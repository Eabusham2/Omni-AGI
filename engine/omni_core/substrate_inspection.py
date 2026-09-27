"""Load-free, generation-bound inspection of sharded neural substrates.

This module never constructs an ``AdaptiveBrain`` or ``NeuralSubstrate``.  It
validates the committed engine pointer and reads only the immutable shard data
needed by an inspection request.  Aggregate indexes are derived caches: they
are atomically published under the source generation and can always be rebuilt
from checksummed authoritative shards.
"""

import base64
import binascii
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file

from .packed_vsa_vectors import PackedTernaryVectors
from .persistence import atomic_write_bytes


SUBSTRATE_STORE_FORMAT = "omni-substrate-shards"
SUBSTRATE_STORE_VERSION = 3
READABLE_SUBSTRATE_STORE_VERSIONS = frozenset((1, 2, 3))
INSPECTION_INDEX_FORMAT = "omni-substrate-inspection-index"
INSPECTION_INDEX_VERSION = 1
INSPECTION_INDEX_RECORDS_PER_SHARD = 512
INSPECTION_TRANSPORT_BYTES = 16 * 1024 * 1024
ADJACENCY_INDEX_FORMAT = "omni-substrate-adjacency-bloom"
ADJACENCY_INDEX_VERSION = 1
ADJACENCY_BLOOM_BITS = 8_192
ADJACENCY_BLOOM_HASHES = 6
INSPECTION_DISK_RESERVE_BYTES = 20 * 1024**3
_KINDS = ("assemblies", "neurons", "synapses")
_BANDS = ("high", "medium", "quiet")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if (
        not relative
        or relative.startswith(("/", "\\"))
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("persisted substrate inspection path is unsafe")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("persisted substrate inspection path escapes its store") from error
    return path


def _read_bytes(path: Path, expected_sha: str, expected_bytes: int) -> bytes:
    payload = path.read_bytes()
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha:
        raise ValueError("persisted substrate inspection blob checksum mismatch")
    return payload


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid persisted substrate %s count" % label)
    return int(value)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    number = _number(value, float("nan"))
    if not math.isfinite(number):
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(number))


def _band(activation: float) -> str:
    return "high" if activation >= 0.66 else ("medium" if activation >= 0.2 else "quiet")


def _adjacency_positions(identifier: str) -> Tuple[int, ...]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[index * 4 : index * 4 + 4], "little")
        % ADJACENCY_BLOOM_BITS
        for index in range(ADJACENCY_BLOOM_HASHES)
    )


def _adjacency_bloom(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    bits = bytearray(ADJACENCY_BLOOM_BITS // 8)
    endpoints: set[str] = set()
    for record in records:
        endpoints.update(
            value
            for value in (
                str(record.get("source_id", "")),
                str(record.get("target_id", "")),
            )
            if value
        )
    for identifier in endpoints:
        for position in _adjacency_positions(identifier):
            bits[position // 8] |= 1 << (position % 8)
    return {
        "format": ADJACENCY_INDEX_FORMAT,
        "formatVersion": ADJACENCY_INDEX_VERSION,
        "bits": ADJACENCY_BLOOM_BITS,
        "hashes": ADJACENCY_BLOOM_HASHES,
        "uniqueEndpoints": len(endpoints),
        "payload": base64.b64encode(bytes(bits)).decode("ascii"),
    }


def _adjacency_might_contain(value: Mapping[str, Any], identifier: str) -> bool:
    if (
        value.get("format") != ADJACENCY_INDEX_FORMAT
        or int(value.get("formatVersion", 0)) != ADJACENCY_INDEX_VERSION
        or value.get("bits") != ADJACENCY_BLOOM_BITS
        or value.get("hashes") != ADJACENCY_BLOOM_HASHES
        or not isinstance(value.get("payload"), str)
    ):
        raise ValueError("persisted adjacency bloom is invalid")
    try:
        bits = base64.b64decode(value["payload"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("persisted adjacency bloom is invalid") from error
    if len(bits) != ADJACENCY_BLOOM_BITS // 8:
        raise ValueError("persisted adjacency bloom is invalid")
    return all(
        bool(bits[position // 8] & (1 << (position % 8)))
        for position in _adjacency_positions(identifier)
    )


def neuron_shard_inspection(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return bounded activation-band partials for one neuron shard."""

    values: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw in records:
        region = str(raw.get("region", "semantic"))
        activation = _number(raw.get("activation"))
        key = (region, _band(activation))
        current = values.setdefault(
            key,
            {
                "region": region,
                "band": key[1],
                "count": 0,
                "activeCount": 0,
                "activationTotal": 0.0,
                "maxActivation": 0.0,
            },
        )
        current["count"] += 1
        current["activeCount"] += int(activation >= 0.1)
        current["activationTotal"] += activation
        current["maxActivation"] = max(float(current["maxActivation"]), activation)
    return {
        "format": INSPECTION_INDEX_FORMAT,
        "formatVersion": INSPECTION_INDEX_VERSION,
        "activationBands": [values[key] for key in sorted(values)],
    }


def synapse_shard_inspection(
    records: Iterable[Mapping[str, Any]],
    region_by_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Return bounded region-pathway partials for one synapse shard."""

    records = list(records)
    values: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw in records:
        source_region = region_by_id.get(str(raw.get("source_id", "")), "unknown")
        target_region = region_by_id.get(str(raw.get("target_id", "")), "unknown")
        key = (source_region, target_region)
        current = values.setdefault(
            key,
            {
                "sourceRegion": source_region,
                "targetRegion": target_region,
                "count": 0,
                "effectiveWeights": {"negative": 0, "zero": 0, "positive": 0},
            },
        )
        current["count"] += 1
        effective = int(raw.get("effective_weight", 0))
        if effective not in {-1, 0, 1}:
            raise ValueError("substrate inspection requires exact ternary synapses")
        key_name = "negative" if effective < 0 else ("positive" if effective > 0 else "zero")
        current["effectiveWeights"][key_name] += 1
    return {
        "format": INSPECTION_INDEX_FORMAT,
        "formatVersion": INSPECTION_INDEX_VERSION,
        "pathways": [values[key] for key in sorted(values)],
        "adjacency": _adjacency_bloom(records),
    }


class PersistedSubstrateView:
    """A validated immutable substrate generation and its shard descriptors."""

    def __init__(
        self,
        engine_directory: Path,
        brain_id: str,
        pointer: Dict[str, Any],
        generation: Dict[str, Any],
        generation_sha256: str,
        attention_epoch: int,
        attention_overlay: Mapping[str, Any],
    ) -> None:
        self.engine_directory = engine_directory.resolve()
        self.brain_id = brain_id
        self.store = self.engine_directory / "substrate"
        self.pointer = pointer
        self.generation = generation
        self.generation_sha256 = generation_sha256
        self.substrate_revision = str(pointer["activeGeneration"])
        self.attention_epoch = int(attention_epoch)
        self.attention_legacy_raw_active = bool(
            attention_overlay["legacyRawActive"]
        )
        self.attention_active_neuron_ids = frozenset(
            str(value) for value in attention_overlay["activeNeuronIds"]
        )
        self.attention_recalled_assembly_ids = frozenset(
            str(value) for value in attention_overlay["recalledAssemblyIds"]
        )
        self.attention_eligible_synapse_ids = frozenset(
            str(value) for value in attention_overlay["eligibleSynapseIds"]
        )
        self.attention_overlay = dict(attention_overlay)
        self.revision = (
            self.substrate_revision
            if self.attention_legacy_raw_active
            and self.attention_epoch == 0
            and not self.attention_active_neuron_ids
            and not self.attention_recalled_assembly_ids
            and not self.attention_eligible_synapse_ids
            else _sha256(
                _canonical_json(
                    {
                        "substrateGeneration": self.substrate_revision,
                        "attentionOverlay": self.attention_overlay,
                    }
                )
            )
        )
        self.counts = {
            kind: _count(generation["counts"].get(kind), kind)
            for kind in _KINDS
        }
        self.shards = list(generation["shards"])

    @classmethod
    def open(cls, engine_directory: Path, expected_brain_id: str) -> "PersistedSubstrateView":
        engine_directory = Path(engine_directory).resolve()
        metadata_path = engine_directory / "brain.json"
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if not isinstance(metadata, dict) or str(metadata.get("brain_id", "")) != expected_brain_id:
            raise ValueError("persisted substrate belongs to another brain")
        substrate = metadata.get("substrate")
        pointer = substrate.get("persistence") if isinstance(substrate, dict) else None
        if not isinstance(pointer, dict):
            raise ValueError("persisted substrate has no sharded generation")
        generation_id = str(pointer.get("activeGeneration", ""))
        generation_relative = str(pointer.get("generationManifest", ""))
        generation_sha = str(pointer.get("generationManifestSha256", ""))
        if (
            pointer.get("format") != SUBSTRATE_STORE_FORMAT
            or int(pointer.get("formatVersion", 0)) not in READABLE_SUBSTRATE_STORE_VERSIONS
            or len(generation_id) != 64
            or any(character not in "0123456789abcdef" for character in generation_id)
            or generation_relative != "generations/%s/manifest.json" % generation_id
            or len(generation_sha) != 64
        ):
            raise ValueError("persisted substrate pointer is invalid")
        store = engine_directory / "substrate"
        root_pointer = json.loads((store / "manifest.json").read_text("utf-8"))
        if _canonical_json(root_pointer) != _canonical_json(pointer):
            raise ValueError("persisted substrate pointer diverges from engine state")
        generation_path = _safe_path(store, generation_relative)
        generation_bytes = _read_bytes(
            generation_path,
            generation_sha,
            generation_path.stat().st_size,
        )
        generation = json.loads(generation_bytes.decode("utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("persisted substrate generation is invalid")
        body = {key: value for key, value in generation.items() if key != "contentSha256"}
        if (
            generation.get("format") != SUBSTRATE_STORE_FORMAT
            or int(generation.get("formatVersion", 0)) != int(pointer.get("formatVersion", 0))
            or (
                int(generation.get("formatVersion", 0)) == 3
                and generation.get("schema") != "neural-substrate-2"
            )
            or generation.get("contentSha256") != generation_id
            or pointer.get("contentSha256") != generation_id
            or _sha256(_canonical_json(body)) != generation_id
            or generation.get("counts") != pointer.get("counts")
            or not isinstance(generation.get("shards"), list)
        ):
            raise ValueError("persisted substrate generation checksum failed")
        observed = {kind: 0 for kind in _KINDS}
        seen: set[Tuple[str, str, int]] = set()
        for shard in generation["shards"]:
            if not isinstance(shard, dict):
                raise ValueError("persisted substrate shard descriptor is invalid")
            kind = str(shard.get("kind", ""))
            bucket = str(shard.get("bucket", ""))
            part = shard.get("part")
            count = shard.get("count")
            if (
                kind not in _KINDS
                or len(bucket) != 1
                or bucket not in "0123456789abcdef"
                or isinstance(part, bool)
                or not isinstance(part, int)
                or part < 0
            ):
                raise ValueError("persisted substrate shard placement is invalid")
            if int(generation.get("formatVersion", 0)) == 3 and (
                (kind == "neurons" and not isinstance(shard.get("tensors"), dict))
                or (kind == "assemblies" and shard.get("tensors") is not None)
            ):
                raise ValueError("persisted packed vector shard layout is invalid")
            observed[kind] += _count(count, kind)
            key = (kind, bucket, part)
            if key in seen:
                raise ValueError("persisted substrate shard is duplicated")
            seen.add(key)
        expected = {
            kind: _count(generation["counts"].get(kind), kind)
            for kind in _KINDS
        }
        if observed != expected:
            raise ValueError("persisted substrate shard totals diverge")
        boundary = metadata.get("fresh_attention_boundary")
        if boundary is not None and not isinstance(boundary, Mapping):
            raise ValueError("persisted attention boundary is invalid")
        raw_epoch = boundary.get("epoch", 0) if isinstance(boundary, Mapping) else 0
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
            raise ValueError("persisted attention epoch is invalid")
        raw_overlay = metadata.get("attention_overlay")
        if raw_overlay is None:
            overlay: Dict[str, Any] = {
                "format": "omni-substrate-attention-overlay",
                "formatVersion": 1,
                "epoch": int(raw_epoch),
                "legacyRawActive": int(raw_epoch) == 0,
                "activeNeuronIds": [],
                "recalledAssemblyIds": [],
                "eligibleSynapseIds": [],
            }
        else:
            if not isinstance(raw_overlay, Mapping):
                raise ValueError("persisted attention overlay is invalid")
            overlay = dict(raw_overlay)
            id_fields = (
                "activeNeuronIds",
                "recalledAssemblyIds",
                "eligibleSynapseIds",
            )
            if (
                overlay.get("format") != "omni-substrate-attention-overlay"
                or overlay.get("formatVersion") != 1
                or overlay.get("epoch") != raw_epoch
                or not isinstance(overlay.get("legacyRawActive"), bool)
                or any(
                    not isinstance(overlay.get(field), list)
                    or not all(
                        isinstance(value, str) and value
                        for value in overlay.get(field, [])
                    )
                    or len(set(overlay.get(field, [])))
                    != len(overlay.get(field, []))
                    for field in id_fields
                )
            ):
                raise ValueError("persisted attention overlay is invalid")
            overlay = {
                "format": "omni-substrate-attention-overlay",
                "formatVersion": 1,
                "epoch": int(raw_epoch),
                "legacyRawActive": bool(overlay["legacyRawActive"]),
                **{
                    field: sorted(str(value) for value in overlay[field])
                    for field in id_fields
                },
            }
        return cls(
            engine_directory,
            expected_brain_id,
            dict(pointer),
            generation,
            generation_sha,
            int(raw_epoch),
            overlay,
        )

    def effective_activation(self, record: Mapping[str, Any]) -> float:
        identifier = str(record.get("id", record.get("neuron_id", "")))
        if (
            not self.attention_legacy_raw_active
            and identifier not in self.attention_active_neuron_ids
        ):
            return 0.0
        return _number(record.get("activation"))

    def effective_eligibility(self, identifier: str, value: Any) -> float:
        if (
            not self.attention_legacy_raw_active
            and identifier not in self.attention_eligible_synapse_ids
        ):
            return 0.0
        return _number(value)

    def kind_shards(self, kind: str) -> List[Dict[str, Any]]:
        return [shard for shard in self.shards if shard.get("kind") == kind]

    def records(self, shard: Mapping[str, Any]) -> Dict[str, Any]:
        spec = shard.get("records")
        if not isinstance(spec, dict):
            raise ValueError("persisted substrate record descriptor is invalid")
        checksum = str(spec.get("sha256", ""))
        relative = str(spec.get("path", ""))
        size = _count(spec.get("bytes"), "record bytes")
        if len(checksum) != 64 or relative != "blobs/%s.json" % checksum:
            raise ValueError("persisted substrate record identity is invalid")
        payload = json.loads(
            _read_bytes(_safe_path(self.store, relative), checksum, size).decode("utf-8")
        )
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != shard.get("kind")
            or not isinstance(payload.get("records"), list)
            or len(payload["records"]) != _count(shard.get("count"), "shard")
        ):
            raise ValueError("persisted substrate record payload is invalid")
        if int(self.generation.get("formatVersion", 0)) == 3:
            kind = str(shard.get("kind", ""))
            if kind in {"neurons", "assemblies"}:
                ids = payload.get("ids")
                if (
                    not isinstance(ids, list)
                    or not all(isinstance(identifier, str) and identifier for identifier in ids)
                    or not all(isinstance(record, Mapping) for record in payload["records"])
                    or ids != [record.get("id") for record in payload["records"]]
                    or payload.get("vectorIds") != ids
                ):
                    raise ValueError("persisted packed vector record IDs are invalid")
                if kind == "neurons":
                    packed = payload.get("packedVectorState")
                    if (
                        not isinstance(packed, dict)
                        or packed.get("ids") != ids
                        or not isinstance(shard.get("tensors"), dict)
                    ):
                        raise ValueError("persisted neuron packed vector state is missing")
                elif (
                    payload.get("vectorStorage") != "shared-neuron-packed"
                    or shard.get("tensors") is not None
                ):
                    raise ValueError("persisted assembly vector must alias a neuron row")
        return payload

    def tensors(self, shard: Mapping[str, Any]) -> Dict[str, Any]:
        spec = shard.get("tensors")
        if not isinstance(spec, dict):
            raise ValueError("persisted substrate tensor descriptor is invalid")
        checksum = str(spec.get("sha256", ""))
        relative = str(spec.get("path", ""))
        size = _count(spec.get("bytes"), "tensor bytes")
        if len(checksum) != 64 or relative != "blobs/%s.safetensors" % checksum:
            raise ValueError("persisted substrate tensor identity is invalid")
        path = _safe_path(self.store, relative)
        _read_bytes(path, checksum, size)
        tensors = load_file(str(path), device="cpu")
        if shard.get("kind") != "synapses":
            if (
                int(self.generation.get("formatVersion", 0)) == 3
                and shard.get("kind") == "neurons"
            ):
                payload = self.records(shard)
                if set(tensors) != {"packed_rows", "update_counters_le"}:
                    raise ValueError("persisted neuron packed tensor fields are invalid")
                PackedTernaryVectors.from_state(
                    payload["packedVectorState"], tensors
                )
            return tensors
        count = _count(shard.get("count"), "synapse count")
        version = int(self.generation.get("formatVersion", 0))
        if version == 1:
            tensors.pop("latent_weight", None)
            effective = tensors.get("effective_weight")
            if effective is None or int(effective.numel()) != count:
                raise ValueError("persisted synapse effective weights are missing")
        else:
            if "latent_weight" in tensors or "effective_weight" in tensors:
                raise ValueError("current synapse shard contains a duplicate weight")
            packed = tensors.get("packed_effective_weight")
            if (
                packed is None
                or packed.dtype != torch.uint8
                or int(packed.numel()) != (count + 3) // 4
            ):
                raise ValueError("packed synapse tensor shard is invalid")
            raw = bytes(packed.reshape(-1).tolist())
            levels = []
            for index in range(len(raw) * 4):
                code = (raw[index // 4] >> ((index % 4) * 2)) & 0x03
                if code == 3 or (index >= count and code != 1):
                    raise ValueError("packed synapse tensor shard is invalid")
                if index < count:
                    levels.append(code - 1)
            tensors["effective_weight"] = torch.tensor(levels, dtype=torch.int8)
        return tensors


def _create_aggregate_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE aggregate (
          group_kind TEXT NOT NULL,
          id TEXT NOT NULL,
          label TEXT NOT NULL,
          cluster_kind TEXT NOT NULL,
          region TEXT,
          source_region TEXT,
          target_region TEXT,
          count INTEGER NOT NULL,
          active_count INTEGER NOT NULL,
          activation_total REAL NOT NULL,
          max_activation REAL NOT NULL,
          negative INTEGER NOT NULL,
          zero_count INTEGER NOT NULL,
          positive INTEGER NOT NULL,
          PRIMARY KEY (group_kind, id)
        ) WITHOUT ROWID
        """
    )
    return connection


def _upsert_band(connection: sqlite3.Connection, value: Mapping[str, Any]) -> None:
    region = str(value.get("region", "semantic"))
    band = str(value.get("band", "quiet"))
    if band not in _BANDS:
        raise ValueError("persisted neuron inspection band is invalid")
    count = _count(value.get("count"), "inspection band")
    active = _count(value.get("activeCount"), "inspection active")
    total = _number(value.get("activationTotal"))
    maximum = _number(value.get("maxActivation"))
    if active > count:
        raise ValueError("persisted neuron inspection active count diverges")
    identifier = "region:%s:%s" % (region, band)
    connection.execute(
        """
        INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, 0, 0, 0)
        ON CONFLICT(group_kind, id) DO UPDATE SET
          count=count+excluded.count,
          active_count=active_count+excluded.active_count,
          activation_total=activation_total+excluded.activation_total,
          max_activation=max(max_activation, excluded.max_activation)
        """,
        ("activation-bands", identifier, "%s · %s" % (region, band), "activation-band", region,
         count, active, total, maximum),
    )


def _upsert_pathway(connection: sqlite3.Connection, value: Mapping[str, Any]) -> None:
    source = str(value.get("sourceRegion", "unknown"))
    target = str(value.get("targetRegion", "unknown"))
    weights = value.get("effectiveWeights")
    if not isinstance(weights, Mapping):
        raise ValueError("persisted pathway inspection weights are invalid")
    count = _count(value.get("count"), "inspection pathway")
    negative = _count(weights.get("negative"), "negative pathway")
    zero = _count(weights.get("zero"), "zero pathway")
    positive = _count(weights.get("positive"), "positive pathway")
    if negative + zero + positive != count:
        raise ValueError("persisted pathway weight counts diverge")
    identifier = "pathway:%s>%s" % (source, target)
    connection.execute(
        """
        INSERT INTO aggregate VALUES (?, ?, ?, ?, NULL, ?, ?, ?, 0, 0, 0, ?, ?, ?)
        ON CONFLICT(group_kind, id) DO UPDATE SET
          count=count+excluded.count,
          negative=negative+excluded.negative,
          zero_count=zero_count+excluded.zero_count,
          positive=positive+excluded.positive
        """,
        ("pathways", identifier, "%s → %s" % (source, target), "pathway", source, target,
         count, negative, zero, positive),
    )


def _validated_partial(shard: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    value = shard.get("inspection")
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("format") != INSPECTION_INDEX_FORMAT
        or int(value.get("formatVersion", 0)) != INSPECTION_INDEX_VERSION
    ):
        raise ValueError("persisted shard inspection summary is invalid")
    return value


def _aggregate_from_shards(
    view: PersistedSubstrateView,
    connection: sqlite3.Connection,
) -> None:
    shards = view.kind_shards("neurons") + view.kind_shards("synapses")
    partials = [_validated_partial(shard) for shard in shards]
    if all(value is not None for value in partials):
        for shard, partial in zip(shards, partials):
            if shard["kind"] == "neurons":
                values = partial.get("activationBands")
                if not isinstance(values, list):
                    raise ValueError("persisted neuron shard summary is invalid")
                for value in values:
                    if not isinstance(value, Mapping):
                        raise ValueError("persisted neuron shard summary entry is invalid")
                    _upsert_band(connection, value)
            else:
                values = partial.get("pathways")
                if not isinstance(values, list):
                    raise ValueError("persisted synapse shard summary is invalid")
                for value in values:
                    if not isinstance(value, Mapping):
                        raise ValueError("persisted pathway shard summary entry is invalid")
                    _upsert_pathway(connection, value)
        connection.commit()
        return

    region_by_id: Dict[str, str] = {}
    for shard in view.kind_shards("neurons"):
        payload = view.records(shard)
        summary = neuron_shard_inspection(payload["records"])
        for record in payload["records"]:
            if not isinstance(record, Mapping):
                raise ValueError("persisted neuron record is invalid")
            region_by_id[str(record.get("id", record.get("neuron_id", "")))] = str(
                record.get("region", "semantic")
            )
        for value in summary["activationBands"]:
            _upsert_band(connection, value)
    connection.commit()

    for shard in view.kind_shards("synapses"):
        payload = view.records(shard)
        tensors = view.tensors(shard)
        effective = tensors.get("effective_weight")
        if effective is None or int(effective.numel()) != len(payload["records"]):
            raise ValueError("persisted synapse effective weights are missing")
        restored: List[Dict[str, Any]] = []
        for index, record in enumerate(payload["records"]):
            if not isinstance(record, Mapping):
                raise ValueError("persisted synapse record is invalid")
            item = dict(record)
            item["effective_weight"] = int(effective[index].item())
            restored.append(item)
        summary = synapse_shard_inspection(restored, region_by_id)
        for value in summary["pathways"]:
            _upsert_pathway(connection, value)
        connection.commit()


def _aggregate_records(connection: sqlite3.Connection, group_kind: str) -> Iterator[Dict[str, Any]]:
    cursor = connection.execute(
        """
        SELECT id, label, cluster_kind, region, source_region, target_region,
               count, active_count, activation_total, max_activation,
               negative, zero_count, positive
        FROM aggregate WHERE group_kind=? ORDER BY id
        """,
        (group_kind,),
    )
    for row in cursor:
        count = int(row[6])
        yield {
            "id": str(row[0]),
            "label": str(row[1]),
            "kind": str(row[2]),
            "count": count,
            "activeCount": int(row[7]),
            "meanActivation": (float(row[8]) / float(count)) if count else 0.0,
            "maxActivation": float(row[9]),
            "effectiveWeights": {
                "negative": int(row[10]),
                "zero": int(row[11]),
                "positive": int(row[12]),
            },
            **({"region": str(row[3])} if row[3] is not None else {}),
            **({"sourceRegion": str(row[4])} if row[4] is not None else {}),
            **({"targetRegion": str(row[5])} if row[5] is not None else {}),
        }


def _write_index(view: PersistedSubstrateView, connection: sqlite3.Connection) -> Path:
    band_total = int(
        connection.execute(
            "SELECT COALESCE(SUM(count), 0) FROM aggregate WHERE group_kind='activation-bands'"
        ).fetchone()[0]
    )
    pathway_total = int(
        connection.execute(
            "SELECT COALESCE(SUM(count), 0) FROM aggregate WHERE group_kind='pathways'"
        ).fetchone()[0]
    )
    if band_total != view.counts["neurons"] or pathway_total != view.counts["synapses"]:
        raise ValueError("persisted inspection aggregate totals diverge from the source generation")
    index_root = view.store / "inspection"
    blob_root = index_root / "blobs"
    blob_root.mkdir(parents=True, exist_ok=True)
    descriptors: List[Dict[str, Any]] = []
    for group_kind in ("activation-bands", "pathways"):
        page: List[Dict[str, Any]] = []
        part = 0
        for record in _aggregate_records(connection, group_kind):
            page.append(record)
            if len(page) < INSPECTION_INDEX_RECORDS_PER_SHARD:
                continue
            descriptors.append(_write_index_page(blob_root, group_kind, part, page))
            part += 1
            page = []
        if page:
            descriptors.append(_write_index_page(blob_root, group_kind, part, page))
    body = {
        "format": INSPECTION_INDEX_FORMAT,
        "formatVersion": INSPECTION_INDEX_VERSION,
        "sourceGeneration": view.substrate_revision,
        "sourceGenerationManifestSha256": view.generation_sha256,
        "counts": dict(view.counts),
        "recordsPerShard": INSPECTION_INDEX_RECORDS_PER_SHARD,
        "shards": descriptors,
    }
    manifest = {**body, "contentSha256": _sha256(_canonical_json(body))}
    path = (
        index_root
        / "generations"
        / view.substrate_revision
        / "manifest.json"
    )
    atomic_write_bytes(path, _canonical_json(manifest))
    return path


def _write_index_page(
    blob_root: Path,
    kind: str,
    part: int,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = _canonical_json({"kind": kind, "records": list(records)})
    checksum = _sha256(payload)
    path = blob_root / (checksum + ".json")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("persisted inspection blob conflicts with its content hash")
    else:
        atomic_write_bytes(path, payload)
    return {
        "kind": kind,
        "part": int(part),
        "count": len(records),
        "firstId": str(records[0]["id"]),
        "lastId": str(records[-1]["id"]),
        "records": {
            "path": "blobs/%s.json" % checksum,
            "sha256": checksum,
            "bytes": len(payload),
        },
    }


def _load_index(view: PersistedSubstrateView) -> Optional[Dict[str, Any]]:
    index_root = view.store / "inspection"
    path = (
        index_root
        / "generations"
        / view.substrate_revision
        / "manifest.json"
    )
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("persisted inspection index manifest is invalid")
    body = {key: value for key, value in manifest.items() if key != "contentSha256"}
    if (
        manifest.get("format") != INSPECTION_INDEX_FORMAT
        or int(manifest.get("formatVersion", 0)) != INSPECTION_INDEX_VERSION
        or manifest.get("sourceGeneration") != view.substrate_revision
        or manifest.get("sourceGenerationManifestSha256") != view.generation_sha256
        or manifest.get("counts") != view.counts
        or manifest.get("contentSha256") != _sha256(_canonical_json(body))
        or not isinstance(manifest.get("shards"), list)
    ):
        raise ValueError("persisted inspection index is stale or corrupt")
    seen: set[Tuple[str, int]] = set()
    for shard in manifest["shards"]:
        if not isinstance(shard, dict):
            raise ValueError("persisted inspection index shard is invalid")
        kind = str(shard.get("kind", ""))
        part = shard.get("part")
        if (
            kind not in {"activation-bands", "pathways"}
            or isinstance(part, bool)
            or not isinstance(part, int)
            or part < 0
            or (kind, part) in seen
        ):
            raise ValueError("persisted inspection index shard placement is invalid")
        seen.add((kind, part))
        _count(shard.get("count"), "inspection shard")
    return manifest


def ensure_inspection_index(view: PersistedSubstrateView) -> Dict[str, Any]:
    existing = _load_index(view)
    if existing is not None:
        return existing
    descriptor, temporary_name = tempfile.mkstemp(prefix="omni-inspection-", suffix=".sqlite3")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = _create_aggregate_database(temporary)
        try:
            _aggregate_from_shards(view, connection)
            _write_index(view, connection)
        finally:
            connection.close()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    loaded = _load_index(view)
    if loaded is None:
        raise RuntimeError("persisted inspection index publication failed")
    return loaded


def _adjacency_index_path(view: PersistedSubstrateView) -> Path:
    return (
        view.store
        / "inspection"
        / "adjacency"
        / "generations"
        / (view.substrate_revision + ".json")
    )


def _validate_adjacency_index(
    view: PersistedSubstrateView,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    body = {key: value for key, value in manifest.items() if key != "contentSha256"}
    entries = manifest.get("shards")
    if (
        manifest.get("format") != ADJACENCY_INDEX_FORMAT
        or int(manifest.get("formatVersion", 0)) != ADJACENCY_INDEX_VERSION
        or manifest.get("sourceGeneration") != view.substrate_revision
        or manifest.get("sourceGenerationManifestSha256") != view.generation_sha256
        or manifest.get("synapseCount") != view.counts["synapses"]
        or manifest.get("contentSha256") != _sha256(_canonical_json(body))
        or not isinstance(entries, list)
    ):
        raise ValueError("persisted adjacency index is stale or corrupt")
    descriptors = {
        (str(shard.get("bucket", "")), int(shard.get("part", -1))): shard
        for shard in view.kind_shards("synapses")
    }
    if len(entries) != len(descriptors):
        raise ValueError("persisted adjacency index shard total diverges")
    seen: set[Tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("persisted adjacency index shard is invalid")
        key = (str(entry.get("bucket", "")), int(entry.get("part", -1)))
        descriptor = descriptors.get(key)
        record_spec = descriptor.get("records") if descriptor else None
        if (
            descriptor is None
            or key in seen
            or not isinstance(record_spec, Mapping)
            or entry.get("recordsSha256") != record_spec.get("sha256")
            or entry.get("count") != descriptor.get("count")
            or not isinstance(entry.get("adjacency"), Mapping)
        ):
            raise ValueError("persisted adjacency index shard binding is invalid")
        _adjacency_might_contain(entry["adjacency"], "__validation_probe__")
        seen.add(key)
    return dict(manifest)


def _load_adjacency_index(
    view: PersistedSubstrateView,
) -> Optional[Dict[str, Any]]:
    try:
        payload = _adjacency_index_path(view).read_bytes()
    except FileNotFoundError:
        return None
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("persisted adjacency index is invalid")
    return _validate_adjacency_index(view, value)


def ensure_adjacency_index(view: PersistedSubstrateView) -> Dict[str, Any]:
    """Return a generation-bound endpoint locator for exact pathway reads.

    Legacy generations are scanned once to derive bounded per-shard Bloom
    filters. False positives only read an extra authoritative shard; false
    negatives are impossible, so effective-zero pathways remain visible.
    """

    existing = _load_adjacency_index(view)
    if existing is not None:
        return existing
    entries: List[Dict[str, Any]] = []
    for shard in view.kind_shards("synapses"):
        partial = _validated_partial(shard)
        adjacency = (
            partial.get("adjacency")
            if isinstance(partial, Mapping)
            else None
        )
        if not isinstance(adjacency, Mapping):
            adjacency = _adjacency_bloom(view.records(shard)["records"])
        else:
            # Decode once before publication so malformed inline indexes never
            # become a trusted fast path.
            _adjacency_might_contain(adjacency, "__validation_probe__")
        record_spec = shard.get("records")
        if not isinstance(record_spec, Mapping):
            raise ValueError("persisted substrate record descriptor is invalid")
        entries.append(
            {
                "bucket": str(shard["bucket"]),
                "part": int(shard["part"]),
                "count": int(shard["count"]),
                "recordsSha256": str(record_spec.get("sha256", "")),
                "adjacency": dict(adjacency),
            }
        )
    body = {
        "format": ADJACENCY_INDEX_FORMAT,
        "formatVersion": ADJACENCY_INDEX_VERSION,
        "sourceGeneration": view.substrate_revision,
        "sourceGenerationManifestSha256": view.generation_sha256,
        "synapseCount": view.counts["synapses"],
        "shards": entries,
    }
    manifest = {**body, "contentSha256": _sha256(_canonical_json(body))}
    payload = _canonical_json(manifest)
    filesystem = os.statvfs(view.store)
    available = int(filesystem.f_bavail) * int(filesystem.f_frsize)
    if available - len(payload) >= INSPECTION_DISK_RESERVE_BYTES:
        atomic_write_bytes(_adjacency_index_path(view), payload)
        loaded = _load_adjacency_index(view)
        if loaded is None:
            raise RuntimeError("persisted adjacency index publication failed")
        return loaded
    return _validate_adjacency_index(view, manifest)


def _connected_synapse_shards(
    view: PersistedSubstrateView,
    identifier: str,
) -> List[Dict[str, Any]]:
    manifest = ensure_adjacency_index(view)
    descriptors = {
        (str(shard.get("bucket", "")), int(shard.get("part", -1))): shard
        for shard in view.kind_shards("synapses")
    }
    return [
        descriptors[(str(entry["bucket"]), int(entry["part"]))]
        for entry in manifest["shards"]
        if _adjacency_might_contain(entry["adjacency"], identifier)
    ]


def _index_records(
    view: PersistedSubstrateView,
    manifest: Mapping[str, Any],
    kind: str,
) -> Iterator[Dict[str, Any]]:
    index_root = view.store / "inspection"
    for shard in manifest["shards"]:
        if shard.get("kind") != kind:
            continue
        spec = shard.get("records")
        if not isinstance(spec, dict):
            raise ValueError("persisted inspection record descriptor is invalid")
        checksum = str(spec.get("sha256", ""))
        relative = str(spec.get("path", ""))
        size = _count(spec.get("bytes"), "inspection bytes")
        if len(checksum) != 64 or relative != "blobs/%s.json" % checksum:
            raise ValueError("persisted inspection record identity is invalid")
        payload = json.loads(
            _read_bytes(_safe_path(index_root, relative), checksum, size).decode("utf-8")
        )
        records = payload.get("records") if isinstance(payload, dict) else None
        if (
            not isinstance(records, list)
            or payload.get("kind") != kind
            or len(records) != _count(shard.get("count"), "inspection shard")
        ):
            raise ValueError("persisted inspection record payload is invalid")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("persisted inspection record is invalid")
            yield record


def _region_clusters(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    totals: Dict[str, float] = defaultdict(float)
    for record in records:
        region = str(record.get("region", "semantic"))
        current = grouped.setdefault(
            region,
            {
                "id": "region:%s" % region,
                "label": region,
                "kind": "region",
                "region": region,
                "count": 0,
                "activeCount": 0,
                "meanActivation": 0.0,
                "maxActivation": 0.0,
                "effectiveWeights": {"negative": 0, "zero": 0, "positive": 0},
            },
        )
        count = _count(record.get("count"), "indexed region")
        current["count"] += count
        current["activeCount"] += _count(record.get("activeCount"), "indexed active")
        totals[region] += _number(record.get("meanActivation")) * count
        current["maxActivation"] = max(
            float(current["maxActivation"]), _number(record.get("maxActivation"))
        )
    for region, record in grouped.items():
        record["meanActivation"] = totals[region] / max(1, int(record["count"]))
    return [grouped[key] for key in sorted(grouped)]


def _attention_masked_bands(
    view: PersistedSubstrateView,
    records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Project immutable raw activation summaries through the live epoch mask.

    The common post-boundary case has an empty overlay and touches only the
    tiny aggregate index. If a later turn activates a sparse set, only neuron
    record shards in those identifiers' hash buckets are read.
    """

    values = [dict(record) for record in records]
    if view.attention_legacy_raw_active:
        return values
    totals: Dict[str, int] = defaultdict(int)
    for record in values:
        totals[str(record.get("region", "semantic"))] += _count(
            record.get("count"), "indexed activation band"
        )

    active_records: Dict[str, Mapping[str, Any]] = {}
    active_ids = set(view.attention_active_neuron_ids)
    if active_ids:
        buckets = {
            hashlib.sha256(("neurons:%s" % identifier).encode("utf-8")).hexdigest()[:1]
            for identifier in active_ids
        }
        for shard in view.kind_shards("neurons"):
            if str(shard.get("bucket", "")) not in buckets:
                continue
            for record in view.records(shard)["records"]:
                if not isinstance(record, Mapping):
                    raise ValueError("persisted neuron record is invalid")
                identifier = str(record.get("id", record.get("neuron_id", "")))
                if identifier in active_ids:
                    active_records[identifier] = record
        if set(active_records) != active_ids:
            raise ValueError("persisted attention overlay names unknown neurons")

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def ensure(region: str, band: str) -> Dict[str, Any]:
        key = (region, band)
        return grouped.setdefault(
            key,
            {
                "id": "region:%s:%s" % (region, band),
                "label": "%s · %s" % (region, band),
                "kind": "activation-band",
                "region": region,
                "count": 0,
                "activeCount": 0,
                "meanActivation": 0.0,
                "maxActivation": 0.0,
                "effectiveWeights": {"negative": 0, "zero": 0, "positive": 0},
                "_activationTotal": 0.0,
            },
        )

    active_by_region: Dict[str, int] = defaultdict(int)
    for record in active_records.values():
        region = str(record.get("region", "semantic"))
        active_by_region[region] += 1
        activation = view.effective_activation(record)
        current = ensure(region, _band(activation))
        current["count"] += 1
        current["activeCount"] += int(activation >= 0.1)
        current["_activationTotal"] += activation
        current["maxActivation"] = max(
            float(current["maxActivation"]), activation
        )
    for region, count in totals.items():
        quiet = count - active_by_region.get(region, 0)
        if quiet < 0:
            raise ValueError("persisted attention overlay diverges from neuron totals")
        if quiet:
            ensure(region, "quiet")["count"] += quiet

    output: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        current = grouped[key]
        count = int(current["count"])
        current["meanActivation"] = (
            float(current.pop("_activationTotal")) / float(count)
            if count
            else 0.0
        )
        if count:
            output.append(current)
    return output


def _cursor(offset: int, revision: str, fingerprint: str) -> str:
    payload = _canonical_json(
        {"offset": max(0, int(offset)), "revision": revision, "fingerprint": fingerprint}
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _cursor_offset(value: str, revision: str, fingerprint: str) -> int:
    if not value or len(value) > 2048:
        raise ValueError("invalid substrate cursor")
    try:
        padding = "=" * ((4 - len(value) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode((value + padding).encode("ascii")))
    except (ValueError, TypeError, binascii.Error, UnicodeError) as error:
        raise ValueError("invalid substrate cursor") from error
    if (
        not isinstance(payload, dict)
        or payload.get("revision") != revision
        or payload.get("fingerprint") != fingerprint
    ):
        raise ValueError("substrate cursor is stale or belongs to another query")
    return _count(payload.get("offset"), "cursor offset")


def _bounded_page(
    records: Iterable[Dict[str, Any]],
    offset: int,
    page_size: int,
) -> Tuple[List[Dict[str, Any]], int, int, bool]:
    page: List[Dict[str, Any]] = []
    matched = 0
    page_bytes = 2
    transport_limited = False
    page_closed = False
    for record in records:
        record_index = matched
        matched += 1
        if record_index < offset or page_closed:
            continue
        encoded = len(_canonical_json(record))
        if len(page) >= page_size:
            page_closed = True
            continue
        separator = int(bool(page))
        if page and page_bytes + separator + encoded > INSPECTION_TRANSPORT_BYTES:
            transport_limited = True
            page_closed = True
            continue
        if not page and encoded > INSPECTION_TRANSPORT_BYTES:
            if "neuronIds" in record:
                record = {**record, "neuronIds": [], "childAssemblyIds": [], "relationshipsPaged": True}
                encoded = len(_canonical_json(record))
            if encoded > INSPECTION_TRANSPORT_BYTES:
                raise ValueError("one substrate inspection record exceeds the transport envelope")
        page.append(record)
        page_bytes += separator + encoded
    return page, matched, page_bytes, transport_limited


def _inspect_neuron(
    view: PersistedSubstrateView, record: Mapping[str, Any]
) -> Dict[str, Any]:
    identifier = str(record.get("id", record.get("neuron_id", "")))
    currently_active = (
        view.attention_legacy_raw_active
        or identifier in view.attention_active_neuron_ids
    )
    return {
        "id": identifier,
        "label": str(record.get("label", "")),
        "region": str(record.get("region", "semantic")),
        "activation": view.effective_activation(record),
        "importance": _number(record.get("importance")),
        "uncertainty": _number(record.get("uncertainty"), 0.5),
        "exposures": _count(max(0, int(record.get("exposures", 0))), "exposures"),
        "createdAt": _timestamp(record.get("created_at")),
        "lastActivatedAt": _timestamp(
            record.get("last_activated_at") if currently_active else None
        ),
        "aliases": [str(value) for value in record.get("aliases", []) if isinstance(value, str)],
    }


def _inspect_assembly(
    view: PersistedSubstrateView, record: Mapping[str, Any]
) -> Dict[str, Any]:
    identifier = str(record.get("id", ""))
    neuron_ids = [str(value) for value in record.get("neuron_ids", [])]
    child_ids = [str(value) for value in record.get("child_assembly_ids", [])]
    return {
        "id": identifier,
        "label": str(record.get("source_label", record.get("kind", "assembly"))),
        "region": "assembly",
        "neuronIds": neuron_ids,
        "childAssemblyIds": child_ids,
        "neuronCount": len(neuron_ids),
        "childAssemblyCount": len(child_ids),
        "relationshipsPaged": False,
        "kind": str(record.get("kind", "knowledge")),
        "source": str(record.get("source", "")),
        "confidence": _number(record.get("confidence"), 0.5),
        "importance": _number(record.get("importance")),
        "rehearsals": _count(max(0, int(record.get("rehearsals", 0))), "rehearsals"),
        "createdAt": _timestamp(record.get("created_at")),
        "lastRecalledAt": _timestamp(
            record.get("last_recalled_at")
            if view.attention_legacy_raw_active
            or identifier in view.attention_recalled_assembly_ids
            else None
        ),
        "sourceLabel": str(record.get("source_label", "")) or None,
        "retainsSourceText": "source_text" in record,
    }


def _synapse_records(
    view: PersistedSubstrateView,
    shards: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    fields = (
        "effective_weight",
        "eligibility",
        "plasticity",
        "uses",
        "stability",
        "last_updated_at",
    )
    for shard in (
        view.kind_shards("synapses") if shards is None else shards
    ):
        payload = view.records(shard)
        tensors = view.tensors(shard)
        if any(name not in tensors or int(tensors[name].numel()) != len(payload["records"]) for name in fields):
            raise ValueError("persisted synapse tensor shard is incomplete")
        for index, raw in enumerate(payload["records"]):
            if not isinstance(raw, Mapping):
                raise ValueError("persisted synapse record is invalid")
            effective = int(tensors["effective_weight"][index].item())
            if effective not in {-1, 0, 1}:
                raise ValueError("persisted synapse is not exact ternary")
            yield {
                "id": str(raw.get("id", "")),
                "sourceId": str(raw.get("source_id", "")),
                "targetId": str(raw.get("target_id", "")),
                "kind": str(raw.get("kind", "associates")),
                "effectiveWeight": effective,
                "eligibility": view.effective_eligibility(
                    str(raw.get("id", "")),
                    float(tensors["eligibility"][index].item()),
                ),
                "plasticity": float(tensors["plasticity"][index].item()),
                "stability": float(tensors["stability"][index].item()),
                "uses": int(tensors["uses"][index].item()),
                "lastUpdatedAt": _timestamp(float(tensors["last_updated_at"][index].item())),
            }


def require_current_substrate_generation(
    view: PersistedSubstrateView,
    expected_live: Mapping[str, Any],
) -> None:
    """Fail closed before using a committed view for a mutable live brain.

    The caller supplies only bounded identity/count metadata, never neuron or
    synapse records. A changed live substrate must use a separate live paged
    inspector; silently substituting the last checkpoint would be incorrect.
    """

    if not isinstance(expected_live, Mapping):
        raise ValueError("live substrate inspection identity is invalid")
    try:
        counts = expected_live["counts"]
        overlay = expected_live["attentionOverlay"]
        if not isinstance(counts, Mapping) or not isinstance(overlay, Mapping):
            raise ValueError("live substrate inspection identity is invalid")
        exact_counts = {kind: _count(counts[kind], kind) for kind in _KINDS}
        state_revision = _count(expected_live["stateRevision"], "state revision")
        active_generation = str(expected_live["activeGeneration"])
        same_overlay = _canonical_json(dict(overlay)) == _canonical_json(
            view.attention_overlay
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("live substrate inspection identity is invalid") from error
    if (
        int(view.generation.get("formatVersion", 0)) != SUBSTRATE_STORE_VERSION
        or active_generation != str(view.pointer.get("activeGeneration", ""))
        or state_revision != _count(view.generation.get("stateRevision"), "state revision")
        or exact_counts != view.counts
        or not same_overlay
    ):
        raise ValueError(
            "live substrate differs from the committed inspection generation"
        )


def query_persisted_substrate(
    engine_directory: Path,
    expected_brain_id: str,
    query: Optional[Mapping[str, Any]] = None,
    *,
    expected_live: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Query one committed generation without loading the neural runtime.

    Pass ``expected_live`` when delegating from an already loaded brain. A
    dirty or mismatched live state is rejected before any shard query.
    """

    view = PersistedSubstrateView.open(engine_directory, expected_brain_id)
    if expected_live is not None:
        require_current_substrate_generation(view, expected_live)
    raw = dict(query or {})
    entity = str(raw.get("entity", "overview"))
    if entity not in {"overview", "neurons", "assemblies", "synapses"}:
        raise ValueError("invalid substrate entity")
    zoom = max(0.0, min(_number(raw.get("zoom"), 0.0), 1.0))
    page_size = raw.get("pageSize", 256)
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("invalid substrate page size")
    region = str(raw.get("region", "")).strip()
    search = str(raw.get("search", "")).strip()
    connected_to = str(raw.get("connectedTo", "")).strip()
    if len(region) > 128 or len(search) > 512 or len(connected_to) > 256:
        raise ValueError("substrate filter is too long")
    if connected_to and entity != "synapses":
        raise ValueError("connectedTo is valid only for synapse queries")
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "entity": entity,
                "zoom": round(zoom, 6),
                "region": region.casefold(),
                "search": search.casefold(),
                "connectedTo": connected_to,
            }
        )
    ).hexdigest()[:20]
    offset = _cursor_offset(str(raw["cursor"]), view.revision, fingerprint) if raw.get("cursor") else 0
    response: Dict[str, Any] = {
        "brainId": expected_brain_id,
        "queriedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "revision": view.revision,
        "substrateContentSha256": view.substrate_revision,
        "attentionEpoch": view.attention_epoch,
        "entity": entity,
        "zoom": zoom,
        "totals": dict(view.counts),
        "matched": 0,
        "offset": offset,
        "returned": 0,
        "pageBytes": 0,
        "transportLimited": False,
        "hasMore": False,
        "clusters": [],
        "neurons": [],
        "assemblies": [],
        "synapses": [],
    }

    search_value = search.casefold()
    if entity == "overview" or zoom < 0.66:
        index = ensure_inspection_index(view)
        bands = _attention_masked_bands(
            view,
            _index_records(view, index, "activation-bands"),
        )
        neuron_clusters = _region_clusters(bands) if zoom < 0.35 else bands
        pathways = list(_index_records(view, index, "pathways"))
        clusters = neuron_clusters + pathways

        def aggregate_matches(record: Mapping[str, Any]) -> bool:
            if region:
                values = {
                    str(record.get("region", "")).casefold(),
                    str(record.get("sourceRegion", "")).casefold(),
                    str(record.get("targetRegion", "")).casefold(),
                }
                if region.casefold() not in values:
                    return False
            if search_value:
                searchable = "%s %s %s %s %s" % (
                    record.get("id", ""),
                    record.get("label", ""),
                    record.get("region", ""),
                    record.get("sourceRegion", ""),
                    record.get("targetRegion", ""),
                )
                if search_value not in searchable.casefold():
                    return False
            return True

        ordered = sorted(
            (dict(record) for record in clusters if aggregate_matches(record)),
            key=lambda item: (str(item["kind"]), -int(item["count"]), str(item["id"])),
        )
        page, matched, page_bytes, transport_limited = _bounded_page(ordered, offset, page_size)
        response["clusters"] = page
    else:
        if entity in {"neurons", "assemblies"}:
            kind = entity

            def records() -> Iterator[Dict[str, Any]]:
                for shard in view.kind_shards(kind):
                    payload = view.records(shard)
                    for raw_record in payload["records"]:
                        if not isinstance(raw_record, Mapping):
                            raise ValueError("persisted substrate detail record is invalid")
                        record = (
                            _inspect_neuron(view, raw_record)
                            if kind == "neurons"
                            else _inspect_assembly(view, raw_record)
                        )
                        searchable = "%s %s %s %s" % (
                            record["id"],
                            record["label"],
                            record.get("region", ""),
                            record.get("sourceLabel", "") or "",
                        )
                        if region and str(record.get("region", "")).casefold() != region.casefold():
                            continue
                        if search_value and search_value not in searchable.casefold():
                            continue
                        yield record

            page, matched, page_bytes, transport_limited = _bounded_page(records(), offset, page_size)
        else:
            region_by_id: Dict[str, str] = {}
            if region:
                for shard in view.kind_shards("neurons"):
                    for raw_record in view.records(shard)["records"]:
                        region_by_id[str(raw_record.get("id", ""))] = str(raw_record.get("region", "semantic"))

            def records() -> Iterator[Dict[str, Any]]:
                selected_shards = (
                    _connected_synapse_shards(view, connected_to)
                    if connected_to
                    else None
                )
                for record in _synapse_records(view, selected_shards):
                    if connected_to and connected_to not in {record["sourceId"], record["targetId"]}:
                        continue
                    if region and region.casefold() not in {
                        region_by_id.get(record["sourceId"], "unknown").casefold(),
                        region_by_id.get(record["targetId"], "unknown").casefold(),
                    }:
                        continue
                    searchable = "%s %s %s %s" % (
                        record["id"], record["sourceId"], record["targetId"], record["kind"]
                    )
                    if search_value and search_value not in searchable.casefold():
                        continue
                    yield record

            page, matched, page_bytes, transport_limited = _bounded_page(records(), offset, page_size)
        response[entity] = page

    end = offset + len(page)
    response["matched"] = matched
    response["returned"] = len(page)
    response["pageBytes"] = page_bytes
    response["transportLimited"] = transport_limited
    response["hasMore"] = end < matched
    if response["hasMore"]:
        response["nextCursor"] = _cursor(end, view.revision, fingerprint)
    return response
