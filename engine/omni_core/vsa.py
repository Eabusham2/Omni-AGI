"""Distributed neural substrate with VSA binding and sparse ternary synapses.

The stable OmniCortex format does not keep an authoritative "idea database"
beside the neural state.  Concepts, experiences, and higher-order ideas are
represented as neuron assemblies connected by plastic synapses.  The
``concepts``/``ideas``/``relations`` properties at the bottom of the class are
compatibility views over that same substrate for the desktop inspector.
"""

import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Mapping as AbstractMapping, MutableMapping
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .persistence import (
    atomic_save_tensors,
    atomic_write_bytes,
    atomic_write_json,
    load_tensors,
    read_json,
)
from .packed_vsa_vectors import PackedTernaryVectors, PackedTernaryVectorView
from .paged_assembly_index import PagedAssemblyIndex, _record_payload
from .paged_assembly_scoring import PagedAssemblyVectorProvider
from .paged_assembly_vector_view import PagedAssemblyVectorView
from .paged_assembly_view import PagedAssemblyView
from .paged_packed_vectors import PagedPackedVectors
from .paged_neuron_metadata import PagedNeuronMetadata
from .paged_vector_scoring import prepare_exact_paged_similarity
from .substrate_inspection import (
    neuron_shard_inspection,
    synapse_shard_inspection,
)


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+\-'.]{1,95}")
_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SUBSTRATE_STORE_FORMAT = "omni-substrate-shards"
_SUBSTRATE_STORE_VERSION = 3
# Older generations remain inspectable for snapshot pruning, but live load
# rejects their higher-precision VSA vectors instead of silently converting.
_READABLE_SUBSTRATE_STORE_VERSIONS = frozenset((1, 2, 3))
_SYNAPSE_TENSOR_FIELDS = (
    "effective_weight",
    "eligibility",
    "plasticity",
    "uses",
    "stability",
    "last_updated_at",
)
_LAZY_SYNAPSE_LOAD_THRESHOLD = 100_000
_FORWARD_INDEX_FORMAT = "omni-substrate-forward-index"
_FORWARD_INDEX_VERSION = 4
_DYNAMIC_ORDER_BASIS = "substrate-shard-record-sha256-v1"
_RECALL_GRAPH_CACHE_BYTES = 64 * 1024 * 1024
_RECALL_GRAPH_EDGE_ESTIMATE = 128
_RECALL_GRAPH_NODE_ESTIMATE = 96
_MISSING = object()


class _PagedAssemblyLookup(AbstractMapping[str, Mapping[str, Any]]):
    """Exact ID/fingerprint reads without a corpus-sized Python dictionary."""

    def __init__(self, view: PagedAssemblyView, field: str) -> None:
        self.view = view
        self.field = field

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        record = (
            self.view.get_by_id(key)
            if self.field == "id"
            else self.view.get_by_fingerprint(key)
        )
        if record is None:
            raise KeyError(key)
        return record

    def __iter__(self) -> Iterator[str]:
        for record in self.view:
            yield str(record[self.field])

    def __len__(self) -> int:
        return len(self.view)


class _RevisionedNodes(dict):
    """Track node membership, which determines whether an edge can fire."""

    def __init__(self, values: Optional[Mapping[str, Dict[str, Any]]] = None):
        super().__init__(values or {})
        self.graph_revision = 0

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        super().__setitem__(key, value)
        self._changed(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._changed(key)

    def clear(self) -> None:
        if self:
            super().clear()
            self._changed(None)

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key in self:
            value = super().pop(key)
            self._changed(key)
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default

    def popitem(self) -> Tuple[str, Dict[str, Any]]:
        value = super().popitem()
        self._changed(value[0])
        return value

    def _changed(self, key: Optional[str] = None) -> None:
        self.graph_revision += 1
        observer = getattr(self, "_recall_graph_observer", None)
        if callable(observer):
            observer((key,) if key is not None else None, self.graph_revision)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Mapping[str, Dict[str, Any]]):
        self.update(other)
        return self


class _TrackedSynapseRecord(dict):
    """Notify its owner when a forward edge or endpoint changes in place."""

    _FORWARD_FIELDS = frozenset(("source_id", "target_id", "effective_weight"))

    def __init__(
        self, value: Mapping[str, Any], changed: Callable[[], None],
        metadata_changed: Optional[Callable[[], None]] = None,
    ):
        super().__init__(value)
        self._changed = changed
        self._metadata_changed = metadata_changed

    def __setitem__(self, key: str, value: Any) -> None:
        updated = key not in self or self[key] != value
        changed = key in self._FORWARD_FIELDS and updated
        super().__setitem__(key, value)
        if changed:
            self._changed()
        elif updated and self._metadata_changed is not None:
            self._metadata_changed()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if key in self._FORWARD_FIELDS:
            self._changed()
        elif self._metadata_changed is not None:
            self._metadata_changed()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default

    def popitem(self) -> Tuple[str, Any]:
        key, value = super().popitem()
        if key in self._FORWARD_FIELDS:
            self._changed()
        elif self._metadata_changed is not None:
            self._metadata_changed()
        return key, value

    def clear(self) -> None:
        updated = bool(self)
        changed = any(key in self for key in self._FORWARD_FIELDS)
        super().clear()
        if changed:
            self._changed()
        elif updated and self._metadata_changed is not None:
            self._metadata_changed()

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other: Mapping[str, Any]):
        self.update(other)
        return self

    def __copy__(self) -> Dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[str, Any]:
        copied = copy.deepcopy(dict(self), memo)
        memo[id(self)] = copied
        return copied


class _RevisionedSynapses(_RevisionedNodes):
    """Eager graph whose direct record edits also invalidate recall."""

    def __init__(self, values: Optional[Mapping[str, Dict[str, Any]]] = None):
        super().__init__()
        for key, value in (values or {}).items():
            dict.__setitem__(self, key, self._wrap(key, value))

    def _changed(self, key: Optional[str] = None) -> None:
        self.graph_revision += 1
        observer = getattr(self, "_recall_graph_observer", None)
        if callable(observer):
            observer(key, self.graph_revision)

    def _wrap(self, key: str, value: Mapping[str, Any]) -> _TrackedSynapseRecord:
        return _TrackedSynapseRecord(value, lambda: self._changed(key))

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        super().__setitem__(key, self._wrap(key, value))


def _is_nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _synapse_id_matches_endpoints(
    record_id: str,
    source_id: str,
    target_id: str,
) -> bool:
    return bool(
        record_id
        and source_id
        and target_id
        and record_id.startswith("%s>%s:" % (source_id, target_id))
    )


def _synapse_id_endpoints(record_id: str) -> Optional[Tuple[str, str]]:
    edge, separator, _kind = str(record_id).partition(":")
    source_id, arrow, target_id = edge.partition(">")
    if not separator or not arrow or not source_id or not target_id:
        return None
    return source_id, target_id


def _pack_ternary_levels(values: Iterable[int]) -> bytes:
    packed = bytearray()
    byte = 0
    lane = 0
    for raw in values:
        value = NeuralSubstrate.exact_effective_weight(raw)
        byte |= (value + 1) << (lane * 2)
        lane += 1
        if lane == 4:
            packed.append(byte)
            byte = 0
            lane = 0
    if lane:
        # Canonical padding is ternary zero (code 01), never -1.
        for index in range(lane, 4):
            byte |= 1 << (index * 2)
        packed.append(byte)
    return bytes(packed)


def _unpack_ternary_level(packed: bytes, index: int) -> int:
    code = (packed[index // 4] >> ((index % 4) * 2)) & 0x03
    if code == 3:
        raise ValueError("packed dynamic synapse contains a reserved code")
    return code - 1


def _unpack_persisted_synapse_weights(
    tensors: Dict[str, torch.Tensor], count: int, version: int
) -> torch.Tensor:
    """Normalize one bounded shard for in-memory use after exact validation."""

    if version == 1:
        weights = tensors.get("effective_weight")
        if weights is None or int(weights.numel()) != count:
            raise ValueError("synapse tensor shard is missing effective_weight")
        for value in weights.reshape(-1):
            NeuralSubstrate.exact_effective_weight(value.item())
        tensors.pop("latent_weight", None)
        return weights.to(torch.int8)
    if "effective_weight" in tensors or "latent_weight" in tensors:
        raise ValueError("current synapse shard contains a duplicate weight")
    packed = tensors.get("packed_effective_weight")
    if (
        packed is None
        or packed.dtype != torch.uint8
        or int(packed.numel()) != (count + 3) // 4
    ):
        raise ValueError("packed synapse tensor shard is invalid")
    raw = bytes(packed.reshape(-1).tolist())
    levels = [_unpack_ternary_level(raw, index) for index in range(count)]
    for index in range(count, len(raw) * 4):
        if _unpack_ternary_level(raw, index) != 0:
            raise ValueError("packed synapse tensor padding is noncanonical")
    return torch.tensor(levels, dtype=torch.int8)


def _forward_hot_ids_sha256(values: Iterable[str], algorithm: Optional[str] = None) -> str:
    from .paged_assembly_membership import ALGORITHM, checksum_for_ids
    checksum = getattr(values, "ids_sha256", None)
    actual_algorithm = getattr(values, "ids_checksum_algorithm", "sha256-sorted-ids-v1")
    chosen = actual_algorithm if algorithm is None else algorithm
    if chosen == ALGORITHM:
        if callable(checksum) and actual_algorithm == chosen:
            value = checksum()
            if not _is_sha256(value):
                raise ValueError("paged hot membership Merkle checksum is invalid")
            return value
        return checksum_for_ids(values)
    if chosen != "sha256-sorted-ids-v1":
        raise ValueError("forward-index membership checksum algorithm is unsupported")
    sorted_ids = getattr(values, "iter_sorted_ids", None)
    if callable(sorted_ids):
        # A paged assembly index supplies canonical ID order directly. The
        # delimiter and UTF-8 encoding reproduce the resident set hash while
        # keeping only one ID in memory at a time.
        digest = hashlib.sha256()
        previous: Optional[str] = None
        for raw in sorted_ids():
            identifier = str(raw)
            if not identifier or (previous is not None and identifier <= previous):
                raise ValueError("paged hot node IDs are not sorted and unique")
            if previous is not None:
                digest.update(b"\0")
            digest.update(identifier.encode("utf-8"))
            previous = identifier
        return digest.hexdigest()
    payload = "\0".join(sorted({str(value) for value in values if str(value)}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _forward_index_path(root: Path, generation: str) -> Path:
    return NeuralSubstrate._safe_store_path(
        Path(root).resolve(),
        "forward-index/generations/%s.json" % str(generation),
    )


def _forward_index_entry(
    descriptor: Mapping[str, Any],
    group: Sequence[Tuple[str, Mapping[str, Any]]],
    hot_node_ids: set[str],
) -> Dict[str, Any]:
    structures: List[List[Any]] = []
    levels: List[int] = []
    hot_locations: List[List[Any]] = []
    synaptic_uses = 0
    for index, (record_id, record) in enumerate(group):
        source = str(record.get("source_id", ""))
        target = str(record.get("target_id", ""))
        weight = NeuralSubstrate.exact_effective_weight(
            record.get("effective_weight", 0)
        )
        uses = record.get("uses", 0)
        if isinstance(uses, bool) or not isinstance(uses, int) or uses < 0:
            raise ValueError("substrate forward index synaptic uses are invalid")
        synaptic_uses += uses
        if weight:
            structures.append([index, record_id, source, target])
            levels.append(weight)
        if source in hot_node_ids:
            hot_locations.append([source, record_id, index])
        if target in hot_node_ids and target != source:
            hot_locations.append([target, record_id, index])
    record_spec = descriptor.get("records")
    tensor_spec = descriptor.get("tensors")
    if not isinstance(record_spec, Mapping) or not isinstance(tensor_spec, Mapping):
        raise ValueError("substrate forward index needs complete shard descriptors")
    identifiers = [record_id for record_id, _record in group]
    return {
        "bucket": str(descriptor["bucket"]),
        "part": int(descriptor["part"]),
        "count": int(descriptor["count"]),
        "recordsSha256": str(record_spec.get("sha256", "")),
        "tensorsSha256": str(tensor_spec.get("sha256", "")),
        "firstId": identifiers[0] if identifiers else None,
        "lastId": identifiers[-1] if identifiers else None,
        "synapticUses": synaptic_uses,
        "forwardRecords": structures,
        "packedEffectiveWeights": base64.b64encode(
            _pack_ternary_levels(levels)
        ).decode("ascii"),
        "hotLocations": sorted(hot_locations),
    }


def _write_forward_index(
    root: Path,
    *,
    generation: str,
    generation_manifest_sha256: str,
    synapse_count: int,
    records_per_shard: int,
    hot_node_ids: Iterable[str],
    entries: Sequence[Mapping[str, Any]],
    growth_guard: Optional[Callable[[int], bool]],
    verified_rebuild: bool = False,
) -> Dict[str, Any]:
    from .paged_forward_index import publish_entries
    normalized_entries = publish_entries(Path(root), entries, growth_guard)
    body = {
        "format": _FORWARD_INDEX_FORMAT, "formatVersion": _FORWARD_INDEX_VERSION,
        "sourceGeneration": str(generation),
        "sourceGenerationManifestSha256": str(generation_manifest_sha256),
        "synapseCount": int(synapse_count), "recordsPerShard": int(records_per_shard),
        "hotNodeIdsSha256": _forward_hot_ids_sha256(hot_node_ids),
        "hotNodeIdsChecksumAlgorithm": getattr(hot_node_ids, "ids_checksum_algorithm", "sha256-sorted-ids-v1"),
        "synapticUses": sum(int(value["synapticUses"]) for value in normalized_entries),
        "shards": normalized_entries,
    }
    encoded = NeuralSubstrate._canonical_json(body)
    manifest = {**body, "contentSha256": hashlib.sha256(encoded).hexdigest()}
    payload = NeuralSubstrate._canonical_json(manifest)
    if growth_guard is not None and not growth_guard(len(payload) + 4096):
        raise SubstrateResourcePause("derived forward descriptor manifest paused at resource reserve")
    path = _forward_index_path(root, generation)
    if path.exists() and not verified_rebuild:
        existing = path.read_bytes()
        if existing != payload:
            try:
                prior = json.loads(existing)
                prior_body = {key: value for key, value in prior.items() if key != "contentSha256"}
                algorithm_migration = bool(
                    isinstance(prior, dict) and prior.get("formatVersion") == 4
                    and prior.get("hotNodeIdsChecksumAlgorithm") in {"sha256-sorted-ids-v1", "sha256-patricia-id-set-v1"}
                    and prior.get("hotNodeIdsChecksumAlgorithm") != body["hotNodeIdsChecksumAlgorithm"]
                )
                if (
                    prior.get("format") != _FORWARD_INDEX_FORMAT
                    or (prior.get("formatVersion") not in {1, 2, 3} and not algorithm_migration)
                    or prior.get("sourceGeneration") != generation
                    or prior.get("sourceGenerationManifestSha256") != generation_manifest_sha256
                    or prior.get("contentSha256") != hashlib.sha256(NeuralSubstrate._canonical_json(prior_body)).hexdigest()
                    or prior.get("hotNodeIdsSha256") != _forward_hot_ids_sha256(
                        hot_node_ids, prior.get("hotNodeIdsChecksumAlgorithm", "sha256-sorted-ids-v1"))
                ):
                    raise ValueError("derived forward index conflicts with its generation")
                if prior["formatVersion"] >= 2:
                    from .paged_forward_index import iter_entries
                    if any(prior.get(key) != body[key] for key in ("synapseCount", "recordsPerShard", "synapticUses")):
                        raise ValueError("derived forward index count conflicts with its generation")
                    prior_entries = prior.get("shards")
                    if not isinstance(prior_entries, list) or len(prior_entries) != len(normalized_entries):
                        raise ValueError("derived forward index group count conflicts")
                    if prior["formatVersion"] == 4:
                        matches = prior_entries == normalized_entries
                    else:
                        matches = not any(old != new for old, new in zip(prior_entries, iter_entries(Path(root), manifest)))
                    if not matches:
                        raise ValueError("derived legacy forward entries conflict")
            except (TypeError, UnicodeError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError("derived forward index conflicts with its generation") from error
        else:
            return manifest
    atomic_write_bytes(path, payload)
    return manifest




def _load_forward_index(
    root: Path,
    *,
    generation: str,
    generation_manifest_sha256: str,
    synapse_count: int,
    records_per_shard: int,
    hot_node_ids: Iterable[str],
    descriptors: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    path = _forward_index_path(root, generation)
    try:
        # Large old inline indexes are expendable. Reconstruct their complete
        # checked entries from bounded canonical source shards into cache v4
        # rather than allocate a corpus-sized JSON tree. No neural record is
        # skipped and authoritative v3 generation hashes do not change.
        if path.stat().st_size > 8 * 1024 * 1024:
            with path.open("rb") as handle:
                header = handle.read(4096)
            if b'"formatVersion":4' not in header:
                return None
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("substrate forward index is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("substrate forward index is invalid")
    if payload != NeuralSubstrate._canonical_json(value):
        raise ValueError("substrate forward index is not canonical")
    body = {key: item for key, item in value.items() if key != "contentSha256"}
    descriptor_map: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for item in descriptors:
        bucket = item.get("bucket")
        part = item.get("part")
        if (
            not isinstance(bucket, str)
            or len(bucket) != 1
            or bucket not in "0123456789abcdef"
            or not _is_nonnegative_int(part)
            or (bucket, part) in descriptor_map
        ):
            raise ValueError("substrate forward index descriptor table is invalid")
        descriptor_map[(bucket, part)] = item
    entries = value.get("shards")
    index_version = value.get("formatVersion")
    legacy_index = index_version == 1
    if (
        value.get("format") != _FORWARD_INDEX_FORMAT
        or type(index_version) is not int
        or index_version not in {1, 2, 3, _FORWARD_INDEX_VERSION}
        or (
            index_version < 3 and "hotNodeIdsChecksumAlgorithm" in value
        )
        or (
            index_version >= 3 and value.get("hotNodeIdsChecksumAlgorithm") not in {
                "sha256-sorted-ids-v1", "sha256-patricia-id-set-v1",
            }
        )
        or value.get("sourceGeneration") != generation
        or not _is_sha256(value.get("sourceGeneration"))
        or value.get("sourceGenerationManifestSha256")
        != generation_manifest_sha256
        or not _is_sha256(value.get("sourceGenerationManifestSha256"))
        or value.get("synapseCount") != int(synapse_count)
        or not _is_nonnegative_int(value.get("synapseCount"))
        or value.get("recordsPerShard") != int(records_per_shard)
        or not _is_nonnegative_int(value.get("recordsPerShard"))
        or value.get("hotNodeIdsSha256")
        != _forward_hot_ids_sha256(
            hot_node_ids, value.get("hotNodeIdsChecksumAlgorithm", "sha256-sorted-ids-v1")
        )
        or not _is_sha256(value.get("hotNodeIdsSha256"))
        or not _is_sha256(value.get("contentSha256"))
        or value.get("contentSha256")
        != hashlib.sha256(NeuralSubstrate._canonical_json(body)).hexdigest()
        or not isinstance(entries, list)
        or len(entries) != len(descriptor_map)
    ):
        raise ValueError("substrate forward index is stale or corrupt")
    seen: set[Tuple[str, int]] = set()
    observed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("substrate forward index shard is invalid")
        bucket = entry.get("bucket")
        part = entry.get("part")
        if not isinstance(bucket, str) or not _is_nonnegative_int(part):
            raise ValueError("substrate forward index shard is invalid")
        key = (bucket, part)
        descriptor = descriptor_map.get(key)
        record_spec = descriptor.get("records") if descriptor else None
        tensor_spec = descriptor.get("tensors") if descriptor else None
        if (
            descriptor is None
            or key in seen
            or not isinstance(record_spec, Mapping)
            or not isinstance(tensor_spec, Mapping)
            or entry.get("count") != descriptor.get("count")
            or not _is_nonnegative_int(entry.get("count"))
            or entry.get("recordsSha256") != record_spec.get("sha256")
            or not _is_sha256(entry.get("recordsSha256"))
            or entry.get("tensorsSha256") != tensor_spec.get("sha256")
            or not _is_sha256(entry.get("tensorsSha256"))
            or (index_version < 4 and (
                not isinstance(entry.get("forwardRecords"), list)
                or not isinstance(entry.get("hotLocations"), list)
                or not isinstance(entry.get("packedEffectiveWeights"), str)
            ))
            or (index_version == 4 and (
                not isinstance(entry.get("entry"), dict)
                or not _is_sha256(entry["entry"].get("sha256"))
                or entry["entry"].get("path") != "forward-index/blobs/%s.json" % entry["entry"].get("sha256", "")
                or not _is_nonnegative_int(entry["entry"].get("bytes"))
                or not _is_nonnegative_int(entry.get("forwardCount"))
                or entry["forwardCount"] > entry["count"]
                or entry.get("packedBytes") != (entry["forwardCount"] + 3) // 4
            ))
            or (
                not legacy_index
                and not _is_nonnegative_int(entry.get("synapticUses"))
            )
        ):
            raise ValueError("substrate forward index shard binding is invalid")
        observed += int(entry["count"])
        seen.add(key)
    if observed != int(synapse_count):
        raise ValueError("substrate forward index count diverges")
    if legacy_index:
        # A valid v1 cache can be upgraded only by scanning its checksummed
        # authoritative shards. Tampered/stale v1 content already failed above.
        return None
    if (
        not _is_nonnegative_int(value.get("synapticUses"))
        or value.get("synapticUses")
        != sum(int(entry["synapticUses"]) for entry in entries)
    ):
        raise ValueError("substrate forward index synaptic uses diverge")
    return value


def _now() -> float:
    return time.time()


class SubstrateResourcePause(RuntimeError):
    """Raised when the host reserve asks structural growth to pause."""


class HypervectorSpace:
    """Deterministic bipolar hypervectors with bind, bundle, and permutation."""

    def __init__(self, dimensions: int = 256, seed: int = 7):
        if dimensions < 16:
            raise ValueError("hypervector dimensions must be at least 16")
        self.dimensions = int(dimensions)
        self.seed = int(seed)

    def symbol(self, name: str) -> torch.Tensor:
        digest = hashlib.sha256(
            ("%d:%s" % (self.seed, name)).encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFFFFFF)
        values = torch.randint(
            0, 2, (self.dimensions,), generator=generator, dtype=torch.float32
        )
        return values.mul(2.0).sub(1.0)

    @staticmethod
    def bind(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right

    @staticmethod
    def bundle(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not vectors:
            raise ValueError("cannot bundle an empty vector sequence")
        summed = torch.stack(list(vectors)).sum(dim=0)
        bundled = torch.sign(summed)
        bundled[bundled == 0] = 1.0
        return bundled

    @staticmethod
    def weighted_bundle(
        vectors: Sequence[torch.Tensor], weights: Sequence[float]
    ) -> torch.Tensor:
        if not vectors or len(vectors) != len(weights):
            raise ValueError("weighted bundle needs equally sized non-empty inputs")
        stacked = torch.stack([vector.float() for vector in vectors])
        scale = torch.tensor(weights, dtype=stacked.dtype).reshape(-1, 1)
        result = torch.sign((stacked * scale).sum(dim=0))
        result[result == 0] = 1.0
        return result

    @staticmethod
    def permute(vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        if vector.ndim == 0:
            raise ValueError("cannot permute a scalar hypervector")
        return torch.roll(vector, shifts=int(steps), dims=-1)

    @staticmethod
    def inverse_permute(vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        return torch.roll(vector, shifts=-int(steps), dims=-1)

    @staticmethod
    def similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(
            F.cosine_similarity(
                left.float().reshape(1, -1),
                right.float().reshape(1, -1),
            ).item()
        )


class NeuralSubstrate:
    """One growable substrate for neurons, assemblies, and ternary synapses.

    Structural growth is governed by a host-resource callback rather than a
    user-set or implementation-defined cardinality.
    """

    SCHEMA = "neural-substrate-2"

    @staticmethod
    def exact_effective_weight(value: Any) -> int:
        """Return a live synapse weight only when it is exactly ternary.

        Do not coerce before validation: ``int(0.5)`` would otherwise turn a
        corrupt higher-precision weight into a seemingly valid zero. This is
        the only persistent learned weight for a sparse substrate synapse.
        """

        if isinstance(value, (bool, str, bytes)):
            raise ValueError(
                "live substrate synapses must have exact ternary weights"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "live substrate synapses must have exact ternary weights"
            ) from error
        if not math.isfinite(numeric) or numeric not in {-1.0, 0.0, 1.0}:
            raise ValueError(
                "live substrate synapses must have exact ternary weights"
            )
        return int(numeric)

    def __init__(
        self,
        dimensions: int = 256,
        seed: int = 7,
        growth_guard: Optional[Callable[[int], bool]] = None,
    ):
        self.space = HypervectorSpace(dimensions, seed)
        self.neurons: Dict[str, Dict[str, Any]] = _RevisionedNodes()
        self.neuron_vectors = PackedTernaryVectors(dimensions, seed=seed)
        self.assemblies: List[Dict[str, Any]] | PagedAssemblyView = []
        self.assembly_vectors = PackedTernaryVectorView(self.neuron_vectors)
        self.synapses: MutableMapping[str, Dict[str, Any]] = _RevisionedSynapses()
        self._recall_graph_cache: Optional[
            Tuple[
                Tuple[int, int, int, int],
                Dict[str, List[Tuple[str, int]]],
                Dict[str, int],
                int,
                int,
            ]
        ] = None
        self._assembly_index_source: Optional[List[Dict[str, Any]]] = None
        self._assembly_indexed_count = 0
        self._assembly_by_id: Dict[str, Dict[str, Any]] = {}
        self._assembly_by_fingerprint: Dict[str, Dict[str, Any]] = {}
        self._statistical_index_source: Optional[List[Dict[str, Any]]] = None
        self._statistical_indexed_count = 0
        self._statistical_pair_index: Dict[str, set[str]] = defaultdict(set)
        self._statistical_atom_index: Dict[str, set[str]] = defaultdict(set)
        self.growth_events = 0
        self.growth_pauses = 0
        # Monotonic live-state identity for inspection cursors.  Structural
        # counts alone cannot detect an STDP update, recall activation, or
        # decay pass that changes existing records in place.
        self.state_revision = 0
        self.growth_guard = growth_guard
        self.persistence_manifest: Optional[Dict[str, Any]] = None
        # Firing/eligibility is a small epoch-scoped overlay. Learned record
        # fields remain byte-identical across a fresh-attention boundary; an
        # empty overlay makes their prior values logically inactive without
        # rewriting millions of durable substrate records.
        self.attention_epoch = 0
        # Native new state has complete active-ID tracking from epoch zero.
        # Loading an old state without an overlay explicitly restores raw mode.
        self.attention_legacy_raw_active = False
        self.attention_active_neuron_ids: set[str] = set()
        self.attention_recalled_assembly_ids: set[str] = set()
        self.attention_eligible_synapse_ids: set[str] = set()
        # Loaded shard membership is operational persistence state, not neural
        # state. Keeping existing IDs in their prior bounded part prevents one
        # newly learned synapse from shifting and rewriting every later part in
        # the same hash bucket.
        self._persistence_record_groups: Dict[
            Tuple[str, str], Tuple[str, int]
        ] = {}
        self._persistence_records_per_shard = 0
        self._last_recall_audit: Dict[str, Any] = {
            "rule": "effective-ternary-recurrent-settling",
            "exactTernaryContribution": True,
            "latentMagnitudeUsed": False,
            "settledRounds": 0,
            "inhibitoryEdges": 0,
            "inhibitorySignals": 0,
            "suppressedAssemblies": 0,
        }

    def enable_paged_vectors(
        self,
        path: Path,
        *,
        cache_bytes: int = 8 * 1024 * 1024,
        resource_policy: Optional[Any] = None,
        shard_rows: int = 512,
    ) -> Dict[str, Any]:
        """Move one in-memory packed authority into bounded disk-backed rows.

        This is an explicit, one-way working-state transition. It never adopts
        an existing SQLite file without generation validation: a later load
        coordinator must rebuild or reconcile that file from a verified neural
        shard generation. An interrupted migration leaves the original map
        untouched and an orphan cache that cannot advance a brain cursor.
        """

        if isinstance(self.neuron_vectors, PagedPackedVectors):
            if self.neuron_vectors.path.resolve() != Path(path).resolve():
                raise ValueError("substrate is already paged at another path")
            return {
                "vectorCount": len(self.neuron_vectors),
                "storageBytes": self.neuron_vectors.storage_bytes,
                "alreadyPaged": True,
            }
        if not isinstance(self.neuron_vectors, PackedTernaryVectors):
            raise ValueError("substrate has no in-memory packed vector authority")
        if type(shard_rows) is not int or not 1 <= shard_rows <= 4096:
            raise ValueError("paged vector migration shard size is invalid")
        destination = Path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("paged vector cache must be a new path")
        self._validate_packed_vector_identity()
        source = self.neuron_vectors

        def reserve(estimated: int, _operation: str) -> bool:
            return True if self.growth_guard is None else bool(self.growth_guard(estimated))

        reserve_options = (
            # Disk offload is invoked precisely when the host RAM watermark
            # is tight. Reapplying that same watermark to a bounded import
            # buffer would make paging impossible when it is needed most.
            {}
            if resource_policy is not None
            else {"disk_reserve": reserve, "memory_reserve": reserve}
        )
        paged = PagedPackedVectors(
            destination,
            self.space.dimensions,
            seed=source.seed,
            zero_deadband=source.zero_deadband,
            cache_bytes=cache_bytes,
            resource_policy=resource_policy,
            **reserve_options,
        )
        identifiers: List[str] = []

        def import_page() -> None:
            if not identifiers:
                return
            metadata, tensors = source.export_state(keys=identifiers)
            paged.import_state(metadata, tensors)
            for identifier in identifiers:
                if (
                    paged.packed_row(identifier) != source.packed_row(identifier)
                    or paged.update_count(identifier) != source.update_count(identifier)
                ):
                    raise ValueError("paged vector migration changed a learned row")
            identifiers.clear()

        for identifier in source:
            identifiers.append(identifier)
            if len(identifiers) >= int(shard_rows):
                import_page()
        import_page()
        if len(paged) != len(source):
            raise ValueError("paged vector migration missed learned rows")
        view = PackedTernaryVectorView(paged)
        for assembly in self.assemblies:
            view.link(str(assembly.get("id", "")))
        previous_view = self.assembly_vectors
        self.neuron_vectors = paged
        self.assembly_vectors = view
        try:
            self._validate_packed_vector_identity()
        except BaseException:
            self.neuron_vectors = source
            self.assembly_vectors = previous_view
            raise
        return {
            "vectorCount": len(paged),
            "storageBytes": paged.storage_bytes,
            "alreadyPaged": False,
        }

    def enable_paged_assemblies(
        self,
        index: PagedAssemblyIndex,
        *,
        page_size: int = 128,
    ) -> Dict[str, Any]:
        """Move structural assembly metadata into bounded indexed pages.

        The destination must be empty. It may be metadata-only or share the
        exact authoritative neuron-vector object in one SQLite cache. A
        second independently writable assembly-vector copy is forbidden.
        A failed migration leaves the in-memory list active and the partial
        destination unusable for attach.
        """

        if not isinstance(index, PagedAssemblyIndex):
            raise TypeError("paged assembly migration needs an assembly index")
        if isinstance(self.assemblies, PagedAssemblyView):
            if self.assemblies.index.path.resolve() != index.path.resolve():
                raise ValueError("assemblies are already paged at another path")
            return {"assemblyCount": len(self.assemblies), "alreadyPaged": True}
        if not isinstance(self.assemblies, list):
            raise ValueError("substrate assembly metadata is not migratable")
        if index.count() != 0:
            raise ValueError("paged assembly migration needs an empty index")
        if index._vectors is not None and index._vectors is not self.neuron_vectors:
            raise ValueError("paged assembly metadata index must not own vector rows in a second store")
        if index._vectors is self.neuron_vectors and (
            not isinstance(self.neuron_vectors, PagedPackedVectors)
            or index.path.resolve() != self.neuron_vectors.path.resolve()
        ):
            raise ValueError("shared assembly cache has a mismatched vector store")
        self._validate_packed_vector_identity()
        view = PagedAssemblyView(index, page_size=page_size)
        window: List[Mapping[str, Any]] = []
        window_charge = 0
        max_payload_bytes = 8 * 1024 * 1024

        def flush_window() -> None:
            nonlocal window_charge
            if not window:
                return
            with index.batch(
                max_rows=256, max_payload_bytes=max_payload_bytes
            ) as batch:
                for item in window:
                    if not batch.upsert(item):
                        raise ValueError("paged assembly migration found duplicate IDs")
            window.clear()
            window_charge = 0

        for record in self.assemblies:
            identifier, fingerprint, payload, _digest = _record_payload(record)
            charge = 4 * (len(payload) + len(identifier) + len(fingerprint)) + 128
            if charge > max_payload_bytes:
                raise ValueError("assembly record exceeds bounded migration window")
            if window and (
                len(window) >= 256 or window_charge + charge > max_payload_bytes
            ):
                flush_window()
            window.append(record)
            window_charge += charge
        flush_window()
        if len(view) != len(self.assemblies):
            raise ValueError("paged assembly migration missed records")
        original = iter(self.assemblies)
        verified = 0
        for page in index.iter_pages(page_size=page_size):
            for paged_record in page.records:
                if paged_record != next(original, None):
                    raise ValueError("paged assembly migration altered structural metadata")
                verified += 1
        if verified != len(self.assemblies) or next(original, None) is not None:
            raise ValueError("paged assembly migration coverage differs")
        previous_records = self.assemblies
        previous_vectors = self.assembly_vectors
        self.assemblies = view
        if isinstance(self.neuron_vectors, PagedPackedVectors):
            self.assembly_vectors = PagedAssemblyVectorView(index, self.neuron_vectors)
        self.invalidate_assembly_index()
        try:
            if (
                len(self.neuron_vectors) != len(self.neurons)
                or len(self.assembly_vectors) != len(view)
                or (
                    index._vectors is self.neuron_vectors
                    and index.status()["packedVectorRows"] != len(view)
                )
            ):
                raise ValueError("paged assembly migration lost neural rows")
        except BaseException:
            self.assemblies = previous_records
            self.assembly_vectors = previous_vectors
            self.invalidate_assembly_index()
            raise
        return {"assemblyCount": len(view), "alreadyPaged": False}

    def attach_verified_paged_assemblies(
        self,
        index: PagedAssemblyIndex,
        *,
        page_size: int = 128,
    ) -> Dict[str, Any]:
        """Attach a *completed* cache rebuilt from this committed generation.

        Unlike empty-index migration, this method never rewrites metadata or
        vectors. It checks the committed binding and compares every existing
        in-memory record against a bounded index page before dropping the
        resident assembly list. A direct load is still not bounded until the
        loader itself streams assembly metadata into this cache.
        """

        if not isinstance(index, PagedAssemblyIndex):
            raise TypeError("verified assembly attach needs a paged index")
        if not isinstance(self.assemblies, list):
            raise ValueError("verified assembly attach needs a resident source list")
        if (
            not isinstance(self.neuron_vectors, PagedPackedVectors)
            or index._vectors is not self.neuron_vectors
            or index.path.resolve() != self.neuron_vectors.path.resolve()
        ):
            raise ValueError("verified assembly cache must share the exact neuron vector store")
        expected_generation = (
            self.persistence_manifest.get("activeGeneration")
            if isinstance(self.persistence_manifest, dict) else None
        )
        status = index.status()
        if (
            not isinstance(expected_generation, str)
            or status["dirtySinceCommit"]
            or status["committedGenerationSha256"] != expected_generation
            or status["count"] != len(self.assemblies)
            or status["packedVectorRows"] != status["count"]
            or len(self.neuron_vectors) != len(self.neurons)
        ):
            raise ValueError("assembly cache is not bound to this committed generation")
        view = PagedAssemblyView(index, page_size=page_size)
        original = iter(self.assemblies)
        verified = 0
        for page in index.iter_pages(page_size=page_size):
            for paged_record in page.records:
                resident = next(original, None)
                if resident is None:
                    raise ValueError("verified assembly cache has excess records")
                _identifier, _fingerprint, projected, _digest = _record_payload(resident)
                if paged_record != json.loads(projected):
                    raise ValueError("verified assembly cache changed structural metadata")
                verified += 1
        if verified != status["count"] or next(original, None) is not None:
            raise ValueError("verified assembly cache missed records")
        final = index.status()
        if any(
            final[key] != status[key]
            for key in ("storeId", "count", "indexRevision", "vectorRevision",
                        "committedGenerationSha256", "dirtySinceCommit")
        ):
            raise ValueError("verified assembly cache changed during attach")
        previous_records = self.assemblies
        previous_vectors = self.assembly_vectors
        self.assemblies = view
        self.assembly_vectors = PagedAssemblyVectorView(index, self.neuron_vectors)
        self.invalidate_assembly_index()
        try:
            if len(self.assembly_vectors) != verified:
                raise ValueError("verified assembly view count differs")
        except BaseException:
            self.assemblies = previous_records
            self.assembly_vectors = previous_vectors
            self.invalidate_assembly_index()
            raise
        return {"assemblyCount": verified, "alreadyPaged": False}

    @property
    def concepts(self) -> Dict[str, Dict[str, Any]]:
        """Inspector view over substrate neurons (not separate storage)."""

        return self.neurons

    @property
    def concept_vectors(self) -> MutableMapping[str, torch.Tensor]:
        return self.neuron_vectors

    @property
    def ideas(self) -> List[Dict[str, Any]]:
        """Inspector view over distributed assemblies."""

        return self.assemblies

    @property
    def idea_vectors(self) -> MutableMapping[str, torch.Tensor]:
        return self.assembly_vectors

    @property
    def assembly_by_id(self) -> Mapping[str, Dict[str, Any]]:
        """Exact append-aware lookup; duplicate assembly ids are last-wins.

        Runtime growth appends. A replaced or shortened list is rebuilt, while
        new rows update a warm index without rescanning older assemblies.
        Call ``invalidate_assembly_index`` after external in-place reordering
        or same-length replacement.
        """

        if isinstance(self.assemblies, PagedAssemblyView):
            return _PagedAssemblyLookup(self.assemblies, "id")

        if (
            self._assembly_index_source is not self.assemblies
            or self._assembly_indexed_count > len(self.assemblies)
        ):
            self._assembly_by_id = {}
            self._assembly_by_fingerprint = {}
            self._assembly_index_source = self.assemblies
            self._assembly_indexed_count = 0
        if self._assembly_indexed_count < len(self.assemblies):
            for record in self.assemblies[self._assembly_indexed_count :]:
                identifier = str(record.get("id", ""))
                if identifier:
                    self._assembly_by_id[identifier] = record
                fingerprint = str(record.get("fingerprint", ""))
                if fingerprint:
                    # Existing exact-repeat admission historically selected
                    # the first record when duplicate fingerprints existed.
                    self._assembly_by_fingerprint.setdefault(fingerprint, record)
            self._assembly_indexed_count = len(self.assemblies)
        return self._assembly_by_id

    @property
    def assembly_by_fingerprint(self) -> Mapping[str, Dict[str, Any]]:
        """Append-aware exact-repeat lookup without a corpus-sized scan."""

        if isinstance(self.assemblies, PagedAssemblyView):
            return _PagedAssemblyLookup(self.assemblies, "fingerprint")
        _ = self.assembly_by_id
        return self._assembly_by_fingerprint

    def _edit_assembly_by_id(
        self, identifier: str, mutator: Callable[[Dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        """Persist one bounded metadata edit or mutate the in-memory record."""

        if isinstance(self.assemblies, PagedAssemblyView):
            with self.assemblies.transaction(max_rows=1) as edits:
                return edits.edit_by_id(identifier, mutator)
        record = self.assembly_by_id.get(identifier)
        if record is None:
            raise KeyError(identifier)
        mutator(record)
        return record

    def invalidate_assembly_index(self) -> None:
        """Invalidate after an external non-append assembly list mutation."""

        self._assembly_index_source = None
        self._assembly_indexed_count = 0
        self._assembly_by_id = {}
        self._assembly_by_fingerprint = {}
        self._statistical_index_source = None
        self._statistical_indexed_count = 0
        self._statistical_pair_index = defaultdict(set)
        self._statistical_atom_index = defaultdict(set)

    @staticmethod
    def _statistical_pair_keys(neuron_ids: Sequence[str]) -> List[str]:
        """Local unordered relations used only to route to neural fields."""

        return list(
            dict.fromkeys(
                "pair:%s:%s" % tuple(sorted((source, target)))
                for source, target in zip(neuron_ids, neuron_ids[1:])
                if source != target
            )
        )

    def _refresh_statistical_index(self) -> None:
        """Index seed atoms of real field assemblies, never source text."""

        if isinstance(self.assemblies, PagedAssemblyView):
            # A full Python postings map would negate metadata paging. The
            # paged candidate path below scans bounded pages instead.
            return

        if (
            self._statistical_index_source is not self.assemblies
            or self._statistical_indexed_count > len(self.assemblies)
        ):
            # A direct load/replacement pays this once. Index memory is
            # explicitly subject to the same physical growth reserve.
            estimated = sum(
                128 + 320 * len(record.get("statistical_anchor_ids", []))
                for record in self.assemblies
                if record.get("compressed_field")
                and isinstance(record.get("statistical_anchor_ids"), list)
            )
            if estimated:
                self._check_growth(estimated)
            self._statistical_pair_index = defaultdict(set)
            self._statistical_atom_index = defaultdict(set)
            self._statistical_index_source = self.assemblies
            self._statistical_indexed_count = 0
        for record in self.assemblies[self._statistical_indexed_count :]:
            field_id = str(record.get("id", ""))
            anchors = record.get("statistical_anchor_ids")
            if (
                not record.get("compressed_field")
                or not field_id
                or not isinstance(anchors, list)
                or not all(isinstance(value, str) and value for value in anchors)
            ):
                continue
            for key in self._statistical_pair_keys(anchors):
                self._statistical_pair_index[key].add(field_id)
            for neuron_id in anchors:
                self._statistical_atom_index[neuron_id].add(field_id)
        self._statistical_indexed_count = len(self.assemblies)

    @property
    def relations(self) -> Dict[str, Dict[str, Any]]:
        """Inspector view over the authoritative synapse store."""

        return self.synapses

    @property
    def capacity_expansions(self) -> int:
        return self.growth_events

    @staticmethod
    def extract_concepts(text: str) -> List[str]:
        """Extract all ordered atomic and compositional units.

        Stable OmniCortex has no fixed 48-item cutoff and exposes no count-
        truncation argument that a caller could accidentally apply to neural
        learning.
        """

        words = [
            match.group(0).lower().strip(".'") for match in _WORD.finditer(text)
        ]
        words = [word for word in words if len(word) >= 2]
        # ``dict`` preserves first occurrence.  This stays O(n) for records
        # with many unique atoms; the prior ``words.index`` sort was O(n²).
        ordered_atoms = list(dict.fromkeys(words))
        units: List[str] = []
        seen = set()

        def add(unit: str) -> None:
            if unit in seen:
                return
            seen.add(unit)
            units.append(unit)

        for word in ordered_atoms:
            add(word)
        for width in (2, 3):
            for index in range(0, max(0, len(words) - width + 1)):
                unit = "::".join(words[index : index + width])
                if len(set(words[index : index + width])) == 1:
                    continue
                add(unit)
        return units

    @staticmethod
    def extract_ordered_atomic_concepts(text: str) -> List[str]:
        """Return every semantic atom in local order, including repetition.

        Persistent semantic-neuron allocation deliberately deduplicates atoms,
        but transient sequence cues must not.  Repetition can be meaningful to
        a local phrase (for example, the second occurrence of a word may be
        part of the phrase used by a later question), so cue construction uses
        this ordered view before converting it to a distributed vector.
        """

        return [
            word
            for match in _WORD.finditer(text)
            if len(word := match.group(0).lower().strip(".'")) >= 2
        ]

    @classmethod
    def extract_atomic_concepts(cls, text: str) -> List[str]:
        """Return ordered semantic atoms without allocating phrase labels.

        A normal interactive experience can form explicit compositional
        assemblies.  A multi-million-record corpus instead represents phrases
        through sparse co-activation synapses between these atoms.  This
        avoids making a permanent bigram and trigram neuron for every row
        while preserving order, recurrence, exposure, and local structure in
        the authoritative neural graph.
        """

        ordered: List[str] = []
        seen = set()
        for word in cls.extract_ordered_atomic_concepts(text):
            if word in seen:
                continue
            seen.add(word)
            ordered.append(word)
        return ordered

    @staticmethod
    def _segments(text: str) -> List[str]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            return []
        segments = [part.strip() for part in _SEGMENT.split(clean) if part.strip()]
        return segments or [clean]

    def _check_growth(self, estimated_bytes: int) -> None:
        if self.growth_guard is None:
            return
        if not self.growth_guard(max(1, int(estimated_bytes))):
            self.growth_pauses += 1
            raise SubstrateResourcePause(
                "neural substrate growth paused at the host resource reserve"
            )

    def _mark_state_changed(self) -> None:
        self.state_revision += 1

    def edit_neuron_by_id(
        self, neuron_id: str, mutator: Callable[[Dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        """Mutate one neuron atomically when metadata is disk-paged.

        A paged read is immutable. Never hand a detached row to callers who
        expect their in-place edit to change learned state. The resident path
        retains its original object identity and mutation behavior.
        """

        if isinstance(self.neurons, PagedNeuronMetadata):
            return self.neurons.edit_by_id(neuron_id, mutator)
        record = self.neurons[neuron_id]
        mutator(record)
        return record

    def mark_attention_neuron(self, identifier: str) -> None:
        # Epoch zero reads its raw activation fields directly. Recording every
        # historically touched id as well would make the first boundary scale
        # with the whole substrate for no semantic benefit.
        if not self.attention_legacy_raw_active and identifier:
            self.attention_active_neuron_ids.add(str(identifier))

    def mark_attention_assembly(self, identifier: str) -> None:
        if not self.attention_legacy_raw_active and identifier:
            self.attention_recalled_assembly_ids.add(str(identifier))

    def mark_attention_synapse(self, identifier: str) -> None:
        if not self.attention_legacy_raw_active and identifier:
            self.attention_eligible_synapse_ids.add(str(identifier))

    def _ensure_neuron(
        self, label: str, timestamp: float, region: str = "semantic"
    ) -> str:
        neuron_id = hashlib.sha256(
            ("%s:%s" % (region, label)).encode("utf-8")
        ).hexdigest()[:24]
        record = self.neurons.get(neuron_id)
        if record is None:
            record = {
                "id": neuron_id,
                "neuron_id": neuron_id,
                "label": label,
                "region": region,
                "activation": 0.0,
                "importance": 0.1,
                "uncertainty": 0.5,
                "exposures": 0,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            # The new record can be fully activated before its first paged
            # commit, avoiding a second WAL write per novel neuron.
            record["activation"] = min(
                1.0, self.effective_activation(record) * 0.68 + 0.32
            )
            record["importance"] = min(
                1.0, float(record["importance"]) + 1.0 / (10.0 + record["exposures"])
            )
            record["exposures"] += 1
            self.neurons[neuron_id] = record
            self.neuron_vectors[neuron_id] = self.space.symbol(
                "%s-neuron:%s" % (region, label)
            )
            self.growth_events += 1
        else:
            def activate(existing: Dict[str, Any]) -> None:
                existing["activation"] = min(
                    1.0, self.effective_activation(existing) * 0.68 + 0.32
                )
                existing["importance"] = min(
                    1.0,
                    float(existing["importance"])
                    + 1.0 / (10.0 + existing["exposures"]),
                )
                existing["exposures"] += 1
                existing["last_activated_at"] = timestamp

            self.edit_neuron_by_id(neuron_id, activate)
        self.mark_attention_neuron(neuron_id)
        self._mark_state_changed()
        return neuron_id

    def vector_for_labels(self, labels: Sequence[str]) -> torch.Tensor:
        if not labels:
            return F.normalize(
                self.space.symbol("empty-assembly").reshape(1, -1), dim=-1
            )[0]
        vectors = []
        for index, label in enumerate(labels):
            neuron_id = hashlib.sha256(
                ("semantic:%s" % label).encode("utf-8")
            ).hexdigest()[:24]
            neuron = self.neuron_vectors.get(
                neuron_id, self.space.symbol("semantic-neuron:" + label)
            )
            role = self.space.symbol("position:%d" % index)
            vectors.append(
                self.space.permute(self.space.bind(neuron, role), steps=index + 1)
            )
        # The cue is transient activity, not a synaptic weight. Keep its
        # continuous superposition at unit norm to match the normalized
        # read view of packed neuron and assembly rows, including sensory
        # assemblies, without persisting a second learned scale.
        activity = torch.stack(vectors).mean(dim=0)
        if float(activity.norm()) <= 1e-8:
            activity = self.space.bundle(vectors)
        return F.normalize(activity.reshape(1, -1), dim=-1)[0]

    def vector_for_text(self, text: str) -> torch.Tensor:
        return self.vector_for_labels(self.extract_concepts(text))

    def _continuous_assembly_activity(
        self, neuron_ids: Sequence[str]
    ) -> torch.Tensor:
        """Compose the *current* distributed activity without sign quantization."""

        pieces = []
        for index, neuron_id in enumerate(neuron_ids):
            neuron = self.neuron_vectors[neuron_id]
            role = self.space.symbol("position:%d" % index)
            pieces.append(
                self.space.permute(self.space.bind(neuron, role), steps=index + 1)
            )
        return torch.stack(pieces).mean(dim=0)

    def _adapt_coactive_vectors(
        self, neuron_ids: Sequence[str], importance: float
    ) -> None:
        """Move neuron traces along recently eligible ternary pathways.

        The local synapses are strengthened first. Their exact effective
        levels and eligibility then gate a small simultaneous update of each
        participating neuron's distributed vector. A snapshot prevents label
        order from causing one neuron's just-updated state to overwrite the
        context seen by its neighbors. No source text or answer is stored.
        """

        if len(neuron_ids) < 2:
            return
        prior = {
            neuron_id: self.neuron_vectors[neuron_id]
            for neuron_id in neuron_ids
        }
        changes: Dict[str, Tuple[torch.Tensor, float]] = {}
        base_rate = 0.02 + 0.05 * max(0.0, min(1.0, float(importance)))
        for index, neuron_id in enumerate(neuron_ids):
            contexts: List[torch.Tensor] = []
            strengths: List[float] = []
            for neighbor_index in range(max(0, index - 2), min(len(neuron_ids), index + 3)):
                if neighbor_index == index:
                    continue
                neighbor_id = neuron_ids[neighbor_index]
                if neighbor_id == neuron_id:
                    continue
                kind = "co-activates" if neighbor_index > index else "associates"
                synapse = self.synapses.get(
                    "%s>%s:%s" % (neuron_id, neighbor_id, kind)
                )
                if synapse is None:
                    continue
                effective = self.exact_effective_weight(
                    synapse.get("effective_weight", 0)
                )
                eligibility = self.effective_eligibility(synapse)
                if effective == 0 or eligibility <= 0.0:
                    continue
                contexts.append(
                    self.space.permute(
                        prior[neighbor_id], steps=neighbor_index - index
                    )
                )
                strengths.append(effective * eligibility)
            magnitude = sum(abs(value) for value in strengths)
            if not contexts or magnitude <= 1e-8:
                continue
            context = sum(
                (item * weight for item, weight in zip(contexts, strengths)),
                torch.zeros_like(prior[neuron_id]),
            ) / magnitude
            exposure = max(0, int(self.neurons[neuron_id].get("exposures", 0)))
            rate = base_rate / (1.0 + 0.05 * math.log1p(exposure))
            changes[neuron_id] = (context, rate)
        if changes:
            for neuron_id, (target, rate) in changes.items():
                self.neuron_vectors.adapt(neuron_id, target, rate)
            self._mark_state_changed()

    def _adapt_assembly_vector(
        self, assembly_id: str, target: torch.Tensor, rate: float
    ) -> torch.Tensor:
        # The assembly view and neuron map refer to the same packed row.
        integrated = self.assembly_vectors.adapt(assembly_id, target, rate)
        self._mark_state_changed()
        return integrated

    def _strengthen_synapse(
        self,
        source: str,
        target: str,
        timestamp: float,
        *,
        kind: str,
        amount: float,
    ) -> None:
        if source == target:
            return
        synapse_id = "%s>%s:%s" % (source, target, kind)
        synapse = self.synapses.get(synapse_id)
        if synapse is None:
            # A first exposure creates an immediately usable, exact ternary
            # pathway. No duplicate floating-point synaptic weight is kept.
            synapse = {
                "id": synapse_id,
                "source_id": source,
                "target_id": target,
                "kind": kind,
                "effective_weight": (1 if amount > 0 else (-1 if amount < 0 else 0)),
                "eligibility": 0.0,
                "plasticity": 1.0,
                "uses": 0,
                "stability": 0.0,
                "last_updated_at": timestamp,
            }
            self.synapses[synapse_id] = synapse
            synapse = self.synapses[synapse_id]
            self.growth_events += 1
            created = True
        else:
            created = False
        synapse["uses"] += 1
        synapse["stability"] = min(
            20.0, float(synapse.get("stability", 0.0)) + 0.01
        )
        local_rate = float(synapse.get("plasticity", 1.0)) / (
            1.0 + float(synapse["stability"])
        )
        # Signed eligibility is a decaying spike-timing trace, not a second
        # synaptic weight. Subthreshold events accumulate in this short-lived
        # state until they cause an exact {-1,0,+1} transition. The trace can
        # be cleared with attention without erasing the learned pathway.
        timing = max(
            -1.0,
            min(
                1.0,
                self._signed_eligibility(synapse) * 0.8
                + amount * local_rate,
            ),
        )
        weight = self.exact_effective_weight(synapse["effective_weight"])
        if not created and timing >= 0.25 and weight < 1:
            weight += 1
            timing -= 0.25
        elif not created and timing <= -0.25 and weight > -1:
            weight -= 1
            timing += 0.25
        synapse["eligibility"] = timing
        synapse["effective_weight"] = weight
        synapse["last_updated_at"] = timestamp
        self.mark_attention_synapse(synapse_id)
        self._mark_state_changed()

    def _wire_local_structure(
        self, neuron_ids: Sequence[str], timestamp: float
    ) -> None:
        # Local sparse wiring scales linearly with experience length.  It
        # preserves order and co-activation without constructing an O(n²)
        # clique for a long document.
        for index, source in enumerate(neuron_ids):
            for distance, target in enumerate(
                neuron_ids[index + 1 : index + 9], start=1
            ):
                self._strengthen_synapse(
                    source,
                    target,
                    timestamp,
                    kind="co-activates",
                    amount=0.09 / float(distance),
                )
                self._strengthen_synapse(
                    target,
                    source,
                    timestamp,
                    kind="associates",
                    amount=0.045 / float(distance),
                )

    def _wire_assembly_competition(
        self,
        assembly_id: str,
        vector: torch.Tensor,
        timestamp: float,
        *,
        exclude_ids: Optional[Sequence[str]] = None,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Form sparse lateral inhibition from measured representational overlap.

        The closest positively overlapping assembly is a local competitor for
        the newly formed assembly.  Each new assembly adds at most one pair,
        but older assemblies can accumulate arbitrarily many such pathways as
        the substrate grows, so this is sparse topology rather than a stored
        relationship-count ceiling.
        """

        excluded = {str(value) for value in (exclude_ids or [])}
        candidate_vectors = (
            self._iter_assembly_activities()
            if candidate_ids is None
            else (
                (identifier, self.assembly_vectors[identifier])
                for identifier in candidate_ids
                if identifier in self.assembly_vectors
            )
        )
        best = max((
            (self.space.similarity(vector, candidate), candidate_id)
            for candidate_id, candidate in candidate_vectors
            if candidate_id != assembly_id and candidate_id not in excluded
        ), default=None)
        if best is None:
            return
        similarity, competitor_id = best
        if similarity <= 0.0:
            return
        amount = -max(0.04, min(0.20, float(similarity) * 0.20))
        self._strengthen_synapse(
            assembly_id,
            competitor_id,
            timestamp,
            kind="competes",
            amount=amount,
        )
        self._strengthen_synapse(
            competitor_id,
            assembly_id,
            timestamp,
            kind="competes",
            amount=amount,
        )

    def _store_assembly(
        self,
        text: str,
        *,
        kind: str,
        source: str,
        source_label: str,
        retain_source_text: bool,
        importance: float,
        timestamp: float,
        child_ids: Optional[Sequence[str]] = None,
        labels_override: Optional[Sequence[str]] = None,
        vector_override: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, bool]:
        if isinstance(self.assemblies, PagedAssemblyView) and retain_source_text:
            raise ValueError("paged assembly metadata cannot retain raw source text")
        labels = (
            [
                str(value).strip().lower()
                for value in labels_override
                if str(value).strip()
            ]
            if labels_override is not None
            else self.extract_concepts(text)
        )
        if not labels:
            labels = ["empty-experience"]
        packed_vector_bytes = (self.space.dimensions + 3) // 4 + 8
        estimated = (
            len(labels) * (packed_vector_bytes + 640)
            + len(labels) * 20 * 320
            + len(child_ids or []) * 320
            + packed_vector_bytes
            + self.space.dimensions * 4  # one transient decoded work row
        )
        self._check_growth(estimated)
        neuron_ids = [
            self._ensure_neuron(label, timestamp, "semantic") for label in labels
        ]
        if vector_override is None:
            vector = self.vector_for_labels(labels)
        else:
            vector = vector_override.detach().cpu().float().reshape(-1)
            if vector.numel() != self.space.dimensions:
                raise ValueError("assembly vector dimensions do not match substrate")
            if not bool(torch.isfinite(vector).all()):
                raise ValueError("assembly vector must contain only finite values")
            norm = vector.norm().clamp_min(1e-8)
            vector = vector / norm
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.assembly_by_fingerprint.get(fingerprint)
        created = existing is None
        if existing is None:
            assembly_id = hashlib.sha256(
                ("assembly:" + fingerprint).encode("ascii")
            ).hexdigest()[:24]
            record: Dict[str, Any] = {
                "id": assembly_id,
                "assembly_neuron_id": assembly_id,
                "fingerprint": fingerprint,
                "neuron_ids": neuron_ids,
                "concept_ids": neuron_ids,
                "child_assembly_ids": list(child_ids or []),
                "kind": kind,
                "source": source,
                "confidence": 0.5,
                "importance": max(0.0, min(float(importance), 1.0)),
                "rehearsals": 1,
                "created_at": timestamp,
                "last_recalled_at": None,
                "source_label": source_label,
            }
            if retain_source_text:
                record["source_text"] = text
            self.assemblies.append(record)
            self.assembly_vectors[assembly_id] = vector
            self.neurons[assembly_id] = {
                "id": assembly_id,
                "neuron_id": assembly_id,
                "label": source_label or kind,
                "region": "assembly",
                "activation": 0.3,
                "importance": record["importance"],
                "uncertainty": 0.5,
                "exposures": 1,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            self.mark_attention_neuron(assembly_id)
            self.growth_events += 1
            for member in neuron_ids:
                self._strengthen_synapse(
                    assembly_id,
                    member,
                    timestamp,
                    kind="contains",
                    amount=0.5,
                )
                self._strengthen_synapse(
                    member,
                    assembly_id,
                    timestamp,
                    kind="participates",
                    amount=0.35,
                )
            for child_id in child_ids or []:
                self._strengthen_synapse(
                    assembly_id,
                    child_id,
                    timestamp,
                    kind="composes",
                    amount=0.55,
                )
            self._wire_assembly_competition(
                assembly_id,
                vector,
                timestamp,
                exclude_ids=child_ids,
            )
        else:
            assembly_id = str(existing["id"])

            def reinforce_metadata(record: Dict[str, Any]) -> None:
                record["rehearsals"] += 1
                record["last_recalled_at"] = timestamp
                record["importance"] = min(
                    1.0, float(record["importance"]) + 0.03
                )

            record = self._edit_assembly_by_id(assembly_id, reinforce_metadata)
            node = self.neurons.get(assembly_id)
            if node is not None:
                def reinforce_node(current: Dict[str, Any]) -> None:
                    current["activation"] = min(
                        1.0, self.effective_activation(current) * 0.7 + 0.3
                    )
                    current["exposures"] += 1
                    current["last_activated_at"] = timestamp

                self.edit_neuron_by_id(assembly_id, reinforce_node)
                self.mark_attention_neuron(assembly_id)
            self.mark_attention_assembly(assembly_id)
            # Re-exposure must change the same pathways used during recall,
            # not only a rehearsal counter or transient node activation.
            # Salient repeated experiences therefore reinforce their existing
            # ternary assembly memberships and composition links in place.
            reinforcement = 0.04 + 0.10 * max(
                0.0, min(1.0, float(importance))
            )
            for member in neuron_ids:
                self._strengthen_synapse(
                    assembly_id,
                    member,
                    timestamp,
                    kind="contains",
                    amount=reinforcement,
                )
                self._strengthen_synapse(
                    member,
                    assembly_id,
                    timestamp,
                    kind="participates",
                    amount=reinforcement * 0.7,
                )
            for child_id in child_ids or []:
                self._strengthen_synapse(
                    assembly_id,
                    child_id,
                    timestamp,
                    kind="composes",
                    amount=reinforcement,
                )
        self._wire_local_structure(neuron_ids, timestamp)
        self._adapt_coactive_vectors(neuron_ids, importance)
        assembly_target = (
            vector if vector_override is not None
            else self._continuous_assembly_activity(neuron_ids)
        )
        # Keep a newly perceived sensory vector exact on first admission;
        # subsequent exposures can adapt it just like any other assembly.
        if vector_override is None or not created:
            self._adapt_assembly_vector(
                assembly_id, assembly_target, 0.08 if created else 0.12
            )
        return record, self.assembly_vectors[assembly_id], created

    def learn_vector(
        self,
        vector: torch.Tensor,
        *,
        fingerprint: str,
        kind: str,
        source: str,
        source_label: str = "",
        importance: float = 0.5,
        child_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Admit a non-text sensory vector into the same neural substrate.

        Image, audio, and video encoders produce continuous internal vectors;
        serializing those vectors into pretend text would throw away their
        content.  This path creates a distributed sensory assembly directly,
        while ordinary source-label atoms provide sparse semantic links for
        later cross-modal spreading activation.  ``fingerprint`` is a content
        digest/window identity, never raw source bytes.
        """

        timestamp = _now()
        clean_kind = str(kind).strip().lower() or "sensory"
        clean_fingerprint = str(fingerprint).strip()
        if not clean_fingerprint:
            raise ValueError("sensory assembly requires a content fingerprint")
        labels = ["%s-perception" % clean_kind]
        labels.extend(self.extract_concepts(source_label))
        # Preserve order while avoiding duplicate semantic neurons.
        labels = list(dict.fromkeys(labels))
        normalized = vector.detach().cpu().float().reshape(-1)
        if normalized.numel() != self.space.dimensions:
            raise ValueError("sensory vector dimensions do not match substrate")
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("sensory vector must contain only finite values")
        normalized = F.normalize(normalized.reshape(1, -1), dim=-1)[0]
        nearest = max((self.space.similarity(normalized, candidate)
                       for _identifier, candidate in self._iter_assembly_activities()), default=-1.0)
        record, stored, created = self._store_assembly(
            "sensory:%s:%s" % (clean_kind, clean_fingerprint),
            kind=clean_kind,
            source=source,
            source_label=source_label or (clean_kind + " perception"),
            retain_source_text=False,
            importance=importance,
            timestamp=timestamp,
            child_ids=child_ids,
            labels_override=labels,
            vector_override=normalized,
        )
        return {
            "idea_id": record["id"],
            "assembly_id": record["id"],
            "vector": stored,
            "novelty": max(0.0, min(1.0, 1.0 - max(0.0, nearest))),
            "concept_ids": list(record["neuron_ids"]),
            "neuron_ids": list(record["neuron_ids"]),
            "labels": [
                self.neurons[item]["label"]
                for item in record["neuron_ids"]
                if item in self.neurons
            ],
            "assemblies_created": int(created),
            "sensory": True,
        }

    def _statistical_field_candidate(
        self,
        neuron_ids: Sequence[str],
        vector: torch.Tensor,
        source: str,
        kind: str,
    ) -> Tuple[Optional[Mapping[str, Any]], Optional[set[str]]]:
        """Route by shared neural atoms/relations, independent of source.

        The pair index normally yields a few fields. Rare atom postings are a
        fallback for paraphrases that rearrange every local pair. This is an
        approximate routing index, not a memory lookup or answer decoder;
        field vectors and live synapses remain the learned representation.
        """

        if isinstance(self.assemblies, PagedAssemblyView):
            # Keep only two winners while visiting bounded metadata pages.
            # Pair overlap has the old fast-path precedence; the fallback
            # considers every shared atom without keeping a corpus-sized
            # postings dictionary or candidate-ID set in RAM.
            atoms = set(neuron_ids)
            query_pairs = set(self._statistical_pair_keys(neuron_ids))
            best_pair: Optional[Mapping[str, Any]] = None
            best_any: Optional[Mapping[str, Any]] = None
            pair_rank: Tuple[Any, ...] = (-1.0, -1, -1, -1, "")
            any_rank: Tuple[Any, ...] = (-1.0, -1, -1, -1, "")
            for record in self.assemblies:
                if not record.get("compressed_field"):
                    continue
                field_id = str(record.get("id", ""))
                anchors = record.get("statistical_anchor_ids", ())
                if not field_id or not isinstance(anchors, (list, tuple)):
                    continue
                shared = len(atoms.intersection(anchors))
                if not shared:
                    continue
                field_vector = self.assembly_vectors.get(field_id)
                if field_vector is None:
                    continue
                dice = 2.0 * shared / float(len(atoms) + len(anchors))
                similarity = max(0.0, self.space.similarity(vector, field_vector))
                rank = (
                    0.8 * dice + 0.2 * similarity,
                    shared,
                    int(
                        kind == record.get("kind")
                        or kind in record.get("kind_provenance", {})
                    ),
                    int(
                        source == record.get("source")
                        or source in record.get("source_provenance", {})
                    ),
                    field_id,
                )
                if rank > any_rank:
                    best_any, any_rank = record, rank
                if query_pairs.intersection(self._statistical_pair_keys(anchors)):
                    if rank > pair_rank:
                        best_pair, pair_rank = record, rank
            if pair_rank[0] >= 0.52:
                return best_pair, None
            return (best_any if any_rank[0] >= 0.52 else None), None

        self._refresh_statistical_index()
        indexed = self.assembly_by_id
        atoms = set(neuron_ids)
        candidates: set[str] = set()
        for key in self._statistical_pair_keys(neuron_ids):
            candidates.update(self._statistical_pair_index.get(key, ()))

        def strongest(ids: Iterable[str]) -> Tuple[Optional[Dict[str, Any]], float]:
            best: Optional[Dict[str, Any]] = None
            best_rank = (-1.0, -1, -1, -1, "")
            for field_id in ids:
                record = indexed.get(field_id)
                field_vector = self.assembly_vectors.get(field_id)
                if record is None or field_vector is None:
                    continue
                anchors = set(record.get("statistical_anchor_ids", []))
                shared = len(atoms.intersection(anchors))
                if not shared:
                    continue
                dice = 2.0 * shared / float(len(atoms) + len(anchors))
                similarity = max(
                    0.0, self.space.similarity(vector, field_vector)
                )
                score = 0.8 * dice + 0.2 * similarity
                # Provenance is a soft tie-break, never a field-identity key
                # or a barrier to cross-source reinforcement.
                rank = (
                    score,
                    shared,
                    int(
                        kind == record.get("kind")
                        or kind in record.get("kind_provenance", {})
                    ),
                    int(
                        source == record.get("source")
                        or source in record.get("source_provenance", {})
                    ),
                    field_id,
                )
                if rank > best_rank:
                    best = record
                    best_rank = rank
            return best, best_rank[0]

        best, score = strongest(candidates)
        if score < 0.52:
            # A small adaptive number of the rarest *nonempty* atom postings
            # avoids a full field scan on ordinary novel rows. It does not
            # bound the number of learned assemblies or the record traversal.
            postings = sorted(
                (
                    (len(values), atom, values)
                    for atom in atoms
                    if (values := self._statistical_atom_index.get(atom))
                ),
                key=lambda item: (item[0], item[1]),
            )
            probe_count = max(1, int(math.sqrt(len(atoms))))
            for _size, _atom, values in postings[:probe_count]:
                candidates.update(values)
            best, score = strongest(candidates)
        return (best if score >= 0.52 else None), candidates

    def learn_statistical(
        self,
        text: str,
        kind: str = "knowledge",
        source: str = "document",
        source_label: str = "",
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        """Adapt semantic field assemblies without one assembly per row.

        Shared atoms/relations route related records to a learned field even
        across sources. Distinct experiences can form new fields under the
        host growth reserve. Every admitted row still updates local ternary
        pathways and continuous neural vectors; raw source text is not kept.
        """

        timestamp = _now()
        clean_kind = str(kind).strip().lower() or "knowledge"
        clean_source = str(source).strip().lower() or "experience"
        labels = self.extract_atomic_concepts(text)
        if not labels:
            labels = ["empty-experience"]

        semantic_ids = [
            hashlib.sha256(
                ("semantic:%s" % label).encode("utf-8")
            ).hexdigest()[:24]
            for label in labels
        ]
        # This read uses existing adapted neuron vectors where available and
        # deterministic symbols for novel atoms. It performs no mutation.
        current_vector = self.vector_for_labels(labels)
        record, candidate_ids = self._statistical_field_candidate(
            semantic_ids, current_vector, clean_source, clean_kind
        )
        if record is None:
            field_fingerprint = hashlib.sha256(
                (
                    "statistical-field-v2:"
                    + ":".join(sorted(semantic_ids))
                ).encode("ascii")
            ).hexdigest()
            field_id = hashlib.sha256(
                ("assembly:" + field_fingerprint).encode("ascii")
            ).hexdigest()[:24]
            existing = self.assembly_by_id.get(field_id)
            if existing is not None and existing.get("compressed_field"):
                record = existing
        if record is not None:
            field_id = str(record["id"])
            field_fingerprint = str(record["fingerprint"])

        new_neurons = sum(int(item not in self.neurons) for item in semantic_ids)
        # This estimate is deliberately conservative and is checked before any
        # mutation. It covers new atoms, local/membership pathways, an indexed
        # seed field when novel, and newly observed source provenance.
        packed_vector_bytes = (self.space.dimensions + 3) // 4 + 8
        estimated = (
            new_neurons * (packed_vector_bytes + 640)
            + len(labels) * 18 * 320
            + (
                packed_vector_bytes
                + 1024
                + len(labels) * 320
                + len(clean_source) * 2
                if record is None
                else 0
            )
            + (
                len(clean_source) * 2 + 128
                if record is not None
                and clean_source not in record.get("source_provenance", [])
                else 0
            )
            + (
                len(clean_kind) * 2 + 128
                if record is not None
                and clean_kind not in record.get("kind_provenance", [])
                else 0
            )
            + self.space.dimensions * 4  # one transient decoded work row
        )
        self._check_growth(max(1, estimated))

        neuron_ids = [
            self._ensure_neuron(label, timestamp, "semantic")
            for label in labels
        ]
        prior_vector = self.assembly_vectors.get(field_id)
        created = record is None
        if record is None:
            record = {
                "id": field_id,
                "assembly_neuron_id": field_id,
                "fingerprint": field_fingerprint,
                # Membership is authoritative in sparse synapses. Keeping a
                # duplicated ever-growing label list here would defeat the
                # resource-safe representation and slow every inspection.
                "neuron_ids": [],
                "concept_ids": [],
                "child_assembly_ids": [],
                # Derived routing metadata from the seed's semantic neuron
                # identities. The learned synapses/vector, not this list, are
                # authoritative for recall and plasticity.
                "statistical_anchor_ids": list(neuron_ids),
                "kind": clean_kind,
                "source": clean_source,
                "kind_provenance": {clean_kind: 1},
                "source_provenance": {clean_source: 1},
                "confidence": 0.5,
                "importance": max(0.0, min(float(importance), 1.0)),
                "rehearsals": 1,
                "statistical_experiences": 1,
                "compressed_field": True,
                "created_at": timestamp,
                "last_recalled_at": None,
                "source_label": "%s experience field" % clean_kind,
            }
            self.assemblies.append(record)
            self.assembly_vectors[field_id] = current_vector
            self.neurons[field_id] = {
                "id": field_id,
                "neuron_id": field_id,
                "label": "%s experience field" % clean_kind,
                "region": "assembly",
                "activation": 0.3,
                "importance": record["importance"],
                "uncertainty": 0.5,
                "exposures": 1,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            self.mark_attention_neuron(field_id)
            self.growth_events += 1
            self._wire_assembly_competition(
                field_id,
                current_vector,
                timestamp,
                candidate_ids=candidate_ids,
            )
        else:
            def revise_field(record: Dict[str, Any]) -> None:
                sources = record.setdefault(
                    "source_provenance", {str(record.get("source", "")): 1}
                )
                sources[clean_source] = int(sources.get(clean_source, 0)) + 1
                kinds = record.setdefault(
                    "kind_provenance", {str(record.get("kind", "")): 1}
                )
                kinds[clean_kind] = int(kinds.get(clean_kind, 0)) + 1
                record["rehearsals"] = int(record.get("rehearsals", 0)) + 1
                record["statistical_experiences"] = int(
                    record.get("statistical_experiences", 0)
                ) + 1
                record["last_recalled_at"] = timestamp
                record["importance"] = min(
                    1.0,
                    float(record.get("importance", 0.0))
                    + 0.01 / (1.0 + math.log1p(record["rehearsals"])),
                )

            record = self._edit_assembly_by_id(field_id, revise_field)
            def revise_node(current: Dict[str, Any]) -> None:
                current["activation"] = min(
                    1.0, self.effective_activation(current) * 0.68 + 0.32
                )
                current["importance"] = record["importance"]
                current["exposures"] = int(current.get("exposures", 0)) + 1
                current["last_activated_at"] = timestamp

            self.edit_neuron_by_id(field_id, revise_node)
            self.mark_attention_neuron(field_id)
            self.mark_attention_assembly(field_id)

        for member in neuron_ids:
            self._strengthen_synapse(
                field_id,
                member,
                timestamp,
                kind="statistically-contains",
                amount=0.12,
            )
            self._strengthen_synapse(
                member,
                field_id,
                timestamp,
                kind="statistically-participates",
                amount=0.08,
            )
        self._wire_local_structure(neuron_ids, timestamp)
        self._adapt_coactive_vectors(neuron_ids, importance)
        # This whole-row activity is an ephemeral afterimage for working
        # focus and slow cortical learning. It is not a stored text key or an
        # answer record. The durable field prototype adapts toward it, while
        # sparse ordered synapses retain immediate fast plasticity.
        episode_activity = self._continuous_assembly_activity(neuron_ids)
        self._adapt_assembly_vector(
            field_id,
            episode_activity,
            0.08 if created else 0.12,
        )

        similarity = (
            self.space.similarity(prior_vector, current_vector)
            if prior_vector is not None
            else -1.0
        )
        novelty = max(
            0.0,
            min(
                1.0,
                0.70 * (new_neurons / float(max(1, len(labels))))
                + 0.30 * (1.0 - max(0.0, similarity)),
            ),
        )
        return {
            "idea_id": field_id,
            "assembly_id": field_id,
            "vector": episode_activity,
            "novelty": novelty,
            "concept_ids": neuron_ids,
            "neuron_ids": neuron_ids,
            "labels": labels,
            "assemblies_created": int(created),
            "statistical_update": True,
            "statistical_experiences": int(
                record.get("statistical_experiences", 1)
            ),
            "raw_source_stored": False,
        }

    def learn(
        self,
        text: str,
        kind: str = "knowledge",
        source: str = "conversation",
        source_label: str = "",
        retain_source_text: bool = False,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        timestamp = _now()
        segments = self._segments(text)
        if not segments:
            segments = ["empty-experience"]
        child_ids: List[str] = []
        created_count = 0
        nearest = -1.0
        for segment in segments:
            record, vector, created = self._store_assembly(
                segment,
                kind=kind,
                source=source,
                source_label=source_label,
                retain_source_text=retain_source_text,
                importance=importance,
                timestamp=timestamp,
            )
            child_ids.append(str(record["id"]))
            created_count += int(created)
            nearest = max(
                nearest,
                max(
                    (
                        self.space.similarity(vector, candidate)
                        for assembly_id, candidate in self._iter_assembly_activities()
                        if assembly_id != record["id"]
                    ),
                    default=-1.0,
                ),
            )

        if len(segments) > 1:
            parent, parent_vector, parent_created = self._store_assembly(
                text,
                kind=kind,
                source=source,
                source_label=source_label,
                retain_source_text=retain_source_text,
                importance=min(1.0, importance + 0.08),
                timestamp=timestamp,
                child_ids=child_ids,
            )
            primary = parent
            primary_vector = parent_vector
            created_count += int(parent_created)
        else:
            primary = self.assembly_by_id[child_ids[0]]
            primary_vector = self.assembly_vectors[str(primary["id"])]

        novelty = max(0.0, min(1.0, 1.0 - max(0.0, nearest)))
        return {
            "idea_id": primary["id"],
            "assembly_id": primary["id"],
            "vector": primary_vector,
            "novelty": novelty,
            "concept_ids": list(primary["neuron_ids"]),
            "neuron_ids": list(primary["neuron_ids"]),
            "labels": [
                self.neurons[item]["label"]
                for item in primary["neuron_ids"]
                if item in self.neurons
            ],
            "assemblies_created": created_count,
        }

    def _effective_recall_graph(
        self,
    ) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, int], int, int]:
        """Reuse exact forward topology until a node or live edge changes.

        Untracked externally replaced mappings are scanned on every call,
        preserving correctness for callers that assign ordinary dictionaries.
        Cache admission is a memory budget, never a limit on recalled edges.
        """

        revision: Optional[Tuple[int, int, int, int]] = None
        if isinstance(self.neurons, _RevisionedNodes) and isinstance(
            self.synapses, (_RevisionedSynapses, LazyPersistedSynapses)
        ):
            revision = (
                id(self.neurons),
                self.neurons.graph_revision,
                id(self.synapses),
                self.synapses.graph_revision,
            )
        cached = self._recall_graph_cache
        if revision is not None and cached is not None and cached[0] == revision:
            return cached[1], cached[2], cached[3], cached[4]
        # Release a stale graph before allocating its replacement.
        self._recall_graph_cache = None
        cached = None

        adjacency: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        incoming: Dict[str, int] = defaultdict(int)
        inhibitory_edges = 0
        eligible_edges = 0
        persisted_edges = (
            self.synapses.iter_effective_edges()
            if isinstance(self.synapses, LazyPersistedSynapses)
            else (
                (
                    str(synapse["source_id"]),
                    str(synapse["target_id"]),
                    self.exact_effective_weight(
                        synapse.get("effective_weight", 0)
                    ),
                )
                for synapse in self.synapses.values()
            )
        )
        for source, target, effective in persisted_edges:
            if effective == 0:
                continue
            if source not in self.neurons or target not in self.neurons:
                continue
            eligible_edges += 1
            inhibitory_edges += int(effective < 0)
            adjacency[source].append((target, effective))
            incoming[target] += 1

        cacheable = revision is not None
        if cacheable:
            # Bound duplicate Python adjacency memory, particularly for the
            # shard-backed graph. Cache only the measured live topology: a
            # large substrate may have millions of exact-zero learned edges.
            # If the graph is too large, the exact scan remains active.
            estimated_bytes = (
                eligible_edges * _RECALL_GRAPH_EDGE_ESTIMATE
                + (len(adjacency) + len(incoming))
                * _RECALL_GRAPH_NODE_ESTIMATE
            )
            cacheable = estimated_bytes <= _RECALL_GRAPH_CACHE_BYTES
        self._recall_graph_cache = (
            (revision, adjacency, incoming, eligible_edges, inhibitory_edges)
            if cacheable and revision is not None
            else None
        )
        return adjacency, incoming, eligible_edges, inhibitory_edges

    def _iter_assembly_activities(self):
        if isinstance(self.assemblies, PagedAssemblyView) and isinstance(self.neuron_vectors, PagedPackedVectors):
            yield from PagedAssemblyVectorProvider(self.assemblies.index, self.neuron_vectors).iter_activities(
                page_size=self.assemblies.page_size,
            )
        else:
            yield from self.assembly_vectors.items()

    def recall_vector(
        self,
        cue: torch.Tensor,
        *,
        workspace_slots: Optional[int] = None,
        record_activity: bool = True,
    ) -> Tuple[torch.Tensor, Sequence[Dict[str, Any]]]:
        """Recall by similarity followed by recurrent spreading activation.

        All assemblies above the adaptive activation floor participate in the
        bundled neural signal. Working-memory size affects salience, not
        permanent addressability, and this API exposes no count-truncation
        parameter.
        """

        if not self.assembly_vectors:
            return cue, []
        if (
            isinstance(self.assemblies, PagedAssemblyView)
            and isinstance(self.neuron_vectors, PagedPackedVectors)
        ):
            return self._recall_vector_paged(cue, workspace_slots=workspace_slots, record_activity=record_activity)
        # The metadata and packed-neuron stores are separate SQLite files in
        # paged mode. Score only assembly IDs through a paired generation
        # provider: an all-neuron scan would mistake ordinary neurons for
        # ideas, and the metadata index contains no duplicate vector rows.
        paged_provider: Optional[PagedAssemblyVectorProvider] = None
        paged_snapshot: Optional[str] = None
        if (
            isinstance(self.assemblies, PagedAssemblyView)
            and isinstance(self.neuron_vectors, PagedPackedVectors)
        ):
            paged_provider = PagedAssemblyVectorProvider(
                self.assemblies.index, self.neuron_vectors
            )
            prepared = prepare_exact_paged_similarity(
                paged_provider, cue,
                page_size=min(self.assemblies.page_size, 4096),
                workspace_slots=workspace_slots,
            )
            if prepared.summary.positive_count == 0:
                paged_provider.assert_unchanged(prepared.summary.snapshot_id)
                return cue, []
            slots = prepared.summary.workspace_slots
            adaptive_floor = prepared.summary.adaptive_floor
            paged_snapshot = prepared.summary.snapshot_id
            # Only active seeds occupy memory. No top-k or fixed assembly cap
            # is imposed; a truly enormous active set still needs a separate
            # paged recurrent overlay before whole-recall memory is bounded.
            seeds = {}
            for match in prepared.iter_matches():
                if len(seeds) % 128 == 0:
                    # Admission is in bounded chunks, not a neural count
                    # ceiling. A denied host reserve leaves recall unchanged.
                    self._check_growth(128 * 256)
                seeds[match.assembly_id] = float(match.score)
            paged_provider.assert_unchanged(paged_snapshot)
        else:
            # In-memory packed vectors retain the original exact two-pass
            # similarity. Avoid a full sorted (score,id,decoded-vector) list.
            best = 0.0
            positive_count = 0
            for vector in self.assembly_vectors.values():
                score = self.space.similarity(cue, vector)
                if score > 0:
                    positive_count += 1
                    best = max(best, score)
            if positive_count == 0:
                return cue, []
            slots = max(1, int(workspace_slots or max(8, math.sqrt(positive_count))))
            adaptive_floor = max(0.01, best / (2.0 + math.log2(slots + 1.0)))
            seeds = {
                assembly_id: float(score)
                for assembly_id, vector in self.assembly_vectors.items()
                if (score := self.space.similarity(cue, vector)) > 0
                and score >= adaptive_floor
            }
        if paged_provider is not None and self._recall_graph_cache is None:
            # The current recurrent graph still materializes adjacency. Do
            # not walk a potentially huge lazy edge store into host OOM. A
            # future paged frontier can replace this conservative admission.
            self._check_growth(max(1, len(self.synapses)) * 256)
        adjacency, incoming, eligible_edges, inhibitory_edges = (
            self._effective_recall_graph()
        )

        # This synchronous recurrent system is a contraction: each target's
        # signed drive is normalized by its incoming degree and damped by 0.52.
        # It therefore settles without a fixed hop ceiling. Crucially, the
        # contribution of each live edge is exactly source_activity * {-1,+1};
        # no separate floating shadow scales the forward signal.
        activation = dict(seeds)
        admitted = set(seeds)
        propagation_rounds = 0
        inhibitory_signals = 0
        convergence_delta = 0.0
        while activation:
            workspace_pressure = max(
                1.0, len(activation) / max(1.0, float(slots))
            )
            pressure_floor = max(
                1e-5,
                adaptive_floor
                * (
                    0.02
                    + 0.03 * math.log2(workspace_pressure + 1.0)
                ),
            )
            drives: Dict[str, float] = defaultdict(float)
            round_inhibitory = 0
            for source, source_activation in activation.items():
                if abs(source_activation) <= 1e-12:
                    continue
                for target, weight in adjacency.get(source, []):
                    contribution = source_activation * float(weight)
                    drives[target] += contribution
                    round_inhibitory += int(contribution < 0.0)

            candidates = admitted.union(seeds).union(drives)
            settled: Dict[str, float] = {}
            for target in candidates:
                recurrent = (
                    0.52
                    * drives.get(target, 0.0)
                    / float(max(1, incoming.get(target, 0)))
                )
                value = max(
                    -1.0,
                    min(1.0, seeds.get(target, 0.0) + recurrent),
                )
                if target in admitted or abs(value) >= pressure_floor:
                    admitted.add(target)
                    settled[target] = value

            convergence_delta = max(
                (
                    abs(settled.get(key, 0.0) - activation.get(key, 0.0))
                    for key in set(settled).union(activation)
                ),
                default=0.0,
            )
            inhibitory_signals = round_inhibitory
            activation = settled
            propagation_rounds += 1
            if convergence_delta <= max(1e-7, pressure_floor * 0.001):
                break

        by_id = self.assembly_by_id
        if paged_provider is not None:
            self._check_growth(max(1, len(activation)) * 160)
        active = sorted(activation.items(), key=lambda item: item[1], reverse=True)
        # Stored packed rows decode as unit-length transient activations.
        # Normalize the current cue only for this readout so a text cue with
        # more dimensions cannot drown out equally salient recalled activity.
        signal_sum = F.normalize(cue.float(), dim=0).clone()
        signal_weight = 1.0
        recalled = []
        for assembly_id, score in active:
            vector = self.assembly_vectors.get(assembly_id)
            if vector is None or score <= 0:
                continue
            weight = max(0.01, float(score))
            signal_sum.add_(vector.float(), alpha=weight)
            signal_weight += weight
            assembly = by_id.get(assembly_id)
            if assembly is None:
                continue
            recalled.append(
                {
                    "idea_id": assembly_id,
                    "assembly_id": assembly_id,
                    "score": float(score),
                    "concept_ids": list(assembly["neuron_ids"]),
                    "neuron_ids": list(assembly["neuron_ids"]),
                }
            )
        if paged_provider is not None and paged_snapshot is not None:
            # The readout above decoded only selected assembly rows. Detect a
            # concurrent structural/vector edit before publishing activity;
            # record_recall_activity itself then advances metadata revision.
            paged_provider.assert_unchanged(paged_snapshot)
        if record_activity:
            self.record_recall_activity(recalled)
        # Recalled firing is a continuous decoder-conditioning signal, not a
        # newly stored bipolar symbol. A sign bundle here silently discarded
        # every sub-unit recall contribution when the cue had weight 1: one
        # remembered assembly could fire through real synapses while the
        # decoder still received the untouched cue. Preserve its measured
        # contribution with a bounded activity-weighted mean. The underlying
        # forward synapses remain exactly ternary; no shadow-weight magnitude
        # or stored source text participates in this readout.
        signal = signal_sum / max(signal_weight, 1e-8)
        # Exposed as inspection metadata only; it is derived from this recall
        # operation and is never an authoritative memory record.
        self._last_recall_rounds = propagation_rounds
        suppressed = sum(
            1
            for assembly_id, seed in seeds.items()
            if seed >= adaptive_floor
            and activation.get(assembly_id, 0.0) < adaptive_floor
        )
        self._last_recall_audit = {
            "rule": "effective-ternary-recurrent-settling",
            "exactTernaryContribution": True,
            "latentMagnitudeUsed": False,
            "eligibleEdges": eligible_edges,
            "inhibitoryEdges": inhibitory_edges,
            "inhibitorySignals": inhibitory_signals,
            "suppressedAssemblies": suppressed,
            "settledRounds": propagation_rounds,
            "convergenceDelta": convergence_delta,
            "damping": 0.52,
            "fanInNormalization": True,
            "activeNeuralNodes": len(activation),
            "signalRule": "continuous-activity-weighted-mean",
            "activationByAssembly": {
                key: float(value)
                for key, value in sorted(activation.items())
                if key in self.assembly_vectors
            },
        }
        return signal, recalled

    def _recall_vector_paged(
        self, cue: torch.Tensor, *, workspace_slots: Optional[int], record_activity: bool,
    ):
        from .paged_recurrent_spreading import PagedRecurrentState
        from .paged_recall_graph import PagedRecallGraph
        provider = PagedAssemblyVectorProvider(self.assemblies.index, self.neuron_vectors)
        prepared = prepare_exact_paged_similarity(
            provider, cue, page_size=self.assemblies.page_size, workspace_slots=workspace_slots,
        )
        if prepared.summary.positive_count == 0:
            provider.assert_unchanged(prepared.summary.snapshot_id)
            return cue, []
        initial_graph_revision = getattr(self.synapses, "graph_revision", None)
        initial_neuron_revision = getattr(self.neurons, "graph_revision", None)
        source_reference = weakref.ref(self)
        def reserve(size: int, operation: str) -> bool:
            # Corpus/frontier bytes live on disk, not in an equally large
            # resident mirror. Charge the actual SQLite write envelope and
            # only the bounded per-connection RAM window.
            source = source_reference()
            if source is None:
                raise RuntimeError("paged recurrent execution source is closed")
            index = source.assemblies.index
            index._reserve_disk(size, operation)
            index._reserve_memory(320 * 1024, operation)
            if getattr(index, "_disk_reserve", None) is None:
                source._check_growth(min(size, 320 * 1024))
            return True
        state = PagedRecurrentState(self.assemblies.index.path.parent / "recall", reserve=reserve)
        try:
            graph_stamp = (id(self.synapses), id(self.neurons))
            graph_cached = getattr(self, "_paged_recall_graph", None)
            if (
                initial_graph_revision is None or initial_neuron_revision is None
                or graph_cached is None or graph_cached[0] != graph_stamp
                or not graph_cached[1].refresh(self.synapses, self.neurons)
            ):
                graph = PagedRecallGraph(self.assemblies.index.path.parent / "recall", reserve=reserve)
                try:
                    edges = (
                        self.synapses.iter_effective_edge_records() if isinstance(self.synapses, LazyPersistedSynapses)
                        else ((str(identifier), str(value["source_id"]), str(value["target_id"]),
                               self.exact_effective_weight(value.get("effective_weight", 0)))
                              for identifier, value in self.synapses.items())
                    )
                    graph.load(edges, node_exists=lambda identifier: identifier in self.neurons,
                               synapse_revision=initial_graph_revision, neuron_revision=initial_neuron_revision)
                except BaseException:
                    graph.close()
                    raise
                if isinstance(self.synapses, (_RevisionedSynapses, LazyPersistedSynapses)):
                    self.synapses._recall_graph_observer = graph.queue_change
                if isinstance(self.neurons, (_RevisionedNodes, PagedNeuronMetadata)):
                    self.neurons._recall_graph_observer = graph.queue_membership_change
                self._paged_recall_graph = (graph_stamp, graph)
            else:
                graph = graph_cached[1]
            state.use_graph(graph)
            state.load_seeds((match.assembly_id, float(match.score)) for match in prepared.iter_matches())
            provider.assert_unchanged(prepared.summary.snapshot_id)
            state.settle(slots=prepared.summary.workspace_slots, adaptive_floor=prepared.summary.adaptive_floor)
            signal_sum = F.normalize(cue.float(), dim=0).clone()
            signal_weight = 1.0
            for assembly_id, score in state.iter_activation(positive_only=True):
                vector = self.assembly_vectors.get(assembly_id)
                if vector is None:
                    continue
                weight = max(0.01, score)
                signal_sum.add_(vector.float(), alpha=weight)
                signal_weight += weight
                assembly = self.assembly_by_id.get(assembly_id)
                if assembly is not None:
                    state.append_recalled({
                        "idea_id": assembly_id, "assembly_id": assembly_id, "score": score,
                        "concept_ids": list(assembly["neuron_ids"]), "neuron_ids": list(assembly["neuron_ids"]),
                    })
            provider.assert_unchanged(prepared.summary.snapshot_id)
            if (
                getattr(self.synapses, "graph_revision", None) != initial_graph_revision
                or getattr(self.neurons, "graph_revision", None) != initial_neuron_revision
            ):
                raise ValueError("recurrent graph changed before readout")
            recalled = state.recalled()
            if record_activity:
                self.record_recall_activity(recalled)
            self._last_recall_paged_state = state
            self._last_recall_rounds = state.rounds
            self._last_recall_audit = {
                "rule": "effective-ternary-recurrent-settling", "exactTernaryContribution": True,
                "latentMagnitudeUsed": False, "eligibleEdges": state.eligible_edges,
                "inhibitoryEdges": state.inhibitory_edges, "inhibitorySignals": state.inhibitory_signals,
                "suppressedAssemblies": state.suppressed_seeds, "settledRounds": state.rounds,
                "convergenceDelta": state.convergence_delta, "damping": 0.52, "fanInNormalization": True,
                "activeNeuralNodes": state.active_count, "signalRule": "continuous-activity-weighted-mean",
                "activationByAssembly": {}, "activationStorage": "paged-recurrent-frontier",
                "activationCount": len(recalled), "allEligibleAssembliesIncluded": True,
            }
            return signal_sum / max(signal_weight, 1e-8), recalled
        except BaseException:
            state.close()
            raise

    def release_paged_recall_scratch(self) -> None:
        """Release disposable graph/frontier files only at runtime shutdown."""

        state = getattr(self, "_last_recall_paged_state", None)
        graph = getattr(self, "_paged_recall_graph", None)
        idle = getattr(self, "_paged_idle_selector", None)
        self._paged_idle_selector = None
        self._last_recall_paged_state = None
        self._paged_recall_graph = None
        if state is not None:
            state.close()
        if idle is not None:
            idle.close()
        if graph is not None:
            owner = graph[1]
            for source in (self.neurons, self.synapses):
                observer = getattr(source, "_recall_graph_observer", None)
                if getattr(observer, "__self__", None) is owner:
                    del source._recall_graph_observer
            owner.close()

    def record_recall_activity(
        self, recalled: Sequence[Mapping[str, Any]]
    ) -> None:
        """Commit activation metadata for a previously settled recall."""

        if not recalled:
            return
        by_id = self.assembly_by_id
        timestamp = _now()
        changed = False
        for item in recalled:
            assembly_id = str(item.get("assembly_id", item.get("idea_id", "")))
            score = float(item.get("score", 0.0))
            assembly = by_id.get(assembly_id)
            if assembly is None or score <= 0.0:
                continue
            if isinstance(self.assemblies, PagedAssemblyView):
                def mark_recalled(record: Dict[str, Any]) -> None:
                    record["last_recalled_at"] = timestamp

                self._edit_assembly_by_id(
                    assembly_id, mark_recalled
                )
            else:
                assembly["last_recalled_at"] = timestamp
            node = self.neurons.get(assembly_id)
            if node is not None:
                def mark_node(current: Dict[str, Any]) -> None:
                    current["activation"] = min(
                        1.0,
                        self.effective_activation(current) * 0.6 + min(0.4, score),
                    )
                    current["last_activated_at"] = timestamp

                self.edit_neuron_by_id(assembly_id, mark_node)
                self.mark_attention_neuron(assembly_id)
            self.mark_attention_assembly(assembly_id)
            changed = True
        if changed:
            self._mark_state_changed()

    def effective_activation(self, neuron: Mapping[str, Any]) -> float:
        identifier = str(neuron.get("id", neuron.get("neuron_id", "")))
        if (
            not self.attention_legacy_raw_active
            and identifier not in self.attention_active_neuron_ids
        ):
            return 0.0
        return float(neuron.get("activation", 0.0) or 0.0)

    def effective_eligibility(self, synapse: Mapping[str, Any]) -> float:
        identifier = str(synapse.get("id", ""))
        if (
            not self.attention_legacy_raw_active
            and identifier not in self.attention_eligible_synapse_ids
        ):
            return 0.0
        return abs(float(synapse.get("eligibility", 0.0) or 0.0))

    def _signed_eligibility(self, synapse: Mapping[str, Any]) -> float:
        identifier = str(synapse.get("id", ""))
        if (
            not self.attention_legacy_raw_active
            and identifier not in self.attention_eligible_synapse_ids
        ):
            return 0.0
        return float(synapse.get("eligibility", 0.0) or 0.0)

    def attention_overlay_metadata(self) -> Dict[str, Any]:
        return {
            "format": "omni-substrate-attention-overlay",
            "formatVersion": 1,
            "epoch": int(self.attention_epoch),
            "legacyRawActive": bool(self.attention_legacy_raw_active),
            "activeNeuronIds": sorted(self.attention_active_neuron_ids),
            "recalledAssemblyIds": sorted(
                self.attention_recalled_assembly_ids
            ),
            "eligibleSynapseIds": sorted(
                self.attention_eligible_synapse_ids
            ),
        }

    def configure_attention_overlay(
        self,
        epoch: int,
        overlay: Optional[Mapping[str, Any]],
    ) -> None:
        self.attention_epoch = max(0, int(epoch))
        if overlay is None:
            self.attention_legacy_raw_active = self.attention_epoch == 0
            self.attention_active_neuron_ids = set()
            self.attention_recalled_assembly_ids = set()
            self.attention_eligible_synapse_ids = set()
            return
        if (
            overlay.get("format") != "omni-substrate-attention-overlay"
            or int(overlay.get("formatVersion", 0)) != 1
            or int(overlay.get("epoch", -1)) != self.attention_epoch
            or not isinstance(overlay.get("legacyRawActive"), bool)
        ):
            raise ValueError("substrate attention overlay is invalid")

        def identifiers(field: str, available: Mapping[str, Any]) -> set[str]:
            values = overlay.get(field, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value in available
                for value in values
            ):
                raise ValueError("substrate attention overlay ids are invalid")
            return set(values)

        assemblies = self.assembly_by_id
        self.attention_legacy_raw_active = bool(
            overlay["legacyRawActive"]
        )
        self.attention_active_neuron_ids = identifiers(
            "activeNeuronIds", self.neurons
        )
        self.attention_recalled_assembly_ids = identifiers(
            "recalledAssemblyIds", assemblies
        )
        self.attention_eligible_synapse_ids = identifiers(
            "eligibleSynapseIds", self.synapses
        )

    def clear_attention_activity(self, next_epoch: int) -> Dict[str, int]:
        """Advance an O(active-overlay) boundary without rewriting substrate."""

        if int(next_epoch) <= self.attention_epoch:
            raise ValueError("attention epoch must increase")
        activated_neurons = len(self.attention_active_neuron_ids)
        recalled_assemblies = len(self.attention_recalled_assembly_ids)
        eligibility_traces = len(self.attention_eligible_synapse_ids)
        legacy_raw_active = int(self.attention_legacy_raw_active)
        self.attention_epoch = int(next_epoch)
        self.attention_legacy_raw_active = False
        self.attention_active_neuron_ids.clear()
        self.attention_recalled_assembly_ids.clear()
        self.attention_eligible_synapse_ids.clear()
        self._last_recall_audit = {
            "rule": "effective-ternary-recurrent-settling",
            "exactTernaryContribution": True,
            "latentMagnitudeUsed": False,
            "settledRounds": 0,
            "inhibitoryEdges": 0,
            "inhibitorySignals": 0,
            "suppressedAssemblies": 0,
        }
        self._last_recall_rounds = 0
        return {
            "activatedNeurons": activated_neurons,
            "recalledAssemblies": recalled_assemblies,
            "eligibilityTraces": eligibility_traces,
            "legacyRawActive": legacy_raw_active,
        }

    def decay(
        self,
        amount: float = 0.002,
        *,
        synapses: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        """Decay activity and either all or one explicitly active edge set.

        Full synaptic forgetting remains the consolidation operation. Online
        settling passes its causally related edges so one tiny experience does
        not rewrite millions of unrelated durable synapses before replying.
        """

        amount = max(0.0, min(float(amount), 1.0))
        if isinstance(self.neurons, PagedNeuronMetadata):
            if amount > 0.0:
                # One durable epoch changes the effective state of all cold
                # neurons. Reads project it; touched-node edits materialize
                # their pending factor before changing exposures. No per-turn
                # full-table WAL sweep or silent decay omission occurs.
                self.neurons.decay(amount)
        else:
            for neuron in self.neurons.values():
                neuron["activation"] *= 1.0 - amount
                neuron["uncertainty"] = min(
                    1.0,
                    float(neuron["uncertainty"])
                    + amount / (1.0 + neuron["exposures"]),
                )
        selected_synapses = (
            self.synapses.values() if synapses is None else synapses
        )
        decayed_synapses = False
        for synapse in selected_synapses:
            decayed_synapses = True
            stability = float(synapse.get("stability", 0.0))
            weight = self.exact_effective_weight(synapse["effective_weight"])
            # Maintenance may continue weakening a cold pathway after Fresh.
            # Its stored timing trace does not make that pathway active for
            # recall or working-memory admission: those still honor the
            # attention overlay through effective_eligibility().
            timing = float(synapse.get("eligibility", 0.0) or 0.0) * (
                1.0 - amount * 0.1
            )
            if weight:
                pressure = (
                    amount
                    * (1.0 - 0.8 * min(1.0, stability))
                    / (1.0 + synapse["uses"])
                )
                timing -= weight * pressure
                if weight * timing <= -0.25:
                    timing += weight * 0.25
                    weight = 0
            synapse["eligibility"] = max(-1.0, min(1.0, timing))
            synapse["stability"] = max(0.0, stability - amount * 0.05)
            synapse["effective_weight"] = weight
        if amount > 0.0 and (self.neurons or decayed_synapses):
            self._mark_state_changed()

    def synaptic_use_count(self) -> int:
        """Return exact cumulative sparse-synapse use events."""

        if isinstance(self.synapses, LazyPersistedSynapses):
            return self.synapses.synaptic_use_count
        total = 0
        for record in self.synapses.values():
            uses = record.get("uses", 0)
            if isinstance(uses, bool) or not isinstance(uses, int) or uses < 0:
                raise ValueError("substrate synaptic uses are invalid")
            total += uses
        return total

    def _validate_packed_vector_identity(self) -> None:
        """Require one packed row per neuron and one assembly view per record."""

        if (
            getattr(self.neuron_vectors, "packed_authoritative", False) is not True
            or getattr(self.neuron_vectors, "dimensions", None) != self.space.dimensions
            or not isinstance(
                self.assembly_vectors,
                (PackedTernaryVectorView, PagedAssemblyVectorView),
            )
            or self.assembly_vectors.backing is not self.neuron_vectors
        ):
            raise ValueError("substrate vectors must share one packed authority")
        if (
            isinstance(self.assemblies, PagedAssemblyView)
            and isinstance(self.assembly_vectors, PagedAssemblyVectorView)
            and isinstance(self.neuron_vectors, PagedPackedVectors)
            and self.assemblies.index._vectors is self.neuron_vectors
            and self.assembly_vectors.index is self.assemblies.index
            and isinstance(self.persistence_manifest, dict)
        ):
            generation = self.persistence_manifest.get("activeGeneration")
            index_state = self.assemblies.index.status()
            vectors_state = self.neuron_vectors.status()
            if (
                isinstance(generation, str)
                and not index_state["dirtySinceCommit"]
                and not vectors_state["dirtySinceCommit"]
                and index_state["committedGenerationSha256"] == generation
                and vectors_state["committedGenerationSha256"] == generation
                and index_state["count"] == index_state["packedVectorRows"]
                and vectors_state["rowCount"] == len(self.neurons)
                and index_state["count"] == len(self.assemblies)
            ):
                # Rebuild verified every committed shard and every shared
                # vector alias before binding this generation. Rechecking
                # those same millions of IDs on each read/save is redundant;
                # a dirty cache falls through to the full exact scan.
                return
        if len(self.neuron_vectors) != len(self.neurons) or any(
            neuron_id not in self.neuron_vectors for neuron_id in self.neurons
        ):
            raise ValueError("substrate neuron records and packed vectors differ")
        if isinstance(self.assemblies, PagedAssemblyView):
            if isinstance(self.assembly_vectors, PagedAssemblyVectorView) and (
                self.assembly_vectors.index is not self.assemblies.index
            ):
                raise ValueError("paged assembly vector view has a different index")
            if len(self.assembly_vectors) != len(self.assemblies):
                raise ValueError("paged assemblies and vector view counts differ")
            for record in self.assemblies:
                assembly_id = str(record.get("id", ""))
                if (
                    not assembly_id
                    or assembly_id not in self.neurons
                    or assembly_id not in self.assembly_vectors
                ):
                    raise ValueError("paged assembly lacks its shared neuron row")
            return
        assembly_ids = [str(item.get("id", "")) for item in self.assemblies]
        if (
            not all(assembly_ids)
            or len(set(assembly_ids)) != len(assembly_ids)
            or set(self.assembly_vectors) != set(assembly_ids)
            or any(assembly_id not in self.neurons for assembly_id in assembly_ids)
        ):
            raise ValueError("substrate assembly vectors must alias neuron rows")

    def metadata(self, include_records: bool = True) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "schema": self.SCHEMA,
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "cardinality_limit": None,
            "authoritative_memory": "packed-ternary-neurons-assemblies-synapses",
            "growth_events": self.growth_events,
            "growth_pauses": self.growth_pauses,
            "state_revision": self.state_revision,
        }
        if self.persistence_manifest is not None:
            metadata["persistence"] = dict(self.persistence_manifest)
        if include_records:
            if isinstance(self.assemblies, PagedAssemblyView):
                raise ValueError(
                    "paged assemblies cannot use monolithic metadata export"
                )
            self._validate_packed_vector_identity()
            packed_metadata, _packed_tensors = self.neuron_vectors.export_state()
            metadata.update(
                {
                    "neurons": list(self.neurons.values()),
                    "assemblies": self.assemblies,
                    "synapses": [
                        {
                            key: value
                            for key, value in record.items()
                            if key != "latent_weight"
                        }
                        for record in self.synapses.values()
                    ],
                    "packed_vector_state": packed_metadata,
                }
            )
        return metadata

    @staticmethod
    def _canonical_json(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _bucket(kind: str, record_id: str) -> str:
        return hashlib.sha256(
            ("%s:%s" % (kind, record_id)).encode("utf-8")
        ).hexdigest()[:1]

    @staticmethod
    def _safe_store_path(root: Path, relative: str) -> Path:
        if (
            not relative
            or relative.startswith(("/", "\\"))
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("substrate shard manifest contains an unsafe path")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("substrate shard path escapes its store") from error
        return resolved

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _store_json_blob(
        self, root: Path, value: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = self._canonical_json(value)
        checksum = hashlib.sha256(payload).hexdigest()
        relative = "blobs/%s.json" % checksum
        path = self._safe_store_path(root, relative)
        if not path.exists():
            self._check_growth(len(payload) + 4096)
            atomic_write_bytes(path, payload)
        return {
            "path": relative,
            "sha256": checksum,
            "bytes": len(payload),
        }

    def _store_tensor_blob(
        self,
        root: Path,
        tensors: Dict[str, torch.Tensor],
        *,
        reusable: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state_digest = hashlib.sha256()
        for name, value in sorted(tensors.items()):
            contiguous = value.detach().cpu().contiguous()
            state_digest.update(name.encode("utf-8"))
            state_digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            state_digest.update(str(contiguous.dtype).encode("ascii"))
            state_digest.update(contiguous.numpy().tobytes())
        state_checksum = state_digest.hexdigest()
        if (
            isinstance(reusable, dict)
            and reusable.get("stateSha256") == state_checksum
        ):
            reused_path = self._safe_store_path(
                root, str(reusable.get("path", ""))
            )
            if (
                reused_path.is_file()
                and self._file_sha256(reused_path)
                == str(reusable.get("sha256", ""))
                and reused_path.stat().st_size == int(reusable.get("bytes", -1))
            ):
                return dict(reusable)
        estimated = 4096 + sum(
            int(value.numel() * value.element_size())
            for value in tensors.values()
        )
        self._check_growth(estimated)
        blobs = root / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="substrate.", suffix=".safetensors.tmp", dir=str(blobs)
        )
        os.close(descriptor)
        os.unlink(temporary_name)
        temporary = Path(temporary_name)
        try:
            atomic_save_tensors(
                temporary,
                tensors,
                metadata={
                    "format": _SUBSTRATE_STORE_FORMAT,
                    "formatVersion": str(_SUBSTRATE_STORE_VERSION),
                },
            )
            checksum = self._file_sha256(temporary)
            relative = "blobs/%s.safetensors" % checksum
            destination = self._safe_store_path(root, relative)
            size = temporary.stat().st_size
            if destination.exists():
                temporary.unlink()
            else:
                os.replace(str(temporary), str(destination))
            return {
                "path": relative,
                "sha256": checksum,
                "bytes": size,
                "stateSha256": state_checksum,
            }
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _groups(
        kind: str,
        records: Iterable[Tuple[str, Any]],
        records_per_shard: int,
    ) -> Iterable[Tuple[str, int, List[Tuple[str, Any]]]]:
        buckets: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        for record_id, record in records:
            buckets[NeuralSubstrate._bucket(kind, str(record_id))].append(
                (str(record_id), record)
            )
        for bucket in sorted(buckets):
            ordered = sorted(buckets[bucket], key=lambda item: item[0])
            for offset in range(0, len(ordered), records_per_shard):
                yield bucket, offset // records_per_shard, ordered[
                    offset : offset + records_per_shard
                ]

    def _stable_groups(
        self,
        kind: str,
        records: Iterable[Tuple[str, Any]],
        records_per_shard: int,
    ) -> Iterable[Tuple[str, int, List[Tuple[str, Any]]]]:
        """Preserve committed shard membership and place only new IDs locally."""

        if (
            not self._persistence_record_groups
            or self._persistence_records_per_shard != records_per_shard
        ):
            yield from self._groups(kind, records, records_per_shard)
            return
        groups: Dict[Tuple[str, int], List[Tuple[str, Any]]] = defaultdict(list)
        additions: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        for raw_record_id, record in records:
            record_id = str(raw_record_id)
            bucket = self._bucket(kind, record_id)
            assigned = self._persistence_record_groups.get((kind, record_id))
            if assigned is None or assigned[0] != bucket:
                additions[bucket].append((record_id, record))
                continue
            groups[(bucket, int(assigned[1]))].append((record_id, record))
        for bucket, added in sorted(additions.items()):
            parts = sorted(
                part for grouped_bucket, part in groups if grouped_bucket == bucket
            )
            part = parts[-1] if parts else 0
            for record in sorted(added, key=lambda item: item[0]):
                if len(groups[(bucket, part)]) >= records_per_shard:
                    part += 1
                groups[(bucket, part)].append(record)
        for (bucket, part), group in sorted(groups.items()):
            if len(group) > records_per_shard:
                raise ValueError("stable substrate shard exceeded its record bound")
            yield bucket, part, sorted(group, key=lambda item: item[0])

    def save_sharded(
        self,
        root: Path,
        *,
        records_per_shard: int = 512,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> Dict[str, Any]:
        """Write a deterministic content-addressed substrate generation.

        Only bounded shards are materialized. Existing content blobs are
        reused, so a growth update rewrites the affected hash bucket rather
        than the complete sparse substrate. ``disk_reserve`` is used only by
        the paged writer; the resident path keeps its existing growth guard.
        """

        if records_per_shard < 1:
            raise ValueError("records_per_shard must be positive")
        if isinstance(self.assemblies, PagedAssemblyView):
            # Keep the paged path separate from _stable_groups, whose resident
            # grouping map is proportional to the entire assembly corpus.
            # Dynamic synapses remain fail-closed until their bounded v3
            # publisher can commit the same generation and forward index.
            from .paged_substrate_writer import write_paged_substrate_generation

            return write_paged_substrate_generation(
                self, root, records_per_shard=records_per_shard,
                disk_reserve=disk_reserve,
            )
        # Validate the live forward values before creating any generation or
        # content blob. A corrupt fractional value must not be truncated to an
        # int8 ternary level or leave a partially staged substrate behind.
        lazy_synapses = (
            self.synapses
            if isinstance(self.synapses, LazyPersistedSynapses)
            else None
        )
        if lazy_synapses is not None:
            lazy_synapses.validate_dirty()
        else:
            for synapse in self.synapses.values():
                self.exact_effective_weight(synapse.get("effective_weight", 0))
        self._validate_packed_vector_identity()
        store = Path(root).resolve()
        (store / "blobs").mkdir(parents=True, exist_ok=True)
        (store / "generations").mkdir(parents=True, exist_ok=True)
        shards: List[Dict[str, Any]] = []
        forward_index_entries: List[Dict[str, Any]] = []
        forward_hot_node_ids = {
            str(record.get("id", ""))
            for record in self.assemblies
            if record.get("id")
        }
        next_record_groups: Dict[Tuple[str, str], Tuple[str, int]] = {}
        prior_shards: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        if self.persistence_manifest is not None:
            try:
                prior_path = self._safe_store_path(
                    store,
                    str(self.persistence_manifest.get("generationManifest", "")),
                )
                prior_generation = read_json(prior_path)
                prior_shards = {
                    (
                        str(item.get("kind", "")),
                        str(item.get("bucket", "")),
                        int(item.get("part", -1)),
                    ): item
                    for item in prior_generation.get("shards", [])
                    if isinstance(item, dict)
                }
            except (FileNotFoundError, OSError, ValueError, TypeError):
                # A complete new generation is still safe. Any malformed prior
                # pointer will be rejected on load rather than trusted for reuse.
                prior_shards = {}

        for kind, values in (
            ("neurons", self.neurons.items()),
            (
                "assemblies",
                (
                    (
                        str(item["id"]),
                        {**item, "__persistence_ordinal": index},
                    )
                    for index, item in enumerate(self.assemblies)
                ),
            ),
        ):
            for bucket, part, group in self._stable_groups(
                kind, values, records_per_shard
            ):
                next_record_groups.update(
                    {
                        (kind, record_id): (bucket, part)
                        for record_id, _record in group
                    }
                )
                ids = [record_id for record_id, _record in group]
                packed_metadata: Optional[Dict[str, Any]] = None
                packed_tensors: Optional[Dict[str, torch.Tensor]] = None
                if kind == "neurons":
                    packed_metadata, packed_tensors = self.neuron_vectors.export_state(
                        keys=ids
                    )
                json_blob = self._store_json_blob(
                    store,
                    {
                        "kind": kind,
                        "ids": ids,
                        "records": [record for _record_id, record in group],
                        "vectorIds": ids,
                        **(
                            {"packedVectorState": packed_metadata}
                            if kind == "neurons"
                            else {"vectorStorage": "shared-neuron-packed"}
                        ),
                    },
                )
                tensor_blob = (
                    self._store_tensor_blob(
                        store,
                        packed_tensors,
                        reusable=prior_shards.get(
                            (kind, bucket, part), {}
                        ).get("tensors"),
                    )
                    if packed_tensors is not None
                    else None
                )
                shards.append(
                    {
                        "kind": kind,
                        "bucket": bucket,
                        "part": part,
                        "count": len(group),
                        "records": json_blob,
                        "tensors": tensor_blob,
                        **(
                            {
                                "inspection": neuron_shard_inspection(
                                    [record for _record_id, record in group]
                                )
                            }
                            if kind == "neurons"
                            else {}
                        ),
                    }
                )

        region_by_id = {
            str(record.get("id", record_id)): str(
                record.get("region", "semantic")
            )
            for record_id, record in self.neurons.items()
        }
        lazy_changed_groups: Dict[
            Tuple[str, int], List[Tuple[str, Dict[str, Any]]]
        ] = {}
        if lazy_synapses is not None:
            reused, lazy_changed_groups = lazy_synapses.save_plan(
                records_per_shard
            )
            migrating_legacy_synapses = (
                lazy_synapses.store_version < _SUBSTRATE_STORE_VERSION
            )
            migration_keys = [
                (str(item["bucket"]), int(item["part"]))
                for item in reused
            ] if migrating_legacy_synapses else []
            if not migrating_legacy_synapses:
                shards.extend(reused)
            synapse_groups: Iterable[
                Tuple[str, int, List[Tuple[str, Dict[str, Any]]]]
            ] = (
                (bucket, part, group)
                for (bucket, part), group in sorted(
                    lazy_changed_groups.items()
                )
            )
            if migration_keys:
                # Rewrite only one legacy shard at a time. The old float
                # shadow is dropped on load; the v2 generation points solely
                # to ternary-weight shards. This avoids materializing a large
                # native brain in RAM just to migrate its storage format.
                changed_stream = synapse_groups

                def migrated_groups() -> Iterator[
                    Tuple[str, int, List[Tuple[str, Dict[str, Any]]]]
                ]:
                    yield from changed_stream
                    for bucket, part in migration_keys:
                        yield (
                            bucket,
                            part,
                            lazy_synapses._group_records((bucket, part)),
                        )

                synapse_groups = migrated_groups()
        else:
            synapse_groups = self._stable_groups(
                "synapses", self.synapses.items(), records_per_shard
            )
        for bucket, part, group in synapse_groups:
            if lazy_synapses is None:
                next_record_groups.update(
                    {
                        ("synapses", record_id): (bucket, part)
                        for record_id, _record in group
                    }
                )
            structures = []
            for record_id, record in group:
                structures.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in _SYNAPSE_TENSOR_FIELDS
                        and key != "latent_weight"
                    }
                )
                if structures[-1].get("id") != record_id:
                    raise ValueError("synapse mapping key does not match record id")
            json_blob = self._store_json_blob(
                store,
                {
                    "kind": "synapses",
                    "ids": [record_id for record_id, _record in group],
                    "records": structures,
                },
            )
            tensors = {
                "packed_effective_weight": torch.tensor(
                    list(_pack_ternary_levels(
                        self.exact_effective_weight(
                            record.get("effective_weight", 0)
                        )
                        for _record_id, record in group
                    )),
                    dtype=torch.uint8,
                ),
                "eligibility": torch.tensor(
                    [
                        float(record.get("eligibility", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "plasticity": torch.tensor(
                    [
                        float(record.get("plasticity", 1.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "uses": torch.tensor(
                    [
                        int(record.get("uses", 0))
                        for _record_id, record in group
                    ],
                    dtype=torch.int64,
                ),
                "stability": torch.tensor(
                    [
                        float(record.get("stability", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "last_updated_at": torch.tensor(
                    [
                        float(record.get("last_updated_at", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
            }
            reusable_synapse = prior_shards.get(
                ("synapses", bucket, part), {}
            ).get("tensors")
            tensor_blob = self._store_tensor_blob(
                store,
                tensors,
                reusable=reusable_synapse,
            )
            synapse_descriptor = {
                "kind": "synapses",
                "bucket": bucket,
                "part": part,
                "count": len(group),
                "records": json_blob,
                "tensors": tensor_blob,
                "inspection": synapse_shard_inspection(
                    [record for _record_id, record in group],
                    region_by_id,
                ),
            }
            shards.append(synapse_descriptor)
            if lazy_synapses is None:
                forward_index_entries.append(
                    _forward_index_entry(
                        synapse_descriptor,
                        group,
                        forward_hot_node_ids,
                    )
                )

        generation_body = {
            "format": _SUBSTRATE_STORE_FORMAT,
            "formatVersion": _SUBSTRATE_STORE_VERSION,
            "schema": self.SCHEMA,
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "growthEvents": self.growth_events,
            "growthPauses": self.growth_pauses,
            "stateRevision": self.state_revision,
            "recordsPerShard": int(records_per_shard),
            "counts": {
                "neurons": len(self.neurons),
                "assemblies": len(self.assemblies),
                "synapses": len(self.synapses),
            },
            "shards": sorted(
                shards,
                key=lambda item: (
                    str(item["kind"]),
                    str(item["bucket"]),
                    int(item["part"]),
                ),
            ),
        }
        content_checksum = hashlib.sha256(
            self._canonical_json(generation_body)
        ).hexdigest()
        generation = {
            **generation_body,
            "contentSha256": content_checksum,
        }
        generation_relative = "generations/%s/manifest.json" % content_checksum
        generation_path = self._safe_store_path(store, generation_relative)
        generation_bytes = self._canonical_json(generation)
        generation_sha = hashlib.sha256(generation_bytes).hexdigest()
        if not generation_path.exists():
            self._check_growth(len(generation_bytes) + 4096)
            atomic_write_bytes(generation_path, generation_bytes)
        synapse_descriptors = [
            item
            for item in generation_body["shards"]
            if item.get("kind") == "synapses"
        ]
        if lazy_synapses is not None:
            forward_index_entries = lazy_synapses.forward_index_entries(
                synapse_descriptors,
                lazy_changed_groups,
                forward_hot_node_ids,
            )
        published_forward_index = _write_forward_index(
            store,
            generation=content_checksum,
            generation_manifest_sha256=generation_sha,
            synapse_count=len(self.synapses),
            records_per_shard=records_per_shard,
            hot_node_ids=forward_hot_node_ids,
            entries=forward_index_entries,
            growth_guard=self.growth_guard,
        )
        pointer = {
            "format": _SUBSTRATE_STORE_FORMAT,
            "formatVersion": _SUBSTRATE_STORE_VERSION,
            "activeGeneration": content_checksum,
            "generationManifest": generation_relative,
            "generationManifestSha256": generation_sha,
            "counts": dict(generation_body["counts"]),
            "shardCount": len(shards),
            "contentSha256": content_checksum,
        }
        atomic_write_json(store / "manifest.json", pointer)
        if lazy_synapses is not None:
            lazy_synapses.commit_generation(
                synapse_descriptors,
                lazy_changed_groups,
                (
                    str(record.get("id", ""))
                    for record in self.assemblies
                ),
                forward_manifest=published_forward_index,
            )
            lazy_synapses.install_forward_index_manifest(
                published_forward_index
            )
        self.persistence_manifest = pointer
        self._persistence_record_groups = next_record_groups
        self._persistence_records_per_shard = int(records_per_shard)
        return dict(pointer)

    @classmethod
    def _generation_references(
        cls, root: Path, pointer: Dict[str, Any]
    ) -> Tuple[str, set[str]]:
        generation_id = str(pointer.get("activeGeneration", ""))
        relative = str(pointer.get("generationManifest", ""))
        if (
            len(generation_id) != 64
            or any(character not in "0123456789abcdef" for character in generation_id)
            or relative != "generations/%s/manifest.json" % generation_id
        ):
            raise ValueError("substrate generation identity is invalid")
        path = cls._safe_store_path(root, relative)
        generation_bytes = path.read_bytes()
        if hashlib.sha256(generation_bytes).hexdigest() != str(
            pointer.get("generationManifestSha256", "")
        ):
            raise ValueError("substrate generation manifest checksum mismatch")
        generation = json.loads(generation_bytes.decode("utf-8"))
        body = {
            key: value
            for key, value in generation.items()
            if key != "contentSha256"
        }
        if (
            generation.get("format") != _SUBSTRATE_STORE_FORMAT
            or int(generation.get("formatVersion", 0))
            not in _READABLE_SUBSTRATE_STORE_VERSIONS
            or int(pointer.get("formatVersion", 0))
            != int(generation.get("formatVersion", 0))
            or hashlib.sha256(cls._canonical_json(body)).hexdigest()
            != generation_id
            or str(generation.get("contentSha256", "")) != generation_id
            or str(pointer.get("contentSha256", "")) != generation_id
        ):
            raise ValueError("substrate generation content checksum mismatch")
        references: set[str] = set()
        for shard in generation.get("shards", []):
            if not isinstance(shard, dict):
                raise ValueError("substrate shard entry is invalid")
            for field, suffix in (
                ("records", ".json"),
                ("tensors", ".safetensors"),
            ):
                spec = shard.get(field)
                if spec is None and field == "tensors":
                    continue
                if not isinstance(spec, dict):
                    raise ValueError("substrate shard blob reference is invalid")
                checksum = str(spec.get("sha256", ""))
                expected = "blobs/%s%s" % (checksum, suffix)
                if (
                    len(checksum) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in checksum
                    )
                    or str(spec.get("path", "")) != expected
                ):
                    raise ValueError("substrate shard blob identity is invalid")
                references.add(expected)
        return generation_id, references

    @classmethod
    def prune_unreferenced(
        cls,
        root: Path,
        pointers: Sequence[Optional[Dict[str, Any]]],
        *, on_blob_removed: Optional[Callable[[Path], None]] = None,
    ) -> Dict[str, Any]:
        """Reclaim shards unreachable from the active and prior checkpoints.

        Sparse neural state itself remains unbounded. This removes only old
        serialized versions after the authoritative engine checkpoint has
        advanced, preventing continual learning from duplicating disk usage on
        every save.
        """

        store = Path(root).resolve()
        try:
            retained_generations: set[str] = set()
            retained_blobs: set[str] = set()
            for pointer in pointers:
                if not isinstance(pointer, dict) or not pointer:
                    continue
                generation_id, references = cls._generation_references(
                    store, pointer
                )
                retained_generations.add(generation_id)
                retained_blobs.update(references)
            if not retained_generations:
                raise ValueError("no committed substrate generation to retain")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            return {
                "completed": False,
                "reason": str(error),
                "generationsRetained": 0,
                "generationsRemoved": 0,
                "blobsRemoved": 0,
                "bytesReclaimed": 0,
            }

        generations_removed = 0
        blobs_removed = 0
        bytes_reclaimed = 0
        forward_cache_gc = True
        retained_forward_blobs: set[str] = set()
        try:
            for generation_id in retained_generations:
                path = _forward_index_path(store, generation_id)
                value = json.loads(path.read_bytes())
                body = {key: item for key, item in value.items() if key != "contentSha256"}
                if value.get("sourceGeneration") != generation_id or value.get("contentSha256") != hashlib.sha256(cls._canonical_json(body)).hexdigest():
                    raise ValueError("retained derived forward manifest is not checked")
                if value.get("formatVersion") == 4:
                    for entry in value["shards"]:
                        spec = entry["entry"]
                        relative = "forward-index/blobs/%s.json" % spec["sha256"]
                        if not _is_sha256(spec["sha256"]) or spec["path"] != relative:
                            raise ValueError("retained derived forward group reference is invalid")
                        retained_forward_blobs.add(relative)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # An optional derived cache cannot veto authoritative neural GC;
            # nor can a missing/bad retained cache justify deleting its blobs.
            forward_cache_gc = False
        generations = store / "generations"
        if generations.is_dir():
            for candidate in generations.iterdir():
                if (
                    candidate.name in retained_generations
                    or len(candidate.name) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in candidate.name
                    )
                    or candidate.is_symlink()
                    or not candidate.is_dir()
                ):
                    continue
                bytes_reclaimed += sum(
                    item.stat().st_size
                    for item in candidate.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
                shutil.rmtree(candidate)
                if on_blob_removed is not None:
                    on_blob_removed(candidate / "manifest.json")
                generations_removed += 1

        blobs = store / "blobs"
        if blobs.is_dir():
            for candidate in blobs.iterdir():
                relative = "blobs/" + candidate.name
                stem = candidate.name.split(".", 1)[0]
                if (
                    relative in retained_blobs
                    or candidate.is_symlink()
                    or not candidate.is_file()
                    or candidate.suffix not in {".json", ".safetensors"}
                    or len(stem) != 64
                    or any(character not in "0123456789abcdef" for character in stem)
                ):
                    continue
                size = candidate.stat().st_size
                candidate.unlink()
                if on_blob_removed is not None:
                    on_blob_removed(candidate)
                blobs_removed += 1
                bytes_reclaimed += size

        if forward_cache_gc:
            for directory, retained, suffix in (
                (store / "forward-index" / "generations", {value + ".json" for value in retained_generations}, ".json"),
                (store / "forward-index" / "blobs", {Path(value).name for value in retained_forward_blobs}, ".json"),
            ):
                if not directory.is_dir() or directory.is_symlink():
                    continue
                for candidate in directory.iterdir():
                    if candidate.name in retained or candidate.is_symlink() or not candidate.is_file() or candidate.suffix != suffix or not _is_sha256(candidate.stem):
                        continue
                    size = candidate.stat().st_size
                    candidate.unlink()
                    if on_blob_removed is not None:
                        on_blob_removed(candidate)
                    blobs_removed += 1
                    bytes_reclaimed += size

        query_indexes_removed = 0
        query_directory = store / "inspection" / "query-generations"
        if query_directory.is_dir() and not query_directory.is_symlink():
            for candidate in query_directory.iterdir():
                if candidate.name in retained_generations or not _is_sha256(candidate.name) or candidate.is_symlink() or not candidate.is_dir():
                    continue
                try:
                    manifest = json.loads((candidate / "manifest.json").read_bytes())
                    files = list(candidate.iterdir())
                    if manifest.get("format") != "omni-immutable-substrate-query-index" or any(
                        item.name not in {"manifest.json", "records.sqlite3"} or item.is_symlink() or not item.is_file() for item in files
                    ):
                        continue
                    bytes_reclaimed += sum(item.stat().st_size for item in files)
                    shutil.rmtree(candidate)
                    if on_blob_removed is not None:
                        for item in files:
                            on_blob_removed(item)
                    query_indexes_removed += 1
                except (OSError, ValueError, TypeError):
                    continue

        return {
            "completed": True,
            "reason": "unreferenced substrate generations reclaimed",
            "generationsRetained": len(retained_generations),
            "generationsRemoved": generations_removed,
            "blobsRemoved": blobs_removed,
            "bytesReclaimed": bytes_reclaimed,
            "forwardCacheGcCompleted": forward_cache_gc,
            "queryIndexesRemoved": query_indexes_removed,
        }

    @classmethod
    def load_sharded(
        cls,
        root: Path,
        metadata: Dict[str, Any],
        *,
        growth_guard: Optional[Callable[[int], bool]] = None,
        lazy_synapses: Optional[bool] = None,
        paged_vectors: Optional[PagedPackedVectors] = None,
        paged_neurons: Optional[PagedNeuronMetadata] = None,
        defer_paged_assemblies: bool = False,
    ) -> "NeuralSubstrate":
        store = Path(root).resolve()
        if metadata.get("schema") != cls.SCHEMA:
            raise ValueError("substrate metadata schema is incompatible")
        persistence = metadata.get("persistence")
        pointer = (
            dict(persistence)
            if isinstance(persistence, dict)
            else read_json(store / "manifest.json")
        )
        if (
            pointer.get("format") != _SUBSTRATE_STORE_FORMAT
            or int(pointer.get("formatVersion", 0))
            not in _READABLE_SUBSTRATE_STORE_VERSIONS
        ):
            raise ValueError("unsupported neural substrate shard format")
        if int(pointer.get("formatVersion", 0)) != _SUBSTRATE_STORE_VERSION:
            raise ValueError(
                "legacy higher-precision VSA vector state cannot be loaded "
                "without an explicit migration"
            )
        generation_path = cls._safe_store_path(
            store, str(pointer.get("generationManifest", ""))
        )
        active_generation = str(pointer.get("activeGeneration", ""))
        if (
            len(active_generation) != 64
            or any(
                character not in "0123456789abcdef"
                for character in active_generation
            )
            or str(pointer.get("generationManifest", ""))
            != "generations/%s/manifest.json" % active_generation
        ):
            raise ValueError("substrate generation identity is invalid")
        generation_bytes = generation_path.read_bytes()
        generation_manifest_sha256 = str(
            pointer.get("generationManifestSha256", "")
        )
        if hashlib.sha256(generation_bytes).hexdigest() != generation_manifest_sha256:
            raise ValueError("substrate generation manifest checksum mismatch")
        generation = json.loads(generation_bytes.decode("utf-8"))
        if (
            not isinstance(generation, dict)
            or generation.get("format") != _SUBSTRATE_STORE_FORMAT
            or int(generation.get("formatVersion", 0))
            != int(pointer.get("formatVersion", 0))
            or generation.get("schema") != cls.SCHEMA
        ):
            raise ValueError("substrate generation manifest is incompatible")
        content_body = {
            key: value
            for key, value in generation.items()
            if key != "contentSha256"
        }
        content_checksum = hashlib.sha256(
            cls._canonical_json(content_body)
        ).hexdigest()
        if (
            content_checksum != str(generation.get("contentSha256", ""))
            or content_checksum != str(pointer.get("activeGeneration", ""))
            or content_checksum != str(pointer.get("contentSha256", ""))
        ):
            raise ValueError("substrate generation content checksum mismatch")
        if (
            int(generation["dimensions"]) != int(metadata.get("dimensions", -1))
            or int(generation["seed"]) != int(metadata.get("seed", -1))
            or generation.get("counts") != pointer.get("counts")
        ):
            raise ValueError("substrate generation does not match engine metadata")
        substrate = cls(
            dimensions=int(generation["dimensions"]),
            seed=int(generation["seed"]),
            growth_guard=growth_guard,
        )
        if paged_vectors is not None:
            if (
                not isinstance(paged_vectors, PagedPackedVectors)
                or paged_vectors.dimensions != substrate.space.dimensions
                or paged_vectors.seed != substrate.space.seed
                or len(paged_vectors) != 0
            ):
                raise ValueError("paged vector load requires an empty compatible cache")
            substrate.neuron_vectors = paged_vectors
            substrate.assembly_vectors = PackedTernaryVectorView(paged_vectors)
        if paged_neurons is not None:
            if (
                not isinstance(paged_neurons, PagedNeuronMetadata)
                or paged_vectors is None
                or paged_neurons.path.resolve() != paged_vectors.path.resolve()
                or len(paged_neurons) != 0
            ):
                raise ValueError("paged neuron load needs the same empty vector cache")
            substrate.neurons = paged_neurons
        if defer_paged_assemblies and paged_neurons is None:
            raise ValueError("deferred assembly load needs paged neuron metadata")
        substrate.growth_events = int(generation.get("growthEvents", 0))
        substrate.growth_pauses = int(generation.get("growthPauses", 0))
        substrate.state_revision = int(generation.get("stateRevision", 0))
        expected_counts = generation.get("counts", {})
        expected_synapses = int(expected_counts.get("synapses", 0))
        use_lazy_synapses = (
            expected_synapses >= _LAZY_SYNAPSE_LOAD_THRESHOLD
            if lazy_synapses is None
            else bool(lazy_synapses)
        )
        if defer_paged_assemblies and not use_lazy_synapses:
            raise ValueError("deferred assembly load requires lazy synapses")
        lazy_synapse_shards: List[Dict[str, Any]] = []
        observed_deferred_assemblies = 0

        for shard in generation.get("shards", []):
            if not isinstance(shard, dict):
                raise ValueError("substrate shard entry is invalid")
            kind = str(shard.get("kind", ""))
            if defer_paged_assemblies and (
                type(shard.get("count")) is not int
                or not 1 <= shard["count"] <= 512
            ):
                raise SubstrateResourcePause(
                    "paged cold-load shard exceeds bounded row window"
                )
            if kind == "synapses" and use_lazy_synapses:
                bucket = str(shard.get("bucket", ""))
                part = int(shard.get("part", -1))
                if (
                    len(bucket) != 1
                    or bucket not in "0123456789abcdef"
                    or part < 0
                ):
                    raise ValueError("substrate shard placement is invalid")
                lazy_synapse_shards.append(dict(shard))
                continue
            record_spec = shard.get("records")
            if not isinstance(record_spec, dict):
                raise ValueError("substrate record shard is invalid")
            record_path = cls._safe_store_path(
                store, str(record_spec.get("path", ""))
            )
            if str(record_spec.get("path", "")) != "blobs/%s.json" % str(
                record_spec.get("sha256", "")
            ):
                raise ValueError("substrate record shard identity is invalid")
            if defer_paged_assemblies and (
                type(record_spec.get("bytes")) is not int
                or not 0 <= record_spec["bytes"] <= 64 * 1024 * 1024
            ):
                raise SubstrateResourcePause(
                    "paged cold-load record blob exceeds bounded read window"
                )
            record_before = record_path.stat()
            if (
                cls._file_sha256(record_path)
                != str(record_spec.get("sha256", ""))
                or record_path.stat().st_size != int(record_spec.get("bytes", -1))
            ):
                raise ValueError("substrate record shard checksum mismatch")
            if paged_vectors is not None:
                from .paged_substrate_writer import remember_verified_blob
                remember_verified_blob(
                    record_path, str(record_spec["sha256"]), before=record_before,
                )
            payload = json.loads(record_path.read_text("utf-8"))
            records = payload.get("records", [])
            if len(records) != int(shard.get("count", -1)):
                raise ValueError("substrate record shard count mismatch")

            tensor_values: Dict[str, torch.Tensor] = {}
            tensor_spec = shard.get("tensors")
            if tensor_spec is not None:
                if not isinstance(tensor_spec, dict):
                    raise ValueError("substrate tensor shard is invalid")
                tensor_path = cls._safe_store_path(
                    store, str(tensor_spec.get("path", ""))
                )
                if str(
                    tensor_spec.get("path", "")
                ) != "blobs/%s.safetensors" % str(
                    tensor_spec.get("sha256", "")
                ):
                    raise ValueError("substrate tensor shard identity is invalid")
                if defer_paged_assemblies and (
                    type(tensor_spec.get("bytes")) is not int
                    or not 0 <= tensor_spec["bytes"] <= 64 * 1024 * 1024
                ):
                    raise SubstrateResourcePause(
                        "paged cold-load tensor blob exceeds bounded read window"
                    )
                tensor_before = tensor_path.stat()
                if (
                    cls._file_sha256(tensor_path)
                    != str(tensor_spec.get("sha256", ""))
                    or tensor_path.stat().st_size
                    != int(tensor_spec.get("bytes", -1))
                ):
                    raise ValueError("substrate tensor shard checksum mismatch")
                if paged_vectors is not None:
                    from .paged_substrate_writer import remember_verified_blob
                    remember_verified_blob(
                        tensor_path, str(tensor_spec["sha256"]), before=tensor_before,
                    )
                tensor_values = load_tensors(tensor_path, device="cpu")
                if int(generation.get("formatVersion", 0)) == 1:
                    # Native v1 may carry a float shadow. Keep only the
                    # effective ternary value after checksum validation.
                    tensor_values.pop("latent_weight", None)

            if kind == "neurons":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("neuron shard identifiers do not match")
                if paged_neurons is None:
                    substrate.neurons.update(
                        {record_id: dict(record) for record_id, record in zip(ids, records)}
                    )
                else:
                    for start in range(0, len(records), 64):
                        page = records[start : start + 64]
                        if paged_neurons.import_page(
                            page, max_rows=64,
                            max_payload_bytes=20 * 1024 * 1024,
                        ) != len(page):
                            raise ValueError("paged neuron shard import missed records")
                vector_ids = [
                    str(item) for item in payload.get("vectorIds", [])
                ]
                if vector_ids != ids or set(tensor_values) != {
                    "packed_rows", "update_counters_le"
                }:
                    raise ValueError("neuron shard lacks exact packed vector state")
                vector_metadata = payload.get("packedVectorState")
                if (
                    not isinstance(vector_metadata, Mapping)
                    or vector_metadata.get("ids") != vector_ids
                ):
                    raise ValueError("neuron packed vector IDs do not match shard")
                if paged_vectors is None:
                    shard_vectors = PackedTernaryVectors.from_state(
                        vector_metadata, tensor_values
                    )
                    substrate.neuron_vectors.update_packed(shard_vectors)
                elif paged_vectors.import_state(vector_metadata, tensor_values) != len(ids):
                    raise ValueError("paged neuron shard import missed packed rows")
            elif kind == "assemblies":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("assembly shard identifiers do not match")
                if defer_paged_assemblies:
                    observed_deferred_assemblies += len(records)
                else:
                    substrate.assemblies.extend(dict(record) for record in records)
                vector_ids = [
                    str(item) for item in payload.get("vectorIds", [])
                ]
                if (
                    vector_ids != ids
                    or payload.get("vectorStorage") != "shared-neuron-packed"
                    or tensor_spec is not None
                ):
                    raise ValueError("assembly shard must alias packed neuron rows")
            elif kind == "synapses":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("synapse shard identifiers do not match")
                tensor_values["effective_weight"] = _unpack_persisted_synapse_weights(
                    tensor_values,
                    len(ids),
                    int(generation.get("formatVersion", 0)),
                )
                for field in _SYNAPSE_TENSOR_FIELDS:
                    values = tensor_values.get(field)
                    if values is None or values.numel() != len(ids):
                        raise ValueError(
                            "synapse tensor shard is missing " + field
                        )
                for index, (record_id, record) in enumerate(zip(ids, records)):
                    restored = dict(record)
                    restored.update(
                        {
                            "effective_weight": int(
                                tensor_values["effective_weight"][index].item()
                            ),
                            "eligibility": float(
                                tensor_values["eligibility"][index].item()
                            ),
                            "plasticity": float(
                                tensor_values["plasticity"][index].item()
                            ),
                            "uses": int(tensor_values["uses"][index].item()),
                            "stability": float(
                                tensor_values["stability"][index].item()
                            ),
                            "last_updated_at": float(
                                tensor_values["last_updated_at"][index].item()
                            ),
                        }
                    )
                    if restored["effective_weight"] not in {-1, 0, 1}:
                        raise ValueError(
                            "substrate shard contains a non-ternary live synapse"
                        )
                    substrate.synapses[record_id] = restored
            else:
                raise ValueError("unknown substrate shard kind")
            bucket = str(shard.get("bucket", ""))
            part = int(shard.get("part", -1))
            if part < 0 or any(
                cls._bucket(kind, record_id) != bucket for record_id in ids
            ):
                raise ValueError("substrate shard placement is invalid")
            if not defer_paged_assemblies:
                substrate._persistence_record_groups.update(
                    {
                        (kind, record_id): (bucket, part)
                        for record_id in ids
                    }
                )

        if use_lazy_synapses and defer_paged_assemblies:
            # The verified-cache finisher rebuilds assembly membership from
            # bounded shard pages, then installs lazy synapses against that
            # paged membership. A half-loaded substrate is never published.
            substrate._deferred_lazy_synapse_shards = lazy_synapse_shards
            substrate._paged_load_incomplete = True
        elif use_lazy_synapses:
            hot_node_ids = {
                str(record.get("id", ""))
                for record in substrate.assemblies
                if record.get("id")
            }
            forward_index = _load_forward_index(
                store,
                generation=active_generation,
                generation_manifest_sha256=generation_manifest_sha256,
                synapse_count=expected_synapses,
                records_per_shard=int(generation.get("recordsPerShard", 0)),
                hot_node_ids=hot_node_ids,
                descriptors=lazy_synapse_shards,
            )
            substrate.synapses = LazyPersistedSynapses(
                store,
                lazy_synapse_shards,
                expected_synapses,
                int(generation.get("recordsPerShard", 0)),
                hot_node_ids,
                forward_index,
                store_version=int(generation.get("formatVersion", 0)),
            )
            if forward_index is None:
                # One bounded legacy migration. Future loads validate only the
                # generation manifest and this compact forward index; cold
                # record/tensor shards remain unopened until addressed.
                entries = substrate.synapses.forward_index_entries(
                    lazy_synapse_shards,
                    {},
                    hot_node_ids,
                )
                try:
                    published_forward_index = _write_forward_index(
                        store,
                        generation=active_generation,
                        generation_manifest_sha256=generation_manifest_sha256,
                        synapse_count=expected_synapses,
                        records_per_shard=int(
                            generation.get("recordsPerShard", 0)
                        ),
                        hot_node_ids=hot_node_ids,
                        entries=entries,
                        growth_guard=growth_guard,
                    )
                    substrate.synapses.install_forward_index_manifest(
                        published_forward_index
                    )
                except SubstrateResourcePause:
                    # The in-memory index is already exact. Never cross the
                    # disk reserve merely to accelerate a future restart.
                    pass
        observed_counts = {
            "neurons": len(substrate.neurons),
            "assemblies": (
                observed_deferred_assemblies
                if defer_paged_assemblies else len(substrate.assemblies)
            ),
            "synapses": (
                sum(int(item.get("count", -1)) for item in lazy_synapse_shards)
                if defer_paged_assemblies and use_lazy_synapses
                else len(substrate.synapses)
            ),
        }
        if observed_counts != expected_counts:
            raise ValueError("substrate shard generation count mismatch")
        if not defer_paged_assemblies:
            substrate.assemblies.sort(
                key=lambda item: int(item.get("__persistence_ordinal", 0))
            )
            substrate.invalidate_assembly_index()
            for item in substrate.assemblies:
                item.pop("__persistence_ordinal", None)
                substrate.assembly_vectors.link(str(item.get("id", "")))
            substrate._validate_packed_vector_identity()
        substrate.persistence_manifest = dict(pointer)
        substrate._persistence_records_per_shard = int(
            generation.get("recordsPerShard", 0)
        )
        # A direct substrate load has no attention overlay to interpret.
        # AdaptiveBrain.configure_attention_overlay replaces this with the
        # saved native epoch mode when its runtime state is loaded.
        substrate.attention_legacy_raw_active = True
        return substrate

    def tensor_state(self, prefix: str = "substrate.") -> Dict[str, torch.Tensor]:
        if isinstance(self.assemblies, PagedAssemblyView):
            raise ValueError("paged assemblies cannot use monolithic tensor export")
        self._validate_packed_vector_identity()
        _metadata, tensors = self.neuron_vectors.export_state(
            prefix=prefix + "vectors."
        )
        return tensors

    @classmethod
    def from_state(
        cls,
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        prefix: str = "substrate.",
    ) -> "NeuralSubstrate":
        if metadata.get("schema") != cls.SCHEMA:
            raise ValueError(
                "legacy higher-precision VSA vector state requires an explicit migration"
            )
        if (
            "neuron_vector_ids" in metadata
            or "assembly_vector_ids" in metadata
            or prefix + "neuron_vectors" in tensors
            or prefix + "assembly_vectors" in tensors
        ):
            raise ValueError("legacy float VSA vector state cannot be loaded")
        substrate = cls(
            dimensions=int(metadata["dimensions"]),
            seed=int(metadata["seed"]),
        )
        substrate.neurons = _RevisionedNodes({
            item["id"]: dict(item) for item in metadata.get("neurons", [])
        })
        substrate.assemblies = [
            dict(item) for item in metadata.get("assemblies", [])
        ]
        substrate.synapses = _RevisionedSynapses({
            item["id"]: {
                key: value for key, value in item.items()
                if key != "latent_weight"
            }
            for item in metadata.get("synapses", [])
        })
        substrate.attention_legacy_raw_active = True
        substrate.growth_events = int(metadata.get("growth_events", 0))
        substrate.growth_pauses = int(metadata.get("growth_pauses", 0))
        substrate.state_revision = int(metadata.get("state_revision", 0))
        substrate.neuron_vectors = PackedTernaryVectors.from_state(
            metadata.get("packed_vector_state"),
            tensors,
            prefix=prefix + "vectors.",
        )
        substrate.assembly_vectors = PackedTernaryVectorView(
            substrate.neuron_vectors
        )
        for item in substrate.assemblies:
            substrate.assembly_vectors.link(str(item.get("id", "")))
        substrate._validate_packed_vector_identity()
        return substrate


class LazyPersistedSynapses(MutableMapping[str, Dict[str, Any]]):
    """Exact shard-backed synapses with a bounded mutable hot overlay.

    Large learned substrates contain millions of cold, usually zero-valued
    edges.  Reconstructing every edge as a Python dictionary multiplies the
    committed shard size into several gigabytes of heap before the first turn.
    A generation-bound, checksummed forward index retains the exact non-zero
    graph and locators without opening every cold shard during startup. A shard
    is checksum-verified before its first page-in, while an explicit/background
    scrub can verify the complete persisted generation independently.

    The mapping contract remains complete: ``len``, membership, iteration,
    mutation, deletion, packed export, and checkpoint save all address the
    full persisted cardinality.  Cold records are never approximated or
    dropped.
    """

    CACHE_SHARDS = 8

    def __init__(
        self,
        root: Path,
        shards: Sequence[Mapping[str, Any]],
        expected_count: int,
        records_per_shard: int,
        hot_node_ids: Iterable[str] = (),
        forward_index: Optional[Mapping[str, Any]] = None,
        *,
        store_version: int = _SUBSTRATE_STORE_VERSION,
    ) -> None:
        self.root = Path(root).resolve()
        self.store_version = int(store_version)
        if self.store_version not in _READABLE_SUBSTRATE_STORE_VERSIONS:
            raise ValueError("unsupported neural substrate shard format")
        self.graph_revision = 0
        self.persistence_revision = 0
        self._descriptors: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._ordered_keys: List[Tuple[str, int]] = []
        self._ranges: Dict[str, List[Tuple[str, str, Tuple[str, int]]]] = (
            defaultdict(list)
        )
        self._range_by_key: Dict[Tuple[str, int], Tuple[str, str]] = {}
        self._forward_by_shard: Dict[
            Tuple[str, int],
            Tuple[List[Tuple[int, str, str, str]], bytes],
        ] = {}
        self._uses_by_shard: Dict[Tuple[str, int], int] = {}
        self._cache: "OrderedDict[Tuple[str, int], Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]]" = OrderedDict()
        self._verified_shards: set[Tuple[str, int]] = set()
        self._locations: Dict[str, Tuple[Tuple[str, int], int]] = {}
        self._dirty: Dict[str, Dict[str, Any]] = {}
        self._original_uses: Dict[str, int] = {}
        self._dirty_locations: Dict[str, Tuple[Tuple[str, int], int]] = {}
        self._deleted_locations: Dict[str, Tuple[Tuple[str, int], int]] = {}
        self._new_ids: set[str] = set()
        # A paged assembly source supports exact indexed membership and
        # canonical ordered iteration without copying every assembly ID into
        # another process-resident Python set.
        if callable(getattr(hot_node_ids, "iter_sorted_ids", None)):
            self._hot_node_ids = hot_node_ids
        else:
            self._hot_node_ids = {
                str(value) for value in hot_node_ids if str(value)
            }
        self._hot_locations: Dict[
            str, List[Tuple[str, Tuple[str, int], int]]
        ] = defaultdict(list)
        self._base_count = int(expected_count)
        self.records_per_shard = int(records_per_shard)
        self._forward_index_manifest = (
            dict(forward_index) if isinstance(forward_index, Mapping) else None
        )
        self._persisted_forward_index = self._forward_index_manifest is not None
        if self._base_count < 0 or self.records_per_shard < 1:
            raise ValueError("lazy substrate records_per_shard is invalid")
        observed = 0
        for raw in shards:
            descriptor = dict(raw)
            bucket = descriptor.get("bucket")
            part = descriptor.get("part")
            count = descriptor.get("count")
            record_spec = descriptor.get("records")
            tensor_spec = descriptor.get("tensors")
            if not isinstance(bucket, str) or not _is_nonnegative_int(part):
                raise ValueError("substrate synapse shard placement is invalid")
            key = (bucket, part)
            if (
                descriptor.get("kind") != "synapses"
                or len(bucket) != 1
                or bucket not in "0123456789abcdef"
                or key in self._descriptors
                or not _is_nonnegative_int(count)
                or count < 1
                or count > self.records_per_shard
                or not isinstance(record_spec, Mapping)
                or not isinstance(tensor_spec, Mapping)
                or record_spec.get("path")
                != "blobs/%s.json" % str(record_spec.get("sha256", ""))
                or tensor_spec.get("path")
                != "blobs/%s.safetensors" % str(tensor_spec.get("sha256", ""))
                or not _is_sha256(record_spec.get("sha256"))
                or not _is_sha256(tensor_spec.get("sha256"))
                or not _is_nonnegative_int(record_spec.get("bytes"))
                or not _is_nonnegative_int(tensor_spec.get("bytes"))
            ):
                raise ValueError("substrate synapse shard placement is invalid")
            self._descriptors[key] = descriptor
            self._ordered_keys.append(key)
        self._ordered_keys.sort()
        if isinstance(forward_index, Mapping) and forward_index.get("formatVersion") == 4:
            from .paged_forward_index import BlobForwardTopology, EndpointHotLocations
            topology = BlobForwardTopology(self.root, forward_index["shards"])
            if set(topology) != set(self._descriptors):
                raise ValueError("paged forward group coverage differs from source descriptors")
            for key in self._ordered_keys:
                descriptor = self._descriptors[key]
                entry = topology.descriptors[key]
                first_id, last_id = entry.get("firstId"), entry.get("lastId")
                if (
                    not isinstance(first_id, str) or not isinstance(last_id, str) or first_id > last_id
                    or NeuralSubstrate._bucket("synapses", first_id) != key[0]
                    or NeuralSubstrate._bucket("synapses", last_id) != key[0]
                    or entry.get("count") != descriptor["count"]
                    or entry.get("recordsSha256") != descriptor["records"]["sha256"]
                    or entry.get("tensorsSha256") != descriptor["tensors"]["sha256"]
                    or not _is_nonnegative_int(entry.get("synapticUses"))
                    or not _is_nonnegative_int(entry.get("forwardCount"))
                    or entry["forwardCount"] > descriptor["count"]
                    or entry.get("packedBytes") != (entry["forwardCount"] + 3) // 4
                ):
                    raise ValueError("paged forward group descriptor binding is invalid")
                self._range_by_key[key] = (first_id, last_id)
                self._ranges[key[0]].append((first_id, last_id, key))
                self._uses_by_shard[key] = entry["synapticUses"]
                observed += descriptor["count"]
            for bucket in self._ranges:
                self._ranges[bucket].sort(key=lambda value: (value[0], value[1], value[2]))
            if observed != self._base_count or sum(self._uses_by_shard.values()) != forward_index.get("synapticUses"):
                raise ValueError("paged forward group aggregate diverges")
            self._forward_by_shard = topology
            self._hot_locations = EndpointHotLocations(self.root, topology, self._hot_node_ids, forward_index)
            return
        indexed_entries: Dict[Tuple[str, int], Mapping[str, Any]] = {}
        if isinstance(forward_index, Mapping):
            for value in forward_index.get("shards", []):
                if not isinstance(value, Mapping):
                    raise ValueError("substrate forward index shard is invalid")
                key = (value.get("bucket"), value.get("part"))
                if key in indexed_entries:
                    raise ValueError("substrate forward index shard is duplicated")
                indexed_entries[key] = value
        for key in self._ordered_keys:
            descriptor = self._descriptors[key]
            entry = indexed_entries.get(key)
            if entry is not None:
                observed += int(descriptor["count"])
                first_id = entry.get("firstId")
                last_id = entry.get("lastId")
                if (
                    not isinstance(first_id, str)
                    or not first_id
                    or not isinstance(last_id, str)
                    or not last_id
                    or first_id > last_id
                    or NeuralSubstrate._bucket("synapses", first_id) != key[0]
                    or NeuralSubstrate._bucket("synapses", last_id) != key[0]
                ):
                    raise ValueError("substrate forward index range is invalid")
                self._range_by_key[key] = (first_id, last_id)
                self._ranges[key[0]].append((first_id, last_id, key))
                raw_forward = entry.get("forwardRecords")
                raw_hot = entry.get("hotLocations")
                if not isinstance(raw_forward, list) or not isinstance(raw_hot, list):
                    raise ValueError("substrate forward index records are invalid")
                forward: List[Tuple[int, str, str, str]] = []
                for value in raw_forward:
                    if (
                        not isinstance(value, list)
                        or len(value) != 4
                        or isinstance(value[0], bool)
                        or not isinstance(value[0], int)
                        or not 0 <= value[0] < int(descriptor["count"])
                        or not all(isinstance(item, str) and item for item in value[1:])
                        or not first_id <= value[1] <= last_id
                        or NeuralSubstrate._bucket("synapses", value[1]) != key[0]
                        or not _synapse_id_matches_endpoints(
                            value[1], value[2], value[3]
                        )
                    ):
                        raise ValueError("substrate forward index edge is invalid")
                    forward.append((value[0], value[1], value[2], value[3]))
                if [value[0] for value in forward] != sorted(
                    {value[0] for value in forward}
                ) or len({value[1] for value in forward}) != len(forward):
                    raise ValueError("substrate forward index edges are duplicated")
                try:
                    packed = base64.b64decode(
                        str(entry.get("packedEffectiveWeights", "")),
                        validate=True,
                    )
                except ValueError as error:
                    raise ValueError("substrate forward index weights are invalid") from error
                levels = [
                    _unpack_ternary_level(packed, index)
                    for index in range(len(forward))
                ]
                if (
                    len(packed) != (len(forward) + 3) // 4
                    or any(level == 0 for level in levels)
                    or _pack_ternary_levels(levels) != packed
                ):
                    raise ValueError("substrate forward index weights are invalid")
                self._forward_by_shard[key] = (forward, packed)
                shard_uses = entry.get("synapticUses")
                if not _is_nonnegative_int(shard_uses):
                    raise ValueError(
                        "substrate forward index synaptic uses are invalid"
                    )
                self._uses_by_shard[key] = shard_uses
                seen_hot: set[Tuple[str, str, int]] = set()
                contains_many = getattr(self._hot_node_ids, "contains_many", None)
                allowed_hot_ids = (
                    contains_many(
                        {
                            str(value[0])
                            for value in raw_hot
                            if isinstance(value, list) and value
                        }
                    )
                    if callable(contains_many)
                    else self._hot_node_ids
                )
                for value in raw_hot:
                    endpoints = (
                        _synapse_id_endpoints(value[1])
                        if isinstance(value, list)
                        and len(value) == 3
                        and isinstance(value[1], str)
                        else None
                    )
                    if (
                        not isinstance(value, list)
                        or len(value) != 3
                        or value[0] not in allowed_hot_ids
                        or not isinstance(value[1], str)
                        or isinstance(value[2], bool)
                        or not isinstance(value[2], int)
                        or not 0 <= value[2] < int(descriptor["count"])
                        or not first_id <= value[1] <= last_id
                        or NeuralSubstrate._bucket("synapses", value[1]) != key[0]
                        or endpoints is None
                        or value[0] not in endpoints
                        or (value[0], value[1], value[2]) in seen_hot
                    ):
                        raise ValueError("substrate forward index hot location is invalid")
                    seen_hot.add((value[0], value[1], value[2]))
                    self._hot_locations[value[0]].append(
                        (value[1], key, value[2])
                    )
                continue
            records, tensors = self._read_shard(key, validate=True)
            observed += len(records)
            if records:
                identifiers = [str(record.get("id", "")) for record in records]
                self._range_by_key[key] = (identifiers[0], identifiers[-1])
                self._ranges[key[0]].append((identifiers[0], identifiers[-1], key))
            forward = []
            forward_levels: List[int] = []
            effective = tensors["effective_weight"]
            uses = tensors["uses"]
            shard_uses = 0
            for index, record in enumerate(records):
                record_id = str(record["id"])
                source_id = str(record["source_id"])
                target_id = str(record["target_id"])
                if source_id in self._hot_node_ids:
                    self._hot_locations[source_id].append((record_id, key, index))
                if target_id in self._hot_node_ids and target_id != source_id:
                    self._hot_locations[target_id].append((record_id, key, index))
                weight = NeuralSubstrate.exact_effective_weight(
                    effective[index].item()
                )
                use_count = int(uses[index].item())
                if use_count < 0:
                    raise ValueError("substrate synaptic uses are invalid")
                shard_uses += use_count
                if weight:
                    forward.append((index, record_id, source_id, target_id))
                    forward_levels.append(weight)
            self._forward_by_shard[key] = (
                forward, _pack_ternary_levels(forward_levels)
            )
            self._uses_by_shard[key] = shard_uses
        if observed != self._base_count:
            raise ValueError("substrate shard generation count mismatch")
        if self._persisted_forward_index and set(indexed_entries) != set(
            self._descriptors
        ):
            raise ValueError("substrate forward index shard coverage diverges")
        if self._persisted_forward_index and int(
            self._forward_index_manifest.get("synapticUses", -1)
        ) != sum(self._uses_by_shard.values()):
            raise ValueError("substrate forward index synaptic uses diverge")
        for bucket in self._ranges:
            self._ranges[bucket].sort(key=lambda value: (value[0], value[1], value[2]))
        # Validation must not itself create resident Python record dictionaries.
        self._cache.clear()

    @property
    def persisted_cold_count(self) -> int:
        return max(0, len(self) - len(self._dirty))

    @property
    def resident_record_count(self) -> int:
        return len(self._dirty)

    @property
    def forward_edge_count(self) -> int:
        descriptors = getattr(self._forward_by_shard, "descriptors", None)
        if descriptors is not None:
            total = sum(row["forwardCount"] for row in descriptors.values())
            # Only changed complete groups are decoded for exact delta stats.
            keys = {key for key, _index in self._dirty_locations.values()}.union(
                key for key, _index in self._deleted_locations.values())
            for key in keys:
                structures, _packed = self._forward_by_shard[key]
                nonzero = {row[0] for row in structures}
                for identifier, (group, index) in self._dirty_locations.items():
                    if group == key and identifier not in self._deleted_locations:
                        total += int(bool(NeuralSubstrate.exact_effective_weight(self._dirty[identifier].get("effective_weight", 0)))) - int(index in nonzero)
                for group, index in self._deleted_locations.values():
                    if group == key:
                        total -= int(index in nonzero)
            total += sum(int(bool(NeuralSubstrate.exact_effective_weight(self._dirty[identifier].get("effective_weight", 0))))
                         for identifier in self._new_ids)
            return total
        return sum(1 for _edge in self.iter_effective_edges())

    @staticmethod
    def _exact_uses(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("substrate synaptic uses are invalid")
        return int(value)

    @property
    def synaptic_use_count(self) -> int:
        """Exact current use aggregate without paging untouched records."""

        total = sum(self._uses_by_shard.values())
        for record_id, record in self._dirty.items():
            current = self._exact_uses(record.get("uses", 0))
            if record_id in self._new_ids:
                total += current
            elif record_id not in self._deleted_locations:
                total += current - self._original_uses[record_id]
        for record_id in self._deleted_locations:
            total -= self._original_uses[record_id]
        if total < 0:
            raise ValueError("substrate synaptic uses aggregate is invalid")
        return total

    def dynamic_pack_state(self) -> Tuple[torch.Tensor, int, str, str]:
        """Build exact dense ternary state from index positions, not shards.

        Persistence-shard order is bound by each record blob checksum. This
        makes packed checkpoint validation O(shards + nonzero edges), while a
        full scrub remains available to re-derive every indexed position.
        """

        if self._new_ids or self._deleted_locations:
            raise RuntimeError(
                "lazy substrate must checkpoint structural changes before packing"
            )
        values = torch.zeros(self._base_count, dtype=torch.int8)
        order_records: List[Dict[str, Any]] = []
        offset = 0
        for key in self._ordered_keys:
            descriptor = self._descriptors[key]
            count = int(descriptor["count"])
            record_spec = descriptor["records"]
            order_records.append(
                {
                    "bucket": key[0],
                    "part": key[1],
                    "count": count,
                    "recordsSha256": str(record_spec["sha256"]),
                }
            )
            structures, packed = self._forward_by_shard.get(key, ([], b""))
            for packed_index, (index, record_id, _source, _target) in enumerate(
                structures
            ):
                replacement = self._dirty.get(record_id)
                level = (
                    NeuralSubstrate.exact_effective_weight(
                        replacement.get("effective_weight", 0)
                    )
                    if replacement is not None
                    else _unpack_ternary_level(packed, packed_index)
                )
                values[offset + index] = level
            indexed_nonzero = {record_id for _index, record_id, _s, _t in structures}
            for record_id, (location_key, index) in self._dirty_locations.items():
                if location_key != key or record_id in indexed_nonzero:
                    continue
                values[offset + index] = NeuralSubstrate.exact_effective_weight(
                    self._dirty[record_id].get("effective_weight", 0)
                )
            offset += count
        if offset != self._base_count:
            raise ValueError("substrate dynamic pack count diverges")
        order_sha256 = hashlib.sha256(
            NeuralSubstrate._canonical_json(order_records)
        ).hexdigest()
        return values, self._base_count, order_sha256, _DYNAMIC_ORDER_BASIS

    def stream_dynamic_pack_state(self, *, disk_reserve=None, memory_reserve=None):
        """Same canonical order/hash, with bounded exact-int8 pages only.

        Sparse zero edges remain present. A disk-sorted current delta avoids
        rescanning all touched IDs for each cold group. No full-edge tensor or
        ID list is allocated, and source revisions guard the complete stream.
        """

        from .streamed_ternary import TernaryPackedSource
        if self._new_ids or self._deleted_locations:
            raise RuntimeError("lazy substrate must checkpoint structural changes before packing")
        revision = self.graph_revision, self.persistence_revision
        digest = hashlib.sha256(b"[")
        for ordinal, key in enumerate(self._ordered_keys):
            descriptor = self._descriptors[key]
            row = {"bucket": key[0], "part": key[1], "count": descriptor["count"],
                   "recordsSha256": descriptor["records"]["sha256"]}
            if ordinal:
                digest.update(b",")
            digest.update(NeuralSubstrate._canonical_json(row))
        digest.update(b"]")
        source_count = self._base_count
        def require(size, operation):
            if disk_reserve is not None and disk_reserve(size, operation) is False:
                raise SubstrateResourcePause("streamed dynamic packing paused at disk reserve")
            if memory_reserve is not None and memory_reserve(2 * 1024 * 1024, operation) is False:
                raise SubstrateResourcePause("streamed dynamic packing paused at memory reserve")
        def pages():
            import sqlite3
            if revision != (self.graph_revision, self.persistence_revision):
                raise ValueError("dynamic source changed before streamed packing")
            require(65536, "dynamic packed delta scratch")
            staging = self.root / "staging"
            if staging.is_symlink():
                raise ValueError("dynamic packing staging path must not be a symlink")
            staging.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".dynamic-pack-", dir=staging) as folder:
                connection = sqlite3.connect(Path(folder) / "delta.sqlite3")
                try:
                    connection.execute("PRAGMA journal_mode=OFF")
                    connection.execute("PRAGMA synchronous=OFF")
                    connection.execute("PRAGMA cache_size=-256")
                    connection.execute("PRAGMA temp_store=FILE")
                    connection.execute("CREATE TABLE delta(bucket TEXT,part INTEGER,position INTEGER,level INTEGER,PRIMARY KEY(bucket,part,position)) WITHOUT ROWID")
                    for identifier, (key, position) in self._dirty_locations.items():
                        require(4096, "dynamic packed dirty position")
                        level = NeuralSubstrate.exact_effective_weight(self._dirty[identifier].get("effective_weight", 0))
                        connection.execute("INSERT INTO delta VALUES(?,?,?,?)", (*key, position, level))
                    connection.commit()
                    seen = 0
                    for key in self._ordered_keys:
                        if revision != (self.graph_revision, self.persistence_revision):
                            raise ValueError("dynamic source changed during streamed packing")
                        count = int(self._descriptors[key]["count"])
                        require(4096 + count * 4, "dynamic packed exact group")
                        values = torch.zeros(count, dtype=torch.int8)
                        structures, packed = self._forward_by_shard.get(key, ([], b""))
                        for offset, (position, _identifier, _source, _target) in enumerate(structures):
                            values[position] = _unpack_ternary_level(packed, offset)
                        for position, level in connection.execute("SELECT position,level FROM delta WHERE bucket=? AND part=?", key):
                            values[position] = level
                        seen += count
                        yield values
                    if seen != source_count or revision != (self.graph_revision, self.persistence_revision):
                        raise ValueError("streamed dynamic packing coverage/source revision differs")
                finally:
                    connection.close()
        return TernaryPackedSource((source_count,), pages), source_count, digest.hexdigest(), _DYNAMIC_ORDER_BASIS

    def paging_status(self) -> Dict[str, Any]:
        return {
            "mode": "generation-indexed-content-addressed-lazy-synapses",
            "totalSynapses": len(self),
            "residentSynapseRecords": self.resident_record_count,
            "persistedColdSynapses": self.persisted_cold_count,
            "forwardEdges": self.forward_edge_count,
            "synapticUses": self.synaptic_use_count,
            "packedForwardBytes": (
                sum(row["packedBytes"] for row in self._forward_by_shard.descriptors.values())
                if hasattr(self._forward_by_shard, "descriptors")
                else sum(len(packed) for _structures, packed in self._forward_by_shard.values())
            ),
            "forwardTopologyResidency": "paged-immutable-groups" if hasattr(self._forward_by_shard, "descriptors") else "legacy-resident",
            "forwardWeightsPerByte": 4,
            "denseForwardWeightsMaterialized": False,
            "shards": len(self._ordered_keys),
            "forwardIndexLoaded": self._persisted_forward_index,
            "verifiedShards": len(self._verified_shards),
            "pendingScrubShards": max(
                0, len(self._ordered_keys) - len(self._verified_shards)
            ),
            "allRecordsAddressable": True,
            "exactRecall": True,
            "transactionalCheckpoint": True,
        }

    def resident_items(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return self._dirty.items()

    def install_forward_index_manifest(
        self, manifest: Mapping[str, Any]
    ) -> None:
        """Record the exact atomic index published for the current generation."""

        self._forward_index_manifest = dict(manifest)
        self._persisted_forward_index = True

    def _persisted_index_entry(self, key: Tuple[str, int]) -> Dict[str, Any]:
        reused = getattr(self._forward_by_shard, "reused_entry", None)
        if callable(reused):
            return reused(key)
        descriptor = self._descriptors[key]
        record_spec = descriptor["records"]
        tensor_spec = descriptor["tensors"]
        bounds = self._range_by_key.get(key)
        if bounds is None:
            raise ValueError("substrate forward index range is missing")
        structures, packed = self._forward_by_shard.get(key, ([], b""))
        hot_locations = sorted(
            [node_id, record_id, index]
            for node_id, values in self._hot_locations.items()
            for record_id, location_key, index in values
            if location_key == key
        )
        return {
            "bucket": key[0],
            "part": key[1],
            "count": int(descriptor["count"]),
            "recordsSha256": str(record_spec["sha256"]),
            "tensorsSha256": str(tensor_spec["sha256"]),
            "firstId": bounds[0],
            "lastId": bounds[1],
            "synapticUses": self._uses_by_shard[key],
            "forwardRecords": [list(value) for value in structures],
            "packedEffectiveWeights": base64.b64encode(packed).decode("ascii"),
            "hotLocations": hot_locations,
        }

    def scrub_persisted_shards(
        self,
        progress: Optional[
            Callable[[int, int, Tuple[str, int]], None]
        ] = None,
    ) -> Dict[str, Any]:
        """Verify every cold shard and prove byte-backed/index parity.

        This is intentionally separate from cold startup. Callers may run it
        explicitly or on a background maintenance worker. It has no record
        cap and stops at the first checksum, schema, or index divergence.
        """

        manifest = self._forward_index_manifest
        if manifest is not None:
            current = _load_forward_index(
                self.root,
                generation=str(manifest.get("sourceGeneration", "")),
                generation_manifest_sha256=str(
                    manifest.get("sourceGenerationManifestSha256", "")
                ),
                synapse_count=self._base_count,
                records_per_shard=self.records_per_shard,
                hot_node_ids=self._hot_node_ids,
                descriptors=[self._descriptors[key] for key in self._ordered_keys],
            )
            if current is None or current != manifest:
                raise ValueError("substrate forward index changed after load")

        started = time.perf_counter()
        total = len(self._ordered_keys)
        verified_records = 0
        for ordinal, key in enumerate(self._ordered_keys, start=1):
            records, tensors = self._descriptor_payload(key, validate=True)
            group: List[Tuple[str, Mapping[str, Any]]] = []
            for index, record in enumerate(records):
                group.append(
                    (
                        str(record["id"]),
                        {
                            **record,
                            "effective_weight": tensors["effective_weight"][
                                index
                            ].item(),
                            "uses": int(tensors["uses"][index].item()),
                        },
                    )
                )
            expected = _forward_index_entry(
                self._descriptors[key], group, self._hot_node_ids
            )
            if expected != self._persisted_index_entry(key):
                raise ValueError(
                    "substrate forward index diverges from persisted shards"
                )
            self._verified_shards.add(key)
            verified_records += len(records)
            if progress is not None:
                progress(ordinal, total, key)
        if verified_records != self._base_count:
            raise ValueError("substrate scrub record count diverges")
        return {
            "verified": True,
            "shards": total,
            "synapses": verified_records,
            "generation": (
                str(manifest.get("sourceGeneration", ""))
                if manifest is not None
                else None
            ),
            "elapsedSeconds": time.perf_counter() - started,
        }

    def connected_records(
        self, node_ids: Iterable[str]
    ) -> Tuple[Dict[str, Any], ...]:
        """Page the complete exact edge set touching known hot assemblies."""

        requested = {str(value) for value in node_ids if str(value)}
        record_ids: List[str] = []
        seen: set[str] = set()
        for node_id in requested:
            for record_id, key, index in self._hot_locations.get(node_id, []):
                if record_id in seen or record_id in self._deleted_locations:
                    continue
                seen.add(record_id)
                self._locations.setdefault(record_id, (key, index))
                record_ids.append(record_id)
        for record_id, record in self._dirty.items():
            if record_id in seen:
                continue
            if requested.intersection(
                {str(record.get("source_id", "")), str(record.get("target_id", ""))}
            ):
                seen.add(record_id)
                record_ids.append(record_id)
        return tuple(self[record_id] for record_id in record_ids)

    def _descriptor_payload(
        self,
        key: Tuple[str, int],
        *,
        validate: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        descriptor = self._descriptors[key]
        record_spec = descriptor.get("records")
        tensor_spec = descriptor.get("tensors")
        if not isinstance(record_spec, dict) or not isinstance(tensor_spec, dict):
            raise ValueError("substrate synapse shard descriptor is invalid")
        record_checksum = str(record_spec.get("sha256", ""))
        record_relative = str(record_spec.get("path", ""))
        if record_relative != "blobs/%s.json" % record_checksum:
            raise ValueError("substrate record shard identity is invalid")
        record_path = NeuralSubstrate._safe_store_path(self.root, record_relative)
        if validate and (
            NeuralSubstrate._file_sha256(record_path) != record_checksum
            or record_path.stat().st_size != int(record_spec.get("bytes", -1))
        ):
            raise ValueError("substrate record shard checksum mismatch")
        try:
            payload = json.loads(record_path.read_text("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("substrate synapse record payload is invalid") from error
        records = payload.get("records") if isinstance(payload, dict) else None
        ids = payload.get("ids") if isinstance(payload, dict) else None
        count = int(descriptor.get("count", -1))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"ids", "kind", "records"}
            or payload.get("kind") != "synapses"
            or not isinstance(records, list)
            or not isinstance(ids, list)
            or len(records) != count
            or len(ids) != count
        ):
            raise ValueError("substrate synapse record payload is invalid")
        normalized: List[Dict[str, Any]] = []
        for raw_id, raw_record in zip(ids, records):
            if not isinstance(raw_record, dict):
                raise ValueError("substrate synapse record is invalid")
            record_id = str(raw_id)
            record = dict(raw_record)
            if (
                str(record.get("id", "")) != record_id
                or not isinstance(raw_id, str)
                or NeuralSubstrate._bucket("synapses", record_id) != key[0]
                or not _synapse_id_matches_endpoints(
                    record_id,
                    str(record.get("source_id", "")),
                    str(record.get("target_id", "")),
                )
            ):
                raise ValueError("synapse shard identifiers do not match")
            normalized.append(record)
        normalized_ids = [str(record["id"]) for record in normalized]
        if normalized_ids != sorted(normalized_ids) or len(
            set(normalized_ids)
        ) != len(normalized_ids):
            raise ValueError("synapse shard identifiers are not canonical")

        tensor_checksum = str(tensor_spec.get("sha256", ""))
        tensor_relative = str(tensor_spec.get("path", ""))
        if tensor_relative != "blobs/%s.safetensors" % tensor_checksum:
            raise ValueError("substrate tensor shard identity is invalid")
        tensor_path = NeuralSubstrate._safe_store_path(self.root, tensor_relative)
        if validate and (
            NeuralSubstrate._file_sha256(tensor_path) != tensor_checksum
            or tensor_path.stat().st_size != int(tensor_spec.get("bytes", -1))
        ):
            raise ValueError("substrate tensor shard checksum mismatch")
        tensors = load_tensors(tensor_path, device="cpu")
        tensors["effective_weight"] = _unpack_persisted_synapse_weights(
            tensors, count, self.store_version
        )
        for field in _SYNAPSE_TENSOR_FIELDS:
            value = tensors.get(field)
            if value is None or int(value.numel()) != count:
                raise ValueError("synapse tensor shard is missing " + field)
        for value in tensors["effective_weight"]:
            NeuralSubstrate.exact_effective_weight(value.item())
        return normalized, tensors

    def _read_shard(
        self,
        key: Tuple[str, int],
        *,
        validate: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        if not validate and key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        must_validate = validate or key not in self._verified_shards
        value = self._descriptor_payload(key, validate=must_validate)
        if must_validate:
            self._verified_shards.add(key)
        if not validate:
            self._cache[key] = value
            while len(self._cache) > self.CACHE_SHARDS:
                self._cache.popitem(last=False)
        return value

    @staticmethod
    def _restore_record(
        record: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
        index: int,
    ) -> Dict[str, Any]:
        restored = dict(record)
        restored.update(
            {
                "effective_weight": NeuralSubstrate.exact_effective_weight(
                    tensors["effective_weight"][index].item()
                ),
                "eligibility": float(tensors["eligibility"][index].item()),
                "plasticity": float(tensors["plasticity"][index].item()),
                "uses": int(tensors["uses"][index].item()),
                "stability": float(tensors["stability"][index].item()),
                "last_updated_at": float(
                    tensors["last_updated_at"][index].item()
                ),
            }
        )
        return restored

    def _locate(self, record_id: str) -> Optional[Tuple[Tuple[str, int], int]]:
        cached = self._locations.get(record_id)
        if cached is not None:
            return cached
        bucket = NeuralSubstrate._bucket("synapses", record_id)
        candidates = [
            key
            for minimum, maximum, key in self._ranges.get(bucket, [])
            if minimum <= record_id <= maximum
        ]
        for key in candidates:
            records, _tensors = self._read_shard(key)
            for index, record in enumerate(records):
                if str(record.get("id", "")) == record_id:
                    location = (key, index)
                    self._locations[record_id] = location
                    return location
        return None

    def __len__(self) -> int:
        return self._base_count - len(self._deleted_locations) + len(self._new_ids)

    def __iter__(self) -> Iterator[str]:
        deleted = set(self._deleted_locations)
        for key in self._ordered_keys:
            records, _tensors = self._read_shard(key)
            for record in records:
                record_id = str(record.get("id", ""))
                if record_id not in deleted:
                    yield record_id
        for record_id in self._dirty:
            if record_id in self._new_ids:
                yield record_id

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        if value in self._new_ids:
            return value in self._dirty
        if value in self._deleted_locations:
            return False
        return value in self._dirty_locations or self._locate(value) is not None

    def __getitem__(self, record_id: str) -> Dict[str, Any]:
        if record_id in self._dirty:
            return self._dirty[record_id]
        if record_id in self._deleted_locations:
            raise KeyError(record_id)
        location = self._locate(record_id)
        if location is None:
            raise KeyError(record_id)
        key, index = location
        records, tensors = self._read_shard(key)
        restored = _TrackedSynapseRecord(
            self._restore_record(records[index], tensors, index),
            lambda: self._graph_changed(record_id),
            self._persistence_changed,
        )
        self._dirty[record_id] = restored
        self._dirty_locations[record_id] = location
        self._original_uses[record_id] = self._exact_uses(
            restored.get("uses", 0)
        )
        return restored

    def __setitem__(self, record_id: str, value: Dict[str, Any]) -> None:
        record_id = str(record_id)
        record = dict(value)
        if str(record.get("id", "")) != record_id:
            raise ValueError("synapse mapping key does not match record id")
        NeuralSubstrate.exact_effective_weight(record.get("effective_weight", 0))
        if record_id in self._deleted_locations:
            location = self._deleted_locations.pop(record_id)
            self._dirty_locations[record_id] = location
        elif record_id not in self._dirty_locations:
            location = self._locate(record_id)
            if location is None:
                self._new_ids.add(record_id)
            else:
                self._dirty_locations[record_id] = location
                _records, tensors = self._read_shard(location[0])
                self._original_uses[record_id] = self._exact_uses(
                    int(tensors["uses"][location[1]].item())
                )
        self._exact_uses(record.get("uses", 0))
        self._dirty[record_id] = _TrackedSynapseRecord(
            record, lambda: self._graph_changed(record_id), self._persistence_changed
        )
        self._graph_changed(record_id)

    def __delitem__(self, record_id: str) -> None:
        if record_id in self._new_ids:
            self._new_ids.remove(record_id)
            self._dirty.pop(record_id, None)
            self._graph_changed(record_id)
            return
        location = self._dirty_locations.get(record_id) or self._locate(record_id)
        if location is None or record_id in self._deleted_locations:
            raise KeyError(record_id)
        if record_id not in self._original_uses:
            _records, tensors = self._read_shard(location[0])
            self._original_uses[record_id] = self._exact_uses(
                int(tensors["uses"][location[1]].item())
            )
        self._dirty.pop(record_id, None)
        self._dirty_locations.pop(record_id, None)
        self._deleted_locations[record_id] = location
        self._graph_changed(record_id)

    def _graph_changed(self, record_id: Optional[str] = None) -> None:
        self.graph_revision += 1
        self._persistence_changed()
        observer = getattr(self, "_recall_graph_observer", None)
        if callable(observer):
            observer(record_id, self.graph_revision)

    def _persistence_changed(self) -> None:
        self.persistence_revision += 1

    def iter_effective_edges(self) -> Iterator[Tuple[str, str, int]]:
        """Yield the exact current non-zero graph without paging cold records."""

        for _identifier, source, target, level in self.iter_effective_edge_records():
            yield source, target, level

    def iter_effective_edge_records(self) -> Iterator[Tuple[str, str, str, int]]:
        """Include actual edge IDs for exact incremental execution indexing."""

        replacements_by_shard: Dict[
            Tuple[str, int], Dict[int, Dict[str, Any]]
        ] = defaultdict(dict)
        for record_id, (key, index) in self._dirty_locations.items():
            replacement = self._dirty.get(record_id)
            if replacement is not None:
                replacements_by_shard[key][index] = replacement
        deleted_by_shard: Dict[Tuple[str, int], set[int]] = defaultdict(set)
        for key, index in self._deleted_locations.values():
            deleted_by_shard[key].add(index)
        for key in self._ordered_keys:
            replacements = replacements_by_shard.get(key, {})
            deleted = deleted_by_shard.get(key, set())
            structures, packed = self._forward_by_shard.get(key, ([], b""))
            if not replacements and not deleted:
                for offset, (_index, record_id, source, target) in enumerate(
                    structures
                ):
                    yield record_id, source, target, _unpack_ternary_level(packed, offset)
                continue
            base = {
                index: (
                    record_id,
                    source,
                    target,
                    _unpack_ternary_level(packed, offset),
                )
                for offset, (index, record_id, source, target) in enumerate(
                    structures
                )
            }
            for index in sorted(set(base).union(replacements)):
                if index in deleted:
                    continue
                replacement = replacements.get(index)
                if replacement is not None:
                    weight = NeuralSubstrate.exact_effective_weight(
                        replacement.get("effective_weight", 0)
                    )
                    if weight:
                        yield (
                            str(replacement["id"]),
                            str(replacement["source_id"]),
                            str(replacement["target_id"]),
                            weight,
                        )
                    continue
                record_id, source, target, weight = base[index]
                yield record_id, source, target, weight
        for record_id, record in self._dirty.items():
            if record_id not in self._new_ids:
                continue
            weight = NeuralSubstrate.exact_effective_weight(
                record.get("effective_weight", 0)
            )
            if weight:
                yield record_id, str(record["source_id"]), str(record["target_id"]), weight

    def observed_effective_levels(self) -> set[int]:
        levels = {weight for _source, _target, weight in self.iter_effective_edges()}
        if len(levels) < len(self):
            levels.add(0)
        return levels

    def dynamic_export(self) -> Tuple[List[str], torch.Tensor]:
        """Materialize only IDs and int8 forward values for explicit packing."""

        values: List[Tuple[str, int]] = []
        for key in self._ordered_keys:
            records, tensors = self._read_shard(key)
            for index, record in enumerate(records):
                record_id = str(record["id"])
                if record_id in self._deleted_locations:
                    continue
                replacement = self._dirty.get(record_id)
                weight = NeuralSubstrate.exact_effective_weight(
                    (
                        replacement.get("effective_weight", 0)
                        if replacement is not None
                        else tensors["effective_weight"][index].item()
                    )
                )
                values.append((record_id, weight))
        for record_id in self._new_ids:
            record = self._dirty[record_id]
            values.append(
                (
                    record_id,
                    NeuralSubstrate.exact_effective_weight(
                        record.get("effective_weight", 0)
                    ),
                )
            )
        values.sort(key=lambda item: item[0])
        return (
            [record_id for record_id, _weight in values],
            torch.tensor([weight for _record_id, weight in values], dtype=torch.int8),
        )

    def validate_dirty(self) -> None:
        for record_id, record in self._dirty.items():
            if str(record.get("id", "")) != record_id:
                raise ValueError("synapse mapping key does not match record id")
            NeuralSubstrate.exact_effective_weight(
                record.get("effective_weight", 0)
            )
            self._exact_uses(record.get("uses", 0))

    def _group_records(self, key: Tuple[str, int]) -> List[Tuple[str, Dict[str, Any]]]:
        records, tensors = self._read_shard(key)
        values: List[Tuple[str, Dict[str, Any]]] = []
        for index, raw in enumerate(records):
            record_id = str(raw["id"])
            if record_id in self._deleted_locations:
                continue
            record = self._dirty.get(record_id)
            if record is None:
                record = self._restore_record(raw, tensors, index)
            values.append((record_id, dict(record)))
        return values

    def save_plan(
        self,
        records_per_shard: int,
    ) -> Tuple[
        List[Dict[str, Any]],
        Dict[Tuple[str, int], List[Tuple[str, Dict[str, Any]]]],
    ]:
        if records_per_shard != self.records_per_shard:
            raise ValueError(
                "lazy substrate checkpoints must preserve records_per_shard"
            )
        changed = {
            location[0] for location in self._dirty_locations.values()
        }.union(location[0] for location in self._deleted_locations.values())
        groups = {key: self._group_records(key) for key in changed}
        additions: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for record_id in self._new_ids:
            additions[NeuralSubstrate._bucket("synapses", record_id)].append(
                (record_id, dict(self._dirty[record_id]))
            )
        for bucket, records in additions.items():
            bucket_keys = [key for key in self._ordered_keys if key[0] == bucket]
            part = max((key[1] for key in bucket_keys), default=-1)
            key = (bucket, part) if part >= 0 else (bucket, 0)
            if key in self._descriptors and key not in groups:
                groups[key] = self._group_records(key)
            group = groups.setdefault(key, [])
            for record in sorted(records, key=lambda item: item[0]):
                if len(group) >= records_per_shard:
                    part += 1
                    key = (bucket, part)
                    group = groups.setdefault(key, [])
                group.append(record)
        reused = [
            dict(self._descriptors[key])
            for key in self._ordered_keys
            if key not in groups
        ]
        return reused, {
            key: sorted(group, key=lambda item: item[0])
            for key, group in groups.items()
            if group
        }

    def forward_index_entries(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        changed_groups: Mapping[
            Tuple[str, int], Sequence[Tuple[str, Mapping[str, Any]]]
        ],
        hot_node_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        if hasattr(self._forward_by_shard, "reused_entry"):
            hot_ids = (hot_node_ids if callable(getattr(hot_node_ids, "iter_sorted_ids", None))
                       else {str(value) for value in hot_node_ids if str(value)})
            prior = self._forward_index_manifest or {}
            same_membership = _forward_hot_ids_sha256(
                hot_ids, prior.get("hotNodeIdsChecksumAlgorithm", "sha256-sorted-ids-v1")
            ) == prior.get("hotNodeIdsSha256")
            def bounded_entries():
                for descriptor in descriptors:
                    key = str(descriptor["bucket"]), int(descriptor["part"])
                    changed = changed_groups.get(key)
                    if changed is not None:
                        yield _forward_index_entry(descriptor, changed, hot_ids)
                    elif same_membership:
                        yield self._forward_by_shard.reused_entry(key)
                    else:
                        # Legacy callers without the production complete
                        # endpoint index must faithfully verify/reindex every
                        # old group when assembly membership changes.
                        yield _forward_index_entry(descriptor, self._group_records(key), hot_ids)
            return bounded_entries()
        hot_ids = {str(value) for value in hot_node_ids if str(value)}
        # Index hot locations once.  The previous comprehension lived inside
        # the descriptor loop and therefore rescanned every hot edge for every
        # persisted shard (9,426 x the hot-location set in the live brain).
        # Grouping by shard makes generation publication O(shards + hot edges)
        # while preserving the exact canonical per-shard ordering.
        hot_locations_by_key: Dict[Tuple[str, int], List[List[Any]]] = (
            defaultdict(list)
        )
        for node_id, values in self._hot_locations.items():
            if node_id not in hot_ids:
                continue
            for record_id, location_key, index in values:
                hot_locations_by_key[location_key].append(
                    [node_id, record_id, index]
                )
        for values in hot_locations_by_key.values():
            values.sort()
        entries: List[Dict[str, Any]] = []
        for descriptor in descriptors:
            key = (str(descriptor["bucket"]), int(descriptor["part"]))
            changed = changed_groups.get(key)
            if changed is not None:
                entries.append(
                    _forward_index_entry(descriptor, changed, hot_ids)
                )
                continue
            bounds = self._range_by_key.get(key)
            structures, packed = self._forward_by_shard.get(key, ([], b""))
            record_spec = descriptor.get("records")
            tensor_spec = descriptor.get("tensors")
            if not isinstance(record_spec, Mapping) or not isinstance(
                tensor_spec, Mapping
            ):
                raise ValueError("substrate forward index descriptor is invalid")
            hot_locations = hot_locations_by_key.get(key, [])
            entries.append(
                {
                    "bucket": key[0],
                    "part": key[1],
                    "count": int(descriptor["count"]),
                    "recordsSha256": str(record_spec.get("sha256", "")),
                    "tensorsSha256": str(tensor_spec.get("sha256", "")),
                    "firstId": bounds[0] if bounds else None,
                    "lastId": bounds[1] if bounds else None,
                    "synapticUses": self._uses_by_shard[key],
                    "forwardRecords": [list(value) for value in structures],
                    "packedEffectiveWeights": base64.b64encode(packed).decode(
                        "ascii"
                    ),
                    "hotLocations": hot_locations,
                }
            )
        return entries

    def commit_generation(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        changed_groups: Mapping[
            Tuple[str, int], Sequence[Tuple[str, Mapping[str, Any]]]
        ],
        hot_node_ids: Iterable[str],
        *, forward_manifest: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if forward_manifest is not None and forward_manifest.get("formatVersion") == 4:
            replacement = LazyPersistedSynapses(self.root, descriptors, len(self), self.records_per_shard,
                                                hot_node_ids, forward_manifest, store_version=_SUBSTRATE_STORE_VERSION)
            revision, persistence_revision = self.graph_revision, self.persistence_revision
            observer = getattr(self, "_recall_graph_observer", None)
            self.__dict__.update(replacement.__dict__)
            self.graph_revision, self.persistence_revision = revision, persistence_revision
            if observer is not None:
                self._recall_graph_observer = observer
            return
        previous_forward = self._forward_by_shard
        previous_ranges = self._range_by_key
        previous_uses = self._uses_by_shard
        self._hot_node_ids = {
            str(value) for value in hot_node_ids if str(value)
        }
        next_hot_locations: Dict[
            str, List[Tuple[str, Tuple[str, int], int]]
        ] = defaultdict(list)
        changed_keys = set(changed_groups)
        for node_id, locations in self._hot_locations.items():
            if node_id not in self._hot_node_ids:
                continue
            next_hot_locations[node_id].extend(
                value for value in locations if value[1] not in changed_keys
            )
        self._descriptors = {
            (str(value["bucket"]), int(value["part"])): dict(value)
            for value in descriptors
        }
        self._ordered_keys = sorted(self._descriptors)
        self._ranges = defaultdict(list)
        self._range_by_key = {}
        next_forward: Dict[
            Tuple[str, int],
            Tuple[List[Tuple[int, str, str, str]], bytes],
        ] = {}
        next_uses: Dict[Tuple[str, int], int] = {}
        for key in self._ordered_keys:
            changed = changed_groups.get(key)
            if changed is None:
                prior_structures, prior_packed = previous_forward.get(
                    key, ([], b"")
                )
                next_forward[key] = (list(prior_structures), prior_packed)
                next_uses[key] = previous_uses[key]
                bounds = previous_ranges.get(key)
                identifiers = list(bounds) if bounds is not None else []
            else:
                identifiers = [record_id for record_id, _record in changed]
                next_uses[key] = sum(
                    self._exact_uses(record.get("uses", 0))
                    for _record_id, record in changed
                )
                for index, (record_id, record) in enumerate(changed):
                    source_id = str(record.get("source_id", ""))
                    target_id = str(record.get("target_id", ""))
                    if source_id in self._hot_node_ids:
                        next_hot_locations[source_id].append(
                            (record_id, key, index)
                        )
                    if (
                        target_id in self._hot_node_ids
                        and target_id != source_id
                    ):
                        next_hot_locations[target_id].append(
                            (record_id, key, index)
                        )
                structures: List[Tuple[int, str, str, str]] = []
                levels: List[int] = []
                for index, (record_id, record) in enumerate(changed):
                    weight = NeuralSubstrate.exact_effective_weight(
                        record.get("effective_weight", 0)
                    )
                    if not weight:
                        continue
                    structures.append(
                        (
                            index,
                            record_id,
                            str(record["source_id"]),
                            str(record["target_id"]),
                        )
                    )
                    levels.append(weight)
                next_forward[key] = (
                    structures,
                    _pack_ternary_levels(levels),
                )
            if identifiers:
                self._range_by_key[key] = (identifiers[0], identifiers[-1])
                self._ranges[key[0]].append((identifiers[0], identifiers[-1], key))
        for bucket in self._ranges:
            self._ranges[bucket].sort(key=lambda value: (value[0], value[1], value[2]))
        self._forward_by_shard = next_forward
        self._uses_by_shard = next_uses
        self._hot_locations = next_hot_locations
        self._verified_shards = {
            key for key in self._verified_shards if key in self._descriptors
        }.union(changed_keys)
        self._base_count = len(self)
        self.store_version = _SUBSTRATE_STORE_VERSION
        self._dirty.clear()
        self._original_uses.clear()
        self._dirty_locations.clear()
        self._deleted_locations.clear()
        self._new_ids.clear()
        self._locations.clear()
        self._cache.clear()


# Source compatibility only.  The stable engine imports ``NeuralSubstrate``;
# external beta checkpoints remain intentionally incompatible.
ConceptMemory = NeuralSubstrate
