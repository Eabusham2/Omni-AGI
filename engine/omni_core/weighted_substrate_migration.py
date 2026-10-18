"""Bounded proofs for native memory's additive sign/strength representation.

The immutable substrate generation is authority. SQLite below is disposable
comparison scratch: it holds hashes, never facts or an answering lookup. No
learned vector is reconstructed as a full floating-point model.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from .bounded_tensor_io import BoundedTensorFile
from .packed_vsa_vectors import PackedTernaryVectors
from .persistence import load_tensors


_MAX_BLOB_BYTES = 64 * 1024 * 1024
_SYNAPSE_FIELDS = {"packed_effective_weight", "eligibility", "plasticity", "uses", "stability", "last_updated_at"}


def _generation(root: Path, pointer: Mapping):
    from .vsa import NeuralSubstrate
    identity = pointer.get("activeGeneration")
    if (not isinstance(identity, str) or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
            or pointer.get("generationManifest") != "generations/" + identity + "/manifest.json"):
        raise ValueError("weighted migration generation identity is invalid")
    payload = NeuralSubstrate._safe_store_path(root, pointer["generationManifest"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != pointer.get("generationManifestSha256"):
        raise ValueError("weighted migration generation manifest checksum mismatch")
    value = json.loads(payload)
    if (not isinstance(value, dict) or value.get("format") != "omni-substrate-shards"
            or pointer.get("format") != value.get("format")
            or hashlib.sha256(NeuralSubstrate._canonical_json({key: item for key, item in value.items() if key != "contentSha256"})).hexdigest() != identity
            or value.get("contentSha256") != identity or pointer.get("contentSha256") != identity
            or pointer.get("formatVersion") != value.get("formatVersion")):
        raise ValueError("weighted migration generation content checksum mismatch")
    if (type(value.get("formatVersion")) is not int or value["formatVersion"] != 3
            or value.get("schema") != NeuralSubstrate.SCHEMA or value.get("counts") != pointer.get("counts")
            or pointer.get("shardCount") != len(value.get("shards", []))
            or not isinstance(value.get("counts"), dict) or set(value["counts"]) != {"neurons", "assemblies", "synapses"}
            or any(type(count) is not int or count < 0 for count in value["counts"].values())):
        raise ValueError("weighted migration needs an exact native substrate generation")
    return value


def _blob(root: Path, specification: Mapping, suffix: str, *, read: bool, identity_sink=None):
    from .vsa import NeuralSubstrate
    if not isinstance(specification, Mapping):
        raise ValueError("weighted migration shard is missing a blob")
    size, digest = specification.get("bytes"), specification.get("sha256")
    if (type(size) is not int or not 0 <= size <= _MAX_BLOB_BYTES
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or specification.get("path") != "blobs/" + digest + suffix):
        raise ValueError("weighted migration blob reference is invalid or exceeds its bounded window")
    path = NeuralSubstrate._safe_store_path(root, specification["path"])
    if (root / specification["path"]).is_symlink() or not path.is_file():
        raise ValueError("weighted migration blob is not a regular local file")
    before = path.stat()
    if before.st_size != size:
        raise ValueError("weighted migration blob size differs")
    payload = path.read_bytes() if read else None
    observed = hashlib.sha256(payload).hexdigest() if payload is not None else NeuralSubstrate._file_sha256(path)
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if observed != digest or identity(before) != identity(after):
        raise ValueError("weighted migration blob checksum differs or changed while read")
    if identity_sink is not None:
        identity_sink(path, identity(after))
    return payload if read else path


def missing_weighted_groups(root: Path, pointer: Mapping) -> dict[str, set[tuple[str, int]]]:
    """Identify additive representation work from the committed generation.

    Do not decide from a flag changed by a failed save. New writer declarations
    avoid re-reading every already-weighted shard during ordinary cold load;
    the loaders still validate each actual payload when it is consumed.
    """
    root = Path(root).resolve()
    generation = _generation(root, pointer)
    marker = generation.get("nativeWeightedRepresentation")
    if marker is not None and (type(marker) is not int or marker != 2):
        raise ValueError("native weighted memory representation is unsupported")
    result = {"neurons": set(), "synapses": set()}
    if marker == 2:
        return result
    for descriptor in generation["shards"]:
        kind = descriptor["kind"]
        key = descriptor["bucket"], descriptor["part"]
        if kind == "neurons":
            payload = json.loads(_blob(root, descriptor["records"], ".json", read=True))
            version = payload.get("packedVectorState", {}).get("formatVersion")
            if type(version) is not int or version not in (1, 2):
                raise ValueError("native vector representation is unsupported")
            if version == 1:
                result[kind].add(key)
        elif kind == "synapses":
            path = _blob(root, descriptor["tensors"], ".safetensors", read=False)
            if "packed_connection_strength" not in BoundedTensorFile(path).specs:
                result[kind].add(key)
    return result


def _framed_hash(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    return digest.hexdigest()


def _records(root: Path, generation: Mapping, *, require_weighted: bool, identity_sink=None):
    """Yield one bounded, authenticated row digest at a time."""
    from .vsa import NeuralSubstrate, _unpack_persisted_synapse_weights, _synapse_id_matches_endpoints
    dimensions = generation["dimensions"]
    for descriptor in generation["shards"]:
        kind, count = descriptor.get("kind"), descriptor.get("count")
        if kind not in {"neurons", "assemblies", "synapses"} or type(count) is not int or not 1 <= count <= generation["recordsPerShard"]:
            raise ValueError("weighted migration shard count/kind is invalid")
        payload = json.loads(_blob(root, descriptor["records"], ".json", read=True, identity_sink=identity_sink))
        if not isinstance(payload, dict):
            raise ValueError("weighted migration record payload is invalid")
        records, ids = payload.get("records"), payload.get("ids")
        if (not isinstance(records, list) or not isinstance(ids, list) or len(records) != count or len(ids) != count
                or any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != count
                or ids != [record.get("id") if isinstance(record, dict) else None for record in records]
                or payload.get("kind") != kind
                or any(NeuralSubstrate._bucket(kind, identifier) != descriptor.get("bucket") for identifier in ids)):
            raise ValueError("weighted migration shard record identities differ")
        tensors = {}
        if descriptor.get("tensors") is not None:
            path = _blob(root, descriptor["tensors"], ".safetensors", read=False, identity_sink=identity_sink)
            identity = path.stat()
            tensors = load_tensors(path)
            if (identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) != (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns):
                raise ValueError("weighted migration tensors changed while loaded")
        if kind == "neurons":
            metadata = payload.get("packedVectorState")
            if (payload.get("vectorIds") != ids or not isinstance(metadata, Mapping)
                    or metadata.get("ids") != ids or metadata.get("dimensions") != dimensions
                    or metadata.get("seed") != generation["seed"]
                    or set(tensors) != PackedTernaryVectors.tensor_names(metadata)
                    or require_weighted and metadata.get("formatVersion") != 2):
                raise ValueError("weighted migration neuron representation is incomplete")
            vectors = PackedTernaryVectors.from_state(metadata, tensors)
            for identifier, record in zip(ids, records):
                # Defaults in a v1 source are expanded only as packed zero
                # control bytes (unit strength); not learned floating values.
                yield kind, identifier, _framed_hash(NeuralSubstrate._canonical_json(record),
                    vectors.row_payload(identifier), struct.pack("<Q", vectors.update_count(identifier)),
                    struct.pack("<d", vectors.zero_deadband))
        elif kind == "assemblies":
            if tensors or descriptor.get("tensors") is not None or payload.get("vectorIds") != ids or payload.get("vectorStorage") != "shared-neuron-packed":
                raise ValueError("weighted migration assembly does not alias the neural row")
            for identifier, record in zip(ids, records):
                yield kind, identifier, _framed_hash(NeuralSubstrate._canonical_json(record))
        else:
            if set(tensors) - {"packed_connection_strength"} != _SYNAPSE_FIELDS or require_weighted and "packed_connection_strength" not in tensors:
                raise ValueError("weighted migration synapse representation is incomplete")
            effective = _unpack_persisted_synapse_weights(tensors, count, 3)
            for field in ("eligibility", "plasticity", "stability", "last_updated_at", "uses"):
                value = tensors[field]
                if tuple(value.shape) != (count,) or value.dtype != (torch.int64 if field == "uses" else torch.float64) or not bool(torch.isfinite(value).all()):
                    raise ValueError("weighted migration synapse state is invalid")
            for position, (identifier, record) in enumerate(zip(ids, records)):
                if not _synapse_id_matches_endpoints(identifier, record.get("source_id"), record.get("target_id")):
                    raise ValueError("weighted migration synapse endpoints differ")
                if type(record.get("parallel_count", 0)) is not int or record.get("parallel_count", 0) < 0:
                    raise ValueError("weighted migration repeated-contact count is invalid")
                parts = [NeuralSubstrate._canonical_json(record), bytes([int(effective[position]) + 1, int(tensors["connection_strength"][position])])]
                parts.extend(tensors[field][position:position + 1].numpy().tobytes() for field in
                    ("eligibility", "plasticity", "uses", "stability", "last_updated_at"))
                yield kind, identifier, _framed_hash(*parts)


def verify_saved_weighted_generation(root: Path, target_pointer: Mapping, *, source_pointer: Mapping,
                                     memory_reserve=None, disk_reserve=None) -> dict:
    """Prove exact native learning preservation through additive conversion."""
    root = Path(root).resolve()
    source = _generation(root, source_pointer)
    target = _generation(root, target_pointer)
    ignored = {"shards", "contentSha256", "recordsPerShard", "nativeWeightedRepresentation"}
    if ({key: value for key, value in source.items() if key not in ignored}
            != {key: value for key, value in target.items() if key not in ignored}):
        raise ValueError("weighted migration changed original substrate metadata/counts")
    if memory_reserve is not None:
        for generation in (source, target):
            peak = max((sum(int(shard.get(field, {}).get("bytes", 0)) for field in ("records", "tensors"))
                        for shard in generation["shards"]), default=0)
            if memory_reserve(8 * peak + 65536) is False:
                raise ValueError("weighted migration awaits bounded verification memory")
    scratch_dir = root / "staging"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if scratch_dir.is_symlink():
        raise ValueError("weighted migration scratch directory must be local")
    if disk_reserve is not None and disk_reserve(65536, "weighted migration comparison scratch") is False:
        raise ValueError("weighted migration awaits verification disk")
    descriptor, temporary = tempfile.mkstemp(prefix="weighted-proof-", suffix=".sqlite3", dir=scratch_dir)
    os.close(descriptor)
    path = Path(temporary)
    connection = sqlite3.connect(path)
    counts = {"neurons": 0, "assemblies": 0, "synapses": 0}
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-1024")
        connection.execute("CREATE TABLE original(kind TEXT, id TEXT, digest TEXT, seen INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(kind,id)) WITHOUT ROWID")
        connection.execute("CREATE TABLE artifacts(path TEXT PRIMARY KEY, identity TEXT NOT NULL) WITHOUT ROWID")

        def remember_identity(path, identity):
            payload = json.dumps(identity, separators=(",", ":"))
            prior = connection.execute("SELECT identity FROM artifacts WHERE path=?", (str(path),)).fetchone()
            if prior is not None and prior[0] != payload:
                raise ValueError("weighted migration artifact changed between reads")
            connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (str(path), payload))

        for kind, identifier, digest in _records(root, source, require_weighted=False, identity_sink=remember_identity):
            if disk_reserve is not None and counts[kind] % 256 == 0 and disk_reserve(256 * 1024, "weighted migration row proof") is False:
                raise ValueError("weighted migration awaits comparison disk")
            connection.execute("INSERT INTO original(kind,id,digest) VALUES (?,?,?)", (kind, identifier, digest))
            counts[kind] += 1
        if counts != source["counts"]:
            raise ValueError("weighted migration did not visit all original records")
        connection.commit()
        observed = {key: 0 for key in counts}
        for kind, identifier, digest in _records(root, target, require_weighted=True, identity_sink=remember_identity):
            changed = connection.execute("UPDATE original SET seen=1 WHERE kind=? AND id=? AND digest=? AND seen=0", (kind, identifier, digest))
            if changed.rowcount != 1:
                raise ValueError("weighted migration changed original learned records or added nonunit state")
            observed[kind] += 1
        if observed != counts or connection.execute("SELECT 1 FROM original WHERE seen=0 LIMIT 1").fetchone() is not None:
            raise ValueError("weighted migration omitted original records")
        # The immutable source and target pointers still name precisely the
        # generation whose rows were compared; this never publishes them.
        if _generation(root, source_pointer) != source or _generation(root, target_pointer) != target:
            raise ValueError("weighted migration generation changed during verification")
        for name, expected in connection.execute("SELECT path,identity FROM artifacts"):
            artifact_path = Path(name)
            details = artifact_path.stat()
            observed = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)
            if artifact_path.is_symlink() or json.dumps(observed, separators=(",", ":")) != expected:
                raise ValueError("weighted migration artifact changed before proof completed")
    finally:
        connection.close()
        path.unlink(missing_ok=True)
    return {"verified": True, "unitDefaultsVerified": True, "sourceRecordsAndSignsPreserved": True,
            "sourceGeneration": source_pointer["activeGeneration"], "targetGeneration": target_pointer["activeGeneration"],
            "preservedNeuronCount": counts["neurons"], "preservedSynapseCount": counts["synapses"],
            "preservedAssemblyCount": counts["assemblies"], "sourceCursorAdvanced": False,
            "comparison": "bounded-authenticated-record-hashes", "newContactsAdded": False}
