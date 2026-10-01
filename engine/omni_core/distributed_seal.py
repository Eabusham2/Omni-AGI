"""Independent source-free binding of distributed progress in brain.json.

This is a transaction consistency seal, not a signature or an authentication
claim against someone rewriting all authoritative files.
"""

import hashlib
import json

from .record_window_wave import validate_record_window_cursor
from .text_spool import bounded_json_sha256
from .sparse_router_state import canonical_module_walk


FIELDS = {"format", "formatVersion", "manifestSha256", "topologySha256", "trainingPolicySha256",
    "rankCursorsSha256", "rankCursors", "worldSize", "recordsPerEpoch", "epochsRequested",
    "committedRecordStop", "globalOptimizerSteps", "contentSha256"}
RANK_FIELDS = {"rank", "worldSize", "epoch", "nextGlobalOrdinal", "ownedRecordsCompleted",
    "optimizerStepsCompleted", "manifestSha256"}


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_distributed_training_seal(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != FIELDS or value["format"] != "omni-distributed-native-cursor-seal" or type(value["formatVersion"]) is not int or value["formatVersion"] != 1:
        raise ValueError("distributed native cursor seal schema is invalid")
    for name in ("worldSize", "recordsPerEpoch", "epochsRequested", "committedRecordStop", "globalOptimizerSteps"):
        if type(value[name]) is not int or not 0 <= value[name] <= (1 << 63) - 1:
            raise ValueError("distributed native cursor seal counter is invalid")
    if value["worldSize"] < 1 or value["epochsRequested"] < 1:
        raise ValueError("distributed native cursor seal request is invalid")
    for name in ("manifestSha256", "topologySha256", "trainingPolicySha256", "rankCursorsSha256", "contentSha256"):
        if not _hash(value[name]):
            raise ValueError("distributed native cursor seal hash is invalid")
    cursors = value["rankCursors"]
    if not isinstance(cursors, list) or len(cursors) != value["worldSize"]:
        raise ValueError("distributed native cursor seal rank coverage is incomplete")
    positions, clean_cursors = [], []
    cardinality, world_size = value["recordsPerEpoch"], value["worldSize"]
    for rank, cursor in enumerate(cursors):
        if not isinstance(cursor, dict) or set(cursor) not in (RANK_FIELDS, RANK_FIELDS | {"recordWindow"}):
            raise ValueError("distributed sealed rank cursor schema is invalid")
        if any(type(cursor[name]) is not int or not 0 <= cursor[name] <= (1 << 63) - 1 for name in RANK_FIELDS - {"manifestSha256"}):
            raise ValueError("distributed sealed rank cursor counter is invalid")
        if cursor["rank"] != rank or cursor["worldSize"] != world_size or cursor["manifestSha256"] != value["manifestSha256"] or cursor["epoch"] > value["epochsRequested"] or cursor["nextGlobalOrdinal"] > cardinality or cursor["optimizerStepsCompleted"] != value["globalOptimizerSteps"]:
            raise ValueError("distributed sealed rank cursor identity/order is invalid")
        owned_per_epoch = 0 if rank >= cardinality else 1 + (cardinality - 1 - rank) // world_size
        ordinal = cursor["nextGlobalOrdinal"]
        owned_prefix = 0 if rank >= ordinal else 1 + (ordinal - 1 - rank) // world_size
        if cursor["ownedRecordsCompleted"] != cursor["epoch"] * owned_per_epoch + owned_prefix:
            raise ValueError("distributed sealed rank coverage is not its exact ordinal prefix")
        clean = dict(cursor)
        if "recordWindow" in cursor:
            window = validate_record_window_cursor(cursor["recordWindow"])
            if window["phase"] == "complete" or not ordinal <= window["ordinal"] < min(cardinality, ordinal + world_size) or window["ordinal"] % world_size != rank:
                raise ValueError("distributed sealed record window does not belong to its rank prefix")
            clean["recordWindow"] = window
        if cursor["epoch"] == value["epochsRequested"] and (ordinal or "recordWindow" in cursor):
            raise ValueError("exhausted distributed seal retains an active record")
        positions.append(cursor["epoch"] * cardinality + ordinal)
        clean_cursors.append(clean)
    if min(positions) != value["committedRecordStop"] or len({cursor["epoch"] for cursor in cursors}) != 1:
        raise ValueError("distributed seal skips a canonical record/epoch prefix")
    if bounded_json_sha256(clean_cursors) != value["rankCursorsSha256"]:
        raise ValueError("distributed seal rank cursor hash is invalid")
    body = {key: item for key, item in value.items() if key != "contentSha256"}
    if bounded_json_sha256(body) != value["contentSha256"]:
        raise ValueError("distributed native cursor seal checksum is invalid")
    return {**value, "rankCursors": clean_cursors}


def make_distributed_training_seal(*, manifest_sha256, topology_sha256, training_policy_sha256,
    record_count, epochs, cursors, committed_record_stop, global_steps):
    rank_cursors = [cursor.to_dict() if hasattr(cursor, "to_dict") else dict(cursor) for cursor in cursors]
    body = {"format": "omni-distributed-native-cursor-seal", "formatVersion": 1,
        "manifestSha256": manifest_sha256, "topologySha256": topology_sha256,
        "trainingPolicySha256": training_policy_sha256, "rankCursorsSha256": bounded_json_sha256(rank_cursors),
        "rankCursors": rank_cursors, "worldSize": len(rank_cursors), "recordsPerEpoch": int(record_count),
        "epochsRequested": int(epochs), "committedRecordStop": int(committed_record_stop),
        "globalOptimizerSteps": int(global_steps)}
    return validate_distributed_training_seal({**body, "contentSha256": bounded_json_sha256(body)})


def native_topology_sha256(brain):
    """Geometry/scales of every native module plus fast substrate cardinality."""
    digest = hashlib.sha256(b"[")
    rows = 0
    def emit(row):
        nonlocal rows
        if rows:
            digest.update(b",")
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        rows += 1
    roots = ("decoder", "memory_bridge", "idea_adapter", "router", "liquid", "modalities")
    for root_name in roots:
        root = getattr(brain, root_name)
        for name, module in canonical_module_walk(root, resource_policy=getattr(brain, "resource_policy", None)):
            row = {"name": root_name + ("." + name if name else ""), "type": type(module).__name__}
            for role in ("_parameters", "_buffers"):
                row[role] = [[key, list(tensor.shape), str(tensor.dtype)] for key, tensor in sorted(getattr(module, role).items())
                    if tensor is not None and (role != "_buffers" or key not in module._non_persistent_buffers_set)]
            if hasattr(module, "ternary_weight_shape"):
                row["logicalWeightShape"] = list(module.ternary_weight_shape)
                row["logicalBiasShape"] = list(module.ternary_bias_shape)
                scale = getattr(module, "_packed_forward_scale", None)
                row["packedScale"] = float(scale.detach().cpu().item()) if scale is not None else None
            emit(row)
    emit({"substrate": {"neurons": len(brain.memory.neurons),
        "assemblies": len(brain.memory.assemblies), "synapses": len(brain.memory.synapses)}})
    digest.update(b"]")
    return digest.hexdigest()
