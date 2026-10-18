"""Rebind owned, byte-identical dataset sources without replaying old records."""
import copy
import hashlib
from pathlib import Path

from .connected_runtime import _source_stat, activate_pending_ingestion


def rebind_retained_source(brain, *, path, original_source_identity, content_hash,
                          source_name, transaction_id="", parent_id="", source_identity_aliases=(), cancelled=None):
    if not brain._sha256_identifier(original_source_identity) or not brain._sha256_identifier(content_hash):
        raise ValueError("retained source needs its original identity and exact content hash")
    if not isinstance(source_identity_aliases, (list, tuple)) or any(not brain._sha256_identifier(key) for key in source_identity_aliases):
        raise ValueError("retained source identity lineage is invalid")
    identities = {original_source_identity, *source_identity_aliases}
    target = Path(path).resolve()
    owner = Path(brain.storage_path).resolve()
    if not target.is_file() or owner not in target.parents:
        raise ValueError("relocated learning source must be an owned regular file")
    name_hash = hashlib.sha256(source_name.encode("utf-8")).hexdigest()
    destination_key = hashlib.sha256(("file\0" + str(target)).encode()).hexdigest()
    candidates = [item for item in getattr(brain, "connected_pending_ingestions", [])
        if item.get("status") != "complete" and (not parent_id or item.get("parentId") == parent_id)
        and item.get("checkpoint", {}).get("sourceIdentity") in identities
        and item.get("checkpoint", {}).get("contentHash") == content_hash
        and item.get("checkpoint", {}).get("sourceNameHash") == name_hash
        and (not transaction_id or item.get("checkpoint", {}).get("transactionId") == transaction_id)]
    if len(candidates) > 1:
        raise ValueError("retained source belongs to multiple parent cursors; choose its exact lineage")
    if candidates:
        return {**activate_pending_ingestion(brain, candidates[0]["id"], target, source_name, cancelled=cancelled), "recordsReplayed": 0}
    old_items = [(key, item) for key, item in brain.ingestion_checkpoints.items() if key in identities]
    if len(old_items) > 1: raise ValueError("retained source has more than one active lineage cursor")
    old_key, old = old_items[0] if old_items else (None, None)
    current = brain.ingestion_checkpoints.get(destination_key)
    selected = old or current
    if selected is not None:
        key = old_key if old is not None else destination_key
        selected = brain._validated_ingestion_checkpoints({key: selected})[key]
    brain._validated_completed_ingestions(brain.completed_ingestions)
    completed = [item for item in brain.completed_ingestions
        if item.get("sourceIdentity") in identities.union({destination_key})
        and item.get("contentHash") == content_hash and item.get("sourceNameHash") == name_hash
        and (not transaction_id or item.get("transactionId") == transaction_id)]
    if selected is None and not completed:
        # The source may not yet have received its first native record. This
        # is not reported as a resumed/committed learning transaction.
        return {"state": "not-started", "rebound": False, "recordsReplayed": 0}
    if selected is not None and (selected.get("contentHash") != content_hash or selected.get("sourceNameHash") != name_hash
            or transaction_id and selected.get("transactionId") != transaction_id
            or selected.get("neuralStateChecksum") != brain.parameter_checksum()):
        raise ValueError("retained source cursor does not match the current neural generation")
    kind = selected["resolvedKind"] if selected is not None else "text"
    before = _source_stat(target, kind)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while True:
            if cancelled is not None and cancelled(): raise RuntimeError("retained source verification paused")
            block = stream.read(1024*1024)
            if not block: break
            digest.update(block)
    if digest.hexdigest() != content_hash or before != _source_stat(target, kind):
        raise ValueError("retained source bytes changed during relocation")
    if selected is not None and selected.get("sourceBytes") != before["size"]:
        raise ValueError("retained source byte count changed")
    if current is not None and old is None and current.get("sourceSnapshot") == before:
        return {"state": "active", "rebound": False, "idempotent": True, "recordsReplayed": 0,
            "committedRecords": current["committedRecords"]}
    prior_cursors, prior_completed, prior_joint = copy.deepcopy(brain.ingestion_checkpoints), copy.deepcopy(brain.completed_ingestions), copy.deepcopy(brain.ingestion_joint_generation)
    try:
        if selected is not None:
            if len(brain.ingestion_checkpoints) != 1:
                raise ValueError("another active source must settle before relocation")
            checkpoint = copy.deepcopy(selected)
            checkpoint.update(sourceIdentity=destination_key, sourceSnapshot=before,
                neuralStateChecksum=brain.parameter_checksum())
            brain.ingestion_checkpoints = {destination_key: checkpoint}
            brain.ingestion_joint_generation = None
        for receipt in brain.completed_ingestions:
            if receipt in completed: receipt["sourceIdentity"] = destination_key
        brain.save()
    except BaseException:
        brain.ingestion_checkpoints, brain.completed_ingestions, brain.ingestion_joint_generation = prior_cursors, prior_completed, prior_joint
        raise
    return {"state": "active" if selected is not None else "complete", "rebound": True,
        "committedRecords": selected["committedRecords"] if selected is not None else None, "recordsReplayed": 0}
