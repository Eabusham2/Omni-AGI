"""Adopt complete native parent activity into one connected identity.

All input is an already isolated, checksummed parent snapshot. Historical
text remains the conversation ledger; only each parent's existing temporary
token window is admitted as context. Source replay cursors keep their exact
positions. No source is trained, answered, or summarized during a merge.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .bounded_tensor_io import BoundedTensorFile
from .conversation_ledger import NeuralConversationLedger, ZERO_HASH
from .memory_lifecycle import OrganicMemoryLifecycle
from .offload import DurableReplayBuffer, PagedWorkingMemory, _tensor_bytes, _tensor_sha256
from .online_replay import validate_replay_cursor
from .persistence import read_json


FORMAT = "omni-connected-runtime-adoption"
VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _check(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("full merge stopped at a runtime adoption boundary")


@contextmanager
def _readonly(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("parent runtime database is not a regular snapshot")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-1024")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("parent runtime database failed integrity verification")
        yield connection
    finally:
        connection.close()


def _embed(value: torch.Tensor, width: int, offset: int, parent_width: int) -> torch.Tensor:
    value = value.detach().cpu().reshape(-1)
    if value.numel() != parent_width or not bool(torch.isfinite(value).all()):
        raise ValueError("parent activity is not a finite vector of its declared width")
    output = torch.zeros(width, dtype=value.dtype)
    output[offset:offset + parent_width].copy_(value)
    return output


def _conversation_rows(path: Path, parent_id: str, saved: Mapping[str, Any], cancelled=None):
    """Authenticate the entire chain, including committed standalone actions."""
    with _readonly(path) as source:
        owner = source.execute("SELECT value FROM meta WHERE key='brain_id'").fetchone()
        if owner != (parent_id,):
            raise ValueError("parent conversation belongs to another identity")
        previous, sequence = ZERO_HASH, 0
        head = int(saved.get("headSequence", -1))
        head_seen = head == 0 and saved.get("headSha256") == ZERO_HASH
        for row in source.execute("SELECT sequence,entry_key,kind,created_at,attention_epoch,payload_json,payload_sha256,previous_sha256,row_sha256 FROM entries ORDER BY sequence"):
            _check(cancelled)
            record = dict(zip(("sequence", "entry_key", "kind", "created_at", "attention_epoch", "payload_json", "payload_sha256", "previous_sha256", "row_sha256"), row))
            sequence += 1
            if (record["sequence"] != sequence or record["previous_sha256"] != previous
                    or hashlib.sha256(record["payload_json"].encode("utf-8")).hexdigest() != record["payload_sha256"]
                    or NeuralConversationLedger._row_sha(record) != record["row_sha256"]):
                raise ValueError("parent conversation hash chain is invalid")
            previous = record["row_sha256"]
            if sequence == head:
                head_seen = previous == saved.get("headSha256")
            value = json.loads(record["payload_json"])
            if not isinstance(value, dict):
                raise ValueError("parent conversation payload is not an object")
            yield record, value
        if not head_seen:
            raise ValueError("parent conversation is not bound to its committed checkpoint")


def _message_id(index: sqlite3.Connection, parent: str, old: str) -> str:
    row = index.execute("SELECT target_id FROM conversation_sources WHERE parent_id=? AND kind='message' AND original_id=?", (parent, str(old))).fetchone()
    if row is None:
        raise ValueError("pending parent learning references a missing message")
    return str(row[0])


def _adopt_conversations(brain, snapshots, metadata, index, cancelled):
    index.executescript("""
      CREATE TABLE IF NOT EXISTS conversation_sources(
        parent_id TEXT NOT NULL, sequence INTEGER NOT NULL, kind TEXT NOT NULL,
        original_id TEXT NOT NULL, target_id TEXT NOT NULL, source_payload_sha256 TEXT NOT NULL,
        source_row_sha256 TEXT NOT NULL, PRIMARY KEY(parent_id,sequence));
      CREATE INDEX IF NOT EXISTS merge_message_ids ON conversation_sources(parent_id,kind,original_id);
      CREATE TABLE IF NOT EXISTS conversation_payloads(
        target_id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL, source_payload_sha256 TEXT NOT NULL);
    """)
    for parent in snapshots:
        saved = metadata[parent.parent_id]
        head = saved.get("conversation")
        if not isinstance(head, Mapping):
            raise ValueError("complete native merge requires each committed conversation head")
        path = brain.storage_path / parent.engine_relative_path / "conversation.sqlite3"
        with index:
            for row, payload in _conversation_rows(path, parent.parent_id, head, cancelled):
                # Exact identical content/identity may share an entry. A reused
                # source ID with different text or provenance does not collide.
                target = _digest(["merged-conversation-v1", row["kind"], row["payload_sha256"]])
                index.execute("INSERT OR IGNORE INTO conversation_payloads VALUES(?,?,?,?,?)", (target, row["kind"], row["created_at"], row["payload_json"], row["payload_sha256"]))
                index.execute("INSERT OR IGNORE INTO conversation_sources VALUES(?,?,?,?,?,?,?)", (parent.parent_id, row["sequence"], row["kind"], str(payload.get("id", "")), target, row["payload_sha256"], row["row_sha256"]))
    for target, kind, _created, raw, source_sha in index.execute("SELECT target_id,kind,created_at,payload_json,source_payload_sha256 FROM conversation_payloads ORDER BY created_at,target_id"):
        _check(cancelled)
        payload = json.loads(raw)
        original_id = payload.get("id")
        payload["id"] = target
        # Keep text and timestamps byte-for-byte; source identity and exact
        # original payload are available in the append-only provenance index.
        if "brainId" in payload:
            payload["brainId"] = brain.brain_id
        if "brain_id" in payload:
            payload["brain_id"] = brain.brain_id
        payload["mergeOriginal"] = {"id": original_id, "payloadSha256": source_sha}
        brain.conversation.append(kind, (payload,))
    return int(index.execute("SELECT COUNT(*) FROM conversation_payloads").fetchone()[0])


def _adopt_replay(brain, path, region_id, offset, parent_width, cancelled, *, after_sequence=0):
    if not path.is_file():
        raise ValueError("complete native merge is missing parent latent replay")
    count = 0
    with _readonly(path) as source, brain.replay._connect() as target:
        target.execute("CREATE TABLE IF NOT EXISTS merge_origins(origin_sha256 TEXT PRIMARY KEY, sequence INTEGER NOT NULL UNIQUE)")
        for row in source.execute("SELECT sequence,created_at,sha256,dtype,shape_json,payload FROM replay WHERE sequence>? ORDER BY sequence", (after_sequence,)):
            _check(cancelled)
            value = DurableReplayBuffer._decode((row[0], row[2], row[3], row[4], row[5]))
            origin = _digest([region_id, row[0], row[1], row[2], row[3], row[4]])
            value = _embed(value, brain.config.idea_dim, offset, parent_width)
            payload = _tensor_bytes(value)
            brain.resource_policy.require_disk(max(65536, len(payload) * 3), "merged latent replay")
            target.execute("BEGIN IMMEDIATE")
            try:
                if target.execute("SELECT 1 FROM merge_origins WHERE origin_sha256=?", (origin,)).fetchone() is None:
                    cursor = target.execute("INSERT INTO replay(created_at,sha256,dtype,shape_json,payload) VALUES(?,?,?,?,?)", (row[1], _tensor_sha256(value), str(value.dtype), _canonical(list(value.shape)), sqlite3.Binary(payload)))
                    target.execute("INSERT INTO merge_origins VALUES(?,?)", (origin, cursor.lastrowid))
                target.execute("COMMIT")
            except BaseException:
                if target.in_transaction:
                    target.execute("ROLLBACK")
                raise
            count += 1
    return count


def _adopt_working_page(brain, origin, value, metadata, timestamp, *, source_activity_now=None, source_activity_at=None):
    value = value.detach().cpu().contiguous().reshape(-1)
    brain.resource_policy.require_disk(max(65536, value.numel() * value.element_size() * 3), "merged working activity")
    if brain.paged_working_memory.activity_clock_bound:
        if source_activity_now is None:
            raise ValueError("merged cold working activity requires its original activity coordinate")
        from .working_activity import rebase_page_activity
        metadata = rebase_page_activity(metadata, source_now=source_activity_now,
            source_at=source_activity_at, target_now=brain.paged_working_memory._ranking_now())
    with brain.paged_working_memory._connect() as target:
        # page_id is derived from the exact source address, not from its value:
        # two distinct experiences with the same activation remain distinct.
        target.execute("INSERT OR IGNORE INTO working_pages(page_id,assembly_id,metadata_json,sha256,dtype,shape_json,payload,updated_at) VALUES(?,?,?,?,?,?,?,?)", (origin, str(metadata.get("assemblyId", "")), _canonical(metadata), _tensor_sha256(value), str(value.dtype), _canonical(list(value.shape)), sqlite3.Binary(_tensor_bytes(value)), float(timestamp)))


def _address(region_id: str, value: Any) -> str:
    return _digest(["regional-activity-v1", region_id, str(value)]) if value else ""


def inherited_ingestion_records(parent, saved, validate_checkpoints):
    """Preserve queued/active/completed cursors when a merged parent merges again."""
    checkpoints = validate_checkpoints(saved.get("ingestion_checkpoints"))
    consumed = set()
    for prior in validate_pending_ingestions(saved.get("connected_pending_ingestions")):
        checkpoint = prior["checkpoint"]
        key = prior["sourceIdentity"]
        if prior["status"] == "active":
            key = prior.get("destinationSourceIdentity")
            if key not in checkpoints:
                raise ValueError("inherited active source lost its actual committed cursor")
            checkpoint = checkpoints[key]
            consumed.add(key)
        pending_id = _digest(["merged-inherited-ingestion-v1", parent.region_id, prior["id"], key, checkpoint])
        value = {"id": pending_id, "status": "complete" if prior["status"] == "complete" else "queued",
            "parentId": parent.parent_id, "regionId": parent.region_id,
            "parentEnginePath": parent.engine_relative_path,
            "parentGenerationSha256": parent.generation_sha256,
            "sourceIdentity": key, "checkpoint": copy.deepcopy(checkpoint),
            "mergeLineage": [*copy.deepcopy(prior.get("mergeLineage", [])),
                {"parentId": prior["parentId"], "pendingId": prior["id"],
                 "parentGenerationSha256": prior["parentGenerationSha256"]}]}
        if prior["status"] == "complete":
            value["completionReceipt"] = copy.deepcopy(prior.get("completionReceipt"))
        yield value
    for source_key, checkpoint in checkpoints.items():
        if source_key in consumed:
            continue
        pending_id = _digest(["merged-ingestion-v1", parent.region_id, source_key, checkpoint])
        yield {"id": pending_id, "status": "queued", "parentId": parent.parent_id,
            "regionId": parent.region_id, "parentEnginePath": parent.engine_relative_path,
            "parentGenerationSha256": parent.generation_sha256,
            "sourceIdentity": source_key, "checkpoint": copy.deepcopy(checkpoint)}


def adopt_connected_runtime(brain, snapshots: Sequence[Any], *, cancelled=None) -> dict[str, Any]:
    """Populate common history, replay, working activity and resumable jobs.

    Destination-only SQLite writes are idempotent. The caller saves the new
    brain atomically before marking the host merge complete.
    """
    connected = brain._connected_regions
    if connected is None:
        raise ValueError("runtime adoption requires the actual connected neural graph")
    parents = [{"parentId": p.parent_id, "enginePath": p.engine_relative_path,
                "generationSha256": p.generation_sha256, "metadataSha256": p.metadata_sha256} for p in snapshots]
    source_sha = _digest(parents)
    existing = getattr(brain, "connected_runtime_adoption", None)
    if existing is not None:
        if existing.get("sourceSha256") != source_sha:
            raise ValueError("connected runtime adoption source changed")
        return copy.deepcopy(existing)
    metadata = {p.parent_id: read_json(brain.storage_path / p.engine_relative_path / "brain.json") for p in snapshots}
    configs = {row["regionId"]: row["config"] for row in connected.descriptor["regions"]}
    offsets, idea_offset, vsa_offset = {}, 0, 0
    for row in connected.descriptor["regions"]:
        conf = configs[row["regionId"]]
        offsets[row["regionId"]] = (idea_offset, vsa_offset)
        idea_offset += int(conf["idea_dim"])
        vsa_offset += int(conf["vsa_dim"])
    if idea_offset != brain.config.idea_dim or vsa_offset != brain.config.vsa_dim:
        raise ValueError("merged activity dimensions differ from its neural regions")
    # Combining only the already-live windows is not hidden history retrieval.
    # Preserve all of them or reject the reviewed window before admitting state.
    from .recent_token_activity import RecentTokenActivity, token_hash
    contexts, context_spans, token_owners, seen_regions = [], [], [], set()
    for parent in sorted(snapshots, key=lambda p: (str(metadata[p.parent_id].get("updated_at", "")), p.parent_id)):
        if parent.region_id in seen_regions:
            continue
        seen_regions.add(parent.region_id)
        tokens = metadata[parent.parent_id].get("recent_token_context", [])
        if not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < brain.config.vocab_size for t in tokens):
            raise ValueError("parent temporary token context is invalid")
        activity = RecentTokenActivity.from_state(metadata[parent.parent_id].get("recent_token_activity"),
            tokens, human_id=brain.tokenizer.human_id)
        context_spans.extend({**span, "afterimageId": _address(parent.region_id, span.get("afterimageId"))}
                             for span in activity.spans)
        contexts.extend(tokens)
        prior_owners = (metadata[parent.parent_id].get("current_context") or {}).get("connectedTokenOwners")
        if prior_owners is not None:
            if (not isinstance(prior_owners, list)
                    or any(not isinstance(row, Mapping) or type(row.get("tokenCount")) is not int or row["tokenCount"] <= 0
                        or not isinstance(row.get("regionPath"), list) or not row["regionPath"] for row in prior_owners)
                    or sum(row["tokenCount"] for row in prior_owners) != len(tokens)):
                raise ValueError("parent live token ownership ranges are invalid")
            token_owners.extend({"regionPath": [parent.region_id, *row["regionPath"]], "tokenCount": row["tokenCount"]}
                for row in prior_owners)
        elif tokens:
            token_owners.append({"regionPath": [parent.region_id], "tokenCount": len(tokens)})
    if len(contexts) > brain.config.max_seq_len:
        raise ValueError("review a larger working window: merged live parent contexts need %s tokens" % len(contexts))
    index_path = brain.engine_path / "connected-runtime.sqlite3"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = sqlite3.connect(str(index_path))
    index.execute("PRAGMA synchronous=FULL")
    index.execute("PRAGMA temp_store=FILE")
    index.execute("PRAGMA cache_size=-1024")
    counts = {"historyEntries": 0, "replaySourceRows": 0, "workingSourceRows": 0, "pendingChatJobs": 0, "pendingDatasetJobs": 0}
    target_cycle = max(int((saved.get("memory_lifecycle") or {}).get("cycle", 0)) for saved in metadata.values())
    try:
        counts["historyEntries"] = _adopt_conversations(brain, snapshots, metadata, index, cancelled)
        from .connected_execution import adopt_connected_executions
        brain.connected_executions = adopt_connected_executions(brain, snapshots, index, metadata, cancelled=cancelled)
        counts["pendingExecutions"] = sum(row["status"] in {"queued", "suspended"}
            for row in brain.connected_executions["entries"])
        seen_regions = set()
        pending_chat, pending_sources, sources = {}, {}, {}
        for parent in snapshots:
            _check(cancelled)
            saved = metadata[parent.parent_id]
            engine = brain.storage_path / parent.engine_relative_path
            config = configs[parent.region_id]
            io, vo = offsets[parent.region_id]
            source_activity_now = None
            source_activity_baseline = None
            if brain.paged_working_memory.activity_clock_bound:
                from .working_activity import checkpoint_activity_time, checkpoint_wall_time, checked_activity_baseline
                source_activity_now = checkpoint_activity_time(saved)
                source_activity_baseline = checked_activity_baseline(source_activity_now, checkpoint_wall_time(saved.get("updated_at")))
            for source in saved.get("training_sources", []):
                if not isinstance(source, Mapping):
                    raise ValueError("parent training provenance is invalid")
                sources.setdefault(_digest(source), dict(source))
            for job in saved.get("pending_chat_slow_learning", []):
                if not isinstance(job, Mapping):
                    raise ValueError("parent pending chat learning is invalid")
                value = copy.deepcopy(dict(job))
                value["humanMessageId"] = _message_id(index, parent.parent_id, value.get("humanMessageId", ""))
                value["replayCursor"] = validate_replay_cursor(value.get("replayCursor"))
                value["afterimageId"] = _address(parent.region_id, value.get("afterimageId"))
                original = str(value.get("jobId", ""))
                if len(original) != 64 or any(c not in "0123456789abcdef" for c in original):
                    raise ValueError("parent pending chat job has an invalid identity")
                value["jobId"] = _digest(["merged-chat-replay-v1", parent.region_id, value])
                value["mergeOriginalJobId"] = original
                value["mergeParentId"] = parent.parent_id
                pending_chat.setdefault(value["jobId"], value)
            for pending in inherited_ingestion_records(parent, saved, brain._validated_ingestion_checkpoints):
                pending_sources.setdefault(pending["id"], pending)
            if parent.region_id in seen_regions:
                continue
            seen_regions.add(parent.region_id)
            counts["replaySourceRows"] += _adopt_replay(brain, engine / "state/replay.sqlite3", parent.region_id, io, int(config["idea_dim"]), cancelled)
            plastic = BoundedTensorFile(engine / "plasticity.safetensors")
            def tensor(name):
                if name not in plastic.specs:
                    return None
                value = torch.empty(plastic.specs[name].shape, dtype=plastic.specs[name].dtype)
                plastic.copy_into(name, value, cancelled=cancelled)
                return value
            working = tensor("state.working_memory")
            items = saved.get("workspace_items", [])
            if working is not None and (working.ndim != 2 or len(items) != working.shape[0]):
                raise ValueError("parent working activity and metadata do not align")
            for position, vector in enumerate(working if working is not None else ()):
                item = dict(items[position])
                prior_path = item.get("mergeRegionPath", [item["mergeRegionId"]] if item.get("mergeRegionId") else [])
                item["assemblyId"] = _address(parent.region_id, item.get("assemblyId"))
                item["mergeRegionId"] = parent.region_id
                item["mergeRegionPath"] = [parent.region_id, *prior_path]
                value = _embed(vector, brain.config.idea_dim, io, int(config["idea_dim"]))
                if len(brain.working_memory) < brain.config.working_memory_slots:
                    brain.working_memory.append(value)
                    brain.workspace_items.append(item)
                else:
                    _adopt_working_page(brain, _digest([parent.neural_sha256, "hot", position]), value, item, 0.0,
                        source_activity_now=source_activity_now)
                counts["workingSourceRows"] += 1
            cold = engine / "state/working-memory.sqlite3"
            if cold.is_file():
                with _readonly(cold) as connection:
                    if source_activity_now is not None:
                        from .working_activity import read_activity_baseline, page_activity_at
                        source_activity_baseline = read_activity_baseline(connection) or source_activity_baseline
                    for row in connection.execute("SELECT sequence,page_id,assembly_id,metadata_json,sha256,dtype,shape_json,payload,updated_at FROM working_pages ORDER BY sequence"):
                        _check(cancelled)
                        value, item = PagedWorkingMemory._decode_page_row(row)
                        prior_path = item.get("mergeRegionPath", [item["mergeRegionId"]] if item.get("mergeRegionId") else [])
                        item["assemblyId"] = _address(parent.region_id, item.get("assemblyId"))
                        item["mergeRegionId"] = parent.region_id
                        item["mergeRegionPath"] = [parent.region_id, *prior_path]
                        source_activity_at = page_activity_at(item, row[8], source_activity_baseline) if source_activity_now is not None else None
                        _adopt_working_page(brain, _digest([parent.region_id, "cold", row[1], row[4]]), _embed(value, brain.config.idea_dim, io, int(config["idea_dim"])), item, row[8],
                            source_activity_now=source_activity_now, source_activity_at=source_activity_at)
                        counts["workingSourceRows"] += 1
            afterimages = tensor("memory_lifecycle.afterimage_vectors")
            if afterimages is None:
                afterimages = tensor("memory_lifecycle.scratch_vectors")
            lifecycle = OrganicMemoryLifecycle.from_state(saved.get("memory_lifecycle"), {} if afterimages is None else {"memory_lifecycle.afterimage_vectors": afterimages})
            for item, vector in zip(lifecycle.afterimage_items, lifecycle.afterimage_vectors):
                item = copy.deepcopy(item)
                prior_path = item.get("mergeRegionPath", [item["mergeRegionId"]] if item.get("mergeRegionId") else [])
                for name in ("id", "assemblyId"):
                    item[name] = _address(parent.region_id, item.get(name))
                item["mergeRegionId"] = parent.region_id
                item["mergeRegionPath"] = [parent.region_id, *prior_path]
                # The common cycle counter is not permission to age an older
                # parent's fresh activity by unrelated cycles from a sibling.
                for key in ("enteredCycle", "lastActiveCycle", "lastRescoredCycle", "lastSettledCycle"):
                    if type(item.get(key)) is int:
                        item[key] += target_cycle - lifecycle.cycle
                brain.memory_lifecycle.afterimage_items.append(item)
                brain.memory_lifecycle.afterimage_vectors.append(_embed(vector, brain.config.idea_dim, io, int(config["idea_dim"])))
            for item in lifecycle.active_focus:
                item = copy.deepcopy(item)
                prior_path = item.get("mergeRegionPath", [item["mergeRegionId"]] if item.get("mergeRegionId") else [])
                item["assemblyId"] = _address(parent.region_id, item.get("assemblyId"))
                item["mergeRegionId"] = parent.region_id
                item["mergeRegionPath"] = [parent.region_id, *prior_path]
                brain.memory_lifecycle.active_focus.append(item)
            for name in ("settled_experiences", "settled_rest_cycles", "expired_afterimages", "reinforcement_events"):
                setattr(brain.memory_lifecycle, name, getattr(brain.memory_lifecycle, name) + getattr(lifecycle, name))
            brain.memory_lifecycle.cycle = target_cycle
            for key, value in saved.get("counters", {}).items():
                if type(value) is not int or value < 0:
                    raise ValueError("parent neural counter is invalid")
                brain.counters[key] = int(brain.counters.get(key, 0)) + value
            for key, value in saved.get("modality_training", {}).items():
                if key in brain.modality_training:
                    if type(value) is not int or value < 0:
                        raise ValueError("parent modality training counter is invalid")
                    brain.modality_training[key] += value
        brain.training_sources = list(sources.values())
        brain.pending_chat_slow_learning = list(pending_chat.values())
        brain.connected_pending_ingestions = list(pending_sources.values())
        brain.recent_token_context = contexts
        brain.recent_token_activity = RecentTokenActivity(context_spans)
        brain.recent_token_activity.metadata(contexts)
        brain.current_context = {**brain.current_context, "tokenCount": len(contexts), "tokenHash": token_hash(contexts),
            "mergedParentWindows": True, "connectedTokenOwners": token_owners}
        counts["pendingChatJobs"] = len(pending_chat)
        counts["pendingDatasetJobs"] = sum(row["status"] != "complete" for row in pending_sources.values())
        receipt = {"format": FORMAT, "formatVersion": VERSION, "sourceSha256": source_sha,
            "parents": parents, "counts": counts, "temporaryTokens": len(contexts),
            "historyInjectedAsPrompt": False, "nativeRegionsActive": True}
        receipt["sha256"] = _digest(receipt)
        brain.connected_runtime_adoption = receipt
        return copy.deepcopy(receipt)
    finally:
        index.close()


def validate_runtime_adoption(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("connected runtime adoption is invalid")
    body = dict(value)
    claimed = body.pop("sha256", None)
    if (body.get("format") != FORMAT or body.get("formatVersion") != VERSION
            or _digest(body) != claimed or body.get("historyInjectedAsPrompt") is not False):
        raise ValueError("connected runtime adoption checksum is invalid")
    return {**body, "sha256": claimed}


def validate_pending_ingestions(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("connected pending source queue is invalid")
    from .brain import AdaptiveBrain
    result, seen = [], set()
    for raw in value:
        if not isinstance(raw, Mapping) or raw.get("status") not in {"queued", "active", "complete"}:
            raise ValueError("connected pending source status is invalid")
        item = copy.deepcopy(dict(raw))
        if not AdaptiveBrain._sha256_identifier(item.get("id")) or item["id"] in seen:
            raise ValueError("connected pending source identity is invalid")
        seen.add(item["id"])
        AdaptiveBrain._validated_ingestion_checkpoints({item.get("sourceIdentity"): item.get("checkpoint")})
        result.append(item)
    return result


def refresh_pending_ingestions(brain) -> None:
    """Reflect only actual committed completion; a vanished cursor is not done."""
    for item in getattr(brain, "connected_pending_ingestions", []):
        if item.get("status") != "active":
            continue
        key = item.get("destinationSourceIdentity")
        checkpoint = brain.ingestion_checkpoints.get(key)
        if checkpoint is not None:
            item["destinationCheckpoint"] = copy.deepcopy(checkpoint)
            continue
        source = item["checkpoint"]
        receipt = next((entry for entry in reversed(brain.completed_ingestions)
                        if entry.get("sourceIdentity") == key
                        and entry.get("transactionId") == source["transactionId"]
                        and entry.get("contentHash") == source["contentHash"]), None)
        if receipt is not None:
            item["status"] = "complete"
            item["completionReceipt"] = copy.deepcopy(receipt)


def _source_stat(path: Path, kind: str) -> dict[str, int]:
    value = path.stat()
    result = {"device": int(value.st_dev), "inode": int(value.st_ino),
              "size": int(value.st_size), "mtimeNs": int(value.st_mtime_ns)}
    if kind == "sqlite":
        for label, suffix in (("Wal", "-wal"), ("Shm", "-shm")):
            try:
                current = Path(str(path) + suffix).stat()
            except FileNotFoundError:
                current = None
            result.update({"sqlite%s%s" % (label, field): number for field, number in (
                ("Present", int(current is not None)),
                ("Device", int(current.st_dev) if current else 0),
                ("Inode", int(current.st_ino) if current else 0),
                ("Size", int(current.st_size) if current else 0),
                ("MtimeNs", int(current.st_mtime_ns) if current else 0),
            )})
    return result


def activate_pending_ingestion(brain, pending_id: str, source_path: Path,
                               source_name: str, *, cancelled=None) -> dict[str, Any]:
    """Bind one retained parent cursor to the new atomic neural generation.

    This does not train a record or guess an old filename. The host then
    resumes its preserved dataset job with the original hash/policy/epoch.
    Multiple parents remain queued until this transaction actually completes.
    """
    queue = validate_pending_ingestions(getattr(brain, "connected_pending_ingestions", []))
    selected = next((item for item in queue if item["id"] == pending_id), None)
    if selected is None:
        raise ValueError("pending merged source is unknown")
    if selected["status"] == "complete":
        return {"id": pending_id, "state": "complete", "idempotent": True}
    was_active = selected["status"] == "active"
    old_key = selected.get("destinationSourceIdentity") if was_active else None
    if was_active:
        active = brain.ingestion_checkpoints.get(old_key)
        if active is None or set(brain.ingestion_checkpoints) != {old_key}:
            raise ValueError("active merged source has no committed cursor")
        checkpoint = copy.deepcopy(active)
        if checkpoint["neuralStateChecksum"] != brain.parameter_checksum():
            raise ValueError("active merged cursor is not the current committed neural state")
    else:
        checkpoint = copy.deepcopy(selected["checkpoint"])
    if ((not was_active and brain.ingestion_checkpoints)
            or any(item["status"] == "active" and item["id"] != pending_id for item in queue)):
        raise ValueError("resume the active dataset before adopting another parent cursor")
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("retained dataset needs an available source file")
    if hashlib.sha256(source_name.encode("utf-8")).hexdigest() != checkpoint["sourceNameHash"]:
        raise ValueError("retained dataset name does not match its original parser manifest")
    before = _source_stat(path, checkpoint["resolvedKind"])
    if before["size"] != checkpoint["sourceBytes"]:
        raise ValueError("retained dataset size changed")
    if checkpoint["resolvedKind"] == "sqlite":
        from .datasets import sqlite_consistent_snapshot_sha256
        content_hash = sqlite_consistent_snapshot_sha256(path)
    else:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                _check(cancelled)
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        content_hash = digest.hexdigest()
    if content_hash != checkpoint["contentHash"] or before != _source_stat(path, checkpoint["resolvedKind"]):
        raise ValueError("retained dataset bytes changed before resumption")
    _check(cancelled)
    key = hashlib.sha256(("file\0" + str(path)).encode("utf-8")).hexdigest()
    if was_active and key == old_key and checkpoint["sourceSnapshot"] == before:
        return {"id": pending_id, "state": "active", "idempotent": True,
                "checkpoint": copy.deepcopy(checkpoint)}
    checkpoint["sourceIdentity"] = key
    checkpoint["sourceSnapshot"] = before
    checkpoint["neuralStateChecksum"] = brain.parameter_checksum()
    prior_queue = copy.deepcopy(brain.connected_pending_ingestions)
    prior_cursors = copy.deepcopy(brain.ingestion_checkpoints)
    prior_joint = copy.deepcopy(brain.ingestion_joint_generation)
    selected.update({"status": "active", "destinationSourceIdentity": key,
                     "destinationCheckpoint": checkpoint})
    brain.connected_pending_ingestions = queue
    brain.ingestion_checkpoints = {key: checkpoint}
    brain.ingestion_joint_generation = None
    try:
        # Normal save creates the current neural/substrate/joint hashes. It
        # retains the original record/window cursor and complete schedule.
        brain.save()
    except BaseException:
        try:
            committed = read_json(brain.engine_path / "brain.json")
            durable = committed.get("connected_pending_ingestions") == brain.connected_pending_ingestions
        except (OSError, ValueError):
            durable = False
        if not durable:
            brain.connected_pending_ingestions = prior_queue
            brain.ingestion_checkpoints = prior_cursors
            brain.ingestion_joint_generation = prior_joint
        raise
    return {"id": pending_id, "state": "active", "idempotent": False,
            "rebound": was_active,
            "checkpoint": copy.deepcopy(brain.ingestion_checkpoints[key])}


def action_result_continuation(brain, selection, evidence_sha256):
    """Select only an authenticated inherited action-result source cursor."""
    if getattr(brain, "_connected_regions", None) is None:
        raise ValueError("action-result continuation requires a connected native merge")
    if (not isinstance(selection, Mapping)
            or set(selection) != {"pendingId", "originalEvidenceId", "parentId"}
            or not all(isinstance(selection.get(key), str) and selection[key] for key in selection)
            or not brain._sha256_identifier(selection["pendingId"])
            or not brain._sha256_identifier(selection["originalEvidenceId"])):
        raise ValueError("merged action-result continuation selector is invalid")
    selected = next((item for item in validate_pending_ingestions(brain.connected_pending_ingestions)
                     if item["id"] == selection["pendingId"]), None)
    source_name = "action-result-" + selection["originalEvidenceId"]
    if selected is None or selected["parentId"] != selection["parentId"]:
        raise ValueError("merged action-result continuation belongs to another source")
    checkpoint = selected["checkpoint"]
    if (checkpoint["contentHash"] != evidence_sha256
            or checkpoint["sourceNameHash"] != hashlib.sha256(source_name.encode("utf-8")).hexdigest()
            or checkpoint["transactionId"] != selection["originalEvidenceId"]
            or checkpoint["policy"] != "pretrain" or checkpoint["resolvedKind"] != "jsonl"):
        raise ValueError("merged action-result continuation does not match its exact original evidence")
    return selected, source_name
