"""Atomic same-turn execution checkpoints, separate from learned checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path

from .execution_state_codec import ExecutionStateReader, ExecutionStateWriter, file_digest
from .persistence import atomic_write_json, read_json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def binding_state_for(brain, turn_id):
    # Runtime resource settings can change without changing these immutable
    # authoritative learned/control generations. No shape/model substitute.
    body = {"brainId": brain.brain_id, "turnId": turn_id,
        "mutable": brain.mutable_state_manifest, "substrate": brain.memory.persistence_manifest,
        "coreSha256": brain.parameter_checksum(),
        "attentionEpoch": brain._attention_epoch()}
    if not isinstance(body["mutable"], dict) or not isinstance(body["substrate"], dict):
        raise ValueError("native suspension requires committed neural generation pointers")
    return body


def binding_for(brain, turn_id):
    return hashlib.sha256(canonical(binding_state_for(brain, turn_id))).hexdigest()


class ExecutionCheckpointStore:
    format = "omni-native-execution-checkpoint"

    def __init__(self, engine_path, brain_id, *, policy=None):
        self.engine = Path(engine_path)
        self.root = self.engine / ".execution"
        self.brain_id, self.policy = brain_id, policy

    def _root(self, create=False):
        if self.engine.is_symlink() or not self.engine.is_dir():
            raise ValueError("native execution root must be an existing private engine directory")
        if self.root.is_symlink():
            raise ValueError("native execution state cannot follow a symbolic link")
        if create:
            self.root.mkdir(mode=0o700, exist_ok=True)
        return self.root

    def _read(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("native execution metadata is not a regular file")
        reserve = getattr(self.policy, "reserve_ram", None)
        with reserve(path.stat().st_size * 12 + 65536, "native continuation metadata") if callable(reserve) else nullcontext():
            return read_json(path)

    def current(self, expected_id=None):
        root = self._root()
        pointer = root / "active.json"
        if not pointer.exists():
            if expected_id is not None:
                raise ValueError("the requested native continuation does not exist")
            return None
        active = self._read(pointer)
        identifier, digest = active.get("checkpointId"), active.get("manifestSha256")
        if (active.get("brainId") != self.brain_id or not isinstance(identifier, str)
                or not re.fullmatch(r"[a-f0-9]{32}", identifier) or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest) or expected_id is not None and identifier != expected_id):
            raise ValueError("native continuation has a different owner or checkpoint identity")
        directory = root / identifier
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("native continuation directory is invalid")
        value = self._read(directory / ("m-" + digest + ".json"))
        if (hashlib.sha256(canonical(value)).hexdigest() != digest or value.get("format") != self.format
                or value.get("version") != 2 or value.get("brainId") != self.brain_id
                or value.get("checkpointId") != identifier or value.get("status") not in {"suspended", "completed", "stopped"}):
            raise ValueError("native continuation manifest checksum or format is invalid")
        frontier, worker, presentation = value.get("frontier"), value.get("worker"), value.get("presentation")
        if value.get("runtimeCompatibility") is not None:
            from .execution_compatibility import validate_execution_compatibility
            validate_execution_compatibility(value["runtimeCompatibility"])
        if (not isinstance(frontier, dict) or not isinstance(worker, dict) or not isinstance(presentation, dict)
                or frontier.get("checkpointId") != identifier or frontier.get("exactSupportedResume") is not True
                or frontier.get("phase") != "native-text-token-boundary"
                or not isinstance(frontier.get("turnId"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", frontier["turnId"])
                or not isinstance(frontier.get("inputSha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", frontier["inputSha256"])
                or any(type(frontier.get(key)) is not int or not 0 <= frontier[key] <= (1 << 53) - 1
                    for key in ("lastStreamSequence", "sampledTokens", "publishedTokens"))
                or frontier["publishedTokens"] > frontier["sampledTokens"]
                or type(value.get("minimumContextTokens")) is not int or value["minimumContextTokens"] < 0
                or worker.get("brainId") != self.brain_id or worker.get("turnId") != frontier["turnId"]
                or worker.get("lastStreamSequence") != frontier["lastStreamSequence"]
                or not isinstance(presentation.get("input"), str) or not isinstance(presentation.get("publishedText"), str)
                or worker.get("publishedText") != presentation["publishedText"] or worker.get("actions") != presentation.get("actions")
                or not isinstance(presentation.get("actions"), list)
                or self._presentation_input_sha256(presentation) != frontier["inputSha256"]):
            raise ValueError("native continuation frontier/presentation ownership is invalid")
        return value

    @staticmethod
    def _presentation_input_sha256(presentation):
        from .native_audio_chat import observed_input_sha256
        raw = presentation.get("inputReceipt")
        audio = raw.get("nativeAudioInput") if isinstance(raw, dict) else None
        return observed_input_sha256({"content": presentation["input"], **({"native_audio_input": audio} if audio is not None else {})})

    def _publish(self, directory, value):
        digest = hashlib.sha256(canonical(value)).hexdigest()
        # New immutable metadata first, one pointer replacement last. A crash
        # cannot make an old pointer refer to overwritten manifest contents.
        target = directory / ("m-" + digest + ".json")
        if self.policy is not None:
            self.policy.require_disk(len(canonical(value)) + 65536, "native continuation metadata commit")
        if not target.exists():
            atomic_write_json(target, value)
        atomic_write_json(self.root / "active.json", {"brainId": self.brain_id,
            "checkpointId": value["checkpointId"], "manifestSha256": digest})
        return value

    def save(self, brain, context, cursor, *, phase, worker_state, journal, continuation):
        if context.get("turn_id") != worker_state.get("turnId"):
            raise ValueError("suspended chat context lost its owned turn")
        self._root(create=True)
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        directory.mkdir(mode=0o700)
        committed = False
        clock = brain._owned_activity_clock()
        previously_paused = clock.paused
        try:
            from .execution_rng import capture_execution_rng
            from .execution_compatibility import capture_execution_compatibility
            clock.pause()
            activity_clock = clock.snapshot()
            compatibility = capture_execution_compatibility(brain.device)
            global_rng = capture_execution_rng(brain.device)
            # Caller owns a safe token boundary and all neural mutation locks.
            # Persist that exact brain before publishing a cursor bound to it.
            brain.save()
            binding = binding_for(brain, context["turn_id"])
            journal_receipt = journal.export_resume_state(directory / "diagnostic", binding_id=binding)
            values = {key: value for key, value in context.items() if key != "before_parameters"}
            writer = ExecutionStateWriter(directory, policy=self.policy)
            payload = writer.encode({"context": values, "cursor": cursor, "continuation": continuation,
                "globalRng": global_rng, "activityClock": activity_clock})
            sampled = int(cursor.generated.shape[1]) - cursor.input_length
            published = cursor.emission_start - cursor.input_length
            frontier = {"checkpointId": identifier, "turnId": context["turn_id"], "inputSha256": context["input_sha256"],
                "lastStreamSequence": worker_state["lastStreamSequence"], "sampledTokens": sampled,
                "publishedTokens": published, "phase": "native-text-token-boundary", "exactSupportedResume": True}
            presentation = {"input": context["clean"], "publishedText": worker_state["publishedText"],
                "actions": worker_state["actions"], "inputReceipt": context["admitted_input"]}
            result = {"format": self.format, "version": 2, "brainId": self.brain_id, "checkpointId": identifier,
                "status": "suspended", "bindingId": binding, "binding": binding_state_for(brain, context["turn_id"]), "phase": phase, "frontier": frontier,
                "minimumContextTokens": min(int(cursor.generated.shape[1]) - cursor.window_start, cursor.options["context_capacity"]),
                "journal": journal_receipt, "files": writer.files, "payload": payload, "worker": worker_state,
                "presentation": presentation, "backend": str(brain.device), "runtimeCompatibility": compatibility,
                "neuralFiles": {name: {"sha256": file_digest(brain.engine_path / name), "bytes": (brain.engine_path / name).stat().st_size}
                    for name in ("core.safetensors", "plasticity.safetensors")}}
            self._publish(directory, result)
            committed = True
            return result
        finally:
            if not committed:
                # This exact directory was generated above; no imported or
                # caller-supplied path is a recursive cleanup target.
                shutil.rmtree(directory, ignore_errors=True)
                if not previously_paused:
                    clock.resume()

    def restore(self, brain, expected_id):
        value = self.current(expected_id)
        if value["status"] != "suspended" or value["bindingId"] != binding_for(brain, value["frontier"]["turnId"]):
            raise ValueError("native continuation no longer matches its committed neural state")
        if value["backend"] != str(brain.device):
            raise ValueError("exact native continuation requires its recorded compute backend")
        self.require_supported_frontier(value)
        directory = self.root / expected_id
        reader = ExecutionStateReader(directory, value["files"], pager=brain.working_attention_pager,
            device=brain.device, policy=self.policy)
        journal = None
        try:
            payload = reader.decode(value["payload"])
            cursor, context = payload["cursor"], payload["context"]
            if payload.get("activityClock") != brain._owned_activity_clock().snapshot() or not brain.activity_clock.paused:
                raise ValueError("exact continuation requires its sealed paused activity clock")
            if (context.get("turn_id") != value["frontier"]["turnId"] or context.get("input_sha256") != value["frontier"]["inputSha256"]
                    or context.get("clean") != value["presentation"]["input"]
                    or int(cursor.generated.shape[1]) - cursor.input_length != value["frontier"]["sampledTokens"]
                    or cursor.emission_start - cursor.input_length != value["frontier"]["publishedTokens"]
                    or brain.tokenizer.decode(cursor.generated[0, cursor.input_length:cursor.emission_start].detach().cpu().tolist())
                        != value["presentation"]["publishedText"]):
                raise ValueError("native continuation activity differs from its published frontier")
            from .parameter_diagnostics import ParameterDeltaJournal
            roots = [(name, getattr(brain, name)) for name in ParameterDeltaJournal.scope["modules"]]
            journal = ParameterDeltaJournal.from_resume_state(roots, directory / "diagnostic", binding_id=value["bindingId"],
                expected_manifest_sha256=value["journal"]["manifestSha256"],
                directory=brain._live_paging_cache_directory / "core-diagnostics",
                reserve_ram=brain._reserve_core_diagnostic_state, reserve_disk=brain.resource_policy.require_disk,
                resource_policy=brain.resource_policy)
            payload["context"]["before_parameters"] = journal
            payload["cursor"].bind_verified_cache(brain.decoder)
            return value, payload, reader
        except BaseException:
            reader.close()
            if journal is not None:
                journal.close()
            raise

    def require_supported_frontier(self, value=None):
        """Qualify execution, distinct from archival reading of old state.

        Complete old exports and explicit Stop remain readable. They may not
        be advertised/adopted as executable without a real saved runtime and
        paused clock, and are never retroactively assigned those seals.
        """
        value = self.current() if value is None else value
        if (not isinstance(value, dict) or value.get("status") != "suspended" or value.get("brainId") != self.brain_id
                or value.get("format") != self.format or value.get("version") != 2):
            raise ValueError("only an actual suspended frontier can be qualified for execution")
        from .execution_compatibility import require_execution_compatibility
        require_execution_compatibility(value.get("runtimeCompatibility"), value["backend"])
        payload = value.get("payload")
        if not isinstance(payload, dict) or payload.get("kind") != "map" or not isinstance(payload.get("entries"), list):
            raise ValueError("exact continuation has no closed activity payload")
        members = [row[1] for row in payload["entries"] if isinstance(row, list) and len(row) == 2 and row[0] == "activityClock"]
        if len(members) != 1 or not isinstance(members[0], dict) or members[0].get("kind") != "map":
            raise ValueError("this saved frontier has no sealed paused activity clock; exact restore is unsupported")
        entries = members[0].get("entries")
        if not isinstance(entries, list) or len(entries) != 4 or any(not isinstance(row, list) or len(row) != 2 or type(row[0]) is not str for row in entries):
            raise ValueError("saved paused activity clock structure is invalid")
        clock = dict(entries)
        from .activity_clock import ActivityClock
        parsed = ActivityClock.from_state(clock)
        if not parsed.paused:
            raise ValueError("saved execution activity clock must be paused")
        metadata = self._read(self.engine / "brain.json")
        if metadata.get("brain_id") != self.brain_id or metadata.get("activity_clock") != clock:
            raise ValueError("saved execution activity clock differs from its exact learned-state owner")
        return value

    def update_worker(self, expected_id, worker_state):
        value = self.current(expected_id)
        if value["status"] != "suspended" or worker_state.get("turnId") != value["frontier"]["turnId"]:
            raise ValueError("observation update does not own a suspended turn")
        value = {**value, "worker": worker_state}
        return self._publish(self.root / expected_id, value)

    def finish(self, expected_id, *, turn_id, stopped=False):
        value = self.current(expected_id)
        if value["frontier"]["turnId"] != turn_id:
            raise ValueError("native completion does not own this suspended turn")
        if value["status"] != "suspended":
            if value["status"] == ("stopped" if stopped else "completed"):
                return value
            raise ValueError("native continuation was already terminated differently")
        value = {**value, "status": "stopped" if stopped else "completed"}
        return self._publish(self.root / expected_id, value)

    def mark_unloaded(self):
        self._root(create=True)
        atomic_write_json(self.root / "unloaded.json", {"brainId": self.brain_id, "explicitLoadRequired": True})

    def explicitly_unloaded(self):
        self._root()
        path = self.root / "unloaded.json"
        if not path.exists():
            return False
        value = self._read(path)
        if value != {"brainId": self.brain_id, "explicitLoadRequired": True}:
            raise ValueError("invalid unloaded instance ownership marker")
        return True

    def mark_loaded(self):
        if self.explicitly_unloaded():
            (self.root / "unloaded.json").unlink()
