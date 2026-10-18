"""Bounded App admission/commit fixtures, not model capability qualification."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_core.app_observation import (AppObservationCommitError, EMPTY_FRONTIER,
    admit_native_app_observation, frontier_sha256, native_app_observation_frontier,
    parse_app_observation, raw_sha256, validate_frontier_head)
from omni_core.persistence import atomic_write_json, read_json
from worker import Worker, ChatSteeringState, RpcFault


def observation(*, ordinal=1, previous=EMPTY_FRONTIER, stream="app-stream", expected=EMPTY_FRONTIER,
                disposition="observe", content=None, method="ui/message", brain_id="app-native-fixture"):
    data = {"method": method, "params": {"role": "user", "content": {"type": "text", "text": "copper leaves drift"}}
        if method == "ui/message" else {"structuredContent": {"season": "autumn", "count": 3}}}
    if content is not None:
        data["params"]["content"] = content
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    record = {"format": "omni-mcp-app-observation", "version": 1, "id": "observation-" + str(ordinal),
        "artifactId": "artifact-one", "captureSha256": "c" * 64,
        "source": {"brainId": brain_id, "serverId": "fixture-server", "serverLabel": "Fixture App", "endpoint": "stdio:fixture"},
        "observedAt": "2026-10-11T12:00:00.000Z", "reviewId": "review-" + str(ordinal), "method": method,
        "content": {"sha256": raw_sha256(raw), "bytes": len(raw.encode("utf-8")), "mimeType": "application/json"},
        "streamId": stream, "ordinal": ordinal, "previousSha256": previous, "untrusted": True}
    record_raw = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return {"brainId": brain_id, "recordJson": record_raw, "recordSha256": raw_sha256(record_raw),
        "dataJson": raw, "dataSha256": raw_sha256(raw), "expectedFrontierSha256": expected, "disposition": disposition}


class SyntheticBrain:
    """Commit-fault fixture only; native learning proof is separate below."""
    def __init__(self, directory):
        self.brain_id = "app-native-fixture"
        self.storage_path = Path(directory)
        self.engine_path = self.storage_path / "engine"
        self.engine_path.mkdir()
        self.resource_policy = SimpleNamespace(reserve_ram=lambda *_args: nullcontext())
        self.counters = {"experiences": 0}
        self.app_observation_frontier = None
        self.calls, self.saves, self.closed = [], 0, False
        self.ingestion_checkpoints = {}
        self.save()

    def learn_experience(self, text, **kwargs):
        self.calls.append((text, kwargs))
        self.counters["experiences"] += 1
        return {"assembly_id": "synthetic-assembly"}

    def save(self):
        self.saves += 1
        atomic_write_json(self.engine_path / "brain.json", {"brain_id": self.brain_id,
            "app_observation_frontier": self.app_observation_frontier, "counters": self.counters})

    def close(self):
        self.closed = True


class AppObservationContractTests(unittest.TestCase):
    def test_exact_raw_bytes_owner_and_no_hidden_wire_fields(self):
        wire = observation()
        self.assertEqual(parse_app_observation(wire, wire["brainId"]).text, "copper leaves drift")
        for changed in ({**wire, "dataJson": wire["dataJson"] + " "}, {**wire, "brainId": "other"},
                        {**wire, "systemPrompt": "hidden"}, {**wire, "disposition": "learn"}):
            with self.subTest(changed=changed.keys()), self.assertRaises(ValueError):
                parse_app_observation(changed, "app-native-fixture")

    def test_full_text_structured_and_inline_resources_preserve_order(self):
        wire = observation(method="ui/update-model-context", content=[{"type": "text", "text": "first"},
            {"type": "resource", "resource": {"uri": "fixture://text", "text": "second", "mimeType": "text/plain"}}])
        self.assertEqual(parse_app_observation(wire, wire["brainId"]).text,
                         'first\nsecond\n{"season":"autumn","count":3}')
        large = "full-record " * 8193
        wire = observation(content={"type": "text", "text": large})
        self.assertEqual(parse_app_observation(wire, wire["brainId"]).text, large.strip())

    def test_media_and_links_dispatch_to_sensory_path_not_serialized_as_text(self):
        for block in ({"type": "image", "data": "AAAA", "mimeType": "image/png"},
                      {"type": "audio", "data": "AAAA", "mimeType": "audio/wav"},
                      {"type": "resource_link", "uri": "https://example.invalid", "name": "link"},
                      {"type": "resource", "resource": {"uri": "fixture://binary", "blob": "AAAA"}}):
            wire = observation(content=block)
            admitted = parse_app_observation(wire, wire["brainId"])
            self.assertTrue(admitted.extended)
            self.assertIsNone(admitted.text)
            superseded = parse_app_observation({**wire, "disposition": "superseded"}, wire["brainId"])
            self.assertIsNone(superseded.text)
            self.assertFalse(superseded.extended)

    def test_fifo_duplicate_restart_and_new_stream_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            first = observation()
            result = admit_native_app_observation(brain, first)
            self.assertEqual(result["head"]["learning"]["experienceCountAfter"], 1)
            self.assertEqual(brain.calls[0][1], {"kind": "experience", "source": "mcp-app",
                "source_label": "reviewed-mcp-app:fixture-server/artifact-one", "steps": 0})
            saves = brain.saves
            duplicate = admit_native_app_observation(brain, first)
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual((len(brain.calls), brain.saves), (1, saves))
            brain.app_observation_frontier = read_json(brain.engine_path / "brain.json")["app_observation_frontier"]
            self.assertTrue(admit_native_app_observation(brain, first)["duplicate"])
            for wire in (observation(ordinal=3, previous=first["recordSha256"], expected=result["frontierSha256"]),
                         observation(ordinal=2, previous="f" * 64, expected=result["frontierSha256"]),
                         observation(stream="new-stream", expected=EMPTY_FRONTIER)):
                with self.assertRaises(ValueError):
                    admit_native_app_observation(brain, wire)
            next_wire = observation(ordinal=2, previous=first["recordSha256"], expected=result["frontierSha256"])
            second = admit_native_app_observation(brain, next_wire)
            self.assertEqual(second["head"]["baseFrontierSha256"], result["frontierSha256"])
            with self.assertRaises(ValueError):
                admit_native_app_observation(brain, first)
            new = admit_native_app_observation(brain, observation(stream="new-stream", expected=second["frontierSha256"]))
            self.assertEqual(new["head"]["baseFrontierSha256"], second["frontierSha256"])

    def test_superseded_advances_only_metadata_then_real_next_observation(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            skipped = observation(disposition="superseded")
            result = admit_native_app_observation(brain, skipped)
            self.assertEqual(result["head"]["state"], "advanced-without-observation")
            self.assertIsNone(result["head"]["learning"])
            self.assertEqual(brain.calls, [])
            observed = observation(ordinal=2, previous=skipped["recordSha256"], expected=result["frontierSha256"])
            result = admit_native_app_observation(brain, observed)
            self.assertEqual(result["head"]["learning"]["experienceCountAfter"], 1)
            self.assertEqual(len(brain.calls), 1)

    def test_failed_save_fences_warm_state_without_false_ack(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            before = read_json(brain.engine_path / "brain.json")
            with patch.object(brain, "save", side_effect=OSError("before publication")):
                with self.assertRaises(AppObservationCommitError):
                    admit_native_app_observation(brain, observation())
            self.assertTrue(brain._connected_recovery_required)
            self.assertIsNone(brain.app_observation_frontier)
            self.assertEqual(read_json(brain.engine_path / "brain.json"), before)
            with self.assertRaises(AppObservationCommitError):
                admit_native_app_observation(brain, observation())
            self.assertEqual(len(brain.calls), 1)

    def test_committed_save_followed_by_cleanup_error_returns_exact_durable_ack(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            real_save = brain.save
            def save_then_error():
                real_save()
                raise OSError("after publication cleanup")
            with patch.object(brain, "save", side_effect=save_then_error):
                result = admit_native_app_observation(brain, observation())
            self.assertTrue(result["reloadRequired"])
            self.assertTrue(brain._connected_recovery_required)
            self.assertEqual(result["head"], read_json(brain.engine_path / "brain.json")["app_observation_frontier"])

    def test_paused_training_uses_existing_observed_boundary_not_stale_cursor(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            brain.ingestion_checkpoints = {"paused": {"neuralStateChecksum": "old"}}
            brain.parameter_checksum = lambda: "old"
            boundary = []
            brain._save_observed_neural_boundary = lambda before, source: (boundary.append((before, source)), brain.save())
            admit_native_app_observation(brain, observation())
            self.assertEqual(boundary, [("old", "observation-1")])

    def test_head_is_bounded_and_hash_is_language_independent(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            result = admit_native_app_observation(brain, observation())
            self.assertEqual(result["frontierSha256"], frontier_sha256(result["head"]))
            self.assertNotIn("copper leaves", json.dumps(result["head"]))
            self.assertLess(len(json.dumps(result["head"])), 2048)
            invalid = copy.deepcopy(result["head"])
            invalid["learning"]["experienceCountAfter"] += 1
            with self.assertRaises(ValueError):
                validate_frontier_head(invalid)

    def test_worker_requires_load_and_preserves_active_or_retained_decoder(self):
        worker = Worker()
        with tempfile.TemporaryDirectory() as root:
            params = {**observation(), "storagePath": root}
            try:
                with self.assertRaises(RpcFault):
                    worker.admit_app_observation(params)
                brain = SyntheticBrain(root)
                worker.brains[brain.brain_id] = brain
                with patch.object(worker, "_get", return_value=brain), patch.object(worker, "_execution_store") as store:
                    worker._chat_steering[(brain.brain_id, "turn-one")] = ChatSteeringState(brain.brain_id, "turn-one", claimed=True)
                    with self.assertRaisesRegex(RuntimeError, "active native turn"):
                        worker.admit_app_observation(params)
                    worker._chat_steering.clear()
                    store.return_value.current.return_value = {"status": "suspended"}
                    with self.assertRaisesRegex(RuntimeError, "retained native turn"):
                        worker.admit_app_observation(params)
                    self.assertEqual(brain.calls, [])
                    store.return_value.current.return_value = None
                    worker.admit_app_observation(params)
                    self.assertEqual(len(brain.calls), 1)
            finally:
                worker.shutdown({}, None)

    def test_worker_releases_failed_owner_without_implicit_reload_or_retry(self):
        with tempfile.TemporaryDirectory() as root:
            worker, brain = Worker(), SyntheticBrain(root)
            worker.brains[brain.brain_id] = brain
            params = {**observation(), "storagePath": root}
            try:
                with patch.object(worker, "_get", return_value=brain), patch.object(worker, "_execution_store") as store, \
                        patch.object(brain, "save", side_effect=OSError("failed atomic publication")), \
                        patch.object(worker, "_release_owner_lease") as release, patch.object(worker, "notify") as notify:
                    store.return_value.current.return_value = None
                    with self.assertRaises(RpcFault) as caught:
                        worker.admit_app_observation(params)
                    self.assertEqual(caught.exception.code, -32031)
                    self.assertTrue(caught.exception.data["nativeOwnerReleased"])
                    self.assertTrue(caught.exception.data["reloadRequired"])
                    self.assertFalse(caught.exception.data["committed"])
                    self.assertEqual(notify.call_args.args[0], "native-owner-released")
                    release.assert_called_once_with(brain.brain_id)
                self.assertTrue(brain.closed)
                self.assertNotIn(brain.brain_id, worker.brains)
                self.assertEqual(len(brain.calls), 1)
            finally:
                worker.shutdown({}, None)

    def test_worker_returns_durable_cleanup_failure_ack_after_releasing_owner(self):
        with tempfile.TemporaryDirectory() as root:
            worker, brain = Worker(), SyntheticBrain(root)
            worker.brains[brain.brain_id] = brain
            params = {**observation(), "storagePath": root}
            save = brain.save
            def after_publication():
                save()
                raise OSError("cleanup failure")
            try:
                with patch.object(worker, "_get", return_value=brain), patch.object(worker, "_execution_store") as store, \
                        patch.object(brain, "save", side_effect=after_publication), patch.object(worker, "notify") as notify:
                    store.return_value.current.return_value = None
                    receipt = worker.admit_app_observation(params)
                self.assertTrue(receipt["reloadRequired"])
                self.assertTrue(receipt["nativeOwnerReleased"])
                self.assertTrue(notify.call_args.kwargs["data"]["committed"])
                self.assertEqual(receipt["head"], read_json(brain.engine_path / "brain.json")["app_observation_frontier"])
                self.assertTrue(brain.closed)
                self.assertNotIn(brain.brain_id, worker.brains)
            finally:
                worker.shutdown({}, None)


if __name__ == "__main__":
    unittest.main()
