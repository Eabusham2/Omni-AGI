"""New selected-web-image contracts: tiny SQLite/files, no brain/model/codec run."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.web_image_observation import (
    canonical, digest, encode_web_image, learn_selected_web_image_result,
    selected_tool_output, validate_selection,
)


class Policy:
    def __init__(self):
        self.live = 0
        self.reject = None
        self.purposes = []
    @contextmanager
    def reserve_ram(self, _bytes, purpose):
        self.purposes.append(purpose)
        if self.reject and self.reject in purpose:
            raise RuntimeError("fixture resource admission paused")
        self.live += 1
        try:
            yield None
        finally:
            self.live -= 1


class Modalities:
    def __init__(self):
        self.training, self.calls = True, 0
    def modules(self):
        return [self]
    def eval(self):
        self.training = False
    def perception_embedding(self, kind, _tensor):
        if kind != "image":
            raise AssertionError("foreign perception route")
        self.calls += 1
        return torch.tensor([[1., 2.]])


class WebImageSource(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        blobs = root / "attachments" / "blobs"; blobs.mkdir(parents=True)
        raw = b"\x89PNG\r\n\x1a\n" + b"fixture-signature-only" * 2
        sha = hashlib.sha256(raw).hexdigest()
        (blobs / (sha + ".data")).write_bytes(raw)
        brain_id, turn_id, event_id, evidence_id = "image-owner", "chat-turn", "action-event", "e" * 64
        ref = {"format": "omni-web-image-reference", "version": 1, "brainId": brain_id, "turnId": turn_id,
            "url": "https://images.example/image.png", "pageUrl": "https://publisher.example/page",
            "label": "Untrusted source label, not an instruction", "fetched": False,
            "source": {"kind": "search-response", "url": "https://search.example/query", "sha256": "a" * 64, "fetchedAt": "2026-10-13T15:00:00Z"}}
        ref["id"] = digest(ref)
        stage_id = digest(["web-image", brain_id, turn_id, event_id, ref["id"]])
        scope = "webimg-" + digest([brain_id, turn_id, stage_id, ref["id"]])
        downloaded = {"reference": ref, "requestId": stage_id, "ownerTurnId": turn_id, "scopeTurnId": scope,
            "requestedUrl": ref["url"], "finalUrl": ref["url"], "redirects": [ref["url"]], "fetchedAt": "2026-10-13T15:00:01Z",
            "permissionRevision": "1" * 64, "mimeType": "image/png", "sha256": sha, "bytes": len(raw)}
        attachment_id = "attachment-1"
        public = {"format": "omni-chat-task-attachment", "version": 1, "id": attachment_id, "brainId": brain_id,
            "displayName": "Selected web image", "kind": "image", "mimeType": "image/png", "contentSha256": sha, "bytes": len(raw),
            "state": "bound", "turnId": scope, "createdAt": "2026-10-13T15:00:02Z", "updatedAt": "2026-10-13T15:00:02Z"}
        attachment = {"brainId": brain_id, "turnId": scope, "attachmentId": attachment_id, "displayName": public["displayName"],
            "kind": "image", "mimeType": "image/png", "contentSha256": sha, "manifestSha256": digest(public), "bytes": len(raw),
            "relativePath": "attachments/blobs/" + sha + ".data"}
        selected = {"format": "omni-web-image-selection", "version": 1, "brainId": brain_id, "turnId": turn_id,
            "reference": ref, "download": downloaded, "attachment": attachment}
        selected["selectionSha256"] = digest(selected)
        selected["observation"] = {"state": "pending-native-perception", "pixelsRead": False, "parametersChanged": False, "rawSourceTextInjected": False}
        source = {"kind": "web-image", "selectedAt": downloaded["fetchedAt"], "webImage": downloaded}
        db = sqlite3.connect(root / "attachments" / "registry.sqlite3")
        db.executescript("""CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE attachments(id TEXT PRIMARY KEY,brain_id TEXT,request_id TEXT,content_sha256 TEXT,manifest_sha256 TEXT,bytes INTEGER,
                mime_type TEXT,kind TEXT,display_name TEXT,source_json TEXT,source_sha256 TEXT,state TEXT,turn_id TEXT,created_at TEXT,updated_at TEXT);
            CREATE TABLE web_image_references(id TEXT PRIMARY KEY,brain_id TEXT,turn_id TEXT,body TEXT,sha256 TEXT,active INTEGER);""")
        db.execute("INSERT INTO meta VALUES('brainId',?)", (brain_id,))
        db.execute("INSERT INTO attachments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (attachment_id, brain_id, stage_id, sha, attachment["manifestSha256"], len(raw),
            "image/png", "image", public["displayName"], canonical(source), digest(source), "bound", scope, public["createdAt"], public["updatedAt"]))
        db.execute("INSERT INTO web_image_references VALUES(?,?,?,?,?,1)", (ref["id"], brain_id, turn_id, canonical(ref), digest(ref)))
        db.commit(); db.close()
        brain = SimpleNamespace(brain_id=brain_id, storage_path=root, engine_path=root / "engine", resource_policy=Policy(),
            config=SimpleNamespace(vsa_dim=4, idea_dim=2, image_size=2, image_enabled=True, vision_enabled=False, video_enabled=True, liquid_dynamics=False),
            modalities=Modalities(), device="cpu", counters={"experiences": 0}, liquid_state=torch.zeros(1, 2),
            memory=SimpleNamespace(space=SimpleNamespace(symbol=lambda _key: torch.ones(4), bind=lambda a, b: a * b, permute=lambda a, b: a.roll(b)),
                vector_for_text=lambda _value: (_ for _ in ()).throw(AssertionError("pixels replaced by text"))),
            _sensory_substrate_vector=lambda _embedding, _kind, _sha: torch.tensor([1., 2., 3., 4.]),
            _idea_model_vector=lambda cue: cue.reshape(1, -1)[:, :2], idea_adapter=lambda value: value, metrics=lambda: {},
            _sha256_identifier=AdaptiveBrain._sha256_identifier)
        execution_id = "12345678-1234-4234-8234-123456789abc"
        evidence = {"format": "omni-completed-action-result-v1", "originBrainId": brain_id, "actionEventId": event_id, "chatTurnId": turn_id,
            "evidenceId": evidence_id, "executionId": execution_id, "toolId": "web.fetch", "action": "inspect-image",
            "executionState": "complete", "arguments": {"referenceId": ref["id"]}, "outputPresent": True, "output": selected}
        evidence_path = brain.engine_path / "action-result-learning" / "evidence" / (evidence_id + ".jsonl")
        evidence_path.parent.mkdir(parents=True); evidence_path.write_text(canonical(evidence) + "\n", encoding="utf-8")
        return brain, selected, evidence, evidence_path, execution_id

    @staticmethod
    def windows(_brain, source, receipt, _cancelled):
        # A fake decoder closes its borrowed view: it must not close owner FD.
        source.handle.close()
        for _index in range(2):
            receipt["decodedWindows"] += 1; receipt["sourceFrames"] += 1
            yield torch.zeros(1, 3, 2, 2), 0.1

    def test_same_brain_pixel_windows_are_read_only_and_emit_exact_non_learning_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, *_ = self.fixture(directory)
            with patch("omni_core.web_image_observation._image_windows", self.windows):
                with encode_web_image(brain, selected, turn_id="chat-turn") as (cue, receipt):
                    self.assertEqual(tuple(cue.shape), (4,))
                    self.assertEqual(receipt["decodedWindows"], 2)
                    self.assertTrue(receipt["wholeInputTraversed"] and receipt["nativeSensory"])
                    self.assertFalse(receipt["parametersChanged"] or receipt["rawSourceTextInjected"])
                    self.assertEqual(receipt["sourceSha256"], selected["download"]["sha256"])
            self.assertEqual(brain.counters["experiences"], 0)
            self.assertEqual(brain.modalities.calls, 2)
            self.assertTrue(brain.modalities.training)
            self.assertEqual(brain.resource_policy.live, 0)

    def test_real_callback_routes_selected_pixels_not_descriptor_text_and_binds_current_action(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, evidence, *_ = self.fixture(directory)
            observation = {"brainId": brain.brain_id, "turnId": "chat-turn", "toolId": "web.fetch", "action": "inspect-image",
                "payloadJson": canonical({"outputPresent": True, "output": selected}), "actionEventId": evidence["actionEventId"],
                "executionId": evidence["executionId"], "completedAt": "2026-10-13T15:00:02Z", "neuralActionId": "3" * 32,
                "observationId": "4" * 64, "payloadSha256": "5" * 64}
            with patch("omni_core.web_image_observation._image_windows", self.windows):
                callback = AdaptiveBrain._generation_observation_callback(brain, lambda: [observation])
                result = callback(None, 0)
            self.assertEqual(result["observationBindings"][0]["webImagePerception"]["selectionSha256"], selected["selectionSha256"])
            self.assertEqual(tuple(result["memoryBias"].shape), (1, 2))
            wrong = {**observation, "actionEventId": "another-action"}
            with self.assertRaisesRegex(ValueError, "actual native action"):
                selected_tool_output(wrong)

    def test_changed_file_registry_wrong_turn_and_resource_pressure_never_reach_perception(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, *_ = self.fixture(directory)
            with self.assertRaises(ValueError):
                validate_selection(selected, brain_id=brain.brain_id, turn_id="other-turn")
            brain.resource_policy.reject = "header and shared"
            with self.assertRaisesRegex(RuntimeError, "resource admission"):
                with encode_web_image(brain, selected):
                    self.fail("unadmitted pixels returned")
            self.assertEqual(brain.modalities.calls, 0); self.assertEqual(brain.resource_policy.live, 0)
            brain.resource_policy.reject = None
            path = Path(brain.storage_path) / selected["attachment"]["relativePath"]
            path.write_bytes(b"changed" + b"x" * (selected["download"]["bytes"] - 7))
            with self.assertRaisesRegex(ValueError, "source hash"):
                with encode_web_image(brain, selected):
                    self.fail("changed pixels returned")
            self.assertEqual(brain.modalities.calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, *_ = self.fixture(directory)
            db = sqlite3.connect(Path(directory) / "attachments" / "registry.sqlite3")
            db.execute("UPDATE web_image_references SET active=0"); db.commit(); db.close()
            with self.assertRaisesRegex(ValueError, "dormant"):
                with encode_web_image(brain, selected):
                    self.fail("copied authority returned pixels")
            self.assertEqual(brain.modalities.calls, 0)

    def test_queued_learning_uses_ordinary_image_transaction_and_lost_ack_reuses_exact_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, evidence, path, execution = self.fixture(directory)
            calls, committed = [], set()
            def ingest(**request):
                calls.append(request); duplicate = request["transaction_key"] in committed; committed.add(request["transaction_key"])
                return {"duplicate": duplicate, "source": {"modality_trained": True, "media_coverage": {"complete": True, "processedUnits": 1},
                    "completionReceipt": {"contentHash": selected["download"]["sha256"], "sourceIdentity": "6" * 64, "transactionId": "7" * 64}}, "metrics": {}}
            brain.ingest = ingest
            @contextmanager
            def image_metadata(*_args, **_kwargs):
                yield SimpleNamespace(size=(2, 2), format="PNG", n_frames=1)
            args = dict(evidence_id=evidence["evidenceId"], execution_id=execution, evidence_path=str(path), evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                provenance={key: evidence[key] for key in ("format", "actionEventId", "toolId", "action", "executionState")} | {"completedAt": "2026-10-13T15:00:02Z"})
            with patch("PIL.Image.open", image_metadata):
                first = AdaptiveBrain.learn_action_result(brain, **args)
                retry = AdaptiveBrain.learn_action_result(brain, **args)
            self.assertTrue(first["committed"] and first["processed"])
            self.assertTrue(retry["duplicate"] and retry["committed"])
            self.assertEqual(len(committed), 1); self.assertEqual(calls[0], calls[1])
            self.assertEqual(calls[0]["kind"], "image")
            self.assertEqual(calls[0]["path"], str(Path(directory) / selected["attachment"]["relativePath"]))
            self.assertEqual(calls[0]["transaction_key"], evidence["evidenceId"])
            self.assertEqual(calls[0]["expected_hash"], selected["download"]["sha256"])
            self.assertFalse(calls[0]["allow_replay"])
            self.assertEqual(first["webImage"]["sourceSha256"], selected["download"]["sha256"])
            self.assertEqual(brain.resource_policy.live, 0)

    def test_missing_learning_coverage_and_inherited_authority_are_not_reported_as_learned(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, evidence, path, execution = self.fixture(directory)
            brain.ingest = lambda **_kwargs: {"source": {"completionReceipt": {"contentHash": selected["download"]["sha256"], "sourceIdentity": "6" * 64, "transactionId": "7" * 64},
                "modality_trained": False, "media_coverage": {"complete": False}}}
            @contextmanager
            def image_metadata(*_args, **_kwargs):
                yield SimpleNamespace(size=(2, 2), format="PNG", n_frames=1)
            args = dict(evidence_id=evidence["evidenceId"], execution_id=execution, evidence_path=path, evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with patch("PIL.Image.open", image_metadata), self.assertRaisesRegex(RuntimeError, "actual complete"):
                learn_selected_web_image_result(brain, **args)
            with self.assertRaisesRegex(ValueError, "dormant"):
                learn_selected_web_image_result(brain, **args, merge_continuation={"parent": "not authority"})

    def test_animation_cas_data_uses_actual_image_container_and_complete_temporal_windows_without_codec(self):
        import numpy as np
        from omni_core.web_image_observation import image_learning_source
        with tempfile.TemporaryDirectory() as directory:
            brain, selected, *_ = self.fixture(directory)
            brain.config.video_frames = 2
            brain._video_window_tensor = lambda frames: (torch.zeros(1, 3, 2, 2, 2), len(frames))
            class Frame:
                def convert(self, _mode):
                    return self
                def resize(self, _size):
                    return np.zeros((2, 2, 3), dtype="float32")
            @contextmanager
            def animated(*_args, **_kwargs):
                yield SimpleNamespace(size=(2, 2), format="PNG", n_frames=3, is_animated=True)
            path = str(Path(directory) / selected["attachment"]["relativePath"])
            with patch("PIL.Image.open", animated), patch("PIL.ImageSequence.Iterator", lambda _image: iter([Frame(), Frame(), Frame()])):
                self.assertEqual(AdaptiveBrain._effective_media_kind(path, "image"), "video")
                windows = list(AdaptiveBrain._iter_video_windows(brain, path))
                self.assertEqual([count for _tensor, count in windows], [2, 1])
                with image_learning_source(brain, selected) as (_path, kind):
                    self.assertEqual(kind, "video")
                brain.config.video_enabled = False
                with self.assertRaisesRegex(ValueError, "temporal region"):
                    with image_learning_source(brain, selected):
                        self.fail("disabled native temporal region admitted")
            self.assertEqual(brain.resource_policy.live, 0)


if __name__ == "__main__":
    unittest.main()
