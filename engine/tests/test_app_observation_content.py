"""Real bounded codecs + synthetic encoder/admission contract, no model build.

These fixtures do not prove learned sensory competence or native packed changes.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.app_observation import (AppObservationCommitError, admit_native_app_observation,
                                      frontier_sha256, native_app_observation_frontier)
from omni_core.app_observation_audit import (app_observation_audit_snapshot_files,
    copy_app_observation_audit_snapshot, verify_audit_file)
from omni_core.offload import NeuralStateResourcePause
from omni_core.vsa import HypervectorSpace
from tests.test_app_observation import SyntheticBrain, observation
from worker import Worker, RpcFault
import tests.test_mcp_sampling_media as codec_fixture


class ContractEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Identity()
        self.child.train(False)
        self.seen = []

    def perception_embedding(self, kind, tensor):
        self.seen.append((kind, tuple(tensor.shape)))
        # Explicit synthetic output; actual same-brain neural proof is separate.
        return torch.tensor([[.2, .4, .6, .8] * 4], dtype=torch.float32)


class ContentContractBrain(SyntheticBrain):
    def __init__(self, directory):
        super().__init__(directory)
        self.resource_policy = codec_fixture.DecoderPolicy()
        self.disk_reservations = []
        self.resource_policy.require_disk = lambda size, label: self.disk_reservations.append((size, label))
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(vsa_dim=16, idea_dim=16, image_size=8, audio_samples=64, modality_channels=2)
        self.modalities = ContractEncoder()
        self.text_seen, self.sensory_admissions = [], []
        self.memory = SimpleNamespace(space=HypervectorSpace(16, 491))
        self.memory.vector_for_text = lambda text: (self.text_seen.append(text), self.memory.space.symbol("text:" + text))[1]
        self._live_paging_cache_directory = self.engine_path / "private-cache"

    def _sensory_substrate_vector(self, embedding, _kind, _fingerprint):
        return embedding.detach()[0]

    def _admit_sensory_embedding(self, vector, **kwargs):
        self.sensory_admissions.append((vector.clone(), kwargs))
        self.counters["experiences"] += 1
        return {"assemblyId": "contract-sensory-assembly", "vector": vector,
                "memorySettling": {"afterimageId": "contract-afterimage"}}


def mixed_wire(**changes):
    blocks = [{"type": "text", "text": "literal before"},
        codec_fixture.content("image", "image/png", codec_fixture.image_bytes()),
        codec_fixture.content("audio", "audio/wav", codec_fixture.wav_bytes()),
        {"type": "resource", "resource": {"uri": "fixture://inline", "text": "literal after", "mimeType": "text/plain"}}]
    return observation(content=blocks, **changes)


class AppContentContractTests(unittest.TestCase):
    def test_real_png_wav_traversal_order_and_one_synthetic_admission(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            modes = [(module, module.training) for module in brain.modalities.modules()]
            result = admit_native_app_observation(brain, mixed_wire())
            summary, audit = result["head"]["content"]["summary"], result["head"]["content"]["audit"]
            self.assertEqual(summary["blockCount"], 4)
            self.assertEqual((summary["textBlocks"], summary["resourceTextBlocks"], summary["imageBlocks"], summary["audioBlocks"]), (1, 1, 1, 1))
            self.assertEqual((summary["decodedWindows"], summary["imageFrames"], summary["audioModelSamples"]), (4, 1, 137))
            self.assertEqual([kind for kind, _shape in brain.modalities.seen], ["image", "audio", "audio", "audio"])
            self.assertEqual(brain.text_seen, ["literal before", "literal after"])
            self.assertEqual(len(brain.sensory_admissions), 1)
            self.assertEqual(brain.calls, [])
            self.assertEqual(brain.counters["experiences"], 1)
            self.assertEqual([(module, module.training) for module in brain.modalities.modules()], modes)
            path = verify_audit_file(brain.engine_path, audit, policy=brain.resource_policy)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows], ["text", "image", "audio", "resource-text"])
            self.assertEqual([row["blockIndex"] for row in rows], list(range(4)))
            self.assertEqual(rows[1]["sourceSha256"], hashlib.sha256(codec_fixture.image_bytes()).hexdigest())
            self.assertEqual(rows[2]["sourceSha256"], hashlib.sha256(codec_fixture.wav_bytes()).hexdigest())
            self.assertTrue(all(lease.released for lease in brain.resource_policy.live))

    def test_media_only_has_no_caption_and_duplicate_recovers_audit_without_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            wire = observation(content=codec_fixture.content("image", "image/png", codec_fixture.image_bytes()))
            result = admit_native_app_observation(brain, wire)
            self.assertEqual(result["head"]["learning"]["textBytes"], 0)
            self.assertEqual(result["head"]["learning"]["textSha256"], hashlib.sha256(b"").hexdigest())
            self.assertEqual(brain.text_seen, [])
            with patch.object(brain.modalities, "perception_embedding", side_effect=AssertionError("duplicate decoded again")), \
                    patch.object(brain, "_admit_sensory_embedding", side_effect=AssertionError("duplicate learned again")):
                duplicate = admit_native_app_observation(brain, wire, cancelled=lambda: True)
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(duplicate["head"]["content"]["audit"], result["head"]["content"]["audit"])

    def test_reference_and_structured_data_are_ordered_unfetched_data(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            wire = observation(method="ui/update-model-context", content=[{"type": "resource_link", "uri": "https://example.invalid/ref", "name": "Reference"}])
            result = admit_native_app_observation(brain, wire)
            summary = result["head"]["content"]["summary"]
            self.assertEqual((summary["referenceBlocks"], summary["structuredBlocks"], summary["blockCount"]), (1, 1, 2))
            self.assertFalse(summary["nativeSensory"])
            self.assertEqual(brain.modalities.seen, [])
            audit = result["head"]["content"]["audit"]
            rows = [json.loads(line) for line in verify_audit_file(brain.engine_path, audit).read_text().splitlines()]
            self.assertTrue(all(row["fetched"] is False and row["sourceSha256"] == wire["dataSha256"] for row in rows))

    def test_current_applied_audit_backpressures_next_and_new_stream_until_archived(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as archive:
            brain = ContentContractBrain(directory)
            first = mixed_wire()
            result = admit_native_app_observation(brain, first)
            audit = result["head"]["content"]["audit"]
            path = verify_audit_file(brain.engine_path, audit)
            next_wire = observation(ordinal=2, previous=first["recordSha256"], expected=result["frontierSha256"])
            for wire in (next_wire, observation(stream="new-stream", expected=result["frontierSha256"]),
                         {**next_wire, "previousAuditArchivedSha256": "f" * 64}):
                with self.assertRaisesRegex(ValueError, "durably archived"):
                    admit_native_app_observation(brain, wire)
            self.assertTrue(path.exists())
            metadata = {"brain_id": brain.brain_id, "app_observation_frontier": result["head"]}
            copy_app_observation_audit_snapshot(brain.engine_path, Path(archive), metadata)
            archived = app_observation_audit_snapshot_files(Path(archive), metadata)
            self.assertEqual(hashlib.sha256(next(iter(archived.values())).read_bytes()).hexdigest(), audit["sha256"])
            admit_native_app_observation(brain, {**next_wire, "previousAuditArchivedSha256": audit["sha256"]})
            self.assertFalse(path.exists())
            self.assertTrue(next(iter(archived.values())).exists())

    def test_copy_recovery_preserves_original_audit_owner_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as copy_root:
            brain = ContentContractBrain(directory)
            result = admit_native_app_observation(brain, mixed_wire())
            metadata = {"brain_id": "copy-owner", "app_observation_frontier": copy.deepcopy(result["head"])}
            copy_app_observation_audit_snapshot(brain.engine_path, Path(copy_root), metadata)
            copied = app_observation_audit_snapshot_files(Path(copy_root), metadata)
            self.assertEqual(len(copied), 1)
            self.assertEqual(metadata["app_observation_frontier"]["content"]["audit"]["sourceBrainId"], brain.brain_id)
            self.assertEqual(next(iter(copied.values())).read_bytes(), verify_audit_file(brain.engine_path, result["head"]["content"]["audit"]).read_bytes())

    def test_decode_resource_rejection_and_cancel_do_not_claim_learning(self):
        for mode in ("resource", "cancel", "invalid-signal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                brain = ContentContractBrain(directory)
                if mode == "resource":
                    brain.resource_policy.reject_pixels = True
                if mode == "invalid-signal":
                    brain.modalities.perception_embedding = lambda *_args: torch.zeros(1, 16)
                error = NeuralStateResourcePause if mode == "resource" else InterruptedError if mode == "cancel" else ValueError
                with self.assertRaises(error):
                    admit_native_app_observation(brain, mixed_wire(), cancelled=lambda: mode == "cancel")
                self.assertEqual(brain.counters["experiences"], 0)
                self.assertEqual(brain.sensory_admissions, [])
                self.assertIsNone(brain.app_observation_frontier)
                self.assertEqual(list((brain.engine_path / "app-observation-audits").iterdir()), [])
                self.assertTrue(all(lease.released for lease in brain.resource_policy.live))

    def test_unpublished_failure_removes_only_own_audit_and_fences_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            with patch.object(brain, "save", side_effect=OSError("not published")):
                with self.assertRaises(AppObservationCommitError):
                    admit_native_app_observation(brain, mixed_wire())
            self.assertTrue(brain._connected_recovery_required)
            self.assertIsNone(brain.app_observation_frontier)
            self.assertEqual(list((brain.engine_path / "app-observation-audits").iterdir()), [])

    def test_published_cleanup_failure_and_late_cancel_keep_recoverable_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            save = brain.save
            def published_then_failed():
                save()
                raise OSError("post-publication cleanup")
            wire = mixed_wire()
            with patch.object(brain, "save", side_effect=published_then_failed):
                result = admit_native_app_observation(brain, wire)
            self.assertTrue(result["reloadRequired"])
            self.assertTrue(verify_audit_file(brain.engine_path, result["head"]["content"]["audit"]).exists())
            brain._connected_recovery_required = False  # synthetic explicit Load
            before = len(brain.modalities.seen)
            duplicate = admit_native_app_observation(brain, wire, cancelled=lambda: True)
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(len(brain.modalities.seen), before)

    def test_current_audit_tamper_is_rejected_not_relearned(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            wire = mixed_wire()
            result = admit_native_app_observation(brain, wire)
            path = verify_audit_file(brain.engine_path, result["head"]["content"]["audit"])
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                admit_native_app_observation(brain, wire)
            self.assertEqual(len(brain.sensory_admissions), 1)

    def test_connected_mutation_prepares_publication_before_same_root_save(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            brain._connected_regions = object()
            calls = []
            class Transaction:
                def __init__(self, owner): self.owner = owner
                def __enter__(self): calls.append("transaction"); return self
                def prepare_publication(self): calls.append("prepare-publication")
                def __exit__(self, kind, *_args): calls.append("close" if kind is None else "rollback")
            brain._idea_model_vector = lambda vector: vector.unsqueeze(0)
            brain.decoder = SimpleNamespace(observe_experience=lambda *_args, **_kwargs: calls.append("regional-observation"))
            save = brain.save
            def saved(): calls.append("root-save"); save()
            with patch("omni_core.connected_transaction.ConnectedExperienceTransaction", Transaction), patch.object(brain, "save", side_effect=saved):
                admit_native_app_observation(brain, mixed_wire())
            self.assertEqual(calls, ["transaction", "regional-observation", "prepare-publication", "root-save", "close"])

    def test_conditional_content_head_hash_matches_shared_utf8_fixture(self):
        z, a, b, d, e = (char * 64 for char in "0abde")
        head = {"format": "omni-native-app-observation-head", "version": 1,
            "streamId": "11111111-1111-1111-1111-111111111111", "ordinal": 1,
            "recordSha256": a, "dataSha256": b, "previousSha256": z, "baseFrontierSha256": z,
            "disposition": "observe", "state": "observed", "committedAt": "2026-10-11T12:00:00.000Z",
            "learning": {"kind": "experience", "source": "mcp-app", "sourceLabel": "reviewed-mcp-app:fixture/é", "steps": 0,
                "experienceCountBefore": 7, "experienceCountAfter": 8, "assemblyId": "assembly-one", "afterimageId": "scratch-one",
                "textSha256": hashlib.sha256(b"").hexdigest(), "textBytes": 0},
            "content": {"summary": {"format": "omni-native-app-content-summary", "version": 1, "blockCount": 1,
                "textBlocks": 0, "resourceTextBlocks": 0, "referenceBlocks": 0, "structuredBlocks": 0, "imageBlocks": 1, "audioBlocks": 0,
                "decodedWindows": 1, "imageFrames": 1, "audioModelSamples": 0, "sourceMediaBytes": 5,
                "orderedInputSha256": e, "wholeInputTraversed": True, "nativeSensory": True},
                "audit": {"format": "omni-native-app-observation-audit", "version": 1, "path": "app-observation-audits/" + d + ".jsonl",
                    "sha256": d, "bytes": 500, "rows": 1, "sourceBrainId": "fixture-owner", "recordSha256": a, "dataSha256": b, "orderedInputSha256": e}}}
        self.assertEqual(frontier_sha256(head), "1fb2e499884681f6a454e8b99408c7e349d20ac928613d7e96156f920f85f099")

    def test_serial_dispatch_cancels_only_read_phase_and_keeps_resident_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, brain = Worker(), ContentContractBrain(directory)
            worker.brains[brain.brain_id] = brain
            def reading(_brain, _params, *, cancelled):
                self.assertEqual(worker._active_request, ("admit_app_observation", "read-one"))
                self.assertTrue(worker.request_cooperative_cancel())
                self.assertTrue(cancelled())
                raise InterruptedError("source reading stopped")
            try:
                with patch.object(worker, "_get", return_value=brain), patch.object(worker, "_execution_store") as store, \
                        patch("worker_app_observation.admit_native_app_observation", side_effect=reading):
                    store.return_value.current.return_value = None
                    with self.assertRaises(RpcFault) as caught:
                        worker.dispatch({"jsonrpc": "2.0", "id": "read-one", "method": "admit_app_observation",
                            "params": {**mixed_wire(), "storagePath": directory}})
                self.assertEqual(caught.exception.code, -32800)
                self.assertFalse(caught.exception.data["learningApplied"])
                self.assertFalse(caught.exception.data["nativeOwnerReleased"])
                self.assertIs(worker.brains[brain.brain_id], brain)
                self.assertFalse(brain.closed)
                self.assertIsNone(worker._active_request)
                self.assertFalse(worker._cooperative_cancel.is_set())
            finally:
                worker.shutdown({}, None)

    def test_unknown_publication_keeps_audit_and_never_claims_nothing_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            save = brain.save
            def uncertain():
                save()
                raise OSError("acknowledgement uncertain")
            wire = mixed_wire()
            with patch.object(brain, "save", side_effect=uncertain), \
                    patch("omni_core.app_observation._durable_head_is", side_effect=OSError("metadata temporarily unavailable")):
                with self.assertRaises(AppObservationCommitError) as caught:
                    admit_native_app_observation(brain, wire)
            self.assertIsNone(caught.exception.committed)
            self.assertTrue(brain._connected_recovery_required)
            saved = json.loads((brain.engine_path / "brain.json").read_text())
            self.assertTrue(verify_audit_file(brain.engine_path, saved["app_observation_frontier"]["content"]["audit"]).exists())
            brain.app_observation_frontier = saved["app_observation_frontier"]
            brain._connected_recovery_required = False  # synthetic explicit Load
            self.assertTrue(admit_native_app_observation(brain, wire)["duplicate"])
            self.assertEqual(len(brain.sensory_admissions), 1)

    def test_committed_but_missing_audit_reports_applied_repair_not_relearning(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = ContentContractBrain(directory)
            save = brain.save
            def damaged_after_publication():
                save()
                audit = brain.app_observation_frontier["content"]["audit"]
                verify_audit_file(brain.engine_path, audit).unlink()  # owned corruption fixture
                raise OSError("committed audit was removed")
            with patch.object(brain, "save", side_effect=damaged_after_publication):
                with self.assertRaises(AppObservationCommitError) as caught:
                    admit_native_app_observation(brain, mixed_wire())
            self.assertIs(caught.exception.committed, True)
            self.assertTrue(brain._connected_recovery_required)
            self.assertEqual(len(brain.sensory_admissions), 1)


if __name__ == "__main__":
    unittest.main()
