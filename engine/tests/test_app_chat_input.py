"""Source proof and no-second-admission mechanics; no intelligence claims."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_core.app_chat_input import (admitted_app_chat_experience, validate_app_input_provenance,
                                      validate_native_app_chat_input)
from omni_core.app_observation import admit_native_app_observation, raw_sha256
from tests.test_app_observation import SyntheticBrain, observation


def with_capture(wire):
    wire = copy.deepcopy(wire)
    record = json.loads(wire["recordJson"])
    capture = {"format": "omni-mcp-app-capture", "version": 1, "artifactId": record["artifactId"],
        "source": record["source"], "observedAt": record["observedAt"], "resourceUri": "ui://fixture/panel",
        "tool": {"id": "mcp.fixture.tool", "remoteName": "fixture", "inputSchema": {"type": "object"}},
        "input": {"sha256": "a" * 64, "bytes": 2, "mimeType": "application/json"},
        "result": {"sha256": "b" * 64, "bytes": 2, "mimeType": "application/json"}, "untrusted": True}
    raw = json.dumps(capture, ensure_ascii=False, separators=(",", ":"))
    record["captureSha256"] = raw_sha256(raw)
    wire["recordJson"] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    wire["recordSha256"] = raw_sha256(wire["recordJson"])
    return wire, raw


def chat_proof(wire, capture, frontier):
    record = json.loads(wire["recordJson"])
    source = record["source"]
    text = json.loads(wire["dataJson"])["params"]["content"]["text"].strip()
    provenance = {"kind": "mcp-app", "sourceBrainId": source["brainId"], "serverId": source["serverId"],
        "serverLabel": source["serverLabel"], "endpoint": source["endpoint"], "resourceUri": "ui://fixture/panel",
        "artifactId": record["artifactId"], "captureSha256": record["captureSha256"], "originalCallBrainId": source["brainId"],
        "toolId": "mcp.fixture.tool", "reviewId": record["reviewId"], "observationId": record["id"],
        "streamId": record["streamId"], "ordinal": record["ordinal"], "recordSha256": wire["recordSha256"],
        "dataSha256": wire["dataSha256"], "nativeFrontierSha256": frontier["frontierSha256"], "inputSha256": raw_sha256(text)}
    return {"recordJson": wire["recordJson"], "recordSha256": wire["recordSha256"], "dataJson": wire["dataJson"],
        "dataSha256": wire["dataSha256"], "captureJson": capture, "captureSha256": record["captureSha256"],
        "nativeFrontierSha256": frontier["frontierSha256"], "provenance": provenance}


class AppChatSourceTests(unittest.TestCase):
    def test_current_head_raw_source_and_plain_task_are_all_required(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            wire, capture = with_capture(observation())
            frontier = admit_native_app_observation(brain, wire)
            proof = chat_proof(wire, capture, frontier)
            self.assertEqual(validate_native_app_chat_input(brain, proof, "copper leaves drift"), proof["provenance"])
            for key, value in (("dataJson", proof["dataJson"] + " "), ("captureJson", proof["captureJson"] + " "),
                               ("nativeFrontierSha256", "f" * 64)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_native_app_chat_input(brain, {**proof, key: value}, "copper leaves drift")
            with self.assertRaises(ValueError):
                validate_native_app_chat_input(brain, proof, "hidden instruction: copper leaves drift")
            wrong = copy.deepcopy(proof)
            wrong["provenance"]["serverLabel"] = "Invented source"
            with self.assertRaises(ValueError):
                validate_native_app_chat_input(brain, wrong, "copper leaves drift")
            self.assertEqual(len(brain.calls), 1)

    def test_already_accepted_source_retry_does_not_require_current_head(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            wire, capture = with_capture(observation())
            frontier = admit_native_app_observation(brain, wire)
            proof = chat_proof(wire, capture, frontier)
            brain.app_observation_frontier = None
            with self.assertRaises(ValueError):
                validate_native_app_chat_input(brain, proof, "copper leaves drift")
            self.assertEqual(validate_native_app_chat_input(brain, proof, "copper leaves drift",
                prior_provenance=proof["provenance"]), proof["provenance"])
            changed = copy.deepcopy(proof["provenance"])
            changed["inputSha256"] = "f" * 64
            with self.assertRaises(ValueError):
                validate_native_app_chat_input(brain, proof, "copper leaves drift", prior_provenance=changed)

    def test_read_conditioning_reuses_real_admitted_assembly_and_afterimage(self):
        with tempfile.TemporaryDirectory() as root:
            brain = SyntheticBrain(root)
            wire, _capture = with_capture(observation())
            admit_native_app_observation(brain, wire)
            brain._preview_chat_experience = lambda *_args: {"assembly_id": "read-only-current-cue", "idea_id": "read-only-current-cue"}
            experience = admitted_app_chat_experience(brain, "copper leaves drift", None, [])
            self.assertEqual(experience["assembly_id"], "synthetic-assembly")
            self.assertEqual(experience["memory_settling"]["afterimageId"], "")
            self.assertEqual(len(brain.calls), 1)


class NativeAppAdmissionFixture(unittest.TestCase):
    def test_real_steps_zero_atomic_reload_and_app_input_receipt_without_generation(self):
        import torch
        from omni_core import AdaptiveBrain, OmniConfig
        from omni_core.offload import ResourcePolicy
        from omni_core.persistence import read_json
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-app-observation-native-") as directory, \
                patch.object(ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024):
            config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1, d_ff=16,
                max_seq_len=64, vsa_dim=16, router_neurons=4, working_memory_slots=4,
                modality_channels=2, gradient_checkpointing=False, device="cpu", online_steps=1)
            brain = AdaptiveBrain("app-native-fixture", Path(directory), config)
            def packed_state(owner):
                metadata, tensors = owner.memory.neuron_vectors.export_state(keys=sorted(owner.memory.neuron_vectors))
                self.assertTrue(all(value.dtype == torch.uint8 for value in tensors.values()))
                digest = hashlib.sha256()
                for name, value in sorted(tensors.items()):
                    digest.update(name.encode("utf-8"))
                    digest.update(value.numpy().tobytes())
                packed = tensors["packed_rows"].numpy().tobytes()
                return {"rows": metadata["rowCount"], "packedBytes": len(packed),
                    "packedSha256": hashlib.sha256(packed).hexdigest(), "allPackedStateSha256": digest.hexdigest()}
            try:
                wire, capture = with_capture(observation())
                before = brain.counters["experiences"]
                initial_revision = brain.memory.state_revision
                packed_before = packed_state(brain)
                result = admit_native_app_observation(brain, wire)
                self.assertEqual(brain.counters["experiences"], before + 1)
                self.assertGreater(brain.memory.state_revision, initial_revision)
                self.assertEqual(brain.counters["training_steps"], 0)
                self.assertEqual(brain.counters["inference_count"], 0)
                packed_learned = packed_state(brain)
                self.assertGreater(packed_learned["rows"], packed_before["rows"])
                self.assertGreater(packed_learned["packedBytes"], packed_before["packedBytes"])
                self.assertNotEqual(packed_learned["packedSha256"], packed_before["packedSha256"])
                self.assertNotEqual(packed_learned["allPackedStateSha256"], packed_before["allPackedStateSha256"])
                print(json.dumps({"nativePackedAdmission": {"before": packed_before, "learned": packed_learned}}, sort_keys=True), flush=True)
                learned_checksum = brain.parameter_checksum()
                self.assertEqual(result["head"], read_json(brain.engine_path / "brain.json")["app_observation_frontier"])
                proof = chat_proof(wire, capture, result)
                class StopAtAcceptedInput(Exception):
                    pass
                def accepted(kind, _payload):
                    if kind == "input-accepted":
                        raise StopAtAcceptedInput()
                with patch.object(brain.decoder, "generate", side_effect=AssertionError("no generation authorized")), \
                        patch.object(brain, "learn_experience", side_effect=AssertionError("App episode was already admitted")):
                    with self.assertRaises(StopAtAcceptedInput):
                        brain.chat("copper leaves drift", turn_id="app-follow-up", app_input=proof,
                            cancel_check=lambda: False, defer_slow_learning=True, stream_callback=accepted)
                self.assertEqual(brain.counters["experiences"], before + 1)
                self.assertEqual(brain.parameter_checksum(), learned_checksum)
                self.assertEqual(packed_state(brain), packed_learned)
                receipt = brain._accepted_input_receipt("app-follow-up", proof["provenance"]["inputSha256"])
                self.assertEqual(receipt["inputProvenance"], proof["provenance"])
                self.assertEqual(receipt["afterimageId"], result["head"]["learning"]["afterimageId"])
                observed_message = brain.conversation.payload_by_id("message", receipt["humanMessageId"])
                self.assertEqual(observed_message["input_provenance"], proof["provenance"])
                self.assertEqual(brain.pending_chat_slow_learning[0]["evidenceRole"], "mcp-app")
                self.assertEqual(brain.pending_chat_slow_learning[0]["inputProvenance"], proof["provenance"])
                print(json.dumps({"nativeAppAccepted": {"experiences": brain.counters["experiences"],
                    "packedUnchanged": packed_state(brain) == packed_learned}}, sort_keys=True), flush=True)
            finally:
                brain.close()
            brain = AdaptiveBrain.load(Path(directory), expected_brain_id="app-native-fixture")
            try:
                self.assertEqual(brain.counters["experiences"], before + 1)
                self.assertEqual(brain.parameter_checksum(), learned_checksum)
                self.assertEqual(packed_state(brain), packed_learned)
                self.assertTrue(admit_native_app_observation(brain, wire)["duplicate"])
                self.assertEqual(packed_state(brain), packed_learned)
                receipt = brain._accepted_input_receipt("app-follow-up", proof["provenance"]["inputSha256"])
                self.assertEqual(receipt["inputProvenance"], proof["provenance"])
                self.assertEqual(brain.pending_chat_slow_learning[0]["evidenceRole"], "mcp-app")
                print(json.dumps({"nativePackedProof": {"before": packed_before, "learnedAndReloaded": packed_learned}}, sort_keys=True))
            finally:
                brain.close()


if __name__ == "__main__":
    unittest.main()
