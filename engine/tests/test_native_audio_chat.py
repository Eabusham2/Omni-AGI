"""Constructor-free native audio contracts; encoder/admission are synthetic."""
import copy
from contextlib import contextmanager
import hashlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

import torch
from torch import nn

from omni_core.native_audio_chat import (audio_identity_sha256, validate_native_audio_input,
    validate_native_audio_receipt, observed_input_sha256, encode_native_audio,
    admit_native_audio_experience, accepted_audio_cue)
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.temporary_steering_context import temporary_steering_inputs


def identity(payload=b"fixture", **capture):
    digest = hashlib.sha256(payload).hexdigest()
    row = {"format": "omni-native-audio-input", "version": 1,
        "attachment": {"brainId": "audio-brain", "turnId": "audio-turn", "attachmentId": "audio-attachment",
            "contentSha256": digest, "bytes": len(payload), "mimeType": "audio/wav", "kind": "audio",
            "relativePath": "attachments/blobs/" + digest + ".data", "displayName": "Microphone audio", "manifestSha256": "b" * 64},
        "capture": {"sessionId": "mic-lease", "utteranceId": "utterance-one", "sequence": 0,
            "capturedAt": "2026-10-11T14:00:00Z", "permissionRevision": 1, **capture}}
    row["inputSha256"] = audio_identity_sha256(row)
    return row


def wave_bytes(count=137):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes(b"".join(int(math.sin(i / 9.) * 20000).to_bytes(2, "little", signed=True) for i in range(count)))
    return stream.getvalue()


class Policy:
    def __init__(self):
        self.charges = []
        self.refuse = False
    @contextmanager
    def reserve_ram(self, amount, label):
        self.charges.append((amount, label))
        if self.refuse:
            raise MemoryError("synthetic resource admission failure")
        yield
    def check_active_pressure(self, *_args):
        return None
    def status(self, **_kwargs):
        return {"memoryPressure": False}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def perception_embedding(self, kind, tensor):
        assert kind == "audio"
        self.calls += 1
        # Explicit synthetic activity, not a trained native model.
        return torch.tensor([[1., float(tensor.abs().mean()) + .1, .5, .25]])


class Space:
    def symbol(self, _position):
        return torch.ones(8)
    def bind(self, vector, position):
        return vector * position
    def permute(self, vector, position):
        return vector.roll(position)


class NativeAudioContractTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-audio-contract-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        payload = wave_bytes()
        data = identity(payload)
        path = root / data["attachment"]["relativePath"]
        path.parent.mkdir(parents=True); path.write_bytes(payload)
        brain = SimpleNamespace(brain_id="audio-brain", storage_path=root, resource_policy=Policy(), device=torch.device("cpu"),
            config=SimpleNamespace(audio_enabled=True, vsa_dim=8, idea_dim=4, audio_samples=64), modalities=Encoder(),
            memory=SimpleNamespace(space=Space(), neuron_vectors={}), working_attention_pager=None)
        brain._sensory_substrate_vector = lambda vector, _kind, _sha: vector.flatten().repeat(2)
        return brain, data, path

    def receipt(self, data):
        return {"input": data, "assemblyId": "audio-assembly", "sourceBytes": data["attachment"]["bytes"],
            "sourceSha256": data["attachment"]["contentSha256"], "decodedWindows": 3, "modelSamples": 137,
            "modelSampleRate": 16000, "wholeInputTraversed": True, "nativeSensory": True, "transcriptUsed": False}

    def test_typed_audio_hash_not_empty_text_or_display_caption(self):
        data = identity()
        self.assertEqual(validate_native_audio_input(data, brain_id="audio-brain", turn_id="audio-turn"), data)
        self.assertNotEqual(data["inputSha256"], hashlib.sha256(b"").hexdigest())
        self.assertNotEqual(data["inputSha256"], hashlib.sha256(b"Microphone audio").hexdigest())
        changed = copy.deepcopy(data); changed["capture"]["sequence"] = 1
        with self.assertRaises(ValueError): validate_native_audio_input(changed)
        with self.assertRaises(ValueError): validate_native_audio_input(data, brain_id="other-brain")

    def test_transcript_and_path_substitution_rejected(self):
        data = identity()
        changed = copy.deepcopy(data); changed["attachment"]["relativePath"] = "../secret"
        changed["inputSha256"] = audio_identity_sha256(changed)
        with self.assertRaises(ValueError): validate_native_audio_input(changed)
        with self.assertRaises(ValueError): observed_input_sha256({"content": "invented transcript", "native_audio_input": self.receipt(data)})

    def test_actual_codec_all_samples_synthetic_encoder_no_admission(self):
        brain, data, path = self.fixture()
        before = path.read_bytes()
        with encode_native_audio(brain, data) as (cue, receipt):
            self.assertEqual(cue.shape, (8,)); self.assertTrue(bool(torch.isfinite(cue).all()))
            self.assertEqual(receipt["modelSamples"], 137)
            self.assertEqual(receipt["sourceSampleFrames"], 137)
            self.assertEqual(receipt["decodedWindows"], 3)
            self.assertEqual(brain.modalities.calls, 3)
            self.assertTrue(receipt["nativeSensory"]); self.assertFalse(receipt["transcriptUsed"])
        self.assertTrue(brain.modalities.training)
        self.assertEqual(path.read_bytes(), before)
        self.assertGreater(len(brain.resource_policy.charges), 3)

    def test_source_tamper_and_resource_refusal_before_encoder(self):
        brain, data, path = self.fixture()
        path.write_bytes(b"x" * data["attachment"]["bytes"])
        with self.assertRaises(ValueError):
            with encode_native_audio(brain, data): pass
        self.assertEqual(brain.modalities.calls, 0)
        brain, data, _path = self.fixture()
        brain.resource_policy.refuse = True
        with self.assertRaises(MemoryError):
            with encode_native_audio(brain, data): pass
        self.assertEqual(brain.modalities.calls, 0)

    def test_cancel_is_read_boundary_and_original_source_retained(self):
        brain, data, path = self.fixture()
        with self.assertRaises(InterruptedError):
            with encode_native_audio(brain, data, cancelled=lambda: True): pass
        self.assertTrue(path.is_file()); self.assertTrue(brain.modalities.training)
        self.assertEqual(brain.modalities.calls, 0)

    def test_one_synthetic_ordinary_admission_no_text_learning(self):
        brain, data, _path = self.fixture()
        calls = []
        settling = {"afterimageId": "audio-afterimage"}
        def admit(vector, **kwargs):
            calls.append(kwargs)
            return {"assemblyId": "audio-assembly", "vector": vector, "novelty": .8, "memorySettling": settling}
        brain._admit_sensory_embedding = admit
        brain._preview_chat_experience = lambda *_args: {"labels": ["placeholder"], "concept_ids": ["placeholder"]}
        brain.learn_experience = lambda *_args, **_kwargs: self.fail("audio cannot be a fabricated text lesson")
        with encode_native_audio(brain, data) as (cue, decoded):
            preview, receipt = admit_native_audio_experience(brain, data, cue, decoded, [])
        self.assertEqual(len(calls), 1); self.assertTrue(calls[0]["substrate_space"])
        self.assertEqual(calls[0]["experience_source"], "conversation-audio")
        self.assertEqual(preview["labels"], []); self.assertEqual(preview["concept_ids"], [])
        self.assertEqual(preview["memory_settling"], settling)
        self.assertEqual(validate_native_audio_receipt(receipt)["input"], data)

    def test_accepted_retry_reads_packed_assembly_not_source_or_decoder(self):
        brain, data, path = self.fixture()
        receipt = self.receipt(data)
        vector = torch.tensor([1., 0., -1., 0., 1., 0., -1., 0.])
        brain.memory.neuron_vectors["audio-assembly"] = vector
        path.unlink()  # Retry uses admitted neural conditioning, not source retrieval.
        cue, restored = accepted_audio_cue(brain, {"inputSha256": data["inputSha256"], "nativeAudioInput": receipt})
        self.assertIs(cue, vector); self.assertEqual(restored, receipt)
        self.assertEqual(brain.modalities.calls, 0)

    def test_exact_suspended_identity_original_owner_preserved(self):
        data = identity(); receipt = self.receipt(data)
        presentation = {"input": "", "inputReceipt": {"nativeAudioInput": receipt}}
        self.assertEqual(ExecutionCheckpointStore._presentation_input_sha256(presentation), data["inputSha256"])
        copied = copy.deepcopy(presentation)
        self.assertEqual(copied["inputReceipt"]["nativeAudioInput"]["input"]["attachment"]["brainId"], "audio-brain")
        copied["input"] = "Audio message"
        with self.assertRaises(ValueError): ExecutionCheckpointStore._presentation_input_sha256(copied)

    def test_temporary_audio_steer_is_typed_not_prose_and_fresh_discards_it(self):
        data = identity()
        carry = {"format": "omni-temporary-steering-input-v1", "brainId": "audio-brain", "successorTurnId": "next-turn",
                 "attentionEpoch": 0, "inputs": [], "audioInputs": [data]}
        text, audit = temporary_steering_inputs(carry, brain_id="audio-brain", turn_id="next-turn", attention_epoch=0, policy=Policy())
        self.assertEqual(text, []); self.assertEqual(audit["audioInputs"], [{"turnId": "audio-turn", "inputSha256": data["inputSha256"]}])
        self.assertFalse(audit["durablePredecessorLearning"])
        text, audit = temporary_steering_inputs(carry, brain_id="audio-brain", turn_id="next-turn", attention_epoch=1, policy=Policy())
        self.assertEqual(text, []); self.assertTrue(audit["freshAttentionDiscarded"])
        with self.assertRaises(ValueError): temporary_steering_inputs(carry, brain_id="other-brain", turn_id="next-turn", attention_epoch=0, policy=Policy())


if __name__ == "__main__":
    unittest.main()
