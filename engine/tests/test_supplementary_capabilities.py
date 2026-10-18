"""Real packed head updates on fixed numeric features, not a chat qualification."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder
from omni_core.persistence import tensor_checksum
from omni_core.supplementary_capabilities import (
    canonical_sha, teach_supplementary_capabilities, validated_capability_receipts,
)


class SupplementaryTeachingTests(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(1); torch.manual_seed(231)
        config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1, d_ff=16,
            vsa_dim=16, router_neurons=4, max_seq_len=128, working_memory_slots=4,
            modality_channels=4, image_size=8, audio_samples=16, video_frames=2)
        decoder = OmniDecoder(config)
        decoder.action_policy.hidden.eval()
        brain = SimpleNamespace(config=config, device=torch.device("cpu"), decoder=decoder,
            counters={"training_steps": 9}, supplementary_capability_receipts={},
            resource_policy=SimpleNamespace(reserve_ram=Mock(return_value=SimpleNamespace(release=Mock()))),
            _normalize_tool_schemas=AdaptiveBrain._normalize_tool_schemas,
            _tool_action_input_schema=AdaptiveBrain._tool_action_input_schema)
        body = {"format": "omni-declared-capability-pack", "version": 1, "id": "fixture-declared-pack-1",
            "sourceKind": "project-authored-declared-examples", "executedHostActions": False,
            "preferenceLabels": False, "rewardModel": False,
            "schemas": [{"id": "data.learn", "actions": ["status"], "grant": "ask",
                "actionInputSchemas": {"status": {"type": "object", "additionalProperties": False,
                    "properties": {"taskId": {"type": "string"}}, "required": ["taskId"]}}}],
            "records": [{"utterance": "Inspect the selected learning task.", "kind": "tool", "toolId": "data.learn",
                "action": "status", "arguments": {"taskId": "fixture-task"}}]}
        pack = {**body, "sha256": canonical_sha(body)}
        features = (torch.tensor([[.2, -.7, .9, -.2, .4, .1, -.3, .6]]),
            torch.tensor([[-.1, .5, .7, -.8, .3, -.2, .6, .4]]))
        return brain, pack, features

    def heads_checksum(self, brain):
        return tensor_checksum(value for name in ("action_policy", "internal_action_policy", "tool_route_head", "action_argument_head")
            for value in getattr(brain.decoder, name).state_dict().values())

    def test_real_packed_attempt_truthful_rollback_receipt_and_idempotent_retry(self):
        brain, pack, features = self.fixture()
        before = self.heads_checksum(brain)
        modes = [module.training for module in brain.decoder.modules()]
        with patch("omni_core.supplementary_capabilities._features", return_value=features) as read:
            receipt = teach_supplementary_capabilities(brain, pack, steps=2)
            self.assertNotEqual(receipt["learnedChecksumBefore"], receipt["learnedChecksumCandidate"])
            self.assertEqual(receipt["attemptedSteps"], 2)
            self.assertFalse(receipt["executedHostActions"])
            self.assertFalse(receipt["groundingEnabledByDeclaration"])
            self.assertEqual(tuple(brain.decoder.parameters()), ())
            self.assertEqual([module.training for module in brain.decoder.modules()], modes)
            if not receipt["candidateAccepted"]:
                self.assertEqual(self.heads_checksum(brain), before)
                self.assertEqual(receipt["ownerChecksumAfter"], receipt["ownerChecksumBefore"])
                self.assertEqual(brain.counters, {"training_steps": 9})
            state = self.heads_checksum(brain)
            duplicate = teach_supplementary_capabilities(brain, pack, steps=2)
            self.assertTrue(duplicate["duplicate"]); self.assertEqual(read.call_count, 1)
            self.assertEqual(self.heads_checksum(brain), state)
            self.assertEqual(validated_capability_receipts(brain.supplementary_capability_receipts), brain.supplementary_capability_receipts)

    def test_per_record_regression_restores_structural_route_growth_and_all_controls(self):
        brain, pack, features = self.fixture()
        before = self.heads_checksum(brain)
        from omni_core.supplementary_capabilities import _measure
        count = 0
        def measure(*args):
            nonlocal count
            result = _measure(*args); count += 1
            if count == 2: result["argumentLosses"][0] += 100
            return result
        with patch("omni_core.supplementary_capabilities._features", return_value=features), patch("omni_core.supplementary_capabilities._measure", side_effect=measure):
            receipt = teach_supplementary_capabilities(brain, pack, steps=1)
        self.assertFalse(receipt["candidateAccepted"]); self.assertTrue(receipt["rollbackVerified"])
        self.assertEqual(self.heads_checksum(brain), before)
        self.assertEqual(brain.decoder.tool_route_head.route_keys.shape[0], 0)
        self.assertEqual(brain.decoder.action_argument_head.route_keys.shape[0], 0)

    def test_failed_commit_and_cancellation_restore_bytes_counters_modes_and_receipts(self):
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                brain, pack, features = self.fixture(); before = self.heads_checksum(brain)
                modes = [module.training for module in brain.decoder.modules()]
                with patch("omni_core.supplementary_capabilities._features", return_value=features), self.assertRaisesRegex(RuntimeError, "boundary|commit"):
                    teach_supplementary_capabilities(brain, pack, steps=1, cancel_check=lambda: cancel,
                        commit=lambda _receipt: (_ for _ in ()).throw(RuntimeError("commit refused")))
                self.assertEqual(self.heads_checksum(brain), before)
                self.assertEqual(brain.supplementary_capability_receipts, {})
                self.assertEqual(brain.counters, {"training_steps": 9})
                self.assertEqual([module.training for module in brain.decoder.modules()], modes)

    def test_declared_placeholders_never_overwrite_confirmed_route_and_argument_owners(self):
        brain, pack, features = self.fixture()
        brain.decoder.tool_route_head.experience_updates.fill_(3)
        brain.decoder.action_argument_head.grounded_steps.fill_(2)
        from omni_core.supplementary_capabilities import _learned_checksum
        protected = {name: getattr(brain.decoder, name) for name in ("tool_route_head", "action_argument_head")}
        before = _learned_checksum(protected)
        with patch("omni_core.supplementary_capabilities._features", return_value=features):
            receipt = teach_supplementary_capabilities(brain, pack, steps=1)
        self.assertEqual(set(receipt["protectedOwners"]), set(protected))
        self.assertEqual(_learned_checksum(protected), before)
        self.assertEqual(int(brain.decoder.tool_route_head.experience_updates), 3)
        self.assertEqual(int(brain.decoder.action_argument_head.grounded_steps), 2)

    def test_invalid_hash_or_typed_record_rejected_before_mutation(self):
        brain, pack, _ = self.fixture(); before = self.heads_checksum(brain)
        for invalid in (dict(pack, sha256="0"*64), dict(pack, executedHostActions=True)):
            with self.assertRaises(ValueError): teach_supplementary_capabilities(brain, invalid)
        bad = copy.deepcopy(pack); bad["records"][0]["arguments"] = {"path": "/ungranted"}
        bad["sha256"] = canonical_sha({key: value for key, value in bad.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "arguments"): teach_supplementary_capabilities(brain, bad)
        self.assertEqual(self.heads_checksum(brain), before)


if __name__ == "__main__": unittest.main()
