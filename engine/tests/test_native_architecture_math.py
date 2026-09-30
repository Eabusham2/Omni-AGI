"""Source-derived arithmetic/config contracts, without any neural constructor."""

import copy
import json
import unittest
from pathlib import Path

from omni_core.config import OmniConfig
from omni_core.native_architecture import (
    native_architecture_sha256, native_core_inventory, validate_native_architecture,
)
from omni_core.native_compute_profile import profile_native_projection_compute
from worker import Worker, RpcFault


def fixture_descriptor():
    shape = {"dModel": 64, "layers": 2, "feedForward": 192, "nHeads": 4,
             "vsaDimensions": 256, "routerNeurons": 64, "modalityChannels": 16,
             "imageSize": 16, "audioSamples": 256, "videoFrames": 4,
             "workingMemoryItems": 32768, "workspaceLatents": 8192,
             "vocabSize": 261, "liquidMode": "cfc"}
    value = {"format": "omni-main-selected-native-architecture", "formatVersion": 1,
             "architecture": "OmniCortex", "externalPretrainedWeights": False,
             "hardwareTier": "personal", "shape": shape, "inventory": native_core_inventory(shape),
             "sizing": {"policy": "fixture", "selectedSystemRamBudgetBytes": 8589934592},
             "qualityEvidence": "unmeasured-native-quality-deferred"}
    value["sha256"] = native_architecture_sha256(value)
    return value


class NativeArchitectureMathTests(unittest.TestCase):
    def test_all_four_recorded_initial_inventories_match_exact_row_arithmetic(self):
        path = Path(__file__).parents[2] / "architecture" / "omnicortex-ground-up-v1.json"
        profiles = json.loads(path.read_text())["profiles"]
        media = {"micro": (8, 64, 2, 8), "personal": (16, 256, 4, 16),
                 "gpu": (32, 512, 6, 24), "workstation": (32, 1024, 8, 32)}
        for tier, source in profiles.items():
            image, audio, frames, channels = media[tier]
            shape = {"dModel": source["dModel"], "layers": source["layers"],
                     "feedForward": source["feedForward"], "vsaDimensions": source["vsaDimensions"],
                     "routerNeurons": source["routerNeurons"], "modalityChannels": channels,
                     "imageSize": image, "audioSamples": audio, "videoFrames": frames,
                     "workspaceLatents": 8, "vocabSize": 261, "liquidMode": "cfc"}
            count = native_core_inventory(shape)
            self.assertEqual(count["logicalParameters"] - 8 * shape["dModel"], source["baseParametersExcludingWorkspaceLatents"])
            self.assertEqual(count["projectionParameters"], source["exactPackedProjectionParameters"])
            self.assertEqual(count["tableParameters"] - 8 * shape["dModel"], source["exactPackedTableParametersExcludingWorkspaceLatents"])
            self.assertEqual(count["packedWeightBytes"] - 8 * ((shape["dModel"] + 3) // 4), source["packedBytesExcludingWorkspaceLatents"])

    def test_closed_form_and_resistance_owner_counts_match_general_inventory(self):
        shape = fixture_descriptor()["shape"]
        shape.update(dModel=48, feedForward=147, layers=3, nHeads=4,
                     vsaDimensions=192, routerNeurons=26, modalityChannels=9,
                     imageSize=12, audioSamples=100, videoFrames=3)
        d, ff, layers, r, c = (shape[key] for key in ("dModel", "feedForward", "layers", "routerNeurons", "modalityChannels"))
        w, s, a, f = shape["workspaceLatents"], shape["imageSize"] // 4, shape["audioSamples"] // 4, shape["videoFrames"]
        expected = ((4 * layers + 31) * d * d + 3 * layers * d * ff
                    + (w + 2 * layers + 2254) * d + 2 * r * d + r * r + r
                    + d * c * (a + f * s * s + 8) + 162 * c * c
                    + c * (368 + (f + 1) * s * s + 2 * a) + 197119)
        counts = native_core_inventory(shape)
        self.assertEqual(counts["logicalParameters"], expected)
        self.assertGreater(counts["packedWeightBytes"] * 4, expected)
        self.assertEqual(counts["resistanceBytes"], w + 28 * d + r + layers * (5 * d + 2 * ff + 2)
                         + c * (a + f * s * s + 97) + 1524)
        self.assertEqual(counts["packedOwners"], 92 + 6 * layers)
        ltc = native_core_inventory({**shape, "liquidMode": "ltc"})
        self.assertEqual(ltc["logicalParameters"] - expected, -d * d + 2 * d)

    def test_hash_matches_mirrored_node_fixture(self):
        descriptor = fixture_descriptor()
        self.assertEqual(descriptor["sha256"], "7caef23bd630bb834fe0c03bad59472466ebe4814e423060b872f1cb9e9dc7a2")
        self.assertEqual(validate_native_architecture(descriptor), descriptor)

    def test_main_shape_persists_and_renderer_descriptor_is_rejected(self):
        descriptor = fixture_descriptor()
        config = OmniConfig.from_external({"hardwareTier": "personal", "workingMemorySlots": 32768}, native_architecture=descriptor)
        self.assertEqual(config.d_model, 64)
        self.assertEqual(config.working_memory_slots, 32768)
        self.assertEqual(OmniConfig.from_dict(config.to_dict()).native_architecture, descriptor)
        with self.assertRaisesRegex(ValueError, "trusted main"):
            OmniConfig.from_external({"nativeArchitecture": descriptor})
        forged = copy.deepcopy(descriptor)
        forged["inventory"]["logicalParameters"] += 1
        forged["sha256"] = native_architecture_sha256(forged)
        with self.assertRaisesRegex(ValueError, "inventory"):
            OmniConfig.from_external({"hardwareTier": "personal"}, native_architecture=forged)
        changed = config.to_dict()
        changed["d_model"] += 32
        changed["idea_dim"] += 32
        with self.assertRaisesRegex(ValueError, "checkpoint dimensions"):
            OmniConfig.from_dict(changed)

    def test_legacy_exact_saved_shapes_not_resized_or_added_to_old_hash_payload(self):
        config = OmniConfig.from_dict({"d_model": 48, "idea_dim": 48, "n_heads": 4, "n_layers": 3, "d_ff": 147})
        self.assertEqual((config.d_model, config.n_layers, config.d_ff), (48, 3, 147))
        self.assertNotIn("native_architecture", config.to_dict())

    def test_designated_pool_and_zero_attention_allowance_roundtrip(self):
        config = OmniConfig.from_external({"storagePoolBytes": 4096, "contextOffloadBudgetBytes": 0})
        self.assertEqual(config.storage_pool_bytes, 4096)
        self.assertEqual(config.working_attention_scratch_budget_bytes, 0)
        self.assertEqual(OmniConfig.from_dict(config.to_dict()).working_attention_scratch_budget_bytes, 0)
        self.assertNotIn("working_attention_scratch_budget_bytes", OmniConfig().to_dict())

    def test_trusted_duplicate_worker_metadata_is_exact_and_embedded_only_rejected(self):
        descriptor = fixture_descriptor()
        raw = {"nativeArchitecture": descriptor, "hardwareTier": "personal", "workingMemorySlots": 32768}
        cleaned = Worker._native_builder_metadata({"nativeArchitecture": descriptor}, raw)
        self.assertNotIn("nativeArchitecture", cleaned)
        self.assertIn("nativeArchitecture", raw)
        resolved = OmniConfig.from_external(cleaned, native_architecture=descriptor)
        self.assertEqual(resolved.native_architecture["sha256"], descriptor["sha256"])
        with self.assertRaises(RpcFault):
            Worker._native_builder_metadata({}, raw)
        other = copy.deepcopy(descriptor)
        other["sizing"]["policy"] = "different-main-declaration"
        other["sha256"] = native_architecture_sha256(other)
        with self.assertRaises(RpcFault):
            Worker._native_builder_metadata({"nativeArchitecture": other}, raw)

    def test_generic_compute_profile_fixture_does_not_run_tensor_or_neural_code(self):
        ticks = iter([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
        calls, reserves = [], []
        result = profile_native_projection_compute(
            clock=lambda: next(ticks), primitive=lambda: calls.append(True),
            reserve=lambda count, device: reserves.append(count),
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["projectionMacsPerRun"], 8 * 64 ** 2)
        self.assertEqual(result["kernel"], "omni_core.model._packed_ternary_forward")
        self.assertEqual(result["implementation"], "injected-fixture")
        self.assertFalse(result["neuralModelConstructed"])
        self.assertFalse(result["neuralQualityMeasured"])
        self.assertLess(reserves[0], 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
