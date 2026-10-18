"""Tiny packed-owner receipt checks, without creating/training a brain."""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.ground_up import (
    GROUND_UP_ACTION_EXAMPLES, GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_TRAJECTORIES, GROUND_UP_V3_SOURCE_RECORDS,
    current_ground_up_curriculum_manifest, current_ground_up_training_receipt_contract,
    seal_ground_up_v3_training_manifest, seal_ground_up_v3_training_receipt,
)
from omni_core.model import PackedAdaptiveBitLinear
from omni_core.spiking import STDPSynapses


def fixture():
    # __init__/create are deliberately not called. These are tiny packed
    # owners and an explicit synthetic receipt, not a trained saved instance.
    brain = AdaptiveBrain.__new__(AdaptiveBrain)
    brain.config = OmniConfig.micro(vision_enabled=False, image_enabled=False, audio_enabled=False, video_enabled=False)
    brain.decoder = PackedAdaptiveBitLinear(3, 2, bias=False)
    brain.router = nn.Module()
    brain.router.synapses = STDPSynapses(4, 4)
    brain.router.population = SimpleNamespace(membrane=torch.zeros(4))
    brain.memory_bridge = brain.router  # Exercise unique-owner accounting.
    brain.idea_adapter = nn.Identity()
    brain.liquid = nn.Identity()
    brain.modalities = nn.Module()
    brain.modalities.imagination_selector = nn.Identity()
    brain.resource_policy = SimpleNamespace(status=lambda **_kwargs: {"memoryPressure": False})
    brain.memory = SimpleNamespace(neurons={}, assemblies={}, synapses={}, neuron_vectors={}, space=SimpleNamespace(dimensions=4))
    initial_count = brain.parameter_accounting()["mutableDenseParameters"]
    before = brain.parameter_checksum()
    names_before = set(brain._named_native_core_tensors())
    brain.decoder.add_parallel_connection_(0, 1, sign=1, strength=2)
    brain.router.synapses._sparse_state.ensure(0, 0)
    brain.router.synapses._sparse_state.ensure(0, 0, layer=1)
    brain.memory.neurons = {"a": {}, "b": {}, "c": {}}
    brain.memory.assemblies = {"a": {}, "b": {}}
    brain.memory.synapses = {"a": {}}
    brain.replay = []
    brain.working_memory = []
    brain.workspace_items = []
    brain.paged_working_memory = SimpleNamespace(count=lambda: 0)
    brain.recent_token_context = []
    brain.fresh_attention_boundary = None
    brain.current_context = {"tokenCount": 0, "sensorySlots": 0, "tokenHash": ""}
    brain.memory_lifecycle = SimpleNamespace(afterimage_items=[], active_focus=[])
    brain.liquid_state = torch.zeros(1)
    after = brain.parameter_checksum()
    names_after = set(brain._named_native_core_tensors())
    changed = sorted(names_after - names_before)
    contract = current_ground_up_training_receipt_contract()
    receipt = seal_ground_up_v3_training_receipt({
        **contract, "recordsVisited": GROUND_UP_V3_SOURCE_RECORDS,
        "recordGroupsVisited": copy.deepcopy(contract["recordGroupsExpected"]),
        "completeCoverage": True, "nativeCoreParameterTensors": len(names_after),
        "changedNativeCoreParameterTensors": len(changed), "changedNativeCoreTensorNames": changed,
        "parameterChecksumBefore": before, "parameterChecksumAfter": after, "parametersChanged": True,
        "modalityParameterChecksumBefore": brain._modality_parameter_checksum(),
        "modalityParameterChecksumAfter": brain._modality_parameter_checksum(), "modalityParametersChanged": False,
        "imaginationSelectorChecksumBefore": brain._imagination_selector_checksum(),
        "imaginationSelectorChecksumAfter": brain._imagination_selector_checksum(), "imaginationSelectorParametersChanged": False,
        "substrateBefore": {"neurons": 0, "assemblies": 0, "synapses": 0},
        "substrateAfter": {"neurons": 3, "assemblies": 2, "synapses": 1}, "substrateChanged": True,
    })
    cfg = brain.config
    manifest = seal_ground_up_v3_training_manifest({
        **current_ground_up_curriculum_manifest(), "originKind": "ground-up", "trainedAt": "2026-10-06T00:00:00Z",
        "randomInitialization": {"algorithm": "torch-seeded-module-initialization-v1", "seed": cfg.seed,
            "parameterChecksum": before, "exactParameterCount": initial_count},
        "architectureScale": {"hardwareTier": cfg.hardware_tier, "dimensions": cfg.d_model, "layers": cfg.n_layers,
            "feedForward": cfg.d_ff, "vsaDimensions": cfg.vsa_dim, "routerNeurons": cfg.router_neurons,
            "denseParameterCount": initial_count, "growthCardinalityLimit": None, "growthBoundary": "live-resource-watermark",
            "diskStateOffload": cfg.disk_state_offload},
        "modalityTraining": {"source": "selected-user-data-only", "enabledModalities": [], "trainedModalities": [],
            "trainingRecords": 0, "steps": 0, "parametersChanged": False, "syntheticFixture": False},
        "transientStateReset": {"complete": True, "replayEntriesAfter": 0, "workingMemoryVectorsAfter": 0,
            "pagedWorkingMemoryAfter": 0, "recentTokensAfter": 0, "lifecycleScratchAfter": 0, "lifecycleFocusAfter": 0,
            "currentContextTokensAfter": 0, "currentContextSensorySlotsAfter": 0, "freshAttentionBoundaryAfter": None,
            "liquidStateAbsoluteSumAfter": 0.0, "routerMembraneAbsoluteSumAfter": 0.0, "routerPreTraceAbsoluteSumAfter": 0.0,
            "routerPostTraceAbsoluteSumAfter": 0.0},
        "toolCurriculum": {"recordsVisited": len(GROUND_UP_TOOL_TRAJECTORIES) + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES), "perActionCoverage": True},
        "actionTraining": {"examples": len(GROUND_UP_ACTION_EXAMPLES)},
        "publicCapabilityReadiness": {"phase": "initial-neural-tool-curriculum", "curriculumVersion": 3,
            "toolTrajectoriesTrained": len(GROUND_UP_TOOL_TRAJECTORIES), "negativeExamplesTrained": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "capabilityOutcomeVerified": False, "readinessProbeExecuted": False, "readinessProbesAreOptimizerInputs": False,
            "systemPrompt": False, "toolDescriptionProse": False, "rewardModel": False, "rlhf": False},
        "trainingReceipt": receipt,
    })
    return brain, manifest


class GroundUpReceiptAccountingTests(unittest.TestCase):
    def test_initial_count_excludes_new_sparse_blocks_and_repeat_layers_but_counts_live_state(self):
        brain, manifest = fixture()
        accounting = brain.parameter_accounting()
        self.assertEqual(manifest["randomInitialization"]["exactParameterCount"], 6)
        self.assertEqual(accounting["mutableDenseParameters"], 6 + 1 + 32)
        self.assertEqual(brain.router.synapses.logical_ternary_parameter_count, 32)
        self.assertEqual(brain._validate_ground_up_v3_training_manifest(manifest), manifest)

    def test_resealed_wrong_initial_counts_still_fail(self):
        for section, field in (("randomInitialization", "exactParameterCount"), ("architectureScale", "denseParameterCount")):
            brain, manifest = fixture()
            manifest[section][field] += 1
            altered = seal_ground_up_v3_training_manifest(manifest)
            with self.subTest(section=section), self.assertRaisesRegex(RuntimeError, "architecture binding"):
                brain._validate_ground_up_v3_training_manifest(altered)

    def test_sparse_signs_and_strengths_are_still_bound_to_the_exact_receipt(self):
        for field in ("_packed_weights", "_connection_state"):
            brain, manifest = fixture()
            block = brain.router.synapses.blocks["r0_c0_p1"]
            getattr(block, field).view(-1)[0] ^= 1
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "architecture binding"):
                brain._validate_ground_up_v3_training_manifest(manifest)

    def test_removing_allocated_connections_does_not_bypass_live_checksum_or_tensor_inventory(self):
        brain, manifest = fixture()
        del brain.router.synapses.blocks["r0_c0_p1"]
        with self.assertRaisesRegex(RuntimeError, "architecture binding"):
            brain._validate_ground_up_v3_training_manifest(manifest)

    def test_record_coverage_and_architecture_dimensions_remain_required(self):
        brain, manifest = fixture()
        manifest["trainingReceipt"]["recordsVisited"] -= 1
        manifest["trainingReceipt"] = seal_ground_up_v3_training_receipt(manifest["trainingReceipt"])
        with self.assertRaisesRegex(RuntimeError, "manifest is invalid"):
            brain._validate_ground_up_v3_training_manifest(manifest)
        brain, manifest = fixture()
        manifest["architectureScale"]["dimensions"] += 1
        with self.assertRaisesRegex(RuntimeError, "architecture binding"):
            brain._validate_ground_up_v3_training_manifest(seal_ground_up_v3_training_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
