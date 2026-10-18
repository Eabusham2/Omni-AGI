"""Failure/retention transaction contract, not origin or fluency evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from omni_core.capability_rehearsal import CapabilityRehearsalPolicy
from omni_core.connected_capabilities import rehearse_inherited_capabilities, _rows_retained, preserved_feature_activity
from omni_core.ground_up import current_ground_up_curriculum_manifest
from omni_core.persistence import tensor_checksum
from tests.test_full_merge_regions import connected


class ConnectedCapabilityTransactionTests(unittest.TestCase):
    def test_feature_failure_preserves_working_router_liquid_and_attention(self):
        graph = connected()
        graph.regions[0].decoder.eval()
        memory = SimpleNamespace(_last_recall_audit={"old": True})
        region_memory = SimpleNamespace()
        workspace = [torch.tensor([1.0, 2.0])]
        lease = Mock()
        brain = SimpleNamespace(decoder=graph, liquid_state=torch.ones(1, 24),
            router=SimpleNamespace(population=SimpleNamespace(membrane=torch.tensor([0.25]), spike_count=torch.tensor([3]))),
            resource_policy=SimpleNamespace(reserve_ram=Mock(return_value=lease)), memory=memory,
            _connected_regions=SimpleNamespace(iter_memories=lambda: iter((("region", region_memory),))),
            working_memory=workspace, current_context={"tokens": 3})
        tensors = [brain.liquid_state, brain.router.population.membrane, brain.router.population.spike_count]
        tensors.extend(value for region in graph.regions for value in (region.liquid_state, region.router.population.membrane, region.router.population.spike_count))
        before = tensor_checksum(tensors)
        modes = [module.training for module in graph.modules()]
        with self.assertRaisesRegex(RuntimeError, "feature interruption"):
            with preserved_feature_activity(brain):
                for tensor in tensors:
                    tensor.add_(1)
                graph.eval()
                graph.last_region_activity = (torch.ones(2),)
                brain.working_memory = []
                brain.current_context = {"tokens": 99}
                memory._last_recall_audit = {"training": True}
                region_memory._last_recall_audit = {"training": True}
                raise RuntimeError("feature interruption")
        self.assertEqual(tensor_checksum(tensors), before)
        self.assertEqual([module.training for module in graph.modules()], modes)
        self.assertEqual(graph.last_region_activity, ())
        self.assertIs(brain.working_memory, workspace)
        self.assertEqual(brain.current_context, {"tokens": 3})
        self.assertEqual(memory._last_recall_audit, {"old": True})
        self.assertFalse(hasattr(region_memory, "_last_recall_audit"))
        lease.release.assert_called_once()

    def test_failure_and_declared_regression_restore_real_packed_heads(self):
        torch.set_num_threads(1)
        graph = connected()
        names = ("action_policy", "internal_action_policy", "tool_route_head", "action_argument_head")
        def checksum():
            return tensor_checksum(value for name in names for value in getattr(graph, name).state_dict().values())
        initial = checksum()
        leaf = graph.regions[0].decoder.action_policy.hidden
        original_events = leaf._pending_stability_events
        leaf.eval()
        lease = Mock()
        brain = SimpleNamespace(decoder=graph, resource_policy=SimpleNamespace(reserve_ram=Mock(return_value=lease)),
            _inherited_capability_curriculum={"curriculum": current_ground_up_curriculum_manifest(), "origins": []},
            counters={"training_steps": 0}, slow_anchors={}, slow_importance={},
            _named_slow_parameters=lambda: {},
            _train_tool_route_head=Mock(return_value={"steps": 1}),
            _train_action_argument_head=Mock(return_value={"steps": 1}))
        def mutate(**_):
            for value in graph.regions[0].decoder.action_policy.buffers():
                if value.dtype == torch.uint8 and value.numel():
                    value.flatten()[0].bitwise_xor_(1)
                    break
            brain.counters["training_steps"] += 1
            brain.counters["new_counter"] = 99
            leaf._pending_stability_events += 5
            leaf.parallel_connections.add(0, 0, sign=1, strength=3)
            return {"steps": 1, "rolledBack": False}
        brain._calibrate_starter_action_policy = mutate
        baseline = {"actionLoss": 1.0, "routeLoss": 1.0, "argumentLoss": 1.0,
                    "correctActions": 2, "correctRoutes": 2, "minimumExpectedProbability": 0.1}
        for failure in (True, False):
            with self.subTest(exception=failure), patch("omni_core.capability_rehearsal.eligible_ground_up_rehearsal", return_value=True), patch(
                    "omni_core.connected_capabilities._samples", return_value=()), patch(
                    "omni_core.connected_capabilities._measure", side_effect=[baseline, {**baseline, "routeLoss": 9.0}]), patch(
                    "omni_core.connected_capabilities._retention_rows", return_value={
                        "actionCorrect": [], "routeCorrect": [], "actionTargetProbabilities": [],
                        "routeTargetProbabilities": [], "argumentLosses": []}):
                brain._train_tool_route_head.side_effect = RuntimeError("cancel after native head mutation") if failure else None
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "cancel after"):
                        rehearse_inherited_capabilities(brain, phase="final", committed_global_waves=1,
                            policy=CapabilityRehearsalPolicy())
                else:
                    result = rehearse_inherited_capabilities(brain, phase="final", committed_global_waves=1,
                        policy=CapabilityRehearsalPolicy())
                    self.assertFalse(result["candidateAccepted"])
                    self.assertEqual(result["parameterSteps"], 0)
                    self.assertNotEqual(result["ownerChecksumBefore"], result["ownerChecksumCandidate"])
                    self.assertEqual(result["ownerChecksumAfter"], result["ownerChecksumBefore"])
                    self.assertFalse(result["readinessProbeExecuted"])
                self.assertEqual(checksum(), initial)
                self.assertEqual(brain.counters, {"training_steps": 0})
                self.assertEqual(leaf._pending_stability_events, original_events)
                self.assertFalse(leaf.training)
        self.assertEqual(lease.release.call_count, brain.resource_policy.reserve_ram.call_count)

    def test_improved_mean_cannot_hide_an_individual_lost_route(self):
        before = {"actionCorrect": [True, False], "routeCorrect": [True],
                  "actionTargetProbabilities": [0.6, 0.1], "routeTargetProbabilities": [0.9],
                  "argumentLosses": [1.0, 8.0]}
        after = {**before, "actionCorrect": [False, True], "actionTargetProbabilities": [0.3, 0.9]}
        self.assertFalse(_rows_retained(before, after, 0.02))
        self.assertFalse(_rows_retained(before, {**before, "argumentLosses": [1.2, 1.0]}, 0.02))
        self.assertTrue(_rows_retained(before, {**before, "argumentLosses": [0.9, 7.0]}, 0.02))

    def test_connected_replay_uses_authorized_storage_and_exactly_rolls_back_a_failed_candidate(self):
        from omni_core.optimizers import PackedMutationSnapshot

        graph = connected()
        graph.regions[0].decoder.action_policy.hidden.eval()
        modes = [module.training for module in graph.modules()]
        initial = tensor_checksum(graph.state_dict().values())
        lease, disk_calls = Mock(), []
        refused = False

        def reserve(_size, label):
            nonlocal refused
            if label.endswith('packed rollback') and not refused:
                refused = True
                raise RuntimeError('simulated whole-head RAM pressure')
            return lease

        with tempfile.TemporaryDirectory(prefix='omni-connected-spill-fixture-') as temporary:
            brain = SimpleNamespace(decoder=graph,
                config=SimpleNamespace(disk_state_offload=True), _live_paging_cache_directory=Path(temporary),
                resource_policy=SimpleNamespace(reserve_ram=reserve,
                    require_disk=lambda size, label: disk_calls.append((size, label))),
                _inherited_capability_curriculum={'curriculum': current_ground_up_curriculum_manifest(), 'origins': []},
                counters={'training_steps': 4}, slow_anchors={}, slow_importance={}, _named_slow_parameters=lambda: {},
                _train_tool_route_head=Mock(side_effect=RuntimeError('interrupted connected native candidate')),
                _train_action_argument_head=Mock())

            def mutate(**_):
                leaf = graph.regions[0].decoder.action_policy.hidden
                leaf.fill_ternary_(0)
                leaf.parallel_connections.add(0, 0, sign=1, strength=3)
                brain.counters.update(training_steps=5, candidate_only=1)
                return {'steps': 1, 'rolledBack': False}

            brain._calibrate_starter_action_policy = mutate
            capture, storage_modes = PackedMutationSnapshot.capture, []

            def observe(*args, **kwargs):
                snapshot = capture(*args, **kwargs)
                storage_modes.append(snapshot.storage_mode)
                return snapshot

            readings = {'actionLoss': 1., 'routeLoss': 1., 'argumentLoss': 1., 'correctActions': 0, 'correctRoutes': 0}
            rows = {'actionCorrect': [], 'routeCorrect': [], 'actionTargetProbabilities': [],
                    'routeTargetProbabilities': [], 'argumentLosses': []}
            with patch.object(PackedMutationSnapshot, 'capture', side_effect=observe), \
                    patch('omni_core.capability_rehearsal.eligible_ground_up_rehearsal', return_value=True), \
                    patch('omni_core.connected_capabilities._samples', return_value=()), \
                    patch('omni_core.connected_capabilities._measure', return_value=readings), \
                    patch('omni_core.connected_capabilities._retention_rows', return_value=rows), \
                    self.assertRaisesRegex(RuntimeError, 'interrupted connected native candidate'):
                rehearse_inherited_capabilities(brain, phase='middle', committed_global_waves=1,
                    policy=CapabilityRehearsalPolicy())
            self.assertEqual(storage_modes, ['private-disk-packed-snapshot'])
            self.assertTrue(disk_calls)
            self.assertEqual(list((Path(temporary) / 'capability-rollback').iterdir()), [])
        self.assertEqual(tensor_checksum(graph.state_dict().values()), initial)
        self.assertEqual(brain.counters, {'training_steps': 4})
        self.assertEqual([module.training for module in graph.modules()], modes)

    def test_generic_replay_does_not_write_confirmed_experience_owners(self):
        graph = connected()
        graph.tool_route_head.experience_updates.fill_(1)
        # A retained per-route receipt protects it even if an older aggregate
        # counter is zero; no generic curriculum overwrites confirmed values.
        index = graph.action_argument_head.register_route("system.files", "read")
        graph.action_argument_head.grounded_route_updates[index].fill_(1)
        before = tensor_checksum([*graph.tool_route_head.state_dict().values(), *graph.action_argument_head.state_dict().values()])
        brain = SimpleNamespace(decoder=graph, resource_policy=SimpleNamespace(),
            _inherited_capability_curriculum={"curriculum": current_ground_up_curriculum_manifest(), "origins": []},
            counters={"training_steps": 0}, slow_anchors={}, slow_importance={}, _named_slow_parameters=lambda: {},
            _calibrate_starter_action_policy=Mock(return_value={"steps": 0, "rolledBack": False}),
            _train_tool_route_head=Mock(side_effect=AssertionError("confirmed route overwritten")),
            _train_action_argument_head=Mock(side_effect=AssertionError("confirmed arguments overwritten")))
        rows = {"actionCorrect": [], "routeCorrect": [], "actionTargetProbabilities": [],
                "routeTargetProbabilities": [], "argumentLosses": []}
        readings = {"actionLoss": 1.0, "routeLoss": 1.0, "argumentLoss": 1.0, "correctActions": 0, "correctRoutes": 0}
        with patch("omni_core.capability_rehearsal.eligible_ground_up_rehearsal", return_value=True), patch(
                "omni_core.connected_capabilities._samples", return_value=()), patch(
                "omni_core.connected_capabilities._measure", return_value=readings), patch(
                "omni_core.connected_capabilities._retention_rows", return_value=rows):
            result = rehearse_inherited_capabilities(brain, phase="final", committed_global_waves=1,
                policy=CapabilityRehearsalPolicy())
        self.assertEqual(result["protectedOwners"], ["tool_route_head", "action_argument_head"])
        self.assertEqual(result["parameterSteps"], 0)
        self.assertEqual(tensor_checksum([*graph.tool_route_head.state_dict().values(),
            *graph.action_argument_head.state_dict().values()]), before)
        brain._train_tool_route_head.assert_not_called()
        brain._train_action_argument_head.assert_not_called()


if __name__ == "__main__":
    unittest.main()
