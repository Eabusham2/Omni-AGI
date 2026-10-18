"""New constructor-free checks for discarded/repeated native Load presentation."""
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from worker import RpcFault, Worker
from engine.tests.test_runtime_settings_shape import fixture as settings_fixture


def summary_owner():
    brain = AdaptiveBrain.__new__(AdaptiveBrain)
    brain.brain_id, brain.storage_path, brain.engine_path = "b", Path("/owned/b"), Path("/nonexistent/owned-b-engine")
    brain.config = OmniConfig.micro()
    brain.created_at = brain.updated_at = "2026-10-13T13:00:00Z"
    brain.device, brain.device_backend = torch.device("cpu"), "cpu"
    brain.counters = {"training_steps": 0}
    brain.ground_up_training_manifest = {"randomInitialization": {"exactParameterCount": 99}}
    brain.packed_ternary_manifest = None
    brain.fresh_attention_boundary = None
    brain.working_memory, brain.training_sources, brain.replay = [], [], []
    brain.installed_modality_packs, brain.modality_training = [], {}
    brain.growth_pause, brain.workspace_items = None, []
    brain.decoder = SimpleNamespace(expert_count=0)
    brain.memory = SimpleNamespace(concepts={}, ideas=[], relations={}, neurons={}, assemblies=[], synapses={}, growth_events=0, growth_pauses=0)
    brain.router = SimpleNamespace(synapses=SimpleNamespace(active_synapse_count=Mock(return_value=0), plasticity_events=torch.tensor(0)))
    brain.conversation = SimpleNamespace(summary=Mock(return_value={"messageCount": 0}))
    brain._ternary_audit = Mock(return_value={"actualCheckRetained": True})
    brain.workspace_snapshot = Mock(return_value={})
    brain._state_offload_status = Mock(return_value={"actualResourceStatusRetained": True})
    brain._organic_state = Mock(return_value={})
    brain.parameter_accounting = Mock(return_value={"mutableDenseParameters": 7, "totalNeuralParameters": 11})
    brain.parameter_checksum = Mock(return_value="c" * 64)
    return brain


class LoadPresentationWorkTests(unittest.TestCase):
    def test_one_summary_measures_inventory_once_and_preserves_independent_returned_dicts(self):
        brain = summary_owner()
        result = brain.summary()
        brain.parameter_checksum.assert_called_once()
        brain.parameter_accounting.assert_called_once()
        brain._ternary_audit.assert_called_once()
        brain._state_offload_status.assert_called_once()
        self.assertEqual(result["runtimeCard"]["randomInitialization"]["exactParameterCount"], 99)
        self.assertEqual(result["metrics"]["parameterAccounting"], result["runtimeCard"]["parameterAccounting"])
        self.assertIsNot(result["metrics"]["parameterAccounting"], result["runtimeCard"]["parameterAccounting"])
        result["metrics"]["parameterAccounting"]["totalNeuralParameters"] = 999
        self.assertEqual(result["runtimeCard"]["parameterAccounting"]["totalNeuralParameters"], 11)
        self.assertEqual(brain.parameter_accounting.return_value["totalNeuralParameters"], 11)
        # This is operation-local, not a cached inventory on the brain.
        brain.parameter_accounting.return_value = {"mutableDenseParameters": 7, "totalNeuralParameters": 12}
        self.assertEqual(brain.summary()["runtimeCard"]["parameterAccounting"]["totalNeuralParameters"], 12)

    def test_standalone_runtime_card_has_no_eager_unused_recount(self):
        brain = summary_owner()
        self.assertEqual(brain.runtime_card()["parameterAccounting"]["totalNeuralParameters"], 11)
        brain.parameter_accounting.assert_called_once()

    def test_internal_no_summary_config_keeps_validation_save_and_event_audit(self):
        brain, _, _ = settings_fixture()
        brain.summary = Mock(side_effect=AssertionError("discarded Load presentation"))
        brain.runtime_card = Mock(return_value={"actualAudit": True})
        result = brain.update_config({"name": "Actual new name"}, persist_if_unchanged=False, include_summary=False)
        self.assertEqual(result, {"changed": ["name"], "ignoredLearnedShapeFields": []})
        self.assertEqual(brain.config.name, "Actual new name")
        brain.save.assert_called_once()
        brain.events.append.assert_called_once_with("config-updated", {"changed": ["name"], "runtimeCard": {"actualAudit": True}})
        brain.runtime_card.assert_called_once()
        brain.summary.assert_not_called()
        brain.save.reset_mock()
        with self.assertRaises(ValueError):
            brain.update_config({"contextWindowTokens": 1}, persist_if_unchanged=False, include_summary=False)
        brain.save.assert_not_called()

    def test_unchanged_internal_config_skips_only_discarded_report_not_control_validation(self):
        brain, _, _ = settings_fixture()
        brain.summary = Mock(side_effect=AssertionError("discarded Load summary"))
        brain.runtime_card = Mock(side_effect=AssertionError("unchanged configuration audit"))
        result = brain.update_config({}, persist_if_unchanged=False, include_summary=False)
        self.assertEqual(result["changed"], [])
        brain.population_controls_from_config.assert_called_once()
        brain.save.assert_not_called()
        brain.summary.assert_not_called()

    def test_worker_load_returns_one_final_summary_and_retains_warm_cancel_check(self):
        store = SimpleNamespace(current=Mock(return_value=None), mark_loaded=Mock())
        clock = SimpleNamespace(resume=Mock())
        brain = SimpleNamespace(update_config=Mock(), summary=Mock(return_value={"actualFinalSummary": True}), _owned_activity_clock=lambda: clock)
        session = SimpleNamespace(_get=Mock(return_value=brain), _execution_store=Mock(return_value=store),
                                  reviewed_execution_runtime_config=Mock(return_value={}), _cooperative_cancel=threading.Event())
        with patch("omni_core.connected_execution.connected_execution_status", return_value={"rootMutationHeld": False}):
            result = Worker.load(session, {"brainId": "b", "config": {}}, "load")
            self.assertEqual(result, {"actualFinalSummary": True})
            brain.update_config.assert_called_once_with({}, persist_if_unchanged=False, include_summary=False)
            brain.summary.assert_called_once()
            store.mark_loaded.assert_called_once()
            clock.resume.assert_called_once()
            brain.summary.reset_mock()
            session._cooperative_cancel.set()
            with self.assertRaisesRegex(RpcFault, "safe boundary"):
                Worker.load(session, {"brainId": "b"}, "cancelled-load")
            brain.summary.assert_not_called()

    def test_pack_comparison_reuses_only_the_unpublished_phase_checksum(self):
        source = (Path(__file__).parents[1] / "omni_core" / "brain.py").read_text()
        start = source.index("current_loaded_parameter_checksum = brain.parameter_checksum()")
        stop = source.index('stored_cpu_rng = plastic.get("state.rng_cpu")', start)
        phase = source[start:stop]
        self.assertIn("packed_parameter_checksum != current_loaded_parameter_checksum", phase)
        self.assertEqual(phase.count("brain.parameter_checksum()"), 1)
        self.assertNotIn("brain.save()", phase)
        self.assertNotIn("brain.export_packed_ternary()", phase)
        self.assertIn("verify_ternary_shards(packed_path, retain_names=())", phase)
        self.assertIn("validate_checkpoint_binding_v3(", phase)


if __name__ == "__main__":
    unittest.main()
