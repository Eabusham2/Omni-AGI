"""Production method wiring fixtures; no torch/model/brain construction."""

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional


SOURCE = Path(__file__).resolve().parents[1] / "omni_core" / "brain.py"


def method_fixture(names, **extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
    methods = [copy.deepcopy(node) for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {method.name for method in methods} != set(names):
        raise AssertionError("production method disappeared")
    namespace = dict(copy=copy, Any=Any, Dict=Dict, Optional=Optional, Callable=Callable, Mapping=Mapping, **extra)
    module = ast.Module(body=[ast.ClassDef(name="MethodFixture", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["MethodFixture"]()


class FakeActivityPager:
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.resident_budget_bytes = kwargs["resident_budget_bytes"]
        self.scratch_budget_bytes = kwargs["scratch_budget_bytes"]
        self.device_tile_budget_bytes = kwargs["device_tile_budget_bytes"]
        self.closed = False

    def status(self):
        return {"residentBytes": 0, "spillBytes": 0, "contextPagedToStorage": False}

    def close(self):
        self.closed = True


class BrainPagingContractTests(unittest.TestCase):
    def test_failed_precommit_restores_only_staged_authority_pointers(self):
        fixture = method_fixture({"save"})
        fixture.memory = SimpleNamespace(persistence_manifest={"generation": "old"})
        fixture.mutable_state_manifest = {"generation": "old-core"}
        fixture.pending_learning = ["valid unsaved mutation remains pending"]

        def failed(**kwargs):
            fixture.memory.persistence_manifest = {"generation": "ahead"}
            fixture.mutable_state_manifest = {"generation": "ahead-core"}
            raise OSError("checkpoint refused")

        fixture._save_checkpoint_impl = failed
        with self.assertRaisesRegex(OSError, "checkpoint refused"):
            fixture.save()
        self.assertEqual(fixture.memory.persistence_manifest, {"generation": "old"})
        self.assertEqual(fixture.mutable_state_manifest, {"generation": "old-core"})
        self.assertEqual(len(fixture.pending_learning), 1)

    def test_postcommit_failure_never_rolls_back_durable_authority(self):
        fixture = method_fixture({"save"})
        fixture.memory = SimpleNamespace(persistence_manifest={"generation": "old"})
        fixture.mutable_state_manifest = {"generation": "old-core"}

        def committed_then_failed(**kwargs):
            fixture.memory.persistence_manifest = {"generation": "new"}
            fixture.mutable_state_manifest = {"generation": "new-core"}
            kwargs["commit_callback"]()
            raise OSError("optional maintenance failed")

        fixture._save_checkpoint_impl = committed_then_failed
        with self.assertRaises(OSError):
            fixture.save()
        self.assertEqual(fixture.memory.persistence_manifest, {"generation": "new"})
        self.assertEqual(fixture.mutable_state_manifest, {"generation": "new-core"})

    def test_working_activity_uses_one_shared_envelope_without_fixed_context_ceiling(self):
        gib = 1024 ** 3
        readings = {
            "systemRamBudgetBytes": 12 * gib, "processMemoryBytes": gib,
            "availableSafeRamBytes": 8 * gib, "diskFreeBytes": 60 * gib,
            "diskReserveBytes": 20 * gib,
        }
        fixture = method_fixture({"_configure_working_attention_resources"}, WorkingAttentionPager=FakeActivityPager)
        fixture.resource_policy = SimpleNamespace(status=lambda: dict(readings))
        fixture.core_pager = SimpleNamespace(status=lambda: {"cpuHeapBytes": 0, "cpuMappedLogicalBytes": 0})
        fixture.config = SimpleNamespace(training_accelerator_budget_bytes=0, training_scratch_budget_bytes=0,
            disk_state_offload=True, storage_pool_bytes=0, working_attention_scratch_budget_bytes=None)
        fixture.device = "cpu"
        attached = []
        fixture.decoder = SimpleNamespace(configure_working_attention=attached.append)
        with tempfile.TemporaryDirectory(prefix="omni-paging-method-") as directory:
            fixture._live_paging_cache_directory = Path(directory)
            fixture._configure_working_attention_resources()
            self.assertEqual(len(attached), 1)
            envelope = fixture._working_attention_envelope
            self.assertEqual(sum(envelope["sharedPartition"].values()), 100)
            self.assertLessEqual(envelope["activityLiveBudgetBytes"] + envelope["computeTileBudgetBytes"], envelope["residualRamBytes"])
            self.assertGreater(envelope["activityCapacityBytes"], 512 * 1024 ** 2)
            self.assertFalse(envelope["capacityShrunk"])
            fixture.config.disk_state_offload = False
            fixture._configure_working_attention_resources()
            self.assertEqual(fixture.working_attention_pager.scratch_budget_bytes, 0)

    def test_close_releases_activity_before_core_without_checkpoint_deletion(self):
        order = []
        fixture = method_fixture({"close"})
        fixture.working_attention_pager = SimpleNamespace(close=lambda: order.append("activity"))
        fixture.core_pager = SimpleNamespace(close=lambda: order.append("core") or {"checkpointTouched": False})
        fixture.conversation = SimpleNamespace(close=lambda: order.append("conversation"))
        fixture.events = SimpleNamespace(close=lambda: order.append("events"))
        fixture.close()
        self.assertEqual(order, ["activity", "core", "conversation", "events"])
        self.assertFalse(fixture._native_core_cleanup_status["checkpointTouched"])

    def test_production_commit_hook_follows_authoritative_json_and_reuse_checks_it(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint_impl")
        text = ast.unparse(method)
        self.assertLess(text.index("atomic_write_json("), text.index("commit_paged_substrate_generation("))
        self.assertIn("persisted_engine.get('substrate')", text)
        self.assertIn("commit_callback()", text)
        self.assertIn("isinstance(self.memory.assemblies, PagedAssemblyView)", text)


if __name__ == "__main__":
    unittest.main()
