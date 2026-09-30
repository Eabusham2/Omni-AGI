"""AST-extracted Worker policy methods; no Worker/brain/model construction."""
import ast
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class WorkerStdioPolicyFixtures(unittest.TestCase):
    def fixture(self):
        source = Path(__file__).parents[1] / "worker.py"
        tree = ast.parse(source.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
        names = {"_stdio_resource_policy", "reserve_stdio_memory", "stdio_memory_headroom"}
        methods = [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"os": os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), "exec"), namespace)
        created = []
        status = {"ramAdmissionVerified": True, "systemRamBudgetBytes": 1000,
            "admissionResidentMemoryBytes": 700, "sharedQuota": {"ramEscrowBytes": 100},
            "availableMemoryBytes": 500, "ramReserveBytes": 100}
        class Policy:
            def __init__(self, *args, **kwargs): created.append(kwargs)
            def status(self): return dict(status)
            def reserve_ram(self, size, operation): return (size, operation)
        module = types.ModuleType("omni_core.offload"); module.ResourcePolicy = Policy
        dummy = types.SimpleNamespace(_default_root=lambda: Path("/fixture"))
        for name in names: setattr(dummy, name, types.MethodType(namespace[name], dummy))
        return dummy, module, created, status

    def test_trusted_stdio_ceiling_is_100_percent_local_then_global_clamp_not_personal65(self):
        worker, module, created, _ = self.fixture()
        with patch.dict(sys.modules, {"omni_core.offload": module}):
            first = worker._stdio_resource_policy()
            self.assertIs(first, worker._stdio_resource_policy())
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["system_ram_share_percent"], 100)
        self.assertFalse(created[0]["include_accelerator_memory"])
        self.assertEqual(created[0]["shared_resource_owner_id"], "stdio:%d" % os.getpid())

    def test_headroom_counts_family_and_atomic_pending_and_unknown_is_not_zero_usage(self):
        worker, module, _, status = self.fixture()
        with patch.dict(sys.modules, {"omni_core.offload": module}):
            self.assertEqual(worker.stdio_memory_headroom(), 200)
            status["ramAdmissionVerified"] = False
            self.assertEqual(worker.stdio_memory_headroom(), 0)

    def test_parser_reservation_goes_through_same_policy_not_a_new_allocator(self):
        worker, module, _, _ = self.fixture()
        with patch.dict(sys.modules, {"omni_core.offload": module}):
            self.assertEqual(worker.reserve_stdio_memory(123, "JSON decode"), (123, "JSON decode"))


if __name__ == "__main__": unittest.main()
