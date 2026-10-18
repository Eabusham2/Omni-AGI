"""New constructor-free byte-accounting cases; no Torch or hardware primitive."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("owned_projection_byte_fixture", ROOT / "omni_core/native_compute_profile.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class NativeComputeTensorByteFixtures(unittest.TestCase):
    def test_weighted_state_and_both_int64_outputs_have_declared_bytes(self):
        budget = PROFILE.native_projection_tensor_budget(8, 64, 64)
        self.assertEqual(budget["packedSignBytes"], 1024)
        self.assertEqual(budget["connectionStateBytes"], 4096)
        self.assertEqual(budget["sampleTensorBytes"], 7172)
        self.assertEqual(budget["integerRowBlockBytes"], 4096)
        self.assertEqual(budget["integerConcatenationBytes"], 4096)
        self.assertEqual(budget["forwardTensorReserveBytes"], 16940)
        self.assertEqual(PROFILE.native_projection_tensor_budget(2, 65, 3)["packedSignBytes"], 51)
        for values in ((True, 64, 64), (0, 1, 1), (1, 1, -1), (1 << 52, 64, 64)):
            with self.assertRaises(ValueError):
                PROFILE.native_projection_tensor_budget(*values)

    def test_profile_admits_new_bytes_before_injected_work_and_rejects_old_envelope(self):
        ticks = iter((0, .001, .002, .003, .004, .005, .006, .007))
        reserved, calls = [], []
        value = PROFILE.profile_native_projection_compute(primitive=lambda: calls.append(True),
            clock=lambda: next(ticks), reserve=lambda size, device: reserved.append((size, device)))
        self.assertEqual(reserved, [(1072688, "cpu")])
        self.assertEqual(len(calls), 4)
        self.assertEqual(value["formatVersion"], 2)
        self.assertEqual(value["integerResultDtype"], "int64")
        self.assertEqual(value["connectionStateBytes"], 4096)
        self.assertEqual(value["repeatedContacts"], 0)
        self.assertEqual(value["repeatedContactPages"], 0)
        self.assertFalse(value["repeatedContactTimingMeasured"])
        self.assertFalse(value["tensorReserveIsExactPeak"])
        calls.clear()

        def old_envelope(size, _device):
            if size > 1053700:
                raise MemoryError("old profile envelope is insufficient")

        with self.assertRaises(MemoryError):
            PROFILE.profile_native_projection_compute(primitive=lambda: calls.append(True), reserve=old_envelope)
        self.assertEqual(calls, [])

    def test_actual_repeat_page_storage_arithmetic_is_not_zero_repeat_profile_timing(self):
        # Execute just the existing pure inventory method on an inert owner.
        # No import/constructor/page tensor or learned connection is created.
        tree = ast.parse((ROOT / "omni_core/packed_connections.py").read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PooledParallelConnections")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "storage_status")
        namespace = {"CONTACT_PAGE_CAPACITY": 1024, "CONTACT_PAGE_METADATA_ALLOWANCE": 8192}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "owned-repeat-storage-arithmetic", "exec"), namespace)

        class InertOwner:
            pages = {"first": None, "second": None}
            def __len__(self):
                return 1  # Allocated/tombstoned slots still cost full pages.

        inventory = namespace["storage_status"](InertOwner())
        self.assertEqual(inventory["parallelConnectionCount"], 1)
        self.assertEqual(inventory["parallelConnectionPackedSignBytes"], 512)
        self.assertEqual(inventory["parallelConnectionLearnedBytes"], 2560)
        self.assertEqual(inventory["parallelConnectionStructuralBytes"], 20488)
        self.assertEqual(inventory["parallelConnectionAllocatedTensorBytes"], 23048)
        self.assertEqual(inventory["parallelConnectionIndexCacheBytes"], 4096)
        self.assertEqual(inventory["parallelConnectionRegistryAdmissionBytes"], 16384)


if __name__ == "__main__":
    unittest.main()
