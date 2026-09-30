"""Pure byte/storage/stub checks; never construct an Omni brain or decoder."""

import ast
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from omni_core.bounded_tensor_io import (
    BoundedTensorFile,
    LazyTensorMapping,
    atomic_save_tensors_bounded,
    load_module_bounded,
)
from omni_core.native_core_paging import (
    BoundedPackedRollback,
    NativeCorePager,
    NativePackedModuleMixin,
    paged_projection_call,
    shared_native_residency_budget,
)
from omni_core.optimizers import PackedMutationSnapshot
from omni_core.persistence import tensor_checksum


class StorageOwner(NativePackedModuleMixin, torch.nn.Module):
    """Tensor ownership stub, not a neural projection/model constructor."""

    def __init__(self, shape=(4, 1024)):
        super().__init__()
        self._init_native_paging()
        self.register_buffer("_packed_forward_weight", self._native_buffer(
            "_packed_forward_weight", shape,
            fill=None if self._native_loading_checkpoint else 0x55,
        ))
        self.register_buffer("_row_stability", self._native_buffer(
            "_row_stability", (shape[0],), fill=0,
        ))
        self.register_buffer("_packed_forward_scale", torch.tensor(0.25))
        self.register_buffer("_autograd_trigger", torch.zeros(()), persistent=False)
        self._pending_stability_events = 0
        self._finish_native_paging()

    def authoritative_packed_tensors(self):
        return (self._packed_forward_weight,)

    def _validate_packed(self):
        if self._packed_forward_weight.dtype != torch.uint8:
            raise ValueError("invalid stub byte dtype")

    @paged_projection_call
    def consume(self, activity):
        # Exercise scope ownership only, with no synapse training/decoder.
        return activity + self._packed_forward_scale


class NativeCorePagingStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-core-storage-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _owner(self, pager, *, shape=(4, 1024), loading=False):
        with pager.construction(from_checkpoint=loading):
            return StorageOwner(shape)

    def test_ram_first_then_exact_cold_spill_and_pressure_reduction(self):
        reserves = []
        pager = NativeCorePager(self.root / "cold", cpu_hot_bytes=5000,
                               reserve_disk=lambda size, label: reserves.append((size, label)),
                               chunk_bytes=127)
        first = self._owner(pager)
        self.assertEqual(pager.status()["cpuMappedLogicalBytes"], 0)
        second = self._owner(pager)
        self.assertGreaterEqual(pager.status()["cpuMappedLogicalBytes"], 4096)
        before = first._packed_forward_weight.clone()
        old_target = first._packed_forward_weight
        pager.cool_to_budget(16)
        self.assertIsNot(first._packed_forward_weight, old_target)
        self.assertTrue(torch.equal(first._packed_forward_weight, before))
        self.assertTrue(torch.equal(second._packed_forward_weight, before))
        self.assertLessEqual(pager.status()["peakTransferScratchBytes"], 127)
        self.assertTrue(reserves)
        self.assertFalse(pager.status()["hardCpuRssBound"])
        self.assertFalse(pager.status()["attentionKvIncluded"])
        expected_allocated = sum(getattr(path.stat(), "st_blocks", 0) * 512
                                 for path in (self.root / "cold").iterdir())
        if expected_allocated:
            self.assertEqual(pager.status()["cpuMappedAllocatedBytes"], expected_allocated)
            self.assertEqual(pager.status()["cpuMappedAllocationBasis"], "posix-st_blocks-times-512")

    def test_shared_residual_budget_no_fixed_ceiling_and_negative_headroom(self):
        gib = 1024 ** 3
        readings = {"systemRamBudgetBytes": 16 * gib, "availableSafeRamBytes": 12 * gib}
        budget = shared_native_residency_budget(readings, baseline_bytes=2 * gib)
        self.assertGreater(budget["corePartitionBytes"], 512 * 1024 * 1024)
        self.assertEqual(budget["corePartitionBytes"] + budget["workingActivityPartitionBytes"]
                         + budget["trainingTransferPartitionBytes"] + budget["measuredBaselineBytes"], 16 * gib)
        pressure = shared_native_residency_budget({
            **readings, "availableMemoryBytes": gib // 2, "ramReserveBytes": gib,
            "processMemoryBytes": 15 * gib,
        }, baseline_bytes=2 * gib, current_core_heap_bytes=4 * gib)
        self.assertEqual(pressure["liveCoreHotBytes"], int(3.5 * gib))

    def test_live_budget_promotes_fitting_core_and_preserves_exported_view(self):
        target = [16]
        pager = NativeCorePager(self.root / "pages", budget_provider=lambda _status: {
            "liveCoreHotBytes": target[0], "acceleratorCoreHotBytes": 0,
        }, chunk_bytes=128)
        owner = self._owner(pager)
        exported = owner._packed_forward_weight.detach()
        self.assertGreater(pager.status()["cpuMappedLogicalBytes"], 0)
        target[0] = 8192
        pager.refresh_budget(force=True)
        self.assertEqual(pager.status()["cpuMappedLogicalBytes"], 0)
        self.assertTrue(bool(exported.eq(0x55).all()))
        self.assertTrue(torch.equal(owner._packed_forward_weight, exported))
        self.assertTrue(any(item["event"] == "cpu-hot-promote" for item in pager.status()["trace"]))

    def test_close_retains_exported_mapping_and_rejects_new_jobs(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        owner = self._owner(pager)
        exported = owner._packed_forward_weight.detach()
        checkpoint = self.root / "unchanged.safetensors"
        atomic_save_tensors_bounded(checkpoint, {"weight": exported})
        before = checkpoint.read_bytes()
        report = pager.close()
        self.assertFalse(report["mappingForceClosed"])
        self.assertFalse(report["checkpointDeleted"])
        self.assertGreaterEqual(report["mappedBytesRetainedByTensorViews"], exported.numel())
        self.assertTrue(bool(exported.eq(0x55).all()))
        self.assertEqual(checkpoint.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            owner.consume(torch.ones(1))

    def test_failed_construction_cleans_only_new_owners(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        stable = self._owner(pager)
        stable_value = stable._packed_forward_weight.clone()
        before_files = {path.name for path in (self.root / "pages").iterdir()}
        with self.assertRaisesRegex(RuntimeError, "construction abort"):
            with pager.construction():
                created = StorageOwner()
                raise RuntimeError("construction abort")
        self.assertFalse(pager.status()["closed"])
        self.assertTrue(torch.equal(stable._packed_forward_weight, stable_value))
        stable.consume(torch.ones(1))
        self.assertEqual(pager.status()["registeredPackedOwners"], 1)
        remaining = {path.name for path in (self.root / "pages").iterdir()}
        self.assertTrue(before_files.issubset(remaining))
        self.assertIsNone(created._packed_forward_weight)

    def test_deferred_copy_is_chunk_bounded_and_checkpoint_is_immutable(self):
        values = {
            "core._packed_forward_weight": torch.full((3, 4097), 0x55, dtype=torch.uint8),
            "core._row_stability": torch.zeros(3, dtype=torch.uint8),
            "core._packed_forward_scale": torch.tensor(0.75),
        }
        path = self.root / "checkpoint.safetensors"
        atomic_save_tensors_bounded(path, values, chunk_bytes=128)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=8, chunk_bytes=128)
        owner = self._owner(pager, shape=(3, 4097), loading=True)
        with self.assertRaisesRegex(RuntimeError, "before.*completed"):
            owner.consume(torch.ones(1))
        reader = BoundedTensorFile(path, chunk_bytes=128)
        transfers = []
        load_module_bounded(owner, reader, "core.", on_chunk=lambda *entry: transfers.append(entry))
        pager.finish_load()
        self.assertLessEqual(reader.peak_transfer_bytes, 128)
        self.assertGreater(len(transfers), 90)
        self.assertTrue(torch.equal(owner._packed_forward_weight, values["core._packed_forward_weight"]))
        owner._packed_forward_weight.fill_(0xAA)
        pager.flush()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
        self.assertFalse(pager.status()["checkpointWritableMapped"])

    def test_to_device_does_not_eager_move_cold_packed_state(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        owner = self._owner(pager)
        original = owner._packed_forward_weight
        owner.to(device="meta")
        self.assertIs(owner._packed_forward_weight, original)
        self.assertEqual(owner._packed_forward_weight.device.type, "cpu")
        self.assertEqual(owner._native_compute_device.type, "meta")
        self.assertEqual(pager.status()["acceleratorAdmittedBytes"], 0)

    def test_isolated_region_copy_preserves_packed_bytes_without_copying_live_pager(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0, chunk_bytes=128)
        owner = self._owner(pager)
        cloned = copy.deepcopy(owner)
        self.assertIsNone(cloned._native_core_pager)
        self.assertEqual(cloned._packed_forward_weight.device.type, "cpu")
        self.assertTrue(torch.equal(cloned._packed_forward_weight, owner._packed_forward_weight))
        cloned._packed_forward_weight.fill_(0)
        self.assertTrue(bool(owner._packed_forward_weight.eq(0x55).all()))

    def test_simulated_device_lru_writes_back_mutations_only(self):
        # Simulate device copies with CPU tensors; no actual model or device
        # kernel is run. This validates ownership/eviction, not backend support.
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0,
                               accelerator_hot_bytes=4200, chunk_bytes=128)
        first = self._owner(pager, shape=(4, 1024))
        second = self._owner(pager, shape=(4, 1024))
        original_empty_like = torch.empty_like
        def fake_device_copy(value, **_kwargs):
            return original_empty_like(value, device="cpu")
        with patch("omni_core.native_core_paging.torch.empty_like", side_effect=fake_device_copy):
            with pager.use(first, torch.device("meta")):
                first._packed_forward_weight.fill_(0xAA)
                first._row_stability.fill_(3)
                with self.assertRaisesRegex(RuntimeError, "pinned"):
                    pager.evict(first)
            # Cached working owner remains hot; another owner forces LRU
            # writeback into the first owner's session-backed packed bytes.
            with pager.use(second, torch.device("meta")):
                pass
        self.assertTrue(bool(first._packed_forward_weight.eq(0xAA).all()))
        self.assertTrue(bool(first._row_stability.eq(3).all()))
        self.assertEqual(pager.status()["acceleratorPageOuts"], 1)
        written = pager.status()["acceleratorWritebackBytes"]
        self.assertEqual(written, 4100)
        pager.flush()
        self.assertEqual(pager.status()["acceleratorWritebackBytes"], written)
        self.assertLessEqual(pager.status()["peakTransferScratchBytes"], 128)

    def test_failed_admission_keeps_exact_cold_owner(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0,
                               accelerator_hot_bytes=8)
        owner = self._owner(pager)
        before = owner._packed_forward_weight.clone()
        with self.assertRaisesRegex(RuntimeError, "exceeds.*budget"):
            with pager.use(owner, torch.device("meta")):
                self.fail("oversized owner admitted")
        self.assertTrue(torch.equal(owner._packed_forward_weight, before))
        self.assertEqual(pager.status()["pinnedOwners"], 0)
        self.assertEqual(pager.status()["acceleratorAdmittedBytes"], 0)

    def test_disk_reserve_and_staging_failure_preserve_owner_bytes(self):
        def reject_disk(_size, _label):
            raise RuntimeError("storage watermark")
        declined = NativeCorePager(self.root / "declined", cpu_hot_bytes=0,
                                   reserve_disk=reject_disk)
        with self.assertRaisesRegex(RuntimeError, "watermark"):
            self._owner(declined)
        self.assertFalse((self.root / "declined").exists())
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0,
                               accelerator_hot_bytes=8192)
        owner = self._owner(pager)
        before = owner._packed_forward_weight.clone()
        original_empty_like = torch.empty_like
        calls = []
        def fail_second_allocation(value, **_kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("simulated allocator pressure")
            return original_empty_like(value, device="cpu")
        with patch("omni_core.native_core_paging.torch.empty_like", side_effect=fail_second_allocation):
            with self.assertRaisesRegex(RuntimeError, "allocator pressure"):
                with pager.use(owner, torch.device("meta")):
                    self.fail("partial admission published")
        self.assertTrue(torch.equal(owner._packed_forward_weight, before))
        self.assertEqual(pager.status()["pinnedOwners"], 0)
        self.assertEqual(pager.status()["acceleratorAdmittedBytes"], 0)

    def test_cancelled_partial_load_is_not_executable_or_checkpoint_mutating(self):
        path = self.root / "checkpoint.safetensors"
        tensor = torch.full((4, 1024), 0xAA, dtype=torch.uint8)
        atomic_save_tensors_bounded(path, {"weight": tensor})
        before = path.read_bytes()
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        owner = self._owner(pager, loading=True)
        calls = []
        def cancelled():
            calls.append(True)
            return len(calls) > 1
        with self.assertRaises(InterruptedError):
            BoundedTensorFile(path, chunk_bytes=128).copy_into(
                "weight", owner._packed_forward_weight, cancelled=cancelled,
            )
        with self.assertRaisesRegex(RuntimeError, "before.*completed"):
            owner.consume(torch.ones(1))
        self.assertEqual(path.read_bytes(), before)

    def test_disk_rollback_resolves_replaced_targets_and_repeats_exactly(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=8192, chunk_bytes=128)
        owner = self._owner(pager)
        expected = owner._packed_forward_weight.clone()
        snapshot = BoundedPackedRollback.capture((owner,), directory=self.root / "rollback")
        pager.cool_to_budget(0)
        owner._packed_forward_weight.fill_(0xAA)
        owner._row_stability.fill_(8)
        owner._pending_stability_events = 9
        snapshot.restore()
        self.assertTrue(torch.equal(owner._packed_forward_weight, expected))
        self.assertTrue(bool(owner._row_stability.eq(0).all()))
        self.assertEqual(owner._pending_stability_events, 0)
        owner._packed_forward_weight.fill_(0)
        snapshot.restore()
        self.assertTrue(torch.equal(owner._packed_forward_weight, expected))
        snapshot.close()
        self.assertFalse(snapshot.path.exists())

    def test_optimizer_snapshot_ram_names_and_disk_fallback_after_spill(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=8192)
        owner = self._owner(pager)
        expected = owner._packed_forward_weight.clone()
        ram = PackedMutationSnapshot.capture((owner,), reserve=lambda _size: None)
        pager.cool_to_budget(0)
        owner._packed_forward_weight.fill_(0xAA)
        ram.restore()
        self.assertTrue(torch.equal(owner._packed_forward_weight, expected))
        ram.close()
        def reject_ram(_size):
            raise RuntimeError("RAM reserve")
        disk = PackedMutationSnapshot.capture((owner,), reserve=reject_ram,
                                               disk_directory=self.root / "rollback")
        self.assertEqual(disk.storage_mode, "private-disk-packed-snapshot")
        pager.promote_hot_to_budget()
        owner._packed_forward_weight.fill_(0)
        disk.restore()
        self.assertTrue(torch.equal(owner._packed_forward_weight, expected))
        disk.close()

    def test_mutation_revision_ignores_unchanged_reads_and_to(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        owner = self._owner(pager)
        revision = pager.mutation_revision
        owner.consume(torch.ones(1))
        owner.to("cpu")
        pager.flush()
        self.assertEqual(pager.mutation_revision, revision)
        owner._packed_forward_weight[0, 0] = 0xAA
        self.assertGreater(pager.mutation_revision, revision)

    def test_activity_is_viewport_observation_and_retargets_without_faking_unseen_rows(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0, chunk_bytes=128)
        owner = self._owner(pager)
        pager.bind_names((("cortex.", owner),))
        first = pager.observation("cortex.", enabled=True, start=1, count=2)
        self.assertFalse(first["observed"])
        pager.observe_output(owner, torch.zeros((1, 4)), torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
        result = pager.observation("cortex.", enabled=True, start=1, count=2)
        self.assertTrue(result["observed"])
        self.assertEqual(result["observation"]["start"], 1)
        self.assertFalse(result["observation"]["fullPopulationObserved"])
        self.assertIsNone(result["observation"]["firingClassification"])
        other = pager.observation("cortex.", enabled=True, start=3, count=1)
        self.assertFalse(other["observed"])
        self.assertIsNone(other["observation"])

    def test_standard_reader_accepts_bounded_output_and_cancel_keeps_old_file(self):
        path = self.root / "tensors.safetensors"
        values = {
            "empty": torch.empty((0, 3), dtype=torch.int16),
            "bytes": torch.arange(255, dtype=torch.uint8),
            "half": torch.tensor([0.25, 1.5], dtype=torch.bfloat16),
            "scalar": torch.tensor(71, dtype=torch.long),
            "bool": torch.tensor([True, False]),
        }
        atomic_save_tensors_bounded(path, values, metadata={"mode": "test"}, chunk_bytes=9)
        actual = load_file(str(path))
        for name, value in values.items():
            self.assertTrue(torch.equal(actual[name], value))
        before = path.read_bytes()
        with self.assertRaises(InterruptedError):
            atomic_save_tensors_bounded(path, values, cancelled=lambda: True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        lazy = LazyTensorMapping(path)
        self.assertEqual(set(lazy), set(values))
        self.assertTrue(torch.equal(lazy["bytes"], values["bytes"]))

    def test_strict_bounded_loader_rejects_dtype_and_inventory_mismatch(self):
        pager = NativeCorePager(self.root / "pages", cpu_hot_bytes=0)
        owner = self._owner(pager, loading=True)
        path = self.root / "bad.safetensors"
        atomic_save_tensors_bounded(path, {"core.unexpected": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            load_module_bounded(owner, BoundedTensorFile(path), "core.")
        values = {"core." + name: value for name, value in owner.state_dict().items()}
        values["core._packed_forward_weight"] = torch.zeros((4, 1024), dtype=torch.float32)
        atomic_save_tensors_bounded(path, values)
        with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
            load_module_bounded(owner, BoundedTensorFile(path), "core.")

    def test_checksum_wire_bytes_are_unchanged_and_production_autograd_has_no_weight_refs(self):
        tensors = [torch.arange(7, dtype=torch.float32), torch.tensor([-1, 0, 1], dtype=torch.int8)]
        digest = hashlib.sha256()
        for value in tensors:
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
        self.assertEqual(tensor_checksum(tensors), digest.hexdigest())
        model_path = Path(__file__).parents[1] / "omni_core" / "model.py"
        tree = ast.parse(model_path.read_text())
        for child in tree.body:
            if isinstance(child, ast.ClassDef) and child.name in {
                "_PackedOnlyTernaryLinear", "_PackedOnlyTernaryConvolution",
            }:
                refs = [node.attr for node in ast.walk(child)
                        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "ctx"]
                self.assertNotIn("packed", refs)
                self.assertNotIn("scale", refs)


if __name__ == "__main__":
    unittest.main()
