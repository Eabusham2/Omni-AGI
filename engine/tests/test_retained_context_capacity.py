"""New stdlib capacity + inert device custody contracts; no neural imports.

The page owner/sequence and paged codec branch are AST-loaded unchanged. A
byte-only tensor adapter simulates device names, never GPU arithmetic/hardware,
model construction, Load or training. Fixture arithmetic is shared with TS.
"""
from __future__ import annotations

import ast
import bisect
import contextvars
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
import unittest
import uuid
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("owned_retained_capacity", ROOT / "engine/omni_core/retained_context_capacity.py")
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)
FIXTURE = json.loads((ROOT / "tests/fixtures/retained-context-capacity.json").read_text())


class Device:
    def __init__(self, value="cpu"):
        self.name = str(value); self.type = self.name.split(":")[0]
    def __str__(self): return self.name
    def __eq__(self, other): return self.name == str(other)


class FakeOOM(RuntimeError): pass


class BytesTensor:
    fail_target = None
    def __init__(self, values, device="cpu"):
        self.values, self.device, self.dtype = bytes(values), Device(device), "uint8"
        self.shape, self.ndim = (len(self.values),), 1
    def numel(self): return len(self.values)
    def element_size(self): return 1
    def detach(self): return self
    def contiguous(self): return self
    def stride(self, _axis): return 1
    def movedim(self, first, second):
        assert first == second == 0
        return self
    def narrow(self, axis, start, count):
        assert axis == 0
        return self[start:start + count]
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        assert shape == (-1,) or tuple(shape) == self.shape
        return self
    def view(self, _dtype): return self
    def numpy(self): return types.SimpleNamespace(tobytes=lambda: self.values)
    def to(self, *, device, copy=False):
        if str(device) == self.fail_target: raise FakeOOM("inert allocation race")
        return BytesTensor(self.values, device) if copy or self.device != device else self
    def __getitem__(self, index): return BytesTensor(self.values[index], self.device)


def cat(values, *, dim):
    assert dim == 0
    if any(item.device != values[0].device for item in values):
        raise AssertionError("mixed-device fragments reached cat")
    return BytesTensor(b"".join(item.values for item in values), values[0].device)


class Pause(RuntimeError):
    def __init__(self, message, status): super().__init__(message); self.status = status


class Lease:
    def __init__(self): self.released = False
    def __enter__(self): return self
    def __exit__(self, *_): self.release()
    def mark_allocated(self, _size): pass
    def release(self): self.released = True


class Policy:
    def __init__(self): self.calls, self.blocked, self.block_ram = [], set(), False
    def reserve_ram(self, size, operation):
        self.calls.append(("ram", size, operation))
        if self.block_ram: raise Pause("inert live RAM pressure", {})
        return Lease()
    def require_device(self, device, size, operation):
        self.calls.append((str(device), size, operation))
        if str(device) in self.blocked: raise Pause("inert current device pressure", {})
        return {"enabled": True, "capacityMeasured": True, "remainingBytes": 4096}
    def device_budget(self, device, *, fresh):
        assert fresh is True
        return {"enabled": True, "capacityMeasured": True, "remainingBytes": 0 if str(device) in self.blocked else 4096}
    def status(self, **_): return {}


def definitions(path, names, namespace, module_name):
    module = types.ModuleType(module_name); module.__dict__.update(namespace); sys.modules[module_name] = module
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(path), "exec"), module.__dict__)
    return module


TORCH = types.SimpleNamespace(device=Device, Tensor=BytesTensor, uint8="uint8", OutOfMemoryError=FakeOOM,
    cat=cat, frombuffer=lambda values, dtype: BytesTensor(values))
P = definitions(ROOT / "engine/omni_core/working_attention_paging.py",
    {"WorkingAttentionCancelled", "_ActivityPage", "_OwnedSpillLease", "WorkingAttentionPager", "PagedTensorSequence", "_saved_activity_scratch"},
    dict(torch=TORCH, bisect=bisect, contextvars=contextvars, hashlib=hashlib, math=math, os=os, time=time, uuid=uuid,
        OrderedDict=OrderedDict, contextmanager=contextmanager, nullcontext=nullcontext, dataclass=dataclass, Path=Path,
        NeuralStateResourcePause=Pause, canonical_device=str, current_regional_target=lambda _: None,
        _ACTIVE_PAGER=contextvars.ContextVar("owned_retained_pager", default=None)), "owned_retained_pager")


def pool(device, size=8):
    return {"deviceId": "physical:" + device, "executionDevice": device, "memoryGroupId": "physical:" + device,
            "budgetBytes": size, "selectedCeilingBytes": 32, "effectiveCopyBytesPerSecond": None, "copyTimingMeasured": False}


def source_method(file, owner, name):
    tree = ast.parse((ROOT / file).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    namespace = dict(NeuralStateResourcePause=Pause, canonical_device=str, time=time)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), "owned-context-source-method", "exec"), namespace)
    return namespace[name]


class RetainedContextCapacityCases(unittest.TestCase):
    def pager(self, *, ram=4, scratch=0, pools=None):
        temporary = TemporaryDirectory(prefix="omni-retained-inert-"); self.addCleanup(temporary.cleanup)
        policy = Policy()
        pager = P.WorkingAttentionPager(Path(temporary.name), resident_budget_bytes=ram, scratch_budget_bytes=scratch,
            device_tile_budget_bytes=1024, resource_policy=policy, page_tokens=4,
            retained_context_pools=tuple(pools if pools is not None else [pool("cuda:0"), pool("cuda:1")]))
        self.addCleanup(pager.close)
        return pager, policy

    def test_shared_fixture_distinct_current_storage_pools_exclude_unified_unknown_and_unsupported(self):
        self.assertEqual(C.retained_context_pools(FIXTURE["topology"], FIXTURE["budgets"]), FIXTURE["expectedPools"])
        self.assertEqual(sum(item["budgetBytes"] for item in FIXTURE["expectedPools"]), 6000)
        self.assertFalse(any(item["nativeComputeSupported"] for item in FIXTURE["topology"]["devices"] if item["backend"] == "cuda"))

    def test_shared_fixture_context_metadata_and_full_checkpoint_arithmetic(self):
        keys = {"ramWorkingBytes":"ram_working_bytes", "deviceContextBytes":"device_context_bytes", "kvBytesPerToken":"kv_bytes_per_token",
            "indexBytesPerToken":"index_bytes_per_token", "modelLimitTokens":"model_limit_tokens", "spillCapacityBytes":"spill_capacity_bytes",
            "pagedKvBytesPerToken":"paged_kv_bytes_per_token", "checkpointCapacityBytes":"checkpoint_capacity_bytes", "pageTokens":"page_tokens", "layers":"layers",
            "sharedStorageCapacityBytes":"shared_storage_capacity_bytes"}
        output = {"residentMaximumTokens":"residentMaximumTokens", "storageMaximumTokens":"storageMaximumTokens",
                  "maximumTokens":"maximumTokens", "metadataMaximumTokens":"metadataMaximumTokens"}
        for case in FIXTURE["capacityCases"]:
            with self.subTest(case=case["name"]):
                values = {keys[key]: value for key, value in case["input"].items()}
                self.assertEqual(C.context_capacity(**values), {output[key]: value for key, value in case["expected"].items()})
                self.assertEqual(C.context_snapshot_bytes(case["snapshotTokens"], **{key: values[key] for key in
                    ("kv_bytes_per_token", "index_bytes_per_token", "page_tokens", "layers")}), case["snapshotBytes"])

    def test_disabled_identity_and_transient_free_observations_do_not_inflate_or_shrink_selected_ceiling(self):
        topology, budgets = copy.deepcopy(FIXTURE["topology"]), copy.deepcopy(FIXTURE["budgets"])
        device = next(item for item in topology["devices"] if item["id"] == "cuda:physical-a")
        device["availableBytes"] = 0
        self.assertEqual(C.retained_context_pools(topology, budgets)[0]["budgetBytes"], 2400)
        for budget in budgets:
            if budget["memoryGroupId"] == "gpu:a": budget["enabled"] = False
        self.assertEqual([item["deviceId"] for item in C.retained_context_pools(topology, budgets)], ["cuda:physical-b"])

    def test_shared_manual_fixture_storage_only_full_owner_and_missing_minimum(self):
        result = C.retained_context_pools(FIXTURE["topology"], FIXTURE["budgets"], manual=True,
            minimums=FIXTURE["manualMinimums"], page_bytes=1000)
        self.assertEqual({item["executionDevice"]: item["budgetBytes"] for item in result}, FIXTURE["manualExpectedBudgets"])
        self.assertEqual(result[0]["minimumPackedOwnerBytes"], 0)
        self.assertEqual(result[1]["minimumPackedOwnerBytes"], 3000)
        self.assertTrue(all(item["partitionMode"] == "manual-admitted-minimums" for item in result))
        missing = C.retained_context_pools(FIXTURE["topology"], FIXTURE["budgets"], manual=True)
        self.assertEqual([item["budgetBytes"] for item in missing], [2400, 3600])
        self.assertTrue(all(item["partitionMode"] == "manual-conservative-missing-minimum" for item in missing))

    def test_several_tiny_GPU_pools_cannot_combine_to_place_one_indivisible_page(self):
        result = C.retained_context_pools(FIXTURE["topology"], FIXTURE["budgets"], page_bytes=4096)
        self.assertGreater(sum(item["budgetBytes"] for item in FIXTURE["expectedPools"]), 4096)
        self.assertEqual(sum(item["budgetBytes"] for item in result), 0)

    def test_actual_page_owner_overflows_RAM_into_two_admitted_pools_before_disk(self):
        pager, policy = self.pager()
        ids = [pager.save_page(BytesTensor(bytes([number]) * 4), retained_context=True) for number in range(5)]
        self.assertEqual([str(pager._pages[item].tensor.device) for item in ids], ["cpu", "cuda:0", "cuda:0", "cuda:1", "cuda:1"])
        self.assertEqual(pager.status()["residentBytes"], 4)
        self.assertEqual(pager.status()["retainedDeviceBytes"], 16)
        self.assertEqual(pager.status()["spillBytes"], 0)
        self.assertEqual(len([call for call in policy.calls if call[0].startswith("cuda:")]), 4)

    def test_mixed_page_read_converts_fragments_before_cat_and_checkpoint_reads_CPU(self):
        pager, _ = self.pager()
        sequence = pager.sequence(BytesTensor(b"abcdefgh"), axis=0)
        self.addCleanup(sequence.close)
        self.assertEqual(sequence.read(0, 8, Device("cuda:1")).values, b"abcdefgh")
        self.assertEqual(str(sequence.read(0, 8, Device("cuda:1")).device), "cuda:1")
        self.assertTrue(all(str(pager.read_page(item).device) == "cpu" for item in sequence.pages))

    def test_fresh_device_pressure_demotes_only_after_successful_CPU_transfer(self):
        pager, policy = self.pager(ram=8)
        identifier = pager.save_page(BytesTensor(b"abcd", "cuda:0"), retained_context=True)
        policy.blocked.add("cuda:0")
        pager.reclaim_device(Device("cuda:0"), 1)
        self.assertEqual(str(pager._pages[identifier].tensor.device), "cpu")
        self.assertEqual(pager.status()["retainedDeviceBytes"], 0)
        self.assertEqual(pager.read_page(identifier).values, b"abcd")

    def test_failed_demotion_and_Stop_keep_original_page_and_charge(self):
        for stop in (False, True):
            with self.subTest(stop=stop):
                pager, policy = self.pager(ram=0)
                identifier = pager.save_page(BytesTensor(b"abcd", "cuda:0"), retained_context=True)
                original = pager._pages[identifier].tensor
                policy.blocked.add("cuda:0")
                pager._cancelled = lambda: stop
                with self.assertRaises(P.WorkingAttentionCancelled if stop else Pause):
                    pager.reclaim_device(Device("cuda:0"), 1)
                self.assertIs(pager._pages[identifier].tensor, original)
                self.assertEqual(pager.status()["retainedDeviceBytes"], 4)

    def test_verified_GPU_spill_keeps_bytes_and_releases_only_actual_resident_owner(self):
        # The payload is four bytes, not a four-byte physical allocation.
        # Windows lacks statvfs and the real pager conservatively reserves a
        # 64-KiB allocation unit. Exercise both geometries on every host;
        # never weaken production physical-quota admission for this fixture.
        for allocation_unit in (4096, 65536):
            with self.subTest(allocationUnitBytes=allocation_unit):
                pager, policy = self.pager(ram=0, scratch=allocation_unit - 1)
                pager._allocation_unit = allocation_unit  # Explicit inert filesystem geometry.
                identifier = pager.save_page(BytesTensor(b"abcd", "cuda:0"), retained_context=True)
                original = pager._pages[identifier].tensor
                policy.blocked.add("cuda:0")
                with self.assertRaises(Pause) as refused:
                    pager.reclaim_device(Device("cuda:0"), 1)
                self.assertEqual(refused.exception.status["requiredAdditionalSpillBytes"], allocation_unit)
                self.assertIs(pager._pages[identifier].tensor, original)
                self.assertIsNone(pager._pages[identifier].path)
                pager.scratch_budget_bytes = allocation_unit * 2
                pager.reclaim_device(Device("cuda:0"), 1)
                self.assertIsNone(pager._pages[identifier].tensor)
                self.assertEqual(pager.status()["retainedDeviceBytes"], 0)
                self.assertEqual(pager.read_page(identifier).values, b"abcd")
                self.assertGreater(pager.status()["spillBytes"], 0)
                self.assertLessEqual(pager.status()["spillBytes"], pager.scratch_budget_bytes)

    def test_transactional_tail_replacement_failure_preserves_original_sequence(self):
        pager, _ = self.pager()
        sequence = pager.sequence(BytesTensor(b"abc"), axis=0); self.addCleanup(sequence.close)
        before = list(sequence.pages), list(sequence.ends), sequence.shape
        save = pager.save_page; calls = [0]
        def fail_second(value, **kwargs):
            calls[0] += 1
            if calls[0] == 2: raise Pause("inert second admission", {})
            return save(value, **kwargs)
        with patch.object(pager, "save_page", fail_second), self.assertRaises(Pause):
            sequence.append(BytesTensor(b"de"))
        self.assertEqual((sequence.pages, sequence.ends, sequence.shape), before)
        self.assertEqual(sequence.read(0, 3, Device("cpu")).values, b"abc")
        self.assertEqual(pager.status()["livePages"], 1)

    def test_partial_tail_replacement_uses_only_temporary_transfer_credit_not_fork_credit(self):
        pager, _ = self.pager(ram=0, pools=[pool("cuda:0", 4)])
        sequence = pager.sequence(BytesTensor(b"abc"), axis=0); self.addCleanup(sequence.close)
        sequence.append(BytesTensor(b"d"))
        self.assertEqual(sequence.read(0, 4, Device("cpu")).values, b"abcd")
        self.assertEqual(pager.status()["retainedDeviceBytes"], 4)
        separate, _ = self.pager(ram=0, pools=[pool("cuda:0", 4)])
        original = separate.sequence(BytesTensor(b"abc"), axis=0); self.addCleanup(original.close)
        branch = original.fork(); self.addCleanup(branch.close)
        with self.assertRaises(Pause): branch.append(BytesTensor(b"d"))
        self.assertEqual(original.read(0, 3, Device("cpu")).values, b"abc")
        self.assertEqual(branch.read(0, 3, Device("cpu")).values, b"abc")
        self.assertEqual(separate.status()["retainedDeviceBytes"], 3)

    def test_shared_GPU_page_references_do_not_duplicate_capacity_or_release_early(self):
        pager, _ = self.pager(ram=0)
        sequence = pager.sequence(BytesTensor(b"abcd"), axis=0)
        fork = sequence.fork(); self.addCleanup(fork.close)
        self.assertEqual(pager.status()["retainedDeviceBytes"], 4)
        sequence.close()
        self.assertEqual(fork.read(0, 4, Device("cpu")).values, b"abcd")
        self.assertEqual(pager.status()["retainedDeviceBytes"], 4)
        fork.close(); self.assertEqual(pager.status()["retainedDeviceBytes"], 0)

    def test_allocation_race_falls_back_but_cancellation_is_never_swallowed(self):
        pager, _ = self.pager(ram=0, scratch=8192)
        with patch.object(BytesTensor, "fail_target", "cuda:0"):
            identifier = pager.save_page(BytesTensor(b"abcd"), retained_context=True)
        self.assertEqual(str(pager._pages[identifier].tensor.device), "cuda:1")
        pager._cancelled = lambda: True
        with self.assertRaises(P.WorkingAttentionCancelled): pager.save_page(BytesTensor(b"next"), retained_context=True)

    def test_paged_checkpoint_restore_uses_CPU_values_and_current_pool_admission(self):
        pager, _ = self.pager(ram=0)
        source = pager.sequence(BytesTensor(b"abcdefgh"), axis=0); self.addCleanup(source.close)
        # Execute the real encoder's exact paged branch, not a replacement
        # codec. Payload I/O here is an inert byte dictionary, not safetensors.
        tree = ast.parse((ROOT / "engine/omni_core/execution_state_codec.py").read_text())
        writer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExecutionStateWriter")
        encode = next(node for node in writer.body if isinstance(node, ast.FunctionDef) and node.name == "encode")
        branch = next(node for node in encode.body if isinstance(node, ast.If) and "isinstance(value, PagedTensorSequence)" == ast.unparse(node.test))
        method = copy.deepcopy(encode); method.body = [branch]; method.decorator_list = []
        reader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExecutionStateReader")
        decode = next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "decode")
        namespace = dict(PagedTensorSequence=P.PagedTensorSequence, SavedActivityTensor=type("NoSavedTraining", (), {}), math=math)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method, decode], type_ignores=[])), "owned-paged-codec-branches", "exec"), namespace)
        values = {}
        def tensor(value):
            self.assertEqual(str(value.device), "cpu")
            name = str(len(values)); values[name] = value.values; return {"file": name}
        encoded = namespace["encode"](types.SimpleNamespace(tensor=tensor), source)
        target, _ = self.pager(ram=0)
        owner = types.SimpleNamespace(pager=target, _owners=[], tensor=lambda name, page: BytesTensor(values[name]))
        restored = namespace["decode"](owner, encoded); self.addCleanup(restored.close)
        self.assertEqual(restored.read(0, 8, Device("cpu")).values, b"abcdefgh")
        self.assertEqual(target.status()["retainedDeviceBytes"], 8)
        self.assertEqual(restored._page_tokens, source._page_tokens)
        blocked, policy = self.pager(ram=0)
        policy.blocked.update({"cuda:0", "cuda:1"})
        failed = types.SimpleNamespace(pager=blocked, _owners=[], tensor=lambda name, page: BytesTensor(values[name]))
        with self.assertRaises(Pause): namespace["decode"](failed, encoded)
        for item in failed._owners: item.close()
        self.assertEqual(values, {"0": b"abcd", "1": b"efgh"})

    def test_manual_core_reclaim_refuses_pins_or_transactions_before_eviction_and_obeys_Stop(self):
        reclaim = source_method("engine/omni_core/native_core_paging.py", "NativeCorePager", "reclaim_device_for_context")
        for failure in ("pin", "transaction", "Stop", "writeback"):
            with self.subTest(failure=failure):
                module = type("InertPackedModule", (), {})(); module._online_transaction = object() if failure == "transaction" else None
                owner = types.SimpleNamespace(device=Device("cuda:0"), pins=1 if failure == "pin" else 0, heat=1)
                events = []
                def pause(message): raise Pause(message, {})
                def stopped():
                    if failure == "Stop": raise P.WorkingAttentionCancelled("inert Stop")
                def evict(_module, *, cancelled):
                    cancelled(); events.append("writeback")
                    if failure == "writeback": raise Pause("inert writeback pressure", {})
                current = types.SimpleNamespace(_lock=nullcontext(), _owners={module: owner}, _pause=pause, evict=evict,
                    shared_resource_policy=types.SimpleNamespace(device_budget=lambda *_, **__: {"enabled": True, "capacityMeasured": True, "remainingBytes": 0}),
                    _accelerator_usage=lambda: {"cuda:0": 8})
                with self.assertRaises((Pause, P.WorkingAttentionCancelled)):
                    reclaim(current, Device("cuda:0"), 4, core_minimum_bytes=4, cancelled=stopped)
                self.assertEqual(str(owner.device), "cuda:0")
                self.assertEqual(events, ["writeback"] if failure == "writeback" else [])

    def test_failed_manual_core_cold_copy_does_not_register_unwritten_backing(self):
        cool = source_method("engine/omni_core/native_core_paging.py", "NativeCorePager", "cool_to_budget")
        source = BytesTensor(bytes(2048)); module = type("InertModule", (), {})(); module._buffers = {"weight": source}
        owner = types.SimpleNamespace(heat=0, pins=0, loaded=True, cold={"weight": source}, maps={}, device=None)
        deleted = []
        def mapping(actual, name, *_args):
            actual.maps[name] = types.SimpleNamespace(path="owned-new-map")
            return BytesTensor(bytes(2048))
        def fail(*_args, **_kwargs): raise Pause("inert bounded transfer failure", {})
        current = types.SimpleNamespace(_lock=nullcontext(), _owners={module: owner}, _observe_mutations=lambda *_: None,
            cpu_hot_bytes=0, _heap_bytes=lambda: 2048, _map_tensor=mapping, _copy_bounded=fail,
            _unlink_owned_page=lambda path: deleted.append(path))
        with self.assertRaises(Pause): cool(current, 0, cancelled=lambda: None)
        self.assertIs(owner.cold["weight"], source); self.assertIs(module._buffers["weight"], source)
        self.assertEqual(owner.maps, {}); self.assertEqual(deleted, ["owned-new-map"])

    def test_live_context_policy_change_refuses_before_retiring_sessions_or_mutating_configuration(self):
        update = source_method("engine/omni_core/brain.py", "AdaptiveBrain", "update_config")
        events = []
        config = types.SimpleNamespace(working_memory_mode="auto", max_seq_len=1024)
        current = types.SimpleNamespace(config=config, working_attention_pager=types.SimpleNamespace(status=lambda: {"livePages": 1}),
            core_pager=types.SimpleNamespace(status=lambda: {"pinnedOwners": 0}),
            _native_operation_auto=types.SimpleNamespace(close=lambda: events.append("retired")))
        with self.assertRaises(Pause): update(current, {"workingMemoryMode": "manual"})
        self.assertEqual(config.working_memory_mode, "auto"); self.assertEqual(events, [])


if __name__ == "__main__": unittest.main()
