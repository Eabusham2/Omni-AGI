"""NEW constructor-free actual-source edges, no Torch/model import or run."""
import ast
import copy
import hashlib
import json
import threading
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def methods(file, owner, names, namespace):
    tree = ast.parse((ROOT / file).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    selected = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names: continue
        node = copy.deepcopy(node)
        node.decorator_list, node.returns = [], None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs): arg.annotation = None
        selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "new-actual-decoder-source-boundary", "exec"), namespace)
    return namespace


class TinyTensor:
    def __init__(self, values=None, version=0):
        self._version, self.values = version, list(values or [])
        self.shape, self.dtype = (len(self.values),), "uint8"
    def numel(self): return len(self.values)
    def element_size(self): return 1
    def reshape(self, _shape): return self
    def __getitem__(self, selection):
        class Slice:
            def __init__(self, tensor, selection): self.tensor, self.selection = tensor, selection
            def copy_(self, source, non_blocking=False):
                self.tensor.values[self.selection] = source.tensor.values[source.selection]
        return Slice(self, selection)


class DecoderSourceBoundaries(unittest.TestCase):
    def test_actual_pager_learned_epoch_excludes_cache_and_known_moves_but_catches_strength_repeats_replacement(self):
        namespace = methods("omni_core/native_core_paging.py", "NativeCorePager",
            {"_observe_mutations", "_rebind_learned_tokens"}, {"torch": types.SimpleNamespace(Tensor=TinyTensor)})
        module = types.SimpleNamespace(_buffers={"_packed": TinyTensor(), "_strength": TinyTensor(),
            "_record_meta": TinyTensor(), "_sort_order": TinyTensor()}, _non_persistent_buffers_set={"_sort_order"})
        owner = types.SimpleNamespace(cold=dict(module._buffers), learned_tokens={}, versions={name: 0 for name in module._buffers},
            maps={}, device=None, dirty_names=set())
        pager = types.SimpleNamespace(_learning_revision=0, _revision=0, _learning_revision_active=True)
        namespace["_rebind_learned_tokens"](module, owner)
        observe = lambda: namespace["_observe_mutations"](pager, module, owner)
        module._buffers["_sort_order"]._version += 1
        observe()
        self.assertEqual((pager._learning_revision, pager._revision), (0, 1))
        module._buffers["_packed"] = TinyTensor(version=9)  # Known exact .to/writeback representation move.
        owner.versions["_packed"] = 9
        namespace["_rebind_learned_tokens"](module, owner)
        observe()
        self.assertEqual(pager._learning_revision, 0)
        module._buffers["_strength"]._version += 1; observe()
        module._buffers["_record_meta"]._version += 1; observe()
        self.assertEqual(pager._learning_revision, 2)
        module._buffers["_packed"] = TinyTensor(version=9)  # Unknown replacement, same numeric _version.
        observe()
        self.assertEqual(pager._learning_revision, 3)

    def test_actual_bounded_packed_copy_accumulates_real_bytes_not_only_largest_transfer(self):
        namespace = methods("omni_core/native_core_paging.py", "NativeCorePager", {"_copy_bounded"}, {})
        pager = types.SimpleNamespace(chunk_bytes=3, _transfer_bytes=0, _peak_transfer_bytes=0)
        source, target = TinyTensor(range(8)), TinyTensor([0] * 8)
        self.assertEqual(namespace["_copy_bounded"](pager, source, target), 8)
        self.assertEqual(target.values, source.values)
        self.assertEqual((pager._transfer_bytes, pager._peak_transfer_bytes), (8, 3))

    def test_actual_placement_restores_original_controls_on_success_admission_error_and_flush_error(self):
        class Module:
            def __init__(self): self._buffers, self._parameters = {}, {}
        class RootModule(Module):
            def modules(self): return [self, packed]
        for failure in (None, "admission", "flush"):
            with self.subTest(failure=failure):
                packed = Module()
                root = RootModule()
                original = TinyTensor([1])
                root._buffers["gain"] = original
                root._native_compute_device = "cpu"
                root.blocks = []
                root.global_workspace = types.SimpleNamespace(query_chunk_slots=3)
                refreshed = []
                core = types.SimpleNamespace(refresh_budget=lambda **kw: refreshed.append(kw))
                def flush():
                    if failure == "flush": raise RuntimeError("inert flush error")
                core.flush = flush
                packed._native_core_pager = core
                packed._native_compute_device = "cpu"
                working = types.SimpleNamespace(compute_device="cpu", resident_budget_bytes=1,
                    scratch_budget_bytes=2, device_tile_budget_bytes=3)
                brain = types.SimpleNamespace(decoder=root, core_pager=core, _trainable_modules=lambda: [root],
                    _native_state_tensors=lambda value: iter([value]) if isinstance(value, TinyTensor) else iter(()),
                    liquid_state=original, device="cpu", device_backend="cpu", _native_backend_residency={"actualBackend": "cpu"},
                    _native_backend_last_changed=7, resource_policy=types.SimpleNamespace(execution_device="cpu"),
                    working_attention_pager=working)
                def prepare(**kwargs):
                    self.assertEqual(kwargs["reviewed_target"], "mps")
                    root._buffers["gain"] = TinyTensor([2])
                    root._native_compute_device = "mps"
                    brain.device, brain.device_backend, brain.liquid_state = "mps", "mps", TinyTensor([9])
                    brain.resource_policy.execution_device = "mps"
                    working.compute_device, working.resident_budget_bytes = "mps", 9
                    root.global_workspace.query_chunk_slots = 1
                    if failure == "admission": raise RuntimeError("inert admission error")
                brain.prepare_native_execution = prepare
                namespace = methods("omni_core/decoder_execution_native.py", "NativeDecoderExecutionOps", {"_placement"},
                    {"torch": types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module)), "copy": copy,
                     "canonical_device": str})
                ops = types.SimpleNamespace(brain=brain, quiescent=lambda: None)
                enter = contextmanager(namespace["_placement"])
                if failure:
                    with self.assertRaisesRegex(RuntimeError, failure):
                        with enter(ops, {"executionDevice": "mps"}): pass
                else:
                    with enter(ops, {"executionDevice": "mps"}):
                        self.assertEqual(brain.device, "mps")
                self.assertIs(root._buffers["gain"], original)
                self.assertIs(brain.liquid_state, original)
                self.assertEqual((brain.device, brain.device_backend, working.compute_device), ("cpu", "cpu", "cpu"))
                self.assertEqual((working.resident_budget_bytes, root.global_workspace.query_chunk_slots), (1, 3))
                self.assertEqual(root._native_compute_device, "cpu")
                self.assertEqual(bool(getattr(brain, "_native_execution_restore_required", False)), failure == "flush")

    def test_actual_control_dispatch_bypasses_get_and_neural_ops_and_keeps_exact_operation_id(self):
        class Fault(RuntimeError): pass
        namespace = methods("worker.py", "Worker", {"dispatch_control", "_decoder_control_owner",
            "decoder_execution_status", "decoder_execution_stop"}, {"RpcFault": Fault, "MCP_SAMPLING_CONTROLS": set()})
        calls = []
        brain = types.SimpleNamespace(storage_path="owned", decoder_execution_status=lambda: {"phase": "calibrating"},
            decoder_execution_stop=lambda **kw: calls.append(kw) or {"phase": "calibrating"})
        worker = types.SimpleNamespace(worker_role="neural", brains={"owned": brain},
            _brain_id=lambda params: params["brainId"], _storage=lambda params, owner: params.get("storagePath", owner),
            _get=lambda *_args: self.fail("flags-only control invoked model/Load getter"))
        for name in ("_decoder_control_owner", "decoder_execution_status", "decoder_execution_stop"):
            setattr(worker, name, types.MethodType(namespace[name], worker))
        def request(method, params):
            return namespace["dispatch_control"](worker, {"jsonrpc": "2.0", "id": "current", "method": method, "params": params})
        self.assertEqual(request("decoder_execution_status", {"brainId": "owned"})["result"]["phase"], "calibrating")
        request("decoder_execution_stop", {"brainId": "owned", "operationId": "exact-main-call"})
        self.assertEqual(calls, [{"operation_id": "exact-main-call"}])
        with self.assertRaises(Fault): request("decoder_execution_status", {"brainId": "missing"})
        with self.assertRaises(Fault): request("decoder_execution_status", {"brainId": "owned", "prompt": "not-authority"})

    def test_new_actual_candidate_union_binds_compute_and_home_completion_without_claiming_cuda_native(self):
        def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        namespace = methods("omni_core/decoder_execution_native.py", "NativeDecoderExecutionOps", {"candidates"},
            {"torch": types.SimpleNamespace(device=str), "canonical_device": str, "digest": digest})
        devices = [{"id": name, "executionDevice": name, "kind": "ram" if name == "cpu" else "gpu",
            "memoryGroupId": "ram:system" if name in {"cpu", "mps"} else "gpu:separate", "nativeComputeSupported": name in {"cpu", "mps"}}
            for name in ("cpu", "mps", "cuda:0", "xpu:0")]
        brain = types.SimpleNamespace(device="mps", _regional_topology=lambda: {"devices": devices},
            resource_policy=types.SimpleNamespace(device_budget=lambda *args, **kw: {"enabled": True, "capacityMeasured": True}))
        ops = types.SimpleNamespace(brain=brain)
        current, plans = namespace["candidates"](ops, False)
        self.assertEqual(current["executionDevice"], "mps")
        self.assertEqual([(row["executionDevice"], row["asyncDevices"]) for row in plans], [("mps", ["mps"]), ("cpu", ["mps"])])
        _current, reviewed = namespace["candidates"](ops, True)
        cuda = next(row for row in reviewed if row["executionDevice"] == "cuda:0")
        self.assertEqual(cuda["backendMode"], "explicit-cuda-cpu-tile")
        self.assertEqual(cuda["asyncDevices"], ["cuda:0", "mps"])
        self.assertTrue(cuda["fallbackAcknowledged"])
        self.assertFalse(any(row["executionDevice"].startswith("xpu") for row in reviewed))

    def test_new_actual_registry_streams_metadata_and_fences_real_controls_not_disposable_cache(self):
        namespace = methods("omni_core/decoder_execution_native.py", "NativeDecoderExecutionOps", {"_registry"},
            {"torch": types.SimpleNamespace(Tensor=TinyTensor), "hashlib": hashlib, "json": json})
        reserved = []
        def reserve(size, _label): reserved.append(size); return nullcontext()
        module = types.SimpleNamespace(_buffers={"state": TinyTensor([1]), "cache": TinyTensor([2])},
            _non_persistent_buffers_set={"cache"}, query_chunk_slots=4, _packed_structural_fields={"_next_page"}, _next_page=1)
        pager = types.SimpleNamespace(_owners={"one": 1})
        ops = types.SimpleNamespace(brain=types.SimpleNamespace(decoder=types.SimpleNamespace(named_modules=lambda: [("actual", module)]),
            core_pager=pager, resource_policy=types.SimpleNamespace(reserve_ram=reserve)))
        first, owner = namespace["_registry"](ops)
        module._buffers["cache"].shape = (999,)
        self.assertEqual(namespace["_registry"](ops)[0], first)
        module._next_page += 1
        second = namespace["_registry"](ops)[0]
        self.assertNotEqual(second, first)
        module.query_chunk_slots += 1
        self.assertNotEqual(namespace["_registry"](ops)[0], second)
        self.assertIs(owner, pager)
        self.assertEqual(reserved, [65536] * 4)

    def test_new_actual_loaded_admission_refuses_other_live_inline_job_without_stopping_it(self):
        class Fault(RuntimeError): pass
        namespace = methods("worker.py", "Worker", {"_decoder_loaded_owner"}, {"RpcFault": Fault, "nullcontext": nullcontext})
        brain = types.SimpleNamespace(brain_id="owned")
        stopped = []
        job = types.SimpleNamespace(brain_id="owned", future=types.SimpleNamespace(done=lambda: False, cancel=lambda: stopped.append(1)))
        worker = types.SimpleNamespace(_regional_loaded_owner=lambda params, extra: brain,
            _inline_generations={"job": job})
        with self.assertRaisesRegex(Fault, "independent inline"):
            namespace["_decoder_loaded_owner"](worker, {"brainId": "owned"}, set())
        self.assertEqual(stopped, [])
        job.brain_id = "different-owner"
        self.assertIs(namespace["_decoder_loaded_owner"](worker, {"brainId": "owned"}, set()), brain)

    def test_new_actual_live_epoch_is_opt_in_resource_admitted_and_retires_metadata(self):
        namespace = methods("omni_core/native_core_paging.py", "NativeCorePager",
            {"_observe_mutations", "_rebind_learned_tokens", "learning_revision", "retire_learning_observation"},
            {"torch": types.SimpleNamespace(Tensor=TinyTensor)})
        class Module: pass
        module = Module()
        module._buffers, module._non_persistent_buffers_set = {"state": TinyTensor([1])}, set()
        owner = types.SimpleNamespace(cold=dict(module._buffers), learned_tokens={}, versions={"state": 0}, maps={}, device=None, dirty_names=set())
        reservations, releases = [], []
        class Lease:
            def mark_allocated(self, size): reservations.append(("allocated", size))
            def release(self): releases.append(1)
        def reserve(size, label): reservations.append((label, size)); return Lease()
        pager = types.SimpleNamespace(_learning_revision=7, _revision=0, _learning_revision_active=False,
            _learning_tracking_lease=None, _lock=threading.RLock(), _owners={module: owner},
            shared_resource_policy=types.SimpleNamespace(reserve_ram=reserve))
        pager._rebind_learned_tokens = namespace["_rebind_learned_tokens"]
        pager._observe_mutations = types.MethodType(namespace["_observe_mutations"], pager)
        pager._observe_mutations(module, owner)
        self.assertEqual(owner.learned_tokens, {})
        self.assertEqual(namespace["learning_revision"](pager), 8)
        self.assertEqual(reservations, [("opt-in live packed-owner version metadata", 448), ("allocated", 448)])
        module._buffers["state"]._version += 1
        self.assertEqual(namespace["learning_revision"](pager), 9)
        namespace["retire_learning_observation"](pager)
        self.assertFalse(pager._learning_revision_active)
        self.assertEqual(owner.learned_tokens, {})
        self.assertEqual(releases, [1])


if __name__ == "__main__": unittest.main()
