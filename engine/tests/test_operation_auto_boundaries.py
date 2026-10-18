"""NEW actual source seams, constructor-free owners and tiny raw CPU tensors.

No brain/module initialization, native kernel, Load, training, provider, device
benchmark or neural calibration. The fake host is not real dispatch evidence.
"""
from __future__ import annotations

import ast
import base64
import binascii
import copy
import importlib
import sys
import threading
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "owned_perception_native_boundary"
package = types.ModuleType(PACKAGE); package.__path__ = [str(ROOT / "omni_core")]; sys.modules[PACKAGE] = package
NATIVE = importlib.import_module(PACKAGE + ".operation_auto_native")
MODAL = importlib.import_module(PACKAGE + ".modalities")
CONNECTED = importlib.import_module(PACKAGE + ".connected_regions")
MODEL = importlib.import_module(PACKAGE + ".model")
CONTACTS = importlib.import_module(PACKAGE + ".packed_connections")


def methods(relative, owner, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    selected = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    for node in selected: node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(ROOT / relative), "exec"), namespace)
    return namespace


class Lease:
    def __init__(self): self.closed = False
    def mark_allocated(self, size): self.size = size
    def release(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *_args): self.release()


class Policy:
    def __init__(self): self.leases, self.admissions = [], []
    def reserve_ram(self, size, label):
        self.admissions.append((int(size), label)); lease = Lease(); self.leases.append(lease); return lease
    def require_device(self, device, size, label): self.admissions.append((int(size), label))


class Owner:
    def __init__(self, **values): self.__dict__.update(values)


def empty_module(kind, *, children=None, buffers=None, attrs=None):
    # No nn.Module or learned-module __init__ runs. Only declared tiny metadata
    # and raw tensors are supplied to read-only source methods.
    module = object.__new__(kind)
    module.__dict__.update({"_modules": children or {}, "_buffers": buffers or {}, "_parameters": {},
        "_non_persistent_buffers_set": set(), "_forward_hooks": {}, "_forward_pre_hooks": {}, "_backward_hooks": {},
        "training": True, **(attrs or {})})
    return module


def metadata_fixture():
    policy = Policy(); pager = types.SimpleNamespace(_owners={})
    pool = empty_module(CONTACTS.PooledParallelConnections, children={"pages": empty_module(torch.nn.ModuleDict)},
                        buffers={"_next_page": torch.zeros(8, dtype=torch.uint8)}, attrs={"_native_core_pager": pager})
    conv = empty_module(MODEL.PackedAdaptiveBitConv2d, children={"parallel_connections": pool},
        buffers={"_packed_forward_weight": torch.zeros((2, 1), dtype=torch.uint8)},
        attrs={"_native_core_pager": pager, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "dilation": (1, 1),
            "groups": 1, "in_channels": 3, "out_channels": 2, "_matrix_width": 27, "inplace": False})
    projection = empty_module(MODEL.PackedAdaptiveBitLinear, buffers={"_packed_forward_weight": torch.zeros((4, 1), dtype=torch.uint8)},
        attrs={"_native_core_pager": pager, "in_features": 2, "out_features": 4})
    encoder = empty_module(torch.nn.Sequential, children={"0": conv, "1": empty_module(torch.nn.SiLU, attrs={"inplace": False}),
                                                        "2": empty_module(torch.nn.AdaptiveAvgPool2d, attrs={"output_size": 1})})
    vision = empty_module(MODAL.TinyVisionEncoder, children={"features": encoder, "projection": projection})
    config = types.SimpleNamespace(vision_enabled=True, image_enabled=True, audio_enabled=False, video_enabled=False,
        image_size=8, audio_samples=16, video_frames=2, to_dict=lambda: {"vision": True, "imageSize": 8})
    leaf = empty_module(MODAL.ModalityHub, children={"vision": vision}, attrs={"config": config})
    pager._owners = {conv: types.SimpleNamespace(loaded=True, device=None), projection: types.SimpleNamespace(loaded=True, device=None)}
    brain = Owner(core_pager=pager, resource_policy=policy, modalities=leaf)
    ops = object.__new__(NATIVE.NativeOperationAutoOps); ops._owner = weakref.ref(brain); ops._stop_event = None
    ops._scopes = {}; ops._activation_transfer_bytes = 0
    return brain, ops, leaf, pool, conv, projection


class NativePerceptionBoundaries(unittest.TestCase):
    def test_worker_control_status_plans_stop_are_exact_owner_bytes_only(self):
        class Fault(RuntimeError): pass
        names = {"dispatch_control", "_decoder_control_owner", "native_operation_auto_status", "native_operation_auto_stop", "native_operation_auto_plans"}
        ns = methods("worker.py", "Worker", names, {"RpcFault": Fault, "MCP_SAMPLING_CONTROLS": set()})
        calls = []
        brain = types.SimpleNamespace(storage_path="owned", native_operation_auto_status=lambda: {"phase": "qualifying"},
            native_operation_auto_stop=lambda **kw: calls.append(("stop", kw)) or {}, native_operation_auto_plans=lambda **kw: calls.append(("plans", kw)) or {"total": 0})
        worker = types.SimpleNamespace(worker_role="neural", brains={"owned": brain}, _brain_id=lambda p: p["brainId"],
            _storage=lambda p, owner: p.get("storagePath", owner), _get=lambda *_args: self.fail("flags control attempted neural getter/Load"))
        for name in names - {"dispatch_control"}: setattr(worker, name, types.MethodType(ns[name], worker))
        def invoke(name, params): return ns["dispatch_control"](worker, {"jsonrpc": "2.0", "id": "r", "method": name, "params": params})["result"]
        self.assertEqual(invoke("native_operation_auto_status", {"brainId": "owned"}), {"phase": "qualifying"})
        invoke("native_operation_auto_plans", {"brainId": "owned"}); invoke("native_operation_auto_stop", {"brainId": "owned", "operationId": "exact"})
        self.assertEqual(calls, [("plans", {"after": None, "limit": 32}), ("stop", {"operation_id": "exact"})])
        for params in ({"brainId": "absent"}, {"brainId": "owned", "storagePath": "different"}, {"brainId": "owned", "tensor": "untrusted"}):
            with self.assertRaises(Fault): invoke("native_operation_auto_status", params)
        with self.assertRaises(Fault): invoke("native_operation_auto_release_stopped", {"brainId": "owned"})

    def test_actual_prepare_arm_are_loaded_serial_metadata_and_reject_inline_job(self):
        class Fault(RuntimeError): pass
        ns = methods("worker.py", "Worker", {"_decoder_loaded_owner", "native_operation_auto_prepare", "native_operation_auto_arm"},
                     {"RpcFault": Fault, "nullcontext": nullcontext})
        calls, event = [], threading.Event()
        brain = types.SimpleNamespace(brain_id="owned", native_operation_auto_prepare=lambda **kw: calls.append(("prepare", kw)),
                                      native_operation_auto_arm=lambda **kw: calls.append(("arm", kw)))
        worker = types.SimpleNamespace(_regional_loaded_owner=lambda _p, _extra: brain, _inline_generations={}, _cooperative_cancel=event)
        worker._decoder_loaded_owner = types.MethodType(ns["_decoder_loaded_owner"], worker)
        ns["native_operation_auto_prepare"](worker, {"brainId": "owned", "topology": {"observed": True}}, "r")
        ns["native_operation_auto_arm"](worker, {"brainId": "owned", "operationId": "exact", "reviewed": {"fence": True}}, "r")
        self.assertEqual(calls, [("prepare", {"topology": {"observed": True}, "acknowledge_cuda": False}),
                                ("arm", {"operation_id": "exact", "reviewed": {"fence": True}, "samples": 3})])
        self.assertIs(brain._native_operation_auto_cancel_flags, event)
        worker._inline_generations["not-cancelled"] = types.SimpleNamespace(brain_id="owned", future=types.SimpleNamespace(done=lambda: False))
        with self.assertRaises(Fault): ns["native_operation_auto_arm"](worker, {"brainId": "owned"}, "r")
        self.assertEqual(len(calls), 2); self.assertIn("not-cancelled", worker._inline_generations)

    def test_failed_exact_restore_pauses_current_packet_owner_without_close_or_load(self):
        class Fault(RuntimeError):
            def __init__(self, code, message, data=None): super().__init__(message); self.code, self.data = code, data
        ns = methods("worker.py", "Worker", {"observe_packet"}, {"RpcFault": Fault, "base64": base64, "binascii": binascii,
            "_number": float, "is_allocator_oom_error": lambda _e: self.fail("restore guard fell into allocator/reload branch"),
            "AdaptiveBrain": types.SimpleNamespace(load=lambda *_a, **_kw: self.fail("failed restore triggered Load"))})
        def failed(**_kwargs): raise RuntimeError("owned exact restoration failed")
        brain = types.SimpleNamespace(brain_id="owned", storage_path="owned", _native_execution_restore_required=True,
            observe_live_packet=failed, close=lambda: self.fail("failed restore closed current brain"))
        session = types.SimpleNamespace(brain_id="owned", modalities=("image",), max_packet_bytes=16, last_sequence=-1,
            last_timestamp_ms=0, retention="neural", permission={"source": "owned-fixture"})
        worker = types.SimpleNamespace(_observation_lock=threading.RLock(), _observation_sessions={"s": session}, brains={"owned": brain})
        with self.assertRaises(Fault) as caught:
            ns["observe_packet"](worker, {"sessionId": "s", "modality": "image", "sequence": 0, "timestampMs": 1,
                "mimeType": "image/png", "dataBase64": base64.b64encode(b"xx").decode()}, "r")
        self.assertEqual(caught.exception.code, -32020)
        pause = caught.exception.data["resourcePause"]
        self.assertTrue(pause["reloadRequired"]); self.assertTrue(pause["nativeOwnerRestoreFailed"])
        self.assertFalse(pause["workerRestarted"]); self.assertFalse(pause["packetCommitted"])
        self.assertIs(worker.brains["owned"], brain); self.assertEqual(session.last_sequence, -1)

    def test_borrowed_connected_closure_keeps_aliases_and_cpu_ordinal_control_not_fake_pager_owner(self):
        brain, ops, leaf, pool, conv, projection = metadata_fixture()
        exchange = empty_module(CONNECTED.RegionExchange, children={"bridges": empty_module(torch.nn.ModuleDict)}, attrs={"widths": (4, 4)})
        regions = [types.SimpleNamespace(identity=types.SimpleNamespace(region_id="r" + str(i), parent_id="p" + str(i), generation_sha256=str(i) * 64),
                                        config=leaf.config, modalities=leaf) for i in range(2)]
        connected = empty_module(CONNECTED.ConnectedModalityHub, attrs={"connected_decoder": types.SimpleNamespace(regions=regions, workspace_exchange=exchange), "config": leaf.config})
        closure = ops._closure(connected, "image")
        self.assertEqual(len(closure["tree"]["children"]), 2)
        packed = ops._validate_closure(closure)
        self.assertEqual(set(packed), {conv, projection}); self.assertNotIn(pool, brain.core_pager._owners)
        scope = types.SimpleNamespace(snapshot=lambda: {str(id(conv)): 0, str(id(projection)): 0})
        first = ops._state(closure, scope, "session")
        pool._next_page.add_(1)
        self.assertNotEqual(ops._state(closure, scope, "session"), first)
        conv._forward_hooks["unknown"] = lambda *_args: None
        with self.assertRaisesRegex(RuntimeError, "callback"): ops._validate_closure(closure)

    def test_real_geometry_counts_int64_intermediates_and_refuses_inplace_unknown_closure(self):
        brain, ops, leaf, _pool, _conv, _projection = metadata_fixture()
        closure = ops._closure(leaf, "image")
        operation = {"tensor": torch.zeros((1, 3, 8, 8)), "modality": "image", "closure": closure}
        actual = ops._activity_bytes(operation)
        # 16 positions, 27 input columns, 2 outputs; actual known int64 rows and
        # concatenation are part of the existing declared primitive budget.
        kernel = NATIVE.native_projection_tensor_budget(16, 27, 2)
        self.assertEqual(kernel["integerRowBlockBytes"] + kernel["integerConcatenationBytes"], 16 * 2 * 8 * 2)
        self.assertGreater(actual, 65536 + kernel["forwardTensorReserveBytes"])
        leaf.vision.features[1].inplace = True
        with self.assertRaisesRegex(RuntimeError, "in-place"): ops._validate_closure(closure)

    def test_actual_execution_stage_restore_and_tiny_output_are_private_same_owner(self):
        policy, calls = Policy(), []
        packed = types.SimpleNamespace()  # A hashable fake owner, not a model.
        packed = type("OwnedPackedMetadata", (), {})(); packed._native_compute_device = torch.device("cpu")
        packed._buffers, packed._parameters = {}, {}
        class Pager:
            _lock = threading.RLock()
            _owners = {packed: types.SimpleNamespace(device=None, loaded=True)}
            def evict(self, module): calls.append("evict")
            def _observe_mutations(self, module, owner): calls.append("verify-persistent-mutations")
            def _regional_limits(self): calls.append("restore-budget")
        activity = types.SimpleNamespace(activate=lambda **_kwargs: nullcontext())
        def encode(modality, tensor):
            self.assertFalse(torch.is_grad_enabled()); self.assertEqual(modality, "image"); calls.append("pure-body")
            return tensor * 2
        brain = Owner(resource_policy=policy, core_pager=Pager(), working_attention_pager=activity,
            device=torch.device("cpu"), _regional_transfer_reservation=lambda *_args: nullcontext())
        ops = NATIVE.NativeOperationAutoOps(brain)
        ops.quiescent = lambda: calls.append("quiescent")
        ops.admit_candidate = lambda *_args: {"enabled": True}
        ops._activity_bytes = lambda _operation: 128
        ops.synchronize = lambda devices: calls.append(("sync", devices))
        tensor = torch.tensor([[1.0, 2.0]])
        operation = {"tensor": tensor, "modality": "image", "hub": types.SimpleNamespace(perception_embedding=encode),
                     "closure": {"modules": {id(packed): packed}}, "residency": [(packed, None)],
                     "ownerObjects": {packed: brain.core_pager._owners[packed]}}
        stages, flags = [], threading.Event()
        with ops.pure_scope(flags):
            result = ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, types.SimpleNamespace(stage_completed=stages.append), flags)
        self.assertEqual(stages, list(NATIVE.STAGES)); self.assertEqual(result.take().tolist(), [[2.0, 4.0]])
        result.close(); self.assertTrue(all(lease.closed for lease in policy.leases))
        self.assertEqual(tensor.tolist(), [[1.0, 2.0]]); self.assertEqual(str(packed._native_compute_device), "cpu")
        self.assertEqual(calls.count("pure-body"), 1); self.assertIn("restore-budget", calls)
        self.assertNotIn("evict", calls); self.assertIn("verify-persistent-mutations", calls)
        def failed_restore(_module): raise RuntimeError("fake restore refuses current allocation")
        brain.core_pager.evict = failed_restore
        brain.core_pager._owners[packed].device = torch.device("mps")  # Inert metadata only; no accelerator allocation.
        with ops.pure_scope(flags), self.assertRaises(NATIVE.OperationAutoRestoreRequired):
            ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, None, flags)
        self.assertTrue(brain._native_execution_restore_required)

    def test_restore_skips_unchanged_canonical_residency_but_restores_changed_and_refuses_new_owner(self):
        calls, policy = [], Policy()
        packed = type("OwnedPackedMetadata", (), {})(); packed._native_compute_device = "mps:0"
        owner = types.SimpleNamespace(device="mps", loaded=True)
        class Pager:
            _lock = threading.RLock()
            _owners = {packed: owner}
            def _observe_mutations(self, module, current):
                self_test.assertIs(current, owner); calls.append("verify")
            def evict(self, module): self._owners[module].device = None; calls.append("evict")
            @contextmanager
            def use(self, module, device): self._owners[module].device = device; calls.append(("readmit", device)); yield
            def _regional_limits(self): pass
        self_test = self
        brain = Owner(core_pager=Pager(), resource_policy=policy, device="mps")
        ops = NATIVE.NativeOperationAutoOps(brain); ops.quiescent = lambda: None; ops.admit_candidate = lambda _candidate: {}
        operation = {"closure": {"modules": {id(packed): packed}}, "residency": [(packed, "mps:0")], "ownerObjects": {packed: owner}}
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()): pass
        self.assertEqual(calls, ["verify"]); self.assertEqual(packed._native_compute_device, "mps:0")
        calls.clear()
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()): owner.device = None
        self.assertEqual(calls, ["verify", "evict", ("readmit", "mps:0")])
        self.assertEqual(packed._native_compute_device, "mps:0")
        with self.assertRaises(NATIVE.OperationAutoRestoreRequired):
            with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()):
                brain.core_pager._owners[packed] = types.SimpleNamespace(device="mps", loaded=True)
        self.assertTrue(brain._native_execution_restore_required)

    def test_actual_public_perception_hooks_reach_policy_once_and_bypass_nested_private_hubs(self):
        calls = []
        class Hub: pass
        leaf = Hub(); leaf.config = types.SimpleNamespace(vision_enabled=True)
        leaf.vision = lambda tensor: tensor * 2
        leaf.perception_embedding = types.MethodType(MODAL.ModalityHub.perception_embedding, leaf)
        root = Hub(); root._resize = lambda tensor, _modality, _config: tensor
        root.connected_decoder = types.SimpleNamespace(regions=[types.SimpleNamespace(modalities=leaf, config={})],
                                                        workspace_exchange=lambda tensor: tensor)
        class Controller:
            def __init__(self, name): self.name = name
            def consume(self, _hub, modality, tensor, normal):
                calls.append(self.name)
                token = NATIVE._BYPASS.set(True)
                try: return normal()
                finally: NATIVE._BYPASS.reset(token)
        parent, child = Controller("complete-connected-entry"), Controller("must-not-repeat-parent")
        NATIVE.bind_operation_auto(root, parent); NATIVE.bind_operation_auto(leaf, child)
        tensor = torch.tensor([[1.0, 2.0]])
        result = CONNECTED.ConnectedModalityHub.perception_embedding(root, "image", tensor)
        self.assertEqual(result.tolist(), [[2.0, 4.0]])
        self.assertEqual(calls, ["complete-connected-entry"])


if __name__ == "__main__": unittest.main()
