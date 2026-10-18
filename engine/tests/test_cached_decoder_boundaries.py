"""NEW actual-source inert boundaries, not a brain/decoder construction."""
from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
import unittest
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("owned_cached_boundary_sampler", ROOT / "omni_core/native_token_sampling.py")
SAMPLER = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAMPLER)


def methods(relative, owner, names, namespace):
    source = ast.parse((ROOT / relative).read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == owner)
    selected = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    for node in selected: node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(ROOT / relative), "exec"), namespace)
    return namespace


class CachedNativeBoundaries(unittest.TestCase):
    def test_actual_worker_status_stop_are_flags_only_exact_owner_and_no_load(self):
        class Fault(RuntimeError): pass
        namespace = methods("worker.py", "Worker", {"dispatch_control", "_decoder_control_owner",
            "cached_decoder_execution_status", "cached_decoder_execution_stop"}, {"RpcFault": Fault, "MCP_SAMPLING_CONTROLS": set()})
        calls = []
        brain = types.SimpleNamespace(storage_path="owned", cached_decoder_execution_status=lambda: {"phase": "calibrating"},
            cached_decoder_execution_stop=lambda **kw: calls.append(kw) or {"phase": "calibrating"})
        worker = types.SimpleNamespace(worker_role="neural", brains={"owned": brain},
            _brain_id=lambda params: params["brainId"], _storage=lambda params, owner: params.get("storagePath", owner),
            _get=lambda *_args: self.fail("metadata control attempted Load/neural getter"))
        for name in ("_decoder_control_owner", "cached_decoder_execution_status", "cached_decoder_execution_stop"):
            setattr(worker, name, types.MethodType(namespace[name], worker))
        def request(method, params):
            return namespace["dispatch_control"](worker, {"jsonrpc": "2.0", "id": "control", "method": method, "params": params})
        self.assertEqual(request("cached_decoder_execution_status", {"brainId": "owned"})["result"]["phase"], "calibrating")
        request("cached_decoder_execution_stop", {"brainId": "owned", "operationId": "exact-arm"})
        self.assertEqual(calls, [{"operation_id": "exact-arm"}])
        for params in ({"brainId": "missing"}, {"brainId": "owned", "storagePath": "other"}, {"brainId": "owned", "prompt": "not-authority"}):
            with self.assertRaises(Fault): request("cached_decoder_execution_status", params)

    def test_actual_arm_loaded_owner_keeps_inline_job_and_never_reloads(self):
        class Fault(RuntimeError): pass
        namespace = methods("worker.py", "Worker", {"_decoder_loaded_owner", "cached_decoder_execution_prepare", "cached_decoder_execution_arm"},
            {"RpcFault": Fault, "nullcontext": nullcontext})
        calls = []
        brain = types.SimpleNamespace(brain_id="owned", cached_decoder_execution_prepare=lambda **kw: calls.append(("prepare", kw)),
            cached_decoder_execution_arm=lambda **kw: calls.append(("arm", kw)))
        worker = types.SimpleNamespace(_regional_loaded_owner=lambda _params, _extra: brain, _inline_generations={})
        worker._decoder_loaded_owner = types.MethodType(namespace["_decoder_loaded_owner"], worker)
        namespace["cached_decoder_execution_prepare"](worker, {"brainId": "owned", "topology": {"actual": True}}, "request")
        namespace["cached_decoder_execution_arm"](worker, {"brainId": "owned", "operationId": "exact", "reviewed": {"owner": True}}, "request")
        self.assertEqual(calls, [("prepare", {"topology": {"actual": True}, "acknowledge_cuda": False}),
            ("arm", {"operation_id": "exact", "reviewed": {"owner": True}, "samples": 3})])
        worker._inline_generations["existing"] = types.SimpleNamespace(brain_id="owned", future=types.SimpleNamespace(done=lambda: False))
        with self.assertRaises(Fault): namespace["cached_decoder_execution_prepare"](worker, {"brainId": "owned"}, "request")
        self.assertEqual(len(calls), 2)
        self.assertIn("existing", worker._inline_generations)

    def test_actual_sampling_loop_reaches_explicit_cached_frontier_and_world_once_per_token(self):
        class Cache:
            def __init__(self, length): self.length, self.closed = length, False
            def close(self): self.closed = True
        class Cancel(RuntimeError): pass
        calls, hooks, outputs = [], [], []
        logits = torch.tensor([[float("-inf")] * 4 + [0.]])
        def forward(window, cue, cache):
            calls.append("ordinary-forward")
            decoder.last_generation_neural_state = torch.tensor([[1.]])
            return logits, Cache(window.shape[1])
        def cached_hook(_decoder, **params):
            hooks.append((params["allowed"], None if params["cache"] is None else params["cache"].length))
            if not params["allowed"] or params["cache"] is None: return params["fallback"]()
            self.assertFalse(params["flags_cancelled"]())
            checked = Cache(params["window"].shape[1])
            self.assertTrue(params["before_sample"](checked, torch.tensor([[2.]])))
            return SAMPLER.sample_native_token(logits, params["generator"], **params["sampling"])
        namespace = methods("omni_core/model.py", "OmniDecoder", {"_generate_scoped_iter"},
            {"torch": torch, "Mapping": Mapping, "WorkingAttentionCancelled": Cancel,
             "sample_native_token": SAMPLER.sample_native_token, "cached_sampling_operation": cached_hook})
        decoder = types.SimpleNamespace(config=types.SimpleNamespace(max_seq_len=8), eval=lambda: None,
            _generation_step=forward, _working_pager=lambda: object())
        def boundary(): calls.append("world"); return False
        owner = [None]
        stream = namespace["_generate_scoped_iter"](decoder, torch.tensor([[1]]), max_new_tokens=2,
            temperature=.65, top_k=1, printable_only=False, cancelled=boundary,
            token_callback=lambda token, _step, _entropy: outputs.append(token.item()), _cache_owner=owner,
            allow_cached_execution=True, cached_cancelled=lambda: False)
        events = list(stream)
        self.assertEqual(len(events), 2); self.assertEqual(outputs, [4, 4])
        self.assertEqual(hooks, [(True, None), (True, 1)])
        self.assertEqual(calls, ["world", "ordinary-forward", "world", "world", "world"])
        self.assertEqual(owner[0].length, 2)
        calls.clear(); hooks.clear()
        list(namespace["_generate_scoped_iter"](decoder, torch.tensor([[1]]), max_new_tokens=2,
            top_k=1, printable_only=False, cancelled=boundary))
        self.assertEqual(hooks, [(False, None), (False, 1)])
        self.assertEqual(calls.count("ordinary-forward"), 2)

    def test_only_visible_single_pass_chat_opt_in_is_bound_to_raw_flags_not_world_tick(self):
        tree = ast.parse((ROOT / "omni_core/brain.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and any(
            keyword.arg == "allow_cached_execution" for keyword in node.keywords)]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(next(keyword.value.value for keyword in call.keywords if keyword.arg == "allow_cached_execution"))
        flags = next(keyword.value for keyword in call.keywords if keyword.arg == "cached_cancelled")
        self.assertIsInstance(flags, ast.Lambda)
        invoked = {node.func.id for node in ast.walk(flags) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertEqual(invoked, {"bool", "original_cancel_check", "steer_check", "suspension_check"})


if __name__ == "__main__": unittest.main()
