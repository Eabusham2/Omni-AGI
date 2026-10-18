"""NEW small CPU activity/sampling checks; never constructs a neural model.

Actual pager/cache classes are AST-loaded without omni_core's brain imports.
Only tiny CPU tensors, owned temporary raw pages and injected resource policy.
"""
from __future__ import annotations

import ast
import bisect
import contextvars
import copy
import gc
import hashlib
import importlib.util
import math
import mmap
import os
import stat
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

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def standalone(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Pause(RuntimeError):
    def __init__(self, message, status): super().__init__(message); self.status = status


def selected_definitions(name, relative, definitions, namespace):
    module = types.ModuleType(name); module.__dict__.update(namespace)
    sys.modules[name] = module
    source = ast.parse((ROOT / relative).read_text())
    selected = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in definitions]
    assert {node.name for node in selected} == set(definitions)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(ROOT / relative), "exec"), module.__dict__)
    return module


REGIONAL = standalone("owned_cached_regional", "omni_core/regional_execution.py")
PAGER = selected_definitions("owned_cached_pager", "omni_core/working_attention_paging.py", {
    "WorkingAttentionCancelled", "_ActivityPage", "_OwnedSpillLease", "WorkingAttentionPager",
    "PagedTensorSequence", "_saved_activity_scratch", "_SavedActivityImage", "_SavedActivityMapping", "SavedActivityTensor",
}, {"torch": torch, "bisect": bisect, "contextvars": contextvars, "hashlib": hashlib,
    "math": math, "mmap": mmap, "os": os, "time": time, "uuid": uuid,
    "OrderedDict": OrderedDict, "contextmanager": contextmanager, "nullcontext": nullcontext,
    "dataclass": dataclass, "Path": Path, "NeuralStateResourcePause": Pause,
    "canonical_device": REGIONAL.canonical_device, "current_regional_target": REGIONAL.current_regional_target,
    "_ACTIVE_PAGER": contextvars.ContextVar("owned_cached_pager", default=None)})
CACHE = selected_definitions("owned_cached_containers", "omni_core/model.py", {
    "_AttentionInferenceCache", "_WorkspaceInferenceChunk", "_DecoderInferenceCache",
}, {"torch": torch, "dataclass": dataclass, "PagedTensorSequence": PAGER.PagedTensorSequence,
    "SavedActivityTensor": PAGER.SavedActivityTensor})
SNAPSHOT = standalone("owned_cached_snapshot", "omni_core/cached_activity_snapshot.py")
SAMPLING = standalone("owned_cached_sampling", "omni_core/native_token_sampling.py")
NATIVE = selected_definitions("owned_cached_native_source", "omni_core/cached_decoder_native.py", {
    "NativeCachedDecoderOps",
}, {"NativeDecoderExecutionOps": object, "torch": torch, "os": os, "stat": stat,
    "hashlib": hashlib, "_identity_proof_supported": lambda _path, _details: True})


class Lease:
    def __init__(self, policy, size):
        self.policy, self.size, self.released, self.allocated = policy, size, False, 0
    def mark_allocated(self, size): self.allocated = size
    def release(self): self.released = True
    def __enter__(self): return self
    def __exit__(self, *_args): self.release()


class Policy:
    def __init__(self): self.leases, self.reject = [], None
    def reserve_ram(self, size, label):
        if size > 1024 * 1024: raise AssertionError("tiny CPU fixture requested over 1 MiB")
        if self.reject and self.reject in label: raise Pause("owned RAM pressure", {})
        lease = Lease(self, size); self.leases.append(lease); return lease
    def status(self, **_kwargs): return {}


def make_pager(root, policy=None, *, hot=0, page_tokens=4):
    return PAGER.WorkingAttentionPager(Path(root), resident_budget_bytes=hot,
        scratch_budget_bytes=1024 * 1024, device_tile_budget_bytes=64 * 1024,
        resource_policy=policy or Policy(), page_tokens=page_tokens)


def legacy_sample(logits, generator, *, temperature, top_k, noise, text_boundaries, remaining_tokens):
    # The pre-extraction inline law, including operation order and draw count.
    if noise > 0:
        jitter = torch.randn(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
        logits = logits + float(noise) * jitter
    logits = logits / max(float(temperature), 1e-4)
    if text_boundaries is not None:
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        for row, boundary in enumerate(text_boundaries):
            allowed[row, boundary.allowed_token_ids(remaining_tokens)] = True
        logits = logits.masked_fill(~allowed, -float("inf"))
    if top_k > 0 and top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, top_k, dim=-1)
        probabilities = F.softmax(values.float(), dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        token = indices.gather(-1, sampled)
        selected = probabilities.gather(-1, sampled)
    else:
        probabilities = F.softmax(logits.float(), dim=-1)
        token = torch.multinomial(probabilities, 1, generator=generator)
        selected = probabilities.gather(-1, token)
    log_probability = float(selected.clamp_min(1e-45).log().mean().item())
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    return token, log_probability, float(entropy.mean().item())


class CachedActivityOwnership(unittest.TestCase):
    def test_cold_cow_tail_keeps_exact_original_pages_until_last_owner(self):
        with TemporaryDirectory(prefix="omni-cached-cow-") as root:
            pager = make_pager(root)
            values = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
            original = pager.sequence(values)
            source_ids = list(original.pages)
            source_hashes = {key: hashlib.sha256(pager._pages[key].path.read_bytes()).hexdigest() for key in source_ids}
            fork = original.fork()
            self.assertTrue(all(pager._pages[key].references == 2 for key in source_ids))
            fork.append(torch.tensor([[[[10., 11.]]]]))
            self.assertEqual((original.length, fork.length), (3, 4))
            self.assertTrue(torch.equal(original.read(0, 3, torch.device("cpu")), values))
            self.assertTrue(torch.equal(fork.read(0, 4, torch.device("cpu")), torch.cat((values, torch.tensor([[[[10., 11.]]]])), dim=2)))
            self.assertEqual(source_hashes, {key: hashlib.sha256(pager._pages[key].path.read_bytes()).hexdigest() for key in source_ids})
            original.close()
            self.assertFalse(any(key in pager._pages for key in source_ids))
            self.assertEqual(fork.read(0, 4, torch.device("cpu")).shape, (1, 1, 4, 2))
            fork.close()
            self.assertEqual((pager.status()["livePages"], pager.status()["spillBytes"]), (0, 0))
            self.assertTrue(all(lease.released for lease in pager.resource_policy.leases))
            pager.close()

    def test_saved_workspace_first_restore_and_shared_image_stay_immutable(self):
        with TemporaryDirectory(prefix="omni-cached-workspace-") as root:
            pager = make_pager(root)
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            original = pager.save_activation(values)
            restored = original.restore()
            self.assertTrue(torch.equal(restored, values))
            image_path = original._restore_image.path
            before = hashlib.sha256(image_path.read_bytes()).hexdigest()
            fork = original.fork()
            restored.zero_()  # ACCESS_COPY storage must not mutate the image.
            other = fork.restore()
            self.assertTrue(torch.equal(other, values))
            self.assertEqual(hashlib.sha256(image_path.read_bytes()).hexdigest(), before)
            original.close(); fork.close()
            self.assertTrue(image_path.exists())  # Both exported tensors own it.
            del restored, other
            gc.collect()
            self.assertFalse(image_path.exists())
            self.assertEqual(pager.status()["spillBytes"], 0)
            pager.close()

    def test_partial_fork_pressure_and_cancel_release_refs_not_source(self):
        with TemporaryDirectory(prefix="omni-cached-pressure-") as root:
            policy = Policy(); pager = make_pager(root, policy, page_tokens=2)
            values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
            original = pager.sequence(values)
            policy.reject = "KV sequence metadata"
            with self.assertRaises(Pause): original.fork()
            policy.reject = None
            checks = [0]
            def stopped(): checks[0] += 1; return checks[0] >= 2
            pager._cancelled = stopped
            with self.assertRaises(PAGER.WorkingAttentionCancelled): original.fork()
            pager._cancelled = None
            self.assertTrue(all(pager._pages[key].references == 1 for key in original.pages))
            self.assertTrue(torch.equal(original.read(0, 4, torch.device("cpu")), values))
            original.close(); pager.close()
            self.assertTrue(all(lease.released for lease in policy.leases))

    def test_failed_last_unlink_retains_physical_quota_until_successful_retry(self):
        with TemporaryDirectory(prefix="omni-cached-unlink-") as root:
            pager = make_pager(root)
            original = pager.sequence(torch.ones((1, 1, 1, 2)))
            fork = original.fork(); original.close()
            key = fork.pages[0]; path = pager._pages[key].path
            allocated = pager.status()["spillBytes"]
            with patch.object(Path, "unlink", side_effect=OSError("owned unlink failure")):
                with self.assertRaisesRegex(OSError, "owned unlink failure"): fork.close()
            self.assertTrue(path.exists()); self.assertIn(key, pager._pages)
            self.assertEqual(pager.status()["spillBytes"], allocated)
            self.assertFalse(fork._closed)
            fork.close(); self.assertFalse(path.exists())
            self.assertEqual(pager.status()["spillBytes"], 0); pager.close()

    def test_decoder_cow_admits_partial_copies_and_transfers_cache_ownership_once(self):
        with TemporaryDirectory(prefix="omni-cached-structural-") as root:
            policy = Policy(); pager = make_pager(root, policy)
            saved = pager.save_activation(torch.ones(4))
            key = pager.sequence(torch.ones((1, 1, 2, 2)))
            value = pager.sequence(torch.zeros((1, 1, 2, 2)))
            original = CACHE._DecoderInferenceCache(2, ("same",), [CACHE._WorkspaceInferenceChunk(saved, saved.fork(), saved.fork())],
                [CACHE._AttentionInferenceCache(key, value)], torch.tensor([[2., 3.]]), None, torch.tensor([[4., 5.]]))
            before = {identifier: page.references for identifier, page in pager._pages.items()}
            copies = [0]
            def tensor(source):
                copies[0] += 1
                if copies[0] == 2: raise Pause("owned second copy pressure", {})
                lease = policy.reserve_ram(source.numel() * source.element_size(), "raw stats")
                result = source.clone(); lease.mark_allocated(result.numel() * result.element_size())
                return result, lease
            kwargs = dict(activity=lambda source: source.fork() if isinstance(source, (PAGER.PagedTensorSequence, PAGER.SavedActivityTensor)) else None,
                tensor=tensor, workspace=CACHE._WorkspaceInferenceChunk, attention=CACHE._AttentionInferenceCache,
                decoder=CACHE._DecoderInferenceCache, reserve_metadata=lambda size: policy.reserve_ram(size, "containers"))
            with self.assertRaises(Pause): SNAPSHOT.fork_decoder_cache(original, **kwargs)
            self.assertEqual(before, {identifier: page.references for identifier, page in pager._pages.items()})
            copies[0] = -100
            fork = SNAPSHOT.fork_decoder_cache(original, **kwargs)
            self.assertIsNot(fork.cache.workspace, original.workspace)
            fork.cache.hidden_sum.zero_()
            self.assertTrue(torch.equal(original.hidden_sum, torch.tensor([[2., 3.]])))
            adopted = fork.take(); fork.close()
            self.assertIsNone(fork.cache)
            original.close()
            self.assertEqual(adopted.attention[0].key.read(0, 2, torch.device("cpu")).shape, (1, 1, 2, 2))
            adopted.close(); pager.close()
            self.assertTrue(all(lease.released for lease in policy.leases))

    def test_sampling_is_exact_legacy_law_and_does_not_accept_or_publish_boundaries(self):
        class Boundary:
            def __init__(self): self.pending = [19]
            def allowed_token_ids(self, remaining): return [0, 2, 4] if remaining > 0 else [0]
            def accept(self, *_args): self.fail("sampling helper must not accept/publish tokens")
        logits = torch.tensor([[.2, -.3, 1.5, .9, -.8], [1., 2., .3, -1., 0.]])
        unchanged = logits.clone(); global_rng = torch.get_rng_state().clone()
        for top_k, noise, temperature, masked in ((0, 0., .65, False), (3, .2, .95, False), (2, .1, 0., True), (99, 0., .8, True)):
            with self.subTest(top_k=top_k, noise=noise, temperature=temperature, masked=masked):
                current, old = torch.Generator().manual_seed(1234), torch.Generator().manual_seed(1234)
                boundaries = [Boundary(), Boundary()] if masked else None
                options = dict(top_k=top_k, noise=noise, temperature=temperature, text_boundaries=boundaries, remaining_tokens=2)
                actual, expected = SAMPLING.sample_native_token(logits, current, **options), legacy_sample(logits, old, **options)
                self.assertTrue(torch.equal(actual[0], expected[0])); self.assertEqual(actual[1:], expected[1:])
                self.assertTrue(torch.equal(current.get_state(), old.get_state()))
                if boundaries: self.assertEqual([boundary.pending for boundary in boundaries], [[19], [19]])
        self.assertTrue(torch.equal(logits, unchanged)); self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))

    def test_exact_opened_page_proof_reuses_only_matching_identity_and_rejects_tamper_symlink_cancel(self):
        with TemporaryDirectory(prefix="omni-cached-proof-") as root:
            pager = make_pager(root); ops = NATIVE.NativeCachedDecoderOps()
            ops.brain = types.SimpleNamespace(resource_policy=pager.resource_policy)
            original = pager.sequence(torch.tensor([[[[1., 2.]]]]))
            identifier = original.pages[0]; page = pager._pages[identifier]
            expected = page.digest
            self.assertEqual(ops._page_sha(pager, identifier), expected)
            first_read = pager._metrics["bytesRead"]
            self.assertEqual(ops._page_sha(pager, identifier), expected)
            if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
                self.assertEqual(pager._metrics["bytesRead"], first_read)
            # Unknown/network/Windows eligibility cannot promote inode/times
            # alone into a reusable crypto proof, even after a matching hit.
            with patch.object(NATIVE, "_identity_proof_supported", return_value=False):
                before_unknown = pager._metrics["bytesRead"]
                self.assertEqual(ops._page_sha(pager, identifier), expected)
                self.assertEqual(ops._page_sha(pager, identifier), expected)
                self.assertEqual(pager._metrics["bytesRead"] - before_unknown, 2 * page.size)
            replacement = page.path.with_suffix(".replacement")
            replacement.write_bytes(page.path.read_bytes()); os.replace(replacement, page.path)
            self.assertEqual(ops._page_sha(pager, identifier), expected)
            self.assertGreater(pager._metrics["bytesRead"], first_read)
            changed = bytearray(page.path.read_bytes()); changed[-1] ^= 1; page.path.write_bytes(changed)
            with self.assertRaisesRegex(RuntimeError, "integrity failed"): ops._page_sha(pager, identifier)
            # Even a cached prior proof cannot follow a subsequently substituted link.
            target = page.path.with_suffix(".owned-original"); os.replace(page.path, target); page.path.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "regular file"): ops._page_sha(pager, identifier)
            page.path.unlink(); os.replace(target, page.path)
            pager._cancelled = lambda: True
            with self.assertRaises(PAGER.WorkingAttentionCancelled): ops._page_sha(pager, identifier)
            pager._cancelled = None
            original.close(); pager.close()

    def test_transient_route_preserves_default_dispatch_and_requires_exact_budget_owner(self):
        policy = object(); calls = []
        class Consumer:
            def packed_limits(self, current):
                if current is not policy: raise RuntimeError("wrong physical resource owner")
                return {"mps": {"memoryGroupId": "ram:system", "hotBytes": 64}}
            def execute(self, module, method, value, args, kwargs):
                calls.append(("transient", module)); return method(module, value, *args, **kwargs)
        class Block:
            @REGIONAL.regional_block_call
            def forward(self, value): calls.append(("actual", self)); return value + 1
        block = Block()
        self.assertEqual(block.forward(1), 2)
        self.assertIsNone(REGIONAL.transient_regional_limits(policy))
        with REGIONAL.transient_regional_scope(Consumer()):
            self.assertEqual(block.forward(2), 3)
            self.assertEqual(REGIONAL.transient_regional_limits(policy)["mps"]["hotBytes"], 64)
            with self.assertRaisesRegex(RuntimeError, "wrong physical"): REGIONAL.transient_regional_limits(object())
        self.assertIsNone(REGIONAL.transient_regional_limits(policy))
        self.assertEqual(block.forward(3), 4)
        self.assertEqual([call[0] for call in calls], ["actual", "transient", "actual", "actual"])


if __name__ == "__main__": unittest.main()
