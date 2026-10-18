"""LOCAL tiny CPU tensor/API contract; deliberately not stdlib-only CI.

No brain/decoder/module construction, Load, training, GPU allocation or kernel
benchmark. AST-owned page/codec definitions and real bounded safetensor I/O;
neural activations stay under64 KiB. Resource policy is explicitly injected.
"""
from __future__ import annotations

import ast
import bisect
import contextvars
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import types
import unittest
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[2]


def standalone(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


INERT = standalone("retained_tensor_inert_helpers", ROOT / "engine/tests/test_retained_context_capacity.py")
P = INERT.definitions(ROOT / "engine/omni_core/working_attention_paging.py",
    {"WorkingAttentionCancelled", "_ActivityPage", "_OwnedSpillLease", "WorkingAttentionPager", "PagedTensorSequence", "_saved_activity_scratch"},
    dict(torch=torch, bisect=bisect, contextvars=contextvars, hashlib=hashlib, math=math, os=os, time=time, uuid=uuid,
        OrderedDict=OrderedDict, contextmanager=contextmanager, nullcontext=nullcontext, dataclass=dataclass, Path=Path,
        NeuralStateResourcePause=INERT.Pause, canonical_device=str, current_regional_target=lambda _: None,
        _ACTIVE_PAGER=contextvars.ContextVar("owned_real_CPU_pages", default=None)), "owned_real_CPU_pages")
PACKAGE = types.ModuleType("owned_retained_runtime"); PACKAGE.__path__ = []
sys.modules[PACKAGE.__name__] = PACKAGE
for source, function, registry in (("native_core_paging.py", "release_native_tensor_chunk", "_MAPPED_PAGERS"),
                                   ("router_state_paging.py", "release_router_tensor_chunk", "_MAPPED")):
    # Exact lazy I/O release functions, no mapped learned owners in this raw
    # activity fixture. Do not import the modules' brain/kernel dependencies.
    INERT.definitions(ROOT / "engine/omni_core" / source, {function},
        {"torch": torch, registry: weakref.WeakValueDictionary()}, PACKAGE.__name__ + "." + source[:-3])
IO = standalone(PACKAGE.__name__ + ".bounded_tensor_io", ROOT / "engine/omni_core/bounded_tensor_io.py")


class RealCPURetainedTensorCase(unittest.TestCase):
    def test_real_noncontiguous_tensor_tail_spill_checksum_fork_and_paged_safe_file_restore(self):
        with TemporaryDirectory(prefix="omni-real-CPU-page-") as temporary:
            root, policy = Path(temporary), INERT.Policy()
            def pager(name):
                value = P.WorkingAttentionPager(root / name, resident_budget_bytes=0, scratch_budget_bytes=1024 * 1024,
                    device_tile_budget_bytes=65536, resource_policy=policy, page_tokens=4)
                self.addCleanup(value.close); return value
            original = torch.arange(84, dtype=torch.float32).reshape(1, 2, 7, 6)[..., ::2]
            self.assertFalse(original.is_contiguous()); self.assertLess(original.numel() * original.element_size(), 65536)
            source = pager("source"); sequence = source.sequence(original[:, :, :3, :], axis=2)
            self.addCleanup(sequence.close)
            sequence.append(original[:, :, 3:, :])
            self.assertEqual(sequence.ends, [4, 7])
            self.assertTrue(torch.equal(sequence.read(1, 7, torch.device("cpu")), original[:, :, 1:7, :]))
            fork = sequence.fork(); self.addCleanup(fork.close)
            sequence.close()
            self.assertTrue(torch.equal(fork.read(0, 7, torch.device("cpu")), original))
            self.assertGreater(source.status()["spillBytes"], 0)

            # Same genuine bounded tensor file methods and exact paged branch;
            # unrelated neural cache variants are intentionally not imported.
            tree = ast.parse((ROOT / "engine/omni_core/execution_state_codec.py").read_text())
            writer = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExecutionStateWriter"))
            reader = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExecutionStateReader"))
            encode = next(node for node in writer.body if isinstance(node, ast.FunctionDef) and node.name == "encode")
            encode.body = [next(node for node in encode.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "isinstance(value, PagedTensorSequence)")]
            digest = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "file_digest")
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            namespace = dict(torch=torch, hashlib=hashlib, math=math, Path=Path, weakref=weakref, nullcontext=nullcontext,
                PagedTensorSequence=P.PagedTensorSequence, SavedActivityTensor=type("NoTrainingGraph", (), {}),
                BoundedTensorFile=IO.BoundedTensorFile, atomic_save_tensors_bounded=IO.atomic_save_tensors_bounded)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, digest, writer, reader], type_ignores=[])),
                "owned-retained-CPU-codec", "exec"), namespace)
            destination = root / "snapshot"; destination.mkdir()
            encoded_writer = namespace["ExecutionStateWriter"](destination, policy=None)
            encoded = encoded_writer.encode(fork)
            restored_pager = pager("restored")
            encoded_reader = namespace["ExecutionStateReader"](destination, encoded_writer.files,
                pager=restored_pager, device="cpu", policy=None)
            restored = encoded_reader.decode(encoded); self.addCleanup(restored.close)
            self.assertTrue(torch.equal(restored.read(0, 7, torch.device("cpu")), original))
            self.assertEqual(restored._page_tokens, 4)
            self.assertTrue(all(item["device"] == "cpu" for item in encoded_writer.files.values()))
            self.assertLess(max(item["bytes"] for item in encoded_writer.files.values()), 65536)
            # A corrupt cold page fails its real checksum without inventing
            # values or dropping the other page/reference owners.
            damaged = restored_pager._pages[restored.pages[0]]
            with damaged.path.open("r+b") as handle:
                handle.seek(0); value = handle.read(1); handle.seek(0); handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                restored.read(0, 1, torch.device("cpu"))
            self.assertEqual(restored_pager.status()["livePages"], 2)


if __name__ == "__main__": unittest.main()
