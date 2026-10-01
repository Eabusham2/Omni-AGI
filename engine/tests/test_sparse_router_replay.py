"""Tiny fast-state protocol/storage fixtures; no model or brain construction."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.distributed_seal import native_topology_sha256, make_distributed_training_seal
from omni_core.distributed_training import apply_authoritative_source_updates
from omni_core.offload import NeuralStateResourcePause
from omni_core.persistence import tensor_checksum
from omni_core.sparse_router_state import canonical_module_walk
if __package__:
    from .test_sparse_router_state import sparse_owner
else:
    from test_sparse_router_state import sparse_owner


def protocol_owner(router):
    return SimpleNamespace(router=router, decoder=torch.nn.Module(), memory_bridge=torch.nn.Module(),
        idea_adapter=torch.nn.Module(), liquid=torch.nn.Module(), modalities=torch.nn.Module(),
        memory=SimpleNamespace(neurons={}, assemblies=[], synapses={}))


def primitive_learned_checksum(owner):
    values = []
    for _path, child in canonical_module_walk(owner):
        getter = getattr(child, "authoritative_packed_tensors", None)
        if callable(getter):
            values.extend(getter())
    return tensor_checksum(values)


class SparseRouterReplayFixtures(unittest.TestCase):
    def test_canonical_ordered_replay_births_roundtrip_with_same_seal_and_learned_hash(self):
        # A protocol callback applies two explicit tiny STDP tensor equations,
        # not a constructed AdaptiveBrain, optimizer, network or training job.
        router = sparse_owner(rows=67, columns=69)
        owner = protocol_owner(router)
        calls = []
        def literal_fast_state(piece, **kwargs):
            calls.append((piece, kwargs))
            router.reset_activity()
            pre_index, post_index = (68, 66) if len(calls) == 1 else (0, 0)
            router.pre_trace[pre_index] = 1.
            post = torch.zeros(67); post[post_index] = 1.
            router.step(torch.zeros(69), post)
        owner.learn_experience = literal_fast_state
        records = [(0, SimpleNamespace(kind="text"), SimpleNamespace(text_payload=SimpleNamespace(
                       windows=lambda: iter((("unaltered source one", 20),))))),
                   (1, SimpleNamespace(kind="text"), SimpleNamespace(text_payload=SimpleNamespace(
                       windows=lambda: iter((("unaltered source two", 20),)))))]
        committed, reports = apply_authoritative_source_updates(owner, iter(records))
        self.assertEqual(committed, 2)
        self.assertEqual(reports, [])
        self.assertEqual([value[0] for value in calls], ["unaltered source one", "unaltered source two"])
        self.assertTrue(all(kwargs["steps"] == 0 and kwargs["structural_detail"] for _, kwargs in calls))
        self.assertEqual(list(router.blocks), ["r1_c1", "r0_c0"])
        before_seal = native_topology_sha256(owner)
        before_learned = primitive_learned_checksum(router)
        before_fast = router.checksum_with_controls()
        with tempfile.TemporaryDirectory(prefix="sparse-replay-fixture-") as directory:
            path = Path(directory) / "state.safetensors"
            atomic_save_tensors_bounded(path, {"router.synapses."+key: value for key, value in router.state_dict().items()}, chunk_bytes=64)
            restored = sparse_owner(rows=67, columns=69, loading=True)
            restored.load_recurrent_state_bounded(BoundedTensorFile(path, chunk_bytes=64), "router.synapses.")
        self.assertEqual(list(restored.blocks), ["r0_c0", "r1_c1"])
        self.assertEqual(native_topology_sha256(protocol_owner(restored)), before_seal)
        self.assertEqual(primitive_learned_checksum(restored), before_learned)
        self.assertEqual(restored.checksum_with_controls(), before_fast)
        cursors = [{"rank": 0, "worldSize": 1, "epoch": 1, "nextGlobalOrdinal": 0,
                    "ownedRecordsCompleted": 2, "optimizerStepsCompleted": 3, "manifestSha256": "a"*64}]
        seal = make_distributed_training_seal(manifest_sha256="a"*64, topology_sha256=before_seal,
            training_policy_sha256="b"*64, record_count=2, epochs=1, cursors=cursors,
            committed_record_stop=2, global_steps=3)
        self.assertEqual(seal["topologySha256"], native_topology_sha256(protocol_owner(restored)))

    def test_non_sparse_traversal_and_shared_module_dedup_are_unchanged(self):
        root = torch.nn.Module()
        shared = torch.nn.Module()
        shared.add_module("z", torch.nn.Module())
        shared.add_module("a", torch.nn.Module())
        root.add_module("late", shared)
        root.add_module("alias", shared)
        root.add_module("first", torch.nn.Module())
        self.assertEqual(list(canonical_module_walk(root)), list(root.named_modules()))

    def test_sparse_registry_sort_is_admitted_before_children_are_yielded(self):
        router = sparse_owner()
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[0, 0] = 1
        router.set_effective_weights(levels)
        calls = []
        class Policy:
            def reserve_ram(self, amount, operation):
                calls.append((amount, operation))
                raise NeuralStateResourcePause("fixture canonical registry metadata ceiling", {})
        walk = canonical_module_walk(router, resource_policy=Policy())
        self.assertEqual(next(walk)[0], "")
        self.assertEqual(next(walk)[0], "blocks")
        with self.assertRaises(NeuralStateResourcePause):
            next(walk)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0][0], 1024+128)

    def test_actual_collective_loads_full_canonical_native_and_strict_slow_module_excludes_router(self):
        source = Path(__file__).parents[1] / "omni_core" / "distributed_training.py"
        tree = ast.parse(source.read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DistributedBrainTrainingModule")
        constructor = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        registered = {target.attr for node in ast.walk(constructor) if isinstance(node, ast.Assign)
                      for target in node.targets if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"}
        self.assertNotIn("router", registered)
        self.assertEqual(registered, {"decoder", "memory_bridge", "idea_adapter", "liquid"})
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DistributedGroundUpTrainer")
        publication = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_native_collective")
        refresh = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_refresh_native_replica")
        body = ast.get_source_segment(source.read_text(), publication)
        self.assertLess(body.index("apply_authoritative_source_updates("), body.index("native_topology_sha256(brain)"))
        self.assertLess(body.index("native_topology_sha256(brain)"), body.index("brain.save()"))
        refreshed = ast.get_source_segment(source.read_text(), refresh)
        self.assertIn("AdaptiveBrain.load(path)", refreshed)
        self.assertNotIn("load_state_dict", refreshed)
        self.assertIn("self._verify_native_seal(restored, checkpoint)", refreshed)


if __name__ == "__main__":
    unittest.main()
