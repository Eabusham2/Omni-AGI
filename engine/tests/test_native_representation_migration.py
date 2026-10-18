"""Tiny native bytes/cursor fixtures; no AdaptiveBrain constructor or training."""
import ast
import copy
import hashlib
import json
import os
import tempfile
import unittest
from typing import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded, load_module_bounded
from omni_core.model import PackedAdaptiveBitLinear
from omni_core.persistence import tensor_checksum, atomic_write_json, read_json
from omni_core.sparse_router_state import canonical_module_walk
from omni_core.ingestion_schedule_v3 import (make_ingestion_schedule_v3, schedule_sha256,
    make_checkpoint_binding_v3, checkpoint_binding_sha256, validate_checkpoint_binding_v3)
from omni_core.joint_generation import VerifiedArtifactCache, recover_joint_generation
from omni_core.offload import DurableReplayBuffer, MutableStateStore
from omni_core.vsa import NeuralSubstrate
from .test_sparse_router_state import sparse_owner
from .test_router_state_paging import owner as dense_owner
from . import test_joint_generation as joint_fixtures


class NativeRepresentationMigrationFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native-migration-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def brain(self):
        value = object.__new__(AdaptiveBrain)
        value.decoder = torch.nn.Module()
        # Deliberately differs from safe-tensor lexical ordering.
        value.decoder.add_module("z_first", PackedAdaptiveBitLinear(17, 5, bias=True))
        value.decoder.add_module("a_last", PackedAdaptiveBitLinear(3, 2, bias=True))
        value.memory_bridge = PackedAdaptiveBitLinear(2, 3, bias=True)
        value.idea_adapter = torch.nn.Module()
        value.router = torch.nn.Module()
        value.router.input_projection = PackedAdaptiveBitLinear(2, 3, bias=True)
        value.router.synapses = sparse_owner(rows=3, columns=3)
        levels = torch.zeros((3, 3), dtype=torch.int8); levels[0, 1] = 1
        value.router.synapses.set_effective_weights(levels)
        value.router.output_projection = PackedAdaptiveBitLinear(3, 2, bias=True)
        value.liquid = torch.nn.Module()
        value.modalities = torch.nn.Module()
        value._reserve_core_diagnostic_state = lambda _size: None
        value.resource_policy = SimpleNamespace(require_disk=lambda *_args: None)
        value.engine_path = self.root
        return value

    def files(self, value, *, old=True, dense=False):
        core, plastic = {}, {}
        for name, module in value._ternary_export_roots().items():
            destination = plastic if name == "router" else core
            for key, tensor in module.state_dict().items():
                if old and (key.endswith(("_connection_state", "_bias_connection_state")) or ".parallel_connections." in "."+key):
                    continue
                destination[name+"."+key] = tensor.clone()
        if old:
            plastic["router.synapses._sparse_layout"][0] = 1
        if dense:
            original = dense_owner(rows=3, columns=3)
            original.set_effective_weights(value.router.synapses.weights.to(torch.int8))
            # Keep exact old control data, without constructing a model.
            for name in ("pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
                getattr(original, name).copy_(getattr(value.router.synapses, name))
            plastic = {key: tensor for key, tensor in plastic.items() if not key.startswith("router.synapses.")}
            plastic.update({"router.synapses."+key: tensor.clone() for key, tensor in original.state_dict().items()})
        for path, data in ((self.root/"core.safetensors", core), (self.root/"plasticity.safetensors", plastic)):
            atomic_save_tensors_bounded(path, data, chunk_bytes=24)
        return BoundedTensorFile(self.root/"core.safetensors", chunk_bytes=24), BoundedTensorFile(self.root/"plasticity.safetensors", chunk_bytes=24), core, plastic

    def original_checksum(self, core, plastic, *, dense=False):
        names = ["decoder.z_first._packed_forward_weight", "decoder.z_first._packed_forward_bias",
                 "decoder.a_last._packed_forward_weight", "decoder.a_last._packed_forward_bias",
                 "memory_bridge._packed_forward_weight", "memory_bridge._packed_forward_bias",
                 "router.input_projection._packed_forward_weight", "router.input_projection._packed_forward_bias",
                 "router.synapses._packed_weights" if dense else "router.synapses.blocks.r0_c0._packed_weights",
                 "router.output_projection._packed_forward_weight", "router.output_projection._packed_forward_bias"]
        return tensor_checksum((plastic if name.startswith("router.") else core)[name] for name in names)

    def proof(self, value, readers, *, dense=False):
        core, plastic, _, _ = readers
        router = {"legacyDenseMigration": dense, "unitStrengthMigration": True}
        with patch.object(core, "tensor", side_effect=AssertionError("full source tensor read")), patch.object(plastic, "tensor", side_effect=AssertionError("full source tensor read")):
            return value._verify_native_representation_migration(core, plastic, router)

    def test_original_getter_order_is_preserved_across_core_and_sparse_migration(self):
        value = self.brain(); readers = self.files(value)
        before = self.original_checksum(readers[2], readers[3])
        proof = self.proof(value, readers)
        self.assertEqual(proof["sourceParameterChecksum"], before)
        self.assertEqual(proof["destinationParameterChecksum"], value.parameter_checksum())
        self.assertNotEqual(proof["sourceParameterChecksum"], proof["destinationParameterChecksum"])
        self.assertTrue(proof["required"])
        self.assertTrue(any(name.endswith("parallel_connections._next_page") for name in proof["verifiedAdditiveFields"]))
        self.assertLessEqual(readers[0].peak_transfer_bytes, 24)
        self.assertLessEqual(readers[1].peak_transfer_bytes, 24)

    def test_dense_original_slot_is_reconstructed_before_sparse_children(self):
        value = self.brain(); readers = self.files(value, dense=True)
        before = self.original_checksum(readers[2], readers[3], dense=True)
        proof = self.proof(value, readers, dense=True)
        self.assertEqual(proof["sourceParameterChecksum"], before)
        self.assertTrue(proof["denseRouterMigrated"])

    def test_parent_getter_resolves_the_actual_registered_child_owner(self):
        value = self.brain()
        value.decoder.authoritative_packed_tensors = lambda: value.decoder.z_first.authoritative_packed_tensors()
        readers = self.files(value)
        proof = self.proof(value, readers)
        self.assertEqual(proof["sourceParameterChecksum"], self.original_checksum(readers[2], readers[3]))
        self.assertIn("decoder.z_first._connection_state", proof["verifiedAdditiveFields"])

    def test_missing_new_strength_is_only_permitted_at_unit_value(self):
        value = self.brain(); readers = self.files(value)
        value.decoder.z_first._connection_state[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "not unit strength"):
            self.proof(value, readers)

    def test_missing_new_pool_counter_must_be_empty(self):
        value = self.brain(); readers = self.files(value)
        value.decoder.z_first.parallel_connections._next_page[0] = 1
        with self.assertRaisesRegex(ValueError, "empty metadata"):
            self.proof(value, readers)

    def test_dense_migration_rejects_unproved_nonunit_new_connection_state(self):
        value = self.brain(); readers = self.files(value, dense=True)
        value.router.synapses.blocks["r0_c0"]._connection_state[0, 0] = 3
        with self.assertRaisesRegex(ValueError, "not unit strength"):
            self.proof(value, readers, dense=True)

    def test_changed_live_signs_are_rejected_even_when_shape_still_matches(self):
        value = self.brain(); readers = self.files(value)
        packed = value.decoder.z_first._packed_forward_weight
        packed[0, 0] = (int(packed[0, 0]) & ~3) | (((int(packed[0, 0]) & 3) + 1) % 3)
        with self.assertRaisesRegex(ValueError, "changed original learned bytes"):
            self.proof(value, readers)

    def test_changed_source_file_during_proof_is_rejected(self):
        value = self.brain(); readers = self.files(value)
        original = readers[0].chunks
        changed = False
        def chunks(name):
            nonlocal changed
            for piece in original(name):
                yield piece
                if not changed:
                    changed = True
                    stamp = readers[0].path.stat()
                    os.utime(readers[0].path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1_000_000))
        with patch.object(readers[0], "chunks", side_effect=chunks):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.proof(value, readers)

    def test_current_weighted_files_keep_strengths_and_pool_state_in_checksum(self):
        value = self.brain()
        value.decoder.z_first._connection_state[0, 0] = 0x23
        value.decoder.z_first.add_parallel_connection_(0, 1, sign=1, strength=5)
        readers = self.files(value, old=False)
        proof = value._verify_native_representation_migration(readers[0], readers[1], {"legacyDenseMigration": False, "unitStrengthMigration": False})
        self.assertFalse(proof["required"])
        self.assertEqual(proof["verifiedAdditiveFields"], [])
        self.assertEqual(proof["sourceParameterChecksum"], value.parameter_checksum())

    def test_current_pool_birth_order_and_checkpoint_load_have_the_same_checksum(self):
        value = self.brain()
        value.decoder.z_first = PackedAdaptiveBitLinear(2, 257, bias=True)
        value.decoder.z_first.add_parallel_connection_(256, 0, sign=1, strength=3)
        value.decoder.z_first.add_parallel_connection_(0, 1, sign=-1, strength=4)
        readers = self.files(value, old=False)
        original = value.parameter_checksum()
        loaded = self.brain()
        loaded.decoder.z_first = PackedAdaptiveBitLinear(2, 257, bias=True)
        for name, module in loaded._ternary_export_roots().items():
            if name != "router":
                load_module_bounded(module, readers[0], name+".")
        loaded.router.synapses = sparse_owner(rows=3, columns=3, loading=True)
        migration = loaded.router.synapses.load_recurrent_state_bounded(readers[1], "router.synapses.")
        load_module_bounded(loaded.router, readers[1], "router.", excluded_prefixes=("synapses.",))
        self.assertEqual(loaded.parameter_checksum(), original)
        self.assertFalse(loaded._verify_native_representation_migration(readers[0], readers[1], migration)["required"])

    def test_unchanged_verified_v2_load_does_not_repeat_the_original_source_scan(self):
        source = Path(__file__).parents[1]/"omni_core"/"brain.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        load = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_impl")
        start = next(index for index, node in enumerate(load.body) if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "current_loaded_parameter_checksum" for target in node.targets))
        end = next(index for index in range(start, len(load.body)) if isinstance(load.body[index], ast.For) and
            isinstance(load.body[index].target, ast.Name) and load.body[index].target.id == "active_checkpoint")
        function = ast.FunctionDef(name="check", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
            ("brain", "core_reader", "plastic_reader", "router_migration", "recovered_mutable")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[*copy.deepcopy(load.body[start:end]), ast.Return(value=ast.Name(id="representation_migration", ctx=ast.Load()))], decorator_list=[])
        namespace = {"cls": AdaptiveBrain, "Mapping": Mapping, "substrate_source_binding": {"fixture": "already verified"},
                     "substrate_migration_required": False}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
        value = self.brain()
        expected = value.parameter_checksum()
        with patch.object(value, "_verify_native_representation_migration", side_effect=AssertionError("unnecessary second source scan")):
            proof = namespace["check"](value, None, None, {"legacyDenseMigration": False, "unitStrengthMigration": False},
                {"generation": {"metadata": {"parameterChecksum": expected}}})
        self.assertFalse(proof["required"])
        self.assertEqual(proof["sourceParameterChecksum"], expected)

    def test_source_generation_anchor_rejects_a_stale_or_missing_hash(self):
        value = self.brain(); proof = self.proof(value, self.files(value))
        for recovered in (None, {"generation": {"metadata": {"parameterChecksum": "f"*64}}}):
            with self.assertRaisesRegex(ValueError, "verified saved generation checksum"):
                value._bind_native_migration_source_generation(proof, recovered)
        value._bind_native_migration_source_generation(proof, {"generation": {"contentSha256": "d"*64,
            "metadata": {"parameterChecksum": proof["sourceParameterChecksum"]}}})
        self.assertTrue(proof["sourceGenerationVerified"])

    def source_pointer(self):
        return {"format": "omni-substrate-shards", "formatVersion": 3,
                "activeGeneration": "c"*64, "contentSha256": "c"*64,
                "generationManifestSha256": "d"*64,
                "counts": {"neurons": 2, "assemblies": 1, "synapses": 3}}

    def test_substrate_load_keeps_the_original_pointer_linked_to_saved_neural_state(self):
        value = self.brain(); pointer = self.source_pointer()
        value.memory = SimpleNamespace(persistence_manifest=copy.deepcopy(pointer))
        recovered = {"generation": {"metadata": {"substrateContentSha256": "c"*64}}}
        proof = value._verify_loaded_substrate_binding(pointer, recovered)
        self.assertTrue(proof["mutableSourceLinked"])
        self.assertTrue(proof["sourcePointerUnchanged"])
        self.assertEqual(proof["sourceCounts"], pointer["counts"])

    def test_uncommitted_substrate_mutation_invalidates_a_pending_representation_migration(self):
        value, proof = self.staged(); pointer = self.source_pointer()
        value.memory = SimpleNamespace(persistence_manifest=copy.deepcopy(pointer), state_revision=8)
        proof["substrateSourceBinding"] = value._verify_loaded_substrate_binding(pointer,
            {"generation": {"metadata": {"substrateContentSha256": "c"*64}}})
        value.memory.state_revision = 9
        value.save = lambda: self.fail("must not publish after independent memory changed")
        with self.assertRaisesRegex(ValueError, "verified substrate changed"):
            value._commit_native_representation_migration(proof)

    def test_substrate_generation_cannot_change_before_original_cursor_validation(self):
        value = self.brain(); pointer = self.source_pointer()
        value.memory = SimpleNamespace(persistence_manifest={**pointer, "activeGeneration": "e"*64, "contentSha256": "e"*64})
        with self.assertRaisesRegex(ValueError, "before cursor verification"):
            value._verify_loaded_substrate_binding(pointer, None)

    def test_stale_or_missing_cross_store_neural_substrate_binding_is_rejected(self):
        value = self.brain(); pointer = self.source_pointer()
        value.memory = SimpleNamespace(persistence_manifest=copy.deepcopy(pointer))
        for metadata in ({}, {"substrateContentSha256": "e"*64}):
            with self.assertRaisesRegex(ValueError, "do not match"):
                value._verify_loaded_substrate_binding(pointer, {"generation": {"metadata": metadata}})

    def test_malformed_original_substrate_counts_are_not_treated_as_empty(self):
        value = self.brain(); pointer = self.source_pointer()
        pointer["counts"]["neurons"] = True
        value.memory = SimpleNamespace(persistence_manifest=copy.deepcopy(pointer))
        with self.assertRaisesRegex(ValueError, "binding is invalid"):
            value._verify_loaded_substrate_binding(pointer, None)

    def weighted_substrate_proof(self, source, target):
        return {"verified": True, "sourceGeneration": source["activeGeneration"],
                "targetGeneration": target["activeGeneration"], "preservedNeuronCount": source["counts"]["neurons"],
                "preservedSynapseCount": source["counts"]["synapses"], "unitDefaultsVerified": True,
                "sourceRecordsAndSignsPreserved": True}

    def test_staged_substrate_checks_exact_original_and_target_generation_evidence(self):
        value = self.brain(); source = self.source_pointer()
        target = {**source, "activeGeneration": "e"*64, "contentSha256": "e"*64}
        value.memory = SimpleNamespace(persistence_manifest=target)
        value._native_substrate_migration_source = source
        expected = self.weighted_substrate_proof(source, target)
        with patch.object(NeuralSubstrate, "verify_saved_weighted_generation", return_value=expected, create=True) as verify:
            self.assertEqual(value._verify_staged_substrate_migration(), expected)
            verify.assert_called_once_with(self.root/"substrate", target, source_pointer=source,
                memory_reserve=value._reserve_core_diagnostic_state, disk_reserve=value.resource_policy.require_disk)
        for change in ({"unitDefaultsVerified": False}, {"sourceRecordsAndSignsPreserved": False},
                       {"sourceGeneration": "f"*64}, {"targetGeneration": "f"*64},
                       {"preservedNeuronCount": 1}, {"preservedSynapseCount": True}):
            with patch.object(NeuralSubstrate, "verify_saved_weighted_generation", return_value={**expected, **change}, create=True):
                with self.assertRaisesRegex(ValueError, "complete original generation"):
                    value._verify_staged_substrate_migration()

    def test_memory_migration_guard_is_present_during_save_and_cleared_afterwards(self):
        value, proof = self.staged(); source = self.source_pointer()
        value.memory = SimpleNamespace(persistence_manifest=source, state_revision=8)
        proof["substrateMigrationRequired"] = True
        proof["substrateSourceBinding"] = value._verify_loaded_substrate_binding(source,
            {"generation": {"metadata": {"substrateContentSha256": "c"*64}}})
        def save():
            self.assertEqual(value._native_substrate_migration_source, source)
            self.assertIsNot(value._native_substrate_migration_source, source)
        value.save = save
        value._commit_native_representation_migration(proof)
        self.assertIsNone(value._native_substrate_migration_source)

    def test_missing_memory_source_evidence_and_nested_migration_do_not_rewrite_cursors(self):
        value, proof = self.staged()
        before = copy.deepcopy(value.ingestion_checkpoints)
        proof["substrateMigrationRequired"] = True
        value.save = lambda: self.fail("save must not run without a complete migration boundary")
        with self.assertRaisesRegex(ValueError, "original substrate binding"):
            value._commit_native_representation_migration(proof)
        self.assertEqual(value.ingestion_checkpoints, before)
        value._native_substrate_migration_source = {"already": "pending"}
        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            value._commit_native_representation_migration(proof)
        self.assertEqual(value.ingestion_checkpoints, before)

    def staged(self):
        value = self.brain(); proof = self.proof(value, self.files(value))
        value._bind_native_migration_source_generation(proof, {"generation": {"contentSha256": "d"*64,
            "metadata": {"parameterChecksum": proof["sourceParameterChecksum"]}}})
        value.ingestion_checkpoints = {"source": {"neuralStateChecksum": proof["sourceParameterChecksum"],
            "committedRecords": 37, "epoch": 2, "commitSequence": 19, "activeRecordWindow": {"nextByte": 1042}}}
        value.ingestion_joint_generation = {"generation": "old-joint"}
        value.mutable_state_manifest = {"generation": "old-neural"}
        atomic_write_json(self.root/"brain.json", {"ingestion_checkpoints": value.ingestion_checkpoints,
            "ingestion_joint_generation": value.ingestion_joint_generation, "mutable_state": value.mutable_state_manifest})
        return value, proof

    def test_rebind_changes_only_neural_binding_and_keeps_record_window_positions(self):
        value, proof = self.staged()
        original = copy.deepcopy(value.ingestion_checkpoints)
        saved = []
        value.save = lambda: saved.append(copy.deepcopy(value.ingestion_checkpoints))
        value._commit_native_representation_migration(proof)
        self.assertEqual(len(saved), 1)
        expected = copy.deepcopy(original)
        expected["source"]["neuralStateChecksum"] = proof["destinationParameterChecksum"]
        self.assertEqual(value.ingestion_checkpoints, expected)

    def test_stale_cursor_prevents_any_publication(self):
        value, proof = self.staged()
        value.ingestion_checkpoints["source"]["neuralStateChecksum"] = "b"*64
        value.save = lambda: self.fail("must not save an unverified original cursor")
        with self.assertRaisesRegex(ValueError, "stale source cursor"):
            value._commit_native_representation_migration(proof)

    def test_precommit_interruption_restores_cursors_and_joint_reference(self):
        value, proof = self.staged()
        original = copy.deepcopy(value.ingestion_checkpoints)
        def fail():
            value.ingestion_joint_generation = {"generation": "uncommitted-joint"}
            raise OSError("fixture before brain.json commit")
        value.save = fail
        with self.assertRaises(OSError):
            value._commit_native_representation_migration(proof)
        self.assertEqual(value.ingestion_checkpoints, original)
        self.assertEqual(value.ingestion_joint_generation, {"generation": "old-joint"})
        self.assertEqual(read_json(self.root/"brain.json")["ingestion_checkpoints"], original)

    def test_postcommit_failure_does_not_undo_durable_rebinding(self):
        value, proof = self.staged()
        def fail_after_commit():
            value.ingestion_joint_generation = {"generation": "new-joint"}
            value.mutable_state_manifest = {"generation": "new-neural"}
            atomic_write_json(self.root/"brain.json", {"ingestion_checkpoints": value.ingestion_checkpoints,
                "ingestion_joint_generation": value.ingestion_joint_generation, "mutable_state": value.mutable_state_manifest})
            raise OSError("fixture after brain.json commit")
        value.save = fail_after_commit
        with self.assertRaises(OSError):
            value._commit_native_representation_migration(proof)
        self.assertEqual(value.ingestion_checkpoints["source"]["neuralStateChecksum"], proof["destinationParameterChecksum"])
        self.assertEqual(value.ingestion_joint_generation, {"generation": "new-joint"})

    def test_production_load_verifies_original_cursor_and_reads_rng_before_migration_save(self):
        source = Path(__file__).parents[1]/"omni_core"/"brain.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        load = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_impl")
        text = ast.get_source_segment(source.read_text(), load)
        self.assertLess(text.index("_bind_native_migration_source_generation("), text.index("for active_checkpoint in brain.ingestion_checkpoints.values()"))
        self.assertLess(text.index("validate_checkpoint_binding_v3("), text.index("_commit_native_representation_migration("))
        self.assertLess(text.index('stored_cpu_rng = plastic.get("state.rng_cpu")'), text.index("_commit_native_representation_migration("))
        self.assertLess(text.index('stored_accelerator_rng = plastic.get("state.rng_accelerator")'), text.index("_commit_native_representation_migration("))
        self.assertNotIn("_legacy_router_parameter_checksum", text)

    def validate_loaded_cursor(self, brain, checksum, *, joint=None):
        # Execute the actual production validation loop against storage-only
        # protocol owners, avoiding construction or training of a whole brain.
        source = Path(__file__).parents[1]/"omni_core"/"brain.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        load = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_impl")
        loop = next(node for node in load.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "active_checkpoint")
        function = ast.FunctionDef(name="validate", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
            ("brain", "loaded_parameter_checksum", "joint")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[copy.deepcopy(loop)], decorator_list=[])
        namespace = {"cls": AdaptiveBrain, "core_recovery_records": [], "representation_migration": {"required": True},
                     "validate_checkpoint_binding_v3": validate_checkpoint_binding_v3}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
        namespace["validate"](brain, checksum, joint)

    def test_v2_original_binding_checks_replay_workspace_and_neural_generation_before_rebind(self):
        value, proof = self.staged()
        before = proof["sourceParameterChecksum"]
        value.ingestion_checkpoints["source"]["formatVersion"] = 2
        value.memory = SimpleNamespace(persistence_manifest={"contentSha256": "c"*64})
        value.mutable_state_manifest = {"activeGeneration": "a"*64, "contentSha256": "a"*64}
        replay = {"count": 4, "highWaterId": 7, "contentSha256": "d"*64}
        workspace = {"count": 3, "highWaterId": 5, "contentSha256": "e"*64}
        value.replay = SimpleNamespace(checkpoint=lambda: dict(replay))
        value.paged_working_memory = SimpleNamespace(checkpoint=lambda: dict(workspace))
        value._fresh_attention_paged_clear_pending = False
        real_checksum = value.parameter_checksum
        value.parameter_checksum = lambda: before
        value._bind_ingestion_checkpoint_generation(value.mutable_state_manifest)
        value.parameter_checksum = real_checksum
        self.validate_loaded_cursor(value, before)
        for observed in (replay, workspace):
            prior = observed["contentSha256"]
            observed["contentSha256"] = "f"*64
            with self.assertRaisesRegex(ValueError, "committed generation"):
                self.validate_loaded_cursor(value, before)
            observed["contentSha256"] = prior
        with self.assertRaisesRegex(ValueError, "committed neural state"):
            self.validate_loaded_cursor(value, "b"*64)

    def test_v3_joint_generation_rebind_preserves_cursor_coverage_and_schedule(self):
        self.assert_v3_joint_rebind(memory_only=False)

    def test_memory_only_migration_rebinds_substrate_with_unchanged_core_checksum(self):
        self.assert_v3_joint_rebind(memory_only=True)

    def assert_v3_joint_rebind(self, *, memory_only):
        fixture = joint_fixtures.JointGenerationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        value = object.__new__(AdaptiveBrain)
        value.engine_path = fixture.root
        before, after = "a"*64, ("a" if memory_only else "b")*64
        value.parameter_checksum = lambda: before
        value.memory = SimpleNamespace(persistence_manifest=fixture.substrate_pointer)
        value.replay = SimpleNamespace(checkpoint=lambda: {"count": 0, "highWaterId": 0, "contentSha256": "d"*64})
        value.paged_working_memory = SimpleNamespace(checkpoint=lambda: {"count": 0, "highWaterId": 0, "contentSha256": "e"*64})
        value._fresh_attention_paged_clear_pending = False
        value._joint_artifact_cache = VerifiedArtifactCache()
        value.mutable_state_manifest = fixture.neural_pointer
        value.ingestion_joint_generation = None
        schedule = make_ingestion_schedule_v3(source_manifest_sha256=fixture.source_hash,
            source_content_sha256=fixture.content_hash, parser_manifest_sha256=fixture.parser_hash,
            physical_batch_records=1, gradient_accumulation=2, training_sequence_tokens=128,
            checkpoint_records=1, assembly_page_records=128)
        cursor = {"formatVersion": 3, "neuralStateChecksum": before,
            "sourceManifestSha256": fixture.source_hash, "parserManifestSha256": fixture.parser_hash,
            "contentHash": fixture.content_hash, "learningSchedule": schedule,
            "learningScheduleSha256": schedule_sha256(schedule), "commitSequence": 1,
            "committedRecords": 2, "recordPrefixSha256": fixture.cursor["recordPrefixSha256"],
            "visitedRecords": 3, "processedRecords": 2, "rejectedRecords": 1,
            "processedBytes": 42, "expectedRecords": 4, "activeRecordWindowSha256": "f"*64}
        value.ingestion_checkpoints = {"source": cursor}
        value._bind_ingestion_checkpoint_generation(fixture.neural_pointer)
        joint = recover_joint_generation(fixture.root/"state"/"ingestion-joint", value.ingestion_joint_generation,
            neural_store_root=fixture.neural_store, substrate_store_root=fixture.substrate_store,
            source_manifest_sha256=fixture.source_hash, parser_manifest_sha256=fixture.parser_hash,
            source_content_sha256=fixture.content_hash)
        self.validate_loaded_cursor(value, before, joint=joint)
        original = copy.deepcopy(cursor)
        with self.assertRaises(ValueError):
            self.validate_loaded_cursor(value, "c"*64, joint=joint)
        changed_joint = SimpleNamespace(manifest={**joint.manifest, "cursor": {**joint.manifest["cursor"], "committedRecords": 3}})
        with self.assertRaisesRegex(ValueError, "diverges"):
            self.validate_loaded_cursor(value, before, joint=changed_joint)

        # Stage a new immutable neural generation with the same source cursor.
        generation = read_json(fixture.neural_store/fixture.neural_pointer["generationManifest"])
        generation.pop("contentSha256")
        generation["metadata"] = {"parameterChecksum": after}
        next_substrate_pointer = fixture.substrate_pointer
        if memory_only:
            substrate_generation = read_json(fixture.substrate_store/fixture.substrate_pointer["generationManifest"])
            substrate_generation.pop("contentSha256")
            substrate_generation["nativeWeightedRepresentation"] = 2
            substrate_digest = hashlib.sha256(json.dumps(substrate_generation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            substrate_relative = "generations/"+substrate_digest+"/manifest.json"
            atomic_write_json(fixture.substrate_store/substrate_relative, {**substrate_generation, "contentSha256": substrate_digest})
            next_substrate_pointer = {**fixture.substrate_pointer, "activeGeneration": substrate_digest, "contentSha256": substrate_digest,
                "generationManifest": substrate_relative,
                "generationManifestSha256": hashlib.sha256((fixture.substrate_store/substrate_relative).read_bytes()).hexdigest()}
            generation["metadata"]["substrateContentSha256"] = substrate_digest
        payload = json.dumps(generation, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        relative = "generations/"+digest+"/manifest.json"
        atomic_write_json(fixture.neural_store/relative, {**generation, "contentSha256": digest})
        next_pointer = {**fixture.neural_pointer, "activeGeneration": digest, "contentSha256": digest,
            "generationManifest": relative,
            "generationManifestSha256": hashlib.sha256((fixture.neural_store/relative).read_bytes()).hexdigest()}
        value.parameter_checksum = lambda: after
        def save():
            value.mutable_state_manifest = next_pointer
            value.memory.persistence_manifest = next_substrate_pointer
            value._bind_ingestion_checkpoint_generation(next_pointer)
            atomic_write_json(value.engine_path/"brain.json", {"ingestion_checkpoints": value.ingestion_checkpoints,
                "ingestion_joint_generation": value.ingestion_joint_generation, "mutable_state": next_pointer})
        value.save = save
        proof = {"format": "omni-native-representation-migration-v2", "required": True,
            "originalLearnedBytesPreserved": True, "sourceGenerationVerified": True,
            "sourceParameterChecksum": before, "destinationParameterChecksum": after}
        value._commit_native_representation_migration(proof)
        for name in original:
            if name not in {"neuralStateChecksum", "pagedCheckpointBinding", "pagedCheckpointBindingSha256"}:
                self.assertEqual(cursor[name], original[name], name)
        reloaded = recover_joint_generation(fixture.root/"state"/"ingestion-joint", value.ingestion_joint_generation,
            neural_store_root=fixture.neural_store, substrate_store_root=fixture.substrate_store,
            source_manifest_sha256=fixture.source_hash, parser_manifest_sha256=fixture.parser_hash,
            source_content_sha256=fixture.content_hash)
        self.validate_loaded_cursor(value, after, joint=reloaded)
        self.assertEqual(reloaded.manifest["neuralState"]["generationId"], digest)
        self.assertEqual(reloaded.manifest["substrateState"]["generationId"], next_substrate_pointer["activeGeneration"])
        self.assertEqual(reloaded.manifest["cursor"], joint.manifest["cursor"])
        self.assertEqual(reloaded.manifest["coverage"], joint.manifest["coverage"])

    def test_interrupted_file_materialization_recovers_from_the_actual_committed_generation(self):
        base = self.root
        for after_commit in (False, True):
            self.root = base/("after" if after_commit else "before")
            self.root.mkdir()
            value = self.brain(); readers = self.files(value)
            proof = self.proof(value, readers)
            policy = SimpleNamespace(require_disk=lambda *_args: None)
            replay = DurableReplayBuffer(self.root/"state"/"replay.sqlite3", policy)
            replay.append(torch.tensor([1., 2.]))
            store = MutableStateStore(self.root/"state", "fixture", policy)
            old_pointer = store.stage_generation(core=readers[2], plasticity=readers[3], optimizer_state={}, replay=replay,
                metadata={"parameterChecksum": proof["sourceParameterChecksum"]})
            store.publish(old_pointer); recovered = store.recover(old_pointer, self.root, replay)
            value._bind_native_migration_source_generation(proof, recovered)
            value.memory = SimpleNamespace(persistence_manifest={"contentSha256": "c"*64})
            value.mutable_state_manifest = old_pointer
            value.ingestion_joint_generation = None
            value.ingestion_checkpoints = {"source": {"formatVersion": 2, "neuralStateChecksum": proof["sourceParameterChecksum"], "committedRecords": 37}}
            origin = self.root/"origin"; origin.mkdir()
            atomic_write_json(origin/"proof.json", {"unchanged": "original identity"})
            origin_before = (origin/"proof.json").read_bytes()
            atomic_write_json(self.root/"brain.json", {"ingestion_checkpoints": value.ingestion_checkpoints,
                "ingestion_joint_generation": None, "mutable_state": old_pointer})
            next_core, next_plastic = {}, {}
            for name, module in value._ternary_export_roots().items():
                destination = next_plastic if name == "router" else next_core
                destination.update({name+"."+field: tensor for field, tensor in module.state_dict().items()})
            staged = []
            def save_impl(*, commit_callback, **_kwargs):
                pointer = store.stage_generation(core=next_core, plasticity=next_plastic, optimizer_state={}, replay=replay,
                    metadata={"parameterChecksum": proof["destinationParameterChecksum"]})
                staged.append(pointer)
                store.materialize(pointer, self.root)
                value.mutable_state_manifest = pointer
                if after_commit:
                    atomic_write_json(self.root/"brain.json", {"ingestion_checkpoints": value.ingestion_checkpoints,
                        "ingestion_joint_generation": None, "mutable_state": pointer})
                    commit_callback()
                raise OSError("fixture interrupted migration")
            value._save_checkpoint_impl = save_impl
            with self.assertRaises(OSError):
                value._commit_native_representation_migration(proof)
            authoritative = read_json(self.root/"brain.json")
            restored = store.recover(authoritative["mutable_state"], self.root, replay)
            expected = proof["destinationParameterChecksum" if after_commit else "sourceParameterChecksum"]
            self.assertEqual(restored["generation"]["metadata"]["parameterChecksum"], expected)
            self.assertEqual(authoritative["ingestion_checkpoints"]["source"]["neuralStateChecksum"], expected)
            self.assertEqual(authoritative["ingestion_checkpoints"]["source"]["committedRecords"], 37)
            self.assertEqual(len(replay), 1)
            self.assertEqual((origin/"proof.json").read_bytes(), origin_before)
            names = BoundedTensorFile(self.root/"core.safetensors").specs
            self.assertEqual("decoder.z_first._connection_state" in names, after_commit)


if __name__ == "__main__":
    unittest.main()
