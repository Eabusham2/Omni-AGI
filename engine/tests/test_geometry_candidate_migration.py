"""Packed-copy/manifest fixtures only: no models, brains, forward or training."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from omni_core.architecture_migration import (
    architecture_mutation_policy, copy_control_geometry, copy_packed_geometry,
    geometry_candidate_config, geometry_candidate_manifest, geometry_owner_plan,
    normalize_architecture_change, reseal_native_descriptor,
    validate_compatible_architecture_lineage, validate_geometry_candidate_manifest,
)
from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.native_architecture import native_architecture_sha256, native_core_inventory
from omni_core.text_spool import bounded_json_sha256


def pack(values):
    rows, width = values.shape
    result = torch.full((rows, (width + 3) // 4), 0x55, dtype=torch.uint8)
    for row in range(rows):
        for column in range(width):
            lane = 2 * (column % 4)
            result[row, column // 4] = (int(result[row, column // 4]) & (0xFF ^ (3 << lane))) | ((int(values[row, column]) + 1) << lane)
    return result


def unpack(values, width):
    return torch.tensor([[(int(values[row, column // 4]) >> (2 * (column % 4)) & 3) - 1
        for column in range(width)] for row in range(values.shape[0])], dtype=torch.int8)


class GeometryCandidateMigrationFixtures(unittest.TestCase):
    def geometry(self):
        return {"d_model": 6, "d_ff": 10, "n_heads": 1, "idea_dim": 6}

    def test_explicit_width_head_protocol_is_candidate_only_and_checks_legal_geometry(self):
        parent = self.geometry()
        original = copy.deepcopy(parent)
        change = normalize_architecture_change({"mutation": "resize-width", "dModel": 8, "feedForward": 16, "nHeads": 2})
        target = geometry_candidate_config(parent, change)
        self.assertEqual((target["d_model"], target["d_ff"], target["n_heads"], target["idea_dim"]), (8, 16, 2, 8))
        self.assertEqual(parent, original)
        self.assertEqual(architecture_mutation_policy(change)["functionPreservingAtInsertion"], False)
        for payload in ({"mutation": "resize-width"}, {"mutation": "repartition-heads", "nHeads": True},
            {"mutation": "resize-width", "dModel": 9}, {"mutation": "repartition-heads", "nHeads": 2}):
            with self.assertRaises(ValueError): geometry_candidate_config(parent, payload)

    def test_segmented_qkv_rows_are_exact_not_flat_padded_and_initialization_is_chunk_invariant(self):
        old, new = self.geometry(), {"d_model": 8, "d_ff": 16, "n_heads": 2, "idea_dim": 8}
        logical = (torch.arange(18 * 6).reshape(18, 6) % 3 - 1).to(torch.int8)
        source = pack(logical)
        before = source.clone()
        plan = geometry_owner_plan("decoder.blocks.0.attention.qkv", (18, 6), (24, 8), old, new)
        results = []
        for chunk in (32, 96):
            target = torch.empty((24, 2), dtype=torch.uint8)
            proof = copy_packed_geometry(source, target, 6, 8, row_segments=plan["rowSegments"],
                column_segments=plan["columnSegments"], seed="same-candidate", tensor_name="qkv", chunk_bytes=chunk)
            values = unpack(target, 8)
            for group in range(3):
                self.assertTrue(torch.equal(values[group * 8:group * 8 + 6, :6], logical[group * 6:(group + 1) * 6]))
            self.assertEqual(proof["copiedLogicalTrits"], 108)
            self.assertFalse(proof["functionPreserved"])
            results.append(target)
        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertTrue(torch.equal(source, before))

    def test_gate_and_fixed_schema_axes_move_to_their_declared_segments(self):
        old, new = self.geometry(), {"d_model": 8, "d_ff": 16, "n_heads": 2, "idea_dim": 8}
        gate = geometry_owner_plan("decoder.blocks.0.feed_forward.up", (20, 6), (32, 8), old, new)
        self.assertEqual(gate["rowSegments"], [[0, 0, 10], [10, 16, 10]])
        condition = geometry_owner_plan("decoder.action_argument_head.condition", (6, 10), (8, 12), old, new)
        self.assertEqual(condition["columnSegments"], [[0, 0, 6], [6, 8, 4]])
        recurrence = geometry_owner_plan("decoder.action_argument_head.transition", (6, 12), (8, 16), old, new)
        self.assertEqual(recurrence["columnSegments"], [[0, 0, 6], [6, 8, 6]])
        bias = geometry_owner_plan("decoder.blocks.0.attention.qkv", (1, 18), (1, 24), old, new, role="bias")
        self.assertEqual(bias["columnSegments"], [[0, 0, 6], [6, 8, 6], [12, 16, 6]])
        with self.assertRaisesRegex(ValueError, "Q/K/V"):
            geometry_owner_plan("decoder.blocks.0.attention.qkv", (19, 6), (24, 8), old, new)

    def test_head_only_preserves_all_packed_bytes_without_promising_same_attention_function(self):
        old = self.geometry()
        new = geometry_candidate_config(old, {"mutation": "repartition-heads", "nHeads": 3})
        source = pack((torch.arange(18 * 6).reshape(18, 6) % 3 - 1).to(torch.int8))
        target = torch.empty_like(source)
        plan = geometry_owner_plan("decoder.blocks.0.attention.qkv", (18, 6), (18, 6), old, new)
        proof = copy_packed_geometry(source, target, 6, 6, seed=1, tensor_name="head-only",
            row_segments=plan["rowSegments"], column_segments=plan["columnSegments"], chunk_bytes=32)
        self.assertTrue(torch.equal(source, target))
        self.assertEqual(proof["initializedLogicalTrits"], 0)
        self.assertEqual(proof["unmappedParentLogicalTrits"], 0)
        self.assertFalse(proof["functionPreserved"])

    def test_immutable_file_source_and_narrowing_copy_only_declared_exact_coordinates(self):
        logical = (torch.arange(9 * 7).reshape(9, 7) % 3 - 1).to(torch.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.safetensors"
            atomic_save_tensors_bounded(path, {"packed": pack(logical)})
            original = path.read_bytes()
            target = torch.empty((5, 2), dtype=torch.uint8)
            proof = copy_packed_geometry(BoundedTensorFile(path, chunk_bytes=32), target, 7, 5,
                source_name="packed", seed=17, tensor_name="narrow", chunk_bytes=32)
            self.assertTrue(torch.equal(unpack(target, 5), logical[:5, :5]))
            self.assertEqual(proof["unmappedParentLogicalTrits"], 9 * 7 - 5 * 5)
            self.assertEqual(path.read_bytes(), original)
            # New row padding is neutral, never copied/random reserved codes.
            self.assertTrue(bool(((target[:, -1] >> 2) == 0x15).all()))

    def test_corruption_aliases_and_cancel_never_mutate_parent(self):
        source = pack(torch.zeros((3, 6), dtype=torch.int8))
        before = source.clone()
        with self.assertRaisesRegex(ValueError, "alias"):
            copy_packed_geometry(source, source.view_as(source), 6, 6, seed=1, tensor_name="alias")
        with self.assertRaisesRegex(ValueError, "alias"):
            copy_packed_geometry(source, torch.empty((4, 2), dtype=torch.uint8), 6, 8,
                row_segments=[[0, 0, 2], [1, 2, 2]], seed=1, tensor_name="bad")
        with self.assertRaises(InterruptedError):
            copy_packed_geometry(source, torch.empty((4, 2), dtype=torch.uint8), 6, 8,
                seed=1, tensor_name="cancel", cancelled=lambda: True)
        self.assertTrue(torch.equal(source, before))
        source[0, 0] = 0xFF
        with self.assertRaisesRegex(ValueError, "reserved"):
            copy_packed_geometry(source, torch.empty((4, 2), dtype=torch.uint8), 6, 8, seed=1, tensor_name="bad")

    def test_partial_candidate_cancellation_has_no_parent_mutation_or_completion_proof(self):
        source = pack(torch.zeros((3, 6), dtype=torch.int8))
        before = source.clone()
        target = torch.full((4, 2), 0xFF, dtype=torch.uint8)
        checks = 0
        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 9
        with self.assertRaises(InterruptedError):
            copy_packed_geometry(source, target, 6, 8, seed=3, tensor_name="partial",
                chunk_bytes=32, cancelled=cancelled)
        self.assertTrue(torch.equal(source, before))
        self.assertFalse(torch.equal(target, torch.full_like(target, 0xFF)))
        controls, target_controls = torch.arange(9, dtype=torch.int16), torch.empty(12, dtype=torch.int16)
        old_controls = controls.clone()
        checks = 0
        with self.assertRaises(InterruptedError):
            copy_control_geometry(controls, target_controls, chunk_bytes=4, cancelled=cancelled)
        self.assertTrue(torch.equal(controls, old_controls))

    def test_registered_page_releases_cover_only_completed_bounded_ranges(self):
        source = pack((torch.arange(5 * 7).reshape(5, 7) % 3 - 1).to(torch.int8))
        target = torch.empty((8, 3), dtype=torch.uint8)
        with patch("omni_core.native_core_paging.release_native_tensor_chunk") as release:
            copy_packed_geometry(source, target, 7, 9, seed=3, tensor_name="page-release", chunk_bytes=32)
        self.assertGreater(release.call_count, 0)
        for args in release.call_args_list:
            tensor, start, count = args.args
            self.assertGreater(count, 0)
            self.assertLessEqual(count, 32)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start + count, tensor.numel() * tensor.element_size())

    def test_row_resistance_and_control_axes_preserve_values_without_averaging(self):
        source = torch.arange(18, dtype=torch.uint8)
        target = torch.empty(24, dtype=torch.uint8)
        proof = copy_control_geometry(source, target, axis_segments=[[[0, 0, 6], [6, 8, 6], [12, 16, 6]]], chunk_bytes=4)
        for index in range(3):
            self.assertTrue(torch.equal(target[index * 8:index * 8 + 6], source[index * 6:(index + 1) * 6]))
            self.assertTrue(bool((target[index * 8 + 6:(index + 1) * 8] == 0).all()))
        self.assertEqual(proof["copiedElements"], 18)
        self.assertLessEqual(proof["peakTransferBytes"], 4)
        before = source.clone()
        with self.assertRaisesRegex(ValueError, "alias"):
            copy_control_geometry(source, source.view_as(source))
        self.assertTrue(torch.equal(source, before))

    def test_migration_manifest_binds_parent_origin_and_cannot_self_claim_activation(self):
        old, change = self.geometry(), {"mutation": "resize-width", "dModel": 8, "nHeads": 2}
        new = geometry_candidate_config(old, change)
        proof = copy_packed_geometry(pack(torch.zeros((2, 6), dtype=torch.int8)), torch.empty((3, 2), dtype=torch.uint8),
            6, 8, seed=1, tensor_name="fixture")
        manifest = geometry_candidate_manifest(candidate_id="isolated", parent_metadata_sha256="a" * 64,
            parent_architecture_sha256="b" * 64, root_architecture_sha256="c" * 64,
            candidate_architecture_sha256="d" * 64, mutation=change, old_geometry=old, new_geometry=new,
            tensor_proofs={"fixture": proof}, owner_inventory={"fixture": {"kind": "packed", "owner": "fixture",
                "role": "weight", "oldShape": [2, 2], "newShape": [3, 2], "dtype": "torch.uint8"}})
        validate_geometry_candidate_manifest(manifest, parent_metadata_sha256="a" * 64, root_architecture_sha256="c" * 64)
        bad = copy.deepcopy(manifest); bad["activationAllowed"] = True
        bad["contentSha256"] = bounded_json_sha256({key: value for key, value in bad.items() if key != "contentSha256"})
        with self.assertRaisesRegex(ValueError, "activate"):
            validate_geometry_candidate_manifest(bad)
        bad = copy.deepcopy(manifest); bad["tensorProofs"]["fixture"]["initializedLogicalTrits"] += 1
        bad["contentSha256"] = bounded_json_sha256({key: value for key, value in bad.items() if key != "contentSha256"})
        with self.assertRaisesRegex(ValueError, "account"):
            validate_geometry_candidate_manifest(bad)

    def test_mixed_compatible_and_geometry_lineage_preserves_immutable_root(self):
        shape = {"dModel": 48, "layers": 1, "feedForward": 96, "nHeads": 4, "vsaDimensions": 64,
            "routerNeurons": 5, "modalityChannels": 4, "imageSize": 16, "audioSamples": 256, "videoFrames": 4,
            "workingMemoryItems": 32, "workspaceLatents": 8, "vocabSize": 261, "liquidMode": "cfc"}
        descriptor = {"format": "omni-main-selected-native-architecture", "formatVersion": 1,
            "architecture": "OmniCortex", "externalPretrainedWeights": False,
            "qualityEvidence": "unmeasured-native-quality-deferred", "hardwareTier": "micro",
            "shape": shape, "inventory": native_core_inventory(shape), "sizing": {"selected": 1}}
        descriptor["sha256"] = native_architecture_sha256(descriptor)
        origin = copy.deepcopy(descriptor)
        config = SimpleNamespace(native_architecture=descriptor, n_layers=2, router_neurons=5)
        config.native_architecture = reseal_native_descriptor(config, {"mutation": "grow-depth", "addLayers": 1})
        config.d_model, config.d_ff, config.n_heads = 64, 128, 4
        config.native_architecture = reseal_native_descriptor(config, {"mutation": "resize-width", "dModel": 64, "feedForward": 128})
        config.n_heads = 8
        config.native_architecture = reseal_native_descriptor(config, {"mutation": "repartition-heads", "nHeads": 8})
        config.router_neurons = 7
        current = reseal_native_descriptor(config, {"mutation": "grow-router", "addNeurons": 2})
        validate_compatible_architecture_lineage(current, origin)
        self.assertEqual(origin, descriptor)
        self.assertTrue(current["evolutionLineage"]["normalizationAndHeadGeometryChanged"])
        self.assertFalse(current["evolutionLineage"]["qualityVerified"])
        self.assertEqual(current["evolutionLineage"]["rootArchitectureSha256"], origin["sha256"])


if __name__ == "__main__": unittest.main()
