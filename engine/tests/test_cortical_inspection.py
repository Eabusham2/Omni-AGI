"""Committed file/tensor inspection only; no neural constructor or execution."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.cortical_inspection import query_committed_cortex
from omni_core.persistence import atomic_save_tensors, atomic_write_json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class CorticalInspectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="omni-cortex-inspection-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.name = "decoder.language_head._packed_forward_weight"
        # LSB codes 0,1,2,1 -> trits -1,0,+1,0. Seven columns with zero padding.
        tensors = {self.name: torch.tensor([[0x64, 0x59], [0x64, 0x59]], dtype=torch.uint8)}
        blobs = self.root / "state" / "blobs"
        blobs.mkdir(parents=True)
        roles = {}
        for role, values in (("core", tensors), ("plasticity", {})):
            path = blobs / (role + ".safetensors")
            atomic_save_tensors(path, values)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            destination = blobs / (digest + ".safetensors")
            path.rename(destination)
            roles[role] = {"path": "blobs/" + destination.name, "sha256": digest, "bytes": len(data)}
        body = {"format": "omni-mutable-state", "formatVersion": 1, "brainId": "b", "roles": roles}
        generation = hashlib.sha256(canonical(body)).hexdigest()
        manifest = {**body, "contentSha256": generation}
        manifest_path = self.root / "state" / "generations" / generation / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        pointer = {"activeGeneration": generation, "contentSha256": generation,
                   "generationManifest": "generations/" + generation + "/manifest.json",
                   "generationManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        atomic_write_json(self.root / "brain.json", {"brain_id": "b", "mutable_state": pointer,
                                                    "recent_token_context": [1, 259, 100, 260, 101, 2]})
        pack = {"tensors": [{"name": "decoder.language_head.weight", "shape": [2, 7]}]}
        packed_path = self.root / "packed-ternary" / "manifest.json"
        atomic_write_json(packed_path, pack)
        checksum = packed_path.with_name("manifest.sha256")
        checksum.write_text(hashlib.sha256(packed_path.read_bytes()).hexdigest())

    def test_inventory_and_exact_selected_columns_are_bounded(self):
        page = query_committed_cortex(self.root, "b", {})
        self.assertEqual(page["moduleCount"], 1)
        self.assertEqual(page["logicalParameters"], 14)
        self.assertTrue(page["logicalInventoryComplete"])
        first = query_committed_cortex(self.root, "b", {"entity": "elements", "moduleId": self.name, "row": 0, "pageSize": 4})
        self.assertEqual([item["value"] for item in first["records"]], [-1, 0, 1, 0])
        self.assertTrue(first["hasMore"])
        rest = query_committed_cortex(self.root, "b", {"entity": "elements", "moduleId": self.name, "row": 0, "pageSize": 4, "cursor": first["nextCursor"]})
        self.assertEqual(rest["offset"], 4)
        self.assertEqual(len(rest["records"]), 3)
        self.assertFalse(rest["activity"]["observed"])
        self.assertFalse(rest["integrity"]["wholeRolePayloadScanned"])

    def test_cursor_and_native_identity_reject_mismatch(self):
        page = query_committed_cortex(self.root, "b", {"entity": "rows", "moduleId": self.name, "pageSize": 1})
        with self.assertRaisesRegex(ValueError, "cursor changed"):
            query_committed_cortex(self.root, "b", {"entity": "rows", "moduleId": self.name, "row": 1, "cursor": page["nextCursor"]})
        with self.assertRaisesRegex(ValueError, "identity"):
            query_committed_cortex(self.root, "wrong", {})

    def test_token_boundaries_are_observed_input_not_firing_or_semantics(self):
        page = query_committed_cortex(self.root, "b", {"entity": "boundaries"})
        self.assertEqual(page["records"][1]["kind"], "human-boundary")
        self.assertEqual(page["records"][3]["kind"], "brain-boundary")
        self.assertTrue(all(item["observedInput"] for item in page["records"]))
        self.assertTrue(all(item["activation"] is None for item in page["records"]))
        links = query_committed_cortex(self.root, "b", {"entity": "links", "moduleId": self.name})
        self.assertTrue(all(item["semanticExplanationVerified"] is False for item in links["records"]))


if __name__ == "__main__":
    unittest.main()
