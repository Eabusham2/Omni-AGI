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
    def replace_core(self, tensors):
        brain = json.loads((self.root / 'brain.json').read_bytes())
        generation = json.loads((self.root / 'state' / brain['mutable_state']['generationManifest']).read_bytes())
        scratch = self.root / 'state' / 'blobs' / 'updated.safetensors'
        atomic_save_tensors(scratch, tensors)
        payload = scratch.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = scratch.with_name(digest + '.safetensors')
        scratch.rename(destination)
        generation['roles']['core'] = {'path': 'blobs/' + destination.name, 'sha256': digest, 'bytes': len(payload)}
        generation.pop('contentSha256')
        identity = hashlib.sha256(canonical(generation)).hexdigest()
        generation['contentSha256'] = identity
        relative = 'generations/' + identity + '/manifest.json'
        path = self.root / 'state' / relative
        atomic_write_json(path, generation)
        brain['mutable_state'] = {'activeGeneration': identity, 'contentSha256': identity,
            'generationManifest': relative, 'generationManifestSha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        atomic_write_json(self.root / 'brain.json', brain)

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

    def test_selected_projection_reads_actual_gain_and_resistance(self):
        self.replace_core({self.name: torch.tensor([[0x64, 0x59], [0x64, 0x59]], dtype=torch.uint8),
            'decoder.language_head._connection_state': torch.tensor([[0xF7, 0, 2, 0, 0, 0, 0], [0] * 7], dtype=torch.uint8)})
        page = query_committed_cortex(self.root, 'b', {'entity': 'elements', 'moduleId': self.name, 'row': 0})
        self.assertEqual(page['records'][0]['value'], -1)
        self.assertEqual(page['records'][0]['strength'], 8)
        self.assertEqual(page['records'][0]['resistance'], 15)
        self.assertEqual(page['records'][0]['effectiveContribution'], -8)
        self.assertEqual(page['connectionStateBytes'], 14)
        links = query_committed_cortex(self.root, 'b', {'entity': 'links', 'moduleId': self.name, 'row': 0})
        self.assertEqual(links['records'][0]['effectiveContribution'], -8)

    def test_repeated_contact_slots_expose_real_endpoints_and_skip_tombstones(self):
        from omni_core.model import PackedAdaptiveBitLinear
        layer = PackedAdaptiveBitLinear(7, 2, bias=False)
        layer.fill_ternary_(1)
        layer.add_parallel_connection_(1, 3, sign=-1, strength=5)
        layer.add_parallel_connection_(1, 3, sign=1, strength=7)
        page = next(iter(layer.parallel_connections.pages.values()))
        page.write(0, 0, 0, 0x55, 0, remove=True)
        self.replace_core({'decoder.language_head.' + name: tensor for name, tensor in layer.state_dict().items()})
        inventory = query_committed_cortex(self.root, 'b', {})
        repeated = next(item for item in inventory['records'] if item.get('storageKind') == 'repeated-contact-page')
        selected = {'entity': 'elements', 'moduleId': repeated['id'], 'row': 0, 'pageSize': 1}
        empty = query_committed_cortex(self.root, 'b', selected)
        self.assertEqual(empty['records'], [])
        self.assertEqual(empty['scannedThrough'], 1)
        next_page = query_committed_cortex(self.root, 'b', {**selected, 'cursor': empty['nextCursor']})
        record = next_page['records'][0]
        self.assertEqual((record['row'], record['column'], record['value'], record['strength'], record['ordinal']), (1, 3, 1, 7, 1))
        self.assertTrue(record['isRepeatedContact'])
        self.assertFalse(record['activationObserved'])
        self.assertFalse(inventory['logicalInventoryComplete'])


if __name__ == "__main__":
    unittest.main()
