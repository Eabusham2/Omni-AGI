"""New inspection contracts using tiny files/fake owners, never a neural constructor."""
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from omni_core.cortical_activity_descriptor import describe_forward_output
from omni_core.cortical_inspection import query_committed_cortex
from omni_core.native_core_paging import NativeCorePager


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def checkpoint(root, tensors, tokens=()):
    blobs = root / "state" / "blobs"
    blobs.mkdir(parents=True)
    roles = {}
    for role, values in (("core", tensors), ("plasticity", {})):
        temporary = blobs / (role + ".safetensors")
        save_file(values, str(temporary))
        encoded = temporary.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        destination = blobs / (digest + ".safetensors")
        temporary.rename(destination)
        roles[role] = {"path": "blobs/" + destination.name, "sha256": digest, "bytes": len(encoded)}
    body = {"format": "omni-mutable-state", "formatVersion": 1, "brainId": "b", "roles": roles}
    identity = hashlib.sha256(canonical(body)).hexdigest()
    encoded = canonical({**body, "contentSha256": identity})
    relative = "generations/" + identity + "/manifest.json"
    manifest = root / "state" / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(encoded)
    (root / "brain.json").write_bytes(canonical({"brain_id": "b", "recent_token_context": list(tokens),
        "mutable_state": {"activeGeneration": identity, "contentSha256": identity, "generationManifest": relative,
                          "generationManifestSha256": hashlib.sha256(encoded).hexdigest()}}))


class DummyOwner:
    pass


class CorticalViewerEvidenceTests(unittest.TestCase):
    def test_bias_links_are_constant_to_actual_output_and_validate_codes(self):
        with tempfile.TemporaryDirectory(prefix="omni-cortex-bias-") as folder:
            root = Path(folder)
            name = "decoder.head._packed_forward_bias"
            checkpoint(root, {name: torch.tensor([[0x62]], dtype=torch.uint8),
                              "decoder.head._bias_connection_state": torch.tensor([[0xF7, 0, 0x23, 0]], dtype=torch.uint8)})
            page = query_committed_cortex(root, "b", {"entity": "links", "moduleId": name, "offset": 2, "pageSize": 1})
            contact = page["records"][0]
            self.assertEqual((contact["sourceAxis"], contact["sourceIndex"], contact["targetAxis"], contact["targetIndex"]),
                             ("constant-one", 0, "output-row", 2))
            self.assertEqual((contact["strength"], contact["resistance"]), (4, 2))
            self.assertTrue(page["integrity"]["selectedTernaryCodesValidated"])

    def test_repeated_inventory_separates_capacity_and_known_base_parameters(self):
        with tempfile.TemporaryDirectory(prefix="omni-cortex-contact-inventory-") as folder:
            root = Path(folder)
            owner = "decoder.head"
            repeated = owner + ".parallel_connections.pages.w_r0_p0"
            registry = torch.zeros((1024, 10), dtype=torch.uint8)
            registry[0, 0:3] = torch.tensor([1, 1, 2], dtype=torch.uint8)
            checkpoint(root, {owner + "._packed_forward_weight": torch.full((2, 1), 0x55, dtype=torch.uint8),
                owner + "._connection_state": torch.zeros((2, 4), dtype=torch.uint8),
                repeated + "._packed_sign": torch.full((1, 256), 0x55, dtype=torch.uint8),
                repeated + "._connection_state": torch.zeros((1, 1024), dtype=torch.uint8),
                repeated + "._record_meta": registry})
            page = query_committed_cortex(root, "b", {})
            self.assertFalse(page["logicalInventoryComplete"])
            self.assertEqual(page["inventory"], {"baseLogicalParameters": 8, "baseLogicalInventoryComplete": True,
                "repeatedContactPages": 1, "allocatedRepeatedContactSlots": 1024, "repeatedLiveContacts": None})
            contact_owner = next(value for value in page["records"] if value["storageKind"] == "repeated-contact-page")
            self.assertEqual(contact_owner["activityOwner"], owner)
            self.assertEqual(page["relationships"]["ideaInputs"], [])
            self.assertEqual(page["relationships"]["tokenInputs"], [])

    def test_malformed_token_metadata_cannot_create_same_offset_paging_loop(self):
        with tempfile.TemporaryDirectory(prefix="omni-cortex-bad-token-") as folder:
            root = Path(folder)
            checkpoint(root, {}, [True, 100])
            with self.assertRaisesRegex(ValueError, "token boundary"):
                query_committed_cortex(root, "b", {"entity": "boundaries", "pageSize": 1})

    def test_actual_owner_descriptors_do_not_infer_from_names_or_storage(self):
        self.assertEqual(describe_forward_output(SimpleNamespace(num_embeddings=99, embedding_dim=7)),
                         {"outputAxis": "embedding-column", "outputSize": 7})
        self.assertEqual(describe_forward_output(SimpleNamespace(out_features=9, name="scale_delta")),
                         {"outputAxis": "output-row", "outputSize": 9})
        self.assertEqual(describe_forward_output(SimpleNamespace(dimensions=2, out_channels=3)),
                         {"outputAxis": "output-channel", "outputSize": 3})
        self.assertIsNone(describe_forward_output(SimpleNamespace(name="decoder.embedding", _packed_sign=b"x")))

    def test_capture_scope_cleanup_cannot_disable_new_or_unrelated_owner(self):
        module, storage = DummyOwner(), DummyOwner()
        module.num_embeddings, module.embedding_dim = 20, 7
        pager = SimpleNamespace(_lock=threading.RLock(), _owners={module: SimpleNamespace(name="real"), storage: SimpleNamespace(name="page")},
            _observations={}, _observation_requests={}, _observation_scopes={}, chunk_bytes=64)
        first = NativeCorePager.observation(pager, "real", enabled=True, start=3, count=99, capture_id="old")
        self.assertEqual((first["outputAxis"], first["admittedCount"]), ("embedding-column", 2))
        NativeCorePager.observation(pager, "real", enabled=True, start=4, count=1, capture_id="new")
        NativeCorePager.observation(pager, "real", enabled=False, capture_id="old")
        self.assertEqual(pager._observation_requests, {"real": (4, 1)})
        self.assertFalse(NativeCorePager.observation(pager, "page", enabled=False)["available"])
        self.assertFalse(NativeCorePager.observation(pager, "not-an-owner", enabled=True)["available"])
        self.assertEqual(pager._observation_requests, {"real": (4, 1)})
        NativeCorePager.observation(pager, "real", enabled=False, capture_id="new")
        self.assertEqual(pager._observation_requests, {})
        self.assertFalse(NativeCorePager.observation(pager, "real", enabled=True, start=7)["available"])


if __name__ == "__main__":
    unittest.main()
