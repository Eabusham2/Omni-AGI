"""Snapshot closure checks for inspection catalogs; no brain/model is built."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omni_core.persistence import snapshot_files, snapshot_required_bytes


class WholeBrainCatalogPersistenceTests(unittest.TestCase):
    def test_snapshot_sizes_and_copies_current_and_lineage_closures(self):
        with tempfile.TemporaryDirectory(prefix="omni-catalog-persistence-") as temporary:
            source = Path(temporary) / "source"
            destination = Path(temporary) / "destination"
            source.mkdir()
            metadata = {"brain_id": "catalog-persistence-owner"}
            (source / "brain.json").write_text(json.dumps(metadata), encoding="utf-8")
            (source / "core.safetensors").write_bytes(b"core")
            (source / "plasticity.safetensors").write_bytes(b"plasticity")
            current = "whole-brain-catalog/blobs/" + "a" * 64 + ".json"
            lineage = "whole-brain-catalog/lineage/blobs/" + "b" * 64 + ".json"
            payloads = {current: b"current catalog page", lineage: b"lineage catalog page"}
            for relative, payload in payloads.items():
                path = source.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            def rows(relative):
                payload = payloads[relative]
                return [(relative, len(payload), hashlib.sha256(payload).hexdigest())]

            with mock.patch("omni_core.whole_brain_catalog.iter_catalog_files", side_effect=lambda *args, **kwargs: iter(rows(current))), \
                 mock.patch("omni_core.whole_brain_catalog.iter_catalog_lineage_files", side_effect=lambda *args, **kwargs: iter(rows(lineage))):
                expected = sum((source / name).stat().st_size for name in
                               ("brain.json", "core.safetensors", "plasticity.safetensors")) + sum(map(len, payloads.values()))
                self.assertEqual(snapshot_required_bytes(source), expected)
                snapshot_files(source, destination)

            for relative, payload in payloads.items():
                self.assertEqual(destination.joinpath(*relative.split("/")).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
