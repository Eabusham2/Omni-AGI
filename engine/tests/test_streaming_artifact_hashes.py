"""Regression checks for bounded-memory artifact checksums."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain


class StreamingArtifactHashTests(unittest.TestCase):
    def test_modality_pack_file_hash_matches_previous_byte_formula(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary) / "modality.safetensors"
            content = b"modality pack\x00" * 83 + b"tail"
            pack.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("artifact hash must stream"),
            ):
                self.assertEqual(AdaptiveBrain._file_sha256(pack), expected)

    def test_snapshot_hash_matches_concatenated_byte_formula_across_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core.safetensors"
            plasticity = root / "plasticity.safetensors"
            core_bytes = b"core\x00" * 209716 + b"end"
            plasticity_bytes = b"plasticity\xff" * 37
            core.write_bytes(core_bytes)
            plasticity.write_bytes(plasticity_bytes)
            substrate_sha = "a" * 64
            mutable_sha = "b" * 64
            expected = hashlib.sha256(
                core_bytes
                + plasticity_bytes
                + substrate_sha.encode("ascii")
                + mutable_sha.encode("ascii")
            ).hexdigest()

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("snapshot hash must stream"),
            ):
                actual = AdaptiveBrain._snapshot_checksum(
                    core, plasticity, substrate_sha, mutable_sha
                )

            self.assertEqual(actual, expected)

    def test_snapshot_hash_rejects_nonregular_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core.safetensors"
            core.write_bytes(b"core")
            with self.assertRaisesRegex(ValueError, "regular file"):
                AdaptiveBrain._snapshot_checksum(core, root, "", "")

            linked = root / "linked.safetensors"
            try:
                linked.symlink_to(core)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable on this platform")
            with self.assertRaisesRegex(ValueError, "regular file"):
                AdaptiveBrain._snapshot_checksum(linked, core, "", "")


if __name__ == "__main__":
    unittest.main()
