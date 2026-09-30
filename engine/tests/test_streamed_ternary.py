"""Only bounded raw tensor/byte math; no brain/model/application constructors."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.streamed_ternary import TernaryPackedSource, source_chunks, source_hashes, dense_source, packed_rows_source, verify_file, PAGE_TRITS
from omni_core.ternary_packing import encode_ternary_2bit, _tensor_sha256, TernaryTensorSpec, export_ternary_shards, verify_ternary_shards, TernaryIntegrityError


class StreamedTernaryTests(unittest.TestCase):
    def test_arbitrary_page_boundaries_preserve_exact_encoding_and_tensor_digest(self):
        for count in (0, 1, 2, 3, 4, 5, 67, 1001):
            values = torch.tensor([(index % 3) - 1 for index in range(count)], dtype=torch.int8)
            source = TernaryPackedSource((count,), lambda: (values[start:start + 5] for start in range(0, count, 5)))
            payload = b"".join(source_chunks(source))
            self.assertEqual(payload, encode_ternary_2bit(values))
            self.assertEqual(source_hashes(source), (hashlib.sha256(payload).hexdigest(), _tensor_sha256(values)))

    def test_internal_row_padding_is_verified_then_repacked_across_rows(self):
        rows = torch.tensor([[-1, 0, 1, 0, -1], [1, 1, 0, -1, 0]], dtype=torch.int8)
        packed = torch.stack([torch.tensor(list(encode_ternary_2bit(row)), dtype=torch.uint8) for row in rows])
        source = packed_rows_source(packed, tuple(rows.shape))
        self.assertEqual(b"".join(source_chunks(source)), encode_ternary_2bit(rows))
        self.assertEqual(source_hashes(source)[1], _tensor_sha256(rows))
        packed[0, -1] |= 0xC0  # Internal padding becomes reserved/nonzero.
        with self.assertRaises(ValueError):
            list(source_chunks(packed_rows_source(packed, tuple(rows.shape))))

    def test_strided_and_empty_dense_compatibility_never_requires_full_contiguous_copy(self):
        values = torch.tensor([[-1, 0, 1], [1, 0, -1]], dtype=torch.int8).T
        self.assertFalse(values.is_contiguous())
        self.assertEqual(b"".join(source_chunks(dense_source(values))), encode_ternary_2bit(values))
        source = packed_rows_source(torch.empty((0, 2), dtype=torch.uint8), (0, 5))
        self.assertEqual(b"".join(source_chunks(source)), b"")

    def test_export_and_empty_retention_verify_more_than_one_page_without_whole_reads(self):
        count = PAGE_TRITS * 2 + 3
        def pages():
            remaining = count
            while remaining:
                size = min(remaining, 10001)
                yield torch.ones(size, dtype=torch.int8)
                remaining -= size
        source = TernaryPackedSource((count,), pages)
        with tempfile.TemporaryDirectory(prefix="omni-trit-stream-storage-") as folder:
            target = Path(folder)
            manifest = export_ternary_shards(target, [TernaryTensorSpec("actual.dynamic", source, kind="dynamic-synapse")])
            original = Path.read_bytes
            def no_whole_bin(path):
                if path.suffix == ".bin":
                    raise AssertionError("whole packed tensor read")
                return original(path)
            with mock.patch.object(Path, "read_bytes", no_whole_bin):
                checked = verify_ternary_shards(target, retain_names=())
            self.assertEqual(checked.tensors, {})
            self.assertEqual(manifest["tensors"][0]["numel"], count)
            self.assertEqual(manifest["tensors"][0]["byteLength"], (count + 3) // 4)
            path = target / manifest["tensors"][0]["shard"]
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                last = handle.read(1)[0]
                handle.seek(-1, 2)
                handle.write(bytes([last | 0xC0]))
            with self.assertRaises(TernaryIntegrityError):
                verify_file(path, source.shape)

    def test_incomplete_or_changed_shape_never_silently_truncates(self):
        source = TernaryPackedSource((9,), lambda: [torch.ones(8, dtype=torch.int8)])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            list(source_chunks(source))
        source = TernaryPackedSource((7,), lambda: [torch.ones(8, dtype=torch.int8)])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            list(source_chunks(source))


if __name__ == "__main__":
    unittest.main()
