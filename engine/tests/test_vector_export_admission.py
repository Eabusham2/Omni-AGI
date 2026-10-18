import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.memory_inference_export import _RowScratchAdmission, vector_tensor_spec
from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.ternary_packing import export_ternary_shards, verify_ternary_shards
from omni_core.weighted_vsa_row import CONTACT


class VectorExportAdmissionTests(unittest.TestCase):
    def test_actual_paged_native_stream_matches_resident_and_retries_refused_page_without_skips(self):
        from omni_core.paged_packed_vectors import PagedPackedVectors
        with tempfile.TemporaryDirectory() as folder:
            resident = PackedTernaryVectors(7)
            paged = PagedPackedVectors(Path(folder) / 'vectors.sqlite3', 7)
            for index in range(37):
                key = str(index)
                resident[key] = paged[key] = torch.tensor([1, 0, -1, 1, 0, 1, -1])
            state, contacts = bytes([15, 3, 0, 2, 1, 5, 1]), CONTACT.pack(4, 0x56, 2) + CONTACT.pack(0, 0x54, 7)
            resident.set_connection_state('1', state, contacts)
            paged.set_packed('1', resident.row_payload('1'))
            attempts = []
            paged._memory_reserve = lambda amount, why: attempts.append((amount, why)) or amount <= 22_000
            first_spec, first_binding = vector_tensor_spec(resident)
            second_spec, second_binding = vector_tensor_spec(paged)
            # No row-read fallback. Larger refused page admissions cannot
            # secretly materialize before their existing provider guard.
            with patch.object(paged, 'row_payload', side_effect=AssertionError('unbounded per-row admission path')):
                export_ternary_shards(Path(folder) / 'resident', [first_spec], metadata=first_binding)
                export_ternary_shards(Path(folder) / 'paged', [second_spec], metadata=second_binding)
            first = verify_ternary_shards(Path(folder) / 'resident')
            second = verify_ternary_shards(Path(folder) / 'paged')
            self.assertEqual(first_binding, second_binding)
            self.assertEqual(first.manifest, second.manifest)
            self.assertEqual(first.parallel_connections, second.parallel_connections)
            self.assertTrue(any(amount > 22_000 for amount, _ in attempts))
            self.assertTrue(any(amount <= 22_000 for amount, _ in attempts))
            self.assertTrue(all(why == 'paged vector page' for _, why in attempts))

    def test_larger_periodic_row_and_cancellation_checks_are_not_skipped(self):
        now, calls, cancelled = [0.], [], [False]
        guard = _RowScratchAdmission(lambda size, reason: calls.append((size, reason)) or True,
            lambda: cancelled[0], clock=lambda: now[0])
        guard.check(100, 'first'); guard.check(80, 'same peak')
        self.assertEqual([size for size, _ in calls], [100])
        guard.check(101, 'larger immediate')
        now[0] = .26; guard.check(80, 'fresh pressure')
        self.assertEqual([size for size, _ in calls], [100, 101, 80])
        guard.check(90, 'larger than current admitted peak')
        self.assertEqual(calls[-1][0], 90)
        for _ in range(256):
            guard.check(80, 'bounded row count')
        self.assertEqual(calls[-1], (80, 'bounded row count'))
        cancelled[0] = True
        with self.assertRaises(InterruptedError):
            guard.check(1, 'cached approval still cancels')
        denied = _RowScratchAdmission(lambda *_args: False, clock=lambda: 0.)
        with self.assertRaisesRegex(RuntimeError, 'scratch memory'):
            denied.check(1, 'denied immediately')
        self.assertEqual(denied.approved, 0)

    def test_streamed_exact_companions_and_order_match_without_per_row_os_sampling(self):
        vectors = PackedTernaryVectors(7)
        for index in range(40):
            vectors[str(index)] = torch.tensor([1, 0, -1, 1, 0, 1, -1])
        vectors.set_connection_state('1', bytes([15, 1, 3, 2, 0, 5, 1]),
            CONTACT.pack(4, 0x56, 2) + CONTACT.pack(0, 0x54, 7))
        approvals = []
        with tempfile.TemporaryDirectory() as folder, patch(
                'omni_core.memory_inference_export.time.monotonic', return_value=1.):
            plain, plain_binding = vector_tensor_spec(vectors)
            guarded, guarded_binding = vector_tensor_spec(vectors,
                reserve=lambda size, why: approvals.append((size, why)) or True)
            export_ternary_shards(Path(folder) / 'plain', [plain], metadata=plain_binding)
            export_ternary_shards(Path(folder) / 'guarded', [guarded], metadata=guarded_binding)
            first = verify_ternary_shards(Path(folder) / 'plain')
            second = verify_ternary_shards(Path(folder) / 'guarded')
            self.assertEqual(plain_binding, guarded_binding)
            self.assertEqual(first.manifest, second.manifest)
            self.assertTrue(torch.equal(first.tensors[plain.name], second.tensors[plain.name]))
            self.assertTrue(torch.equal(first.connection_states[plain.name], second.connection_states[plain.name]))
            self.assertEqual(first.parallel_connections, second.parallel_connections)
            # Four complete traversals still visit all40 rows. OS admission
            # probes are bounded pressure/peak checks, not a count-based cap.
            self.assertGreaterEqual(len(approvals), 4)
            self.assertLess(len(approvals), 20)


if __name__ == '__main__':
    unittest.main()
