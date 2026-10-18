"""Transient cue scaling only: no cortical model initialization or training."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.streamed_vsa_activity import (
    activity_block_rows, activity_reduction_scratch_bytes, streamed_activity_mean,
)
from omni_core.vsa import NeuralSubstrate
from omni_core.weighted_vsa_row import CONTACT


def semantic_id(label):
    return hashlib.sha256(("semantic:" + label).encode()).hexdigest()[:24]


def reference_cue(memory, labels):
    if not labels:
        return F.normalize(memory.space.symbol("empty-assembly").reshape(1, -1), dim=-1)[0]
    rows = []
    for index, label in enumerate(labels):
        neuron = memory.neuron_vectors.get(semantic_id(label), memory.space.symbol("semantic-neuron:" + label))
        rows.append(memory.space.permute(memory.space.bind(neuron, memory.space.symbol("position:%d" % index)), index + 1))
    activity = torch.stack(rows).mean(0)
    if float(activity.norm()) <= 1e-8:
        activity = memory.space.bundle(rows)
    return F.normalize(activity.reshape(1, -1), dim=-1)[0]


class StreamedVsaActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def memory(self):
        memory = NeuralSubstrate(64, seed=947)
        for index in range(11):
            label = "learned-%d" % index
            identifier = semantic_id(label)
            memory.neuron_vectors[identifier] = memory.space.symbol("source:" + label)
            memory.neuron_vectors.set_connection_state(identifier, bytes([index % 16] * 64), CONTACT.pack(index, 0x56, 7))
        return memory

    def test_empty_and_short_cues_are_bit_identical_to_existing_mean(self):
        memory = self.memory()
        for count in (0, 1, 7, 31, 32):
            with self.subTest(count=count):
                labels = ["learned-%d" % (index % 11) for index in range(count)]
                self.assertTrue(torch.equal(reference_cue(memory, labels), memory.vector_for_labels(labels)))

    def test_long_mixed_weighted_cues_preserve_numerical_superposition(self):
        memory = self.memory()
        before = {key: memory.neuron_vectors.row_payload(key) for key in memory.neuron_vectors}
        rng = torch.random.get_rng_state().clone()
        for count in (33, 257, 2049):
            with self.subTest(count=count):
                labels = ["learned-%d" % (index % 11) if index % 3 else "new-%d" % index for index in range(count)]
                expected = reference_cue(memory, labels)
                actual = memory.vector_for_labels(labels)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        self.assertEqual(before, {key: memory.neuron_vectors.row_payload(key) for key in memory.neuron_vectors})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(memory.state_revision, 0)

    def test_stack_scratch_is_bounded_without_limiting_label_traversal(self):
        memory = self.memory()
        labels = ["learned-%d" % (index % 11) for index in range(4097)]
        stack = torch.stack
        shapes = []
        def checked_stack(rows, *args, **kwargs):
            shapes.append(len(rows))
            self.assertLessEqual(len(rows), activity_block_rows(memory.space.dimensions))
            return stack(rows, *args, **kwargs)
        with patch("torch.stack", side_effect=checked_stack):
            value = memory.vector_for_labels(labels)
        self.assertEqual(sum(shapes), len(labels))
        self.assertAlmostEqual(float(value.norm()), 1.0, places=6)
        self.assertEqual(activity_reduction_scratch_bytes(64, 4097), activity_reduction_scratch_bytes(64, 4_000_000))

    def test_learned_rows_do_not_generate_unused_default_symbols(self):
        memory = self.memory()
        original = memory.space.symbol
        names = []
        def symbol(name):
            names.append(name)
            return original(name)
        with patch.object(memory.space, "symbol", side_effect=symbol):
            memory.vector_for_labels(["learned-0", "learned-1", "not-yet-learned"])
        self.assertEqual(names, ["position:0", "position:1", "semantic-neuron:not-yet-learned", "position:2"])

    def test_zero_superposition_fallback_and_continuous_assembly_mean(self):
        memory = self.memory()
        with patch.object(memory.space, "bind", return_value=torch.zeros(64)):
            actual = memory.vector_for_labels(["one", "two"])
            torch.testing.assert_close(actual, torch.full((64,), 1 / 8))
        identifiers = [semantic_id("learned-%d" % (index % 11)) for index in range(257)]
        rows = [memory.space.permute(memory.space.bind(memory.neuron_vectors[identifier], memory.space.symbol("position:%d" % index)), index + 1)
                for index, identifier in enumerate(identifiers)]
        torch.testing.assert_close(memory._continuous_assembly_activity(identifiers), torch.stack(rows).mean(0), rtol=2e-6, atol=2e-7)

    def test_paged_weighted_rows_remain_unchanged(self):
        memory = self.memory()
        metadata, tensors = memory.neuron_vectors.export_state()
        with tempfile.TemporaryDirectory() as directory:
            paged = PagedPackedVectors(Path(directory) / "rows.sqlite3", 64, seed=947)
            paged.import_state(metadata, tensors)
            memory.neuron_vectors = paged
            before = {key: paged.row_payload(key) for key in paged}
            labels = ["learned-%d" % (index % 11) for index in range(129)]
            torch.testing.assert_close(memory.vector_for_labels(labels), reference_cue(memory, labels), rtol=2e-6, atol=2e-7)
            self.assertEqual(before, {key: paged.row_payload(key) for key in paged})

    def test_cooperative_cancellation_precedes_next_row_allocation(self):
        class Cancelled(RuntimeError):
            pass
        class Pager:
            calls = 0
            def check_cancelled(self):
                self.calls += 1
                if self.calls == 5:
                    raise Cancelled()
        visited = []
        def rows():
            for index in range(100):
                visited.append(index)
                yield torch.ones(64)
        with patch("omni_core.working_attention_paging.current_working_attention_pager", return_value=Pager()):
            with self.assertRaises(Cancelled):
                streamed_activity_mean(rows(), 64)
        self.assertEqual(visited, [0, 1, 2, 3])

    def test_wide_vectors_shrink_the_block_instead_of_a_neuron_limit(self):
        self.assertEqual(activity_block_rows(64), 32)
        self.assertEqual(activity_block_rows(4096), 4)
        self.assertEqual(activity_block_rows(1_000_000), 1)
        with self.assertRaises(ValueError):
            streamed_activity_mean(iter(()), 64)


if __name__ == "__main__":
    unittest.main()
