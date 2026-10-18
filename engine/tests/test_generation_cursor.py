"""Tiny native sampling fixture. No user brain or language training."""
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.config import OmniConfig
from omni_core.generation_cursor import NativeGenerationSuspended
from omni_core.model import OmniDecoder
from omni_core.working_attention_paging import WorkingAttentionPager
from omni_core.execution_state_codec import ExecutionStateWriter, ExecutionStateReader


class GenerationCursorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(912)
        self.decoder = OmniDecoder(OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1,
            n_layers=1, d_ff=16, max_seq_len=16, working_memory_slots=4,
            vsa_dim=16, router_neurons=4)).eval()
        self.prompt = torch.tensor([[259, 65, 260]])
        self.options = dict(max_new_tokens=9, top_k=12, noise=.05, seed=5, temperature=.75)

    def test_same_invocation_resumes_without_duplicate_tokens_or_actions(self):
        reference_tokens, reference_entropy = self.decoder.generate(self.prompt, **self.options)
        output, actions = [], []
        def emit(value, step, entropy):
            output.extend(value.reshape(-1).tolist())
        def activity(_state, step, entropy):
            actions.append(step)
        with self.assertRaises(NativeGenerationSuspended) as suspended:
            self.decoder.generate(self.prompt, **self.options, token_callback=emit, activity_callback=activity,
                suspension_check=lambda: len(actions) >= 3)
        cursor = suspended.exception.cursor
        old_count = len(output)
        tokens, entropy = self.decoder.generate(self.prompt, **self.options, token_callback=emit,
            activity_callback=activity, resume_cursor=cursor)
        self.assertTrue(torch.equal(tokens, reference_tokens))
        self.assertEqual(entropy, reference_entropy)
        self.assertEqual(output, tokens[0, self.prompt.shape[1]:].tolist())
        self.assertEqual(actions, list(range(len(reference_entropy))))
        self.assertGreater(old_count, 0)
        self.assertIsNone(cursor.cache)

    def test_suspension_transfers_live_paged_cache_until_the_owner_releases_it(self):
        with tempfile.TemporaryDirectory(prefix="omni-cursor-") as root:
            pager = WorkingAttentionPager(Path(root), resident_budget_bytes=4096,
                scratch_budget_bytes=65536, device_tile_budget_bytes=65536)
            self.decoder.configure_working_attention(pager)
            calls = []
            with self.assertRaises(NativeGenerationSuspended) as suspended:
                self.decoder.generate(self.prompt, **self.options, activity_callback=lambda *_: calls.append(1),
                    suspension_check=lambda: len(calls) >= 2)
            self.assertGreater(pager.status()["livePages"], 0)
            suspended.exception.cursor.close()
            self.assertEqual(pager.status()["livePages"], 0)
            pager.close()

    def test_resume_refuses_changed_sampling_policy_or_input(self):
        with self.assertRaises(NativeGenerationSuspended) as suspended:
            self.decoder.generate(self.prompt, **self.options, suspension_check=lambda: True)
        cursor = suspended.exception.cursor
        with self.assertRaisesRegex(ValueError, "sampling options"):
            self.decoder.generate(self.prompt, **{**self.options, "temperature": .2}, resume_cursor=cursor)
        with self.assertRaisesRegex(ValueError, "different input"):
            self.decoder.generate(torch.tensor([[259, 66, 260]]), **self.options, resume_cursor=cursor)
        cursor.close()

    def test_cold_paged_cache_roundtrip_keeps_exact_remaining_tokens_and_rng(self):
        with tempfile.TemporaryDirectory(prefix="omni-cold-cursor-") as folder:
            root = Path(folder); snapshot = root / "snapshot"; snapshot.mkdir()
            pager = WorkingAttentionPager(root / "old", resident_budget_bytes=4096, scratch_budget_bytes=65536, device_tile_budget_bytes=65536)
            self.decoder.configure_working_attention(pager)
            bias = torch.arange(8, dtype=torch.float32).reshape(1, 8) / 8
            expected, entropy = self.decoder.generate(self.prompt, memory_bias=bias, **self.options)
            emitted, activity = [], []
            def emit(value, *_): emitted.extend(value.reshape(-1).tolist())
            with self.assertRaises(NativeGenerationSuspended) as suspended:
                self.decoder.generate(self.prompt, memory_bias=bias, **self.options, token_callback=emit,
                    activity_callback=lambda *_: activity.append(1), suspension_check=lambda: len(activity) >= 3)
            original = suspended.exception.cursor
            writer = ExecutionStateWriter(snapshot)
            tree = writer.encode(original)
            original.close(); pager.close()
            replacement = OmniDecoder(self.decoder.config).eval()
            replacement.load_state_dict(self.decoder.state_dict())
            restored_pager = WorkingAttentionPager(root / "new", resident_budget_bytes=4096, scratch_budget_bytes=65536, device_tile_budget_bytes=65536)
            replacement.configure_working_attention(restored_pager)
            reader = ExecutionStateReader(snapshot, writer.files, pager=restored_pager, device="cpu")
            cursor = reader.decode(tree)
            cursor.bind_verified_cache(replacement)
            reader.release()
            actual, actual_entropy = replacement.generate(self.prompt, **self.options, resume_cursor=cursor, token_callback=emit)
            self.assertTrue(torch.equal(actual, expected)); self.assertEqual(actual_entropy, entropy)
            self.assertEqual(emitted, actual[0, self.prompt.shape[1]:].tolist())
            self.assertEqual(restored_pager.status()["livePages"], 0)
            restored_pager.close()

    def test_codec_rejects_changed_activity_and_unknown_execution_types(self):
        with tempfile.TemporaryDirectory(prefix="omni-corrupt-cursor-") as folder:
            root = Path(folder)
            writer = ExecutionStateWriter(root)
            tree = writer.encode(torch.tensor([1, 2, 3]))
            path = root / tree["file"]
            content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
            reader = ExecutionStateReader(root, writer.files, pager=None, device="cpu")
            with self.assertRaisesRegex(ValueError, "checksum"):
                reader.decode(tree)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                writer.encode(lambda: None)


if __name__ == "__main__":
    unittest.main()
