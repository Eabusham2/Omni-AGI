import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder
from omni_core.working_attention_paging import WorkingAttentionPager, current_working_attention_pager


class CheckpointPagerContextTests(unittest.TestCase):
    def test_backward_recompute_keeps_same_pager_after_forward_scope_exits(self):
        torch.set_num_threads(1)
        config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1,
            d_ff=16, vsa_dim=16, router_neurons=4, max_seq_len=16,
            working_memory_slots=4, gradient_checkpointing=True)
        with tempfile.TemporaryDirectory() as directory:
            pager = WorkingAttentionPager(Path(directory), resident_budget_bytes=4096,
                scratch_budget_bytes=4 * 1024**2, device_tile_budget_bytes=2 * 1024**2, page_tokens=2)
            decoder = OmniDecoder(config).train()
            decoder.working_attention_pager = pager
            observed = []
            hook = decoder.blocks[0].register_forward_pre_hook(
                lambda _module, _inputs: observed.append(current_working_attention_pager()))
            try:
                tokens = torch.tensor([[1, 68, 2]])
                loss = decoder(tokens, labels=tokens)["loss"]
                self.assertIsNone(current_working_attention_pager())
                loss.backward()
                self.assertGreaterEqual(len(observed), 2)
                self.assertTrue(all(value is pager for value in observed))
                self.assertIsNone(current_working_attention_pager())
            finally:
                hook.remove()
                pager.close()


if __name__ == "__main__":
    unittest.main()
