"""Small, source-free tests of the local decoder's learned memory cue."""

import unittest

import torch

from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder, packed_runtime_status
from omni_core.tokenizer import ByteTokenizer


class DecoderLatentRecallTests(unittest.TestCase):
    def test_workspace_language_logits_are_prefix_causal_and_padding_safe(self):
        torch.manual_seed(7102)
        torch.set_num_threads(1)
        model = OmniDecoder(
            OmniConfig.micro(
                max_seq_len=32,
                working_memory_slots=16,
                memory_resident_items=16,
                dropout=0.0,
            )
        ).eval()
        first = torch.tensor([[1, 259, 70, 71, 72, 260, 80, 81]])
        changed_future = first.clone()
        changed_future[0, -2:] = torch.tensor([110, 111])
        with torch.no_grad():
            for explicit in (None, True):
                left = model(first, use_global_workspace=explicit)["logits"]
                right = model(changed_future, use_global_workspace=explicit)[
                    "logits"
                ]
                self.assertTrue(torch.allclose(left[:, :-2], right[:, :-2], atol=1e-6))

            padded = torch.cat([first, torch.tensor([[7, 8]])], dim=1)
            mask = torch.tensor([[True] * 8 + [False, False]])
            masked = model(padded, attention_mask=mask)["logits"][:, :8]
            individual = model(first)["logits"]
            self.assertTrue(torch.allclose(masked, individual, atol=1e-6))

    def test_paraphrased_question_can_decode_a_parameter_learned_latent_answer(self):
        torch.manual_seed(7103)
        torch.set_num_threads(1)
        config = OmniConfig.micro(
            max_seq_len=64,
            working_memory_slots=16,
            memory_resident_items=16,
            gradient_checkpointing=False,
            dropout=0.0,
        )
        model = OmniDecoder(config)
        tokenizer = ByteTokenizer()
        memory = torch.zeros(2, config.idea_dim)
        memory[0, 0] = 4.0
        memory[1, 1] = 4.0
        answers = ("amber", "cobalt")
        questions = (
            "What color was the map?",
            "Tell me the map color.",
        )
        examples = []
        for index, answer in enumerate(answers):
            for question in questions:
                token_ids = tokenizer.dialogue(question, answer, complete=True)
                ids = torch.tensor([token_ids], dtype=torch.long)
                labels = ids.clone()
                labels[:, : token_ids.index(tokenizer.brain_id) + 1] = 0
                examples.append((ids, labels, memory[index : index + 1]))

        initial_workspace = (
            model.global_workspace.query.packed_forward_weight().detach().clone()
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        early_losses = None
        for step in range(80):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                model(ids, memory_bias=cue, labels=labels)["loss"]
                for ids, labels, cue in examples
            ) / len(examples)
            loss.backward()
            optimizer.step()
            if step == 9:
                with torch.no_grad():
                    probe_ids = tokenizer.dialogue(
                        "Which color is the map?", answers[0], complete=True
                    )
                    probe = torch.tensor([probe_ids], dtype=torch.long)
                    probe_labels = probe.clone()
                    probe_labels[:, : probe_ids.index(tokenizer.brain_id) + 1] = 0
                    active = model(
                        probe, memory_bias=memory[0:1], labels=probe_labels
                    )["loss"].item()
                    disabled = model(
                        probe,
                        memory_bias=memory[0:1],
                        labels=probe_labels,
                        use_global_workspace=False,
                    )["loss"].item()
                early_losses = (active, disabled)

        model.eval()
        self.assertIsNotNone(early_losses)
        self.assertLess(early_losses[0], early_losses[1])
        self.assertFalse(
            torch.equal(
                model.global_workspace.query.packed_forward_weight(),
                initial_workspace,
            )
        )
        self.assertTrue(packed_runtime_status(model)["complete"])
        for index, answer in enumerate(answers):
            heldout_ids = tokenizer.dialogue(
                "Which color is the map?", answer, complete=True
            )
            heldout = torch.tensor([heldout_ids], dtype=torch.long)
            heldout_labels = heldout.clone()
            heldout_labels[
                :, : heldout_ids.index(tokenizer.brain_id) + 1
            ] = 0
            prefix = torch.tensor(
                [tokenizer.dialogue("Which color is the map?", complete=False)],
                dtype=torch.long,
            )
            self.assertNotIn(answer, tokenizer.decode(prefix[0].tolist()))
            with torch.no_grad():
                conditioned_loss = model(
                    heldout,
                    memory_bias=memory[index : index + 1],
                    labels=heldout_labels,
                )["loss"].item()
            self.assertLess(conditioned_loss, 0.1)
            generated, _ = model.generate(
                prefix,
                memory_bias=memory[index : index + 1],
                max_new_tokens=8,
                temperature=1.0,
                top_k=1,
                noise=0.0,
                seed=1,
            )
            reply = tokenizer.decode(generated[0, prefix.shape[1] :].tolist())
            self.assertTrue(
                reply.startswith(answer),
                f"latent answer {answer!r} was decoded as {reply!r}",
            )


if __name__ == "__main__":
    unittest.main()
