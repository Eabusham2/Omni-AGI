"""Pure wire/state and scripted method fixtures, not a native neural run.

No brain/model constructor, learned projection, forward kernel, training,
optimizer, app or inference-quality claim. Programmed logits exercise only the
actual token-loop/cache/callback protocol against primitive tiny tensors.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.chat_steering import generation_completion
from omni_core.model import OmniDecoder
from omni_core.native_action_protocol import NativeActionEmissionLedger
from omni_core.tokenizer import ByteTokenizer
from omni_core.utf8_text_boundary import Utf8TextBoundary


class CacheFixture:
    def __init__(self): self.closes = 0
    def close(self): self.closes += 1


class ScriptedTokenLoop:
    """No neural model or module: only scripted logits and cache ownership."""
    _generate_scoped = OmniDecoder._generate_scoped

    def __init__(self, choice):
        self.config = SimpleNamespace(max_seq_len=64)
        self.choice = choice
        self.calls, self.caches = [], []
    def eval(self): pass
    def _working_pager(self): return None
    def logits(self, window, memory_bias, cache):
        self.calls.append((window.detach().clone(), memory_bias.detach().clone() if memory_bias is not None else None, cache))
        self.last_generation_neural_state = torch.ones((window.shape[0], 4))
        result = torch.full((window.shape[0], 261), -30.)
        result[:, 2] = 20. # fallback EOS when the scripted byte is invalid wire syntax
        result[:, self.choice(len(self.calls) - 1, memory_bias)] = 30.
        return result
    def _generation_step(self, window, memory_bias, cache):
        result = self.logits(window, memory_bias, cache)
        if cache is None:
            cache = CacheFixture()
            self.caches.append(cache)
        return result, cache
    def forward(self, window, memory_bias=None):
        logits = self.logits(window, memory_bias, None)
        return {"logits": logits[:, None], "hidden": self.last_generation_neural_state[:, None]}


class NativeTextConditioningBoundaryTests(unittest.TestCase):
    prompt = torch.tensor([[1, 259, 100, 260]], dtype=torch.long)

    def run_protocol(self, fixture, **kwargs):
        return OmniDecoder.generate(fixture, self.prompt, top_k=1, noise=0., seed=4, **kwargs)

    def test_initial_boundary_allows_eos_and_all_scalar_languages_not_only_ascii(self):
        boundary = Utf8TextBoundary()
        allowed = boundary.allowed_token_ids(4)
        self.assertIn(2, allowed)
        self.assertNotIn(3, allowed) # saved-message transport rejects NUL, not a language
        self.assertIn(0xD9 + 3, allowed)
        self.assertIn(0xE4 + 3, allowed)
        self.assertIn(0xF0 + 3, allowed)
        self.assertNotIn(0x80 + 3, allowed)
        self.assertNotIn(259, allowed) # role tokens are not response bytes

    def test_utf8_guards_reject_only_overlong_surrogate_and_out_of_range_encoding(self):
        for lead, invalid in ((0xE0, 0x9F), (0xED, 0xA0), (0xF0, 0x8F), (0xF4, 0x90)):
            boundary = Utf8TextBoundary()
            boundary.accept(lead + 3)
            self.assertNotIn(2, boundary.allowed_token_ids(3))
            with self.assertRaises(ValueError): boundary.accept(invalid + 3)
        for invalid in (0x80, 0xC0, 0xC1, 0xF5, 0xFF):
            with self.assertRaises(ValueError): Utf8TextBoundary().accept(invalid + 3)
        for text in ("العربية", "中文", "🙂", "\t\n"):
            boundary = Utf8TextBoundary()
            for value in text.encode(): boundary.accept(value + 3)
            self.assertTrue(boundary.complete)
            boundary.accept(2)

    def test_nul_cannot_be_generated_into_a_receipt_that_the_host_would_reject(self):
        with self.assertRaisesRegex(ValueError, "saved-message transport"):
            Utf8TextBoundary().accept(3)
        fixture = ScriptedTokenLoop(lambda *_: 3)
        generated, _ = self.run_protocol(fixture, max_new_tokens=4)
        self.assertEqual(generated[0, self.prompt.shape[1]:].tolist(), [2])

    def test_actual_initial_eos_reaches_existing_honest_no_reply_completion(self):
        fixture = ScriptedTokenLoop(lambda *_: 2)
        emitted = []
        generated, entropies = self.run_protocol(fixture, max_new_tokens=8, token_callback=lambda value, *_: emitted.extend(value.reshape(-1).tolist()))
        suffix = generated[0, self.prompt.shape[1]:].tolist()
        self.assertEqual(suffix, [2])
        self.assertEqual(fixture.last_generation_stop_reason, "learned-boundary")
        self.assertEqual(len(entropies), 1)
        completion = generation_completion(suffix, ByteTokenizer().decode)
        self.assertEqual(completion["text"], "")
        self.assertTrue(completion["noReply"])
        self.assertFalse(completion["zeroTokenYield"])
        self.assertFalse(completion["steered"])
        self.assertEqual(ByteTokenizer().decode(emitted), "")

    def test_multibyte_stream_emits_only_complete_scalars_and_equals_final_exact_text(self):
        text = "مرحباً 中文🙂"
        script = [*(value + 3 for value in text.encode()), 2]
        fixture = ScriptedTokenLoop(lambda index, _bias: script[index])
        emitted = []
        generated, _ = self.run_protocol(fixture, max_new_tokens=len(script), token_callback=lambda value, *_: emitted.append(ByteTokenizer().decode(value.reshape(-1).tolist())))
        actual = ByteTokenizer().decode(generated[0, self.prompt.shape[1]:].tolist())
        self.assertEqual(actual, text)
        self.assertEqual("".join(emitted), text)
        self.assertNotIn("�", "".join(emitted))

    def test_budget_does_not_start_an_unfinishable_scalar_or_force_an_ascii_answer(self):
        fixture = ScriptedTokenLoop(lambda *_: 0xF0 + 3)
        generated, _ = self.run_protocol(fixture, max_new_tokens=3)
        self.assertEqual(generated[0, self.prompt.shape[1]:].tolist(), [2])
        fixture = ScriptedTokenLoop(lambda index, _bias: [0xF0 + 3, 0x9F + 3, 0x98 + 3, 0x80 + 3][index])
        generated, _ = self.run_protocol(fixture, max_new_tokens=4)
        self.assertEqual(ByteTokenizer().decode(generated[0, self.prompt.shape[1]:].tolist()), "😀")
        self.assertEqual(fixture.last_generation_stop_reason, "token-budget")

    def test_steer_inside_scalar_preserves_only_actual_emitted_prefix_without_replacement(self):
        fixture = ScriptedTokenLoop(lambda index, _bias: [ord("A") + 3, 0xE4 + 3, 0xB8 + 3][index])
        emitted = []
        generated, entropies = self.run_protocol(fixture, max_new_tokens=8,
            steer_check=lambda: len(fixture.calls) >= 3,
            token_callback=lambda value, *_: emitted.extend(value.reshape(-1).tolist()))
        suffix = generated[0, self.prompt.shape[1]:].tolist()
        self.assertEqual(ByteTokenizer().decode(suffix), "A")
        self.assertEqual(suffix, emitted)
        self.assertEqual(len(entropies), 1)
        self.assertEqual(fixture.last_generation_stop_reason, "steered")

    def test_native_stop_inside_scalar_retains_zero_visible_output_not_invented_text(self):
        fixture = ScriptedTokenLoop(lambda index, _bias: [0xE4 + 3, 0xB8 + 3][index])
        generated, entropies = self.run_protocol(fixture, max_new_tokens=8,
            activity_callback=lambda _hidden, step, _entropy: {"stop": step == 1})
        self.assertTrue(torch.equal(generated, self.prompt))
        self.assertEqual(entropies, [])
        self.assertEqual(fixture.last_generation_stop_reason, "native-action-stop")

    def test_recurrent_ponder_cue_conditions_next_token_and_refills_exact_prefix(self):
        for cached in (True, False):
            def choice(index, bias):
                return 2 if index > 1 else ord("B" if bias is not None and bias[0, 0].item() == 2 else "A") + 3
            fixture = ScriptedTokenLoop(choice)
            generated, _ = self.run_protocol(fixture, memory_bias=torch.zeros(1, 4), max_new_tokens=8,
                use_cache=cached, activity_callback=lambda _hidden, step, _entropy: {"memoryBias": torch.full((1, 4), 2.)} if step == 0 else {})
            self.assertEqual(ByteTokenizer().decode(generated[0, self.prompt.shape[1]:].tolist()), "AB")
            self.assertEqual(fixture.calls[0][1][0, 0].item(), 0)
            self.assertEqual(fixture.calls[1][1][0, 0].item(), 2)
            self.assertIsNone(fixture.calls[1][2])
            self.assertTrue(torch.equal(fixture.calls[1][0][:, :-1], self.prompt))
            if cached: self.assertTrue(all(cache.closes == 1 for cache in fixture.caches))

    def test_invalid_recurrent_conditioning_raises_and_closes_the_real_owned_cache(self):
        fixture = ScriptedTokenLoop(lambda *_: ord("A") + 3)
        with self.assertRaisesRegex(ValueError, "invalid recurrent text conditioning"):
            self.run_protocol(fixture, max_new_tokens=8, activity_callback=lambda *_: {"memoryBias": torch.full((1, 4), float("nan"))})
        self.assertEqual(fixture.caches[0].closes, 1)

    def test_actual_brain_callback_returns_refined_same_cortex_state_to_text_loop(self):
        refined = torch.full((1, 4), .75)
        brain = SimpleNamespace(config=SimpleNamespace(vocab_size=261),
            decoder=SimpleNamespace(action_policy=lambda _hidden: torch.zeros(1, 8), internal_action_policy=lambda _hidden: torch.zeros(1, 8)),
            _select_structured_actions=lambda *_args, **_kwargs: ({}, [{"kind": "ponder", "confidence": .95, "arguments": {"organic": True}}]),
            _native_pre_speech_ponder=lambda *_args, **_kwargs: (refined, {"phase": "pre-speech"}))
        events = []
        callback = AdaptiveBrain._generation_activity_callback(brain, schemas=[], input_text="", assembly_ids=[],
            organic_state={}, action_cue=torch.zeros(1, 4), ledger=NativeActionEmissionLedger("turn"), seed=4,
            emit=lambda *event: events.append(event))
        directive = callback(torch.ones(1, 4), 3, .5)
        self.assertIs(directive["memoryBias"], refined)
        self.assertFalse(directive["stop"])
        self.assertEqual(events[0][1]["action"]["arguments"]["ponderTrace"]["phase"], "mid-generation")
        # The same learned kind at a later genuine decision is available;
        # replay of that same boundary must not compute or emit twice.
        second = callback(torch.ones(1, 4), 4, .5)
        self.assertIs(second["memoryBias"], refined)
        self.assertEqual(len(events), 2)
        self.assertNotEqual(events[0][1]["actionId"], events[1][1]["actionId"])
        self.assertNotIn("memoryBias", callback(torch.ones(1, 4), 4, .5))
        self.assertEqual(len(events), 2)


if __name__ == "__main__": unittest.main()
