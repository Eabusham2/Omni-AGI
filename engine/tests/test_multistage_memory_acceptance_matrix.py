import re
import unittest
from dataclasses import replace

if __package__:
    from .memory_acceptance_harness import (
        GeneratedAnswerEvidence,
        PATH_FIXTURES,
        PATH_SPECS,
        PathObservation,
        assess_path_observation,
    )
else:
    # `unittest discover -s engine/tests` imports test files as top-level
    # modules, while package-mode runs give this module an engine.tests parent.
    from memory_acceptance_harness import (
        GeneratedAnswerEvidence,
        PATH_FIXTURES,
        PATH_SPECS,
        PathObservation,
        assess_path_observation,
    )


def observation(
    path,
    *,
    representation="detailed",
    segment_count=3,
    exact_eligible_segments=None,
    query_contains_learned_cue=True,
    typed_target_present=False,
    answer=None,
):
    return PathObservation(
        path=path,
        representation=representation,
        segment_count=segment_count,
        query_contains_learned_cue=query_contains_learned_cue,
        typed_target_present=typed_target_present,
        attributable_source=True,
        accepted_for_learning=True,
        coverage_complete=(
            True if PATH_SPECS[path].coverage_required else None
        ),
        grammar_specific_gate_used=False,
        statistical_updates=max(1, segment_count),
        semantic_target_present=True,
        semantic_target_score=0.72,
        semantic_control_score=0.11,
        statistical_forward_precision="exact ternary {-1,0,+1}",
        recent_dialogue_tokens=0,
        prior_prompt_text_expanded=False,
        raw_source_text_injected=False,
        exact_eligible_segments=(
            max(1, segment_count - 1)
            if exact_eligible_segments is None
            else exact_eligible_segments
        ),
        answer=answer,
    )


def generated_answer(text="The amber ferry departs from Pier Nine."):
    return GeneratedAnswerEvidence(
        expected_text=text,
        generated_text=text,
        backend="mutable-omni-decoder",
        generated_token_count=len(text),
        answer_quality_passed=True,
        fresh_attention_committed=True,
        fresh_attention_epoch=1,
        worker_restarted=True,
        cortical_checksum_before_learning="a" * 64,
        cortical_checksum_after_learning="b" * 64,
        cortical_checksum_after_restart="b" * 64,
        training_steps_before=0,
        training_steps_after=1,
        raw_text_retrieved=False,
        raw_token_ids_retrieved=False,
    )


class MultiStageMemoryAcceptanceMatrixTests(unittest.TestCase):
    """Synthetic decision-rule tests; live answer quality is gated elsewhere."""

    def test_matrix_covers_every_requested_learning_path(self):
        self.assertEqual(
            set(PATH_SPECS),
            {
                "messy-chat",
                "direct-learn-experience",
                "typed-jsonl",
                "plain-file",
                "parquet-text-row",
                "web-crawl",
            },
        )
        self.assertEqual(PATH_SPECS["typed-jsonl"].exact_mode, "typed-target")
        self.assertTrue(PATH_SPECS["web-crawl"].coverage_required)
        self.assertEqual(set(PATH_FIXTURES), set(PATH_SPECS))
        self.assertTrue(
            all(
                "grammar" not in spec.exact_condition
                for spec in PATH_SPECS.values()
            )
        )
        encoded = " ".join(
            fixture.experience.casefold() for fixture in PATH_FIXTURES.values()
        )
        for prohibited_template in (
            "the code is",
            "remember this fact",
            "reply only with the code",
        ):
            self.assertNotIn(prohibited_template, encoded)

    def test_non_typed_exact_fixtures_use_only_generic_adjacency(self):
        for path, fixture in PATH_FIXTURES.items():
            if path == "typed-jsonl":
                self.assertEqual(
                    fixture.expected_exact_text, fixture.declared_response
                )
                continue
            with self.subTest(path=path):
                segments = [
                    value.strip()
                    for value in re.split(
                        r"(?<=[.!?])\s+|\n+", fixture.experience
                    )
                    if value.strip()
                ]
                learned_cue = next(
                    value
                    for value in segments
                    if value in fixture.query
                )
                cue_index = segments.index(learned_cue)
                self.assertGreater(cue_index, 0)
                self.assertEqual(
                    segments[cue_index - 1], fixture.expected_exact_text
                )

    def test_detailed_adjacent_episode_requires_exact_generated_content(self):
        for path in (
            "messy-chat",
            "direct-learn-experience",
            "plain-file",
            "parquet-text-row",
            "web-crawl",
        ):
            with self.subTest(path=path):
                result = assess_path_observation(
                    observation(path, answer=generated_answer())
                )
                self.assertTrue(result.passed, result.failures)
                self.assertEqual(result.expected_level, "exact-generated")

    def test_typed_target_is_exact_only_when_resource_plan_retains_it(self):
        detailed = assess_path_observation(
            observation(
                "typed-jsonl",
                typed_target_present=True,
                answer=generated_answer("The taught assistant response."),
            )
        )
        self.assertTrue(detailed.passed, detailed.failures)
        self.assertEqual(detailed.expected_level, "exact-generated")

        compact = assess_path_observation(
            observation(
                "typed-jsonl",
                representation="statistical",
                typed_target_present=True,
                answer=generated_answer("The taught assistant response."),
            )
        )
        self.assertTrue(compact.passed, compact.failures)
        self.assertEqual(compact.expected_level, "semantic-generated")

    def test_question_only_or_paraphrased_cue_has_semantic_acceptance(self):
        question_only = assess_path_observation(
            observation(
                "plain-file",
                segment_count=1,
                exact_eligible_segments=0,
                answer=generated_answer(),
            )
        )
        paraphrase = assess_path_observation(
            observation(
                "messy-chat",
                query_contains_learned_cue=False,
                answer=generated_answer(),
            )
        )
        self.assertTrue(question_only.passed, question_only.failures)
        self.assertTrue(paraphrase.passed, paraphrase.failures)
        self.assertEqual(question_only.expected_level, "semantic-generated")
        self.assertEqual(paraphrase.expected_level, "semantic-generated")

    def test_harness_rejects_context_leak_and_grammar_gate(self):
        base = observation(
            "parquet-text-row",
            representation="statistical",
            answer=generated_answer(),
        )
        compromised = PathObservation(
            **{
                **base.__dict__,
                "grammar_specific_gate_used": True,
                "recent_dialogue_tokens": 4,
                "raw_source_text_injected": True,
            }
        )
        result = assess_path_observation(compromised)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("grammar-specific" in value for value in result.failures)
        )
        self.assertTrue(
            any("prior recent-dialogue" in value for value in result.failures)
        )
        self.assertTrue(
            any("raw source text" in value for value in result.failures)
        )

    def test_semantic_metric_must_beat_an_unrelated_control(self):
        base = observation(
            "web-crawl", segment_count=1, answer=generated_answer()
        )
        failed = PathObservation(
            **{
                **base.__dict__,
                "semantic_target_score": 0.05,
                "semantic_control_score": 0.20,
            }
        )
        result = assess_path_observation(failed)
        self.assertFalse(result.passed)
        self.assertIn(
            "semantic target did not outrank the unrelated control",
            result.failures,
        )

    def test_non_neural_lookup_cannot_pass_as_generated_answer(self):
        replay = replace(
            generated_answer(),
            backend="lookup-index",
        )
        result = assess_path_observation(
            observation("plain-file", answer=replay)
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "answer did not come from a model generation backend",
            result.failures,
        )
        missing_generation = assess_path_observation(observation("plain-file"))
        self.assertFalse(missing_generation.passed)
        self.assertIn(
            "no model-generated answer after Fresh and worker restart",
            missing_generation.failures,
        )

    def test_generated_answer_requires_fresh_restart_and_persisted_weight_change(self):
        base = generated_answer()
        invalid = replace(
            base,
            fresh_attention_committed=False,
            fresh_attention_epoch=0,
            worker_restarted=False,
            cortical_checksum_after_learning=base.cortical_checksum_before_learning,
            cortical_checksum_after_restart="c" * 64,
            training_steps_after=base.training_steps_before,
        )
        result = assess_path_observation(
            observation("typed-jsonl", typed_target_present=True, answer=invalid)
        )
        self.assertFalse(result.passed)
        self.assertIn("Fresh attention did not commit before the answer", result.failures)
        self.assertIn("answer was not generated after a worker restart", result.failures)
        self.assertIn("learning did not change cortical weights", result.failures)
        self.assertIn("no cortical training step was committed", result.failures)

        not_persisted = replace(base, cortical_checksum_after_restart="c" * 64)
        result = assess_path_observation(
            observation("plain-file", answer=not_persisted)
        )
        self.assertIn(
            "learned cortical weights did not survive restart", result.failures
        )

    def test_generated_answer_keeps_exact_quality_gate(self):
        wrong = replace(
            generated_answer(),
            generated_text="A different answer.",
            answer_quality_passed=False,
        )
        result = assess_path_observation(
            observation("plain-file", answer=wrong)
        )
        self.assertFalse(result.passed)
        self.assertIn("generated response was not an exact content match", result.failures)
        self.assertIn("generated answer failed its answer-quality check", result.failures)


if __name__ == "__main__":
    unittest.main()
