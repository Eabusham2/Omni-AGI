"""Explicit live chat values needed by the unchanged commit/finalization path."""

FINAL_FIELDS = (
    "action_scores", "action_state", "admitted_human_message", "admitted_input",
    "before_checksum", "before_parameters", "best_work_trace", "candidates", "clean", "cue",
    "decision_prediction_loss", "decoder_training_before_generation", "defer_slow_learning",
    "entropies", "experience", "expert_route", "generated", "generation_budget",
    "generation_started_at", "generation_tokens", "input_sha256", "internal_action_logits",
    "language_action_logits", "native_ponder_trace", "normalized_tools", "pending_current_context",
    "ponder_activity_passes", "ponder_activity_vector", "ponder_factors", "ponder_steps",
    "prior_replay", "private_generation_steered", "prompt_ids", "prompt_token_hash", "proposed_actions",
    "recall_audit", "recalled", "recalled_vector", "recent_prompt_tokens", "response_boundary_checksum",
    "seed", "selected_branch", "single_pass_generation", "temporary_input_audit", "tool_model",
    "transactional_generation", "turn_id", "working_memory_used", "working_model",
)


def final_context(values):
    return {name: values.get(name) for name in FINAL_FIELDS}
