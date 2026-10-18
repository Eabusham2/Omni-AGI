"""Operation declarations and private dispatch guards; no neural work."""
from contextvars import ContextVar

PERCEPTION_KINDS = ["native.perception.image-embedding", "native.perception.audio-embedding", "native.perception.video-embedding"]
CACHED_KINDS = ["native.decoder.cached-whole-step-sampling", "native.connected-decoder.cached-whole-step-sampling"]
MEDIA_KINDS = ["native.media.image-artifact", "native.media.audio-artifact", "native.media.video-artifact"]
TRAINING_KINDS = ["native.training.prepared-whole-wave"]
PREFILL_KINDS = ["native.decoder.full-prefix-sampling", "native.connected-decoder.full-prefix-sampling"]
FORWARD_KINDS = ["native.decoder.uncached-full-forward", "native.connected-decoder.uncached-full-forward"]
PERCEPTION_STAGES = ["operation-owner-admission", "input-activity-transfer", "complete-perception-embedding", "output-home-transfer", "original-owner-restore"]
CACHED_STAGES = ["frontier-cow-owner-admission", "input-cue-cache-activity-transfer", "complete-causal-workspace-conditioning",
    "all-decoder-regions-recurrent-exchange", "experts-norm-language-readout", "cache-neural-output-home-return",
    "original-owner-restore", "same-home-sampling"]
MEDIA_STAGES = ["prepared-media-owner-admission", "home-RNG-and-activity-transfer", "all-native-media-regions",
    "overlap-composition-and-preview", "deterministic-codec-artifact-checksum", "original-owner-restore"]
TRAINING_STAGES = ["prepared-wave-owner-and-policy-admission", "private-native-and-graph-COW",
    "same-home-RNG-input-and-activity-transfer", "all-region-forward-backward-and-native-update",
    "exact-complete-wave-state-and-effects-proof", "original-owner-and-RNG-fence"]
PREFILL_STAGES = ["prefix-owner-and-policy-admission", "input-cue-and-empty-cache-activity",
    "complete-prefix-causal-workspace", "all-decoder-regions-recurrent-exchange", "experts-norm-language-readout",
    "new-cache-neural-output-home-return", "original-owner-restore", "same-home-sampling"]
FORWARD_STAGES = ["forward-owner-and-policy-admission", "input-cue-mask-activity-transfer",
    "complete-causal-and-requested-whole-workspace", "all-decoder-connected-recurrent-exchange",
    "experts-norm-language-action-readout", "all-outputs-and-activity-home-return", "original-owner-restore"]
# Each declared consumer has its strict shared/main contract. An older review
# cannot acquire additional operation kinds without a fresh owner review.
STAGE_COVERAGE = {**{kind: PERCEPTION_STAGES for kind in PERCEPTION_KINDS}, **{kind: CACHED_STAGES for kind in CACHED_KINDS},
    **{kind: MEDIA_STAGES for kind in MEDIA_KINDS}, **{kind: TRAINING_STAGES for kind in TRAINING_KINDS},
    **{kind: PREFILL_STAGES for kind in PREFILL_KINDS}, **{kind: FORWARD_STAGES for kind in FORWARD_KINDS}}
PRIVATE_OPERATION = ContextVar("omni_private_complete_operation_body", default=False)


def coverage(work_kinds):
    if len(set(work_kinds)) != len(work_kinds) or any(kind not in STAGE_COVERAGE for kind in work_kinds):
        raise ValueError("unknown or duplicated current native Auto operation")
    return [{"workKind": kind, "stages": list(STAGE_COVERAGE[kind])} for kind in work_kinds]
