"""Pure v3 contract for paged, detailed record ingestion.

This module does not start an ingestion or migrate neural state.  In particular,
a v2 cursor is never read as v3.  The writer must compute the manifest and
committed-generation digests from durable artifacts, commit neural state,
paged vectors, the assembly index, and the cursor as one recoverable boundary,
then validate those same artifacts before resume.  Hashes here bind identities;
they do not replace that external durability/coverage verification.

Only fixed protocol names, counts, booleans, and SHA-256 identifiers can enter
these contracts.  Source text, token IDs, embeddings, and answer payloads do
not belong in an ingestion schedule or checkpoint binding.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional


SCHEDULE_FORMAT = "omni-ingestion-learning-schedule"
SCHEDULE_VERSION = 3
CHECKPOINT_FORMAT = "omni-paged-ingestion-checkpoint-binding"
CHECKPOINT_VERSION = 3
MIGRATION_FORMAT = "omni-ingestion-schedule-boundary-migration"
MIGRATION_VERSION = 1
PARSER_CONTRACT = "omni-dataset-record-stream-v1"
RECORD_PREFIX_CONTRACT = "omni-record-prefix-v1"
EMPTY_RECORD_PREFIX_SHA256 = hashlib.sha256(
    RECORD_PREFIX_CONTRACT.encode("ascii")
).hexdigest()
LOCAL_TYPED_TARGET_WINDOW_POLICY = "role-bounded-causal-exact-byte-windows-v1"
CORPUS_REPRESENTATION = "paged-detailed-assemblies"
SLOW_GRADIENT_MODE = "streaming-microbatch-gradient-accumulation"
PAUSE_POLICY = {
    "mode": "per-record-and-page-reserve-v1",
    "checkBeforeEveryRecord": True,
    "checkBeforeEveryPageWrite": True,
    "onPressure": "pause-without-cursor-advance",
    "onPartialRecord": "rollback-to-last-committed-generation",
    "allowRepresentationDowngrade": False,
    "allowRecordSkipping": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CONTRACT_BYTES = 8192
_MAX_COUNT = (1 << 63) - 1


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("ingestion contract is not finite JSON") from error
    if len(payload) > _MAX_CONTRACT_BYTES:
        raise ValueError("ingestion contract is too large")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, fields: set[str], label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("%s fields are invalid" % label)
    return dict(value)


def _schedule_identity(value: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("v3 schedule fields are invalid")
    return (
        value.get("sourceManifestSha256"),
        value.get("parserManifestSha256"),
        value.get("sourceContentSha256"),
    )


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("%s must be a lowercase SHA-256" % label)
    return value


def _count(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool) or not isinstance(value, int)
        or not minimum <= value <= _MAX_COUNT
    ):
        raise ValueError("%s count is invalid" % label)
    return value


def source_parser_manifest_sha256(
    source_manifest_sha256: str, parser_manifest_sha256: str,
) -> str:
    """Bind the two externally computed manifest digests in a fixed order."""

    return _digest({
        "sourceManifestSha256": _hash(source_manifest_sha256, "source manifest"),
        "parserManifestSha256": _hash(parser_manifest_sha256, "parser manifest"),
    })


def schedule_sha256(value: Mapping[str, Any]) -> str:
    """Canonical schedule digest; validation is still required on load."""

    return _digest(value)


def make_ingestion_schedule_v3(
    *,
    source_manifest_sha256: str,
    source_content_sha256: str,
    parser_manifest_sha256: str,
    physical_batch_records: int,
    gradient_accumulation: int,
    training_sequence_tokens: int,
    checkpoint_records: int,
    assembly_page_records: int,
) -> Dict[str, Any]:
    """Freeze independent representation and gradient choices for one source."""

    schedule = {
        "format": SCHEDULE_FORMAT,
        "formatVersion": SCHEDULE_VERSION,
        "sourceManifestSha256": source_manifest_sha256,
        "sourceContentSha256": source_content_sha256,
        "parserManifestSha256": parser_manifest_sha256,
        "sourceParserManifestSha256": source_parser_manifest_sha256(
            source_manifest_sha256, parser_manifest_sha256
        ),
        "parserContract": PARSER_CONTRACT,
        "recordPrefixContract": RECORD_PREFIX_CONTRACT,
        "detailedRecordAssemblies": True,
        "corpusRepresentation": CORPUS_REPRESENTATION,
        "slowGradientMode": SLOW_GRADIENT_MODE,
        "physicalBatchRecords": physical_batch_records,
        "gradientAccumulation": gradient_accumulation,
        "trainingSequenceTokens": training_sequence_tokens,
        "checkpointRecords": checkpoint_records,
        "assemblyPageRecords": assembly_page_records,
        "localTypedTargetWindowPolicy": LOCAL_TYPED_TARGET_WINDOW_POLICY,
        "resourcePausePolicy": dict(PAUSE_POLICY),
    }
    return validate_ingestion_schedule_v3(
        schedule, schedule_sha256(schedule),
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
    )


def validate_ingestion_schedule_v3(
    value: Any,
    expected_sha256: str,
    *,
    source_manifest_sha256: str,
    parser_manifest_sha256: str,
    source_content_sha256: str,
) -> Dict[str, Any]:
    """Fail closed on version, unknown fields, altered source, or mode drift."""

    schedule = _mapping(value, {
        "format", "formatVersion", "sourceManifestSha256",
        "sourceContentSha256", "parserManifestSha256",
        "sourceParserManifestSha256", "parserContract", "recordPrefixContract",
        "detailedRecordAssemblies", "corpusRepresentation",
        "slowGradientMode", "physicalBatchRecords", "gradientAccumulation",
        "trainingSequenceTokens", "checkpointRecords", "assemblyPageRecords",
        "localTypedTargetWindowPolicy", "resourcePausePolicy",
    }, "v3 schedule")
    if (
        schedule["format"] != SCHEDULE_FORMAT
        or type(schedule["formatVersion"]) is not int
        or schedule["formatVersion"] != SCHEDULE_VERSION
        or schedule["parserContract"] != PARSER_CONTRACT
        or schedule["recordPrefixContract"] != RECORD_PREFIX_CONTRACT
        or schedule["detailedRecordAssemblies"] is not True
        or schedule["corpusRepresentation"] != CORPUS_REPRESENTATION
        or schedule["slowGradientMode"] != SLOW_GRADIENT_MODE
        or schedule["localTypedTargetWindowPolicy"]
        != LOCAL_TYPED_TARGET_WINDOW_POLICY
    ):
        raise ValueError("v3 schedule protocol or modes are invalid")
    # These are protocol-valid counts, not resource-derived training caps.
    # The live resource policy decides whether a requested schedule can run.
    for field in (
        "physicalBatchRecords", "gradientAccumulation",
        "trainingSequenceTokens", "checkpointRecords",
    ):
        _count(schedule[field], field, minimum=1)
    page_records = _count(
        schedule["assemblyPageRecords"], "assemblyPageRecords", minimum=1
    )
    if page_records > 4096:
        raise ValueError("assemblyPageRecords page size is invalid")
    expected_source = _hash(source_manifest_sha256, "source manifest")
    expected_parser = _hash(parser_manifest_sha256, "parser manifest")
    expected_content = _hash(source_content_sha256, "source content")
    if (
        schedule["sourceManifestSha256"] != expected_source
        or schedule["parserManifestSha256"] != expected_parser
        or schedule["sourceContentSha256"] != expected_content
        or schedule["sourceParserManifestSha256"]
        != source_parser_manifest_sha256(expected_source, expected_parser)
    ):
        raise ValueError("v3 schedule source/parser manifest binding is invalid")
    pause_policy = _mapping(schedule["resourcePausePolicy"], set(PAUSE_POLICY), "pause policy")
    if (
        pause_policy != PAUSE_POLICY
        or any(type(pause_policy[field]) is not type(expected)
               for field, expected in PAUSE_POLICY.items())
    ):
        raise ValueError("v3 schedule per-record resource pause policy is invalid")
    if _digest(schedule) != _hash(expected_sha256, "schedule"):
        raise ValueError("v3 schedule checksum mismatch")
    schedule["resourcePausePolicy"] = dict(PAUSE_POLICY)
    return schedule


def checkpoint_binding_sha256(value: Mapping[str, Any]) -> str:
    """Canonical digest for a committed v3 cursor/generation binding."""

    return _digest(value)


def _validate_generation(value: Any, label: str, *, index: bool) -> Dict[str, Any]:
    fields = {"generationId", "contentSha256", "recordCount"}
    if index:
        fields |= {"highWaterSequence"}
    generation = _mapping(value, fields, label)
    _hash(generation["generationId"], label + " id")
    _hash(generation["contentSha256"], label + " checksum")
    _count(generation["recordCount"], label + " record")
    if index:
        # This identity must describe committed neural shards, not a mutable
        # SQLite cache instance. A cache rebuilt from the same committed
        # shards receives a new local store ID and must still validate the
        # cursor against the unchanged authoritative generation.
        high_water = _count(generation["highWaterSequence"], "index high-water")
        if high_water < generation["recordCount"]:
            raise ValueError("index high-water is behind record count")
    return generation


def _validate_checkpoint_shape(value: Any) -> Dict[str, Any]:
    binding = _mapping(value, {
        "format", "formatVersion", "scheduleSha256",
        "sourceManifestSha256", "parserManifestSha256",
        "sourceParserManifestSha256", "sourceContentSha256",
        "neuralStateSha256", "checkpointSequence", "cursor", "coverage",
        "vectorGeneration", "indexGeneration",
    }, "v3 checkpoint")
    if (
        binding["format"] != CHECKPOINT_FORMAT
        or type(binding["formatVersion"]) is not int
        or binding["formatVersion"] != CHECKPOINT_VERSION
    ):
        raise ValueError("v3 checkpoint version is invalid")
    for field in (
        "scheduleSha256", "sourceManifestSha256", "parserManifestSha256",
        "sourceParserManifestSha256", "sourceContentSha256",
        "neuralStateSha256",
    ):
        _hash(binding[field], field)
    _count(binding["checkpointSequence"], "checkpoint sequence", minimum=1)
    cursor = _mapping(
        binding["cursor"], {"committedRecords", "recordPrefixSha256"},
        "v3 checkpoint cursor",
    )
    _count(cursor["committedRecords"], "committed records")
    _hash(cursor["recordPrefixSha256"], "record prefix")
    if (
        cursor["committedRecords"] == 0
        and cursor["recordPrefixSha256"] != EMPTY_RECORD_PREFIX_SHA256
    ):
        raise ValueError("empty v3 checkpoint record prefix is invalid")
    coverage = _mapping(binding["coverage"], {
        "visitedRecords", "processedRecords", "rejectedRecords",
        "processedBytes", "expectedRecords", "sourceStreamExhausted",
        "sourceContentReverifiedSha256",
    }, "v3 checkpoint coverage")
    for field in ("visitedRecords", "processedRecords", "rejectedRecords", "processedBytes"):
        _count(coverage[field], field)
    expected = coverage["expectedRecords"]
    if expected is not None:
        _count(expected, "expected records")
    if type(coverage["sourceStreamExhausted"]) is not bool:
        raise ValueError("v3 checkpoint source exhaustion flag is invalid")
    if (
        coverage["visitedRecords"] != coverage["processedRecords"] + coverage["rejectedRecords"]
        or cursor["committedRecords"] > coverage["visitedRecords"]
        or (expected is not None and coverage["visitedRecords"] > expected)
    ):
        raise ValueError("v3 checkpoint record coverage is incomplete")
    if coverage["sourceStreamExhausted"]:
        if expected is not None and coverage["visitedRecords"] != expected:
            raise ValueError("v3 checkpoint final record coverage is incomplete")
        if coverage["sourceContentReverifiedSha256"] != binding["sourceContentSha256"]:
            raise ValueError("v3 checkpoint final source hash was not reverified")
    elif coverage["sourceContentReverifiedSha256"] is not None:
        raise ValueError("v3 checkpoint has premature final source verification")
    binding["cursor"] = cursor
    binding["coverage"] = coverage
    binding["vectorGeneration"] = _validate_generation(
        binding["vectorGeneration"], "vector generation", index=False
    )
    binding["indexGeneration"] = _validate_generation(
        binding["indexGeneration"], "index generation", index=True
    )
    return binding


def make_checkpoint_binding_v3(
    *, schedule: Mapping[str, Any], schedule_sha256_value: str,
    neural_state_sha256: str, checkpoint_sequence: int,
    cursor: Mapping[str, Any], coverage: Mapping[str, Any],
    vector_generation: Mapping[str, Any], index_generation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct a checkpoint header after the external atomic commit."""

    source_hash, parser_hash, content_hash = _schedule_identity(schedule)
    validated_schedule = validate_ingestion_schedule_v3(
        schedule, schedule_sha256_value,
        source_manifest_sha256=source_hash,
        parser_manifest_sha256=parser_hash,
        source_content_sha256=content_hash,
    )
    binding = {
        "format": CHECKPOINT_FORMAT,
        "formatVersion": CHECKPOINT_VERSION,
        "scheduleSha256": schedule_sha256_value,
        "sourceManifestSha256": validated_schedule["sourceManifestSha256"],
        "parserManifestSha256": validated_schedule["parserManifestSha256"],
        "sourceParserManifestSha256": validated_schedule["sourceParserManifestSha256"],
        "sourceContentSha256": validated_schedule["sourceContentSha256"],
        "neuralStateSha256": neural_state_sha256,
        "checkpointSequence": checkpoint_sequence,
        "cursor": dict(cursor),
        "coverage": dict(coverage),
        "vectorGeneration": dict(vector_generation),
        "indexGeneration": dict(index_generation),
    }
    return _validate_checkpoint_shape(binding)


def validate_unobserved_checkpoint_binding_v3(
    value: Any,
    expected_sha256: str,
    *, schedule: Mapping[str, Any], schedule_sha256_value: str,
    source_manifest_sha256: str, parser_manifest_sha256: str,
    source_content_sha256: str,
) -> Dict[str, Any]:
    """Parse a persisted v3 binding without authorizing resume.

    This verifies the closed schema, source/schedule binding, and checksum.
    It deliberately does not claim the stored paged generations still match
    durable vector/index artifacts.  Resume must call the observed validator.
    """

    validated_schedule = validate_ingestion_schedule_v3(
        schedule, schedule_sha256_value,
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
    )
    binding = _validate_checkpoint_shape(value)
    for field in (
        "sourceManifestSha256", "parserManifestSha256",
        "sourceParserManifestSha256", "sourceContentSha256",
    ):
        if binding[field] != validated_schedule[field]:
            raise ValueError("v3 checkpoint source/parser binding mismatch")
    if binding["scheduleSha256"] != schedule_sha256_value:
        raise ValueError("v3 checkpoint schedule mismatch")
    if _digest(binding) != _hash(expected_sha256, "checkpoint"):
        raise ValueError("v3 checkpoint checksum mismatch")
    return binding


def validate_checkpoint_binding_v3(
    value: Any,
    expected_sha256: str,
    *, schedule: Mapping[str, Any], schedule_sha256_value: str,
    source_manifest_sha256: str, parser_manifest_sha256: str,
    source_content_sha256: str, neural_state_sha256: str,
    vector_generation: Mapping[str, Any], index_generation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a resume cursor against freshly observed committed artifacts."""

    binding = validate_unobserved_checkpoint_binding_v3(
        value, expected_sha256, schedule=schedule,
        schedule_sha256_value=schedule_sha256_value,
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
    )
    if binding["neuralStateSha256"] != _hash(neural_state_sha256, "neural state"):
        raise ValueError("v3 checkpoint neural generation mismatch")
    observed_vectors = _validate_generation(vector_generation, "observed vector generation", index=False)
    observed_index = _validate_generation(index_generation, "observed index generation", index=True)
    if binding["vectorGeneration"] != observed_vectors or binding["indexGeneration"] != observed_index:
        raise ValueError("v3 checkpoint paged generation mismatch")
    return binding


def _validate_v2_schedule_for_migration(value: Any, expected_sha256: str) -> Dict[str, Any]:
    """Recognize the exact legacy contract, without upgrading its meaning."""

    schedule = _mapping(value, {
        "format", "formatVersion", "detailedRecordAssemblies",
        "physicalBatchRecords", "gradientAccumulation", "trainingSequenceTokens",
        "checkpointRecords", "corpusRepresentation", "slowGradientMode",
        "localTypedTargetWindowPolicy",
    }, "v2 schedule")
    if (
        schedule["format"] != SCHEDULE_FORMAT
        or type(schedule["formatVersion"]) is not int
        or schedule["formatVersion"] != 2
        or type(schedule["detailedRecordAssemblies"]) is not bool
        or schedule["localTypedTargetWindowPolicy"] != LOCAL_TYPED_TARGET_WINDOW_POLICY
    ):
        raise ValueError("v2 schedule protocol is invalid")
    detailed = schedule["detailedRecordAssemblies"]
    if (
        schedule["corpusRepresentation"] != (
            "detailed-distributed-assemblies" if detailed
            else "shared-semantic-field-and-local-synapses"
        )
        or schedule["slowGradientMode"] != (
            "per-experience" if detailed
            else SLOW_GRADIENT_MODE
        )
    ):
        raise ValueError("v2 schedule modes are invalid")
    for field, minimum in (
        ("physicalBatchRecords", 1), ("gradientAccumulation", 1),
        ("trainingSequenceTokens", 8), ("checkpointRecords", 1),
    ):
        _count(schedule[field], field, minimum=minimum)
    if _digest(schedule) != _hash(expected_sha256, "v2 schedule"):
        raise ValueError("v2 schedule checksum mismatch")
    return schedule


def make_v2_to_v3_boundary_plan(
    *, v2_schedule: Mapping[str, Any], v2_schedule_sha256: str,
    v2_completion_receipt_sha256: str, v2_source_content_sha256: str,
    v2_completion_confirmed: bool, active_v2_checkpoint: bool, v2_epoch: int,
    v3_schedule: Mapping[str, Any], v3_schedule_sha256: str, v3_epoch: int,
) -> Dict[str, Any]:
    """Plan a *new epoch* after a verified v2 completion receipt.

    The caller must authenticate the completed receipt against durable brain
    state and prove no active v2 checkpoint remains.  This plan never carries
    a v2 cursor, asserts v2 records were detailed, or mutates neural state.
    """

    _validate_v2_schedule_for_migration(v2_schedule, v2_schedule_sha256)
    _hash(v2_completion_receipt_sha256, "v2 completion receipt")
    _hash(v2_source_content_sha256, "v2 source content")
    if v2_completion_confirmed is not True or active_v2_checkpoint is not False:
        raise ValueError("v2 must be completed with no active checkpoint")
    source_hash, parser_hash, content_hash = _schedule_identity(v3_schedule)
    validated_v3 = validate_ingestion_schedule_v3(
        v3_schedule, v3_schedule_sha256,
        source_manifest_sha256=source_hash,
        parser_manifest_sha256=parser_hash,
        source_content_sha256=content_hash,
    )
    if v2_source_content_sha256 != validated_v3["sourceContentSha256"]:
        raise ValueError("v2/v3 migration source content differs")
    _count(v2_epoch, "v2 epoch")
    _count(v3_epoch, "v3 epoch")
    if v3_epoch <= v2_epoch:
        raise ValueError("v3 migration requires a new epoch")
    return {
        "format": MIGRATION_FORMAT,
        "formatVersion": MIGRATION_VERSION,
        "boundary": "completed-v2-transaction-then-new-v3-epoch",
        "v2ScheduleSha256": v2_schedule_sha256,
        "v2CompletionReceiptSha256": v2_completion_receipt_sha256,
        "v2SourceContentSha256": v2_source_content_sha256,
        "v2Epoch": v2_epoch,
        "v3ScheduleSha256": v3_schedule_sha256,
        "v3Epoch": v3_epoch,
        "sourceManifestSha256": validated_v3["sourceManifestSha256"],
        "parserManifestSha256": validated_v3["parserManifestSha256"],
        "sourceContentSha256": validated_v3["sourceContentSha256"],
        "newTransactionRequired": True,
        "resetCursorToZero": True,
        "replayFullSourceInV3": True,
        "reinterpretV2AsV3": False,
    }


def migration_plan_sha256(value: Mapping[str, Any]) -> str:
    return _digest(value)


def validate_v2_to_v3_boundary_plan(
    value: Any, expected_sha256: str, *,
    v2_schedule: Mapping[str, Any], v2_schedule_sha256: str,
    v2_completion_receipt_sha256: str, v2_source_content_sha256: str,
    v2_completion_confirmed: bool, active_v2_checkpoint: bool,
    v3_schedule: Mapping[str, Any],
    v3_schedule_sha256: str,
) -> Dict[str, Any]:
    """Validate an explicit migration plan before any new v3 transaction."""

    plan = _mapping(value, {
        "format", "formatVersion", "boundary", "v2ScheduleSha256",
        "v2CompletionReceiptSha256", "v2SourceContentSha256",
        "v2Epoch", "v3ScheduleSha256",
        "v3Epoch", "sourceManifestSha256", "parserManifestSha256",
        "sourceContentSha256", "newTransactionRequired", "resetCursorToZero",
        "replayFullSourceInV3", "reinterpretV2AsV3",
    }, "v2-to-v3 migration plan")
    expected = make_v2_to_v3_boundary_plan(
        v2_schedule=v2_schedule,
        v2_schedule_sha256=v2_schedule_sha256,
        v2_completion_receipt_sha256=v2_completion_receipt_sha256,
        v2_source_content_sha256=v2_source_content_sha256,
        v2_completion_confirmed=v2_completion_confirmed,
        active_v2_checkpoint=active_v2_checkpoint,
        v2_epoch=plan["v2Epoch"], v3_schedule=v3_schedule,
        v3_schedule_sha256=v3_schedule_sha256, v3_epoch=plan["v3Epoch"],
    )
    if plan != expected or any(
        type(plan[field]) is not type(expected[field]) for field in expected
    ):
        raise ValueError("v2-to-v3 migration boundary is invalid")
    if _digest(plan) != _hash(expected_sha256, "migration plan"):
        raise ValueError("v2-to-v3 migration plan checksum mismatch")
    return plan
