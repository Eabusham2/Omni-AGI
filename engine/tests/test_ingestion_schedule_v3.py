"""Pure v3 ingestion contract tests; no model initialization or training."""

import copy
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ENGINE / "omni_core" / "ingestion_schedule_v3.py"
# Loading this file directly keeps the schema tests independent of the
# omni_core package initializer, which imports the model and torch.
SCHEMA_SPEC = importlib.util.spec_from_file_location(
    "omni_ingestion_schedule_v3_pure_test", SCHEMA_PATH
)
if SCHEMA_SPEC is None or SCHEMA_SPEC.loader is None:
    raise RuntimeError("v3 ingestion schema could not be loaded")
schema = importlib.util.module_from_spec(SCHEMA_SPEC)
SCHEMA_SPEC.loader.exec_module(schema)
EMPTY_RECORD_PREFIX_SHA256 = schema.EMPTY_RECORD_PREFIX_SHA256
checkpoint_binding_sha256 = schema.checkpoint_binding_sha256
make_checkpoint_binding_v3 = schema.make_checkpoint_binding_v3
make_ingestion_schedule_v3 = schema.make_ingestion_schedule_v3
make_v2_to_v3_boundary_plan = schema.make_v2_to_v3_boundary_plan
migration_plan_sha256 = schema.migration_plan_sha256
schedule_sha256 = schema.schedule_sha256
source_parser_manifest_sha256 = schema.source_parser_manifest_sha256
validate_checkpoint_binding_v3 = schema.validate_checkpoint_binding_v3
validate_ingestion_schedule_v3 = schema.validate_ingestion_schedule_v3
validate_unobserved_checkpoint_binding_v3 = (
    schema.validate_unobserved_checkpoint_binding_v3
)
validate_v2_to_v3_boundary_plan = schema.validate_v2_to_v3_boundary_plan


def sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def legacy_schedule():
    return {
        "format": "omni-ingestion-learning-schedule",
        "formatVersion": 2,
        "detailedRecordAssemblies": False,
        "physicalBatchRecords": 2,
        "gradientAccumulation": 4,
        "trainingSequenceTokens": 64,
        "checkpointRecords": 512,
        "corpusRepresentation": "shared-semantic-field-and-local-synapses",
        "slowGradientMode": "streaming-microbatch-gradient-accumulation",
        "localTypedTargetWindowPolicy": (
            "role-bounded-causal-exact-byte-windows-v1"
        ),
    }


class IngestionScheduleV3Tests(unittest.TestCase):
    def setUp(self):
        self.source = sha("source-manifest")
        self.content = sha("source-content")
        self.parser = sha("parser-manifest")
        self.schedule = make_ingestion_schedule_v3(
            source_manifest_sha256=self.source,
            source_content_sha256=self.content,
            parser_manifest_sha256=self.parser,
            physical_batch_records=2,
            gradient_accumulation=8,
            training_sequence_tokens=128,
            checkpoint_records=512,
            assembly_page_records=64,
        )
        self.schedule_hash = schedule_sha256(self.schedule)
        self.vector = {
            "generationId": sha("vector-generation-3"),
            "contentSha256": sha("vector-bytes-3"),
            "recordCount": 40,
        }
        self.index = {
            "generationId": sha("index-generation-3"),
            "contentSha256": sha("index-rows-3"),
            "recordCount": 40,
            "highWaterSequence": 45,
        }
        self.neural = sha("neural-generation-3")
        self.cursor = {
            "committedRecords": 40,
            "recordPrefixSha256": sha("record-prefix-40"),
        }
        self.coverage = {
            "visitedRecords": 43,
            "processedRecords": 40,
            "rejectedRecords": 3,
            "processedBytes": 4096,
            "expectedRecords": 100,
            "sourceStreamExhausted": False,
            "sourceContentReverifiedSha256": None,
        }

    def validate_schedule(self, value, digest=None, **overrides):
        options = {
            "source_manifest_sha256": self.source,
            "parser_manifest_sha256": self.parser,
            "source_content_sha256": self.content,
        }
        options.update(overrides)
        return validate_ingestion_schedule_v3(
            value, self.schedule_hash if digest is None else digest,
            **options,
        )

    def binding(self, *, coverage=None):
        return make_checkpoint_binding_v3(
            schedule=self.schedule,
            schedule_sha256_value=self.schedule_hash,
            neural_state_sha256=self.neural,
            checkpoint_sequence=3,
            cursor=self.cursor,
            coverage=self.coverage if coverage is None else coverage,
            vector_generation=self.vector,
            index_generation=self.index,
        )

    def validate_binding(self, value, digest=None, **overrides):
        options = {
            "schedule": self.schedule,
            "schedule_sha256_value": self.schedule_hash,
            "source_manifest_sha256": self.source,
            "parser_manifest_sha256": self.parser,
            "source_content_sha256": self.content,
            "neural_state_sha256": self.neural,
            "vector_generation": self.vector,
            "index_generation": self.index,
        }
        options.update(overrides)
        return validate_checkpoint_binding_v3(
            value, checkpoint_binding_sha256(value) if digest is None else digest,
            **options,
        )

    def test_schedule_separates_paged_detail_from_slow_gradient_batching(self):
        self.assertTrue(self.schedule["detailedRecordAssemblies"])
        self.assertEqual(
            self.schedule["corpusRepresentation"],
            "paged-detailed-assemblies",
        )
        self.assertEqual(
            self.schedule["slowGradientMode"],
            "streaming-microbatch-gradient-accumulation",
        )
        self.assertEqual(self.schedule["formatVersion"], 3)
        self.assertEqual(
            self.schedule["recordPrefixContract"], "omni-record-prefix-v1"
        )
        self.assertEqual(
            self.schedule["sourceParserManifestSha256"],
            source_parser_manifest_sha256(self.source, self.parser),
        )
        self.assertEqual(self.validate_schedule(self.schedule), self.schedule)
        self.assertEqual(
            self.schedule_hash,
            hashlib.sha256(json.dumps(
                self.schedule, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
        )

    def test_schedule_rejects_v2_and_unknown_payloads(self):
        with self.assertRaisesRegex(ValueError, "v3 schedule"):
            self.validate_schedule(legacy_schedule())
        with self.assertRaisesRegex(ValueError, "v3 schedule"):
            self.validate_schedule({**self.schedule, "sourceText": "private data"})
        with self.assertRaisesRegex(ValueError, "v3 schedule"):
            self.validate_schedule({**self.schedule, "tokenIds": [12, 45]})

    def test_schedule_rejects_mode_drift_and_bool_counts(self):
        for change in (
            {"corpusRepresentation": "shared-semantic-field-and-local-synapses"},
            {"slowGradientMode": "per-experience"},
            {"detailedRecordAssemblies": False},
            {"formatVersion": True},
            {"physicalBatchRecords": True},
            {"assemblyPageRecords": 4097},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate_schedule({**self.schedule, **change})

    def test_training_counts_have_signed_64_bounds_and_two_position_label_floor(self):
        large = {
            **self.schedule,
            "physicalBatchRecords": (1 << 63) - 1,
            "gradientAccumulation": (1 << 63) - 1,
            "trainingSequenceTokens": (1 << 63) - 1,
            "checkpointRecords": (1 << 63) - 1,
        }
        self.assertEqual(
            self.validate_schedule(large, schedule_sha256(large)), large
        )
        small = {**self.schedule, "trainingSequenceTokens": 2}
        self.assertEqual(
            self.validate_schedule(small, schedule_sha256(small)), small
        )
        one_position = {**self.schedule, "trainingSequenceTokens": 1}
        with self.assertRaises(ValueError):
            self.validate_schedule(one_position, schedule_sha256(one_position))
        for field in (
            "physicalBatchRecords", "gradientAccumulation",
            "trainingSequenceTokens", "checkpointRecords",
        ):
            for invalid in (0, -1, 1 << 63, True):
                altered = {**self.schedule, field: invalid}
                with self.subTest(field=field, invalid=invalid), self.assertRaises(
                    ValueError
                ):
                    self.validate_schedule(altered, schedule_sha256(altered))

    def test_schedule_rejects_resource_policy_relaxation(self):
        for field, value in (
            ("allowRecordSkipping", True),
            ("allowRepresentationDowngrade", 0),
            ("checkBeforeEveryRecord", 1),
            ("onPartialRecord", "continue-from-half-record"),
        ):
            modified = copy.deepcopy(self.schedule)
            modified["resourcePausePolicy"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "pause policy"
            ):
                self.validate_schedule(modified, schedule_sha256(modified))

    def test_schedule_rejects_manifest_rebinding_and_checksum_tamper(self):
        with self.assertRaisesRegex(ValueError, "source/parser"):
            self.validate_schedule(
                self.schedule, source_manifest_sha256=sha("other source")
            )
        modified = {**self.schedule, "checkpointRecords": 1024}
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.validate_schedule(modified)
        modified = {
            **self.schedule,
            "sourceParserManifestSha256": sha("wrong pair"),
        }
        with self.assertRaisesRegex(ValueError, "source/parser"):
            self.validate_schedule(modified, schedule_sha256(modified))

    def test_checkpoint_binds_coverage_cursor_and_observed_generations(self):
        binding = self.binding()
        self.assertEqual(self.validate_binding(binding), binding)
        self.assertEqual(
            validate_unobserved_checkpoint_binding_v3(
                binding,
                checkpoint_binding_sha256(binding),
                schedule=self.schedule,
                schedule_sha256_value=self.schedule_hash,
                source_manifest_sha256=self.source,
                parser_manifest_sha256=self.parser,
                source_content_sha256=self.content,
            ),
            binding,
        )
        self.assertEqual(binding["cursor"]["committedRecords"], 40)
        self.assertEqual(binding["coverage"]["visitedRecords"], 43)
        self.assertEqual(binding["vectorGeneration"], self.vector)
        self.assertEqual(binding["indexGeneration"], self.index)

    def test_checkpoint_rejects_stale_or_mixed_generations(self):
        binding = self.binding()
        for field, actual in (
            ("vector_generation", {**self.vector, "contentSha256": sha("new-vector")}),
            ("index_generation", {**self.index, "contentSha256": sha("new-index")}),
            ("neural_state_sha256", sha("new-neural")),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "generation mismatch"
            ):
                self.validate_binding(binding, **{field: actual})

    def test_checkpoint_rejects_tampered_cursor_coverage_and_payload(self):
        binding = self.binding()
        modified = copy.deepcopy(binding)
        modified["cursor"]["committedRecords"] = 44
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.validate_binding(modified)
        modified = copy.deepcopy(binding)
        modified["coverage"]["rejectedRecords"] = 2
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.validate_binding(modified)
        modified = copy.deepcopy(binding)
        modified["cursor"]["recordText"] = "do not retain"
        with self.assertRaisesRegex(ValueError, "cursor"):
            self.validate_binding(modified)
        modified = copy.deepcopy(binding)
        modified["coverage"]["tokenIds"] = [1]
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.validate_binding(modified)

    def test_empty_cursor_has_one_deterministic_prefix_digest(self):
        empty_cursor = {
            "committedRecords": 0,
            "recordPrefixSha256": EMPTY_RECORD_PREFIX_SHA256,
        }
        empty_coverage = {
            **self.coverage,
            "visitedRecords": 0,
            "processedRecords": 0,
            "rejectedRecords": 0,
            "processedBytes": 0,
        }
        binding = make_checkpoint_binding_v3(
            schedule=self.schedule,
            schedule_sha256_value=self.schedule_hash,
            neural_state_sha256=self.neural,
            checkpoint_sequence=1,
            cursor=empty_cursor,
            coverage=empty_coverage,
            vector_generation=self.vector,
            index_generation=self.index,
        )
        self.assertEqual(self.validate_binding(binding), binding)
        with self.assertRaisesRegex(ValueError, "empty.*prefix"):
            make_checkpoint_binding_v3(
                schedule=self.schedule,
                schedule_sha256_value=self.schedule_hash,
                neural_state_sha256=self.neural,
                checkpoint_sequence=1,
                cursor={**empty_cursor, "recordPrefixSha256": sha("wrong")},
                coverage=empty_coverage,
                vector_generation=self.vector,
                index_generation=self.index,
            )

    def test_checkpoint_rejects_hash_tamper_even_when_shape_is_valid(self):
        binding = self.binding()
        modified = copy.deepcopy(binding)
        modified["cursor"]["recordPrefixSha256"] = sha("other prefix")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.validate_binding(modified, checkpoint_binding_sha256(binding))

    def test_unobserved_parser_does_not_replace_generation_verification(self):
        binding = self.binding()
        observed = {**self.vector, "contentSha256": sha("disk changed")}
        # The pure reader can authenticate the header, but only the observed
        # validator is allowed to authorize resume against durable artifacts.
        self.assertEqual(
            validate_unobserved_checkpoint_binding_v3(
                binding, checkpoint_binding_sha256(binding),
                schedule=self.schedule,
                schedule_sha256_value=self.schedule_hash,
                source_manifest_sha256=self.source,
                parser_manifest_sha256=self.parser,
                source_content_sha256=self.content,
            ), binding,
        )
        with self.assertRaisesRegex(ValueError, "paged generation mismatch"):
            self.validate_binding(binding, vector_generation=observed)
        with self.assertRaisesRegex(ValueError, "source/parser"):
            validate_unobserved_checkpoint_binding_v3(
                binding, checkpoint_binding_sha256(binding),
                schedule=self.schedule,
                schedule_sha256_value=self.schedule_hash,
                source_manifest_sha256=sha("other manifest"),
                parser_manifest_sha256=self.parser,
                source_content_sha256=self.content,
            )

    def test_final_checkpoint_requires_full_coverage_and_reverified_content(self):
        complete = {
            **self.coverage,
            "visitedRecords": 100,
            "processedRecords": 97,
            "sourceStreamExhausted": True,
            "sourceContentReverifiedSha256": self.content,
        }
        binding = self.binding(coverage=complete)
        self.assertEqual(self.validate_binding(binding), binding)
        for change in (
            {"visitedRecords": 99, "processedRecords": 96},
            {"sourceContentReverifiedSha256": sha("wrong source")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.binding(coverage={**complete, **change})

    def test_v2_migration_requires_verified_completion_and_new_epoch(self):
        v2 = legacy_schedule()
        v2_hash = schedule_sha256(v2)
        args = {
            "v2_schedule": v2,
            "v2_schedule_sha256": v2_hash,
            "v2_completion_receipt_sha256": sha("v2 completed receipt"),
            "v2_source_content_sha256": self.content,
            "v2_completion_confirmed": True,
            "active_v2_checkpoint": False,
            "v2_epoch": 0,
            "v3_schedule": self.schedule,
            "v3_schedule_sha256": self.schedule_hash,
            "v3_epoch": 1,
        }
        plan = make_v2_to_v3_boundary_plan(**args)
        self.assertTrue(plan["resetCursorToZero"])
        self.assertTrue(plan["replayFullSourceInV3"])
        self.assertFalse(plan["reinterpretV2AsV3"])
        self.assertNotIn("committedRecords", plan)
        self.assertEqual(validate_v2_to_v3_boundary_plan(
            plan, migration_plan_sha256(plan),
            **{key: value for key, value in args.items() if key not in {"v2_epoch", "v3_epoch"}},
        ), plan)
        for change in (
            {"v2_completion_confirmed": False},
            {"active_v2_checkpoint": True},
            {"v3_epoch": 0},
            {"v2_source_content_sha256": sha("other source")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_v2_to_v3_boundary_plan(**{**args, **change})

    def test_v2_migration_rejects_protocol_drift_and_plan_tamper(self):
        v2 = legacy_schedule()
        v2_hash = schedule_sha256(v2)
        common = {
            "v2_schedule": v2,
            "v2_schedule_sha256": v2_hash,
            "v2_completion_receipt_sha256": sha("v2 receipt"),
            "v2_source_content_sha256": self.content,
            "v2_completion_confirmed": True,
            "active_v2_checkpoint": False,
            "v3_schedule": self.schedule,
            "v3_schedule_sha256": self.schedule_hash,
        }
        plan = make_v2_to_v3_boundary_plan(**common, v2_epoch=2, v3_epoch=3)
        forged = {**plan, "reinterpretV2AsV3": True}
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_v2_to_v3_boundary_plan(
                forged, migration_plan_sha256(forged), **common
            )
        forged = {**plan, "newTransactionRequired": 1}
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_v2_to_v3_boundary_plan(
                forged, migration_plan_sha256(forged), **common
            )
        with self.assertRaisesRegex(ValueError, "v2 schedule"):
            make_v2_to_v3_boundary_plan(
                **{**common, "v2_schedule": {**v2, "formatVersion": 3}},
                v2_epoch=2, v3_epoch=3,
            )


class BrainCheckpointReadPathSourceTests(unittest.TestCase):
    """Inspect wiring without importing the model or running a brain."""

    @staticmethod
    def method_source(source, name):
        start = re.search(
            r"^    def " + re.escape(name) + r"\(", source, re.MULTILINE
        )
        if start is None:
            raise AssertionError("brain method %s is missing" % name)
        next_method = re.search(
            r"^    (?:@|def |async def )", source[start.end():], re.MULTILINE
        )
        end = (
            start.end() + next_method.start()
            if next_method is not None else len(source)
        )
        return source[start.start():end]

    def test_v2_resume_stays_frozen_and_v3_commits_verified_shards(self):
        source = (ENGINE / "omni_core" / "brain.py").read_text(
            encoding="utf-8"
        )
        method = lambda name: self.method_source(source, name)
        self.assertIn("INGESTION_CHECKPOINT_VERSION = 2", source)
        self.assertIn("INGESTION_LEARNING_SCHEDULE_VERSION = 2", source)
        self.assertNotIn("Backward-safe internal stable-v1 loading", source)
        self.assertIn(
            '"formatVersion": INGESTION_LEARNING_SCHEDULE_VERSION',
            method("_ingestion_learning_schedule"),
        )
        self.assertIn(
            '3 if v3_checkpoint else INGESTION_CHECKPOINT_VERSION',
            method("ingest"),
        )
        reader = method("_validated_ingestion_checkpoints")
        self.assertLess(
            reader.index("_validated_paged_ingestion_checkpoint_v3"),
            reader.index("INGESTION_CHECKPOINT_VERSION"),
        )
        self.assertIn('validated[str(key)] = checkpoint', reader)
        self.assertIn("make_ingestion_schedule_v3", method("ingest"))
        self.assertIn("verify_source_snapshot(full_hash=True)", method("ingest"))
        self.assertIn("validate_checkpoint_binding_v3", method("ingest"))
        self.assertIn("recover_joint_generation", method("ingest"))
        self.assertIn("stage_joint_generation", method("_bind_ingestion_checkpoint_generation"))
        self.assertNotIn(
            "validate_ingestion_schedule_v3",
            method("_validated_ingestion_learning_schedule"),
        )
        v3 = method("_validated_paged_ingestion_checkpoint_v3")
        self.assertIn("validate_ingestion_schedule_v3", v3)
        self.assertIn("validate_unobserved_checkpoint_binding_v3", v3)
        self.assertNotIn("validate_checkpoint_binding_v3(", v3)


if __name__ == "__main__":
    unittest.main()
