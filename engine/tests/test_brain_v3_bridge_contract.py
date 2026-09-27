"""Pure v3 ingestion contract checks; no brain/model is constructed."""

import hashlib
import inspect
import unittest

from omni_core.brain import AdaptiveBrain, INGESTION_CHECKPOINT_FORMAT
from omni_core.ingestion_schedule_v3 import (
    checkpoint_binding_sha256,
    make_checkpoint_binding_v3,
    make_ingestion_schedule_v3,
    schedule_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class BrainV3BridgeContractTests(unittest.TestCase):
    def test_source_and_parser_manifests_bind_observed_identity(self) -> None:
        args = {
            "content_hash": _sha("actual bytes"),
            "source_bytes": 12,
            "resolved_kind": "jsonl",
            "source_name_hash": _sha("dataset.jsonl"),
            "record_count_hint": 1,
        }
        first = AdaptiveBrain._ingestion_v3_manifest_hashes(**args)
        self.assertEqual(first, AdaptiveBrain._ingestion_v3_manifest_hashes(**args))
        self.assertNotEqual(first[0], AdaptiveBrain._ingestion_v3_manifest_hashes(
            **{**args, "source_bytes": 13}
        )[0])
        self.assertNotEqual(first[1], AdaptiveBrain._ingestion_v3_manifest_hashes(
            **{**args, "resolved_kind": "csv"}
        )[1])

    def test_generation_binding_uses_committed_shard_hash_not_sqlite_id(self) -> None:
        generation = _sha("substrate generation")
        vectors, index = AdaptiveBrain._v3_committed_substrate_generations({
            "format": "omni-substrate-shards", "formatVersion": 3,
            "activeGeneration": generation, "contentSha256": generation,
            "counts": {"neurons": 23, "assemblies": 7, "synapses": 41},
        })
        self.assertEqual(vectors, {
            "generationId": generation, "contentSha256": generation,
            "recordCount": 23,
        })
        self.assertEqual(index["generationId"], generation)
        self.assertEqual(index["highWaterSequence"], 7)
        with self.assertRaises(ValueError):
            AdaptiveBrain._v3_committed_substrate_generations({
                "format": "omni-substrate-shards", "formatVersion": 2,
                "activeGeneration": generation, "contentSha256": generation,
                "counts": {"neurons": 23, "assemblies": 7, "synapses": 41},
            })

    def test_v3_cursor_accepts_closed_hash_count_schema_and_rejects_raw_text(self) -> None:
        content = _sha("source")
        source, parser = AdaptiveBrain._ingestion_v3_manifest_hashes(
            content_hash=content, source_bytes=10, resolved_kind="jsonl",
            source_name_hash=_sha("file"), record_count_hint=1,
        )
        schedule = make_ingestion_schedule_v3(
            source_manifest_sha256=source, source_content_sha256=content,
            parser_manifest_sha256=parser, physical_batch_records=1,
            gradient_accumulation=2, training_sequence_tokens=128,
            checkpoint_records=1, assembly_page_records=128,
        )
        schedule_hash = schedule_sha256(schedule)
        generation = _sha("generation")
        vectors = {"generationId": generation, "contentSha256": generation,
                   "recordCount": 3}
        index = {"generationId": generation, "contentSha256": generation,
                 "recordCount": 1, "highWaterSequence": 1}
        coverage = {
            "visitedRecords": 1, "processedRecords": 1,
            "rejectedRecords": 0, "processedBytes": 10,
            "expectedRecords": 1, "sourceStreamExhausted": False,
            "sourceContentReverifiedSha256": None,
        }
        binding = make_checkpoint_binding_v3(
            schedule=schedule, schedule_sha256_value=schedule_hash,
            neural_state_sha256=_sha("weights"), checkpoint_sequence=1,
            cursor={"committedRecords": 1, "recordPrefixSha256": _sha("record")},
            coverage=coverage, vector_generation=vectors, index_generation=index,
        )
        key = _sha("source identity")
        baseline = {
            "parameterChecksum": _sha("before"),
            "concepts": 0, "ideas": 0, "plasticityEvents": 0,
            "memoryNeurons": 0, "memorySynapses": 0,
            "memorySynapticUses": 0, "trainingSteps": 0,
            "statisticalExperiences": 0,
        }
        coverage_snapshot = {
            "discoveredFiles": 1, "completedFiles": 0,
            "processedFiles": 0, "rejectedFiles": 0,
            "discoveredRecords": 1, "processedRecords": 1,
            "rejectedRecords": 0, "processedBytes": 10,
            "shards": 0, "modalityCounts": {"text": 1},
            "errors": [], "errorCount": 0,
            "errorsTruncated": False, "complete": False,
        }
        checkpoint = {
            "format": INGESTION_CHECKPOINT_FORMAT, "formatVersion": 3,
            "parserContract": "omni-dataset-record-stream-v1",
            "status": "active", "sourceIdentity": key,
            "transactionId": _sha("transaction"), "contentHash": content,
            "sourceNameHash": _sha("file"),
            "neuralStateChecksum": _sha("weights"),
            "recordPrefixSha256": _sha("record"),
            "sourceSnapshot": {"device": 1, "inode": 2, "size": 10, "mtimeNs": 3},
            "sourceBytes": 10, "resolvedKind": "jsonl",
            "policy": "encode", "epoch": 0, "committedRecords": 1,
            "visitedRecords": 1, "processedRecords": 1,
            "rejectedRecords": 0, "processedBytes": 10,
            "commitSequence": 1, "coverageAtCommit": coverage_snapshot,
            "learningSchedule": schedule, "learningScheduleSha256": schedule_hash,
            "baseline": baseline,
            "aggregate": {
                "lossTotal": 0.1, "learnedChunks": 1,
                "readingReportCount": 1, "streamingGradientRecords": 1,
                "streamingGradientOptimizerSteps": 1,
                "mediaAccumulator": AdaptiveBrain._empty_media_accumulator(),
            },
            "capabilityRehearsal": None, "capabilityRehearsalCadence": None,
            "committedAt": "2026-01-01T00:00:00Z",
            "sourceManifestSha256": source, "parserManifestSha256": parser,
            "expectedRecords": 1,
            "pagedCheckpointBinding": binding,
            "pagedCheckpointBindingSha256": checkpoint_binding_sha256(binding),
        }
        self.assertEqual(
            AdaptiveBrain._validated_ingestion_checkpoints({key: checkpoint})[key],
            checkpoint,
        )
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            AdaptiveBrain._validated_ingestion_checkpoints({
                key: {**checkpoint, "sourceText": "private passage"}
            })

    def test_first_chat_is_not_gated_by_build_time_tool_quiz(self) -> None:
        curriculum = inspect.getsource(AdaptiveBrain._train_ground_up_curriculum)
        self.assertNotIn("rehearse_public_capability_routes(", curriculum)
        action_training = inspect.getsource(AdaptiveBrain._train_starter_action_policy)
        self.assertIn("strict=False", action_training)


if __name__ == "__main__":
    unittest.main()
