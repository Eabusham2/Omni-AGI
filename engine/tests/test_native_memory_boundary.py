"""Cheap production-boundary checks that do not construct or train a brain."""

import inspect
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from omni_core.brain import AdaptiveBrain, ENGINE_SCHEMA_VERSION
from omni_core.distributed_training import DynamicNeuralUpdate


class NativeMemoryBoundaryTests(unittest.TestCase):
    def test_native_brain_exposes_no_cue_to_answer_bank(self):
        self.assertFalse(hasattr(AdaptiveBrain, "_learn_sequence_associations"))
        self.assertNotIn("sequence_memory", inspect.getsource(AdaptiveBrain.__init__))
        self.assertNotIn("sequence_memory", inspect.getsource(AdaptiveBrain.learn_experience))

    def test_distributed_update_has_no_answer_summary(self):
        self.assertNotIn(
            "response_summary", {field.name for field in fields(DynamicNeuralUpdate)}
        )

    def test_legacy_origin_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            engine.mkdir()
            metadata = {
                "schema_version": ENGINE_SCHEMA_VERSION,
                "release_format": "stable-1.0",
                "brain_id": "old-brain",
                "config": {
                    "origin_kind": "starter",
                },
            }
            (engine / "brain.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only native ground-up"):
                AdaptiveBrain.load(Path(directory))

    def test_cue_to_answer_metadata_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            engine.mkdir()
            metadata = {
                "schema_version": ENGINE_SCHEMA_VERSION,
                "release_format": "stable-1.0",
                "brain_id": "old-brain",
                "config": {
                    "origin_kind": "ground-up",
                },
                "neural_sequence_memory": {"entries": 1},
            }
            (engine / "brain.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obsolete cue-to-answer"):
                AdaptiveBrain.load(Path(directory))


if __name__ == "__main__":
    unittest.main()
