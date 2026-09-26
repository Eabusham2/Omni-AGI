import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import distributed_train
from distributed_train import _resolve_initial_ground_up_locator
from omni_core.config import OmniConfig
from omni_core.ground_up import (
    current_ground_up_curriculum_manifest,
    ground_up_curriculum_manifest,
)


class DistributedCliOriginTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-distributed-cli-origin-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _origin_metadata(curriculum=None):
        config = OmniConfig.micro(
            origin_kind="ground-up",
        ).to_dict()
        before = "a" * 64
        after = "b" * 64
        selected_curriculum = curriculum or ground_up_curriculum_manifest()
        manifest = {
            **selected_curriculum,
            "originKind": "ground-up",
            "externalWeightFiles": [],
            "pretrainedTextCortex": None,
            "baseFrozen": False,
            "randomInitialization": {
                "algorithm": "torch-seeded-module-initialization-v1",
                "seed": config["seed"],
                "parameterChecksum": before,
                "exactParameterCount": 143_064,
            },
            "architectureScale": {
                "hardwareTier": config["hardware_tier"],
                "dimensions": config["d_model"],
                "layers": config["n_layers"],
                "denseParameterCount": 143_064,
            },
            "trainingReceipt": {
                "format": "omni-ground-up-training-receipt",
                "formatVersion": (
                    2 if selected_curriculum.get("formatVersion") == 3 else 1
                ),
                "completeCoverage": True,
                "parametersChanged": True,
                "baseFrozen": False,
                "parameterChecksumBefore": before,
                "parameterChecksumAfter": after,
            },
        }
        return {
            "config": config,
            "ground_up_training_manifest": manifest,
            "packed_ternary_manifest": {
                "originKind": "ground-up",
                "pretrainedTextCortex": None,
                "baseFrozen": False,
                "contentSha256": "c" * 64,
                "parameterChecksum": after,
            },
            "conversation": {
                "totalEntries": 0,
                "messageCount": 0,
                "actionCount": 0,
                "traceCount": 0,
            },
            "training_sources": [],
            "ingestion_checkpoints": {},
            "completed_ingestions": [],
            "completed_chat_turns": [],
            "workspace_items": [],
            "installed_modality_packs": [],
            "counters": {"inference_count": 0},
        }

    def _brain_locator(self):
        live = self.root / "live-brain"
        engine = live / "engine"
        origin = engine / "origin"
        origin.mkdir(parents=True)
        # The mutable checkpoint is deliberately mature. It is only a locator;
        # none of these private fields may be used by distributed training.
        (engine / "brain.json").write_text(
            json.dumps(
                {
                    "training_sources": [{"name": "private-user-data"}],
                    "conversation": {"messageCount": 7, "traceCount": 3},
                }
            ),
            encoding="utf-8",
        )
        metadata = self._origin_metadata()
        (origin / "brain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        for relative in (
            "core.safetensors",
            "plasticity.safetensors",
            "substrate/manifest.json",
            "state/manifest.json",
        ):
            path = origin / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        packed_manifest = origin / "packed-ternary" / "manifest.json"
        packed_manifest.parent.mkdir(parents=True, exist_ok=True)
        packed_manifest.write_bytes(b'{"format":"omni-packed-ternary"}')
        (packed_manifest.parent / "manifest.sha256").write_text(
            hashlib.sha256(packed_manifest.read_bytes()).hexdigest() + "\n",
            encoding="ascii",
        )
        return live, origin

    def test_live_brain_is_only_a_locator_for_its_pristine_origin(self):
        live, _origin = self._brain_locator()

        self.assertEqual(
            _resolve_initial_ground_up_locator(live), live.resolve()
        )

    def test_direct_origin_and_symlink_escape_are_rejected(self):
        live, origin = self._brain_locator()
        with self.assertRaisesRegex(ValueError, "live brain root"):
            _resolve_initial_ground_up_locator(origin)

        external = self.root / "external-origin"
        origin.rename(external)
        try:
            (live / "engine" / "origin").symlink_to(
                external, target_is_directory=True
            )
        except OSError as error:
            self.skipTest("directory symlinks unavailable: %s" % error)
        with self.assertRaisesRegex(ValueError, "distinct immutable"):
            _resolve_initial_ground_up_locator(live)

    def test_missing_or_non_distinct_origin_files_are_rejected(self):
        live, origin = self._brain_locator()
        (origin / "plasticity.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "missing plasticity"):
            _resolve_initial_ground_up_locator(live)

        (origin / "plasticity.safetensors").write_bytes(b"fixture")
        packed_manifest = origin / "packed-ternary" / "manifest.json"
        packed_checksum = origin / "packed-ternary" / "manifest.sha256"
        packed_checksum.write_text("0" * 64 + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "manifest checksum is invalid"):
            _resolve_initial_ground_up_locator(live)
        packed_checksum.write_text(
            hashlib.sha256(packed_manifest.read_bytes()).hexdigest() + "\n",
            encoding="ascii",
        )
        current = live / "engine" / "brain.json"
        current.unlink()
        try:
            os.link(origin / "brain.json", current)
        except OSError as error:
            self.skipTest("hard links unavailable: %s" % error)
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            _resolve_initial_ground_up_locator(live)

    def test_missing_provenance_or_post_origin_user_state_is_rejected(self):
        live, origin = self._brain_locator()
        metadata = self._origin_metadata()
        metadata.pop("ground_up_training_manifest")
        (origin / "brain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            _resolve_initial_ground_up_locator(live)

        metadata = self._origin_metadata()
        metadata["conversation"] = {
            "totalEntries": 1,
            "messageCount": 0,
            "actionCount": 1,
            "traceCount": 0,
        }
        (origin / "brain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            _resolve_initial_ground_up_locator(live)

        v3 = current_ground_up_curriculum_manifest()
        for field, value in (
            ("recent_token_context", [7, 11]),
            ("paged_working_memory", {"count": 1}),
            ("fresh_attention_boundary", {"epoch": 1}),
            ("memory_lifecycle", {"scratchItems": [{"id": "private"}]}),
        ):
            with self.subTest(v3_transient_user_state=field):
                metadata = self._origin_metadata(v3)
                metadata[field] = value
                (origin / "brain.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    _resolve_initial_ground_up_locator(live)

        metadata = self._origin_metadata()
        metadata["training_sources"] = [{"name": "private-user-data"}]
        (origin / "brain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            _resolve_initial_ground_up_locator(live)

    def test_train_command_passes_only_the_resolved_locator_to_trainer(self):
        supplied = self.root / "supplied-live-brain"
        resolved = self.root / "resolved-live-brain"
        context = SimpleNamespace(
            device="cpu",
            is_rank_zero=True,
            close=mock.Mock(),
        )
        trainer = mock.Mock()
        trainer.run.return_value = {"state": "complete"}
        with mock.patch.object(
            distributed_train,
            "_resolve_initial_ground_up_locator",
            return_value=resolved,
        ) as resolve, mock.patch.object(
            distributed_train,
            "initialize_distributed",
            return_value=context,
        ), mock.patch.object(
            distributed_train,
            "DistributedGroundUpTrainer",
            return_value=trainer,
        ) as trainer_type, contextlib.redirect_stdout(io.StringIO()):
            status = distributed_train.main(
                [
                    "train",
                    "--dataset",
                    str(self.root / "dataset.jsonl"),
                    "--output",
                    str(self.root / "output"),
                    "--initial-ground-up",
                    str(supplied),
                    "--profile",
                    "micro",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(status, 0)
        resolve.assert_called_once_with(supplied)
        self.assertEqual(
            trainer_type.call_args.kwargs["initial_brain_path"], resolved
        )
        trainer.run.assert_called_once_with()
        context.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
