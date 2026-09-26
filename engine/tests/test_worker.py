import base64
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from worker import DeferredEventLog, InlineGeneration, RpcFault, Worker
from omni_core import OmniConfig
from omni_core.brain import ChatGenerationCancelled
from omni_core.offload import NeuralStateResourcePause
from omni_core.persistence import EventLog


class WorkerProtocolTests(unittest.TestCase):
    def test_checkpoint_rpc_flushes_without_calling_snapshot(self):
        worker = Worker()
        brain = MagicMock()
        receipt = {
            "format": "omni-neural-checkpoint",
            "formatVersion": 1,
            "brainId": "checkpoint-brain",
            "operationId": "checkpoint-operation",
            "committed": True,
            "snapshotCreated": False,
        }
        brain.checkpoint.return_value = receipt
        with patch.object(worker, "_get", return_value=brain) as get_brain:
            result = worker.checkpoint(
                {"brainId": "checkpoint-brain", "operationId": "checkpoint-operation"},
                "request-checkpoint",
            )
        self.assertEqual(result, receipt)
        get_brain.assert_called_once()
        brain.checkpoint.assert_called_once_with("checkpoint-operation")
        brain.snapshot.assert_not_called()
        self.assertIn("checkpoint", worker.methods)
        with patch.object(worker, "_get") as invalid_get:
            with self.assertRaisesRegex(RpcFault, "operationId is required"):
                worker.checkpoint({"brainId": "checkpoint-brain"}, "missing-operation")
            invalid_get.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_manual_consolidation_is_not_an_rpc_and_empty_train_never_loads_a_brain(self):
        worker = Worker()
        self.assertNotIn("consolidate", worker.methods)
        invalid = (
            {},
            {"texts": []},
            {"text": " \x00 "},
            {"sourceIds": []},
        )
        with patch.object(
            worker,
            "_job",
            side_effect=AssertionError("invalid train must not load a brain"),
        ) as start_job:
            for params in invalid:
                with self.subTest(params=params):
                    with self.assertRaisesRegex(
                        RpcFault,
                        "requires non-empty text",
                    ):
                        worker.train(params, "invalid-train")
        start_job.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_workspace_carries_worker_owned_runtime_card_to_renderer(self):
        worker = Worker()
        brain = MagicMock()
        brain.workspace_snapshot.return_value = {
            "brainId": "runtime-placement",
            "queriedAt": "2026-08-26T00:00:00Z",
        }
        brain.runtime_card.return_value = {
            "pretrained_text_cortex": {
                "loaded": True,
                "resources": {
                    "residency": "tiered-ram-disk",
                    "diskPagingPerStep": True,
                    "estimatedSlowdownPercent": 78.0,
                    "actualPlacement": {
                        "reportedByBackend": True,
                        "tiers": {"cpu": 2, "disk": 1},
                    },
                },
            },
        }
        with patch.object(worker, "_get", return_value=brain):
            result = worker.workspace(
                {"brainId": "runtime-placement"}, "workspace-runtime"
            )

        self.assertEqual(
            result["runtimeCard"]["pretrained_text_cortex"]["resources"][
                "actualPlacement"
            ]["tiers"],
            {"cpu": 2, "disk": 1},
        )
        self.assertTrue(
            result["runtimeCard"]["pretrained_text_cortex"]["resources"][
                "diskPagingPerStep"
            ]
        )
        brain.workspace_snapshot.assert_called_once_with()
        brain.runtime_card.assert_called_once_with()
        worker.shutdown({}, "shutdown")

    def test_fresh_attention_rpc_commits_exact_operation_and_reports_epoch(self):
        worker = Worker()
        brain = MagicMock()
        brain.start_fresh_attention.return_value = {
            "format": "omni-fresh-attention-boundary",
            "formatVersion": 1,
            "brainId": "fresh-brain",
            "committed": True,
            "idempotent": False,
            "boundary": {"epoch": 3},
            "pagedCleanupPending": False,
        }
        worker.brains["fresh-brain"] = brain

        with patch.object(worker, "_get", return_value=brain), patch.object(
            worker, "notify"
        ) as notify:
            result = worker.fresh_attention(
                {
                    "brainId": "fresh-brain",
                    "operationId": "fresh-operation",
                },
                "request-id",
            )

        brain.start_fresh_attention.assert_called_once_with("fresh-operation")
        self.assertEqual(result["boundary"]["epoch"], 3)
        self.assertEqual(result["inlineGenerationsCancelled"], 0)
        self.assertEqual(result["observationSessionsCancelled"], 0)
        self.assertIs(worker.brains["fresh-brain"], brain)
        notify.assert_called_once()
        worker.shutdown({}, "shutdown")

    def test_fresh_attention_rpc_discards_partial_or_unclean_cached_state(self):
        for pending_cleanup, failure in ((False, True), (True, False)):
            with self.subTest(
                pending_cleanup=pending_cleanup,
                failure=failure,
            ):
                worker = Worker()
                brain = MagicMock()
                if failure:
                    brain.start_fresh_attention.side_effect = RuntimeError(
                        "precommit reset failure"
                    )
                else:
                    brain.start_fresh_attention.return_value = {
                        "format": "omni-fresh-attention-boundary",
                        "formatVersion": 1,
                        "brainId": "fresh-brain",
                        "committed": True,
                        "idempotent": False,
                        "boundary": {"epoch": 1},
                        "pagedCleanupPending": pending_cleanup,
                    }
                worker.brains["fresh-brain"] = brain
                with patch.object(worker, "_get", return_value=brain):
                    if failure:
                        with self.assertRaisesRegex(
                            RuntimeError, "precommit reset failure"
                        ):
                            worker.fresh_attention(
                                {"brainId": "fresh-brain"}, "request-id"
                            )
                    else:
                        worker.fresh_attention(
                            {"brainId": "fresh-brain"}, "request-id"
                        )
                self.assertNotIn("fresh-brain", worker.brains)
                brain.close.assert_called_once_with()
                worker.shutdown({}, "shutdown")

    def test_new_creation_rejects_private_architecture_shape_fields(self):
        worker = Worker()
        with patch("worker.AdaptiveBrain.create") as create:
            with self.assertRaisesRegex(
                RpcFault, "versioned hardwareTier profile"
            ):
                worker.create(
                    {
                        "brainId": "caller-shaped-brain",
                        "storagePath": "/tmp/omni-caller-shaped-brain",
                        "origin": "ground-up",
                        "hardwareTier": "micro",
                        "config": {
                            "name": "Caller-shaped brain",
                            "d_model": 4096,
                            "n_layers": 96,
                        },
                    },
                    "create-caller-shaped-brain",
                )
            create.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_new_creation_preflights_existing_checkpoint_profile_before_finalize(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-existing-profile-preflight-"
        ) as folder:
            engine = Path(folder) / "engine"
            engine.mkdir()
            (engine / "brain.json").write_text(
                json.dumps(
                    {
                        "config": OmniConfig.from_external(
                            {"hardwareTier": "personal"}
                        ).to_dict()
                    }
                ),
                "utf-8",
            )
            with patch("worker.AdaptiveBrain.create") as create, patch.object(
                worker, "notify"
            ) as notify:
                with self.assertRaisesRegex(
                    RpcFault, "existing brain architecture does not match"
                ):
                    worker.create(
                        {
                            "brainId": "mismatched-existing-profile",
                            "storagePath": folder,
                            "origin": "ground-up",
                            "hardwareTier": "micro",
                            "config": {"name": "Canonical micro build"},
                        },
                        "create-mismatched-existing-profile",
                    )
                create.assert_not_called()
                notify.assert_not_called()
                self.assertNotIn("mismatched-existing-profile", worker.brains)
        worker.shutdown({}, "shutdown")

    def test_new_creation_closes_post_create_profile_mismatch(self):
        worker = Worker()
        returned = MagicMock()
        returned.config = OmniConfig.from_external({"hardwareTier": "personal"})
        with tempfile.TemporaryDirectory(
            prefix="omni-returned-profile-postcondition-"
        ) as folder, patch(
            "worker.AdaptiveBrain.create", return_value=returned
        ) as create, patch.object(
            worker, "notify"
        ) as notify:
            with self.assertRaisesRegex(
                RpcFault, "created brain architecture does not match"
            ):
                worker.create(
                    {
                        "brainId": "mismatched-returned-profile",
                        "storagePath": folder,
                        "origin": "ground-up",
                        "hardwareTier": "micro",
                        "config": {"name": "Canonical micro build"},
                    },
                    "create-mismatched-returned-profile",
                )

            create.assert_called_once()
            returned.close.assert_called_once_with()
            notify.assert_not_called()
            self.assertNotIn("mismatched-returned-profile", worker.brains)
        worker.shutdown({}, "shutdown")

    def test_new_creation_rejects_invalid_or_conflicting_hardware_tiers(self):
        worker = Worker()
        with patch("worker.AdaptiveBrain.create") as create:
            for params, message in (
                (
                    {
                        "hardwareTier": "unbounded",
                        "config": {"name": "Invalid tier"},
                    },
                    "hardwareTier is invalid",
                ),
                (
                    {
                        "hardwareTier": "micro",
                        "config": {
                            "name": "Conflicting tier",
                            "hardwareTier": "workstation",
                        },
                    },
                    "declarations do not match",
                ),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(
                    RpcFault, message
                ):
                    worker.create(
                        {
                            "brainId": "invalid-tier-brain",
                            "storagePath": "/tmp/omni-invalid-tier-brain",
                            "origin": "ground-up",
                            **params,
                        },
                        "create-invalid-tier-brain",
                    )
            create.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_new_creation_rejects_unknown_modality_identifiers(self):
        worker = Worker()
        with patch("worker.AdaptiveBrain.create") as create:
            with self.assertRaisesRegex(RpcFault, "modalities are invalid"):
                worker.create(
                    {
                        "brainId": "invalid-modality-brain",
                        "storagePath": "/tmp/omni-invalid-modality-brain",
                        "origin": "ground-up",
                        "hardwareTier": "micro",
                        "modalities": ["vision", "unknown-modality"],
                        "config": {"name": "Invalid modality"},
                    },
                    "create-invalid-modality-brain",
                )
            create.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_runtime_requests_cannot_materialize_pending_brain_before_explicit_create(self):
        """An eager idle/load must not replace the builder's chosen cortex."""

        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-pending-create-race-"
        ) as folder:
            pending = {
                "brainId": "pending-ground-up-brain",
                "storagePath": folder,
                # This config mirrors the repository record that is visible
                # while the explicit create RPC is still training its core.
                "config": {
                    "name": "Pending ground-up construction",
                    "idleCognition": True,
                },
                "origin": "ground-up",
                "hardwareTier": "workstation",
            }

            with patch("worker.AdaptiveBrain.create") as create, patch(
                "worker.AdaptiveBrain.load"
            ) as load:
                with self.assertRaisesRegex(
                    RpcFault, "explicit create request must complete"
                ):
                    worker.load(pending, "early-load")
                with self.assertRaisesRegex(
                    RpcFault, "explicit create request must complete"
                ):
                    worker.idle_cycle(
                        {
                            **pending,
                            "toolSchemas": [],
                            "minimumIdleSeconds": 0,
                        },
                        "early-idle",
                    )
                create.assert_not_called()
                load.assert_not_called()
                self.assertNotIn(pending["brainId"], worker.brains)
                self.assertFalse(
                    (Path(folder) / "engine" / "brain.json").exists()
                )

            created = MagicMock()
            created.brain_id = pending["brainId"]
            created.storage_path = Path(folder).resolve()
            created.config = OmniConfig.from_external(
                {"hardwareTier": pending["hardwareTier"]}
            )
            created.summary.return_value = {"brainId": pending["brainId"]}
            with patch(
                "worker.AdaptiveBrain.create", return_value=created
            ) as create, contextlib.redirect_stdout(io.StringIO()):
                result = worker.create(pending, "authoritative-create")

            self.assertEqual(result["brainId"], pending["brainId"])
            create.assert_called_once()
            created_config = create.call_args.args[2]
            self.assertEqual(created_config.origin_kind, "ground-up")
            self.assertEqual(created_config.hardware_tier, "workstation")
            self.assertTrue(create.call_args.kwargs["initialize_ground_up"])
            self.assertIs(worker.brains[pending["brainId"]], created)
        worker.shutdown({}, "shutdown")

    def test_background_idle_defers_large_checkpoint_before_loading_or_mutating(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-bounded-idle-admission-"
        ) as folder:
            brain_id = "large-idle-checkpoint"
            engine = Path(folder) / "engine"
            engine.mkdir()
            metadata_path = engine / "brain.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "brain_id": brain_id,
                        "substrate": {
                            "persistence": {
                                "shardCount": 9_456,
                                "counts": {
                                    "assemblies": 117,
                                    "neurons": 14_941,
                                    "synapses": 4_811_811,
                                },
                            }
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            before = metadata_path.read_bytes()
            with patch.object(worker, "_get") as load:
                result = worker.idle_cycle(
                    {
                        "brainId": brain_id,
                        "storagePath": folder,
                        "toolSchemas": [],
                        "minimumIdleSeconds": 0,
                    },
                    "background-idle-large",
                )

            load.assert_not_called()
            self.assertFalse(result["ran"])
            self.assertEqual(result["reason"], "resource-envelope")
            self.assertEqual(result["retryAfterSeconds"], 15 * 60)
            self.assertEqual(result["actions"], [])
            self.assertEqual(
                result["admission"]["policy"],
                "bounded-persisted-idle-v1",
            )
            self.assertEqual(
                result["admission"]["substrateShards"], 9_456
            )
            self.assertEqual(
                result["admission"]["substrateEntities"],
                4_826_869,
            )
            self.assertFalse(result["admission"]["checkpointMutated"])
            self.assertFalse(result["admission"]["brainLoadedByRequest"])
            self.assertEqual(metadata_path.read_bytes(), before)
            self.assertNotIn(brain_id, worker.brains)

            foreground_brain = MagicMock()
            foreground_brain.brain_id = brain_id
            foreground_brain.chat.return_value = {
                "text": "foreground response",
                "trace": {"id": "foreground-trace"},
            }
            with patch.object(
                worker, "_get", return_value=foreground_brain
            ), contextlib.redirect_stdout(io.StringIO()):
                chat = worker.chat(
                    {
                        "brainId": brain_id,
                        "storagePath": folder,
                        "input": "hi",
                        "toolSchemas": [],
                    },
                    "foreground-chat",
                )
            self.assertEqual(chat["text"], "foreground response")
            foreground_brain.chat.assert_called_once()
        worker.shutdown({}, "shutdown")

    def test_background_idle_admits_small_checkpoint_to_normal_neural_path(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-small-idle-admission-"
        ) as folder:
            brain_id = "small-idle-checkpoint"
            engine = Path(folder) / "engine"
            engine.mkdir()
            (engine / "brain.json").write_text(
                json.dumps(
                    {
                        "brain_id": brain_id,
                        "substrate": {
                            "persistence": {
                                "shardCount": 2,
                                "counts": {
                                    "assemblies": 4,
                                    "neurons": 8,
                                    "synapses": 16,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            brain = MagicMock()
            brain.idle_cycle.return_value = {
                "brainId": brain_id,
                "ran": True,
                "actions": [],
            }
            with patch.object(worker, "_get", return_value=brain) as load:
                result = worker.idle_cycle(
                    {
                        "brainId": brain_id,
                        "storagePath": folder,
                        "toolSchemas": [],
                        "minimumIdleSeconds": 7,
                    },
                    "background-idle-small",
                )

            load.assert_called_once()
            brain.idle_cycle.assert_called_once_with(
                tool_schemas=[], minimum_idle_seconds=7.0
            )
            self.assertTrue(result["ran"])
        worker.shutdown({}, "shutdown")

    def test_send_writes_utf8_protocol_bytes_through_cp1252_stdout(self):
        raw = io.BytesIO()
        cp1252_stdout = io.TextIOWrapper(raw, encoding="cp1252")
        try:
            with contextlib.redirect_stdout(cp1252_stdout):
                Worker._send(
                    {
                        "jsonrpc": "2.0",
                        "id": "unicode",
                        "result": {"transition": "working memory → disk"},
                    }
                )
            payload = raw.getvalue().decode("utf-8")
        finally:
            cp1252_stdout.detach()
        self.assertEqual(
            json.loads(payload)["result"]["transition"],
            "working memory → disk",
        )

    def test_cancelling_inline_job_removes_staged_artifacts(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-inline-cancel-"
        ) as folder:
            action_id = "b" * 32
            staging = (
                Path(folder)
                / "engine"
                / ".inline-imagination"
                / action_id
            )
            artifact = staging / "artifacts" / "partial.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"partial")
            future = Future()
            future.set_result({"path": str(artifact)})
            record = InlineGeneration(
                brain_id="cancel-brain",
                action_id=action_id,
                stream_id="cancel-stream",
                signature="fixture",
                staging_root=staging,
                events=DeferredEventLog(),
                future=future,
                job_id="cancel-job",
            )
            worker._inline_generations[(record.brain_id, action_id)] = record

            result = worker.cancel({"jobId": "cancel-job"}, "cancel")

            self.assertEqual(result["inlineGenerationsCancelled"], 1)
            self.assertFalse(staging.exists())
            self.assertFalse(staging.parent.exists())
            self.assertEqual(worker._inline_generations, {})
        worker.shutdown({}, "shutdown")

    def test_cancel_acknowledges_durable_job_and_rejects_only_matching_evolution_candidate(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-worker-cancel-ack-"
        ) as folder:
            root = Path(folder)
            engine = root / "engine"
            matching_id = "a" * 32
            unrelated_id = "b" * 32
            for candidate_id, runtime_request_id in (
                (matching_id, "evolution-job"),
                (unrelated_id, "another-job"),
            ):
                directory = engine / "candidates" / candidate_id
                directory.mkdir(parents=True)
                (directory / "candidate.json").write_text(
                    json.dumps(
                        {
                            "id": candidate_id,
                            "kind": "neural-evolution",
                            "status": "training",
                            "provenance": {
                                "runtimeRequestId": runtime_request_id
                            },
                        }
                    ),
                    "utf-8",
                )

            request = {
                "brainId": "cancel-brain",
                "storagePath": str(root),
                "jobId": "evolution-job",
                "kind": "neural-evolution-proposal",
                "reason": "Stopped by operator.",
            }
            with contextlib.redirect_stdout(io.StringIO()):
                first = worker.cancel(request, "cancel-1")
                second = worker.cancel(request, "cancel-2")

            self.assertTrue(first["acknowledged"])
            self.assertEqual(first["candidateIds"], [matching_id])
            self.assertEqual(second["eventId"], first["eventId"])
            matching = json.loads(
                (
                    engine / "candidates" / matching_id / "candidate.json"
                ).read_text("utf-8")
            )
            unrelated = json.loads(
                (
                    engine / "candidates" / unrelated_id / "candidate.json"
                ).read_text("utf-8")
            )
            self.assertEqual(matching["status"], "rejected")
            self.assertTrue(matching["cancellationAcknowledged"])
            self.assertEqual(unrelated["status"], "training")

            events = EventLog(engine / "events.sqlite3", "cancel-brain")
            try:
                cancellations = [
                    event
                    for event in events.recent(100)
                    if event["kind"] == "job-cancelled"
                    and event["jobId"] == "evolution-job"
                ]
            finally:
                events.close()
            self.assertEqual(len(cancellations), 1)
        worker.shutdown({}, "shutdown")

    def test_chat_cooperative_cancel_acknowledges_one_safe_boundary_without_shutdown(self):
        worker = Worker()
        brain = MagicMock()
        brain.brain_id = "warm-cancel-brain"

        def cancelled_chat(*_args, **kwargs):
            self.assertTrue(worker.request_cooperative_cancel())
            self.assertTrue(kwargs["cancel_check"]())
            raise ChatGenerationCancelled("chat generation was cancelled")

        brain.chat.side_effect = cancelled_chat
        request = {
            "jsonrpc": "2.0",
            "id": "warm-turn",
            "method": "chat",
            "params": {
                "brainId": brain.brain_id,
                "input": "stop safely",
                "streamId": "warm-turn",
                "toolSchemas": [],
            },
        }
        with patch.object(worker, "_get", return_value=brain), self.assertRaises(
            RpcFault
        ) as raised:
            worker.dispatch(request)

        self.assertEqual(raised.exception.code, -32800)
        self.assertEqual(raised.exception.data["safeBoundary"], True)
        self.assertTrue(worker.running)
        self.assertIsNone(worker._active_request)
        self.assertFalse(worker._cooperative_cancel.is_set())
        worker.shutdown({}, "shutdown")

    def test_load_cancel_keeps_the_completed_warm_cache_alive(self):
        worker = Worker()
        brain = MagicMock()
        brain.summary.return_value = {"brainId": "warm-load-brain"}

        def completed_load(_params):
            self.assertTrue(worker.request_cooperative_cancel())
            return brain

        request = {
            "jsonrpc": "2.0",
            "id": "warm-load",
            "method": "load",
            "params": {"brainId": "warm-load-brain"},
        }
        with patch.object(worker, "_get", side_effect=completed_load), self.assertRaises(
            RpcFault
        ) as raised:
            worker.dispatch(request)

        self.assertEqual(raised.exception.code, -32800)
        self.assertTrue(raised.exception.data["warm"])
        self.assertTrue(worker.running)
        brain.close.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_inline_imagination_never_starts_before_ask_approval(self):
        worker = Worker()
        brain = MagicMock()
        brain.brain_id = "permission-brain"

        def chat_side_effect(*_args, **kwargs):
            callback = kwargs["stream_callback"]
            callback(
                "action",
                {
                    "actionId": "a" * 32,
                    "action": {
                        "kind": "imagine",
                        "toolId": "modality.imagine",
                        "action": "generate",
                        "arguments": {"modality": "image"},
                    },
                },
            )
            callback("token", {"delta": "permission retained"})
            return {"text": "permission retained", "trace": {"id": "trace"}}

        brain.chat.side_effect = chat_side_effect
        with patch.object(worker, "_get", return_value=brain), patch.object(
            worker, "_start_inline_generation"
        ) as start_inline, contextlib.redirect_stdout(io.StringIO()):
            result = worker.chat(
                {
                    "brainId": brain.brain_id,
                    "input": "Imagine only after approval.",
                    "streamId": "permission-stream",
                    "toolSchemas": [
                        {
                            "id": "modality.imagine",
                            "actions": ["generate"],
                            "grant": "ask",
                        }
                    ],
                },
                "permission-chat",
            )
        self.assertEqual(result["text"], "permission retained")
        start_inline.assert_not_called()
        worker.shutdown({}, "shutdown")

    def test_chat_stream_marks_reply_complete_before_turn_commit(self):
        worker = Worker()
        brain = MagicMock()
        brain.brain_id = "phase-brain"
        expected_phase = {
            "phase": "reply-complete-learning",
            "replyComplete": True,
            "turnCommitted": False,
            "learning": True,
            "saving": True,
        }

        def chat_side_effect(*_args, **kwargs):
            callback = kwargs["stream_callback"]
            callback("token", {"delta": "hi"})
            callback("phase", dict(expected_phase))
            return {"text": "hi", "trace": {"id": "phase-trace"}}

        brain.chat.side_effect = chat_side_effect
        output = io.StringIO()
        with patch.object(
            worker, "_get", return_value=brain
        ), contextlib.redirect_stdout(output):
            result = worker.chat(
                {
                    "brainId": brain.brain_id,
                    "input": "hello",
                    "streamId": "phase-stream",
                    "toolSchemas": [],
                },
                "phase-chat",
            )

        self.assertEqual(result["text"], "hi")
        events = [
            value["params"]
            for value in (
                json.loads(line)
                for line in output.getvalue().splitlines()
                if line.strip()
            )
            if value.get("method") == "event"
            and value.get("params", {}).get("streamId") == "phase-stream"
        ]
        self.assertEqual(
            [event["type"] for event in events],
            ["chat-token", "chat-phase"],
        )
        self.assertEqual([event["sequence"] for event in events], [0, 1])
        self.assertEqual(events[-1]["data"], expected_phase)
        self.assertEqual(
            brain.chat.call_args.kwargs["turn_id"], "phase-stream"
        )
        worker.shutdown({}, "shutdown")

    def test_chat_receipt_reads_exact_atomic_commit_without_loading_brain(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-chat-receipt-"
        ) as folder:
            brain_id = "receipt-brain"
            turn_id = "receipt-turn"
            human = "hello receipt"
            response = "committed response"
            input_sha = hashlib.sha256(human.encode("utf-8")).hexdigest()
            follow_up = "tool follow-up"
            follow_up_sha = hashlib.sha256(
                follow_up.encode("utf-8")
            ).hexdigest()
            parameter_sha = "a" * 64
            trace = {
                "id": "trace-id",
                "turn_id": turn_id,
                "input_sha256": input_sha,
                "parameter_checksum_after": parameter_sha,
                "generation_backend": "fixture",
            }
            follow_up_trace = {
                "id": "follow-up-trace-id",
                "turn_id": turn_id,
                "input_sha256": follow_up_sha,
                "parameter_checksum_after": "d" * 64,
                "generation_backend": "fixture",
            }
            metadata = {
                "brain_id": brain_id,
                "updated_at": "2026-09-07T05:00:00Z",
                "messages": [
                    {
                        "id": "human-id",
                        "role": "human",
                        "content": human,
                        "created_at": "2026-09-07T04:59:00Z",
                        "turn_id": turn_id,
                    },
                    {
                        "id": "brain-id",
                        "role": "brain",
                        "content": response,
                        "created_at": "2026-09-07T04:59:01Z",
                        "turn_id": turn_id,
                    },
                ],
                "traces": [trace],
                "counters": {
                    "inference_count": 4,
                    "plasticity_events": 91,
                    "consolidation_cycles": 3,
                },
                "completed_chat_turns": [
                    {
                        "format": "omni-completed-chat-turn",
                        "formatVersion": 1,
                        "turnId": turn_id,
                        "inputSha256": input_sha,
                        "humanMessageId": "human-id",
                        "brainMessageId": "brain-id",
                        "traceId": "trace-id",
                        "inferenceCount": 4,
                        "parameterChecksumAfter": parameter_sha,
                        "committedAt": "2026-09-07T05:00:00Z",
                    },
                ],
                "substrate": {
                    "persistence": {"activeGeneration": "b" * 64}
                },
                "mutable_state": {"activeGeneration": "c" * 64},
            }
            engine = Path(folder) / "engine"
            engine.mkdir()
            (engine / "brain.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            params = {
                "brainId": brain_id,
                "storagePath": folder,
                "turnId": turn_id,
                "inputSha256": input_sha,
                "minimumInferenceCount": 3,
            }

            with patch.object(worker, "_get") as load:
                result = worker.chat_receipt(params, "receipt-query")

            load.assert_not_called()
            self.assertTrue(result["committed"])
            self.assertTrue(result["turnCommitted"])
            self.assertFalse(result["legacyMatched"])
            self.assertEqual(result["humanMessage"]["content"], human)
            self.assertEqual(result["brainMessage"]["content"], response)
            self.assertEqual(
                result["brainMessage"]["traceId"], trace["id"]
            )
            self.assertEqual(result["trace"], trace)
            self.assertEqual(result["inferenceCount"], 4)
            self.assertEqual(result["plasticityEvents"], 91)
            self.assertEqual(result["consolidationCycles"], 3)
            self.assertEqual(result["substrateGeneration"], "b" * 64)
            self.assertEqual(result["mutableStateGeneration"], "c" * 64)
            stale = worker.chat_receipt(
                {**params, "minimumInferenceCount": 4}, "stale-query"
            )
            self.assertFalse(stale["committed"])
            wrong_earlier_baseline = worker.chat_receipt(
                {**params, "minimumInferenceCount": 2},
                "earlier-baseline-query",
            )
            self.assertFalse(wrong_earlier_baseline["committed"])
            metadata["messages"].extend(
                [
                    {
                        "id": "follow-up-human-id",
                        "role": "human",
                        "content": follow_up,
                        "created_at": "2026-09-07T04:59:02Z",
                        "turn_id": turn_id,
                    },
                    {
                        "id": "follow-up-brain-id",
                        "role": "brain",
                        "content": "follow-up response",
                        "created_at": "2026-09-07T04:59:03Z",
                        "turn_id": turn_id,
                    },
                ]
            )
            metadata["traces"].append(follow_up_trace)
            metadata["counters"]["inference_count"] = 5
            metadata["completed_chat_turns"].append(
                {
                    "format": "omni-completed-chat-turn",
                    "formatVersion": 1,
                    "turnId": turn_id,
                    "inputSha256": follow_up_sha,
                    "humanMessageId": "follow-up-human-id",
                    "brainMessageId": "follow-up-brain-id",
                    "traceId": "follow-up-trace-id",
                    "inferenceCount": 5,
                    "parameterChecksumAfter": "d" * 64,
                    "committedAt": "2026-09-07T05:00:01Z",
                }
            )
            (engine / "brain.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            superseded = worker.chat_receipt(
                params, "superseded-receipt-query"
            )
            self.assertFalse(superseded["committed"])
            follow_up_result = worker.chat_receipt(
                {
                    **params,
                    "inputSha256": follow_up_sha,
                    "minimumInferenceCount": 4,
                },
                "follow-up-query",
            )
            self.assertTrue(follow_up_result["committed"])
            self.assertEqual(
                follow_up_result["humanMessage"]["id"],
                "follow-up-human-id",
            )
            self.assertEqual(follow_up_result["inferenceCount"], 5)
            wrong_turn = worker.chat_receipt(
                {
                    **params,
                    "turnId": "other-turn",
                    "inputSha256": follow_up_sha,
                    "minimumInferenceCount": 4,
                },
                "wrong-turn-query",
            )
            self.assertFalse(wrong_turn["committed"])
        worker.shutdown({}, "shutdown")

    def test_chat_receipt_recovers_one_latest_pre_receipt_commit_only(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-legacy-chat-receipt-"
        ) as folder:
            brain_id = "legacy-receipt-brain"
            turn_id = "cancelled-ui-turn"
            human = "legacy committed input"
            input_sha = hashlib.sha256(human.encode("utf-8")).hexdigest()
            parameter_sha = "d" * 64
            trace = {
                "id": "legacy-trace",
                "created_at": "2026-09-07T04:59:01Z",
                "input_sha256": input_sha,
                "parameter_checksum_after": parameter_sha,
            }
            metadata = {
                "brain_id": brain_id,
                "updated_at": "2026-09-07T05:00:00Z",
                "messages": [
                    {
                        "id": "legacy-human",
                        "role": "human",
                        "content": human,
                        "created_at": "2026-09-07T04:59:00Z",
                    },
                    {
                        "id": "legacy-brain",
                        "role": "brain",
                        "content": "legacy response",
                        "created_at": "2026-09-07T04:59:01Z",
                    },
                ],
                "traces": [trace],
                "counters": {"inference_count": 1},
                "substrate": {"persistence": {}},
                "mutable_state": {},
            }
            engine = Path(folder) / "engine"
            engine.mkdir()
            metadata_path = engine / "brain.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            params = {
                "brainId": brain_id,
                "storagePath": folder,
                "turnId": turn_id,
                "inputSha256": input_sha,
                "minimumInferenceCount": 0,
            }

            result = worker.chat_receipt(params, "legacy-query")

            self.assertTrue(result["committed"])
            self.assertTrue(result["legacyMatched"])
            self.assertEqual(result["turnId"], turn_id)
            self.assertEqual(result["trace"], trace)
            self.assertEqual(
                result["brainMessage"]["traceId"], "legacy-trace"
            )
            no_second_turn = worker.chat_receipt(
                {**params, "minimumInferenceCount": 1}, "legacy-stale"
            )
            self.assertFalse(no_second_turn["committed"])
            wrong_input = worker.chat_receipt(
                {
                    **params,
                    "inputSha256": hashlib.sha256(b"other").hexdigest(),
                },
                "legacy-wrong-input",
            )
            self.assertFalse(wrong_input["committed"])
            metadata["messages"] = metadata["messages"][:1]
            metadata["traces"] = []
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            partial = worker.chat_receipt(params, "legacy-partial-state")
            self.assertFalse(partial["committed"])
        worker.shutdown({}, "shutdown")

    def test_authoritative_overlay_digest_rejects_post_review_mutation(self):
        worker = Worker()
        with tempfile.TemporaryDirectory(
            prefix="omni-worker-overlay-target-"
        ) as target_folder, tempfile.TemporaryDirectory(
            prefix="omni-worker-overlay-source-"
        ) as source_folder:
            config = {
                "name": "Overlay fixture",
                "onlineLearning": False,
                "learnFromOwnMessages": False,
                "spikingDynamics": False,
            }
            worker.create(
                {
                    "brainId": "overlay-target",
                    "storagePath": target_folder,
                    "config": config,
                    "hardwareTier": "micro",
                },
                "create-target",
            )
            worker.create(
                {
                    "brainId": "overlay-source",
                    "storagePath": source_folder,
                    "config": config,
                    "hardwareTier": "micro",
                },
                "create-source",
            )
            source = worker.brains["overlay-source"]
            source.learn_experience(
                "A fork-local causal assembly enters replay.",
                steps=1,
                importance=1.0,
            )
            source.save()
            params = {
                "targetBrainId": "overlay-target",
                "targetStoragePath": target_folder,
                "sourceBrainId": "overlay-source",
                "sourceStoragePath": source_folder,
            }
            reviewed = worker.preview_overlay(params, "preview")
            self.assertRegex(reviewed["digest"], r"^[a-f0-9]{64}$")
            self.assertGreater(reviewed["additions"]["neurons"], 0)
            self.assertGreater(reviewed["additions"]["assemblies"], 0)
            self.assertGreater(reviewed["additions"]["replayExamples"], 0)
            self.assertFalse(reviewed["weightsAveraged"])

            target = worker.brains["overlay-target"]
            target.learn_experience(
                "The target base changed after the operator reviewed it.",
                steps=1,
                importance=1.0,
            )
            target.save()
            with self.assertRaisesRegex(RpcFault, "changed after review"):
                worker.merge_overlay(
                    {
                        **params,
                        "expectedPreviewDigest": reviewed["digest"],
                    },
                    "stale-target-merge",
                )

            reviewed = worker.preview_overlay(params, "preview-after-target")
            source.learn_experience(
                "This mutation happened after the operator reviewed the fork.",
                steps=1,
                importance=1.0,
            )
            source.save()
            with self.assertRaisesRegex(RpcFault, "changed after review"):
                worker.merge_overlay(
                    {
                        **params,
                        "expectedPreviewDigest": reviewed["digest"],
                    },
                    "stale-merge",
                )

            refreshed = worker.preview_overlay(params, "preview-again")
            merged = worker.merge_overlay(
                {
                    **params,
                    "expectedPreviewDigest": refreshed["digest"],
                },
                "merge",
            )
            self.assertEqual(merged["reviewedDigest"], refreshed["digest"])
            self.assertGreater(merged["ideas"], 0)
            self.assertGreater(merged["replayExamples"], 0)
            self.assertFalse(merged["weightsAveraged"])
            worker.brains["overlay-target"].events.close()
            source.events.close()

    def test_failed_ingestion_discards_partial_state_and_reloads_checkpoint(self):
        worker = Worker()
        partial = MagicMock()
        partial.brain_id = "rollback-brain"
        partial.storage_path = Path("/tmp/omni-rollback-brain")
        partial.events = MagicMock()
        partial.ingest.side_effect = RuntimeError("cancelled mid-record")
        restored = MagicMock()
        restored.brain_id = partial.brain_id
        restored.storage_path = partial.storage_path
        worker.brains[partial.brain_id] = partial

        with patch.object(worker, "_get", return_value=partial), patch(
            "worker.AdaptiveBrain.load", return_value=restored
        ) as load, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                worker.ingest(
                    {
                        "brainId": partial.brain_id,
                        "text": "record that must roll back",
                        "jobId": "job-rollback",
                    },
                    "request-rollback",
                )

        partial.close.assert_called_once()
        load.assert_called_once_with(
            partial.storage_path,
            expected_brain_id=partial.brain_id,
        )
        self.assertIs(worker.brains[partial.brain_id], restored)

    def test_ingestion_streams_structured_native_dataset_progress(self):
        worker = Worker()
        brain = MagicMock()
        brain.brain_id = "dataset-progress-brain"
        brain.storage_path = Path("/tmp/dataset-progress-brain")
        brain.events = MagicMock()
        structured = {
            "coverage": {
                "discoveredRecords": 2,
                "processedRecords": 1,
                "rejectedRecords": 0,
            },
            "currentRecord": 2,
            "committedRecords": 1,
            "checkpointCommitted": False,
            "expectedRecords": 2,
            "recordTotalKnown": True,
        }

        def ingest_with_progress(**kwargs):
            kwargs["progress"](
                0.5,
                "Learning dataset record 2/2",
                {"datasetProgress": structured},
            )
            return {"promoted": False}

        brain.ingest.side_effect = ingest_with_progress
        worker.brains[brain.brain_id] = brain
        output = io.StringIO()
        with patch.object(
            worker, "_get", return_value=brain
        ), contextlib.redirect_stdout(output):
            result = worker.ingest(
                {
                    "brainId": brain.brain_id,
                    "text": "private source must not enter progress",
                    "jobId": "dataset-progress-job",
                },
                "dataset-progress-request",
            )

        self.assertFalse(result["promoted"])
        messages = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.strip()
        ]
        progress = next(
            message["params"]
            for message in messages
            if message.get("method") == "event"
            and message.get("params", {}).get("type") == "job-progress"
            and isinstance(message.get("params", {}).get("data"), dict)
            and "datasetProgress"
            in message["params"]["data"]
        )
        self.assertEqual(
            progress["data"]["datasetProgress"], structured
        )
        self.assertNotIn("private source", json.dumps(progress))

        with patch.object(
            worker, "_get", return_value=brain
        ), contextlib.redirect_stdout(io.StringIO()):
            _brain, job_id, callback = worker._job(
                {
                    "brainId": brain.brain_id,
                    "jobId": "cancel-dataset-progress",
                },
                "cancel-dataset-request",
                "ingestion",
            )
            worker.cancelled_jobs.add(job_id)
            with self.assertRaisesRegex(RpcFault, "job was cancelled"):
                callback(
                    0.6,
                    "Learning dataset record 2/2",
                    {"datasetProgress": structured},
                )
        worker.cancelled_jobs.discard(job_id)
        worker.shutdown({}, "shutdown")

    def test_native_resource_pause_is_persisted_and_returned_as_recoverable(self):
        worker = Worker()
        partial = MagicMock()
        partial.brain_id = "native-pause-brain"
        partial.storage_path = Path("/tmp/omni-native-pause-brain")
        status = {
            "systemRamBudgetBytes": 11_274_289_152,
            "currentOmniAvailableBytes": 10_468_982_784,
            "processMemoryBytes": 4_776_187_776,
            "acceleratorFreeMemoryBytes": 8_904_081_408,
            "acceleratorAllocatedMemoryBytes": 80_342_784,
        }
        partial.ingest.side_effect = NeuralStateResourcePause(
            "neural state paused at the mandatory disk reserve",
            status,
        )
        restored = MagicMock()
        restored.brain_id = partial.brain_id
        restored.storage_path = partial.storage_path
        restored.ingestion_checkpoints = {}
        worker.brains[partial.brain_id] = partial

        with patch.object(worker, "_get", return_value=partial), patch.object(
            worker, "notify"
        ), patch("worker.AdaptiveBrain.load", return_value=restored):
            with self.assertRaises(RpcFault) as raised:
                worker.ingest(
                    {
                        "brainId": partial.brain_id,
                        "text": "record that must remain uncommitted",
                        "jobId": "job-native-pause",
                    },
                    "request-native-pause",
                )

        self.assertEqual(raised.exception.code, -32020)
        returned = raised.exception.data["resourcePause"]
        for field, expected in status.items():
            self.assertEqual(returned[field], expected)
        self.assertTrue(returned["recoverable"])

        persisted = {
            call.args[0]: call.args[1]
            for call in restored.events.append.call_args_list
        }
        rollback = persisted["ingestion-rollback"]
        self.assertEqual(rollback["resourcePause"], returned)
        self.assertEqual(rollback["activeCheckpoints"], [])
        self.assertFalse(rollback["uncommittedLearningRepresentedAsCommitted"])
        self.assertIn("uncommitted record batch discarded", rollback["reason"])
        self.assertEqual(
            persisted["job-paused"]["resourcePause"], returned
        )
        self.assertIs(worker.brains[partial.brain_id], restored)

    def test_gpu_profiles_select_available_backend_without_overriding_explicit_device(self):
        worker = Worker()
        with patch("worker.torch.cuda.is_available", return_value=True):
            selected = worker._builder_config({"hardwareTier": "gpu"}, {})
        self.assertEqual(selected["device"], "cuda")

        with patch("worker.torch.cuda.is_available", return_value=False), patch.object(
            worker, "_mps_available", return_value=True
        ), patch.object(worker, "_directml_available", return_value=True):
            selected = worker._builder_config({"hardwareTier": "gpu"}, {})
        self.assertEqual(selected["device"], "mps")

        with patch("worker.torch.cuda.is_available", return_value=False), patch.object(
            worker, "_mps_available", return_value=False
        ), patch.object(
            worker, "_directml_available", return_value=True
        ):
            selected = worker._builder_config({"hardwareTier": "workstation"}, {})
        self.assertEqual(selected["device"], "directml")

        with patch("worker.torch.cuda.is_available", return_value=True):
            selected = worker._builder_config(
                {"hardwareTier": "gpu"}, {"device": "cpu"}
            )
        self.assertEqual(selected["device"], "cpu")

    def test_jsonrpc_health_create_state_and_shutdown(self):
        worker = Worker()

        def request(identifier, method, params):
            with contextlib.redirect_stdout(io.StringIO()):
                response = worker.dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": identifier,
                        "method": method,
                        "params": params,
                    }
                )
            self.assertEqual(response["jsonrpc"], "2.0")
            self.assertEqual(response["id"], identifier)
            self.assertNotIn("error", response)
            return response["result"]

        health = request("health", "health", {})
        self.assertTrue(health["ready"])
        self.assertIn("directml", health["capabilities"])
        self.assertIn("mps", health["capabilities"])
        self.assertEqual(health["operatingSystem"], sys.platform)

        with tempfile.TemporaryDirectory(prefix="omni-worker-test-") as folder:
            config = {
                "name": "RPC brain",
                "ternaryWeights": True,
                "spikingDynamics": True,
                "stdpPlasticity": True,
                "liquidDynamics": True,
                "vectorSymbolicMemory": True,
                "onlineLearning": False,
                "consolidation": True,
                "metaplasticity": True,
            }
            created = request(
                "create",
                "create",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "config": config,
                    "hardwareTier": "micro",
                    "modalities": ["image"],
                },
            )
            self.assertEqual(created["runtimeCard"]["hardware_tier"], "micro")
            self.assertEqual(created["runtimeCard"]["enabled_modalities"], ["image"])

            packed = request(
                "packed",
                "export_ternary",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                },
            )
            self.assertEqual(
                packed["summary"]["eligibleTensorCount"],
                len(packed["manifest"]["coverage"]["eligibleTensorNames"]),
            )
            self.assertTrue(packed["manifest"]["coverage"]["complete"])
            self.assertTrue(
                (Path(packed["path"]) / "manifest.json").is_file()
            )
            self.assertTrue(
                (Path(packed["path"]) / "manifest.sha256").is_file()
            )

            streamed_output = io.StringIO()
            forced_action = {
                "kind": "imagine",
                "toolId": "modality.imagine",
                "action": "generate",
                "arguments": {
                    "modality": "image",
                    "conceptIds": ["organic-fixture"],
                    "organic": True,
                },
                "confidence": 0.91,
            }
            with patch.object(
                worker.brains["rpc-brain"],
                "_select_structured_actions",
                return_value=(
                    {
                        "talk": 0.01,
                        "tool": 0.01,
                        "imagine": 0.91,
                        "agent": 0.01,
                        "ponder": 0.01,
                        "learn": 0.01,
                        "evolve": 0.01,
                        "stop": 0.03,
                    },
                    [forced_action],
                ),
            ), contextlib.redirect_stdout(streamed_output):
                streamed_response = worker.dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": "chat",
                        "method": "chat",
                        "params": {
                            "brainId": "rpc-brain",
                            "storagePath": folder,
                            "input": "Inspect this workspace.",
                            "maxNewTokens": 2,
                            "seed": 17,
                            "streamId": "turn-stream",
                            "toolSchemas": [
                                {
                                    "id": "windows.files",
                                    "actions": ["list", "read"],
                                    "grant": "ask",
                                },
                                {
                                    "id": "modality.imagine",
                                    "actions": ["generate"],
                                    "grant": "auto",
                                },
                            ],
                        },
                    }
                )
                # Emulate the protocol loop exactly: dispatch returns first,
                # then the worker writes the chat response. At least one real
                # modality decoder preview must already be in the stream.
                Worker._send(streamed_response)
                with worker._inline_lock:
                    inline_action_ids = [
                        record.action_id
                        for record in worker._inline_generations.values()
                        if record.brain_id == "rpc-brain"
                    ]
                self.assertEqual(len(inline_action_ids), 1)
                modality_response = worker.dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": "image-preview",
                        "method": "generate_modality",
                        "params": {
                            "brainId": "rpc-brain",
                            "storagePath": folder,
                            "jobId": "image-job",
                            "neuralActionId": inline_action_ids[0],
                            "modality": "image",
                            "conceptIds": ["organic-fixture"],
                        },
                    }
                )
            self.assertNotIn("error", streamed_response)
            self.assertNotIn("error", modality_response)
            self.assertTrue(
                modality_response["result"]["generatedDuringChat"]
            )
            self.assertEqual(
                modality_response["result"]["neuralActionId"],
                inline_action_ids[0],
            )
            chatted = streamed_response["result"]
            stream_messages = [
                json.loads(line)
                for line in streamed_output.getvalue().splitlines()
                if line.strip()
            ]
            stream_events = [
                message["params"]
                for message in stream_messages
                if message.get("method") == "event"
                and message.get("params", {}).get("streamId")
                == "turn-stream"
            ]
            self.assertEqual(
                [event["sequence"] for event in stream_events],
                list(range(len(stream_events))),
            )
            self.assertEqual(stream_events[0]["type"], "chat-action")
            self.assertEqual(
                stream_events[0]["data"]["action"]["kind"], "imagine"
            )
            token_events = [
                event for event in stream_events
                if event["type"] == "chat-token"
            ]
            self.assertTrue(token_events)
            self.assertEqual(
                "".join(event["data"]["delta"] for event in token_events).strip(),
                chatted["text"],
            )
            self.assertEqual(
                chatted["trace"]["available_tool_ids"],
                ["modality.imagine", "windows.files"],
            )
            self.assertFalse(chatted["trace"]["tool_schema_text_injected"])
            self.assertEqual(
                chatted["runtimeCard"]["tool_schema_channel"],
                "substrate-capability-embedding",
            )
            first_preview_index = next(
                index
                for index, message in enumerate(stream_messages)
                if message.get("method") == "event"
                and message.get("params", {}).get("type")
                == "modality-preview"
                and message.get("params", {}).get("streamId")
                == "turn-stream"
            )
            chat_response_index = next(
                index
                for index, message in enumerate(stream_messages)
                if message.get("id") == "chat"
            )
            self.assertLess(
                first_preview_index,
                chat_response_index,
                "a real generated preview must arrive before chat RPC completion",
            )
            first_preview = stream_messages[first_preview_index]["params"]
            self.assertEqual(
                first_preview["actionId"], inline_action_ids[0]
            )
            self.assertTrue(
                first_preview["data"]["preview"]["dataUrl"].startswith(
                    "data:image/png;base64,"
                )
            )
            inline_preview = first_preview["data"]["preview"]
            self.assertEqual(inline_preview["schemaVersion"], 1)
            self.assertEqual(inline_preview["producer"], "same-brain-decoder")
            self.assertEqual(inline_preview["stage"], "diffusion-vq-decode")
            self.assertTrue(inline_preview["actualDecoderOutput"])
            self.assertFalse(inline_preview["spatialResolutionReduced"])
            inline_payload = base64.b64decode(
                inline_preview["dataUrl"].split(",", 1)[1], validate=True
            )
            self.assertEqual(
                hashlib.sha256(inline_payload).hexdigest(),
                inline_preview["payloadSha256"],
            )
            preview_events = [
                message["params"]
                for message in stream_messages
                if message.get("method") == "event"
                and message.get("params", {}).get("type")
                == "modality-preview"
                and message.get("params", {}).get("jobId") == "image-job"
            ]
            self.assertGreaterEqual(len(preview_events), 1)
            self.assertTrue(
                all(
                    event["data"]["preview"].get("mimeType") == "image/png"
                    and (
                        str(event["data"]["preview"].get("dataUrl", "")).startswith(
                            "data:image/png;base64,"
                        )
                        or str(
                            event["data"]["preview"].get("artifactPath", "")
                        ).endswith(".png")
                    )
                    for event in preview_events
                )
            )
            for event in preview_events:
                preview = event["data"]["preview"]
                self.assertEqual(preview["producer"], "same-brain-decoder")
                self.assertRegex(preview["payloadSha256"], r"^[a-f0-9]{64}$")
                if "dataUrl" in preview:
                    payload = base64.b64decode(
                        preview["dataUrl"].split(",", 1)[1], validate=True
                    )
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        preview["payloadSha256"],
                    )
            self.assertEqual(
                sorted(event["progress"] for event in preview_events),
                [event["progress"] for event in preview_events],
            )
            self.assertEqual(
                len(list((Path(folder) / "engine" / "artifacts").glob("*.png"))),
                1,
                "the typed tool job must claim the inline artifact, not generate twice",
            )
            self.assertFalse(
                (Path(folder) / "engine" / ".inline-imagination").exists()
            )
            modality_audit = next(
                event
                for event in worker.brains["rpc-brain"].events.recent(20)
                if event["kind"] == "modality-generation"
            )
            self.assertEqual(modality_audit["jobId"], "image-job")
            self.assertTrue(
                modality_audit["payload"]["generatedDuringChat"]
            )
            self.assertEqual(
                modality_audit["payload"]["neuralActionId"],
                inline_action_ids[0],
            )

            feedback = request(
                "feedback",
                "feedback",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "text": chatted["text"],
                    "direction": "up",
                    "messageId": "desktop-message",
                    "traceId": chatted["trace"]["id"],
                },
            )
            self.assertFalse(feedback["rewardModel"])
            self.assertFalse(feedback["rlhf"])
            self.assertGreater(feedback["stdp"]["stdp_update"], 0.0)

            idle = request(
                "idle",
                "idle_cycle",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "minimumIdleSeconds": 0,
                    "toolSchemas": [
                        {
                            "id": "modality.imagine",
                            "actions": ["generate"],
                            "grant": "ask",
                        }
                    ],
                },
            )
            self.assertTrue(idle["ran"])
            self.assertEqual(idle["trace"]["promptTokenCount"], 0)
            self.assertFalse(idle["trace"]["hiddenBehavioralPrompt"])

            workspace = request(
                "workspace",
                "workspace",
                {"brainId": "rpc-brain", "storagePath": folder},
            )
            self.assertEqual(workspace["brainId"], "rpc-brain")
            self.assertGreaterEqual(
                workspace["latentWorkspace"]["occupancy"], 1
            )
            self.assertFalse(workspace["hiddenBehavioralPrompt"])
            self.assertFalse(workspace["rawLongTermTextInjected"])

            first_page = request(
                "substrate-1",
                "query_substrate",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "query": {
                        "entity": "neurons",
                        "zoom": 1,
                        "pageSize": 1,
                    },
                },
            )
            self.assertEqual(first_page["entity"], "neurons")
            self.assertEqual(len(first_page["neurons"]), 1)
            self.assertGreater(first_page["totals"]["neurons"], 1)
            self.assertTrue(first_page["hasMore"])
            second_page = request(
                "substrate-2",
                "query_substrate",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "query": {
                        "entity": "neurons",
                        "zoom": 1,
                        "pageSize": 1,
                        "cursor": first_page["nextCursor"],
                    },
                },
            )
            self.assertNotEqual(
                first_page["neurons"][0]["id"],
                second_page["neurons"][0]["id"],
            )
            overview = request(
                "substrate-overview",
                "query_substrate",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "query": {"entity": "overview", "zoom": 0},
                },
            )
            self.assertGreater(len(overview["clusters"]), 0)
            self.assertEqual(overview["neurons"], [])

            state = request(
                "state",
                "state",
                {"brainId": "rpc-brain", "storagePath": folder},
            )
            self.assertEqual(state["eventLogIntegrity"], "ok")
            self.assertTrue(Path(state["files"]["core"]).is_file())
            for brain in worker.brains.values():
                brain.events.close()
            worker.brains.clear()

        shutdown = request("shutdown", "shutdown", {})
        self.assertTrue(shutdown["stopping"])
        self.assertFalse(worker.running)


if __name__ == "__main__":
    unittest.main()
