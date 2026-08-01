import contextlib
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


class WorkerProtocolTests(unittest.TestCase):
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

        partial.events.close.assert_called_once()
        load.assert_called_once_with(
            partial.storage_path,
            expected_brain_id=partial.brain_id,
        )
        self.assertIs(worker.brains[partial.brain_id], restored)

    def test_gpu_profiles_select_available_backend_without_overriding_explicit_device(self):
        worker = Worker()
        with patch("worker.torch.cuda.is_available", return_value=True):
            selected = worker._builder_config({"hardwareTier": "gpu"}, {})
        self.assertEqual(selected["device"], "cuda")

        with patch("worker.torch.cuda.is_available", return_value=False), patch.object(
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
                    event["data"]["preview"]["dataUrl"].startswith(
                        "data:image/png;base64,"
                    )
                    for event in preview_events
                )
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
