import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from worker import Worker


class WorkerProtocolTests(unittest.TestCase):
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
            self.assertNotIn("error", streamed_response)
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

            preview_output = io.StringIO()
            with contextlib.redirect_stdout(preview_output):
                modality_response = worker.dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": "image-preview",
                        "method": "generate_modality",
                        "params": {
                            "brainId": "rpc-brain",
                            "storagePath": folder,
                            "jobId": "image-job",
                            "modality": "image",
                            "conceptIds": ["organic-fixture"],
                            "seed": 23,
                        },
                    }
                )
            self.assertNotIn("error", modality_response)
            preview_events = [
                message["params"]
                for message in (
                    json.loads(line)
                    for line in preview_output.getvalue().splitlines()
                    if line.strip()
                )
                if message.get("method") == "event"
                and message.get("params", {}).get("type")
                == "modality-preview"
            ]
            self.assertGreaterEqual(len(preview_events), 2)
            self.assertEqual(
                [event["sequence"] for event in preview_events],
                list(range(len(preview_events))),
            )
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
