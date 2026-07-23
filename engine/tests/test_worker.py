import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from worker import Worker


class WorkerProtocolTests(unittest.TestCase):
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

            chatted = request(
                "chat",
                "chat",
                {
                    "brainId": "rpc-brain",
                    "storagePath": folder,
                    "input": "Inspect this workspace.",
                    "maxNewTokens": 2,
                    "seed": 17,
                    "toolSchemas": [
                        {
                            "id": "windows.files",
                            "actions": ["list", "read"],
                            "grant": "ask",
                        }
                    ],
                },
            )
            self.assertEqual(
                chatted["trace"]["available_tool_ids"], ["windows.files"]
            )
            self.assertFalse(chatted["trace"]["tool_schema_text_injected"])
            self.assertEqual(
                chatted["runtimeCard"]["tool_schema_channel"], "vsa-internal"
            )

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
