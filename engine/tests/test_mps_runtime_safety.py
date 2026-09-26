import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.offload import ResourcePolicy
from worker import Worker


class _MpsLabelledBrain:
    """CPU tensors labelled as MPS so the placement policy is testable on CI."""

    def __init__(self, root: Path, release: threading.Event):
        self.brain_id = "mps-inline-fixture"
        self.engine_path = root / "engine"
        self.engine_path.mkdir(parents=True)
        self.device = torch.device("mps")
        self.device_backend = "mps"
        self.modalities = SimpleNamespace(image=torch.nn.Linear(4, 4))
        self.liquid_state = torch.ones(1, 4)
        self.resource_policy = SimpleNamespace(include_accelerator_memory=True)
        self.counters = {"inference_count": 1}
        self.modality_training = {"image": 1}
        self.installed_modality_packs = []
        self.release = release
        self.started = threading.Event()
        self.observed = {}

    def _modality_idea(self, _prompt, _concept_ids):
        return torch.ones(1, 4)

    def _modality_idea_evidence(self, _prompt, _concept_ids):
        return {"source": "active-working-memory", "sameBrain": True}

    def generate_modality(self, **values):
        self.observed.update(
            {
                "device": self.device.type,
                "backend": self.device_backend,
                "decoderDevice": next(self.modalities.parameters()).device.type,
                "ideaDevice": self._modality_idea().device.type,
                "acceleratorReadEnabled": (
                    self.resource_policy.include_accelerator_memory
                ),
                "evidence": self._modality_idea_evidence(),
            }
        )
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release inline generation")
        values["preview_callback"](
            1.0,
            "image/png",
            b"\x89PNG\r\n\x1a\nfixture",
            {
                "stage": "diffusion-vq-decode",
                "completedUnits": 1,
                "totalUnits": 1,
            },
        )
        artifact = self.engine_path / "artifacts" / "fixture.png"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        return {"path": str(artifact), "mimeType": "image/png"}


class MpsRuntimeSafetyTests(unittest.TestCase):
    def test_concurrent_inline_snapshot_is_entirely_cpu_owned_for_mps(self):
        worker = Worker()
        release = threading.Event()
        previews = []
        try:
            with tempfile.TemporaryDirectory(prefix="omni-mps-inline-") as folder:
                brain = _MpsLabelledBrain(Path(folder), release)
                record = worker._start_inline_generation(
                    brain,
                    "a" * 32,
                    "mps-stream",
                    {
                        "kind": "imagine",
                        "toolId": "modality.imagine",
                        "action": "generate",
                        "arguments": {"modality": "image"},
                    },
                    lambda _record, preview: previews.append(preview),
                )
                self.assertIsNotNone(record)
                assert record is not None and record.future is not None
                self.assertTrue(brain.started.wait(timeout=5))
                self.assertFalse(record.future.done())
                self.assertEqual(record.execution_device, "cpu")
                self.assertTrue(record.authoritative_accelerator_isolated)
                release.set()
                result = record.future.result(timeout=5)
                self.assertEqual(result["inlineExecutionDevice"], "cpu")
                self.assertTrue(result["authoritativeAcceleratorIsolated"])
                self.assertEqual(
                    brain.observed,
                    {
                        "device": "cpu",
                        "backend": "cpu",
                        "decoderDevice": "cpu",
                        "ideaDevice": "cpu",
                        "acceleratorReadEnabled": False,
                        "evidence": {
                            "source": "active-working-memory",
                            "sameBrain": True,
                        },
                    },
                )
                self.assertEqual(previews[-1]["executionDevice"], "cpu")
                self.assertTrue(
                    previews[-1]["authoritativeAcceleratorIsolated"]
                )
        finally:
            release.set()
            worker._shutdown_inline_generations()

    def test_inline_cpu_policy_never_queries_accelerator_allocator(self):
        with tempfile.TemporaryDirectory(prefix="omni-cpu-policy-") as folder:
            policy = ResourcePolicy(
                Path(folder), include_accelerator_memory=False
            )
            with patch("omni_core.offload._accelerator_memory") as accelerator:
                reading = policy.readings()
            accelerator.assert_not_called()
            self.assertIsNone(reading.accelerator_total_memory_bytes)
            self.assertIsNone(reading.accelerator_free_memory_bytes)
            self.assertIsNone(reading.accelerator_allocated_memory_bytes)

    def test_runtime_has_no_explicit_mps_graph_cache_clear(self):
        sources = [ENGINE / "worker.py", *sorted((ENGINE / "omni_core").glob("*.py"))]
        unsafe_call = "torch.mps." + "empty_cache("
        offenders = [
            str(path.relative_to(ENGINE))
            for path in sources
            if unsafe_call in path.read_text("utf-8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
