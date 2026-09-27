"""Worker lease integration without constructing an OmniCortex model."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from worker import Worker
from omni_core.brain_lease import BrainOwnerLease


class WorkerBrainLeaseTests(unittest.TestCase):
    def test_load_holds_exclusive_lease_until_unload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omni-worker-lease-") as folder:
            storage = Path(folder) / "brain-a"
            engine = storage / "engine"
            engine.mkdir(parents=True)
            (engine / "brain.json").write_text("{}", encoding="utf-8")
            worker = Worker()
            fake_brain = SimpleNamespace(
                brain_id="brain-a", storage_path=storage.resolve(),
                close=mock.Mock(),
            )
            params = {"brainId": "brain-a", "storagePath": str(storage)}
            try:
                with (
                    mock.patch.object(worker, "_clear_inline_staging"),
                    mock.patch.object(worker, "_clear_preview_cache"),
                    mock.patch.object(worker, "_prune_verified_abandoned_live_caches"),
                    mock.patch("worker.AdaptiveBrain.load", return_value=fake_brain) as load,
                ):
                    self.assertIs(worker._get(params), fake_brain)
                    lease = worker._brain_leases["brain-a"]
                    lease.assert_exclusive()
                    self.assertIs(worker._get(params), fake_brain)
                    load.assert_called_once()
                    worker.unload(params, None)
                    fake_brain.close.assert_called_once()
                    self.assertNotIn("brain-a", worker._brain_leases)
            finally:
                worker._shutdown_inline_generations()
                worker._release_all_owner_leases()

    def test_prune_callback_requires_lease_and_no_local_brain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omni-prune-lease-") as folder:
            storage = Path(folder) / "brain-a"
            engine = storage / "engine"
            child = engine / "state" / "live-substrate-cache" / ("a" * 32)
            child.mkdir(parents=True)
            worker = Worker()
            lease = BrainOwnerLease(storage).acquire()
            worker._brain_leases["brain-a"] = lease
            brain = SimpleNamespace(
                brain_id="brain-a", engine_path=engine,
                _verified_paged_rebuild=object(),
                _live_paging_cache_directory=child.parent / ("b" * 32),
            )
            observed = []

            def pretend_prune(cache_directory, _engine, *, no_consumers, **_kwargs):
                observed.append(no_consumers(cache_directory / "live-paged"))
                return {"removed": False}

            try:
                with mock.patch(
                    "omni_core.live_paging_migration.prune_abandoned_live_paging_cache",
                    side_effect=pretend_prune,
                ):
                    worker._prune_verified_abandoned_live_caches("brain-a", brain, lease)
                    self.assertTrue(observed)
                    self.assertTrue(all(observed))
                    observed.clear()
                    worker.brains["brain-a"] = brain
                    worker._prune_verified_abandoned_live_caches("brain-a", brain, lease)
                    self.assertTrue(observed)
                    self.assertFalse(any(observed))
            finally:
                worker.brains.clear()
                worker._shutdown_inline_generations()
                worker._release_all_owner_leases()


if __name__ == "__main__":
    unittest.main()
