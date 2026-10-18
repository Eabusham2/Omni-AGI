"""Tiny event-ring and source-hook contracts; no LIF/brain construction or run."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.router_spike_capture import RouterSpikeCapture, control_router_spikes, query_router_spikes, retire_population_spike_capture, retire_router_spike_capture


class Lease:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


class Population:
    neurons = 8


class RouterSpikeCaptureTests(unittest.TestCase):
    def capture(self, population=None, **options):
        population = population or Population()
        lease = Lease()
        reservations = []

        def reserve(size, operation):
            reservations.append((size, operation))
            return lease

        capture = RouterSpikeCapture("b", population, "one", options.pop("start", 2), options.pop("count", 3), reserve, **options)
        self.addCleanup(capture.close)
        return capture, lease, reservations, population

    def test_uses_actual_binary_admitted_range_without_full_tensor_copy(self):
        population = Population()
        population.neurons = 1024
        capture, lease, reservations, _ = self.capture(population, start=700, count=5)
        spikes = torch.zeros(1024)
        spikes[701] = 1
        before = spikes.clone()
        transferred = []
        actual_to = torch.Tensor.to

        def counted_to(value, *args, **kwargs):
            if kwargs.get("device") == "cpu":
                transferred.append(value.numel())
            return actual_to(value, *args, **kwargs)

        with patch.object(torch.Tensor, "to", counted_to):
            capture.record_hard_spikes(spikes)
        page = capture.snapshot()
        self.assertEqual(transferred, [5])
        self.assertTrue(torch.equal(spikes, before))
        self.assertEqual(page["frames"][0]["spikes"], [0, 1, 0, 0, 0])
        self.assertEqual(page["frames"][0]["channelStart"], 700)
        self.assertTrue(page["frames"][0]["monotonicNs"].isdigit())
        self.assertGreater(reservations[0][0], page["ringBytes"])
        self.assertFalse(page["semanticNeuronMapping"])
        self.assertFalse(page["synapseFiringObserved"])
        self.assertEqual(lease.releases, 0)

    def test_ring_is_packed_bounded_and_reports_real_drops_sequence_and_epoch(self):
        capture, _, _, _ = self.capture(start=0, count=4, ring_bytes=72)
        self.assertEqual(capture.capacity, 2)
        for _ in range(4):
            capture.record_hard_spikes(torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.]))
        page = capture.snapshot(page_size=1)
        self.assertEqual((page["ringBytes"], page["droppedFrames"], page["missedFrames"]), (66, 2, 2))
        self.assertEqual([frame["sequence"] for frame in page["frames"]], [3])
        self.assertTrue(page["hasMore"])
        rest = capture.snapshot(epoch=page["epoch"], after=page["nextSequence"])
        self.assertEqual([frame["sequence"] for frame in rest["frames"]], [4])
        self.assertFalse(rest["hasMore"])
        with self.assertRaisesRegex(ValueError, "epoch changed"):
            capture.snapshot(epoch="0" * 32)
        with self.assertRaisesRegex(ValueError, "ahead"):
            capture.snapshot(after=5)

    def test_nonbinary_payload_retires_only_observer_without_raising_into_neural_step(self):
        capture, lease, _, _ = self.capture()
        spikes = torch.tensor([0., 0., 0.7, 1., 0., 0., 0., 0.])
        capture.record_hard_spikes(spikes)
        page = capture.snapshot()
        self.assertFalse(page["enabled"])
        self.assertEqual(page["frames"], [])
        self.assertIn("not binary", page["reason"])
        self.assertEqual(lease.releases, 1)
        self.assertEqual(spikes[2].item(), torch.tensor(0.7).item())

    def test_resource_refusal_preserves_previous_capture_and_exact_owning_cleanup(self):
        leases = []
        reject = False

        def reserve(*_args):
            if reject:
                raise RuntimeError("RAM refused")
            lease = Lease(); leases.append(lease); return lease

        population = Population()
        brain = SimpleNamespace(brain_id="b", router=SimpleNamespace(population=population), resource_policy=SimpleNamespace(reserve_ram=reserve))
        first = query_router_spikes(brain, "b", {"captureId": "first", "enabled": True, "start": 2, "count": 3})
        original = population._native_spike_observer
        reject = True
        with self.assertRaisesRegex(RuntimeError, "RAM refused"):
            query_router_spikes(brain, "b", {"captureId": "new", "enabled": True})
        self.assertIs(population._native_spike_observer, original)
        self.assertFalse(original.closed)
        reject = False
        second = query_router_spikes(brain, "b", {"captureId": "new", "enabled": True})
        self.assertEqual(leases[0].releases, 1)
        query_router_spikes(brain, "b", {"captureId": "first", "enabled": False, "epoch": first["epoch"]})
        self.assertFalse(population._native_spike_observer.closed)
        with self.assertRaisesRegex(ValueError, "epoch changed"):
            query_router_spikes(brain, "b", {"captureId": "new", "enabled": True, "epoch": first["epoch"]})
        query_router_spikes(brain, "b", {"captureId": "new", "enabled": False, "epoch": second["epoch"]})
        self.assertIsNone(population._native_spike_observer)
        self.assertEqual(leases[1].releases, 1)

    def test_reset_growth_close_retire_observer_and_loaded_identity_is_required(self):
        capture, lease, _, population = self.capture()
        population._native_spike_observer = capture
        retire_population_spike_capture(population, "activity-reset")
        self.assertIsNone(population._native_spike_observer)
        self.assertEqual(lease.releases, 1)
        capture.close()
        self.assertEqual(lease.releases, 1)
        self.assertEqual(capture.snapshot()["ringBytes"], 0)
        unloaded = query_router_spikes(None, "b", {"captureId": "one", "enabled": True})
        self.assertFalse(unloaded["available"])
        self.assertIn("not-loaded", unloaded["reason"])
        with self.assertRaisesRegex(ValueError, "brain identity"):
            query_router_spikes(SimpleNamespace(brain_id="other"), "b", {"captureId": "one", "enabled": True})

    def test_silent_actual_frames_are_observed_but_do_not_invent_spikes(self):
        capture, _, _, _ = self.capture()
        capture.record_hard_spikes(torch.zeros(8))
        page = capture.snapshot()
        self.assertTrue(page["observed"])
        self.assertEqual(page["frames"][0]["spikes"], [0, 0, 0])
        self.assertFalse(page["synapseFiringObserved"])

    def test_live_control_reads_only_owned_bytes_and_never_allocates_or_uses_tensor_code(self):
        population = Population()
        lease = Lease()
        brain = SimpleNamespace(brain_id="b", router=SimpleNamespace(population=population),
                                resource_policy=SimpleNamespace(reserve_ram=lambda *_args: lease))
        started = query_router_spikes(brain, "b", {"captureId": "live", "enabled": True, "start": 3, "count": 2})
        brain._router_spike_capture.record_hard_spikes(torch.tensor([0., 0., 0., 1., 0., 0., 0., 0.]))
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("control cannot run tensor operations")), \
                patch("omni_core.router_spike_capture.RouterSpikeCapture", side_effect=AssertionError("control cannot allocate a capture")):
            polled = control_router_spikes(brain, "b", {"captureId": "live", "enabled": True, "epoch": started["epoch"]})
            self.assertEqual(polled["frames"][0]["spikes"], [1, 0])
            control_router_spikes(brain, "b", {"captureId": "other", "enabled": False, "epoch": started["epoch"]})
            self.assertFalse(brain._router_spike_capture.closed)
            stopped = control_router_spikes(brain, "b", {"captureId": "live", "enabled": False, "epoch": started["epoch"], "afterSequence": 1})
            self.assertFalse(stopped["enabled"])
        self.assertEqual(lease.releases, 1)

    def test_source_hook_uses_hard_spikes_before_surrogate_and_retires_on_growth_reset_close(self):
        source = (Path(__file__).parents[1] / "omni_core" / "spiking.py").read_text()
        hook = source.index("observer.record_hard_spikes(spikes)")
        self.assertLess(source.index("self.spike_count.add_(spikes)"), hook)
        self.assertLess(hook, source.index("soft_spike ="))
        self.assertIn('retire_population_spike_capture(self, "LIF-activity-reset")', source)
        self.assertIn('retire_population_spike_capture(self.population, "LIF-population-grown")', source)
        brain_source = (Path(__file__).parents[1] / "omni_core" / "brain.py").read_text()
        self.assertIn('retire_router_spike_capture(self, "brain-closed-or-unloaded")', brain_source)

    def test_retirement_rechecks_exact_capture_before_releasing_a_replaced_owner(self):
        population = Population()
        leases = []

        def reserve(*_args):
            lease = Lease(); leases.append(lease); return lease

        brain = SimpleNamespace(brain_id="b", router=SimpleNamespace(population=population), resource_policy=SimpleNamespace(reserve_ram=reserve))
        query_router_spikes(brain, "b", {"captureId": "old", "enabled": True})
        old = brain._router_spike_capture
        query_router_spikes(brain, "b", {"captureId": "new", "enabled": True})
        replacement = brain._router_spike_capture
        retire_router_spike_capture(brain, "late-old-control", expected_capture=old)
        self.assertIs(population._native_spike_observer, replacement)
        self.assertFalse(replacement.closed)
        self.assertEqual(leases[1].releases, 0)
        retire_router_spike_capture(brain, expected_capture=replacement)
        self.assertEqual(leases[1].releases, 1)

    def test_worker_control_checks_storage_and_stays_in_non_neural_dispatch_branch(self):
        from worker import RpcFault, Worker
        population = Population()
        lease = Lease()
        brain = SimpleNamespace(brain_id="b", storage_path=Path("/private/owned-b"), router=SimpleNamespace(population=population),
                                resource_policy=SimpleNamespace(reserve_ram=lambda *_args: lease))
        started = query_router_spikes(brain, "b", {"captureId": "worker", "enabled": True})
        request = {"jsonrpc": "2.0", "id": "poll", "method": "router_activity_control", "params": {
            "brainId": "b", "storagePath": "/private/owned-b", "query": {"captureId": "worker", "enabled": True, "epoch": started["epoch"]}}}
        session = SimpleNamespace(worker_role="neural", brains={"b": brain},
                                  _brain_id=lambda params: params["brainId"], _storage=lambda params, _brain_id: Path(params["storagePath"]))
        session.router_activity_control = lambda params: Worker.router_activity_control(session, params)
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("control must remain bytes-only")):
            response = Worker.dispatch_control(session, request)
        self.assertEqual(response["id"], "poll")
        self.assertEqual(response["result"]["frames"], [])
        with self.assertRaisesRegex(RpcFault, "storage identity"):
            Worker.router_activity_control(session, {**request["params"], "storagePath": "/private/other"})
        retire_router_spike_capture(brain)


if __name__ == "__main__":
    unittest.main()
