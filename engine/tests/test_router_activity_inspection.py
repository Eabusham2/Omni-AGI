"""Exact router-contact inspection without constructing a brain or LIF model."""
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.packed_connections import encode_connection_state
from omni_core.router_activity_inspection import capture_router_contacts
from omni_core.router_state_paging import pack_tile, unpack_tile
from omni_core.router_spike_capture import control_router_spikes, query_router_spikes, retire_router_spike_capture
from omni_core.sparse_router_state import SparseRouterState, sparse_layout_digest


class Lease:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


class Population:
    neurons = 4


def block(levels, *, states=None, eligibility=None, stability=None, uses=None):
    rows, columns = levels.shape
    return SimpleNamespace(
        rows=rows,
        columns=columns,
        _router_unpublished=False,
        _packed_weights=pack_tile(levels),
        _connection_state=states if states is not None else torch.zeros((rows, columns), dtype=torch.uint8),
        eligibility_accumulator=eligibility if eligibility is not None else torch.zeros((rows, columns), dtype=torch.int16),
        stability=stability if stability is not None else torch.zeros((rows, columns), dtype=torch.float32),
        uses=uses if uses is not None else torch.zeros((rows, columns), dtype=torch.float32),
    )


class RouterActivityInspectionTests(unittest.TestCase):
    def fixture(self):
        levels = torch.zeros((4, 4), dtype=torch.int8)
        levels[0, 0] = 1  # deliberately outside the selected 1..3 viewport
        levels[1, 1] = 1
        levels[2, 2] = -1
        strengths = torch.ones((4, 4), dtype=torch.uint8)
        resistance = torch.zeros((4, 4), dtype=torch.uint8)
        strengths[1, 1], resistance[1, 1] = 3, 2
        strengths[3, 1], resistance[3, 1] = 4, 5
        state = encode_connection_state(strengths, resistance)
        eligibility = torch.zeros((4, 4), dtype=torch.int16)
        stability = torch.zeros((4, 4), dtype=torch.float32)
        uses = torch.zeros((4, 4), dtype=torch.float32)
        eligibility[2, 2], stability[1, 1], uses[3, 1] = 7, 0.5, 2
        repeated = torch.zeros((4, 4), dtype=torch.int8)
        repeated[1, 1] = -1
        owner = SimpleNamespace(pre_neurons=4, post_neurons=4, blocks={
            "r0_c0": block(levels, states=state, eligibility=eligibility, stability=stability, uses=uses),
            "r0_c0_p1": block(repeated),
        })
        releases = []
        sparse = SimpleNamespace(owner=owner, ready=True, _generation=7,
                                 release=lambda value, region: releases.append((value, region)))
        checks = []
        synapses = SimpleNamespace(
            pre_neurons=4, post_neurons=4, _sparse_state=sparse,
            _operation=lambda: nullcontext(), _ram=lambda *_args: nullcontext(),
            _check=lambda: checks.append(True),
        )
        leases, reservations = [], []

        def reserve(size, operation):
            lease = Lease(); leases.append(lease); reservations.append((size, operation)); return lease

        brain = SimpleNamespace(
            brain_id="brain-a",
            router=SimpleNamespace(neurons=4, population=Population(), synapses=synapses),
            resource_policy=SimpleNamespace(reserve_ram=reserve),
        )
        return brain, leases, reservations, releases, checks

    def test_pages_exact_selected_contacts_and_preserves_parallel_identity(self):
        brain, leases, reservations, releases, checks = self.fixture()
        snapshot = capture_router_contacts(brain, "brain-a", 1, 3)
        first = snapshot.page(0, 2)
        second = snapshot.page(2, 2)

        self.assertTrue(first["available"])
        self.assertEqual((first["topologyGeneration"], first["totalContacts"]), (7, 4))
        self.assertGreater(first["capturedAtUnix"], 0)
        self.assertEqual((first["returned"], first["hasMore"]), (2, True))
        self.assertEqual((second["returned"], second["hasMore"]), (2, False))
        contacts = first["contacts"] + second["contacts"]
        self.assertEqual([(item["targetChannel"], item["sourceChannel"]) for item in contacts],
                         [(1, 1), (2, 2), (3, 1), (1, 1)])
        self.assertNotIn(0, [item["sourceChannel"] for item in contacts])
        self.assertEqual(contacts[0]["effectiveContribution"], 3)
        self.assertEqual((contacts[0]["connectionStrength"], contacts[0]["connectionResistance"]), (3, 2))
        self.assertEqual(contacts[1]["effectiveContribution"], -1)
        self.assertEqual(contacts[2]["effectiveWeight"], 0)
        self.assertEqual((contacts[2]["connectionStrength"], contacts[2]["connectionResistance"]), (4, 5))
        self.assertEqual(contacts[3]["parallelLayer"], 1)
        self.assertTrue(contacts[3]["isRepeatedContact"])
        self.assertEqual(contacts[3]["parallelRootId"], contacts[0]["id"])
        self.assertTrue(all(item["firingObserved"] is False for item in contacts))
        self.assertFalse(first["synapseFiringObserved"])
        # The selected viewport began at an unaligned column. Both retained
        # reads start at the packed byte boundary without exposing channel 0.
        self.assertEqual([region for _, region in releases], [(1, 4, 0, 4), (1, 4, 0, 4)])
        self.assertEqual(len(checks), 2)
        self.assertGreater(reservations[0][0], 0)
        self.assertEqual(leases[0].releases, 0)
        snapshot.close()
        self.assertEqual(leases[0].releases, 1)
        self.assertFalse(snapshot.page()["available"])

    def test_missing_owner_is_explicit_and_resource_refusal_does_not_publish(self):
        absent = SimpleNamespace(router=SimpleNamespace(neurons=4, synapses=None))
        unavailable = capture_router_contacts(absent, "brain-a", 0, 4).page()
        self.assertFalse(unavailable["available"])
        self.assertIn("no-exact-loaded", unavailable["reason"])

        brain, _, _, _, _ = self.fixture()
        brain.resource_policy.reserve_ram = lambda *_args: (_ for _ in ()).throw(RuntimeError("RAM refused"))
        with self.assertRaisesRegex(RuntimeError, "RAM refused"):
            capture_router_contacts(brain, "brain-a", 0, 4)

        brain, _, _, _, _ = self.fixture()
        contact_lease, calls = Lease(), 0

        def refuse_ring(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return contact_lease
            raise RuntimeError("ring RAM refused")

        brain.resource_policy.reserve_ram = refuse_ring
        with self.assertRaisesRegex(RuntimeError, "ring RAM refused"):
            query_router_spikes(brain, "brain-a", {"captureId": "refused", "enabled": True})
        self.assertEqual(contact_lease.releases, 1)
        self.assertIsNone(getattr(brain, "_router_spike_capture", None))
        self.assertIsNone(getattr(brain.router.population, "_native_spike_observer", None))

    def test_activity_epoch_pages_the_immutable_contacts_without_tensor_work_on_poll(self):
        brain, leases, _, _, _ = self.fixture()
        initial = query_router_spikes(brain, "brain-a", {
            "captureId": "owned", "enabled": True, "start": 1, "count": 3,
            "contactOffset": 0, "contactPageSize": 2,
        })
        contacts = initial["contactSnapshot"]
        self.assertEqual((contacts["offset"], contacts["returned"], contacts["hasMore"]), (0, 2, True))
        captured_at = contacts["capturedAtUnix"]
        observer = brain._router_spike_capture.contact_snapshot
        block = brain.router.synapses._sparse_state.owner.blocks["r0_c0"]
        # Change operands in the same allocated block after the inspection
        # snapshot. Transmission must report these live values, not the old
        # enable-time sign/strength metadata.
        live_levels = unpack_tile(block._packed_weights, 0, 4, 0, 4)
        live_levels[1, 1] = -1
        block._packed_weights.copy_(pack_tile(live_levels))
        block._connection_state[1, 1] = (5-1) | (4 << 4)
        operation = observer.begin_recurrence(brain.router.synapses)
        observer.record_packed_tile(operation, brain.router.synapses,
            ("r0_c0", observer.layout_digest), (0, 4, 0, 4),
            torch.tensor([0., 2., 1., 0.5]), block._packed_weights,
            block._connection_state)
        observer.finish_recurrence(operation)
        brain._router_spike_capture.record_hard_spikes(torch.tensor([0., 1., 0., 1.]))
        observed = control_router_spikes(brain, "brain-a", {
            "captureId": "owned", "enabled": True, "epoch": initial["epoch"],
            "contactOffset": 0, "contactPageSize": 2,
            "transmissionEpoch": initial["transmissionPage"]["epoch"],
            "transmissionPageSize": 4,
        })["transmissionPage"]
        self.assertEqual([(event["sourceInput"], event["effectiveWeight"],
                           event["connectionStrength"], event["effectiveContribution"],
                           event["classification"]) for event in observed["events"]],
                         [(2.0, -1, 5, -10.0, "inhibitory"),
                          (1.0, -1, 1, -1.0, "inhibitory")])
        self.assertEqual((initial["contactSnapshot"]["contacts"][0]["effectiveWeight"],
                          initial["contactSnapshot"]["contacts"][0]["connectionStrength"]),
                         (1, 3))
        self.assertTrue(all(event["transmissionObserved"] for event in observed["events"]))
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("poll must use retained bytes/tuples only")):
            next_page = control_router_spikes(brain, "brain-a", {
                "captureId": "owned", "enabled": True, "epoch": initial["epoch"],
                "contactOffset": 2, "contactPageSize": 2,
            })
        self.assertEqual(next_page["frames"][0]["spikes"], [1, 0, 1])
        self.assertGreaterEqual(next_page["frames"][0]["observedAtUnix"], captured_at)
        self.assertEqual(next_page["contactSnapshot"]["capturedAtUnix"], captured_at)
        self.assertEqual((next_page["contactSnapshot"]["offset"], next_page["contactSnapshot"]["returned"]), (2, 2))
        # Paging changed the selected contacts, so the transmission epoch and
        # retained ring reset instead of mixing operands from two selections.
        self.assertNotEqual(next_page["transmissionPage"]["epoch"],
                            initial["transmissionPage"]["epoch"])
        repeated_selection = control_router_spikes(brain, "brain-a", {
            "captureId": "owned", "enabled": True, "epoch": initial["epoch"],
            "contactOffset": 3, "contactPageSize": 1,
        })
        repeated = brain.router.synapses._sparse_state.owner.blocks["r0_c0_p1"]
        repeated_operation = observer.begin_recurrence(brain.router.synapses)
        observer.record_packed_tile(repeated_operation, brain.router.synapses,
            ("r0_c0_p1", observer.layout_digest), (0, 4, 0, 4),
            torch.tensor([0., 2., 1., 0.5]), repeated._packed_weights,
            repeated._connection_state)
        observer.finish_recurrence(repeated_operation)
        repeated_page = control_router_spikes(brain, "brain-a", {
            "captureId": "owned", "enabled": True, "epoch": initial["epoch"],
            "contactOffset": 3, "contactPageSize": 1,
            "transmissionEpoch": repeated_selection["transmissionPage"]["epoch"],
            "transmissionPageSize": 4,
        })["transmissionPage"]
        self.assertEqual(len(repeated_page["events"]), 1)
        self.assertEqual(repeated_page["events"][0]["contactId"],
                         "brain-a:root.router.synapses:r0_c0_p1:1:1")
        self.assertEqual((repeated_page["events"][0]["sourceInput"],
                          repeated_page["events"][0]["effectiveWeight"],
                          repeated_page["events"][0]["effectiveContribution"],
                          repeated_page["events"][0]["classification"]),
                         (2.0, -1, -2.0, "inhibitory"))
        self.assertFalse(repeated_page["events"][0]["postsynapticFiringObserved"])
        retire_router_spike_capture(brain)
        self.assertEqual([lease.releases for lease in leases], [1, 1])

    def test_recurrent_hook_is_after_actual_packed_compute_and_before_add_or_stdp(self):
        source = (Path(__file__).parents[1] / "omni_core" / "spiking.py").read_text()
        recurrent = source[source.index("    def recurrent(self, inputs:"):source.index("    def active_synapse_count", source.index("    def recurrent(self, inputs:"))]
        self.assertLess(recurrent.index("contribution = packed_forward_float"),
                        recurrent.index("record_packed_tile"))
        self.assertLess(recurrent.index("record_packed_tile"),
                        recurrent.index("output[r0:r1].add_"))
        self.assertIn("except Exception as error", recurrent)
        self.assertIn("transmission-observer-retired", recurrent)
        route = source[source.index("        for _ in range(max(1, int(steps))):"):source.index("        activity = total_spikes")]
        self.assertLess(route.index("synapses.recurrent"), route.index("population.step"))
        self.assertLess(route.index("population.step"), route.index("synapses.step"))

    def test_packed_iterator_default_stays_three_tuple_and_identity_is_opt_in(self):
        levels = torch.zeros((4, 4), dtype=torch.int8); levels[1, 1] = 1
        value = block(levels)
        owner = SimpleNamespace(pre_neurons=4, post_neurons=4,
            blocks={"r0_c0": value}, _ram=lambda *_args: nullcontext(),
            _check=lambda: None)
        sparse = SimpleNamespace(owner=owner,
            regions=lambda _block: iter(((0, 4, 0, 4),)),
            release=lambda *_args: None)
        ordinary = list(SparseRouterState.iter_packed_tiles(sparse))
        observed = list(SparseRouterState.iter_packed_tiles(sparse, include_identity=True))
        self.assertEqual(len(ordinary[0]), 3)
        self.assertEqual(len(observed[0]), 4)
        self.assertEqual(observed[0][3][0], "r0_c0")
        self.assertRegex(observed[0][3][1], r"^[a-f0-9]{64}$")

    def test_layout_seal_tracks_live_owners_not_weight_mutations(self):
        levels = torch.zeros((4, 4), dtype=torch.int8); levels[1, 1] = 1
        first = block(levels); first._router_birth = 3
        owner = SimpleNamespace(pre_neurons=4, post_neurons=4,
                                blocks={"r0_c0": first})
        initial = sparse_layout_digest(owner, ["r0_c0"])
        first._packed_weights.copy_(pack_tile(-levels))
        self.assertEqual(sparse_layout_digest(owner, ["r0_c0"]), initial)
        replacement = block(levels); replacement._router_birth = 3
        owner.blocks["r0_c0"] = replacement
        self.assertNotEqual(sparse_layout_digest(owner, ["r0_c0"]), initial)
        replacement._router_unpublished = True
        with self.assertRaisesRegex(RuntimeError, "unpublished"):
            sparse_layout_digest(owner, ["r0_c0"])

    def test_observer_failure_on_first_tile_cannot_change_later_neural_tiles(self):
        source_path = Path(__file__).parents[1] / "omni_core" / "spiking.py"
        tree = ast.parse(source_path.read_text())
        class_node = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "STDPSynapses")
        method = next(node for node in class_node.body
                      if isinstance(node, ast.FunctionDef) and node.name == "recurrent")

        def packed_forward(activity, signs, _states, _width, _rows):
            return activity * float(signs[0, 0])

        namespace = {"torch": torch, "closing": __import__("contextlib").closing,
                     "packed_forward_float": packed_forward}
        ast.fix_missing_locations(method)
        exec(compile(ast.Module(body=[method], type_ignores=[]),
                     str(source_path), "exec"), namespace)
        recurrent = namespace["recurrent"]

        class FailingContactObserver:
            def __init__(self):
                self.records = 0
                self.retirements = 0

            def begin_recurrence(self, _owner):
                return object()

            def record_packed_tile(self, *_args):
                self.records += 1
                raise RuntimeError("inspection failed")

            def retire_transmissions(self, _reason):
                self.retirements += 1
                raise RuntimeError("retirement also failed")

            def finish_recurrence(self, _operation):
                raise AssertionError("retired observer must not finish")

        contact_observer = FailingContactObserver()
        observer = SimpleNamespace(contact_snapshot=contact_observer)
        identity_modes = []

        def tiles(*, include_identity=False):
            identity_modes.append(include_identity)
            yield ((0, 1, 0, 1), torch.tensor([[2]], dtype=torch.uint8),
                   torch.zeros((1, 1), dtype=torch.uint8), ("r0_c0", "a" * 64))
            yield ((1, 2, 1, 2), torch.tensor([[3]], dtype=torch.uint8),
                   torch.zeros((1, 1), dtype=torch.uint8), ("r0_c0", "a" * 64))

        synapses = SimpleNamespace(pre_neurons=2, post_neurons=2,
            _operation=lambda: nullcontext(), _ram=lambda *_args: nullcontext(),
            iter_packed_tiles=tiles, _native_transmission_observer=observer)
        result = recurrent(synapses, torch.tensor([1.0, 1.0]))
        self.assertTrue(torch.equal(result, torch.tensor([2.0, 3.0])))
        self.assertEqual(identity_modes, [True])
        self.assertEqual((contact_observer.records, contact_observer.retirements), (1, 1))
        self.assertIsNone(synapses._native_transmission_observer)


if __name__ == "__main__":
    unittest.main()
