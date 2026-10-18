"""New stdlib-only reply executor contracts, not a learned-brain evaluation."""
from __future__ import annotations

import ast
import copy
import importlib
import sqlite3
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "owned_reply_execution_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "omni_core")]
sys.modules[PACKAGE] = package
P = importlib.import_module(PACKAGE + ".reply_execution_planner")
S = importlib.import_module(PACKAGE + ".whole_path_scheduler")
R = importlib.import_module(PACKAGE + ".operation_auto_registry")
W = importlib.import_module(PACKAGE + ".whole_path_execution")


class Fixture:
    """Only inert owner metadata, Events and a real tiny SQLite effect ledger."""
    def __init__(self):
        self.brain = types.SimpleNamespace(brain_id="brain", decoder=object())
        self.binding_value = {"ownerSessionId": "session", "policySha256": "1" * 64, "budgetSha256": "4" * 64}
        self.active, self.clock_value = True, 1
        self.calls, self.rows, self.leases = [], [], []
        self.retained, self.refill = True, False
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.execute("CREATE TABLE effects(value TEXT)")
        self.controller = types.SimpleNamespace(ops=types.SimpleNamespace(
            cached_operations=types.SimpleNamespace(valid_frontier=lambda *_: self.retained),
            prefill_operations=types.SimpleNamespace(valid_prefix=lambda *_: self.refill)))
        self.brain._native_operation_auto = self.controller
        self.scheduler = S.WholePathScheduler(self, clock_ns=self.clock)
        self.cancel = threading.Event()
        self.session = types.SimpleNamespace(brain_id="brain", stream_id="turn", requested=threading.Event(),
            suspension_requested=threading.Event())
        self.sessions = {("brain", "turn"): self.session}
    def clock(self): return self.clock_value
    def advance(self, count): self.clock_value += count
    def armed(self): return self.active
    def binding(self): return copy.deepcopy(self.binding_value)
    def owns(self, owner): return owner is self.brain
    def frontier(self): return {"changes": self.db.total_changes}
    def reserve_metadata(self, _size):
        lease = types.SimpleNamespace(closed=False)
        lease.release = lambda: setattr(lease, "closed", True)
        self.leases.append(lease)
        return lease
    def admit_effect(self, stage): self.calls.append(("admit", stage))
    def drain_prior_effect(self, stage): self.calls.append(("drain", stage))
    def complete_effect(self, stage): self.calls.append(("complete", stage))
    def record(self, row): self.rows.append(copy.deepcopy(row))
    def admit_selection(self, identity, candidate):
        if not self.active or identity["policySha256"] != self.binding_value["policySha256"]:
            raise RuntimeError("actual selection policy retired")
        self.calls.append(("selection-admission", candidate["planSha256"]))
    @contextmanager
    def resource_observation(self):
        first = [True]
        def sample():
            result = {"budgetSha256": "4" * 64, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1000, "observedPeakBytes": 100,
                    "peakObservation": "operation-start" if first[0] else "sampled-full-operation"}}}
            first[0] = False
            return result
        yield sample
    @contextmanager
    def scope(self):
        with self.scheduler.request("chat", "request"):
            controls = P.bind_reply_controls(self.brain, "request", "turn", sessions=self.sessions,
                key=("brain", "turn"), session=self.session, cancellation=self.cancel)
            try: yield controls
            finally: P.retire_reply_controls(self.brain, controls)
    def sample(self, *, cache=object(), **changes):
        def consume(name):
            def run(_decoder, **kwargs):
                self.calls.append((name, kwargs))
                kwargs["before_sample"]("actual-cache", "actual-neural-state")
                return name
            return run
        args = dict(use_cache=True, resumed=False, allowed=True, window="window", cue="cue", cache=cache,
            generator="same-home-generator", sampling={"temperature": .7}, flags_cancelled=lambda: False,
            before_sample=lambda *_: self.db.execute("INSERT INTO effects VALUES('boundary')"),
            cached=consume("retained"), prefix=consume("prefix"), one_shot=consume("one-shot"),
            ordinary=lambda: self.calls.append(("ordinary", S.qualification_allowed())) or "ordinary",
            ordinary_after_boundary=lambda: self.calls.append(("after-boundary",)) or "after-boundary")
        args.update(changes)
        return P.dispatch_reply_sample(self.brain.decoder, **args)


class ReplyPlannerCases(unittest.TestCase):
    def fixture(self):
        value = Fixture(); self.addCleanup(value.db.close)
        return value

    def test_real_dispatch_chooses_retained_or_initial_refill_consumer_and_one_boundary(self):
        for retained, refill, cache, expected in ((True, False, object(), "retained"),
                (False, True, None, "prefix"), (False, True, object(), "prefix")):
            with self.subTest(expected=expected, initial=cache is None):
                f = self.fixture(); f.retained, f.refill = retained, refill
                @P.owned_reply
                def reply(brain): return f.sample(cache=cache)
                with f.scope(): self.assertEqual(reply(f.brain), expected)
                self.assertEqual(f.db.total_changes, 1)
                self.assertEqual(f.rows[-1]["replyExecution"]["tokenFrontiersDispatched"], 1)

    def test_explicit_uncached_keeps_original_arithmetic_and_never_calls_cached_consumer(self):
        f = self.fixture()
        @P.owned_reply
        def reply(brain): return f.sample(use_cache=False, allowed=False)
        with f.scope(): self.assertEqual(reply(f.brain), "ordinary")
        self.assertIn(("ordinary", True), f.calls)
        self.assertFalse(any(row[0] in {"prefix", "retained", "one-shot"} for row in f.calls))

    def test_unknown_schedule_or_control_namespace_is_ordinary_not_inferred_authority(self):
        for changed in ("missing-controls", "resumed", "unknown-stage", "policy", "stopped", "foreign-decoder"):
            with self.subTest(changed=changed):
                f = self.fixture()
                @P.owned_reply
                def reply(brain):
                    if changed == "unknown-stage":
                        with P.reply_stage(brain, "new-unreviewed-thing"): pass
                    if changed == "policy": f.binding_value["policySha256"] = "9" * 64
                    if changed == "stopped": f.active = False
                    if changed == "foreign-decoder": brain.decoder = object()
                    return f.sample(resumed=changed == "resumed")
                with f.scope():
                    if changed == "missing-controls": del f.brain._native_reply_private_flags
                    # An owner replacement is not a supported route. All other
                    # cases preserve ordinary execution rather than retrying it.
                    if changed == "foreign-decoder":
                        old = f.brain.decoder
                        f.sample = lambda **_: P.dispatch_reply_sample(old, use_cache=True, resumed=False, allowed=True,
                            window=None, cue=None, cache=None, cached=lambda *_a, **_k: "bad",
                            prefix=lambda *_a, **_k: "bad", one_shot=lambda *_a, **_k: "bad",
                            ordinary=lambda: "ordinary", ordinary_after_boundary=lambda: "bad")
                    self.assertEqual(reply(f.brain), "ordinary")

    def test_world_callback_protocol_and_error_are_once_only_and_private_entry_is_blocked(self):
        f = self.fixture()
        class Inbox:
            wait_seconds = 30
            def __call__(self):
                self.check(); return "input"
            def wait_pending(self):
                self.check(); return "result"
            def check(self):
                if S.qualification_allowed(): raise AssertionError("callback could privately qualify")
                f.db.execute("INSERT INTO effects VALUES('world')")
        wrapped = P.world_callback(Inbox())
        self.assertEqual(wrapped.wait_seconds, 30)
        self.assertEqual(wrapped(), "input")
        self.assertEqual(wrapped.wait_pending(), "result")
        token = R.PRIVATE_OPERATION.set(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "real world callback"): wrapped()
        finally: R.PRIVATE_OPERATION.reset(token)
        self.assertEqual(f.db.total_changes, 2)
        def fail():
            f.db.execute("INSERT INTO effects VALUES('before-error')")
            raise ValueError("world failed after commit")
        with self.assertRaisesRegex(ValueError, "after commit"): P.world_callback(fail)()
        self.assertEqual(f.db.total_changes, 3)
        self.assertTrue(S.qualification_allowed())

    def test_real_pinned_heads_arguments_Ponder_learning_are_admitted_and_never_privately_repeated(self):
        f = self.fixture()
        @P.owned_reply
        def reply(brain):
            for name in ("action-readout", "structured-action-arguments", "voluntary-ponder", "reply-finalization"):
                with P.reply_stage(brain, name):
                    self.assertFalse(S.qualification_allowed())
                    f.db.execute("INSERT INTO effects VALUES(?)", (name,)); f.advance(10)
        with f.scope(): reply(f.brain)
        self.assertEqual(f.db.total_changes, 4)
        self.assertEqual(f.rows[-1]["fixedMemoryNanoseconds"], "40")
        self.assertEqual(f.rows[-1]["effectsObserved"], 4)
        self.assertFalse(f.rows[-1]["replyExecution"]["wholeReplySpeedupMeasured"])
        self.assertTrue(all(lease.closed for lease in f.leases))

    def test_real_sampler_or_boundary_failure_does_not_fallback_and_repeat_world_effects(self):
        f = self.fixture()
        def selected(_decoder, **kwargs):
            kwargs["before_sample"]()
            raise ValueError("selected path failed after real boundary")
        @P.owned_reply
        def reply(brain):
            return f.sample(cached=selected, before_sample=P.world_callback(lambda: f.db.execute("INSERT INTO effects VALUES('once')")))
        with self.assertRaisesRegex(ValueError, "after real boundary"), f.scope(): reply(f.brain)
        self.assertEqual(f.db.total_changes, 1)
        self.assertFalse(any(row[0] == "ordinary" for row in f.calls))
        self.assertIsNone(P.current_reply_planner())
        self.assertIsNone(S.current_owned_path())

    def test_exact_Event_binding_detects_turn_owner_request_and_Event_replacement_without_resetting(self):
        f = self.fixture()
        with f.scope() as controls:
            self.assertTrue(controls.current())
            self.assertFalse(controls.is_set())
            f.session.requested.set(); self.assertTrue(controls.is_set())
            f.session.requested = threading.Event(); self.assertFalse(controls.current())
            self.assertTrue(controls.events[1].is_set())
            self.assertIsNone(P.bind_reply_controls(f.brain, "other-request", "turn", sessions=f.sessions,
                key=("brain", "turn"), session=f.session, cancellation=f.cancel))
            replacement = object(); f.brain._native_reply_private_flags = replacement
            P.retire_reply_controls(f.brain, controls)
            self.assertIs(f.brain._native_reply_private_flags, replacement)
        self.assertTrue(controls.events[1].is_set())

    def test_saved_or_finished_choice_cannot_recreate_current_request_authority(self):
        f = self.fixture()
        identity = {"ownerId": "brain", **f.binding_value, "workKind": R.CACHED_KINDS[0]}
        with f.scope():
            with S.owned_operation_choice(f.controller, identity, {"planSha256": "a" * 64}) as choice:
                self.assertIs(S.current_operation_choice(), choice)
                self.assertIs(choice["controller"], f.controller)
                f.advance(5)
            saved = dict(choice)
            self.assertIsNone(S.current_operation_choice())
        self.assertIsNone(S.current_operation_choice())
        with f.scope():
            token = S._CHOICE.set(saved)
            try: self.assertIsNone(S.current_operation_choice())
            finally: S._CHOICE.reset(token)
        self.assertEqual(f.rows[0]["actualPreparedOperations"][R.CACHED_KINDS[0]]["completed"], 1)

    def test_nested_reply_scope_and_callback_failure_do_not_leak_qualification_or_planner(self):
        f = self.fixture()
        @P.owned_reply
        def inner(brain):
            self.assertTrue(S.qualification_allowed())
            raise ValueError("nested")
        @P.owned_reply
        def outer(brain, stream_callback=None):
            current = P.current_reply_planner()
            with self.assertRaisesRegex(ValueError, "nested"): inner(brain)
            self.assertIs(P.current_reply_planner(), current)
            stream_callback("one", {})
        with f.scope():
            outer(f.brain, stream_callback=lambda *_: f.db.execute("INSERT INTO effects VALUES('one')"))
        self.assertEqual(f.db.total_changes, 1)
        self.assertIsNone(P.current_reply_planner())
        self.assertTrue(S.qualification_allowed())

    def test_actual_dispatch_consumes_complete_operation_samples_plus_fixed_owner_cost_before_delivery(self):
        f = self.fixture()
        home = {"planSha256": "a" * 64, "backendMode": "native-cpu", "asyncDevices": [],
            "supported": True, "fallbackAcknowledged": False}
        other = {"planSha256": "b" * 64, "backendMode": "native-metal", "asyncDevices": ["mps"],
            "supported": True, "fallbackAcknowledged": False}
        state = {"kind": "live-owner-epochs", "bindingSha256": "8" * 64, "ownerSessionId": "session",
            "complete": True, "ownerVersions": {"original-owner": 1}}
        identity = {"ownerId": "brain", **f.binding_value, "workKind": R.CACHED_KINDS[0],
            "stageCoverage": R.CACHED_STAGES, "workloadSha256": "2" * 64, "geometrySha256": "3" * 64,
            "samplingContractSha256": "5" * 64, "topologySha256": "6" * 64,
            "cache": {"heat": "warm", "logicalStateSha256": "7" * 64, "backendInitialization": "warm"},
            "nativeState": state}
        # Exercise the real source-callback custody contract with an inert
        # clock/operation. This is not hardware or neural speed evidence.
        recorder = W.WholePathRecorder(clock_ns=f.clock)
        rows, chosen = [], []
        def actual_consumer(_decoder, **kwargs):
            for _ in range(3):
                for candidate in (home, other):
                    with f.resource_observation() as resources:
                        observed = recorder.begin(identity, candidate, synchronize=lambda devices: list(devices),
                            counters=lambda: dict.fromkeys(W.COUNTERS, 0), resources=resources)
                        f.advance(100 if candidate is home else 10)
                        for stage in R.CACHED_STAGES: observed.stage_completed(stage)
                        rows.append(observed.finish(native_state_after=state, output_contract_sha256="9" * 64))
            decision = f.scheduler.select(iter(rows), selector=W.select_whole_path_candidate,
                identity=identity, current_candidate=home, minimum_samples=3,
                verify_receipt=lambda row: row["observerId"] == recorder.observer_id)
            self.assertEqual(decision["decision"], "propose-at-quiescence", repr(decision))
            self.assertEqual(decision["candidateEvidence"]["maximumNanoseconds"], "50")
            self.assertEqual(decision["aggregateSchedule"]["fixedMemoryNanoseconds"], "40")
            candidate = decision["proposedCandidate"]
            with S.owned_operation_choice(f.controller, identity, candidate):
                chosen.append(candidate)
                kwargs["before_sample"]()
            return "actual-delivery"
        @P.owned_reply
        def reply(brain):
            with P.reply_stage(brain, "structured-action-arguments"):
                f.db.execute("INSERT INTO effects VALUES('arguments')"); f.advance(40)
            return f.sample(cached=actual_consumer, before_sample=P.world_callback(
                lambda: f.db.execute("INSERT INTO effects VALUES('published-once')")))
        with f.scope(): self.assertEqual(reply(f.brain), "actual-delivery")
        self.assertEqual(chosen, [other])
        self.assertEqual(f.db.total_changes, 2)
        self.assertEqual(f.rows[-1]["actualPreparedOperations"][R.CACHED_KINDS[0]]["completed"], 1)

    def test_private_stop_reads_exact_existing_events_without_polling_world_callback(self):
        f = self.fixture(); polled = []
        def callback(): polled.append("world-poll"); return False
        def actual_consumer(_decoder, **kwargs):
            self.assertFalse(kwargs["flags_cancelled"]())
            private = R.PRIVATE_OPERATION.set(True)
            try:
                f.session.requested.set()
                self.assertTrue(kwargs["flags_cancelled"]())
                with self.assertRaisesRegex(RuntimeError, "real world callback"):
                    P.world_callback(callback)()
            finally: R.PRIVATE_OPERATION.reset(private)
            return kwargs["fallback"]()
        @P.owned_reply
        def reply(brain, cancel_check=None):
            return f.sample(cached=actual_consumer, flags_cancelled=cancel_check)
        with f.scope(): self.assertEqual(reply(f.brain, cancel_check=callback), "ordinary")
        self.assertEqual(polled, [])
        self.assertTrue(f.session.requested.is_set())

    def test_private_or_nested_world_entry_cannot_run_reply_or_requalify_any_native_kind(self):
        f = self.fixture(); executed = []
        @P.owned_reply
        def reply(brain): executed.append("chat")
        with f.scope():
            private = R.PRIVATE_OPERATION.set(True)
            try:
                with self.assertRaisesRegex(RuntimeError, "chat request"): reply(f.brain)
                with self.assertRaisesRegex(RuntimeError, "generation frontier"): f.sample()
            finally: R.PRIVATE_OPERATION.reset(private)
            with S.world_effect_barrier(), S.request_qualification_gate(lambda: True):
                self.assertFalse(S.qualification_allowed())
        self.assertEqual(executed, [])

    def test_actual_native_entry_guards_preserve_one_original_body_inside_world_barrier(self):
        f = self.fixture()
        auto = importlib.import_module(PACKAGE + ".operation_auto_policy")
        cached = importlib.import_module(PACKAGE + ".cached_whole_operation_auto")
        prefix = importlib.import_module(PACKAGE + ".prefill_whole_operation_auto")
        forward = importlib.import_module(PACKAGE + ".forward_whole_operation_auto")
        media = importlib.import_module(PACKAGE + ".media_whole_operation_auto")
        one_shot = importlib.import_module(PACKAGE + ".cached_decoder_execution")
        def ordinary():
            f.db.execute("INSERT INTO effects VALUES('original')")
            return "original"
        arguments = dict(window=None, cue=None, cache=None, generator=None, sampling=None,
            flags_cancelled=None, before_sample=None, fallback=ordinary, fallback_after_boundary=ordinary)
        # No controller/model construction: the new guard must be evaluated
        # before accessing even an owner lock, source, tensor or callback pool.
        with S.world_effect_barrier():
            self.assertEqual(auto.OperationAutoPolicy.consume(object(), None, None, None, ordinary), "original")
            self.assertEqual(cached.CachedWholeOperationPolicy.consume_cached(object(), **arguments), "original")
            self.assertEqual(prefix.FullPrefixOperationPolicy.consume_prefill(object(), **arguments), "original")
            self.assertEqual(forward.FullForwardOperationPolicy.consume_full_forward(object(), window=None, cue=None,
                use_global_workspace=None, attention_mask=None, fallback=ordinary), "original")
            self.assertEqual(media.WholeMediaOperationPolicy.consume_media(object(), None, {"idea": None},
                lambda *_: ordinary(), None, None, None, None), "original")
            self.assertEqual(one_shot.CachedDecoderExecution.consume(object(), **arguments), "original")
        self.assertEqual(f.db.total_changes, 6)

    def test_aborted_or_failed_genuine_delivery_does_not_claim_a_completed_native_sample(self):
        f = self.fixture()
        identity = {"ownerId": "brain", **f.binding_value, "workKind": R.CACHED_KINDS[0]}
        with f.scope():
            with S.owned_operation_choice(f.controller, identity, {}) as choice:
                choice["aborted"] = True
            with self.assertRaisesRegex(ValueError, "real failure"):
                with S.owned_operation_choice(f.controller, identity, {}):
                    raise ValueError("real failure")
        row = f.rows[-1]["actualPreparedOperations"][R.CACHED_KINDS[0]]
        self.assertEqual(row["entered"], 2)
        self.assertEqual(row["completed"], 0)
        self.assertEqual(row["aborted"], 1)

    def test_publication_phase_does_not_add_new_stop_or_pressure_gate_after_an_applied_update(self):
        f = self.fixture()
        def admission(_stage): raise AssertionError("extra admission must not precede original atomic save")
        f.admit_effect = admission
        with f.scope():
            f.db.execute("INSERT INTO effects VALUES('already-applied-update')")
            with S.owned_phase(f.brain, "canonical-checkpoint-publication", publication=True):
                # Original save's real disk/transaction gates remain its own.
                f.db.execute("INSERT INTO effects VALUES('original-atomic-save')"); f.advance(5)
        self.assertEqual(f.db.total_changes, 2)
        self.assertEqual(f.rows[-1]["effectsObserved"], 1)
        self.assertEqual(f.rows[-1]["fixedMemoryNanoseconds"], "5")

    def test_request_metadata_release_failure_does_not_fail_or_repeat_a_committed_request(self):
        f = self.fixture()
        def reserve(_size):
            lease = types.SimpleNamespace(closed=False)
            def release():
                lease.closed = True
                raise RuntimeError("diagnostic lease release failed")
            lease.release = release
            f.leases.append(lease)
            return lease
        f.reserve_metadata = reserve
        with f.scope():
            f.db.execute("INSERT INTO effects VALUES('committed-request')")
        self.assertEqual(f.db.total_changes, 1)
        self.assertEqual(len(f.rows), 1)
        self.assertTrue(f.rows[0]["completed"])
        self.assertTrue(f.leases[0].closed)
        self.assertIsNone(S.current_owned_path())

    def test_new_actual_source_seams_keep_existing_sampling_calls_and_once_only_world_boundaries(self):
        brain = ast.parse((ROOT / "omni_core" / "brain.py").read_text())
        chat = next(node for node in ast.walk(brain) if isinstance(node, ast.FunctionDef) and node.name == "chat")
        self.assertIn("owned_reply", [ast.unparse(node) for node in chat.decorator_list])
        phases = {node.args[1].value for node in ast.walk(chat) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "reply_stage"}
        self.assertEqual(phases, P._STAGES)
        model = ast.parse((ROOT / "omni_core" / "model.py").read_text())
        generate = next(node for node in ast.walk(model) if isinstance(node, ast.FunctionDef) and node.name == "_generate_scoped_iter")
        calls = [node for node in ast.walk(generate) if isinstance(node, ast.Call)]
        self.assertEqual(sum(ast.unparse(node.func) == "dispatch_reply_sample" for node in calls), 1)
        self.assertEqual(sum(ast.unparse(node.func) == "sample_native_token" for node in calls), 2)
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == "PRIVATE_OPERATION" for node in ast.walk(generate)))
        worker = (ROOT / "worker.py").read_text()
        self.assertIn("bind_reply_controls(brain, request_id, control_turn_id", worker)
        self.assertIn("finally:\n            retire_reply_controls(brain, reply_controls)", worker)
        for name in ("operation_auto_policy", "cached_whole_operation_auto", "prefill_whole_operation_auto", "forward_whole_operation_auto"):
            source = (ROOT / "omni_core" / (name + ".py")).read_text()
            self.assertIn("qualification_allowed()", source)
            self.assertIn("with owned_operation_choice(", source)


if __name__ == "__main__": unittest.main()
