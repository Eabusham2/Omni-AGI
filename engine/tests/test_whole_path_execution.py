"""NEW synthetic callback receipts only; never a native/hardware benchmark."""
import copy
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("owned_whole_path_fixture", Path(__file__).resolve().parents[1] / "omni_core/whole_path_execution.py")
WHOLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WHOLE)


def identity():
    return {"ownerId": "fake-owned-brain", "ownerSessionId": "fake-same-live-owner", "workKind": "native.decoder.step",
        "stageCoverage": ["input-transfer", "workspace", "decoder-blocks", "heads", "result-readback"],
        "workloadSha256": "1" * 64, "geometrySha256": "2" * 64, "samplingContractSha256": "3" * 64,
        "policySha256": "4" * 64, "topologySha256": "5" * 64, "budgetSha256": "6" * 64,
        "cache": {"heat": "cold", "logicalStateSha256": "7" * 64, "backendInitialization": "warm"},
        "nativeState": {"kind": "live-owner-epochs", "bindingSha256": "8" * 64,
            "ownerSessionId": "fake-same-live-owner", "complete": True,
            "ownerVersions": {"fake-packed": 11, "fake-fast": 22, "fake-recurrent": 33}}}


def candidate(mode="native-cpu", supported=True):
    return {"planSha256": ("a" if mode == "native-cpu" else "b") * 64, "backendMode": mode,
        "asyncDevices": [] if mode == "native-cpu" else ["mps"], "supported": supported, "fallbackAcknowledged": False}


class FakeClock:
    def __init__(self, now=100): self.now = now
    def __call__(self): return self.now


class FakeObservedRuntime:
    """Runtime-shaped fixtures test mechanism; these are NOT actual measurements."""
    def __init__(self):
        self.clock = FakeClock()
        self.recorder = WHOLE.WholePathRecorder(clock_ns=self.clock)
        self.owned = {}
    def receipt(self, duration=100, *, work=None, plan=None, tweak=None, status="completed", after_state=None):
        work, plan = copy.deepcopy(work or identity()), copy.deepcopy(plan or candidate())
        counters = dict.fromkeys(WHOLE.COUNTERS, 0)
        reading = {"budgetSha256": work["budgetSha256"], "admissionComplete": True, "pressureObserved": False,
            "physicalGroups": {"ram:system": {"ceilingBytes": 512, "observedPeakBytes": 100, "peakObservation": "operation-start"}}}
        phase = {"value": 0}
        def sync(devices):
            phase["value"] += 1
            self.clock.now += 20 if phase["value"] == 1 else 30
            return devices
        observation = self.recorder.begin(work, plan, synchronize=sync,
            counters=lambda: copy.deepcopy(counters), resources=lambda: copy.deepcopy(reading))
        for stage in work["stageCoverage"]:
            observation.stage_completed(stage)
        counters["coreReadBytes"] += 16
        counters["activationTransferBytes"] += 8
        reading["physicalGroups"]["ram:system"]["peakObservation"] = "sampled-full-operation"
        self.clock.now += duration - 30  # Completion sync accounts for the last30.
        if tweak:
            tweak(observation, counters, reading)
        receipt = observation.finish(native_state_after=copy.deepcopy(after_state or work["nativeState"]),
            output_contract_sha256="e" * 64, status=status)
        self.owned[receipt["id"]] = receipt["receiptSha256"]
        return receipt
    def verify(self, receipt):
        return self.owned.get(receipt.get("id")) == receipt.get("receiptSha256")
    def paired(self):
        return [self.receipt(duration) for duration in (100, 110, 120)] + [self.receipt(duration, plan=candidate("native-metal")) for duration in (50, 60, 70)]
    def select(self, rows, work=None):
        return WHOLE.select_whole_path_candidate(iter(rows), identity=work or identity(), current_candidate=candidate(), verify_receipt=self.verify)


class WholePathExecutionFixtures(unittest.TestCase):
    def test_total_includes_completion_sync_and_declared_stages_but_excludes_prior_work_drain(self):
        runtime = FakeObservedRuntime()
        receipt = runtime.receipt(80, plan=candidate("native-metal"))
        self.assertTrue(receipt["eligible"])
        self.assertEqual(receipt["totalNanoseconds"], "80")
        self.assertEqual(receipt["completionSynchronizationNanoseconds"], "30")
        self.assertEqual(receipt["priorWorkDrainExcludedNanoseconds"], "20")
        self.assertEqual(receipt["counterDeltas"]["coreReadBytes"], "16")
        self.assertEqual(receipt["counterDeltas"]["activationTransferBytes"], "8")
        self.assertEqual(receipt["identity"]["workKind"], "native.decoder.step")
        self.assertIn("undeclared-brain-stages", receipt["scope"])
        self.assertEqual(receipt["completedStages"], identity()["stageCoverage"])

    def test_proven_live_uncommitted_fence_is_accepted_but_unknown_or_changed_fences_are_diagnostic(self):
        runtime = FakeObservedRuntime()
        self.assertTrue(runtime.receipt()["eligible"])
        unknown = identity()
        unknown["nativeState"].update(kind="unproven", bindingSha256=None, complete=False)
        self.assertIn("native-owner-fence-unproven", runtime.receipt(work=unknown)["ineligibleReasons"])
        changed = copy.deepcopy(identity()["nativeState"])
        changed["ownerVersions"]["fake-fast"] += 1
        self.assertIn("native-owner-fence-changed-during-operation", runtime.receipt(after_state=changed)["ineligibleReasons"])

    def test_selector_requires_actual_matching_work_heat_geometry_budget_and_output_contract(self):
        runtime = FakeObservedRuntime()
        paired = runtime.paired()
        result = runtime.select(paired)
        self.assertEqual(result["decision"], "propose-at-quiescence")
        self.assertEqual(result["proposedCandidate"]["backendMode"], "native-metal")
        self.assertEqual(result["baselineEvidence"]["sumNanoseconds"], "330")
        self.assertFalse(result["automaticAdoptionPerformed"])
        self.assertFalse(result["neuralQualityMeasured"])
        for field in ("workKind", "geometrySha256", "workloadSha256", "samplingContractSha256", "budgetSha256"):
            work = identity()
            work[field] = "native.media.operation" if field == "workKind" else "f" * 64
            self.assertEqual(runtime.select(paired, work)["decision"], "retain-supported-current-fallback")
        warm = identity()
        warm["cache"]["heat"] = "warm"
        self.assertEqual(runtime.select(paired, warm)["decision"], "retain-supported-current-fallback")

    def test_duplicate_page_receipts_do_not_create_samples_and_public_rehash_is_not_authority(self):
        runtime = FakeObservedRuntime()
        first, alternative = runtime.receipt(100), runtime.receipt(50, plan=candidate("native-metal"))
        result = runtime.select([first, first, first, alternative, alternative, alternative])
        self.assertEqual(result["decision"], "retain-supported-current-fallback")
        self.assertEqual(result["ignoredReceipts"], 4)
        forged = copy.deepcopy(first)
        forged["totalNanoseconds"] = "1"
        forged["receiptSha256"] = WHOLE.digest({key: value for key, value in forged.items() if key != "receiptSha256"})
        self.assertEqual(runtime.select([forged])["ignoredReceipts"], 1)
        self.assertEqual(WHOLE.select_whole_path_candidate(runtime.paired(), identity=identity(), current_candidate=candidate(),
            verify_receipt=lambda _receipt: False)["decision"], "retain-supported-current-fallback")

    def test_partial_sync_stage_counter_pressure_and_cancel_or_failure_cannot_select(self):
        runtime = FakeObservedRuntime()
        def partial_sync(observation, _counters, _reading): observation.synchronize = lambda _devices: []
        def partial_stage(observation, _counters, _reading): observation.completed_stages.pop()
        def missing_counter(_observation, counters, _reading): counters["coreTransferBytes"] = None
        def over_budget(_observation, _counters, reading): reading["physicalGroups"]["ram:system"]["observedPeakBytes"] = 513
        for tweak in (partial_sync, partial_stage, missing_counter, over_budget):
            receipt = runtime.receipt(80, plan=candidate("native-metal"), tweak=tweak)
            self.assertFalse(receipt["eligible"])
            self.assertTrue(receipt["ineligibleReasons"])
        for status in ("cancelled", "failed"):
            receipt = runtime.receipt(status=status)
            self.assertEqual(receipt["status"], status)
            self.assertFalse(receipt["eligible"])
        def broken_capture(observation, _counters, _reading):
            observation.counters = lambda: (_ for _ in ()).throw(RuntimeError("fake counter read failed"))
        self.assertIn("terminal-counter-capture-failed", runtime.receipt(tweak=broken_capture)["ineligibleReasons"])

    def test_inert_origin_unsupported_backend_output_changes_and_overlapping_latency_ranges_retain_fallback(self):
        runtime = FakeObservedRuntime()
        runtime.recorder.origin = "inert-fixture"
        self.assertEqual(runtime.select(runtime.paired())["decision"], "retain-supported-current-fallback")
        runtime = FakeObservedRuntime()
        baseline = [runtime.receipt(value) for value in (100, 110, 120)]
        unsupported = [runtime.receipt(40, plan=candidate("native-metal", supported=False)) for _ in range(3)]
        overlap = [runtime.receipt(value, plan=candidate("native-metal")) for value in (50, 60, 105)]
        self.assertEqual(runtime.select(baseline + unsupported)["decision"], "retain-supported-current-fallback")
        self.assertEqual(runtime.select(baseline + overlap)["decision"], "retain-supported-current-fallback")
        other_output = [runtime.receipt(40, plan=candidate("native-metal")) for _ in range(3)]
        for receipt in other_output:
            receipt["outputContractSha256"] = "f" * 64
            receipt["receiptSha256"] = WHOLE.digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
            runtime.owned[receipt["id"]] = receipt["receiptSha256"]  # Synthetic source-shaped other-result fixture.
        self.assertEqual(runtime.select(baseline + other_output)["decision"], "retain-supported-current-fallback")

    def test_apply_is_explicit_one_use_quiescent_and_readmission_bound_never_auto_arms_routes(self):
        runtime = FakeObservedRuntime()
        proposal, calls = runtime.select(runtime.paired()), []
        authority = {"available": True}
        def authorize(_proposal):
            value, authority["available"] = authority["available"], False
            return value
        result = WHOLE.apply_whole_path_proposal(proposal, identity=identity(), current_candidate=candidate(), authorize_proposal=authorize,
            revalidate=lambda: (identity(), candidate()),
            quiescent=lambda: calls.append("safe-boundary"), admit_candidate=lambda _candidate: calls.append("fresh-admission"),
            apply_candidate=lambda chosen: calls.append(chosen["backendMode"]))
        self.assertEqual(calls, ["safe-boundary", "fresh-admission", "native-metal"])
        self.assertTrue(result["applied"])
        self.assertFalse(result["savedManualRoutesArmed"])
        with self.assertRaisesRegex(ValueError, "one-use"):
            WHOLE.apply_whole_path_proposal(proposal, identity=identity(), current_candidate=candidate(), authorize_proposal=authorize,
                revalidate=lambda: (identity(), candidate()),
                quiescent=lambda: self.fail("replay reached boundary"), admit_candidate=lambda _candidate: None, apply_candidate=lambda _candidate: None)
        changed = identity()
        changed["nativeState"]["ownerVersions"]["fake-packed"] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            WHOLE.apply_whole_path_proposal(proposal, identity=changed, current_candidate=candidate(), authorize_proposal=lambda _proposal: True,
                revalidate=lambda: (changed, candidate()),
                quiescent=lambda: self.fail("stale work reached boundary"), admit_candidate=lambda _candidate: None, apply_candidate=lambda _candidate: None)

    def test_unsafe_boundary_and_admission_failure_do_not_apply_or_retry(self):
        runtime = FakeObservedRuntime()
        proposal = runtime.select(runtime.paired())
        for failure in ("boundary", "admission"):
            calls = []
            def boundary():
                calls.append("boundary")
                if failure == "boundary": raise RuntimeError("live cache")
            def admission(_candidate):
                calls.append("admission")
                raise RuntimeError("physical budget waits")
            with self.assertRaises(RuntimeError):
                WHOLE.apply_whole_path_proposal(proposal, identity=identity(), current_candidate=candidate(), authorize_proposal=lambda _proposal: True,
                    revalidate=lambda: (identity(), candidate()),
                    quiescent=boundary, admit_candidate=admission, apply_candidate=lambda _candidate: self.fail("unsafe apply"))
            self.assertEqual(calls, ["boundary"] if failure == "boundary" else ["boundary", "admission"])

    def test_clock_and_counter_precision_do_not_become_float_over_long_uptime(self):
        runtime = FakeObservedRuntime()
        runtime.clock.now = 2 ** 80
        receipt = runtime.receipt(123)
        self.assertEqual(receipt["totalNanoseconds"], "123")
        self.assertEqual(receipt["completionSynchronizationNanoseconds"], "30")

    def test_apply_revalidates_live_fence_after_quiescence_and_after_fresh_admission(self):
        runtime = FakeObservedRuntime()
        proposal = runtime.select(runtime.paired())
        for change_at in ("quiescence", "admission"):
            current, calls = identity(), []
            def mutate(): current["nativeState"]["ownerVersions"]["fake-fast"] += 1
            def boundary():
                calls.append("quiescence")
                if change_at == "quiescence": mutate()
            def admit(_candidate):
                calls.append("admission")
                if change_at == "admission": mutate()
            with self.assertRaisesRegex(ValueError, "current safe/admission"):
                WHOLE.apply_whole_path_proposal(proposal, identity=identity(), current_candidate=candidate(),
                    authorize_proposal=lambda _proposal: True, revalidate=lambda: (current, candidate()),
                    quiescent=boundary, admit_candidate=admit, apply_candidate=lambda _candidate: self.fail("stale apply"))
            self.assertEqual(calls, ["quiescence"] if change_at == "quiescence" else ["quiescence", "admission"])

    def test_proven_cold_backend_is_not_initialized_by_an_excluded_prestart_barrier(self):
        work, clock, sync_calls = identity(), FakeClock(), []
        work["cache"]["backendInitialization"] = "cold"
        reading = {"budgetSha256": work["budgetSha256"], "admissionComplete": True, "pressureObserved": False,
            "physicalGroups": {"ram:system": {"ceilingBytes": 512, "observedPeakBytes": 100, "peakObservation": "operation-start"}}}
        def synchronize(devices):
            sync_calls.append(devices)
            clock.now += 30
            return devices
        observation = WHOLE.WholePathRecorder(clock_ns=clock).begin(work, candidate("native-metal"), synchronize=synchronize,
            counters=lambda: dict.fromkeys(WHOLE.COUNTERS, 0), resources=lambda: copy.deepcopy(reading))
        self.assertEqual(sync_calls, [])
        clock.now += 50
        reading["physicalGroups"]["ram:system"]["peakObservation"] = "sampled-full-operation"
        for stage in work["stageCoverage"]: observation.stage_completed(stage)
        receipt = observation.finish(native_state_after=work["nativeState"], output_contract_sha256="e" * 64)
        self.assertTrue(receipt["eligible"])
        self.assertEqual(receipt["totalNanoseconds"], "80")
        self.assertEqual(receipt["priorWorkDrainExcludedNanoseconds"], "0")
        self.assertEqual(receipt["priorWorkDrainMode"], "not-required-proven-cold-backend")
        self.assertEqual(sync_calls, [("mps",)])


if __name__ == "__main__":
    unittest.main()
