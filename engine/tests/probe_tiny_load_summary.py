"""Opt-in bounded native cold Load; --source-root never initializes a brain.

The parent watches only its own child family. This is not a connected inherited
frontier, packaged application, quality test or whole-size speed benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import sys
import time

try:
    from .tiny_load_probe_receipts import (content_snapshot, prepare_private_copy,
        runtime_source_snapshot, sqlite_snapshot, strict_private_files, tree_snapshot, write_receipt)
except ImportError:
    from tiny_load_probe_receipts import (content_snapshot, prepare_private_copy,
        runtime_source_snapshot, sqlite_snapshot, strict_private_files, tree_snapshot, write_receipt)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            result.update(chunk)
    return result.hexdigest()


def child_probe(root):
    import gc
    from unittest.mock import patch
    import torch
    from omni_core.brain import AdaptiveBrain
    from omni_core.config import OmniConfig
    from worker import Worker

    torch.set_num_threads(1)
    config = OmniConfig.micro(d_model=8, n_heads=1, n_layers=1, d_ff=16, idea_dim=8,
        vsa_dim=32, router_neurons=4, working_memory_slots=8, max_seq_len=32, modality_channels=4,
        vision_enabled=False, image_enabled=False, audio_enabled=False, video_enabled=False,
        online_learning=False, idle_cognition=False, recursive_improvement=False,
        system_ram_share_percent=30.0, device="cpu")
    config.validate()
    storage = root / "brain"
    model = worker = None
    started = time.perf_counter()
    forbidden = AssertionError("This cold-Load fixture must not train, ingest, chat or run a curriculum")
    with patch.object(AdaptiveBrain, "_train_ground_up_curriculum", side_effect=forbidden), \
            patch.object(AdaptiveBrain, "train", side_effect=forbidden), \
            patch.object(AdaptiveBrain, "ingest", side_effect=forbidden), \
            patch.object(AdaptiveBrain, "chat", side_effect=forbidden):
        try:
            model = AdaptiveBrain("tiny-load", storage, config)
            emit("random-initialized", elapsedSeconds=time.perf_counter() - started)
            initial_checksum = model.parameter_checksum()
            initial_count = model.parameter_accounting()["totalNeuralParameters"]
            model.save()
            model.export_packed_ternary()
            paths = [storage / "engine" / name for name in ("core.safetensors", "plasticity.safetensors", "brain.json")]
            before = {path.name: file_sha(path) for path in paths}
            (root / "preload-receipt.json").write_text(json.dumps({"parameterChecksum": initial_checksum,
                "exactParameters": initial_count, "sourceFiles": before, "curriculumRun": False}, indent=2) + "\n")
            model.close(); model = None; gc.collect()
            emit("original-closed-before-cold-load", exactParameters=initial_count, elapsedSeconds=time.perf_counter() - started)
            worker = Worker()
            counts = {"summaryCalls": 0, "accountingCalls": 0, "checksumCalls": 0}
            original_summary = AdaptiveBrain.summary
            original_accounting = AdaptiveBrain.parameter_accounting
            original_checksum = AdaptiveBrain.parameter_checksum

            def summary(owner):
                counts["summaryCalls"] += 1
                return original_summary(owner)

            def accounting(owner):
                counts["accountingCalls"] += 1
                return original_accounting(owner)

            def checksum(owner):
                counts["checksumCalls"] += 1
                return original_checksum(owner)

            load_started = time.perf_counter()
            with patch.object(AdaptiveBrain, "summary", summary), patch.object(AdaptiveBrain, "parameter_accounting", accounting), patch.object(AdaptiveBrain, "parameter_checksum", checksum):
                result = worker.load({"brainId": "tiny-load", "storagePath": str(storage), "config": {"contextWindowTokens": 32}}, "tiny-cold-load")
            elapsed = time.perf_counter() - load_started
            loaded = worker.brains["tiny-load"]
            after = {path.name: file_sha(path) for path in paths}
            loaded_checksum = loaded.parameter_checksum()
            observation = {"coldLoadReturned": True, "coldLoadSeconds": elapsed, "loadCalls": counts,
                "initialParameterChecksum": initial_checksum, "returnedParameterChecksum": result["parameterChecksum"],
                "loadedParameterChecksum": loaded_checksum, "sourceFilesBefore": before, "sourceFilesAfter": after,
                "trainingSteps": loaded.counters["training_steps"], "inferenceCount": loaded.counters["inference_count"],
                "experiences": loaded.counters["experiences"]}
            (root / "postload-observation.json").write_text(json.dumps(observation, indent=2) + "\n")
            emit("postload-observation", **observation)
            if result["parameterChecksum"] != initial_checksum or loaded_checksum != initial_checksum or before != after:
                raise AssertionError("Cold Load changed the saved random learned/source bytes")
            if loaded.counters["training_steps"] or loaded.counters["inference_count"] or loaded.counters["experiences"]:
                raise AssertionError("Cold-Load-only fixture performed learning or inference")
            if counts["summaryCalls"] != 1:
                raise AssertionError("Cold Load produced more than its one returned final summary")
            if result["metrics"]["parameterAccounting"] != result["runtimeCard"]["parameterAccounting"]:
                raise AssertionError("Summary inventories differ")
            receipt = {"format": "omni-tiny-random-cold-load-probe", "version": 1,
                "passed": True, "root": str(root), "brainId": "tiny-load", "backend": "cpu",
                "coldLoadSeconds": elapsed, "totalChildSeconds": time.perf_counter() - started,
                "exactParameters": initial_count, "loadCalls": counts, "parameterChecksum": initial_checksum,
                "sourceFilesBefore": before, "sourceFilesAfter": after, "sourceFilesUnchanged": before == after,
                "trainingSteps": 0, "inferenceCount": 0, "experiences": 0,
                "curriculumRun": False, "connectedFrontierProof": False, "userBrainTouched": False,
                "simultaneousModelOwners": 1, "mode": "fresh-random-save-close-cold-Worker.Load-only"}
            (root / "child-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            emit("passed", **receipt)
        finally:
            if model is not None:
                model.close()
            if worker is not None:
                worker.shutdown({}, None)


def child_load_only(root, source_root):
    """One actual Load owner; every clause is persisted before asserting it."""
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    from omni_core.brain import AdaptiveBrain
    from worker import Worker

    torch.set_num_threads(1)
    started = time.monotonic()
    storage = root / "brain"
    custody = json.loads((root / "copy-receipt.json").read_text())
    metadata = json.loads((storage / "engine" / "brain.json").read_text())
    generation_path = storage / "engine" / "state" / metadata["mutable_state"]["generationManifest"]
    generation = json.loads(generation_path.read_text())
    expected_checksum = generation["metadata"]["parameterChecksum"]
    expected_parameters = metadata["runtime_card"]["parameterAccounting"]["totalNeuralParameters"]
    before_files = tree_snapshot(storage)
    before_sql = sqlite_snapshot(storage)
    before_source = runtime_source_snapshot()
    pre = {"mode": "private-copy-existing-native-Load-only", "sourceRoot": str(source_root),
           "brainId": metadata["brain_id"], "expectedParameterChecksum": expected_checksum,
           "savedReportedLogicalParameters": expected_parameters, "privateFiles": before_files,
           "privateSqlite": before_sql, "runtimeSources": before_source,
           "savedClock": metadata.get("activity_clock"), "savedCounters": metadata["counters"],
           "mutablePointer": metadata["mutable_state"], "substratePointer": metadata["substrate"]["persistence"],
           "originDirectoryPresent": custody["originDirectoryPresent"],
           "newInitializationAllowed": False, "forwardAllowed": False, "learningAllowed": False}
    write_receipt(root / "preload-observation.json", pre)
    emit("private-copy-preload-observed", exactParameters=expected_parameters,
         parameterChecksum=expected_checksum, newInitialization=False)
    worker = None
    result = None
    observation = {}
    error = None
    counts = {"summaryCalls": 0, "accountingCalls": 0, "checksumCalls": 0,
              "checkpointLoadConstructors": 0, "forbiddenCalls": 0}
    original_init = AdaptiveBrain.__init__
    original_summary = AdaptiveBrain.summary
    original_accounting = AdaptiveBrain.parameter_accounting
    original_checksum = AdaptiveBrain.parameter_checksum

    def checkpoint_init(owner, brain_id, path, config, **kwargs):
        if (kwargs.get("_loading_checkpoint") is not True or Path(path).resolve() != storage.resolve()
                or brain_id != metadata["brain_id"]):
            counts["forbiddenCalls"] += 1
            raise AssertionError("Load-only probe attempted fresh or foreign neural initialization")
        counts["checkpointLoadConstructors"] += 1
        return original_init(owner, brain_id, path, config, **kwargs)

    def forbidden(*_args, **_kwargs):
        counts["forbiddenCalls"] += 1
        raise AssertionError("Load-only probe cannot initialize, forward, train, save/repack, ingest or generate")

    def counted(name, method):
        def invoke(owner, *args, **kwargs):
            counts[name] += 1
            return method(owner, *args, **kwargs)
        return invoke

    try:
        worker = Worker()
        with ExitStack() as stack:
            stack.enter_context(patch.object(AdaptiveBrain, "__init__", checkpoint_init))
            for name in ("_train_ground_up_curriculum", "train", "ingest", "chat", "generate_modality", "save", "export_packed_ternary"):
                stack.enter_context(patch.object(AdaptiveBrain, name, forbidden))
            stack.enter_context(patch.object(Worker, "create", forbidden))
            stack.enter_context(patch.object(torch.nn.Module, "_call_impl", forbidden))
            stack.enter_context(patch.object(AdaptiveBrain, "summary", counted("summaryCalls", original_summary)))
            stack.enter_context(patch.object(AdaptiveBrain, "parameter_accounting", counted("accountingCalls", original_accounting)))
            stack.enter_context(patch.object(AdaptiveBrain, "parameter_checksum", counted("checksumCalls", original_checksum)))
            load_started = time.monotonic()
            result = worker.load({"brainId": metadata["brain_id"], "storagePath": str(storage),
                                  "config": {"contextWindowTokens": metadata["config"]["max_seq_len"]}},
                                 "tiny-private-cold-load")
            load_finished = time.monotonic()
            load_counts = dict(counts)
            loaded = worker.brains[metadata["brain_id"]]
            live_clock = loaded._owned_activity_clock().snapshot()
            clock_observed_at = time.monotonic()
            observation = {"coldLoadReturned": True, "coldLoadSeconds": load_finished - load_started,
                "loadCalls": load_counts, "returnedParameterChecksum": result["parameterChecksum"],
                "loadedParameterChecksum": original_checksum(loaded),
                "loadedCounters": dict(loaded.counters), "liveClock": live_clock,
                "maximumObservedClockAdvanceSeconds": clock_observed_at - load_started,
                "loadedActivityMarkers": {name: getattr(loaded, name) for name in
                    ("last_activity_decay", "last_idle_cycle_at", "last_idle_visible_action_at")},
                "loadedMutablePointer": dict(loaded.mutable_state_manifest),
                "loadedSubstratePointer": dict(loaded.memory.persistence_manifest),
                "workingClockBound": loaded.paged_working_memory.activity_clock_bound,
                "workingActivityBaseline": loaded.paged_working_memory._activity_baseline,
                "metricsParameterAccounting": result["metrics"]["parameterAccounting"],
                "runtimeParameterAccounting": result["runtimeCard"]["parameterAccounting"],
                "derivedSidecarStatus": result.get("runtimeCard", {}).get("packedInferenceSidecar")}
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        if worker is not None:
            try:
                worker.shutdown({}, None)
            except BaseException as caught:
                observation["closeError"] = {"type": type(caught).__name__, "message": str(caught)}

    after_files = tree_snapshot(storage)
    after_sql = sqlite_snapshot(storage)
    after_source = runtime_source_snapshot()
    original_after = tree_snapshot(source_root)
    clauses = {
        "loadReturned": result is not None and error is None,
        "originalFixtureAllFilesAndIdentitiesUnchanged": custody["sourceTreeBefore"] == original_after,
        "privateAuthoritativeAndDerivedFilesUnchanged": strict_private_files(before_files) == strict_private_files(after_files),
        "privateSqliteSchemaAndEveryLogicalRowUnchanged": before_sql == after_sql,
        "runtimeSourcesUnchanged": before_source == after_source,
        "noForbiddenCall": counts["forbiddenCalls"] == 0,
        "oneCheckpointLoadOwner": counts["checkpointLoadConstructors"] == 1,
        "closeSucceeded": "closeError" not in observation,
    }
    if result is not None:
        live = observation.get("liveClock", {})
        saved = metadata["activity_clock"]
        accounting = observation.get("metricsParameterAccounting", {})
        clauses.update({
            "returnedLearnedChecksumMatchesSavedGeneration": observation.get("returnedParameterChecksum") == expected_checksum,
            "loadedLearnedChecksumMatchesSavedGeneration": observation.get("loadedParameterChecksum") == expected_checksum,
            "allSavedCountersUnchanged": all(observation.get("loadedCounters", {}).get(name) == value for name, value in metadata["counters"].items()),
            "noNewNonzeroCounter": all(value == 0 for name, value in observation.get("loadedCounters", {}).items() if name not in metadata["counters"]),
            "oneReturnedSummary": observation.get("loadCalls", {}).get("summaryCalls") == 1,
            "returnedInventoriesEqual": accounting == observation.get("runtimeParameterAccounting"),
            "logicalParameterCountUnchanged": accounting.get("totalNeuralParameters") == expected_parameters,
            "mutableGenerationUnchanged": observation.get("loadedMutablePointer") == metadata["mutable_state"],
            "substrateGenerationUnchanged": observation.get("loadedSubstratePointer") == metadata["substrate"]["persistence"],
            "clockResumedWithoutOfflineAdvance": (live.get("paused") is False and saved["elapsed"] <= live.get("elapsed", -1)
                <= saved["elapsed"] + observation.get("maximumObservedClockAdvanceSeconds", -1)),
            "activityMarkersUnchanged": observation.get("loadedActivityMarkers") == {name: metadata[name] for name in
                ("last_activity_decay", "last_idle_cycle_at", "last_idle_visible_action_at")},
            "readOnlyWorkingClockBound": observation.get("workingClockBound") is True,
        })
    receipt = {"format": "omni-existing-tiny-private-cold-load-probe", "version": 1,
        "passed": all(clauses.values()), "root": str(root), "sourceRoot": str(source_root),
        "observation": observation, "error": error, "clauses": clauses,
        "privateFilesAfter": after_files, "privateSqliteAfter": after_sql,
        "runtimeSourcesAfter": after_source, "sourceTreeAfter": original_after,
        "totalChildSeconds": time.monotonic() - started,
        "newBrainInitialized": False, "curriculumRun": False, "forwardOrLearningRun": False,
        "connectedFrontierProof": False, "trainedCapabilityProof": False, "userBrainTouched": False}
    # Clause details survive an assertion failure; no old failed receipt is
    # overwritten or retrospectively relabeled by this separate attempt.
    write_receipt(root / "postload-observation.json", receipt)
    emit("private-copy-postload-observed", passed=receipt["passed"], clauses=clauses,
         coldLoadSeconds=observation.get("coldLoadSeconds"), error=error)
    if not receipt["passed"]:
        raise AssertionError("Load-only preservation failed: " + ", ".join(name for name, passed in clauses.items() if not passed))
    write_receipt(root / "child-receipt.json", receipt)
    emit("passed", coldLoadSeconds=observation["coldLoadSeconds"], exactParameters=expected_parameters)


def family_rss(root_pid):
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], check=True, capture_output=True, text=True).stdout
    rows = [tuple(map(int, row.split())) for row in output.splitlines() if row.strip()]
    family = {root_pid}
    while True:
        children = {pid for pid, parent, _rss in rows if parent in family}
        if children <= family:
            break
        family.update(children)
    return sum(rss * 1024 for pid, _parent, rss in rows if pid in family)


def allocated_bytes(root):
    result = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # An atomic publication can retire a staging file.
            result += stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size
    return result


def supervise(root, *, source_root=None):
    started = time.monotonic()
    minimum_free_disk = 20 * 1024**3
    if shutil.disk_usage(root).free < minimum_free_disk:
        raise ValueError("Load-only fixture requires the actual20GiB free-disk floor")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    arguments = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--root", str(root)]
    if source_root is not None:
        arguments.extend(["--source-root", str(source_root)])
    child = subprocess.Popen(arguments,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    peak, disk_peak, killed, last_sample = 0, 0, None, 0.0
    minimum_observed_free_disk = shutil.disk_usage(root).free
    try:
        while child.poll() is None:
            now = time.monotonic()
            if now - last_sample >= 0.5:
                peak = max(peak, family_rss(os.getpid()))
                disk_peak = max(disk_peak, allocated_bytes(root))
                minimum_observed_free_disk = min(minimum_observed_free_disk, shutil.disk_usage(root).free)
                last_sample = now
            if (now - started > 120 or peak > 512 * 1024**2 or disk_peak > 128 * 1024**2
                    or minimum_observed_free_disk < minimum_free_disk):
                killed = ("time" if now - started > 120 else "RAM" if peak > 512 * 1024**2
                          else "disk" if disk_peak > 128 * 1024**2 else "free-disk-floor")
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            for key, _events in selector.select(timeout=0.25):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        remaining = child.communicate(timeout=5)[0]
        if remaining:
            print(remaining.decode("utf-8", errors="replace"), end="", flush=True)
    finally:
        selector.close()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
    summary = {"root": str(root), "childExitCode": child.returncode, "capKill": killed,
               "elapsedSeconds": time.monotonic() - started, "peakFamilyRssBytes": peak, "peakAllocatedFixtureBytes": disk_peak,
               "minimumObservedFreeDiskBytes": minimum_observed_free_disk,
               "caps": {"seconds": 120, "familyRamBytes": 512 * 1024**2, "fixtureAllocatedBytes": 128 * 1024**2,
                        "minimumFreeDiskBytes": minimum_free_disk}}
    if source_root is not None:
        custody = json.loads((root / "copy-receipt.json").read_text())
        source_after = tree_snapshot(source_root)
        summary["originalFixtureAllFilesAndIdentitiesUnchanged"] = custody["sourceTreeBefore"] == source_after
        write_receipt(root / "source-after-supervisor.json", {"sourceTreeAfter": source_after,
            "originalFixtureAllFilesAndIdentitiesUnchanged": summary["originalFixtureAllFilesAndIdentitiesUnchanged"]})
    summary["resourceSampleIntervalSeconds"] = 0.5
    (root / "supervisor-receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit("supervisor-finished", **summary)
    return 0 if child.returncode == 0 and killed is None and summary.get("originalFixtureAllFilesAndIdentitiesUnchanged", True) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--initialize-random", action="store_true",
                        help="Separate explicit legacy creation mode; never used by Load-only verification")
    options = parser.parse_args()
    folder = options.root.resolve()
    if options.root.is_symlink() or not folder.is_dir() or not folder.name.startswith("omni-native-cold-load-"):
        raise SystemExit("Probe requires its explicit generated temporary directory")
    if options.child:
        if options.source_root is not None:
            child_load_only(folder, options.source_root.resolve())
        elif options.initialize_random:
            child_probe(folder)
        else:
            raise SystemExit("Choose explicit Load-only source; random initialization is not implicit")
    else:
        if any(folder.iterdir()):
            raise SystemExit("Do not rerun or replace a populated native probe directory")
        if options.source_root is not None and not options.initialize_random:
            prepare_private_copy(options.source_root, folder, minimum_free_disk_bytes=20 * 1024**3)
            raise SystemExit(supervise(folder, source_root=options.source_root.resolve()))
        if options.initialize_random and options.source_root is None:
            # Kept only as an explicitly selected historical fixture mode.
            raise SystemExit("Legacy creation mode is not authorized by the Load-only gate")
        raise SystemExit("Load-only verification requires --source-root; no brain will be initialized")
