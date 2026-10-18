"""Bounded, reproducible retention screening for the real packed projection.

This is an opt-in component experiment, not model training, an implementation
of the named literature algorithms, or authority to change runtime defaults.
The four arms differ only in existing resistance and training-only rehearsal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .model import PackedAdaptiveBitLinear


VARIANTS = {
    "native-default": {"row_resistance": True, "replay_fraction": 0.0},
    "row-resistance-off": {"row_resistance": False, "replay_fraction": 0.0},
    "native-rehearsal": {"row_resistance": True, "replay_fraction": 0.25},
    "row-off-rehearsal": {"row_resistance": False, "replay_fraction": 0.25},
}
CONDITIONS = ("shortcut-shift", "disjoint-support-control")


@dataclass(frozen=True)
class Protocol:
    version: int = 1
    seeds: tuple[int, ...] = (17, 29, 43)
    tasks: int = 3
    classes: int = 4
    dimensions: int = 12
    train_samples: int = 64
    heldout_samples: int = 96
    batch_size: int = 16
    steps_per_task: int = 64
    replay_samples_per_task: int = 16
    learning_rate: float = 0.30
    projection_scale: float = 0.25
    stability_strength: float = 0.025
    max_state_and_replay_bytes: int = 65536
    max_train_time_ratio: float = 2.0
    max_worker_rss_bytes: int = 768 * 1024 * 1024
    accuracy_regression_tolerance: float = 0.0
    minimum_acquisition_gain: float = 0.10

    def validate(self) -> None:
        if (self.version != 1 or self.tasks != 3 or self.classes != 4 or self.dimensions != 12
                or self.train_samples != 64 or self.heldout_samples != 96 or self.batch_size != 16
                or self.replay_samples_per_task != 16 or not 1 <= self.steps_per_task <= 256
                or not self.seeds or len(self.seeds) > 16 or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 0 <= seed < 100000 for seed in self.seeds)
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1
                or self.projection_scale != 0.25 or self.stability_strength != 0.025):
            raise ValueError("comparison exceeds the fixed bounded native protocol")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tensor_digest(values) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dataset(protocol: Protocol, condition: str, seed: int, task: int, split: str):
    """Two jointly learnable, fixed data conditions; no learned teacher model.

    Shortcut labels have a permanent class cue and task-dependent nuisance cue.
    The control uses separate feature blocks, so it does not require forgetting.
    Held-out RNG streams never enter updates or the rehearsal reservoir.
    """
    if condition not in CONDITIONS or split not in {"train", "heldout"} or not 0 <= task < 3:
        raise ValueError("unknown comparison data partition")
    count = protocol.train_samples if split == "train" else protocol.heldout_samples
    rng = torch.Generator().manual_seed(seed * 100000 + task * 100 + (1 if split == "train" else 2))
    labels = torch.arange(count, dtype=torch.int64) % protocol.classes
    features = torch.zeros((count, protocol.dimensions), dtype=torch.float32)
    rows = torch.arange(count)
    if condition == "shortcut-shift":
        features += torch.randn(features.shape, generator=rng) * 0.08
        features[rows, labels] += 0.6
        features[rows, 4 + (labels + task) % protocol.classes] += 1.8
    else:
        block = slice(task * protocol.classes, (task + 1) * protocol.classes)
        features[:, block] = torch.randn((count, protocol.classes), generator=rng) * 0.08
        features[rows, task * protocol.classes + labels] += 1.0
    return features, labels


def model_bytes(model: PackedAdaptiveBitLinear) -> dict[str, int]:
    tensors = model.state_dict()
    persistent = sum(value.numel() * value.element_size() for value in tensors.values())
    storages = {}
    for value in model.buffers():
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return {"checkpointTensorBytes": persistent, "residentTensorStorageBytes": sum(storages.values()),
            "floatingParameterBytes": sum(value.numel() * value.element_size() for value in model.parameters()
                                          if value.is_floating_point())}


def assert_native_model(model: PackedAdaptiveBitLinear) -> None:
    if tuple(model.parameters()) or model.packed_forward_status()["latentMasterLearningState"]:
        raise RuntimeError("comparison introduced a learned floating master")
    allowed = {"_packed_forward_scale", "_online_learning_rate", "_autograd_trigger"}
    for name, value in model.named_buffers():
        if value.is_floating_point() and (name not in allowed or value.numel() != 1):
            raise RuntimeError("comparison found non-scalar floating learned storage: " + name)


def accuracy(model, data):
    with torch.no_grad():
        logits = model(data[0])
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("non-finite packed comparison output")
        return {"accuracy": float((logits.argmax(-1) == data[1]).float().mean()),
                "crossEntropy": float(F.cross_entropy(logits, data[1]))}


def worker_peak_rss() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def run_trial(protocol: Protocol, condition: str, seed: int, variant: str) -> dict[str, Any]:
    protocol.validate()
    sources_before = source_hashes()
    if variant not in VARIANTS or seed not in protocol.seeds:
        raise ValueError("trial is outside the declared comparison")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    arm = VARIANTS[variant]
    model = PackedAdaptiveBitLinear(protocol.dimensions, protocol.classes, scale=protocol.projection_scale).eval()
    model.configure_packed_stability(enabled=bool(arm["row_resistance"]), strength=protocol.stability_strength)
    assert_native_model(model)
    initial_checksum = tensor_digest(sorted(model.state_dict().items()))
    train = [dataset(protocol, condition, seed, task, "train") for task in range(protocol.tasks)]
    heldout = [dataset(protocol, condition, seed, task, "heldout") for task in range(protocol.tasks)]
    data_hash = tensor_digest([(f"{split}/{task}/{kind}", tensor)
        for split, partitions in (("train", train), ("heldout", heldout))
        for task, pair in enumerate(partitions) for kind, tensor in zip(("x", "y"), pair)])
    initial = [accuracy(model, data) for data in heldout]  # warms the actual packed kernel before timing
    history, task_times, exposures, changes = [], [], [], []
    replay_x = torch.empty((0, protocol.dimensions), dtype=torch.float32)
    replay_y = torch.empty((0,), dtype=torch.int64)
    initial_storage = model_bytes(model)
    peak_state_bytes = initial_storage["checkpointTensorBytes"]
    peak_replay_bytes = 0
    peak_batch_tensor_bytes = 0
    for task in range(protocol.tasks):
        changed, replay_seen, current_seen = 0, 0, 0
        start = time.perf_counter_ns()
        for step in range(protocol.steps_per_task):
            indices = (torch.arange(protocol.batch_size) + step * protocol.batch_size) % protocol.train_samples
            x, y = train[task][0][indices].clone(), train[task][1][indices].clone()
            count = int(protocol.batch_size * float(arm["replay_fraction"])) if len(replay_y) else 0
            if count:
                replay_indices = (torch.arange(count) + step * count) % len(replay_y)
                x[-count:], y[-count:] = replay_x[replay_indices], replay_y[replay_indices]
            replay_seen += count
            current_seen += protocol.batch_size - count
            with torch.no_grad():
                logits = model(x)
                # learn_from_gradient already averages over the input batch.
                # Supply per-example d(CE)/d(logits), not an extra /batch.
                gradient = logits.softmax(-1) - F.one_hot(y, protocol.classes).float()
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("non-finite comparison gradient")
                rng = torch.Generator().manual_seed(seed * 100000 + task * 1000 + step)
                changed += model.learn_from_gradient(x, gradient, protocol.learning_rate, generator=rng)
            peak_batch_tensor_bytes = max(peak_batch_tensor_bytes,
                sum(value.numel() * value.element_size() for value in (x, y, logits, gradient)))
        task_times.append((time.perf_counter_ns() - start) / 1e9)
        history.append([accuracy(model, data) for data in heldout])
        exposures.append({"newTaskExamples": current_seen, "replayedExamples": replay_seen,
                          "updates": protocol.steps_per_task})
        changes.append(changed)
        if arm["replay_fraction"]:
            count = protocol.replay_samples_per_task
            replay_x = torch.cat((replay_x, train[task][0][:count].clone()))
            replay_y = torch.cat((replay_y, train[task][1][:count].clone()))
        peak_replay_bytes = max(peak_replay_bytes, replay_x.numel() * replay_x.element_size() + replay_y.numel() * replay_y.element_size())
        peak_state_bytes = max(peak_state_bytes, model_bytes(model)["checkpointTensorBytes"])
        assert_native_model(model)
    acquisition = [history[task][task]["accuracy"] for task in range(protocol.tasks)]
    retention = [history[-1][task]["accuracy"] for task in range(protocol.tasks - 1)]
    forgetting = [max(history[stage][task]["accuracy"] for stage in range(task, protocol.tasks)) - history[-1][task]["accuracy"]
                  for task in range(protocol.tasks - 1)]
    if source_hashes() != sources_before:
        raise RuntimeError("native source changed during the paired comparison trial")
    return {"condition": condition, "seed": seed, "variant": variant, "dataSha256": data_hash, "sourceSha256": sources_before,
        "initialStateSha256": initial_checksum, "finalStateSha256": tensor_digest(sorted(model.state_dict().items())),
        "initialHeldout": initial, "afterEachTask": history, "acquisition": acquisition, "retention": retention,
        "acquisitionGain": [acquisition[i] - initial[i]["accuracy"] for i in range(protocol.tasks)],
        "forgetting": forgetting, "newAndReplayExposure": exposures, "changedContactsByTask": changes,
        "trainingSecondsByTask": task_times, "trainingSeconds": sum(task_times),
        "storage": {**model_bytes(model), "peakCheckpointTensorBytes": peak_state_bytes,
            "peakReplayTensorBytes": peak_replay_bytes, "peakAccountedBatchTensorBytes": peak_batch_tensor_bytes,
            "peakWorkerRssBytes": worker_peak_rss()},
        "nativeContract": {"learnedFloatParameters": 0, "latentMasterLearningState": False,
            "stability": model.packed_stability_status(), "packedKernel": model.packed_forward_status()},
        "evaluationEnteredReplay": False}


def compare_trials(trials: list[dict[str, Any]], protocol: Protocol) -> dict[str, Any]:
    protocol.validate()
    expected = {(condition, seed, variant) for condition in CONDITIONS for seed in protocol.seeds for variant in VARIANTS}
    indexed = {(row["condition"], row["seed"], row["variant"]): row for row in trials}
    if len(indexed) != len(trials) or set(indexed) != expected:
        raise ValueError("comparison requires every unique preregistered paired trial")
    comparisons = {}
    for variant in VARIANTS:
        if variant == "native-default":
            continue
        reasons, acquisition_deltas, retention_deltas, time_ratios = [], [], [], []
        for condition in CONDITIONS:
            for seed in protocol.seeds:
                baseline = indexed[(condition, seed, "native-default")]
                candidate = indexed[(condition, seed, variant)]
                label = f"{condition}/seed-{seed}"
                if (baseline["dataSha256"] != candidate["dataSha256"] or
                    baseline["initialStateSha256"] != candidate["initialStateSha256"] or
                    baseline["sourceSha256"] != candidate["sourceSha256"]):
                    raise ValueError("paired comparison changed data, initial state, or native source")
                ad = [a - b for a, b in zip(candidate["acquisition"], baseline["acquisition"])]
                rd = [a - b for a, b in zip(candidate["retention"], baseline["retention"])]
                acquisition_deltas.extend(ad); retention_deltas.extend(rd)
                ratio = candidate["trainingSeconds"] / max(1e-9, baseline["trainingSeconds"])
                time_ratios.append(ratio)
                if any(value < -protocol.accuracy_regression_tolerance for value in ad): reasons.append(label + ": acquisition regression")
                if any(value < -protocol.accuracy_regression_tolerance for value in rd): reasons.append(label + ": retention regression")
                if min(candidate["acquisitionGain"]) < protocol.minimum_acquisition_gain: reasons.append(label + ": insufficient acquisition")
                storage = candidate["storage"]
                if storage["peakCheckpointTensorBytes"] + storage["peakReplayTensorBytes"] > protocol.max_state_and_replay_bytes:
                    reasons.append(label + ": state/replay memory budget")
                if storage["peakWorkerRssBytes"] is None: reasons.append(label + ": peak RSS unavailable")
                elif storage["peakWorkerRssBytes"] > protocol.max_worker_rss_bytes: reasons.append(label + ": worker RSS budget")
                if ratio > protocol.max_train_time_ratio: reasons.append(label + ": training time budget")
                if candidate["evaluationEnteredReplay"] or candidate["nativeContract"]["latentMasterLearningState"] or storage["floatingParameterBytes"]:
                    reasons.append(label + ": native/evaluation isolation violation")
        comparisons[variant] = {"componentScreenPassed": not reasons, "reasons": reasons,
            "meanAcquisitionDelta": statistics.mean(acquisition_deltas), "worstAcquisitionDelta": min(acquisition_deltas),
            "meanRetentionDelta": statistics.mean(retention_deltas), "worstRetentionDelta": min(retention_deltas),
            "meanTrainingTimeRatio": statistics.mean(time_ratios), "maximumTrainingTimeRatio": max(time_ratios),
            "defaultChangeAuthorized": False}
    return comparisons


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
        ("retention_comparison.py", "model.py", "packed_connections.py", "packed_native_kernel.py")}


def run_comparison(protocol: Protocol) -> dict[str, Any]:
    protocol.validate()
    sources_before = source_hashes()
    trials = []
    started = time.perf_counter_ns()
    # One native thread and one subprocess at a time; no model-size/corpus growth.
    # Rotate arm order per seed to reduce systematic warmed-machine bias.
    variants = tuple(VARIANTS)
    for condition in CONDITIONS:
        for index, seed in enumerate(protocol.seeds):
            for variant in variants[index % len(variants):] + variants[:index % len(variants)]:
                print(f"Measuring {condition}, seed {seed}, {variant}", file=sys.stderr, flush=True)
                command = [sys.executable, "-m", "omni_core.retention_comparison", "--worker", condition,
                           str(seed), variant, "--steps", str(protocol.steps_per_task)]
                child = subprocess.run(command, input=json.dumps(asdict(protocol)), text=True, capture_output=True,
                    check=False, timeout=60, cwd=Path(__file__).parents[1])
                if child.returncode:
                    raise RuntimeError("comparison worker failed: " + child.stderr[-4000:])
                trial = json.loads(child.stdout)
                if trial["sourceSha256"] != sources_before:
                    raise RuntimeError("native source changed between paired comparison workers")
                trials.append(trial)
    if source_hashes() != sources_before:
        raise RuntimeError("native source changed during comparison")
    return {"format": "omni-packed-retention-comparison", "formatVersion": 1,
        "scope": "fixed tiny sequential-task native projection; not whole-brain learned capability",
        "protocol": asdict(protocol), "protocolSha256": hashlib.sha256(canonical(asdict(protocol))).hexdigest(),
        "sourceSha256": sources_before, "environment": {"python": platform.python_version(),
            "torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(), "threads": 1},
        "measurement": {"time": "perf_counter_ns update loop; initial kernel warmup and evaluation excluded",
            "memory": "exact tensor payload/storage bytes; isolated process high-water RSS includes Python/Torch/loader overhead",
            "workerIsolation": True, "maximumTrialSeconds": 60, "dataset": "deterministic constructed fixtures; no external/user corpus",
            "replay": "first 16 balanced training records per past task; evaluation never used; 25 percent replaces new examples"},
        "elapsedSeconds": (time.perf_counter_ns() - started) / 1e9, "trials": trials,
        "comparisons": compare_trials(trials, protocol), "defaultChangeAuthorized": False,
        "nextGate": "full native sequential held-out multimodal/source tests, persistence and resource evidence before any default proposal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--worker", nargs=3, metavar=("CONDITION", "SEED", "VARIANT"))
    args = parser.parse_args()
    if args.worker:
        raw = json.load(sys.stdin); raw["seeds"] = tuple(raw["seeds"])
        protocol = Protocol(**raw)
        result = run_trial(protocol, args.worker[0], int(args.worker[1]), args.worker[2])
    else:
        result = run_comparison(Protocol(steps_per_task=args.steps))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Reports are opt-in new artifacts, never model state or settings.
        with args.output.open("x", encoding="utf8") as destination:
            destination.write(encoded)
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
