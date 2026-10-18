"""Explicit hardware profiling of the current small packed projection primitive.

Import runs no probe. Synthetic float32 activity plus packed exact ternary rows
never constructs a neural module/brain or performs a model-quality quiz.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

PROFILE_FORMAT = "omni-bounded-packed-projection-compute-profile"
PROFILE_KERNEL = "omni_core.model._packed_ternary_forward"
PROFILE_KERNEL_REVISION = "float32-activity-int8-quantization-weighted-packed-rows-int64-dispatch-v2"
PROFILE_ACTIVITY_ROWS = 8
PROFILE_INPUT_FEATURES = 64
PROFILE_OUTPUT_FEATURES = 64
PROFILE_RUNS = 4
PROFILE_SCRATCH_RESERVE_BYTES = 1024 * 1024


def native_projection_tensor_budget(activity_rows: int, input_features: int, output_features: int) -> dict[str, int]:
    """Declared float32 forward tensors; no Torch/import/allocator probe.

    The native forward retains its int64 row blocks while cat creates a second
    int64 result. The scaling expression can retain two float32 outputs at once.
    Sum the known buffers conservatively, not an exact allocator/RSS peak. This
    geometry has base contact state but no repeated-contact owner/pages; dynamic
    page signs/state/endpoints/index caches are admitted by their actual owner.
    """
    if any(type(value) is not int or not 1 <= value <= (1 << 53) - 1
           for value in (activity_rows, input_features, output_features)):
        raise ValueError("native projection tensor geometry requires positive safe integers")
    samples, width, outputs = activity_rows, input_features, output_features
    values = {
        "inputActivityBytes": samples * width * 4,
        "packedSignBytes": outputs * ((width + 3) // 4),
        "connectionStateBytes": outputs * width,
        "weightScaleBytes": 4,
        "quantizedActivityBytes": samples * width,
        "activationScaleBytes": samples * 4,
        "integerRowBlockBytes": samples * outputs * 8,
        "integerConcatenationBytes": samples * outputs * 8,
        "floatingOutputWorkingBytes": samples * outputs * 4 * 2,
        "quantizationWorkingBytes": samples * width * 4 * 2,
        # Metal's three int32 dispatch fields; safe extra allowance on CPU.
        "backendDescriptorBytes": 12,
    }
    values["sampleTensorBytes"] = sum(values[key] for key in (
        "inputActivityBytes", "packedSignBytes", "connectionStateBytes", "weightScaleBytes"))
    values["forwardTensorReserveBytes"] = sum(values[key] for key in (
        "quantizedActivityBytes", "activationScaleBytes", "integerRowBlockBytes",
        "integerConcatenationBytes", "floatingOutputWorkingBytes",
        "quantizationWorkingBytes", "backendDescriptorBytes"))
    if any(value > (1 << 53) - 1 for value in values.values()):
        raise ValueError("native projection tensor bytes exceed exact integer accounting")
    return values


def profile_native_projection_compute(
    *, device: str = "cpu", reserve: Optional[Callable[[int, str], Any]] = None,
    clock: Callable[[], float] = time.perf_counter,
    primitive: Optional[Callable[[], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    rows, width, outputs = PROFILE_ACTIVITY_ROWS, PROFILE_INPUT_FEATURES, PROFILE_OUTPUT_FEATURES
    tensor_budget = native_projection_tensor_budget(rows, width, outputs)
    admitted_bytes = (tensor_budget["sampleTensorBytes"] + tensor_budget["forwardTensorReserveBytes"]
                      + PROFILE_SCRATCH_RESERVE_BYTES)
    injected = primitive is not None
    actual = device

    def boundary():
        if cancelled is not None and cancelled():
            raise InterruptedError("hardware projection profiling cancelled at a primitive boundary")

    boundary()
    if reserve is not None:
        reserve(admitted_bytes, device)
    if primitive is None:
        import torch
        from .model import _packed_ternary_forward, pack_ternary_weight
        try:
            if device == "directml":
                import torch_directml
                target = torch_directml.device()
            else:
                target = torch.device(device)
            actual = str(target)
            levels = (torch.arange(outputs * width, dtype=torch.int32, device=target) % 3 - 1).to(torch.int8).reshape(outputs, width)
            packed = pack_ternary_weight(levels)
            del levels
            # Zero strength/resistance codes preserve the previous unit-strength
            # numerical result while exercising the real weighted kernel input.
            connection_state = torch.zeros((outputs, width), dtype=torch.uint8, device=target)
            activity = (torch.arange(rows * width, dtype=torch.int32, device=target) % 3 - 1).float().reshape(rows, width)
            scale = torch.tensor(width ** -0.5, dtype=torch.float32, device=target)

            def primitive():
                value = _packed_ternary_forward(activity, packed, width, outputs, scale,
                                                connection_state=connection_state)
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                elif target.type == "mps":
                    torch.mps.synchronize()
                elif target.type != "cpu":
                    # A bounded readback observes completion on DirectML.
                    value.detach().to(device="cpu")

            boundary()
            primitive()
        except (ImportError, RuntimeError, NotImplementedError) as error:
            message = str(error).lower()
            if device == "cpu" or any(value in message for value in (
                "out of memory", "cannot allocate memory", "can't allocate memory", "allocation failed",
            )):
                # Temporary allocator pressure is not evidence that a device
                # is intrinsically a slower CPU. Let the collector defer it.
                raise
            cpu = profile_native_projection_compute(device="cpu", reserve=reserve, clock=clock, cancelled=cancelled)
            return {**cpu, "requestedDevice": device, "fallbackReason": type(error).__name__}
    durations = []
    for _ in range(PROFILE_RUNS):
        boundary()
        start = clock()
        primitive()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("projection primitive compute timer is invalid")
        durations.append(elapsed)
        if sum(durations) >= 0.25:
            break
    boundary()
    median_us = max(1, math.ceil(sorted(durations)[len(durations) // 2] * 1_000_000))
    macs = rows * width * outputs
    return {
        "format": PROFILE_FORMAT, "formatVersion": 2,
        "measuredAt": datetime.now(timezone.utc).isoformat(),
        "requestedDevice": device, "actualDevice": actual,
        "kernel": PROFILE_KERNEL, "kernelRevision": PROFILE_KERNEL_REVISION,
        "activityDtype": "float32", "packedDtype": "uint8", "quantizedActivityDtype": "int8",
        "integerResultDtype": "int64", "intermediateAccumulationDtypeVerified": False, "outputDtype": "float32",
        "connectionStateDtype": "uint8", "connectionStateEncoding": "strength-low4-resistance-high4",
        "repeatedContacts": 0, "repeatedContactPages": 0, "repeatedContactTimingMeasured": False,
        "activityRows": rows, "inputFeatures": width, "outputFeatures": outputs,
        **tensor_budget, "scratchReserveBytes": PROFILE_SCRATCH_RESERVE_BYTES,
        "tensorReserveIsExactPeak": False,
        "resourceByteBasis": "known-forward-tensors-plus-allocator-initialization-reserve",
        "admittedBytes": admitted_bytes, "runs": len(durations),
        "projectionMacsPerRun": macs, "medianRunMicroseconds": median_us,
        "projectionMacsPerSecond": max(1, macs * 1_000_000 // median_us),
        "elapsedMicroseconds": max(1, math.ceil(sum(durations) * 1_000_000)),
        "samplingBudgetMicroseconds": 250000,
        "hardWallClockDeadline": False,
        "fallbackReason": None, "warmupPerformed": not injected,
        "implementation": "injected-fixture" if injected else "native-packed-kernel",
        "integerBackend": "runtime-dispatch-may-use-bounded-cpu-fallback",
        "backendInitializationIncluded": False,
        "neuralModelConstructed": False, "neuralQualityMeasured": False,
        "fullNeuralThroughputMeasured": False,
        "evidence": "cache-tile-packed-primitive-not-complete-neural-throughput",
    }
