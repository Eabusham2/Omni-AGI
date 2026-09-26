"""Bounded CPU benchmark for GlobalWorkspace summary implementations.

This is intentionally a standalone script rather than a collected test.  It
compares the legacy full-slot graph, a reconstructed nonreentrant-checkpoint
design, and the sequential custom-autograd implementation without touching
MPS, a live brain, or the desktop application.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import GlobalWorkspace


def reconstructed_checkpoint_summary(
    workspace: GlobalWorkspace,
    inputs: Tensor,
    attention_mask: Optional[Tensor],
    chunk_slots: int,
) -> Tensor:
    """Reconstruct the prior per-iteration chunk/checkpoint design.

    This helper is a characterization implementation, not source recovered
    from an authoritative commit.  Every recurrent iteration checkpoints
    contiguous slot chunks, concatenates them in slot order, and retains the
    original full-slot mean.
    """

    attention_mask = workspace._validated_attention_mask(
        inputs, attention_mask
    )
    latents = workspace.latents.unsqueeze(0).expand(
        inputs.shape[0], -1, -1
    )
    keys = workspace.key(inputs)
    values = workspace.value(inputs)
    width = max(1, int(chunk_slots))
    for _ in range(workspace.iterations):
        chunks = []
        for start in range(0, workspace.slots, width):
            chunks.append(
                checkpoint(
                    workspace._attend,
                    latents[:, start : start + width],
                    keys,
                    values,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            )
        latents = torch.cat(chunks, dim=1)
    return workspace.broadcast(workspace.norm(latents)).mean(dim=1)


def _fixture(
    *,
    batch: int,
    sequence: int,
    dimensions: int,
    slots: int,
    iterations: int,
) -> Tuple[GlobalWorkspace, Tensor, Tensor, Tensor]:
    torch.manual_seed(1701)
    workspace = GlobalWorkspace(
        dimensions=dimensions,
        slots=slots,
        iterations=iterations,
    ).train()
    inputs = torch.randn(
        batch, sequence, dimensions, dtype=torch.float32
    ).requires_grad_(True)
    attention_mask = torch.ones(batch, sequence, dtype=torch.bool)
    for row in range(1, batch):
        attention_mask[row, sequence - row * sequence // (batch * 3) :] = False
    probe = torch.linspace(
        -0.75,
        0.75,
        batch * dimensions,
        dtype=torch.float32,
    ).reshape(batch, dimensions)
    return workspace, inputs, attention_mask, probe


def _summary(
    method: str,
    workspace: GlobalWorkspace,
    inputs: Tensor,
    attention_mask: Tensor,
    chunk_slots: int,
) -> Tensor:
    if method == "legacy":
        return workspace(inputs, attention_mask=attention_mask)[1]
    if method == "checkpoint":
        return reconstructed_checkpoint_summary(
            workspace, inputs, attention_mask, chunk_slots
        )
    if method == "sequential":
        workspace.query_chunk_slots = chunk_slots
        return workspace.summarize(inputs, attention_mask=attention_mask)
    raise ValueError(f"unknown method: {method}")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs exposed by CI report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _gradient_snapshot(
    workspace: GlobalWorkspace, inputs: Tensor
) -> Dict[str, Tensor]:
    result = {"inputs": inputs.grad.detach().clone()}
    for name, parameter in workspace.named_parameters():
        if parameter.grad is not None:
            result[name] = parameter.grad.detach().clone()
    return result


def _error(reference: Tensor, actual: Tensor) -> Dict[str, float]:
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).eps)
    return {
        "max_abs": float(difference.max()),
        "max_rel": float((difference / denominator).max()),
    }


def exactness(args: argparse.Namespace) -> Dict[str, object]:
    base, base_inputs, attention_mask, probe = _fixture(
        batch=args.batch,
        sequence=args.sequence,
        dimensions=args.dimensions,
        slots=args.slots,
        iterations=args.iterations,
    )
    state = base.state_dict()
    outputs: Dict[str, Tensor] = {}
    gradients: Dict[str, Dict[str, Tensor]] = {}
    labels = ("legacy", "checkpoint", "sequential")
    for method in labels:
        workspace = GlobalWorkspace(
            args.dimensions, args.slots, args.iterations
        ).train()
        workspace.load_state_dict(state, strict=True)
        inputs = base_inputs.detach().clone().requires_grad_(True)
        summary = _summary(
            method,
            workspace,
            inputs,
            attention_mask,
            args.chunk_slots,
        )
        (summary * probe).sum().backward()
        outputs[method] = summary.detach().clone()
        gradients[method] = _gradient_snapshot(workspace, inputs)

    comparisons: Dict[str, object] = {}
    for method in ("checkpoint", "sequential"):
        gradient_errors = {
            name: _error(reference, gradients[method][name])
            for name, reference in gradients["legacy"].items()
        }
        comparisons[method] = {
            "output": _error(outputs["legacy"], outputs[method]),
            "gradients": gradient_errors,
            "max_gradient_abs": max(
                value["max_abs"] for value in gradient_errors.values()
            ),
            "max_gradient_rel": max(
                value["max_rel"] for value in gradient_errors.values()
            ),
        }
    return {
        "fixture": _fixture_description(args),
        "comparisons_to_legacy": comparisons,
    }


def _fixture_description(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "batch": args.batch,
        "sequence": args.sequence,
        "dimensions": args.dimensions,
        "slots": args.slots,
        "iterations": args.iterations,
        "chunk_slots": args.chunk_slots,
    }


def benchmark(args: argparse.Namespace) -> Dict[str, object]:
    # Trigger method-specific lazy imports, autograd setup, and BLAS setup
    # before the RSS baseline.  In particular, the first nonreentrant
    # checkpoint call initializes framework machinery unrelated to fixture
    # size and would otherwise dominate a bounded comparison.
    warm_workspace, warm_inputs, warm_mask, warm_probe = _fixture(
        batch=1,
        sequence=8,
        dimensions=8,
        slots=8,
        iterations=1,
    )
    warm_summary = _summary(
        args.method,
        warm_workspace,
        warm_inputs,
        warm_mask,
        min(2, args.chunk_slots),
    )
    (warm_summary * warm_probe).sum().backward()
    del warm_workspace, warm_inputs, warm_mask, warm_probe, warm_summary
    gc.collect()

    workspace, inputs, attention_mask, probe = _fixture(
        batch=args.batch,
        sequence=args.sequence,
        dimensions=args.dimensions,
        slots=args.slots,
        iterations=args.iterations,
    )
    gc.collect()
    baseline = _peak_rss_bytes()
    started = time.perf_counter()
    summary = _summary(
        args.method,
        workspace,
        inputs,
        attention_mask,
        args.chunk_slots,
    )
    (summary * probe).sum().backward()
    elapsed = time.perf_counter() - started
    peak = _peak_rss_bytes()
    return {
        "method": args.method,
        "fixture": _fixture_description(args),
        "elapsed_seconds": elapsed,
        "baseline_peak_rss_mib": baseline / math.pow(2, 20),
        "peak_rss_mib": peak / math.pow(2, 20),
        "incremental_peak_rss_mib": (peak - baseline) / math.pow(2, 20),
        "output_checksum": float(summary.detach().double().sum()),
        "input_gradient_checksum": float(inputs.grad.detach().double().sum()),
    }


def parse_args(arguments: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("legacy", "checkpoint", "sequential", "exactness"),
        required=True,
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=384)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--slots", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--chunk-slots", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    result = exactness(args) if args.method == "exactness" else benchmark(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
