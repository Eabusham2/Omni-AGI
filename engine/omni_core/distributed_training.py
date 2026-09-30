"""Experimental torchrun ground-up corpus training for OmniCortex.

Only ordinary next-token/reconstruction learning is performed here. A run
must use a randomly initialized OmniCortex brain; there is no reward model,
preference objective, RLHF stage, or imported model.

Packed derivatives are staged, reduced in a fixed owner/row order, and applied
once to a canonical uint8 identity. Literal source windows are not sampled.
Fast substrate/STDP state is replayed in global source order on the canonical
native checkpoint, then every replica refreshes from that exact publication.
Runtime neural quality and distributed throughput remain separate acceptance
gates; source and primitive checks do not establish either.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .brain import AdaptiveBrain
from .capability_rehearsal import (
    CapabilityRehearsalPolicy,
    CapabilityScheduleState,
    advance_schedule_state,
    due_rehearsal_phase,
    eligible_ground_up_rehearsal,
    rehearse_capabilities,
)
from .config import OmniConfig
from .distributed_runtime import (
    DatasetManifest,
    DatasetManifestEntry,
    DistributedContext,
    DistributedRunStore,
    DistributedRunLease,
    RankCursor,
    ResourceReading,
    aggregate_resource_readings,
    initial_rank_cursors,
    sample_resources,
)
from .ground_up import (
    GROUND_UP_READINESS_PROBE_FIXTURES,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_TRAJECTORIES,
    ground_up_curriculum_manifest,
    validate_ground_up_v3_training_manifest,
)
from .model import PACKED_AUTHORITATIVE_PROJECTION_TYPES, _apply_packed_gradient_rows
from .packed_collective import PackedCollectiveController
from .packed_collective_hooks import packed_derivative_sink
from .collective_controls import synchronize_control_state
from .window_wave_buffer import PreparedWindowWave
from .optimizers import PackedOnlyOptimizer, adamw_for_remaining_parameters
from .persistence import tensor_checksum
from .ternary_packing import verify_ternary_shards
from .text_spool import DatasetResourcePause
from .record_window_wave import RecordWindowStream
from .distributed_seal import make_distributed_training_seal, native_topology_sha256
from .text_spool import bounded_json_sha256


_DISTRIBUTED_ORIGIN_IDENTITY_FIELDS = (
    "seed",
    "vocab_size",
    "max_seq_len",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "idea_dim",
    "vsa_dim",
    "router_neurons",
    "hardware_tier",
    "origin_kind",
    "working_memory_slots",
    "image_size",
    "audio_samples",
    "video_frames",
    "modality_channels",
    "vision_enabled",
    "image_enabled",
    "audio_enabled",
    "video_enabled",
)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def _device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, Mapping):
        return {key: _device_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_device_tree(item, device) for item in value]
    return copy.deepcopy(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_distributed_coverage(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep exhaustive counts while excluding host paths and error prose."""

    integer_fields = (
        "schemaVersion",
        "discoveredFiles",
        "processedFiles",
        "rejectedFiles",
        "discoveredRecords",
        "processedRecords",
        "rejectedRecords",
        "discoveredBytes",
        "processedBytes",
        "shards",
    )
    result = {
        name: max(0, int(value.get(name, 0)))
        for name in integer_fields
    }
    modality_counts = value.get("modalityCounts")
    result["modalityCounts"] = {
        str(name): max(0, int(count))
        for name, count in (
            modality_counts.items()
            if isinstance(modality_counts, Mapping)
            else ()
        )
    }
    errors = value.get("errors")
    result["errorCount"] = (
        len(errors) if isinstance(errors, list) else max(0, int(value.get("errorCount", 0)))
    )
    result["complete"] = value.get("complete") is True
    return result


def _safe_resource_telemetry(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Preserve measured resource accounting without portable host identity."""

    scalar_fields = (
        "measuredAt",
        "rankCount",
        "minimumDiskFreeBytes",
        "maximumDiskReserveBytes",
        "minimumDiskAboveReserveBytes",
        "minimumAvailableMemoryBytes",
        "totalProcessPeakRssBytes",
        "totalAcceleratorAllocatedBytes",
        "totalAcceleratorReservedBytes",
        "diskPressure",
    )
    result = {
        name: value.get(name)
        for name in scalar_fields
        if name in value
    }
    per_rank = value.get("perRank")
    permitted_rank_fields = (
        "rank",
        "device",
        "measuredAt",
        "diskTotalBytes",
        "diskFreeBytes",
        "diskReserveBytes",
        "availableMemoryBytes",
        "ramReserveBytes",
        "processPeakRssBytes",
        "acceleratorAllocatedBytes",
        "acceleratorReservedBytes",
    )
    result["perRank"] = (
        [
            {
                name: row.get(name)
                for name in permitted_rank_fields
                if name in row
            }
            for row in per_rank
            if isinstance(row, Mapping)
        ]
        if isinstance(per_rank, list)
        else []
    )
    return result


def _telemetry_ledger_receipt(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise RuntimeError("distributed telemetry ledger has a partial record")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("distributed telemetry ledger is invalid") from error
            if not isinstance(value, dict):
                raise RuntimeError("distributed telemetry ledger record is invalid")
            digest.update(line)
            entries += 1
    if entries < 1:
        raise RuntimeError("distributed telemetry ledger is empty")
    return {
        "format": "jsonl",
        "records": entries,
        "sha256": digest.hexdigest(),
    }


def _verified_content_sha256(value: Mapping[str, Any], label: str) -> str:
    claimed = value.get("contentSha256")
    body = {key: item for key, item in value.items() if key != "contentSha256"}
    calculated = _canonical_sha256(body)
    if claimed != calculated:
        raise RuntimeError("%s content checksum is invalid" % label)
    return calculated


def _validate_promotion_receipt(
    value: Mapping[str, Any], expected_run_identity: str
) -> Dict[str, Any]:
    dataset = value.get("dataset")
    packed = value.get("packedTernary")
    resources = value.get("resources")
    accounting = value.get("parameterAccounting")
    parameter_evidence = value.get("parameterEvidence")
    coverage = dataset.get("coverage") if isinstance(dataset, Mapping) else None
    telemetry_ledger = (
        resources.get("telemetryLedger")
        if isinstance(resources, Mapping)
        else None
    )
    if (
        value.get("format") != "omni-distributed-ground-up-promotion"
        or value.get("formatVersion") != 2
        or value.get("runIdentitySha256") != expected_run_identity
        or value.get("runtimeReady") is not True
        or not _sha256_identifier(value.get("contentSha256"))
        or value.get("originKind") != "ground-up"
        or value.get("externalWeightFiles") != []
        or value.get("rlhf") is not False
        or value.get("rewardModel") is not False
        or value.get("preferenceLabels") is not False
        or not isinstance(accounting, Mapping)
        or not _sha256_identifier(value.get("parameterChecksum"))
        or not isinstance(parameter_evidence, Mapping)
        or not _sha256_identifier(parameter_evidence.get("before"))
        or parameter_evidence.get("after") != value.get("parameterChecksum")
        or parameter_evidence.get("changed") != (
            parameter_evidence.get("before")
            != parameter_evidence.get("after")
        )
        or not isinstance(dataset, Mapping)
        or not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or not _strict_nonnegative_integer(dataset.get("recordsExpected"))
        or dataset.get("recordsExpected") != dataset.get("recordsVisited")
        or dataset.get("recordsExpected") != dataset.get("dynamicHighWater")
        or not isinstance(packed, Mapping)
        or packed.get("coverageComplete") is not True
        or not _sha256_identifier(packed.get("contentSha256"))
        or packed.get("parameterChecksum") != value.get("parameterChecksum")
        or not isinstance(resources, Mapping)
        or not isinstance(resources.get("finalCheckpoint"), Mapping)
        or not isinstance(telemetry_ledger, Mapping)
        or not _sha256_identifier(telemetry_ledger.get("sha256"))
    ):
        raise RuntimeError("distributed promotion receipt identity is invalid")
    _verified_content_sha256(value, "distributed promotion receipt")
    return dict(value)


def _sha256_identifier(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _final_capability_receipt_ready(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not _sha256_identifier(value.get("contentSha256")):
        return False
    after = value.get("after")
    action = value.get("action")
    all_routes = (
        after.get("allToolTrajectories")
        if isinstance(after, Mapping)
        else None
    )
    curriculum = ground_up_curriculum_manifest()
    try:
        _verified_content_sha256(value, "capability rehearsal receipt")
    except RuntimeError:
        return False
    return bool(
        value.get("phase") == "final"
        and value.get("appliedRank") == 0
        and value.get("regressionGatePassed") is True
        and value.get("curriculumId") == curriculum["id"]
        and value.get("curriculumSha256") == curriculum["sha256"]
        and value.get("recordsVisited") == value.get("expectedRecords")
        and isinstance(after, Mapping)
        and after.get("passed") is True
        and after.get("correct") == len(GROUND_UP_READINESS_PROBE_FIXTURES)
        and isinstance(action, Mapping)
        and action.get("calibrated") is True
        and isinstance(all_routes, Mapping)
        and all_routes.get("passed") is True
        and all_routes.get("routeCount") == len(GROUND_UP_TOOL_TRAJECTORIES)
        and all_routes.get("correctRoutes")
        == len(GROUND_UP_TOOL_TRAJECTORIES)
        and all_routes.get("negativeCount")
        == len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
        and all_routes.get("negativeNoActionCount")
        == len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
    )


def _distributed_origin_identity(config: OmniConfig) -> Dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in _DISTRIBUTED_ORIGIN_IDENTITY_FIELDS
    }


def _validate_distributed_checkpoint_identity(
    brain: AdaptiveBrain,
    expected_config: OmniConfig,
) -> None:
    actual_identity = _distributed_origin_identity(brain.config)
    expected_identity = _distributed_origin_identity(expected_config)
    mismatches = [
        name
        for name in _DISTRIBUTED_ORIGIN_IDENTITY_FIELDS
        if actual_identity[name] != expected_identity[name]
    ]
    expected_curriculum = ground_up_curriculum_manifest()
    manifest = brain.ground_up_training_manifest
    if mismatches:
        raise RuntimeError(
            "distributed checkpoint does not match the requested architecture: %s"
            % ", ".join(mismatches)
        )
    if (
        brain.config.origin_kind != "ground-up"
        or not isinstance(manifest, Mapping)
        or manifest.get("id") != expected_curriculum["id"]
        or manifest.get("sha256") != expected_curriculum["sha256"]
        or not eligible_ground_up_rehearsal(brain)
        or not brain._ground_up_action_origin_verified
    ):
        raise RuntimeError(
            "distributed checkpoint is not the current verified "
            "foundation-free ground-up architecture"
        )


def _validate_distributed_origin_template(
    brain: AdaptiveBrain,
    expected_config: OmniConfig,
) -> None:
    """Accept only a pristine immutable origin for a new distributed run.

    The caller copies ``engine/origin`` before loading ``brain``.  These checks
    prevent a mutable post-Build checkpoint (including prior user data or
    conversation learning) from being presented as a fresh random-initialized
    run template.
    """

    _validate_distributed_checkpoint_identity(brain, expected_config)
    manifest = brain.ground_up_training_manifest
    validated_manifest = brain._validate_ground_up_training_manifest()
    if (
        not isinstance(manifest, Mapping)
        or validated_manifest != dict(manifest)
    ):
        raise RuntimeError(
            "initial brain must be the verified immutable origin of the "
            "current foundation-free ground-up curriculum"
        )
    verify_ternary_shards(brain.engine_path / "packed-ternary", retain_names=())
    conversation = brain.conversation.summary()
    current_context = (
        brain.current_context
        if isinstance(brain.current_context, Mapping)
        else {}
    )
    strict_pristine_transients = int(manifest.get("formatVersion", 0)) >= 3
    if (
        brain.messages
        or brain.traces
        or brain.training_sources
        or brain.ingestion_checkpoints
        or brain.completed_ingestions
        or brain.completed_chat_turns
        or (
            strict_pristine_transients
            and (
                len(brain.replay) != 0
                or len(brain.working_memory) != 0
                or brain.paged_working_memory.count() != 0
                or brain.workspace_items
                or brain.recent_token_context
                or brain.fresh_attention_boundary is not None
                or int(current_context.get("tokenCount", 0)) != 0
                or int(current_context.get("recentTokenCount", 0)) != 0
                or int(current_context.get("sensorySlots", 0)) != 0
                or str(current_context.get("tokenHash", "")) != ""
                or brain.memory_lifecycle.scratch_items
                or brain.memory_lifecycle.active_focus
            )
        )
        or brain.installed_modality_packs
        or int(brain.counters.get("inference_count", 0)) != 0
        or int(conversation.get("totalEntries", 0)) != 0
        or int(conversation.get("messageCount", 0)) != 0
        or int(conversation.get("actionCount", 0)) != 0
        or int(conversation.get("traceCount", 0)) != 0
    ):
        raise RuntimeError(
            "initial ground-up template contains post-origin learning or conversation state"
        )


@dataclass(frozen=True)
class DistributedTrainingOptions:
    epochs: int = 1
    global_batch_records: int = 16
    micro_batch_records: int = 0  # Auto: admitted capacity, not a fixed ceiling.
    gradient_accumulation: int = 0
    learning_rate: Optional[float] = None
    strategy: str = "auto"
    fsdp_min_parameter_bytes: int = 2 * 1024**3
    amp: str = "auto"
    checkpoint_steps: int = 1
    keep_checkpoints: int = 2
    resume: str = "auto"
    replace_output: bool = False
    failure_injection: str = ""
    capability_rehearsal_waves: int = 128

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("distributed training epochs must be positive")
        if self.global_batch_records < 1:
            raise ValueError("global batch records must be positive")
        if self.micro_batch_records < 0:
            raise ValueError("micro batch records must be auto (0) or positive")
        if self.gradient_accumulation < 0:
            raise ValueError("gradient accumulation cannot be negative")
        if self.strategy not in {"auto", "ddp", "fsdp"}:
            raise ValueError("strategy must be auto, ddp, or fsdp")
        if self.amp not in {"auto", "off", "fp16", "bf16"}:
            raise ValueError("AMP must be auto, off, fp16, or bf16")
        if self.checkpoint_steps < 1:
            raise ValueError("checkpoint steps must be positive")
        if self.keep_checkpoints < 1:
            raise ValueError("at least one checkpoint must be retained")
        if self.resume not in {"auto", "required", "never"}:
            raise ValueError("resume must be auto, required, or never")
        if self.capability_rehearsal_waves < 1:
            raise ValueError("capability rehearsal waves must be positive")


def _finite_wave_progress(
    *,
    record_count: int,
    global_batch_records: int,
    epochs: int,
    completed_epochs: int,
    next_global_ordinal: int,
) -> Tuple[int, int]:
    """Count only fully trained waves at a cursor eligible for publication."""

    if record_count < 1 or global_batch_records < 1 or epochs < 1:
        raise ValueError("finite rehearsal dataset or batch is invalid")
    if not 0 <= completed_epochs <= epochs:
        raise ValueError("finite rehearsal epoch cursor is invalid")
    if not 0 <= next_global_ordinal <= record_count:
        raise ValueError("finite rehearsal record cursor is invalid")
    if completed_epochs == epochs and next_global_ordinal != 0:
        raise ValueError("completed finite rehearsal epoch has a record cursor")
    if (
        next_global_ordinal not in {0, record_count}
        and next_global_ordinal % global_batch_records != 0
    ):
        raise ValueError("finite rehearsal cursor is not a completed wave")
    waves_per_epoch = (record_count + global_batch_records - 1) // global_batch_records
    completed_waves = (
        completed_epochs * waves_per_epoch
        + (
            next_global_ordinal + global_batch_records - 1
        ) // global_batch_records
    )
    return completed_waves, epochs * waves_per_epoch


def _due_distributed_rehearsal_phase(
    state: CapabilityScheduleState,
    policy: CapabilityRehearsalPolicy,
    *,
    committed_global_waves: int,
    record_count: int,
    global_batch_records: int,
    epochs: int,
    completed_epochs: int,
    next_global_ordinal: int,
    final: bool = False,
) -> Optional[str]:
    completed_waves, total_waves = _finite_wave_progress(
        record_count=record_count,
        global_batch_records=global_batch_records,
        epochs=epochs,
        completed_epochs=completed_epochs,
        next_global_ordinal=next_global_ordinal,
    )
    return due_rehearsal_phase(
        state,
        policy,
        committed_global_waves=committed_global_waves,
        final=final,
        completed_finite_waves=completed_waves,
        total_finite_waves=total_waves,
    )


@dataclass(frozen=True)
class DynamicNeuralUpdate:
    """One source-free, canonically ordered fast-plasticity operation."""

    epoch: int
    ordinal: int
    record_id: str
    labels: Tuple[str, ...]
    cue: torch.Tensor
    kind: str

    def validate(
        self, *, manifest: DatasetManifest, dimensions: int
    ) -> None:
        if self.epoch < 0 or self.ordinal < 0 or self.ordinal >= len(manifest.entries):
            raise ValueError("dynamic neural update has an invalid position")
        entry = manifest.entries[self.ordinal]
        if self.record_id != entry.record_id or self.kind != entry.kind:
            raise ValueError("dynamic neural update does not match the dataset manifest")
        if len(self.labels) > 4096 or any(
            not isinstance(value, str) or not value or len(value) > 512
            for value in self.labels
        ):
            raise ValueError("dynamic neural update labels are invalid")
        for name, tensor in (("cue", self.cue),):
            if not isinstance(tensor, torch.Tensor) or tensor.numel() != dimensions:
                raise ValueError("dynamic neural update %s shape is invalid" % name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("dynamic neural update %s is non-finite" % name)


class DistributedBrainTrainingModule(nn.Module):
    """The corpus objective over packed synapses and any residual controls.

    Only modules connected to the corpus objective are registered.  Growable
    dictionaries and STDP buffers stay outside the
    reducer and are merged through :class:`DynamicNeuralUpdate` instead.
    """

    def __init__(self, brain: AdaptiveBrain):
        super().__init__()
        self.decoder = brain.decoder
        self.memory_bridge = brain.memory_bridge
        self.idea_adapter = brain.idea_adapter
        self.liquid = brain.liquid
        # Avoid registering AdaptiveBrain itself (it is not an nn.Module) while
        # retaining access to metaplastic anchors and the immutable tokenizer.
        object.__setattr__(self, "_brain", brain)

    @property
    def brain(self) -> AdaptiveBrain:
        return object.__getattribute__(self, "_brain")

    def _zero_loss(self) -> torch.Tensor:
        values = [
            parameter.reshape(-1)[0] * 0.0
            for parameter in self.parameters()
            if parameter.numel()
        ]
        if not values:
            # A packed-only brain has no floating Parameter. Empty gradient
            # accumulation slots still need a differentiable zero; the
            # packed module's scalar autograd trigger is not a learned weight.
            values = [
                buffer.reshape(-1)[0] * 0.0
                for buffer in self.buffers()
                if buffer.requires_grad and buffer.numel()
            ]
        if not values:
            raise RuntimeError("distributed training module has no autograd path")
        return torch.stack(values).sum()

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vsa_vectors: torch.Tensor,
        noise: torch.Tensor,
        *,
        world_size: int,
        global_window_count: int,
        global_label_count: int,
        include_stability: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_window_count < 1 or global_label_count < 1:
            raise ValueError("global training wave contains no token windows")
        if ids.shape[0] == 0:
            loss = self._zero_loss()
            measurements = torch.zeros((6,), dtype=torch.float32, device=loss.device)
        else:
            if (
                ids.ndim != 2
                or attention_mask.shape != ids.shape
                or vsa_vectors.ndim != 2
                or noise.ndim != 2
                or vsa_vectors.shape[0] != ids.shape[0]
                or noise.shape[0] != ids.shape[0]
            ):
                raise ValueError("distributed training tensors do not align")
            idea = torch.tanh(
                self.memory_bridge(vsa_vectors.to(dtype=torch.float32))
            )
            reconstructed = self.idea_adapter(idea + noise.to(idea))
            idea_losses = F.mse_loss(
                reconstructed, idea.detach(), reduction="none"
            ).mean(dim=-1)
            liquid_state = self.brain.liquid_state.detach().to(idea).expand(
                idea.shape[0], -1
            )
            temporal, _controls = self.liquid(
                idea, state=liquid_state, elapsed=1.0
            )
            temporal_losses = F.mse_loss(
                temporal, idea.detach(), reduction="none"
            ).mean(dim=-1)
            embedded = self.decoder.embedding(ids)
            whole = self.decoder.global_workspace.summarize(
                embedded, attention_mask=attention_mask
            )
            workspace_losses = (
                F.normalize(whole, dim=-1) - F.normalize(idea.detach(), dim=-1)
            ).pow(2).mean(dim=-1)
            logits = self.decoder(
                ids,
                memory_bias=reconstructed,
                attention_mask=attention_mask,
            )["logits"]
            token_losses = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                ids[:, 1:].contiguous().view(-1),
                ignore_index=self.brain.tokenizer.pad_id,
                reduction="none",
            ).view(ids.shape[0], -1)
            prediction_mask = attention_mask[:, 1:]
            language_losses = (
                (token_losses * prediction_mask).sum(dim=1)
                / prediction_mask.sum(dim=1).clamp_min(1)
            )
            auxiliary = (
                0.2 * idea_losses
                + 0.05 * temporal_losses
                + 0.1 * workspace_losses
            )
            if not bool(torch.isfinite(auxiliary).all()) or not bool(torch.isfinite(token_losses).all()):
                raise RuntimeError("non-finite distributed corpus loss")
            # Context/padding is unlabelled. Short tails carry their actual
            # target count, not the weight of an entire padded long window.
            label_sum = (token_losses * prediction_mask).sum()
            loss = float(world_size) * (label_sum / float(global_label_count)
                + auxiliary.sum() / float(global_window_count))
            measurements = torch.stack(
                (
                    auxiliary.detach().sum(),
                    label_sum.detach(),
                    idea_losses.detach().sum(),
                    workspace_losses.detach().sum(),
                    torch.as_tensor(
                        float(ids.shape[0]), device=ids.device, dtype=torch.float32
                    ),
                    prediction_mask.sum().detach().float(),
                )
            ).float()
        if include_stability:
            stability = self.brain._stability_penalty(
                self.brain._streaming_experience_parameters()
            )
            if not bool(torch.isfinite(stability)):
                raise RuntimeError("non-finite distributed stability loss")
            # Every rank contributes the same stability objective; DDP's mean
            # therefore preserves it exactly rather than multiplying it.
            loss = loss + stability
        return loss, measurements


def _module_parameter_bytes(module: nn.Module) -> int:
    return sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in module.parameters()
    )


def _resolve_strategy(
    options: DistributedTrainingOptions,
    context: DistributedContext,
    module: nn.Module,
) -> str:
    if context.world_size == 1:
        return "single"
    if any(isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES) for child in module.modules()):
        return "packed-collective"
    if options.strategy == "fsdp":
        if (
            context.device.type != "cuda"
            or platform.system().lower() == "windows"
        ):
            raise ValueError(
                "FSDP requires a supported non-Windows multi-rank CUDA launch"
            )
        return "fsdp"
    if options.strategy == "ddp":
        return "ddp"
    if (
        context.device.type == "cuda"
        and platform.system().lower() != "windows"
        and _module_parameter_bytes(module) >= options.fsdp_min_parameter_bytes
    ):
        return "fsdp"
    return "ddp"


def _wrap_module(
    module: DistributedBrainTrainingModule,
    *,
    strategy: str,
    context: DistributedContext,
) -> nn.Module:
    has_packed = any(isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES) for child in module.modules())
    if has_packed and context.world_size > 1 and strategy != "packed-collective":
        raise RuntimeError("multi-rank native packed updates require the canonical packed collective, not DDP/FSDP or independent single replicas")
    # Packed derivatives use an explicit collective, not DDP/FSDP's parameter
    # reducer. Each rank shares the same authoritative ternary bytes.
    if strategy == "packed-collective":
        return module
    if strategy == "single":
        return module
    if strategy == "ddp":
        return DistributedDataParallel(
            module,
            device_ids=(
                [context.local_rank] if context.device.type == "cuda" else None
            ),
            output_device=(
                context.local_rank if context.device.type == "cuda" else None
            ),
            broadcast_buffers=False,
            # Tail waves can leave one rank with the differentiable zero path
            # while another owns real windows, so their used-parameter sets
            # legitimately differ even though full waves often use all roots.
            find_unused_parameters=True,
        )
    if strategy == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError as error:  # pragma: no cover - old unsupported torch.
            raise RuntimeError("this PyTorch build does not provide FSDP") from error
        return FSDP(
            module,
            device_id=context.device,
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
        )
    raise ValueError("unknown distributed strategy")


def _unwrap(module: nn.Module) -> DistributedBrainTrainingModule:
    current: nn.Module = module
    while hasattr(current, "module"):
        current = getattr(current, "module")
    if not isinstance(current, DistributedBrainTrainingModule):
        raise TypeError("distributed brain module wrapper is invalid")
    return current


def _new_training_optimizer(
    brain: AdaptiveBrain,
    module: DistributedBrainTrainingModule,
    learning_rate: Optional[float],
) -> torch.optim.Optimizer | PackedOnlyOptimizer:
    rate = max(
        1e-6,
        min(
            0.02,
            float(
                brain.config.learning_rate
                if learning_rate is None
                else learning_rate
            ),
        ),
    )
    connected_ids = {
        id(parameter)
        for parameter in brain._streaming_experience_parameters()
    }
    parameters = [
        parameter
        for parameter in module.parameters()
        if id(parameter) in connected_ids
    ]
    optimizer = adamw_for_remaining_parameters(
        parameters,
        lr=rate,
        weight_decay=brain.config.weight_decay,
    )
    # Native checkpoints retain a full optimizer (including tool/modality
    # parameters). Transfer only matching corpus states into this reducer-
    # owned optimizer; unrelated native moments remain untouched on save.
    for parameter in parameters:
        native_state = brain._optimizer.state.get(parameter)
        if native_state:
            optimizer.state[parameter] = _device_tree(
                native_state, parameter.device
            )
    return optimizer


def _amp_settings(
    amp: str, device: torch.device
) -> Tuple[bool, Optional[torch.dtype], bool]:
    requested = amp
    if requested == "auto":
        requested = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "off"
        )
    if requested == "off" or device.type == "mps":
        return False, None, False
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 AMP is supported only for CUDA distributed training")
        return True, torch.float16, True
    if requested == "bf16":
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("BF16 AMP is unavailable for this device")
        return True, torch.bfloat16, False
    raise ValueError("invalid AMP mode")


def _autocast_context(
    enabled: bool, dtype: Optional[torch.dtype], device: torch.device
) -> contextlib.AbstractContextManager[Any]:
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _deterministic_noise(
    *,
    manifest_sha256: str,
    epoch: int,
    record_id: str,
    window_index: int,
    dimensions: int,
) -> torch.Tensor:
    material = "%s:%d:%s:%d" % (
        manifest_sha256,
        int(epoch),
        record_id,
        int(window_index),
    )
    seed = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "little")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    return torch.randn((dimensions,), generator=generator, dtype=torch.float32) * 0.06


def _dynamic_update(
    brain: AdaptiveBrain,
    manifest: DatasetManifest,
    entry: DatasetManifestEntry,
    text: str,
    epoch: int,
) -> DynamicNeuralUpdate:
    clean = str(text).replace("\x00", "").strip()
    labels = tuple(brain.memory.extract_atomic_concepts(clean))
    if not labels:
        # Binary records and empty-but-valid structured values still receive a
        # source-free identity assembly. Content bytes are represented only by
        # their committed digest, never copied into checkpoint metadata.
        labels = (
            "%s perception" % entry.kind,
            "content %s" % entry.content_sha256[:24],
        )
    cue = brain.memory.vector_for_labels(labels).detach().cpu().float()
    return DynamicNeuralUpdate(
        epoch=int(epoch),
        ordinal=entry.ordinal,
        record_id=entry.record_id,
        labels=labels,
        cue=cue,
        kind=entry.kind,
    )


def apply_dynamic_updates(
    brain: AdaptiveBrain,
    manifest: DatasetManifest,
    updates: Sequence[DynamicNeuralUpdate],
    *,
    already_committed: int,
) -> int:
    """Validate, deduplicate and apply rank updates in canonical order."""

    dimensions = int(brain.config.vsa_dim)
    committed = max(0, int(already_committed))
    by_position: Dict[int, DynamicNeuralUpdate] = {}
    for update in updates:
        update.validate(manifest=manifest, dimensions=dimensions)
        position = update.epoch * len(manifest.entries) + update.ordinal
        if position < committed:
            continue
        previous = by_position.get(position)
        if previous is not None:
            if (
                previous.record_id != update.record_id
                or previous.labels != update.labels
                or not torch.equal(previous.cue, update.cue)
            ):
                raise ValueError("ranks emitted conflicting dynamic updates")
            # A duplicated position is a scheduling error even if its payload
            # matches. Silently coalescing it would hide duplicate ownership.
            raise ValueError("multiple ranks emitted the same dynamic update")
        by_position[position] = update

    ordered_positions = sorted(by_position)
    if ordered_positions != list(
        range(committed, committed + len(ordered_positions))
    ):
        raise ValueError(
            "dynamic rank updates contain a gap, duplicate, or out-of-order suffix"
        )
    # NeuralSubstrate timestamps are inspection metadata, but wall-clock time
    # would still make two equivalent rank layouts serialize differently.
    # This checkpoint brain is isolated and single-threaded, so temporarily
    # supply a stable logical timestamp while applying each canonical update.
    from . import vsa as vsa_module

    original_now = vsa_module._now
    try:
        for position in ordered_positions:
            update = by_position[position]
            logical_time = 1_700_000_000.0 + float(position) / 1_000_000.0
            vsa_module._now = lambda value=logical_time: value
            synthetic = " ".join(update.labels)
            brain.memory.learn_statistical(
                synthetic,
                kind=("sensory" if update.kind in {"image", "audio", "video"} else "knowledge"),
                source="distributed-dataset",
                source_label="distributed manifest record",
                importance=0.5,
            )
            with torch.no_grad():
                idea = torch.tanh(
                    brain.memory_bridge(
                        update.cue.to(brain.device).reshape(1, -1)
                    )
                )
                _routed, _metrics = brain.router.route(
                    idea,
                    steps=2,
                    learn=True,
                    threshold_offset=0.0,
                )
            brain.counters["experiences"] += 1
            brain.counters["plasticity_events"] = int(
                brain.router.synapses.plasticity_events.item()
            )
            committed = position + 1
    finally:
        vsa_module._now = original_now

    # Stable dictionaries must never contain duplicate logical rows.
    if len(brain.memory.neurons) != len(set(brain.memory.neurons)):
        raise RuntimeError("distributed substrate contains duplicate neuron rows")
    if len(brain.memory.synapses) != len(set(brain.memory.synapses)):
        raise RuntimeError("distributed substrate contains duplicate synapse rows")
    assembly_ids = [str(item.get("id", "")) for item in brain.memory.assemblies]
    if len(assembly_ids) != len(set(assembly_ids)):
        raise RuntimeError("distributed substrate contains duplicate assembly rows")
    return committed


def _media_parameter_checksum(brain: AdaptiveBrain) -> str:
    if hasattr(brain, "core_pager"):
        brain.core_pager.flush()
    return tensor_checksum(
        value for _name, value in sorted(brain.modalities.state_dict().items())
    )


class MonotonicManifestReplay:
    """One rank-zero verified source pass, resumed from one global position."""

    def __init__(self, manifest: DatasetManifest, start_position: int = 0, resource_admission=None):
        self.manifest = manifest
        self.cardinality = len(manifest.entries)
        self.position = max(0, int(start_position))
        self.resource_admission = resource_admission
        if self.cardinality == 0 and self.position:
            raise ValueError("manifest replay position exceeds an empty manifest")
        self.epoch = (
            self.position // self.cardinality if self.cardinality else 0
        )
        self._iterator: Optional[
            Iterator[Tuple[DatasetManifestEntry, Any]]
        ] = None

    def _open(self) -> None:
        if self.cardinality == 0:
            self._iterator = iter(())
            return
        start = self.position % self.cardinality
        self._iterator = iter(
            self.manifest.iter_verified_records(
                rank=0,
                world_size=1,
                start_ordinal=start,
                resource_admission=self.resource_admission,
            )
        )

    def consume_until(
        self, stop_position: int
    ) -> Iterator[Tuple[int, DatasetManifestEntry, Any]]:
        stop = max(0, int(stop_position))
        if stop < self.position:
            raise ValueError("manifest media replay cannot move backward")
        while self.position < stop:
            if self._iterator is None:
                self._open()
            assert self._iterator is not None
            try:
                entry, record = next(self._iterator)
            except StopIteration:
                if self.cardinality == 0:
                    raise ValueError("manifest replay ended before its cursor")
                self.epoch += 1
                self._iterator = None
                continue
            expected_ordinal = self.position % self.cardinality
            if entry.ordinal != expected_ordinal:
                raise ValueError("manifest replay ordinal diverged")
            position = self.position
            self.position += 1
            yield position, entry, record

    def close(self):
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None


def apply_media_updates(
    brain: AdaptiveBrain,
    records: Iterable[Tuple[int, DatasetManifestEntry, Any]],
) -> List[Dict[str, Any]]:
    """Reopen and train every newly committed media record exactly once.

    Archive members are consumed while their iterator lease is live. No path
    or raw media bytes enter the distributed checkpoint receipt.
    """

    reports: List[Dict[str, Any]] = []
    for position, entry, record in records:
        if entry.kind not in {"image", "audio", "video"}:
            continue
        local_path = str(record.local_path or "").strip()
        if not local_path:
            raise RuntimeError(
                "distributed media record has no leased decoder path"
            )
        effective_kind = brain._effective_media_kind(local_path, entry.kind)
        before = _media_parameter_checksum(brain)
        result = brain._train_media(
            local_path,
            effective_kind,
            "distributed manifest record",
            steps=1,
            content_sha256=entry.content_sha256,
            **({"speech_text": record.provenance["speech_text"]} if "speech_text" in record.provenance else {}),
        )
        after = _media_parameter_checksum(brain)
        coverage = result.get("coverage")
        if (
            not bool(result.get("trained", False))
            or int(result.get("steps", 0)) < 1
            or not isinstance(coverage, Mapping)
            or not bool(coverage.get("complete", False))
            or before == after
        ):
            raise RuntimeError(
                "distributed %s record did not mutate its modality pack with full coverage"
                % effective_kind
            )
        reports.append(
            {
                "position": int(position),
                "recordId": entry.record_id,
                "kind": effective_kind,
                "steps": int(result["steps"]),
                "loss": float(result.get("loss", 0.0)),
                "parameterChecksumBefore": before,
                "parameterChecksumAfter": after,
                "coverage": brain._safe_media_coverage(
                    effective_kind, coverage
                ),
                "rawSourceStored": False,
            }
        )
    return reports


def apply_authoritative_source_updates(brain, records):
    """Canonical fast learning from every literal source section, not labels.

    The surrounding native publication is the durable transaction boundary.
    A failure discards the private replica and leaves the previous pointer.
    """
    reports = []
    committed = None
    from . import vsa as vsa_module
    original_now = vsa_module._now
    try:
        for position, entry, record in records:
            if entry.kind in {"image", "audio", "video"}:
                reports.extend(apply_media_updates(brain, [(position, entry, record)]))
            else:
                payload = getattr(record, "text_payload", None)
                sections = payload.windows() if payload is not None else (
                    (piece, 0) for piece in brain._experience_chunks(record.text))
                for section_index, (piece, _end) in enumerate(sections):
                    logical_time = 1_700_000_000.0 + float(position) + float(section_index) / 1_000_000.0
                    vsa_module._now = lambda value=logical_time: value
                    brain.learn_experience(piece, kind="knowledge", source="distributed-dataset",
                        source_label="distributed manifest record", steps=0, importance=0.5,
                        structural_detail=True)
            committed = position + 1
    finally:
        vsa_module._now = original_now
    return committed, reports


def merge_media_training_state(
    prior: Optional[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    state = dict(prior) if isinstance(prior, Mapping) else {}
    records = max(0, int(state.get("trainedRecords", 0)))
    steps = max(0, int(state.get("steps", 0)))
    by_modality = {
        str(key): max(0, int(value))
        for key, value in (
            state.get("byModality", {}).items()
            if isinstance(state.get("byModality"), Mapping)
            else ()
        )
    }
    chain = str(state.get("receiptChainSha256", "0" * 64))
    samples = [
        dict(value)
        for value in state.get("recentReports", [])
        if isinstance(value, Mapping)
    ][-16:]
    for report in sorted(reports, key=lambda value: int(value["position"])):
        encoded = {
            key: value
            for key, value in report.items()
            if key not in {"rawText", "rawBytes", "path"}
        }
        chain = hashlib.sha256(
            chain.encode("ascii")
            + json.dumps(
                encoded,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        records += 1
        steps += max(0, int(report.get("steps", 0)))
        kind = str(report.get("kind", "unknown"))
        by_modality[kind] = by_modality.get(kind, 0) + 1
        samples.append(encoded)
        samples = samples[-16:]
    return {
        "format": "omni-distributed-media-training",
        "formatVersion": 1,
        "trainedRecords": records,
        "steps": steps,
        "byModality": dict(sorted(by_modality.items())),
        "receiptChainSha256": chain,
        "recentReports": samples,
        "allParameterChecksumsChanged": all(
            value.get("parameterChecksumBefore")
            != value.get("parameterChecksumAfter")
            for value in samples
        ),
        "rawSourceStored": False,
    }


def _pad_windows(
    brain: AdaptiveBrain,
    windows: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not windows:
        return (
            torch.empty((0, 2), dtype=torch.long, device=brain.device),
            torch.empty((0, 2), dtype=torch.bool, device=brain.device),
            torch.empty((0, brain.config.vsa_dim), dtype=torch.float32, device=brain.device),
            torch.empty((0, brain.config.idea_dim), dtype=torch.float32, device=brain.device),
        )
    width = max(int(values[0].numel()) for values in windows)
    ids = torch.full(
        (len(windows), width),
        brain.tokenizer.pad_id,
        dtype=torch.long,
        device=brain.device,
    )
    vectors: List[torch.Tensor] = []
    noises: List[torch.Tensor] = []
    for index, (token_ids, vector, noise) in enumerate(windows):
        values = token_ids.to(brain.device).reshape(-1)
        ids[index, : values.numel()] = values
        vectors.append(vector.detach().cpu().float().reshape(-1))
        noises.append(noise.detach().cpu().float().reshape(-1))
    return (
        ids,
        ids.ne(brain.tokenizer.pad_id),
        torch.stack(vectors).to(brain.device),
        torch.stack(noises).to(brain.device),
    )


def _optimizer_state_by_name(
    module: DistributedBrainTrainingModule,
    optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
) -> Dict[str, Any]:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    state = {
        names[id(parameter)]: _cpu_tree(value)
        for parameter, value in optimizer.state.items()
        if id(parameter) in names
    }
    return {
        "state": state,
        "paramGroups": [
            {
                **{
                    key: copy.deepcopy(value)
                    for key, value in group.items()
                    if key != "params"
                },
                "params": [
                    names[id(parameter)]
                    for parameter in group["params"]
                    if id(parameter) in names
                ],
            }
            for group in optimizer.param_groups
        ],
    }


@torch.no_grad()
def _canonicalize_distributed_learning_state(
    brain: AdaptiveBrain,
    optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
    parameters: Sequence[nn.Parameter],
) -> None:
    """Canonicalize remaining high-precision controls and Adam moments."""

    seen: set[int] = set()

    def canonicalize(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) in seen:
                return
            seen.add(id(value))
            if value.dtype == torch.float32 and value.is_contiguous():
                AdaptiveBrain._canonicalize_streaming_float_tensor(value)
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()
            ):
                raise RuntimeError("distributed learning state became non-finite")
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                canonicalize(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                canonicalize(nested)

    parameter_ids = {id(value) for value in parameters}
    named_ids = {
        name: id(parameter)
        for name, parameter in brain._named_slow_parameters().items()
    }
    for parameter in parameters:
        canonicalize(parameter)
        canonicalize(optimizer.state.get(parameter, {}))
    for name, anchor in brain.slow_anchors.items():
        if named_ids.get(name) in parameter_ids:
            canonicalize(anchor)
    for name, importance in brain.slow_importance.items():
        if named_ids.get(name) in parameter_ids:
            canonicalize(importance)


def _full_training_state(
    wrapped: nn.Module,
    optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
    *,
    strategy: str,
    context: DistributedContext,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Any]]]:
    module = _unwrap(wrapped)
    if strategy != "fsdp":
        if not context.is_rank_zero:
            return None, None
        return (
            {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
            _optimizer_state_by_name(module, optimizer),
        )

    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    with FSDP.state_dict_type(
        wrapped,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = wrapped.state_dict()
    optimizer_state = FSDP.full_optim_state_dict(
        wrapped, optimizer, rank0_only=True
    )
    if not context.is_rank_zero:
        return None, None
    normalized_model = {
        str(key).removeprefix("module.").removeprefix("_fsdp_wrapped_module."): value.detach().cpu().clone()
        for key, value in model_state.items()
    }
    raw_state = optimizer_state.get("state", {})
    normalized_optimizer = {
        "state": {
            str(key).removeprefix("module.").removeprefix("_fsdp_wrapped_module."): _cpu_tree(value)
            for key, value in raw_state.items()
        },
        "paramGroups": [
            {
                **{key: copy.deepcopy(value) for key, value in group.items() if key != "params"},
                "params": [
                    str(value).removeprefix("module.").removeprefix("_fsdp_wrapped_module.")
                    for value in group.get("params", [])
                ],
            }
            for group in optimizer_state.get("param_groups", [])
        ],
    }
    return normalized_model, normalized_optimizer


def _merge_optimizer_state(
    brain: AdaptiveBrain,
    module: DistributedBrainTrainingModule,
    named_state: Mapping[str, Any],
) -> None:
    parameters = dict(module.named_parameters())
    raw_state = named_state.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("distributed optimizer state is invalid")
    unknown = sorted(set(str(key) for key in raw_state).difference(parameters))
    if unknown:
        raise ValueError("distributed optimizer state names an unknown parameter")
    for name, value in raw_state.items():
        parameter = parameters[str(name)]
        brain._optimizer.state[parameter] = _device_tree(value, parameter.device)


def _copytree_transactional(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".%s-%s.next" % (destination.name, uuid.uuid4().hex))
    previous = destination.parent / (".%s-%s.previous" % (destination.name, uuid.uuid4().hex))
    shutil.copytree(source, temporary)
    replaced = False
    try:
        if destination.exists():
            os.replace(str(destination), str(previous))
            replaced = True
        os.replace(str(temporary), str(destination))
        if previous.exists():
            shutil.rmtree(previous)
        replaced = False
    except BaseException:
        if destination.exists() and replaced:
            shutil.rmtree(destination)
        if replaced and previous.exists():
            os.replace(str(previous), str(destination))
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if previous.exists():
            shutil.rmtree(previous)


def _rewrite_brain_device(path: Path, device: str) -> None:
    """Bind a private rank clone without changing learned neural state."""

    from .persistence import atomic_write_json, read_json

    metadata_path = Path(path) / "engine" / "brain.json"
    metadata = read_json(metadata_path)
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("brain checkpoint config is invalid")
    config = {**config, "device": str(device)}
    atomic_write_json(metadata_path, {**metadata, "config": config})


def _checkpoint_brain(
    *,
    store: DistributedRunStore,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    slow_anchors: Mapping[str, torch.Tensor],
    slow_importance: Mapping[str, torch.Tensor],
    manifest: DatasetManifest,
    dynamic_updates: Sequence[DynamicNeuralUpdate],
    dynamic_high_water: int,
    training_steps: int,
    final_pack: bool,
    rehearsal_phase: Optional[str],
    committed_global_waves: int,
    rehearsal_policy: CapabilityRehearsalPolicy,
    baseline_minimum_probability: float,
    media_replay: MonotonicManifestReplay,
) -> Tuple[
    AdaptiveBrain,
    int,
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    source = store.path / "checkpoint-brain"
    temporary = store.path / (".checkpoint-brain-%s.next" % uuid.uuid4().hex)
    shutil.copytree(source, temporary)
    brain: Optional[AdaptiveBrain] = None
    try:
        brain = AdaptiveBrain.load(temporary)
        module = DistributedBrainTrainingModule(brain)
        module.load_state_dict(dict(model_state), strict=True)
        _merge_optimizer_state(brain, module, optimizer_state)
        brain.slow_anchors = {
            name: value.detach().cpu().clone() for name, value in slow_anchors.items()
        }
        brain.slow_importance = {
            name: value.detach().cpu().clone() for name, value in slow_importance.items()
        }
        brain._sync_stability_state()
        committed = apply_dynamic_updates(
            brain,
            manifest,
            dynamic_updates,
            already_committed=dynamic_high_water,
        )
        media_reports = apply_media_updates(
            brain,
            media_replay.consume_until(committed),
        )
        brain.counters["training_steps"] = max(
            int(brain.counters.get("training_steps", 0)), int(training_steps)
        )
        rehearsal_receipt: Optional[Dict[str, Any]] = None
        if rehearsal_phase is not None:
            rehearsal_receipt = rehearse_capabilities(
                brain,
                phase=rehearsal_phase,
                committed_global_waves=committed_global_waves,
                policy=rehearsal_policy,
                baseline_minimum_probability=baseline_minimum_probability,
            )
            from .persistence import atomic_write_json

            atomic_write_json(
                brain.engine_path / "capability-rehearsal.json",
                rehearsal_receipt,
            )
        brain.save()
        if final_pack:
            brain.export_packed_ternary()
        brain.events.close()
        brain = None
        _copytree_transactional(temporary, source)
    finally:
        if brain is not None:
            brain.events.close()
        if temporary.exists():
            shutil.rmtree(temporary)
    result = AdaptiveBrain.load(source)
    return result, committed, rehearsal_receipt, media_reports


def _collect_objects(context: DistributedContext, value: Any) -> List[Any]:
    if not context.distributed:
        return [value]
    gathered: List[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _failure_payload(failure):
    status = getattr(failure, "status", None)
    message = "%s: %s" % (type(failure).__name__, str(failure)[:2_000])
    resource = isinstance(failure, (DatasetResourcePause, MemoryError)) or (
        isinstance(status, Mapping) and any(bool(status.get(key)) for key in ("recoverable", "paused", "memoryPressure", "diskPressure"))) or any(
            marker in message.lower() for marker in ("out of memory", "no space left", "disk is full", "cannot allocate memory"))
    return {"message": message, "resource": bool(resource), "status": dict(status) if isinstance(status, Mapping) else {}}


def _raise_phase_failures(values, label):
    failures = [value for value in values if value]
    if not failures:
        return
    selected = next((value for value in failures if isinstance(value, Mapping) and not value.get("resource")), failures[0])
    message = selected.get("message", "unknown phase failure") if isinstance(selected, Mapping) else str(selected)
    if isinstance(selected, Mapping) and selected.get("resource"):
        raise DatasetResourcePause("%s: %s" % (label, message), selected.get("status", {}))
    raise RuntimeError("%s: %s" % (label, message))


def _collective_cancelled(
    context: DistributedContext, local_cancelled: bool
) -> bool:
    value = torch.tensor(
        [1 if local_cancelled else 0],
        dtype=torch.int32,
        device=(context.device if context.backend == "nccl" else torch.device("cpu")),
    )
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def _broadcast_rank_zero_error(
    context: DistributedContext, error: Optional[str]
) -> None:
    values: List[Optional[str]] = [error if context.is_rank_zero else None]
    if context.distributed:
        dist.broadcast_object_list(values, src=0)
    _raise_phase_failures(values, "rank-zero phase failed")


def _apply_rehearsal_sync(
    *,
    wrapped: nn.Module,
    optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
    strategy: str,
    payload: Optional[Mapping[str, Any]],
) -> None:
    """Install rank-zero's rehearsed action heads on every dense replica."""

    if payload is None:
        return
    module = _unwrap(wrapped)

    def load_heads() -> None:
        module.decoder.action_policy.load_state_dict(
            dict(payload["languageActionHead"]), strict=True
        )
        module.decoder.internal_action_policy.load_state_dict(
            dict(payload["internalActionHead"]), strict=True
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(
            wrapped, recurse=True, writeback=True, rank0_only=False
        ):
            load_heads()
    else:
        load_heads()
    head_parameters = {
        *module.decoder.action_policy.parameters(),
        *module.decoder.internal_action_policy.parameters(),
    }
    for parameter in head_parameters:
        optimizer.state.pop(parameter, None)
    anchors = payload.get("slowAnchors")
    importance = payload.get("slowImportance")
    if not isinstance(anchors, Mapping) or not isinstance(importance, Mapping):
        raise ValueError("rehearsal metaplastic sync state is invalid")
    for name, value in anchors.items():
        if isinstance(value, torch.Tensor):
            module.brain.slow_anchors[str(name)] = value.detach().cpu().clone()
    for name, value in importance.items():
        if isinstance(value, torch.Tensor):
            module.brain.slow_importance[str(name)] = value.detach().cpu().clone()
    module.brain._sync_stability_state()


class DistributedGroundUpTrainer:
    """End-to-end exact-shard trainer used by the standalone CLI."""

    def __init__(
        self,
        *,
        context: DistributedContext,
        store: DistributedRunStore,
        dataset_path: Path,
        output_path: Path,
        config: OmniConfig,
        options: DistributedTrainingOptions,
        brain_id: str,
        requested_kind: str = "",
        initial_brain_path: Optional[Path] = None,
    ) -> None:
        options.validate()
        self.context = context
        self.store = store
        self.dataset_path = Path(dataset_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.config = config
        self.options = options
        self.brain_id = str(brain_id)
        self.requested_kind = str(requested_kind or "")
        self.initial_brain_path = (
            Path(initial_brain_path).resolve()
            if initial_brain_path is not None
            else None
        )
        self._signal_cancelled = False
        self._prior_handlers: Dict[int, Any] = {}

    def _install_signal_handlers(self) -> None:
        def request_cancel(_signum: int, _frame: Any) -> None:
            self._signal_cancelled = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._prior_handlers.items():
            signal.signal(signum, handler)
        self._prior_handlers.clear()

    def _training_policy_sha256(self):
        cached = getattr(self, "_training_policy_identity", None)
        if cached is not None:
            return cached
        options = {key: value for key, value in asdict(self.options).items()
                   if key not in {"resume", "failure_injection", "replace_output", "keep_checkpoints", "strategy", "amp", "fsdp_min_parameter_bytes"}}
        support = {}
        for name in ("brain.py", "datasets.py", "text_spool.py", "columnar_admission.py", "tokenizer.py",
            "record_window_wave.py", "distributed_seal.py", "distributed_runtime.py", "distributed_training.py",
            "packed_collective.py", "packed_collective_hooks.py", "collective_controls.py", "window_wave_buffer.py", "model.py", "modalities.py", "liquid.py", "spiking.py", "optimizers.py"):
            digest = hashlib.sha256()
            with open(Path(__file__).parent / name, "rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            support[name] = digest.hexdigest()
        self._training_policy_identity = bounded_json_sha256({"policy": "resource-admitted-window-waves/canonical-packed-mutation/source-fast-replay-v2",
            "options": options, "supportCodeSha256": support})
        return self._training_policy_identity

    def _acquire_run_leases(self):
        owner, local, error = None, None, None
        if self.context.is_rank_zero:
            try:
                owner = self.store.acquire_run_lease()
            except BaseException as failure:
                error = _failure_payload(failure)
        _broadcast_rank_zero_error(self.context, error)
        try:
            rank_path = self.store.ranks_path / ("rank-%05d" % self.context.rank)
            rank_path.mkdir(parents=True, exist_ok=True)
            local = DistributedRunLease(rank_path / ".rank-owner.lock")
        except BaseException as failure:
            error = _failure_payload(failure)
        errors = _collect_objects(self.context, error)
        if any(errors):
            if local is not None: local.close()
            if owner is not None: owner.close()
        _raise_phase_failures(errors, "distributed private rank replica is still owned by another run")
        return local, owner

    def _distributed_window_plan(self, brain, group_start, group_end, sequence_tokens=None):
        """Use real CPU/CUDA headroom and requested work, with no Auto cap."""
        base = brain._training_resource_plan()
        memory = base["memory"]
        first = group_start + (self.context.rank - group_start) % self.context.world_size
        owned = 0 if first >= group_end else 1 + (group_end - 1 - first) // self.context.world_size
        accumulation = max(1, int(self.options.gradient_accumulation))
        requested_micro = int(self.options.micro_batch_records) or max(1, math.ceil(max(1, owned) / accumulation))
        effective_work = max(1, owned, requested_micro * accumulation)
        resident = bool(brain._optimizer.state) and not brain._optimizer_offloaded
        plan = brain.resource_policy.training_plan(
            max_window_tokens=int(sequence_tokens or min(brain.config.max_seq_len, brain._runtime_training_max_seq_len)),
            requested_batch_size=requested_micro, requested_gradient_accumulation=accumulation,
            effective_batch_target=effective_work, require_physical_batch_divisor=False,
            trainable_parameter_bytes=int(memory["optimizerAndGradientBytes"]) // (1 if resident else 3),
            packed_update_scratch_bytes=int(memory["packedUpdateScratchBytes"]),
            activation_bytes_per_token=int(memory["activationBytesPerToken"]), optimizer_state_resident=resident,
            resource_mode=brain.config.training_resource_mode, manual_ram_budget_bytes=brain.config.training_ram_budget_bytes,
            manual_accelerator_budget_bytes=brain.config.training_accelerator_budget_bytes,
            manual_scratch_budget_bytes=brain.config.training_scratch_budget_bytes,
            storage_bytes_per_second=brain.config.storage_bytes_per_second, disk_state_offload=brain.config.disk_state_offload)
        selected_sequence = int(plan["windowTokens"])
        if sequence_tokens is not None and selected_sequence < sequence_tokens:
            raise DatasetResourcePause("resume is waiting for its frozen causal context", plan)
        physical = int(plan["physicalBatchRecords"])
        if hasattr(brain.decoder, "maximum_forward_tokens"):
            low, high = 1, physical
            while low < high:
                candidate = (low + high + 1) // 2
                if brain.decoder.maximum_forward_tokens(candidate) >= selected_sequence:
                    low = candidate
                else:
                    high = candidate - 1
            physical = low
            if brain.decoder.maximum_forward_tokens(physical) < selected_sequence:
                if sequence_tokens is not None:
                    raise DatasetResourcePause("frozen context cannot fit the working compute reservation", plan)
                selected_sequence = min(selected_sequence, int(brain.decoder.maximum_forward_tokens(physical)))
        if selected_sequence < 2 or plan["pauseBeforeStep"]:
            raise DatasetResourcePause("distributed labelled windows cannot fit admitted CPU/CUDA resources", plan)
        peak = (int(plan["memory"]["optimizerAndGradientBytes"]) + int(plan["memory"]["packedUpdateScratchBytes"])
            + int(plan["memory"]["allocatorMarginBytes"]) + physical * selected_sequence * int(plan["memory"]["activationBytesPerToken"]))
        plan.update(windowTokens=selected_sequence, physicalBatchRecords=physical,
            waveWindowTarget=effective_work, requestedRecordGroup=[group_start, group_end],
            requestedMicroBatchRecords=int(self.options.micro_batch_records),
            inputRamBudgetBytes=max(0, int(plan["memory"]["ramBudgetBytes"]) - peak),
            accumulationSlots=max(accumulation, math.ceil(effective_work / physical)))
        return plan

    def _prepare_manifest(self) -> DatasetManifest:
        error = None
        if self.context.is_rank_zero:
            try:
                from .offload import ResourcePolicy
                policy = ResourcePolicy(self.store.path, ram_reserve_bytes=self.config.ram_reserve_bytes,
                    disk_reserve_bytes=self.config.disk_reserve_bytes,
                    system_ram_share_percent=self.config.system_ram_share_percent,
                    hardware_tier=self.config.hardware_tier)
                def admit(stage, ram_bytes, disk_bytes):
                    status = policy.status(estimated_ram_bytes=ram_bytes, estimated_write_bytes=disk_bytes)
                    if status["memoryPressure"] or status["diskPressure"]:
                        raise DatasetResourcePause("%s is waiting for manifest parser resources" % stage, status)
                self.store.initialize()
                if self.store.manifest_path.is_file():
                    manifest = DatasetManifest.read(self.store.manifest_path)
                    manifest.verify_current_source(resource_admission=admit)
                else:
                    manifest = DatasetManifest.build(self.dataset_path, self.requested_kind,
                        database_path=self.store.manifest_path.with_suffix(".sqlite3"), resource_admission=admit)
                    manifest.write(self.store.manifest_path)
            except BaseException as failure:
                error = _failure_payload(failure)
        _broadcast_rank_zero_error(self.context, error)
        local_error, manifest = None, None
        try:
            manifest = DatasetManifest.read(self.store.manifest_path)
            if str(self.dataset_path) != manifest.source:
                raise ValueError("run directory belongs to another dataset path")
        except BaseException as failure:
            local_error = _failure_payload(failure)
        _raise_phase_failures(_collect_objects(self.context, local_error), "distributed manifest setup failed")
        return manifest

    def _new_ground_up_brain(self, path: Path) -> AdaptiveBrain:
        if self.initial_brain_path is not None:
            # A caller may point at a live brain only as a locator for its
            # immutable post-curriculum origin. Never copy the mutable current
            # checkpoint: it may contain prior user datasets, conversations,
            # replay, modality packs, or later online learning.
            origin = self.initial_brain_path / "engine" / "origin"
            if not origin.is_dir():
                raise RuntimeError(
                    "--initial-ground-up must name a brain with a verified "
                    "immutable engine/origin"
                )
            source: Optional[AdaptiveBrain] = None
            try:
                _copytree_transactional(origin, path / "engine")
                # Ground-up origin authentication intentionally resolves
                # through the normal load path. Recreate the nested immutable
                # reference so the copied current checkpoint can prove it came
                # from those exact bytes instead of trusting the caller's
                # directory label.
                _copytree_transactional(origin, path / "engine" / "origin")
                _rewrite_brain_device(path, "cpu")
                source = AdaptiveBrain.load(
                    path, expected_brain_id=self.brain_id
                )
                _validate_distributed_origin_template(source, self.config)
                return source
            except BaseException:
                if source is not None:
                    source.close()
                if path.exists():
                    shutil.rmtree(path)
                raise
        config = OmniConfig.from_dict(self.config.to_dict())
        config.origin_kind = "ground-up"
        # Durable checkpoint/promotion state stays CPU-portable. Each private
        # rank clone is rebound to LOCAL_RANK immediately before loading.
        config.device = "cpu"
        return AdaptiveBrain.create(
            self.brain_id,
            path,
            config,
            initialize_ground_up=True,
        )

    def _prepare_brains_rank_zero(self, manifest):
        checkpoint = None
        self.store.initialize()
        checkpoint = self.store.load_active_checkpoint(
            manifest_sha256=manifest.content_sha256,
            world_size=self.context.world_size,
        )
        if self.options.resume == "required" and checkpoint is None:
            raise RuntimeError("resume was required but no checkpoint exists")
        if self.options.resume == "never" and checkpoint is not None:
            raise RuntimeError("run already has a checkpoint; use a new run directory")
        checkpoint_brain_path = self.store.path / "checkpoint-brain"
        if checkpoint is None:
            if not self.store.template_path.exists():
                template = self._new_ground_up_brain(
                    self.store.template_path
                )
                template.close()
            template = AdaptiveBrain.load(
                self.store.template_path,
                expected_brain_id=self.brain_id,
            )
            try:
                _validate_distributed_origin_template(
                    template, self.config
                )
            finally:
                template.close()
            # A failed attempt without a published cursor owns no durable
            # progress. Always reset its working checkpoint before the
            # start rehearsal so retry cannot apply that event twice.
            _copytree_transactional(
                self.store.template_path, checkpoint_brain_path
            )
        else:
            _copytree_transactional(self.store.published_native_path(checkpoint), checkpoint_brain_path)
            recovered = AdaptiveBrain.load(checkpoint_brain_path)
            try:
                _validate_distributed_checkpoint_identity(
                    recovered, self.config
                )
                self._verify_native_seal(recovered, checkpoint)
            finally:
                recovered.close()
        schedule_state = CapabilityScheduleState.from_dict(
            checkpoint.get("capabilityRehearsal")
            if checkpoint is not None
            and isinstance(checkpoint.get("capabilityRehearsal"), Mapping)
            else None
        )
        media_training_state = (
            dict(checkpoint["mediaTraining"])
            if checkpoint is not None
            and isinstance(checkpoint.get("mediaTraining"), Mapping)
            else merge_media_training_state(None, ())
        )
        start_phase = due_rehearsal_phase(
            schedule_state,
            CapabilityRehearsalPolicy(
                periodic_global_waves=self.options.capability_rehearsal_waves
            ),
            committed_global_waves=(
                int(checkpoint.get("globalOptimizerSteps", 0))
                if checkpoint is not None
                else 0
            ),
        )
        if start_phase == "start":
            start_brain = AdaptiveBrain.load(checkpoint_brain_path)
            try:
                receipt = rehearse_capabilities(
                    start_brain,
                    phase="start",
                    committed_global_waves=0,
                    policy=CapabilityRehearsalPolicy(
                        periodic_global_waves=(
                            self.options.capability_rehearsal_waves
                        )
                    ),
                )
                schedule_state = advance_schedule_state(
                    schedule_state, receipt
                )
                from .persistence import atomic_write_json

                atomic_write_json(
                    start_brain.engine_path / "capability-rehearsal.json",
                    receipt,
                )
                start_brain.save()
            finally:
                start_brain.events.close()
        cursors = (
            [
                RankCursor.from_dict(value)
                for value in checkpoint["rankCursors"]
            ]
            if checkpoint is not None
            else initial_rank_cursors(
                world_size=self.context.world_size,
                manifest_sha256=manifest.content_sha256,
            )
        )
        dynamic_high_water = (
            int(checkpoint.get("dynamicHighWater", 0))
            if checkpoint is not None
            else 0
        )
        global_steps = int(checkpoint.get("globalOptimizerSteps", 0)) if checkpoint else 0
        payload: List[Any] = [
            [cursor.to_dict() for cursor in cursors],
            dynamic_high_water,
            global_steps,
            checkpoint is not None,
            schedule_state.to_dict(),
            media_training_state,
        ]
        return payload

    def _prepare_brains(
        self, manifest: DatasetManifest
    ) -> Tuple[
        AdaptiveBrain,
        List[RankCursor],
        int,
        int,
        bool,
        CapabilityScheduleState,
        Dict[str, Any],
    ]:
        error = None
        payload = [None, None, None, None, None, None]
        if self.context.is_rank_zero:
            try:
                payload = self._prepare_brains_rank_zero(manifest)
            except BaseException as failure:
                error = _failure_payload(failure)
        _broadcast_rank_zero_error(self.context, error)
        if self.context.distributed:
            dist.broadcast_object_list(payload, src=0)
        cursors = [RankCursor.from_dict(value) for value in payload[0]]
        dynamic_high_water = int(payload[1])
        global_steps = int(payload[2])
        resumed = bool(payload[3])
        schedule_state = CapabilityScheduleState.from_dict(payload[4])
        media_training_state = dict(payload[5])

        self.context.barrier()
        rank_brain_path = self.store.ranks_path / ("rank-%05d" % self.context.rank) / "brain"
        brain, local_error = None, None
        try:
            _copytree_transactional(self.store.path / "checkpoint-brain", rank_brain_path)
            _rewrite_brain_device(rank_brain_path, str(self.context.device))
            brain = AdaptiveBrain.load(rank_brain_path)
            _validate_distributed_checkpoint_identity(brain, self.config)
        except BaseException as failure:
            local_error = _failure_payload(failure)
        errors = _collect_objects(self.context, local_error)
        if any(errors) and brain is not None:
            brain.close()
        _raise_phase_failures(errors, "distributed native replica setup failed")
        return (
            brain,
            cursors,
            dynamic_high_water,
            global_steps,
            resumed,
            schedule_state,
            media_training_state,
        )

    def _bind_actual_native_device(self, brain):
        actual_devices = _collect_objects(self.context, str(brain.device))
        if self.context.backend == "nccl" and any(not value.startswith("cuda") for value in actual_devices):
            raise DatasetResourcePause("native CPU fallback requires relaunch with CPU/Gloo; an existing NCCL group cannot migrate", {
                "actualDevices": actual_devices, "backend": self.context.backend, "relaunchRequired": True})
        self.context = replace(self.context, device=torch.device(brain.device))

    def _verify_native_seal(self, brain, checkpoint):
        seal = brain.distributed_training_seal
        if seal is None or seal["contentSha256"] != checkpoint.get("distributedSealSha256") or seal["rankCursors"] != checkpoint["rankCursors"] or seal["manifestSha256"] != checkpoint["manifestSha256"] or seal["trainingPolicySha256"] != self._training_policy_sha256() or seal["topologySha256"] != native_topology_sha256(brain):
            raise ValueError("distributed resume does not match its independently committed native cursor/topology/policy seal")

    def _failure_injection_matches(self, global_step: int) -> bool:
        raw = self.options.failure_injection.strip()
        if not raw:
            return False
        try:
            rank_text, step_text = raw.split(":", 1)
            rank = int(rank_text)
            step = int(step_text)
        except (ValueError, TypeError) as error:
            raise ValueError("failure injection must be RANK:GLOBAL_STEP") from error
        if rank != self.context.rank or step != global_step:
            return False
        marker = self.store.failures_path / (
            "injected-rank-%05d-step-%08d.once" % (rank, step)
        )
        try:
            descriptor = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        else:
            os.close(descriptor)
            return True

    def _rank_records_for_wave(
        self,
        brain: AdaptiveBrain,
        manifest: DatasetManifest,
        epoch: int,
        records: Sequence[Tuple[DatasetManifestEntry, Any]],
    ) -> Tuple[
        List[Tuple[DatasetManifestEntry, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]],
        List[DynamicNeuralUpdate],
    ]:
        prepared = []
        dynamic = []
        for entry, literal_window in records:
            if literal_window.human is not None:
                seal = getattr(brain, "distributed_training_seal", None) or {}
                key = (epoch, entry.record_id, literal_window.human.sha256, seal.get("committedRecordStop", 0))
                cached = getattr(self, "_leased_human_cue_cache", None)
                if cached is None or cached[0] != key:
                    cached = (key, brain._leased_dialogue_cue(literal_window.human).detach().cpu().float())
                    self._leased_human_cue_cache = cached
                cue = cached[1]
            else:
                cue = brain.memory.vector_for_text(literal_window.text or " ").detach().cpu().float()
            prepared.append((entry, [(
                torch.tensor(literal_window.ids, dtype=torch.long), cue,
                _deterministic_noise(manifest_sha256=manifest.content_sha256, epoch=epoch,
                    record_id=entry.record_id, window_index=literal_window.window_index,
                    dimensions=brain.config.idea_dim),
            )]))
        return prepared, dynamic

    def _train_wave(
        self,
        *,
        brain: AdaptiveBrain,
        wrapped: nn.Module,
        optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
        scaler: Any,
        amp_enabled: bool,
        amp_dtype: Optional[torch.dtype],
        local_records: PreparedWindowWave,
        manifest: DatasetManifest,
        epoch: int,
    ) -> Tuple[Dict[str, float], List[DynamicNeuralUpdate], bool]:
        direct_packed = any(
            isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES)
            for child in _unwrap(wrapped).modules()
        )
        if isinstance(optimizer, PackedOnlyOptimizer) and not direct_packed:
            raise RuntimeError("corpus objective has no learnable weights")
        if direct_packed and amp_enabled:
            raise RuntimeError(
                "direct packed backward updates require unscaled execution"
            )
        controller = None
        if direct_packed:
            controller = getattr(self, "_packed_collective_controller", None)
            if controller is None:
                def reserve(**amounts):
                    if self._signal_cancelled or self.store.cancel_requested():
                        raise DatasetResourcePause("distributed cancellation stopped an uncommitted packed window", {"cancelled": True})
                    status = brain.resource_policy.status(
                        estimated_ram_bytes=amounts.get("ram_bytes", 0),
                        estimated_write_bytes=amounts.get("disk_bytes", 0))
                    if status["memoryPressure"] or status["diskPressure"]:
                        raise DatasetResourcePause("packed collective is waiting for derivative/rollback resources", status)
                controller = PackedCollectiveController(
                    {name: child for name, child in _unwrap(wrapped).named_modules()
                     if isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES)},
                    self.context, self.store.path / ("rank-packed-scratch-%05d" % self.context.rank),
                    _apply_packed_gradient_rows, reserve)
                self._packed_collective_controller = controller
            step_id = "%s:%d:%d" % (manifest.content_sha256, epoch, int(brain.counters["training_steps"]))
        if not isinstance(local_records, PreparedWindowWave):
            raise ValueError("distributed learner requires a bounded prepared-input wave")
        dynamic = []
        local_window_count = local_records.window_count
        count = torch.tensor(
            [local_window_count, local_records.label_count],
            dtype=torch.int64,
            device=(self.context.device if self.context.backend == "nccl" else torch.device("cpu")),
        )
        if self.context.distributed:
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_window_count = int(count[0].item())
        global_label_count = int(count[1].item())
        if global_window_count == 0:
            return {
                "loss": 0.0,
                "languageLoss": 0.0,
                "ideaLoss": 0.0,
                "workspaceLoss": 0.0,
                "windows": 0.0,
            }, dynamic, False
        if controller is not None:
            controller.begin(step_id)

        local_required = max(1, math.ceil(local_window_count / local_records.physical_batch))
        required = torch.tensor(
            [local_required],
            dtype=torch.int64,
            device=(self.context.device if self.context.backend == "nccl" else torch.device("cpu")),
        )
        if self.context.distributed:
            dist.all_reduce(required, op=dist.ReduceOp.MAX)
        slots = max(int(required.item()), int(self.options.gradient_accumulation))
        chunks = iter(local_records.batches())

        optimizer.zero_grad(set_to_none=True)
        measurement = torch.zeros((6,), dtype=torch.float64, device=(
            self.context.device if self.context.backend == "nccl" else torch.device("cpu")
        ))
        for slot in range(slots):
            final = slot == slots - 1
            sync_context = (
                contextlib.nullcontext()
                if final or not hasattr(wrapped, "no_sync")
                else wrapped.no_sync()  # type: ignore[attr-defined]
            )
            local_error = None
            try:
                windows = next(chunks, [])
                ids, mask, vectors, noise = _pad_windows(brain, windows)
                with sync_context, packed_derivative_sink(controller):
                    with _autocast_context(amp_enabled, amp_dtype, self.context.device):
                        loss, measured = wrapped(
                            ids, mask, vectors, noise,
                            world_size=1 if controller is not None else self.context.world_size,
                            global_window_count=global_window_count,
                            global_label_count=global_label_count,
                            include_stability=final and (controller is None or self.context.is_rank_zero))
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("non-finite distributed corpus loss")
                    if direct_packed:
                        loss.backward()
                    else:
                        scaler.scale(loss).backward()
            except BaseException as failure:
                local_error = _failure_payload(failure)
            slot_errors = _collect_objects(self.context, local_error)
            if any(slot_errors):
                if controller is not None: controller.rollback()
                optimizer.zero_grad(set_to_none=True)
                _raise_phase_failures(slot_errors, "distributed backward slot failed")
            measurement += measured.to(measurement).detach()

        if not direct_packed:
            scaler.unscale_(optimizer)
        if controller is not None:
            # Residual scalar controls, if any, get the same explicit SUM as
            # packed derivatives. There is no private DDP gradient identity.
            for _name, parameter in sorted(_unwrap(wrapped).named_parameters()):
                present = torch.tensor([int(parameter.grad is not None)], dtype=torch.int32,
                    device=self.context.device if self.context.backend == "nccl" else torch.device("cpu"))
                dist.all_reduce(present, op=dist.ReduceOp.MAX)
                if not present.item(): continue
                if parameter.grad is None: parameter.grad = torch.zeros_like(parameter)
                communication_device = self.context.device if self.context.backend == "nccl" else torch.device("cpu")
                def prepare_control():
                    return parameter.grad.to(communication_device).contiguous()
                reduced = controller._phase(prepare_control)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                controller._phase(lambda: parameter.grad.copy_(reduced.to(parameter.grad.device)))
        local_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in _unwrap(wrapped).parameters()
        )
        finite = torch.tensor(
            [1 if local_finite else 0],
            dtype=torch.int32,
            device=(self.context.device if self.context.backend == "nccl" else torch.device("cpu")),
        )
        if self.context.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            if controller is not None: controller.rollback()
            optimizer.zero_grad(set_to_none=True)
            if not direct_packed:
                scaler.update()
            raise RuntimeError("one or more ranks produced non-finite gradients")

        parameters = [
            parameter
            for parameter in _unwrap(wrapped).parameters()
            if parameter.grad is not None
        ]
        if isinstance(optimizer, PackedOnlyOptimizer) and parameters:
            raise RuntimeError(
                "packed-only optimizer received unexpected floating gradients"
            )
        if controller is not None:
            controller._phase(brain._drain_packed_stability_events)
            controller._phase(lambda: brain._accumulate_slow_importance(parameters) if self.context.is_rank_zero else None)
        else:
            brain._accumulate_slow_importance(parameters)
        if not parameters:
            norm = measurement.new_zeros(())
        elif hasattr(wrapped, "clip_grad_norm_") and not isinstance(
            wrapped, DistributedDataParallel
        ):
            norm = wrapped.clip_grad_norm_(brain.config.grad_clip)  # type: ignore[attr-defined]
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, brain.config.grad_clip, error_if_nonfinite=True
            )
        if not bool(torch.isfinite(torch.as_tensor(norm))):
            raise RuntimeError("distributed gradient norm is non-finite")
        if direct_packed:
            if controller is not None: controller.commit(retain_rollback=True)
            try:
                if controller is not None:
                    controller._phase(lambda: optimizer.step() if self.context.is_rank_zero else None)
                else:
                    optimizer.step()
            except BaseException:
                if controller is not None: controller.rollback()
                raise
        else:
            scaler.step(optimizer)
            scaler.update()
        def finish_controls():
            brain._commit_slow_anchors(rate=0.08, parameters=parameters)
            if _resolve_strategy(self.options, self.context, _unwrap(wrapped)) != "fsdp":
                _canonicalize_distributed_learning_state(brain, optimizer, parameters)
        if controller is not None:
            controller._phase(lambda: finish_controls() if self.context.is_rank_zero else None)
            synchronize_control_state(controller, _unwrap(wrapped), optimizer, brain)
        else:
            finish_controls()
        optimizer.zero_grad(set_to_none=True)
        if controller is not None: controller.finalize()
        brain.counters["training_steps"] += 1
        if self.context.distributed:
            dist.all_reduce(measurement, op=dist.ReduceOp.SUM)
        divisor = max(1.0, float(measurement[4].item()))
        labels = max(1.0, float(measurement[5].item()))
        return {
            "loss": float(measurement[0].item()) / divisor + float(measurement[1].item()) / labels,
            "languageLoss": float(measurement[1].item()) / labels,
            "ideaLoss": float(measurement[2].item()) / divisor,
            "workspaceLoss": float(measurement[3].item()) / divisor,
            "windows": float(measurement[4].item()),
            "labels": float(measurement[5].item()),
            "physicalMicrobatchWindows": float(local_records.physical_batch),
            "accumulationSlots": float(slots),
            "gradientNorm": float(torch.as_tensor(norm).detach().cpu().item()),
        }, dynamic, True

    def _checkpoint(
        self,
        *,
        brain: AdaptiveBrain,
        wrapped: nn.Module,
        optimizer: torch.optim.Optimizer | PackedOnlyOptimizer,
        manifest: DatasetManifest,
        cursors: Sequence[RankCursor],
        local_dynamic: Sequence[DynamicNeuralUpdate],
        dynamic_high_water: int,
        global_steps: int,
        strategy: str,
        final_pack: bool,
        schedule_state: CapabilityScheduleState,
        rehearsal_phase: Optional[str],
        media_training_state: Mapping[str, Any],
        media_replay: Optional[MonotonicManifestReplay],
    ) -> Tuple[int, CapabilityScheduleState, Dict[str, Any]]:
        if any(isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES) for child in _unwrap(wrapped).modules()):
            return self._checkpoint_native_collective(brain=brain, wrapped=wrapped, optimizer=optimizer,
                manifest=manifest, cursors=cursors, dynamic_high_water=dynamic_high_water,
                global_steps=global_steps, strategy=strategy, final_pack=final_pack,
                schedule_state=schedule_state, rehearsal_phase=rehearsal_phase,
                media_training_state=media_training_state, media_replay=media_replay)
        gathered_dynamic = _collect_objects(self.context, list(local_dynamic))
        readings = _collect_objects(
            self.context,
            sample_resources(
                rank=self.context.rank,
                device=self.context.device,
                path=brain.engine_path,
                policy_status=brain.resource_policy.status(),
            ),
        )
        model_state, optimizer_state = _full_training_state(
            wrapped, optimizer, strategy=strategy, context=self.context
        )
        error: Optional[str] = None
        committed = int(dynamic_high_water)
        next_schedule_state = schedule_state
        next_media_training_state = dict(media_training_state)
        rehearsal_sync: Optional[Dict[str, Any]] = None
        if self.context.is_rank_zero:
            checkpoint_brain: Optional[AdaptiveBrain] = None
            try:
                telemetry = aggregate_resource_readings(
                    [value for value in readings if isinstance(value, ResourceReading)]
                )
                if bool(telemetry["diskPressure"]):
                    raise RuntimeError(
                        "distributed checkpoint paused at the disk reserve"
                    )
                updates = list(
                    itertools.chain.from_iterable(
                        value for value in gathered_dynamic if isinstance(value, list)
                    )
                )
                if model_state is None or optimizer_state is None:
                    raise RuntimeError("rank zero did not receive full training state")
                if media_replay is None:
                    raise RuntimeError("rank zero media replay cursor is missing")
                (
                    checkpoint_brain,
                    committed,
                    rehearsal_receipt,
                    media_reports,
                ) = _checkpoint_brain(
                    store=self.store,
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                    slow_anchors=_unwrap(wrapped).brain.slow_anchors,
                    slow_importance=_unwrap(wrapped).brain.slow_importance,
                    manifest=manifest,
                    dynamic_updates=updates,
                    dynamic_high_water=dynamic_high_water,
                    training_steps=int(
                        _unwrap(wrapped).brain.counters.get(
                            "training_steps", global_steps
                        )
                    ),
                    final_pack=final_pack,
                    rehearsal_phase=rehearsal_phase,
                    committed_global_waves=global_steps,
                    rehearsal_policy=CapabilityRehearsalPolicy(
                        periodic_global_waves=(
                            self.options.capability_rehearsal_waves
                        )
                    ),
                    baseline_minimum_probability=(
                        schedule_state.baseline_minimum_probability
                    ),
                    media_replay=media_replay,
                )
                if rehearsal_receipt is not None:
                    next_schedule_state = advance_schedule_state(
                        schedule_state, rehearsal_receipt
                    )
                    action_names = {
                        name
                        for name in checkpoint_brain.slow_anchors
                        if name.startswith("decoder.action_policy.")
                        or name.startswith("decoder.internal_action_policy.")
                        or name.startswith("modalities.imagination_selector.")
                    }
                    rehearsal_sync = {
                        "languageActionHead": _cpu_tree(
                            checkpoint_brain.decoder.action_policy.state_dict()
                        ),
                        "internalActionHead": _cpu_tree(
                            checkpoint_brain.decoder.internal_action_policy.state_dict()
                        ),
                        "slowAnchors": {
                            name: checkpoint_brain.slow_anchors[name]
                            for name in sorted(action_names)
                        },
                        "slowImportance": {
                            name: checkpoint_brain.slow_importance[name]
                            for name in sorted(action_names)
                        },
                    }
                next_media_training_state = merge_media_training_state(
                    media_training_state, media_reports
                )
                checkpoint = self.store.publish_checkpoint(
                    brain_json_path=checkpoint_brain.engine_path / "brain.json",
                    manifest=manifest,
                    cursors=cursors,
                    epochs_requested=self.options.epochs,
                    global_optimizer_steps=global_steps,
                    dynamic_high_water=committed,
                    strategy=strategy,
                    telemetry=telemetry,
                    capability_rehearsal=next_schedule_state.to_dict(),
                    media_training=next_media_training_state,
                )
                self.store.append_telemetry(
                    {
                        **telemetry,
                        "globalOptimizerSteps": global_steps,
                        "checkpoint": checkpoint["contentSha256"],
                    }
                )
                self.store.prune_checkpoints(self.options.keep_checkpoints)
            except BaseException as caught:
                error = "%s: %s" % (type(caught).__name__, caught)
            finally:
                if checkpoint_brain is not None:
                    checkpoint_brain.events.close()
        _broadcast_rank_zero_error(self.context, error)
        values: List[Any] = [
            committed if self.context.is_rank_zero else None,
            (
                next_schedule_state.to_dict()
                if self.context.is_rank_zero
                else None
            ),
            rehearsal_sync if self.context.is_rank_zero else None,
            (
                next_media_training_state
                if self.context.is_rank_zero
                else None
            ),
        ]
        if self.context.distributed:
            dist.broadcast_object_list(values, src=0)
        _apply_rehearsal_sync(
            wrapped=wrapped,
            optimizer=optimizer,
            strategy=strategy,
            payload=values[2],
        )
        return (
            int(values[0]),
            CapabilityScheduleState.from_dict(values[1]),
            dict(values[3]),
        )

    def _checkpoint_native_collective(self, *, brain, wrapped, optimizer, manifest, cursors,
        dynamic_high_water, global_steps, strategy, final_pack, schedule_state,
        rehearsal_phase, media_training_state, media_replay):
        """Publish one native identity without a whole-model CPU state clone."""
        readings = _collect_objects(self.context, sample_resources(rank=self.context.rank,
            device=self.context.device, path=brain.engine_path, policy_status=brain.resource_policy.status()))
        result, error = None, None
        if self.context.is_rank_zero:
            try:
                telemetry = aggregate_resource_readings([value for value in readings if isinstance(value, ResourceReading)])
                if telemetry["diskPressure"] or telemetry.get("memoryPressure", False):
                    raise DatasetResourcePause("canonical native checkpoint is waiting for physical resources", telemetry)
                if media_replay is None or media_replay.position != dynamic_high_water:
                    raise ValueError("canonical source replay is not at its published record boundary")
                positions = [cursor.epoch * len(manifest.entries) + cursor.next_global_ordinal for cursor in cursors]
                committed_stop = min(positions)
                if committed_stop < dynamic_high_water:
                    raise ValueError("canonical record coverage cannot move backward")
                # Optimizer controls belong to the same parameter objects; no
                # full-precision projection mirror or private state average.
                for parameter, state in optimizer.state.items():
                    brain._optimizer.state[parameter] = state
                committed, media_reports = apply_authoritative_source_updates(
                    brain, media_replay.consume_until(committed_stop))
                if committed is not None and committed != committed_stop:
                    raise ValueError("canonical source replay did not reach complete record coverage")
                if media_replay.position != committed_stop:
                    raise ValueError("canonical source replay cursor did not exhaust its requested prefix")
                next_schedule = schedule_state
                if rehearsal_phase is not None:
                    if any(cursor.record_window is not None for cursor in cursors):
                        raise RuntimeError("architecture/capability rehearsal cannot mutate an active record transaction")
                    receipt = rehearse_capabilities(brain, phase=rehearsal_phase,
                        committed_global_waves=global_steps,
                        policy=CapabilityRehearsalPolicy(periodic_global_waves=self.options.capability_rehearsal_waves),
                        baseline_minimum_probability=schedule_state.baseline_minimum_probability)
                    next_schedule = advance_schedule_state(schedule_state, receipt)
                next_media = merge_media_training_state(media_training_state, media_reports)
                seal = make_distributed_training_seal(manifest_sha256=manifest.content_sha256,
                    topology_sha256=native_topology_sha256(brain), training_policy_sha256=self._training_policy_sha256(),
                    record_count=len(manifest.entries), epochs=self.options.epochs, cursors=cursors,
                    committed_record_stop=committed_stop, global_steps=global_steps)
                brain.stage_distributed_training_seal(seal)
                brain.save()
                if final_pack:
                    brain.export_packed_ternary()
                source_path = brain.storage_path
                brain.close()
                copy_bytes = sum(path.stat().st_size for path in source_path.rglob("*") if path.is_file())
                brain.resource_policy.require_disk(copy_bytes, "canonical distributed native checkpoint copy")
                checkpoint = self.store.publish_checkpoint(brain_json_path=source_path / "engine" / "brain.json",
                    manifest=manifest, cursors=cursors, epochs_requested=self.options.epochs,
                    global_optimizer_steps=global_steps, dynamic_high_water=committed_stop,
                    strategy=strategy, telemetry=telemetry, capability_rehearsal=next_schedule.to_dict(),
                    media_training=next_media, native_brain_path=source_path)
                brain.resource_policy.require_disk(copy_bytes, "canonical distributed working replica copy")
                _copytree_transactional(self.store.published_native_path(checkpoint), self.store.path / "checkpoint-brain")
                self.store.append_telemetry({**telemetry, "globalOptimizerSteps": global_steps,
                    "checkpoint": checkpoint["contentSha256"]})
                self.store.prune_checkpoints(self.options.keep_checkpoints)
                result = [committed_stop, next_schedule.to_dict(), next_media, checkpoint["contentSha256"]]
            except BaseException as failure:
                error = _failure_payload(failure)
        _broadcast_rank_zero_error(self.context, error)
        values = [result]
        if self.context.distributed:
            dist.broadcast_object_list(values, src=0)
        if not isinstance(values[0], list) or len(values[0]) != 4:
            raise RuntimeError("canonical native checkpoint publication did not return an identity")
        self._last_native_publication_sha256 = values[0][3]
        return int(values[0][0]), CapabilityScheduleState.from_dict(values[0][1]), dict(values[0][2])

    def _refresh_native_replica(self, brain, manifest):
        """All fast/slow/nonweight owners come from the same publication."""
        controller = getattr(self, "_packed_collective_controller", None)
        if controller is not None:
            if controller.step_id:
                raise RuntimeError("cannot refresh native topology during an active packed transaction")
            controller.close()
            self._packed_collective_controller = None
        path = brain.storage_path
        policy = brain.resource_policy
        brain.close()
        local_error, restored = None, None
        try:
            checkpoint = self.store.load_active_checkpoint(manifest_sha256=manifest.content_sha256,
                world_size=self.context.world_size)
            if checkpoint is None or checkpoint["contentSha256"] != self._last_native_publication_sha256:
                raise RuntimeError("rank cannot see the exact canonical native publication")
            canonical = self.store.published_native_path(checkpoint)
            copy_bytes = sum(item.stat().st_size for item in canonical.rglob("*") if item.is_file())
            # The previous replica is quiescent; admission never changes the
            # requested architecture or skips the current leased record.
            policy.require_disk(copy_bytes, "distributed replica checkpoint refresh")
            _copytree_transactional(canonical, path)
            _rewrite_brain_device(path, str(self.context.device))
            restored = AdaptiveBrain.load(path)
            _validate_distributed_checkpoint_identity(restored, self.config)
            self._verify_native_seal(restored, checkpoint)
        except BaseException as failure:
            local_error = _failure_payload(failure)
        errors = _collect_objects(self.context, local_error)
        if any(errors):
            if restored is not None:
                restored.close()
            _raise_phase_failures(errors, "native replica refresh failed")
        return restored

    def _promote_output(
        self,
        *,
        manifest: DatasetManifest,
        cursors: Sequence[RankCursor],
        global_steps: int,
        dynamic_high_water: int,
        strategy: str,
        schedule_state: CapabilityScheduleState,
        media_training_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        expected_records = len(manifest.entries) * self.options.epochs
        coverage = _safe_distributed_coverage(manifest.coverage)
        expected_media_records = self.options.epochs * sum(
            int(coverage["modalityCounts"].get(name, 0))
            for name in ("image", "audio", "video")
        )
        ordered_cursors = sorted(cursors, key=lambda value: value.rank)
        visited_records = sum(
            int(value.owned_records_completed) for value in ordered_cursors
        )
        run_identity = _canonical_sha256(
            {
                "brainId": self.brain_id,
                "datasetManifestSha256": manifest.content_sha256,
                "epochs": self.options.epochs,
                "worldSize": self.context.world_size,
                "strategy": str(strategy),
            }
        )
        if (
            [value.rank for value in ordered_cursors]
            != list(range(self.context.world_size))
            or any(
                value.world_size != self.context.world_size
                or value.epoch != self.options.epochs
                or value.next_global_ordinal != 0
                or value.manifest_sha256 != manifest.content_sha256
                for value in ordered_cursors
            )
            or visited_records != expected_records
            or int(dynamic_high_water) != expected_records
            or not schedule_state.final_completed
            or not _final_capability_receipt_ready(
                schedule_state.last_receipt
            )
            or coverage.get("complete") is not True
            or int(coverage.get("processedRecords", -1))
            != len(manifest.entries)
            or int(coverage.get("discoveredRecords", -1))
            != int(coverage.get("processedRecords", -1))
            + int(coverage.get("rejectedRecords", -1))
            or int(coverage.get("discoveredFiles", -1))
            != int(coverage.get("processedFiles", -1))
            + int(coverage.get("rejectedFiles", -1))
            or int(media_training_state.get("trainedRecords", -1))
            != expected_media_records
            or media_training_state.get("rawSourceStored") is not False
            or (
                expected_media_records > 0
                and media_training_state.get("allParameterChecksumsChanged")
                is not True
            )
        ):
            raise RuntimeError(
                "distributed promotion requires complete final cursors and capability readiness"
            )
        final_checkpoint = self.store.load_active_checkpoint(
            manifest_sha256=manifest.content_sha256,
            world_size=self.context.world_size,
        )
        if not isinstance(final_checkpoint, Mapping):
            raise RuntimeError("distributed promotion has no final checkpoint")
        final_telemetry = final_checkpoint.get("telemetry")
        per_rank_telemetry = (
            final_telemetry.get("perRank")
            if isinstance(final_telemetry, Mapping)
            else None
        )
        if (
            not isinstance(final_telemetry, Mapping)
            or int(final_telemetry.get("rankCount", 0))
            != self.context.world_size
            or not isinstance(per_rank_telemetry, list)
            or len(per_rank_telemetry) != self.context.world_size
            or [
                row.get("rank") if isinstance(row, Mapping) else None
                for row in per_rank_telemetry
            ]
            != list(range(self.context.world_size))
            or final_telemetry.get("diskPressure") is True
            or final_checkpoint.get("rankCursors")
            != [value.to_dict() for value in ordered_cursors]
            or int(final_checkpoint.get("globalOptimizerSteps", -1))
            != int(global_steps)
            or int(final_checkpoint.get("dynamicHighWater", -1))
            != expected_records
            or final_checkpoint.get("capabilityRehearsal")
            != schedule_state.to_dict()
            or final_checkpoint.get("mediaTraining")
            != dict(media_training_state)
        ):
            raise RuntimeError(
                "distributed promotion checkpoint accounting is incomplete"
            )
        telemetry_ledger = _telemetry_ledger_receipt(
            self.store.telemetry_path
        )
        expected_dataset_receipt = {
            "manifestSha256": manifest.content_sha256,
            "recordChainSha256": manifest.record_chain_sha256,
            "validRecords": len(manifest.entries),
            "epochs": self.options.epochs,
            "recordsExpected": expected_records,
            "recordsVisited": visited_records,
            "dynamicHighWater": int(dynamic_high_water),
            "coverage": coverage,
            "rawSourceStored": False,
        }
        expected_resource_receipt = {
            "finalCheckpoint": _safe_resource_telemetry(final_telemetry),
            "telemetryLedger": telemetry_ledger,
            "checkpointContentSha256": final_checkpoint.get(
                "contentSha256"
            ),
        }
        source = self.store.path / "checkpoint-brain"
        existing_receipt = (
            self.output_path / "engine" / "distributed-training.json"
        )
        if self.output_path.is_dir() and existing_receipt.is_file():
            with existing_receipt.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            if isinstance(value, dict) and value.get("formatVersion") == 2:
                verified_receipt = _validate_promotion_receipt(
                    value, run_identity
                )
                if (
                    verified_receipt.get("dataset")
                    != expected_dataset_receipt
                    or verified_receipt.get("rankCursors")
                    != [cursor.to_dict() for cursor in ordered_cursors]
                    or verified_receipt.get("globalOptimizerSteps")
                    != int(global_steps)
                    or verified_receipt.get("capabilityRehearsal")
                    != schedule_state.to_dict()
                    or verified_receipt.get("mediaTraining")
                    != dict(media_training_state)
                    or verified_receipt.get("resources")
                    != expected_resource_receipt
                ):
                    raise RuntimeError(
                        "existing distributed promotion receipt does not "
                        "match the final run accounting"
                    )
                existing = AdaptiveBrain.load(
                    self.output_path, expected_brain_id=self.brain_id
                )
                try:
                    packed = verify_ternary_shards(
                        existing.engine_path / "packed-ternary", retain_names=()
                    ).manifest
                    packed_metadata = packed.get("metadata")
                    packed_coverage = packed.get("coverage")
                    existing_ground = existing.ground_up_training_manifest
                    expected_curriculum = ground_up_curriculum_manifest()
                    if int(expected_curriculum.get("formatVersion", 0)) >= 3:
                        existing_ground = (
                            validate_ground_up_v3_training_manifest(
                                existing_ground
                            )
                        )
                    existing_ground_receipt = (
                        existing_ground.get("trainingReceipt")
                        if isinstance(existing_ground, Mapping)
                        else None
                    )
                    embedded = next(
                        (
                            item.get("distributed_training_receipt")
                            for item in existing.training_sources
                            if item.get("distributed_receipt_sha256")
                            == verified_receipt.get("contentSha256")
                        ),
                        None,
                    )
                    if (
                        existing.config.origin_kind != "ground-up"
                        or not eligible_ground_up_rehearsal(existing)
                        or not existing._ground_up_action_origin_verified
                        or not isinstance(existing_ground, Mapping)
                        or not isinstance(existing_ground_receipt, Mapping)
                        or not isinstance(packed_metadata, Mapping)
                        or not isinstance(packed_coverage, Mapping)
                        or verified_receipt.get("groundUpCurriculum")
                        != {
                            "id": expected_curriculum["id"],
                            "sha256": expected_curriculum["sha256"],
                        }
                        or verified_receipt.get(
                            "groundUpTrainingManifestSha256"
                        )
                        != _verified_content_sha256(
                            existing_ground,
                            "ground-up training manifest",
                        )
                        or verified_receipt.get(
                            "groundUpTrainingReceiptSha256"
                        )
                        != _verified_content_sha256(
                            existing_ground_receipt,
                            "ground-up training receipt",
                        )
                        or existing.parameter_checksum()
                        != verified_receipt.get("parameterChecksum")
                        or existing.parameter_accounting()
                        != verified_receipt.get("parameterAccounting")
                        or verified_receipt.get("parameterEvidence", {}).get(
                            "before"
                        )
                        != existing_ground_receipt.get(
                            "parameterChecksumAfter"
                        )
                        or (
                            int(
                                verified_receipt.get("dataset", {}).get(
                                    "recordsExpected", 0
                                )
                            )
                            > 0
                            and verified_receipt.get(
                                "parameterEvidence", {}
                            ).get("changed")
                            is not True
                        )
                        or packed.get("contentSha256")
                        != verified_receipt.get("packedTernary", {}).get(
                            "contentSha256"
                        )
                        or packed_coverage.get("complete") is not True
                        or packed_coverage.get("eligibleTensorCount")
                        != verified_receipt.get("packedTernary", {}).get(
                            "eligibleTensorCount"
                        )
                        or packed_metadata.get("parameterChecksum")
                        != verified_receipt.get("parameterChecksum")
                        or packed_metadata.get("originKind") != "ground-up"
                        or packed_metadata.get("baseFrozen") is not False
                        or packed_metadata.get("groundUpCurriculumSha256")
                        != expected_curriculum["sha256"]
                        or packed_metadata.get(
                            "groundUpTrainingManifestSha256"
                        )
                        != existing_ground.get("contentSha256")
                        or packed_metadata.get(
                            "groundUpTrainingReceiptSha256"
                        )
                        != existing_ground_receipt.get("contentSha256")
                        or embedded != verified_receipt
                    ):
                        raise RuntimeError(
                            "existing distributed promotion no longer matches its receipt"
                        )
                finally:
                    existing.close()
                return {
                    **verified_receipt,
                    "idempotentCompletion": True,
                }
        if self.output_path.exists() and not self.options.replace_output:
            raise FileExistsError(
                "%s already exists; pass --replace-output to replace it transactionally"
                % self.output_path
            )
        _copytree_transactional(source, self.output_path)
        brain = AdaptiveBrain.load(self.output_path)
        try:
            ground_manifest = brain.ground_up_training_manifest
            curriculum = ground_up_curriculum_manifest()
            if int(curriculum.get("formatVersion", 0)) >= 3:
                ground_manifest = validate_ground_up_v3_training_manifest(
                    ground_manifest
                )
            if (
                brain.config.origin_kind != "ground-up"
                or not isinstance(ground_manifest, Mapping)
                or not eligible_ground_up_rehearsal(brain)
                or not brain._ground_up_action_origin_verified
                or ground_manifest.get("id") != curriculum["id"]
                or ground_manifest.get("sha256") != curriculum["sha256"]
            ):
                raise RuntimeError(
                    "distributed promotion is not the current foundation-free ground-up curriculum"
                )
            ground_receipt = ground_manifest.get("trainingReceipt")
            if not isinstance(ground_receipt, Mapping):
                raise RuntimeError(
                    "distributed promotion has no ground-up training receipt"
                )
            packed = verify_ternary_shards(
                brain.engine_path / "packed-ternary", retain_names=()
            ).manifest
            packed_metadata = packed.get("metadata")
            packed_coverage = packed.get("coverage")
            parameter_checksum = brain.parameter_checksum()
            if (
                not isinstance(packed_metadata, Mapping)
                or not isinstance(packed_coverage, Mapping)
                or packed_coverage.get("complete") is not True
                or packed_metadata.get("parameterChecksum")
                != parameter_checksum
                or packed_metadata.get("originKind") != "ground-up"
                or packed_metadata.get("baseFrozen") is not False
                or packed_metadata.get("groundUpCurriculumSha256")
                != curriculum["sha256"]
                or packed_metadata.get("groundUpTrainingManifestSha256")
                != ground_manifest.get("contentSha256")
                or packed_metadata.get("groundUpTrainingReceiptSha256")
                != ground_receipt.get("contentSha256")
            ):
                raise RuntimeError(
                    "distributed promotion packed ternary readiness is incomplete"
                )
            origin_parameter_checksum = ground_receipt.get(
                "parameterChecksumAfter"
            )
            if not _sha256_identifier(origin_parameter_checksum):
                raise RuntimeError(
                    "distributed promotion origin parameter checksum is invalid"
                )
            parameters_changed = (
                origin_parameter_checksum != parameter_checksum
            )
            if expected_records > 0 and not parameters_changed:
                raise RuntimeError(
                    "distributed dataset completed without changing neural parameters"
                )
            receipt_body = {
                "format": "omni-distributed-ground-up-promotion",
                "formatVersion": 2,
                "promotedAt": _utc_now(),
                "brainId": brain.brain_id,
                "runIdentitySha256": run_identity,
                "parameterChecksum": parameter_checksum,
                "parameterEvidence": {
                    "before": origin_parameter_checksum,
                    "after": parameter_checksum,
                    "changed": parameters_changed,
                },
                "parameterAccounting": brain.parameter_accounting(),
                "originKind": "ground-up",
                "externalWeightFiles": [],
                "groundUpCurriculum": {
                    "id": curriculum["id"],
                    "sha256": curriculum["sha256"],
                },
                "groundUpTrainingManifestSha256": (
                    _verified_content_sha256(
                        ground_manifest,
                        "ground-up training manifest",
                    )
                ),
                "groundUpTrainingReceiptSha256": (
                    _verified_content_sha256(
                        ground_receipt,
                        "ground-up training receipt",
                    )
                ),
                "dataset": expected_dataset_receipt,
                "rankCursors": [
                    value.to_dict() for value in ordered_cursors
                ],
                "globalOptimizerSteps": int(global_steps),
                "capabilityRehearsal": schedule_state.to_dict(),
                "mediaTraining": dict(media_training_state),
                "resources": expected_resource_receipt,
                "packedTernary": {
                    "contentSha256": packed.get("contentSha256"),
                    "parameterChecksum": packed_metadata.get(
                        "parameterChecksum"
                    ),
                    "eligibleTensorCount": packed_coverage.get(
                        "eligibleTensorCount"
                    ),
                    "coverageComplete": True,
                },
                "rlhf": False,
                "rewardModel": False,
                "preferenceLabels": False,
                "runtimeReady": True,
            }
            receipt = {
                **receipt_body,
                "contentSha256": _canonical_sha256(receipt_body),
            }
            origin_substrate = ground_receipt.get("substrateAfter")
            origin_counts = (
                origin_substrate
                if isinstance(origin_substrate, Mapping)
                else {}
            )
            source_id = "distributed-%s" % manifest.content_sha256[:32]
            source_record = {
                "id": source_id,
                "name": Path(manifest.source).name or "distributed dataset",
                "kind": "dataset",
                "dataset_kind": manifest.requested_kind or "auto",
                "bytes": int(coverage.get("processedBytes", 0)),
                "content_hash": manifest.content_sha256,
                "policy": "pretrain",
                "imported_at": receipt["promotedAt"],
                "last_trained_at": receipt["promotedAt"],
                "training_epochs": self.options.epochs,
                "learned_records": visited_records,
                "assemblies_created": max(
                    0,
                    len(brain.memory.assemblies)
                    - int(origin_counts.get("assemblies", 0)),
                ),
                "semantic_neurons_created": max(
                    0,
                    len(brain.memory.neurons)
                    - int(origin_counts.get("neurons", 0)),
                ),
                "sparse_synapses_created": max(
                    0,
                    len(brain.memory.synapses)
                    - int(origin_counts.get("synapses", 0)),
                ),
                "canonical_source_record_updates": expected_records,
                "parameter_update_steps": int(global_steps),
                "parameter_checksum_before": ground_receipt.get(
                    "parameterChecksumAfter"
                ),
                "parameter_checksum_after": parameter_checksum,
                "parameters_changed": (
                    parameters_changed
                ),
                "raw_text_retained": False,
                "coverage": coverage,
                "distributed_receipt_sha256": receipt["contentSha256"],
                # This bounded receipt remains inside brain.json and therefore
                # survives portable .omni export/import even when the local
                # convenience file below is omitted.
                "distributed_training_receipt": receipt,
            }
            existing_source = next(
                (
                    item
                    for item in brain.training_sources
                    if item.get("id") == source_id
                ),
                None,
            )
            if existing_source is not None and existing_source != source_record:
                raise RuntimeError(
                    "distributed training source conflicts with its promotion receipt"
                )
            if existing_source is None:
                brain.training_sources.append(source_record)
            brain.save()
            from .persistence import atomic_write_json

            atomic_write_json(
                brain.engine_path / "distributed-training.json", receipt
            )
            return receipt
        finally:
            brain.close()

    def run(self) -> Dict[str, Any]:
        self._install_signal_handlers()
        brain: Optional[AdaptiveBrain] = None
        leases = ()
        write_authority = False
        try:
            leases = self._acquire_run_leases()
            write_authority = True
            manifest = self._prepare_manifest()
            if _collective_cancelled(
                self.context,
                self._signal_cancelled or self.store.cancel_requested(),
            ):
                if self.context.is_rank_zero:
                    self.store.write_status(
                        state="cancelled",
                        reason="cancellation requested before model allocation",
                        resumable=self.store.active_path.is_file(),
                    )
                return {"state": "cancelled", "resumable": self.store.active_path.is_file()}
            (
                brain,
                cursors,
                dynamic_high_water,
                global_steps,
                resumed,
                schedule_state,
                media_training_state,
            ) = (
                self._prepare_brains(manifest)
            )
            self._bind_actual_native_device(brain)
            media_replay = (
                MonotonicManifestReplay(manifest, dynamic_high_water)
                if self.context.is_rank_zero
                else None
            )
            cursor = cursors[self.context.rank]
            module = DistributedBrainTrainingModule(brain).to(self.context.device)
            strategy = _resolve_strategy(self.options, self.context, module)
            wrapped = _wrap_module(
                module, strategy=strategy, context=self.context
            )
            optimizer = _new_training_optimizer(
                brain, module, self.options.learning_rate
            )
            # Packed-authoritative synapses consume the real backward signal
            # immediately. GradScaler cannot unscale an update already made
            # inside backward, so keep this path entirely unscaled.
            direct_packed = any(
                isinstance(child, PACKED_AUTHORITATIVE_PROJECTION_TYPES)
                for child in module.modules()
            )
            if not direct_packed:
                raise RuntimeError("distributed corpus runs require the native packed-authoritative architecture")
            amp_enabled, amp_dtype, scaled_amp = (
                (False, None, False)
                if direct_packed
                else _amp_settings(self.options.amp, self.context.device)
            )
            scaler = _make_grad_scaler(scaled_amp)
            if self.context.is_rank_zero:
                self.store.write_status(
                    state="running",
                    environment=self.context.status(),
                    strategy=strategy,
                    amp=(str(amp_dtype).replace("torch.", "") if amp_enabled else "off"),
                    manifestSha256=manifest.content_sha256,
                    validRecords=len(manifest.entries),
                    epochsRequested=self.options.epochs,
                    resumed=resumed,
                    cancelPath=str(self.store.cancel_path),
                    outputPath=str(self.output_path),
                    noPretrainedFoundation=True,
                    rlhf=False,
                    failures=self.store.recent_failures(),
                )

            pending_dynamic: List[DynamicNeuralUpdate] = []
            last_metrics: Dict[str, float] = {}
            rehearsal_policy = CapabilityRehearsalPolicy(
                periodic_global_waves=self.options.capability_rehearsal_waves
            )

            def phase_at_completed_wave(
                *, completed_epochs: int, next_global_ordinal: int, final: bool = False
            ) -> Optional[str]:
                return _due_distributed_rehearsal_phase(
                    schedule_state,
                    rehearsal_policy,
                    committed_global_waves=global_steps,
                    record_count=len(manifest.entries),
                    global_batch_records=self.options.global_batch_records,
                    epochs=self.options.epochs,
                    completed_epochs=completed_epochs,
                    next_global_ordinal=next_global_ordinal,
                    final=final,
                )

            def admit_source(stage, ram_bytes, disk_bytes):
                status = brain.resource_policy.status(estimated_ram_bytes=ram_bytes, estimated_write_bytes=disk_bytes)
                if status["memoryPressure"] or status["diskPressure"]:
                    raise DatasetResourcePause("%s is waiting for distributed parser resources" % stage, status)

            if media_replay is not None:
                media_replay.resource_admission = admit_source

            def refresh_replica():
                nonlocal brain, module, wrapped, optimizer, scaler
                brain = self._refresh_native_replica(brain, manifest)
                self._bind_actual_native_device(brain)
                module = DistributedBrainTrainingModule(brain).to(self.context.device)
                wrapped = _wrap_module(module, strategy=strategy, context=self.context)
                optimizer = _new_training_optimizer(brain, module, self.options.learning_rate)
                scaler = _make_grad_scaler(False)

            while cursor.epoch < self.options.epochs:
                epoch = cursor.epoch
                if any(value.epoch != epoch for value in cursors):
                    raise ValueError("distributed ranks disagree on the requested epoch boundary")
                start = cursor.next_global_ordinal
                owned_iterator = iter(
                    manifest.iter_verified_records(
                        rank=self.context.rank,
                        world_size=self.context.world_size,
                        start_ordinal=start,
                        resource_admission=admit_source,
                    )
                )
                active_stream = None
                active_entry = None
                minimum_ordinal = min(value.next_global_ordinal for value in cursors)
                wave_start = (minimum_ordinal if minimum_ordinal == len(manifest.entries) else
                    minimum_ordinal // self.options.global_batch_records * self.options.global_batch_records)
                while wave_start < len(manifest.entries):
                    if _collective_cancelled(
                        self.context,
                        self._signal_cancelled or self.store.cancel_requested(),
                    ):
                        dynamic_high_water, schedule_state, media_training_state = self._checkpoint(
                            brain=brain, wrapped=wrapped, optimizer=optimizer, manifest=manifest,
                            cursors=cursors, local_dynamic=(), dynamic_high_water=dynamic_high_water,
                            global_steps=global_steps, strategy=strategy, final_pack=False,
                            schedule_state=schedule_state, rehearsal_phase=None,
                            media_training_state=media_training_state, media_replay=media_replay)
                        if self.context.is_rank_zero:
                            self.store.write_status(
                                state="cancelled",
                                reason="cancellation requested",
                                resumable=True,
                                globalOptimizerSteps=global_steps,
                            )
                        return {
                            "state": "cancelled",
                            "resumable": True,
                            "globalOptimizerSteps": global_steps,
                        }
                    wave_end = min(
                        len(manifest.entries),
                        wave_start + self.options.global_batch_records,
                    )
                    # The requested record group bounds lookahead. Each rank
                    # opens only one record lease at a time and prepares its
                    # data before advancing that lease.
                    local_records, source_error = None, None
                    owned_completed = cursor.owned_records_completed
                    next_ordinal = cursor.next_global_ordinal
                    next_window = cursor.record_window
                    try:
                        frozen_sequences = {value.record_window["sequenceTokens"] for value in cursors if value.record_window is not None}
                        if len(frozen_sequences) > 1:
                            raise ValueError("distributed ranks have incompatible frozen record-window shapes")
                        frozen_sequence = next(iter(frozen_sequences)) if frozen_sequences else None
                        plan = self._distributed_window_plan(brain, wave_start, wave_end, frozen_sequence)
                        proposed_sequence = int(plan["windowTokens"])
                    except BaseException as failure:
                        source_error = _failure_payload(failure)
                        proposed_sequence = 0
                    source_errors = _collect_objects(self.context, source_error)
                    _raise_phase_failures(source_errors, "distributed record-wave admission failed")
                    proposals = _collect_objects(self.context, proposed_sequence)
                    sequence_tokens = next(iter(frozen_sequences)) if frozen_sequences else min(proposals)
                    if min(proposals) < sequence_tokens:
                        raise DatasetResourcePause("resume is waiting for its frozen causal-window resources", {
                            "requestedSequenceTokens": sequence_tokens, "availableSequenceTokens": min(proposals)})
                    source_error = None
                    try:
                        # Re-admit the chosen common context with this rank's
                        # own CPU/CUDA budget. Physical batches can differ,
                        # and adapt only between committed optimizer waves.
                        plan = self._distributed_window_plan(brain, wave_start, wave_end, sequence_tokens)
                        def reserve_input(**amounts):
                            if self._signal_cancelled or self.store.cancel_requested():
                                raise DatasetResourcePause("cancelled during input-wave preparation", {"cancelled": True})
                            admit_source("distributed input wave", amounts.get("ram_bytes", 0), amounts.get("disk_bytes", 0))
                        local_records = PreparedWindowWave(physical_batch=plan["physicalBatchRecords"],
                            ram_budget=plan["inputRamBudgetBytes"], directory=self.store.path / ("rank-input-scratch-%05d" % self.context.rank),
                            reserve=reserve_input, allow_spill=brain.config.disk_state_offload)
                        while next_ordinal < wave_end and local_records.window_count < plan["waveWindowTarget"]:
                            owned_ordinal = next_ordinal + (self.context.rank - next_ordinal) % self.context.world_size
                            if owned_ordinal >= wave_end:
                                next_ordinal, next_window = wave_end, None
                                break
                            else:
                                if active_stream is None:
                                    active_entry, record = next(owned_iterator)
                                    if active_entry.ordinal != owned_ordinal:
                                        raise ValueError("rank-owned record order differs from its exact global group")
                                    active_stream = RecordWindowStream(record, active_entry, brain.tokenizer,
                                        sequence_tokens, cursor.record_window)
                                    cursor_window = cursor.record_window
                                    if cursor_window is not None:
                                        cursor = RankCursor(cursor.rank, cursor.world_size, cursor.epoch,
                                            cursor.next_global_ordinal, cursor.owned_records_completed,
                                            cursor.optimizer_steps_completed, cursor.manifest_sha256)
                                literal = [(active_entry, item) for item in active_stream.next_batch(
                                    min(plan["physicalBatchRecords"], plan["waveWindowTarget"] - local_records.window_count))]
                                prepared, _ = self._rank_records_for_wave(brain, manifest, epoch, literal)
                                for _entry, values in prepared:
                                    for ids, cue, noise in values:
                                        local_records.append(ids, cue, noise)
                                if active_stream.complete:
                                    owned_completed += 1
                                    next_ordinal, next_window = min(wave_end, active_entry.ordinal + self.context.world_size), None
                                    active_stream.close()
                                    active_stream, active_entry = None, None
                                else:
                                    next_ordinal, next_window = active_entry.ordinal, active_stream.state()
                                    next_window["lastWavePlan"] = {"physicalBatchWindows": plan["physicalBatchRecords"],
                                        "waveWindowTarget": plan["waveWindowTarget"], "recordGroup": [wave_start, wave_end],
                                        "adaptAtCommittedBoundary": True}
                    except BaseException as failure:
                        source_error = _failure_payload(failure)
                    source_errors = _collect_objects(self.context, source_error)
                    _raise_phase_failures(source_errors, "distributed exact record traversal failed before mutation")
                    try:
                        metrics, dynamic, stepped = self._train_wave(brain=brain, wrapped=wrapped,
                            optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                            local_records=local_records, manifest=manifest, epoch=epoch)
                    finally:
                        if local_records is not None: local_records.close()
                    last_metrics = metrics
                    if stepped:
                        global_steps += 1
                    if self._failure_injection_matches(global_steps):
                        raise RuntimeError(
                            "injected distributed rank failure before cursor publication"
                        )
                    cursor = RankCursor(
                        rank=self.context.rank,
                        world_size=self.context.world_size,
                        epoch=epoch,
                        next_global_ordinal=next_ordinal,
                        owned_records_completed=owned_completed,
                        optimizer_steps_completed=(
                            cursor.optimizer_steps_completed + int(stepped)
                        ),
                        manifest_sha256=manifest.content_sha256,
                        record_window=next_window,
                    )
                    cursor_payloads = _collect_objects(
                        self.context, cursor.to_dict()
                    )
                    cursors = [RankCursor.from_dict(value) for value in cursor_payloads]
                    group_complete = all(value.next_global_ordinal >= wave_end and value.record_window is None for value in cursors)
                    rehearsal_phase = phase_at_completed_wave(
                        completed_epochs=epoch,
                        next_global_ordinal=wave_end) if group_complete else None
                    if (
                        group_complete or global_steps % self.options.checkpoint_steps == 0
                    ):
                        (
                            dynamic_high_water,
                            schedule_state,
                            media_training_state,
                        ) = self._checkpoint(
                            brain=brain,
                            wrapped=wrapped,
                            optimizer=optimizer,
                            manifest=manifest,
                            cursors=cursors,
                            local_dynamic=pending_dynamic,
                            dynamic_high_water=dynamic_high_water,
                            global_steps=global_steps,
                            strategy=strategy,
                            final_pack=False,
                            schedule_state=schedule_state,
                            rehearsal_phase=rehearsal_phase,
                            media_training_state=media_training_state,
                            media_replay=media_replay,
                        )
                        pending_dynamic.clear()
                        refresh_replica()
                    if self.context.is_rank_zero:
                        committed_ordinal = min(value.next_global_ordinal for value in cursors)
                        completed_units = epoch * len(manifest.entries) + committed_ordinal
                        total_units = self.options.epochs * max(1, len(manifest.entries))
                        self.store.write_status(
                            state="running",
                            globalOptimizerSteps=global_steps,
                            epoch=epoch,
                            nextGlobalOrdinal=committed_ordinal,
                            activeRecordWindows=[value.record_window for value in cursors],
                            learnerWave={"physicalBatchWindows": plan["physicalBatchRecords"],
                                "windowTargetPerRank": plan["waveWindowTarget"], "requestedRecordGroup": [wave_start, wave_end]},
                            progress=min(0.99, completed_units / float(total_units)),
                            metrics=last_metrics,
                        )
                    if group_complete:
                        if active_stream is not None:
                            active_stream.close()
                        active_stream, active_entry = None, None
                        self._leased_human_cue_cache = None
                        wave_start = wave_end

                # Publish the epoch transition itself so resume never repeats
                # a completed last wave under the next epoch number.
                tail_error = None
                try:
                    if next(owned_iterator, None) is not None:
                        raise ValueError("distributed rank iterator retained an uncommitted source record")
                except BaseException as failure:
                    tail_error = _failure_payload(failure)
                _raise_phase_failures(_collect_objects(self.context, tail_error), "distributed epoch source exhaustion failed")
                cursor = RankCursor(
                    rank=self.context.rank,
                    world_size=self.context.world_size,
                    epoch=epoch + 1,
                    next_global_ordinal=0,
                    owned_records_completed=cursor.owned_records_completed,
                    optimizer_steps_completed=cursor.optimizer_steps_completed,
                    manifest_sha256=manifest.content_sha256,
                )
                cursor_payloads = _collect_objects(self.context, cursor.to_dict())
                cursors = [RankCursor.from_dict(value) for value in cursor_payloads]
                final_boundary = cursor.epoch == self.options.epochs
                rehearsal_phase = phase_at_completed_wave(
                    completed_epochs=cursor.epoch,
                    next_global_ordinal=0,
                    final=final_boundary,
                )
                (
                    dynamic_high_water,
                    schedule_state,
                    media_training_state,
                ) = self._checkpoint(
                    brain=brain,
                    wrapped=wrapped,
                    optimizer=optimizer,
                    manifest=manifest,
                    cursors=cursors,
                    local_dynamic=pending_dynamic,
                    dynamic_high_water=dynamic_high_water,
                    global_steps=global_steps,
                    strategy=strategy,
                    final_pack=final_boundary,
                    schedule_state=schedule_state,
                    rehearsal_phase=rehearsal_phase,
                    media_training_state=media_training_state,
                    media_replay=media_replay,
                )
                pending_dynamic.clear()
                refresh_replica()
                owned_iterator.close()

            error: Optional[str] = None
            promotion: Optional[Dict[str, Any]] = None
            if self.context.is_rank_zero:
                try:
                    promotion = self._promote_output(
                        manifest=manifest,
                        cursors=cursors,
                        global_steps=global_steps,
                        dynamic_high_water=dynamic_high_water,
                        strategy=strategy,
                        schedule_state=schedule_state,
                        media_training_state=media_training_state,
                    )
                    self.store.write_status(
                        state="complete",
                        progress=1.0,
                        globalOptimizerSteps=global_steps,
                        outputPath=str(self.output_path),
                        promotion=promotion,
                        capabilityRehearsal=schedule_state.to_dict(),
                        mediaTraining=media_training_state,
                    )
                except BaseException as caught:
                    error = "%s: %s" % (type(caught).__name__, caught)
            _broadcast_rank_zero_error(self.context, error)
            values: List[Any] = [promotion if self.context.is_rank_zero else None]
            if self.context.distributed:
                dist.broadcast_object_list(values, src=0)
            return {
                "state": "complete",
                "brainId": self.brain_id,
                "manifestSha256": manifest.content_sha256,
                "validRecords": len(manifest.entries),
                "epochs": self.options.epochs,
                "worldSize": self.context.world_size,
                "strategy": strategy,
                "globalOptimizerSteps": global_steps,
                "dynamicUpdates": int(dynamic_high_water),
                "resumed": resumed,
                "metrics": last_metrics,
                "capabilityRehearsal": schedule_state.to_dict(),
                "mediaTraining": media_training_state,
                "promotion": values[0],
            }
        except DatasetResourcePause as error:
            state = "cancelled" if error.status.get("cancelled") else "paused"
            if write_authority and self.context.is_rank_zero:
                self.store.write_status(state=state, reason=str(error),
                    resumable=self.store.active_path.is_file(), resourceReadings=error.status)
            return {"state": state, "resumable": self.store.active_path.is_file(), "reason": str(error)}
        except BaseException as error:
            # Packed synapses may have mutated before backward/finite/step
            # failed. This rank brain is a private, unsaved clone: never save
            # it on this path. The next run recreates it from the last
            # published native generation and cursor, discarding the entire
            # failed wave rather than replaying a partially learned state.
            if write_authority:
                self.store.record_failure(self.context.rank, error)
            if write_authority and self.context.is_rank_zero:
                self.store.write_status(
                    state="failed",
                    reason="%s: %s" % (type(error).__name__, error),
                    resumable=self.store.active_path.is_file(),
                    failures=self.store.recent_failures(),
                )
            raise
        finally:
            # All owned leases are released even if another teardown fails.
            # No cleanup path publishes an unsaved failed replica.
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(self._restore_signal_handlers)
                for lease in reversed(leases):
                    if lease is not None: cleanup.callback(lease.close)
                if brain is not None: cleanup.callback(brain.close)
                controller = getattr(self, "_packed_collective_controller", None)
                if controller is not None:
                    def close_controller():
                        try:
                            if controller.step_id: controller.rollback()
                        finally:
                            controller.close()
                            self._packed_collective_controller = None
                    cleanup.callback(close_controller)
                for name in ("media_replay", "owned_iterator", "active_stream"):
                    resource = locals().get(name)
                    if resource is not None: cleanup.callback(resource.close)
                input_wave = locals().get("local_records")
                if isinstance(input_wave, PreparedWindowWave): cleanup.callback(input_wave.close)


__all__ = [
    "DistributedBrainTrainingModule",
    "DistributedGroundUpTrainer",
    "DistributedTrainingOptions",
    "DynamicNeuralUpdate",
    "apply_dynamic_updates",
]
