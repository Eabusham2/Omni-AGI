"""Adaptive OmniCortex brain lifecycle and persistence."""

import array
import base64
import binascii
import copy
import errno
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave
import zipfile
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import torch
from torch import nn
from torch.nn import functional as F

from .capability_rehearsal import (
    CapabilityRehearsalPolicy,
    CapabilityScheduleState,
    advance_schedule_state,
    due_rehearsal_phase,
    eligible_ground_up_rehearsal,
    rehearse_capabilities,
    structural_capability_schemas,
)
from .committed_paged_cache import (
    finish_verified_index_from_loaded_vectors,
    prepare_committed_paged_cache,
)
from .config import OmniConfig, safe_rounded_storage_bytes_per_second
from .conversation_ledger import NeuralConversationLedger
from .datasets import (
    DatasetCoverage,
    dataset_format,
    dataset_record_count_hint,
    iter_dataset_records,
    sqlite_consistent_snapshot_sha256,
)
from .ground_up import (
    GROUND_UP_ACTION_EXAMPLES,
    GROUND_UP_V3_SOURCE_RECORDS,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_TRAJECTORIES,
    current_ground_up_curriculum_manifest,
    current_ground_up_training_receipt_contract,
    resolve_ground_up_curriculum_manifest,
    seal_ground_up_v3_training_manifest,
    seal_ground_up_v3_training_receipt,
    validate_ground_up_v3_training_manifest,
    validate_ground_up_v3_training_receipt,
)
from .ingestion_schedule_v3 import (
    checkpoint_binding_sha256,
    make_checkpoint_binding_v3,
    make_ingestion_schedule_v3,
    schedule_sha256,
    validate_checkpoint_binding_v3,
    validate_ingestion_schedule_v3,
    validate_unobserved_checkpoint_binding_v3,
)
from .joint_generation import (
    VerifiedArtifactCache,
    recover_joint_generation,
    stage_joint_generation,
)
from .liquid import LiquidController
from .memory_lifecycle import OrganicMemoryLifecycle
from .modalities import (
    IMAGINATION_MODALITIES,
    ModalityGenerationCancelled,
    ModalityHub,
)
from .media_planning import (
    MediaGenerationMeasurements,
    MediaOutputRequest,
    MediaResourceDemand,
    NeuralMediaWindows,
    inline_media_data_url,
    media_resource_headroom,
    plan_media_output,
)
from .model import (
    ACTION_KINDS,
    TERNARY_PROJECTION_TYPES,
    PackedAdaptiveBitLinear as BitLinear,
    OmniDecoder,
    packed_runtime_status,
)
from .offload import (
    DurableReplayBuffer,
    HotStateResidencyPlanner,
    MutableStateStore,
    NeuralStateResourcePause,
    PagedWorkingMemory,
    ResourcePolicy,
    copy_mutable_state_snapshot,
)
from .optimizers import (
    PackedMutationSnapshot,
    PackedOnlyOptimizer,
    adamw_for_remaining_parameters,
)
from .persistence import (
    EventLog,
    atomic_save_tensors,
    atomic_write_json,
    copy_substrate_snapshot,
    load_tensors,
    read_json,
    snapshot_required_bytes,
    snapshot_files,
    tensor_checksum,
)
from .spiking import AssociativeSpikingRouter
from .ternary_packing import (
    collect_module_ternary_tensors,
    export_module_ternary_shards,
    inspect_module_ternary_layout,
    verify_ternary_shards,
)
from .tokenizer import ByteTokenizer
from .vsa import LazyPersistedSynapses, NeuralSubstrate, SubstrateResourcePause


ENGINE_SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
ACTION_PROPOSAL_CONFIDENCE = 0.62
# A strict majority in one independent eight-way head is meaningful evidence,
# unlike a random/uniform vote.  It can support a talk-to-tool recovery only
# when the structural materializer separately proves a complete typed action.
ACTION_INDEPENDENT_TOOL_SUPPORT_CONFIDENCE = 0.5
NATIVE_ACTION_TARGET_CONFIDENCE = 0.70
ACTION_KIND_EMISSION_CONFIDENCE = {
    "talk": 0.0,
    "tool": ACTION_PROPOSAL_CONFIDENCE,
    "imagine": ACTION_PROPOSAL_CONFIDENCE,
    "agent": ACTION_PROPOSAL_CONFIDENCE,
    "ponder": 0.30,
    "learn": 0.30,
    "evolve": ACTION_PROPOSAL_CONFIDENCE,
    "stop": 0.90,
}
INGESTION_CHECKPOINT_FORMAT = "omni-record-ingestion-checkpoint"
INGESTION_CHECKPOINT_VERSION = 2
INGESTION_PARSER_CONTRACT = "omni-dataset-record-stream-v1"
INGESTION_LEARNING_SCHEDULE_FORMAT = "omni-ingestion-learning-schedule"
INGESTION_LEARNING_SCHEDULE_VERSION = 2
LOCAL_TYPED_TARGET_WINDOW_POLICY = (
    "role-bounded-causal-exact-byte-windows-v1"
)
INGESTION_CHECKPOINT_RECORDS = 512
COMPLETED_INGESTION_TOMBSTONES = 256
COMPLETED_CHAT_TURN_RECEIPTS = 256
CHAT_TURN_RECEIPT_FORMAT = "omni-completed-chat-turn"
COMPLETED_CHAT_SLOW_LEARNING = 256
FRESH_ATTENTION_FORMAT = "omni-fresh-attention-boundary"
FRESH_ATTENTION_VERSION = 1
MEDIA_ACCUMULATOR_FORMAT = "omni-media-diagnostics"
MEDIA_ACCUMULATOR_VERSION = 1
MEDIA_DIAGNOSTIC_SAMPLES = 64
MEDIA_COUNTER_MAX = (1 << 63) - 1
# Inspection pages have no record-count ceiling.  This byte envelope keeps a
# single JSON-RPC response below the desktop worker's protocol-line guard;
# continuation cursors make every matching record addressable.
SUBSTRATE_INSPECTION_TRANSPORT_BYTES = 16 * 1024 * 1024
ALLOCATOR_OOM_MARKERS = (
    "cuda out of memory",
    "hip out of memory",
    "mps backend out of memory",
    "cannot allocate memory",
    "can't allocate memory",
    "std::bad_alloc",
    "cudnn_status_alloc_failed",
    "cuda_error_out_of_memory",
    "not enough memory resources are available",
    "e_outofmemory",
    "0x8007000e",
)
STREAMING_CANONICAL_CHUNK_ELEMENTS = 1 << 20
MODALITY_DECODER_STEPS = {
    "image": 4,
    "audio": 3,
    "video": 3,
}
# The tier changes preview publication cadence only. Decoder steps, spatial
# resolution, codec length, frame count, and final artifact quality remain the
# exact configured values for this brain.
MODALITY_PREVIEW_BUDGET_BY_TIER = {
    "micro": 2,
    "personal": 3,
    "gpu": 4,
    "workstation": 6,
}


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _prefixed_state(module: nn.Module, prefix: str) -> Dict[str, torch.Tensor]:
    return {prefix + key: value for key, value in module.state_dict().items()}


def _load_prefixed(
    module: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    prefix: str,
    strict: bool = True,
) -> None:
    state = {
        key[len(prefix) :]: value
        for key, value in tensors.items()
        if key.startswith(prefix)
    }
    result = module.load_state_dict(state, strict=False)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise ValueError(
            "%s checkpoint mismatch (missing=%s, unexpected=%s)"
            % (prefix, result.missing_keys, result.unexpected_keys)
        )


def _finite_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clone_state_to_cpu(value: Any) -> Any:
    """Clone nested optimizer state without retaining accelerator storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def is_allocator_oom_error(error: BaseException) -> bool:
    """Recognize allocator exhaustion without treating unrelated failures as OOM.

    PyTorch uses different exception classes and messages across CPU, CUDA,
    MPS, and DirectML.  Following the chained exception is important because
    some backends wrap the allocator error in a plain ``RuntimeError``.
    """

    current: Optional[BaseException] = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        inspected = current
        visited.add(id(current))
        if isinstance(inspected, MemoryError):
            return True
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(cuda_oom, type) and isinstance(inspected, cuda_oom):
            return True
        if isinstance(inspected, OSError) and inspected.errno == errno.ENOMEM:
            return True
        message = str(inspected).strip().lower()
        if isinstance(inspected, (RuntimeError, OSError)) and any(
            marker in message for marker in ALLOCATOR_OOM_MARKERS
        ):
            return True
        current = inspected.__cause__
        if current is None and not inspected.__suppress_context__:
            current = inspected.__context__
    return False


class ChatGenerationCancelled(RuntimeError):
    """Raised only at a rollback-safe cooperative chat boundary."""


class AdaptiveBrain:
    """One persistent, mutable model identity.

    ``storage_path`` is the desktop brain directory.  All Python-owned files
    live below ``storage_path/engine`` so the UI's ``brain.json`` remains
    authoritative and untouched.
    """

    def __init__(
        self,
        brain_id: str,
        storage_path: Path,
        config: OmniConfig,
    ):
        # Native OmniCortex keeps eligible learned synapses in packed ternary
        # storage while higher-precision activity and control state remain
        # separate. Structural growth is governed by live resources.
        config.ternary_weights = True
        config.spiking_dynamics = True
        config.stdp_plasticity = True
        config.liquid_dynamics = True
        config.vector_symbolic_memory = True
        config.memory_injection = "working-memory"
        config.validate()
        self.brain_id = str(brain_id)
        self.storage_path = Path(storage_path).resolve()
        self.engine_path = self.storage_path / "engine"
        # A live paging cache is expendable, not checkpoint authority. A new
        # process gets a new child so a crash before the first brain.json
        # commit cannot make the next retry adopt or overwrite stale pages.
        self._live_paging_cache_directory = (
            self.engine_path / "state" / "live-substrate-cache" / uuid.uuid4().hex
        )
        self.config = config
        self.resource_policy = ResourcePolicy(
            self.engine_path,
            ram_reserve_bytes=config.ram_reserve_bytes,
            disk_reserve_bytes=config.disk_reserve_bytes,
            system_ram_share_percent=config.system_ram_share_percent,
            storage_bytes_per_second=config.storage_bytes_per_second,
            hardware_tier=config.hardware_tier,
        )
        self.state_store = MutableStateStore(
            self.engine_path / "state", self.brain_id, self.resource_policy
        )
        self.mutable_state_manifest: Optional[Dict[str, Any]] = None
        self._substrate_gc_status: Dict[str, Any] = {
            "completed": False,
            "reason": "no committed checkpoint yet",
            "generationsRetained": 0,
            "generationsRemoved": 0,
            "blobsRemoved": 0,
            "bytesReclaimed": 0,
        }
        self.resource_pause: Optional[Dict[str, Any]] = None
        # These are runtime performance bounds, never neural-state limits.
        # Allocator pressure may lower them without rewriting the user's
        # hardware-derived configuration or skipping any source tokens.
        self._runtime_train_batch_size = max(1, int(config.train_batch_size))
        self._runtime_training_max_seq_len = max(8, int(config.max_seq_len))
        self._allocator_oom_count = 0
        self._optimizer_offloaded = False
        self._optimizer_scratch_pointer: Optional[Dict[str, Any]] = None
        self._last_pressure_scratch_at = 0.0
        self.hot_state_residency = HotStateResidencyPlanner()
        requested_device = config.device
        self.device_backend = "cpu"
        if requested_device.lower() in {"directml", "dml", "privateuseone"}:
            try:
                import torch_directml

                self.device = torch_directml.device()
                self.device_backend = "directml"
            except (ImportError, RuntimeError, OSError, ValueError):
                self.device = torch.device("cpu")
        elif requested_device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested_device)
            self.device_backend = "cuda"
        elif (
            requested_device.lower() == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
            self.device_backend = "mps"
        else:
            self.device = torch.device("cpu")
        torch.manual_seed(config.seed)

        self.tokenizer = ByteTokenizer()
        self.decoder = OmniDecoder(config).to(self.device)
        self.memory_bridge = BitLinear(config.vsa_dim, config.idea_dim, bias=True).to(
            self.device
        )
        self.idea_adapter = nn.Sequential(
            BitLinear(config.idea_dim, config.idea_dim * 2, bias=True),
            nn.SiLU(),
            BitLinear(config.idea_dim * 2, config.idea_dim, bias=True),
        ).to(self.device)
        self.router = AssociativeSpikingRouter(
            config.idea_dim,
            config.router_neurons,
            leak=config.membrane_leak,
            threshold=config.firing_threshold,
            learning_rate=config.stdp_learning_rate,
            tau_pre=config.stdp_tau_pre,
            tau_post=config.stdp_tau_post,
            a_plus=config.stdp_a_plus,
            a_minus=config.stdp_a_minus,
            metaplasticity_rate=(
                config.metaplasticity_rate if config.metaplasticity else 0.0
            ),
        ).to(self.device)
        self.liquid = LiquidController(
            config.idea_dim,
            mode=config.liquid_mode,
            solver_steps=config.liquid_steps,
        ).to(self.device)
        self.modalities = ModalityHub(config).to(self.device)
        for root_module in (
            self.decoder,
            self.memory_bridge,
            self.idea_adapter,
            self.router,
            self.liquid,
            self.modalities,
        ):
            for module in root_module.modules():
                if isinstance(module, TERNARY_PROJECTION_TYPES):
                    module.ternary = True
                    module.configure_packed_stability(
                        enabled=config.metaplasticity,
                        strength=config.slow_stability_strength,
                    )
        self.memory = NeuralSubstrate(
            config.vsa_dim,
            seed=config.seed,
            growth_guard=self._allow_substrate_growth,
        )
        self._paged_vector_cache: Optional[Dict[str, Any]] = None
        self.liquid_state = torch.zeros(1, config.idea_dim, device=self.device)
        self.working_memory: List[torch.Tensor] = []
        self.workspace_items: List[Dict[str, Any]] = []
        self.memory_lifecycle = OrganicMemoryLifecycle()
        # Temporary multi-turn language-boundary state. Long-term facts remain
        # authoritative only in the neural substrate and learned parameters;
        # this bounded ring is the explicit token working memory shown in the
        # Runtime Card.
        self.recent_token_context: List[int] = []
        self.current_context: Dict[str, Any] = {
            "tokenCount": 0,
            "tokenHash": "",
            "sensorySlots": 0,
            "updatedAt": self.created_at if hasattr(self, "created_at") else _iso_now(),
        }
        self.fresh_attention_boundary: Optional[Dict[str, Any]] = None
        self._fresh_attention_paged_clear_pending = False
        # Replay is durable immediately, not a capacity-limited Python list.
        # This keeps all admitted latent examples available across RAM pressure
        # and process restarts without silently thinning older experience.
        self.replay = DurableReplayBuffer(
            self.engine_path / "state" / "replay.sqlite3",
            self.resource_policy,
        )
        self.paged_working_memory = PagedWorkingMemory(
            self.engine_path / "state" / "working-memory.sqlite3",
            self.resource_policy,
        )
        self.paged_working_memory_recovery: Dict[str, Any] = {
            "committedPages": 0,
            "rolledBackPages": 0,
            "scratchReset": False,
            "learningReadable": False,
        }
        self.messages: List[Dict[str, Any]] = []
        self.traces: List[Dict[str, Any]] = []
        self.training_sources: List[Dict[str, Any]] = []
        # A file-level desktop cursor is not sufficient for multi-gigabyte
        # JSONL/Parquet shards.  These engine-authoritative records are
        # committed in the same brain.json generation as the neural tensors
        # they describe, so a retry can replay only the uncommitted suffix.
        # The checkpoint contains counts, hashes, and neural audit summaries;
        # it never contains source text or token ids.
        self.ingestion_checkpoints: Dict[str, Dict[str, Any]] = {}
        # A staged joint generation is authoritative only when this exact
        # reference is atomically included in brain.json with the v3 cursor.
        # The SQLite assembly/vector cache is derived and never referenced as
        # an independent recovery authority.
        self.ingestion_joint_generation: Optional[Dict[str, Any]] = None
        self._joint_artifact_cache = VerifiedArtifactCache()
        self._paged_substrate_required = False
        self._verified_paged_rebuild: Optional[Any] = None
        self.completed_ingestions: List[Dict[str, Any]] = []
        self.completed_chat_turns: List[Dict[str, Any]] = []
        # Every accepted turn enters fast episodic neural state. Slow replay
        # jobs are checkpointed beside that state so cortical consolidation
        # can be preempted/retried without loss or duplicate optimizer steps.
        self.pending_chat_slow_learning: List[Dict[str, Any]] = []
        self.completed_chat_slow_learning: List[str] = []
        self._ingestion_checkpoint_records = INGESTION_CHECKPOINT_RECORDS
        self.created_at = _iso_now()
        self.updated_at = self.created_at
        self.counters: Dict[str, int] = {
            "experiences": 0,
            "training_steps": 0,
            "consolidation_cycles": 0,
            "inference_count": 0,
            "plasticity_events": 0,
            "snapshots": 0,
            "metaplastic_updates": 0,
            "workspace_evictions": 0,
            "workspace_rehearsals": 0,
            "context_token_evictions": 0,
            "idle_cognition_cycles": 0,
            "action_retention_checks": 0,
            "action_retention_replays": 0,
            "action_retention_failures": 0,
        }
        self.modality_training: Dict[str, int] = {
            "vision": 0,
            "image": 0,
            "audio": 0,
            "video": 0,
        }
        self.installed_modality_packs: List[Dict[str, Any]] = []
        # Every build uses the native, randomly initialized core.
        self.ground_up_training_manifest: Optional[Dict[str, Any]] = None
        self.packed_ternary_manifest: Optional[Dict[str, Any]] = None
        self._starter_action_language_cache: Optional[torch.Tensor] = None
        self._starter_action_internal_cache: Optional[torch.Tensor] = None
        self._starter_action_target_cache: Optional[torch.Tensor] = None
        self._starter_records_visited = 0
        self._ground_up_action_origin_verified = False
        self.novelty_streak = 0
        self.growth_pause: Optional[Dict[str, Any]] = None
        self.last_activity_decay = time.time()
        self.last_idle_cycle_at = 0.0
        # Prompt-free recurrent activity may run frequently, but surfacing an
        # unsolicited message or external action every cycle is not organic
        # behavior.  This persisted refractory timestamp leaves the internal
        # neural dynamics active while spacing user-visible initiative.  It is
        # a desktop attention safeguard, not a personality/curiosity control.
        self.last_idle_visible_action_at = 0.0
        self.slow_anchors: Dict[str, torch.Tensor] = {}
        self.slow_importance: Dict[str, torch.Tensor] = {}
        self._sync_stability_state()
        self._optimizer = self._new_optimizer()
        self.events = EventLog(self.engine_path / "events.sqlite3", self.brain_id)
        self.conversation = NeuralConversationLedger(
            self.engine_path / "conversation.sqlite3",
            self.brain_id,
        )

    def close(self) -> None:
        self.conversation.close()
        self.events.close()

    def _trainable_modules(self) -> Iterable[nn.Module]:
        modules: List[nn.Module] = [
            self.decoder,
            self.memory_bridge,
            self.idea_adapter,
            self.router,
            self.liquid,
            self.modalities,
        ]
        return tuple(modules)

    @staticmethod
    def _learned_parameter_tensors(
        modules: Iterable[nn.Module],
    ) -> Iterator[torch.Tensor]:
        """Yield each learned tensor once, including packed ternary synapses.

        A packed-authoritative projection has no floating ``Parameter`` for
        its weights. Counting or hashing only ``module.parameters()`` would
        falsely report no cortical learning even while its synapses change.
        Scales, online rates, activity, and eligibility are not extra learned
        synapses and are deliberately excluded here.
        """

        seen: set[int] = set()
        for root in modules:
            for parameter in root.parameters():
                identity = id(parameter)
                if identity not in seen:
                    seen.add(identity)
                    yield parameter
            for child in root.modules():
                packed_tensors = getattr(
                    child, "authoritative_packed_tensors", None
                )
                if not callable(packed_tensors):
                    continue
                for tensor in packed_tensors():
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError("packed synapse owner returned a non-tensor")
                    identity = id(tensor)
                    if identity not in seen:
                        seen.add(identity)
                        yield tensor

    @staticmethod
    def _packed_logical_parameter_count(modules: Iterable[nn.Module]) -> int:
        seen: set[int] = set()
        total = 0
        for root in modules:
            for child in root.modules():
                identity = id(child)
                if identity in seen:
                    continue
                seen.add(identity)
                count = getattr(child, "logical_ternary_parameter_count", 0)
                total += int(count() if callable(count) else count)
        return total

    def _new_optimizer(
        self, learning_rate: Optional[float] = None
    ) -> torch.optim.Optimizer | PackedOnlyOptimizer:
        base_learning_rate = max(
            1e-6,
            min(
                0.02,
                float(
                    self.config.learning_rate
                    if learning_rate is None
                    else learning_rate
                ),
            ),
        )
        parameters = []
        for module in self._trainable_modules():
            parameters.extend(module.parameters())
        groups: List[Dict[str, Any]] = [
            {"params": parameters, "lr": base_learning_rate}
        ]
        return adamw_for_remaining_parameters(
            groups,
            lr=base_learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _replace_optimizer(
        self, learning_rate: Optional[float] = None
    ) -> torch.optim.Optimizer | PackedOnlyOptimizer:
        """Install a fresh optimizer and retire any stale offload pointer."""

        self._optimizer = self._new_optimizer(learning_rate)
        self._optimizer_offloaded = False
        self._optimizer_scratch_pointer = None
        return self._optimizer

    def _load_optimizer_state(self, state: Mapping[str, Any]) -> bool:
        """Load moments for the single native OmniCortex optimizer."""

        self._optimizer.load_state_dict(copy.deepcopy(dict(state)))
        return False

    def _ensure_optimizer_resident(self) -> None:
        """Rehydrate Adam moments from safe-tensor scratch on demand."""

        if not self._optimizer_offloaded:
            return
        if self._optimizer_scratch_pointer is None:
            raise RuntimeError("optimizer is offloaded without a durable pointer")
        state = self.state_store.load_pressure_optimizer(
            self._optimizer_scratch_pointer
        )
        if not isinstance(state, Mapping):
            raise ValueError("pressure optimizer state is invalid")
        self._optimizer.load_state_dict(dict(state))
        self._optimizer_offloaded = False

    def _activation_scratch_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {"state.liquid": self.liquid_state.detach()}
        if self.working_memory:
            tensors["state.working_memory"] = torch.stack(
                self.working_memory
            )
        for name, value in self.router.state_dict().items():
            tensors["router." + name] = value.detach()
        return tensors

    def _memory_pressure_wait(
        self,
        readings: Mapping[str, Any],
        *,
        stage: str,
        retry_after_seconds: int = 5,
        detail: str = "",
    ) -> Dict[str, Any]:
        """Publish a recoverable wait without changing persisted capacity."""

        status = {
            **dict(readings),
            "mode": "memory-pressure-wait-retry",
            "paused": True,
            "recoverable": True,
            "waitForMemory": True,
            "retryAfterSeconds": max(1, int(retry_after_seconds)),
            "pressureStage": str(stage),
            "configuredContextTokens": int(self.config.max_seq_len),
            "configuredCapacityPreserved": True,
            "capacityPersistsAcrossPressure": True,
            "contextWindowShrunk": False,
            "contextPagedToStorage": False,
            "activeCortexResident": True,
            "userAction": (
                "Close memory-heavy applications, wait for memory to become "
                "available, then retry. Omni keeps the saved context capacity."
            ),
            "detail": str(detail),
        }
        reason = (
            "memory pressure paused %s; close memory-heavy applications, "
            "then retry without reducing the saved context capacity" % stage
        )
        self.resource_pause = {
            "reason": reason,
            "readings": status,
            "at": _iso_now(),
        }
        return status

    def _maintain_neural_state_resources(self) -> Dict[str, Any]:
        """Spill transient neural state when the adaptive RAM reserve is near.

        Replay is always disk-backed. Under pressure this additionally writes
        optimizer moments and activation/recurrent scratch, then releases the
        resident optimizer state and accelerator allocator caches. The next
        optimizer use transparently rehydrates the exact moments.
        """

        status = self.resource_policy.status()
        if status["diskPressure"]:
            self.resource_pause = {
                "reason": "available disk reached the neural-state reserve",
                "readings": status,
                "at": _iso_now(),
            }
            return self._state_offload_status(status)
        if status["memoryPressure"] and self.config.disk_state_offload:
            from .paged_assembly_view import PagedAssemblyView

            if not isinstance(self.memory.assemblies, PagedAssemblyView):
                self._ensure_paged_ingestion_substrate(0, force=True)
                status = self.resource_policy.status()
                if not status["memoryPressure"]:
                    self.resource_pause = None
                    return self._state_offload_status(status)
        if (
            status["memoryPressure"]
            and self.config.disk_state_offload
        ):
            training_plan = self._training_resource_plan()
            scratch_plan = dict(training_plan["scratch"])
            if not bool(scratch_plan["available"]):
                wait = self._memory_pressure_wait(
                    {
                        **status,
                        "trainingResourcePlan": training_plan,
                    },
                    stage="neural-state offload",
                    detail=(
                        "No safe emergency-checkpoint space remains above "
                        "the mandatory disk reserve."
                    ),
                )
                return self._state_offload_status(wait)
            now = time.monotonic()
            minimum_interval = float(
                scratch_plan["minimumWriteIntervalSeconds"]
            )
            if (
                not self._optimizer_offloaded
                and self._optimizer_scratch_pointer is not None
                and now - self._last_pressure_scratch_at < minimum_interval
            ):
                # Keep the newer state in RAM and pause before another step.
                # Rewriting full optimizer moments every microbatch would turn
                # an HDD into a seek bottleneck and needlessly wear an SSD.
                wait = self._memory_pressure_wait(
                    {
                        **status,
                        "trainingResourcePlan": training_plan,
                        "scratchRetryAfterSeconds": max(
                            0,
                            int(
                                minimum_interval
                                - (now - self._last_pressure_scratch_at)
                            ),
                        ),
                    },
                    stage="neural-state checkpoint",
                    retry_after_seconds=max(
                        1,
                        int(
                            minimum_interval
                            - (now - self._last_pressure_scratch_at)
                        ),
                    ),
                    detail=(
                        "The newer optimizer state remains in RAM until the "
                        "rate-limited sequential scratch write is eligible."
                    ),
                )
                return self._state_offload_status(wait)
            if self._optimizer_offloaded:
                if self._optimizer_scratch_pointer is None:
                    raise RuntimeError(
                        "optimizer is offloaded without durable scratch"
                    )
                # The cold optimizer moments are already durable and absent
                # from RAM. Rewriting the same full blob under sustained
                # pressure would only wear storage and cannot free more RAM.
                wait = self._memory_pressure_wait(
                    status,
                    stage="neural execution",
                    detail=(
                        "Eligible optimizer moments are already paged; active "
                        "cortex and attention state remain resident."
                    ),
                )
                return self._state_offload_status(wait)
            else:
                optimizer_state = _clone_state_to_cpu(
                    self._optimizer.state_dict()
                )
            pointer = self.state_store.save_pressure_scratch(
                optimizer_state=optimizer_state,
                activations=self._activation_scratch_tensors(),
                metadata={
                    "brainId": self.brain_id,
                    "parameterChecksum": self.parameter_checksum(),
                    "substrateContentSha256": str(
                        (self.memory.persistence_manifest or {}).get(
                            "contentSha256", ""
                        )
                    ),
                    "trainingSteps": self.counters["training_steps"],
                    "reason": "adaptive RAM reserve",
                    "ioMode": "bounded-sequential-emergency-checkpoint",
                    "sequentialChunkBytes": scratch_plan[
                        "sequentialChunkBytes"
                    ],
                    "minimumWriteIntervalSeconds": minimum_interval,
                },
            )
            if not self._optimizer_offloaded:
                self._optimizer.state.clear()
            self._optimizer_scratch_pointer = dict(pointer)
            self._optimizer_offloaded = True
            self._last_pressure_scratch_at = now
            if self.device_backend == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Do not invoke the MPS allocator cache-clear API: it clears MPSGraphCache and
            # can deallocate a graph still executing on another runtime queue.
            # Unified-memory pressure remains governed by the live policy and
            # pauses truthfully when offloading alone does not restore reserve.
            post_status = self.resource_policy.status()
            if bool(post_status.get("memoryPressure")):
                post_status = self._memory_pressure_wait(
                    post_status,
                    stage="neural execution",
                    detail=(
                        "Cold optimizer moments were paged successfully, but "
                        "the current live RAM watermark is still unavailable."
                    ),
                )
            else:
                self.resource_pause = None
            return self._state_offload_status(post_status)
        if status["memoryPressure"]:
            wait = self._memory_pressure_wait(
                status,
                stage="neural execution",
                detail=(
                    "Disk state offload is unavailable; active cortex state "
                    "was kept intact."
                ),
            )
            return self._state_offload_status(wait)
        self.resource_pause = None
        return self._state_offload_status(status)

    def _state_offload_status(
        self, resource_status: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        readings = dict(resource_status or self.resource_policy.status())
        replay_status = self.replay.status()
        system_budget = readings.get("systemRamBudgetBytes")
        process_memory = readings.get("processMemoryBytes")
        resident_budget: Optional[int] = None
        if isinstance(system_budget, int) and isinstance(process_memory, int):
            # Metadata-rich sparse records vary in size; 1 KiB is a
            # conservative ordering estimate, not a cardinality ceiling.
            resident_budget = max(
                0, (system_budget - process_memory) // 1024
            )
        unfinished_ids = [
            str(item.get("assemblyId", ""))
            for item in self.workspace_items
            if isinstance(item, Mapping) and item.get("assemblyId")
        ]
        residency = self.hot_state_residency.update(
            neurons=self.memory.neurons,
            assemblies=self.memory.assemblies,
            synapses=self.memory.synapses,
            unfinished_ids=unfinished_ids,
            attention_active_ids=(
                self.memory.attention_active_neuron_ids
                | self.memory.attention_eligible_synapse_ids
            ),
            attention_legacy_raw_active=(
                self.memory.attention_legacy_raw_active
            ),
            resident_budget=resident_budget,
            paged_assembly_ids=(
                ()
                if self._fresh_attention_paged_clear_pending
                else self.paged_working_memory.assembly_ids()
            ),
        )
        paged_status = self.paged_working_memory.status()
        if self._fresh_attention_paged_clear_pending:
            paged_status = {**paged_status, "count": 0}
        orphan_count = 0
        orphan_database_bytes = 0
        cache_parent = self._live_paging_cache_directory.parent
        if cache_parent.is_dir() and not cache_parent.is_symlink():
            for child in cache_parent.iterdir():
                if (
                    child == self._live_paging_cache_directory
                    or not re.fullmatch(r"[0-9a-f]{32}", child.name)
                    or not child.is_dir()
                    or child.is_symlink()
                ):
                    continue
                orphan_count += 1
                database = child / "live-paged" / "working.sqlite3"
                try:
                    if database.is_file() and not database.is_symlink():
                        orphan_database_bytes += database.stat().st_size
                except OSError:
                    # Another process may still be changing a derived cache.
                    # Telemetry must not interrupt neural checkpointing.
                    pass
        return {
            "mode": "transactional-disk-backed",
            "enabled": self.config.disk_state_offload,
            "replay": replay_status,
            "optimizer": {
                "durable": self.mutable_state_manifest is not None
                or self._optimizer_scratch_pointer is not None,
                "resident": not self._optimizer_offloaded,
                "pressureScratch": self._optimizer_scratch_pointer,
                "serialization": "safe-tensors+typed-json",
            },
            "activationScratch": {
                "durableOnPressure": True,
                "mirroredNotEvicted": True,
                "includes": [
                    "liquid recurrent state",
                    "working-memory vectors",
                    "router spike/STDP state",
                ],
            },
            "workingMemoryPaging": paged_status,
            "derivedLiveCacheOrphans": {
                "count": orphan_count,
                "knownDatabaseBytes": orphan_database_bytes,
                "authoritative": False,
                "autoDeletedWithoutLease": False,
            },
            "hotStateResidency": residency,
            "pagingSemantics": {
                "contextPagedToStorage": False,
                "capacityPersistsAcrossPressure": True,
                "storagePoolShareRule": "largest-brain-not-sum",
                "activeCortexResident": True,
                "hotRamPriority": [
                    "currently firing",
                    "frequently used",
                    "stable or rooted",
                    "unfinished activity",
                ],
                "spillOrder": [
                    "cold scratch trail",
                    "replay batches",
                    "optimizer moments",
                    "inactive working patterns",
                ],
            },
            "configuredWorkingMemorySpillBytes": self.config.memory_offload_bytes,
            "estimatedStorageSlowdownPercent": (
                self.config.memory_offload_slowdown_percent
            ),
            "checkpoint": {
                "activeGeneration": (
                    self.mutable_state_manifest or {}
                ).get("activeGeneration"),
                "contentAddressed": True,
                "atomicPointer": True,
                "lastRecovery": dict(self.state_store.last_recovery),
                "garbageCollection": {
                    "mutableState": dict(self.state_store.last_gc),
                    "substrate": dict(self._substrate_gc_status),
                },
            },
            "resources": readings,
            "paused": self.resource_pause is not None
            or bool(readings.get("diskPressure")),
            "pause": self.resource_pause,
        }

    def _named_slow_parameters(self) -> Dict[str, nn.Parameter]:
        named: Dict[str, nn.Parameter] = {}
        modules: List[Tuple[str, nn.Module]] = [
            ("decoder", self.decoder),
            ("memory_bridge", self.memory_bridge),
            ("idea_adapter", self.idea_adapter),
            ("router", self.router),
            ("liquid", self.liquid),
            ("modalities", self.modalities),
        ]
        for prefix, module in modules:
            for name, parameter in module.named_parameters():
                named["%s.%s" % (prefix, name)] = parameter
        return named

    def _configure_packed_stability(self) -> None:
        """Apply the saved stability policy to newly grown packed modules."""

        for root in self._trainable_modules():
            for module in root.modules():
                if isinstance(module, TERNARY_PROJECTION_TYPES):
                    module.configure_packed_stability(
                        enabled=self.config.metaplasticity,
                        strength=self.config.slow_stability_strength,
                    )

    def _named_native_core_tensors(self) -> Dict[str, torch.Tensor]:
        """Name actual mutable weights for origin training receipts.

        The old receipt enumerated only ``nn.Parameter`` objects, so a packed
        layer could change its real ternary synapses while the receipt claimed
        that no native weight changed. Keep metaplastic float anchors separate.
        """

        roots: List[Tuple[str, nn.Module]] = [
            ("decoder", self.decoder),
            ("memory_bridge", self.memory_bridge),
            ("idea_adapter", self.idea_adapter),
            ("router", self.router),
            ("liquid", self.liquid),
            ("modalities", self.modalities),
        ]
        named: Dict[str, torch.Tensor] = {}
        for prefix, root in roots:
            for name, parameter in root.named_parameters():
                named["%s.%s" % (prefix, name)] = parameter
            for path, child in root.named_modules():
                packed_tensors = getattr(
                    child, "authoritative_packed_tensors", None
                )
                if not callable(packed_tensors):
                    continue
                for index, tensor in enumerate(packed_tensors()):
                    name = "%s.%s%s%s" % (
                        prefix,
                        path + "." if path else "",
                        "packed_synapses",
                        "" if index == 0 else ".%d" % index,
                    )
                    named[name] = tensor
        return named

    def _sync_stability_state(self) -> None:
        """Keep legacy floating anchors aligned if any remain.

        A native all-packed brain has no such parameters. Its live retention
        mechanism is per-row uint8 resistance applied in packed updates, not
        a fictitious differentiable FP32 anchor penalty.
        """

        named = self._named_slow_parameters()
        for name, parameter in named.items():
            anchor = self.slow_anchors.get(name)
            if anchor is None or tuple(anchor.shape) != tuple(parameter.shape):
                self.slow_anchors[name] = parameter.detach().cpu().clone()
                self.slow_importance[name] = torch.zeros_like(
                    parameter.detach().cpu(), dtype=torch.float32
                )
        stale = set(self.slow_anchors).difference(named)
        for name in stale:
            self.slow_anchors.pop(name, None)
            self.slow_importance.pop(name, None)

    def _stability_penalty(
        self, parameters: Optional[Iterable[nn.Parameter]] = None
    ) -> torch.Tensor:
        # Packed synapses receive their resistance in the direct stochastic
        # transition, where a differentiable weight penalty cannot act.
        named = self._named_slow_parameters()
        allowed = None if parameters is None else {id(item) for item in parameters}
        terms: List[torch.Tensor] = []
        if not self.config.metaplasticity or self.config.slow_stability_strength <= 0:
            return torch.zeros((), device=self.device)
        self._sync_stability_state()
        for name, parameter in named.items():
            if allowed is not None and id(parameter) not in allowed:
                continue
            anchor = self.slow_anchors[name].to(
                parameter.device, dtype=parameter.dtype
            )
            importance = self.slow_importance[name].to(
                parameter.device, dtype=parameter.dtype
            )
            terms.append((importance * (parameter - anchor).pow(2)).mean())
        if not terms:
            return torch.zeros((), device=self.device)
        return torch.stack(terms).mean() * self.config.slow_stability_strength

    def _accumulate_slow_importance(
        self, parameters: Optional[Iterable[nn.Parameter]] = None
    ) -> None:
        if not self.config.metaplasticity:
            return
        # Packed projections do not have an FP32 weight Parameter, so the
        # differentiable EWC path below is intentionally empty for an all-
        # ternary brain. Their row-wise uint8 resistance is updated directly
        # during a successful packed synapse transition; count only those
        # actual updates, never a no-op floating-anchor call.
        self._drain_packed_stability_events()
        allowed = None if parameters is None else {id(item) for item in parameters}
        decay = self.config.slow_importance_decay
        self._sync_stability_state()
        updated = 0
        for name, parameter in self._named_slow_parameters().items():
            if allowed is not None and id(parameter) not in allowed:
                continue
            if parameter.grad is None:
                continue
            evidence = parameter.grad.detach().float().cpu().pow(2)
            scale = float(evidence.mean().item())
            if scale > 0:
                evidence = (evidence / (scale + 1e-12)).clamp_(0.0, 100.0)
            self.slow_importance[name].mul_(decay).add_(
                evidence, alpha=1.0 - decay
            )
            updated += 1
        if updated:
            self.counters["metaplastic_updates"] += 1

    def _drain_packed_stability_events(self) -> int:
        events = 0
        for root in self._trainable_modules():
            for module in root.modules():
                drain = getattr(module, "drain_packed_stability_events", None)
                if callable(drain):
                    events += int(drain())
        if events:
            self.counters["metaplastic_updates"] += events
        return events

    def _commit_slow_anchors(
        self,
        rate: float = 1.0,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ) -> None:
        if not self.config.metaplasticity:
            return
        rate = max(0.0, min(float(rate), 1.0))
        allowed = None if parameters is None else {id(item) for item in parameters}
        self._sync_stability_state()
        for name, parameter in self._named_slow_parameters().items():
            if allowed is not None and id(parameter) not in allowed:
                continue
            current = parameter.detach().float().cpu()
            self.slow_anchors[name].lerp_(current, rate)

    def _packed_stability_accounting(self) -> Dict[str, Any]:
        modules = 0
        bytes_resident = 0
        for root in self._trainable_modules():
            for module in root.modules():
                status = getattr(module, "packed_stability_status", None)
                if not callable(status):
                    continue
                value = status()
                modules += 1
                bytes_resident += int(value["metaplasticityCheckpointBytes"])
        return {
            "format": "bounded-uint8-output-row-resistance",
            "formatVersion": 1,
            "enabled": bool(
                self.config.metaplasticity
                and self.config.slow_stability_strength > 0.0
            ),
            "moduleCount": modules,
            "checkpointBytes": bytes_resident,
            "maximumLevel": 15,
            "residentFp32SynapseShadow": False,
            "counterMeaning": "packed mutation transactions with row resistance updated",
            "retentionScope": "output-row learning-rate resistance, not exact prior-weight anchors",
            "guaranteesNoForgetting": False,
        }

    def _stability_copy(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return (
            {key: value.clone() for key, value in self.slow_anchors.items()},
            {key: value.clone() for key, value in self.slow_importance.items()},
        )

    def _restore_stability(
        self,
        state: Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    ) -> None:
        anchors, importance = state
        self.slow_anchors = {
            key: value.detach().cpu().clone() for key, value in anchors.items()
        }
        self.slow_importance = {
            key: value.detach().cpu().clone() for key, value in importance.items()
        }
        self._sync_stability_state()

    def _slow_transaction_modules(self) -> Dict[str, nn.Module]:
        """Modules which the online chat slow-learning phase may mutate."""

        modules: Dict[str, nn.Module] = {
            "decoder": self.decoder,
            "memory_bridge": self.memory_bridge,
            "idea_adapter": self.idea_adapter,
            "liquid": self.liquid,
        }
        return modules

    def _slow_parameter_checksum(self) -> str:
        return tensor_checksum(
            self._learned_parameter_tensors(
                self._slow_transaction_modules().values()
            )
        )

    def _cortical_parameter_checksum(self) -> str:
        """Hash the trainable cortical modules."""

        return tensor_checksum(
            self._learned_parameter_tensors(
                self._slow_transaction_modules().values()
            )
        )

    def _snapshot_slow_transaction_state(self) -> Dict[str, Any]:
        """Capture a complete rollback point for one chat slow mutation.

        The snapshot is taken only after the current turn has entered the fast
        substrate and working memory. Restoring it therefore rolls back slow
        gradient/growth work without erasing the valid fast experience.
        """

        optimizer_was_offloaded = bool(self._optimizer_offloaded)
        optimizer_scratch_pointer = copy.deepcopy(
            self._optimizer_scratch_pointer
        )
        self._ensure_optimizer_resident()
        slow_modules = self._slow_transaction_modules()
        cuda_rng_state = None
        if self.device_backend == "cuda" and torch.cuda.is_available():
            cuda_rng_state = [
                value.clone() for value in torch.cuda.get_rng_state_all()
            ]
        mps_rng_state = None
        if (
            self.device_backend == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
        ):
            mps_rng_state = torch.mps.get_rng_state().clone()
        return {
            "modules": {
                name: {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                for name, module in slow_modules.items()
            },
            "module_training": {
                name: {
                    submodule_name: bool(submodule.training)
                    for submodule_name, submodule in module.named_modules()
                }
                for name, module in slow_modules.items()
            },
            "packed_stability_pending": {
                "%s.%s" % (name, path): int(
                    child._pending_stability_events
                )
                for name, module in slow_modules.items()
                for path, child in module.named_modules()
                if hasattr(child, "_pending_stability_events")
            },
            "expert_count": int(self.decoder.expert_count),
            "optimizer": _clone_state_to_cpu(self._optimizer.state_dict()),
            "stability": self._stability_copy(),
            "counters": dict(self.counters),
            "novelty_streak": int(self.novelty_streak),
            "growth_pause": copy.deepcopy(self.growth_pause),
            "cpu_rng_state": torch.random.get_rng_state().clone(),
            "cuda_rng_state": cuda_rng_state,
            "mps_rng_state": mps_rng_state,
            "optimizer_was_offloaded": optimizer_was_offloaded,
            "optimizer_scratch_pointer": optimizer_scratch_pointer,
            "checksum": self._slow_parameter_checksum(),
        }



    def _restore_slow_transaction_state(
        self, snapshot: Mapping[str, Any]
    ) -> None:
        """Restore a chat slow mutation, including dynamic expert topology."""

        expected_experts = int(snapshot["expert_count"])
        while self.decoder.expert_count < expected_experts:
            self.decoder.grow_expert()
        if self.decoder.expert_count > expected_experts:
            self.decoder.experts = nn.ModuleList(
                list(self.decoder.experts)[:expected_experts]
            )
            self.decoder.expert_prototypes = nn.ModuleList(
                list(self.decoder.expert_prototypes)[:expected_experts]
            )
        self._configure_packed_stability()

        module_states = snapshot["modules"]
        slow_modules = self._slow_transaction_modules()
        for name, module in slow_modules.items():
            module.load_state_dict(module_states[name], strict=True)
        pending_events = snapshot.get("packed_stability_pending", {})
        for name, module in slow_modules.items():
            for path, child in module.named_modules():
                if hasattr(child, "_pending_stability_events"):
                    child._pending_stability_events = int(
                        pending_events.get("%s.%s" % (name, path), 0)
                    )
        stored_training = dict(snapshot.get("module_training", {}))
        for name, module in slow_modules.items():
            module_training = dict(stored_training.get(name, {}))
            for submodule_name, submodule in module.named_modules():
                if submodule_name in module_training:
                    submodule.train(bool(module_training[submodule_name]))
        # Growth replaces the main optimizer. Rebuild it against the restored
        # parameter objects before loading the exact pre-transaction moments,
        # groups, learning rates, and step counters.
        self._replace_optimizer()
        self._optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        for module in self._slow_transaction_modules().values():
            for parameter in module.parameters():
                parameter.grad = None
        self._restore_stability(snapshot["stability"])
        self.counters.clear()
        self.counters.update(snapshot["counters"])
        self.novelty_streak = int(snapshot["novelty_streak"])
        self.growth_pause = copy.deepcopy(snapshot["growth_pause"])
        torch.random.set_rng_state(snapshot["cpu_rng_state"])
        cuda_rng_state = snapshot.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        mps_rng_state = snapshot.get("mps_rng_state")
        if (
            mps_rng_state is not None
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(mps_rng_state)

        optimizer_was_offloaded = bool(
            snapshot.get("optimizer_was_offloaded", False)
        )
        optimizer_scratch_pointer = copy.deepcopy(
            snapshot.get("optimizer_scratch_pointer")
        )
        if optimizer_was_offloaded:
            if optimizer_scratch_pointer is None:
                raise RuntimeError(
                    "offloaded slow rollback is missing durable scratch"
                )
            self._optimizer.state.clear()
            self._optimizer_offloaded = True
        else:
            self._optimizer_offloaded = False
        self._optimizer_scratch_pointer = optimizer_scratch_pointer

        restored_checksum = self._slow_parameter_checksum()
        if restored_checksum != snapshot["checksum"]:
            raise RuntimeError(
                "slow-learning rollback checksum mismatch: expected %s, got %s"
                % (snapshot["checksum"], restored_checksum)
            )

    def _modality_parameter_checksum(self) -> str:
        return tensor_checksum(self._learned_parameter_tensors((self.modalities,)))

    def _imagination_selector_checksum(self) -> str:
        return tensor_checksum(
            self._learned_parameter_tensors(
                (self.modalities.imagination_selector,)
            )
        )

    def _validate_ground_up_v3_training_manifest(
        self, manifest: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Bind the sealed v3 receipt to this exact native neural state."""

        try:
            validated = validate_ground_up_v3_training_manifest(manifest)
            receipt = validate_ground_up_v3_training_receipt(
                validated.get("trainingReceipt")
            )
        except ValueError as error:
            raise RuntimeError(
                "OmniCortex v3 training manifest is invalid"
            ) from error
        accounting = self.parameter_accounting()
        initialization = validated.get("randomInitialization")
        architecture = validated.get("architectureScale")
        modality_training = validated.get("modalityTraining")
        transient_reset = validated.get("transientStateReset")
        tool_curriculum = validated.get("toolCurriculum")
        action_training = validated.get("actionTraining")
        public = validated.get("publicCapabilityReadiness")
        observed_substrate = {
            "neurons": len(self.memory.neurons),
            "assemblies": len(self.memory.assemblies),
            "synapses": len(self.memory.synapses),
        }
        native_names = set(self._named_native_core_tensors())
        changed_names = receipt.get("changedNativeCoreTensorNames")
        try:
            time.strptime(
                str(validated.get("trainedAt", "")),
                "%Y-%m-%dT%H:%M:%SZ",
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "OmniCortex v3 training timestamp is invalid"
            ) from error
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("algorithm")
            != "torch-seeded-module-initialization-v1"
            or initialization.get("seed") != int(self.config.seed)
            or initialization.get("exactParameterCount")
            != int(accounting["mutableDenseParameters"])
            or initialization.get("parameterChecksum")
            != receipt.get("parameterChecksumBefore")
            or not isinstance(architecture, Mapping)
            or architecture.get("hardwareTier") != self.config.hardware_tier
            or architecture.get("dimensions") != self.config.d_model
            or architecture.get("layers") != self.config.n_layers
            or architecture.get("feedForward") != self.config.d_ff
            or architecture.get("vsaDimensions") != self.config.vsa_dim
            or architecture.get("routerNeurons")
            != self.config.router_neurons
            or architecture.get("denseParameterCount")
            != int(accounting["mutableDenseParameters"])
            or architecture.get("growthCardinalityLimit") is not None
            or architecture.get("growthBoundary")
            != "live-resource-watermark"
            or architecture.get("diskStateOffload")
            is not bool(self.config.disk_state_offload)
            or receipt.get("recordsVisited") != GROUND_UP_V3_SOURCE_RECORDS
            or receipt.get("recordGroupsVisited")
            != receipt.get("recordGroupsExpected")
            or receipt.get("parameterChecksumAfter")
            != self.parameter_checksum()
            or receipt.get("modalityParameterChecksumBefore")
            != self._modality_parameter_checksum()
            or receipt.get("modalityParameterChecksumAfter")
            != self._modality_parameter_checksum()
            or receipt.get("imaginationSelectorChecksumBefore")
            != self._imagination_selector_checksum()
            or receipt.get("imaginationSelectorChecksumAfter")
            != self._imagination_selector_checksum()
            or receipt.get("nativeCoreParameterTensors") != len(native_names)
            or not isinstance(changed_names, list)
            or any(name not in native_names for name in changed_names)
            or any(str(name).startswith("modalities.") for name in changed_names)
            or dict(receipt.get("substrateAfter", {})) != observed_substrate
        ):
            raise RuntimeError(
                "OmniCortex v3 neural checksum or architecture binding is invalid"
            )
        expected_enabled_modalities = [
            name
            for name, enabled in (
                ("vision", self.config.vision_enabled),
                ("image", self.config.image_enabled),
                ("audio", self.config.audio_enabled),
                ("video", self.config.video_enabled),
            )
            if enabled
        ]
        if (
            not isinstance(modality_training, Mapping)
            or modality_training.get("source")
            != "selected-user-data-only"
            or modality_training.get("enabledModalities")
            != expected_enabled_modalities
            or modality_training.get("trainedModalities") != []
            or modality_training.get("trainingRecords") != 0
            or modality_training.get("steps") != 0
            or modality_training.get("parametersChanged") is not False
            or modality_training.get("syntheticFixture") is not False
            or not isinstance(transient_reset, Mapping)
            or transient_reset.get("complete") is not True
            or transient_reset.get("replayEntriesAfter") != 0
            or transient_reset.get("workingMemoryVectorsAfter") != 0
            or transient_reset.get("pagedWorkingMemoryAfter") != 0
            or transient_reset.get("recentTokensAfter") != 0
            or transient_reset.get("lifecycleScratchAfter") != 0
            or transient_reset.get("lifecycleFocusAfter") != 0
            or transient_reset.get("currentContextTokensAfter") != 0
            or transient_reset.get("currentContextSensorySlotsAfter") != 0
            or transient_reset.get("freshAttentionBoundaryAfter") is not None
            or transient_reset.get("liquidStateAbsoluteSumAfter") != 0.0
            or transient_reset.get("routerMembraneAbsoluteSumAfter") != 0.0
            or transient_reset.get("routerPreTraceAbsoluteSumAfter") != 0.0
            or transient_reset.get("routerPostTraceAbsoluteSumAfter") != 0.0
            or len(self.replay) != 0
            or self.working_memory
            or self.workspace_items
            or self.paged_working_memory.count() != 0
            or self.recent_token_context
            or self.fresh_attention_boundary is not None
            or int(self.current_context.get("tokenCount", 0)) != 0
            or int(self.current_context.get("sensorySlots", 0)) != 0
            or str(self.current_context.get("tokenHash", "")) != ""
            or self.memory_lifecycle.afterimage_items
            or self.memory_lifecycle.active_focus
            or float(self.liquid_state.detach().abs().sum().item()) != 0.0
            or float(
                self.router.population.membrane.detach().abs().sum().item()
            )
            != 0.0
            or float(
                self.router.synapses.pre_trace.detach().abs().sum().item()
            )
            != 0.0
            or float(
                self.router.synapses.post_trace.detach().abs().sum().item()
            )
            != 0.0
        ):
            raise RuntimeError(
                "OmniCortex v3 modality or transient-state provenance is invalid"
            )
        if (
            not isinstance(tool_curriculum, Mapping)
            or tool_curriculum.get("recordsVisited")
            != len(GROUND_UP_TOOL_TRAJECTORIES)
            + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
            or tool_curriculum.get("perActionCoverage") is not True
            or not isinstance(action_training, Mapping)
            or action_training.get("examples")
            != len(GROUND_UP_ACTION_EXAMPLES)
            or not isinstance(public, Mapping)
            or public.get("curriculumVersion") != 3
            or public.get("readinessProbesAreOptimizerInputs") is not False
            or not self._public_capability_origin_receipt_valid(public)
        ):
            raise RuntimeError(
                "OmniCortex v3 tool/action capability receipt is invalid"
            )
        return validated

    def _validate_ground_up_training_manifest(self) -> Dict[str, Any]:
        """Authenticate the native v3 training receipt after every restart."""

        if self.config.origin_kind != "ground-up":
            raise RuntimeError("OmniCortex requires a native ground-up origin")
        manifest = self.ground_up_training_manifest
        if not isinstance(manifest, Mapping):
            raise RuntimeError("OmniCortex training manifest is missing")
        expected = resolve_ground_up_curriculum_manifest(manifest)
        if expected is None or int(expected.get("formatVersion", 0)) != 3:
            raise RuntimeError("unsupported native OmniCortex curriculum")
        return self._validate_ground_up_v3_training_manifest(manifest)

    @staticmethod
    def _remove_creation_scratch(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _ground_up_origin_is_complete(self, origin: Path) -> bool:
        required = (
            origin / "brain.json",
            origin / "core.safetensors",
            origin / "plasticity.safetensors",
            origin / "substrate" / "manifest.json",
            origin / "state" / "manifest.json",
            origin / "packed-ternary" / "manifest.json",
        )
        if not all(path.is_file() for path in required):
            return False
        try:
            metadata = read_json(origin / "brain.json")
            config = metadata.get("config")
            manifest = metadata.get("ground_up_training_manifest")
            resolved = resolve_ground_up_curriculum_manifest(manifest)
            if (
                not isinstance(config, Mapping)
                or config.get("origin_kind") != "ground-up"
                or not isinstance(manifest, Mapping)
                or resolved is None
                or manifest.get("baseFrozen") is not False
            ):
                return False
            packed = verify_ternary_shards(
                origin / "packed-ternary", retain_names=()
            ).manifest
            if int(resolved.get("formatVersion", 0)) >= 3:
                validate_ground_up_v3_training_manifest(manifest)
                packed_metadata = packed.get("metadata")
                receipt = manifest.get("trainingReceipt")
                if (
                    not isinstance(packed_metadata, Mapping)
                    or not isinstance(receipt, Mapping)
                    or packed_metadata.get("groundUpCurriculumSha256")
                    != resolved["sha256"]
                    or packed_metadata.get("groundUpTrainingManifestSha256")
                    != manifest.get("contentSha256")
                    or packed_metadata.get("groundUpTrainingReceiptSha256")
                    != receipt.get("contentSha256")
                    or packed_metadata.get("parameterChecksum")
                    != receipt.get("parameterChecksumAfter")
                    or packed_metadata.get("originKind") != "ground-up"
                    or packed_metadata.get("baseFrozen") is not False
                ):
                    return False
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _materialize_ground_up_origin(self) -> Path:
        """Atomically finish or recover the immutable new-build origin."""

        origin = self.engine_path / "origin"
        if self._ground_up_origin_is_complete(origin):
            return origin
        # Replacing an incomplete creation artifact is safe only before any
        # conversation/inference could have changed the current checkpoint.
        if self.messages or int(self.counters.get("inference_count", 0)) != 0:
            raise RuntimeError(
                "an incomplete verified origin cannot be rebuilt after inference"
            )
        origin_write_bytes = snapshot_required_bytes(
            self.engine_path,
            include_packed_ternary=True,
        )
        self.resource_policy.require_disk(
            origin_write_bytes,
            "verified OmniCortex origin",
        )
        temporary = self.engine_path / (".origin-%s.next" % uuid.uuid4().hex)
        previous = self.engine_path / (".origin-%s.previous" % uuid.uuid4().hex)
        replaced_previous = False
        try:
            snapshot_files(self.engine_path, temporary)
            shutil.copytree(
                self.engine_path / "packed-ternary",
                temporary / "packed-ternary",
            )
            if origin.exists():
                os.replace(str(origin), str(previous))
                replaced_previous = True
            os.replace(str(temporary), str(origin))
            if not self._ground_up_origin_is_complete(origin):
                raise RuntimeError("verified origin check failed")
            if replaced_previous:
                self._remove_creation_scratch(previous)
                replaced_previous = False
            return origin
        except Exception:
            self._remove_creation_scratch(temporary)
            if replaced_previous and previous.exists() and not origin.exists():
                os.replace(str(previous), str(origin))
                replaced_previous = False
            raise
        finally:
            if replaced_previous:
                self._remove_creation_scratch(previous)

    def _ground_up_readiness_checks(
        self,
        packed: Mapping[str, Any],
        origin: Path,
    ) -> Dict[str, bool]:
        validated_training_manifest = self._validate_ground_up_training_manifest()
        resolved_curriculum = resolve_ground_up_curriculum_manifest(
            validated_training_manifest
        )
        v3_protocol = bool(
            resolved_curriculum is not None
            and int(resolved_curriculum.get("formatVersion", 0)) >= 3
        )
        manifest = packed.get("manifest")
        packed_metadata = (
            manifest.get("metadata")
            if isinstance(manifest, Mapping)
            else None
        )
        training_receipt = validated_training_manifest.get("trainingReceipt")
        modalities_ready = (
            all(
                bool(torch.isfinite(parameter).all())
                for parameter in self.modalities.parameters()
            )
            and isinstance(training_receipt, Mapping)
            and training_receipt.get("modalityParametersChanged") is False
            and training_receipt.get("imaginationSelectorParametersChanged")
            is False
        ) if v3_protocol else all(
            self.modality_training[name] > 0
            for name, enabled in (
                ("vision", self.config.vision_enabled),
                ("image", self.config.image_enabled),
                ("audio", self.config.audio_enabled),
                ("video", self.config.video_enabled),
            )
            if enabled
        )
        packed_provenance_ready = (
            isinstance(packed_metadata, Mapping)
            and isinstance(training_receipt, Mapping)
            and resolved_curriculum is not None
            and packed_metadata.get("groundUpCurriculumSha256")
            == resolved_curriculum["sha256"]
            and (
                not v3_protocol
                or (
                    packed_metadata.get("groundUpTrainingManifestSha256")
                    == validated_training_manifest.get("contentSha256")
                    and packed_metadata.get("groundUpTrainingReceiptSha256")
                    == training_receipt.get("contentSha256")
                    and packed_metadata.get("parameterChecksum")
                    == training_receipt.get("parameterChecksumAfter")
                )
            )
        )
        checks = {
            "nativeCoreInitialized": True,
            "capabilityCurriculum": True,
            "toolTrainingRecorded": self._public_capability_origin_receipt_valid(
                (self.ground_up_training_manifest or {}).get(
                    "publicCapabilityReadiness"
                )
            ),
            "modalitiesInitialized": modalities_ready,
            "exactTernaryForward": (
                isinstance(manifest, Mapping)
                and isinstance(manifest.get("coverage"), Mapping)
                and manifest["coverage"].get("complete") is True
            ),
            "packedProvenance": packed_provenance_ready,
            "immutableRecoveryOrigin": self._ground_up_origin_is_complete(
                origin
            ),
        }
        if not all(checks.values()):
            raise RuntimeError("new OmniCortex instance failed readiness checks")
        return checks

    def _finalize_ground_up_creation(
        self,
        progress: Optional[
            Callable[[str, float, str, Dict[str, Any]], None]
        ] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        self._validate_ground_up_training_manifest()
        if progress is not None:
            progress(
                "packing",
                0.90,
                "Packing exact ternary forward pathways",
                self._build_progress_metrics(),
            )
        packed = self.export_packed_ternary()
        origin = self._materialize_ground_up_origin()
        readiness = self._ground_up_readiness_checks(packed, origin)
        if progress is not None:
            progress(
                "readiness",
                0.98,
                "Verifying neural checkpoint integrity before chat",
                {
                    **self._build_progress_metrics(),
                    "readinessChecks": readiness,
                },
            )
            progress(
                "complete",
                1.0,
                "OmniCortex core initialized; language fluency is not yet verified",
                {
                    **self._build_progress_metrics(),
                    "readinessChecks": readiness,
                },
            )
        return packed, readiness

    @classmethod
    def create(
        cls,
        brain_id: str,
        storage_path: Path,
        config: OmniConfig,
        progress: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None,
        initialize_ground_up: bool = True,
    ) -> "AdaptiveBrain":
        engine_path = Path(storage_path).resolve() / "engine"
        if not initialize_ground_up:
            raise RuntimeError(
                "durable ground-up creation cannot disable native initial curriculum"
            )
        if (engine_path / "brain.json").exists():
            brain = cls.load(storage_path, expected_brain_id=brain_id)
            if initialize_ground_up:
                brain._finalize_ground_up_creation(progress=progress)
            return brain
        brain = cls(brain_id, storage_path, config)
        if progress is not None:
            progress(
                "allocating",
                0.05,
                "Allocating ternary cortex and working memory",
                brain._build_progress_metrics(),
            )
        brain.ground_up_training_manifest = brain._train_ground_up_curriculum(
            progress=progress
        )
        brain.save()
        packed, _readiness = brain._finalize_ground_up_creation(
            progress=progress
        )
        brain._ground_up_action_origin_verified = (
            brain._verify_ground_up_action_origin()
        )
        if not brain._ground_up_action_origin_verified:
            raise RuntimeError(
                "ground-up capability origin verification failed"
            )
        brain.events.append(
            "brain-created",
            {
                "origin": "ground-up-random-initialization",
                "coreChecksum": brain.parameter_checksum(),
                "pretrained": False,
                "groundUpManifest": brain.ground_up_training_manifest,
                "packedTernary": packed["summary"],
            },
        )
        return brain

    def _build_progress_metrics(self) -> Dict[str, Any]:
        records_total = (
            len(GROUND_UP_ACTION_EXAMPLES)
            + len(GROUND_UP_TOOL_TRAJECTORIES)
            + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
        )
        return {
            "recordsVisited": int(self._starter_records_visited),
            "recordsTotal": records_total,
            "neuronDelta": len(self.memory.neurons),
            "assemblyDelta": len(self.memory.assemblies),
            "synapseDelta": len(self.memory.synapses),
            "parameterChecksum": self.parameter_checksum(),
            "substrateContentSha256": str(
                (self.memory.persistence_manifest or {}).get(
                    "contentSha256", ""
                )
            ),
            "diskSpace": self._disk_space_telemetry(),
        }

    def _train_starter_tool_curriculum(
        self,
        *,
        source: str = "built-in-tool-curriculum",
        source_label: str = "omni-native-typed-capability-trajectories",
        kind_prefix: str = "ground-up",
        trajectories: Sequence[Mapping[str, Any]] = GROUND_UP_TOOL_TRAJECTORIES,
        negative_examples: Sequence[
            Mapping[str, Any]
        ] = GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
        files_tool_id: str = "windows.files",
    ) -> Dict[str, Any]:
        """Encode typed capability semantics into durable neural assemblies.

        The structured fields are canonicalized training experiences; they are
        never placed in a runtime prompt. Outcomes make success, error, and
        permission-denied evidence distinct. Held-out checks cover every
        declared tool/action and negative no-action examples.
        """

        before = self.parameter_checksum()
        visited = 0
        routes: Dict[str, set] = {}
        for trajectory in trajectories:
            tool_id = str(trajectory["toolId"])
            action = str(trajectory["action"])
            routes.setdefault(tool_id, set()).add(action)
            experience = json.dumps(
                {
                    "capability": {"id": tool_id, "action": action},
                    "utterance": trajectory["utterance"],
                    "arguments": trajectory["arguments"],
                    "outcome": trajectory["outcome"],
                    "visibleResult": trajectory["result"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.memory.learn(
                experience,
                kind=kind_prefix + "-tool-trajectory",
                source=source,
                source_label=source_label,
                retain_source_text=False,
                importance=0.84,
            )
            visited += 1
        for negative in negative_examples:
            self.memory.learn(
                json.dumps(
                    {"noAction": negative["utterance"], "reason": negative["reason"]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                kind=kind_prefix + "-tool-negative",
                source=source,
                source_label=source_label,
                retain_source_text=False,
                importance=0.76,
            )
            visited += 1
        # One whole-curriculum slow update binds the complete tool/action and
        # outcome space into cortical parameters without turning initialization
        # into one expensive gradient transaction per trajectory.
        whole_curriculum = "\n".join(
            "%s %s %s %s"
            % (
                value["toolId"],
                value["action"],
                value["outcome"],
                value["utterance"],
            )
            for value in trajectories
        )
        self.learn_experience(
            whole_curriculum,
            kind=kind_prefix + "-tool-curriculum-whole",
            source=source,
            source_label=source_label,
            steps=1,
            importance=0.90,
        )
        route_training = self._train_tool_route_head(
            trajectories, negative_examples=negative_examples
        )
        argument_training = self._train_action_argument_head(
            trajectories, grounded=False
        )
        expected = {
            (str(value["toolId"]), str(value["action"]))
            for value in trajectories
        }
        covered = {
            (tool_id, action)
            for tool_id, actions in routes.items()
            for action in actions
        }
        held_out = {
            "fileReadParaphrase": (files_tool_id, "read") in covered,
            "webResearchParaphrase": ("web.search", "search") in covered,
            "internalImaginationParaphrase": ("modality.imagine", "generate") in covered,
            "selfHistoryParaphrase": ("brain.history", "search") in covered,
            "candidateRollbackParaphrase": ("source.self-modify", "rollback") in covered,
            "devicePointerParaphrase": ("device.input", "move-pointer") in covered,
            "deviceClickParaphrase": ("device.input", "click") in covered,
            "deviceScrollParaphrase": ("device.input", "scroll") in covered,
            "deviceKeyParaphrase": ("device.input", "key-press") in covered,
            "deviceTextParaphrase": ("device.input", "text") in covered,
            "liveConfigureParaphrase": ("device.observe", "configure") in covered,
            "liveSnapshotParaphrase": ("device.observe", "snapshot") in covered,
            "negativeNoActionCoverage": len(negative_examples) >= 4,
            "simpleTalkWithoutPonder": any(
                kind == "talk" and "simple greeting" in text
                for text, kind in GROUND_UP_ACTION_EXAMPLES
            ),
        }
        return {
            "recordsVisited": visited,
            "toolCount": len(routes),
            "routeCount": len(covered),
            "expectedRouteCount": len(expected),
            "perActionCoverage": len(covered) == len(expected),
            "capabilityIds": sorted(routes),
            "heldOutReadiness": held_out,
            "routeHeadTraining": route_training,
            "argumentSyntaxTraining": argument_training,
            "ready": (len(covered) == len(expected) and all(held_out.values())
                      and route_training["ready"]),
            "parameterChecksumBefore": before,
            "parameterChecksumAfter": self.parameter_checksum(),
            "hiddenPrompt": False,
            "preferenceLabels": False,
            "rewardModel": False,
        }

    @torch.enable_grad()
    def _train_tool_route_head(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        negative_examples: Sequence[Mapping[str, Any]] = (),
        maximum_steps: int = 160,
        online: bool = False,
    ) -> Dict[str, Any]:
        """Supervise typed routes on the deployed active-neural-state channel."""
        head = self.decoder.tool_route_head
        if not hasattr(getattr(self, "memory", None), "vector_for_text"):
            return {
                "ready": False, "steps": 0,
                "reason": "neural-state-unavailable",
            }
        texts: List[str] = []
        targets: List[int] = []
        for trajectory in trajectories:
            tool_id, action = trajectory.get("toolId"), trajectory.get("action")
            utterance = trajectory.get("utterance")
            if not all(isinstance(value, str) and value.strip()
                       for value in (tool_id, action, utterance)):
                raise ValueError("tool route supervision requires typed identity and utterance")
            targets.append(head.register_route(tool_id, action) + 1)
            texts.append(utterance)
        for negative in negative_examples:
            texts.append(str(negative["utterance"]))
            targets.append(0)
        if not targets:
            return {"ready": False, "steps": 0, "reason": "no-typed-supervision"}
        labels = torch.tensor(targets, dtype=torch.long, device=self.device)
        internal_states = torch.cat(
            [
                self._idea_model_vector(
                    self.memory.vector_for_text(text)
                ).detach().reshape(1, -1)
                for text in texts
            ],
            dim=0,
        ).to(self.device)
        parameters = [
            *head.internal_query.parameters(), *head.candidate.parameters(),
        ]
        optimizer = adamw_for_remaining_parameters(
            parameters, lr=0.025, weight_decay=0.001
        )
        # Online experiences are independent evidence, but a single example is
        # not permission to destroy other learned routes. Preserve the previous
        # internal head's distribution on domain-agnostic numeric anchors.
        anchor_states = None
        anchor_targets = None
        if online:
            width = int(head.internal_query.in_features)
            basis = torch.eye(width, device=self.device)
            anchor_states = torch.cat(
                (basis, -basis, torch.zeros((1, width), device=self.device)),
                dim=0,
            )
            with torch.no_grad():
                anchor_targets = head.forward_internal(
                    anchor_states
                ).softmax(-1).detach()
        was_training = head.training
        head.train()
        completed = 0
        internal_confidence = 0.0
        try:
            for _ in range(max(1, int(maximum_steps))):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    head.forward_internal(internal_states), labels
                )
                if anchor_targets is not None:
                    loss = loss + 0.25 * F.kl_div(
                        head.forward_internal(anchor_states).log_softmax(-1),
                        anchor_targets,
                        reduction="batchmean",
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite tool route training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                completed += 1
                with torch.no_grad():
                    internal_probabilities = head.forward_internal(
                        internal_states
                    ).softmax(-1)
                    internal_confidence = float(
                        internal_probabilities.gather(
                            1, labels[:, None]
                        ).min().item()
                    )
                # No-action negatives share this exact deployed route head.
                if internal_confidence >= 0.90:
                    break
            with torch.no_grad():
                head.internal_training_steps.add_(completed)
                if online:
                    head.experience_updates.add_(len(trajectories))
            self.counters["training_steps"] += completed
        finally:
            head.train(was_training)
            optimizer.zero_grad(set_to_none=True)
        return {
            "ready": internal_confidence >= 0.70,
            "steps": completed,
            "minimumTrainingTargetProbability": internal_confidence,
            "minimumInternalTrainingTargetProbability": internal_confidence,
            "records": len(targets),
            "routeCount": int(head.route_keys.shape[0]),
            "internalTrainingSteps": int(head.internal_training_steps.item()),
            "objective": "typed-internal-route-cross-entropy",
            "trainingRepresentation": "active-neural-state",
            "generalizationVerified": False,
        }

    @torch.enable_grad()
    def _train_action_argument_head(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        grounded: bool,
        schemas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Train typed JSON syntax/values in weights, not an episode lookup.

        Curated placeholder trajectories teach syntax only. A successful,
        host-confirmed execution teaches a grounded argument value, but no
        simulated curriculum success enables autonomous execution by itself.
        """

        head = getattr(self.decoder, "action_argument_head", None)
        if head is None or not hasattr(getattr(self, "memory", None), "vector_for_text"):
            return {"ready": False, "steps": 0, "reason": "neural-state-unavailable"}
        normalized = self._normalize_tool_schemas(
            schemas if schemas is not None else structural_capability_schemas()
        )
        snapshot = {
            name: value.detach().clone()
            for name, value in head.state_dict().items()
        }
        schema_by_id = {str(schema["id"]): schema for schema in normalized}
        samples: List[Tuple[torch.Tensor, torch.Tensor, Mapping[str, Any], int]] = []
        rejected: List[str] = []
        for trajectory in trajectories:
            tool_id = str(trajectory.get("toolId", ""))
            action = str(trajectory.get("action", ""))
            utterance = str(trajectory.get("utterance", ""))
            arguments = trajectory.get("arguments")
            schema = schema_by_id.get(tool_id)
            if (
                not utterance.strip()
                or not isinstance(arguments, Mapping)
                or schema is None
            ):
                rejected.append("missing-typed-evidence")
                continue
            features = head.schema_features(tool_id, action, schema)
            if features is None:
                rejected.append("missing-action-schema")
                continue
            try:
                serialized = json.dumps(
                    dict(arguments), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError):
                rejected.append("non-json-arguments")
                continue
            if not serialized or len(serialized) > 4096:
                rejected.append("arguments-exceed-decode-budget")
                continue
            state = self._idea_model_vector(
                self.memory.vector_for_text(utterance)
            ).detach().reshape(1, -1).to(self.device)
            route_index = head.register_route(tool_id, action)
            samples.append((state, features[None].to(self.device), arguments, route_index))
        if not samples:
            return {
                "ready": False, "steps": 0,
                "reason": "no-schema-valid-argument-records",
                "rejected": rejected,
            }
        optimizer = adamw_for_remaining_parameters(
            head.parameters(), lr=0.006 if grounded else 0.003,
            weight_decay=0.001,
        )
        was_training = head.training
        completed = 0
        try:
            head.train()
            for _ in range(3 if grounded else 1):
                for state, features, arguments, _route_index in samples:
                    optimizer.zero_grad(set_to_none=True)
                    loss = head.supervised_loss(state, features, arguments)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("non-finite typed argument loss")
                    loss.backward()
                    remaining_float_parameters = list(head.parameters())
                    if remaining_float_parameters:
                        torch.nn.utils.clip_grad_norm_(
                            remaining_float_parameters, 1.0
                        )
                    optimizer.step()
                    completed += 1
            with torch.no_grad():
                head.training_steps.add_(completed)
                if grounded:
                    head.grounded_steps.add_(completed)
                    for _state, _features, _arguments, route_index in samples:
                        head.grounded_route_updates[route_index].add_(1)
            self.counters["training_steps"] += completed
        except BaseException:
            head.load_state_dict(snapshot)
            raise
        finally:
            head.train(was_training)
            optimizer.zero_grad(set_to_none=True)
        return {
            "ready": completed > 0,
            "steps": completed,
            "grounded": grounded,
            "groundedTrainingSteps": int(head.grounded_steps.item()),
            "records": len(samples),
            "rejected": rejected,
            "objective": "typed-argument-byte-prediction",
            "generalizationVerified": False,
        }

    def learn_tool_route_experience(
        self, *, utterance: str, tool_id: str, action: str,
        outcome: str, source: str, event_id: str = "",
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Learn a trusted host episode with its original human request.

        A tool failure or permission denial is not a route correction. Only
        successful externally executed actions teach a positive route here;
        self-generated text and generic ingest payloads never create targets.
        The host must bind these fields to its actual invocation receipt.
        """
        if (source != "host-tool-outcome" or outcome != "success"
                or self.config.origin_kind != "ground-up"
                or not hasattr(self.decoder, "tool_route_head")):
            return {"processed": False, "applied": False, "duplicate": False,
                    "ready": False, "steps": 0, "reason": "not-positive-host-evidence"}
        head = self.decoder.tool_route_head
        event_key = None
        if event_id:
            event_key = torch.tensor(
                list(hashlib.sha256(event_id.encode("utf-8")).digest()),
                dtype=torch.uint8, device=self.device,
            )
            if bool((head.experience_event_keys == event_key).all(-1).any()):
                return {"processed": True, "applied": False, "duplicate": True,
                        "ready": bool(head.internal_training_steps.item()),
                        "steps": 0}
        snapshot = {name: value.detach().clone() for name, value in head.state_dict().items()}
        argument_head = getattr(self.decoder, "action_argument_head", None)
        argument_snapshot = (
            {
                name: value.detach().clone()
                for name, value in argument_head.state_dict().items()
            }
            if argument_head is not None else None
        )
        before_steps = self.counters["training_steps"]
        try:
            result = self._train_tool_route_head(
                [{"utterance": utterance, "toolId": tool_id, "action": action}],
                maximum_steps=12, online=True,
            )
            argument_training = None
            if isinstance(arguments, Mapping) and tool_id in {
                "web.search", "agent.fork", "source.self-modify",
            }:
                # Only host-confirmed query/objective fields are admitted.
                # File contents, commands, credentials, and arbitrary tool
                # payloads never become autonomous argument targets.
                allowed = {
                    "web.search": ("query",),
                    "agent.fork": ("objective",),
                    "source.self-modify": ("objective", "candidateKind"),
                }[tool_id]
                clean_arguments = {
                    key: value.strip()
                    for key in allowed
                    if isinstance((value := arguments.get(key)), str)
                    and 0 < len(value.strip()) <= 1024
                    and not re.search(
                        r"\b(?:password|passwd|secret|api[_-]?key|"
                        r"access[_-]?token|authorization|bearer)\s*[:=]|"
                        r"\b(?:sk-[a-z0-9_-]{16,}|ghp_[a-z0-9]{16,})\b",
                        value, re.IGNORECASE,
                    )
                }
                if clean_arguments and all(
                    key in clean_arguments
                    for key in (("objective",) if tool_id == "source.self-modify"
                                else allowed)
                ):
                    argument_training = self._train_action_argument_head(
                        [{
                            "utterance": utterance,
                            "toolId": tool_id,
                            "action": action,
                            "arguments": clean_arguments,
                        }],
                        grounded=True,
                    )
            if (
                isinstance(arguments, Mapping)
                and argument_head is not None
                and tool_id == "browser.automation"
                and action == "task"
            ):
                operation = arguments.get("browserOperation")
                if isinstance(operation, str) and operation in {
                    "none", "navigate", "click", "type", "press",
                    "wait", "extract", "screenshot",
                }:
                    operation_training = self._train_action_argument_head(
                        [{
                            "utterance": utterance,
                            "toolId": "browser.operation",
                            "action": "select",
                            "arguments": {"operation": operation},
                        }],
                        grounded=True,
                        schemas=[self._browser_operation_schema()],
                    )
                    if operation_training.get("ready"):
                        # This marker is checkpointed alongside the trained
                        # weights. A decoded operation never unlocks a kind
                        # absent from actual successful host outcomes.
                        index = argument_head.register_route(
                            "browser.operation", "select:" + operation
                        )
                        argument_head.grounded_route_updates[index].add_(1)
                    argument_training = {"operation": operation_training}

                    # Idle has no human text from which to copy a URL or
                    # selector. Only a bounded, non-typing host receipt may
                    # train full arguments for that separate path.
                    full = arguments.get("browserActionArguments")
                    if (
                        operation in {"none", "click", "screenshot"}
                        and isinstance(full, Mapping)
                        and isinstance(full.get("url"), str)
                        and self._explicit_https_urls(full["url"]) == [full["url"]]
                        and self._browser_operation_kind(full) == operation
                        and self._materialized_tool_action_matches_schema(
                            structural_capability_schemas(),
                            {
                                "toolId": "browser.automation",
                                "action": "task",
                                "arguments": full,
                            },
                        )
                    ):
                        full_training = self._train_action_argument_head(
                            [{
                                "utterance": utterance,
                                "toolId": "browser.automation",
                                "action": "task",
                                "arguments": {
                                    "url": full["url"],
                                    "steps": list(full.get("steps", [])),
                                },
                            }],
                            grounded=True,
                        )
                        if full_training.get("ready"):
                            full_index = argument_head.register_route(
                                "browser.operation", "full:" + operation
                            )
                            argument_head.grounded_route_updates[full_index].add_(1)
                        argument_training["fullArguments"] = full_training
            if event_key is not None:
                head.experience_event_keys = torch.cat(
                    (head.experience_event_keys, event_key[None]), dim=0
                )[-256:]
        except BaseException:
            head.load_state_dict(snapshot)
            if argument_head is not None and argument_snapshot is not None:
                argument_head.load_state_dict(argument_snapshot)
            self.counters["training_steps"] = before_steps
            raise
        return {
            **result,
            "processed": True, "applied": True, "duplicate": False,
            "argumentTraining": argument_training,
        }


    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _snapshot_checksum(
        core_path: Path,
        plasticity_path: Path,
        substrate_content_sha256: Any,
        mutable_state_content_sha256: Any,
    ) -> str:
        """Hash snapshot files and metadata in their original byte order."""

        digest = hashlib.sha256()
        for path in (core_path, plasticity_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("snapshot checksum input must be a regular file")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(str(substrate_content_sha256).encode("ascii"))
        digest.update(str(mutable_state_content_sha256).encode("ascii"))
        return digest.hexdigest()




    @staticmethod
    def _public_capability_origin_receipt_valid(value: Any) -> bool:
        """Authenticate initial training without claiming tool competence."""

        return bool(
            isinstance(value, Mapping)
            and value.get("phase") == "initial-neural-tool-curriculum"
            and value.get("curriculumVersion") == 3
            and value.get("toolTrajectoriesTrained")
            == len(GROUND_UP_TOOL_TRAJECTORIES)
            and value.get("negativeExamplesTrained")
            == len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
            and value.get("capabilityOutcomeVerified") is False
            and value.get("readinessProbeExecuted") is False
            and value.get("readinessProbesAreOptimizerInputs") is False
            and value.get("systemPrompt") is False
            and value.get("toolDescriptionProse") is False
            and value.get("rewardModel") is False
            and value.get("rlhf") is False
        )

    @staticmethod
    def _public_capability_readiness_ready(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        probe = value.get("probe")
        records = probe.get("records") if isinstance(probe, Mapping) else None
        if not isinstance(records, list):
            return False
        count = probe.get("probeCount")
        correct = probe.get("correct")
        all_trajectories = probe.get("allToolTrajectories")
        trajectory_records = (
            all_trajectories.get("records")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        negative_records = (
            all_trajectories.get("negativeRecords")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        route_count = (
            all_trajectories.get("routeCount")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        correct_routes = (
            all_trajectories.get("correctRoutes")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        negative_count = (
            all_trajectories.get("negativeCount")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        negative_correct = (
            all_trajectories.get("negativeNoActionCount")
            if isinstance(all_trajectories, Mapping)
            else None
        )
        return bool(
            value.get("systemPrompt") is False
            and value.get("toolDescriptionProse") is False
            and value.get("rewardModel") is False
            and value.get("rlhf") is False
            and probe.get("passed") is True
            and probe.get("structuralSchemasOnly") is True
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and isinstance(correct, int)
            and not isinstance(correct, bool)
            and correct == count == len(records)
            and all(
                isinstance(record, Mapping)
                and record.get("correct") is True
                and record.get("schemaValidArguments") is True
                for record in records
            )
            and isinstance(all_trajectories, Mapping)
            and all_trajectories.get("passed") is True
            and all_trajectories.get("allLearnedRoutesValidated") is True
            and all_trajectories.get("executedActions") is False
            and all_trajectories.get("systemPrompt") is False
            and all_trajectories.get("toolDescriptionProse") is False
            and isinstance(trajectory_records, list)
            and isinstance(route_count, int)
            and not isinstance(route_count, bool)
            and route_count > 0
            and isinstance(correct_routes, int)
            and not isinstance(correct_routes, bool)
            and correct_routes == route_count == len(trajectory_records)
            and all(
                isinstance(record, Mapping)
                and record.get("correct") is True
                and record.get("schemaValidArguments") is True
                and record.get("materializationStatus")
                in {
                    "exact-materialized",
                    (
                        "learned-route-validated-materialization-"
                        "deferred-until-runtime-state"
                    ),
                }
                for record in trajectory_records
            )
            and isinstance(negative_records, list)
            and isinstance(negative_count, int)
            and not isinstance(negative_count, bool)
            and isinstance(negative_correct, int)
            and not isinstance(negative_correct, bool)
            and negative_correct
            == negative_count
            == len(negative_records)
            and all(
                isinstance(record, Mapping)
                and record.get("noAction") is True
                for record in negative_records
            )
            and isinstance(all_trajectories.get("perTool"), Mapping)
            and all(
                isinstance(record, Mapping)
                and record.get("complete") is True
                for record in all_trajectories["perTool"].values()
            )
        )

    def _verify_ground_up_action_origin(self) -> bool:
        """Authenticate action rehearsal against the immutable local origin."""

        if self.config.origin_kind != "ground-up":
            return False
        origin = self.engine_path / "origin"
        required = (
            origin / "brain.json",
            origin / "plasticity.safetensors",
            origin / "substrate" / "manifest.json",
            origin / "state" / "manifest.json",
            origin / "packed-ternary" / "manifest.json",
        )
        if not all(path.is_file() for path in required):
            return False
        if not self._ground_up_origin_is_complete(origin):
            return False
        try:
            metadata = read_json(origin / "brain.json")
            config = metadata.get("config")
            manifest = metadata.get("ground_up_training_manifest")
            expected = resolve_ground_up_curriculum_manifest(manifest)
            tool = manifest.get("toolCurriculum") if isinstance(manifest, Mapping) else None
            action = manifest.get("actionTraining") if isinstance(manifest, Mapping) else None
            public = (
                manifest.get("publicCapabilityReadiness")
                if isinstance(manifest, Mapping)
                else None
            )
            return bool(
                isinstance(config, Mapping)
                and config.get("origin_kind") == "ground-up"
                and isinstance(manifest, Mapping)
                and expected is not None
                and all(manifest.get(key) == value for key, value in expected.items())
                and (
                    int(expected.get("formatVersion", 0)) < 3
                    or validate_ground_up_v3_training_manifest(manifest)
                    == dict(manifest)
                )
                and isinstance(tool, Mapping)
                and tool.get("perActionCoverage") is True
                and isinstance(action, Mapping)
                and action.get("examples") == len(GROUND_UP_ACTION_EXAMPLES)
                and self._public_capability_origin_receipt_valid(public)
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _action_chat_tensor(self, text: str) -> torch.Tensor:
        """Encode the current turn exactly as the chat action channel sees it."""

        payload = self.tokenizer.encode(text)
        payload_budget = max(0, int(self.config.max_seq_len) - 3)
        if len(payload) > payload_budget:
            payload = payload[-payload_budget:]
        return torch.tensor(
            [[
                self.tokenizer.bos_id,
                self.tokenizer.human_id,
                *payload,
                self.tokenizer.brain_id,
            ]],
            dtype=torch.long,
            device=self.device,
        )

    def _starter_action_features(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize current neural features in one mask-correct batch.

        The rows have different chat-boundary lengths. Right padding is hidden
        from the bidirectional global workspace and expert pooling, while the
        causal blocks cannot see padding after each row's final valid token.
        Consequently every gathered feature is the same current neural route
        as an individual public-chat forward, without sixteen serial passes.
        """

        self.decoder.eval()
        assembly_features: List[torch.Tensor] = []
        token_rows: List[torch.Tensor] = []
        targets: List[int] = []
        with torch.no_grad():
            for text, kind in GROUND_UP_ACTION_EXAMPLES:
                # This is the real public chat boundary, not a bare-text or
                # synthetic instruction representation: bos, human, text,
                # brain. The same helper is used by runtime action selection.
                ids = self._action_chat_tensor(text)
                token_rows.append(ids[0])
                assembly_feature = self._idea_model_vector(
                    self.memory.vector_for_text(text)
                )
                targets.append(ACTION_KINDS.index(kind))
                # Idle cognition selects from active internal assemblies rather
                # than token-decoder hidden states. Train the same typed head on
                # that neural channel so spontaneous actions do not need a
                # synthetic instruction or hidden prompt.
                assembly_features.append(assembly_feature.detach())

            lengths = torch.tensor(
                [int(row.numel()) for row in token_rows],
                dtype=torch.long,
                device=self.device,
            )
            maximum = int(lengths.max().item())
            input_ids = torch.full(
                (len(token_rows), maximum),
                int(self.tokenizer.pad_id),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros(
                (len(token_rows), maximum),
                dtype=torch.bool,
                device=self.device,
            )
            for row_index, row in enumerate(token_rows):
                size = int(row.numel())
                input_ids[row_index, :size] = row
                attention_mask[row_index, :size] = True
            assembly_batch = torch.cat(assembly_features, dim=0)
            routed = self.decoder(
                input_ids,
                memory_bias=self.idea_adapter(assembly_batch),
                use_global_workspace=True,
                attention_mask=attention_mask,
            )
            hidden = routed["hidden"][
                torch.arange(len(token_rows), device=self.device),
                lengths - 1,
            ]
            # The language action channel binds the actual chat-framed decoder
            # state to its current neural idea. Runtime uses the identical
            # fusion, with capability embeddings supplied as a bounded memory
            # bias rather than behavioral prose.
            language_batch = hidden + 0.5 * assembly_batch
        return (
            language_batch.detach(),
            assembly_batch.detach(),
            torch.tensor(targets, dtype=torch.long, device=self.device),
        )

    @staticmethod
    def _action_calibration_reading(
        language_logits: torch.Tensor,
        internal_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        target_column = targets.unsqueeze(-1)
        language_probabilities = F.softmax(
            language_logits.detach().float(), dim=-1
        )
        internal_probabilities = F.softmax(
            internal_logits.detach().float(), dim=-1
        )
        deployed_probabilities = F.softmax(
            0.35 * language_logits.detach().float()
            + 0.65 * internal_logits.detach().float(),
            dim=-1,
        )
        language_targets = language_probabilities.gather(
            1, target_column
        ).squeeze(-1)
        internal_targets = internal_probabilities.gather(
            1, target_column
        ).squeeze(-1)
        deployed_targets = deployed_probabilities.gather(
            1, target_column
        ).squeeze(-1)
        required = torch.tensor(
            [
                max(
                    NATIVE_ACTION_TARGET_CONFIDENCE,
                    ACTION_KIND_EMISSION_CONFIDENCE[ACTION_KINDS[int(index)]]
                    + 0.02,
                )
                for index in targets.detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=language_targets.device,
        )
        return {
            "languageAccuracy": float(
                language_probabilities.argmax(dim=-1)
                .eq(targets)
                .float()
                .mean()
                .item()
            ),
            "internalAccuracy": float(
                internal_probabilities.argmax(dim=-1)
                .eq(targets)
                .float()
                .mean()
                .item()
            ),
            "deployedAccuracy": float(
                deployed_probabilities.argmax(dim=-1)
                .eq(targets)
                .float()
                .mean()
                .item()
            ),
            "minimumLanguageTargetConfidence": float(
                language_targets.min().item()
            ),
            "minimumInternalTargetConfidence": float(
                internal_targets.min().item()
            ),
            "minimumDeployedTargetConfidence": float(
                deployed_targets.min().item()
            ),
            "minimumLanguageThresholdMargin": float(
                (language_targets - required).min().item()
            ),
            "minimumInternalThresholdMargin": float(
                (internal_targets - required).min().item()
            ),
            "minimumDeployedThresholdMargin": float(
                (deployed_targets - required).min().item()
            ),
        }

    def _calibrate_starter_action_policy(
        self,
        *,
        max_steps: int,
        minimum_steps: int = 0,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Rehearse typed trajectories after shared representations learn.

        Only the two neural action heads are optimized. The decoder and neural
        substrate still supply the features, so this is latent replay rather
        than a phrase/regular-expression command path.
        """

        parameters = [
            *self.decoder.action_policy.parameters(),
            *self.decoder.internal_action_policy.parameters(),
        ]
        parameter_ids = {id(parameter) for parameter in parameters}
        self._sync_stability_state()
        action_parameter_names = {
            name
            for name, parameter in self._named_slow_parameters().items()
            if id(parameter) in parameter_ids
        }
        head_snapshot = {
            "language": {
                key: value.detach().clone()
                for key, value in self.decoder.action_policy.state_dict().items()
            },
            "internal": {
                key: value.detach().clone()
                for key, value in self.decoder.internal_action_policy.state_dict().items()
            },
        }
        stability_snapshot = {
            name: (
                self.slow_anchors[name].clone(),
                self.slow_importance[name].clone(),
            )
            for name in action_parameter_names
        }
        counter_snapshot = {
            key: int(self.counters[key])
            for key in ("training_steps", "metaplastic_updates")
        }
        initial_loss = 0.0
        final_loss = 0.0
        completed = 0
        readings: Dict[str, float] = {}
        calibrated = False
        language_head_was_training = self.decoder.action_policy.training
        internal_head_was_training = self.decoder.internal_action_policy.training
        try:
            language_batch, assembly_batch, target_batch = (
                self._starter_action_features()
            )
            # Feature extraction puts the decoder in eval mode. Packed
            # synapses update directly during backward only while their head
            # is training; without restoring these modes, this loop silently
            # trains only the remaining floating normalization scales.
            self.decoder.action_policy.train()
            self.decoder.internal_action_policy.train()
            if self._starter_action_language_cache is None:
                self._starter_action_language_cache = (
                    language_batch.detach().cpu().clone()
                )
                self._starter_action_internal_cache = (
                    assembly_batch.detach().cpu().clone()
                )
                self._starter_action_target_cache = (
                    target_batch.detach().cpu().clone()
                )
            optimizer = adamw_for_remaining_parameters(
                parameters, lr=0.02, weight_decay=1e-5
            )
            with torch.no_grad():
                initial_language_logits = self.decoder.action_policy(
                    language_batch
                )
                initial_internal_logits = self.decoder.internal_action_policy(
                    assembly_batch
                )
                initial_loss = float(
                    (
                        F.cross_entropy(initial_language_logits, target_batch)
                        + F.cross_entropy(initial_internal_logits, target_batch)
                    ).item()
                )
                readings = self._action_calibration_reading(
                    initial_language_logits,
                    initial_internal_logits,
                    target_batch,
                )
            final_loss = initial_loss
            calibrated = (
                readings["minimumLanguageThresholdMargin"] > 0.0
                and readings["minimumInternalThresholdMargin"] > 0.0
                and readings["minimumDeployedThresholdMargin"] > 0.0
            )
            for step in range(max(0, int(max_steps))):
                if calibrated and completed >= max(0, int(minimum_steps)):
                    break
                optimizer.zero_grad(set_to_none=True)
                language_logits = self.decoder.action_policy(language_batch)
                internal_logits = self.decoder.internal_action_policy(
                    assembly_batch
                )
                loss = F.cross_entropy(
                    language_logits, target_batch
                ) + F.cross_entropy(internal_logits, target_batch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "non-finite native action loss"
                    )
                loss.backward()
                self._accumulate_slow_importance(parameters)
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.grad_clip
                )
                optimizer.step()
                completed = step + 1
                final_loss = float(loss.detach().item())
                with torch.no_grad():
                    readings = self._action_calibration_reading(
                        self.decoder.action_policy(language_batch),
                        self.decoder.internal_action_policy(assembly_batch),
                        target_batch,
                    )
                    calibrated = (
                        readings["minimumLanguageThresholdMargin"] > 0.0
                        and readings["minimumInternalThresholdMargin"] > 0.0
                        and readings["minimumDeployedThresholdMargin"] > 0.0
                    )
                self.counters["training_steps"] += 1
            if not calibrated and strict:
                raise RuntimeError(
                    "native action policy did not retain its neural "
                    "confidence margin "
                    "(language=%.4f, internal=%.4f, deployed=%.4f, "
                    "steps=%d)"
                    % (
                        readings.get("minimumLanguageThresholdMargin", float("nan")),
                        readings.get("minimumInternalThresholdMargin", float("nan")),
                        readings.get("minimumDeployedThresholdMargin", float("nan")),
                        completed,
                    )
                )
            if completed:
                self._commit_slow_anchors(
                    rate=1.0,
                    parameters=parameters,
                )
        except Exception as error:
            self.decoder.action_policy.load_state_dict(
                head_snapshot["language"]
            )
            self.decoder.internal_action_policy.load_state_dict(
                head_snapshot["internal"]
            )
            for name, (anchor, importance) in stability_snapshot.items():
                self.slow_anchors[name] = anchor
                self.slow_importance[name] = importance
            self.counters.update(counter_snapshot)
            if strict:
                raise
            return {
                "examples": len(GROUND_UP_ACTION_EXAMPLES),
                "trainingVectors": len(GROUND_UP_ACTION_EXAMPLES) * 2,
                "neuralChannels": [
                    "language-decoder",
                    "internal-assembly",
                ],
                "languageChatFraming": [
                    "bos",
                    "human",
                    "text",
                    "brain",
                ],
                "steps": 0,
                "attemptedSteps": completed,
                "featurePasses": 1,
                "featureVectors": len(GROUND_UP_ACTION_EXAMPLES),
                "featureBatching": "right-padded-mask-correct",
                "applied": False,
                "initialLoss": initial_loss,
                "finalLoss": final_loss,
                **readings,
                "proposalConfidenceThreshold": ACTION_PROPOSAL_CONFIDENCE,
                "requiredTargetConfidence": NATIVE_ACTION_TARGET_CONFIDENCE,
                "perKindEmissionConfidence": (
                    ACTION_KIND_EMISSION_CONFIDENCE
                ),
                "deployedBlend": {
                    "language": 0.35,
                    "internal": 0.65,
                },
                "minimumConfidenceMargin": None,
                "calibrated": False,
                "rolledBack": True,
                "failureType": type(error).__name__,
                "failure": str(error)[:240],
                "objective": "structured action trajectory imitation",
                "preferenceLabels": False,
                "rewardModel": False,
            }
        finally:
            self.decoder.action_policy.train(language_head_was_training)
            self.decoder.internal_action_policy.train(internal_head_was_training)
        accuracy = 0.5 * (
            readings["languageAccuracy"] + readings["internalAccuracy"]
        )
        return {
            "examples": len(GROUND_UP_ACTION_EXAMPLES),
            "trainingVectors": int(target_batch.numel()) * 2,
            "neuralChannels": ["language-decoder", "internal-assembly"],
            "languageChatFraming": ["bos", "human", "text", "brain"],
            "steps": completed,
            "attemptedSteps": completed,
            "featurePasses": 1,
            "featureVectors": len(GROUND_UP_ACTION_EXAMPLES),
            "featureBatching": "right-padded-mask-correct",
            "applied": completed > 0,
            "initialLoss": initial_loss,
            "finalLoss": final_loss,
            "accuracy": accuracy,
            **readings,
            "proposalConfidenceThreshold": ACTION_PROPOSAL_CONFIDENCE,
            "requiredTargetConfidence": NATIVE_ACTION_TARGET_CONFIDENCE,
            "perKindEmissionConfidence": ACTION_KIND_EMISSION_CONFIDENCE,
            "deployedBlend": {"language": 0.35, "internal": 0.65},
            "minimumConfidenceMargin": min(
                readings["minimumLanguageThresholdMargin"],
                readings["minimumInternalThresholdMargin"],
                readings["minimumDeployedThresholdMargin"],
            ),
            "calibrated": calibrated,
            "rolledBack": False,
            "objective": "structured action trajectory imitation",
            "preferenceLabels": False,
            "rewardModel": False,
        }

    def _can_retain_native_action_policy(self) -> bool:
        """Retain learned routes only for an authenticated native origin."""

        return bool(
            self.config.origin_kind == "ground-up"
            and self._ground_up_action_origin_verified
            and isinstance(self.ground_up_training_manifest, Mapping)
            and self._starter_action_language_cache is not None
            and self._starter_action_internal_cache is not None
            and self._starter_action_target_cache is not None
        )

    def _retain_native_action_policy(
        self,
        *,
        pre_language_logits: torch.Tensor,
        pre_internal_logits: torch.Tensor,
        pre_action_emitted: bool,
        post_language_feature: torch.Tensor,
        post_internal_feature: torch.Tensor,
        exact_route_decoder_forwards: int = 0,
        max_steps: int = 96,
    ) -> Optional[Dict[str, Any]]:
        """Distill one exact routed decision after a slow representation update.

        The caller performs one post-update forward through the exact runtime
        route. This method performs one additional, mask-correct decoder batch
        over all native trajectories using the *current* decoder, memory
        bridge, global workspace, expert router, and action inputs. Origin
        caches authenticate eligibility but never stand in for current neural
        features.
        """

        if not self._can_retain_native_action_policy():
            return None
        assert self._starter_action_language_cache is not None
        assert self._starter_action_internal_cache is not None
        assert self._starter_action_target_cache is not None
        parameters = [
            *self.decoder.action_policy.parameters(),
            *self.decoder.internal_action_policy.parameters(),
        ]
        parameter_ids = {id(parameter) for parameter in parameters}
        self._sync_stability_state()
        action_names = {
            name
            for name, parameter in self._named_slow_parameters().items()
            if id(parameter) in parameter_ids
        }
        head_snapshot = {
            "language": {
                key: value.detach().clone()
                for key, value in self.decoder.action_policy.state_dict().items()
            },
            "internal": {
                key: value.detach().clone()
                for key, value in self.decoder.internal_action_policy.state_dict().items()
            },
        }
        stability_snapshot = {
            name: (
                self.slow_anchors[name].clone(),
                self.slow_importance[name].clone(),
            )
            for name in action_names
        }
        counter_snapshot = {
            key: int(self.counters[key])
            for key in ("training_steps", "metaplastic_updates")
        }
        pre_language = pre_language_logits.detach().to(self.device)
        pre_internal = pre_internal_logits.detach().to(self.device)
        pre_deployed = F.softmax(
            0.35 * pre_language.float() + 0.65 * pre_internal.float(),
            dim=-1,
        )
        target = pre_deployed.argmax(dim=-1)
        target_index = int(target.item())
        target_kind = ACTION_KINDS[target_index]
        pre_confidence = float(pre_deployed[0, target_index].item())
        emission_threshold = ACTION_KIND_EMISSION_CONFIDENCE[target_kind]
        required_confidence = (
            max(emission_threshold + 0.02, min(pre_confidence, 0.90))
            if pre_action_emitted
            else max(0.0, pre_confidence - 0.02)
        )
        language_head_was_training = self.decoder.action_policy.training
        internal_head_was_training = self.decoder.internal_action_policy.training
        canonical_language, canonical_internal, canonical_targets = (
            self._starter_action_features()
        )
        self.decoder.action_policy.train()
        self.decoder.internal_action_policy.train()
        post_language = post_language_feature.detach().to(self.device)
        post_internal = post_internal_feature.detach().to(self.device)
        optimizer = adamw_for_remaining_parameters(
            parameters,
            lr=0.02,
            weight_decay=1e-5,
        )
        completed = 0
        actual_confidence = 0.0
        actual_kind = "talk"
        canonical_readings: Dict[str, float] = {}

        def reading() -> Tuple[bool, bool]:
            nonlocal actual_confidence, actual_kind, canonical_readings
            with torch.no_grad():
                canonical_language_logits = self.decoder.action_policy(
                    canonical_language
                )
                canonical_internal_logits = (
                    self.decoder.internal_action_policy(canonical_internal)
                )
                canonical_readings = self._action_calibration_reading(
                    canonical_language_logits,
                    canonical_internal_logits,
                    canonical_targets,
                )
                actual_language_logits = self.decoder.action_policy(
                    post_language
                )
                actual_internal_logits = self.decoder.internal_action_policy(
                    post_internal
                )
                actual_probabilities = F.softmax(
                    0.35 * actual_language_logits.float()
                    + 0.65 * actual_internal_logits.float(),
                    dim=-1,
                )
                actual_index = int(actual_probabilities.argmax(dim=-1).item())
                actual_kind = ACTION_KINDS[actual_index]
                actual_confidence = float(
                    actual_probabilities[0, target_index].item()
                )
                exact_ready = (
                    actual_index == target_index
                    and actual_confidence >= required_confidence
                )
                if (
                    not pre_action_emitted
                    and emission_threshold > 0.0
                    and pre_confidence < emission_threshold
                ):
                    exact_ready = (
                        exact_ready
                        and actual_confidence < emission_threshold
                    )
                canonical_ready = (
                    canonical_readings["minimumLanguageThresholdMargin"] > 0.0
                    and canonical_readings["minimumInternalThresholdMargin"] > 0.0
                    and canonical_readings["minimumDeployedThresholdMargin"] > 0.0
                )
                return exact_ready, canonical_ready

        try:
            exact_ready, canonical_ready = reading()
            for step in range(max(0, min(96, int(max_steps)))):
                if exact_ready and canonical_ready:
                    break
                optimizer.zero_grad(set_to_none=True)
                language_features = torch.cat(
                    (canonical_language, post_language), dim=0
                )
                internal_features = torch.cat(
                    (canonical_internal, post_internal), dim=0
                )
                language_logits = self.decoder.action_policy(
                    language_features
                )
                internal_logits = self.decoder.internal_action_policy(
                    internal_features
                )
                canonical_count = int(canonical_targets.numel())
                canonical_loss = F.cross_entropy(
                    language_logits[:canonical_count], canonical_targets
                ) + F.cross_entropy(
                    internal_logits[:canonical_count], canonical_targets
                )
                actual_language_logits = language_logits[-1:]
                actual_internal_logits = internal_logits[-1:]
                actual_logits = (
                    0.35 * actual_language_logits
                    + 0.65 * actual_internal_logits
                )
                hard_loss = (
                    F.cross_entropy(actual_logits, target)
                    if pre_action_emitted
                    else torch.zeros((), device=self.device)
                )
                distillation_loss = F.kl_div(
                    F.log_softmax(actual_language_logits.float(), dim=-1),
                    F.softmax(pre_language.float(), dim=-1),
                    reduction="batchmean",
                ) + F.kl_div(
                    F.log_softmax(actual_internal_logits.float(), dim=-1),
                    F.softmax(pre_internal.float(), dim=-1),
                    reduction="batchmean",
                )
                loss = (
                    0.25 * canonical_loss
                    + hard_loss
                    + (0.20 if pre_action_emitted else 1.0)
                    * distillation_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite exact-route retention loss")
                loss.backward()
                self._accumulate_slow_importance(parameters)
                torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
                optimizer.step()
                self.counters["training_steps"] += 1
                completed = step + 1
                exact_ready, canonical_ready = reading()
            if not exact_ready or not canonical_ready:
                raise RuntimeError(
                    "exact-route action retention did not recover its margin"
                )
            if completed:
                self._commit_slow_anchors(rate=1.0, parameters=parameters)
            cleared_main_optimizer_states = 0
            if completed:
                for parameter in parameters:
                    if parameter in self._optimizer.state:
                        self._optimizer.state.pop(parameter)
                        cleared_main_optimizer_states += 1
            result: Dict[str, Any] = {
                "mode": "exact-route-self-distillation+current-neural-replay",
                "calibrated": True,
                "rolledBack": False,
                "steps": completed,
                "attemptedSteps": completed,
                "applied": completed > 0,
                "exactRouteDecoderForwards": max(
                    0, int(exact_route_decoder_forwards)
                ),
                "canonicalDecoderForwards": 1,
                "canonicalFeatureVectors": int(canonical_targets.numel()),
                "actualPreKind": target_kind,
                "actualPreConfidence": pre_confidence,
                "actualPreActionEmitted": pre_action_emitted,
                "actualPostKind": actual_kind,
                "actualPostConfidence": actual_confidence,
                "actualRequiredConfidence": required_confidence,
                "actualRoutePreserved": True,
                "canonicalReplayReady": canonical_ready,
                "canonicalReplayMetrics": canonical_readings,
                "mainOptimizerHeadStatesCleared": (
                    cleared_main_optimizer_states
                ),
                "syntheticDeployedGuarantee": False,
            }
        except Exception as error:
            self.decoder.action_policy.load_state_dict(head_snapshot["language"])
            self.decoder.internal_action_policy.load_state_dict(
                head_snapshot["internal"]
            )
            for name, (anchor, importance) in stability_snapshot.items():
                self.slow_anchors[name] = anchor
                self.slow_importance[name] = importance
            self.counters.update(counter_snapshot)
            result = {
                "mode": "exact-route-self-distillation+current-neural-replay",
                "calibrated": False,
                "rolledBack": True,
                "steps": 0,
                "attemptedSteps": completed,
                "applied": False,
                "exactRouteDecoderForwards": max(
                    0, int(exact_route_decoder_forwards)
                ),
                "canonicalDecoderForwards": 1,
                "canonicalFeatureVectors": int(canonical_targets.numel()),
                "actualPreKind": target_kind,
                "actualPreConfidence": pre_confidence,
                "actualPreActionEmitted": pre_action_emitted,
                "actualPostKind": actual_kind,
                "actualPostConfidence": actual_confidence,
                "actualRequiredConfidence": required_confidence,
                "actualRoutePreserved": False,
                "canonicalReplayReady": False,
                "syntheticDeployedGuarantee": False,
                "failureType": type(error).__name__,
                "failure": str(error)[:240],
            }
        self.decoder.action_policy.train(language_head_was_training)
        self.decoder.internal_action_policy.train(internal_head_was_training)
        self.counters["action_retention_checks"] += 1
        if result["calibrated"]:
            self.counters["action_retention_replays"] += int(result["steps"])
        else:
            self.counters["action_retention_failures"] += 1
        return result

    def _train_starter_action_policy(self) -> Dict[str, Any]:
        """Imitate typed action trajectories without a reward/preference model."""

        return self._calibrate_starter_action_policy(
            # The capability curriculum intentionally shifts the shared
            # language representation before this head is calibrated. The
            # Teach the neural action head from the declared examples. Do not
            # quiz it into a perfect route margin before its first user turn.
            max_steps=32,
            minimum_steps=16,
            strict=False,
        )




    def _clear_ground_up_transient_state(self) -> Dict[str, Any]:
        """Remove Build-time scratch while preserving learned neural state."""

        before = {
            "replayEntries": len(self.replay),
            "workingMemoryVectors": len(self.working_memory),
            "pagedWorkingMemory": self.paged_working_memory.count(),
            "recentTokens": len(self.recent_token_context),
            "lifecycleScratch": len(self.memory_lifecycle.afterimage_items),
            "lifecycleFocus": len(self.memory_lifecycle.active_focus),
        }
        self.replay.truncate(0)
        self.paged_working_memory.clear()
        self.working_memory = []
        self.workspace_items = []
        self.recent_token_context = []
        self.current_context = {
            "tokenCount": 0,
            "tokenHash": "",
            "sensorySlots": 0,
            "updatedAt": _iso_now(),
        }
        self.fresh_attention_boundary = None
        self._fresh_attention_paged_clear_pending = False
        self.memory_lifecycle.clear_attention()
        self.router.reset_activity()
        self.liquid_state.zero_()
        return {
            **{key + "Before": value for key, value in before.items()},
            "replayEntriesAfter": len(self.replay),
            "workingMemoryVectorsAfter": len(self.working_memory),
            "pagedWorkingMemoryAfter": self.paged_working_memory.count(),
            "recentTokensAfter": len(self.recent_token_context),
            "lifecycleScratchAfter": len(
                self.memory_lifecycle.afterimage_items
            ),
            "lifecycleFocusAfter": len(self.memory_lifecycle.active_focus),
            "currentContextTokensAfter": int(
                self.current_context.get("tokenCount", 0)
            ),
            "currentContextSensorySlotsAfter": int(
                self.current_context.get("sensorySlots", 0)
            ),
            "freshAttentionBoundaryAfter": None,
            "liquidStateAbsoluteSumAfter": float(
                self.liquid_state.detach().abs().sum().item()
            ),
            "routerMembraneAbsoluteSumAfter": float(
                self.router.population.membrane.detach().abs().sum().item()
            ),
            "routerPreTraceAbsoluteSumAfter": float(
                self.router.synapses.pre_trace.detach().abs().sum().item()
            ),
            "routerPostTraceAbsoluteSumAfter": float(
                self.router.synapses.post_trace.detach().abs().sum().item()
            ),
            "complete": True,
            "learnedParametersPreserved": True,
            "learnedSubstratePreserved": True,
        }

    def _train_ground_up_curriculum(
        self,
        progress: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train the native random core on the transparent local curriculum.

        Construction has no committed checkpoint before this method returns.
        Any exception therefore aborts the entire build; only the subsequently
        snapshotted origin can become durable/discoverable.
        """

        curriculum = current_ground_up_curriculum_manifest()
        initial_checksum = self.parameter_checksum()
        initial_accounting = self.parameter_accounting()
        modality_checksum_before = self._modality_parameter_checksum()
        selector_checksum_before = self._imagination_selector_checksum()
        parameter_checksums_before = {
            name: tensor_checksum([parameter])
            for name, parameter in self._named_native_core_tensors().items()
        }
        substrate_before = {
            "neurons": len(self.memory.neurons),
            "assemblies": len(self.memory.assemblies),
            "synapses": len(self.memory.synapses),
        }
        if progress is not None:
            progress(
                "capability-curriculum",
                0.18,
                "Training the random native core on local tool/action examples",
                self._build_progress_metrics(),
            )
        tool_curriculum = self._train_starter_tool_curriculum(
            source="built-in-tool-curriculum",
            source_label=str(curriculum["id"]),
            kind_prefix="ground-up",
            trajectories=GROUND_UP_TOOL_TRAJECTORIES,
            negative_examples=GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
            files_tool_id="system.files",
        )
        self._starter_records_visited = int(tool_curriculum["recordsVisited"])
        action_training = self._train_starter_action_policy()
        self._starter_records_visited += len(GROUND_UP_ACTION_EXAMPLES)
        if progress is not None:
            progress(
                "action-policy",
                0.66,
                "Training native action routes from the same local examples",
                self._build_progress_metrics(),
            )
        # Training coverage is recorded; capability remains unverified until
        # the user's actual interactions. No build-time tool quiz or hidden
        # behavioral prompt is part of the first conversation.
        public_capability_readiness = {
            "phase": "initial-neural-tool-curriculum",
            "curriculumVersion": 3,
            "toolTrajectoriesTrained": len(GROUND_UP_TOOL_TRAJECTORIES),
            "negativeExamplesTrained": len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "capabilityOutcomeVerified": False,
            "readinessProbeExecuted": False,
            "readinessProbesAreOptimizerInputs": False,
            "systemPrompt": False,
            "toolDescriptionProse": False,
            "rewardModel": False,
            "rlhf": False,
        }
        transient_reset = self._clear_ground_up_transient_state()
        parameter_checksums_after = {
            name: tensor_checksum([parameter])
            for name, parameter in self._named_native_core_tensors().items()
        }
        changed_tensors = sorted(
            name
            for name, checksum in parameter_checksums_after.items()
            if parameter_checksums_before.get(name) != checksum
        )
        trained_checksum = self.parameter_checksum()
        modality_checksum_after = self._modality_parameter_checksum()
        selector_checksum_after = self._imagination_selector_checksum()
        expected_records = GROUND_UP_V3_SOURCE_RECORDS
        if self._starter_records_visited != expected_records:
            raise RuntimeError("local curriculum record coverage is incomplete")
        if not changed_tensors or trained_checksum == initial_checksum:
            raise RuntimeError("local curriculum did not update native core weights")
        if (
            modality_checksum_after != modality_checksum_before
            or selector_checksum_after != selector_checksum_before
            or any(name.startswith("modalities.") for name in changed_tensors)
        ):
            raise RuntimeError(
                "ground-up tool/action curriculum changed modality parameters"
            )
        substrate_after = {
            "neurons": len(self.memory.neurons),
            "assemblies": len(self.memory.assemblies),
            "synapses": len(self.memory.synapses),
        }
        if substrate_after == substrate_before:
            raise RuntimeError("local curriculum did not update the neural substrate")
        receipt = seal_ground_up_v3_training_receipt({
            **current_ground_up_training_receipt_contract(),
            "recordsVisited": self._starter_records_visited,
            "recordGroupsVisited": {
                "actionExamples": len(GROUND_UP_ACTION_EXAMPLES),
                "toolTrajectories": len(GROUND_UP_TOOL_TRAJECTORIES),
                "negativeToolExamples": len(
                    GROUND_UP_TOOL_NEGATIVE_EXAMPLES
                ),
                "syntheticModalityFixtures": 0,
                "imaginationSelectorFixtures": 0,
            },
            "completeCoverage": True,
            "nativeCoreParameterTensors": len(parameter_checksums_after),
            "changedNativeCoreParameterTensors": len(changed_tensors),
            "changedNativeCoreTensorNames": changed_tensors,
            "parameterChecksumBefore": initial_checksum,
            "parameterChecksumAfter": trained_checksum,
            "parametersChanged": True,
            "modalityParameterChecksumBefore": modality_checksum_before,
            "modalityParameterChecksumAfter": modality_checksum_after,
            "modalityParametersChanged": False,
            "imaginationSelectorChecksumBefore": selector_checksum_before,
            "imaginationSelectorChecksumAfter": selector_checksum_after,
            "imaginationSelectorParametersChanged": False,
            "substrateBefore": substrate_before,
            "substrateAfter": substrate_after,
            "substrateChanged": True,
        })
        validate_ground_up_v3_training_receipt(receipt)
        enabled_modalities = [
            name
            for name, enabled in (
                ("vision", self.config.vision_enabled),
                ("image", self.config.image_enabled),
                ("audio", self.config.audio_enabled),
                ("video", self.config.video_enabled),
            )
            if enabled
        ]
        modality_training = {
            "source": "selected-user-data-only",
            "enabledModalities": enabled_modalities,
            "trainedModalities": [],
            "trainingRecords": 0,
            "steps": 0,
            "parametersChanged": False,
            "syntheticFixture": False,
            "initialization": "seeded-random-untrained",
        }
        manifest = seal_ground_up_v3_training_manifest({
            **curriculum,
            "originKind": "ground-up",
            "trainedAt": _iso_now(),
            "randomInitialization": {
                "algorithm": "torch-seeded-module-initialization-v1",
                "seed": int(self.config.seed),
                "parameterChecksum": initial_checksum,
                "exactParameterCount": int(
                    initial_accounting["totalNeuralParameters"]
                ),
            },
            "architectureScale": {
                "hardwareTier": self.config.hardware_tier,
                "dimensions": self.config.d_model,
                "layers": self.config.n_layers,
                "feedForward": self.config.d_ff,
                "vsaDimensions": self.config.vsa_dim,
                "routerNeurons": self.config.router_neurons,
                "denseParameterCount": int(
                    initial_accounting["mutableDenseParameters"]
                ),
                "growthCardinalityLimit": None,
                "growthBoundary": "live-resource-watermark",
                "diskStateOffload": bool(self.config.disk_state_offload),
            },
            "toolCurriculum": tool_curriculum,
            "actionTraining": action_training,
            "modalityTraining": modality_training,
            "transientStateReset": transient_reset,
            "publicCapabilityReadiness": public_capability_readiness,
            "trainingReceipt": receipt,
            "externalWeightFiles": [],
            "pretrainedTextCortex": None,
            "baseFrozen": False,
        })
        self._validate_ground_up_v3_training_manifest(manifest)
        self.events.append("ground-up-curriculum-trained", dict(receipt))
        return manifest


    @staticmethod
    def _restore_candidate_checkpoint(candidate_dir: Path) -> None:
        """Restore the last complete checkpoint captured before candidate work.

        Promotion writes three independently atomic files.  The candidate phase
        record and this backup turn those writes into a recoverable transaction:
        if a process dies between replacements, the next load restores the
        complete pre-candidate set before reading any tensors.
        """

        stable = candidate_dir / "stable"
        engine_path = candidate_dir.parent.parent
        filenames = ("brain.json", "core.safetensors", "plasticity.safetensors")
        missing = [name for name in filenames if not (stable / name).is_file()]
        if missing:
            raise RuntimeError(
                "candidate stable checkpoint is incomplete: %s"
                % ", ".join(missing)
            )
        for filename in ("core.safetensors", "plasticity.safetensors"):
            source = stable / filename
            temporary = engine_path / (filename + ".recovery.tmp")
            shutil.copy2(str(source), str(temporary))
            os.replace(str(temporary), str(engine_path / filename))
        copy_substrate_snapshot(stable, engine_path)
        copy_mutable_state_snapshot(stable, engine_path)
        source = stable / "brain.json"
        temporary = engine_path / "brain.json.recovery.tmp"
        shutil.copy2(str(source), str(temporary))
        os.replace(str(temporary), str(engine_path / "brain.json"))

    @staticmethod
    def _recover_interrupted_candidates(engine_path: Path) -> List[Dict[str, Any]]:
        """Quarantine unfinished candidates and roll back interrupted promotion."""

        recovered: List[Dict[str, Any]] = []
        candidates_root = engine_path / "candidates"
        if not candidates_root.is_dir():
            return recovered
        for candidate_dir in sorted(candidates_root.iterdir()):
            if not candidate_dir.is_dir():
                continue
            record_path = candidate_dir / "candidate.json"
            if record_path.is_file():
                record = read_json(record_path)
                previous_status = str(record.get("status", "unknown"))
            else:
                record = {
                    "id": candidate_dir.name,
                    "kind": "unknown",
                    "createdAt": _iso_now(),
                }
                previous_status = "unrecorded"
            if previous_status not in {"training", "promoting", "unrecorded"}:
                continue
            restored = False
            if previous_status == "promoting":
                AdaptiveBrain._restore_candidate_checkpoint(candidate_dir)
                restored = True
            recovered_record = {
                **record,
                "status": "interrupted",
                "previousStatus": previous_status,
                "reason": (
                    "worker stopped during promotion; stable checkpoint restored"
                    if restored
                    else "worker stopped before candidate promotion"
                ),
                "stableCheckpointRestored": restored,
                "recoveredAt": _iso_now(),
            }
            atomic_write_json(record_path, recovered_record)
            recovered.append(
                {
                    "candidateId": str(
                        recovered_record.get("id", candidate_dir.name)
                    ),
                    "kind": str(recovered_record.get("kind", "unknown")),
                    "previousStatus": previous_status,
                    "stableCheckpointRestored": restored,
                }
            )
        return recovered

    @classmethod
    def load(
        cls, storage_path: Path, expected_brain_id: Optional[str] = None
    ) -> "AdaptiveBrain":
        constructed: List["AdaptiveBrain"] = []
        try:
            return cls._load_impl(
                storage_path,
                expected_brain_id=expected_brain_id,
                constructed=constructed,
            )
        except BaseException:
            # A failed integrity or compatibility check must not leave the
            # SQLite event journal open. Windows will otherwise refuse to
            # clean up, replace, or restore the containing brain directory.
            if constructed:
                constructed[0].close()
            raise

    @staticmethod
    def _repair_nonfinite_core_tensors(
        core: Mapping[str, torch.Tensor],
        recovery_sources: Sequence[Tuple[str, Mapping[str, torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Repair only corrupt elements from verified checkpoint generations.

        Finite values in the active checkpoint are authoritative learned state
        and are never replaced.  A matching, finite value from the newest
        verified recovery generation fills each non-finite position.  If no
        trusted source can repair every position, loading fails closed rather
        than silently reinitializing or discarding a learned tensor.
        """

        repaired = dict(core)
        masks: Dict[str, torch.Tensor] = {}
        records: List[Dict[str, Any]] = []
        for name, active in core.items():
            if not active.is_floating_point():
                continue
            invalid = ~torch.isfinite(active)
            invalid_count = int(invalid.sum().item())
            if invalid_count == 0:
                continue
            value = active.detach().clone()
            unresolved = invalid.clone()
            source_counts: Dict[str, int] = {}
            for source_name, source_core in recovery_sources:
                fallback = source_core.get(name)
                if (
                    fallback is None
                    or not fallback.is_floating_point()
                    or fallback.shape != active.shape
                ):
                    continue
                usable = unresolved & torch.isfinite(fallback)
                count = int(usable.sum().item())
                if count == 0:
                    continue
                value[usable] = fallback.to(dtype=value.dtype)[usable]
                unresolved &= ~usable
                source_counts[source_name] = count
                if not bool(unresolved.any()):
                    break
            if bool(unresolved.any()):
                raise ValueError(
                    "%s contains %d non-finite checkpoint values and no "
                    "verified recovery generation can restore all of them"
                    % (name, int(unresolved.sum().item()))
                )
            # This assertion protects future edits from accidentally replacing
            # valid learned positions while broadening checkpoint recovery.
            if not torch.equal(value[~invalid], active[~invalid]):
                raise RuntimeError(
                    "checkpoint recovery changed finite learned state in %s"
                    % name
                )
            repaired[name] = value
            masks[name] = invalid.detach().cpu()
            records.append(
                {
                    "tensor": name,
                    "repairedElements": invalid_count,
                    "totalElements": int(active.numel()),
                    "sources": source_counts,
                    "finiteLearnedElementsPreserved": int(
                        active.numel() - invalid_count
                    ),
                }
            )
        return repaired, masks, records

    @classmethod
    def _load_impl(
        cls,
        storage_path: Path,
        expected_brain_id: Optional[str] = None,
        constructed: Optional[List["AdaptiveBrain"]] = None,
    ) -> "AdaptiveBrain":
        engine_path = Path(storage_path).resolve() / "engine"
        def require_native_metadata(candidate: Mapping[str, Any]) -> None:
            saved_config = candidate.get("config")
            if not isinstance(saved_config, Mapping) or (
                saved_config.get("origin_kind") != "ground-up"
                or "foundation_model_id" in saved_config
                or "foundationModelId" in saved_config
            ):
                raise ValueError(
                    "Only native ground-up OmniCortex brains can be loaded; "
                    "imported foundations and legacy origins are unsupported"
                )
            if candidate.get("neural_sequence_memory") is not None:
                raise ValueError(
                    "checkpoint contains obsolete cue-to-answer sequence memory"
                )
            if "starter_training_manifest" in candidate:
                raise ValueError("checkpoint contains an imported starter manifest")
            if candidate.get("messages") or candidate.get("traces"):
                raise ValueError("native checkpoint contains obsolete inline conversation")

        # Reject incompatible identities before recovery has an opportunity
        # to write anything into their app-managed directories.
        metadata = read_json(engine_path / "brain.json")
        require_native_metadata(metadata)
        recovered_candidates = cls._recover_interrupted_candidates(engine_path)
        metadata = read_json(engine_path / "brain.json")
        require_native_metadata(metadata)
        if int(metadata.get("schema_version", 0)) != ENGINE_SCHEMA_VERSION:
            raise ValueError("unsupported OmniCortex engine schema")
        if metadata.get("release_format") != "stable-1.0":
            raise ValueError(
                "incompatible OmniCortex beta brain; create a stable v1 brain"
            )
        brain_id = str(metadata["brain_id"])
        if expected_brain_id is not None and str(expected_brain_id) != brain_id:
            raise ValueError("brain id does not match the requested storage path")
        config = OmniConfig.from_dict(metadata["config"])
        brain = cls(brain_id, storage_path, config)
        if constructed is not None:
            constructed.append(brain)
        stored_working_pages = metadata.get("paged_working_memory")
        if isinstance(stored_working_pages, Mapping):
            brain.paged_working_memory_recovery = (
                brain.paged_working_memory.recover_checkpoint(
                    stored_working_pages
                )
            )
        for _ in range(int(metadata.get("expert_count", 0))):
            brain.decoder.grow_expert()
        brain._configure_packed_stability()
        recovered_mutable: Optional[Dict[str, Any]] = None
        mutable_pointer = metadata.get("mutable_state")
        if isinstance(mutable_pointer, Mapping):
            recovered_mutable = brain.state_store.recover(
                mutable_pointer,
                engine_path,
                brain.replay,
            )
            recovered_pointer = recovered_mutable.get("pointer")
            brain.mutable_state_manifest = dict(
                recovered_pointer
                if isinstance(recovered_pointer, Mapping)
                else mutable_pointer
            )
        core = load_tensors(engine_path / "core.safetensors", device="cpu")
        origin_verified = brain._ground_up_origin_is_complete(
            engine_path / "origin"
        )
        core_recovery_sources: List[Tuple[str, Mapping[str, torch.Tensor]]] = []
        if any(
            value.is_floating_point()
            and not bool(torch.isfinite(value).all())
            for value in core.values()
        ):
            if isinstance(mutable_pointer, Mapping):
                core_recovery_sources.extend(
                    (
                        "previous-generation:%s" % generation_id,
                        generation_core,
                    )
                    for generation_id, generation_core
                    in brain.state_store.prior_core_generations(mutable_pointer)
                )
            if origin_verified:
                core_recovery_sources.append(
                    (
                        "verified-omni-origin",
                        load_tensors(
                            engine_path / "origin" / "core.safetensors",
                            device="cpu",
                        ),
                    )
                )
        core, repaired_core_masks, core_recovery_records = (
            cls._repair_nonfinite_core_tensors(core, core_recovery_sources)
        )
        plastic = load_tensors(
            engine_path / "plasticity.safetensors", device="cpu"
        )
        _load_prefixed(brain.decoder, core, "decoder.")
        _load_prefixed(brain.memory_bridge, core, "memory_bridge.")
        _load_prefixed(brain.idea_adapter, core, "idea_adapter.")
        has_foundation_adapter = any(
            key.startswith("foundation_adapter.") for key in core
        )
        if has_foundation_adapter:
            raise ValueError("checkpoint contains an imported foundation adapter")
        _load_prefixed(brain.liquid, core, "liquid.")
        _load_prefixed(brain.modalities, core, "modalities.")
        _load_prefixed(brain.router, plastic, "router.")
        if "state.liquid" in plastic:
            brain.liquid_state = plastic["state.liquid"].to(brain.device)
        working = plastic.get("state.working_memory")
        if working is not None:
            brain.working_memory = [
                row.detach().cpu()
                for row in working[-brain.config.working_memory_slots :]
            ]
        stored_workspace = [
            dict(item)
            for item in metadata.get("workspace_items", [])
            if isinstance(item, Mapping)
        ]
        brain.workspace_items = stored_workspace[-len(brain.working_memory) :]
        while len(brain.workspace_items) < len(brain.working_memory):
            brain.workspace_items.insert(
                0,
                {
                    "id": uuid.uuid4().hex,
                    "assemblyId": "",
                    "source": "restored",
                    "salience": 0.5,
                    "rehearsals": 1,
                    "enteredAt": _iso_now(),
                    "lastActiveAt": _iso_now(),
                },
            )
        brain.memory_lifecycle = OrganicMemoryLifecycle.from_state(
            metadata.get("memory_lifecycle"),
            plastic,
        )
        brain.current_context = dict(
            metadata.get("current_context", brain.current_context)
        )
        stored_recent = metadata.get("recent_token_context", [])
        if isinstance(stored_recent, list):
            brain.recent_token_context = [
                int(token)
                for token in stored_recent[-brain.config.max_seq_len :]
                if isinstance(token, int)
                and 0 <= int(token) < brain.config.vocab_size
            ]
        substrate_metadata = metadata.get("substrate", {})
        if not (
            isinstance(substrate_metadata, dict)
            and isinstance(substrate_metadata.get("persistence"), dict)
            and substrate_metadata["persistence"].get("formatVersion") == 3
        ):
            raise ValueError(
                "native OmniCortex requires a committed packed v3 substrate"
            )
        if isinstance(substrate_metadata, dict) and isinstance(
            substrate_metadata.get("persistence"), dict
        ):
            counts = substrate_metadata["persistence"].get("counts", {})
            neuron_count = (
                counts.get("neurons", 0)
                if isinstance(counts, Mapping)
                else 0
            )
            resource_status = brain.resource_policy.status()
            ram_budget = resource_status.get("systemRamBudgetBytes")
            if not isinstance(ram_budget, int) or ram_budget <= 0:
                ram_budget = resource_status.get("availableMemoryBytes")
            vector_bytes = (
                max(0, neuron_count) * ((brain.config.vsa_dim + 3) // 4 + 8)
                if type(neuron_count) is int else 0
            )
            page_vectors = bool(
                brain.config.disk_state_offload
                and (vector_bytes > 0 or metadata.get("paged_substrate_required") is True)
                and (
                    metadata.get("paged_substrate_required") is True
                    or
                    resource_status.get("memoryPressure")
                    or (
                        isinstance(ram_budget, int)
                        and ram_budget > 0
                        and vector_bytes > max(8 * 1024 * 1024, ram_budget // 64)
                    )
                )
            )
            if page_vectors:
                with prepare_committed_paged_cache(
                    engine_path,
                    engine_path / "state" / "paged-substrate-cache",
                    resource_policy=brain.resource_policy,
                ) as prepared:
                    brain.memory = NeuralSubstrate.load_sharded(
                        engine_path / "substrate",
                        substrate_metadata,
                        growth_guard=brain._allow_substrate_growth,
                        lazy_synapses=True,
                        paged_vectors=prepared.vectors,
                        paged_neurons=prepared.neurons,
                        defer_paged_assemblies=True,
                    )
                    rebuilt = finish_verified_index_from_loaded_vectors(
                        prepared, brain.memory
                    )
                    brain._verified_paged_rebuild = rebuilt
                    brain._paged_vector_cache = {
                        "generationSha256": rebuilt.generation_sha256,
                        "vectorCount": rebuilt.neurons,
                        "assemblyCount": rebuilt.assemblies,
                    }
                    brain._paged_substrate_required = True
            else:
                brain.memory = NeuralSubstrate.load_sharded(
                    engine_path / "substrate",
                    substrate_metadata,
                    growth_guard=brain._allow_substrate_growth,
                    lazy_synapses=True,
                )
        if any(str(name).startswith("sequence_memory.") for name in plastic):
            raise ValueError("native OmniCortex cannot load cue-to-answer weights")
        replay = plastic.get("state.replay")
        if replay is not None and len(brain.replay) == 0:
            # One-way migration for early internal stable-v1 checkpoints. New
            # saves keep replay exclusively in transactional SQLite.
            brain.replay.extend(row.detach().cpu() for row in replay)
        cached_language = plastic.get("state.starter_action_language_features")
        cached_internal = plastic.get("state.starter_action_internal_features")
        cached_targets = plastic.get("state.starter_action_targets")
        if (
            cached_language is not None
            and cached_internal is not None
            and cached_targets is not None
        ):
            brain._starter_action_language_cache = cached_language.detach().cpu()
            brain._starter_action_internal_cache = cached_internal.detach().cpu()
            brain._starter_action_target_cache = cached_targets.detach().cpu()
        anchors = {
            key[len("stability.anchor.") :]: value.detach().cpu()
            for key, value in plastic.items()
            if key.startswith("stability.anchor.")
        }
        importance = {
            key[len("stability.importance.") :]: value.detach().cpu()
            for key, value in plastic.items()
            if key.startswith("stability.importance.")
        }
        if anchors:
            brain.slow_anchors = anchors
            brain.slow_importance = {
                name: importance.get(name, torch.zeros_like(value)).float()
                for name, value in anchors.items()
            }
            brain._sync_stability_state()
        brain.messages = []
        brain.traces = []
        brain.training_sources = [
            cls._source_without_inline_text(value)
            for value in metadata.get("training_sources", [])
            if isinstance(value, Mapping)
        ]
        # Stable source retention is owned by the desktop CAS. Remove inline
        # text from any early internal stable checkpoint on first load.
        for assembly in brain.memory.assemblies:
            if isinstance(assembly, dict):
                assembly.pop("source_text", None)
        brain.ingestion_checkpoints = cls._validated_ingestion_checkpoints(
            metadata.get("ingestion_checkpoints")
        )
        paging_marker = metadata.get("paged_substrate_required")
        if paging_marker is not None and type(paging_marker) is not bool:
            raise ValueError("paged substrate persistence marker is invalid")
        brain._paged_substrate_required = bool(
            paging_marker
        )
        v3_checkpoints = [
            item for item in brain.ingestion_checkpoints.values()
            if item.get("formatVersion") == 3
        ]
        committed_joint_reference = metadata.get("ingestion_joint_generation")
        if v3_checkpoints:
            if len(v3_checkpoints) != 1 or len(brain.ingestion_checkpoints) != 1:
                raise ValueError("v3 ingestion has more than one active cursor")
            if not isinstance(committed_joint_reference, Mapping):
                raise ValueError("v3 ingestion has no brain.json joint generation")
            active_v3 = v3_checkpoints[0]
            joint = recover_joint_generation(
                engine_path / "state" / "ingestion-joint",
                committed_joint_reference,
                neural_store_root=engine_path / "state",
                substrate_store_root=engine_path / "substrate",
                source_manifest_sha256=active_v3["sourceManifestSha256"],
                parser_manifest_sha256=active_v3["parserManifestSha256"],
                source_content_sha256=active_v3["contentHash"],
            )
            if joint is None or (
                joint.manifest["neuralState"]["generationId"]
                != (brain.mutable_state_manifest or {}).get("activeGeneration")
                or joint.manifest["substrateState"]["generationId"]
                != (brain.memory.persistence_manifest or {}).get("activeGeneration")
                or joint.manifest["checkpointSequence"]
                != active_v3["commitSequence"]
                or joint.manifest["cursor"]["committedRecords"]
                != active_v3["committedRecords"]
            ):
                raise ValueError("v3 cursor is not the brain.json-committed neural generation")
            brain.ingestion_joint_generation = dict(committed_joint_reference)
        elif committed_joint_reference is not None:
            raise ValueError("ingestion joint reference has no active v3 cursor")
        brain.completed_ingestions = cls._validated_completed_ingestions(
            metadata.get("completed_ingestions")
        )
        brain.completed_chat_turns = cls._validated_completed_chat_turns(
            metadata.get("completed_chat_turns")
        )
        raw_completed_chat_slow = metadata.get(
            "completed_chat_slow_learning", []
        )
        if not isinstance(raw_completed_chat_slow, list):
            raise ValueError("completed chat slow-learning records are invalid")
        brain.completed_chat_slow_learning = [
            str(value)
            for value in raw_completed_chat_slow
            if brain._sha256_identifier(value)
        ][-COMPLETED_CHAT_SLOW_LEARNING:]
        raw_pending_chat_slow = metadata.get("pending_chat_slow_learning", [])
        if not isinstance(raw_pending_chat_slow, list):
            raise ValueError("pending chat slow-learning records are invalid")
        brain.pending_chat_slow_learning = [
            dict(item)
            for item in raw_pending_chat_slow
            if isinstance(item, Mapping)
            and brain._sha256_identifier(item.get("jobId"))
            and brain._sha256_identifier(item.get("inputSha256"))
            and isinstance(item.get("humanMessageId"), str)
            and bool(item.get("humanMessageId"))
        ]
        brain.fresh_attention_boundary = (
            cls._validated_fresh_attention_boundary(
                metadata.get("fresh_attention_boundary")
            )
        )
        brain.memory.configure_attention_overlay(
            (
                int(brain.fresh_attention_boundary["epoch"])
                if brain.fresh_attention_boundary is not None
                else 0
            ),
            (
                metadata.get("attention_overlay")
                if isinstance(metadata.get("attention_overlay"), Mapping)
                else None
            ),
        )
        saved_conversation = metadata.get("conversation")
        if not isinstance(saved_conversation, Mapping):
            raise ValueError("native checkpoint is missing its committed conversation head")
        try:
            brain.conversation.truncate_after_head(
                int(saved_conversation["headSequence"]),
                str(saved_conversation["headSha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "native conversation ledger does not match the committed checkpoint"
            ) from error
        active_epoch = brain._attention_epoch()
        runtime_rows = max(32, min(1000, int(brain.config.max_seq_len)))
        brain.messages = brain.conversation.recent_payloads(
            "message", runtime_rows, attention_epoch=active_epoch
        )
        brain.traces = brain.conversation.recent_payloads(
            "trace", min(200, runtime_rows), attention_epoch=active_epoch
        )
        brain.created_at = str(metadata.get("created_at", _iso_now()))
        brain.updated_at = str(metadata.get("updated_at", brain.created_at))
        brain.counters.update(
            {key: int(value) for key, value in metadata.get("counters", {}).items()}
        )
        if brain.completed_chat_turns:
            brain.counters["inference_count"] = max(
                int(brain.counters.get("inference_count", 0)),
                max(
                    int(item["inferenceCount"])
                    for item in brain.completed_chat_turns
                ),
            )
        brain.modality_training.update(
            {
                key: int(value)
                for key, value in metadata.get("modality_training", {}).items()
                if key in brain.modality_training
            }
        )
        brain.installed_modality_packs = [
            dict(item)
            for item in metadata.get("installed_modality_packs", [])
            if isinstance(item, Mapping)
        ]
        stored_ground_up_manifest = metadata.get("ground_up_training_manifest")
        brain.ground_up_training_manifest = (
            dict(stored_ground_up_manifest)
            if isinstance(stored_ground_up_manifest, Mapping)
            else None
        )
        stored_packed_manifest = metadata.get("packed_ternary_manifest")
        brain.packed_ternary_manifest = (
            dict(stored_packed_manifest)
            if isinstance(stored_packed_manifest, Mapping)
            else None
        )
        brain.novelty_streak = int(metadata.get("novelty_streak", 0))
        brain.growth_pause = metadata.get("growth_pause")
        brain.resource_pause = metadata.get("resource_pause")
        brain.last_activity_decay = float(
            metadata.get("last_activity_decay", time.time())
        )
        brain.last_idle_cycle_at = float(metadata.get("last_idle_cycle_at", 0.0))
        brain.last_idle_visible_action_at = float(
            metadata.get("last_idle_visible_action_at", 0.0)
        )
        brain._ground_up_action_origin_verified = (
            brain._verify_ground_up_action_origin()
        )
        if brain._ground_up_action_origin_verified:
            origin_plastic = load_tensors(
                engine_path / "origin" / "plasticity.safetensors",
                device="cpu",
            )
            origin_language = origin_plastic.get(
                "state.starter_action_language_features"
            )
            origin_internal = origin_plastic.get(
                "state.starter_action_internal_features"
            )
            origin_targets = origin_plastic.get(
                "state.starter_action_targets"
            )
            if (
                origin_language is None
                or origin_internal is None
                or origin_targets is None
            ):
                brain._ground_up_action_origin_verified = False
            else:
                brain._starter_action_language_cache = (
                    origin_language.detach().cpu()
                )
                brain._starter_action_internal_cache = (
                    origin_internal.detach().cpu()
                )
                brain._starter_action_target_cache = (
                    origin_targets.detach().cpu()
                )
        brain._replace_optimizer()
        if recovered_mutable is not None:
            optimizer_state = recovered_mutable.get("optimizerState")
            if isinstance(optimizer_state, Mapping):
                try:
                    brain._load_optimizer_state(optimizer_state)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        "mutable-state optimizer checkpoint is incompatible"
                    ) from error
        optimizer_recovery = brain._repair_optimizer_for_core_recovery(
            repaired_core_masks
        )
        loaded_parameter_checksum = brain.parameter_checksum()
        for active_checkpoint in brain.ingestion_checkpoints.values():
            if active_checkpoint.get("formatVersion") == 3:
                if core_recovery_records:
                    raise ValueError(
                        "v3 cursor cannot be replayed after neural core repair"
                    )
                vector_generation, index_generation = (
                    cls._v3_committed_substrate_generations(
                        brain.memory.persistence_manifest or {}
                    )
                )
                observed = validate_checkpoint_binding_v3(
                    active_checkpoint["pagedCheckpointBinding"],
                    active_checkpoint["pagedCheckpointBindingSha256"],
                    schedule=active_checkpoint["learningSchedule"],
                    schedule_sha256_value=active_checkpoint["learningScheduleSha256"],
                    source_manifest_sha256=active_checkpoint["sourceManifestSha256"],
                    parser_manifest_sha256=active_checkpoint["parserManifestSha256"],
                    source_content_sha256=active_checkpoint["contentHash"],
                    neural_state_sha256=loaded_parameter_checksum,
                    vector_generation=vector_generation,
                    index_generation=index_generation,
                )
                if (
                    observed["cursor"] != joint.manifest["cursor"]
                    or observed["coverage"] != joint.manifest["coverage"]
                ):
                    raise ValueError(
                        "v3 cursor diverges from its committed joint generation"
                    )
                continue
            if core_recovery_records:
                # The record cursor and every finite neural value remain
                # authoritative. The repaired generation is committed below
                # and receives a fresh exact transactional binding.
                active_checkpoint["neuralStateChecksum"] = (
                    loaded_parameter_checksum
                )
                continue
            if (
                active_checkpoint.get("neuralStateChecksum")
                != loaded_parameter_checksum
            ):
                raise ValueError(
                    "ingestion checkpoint does not match committed neural state"
                )
            binding = dict(active_checkpoint["committedGeneration"])
            mutable = brain.mutable_state_manifest or {}
            substrate = brain.memory.persistence_manifest or {}
            replay_checkpoint = brain.replay.checkpoint()
            workspace_checkpoint = brain.paged_working_memory.checkpoint()
            observed_binding = {
                "format": "omni-ingestion-generation-binding",
                "formatVersion": 1,
                "mutableStateActiveGeneration": str(
                    mutable.get("activeGeneration", "")
                ),
                "mutableStateContentSha256": str(
                    mutable.get("contentSha256", "")
                ),
                "substrateContentSha256": str(
                    substrate.get("contentSha256", "")
                ),
                "replayCount": int(replay_checkpoint["count"]),
                "replayHighWaterId": int(replay_checkpoint["highWaterId"]),
                "replayContentSha256": str(
                    replay_checkpoint["contentSha256"]
                ),
                "workspacePageCount": int(workspace_checkpoint["count"]),
                "workspacePageHighWaterId": int(
                    workspace_checkpoint["highWaterId"]
                ),
                "workspacePageContentSha256": str(
                    workspace_checkpoint["contentSha256"]
                ),
                "workspacePagesLearningReadable": False,
                "neuralStateChecksum": loaded_parameter_checksum,
            }
            if binding != observed_binding:
                raise ValueError(
                    "ingestion checkpoint does not match its committed generation"
                )
        packed_path = engine_path / "packed-ternary"
        if (
            not core_recovery_records
            and (packed_path / "manifest.json").is_file()
        ):
            (
                dynamic_values,
                dynamic_synapse_count,
                expected_synapse_order_hash,
                dynamic_order_basis,
            ) = brain._dynamic_synapse_pack_state()
            dynamic_name = "substrate.dynamic_synapses.weights"
            dynamic_tensors = brain._dynamic_ternary_tensors(dynamic_values)
            expected_layout = inspect_module_ternary_layout(
                brain._ternary_export_roots(),
                dynamic_synapses=dynamic_tensors,
            )
            verified_packed = verify_ternary_shards(
                packed_path, retain_names=(dynamic_name,)
            )
            packed_layout = {
                str(entry.get("name", "")): (
                    tuple(int(value) for value in entry.get("shape", [])),
                    str(entry.get("kind", "")),
                )
                for entry in verified_packed.manifest.get("tensors", [])
            }
            # A valid earlier pack may have fewer dynamic synapses than the
            # newly committed neural generation. Regenerate it from the
            # authoritative checkpoint below instead of rejecting the brain.
            packed_layout_is_stale = packed_layout != expected_layout
            packed_metadata = verified_packed.manifest.get("metadata")
            if not isinstance(packed_metadata, Mapping):
                raise ValueError("packed ternary metadata is invalid")
            ground_up_curriculum = resolve_ground_up_curriculum_manifest(
                brain.ground_up_training_manifest
            )
            if (
                ground_up_curriculum is not None
                and int(ground_up_curriculum.get("formatVersion", 0)) >= 3
            ):
                ground_up_manifest = validate_ground_up_v3_training_manifest(
                    brain.ground_up_training_manifest
                )
                ground_up_receipt = ground_up_manifest.get("trainingReceipt")
                if (
                    not isinstance(ground_up_receipt, Mapping)
                    or packed_metadata.get("groundUpCurriculumSha256")
                    != ground_up_curriculum["sha256"]
                    or packed_metadata.get("groundUpTrainingManifestSha256")
                    != ground_up_manifest.get("contentSha256")
                    or packed_metadata.get("groundUpTrainingReceiptSha256")
                    != ground_up_receipt.get("contentSha256")
                    or packed_metadata.get("originKind") != "ground-up"
                    or packed_metadata.get("baseFrozen") is not False
                    or "foundationModelId" in packed_metadata
                ):
                    raise ValueError(
                        "packed ternary ground-up provenance is invalid"
                    )
            packed_parameter_checksum = str(
                packed_metadata.get("parameterChecksum", "")
            )
            packed_dynamic_values = verified_packed.tensors.get(dynamic_name)
            packed_dynamic_count = packed_metadata.get("dynamicSynapseCount")
            packed_dynamic_order_hash = packed_metadata.get(
                "dynamicSynapseOrderSha256"
            )
            packed_total_dynamic = packed_metadata.get(
                "totalDynamicSynapseCount"
            )
            if any(
                name.startswith("substrate.sequence_memory.")
                for name in packed_layout
            ) or any(
                str(name).startswith("neuralSequence")
                or name == "totalPackedSequenceSynapseSlots"
                for name in packed_metadata
            ):
                raise ValueError("packed checkpoint contains obsolete answer-key state")
            dynamic_pack_is_stale = (
                packed_total_dynamic != dynamic_synapse_count
                or packed_metadata.get("dynamicSynapseCountingBasis")
                != "substrate-records-v1"
                or isinstance(packed_dynamic_count, bool)
                or not isinstance(packed_dynamic_count, int)
                or packed_dynamic_count != dynamic_synapse_count
                or not isinstance(packed_dynamic_order_hash, str)
                or packed_dynamic_order_hash != expected_synapse_order_hash
                or packed_metadata.get("dynamicSynapseOrderBasis")
                != dynamic_order_basis
                or packed_metadata.get("substrateContentSha256")
                != str(
                    (brain.memory.persistence_manifest or {}).get(
                        "contentSha256", ""
                    )
                )
                or packed_dynamic_values is None
                or packed_dynamic_values.dtype != torch.int8
                or tuple(packed_dynamic_values.shape)
                != tuple(dynamic_values.shape)
                or not torch.equal(
                    packed_dynamic_values.detach().cpu(),
                    dynamic_values.detach().cpu(),
                )
            )
            if (
                packed_parameter_checksum != brain.parameter_checksum()
                or dynamic_pack_is_stale
                or packed_layout_is_stale
            ):
                brain.export_packed_ternary()
        stored_cpu_rng = plastic.get("state.rng_cpu")
        if stored_cpu_rng is not None:
            torch.set_rng_state(stored_cpu_rng.detach().cpu().to(torch.uint8))
        stored_accelerator_rng = plastic.get("state.rng_accelerator")
        if stored_accelerator_rng is not None:
            if brain.device_backend == "cuda":
                torch.cuda.set_rng_state(
                    stored_accelerator_rng.detach().cpu().to(torch.uint8),
                    brain.device,
                )
            elif brain.device_backend == "mps" and hasattr(
                torch, "mps"
            ) and hasattr(torch.mps, "set_rng_state"):
                torch.mps.set_rng_state(
                    stored_accelerator_rng.detach().cpu().to(torch.uint8)
                )
        if core_recovery_records:
            previous_generation = str(
                (brain.mutable_state_manifest or {}).get(
                    "activeGeneration", ""
                )
            )
            # Keep the newest verified generation actually used for repair as
            # the post-recovery rollback point. Retaining the corrupt active
            # generation instead would discard the only known-good fallback
            # when save performs bounded generation garbage collection.
            used_sources = {
                str(source)
                for record in core_recovery_records
                for source in dict(record.get("sources", {}))
            }
            retained_recovery_generation = next(
                (
                    source.split(":", 1)[1]
                    for source, _source_core in core_recovery_sources
                    if source in used_sources
                    and source.startswith("previous-generation:")
                ),
                None,
            )
            if retained_recovery_generation is not None:
                brain.mutable_state_manifest = (
                    brain.state_store.generation_pointer(
                        retained_recovery_generation
                    )
                )
            brain.save()
            brain.export_packed_ternary()
            recovery_event = {
                "reason": "non-finite checkpoint master state",
                "previousActiveGeneration": previous_generation,
                "recoveredActiveGeneration": str(
                    (brain.mutable_state_manifest or {}).get(
                        "activeGeneration", ""
                    )
                ),
                "tensors": core_recovery_records,
                "tensorCount": len(core_recovery_records),
                "repairedElements": sum(
                    int(record["repairedElements"])
                    for record in core_recovery_records
                ),
                "optimizer": optimizer_recovery,
                "finiteLearnedStatePreserved": True,
                "packedTernaryRegenerated": True,
                "at": _iso_now(),
            }
            brain.state_store.last_recovery = {
                "recovered": True,
                "reason": "repaired non-finite neural checkpoint elements",
                "activeGeneration": recovery_event[
                    "recoveredActiveGeneration"
                ],
                "previousActiveGeneration": previous_generation,
            }
            brain.events.append("checkpoint-tensor-recovered", recovery_event)
        for recovered in recovered_candidates:
            brain.events.append("candidate-recovered", recovered)
        return brain

    def _begin_candidate(self, kind: str) -> Tuple[str, Path]:
        candidate_id = uuid.uuid4().hex
        candidate_dir = self.engine_path / "candidates" / candidate_id
        estimated_bytes = snapshot_required_bytes(self.engine_path)
        self.resource_policy.require_disk(
            estimated_bytes,
            "%s candidate snapshot" % str(kind),
        )
        try:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            snapshot_files(self.engine_path, candidate_dir / "stable")
            atomic_write_json(
                candidate_dir / "candidate.json",
                {
                    "id": candidate_id,
                    "kind": str(kind),
                    "status": "training",
                    "createdAt": _iso_now(),
                },
            )
        except BaseException:
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            raise
        return candidate_id, candidate_dir

    @staticmethod
    def _record_candidate(candidate_dir: Path, **updates: Any) -> None:
        record_path = candidate_dir / "candidate.json"
        record = read_json(record_path) if record_path.is_file() else {}
        atomic_write_json(record_path, {**record, **updates})

    def _core_tensors(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        tensors.update(_prefixed_state(self.decoder, "decoder."))
        tensors.update(_prefixed_state(self.memory_bridge, "memory_bridge."))
        tensors.update(_prefixed_state(self.idea_adapter, "idea_adapter."))
        tensors.update(_prefixed_state(self.liquid, "liquid."))
        tensors.update(_prefixed_state(self.modalities, "modalities."))
        return tensors

    def _core_parameter_map(self) -> Dict[str, nn.Parameter]:
        roots: List[Tuple[str, nn.Module]] = [
            ("decoder.", self.decoder),
            ("memory_bridge.", self.memory_bridge),
            ("idea_adapter.", self.idea_adapter),
            ("liquid.", self.liquid),
            ("modalities.", self.modalities),
        ]
        return {
            prefix + name: parameter
            for prefix, module in roots
            for name, parameter in module.named_parameters()
        }

    def parameter_accounting(self) -> Dict[str, Any]:
        """Return the authoritative count of effective neural parameters.

        Floating trainable parameters and logical packed ternary synapses are
        counted once by object identity; sparse substrate synapses are counted
        by their authoritative records. The number of packed *bytes*, optimizer
        moments, scales, and transient activity are not extra parameters.
        """
        seen_parameters: set[int] = set()
        mutable_parameters = 0
        for module in self._trainable_modules():
            for parameter in module.parameters():
                identity = id(parameter)
                if identity in seen_parameters:
                    continue
                seen_parameters.add(identity)
                mutable_parameters += int(parameter.numel())
        packed_parameters = self._packed_logical_parameter_count(
            self._trainable_modules()
        )
        mutable_dense_parameters = mutable_parameters + packed_parameters
        substrate_dynamic_synapses = len(self.memory.synapses)
        total_parameters = mutable_dense_parameters + substrate_dynamic_synapses
        return {
            "mutableDenseParameters": mutable_dense_parameters,
            "floatingTrainableParameters": mutable_parameters,
            "packedTernaryParameters": packed_parameters,
            "substrateDynamicSparseSynapses": substrate_dynamic_synapses,
            "dynamicSparseSynapses": substrate_dynamic_synapses,
            "totalNeuralParameters": total_parameters,
            "countingRule": (
                "unique floating trainable parameters plus logical packed "
                "ternary synapses and authoritative sparse substrate synapses; "
                "excludes packing bytes, scales, optimizer state, and activity"
            ),
        }

    def _repair_optimizer_for_core_recovery(
        self, repaired_masks: Mapping[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Invalidate only optimizer moments that no longer match a master.

        Restored master elements came from an older verified checkpoint, so
        their current Adam moments cannot be trusted.  Those exact positions
        are reset while every other finite moment is preserved. Independently
        non-finite optimizer values are also cleared before the recovered
        generation is made durable.
        """

        if not repaired_masks:
            return {
                "parameters": 0,
                "momentElementsReset": 0,
                "nonFiniteElementsCleared": 0,
            }
        parameters = self._core_parameter_map()
        touched_parameters = 0
        moment_elements = 0
        nonfinite_elements = 0
        with torch.no_grad():
            for name, repaired_mask in repaired_masks.items():
                parameter = parameters.get(name)
                if parameter is None:
                    continue
                state = self._optimizer.state.get(parameter)
                if not isinstance(state, dict):
                    continue
                parameter_touched = False
                for _key, value in state.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or not value.is_floating_point()
                    ):
                        continue
                    invalid = ~torch.isfinite(value)
                    invalid_count = int(invalid.sum().item())
                    reset = invalid
                    if value.shape == parameter.shape:
                        restored = repaired_mask.to(device=value.device)
                        reset = reset | restored
                        moment_elements += int(restored.sum().item())
                    reset_count = int(reset.sum().item())
                    if reset_count:
                        value.masked_fill_(reset, 0.0)
                        parameter_touched = True
                    nonfinite_elements += invalid_count
                if parameter_touched:
                    touched_parameters += 1
        return {
            "parameters": touched_parameters,
            "momentElementsReset": moment_elements,
            "nonFiniteElementsCleared": nonfinite_elements,
        }

    def _plastic_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = _prefixed_state(self.router, "router.")
        tensors["state.liquid"] = self.liquid_state.detach()
        # Record-batch restart must continue the same stochastic optimization
        # stream as an uninterrupted run. These are generator states, not
        # source tokens, and live in the same atomic neural checkpoint.
        tensors["state.rng_cpu"] = torch.get_rng_state()
        if self.device_backend == "cuda":
            tensors["state.rng_accelerator"] = torch.cuda.get_rng_state(
                self.device
            )
        elif self.device_backend == "mps" and hasattr(torch, "mps") and hasattr(
            torch.mps, "get_rng_state"
        ):
            tensors["state.rng_accelerator"] = torch.mps.get_rng_state()
        if self.working_memory:
            tensors["state.working_memory"] = torch.stack(self.working_memory)
        tensors.update(self.memory_lifecycle.state_tensors())
        if self._starter_action_language_cache is not None:
            tensors["state.starter_action_language_features"] = (
                self._starter_action_language_cache
            )
        if self._starter_action_internal_cache is not None:
            tensors["state.starter_action_internal_features"] = (
                self._starter_action_internal_cache
            )
        if self._starter_action_target_cache is not None:
            tensors["state.starter_action_targets"] = (
                self._starter_action_target_cache
            )
        self._sync_stability_state()
        for name, value in self.slow_anchors.items():
            tensors["stability.anchor." + name] = value
        for name, value in self.slow_importance.items():
            tensors["stability.importance." + name] = value
        return tensors

    @staticmethod
    def _sha256_identifier(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _ingestion_v3_manifest_hashes(
        *, content_hash: str, source_bytes: int, resolved_kind: str,
        source_name_hash: str, record_count_hint: Optional[int],
    ) -> Tuple[str, str]:
        """Hash the actual source identity and parser implementation contract.

        The source hash is observed from bytes before learning; neither this
        manifest nor the cursor stores a path, passage, token, or answer.
        A parser-code/dependency change refuses resume rather than silently
        changing the unvisited suffix of a committed training stream.
        """

        if not AdaptiveBrain._sha256_identifier(content_hash) or not AdaptiveBrain._sha256_identifier(source_name_hash):
            raise ValueError("v3 source manifest lacks a verified content/name hash")
        if type(source_bytes) is not int or source_bytes < 0:
            raise ValueError("v3 source size is invalid")
        if record_count_hint is not None and (
            type(record_count_hint) is not int or record_count_hint < 0
        ):
            raise ValueError("v3 source row-count hint is invalid")
        if not isinstance(resolved_kind, str) or not resolved_kind:
            raise ValueError("v3 parser kind is invalid")
        source_manifest = {
            "format": "omni-observed-source-manifest",
            "formatVersion": 1,
            "contentSha256": content_hash,
            "sourceBytes": source_bytes,
            "resolvedKind": resolved_kind,
            "sourceNameSha256": source_name_hash,
            "recordCountHint": record_count_hint,
        }
        parser_path = Path(__file__).with_name("datasets.py")
        parser_digest = hashlib.sha256()
        with parser_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                parser_digest.update(block)
        dependencies: Dict[str, str] = {}
        for package in (
            ("pyarrow",) if resolved_kind in {"parquet", "arrow"}
            else (("pypdf",) if resolved_kind == "pdf" else ())
        ):
            try:
                dependencies[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependencies[package] = "unavailable"
        parser_manifest = {
            "format": "omni-observed-parser-manifest",
            "formatVersion": 1,
            "contract": INGESTION_PARSER_CONTRACT,
            "resolvedKind": resolved_kind,
            "datasetsCodeSha256": parser_digest.hexdigest(),
            "pythonMajorMinor": "%d.%d" % sys.version_info[:2],
            "dependencies": dependencies,
        }
        return (
            hashlib.sha256(NeuralSubstrate._canonical_json(source_manifest)).hexdigest(),
            hashlib.sha256(NeuralSubstrate._canonical_json(parser_manifest)).hexdigest(),
        )

    @staticmethod
    def _v3_committed_substrate_generations(
        pointer: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Bind both logical paged regions to one immutable v3 shard generation."""

        generation_id = pointer.get("activeGeneration")
        counts = pointer.get("counts")
        if (
            pointer.get("format") != "omni-substrate-shards"
            or pointer.get("formatVersion") != 3
            or not AdaptiveBrain._sha256_identifier(generation_id)
            or pointer.get("contentSha256") != generation_id
            or not isinstance(counts, Mapping)
            or set(counts) != {"neurons", "assemblies", "synapses"}
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            raise ValueError("v3 cursor requires a committed packed substrate generation")
        vector = {
            "generationId": generation_id,
            "contentSha256": generation_id,
            "recordCount": counts["neurons"],
        }
        assembly_index = {
            "generationId": generation_id,
            "contentSha256": generation_id,
            "recordCount": counts["assemblies"],
            "highWaterSequence": counts["assemblies"],
        }
        return vector, assembly_index

    @staticmethod
    def _ingestion_learning_schedule(
        neural_storage_plan: Mapping[str, Any], checkpoint_records: int
    ) -> Dict[str, Any]:
        """Freeze every resource-derived choice that changes neural updates.

        The schedule deliberately contains only booleans, counts, and fixed
        protocol identifiers. Source text, token ids, embeddings, and record
        metadata are never checkpointed. A resumed transaction can therefore
        reproduce its committed learning trajectory without turning the
        cursor into a retrieval store.
        """

        return {
            "format": INGESTION_LEARNING_SCHEDULE_FORMAT,
            "formatVersion": INGESTION_LEARNING_SCHEDULE_VERSION,
            "detailedRecordAssemblies": bool(
                neural_storage_plan["detailedRecordAssemblies"]
            ),
            "physicalBatchRecords": max(
                1, int(neural_storage_plan["physicalBatchRecords"])
            ),
            "gradientAccumulation": max(
                1, int(neural_storage_plan["gradientAccumulation"])
            ),
            "trainingSequenceTokens": max(
                8, int(neural_storage_plan["trainingSequenceTokens"])
            ),
            "checkpointRecords": max(1, int(checkpoint_records)),
            "corpusRepresentation": str(
                neural_storage_plan["corpusRepresentation"]
            ),
            "slowGradientMode": str(neural_storage_plan["slowGradientMode"]),
            "localTypedTargetWindowPolicy": str(
                neural_storage_plan.get(
                    "localTypedTargetWindowPolicy",
                    LOCAL_TYPED_TARGET_WINDOW_POLICY,
                )
            ),
        }

    @classmethod
    def _validated_ingestion_learning_schedule(
        cls, value: Any, expected_sha256: Any
    ) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("ingestion checkpoint learning schedule is invalid")
        schedule = dict(value)
        base_fields = {
            "format",
            "formatVersion",
            "detailedRecordAssemblies",
            "physicalBatchRecords",
            "gradientAccumulation",
            "trainingSequenceTokens",
            "checkpointRecords",
            "corpusRepresentation",
            "slowGradientMode",
            "localTypedTargetWindowPolicy",
        }
        schedule_version = schedule.get("formatVersion")
        if schedule_version != INGESTION_LEARNING_SCHEDULE_VERSION:
            raise ValueError("ingestion checkpoint learning schedule is invalid")
        if set(schedule) != base_fields:
            raise ValueError("ingestion checkpoint learning schedule is invalid")
        if (
            schedule.get("format") != INGESTION_LEARNING_SCHEDULE_FORMAT
            or isinstance(schedule_version, bool)
            or not isinstance(schedule_version, int)
            or not isinstance(schedule.get("detailedRecordAssemblies"), bool)
        ):
            raise ValueError("ingestion checkpoint learning schedule is invalid")
        for field, minimum in (
            ("physicalBatchRecords", 1),
            ("gradientAccumulation", 1),
            ("trainingSequenceTokens", 8),
            ("checkpointRecords", 1),
        ):
            field_value = schedule.get(field)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < minimum
            ):
                raise ValueError(
                    "ingestion checkpoint learning schedule %s is invalid"
                    % field
                )
        if schedule.get("localTypedTargetWindowPolicy") != LOCAL_TYPED_TARGET_WINDOW_POLICY:
            raise ValueError("ingestion checkpoint local typed window policy is invalid")
        detailed = bool(schedule["detailedRecordAssemblies"])
        expected_modes = {
            "corpusRepresentation": (
                "detailed-distributed-assemblies"
                if detailed
                else "shared-semantic-field-and-local-synapses"
            ),
            "slowGradientMode": (
                "per-experience"
                if detailed
                else "streaming-microbatch-gradient-accumulation"
            ),
        }
        if any(
            schedule.get(field) != expected
            for field, expected in expected_modes.items()
        ):
            raise ValueError("ingestion checkpoint learning schedule is invalid")
        encoded = json.dumps(
            schedule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if (
            len(encoded) > 4096
            or not cls._sha256_identifier(expected_sha256)
            or hashlib.sha256(encoded).hexdigest() != expected_sha256
        ):
            raise ValueError(
                "ingestion checkpoint learning schedule hash is invalid"
            )
        return schedule

    @classmethod
    def _validated_paged_ingestion_checkpoint_v3(
        cls, source_identity: str, checkpoint: Mapping[str, Any]
    ) -> None:
        """Parse an explicit v3 header without authorizing paged resume.

        The v3 schema can validate its own hashes and coverage now.  A resume
        additionally needs independently observed, jointly committed paged
        vector/index generations.  This method must not use persisted values
        as their own observation or reinterpret a v2 cursor as v3.
        """

        if set(checkpoint) != {
            "format", "formatVersion", "parserContract", "status",
            "sourceIdentity", "transactionId", "contentHash",
            "sourceNameHash", "neuralStateChecksum", "recordPrefixSha256",
            "sourceSnapshot", "sourceBytes", "resolvedKind", "policy",
            "epoch", "committedRecords", "visitedRecords",
            "processedRecords", "rejectedRecords", "processedBytes",
            "commitSequence", "coverageAtCommit", "learningSchedule",
            "learningScheduleSha256", "baseline", "aggregate",
            "capabilityRehearsal", "capabilityRehearsalCadence",
            "committedAt", "sourceManifestSha256",
            "parserManifestSha256", "expectedRecords",
            "pagedCheckpointBinding", "pagedCheckpointBindingSha256",
        }:
            raise ValueError("v3 paged checkpoint contains unsupported fields")
        if (
            checkpoint.get("format") != INGESTION_CHECKPOINT_FORMAT
            or type(checkpoint.get("formatVersion")) is not int
            or checkpoint.get("formatVersion") != 3
            or checkpoint.get("parserContract") != INGESTION_PARSER_CONTRACT
            or checkpoint.get("status") != "active"
            or checkpoint.get("sourceIdentity") != source_identity
            or checkpoint.get("policy")
            not in {"encode", "consolidate", "pretrain"}
            or not isinstance(checkpoint.get("resolvedKind"), str)
            or not checkpoint.get("resolvedKind")
        ):
            raise ValueError("v3 paged ingestion checkpoint header is invalid")
        for field in (
            "transactionId", "contentHash", "sourceNameHash",
            "neuralStateChecksum", "recordPrefixSha256",
            "sourceManifestSha256", "parserManifestSha256",
        ):
            if not cls._sha256_identifier(checkpoint.get(field)):
                raise ValueError("v3 paged ingestion checkpoint %s is invalid" % field)
        for field, minimum in (
            ("epoch", 0), ("sourceBytes", 0),
            ("committedRecords", 0), ("visitedRecords", 0),
            ("processedRecords", 0), ("rejectedRecords", 0),
            ("processedBytes", 0), ("commitSequence", 1),
        ):
            count = checkpoint.get(field)
            if (
                type(count) is not int
                or not minimum <= count <= (1 << 63) - 1
            ):
                raise ValueError("v3 paged ingestion checkpoint %s is invalid" % field)
        expected_records = checkpoint.get("expectedRecords")
        if expected_records is not None and (
            type(expected_records) is not int or expected_records < 0
        ):
            raise ValueError("v3 paged checkpoint expected row count is invalid")
        snapshot = checkpoint.get("sourceSnapshot")
        snapshot_base = {"device", "inode", "size", "mtimeNs"}
        sqlite_extra = {
            "sqlite%s%s" % (sidecar, suffix)
            for sidecar in ("Wal", "Shm")
            for suffix in ("Present", "Device", "Inode", "Size", "MtimeNs")
        }
        if (
            not isinstance(snapshot, Mapping)
            or not snapshot_base <= set(snapshot)
            or set(snapshot) - snapshot_base not in (set(), sqlite_extra)
            or any(type(value) is not int or value < 0 for value in snapshot.values())
        ):
            raise ValueError("v3 paged checkpoint source snapshot is invalid")
        baseline = checkpoint.get("baseline")
        baseline_counts = {
            "concepts", "ideas", "plasticityEvents", "memoryNeurons",
            "memorySynapses", "memorySynapticUses", "trainingSteps",
            "statisticalExperiences",
        }
        if (
            not isinstance(baseline, Mapping)
            or set(baseline) != baseline_counts | {"parameterChecksum"}
            or not cls._sha256_identifier(baseline.get("parameterChecksum"))
            or any(type(baseline[field]) is not int or baseline[field] < 0
                   for field in baseline_counts)
        ):
            raise ValueError("v3 paged checkpoint baseline is invalid")
        aggregate = checkpoint.get("aggregate")
        aggregate_counts = {
            "learnedChunks", "readingReportCount", "streamingGradientRecords",
            "streamingGradientOptimizerSteps",
        }
        if (
            not isinstance(aggregate, Mapping)
            or set(aggregate) != aggregate_counts | {"lossTotal", "mediaAccumulator"}
            or any(type(aggregate[field]) is not int or aggregate[field] < 0
                   for field in aggregate_counts)
            or isinstance(aggregate.get("lossTotal"), bool)
            or not isinstance(aggregate.get("lossTotal"), (int, float))
            or not math.isfinite(float(aggregate["lossTotal"]))
            or not isinstance(aggregate.get("mediaAccumulator"), Mapping)
            or dict(aggregate["mediaAccumulator"])
            != cls._checkpoint_media_accumulator(aggregate["mediaAccumulator"])
        ):
            raise ValueError("v3 paged checkpoint aggregate is invalid")
        covered = checkpoint.get("coverageAtCommit")
        coverage_counts = {
            "discoveredFiles", "completedFiles", "processedFiles",
            "rejectedFiles", "discoveredRecords", "processedRecords",
            "rejectedRecords", "processedBytes", "shards", "errorCount",
        }
        if (
            not isinstance(covered, Mapping)
            or set(covered) != coverage_counts | {
                "modalityCounts", "errors", "errorsTruncated", "complete"
            }
            or any(type(covered[field]) is not int or covered[field] < 0
                   for field in coverage_counts)
            or covered["errors"] != []
            or type(covered["errorsTruncated"]) is not bool
            or type(covered["complete"]) is not bool
            or not isinstance(covered["modalityCounts"], Mapping)
            or any(
                not isinstance(kind, str) or len(kind) > 32
                or type(count) is not int or count < 0
                for kind, count in covered["modalityCounts"].items()
            )
            or covered["discoveredRecords"] != checkpoint["visitedRecords"]
            or covered["processedRecords"] != checkpoint["processedRecords"]
            or covered["rejectedRecords"] != checkpoint["rejectedRecords"]
            or covered["processedBytes"] != checkpoint["processedBytes"]
        ):
            raise ValueError("v3 paged checkpoint coverage snapshot is invalid")
        schedule = validate_ingestion_schedule_v3(
            checkpoint.get("learningSchedule"),
            checkpoint.get("learningScheduleSha256"),
            source_manifest_sha256=checkpoint["sourceManifestSha256"],
            parser_manifest_sha256=checkpoint["parserManifestSha256"],
            source_content_sha256=checkpoint["contentHash"],
        )
        binding = validate_unobserved_checkpoint_binding_v3(
            checkpoint.get("pagedCheckpointBinding"),
            checkpoint.get("pagedCheckpointBindingSha256"),
            schedule=schedule,
            schedule_sha256_value=checkpoint["learningScheduleSha256"],
            source_manifest_sha256=checkpoint["sourceManifestSha256"],
            parser_manifest_sha256=checkpoint["parserManifestSha256"],
            source_content_sha256=checkpoint["contentHash"],
        )
        cursor = binding["cursor"]
        coverage = binding["coverage"]
        for field, expected in (
            ("neuralStateChecksum", binding["neuralStateSha256"]),
            ("recordPrefixSha256", cursor["recordPrefixSha256"]),
            ("committedRecords", cursor["committedRecords"]),
            ("visitedRecords", coverage["visitedRecords"]),
            ("processedRecords", coverage["processedRecords"]),
            ("rejectedRecords", coverage["rejectedRecords"]),
            ("processedBytes", coverage["processedBytes"]),
            ("commitSequence", binding["checkpointSequence"]),
        ):
            if checkpoint[field] != expected:
                raise ValueError(
                    "v3 paged ingestion checkpoint %s binding mismatch" % field
                )

    @classmethod
    def _validated_ingestion_checkpoints(
        cls, value: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Validate resumable record cursors before any neural state is used.

        A malformed cursor must never be treated as zero: doing that would
        replay already committed neural updates.  Conversely, the cursor is
        inspection/transaction metadata only and is not allowed to carry raw
        source text or token sequences.
        """

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("ingestion checkpoint registry is invalid")
        validated: Dict[str, Dict[str, Any]] = {}
        for key, raw in value.items():
            if not cls._sha256_identifier(key) or not isinstance(raw, Mapping):
                raise ValueError("ingestion checkpoint identity is invalid")
            checkpoint = dict(raw)
            if (
                type(checkpoint.get("formatVersion")) is int
                and checkpoint["formatVersion"] == 3
            ):
                cls._validated_paged_ingestion_checkpoint_v3(key, checkpoint)
                validated[str(key)] = checkpoint
                continue
            if (
                checkpoint.get("format") != INGESTION_CHECKPOINT_FORMAT
                or int(checkpoint.get("formatVersion", 0))
                != INGESTION_CHECKPOINT_VERSION
                or checkpoint.get("parserContract")
                != INGESTION_PARSER_CONTRACT
                or checkpoint.get("status") != "active"
                or checkpoint.get("sourceIdentity") != key
                or not cls._sha256_identifier(checkpoint.get("transactionId"))
                or not cls._sha256_identifier(checkpoint.get("contentHash"))
                or not cls._sha256_identifier(checkpoint.get("sourceNameHash"))
                or not cls._sha256_identifier(
                    checkpoint.get("neuralStateChecksum")
                )
                or not cls._sha256_identifier(
                    checkpoint.get("recordPrefixSha256")
                )
                or checkpoint.get("policy")
                not in {"encode", "consolidate", "pretrain"}
                or not isinstance(checkpoint.get("resolvedKind"), str)
                or not checkpoint.get("resolvedKind")
            ):
                raise ValueError("ingestion checkpoint contract is invalid")
            integer_fields = (
                "epoch",
                "sourceBytes",
                "committedRecords",
                "visitedRecords",
                "processedRecords",
                "rejectedRecords",
                "commitSequence",
            )
            for field in integer_fields:
                field_value = checkpoint.get(field)
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint %s is invalid" % field
                    )
            if checkpoint["commitSequence"] < 1:
                raise ValueError("ingestion checkpoint sequence is invalid")
            if checkpoint["visitedRecords"] < checkpoint["committedRecords"]:
                raise ValueError("ingestion checkpoint coverage is invalid")
            if checkpoint["visitedRecords"] != (
                checkpoint["processedRecords"]
                + checkpoint["rejectedRecords"]
            ):
                raise ValueError("ingestion checkpoint coverage is incomplete")
            baseline = checkpoint.get("baseline")
            aggregate = checkpoint.get("aggregate")
            coverage_at_commit = checkpoint.get("coverageAtCommit")
            source_snapshot = checkpoint.get("sourceSnapshot")
            generation_binding = checkpoint.get("committedGeneration")
            learning_schedule = cls._validated_ingestion_learning_schedule(
                checkpoint.get("learningSchedule"),
                checkpoint.get("learningScheduleSha256"),
            )
            checkpoint["learningSchedule"] = learning_schedule
            if not isinstance(baseline, Mapping) or not isinstance(
                aggregate, Mapping
            ) or not isinstance(coverage_at_commit, Mapping) or not isinstance(
                source_snapshot, Mapping
            ) or not isinstance(generation_binding, Mapping):
                raise ValueError("ingestion checkpoint neural summary is invalid")
            binding_sha = hashlib.sha256(
                json.dumps(
                    dict(generation_binding),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                generation_binding.get("format")
                != "omni-ingestion-generation-binding"
                or int(generation_binding.get("formatVersion", 0)) != 1
                or binding_sha != checkpoint.get("committedGenerationSha256")
            ):
                raise ValueError(
                    "ingestion checkpoint generation binding is invalid"
                )
            for field in (
                "mutableStateActiveGeneration",
                "mutableStateContentSha256",
                "substrateContentSha256",
                "replayContentSha256",
                "workspacePageContentSha256",
                "neuralStateChecksum",
            ):
                if not cls._sha256_identifier(generation_binding.get(field)):
                    raise ValueError(
                        "ingestion checkpoint generation binding is invalid"
                    )
            for field in (
                "replayCount",
                "replayHighWaterId",
                "workspacePageCount",
                "workspacePageHighWaterId",
            ):
                field_value = generation_binding.get(field)
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint generation binding is invalid"
                    )
            if generation_binding.get("workspacePagesLearningReadable") is not False:
                raise ValueError(
                    "ingestion checkpoint workspace binding is invalid"
                )
            for field in ("device", "inode", "size", "mtimeNs"):
                field_value = source_snapshot.get(field)
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint source snapshot is invalid"
                    )
            if (
                int(coverage_at_commit.get("discoveredRecords", -1))
                != checkpoint["visitedRecords"]
                or int(coverage_at_commit.get("processedRecords", -1))
                != checkpoint["processedRecords"]
                or int(coverage_at_commit.get("rejectedRecords", -1))
                != checkpoint["rejectedRecords"]
                or len(coverage_at_commit.get("errors", []))
                > DatasetCoverage._ERROR_SAMPLE_LIMIT
            ):
                raise ValueError("ingestion checkpoint coverage snapshot is invalid")
            if not cls._sha256_identifier(
                baseline.get("parameterChecksum")
            ):
                raise ValueError("ingestion checkpoint baseline is invalid")
            for field in (
                "concepts",
                "ideas",
                "plasticityEvents",
            ):
                field_value = baseline.get(field)
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint baseline %s is invalid" % field
                    )
            for field in ("memorySynapticUses", "trainingSteps"):
                field_value = baseline.get(field)
                if field_value is not None and (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint baseline %s is invalid" % field
                    )
            if not isinstance(aggregate.get("mediaAccumulator", {}), Mapping):
                raise ValueError(
                    "ingestion checkpoint media accumulator is invalid"
                )
            for field in (
                "learnedChunks",
                "readingReportCount",
                "streamingGradientRecords",
                "streamingGradientOptimizerSteps",
            ):
                field_value = aggregate.get(field, 0)
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        "ingestion checkpoint aggregate %s is invalid" % field
                    )
            if any(
                key.startswith(("sequence", "multiStage"))
                for key in (*baseline, *aggregate)
            ):
                raise ValueError("ingestion checkpoint contains obsolete answer-key state")
            media_accumulator = dict(aggregate.get("mediaAccumulator", {}))
            safe_media = cls._checkpoint_media_accumulator(media_accumulator)
            encoded_media = json.dumps(
                safe_media,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if media_accumulator != safe_media or len(encoded_media) > 128 * 1024:
                raise ValueError(
                    "ingestion checkpoint media diagnostics violate the "
                    "bounded scalar/count/hash contract"
                )
            validated[str(key)] = checkpoint
        return validated

    @classmethod
    def _validated_completed_ingestions(cls, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > COMPLETED_INGESTION_TOMBSTONES:
            raise ValueError("completed ingestion tombstones are invalid")
        result: List[Dict[str, Any]] = []
        seen: set = set()
        for raw in value:
            if not isinstance(raw, Mapping):
                raise ValueError("completed ingestion tombstone is invalid")
            item = dict(raw)
            if (
                item.get("format") != "omni-completed-ingestion"
                or int(item.get("formatVersion", 0)) != 1
                or not cls._sha256_identifier(item.get("transactionId"))
                or not cls._sha256_identifier(item.get("contentHash"))
                or not cls._sha256_identifier(item.get("sourceIdentity"))
                or not cls._sha256_identifier(item.get("sourceNameHash"))
                or not cls._sha256_identifier(item.get("parameterChecksumAfter"))
                or item.get("policy")
                not in {"encode", "consolidate", "pretrain"}
                or isinstance(item.get("epoch"), bool)
                or not isinstance(item.get("epoch"), int)
                or int(item.get("epoch", -1)) < 0
                or not isinstance(item.get("coverage"), Mapping)
            ):
                raise ValueError("completed ingestion tombstone contract is invalid")
            transaction = str(item["transactionId"])
            if transaction in seen:
                raise ValueError("completed ingestion tombstone is duplicated")
            seen.add(transaction)
            result.append(item)
        return result

    def _enqueue_chat_slow_learning(
        self,
        *,
        turn_id: str,
        input_sha256: str,
        human_message_id: str,
        experience: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Queue one idempotent cortical replay after fast chat admission."""

        if not (
            self.config.online_learning and int(self.config.online_steps) > 0
        ):
            return None
        job_id = hashlib.sha256(
            (
                "%s\0%s\0%s\0chat-slow-learning-v1"
                % (self.brain_id, turn_id, input_sha256)
            ).encode("utf-8")
        ).hexdigest()
        if job_id in self.completed_chat_slow_learning:
            return None
        existing = next(
            (
                item
                for item in self.pending_chat_slow_learning
                if item.get("jobId") == job_id
            ),
            None,
        )
        if existing is not None:
            return existing
        settling = experience.get("memory_settling")
        settling = settling if isinstance(settling, Mapping) else {}
        signals = settling.get("signals")
        signals = signals if isinstance(signals, Mapping) else {}
        assembly_id = str(experience.get("assembly_id", ""))
        assembly = next(
            (
                item
                for item in self.memory.assemblies
                if str(item.get("id", "")) == assembly_id
            ),
            {},
        )
        retention_assessment = self.memory_lifecycle.assess_retention_candidate(
            novelty=float(experience.get("novelty", 0.0)),
            reuse=float(signals.get("reuse", 0.0)),
            salience=float(signals.get("salience", 0.0)),
            prediction_error=float(
                experience.get("retention_prediction_error", 0.0)
            ),
            rehearsals=max(1, int(assembly.get("rehearsals", 1))),
            stability=float(signals.get("stability", 0.0)),
            related_coactivation=float(signals.get("recurrence", 0.0)),
            interference=float(signals.get("interference", 0.0)),
            recurrence=float(signals.get("recurrence", 0.0)),
            activation=float(signals.get("activation", 0.0)),
            observations=max(1, int(assembly.get("rehearsals", 1))),
        )
        retention = float(retention_assessment["retentionScore"])
        priority = float(retention_assessment["slowReplayPriority"])
        replay_strength = max(
            0.02,
            min(
                1.0,
                priority
                * max(0.05, float(self.config.consolidation_rate) / 0.06),
            ),
        )
        reinforcement = max(
            0.0,
            min(1.0, float(settling.get("reinforcementDrive", 0.0))),
        )
        unfinished = max(
            0.0, min(1.0, float(settling.get("unfinishedScore", 0.0)))
        )
        interference = max(
            0.0, min(1.0, float(signals.get("interference", 0.0)))
        )
        record = {
            "format": "omni-chat-slow-learning-job",
            "formatVersion": 1,
            "jobId": job_id,
            "turnId": turn_id,
            "inputSha256": input_sha256,
            "humanMessageId": human_message_id,
            "queuedAt": _iso_now(),
            "onlineSteps": int(self.config.online_steps),
            "priority": priority,
            "replayStrength": replay_strength,
            "retentionScore": retention,
            "reinforcementDrive": reinforcement,
            "unfinishedScore": unfinished,
            "interference": interference,
            "fadePressure": float(retention_assessment["fadePressure"]),
            "retentionEvidence": retention_assessment["evidence"],
            "retentionReasons": retention_assessment["reasons"],
            "fastEpisodePolicy": retention_assessment["fastEpisodePolicy"],
            "novelty": max(
                0.0, min(1.0, float(experience.get("novelty", 0.0)))
            ),
            "candidateCheckpoint": "last-atomic-neural-generation",
        }
        self.pending_chat_slow_learning.append(record)
        return record

    def consolidate_pending_chat_learning(
        self,
        job_id: str = "",
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Promote one queued replay as an atomic, retry-safe slow update."""

        requested = str(job_id or "").strip()
        candidates = [
            item
            for item in self.pending_chat_slow_learning
            if not requested or str(item.get("jobId", "")) == requested
        ]
        if not candidates:
            return {
                "brainId": self.brain_id,
                "processed": False,
                "pending": len(self.pending_chat_slow_learning),
                "idempotent": bool(
                    requested and requested in self.completed_chat_slow_learning
                ),
            }
        record = max(
            candidates,
            key=lambda item: (
                float(item.get("priority", 0.0)),
                str(item.get("queuedAt", "")),
            ),
        )
        selected_job_id = str(record["jobId"])
        if cancel_check is not None and cancel_check():
            raise ChatGenerationCancelled("background chat learning was cancelled")
        message = self.conversation.payload_by_id(
            "message", str(record["humanMessageId"])
        )
        if (
            not isinstance(message, Mapping)
            or message.get("role") != "human"
            or hashlib.sha256(
                str(message.get("content", "")).encode("utf-8")
            ).hexdigest()
            != record["inputSha256"]
        ):
            raise RuntimeError("queued chat replay evidence is unavailable")
        text = str(message["content"])
        snapshot = self._snapshot_slow_transaction_state()
        pending_before = list(self.pending_chat_slow_learning)
        completed_before = list(self.completed_chat_slow_learning)
        mutable_pointer_before = copy.deepcopy(self.mutable_state_manifest)
        substrate_pointer_before = copy.deepcopy(self.memory.persistence_manifest)
        before = str(snapshot["checksum"])
        cortical_before = self._cortical_parameter_checksum()
        module_checksums_before = {
            name: tensor_checksum(module.parameters())
            for name, module in self._slow_transaction_modules().items()
        }
        try:
            cue = self.memory.vector_for_text(text)
            replay_strength = max(
                0.02, min(1.0, float(record.get("replayStrength", 1.0)))
            )
            self._ensure_optimizer_resident()
            optimizer_groups = list(self._optimizer.param_groups)
            learning_rates = [float(group["lr"]) for group in optimizer_groups]
            try:
                for group, learning_rate in zip(
                    optimizer_groups, learning_rates
                ):
                    group["lr"] = learning_rate * replay_strength
                training = self._optimize_experience(
                    text,
                    cue,
                    steps=max(1, int(record.get("onlineSteps", 1))),
                )
                if cancel_check is not None and cancel_check():
                    raise ChatGenerationCancelled(
                        "background chat learning was cancelled"
                    )
            finally:
                for group, learning_rate in zip(
                    optimizer_groups, learning_rates
                ):
                    group["lr"] = learning_rate
            grew = self._maybe_grow(
                float(record.get("novelty", 0.0)) * replay_strength,
                self._idea_model_vector(cue)[0],
            )
            calibration = None
            if self._can_retain_native_action_policy():
                calibration = self._calibrate_starter_action_policy(
                    max_steps=96,
                    minimum_steps=0,
                    strict=True,
                )
            if cancel_check is not None and cancel_check():
                raise ChatGenerationCancelled(
                    "background chat learning was cancelled"
                )
            after = self._slow_parameter_checksum()
            cortical_after = self._cortical_parameter_checksum()
            module_checksums_after = {
                name: tensor_checksum(module.parameters())
                for name, module in self._slow_transaction_modules().items()
            }
            updated_modules = sorted(
                name
                for name, checksum in module_checksums_after.items()
                if checksum != module_checksums_before[name]
            )
            self.pending_chat_slow_learning = [
                item
                for item in self.pending_chat_slow_learning
                if item.get("jobId") != selected_job_id
            ]
            self.completed_chat_slow_learning.append(selected_job_id)
            self.completed_chat_slow_learning = (
                self.completed_chat_slow_learning[
                    -COMPLETED_CHAT_SLOW_LEARNING:
                ]
            )
            result = {
                "brainId": self.brain_id,
                "processed": True,
                "jobId": selected_job_id,
                "turnId": str(record.get("turnId", "")),
                "priority": float(record.get("priority", 0.0)),
                "replayStrength": replay_strength,
                "parameterChecksumBefore": before,
                "parameterChecksumAfter": after,
                "corticalParameterChecksumBefore": cortical_before,
                "corticalParameterChecksumAfter": cortical_after,
                "corticalParametersUpdated": cortical_before != cortical_after,
                "updatedTrainableModules": updated_modules,
                "training": training,
                "expertGrew": grew,
                "actionCalibration": calibration,
                "pending": len(self.pending_chat_slow_learning),
                "nextPriority": max(
                    (
                        float(item.get("priority", 0.0))
                        for item in self.pending_chat_slow_learning
                    ),
                    default=0.0,
                ),
                "transaction": "atomic-candidate-promoted",
            }
            # Queue removal, completed tombstone, weights, optimizer moments,
            # and stability tensors become authoritative in one generation.
            self.save()
            try:
                self.events.append(
                    "chat-slow-learning-complete",
                    result,
                    job_id=selected_job_id,
                )
            except Exception:
                # The checkpoint is already authoritative. A diagnostics-log
                # failure must never roll it back in RAM and invite duplicate
                # optimizer replay on the still-running worker.
                pass
            return result
        except BaseException as error:
            # brain.json is the authoritative commit record. Finalization
            # (publishing the convenience pointer or pruning old blobs) may
            # fail after that atomic replacement. Never roll a committed job
            # back into RAM and replay its optimizer step a second time.
            try:
                committed = read_json(self.engine_path / "brain.json")
                committed_ids = committed.get("completed_chat_slow_learning")
                committed_pending = committed.get("pending_chat_slow_learning")
                committed_pointer = committed.get("mutable_state")
                job_committed = (
                    isinstance(committed_ids, list)
                    and selected_job_id in committed_ids
                    and isinstance(committed_pending, list)
                    and not any(
                        isinstance(item, Mapping)
                        and item.get("jobId") == selected_job_id
                        for item in committed_pending
                    )
                    and committed_pointer == self.mutable_state_manifest
                )
            except (OSError, ValueError, TypeError):
                job_committed = False
            if job_committed:
                result["postCommitFinalizationError"] = str(error)[:512]
                return result
            self._restore_slow_transaction_state(snapshot)
            self.pending_chat_slow_learning = pending_before
            self.completed_chat_slow_learning = completed_before
            self.mutable_state_manifest = mutable_pointer_before
            self.memory.persistence_manifest = substrate_pointer_before
            raise

    @classmethod
    def _validated_completed_chat_turns(
        cls, value: Any
    ) -> List[Dict[str, Any]]:
        """Validate bounded references committed with an atomic chat save."""

        if value is None:
            return []
        if not isinstance(value, list) or len(value) > COMPLETED_CHAT_TURN_RECEIPTS:
            raise ValueError("completed chat turn receipts are invalid")
        required = {
            "format",
            "formatVersion",
            "turnId",
            "inputSha256",
            "humanMessageId",
            "brainMessageId",
            "traceId",
            "inferenceCount",
            "parameterChecksumAfter",
            "committedAt",
        }
        receipts: List[Dict[str, Any]] = []
        seen = set()
        for raw in value:
            if not isinstance(raw, Mapping):
                raise ValueError("completed chat turn receipt is invalid")
            receipt = dict(raw)
            turn_id = receipt.get("turnId")
            if (
                set(receipt) != required
                or receipt.get("format") != CHAT_TURN_RECEIPT_FORMAT
                or receipt.get("formatVersion") != 1
                or not isinstance(turn_id, str)
                or not turn_id
                or len(turn_id) > 128
                or "\x00" in turn_id
                or any(character in "\r\n" for character in turn_id)
                or not cls._sha256_identifier(receipt.get("inputSha256"))
                or not cls._sha256_identifier(
                    receipt.get("parameterChecksumAfter")
                )
                or not all(
                    isinstance(receipt.get(field), str)
                    and 0 < len(str(receipt[field])) <= 128
                    for field in (
                        "humanMessageId",
                        "brainMessageId",
                        "traceId",
                    )
                )
                or isinstance(receipt.get("inferenceCount"), bool)
                or not isinstance(receipt.get("inferenceCount"), int)
                or int(receipt["inferenceCount"]) < 1
                or not isinstance(receipt.get("committedAt"), str)
                or not receipt["committedAt"]
            ):
                raise ValueError("completed chat turn receipt contract is invalid")
            receipt_key = (turn_id, str(receipt["inputSha256"]))
            if receipt_key in seen:
                raise ValueError("completed chat turn receipt is duplicated")
            seen.add(receipt_key)
            receipts.append(receipt)
        return receipts

    @classmethod
    def _validated_fresh_attention_boundary(
        cls, value: Any
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("fresh attention boundary is invalid")
        boundary = dict(value)
        required = {
            "format",
            "formatVersion",
            "operationId",
            "epoch",
            "createdAt",
            "messagesPreserved",
            "tracesPreserved",
            "synapsesPreserved",
            "replayEntries",
            "parameterChecksum",
            "fastSynapseChecksum",
            "substrateContentSha256",
            "cleared",
        }
        cleared_fields = {
            "recentTokens",
            "currentPromptTokens",
            "residentWorkingMemory",
            "pagedWorkingMemory",
            "lifecycleScratch",
            "activeFocus",
            "activatedNeurons",
            "recalledAssemblies",
            "substrateEligibilityTraces",
            "liquidStateUnits",
            "rawHistoryMessages",
            "automaticTraceInfluence",
            "recallAuditEntries",
            "noveltyStreak",
            "legacyRawAttentionOverlay",
            "routerMembraneUnits",
            "routerSpikeUnits",
            "routerPreTraceUnits",
            "routerPostTraceUnits",
        }
        operation_id = boundary.get("operationId")
        cleared = boundary.get("cleared")
        integer_fields = (
            "epoch",
            "messagesPreserved",
            "tracesPreserved",
            "synapsesPreserved",
            "replayEntries",
        )
        if (
            set(boundary) != required
            or boundary.get("format") != FRESH_ATTENTION_FORMAT
            or boundary.get("formatVersion") != FRESH_ATTENTION_VERSION
            or not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id) > 128
            or "\x00" in operation_id
            or any(character in "\r\n" for character in operation_id)
            or not isinstance(boundary.get("createdAt"), str)
            or not boundary["createdAt"]
            or not cls._sha256_identifier(boundary.get("parameterChecksum"))
            or not cls._sha256_identifier(boundary.get("fastSynapseChecksum"))
            or not cls._sha256_identifier(
                boundary.get("substrateContentSha256")
            )
            or not isinstance(cleared, Mapping)
            or set(cleared) != cleared_fields
            or any(
                isinstance(boundary.get(field), bool)
                or not isinstance(boundary.get(field), int)
                or int(boundary[field]) < (1 if field == "epoch" else 0)
                for field in integer_fields
            )
            or any(
                isinstance(cleared.get(field), bool)
                or not isinstance(cleared.get(field), int)
                or int(cleared[field]) < 0
                for field in cleared_fields
            )
        ):
            raise ValueError("fresh attention boundary contract is invalid")
        return boundary

    @staticmethod
    def _record_prefix_digest(
        previous: str, ordinal: int, record: Any
    ) -> str:
        """Extend a source-free digest over one deterministic yielded record."""

        provenance = dict(getattr(record, "provenance", {}) or {})
        provenance_sha = hashlib.sha256(
            json.dumps(
                provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        body = {
            "ordinal": max(1, int(ordinal)),
            "kind": str(getattr(record, "kind", "text")),
            "nameSha256": hashlib.sha256(
                str(getattr(record, "name", "")).encode("utf-8")
            ).hexdigest(),
            "textSha256": hashlib.sha256(
                str(getattr(record, "text", "")).encode("utf-8")
            ).hexdigest(),
            "contentSha256": str(getattr(record, "content_sha256", "")),
            "provenanceSha256": provenance_sha,
        }
        return hashlib.sha256(
            bytes.fromhex(previous)
            + json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _bind_ingestion_checkpoint_generation(
        self, pointer: Mapping[str, Any]
    ) -> None:
        if not self.ingestion_checkpoints:
            self.ingestion_joint_generation = None
            return
        current_neural_checksum = self.parameter_checksum()
        substrate_pointer = self.memory.persistence_manifest or {}
        replay_checkpoint = self.replay.checkpoint()
        workspace_checkpoint = (
            PagedWorkingMemory.empty_checkpoint()
            if self._fresh_attention_paged_clear_pending
            else self.paged_working_memory.checkpoint()
        )
        for checkpoint in self.ingestion_checkpoints.values():
            if checkpoint.get("neuralStateChecksum") != current_neural_checksum:
                raise RuntimeError(
                    "active ingestion cursor does not describe current neural state"
                )
            if checkpoint.get("formatVersion") == 3:
                schedule = validate_ingestion_schedule_v3(
                    checkpoint.get("learningSchedule"),
                    checkpoint.get("learningScheduleSha256"),
                    source_manifest_sha256=checkpoint["sourceManifestSha256"],
                    parser_manifest_sha256=checkpoint["parserManifestSha256"],
                    source_content_sha256=checkpoint["contentHash"],
                )
                counts = self.memory.persistence_manifest or {}
                vector_generation, index_generation = (
                    self._v3_committed_substrate_generations(counts)
                )
                coverage = {
                    "visitedRecords": int(checkpoint["visitedRecords"]),
                    "processedRecords": int(checkpoint["processedRecords"]),
                    "rejectedRecords": int(checkpoint["rejectedRecords"]),
                    "processedBytes": int(checkpoint["processedBytes"]),
                    "expectedRecords": checkpoint.get("expectedRecords"),
                    "sourceStreamExhausted": False,
                    "sourceContentReverifiedSha256": None,
                }
                cursor = {
                    "committedRecords": int(checkpoint["committedRecords"]),
                    "recordPrefixSha256": str(checkpoint["recordPrefixSha256"]),
                }
                binding = make_checkpoint_binding_v3(
                    schedule=schedule,
                    schedule_sha256_value=checkpoint["learningScheduleSha256"],
                    neural_state_sha256=current_neural_checksum,
                    checkpoint_sequence=int(checkpoint["commitSequence"]),
                    cursor=cursor,
                    coverage=coverage,
                    vector_generation=vector_generation,
                    index_generation=index_generation,
                )
                # Stage immutable neural and substrate references without a
                # full SQLite backup. The cache can be rebuilt from verified
                # v3 shards; only brain.json commits this reference/cursor.
                reference = stage_joint_generation(
                    self.engine_path / "state" / "ingestion-joint",
                    neural_store_root=self.engine_path / "state",
                    neural_pointer=pointer,
                    substrate_store_root=self.engine_path / "substrate",
                    substrate_pointer=counts,
                    sqlite_backup=None,
                    source_manifest_sha256=checkpoint["sourceManifestSha256"],
                    parser_manifest_sha256=checkpoint["parserManifestSha256"],
                    source_content_sha256=checkpoint["contentHash"],
                    checkpoint_sequence=int(checkpoint["commitSequence"]),
                    cursor=cursor,
                    coverage=coverage,
                    previous_reference=self.ingestion_joint_generation,
                    verified_cache=self._joint_artifact_cache,
                )
                checkpoint["pagedCheckpointBinding"] = binding
                checkpoint["pagedCheckpointBindingSha256"] = (
                    checkpoint_binding_sha256(binding)
                )
                self.ingestion_joint_generation = reference
                continue
            binding = {
                "format": "omni-ingestion-generation-binding",
                "formatVersion": 1,
                "mutableStateActiveGeneration": str(
                    pointer.get("activeGeneration", "")
                ),
                "mutableStateContentSha256": str(
                    pointer.get("contentSha256", "")
                ),
                "substrateContentSha256": str(
                    substrate_pointer.get("contentSha256", "")
                ),
                "replayCount": int(replay_checkpoint["count"]),
                "replayHighWaterId": int(replay_checkpoint["highWaterId"]),
                "replayContentSha256": str(
                    replay_checkpoint["contentSha256"]
                ),
                "workspacePageCount": int(workspace_checkpoint["count"]),
                "workspacePageHighWaterId": int(
                    workspace_checkpoint["highWaterId"]
                ),
                "workspacePageContentSha256": str(
                    workspace_checkpoint["contentSha256"]
                ),
                "workspacePagesLearningReadable": False,
                "neuralStateChecksum": current_neural_checksum,
            }
            checkpoint["committedGeneration"] = binding
            checkpoint["committedGenerationSha256"] = hashlib.sha256(
                json.dumps(
                    binding,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

    @staticmethod
    def _source_without_inline_text(value: Mapping[str, Any]) -> Dict[str, Any]:
        source = dict(value)
        source.pop("raw_text", None)
        source["raw_text_retained"] = False
        return source

    def _fresh_attention_runtime_card(
        self, previous: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Patch the committed card without rescanning the durable substrate."""

        if self.fresh_attention_boundary is None:
            raise RuntimeError("fresh attention runtime card requires a boundary")
        card = copy.deepcopy(dict(previous))
        card["fresh_attention"] = {
            "active": True,
            "epoch": int(self.fresh_attention_boundary["epoch"]),
            "createdAt": str(self.fresh_attention_boundary["createdAt"]),
            "messagesPreserved": int(
                self.fresh_attention_boundary["messagesPreserved"]
            ),
            "rawPriorDialogueEligible": False,
            "cleared": dict(self.fresh_attention_boundary["cleared"]),
        }
        card["working_memory_vectors"] = 0
        workspace = card.get("workspace")
        if isinstance(workspace, Mapping):
            workspace = copy.deepcopy(dict(workspace))
            context = workspace.get("contextWindow")
            if isinstance(context, Mapping):
                workspace["contextWindow"] = {
                    **dict(context),
                    "tokenCount": 0,
                    "tokenHash": "",
                    "recentTokenCount": 0,
                    "recentTokenHash": self._token_sequence_hash([]),
                    "sensorySlots": 0,
                    "updatedAt": str(
                        self.current_context.get("updatedAt", self.updated_at)
                    ),
                }
            latent = workspace.get("latentWorkspace")
            if isinstance(latent, Mapping):
                workspace["latentWorkspace"] = {
                    **dict(latent),
                    "occupancy": 0,
                    "resident": 0,
                    "paged": 0,
                    "items": [],
                }
            liquid = workspace.get("liquidState")
            if isinstance(liquid, Mapping):
                workspace["liquidState"] = {
                    **dict(liquid),
                    "mean": 0.0,
                    "norm": 0.0,
                }
            memory = workspace.get("memory")
            if isinstance(memory, Mapping):
                memory = copy.deepcopy(dict(memory))
                memory.pop("fadingScratchTrail", None)
                memory.pop("lastingConnections", None)
                for field in ("activeFocus", "afterimageTrail"):
                    value = memory.get(field)
                    if isinstance(value, Mapping):
                        memory[field] = {
                            **dict(value),
                            "count": 0,
                            "items": [],
                            **(
                                {"averageStrength": 0.0}
                                if field == "afterimageTrail"
                                else {}
                            ),
                        }
                retention = memory.get("retentionDynamics")
                if isinstance(retention, Mapping):
                    memory["retentionDynamics"] = {
                        **dict(retention),
                        "fixedStages": False,
                        "reversible": True,
                    }
                recent = memory.get("recentWords")
                if isinstance(recent, Mapping):
                    memory["recentWords"] = {**dict(recent), "count": 0}
                thoughts = memory.get("workingThoughts")
                if isinstance(thoughts, Mapping):
                    memory["workingThoughts"] = {
                        **dict(thoughts),
                        "count": 0,
                        "resident": 0,
                        "paged": 0,
                        "items": [],
                    }
                workspace["memory"] = memory
            workspace["freshAttentionBoundary"] = dict(
                self.fresh_attention_boundary
            )
            card["workspace"] = workspace
        state_offload = card.get("state_offload")
        if isinstance(state_offload, Mapping):
            state_offload = copy.deepcopy(dict(state_offload))
            paging = state_offload.get("workingMemoryPaging")
            if isinstance(paging, Mapping):
                state_offload["workingMemoryPaging"] = {
                    **dict(paging),
                    "count": 0,
                }
            residency = state_offload.get("hotStateResidency")
            if isinstance(residency, Mapping):
                state_offload["hotStateResidency"] = {
                    **dict(residency),
                    "attentionEpoch": self._attention_epoch(),
                    "attentionOverlayActiveEntities": 0,
                    "orderingRefreshDeferred": True,
                }
            card["state_offload"] = state_offload
        dynamics = card.get("intrinsic_dynamics")
        if isinstance(dynamics, Mapping):
            card["intrinsic_dynamics"] = {
                **dict(dynamics),
                "activeFraction": 0.0,
                "learningProgress": 0.0,
            }
        return card

    def _metadata(
        self,
        *,
        runtime_card_override: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "release_format": "stable-1.0",
            "format": "omni-cortex-engine",
            "brain_id": self.brain_id,
            "name": self.config.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config.to_dict(),
            "expert_count": self.decoder.expert_count,
            "novelty_streak": self.novelty_streak,
            "growth_pause": self.growth_pause,
            "resource_pause": self.resource_pause,
            "last_activity_decay": self.last_activity_decay,
            "last_idle_cycle_at": self.last_idle_cycle_at,
            "last_idle_visible_action_at": self.last_idle_visible_action_at,
            "conversation": self.conversation.summary(),
            "training_sources": [
                self._source_without_inline_text(value)
                for value in self.training_sources
                if isinstance(value, Mapping)
            ],
            "ingestion_checkpoints": self.ingestion_checkpoints,
            "ingestion_joint_generation": self.ingestion_joint_generation,
            "paged_substrate_required": self._paged_substrate_required,
            "completed_ingestions": self.completed_ingestions[
                -COMPLETED_INGESTION_TOMBSTONES:
            ],
            "completed_chat_turns": self.completed_chat_turns[
                -COMPLETED_CHAT_TURN_RECEIPTS:
            ],
            "completed_chat_slow_learning": self.completed_chat_slow_learning[
                -COMPLETED_CHAT_SLOW_LEARNING:
            ],
            "pending_chat_slow_learning": self.pending_chat_slow_learning,
            "workspace_items": self.workspace_items,
            "memory_lifecycle": self.memory_lifecycle.metadata(),
            "recent_token_context": self.recent_token_context,
            "current_context": self.current_context,
            "fresh_attention_boundary": self.fresh_attention_boundary,
            "attention_overlay": self.memory.attention_overlay_metadata(),
            "counters": self.counters,
            "packed_metaplasticity": self._packed_stability_accounting(),
            "modality_training": self.modality_training,
            "installed_modality_packs": self.installed_modality_packs,
            "ground_up_training_manifest": self.ground_up_training_manifest,
            "packed_ternary_manifest": self.packed_ternary_manifest,
            "substrate": self.memory.metadata(include_records=False),
            "mutable_state": self.mutable_state_manifest,
            "paged_working_memory": (
                PagedWorkingMemory.empty_checkpoint()
                if self._fresh_attention_paged_clear_pending
                else self.paged_working_memory.checkpoint()
            ),
            "runtime_card": (
                copy.deepcopy(dict(runtime_card_override))
                if isinstance(runtime_card_override, Mapping)
                else self.runtime_card()
            ),
            "files": {
                "core": "core.safetensors",
                "plasticity": "plasticity.safetensors",
                "substrate": "substrate/manifest.json",
                "mutableState": "state/manifest.json",
                "replay": "state/replay.sqlite3",
                "origin": "origin/",
                "snapshots": "snapshots/",
                "artifacts": "artifacts/",
                "events": "events.sqlite3",
                "conversation": "conversation.sqlite3",
            },
        }

    def save(self, *, reuse_substrate_generation: bool = False) -> None:
        self.updated_at = _iso_now()
        self.engine_path.mkdir(parents=True, exist_ok=True)
        self._drain_packed_stability_events()
        self._ensure_optimizer_resident()
        if not reuse_substrate_generation:
            for assembly in self.memory.assemblies:
                if isinstance(assembly, dict):
                    assembly.pop("source_text", None)
        previous_substrate_pointer = (
            dict(self.memory.persistence_manifest)
            if isinstance(self.memory.persistence_manifest, Mapping)
            else None
        )
        previous_mutable_pointer = (
            dict(self.mutable_state_manifest)
            if isinstance(self.mutable_state_manifest, Mapping)
            else None
        )
        runtime_card_override: Optional[Dict[str, Any]] = None
        if reuse_substrate_generation:
            persisted_engine = read_json(self.engine_path / "brain.json")
            persisted_substrate = read_json(
                self.engine_path / "substrate" / "manifest.json"
            )
            if (
                not isinstance(self.memory.persistence_manifest, Mapping)
                or persisted_substrate != self.memory.persistence_manifest
            ):
                raise RuntimeError(
                    "cannot reuse an uncommitted substrate generation"
                )
            persisted_runtime_card = persisted_engine.get("runtime_card")
            if not isinstance(persisted_runtime_card, Mapping):
                raise RuntimeError(
                    "cannot reuse substrate without a committed runtime card"
                )
            runtime_card_override = self._fresh_attention_runtime_card(
                persisted_runtime_card
            )
        else:
            # The bounded, content-addressed substrate generation is complete
            # before metadata can point at it. Unchanged shard blobs are reused.
            self.memory.save_sharded(
                self.engine_path / "substrate",
                disk_reserve=self.resource_policy.require_disk,
            )
        core = self._core_tensors()
        plasticity = self._plastic_tensors()
        try:
            pointer = self.state_store.stage_generation(
                core=core,
                plasticity=plasticity,
                optimizer_state=_clone_state_to_cpu(
                    self._optimizer.state_dict()
                ),
                replay=self.replay,
                metadata={
                    "schemaVersion": ENGINE_SCHEMA_VERSION,
                    "expertCount": self.decoder.expert_count,
                    "parameterChecksum": self.parameter_checksum(),
                    "substrateContentSha256": (
                        self.memory.persistence_manifest or {}
                    ).get("contentSha256"),
                    "trainingSteps": self.counters["training_steps"],
                },
            )
            # The immutable blobs exist before compatibility paths change. If
            # the process dies before brain.json is replaced, the old metadata
            # still names its old generation and load restores those blobs.
            self.state_store.materialize(pointer, self.engine_path)
            self.mutable_state_manifest = dict(pointer)
            self.resource_pause = None
            self._bind_ingestion_checkpoint_generation(pointer)
            # The neural engine owns committed human/brain messages and
            # measured traces. The desktop ledger is only a stable-ID paged
            # presentation projection plus host action lifecycle; it never
            # invents or renumbers these rows.
            self.conversation.backfill(self.messages, self.traces)
            atomic_write_json(
                self.engine_path / "brain.json",
                self._metadata(runtime_card_override=runtime_card_override),
            )
            self.state_store.publish(pointer)
            self.state_store.last_recovery = {
                "recovered": False,
                "reason": "checkpoint generation committed",
                "activeGeneration": pointer["activeGeneration"],
            }
            # Both authoritative pointers now name the new generation. Keep
            # one immediately previous recovery point and reclaim only blobs
            # that no current/prior checkpoint can reach. This is serialized
            # recovery garbage, never learned neural state.
            self.state_store.prune_unreferenced(
                [pointer, previous_mutable_pointer]
            )
            if not reuse_substrate_generation:
                self._substrate_gc_status = self.memory.prune_unreferenced(
                    self.engine_path / "substrate",
                    [self.memory.persistence_manifest, previous_substrate_pointer],
                )
            if not reuse_substrate_generation:
                try:
                    self._maintain_neural_state_resources()
                except NeuralStateResourcePause as pressure_error:
                    # The authoritative checkpoint is already committed. A racing
                    # reserve change can prevent optional pressure scratch without
                    # making that completed checkpoint a failure.
                    self.resource_pause = {
                        "reason": str(pressure_error),
                        "readings": pressure_error.status,
                        "at": _iso_now(),
                    }
            active_epoch = self._attention_epoch()
            runtime_rows = max(32, min(1000, int(self.config.max_seq_len)))
            self.messages = self.conversation.recent_payloads(
                "message", runtime_rows, attention_epoch=active_epoch
            )
            self.traces = self.conversation.recent_payloads(
                "trace", min(200, runtime_rows), attention_epoch=active_epoch
            )
        except NeuralStateResourcePause as error:
            self.resource_pause = {
                "reason": str(error),
                "readings": error.status,
                "at": _iso_now(),
            }
            raise

    def _ternary_export_roots(self) -> Dict[str, nn.Module]:
        roots: Dict[str, nn.Module] = {
            "decoder": self.decoder,
            "memory_bridge": self.memory_bridge,
            "idea_adapter": self.idea_adapter,
            "router": self.router,
            "liquid": self.liquid,
            "modalities": self.modalities,
        }
        return roots

    def packed_runtime_audit(self) -> Dict[str, Any]:
        roots = self._ternary_export_roots()
        linear_blockers: List[str] = []
        convolution_blockers: List[str] = []
        embedding_blockers: List[str] = []
        floating_parameter_blockers: List[str] = []
        master_blockers: List[str] = []
        dense_bf16_linear_materialized = False
        packed_linear_modules = 0
        packed_convolution_modules = 0
        authoritative_linear_modules = 0
        authoritative_convolution_modules = 0
        authoritative_embedding_modules = 0
        for root_name, root in roots.items():
            status = packed_runtime_status(root)
            packed_linear_modules += int(status["packedBitLinearModules"])
            packed_convolution_modules += int(
                status["packedConvolutionModules"]
            )
            authoritative_linear_modules += int(
                status.get("packedAuthoritativeLinearModules", 0)
            )
            authoritative_convolution_modules += int(
                status.get("packedAuthoritativeConvolutionModules", 0)
            )
            authoritative_embedding_modules += int(
                status.get("packedAuthoritativeEmbeddingModules", 0)
            )
            dense_bf16_linear_materialized |= bool(
                status["denseBf16LinearWeightMaterialized"]
            )
            linear_blockers.extend(
                "%s.%s" % (root_name, name)
                for name in status["denseLinearBlockers"]
            )
            convolution_blockers.extend(
                "%s.%s" % (root_name, name)
                for name in status["denseConvolutionBlockers"]
            )
            embedding_blockers.extend(
                "%s.%s" % (root_name, name)
                for name in status.get("denseEmbeddingBlockers", ())
            )
            floating_parameter_blockers.extend(
                "%s.%s" % (root_name, name)
                for name in status.get("floatingLearnedParameterBlockers", ())
            )
            master_blockers.extend(
                "%s.%s" % (root_name, name)
                for name in status.get("residentFloatMasterBlockers", ())
            )
        return {
            "format": "omni-native-packed-runtime-audit",
            "formatVersion": 1,
            "complete": not linear_blockers
            and not convolution_blockers
            and not embedding_blockers
            and not floating_parameter_blockers
            and not master_blockers,
            "weightFormat": "two-bit-packed-ternary",
            "activationFormat": "signed-int8-dynamic-scale",
            "packedBitLinearModules": packed_linear_modules,
            "packedConvolutionModules": packed_convolution_modules,
            "packedAuthoritativeLinearModules": authoritative_linear_modules,
            "packedAuthoritativeConvolutionModules": authoritative_convolution_modules,
            "packedAuthoritativeEmbeddingModules": authoritative_embedding_modules,
            "denseLinearBlockers": sorted(linear_blockers),
            "denseConvolutionBlockers": sorted(convolution_blockers),
            "denseEmbeddingBlockers": sorted(embedding_blockers),
            "floatingLearnedParameterBlockers": sorted(floating_parameter_blockers),
            "residentFloatMasterBlockers": sorted(master_blockers),
            "denseBf16LinearWeightMaterialized": dense_bf16_linear_materialized,
        }

    def require_complete_packed_runtime(self) -> Dict[str, Any]:
        status = self.packed_runtime_audit()
        if status["complete"] is not True:
            raise RuntimeError(
                "final packed runtime is incomplete: "
                + ", ".join(
                    [
                        *status["denseLinearBlockers"],
                        *status["denseConvolutionBlockers"],
                        *status["denseEmbeddingBlockers"],
                        *status["floatingLearnedParameterBlockers"],
                        *status["residentFloatMasterBlockers"],
                    ]
                )
            )
        return status

    def _dynamic_synapse_export(self) -> Tuple[List[str], torch.Tensor]:
        if isinstance(self.memory.synapses, LazyPersistedSynapses):
            return self.memory.synapses.dynamic_export()
        synapse_ids = sorted(self.memory.synapses)
        values = torch.tensor(
            [
                self.memory.exact_effective_weight(
                    self.memory.synapses[synapse_id]["effective_weight"]
                )
                for synapse_id in synapse_ids
            ],
            dtype=torch.int8,
        )
        return synapse_ids, values

    def _dynamic_synapse_pack_state(
        self,
    ) -> Tuple[torch.Tensor, int, str, str]:
        """Return exact packed state without paging indexed cold records."""

        if isinstance(self.memory.synapses, LazyPersistedSynapses):
            return self.memory.synapses.dynamic_pack_state()
        synapse_ids, values = self._dynamic_synapse_export()
        return (
            values,
            len(synapse_ids),
            hashlib.sha256("\0".join(synapse_ids).encode("utf-8")).hexdigest(),
            "record-id-lexicographic-v1",
        )

    def _dynamic_ternary_tensors(
        self,
        substrate_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """All non-module synapses eligible for packed forward inference."""

        if substrate_values is None:
            _synapse_ids, substrate_values = self._dynamic_synapse_export()
        return {"substrate.dynamic_synapses.weights": substrate_values}

    def export_packed_ternary(
        self, destination: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Materialize and verify the authoritative 2-bit synapse shards.

        Native learning changes these packed weights directly. The sidecar
        covers eligible projections, learned tables, and dynamic synapses;
        small non-synaptic controls remain higher precision in core tensors.
        """

        roots = self._ternary_export_roots()
        self.require_complete_packed_runtime()
        dense_types = (
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        dense_violations: List[str] = []
        for root_name, root in roots.items():
            for module_name, module in root.named_modules():
                if isinstance(module, TERNARY_PROJECTION_TYPES):
                    continue
                if isinstance(module, dense_types):
                    dense_violations.append(
                        "%s.%s" % (root_name, module_name or "<root>")
                    )
        if dense_violations:
            raise RuntimeError(
                "eligible dense forward projections cannot be packed: %s"
                % ", ".join(sorted(dense_violations))
            )
        audit = self._ternary_audit()
        if audit["violations"] or audit["coverage"] != 1.0:
            raise RuntimeError("ternary coverage audit failed before packing")

        (
            dynamic_values,
            dynamic_synapse_count,
            synapse_order_hash,
            dynamic_order_basis,
        ) = self._dynamic_synapse_pack_state()
        dynamic = self._dynamic_ternary_tensors(dynamic_values)
        specs = collect_module_ternary_tensors(
            roots,
            dynamic_synapses=dynamic,
        )
        expected_names = [spec.name for spec in specs]
        packed_write_bytes = sum(
            (int(spec.values.numel()) + 3) // 4
            for spec in specs
        ) + max(64 * 1024, len(specs) * 1024)
        self.resource_policy.require_disk(
            packed_write_bytes,
            "packed ternary export",
        )
        target = (
            Path(destination).resolve()
            if destination is not None
            else (self.engine_path / "packed-ternary").resolve()
        )
        if target == self.engine_path.resolve():
            raise ValueError("packed ternary destination cannot be the engine root")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / (
            ".%s-%s.next" % (target.name, uuid.uuid4().hex)
        )
        previous = target.parent / (
            ".%s-%s.previous" % (target.name, uuid.uuid4().hex)
        )
        replaced_previous = False
        try:
            manifest = export_module_ternary_shards(
                temporary,
                roots,
                dynamic_synapses=dynamic,
                expected_names=expected_names,
                metadata={
                    "brainId": self.brain_id,
                    "engineSchemaVersion": ENGINE_SCHEMA_VERSION,
                    "parameterChecksum": self.parameter_checksum(),
                    "originKind": self.config.origin_kind,
                    "baseFrozen": False,
                    "randomInitializationSeed": int(self.config.seed),
                    "groundUpCurriculumSha256": (
                        (self.ground_up_training_manifest or {}).get("sha256")
                        if self.config.origin_kind == "ground-up"
                        else None
                    ),
                    "groundUpTrainingManifestSha256": (
                        (self.ground_up_training_manifest or {}).get(
                            "contentSha256"
                        )
                        if self.config.origin_kind == "ground-up"
                        else None
                    ),
                    "groundUpTrainingReceiptSha256": (
                        (
                            (self.ground_up_training_manifest or {}).get(
                                "trainingReceipt", {}
                            )
                            or {}
                        ).get("contentSha256")
                        if self.config.origin_kind == "ground-up"
                        else None
                    ),
                    "dynamicSynapseCount": dynamic_synapse_count,
                    "dynamicSynapseOrderSha256": synapse_order_hash,
                    "dynamicSynapseOrderBasis": dynamic_order_basis,
                    "substrateContentSha256": str(
                        (self.memory.persistence_manifest or {}).get(
                            "contentSha256", ""
                        )
                    ),
                    "totalDynamicSynapseCount": dynamic_synapse_count,
                    "dynamicSynapseCountingBasis": "substrate-records-v1",
                    "nativePackedForward": audit["packedExecution"],
                },
            )
            verify_ternary_shards(
                temporary,
                expected_names=expected_names,
                retain_names=(),
            )
            if target.exists():
                os.replace(str(target), str(previous))
                replaced_previous = True
            os.replace(str(temporary), str(target))
            if replaced_previous:
                shutil.rmtree(previous)
                replaced_previous = False
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            if replaced_previous and previous.exists() and not target.exists():
                os.replace(str(previous), str(target))
                replaced_previous = False
            raise
        finally:
            if previous.exists():
                shutil.rmtree(previous)

        summary = {
            "format": manifest["format"],
            "formatVersion": manifest["formatVersion"],
            "contentSha256": manifest["contentSha256"],
            "eligibleTensorCount": manifest["coverage"][
                "eligibleTensorCount"
            ],
            "dynamicSynapseCount": dynamic_synapse_count,
            "dynamicSynapseOrderSha256": synapse_order_hash,
            "dynamicSynapseOrderBasis": dynamic_order_basis,
            "substrateContentSha256": str(
                (self.memory.persistence_manifest or {}).get(
                    "contentSha256", ""
                )
            ),
            "totalDynamicSynapseCount": dynamic_synapse_count,
            "dynamicSynapseCountingBasis": "substrate-records-v1",
            "nativePackedForward": audit["packedExecution"],
            "parameterChecksum": self.parameter_checksum(),
            "originKind": self.config.origin_kind,
            "groundUpCurriculumSha256": (
                (self.ground_up_training_manifest or {}).get("sha256")
                if self.config.origin_kind == "ground-up"
                else None
            ),
            "groundUpTrainingManifestSha256": (
                (self.ground_up_training_manifest or {}).get(
                    "contentSha256"
                )
                if self.config.origin_kind == "ground-up"
                else None
            ),
            "groundUpTrainingReceiptSha256": (
                (
                    (self.ground_up_training_manifest or {}).get(
                        "trainingReceipt", {}
                    )
                    or {}
                ).get("contentSha256")
                if self.config.origin_kind == "ground-up"
                else None
            ),
            "baseFrozen": False,
            "pretrainedTextCortex": None,
            "relativePath": "packed-ternary/",
        }
        self.packed_ternary_manifest = summary
        self.updated_at = _iso_now()
        atomic_write_json(self.engine_path / "brain.json", self._metadata())
        return {
            "brainId": self.brain_id,
            "path": str(target),
            "summary": summary,
            "manifest": manifest,
        }

    def _ternary_audit(self) -> Dict[str, Any]:
        eligible = []
        violations = []
        observed_levels = set()

        def audit_tensor(name: str, values: torch.Tensor) -> None:
            invalid = False
            for value in torch.unique(values.detach()).cpu().tolist():
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError):
                    invalid = True
                    continue
                if not math.isfinite(numeric) or numeric not in {
                    -1.0,
                    0.0,
                    1.0,
                }:
                    invalid = True
                    continue
                observed_levels.add(int(numeric))
            if invalid and name not in violations:
                violations.append(name)

        roots: List[Tuple[str, nn.Module]] = [
            ("decoder", self.decoder),
            ("memory_bridge", self.memory_bridge),
            ("idea_adapter", self.idea_adapter),
            ("router", self.router),
            ("liquid", self.liquid),
            ("modalities", self.modalities),
        ]
        for root_name, root in roots:
            for module_name, module in root.named_modules():
                if not (
                    isinstance(module, TERNARY_PROJECTION_TYPES)
                    or module is self.router.synapses
                ):
                    continue
                name = "%s.%s" % (root_name, module_name or "<root>")
                eligible.append(name)
                if getattr(module, "ternary", False) is not True:
                    violations.append(name)
                audit_tensor(name, module.effective_weight())
        dynamic_name = "substrate.dynamic_synapses.weights"
        eligible.append(dynamic_name)
        dynamic_levels = set()
        if isinstance(self.memory.synapses, LazyPersistedSynapses):
            try:
                self.memory.synapses.validate_dirty()
                dynamic_levels.update(
                    self.memory.synapses.observed_effective_levels()
                )
            except ValueError:
                violations.append(dynamic_name)
        else:
            for synapse in self.memory.synapses.values():
                try:
                    dynamic_levels.add(
                        self.memory.exact_effective_weight(
                            synapse.get("effective_weight", 0)
                        )
                    )
                except ValueError:
                    if dynamic_name not in violations:
                        violations.append(dynamic_name)
        observed_levels.update(dynamic_levels)
        return {
            "eligibleProjections": len(eligible),
            "ternaryProjections": len(eligible) - len(violations),
            "coverage": (
                1.0
                if not eligible
                else (len(eligible) - len(violations)) / float(len(eligible))
            ),
            "violations": violations,
            "observedLevels": sorted(observed_levels),
            "authoritativeWeightStorage": "packed-ternary-synapses",
            "forwardPrecision": "exact ternary {-1,0,+1}",
            "packedExecution": self.packed_runtime_audit(),
        }

    def _organic_state(self) -> Dict[str, Any]:
        from .paged_neuron_metadata import PagedNeuronMetadata

        estimated_activity = False
        sampled_neurons = 0
        if isinstance(self.memory.neurons, PagedNeuronMetadata):
            measured = self.memory.neurons.activity_metrics(
                active_ids=self.memory.attention_active_neuron_ids,
                legacy_raw_active=self.memory.attention_legacy_raw_active,
                threshold=0.1,
                sample_rows=512,
            )
            uncertainty = float(measured["meanUncertainty"])
            active = float(measured["activeFraction"])
            estimated_activity = bool(measured["estimated"])
            sampled_neurons = int(measured["sampledRows"])
        else:
            count = 0
            uncertainty_total = 0.0
            active_count = 0
            for item in self.memory.neurons.values():
                count += 1
                uncertainty_total += float(item.get("uncertainty", 0.5))
                active_count += int(
                    self.memory.effective_activation(item) >= 0.1
                )
            uncertainty = uncertainty_total / count if count else 0.5
            active = active_count / count if count else 0.0
            sampled_neurons = count
        attention_epoch = self._attention_epoch()
        attention_traces = [
            trace
            for trace in self.traces
            if (
                trace.get("attention_epoch", trace.get("attentionEpoch", 0))
                == attention_epoch
            )
        ]
        recent_losses = [
            float(trace.get("train_loss", 0.0))
            for trace in attention_traces[-2:]
            if math.isfinite(float(trace.get("train_loss", 0.0)))
        ]
        prediction_error = (
            recent_losses[-1] / (1.0 + abs(recent_losses[-1]))
            if recent_losses
            else 0.5
        )
        learning_progress = (
            max(0.0, recent_losses[-2] - recent_losses[-1])
            / (1.0 + abs(recent_losses[-2]))
            if len(recent_losses) >= 2
            else 0.0
        )
        recent = attention_traces[-1] if attention_traces else {}
        novelty = float(
            recent.get("organic_state", {}).get(
                "novelty",
                recent.get("ponder_factors", {}).get("novelty", 0.5),
            )
        )
        novelty = max(0.0, min(1.0, novelty))
        tension = max(
            0.0,
            min(
                1.0,
                0.38 * prediction_error
                + 0.28 * uncertainty
                + 0.22 * novelty
                + 0.12 * (1.0 - active),
            ),
        )
        curiosity = max(
            0.0,
            min(
                1.0,
                0.42 * novelty
                + 0.34 * prediction_error
                + 0.18 * uncertainty
                + 0.06 * learning_progress,
            ),
        )
        return {
            "novelty": novelty,
            "uncertainty": uncertainty,
            "predictionError": prediction_error,
            "learningProgress": learning_progress,
            "activeFraction": active,
            "activityEstimated": estimated_activity,
            "sampledNeurons": sampled_neurons,
            "tension": tension,
            "curiosity": curiosity,
        }

    def runtime_card(self) -> Dict[str, Any]:
        card: Dict[str, Any] = {
            "architecture": "OmniCortex",
            "pretrained": False,
            "trained": bool(
                self.ground_up_training_manifest
                or self.counters["training_steps"] > 0
                or self.training_sources
            ),
            "origin_kind": self.config.origin_kind,
            "baseFrozen": False,
            "ground_up_training_manifest": self.ground_up_training_manifest,
            "packed_ternary_manifest": self.packed_ternary_manifest,
            "hidden_behavioral_prompt": False,
            "reward_model": False,
            "rlhf": False,
            "memory_injection": self.config.memory_injection,
            "textual_long_term_memory_injected": False,
            "fresh_attention": (
                None
                if self.fresh_attention_boundary is None
                else {
                    "active": True,
                    "epoch": int(self.fresh_attention_boundary["epoch"]),
                    "createdAt": str(
                        self.fresh_attention_boundary["createdAt"]
                    ),
                    "messagesPreserved": int(
                        self.fresh_attention_boundary["messagesPreserved"]
                    ),
                    "rawPriorDialogueEligible": False,
                    "cleared": dict(
                        self.fresh_attention_boundary["cleared"]
                    ),
                }
            ),
            "tokenizer_boundary": "UTF-8 bytes",
            "weight_forward": "scaled ternary {-1,0,+1}",
            "parameterAccounting": self.parameter_accounting(),
            "ternary_audit": self._ternary_audit(),
            "device": str(self.device),
            "device_backend": self.device_backend,
            "working_tokens": self.config.max_seq_len,
            "working_memory_vectors": len(self.working_memory),
            "workspace": self.workspace_snapshot(),
            "expert_count": self.decoder.expert_count,
            "hardware_tier": self.config.hardware_tier,
            "scale": {
                "dimensions": self.config.d_model,
                "layers": self.config.n_layers,
                "contextTokens": self.config.max_seq_len,
                "recurrentPagedMemoryItems": self.config.working_memory_slots,
                "workingMemoryMode": self.config.working_memory_mode,
                "denseAttentionClaim": False,
                "configuredMemorySpillBytes": self.config.memory_offload_bytes,
                "estimatedStorageSlowdownPercent": (
                    self.config.memory_offload_slowdown_percent
                ),
                "trainBatchSize": self.config.train_batch_size,
                "gradientAccumulation": self.config.gradient_accumulation,
                "gradientCheckpointing": self.config.gradient_checkpointing,
                "replayOffload": "transactional-sqlite-disk",
                "optimizerOffload": "safe-tensor-pressure-scratch",
                "imageSize": self.config.image_size,
                "audioSamples": self.config.audio_samples,
                "videoFrames": self.config.video_frames,
            },
            "active_modules": {
                "ternary": True,
                "dense": False,
                "spiking": self.config.spiking_dynamics,
                "stdp": self.config.spiking_dynamics
                and self.config.stdp_plasticity,
                "liquid": self.config.liquid_dynamics,
                "vsa": self.config.vector_symbolic_memory,
                "onlineLearning": self.config.online_learning,
                "consolidation": self.config.consolidation_enabled,
                "metaplasticity": self.config.metaplasticity,
                "gradientCheckpointing": self.config.gradient_checkpointing,
            },
            "enabled_modalities": [
                name
                for name, enabled in (
                    ("vision", self.config.vision_enabled),
                    ("image", self.config.image_enabled),
                    ("audio", self.config.audio_enabled),
                    ("video", self.config.video_enabled),
                )
                if enabled
            ],
            "growth": {
                "policy": "resource-governed-unbounded",
                "experts": self.decoder.expert_count,
                "elasticLimit": None,
                "paused": self.growth_pause is not None,
                "pause": self.growth_pause,
                "substrate": {
                    "neurons": len(self.memory.neurons),
                    "assemblies": len(self.memory.assemblies),
                    "synapses": len(self.memory.synapses),
                    "growthEvents": self.memory.growth_events,
                    "growthPauses": self.memory.growth_pauses,
                    "cardinalityLimit": None,
                },
            },
            "memory_timescales": {
                "workingMemorySlots": self.config.working_memory_slots,
                "shortTermHalfLifeMinutes": self.config.short_term_half_life_minutes,
                "replayAdmission": "continuous-organic-weighted",
                "forgettingRate": self.config.forgetting_rate,
                "consolidationRate": self.config.consolidation_rate,
                "activeWorkingVectors": len(self.working_memory),
                "injectionChannel": (
                    "internal-recurrent-vectors"
                    if self.config.memory_injection == "working-memory"
                    else "semantic-parameters-and-vsa"
                ),
            },
            "state_offload": self._state_offload_status(),
            "modality_training": {
                name: {
                    "steps": steps,
                    "initialized": (
                        "trained"
                        if steps > 0
                        else (
                            "random"
                        )
                    ),
                }
                for name, steps in self.modality_training.items()
            },
            "installed_modality_packs": [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "modalities": list(item.get("modalities", [])),
                    "sha256": item.get("sha256"),
                    "license": item.get("license"),
                }
                for item in self.installed_modality_packs
            ],
            "intrinsic_dynamics": self._organic_state(),
        }
        if self.config.origin_kind == "ground-up":
            initialization = (
                self.ground_up_training_manifest or {}
            ).get("randomInitialization", {})
            card["randomInitialization"] = {
                "algorithm": initialization.get(
                    "algorithm", "torch-seeded-module-initialization-v1"
                ),
                "seed": int(self.config.seed),
                "parameterChecksum": initialization.get("parameterChecksum"),
                "exactParameterCount": initialization.get(
                    "exactParameterCount",
                    self.parameter_accounting()["totalNeuralParameters"],
                ),
            }
        return card

    def parameter_checksum(self) -> str:
        return tensor_checksum(
            self._learned_parameter_tensors(self._trainable_modules())
        )

    def _parameter_copy(self) -> List[torch.Tensor]:
        modules: List[nn.Module] = [
            self.decoder,
            self.memory_bridge,
            self.idea_adapter,
            self.liquid,
        ]
        return [
            parameter.detach().cpu().clone()
            for parameter in self._learned_parameter_tensors(modules)
        ]

    def _parameter_delta_norm(self, before: Sequence[torch.Tensor]) -> float:
        total = 0.0
        modules: List[nn.Module] = [
            self.decoder,
            self.memory_bridge,
            self.idea_adapter,
            self.liquid,
        ]
        current = [
            parameter.detach().cpu()
            for parameter in self._learned_parameter_tensors(modules)
        ]
        for index, parameter in enumerate(current):
            if (
                index < len(before)
                and parameter.shape == before[index].shape
                and parameter.dtype == torch.uint8
                and before[index].dtype == torch.uint8
            ):
                # Four signed ternary codes live in each byte. Decode only a
                # bounded byte block for an exact mutation norm; never create
                # a full dense copy of a packed cortical matrix.
                current_bytes = parameter.reshape(-1)
                previous_bytes = before[index].reshape(-1)
                for offset in range(0, current_bytes.numel(), 65_536):
                    now = current_bytes[offset : offset + 65_536].to(torch.int16)
                    prior = previous_bytes[offset : offset + 65_536].to(torch.int16)
                    for shift in (0, 2, 4, 6):
                        delta = ((now >> shift) & 3) - ((prior >> shift) & 3)
                        total += float(delta.float().pow(2).sum().item())
                continue
            if index < len(before) and parameter.shape == before[index].shape:
                difference = parameter.float() - before[index].float()
            else:
                difference = parameter.float()
            total += float(difference.pow(2).sum().item())
        return math.sqrt(total)

    def _idea_model_vector(self, vsa_vector: torch.Tensor) -> torch.Tensor:
        raw = vsa_vector.to(self.device, dtype=torch.float32).reshape(1, -1)
        return torch.tanh(self.memory_bridge(raw))

    @staticmethod
    def _replay_admission_probability(
        importance: float, replay_priority: float = 0.0
    ) -> float:
        """Sample cold latent replay smoothly; never gate fast learning.

        The small floor gives a weak one-off experience a chance to enter
        latent storage for later replay,
        while recurrence and other measured retention signals can raise its
        chance on subsequent encounters. Sampling avoids one SQLite commit
        and WAL checkpoint for every section of a large document.
        """

        def bounded(value: float) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0

        attention = bounded(importance) ** 3
        organic = bounded(replay_priority)
        return 0.02 + 0.98 * (attention + organic - attention * organic)

    def _organic_replay_priority(
        self,
        *,
        assembly_id: str,
        salience: float,
        novelty: float,
        prediction_error: float,
        spike_rate: float,
    ) -> float:
        """Measure current organic signals without advancing lifecycle state.

        Replay must be admitted before settle can mutate afterimages or slow
        anchors, preserving the existing disk-reserve pause boundary. This
        read-only pre-settle estimate is rescored by settle immediately after.
        """

        signals, rehearsals, _ = self.memory_lifecycle._measure_signals(
            self.memory,
            str(assembly_id),
            salience_boost=salience,
            spike_rate=spike_rate,
            record_index=self.memory.assembly_by_id,
        )
        return self._replay_priority_from_signals(
            signals,
            rehearsals=rehearsals,
            novelty=novelty,
            prediction_error=prediction_error,
        )

    @staticmethod
    def _replay_priority_from_signals(
        signals: Mapping[str, Any],
        *,
        rehearsals: int,
        novelty: float,
        prediction_error: float,
    ) -> float:
        assessment = OrganicMemoryLifecycle.assess_retention_candidate(
            novelty=novelty,
            reuse=float(signals.get("reuse", 0.0)),
            salience=float(signals.get("salience", 0.0)),
            prediction_error=prediction_error,
            rehearsals=rehearsals,
            stability=float(signals.get("stability", 0.0)),
            related_coactivation=float(signals.get("recurrence", 0.0)),
            interference=float(signals.get("interference", 0.0)),
            recurrence=float(signals.get("recurrence", 0.0)),
            activation=float(signals.get("activation", 0.0)),
            observations=rehearsals,
        )
        return float(assessment["slowReplayPriority"])

    def _append_replay(
        self,
        idea: torch.Tensor,
        importance: float = 1.0,
        *,
        replay_priority: float = 0.0,
        assembly_id: str = "",
    ) -> bool:
        probability = self._replay_admission_probability(
            importance, replay_priority
        )
        if probability < 1.0:
            # Cycle and counters are checkpointed with the neural state. A
            # retry from the same boundary chooses the same lane, while a
            # later related exposure gets a fresh opportunity without keeping
            # source text or a cue-to-answer key.
            seed = "\0".join(
                (
                    str(self.brain_id),
                    str(self.memory_lifecycle.cycle),
                    str(self.counters.get("experiences", 0)),
                    str(self.counters.get("idle_cognition_cycles", 0)),
                    str(assembly_id),
                )
            ).encode("utf-8")
            draw = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") / float(
                1 << 64
            )
            if draw >= probability:
                return False
        try:
            self.replay.append(idea.detach().cpu().reshape(-1))
        except NeuralStateResourcePause as error:
            self.resource_pause = {
                "reason": str(error),
                "readings": error.status,
                "at": _iso_now(),
            }
            raise
        return True

    def _append_selected_replay_batch(
        self, ideas: Iterable[torch.Tensor]
    ) -> None:
        """Persist already-admitted replay vectors in one microbatch commit."""

        if not isinstance(self.replay, DurableReplayBuffer):
            # Lightweight replay test doubles retain their existing append
            # contract; production replay takes the atomic batch path.
            for idea in ideas:
                self._append_replay(idea)
            return
        try:
            self.replay.append_many(
                idea.detach().cpu().reshape(-1) for idea in ideas
            )
        except NeuralStateResourcePause as error:
            self.resource_pause = {
                "reason": str(error),
                "readings": error.status,
                "at": _iso_now(),
            }
            raise

    def _append_working_memory(
        self,
        idea: torch.Tensor,
        *,
        assembly_id: str = "",
        source: str = "experience",
        salience: float = 0.5,
    ) -> None:
        vector = idea.detach().cpu().reshape(-1)
        timestamp = _iso_now()
        salience = max(0.0, min(1.0, float(salience)))
        assembly_key = str(assembly_id)
        restored_item: Dict[str, Any] = {}

        def near_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
            first = left.float().reshape(-1)
            second = right.float().reshape(-1)
            if first.shape != second.shape:
                return False
            similarity = float(
                F.cosine_similarity(
                    first.reshape(1, -1), second.reshape(1, -1)
                ).item()
            )
            return (
                math.isfinite(similarity)
                and similarity >= self.memory_lifecycle.EPISODE_MERGE_COSINE
            )

        if assembly_key:
            preview = self.paged_working_memory.peek_hot(
                hot_assembly_ids=(assembly_key,),
                unfinished_ids=(),
                limit=1,
            )
            restored = False
            if preview and near_identical(preview[0][0], vector):
                try:
                    restored_vector, restored_metadata = (
                        self.paged_working_memory.page_in(
                            str(preview[0][1]["pageId"])
                        )
                    )
                except KeyError:
                    # A competing page-in won; the current vector still gets
                    # its own resident episode below.
                    pass
                else:
                    restored = True
                    restored_item = dict(restored_metadata)
                    vector = (
                        restored_vector.detach().cpu().float().reshape(-1) * 0.72
                        + vector.float() * 0.28
                    )
                    salience = max(
                        salience,
                        float(restored_item.get("salience", 0.0) or 0.0),
                    )
            self.hot_state_residency.note_access(
                (assembly_key,), page_in=restored
            )
        for index, existing in enumerate(self.working_memory):
            item = self.workspace_items[index]
            if not near_identical(existing, vector):
                continue
            self.working_memory[index] = (
                existing.float() * 0.72 + vector.float() * 0.28
            )
            item["salience"] = min(
                1.0, float(item.get("salience", 0.5)) * 0.8 + salience * 0.2
            )
            item["rehearsals"] = (
                int(item.get("rehearsals", 1))
                + int(restored_item.get("rehearsals", 0) or 0)
                + 1
            )
            item["lastActiveAt"] = timestamp
            self.counters["workspace_rehearsals"] += 1
            return

        for item in self.workspace_items:
            item["salience"] = max(
                0.0, float(item.get("salience", 0.5)) * 0.97
            )
        self.working_memory.append(vector)
        self.workspace_items.append(
            {
                "id": str(restored_item.get("id", "")) or uuid.uuid4().hex,
                "assemblyId": assembly_key,
                "source": source,
                "salience": salience,
                "rehearsals": int(
                    restored_item.get("rehearsals", 0) or 0
                )
                + 1,
                "enteredAt": str(restored_item.get("enteredAt", ""))
                or timestamp,
                "lastActiveAt": timestamp,
            }
        )
        resident_limit = max(
            1,
            min(
                self.config.working_memory_slots,
                self.config.memory_resident_items,
            ),
        )
        while len(self.working_memory) > resident_limit:
            eviction_candidates = [
                index
                for index, item in enumerate(self.workspace_items)
                if not assembly_key
                or str(item.get("assemblyId", "")) != assembly_key
            ]
            if not eviction_candidates:
                eviction_candidates = list(range(len(self.workspace_items)))
            eviction = min(
                eviction_candidates,
                key=lambda index: (
                    float(self.workspace_items[index].get("salience", 0.0))
                    * (
                        1.0
                        + math.log1p(
                            int(self.workspace_items[index].get("rehearsals", 1))
                        )
                    ),
                    self.workspace_items[index].get("lastActiveAt", ""),
                ),
            )
            cold_vector = self.working_memory.pop(eviction)
            cold_item = self.workspace_items.pop(eviction)
            self.paged_working_memory.append(cold_vector, cold_item)
            cold_capacity = max(
                0, self.config.working_memory_slots - resident_limit
            )
            self.counters["workspace_evictions"] += (
                self.paged_working_memory.trim_to(cold_capacity)
            )
            self.counters["workspace_evictions"] += 1

    @staticmethod
    def _prediction_error_from_loss(value: Any) -> float:
        try:
            loss = max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(loss):
            return 1.0
        return loss / (1.0 + loss)

    def _settle_memory_automatically(
        self,
        vector: torch.Tensor,
        *,
        assembly_id: str,
        source: str,
        salience: float,
        novelty: float,
        prediction_error: float,
        importance: float,
        spike_rate: float,
        resting: bool = False,
    ) -> Dict[str, Any]:
        """Continuously settle activity without a manual memory command.

        Settling is not Ponder and cannot create visible text or actions.  An
        unfinished trace may quietly re-enter working activity, where the
        ordinary learned action head can later decide whether any deliberate
        Ponder, talk, imagination, or tool action is warranted.
        """

        result = self.memory_lifecycle.settle(
            memory=self.memory,
            router=self.router,
            vector=vector,
            assembly_id=str(assembly_id),
            source=str(source),
            salience=float(salience),
            novelty=float(novelty),
            prediction_error=float(prediction_error),
            importance=float(importance),
            spike_rate=float(spike_rate),
            forgetting_rate=self.config.forgetting_rate,
            long_term_threshold=self.config.long_term_threshold,
            resting=resting,
        )
        reinforcement_drive = float(result.get("reinforcementDrive", 0.0))
        if reinforcement_drive > 0.0:
            # Current slow weights were already changed by ordinary corpus or
            # modality learning. Continuous measured reinforcement increases
            # metaplastic anchoring around that learned state; no categorical
            # memory stage controls whether an experience can keep changing.
            stability_rate = min(0.22, 0.18 * reinforcement_drive)
            self._commit_slow_anchors(rate=stability_rate)
            result["slowWeightStabilityRate"] = stability_rate
        else:
            result["slowWeightStabilityRate"] = 0.0
        recurring_id = str(result.get("recurringAssemblyId", ""))
        if recurring_id:
            for index, item in enumerate(
                self.memory_lifecycle.afterimage_items
            ):
                if str(item.get("assemblyId", "")) != recurring_id:
                    continue
                item["lastActiveCycle"] = self.memory_lifecycle.cycle
                self._append_working_memory(
                    self.memory_lifecycle.afterimage_vectors[index],
                    assembly_id=recurring_id,
                    source="unfinished-memory",
                    salience=max(0.35, float(item.get("salience", 0.35))),
                )
                result["unfinishedReenteredWorkingMemory"] = True
                break
        result.setdefault("unfinishedReenteredWorkingMemory", False)
        return result

    def _active_working_memory_vector(self) -> Optional[torch.Tensor]:
        """Return the measured current idea mixture independent of text policy."""

        if not self.working_memory:
            return None
        recent = torch.stack(self.working_memory).to(
            self.device, dtype=torch.float32
        )
        salience = torch.tensor(
            [
                max(0.01, float(item.get("salience", 0.5)))
                * (1.0 + 0.1 * math.log1p(int(item.get("rehearsals", 1))))
                for item in self.workspace_items
            ],
            dtype=torch.float32,
            device=self.device,
        )
        recency = torch.linspace(
            0.55, 1.0, recent.shape[0], device=self.device
        )
        weights = salience * recency
        weights = weights / weights.sum()
        return torch.tanh((recent * weights[:, None]).sum(dim=0, keepdim=True))

    def _working_memory_vector(self) -> Optional[torch.Tensor]:
        if self.config.memory_injection != "working-memory":
            return None
        return self._active_working_memory_vector()

    @staticmethod
    def _token_sequence_hash(tokens: Sequence[int]) -> str:
        return hashlib.sha256(
            ",".join(str(int(value)) for value in tokens).encode("ascii")
        ).hexdigest()

    def _bounded_completed_turn_tokens(
        self, human: str, brain: str
    ) -> Tuple[List[int], int]:
        """Encode one completed turn while preserving both role markers."""

        human_tokens = self.tokenizer.encode(human)
        brain_tokens = self.tokenizer.encode(brain)
        capacity = max(3, int(self.config.max_seq_len))
        payload_capacity = capacity - 3
        removed = max(
            0, len(human_tokens) + len(brain_tokens) - payload_capacity
        )
        if removed:
            # Preserve recent material from both sides of the exchange. A
            # brain response receives half the available bytes and the human
            # side receives the remainder.
            brain_budget = min(len(brain_tokens), payload_capacity // 2)
            human_budget = min(
                len(human_tokens), payload_capacity - brain_budget
            )
            unused = payload_capacity - human_budget - brain_budget
            if unused and len(brain_tokens) > brain_budget:
                add = min(unused, len(brain_tokens) - brain_budget)
                brain_budget += add
                unused -= add
            if unused and len(human_tokens) > human_budget:
                human_budget += min(unused, len(human_tokens) - human_budget)
            human_tokens = human_tokens[-human_budget:] if human_budget else []
            brain_tokens = brain_tokens[-brain_budget:] if brain_budget else []
        return (
            [self.tokenizer.human_id]
            + human_tokens
            + [self.tokenizer.brain_id]
            + brain_tokens
            + [self.tokenizer.eos_id],
            removed,
        )

    def _append_recent_dialogue(self, human: str, brain: str) -> None:
        turn, removed = self._bounded_completed_turn_tokens(human, brain)
        combined = self.recent_token_context + turn
        overflow = max(0, len(combined) - self.config.max_seq_len)
        if overflow:
            combined = combined[overflow:]
            # Avoid retaining an unlabelled fragment of an evicted old turn.
            try:
                boundary = combined.index(self.tokenizer.human_id)
            except ValueError:
                boundary = 0
            if boundary:
                combined = combined[boundary:]
                overflow += boundary
        self.recent_token_context = combined
        self.counters["context_token_evictions"] += removed + overflow

    def _prompt_with_recent_context(
        self, human: str
    ) -> Tuple[List[int], List[int]]:
        """Build a bounded prompt from explicit recent working context."""

        capacity = max(3, int(self.config.max_seq_len))
        human_payload = self.tokenizer.encode(human)
        current_budget = capacity - 3
        if len(human_payload) > current_budget:
            human_payload = human_payload[-current_budget:]
        current = (
            [self.tokenizer.human_id]
            + human_payload
            + [self.tokenizer.brain_id]
        )
        history_budget = max(0, capacity - 1 - len(current))
        history = (
            self.recent_token_context[-history_budget:]
            if history_budget
            else []
        )
        if history:
            # Use only role-labelled history. When pressure cuts into the
            # oldest retained turn, drop that fragment rather than presenting
            # it as unlabelled hidden text.
            role_boundaries = {
                self.tokenizer.human_id,
                self.tokenizer.brain_id,
            }
            while history and history[0] not in role_boundaries:
                history = history[1:]
        return [self.tokenizer.bos_id] + history + current, history

    def _fast_synapse_checksum(self) -> str:
        return tensor_checksum(
            [
                self.router.synapses.weights,
                self.router.synapses.stability,
                self.router.synapses.uses,
                self.router.synapses.plasticity_events,
            ]
        )

    def _fresh_attention_result(
        self,
        boundary: Mapping[str, Any],
        *,
        idempotent: bool,
        paged_cleanup_pending: bool = False,
    ) -> Dict[str, Any]:
        conversation_counts = self.conversation.counts()
        return {
            "format": FRESH_ATTENTION_FORMAT,
            "formatVersion": FRESH_ATTENTION_VERSION,
            "brainId": self.brain_id,
            "committed": True,
            "idempotent": bool(idempotent),
            "boundary": dict(boundary),
            "parameterChecksum": self.parameter_checksum(),
            "fastSynapseChecksum": self._fast_synapse_checksum(),
            "substrateContentSha256": str(
                (self.memory.persistence_manifest or {}).get(
                    "contentSha256", ""
                )
            ),
            "messagesPreserved": int(conversation_counts["messageCount"]),
            "tracesPreserved": int(conversation_counts["traceCount"]),
            "synapsesPreserved": len(self.memory.synapses),
            "replayEntries": len(self.replay),
            "pagedCleanupPending": bool(paged_cleanup_pending),
            "rawPriorDialogueEligible": False,
        }

    def start_fresh_attention(self, operation_id: str) -> Dict[str, Any]:
        """Atomically forget attention while preserving learned memory/history."""

        operation_id = self._validated_chat_turn_id(operation_id)
        if not operation_id:
            raise ValueError("fresh attention operation id is required")
        existing = self.fresh_attention_boundary
        if existing is not None and existing.get("operationId") == operation_id:
            persisted = read_json(self.engine_path / "brain.json")
            committed_boundary = self._validated_fresh_attention_boundary(
                persisted.get("fresh_attention_boundary")
            )
            if (
                committed_boundary is None
                or committed_boundary.get("operationId") != operation_id
            ):
                raise RuntimeError(
                    "fresh attention boundary is not atomically committed"
                )
            return self._fresh_attention_result(
                existing,
                idempotent=True,
                paged_cleanup_pending=(
                    self._fresh_attention_paged_clear_pending
                ),
            )

        parameter_checksum = self.parameter_checksum()
        fast_synapse_checksum = self._fast_synapse_checksum()
        substrate_content_sha256 = str(
            (self.memory.persistence_manifest or {}).get("contentSha256", "")
        )
        if not self._sha256_identifier(substrate_content_sha256):
            raise RuntimeError(
                "fresh attention requires a committed substrate generation"
            )
        synapse_count = len(self.memory.synapses)
        replay_entries = len(self.replay)
        prior_epoch = self._attention_epoch()
        # Persist every already-materialized row before taking the preservation
        # snapshot. The append-only ledger, rather than the bounded runtime
        # cache, is authoritative for visible history across attention epochs.
        self.conversation.backfill(self.messages, self.traces)
        conversation_counts = self.conversation.counts()
        epoch_counts = self.conversation.counts(attention_epoch=prior_epoch)
        messages_preserved = int(conversation_counts["messageCount"])
        traces_preserved = int(conversation_counts["traceCount"])
        raw_history_messages = int(epoch_counts["messageCount"])
        automatic_trace_influence = int(epoch_counts["traceCount"])
        paged_count = self.paged_working_memory.count()
        recall_audit_entries = len(
            self.memory._last_recall_audit.get("activationByAssembly", {})
        )
        cleared = {
            "recentTokens": len(self.recent_token_context),
            "currentPromptTokens": max(
                0, int(self.current_context.get("tokenCount", 0))
            ),
            "residentWorkingMemory": len(self.working_memory),
            "pagedWorkingMemory": paged_count,
            "lifecycleScratch": len(self.memory_lifecycle.afterimage_items),
            "activeFocus": len(self.memory_lifecycle.active_focus),
            "activatedNeurons": 0,
            "recalledAssemblies": 0,
            "substrateEligibilityTraces": 0,
            "liquidStateUnits": int(
                torch.count_nonzero(self.liquid_state.detach()).item()
            ),
            "rawHistoryMessages": raw_history_messages,
            "automaticTraceInfluence": automatic_trace_influence,
            "recallAuditEntries": recall_audit_entries,
            "noveltyStreak": max(0, int(self.novelty_streak)),
            "legacyRawAttentionOverlay": 0,
            "routerMembraneUnits": int(
                torch.count_nonzero(
                    self.router.population.membrane.detach()
                ).item()
            ),
            "routerSpikeUnits": int(
                torch.count_nonzero(
                    self.router.population.spike_count.detach()
                ).item()
            ),
            "routerPreTraceUnits": int(
                torch.count_nonzero(
                    self.router.synapses.pre_trace.detach()
                ).item()
            ),
            "routerPostTraceUnits": int(
                torch.count_nonzero(
                    self.router.synapses.post_trace.detach()
                ).item()
            ),
        }

        substrate_clear = self.memory.clear_attention_activity(prior_epoch + 1)
        lifecycle_clear = self.memory_lifecycle.clear_attention()
        cleared["activatedNeurons"] = int(
            substrate_clear["activatedNeurons"]
        )
        cleared["recalledAssemblies"] = int(
            substrate_clear["recalledAssemblies"]
        )
        cleared["substrateEligibilityTraces"] = int(
            substrate_clear["eligibilityTraces"]
        )
        cleared["legacyRawAttentionOverlay"] = int(
            substrate_clear["legacyRawActive"]
        )
        cleared["lifecycleScratch"] = int(lifecycle_clear["afterimageItems"])
        cleared["activeFocus"] = int(lifecycle_clear["activeFocus"])
        self.recent_token_context = []
        self.current_context = {
            "tokenCount": 0,
            "tokenHash": "",
            "recentTokenCount": 0,
            "recentTokenHash": self._token_sequence_hash([]),
            "sensorySlots": 0,
            "updatedAt": _iso_now(),
            "freshAttentionEpoch": prior_epoch + 1,
        }
        self.working_memory = []
        self.workspace_items = []
        self.liquid_state.zero_()
        self.router.reset_activity()
        self.novelty_streak = 0
        for module in self._trainable_modules():
            for parameter in module.parameters():
                parameter.grad = None
        reset_clock = time.time()
        self.last_activity_decay = reset_clock
        self.last_idle_cycle_at = reset_clock
        self.last_idle_visible_action_at = reset_clock
        boundary = {
            "format": FRESH_ATTENTION_FORMAT,
            "formatVersion": FRESH_ATTENTION_VERSION,
            "operationId": operation_id,
            "epoch": prior_epoch + 1,
            "createdAt": _iso_now(),
            "messagesPreserved": messages_preserved,
            "tracesPreserved": traces_preserved,
            "synapsesPreserved": synapse_count,
            "replayEntries": replay_entries,
            "parameterChecksum": parameter_checksum,
            "fastSynapseChecksum": fast_synapse_checksum,
            "substrateContentSha256": substrate_content_sha256,
            "cleared": cleared,
        }
        self.fresh_attention_boundary = boundary
        self._fresh_attention_paged_clear_pending = True
        if (
            self.parameter_checksum() != parameter_checksum
            or self._fast_synapse_checksum() != fast_synapse_checksum
            or str(
                (self.memory.persistence_manifest or {}).get(
                    "contentSha256", ""
                )
            )
            != substrate_content_sha256
            or len(self.memory.synapses) != synapse_count
            or len(self.replay) != replay_entries
            or self.conversation.counts() != conversation_counts
        ):
            raise RuntimeError("fresh attention changed durable neural state")

        try:
            self.save(reuse_substrate_generation=True)
        except Exception:
            persisted = read_json(self.engine_path / "brain.json")
            committed_boundary = self._validated_fresh_attention_boundary(
                persisted.get("fresh_attention_boundary")
            )
            if (
                committed_boundary is None
                or committed_boundary.get("operationId") != operation_id
            ):
                raise

        if str(
            (self.memory.persistence_manifest or {}).get("contentSha256", "")
        ) != substrate_content_sha256:
            raise RuntimeError(
                "fresh attention rewrote the durable substrate generation"
            )

        paged_cleanup_pending = False
        try:
            self.paged_working_memory.clear()
            self._fresh_attention_paged_clear_pending = False
        except (OSError, sqlite3.Error):
            # brain.json already names the empty scratch checkpoint. Reload
            # recovery will delete these now-unreachable temporary rows.
            paged_cleanup_pending = True
        result = self._fresh_attention_result(
            boundary,
            idempotent=False,
            paged_cleanup_pending=paged_cleanup_pending,
        )
        self.events.append(
            "fresh-attention",
            {
                key: value
                for key, value in result.items()
                if key not in {"brainId"}
            },
        )
        return result

    def workspace_snapshot(self) -> Dict[str, Any]:
        items = [
            {
                "id": str(item.get("id", "")) or None,
                "kind": str(item.get("source", "experience")),
                "salience": self._inspection_number(item.get("salience"), 0.5),
                "rehearsals": max(0, int(item.get("rehearsals", 0))),
                "enteredAt": str(item.get("enteredAt", "")) or None,
                "lastActiveAt": str(item.get("lastActiveAt", "")) or None,
            }
            for item in self.workspace_items
        ]
        plain_memory = self.memory_lifecycle.snapshot(self.memory)
        plain_memory["recentWords"] = {
            "capacity": self.config.max_seq_len,
            "count": len(self.recent_token_context),
            "evictions": self.counters["context_token_evictions"],
            "temporary": True,
        }
        paged_count = (
            0
            if self._fresh_attention_paged_clear_pending
            else self.paged_working_memory.count()
        )
        plain_memory["workingThoughts"] = {
            "capacity": self.config.working_memory_slots,
            "count": len(self.working_memory) + paged_count,
            "resident": len(self.working_memory),
            "paged": paged_count,
            "items": items,
            "evictions": self.counters["workspace_evictions"],
            "rehearsals": self.counters["workspace_rehearsals"],
        }
        return {
            "brainId": self.brain_id,
            "queriedAt": _iso_now(),
            "contextWindow": {
                "capacityTokens": self.config.max_seq_len,
                "generationBudgetTokens": (
                    self.config.generation_token_budget()
                ),
                "capacityPolicy": "hardware-derived-resource-guarded",
                "expandable": True,
                "extended": self.config.extended_working_memory,
                "tokenCount": max(
                    0, int(self.current_context.get("tokenCount", 0))
                ),
                "tokenHash": str(self.current_context.get("tokenHash", "")),
                "recentTokenCount": len(self.recent_token_context),
                "recentTokenHash": self._token_sequence_hash(
                    self.recent_token_context
                ),
                "evictions": self.counters["context_token_evictions"],
                "sensorySlots": max(
                    0, int(self.current_context.get("sensorySlots", 0))
                ),
                "updatedAt": str(
                    self.current_context.get("updatedAt", self.updated_at)
                ),
            },
            "latentWorkspace": {
                "capacity": self.config.working_memory_slots,
                "occupancy": len(self.working_memory) + paged_count,
                "resident": len(self.working_memory),
                "paged": paged_count,
                "items": items,
                "evictions": self.counters["workspace_evictions"],
                "rehearsals": self.counters["workspace_rehearsals"],
            },
            "liquidState": {
                "dimensions": int(self.liquid_state.numel()),
                "mean": float(self.liquid_state.detach().float().mean().item()),
                "norm": float(self.liquid_state.detach().float().norm().item()),
            },
            # Plain-language stable surface. ``latentWorkspace`` above remains
            # a compatibility alias for early v1 desktop builds; new UI uses
            # these four human-readable memory views.
            "memory": plain_memory,
            "hiddenBehavioralPrompt": False,
            "rawLongTermTextInjected": False,
            "freshAttentionBoundary": (
                None
                if self.fresh_attention_boundary is None
                else dict(self.fresh_attention_boundary)
            ),
        }

    @staticmethod
    def _inspection_timestamp(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(timestamp):
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))

    @staticmethod
    def _inspection_number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return number if math.isfinite(number) else float(default)

    def _substrate_revision(self) -> str:
        value = "%d:%d:%d:%d:%d:%s" % (
            len(self.memory.neurons),
            len(self.memory.assemblies),
            len(self.memory.synapses),
            int(self.memory.growth_events),
            int(getattr(self.memory, "state_revision", 0)),
            str(getattr(self, "updated_at", "")),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _encode_substrate_cursor(
        offset: int, revision: str, fingerprint: str
    ) -> str:
        payload = {
            "offset": max(0, int(offset)),
            "revision": revision,
            "fingerprint": fingerprint,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_substrate_cursor(
        cursor: str, revision: str, fingerprint: str
    ) -> int:
        if not cursor or len(cursor) > 2048:
            raise ValueError("invalid substrate cursor")
        try:
            padding = "=" * ((4 - len(cursor) % 4) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
            )
        except (ValueError, TypeError, binascii.Error, UnicodeError) as error:
            raise ValueError("invalid substrate cursor") from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("revision") != revision
            or payload.get("fingerprint") != fingerprint
        ):
            raise ValueError(
                "substrate cursor is stale or belongs to another query"
            )
        offset = payload.get("offset")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("invalid substrate cursor offset")
        return offset

    def _inspect_neuron(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        effective_activation = getattr(
            self.memory, "effective_activation", None
        )
        legacy_raw_active = bool(
            getattr(self.memory, "attention_legacy_raw_active", True)
        )
        active_ids = getattr(
            self.memory, "attention_active_neuron_ids", frozenset()
        )
        identifier = str(record.get("id", record.get("neuron_id", "")))
        return {
            "id": identifier,
            "label": str(record.get("label", "")),
            "region": str(record.get("region", "semantic")),
            "activation": (
                float(effective_activation(record))
                if callable(effective_activation)
                else self._inspection_number(record.get("activation"))
            ),
            "importance": self._inspection_number(record.get("importance")),
            "uncertainty": self._inspection_number(
                record.get("uncertainty"), 0.5
            ),
            "exposures": max(0, int(record.get("exposures", 0))),
            "createdAt": self._inspection_timestamp(record.get("created_at")),
            "lastActivatedAt": self._inspection_timestamp(
                record.get("last_activated_at")
                if legacy_raw_active or identifier in active_ids
                else None
            ),
            "aliases": [
                str(value)
                for value in record.get("aliases", [])
                if isinstance(value, str)
            ],
        }

    def _inspect_assembly(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        assembly_id = str(record.get("id", ""))
        legacy_raw_active = bool(
            getattr(self.memory, "attention_legacy_raw_active", True)
        )
        recalled_ids = getattr(
            self.memory, "attention_recalled_assembly_ids", frozenset()
        )
        inspector_node = self.memory.neurons.get(assembly_id, {})
        neuron_ids = [str(value) for value in record.get("neuron_ids", [])]
        child_assembly_ids = [
            str(value) for value in record.get("child_assembly_ids", [])
        ]
        return {
            "id": assembly_id,
            "label": str(
                inspector_node.get(
                    "label",
                    record.get("source_label", record.get("kind", "assembly")),
                )
            ),
            "region": "assembly",
            "neuronIds": neuron_ids,
            "childAssemblyIds": child_assembly_ids,
            "neuronCount": len(neuron_ids),
            "childAssemblyCount": len(child_assembly_ids),
            "relationshipsPaged": False,
            "kind": str(record.get("kind", "knowledge")),
            "source": str(record.get("source", "")),
            "confidence": self._inspection_number(
                record.get("confidence"), 0.5
            ),
            "importance": self._inspection_number(record.get("importance")),
            "rehearsals": max(0, int(record.get("rehearsals", 0))),
            "createdAt": self._inspection_timestamp(record.get("created_at")),
            "lastRecalledAt": self._inspection_timestamp(
                record.get("last_recalled_at")
                if legacy_raw_active or assembly_id in recalled_ids
                else None
            ),
            "sourceLabel": str(record.get("source_label", "")) or None,
            # Inspection reveals whether exact text exists, never the retained
            # passage itself.
            "retainsSourceText": "source_text" in record,
        }

    def _inspect_synapse(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        effective = NeuralSubstrate.exact_effective_weight(
            record.get("effective_weight", 0)
        )
        effective_eligibility = getattr(
            self.memory, "effective_eligibility", None
        )
        return {
            "id": str(record.get("id", "")),
            "sourceId": str(record.get("source_id", "")),
            "targetId": str(record.get("target_id", "")),
            "kind": str(record.get("kind", "associates")),
            "effectiveWeight": effective,
            "eligibility": self._inspection_number(
                (
                    effective_eligibility(record)
                    if callable(effective_eligibility)
                    else record.get("eligibility")
                )
            ),
            "plasticity": self._inspection_number(
                record.get("plasticity"), 1.0
            ),
            "stability": self._inspection_number(record.get("stability")),
            "uses": max(0, int(record.get("uses", 0))),
            "lastUpdatedAt": self._inspection_timestamp(
                record.get("last_updated_at")
            ),
        }

    def _substrate_clusters(
        self, zoom: float, region_filter: str, search: str
    ) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}

        def ensure(
            cluster_id: str,
            label: str,
            kind: str,
            **fields: Any,
        ) -> Dict[str, Any]:
            cluster = clusters.get(cluster_id)
            if cluster is None:
                cluster = {
                    "id": cluster_id,
                    "label": label,
                    "kind": kind,
                    "count": 0,
                    "activeCount": 0,
                    "meanActivation": 0.0,
                    "maxActivation": 0.0,
                    "effectiveWeights": {
                        "negative": 0,
                        "zero": 0,
                        "positive": 0,
                    },
                    "_activationTotal": 0.0,
                    **fields,
                }
                clusters[cluster_id] = cluster
            return cluster

        search_value = search.casefold()
        for raw in self.memory.neurons.values():
            neuron = self._inspect_neuron(raw)
            region = neuron["region"]
            searchable = "%s %s %s" % (
                neuron["id"],
                neuron["label"],
                region,
            )
            if region_filter and region.casefold() != region_filter.casefold():
                continue
            if search_value and search_value not in searchable.casefold():
                continue
            activation = float(neuron["activation"])
            if zoom < 0.35:
                key = "region:%s" % region
                label = region
                kind = "region"
            else:
                band = (
                    "high"
                    if activation >= 0.66
                    else ("medium" if activation >= 0.2 else "quiet")
                )
                key = "region:%s:%s" % (region, band)
                label = "%s · %s" % (region, band)
                kind = "activation-band"
            cluster = ensure(key, label, kind, region=region)
            cluster["count"] += 1
            cluster["activeCount"] += int(activation >= 0.1)
            cluster["_activationTotal"] += activation
            cluster["maxActivation"] = max(
                float(cluster["maxActivation"]), activation
            )

        region_by_id = {
            str(record.get("id", "")): str(record.get("region", "semantic"))
            for record in self.memory.neurons.values()
        }
        for raw in self.memory.synapses.values():
            synapse = self._inspect_synapse(raw)
            source_region = region_by_id.get(synapse["sourceId"], "unknown")
            target_region = region_by_id.get(synapse["targetId"], "unknown")
            if region_filter and region_filter.casefold() not in {
                source_region.casefold(),
                target_region.casefold(),
            }:
                continue
            searchable = "%s %s %s %s %s" % (
                synapse["id"],
                synapse["kind"],
                source_region,
                target_region,
                synapse["effectiveWeight"],
            )
            if search_value and search_value not in searchable.casefold():
                continue
            key = "pathway:%s>%s" % (source_region, target_region)
            cluster = ensure(
                key,
                "%s → %s" % (source_region, target_region),
                "pathway",
                sourceRegion=source_region,
                targetRegion=target_region,
            )
            cluster["count"] += 1
            weight_key = (
                "negative"
                if synapse["effectiveWeight"] < 0
                else ("positive" if synapse["effectiveWeight"] > 0 else "zero")
            )
            cluster["effectiveWeights"][weight_key] += 1

        result: List[Dict[str, Any]] = []
        for cluster in clusters.values():
            count = int(cluster["count"])
            activation_total = float(cluster.pop("_activationTotal"))
            if cluster["kind"] != "pathway" and count:
                cluster["meanActivation"] = activation_total / float(count)
            result.append(cluster)
        return sorted(
            result,
            key=lambda item: (
                str(item["kind"]),
                -int(item["count"]),
                str(item["id"]),
            ),
        )

    @staticmethod
    def _substrate_page(
        records: Iterable[Mapping[str, Any]],
        *,
        inspect: Callable[[Mapping[str, Any]], Dict[str, Any]],
        matches: Callable[[Dict[str, Any]], bool],
        offset: int,
        requested_page_size: int,
    ) -> Tuple[List[Dict[str, Any]], int, int, bool]:
        """Select one contiguous page without imposing a cardinality cap.

        ``requested_page_size`` is a caller preference, not a product limit.
        The response stops only when that request is satisfied or the bounded
        JSON-RPC byte envelope is full.  Scanning continues to calculate the
        exact match count, while the next cursor resumes at the first record
        that was not returned.
        """

        page: List[Dict[str, Any]] = []
        matched = 0
        page_bytes = 2
        transport_limited = False
        page_closed = False
        for raw_record in records:
            record = inspect(raw_record)
            if not matches(record):
                continue
            record_index = matched
            matched += 1
            if record_index < offset or page_closed:
                continue
            if len(page) >= requested_page_size:
                page_closed = True
                continue
            encoded_bytes = len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            if (
                encoded_bytes > SUBSTRATE_INSPECTION_TRANSPORT_BYTES
                and "neuronIds" in record
                and "childAssemblyIds" in record
            ):
                # Assembly membership is also represented by authoritative
                # contains/participates/composes synapses.  If one unusually
                # large assembly cannot fit in one protocol line, return its
                # metadata and exact counts while routing every relationship
                # through the ordinary `connectedTo` synapse cursor instead of
                # imposing a hidden member-count cutoff.
                record = {
                    **record,
                    "neuronIds": [],
                    "childAssemblyIds": [],
                    "relationshipsPaged": True,
                }
                encoded_bytes = len(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
            separator_bytes = 1 if page else 0
            if (
                page
                and page_bytes + separator_bytes + encoded_bytes
                > SUBSTRATE_INSPECTION_TRANSPORT_BYTES
            ):
                transport_limited = True
                page_closed = True
                continue
            if not page and encoded_bytes > SUBSTRATE_INSPECTION_TRANSPORT_BYTES:
                raise ValueError(
                    "one substrate inspection record exceeds the JSON-RPC "
                    "transport envelope; narrow the relationship query"
                )
            page.append(record)
            page_bytes += separator_bytes + encoded_bytes
        return page, matched, page_bytes, transport_limited

    def query_substrate(
        self, query: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a read-only, cursor-paged multiresolution substrate view.

        A byte-based transport envelope protects the JSON-RPC channel without
        capping a page at an arbitrary record count. ``nextCursor`` makes the
        total addressable result unbounded. Cursors are invalidated by every
        live substrate mutation, including activation and STDP changes that do
        not alter structural counts, so a traversal never mixes revisions.
        """

        raw = dict(query or {})
        entity = str(raw.get("entity", "overview"))
        if entity not in {"overview", "neurons", "assemblies", "synapses"}:
            raise ValueError("invalid substrate entity")
        zoom = max(0.0, min(self._inspection_number(raw.get("zoom"), 0.0), 1.0))
        page_size_raw = raw.get("pageSize", 256)
        if isinstance(page_size_raw, bool) or not isinstance(page_size_raw, int):
            raise ValueError("invalid substrate page size")
        page_size = page_size_raw
        if page_size < 1:
            raise ValueError("substrate page size must be positive")
        region = str(raw.get("region", "")).strip()
        search = str(raw.get("search", "")).strip()
        connected_to = str(raw.get("connectedTo", "")).strip()
        if len(region) > 128 or len(search) > 512 or len(connected_to) > 256:
            raise ValueError("substrate filter is too long")
        if connected_to and entity != "synapses":
            raise ValueError("connectedTo is valid only for synapse queries")
        revision = self._substrate_revision()
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "entity": entity,
                    "zoom": round(zoom, 6),
                    "region": region.casefold(),
                    "search": search.casefold(),
                    "connectedTo": connected_to,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        cursor = raw.get("cursor")
        offset = (
            self._decode_substrate_cursor(
                str(cursor), revision, fingerprint
            )
            if cursor
            else 0
        )
        totals = {
            "neurons": len(self.memory.neurons),
            "assemblies": len(self.memory.assemblies),
            "synapses": len(self.memory.synapses),
        }
        response: Dict[str, Any] = {
            "brainId": self.brain_id,
            "queriedAt": _iso_now(),
            "revision": revision,
            "entity": entity,
            "zoom": zoom,
            "totals": totals,
            "matched": 0,
            "offset": offset,
            "returned": 0,
            "pageBytes": 0,
            "transportLimited": False,
            "hasMore": False,
            "clusters": [],
            "neurons": [],
            "assemblies": [],
            "synapses": [],
        }

        # Overview and zoomed-out views intentionally aggregate the entire
        # substrate. Zoom in to receive individual cursor-paged records.
        if entity == "overview" or zoom < 0.66:
            clusters = self._substrate_clusters(zoom, region, search)
            page, matched, page_bytes, transport_limited = self._substrate_page(
                clusters,
                inspect=lambda record: dict(record),
                matches=lambda _record: True,
                offset=offset,
                requested_page_size=page_size,
            )
            end = offset + len(page)
            response["clusters"] = page
            response["matched"] = matched
            response["returned"] = len(page)
            response["pageBytes"] = page_bytes
            response["transportLimited"] = transport_limited
            response["hasMore"] = end < matched
            if response["hasMore"]:
                response["nextCursor"] = self._encode_substrate_cursor(
                    end, revision, fingerprint
                )
            return response

        search_value = search.casefold()
        if entity == "neurons":
            raw_records = (
                self.memory.neurons[record_id]
                for record_id in sorted(self.memory.neurons)
            )
            inspect = self._inspect_neuron

            def matches(record: Dict[str, Any]) -> bool:
                return (
                    (
                        not region
                        or record["region"].casefold() == region.casefold()
                    )
                    and (
                        not search_value
                        or search_value
                        in (
                            "%s %s %s"
                            % (record["id"], record["label"], record["region"])
                        ).casefold()
                    )
                )

        elif entity == "assemblies":
            raw_records = iter(
                sorted(
                    self.memory.assemblies,
                    key=lambda record: str(record.get("id", "")),
                )
            )
            inspect = self._inspect_assembly

            def matches(record: Dict[str, Any]) -> bool:
                return (
                    (not region or region.casefold() == "assembly")
                    and (
                        not search_value
                        or search_value
                        in (
                            "%s %s %s %s"
                            % (
                                record["id"],
                                record["label"],
                                record["kind"],
                                record["sourceLabel"] or "",
                            )
                        ).casefold()
                    )
                )

        else:
            region_by_id = {
                str(record.get("id", "")): str(
                    record.get("region", "semantic")
                )
                for record in self.memory.neurons.values()
            }
            raw_records = (
                self.memory.synapses[record_id]
                for record_id in sorted(self.memory.synapses)
            )
            inspect = self._inspect_synapse

            def matches(record: Dict[str, Any]) -> bool:
                return (
                    (
                        not region
                        or region.casefold()
                        in {
                            region_by_id.get(
                                record["sourceId"], "unknown"
                            ).casefold(),
                            region_by_id.get(
                                record["targetId"], "unknown"
                            ).casefold(),
                        }
                    )
                    and (
                        not connected_to
                        or connected_to
                        in {record["sourceId"], record["targetId"]}
                    )
                    and (
                        not search_value
                        or search_value
                        in (
                            "%s %s %s %s"
                            % (
                                record["id"],
                                record["sourceId"],
                                record["targetId"],
                                record["kind"],
                            )
                        ).casefold()
                    )
                )

        page, matched, page_bytes, transport_limited = self._substrate_page(
            raw_records,
            inspect=inspect,
            matches=matches,
            offset=offset,
            requested_page_size=page_size,
        )
        end = offset + len(page)
        response[entity] = page
        response["matched"] = matched
        response["returned"] = len(page)
        response["pageBytes"] = page_bytes
        response["transportLimited"] = transport_limited
        response["hasMore"] = end < matched
        if response["hasMore"]:
            response["nextCursor"] = self._encode_substrate_cursor(
                end, revision, fingerprint
            )
        return response

    def _optimize_experience(
        self,
        text: str,
        vsa_vector: torch.Tensor,
        steps: int,
        learning_rate: Optional[float] = None,
        commit_stability: bool = True,
    ) -> Dict[str, float]:
        if learning_rate is not None:
            optimizer = self._new_optimizer(learning_rate)
        else:
            self._ensure_optimizer_resident()
            optimizer = self._optimizer
        learning_parameters = self._streaming_experience_parameters()
        losses: List[float] = []
        language_losses: List[float] = []
        idea_losses: List[float] = []
        workspace_losses: List[float] = []
        stability_losses: List[float] = []
        self.decoder.train()
        self.memory_bridge.train()
        self.idea_adapter.train()
        self.liquid.train()
        for _ in range(max(1, int(steps))):
            for ids in self.tokenizer.window_tensors(
                text,
                self.device,
                max_length=min(
                    self.config.max_seq_len,
                    self._runtime_training_max_seq_len,
                ),
                add_bos=True,
                add_eos=True,
            ):
                if ids.shape[1] < 2:
                    continue
                if optimizer is self._optimizer:
                    self._ensure_optimizer_resident()
                optimizer.zero_grad(set_to_none=True)
                idea = self._idea_model_vector(vsa_vector)
                noise = torch.randn_like(idea) * 0.06
                reconstructed = self.idea_adapter(idea + noise)
                idea_loss = F.mse_loss(reconstructed, idea.detach())
                temporal, _ = self.liquid(
                    idea, state=self.liquid_state.detach(), elapsed=1.0
                )
                temporal_loss = F.mse_loss(temporal, idea.detach())
                whole = self.decoder.encode_whole(ids)
                workspace_loss = F.mse_loss(
                    F.normalize(whole, dim=-1),
                    F.normalize(idea.detach(), dim=-1),
                )
                language = self.decoder(
                    ids, memory_bias=reconstructed, labels=ids
                )["loss"]
                stability_loss = self._stability_penalty(
                    learning_parameters
                )
                loss = (
                    language
                    + 0.2 * idea_loss
                    + 0.05 * temporal_loss
                    + 0.1 * workspace_loss
                    + stability_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                self._accumulate_slow_importance(learning_parameters)
                parameters = [
                    parameter for parameter in learning_parameters
                    if parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.grad_clip
                )
                optimizer.step()
                if optimizer is self._optimizer:
                    self._maintain_neural_state_resources()
                losses.append(float(loss.detach().item()))
                language_losses.append(float(language.detach().item()))
                idea_losses.append(float(idea_loss.detach().item()))
                workspace_losses.append(float(workspace_loss.detach().item()))
                stability_losses.append(float(stability_loss.detach().item()))
                self.counters["training_steps"] += 1
        if not losses:
            return {
                "loss": 0.0,
                "language_loss": 0.0,
                "idea_loss": 0.0,
                "workspace_loss": 0.0,
                "stability_loss": 0.0,
            }
        if commit_stability:
            self._commit_slow_anchors(
                rate=0.08, parameters=learning_parameters
            )
        return {
            "loss": sum(losses) / len(losses),
            "language_loss": sum(language_losses) / len(language_losses),
            "idea_loss": sum(idea_losses) / len(idea_losses),
            "workspace_loss": sum(workspace_losses) / len(workspace_losses),
            "stability_loss": sum(stability_losses) / len(stability_losses),
        }

    def _local_exact_dialogue_windows(
        self,
        human: str,
        brain: str,
        *,
        training_sequence_tokens: int,
    ) -> Iterator[Tuple[List[int], int, int]]:
        """Yield bounded local-decoder windows with exhaustive byte targets.

        Later windows include the preceding response byte as masked causal
        context. Only the final window targets EOS, so concatenated supervised
        labels are exactly every UTF-8 response byte followed by one EOS.
        """

        context_limit = max(
            8,
            min(
                int(self.config.max_seq_len),
                int(training_sequence_tokens),
            ),
        )
        human_bytes = human.encode("utf-8")
        response_bytes = brain.encode("utf-8")
        if not response_bytes:
            raise ValueError("exact local dialogue response must be non-empty")
        # BOS, HUMAN, BRAIN, one causal continuation byte, and final EOS are
        # reserved before sharing the remaining live context between the
        # human suffix and each exact target window.
        shared_payload = max(2, context_limit - 5)
        response_budget = max(1, shared_payload // 2)
        human_budget = max(0, shared_payload - response_budget)
        if len(human_bytes) < human_budget:
            human_budget = len(human_bytes)
            response_budget = max(1, shared_payload - human_budget)
        human_context = human_bytes[-human_budget:] if human_budget else b""
        previous_response_id: Optional[int] = None
        for offset in range(0, len(response_bytes), response_budget):
            target_bytes = response_bytes[offset : offset + response_budget]
            final_window = offset + len(target_bytes) >= len(response_bytes)
            prefix_ids = (
                [previous_response_id]
                if previous_response_id is not None
                else []
            )
            target_ids = [
                int(value) + self.tokenizer.byte_offset
                for value in target_bytes
            ]
            ids_list = [
                self.tokenizer.bos_id,
                self.tokenizer.human_id,
                *(
                    int(value) + self.tokenizer.byte_offset
                    for value in human_context
                ),
                self.tokenizer.brain_id,
                *prefix_ids,
                *target_ids,
                *([self.tokenizer.eos_id] if final_window else []),
            ]
            if len(ids_list) > context_limit:
                raise RuntimeError(
                    "exact local dialogue window exceeds the training context"
                )
            target_start = (
                3 + len(human_context) + len(prefix_ids)
            )
            target_count = len(target_ids) + (1 if final_window else 0)
            yield ids_list, target_start, target_count
            previous_response_id = target_ids[-1]

    def _optimize_exact_dialogue_pair(
        self,
        human: str,
        brain: str,
        vsa_vector: torch.Tensor,
        steps: int,
        commit_stability: bool,
        training_sequence_tokens: int,
    ) -> Dict[str, Any]:
        """Apply one token-weighted local step across all exact windows."""

        target_tokens = len(brain.encode("utf-8")) + 1
        losses: List[float] = []
        language_losses: List[float] = []
        target_windows = 0
        trained_parameters: Dict[int, nn.Parameter] = {}
        try:
            for _ in range(max(1, int(steps))):
                self._ensure_optimizer_resident()
                self._optimizer.zero_grad(set_to_none=True)
                weighted_language_sum = 0.0
                visited_targets = 0
                step_windows = 0
                for ids_list, target_start, target_count in (
                    self._local_exact_dialogue_windows(
                        human,
                        brain,
                        training_sequence_tokens=training_sequence_tokens,
                    )
                ):
                    ids = torch.tensor(
                        [ids_list], dtype=torch.long, device=self.device
                    )
                    labels = ids.clone()
                    labels[:, :target_start] = self.tokenizer.pad_id
                    idea = self._idea_model_vector(vsa_vector)
                    adapted = self.idea_adapter(idea)
                    prediction_loss = self.decoder(
                        ids, memory_bias=adapted, labels=labels
                    )["loss"]
                    if not bool(torch.isfinite(prediction_loss)):
                        raise RuntimeError(
                            "non-finite exact dialogue-pair loss"
                        )
                    objective = prediction_loss * (
                        float(target_count) / float(target_tokens)
                    )
                    objective.backward()
                    weighted_language_sum += (
                        float(prediction_loss.detach().item())
                        * target_count
                    )
                    visited_targets += target_count
                    step_windows += 1
                if visited_targets != target_tokens or step_windows < 1:
                    raise RuntimeError(
                        "exact local dialogue target coverage is invalid"
                    )
                connected_parameters = tuple(
                    parameter
                    for group in self._optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                )
                if not connected_parameters:
                    raise RuntimeError(
                        "exact local dialogue produced no connected gradients"
                    )
                stability_loss = self._stability_penalty(
                    connected_parameters
                )
                if not bool(torch.isfinite(stability_loss)):
                    raise RuntimeError("non-finite dialogue stability loss")
                if stability_loss.requires_grad:
                    stability_loss.backward()
                self._accumulate_slow_importance(connected_parameters)
                torch.nn.utils.clip_grad_norm_(
                    connected_parameters, self.config.grad_clip
                )
                self._optimizer.step()
                self._maintain_neural_state_resources()
                self.counters["training_steps"] += 1
                language_value = weighted_language_sum / float(target_tokens)
                language_losses.append(language_value)
                losses.append(
                    language_value + float(stability_loss.detach().item())
                )
                target_windows = step_windows
                trained_parameters.update(
                    {
                        id(parameter): parameter
                        for parameter in connected_parameters
                    }
                )
        except Exception:
            self._optimizer.zero_grad(set_to_none=True)
            raise
        if commit_stability:
            self._commit_slow_anchors(
                rate=0.08,
                parameters=trained_parameters.values(),
            )
        parameter_names = {
            id(parameter): name
            for name, parameter in self._named_slow_parameters().items()
        }
        return {
            "loss": sum(losses) / len(losses),
            "language_loss": sum(language_losses) / len(language_losses),
            "target_tokens": target_tokens,
            "target_windows": target_windows,
            "optimizer_steps": len(losses),
            "target_window_policy": LOCAL_TYPED_TARGET_WINDOW_POLICY,
            "logical_token_weighted_objective": True,
            "_stability_parameter_names": tuple(
                parameter_names[parameter_id]
                for parameter_id in trained_parameters
                if parameter_id in parameter_names
            ),
        }

    def _optimize_dialogue_pair(
        self,
        human: str,
        brain: str,
        vsa_vector: torch.Tensor,
        steps: int = 1,
        commit_stability: bool = True,
        exact_response_windows: bool = False,
        exact_training_sequence_tokens: Optional[int] = None,
        exact_target_window_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        if exact_response_windows:
            if exact_target_window_policy != LOCAL_TYPED_TARGET_WINDOW_POLICY:
                raise ValueError("local typed target window policy is invalid")
            if exact_training_sequence_tokens is None:
                raise ValueError(
                    "local typed target window context is not schedule-frozen"
                )
            return self._optimize_exact_dialogue_pair(
                human,
                brain,
                vsa_vector,
                steps=max(1, int(steps)),
                commit_stability=commit_stability,
                training_sequence_tokens=int(
                    exact_training_sequence_tokens
                ),
            )
        ids_list = self.tokenizer.dialogue(human, brain, complete=True)
        training_context_limit = min(
            self.config.max_seq_len,
            self._runtime_training_max_seq_len,
        )
        if len(ids_list) > training_context_limit:
            # Preserve the role boundary and response when a long human turn is
            # clipped to the physical working-token budget.
            response_ids = [
                value + self.tokenizer.byte_offset
                for value in brain.encode("utf-8")
            ]
            response_ids = response_ids[-max(1, training_context_limit // 2) :]
            human_budget = max(
                1, training_context_limit - len(response_ids) - 4
            )
            human_ids = [
                value + self.tokenizer.byte_offset
                for value in human.encode("utf-8")
            ][-human_budget:]
            ids_list = [
                self.tokenizer.bos_id,
                self.tokenizer.human_id,
                *human_ids,
                self.tokenizer.brain_id,
                *response_ids,
                self.tokenizer.eos_id,
            ]
        ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        labels = ids.clone()
        boundary = ids_list.index(self.tokenizer.brain_id)
        labels[:, : boundary + 1] = self.tokenizer.pad_id
        learning_parameters = self._streaming_experience_parameters()
        losses: List[float] = []
        for _ in range(max(1, int(steps))):
            self._ensure_optimizer_resident()
            self._optimizer.zero_grad(set_to_none=True)
            idea = self._idea_model_vector(vsa_vector)
            adapted = self.idea_adapter(idea)
            prediction_loss = self.decoder(
                ids, memory_bias=adapted, labels=labels
            )["loss"]
            stability_loss = self._stability_penalty(
                learning_parameters
            )
            loss = prediction_loss + stability_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite dialogue-pair loss")
            loss.backward()
            self._accumulate_slow_importance(learning_parameters)
            parameters = [
                parameter for parameter in learning_parameters
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
            self._optimizer.step()
            self._maintain_neural_state_resources()
            self.counters["training_steps"] += 1
            losses.append(float(loss.detach().item()))
        if commit_stability:
            self._commit_slow_anchors(
                rate=0.08, parameters=learning_parameters
            )
        return {"loss": sum(losses) / len(losses)}



    def _apply_supervised_dialogue(
        self,
        human: str,
        response: str,
        *,
        steps: int = 1,
        train_local: bool = True,
        local_exact_response_windows: bool = False,
        local_exact_training_sequence_tokens: Optional[int] = None,
        local_exact_target_window_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply one typed target as an atomic slow-learning transaction."""

        clean_human = human.replace("\x00", "").strip()
        clean_response = response.replace("\x00", "").strip()
        if not clean_human or not clean_response:
            raise ValueError("dialogue training requires non-empty human and brain text")
        if not train_local:
            return {
                "humanSha256": hashlib.sha256(
                    clean_human.encode("utf-8")
                ).hexdigest(),
                "responseSha256": hashlib.sha256(
                    clean_response.encode("utf-8")
                ).hexdigest(),
                "local": {
                    "deferredToStreamingBatch": True,
                    "loss": 0.0,
                },
            }
        snapshot = self._snapshot_slow_transaction_state()
        try:
            cue = self.memory.vector_for_text(clean_human)
            local = (
                self._optimize_dialogue_pair(
                    clean_human,
                    clean_response,
                    cue,
                    steps=max(1, int(steps)),
                    commit_stability=False,
                    exact_response_windows=local_exact_response_windows,
                    exact_training_sequence_tokens=(
                        local_exact_training_sequence_tokens
                    ),
                    exact_target_window_policy=(
                        local_exact_target_window_policy
                    ),
                )
                if train_local
                else {"deferredToStreamingBatch": True, "loss": 0.0}
            )
            local_stability_names = tuple(
                str(value)
                for value in local.pop(
                    "_stability_parameter_names", ()
                )
            )
            if local_exact_response_windows:
                named_slow_parameters = self._named_slow_parameters()
                stability_parameters = [
                    named_slow_parameters[name]
                    for name in local_stability_names
                    if name in named_slow_parameters
                ]
                self._commit_slow_anchors(
                    rate=0.08,
                    parameters=stability_parameters,
                )
            else:
                self._commit_slow_anchors(rate=0.08)
            return {
                "humanSha256": hashlib.sha256(
                    clean_human.encode("utf-8")
                ).hexdigest(),
                "responseSha256": hashlib.sha256(
                    clean_response.encode("utf-8")
                ).hexdigest(),
                "local": local,
            }
        except Exception:
            self._restore_slow_transaction_state(snapshot)
            raise







    @staticmethod
    def _typed_dialogue_pairs(provenance: Mapping[str, Any]) -> List[Tuple[str, str]]:
        if provenance.get("format") != "typed-dialogue":
            return []
        raw_pairs = provenance.get("dialoguePairs", [])
        if not isinstance(raw_pairs, list):
            return []
        pairs: List[Tuple[str, str]] = []
        for value in raw_pairs:
            if not isinstance(value, Mapping):
                continue
            human = str(value.get("human", "")).replace("\x00", "").strip()
            response = str(value.get("brain", "")).replace("\x00", "").strip()
            if human and response:
                pairs.append((human, response))
        return pairs

    def _release_training_allocator_cache(self) -> None:
        """Release only disposable allocator caches after a failed allocation."""

        if self.device_backend == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # MPS cache clearing is intentionally forbidden here: PyTorch may
        # release MPSGraph objects still referenced by an in-flight command
        # queue. The caller reports a resource pause instead.

    def _allocator_resource_pause(
        self,
        error: BaseException,
        *,
        stage: str,
        physical_batch: Optional[int] = None,
        sequence_tokens: Optional[int] = None,
    ) -> NeuralStateResourcePause:
        """Describe a recoverable allocator pause without committing mutations."""

        physical = max(
            1,
            int(
                physical_batch
                if physical_batch is not None
                else self._runtime_train_batch_size
            ),
        )
        sequence = max(
            8,
            int(
                sequence_tokens
                if sequence_tokens is not None
                else self._runtime_training_max_seq_len
            ),
        )
        recommended_batch = max(1, physical // 2)
        recommended_sequence = (
            sequence if physical > 1 else max(8, sequence // 2)
        )
        status = self.resource_policy.status()
        status.update(
            {
                "mode": "allocator-oom-checkpoint-recovery",
                "paused": True,
                "recoverable": True,
                "allocatorOutOfMemory": True,
                "allocatorBackend": self.device_backend,
                "failureStage": str(stage),
                "rollbackRequired": True,
                "resumeFromLastCheckpoint": True,
                "sourceRecordsSkipped": False,
                "scratchWriteAttempted": False,
                "recoveryPlan": {
                    "physicalBatchSize": recommended_batch,
                    "sequenceTokens": recommended_sequence,
                    "streamEveryRecord": True,
                    "retryUncommittedSuffix": True,
                },
                "oomCount": int(self._allocator_oom_count),
                "errorType": error.__class__.__name__,
            }
        )
        message = (
            "training paused after the %s allocator ran out of memory; "
            "the uncommitted batch must be rolled back and the exact dataset "
            "suffix can resume with a smaller RAM microbatch"
            % self.device_backend
        )
        self.resource_pause = {
            "reason": message,
            "readings": status,
            "at": _iso_now(),
        }
        return NeuralStateResourcePause(message, status)

    def apply_allocator_oom_downgrade(
        self, status: Mapping[str, Any]
    ) -> Dict[str, int]:
        """Apply a non-neural, process-local batch downgrade after rollback."""

        raw_plan = status.get("recoveryPlan", {})
        plan = raw_plan if isinstance(raw_plan, Mapping) else {}
        physical = max(1, int(plan.get("physicalBatchSize", 1)))
        sequence = max(8, int(plan.get("sequenceTokens", 8)))
        self._runtime_train_batch_size = min(
            self._runtime_train_batch_size, physical
        )
        self._runtime_training_max_seq_len = min(
            self._runtime_training_max_seq_len, sequence
        )
        self._allocator_oom_count = max(
            self._allocator_oom_count, int(status.get("oomCount", 0))
        )
        self.resource_pause = {
            "reason": (
                "allocator recovery is ready to resume from the last atomic "
                "dataset checkpoint"
            ),
            "readings": dict(status),
            "at": _iso_now(),
        }
        return {
            "physicalBatchSize": self._runtime_train_batch_size,
            "sequenceTokens": self._runtime_training_max_seq_len,
        }

    def _clear_allocator_recovery_pause(self) -> None:
        """Clear only a completed allocator pause, preserving real pressure."""

        pause = self.resource_pause
        if not isinstance(pause, Mapping):
            return
        readings = pause.get("readings")
        if isinstance(readings, Mapping) and readings.get("mode") == (
            "allocator-oom-checkpoint-recovery"
        ):
            self.resource_pause = None

    def _streaming_experience_window_batches(
        self,
        experiences: Sequence[Tuple[str, torch.Tensor]],
        *,
        physical_batch: int,
        sequence_tokens: int,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Yield bounded token-window batches while visiting every source byte."""

        pending_ids: List[torch.Tensor] = []
        pending_vectors: List[torch.Tensor] = []
        physical_batch = max(1, int(physical_batch))
        sequence_tokens = max(8, int(sequence_tokens))
        for text, vector in experiences:
            for window in self.tokenizer.window_tensors(
                text,
                self.device,
                max_length=sequence_tokens,
                add_bos=True,
                add_eos=True,
            ):
                if window.shape[1] < 2:
                    continue
                pending_ids.append(window[0])
                pending_vectors.append(vector)
                if len(pending_ids) >= physical_batch:
                    yield pending_ids, pending_vectors
                    pending_ids = []
                    pending_vectors = []
        if pending_ids:
            yield pending_ids, pending_vectors

    def _experience_batch_loss(
        self,
        texts: Sequence[str],
        vsa_vectors: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute one padded micro-batch without stepping the optimizer."""

        encoded: List[torch.Tensor] = []
        expanded_vectors: List[torch.Tensor] = []
        for text, vector in zip(texts, vsa_vectors):
            for window in self.tokenizer.window_tensors(
                text,
                self.device,
                max_length=min(
                    self.config.max_seq_len,
                    self._runtime_training_max_seq_len,
                ),
                add_bos=True,
                add_eos=True,
            ):
                if window.shape[1] < 2:
                    continue
                encoded.append(window[0])
                expanded_vectors.append(vector)
        if not encoded:
            raise ValueError("training batch contained no token windows")
        return self._experience_ids_batch_loss(encoded, expanded_vectors)

    def _streaming_experience_parameters(self) -> Tuple[nn.Parameter, ...]:
        """Return only parameters connected to the corpus batch objective.

        ``OmniDecoder.forward`` exposes action logits alongside language
        logits, but a corpus loss does not consume either action head. Adding
        an all-model stability penalty would nevertheless materialize zero
        gradients for those learned heads, causing AdamW to advance their
        state and apply weight decay. Modality generators are likewise
        unrelated to this objective. Keep stability scoped to the
        modules used by ``_experience_ids_batch_loss``. Packed residual gains
        and expert routes learn in backward without optimizer parameters.
        """

        parameters: List[nn.Parameter] = []
        modules: Tuple[nn.Module, ...] = (
            self.decoder.embedding,
            self.decoder.global_workspace,
            self.decoder.workspace_strength,
            self.decoder.memory_projection,
            self.decoder.memory_strength,
            self.decoder.blocks,
            self.decoder.final_norm,
            self.decoder.language_head,
            self.decoder.experts,
            self.decoder.expert_prototypes,
            self.memory_bridge,
            self.idea_adapter,
            self.liquid,
        )
        for module in modules:
            parameters.extend(module.parameters())
        return tuple(parameters)

    def _experience_ids_batch_loss(
        self,
        encoded: Sequence[torch.Tensor],
        vsa_vectors: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute a padded batch from already bounded token windows."""

        if not encoded or len(encoded) != len(vsa_vectors):
            raise ValueError("token windows and idea vectors must align")
        width = max(int(ids.shape[0]) for ids in encoded)
        ids = torch.full(
            (len(encoded), width),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for index, values in enumerate(encoded):
            ids[index, : values.shape[0]] = values
        idea = torch.cat(
            [self._idea_model_vector(vector) for vector in vsa_vectors],
            dim=0,
        )
        reconstructed = self.idea_adapter(
            idea + torch.randn_like(idea) * 0.06
        )
        idea_loss = F.mse_loss(reconstructed, idea.detach())
        liquid_state = self.liquid_state.detach().expand(
            idea.shape[0], -1
        )
        temporal, _ = self.liquid(
            idea, state=liquid_state, elapsed=1.0
        )
        temporal_loss = F.mse_loss(temporal, idea.detach())
        attention_mask = ids.ne(self.tokenizer.pad_id)
        whole = self.decoder.global_workspace.summarize(
            self.decoder.embedding(ids),
            attention_mask=attention_mask,
        )
        workspace_loss = F.mse_loss(
            F.normalize(whole, dim=-1),
            F.normalize(idea.detach(), dim=-1),
        )
        logits = self.decoder(
            ids,
            memory_bias=reconstructed,
            attention_mask=attention_mask,
        )["logits"]
        token_losses = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            ids[:, 1:].contiguous().view(-1),
            ignore_index=self.tokenizer.pad_id,
            reduction="none",
        ).view(ids.shape[0], -1)
        prediction_mask = attention_mask[:, 1:]
        language = (
            (token_losses * prediction_mask).sum(dim=1)
            / prediction_mask.sum(dim=1).clamp_min(1)
        ).mean()
        stability_loss = self._stability_penalty(
            self._streaming_experience_parameters()
        )
        loss_components = (
            ("idea_loss", idea_loss),
            ("temporal_loss", temporal_loss),
            ("workspace_loss", workspace_loss),
            ("language_loss", language),
            ("stability_loss", stability_loss),
        )
        for component_name, component in loss_components:
            if not bool(torch.isfinite(component).all()):
                raise RuntimeError(
                    "non-finite batch training loss component: %s"
                    % component_name
                )
        loss = (
            language
            + 0.2 * idea_loss
            + 0.05 * temporal_loss
            + 0.1 * workspace_loss
            + stability_loss
        )
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError("non-finite batch training loss aggregate")
        return loss, {
            "loss": float(loss.detach().item()),
            "language_loss": float(language.detach().item()),
            "idea_loss": float(idea_loss.detach().item()),
            "workspace_loss": float(workspace_loss.detach().item()),
            "stability_loss": float(stability_loss.detach().item()),
        }

    @staticmethod
    def _round_streaming_ieee_bits(values: torch.Tensor) -> torch.Tensor:
        """Round normal FP32 values to twenty significant bits exactly."""

        magnitude_bits = values.abs().view(torch.int32)
        remainder = magnitude_bits.bitwise_and(0xF)
        truncated = magnitude_bits.bitwise_and(-16)
        retained_lsb = magnitude_bits.bitwise_right_shift(4).bitwise_and(1)
        increment = (remainder > 8) | (
            (remainder == 8) & (retained_lsb == 1)
        )
        rounded_bits = truncated + increment.to(torch.int32) * 16
        canonical_magnitude = rounded_bits.view(torch.float32).clamp_max(
            torch.finfo(torch.float32).max
        )
        return torch.where(values < 0, -canonical_magnitude, canonical_magnitude)

    @staticmethod
    def _round_streaming_ieee_bits_on_device(
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Device hook kept separate so unsupported backends can fall back."""

        return AdaptiveBrain._round_streaming_ieee_bits(values)

    @staticmethod
    def _unsupported_bitwise_canonicalization(error: BaseException) -> bool:
        if isinstance(error, NotImplementedError):
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "not implemented",
                "not supported",
                "could not run 'aten::",
                "could not run aten::",
                "privateuse1",
                "privateuseone",
            )
        )

    @staticmethod
    @torch.no_grad()
    def _canonicalize_streaming_float_tensor(tensor: torch.Tensor) -> float:
        """Remove low FP32 reduction jitter on a relative binary lattice.

        CPU/GPU reduction kernels can differ by a few terminal mantissa bits
        solely because of allocation alignment or execution grouping. That is
        enough to break a cryptographic resume checksum even when the learning
        objective, RNG, and logical schedule are identical. Keeping twenty
        significant binary mantissa bits removes that non-semantic jitter while
        preserving sign and relative precision across tiny and large values.
        Non-finite values are deliberately untouched so this routine can never
        hide an invalid optimizer state.
        """

        if tensor.dtype != torch.float32 or tensor.numel() == 0:
            return 0.0
        # Clear four of FP32's twenty-three stored mantissa bits using exact
        # IEEE-754 bit operations and round-to-nearest-even. This retains
        # twenty significant binary bits without log2/exp2. In particular,
        # torch 2.13 MPS flushes exp2(-126) to zero even though 2**-126 is a
        # valid normal FP32 value; the prior arithmetic lattice consequently
        # turned finite tiny Adam second moments into NaN via inf * 0.
        if not tensor.is_contiguous():
            raise RuntimeError(
                "streaming FP32 canonicalization requires contiguous state"
            )
        flat = tensor.view(-1)
        maximum_delta = 0.0
        cpu_fallback = False
        for offset in range(
            0, flat.numel(), STREAMING_CANONICAL_CHUNK_ELEMENTS
        ):
            chunk = flat[
                offset : offset + STREAMING_CANONICAL_CHUNK_ELEMENTS
            ]
            absolute = chunk.abs()
            normal = torch.isfinite(chunk) & (
                absolute >= torch.finfo(torch.float32).tiny
            )
            if not bool(normal.any()):
                continue
            original = chunk[normal]
            if cpu_fallback:
                canonical = AdaptiveBrain._round_streaming_ieee_bits(
                    original.detach().cpu()
                ).to(device=original.device)
            else:
                try:
                    canonical = (
                        AdaptiveBrain._round_streaming_ieee_bits_on_device(
                            original
                        )
                    )
                except (RuntimeError, NotImplementedError) as error:
                    if not AdaptiveBrain._unsupported_bitwise_canonicalization(
                        error
                    ):
                        raise
                    cpu_fallback = True
                    canonical = AdaptiveBrain._round_streaming_ieee_bits(
                        original.detach().cpu()
                    ).to(device=original.device)
            if not bool(torch.isfinite(canonical).all()):
                raise RuntimeError(
                    "streaming canonicalization produced a non-finite FP32 value"
                )
            maximum_delta = max(
                maximum_delta,
                float(
                    (canonical - original)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                    .item()
                ),
            )
            chunk[normal] = canonical
        return maximum_delta

    @staticmethod
    def _validate_streaming_adam_step(
        value: Any,
        *,
        parameter_name: str,
        stage: str,
    ) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                numeric = float("nan")
            else:
                numeric = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            numeric = float("nan")
        if (
            not math.isfinite(numeric)
            or numeric < 0.0
            or not numeric.is_integer()
        ):
            raise RuntimeError(
                "invalid Adam optimizer step %s: %s.step"
                % (stage, parameter_name)
            )

    def _validate_streaming_optimizer_state(
        self,
        active_parameters: Sequence[
            Tuple[str, nn.Parameter, Mapping[str, Any]]
        ],
        *,
        stage: str,
        require_state: bool,
    ) -> None:
        adam_optimizer = isinstance(
            self._optimizer,
            (torch.optim.Adam, torch.optim.AdamW),
        )
        for parameter_name, parameter, group in active_parameters:
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(
                    "non-finite parameter %s: %s"
                    % (stage, parameter_name)
                )
            if not adam_optimizer:
                continue
            state = self._optimizer.state.get(parameter)
            if not state and not require_state:
                continue
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    "missing Adam optimizer state %s: %s"
                    % (stage, parameter_name)
                )
            required_state = {"step", "exp_avg", "exp_avg_sq"}
            if bool(group.get("amsgrad", False)):
                required_state.add("max_exp_avg_sq")
            missing_state = sorted(required_state.difference(state))
            if missing_state:
                raise RuntimeError(
                    "missing Adam optimizer state %s: %s.%s"
                    % (stage, parameter_name, missing_state[0])
                )
            self._validate_streaming_adam_step(
                state["step"],
                parameter_name=parameter_name,
                stage=stage,
            )
            for state_name, state_value in state.items():
                if state_name == "step":
                    continue
                if isinstance(state_value, torch.Tensor):
                    finite_state = bool(torch.isfinite(state_value).all())
                elif isinstance(state_value, (int, float)):
                    finite_state = math.isfinite(float(state_value))
                else:
                    finite_state = False
                if not finite_state:
                    raise RuntimeError(
                        "non-finite Adam optimizer state %s: %s.%s"
                        % (stage, parameter_name, state_name)
                    )
                if state_name in {"exp_avg_sq", "max_exp_avg_sq"}:
                    if isinstance(state_value, torch.Tensor):
                        nonnegative = bool(state_value.ge(0).all())
                    else:
                        nonnegative = float(state_value) >= 0.0
                    if not nonnegative:
                        raise RuntimeError(
                            "negative Adam optimizer second moment %s: %s.%s"
                            % (stage, parameter_name, state_name)
                        )

    @torch.no_grad()
    def _canonicalize_streaming_learning_state(
        self,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ) -> float:
        """Canonicalize one streaming optimizer commit and its slow state."""

        tensors: List[Tuple[str, torch.Tensor]] = []
        seen: set[int] = set()
        allowed = (
            None if parameters is None else {id(value) for value in parameters}
        )

        def include(label: str, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                if id(value) not in seen:
                    seen.add(id(value))
                    tensors.append((label, value))
                return
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    include("%s.%s" % (label, key), nested)
            elif isinstance(value, (list, tuple)):
                for index, nested in enumerate(value):
                    include("%s[%d]" % (label, index), nested)

        named_parameters = {
            id(parameter): name
            for name, parameter in self._named_slow_parameters().items()
        }
        parameter_ids_by_name = {
            name: id(parameter)
            for name, parameter in self._named_slow_parameters().items()
        }
        for group_index, group in enumerate(self._optimizer.param_groups):
            for parameter_index, parameter in enumerate(
                group.get("params", [])
            ):
                if allowed is not None and id(parameter) not in allowed:
                    continue
                parameter_name = named_parameters.get(
                    id(parameter),
                    "optimizer-group-%d-parameter-%d"
                    % (group_index, parameter_index),
                )
                include("parameter.%s" % parameter_name, parameter)
                state = self._optimizer.state.get(parameter)
                if state is not None:
                    for state_name, state_value in state.items():
                        if state_name == "step":
                            self._validate_streaming_adam_step(
                                state_value,
                                parameter_name=parameter_name,
                                stage="after canonicalization",
                            )
                            continue
                        include(
                            "optimizer.%s.%s"
                            % (parameter_name, state_name),
                            state_value,
                        )
        for name, anchor in self.slow_anchors.items():
            if (
                allowed is not None
                and parameter_ids_by_name.get(name) not in allowed
            ):
                continue
            include("slow-anchor.%s" % name, anchor)
        for name, importance in self.slow_importance.items():
            if (
                allowed is not None
                and parameter_ids_by_name.get(name) not in allowed
            ):
                continue
            include("slow-importance.%s" % name, importance)
        maximum_delta = 0.0
        for label, tensor in tensors:
            maximum_delta = max(
                maximum_delta,
                self._canonicalize_streaming_float_tensor(tensor),
            )
            if (
                tensor.is_floating_point() or tensor.is_complex()
            ) and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(
                    "non-finite streaming learning state after "
                    "canonicalization: %s" % label
                )
        return maximum_delta

    def _optimize_streaming_experience_batch(
        self,
        experiences: Sequence[Tuple[str, torch.Tensor]],
        *,
        learning_schedule: Optional[Mapping[str, Any]] = None,
        schedule_locked: bool = False,
    ) -> Dict[str, float]:
        """Commit one accumulated slow update for exhaustive stream records.

        Each supplied record participates in a loss. Gradients accumulate over
        physical microbatches, then one optimizer step commits the group. A
        resumable ingestion supplies its frozen, source-free schedule so a RAM
        reading cannot silently change the parameter trajectory. No source
        text is retained after this call. One CPU uint8 rollback image of the
        connected packed modules is held for the entire logical batch, not
        each microbatch; this costs O(packed model bytes) RAM and device-copy
        I/O per batch. The live resource reserve can pause before that copy.
        """

        if not experiences:
            return {
                "loss": 0.0,
                "language_loss": 0.0,
                "idea_loss": 0.0,
                "workspace_loss": 0.0,
                "stability_loss": 0.0,
                "records": 0.0,
                "optimizer_steps": 0.0,
            }
        training_plan = self._training_resource_plan()
        if bool(training_plan["pauseBeforeStep"]):
            status = {
                **self.resource_policy.status(),
                "paused": True,
                "recoverable": True,
                "trainingResourcePlan": training_plan,
                "resumeFromLastCheckpoint": True,
                "sourceRecordsSkipped": False,
            }
            raise NeuralStateResourcePause(
                "training paused before allocating a step outside the safe Omni RAM envelope",
                status,
            )
        try:
            self._ensure_optimizer_resident()
        except BaseException as error:
            if not is_allocator_oom_error(error):
                raise
            self._allocator_oom_count += 1
            self._release_training_allocator_cache()
            raise self._allocator_resource_pause(
                error,
                stage="optimizer-rehydrate",
            ) from error
        self.decoder.train()
        self.memory_bridge.train()
        self.idea_adapter.train()
        self.liquid.train()
        if learning_schedule is None:
            physical = max(
                1,
                int(
                    training_plan.get(
                        "physicalBatchRecords",
                        self._runtime_train_batch_size,
                    )
                ),
            )
            sequence_tokens = max(
                8,
                int(
                    training_plan.get(
                        "windowTokens",
                        self._runtime_training_max_seq_len,
                    )
                ),
            )
        else:
            physical = max(
                1, int(learning_schedule["physicalBatchRecords"])
            )
            sequence_tokens = max(
                8, int(learning_schedule["trainingSequenceTokens"])
            )
        frozen_logical_batch_target = (
            max(1, int(learning_schedule["physicalBatchRecords"]))
            * max(1, int(learning_schedule["gradientAccumulation"]))
            if learning_schedule is not None
            else None
        )
        def reserve_packed_rollback(byte_count: int) -> None:
            status = self.resource_policy.status(estimated_ram_bytes=byte_count)
            if status["memoryPressure"]:
                raise NeuralStateResourcePause(
                    "training paused before packed rollback state crossed the RAM reserve",
                    {
                        **status,
                        "paused": True,
                        "recoverable": True,
                        "trainingResourcePlan": training_plan,
                        "packedRollbackBytes": byte_count,
                        "resumeFromLastCheckpoint": True,
                        "sourceRecordsSkipped": False,
                    },
                )

        try:
            packed_snapshot = PackedMutationSnapshot.capture(
                self._slow_transaction_modules().values(),
                reserve=reserve_packed_rollback,
            )
        except BaseException as error:
            if not is_allocator_oom_error(error):
                raise
            self._allocator_oom_count += 1
            self._release_training_allocator_cache()
            raise self._allocator_resource_pause(
                error, stage="packed-rollback-snapshot",
                physical_batch=physical, sequence_tokens=sequence_tokens,
            ) from error
        measurements: List[Tuple[Dict[str, float], int]] = []
        canonicalization_max_delta = 0.0
        while True:
            measurements = []
            completed_windows = 0
            mutation_stage = "forward-backward"
            cpu_rng_state = torch.get_rng_state().clone()
            accelerator_rng_state: Optional[torch.Tensor] = None
            if self.device_backend == "cuda" and torch.cuda.is_available():
                accelerator_rng_state = torch.cuda.get_rng_state(
                    self.device
                ).clone()
            elif (
                self.device_backend == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "get_rng_state")
            ):
                accelerator_rng_state = torch.mps.get_rng_state().clone()
            self._optimizer.zero_grad(set_to_none=True)
            try:
                for encoded, vectors in self._streaming_experience_window_batches(
                    experiences,
                    physical_batch=physical,
                    sequence_tokens=sequence_tokens,
                ):
                    loss, measured = self._experience_ids_batch_loss(
                        encoded, vectors
                    )
                    if self.device_backend == "mps":
                        # Checkpointed workspace chunks leave only inactive
                        # allocator blocks after forward. Release those blocks
                        # before backward recomputes each exact slot chunk.
                        self._release_training_allocator_cache()
                    # Each batch loss is a mean over its token windows. Weight
                    # it by the actual row count so a short final microbatch has
                    # the same objective as any other physical partition.
                    window_count = len(encoded)
                    (loss * float(window_count)).backward()
                    if self.device_backend == "mps":
                        # The next microbatch must not inherit disposable MPS
                        # cache from checkpoint recomputation.
                        self._release_training_allocator_cache()
                    completed_windows += window_count
                    measurements.append((measured, window_count))
                if completed_windows < 1:
                    raise ValueError("training batch contained no token windows")
                named_parameters = {
                    id(parameter): name
                    for name, parameter in self._named_slow_parameters().items()
                }
                active_parameters: List[
                    Tuple[str, nn.Parameter, Mapping[str, Any]]
                ] = []
                for group_index, group in enumerate(
                    self._optimizer.param_groups
                ):
                    for parameter_index, parameter in enumerate(
                        group["params"]
                    ):
                        if parameter.grad is not None:
                            parameter.grad.div_(float(completed_windows))
                            parameter_name = named_parameters.get(
                                id(parameter),
                                "optimizer-group-%d-parameter-%d"
                                % (group_index, parameter_index),
                            )
                            if not bool(
                                torch.isfinite(parameter.grad).all()
                            ):
                                raise RuntimeError(
                                    "non-finite gradient before slow-importance: %s"
                                    % parameter_name
                                )
                            active_parameters.append(
                                (parameter_name, parameter, group)
                            )
                mutation_stage = "pre-step-state-validation"
                self._validate_streaming_optimizer_state(
                    active_parameters,
                    stage="before optimizer step",
                    require_state=False,
                )
                parameters = [
                    parameter for _, parameter, _ in active_parameters
                ]
                mutation_stage = "slow-importance"
                self._accumulate_slow_importance(parameters)
                mutation_stage = "gradient-clipping"
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.config.grad_clip,
                    error_if_nonfinite=True,
                )
                gradient_norm_value = (
                    float(gradient_norm.detach().item())
                    if isinstance(gradient_norm, torch.Tensor)
                    else float(gradient_norm)
                )
                if not math.isfinite(gradient_norm_value):
                    raise RuntimeError(
                        "non-finite total gradient norm after clipping"
                    )
                mutation_stage = "optimizer-step"
                self._optimizer.step()
                mutation_stage = "post-step-finite-validation"
                self._validate_streaming_optimizer_state(
                    active_parameters,
                    stage="after optimizer step",
                    require_state=True,
                )
                mutation_stage = "slow-anchor-commit"
                self._commit_slow_anchors(
                    rate=0.08,
                    parameters=parameters,
                )
                mutation_stage = "canonicalize-learning-state"
                canonicalization_max_delta = (
                    self._canonicalize_streaming_learning_state(parameters)
                )
                self._validate_streaming_optimizer_state(
                    active_parameters,
                    stage="after canonicalization",
                    require_state=True,
                )
                break
            except BaseException as error:
                if is_allocator_oom_error(error):
                    self._release_training_allocator_cache()
                try:
                    packed_snapshot.restore()
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "packed logical-batch rollback failed; refusing retry"
                    ) from rollback_error
                self._optimizer.zero_grad(set_to_none=True)
                if not is_allocator_oom_error(error):
                    raise
                self._allocator_oom_count += 1
                # Backward may have changed packed synapses; exact uint8
                # rollback above is required before replaying this batch.
                # Once slow importance or an optimizer step begins, only the
                # worker's atomic-generation rollback may recover float state.
                if mutation_stage == "forward-backward":
                    torch.set_rng_state(cpu_rng_state)
                    if accelerator_rng_state is not None:
                        if self.device_backend == "cuda":
                            torch.cuda.set_rng_state(
                                accelerator_rng_state, self.device
                            )
                        elif self.device_backend == "mps":
                            torch.mps.set_rng_state(accelerator_rng_state)
                if (
                    mutation_stage == "forward-backward"
                    and not schedule_locked
                    and physical > 1
                ):
                    if frozen_logical_batch_target is None:
                        physical = max(1, physical // 2)
                    else:
                        # A scheduled ingestion publishes physical batch and
                        # accumulation together. Select the next exact divisor
                        # before replaying any forward work, so the successful
                        # optimizer mutation can always be frozen without a
                        # later post-step rejection. This also preserves useful
                        # headroom for non-power-of-two targets (3 -> 2 for 6)
                        # while safely taking a prime target such as 5 to 1.
                        physical = next(
                            candidate
                            for candidate in range(physical - 1, 0, -1)
                            if frozen_logical_batch_target % candidate == 0
                        )
                    self._runtime_train_batch_size = physical
                    continue
                if (
                    mutation_stage == "forward-backward"
                    and not schedule_locked
                    and sequence_tokens > 8
                ):
                    sequence_tokens = max(8, sequence_tokens // 2)
                    self._runtime_training_max_seq_len = sequence_tokens
                    continue
                raise self._allocator_resource_pause(
                    error,
                    stage=mutation_stage,
                    physical_batch=physical,
                    sequence_tokens=sequence_tokens,
                ) from error
        self._optimizer.zero_grad(set_to_none=True)
        self.counters["training_steps"] += 1
        self._clear_allocator_recovery_pause()
        self._maintain_neural_state_resources()
        keys = (
            "loss",
            "language_loss",
            "idea_loss",
            "workspace_loss",
            "stability_loss",
        )
        result = {
            key: sum(float(item[key]) * count for item, count in measurements)
            / float(max(1, sum(count for _, count in measurements)))
            for key in keys
        }
        result["records"] = float(len(experiences))
        result["optimizer_steps"] = 1.0
        result["physical_batch_records"] = float(physical)
        result["training_sequence_tokens"] = float(sequence_tokens)
        result["canonicalization_max_delta"] = canonicalization_max_delta
        return result

    def _maybe_grow(self, novelty: float, prototype: torch.Tensor) -> bool:
        if novelty >= self.config.growth_novelty_threshold:
            self.novelty_streak += 1
        else:
            self.novelty_streak = max(0, self.novelty_streak - 1)
        if self.novelty_streak < self.config.growth_patience:
            return False
        expert_parameters = (
            self.config.d_model * max(16, self.config.d_ff // 2) * 3
        )
        estimated_bytes = expert_parameters * 12
        if not self._allow_substrate_growth(estimated_bytes):
            return False
        self.decoder.grow_expert(prototype.detach().reshape(-1))
        self._configure_packed_stability()
        self.novelty_streak = 0
        self.growth_pause = None
        self._replace_optimizer()
        self._sync_stability_state()
        return True

    def _maybe_expand_memory(self) -> bool:
        """Compatibility hook; the substrate now grows directly as it learns."""

        return False

    def _allow_substrate_growth(self, estimated_bytes: int) -> bool:
        """Apply host reserve watermarks without imposing neuron-count caps."""

        estimated_bytes = max(1, int(estimated_bytes))
        policy = self.resource_policy.status(
            estimated_write_bytes=estimated_bytes * 2,
            estimated_ram_bytes=estimated_bytes,
        )
        readings = self._resource_readings()
        reason = ""
        if policy["diskPressure"]:
            reason = "available disk is below the neural growth reserve"
        elif policy["memoryPressure"]:
            # First spill eligible scratch, then sample the same projected
            # growth again. A successful spill must not leave this operation
            # falsely marked blocked from the pre-spill reading.
            try:
                self._maintain_neural_state_resources()
            except NeuralStateResourcePause:
                pass
            policy = self.resource_policy.status(
                estimated_write_bytes=estimated_bytes * 2,
                estimated_ram_bytes=estimated_bytes,
            )
            readings = self._resource_readings()
            if policy["diskPressure"]:
                reason = "available disk is below the neural growth reserve"
            elif policy["memoryPressure"]:
                reason = "available memory is below the neural growth reserve"
        if reason:
            self.growth_pause = {
                "reason": reason,
                "readings": {**readings, "policy": policy},
                "estimatedGrowthBytes": estimated_bytes,
                "at": _iso_now(),
            }
            return False
        self.growth_pause = None
        return True

    def _resource_readings(self) -> Dict[str, Any]:
        # One live policy sample keeps RAM and disk fields correlated and
        # exposes the same projected/reserve values used by actual allocation
        # gates. Callers must not invent a smaller feature-specific disk floor.
        return dict(self.resource_policy.status())

    def _disk_space_telemetry(
        self,
        *,
        estimated_write_bytes: int = 0,
    ) -> Dict[str, Any]:
        status = self.resource_policy.status(
            estimated_write_bytes=max(0, int(estimated_write_bytes))
        )
        free = max(0, int(status["diskFreeBytes"]))
        total = max(free, int(status["diskTotalBytes"]))
        reserve = max(0, int(status["diskReserveBytes"]))
        projected = max(0, int(status["projectedDiskFreeBytes"]))
        return {
            "schemaVersion": 1,
            "measuredAt": _iso_now(),
            "diskTotalBytes": total,
            "diskFreeBytes": free,
            "mandatoryReserveBytes": reserve,
            "selectedDatasetBytes": 0,
            "modelBytes": 0,
            "checkpointBytes": 0,
            "maximumWorkingMemorySpillBytes": 0,
            "futureGrowthBytes": 0,
            "operationWriteBytes": max(0, int(estimated_write_bytes)),
            "projectedRemainingBytes": projected,
            "projectedAboveReserveBytes": max(0, projected - reserve),
            "paused": bool(status["diskPressure"]),
        }

    def _training_resource_plan(self) -> Dict[str, Any]:
        """Measure the RAM-first envelope for the next exhaustive update."""

        parameters = {
            id(parameter): parameter
            for module in self._trainable_modules()
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        trainable_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in parameters.values()
        )
        packed_update_scratch_bytes = 0
        for root in self._trainable_modules():
            for module in root.modules():
                packed_tensors = getattr(
                    module, "authoritative_packed_tensors", None
                )
                if not callable(packed_tensors):
                    continue
                weights = packed_tensors()
                if not weights or weights[0].ndim != 2:
                    continue
                rows, packed_columns = weights[0].shape
                # One row block is decoded/graded at a time. Four logical
                # levels share each byte; 16 bytes per active level covers
                # local float gradient, probability, draw, and code scratch.
                # This is a maximum transient block, not a full FP32 mirror.
                packed_update_scratch_bytes = max(
                    packed_update_scratch_bytes,
                    min(64, int(rows)) * int(packed_columns) * 4 * 16,
                )
        activation_multiplier = (
            24 if self.config.gradient_checkpointing else 48
        )
        activation_bytes_per_token = max(
            16 * 1024,
            int(self.config.d_model)
            * max(1, int(self.config.n_layers))
            * activation_multiplier,
        )
        effective_batch_target = max(
            1,
            int(self.config.train_batch_size)
            * int(self.config.gradient_accumulation),
        )
        auto_divisor_policy = self.config.training_resource_mode == "auto"
        requested_physical_batch = (
            min(4, effective_batch_target)
            if auto_divisor_policy
            else min(
                int(self.config.train_batch_size),
                int(self._runtime_train_batch_size),
            )
        )
        plan = self.resource_policy.training_plan(
            max_window_tokens=min(
                int(self.config.max_seq_len),
                int(self._runtime_training_max_seq_len),
            ),
            requested_batch_size=requested_physical_batch,
            requested_gradient_accumulation=int(
                self.config.gradient_accumulation
            ),
            effective_batch_target=(
                effective_batch_target if auto_divisor_policy else None
            ),
            require_physical_batch_divisor=auto_divisor_policy,
            trainable_parameter_bytes=trainable_bytes,
            packed_update_scratch_bytes=packed_update_scratch_bytes,
            activation_bytes_per_token=activation_bytes_per_token,
            optimizer_state_resident=bool(self._optimizer.state)
            and not self._optimizer_offloaded,
            resource_mode=self.config.training_resource_mode,
            manual_ram_budget_bytes=self.config.training_ram_budget_bytes,
            manual_accelerator_budget_bytes=(
                self.config.training_accelerator_budget_bytes
            ),
            manual_scratch_budget_bytes=(
                self.config.training_scratch_budget_bytes
            ),
            storage_bytes_per_second=self.config.storage_bytes_per_second,
            disk_state_offload=self.config.disk_state_offload,
        )
        # Live pressure may choose a smaller physical microbatch/window for
        # this operation, but it must not become a monotonic process-lifetime
        # context downgrade. The runtime ceilings change only after a real
        # allocator refusal; a later plan can grow back toward the persisted
        # configuration when other applications release memory.
        plan["configuredContextTokens"] = int(self.config.max_seq_len)
        plan["capacityPersistsAcrossPressure"] = True
        plan["contextWindowShrunk"] = False
        plan["autoPhysicalBatchDivisorPolicy"] = auto_divisor_policy
        plan["autoPhysicalBatchCandidate"] = (
            requested_physical_batch if auto_divisor_policy else None
        )
        return plan

    def _streaming_neural_storage_plan(
        self, source_bytes: int
    ) -> Dict[str, Any]:
        """Keep detailed neural encoding independent of total source size.

        The old whole-source estimate multiplied compressed bytes by every
        potential structure, then silently mapped a large corpus to a shared
        semantic field. That changed what the brain could learn from each
        record. Admit one bounded checkpoint window instead; every record
        remains eligible for its own distributed assembly. A real RAM/disk
        refusal pauses at the last committed cursor, never downgrades the
        representation or skips the remainder of a dataset.
        """

        source_bytes = max(0, int(source_bytes))
        status = self.resource_policy.status()
        training_plan = self._training_resource_plan()
        disk_headroom = max(
            0,
            int(status["diskFreeBytes"]) - int(status["diskReserveBytes"]),
        )
        available_memory = status.get("availableMemoryBytes")
        ram_reserve = int(status["ramReserveBytes"])
        ram_headroom = (
            max(0, int(available_memory) - ram_reserve)
            if isinstance(available_memory, int)
            else disk_headroom
        )
        system_ram_budget = int(status.get("systemRamBudgetBytes", 0) or 0)
        nominal_ram_capacity = (
            system_ram_budget if system_ram_budget > 0 else ram_headroom
        )
        # This is only a bounded-window admission estimate, not a claim that
        # every future record is the same size. Actual growth and checkpoint
        # writes remain guarded at their own allocation boundaries.
        projected_detailed_bytes = max(
            64 * 1024,
            int(self._ingestion_checkpoint_records)
            * (max(64, (self.config.vsa_dim + 3) // 4) + 2048),
        )
        detailed_budget = min(disk_headroom, nominal_ram_capacity)
        detailed = True
        detailed_admission = self.resource_policy.status(
            estimated_write_bytes=projected_detailed_bytes * 2,
            estimated_ram_bytes=projected_detailed_bytes,
        )
        detailed_deferred = bool(
            detailed
            and (
                detailed_admission["diskPressure"]
                or detailed_admission["memoryPressure"]
            )
        )
        return {
            "policy": "bounded-window-detailed-neural-representation",
            "sourceBytes": source_bytes,
            "projectedDetailedBytes": projected_detailed_bytes,
            "detailedBudgetBytes": detailed_budget,
            "projectionScope": "next-checkpoint-window-not-whole-source",
            "nominalRamCapacityBytes": nominal_ram_capacity,
            "diskHeadroomBytes": disk_headroom,
            "ramHeadroomBytes": ram_headroom,
            "detailedRecordAssemblies": detailed,
            "detailedRepresentationPreferred": detailed,
            "detailedRepresentationDeferred": detailed_deferred,
            "representationDecision": (
                "detailed-awaiting-resources"
                if detailed_deferred
                else "detailed-admitted"
            ),
            "representationDowngradedForTransientPressure": False,
            "detailedAdmissionStatus": detailed_admission,
            "corpusRepresentation": "detailed-distributed-assemblies",
            "slowGradientMode": "per-experience",
            "physicalBatchRecords": int(training_plan["physicalBatchRecords"]),
            "gradientAccumulation": int(training_plan["gradientAccumulation"]),
            "trainingSequenceTokens": int(
                training_plan["windowTokens"]
            ),
            "trainingResourcePlan": training_plan,
            "localTypedTargetWindowPolicy": (
                LOCAL_TYPED_TARGET_WINDOW_POLICY
            ),
            "recordCardinalityLimit": None,
            "silentRecordSkipping": False,
            "resourceStatus": status,
        }

    def _ensure_paged_ingestion_substrate(
        self, source_bytes: int, *, force: bool = False
    ) -> None:
        """Page packed state when physical pressure warrants it.

        A high-RAM brain may keep its live assemblies resident while its saved
        generation remains packed v3 shards. Paging is a physical choice,
        never a change in semantic detail or a prerequisite for learning.
        The working SQLite file is derived; the next save must publish its
        neural rows through packed v3 shards and brain.json before the cursor
        can advance. A stale cache fails closed for explicit reconciliation.
        """

        from .live_paging_migration import migrate_live_substrate_to_paged
        from .paged_assembly_vector_view import PagedAssemblyVectorView
        from .paged_assembly_view import PagedAssemblyView
        from .paged_packed_vectors import PagedPackedVectors

        if (
            isinstance(self.memory.assemblies, PagedAssemblyView)
            and isinstance(self.memory.neuron_vectors, PagedPackedVectors)
        ):
            index = self.memory.assemblies.index
            vectors = self.memory.neuron_vectors
            view = self.memory.assembly_vectors
            if (
                index._vectors is not vectors
                or index.path.resolve() != vectors.path.resolve()
                or not isinstance(view, PagedAssemblyVectorView)
                or view.index is not index
                or view.backing is not vectors
            ):
                raise ValueError("paged neural regions lost their shared row authority")
            self._paged_substrate_required = True
            return
        readings = self.resource_policy.status()
        if not self.config.disk_state_offload:
            if readings["memoryPressure"]:
                raise NeuralStateResourcePause(
                    "resident neural state reached its RAM reserve and disk "
                    "offload is disabled", readings,
                )
            return
        if readings["diskPressure"]:
            raise NeuralStateResourcePause(
                "paged neural ingestion reached the mandatory disk reserve",
                readings,
            )
        system_budget = int(readings.get("systemRamBudgetBytes", 0) or 0)
        resident_projection = (
            len(self.memory.neurons)
            * ((self.config.vsa_dim + 3) // 4 + 96)
            + len(self.memory.assemblies) * 1536
        )
        page_worthwhile = bool(
            force
            or readings["memoryPressure"]
            or (
                system_budget > 0
                and (
                    resident_projection > system_budget // 8
                    or max(0, int(source_bytes)) > system_budget // 4
                )
            )
        )
        if not page_worthwhile:
            return
        try:
            migrate_live_substrate_to_paged(
                self.memory,
                self._live_paging_cache_directory,
                disk_reserve=self.resource_policy.require_disk,
            )
        except SubstrateResourcePause as error:
            raise NeuralStateResourcePause(
                str(error), self.resource_policy.status()
            ) from error
        self._paged_substrate_required = True

    def _preview_chat_experience(
        self,
        text: str,
        cue: torch.Tensor,
        recalled: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Compute generation features without committing the candidate turn."""

        labels = self.memory.extract_concepts(text)
        if not labels:
            labels = ["empty-experience"]
        concept_ids = [
            hashlib.sha256(
                ("semantic:%s" % label).encode("utf-8")
            ).hexdigest()[:24]
            for label in labels
        ]
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assembly_id = hashlib.sha256(
            ("assembly:" + fingerprint).encode("ascii")
        ).hexdigest()[:24]
        nearest = max(
            (float(item.get("score", 0.0)) for item in recalled),
            default=-1.0,
        )
        novelty = max(0.0, min(1.0, 1.0 - max(0.0, nearest)))
        with torch.no_grad():
            idea = self._idea_model_vector(cue)
            if self.config.liquid_dynamics:
                _next_liquid_state, controls = self.liquid(
                    idea,
                    state=self.liquid_state.detach(),
                    elapsed=1.0,
                )
            else:
                controls = {
                    "retention": torch.ones(1, device=self.device),
                    "threshold_offset": torch.zeros(1, device=self.device),
                    "noise_scale": torch.ones(1, device=self.device),
                    "ponder_scale": torch.ones(1, device=self.device),
                }
            threshold = float(
                controls["threshold_offset"].detach().mean().item()
            )
            if self.config.spiking_dynamics:
                membrane = self.router.population.membrane.detach().clone()
                spike_count = (
                    self.router.population.spike_count.detach().clone()
                )
                try:
                    routed, spike_metrics = self.router.route(
                        idea,
                        steps=max(
                            2,
                            int(
                                round(
                                    float(
                                        controls["ponder_scale"].mean().item()
                                    )
                                )
                            ),
                        ),
                        learn=False,
                        threshold_offset=threshold,
                    )
                finally:
                    self.router.population.membrane.copy_(membrane)
                    self.router.population.spike_count.copy_(spike_count)
            else:
                routed = idea
                spike_metrics = {
                    "spike_rate": 0.0,
                    "spikes": 0.0,
                    "stdp_update": 0.0,
                    "mean_stability": 0.0,
                    "active_synapses": 0.0,
                }
        return {
            "idea_id": assembly_id,
            "assembly_id": assembly_id,
            "concept_ids": concept_ids,
            "neuron_ids": concept_ids,
            "labels": labels,
            "assemblies_created": 0,
            "novelty": novelty,
            "idea": routed.detach(),
            "spiking": spike_metrics,
            "liquid_controls": {
                key: float(value.detach().mean().item())
                for key, value in controls.items()
            },
            "training": {
                "loss": 0.0,
                "language_loss": 0.0,
                "idea_loss": 0.0,
                "stability_loss": 0.0,
            },
            "memory_settling": None,
            "grew_expert": False,
            "state_offload": None,
            "preview": True,
        }

    def learn_experience(
        self,
        text: str,
        kind: str = "knowledge",
        source: str = "conversation",
        source_label: str = "",
        steps: Optional[int] = None,
        importance: float = 0.5,
        structural_detail: bool = True,
    ) -> Dict[str, Any]:
        """Learn a whole experience through shared activity, synapses and cortex."""

        return self._learn_experience_impl(
            text,
            kind=kind,
            source=source,
            source_label=source_label,
            steps=steps,
            importance=importance,
            structural_detail=structural_detail,
        )

    def _learn_experience_impl(
        self,
        text: str,
        *,
        kind: str,
        source: str,
        source_label: str,
        steps: Optional[int],
        importance: float,
        structural_detail: bool,
    ) -> Dict[str, Any]:
        now = time.time()
        elapsed = max(0.0, now - self.last_activity_decay)
        half_life = self.config.short_term_half_life_minutes * 60.0
        if elapsed > 0 and self.config.vector_symbolic_memory:
            activity_decay = 1.0 - math.exp(
                -math.log(2.0) * elapsed / max(half_life, 1.0)
            )
            # Per-turn elapsed-time settling decays transient neuron activity.
            # Synaptic forgetting remains global during explicit consolidation
            # and local to causally active edges in organic memory settling.
            self.memory.decay(min(activity_decay, 0.25), synapses=())
        self.last_activity_decay = now
        if self.config.vector_symbolic_memory:
            try:
                if structural_detail:
                    learned = self.memory.learn(
                        text,
                        kind=kind,
                        source=source,
                        source_label=source_label,
                        # Exact source bytes live only in the desktop's
                        # content-addressed store. Neural assemblies retain a
                        # fingerprint and learned state, never an inline copy.
                        retain_source_text=False,
                        importance=importance,
                    )
                else:
                    learned = self.memory.learn_statistical(
                        text,
                        kind=kind,
                        source=source,
                        source_label=source_label,
                        importance=importance,
                    )
            except SubstrateResourcePause:
                self.events.append(
                    "substrate-growth-paused",
                    {
                        "reason": (
                            self.growth_pause or {}
                        ).get("reason", "host resource reserve"),
                        "readings": self._resource_readings(),
                    },
                )
                raise
        else:
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            learned = {
                "idea_id": "transient-" + fingerprint[:20],
                "vector": self.memory.space.symbol("experience:" + fingerprint),
                "novelty": 1.0,
                "concept_ids": [],
                "labels": [],
            }
        idea = self._idea_model_vector(learned["vector"])
        if self.config.liquid_dynamics:
            self.liquid_state, controls = self.liquid(
                idea, state=self.liquid_state.detach(), elapsed=1.0
            )
            self.liquid_state = self.liquid_state.detach()
        else:
            controls = {
                "retention": torch.ones(1, device=self.device),
                "threshold_offset": torch.zeros(1, device=self.device),
                "noise_scale": torch.ones(1, device=self.device),
                "ponder_scale": torch.ones(1, device=self.device),
            }
        threshold = float(controls["threshold_offset"].detach().mean().item())
        if self.config.spiking_dynamics:
            routed, spike_metrics = self.router.route(
                idea,
                steps=max(
                    2, int(round(float(controls["ponder_scale"].mean().item())))
                ),
                learn=self.config.stdp_plasticity,
                threshold_offset=threshold,
            )
        else:
            routed = idea
            spike_metrics = {
                "spike_rate": 0.0,
                "spikes": 0.0,
                "stdp_update": 0.0,
                "mean_stability": 0.0,
                "active_synapses": 0.0,
            }
        requested_steps = self.config.online_steps if steps is None else steps
        slow_learning_applied = bool(
            self.config.online_learning and int(requested_steps) > 0
        )
        if slow_learning_applied:
            train_result = self._optimize_experience(
                text,
                learned["vector"],
                steps=int(requested_steps),
            )
        else:
            train_result = {
                "loss": 0.0,
                "language_loss": 0.0,
                "idea_loss": 0.0,
                "stability_loss": 0.0,
            }
        workspace_salience = max(
            0.0,
            min(
                1.0,
                0.4 * float(learned["novelty"])
                + 0.35 * float(importance)
                + 0.25 * float(spike_metrics["spike_rate"]),
            ),
        )
        prediction_error = self._prediction_error_from_loss(
            train_result.get("loss", 0.0)
        )
        self._append_replay(
            routed,
            importance=importance,
            replay_priority=self._organic_replay_priority(
                assembly_id=str(learned["idea_id"]),
                salience=workspace_salience,
                novelty=float(learned["novelty"]),
                prediction_error=prediction_error,
                spike_rate=float(spike_metrics["spike_rate"]),
            ),
            assembly_id=str(learned["idea_id"]),
        )
        self._append_working_memory(
            routed,
            assembly_id=str(learned["idea_id"]),
            source=source,
            salience=workspace_salience,
        )
        memory_settling = self._settle_memory_automatically(
            routed,
            assembly_id=str(learned["idea_id"]),
            source=source,
            salience=workspace_salience,
            novelty=float(learned["novelty"]),
            prediction_error=prediction_error,
            importance=importance,
            spike_rate=float(spike_metrics["spike_rate"]),
        )
        # Expert allocation changes decoder topology and therefore belongs to
        # the same slow-mutation transaction as gradient learning. Fast-only
        # experiences (steps=0) may update substrate/STDP/working activity, but
        # must not append random decoder parameters before an action decision.
        grew = (
            self._maybe_grow(float(learned["novelty"]), routed[0])
            if slow_learning_applied
            else False
        )
        self.counters["experiences"] += 1
        self.counters["plasticity_events"] = int(
            self.router.synapses.plasticity_events.item()
        )
        offload = self._maintain_neural_state_resources()
        result = {
            "idea_id": learned["idea_id"],
            "assembly_id": learned.get("assembly_id", learned["idea_id"]),
            "concept_ids": learned["concept_ids"],
            "neuron_ids": learned.get("neuron_ids", learned["concept_ids"]),
            "labels": learned["labels"],
            "assemblies_created": int(learned.get("assemblies_created", 1)),
            "novelty": learned["novelty"],
            "idea": routed.detach(),
            "spiking": spike_metrics,
            "liquid_controls": {
                key: float(value.detach().mean().item())
                for key, value in controls.items()
            },
            "training": train_result,
            "memory_settling": memory_settling,
            "grew_expert": grew,
            "state_offload": offload,
        }
        return result

    def _latent_rehearsal_step(
        self, substrate_vector: torch.Tensor, seed: int
    ) -> Dict[str, float]:
        """Consolidate an internal assembly without manufacturing prompt text."""

        self.decoder.train()
        self.memory_bridge.train()
        self.idea_adapter.train()
        self.liquid.train()
        self._ensure_optimizer_resident()
        self._optimizer.zero_grad(set_to_none=True)
        learning_parameters = tuple(
            parameter
            for module in (
                self.memory_bridge,
                self.idea_adapter,
                self.liquid,
            )
            for parameter in module.parameters()
        )
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            idea = self._idea_model_vector(substrate_vector)
            state_noise = torch.randn_like(idea) * 0.025
            reconstructed = self.idea_adapter(idea + state_noise)
            temporal, _ = self.liquid(
                idea,
                state=self.liquid_state.detach(),
                elapsed=1.0,
            )
            reconstruction_loss = F.mse_loss(reconstructed, idea.detach())
            temporal_loss = F.mse_loss(temporal, idea.detach())
            stability_loss = self._stability_penalty(
                learning_parameters
            )
            loss = (
                reconstruction_loss
                + 0.15 * temporal_loss
                + stability_loss
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite idle rehearsal loss")
        loss.backward()
        parameters = [
            parameter for parameter in learning_parameters
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
        self._accumulate_slow_importance(parameters)
        self._optimizer.step()
        self._maintain_neural_state_resources()
        self._commit_slow_anchors(
            rate=0.03, parameters=learning_parameters
        )
        self.counters["training_steps"] += 1
        return {
            "loss": float(loss.detach().item()),
            "reconstructionLoss": float(reconstruction_loss.detach().item()),
            "temporalLoss": float(temporal_loss.detach().item()),
            "stabilityLoss": float(stability_loss.detach().item()),
        }

    def feedback(
        self,
        text: str,
        direction: str,
        *,
        trace_id: str = "",
        message_id: str = "",
    ) -> Dict[str, Any]:
        """Integrate user feedback through local timing and ordinary learning.

        This path has no reward model, preference classifier, or RLHF loss.
        Positive feedback causally replays active assemblies and may run one
        normal corpus-prediction step. Negative feedback reverses spike timing
        to weaken the association without training a refusal/persona target.
        """

        clean = text.replace("\x00", "").strip()
        if not clean:
            raise ValueError("feedback text cannot be empty")
        if len(clean) > 1_000_000:
            raise ValueError("feedback text is too large")
        if direction not in {"up", "down"}:
            raise ValueError("feedback direction must be up or down")
        sign = 1 if direction == "up" else -1
        cue = self.memory.vector_for_text(clean)
        recalled_vector, recalled = self.memory.recall_vector(
            cue, workspace_slots=self.config.working_memory_slots
        )
        active_vector = recalled_vector if recalled else cue
        idea = self._idea_model_vector(active_vector).detach()
        parameter_before = self.parameter_checksum()
        synapse_before = tensor_checksum([self.router.synapses.weights])
        stdp = self.router.apply_feedback(idea, sign)
        slow_learning: Optional[Dict[str, float]] = None
        if direction == "up" and self.config.online_learning:
            slow_learning = self._optimize_experience(
                clean,
                active_vector,
                steps=1,
                commit_stability=True,
            )

        # Confidence/uncertainty are inspection metadata derived from the same
        # neural assemblies. They are not a second authoritative memory store.
        for recalled_item in recalled:
            assembly_id = str(recalled_item.get("assembly_id", ""))
            if assembly_id in self.memory.neurons:
                def revise_feedback_node(node: Dict[str, Any]) -> None:
                    current_activation = self.memory.effective_activation(node)
                    uncertainty = float(node.get("uncertainty", 0.5))
                    node["uncertainty"] = max(
                        0.0, min(1.0, uncertainty - sign * 0.04)
                    )
                    node["activation"] = max(
                        0.0, min(1.0, current_activation + sign * 0.03)
                    )

                self.memory.edit_neuron_by_id(
                    assembly_id, revise_feedback_node
                )
                self.memory.mark_attention_neuron(assembly_id)
        self._append_working_memory(
            idea,
            assembly_id=(
                str(recalled[0].get("assembly_id", "")) if recalled else ""
            ),
            source="feedback",
            salience=0.82 if direction == "up" else 0.45,
        )
        if direction == "up":
            self._append_replay(idea, importance=0.82)
        self.counters["plasticity_events"] = int(
            self.router.synapses.plasticity_events.item()
        )
        parameter_after = self.parameter_checksum()
        synapse_after = tensor_checksum([self.router.synapses.weights])
        record = {
            "id": uuid.uuid4().hex,
            "createdAt": _iso_now(),
            "direction": direction,
            "messageId": message_id,
            "traceId": trace_id,
            "textSha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
            "recalledAssemblyIds": [
                str(item.get("assembly_id", "")) for item in recalled
            ],
            "stdp": stdp,
            "slowLearning": slow_learning,
            "parameterChecksumBefore": parameter_before,
            "parameterChecksumAfter": parameter_after,
            "synapseChecksumBefore": synapse_before,
            "synapseChecksumAfter": synapse_after,
            "rewardModel": False,
            "rlhf": False,
        }
        self.events.append("neural-feedback", record)
        self.save()
        return {
            "brainId": self.brain_id,
            **record,
            "metrics": self.metrics(),
        }

    def idle_cycle(
        self,
        *,
        tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
        minimum_idle_seconds: float = 45.0,
    ) -> Dict[str, Any]:
        """Run one organic, prompt-free recurrent cognition cycle."""

        now = time.time()
        minimum_idle_seconds = max(0.0, float(minimum_idle_seconds))
        since_activity = now - max(
            self.last_activity_decay, self.last_idle_cycle_at
        )
        if since_activity < minimum_idle_seconds:
            return {
                "brainId": self.brain_id,
                "ran": False,
                "reason": "cooldown",
                "retryAfterSeconds": max(
                    0.0, minimum_idle_seconds - since_activity
                ),
                "actions": [],
            }
        available_assemblies = [
            assembly
            for assembly in self.memory.assemblies
            if str(assembly.get("id", "")) in self.memory.assembly_vectors
        ]
        if not available_assemblies:
            self.last_idle_cycle_at = now
            return {
                "brainId": self.brain_id,
                "ran": False,
                "reason": "no-learned-assemblies",
                "actions": [],
            }

        organic = self._organic_state()
        ranked = sorted(
            available_assemblies,
            key=lambda assembly: (
                self.memory.effective_activation(
                    self.memory.neurons.get(
                        str(assembly.get("id", "")), {}
                    )
                )
                * 0.35
                + float(assembly.get("importance", 0.0)) * 0.30
                + float(
                    self.memory.neurons.get(
                        str(assembly.get("id", "")), {}
                    ).get("uncertainty", 0.5)
                )
                * 0.20
                + 1.0
                / (1.0 + max(0, int(assembly.get("rehearsals", 0))))
                * 0.15
            ),
            reverse=True,
        )
        workspace_pressure = max(1, self.config.working_memory_slots)
        active = ranked[:workspace_pressure]
        vectors = [
            self.memory.assembly_vectors[str(assembly["id"])]
            for assembly in active
        ]
        weights = [
            max(
                0.05,
                float(assembly.get("importance", 0.0))
                + 1.0
                / (1.0 + max(0, int(assembly.get("rehearsals", 0)))),
            )
            for assembly in active
        ]
        substrate_vector = self.memory.space.weighted_bundle(vectors, weights)
        model_idea = self._idea_model_vector(substrate_vector)
        self.liquid_state, controls = self.liquid(
            model_idea,
            state=self.liquid_state.detach(),
            elapsed=max(1.0, since_activity),
        )
        self.liquid_state = self.liquid_state.detach()
        ponder_scale = float(controls["ponder_scale"].detach().mean().item())
        compute_demand = max(
            0.0,
            min(
                1.0,
                0.38 * float(organic["tension"])
                + 0.30 * float(organic["curiosity"])
                + 0.20 * float(organic["uncertainty"])
                + 0.12 * max(0.0, ponder_scale - 1.0),
            ),
        )
        threshold_offset = float(
            controls["threshold_offset"].detach().mean().item()
        )
        routed, spike_metrics = self.router.route(
            model_idea.detach(),
            steps=max(2, int(round(ponder_scale + 3.0 * compute_demand))),
            learn=True,
            threshold_offset=threshold_offset,
        )
        before_parameters = self._parameter_copy()
        parameter_before = self.parameter_checksum()
        seed_material = "%s:%d:%s" % (
            self.brain_id,
            self.counters["idle_cognition_cycles"],
            ",".join(str(assembly["id"]) for assembly in active),
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            "little",
        ) & 0x7FFFFFFF
        rehearsal = self._latent_rehearsal_step(substrate_vector, seed)
        parameter_after = self.parameter_checksum()
        parameter_delta = self._parameter_delta_norm(before_parameters)
        self._append_working_memory(
            routed,
            assembly_id=str(active[0]["id"]),
            source="idle",
            salience=max(0.2, compute_demand),
        )
        self._append_replay(
            routed,
            importance=max(0.4, compute_demand),
            replay_priority=self._organic_replay_priority(
                assembly_id=str(active[0]["id"]),
                salience=max(0.2, compute_demand),
                novelty=float(organic["novelty"]),
                prediction_error=float(
                    organic.get("predictionError", organic.get("uncertainty", 0.0))
                ),
                spike_rate=float(spike_metrics["spike_rate"]),
            ),
            assembly_id=str(active[0]["id"]),
        )
        memory_settling = self._settle_memory_automatically(
            routed,
            assembly_id=str(active[0]["id"]),
            source="rest",
            salience=max(0.2, compute_demand),
            novelty=float(organic["novelty"]),
            prediction_error=float(
                organic.get("predictionError", organic.get("uncertainty", 0.0))
            ),
            importance=float(active[0].get("importance", 0.5)),
            spike_rate=float(spike_metrics["spike_rate"]),
            resting=True,
        )
        normalized_tools = self._normalize_tool_schemas(tool_schemas)
        action_state = {
            **organic,
            "computeDemand": compute_demand,
            "ponderScale": ponder_scale,
            # This is typed internal state, not prompt text.  It distinguishes
            # an actual idle wake cycle from a human turn whose text happens
            # to be empty or from backend-generated focus labels.
            "promptFree": 1.0,
        }
        with torch.no_grad():
            # The bundled internal-action trajectories are calibrated on the
            # assembly/model channel.  The spiking router is intentionally
            # plastic and stateful, so feeding its post-spike output into the
            # same classifier creates an out-of-distribution feature shift
            # after ordinary learning (observed as a near-certain idle
            # Ponder).  Keep routed activity responsible for recurrent memory
            # and STDP, while selecting typed actions from the feature channel
            # on which the head was actually trained.
            action_scores, actions = self._select_structured_actions(
                self.decoder.internal_action_policy(model_idea),
                schemas=normalized_tools,
                input_text="",
                neural_state=model_idea,
                assembly_ids=[str(assembly["id"]) for assembly in active],
                organic_state=action_state,
            )
            # This method has already performed the prompt-free recurrent,
            # liquid, spiking, rehearsal, and plasticity pass. Returning a
            # Ponder proposal would make the desktop invoke idle_cycle(0) again,
            # producing a redundant post-idle loop and a misleading visible
            # action card. Keep the neural choice in the trace while treating it
            # as completed internal cognition. Human-turn Ponder proposals still
            # trigger the dedicated pre-speech recurrent path in chat().
            internally_settled_actions = [
                action
                for action in actions
                if str(action.get("kind", "")) == "ponder"
            ]
            actions = [
                action
                for action in actions
                if str(action.get("kind", "")) != "ponder"
            ]
            visible_action_refractory_seconds = max(
                45.0,
                minimum_idle_seconds * 4.0,
            )
            visible_action_ready = (
                now - self.last_idle_visible_action_at
                >= visible_action_refractory_seconds
            )
            if actions and not visible_action_ready:
                # Keep the measured recurrent/STDP/rehearsal work below, but
                # suppress repeated unsolicited UI/tool/message activity.
                actions = []
            # ``talk`` during an idle cycle is generated directly from active
            # neural state. The two decoder input symbols are only the
            # language-boundary markers; there is no human text, behavioral
            # instruction, persona, or hidden prompt.
            talk_confidence = float(action_scores.get("talk", 0.0))
            if (
                not actions
                and max(action_scores, key=action_scores.get) == "talk"
                and talk_confidence >= ACTION_PROPOSAL_CONFIDENCE
                and compute_demand >= 0.35
                and visible_action_ready
            ):
                boundary = torch.tensor(
                    [[self.tokenizer.bos_id, self.tokenizer.brain_id]],
                    dtype=torch.long,
                    device=self.device,
                )
                maximum = max(
                    8,
                    min(
                        128,
                        self.config.max_seq_len - boundary.shape[1],
                    ),
                )
                generated, _entropies = self.decoder.generate(
                    boundary,
                    memory_bias=self.idea_adapter(routed),
                    max_new_tokens=maximum,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    noise=max(
                        0.005,
                        min(
                            0.35,
                            0.02
                            + 0.14 * float(organic["curiosity"])
                            + 0.08 * float(organic["uncertainty"]),
                        ),
                    ),
                    seed=seed,
                    printable_only=True,
                )
                message = self.tokenizer.decode(
                    generated[0, boundary.shape[1] :].detach().cpu().tolist()
                ).replace("\x00", "").strip()
                if message:
                    actions.append(
                        {
                            "kind": "talk",
                            "arguments": {
                                "message": message,
                                "assemblyIds": [
                                    str(assembly["id"])
                                    for assembly in active
                                ],
                                "organic": True,
                                "promptTokenCount": 0,
                            },
                            "confidence": talk_confidence,
                        }
                    )
                    self.messages.append(
                        {
                            "role": "brain",
                            "content": message,
                            "created_at": _iso_now(),
                            "organic": True,
                            "attention_epoch": self._attention_epoch(),
                        }
                    )
                    self._append_recent_dialogue("", message)
            if actions:
                self.last_idle_visible_action_at = now
        mode = (
            "ponder"
            if organic["tension"] >= max(0.5, organic["curiosity"])
            else (
                "imagine"
                if organic["curiosity"] >= 0.62
                else "rehearse"
            )
        )
        self.last_idle_cycle_at = now
        self.last_activity_decay = now
        self.counters["idle_cognition_cycles"] += 1
        self.counters["plasticity_events"] = int(
            self.router.synapses.plasticity_events.item()
        )
        trace = {
            "id": uuid.uuid4().hex,
            "createdAt": _iso_now(),
            "attentionEpoch": self._attention_epoch(),
            "mode": mode,
            "seed": seed,
            "promptTokenCount": 0,
            "hiddenBehavioralPrompt": False,
            "activeAssemblyIds": [
                str(assembly["id"]) for assembly in active
            ],
            "organicState": action_state,
            "liquidControls": {
                key: float(value.detach().mean().item())
                for key, value in controls.items()
            },
            "stdpUpdate": float(spike_metrics["stdp_update"]),
            "spikeRate": float(spike_metrics["spike_rate"]),
            "rehearsal": rehearsal,
            "memorySettling": memory_settling,
            "parameterChecksumBefore": parameter_before,
            "parameterChecksumAfter": parameter_after,
            "parameterDeltaNorm": parameter_delta,
            "actionPolicyFeatureChannel": "assembly-model",
            "actionPolicyScores": action_scores,
            "proposedActionKinds": [
                str(action.get("kind", "")) for action in actions
            ],
            "internallySettledActionKinds": [
                str(action.get("kind", ""))
                for action in internally_settled_actions
            ],
            "note": (
                "Measured prompt-free recurrent activity; not a hidden "
                "chain-of-thought transcript."
            ),
        }
        self.events.append("idle-cognition", trace)
        self.save()
        return {
            "brainId": self.brain_id,
            "ran": True,
            "trace": trace,
            "actions": actions,
            "metrics": self.metrics(),
            "runtimeCard": self.runtime_card(),
        }

    @staticmethod
    def _normalize_tool_schemas(
        schemas: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Keep structural tool identifiers and action names without a count ceiling."""

        if schemas is None:
            return []
        if not isinstance(schemas, (list, tuple)):
            raise ValueError("tool schemas must be a list")

        def structural_input_schema(raw: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(raw, Mapping):
                return None
            raw_properties = raw.get("properties", {})
            clean_properties: Dict[str, Dict[str, str]] = {}
            if isinstance(raw_properties, Mapping):
                for raw_name, raw_spec in raw_properties.items():
                    name = str(raw_name).strip()
                    if not name or len(name) > 128:
                        continue
                    spec = raw_spec if isinstance(raw_spec, Mapping) else {}
                    value_type = str(spec.get("type", "unknown")).strip().lower()
                    if value_type not in {
                        "string", "number", "integer", "boolean",
                        "array", "object", "null", "unknown",
                    }:
                        value_type = "unknown"
                    clean_properties[name] = {"type": value_type}
            raw_required = raw.get("required", [])
            required = []
            if isinstance(raw_required, (list, tuple)):
                required = [str(value) for value in raw_required if str(value) in clean_properties]
            return {
                "type": "object", "properties": clean_properties,
                "required": sorted(set(required)),
            }

        normalized: Dict[str, Dict[str, Any]] = {}
        for schema in schemas:
            if not isinstance(schema, Mapping):
                raise ValueError("each tool schema must be an object")
            tool_id = schema.get("id")
            actions = schema.get("actions")
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise ValueError("each tool schema needs a non-empty id")
            tool_id = tool_id.strip()
            if len(tool_id) > 128:
                raise ValueError("tool schema id exceeds 128 characters")
            if not isinstance(actions, (list, tuple)):
                raise ValueError("tool schema actions must be a list")
            clean_actions = []
            for action in actions:
                if not isinstance(action, str) or not action.strip():
                    raise ValueError("tool actions must be non-empty strings")
                action = action.strip()
                if len(action) > 64:
                    raise ValueError("tool action exceeds 64 characters")
                clean_actions.append(action)
            grant = (
                str(schema.get("grant", "ask")).strip().lower()[:32] or "ask"
            )
            # MCP descriptions are deliberately ignored. Preserve only the
            # structural JSON-schema field names, primitive types, and required
            # set so this channel cannot become behavioral prompt prose.
            clean_input_schema = structural_input_schema(schema.get("inputSchema"))
            clean_action_schemas: Dict[str, Dict[str, Any]] = {}
            raw_action_schemas = schema.get("actionInputSchemas")
            if not tool_id.startswith("mcp.") and isinstance(raw_action_schemas, Mapping):
                for action in clean_actions:
                    action_schema = structural_input_schema(raw_action_schemas.get(action))
                    if action_schema is not None:
                        clean_action_schemas[action] = action_schema
            # Electron already excludes Off tools. Enforce the same boundary
            # in the worker so direct JSON-RPC callers cannot make an Off
            # capability participate in neural routing.
            if grant == "off":
                continue
            existing = normalized.setdefault(
                tool_id, {"id": tool_id, "actions": [], "grant": grant}
            )
            existing["actions"] = sorted(
                set(existing["actions"]).union(clean_actions)
            )
            if clean_input_schema:
                existing["inputSchema"] = clean_input_schema
            if not tool_id.startswith("mcp.") and isinstance(raw_action_schemas, Mapping):
                # Keep an empty map meaningful: a missing action schema must
                # fail closed, not fall back to another action's requirements.
                existing.setdefault("actionInputSchemas", {}).update(clean_action_schemas)
        return [normalized[key] for key in sorted(normalized)]

    @staticmethod
    def _tool_action_input_schema(
        schema: Mapping[str, Any], action: str,
    ) -> Optional[Mapping[str, Any]]:
        """Resolve the selected built-in action or MCP's single input schema."""
        action_schemas = schema.get("actionInputSchemas")
        if not str(schema.get("id", "")).startswith("mcp.") and isinstance(action_schemas, Mapping):
            selected = action_schemas.get(action)
        else:
            selected = schema.get("inputSchema")
        return selected if isinstance(selected, Mapping) else None

    @staticmethod
    def _browser_operation_schema() -> Dict[str, Any]:
        """A structural, checkpointed argument target for one browser choice.

        The operation label is learned from a confirmed host outcome. Selectors,
        typed text, and URLs never enter this operation-kind target.
        """

        return {
            "id": "browser.operation",
            "actions": ["select"],
            "grant": "ask",
            "inputSchema": {
                "type": "object",
                "properties": {"operation": {"type": "string"}},
                "required": ["operation"],
            },
        }

    @staticmethod
    def _browser_operation_kind(arguments: Mapping[str, Any]) -> Optional[str]:
        steps = arguments.get("steps", [])
        if not isinstance(steps, list) or len(steps) > 1:
            return None
        if not steps:
            return "none"
        if not AdaptiveBrain._valid_explicit_browser_steps(steps):
            return None
        return str(steps[0]["kind"])

    def _tool_schema_vector(
        self, schemas: Sequence[Mapping[str, Any]]
    ) -> Optional[torch.Tensor]:
        """Encode tool capability structure without producing prompt tokens."""

        vectors = []
        for schema in schemas:
            tool_id = str(schema["id"])
            identity = self.memory.space.symbol("tool-id:" + tool_id)
            parts = [
                self.memory.space.bind(
                    identity,
                    self.memory.space.symbol(
                        "tool-grant:" + str(schema.get("grant", "ask"))
                    ),
                )
            ]
            for action in schema.get("actions", []):
                parts.append(
                    self.memory.space.bind(
                        identity,
                        self.memory.space.symbol(
                            "tool-action:" + str(action)
                        ),
                    )
                )
            input_schema = schema.get("inputSchema", {})
            properties = (
                input_schema.get("properties", {})
                if isinstance(input_schema, Mapping)
                else {}
            )
            if isinstance(properties, Mapping):
                for name, spec in properties.items():
                    value_type = (
                        str(spec.get("type", "unknown"))
                        if isinstance(spec, Mapping)
                        else "unknown"
                    )
                    parts.append(
                        self.memory.space.bind(
                            identity,
                            self.memory.space.symbol(
                                "tool-field:%s:%s" % (str(name), value_type)
                            ),
                        )
                    )
            vectors.append(self.memory.space.bundle(parts))
        if not vectors:
            return None
        bundled = self.memory.space.bundle(vectors)
        return self._idea_model_vector(bundled)

    @staticmethod
    def _explicit_https_urls(text: str) -> List[str]:
        """Return explicit public HTTPS URLs without guessing a destination."""

        urls: List[str] = []
        for match in re.finditer(r"https://[^\s<>\"'`]+", text, re.IGNORECASE):
            candidate = match.group(0).rstrip(".,;!?)]}")
            try:
                parsed = urlsplit(candidate)
            except ValueError:
                continue
            if (
                parsed.scheme.lower() != "https"
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or len(candidate) > 16_000
            ):
                continue
            if candidate not in urls:
                urls.append(candidate)
        return urls

    @staticmethod
    def _explicit_absolute_paths(text: str) -> List[str]:
        """Extract only paths the user actually supplied.

        Relative paths are intentionally not resolved against an implicit
        directory. Quoted paths may contain spaces; unquoted paths end at
        whitespace. Both Windows drive/UNC and POSIX absolute forms are
        recognized so a checkpoint behaves consistently across hosts.
        """

        candidates: List[str] = []
        quoted = re.compile(
            r"(?P<quote>[\"'`])(?P<value>.*?)(?P=quote)",
            re.DOTALL,
        )
        for match in quoted.finditer(text):
            candidates.append(match.group("value").strip())
        for pattern in (
            r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|\\\\)[^\s<>\"'`]+",
            r"(?<![A-Za-z0-9:/])/[^\s<>\"'`]+",
        ):
            candidates.extend(match.group(0) for match in re.finditer(pattern, text))

        paths: List[str] = []
        for raw in candidates:
            candidate = raw.replace("\x00", "").strip().rstrip(".,;!?)]}")
            absolute = (
                candidate.startswith("/")
                or bool(re.match(r"^[A-Za-z]:[\\/]", candidate))
                or bool(re.match(r"^\\\\[^\\/]+[\\/][^\\/]+", candidate))
            )
            if (
                not absolute
                or "\r" in candidate
                or "\n" in candidate
                or len(candidate) > 32_000
            ):
                continue
            if candidate not in paths:
                paths.append(candidate)
        return paths

    @staticmethod
    def _explicit_write_content(text: str, paths: Sequence[str]) -> Optional[str]:
        """Extract explicitly delimited write content, never infer it."""

        fenced = re.search(
            r"\b(?:content|text)\s*:\s*```(?:[A-Za-z0-9_+.-]+)?\s*\n"
            r"(?P<value>[\s\S]*?)```",
            text,
            re.IGNORECASE,
        )
        if fenced is not None:
            value = fenced.group("value").replace("\x00", "")
            return value[:8 * 1024 * 1024] if value else None

        quoted = re.compile(
            r"(?P<quote>[\"'`])(?P<value>.*?)(?P=quote)",
            re.DOTALL,
        )
        for match in quoted.finditer(text):
            value = match.group("value").replace("\x00", "")
            if value in paths or not value:
                continue
            # Content is accepted only when its delimiter participates in a
            # clear write/content construction. A quoted label elsewhere in
            # the request is not silently treated as file bytes.
            prefix = text[max(0, match.start() - 48) : match.start()].lower()
            suffix = text[match.end() : min(len(text), match.end() + 48)].lower()
            if (
                re.search(r"\b(?:write|save|replace|content|text)\s*$", prefix)
                or re.match(r"\s*(?:to|into|in)\b", suffix)
                or re.search(r"\b(?:content|text)\s*[:=]\s*$", prefix)
            ):
                return value[:8 * 1024 * 1024]
        return None

    @classmethod
    def _explicit_shell_arguments(
        cls, text: str
    ) -> Optional[Dict[str, str]]:
        """Read a user-delimited command and working directory verbatim.

        A native-shell proposal is intentionally impossible from vague prose.
        Both the command and an absolute, explicitly labelled cwd must occur
        in the user's current message; generated text is never consulted and
        the host dialect is selected only by the desktop execution adapter.
        """

        shell_name = (
            r"(?:system\s+shell|native\s+shell|shell|terminal|powershell|pwsh)"
            if os.name == "nt"
            else r"(?:system\s+shell|native\s+shell|shell|terminal|bash|zsh|sh)"
        )
        fence_name = (
            r"(?:powershell|pwsh|shell)?"
            if os.name == "nt"
            else r"(?:bash|zsh|sh|shell)?"
        )
        if not re.search(r"\b%s\b" % shell_name, text, re.IGNORECASE):
            return None
        fenced = re.search(
            r"\b%s(?:\s+command)?\s*(?:is\s*)?[:=]?\s*" % shell_name
            + r"```" + fence_name + r"\s*\n"
            r"(?P<command>[\s\S]*?)```",
            text,
            re.IGNORECASE,
        )
        quoted = re.search(
            r"\b(?:run|execute)\s+(?:this\s+)?" + shell_name
            + r"(?:\s+command)?\s*(?:is\s*)?[:=]?\s*"
            r"(?P<quote>[\"'`])(?P<command>[\s\S]*?)(?P=quote)",
            text,
            re.IGNORECASE,
        )
        command_match = fenced or quoted
        if command_match is None:
            return None
        command = command_match.group("command").replace("\x00", "").strip()
        if not command or len(command) > 100_000:
            return None

        cwd_quoted = re.search(
            r"\b(?:cwd|working\s+directory)\s*(?:is\s*)?(?:[:=]|to)?\s*"
            r"(?P<quote>[\"'`])(?P<cwd>[\s\S]*?)(?P=quote)",
            text,
            re.IGNORECASE,
        )
        cwd_plain = re.search(
            r"\b(?:cwd|working\s+directory)\s*(?:is\s*)?(?:[:=]|to)?\s*"
            r"(?P<cwd>(?:[A-Za-z]:[\\/]|\\\\|/)[^\s<>\"'`]+)",
            text,
            re.IGNORECASE,
        )
        cwd_match = cwd_quoted or cwd_plain
        if cwd_match is None:
            return None
        cwd_value = cwd_match.group("cwd").replace("\x00", "").strip()
        cwd_paths = cls._explicit_absolute_paths(cwd_value)
        if len(cwd_paths) != 1:
            return None
        return {"command": command, "cwd": cwd_paths[0]}

    def _neural_browser_operation(
        self, neural_state: torch.Tensor,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Select one operation with checkpointed weights, not prose rules."""

        head = getattr(self.decoder, "action_argument_head", None)
        if head is None or not hasattr(head, "grounded_for"):
            return None, {"reason": "missing-argument-head"}
        if (
            head.grounded_for("browser.operation", "select") < 1
            or head.grounded_for("browser.operation", "select:none") < 1
        ):
            return None, {"reason": "untrained-browser-no-operation-boundary"}
        schema = self._browser_operation_schema()
        features = head.schema_features("browser.operation", "select", schema)
        if features is None:
            return None, {"reason": "missing-browser-operation-schema"}
        decoded = head.decode(
            neural_state,
            features[None].to(neural_state.device),
            tool_id="browser.operation", action="select",
            max_output_bytes=64,
        )
        if (
            decoded.get("reason") != "learned-typed-arguments"
            or not isinstance(decoded.get("meanTokenProbability"), (int, float))
            or isinstance(decoded["meanTokenProbability"], bool)
            or not math.isfinite(float(decoded["meanTokenProbability"]))
            or float(decoded["meanTokenProbability"]) < 0.85
        ):
            return None, {
                "reason": "uncertain-browser-operation",
                "decoder": {key: value for key, value in decoded.items() if key != "arguments"},
            }
        arguments = decoded.get("arguments")
        operation = arguments.get("operation") if isinstance(arguments, Mapping) else None
        if (
            not isinstance(operation, str)
            or operation not in {
                "none", "navigate", "click", "type", "press",
                "wait", "extract", "screenshot",
            }
            or set(arguments) != {"operation"}
            or head.grounded_for("browser.operation", "select:" + operation) < 1
        ):
            return None, {
                "reason": "ungrounded-or-invalid-browser-operation",
                "decoder": {key: value for key, value in decoded.items() if key != "arguments"},
            }
        return operation, {
            "reason": "grounded-neural-browser-operation",
            "decoder": {key: value for key, value in decoded.items() if key != "arguments"},
        }

    @staticmethod
    def _literal_browser_operand(
        text: str, operation: str,
    ) -> Optional[Dict[str, Any]]:
        """Copy only explicit operands after the neural operation is fixed."""

        if operation == "screenshot":
            return {"kind": operation}
        if operation not in {"click", "press", "wait", "extract"}:
            # Typing, navigation, and complex steps require explicit JSON.
            return None
        quoted = [
            match.group("value")
            for match in re.finditer(
                r"(?<![A-Za-z0-9])(?P<quote>[\"'`])(?P<value>.*?)(?P=quote)",
                text, re.DOTALL,
            )
            if match.group("value").strip()
            and "\r" not in match.group("value")
            and "\n" not in match.group("value")
            and not match.group("value").startswith("https://")
        ]
        if len(quoted) != 1:
            return None
        value = quoted[0]
        key = "key" if operation == "press" else "selector"
        limit = 64 if key == "key" else 2_000
        if "\x00" in value or len(value) > limit:
            return None
        return {"kind": operation, key: value}

    def _materialize_generic_tool_action(
        self,
        *,
        schemas: Sequence[Mapping[str, Any]],
        input_text: str,
        assembly_ids: Sequence[str],
        organic_state: Mapping[str, float],
        neural_state: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the trained neural route, then copy only explicit values.

        The utterance is never fed to a text-route classifier here. It is
        consulted only after a neural winner to materialize that one schema.
        """
        # An idle cycle has no user utterance to supply literal arguments.
        # Never manufacture a tool request from diagnostic assembly labels.
        if not input_text.strip():
            self._last_tool_route_evidence = {
                "kind": "trained-internal-schema-route",
                "selected": None,
                "reason": "no-explicit-action-arguments",
            }
            return None
        if not isinstance(neural_state, torch.Tensor) or neural_state.ndim != 2:
            self._last_tool_route_evidence = {
                "kind": "trained-internal-schema-route",
                "selected": None,
                "reason": "missing-active-neural-state",
            }
            return None
        head = getattr(self.decoder, "tool_route_head", None)
        evidence = (
            head.select_internal(neural_state, schemas) if head is not None else
            {"kind": "trained-internal-schema-route", "trainingSteps": 0,
             "selected": None, "reason": "missing-internal-route-head"}
        )
        self._last_tool_route_evidence = evidence
        selected = evidence["selected"]
        if selected is None:
            return None
        schema = next(
            (
                value for value in schemas
                if value.get("id") == selected["toolId"]
                and selected["action"] in value.get("actions", ())
                and value.get("grant") != "off"
            ),
            None,
        )
        if schema is None:
            evidence["materialization"] = "route-not-enabled"
            return None
        arguments = self._literal_tool_route_arguments(
            selected["toolId"], selected["action"], input_text, schema
        )
        if arguments is None:
            evidence["materialization"] = "missing-explicit-arguments"
            return None
        neural_browser_step = False
        if selected["toolId"] == "browser.automation" and "steps" not in arguments:
            operation, operation_evidence = self._neural_browser_operation(neural_state)
            evidence["browserOperationEvidence"] = operation_evidence
            if operation not in {None, "none"}:
                step = self._literal_browser_operand(input_text, operation)
                if step is not None:
                    arguments["steps"] = [step]
                    neural_browser_step = True
        candidate = {**selected, "arguments": arguments, "routeEvidence": evidence}
        if not self._materialized_tool_action_matches_schema(schemas, candidate):
            evidence["materialization"] = "schema-rejected"
            return None
        evidence["materialization"] = (
            "grounded-neural-browser-operation-literal-operands-schema-validated"
            if neural_browser_step else "literal-arguments-schema-validated"
        )
        return candidate

    def _materialize_internal_action(
        self,
        *,
        schemas: Sequence[Mapping[str, Any]],
        neural_state: Optional[torch.Tensor],
        tool_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Decode one typed idle action from weights and active neural state.

        No diagnostic labels, source passages, fabricated utterances, or
        textual schema descriptions participate. The trained route head is
        needed for generic tools; a selected agent/evolution kind fixes only
        its structural protocol, never its objective text.
        """

        if neural_state is None or neural_state.ndim != 2:
            self._last_tool_route_evidence = {
                "kind": "trained-internal-schema-route",
                "selected": None,
                "reason": "missing-active-neural-state",
            }
            return None
        route_head = getattr(self.decoder, "tool_route_head", None)
        argument_head = getattr(self.decoder, "action_argument_head", None)
        if argument_head is None:
            return None
        if tool_id is None or action is None:
            route_evidence = (
                route_head.select_internal(neural_state, schemas)
                if route_head is not None
                else {"selected": None, "reason": "missing-internal-route-head"}
            )
            selected = route_evidence.get("selected")
            if not isinstance(selected, Mapping):
                self._last_tool_route_evidence = route_evidence
                return None
            tool_id = str(selected.get("toolId", ""))
            action = str(selected.get("action", ""))
        else:
            route_evidence = {
                "kind": "learned-action-kind-protocol",
                "selected": {"toolId": tool_id, "action": action},
                "hiddenPrompt": False,
            }
        schema = next(
            (
                value for value in schemas
                if value.get("id") == tool_id
                and action in value.get("actions", ())
                and str(value.get("grant", "ask")).lower() != "off"
            ),
            None,
        )
        if schema is None:
            self._last_tool_route_evidence = {
                **route_evidence, "materialization": "route-not-enabled"
            }
            return None
        browser_operation: Optional[str] = None
        if tool_id == "browser.automation":
            browser_operation, operation_evidence = self._neural_browser_operation(
                neural_state
            )
            if (
                browser_operation is None
                or argument_head.grounded_for(
                    "browser.operation", "full:" + browser_operation
                ) < 1
            ):
                self._last_tool_route_evidence = {
                    **route_evidence,
                    "materialization": "ungrounded-browser-operation",
                    "browserOperationEvidence": operation_evidence,
                }
                return None
            route_evidence = {
                **route_evidence, "browserOperationEvidence": operation_evidence
            }
        features = argument_head.schema_features(tool_id, action, schema)
        if features is None:
            self._last_tool_route_evidence = {
                **route_evidence, "materialization": "missing-action-schema"
            }
            return None
        argument_evidence = argument_head.decode(
            neural_state.to(self.device), features[None].to(self.device),
            tool_id=tool_id, action=action,
            max_output_bytes=min(4096, max(128, int(self.config.max_seq_len))),
        )
        arguments = argument_evidence.get("arguments")
        if not isinstance(arguments, Mapping):
            self._last_tool_route_evidence = {
                **route_evidence,
                "materialization": str(argument_evidence.get("reason", "no-arguments")),
                "argumentEvidence": argument_evidence,
            }
            return None
        candidate = {
            "toolId": tool_id,
            "action": action,
            "arguments": dict(arguments),
        }
        if tool_id == "browser.automation" and (
            self._browser_operation_kind(arguments) != browser_operation
            or not isinstance(arguments.get("url"), str)
            or self._explicit_https_urls(arguments["url"]) != [arguments["url"]]
        ):
            self._last_tool_route_evidence = {
                **route_evidence,
                "materialization": "browser-operation-argument-mismatch",
            }
            return None
        required_content = (
            "query" if tool_id == "web.search"
            else "objective" if tool_id in {"agent.fork", "source.self-modify"}
            else None
        )
        if required_content is not None and not str(
            arguments.get(required_content, "")
        ).strip():
            self._last_tool_route_evidence = {
                **route_evidence, "materialization": "empty-learned-objective",
                "argumentEvidence": {
                    key: value for key, value in argument_evidence.items()
                    if key != "arguments"
                },
            }
            return None
        if (
            tool_id == "source.self-modify"
            and "candidateKind" in arguments
            and (
                not isinstance(arguments["candidateKind"], str)
                or arguments["candidateKind"] not in {
                    "neural", "data", "substrate", "architecture",
                }
            )
        ):
            self._last_tool_route_evidence = {
                **route_evidence,
                "materialization": "unsupported-learned-candidate-kind",
            }
            return None
        if not self._materialized_tool_action_matches_schema(schemas, candidate):
            self._last_tool_route_evidence = {
                **route_evidence, "materialization": "schema-rejected",
                "argumentEvidence": argument_evidence,
            }
            return None
        evidence = {
            **route_evidence,
            "materialization": "trained-arguments-schema-validated",
            "argumentEvidence": {
                key: value for key, value in argument_evidence.items()
                if key != "arguments"
            },
        }
        self._last_tool_route_evidence = evidence
        return {**candidate, "routeEvidence": evidence}

    def _literal_tool_route_arguments(
        self, tool_id: str, action: str, text: str, schema: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract values *after* neural selection; never rank another route.

        Missing values fail closed. JSON fields are an explicit general
        interface for schemas without a built-in literal-value extractor.
        These parsers cannot promote a different capability when they fail.
        """
        input_schema = self._tool_action_input_schema(schema, action)
        if input_schema is None:
            return None
        properties = input_schema.get("properties", {})
        arguments: Dict[str, Any] = {}
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match is not None:
            try:
                value = json.loads(json_match.group(0))
                if isinstance(value, dict):
                    arguments = {key: item for key, item in value.items() if key in properties}
            except (ValueError, TypeError):
                pass
        paths = self._explicit_absolute_paths(text)
        urls = self._explicit_https_urls(text)
        lowered = text.casefold()
        if tool_id in {"system.files", "windows.files"}:
            if len(paths) == 1:
                arguments.setdefault("path", paths[0])
            if action == "write":
                content = self._explicit_write_content(text, paths)
                if content is not None:
                    arguments.setdefault("content", content)
            if "path" not in arguments or (action == "write" and "content" not in arguments):
                return None
        elif tool_id in {"system.shell", "windows.powershell"}:
            arguments = {**(self._explicit_shell_arguments(text) or {}), **arguments}
            if not {"command", "cwd"}.issubset(arguments):
                return None
        elif tool_id == "code.execute":
            if len(paths) == 1:
                arguments.setdefault("entryPath", paths[0])
                # File extension denotes the literal language; it does not
                # decide whether execute, read, or write is the selected action.
                language = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript"}.get(Path(paths[0]).suffix.lower())
                if language is not None:
                    arguments.setdefault("language", language)
            if not {"entryPath", "language"}.issubset(arguments):
                return None
        elif tool_id in {"web.fetch", "browser.automation"}:
            if len(urls) == 1:
                arguments.setdefault("url", urls[0])
            if "url" not in arguments:
                return None
        elif tool_id in {"web.search", "brain.history"} and action == "search":
            match = re.search(r"\b(?:search|research|look\s+up|find)\b(?:\s+(?:the\s+)?(?:web|internet|online))?(?:\s+(?:for|about|on))?\s+(?P<query>.+)", text, re.I | re.S)
            if match is not None:
                arguments.setdefault("query", match.group("query").strip())
            if not arguments.get("query"):
                return None
        elif tool_id == "device.input":
            if action == "move-pointer":
                match = re.search(r"(?:\bx\s*=?\s*|\()(\-?\d{1,6})\s*(?:,\s*(?:y\s*=?\s*)?|\s+y\s*=?\s*)(\-?\d{1,6})", text, re.I)
                if match:
                    arguments.update({"x": int(match[1]), "y": int(match[2])})
                if not {"x", "y"}.issubset(arguments):
                    return None
            elif action == "click":
                match = re.search(r"\b(left|right|middle)\b", text, re.I)
                if match:
                    arguments.setdefault("button", match[1].lower())
                if "button" not in arguments:
                    return None
            elif action == "scroll":
                match = re.search(r"\b(up|down|left|right)\s+(\d{1,5})\b", text, re.I)
                if match:
                    direction, amount = match[1].lower(), int(match[2])
                    arguments.setdefault("deltaY" if direction in {"up", "down"} else "deltaX",
                                         amount if direction in {"up", "right"} else -amount)
                if not ("deltaX" in arguments or "deltaY" in arguments):
                    return None
            elif action == "key-press":
                match = re.search(r"\bpress\s+[`'\"]?([A-Za-z0-9_.-]+(?:\s*\+\s*[A-Za-z0-9_.-]+){0,4})", text, re.I)
                if match:
                    chord = [item.strip().lower() for item in match[1].split("+")]
                    arguments.setdefault("key", chord[-1])
                    if len(chord) > 1:
                        arguments.setdefault("modifiers", chord[:-1])
                if "key" not in arguments:
                    return None
            elif action == "text":
                match = re.search(r"\b(?:type|enter|input)\s+(?P<q>[\"'`])(?P<text>.*?)(?P=q)", text, re.I | re.S)
                if match:
                    arguments.setdefault("text", match.group("text"))
                if not arguments.get("text"):
                    return None
        elif tool_id == "device.observe":
            resolution = re.search(r"(\d{2,5})\s*(?:x|×|by)\s*(\d{2,5})", lowered)
            if resolution:
                arguments.update({"width": int(resolution[1]), "height": int(resolution[2])})
            if action == "snapshot":
                match = re.search(r"\b(native|current|custom)\b", lowered)
                if match:
                    arguments.setdefault("resolutionMode", match[1])
                count = re.search(r"\b(\d{1,3})\s+(?:frames?|photos?|snapshots?)\b", lowered)
                if count:
                    arguments.setdefault("burstCount", int(count[1]))
                interval = re.search(r"\bevery\s+(\d+)\s*(ms|milliseconds?|s|seconds?)\b", lowered)
                if interval:
                    arguments.setdefault("intervalMs", int(interval[1]) * (1 if interval[2].startswith("m") else 1000))
            else:
                mode = re.search(r"\b(auto|motion|balanced|detail)\b", lowered)
                if mode:
                    arguments.setdefault("mode", mode[1])
                fps = re.search(r"\b(\d{1,3})\s*fps\b", lowered)
                if fps:
                    arguments.setdefault("fps", int(fps[1]))
                duration = re.search(r"\bfor\s+(\d+)\s*(ms|milliseconds?|s|seconds?)\b", lowered)
                if duration:
                    arguments.setdefault("durationMs", int(duration[1]) * (1 if duration[2].startswith("m") else 1000))
            if not arguments:
                return None
        else:
            # Generic external schemas accept only explicit JSON and literal
            # path/URL values, never guessed free-form strings or defaults.
            for name in properties:
                if name.casefold() in {"url", "uri"} and len(urls) == 1:
                    arguments.setdefault(name, urls[0])
                elif name.casefold() in {"path", "file", "filepath", "directory"} and len(paths) == 1:
                    arguments.setdefault(name, paths[0])
        return arguments

    @staticmethod
    def _valid_explicit_browser_steps(value: Any) -> bool:
        """Validate structured steps; prose must not synthesize side effects.

        The neural route may select the browser, but a regex match such as
        ``do not click`` is not evidence that the brain chose a click. A future
        grounded argument decoder can supply the same typed structure.
        """

        if not isinstance(value, list) or len(value) > 200:
            return False

        def bounded_text(
            step: Mapping[str, Any], key: str, limit: int,
            *, allow_newlines: bool = False,
        ) -> bool:
            item = step.get(key)
            return (
                isinstance(item, str)
                and 0 < len(item) <= limit
                and "\x00" not in item
                and (allow_newlines or ("\r" not in item and "\n" not in item))
            )

        allowed = {
            "navigate": {"kind", "url"},
            "click": {"kind", "selector", "timeoutMs"},
            "type": {"kind", "selector", "value", "clear", "sensitive"},
            "press": {"kind", "key", "timeoutMs"},
            "wait": {"kind", "selector", "milliseconds", "timeoutMs"},
            "extract": {"kind", "selector"},
            "screenshot": {"kind"},
        }
        for step in value:
            if not isinstance(step, Mapping):
                return False
            kind = step.get("kind")
            if (
                not isinstance(kind, str)
                or kind not in allowed
                or set(step) - allowed[kind]
            ):
                return False
            if "timeoutMs" in step and (
                isinstance(step["timeoutMs"], bool)
                or not isinstance(step["timeoutMs"], (int, float))
                or not math.isfinite(float(step["timeoutMs"]))
                or not 0 < float(step["timeoutMs"]) <= 30_000
            ):
                return False
            if kind == "navigate" and not (
                bounded_text(step, "url", 16_000)
                and str(step["url"]).startswith("https://")
            ):
                return False
            if kind == "click" and not bounded_text(step, "selector", 2_000):
                return False
            if kind == "type" and not (
                bounded_text(step, "selector", 2_000)
                and bounded_text(step, "value", 100_000, allow_newlines=True)
                and ("clear" not in step or isinstance(step["clear"], bool))
                and ("sensitive" not in step or isinstance(step["sensitive"], bool))
            ):
                return False
            if kind == "press" and not bounded_text(step, "key", 64):
                return False
            if kind == "wait":
                selector = "selector" in step
                milliseconds = "milliseconds" in step
                if selector == milliseconds:
                    return False
                if selector and not bounded_text(step, "selector", 2_000):
                    return False
                if milliseconds and (
                    isinstance(step["milliseconds"], bool)
                    or not isinstance(step["milliseconds"], int)
                    or not 0 < step["milliseconds"] <= 30_000
                ):
                    return False
            if kind == "extract" and "selector" in step and not bounded_text(
                step, "selector", 2_000
            ):
                return False
        return True


    @staticmethod
    def _materialized_tool_action_matches_schema(
        schemas: Sequence[Mapping[str, Any]],
        materialized: Mapping[str, Any],
    ) -> bool:
        """Fail closed unless a materialized action exactly fits its schema."""

        tool_id = materialized.get("toolId")
        action = materialized.get("action")
        arguments = materialized.get("arguments")
        if (
            not isinstance(tool_id, str)
            or not isinstance(action, str)
            or not isinstance(arguments, Mapping)
        ):
            return False
        schema = next(
            (
                value
                for value in schemas
                if value.get("id") == tool_id
                and str(value.get("grant", "ask")).strip().lower()
                != "off"
            ),
            None,
        )
        if schema is None or action not in schema.get("actions", ()):
            return False
        input_schema = AdaptiveBrain._tool_action_input_schema(schema, action)
        if not isinstance(input_schema, Mapping):
            return False
        properties = input_schema.get("properties")
        required = input_schema.get("required", ())
        if not isinstance(properties, Mapping) or not isinstance(
            required, (list, tuple)
        ):
            return False
        if not set(str(value) for value in required).issubset(arguments):
            return False
        if any(str(name) not in properties for name in arguments):
            return False
        if tool_id == "browser.automation" and "steps" in arguments:
            if not AdaptiveBrain._valid_explicit_browser_steps(arguments["steps"]):
                return False

        def valid_type(value: Any, expected: str) -> bool:
            if expected == "string":
                return isinstance(value, str)
            if expected == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if expected == "number":
                return (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                )
            if expected == "boolean":
                return isinstance(value, bool)
            if expected == "array":
                return isinstance(value, (list, tuple))
            if expected == "object":
                return isinstance(value, Mapping)
            if expected == "null":
                return value is None
            return False

        return all(
            isinstance(properties.get(name), Mapping)
            and valid_type(
                value,
                str(properties[name].get("type", "unknown")).lower(),
            )
            for name, value in arguments.items()
        )


    def _select_structured_actions(
        self,
        action_logits: torch.Tensor,
        *,
        schemas: Sequence[Mapping[str, Any]],
        input_text: str,
        assembly_ids: Sequence[str],
        organic_state: Mapping[str, float],
        supporting_action_logits: Sequence[torch.Tensor] = (),
        neural_state: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Convert the learned action head into a typed, permission-gated proposal.

        The neural worker proposes; the trusted desktop process still validates
        the schema, checks the build's tool grant, exposes an action card, and
        executes or asks. No behavioral instruction text is inserted into the
        model context.
        """

        probabilities = F.softmax(action_logits.detach().float(), dim=-1)[0]
        scores = {
            kind: float(probabilities[index].item())
            for index, kind in enumerate(ACTION_KINDS)
        }
        selected_index = int(probabilities.argmax().item())
        kind = ACTION_KINDS[selected_index]
        confidence = scores[kind]
        actions: List[Dict[str, Any]] = []
        self._last_tool_route_evidence = None
        available = {
            str(schema["id"]): set(
                str(value) for value in schema.get("actions", [])
            )
            for schema in schemas
            if str(schema.get("grant", "ask")).strip().lower() != "off"
        }
        tension = float(organic_state.get("computeDemand", 0.0))
        selected_tool: Optional[Dict[str, Any]] = None
        independent_tool_confidence = 0.0
        tool_support_evidence: Optional[Dict[str, Any]] = None
        materialized_candidate: Optional[Dict[str, Any]] = None

        def ordinary_tool_candidate(candidate: Mapping[str, Any]) -> bool:
            return str(candidate.get("toolId", "")) not in {
                "modality.imagine",
                "agent.fork",
                "source.self-modify",
            }

        if kind == "talk" and supporting_action_logits:
            support_candidates = list(supporting_action_logits)
            tool_index = ACTION_KINDS.index("tool")
            for supporting_logits in support_candidates:
                if (
                    not isinstance(supporting_logits, torch.Tensor)
                    or supporting_logits.ndim not in {1, 2}
                    or int(supporting_logits.shape[-1]) != len(ACTION_KINDS)
                ):
                    continue
                support_probabilities = F.softmax(
                    supporting_logits.detach().float().reshape(
                        -1, len(ACTION_KINDS)
                    )[0],
                    dim=-1,
                )
                support_confidence = float(
                    support_probabilities[tool_index].item()
                )
                if (
                    int(support_probabilities.argmax().item()) == tool_index
                    and support_confidence
                    > ACTION_INDEPENDENT_TOOL_SUPPORT_CONFIDENCE
                ):
                    independent_tool_confidence = max(
                        independent_tool_confidence, support_confidence
                    )
            if independent_tool_confidence > 0.0:
                materialized_candidate = self._materialize_generic_tool_action(
                    schemas=schemas,
                    input_text=input_text,
                    assembly_ids=assembly_ids,
                    organic_state=organic_state,
                    neural_state=neural_state,
                )
                if (
                    materialized_candidate is not None
                    and ordinary_tool_candidate(materialized_candidate)
                    and self._materialized_tool_action_matches_schema(
                        schemas, materialized_candidate
                    )
                ):
                    selected_tool = materialized_candidate
                    kind = "tool"
                    confidence = independent_tool_confidence
                    tool_support_evidence = {
                        "kind": "independent-head-majority-and-schema",
                        "toolProbability": independent_tool_confidence,
                        "minimumToolProbability": (
                            ACTION_INDEPENDENT_TOOL_SUPPORT_CONFIDENCE
                        ),
                        "sourceTextRead": False,
                    }
        # A blank/random brain has a nearly uniform head. It must learn a
        # decisive distribution before proposing external activity. Safe
        # internal cognition may emerge sooner when unresolved neural tension
        # is high.
        if kind == "talk":
            return scores, actions
        if kind == "learn":
            # Transactional chat may still be decoding before its fast
            # experience commit; idle replay has a separate durable receipt.
            # Do not surface a card claiming a mutation that has not happened.
            return scores, actions
        if kind == "ponder":
            if (
                self.config.idle_cognition
                and confidence >= 0.30
                and tension >= 0.50
            ):
                actions.append(
                    {
                        "kind": kind,
                        "arguments": {
                            "assemblyIds": list(assembly_ids),
                            "organic": True,
                        },
                        "confidence": confidence,
                    }
                )
            return scores, actions
        if kind == "stop":
            if confidence >= 0.90:
                actions.append(
                    {
                        "kind": "stop",
                        "arguments": {"reason": "neural-action-head"},
                        "confidence": confidence,
                    }
                )
            return scores, actions
        if (
            confidence < ACTION_PROPOSAL_CONFIDENCE
            and selected_tool is None
        ):
            return scores, actions

        base_arguments: Dict[str, Any] = {
            "assemblyIds": list(assembly_ids),
            "organic": True,
        }
        if kind == "tool":
            if selected_tool is None:
                selected_tool = (
                    self._materialize_generic_tool_action(
                        schemas=schemas,
                        input_text=input_text,
                        assembly_ids=assembly_ids,
                        organic_state=organic_state,
                        neural_state=neural_state,
                    )
                    if input_text.strip()
                    else self._materialize_internal_action(
                        schemas=schemas,
                        neural_state=neural_state,
                    )
                )
            if selected_tool is not None and ordinary_tool_candidate(selected_tool):
                # MCP inputSchema describes the complete remote argument object.
                # Neural routing metadata is local action context, not a tool
                # argument; built-in tools still receive that context as before.
                tool_arguments = (
                    dict(selected_tool["arguments"])
                    if selected_tool["toolId"].startswith("mcp.")
                    else {**base_arguments, **selected_tool["arguments"]}
                )
                actions.append(
                    {
                        "kind": "tool",
                        "toolId": selected_tool["toolId"],
                        "action": selected_tool["action"],
                        "arguments": tool_arguments,
                        "confidence": confidence,
                        **({"routeEvidence": selected_tool["routeEvidence"]}
                           if "routeEvidence" in selected_tool else {}),
                        **(
                            {"supportEvidence": tool_support_evidence}
                            if tool_support_evidence is not None
                            else {}
                        ),
                    }
                )
        elif (
            kind == "imagine"
            and "generate" in available.get("modality.imagine", set())
        ):
            # The same checkpointed OmniCortex selects the medium through a
            # learned exact-ternary projection. Active assemblies and liquid
            # state are neural inputs; no hidden prompt or slash command exists.
            enabled_modalities = [
                name
                for name, enabled in (
                    ("image", self.config.image_enabled),
                    ("audio", self.config.audio_enabled),
                    ("video", self.config.video_enabled),
                )
                if enabled
            ]
            # A randomly initialized media pack is a real decoder but has not
            # learned to express an idea. Manual generation can still expose
            # that research baseline; an organic action must not present it as
            # meaningful learned imagination or silently route to a disabled
            # pack. Media training or an installed compatible pack enables it.
            route_modalities = [
                name for name in enabled_modalities
                if int(self.modality_training.get(name, 0)) > 0
                or any(
                    name in item.get("modalities", ())
                    for item in self.installed_modality_packs
                )
            ]
            if not route_modalities:
                return scores, actions
            route_idea = self._modality_idea("", assembly_ids)
            route_idea = torch.tanh(
                0.78 * route_idea + 0.22 * self.liquid_state.detach()
            )
            modality, modality_scores = self.modalities.select_imagination(
                route_idea, enabled=route_modalities
            )
            actions.append(
                {
                    "kind": "imagine",
                    "toolId": "modality.imagine",
                    "action": "generate",
                    "arguments": {
                        **base_arguments,
                        "conceptIds": list(assembly_ids),
                        "modality": modality,
                        "localPackEnabled": True,
                        "trainedPackAvailable": True,
                        "neuralRoute": {
                            "kind": "exact-ternary-same-brain-head",
                            "scores": modality_scores,
                            "hiddenPrompt": False,
                        },
                    },
                    "confidence": confidence,
                }
            )
        elif (
            kind == "agent"
            and "start" in available.get("agent.fork", set())
        ):
            organic_agent = (
                self._materialize_internal_action(
                    schemas=schemas, neural_state=neural_state,
                    tool_id="agent.fork", action="start",
                ) if not input_text.strip() else None
            )
            if input_text.strip() or organic_agent is not None:
                actions.append(
                    {
                        "kind": "agent",
                        "toolId": "agent.fork",
                        "action": "start",
                        "arguments": {
                            **base_arguments,
                            **(
                                organic_agent["arguments"]
                                if organic_agent is not None
                                else {"objective": input_text}
                            ),
                        },
                        "confidence": confidence,
                        **(
                            {"routeEvidence": organic_agent["routeEvidence"]}
                            if organic_agent is not None else {}
                        ),
                    }
                )
        elif (
            kind == "evolve"
            and self.config.recursive_improvement
            and "propose" in available.get("source.self-modify", set())
        ):
            # The neural action head chooses to evolve, but strategy and
            # objective operands still need explicit typed user data or a
            # grounded neural argument decode. Source edits are never
            # synthesized here; an incomplete strategy emits no proposal.
            explicit_evolution: Optional[Dict[str, Any]] = None
            explicit_kind_supplied = False
            if input_text.strip():
                evolution_schema = next(
                    (
                        schema for schema in schemas
                        if schema.get("id") == "source.self-modify"
                        and "propose" in schema.get("actions", ())
                    ),
                    None,
                )
                if evolution_schema is not None:
                    parsed = self._literal_tool_route_arguments(
                        "source.self-modify", "propose", input_text,
                        evolution_schema,
                    )
                    explicit_kind_supplied = (
                        isinstance(parsed, Mapping)
                        and "candidateKind" in parsed
                    )
                    if (
                        isinstance(parsed, Mapping)
                        and isinstance(parsed.get("objective"), str)
                        and parsed["objective"].strip()
                        and isinstance(parsed.get("candidateKind"), str)
                        and parsed.get("candidateKind") in {
                            "neural", "data", "substrate", "architecture",
                        }
                    ):
                        explicit_evolution = {
                            "objective": parsed["objective"].strip(),
                            "candidateKind": parsed["candidateKind"],
                        }
                        add_experts = parsed.get("addExperts")
                        if (
                            isinstance(add_experts, int)
                            and not isinstance(add_experts, bool)
                            and add_experts > 0
                        ):
                            explicit_evolution["addExperts"] = add_experts
            selected_evolution: Optional[Dict[str, Any]] = None
            route_evidence: Optional[Dict[str, Any]] = None
            if explicit_evolution is not None:
                selected_evolution = explicit_evolution
            elif not explicit_kind_supplied:
                learned_evolution = self._materialize_internal_action(
                    schemas=schemas, neural_state=neural_state,
                    tool_id="source.self-modify", action="propose",
                )
                if learned_evolution is not None:
                    selected_evolution = dict(learned_evolution["arguments"])
                    route_evidence = learned_evolution["routeEvidence"]
                    if input_text.strip():
                        # The visible human request is the chat objective;
                        # the grounded neural decoder chooses the strategy.
                        selected_evolution["objective"] = input_text
            else:
                self._last_tool_route_evidence = {
                    "kind": "evolution-argument-gate",
                    "reason": "invalid-explicit-candidate-kind",
                }
            candidate_kind = (
                selected_evolution.get("candidateKind")
                if selected_evolution is not None else None
            )
            if selected_evolution is not None and (
                not isinstance(candidate_kind, str)
                or candidate_kind not in {
                    "neural", "data", "substrate", "architecture",
                }
            ):
                selected_evolution = None
                self._last_tool_route_evidence = {
                    "kind": "evolution-argument-gate",
                    "reason": "candidate-kind-unavailable",
                }
            if selected_evolution is not None and candidate_kind == "architecture" and not (
                isinstance(selected_evolution.get("addExperts"), int)
                and not isinstance(selected_evolution.get("addExperts"), bool)
                and selected_evolution["addExperts"] > 0
            ):
                selected_evolution = None
                self._last_tool_route_evidence = {
                    "kind": "evolution-argument-gate",
                    "reason": "architecture-mutation-arguments-missing",
                }
            if selected_evolution is not None and candidate_kind == "data" and not (
                selected_evolution.get("latentReplay") is True
                or selected_evolution.get("texts")
                or selected_evolution.get("sourceIds")
            ):
                selected_evolution = None
                self._last_tool_route_evidence = {
                    "kind": "evolution-argument-gate",
                    "reason": "data-evidence-missing",
                }
            if selected_evolution is None:
                self._last_tool_route_evidence = (
                    self._last_tool_route_evidence
                    if isinstance(self._last_tool_route_evidence, Mapping)
                    else {"kind": "evolution-argument-gate",
                          "reason": "candidate-kind-unavailable"}
                )
            else:
                actions.append(
                    {
                        "kind": "evolve",
                        "toolId": "source.self-modify",
                        "action": "propose",
                        "arguments": {
                            **base_arguments,
                            **selected_evolution,
                            "recursive": True,
                        },
                        "confidence": confidence,
                        **(
                            {"routeEvidence": route_evidence}
                            if route_evidence is not None else {}
                        ),
                    }
                )
        return scores, actions

    @torch.no_grad()
    def _native_pre_speech_ponder(
        self,
        combined_memory: torch.Tensor,
        internal_memory: torch.Tensor,
        *,
        seed: int,
        confidence: float,
        pass_budget: int,
        resource_budget_ms: int,
        noise: float,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Refine this turn's latent memory after a learned Ponder proposal.

        This is inference, not idle rehearsal or an optimizer step. Liquid/LIF
        scratch stays private to this attempt, while the returned memory bias
        conditions every response candidate and its deterministic replay.
        """

        started = time.perf_counter()
        current = combined_memory.detach()
        refined = internal_memory.detach()
        liquid_state = self.liquid_state.detach().clone()
        membrane = self.router.population.membrane.detach().clone()
        spike_count = self.router.population.spike_count.detach().clone()
        # A local CPU generator works on all deployed accelerators and never
        # changes the global training/decode RNG. Keep the exploratory drive
        # fixed across passes so convergence is meaningful.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        exploration = torch.randn(
            current.shape, generator=generator, dtype=torch.float32
        ).to(current) * max(0.0, float(noise)) * 0.025
        anchor = current + exploration
        passes = 0
        router_steps = 0
        final_delta = 0.0
        converged = False
        stop_reason = "neural-budget"
        try:
            while passes < max(1, int(pass_budget)):
                if cancel_check is not None and cancel_check():
                    raise ChatGenerationCancelled("chat generation was cancelled")
                readings = self.resource_policy.status()
                if readings.get("diskPressure") or readings.get("memoryPressure"):
                    stop_reason = "resource-pressure"
                    break
                if (time.perf_counter() - started) * 1000.0 >= max(
                    1, int(resource_budget_ms)
                ):
                    stop_reason = "resource-budget"
                    break
                if self.config.liquid_dynamics:
                    liquid_state, controls = self.liquid(
                        current, state=liquid_state, elapsed=1.0
                    )
                    driven = 0.65 * anchor + 0.25 * current + 0.10 * liquid_state
                    threshold = float(controls["threshold_offset"].mean().item())
                else:
                    driven = 0.65 * anchor + 0.35 * current
                    threshold = 0.0
                if self.config.spiking_dynamics:
                    updated, _metrics = self.router.route(
                        driven, steps=2, learn=False, threshold_offset=threshold
                    )
                    router_steps += 2
                else:
                    updated = driven
                next_refined = self.idea_adapter(updated)
                if not bool(torch.isfinite(next_refined).all()):
                    raise RuntimeError("non-finite native pre-speech Ponder state")
                final_delta = float(
                    (next_refined.float() - refined.float())
                    .square().mean().sqrt().item()
                )
                current = updated
                refined = next_refined
                passes += 1
                if cancel_check is not None and cancel_check():
                    raise ChatGenerationCancelled("chat generation was cancelled")
                scale = float(refined.float().square().mean().sqrt().item())
                if final_delta <= 1e-4 * (1.0 + scale):
                    converged = True
                    stop_reason = "converged"
                    break
        finally:
            self.router.population.membrane.copy_(membrane)
            self.router.population.spike_count.copy_(spike_count)
        return refined, {
            "activated": True,
            "activated_by": "learned-action-head",
            "passes": passes,
            "converged": converged,
            "stop_reason": stop_reason,
            "initial_score": float(confidence),
            "final_delta": final_delta,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "seed": int(seed),
            "router_steps": router_steps,
            "memory_bias_delta": float(
                (refined.float() - internal_memory.detach().float())
                .square().mean().sqrt().item()
            ),
            "phase": "pre-speech",
            "private": True,
            "visibleMagicTags": False,
        }

    @torch.no_grad()
    def _candidate_nll(
        self,
        prompt_ids: torch.Tensor,
        generated: torch.Tensor,
        memory_bias: torch.Tensor,
    ) -> float:
        sequence = generated[:, -self.config.max_seq_len :]
        offset = max(0, generated.shape[1] - self.config.max_seq_len)
        prompt_remaining = max(0, prompt_ids.shape[1] - offset)
        labels = sequence.clone()
        labels[:, :prompt_remaining] = self.tokenizer.pad_id
        if bool(labels[:, 1:].ne(self.tokenizer.pad_id).any()):
            loss = self.decoder(
                sequence, memory_bias=memory_bias, labels=labels
            )["loss"]
            return float(loss.item())
        return 100.0

    @staticmethod
    def _validated_chat_turn_id(value: str) -> str:
        turn_id = str(value or "").strip()
        if not turn_id:
            return ""
        if (
            len(turn_id) > 128
            or "\x00" in turn_id
            or any(character in "\r\n" for character in turn_id)
        ):
            raise ValueError("chat turn id is invalid")
        return turn_id

    @staticmethod
    def _explicit_response_length_constraint(
        text: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse an explicit user length instruction without prompt expansion."""

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        segments = [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text))
            if value.strip()
        ]
        instructions = [
            value
            for value in segments
            if re.match(
                r"^(?:please\s+)?(?:reply|respond|answer)\b",
                value,
                flags=re.IGNORECASE,
            )
        ]
        for instruction in instructions:
            count_match = re.match(
                r"^(?:please\s+)?(?:reply|respond|answer)\b"
                r"[^\r\n.!?]{0,40}?\b(?:in|with|using)\s+"
                r"(?:(?:only|exactly|at\s+most|no\s+more\s+than)\s+)?"
                r"(?P<count>\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|a\s+single)\s+"
                r"(?P<unit>words?|tokens?|sentences?|characters?)\b",
                instruction,
                flags=re.IGNORECASE,
            )
            if count_match is None:
                continue
            raw_count = count_match.group("count").casefold()
            count = (
                1
                if raw_count == "a single"
                else number_words.get(
                    raw_count,
                    int(raw_count) if raw_count.isdigit() else 0,
                )
            )
            if count > 0:
                return {
                    "kind": "count",
                    "unit": count_match.group("unit").casefold().rstrip("s"),
                    "count": min(4096, count),
                }
        literal_match = next(
            (
                matched
                for instruction in instructions
                if (
                    matched := re.match(
                        r"^(?:please\s+)?(?:reply|respond|answer)\s+"
                        r"(?:only\s+)?(?:with|using)\s+(?:only\s+)?"
                        r"(?P<literal>[^\r\n.!?]{1,160})\s*(?:[.!?]|$)",
                        instruction,
                        flags=re.IGNORECASE,
                    )
                )
                is not None
            ),
            None,
        )
        if literal_match is None:
            return None
        literal = literal_match.group("literal").strip().strip("\"'`").strip()
        literal = re.sub(
            r"^(?:the\s+)?(?:word|phrase)\s+",
            "",
            literal,
            count=1,
            flags=re.IGNORECASE,
        ).strip().strip("\"'`").strip()
        if not literal:
            return None
        # These are semantic references to the requested value, not literal
        # strings the user asked the decoder to emit. Treating "the code" as
        # a two-word literal previously capped an exact neural association at
        # four boundary tokens and turned ORCHID-7421 into ORCH.
        referential = re.fullmatch(
            r"(?:it|this|that|"
            r"(?:the|your|its|my|our)\s+"
            r"(?:(?:exact|final|correct|requested|remembered)\s+)?"
            r"(?:answer|code|color|value|result|name|number|fact|marker|response|call\s+sign))"
            r"(?:\s+and\s+nothing\s+else)?",
            re.sub(r"\s+", " ", literal).casefold(),
        )
        if referential is not None:
            return None
        return {
            "kind": "literal",
            "unit": "literal",
            "count": max(1, len(re.findall(r"\S+", literal))),
            "literal": literal,
            "literalSha256": hashlib.sha256(
                literal.encode("utf-8")
            ).hexdigest(),
        }

    def _response_generation_budget(
        self,
        text: str,
        *,
        cognitive_demand: float,
        caller_limit: Optional[int],
    ) -> Tuple[int, Dict[str, Any]]:
        """Intersect state, caller, and typed user response-length bounds."""

        state_ceiling = int(
            self.config.generation_token_budget(cognitive_demand)
        )
        caller_ceiling = (
            max(1, int(caller_limit))
            if caller_limit is not None
            else state_ceiling
        )
        resolved = min(state_ceiling, caller_ceiling)
        constraint = self._explicit_response_length_constraint(text)
        public_constraint: Optional[Dict[str, Any]] = None
        explicit_ceiling: Optional[int] = None
        if constraint is not None:
            unit = str(constraint["unit"])
            count = max(1, int(constraint["count"]))
            if constraint["kind"] == "literal":
                literal = str(constraint["literal"])
                native_token_count = len(self.tokenizer.encode(literal))
                token_count = max(
                    1,
                    count,
                    native_token_count,
                )
                explicit_ceiling = token_count + 2
            elif unit == "token":
                explicit_ceiling = count + 1
            elif unit == "word":
                explicit_ceiling = count * 4 + 2
            elif unit == "sentence":
                explicit_ceiling = count * 48
            else:
                explicit_ceiling = math.ceil(count / 2.0) + 2
            explicit_ceiling = max(2, int(explicit_ceiling))
            resolved = min(resolved, explicit_ceiling)
            public_constraint = {
                key: value
                for key, value in constraint.items()
                if key != "literal"
            }
            public_constraint["resolvedTokenCeiling"] = explicit_ceiling
        source = "hardware-and-organic-state"
        if caller_limit is not None:
            source = "caller-within-hardware-and-organic-state"
        if explicit_ceiling is not None and explicit_ceiling <= min(
            state_ceiling, caller_ceiling
        ):
            source = "explicit-user-response-length"
        return max(1, resolved), {
            "source": source,
            "stateCeilingTokens": state_ceiling,
            "callerCeilingTokens": (
                caller_ceiling if caller_limit is not None else None
            ),
            "explicitConstraint": public_constraint,
            "explicitLiteral": (
                str(constraint["literal"])
                if constraint is not None and constraint["kind"] == "literal"
                else None
            ),
            "promptTextExpanded": False,
            "stateLengthHead": "hardware-and-organic-state-v1",
        }

    def _completed_chat_result(
        self, receipt: Mapping[str, Any]
    ) -> Dict[str, Any]:
        human_message = next(
            (
                dict(message)
                for message in self.messages
                if message.get("id") == receipt["humanMessageId"]
                and message.get("role") == "human"
            ),
            None,
        )
        brain_message = next(
            (
                dict(message)
                for message in self.messages
                if message.get("id") == receipt["brainMessageId"]
                and message.get("role") == "brain"
            ),
            None,
        )
        trace = next(
            (
                dict(value)
                for value in self.traces
                if value.get("id") == receipt["traceId"]
            ),
            None,
        )
        if human_message is None or brain_message is None or trace is None:
            raise RuntimeError("committed chat receipt references missing state")
        if (
            human_message.get("turn_id") != receipt["turnId"]
            or brain_message.get("turn_id") != receipt["turnId"]
            or trace.get("turn_id") != receipt["turnId"]
            or hashlib.sha256(
                str(human_message.get("content", "")).encode("utf-8")
            ).hexdigest()
            != receipt["inputSha256"]
            or trace.get("input_sha256") != receipt["inputSha256"]
            or trace.get("parameter_checksum_after")
            != receipt["parameterChecksumAfter"]
            or int(receipt["inferenceCount"])
            > int(self.counters.get("inference_count", 0))
        ):
            raise RuntimeError("committed chat receipt binding diverged")
        runtime_card = self.runtime_card()
        runtime_card["available_tool_ids"] = list(
            trace.get("available_tool_ids", [])
        )
        return {
            "brainId": self.brain_id,
            "text": str(brain_message.get("content", "")),
            "response": str(brain_message.get("content", "")),
            "content": str(brain_message.get("content", "")),
            "humanMessage": human_message,
            "message": brain_message,
            "trace": trace,
            "metrics": self.metrics(),
            "runtimeCard": runtime_card,
            "availableToolIds": list(trace.get("available_tool_ids", [])),
            # Never replay a tool/action side effect after an acknowledgement
            # was lost. The persisted trace remains the audit record.
            "actions": [],
            "turnReceipt": dict(receipt),
            "turnCommitted": True,
            "idempotentCompletion": True,
        }

    def chat(
        self,
        text: str,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
        stream_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
        turn_id: str = "",
        defer_slow_learning: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            raise ValueError("chat input cannot be empty")
        if len(clean) > 1_000_000:
            raise ValueError("chat input is too large")

        def cancellation_boundary() -> None:
            if cancel_check is not None and cancel_check():
                raise ChatGenerationCancelled("chat generation was cancelled")

        cancellation_boundary()
        turn_id = self._validated_chat_turn_id(turn_id)
        input_sha256 = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if turn_id:
            existing_receipt = next(
                (
                    receipt
                    for receipt in reversed(self.completed_chat_turns)
                    if receipt.get("turnId") == turn_id
                    and receipt.get("inputSha256") == input_sha256
                ),
                None,
            )
            if existing_receipt is not None:
                return self._completed_chat_result(existing_receipt)
        requested_generation_tokens = (
            max(1, int(max_new_tokens))
            if max_new_tokens is not None
            else None
        )
        before_checksum = self.parameter_checksum()
        before_parameters = self._parameter_copy()
        normalized_tools = self._normalize_tool_schemas(tool_schemas)

        cue = self.memory.vector_for_text(clean)
        # Default speech is conditioned by the shared neural substrate and
        # transient working activity. Exact episode keys and the separate
        # sequence-statistical field must not act as an answer database.
        if self.config.vector_symbolic_memory:
            recalled_vector, recalled = self.memory.recall_vector(
                cue,
                workspace_slots=self.config.working_memory_slots,
                record_activity=False,
            )
        else:
            recalled_vector, recalled = cue, []
        recall_audit = dict(self.memory._last_recall_audit)
        transactional_generation = cancel_check is not None

        def commit_fast_experience() -> Dict[str, Any]:
            self.memory.record_recall_activity(recalled)
            return self.learn_experience(
                clean,
                kind="experience",
                source="conversation",
                source_label="chat",
                steps=0,
                importance=0.7,
            )

        if transactional_generation:
            # A cancelled decode must not admit an incomplete turn to memory.
            experience = self._preview_chat_experience(clean, cue, recalled)
        else:
            # Preserve the original organic path for mutable/associative chat:
            # valid fast activity participates in its own action decision.
            experience = commit_fast_experience()
        recall_model = self._idea_model_vector(recalled_vector)
        tool_model = self._tool_schema_vector(normalized_tools)
        working_model = self._working_memory_vector()
        working_memory_used = (
            len(self.working_memory) if working_model is not None else 0
        )
        components = [
            (0.58 if working_model is not None else 0.65, experience["idea"]),
            (0.25 if working_model is not None else 0.35, recall_model),
        ]
        if working_model is not None:
            components.append((0.17, working_model))
        if tool_model is not None:
            components = [(weight * 0.88, value) for weight, value in components]
            components.append((0.12, tool_model))
        combined_memory = sum(
            weight * value for weight, value in components
        )
        internal_memory = self.idea_adapter(combined_memory)

        if seed is None:
            seed_material = (
                "%s:%d:%s"
                % (self.brain_id, self.counters["inference_count"], clean)
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
            seed &= 0x7FFFFFFF
        prompt_list, recent_prompt_tokens = self._prompt_with_recent_context(
            clean
        )
        prompt_ids = torch.tensor(
            [prompt_list],
            dtype=torch.long,
            device=self.device,
        )
        prompt_token_hash = self._token_sequence_hash(
            prompt_ids[0].tolist()
        )
        pending_current_context = {
            "tokenCount": int(prompt_ids.shape[1]),
            "tokenHash": prompt_token_hash,
            "recentTokenCount": len(self.recent_token_context),
            "recentTokenHash": self._token_sequence_hash(
                self.recent_token_context
            ),
            "sensorySlots": 0,
            "updatedAt": _iso_now(),
        }
        if not transactional_generation:
            self.current_context = pending_current_context
        decoder_training_before_generation = bool(self.decoder.training)
        self.decoder.eval()
        with torch.no_grad():
            # This is the exact deployed action route: the complete bounded
            # runtime prompt plus the same recalled, working, sensory, and
            # capability-conditioned memory used by generation.
            routing_output = self.decoder(
                prompt_ids,
                memory_bias=internal_memory,
                use_global_workspace=True,
            )
            if prompt_ids.shape[1] > 1:
                routed_ids = prompt_ids[
                    :, -routing_output["logits"].shape[1] :
                ]
                decision_prediction_loss = float(
                    F.cross_entropy(
                        routing_output["logits"][:, :-1].float().reshape(
                            -1, routing_output["logits"].shape[-1]
                        ),
                        routed_ids[:, 1:].reshape(-1),
                    ).item()
                )
            else:
                decision_prediction_loss = 0.0
            action_cue = self._idea_model_vector(cue)
            language_action_feature = (
                routing_output["hidden"][:, -1] + 0.5 * action_cue
            )
            language_action_logits = self.decoder.action_policy(
                language_action_feature
            )
            internal_action_logits = self.decoder.internal_action_policy(
                action_cue
            )
        uncertainties = [
            float(self.memory.concepts[concept_id]["uncertainty"])
            for concept_id in experience["concept_ids"]
            if concept_id in self.memory.concepts
        ]
        uncertainty = (
            sum(uncertainties) / len(uncertainties) if uncertainties else 0.5
        )
        training_loss = decision_prediction_loss
        prediction_error = training_loss / (1.0 + abs(training_loss))
        experience["retention_prediction_error"] = prediction_error
        novelty = float(experience["novelty"])
        learning_progress = self._organic_state()["learningProgress"]
        curiosity = max(
            0.0,
            min(
                1.0,
                0.42 * novelty
                + 0.34 * prediction_error
                + 0.18 * uncertainty
                + 0.06 * learning_progress,
            ),
        )
        liquid_ponder = float(experience["liquid_controls"]["ponder_scale"])
        compute_demand = max(
            0.0,
            min(
                1.0,
                0.36 * curiosity
                + 0.28 * uncertainty
                + 0.22 * novelty
                + 0.14 * max(0.0, liquid_ponder - 1.0),
            ),
        )
        generation_tokens, generation_budget = self._response_generation_budget(
            clean,
            cognitive_demand=compute_demand,
            caller_limit=requested_generation_tokens,
        )
        ponder_factors = {
            "liquid": liquid_ponder,
            "novelty": novelty,
            "uncertainty": uncertainty,
            "predictionError": prediction_error,
            "learningProgress": learning_progress,
            "curiosity": curiosity,
        }
        ponder_pass_budget = max(
            1,
            int(
                round(
                    ponder_factors["liquid"]
                    + 3.0 * compute_demand
                    + 1.5 * ponder_factors["uncertainty"]
                )
            ),
        )
        organic_noise = max(
            0.005,
            min(
                0.35,
                0.015
                + 0.16 * curiosity
                + 0.09 * uncertainty
                + 0.05
                * float(experience["liquid_controls"]["noise_scale"]),
            ),
        )
        with torch.no_grad():
            expert_route = (
                routing_output["expert_routing"][0].detach().cpu().tolist()
                if "expert_routing" in routing_output
                else []
            )
            action_state = {
                "novelty": novelty,
                "uncertainty": uncertainty,
                "predictionError": prediction_error,
                "learningProgress": learning_progress,
                "curiosity": curiosity,
                "computeDemand": compute_demand,
                "organicNoise": organic_noise,
                "branches": 1,
            }
            active_assemblies = [
                str(experience.get("assembly_id", experience["idea_id"]))
            ]
            active_assemblies.extend(
                str(item["idea_id"])
                for item in recalled
                if str(item.get("idea_id", "")) not in active_assemblies
            )
            action_scores, proposed_actions = self._select_structured_actions(
                0.35 * language_action_logits
                + 0.65 * internal_action_logits,
                schemas=normalized_tools,
                input_text=clean,
                assembly_ids=active_assemblies,
                organic_state=action_state,
                supporting_action_logits=(
                    language_action_logits,
                    internal_action_logits,
                ),
                neural_state=action_cue,
            )

        # Measure only neural selection/decoding (including deterministic
        # visible-stream replay), not learning committed after the response.
        # The measurement is operational state and never model-facing text.
        generation_started_at = time.perf_counter()
        candidates: List[Dict[str, Any]] = []
        native_ponder_trace: Dict[str, Any] = {
            "activated": False,
            "activated_by": "inactive",
            "passes": 0,
            "converged": False,
            "stop_reason": "not-selected",
            "initial_score": float(action_scores.get("ponder", 0.0)),
            "final_delta": 0.0,
            "elapsed_ms": 0.0,
            "seed": int(seed),
            "router_steps": 0,
            "memory_bias_delta": 0.0,
            "phase": "pre-speech",
            "private": True,
            "visibleMagicTags": False,
        }
        ponder_steps = 0



        native_ponder_action = next(
            (action for action in proposed_actions if action.get("kind") == "ponder"),
            None,
        )
        if native_ponder_action is not None and not any(
            action.get("kind") == "stop" for action in proposed_actions
        ):
            try:
                internal_memory, native_ponder_trace = self._native_pre_speech_ponder(
                    combined_memory,
                    internal_memory,
                    seed=int(seed),
                    confidence=float(native_ponder_action.get("confidence", 0.0)),
                    pass_budget=ponder_pass_budget,
                    resource_budget_ms=max(
                        250, min(10_000, int(500 + 3500 * compute_demand))
                    ),
                    noise=organic_noise,
                    cancel_check=cancel_check,
                )
            except Exception:
                self.decoder.train(decoder_training_before_generation)
                raise
            ponder_steps = int(native_ponder_trace["passes"])
            arguments = native_ponder_action.setdefault("arguments", {})
            arguments["completedInTurn"] = True
            arguments["ponderTrace"] = dict(native_ponder_trace)
        # Candidate computation has no fixed "parallel thoughts" or
        # hardware branch ceiling. Neural energy is derived from the
        # current organic state and working workspace, then spent until
        # activity converges or the host reserve reports pressure.
        remaining_neural_energy = max(
            1.0,
            1.0
            + compute_demand
            * (1.0 + math.log2(max(2, self.config.working_memory_slots))),
        )
        branch = 0
        best_score = -float("inf")
        score_change = float("inf")
        while remaining_neural_energy > 0.0:
            cancellation_boundary()
            branch_seed = int(seed) + branch * 7919
            candidate, branch_entropies = self.decoder.generate(
                prompt_ids,
                memory_bias=internal_memory,
                max_new_tokens=generation_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                noise=organic_noise * (1.0 + 0.04 * ponder_steps),
                seed=branch_seed,
                printable_only=True,
                cancelled=cancel_check,
            )
            cancellation_boundary()
            nll = self._candidate_nll(
                prompt_ids, candidate, internal_memory
            )
            entropy = (
                sum(branch_entropies) / len(branch_entropies)
                if branch_entropies
                else 0.0
            )
            intrinsic_score = (
                -(1.0 + 0.35 * (1.0 - uncertainty)) * nll
                + curiosity * entropy * 0.08
                + novelty
                * min(candidate.shape[1] - prompt_ids.shape[1], 32)
                / 320.0
            )
            candidates.append(
                {
                    "tensor": candidate,
                    "entropies": branch_entropies,
                    "seed": branch_seed,
                    "selfNll": nll,
                    "entropy": entropy,
                    "score": intrinsic_score,
                    "backend": "mutable-omni-decoder",
                }
            )
            improvement = intrinsic_score - best_score
            score_change = (
                abs(improvement) if math.isfinite(best_score) else float("inf")
            )
            best_score = max(best_score, intrinsic_score)
            normalized_entropy = max(0.0, min(4.0, entropy)) / 4.0
            remaining_neural_energy -= (
                1.0
                + 0.35 * normalized_entropy
                + 0.20 * max(0.0, 1.0 - compute_demand)
            )
            branch += 1
            convergence_floor = 0.0025 * (1.0 + abs(best_score))
            if len(candidates) > 1 and score_change <= convergence_floor:
                break
            readings = self._resource_readings()
            disk_free = readings.get("diskFreeBytes")
            memory_free = readings.get("availableMemoryBytes")
            if (
                isinstance(disk_free, int)
                and disk_free <= 512 * 1024 * 1024
            ) or (
                isinstance(memory_free, int)
                and memory_free <= 384 * 1024 * 1024
            ):
                break
        branch_count = len(candidates)
        action_state["branches"] = branch_count
        with torch.no_grad():
            action_scores, proposed_actions = self._select_structured_actions(
                0.35 * language_action_logits
                + 0.65 * internal_action_logits,
                schemas=normalized_tools,
                input_text=clean,
                assembly_ids=active_assemblies,
                organic_state=action_state,
                supporting_action_logits=(
                    language_action_logits,
                    internal_action_logits,
                ),
                neural_state=action_cue,
            )
        if native_ponder_action is not None and native_ponder_trace["activated"]:
            # The final branch-aware materialization must preserve the
            # already completed action, not schedule post-response idle
            # rehearsal or lose evidence of this turn's actual work.
            proposed_actions = [
                action for action in proposed_actions
                if action.get("kind") != "ponder"
            ]
            proposed_actions.append(native_ponder_action)
        selected_branch = max(
            range(len(candidates)),
            key=lambda index: candidates[index]["score"],
        )
        selected = candidates[selected_branch]
        generated = selected["tensor"]
        entropies = selected["entropies"]

        if stream_callback is not None:
            for action in proposed_actions:
                stream_callback(
                    "action",
                    {
                        "actionId": uuid.uuid4().hex,
                        "action": action,
                    },
                )

            def stream_token(
                token: torch.Tensor, step: int, entropy: float
            ) -> None:
                token_ids = token.reshape(-1).tolist()
                delta = self.tokenizer.decode(token_ids)
                if delta:
                    stream_callback(
                        "token",
                        {
                            "delta": delta,
                            "step": int(step),
                            "entropy": float(entropy),
                        },
                    )

            # Candidate selection remains private neural computation.
            # Replay the chosen deterministic branch once so the visible
            # stream is exactly the committed response.
            replayed, replay_entropies = self.decoder.generate(
                prompt_ids,
                memory_bias=internal_memory,
                max_new_tokens=generation_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                noise=organic_noise * (1.0 + 0.04 * ponder_steps),
                seed=int(selected["seed"]),
                printable_only=True,
                token_callback=stream_token,
                cancelled=cancel_check,
            )
            cancellation_boundary()
            if not torch.equal(replayed, generated):
                raise RuntimeError(
                    "deterministic selected-branch replay diverged"
                )
            generated = replayed
            entropies = replay_entropies

        new_ids = generated[
            0, prompt_ids.shape[1] :
        ].detach().cpu().tolist()
        generated_token_count = len(new_ids)
        response = self.tokenizer.decode(new_ids).strip()
        if not response:
            response = (
                self.tokenizer.decode(new_ids, skip_special=False) or "?"
            )

        generation_stop_reason = (
            "token-budget"
            if generated_token_count >= generation_tokens
            else "learned-boundary"
        )
        generation_budget_truncated = bool(
            generation_stop_reason == "token-budget"
            and generated_token_count >= generation_tokens
        )

        if stream_callback is not None:
            cancellation_boundary()
            # Visible decoding is finished, but the human/brain pair is not an
            # authoritative turn until the immediate neural updates and atomic
            # checkpoint below succeed. This phase lets the desktop stop its
            # token cursor and accept queued input without mislabeling the turn
            # as committed.
            try:
                stream_callback(
                    "phase",
                    {
                        "phase": "reply-complete-learning",
                        "replyComplete": True,
                        "turnCommitted": False,
                        "learning": True,
                        "saving": True,
                    },
                )
            except Exception:
                self.decoder.train(decoder_training_before_generation)
                raise

        cancellation_boundary()
        if transactional_generation:
            # Generation and the visible stream have succeeded. The exact
            # fast experience can now enter substrate/STDP/working memory;
            # slow shared-representation learning remains transactional below.
            experience = commit_fast_experience()
            self.current_context = pending_current_context

        generation_elapsed_seconds = max(
            1e-9, time.perf_counter() - generation_started_at
        )
        generation_tokens_per_second = (
            float(generated_token_count) / generation_elapsed_seconds
        )

        # A brain's own same-turn output is not independent teaching evidence.
        # The full user experience always learns, while generated speech must
        # receive later feedback/evidence before it can become a target. This
        # prevents fluent mistakes and native-model nonsense from reinforcing
        # themselves without imposing a behavioral preference objective.
        generated_response_supervision_eligible = False
        generated_response_exclusion_reason = (
            "truncated-neural-or-token-budget-response"
            if generation_budget_truncated
            else "same-turn-generated-output-awaits-independent-evidence"
        )
        own_training = None
        if (
            self.config.learn_from_own_messages
            and generated_response_supervision_eligible
        ):
            own_training = self.learn_experience(
                response,
                kind="experience",
                source="self",
                source_label="self-response",
                steps=0,
                importance=0.35,
            )
        pair_training = None
        # The unconditional fast update above already admitted the complete
        # experience to the shared substrate and STDP pathways. Do not create
        # a second cue-to-token answer index from the chat turn.
        action_calibration = None
        slow_mutation_requested = bool(
            not defer_slow_learning
            and self.config.online_learning
            and int(self.config.online_steps) > 0
        )
        slow_mutation_applied = False
        slow_mutation_rolled_back = False
        slow_mutation_failure: Optional[Dict[str, Any]] = None
        slow_mutation_stage = "disabled"
        slow_parameter_checksum_before = self._slow_parameter_checksum()
        slow_parameter_checksum_after = slow_parameter_checksum_before
        cortical_parameter_checksum_before = None
        cortical_parameter_checksum_after = None
        if slow_mutation_requested:
            cortical_parameter_checksum_before = (
                self._cortical_parameter_checksum()
            )
            slow_snapshot = self._snapshot_slow_transaction_state()
            pre_slow_training = copy.deepcopy(experience["training"])
            slow_parameter_checksum_before = str(slow_snapshot["checksum"])
            try:
                slow_mutation_stage = "experience-learning"
                experience["training"] = self._optimize_experience(
                    clean,
                    cue,
                    steps=int(self.config.online_steps),
                )
                slow_mutation_stage = "dialogue-learning"
                if generated_response_supervision_eligible:
                    pair_training = self._optimize_dialogue_pair(
                        clean, response, cue, steps=1
                    )
                # The current turn and optional self-response entered fast
                # neural state before action selection with steps=0. Defer
                # topology growth until the full slow update is available for
                # native action-path retention to validate atomically.
                slow_mutation_stage = "expert-growth"
                experience["grew_expert"] = self._maybe_grow(
                    float(experience["novelty"]),
                    experience["idea"][0],
                )
                if own_training is not None:
                    own_training["grew_expert"] = self._maybe_grow(
                        float(own_training["novelty"]),
                        own_training["idea"][0],
                    )
                if self._can_retain_native_action_policy():
                    # Recompute this exact runtime route once after the shared
                    # representation mutation. Retention then revalidates every
                    # bundled trajectory against current neural representations
                    # in one separate mask-correct batch.
                    slow_mutation_stage = "action-retention"
                    post_recall_model = self._idea_model_vector(recalled_vector)
                    post_tool_model = self._tool_schema_vector(normalized_tools)
                    post_working_model = self._working_memory_vector()
                    post_components = [
                        (
                            0.58 if post_working_model is not None else 0.65,
                            experience["idea"],
                        ),
                        (
                            0.25 if post_working_model is not None else 0.35,
                            post_recall_model,
                        ),
                    ]
                    if post_working_model is not None:
                        post_components.append((0.17, post_working_model))
                    if post_tool_model is not None:
                        post_components = [
                            (weight * 0.88, value)
                            for weight, value in post_components
                        ]
                        post_components.append((0.12, post_tool_model))
                    post_combined_memory = sum(
                        weight * value for weight, value in post_components
                    )
                    self.decoder.eval()
                    with torch.no_grad():
                        post_internal_memory = self.idea_adapter(
                            post_combined_memory
                        )
                        post_routing_output = self.decoder(
                            prompt_ids,
                            memory_bias=post_internal_memory,
                            use_global_workspace=True,
                        )
                        post_action_cue = self._idea_model_vector(cue)
                        post_language_feature = (
                            post_routing_output["hidden"][:, -1]
                            + 0.5 * post_action_cue
                        )
                    action_calibration = self._retain_native_action_policy(
                        pre_language_logits=language_action_logits,
                        pre_internal_logits=internal_action_logits,
                        pre_action_emitted=bool(proposed_actions),
                        post_language_feature=post_language_feature,
                        post_internal_feature=post_action_cue,
                        exact_route_decoder_forwards=1,
                    )
                    if (
                        action_calibration is None
                        or not bool(action_calibration.get("calibrated"))
                        or bool(action_calibration.get("rolledBack"))
                    ):
                        raise RuntimeError(
                            "native action retention rejected the slow mutation"
                        )
                slow_mutation_stage = "committed"
                slow_mutation_applied = True
            except Exception as error:
                failure_stage = slow_mutation_stage
                failed_training = copy.deepcopy(experience["training"])
                failed_pair_training = copy.deepcopy(pair_training)
                failed_calibration = copy.deepcopy(action_calibration)
                self._restore_slow_transaction_state(slow_snapshot)
                if failure_stage == "action-retention":
                    # The neural mutation is atomic, but its failed validation
                    # remains an auditable event rather than disappearing with
                    # the optimizer/training counters it caused.
                    self.counters["action_retention_checks"] += 1
                    self.counters["action_retention_failures"] += 1
                experience["training"] = pre_slow_training
                experience["grew_expert"] = False
                if own_training is not None:
                    own_training["grew_expert"] = False
                pair_training = None
                slow_mutation_applied = False
                slow_mutation_rolled_back = True
                slow_mutation_stage = "rolled-back"
                slow_mutation_failure = {
                    "stage": failure_stage,
                    "type": type(error).__name__,
                    "message": str(error)[:240],
                    "attemptedTraining": failed_training,
                    "attemptedDialogueTraining": failed_pair_training,
                }
                if failed_calibration is not None:
                    action_calibration = {
                        **failed_calibration,
                        "transactionRolledBack": True,
                    }
                elif failure_stage == "action-retention":
                    action_calibration = {
                        "mode": (
                            "exact-route-self-distillation+"
                            "current-neural-replay"
                        ),
                        "calibrated": False,
                        "rolledBack": True,
                        "transactionRolledBack": True,
                        "failureType": type(error).__name__,
                        "failure": str(error)[:240],
                    }
            slow_parameter_checksum_after = self._slow_parameter_checksum()
            cortical_parameter_checksum_after = (
                self._cortical_parameter_checksum()
            )

        after_checksum = self.parameter_checksum()
        delta_norm = self._parameter_delta_norm(before_parameters)
        self.counters["inference_count"] += 1
        now = _iso_now()
        user_message = {
            "id": uuid.uuid4().hex,
            "role": "human",
            "content": clean,
            "created_at": now,
            "attention_epoch": self._attention_epoch(),
            **({"turn_id": turn_id} if turn_id else {}),
        }
        assistant_message = {
            "id": uuid.uuid4().hex,
            "role": "brain",
            "content": response,
            "created_at": _iso_now(),
            "attention_epoch": self._attention_epoch(),
            **({"turn_id": turn_id} if turn_id else {}),
        }
        self.messages.extend([user_message, assistant_message])
        queued_slow_learning = None
        if defer_slow_learning:
            queued_slow_learning = self._enqueue_chat_slow_learning(
                turn_id=turn_id or str(user_message["id"]),
                input_sha256=input_sha256,
                human_message_id=str(user_message["id"]),
                experience=experience,
            )
            if queued_slow_learning is not None:
                slow_mutation_requested = True
                slow_mutation_stage = "queued-background"
        self._append_recent_dialogue(clean, response)
        self.current_context.update(
            {
                "recentTokenCount": len(self.recent_token_context),
                "recentTokenHash": self._token_sequence_hash(
                    self.recent_token_context
                ),
                "updatedAt": _iso_now(),
            }
        )
        train_loss = float(experience["training"]["loss"])
        if pair_training is not None:
            train_loss = (train_loss + float(pair_training["loss"])) / 2.0

        # A trace is a bag of measured mechanisms from this turn, not a
        # numbered story that implies every chat traversed the same stages.
        # Core observations are always recorded; optional mechanisms appear
        # only when their activation, mutation, rollback, or gate was measured.
        measured_mechanisms: List[Dict[str, str]] = [
            {
                "stage": "input-boundary",
                "detail": (
                    "Encoded the current turn plus explicit bounded recent "
                    "dialogue at the UTF-8 token boundary."
                ),
                "value": "%d tokens" % prompt_ids.shape[1],
            },
            {
                "stage": "recurrent-recall",
                "detail": (
                    "Settled signed recurrent activation using exact ternary "
                    "synapse contributions; inhibitory pathways competed "
                    "without using latent master magnitude, and no remembered "
                    "source text entered the token stream."
                ),
                "value": (
                    "%d active, %d inhibited signals, %d settling rounds"
                    % (
                        len(recalled),
                        int(recall_audit.get("inhibitorySignals", 0)),
                        int(recall_audit.get("settledRounds", 0)),
                    )
                ),
            },
            {
                "stage": "active-context",
                "detail": (
                    "Used the explicit bounded recent-dialogue token ring "
                    "and blended recurrent activity vectors; no long-term "
                    "source, behavioral prompt, or tool-schema prose was "
                    "added."
                    if working_model is not None
                    else (
                        "Used only the explicit bounded recent-dialogue "
                        "token ring; recurrent-vector injection is disabled."
                    )
                ),
                "value": (
                    "%d recent tokens, %d active vectors"
                    % (len(recent_prompt_tokens), working_memory_used)
                    if working_model is not None
                    else "%d recent tokens" % len(recent_prompt_tokens)
                ),
            },
        ]
        if normalized_tools:
            measured_mechanisms.append(
                {
                    "stage": "capability-conditioning",
                    "detail": (
                        "Encoded enabled tool IDs and actions through the "
                        "neural capability channel; no schema text was added "
                        "to prompt tokens."
                    ),
                    "value": "%d available tools" % len(normalized_tools),
                }
            )
        measured_ponder_passes = int(ponder_steps)
        measured_ponder_activated = bool(native_ponder_trace["activated"])
        if measured_ponder_activated or measured_ponder_passes > 0:
            measured_mechanisms.append(
                {
                    "stage": "recurrent-refinement",
                    "detail": (
                        "Ran private recurrent refinement selected by the "
                        "learned action route or measured neural uncertainty; "
                        "only operational convergence metrics are exposed."
                        if measured_ponder_passes > 0
                        else "Selected private recurrent refinement, but the "
                        "resource gate allowed no passes; no refinement is "
                        "claimed."
                    ),
                    "value": "%d native pre-speech passes; %s"
                    % (measured_ponder_passes, native_ponder_trace["stop_reason"]),
                }
            )
        generation_backend = "mutable-omni-decoder"
        measured_mechanisms.append(
            {
                "stage": "response-decoding",
                "detail": "Decoded through the mutable Omni byte cortex.",
                "value": generation_backend,
            }
        )
        stdp_update = float(experience["spiking"]["stdp_update"])
        spike_rate = float(experience["spiking"]["spike_rate"])
        if abs(stdp_update) > 0.0 or abs(spike_rate) > 0.0:
            measured_mechanisms.append(
                {
                    "stage": "connection-learning",
                    "detail": (
                        "Measured local spike-timing-dependent synaptic "
                        "activity for this turn."
                    ),
                    "value": "%.6f L1 update" % stdp_update,
                }
            )
        if slow_mutation_requested:
            measured_mechanisms.append(
                {
                    "stage": "slow-adaptation",
                    "detail": (
                        "Updated ternary decoder and idea-adapter master "
                        "parameters and committed the validated transaction."
                        if slow_mutation_applied
                        else (
                            "Queued this fast episode for priority-driven "
                            "cortical replay; the atomic candidate update "
                            "runs in preemptible background work."
                            if slow_mutation_stage == "queued-background"
                            else (
                            "Attempted the slow neural update, then restored "
                            "all slow parameters, expert topology, optimizer, "
                            "metaplastic state, and counters after validation "
                            "failed. Fast substrate and working-memory state "
                            "from the turn remain active."
                            )
                        )
                    ),
                    "value": (
                        "loss %.6f" % train_loss
                        if slow_mutation_applied
                        else (
                            "priority %.6f"
                            % float(queued_slow_learning.get("priority", 0.0))
                            if slow_mutation_stage == "queued-background"
                            and queued_slow_learning is not None
                            else "rolled back at %s"
                            % (
                                slow_mutation_failure.get("stage", "unknown")
                                if slow_mutation_failure is not None
                                else "unknown"
                            )
                        )
                    ),
                }
            )
        if pair_training is not None or not generated_response_supervision_eligible:
            measured_mechanisms.append(
                {
                    "stage": "dialogue-supervision-gate",
                    "detail": (
                        "Trained and committed human-to-brain role-boundary "
                        "prediction."
                        if pair_training is not None
                        else (
                            "The generated response was excluded from self and "
                            "dialogue supervision "
                            "until independent evidence is available."
                        )
                    ),
                    "value": (
                        "loss %.6f" % pair_training["loss"]
                        if pair_training is not None
                        else generated_response_exclusion_reason
                    ),
                }
            )
        measured_mechanisms.append(
            {
                "stage": "response-generation",
                "detail": (
                    "Sampled from the brain's mutable decoder with "
                    "liquid-controlled noise."
                ),
                "value": (
                    "%d generated tokens in %.1f ms (%.2f tokens/s)"
                    % (
                        generated_token_count,
                        generation_elapsed_seconds * 1000.0,
                        generation_tokens_per_second,
                    )
                ),
            }
        )
        trace = {
            "id": uuid.uuid4().hex,
            "created_at": _iso_now(),
            "attention_epoch": self._attention_epoch(),
            "seed": int(seed),
            "input_sha256": input_sha256,
            **({"turn_id": turn_id} if turn_id else {}),
            "textual_memory_injected": False,
            "long_term_source_text_injected": False,
            "generated_response_supervision": {
                "eligible": generated_response_supervision_eligible,
                "policy": "independent-evidence-required",
                "selfExperienceApplied": own_training is not None,
                "dialogueUpdateApplied": pair_training is not None,
                "exclusionReason": generated_response_exclusion_reason,
            },
            "tool_schema_text_injected": False,
            "hidden_prompt_text_expanded": False,
            "prompt_text_expanded": bool(recent_prompt_tokens),
            "prompt_token_count": int(prompt_ids.shape[1]),
            "recent_dialogue_context_injected": bool(recent_prompt_tokens),
            "recent_dialogue_token_count": len(recent_prompt_tokens),
            "fresh_attention_boundary": (
                None
                if self.fresh_attention_boundary is None
                else dict(self.fresh_attention_boundary)
            ),
            "recent_dialogue_token_ids_sha256": self._token_sequence_hash(
                recent_prompt_tokens
            ),
            "context_token_evictions": self.counters[
                "context_token_evictions"
            ],
            "working_context_capacity_tokens": self.config.max_seq_len,
            "generation_budget_tokens": generation_tokens,
            "generation_budget_source": generation_budget["source"],
            "generation_budget_state_ceiling_tokens": generation_budget[
                "stateCeilingTokens"
            ],
            "generation_budget_caller_ceiling_tokens": generation_budget[
                "callerCeilingTokens"
            ],
            "generation_budget_explicit_constraint": generation_budget[
                "explicitConstraint"
            ],
            "generation_budget_state_length_head": generation_budget[
                "stateLengthHead"
            ],
            "generation_stop_reason": generation_stop_reason,
            "generation_budget_truncated": generation_budget_truncated,
            "prompt_token_ids_sha256": prompt_token_hash,
            "available_tool_ids": [
                schema["id"] for schema in normalized_tools
            ],
            "available_tool_actions": {
                schema["id"]: list(schema["actions"])
                for schema in normalized_tools
            },
            "tool_schema_channel": (
                "substrate-capability-embedding" if normalized_tools else "none"
            ),
            "memory_injection": self.config.memory_injection,
            "working_memory_channel": (
                "recurrent-vector"
                if working_model is not None
                else "disabled"
            ),
            "working_memory_vectors": (
                working_memory_used
            ),
            "parameter_checksum_before": before_checksum,
            "parameter_checksum_after": after_checksum,
            "parameter_delta_norm": delta_norm,
            "stdp_update": float(experience["spiking"]["stdp_update"]),
            "spike_rate": float(experience["spiking"]["spike_rate"]),
            "memory_settling": dict(experience["memory_settling"]),
            "liquid_controls": experience["liquid_controls"],
            "recalled_idea_ids": [
                item["idea_id"] for item in recalled
            ],
            "spreading_activation": recall_audit,
            "expert_route": expert_route,
            "expert_grew": bool(
                experience["grew_expert"]
                or (
                    own_training is not None
                    and own_training["grew_expert"]
                )
            ),
            "own_response_expert_grew": bool(
                own_training is not None
                and own_training["grew_expert"]
            ),
            "growth_pause": self.growth_pause,
            "train_loss": train_loss,
            "decision_prediction_loss": decision_prediction_loss,
            "slow_mutation_requested": slow_mutation_requested,
            "slow_mutation_applied": slow_mutation_applied,
            "slow_mutation_rolled_back": slow_mutation_rolled_back,
            "slow_mutation_stage": slow_mutation_stage,
            "slow_mutation_failure": slow_mutation_failure,
            "slow_learning_job": (
                dict(queued_slow_learning)
                if queued_slow_learning is not None
                else None
            ),
            "slow_parameter_checksum_before": (
                slow_parameter_checksum_before
            ),
            "slow_parameter_checksum_after": slow_parameter_checksum_after,
            "cortical_parameter_checksum_before": (
                cortical_parameter_checksum_before
            ),
            "cortical_parameter_checksum_after": (
                cortical_parameter_checksum_after
            ),
            "cortical_parameters_updated": (
                cortical_parameter_checksum_before is not None
                and cortical_parameter_checksum_after is not None
                and
                cortical_parameter_checksum_before
                != cortical_parameter_checksum_after
            ),
            "response_mode": "neural-generation",
            "response_quality_assessed": False,
            "response_quality_passed": None,
            "generation_entropy": (
                sum(entropies) / len(entropies) if entropies else 0.0
            ),
            "generation_elapsed_ms": generation_elapsed_seconds * 1000.0,
            "generated_token_count": int(generated_token_count),
            "generation_tokens_per_second": generation_tokens_per_second,
            "generation_measurement_scope": (
                "neural-selection-decode-and-visible-stream-replay"
            ),
            "generation_backend": generation_backend,
            "generation_cache_mode": str(
                getattr(self.decoder, "last_generation_cache_mode", "unknown")
            ),
            "ponder": dict(native_ponder_trace),
            "ponder_steps": ponder_steps,
            "ponder_factors": ponder_factors,
            "organic_state": action_state,
            "action_policy_scores": action_scores,
            "action_policy_calibration": action_calibration,
            "action_policy_channel": (
                "exact-runtime-prompt+internal-memory+idea-fusion"
            ),
            "action_policy_prompt_token_ids_sha256": prompt_token_hash,
            "action_policy_recent_dialogue_tokens": len(
                recent_prompt_tokens
            ),
            "action_policy_working_memory_vectors": working_memory_used,
            "action_policy_capability_conditioned": tool_model is not None,
            "action_policy_deployed_kind": max(
                action_scores, key=action_scores.get
            ),
            "action_policy_deployed_confidence": max(
                action_scores.values()
            ),
            "action_policy_tool_support_evidence": [
                dict(action["supportEvidence"])
                for action in proposed_actions
                if isinstance(action.get("supportEvidence"), Mapping)
            ],
            "tool_route_evidence": getattr(self, "_last_tool_route_evidence", None),
            "action_policy_synthetic_guarantee": False,
            "proposed_action_kinds": [
                action["kind"] for action in proposed_actions
            ],
            "branches": [
                {
                    "index": index,
                    "seed": candidate["seed"],
                    "selfNll": candidate["selfNll"],
                    "entropy": candidate["entropy"],
                    "intrinsicScore": candidate["score"],
                    "backend": candidate.get("backend"),
                    "quality": candidate.get("quality"),
                }
                for index, candidate in enumerate(candidates)
            ],
            "selected_branch": selected_branch,
            "mechanism_order": "event-derived-unordered",
            "steps": measured_mechanisms,
            "note": (
                "Operational trace of measured activations and mutations; it is "
                "not a hidden chain-of-thought transcript."
            ),
        }
        self.traces.append(trace)
        self.events.append(
            "chat-mutation",
            {
                "traceId": trace["id"],
                "parameterChecksumBefore": before_checksum,
                "parameterChecksumAfter": after_checksum,
                "parameterDeltaNorm": delta_norm,
                "stdpUpdate": trace["stdp_update"],
                "spikeRate": trace["spike_rate"],
                "trainLoss": trace["train_loss"],
                "availableToolIds": trace["available_tool_ids"],
                **({"turnId": turn_id} if turn_id else {}),
            },
        )
        turn_receipt: Optional[Dict[str, Any]] = None
        if turn_id:
            turn_receipt = {
                "format": CHAT_TURN_RECEIPT_FORMAT,
                "formatVersion": 1,
                "turnId": turn_id,
                "inputSha256": input_sha256,
                "humanMessageId": str(user_message["id"]),
                "brainMessageId": str(assistant_message["id"]),
                "traceId": str(trace["id"]),
                "inferenceCount": int(self.counters["inference_count"]),
                "parameterChecksumAfter": str(
                    trace["parameter_checksum_after"]
                ),
                "committedAt": _iso_now(),
            }
            self.completed_chat_turns.append(turn_receipt)
            self.completed_chat_turns = self.completed_chat_turns[
                -COMPLETED_CHAT_TURN_RECEIPTS:
            ]
        try:
            self.save()
        except Exception:
            if turn_receipt is not None:
                self.completed_chat_turns = [
                    value
                    for value in self.completed_chat_turns
                    if value is not turn_receipt
                ]
            raise
        runtime_card = self.runtime_card()
        runtime_card["available_tool_ids"] = trace["available_tool_ids"]
        runtime_card["tool_schema_channel"] = trace["tool_schema_channel"]
        runtime_card["tool_schema_text_injected"] = False
        runtime_card["measured_generation"] = {
            "elapsedMs": trace["generation_elapsed_ms"],
            "generatedTokens": trace["generated_token_count"],
            "tokensPerSecond": trace["generation_tokens_per_second"],
            "scope": trace["generation_measurement_scope"],
            "cacheMode": trace["generation_cache_mode"],
        }
        return {
            "brainId": self.brain_id,
            "text": response,
            "response": response,
            "content": response,
            "humanMessage": user_message,
            "message": assistant_message,
            "trace": trace,
            "metrics": self.metrics(),
            "runtimeCard": runtime_card,
            "availableToolIds": trace["available_tool_ids"],
            "actions": proposed_actions,
            "turnReceipt": (
                dict(turn_receipt) if turn_receipt is not None else None
            ),
            "turnCommitted": True,
            "idempotentCompletion": False,
        }

    @torch.no_grad()
    def _evaluate_experience(
        self, text: str, vsa_vector: torch.Tensor
    ) -> float:
        self.decoder.eval()
        self.memory_bridge.eval()
        self.idea_adapter.eval()
        self.liquid.eval()
        idea = self._idea_model_vector(vsa_vector)
        reconstructed = self.idea_adapter(idea)
        temporal, _ = self.liquid(
            idea, state=self.liquid_state.detach(), elapsed=1.0
        )
        fixed = (
            0.2 * F.mse_loss(reconstructed, idea)
            + 0.05 * F.mse_loss(temporal, idea)
        )
        losses = []
        for ids in self.tokenizer.window_tensors(
            text,
            self.device,
            max_length=min(
                self.config.max_seq_len,
                self._runtime_training_max_seq_len,
            ),
            add_bos=True,
            add_eos=True,
        ):
            if ids.shape[1] < 2:
                continue
            language = self.decoder(
                ids, memory_bias=reconstructed, labels=ids
            )["loss"]
            losses.append(float((language + fixed).item()))
        return sum(losses) / len(losses) if losses else 0.0

    def _restore_core(self, tensors: Mapping[str, torch.Tensor]) -> None:
        _load_prefixed(self.decoder, tensors, "decoder.")
        _load_prefixed(self.memory_bridge, tensors, "memory_bridge.")
        _load_prefixed(self.idea_adapter, tensors, "idea_adapter.")
        _load_prefixed(self.liquid, tensors, "liquid.")
        _load_prefixed(self.modalities, tensors, "modalities.")
        # Core candidate rollback also rolls back every checkpointed row
        # resistance buffer. Discard pending mutation counters from the
        # rejected branch so they cannot later be reported as accepted
        # metaplastic learning.
        for root in self._trainable_modules():
            for module in root.modules():
                if hasattr(module, "_pending_stability_events"):
                    module._pending_stability_events = 0

    def train(
        self,
        texts: Optional[Sequence[str]] = None,
        epochs: int = 1,
        learning_rate: Optional[float] = None,
        source_ids: Optional[Sequence[str]] = None,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be an explicit sequence of strings")
        samples: List[str] = []
        for text in texts or []:
            if not isinstance(text, str):
                raise ValueError("texts must contain only strings")
            clean_text = text.replace("\x00", "")
            if clean_text.strip():
                samples.append(clean_text)
        if isinstance(source_ids, (str, bytes)):
            raise ValueError("source_ids must be an explicit sequence of identifiers")
        selected: List[str] = []
        for value in source_ids or []:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("source_ids must contain non-empty string identifiers")
            identifier = value.strip()
            if identifier not in selected:
                selected.append(identifier)
        if selected:
            sources = {
                str(source.get("id", "")): source
                for source in self.training_sources
                if isinstance(source, Mapping)
            }
            missing = [identifier for identifier in selected if identifier not in sources]
            if missing:
                raise ValueError(
                    "unknown training source_ids: %s" % ", ".join(missing)
                )
            retained = {
                identifier: str(sources[identifier].get("raw_text", "")).replace(
                    "\x00", ""
                )
                for identifier in selected
                if isinstance(sources[identifier].get("raw_text"), str)
            }
            unretained = [
                identifier
                for identifier in selected
                if identifier not in retained or not retained[identifier].strip()
            ]
            if unretained:
                raise ValueError(
                    "training source_ids have no retained text: %s"
                    % ", ".join(unretained)
                )
            samples.extend(retained[identifier] for identifier in selected)
        if not samples:
            raise ValueError(
                "training requires non-empty text or explicit retained source_ids"
            )

        training_plan = self._training_resource_plan()
        if bool(training_plan["pauseBeforeStep"]):
            raise NeuralStateResourcePause(
                "training paused before allocating a step outside the safe Omni RAM envelope",
                {
                    **self.resource_policy.status(),
                    "paused": True,
                    "recoverable": True,
                    "trainingResourcePlan": training_plan,
                },
            )
        self._ensure_optimizer_resident()
        before = self.parameter_checksum()
        training_steps_before = self.counters["training_steps"]
        optimizer_backup = _clone_state_to_cpu(self._optimizer.state_dict())
        backup = {
            key: value.detach().cpu().clone()
            for key, value in self._core_tensors().items()
        }
        stability_backup = self._stability_copy()
        replay_length = len(self.replay)
        baseline_losses = [
            self._evaluate_experience(sample, self.memory.vector_for_text(sample))
            for sample in samples
        ]
        baseline_loss = sum(baseline_losses) / len(baseline_losses)
        losses: List[float] = []
        sequence_tokens = int(training_plan["windowTokens"])
        payload_size = max(1, sequence_tokens - 2)
        windows_per_epoch = sum(
            max(
                1,
                math.ceil(
                    len(sample.encode("utf-8")) / float(payload_size)
                ),
            )
            for sample in samples
        )
        total = epochs * windows_per_epoch
        completed = 0
        optimizer_steps = 0
        optimizer = (
            self._new_optimizer(learning_rate)
            if learning_rate is not None
            else self._optimizer
        )
        if optimizer is self._optimizer:
            self._ensure_optimizer_resident()
        physical_batch = max(
            1, int(training_plan["physicalBatchRecords"])
        )
        accumulation = max(1, int(training_plan["gradientAccumulation"]))
        effective_batch = physical_batch * accumulation
        candidate_id, candidate_dir = self._begin_candidate("slow-training")
        promoted = False
        rejection = ""
        try:
            self.decoder.train()
            self.memory_bridge.train()
            self.idea_adapter.train()
            self.liquid.train()
            for _ in range(epochs):
                def epoch_units():
                    for sample in samples:
                        vector = self.memory.vector_for_text(sample)
                        for window in self.tokenizer.window_tensors(
                            sample,
                            self.device,
                            max_length=sequence_tokens,
                            add_bos=True,
                            add_eos=True,
                        ):
                            if window.shape[1] >= 2:
                                yield window[0], vector

                units = iter(epoch_units())
                while True:
                    group = list(itertools.islice(units, effective_batch))
                    if not group:
                        break
                    next_checkpoint_bytes = int(
                        training_plan.get("scratch", {}).get(
                            "estimatedCheckpointBytes", 0
                        )
                    )
                    self.resource_policy.require_disk(
                        next_checkpoint_bytes,
                        "slow training",
                    )
                    micro_batches = [
                        group[index : index + physical_batch]
                        for index in range(0, len(group), physical_batch)
                    ]
                    if optimizer is self._optimizer:
                        self._ensure_optimizer_resident()
                    optimizer.zero_grad(set_to_none=True)
                    for micro_batch in micro_batches:
                        token_windows = [
                            item[0] for item in micro_batch
                        ]
                        vectors = [item[1] for item in micro_batch]
                        loss, measurements = self._experience_ids_batch_loss(
                            token_windows, vectors
                        )
                        (loss / float(len(micro_batches))).backward()
                        losses.append(measurements["loss"])
                        # The default training admission importance is 1.0,
                        # so every vector in this existing physical microbatch
                        # is already selected. Keep the candidate rollback and
                        # progress/cancellation boundary at the microbatch.
                        self._append_selected_replay_batch(
                            self._idea_model_vector(vector) for vector in vectors
                        )
                        completed += len(micro_batch)
                        if progress is not None:
                            progress(
                                completed / float(total),
                                "Training candidate",
                                {
                                    "resourceReadings": self.resource_policy.status(
                                        estimated_write_bytes=next_checkpoint_bytes
                                    )
                                },
                            )
                    self._accumulate_slow_importance()
                    parameters = [
                        parameter
                        for group_record in optimizer.param_groups
                        for parameter in group_record["params"]
                        if parameter.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.config.grad_clip
                    )
                    optimizer.step()
                    optimizer_steps += 1
                    self.counters["training_steps"] += 1
                    if optimizer is self._optimizer:
                        self._maintain_neural_state_resources()
            final_losses = [
                self._evaluate_experience(
                    sample, self.memory.vector_for_text(sample)
                )
                for sample in samples
            ]
            final_loss = sum(final_losses) / len(final_losses)
            atomic_save_tensors(
                candidate_dir / "core.safetensors",
                self._core_tensors(),
                metadata={"status": "candidate", "brain_id": self.brain_id},
            )
            if not math.isfinite(final_loss):
                rejection = "candidate loss was non-finite"
            elif final_loss > baseline_loss * 1.05 + 1e-6:
                rejection = "candidate regressed validation loss"
            else:
                self._record_candidate(
                    candidate_dir,
                    status="promoting",
                    baselineLoss=baseline_loss,
                    finalLoss=final_loss,
                )
                promoted = True
                self._commit_slow_anchors(rate=1.0)
                self.save()
        except Exception:
            self._restore_core(backup)
            self._restore_stability(stability_backup)
            self.replay.truncate(replay_length)
            self.counters["training_steps"] = training_steps_before
            self._replace_optimizer()
            self._optimizer.load_state_dict(copy.deepcopy(optimizer_backup))
            self._restore_candidate_checkpoint(candidate_dir)
            self._record_candidate(
                candidate_dir,
                status="rejected",
                reason="training exception",
                rejectedAt=_iso_now(),
            )
            raise
        if not promoted:
            self._restore_core(backup)
            self._restore_stability(stability_backup)
            self.replay.truncate(replay_length)
            self.counters["training_steps"] = training_steps_before
            self._replace_optimizer()
            self._optimizer.load_state_dict(copy.deepcopy(optimizer_backup))
            final_loss = baseline_loss
        self._record_candidate(
            candidate_dir,
            status="promoted" if promoted else "rejected",
            reason=rejection,
            baselineLoss=baseline_loss,
            finalLoss=final_loss,
            completedAt=_iso_now(),
        )
        self.events.append(
            "slow-training",
            {
                "epochs": epochs,
                "samples": len(samples),
                "steps": completed,
                "optimizerSteps": optimizer_steps,
                "physicalBatchSize": physical_batch,
                "gradientAccumulation": accumulation,
                "trainingSequenceTokens": sequence_tokens,
                "trainingResourcePlan": training_plan,
                "meanLoss": sum(losses) / len(losses),
                "baselineLoss": baseline_loss,
                "finalLoss": final_loss,
                "parameterChecksumBefore": before,
                "parameterChecksumAfter": self.parameter_checksum(),
                "candidateId": candidate_id,
                "promoted": promoted,
                "rejection": rejection,
            },
        )
        return {
            "brainId": self.brain_id,
            "epochs": epochs,
            "samples": len(samples),
            "steps": completed,
            "optimizerSteps": optimizer_steps,
            "physicalBatchSize": physical_batch,
            "gradientAccumulation": accumulation,
            "trainingSequenceTokens": sequence_tokens,
            "trainingResourcePlan": training_plan,
            "meanLoss": sum(losses) / len(losses),
            "baselineLoss": baseline_loss,
            "finalLoss": final_loss,
            "parameterChecksumBefore": before,
            "parameterChecksumAfter": self.parameter_checksum(),
            "candidateId": candidate_id,
            "promoted": promoted,
            "rejection": rejection,
            "metrics": self.metrics(),
        }

    def _train_evolution_replay_candidate(
        self, steps: int = 4, progress: Optional[Any] = None
    ) -> Dict[str, Any]:
        if not self.config.consolidation_enabled:
            return {
                "brainId": self.brain_id,
                "disabled": True,
                "steps": 0,
                "promoted": False,
                "rejection": "evolution replay is disabled by this configuration",
                "metrics": self.metrics(),
            }
        steps = int(steps)
        if steps < 1:
            raise ValueError("evolution replay steps must be positive")
        before_checksum = self.parameter_checksum()
        if not self.replay:
            raise ValueError("evolution replay requires retained neural replay")

        backup = {
            key: value.detach().cpu().clone()
            for key, value in self._core_tensors().items()
        }
        stability_backup = self._stability_copy()
        optimizer = self._new_optimizer(
            self.config.learning_rate
            * max(0.05, self.config.consolidation_rate / 0.06)
        )
        training_steps_before = self.counters["training_steps"]
        validation = torch.stack(self.replay[: min(32, len(self.replay))]).to(
            self.device
        )
        with torch.no_grad():
            baseline_loss = float(
                F.mse_loss(self.idea_adapter(validation), validation).item()
            )
        losses: List[float] = []
        self.idea_adapter.train()
        candidate_id, candidate_dir = self._begin_candidate("evolution-replay")
        promoted = False
        rejection = ""
        organic = self._organic_state()
        rehearsal_noise = max(
            0.01,
            min(
                0.12,
                0.015
                + 0.045 * float(organic["predictionError"])
                + 0.030 * float(organic["uncertainty"])
                + 0.020 * float(organic["novelty"])
                + 0.010 * float(organic["tension"]),
            ),
        )
        try:
            for index in range(steps):
                self.resource_policy.require_disk(0, "evolution replay")
                target = self.replay[index % len(self.replay)].to(self.device).reshape(1, -1)
                optimizer.zero_grad(set_to_none=True)
                noisy = target + torch.randn_like(target) * rehearsal_noise
                prediction = self.idea_adapter(noisy)
                reconstruction_loss = F.mse_loss(prediction, target.detach())
                stability_loss = self._stability_penalty(
                    self.idea_adapter.parameters()
                )
                loss = reconstruction_loss + stability_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite evolution replay loss")
                loss.backward()
                self._accumulate_slow_importance(
                    self.idea_adapter.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    self.idea_adapter.parameters(), self.config.grad_clip
                )
                optimizer.step()
                losses.append(float(loss.detach().item()))
                self.counters["training_steps"] += 1
                if progress is not None:
                    progress(
                        (index + 1) / float(steps),
                        "Evaluating internal replay candidate",
                        {"resourceReadings": self._resource_readings()},
                    )
            self.idea_adapter.eval()
            with torch.no_grad():
                final_loss = float(
                    F.mse_loss(self.idea_adapter(validation), validation).item()
                )
            atomic_save_tensors(
                candidate_dir / "core.safetensors",
                self._core_tensors(),
                metadata={"status": "candidate", "brain_id": self.brain_id},
            )
            if not math.isfinite(final_loss):
                rejection = "candidate loss was non-finite"
            elif final_loss > baseline_loss * 1.05 + 1e-6:
                rejection = "candidate regressed validation loss"
            else:
                self._record_candidate(
                    candidate_dir,
                    status="promoting",
                    baselineLoss=baseline_loss,
                    finalLoss=final_loss,
                )
                promoted = True
                self._commit_slow_anchors(rate=1.0)
                self.memory.decay(self.config.forgetting_rate)
                self.router.synapses.decay_unused(
                    self.config.forgetting_rate * 0.5
                )
                self.counters["consolidation_cycles"] += 1
                self.counters["plasticity_events"] = int(
                    self.router.synapses.plasticity_events.item()
                )
                self.save()
        except Exception:
            self._restore_core(backup)
            self._restore_stability(stability_backup)
            self.counters["training_steps"] = training_steps_before
            self._replace_optimizer()
            self._restore_candidate_checkpoint(candidate_dir)
            self._record_candidate(
                candidate_dir,
                status="rejected",
                reason="evolution replay exception",
                rejectedAt=_iso_now(),
            )
            raise
        if not promoted:
            self._restore_core(backup)
            self._restore_stability(stability_backup)
            self.counters["training_steps"] = training_steps_before
            self._replace_optimizer()
            final_loss = baseline_loss
        self._record_candidate(
            candidate_dir,
            status="promoted" if promoted else "rejected",
            reason=rejection,
            baselineLoss=baseline_loss,
            finalLoss=final_loss,
            completedAt=_iso_now(),
        )
        self.events.append(
            "evolution-replay",
            {
                "steps": steps,
                "replayExamples": len(self.replay),
                "meanLoss": sum(losses) / len(losses),
                "baselineLoss": baseline_loss,
                "finalLoss": final_loss,
                "parameterChecksumBefore": before_checksum,
                "parameterChecksumAfter": self.parameter_checksum(),
                "candidateId": candidate_id,
                "promoted": promoted,
                "rejection": rejection,
            },
        )
        return {
            "brainId": self.brain_id,
            "steps": steps,
            "meanLoss": sum(losses) / len(losses),
            "baselineLoss": baseline_loss,
            "finalLoss": final_loss,
            "parameterChecksumBefore": before_checksum,
            "parameterChecksumAfter": self.parameter_checksum(),
            "candidateId": candidate_id,
            "promoted": promoted,
            "rejection": rejection,
            "replayExamples": len(self.replay),
            "metrics": self.metrics(),
        }

    @staticmethod
    def _read_document(path: Path, kind: str = "") -> Tuple[str, bytes, str]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        data = path.read_bytes()
        extension = path.suffix.lower()
        resolved_kind = kind or (
            "pdf"
            if extension == ".pdf"
            else "text"
            if extension
            in {
                ".txt",
                ".md",
                ".markdown",
                ".json",
                ".jsonl",
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".rs",
                ".go",
                ".java",
                ".c",
                ".cc",
                ".cpp",
                ".h",
                ".hpp",
            }
            else "binary"
        )
        if resolved_kind == "pdf":
            try:
                from pypdf import PdfReader
            except ImportError as error:
                raise RuntimeError(
                    "PDF ingestion requires pypdf; install engine/requirements.txt"
                ) from error
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif resolved_kind in {"text", "markdown", "code", "json", "unknown"}:
            text = data.decode("utf-8", errors="replace")
            if text and text.count("\ufffd") / len(text) > 0.02:
                raise ValueError("file is not valid enough UTF-8 text")
        else:
            text = ""
        return text.replace("\x00", ""), data, resolved_kind

    @staticmethod
    def _experience_chunks(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        chunks: List[str] = []
        pending = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            # A single long sentence or source-code line must not be silently
            # truncated. Emit all of it in bounded neural experiences.
            while len(piece) > 4000:
                if pending:
                    chunks.append(pending)
                    pending = ""
                chunks.append(piece[:4000])
                piece = piece[4000:]
            if pending and len(pending) + len(piece) + 1 <= 900:
                pending += " " + piece
            else:
                if pending:
                    chunks.append(pending)
                pending = piece
        if pending:
            chunks.append(pending)
        return chunks

    @staticmethod
    def _reading_chunk_importances(chunks: Sequence[str]) -> List[float]:
        """Derive selective attention weights without skipping any chunk.

        Dataset coverage still visits and trains every valid record.  These
        weights only control how strongly each section enters transient and
        lasting neural memory, analogous to a reader attending differently to
        a heading, a recurring idea, and repetitive filler.
        """

        if not chunks:
            return []
        token_rows = [
            re.findall(r"[A-Za-z0-9_'-]{2,}", chunk.casefold())
            for chunk in chunks
        ]
        document_frequency: Dict[str, int] = {}
        for tokens in token_rows:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        count = max(1, len(chunks))
        values: List[float] = []
        for chunk, tokens in zip(chunks, token_rows):
            unique_ratio = len(set(tokens)) / float(max(1, len(tokens)))
            recurrence = (
                sum(document_frequency[token] / float(count) for token in set(tokens))
                / float(max(1, len(set(tokens))))
            )
            structural = float(
                bool(
                    re.search(
                        r"(^|\n)\s*(?:#{1,6}\s|[A-Z][A-Z0-9 _-]{3,}:|"
                        r"(?:def|class|function|interface|chapter|section)\b)",
                        chunk,
                        re.MULTILINE,
                    )
                )
            )
            length_signal = min(1.0, len(tokens) / 160.0)
            value = (
                0.30
                + 0.09 * unique_ratio
                + 0.08 * recurrence
                + 0.07 * structural
                + 0.04 * length_signal
            )
            values.append(max(0.30, min(0.58, value)))
        return values

    def _integrate_reading_record(
        self,
        text: str,
        *,
        source_name: str,
        child_assembly_ids: Sequence[str],
        child_weights: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        """Bind all visited sections into one whole-record neural assembly."""

        aligned = [
            (assembly_id, float(weight))
            for assembly_id, weight in zip(child_assembly_ids, child_weights)
            if assembly_id in self.memory.assembly_vectors
        ]
        if not aligned:
            return None
        vectors = [self.memory.assembly_vectors[item[0]] for item in aligned]
        weights = [max(0.05, item[1]) for item in aligned]
        whole = self.memory.space.weighted_bundle(vectors, weights)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        admitted = self._admit_sensory_embedding(
            whole,
            kind="document",
            source_name=source_name,
            fingerprint="record:" + fingerprint,
            child_ids=[item[0] for item in aligned],
            importance=0.90,
            substrate_space=True,
            experience_source="reading",
            settling_salience=0.92,
            settling_prediction_error=0.85,
        )
        admitted.pop("vector", None)
        admitted["sectionAssemblyIds"] = [item[0] for item in aligned]
        admitted["sectionWeights"] = weights
        admitted["wholeRecord"] = True
        admitted["rawSourceTextStored"] = False
        return admitted

    def _media_idea(self, source_name: str) -> torch.Tensor:
        vector = self.memory.vector_for_text(source_name or "media experience")
        return self._idea_model_vector(vector).detach()

    def _live_image_tensor(self, payload: bytes) -> torch.Tensor:
        try:
            from PIL import Image
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "live image observation requires Pillow and NumPy"
            ) from error
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                image = opened.convert("RGB").resize(
                    (self.config.image_size, self.config.image_size)
                )
                values = np.asarray(image, dtype="float32").copy()
        except (OSError, ValueError) as error:
            raise ValueError("live image packet could not be decoded") from error
        return (
            torch.from_numpy(values)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 127.5
            - 1.0
        )

    def _live_visual_embedding(
        self,
        payload: bytes,
        modality: str,
        settings: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode global shape and resource-scaled source-detail tiles.

        A native/high-resolution frame is not reduced to one tiny square. The
        global view travels through the selected modality encoder, while
        uniformly distributed source tiles travel through the same brain's
        ternary image encoder and are VSA-bound to their positions. Raw pixels
        remain transient.
        """

        try:
            from PIL import Image
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "live multiresolution perception requires Pillow and NumPy"
            ) from error
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                source = opened.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError("live image packet could not be decoded") from error
        source_width, source_height = source.size
        declared_width = int(settings.get("width", source_width))
        declared_height = int(settings.get("height", source_height))
        if declared_width != source_width or declared_height != source_height:
            raise ValueError("capture dimensions do not match the decoded frame")
        target_size = int(self.config.image_size)

        def as_tensor(image: Any) -> torch.Tensor:
            resized = image.resize((target_size, target_size))
            values = np.asarray(resized, dtype="float32").copy()
            return (
                torch.from_numpy(values)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
                / 127.5
                - 1.0
            )

        global_tensor = as_tensor(source)
        if modality == "video":
            temporal = global_tensor.unsqueeze(2).expand(
                -1, -1, self.config.video_frames, -1, -1
            ).contiguous()
            global_embedding = self.modalities.perception_embedding(
                "video", temporal
            )
        else:
            global_embedding = self.modalities.perception_embedding(
                "image", global_tensor
            )

        columns = max(1, math.ceil(source_width / target_size))
        rows = max(1, math.ceil(source_height / target_size))
        tiles_available = rows * columns
        tile_tensor_bytes = max(1, 3 * target_size * target_size * 4)
        readings = self._resource_readings()
        available_memory = readings.get("availableMemoryBytes")
        memory_tile_budget = (
            max(1, int(available_memory) // (tile_tensor_bytes * 8))
            if isinstance(available_memory, int) and available_memory > 0
            else max(1, int(self.config.working_memory_slots))
        )
        tile_budget = min(
            tiles_available,
            max(1, int(self.config.working_memory_slots)),
            memory_tile_budget,
        )
        if tile_budget == 1:
            selected_indices = [0]
        elif tile_budget == tiles_available:
            selected_indices = list(range(tiles_available))
        else:
            selected_indices = sorted(
                {
                    round(
                        index
                        * (tiles_available - 1)
                        / float(tile_budget - 1)
                    )
                    for index in range(tile_budget)
                }
            )
        bound_tiles: List[torch.Tensor] = []
        for linear_index in selected_indices:
            row = linear_index // columns
            column = linear_index % columns
            left = column * target_size
            top = row * target_size
            right = min(source_width, left + target_size)
            bottom = min(source_height, top + target_size)
            tile_tensor = as_tensor(source.crop((left, top, right, bottom)))
            tile_embedding = self.modalities.image.encode(tile_tensor)
            position_symbol = self.memory.space.symbol(
                "visual-tile:%d:%d:%d:%d"
                % (row, column, rows, columns)
            )
            position = self._idea_model_vector(position_symbol).detach()
            ternary_position = torch.sign(position)
            bound_tiles.append(
                F.normalize(
                    tile_embedding * ternary_position + 0.10 * position,
                    dim=-1,
                )
            )
        tile_field = torch.stack(bound_tiles, dim=0).mean(dim=0)
        embedding = F.normalize(
            0.40 * global_embedding + 0.60 * tile_field,
            dim=-1,
        )
        return embedding, {
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "globalDecodedWidth": target_size,
            "globalDecodedHeight": target_size,
            "tilesAvailable": tiles_available,
            "tilesEncoded": len(selected_indices),
            "tileCoverage": len(selected_indices) / float(tiles_available),
            "tileInputSize": target_size,
            "tileGridRows": rows,
            "tileGridColumns": columns,
            "tileSelection": "uniform-resource-scaled",
            "tileBinding": "ternary-position-vsa",
            "spatialTileEncoder": "same-brain-ternary-image-pack",
            "temporalFrames": (
                int(self.config.video_frames) if modality == "video" else 1
            ),
            "rawTilesStored": False,
        }

    def _live_audio_tensor(
        self,
        payload: bytes,
        mime_type: str,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        channels = max(1, min(32, int(settings.get("channels", 1))))
        if mime_type in {"audio/pcm-f32le", "audio/x-pcm-f32le"}:
            if not payload or len(payload) % 4:
                raise ValueError("float PCM packet has an invalid byte length")
            samples = array.array("f")
            samples.frombytes(payload)
            if os.sys.byteorder != "little":
                samples.byteswap()
            values = torch.tensor(samples, dtype=torch.float32)
        elif mime_type in {"audio/pcm-s16le", "audio/x-pcm-s16le"}:
            if not payload or len(payload) % 2:
                raise ValueError("16-bit PCM packet has an invalid byte length")
            samples = array.array("h")
            samples.frombytes(payload)
            if os.sys.byteorder != "little":
                samples.byteswap()
            values = torch.tensor(samples, dtype=torch.float32).div_(32768.0)
        elif mime_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
            try:
                with wave.open(io.BytesIO(payload), "rb") as handle:
                    channels = max(1, int(handle.getnchannels()))
                    values = self._pcm_mono_values(
                        handle.readframes(handle.getnframes()),
                        int(handle.getsampwidth()),
                        channels,
                    )
                    channels = 1
            except (wave.Error, EOFError, ValueError) as error:
                raise ValueError("live WAV packet could not be decoded") from error
        elif mime_type in {
            "audio/webm",
            "audio/ogg",
            "audio/opus",
            "audio/mp4",
            "audio/aac",
        }:
            try:
                import imageio_ffmpeg

                decoded = subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-i",
                        "pipe:0",
                        "-t",
                        "10",
                        "-f",
                        "f32le",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "pipe:1",
                    ],
                    input=payload,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=15,
                ).stdout
                if not decoded or len(decoded) % 4:
                    raise ValueError("decoded live audio has invalid PCM length")
                samples = array.array("f")
                samples.frombytes(decoded)
                if os.sys.byteorder != "little":
                    samples.byteswap()
                values = torch.tensor(samples, dtype=torch.float32)
                channels = 1
            except (
                ImportError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                raise ValueError(
                    "live encoded audio packet needs a configured external FFmpeg decoder"
                ) from error
        else:
            raise ValueError("unsupported live audio packet encoding")
        if channels > 1:
            usable = values.numel() - values.numel() % channels
            if usable <= 0:
                raise ValueError("live audio packet contains no complete frame")
            values = values[:usable].reshape(-1, channels).mean(dim=1)
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("live audio packet contains no finite samples")
        return self._bounded_audio_target(values.clamp(-1.0, 1.0))[0]

    def observe_live_packet(
        self,
        *,
        modality: str,
        mime_type: str,
        payload: bytes,
        session_id: str,
        sequence: int,
        timestamp_ms: float,
        retention: str = "neural",
        settings: Optional[Mapping[str, Any]] = None,
        permission_source: str = "device",
    ) -> Dict[str, Any]:
        """Admit one bounded live perception without retaining its raw bytes.

        ``working`` updates only recurrent/working activity. ``neural`` also
        forms a persistent assembly, applies STDP, and trains the same ternary
        modality route. Dataset cursors are deliberately untouched: capture
        backpressure is not reported as completed corpus learning.
        """

        if modality not in IMAGINATION_MODALITIES:
            raise ValueError("live modality must be image, audio, or video")
        if retention not in {"working", "neural"}:
            raise ValueError("live retention must be working or neural")
        if not payload:
            raise ValueError("live observation packet is empty")
        normalized_mime = mime_type.strip().lower().split(";", 1)[0]
        packet_settings = dict(settings or {})
        visual_diagnostics: Dict[str, Any] = {}
        if modality in {"image", "video"}:
            if normalized_mime not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                raise ValueError(
                    "live image/video packets must be JPEG, PNG, or WebP frames"
                )
            with torch.no_grad():
                embedding, visual_diagnostics = self._live_visual_embedding(
                    payload, modality, packet_settings
                )
        else:
            tensor = self._live_audio_tensor(
                payload, normalized_mime, packet_settings
            )
            with torch.no_grad():
                embedding = self.modalities.perception_embedding(
                    modality, tensor
                )
        content_hash = hashlib.sha256(payload).hexdigest()
        source_name = "live %s %s" % (
            permission_source.replace("\x00", " ").strip()[:64] or "device",
            modality,
        )
        if retention == "neural":
            admitted = self._admit_sensory_embedding(
                embedding,
                kind="%s-perception" % modality,
                source_name=source_name,
                fingerprint="live:%s:%s" % (modality, content_hash),
                importance=0.62,
                experience_source="live-observation",
            )
            selector_parameters = list(
                self.modalities.imagination_selector.parameters()
            )
            selector_optimizer = adamw_for_remaining_parameters(
                selector_parameters,
                lr=max(1e-5, min(0.01, self.config.learning_rate)),
                weight_decay=1e-5,
            )
            selector_optimizer.zero_grad(set_to_none=True)
            target = torch.full(
                (embedding.shape[0],),
                IMAGINATION_MODALITIES.index(modality),
                dtype=torch.long,
                device=self.device,
            )
            route_loss = F.cross_entropy(
                self.modalities.imagination_logits(embedding.detach()), target
            )
            route_loss.backward()
            self._accumulate_slow_importance(selector_parameters)
            torch.nn.utils.clip_grad_norm_(
                selector_parameters, self.config.grad_clip
            )
            selector_optimizer.step()
            self._commit_slow_anchors(
                rate=0.04, parameters=selector_parameters
            )
            self.counters["training_steps"] += 1
            assembly_id: Optional[str] = str(admitted["assemblyId"])
            spike_rate = float(admitted["spikeRate"])
            novelty = float(admitted["novelty"])
            route_loss_value = float(route_loss.detach().item())
        else:
            substrate = self._sensory_substrate_vector(
                embedding, modality, content_hash
            )
            idea = self._idea_model_vector(substrate)
            self.liquid_state, controls = self.liquid(
                idea, state=self.liquid_state.detach(), elapsed=1.0
            )
            self.liquid_state = self.liquid_state.detach()
            routed, spike_metrics = self.router.route(
                idea,
                steps=max(
                    2,
                    int(
                        round(
                            float(controls["ponder_scale"].mean().item())
                        )
                    ),
                ),
                learn=False,
                threshold_offset=float(
                    controls["threshold_offset"].detach().mean().item()
                ),
            )
            self._append_working_memory(
                routed,
                source="live-%s" % modality,
                salience=0.58,
            )
            assembly_id = None
            spike_rate = float(spike_metrics["spike_rate"])
            novelty = 0.0
            route_loss_value = 0.0
        self.current_context = {
            **self.current_context,
            "sensorySlots": min(
                self.config.working_memory_slots,
                max(0, int(self.current_context.get("sensorySlots", 0))) + 1,
            ),
            "updatedAt": _iso_now(),
        }
        result = {
            "brainId": self.brain_id,
            "sessionId": session_id,
            "sequence": int(sequence),
            "timestampMs": float(timestamp_ms),
            "modality": modality,
            "retention": retention,
            "assemblyId": assembly_id,
            "spikeRate": spike_rate,
            "novelty": novelty,
            "routeLoss": route_loss_value,
            "packetSha256": content_hash,
            "packetBytes": len(payload),
            "rawPacketStored": False,
            "datasetCoverageCommitted": False,
            "sameBrainSharedIdeaSpace": True,
            "hiddenBehavioralPrompt": False,
            "perception": visual_diagnostics,
            "observationControlId": str(
                packet_settings.get("observationControlId", "")
            )[:128]
            or None,
            "resolutionMode": (
                str(packet_settings.get("resolutionMode"))
                if packet_settings.get("resolutionMode")
                in {"native", "current", "custom"}
                else None
            ),
            "burstIndex": (
                int(packet_settings["burstIndex"])
                if "burstIndex" in packet_settings
                else None
            ),
            "burstCount": (
                int(packet_settings["burstCount"])
                if "burstCount" in packet_settings
                else None
            ),
        }
        self.events.append(
            "live-observation",
            {
                key: value
                for key, value in result.items()
                if key not in {"brainId"}
            },
        )
        return result

    def _attention_epoch(self) -> int:
        boundary = self.fresh_attention_boundary
        return int(boundary.get("epoch", 0)) if boundary is not None else 0


    def record_live_observation_control(
        self, *, session_id: str, control: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist an operational device-control result without prompt text."""

        identifier = str(control.get("id", "")).strip()
        kind = str(control.get("kind", "")).strip().lower()
        state = str(control.get("state", "")).strip().lower()
        if (
            not session_id
            or not identifier
            or kind not in {"configure", "snapshot"}
            or state
            not in {"requested", "applied", "rejected", "cancelled", "reverted"}
        ):
            raise ValueError("live observation control is invalid")
        if control.get("sessionId") != session_id:
            raise ValueError("live observation control session does not match")
        serialized = json.dumps(control, sort_keys=True, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > 64 * 1024:
            raise ValueError("live observation control evidence is too large")
        lowered_keys = {str(key).lower() for key in control}
        if lowered_keys.intersection({"data", "bytes", "database64", "rawpacket"}):
            raise ValueError("raw capture bytes cannot enter control evidence")
        evidence = json.loads(serialized)
        self.events.append(
            "live-observation-control",
            {
                "sessionId": session_id,
                "control": evidence,
                "hiddenBehavioralPrompt": False,
                "rawPacketStored": False,
                "datasetCoverageCommitted": False,
            },
        )
        return {
            "brainId": self.brain_id,
            "sessionId": session_id,
            "controlId": identifier,
            "state": state,
            "recorded": True,
            "hiddenBehavioralPrompt": False,
            "rawPacketStored": False,
        }

    def _sensory_substrate_vector(
        self, embedding: torch.Tensor, kind: str, fingerprint: str
    ) -> torch.Tensor:
        """Lift a learned modality embedding through exact ternary synapses.

        The modality encoders live in the shared idea space, while persistent
        assemblies live in the wider VSA substrate.  Reusing the transpose of
        the cortex's mandatory ternary memory bridge keeps this a neural
        projection instead of converting a perception into synthetic text.
        """

        value = embedding.detach().to(self.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[-1] != self.config.idea_dim:
            raise ValueError("sensory embedding dimensions do not match idea space")
        ternary = self.memory_bridge.effective_weight().to(
            self.device, dtype=torch.float32
        )
        lifted = value.mean(dim=0, keepdim=True) @ ternary
        lifted = lifted[0].detach().cpu()
        if not bool(torch.isfinite(lifted).all()) or float(lifted.norm()) <= 1e-8:
            raise RuntimeError(
                "sensory neural projection produced no finite signal; "
                "the input was not admitted as a learned assembly"
            )
        return F.normalize(lifted.reshape(1, -1), dim=-1)[0]

    def _admit_sensory_embedding(
        self,
        embedding: torch.Tensor,
        *,
        kind: str,
        source_name: str,
        fingerprint: str,
        child_ids: Optional[Sequence[str]] = None,
        importance: float = 0.72,
        substrate_space: bool = False,
        experience_source: str = "media",
        settling_salience: Optional[float] = None,
        settling_prediction_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a persistent sensory assembly and activate brain dynamics."""

        if substrate_space:
            substrate_vector = embedding.detach().cpu().float().reshape(-1)
            if substrate_vector.numel() != self.config.vsa_dim:
                raise ValueError(
                    "sensory substrate vector dimensions do not match VSA space"
                )
            substrate_vector = F.normalize(
                substrate_vector.reshape(1, -1), dim=-1
            )[0]
        else:
            substrate_vector = self._sensory_substrate_vector(
                embedding, kind, fingerprint
            )
        learned = self.memory.learn_vector(
            substrate_vector,
            fingerprint=fingerprint,
            kind=kind,
            source=experience_source,
            source_label=source_name,
            importance=importance,
            child_ids=child_ids,
        )
        idea = self._idea_model_vector(learned["vector"])
        self.liquid_state, controls = self.liquid(
            idea, state=self.liquid_state.detach(), elapsed=1.0
        )
        self.liquid_state = self.liquid_state.detach()
        routed, spike_metrics = self.router.route(
            idea,
            steps=max(
                2, int(round(float(controls["ponder_scale"].mean().item())))
            ),
            learn=True,
            threshold_offset=float(
                controls["threshold_offset"].detach().mean().item()
            ),
        )
        workspace_salience = max(
            0.0,
            min(
                1.0,
                0.45 * float(learned["novelty"])
                + 0.35 * importance
                + 0.20 * float(spike_metrics["spike_rate"]),
            ),
        )
        if settling_salience is not None:
            workspace_salience = max(
                0.0, min(1.0, float(settling_salience))
            )
        prediction_error = (
            float(learned["novelty"])
            if settling_prediction_error is None
            else float(settling_prediction_error)
        )
        self._append_replay(
            routed,
            importance=importance,
            replay_priority=self._organic_replay_priority(
                assembly_id=str(learned["assembly_id"]),
                salience=workspace_salience,
                novelty=float(learned["novelty"]),
                prediction_error=prediction_error,
                spike_rate=float(spike_metrics["spike_rate"]),
            ),
            assembly_id=str(learned["assembly_id"]),
        )
        self._append_working_memory(
            routed,
            assembly_id=str(learned["assembly_id"]),
            source=(
                "media-%s" % kind
                if experience_source == "media"
                else experience_source
            ),
            salience=workspace_salience,
        )
        memory_settling = self._settle_memory_automatically(
            routed,
            assembly_id=str(learned["assembly_id"]),
            source=(
                "media-%s" % kind
                if experience_source == "media"
                else experience_source
            ),
            salience=workspace_salience,
            novelty=float(learned["novelty"]),
            prediction_error=prediction_error,
            importance=importance,
            spike_rate=float(spike_metrics["spike_rate"]),
        )
        self.counters["experiences"] += 1
        self.counters["plasticity_events"] = int(
            self.router.synapses.plasticity_events.item()
        )
        return {
            "assemblyId": str(learned["assembly_id"]),
            "neuronIds": list(learned["neuron_ids"]),
            "novelty": float(learned["novelty"]),
            "spikeRate": float(spike_metrics["spike_rate"]),
            "forwardProjection": "exact-ternary-memory-bridge-transpose",
            "rawSourceTextInjected": False,
            "memorySettling": memory_settling,
            "vector": substrate_vector,
        }

    @staticmethod
    def _effective_media_kind(path: str, requested_kind: str) -> str:
        """Route animated GIF experiences through temporal video learning."""

        if requested_kind != "image" or Path(path).suffix.lower() != ".gif":
            return requested_kind
        try:
            from PIL import Image

            with Image.open(path) as opened:
                if bool(getattr(opened, "is_animated", False)) and int(
                    getattr(opened, "n_frames", 1)
                ) > 1:
                    return "video"
        except (ImportError, OSError, ValueError):
            # The normal image decoder will produce the attributable error.
            return requested_kind
        return requested_kind

    def _decode_image(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image
            import numpy as np
        except ImportError as error:
            raise RuntimeError("image training requires Pillow and NumPy") from error
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize(
                (self.config.image_size, self.config.image_size)
            )
            values = np.asarray(image, dtype="float32").copy()
        return (
            torch.from_numpy(values)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 127.5
            - 1.0
        )

    @staticmethod
    def _pcm_mono_values(
        raw: bytes, sample_width: int, channels: int
    ) -> torch.Tensor:
        """Decode one bounded PCM block without retaining the source stream."""

        if sample_width == 1:
            values = torch.tensor(list(raw), dtype=torch.float32)
            values.sub_(128.0).div_(128.0)
        elif sample_width == 2:
            samples = array.array("h")
            samples.frombytes(raw)
            if os.sys.byteorder != "little":
                samples.byteswap()
            values = torch.tensor(samples, dtype=torch.float32).div_(32768.0)
        elif sample_width == 4:
            samples = array.array("i")
            samples.frombytes(raw)
            if os.sys.byteorder != "little":
                samples.byteswap()
            values = torch.tensor(samples, dtype=torch.float32).div_(2147483648.0)
        else:
            raise ValueError("only 8, 16, or 32-bit PCM WAV is supported")
        if channels > 1:
            usable = values.numel() - values.numel() % channels
            values = values[:usable].reshape(-1, channels).mean(dim=1)
        return values

    def _bounded_audio_target(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Fit one source block to the codec while preserving its true length."""

        values = values.detach().float().flatten()
        actual_samples = min(values.numel(), self.config.audio_samples)
        values = values[: self.config.audio_samples]
        if values.numel() < self.config.audio_samples:
            values = F.pad(values, (0, self.config.audio_samples - values.numel()))
        return values.reshape(1, 1, -1).to(self.device), actual_samples

    def _iter_audio_windows(
        self, path: str
    ) -> Iterator[Tuple[torch.Tensor, int]]:
        """Yield every audio sample exactly once in bounded codec windows."""

        window_samples = self.config.audio_samples
        suffix = Path(path).suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(path, "rb") as handle:
                    channels = handle.getnchannels()
                    width = handle.getsampwidth()
                    while True:
                        raw = handle.readframes(window_samples)
                        if not raw:
                            break
                        values = self._pcm_mono_values(raw, width, channels)
                        if values.numel():
                            yield self._bounded_audio_target(values)
                return
            except (wave.Error, EOFError):
                # Some containers use a WAV suffix without PCM payload. The
                # streaming SoundFile/FFmpeg fallbacks below can decode them.
                pass

        try:
            import soundfile

            yielded = False
            with soundfile.SoundFile(path, mode="r") as handle:
                while True:
                    decoded = handle.read(
                        frames=window_samples,
                        dtype="float32",
                        always_2d=True,
                    )
                    if decoded.shape[0] == 0:
                        break
                    yielded = True
                    values = torch.from_numpy(decoded).float().mean(dim=1)
                    yield self._bounded_audio_target(values)
            if yielded:
                return
        except (ImportError, OSError, RuntimeError):
            pass

        process: Optional[subprocess.Popen[bytes]] = None
        try:
            import imageio_ffmpeg

            process = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-i",
                    str(Path(path).resolve()),
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if process.stdout is None:
                raise RuntimeError("FFmpeg audio decoder did not open a stream")
            bytes_per_window = window_samples * 4
            pending = bytearray()
            yielded = False
            while True:
                block = process.stdout.read(bytes_per_window - len(pending))
                if block:
                    pending.extend(block)
                if len(pending) == bytes_per_window or (not block and pending):
                    usable = len(pending) - len(pending) % 4
                    samples = array.array("f")
                    samples.frombytes(bytes(pending[:usable]))
                    if os.sys.byteorder != "little":
                        samples.byteswap()
                    values = torch.tensor(samples, dtype=torch.float32)
                    if values.numel():
                        yielded = True
                        yield self._bounded_audio_target(values)
                    pending.clear()
                if not block:
                    break
            return_code = process.wait()
            if return_code != 0 or not yielded:
                raise RuntimeError("FFmpeg could not decode the audio stream")
        except (ImportError, OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "audio format needs soundfile or a configured external FFmpeg decoder"
            ) from error
        finally:
            if process is not None:
                if process.stdout is not None:
                    process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def _decode_wav(self, path: str) -> torch.Tensor:
        """Compatibility helper returning the first bounded audio window."""

        windows = self._iter_audio_windows(path)
        try:
            target, _actual_samples = next(windows)
            return target
        except StopIteration as error:
            raise ValueError("audio contained no decodable samples") from error
        finally:
            windows.close()

    def _video_window_tensor(
        self, frames: Sequence[Any]
    ) -> Tuple[torch.Tensor, int]:
        actual_frames = len(frames)
        if actual_frames == 0:
            raise ValueError("video window contained no frames")
        padded = list(frames)
        while len(padded) < self.config.video_frames:
            padded.append(padded[-1])
        tensor = torch.stack(
            [torch.from_numpy(frame).permute(2, 0, 1) for frame in padded],
            dim=1,
        )
        return tensor.unsqueeze(0).to(self.device) / 127.5 - 1.0, actual_frames

    def _iter_video_windows(
        self, path: str
    ) -> Iterator[Tuple[torch.Tensor, int]]:
        """Stream every decoded frame through bounded temporal windows."""

        try:
            from PIL import Image, ImageSequence
            import numpy as np
        except ImportError as error:
            raise RuntimeError("video training requires Pillow and NumPy") from error

        def normalized_frame(frame: Any) -> Any:
            resized = frame.convert("RGB").resize(
                (self.config.image_size, self.config.image_size)
            )
            return np.asarray(resized, dtype="float32").copy()

        pending: List[Any] = []
        suffix = Path(path).suffix.lower()
        if suffix in {".gif", ".webp"}:
            try:
                with Image.open(path) as opened:
                    for frame in ImageSequence.Iterator(opened):
                        pending.append(normalized_frame(frame))
                        if len(pending) == self.config.video_frames:
                            yield self._video_window_tensor(pending)
                            pending.clear()
            except (OSError, ValueError) as error:
                raise RuntimeError("Pillow could not decode the animated image") from error
        else:
            process: Optional[subprocess.Popen[bytes]] = None
            try:
                import imageio_ffmpeg

                process = subprocess.Popen(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-i",
                        str(Path(path).resolve()),
                        "-an",
                        "-sn",
                        "-vf",
                        "scale=%d:%d"
                        % (self.config.image_size, self.config.image_size),
                        "-vsync",
                        "0",
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                if process.stdout is None:
                    raise RuntimeError("FFmpeg video decoder did not open a stream")
                frame_bytes = self.config.image_size * self.config.image_size * 3
                raw_pending = bytearray()
                decoded_frames = 0
                while True:
                    block = process.stdout.read(frame_bytes - len(raw_pending))
                    if block:
                        raw_pending.extend(block)
                    if len(raw_pending) == frame_bytes:
                        frame = np.frombuffer(
                            bytes(raw_pending), dtype="uint8"
                        ).reshape(
                            self.config.image_size,
                            self.config.image_size,
                            3,
                        )
                        pending.append(frame.astype("float32", copy=True))
                        decoded_frames += 1
                        raw_pending.clear()
                        if len(pending) == self.config.video_frames:
                            yield self._video_window_tensor(pending)
                            pending.clear()
                    if not block:
                        break
                if raw_pending:
                    raise RuntimeError("FFmpeg returned an incomplete video frame")
                return_code = process.wait()
                if return_code != 0 or decoded_frames == 0:
                    raise RuntimeError("FFmpeg could not decode the video stream")
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "MP4/WebM/MOV video needs a configured external FFmpeg decoder; GIF works with Pillow"
                ) from error
            finally:
                if process is not None:
                    if process.stdout is not None:
                        process.stdout.close()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        if pending:
            yield self._video_window_tensor(pending)

    def _decode_video(self, path: str) -> torch.Tensor:
        """Compatibility helper returning the first bounded video window."""

        windows = self._iter_video_windows(path)
        try:
            target, _actual_frames = next(windows)
            return target
        except StopIteration as error:
            raise ValueError("video contained no decodable frames") from error
        finally:
            windows.close()

    def _train_media(
        self,
        path: str,
        kind: str,
        source_name: str,
        steps: int = 2,
        progress: Optional[Any] = None,
        content_sha256: str = "",
        _include_embedded_audio: bool = True,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        idea = self._media_idea(source_name)
        window_size = 1
        unit = "images"
        if kind == "image":
            if not (self.config.image_enabled or self.config.vision_enabled):
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Image and vision packs are disabled."],
                    "coverage": {
                        "unit": "images",
                        "windowSize": 1,
                        "windows": 0,
                        "discoveredUnits": 0,
                        "processedUnits": 0,
                        "tailUnits": 0,
                        "complete": False,
                    },
                }
            target = self._decode_image(path)
            window_stream: Iterable[Tuple[torch.Tensor, int]] = [(target, 1)]
            parameters = []
            if self.config.image_enabled:
                parameters.extend(self.modalities.image.parameters())
            if self.config.vision_enabled:
                parameters.extend(self.modalities.vision.parameters())
        elif kind == "audio":
            unit = "samples"
            window_size = self.config.audio_samples
            if not self.config.audio_enabled:
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Audio pack is disabled."],
                    "coverage": {
                        "unit": unit,
                        "windowSize": window_size,
                        "windows": 0,
                        "discoveredUnits": 0,
                        "processedUnits": 0,
                        "tailUnits": 0,
                        "complete": False,
                    },
                }
            window_stream = self._iter_audio_windows(path)
            parameters = list(self.modalities.audio.parameters())
        elif kind == "video":
            unit = "frames"
            window_size = self.config.video_frames
            if not self.config.video_enabled:
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Video pack is disabled."],
                    "coverage": {
                        "unit": unit,
                        "windowSize": window_size,
                        "windows": 0,
                        "discoveredUnits": 0,
                        "processedUnits": 0,
                        "tailUnits": 0,
                        "complete": False,
                    },
                }
            window_stream = self._iter_video_windows(path)
            parameters = list(self.modalities.video.parameters())
        else:
            return {
                "trained": False,
                "loss": 0.0,
                "warnings": ["Unsupported binary modality; no parameters changed."],
                "coverage": {
                    "unit": "unknown",
                    "windowSize": 0,
                    "windows": 0,
                    "discoveredUnits": 0,
                    "processedUnits": 0,
                    "tailUnits": 0,
                    "sensoryAssemblies": 0,
                    "complete": False,
                },
            }
        parameters.extend(self.modalities.imagination_selector.parameters())
        optimizer = adamw_for_remaining_parameters(
            parameters, lr=self.config.learning_rate, weight_decay=1e-5
        )
        steps_per_window = max(1, int(steps))
        windows = 0
        processed_units = 0
        tail_units = 0
        loss_sum = 0.0
        loss_count = 0
        initial_loss = 0.0
        final_loss = 0.0
        sensory_sum = torch.zeros(self.config.vsa_dim, dtype=torch.float32)
        sensory_assemblies: List[str] = []
        fingerprint_base = content_sha256 or hashlib.sha256(
            ("%s:%s:%s" % (kind, source_name, Path(path).name)).encode("utf-8")
        ).hexdigest()
        for target, actual_units in window_stream:
            windows += 1
            processed_units += int(actual_units)
            if int(actual_units) < window_size:
                tail_units = int(actual_units)
            for index in range(steps_per_window):
                optimizer.zero_grad(set_to_none=True)
                window_embedding: Optional[torch.Tensor] = None
                if kind == "image":
                    components = []
                    if self.config.image_enabled:
                        image_output = self.modalities.image(target, idea)
                        components.append(
                            image_output["loss"]
                            + 0.2
                            * (
                                1.0
                                - F.cosine_similarity(
                                    image_output["embedding"],
                                    F.normalize(idea, dim=-1),
                                ).mean()
                            )
                        )
                        window_embedding = image_output["embedding"]
                    if self.config.vision_enabled:
                        embedding = self.modalities.vision(target)
                        window_embedding = (
                            embedding
                            if window_embedding is None
                            else F.normalize(window_embedding + embedding, dim=-1)
                        )
                        components.append(
                            0.2
                            * (
                                1.0
                                - F.cosine_similarity(
                                    embedding, F.normalize(idea, dim=-1)
                                ).mean()
                            )
                        )
                    loss = torch.stack(components).sum()
                elif kind == "audio":
                    output = self.modalities.audio(target, idea)
                    window_embedding = output["embedding"]
                    loss = output["loss"] + 0.2 * (
                        1.0
                        - F.cosine_similarity(
                            output["embedding"], F.normalize(idea, dim=-1)
                        ).mean()
                    )
                else:
                    output = self.modalities.video(target, idea)
                    window_embedding = output["embedding"]
                    loss = output["loss"] + 0.2 * (
                        1.0
                        - F.cosine_similarity(
                            output["embedding"], F.normalize(idea, dim=-1)
                        ).mean()
                    )
                if window_embedding is None:
                    raise RuntimeError(
                        "modality encoder produced no sensory embedding"
                    )
                route_target = torch.full(
                    (window_embedding.shape[0],),
                    IMAGINATION_MODALITIES.index(kind),
                    dtype=torch.long,
                    device=self.device,
                )
                route_loss = F.cross_entropy(
                    self.modalities.imagination_logits(
                        window_embedding.detach()
                    ),
                    route_target,
                )
                loss = loss + 0.08 * route_loss
                stability_loss = self._stability_penalty(parameters)
                loss = loss + stability_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite modality training loss")
                loss.backward()
                self._accumulate_slow_importance(parameters)
                torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
                optimizer.step()
                loss_value = float(loss.detach().item())
                if loss_count == 0:
                    initial_loss = loss_value
                final_loss = loss_value
                loss_sum += loss_value
                loss_count += 1
                self.counters["training_steps"] += 1
                if progress is not None:
                    # The stream may not know its total duration in advance.
                    # This monotonic asymptotic estimate is finalized by ingest.
                    progress(
                        min(
                            0.99,
                            (
                                windows
                                - 1
                                + (index + 1) / float(steps_per_window)
                            )
                            / float(windows + 1),
                        ),
                        "Training %s modality window %d" % (kind, windows),
                    )
            sensory = self._admit_sensory_embedding(
                window_embedding,
                kind=kind,
                source_name=source_name,
                fingerprint="%s:window:%d" % (fingerprint_base, windows),
            )
            sensory_sum.add_(sensory.pop("vector"))
            sensory_assemblies.append(str(sensory["assemblyId"]))
            del target
        if windows == 0:
            raise ValueError("%s contained no decodable %s" % (kind, unit))
        self._commit_slow_anchors(rate=0.12)
        if kind == "image":
            if self.config.image_enabled:
                self.modality_training["image"] += loss_count
            if self.config.vision_enabled:
                self.modality_training["vision"] += loss_count
        else:
            self.modality_training[kind] += loss_count
        record_sensory = self._admit_sensory_embedding(
            sensory_sum / float(max(1, windows)),
            kind=kind,
            source_name=source_name,
            fingerprint="%s:record" % fingerprint_base,
            child_ids=sensory_assemblies,
            importance=0.82,
            substrate_space=True,
        )
        record_sensory.pop("vector", None)
        media_coverage: Dict[str, Any] = {
            "unit": unit,
            "windowSize": window_size,
            "windows": windows,
            "discoveredUnits": processed_units,
            "processedUnits": processed_units,
            "tailUnits": tail_units,
            "sensoryAssemblies": len(sensory_assemblies) + 1,
            "wholeRecordAssemblyId": record_sensory["assemblyId"],
            "complete": True,
        }
        if unit == "samples":
            media_coverage["processedSamples"] = processed_units
            media_coverage["tailSamples"] = tail_units
        elif unit == "frames":
            media_coverage["processedFrames"] = processed_units
            media_coverage["tailFrames"] = tail_units
        embedded_audio: Optional[Dict[str, Any]] = None
        embedded_steps = 0
        embedded_loss = 0.0
        if (
            kind == "video"
            and _include_embedded_audio
            and self.config.audio_enabled
            and Path(path).suffix.lower() not in {".gif", ".webp"}
        ):
            try:
                audio_result = self._train_media(
                    path,
                    "audio",
                    source_name + " (embedded audio)",
                    steps=steps_per_window,
                    progress=progress,
                    content_sha256=fingerprint_base + ":embedded-audio",
                    _include_embedded_audio=False,
                )
                embedded_steps = int(audio_result.get("steps", 0))
                embedded_loss = float(audio_result.get("loss", 0.0))
                embedded_audio = {
                    "detected": True,
                    "trained": bool(audio_result.get("trained", False)),
                    "steps": embedded_steps,
                    "loss": embedded_loss,
                    "coverage": dict(audio_result.get("coverage", {})),
                }
            except (RuntimeError, ValueError, OSError):
                # A silent video is valid visual training data. The visual
                # traversal remains complete while the audit explicitly says
                # that no decodable audio track was admitted.
                embedded_audio = {
                    "detected": False,
                    "trained": False,
                    "steps": 0,
                    "loss": 0.0,
                    "coverage": self._empty_media_coverage("audio"),
                }
            media_coverage["embeddedAudio"] = embedded_audio
        total_steps = loss_count + embedded_steps
        combined_loss = (
            loss_sum + embedded_loss * embedded_steps
        ) / float(max(1, total_steps))
        return {
            "trained": True,
            "loss": combined_loss,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "steps": total_steps,
            "primarySteps": loss_count,
            "stepsPerWindow": steps_per_window,
            "windows": windows,
            "coverage": media_coverage,
            "sensory": {
                "windowAssemblyIds": sensory_assemblies,
                **record_sensory,
                "rawSourceTextInjected": False,
            },
            "embeddedAudio": embedded_audio,
            "warnings": warnings,
        }

    def _empty_media_coverage(
        self, kind: str = "", complete: bool = False
    ) -> Dict[str, Any]:
        unit = {
            "image": "images",
            "audio": "samples",
            "video": "frames",
        }.get(kind, "none")
        window_size = {
            "image": 1,
            "audio": self.config.audio_samples,
            "video": self.config.video_frames,
        }.get(kind, 0)
        return {
            "unit": unit,
            "windowSize": window_size,
            "windows": 0,
            "discoveredUnits": 0,
            "processedUnits": 0,
            "tailUnits": 0,
            "records": 0,
            "trainedRecords": 0,
            "failedRecords": 0,
            "complete": bool(complete),
            "byModality": {},
        }

    def _aggregate_media_reports(
        self,
        reports: Sequence[Dict[str, Any]],
        *,
        complete_when_empty: bool = False,
    ) -> Dict[str, Any]:
        accumulator = self._empty_media_accumulator()
        for report in reports:
            self._accumulate_media_report(accumulator, report)
        return self._media_result_from_accumulator(
            accumulator, complete_when_empty=complete_when_empty
        )

    @staticmethod
    def _empty_media_accumulator() -> Dict[str, Any]:
        return {
            "format": MEDIA_ACCUMULATOR_FORMAT,
            "formatVersion": MEDIA_ACCUMULATOR_VERSION,
            "records": 0,
            "trainedRecords": 0,
            "weightedLoss": 0.0,
            "steps": 0,
            "warnings": [],
            "warningCount": 0,
            "warningsTruncated": False,
            "byModality": {},
            "recordSamples": [],
            "recordSamplesTruncated": False,
        }

    @staticmethod
    def _media_counter(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return min(MEDIA_COUNTER_MAX, max(0, int(value)))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _media_hash(value: Any) -> str:
        return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()

    @classmethod
    def _safe_media_coverage(
        cls, kind: str, value: Any
    ) -> Dict[str, Any]:
        raw = dict(value) if isinstance(value, Mapping) else {}
        normalized_kind = kind if kind in {"image", "audio", "video"} else "unknown"
        unit = {
            "image": "images",
            "audio": "samples",
            "video": "frames",
        }.get(normalized_kind, "unknown")
        safe: Dict[str, Any] = {
            "unit": unit,
            "windowSize": cls._media_counter(raw.get("windowSize", 0)),
            "windows": cls._media_counter(raw.get("windows", 0)),
            "discoveredUnits": cls._media_counter(
                raw.get("discoveredUnits", 0)
            ),
            "processedUnits": cls._media_counter(raw.get("processedUnits", 0)),
            "tailUnits": cls._media_counter(raw.get("tailUnits", 0)),
            "sensoryAssemblies": cls._media_counter(
                raw.get("sensoryAssemblies", 0)
            ),
            "complete": bool(raw.get("complete", False)),
        }
        assembly_id = raw.get("wholeRecordAssemblyId")
        if isinstance(assembly_id, str) and assembly_id:
            safe["wholeRecordAssemblySha256"] = cls._media_hash(assembly_id)
            if re.fullmatch(r"[a-f0-9]{16,64}", assembly_id):
                safe["wholeRecordAssemblyId"] = assembly_id
        else:
            assembly_hash = str(raw.get("wholeRecordAssemblySha256", ""))
            if re.fullmatch(r"[a-f0-9]{64}", assembly_hash):
                safe["wholeRecordAssemblySha256"] = assembly_hash
        return safe

    @classmethod
    def _safe_media_sample(
        cls,
        report: Mapping[str, Any],
        *,
        kind: str,
        trained: bool,
        steps: int,
        warning_count: int,
    ) -> Dict[str, Any]:
        content_hash = str(report.get("contentSha256", "")).strip().lower()
        if not re.fullmatch(r"[a-f0-9]{64}", content_hash):
            content_hash = cls._media_hash(content_hash)
        sample: Dict[str, Any] = {
            "nameSha256": cls._media_hash(report.get("name", "")),
            "kind": kind,
            "contentSha256": content_hash,
            "trained": trained,
            "loss": _finite_number(report.get("loss"), 0.0),
            "steps": steps,
            "warningCount": warning_count,
            "coverage": cls._safe_media_coverage(
                kind, report.get("coverage", {})
            ),
        }
        sensory = report.get("sensory")
        if isinstance(sensory, Mapping):
            assembly_id = sensory.get("assemblyId") or sensory.get("assembly_id")
            if isinstance(assembly_id, str) and assembly_id:
                sample["sensoryAssemblySha256"] = cls._media_hash(assembly_id)
                if re.fullmatch(r"[a-f0-9]{16,64}", assembly_id):
                    sample["sensory"] = {"assemblyId": assembly_id}
            window_ids = sensory.get("windowAssemblyIds")
            if isinstance(window_ids, (list, tuple)):
                sample["sensoryWindowAssemblyCount"] = min(
                    MEDIA_COUNTER_MAX, len(window_ids)
                )
        diagnostic = json.dumps(
            sample,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        sample["diagnosticSha256"] = hashlib.sha256(diagnostic).hexdigest()
        return sample

    @classmethod
    def _accumulate_media_report(
        cls, accumulator: Dict[str, Any], report: Mapping[str, Any]
    ) -> None:
        """Fold media results into a fixed-schema, privacy-safe audit sample."""

        kind_value = str(report.get("kind", "unknown"))
        kind = kind_value if kind_value in {"image", "audio", "video"} else "unknown"
        trained = bool(report.get("trained", False))
        steps = cls._media_counter(report.get("steps", 0))
        accumulator["format"] = MEDIA_ACCUMULATOR_FORMAT
        accumulator["formatVersion"] = MEDIA_ACCUMULATOR_VERSION
        accumulator["records"] = min(
            MEDIA_COUNTER_MAX, cls._media_counter(accumulator.get("records", 0)) + 1
        )
        accumulator["trainedRecords"] = min(
            MEDIA_COUNTER_MAX,
            cls._media_counter(accumulator.get("trainedRecords", 0))
            + (1 if trained else 0),
        )
        accumulator["weightedLoss"] = _finite_number(
            accumulator.get("weightedLoss"), 0.0
        ) + _finite_number(report.get("loss"), 0.0) * max(1, steps)
        accumulator["steps"] = min(
            MEDIA_COUNTER_MAX,
            cls._media_counter(accumulator.get("steps", 0)) + steps,
        )
        report_warnings = [str(value) for value in report.get("warnings", [])]
        accumulator["warningCount"] = min(
            MEDIA_COUNTER_MAX,
            cls._media_counter(accumulator.get("warningCount", 0))
            + len(report_warnings),
        )
        warning_samples = accumulator.setdefault("warnings", [])
        for warning in report_warnings:
            if len(warning_samples) < MEDIA_DIAGNOSTIC_SAMPLES:
                warning_samples.append("sha256:" + cls._media_hash(warning))
            else:
                accumulator["warningsTruncated"] = True

        record_coverage = cls._safe_media_coverage(
            kind, report.get("coverage", {})
        )
        by_modality = accumulator.setdefault("byModality", {})
        bucket = by_modality.setdefault(
            kind,
            {
                "unit": record_coverage.get("unit", "unknown"),
                "windowSize": int(record_coverage.get("windowSize", 0)),
                "windows": 0,
                "discoveredUnits": 0,
                "processedUnits": 0,
                "tailUnits": 0,
                "sensoryAssemblies": 0,
                "records": 0,
                "trainedRecords": 0,
                "failedRecords": 0,
                "complete": True,
            },
        )
        for field in (
            "windows",
            "discoveredUnits",
            "processedUnits",
            "tailUnits",
            "sensoryAssemblies",
        ):
            bucket[field] = min(
                MEDIA_COUNTER_MAX,
                cls._media_counter(bucket.get(field, 0))
                + cls._media_counter(record_coverage.get(field, 0)),
            )
        bucket["records"] = min(
            MEDIA_COUNTER_MAX, cls._media_counter(bucket.get("records", 0)) + 1
        )
        bucket["trainedRecords"] = min(
            MEDIA_COUNTER_MAX,
            cls._media_counter(bucket.get("trainedRecords", 0))
            + (1 if trained else 0),
        )
        bucket["failedRecords"] = min(
            MEDIA_COUNTER_MAX,
            cls._media_counter(bucket.get("failedRecords", 0))
            + (0 if trained else 1),
        )
        bucket["complete"] = bool(bucket.get("complete", True)) and bool(
            record_coverage.get("complete", False)
        )
        samples = accumulator.setdefault("recordSamples", [])
        if len(samples) < MEDIA_DIAGNOSTIC_SAMPLES:
            samples.append(
                cls._safe_media_sample(
                    report,
                    kind=kind,
                    trained=trained,
                    steps=steps,
                    warning_count=len(report_warnings),
                )
            )
        else:
            accumulator["recordSamplesTruncated"] = True

    @classmethod
    def _checkpoint_media_accumulator(
        cls, accumulator: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Rebuild the checkpoint from a strict scalar/count/hash allowlist."""

        raw = dict(accumulator) if isinstance(accumulator, Mapping) else {}
        safe = cls._empty_media_accumulator()
        safe["records"] = cls._media_counter(raw.get("records", 0))
        safe["trainedRecords"] = min(
            safe["records"], cls._media_counter(raw.get("trainedRecords", 0))
        )
        safe["weightedLoss"] = _finite_number(raw.get("weightedLoss"), 0.0)
        safe["steps"] = cls._media_counter(raw.get("steps", 0))
        safe["warningCount"] = cls._media_counter(raw.get("warningCount", 0))
        warnings = raw.get("warnings", [])
        if isinstance(warnings, list):
            safe["warnings"] = [
                value
                for value in warnings[:MEDIA_DIAGNOSTIC_SAMPLES]
                if isinstance(value, str)
                and re.fullmatch(r"sha256:[a-f0-9]{64}", value)
            ]
        safe["warningsTruncated"] = bool(raw.get("warningsTruncated", False)) or (
            isinstance(warnings, list) and len(warnings) > MEDIA_DIAGNOSTIC_SAMPLES
        )

        raw_modalities = raw.get("byModality", {})
        if isinstance(raw_modalities, Mapping):
            for kind in ("audio", "image", "video", "unknown"):
                bucket = raw_modalities.get(kind)
                if not isinstance(bucket, Mapping):
                    continue
                coverage = cls._safe_media_coverage(kind, bucket)
                safe["byModality"][kind] = {
                    **coverage,
                    "records": cls._media_counter(bucket.get("records", 0)),
                    "trainedRecords": cls._media_counter(
                        bucket.get("trainedRecords", 0)
                    ),
                    "failedRecords": cls._media_counter(
                        bucket.get("failedRecords", 0)
                    ),
                }
                safe["byModality"][kind].pop(
                    "wholeRecordAssemblySha256", None
                )
                safe["byModality"][kind].pop("wholeRecordAssemblyId", None)

        raw_samples = raw.get("recordSamples", [])
        if isinstance(raw_samples, list):
            for raw_sample in raw_samples[:MEDIA_DIAGNOSTIC_SAMPLES]:
                if not isinstance(raw_sample, Mapping):
                    continue
                kind = str(raw_sample.get("kind", "unknown"))
                if kind not in {"image", "audio", "video"}:
                    kind = "unknown"
                name_hash = str(raw_sample.get("nameSha256", ""))
                content_hash = str(raw_sample.get("contentSha256", ""))
                if not all(
                    re.fullmatch(r"[a-f0-9]{64}", value)
                    for value in (name_hash, content_hash)
                ):
                    continue
                sample: Dict[str, Any] = {
                    "nameSha256": name_hash,
                    "kind": kind,
                    "contentSha256": content_hash,
                    "trained": bool(raw_sample.get("trained", False)),
                    "loss": _finite_number(raw_sample.get("loss"), 0.0),
                    "steps": cls._media_counter(raw_sample.get("steps", 0)),
                    "warningCount": cls._media_counter(
                        raw_sample.get("warningCount", 0)
                    ),
                    "coverage": cls._safe_media_coverage(
                        kind, raw_sample.get("coverage", {})
                    ),
                }
                sensory_hash = str(
                    raw_sample.get("sensoryAssemblySha256", "")
                )
                if re.fullmatch(r"[a-f0-9]{64}", sensory_hash):
                    sample["sensoryAssemblySha256"] = sensory_hash
                sensory = raw_sample.get("sensory")
                if isinstance(sensory, Mapping):
                    assembly_id = str(sensory.get("assemblyId", ""))
                    if re.fullmatch(r"[a-f0-9]{16,64}", assembly_id):
                        sample["sensory"] = {"assemblyId": assembly_id}
                if "sensoryWindowAssemblyCount" in raw_sample:
                    sample["sensoryWindowAssemblyCount"] = cls._media_counter(
                        raw_sample.get("sensoryWindowAssemblyCount", 0)
                    )
                sample["diagnosticSha256"] = hashlib.sha256(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                safe["recordSamples"].append(sample)
        safe["recordSamplesTruncated"] = bool(
            raw.get("recordSamplesTruncated", False)
        ) or (isinstance(raw_samples, list) and len(raw_samples) > MEDIA_DIAGNOSTIC_SAMPLES)
        return safe

    def _media_result_from_accumulator(
        self,
        accumulator: Mapping[str, Any],
        *,
        complete_when_empty: bool = False,
    ) -> Dict[str, Any]:
        # The final source record and an interrupted checkpoint use the same
        # privacy-safe representation, so resuming cannot change its schema.
        accumulator = self._checkpoint_media_accumulator(accumulator)
        records = max(0, int(accumulator.get("records", 0)))
        if records == 0:
            return {
                "trained": False,
                "loss": 0.0,
                "steps": 0,
                "warnings": [],
                "warningCount": 0,
                "warningsTruncated": False,
                "coverage": self._empty_media_coverage(
                    complete=complete_when_empty
                ),
                "records": [],
                "recordSamplesTruncated": False,
            }
        by_modality = {
            str(key): dict(value)
            for key, value in dict(accumulator.get("byModality", {})).items()
            if isinstance(value, Mapping)
        }
        units = {str(value["unit"]) for value in by_modality.values()}
        window_sizes = {
            int(value["windowSize"]) for value in by_modality.values()
        }
        coverage: Dict[str, Any] = {
            "unit": next(iter(units)) if len(units) == 1 else "mixed",
            "windowSize": (
                next(iter(window_sizes)) if len(window_sizes) == 1 else 0
            ),
            "windows": sum(int(value["windows"]) for value in by_modality.values()),
            "discoveredUnits": sum(
                int(value["discoveredUnits"]) for value in by_modality.values()
            ),
            "processedUnits": sum(
                int(value["processedUnits"]) for value in by_modality.values()
            ),
            "tailUnits": sum(
                int(value["tailUnits"]) for value in by_modality.values()
            ),
            "sensoryAssemblies": sum(
                int(value["sensoryAssemblies"])
                for value in by_modality.values()
            ),
            "records": records,
            "trainedRecords": max(
                0, int(accumulator.get("trainedRecords", 0))
            ),
            "failedRecords": records
            - max(0, int(accumulator.get("trainedRecords", 0))),
            "complete": all(
                bool(value["complete"]) for value in by_modality.values()
            ),
            "byModality": by_modality,
        }
        if set(by_modality) == {"audio"}:
            coverage["processedSamples"] = coverage["processedUnits"]
            coverage["tailSamples"] = coverage["tailUnits"]
        elif set(by_modality) == {"video"}:
            coverage["processedFrames"] = coverage["processedUnits"]
            coverage["tailFrames"] = coverage["tailUnits"]
        return {
            "trained": int(accumulator.get("trainedRecords", 0)) > 0,
            "loss": _finite_number(accumulator.get("weightedLoss"), 0.0)
            / max(1, int(accumulator.get("steps", 0))),
            "steps": max(0, int(accumulator.get("steps", 0))),
            "warnings": [str(value) for value in accumulator.get("warnings", [])],
            "warningCount": max(0, int(accumulator.get("warningCount", 0))),
            "warningsTruncated": bool(
                accumulator.get("warningsTruncated", False)
            ),
            "coverage": coverage,
            "records": [
                dict(value)
                for value in accumulator.get("recordSamples", [])
                if isinstance(value, Mapping)
            ],
            "recordSamplesTruncated": bool(
                accumulator.get("recordSamplesTruncated", False)
            ),
        }

    @staticmethod
    def _require_complete_ingestion_coverage(
        coverage: DatasetCoverage,
        *,
        policy: str,
        resolved_kind: str,
        record_count_hint: Optional[int],
    ) -> None:
        if coverage.as_dict()["complete"] is not True:
            raise RuntimeError(
                "dataset traversal stopped before every valid record was visited; "
                "the source remains incomplete and cannot be marked trained"
            )
        if (
            policy != "archive"
            and resolved_kind == "parquet"
            and record_count_hint is not None
            and coverage.discovered_records != int(record_count_hint)
        ):
            raise RuntimeError(
                "Parquet traversal does not match its footer-declared row count"
            )

    def ingest(
        self,
        path: Optional[str] = None,
        text: Optional[str] = None,
        name: str = "",
        kind: str = "",
        policy: str = "encode",
        expected_hash: str = "",
        committed_sqlite_snapshot: bool = False,
        allow_replay: bool = False,
        epoch: int = 0,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if policy not in {"encode", "consolidate", "pretrain", "archive"}:
            raise ValueError("unsupported ingestion policy")

        def capture_source_stat(
            target: Path, format_name: str
        ) -> Dict[str, int]:
            current = target.stat()
            snapshot = {
                "device": int(current.st_dev),
                "inode": int(current.st_ino),
                "size": int(current.st_size),
                "mtimeNs": int(current.st_mtime_ns),
            }
            if format_name == "sqlite":
                # Committed rows may live only in WAL.  Sidecar identity is a
                # cheap between-batch mutation guard; the transactional backup
                # digest below remains the authoritative content identity.
                for label, suffix in (("Wal", "-wal"), ("Shm", "-shm")):
                    sidecar = Path(str(target) + suffix)
                    try:
                        stat = sidecar.stat()
                    except FileNotFoundError:
                        snapshot["sqlite%sPresent" % label] = 0
                        snapshot["sqlite%sDevice" % label] = 0
                        snapshot["sqlite%sInode" % label] = 0
                        snapshot["sqlite%sSize" % label] = 0
                        snapshot["sqlite%sMtimeNs" % label] = 0
                    else:
                        snapshot["sqlite%sPresent" % label] = 1
                        snapshot["sqlite%sDevice" % label] = int(stat.st_dev)
                        snapshot["sqlite%sInode" % label] = int(stat.st_ino)
                        snapshot["sqlite%sSize" % label] = int(stat.st_size)
                        snapshot["sqlite%sMtimeNs" % label] = int(
                            stat.st_mtime_ns
                        )
            return snapshot

        source_path: Optional[Path] = None
        source_snapshot: Optional[Dict[str, int]] = None
        raw_bytes: Optional[bytes] = None
        extracted = ""
        coverage = DatasetCoverage()
        record_count_hint: Optional[int] = None
        if path:
            source_path = Path(path).resolve()
            if not source_path.exists() or not source_path.is_file():
                raise FileNotFoundError(str(source_path))
            resolved_kind = dataset_format(source_path, kind)
            source_name = name or source_path.name
            opened_stat = source_path.stat()
            source_bytes = int(opened_stat.st_size)
            record_count_hint = dataset_record_count_hint(
                source_path, resolved_kind
            )
            if resolved_kind == "sqlite":
                if committed_sqlite_snapshot:
                    if not re.fullmatch(r"[a-f0-9]{64}", expected_hash.lower()):
                        raise ValueError(
                            "committed SQLite snapshot requires its manifest sha256"
                        )
                    source_snapshot = capture_source_stat(
                        source_path, resolved_kind
                    )
                    committed_digest = hashlib.sha256()
                    with source_path.open("rb") as source_stream:
                        for block in iter(
                            lambda: source_stream.read(1024 * 1024), b""
                        ):
                            committed_digest.update(block)
                    content_hash = committed_digest.hexdigest()
                    if source_snapshot != capture_source_stat(
                        source_path, resolved_kind
                    ):
                        raise ValueError(
                            "committed SQLite snapshot changed while it was hashed"
                        )
                else:
                    # Hash the same transactionally consistent SQLite view that
                    # record traversal uses, including committed WAL pages. Two
                    # adjacent snapshots plus sidecar stats close the preflight
                    # race before any neural update is allowed.
                    first_hash = sqlite_consistent_snapshot_sha256(source_path)
                    source_snapshot = capture_source_stat(
                        source_path, resolved_kind
                    )
                    content_hash = sqlite_consistent_snapshot_sha256(source_path)
                    if (
                        first_hash != content_hash
                        or source_snapshot
                        != capture_source_stat(source_path, resolved_kind)
                    ):
                        raise ValueError(
                            "ingestion SQLite source changed while it was snapshotted"
                        )
            else:
                source_snapshot = capture_source_stat(
                    source_path, resolved_kind
                )
                if re.fullmatch(r"[a-f0-9]{64}", expected_hash.lower()):
                    # Electron just hashed the immutable manifest entry and
                    # passes that identity into this transaction. Avoid a
                    # second complete pre-training scan of 30+ GB sources; the
                    # final verification below still proves the source did not
                    # change while neural updates streamed.
                    content_hash = expected_hash.lower()
                else:
                    digest = hashlib.sha256()
                    with source_path.open("rb") as source_stream:
                        for block in iter(
                            lambda: source_stream.read(1024 * 1024), b""
                        ):
                            digest.update(block)
                    content_hash = digest.hexdigest()
                if source_snapshot != capture_source_stat(
                    source_path, resolved_kind
                ):
                    raise ValueError(
                        "ingestion source changed while it was hashed"
                    )
        elif text is not None:
            extracted = str(text).replace("\x00", "")
            raw_bytes = extracted.encode("utf-8")
            resolved_kind = kind or "text"
            source_name = name or "pasted-text"
            source_bytes = len(raw_bytes)
            content_hash = hashlib.sha256(raw_bytes).hexdigest()
            record_count_hint = 1 if extracted.strip() else 0
        else:
            raise ValueError("ingest requires path or text")
        if expected_hash and expected_hash.lower() != content_hash:
            raise ValueError("ingestion content hash mismatch")

        def verify_source_snapshot(*, full_hash: bool = False) -> None:
            if source_path is None or source_snapshot is None:
                return
            observed = capture_source_stat(source_path, resolved_kind)
            if observed != source_snapshot:
                raise ValueError(
                    "ingestion source changed after its immutable snapshot"
                )
            if full_hash:
                if resolved_kind == "sqlite":
                    if committed_sqlite_snapshot:
                        current_digest = hashlib.sha256()
                        with source_path.open("rb") as source_stream:
                            for block in iter(
                                lambda: source_stream.read(1024 * 1024), b""
                            ):
                                current_digest.update(block)
                        observed_hash = current_digest.hexdigest()
                    else:
                        observed_hash = sqlite_consistent_snapshot_sha256(
                            source_path
                        )
                else:
                    current_digest = hashlib.sha256()
                    with source_path.open("rb") as source_stream:
                        for block in iter(
                            lambda: source_stream.read(1024 * 1024), b""
                        ):
                            current_digest.update(block)
                    observed_hash = current_digest.hexdigest()
                if observed_hash != content_hash:
                    raise ValueError(
                        "ingestion source content changed during traversal"
                    )
                if capture_source_stat(
                    source_path, resolved_kind
                ) != source_snapshot:
                    raise ValueError(
                        "ingestion source changed during final verification"
                    )
        checkpoint_key = ""
        checkpoint: Optional[Dict[str, Any]] = None
        source_manifest_sha256 = ""
        parser_manifest_sha256 = ""
        transaction_id = ""
        source_name_hash = hashlib.sha256(
            source_name.encode("utf-8")
        ).hexdigest()
        if source_path is not None and policy != "archive":
            checkpoint_key = hashlib.sha256(
                ("file\0" + str(source_path)).encode("utf-8")
            ).hexdigest()
            transaction_id = hashlib.sha256(
                (
                    "%s\0%d\0%s\0%s\0%d"
                    % (
                        content_hash,
                        max(0, int(epoch)),
                        policy,
                        resolved_kind,
                        source_bytes,
                    )
                ).encode("utf-8")
            ).hexdigest()
            stored_checkpoint = self.ingestion_checkpoints.get(checkpoint_key)
            if stored_checkpoint is not None:
                # Revalidate the in-memory value as rigorously as load(). A
                # test harness or future caller must not be able to smuggle a
                # malformed cursor past the durable-state gate.
                checkpoint = self._validated_ingestion_checkpoints(
                    {checkpoint_key: stored_checkpoint}
                )[checkpoint_key]
                if (
                    checkpoint["transactionId"] != transaction_id
                    or checkpoint["contentHash"] != content_hash
                    or checkpoint["epoch"] != max(0, int(epoch))
                    or checkpoint["policy"] != policy
                    or checkpoint["resolvedKind"] != resolved_kind
                    or checkpoint["sourceBytes"] != source_bytes
                    or checkpoint["sourceNameHash"] != source_name_hash
                    or checkpoint["sourceSnapshot"] != source_snapshot
                ):
                    raise ValueError(
                        "active ingestion checkpoint does not match content hash, "
                        "epoch, policy, kind, size, or source name"
                    )
            if checkpoint is None or checkpoint.get("formatVersion") == 3:
                # v3 may not treat a caller-supplied expected hash as a fresh
                # observation. Read the actual file before the first neural
                # update and again at final traversal verification.
                verify_source_snapshot(full_hash=True)
                source_manifest_sha256, parser_manifest_sha256 = (
                    self._ingestion_v3_manifest_hashes(
                        content_hash=content_hash,
                        source_bytes=source_bytes,
                        resolved_kind=resolved_kind,
                        source_name_hash=source_name_hash,
                        record_count_hint=record_count_hint,
                    )
                )
            if checkpoint is not None and checkpoint.get("formatVersion") == 3:
                if (
                    checkpoint["sourceManifestSha256"] != source_manifest_sha256
                    or checkpoint["parserManifestSha256"] != parser_manifest_sha256
                    or checkpoint["expectedRecords"] != record_count_hint
                ):
                    raise ValueError(
                        "v3 source/parser manifest changed before cursor resume"
                    )
                observed_joint = recover_joint_generation(
                    self.engine_path / "state" / "ingestion-joint",
                    self.ingestion_joint_generation,
                    neural_store_root=self.engine_path / "state",
                    substrate_store_root=self.engine_path / "substrate",
                    source_manifest_sha256=source_manifest_sha256,
                    parser_manifest_sha256=parser_manifest_sha256,
                    source_content_sha256=content_hash,
                )
                if observed_joint is None or (
                    observed_joint.manifest["neuralState"]["generationId"]
                    != (self.mutable_state_manifest or {}).get("activeGeneration")
                    or observed_joint.manifest["substrateState"]["generationId"]
                    != (self.memory.persistence_manifest or {}).get("activeGeneration")
                ):
                    raise ValueError("v3 resume has no matching committed neural state")
                vector_generation, index_generation = (
                    self._v3_committed_substrate_generations(
                        self.memory.persistence_manifest or {}
                    )
                )
                observed_binding = validate_checkpoint_binding_v3(
                    checkpoint["pagedCheckpointBinding"],
                    checkpoint["pagedCheckpointBindingSha256"],
                    schedule=checkpoint["learningSchedule"],
                    schedule_sha256_value=checkpoint["learningScheduleSha256"],
                    source_manifest_sha256=source_manifest_sha256,
                    parser_manifest_sha256=parser_manifest_sha256,
                    source_content_sha256=content_hash,
                    neural_state_sha256=self.parameter_checksum(),
                    vector_generation=vector_generation,
                    index_generation=index_generation,
                )
                if (
                    observed_binding["cursor"] != observed_joint.manifest["cursor"]
                    or observed_binding["coverage"] != observed_joint.manifest["coverage"]
                ):
                    raise ValueError("v3 resume cursor differs from committed coverage")
        if self.ingestion_checkpoints and checkpoint is None:
            # Neural state may only advance along the transaction named by its
            # committed cursor. Interleaving another source would make a later
            # rollback ambiguous, so the caller must first resume the exact
            # hash/epoch/policy tuple already in progress.
            raise RuntimeError(
                "another record ingestion checkpoint is active; resume that "
                "exact dataset transaction before starting a different source"
            )
        completed_transaction = next(
            (
                value
                for value in reversed(self.completed_ingestions)
                if value.get("transactionId") == transaction_id
                and value.get("contentHash") == content_hash
                and value.get("epoch") == max(0, int(epoch))
                and value.get("policy") == policy
                and value.get("sourceIdentity") == checkpoint_key
                and value.get("sourceNameHash") == source_name_hash
            ),
            None,
        )
        if completed_transaction is None:
            # The bounded global list is only a recent audit index. The latest
            # receipt for every retained source is also embedded on that
            # source, so correctness never depends on an arbitrary tombstone
            # count when an RPC acknowledgement is lost.
            for stored_source in self.training_sources:
                raw_receipt = stored_source.get("completionReceipt")
                if not isinstance(raw_receipt, Mapping):
                    continue
                receipt = self._validated_completed_ingestions(
                    [raw_receipt]
                )[0]
                if (
                    receipt.get("transactionId") == transaction_id
                    and receipt.get("contentHash") == content_hash
                    and receipt.get("epoch") == max(0, int(epoch))
                    and receipt.get("policy") == policy
                    and receipt.get("sourceIdentity") == checkpoint_key
                    and receipt.get("sourceNameHash") == source_name_hash
                ):
                    completed_transaction = receipt
                    break
        if completed_transaction is not None:
            source_id = str(completed_transaction.get("sourceId", ""))
            completed_source = next(
                (
                    value
                    for value in self.training_sources
                    if str(value.get("id", "")) == source_id
                ),
                None,
            )
            if completed_source is None:
                raise ValueError(
                    "completed ingestion receipt references a missing source"
                )
            return {
                "brainId": self.brain_id,
                "duplicate": True,
                "idempotentCompletion": True,
                "transactionId": transaction_id,
                "source": completed_source,
                "coverage": dict(completed_transaction["coverage"]),
                "recordRecovery": {
                    "transactionId": transaction_id,
                    "resumed": False,
                    "resumedRecords": 0,
                    "committedRecords": int(
                        completed_transaction.get("committedRecords", 0)
                    ),
                    "visitedRecords": int(
                        completed_transaction.get("visitedRecords", 0)
                    ),
                    "rejectedRecords": int(
                        completed_transaction.get("rejectedRecords", 0)
                    ),
                    "batchCommits": int(
                        completed_transaction.get("batchCommits", 0)
                    ),
                    "checkpointActive": False,
                    "completionReceiptReused": True,
                    "rawSourceTextStored": False,
                    "rawTokenIdsStored": False,
                },
                "parameterChecksumAfter": str(
                    completed_transaction["parameterChecksumAfter"]
                ),
                "metrics": self.metrics(),
            }
        neural_storage_plan = self._streaming_neural_storage_plan(source_bytes)
        if checkpoint is None:
            if checkpoint_key:
                self._ensure_paged_ingestion_substrate(source_bytes)
                learning_schedule = make_ingestion_schedule_v3(
                    source_manifest_sha256=source_manifest_sha256,
                    source_content_sha256=content_hash,
                    parser_manifest_sha256=parser_manifest_sha256,
                    physical_batch_records=max(
                        1, int(neural_storage_plan["physicalBatchRecords"])
                    ),
                    gradient_accumulation=max(
                        1, int(neural_storage_plan["gradientAccumulation"])
                    ),
                    training_sequence_tokens=max(
                        8, int(neural_storage_plan["trainingSequenceTokens"])
                    ),
                    checkpoint_records=max(1, int(self._ingestion_checkpoint_records)),
                    assembly_page_records=128,
                )
                for field in (
                    "detailedRecordAssemblies", "corpusRepresentation",
                    "slowGradientMode", "physicalBatchRecords",
                    "gradientAccumulation", "trainingSequenceTokens",
                ):
                    neural_storage_plan[field] = learning_schedule[field]
            else:
                learning_schedule = self._ingestion_learning_schedule(
                    neural_storage_plan, self._ingestion_checkpoint_records
                )
        else:
            # The current plan remains useful for live pause/readiness signals,
            # but it must not rewrite choices that already produced committed
            # neural state. Those choices are part of the transaction.
            learning_schedule = dict(checkpoint["learningSchedule"])
            for field in (
                "detailedRecordAssemblies",
                "physicalBatchRecords",
                "gradientAccumulation",
                "trainingSequenceTokens",
                "corpusRepresentation",
                "slowGradientMode",
            ):
                neural_storage_plan[field] = learning_schedule[field]
            for field in ("localTypedTargetWindowPolicy",):
                if field in learning_schedule:
                    neural_storage_plan[field] = learning_schedule[field]
        if policy != "archive" and bool(
            neural_storage_plan["detailedRecordAssemblies"]
        ):
            estimated_detailed_bytes = max(
                1, int(neural_storage_plan["projectedDetailedBytes"])
            )
            detailed_admission = self.resource_policy.status(
                estimated_write_bytes=estimated_detailed_bytes * 2,
                estimated_ram_bytes=estimated_detailed_bytes,
            )
            offload_status: Optional[Dict[str, Any]] = None
            if bool(detailed_admission["memoryPressure"]):
                # Free only safely spillable state, then remeasure the exact
                # allocation. A transient watermark must never rewrite this
                # source's frozen learning schedule to statistical-only.
                offload_status = self._maintain_neural_state_resources()
                detailed_admission = self.resource_policy.status(
                    estimated_write_bytes=estimated_detailed_bytes * 2,
                    estimated_ram_bytes=estimated_detailed_bytes,
                )
            neural_storage_plan["detailedAdmissionStatus"] = (
                detailed_admission
            )
            if bool(detailed_admission["memoryPressure"]):
                wait = self._memory_pressure_wait(
                    {
                        **detailed_admission,
                        **(
                            {"stateOffload": offload_status}
                            if offload_status is not None
                            else {}
                        ),
                        "detailedRepresentationPreferred": True,
                        "detailedRepresentationDeferred": True,
                        "representationDowngradedForTransientPressure": False,
                        "sourceRecordsSkipped": False,
                        "resumeFromLastCheckpoint": checkpoint is not None,
                    },
                    stage="detailed neural ingestion",
                    detail=(
                        "The small source remains scheduled for detailed "
                        "neural sequences. Spillable state was considered, "
                        "but the exact allocation still needs more RAM."
                    ),
                )
                raise NeuralStateResourcePause(
                    "detailed neural ingestion is waiting for memory without "
                    "downgrading its learning schedule",
                    wait,
                )
            neural_storage_plan["detailedRepresentationDeferred"] = False
            neural_storage_plan["representationDecision"] = (
                "detailed-admitted-after-offload"
                if offload_status is not None
                else "detailed-admitted"
            )
            neural_storage_plan["resourceStatus"] = (
                self.resource_policy.status()
            )
        if policy != "archive" and bool(
            neural_storage_plan["resourceStatus"]["diskPressure"]
        ):
            raise NeuralStateResourcePause(
                "dataset learning paused before crossing the neural-state disk reserve",
                neural_storage_plan["resourceStatus"],
            )
        duplicate = next(
            (
                source
                for source in self.training_sources
                if source.get("content_hash") == content_hash
            ),
            None,
        )
        if duplicate is not None and not allow_replay and checkpoint is None:
            return {
                "brainId": self.brain_id,
                "duplicate": True,
                "source": duplicate,
                "metrics": self.metrics(),
            }

        checkpoint_baseline = (
            dict(checkpoint["baseline"]) if checkpoint is not None else None
        )
        before_checksum = (
            str(checkpoint_baseline["parameterChecksum"])
            if checkpoint_baseline is not None
            else self.parameter_checksum()
        )
        before_concepts = (
            int(checkpoint_baseline["concepts"])
            if checkpoint_baseline is not None
            else len(self.memory.concepts)
        )
        before_ideas = (
            int(checkpoint_baseline["ideas"])
            if checkpoint_baseline is not None
            else len(self.memory.ideas)
        )
        before_events = (
            int(checkpoint_baseline["plasticityEvents"])
            if checkpoint_baseline is not None
            else int(self.router.synapses.plasticity_events.item())
        )
        before_memory_neurons = (
            int(checkpoint_baseline.get("memoryNeurons", len(self.memory.neurons)))
            if checkpoint_baseline is not None
            else len(self.memory.neurons)
        )
        before_memory_synapses = (
            int(checkpoint_baseline.get("memorySynapses", len(self.memory.synapses)))
            if checkpoint_baseline is not None
            else len(self.memory.synapses)
        )
        current_memory_synaptic_uses = self.memory.synaptic_use_count()
        before_memory_synaptic_uses = (
            int(
                checkpoint_baseline.get(
                    "memorySynapticUses", current_memory_synaptic_uses
                )
            )
            if checkpoint_baseline is not None
            else current_memory_synaptic_uses
        )
        before_training_steps = (
            int(
                checkpoint_baseline.get(
                    "trainingSteps", self.counters["training_steps"]
                )
            )
            if checkpoint_baseline is not None
            else int(self.counters["training_steps"])
        )
        before_statistical_experiences = (
            int(
                checkpoint_baseline.get(
                    "statisticalExperiences",
                    sum(
                        int(value.get("statistical_experiences", 0))
                        for value in self.memory.assemblies
                    ),
                )
            )
            if checkpoint_baseline is not None
            else sum(
                int(value.get("statistical_experiences", 0))
                for value in self.memory.assemblies
            )
        )
        aggregate = dict(checkpoint.get("aggregate", {})) if checkpoint else {}
        loss_total = _finite_number(aggregate.get("lossTotal"), 0.0)
        learned_chunks = max(0, int(aggregate.get("learnedChunks", 0)))
        media_accumulator = (
            self._checkpoint_media_accumulator(aggregate["mediaAccumulator"])
            if isinstance(aggregate.get("mediaAccumulator"), Mapping)
            else self._empty_media_accumulator()
        )
        reading_report_count = max(
            0, int(aggregate.get("readingReportCount", 0))
        )
        streaming_gradient_records = max(
            0, int(aggregate.get("streamingGradientRecords", 0))
        )
        streaming_gradient_optimizer_steps = max(
            0, int(aggregate.get("streamingGradientOptimizerSteps", 0))
        )
        streaming_local_pending: List[Tuple[str, torch.Tensor]] = []
        learning_schedule_started = checkpoint is not None
        # Representation and gradient cadence are independent in the paged
        # ingestion contract: a large source needs a distinct distributed
        # assembly for each experience without forcing an optimizer step per
        # section. Version 2 schedules still encode the old coupled choices,
        # so this preserves their behavior while allowing v3 to use detailed
        # assemblies with bounded streaming microbatches.
        compact_streaming = (
            str(neural_storage_plan["slowGradientMode"])
            == "streaming-microbatch-gradient-accumulation"
        )
        capability_rehearsal_enabled = bool(
            policy == "pretrain" and eligible_ground_up_rehearsal(self)
        )
        capability_rehearsal_policy = CapabilityRehearsalPolicy(
            # Unknown/indefinite streams rehearse at a bounded checkpoint
            # cadence. Finite sources use one exact midpoint below.
            periodic_global_waves=128
        )
        capability_rehearsal_state = CapabilityScheduleState.from_dict(
            (
                checkpoint.get("capabilityRehearsal")
                if checkpoint is not None
                and isinstance(
                    checkpoint.get("capabilityRehearsal"), Mapping
                )
                else None
            )
        )
        capability_committed_waves = (
            int(checkpoint["commitSequence"])
            if checkpoint is not None
            else 0
        )
        raw_capability_cadence = (
            checkpoint.get("capabilityRehearsalCadence")
            if checkpoint is not None
            else None
        )
        if isinstance(raw_capability_cadence, Mapping):
            capability_rehearsal_cadence = dict(raw_capability_cadence)
        elif record_count_hint is not None:
            total_checkpoint_waves = max(
                1,
                math.ceil(
                    int(record_count_hint)
                    / float(max(1, int(learning_schedule["checkpointRecords"])))
                ),
            )
            capability_rehearsal_cadence = {
                "mode": "finite-midpoint",
                "checkpointRecords": int(
                    learning_schedule["checkpointRecords"]
                ),
                "expectedCheckpointWaves": total_checkpoint_waves,
                "middleWave": (
                    math.ceil(total_checkpoint_waves / 2.0)
                    if total_checkpoint_waves >= 2
                    else None
                ),
            }
        else:
            capability_rehearsal_cadence = {
                "mode": "indefinite-periodic",
                "checkpointRecords": int(
                    learning_schedule["checkpointRecords"]
                ),
                "periodicWaves": int(
                    capability_rehearsal_policy.periodic_global_waves
                ),
            }
        if (
            capability_rehearsal_cadence.get("mode")
            not in {"finite-midpoint", "indefinite-periodic"}
            or int(
                capability_rehearsal_cadence.get(
                    "checkpointRecords", -1
                )
            )
            != int(learning_schedule["checkpointRecords"])
        ):
            raise ValueError(
                "ingestion capability rehearsal cadence is invalid"
            )
        if capability_rehearsal_enabled and due_rehearsal_phase(
            capability_rehearsal_state,
            capability_rehearsal_policy,
            committed_global_waves=capability_committed_waves,
        ) == "start":
            capability_receipt = rehearse_capabilities(
                self,
                phase="start",
                committed_global_waves=capability_committed_waves,
                policy=capability_rehearsal_policy,
            )
            capability_rehearsal_state = advance_schedule_state(
                capability_rehearsal_state, capability_receipt
            )

        def flush_streaming_local() -> None:
            nonlocal loss_total
            nonlocal streaming_gradient_records
            nonlocal streaming_gradient_optimizer_steps
            nonlocal learning_schedule_started
            if not streaming_local_pending:
                return
            effective_batch_target = max(
                1,
                int(learning_schedule["physicalBatchRecords"])
                * int(learning_schedule["gradientAccumulation"]),
            )
            report = self._optimize_streaming_experience_batch(
                streaming_local_pending,
                learning_schedule=learning_schedule,
                schedule_locked=learning_schedule_started,
            )
            effective_physical = int(report["physical_batch_records"])
            effective_tokens = int(report["training_sequence_tokens"])
            if not learning_schedule_started:
                # The first allocation may recover from an OOM by selecting a
                # smaller schedule. Freeze the schedule that actually produced
                # the first successful mutation; all later flushes are exact.
                if effective_batch_target % effective_physical:
                    raise RuntimeError(
                        "allocator fallback cannot preserve the exact logical "
                        "batch target"
                    )
                effective_accumulation = (
                    effective_batch_target // effective_physical
                )
                learning_schedule["physicalBatchRecords"] = effective_physical
                learning_schedule["gradientAccumulation"] = (
                    effective_accumulation
                )
                learning_schedule["trainingSequenceTokens"] = effective_tokens
                neural_storage_plan["physicalBatchRecords"] = effective_physical
                neural_storage_plan["gradientAccumulation"] = (
                    effective_accumulation
                )
                neural_storage_plan["trainingSequenceTokens"] = effective_tokens
            elif (
                effective_physical
                != int(learning_schedule["physicalBatchRecords"])
                or effective_tokens
                != int(learning_schedule["trainingSequenceTokens"])
            ):
                raise RuntimeError(
                    "committed ingestion learning schedule changed unexpectedly"
                )
            learning_schedule_started = True
            loss_total += float(report["loss"]) * int(report["records"])
            streaming_gradient_records += int(report["records"])
            streaming_gradient_optimizer_steps += int(
                report["optimizer_steps"]
            )
            streaming_local_pending.clear()

        def queue_streaming_local(text_value: str) -> None:
            if not text_value.strip():
                return
            streaming_local_pending.append(
                (
                    text_value,
                    self.memory.vector_for_text(text_value).detach().cpu(),
                )
            )
            effective_batch = max(
                1,
                int(learning_schedule["physicalBatchRecords"])
                * int(learning_schedule["gradientAccumulation"]),
            )
            if len(streaming_local_pending) >= effective_batch:
                flush_streaming_local()






        committed_record_cursor = (
            int(checkpoint["committedRecords"]) if checkpoint is not None else 0
        )
        last_committed_record_cursor = committed_record_cursor
        resumed_record_count = committed_record_cursor
        commit_sequence = (
            int(checkpoint["commitSequence"]) if checkpoint is not None else 0
        )
        record_cursor = 0
        record_prefix_sha256 = hashlib.sha256(
            b"omni-record-prefix-v1"
        ).hexdigest()
        resume_boundary_validated = committed_record_cursor == 0

        def record_progress_value(
            ordinal: int, within_record: float = 0.0
        ) -> float:
            if record_count_hint is not None and record_count_hint > 0:
                return min(
                    0.99,
                    max(
                        0.0,
                        (max(0, ordinal - 1) + max(0.0, within_record))
                        / float(record_count_hint),
                    ),
                )
            if resolved_kind in {"text", "json", "jsonl", "csv", "tsv"}:
                return min(
                    0.99,
                    max(
                        0.0,
                        int(coverage.processed_bytes)
                        / float(max(1, source_bytes)),
                    ),
                )
            # Compressed/archive bytes and decoded media units are not a valid
            # denominator. Stay honest until the entry is fully traversed.
            return 0.0

        def record_position(ordinal: int) -> str:
            if record_count_hint is not None:
                return "%d/%d" % (ordinal, max(0, record_count_hint))
            return "%d (total discovered while streaming)" % ordinal

        progress_resource_readings: Dict[str, Any] = {}
        progress_resource_readings_at = 0.0

        def dataset_progress_data(
            *, checkpoint_committed: bool = False
        ) -> Dict[str, Any]:
            nonlocal progress_resource_readings
            nonlocal progress_resource_readings_at
            resource_fields = {
                "processMemoryBytes",
                "processPeakMemoryBytes",
                "availableMemoryBytes",
                "totalMemoryBytes",
                "acceleratorFreeMemoryBytes",
                "acceleratorTotalMemoryBytes",
                "diskFreeBytes",
                "diskTotalBytes",
                "diskReserveBytes",
                "mandatoryFreeDiskBytes",
                "projectedDiskFreeBytes",
                "estimatedWriteBytes",
            }
            measured_at = time.monotonic()
            if (
                progress_resource_readings_at <= 0.0
                or measured_at - progress_resource_readings_at >= 1.0
            ):
                live_policy = self.resource_policy.status(
                    estimated_write_bytes=max(
                        0,
                        int(
                            neural_storage_plan
                            .get("trainingResourcePlan", {})
                            .get("scratch", {})
                            .get("estimatedCheckpointBytes", 0)
                        ),
                    )
                )
                progress_resource_readings = {
                    key: value
                    for key, value in live_policy.items()
                    if key in resource_fields
                    and not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                }
                if isinstance(live_policy.get("diskPressure"), bool):
                    progress_resource_readings["diskPressure"] = bool(
                        live_policy["diskPressure"]
                    )
                progress_resource_readings_at = measured_at
            if bool(progress_resource_readings.get("diskPressure")):
                raise NeuralStateResourcePause(
                    "dataset learning paused at the mandatory disk reserve",
                    dict(progress_resource_readings),
                )
            return {
                "datasetProgress": {
                    "coverage": coverage.as_dict(),
                    "currentRecord": int(record_cursor),
                    "committedRecords": int(last_committed_record_cursor),
                    "checkpointCommitted": bool(checkpoint_committed),
                    "expectedRecords": (
                        int(record_count_hint)
                        if record_count_hint is not None
                        else None
                    ),
                    "recordTotalKnown": record_count_hint is not None,
                },
                "resourceReadings": dict(progress_resource_readings),
            }

        def restore_committed_coverage() -> None:
            nonlocal resume_boundary_validated
            if checkpoint is None or resume_boundary_validated:
                return
            stored = checkpoint.get("coverageAtCommit")
            if not isinstance(stored, Mapping):
                raise ValueError("ingestion checkpoint coverage snapshot is invalid")
            if int(coverage.discovered_records) != int(
                checkpoint["visitedRecords"]
            ):
                raise ValueError(
                    "dataset record stream no longer matches its committed checkpoint"
                )
            if record_prefix_sha256 != str(
                checkpoint["recordPrefixSha256"]
            ):
                raise ValueError(
                    "dataset record content no longer matches its committed checkpoint"
                )
            coverage.discovered_files = int(stored.get("discoveredFiles", 0))
            coverage.completed_files = int(stored.get("completedFiles", 0))
            coverage.rejected_files = int(stored.get("rejectedFiles", 0))
            coverage.discovered_records = int(stored.get("discoveredRecords", 0))
            coverage.processed_records = int(stored.get("processedRecords", 0))
            coverage.rejected_records = int(stored.get("rejectedRecords", 0))
            coverage.processed_bytes = int(stored.get("processedBytes", 0))
            coverage.shards = int(stored.get("shards", 0))
            coverage.modality_counts = {
                str(key): int(value)
                for key, value in dict(
                    stored.get("modalityCounts", {})
                ).items()
            }
            coverage.errors = [
                dict(value)
                for value in stored.get("errors", [])
                if isinstance(value, Mapping)
            ][: coverage._ERROR_SAMPLE_LIMIT]
            coverage.error_count = int(stored.get("errorCount", 0))
            coverage.errors_truncated = bool(
                stored.get("errorsTruncated", False)
            )
            resume_boundary_validated = True

        def commit_record_checkpoint() -> None:
            nonlocal checkpoint
            nonlocal commit_sequence
            nonlocal last_committed_record_cursor
            nonlocal capability_rehearsal_state
            if not checkpoint_key or record_cursor <= last_committed_record_cursor:
                return
            verify_source_snapshot()
            # Pending raw strings are consumed by their neural update before
            # the cursor can move.  Only aggregate numbers/hashes survive.
            flush_streaming_local()
            coverage_snapshot = coverage.as_dict()
            commit_sequence += 1
            finite_middle = capability_rehearsal_cadence.get("middleWave")
            middle_due = bool(
                capability_rehearsal_enabled
                and (
                    (
                        capability_rehearsal_cadence["mode"]
                        == "finite-midpoint"
                        and finite_middle is not None
                        and commit_sequence == int(finite_middle)
                        and capability_rehearsal_state.last_periodic_wave == 0
                    )
                    or (
                        capability_rehearsal_cadence["mode"]
                        == "indefinite-periodic"
                        and due_rehearsal_phase(
                            capability_rehearsal_state,
                            capability_rehearsal_policy,
                            committed_global_waves=commit_sequence,
                        )
                        == "middle"
                    )
                )
            )
            if middle_due:
                capability_receipt = rehearse_capabilities(
                    self,
                    phase="middle",
                    committed_global_waves=commit_sequence,
                    policy=capability_rehearsal_policy,
                    baseline_minimum_probability=(
                        capability_rehearsal_state.baseline_minimum_probability
                    ),
                )
                capability_rehearsal_state = advance_schedule_state(
                    capability_rehearsal_state, capability_receipt
                )
            baseline = {
                "parameterChecksum": before_checksum,
                "concepts": before_concepts,
                "ideas": before_ideas,
                "plasticityEvents": before_events,
                "memoryNeurons": before_memory_neurons,
                "memorySynapses": before_memory_synapses,
                "memorySynapticUses": before_memory_synaptic_uses,
                "trainingSteps": before_training_steps,
                "statisticalExperiences": before_statistical_experiences,
            }
            v3_checkpoint = learning_schedule.get("formatVersion") == 3
            checkpoint_coverage = dict(coverage_snapshot)
            if v3_checkpoint:
                # Detailed diagnostics can retain source paths/error text in
                # RAM for the UI, but a neural cursor persists only counts,
                # hashes, and fixed protocol metadata.
                checkpoint_coverage["errors"] = []
                checkpoint_coverage["errorsTruncated"] = bool(
                    checkpoint_coverage.get("errorCount", 0)
                )
            checkpoint = {
                "format": INGESTION_CHECKPOINT_FORMAT,
                "formatVersion": (
                    3 if v3_checkpoint else INGESTION_CHECKPOINT_VERSION
                ),
                "parserContract": INGESTION_PARSER_CONTRACT,
                "status": "active",
                "sourceIdentity": checkpoint_key,
                "transactionId": transaction_id,
                "contentHash": content_hash,
                "sourceNameHash": source_name_hash,
                "neuralStateChecksum": self.parameter_checksum(),
                "recordPrefixSha256": record_prefix_sha256,
                "sourceSnapshot": dict(source_snapshot or {}),
                "sourceBytes": source_bytes,
                "resolvedKind": resolved_kind,
                "policy": policy,
                "epoch": max(0, int(epoch)),
                "committedRecords": record_cursor,
                "visitedRecords": int(coverage_snapshot["discoveredRecords"]),
                "processedRecords": int(coverage_snapshot["processedRecords"]),
                "rejectedRecords": int(coverage_snapshot["rejectedRecords"]),
                "processedBytes": int(coverage_snapshot["processedBytes"]),
                "commitSequence": commit_sequence,
                "coverageAtCommit": checkpoint_coverage,
                "learningSchedule": dict(learning_schedule),
                "learningScheduleSha256": schedule_sha256(learning_schedule),
                "baseline": baseline,
                "aggregate": {
                    "lossTotal": float(loss_total),
                    "learnedChunks": int(learned_chunks),
                    "readingReportCount": int(reading_report_count),
                    "streamingGradientRecords": int(
                        streaming_gradient_records
                    ),
                    "streamingGradientOptimizerSteps": int(
                        streaming_gradient_optimizer_steps
                    ),
                    "mediaAccumulator": self._checkpoint_media_accumulator(
                        media_accumulator
                    ),
                },
                "capabilityRehearsal": (
                    capability_rehearsal_state.to_dict()
                    if capability_rehearsal_enabled
                    else None
                ),
                "capabilityRehearsalCadence": (
                    dict(capability_rehearsal_cadence)
                    if capability_rehearsal_enabled
                    else None
                ),
                "committedAt": _iso_now(),
            }
            if v3_checkpoint:
                checkpoint.update({
                    "sourceManifestSha256": source_manifest_sha256,
                    "parserManifestSha256": parser_manifest_sha256,
                    "expectedRecords": record_count_hint,
                    # Populated only after immutable neural+substrate staging,
                    # before the single authoritative brain.json replacement.
                    "pagedCheckpointBinding": None,
                    "pagedCheckpointBindingSha256": None,
                })
            self.ingestion_checkpoints[checkpoint_key] = checkpoint
            # save() publishes neural tensors, optimizer/replay state, and this
            # cursor through one generation pointer plus one atomic brain.json
            # replacement. A crash observes either the prior batch or this one.
            self._clear_allocator_recovery_pause()
            self.save()
            last_committed_record_cursor = record_cursor
            self.events.append(
                "ingestion-batch-commit",
                {
                    "transactionId": transaction_id,
                    "contentHash": content_hash,
                    "epoch": max(0, int(epoch)),
                    "policy": policy,
                    "commitSequence": commit_sequence,
                    "committedRecords": record_cursor,
                    "visitedRecords": int(
                        coverage_snapshot["discoveredRecords"]
                    ),
                    "processedRecords": int(
                        coverage_snapshot["processedRecords"]
                    ),
                    "rejectedRecords": int(
                        coverage_snapshot["rejectedRecords"]
                    ),
                    "parameterChecksum": self.parameter_checksum(),
                    "rawSourceRetained": False,
                    "rawTokenIdsRetained": False,
                },
            )
            if progress is not None:
                progress(
                    record_progress_value(record_cursor, 1.0),
                    "Committed %d learned records (%d visited, %d rejected)"
                    % (
                        record_cursor,
                        int(coverage_snapshot["discoveredRecords"]),
                        int(coverage_snapshot["rejectedRecords"]),
                    ),
                    dataset_progress_data(checkpoint_committed=True),
                )

        def maybe_commit_record_checkpoint() -> None:
            if (
                checkpoint_key
                and record_cursor - last_committed_record_cursor
                >= int(learning_schedule["checkpointRecords"])
            ):
                commit_record_checkpoint()
        if policy != "archive":
            if source_path is not None:
                record_stream = iter_dataset_records(
                    source_path,
                    requested_kind=resolved_kind,
                    coverage=coverage,
                    _committed_sqlite_snapshot_sha256=(
                        content_hash
                        if resolved_kind == "sqlite"
                        and committed_sqlite_snapshot
                        else ""
                    ),
                )
            else:
                coverage.discovered_files = 1
                coverage.completed_files = 1
                coverage.discovered_records = 1 if extracted.strip() else 0
                coverage.processed_records = 1 if extracted.strip() else 0
                coverage.processed_bytes = source_bytes
                record_stream = []

            for record in record_stream:
                if resolved_kind == "sqlite":
                    record_snapshot_hash = str(
                        dict(getattr(record, "provenance", {})).get(
                            "sqlite_snapshot_sha256", ""
                        )
                    )
                    if record_snapshot_hash != content_hash:
                        raise ValueError(
                            "SQLite traversal snapshot does not match its "
                            "ingestion transaction identity"
                        )
                record_cursor += 1
                record_prefix_sha256 = self._record_prefix_digest(
                    record_prefix_sha256, record_cursor, record
                )
                record_kind = str(getattr(record, "kind", "text"))
                record_name = str(getattr(record, "name", source_name))
                if record_cursor <= committed_record_cursor:
                    # Parsing the prefix again validates the deterministic
                    # record boundary, but no committed record is presented to
                    # any neural learner a second time. Exact source bytes are
                    # owned by the desktop CAS, never rebuilt inside the worker.
                    if record_cursor == committed_record_cursor:
                        restore_committed_coverage()
                    continue
                if record_kind in {"image", "audio", "video"}:
                    record_path = getattr(record, "local_path", None)
                    if not record_path:
                        failure_message = (
                            "Binary modality record has no leased local path."
                        )
                        coverage.reject_processed(
                            record_name,
                            failure_message,
                        )
                        failure_coverage = self._empty_media_coverage(record_kind)
                        self._accumulate_media_report(
                            media_accumulator,
                            {
                                "name": record_name,
                                "kind": record_kind,
                                "contentSha256": str(
                                    getattr(record, "content_sha256", "")
                                ),
                                "provenance": dict(
                                    getattr(record, "provenance", {})
                                ),
                                "trained": False,
                                "loss": 0.0,
                                "steps": 0,
                                "coverage": failure_coverage,
                                "warnings": [failure_message],
                            },
                        )
                        maybe_commit_record_checkpoint()
                        continue
                    effective_kind = self._effective_media_kind(
                        str(record_path), record_kind
                    )
                    if effective_kind != record_kind:
                        prior_count = int(
                            coverage.modality_counts.get(record_kind, 0)
                        )
                        if prior_count <= 1:
                            coverage.modality_counts.pop(record_kind, None)
                        else:
                            coverage.modality_counts[record_kind] = prior_count - 1
                        coverage.modality_counts[effective_kind] = (
                            coverage.modality_counts.get(effective_kind, 0) + 1
                        )
                        record_kind = effective_kind
                    # Archive and remote-manifest paths are leases. They must be
                    # decoded and trained completely before the iterator is
                    # advanced, because advancing removes the temporary file.
                    try:
                        trained_media = self._train_media(
                            str(record_path),
                            record_kind,
                            record_name,
                            steps=3 if policy == "pretrain" else 2,
                            progress=progress,
                            content_sha256=str(
                                getattr(record, "content_sha256", "")
                            ),
                        )
                    except (RuntimeError, ValueError, OSError) as error:
                        if is_allocator_oom_error(error):
                            self._allocator_oom_count += 1
                            self._release_training_allocator_cache()
                            raise self._allocator_resource_pause(
                                error,
                                stage="modality-training",
                            ) from error
                        trained_media = {
                            "trained": False,
                            "loss": 0.0,
                            "steps": 0,
                            "coverage": self._empty_media_coverage(record_kind),
                            "warnings": [str(error)],
                        }
                    if not bool(trained_media.get("trained", False)):
                        training_warnings = [
                            str(value)
                            for value in trained_media.get("warnings", [])
                            if str(value).strip()
                        ]
                        coverage.reject_processed(
                            record_name,
                            "; ".join(training_warnings)
                            or "%s media decoding/training failed" % record_kind,
                        )
                    self._accumulate_media_report(
                        media_accumulator,
                        {
                            "name": record_name,
                            "kind": record_kind,
                            "contentSha256": str(
                                getattr(record, "content_sha256", "")
                            ),
                            "provenance": dict(
                                getattr(record, "provenance", {})
                            ),
                            **trained_media,
                        },
                    )
                    maybe_commit_record_checkpoint()
                    continue

                record_text = str(getattr(record, "text", ""))
                record_ordinal = max(1, int(coverage.processed_records))
                record_provenance = dict(
                    getattr(record, "provenance", {}) or {}
                )
                dialogue_pairs = self._typed_dialogue_pairs(record_provenance)
                record_chunks = self._experience_chunks(record_text)
                chunk_importances = self._reading_chunk_importances(record_chunks)
                section_assemblies: List[str] = []
                for section_index, (chunk, chunk_importance) in enumerate(
                    zip(record_chunks, chunk_importances), start=1
                ):
                    if progress is not None:
                        progress(
                            record_progress_value(
                                record_ordinal,
                                0.02
                                + 0.78
                                * (section_index - 1)
                                / float(max(1, len(record_chunks))),
                            ),
                            "Learning %s record %s, section %d/%d"
                            % (
                                record_name,
                                record_position(record_ordinal),
                                section_index,
                                len(record_chunks),
                            ),
                            dataset_progress_data(),
                        )
                    learned = self.learn_experience(
                        chunk,
                        kind="knowledge",
                        source="document",
                        source_label=record_name,
                        steps=(
                            0
                            if compact_streaming
                            else (2 if policy == "pretrain" else 1)
                        ),
                        importance=chunk_importance,
                        structural_detail=bool(
                            neural_storage_plan["detailedRecordAssemblies"]
                        ),
                    )
                    section_assemblies.append(str(learned["assembly_id"]))
                    if compact_streaming:
                        queue_streaming_local(chunk)
                    else:
                        loss_total += float(learned["training"]["loss"])
                    learned_chunks += 1
                integrated_record = (
                    self._integrate_reading_record(
                        record_text,
                        source_name=record_name,
                        child_assembly_ids=section_assemblies,
                        child_weights=chunk_importances,
                    )
                    if bool(
                        neural_storage_plan["detailedRecordAssemblies"]
                    )
                    else None
                )
                if integrated_record is not None:
                    reading_report_count += 1
                if dialogue_pairs:
                    for human, response in dialogue_pairs:
                        self._apply_supervised_dialogue(
                            human,
                            response,
                            steps=2 if policy == "pretrain" else 1,
                            train_local=not compact_streaming,
                            local_exact_response_windows=(
                                int(learning_schedule["formatVersion"])
                                in {INGESTION_LEARNING_SCHEDULE_VERSION, 3}
                                and learning_schedule.get(
                                    "localTypedTargetWindowPolicy"
                                )
                                == LOCAL_TYPED_TARGET_WINDOW_POLICY
                            ),
                            local_exact_training_sequence_tokens=(
                                int(learning_schedule["trainingSequenceTokens"])
                                if int(learning_schedule["formatVersion"])
                                in {INGESTION_LEARNING_SCHEDULE_VERSION, 3}
                                else None
                            ),
                            local_exact_target_window_policy=(
                                str(
                                    learning_schedule[
                                        "localTypedTargetWindowPolicy"
                                    ]
                                )
                                if int(learning_schedule["formatVersion"])
                                in {INGESTION_LEARNING_SCHEDULE_VERSION, 3}
                                else None
                            ),
                        )
                        if compact_streaming:
                            dialogue_text = "%s\n\n%s" % (human, response)
                            queue_streaming_local(dialogue_text)
                if progress is not None:
                    progress(
                        record_progress_value(record_ordinal, 1.0),
                        "Encoding %s (record %s)"
                        % (source_name, record_position(record_ordinal)),
                        dataset_progress_data(),
                    )
                maybe_commit_record_checkpoint()
            if record_cursor < committed_record_cursor:
                raise ValueError(
                    "dataset ended before its committed ingestion checkpoint"
                )
            if source_path is None and resolved_kind in {
                "image",
                "audio",
                "video",
            }:
                missing_path_message = (
                    "Binary modality ingestion requires a local file path."
                )
                if coverage.processed_records > 0:
                    coverage.reject_processed(
                        source_name,
                        missing_path_message,
                    )
                else:
                    coverage.reject(source_name, missing_path_message)
                self._accumulate_media_report(
                    media_accumulator,
                    {
                        "name": source_name,
                        "kind": resolved_kind,
                        "contentSha256": content_hash,
                        "provenance": {},
                        "trained": False,
                        "loss": 0.0,
                        "steps": 0,
                        "coverage": self._empty_media_coverage(resolved_kind),
                        "warnings": [missing_path_message],
                    },
                )
            elif source_path is None and extracted.strip():
                extracted_chunks = self._experience_chunks(extracted)
                extracted_importances = self._reading_chunk_importances(
                    extracted_chunks
                )
                extracted_assemblies: List[str] = []
                for section_index, (chunk, chunk_importance) in enumerate(
                    zip(extracted_chunks, extracted_importances), start=1
                ):
                    if progress is not None:
                        progress(
                            min(
                                0.82,
                                0.02
                                + 0.78
                                * (section_index - 1)
                                / float(max(1, len(extracted_chunks))),
                            ),
                            "Learning %s section %d/%d"
                            % (source_name, section_index, len(extracted_chunks)),
                        )
                    learned = self.learn_experience(
                        chunk,
                        kind="knowledge",
                        source="document",
                        source_label=source_name,
                        steps=(
                            0
                            if compact_streaming
                            else (2 if policy == "pretrain" else 1)
                        ),
                        importance=chunk_importance,
                        structural_detail=bool(
                            neural_storage_plan["detailedRecordAssemblies"]
                        ),
                    )
                    extracted_assemblies.append(str(learned["assembly_id"]))
                    if compact_streaming:
                        queue_streaming_local(chunk)
                    else:
                        loss_total += float(learned["training"]["loss"])
                    learned_chunks += 1
                integrated_record = (
                    self._integrate_reading_record(
                        extracted,
                        source_name=source_name,
                        child_assembly_ids=extracted_assemblies,
                        child_weights=extracted_importances,
                    )
                    if bool(
                        neural_storage_plan["detailedRecordAssemblies"]
                    )
                    else None
                )
                if integrated_record is not None:
                    reading_report_count += 1
        else:
            coverage.discovered_files = 1
            coverage.completed_files = 1
            coverage.processed_bytes = source_bytes
        self._require_complete_ingestion_coverage(
            coverage,
            policy=policy,
            resolved_kind=resolved_kind,
            record_count_hint=record_count_hint,
        )
        verify_source_snapshot(full_hash=True)
        # Commit the final partial accumulation only after every visited record
        # has entered fast substrate/statistical state. These calls retain no
        # raw source buffers after returning.
        flush_streaming_local()
        media_result = self._media_result_from_accumulator(
            media_accumulator,
            complete_when_empty=policy == "archive",
        )
        if progress is not None:
            progress(1.0, "Encoded %s" % source_name)
        dataset_warnings = [
            "%s: %s" % (entry["source"], entry["message"])
            for entry in coverage.errors
        ]
        media_result["warnings"] = [
            *list(media_result["warnings"]),
            *dataset_warnings,
        ]
        effective_source_kind = resolved_kind
        if (
            resolved_kind == "image"
            and int(media_result["coverage"].get("records", 0)) == 1
            and set(media_result["coverage"].get("byModality", {}))
            == {"video"}
        ):
            effective_source_kind = "video"
        semantic_neurons_created = max(
            0, len(self.memory.neurons) - before_memory_neurons
        )
        sparse_synapses_created = max(
            0, len(self.memory.synapses) - before_memory_synapses
        )
        memory_synapse_update_events = max(
            0,
            self.memory.synaptic_use_count()
            - before_memory_synaptic_uses,
        )
        statistical_experience_updates = max(
            0,
            sum(
                int(value.get("statistical_experiences", 0))
                for value in self.memory.assemblies
            )
            - before_statistical_experiences,
        )
        spike_plasticity_events = max(
            0,
            int(self.router.synapses.plasticity_events.item()) - before_events,
        )
        # This is a count of observed synaptic mutation events, not a count of
        # dense optimizer records. Sparse Hebbian reinforcements and router
        # STDP stay independently auditable.
        synaptic_update_events = (
            memory_synapse_update_events
            + spike_plasticity_events
        )
        parameter_update_steps = max(
            0, int(self.counters["training_steps"]) - before_training_steps
        )
        neural_update_events = synaptic_update_events + parameter_update_steps
        source_record: Dict[str, Any] = {
            "id": (
                str(duplicate.get("id"))
                if duplicate is not None
                else uuid.uuid4().hex
            ),
            "name": source_name,
            "kind": effective_source_kind,
            "bytes": source_bytes,
            "content_hash": content_hash,
            "policy": policy,
            "imported_at": (
                str(duplicate.get("imported_at", _iso_now()))
                if duplicate is not None
                else _iso_now()
            ),
            "last_trained_at": _iso_now(),
            "training_epochs": (
                int(duplicate.get("training_epochs", 1)) + 1
                if duplicate is not None
                else 1
            ),
            "last_epoch": max(0, int(epoch)),
            "transaction_id": transaction_id or None,
            "transactionId": transaction_id or None,
            "source_identity": checkpoint_key or None,
            "source_name_hash": source_name_hash,
            "learned_ideas": len(self.memory.ideas) - before_ideas,
            "learned_concepts": len(self.memory.concepts) - before_concepts,
            # Compatibility field now reports all durable neural update events,
            # not only router STDP. Large statistical corpora can reinforce an
            # existing assembly millions of times without inventing millions
            # of fake new ideas.
            "plasticity_events": neural_update_events,
            "neural_update_events": neural_update_events,
            "synaptic_update_events": synaptic_update_events,
            "memory_synapse_update_events": memory_synapse_update_events,
            "parameter_update_steps": parameter_update_steps,
            "spike_plasticity_events": spike_plasticity_events,
            "semantic_neurons_created": semantic_neurons_created,
            "sparse_synapses_created": sparse_synapses_created,
            "statistical_experience_updates": statistical_experience_updates,
            "raw_text_retained": False,
            "modality_trained": bool(media_result["trained"]),
            "media_coverage": dict(media_result["coverage"]),
            "media_records": list(media_result["records"]),
            "warnings": list(media_result["warnings"]),
            "coverage": coverage.as_dict(),
            "streaming_gradient_records": streaming_gradient_records,
            "streaming_gradient_optimizer_steps": (
                streaming_gradient_optimizer_steps
            ),
            "whole_record_assemblies": reading_report_count,
            "record_batch_commits": commit_sequence,
            "resumed_record_count": resumed_record_count,
            "neural_storage_plan": {
                key: value
                for key, value in neural_storage_plan.items()
                if key != "resourceStatus"
            },
        }
        if duplicate is None:
            self.training_sources.append(source_record)
        else:
            self.training_sources = [
                source_record
                if source.get("id") == duplicate.get("id")
                else source
                for source in self.training_sources
            ]
        # ``consolidate`` is retained as an ingestion-policy spelling for API
        # compatibility, but it no longer starts a separate, manually bounded
        # post-pass here.  Every accepted experience has already gone through
        # the continuously scheduled neural settling path.  Starting another
        # save-owning phase between the last record checkpoint and the final
        # completion receipt would make that phase repeatable after a crash and
        # could publish a cursor for a different neural generation.
        action_policy_refresh: Optional[Dict[str, Any]] = None
        capability_rehearsal_receipt: Optional[Dict[str, Any]] = None
        if capability_rehearsal_enabled:
            final_phase = due_rehearsal_phase(
                capability_rehearsal_state,
                capability_rehearsal_policy,
                committed_global_waves=commit_sequence,
                final=True,
            )
            if final_phase != "final":
                raise RuntimeError(
                    "ground-up pretrain capability final gate was not scheduled"
                )
            capability_rehearsal_receipt = rehearse_capabilities(
                self,
                phase="final",
                committed_global_waves=commit_sequence,
                policy=capability_rehearsal_policy,
                baseline_minimum_probability=(
                    capability_rehearsal_state.baseline_minimum_probability
                ),
            )
            capability_rehearsal_state = advance_schedule_state(
                capability_rehearsal_state,
                capability_rehearsal_receipt,
            )
            action_policy_refresh = dict(
                capability_rehearsal_receipt["action"]
            )
            source_record["capability_rehearsal"] = (
                capability_rehearsal_state.to_dict()
            )
        elif policy != "archive" and self._can_retain_native_action_policy():
            # Corpus learning changes the representations feeding both action
            # heads even when those heads receive no direct gradient. Rehearse
            # the project-authored typed trajectories once after the complete
            # ingestion transaction so ordinary conversation does not drift
            # into a high-confidence Ponder/tool proposal. This is neural
            # trajectory replay, not a runtime prompt or preference objective.
            action_policy_refresh = self._calibrate_starter_action_policy(
                max_steps=256,
                minimum_steps=0,
                strict=False,
            )
            if not bool(action_policy_refresh.get("calibrated", False)):
                source_record["warnings"].append(
                    "Typed action pathways could not be refreshed after learning; "
                    "the prior action heads were restored."
                )
        elif policy == "pretrain" and self.config.origin_kind == "ground-up":
            raise RuntimeError(
                "ground-up pretrain cannot authenticate its capability curriculum"
            )
        parameter_update_steps = max(
            0, int(self.counters["training_steps"]) - before_training_steps
        )
        source_record["parameter_update_steps"] = parameter_update_steps
        source_record["action_policy_parameter_steps"] = int(
            (action_policy_refresh or {}).get("steps", 0)
        )
        source_record["neural_update_events"] = (
            synaptic_update_events + parameter_update_steps
        )
        # Retain the old field solely for stable callers. New callers separate
        # synaptic mutations from slow-parameter optimizer steps.
        source_record["plasticity_events"] = source_record[
            "neural_update_events"
        ]
        completion_parameter_checksum = self.parameter_checksum()
        source_record["parameter_checksum_before"] = before_checksum
        source_record["parameter_checksum_after"] = (
            completion_parameter_checksum
        )
        source_record["parameter_checksum_changed"] = (
            completion_parameter_checksum != before_checksum
        )
        # Completion is itself atomic: the same brain generation both exposes
        # the final source record and removes the active cursor. If promotion
        # fails, load() sees the preceding active checkpoint and resumes.
        if transaction_id:
            completion_receipt = {
                "format": "omni-completed-ingestion",
                "formatVersion": 1,
                "transactionId": transaction_id,
                "contentHash": content_hash,
                "sourceIdentity": checkpoint_key,
                "sourceNameHash": source_name_hash,
                "sourceId": str(source_record["id"]),
                "resolvedKind": resolved_kind,
                "policy": policy,
                "epoch": max(0, int(epoch)),
                "coverage": coverage.as_dict(),
                "committedRecords": record_cursor,
                "visitedRecords": int(coverage.discovered_records),
                "rejectedRecords": int(coverage.rejected_records),
                "batchCommits": commit_sequence,
                "parameterChecksumAfter": completion_parameter_checksum,
                "completedAt": _iso_now(),
                "rawSourceTextStored": False,
                "rawTokenIdsStored": False,
                "capabilityRehearsal": (
                    capability_rehearsal_state.to_dict()
                    if capability_rehearsal_enabled
                    else None
                ),
            }
            source_record["completionReceipt"] = completion_receipt
            self.completed_ingestions = [
                value
                for value in self.completed_ingestions
                if value.get("transactionId") != transaction_id
            ]
            self.completed_ingestions.append(completion_receipt)
            self.completed_ingestions = self.completed_ingestions[
                -COMPLETED_INGESTION_TOMBSTONES:
            ]
        if checkpoint_key:
            self.ingestion_checkpoints.pop(checkpoint_key, None)
        verify_source_snapshot()
        self._clear_allocator_recovery_pause()
        self.save()
        after_checksum = self.parameter_checksum()
        self.events.append(
            "ingestion",
            {
                "sourceId": source_record["id"],
                "contentHash": content_hash,
                "kind": resolved_kind,
                "policy": policy,
                "learnedIdeas": source_record["learned_ideas"],
                "learnedConcepts": source_record["learned_concepts"],
                "synapticUpdateEvents": source_record[
                    "synaptic_update_events"
                ],
                "memorySynapseUpdateEvents": source_record[
                    "memory_synapse_update_events"
                ],
                "parameterUpdateSteps": source_record[
                    "parameter_update_steps"
                ],
                "parameterChecksumBefore": before_checksum,
                "parameterChecksumAfter": after_checksum,
                "parameterChecksumChanged": before_checksum != after_checksum,
                "rawTextRetained": source_record["raw_text_retained"],
                "modalityTrained": source_record["modality_trained"],
                "mediaCoverage": source_record["media_coverage"],
                "warnings": source_record["warnings"],
                "streamingGradientRecords": source_record[
                    "streaming_gradient_records"
                ],
                "streamingGradientOptimizerSteps": source_record[
                    "streaming_gradient_optimizer_steps"
                ],
                "wholeRecordAssemblies": source_record[
                    "whole_record_assemblies"
                ],
                "recordBatchCommits": source_record["record_batch_commits"],
                "resumedRecordCount": source_record["resumed_record_count"],
                "visitedRecords": int(coverage.discovered_records),
                "committedRecords": record_cursor,
                "actionPolicyRefresh": action_policy_refresh,
            },
        )
        return {
            "brainId": self.brain_id,
            "transactionId": transaction_id or None,
            "duplicate": False,
            "source": source_record,
            "meanLoss": loss_total / learned_chunks if learned_chunks else 0.0,
            "modalityLoss": float(media_result["loss"]),
            "mediaCoverage": dict(media_result["coverage"]),
            "warnings": list(media_result["warnings"]),
            "coverage": coverage.as_dict(),
            "streamingGradientTraining": {
                "records": source_record["streaming_gradient_records"],
                "optimizerSteps": source_record[
                    "streaming_gradient_optimizer_steps"
                ],
                "mode": source_record["neural_storage_plan"][
                    "slowGradientMode"
                ],
                "everyQueuedRecordContributed": True,
            },
            "readingIntegration": {
                "wholeRecordAssemblies": source_record[
                    "whole_record_assemblies"
                ],
                "everySectionVisited": True,
                "sectionWeights": "automatic-neural-salience",
                "rawSourceTextStored": False,
            },
            "neuralStoragePlan": source_record["neural_storage_plan"],
            "recordRecovery": {
                "transactionId": transaction_id or None,
                "resumed": resumed_record_count > 0,
                "resumedRecords": resumed_record_count,
                "committedRecords": record_cursor,
                "visitedRecords": int(coverage.discovered_records),
                "rejectedRecords": int(coverage.rejected_records),
                "batchCommits": commit_sequence,
                "checkpointActive": False,
                "rawSourceTextStored": False,
                "rawTokenIdsStored": False,
            },
            "synapticUpdateEvents": source_record[
                "synaptic_update_events"
            ],
            "parameterUpdateSteps": source_record[
                "parameter_update_steps"
            ],
            "parameterChecksumChanged": before_checksum != after_checksum,
            "actionPolicyRefresh": action_policy_refresh,
            "parameterChecksumBefore": before_checksum,
            "parameterChecksumAfter": after_checksum,
            "metrics": self.metrics(),
        }

    @staticmethod
    def _ppm_bytes(image: torch.Tensor) -> bytes:
        value = image.detach().cpu().float()
        if value.ndim == 4:
            value = value[0]
        value = ((value.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        value = value.permute(1, 2, 0).contiguous()
        height, width = value.shape[:2]
        return (
            ("P6\n%d %d\n255\n" % (width, height)).encode("ascii")
            + value.numpy().tobytes()
        )

    @staticmethod
    def _png_chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    @classmethod
    def _png_bytes(cls, image: torch.Tensor) -> bytes:
        value = image.detach().cpu().float()
        if value.ndim == 4:
            value = value[0]
        value = ((value.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        value = value.permute(1, 2, 0).contiguous()
        height, width = value.shape[:2]
        raw = b"".join(
            b"\x00" + value[row].numpy().tobytes() for row in range(height)
        )
        return (
            b"\x89PNG\r\n\x1a\n"
            + cls._png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            )
            + cls._png_chunk(b"IDAT", zlib.compress(raw, level=6))
            + cls._png_chunk(b"IEND", b"")
        )

    @classmethod
    def _apng_bytes(cls, video: torch.Tensor, fps: int = 8) -> bytes:
        if isinstance(fps, bool) or not 1 <= int(fps) <= 65_535:
            raise ValueError("APNG FPS must fit its positive 16-bit timebase")
        fps = int(fps)
        value = video.detach().cpu().float()
        if value.ndim == 5:
            value = value[0]
        value = ((value.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        value = value.permute(1, 2, 3, 0).contiguous()
        frames, height, width = value.shape[:3]
        output = [
            b"\x89PNG\r\n\x1a\n",
            cls._png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            cls._png_chunk(b"acTL", struct.pack(">II", frames, 0)),
        ]
        sequence = 0
        for index in range(frames):
            output.append(
                cls._png_chunk(
                    b"fcTL",
                    struct.pack(
                        ">IIIIIHHBB",
                        sequence,
                        width,
                        height,
                        0,
                        0,
                        1,
                        max(1, fps),
                        0,
                        0,
                    ),
                )
            )
            sequence += 1
            raw = b"".join(
                b"\x00" + value[index, row].numpy().tobytes()
                for row in range(height)
            )
            compressed = zlib.compress(raw, level=6)
            if index == 0:
                output.append(cls._png_chunk(b"IDAT", compressed))
            else:
                output.append(
                    cls._png_chunk(
                        b"fdAT", struct.pack(">I", sequence) + compressed
                    )
                )
                sequence += 1
        output.append(cls._png_chunk(b"IEND", b""))
        return b"".join(output)

    @classmethod
    def _mp4_bytes(
        cls,
        video: torch.Tensor,
        fps: int = 8,
        audio: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> bytes:
        """Encode H.264 MP4 and, when supplied, mux same-idea audio as AAC."""

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("MP4 output requires a configured external FFmpeg runtime") from error
        value = video.detach().cpu().float()
        if value.ndim == 5:
            value = value[0]
        if value.ndim != 4 or value.shape[0] != 3:
            raise ValueError("video output must have shape [3, frames, height, width]")
        value = ((value.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        frames = value.permute(1, 2, 3, 0).contiguous()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        frame_rate = int(fps)
        if not 1 <= frame_rate <= 65_535:
            raise ValueError("video FPS must fit the portable container timebase")
        with tempfile.TemporaryDirectory(prefix="omni-video-output-") as temporary:
            output = Path(temporary) / "generated.mp4"
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                "%dx%d" % (width, height),
                "-r",
                str(frame_rate),
                "-i",
                "-",
            ]
            if audio is not None:
                audio_path = Path(temporary) / "same-brain-audio.wav"
                audio_path.write_bytes(
                    cls._wav_bytes(audio, sample_rate=sample_rate)
                )
                command.extend(
                    [
                        "-i",
                        str(audio_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                    ]
                )
            else:
                command.append("-an")
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ]
            )
            if audio is not None:
                command.extend(
                    [
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-shortest",
                    ]
                )
            command.append(str(output))
            subprocess.run(
                command,
                input=frames.numpy().tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )
            encoded = output.read_bytes()
        if len(encoded) < 12 or encoded[4:8] != b"ftyp":
            raise RuntimeError("FFmpeg did not produce a valid MP4 container")
        if audio is not None and b"soun" not in encoded:
            raise RuntimeError("FFmpeg MP4 output did not contain an audio track")
        return encoded

    @staticmethod
    def _wav_bytes(waveform: torch.Tensor, sample_rate: int = 16000) -> bytes:
        if isinstance(sample_rate, bool) or not 1 <= int(sample_rate) <= 0xFFFFFFFF:
            raise ValueError("WAV sample rate must fit its positive 32-bit field")
        sample_rate = int(sample_rate)
        value = waveform.detach().cpu().float().reshape(-1).clamp(-1, 1)
        samples = array.array("h", (value * 32767.0).to(torch.int16).tolist())
        if os.sys.byteorder != "little":
            samples.byteswap()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())
        return buffer.getvalue()

    def _modality_idea(
        self, prompt: str = "", concept_ids: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        vectors = []
        seen_identifiers = set()
        for concept_id in concept_ids or []:
            identifier = str(concept_id)
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            # Runtime action proposals carry active assembly ids. Early builds
            # looked only in the legacy concept alias and silently fell back to
            # an unrelated unprompted symbol. Resolve every neural inspection
            # alias while keeping the actual vectors authoritative.
            vector = self.memory.assembly_vectors.get(identifier)
            if vector is None:
                vector = self.memory.idea_vectors.get(identifier)
            if vector is None:
                vector = self.memory.concept_vectors.get(identifier)
            if vector is not None:
                vectors.append(vector)
        prompt_provided = bool(prompt.strip())
        if prompt_provided:
            vectors.append(self.memory.vector_for_text(prompt))
        projected: Optional[torch.Tensor] = None
        if vectors:
            cue = self.memory.space.bundle(vectors)
            if self.config.vector_symbolic_memory:
                recalled, _ = self.memory.recall_vector(
                    cue, workspace_slots=self.config.working_memory_slots
                )
            else:
                recalled = cue
            projected = self.idea_adapter(self._idea_model_vector(recalled))

        # A blank manual prompt is not converted into a hidden text prompt.
        # It uses the same brain's currently active neural vectors: resolved
        # assemblies above, weighted working memory, and liquid recurrent
        # state. The intrinsic VSA symbol is only the cold-start fallback when
        # this new identity has no measurable activity yet.
        if not prompt_provided:
            active_vectors: List[torch.Tensor] = []
            working = self._active_working_memory_vector()
            if working is not None:
                active_vectors.append(working.detach())
            liquid = self.liquid_state.detach().to(
                self.device, dtype=torch.float32
            )
            if bool(torch.isfinite(liquid).all()) and bool(
                torch.count_nonzero(liquid).item()
            ):
                active_vectors.append(liquid)
            if active_vectors:
                internal = torch.stack(active_vectors).mean(dim=0)
                projected = (
                    torch.tanh(0.78 * projected + 0.22 * internal)
                    if projected is not None
                    else torch.tanh(internal)
                )
        if projected is not None:
            return projected
        cold_start = self.memory.space.symbol("unprompted-imagination")
        return self.idea_adapter(self._idea_model_vector(cold_start))

    def _modality_idea_evidence(
        self, prompt: str, concept_ids: Optional[Sequence[str]]
    ) -> Dict[str, Any]:
        resolved = set()
        for value in concept_ids or []:
            identifier = str(value)
            if (
                identifier in self.memory.assembly_vectors
                or identifier in self.memory.idea_vectors
                or identifier in self.memory.concept_vectors
            ):
                resolved.add(identifier)
        prompt_provided = bool(prompt.strip())
        if prompt_provided and resolved:
            source = "manual-prompt-and-active-assemblies"
        elif prompt_provided:
            source = "manual-prompt"
        elif resolved:
            source = "active-assemblies"
        elif self.working_memory:
            source = "active-working-memory"
        elif bool(torch.count_nonzero(self.liquid_state.detach()).item()):
            source = "active-liquid-state"
        else:
            source = "intrinsic-neural-cold-start"
        return {
            "source": source,
            "promptProvided": prompt_provided,
            "activeAssemblyCount": len(resolved),
            "workingMemoryVectorCount": len(self.working_memory),
            "sameBrain": True,
            "hiddenBehavioralPrompt": False,
        }

    def _modality_preview_budget(self, modality: str) -> int:
        total = MODALITY_DECODER_STEPS.get(modality, 1)
        tier_budget = MODALITY_PREVIEW_BUDGET_BY_TIER.get(
            self.config.hardware_tier,
            MODALITY_PREVIEW_BUDGET_BY_TIER["personal"],
        )
        return max(1, min(total, tier_budget))

    def generate_modality(
        self,
        modality: str,
        prompt: str = "",
        concept_ids: Optional[Sequence[str]] = None,
        input_path: str = "",
        settings: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        preview_callback: Optional[
            Callable[[float, str, bytes, Dict[str, Any]], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        created_artifact: Optional[Path] = None

        def ensure_active() -> None:
            if cancel_check is not None and bool(cancel_check()):
                if created_artifact is not None:
                    created_artifact.unlink(missing_ok=True)
                raise ModalityGenerationCancelled(
                    "modality generation was cancelled"
                )

        ensure_active()
        generation_started_at = time.perf_counter()
        enabled = {
            "vision": self.config.vision_enabled,
            "image": self.config.image_enabled,
            "audio": self.config.audio_enabled,
            "video": self.config.video_enabled,
        }
        if modality not in enabled:
            raise ValueError("modality must be image, audio, video, or vision")
        if not enabled[modality]:
            raise ValueError("%s modality pack is disabled for this brain" % modality)
        if seed is None:
            material = (
                "%s:%s:%s:%d"
                % (
                    self.brain_id,
                    modality,
                    prompt,
                    self.counters["inference_count"],
                )
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
            seed &= 0x7FFFFFFF
        idea_seed = self._modality_idea_evidence(prompt, concept_ids)
        idea = self._modality_idea(prompt, concept_ids)
        modality_steps = int(self.modality_training.get(modality, 0))
        installed_pack = next(
            (
                item
                for item in reversed(self.installed_modality_packs)
                if modality in item.get("modalities", [])
            ),
            None,
        )
        initialization = (
            "installed-pack:%s" % installed_pack.get("id")
            if installed_pack is not None
            else ("locally-trained" if modality_steps > 0 else "random")
        )
        randomly_initialized = initialization == "random"
        raw_settings = dict(settings or {})
        output_mode = str(raw_settings.get("outputMode", "legacy"))
        if output_mode not in {"legacy", "auto", "exact"}:
            raise ValueError("outputMode must be legacy, auto, or exact")
        scaled_output = output_mode in {"auto", "exact"} and modality != "vision"
        if scaled_output:
            allowed_settings = {
                "outputMode",
                "width",
                "height",
                "durationMs",
                "sampleRate",
                "fps",
                "includeAudio",
                "targetLatencyMs",
                "previewIntervalMs",
            }
            unknown_settings = sorted(set(raw_settings).difference(allowed_settings))
            if unknown_settings:
                raise ValueError(
                    "unsupported scaled media settings: %s"
                    % ", ".join(unknown_settings)
                )

            def optional_output_integer(key: str) -> Optional[int]:
                value = raw_settings.get(key)
                if value is None:
                    return None
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError("%s must be a positive integer" % key)
                return value

            output_width = optional_output_integer("width")
            output_height = optional_output_integer("height")
            output_sample_rate = optional_output_integer("sampleRate") or 16_000
            output_fps = optional_output_integer("fps") or 8
            duration_value = raw_settings.get("durationMs")
            output_duration_ms = (
                None
                if duration_value is None
                else _finite_number(duration_value, -1.0)
            )
            if output_duration_ms is not None and output_duration_ms <= 0.0:
                raise ValueError("durationMs must be finite and positive")
            if output_sample_rate > 0xFFFFFFFF:
                raise ValueError("sampleRate exceeds the WAV container field")
            if output_fps > 65_535:
                raise ValueError("fps exceeds the portable video timebase")
            if not isinstance(raw_settings.get("includeAudio", True), bool):
                raise ValueError("includeAudio must be boolean")
        else:
            output_width = output_height = None
            output_duration_ms = None
            output_sample_rate = int(
                _finite_number(raw_settings.get("sampleRate"), 16_000)
            )
            output_fps = max(
                1,
                min(60, int(_finite_number(raw_settings.get("fps"), 8))),
            )
        media_output_plan = None
        media_output_metadata = None
        native_unit_ms = 0.0
        actual_preview_count = 0
        first_preview_ms: Optional[float] = None
        if modality == "vision":
            ensure_active()
            if not input_path:
                raise ValueError("vision requires inputPath")
            try:
                from PIL import Image
                import numpy as np
            except ImportError as error:
                raise RuntimeError(
                    "vision file input requires Pillow and NumPy"
                ) from error
            with Image.open(input_path) as opened:
                image = opened.convert("RGB").resize(
                    (self.config.image_size, self.config.image_size)
                )
                tensor = torch.from_numpy(
                    np.asarray(image, dtype="float32").copy()
                ).permute(2, 0, 1)
            ensure_active()
            tensor = (tensor / 127.5 - 1.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding_tensor = self.modalities.vision(tensor)
                embedding = embedding_tensor[0].cpu().tolist()
                fingerprint = hashlib.sha256(
                    tensor.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                sensory_cue = self._sensory_substrate_vector(
                    embedding_tensor,
                    "image",
                    fingerprint,
                )
                _recalled_signal, recalled = self.memory.recall_vector(
                    sensory_cue,
                    workspace_slots=self.config.working_memory_slots,
                )
            assemblies_by_id = {
                str(item["id"]): item for item in self.memory.assemblies
            }
            association_summaries = []
            association_bytes = 2
            association_transport_limited = False
            for association in recalled:
                assembly_id = str(association["assembly_id"])
                assembly = assemblies_by_id.get(assembly_id, {})
                neuron_ids = [str(value) for value in association["neuron_ids"]]
                labels = [
                    str(self.memory.neurons[neuron_id].get("label", ""))
                    for neuron_id in neuron_ids
                    if neuron_id in self.memory.neurons
                ]
                summary = {
                    "assemblyId": assembly_id,
                    "score": float(association["score"]),
                    "kind": str(assembly.get("kind", "")),
                    "sourceLabel": str(assembly.get("source_label", "")),
                    "labels": labels,
                    "labelCount": len(labels),
                    "relationshipsPaged": False,
                }
                encoded_bytes = len(
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
                if encoded_bytes > SUBSTRATE_INSPECTION_TRANSPORT_BYTES:
                    summary = {
                        **summary,
                        "labels": [],
                        "relationshipsPaged": True,
                    }
                    encoded_bytes = len(
                        json.dumps(
                            summary,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                separator_bytes = 1 if association_summaries else 0
                if (
                    association_summaries
                    and association_bytes + separator_bytes + encoded_bytes
                    > SUBSTRATE_INSPECTION_TRANSPORT_BYTES
                ):
                    association_transport_limited = True
                    break
                association_summaries.append(summary)
                association_bytes += separator_bytes + encoded_bytes
            result = {
                "brainId": self.brain_id,
                "modality": "vision",
                "embedding": embedding,
                "associations": association_summaries,
                "associationCount": len(recalled),
                "associationViewTruncated": len(recalled) > len(
                    association_summaries
                ),
                "associationViewTransportLimited": association_transport_limited,
                "associationViewBytes": association_bytes,
                "associationMode": "exact-ternary-recurrent-spreading",
                "inputPath": str(Path(input_path).resolve()),
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
            }
        else:
            def encode_preview(
                progress_value: float,
                tensor: torch.Tensor,
                scaled_details: Optional[Dict[str, Any]] = None,
            ) -> None:
                nonlocal actual_preview_count
                nonlocal first_preview_ms
                ensure_active()
                if preview_callback is None:
                    return
                actual_preview_count += 1
                if first_preview_ms is None:
                    first_preview_ms = (
                        time.perf_counter() - generation_started_at
                    ) * 1_000.0
                total_steps = (
                    int(media_output_plan.work_units)
                    if media_output_plan is not None
                    else MODALITY_DECODER_STEPS[modality]
                )
                completed_step = max(
                    1,
                    min(total_steps, int(round(progress_value * total_steps))),
                )
                details: Dict[str, Any] = {
                    "schemaVersion": 1,
                    "modality": modality,
                    "completedUnits": completed_step,
                    "totalUnits": total_steps,
                    "actualDecoderOutput": True,
                    "spatialResolutionReduced": False,
                    "cadence": "hardware-aware-bounded-synchronous",
                    "hardwareTier": self.config.hardware_tier,
                    "previewCount": (
                        int(media_output_plan.estimated_previews)
                        if media_output_plan is not None
                        else self._modality_preview_budget(modality)
                    ),
                    "ideaSource": idea_seed["source"],
                    "activeAssemblyCount": idea_seed["activeAssemblyCount"],
                    "promptProvided": idea_seed["promptProvided"],
                    **(scaled_details or {}),
                }
                if modality == "image":
                    details.update(
                        {
                            "stage": "diffusion-vq-decode",
                            "width": int(tensor.shape[-1]),
                            "height": int(tensor.shape[-2]),
                        }
                    )
                    preview_callback(
                        progress_value,
                        "image/png",
                        self._png_bytes(tensor),
                        details,
                    )
                elif modality == "audio":
                    sample_rate = int(
                        media_output_plan.sample_rate
                        if media_output_plan is not None
                        and media_output_plan.sample_rate is not None
                        else output_sample_rate
                    )
                    sample_count = int(tensor.shape[-1])
                    details.update(
                        {
                            "stage": "codec-waveform",
                            "sampleCount": sample_count,
                            "totalSamples": int(
                                media_output_plan.total_samples
                                if media_output_plan is not None
                                and media_output_plan.total_samples is not None
                                else self.config.audio_samples
                            ),
                            "durationMs": sample_count / float(sample_rate) * 1_000.0,
                        }
                    )
                    preview_callback(
                        progress_value,
                        "audio/wav",
                        self._wav_bytes(
                            tensor,
                            sample_rate=sample_rate,
                        ),
                        details,
                    )
                elif modality == "video":
                    frame_count = int(
                        tensor.shape[2] if tensor.ndim == 5 else tensor.shape[1]
                    )
                    details.update(
                        {
                            "stage": "temporal-frame-timeline",
                            "frameCount": frame_count,
                            "totalFrames": int(
                                media_output_plan.total_frames
                                if media_output_plan is not None
                                and media_output_plan.total_frames is not None
                                else self.config.video_frames
                            ),
                            "width": int(tensor.shape[-1]),
                            "height": int(tensor.shape[-2]),
                        }
                    )
                    preview_callback(
                        progress_value,
                        "image/apng",
                        self._apng_bytes(
                            tensor,
                            fps=int(
                                _finite_number(
                                    output_fps, 8
                                )
                            ),
                        ),
                        details,
                    )
                ensure_active()

            if scaled_output:
                benchmark_started = time.perf_counter()
                self.modalities.generate(
                    modality,
                    idea,
                    seed=int(seed) ^ 0x4D454449,
                    cancel_check=cancel_check,
                    maximum_previews=1,
                )
                native_unit_ms = max(
                    0.001,
                    (time.perf_counter() - benchmark_started) * 1_000.0,
                )
                readings = self._resource_readings()
                available_memory, available_storage = media_resource_headroom(
                    readings
                )
                target_latency_ms = _finite_number(
                    raw_settings.get("targetLatencyMs"), 30_000.0
                )
                preview_interval_ms = _finite_number(
                    raw_settings.get("previewIntervalMs"), 500.0
                )
                if target_latency_ms <= 0.0 or preview_interval_ms <= 0.0:
                    raise ValueError(
                        "media latency and preview intervals must be positive"
                    )
                measurements = MediaGenerationMeasurements.for_modality(
                    modality,
                    available_memory_bytes=available_memory,
                    available_storage_bytes=available_storage,
                    native_unit_ms=native_unit_ms,
                    target_latency_ms=target_latency_ms,
                    preview_interval_ms=preview_interval_ms,
                    source="live-native-unit-benchmark",
                )
                request = MediaOutputRequest(
                    width=output_width,
                    height=output_height,
                    duration_ms=output_duration_ms,
                    sample_rate=output_sample_rate,
                    fps=output_fps,
                )
                if output_mode == "exact" and not request.explicit_for(modality):
                    raise ValueError(
                        "exact media output requires dimensions or duration"
                    )
                media_output_plan = plan_media_output(
                    modality,
                    NeuralMediaWindows.from_config(self.config),
                    measurements,
                    request,
                )

                def media_watermark(demand: MediaResourceDemand) -> bool:
                    status = self.resource_policy.status(
                        estimated_write_bytes=demand.output_bytes,
                        estimated_ram_bytes=demand.working_bytes,
                    )
                    admitted = not bool(
                        status.get("memoryPressure")
                        or status.get("diskPressure")
                    )
                    if not admitted:
                        self.resource_pause = {
                            "reason": "media generation reached the live resource watermark",
                            "readings": status,
                            "mediaDemand": {
                                "stage": demand.stage,
                                "workingBytes": demand.working_bytes,
                                "outputBytes": demand.output_bytes,
                                "completedUnits": demand.completed_units,
                                "totalUnits": demand.total_units,
                            },
                            "at": _iso_now(),
                        }
                    return admitted

                scaled = self.modalities.generate_scaled(
                    media_output_plan,
                    idea,
                    seed=int(seed),
                    training_steps=modality_steps,
                    installed_pack_id=(
                        str(installed_pack.get("id"))
                        if isinstance(installed_pack, Mapping)
                        else None
                    ),
                    preview_callback=(
                        encode_preview if preview_callback is not None else None
                    ),
                    cancel_check=cancel_check,
                    resource_watermark=media_watermark,
                )
                output = scaled.tensor
                media_output_metadata = scaled.metadata
            else:
                output = self.modalities.generate(
                    modality,
                    idea,
                    seed=seed,
                    preview_callback=(
                        encode_preview if preview_callback is not None else None
                    ),
                    cancel_check=cancel_check,
                    maximum_previews=self._modality_preview_budget(modality),
                )
            ensure_active()
            artifact_dir = self.engine_path / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_id = uuid.uuid4().hex
            if modality == "image":
                artifact = artifact_dir / (artifact_id + ".png")
                created_artifact = artifact
                artifact_bytes = self._png_bytes(output)
                artifact.write_bytes(artifact_bytes)
                mime_type = "image/png"
                ensure_active()
            elif modality == "audio":
                artifact = artifact_dir / (artifact_id + ".wav")
                created_artifact = artifact
                artifact_bytes = self._wav_bytes(
                    output,
                    sample_rate=int(
                        media_output_plan.sample_rate
                        if media_output_plan is not None
                        and media_output_plan.sample_rate is not None
                        else output_sample_rate
                    ),
                )
                artifact.write_bytes(artifact_bytes)
                mime_type = "audio/wav"
                ensure_active()
            elif modality == "video":
                fps = int(
                    media_output_plan.fps
                    if media_output_plan is not None
                    and media_output_plan.fps is not None
                    else output_fps
                )
                sample_rate = int(
                    output_sample_rate
                )
                include_audio_value = raw_settings.get("includeAudio", True)
                audio_requested = (
                    include_audio_value
                    if isinstance(include_audio_value, bool)
                    else True
                )
                audio_installed = any(
                    isinstance(item, Mapping)
                    and "audio" in item.get("modalities", [])
                    for item in self.installed_modality_packs
                )
                audio_supported = bool(
                    self.config.audio_enabled
                    and (
                        int(self.modality_training.get("audio", 0)) > 0
                        or audio_installed
                    )
                )
                synchronized_audio: Dict[str, Any] = {
                    "requested": audio_requested,
                    "supported": audio_supported,
                    "decoded": False,
                    "generated": False,
                    "sameBrainIdea": False,
                    "lengthAlignedToVideo": False,
                    "speechSynthesis": False,
                    "hiddenBehavioralPrompt": False,
                }
                audio_output: Optional[torch.Tensor] = None
                if not audio_requested:
                    synchronized_audio["reason"] = (
                        "Synchronized neural sound was disabled for this generation."
                    )
                elif not audio_supported:
                    synchronized_audio["reason"] = (
                        "No trained same-brain audio pack is available; the video remains silent."
                    )
                else:
                    audio_seed_material = (
                        "%d:same-brain-video-audio" % int(seed)
                    ).encode("utf-8")
                    audio_seed = int.from_bytes(
                        hashlib.sha256(audio_seed_material).digest()[:8],
                        "little",
                    ) & 0x7FFFFFFF
                    if output.ndim == 5:
                        frame_count = int(output.shape[2])
                    elif output.ndim == 4:
                        frame_count = int(output.shape[1])
                    else:
                        raise ValueError(
                            "video output must have shape [3, frames, height, width]"
                        )
                    duration_ms = frame_count / float(fps) * 1_000.0
                    if scaled_output:
                        audio_benchmark_started = time.perf_counter()
                        self.modalities.generate(
                            "audio",
                            idea,
                            seed=audio_seed ^ 0x41554449,
                            cancel_check=cancel_check,
                            maximum_previews=1,
                        )
                        audio_native_ms = max(
                            0.001,
                            (time.perf_counter() - audio_benchmark_started)
                            * 1_000.0,
                        )
                        readings = self._resource_readings()
                        audio_memory, audio_storage = media_resource_headroom(
                            readings
                        )
                        audio_measurements = (
                            MediaGenerationMeasurements.for_modality(
                                "audio",
                                available_memory_bytes=audio_memory,
                                available_storage_bytes=audio_storage,
                                native_unit_ms=audio_native_ms,
                                target_latency_ms=_finite_number(
                                    raw_settings.get("targetLatencyMs"),
                                    30_000.0,
                                ),
                                preview_interval_ms=_finite_number(
                                    raw_settings.get("previewIntervalMs"),
                                    500.0,
                                ),
                                source="live-native-unit-benchmark",
                            )
                        )
                        audio_plan = plan_media_output(
                            "audio",
                            NeuralMediaWindows.from_config(self.config),
                            audio_measurements,
                            MediaOutputRequest(
                                duration_ms=duration_ms,
                                sample_rate=sample_rate,
                            ),
                        )
                        audio_pack = next(
                            (
                                item
                                for item in reversed(
                                    self.installed_modality_packs
                                )
                                if "audio" in item.get("modalities", [])
                            ),
                            None,
                        )
                        audio_scaled = self.modalities.generate_scaled(
                            audio_plan,
                            idea,
                            seed=audio_seed,
                            training_steps=int(
                                self.modality_training.get("audio", 0)
                            ),
                            installed_pack_id=(
                                str(audio_pack.get("id"))
                                if isinstance(audio_pack, Mapping)
                                else None
                            ),
                            cancel_check=cancel_check,
                            resource_watermark=media_watermark,
                        )
                        audio_output = audio_scaled.tensor.reshape(-1)
                    else:
                        audio_native = self.modalities.generate(
                            "audio",
                            idea,
                            seed=audio_seed,
                            cancel_check=cancel_check,
                        )
                        duration_samples = max(
                            1,
                            int(round(frame_count * sample_rate / float(fps))),
                        )
                        audio_output = F.interpolate(
                            audio_native.detach().reshape(1, 1, -1),
                            size=duration_samples,
                            mode="linear",
                            align_corners=False,
                        ).reshape(-1)
                    ensure_active()
                    synchronized_audio.update(
                        {
                            "decoded": True,
                            "sameBrainIdea": True,
                            "audioSeed": audio_seed,
                            "sampleRate": sample_rate,
                            "durationMs": duration_ms,
                            "lengthAlignedToVideo": True,
                            **(
                                {
                                    "mediaOutputPlan": audio_plan.as_dict(),
                                    "mediaOutput": audio_scaled.metadata,
                                }
                                if scaled_output
                                else {}
                            ),
                        }
                    )
                container_fallback = ""
                try:
                    artifact_bytes = self._mp4_bytes(
                        output,
                        fps=fps,
                        audio=audio_output,
                        sample_rate=sample_rate,
                    )
                    artifact = artifact_dir / (artifact_id + ".mp4")
                    mime_type = "video/mp4"
                    if audio_output is not None:
                        synchronized_audio["generated"] = True
                        synchronized_audio.pop("reason", None)
                except (
                    ImportError,
                    OSError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as error:
                    artifact_bytes = self._apng_bytes(output, fps=fps)
                    artifact = artifact_dir / (artifact_id + ".png")
                    mime_type = "image/apng"
                    container_fallback = str(error)
                    if audio_output is not None:
                        synchronized_audio["generated"] = False
                        synchronized_audio["reason"] = (
                            "The MP4/AAC encoder was unavailable, so the honest "
                            "APNG fallback cannot carry the generated neural sound: %s"
                            % error
                        )
                created_artifact = artifact
                artifact.write_bytes(artifact_bytes)
                ensure_active()
            else:
                raise ValueError("modality must be image, audio, video, or vision")
            artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
            content_artifact = artifact_dir / (
                artifact_sha256 + artifact.suffix.lower()
            )
            if artifact != content_artifact:
                if content_artifact.exists():
                    if hashlib.sha256(content_artifact.read_bytes()).hexdigest() != artifact_sha256:
                        raise RuntimeError(
                            "content-addressed modality artifact checksum collision"
                        )
                    artifact.unlink(missing_ok=True)
                    created_artifact = None
                else:
                    os.replace(str(artifact), str(content_artifact))
                    created_artifact = content_artifact
                artifact = content_artifact
            ensure_active()
            result = {
                "brainId": self.brain_id,
                "modality": modality,
                "path": str(artifact),
                "mimeType": mime_type,
                "shape": list(output.shape),
                "seed": int(seed),
                "artifactSha256": artifact_sha256,
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
                "ideaSeed": idea_seed,
                "qualityNote": (
                    "Hardware-scaled output assembled from actual same-brain "
                    "neural patches/windows; size and training evidence do not "
                    "claim semantic quality."
                    if media_output_metadata is not None
                    else (
                        "Generated by a tiny research baseline; output quality depends "
                        "on the disclosed pack and local modality training."
                    )
                ),
                **(
                    {
                        "mediaOutputPlan": media_output_plan.as_dict(),
                        "mediaOutput": media_output_metadata,
                    }
                    if media_output_plan is not None
                    and media_output_metadata is not None
                    else {}
                ),
            }
            if modality == "video":
                result["containerFallback"] = container_fallback
                result["synchronizedAudio"] = synchronized_audio
            embedded_media = inline_media_data_url(mime_type, artifact_bytes)
            if embedded_media is not None:
                result["dataUrl"] = embedded_media
        generation_elapsed_seconds = max(
            1e-9, time.perf_counter() - generation_started_at
        )
        decoder_steps = {
            "vision": 1,
            "image": 4,
            "audio": 3,
            "video": 3,
        }[modality]
        if media_output_plan is not None:
            decoder_steps = (
                int(media_output_plan.work_units)
                * MODALITY_DECODER_STEPS[modality]
                + MODALITY_DECODER_STEPS[modality]
            )
        if (
            modality == "video"
            and isinstance(result.get("synchronizedAudio"), Mapping)
            and bool(result["synchronizedAudio"].get("decoded", False))
        ):
            decoder_steps += 3
        result["generationPerformance"] = {
            "elapsedMs": generation_elapsed_seconds * 1000.0,
            "decoderSteps": decoder_steps,
            "stepsPerSecond": decoder_steps / generation_elapsed_seconds,
            "progressivePreviews": (
                0
                if modality == "vision" or preview_callback is None
                else actual_preview_count
            ),
            "previewCadence": (
                "measured-resource-plan"
                if media_output_plan is not None
                else "hardware-aware"
            ),
            "previewOnlyCadenceAdjusted": media_output_plan is None,
            "generationResolutionReducedForPreview": False,
            "measured": True,
            "hiddenBehavioralPrompt": False,
            **(
                {
                    "nativeUnitBenchmarkMs": native_unit_ms,
                    "firstPreviewMs": first_preview_ms,
                    "mediaOutputPlan": media_output_plan.as_dict(),
                }
                if media_output_plan is not None
                else {}
            ),
        }
        self.events.append(
            "modality-generation",
            {
                "modality": modality,
                "seed": int(seed),
                "outputPath": result.get("path"),
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
                "generationPerformance": result["generationPerformance"],
                **(
                    {"synchronizedAudio": result.get("synchronizedAudio")}
                    if modality == "video"
                    else {}
                ),
            },
        )
        return result

    def install_modality_pack(
        self, pack_path: Path, manifest: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Transactionally install compatible safe modality tensors.

        The desktop validates the outer archive, but the engine independently
        validates architecture, shapes, finite values, and tensor namespaces.
        No code, pickle, optimizer, or arbitrary repository script is loaded.
        """

        path = Path(pack_path).resolve()
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise ValueError("modality pack must be a local .safetensors file")
        if manifest.get("format") != "omni-modality-pack":
            raise ValueError("unsupported modality pack format")
        if int(manifest.get("formatVersion", 0)) != 1:
            raise ValueError("unsupported modality pack version")
        if manifest.get("architecture") != "OmniCortex":
            raise ValueError("modality pack architecture must be OmniCortex")
        if int(manifest.get("architectureSchemaVersion", 0)) != ENGINE_SCHEMA_VERSION:
            raise ValueError("modality pack architecture schema is incompatible")
        pack = manifest.get("pack")
        compatibility = manifest.get("compatibility")
        ledger = manifest.get("licenseLedger")
        if not isinstance(pack, Mapping) or not isinstance(
            compatibility, Mapping
        ):
            raise ValueError("modality pack manifest is incomplete")
        if not isinstance(ledger, Mapping) or not str(
            ledger.get("license", "")
        ).strip():
            raise ValueError("modality pack requires a declared license")
        modalities = pack.get("modalities")
        if not isinstance(modalities, (list, tuple)) or not modalities:
            raise ValueError("modality pack must declare at least one modality")
        allowed = {"vision", "image", "audio", "video"}
        selected = [str(item) for item in modalities]
        if len(set(selected)) != len(selected) or set(selected).difference(allowed):
            raise ValueError("modality pack contains unsupported modalities")
        expected_compatibility = {
            "dModel": self.config.d_model,
            "modalityChannels": self.config.modality_channels,
            "imageSize": self.config.image_size,
            "audioSamples": self.config.audio_samples,
            "videoFrames": self.config.video_frames,
        }
        mismatches = [
            key
            for key, value in expected_compatibility.items()
            if int(compatibility.get(key, -1)) != int(value)
        ]
        if mismatches:
            raise ValueError(
                "modality pack compatibility mismatch: %s"
                % ", ".join(mismatches)
            )
        tensors = load_tensors(path, device="cpu")
        if not tensors:
            raise ValueError("modality pack contains no tensors")
        prefix = "modalities."
        if any(not key.startswith(prefix) for key in tensors):
            raise ValueError("modality pack contains a tensor outside modalities.*")
        provided = {
            key[len(prefix) :]: value for key, value in tensors.items()
        }
        current = self.modalities.state_dict()
        expected_keys = {
            key
            for key in current
            if any(key.startswith(name + ".") for name in selected)
        }
        provided_keys = set(provided)
        if provided_keys != expected_keys:
            missing = sorted(expected_keys.difference(provided_keys))
            extra = sorted(provided_keys.difference(expected_keys))
            raise ValueError(
                "modality pack tensor inventory mismatch (missing=%s, extra=%s)"
                % (missing[:8], extra[:8])
            )
        replacement = {
            key: value.detach().cpu().clone() for key, value in current.items()
        }
        for key, value in provided.items():
            expected = current[key]
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError("modality pack tensor shape mismatch: " + key)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("modality pack contains non-finite tensor: " + key)
            replacement[key] = value.to(dtype=expected.dtype)
        previous = {
            key: value.detach().cpu().clone() for key, value in current.items()
        }
        previous_packs = [dict(item) for item in self.installed_modality_packs]
        checksum = self._file_sha256(path)
        record = {
            "id": str(pack.get("id", "")).strip() or checksum[:16],
            "name": str(pack.get("name", "")).strip() or "Omni modality pack",
            "modalities": selected,
            "sha256": checksum,
            "license": str(ledger["license"]).strip(),
            "provenanceUrl": str(ledger.get("provenanceUrl", "")).strip(),
            "installedAt": _iso_now(),
        }
        try:
            self.modalities.load_state_dict(replacement, strict=True)
            self._sync_stability_state()
            for key in expected_keys:
                name = "modalities." + key
                parameter = self._named_slow_parameters().get(name)
                if parameter is not None:
                    self.slow_anchors[name] = parameter.detach().cpu().clone()
                    self.slow_importance[name] = torch.zeros_like(
                        parameter.detach().cpu(), dtype=torch.float32
                    )
            self.installed_modality_packs.append(record)
            self.save()
        except Exception:
            self.modalities.load_state_dict(previous, strict=True)
            self.installed_modality_packs = previous_packs
            self._sync_stability_state()
            raise
        trace = {
            "id": uuid.uuid4().hex,
            "created_at": _iso_now(),
            "kind": "modality-pack-install",
            "pack_id": record["id"],
            "modalities": selected,
            "sha256": checksum,
            "code_executed": False,
            "steps": [
                {
                    "stage": "validate",
                    "detail": "Validated safe tensor inventory, shapes, finite values, compatibility, provenance, and license.",
                    "value": "%d tensors" % len(provided),
                },
                {
                    "stage": "install",
                    "detail": "Replaced only the declared modality namespace and reset its stability anchors.",
                    "value": ", ".join(selected),
                },
            ],
        }
        self.traces.append(trace)
        self.events.append("modality-pack-installed", {**record, "traceId": trace["id"]})
        self.save()
        return {
            "brainId": self.brain_id,
            "pack": record,
            "tensorCount": len(provided),
            "parameterChecksum": self.parameter_checksum(),
            "trace": trace,
        }

    def checkpoint(self, operation_id: str) -> Dict[str, Any]:
        """Commit live neural state and its packed inference manifest only.

        Recovery-point materialization belongs to the desktop repository. This
        flush deliberately creates no engine/snapshots directory, avoiding a
        second full copy for one UI operation.
        """

        operation_id = str(operation_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", operation_id):
            raise ValueError("checkpoint operation_id is invalid")
        self.save()
        packed = self.export_packed_ternary()
        metadata_path = self.engine_path / "brain.json"
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        substrate = (
            metadata.get("substrate", {}).get("persistence", {})
        )
        mutable_state = metadata.get("mutable_state", {})
        packed_manifest_path = self.engine_path / "packed-ternary" / "manifest.json"
        receipt = {
            "format": "omni-neural-checkpoint",
            "formatVersion": 1,
            "brainId": self.brain_id,
            "operationId": operation_id,
            "committed": True,
            "createdAt": _iso_now(),
            "parameterChecksum": self.parameter_checksum(),
            "metadataSha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "substrateContentSha256": str(
                substrate.get("contentSha256", "")
            ),
            "mutableStateContentSha256": str(
                mutable_state.get("contentSha256", "")
            ),
            "packedManifestSha256": hashlib.sha256(
                packed_manifest_path.read_bytes()
            ).hexdigest(),
            "packedContentSha256": str(
                packed["summary"].get("contentSha256", "")
            ),
            "snapshotCreated": False,
        }
        if not all(
            re.fullmatch(r"[a-f0-9]{64}", str(receipt[field]))
            for field in (
                "parameterChecksum",
                "metadataSha256",
                "substrateContentSha256",
                "mutableStateContentSha256",
                "packedManifestSha256",
                "packedContentSha256",
            )
        ):
            raise RuntimeError("neural checkpoint did not commit complete hashes")
        return receipt

    def snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        self.counters["snapshots"] += 1
        self.save()
        packed = self.export_packed_ternary()
        clean_label = SAFE_NAME.sub("-", label.strip()).strip(".-")[:48] or "snapshot"
        snapshot_id = "%s-%s" % (clean_label, uuid.uuid4().hex[:12])
        destination = self.engine_path / "snapshots" / snapshot_id
        snapshot_files(self.engine_path, destination)
        shutil.copytree(
            self.engine_path / "packed-ternary",
            destination / "packed-ternary",
        )
        snapshot_metadata = read_json(destination / "brain.json")
        checksum = self._snapshot_checksum(
            destination / "core.safetensors",
            destination / "plasticity.safetensors",
            snapshot_metadata.get("substrate", {})
            .get("persistence", {})
            .get("contentSha256", ""),
            snapshot_metadata.get("mutable_state", {}).get(
                "contentSha256", ""
            ),
        )
        result = {
            "id": snapshot_id,
            "brainId": self.brain_id,
            "label": label,
            "createdAt": _iso_now(),
            "path": str(destination),
            "checksum": checksum,
            "packedTernary": packed["summary"],
            "metrics": self.metrics(),
        }
        atomic_write_json(destination / "snapshot.json", result)
        self.events.append("snapshot", result)
        return result

    def metrics(self) -> Dict[str, Any]:
        parameter_accounting = self.parameter_accounting()
        trainable_parameters = int(
            parameter_accounting["mutableDenseParameters"]
        )
        files_bytes = 0
        if self.engine_path.exists():
            for path in self.engine_path.rglob("*"):
                if path.is_file():
                    try:
                        files_bytes += path.stat().st_size
                    except OSError:
                        pass
        return {
            "concepts": len(self.memory.concepts),
            "ideas": len(self.memory.ideas),
            "synapses": len(self.memory.relations)
            + int(
                self.router.synapses.effective_weight().ne(0).sum().item()
            ),
            "activeSynapses": int(
                self.router.synapses.effective_weight().ne(0).sum().item()
            ),
            "plasticityEvents": int(
                self.router.synapses.plasticity_events.item()
            ),
            "messages": int(self.conversation.summary()["messageCount"]),
            "trainingSources": len(self.training_sources),
            "replayExamples": len(self.replay),
            "workingMemoryVectors": len(self.working_memory),
            "experts": self.decoder.expert_count,
            "substrateGrowth": {
                "cardinalityLimit": None,
                "neurons": len(self.memory.neurons),
                "assemblies": len(self.memory.assemblies),
                "synapses": len(self.memory.synapses),
                "growthEvents": self.memory.growth_events,
                "growthPauses": self.memory.growth_pauses,
                "paused": self.growth_pause is not None,
            },
            "modalityTraining": dict(self.modality_training),
            "trainableParameters": trainable_parameters,
            "parameterAccounting": parameter_accounting,
            "estimatedBytes": files_bytes,
            "counters": dict(self.counters),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "brainId": self.brain_id,
            "name": self.config.name,
            "storagePath": str(self.storage_path),
            "enginePath": str(self.engine_path),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "parameterChecksum": self.parameter_checksum(),
            "metrics": self.metrics(),
            "runtimeCard": self.runtime_card(),
        }

    def state(self, include_events: int = 20) -> Dict[str, Any]:
        result = self.summary()
        result["files"] = {
            "metadata": str(self.engine_path / "brain.json"),
            "core": str(self.engine_path / "core.safetensors"),
            "plasticity": str(self.engine_path / "plasticity.safetensors"),
            "substrate": str(self.engine_path / "substrate" / "manifest.json"),
            "mutableState": str(self.engine_path / "state" / "manifest.json"),
            "replay": str(self.engine_path / "state" / "replay.sqlite3"),
            "events": str(self.engine_path / "events.sqlite3"),
            "origin": str(self.engine_path / "origin"),
            "snapshots": str(self.engine_path / "snapshots"),
        }
        result["eventLogIntegrity"] = self.events.integrity()
        result["events"] = self.events.recent(include_events)
        return result

    def update_config(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Apply builder controls that do not change checkpoint tensor shapes."""

        changed: List[str] = []

        def assign(attribute: str, key: str, transform: Any = None) -> None:
            if key not in raw:
                return
            value = raw[key]
            if transform is not None:
                value = transform(value)
            if getattr(self.config, attribute) != value:
                setattr(self.config, attribute, value)
                changed.append(attribute)

        assign("name", "name", str)
        if "contextWindowTokens" in raw:
            next_context_tokens = int(raw["contextWindowTokens"])
            if next_context_tokens < 8:
                raise ValueError("contextWindowTokens must be at least 8")
            if next_context_tokens != self.config.max_seq_len:
                previous_context_tokens = int(self.config.max_seq_len)
                # Rotary tables are non-persistent derived buffers, so they can
                # be rebuilt without changing any learned parameter or
                # optimizer moment. They grow lazily from actual tokens rather
                # than allocating the selected maximum window here.
                try:
                    for block in self.decoder.blocks:
                        block.attention.rotary.configure_max_seq_len(
                            next_context_tokens
                        )
                except Exception:
                    raise
                self.config.max_seq_len = next_context_tokens
                if (
                    self._runtime_training_max_seq_len
                    >= previous_context_tokens
                ):
                    self._runtime_training_max_seq_len = (
                        next_context_tokens
                    )
                else:
                    # Preserve a measured allocator-recovery ceiling until the
                    # worker restarts; never confuse it with saved capacity.
                    self._runtime_training_max_seq_len = min(
                        max(8, self._runtime_training_max_seq_len),
                        next_context_tokens,
                    )
                self.recent_token_context = self.recent_token_context[
                    -next_context_tokens:
                ]
                changed.append("max_seq_len")
        for attribute, key in (
            ("online_learning", "onlineLearning"),
            ("consolidation_enabled", "consolidation"),
            ("metaplasticity", "metaplasticity"),
            ("retain_source_text", "retainSourceText"),
            ("learn_from_own_messages", "learnFromOwnMessages"),
        ):
            assign(attribute, key, bool)
        for attribute, key, transform in (
            ("firing_threshold", "firingThreshold", float),
            ("membrane_leak", "membraneLeak", float),
            ("working_memory_slots", "workingMemorySlots", int),
            (
                "short_term_half_life_minutes",
                "shortTermHalfLifeMinutes",
                float,
            ),
            ("forgetting_rate", "forgettingRate", float),
            ("consolidation_rate", "consolidationRate", float),
            ("memory_offload_bytes", "memoryOffloadBytes", int),
            ("memory_resident_items", "memoryResidentItems", int),
            (
                "memory_offload_slowdown_percent",
                "memoryOffloadSlowdownPercent",
                float,
            ),
            (
                "storage_bytes_per_second",
                "storageBytesPerSecond",
                safe_rounded_storage_bytes_per_second,
            ),
        ):
            assign(attribute, key, transform)
        assign("working_memory_mode", "workingMemoryMode", str)
        if "systemRamMode" in raw or "systemRamSharePercent" in raw:
            system_ram_mode = str(
                raw.get(
                    "systemRamMode",
                    "manual"
                    if "systemRamSharePercent" in raw
                    else (
                        "manual"
                        if self.config.system_ram_share_percent > 0.0
                        else "auto"
                    ),
                )
            )
            if system_ram_mode not in {"auto", "manual"}:
                raise ValueError("systemRamMode must be auto or manual")
            system_ram_share = (
                float(raw.get("systemRamSharePercent", 0.0))
                if system_ram_mode == "manual"
                else 0.0
            )
            if system_ram_mode == "manual" and not (
                30.0 <= system_ram_share <= 100.0
            ):
                raise ValueError(
                    "manual systemRamSharePercent must be in [30, 100]"
                )
            if self.config.system_ram_share_percent != system_ram_share:
                self.config.system_ram_share_percent = system_ram_share
                changed.append("system_ram_share_percent")
        if "learningRate" in raw:
            neural_rate = max(
                1e-5, min(0.02, float(raw["learningRate"]) * 0.02)
            )
            if self.config.learning_rate != neural_rate:
                self.config.learning_rate = neural_rate
                changed.append("learning_rate")
        assign(
            "memory_recipe",
            "memoryRecipe",
            lambda value: (
                "adaptive-retention"
                if str(value) in {"human", "human-consolidation"}
                else str(value)
            ),
        )
        self.config.ternary_weights = True
        self.config.spiking_dynamics = True
        self.config.stdp_plasticity = True
        self.config.liquid_dynamics = True
        self.config.vector_symbolic_memory = True
        self.config.memory_injection = "working-memory"
        requested_liquid = str(raw.get("liquidMode", self.config.liquid_mode))
        if requested_liquid not in {"cfc", "ltc"}:
            raise ValueError("liquidMode must be cfc or ltc")
        if requested_liquid != self.config.liquid_mode:
            self.config.liquid_mode = requested_liquid
            self.liquid = LiquidController(
                self.config.idea_dim,
                mode=requested_liquid,
                solver_steps=self.config.liquid_steps,
            ).to(self.device)
            changed.append("liquid_mode")
        self.config.validate()
        if any(
            value in changed
            for value in {
                "system_ram_share_percent",
                "storage_bytes_per_second",
                "memory_offload_bytes",
                "memory_resident_items",
            }
        ):
            self.resource_policy = ResourcePolicy(
                self.engine_path,
                ram_reserve_bytes=self.config.ram_reserve_bytes,
                disk_reserve_bytes=self.config.disk_reserve_bytes,
                system_ram_share_percent=self.config.system_ram_share_percent,
                storage_bytes_per_second=self.config.storage_bytes_per_second,
                hardware_tier=self.config.hardware_tier,
            )
            self.state_store.policy = self.resource_policy
            self.replay.policy = self.resource_policy
            self.paged_working_memory.policy = self.resource_policy
        self.population_controls_from_config()
        # Resource-only settings must not discard learned Adam moments. The
        # optimizer is rebuilt only when its configured learning rate changes.
        if "learning_rate" in changed:
            self._replace_optimizer()
        self.save()
        self.events.append(
            "config-updated", {"changed": changed, "runtimeCard": self.runtime_card()}
        )
        result = self.summary()
        result["changed"] = changed
        return result

    def population_controls_from_config(self) -> None:
        self.router.population.leak = self.config.membrane_leak
        self.router.population.threshold = self.config.firing_threshold
        self.router.synapses.metaplasticity_rate = (
            self.config.metaplasticity_rate
            if self.config.metaplasticity
            else 0.0
        )
        for root in self._trainable_modules():
            for module in root.modules():
                if isinstance(module, BitLinear):
                    module.ternary = True

    @staticmethod
    def _overlay_tensor_sha256(value: Optional[torch.Tensor]) -> Optional[str]:
        if value is None:
            return None
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(dimension) for dimension in tensor.shape).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _overlay_state_identity(self) -> Dict[str, Any]:
        """Hash authoritative merge inputs without materializing one huge object."""

        digest = hashlib.sha256()
        config_sha256 = hashlib.sha256(
            json.dumps(
                self.config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        digest.update(b"engine-schema\0")
        digest.update(str(ENGINE_SCHEMA_VERSION).encode("ascii"))
        digest.update(b"\nbrain\0")
        digest.update(self.brain_id.encode("utf-8"))
        digest.update(b"\nconfig\0")
        digest.update(config_sha256.encode("ascii"))
        digest.update(b"\n")

        def add_record(
            kind: str,
            identity: str,
            record: Mapping[str, Any],
            vector: Optional[torch.Tensor] = None,
        ) -> None:
            digest.update(kind.encode("ascii"))
            digest.update(b"\0")
            digest.update(identity.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    dict(record),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(b"\0")
            vector_sha256 = self._overlay_tensor_sha256(vector)
            digest.update((vector_sha256 or "-").encode("ascii"))
            digest.update(b"\n")

        for neuron_id in sorted(self.memory.neurons):
            add_record(
                "neuron",
                neuron_id,
                self.memory.neurons[neuron_id],
                self.memory.neuron_vectors.get(neuron_id),
            )
        assemblies = sorted(
            self.memory.assemblies, key=lambda item: str(item.get("id", ""))
        )
        for assembly in assemblies:
            assembly_id = str(assembly.get("id", ""))
            add_record(
                "assembly",
                assembly_id,
                assembly,
                self.memory.assembly_vectors.get(assembly_id),
            )
        for synapse_id in sorted(self.memory.synapses):
            add_record(
                "synapse",
                synapse_id,
                self.memory.synapses[synapse_id],
            )
        replay_sha256 = []
        for index, vector in enumerate(self.replay):
            checksum = self._overlay_tensor_sha256(vector)
            if checksum is None:
                continue
            replay_sha256.append(checksum)
            digest.update(b"replay\0")
            digest.update(str(index).encode("ascii"))
            digest.update(b"\0")
            digest.update(checksum.encode("ascii"))
            digest.update(b"\n")
        parameter_sha256 = self.parameter_checksum()
        digest.update(b"parameters\0")
        digest.update(parameter_sha256.encode("ascii"))
        return {
            "stateSha256": digest.hexdigest(),
            "parameterSha256": parameter_sha256,
            "configSha256": config_sha256,
            "engineSchemaVersion": ENGINE_SCHEMA_VERSION,
            "counts": {
                "neurons": len(self.memory.neurons),
                "assemblies": len(self.memory.assemblies),
                "synapses": len(self.memory.synapses),
                "replayExamples": len(self.replay),
            },
            "replaySha256": replay_sha256,
        }

    def preview_overlay(self, source: "AdaptiveBrain") -> Dict[str, Any]:
        """Describe and bind the exact worker state that a fork merge can add."""

        if source.brain_id == self.brain_id:
            raise ValueError("cannot preview a brain overlay into itself")
        source_identity = source._overlay_state_identity()
        target_identity = self._overlay_state_identity()
        target_fingerprints = {
            str(idea.get("fingerprint", "")) for idea in self.memory.ideas
        }
        target_assemblies_by_id = {
            str(idea.get("id", "")): idea for idea in self.memory.ideas
        }
        target_assemblies_by_fingerprint = {
            str(idea.get("fingerprint", "")): idea
            for idea in self.memory.ideas
        }
        source_replay = source_identity["replaySha256"]
        target_replay = set(target_identity["replaySha256"])
        divergent_neurons = sum(
            1
            for neuron_id, neuron in source.memory.neurons.items()
            if neuron_id in self.memory.neurons
            and (
                self.memory.neurons[neuron_id] != neuron
                or self._overlay_tensor_sha256(
                    self.memory.neuron_vectors.get(neuron_id)
                )
                != self._overlay_tensor_sha256(
                    source.memory.neuron_vectors.get(neuron_id)
                )
            )
        )
        divergent_synapses = sum(
            1
            for synapse_id, synapse in source.memory.synapses.items()
            if synapse_id in self.memory.synapses
            and self.memory.synapses[synapse_id] != synapse
        )
        divergent_assemblies = 0
        for assembly in source.memory.ideas:
            assembly_id = str(assembly.get("id", ""))
            fingerprint = str(assembly.get("fingerprint", ""))
            existing = (
                target_assemblies_by_fingerprint.get(fingerprint)
                or target_assemblies_by_id.get(assembly_id)
            )
            if existing is None:
                continue
            existing_id = str(existing.get("id", ""))
            if (
                existing != assembly
                or self._overlay_tensor_sha256(
                    self.memory.assembly_vectors.get(existing_id)
                )
                != self._overlay_tensor_sha256(
                    source.memory.assembly_vectors.get(assembly_id)
                )
            ):
                divergent_assemblies += 1
        additions = {
            "neurons": sum(
                neuron_id not in self.memory.neurons
                for neuron_id in source.memory.neurons
            ),
            "assemblies": sum(
                str(idea.get("fingerprint", "")) not in target_fingerprints
                and str(idea.get("id", "")) not in target_assemblies_by_id
                for idea in source.memory.ideas
            ),
            "synapses": sum(
                synapse_id not in self.memory.synapses
                for synapse_id in source.memory.synapses
            ),
            "replayExamples": sum(
                checksum not in target_replay for checksum in source_replay
            ),
        }
        duplicates = {
            "neurons": len(source.memory.neurons) - additions["neurons"],
            "assemblies": len(source.memory.ideas) - additions["assemblies"],
            "synapses": len(source.memory.synapses) - additions["synapses"],
            "replayExamples": len(source_replay) - additions["replayExamples"],
        }
        descriptor = {
            "schemaVersion": 1,
            "sourceBrainId": source.brain_id,
            "targetBrainId": self.brain_id,
            "sourceStateSha256": source_identity["stateSha256"],
            "targetStateSha256": target_identity["stateSha256"],
            "sourceParameterSha256": source_identity["parameterSha256"],
            "targetParameterSha256": target_identity["parameterSha256"],
            "sourceConfigSha256": source_identity["configSha256"],
            "targetConfigSha256": target_identity["configSha256"],
            "engineSchemaVersion": ENGINE_SCHEMA_VERSION,
            "sourceCounts": source_identity["counts"],
            "targetCounts": target_identity["counts"],
            "additions": additions,
            "duplicates": duplicates,
            "divergent": {
                "neurons": divergent_neurons,
                "assemblies": divergent_assemblies,
                "synapses": divergent_synapses,
            },
            "weightsAveraged": False,
        }
        digest = hashlib.sha256(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {**descriptor, "digest": digest}

    def merge_overlay(
        self, source: "AdaptiveBrain", expected_preview_digest: str
    ) -> Dict[str, Any]:
        """Merge a reviewed authoritative overlay, never whole-model weights."""

        if source.brain_id == self.brain_id:
            raise ValueError("cannot merge a brain overlay into itself")
        if not re.fullmatch(r"[a-f0-9]{64}", expected_preview_digest or ""):
            raise ValueError("an exact authoritative overlay preview digest is required")
        preview = self.preview_overlay(source)
        if preview["digest"] != expected_preview_digest:
            raise ValueError(
                "authoritative overlay changed after review; request a new preview"
            )
        existing_fingerprints = {
            idea["fingerprint"] for idea in self.memory.ideas
        }
        existing_assembly_ids = {
            str(idea.get("id", "")) for idea in self.memory.ideas
        }
        added_concepts = 0
        added_ideas = 0
        added_relations = 0
        added_replay = 0
        additions = preview["additions"]
        estimated_bytes = (
            int(additions["neurons"])
            * (self.config.vsa_dim * 4 + 640)
            + int(additions["assemblies"])
            * (self.config.vsa_dim * 4 + 1024)
            + int(additions["synapses"]) * 384
            + int(additions["replayExamples"])
            * (max(self.config.idea_dim, self.config.d_model) * 4 + 128)
        )
        if not self._allow_substrate_growth(estimated_bytes):
            raise SubstrateResourcePause(
                "overlay merge paused at the host resource reserve"
            )
        for concept_id, concept in source.memory.concepts.items():
            if concept_id in self.memory.concepts:
                continue
            self.memory.concepts[concept_id] = dict(concept)
            vector = source.memory.concept_vectors.get(concept_id)
            if vector is not None:
                self.memory.concept_vectors[concept_id] = vector.detach().cpu().clone()
            added_concepts += 1
        for idea in source.memory.ideas:
            if (
                idea["fingerprint"] in existing_fingerprints
                or str(idea.get("id", "")) in existing_assembly_ids
            ):
                continue
            copied = dict(idea)
            if not (
                self.config.memory_recipe == "total-recall"
                and self.config.retain_source_text
            ):
                copied.pop("source_text", None)
            self.memory.ideas.append(copied)
            vector = source.memory.idea_vectors.get(idea["id"])
            if vector is not None:
                self.memory.idea_vectors[idea["id"]] = vector.detach().cpu().clone()
            existing_fingerprints.add(idea["fingerprint"])
            existing_assembly_ids.add(str(idea.get("id", "")))
            added_ideas += 1
        for relation_id, relation in source.memory.relations.items():
            if relation_id in self.memory.relations:
                continue
            if (
                relation["source_id"] in self.memory.concepts
                and relation["target_id"] in self.memory.concepts
            ):
                self.memory.relations[relation_id] = dict(relation)
                added_relations += 1
        replay_hashes = {
            hashlib.sha256(vector.numpy().tobytes()).hexdigest()
            for vector in self.replay
        }
        for vector in source.replay:
            checksum = hashlib.sha256(vector.numpy().tobytes()).hexdigest()
            if checksum in replay_hashes:
                continue
            self._append_replay(vector, importance=1.0)
            replay_hashes.add(checksum)
            added_replay += 1
        self.save()
        result = {
            "targetBrainId": self.brain_id,
            "sourceBrainId": source.brain_id,
            "concepts": added_concepts,
            "ideas": added_ideas,
            "relations": added_relations,
            "replayExamples": added_replay,
            "weightsAveraged": False,
            "reviewedDigest": expected_preview_digest,
            "metrics": self.metrics(),
        }
        self.events.append("overlay-merged", result)
        return result
