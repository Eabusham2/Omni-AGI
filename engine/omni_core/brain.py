"""Adaptive OmniCortex brain lifecycle and persistence."""

import array
import base64
import binascii
import hashlib
import io
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import wave
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import OmniConfig
from .liquid import LiquidController
from .modalities import ModalityHub
from .model import BitLinear, OmniDecoder
from .persistence import (
    EventLog,
    atomic_save_tensors,
    atomic_write_json,
    load_tensors,
    read_json,
    snapshot_files,
    tensor_checksum,
)
from .spiking import AssociativeSpikingRouter
from .tokenizer import ByteTokenizer
from .vsa import ConceptMemory


ENGINE_SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


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


class AdaptiveBrain:
    """One persistent, mutable model identity.

    ``storage_path`` is the desktop brain directory.  All Python-owned files
    live below ``storage_path/engine`` so the UI's ``brain.json`` remains
    authoritative and untouched.
    """

    def __init__(self, brain_id: str, storage_path: Path, config: OmniConfig):
        config.validate()
        self.brain_id = str(brain_id)
        self.storage_path = Path(storage_path).resolve()
        self.engine_path = self.storage_path / "engine"
        self.config = config
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
                if isinstance(module, BitLinear):
                    module.ternary = config.ternary_weights
        self.memory = ConceptMemory(
            config.vsa_dim,
            seed=config.seed,
            max_concepts=config.max_concepts,
            max_ideas=config.max_ideas,
            max_relations=config.max_synapses,
        )
        self.liquid_state = torch.zeros(1, config.idea_dim, device=self.device)
        self.working_memory: List[torch.Tensor] = []
        self.replay: List[torch.Tensor] = []
        self.messages: List[Dict[str, Any]] = []
        self.traces: List[Dict[str, Any]] = []
        self.training_sources: List[Dict[str, Any]] = []
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
        }
        self.modality_training: Dict[str, int] = {
            "vision": 0,
            "image": 0,
            "audio": 0,
            "video": 0,
        }
        self.installed_modality_packs: List[Dict[str, Any]] = []
        self.novelty_streak = 0
        self.growth_pause: Optional[Dict[str, Any]] = None
        self.last_activity_decay = time.time()
        self.slow_anchors: Dict[str, torch.Tensor] = {}
        self.slow_importance: Dict[str, torch.Tensor] = {}
        self._sync_stability_state()
        self._optimizer = self._new_optimizer()
        self.events = EventLog(self.engine_path / "events.sqlite3", self.brain_id)

    def _trainable_modules(self) -> Iterable[nn.Module]:
        return (
            self.decoder,
            self.memory_bridge,
            self.idea_adapter,
            self.router,
            self.liquid,
            self.modalities,
        )

    def _new_optimizer(
        self, learning_rate: Optional[float] = None
    ) -> torch.optim.Optimizer:
        parameters = []
        for module in self._trainable_modules():
            parameters.extend(parameter for parameter in module.parameters())
        return torch.optim.AdamW(
            parameters,
            lr=max(
                1e-6,
                min(0.02, float(learning_rate or self.config.learning_rate)),
            ),
            weight_decay=self.config.weight_decay,
        )

    def _named_slow_parameters(self) -> Dict[str, nn.Parameter]:
        named: Dict[str, nn.Parameter] = {}
        for prefix, module in (
            ("decoder", self.decoder),
            ("memory_bridge", self.memory_bridge),
            ("idea_adapter", self.idea_adapter),
            ("liquid", self.liquid),
            ("modalities", self.modalities),
        ):
            for name, parameter in module.named_parameters():
                named["%s.%s" % (prefix, name)] = parameter
        return named

    def _sync_stability_state(self) -> None:
        """Keep EWC-like anchors aligned with dynamically grown parameters."""

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

    def _commit_slow_anchors(self, rate: float = 1.0) -> None:
        if not self.config.metaplasticity:
            return
        rate = max(0.0, min(float(rate), 1.0))
        self._sync_stability_state()
        for name, parameter in self._named_slow_parameters().items():
            current = parameter.detach().float().cpu()
            self.slow_anchors[name].lerp_(current, rate)

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

    @classmethod
    def create(
        cls,
        brain_id: str,
        storage_path: Path,
        config: OmniConfig,
    ) -> "AdaptiveBrain":
        engine_path = Path(storage_path).resolve() / "engine"
        if (engine_path / "brain.json").exists():
            return cls.load(storage_path, expected_brain_id=brain_id)
        brain = cls(brain_id, storage_path, config)
        brain.save()
        origin = brain.engine_path / "origin"
        snapshot_files(brain.engine_path, origin)
        brain.events.append(
            "brain-created",
            {
                "origin": (
                    "compatible-starter"
                    if config.origin_kind == "starter"
                    else "random-initialization"
                ),
                "coreChecksum": brain.parameter_checksum(),
                "pretrained": config.origin_kind == "starter",
            },
        )
        return brain

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
        for filename in filenames:
            source = stable / filename
            temporary = engine_path / (filename + ".recovery.tmp")
            shutil.copy2(str(source), str(temporary))
            os.replace(str(temporary), str(engine_path / filename))

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
        engine_path = Path(storage_path).resolve() / "engine"
        recovered_candidates = cls._recover_interrupted_candidates(engine_path)
        metadata = read_json(engine_path / "brain.json")
        if int(metadata.get("schema_version", 0)) != ENGINE_SCHEMA_VERSION:
            raise ValueError("unsupported OmniCortex engine schema")
        brain_id = str(metadata["brain_id"])
        if expected_brain_id is not None and str(expected_brain_id) != brain_id:
            raise ValueError("brain id does not match the requested storage path")
        config = OmniConfig.from_dict(metadata["config"])
        brain = cls(brain_id, storage_path, config)
        for _ in range(int(metadata.get("expert_count", 0))):
            brain.decoder.grow_expert()
        core = load_tensors(engine_path / "core.safetensors", device="cpu")
        plastic = load_tensors(
            engine_path / "plasticity.safetensors", device="cpu"
        )
        _load_prefixed(brain.decoder, core, "decoder.")
        _load_prefixed(brain.memory_bridge, core, "memory_bridge.")
        _load_prefixed(brain.idea_adapter, core, "idea_adapter.")
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
        brain.memory = ConceptMemory.from_state(
            metadata.get("memory", {}), plastic, prefix="memory."
        )
        replay = plastic.get("state.replay")
        if replay is not None:
            brain.replay = [row.detach().cpu() for row in replay]
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
        brain.messages = list(metadata.get("messages", []))
        brain.traces = list(metadata.get("traces", []))
        brain.training_sources = list(metadata.get("training_sources", []))
        brain.created_at = str(metadata.get("created_at", _iso_now()))
        brain.updated_at = str(metadata.get("updated_at", brain.created_at))
        brain.counters.update(
            {key: int(value) for key, value in metadata.get("counters", {}).items()}
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
        brain.novelty_streak = int(metadata.get("novelty_streak", 0))
        brain.growth_pause = metadata.get("growth_pause")
        brain.last_activity_decay = float(
            metadata.get("last_activity_decay", time.time())
        )
        brain._optimizer = brain._new_optimizer()
        for recovered in recovered_candidates:
            brain.events.append("candidate-recovered", recovered)
        return brain

    def _begin_candidate(self, kind: str) -> Tuple[str, Path]:
        candidate_id = uuid.uuid4().hex
        candidate_dir = self.engine_path / "candidates" / candidate_id
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

    def _plastic_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = _prefixed_state(self.router, "router.")
        tensors.update(self.memory.tensor_state(prefix="memory."))
        tensors["state.liquid"] = self.liquid_state.detach()
        if self.working_memory:
            tensors["state.working_memory"] = torch.stack(self.working_memory)
        if self.replay:
            tensors["state.replay"] = torch.stack(self.replay)
        self._sync_stability_state()
        for name, value in self.slow_anchors.items():
            tensors["stability.anchor." + name] = value
        for name, value in self.slow_importance.items():
            tensors["stability.importance." + name] = value
        return tensors

    def _metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "format": "omni-cortex-engine",
            "brain_id": self.brain_id,
            "name": self.config.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config.to_dict(),
            "expert_count": self.decoder.expert_count,
            "novelty_streak": self.novelty_streak,
            "growth_pause": self.growth_pause,
            "last_activity_decay": self.last_activity_decay,
            "messages": self.messages[-10000:],
            "traces": self.traces[-1000:],
            "training_sources": self.training_sources,
            "counters": self.counters,
            "modality_training": self.modality_training,
            "installed_modality_packs": self.installed_modality_packs,
            "memory": self.memory.metadata(),
            "runtime_card": self.runtime_card(),
            "files": {
                "core": "core.safetensors",
                "plasticity": "plasticity.safetensors",
                "origin": "origin/",
                "snapshots": "snapshots/",
                "artifacts": "artifacts/",
                "events": "events.sqlite3",
            },
        }

    def save(self) -> None:
        self.updated_at = _iso_now()
        self.engine_path.mkdir(parents=True, exist_ok=True)
        # Candidate tensors are fully written before metadata points at them.
        atomic_save_tensors(
            self.engine_path / "core.safetensors",
            self._core_tensors(),
            metadata={
                "format": "omni-core",
                "schema_version": str(ENGINE_SCHEMA_VERSION),
                "brain_id": self.brain_id,
            },
        )
        atomic_save_tensors(
            self.engine_path / "plasticity.safetensors",
            self._plastic_tensors(),
            metadata={
                "format": "omni-plasticity",
                "schema_version": str(ENGINE_SCHEMA_VERSION),
                "brain_id": self.brain_id,
            },
        )
        atomic_write_json(self.engine_path / "brain.json", self._metadata())

    def runtime_card(self) -> Dict[str, Any]:
        has_prior_training = (
            self.config.origin_kind == "starter"
            or self.counters["training_steps"] > 0
            or bool(self.training_sources)
        )
        return {
            "architecture": "OmniCortex",
            "pretrained": has_prior_training,
            "origin_kind": self.config.origin_kind,
            "hidden_behavioral_prompt": False,
            "reward_model": False,
            "rlhf": False,
            "memory_injection": self.config.memory_injection,
            "textual_long_term_memory_injected": False,
            "tokenizer_boundary": "UTF-8 bytes",
            "weight_forward": (
                "scaled ternary {-1,0,+1}"
                if self.config.ternary_weights
                else "dense floating point"
            ),
            "device": str(self.device),
            "device_backend": self.device_backend,
            "working_tokens": self.config.max_seq_len,
            "working_memory_vectors": len(self.working_memory),
            "expert_count": self.decoder.expert_count,
            "hardware_tier": self.config.hardware_tier,
            "scale": {
                "dimensions": self.config.d_model,
                "layers": self.config.n_layers,
                "contextTokens": self.config.max_seq_len,
                "trainBatchSize": self.config.train_batch_size,
                "gradientAccumulation": self.config.gradient_accumulation,
                "gradientCheckpointing": self.config.gradient_checkpointing,
                "replayOffload": "cpu-with-durable-safetensors",
                "imageSize": self.config.image_size,
                "audioSamples": self.config.audio_samples,
                "videoFrames": self.config.video_frames,
            },
            "active_modules": {
                "ternary": self.config.ternary_weights,
                "dense": not self.config.ternary_weights,
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
                "policy": self.config.growth_policy,
                "experts": self.decoder.expert_count,
                "elasticLimit": (
                    self.config.max_experts
                    if self.config.growth_policy == "elastic"
                    else None
                ),
                "paused": self.growth_pause is not None,
                "pause": self.growth_pause,
                "memoryCapacities": {
                    "concepts": self.memory.max_concepts,
                    "ideas": self.memory.max_ideas,
                    "synapses": self.memory.max_relations,
                    "expansions": self.memory.capacity_expansions,
                },
            },
            "memory_timescales": {
                "workingMemorySlots": self.config.working_memory_slots,
                "shortTermHalfLifeMinutes": self.config.short_term_half_life_minutes,
                "longTermThreshold": self.config.long_term_threshold,
                "forgettingRate": self.config.forgetting_rate,
                "consolidationRate": self.config.consolidation_rate,
                "activeWorkingVectors": len(self.working_memory),
                "injectionChannel": (
                    "internal-recurrent-vectors"
                    if self.config.memory_injection == "working-memory"
                    else "semantic-parameters-and-vsa"
                ),
            },
            "modality_training": {
                name: {
                    "steps": steps,
                    "initialized": (
                        "trained"
                        if steps > 0
                        else (
                            "starter"
                            if self.config.origin_kind == "starter"
                            else "random"
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
            "intrinsic_drives": {
                "novelty": self.config.novelty_drive,
                "coherence": self.config.coherence_drive,
                "curiosity": self.config.curiosity_drive,
                "parallelThoughts": self.config.parallel_thoughts,
            },
        }

    def parameter_checksum(self) -> str:
        parameters: List[torch.Tensor] = [
            parameter
            for module in self._trainable_modules()
            for parameter in module.parameters()
        ]
        return tensor_checksum(parameters)

    def _parameter_copy(self) -> List[torch.Tensor]:
        return [
            parameter.detach().cpu().clone()
            for module in (
                self.decoder,
                self.memory_bridge,
                self.idea_adapter,
                self.liquid,
            )
            for parameter in module.parameters()
        ]

    def _parameter_delta_norm(self, before: Sequence[torch.Tensor]) -> float:
        total = 0.0
        current = [
            parameter.detach().cpu()
            for module in (
                self.decoder,
                self.memory_bridge,
                self.idea_adapter,
                self.liquid,
            )
            for parameter in module.parameters()
        ]
        for index, parameter in enumerate(current):
            if index < len(before) and parameter.shape == before[index].shape:
                difference = parameter - before[index]
            else:
                difference = parameter
            total += float(difference.float().pow(2).sum().item())
        return math.sqrt(total)

    def _idea_model_vector(self, vsa_vector: torch.Tensor) -> torch.Tensor:
        raw = vsa_vector.to(self.device, dtype=torch.float32).reshape(1, -1)
        return torch.tanh(self.memory_bridge(raw))

    def _append_replay(self, idea: torch.Tensor, importance: float = 1.0) -> None:
        if float(importance) < self.config.long_term_threshold:
            return
        self.replay.append(idea.detach().cpu().reshape(-1))
        if len(self.replay) > self.config.replay_capacity:
            # Deterministic reservoir-like thinning retains old and new eras.
            self.replay = self.replay[::2] + self.replay[-self.config.replay_capacity // 2 :]
            self.replay = self.replay[-self.config.replay_capacity :]

    def _append_working_memory(self, idea: torch.Tensor) -> None:
        self.working_memory.append(idea.detach().cpu().reshape(-1))
        self.working_memory = self.working_memory[
            -self.config.working_memory_slots :
        ]

    def _working_memory_vector(self) -> Optional[torch.Tensor]:
        if (
            self.config.memory_injection != "working-memory"
            or not self.working_memory
        ):
            return None
        recent = torch.stack(self.working_memory).to(
            self.device, dtype=torch.float32
        )
        # Recency-weighted activity is a bounded recurrent state, not prose.
        weights = torch.linspace(
            0.35, 1.0, recent.shape[0], device=self.device
        )
        weights = weights / weights.sum()
        return torch.tanh((recent * weights[:, None]).sum(dim=0, keepdim=True))

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
            optimizer = self._optimizer
        ids = self.tokenizer.tensor(
            text,
            self.device,
            max_length=self.config.max_seq_len,
            add_bos=True,
            add_eos=True,
        )
        if ids.shape[1] < 2:
            return {
                "loss": 0.0,
                "language_loss": 0.0,
                "idea_loss": 0.0,
                "stability_loss": 0.0,
            }
        losses: List[float] = []
        language_losses: List[float] = []
        idea_losses: List[float] = []
        stability_losses: List[float] = []
        self.decoder.train()
        self.memory_bridge.train()
        self.idea_adapter.train()
        self.liquid.train()
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            idea = self._idea_model_vector(vsa_vector)
            noise = torch.randn_like(idea) * 0.06
            reconstructed = self.idea_adapter(idea + noise)
            idea_loss = F.mse_loss(reconstructed, idea.detach())
            temporal, _ = self.liquid(
                idea, state=self.liquid_state.detach(), elapsed=1.0
            )
            temporal_loss = F.mse_loss(temporal, idea.detach())
            language = self.decoder(
                ids, memory_bias=reconstructed, labels=ids
            )["loss"]
            stability_loss = self._stability_penalty()
            loss = (
                language
                + 0.2 * idea_loss
                + 0.05 * temporal_loss
                + stability_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            self._accumulate_slow_importance()
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            language_losses.append(float(language.detach().item()))
            idea_losses.append(float(idea_loss.detach().item()))
            stability_losses.append(float(stability_loss.detach().item()))
            self.counters["training_steps"] += 1
        if commit_stability:
            self._commit_slow_anchors(rate=0.08)
        return {
            "loss": sum(losses) / len(losses),
            "language_loss": sum(language_losses) / len(language_losses),
            "idea_loss": sum(idea_losses) / len(idea_losses),
            "stability_loss": sum(stability_losses) / len(stability_losses),
        }

    def _optimize_dialogue_pair(
        self,
        human: str,
        brain: str,
        vsa_vector: torch.Tensor,
        steps: int = 1,
        commit_stability: bool = True,
    ) -> Dict[str, float]:
        ids_list = self.tokenizer.dialogue(human, brain, complete=True)
        if len(ids_list) > self.config.max_seq_len:
            # Preserve the role boundary and response when a long human turn is
            # clipped to the physical working-token budget.
            response_ids = [
                value + self.tokenizer.byte_offset
                for value in brain.encode("utf-8")
            ]
            response_ids = response_ids[-max(1, self.config.max_seq_len // 2) :]
            human_budget = max(
                1, self.config.max_seq_len - len(response_ids) - 4
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
        losses: List[float] = []
        for _ in range(max(1, int(steps))):
            self._optimizer.zero_grad(set_to_none=True)
            idea = self._idea_model_vector(vsa_vector)
            adapted = self.idea_adapter(idea)
            prediction_loss = self.decoder(
                ids, memory_bias=adapted, labels=labels
            )["loss"]
            stability_loss = self._stability_penalty()
            loss = prediction_loss + stability_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite dialogue-pair loss")
            loss.backward()
            self._accumulate_slow_importance()
            parameters = [
                parameter
                for group in self._optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
            self._optimizer.step()
            self.counters["training_steps"] += 1
            losses.append(float(loss.detach().item()))
        if commit_stability:
            self._commit_slow_anchors(rate=0.08)
        return {"loss": sum(losses) / len(losses)}

    def _experience_batch_loss(
        self,
        texts: Sequence[str],
        vsa_vectors: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute one padded micro-batch without stepping the optimizer."""

        encoded = [
            self.tokenizer.tensor(
                text,
                self.device,
                max_length=self.config.max_seq_len,
                add_bos=True,
                add_eos=True,
            )[0]
            for text in texts
        ]
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
        language = self.decoder(
            ids, memory_bias=reconstructed, labels=ids
        )["loss"]
        stability_loss = self._stability_penalty()
        loss = (
            language
            + 0.2 * idea_loss
            + 0.05 * temporal_loss
            + stability_loss
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite batch training loss")
        return loss, {
            "loss": float(loss.detach().item()),
            "language_loss": float(language.detach().item()),
            "idea_loss": float(idea_loss.detach().item()),
            "stability_loss": float(stability_loss.detach().item()),
        }

    def _maybe_grow(self, novelty: float, prototype: torch.Tensor) -> bool:
        if self.config.growth_policy == "fixed":
            return False
        if novelty >= self.config.growth_novelty_threshold:
            self.novelty_streak += 1
        else:
            self.novelty_streak = max(0, self.novelty_streak - 1)
        if (
            self.config.growth_policy == "elastic"
            and self.decoder.expert_count >= self.config.max_experts
        ):
            self.growth_pause = {
                "reason": "elastic expert limit reached",
                "readings": self._resource_readings(),
                "at": _iso_now(),
            }
            return False
        if self.novelty_streak < self.config.growth_patience:
            return False
        if self.config.growth_policy == "unbounded":
            readings = self._resource_readings()
            expert_parameters = (
                self.config.d_model * max(16, self.config.d_ff // 2) * 3
            )
            estimated_bytes = expert_parameters * 12
            disk_free = readings.get("diskFreeBytes")
            ram_free = readings.get("availableMemoryBytes")
            reason = ""
            if isinstance(disk_free, int) and disk_free < max(
                256 * 1024 * 1024, estimated_bytes * 12
            ):
                reason = "available disk is below the growth reserve"
            elif isinstance(ram_free, int) and ram_free < max(
                256 * 1024 * 1024, estimated_bytes * 6
            ):
                reason = "available memory is below the growth reserve"
            if reason:
                self.growth_pause = {
                    "reason": reason,
                    "readings": readings,
                    "estimatedExpertBytes": estimated_bytes,
                    "at": _iso_now(),
                }
                return False
        self.decoder.grow_expert(prototype.detach().reshape(-1))
        self.novelty_streak = 0
        self.growth_pause = None
        self._optimizer = self._new_optimizer()
        self._sync_stability_state()
        return True

    def _maybe_expand_memory(self) -> bool:
        """Expand sparse idea storage for unbounded builds while resources allow."""

        if self.config.growth_policy != "unbounded":
            return False
        pressure = (
            len(self.memory.concepts) >= self.memory.max_concepts,
            len(self.memory.ideas) >= self.memory.max_ideas,
            len(self.memory.relations) >= self.memory.max_relations,
        )
        if not any(pressure):
            return False
        next_capacities = {
            "concepts": (
                max(self.memory.max_concepts + 64, self.memory.max_concepts * 2)
                if pressure[0]
                else self.memory.max_concepts
            ),
            "ideas": (
                max(self.memory.max_ideas + 32, self.memory.max_ideas * 2)
                if pressure[1]
                else self.memory.max_ideas
            ),
            "relations": (
                max(self.memory.max_relations + 256, self.memory.max_relations * 2)
                if any(pressure)
                else self.memory.max_relations
            ),
        }
        added_vectors = (
            next_capacities["concepts"]
            - self.memory.max_concepts
            + next_capacities["ideas"]
            - self.memory.max_ideas
        )
        added_relations = (
            next_capacities["relations"] - self.memory.max_relations
        )
        estimated_bytes = (
            added_vectors * self.config.vsa_dim * 4
            + added_relations * 512
        )
        readings = self._resource_readings()
        disk_free = readings.get("diskFreeBytes")
        ram_free = readings.get("availableMemoryBytes")
        reason = ""
        if isinstance(disk_free, int) and disk_free < max(
            256 * 1024 * 1024, estimated_bytes * 8
        ):
            reason = "available disk is below the sparse-memory growth reserve"
        elif isinstance(ram_free, int) and ram_free < max(
            256 * 1024 * 1024, estimated_bytes * 4
        ):
            reason = "available memory is below the sparse-memory growth reserve"
        if reason:
            self.growth_pause = {
                "reason": reason,
                "readings": readings,
                "estimatedMemoryBytes": estimated_bytes,
                "at": _iso_now(),
            }
            return False
        before = {
            "concepts": self.memory.max_concepts,
            "ideas": self.memory.max_ideas,
            "relations": self.memory.max_relations,
        }
        after = self.memory.expand_capacities(
            concepts=next_capacities["concepts"],
            ideas=next_capacities["ideas"],
            relations=next_capacities["relations"],
        )
        self.growth_pause = None
        self.events.append(
            "memory-capacity-growth",
            {
                "before": before,
                "after": after,
                "estimatedAdditionalBytes": estimated_bytes,
                "readings": readings,
            },
        )
        return True

    def _resource_readings(self) -> Dict[str, Any]:
        disk = shutil.disk_usage(str(self.engine_path))
        available_memory: Optional[int] = None
        try:
            if os.name == "nt":
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("length", ctypes.c_ulong),
                        ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong),
                        ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.length = ctypes.sizeof(MemoryStatus)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                available_memory = int(status.available_physical)
            elif hasattr(os, "sysconf"):
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                available_memory = int(pages * page_size)
        except (OSError, ValueError, AttributeError):
            available_memory = None
        return {
            "diskFreeBytes": int(disk.free),
            "availableMemoryBytes": available_memory,
        }

    def learn_experience(
        self,
        text: str,
        kind: str = "knowledge",
        source: str = "conversation",
        source_label: str = "",
        steps: Optional[int] = None,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        now = time.time()
        elapsed = max(0.0, now - self.last_activity_decay)
        half_life = self.config.short_term_half_life_minutes * 60.0
        if elapsed > 0 and self.config.vector_symbolic_memory:
            activity_decay = 1.0 - math.exp(
                -math.log(2.0) * elapsed / max(half_life, 1.0)
            )
            self.memory.decay(min(activity_decay, 0.25))
        self.last_activity_decay = now
        if self.config.vector_symbolic_memory:
            self._maybe_expand_memory()
            retain = (
                self.config.memory_recipe == "total-recall"
                and self.config.retain_source_text
            )
            learned = self.memory.learn(
                text,
                kind=kind,
                source=source,
                source_label=source_label,
                retain_source_text=retain,
                importance=importance,
            )
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
        if self.config.online_learning and int(requested_steps) > 0:
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
        self._append_replay(routed, importance=importance)
        self._append_working_memory(routed)
        grew = self._maybe_grow(float(learned["novelty"]), routed[0])
        self.counters["experiences"] += 1
        self.counters["plasticity_events"] = int(
            self.router.synapses.plasticity_events.item()
        )
        return {
            "idea_id": learned["idea_id"],
            "concept_ids": learned["concept_ids"],
            "labels": learned["labels"],
            "novelty": learned["novelty"],
            "idea": routed.detach(),
            "spiking": spike_metrics,
            "liquid_controls": {
                key: float(value.detach().mean().item())
                for key, value in controls.items()
            },
            "training": train_result,
            "grew_expert": grew,
        }

    @staticmethod
    def _normalize_tool_schemas(
        schemas: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Keep only bounded structural tool identifiers and action names."""

        if schemas is None:
            return []
        if not isinstance(schemas, (list, tuple)):
            raise ValueError("tool schemas must be a list")
        if len(schemas) > 100:
            raise ValueError("at most 100 tool schemas may be active")
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
            if len(actions) > 32:
                raise ValueError("a tool schema may expose at most 32 actions")
            clean_actions = []
            for action in actions:
                if not isinstance(action, str) or not action.strip():
                    raise ValueError("tool actions must be non-empty strings")
                action = action.strip()
                if len(action) > 64:
                    raise ValueError("tool action exceeds 64 characters")
                clean_actions.append(action)
            grant = str(schema.get("grant", "ask")).strip()[:32] or "ask"
            existing = normalized.setdefault(
                tool_id, {"id": tool_id, "actions": [], "grant": grant}
            )
            existing["actions"] = sorted(
                set(existing["actions"]).union(clean_actions)
            )
        return [normalized[key] for key in sorted(normalized)]

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
            vectors.append(self.memory.space.bundle(parts))
        if not vectors:
            return None
        bundled = self.memory.space.bundle(vectors)
        return self._idea_model_vector(bundled)

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

    def chat(
        self,
        text: str,
        max_new_tokens: int = 48,
        seed: Optional[int] = None,
        tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            raise ValueError("chat input cannot be empty")
        if len(clean) > 1_000_000:
            raise ValueError("chat input is too large")
        before_checksum = self.parameter_checksum()
        before_parameters = self._parameter_copy()
        normalized_tools = self._normalize_tool_schemas(tool_schemas)

        cue = self.memory.vector_for_text(clean)
        if self.config.vector_symbolic_memory:
            recalled_vector, recalled = self.memory.recall_vector(
                cue, limit=min(self.config.working_memory_slots, 16)
            )
        else:
            recalled_vector, recalled = cue, []
        experience = self.learn_experience(
            clean,
            kind="question" if clean.rstrip().endswith("?") else "experience",
            source="conversation",
            source_label="chat",
            steps=self.config.online_steps,
            importance=0.7,
        )
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
        prompt_list = self.tokenizer.dialogue(clean, brain="", complete=False)
        prompt_ids = torch.tensor(
            [prompt_list[-self.config.max_seq_len :]],
            dtype=torch.long,
            device=self.device,
        )
        prompt_token_hash = hashlib.sha256(
            ",".join(str(value) for value in prompt_ids[0].tolist()).encode(
                "ascii"
            )
        ).hexdigest()
        liquid_noise = experience["liquid_controls"]["noise_scale"]
        uncertainties = [
            float(self.memory.concepts[concept_id]["uncertainty"])
            for concept_id in experience["concept_ids"]
            if concept_id in self.memory.concepts
        ]
        uncertainty = (
            sum(uncertainties) / len(uncertainties) if uncertainties else 0.5
        )
        ponder_factors = {
            "liquid": float(experience["liquid_controls"]["ponder_scale"]),
            "novelty": float(experience["novelty"])
            * self.config.novelty_drive,
            "uncertainty": uncertainty * self.config.curiosity_drive,
        }
        ponder_steps = max(
            1,
            int(
                round(
                    ponder_factors["liquid"]
                    + 2.0 * ponder_factors["novelty"]
                    + 2.0 * ponder_factors["uncertainty"]
                )
            ),
        )
        branch_count = max(1, min(self.config.parallel_thoughts, 8))
        candidates = []
        for branch in range(branch_count):
            branch_seed = int(seed) + branch * 7919
            candidate, branch_entropies = self.decoder.generate(
                prompt_ids,
                memory_bias=internal_memory,
                max_new_tokens=max(1, min(int(max_new_tokens), 512)),
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                noise=self.config.noise
                * liquid_noise
                * (1.0 + 0.04 * ponder_steps),
                seed=branch_seed,
                printable_only=True,
            )
            nll = self._candidate_nll(
                prompt_ids, candidate, internal_memory
            )
            entropy = (
                sum(branch_entropies) / len(branch_entropies)
                if branch_entropies
                else 0.0
            )
            intrinsic_score = (
                -self.config.coherence_drive * nll
                + self.config.curiosity_drive * entropy * 0.08
                + self.config.novelty_drive
                * float(experience["novelty"])
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
                }
            )
        selected_branch = max(
            range(len(candidates)), key=lambda index: candidates[index]["score"]
        )
        selected = candidates[selected_branch]
        generated = selected["tensor"]
        entropies = selected["entropies"]
        new_ids = generated[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        response = self.tokenizer.decode(new_ids).strip()
        if not response:
            # The first generated byte is constrained to visible ASCII, so this
            # only covers a pathological tokenizer/checkpoint corruption case.
            response = self.tokenizer.decode(new_ids, skip_special=False) or "?"

        own_training = None
        if self.config.learn_from_own_messages:
            own_training = self.learn_experience(
                response,
                kind="experience",
                source="self",
                source_label="self-response",
                steps=0,
                importance=0.35,
            )
        pair_training = None
        if self.config.online_learning:
            pair_training = self._optimize_dialogue_pair(
                clean, response, cue, steps=1
            )

        with torch.no_grad():
            routing_output = self.decoder(
                prompt_ids,
                memory_bias=internal_memory,
            )
            expert_route = (
                routing_output["expert_routing"][0].detach().cpu().tolist()
                if "expert_routing" in routing_output
                else []
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
        }
        assistant_message = {
            "id": uuid.uuid4().hex,
            "role": "brain",
            "content": response,
            "created_at": _iso_now(),
        }
        self.messages.extend([user_message, assistant_message])
        train_loss = float(experience["training"]["loss"])
        if pair_training is not None:
            train_loss = (train_loss + float(pair_training["loss"])) / 2.0
        trace = {
            "id": uuid.uuid4().hex,
            "created_at": _iso_now(),
            "seed": int(seed),
            "input_sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
            "textual_memory_injected": False,
            "tool_schema_text_injected": False,
            "prompt_text_expanded": False,
            "prompt_token_count": int(prompt_ids.shape[1]),
            "prompt_token_ids_sha256": prompt_token_hash,
            "available_tool_ids": [
                schema["id"] for schema in normalized_tools
            ],
            "available_tool_actions": {
                schema["id"]: list(schema["actions"])
                for schema in normalized_tools
            },
            "tool_schema_channel": (
                "vsa-internal" if normalized_tools else "none"
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
            "liquid_controls": experience["liquid_controls"],
            "recalled_idea_ids": [
                item["idea_id"] for item in recalled
            ],
            "expert_route": expert_route,
            "expert_grew": bool(experience["grew_expert"]),
            "growth_pause": self.growth_pause,
            "train_loss": train_loss,
            "generation_entropy": (
                sum(entropies) / len(entropies) if entropies else 0.0
            ),
            "ponder_steps": ponder_steps,
            "ponder_factors": ponder_factors,
            "branches": [
                {
                    "index": index,
                    "seed": candidate["seed"],
                    "selfNll": candidate["selfNll"],
                    "entropy": candidate["entropy"],
                    "intrinsicScore": candidate["score"],
                }
                for index, candidate in enumerate(candidates)
            ],
            "selected_branch": selected_branch,
            "steps": [
                {
                    "stage": "encode",
                    "detail": "Encoded the current turn at the UTF-8 token boundary.",
                    "value": "%d tokens" % prompt_ids.shape[1],
                },
                {
                    "stage": "idea-memory",
                    "detail": "Activated compositional hypervectors; no remembered source text entered the token stream.",
                    "value": "%d recalled idea vectors" % len(recalled),
                },
                {
                    "stage": "working-memory",
                    "detail": (
                        "Blended bounded recurrent activity vectors; no prior "
                        "message text was added to prompt tokens."
                        if working_model is not None
                        else "Recurrent working-memory injection is disabled."
                    ),
                    "value": (
                        "%d active vectors" % working_memory_used
                        if working_model is not None
                        else "parameter-only mode"
                    ),
                },
                {
                    "stage": "tool-schema",
                    "detail": (
                        "Encoded enabled tool IDs and actions through the "
                        "internal VSA channel; no schema text was added to "
                        "prompt tokens."
                    ),
                    "value": "%d available tools" % len(normalized_tools),
                },
                {
                    "stage": "plasticity",
                    "detail": "Applied local spike-timing-dependent synaptic updates.",
                    "value": "%.6f L1 update" % experience["spiking"]["stdp_update"],
                },
                {
                    "stage": "slow-learning",
                    "detail": "Updated ternary decoder and idea-consolidation master parameters.",
                    "value": "loss %.6f" % train_loss,
                },
                {
                    "stage": "dialogue-learning",
                    "detail": "Learned the completed human-to-brain role-boundary sequence.",
                    "value": (
                        "loss %.6f" % pair_training["loss"]
                        if pair_training is not None
                        else "online slow learning disabled"
                    ),
                },
                {
                    "stage": "generation",
                    "detail": "Sampled from the brain's own decoder with liquid-controlled noise.",
                    "value": "%d generated tokens" % len(new_ids),
                },
            ],
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
            },
        )
        self.save()
        runtime_card = self.runtime_card()
        runtime_card["available_tool_ids"] = trace["available_tool_ids"]
        runtime_card["tool_schema_channel"] = trace["tool_schema_channel"]
        runtime_card["tool_schema_text_injected"] = False
        return {
            "brainId": self.brain_id,
            "text": response,
            "response": response,
            "content": response,
            "message": assistant_message,
            "trace": trace,
            "metrics": self.metrics(),
            "runtimeCard": runtime_card,
            "availableToolIds": trace["available_tool_ids"],
        }

    @torch.no_grad()
    def _evaluate_experience(
        self, text: str, vsa_vector: torch.Tensor
    ) -> float:
        ids = self.tokenizer.tensor(
            text,
            self.device,
            max_length=self.config.max_seq_len,
            add_bos=True,
            add_eos=True,
        )
        self.decoder.eval()
        self.memory_bridge.eval()
        self.idea_adapter.eval()
        self.liquid.eval()
        idea = self._idea_model_vector(vsa_vector)
        reconstructed = self.idea_adapter(idea)
        temporal, _ = self.liquid(
            idea, state=self.liquid_state.detach(), elapsed=1.0
        )
        language = self.decoder(
            ids, memory_bias=reconstructed, labels=ids
        )["loss"]
        return float(
            (
                language
                + 0.2 * F.mse_loss(reconstructed, idea)
                + 0.05 * F.mse_loss(temporal, idea)
            ).item()
        )

    def _restore_core(self, tensors: Mapping[str, torch.Tensor]) -> None:
        _load_prefixed(self.decoder, tensors, "decoder.")
        _load_prefixed(self.memory_bridge, tensors, "memory_bridge.")
        _load_prefixed(self.idea_adapter, tensors, "idea_adapter.")
        _load_prefixed(self.liquid, tensors, "liquid.")
        _load_prefixed(self.modalities, tensors, "modalities.")

    def train(
        self,
        texts: Optional[Sequence[str]] = None,
        epochs: int = 1,
        learning_rate: Optional[float] = None,
        source_ids: Optional[Sequence[str]] = None,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        epochs = max(1, min(int(epochs), 10000))
        samples = [
            text.replace("\x00", "")
            for text in (texts or [])
            if isinstance(text, str) and text.strip()
        ]
        selected = set(source_ids or [])
        if not samples:
            for source in self.training_sources:
                if selected and source.get("id") not in selected:
                    continue
                retained = source.get("raw_text")
                if isinstance(retained, str) and retained.strip():
                    samples.append(retained)
        if not samples:
            return self.consolidate(steps=epochs, progress=progress)

        before = self.parameter_checksum()
        training_steps_before = self.counters["training_steps"]
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
        total = epochs * len(samples)
        completed = 0
        optimizer_steps = 0
        optimizer = (
            self._new_optimizer(learning_rate)
            if learning_rate is not None
            else self._optimizer
        )
        physical_batch = max(1, int(self.config.train_batch_size))
        accumulation = max(1, int(self.config.gradient_accumulation))
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
                for group_start in range(0, len(samples), effective_batch):
                    group = samples[group_start : group_start + effective_batch]
                    micro_batches = [
                        group[index : index + physical_batch]
                        for index in range(0, len(group), physical_batch)
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    for micro_batch in micro_batches:
                        vectors = [
                            self.memory.vector_for_text(sample)
                            for sample in micro_batch
                        ]
                        loss, measurements = self._experience_batch_loss(
                            micro_batch, vectors
                        )
                        (loss / float(len(micro_batches))).backward()
                        losses.append(measurements["loss"])
                        for vector in vectors:
                            self._append_replay(
                                self._idea_model_vector(vector)
                            )
                        completed += len(micro_batch)
                        if progress is not None:
                            progress(
                                completed / float(total),
                                "Training candidate",
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
            self.replay = self.replay[:replay_length]
            self.counters["training_steps"] = training_steps_before
            self._optimizer = self._new_optimizer()
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
            self.replay = self.replay[:replay_length]
            self.counters["training_steps"] = training_steps_before
            self._optimizer = self._new_optimizer()
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

    def consolidate(
        self, steps: int = 4, progress: Optional[Any] = None
    ) -> Dict[str, Any]:
        if not self.config.consolidation_enabled:
            return {
                "brainId": self.brain_id,
                "disabled": True,
                "steps": 0,
                "promoted": False,
                "rejection": "consolidation is disabled by this brain recipe",
                "metrics": self.metrics(),
            }
        steps = max(1, min(int(steps), 10000))
        before_checksum = self.parameter_checksum()
        if not self.replay:
            self.memory.decay(self.config.forgetting_rate)
            self.router.synapses.decay_unused(
                self.config.forgetting_rate * 0.5
            )
            self.counters["consolidation_cycles"] += 1
            self.save()
            self.events.append(
                "consolidation",
                {
                    "steps": 0,
                    "replayExamples": 0,
                    "parameterChecksumBefore": before_checksum,
                    "parameterChecksumAfter": self.parameter_checksum(),
                },
            )
            return {
                "brainId": self.brain_id,
                "steps": 0,
                "meanLoss": 0.0,
                "parameterChecksumBefore": before_checksum,
                "parameterChecksumAfter": self.parameter_checksum(),
                "promoted": True,
                "metrics": self.metrics(),
            }

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
        candidate_id, candidate_dir = self._begin_candidate("consolidation")
        promoted = False
        rejection = ""
        try:
            for index in range(steps):
                target = self.replay[index % len(self.replay)].to(self.device).reshape(1, -1)
                optimizer.zero_grad(set_to_none=True)
                noisy = target + torch.randn_like(target) * (
                    0.03 + self.config.noise * 0.05
                )
                prediction = self.idea_adapter(noisy)
                reconstruction_loss = F.mse_loss(prediction, target.detach())
                stability_loss = self._stability_penalty(
                    self.idea_adapter.parameters()
                )
                loss = reconstruction_loss + stability_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite consolidation loss")
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
                        "Consolidating candidate latent replay",
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
            self._optimizer = self._new_optimizer()
            self._restore_candidate_checkpoint(candidate_dir)
            self._record_candidate(
                candidate_dir,
                status="rejected",
                reason="consolidation exception",
                rejectedAt=_iso_now(),
            )
            raise
        if not promoted:
            self._restore_core(backup)
            self._restore_stability(stability_backup)
            self.counters["training_steps"] = training_steps_before
            self._optimizer = self._new_optimizer()
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
            "consolidation",
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
        size = path.stat().st_size
        if size > 128 * 1024 * 1024:
            raise ValueError("ingestion file exceeds the 128 MB limit")
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
    def _experience_chunks(text: str, limit: int = 64) -> List[str]:
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
            if pending and len(pending) + len(piece) + 1 <= 900:
                pending += " " + piece
            else:
                if pending:
                    chunks.append(pending)
                pending = piece[:4000]
            if len(chunks) >= limit:
                break
        if pending and len(chunks) < limit:
            chunks.append(pending)
        return chunks

    def _media_idea(self, source_name: str) -> torch.Tensor:
        vector = self.memory.vector_for_text(source_name or "media experience")
        return self._idea_model_vector(vector).detach()

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

    def _decode_wav(self, path: str) -> torch.Tensor:
        suffix = Path(path).suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(path, "rb") as handle:
                    channels = handle.getnchannels()
                    width = handle.getsampwidth()
                    frames = handle.getnframes()
                    raw = handle.readframes(frames)
                if width == 1:
                    values = torch.tensor(
                        list(raw), dtype=torch.float32
                    ).sub_(128.0)
                    values.div_(128.0)
                elif width == 2:
                    samples = array.array("h")
                    samples.frombytes(raw)
                    if os.sys.byteorder != "little":
                        samples.byteswap()
                    values = torch.tensor(
                        samples, dtype=torch.float32
                    ).div_(32768.0)
                elif width == 4:
                    samples = array.array("i")
                    samples.frombytes(raw)
                    if os.sys.byteorder != "little":
                        samples.byteswap()
                    values = torch.tensor(
                        samples, dtype=torch.float32
                    ).div_(2147483648.0)
                else:
                    raise ValueError(
                        "only 8, 16, or 32-bit PCM WAV is supported"
                    )
                if channels > 1:
                    values = values[
                        : values.numel() - values.numel() % channels
                    ]
                    values = values.reshape(-1, channels).mean(dim=1)
            except (wave.Error, EOFError):
                values = torch.empty(0)
        else:
            values = torch.empty(0)
        if values.numel() == 0:
            try:
                import soundfile
                decoded, _sample_rate = soundfile.read(
                    path, dtype="float32", always_2d=True
                )
                values = torch.from_numpy(decoded).float().mean(dim=1)
            except (ImportError, RuntimeError):
                values = torch.empty(0)
        if values.numel() == 0:
            try:
                import imageio_ffmpeg

                executable = imageio_ffmpeg.get_ffmpeg_exe()
                decoded = subprocess.run(
                    [
                        executable,
                        "-v",
                        "error",
                        "-i",
                        str(Path(path).resolve()),
                        "-t",
                        "60",
                        "-f",
                        "f32le",
                        "-ac",
                        "1",
                        "-",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                ).stdout
                samples = array.array("f")
                samples.frombytes(decoded)
                if os.sys.byteorder != "little":
                    samples.byteswap()
                values = torch.tensor(samples, dtype=torch.float32)
            except (
                ImportError,
                OSError,
                subprocess.SubprocessError,
            ) as error:
                raise RuntimeError(
                    "audio format needs soundfile or the bundled FFmpeg decoder"
                ) from error
        if values.numel() == 0:
            raise ValueError("audio contained no decodable samples")
        values = F.interpolate(
            values.reshape(1, 1, -1),
            size=self.config.audio_samples,
            mode="linear",
            align_corners=False,
        )
        return values.to(self.device)

    def _decode_video(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image, ImageSequence
            import numpy as np
        except ImportError as error:
            raise RuntimeError("video training requires Pillow and NumPy") from error
        frames = []
        suffix = Path(path).suffix.lower()
        if suffix in {".gif", ".webp"}:
            with Image.open(path) as opened:
                for frame in ImageSequence.Iterator(opened):
                    frames.append(
                        np.asarray(
                            frame.convert("RGB").resize(
                                (self.config.image_size, self.config.image_size)
                            ),
                            dtype="float32",
                        ).copy()
                    )
        else:
            try:
                import imageio_ffmpeg

                executable = imageio_ffmpeg.get_ffmpeg_exe()
                with tempfile.TemporaryDirectory(
                    prefix="omni-video-decode-"
                ) as temporary:
                    output_pattern = str(
                        Path(temporary) / "frame-%06d.png"
                    )
                    subprocess.run(
                        [
                            executable,
                            "-v",
                            "error",
                            "-i",
                            str(Path(path).resolve()),
                            "-frames:v",
                            str(max(16, self.config.video_frames * 8)),
                            output_pattern,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    for frame_path in sorted(Path(temporary).glob("frame-*.png")):
                        with Image.open(frame_path) as opened_frame:
                            frame = opened_frame.convert("RGB").resize(
                                (
                                    self.config.image_size,
                                    self.config.image_size,
                                )
                            )
                            frames.append(
                                np.asarray(
                                    frame,
                                    dtype="float32",
                                ).copy()
                            )
            except (
                ImportError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                raise RuntimeError(
                    "MP4/WebM/MOV video needs the bundled FFmpeg decoder; GIF works with Pillow"
                ) from error
        if not frames:
            raise ValueError("video contained no decodable frames")
        indices = torch.linspace(
            0, len(frames) - 1, self.config.video_frames
        ).round().to(torch.long)
        selected = [frames[int(index)] for index in indices]
        tensor = torch.stack(
            [torch.from_numpy(frame).permute(2, 0, 1) for frame in selected],
            dim=1,
        )
        return tensor.unsqueeze(0).to(self.device) / 127.5 - 1.0

    def _train_media(
        self,
        path: str,
        kind: str,
        source_name: str,
        steps: int = 2,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        idea = self._media_idea(source_name)
        if kind == "image":
            if not (self.config.image_enabled or self.config.vision_enabled):
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Image and vision packs are disabled."],
                }
            target = self._decode_image(path)
            parameters = []
            if self.config.image_enabled:
                parameters.extend(self.modalities.image.parameters())
            if self.config.vision_enabled:
                parameters.extend(self.modalities.vision.parameters())
        elif kind == "audio":
            if not self.config.audio_enabled:
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Audio pack is disabled."],
                }
            target = self._decode_wav(path)
            parameters = list(self.modalities.audio.parameters())
        elif kind == "video":
            if not self.config.video_enabled:
                return {
                    "trained": False,
                    "loss": 0.0,
                    "warnings": ["Video pack is disabled."],
                }
            target = self._decode_video(path)
            parameters = list(self.modalities.video.parameters())
        else:
            return {
                "trained": False,
                "loss": 0.0,
                "warnings": ["Unsupported binary modality; no parameters changed."],
            }
        optimizer = torch.optim.AdamW(
            parameters, lr=self.config.learning_rate, weight_decay=1e-5
        )
        losses = []
        for index in range(max(1, steps)):
            optimizer.zero_grad(set_to_none=True)
            if kind == "image":
                components = []
                if self.config.image_enabled:
                    components.append(self.modalities.image(target, idea)["loss"])
                if self.config.vision_enabled:
                    embedding = self.modalities.vision(target)
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
                loss = output["loss"] + 0.2 * (
                    1.0
                    - F.cosine_similarity(
                        output["embedding"], F.normalize(idea, dim=-1)
                    ).mean()
                )
            else:
                output = self.modalities.video(target, idea)
                loss = output["loss"] + 0.2 * (
                    1.0
                    - F.cosine_similarity(
                        output["embedding"], F.normalize(idea, dim=-1)
                    ).mean()
                )
            stability_loss = self._stability_penalty(parameters)
            loss = loss + stability_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite modality training loss")
            loss.backward()
            self._accumulate_slow_importance(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            self.counters["training_steps"] += 1
            if progress is not None:
                progress(
                    (index + 1) / float(max(1, steps)),
                    "Training %s modality pack" % kind,
                )
        self._commit_slow_anchors(rate=0.12)
        if kind == "image":
            if self.config.image_enabled:
                self.modality_training["image"] += len(losses)
            if self.config.vision_enabled:
                self.modality_training["vision"] += len(losses)
        else:
            self.modality_training[kind] += len(losses)
        return {
            "trained": True,
            "loss": sum(losses) / len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "steps": len(losses),
            "warnings": warnings,
        }

    def ingest(
        self,
        path: Optional[str] = None,
        text: Optional[str] = None,
        name: str = "",
        kind: str = "",
        policy: str = "encode",
        expected_hash: str = "",
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if policy not in {"encode", "consolidate", "pretrain", "archive"}:
            raise ValueError("unsupported ingestion policy")
        raw_bytes: bytes
        if path:
            source_path = Path(path).resolve()
            extracted, raw_bytes, resolved_kind = self._read_document(
                source_path, kind=kind
            )
            source_name = name or source_path.name
        elif text is not None:
            extracted = str(text).replace("\x00", "")
            raw_bytes = extracted.encode("utf-8")
            resolved_kind = kind or "text"
            source_name = name or "pasted-text"
        else:
            raise ValueError("ingest requires path or text")
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        if expected_hash and expected_hash.lower() != content_hash:
            raise ValueError("ingestion content hash mismatch")
        duplicate = next(
            (
                source
                for source in self.training_sources
                if source.get("content_hash") == content_hash
            ),
            None,
        )
        if duplicate is not None:
            return {
                "brainId": self.brain_id,
                "duplicate": True,
                "source": duplicate,
                "metrics": self.metrics(),
            }

        before_checksum = self.parameter_checksum()
        before_concepts = len(self.memory.concepts)
        before_ideas = len(self.memory.ideas)
        before_events = int(self.router.synapses.plasticity_events.item())
        chunks = self._experience_chunks(extracted)
        losses: List[float] = []
        media_result: Dict[str, Any] = {
            "trained": False,
            "loss": 0.0,
            "warnings": [],
        }
        if policy != "archive":
            total = max(1, len(chunks))
            for index, chunk in enumerate(chunks):
                learned = self.learn_experience(
                    chunk,
                    kind="knowledge",
                    source="document",
                    source_label=source_name,
                    steps=2 if policy == "pretrain" else 1,
                    importance=0.65,
                )
                losses.append(float(learned["training"]["loss"]))
                if progress is not None:
                    progress(
                        (index + 1) / float(total),
                        "Encoding %s" % source_name,
                    )
            if resolved_kind in {"image", "audio", "video"}:
                if not path:
                    media_result["warnings"].append(
                        "Binary modality ingestion requires a local file path."
                    )
                else:
                    try:
                        media_result = self._train_media(
                            str(Path(path).resolve()),
                            resolved_kind,
                            source_name,
                            steps=3 if policy == "pretrain" else 2,
                            progress=progress,
                        )
                    except (RuntimeError, ValueError, OSError) as error:
                        media_result = {
                            "trained": False,
                            "loss": 0.0,
                            "warnings": [str(error)],
                        }
        source_record: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "name": source_name,
            "kind": resolved_kind,
            "bytes": len(raw_bytes),
            "content_hash": content_hash,
            "policy": policy,
            "imported_at": _iso_now(),
            "learned_ideas": len(self.memory.ideas) - before_ideas,
            "learned_concepts": len(self.memory.concepts) - before_concepts,
            "plasticity_events": int(
                self.router.synapses.plasticity_events.item()
            )
            - before_events,
            "raw_text_retained": False,
            "modality_trained": bool(media_result["trained"]),
            "warnings": list(media_result["warnings"]),
        }
        if (
            extracted
            and self.config.memory_recipe == "total-recall"
            and self.config.retain_source_text
        ):
            source_record["raw_text"] = extracted
            source_record["raw_text_retained"] = True
        self.training_sources.append(source_record)
        if policy == "consolidate":
            self.consolidate(max(1, min(8, len(chunks))), progress=progress)
        else:
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
                "parameterChecksumBefore": before_checksum,
                "parameterChecksumAfter": after_checksum,
                "rawTextRetained": source_record["raw_text_retained"],
                "modalityTrained": source_record["modality_trained"],
                "warnings": source_record["warnings"],
            },
        )
        return {
            "brainId": self.brain_id,
            "duplicate": False,
            "source": source_record,
            "meanLoss": sum(losses) / len(losses) if losses else 0.0,
            "modalityLoss": float(media_result["loss"]),
            "warnings": list(media_result["warnings"]),
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

    @staticmethod
    def _wav_bytes(waveform: torch.Tensor, sample_rate: int = 16000) -> bytes:
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
        for concept_id in concept_ids or []:
            vector = self.memory.concept_vectors.get(str(concept_id))
            if vector is not None:
                vectors.append(vector)
        if prompt:
            vectors.append(self.memory.vector_for_text(prompt))
        if not vectors:
            vectors.append(self.memory.space.symbol("unprompted-imagination"))
        cue = self.memory.space.bundle(vectors)
        if self.config.vector_symbolic_memory:
            recalled, _ = self.memory.recall_vector(
                cue, limit=min(self.config.working_memory_slots, 16)
            )
        else:
            recalled = cue
        return self.idea_adapter(self._idea_model_vector(recalled))

    def generate_modality(
        self,
        modality: str,
        prompt: str = "",
        concept_ids: Optional[Sequence[str]] = None,
        input_path: str = "",
        settings: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
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
            else (
                "locally-trained"
                if modality_steps > 0
                else (
                    "compatible-starter"
                    if self.config.origin_kind == "starter"
                    else "random"
                )
            )
        )
        randomly_initialized = initialization == "random"
        if modality == "vision":
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
            tensor = (tensor / 127.5 - 1.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.modalities.vision(tensor)[0].cpu().tolist()
            result = {
                "brainId": self.brain_id,
                "modality": "vision",
                "embedding": embedding,
                "inputPath": str(Path(input_path).resolve()),
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
            }
        else:
            output = self.modalities.generate(modality, idea, seed=seed)
            artifact_dir = self.engine_path / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_id = uuid.uuid4().hex
            if modality == "image":
                artifact = artifact_dir / (artifact_id + ".png")
                artifact_bytes = self._png_bytes(output)
                artifact.write_bytes(artifact_bytes)
                mime_type = "image/png"
            elif modality == "audio":
                artifact = artifact_dir / (artifact_id + ".wav")
                artifact_bytes = self._wav_bytes(
                    output,
                    sample_rate=int(
                        _finite_number((settings or {}).get("sampleRate"), 16000)
                    ),
                )
                artifact.write_bytes(artifact_bytes)
                mime_type = "audio/wav"
            elif modality == "video":
                artifact = artifact_dir / (artifact_id + ".png")
                artifact_bytes = self._apng_bytes(
                    output,
                    fps=int(_finite_number((settings or {}).get("fps"), 8)),
                )
                artifact.write_bytes(artifact_bytes)
                mime_type = "image/apng"
            else:
                raise ValueError("modality must be image, audio, video, or vision")
            result = {
                "brainId": self.brain_id,
                "modality": modality,
                "path": str(artifact),
                "mimeType": mime_type,
                "shape": list(output.shape),
                "seed": int(seed),
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
                "qualityNote": (
                    "Generated by a tiny research baseline; output quality depends "
                    "on the disclosed pack and local modality training."
                ),
            }
            if len(artifact_bytes) <= 8 * 1024 * 1024:
                result["dataUrl"] = (
                    "data:%s;base64,%s"
                    % (
                        mime_type,
                        base64.b64encode(artifact_bytes).decode("ascii"),
                    )
                )
        self.events.append(
            "modality-generation",
            {
                "modality": modality,
                "seed": int(seed),
                "outputPath": result.get("path"),
                "randomlyInitialized": randomly_initialized,
                "initialization": initialization,
                "trainingSteps": modality_steps,
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
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
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

    def snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        self.counters["snapshots"] += 1
        self.save()
        clean_label = SAFE_NAME.sub("-", label.strip()).strip(".-")[:48] or "snapshot"
        snapshot_id = "%s-%s" % (clean_label, uuid.uuid4().hex[:12])
        destination = self.engine_path / "snapshots" / snapshot_id
        snapshot_files(self.engine_path, destination)
        checksum = hashlib.sha256(
            (destination / "core.safetensors").read_bytes()
            + (destination / "plasticity.safetensors").read_bytes()
        ).hexdigest()
        result = {
            "id": snapshot_id,
            "brainId": self.brain_id,
            "label": label,
            "createdAt": _iso_now(),
            "path": str(destination),
            "checksum": checksum,
            "metrics": self.metrics(),
        }
        atomic_write_json(destination / "snapshot.json", result)
        self.events.append("snapshot", result)
        return result

    def metrics(self) -> Dict[str, Any]:
        parameters = sum(
            parameter.numel()
            for module in self._trainable_modules()
            for parameter in module.parameters()
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
            + int(self.router.synapses.weights.ne(0).sum().item()),
            "activeSynapses": int(self.router.synapses.weights.ne(0).sum().item()),
            "plasticityEvents": int(
                self.router.synapses.plasticity_events.item()
            ),
            "messages": len(self.messages),
            "trainingSources": len(self.training_sources),
            "replayExamples": len(self.replay),
            "workingMemoryVectors": len(self.working_memory),
            "experts": self.decoder.expert_count,
            "memoryCapacities": {
                "concepts": self.memory.max_concepts,
                "ideas": self.memory.max_ideas,
                "synapses": self.memory.max_relations,
                "expansions": self.memory.capacity_expansions,
            },
            "modalityTraining": dict(self.modality_training),
            "trainableParameters": parameters,
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
        for attribute, key in (
            ("ternary_weights", "ternaryWeights"),
            ("spiking_dynamics", "spikingDynamics"),
            ("stdp_plasticity", "stdpPlasticity"),
            ("liquid_dynamics", "liquidDynamics"),
            ("vector_symbolic_memory", "vectorSymbolicMemory"),
            ("online_learning", "onlineLearning"),
            ("consolidation_enabled", "consolidation"),
            ("metaplasticity", "metaplasticity"),
            ("retain_source_text", "retainSourceText"),
            ("learn_from_own_messages", "learnFromOwnMessages"),
        ):
            assign(attribute, key, bool)
        for attribute, key, transform in (
            ("noise", "noise", float),
            ("firing_threshold", "firingThreshold", float),
            ("membrane_leak", "membraneLeak", float),
            ("working_memory_slots", "workingMemorySlots", int),
            (
                "short_term_half_life_minutes",
                "shortTermHalfLifeMinutes",
                float,
            ),
            ("long_term_threshold", "longTermThreshold", float),
            ("forgetting_rate", "forgettingRate", float),
            ("consolidation_rate", "consolidationRate", float),
            ("novelty_drive", "noveltyDrive", float),
            ("coherence_drive", "coherenceDrive", float),
            ("curiosity_drive", "curiosityDrive", float),
            ("parallel_thoughts", "parallelThoughts", int),
            ("max_concepts", "maxConcepts", int),
            ("max_synapses", "maxSynapses", int),
        ):
            assign(attribute, key, transform)
        if "learningRate" in raw:
            neural_rate = max(
                1e-5, min(0.02, float(raw["learningRate"]) * 0.02)
            )
            if self.config.learning_rate != neural_rate:
                self.config.learning_rate = neural_rate
                changed.append("learning_rate")
        assign("growth_policy", "growthPolicy", str)
        assign("memory_recipe", "memoryRecipe", str)
        assign("memory_injection", "memoryInjection", str)
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
        self.population_controls_from_config()
        self.memory.max_concepts = self.config.max_concepts
        self.memory.max_relations = self.config.max_synapses
        self._optimizer = self._new_optimizer()
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
                    module.ternary = self.config.ternary_weights

    def merge_overlay(self, source: "AdaptiveBrain") -> Dict[str, Any]:
        """Merge inspectable ideas/replay, never whole-model weight averages."""

        if source.brain_id == self.brain_id:
            raise ValueError("cannot merge a brain overlay into itself")
        existing_fingerprints = {
            idea["fingerprint"] for idea in self.memory.ideas
        }
        added_concepts = 0
        added_ideas = 0
        added_relations = 0
        added_replay = 0
        for concept_id, concept in source.memory.concepts.items():
            if concept_id in self.memory.concepts:
                continue
            if len(self.memory.concepts) >= self.memory.max_concepts:
                break
            self.memory.concepts[concept_id] = dict(concept)
            vector = source.memory.concept_vectors.get(concept_id)
            if vector is not None:
                self.memory.concept_vectors[concept_id] = vector.detach().cpu().clone()
            added_concepts += 1
        for idea in source.memory.ideas:
            if idea["fingerprint"] in existing_fingerprints:
                continue
            if len(self.memory.ideas) >= self.memory.max_ideas:
                break
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
            added_ideas += 1
        for relation_id, relation in source.memory.relations.items():
            if relation_id in self.memory.relations:
                continue
            if len(self.memory.relations) >= self.memory.max_relations:
                break
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
            "metrics": self.metrics(),
        }
        self.events.append("overlay-merged", result)
        return result
