"""Adaptive OmniCortex brain lifecycle and persistence."""

import array
import base64
import binascii
import copy
import hashlib
import io
import itertools
import json
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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import torch
from torch import nn
from torch.nn import functional as F

from .config import OmniConfig
from .datasets import DatasetCoverage, dataset_format, iter_dataset_records
from .liquid import LiquidController
from .modalities import ModalityHub
from .model import (
    ACTION_KINDS,
    TERNARY_PROJECTION_TYPES,
    BitLinear,
    OmniDecoder,
)
from .persistence import (
    EventLog,
    atomic_save_tensors,
    atomic_write_json,
    copy_substrate_snapshot,
    load_tensors,
    read_json,
    snapshot_files,
    tensor_checksum,
)
from .spiking import AssociativeSpikingRouter
from .starter import (
    STARTER_ACTION_EXAMPLES,
    STARTER_CORPUS,
    starter_manifest,
)
from .ternary_packing import (
    collect_module_ternary_tensors,
    export_module_ternary_shards,
    verify_ternary_shards,
)
from .tokenizer import ByteTokenizer
from .vsa import NeuralSubstrate, SubstrateResourcePause


ENGINE_SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
ACTION_PROPOSAL_CONFIDENCE = 0.62
STARTER_ACTION_TARGET_CONFIDENCE = 0.70
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


class AdaptiveBrain:
    """One persistent, mutable model identity.

    ``storage_path`` is the desktop brain directory.  All Python-owned files
    live below ``storage_path/engine`` so the UI's ``brain.json`` remains
    authoritative and untouched.
    """

    def __init__(self, brain_id: str, storage_path: Path, config: OmniConfig):
        # Stable OmniCortex always uses ternary forward synapses and
        # resource-governed structural growth.  Floating master weights remain
        # available only to the learning algorithm.
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
                if isinstance(module, TERNARY_PROJECTION_TYPES):
                    module.ternary = True
        self.memory = NeuralSubstrate(
            config.vsa_dim,
            seed=config.seed,
            growth_guard=self._allow_substrate_growth,
        )
        self.liquid_state = torch.zeros(1, config.idea_dim, device=self.device)
        self.working_memory: List[torch.Tensor] = []
        self.workspace_items: List[Dict[str, Any]] = []
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
        self.starter_training_manifest: Optional[Dict[str, Any]] = None
        self.packed_ternary_manifest: Optional[Dict[str, Any]] = None
        self._starter_action_language_cache: Optional[torch.Tensor] = None
        self._starter_action_internal_cache: Optional[torch.Tensor] = None
        self._starter_action_target_cache: Optional[torch.Tensor] = None
        self._bundled_origin_verified = False
        self.novelty_streak = 0
        self.growth_pause: Optional[Dict[str, Any]] = None
        self.last_activity_decay = time.time()
        self.last_idle_cycle_at = 0.0
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

        return {
            "decoder": self.decoder,
            "memory_bridge": self.memory_bridge,
            "idea_adapter": self.idea_adapter,
            "liquid": self.liquid,
        }

    def _slow_parameter_checksum(self) -> str:
        return tensor_checksum(
            [
                parameter
                for module in self._slow_transaction_modules().values()
                for parameter in module.parameters()
            ]
        )

    def _snapshot_slow_transaction_state(self) -> Dict[str, Any]:
        """Capture a complete rollback point for one chat slow mutation.

        The snapshot is taken only after the current turn has entered the fast
        substrate and working memory. Restoring it therefore rolls back slow
        gradient/growth work without erasing the valid fast experience.
        """

        cuda_rng_state = None
        if self.device_backend == "cuda" and torch.cuda.is_available():
            cuda_rng_state = [
                value.clone() for value in torch.cuda.get_rng_state_all()
            ]
        return {
            "modules": {
                name: {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                for name, module in self._slow_transaction_modules().items()
            },
            "expert_count": int(self.decoder.expert_count),
            "optimizer": _clone_state_to_cpu(self._optimizer.state_dict()),
            "stability": self._stability_copy(),
            "counters": dict(self.counters),
            "novelty_streak": int(self.novelty_streak),
            "growth_pause": copy.deepcopy(self.growth_pause),
            "cpu_rng_state": torch.random.get_rng_state().clone(),
            "cuda_rng_state": cuda_rng_state,
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
            self.decoder.expert_prototypes = nn.ParameterList(
                list(self.decoder.expert_prototypes)[:expected_experts]
            )

        module_states = snapshot["modules"]
        for name, module in self._slow_transaction_modules().items():
            module.load_state_dict(module_states[name], strict=True)

        # Growth replaces the main optimizer. Rebuild it against the restored
        # parameter objects before loading the exact pre-transaction moments,
        # groups, learning rates, and step counters.
        self._optimizer = self._new_optimizer()
        self._optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        self._restore_stability(snapshot["stability"])
        self.counters.clear()
        self.counters.update(snapshot["counters"])
        self.novelty_streak = int(snapshot["novelty_streak"])
        self.growth_pause = copy.deepcopy(snapshot["growth_pause"])
        torch.random.set_rng_state(snapshot["cpu_rng_state"])
        cuda_rng_state = snapshot.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

        restored_checksum = self._slow_parameter_checksum()
        if restored_checksum != snapshot["checksum"]:
            raise RuntimeError(
                "slow-learning rollback checksum mismatch: expected %s, got %s"
                % (snapshot["checksum"], restored_checksum)
            )

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
        if config.origin_kind == "starter":
            brain.starter_training_manifest = brain._train_bundled_starter()
        brain.save()
        packed = brain.export_packed_ternary()
        origin = brain.engine_path / "origin"
        snapshot_files(brain.engine_path, origin)
        shutil.copytree(
            brain.engine_path / "packed-ternary",
            origin / "packed-ternary",
        )
        if config.origin_kind == "starter":
            brain._write_bundled_origin_provenance()
            brain._bundled_origin_verified = (
                brain._verify_bundled_origin_provenance()
            )
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
                "starterManifest": brain.starter_training_manifest,
                "packedTernary": packed["summary"],
            },
        )
        return brain

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _bundled_origin_provenance_payload(self) -> Dict[str, Any]:
        origin = self.engine_path / "origin"
        metadata = read_json(origin / "brain.json")
        substrate = metadata.get("substrate", {})
        persistence = (
            substrate.get("persistence", {})
            if isinstance(substrate, Mapping)
            else {}
        )
        manifest = metadata.get("starter_training_manifest", {})
        return {
            "format": "omni-bundled-origin-provenance-1",
            # A fork/duplicate keeps the immutable origin but receives a new
            # live identity. Provenance therefore binds to the identity stored
            # inside the origin snapshot, not the mutable current brain id.
            "originBrainId": metadata.get("brain_id"),
            "starterId": manifest.get("id"),
            "starterManifestSha256": manifest.get("sha256"),
            "originParameterChecksum": manifest.get(
                "trainedParameterChecksum"
            ),
            "coreSha256": self._file_sha256(
                origin / "core.safetensors"
            ),
            "plasticitySha256": self._file_sha256(
                origin / "plasticity.safetensors"
            ),
            "brainMetadataSha256": self._file_sha256(
                origin / "brain.json"
            ),
            "substrateContentSha256": persistence.get("contentSha256"),
            "packedManifestSha256": self._file_sha256(
                origin / "packed-ternary" / "manifest.json"
            ),
        }

    def _write_bundled_origin_provenance(self) -> None:
        payload = self._bundled_origin_provenance_payload()
        content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        atomic_write_json(
            self.engine_path / "origin" / "provenance.json",
            {
                **payload,
                "contentSha256": hashlib.sha256(content).hexdigest(),
            },
        )

    def _verify_bundled_origin_provenance(self) -> bool:
        provenance_path = self.engine_path / "origin" / "provenance.json"
        if self.config.origin_kind != "starter" or not provenance_path.is_file():
            return False
        try:
            recorded = read_json(provenance_path)
            content_sha = str(recorded.pop("contentSha256", ""))
            expected_content = hashlib.sha256(
                json.dumps(
                    recorded,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if content_sha != expected_content:
                return False
            expected = starter_manifest()
            actual = self._bundled_origin_provenance_payload()
            return (
                recorded == actual
                and actual["starterId"] == expected["id"]
                and actual["starterManifestSha256"] == expected["sha256"]
                and isinstance(actual["originParameterChecksum"], str)
                and len(actual["originParameterChecksum"]) == 64
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
            for text, kind in STARTER_ACTION_EXAMPLES:
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
                    STARTER_ACTION_TARGET_CONFIDENCE,
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
        try:
            language_batch, assembly_batch, target_batch = (
                self._starter_action_features()
            )
            if strict and self._starter_action_language_cache is None:
                self._starter_action_language_cache = (
                    language_batch.detach().cpu().clone()
                )
                self._starter_action_internal_cache = (
                    assembly_batch.detach().cpu().clone()
                )
                self._starter_action_target_cache = (
                    target_batch.detach().cpu().clone()
                )
            optimizer = torch.optim.AdamW(
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
                        "non-finite bundled starter action loss"
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
            if not calibrated:
                raise RuntimeError(
                    "bundled starter action policy did not retain its neural "
                    "confidence margin"
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
                "examples": len(STARTER_ACTION_EXAMPLES),
                "trainingVectors": len(STARTER_ACTION_EXAMPLES) * 2,
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
                "featureVectors": len(STARTER_ACTION_EXAMPLES),
                "featureBatching": "right-padded-mask-correct",
                "applied": False,
                "initialLoss": initial_loss,
                "finalLoss": final_loss,
                **readings,
                "proposalConfidenceThreshold": ACTION_PROPOSAL_CONFIDENCE,
                "requiredTargetConfidence": STARTER_ACTION_TARGET_CONFIDENCE,
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
        accuracy = 0.5 * (
            readings["languageAccuracy"] + readings["internalAccuracy"]
        )
        return {
            "examples": len(STARTER_ACTION_EXAMPLES),
            "trainingVectors": int(target_batch.numel()) * 2,
            "neuralChannels": ["language-decoder", "internal-assembly"],
            "languageChatFraming": ["bos", "human", "text", "brain"],
            "steps": completed,
            "attemptedSteps": completed,
            "featurePasses": 1,
            "featureVectors": len(STARTER_ACTION_EXAMPLES),
            "featureBatching": "right-padded-mask-correct",
            "applied": completed > 0,
            "initialLoss": initial_loss,
            "finalLoss": final_loss,
            "accuracy": accuracy,
            **readings,
            "proposalConfidenceThreshold": ACTION_PROPOSAL_CONFIDENCE,
            "requiredTargetConfidence": STARTER_ACTION_TARGET_CONFIDENCE,
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

    def _train_starter_action_policy(self) -> Dict[str, Any]:
        """Imitate typed action trajectories without a reward/preference model."""

        return self._calibrate_starter_action_policy(
            max_steps=96,
            minimum_steps=16,
        )

    def _can_retain_bundled_action_policy(self) -> bool:
        expected = starter_manifest()
        return bool(
            self.config.origin_kind == "starter"
            and self._bundled_origin_verified
            and self.starter_training_manifest is not None
            and self.starter_training_manifest.get("id") == expected["id"]
            and self.starter_training_manifest.get("sha256")
            == expected["sha256"]
            and self._starter_action_language_cache is not None
            and self._starter_action_internal_cache is not None
            and self._starter_action_target_cache is not None
        )

    def _retain_starter_action_policy(
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
        over all bundled trajectories using the *current* decoder, memory
        bridge, global workspace, expert router, and action inputs. Origin
        caches authenticate eligibility but never stand in for current neural
        features.
        """

        if not self._can_retain_bundled_action_policy():
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
        canonical_language, canonical_internal, canonical_targets = (
            self._starter_action_features()
        )
        post_language = post_language_feature.detach().to(self.device)
        post_internal = post_internal_feature.detach().to(self.device)
        optimizer = torch.optim.AdamW(
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
        self.counters["action_retention_checks"] += 1
        if result["calibrated"]:
            self.counters["action_retention_replays"] += int(result["steps"])
        else:
            self.counters["action_retention_failures"] += 1
        return result

    def _train_starter_modalities(self) -> Dict[str, Any]:
        """Train every enabled baseline on deterministic synthetic perceptions."""

        height = self.config.image_size
        width = self.config.image_size
        horizontal = torch.linspace(
            -1.0, 1.0, width, device=self.device
        ).reshape(1, 1, 1, width)
        vertical = torch.linspace(
            -1.0, 1.0, height, device=self.device
        ).reshape(1, 1, height, 1)
        image = torch.cat(
            (
                horizontal.expand(1, 1, height, width),
                vertical.expand(1, 1, height, width),
                0.5
                * (
                    horizontal.expand(1, 1, height, width)
                    + vertical.expand(1, 1, height, width)
                ),
            ),
            dim=1,
        )
        audio_time = torch.linspace(
            0.0, 1.0, self.config.audio_samples, device=self.device
        )
        audio = (
            0.55 * torch.sin(audio_time * (2.0 * math.pi * 3.0))
        ).reshape(1, 1, -1)
        frames = [
            torch.roll(image, shifts=index, dims=-1)
            for index in range(self.config.video_frames)
        ]
        video = torch.stack(frames, dim=2)
        idea = self._media_idea("bundled starter multimodal experience")

        parameters: List[nn.Parameter] = []
        if self.config.vision_enabled:
            parameters.extend(self.modalities.vision.parameters())
        if self.config.image_enabled:
            parameters.extend(self.modalities.image.parameters())
        if self.config.audio_enabled:
            parameters.extend(self.modalities.audio.parameters())
        if self.config.video_enabled:
            parameters.extend(self.modalities.video.parameters())
        if not parameters:
            return {"steps": 0, "loss": 0.0, "modalities": []}

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )
        optimizer.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        trained: List[str] = []
        if self.config.image_enabled:
            losses.append(self.modalities.image(image, idea)["loss"])
            trained.append("image")
        if self.config.vision_enabled:
            embedding = self.modalities.vision(image)
            losses.append(
                0.2
                * (
                    1.0
                    - F.cosine_similarity(
                        embedding,
                        F.normalize(idea, dim=-1),
                    ).mean()
                )
            )
            trained.append("vision")
        if self.config.audio_enabled:
            audio_output = self.modalities.audio(audio, idea)
            losses.append(
                audio_output["loss"]
                + 0.2
                * (
                    1.0
                    - F.cosine_similarity(
                        audio_output["embedding"],
                        F.normalize(idea, dim=-1),
                    ).mean()
                )
            )
            trained.append("audio")
        if self.config.video_enabled:
            video_output = self.modalities.video(video, idea)
            losses.append(
                video_output["loss"]
                + 0.2
                * (
                    1.0
                    - F.cosine_similarity(
                        video_output["embedding"],
                        F.normalize(idea, dim=-1),
                    ).mean()
                )
            )
            trained.append("video")
        loss = torch.stack(losses).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite bundled starter modality loss")
        loss.backward()
        self._accumulate_slow_importance(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
        optimizer.step()
        self._commit_slow_anchors(rate=1.0)
        self.counters["training_steps"] += 1
        for modality in trained:
            self.modality_training[modality] += 1
        return {
            "steps": 1,
            "loss": float(loss.detach().item()),
            "modalities": trained,
            "syntheticFixture": True,
        }

    def _train_bundled_starter(self) -> Dict[str, Any]:
        """Materialize the project-authored trained origin before snapshotting."""

        initial_checksum = self.parameter_checksum()
        losses: List[float] = []
        for text in STARTER_CORPUS:
            learned = self.learn_experience(
                text,
                kind="starter-training",
                source="bundled-starter",
                source_label="omni-starter-bundled-1",
                steps=1,
                importance=0.72,
            )
            losses.append(float(learned["training"]["loss"]))
        action_training = self._train_starter_action_policy()
        modality_training = self._train_starter_modalities()
        manifest = {
            **starter_manifest(),
            "trainedAt": _iso_now(),
            "initialParameterChecksum": initial_checksum,
            "trainedParameterChecksum": self.parameter_checksum(),
            "corpusPassagesVisited": len(STARTER_CORPUS),
            "meanCorpusLoss": sum(losses) / max(1, len(losses)),
            "corpusLossCurve": losses,
            "actionTraining": action_training,
            "modalityTraining": modality_training,
            "hiddenBehavioralPrompt": False,
        }
        self.events.append(
            "bundled-starter-trained",
            {
                "manifestId": manifest["id"],
                "manifestSha256": manifest["sha256"],
                "corpusPassagesVisited": len(STARTER_CORPUS),
                "actionTraining": action_training,
                "modalityTraining": modality_training,
                "parameterChecksumBefore": initial_checksum,
                "parameterChecksumAfter": manifest["trainedParameterChecksum"],
            },
        )
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
                constructed[0].events.close()
            raise

    @classmethod
    def _load_impl(
        cls,
        storage_path: Path,
        expected_brain_id: Optional[str] = None,
        constructed: Optional[List["AdaptiveBrain"]] = None,
    ) -> "AdaptiveBrain":
        engine_path = Path(storage_path).resolve() / "engine"
        recovered_candidates = cls._recover_interrupted_candidates(engine_path)
        metadata = read_json(engine_path / "brain.json")
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
        if isinstance(substrate_metadata, dict) and isinstance(
            substrate_metadata.get("persistence"), dict
        ):
            brain.memory = NeuralSubstrate.load_sharded(
                engine_path / "substrate",
                substrate_metadata,
                growth_guard=brain._allow_substrate_growth,
            )
        else:
            # Backward-safe internal stable-v1 loading. The public beta format
            # remains rejected above; early stable checkpoints stored these
            # vectors in plasticity.safetensors.
            brain.memory = NeuralSubstrate.from_state(
                substrate_metadata, plastic, prefix="substrate."
            )
            brain.memory.growth_guard = brain._allow_substrate_growth
        replay = plastic.get("state.replay")
        if replay is not None:
            brain.replay = [row.detach().cpu() for row in replay]
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
        stored_starter_manifest = metadata.get("starter_training_manifest")
        brain.starter_training_manifest = (
            dict(stored_starter_manifest)
            if isinstance(stored_starter_manifest, Mapping)
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
        brain.last_activity_decay = float(
            metadata.get("last_activity_decay", time.time())
        )
        brain.last_idle_cycle_at = float(metadata.get("last_idle_cycle_at", 0.0))
        brain._bundled_origin_verified = (
            brain._verify_bundled_origin_provenance()
        )
        if brain._bundled_origin_verified:
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
                brain._bundled_origin_verified = False
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
        brain._optimizer = brain._new_optimizer()
        packed_path = engine_path / "packed-ternary"
        if (packed_path / "manifest.json").is_file():
            synapse_ids, dynamic_values = brain._dynamic_synapse_export()
            dynamic_name = "substrate.dynamic_synapses.weights"
            expected_synapse_order_hash = hashlib.sha256(
                "\0".join(synapse_ids).encode("utf-8")
            ).hexdigest()
            expected_specs = collect_module_ternary_tensors(
                brain._ternary_export_roots(),
                dynamic_synapses={
                    dynamic_name: dynamic_values,
                },
            )
            expected_names = [spec.name for spec in expected_specs]
            verified_packed = verify_ternary_shards(packed_path)
            packed_metadata = verified_packed.manifest.get("metadata")
            if not isinstance(packed_metadata, Mapping):
                raise ValueError("packed ternary metadata is invalid")
            packed_parameter_checksum = str(
                packed_metadata.get("parameterChecksum", "")
            )
            packed_dynamic_values = verified_packed.tensors.get(dynamic_name)
            packed_dynamic_count = packed_metadata.get("dynamicSynapseCount")
            packed_dynamic_order_hash = packed_metadata.get(
                "dynamicSynapseOrderSha256"
            )
            dynamic_pack_is_stale = (
                isinstance(packed_dynamic_count, bool)
                or not isinstance(packed_dynamic_count, int)
                or packed_dynamic_count != len(synapse_ids)
                or not isinstance(packed_dynamic_order_hash, str)
                or packed_dynamic_order_hash != expected_synapse_order_hash
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
            ):
                brain.export_packed_ternary()
            else:
                verify_ternary_shards(
                    packed_path,
                    expected_names=expected_names,
                )
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
        tensors["state.liquid"] = self.liquid_state.detach()
        if self.working_memory:
            tensors["state.working_memory"] = torch.stack(self.working_memory)
        if self.replay:
            tensors["state.replay"] = torch.stack(self.replay)
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

    def _metadata(self) -> Dict[str, Any]:
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
            "last_activity_decay": self.last_activity_decay,
            "last_idle_cycle_at": self.last_idle_cycle_at,
            "messages": self.messages[-10000:],
            "traces": self.traces[-1000:],
            "training_sources": self.training_sources,
            "workspace_items": self.workspace_items,
            "recent_token_context": self.recent_token_context,
            "current_context": self.current_context,
            "counters": self.counters,
            "modality_training": self.modality_training,
            "installed_modality_packs": self.installed_modality_packs,
            "starter_training_manifest": self.starter_training_manifest,
            "packed_ternary_manifest": self.packed_ternary_manifest,
            "substrate": self.memory.metadata(include_records=False),
            "runtime_card": self.runtime_card(),
            "files": {
                "core": "core.safetensors",
                "plasticity": "plasticity.safetensors",
                "substrate": "substrate/manifest.json",
                "origin": "origin/",
                "snapshots": "snapshots/",
                "artifacts": "artifacts/",
                "events": "events.sqlite3",
            },
        }

    def save(self) -> None:
        self.updated_at = _iso_now()
        self.engine_path.mkdir(parents=True, exist_ok=True)
        # The bounded, content-addressed substrate generation is complete
        # before metadata can point at it. Unchanged shard blobs are reused.
        self.memory.save_sharded(self.engine_path / "substrate")
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

    def _ternary_export_roots(self) -> Dict[str, nn.Module]:
        return {
            "decoder": self.decoder,
            "memory_bridge": self.memory_bridge,
            "idea_adapter": self.idea_adapter,
            "router": self.router,
            "liquid": self.liquid,
            "modalities": self.modalities,
        }

    def _dynamic_synapse_export(self) -> Tuple[List[str], torch.Tensor]:
        synapse_ids = sorted(self.memory.synapses)
        values = torch.tensor(
            [
                int(self.memory.synapses[synapse_id]["effective_weight"])
                for synapse_id in synapse_ids
            ],
            dtype=torch.int8,
        )
        return synapse_ids, values

    def export_packed_ternary(
        self, destination: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Materialize and verify exact 2-bit inference shards.

        Floating master weights remain in safe tensors for continued learning.
        This sidecar contains only the effective ``{-1, 0, +1}`` projections
        and dynamically grown substrate synapses used by inference.
        """

        roots = self._ternary_export_roots()
        dense_types = (
            nn.Linear,
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

        synapse_ids, dynamic_values = self._dynamic_synapse_export()
        dynamic = {
            "substrate.dynamic_synapses.weights": dynamic_values,
        }
        specs = collect_module_ternary_tensors(
            roots,
            dynamic_synapses=dynamic,
        )
        expected_names = [spec.name for spec in specs]
        synapse_order_hash = hashlib.sha256(
            "\0".join(synapse_ids).encode("utf-8")
        ).hexdigest()
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
                    "dynamicSynapseCount": len(synapse_ids),
                    "dynamicSynapseOrderSha256": synapse_order_hash,
                },
            )
            verify_ternary_shards(
                temporary,
                expected_names=expected_names,
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
            "dynamicSynapseCount": len(synapse_ids),
            "dynamicSynapseOrderSha256": synapse_order_hash,
            "parameterChecksum": self.parameter_checksum(),
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
        for root_name, root in (
            ("decoder", self.decoder),
            ("memory_bridge", self.memory_bridge),
            ("idea_adapter", self.idea_adapter),
            ("router", self.router),
            ("liquid", self.liquid),
            ("modalities", self.modalities),
        ):
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
                levels = {
                    int(value)
                    for value in torch.unique(module.effective_weight()).tolist()
                }
                observed_levels.update(levels)
                if not levels.issubset({-1, 0, 1}):
                    violations.append(name)
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
            "masterWeightPrecision": "floating-learning-state",
            "forwardPrecision": "exact ternary {-1,0,+1}",
        }

    def _organic_state(self) -> Dict[str, float]:
        neurons = list(self.memory.neurons.values())
        uncertainty = (
            sum(float(item.get("uncertainty", 0.5)) for item in neurons)
            / float(len(neurons))
            if neurons
            else 0.5
        )
        active = (
            sum(
                1
                for item in neurons
                if float(item.get("activation", 0.0)) >= 0.1
            )
            / float(len(neurons))
            if neurons
            else 0.0
        )
        recent_losses = [
            float(trace.get("train_loss", 0.0))
            for trace in self.traces[-2:]
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
        recent = self.traces[-1] if self.traces else {}
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
            "tension": tension,
            "curiosity": curiosity,
        }

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
            "starter_training_manifest": self.starter_training_manifest,
            "packed_ternary_manifest": self.packed_ternary_manifest,
            "hidden_behavioral_prompt": False,
            "reward_model": False,
            "rlhf": False,
            "memory_injection": self.config.memory_injection,
            "textual_long_term_memory_injected": False,
            "tokenizer_boundary": "UTF-8 bytes",
            "weight_forward": "scaled ternary {-1,0,+1}",
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
                "trainBatchSize": self.config.train_batch_size,
                "gradientAccumulation": self.config.gradient_accumulation,
                "gradientCheckpointing": self.config.gradient_checkpointing,
                "replayOffload": "cpu-with-durable-safetensors",
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
            "intrinsic_dynamics": self._organic_state(),
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
        for index, existing in enumerate(self.working_memory):
            similarity = float(
                F.cosine_similarity(
                    existing.float().reshape(1, -1),
                    vector.float().reshape(1, -1),
                ).item()
            )
            if similarity < 0.985:
                continue
            self.working_memory[index] = (
                existing.float() * 0.72 + vector.float() * 0.28
            )
            item = self.workspace_items[index]
            item["salience"] = min(
                1.0, float(item.get("salience", 0.5)) * 0.8 + salience * 0.2
            )
            item["rehearsals"] = int(item.get("rehearsals", 1)) + 1
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
                "id": uuid.uuid4().hex,
                "assemblyId": assembly_id,
                "source": source,
                "salience": salience,
                "rehearsals": 1,
                "enteredAt": timestamp,
                "lastActiveAt": timestamp,
            }
        )
        while len(self.working_memory) > self.config.working_memory_slots:
            eviction = min(
                range(len(self.workspace_items)),
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
            self.working_memory.pop(eviction)
            self.workspace_items.pop(eviction)
            self.counters["workspace_evictions"] += 1

    def _working_memory_vector(self) -> Optional[torch.Tensor]:
        if (
            self.config.memory_injection != "working-memory"
            or not self.working_memory
        ):
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
                "occupancy": len(self.working_memory),
                "items": items,
                "evictions": self.counters["workspace_evictions"],
                "rehearsals": self.counters["workspace_rehearsals"],
            },
            "liquidState": {
                "dimensions": int(self.liquid_state.numel()),
                "mean": float(self.liquid_state.detach().float().mean().item()),
                "norm": float(self.liquid_state.detach().float().norm().item()),
            },
            "hiddenBehavioralPrompt": False,
            "rawLongTermTextInjected": False,
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
        value = "%d:%d:%d:%d" % (
            len(self.memory.neurons),
            len(self.memory.assemblies),
            len(self.memory.synapses),
            int(self.memory.growth_events),
        )
        return hashlib.sha256(value.encode("ascii")).hexdigest()[:16]

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
        return {
            "id": str(record.get("id", record.get("neuron_id", ""))),
            "label": str(record.get("label", "")),
            "region": str(record.get("region", "semantic")),
            "activation": self._inspection_number(record.get("activation")),
            "importance": self._inspection_number(record.get("importance")),
            "uncertainty": self._inspection_number(
                record.get("uncertainty"), 0.5
            ),
            "exposures": max(0, int(record.get("exposures", 0))),
            "createdAt": self._inspection_timestamp(record.get("created_at")),
            "lastActivatedAt": self._inspection_timestamp(
                record.get("last_activated_at")
            ),
            "aliases": [
                str(value)
                for value in record.get("aliases", [])
                if isinstance(value, str)
            ],
        }

    def _inspect_assembly(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        assembly_id = str(record.get("id", ""))
        inspector_node = self.memory.neurons.get(assembly_id, {})
        return {
            "id": assembly_id,
            "label": str(
                inspector_node.get(
                    "label",
                    record.get("source_label", record.get("kind", "assembly")),
                )
            ),
            "region": "assembly",
            "neuronIds": [
                str(value) for value in record.get("neuron_ids", [])
            ],
            "childAssemblyIds": [
                str(value)
                for value in record.get("child_assembly_ids", [])
            ],
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
            ),
            "sourceLabel": str(record.get("source_label", "")) or None,
            # Inspection reveals whether exact text exists, never the retained
            # passage itself.
            "retainsSourceText": "source_text" in record,
        }

    def _inspect_synapse(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        effective = int(record.get("effective_weight", 0))
        effective = -1 if effective < 0 else (1 if effective > 0 else 0)
        return {
            "id": str(record.get("id", "")),
            "sourceId": str(record.get("source_id", "")),
            "targetId": str(record.get("target_id", "")),
            "kind": str(record.get("kind", "associates")),
            "effectiveWeight": effective,
            "latentWeight": self._inspection_number(
                record.get("latent_weight")
            ),
            "eligibility": self._inspection_number(
                record.get("eligibility")
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

    def query_substrate(
        self, query: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a read-only, cursor-paged multiresolution substrate view.

        A page-size bound protects the JSON-RPC channel, while ``nextCursor``
        makes the total addressable result unbounded. Cursors are invalidated
        by structural growth so a traversal never silently mixes revisions.
        """

        raw = dict(query or {})
        entity = str(raw.get("entity", "overview"))
        if entity not in {"overview", "neurons", "assemblies", "synapses"}:
            raise ValueError("invalid substrate entity")
        zoom = max(0.0, min(self._inspection_number(raw.get("zoom"), 0.0), 1.0))
        page_size_raw = raw.get("pageSize", 256)
        if isinstance(page_size_raw, bool):
            raise ValueError("invalid substrate page size")
        page_size = max(1, min(int(page_size_raw), 5000))
        region = str(raw.get("region", "")).strip()
        search = str(raw.get("search", "")).strip()
        if len(region) > 128 or len(search) > 512:
            raise ValueError("substrate filter is too long")
        revision = self._substrate_revision()
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "entity": entity,
                    "zoom": round(zoom, 6),
                    "region": region.casefold(),
                    "search": search.casefold(),
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
            "entity": entity,
            "zoom": zoom,
            "totals": totals,
            "matched": 0,
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
            page = clusters[offset : offset + page_size]
            end = offset + len(page)
            response["clusters"] = page
            response["matched"] = len(clusters)
            response["hasMore"] = end < len(clusters)
            if response["hasMore"]:
                response["nextCursor"] = self._encode_substrate_cursor(
                    end, revision, fingerprint
                )
            return response

        search_value = search.casefold()
        if entity == "neurons":
            records = [
                self._inspect_neuron(record)
                for record in self.memory.neurons.values()
            ]
            records = [
                record
                for record in records
                if (
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
            ]
        elif entity == "assemblies":
            records = [
                self._inspect_assembly(record)
                for record in self.memory.assemblies
            ]
            records = [
                record
                for record in records
                if (not region or region.casefold() == "assembly")
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
            ]
        else:
            region_by_id = {
                str(record.get("id", "")): str(
                    record.get("region", "semantic")
                )
                for record in self.memory.neurons.values()
            }
            records = [
                self._inspect_synapse(record)
                for record in self.memory.synapses.values()
            ]
            records = [
                record
                for record in records
                if (
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
            ]
        records.sort(key=lambda record: str(record["id"]))
        page = records[offset : offset + page_size]
        end = offset + len(page)
        response[entity] = page
        response["matched"] = len(records)
        response["hasMore"] = end < len(records)
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
            optimizer = self._optimizer
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
                max_length=self.config.max_seq_len,
                add_bos=True,
                add_eos=True,
            ):
                if ids.shape[1] < 2:
                    continue
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
                stability_loss = self._stability_penalty()
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
                self._accumulate_slow_importance()
                parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.grad_clip
                )
                optimizer.step()
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
            self._commit_slow_anchors(rate=0.08)
        return {
            "loss": sum(losses) / len(losses),
            "language_loss": sum(language_losses) / len(language_losses),
            "idea_loss": sum(idea_losses) / len(idea_losses),
            "workspace_loss": sum(workspace_losses) / len(workspace_losses),
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

        encoded: List[torch.Tensor] = []
        expanded_vectors: List[torch.Tensor] = []
        for text, vector in zip(texts, vsa_vectors):
            for window in self.tokenizer.window_tensors(
                text,
                self.device,
                max_length=self.config.max_seq_len,
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
        whole = self.decoder.encode_whole(ids)
        workspace_loss = F.mse_loss(
            F.normalize(whole, dim=-1),
            F.normalize(idea.detach(), dim=-1),
        )
        language = self.decoder(
            ids, memory_bias=reconstructed, labels=ids
        )["loss"]
        stability_loss = self._stability_penalty()
        loss = (
            language
            + 0.2 * idea_loss
            + 0.05 * temporal_loss
            + 0.1 * workspace_loss
            + stability_loss
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite batch training loss")
        return loss, {
            "loss": float(loss.detach().item()),
            "language_loss": float(language.detach().item()),
            "idea_loss": float(idea_loss.detach().item()),
            "workspace_loss": float(workspace_loss.detach().item()),
            "stability_loss": float(stability_loss.detach().item()),
        }

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
        self.novelty_streak = 0
        self.growth_pause = None
        self._optimizer = self._new_optimizer()
        self._sync_stability_state()
        return True

    def _maybe_expand_memory(self) -> bool:
        """Compatibility hook; the substrate now grows directly as it learns."""

        return False

    def _allow_substrate_growth(self, estimated_bytes: int) -> bool:
        """Apply host reserve watermarks without imposing neuron-count caps."""

        estimated_bytes = max(1, int(estimated_bytes))
        readings = self._resource_readings()
        disk_free = readings.get("diskFreeBytes")
        ram_free = readings.get("availableMemoryBytes")
        reason = ""
        if isinstance(disk_free, int) and disk_free < max(
            512 * 1024 * 1024, estimated_bytes * 8
        ):
            reason = "available disk is below the neural growth reserve"
        elif isinstance(ram_free, int) and ram_free < max(
            384 * 1024 * 1024, estimated_bytes * 4
        ):
            reason = "available memory is below the neural growth reserve"
        if reason:
            self.growth_pause = {
                "reason": reason,
                "readings": readings,
                "estimatedGrowthBytes": estimated_bytes,
                "at": _iso_now(),
            }
            return False
        return True

    def _resource_readings(self) -> Dict[str, Any]:
        probe = self.engine_path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        disk = shutil.disk_usage(str(probe))
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
            retain = (
                self.config.memory_recipe == "total-recall"
                and self.config.retain_source_text
            )
            try:
                learned = self.memory.learn(
                    text,
                    kind=kind,
                    source=source,
                    source_label=source_label,
                    retain_source_text=retain,
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
        self._append_replay(routed, importance=importance)
        workspace_salience = max(
            0.0,
            min(
                1.0,
                0.4 * float(learned["novelty"])
                + 0.35 * float(importance)
                + 0.25 * float(spike_metrics["spike_rate"]),
            ),
        )
        self._append_working_memory(
            routed,
            assembly_id=str(learned["idea_id"]),
            source=source,
            salience=workspace_salience,
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
        return {
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
            "grew_expert": grew,
        }

    def _latent_rehearsal_step(
        self, substrate_vector: torch.Tensor, seed: int
    ) -> Dict[str, float]:
        """Consolidate an internal assembly without manufacturing prompt text."""

        self.decoder.train()
        self.memory_bridge.train()
        self.idea_adapter.train()
        self.liquid.train()
        self._optimizer.zero_grad(set_to_none=True)
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
            stability_loss = self._stability_penalty()
            loss = (
                reconstruction_loss
                + 0.15 * temporal_loss
                + stability_loss
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite idle rehearsal loss")
        loss.backward()
        parameters = [
            parameter
            for module in (
                self.memory_bridge,
                self.idea_adapter,
                self.liquid,
            )
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
        self._accumulate_slow_importance(parameters)
        self._optimizer.step()
        self._commit_slow_anchors(rate=0.03)
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
            node = self.memory.neurons.get(assembly_id)
            if node is not None:
                uncertainty = float(node.get("uncertainty", 0.5))
                node["uncertainty"] = max(
                    0.0, min(1.0, uncertainty - sign * 0.04)
                )
                node["activation"] = max(
                    0.0,
                    min(
                        1.0,
                        float(node.get("activation", 0.0))
                        + sign * 0.03,
                    ),
                )
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
        if not self.config.idle_cognition:
            return {
                "brainId": self.brain_id,
                "ran": False,
                "reason": "idle-cognition-disabled",
                "actions": [],
            }
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
                float(
                    self.memory.neurons.get(
                        str(assembly.get("id", "")), {}
                    ).get("activation", 0.0)
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
        self._append_replay(routed, importance=max(0.4, compute_demand))
        normalized_tools = self._normalize_tool_schemas(tool_schemas)
        focus_labels = [
            str(
                self.memory.neurons.get(str(assembly["id"]), {}).get(
                    "label", assembly.get("kind", "assembly")
                )
            )
            for assembly in active[:8]
        ]
        action_state = {
            **organic,
            "computeDemand": compute_demand,
            "ponderScale": ponder_scale,
        }
        with torch.no_grad():
            action_scores, actions = self._select_structured_actions(
                self.decoder.internal_action_policy(routed),
                schemas=normalized_tools,
                input_text="neural focus: " + "; ".join(focus_labels),
                assembly_ids=[str(assembly["id"]) for assembly in active],
                organic_state=action_state,
            )
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
                        }
                    )
                    self._append_recent_dialogue("", message)
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
            "parameterChecksumBefore": parameter_before,
            "parameterChecksumAfter": parameter_after,
            "parameterDeltaNorm": parameter_delta,
            "actionPolicyScores": action_scores,
            "proposedActionKinds": [
                str(action.get("kind", "")) for action in actions
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
            grant = (
                str(schema.get("grant", "ask")).strip().lower()[:32] or "ask"
            )
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
    def _explicit_powershell_arguments(
        cls, text: str
    ) -> Optional[Dict[str, str]]:
        """Read a user-delimited command and working directory verbatim.

        A PowerShell proposal is intentionally impossible from vague prose.
        Both the command and an absolute, explicitly labelled cwd must occur
        in the user's current message; generated text is never consulted.
        """

        if not re.search(r"\b(?:powershell|pwsh)\b", text, re.IGNORECASE):
            return None
        fenced = re.search(
            r"\b(?:powershell|pwsh)(?:\s+command)?\s*(?:is\s*)?[:=]?\s*"
            r"```(?:powershell|pwsh)?\s*\n(?P<command>[\s\S]*?)```",
            text,
            re.IGNORECASE,
        )
        quoted = re.search(
            r"\b(?:run|execute)\s+(?:this\s+)?(?:powershell|pwsh)"
            r"(?:\s+command)?\s*(?:is\s*)?[:=]?\s*"
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

    @staticmethod
    def _explicit_browser_steps(
        text: str,
        initial_url: str,
    ) -> List[Dict[str, Any]]:
        """Encode only literal browser steps present in the user's message."""

        positioned: List[Tuple[int, Dict[str, Any]]] = []

        def quoted_value(match: re.Match[str], name: str) -> Optional[str]:
            value = match.group(name).replace("\x00", "").strip()
            if not value or "\r" in value or "\n" in value:
                return None
            return value

        for match in re.finditer(
            r"\bnavigate\s+(?:to\s+)?(?P<url>https://[^\s<>\"'`]+)",
            text,
            re.IGNORECASE,
        ):
            url = match.group("url").rstrip(".,;!?)]}")
            if url != initial_url and url in AdaptiveBrain._explicit_https_urls(url):
                positioned.append((match.start(), {"kind": "navigate", "url": url}))

        for match in re.finditer(
            r"\bclick(?:\s+on)?\s+(?P<quote>[\"'`])"
            r"(?P<selector>.*?)(?P=quote)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            selector = quoted_value(match, "selector")
            if selector and len(selector) <= 2_000:
                positioned.append(
                    (match.start(), {"kind": "click", "selector": selector})
                )

        for match in re.finditer(
            r"\btype\s+(?P<value_quote>[\"'`])(?P<value>.*?)"
            r"(?P=value_quote)\s+(?:into|in)\s+"
            r"(?P<selector_quote>[\"'`])(?P<selector>.*?)"
            r"(?P=selector_quote)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            value = quoted_value(match, "value")
            selector = quoted_value(match, "selector")
            if value and selector and len(value) <= 100_000 and len(selector) <= 2_000:
                positioned.append(
                    (
                        match.start(),
                        {
                            "kind": "type",
                            "selector": selector,
                            "value": value,
                            "clear": True,
                        },
                    )
                )

        for match in re.finditer(
            r"\bpress\s+(?:(?P<quote>[\"'`])(?P<quoted_key>.*?)"
            r"(?P=quote)|(?P<plain_key>[A-Za-z0-9_+.-]{1,32}))",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            key = (
                match.group("quoted_key")
                if match.group("quoted_key") is not None
                else match.group("plain_key")
            )
            key = key.replace("\x00", "").strip()
            if key and "\r" not in key and "\n" not in key and len(key) <= 64:
                positioned.append((match.start(), {"kind": "press", "key": key}))

        for match in re.finditer(
            r"\bwait\s+(?:for\s+)?(?P<quote>[\"'`])"
            r"(?P<selector>.*?)(?P=quote)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            selector = quoted_value(match, "selector")
            if selector and len(selector) <= 2_000:
                positioned.append(
                    (match.start(), {"kind": "wait", "selector": selector})
                )
        for match in re.finditer(
            r"\bwait\s+(?P<milliseconds>\d{1,8})\s*(?:ms|milliseconds?)\b",
            text,
            re.IGNORECASE,
        ):
            milliseconds = int(match.group("milliseconds"))
            if 0 < milliseconds <= 30_000:
                positioned.append(
                    (match.start(), {"kind": "wait", "milliseconds": milliseconds})
                )

        for match in re.finditer(
            r"\bextract(?:\s+(?:text|links|data))?(?:\s+from)?\s+"
            r"(?P<quote>[\"'`])(?P<selector>.*?)(?P=quote)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            selector = quoted_value(match, "selector")
            if selector and len(selector) <= 2_000:
                positioned.append(
                    (match.start(), {"kind": "extract", "selector": selector})
                )

        for match in re.finditer(
            r"\b(?:take\s+(?:a\s+)?)?screenshot\b",
            text,
            re.IGNORECASE,
        ):
            positioned.append((match.start(), {"kind": "screenshot"}))

        positioned.sort(key=lambda item: item[0])
        return [step for _position, step in positioned[:200]]

    def _materialize_generic_tool_action(
        self,
        *,
        schemas: Sequence[Mapping[str, Any]],
        input_text: str,
        assembly_ids: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Select one enabled standard tool and build validated typed inputs.

        The learned action head has already selected the external ``tool``
        channel. This second stage ranks only enabled schema/action pairs using
        explicit argument evidence plus VSA similarity to the current turn and
        active assemblies. It never reads generated response prose.
        """

        enabled = {
            str(schema.get("id", "")): {
                str(action) for action in schema.get("actions", [])
            }
            for schema in schemas
            if str(schema.get("grant", "ask")).strip().lower() != "off"
        }
        lowered = input_text.lower()
        paths = self._explicit_absolute_paths(input_text)
        urls = self._explicit_https_urls(input_text)
        candidates: List[
            Tuple[float, str, str, Dict[str, Any], str]
        ] = []

        def add(
            score: float,
            tool_id: str,
            action: str,
            arguments: Dict[str, Any],
            prototype: str,
        ) -> None:
            if action not in enabled.get(tool_id, set()):
                return
            candidates.append(
                (float(score), tool_id, action, arguments, prototype)
            )

        run_intent = bool(
            re.search(r"\b(?:run|execute|launch|test)\b", lowered)
        )
        write_intent = bool(
            re.search(r"\b(?:write|save|replace|overwrite|update)\b", lowered)
        )
        read_intent = bool(
            re.search(r"\b(?:read|inspect|view|show|display)\b", lowered)
        )
        list_intent = bool(
            re.search(
                r"\b(?:list|enumerate|folder|directory|files|contents)\b",
                lowered,
            )
        )
        powershell_arguments = self._explicit_powershell_arguments(input_text)
        if powershell_arguments is not None:
            add(
                8.0,
                "windows.powershell",
                "run",
                powershell_arguments,
                "run execute powershell command in explicit working directory",
            )
        if len(paths) == 1:
            path = paths[0]
            extension = Path(path.replace("\\", "/")).suffix.lower()
            languages = {
                ".py": "python",
                ".js": "javascript",
                ".mjs": "javascript",
                ".cjs": "javascript",
                ".ps1": "powershell",
            }
            if run_intent and extension in languages:
                add(
                    6.0,
                    "code.execute",
                    "run",
                    {
                        "language": languages[extension],
                        "entryPath": path,
                    },
                    "run execute code program script file",
                )
            if write_intent:
                content = self._explicit_write_content(input_text, paths)
                if content is not None:
                    add(
                        6.5,
                        "windows.files",
                        "write",
                        {"path": path, "content": content},
                        "write save replace file content",
                    )
            if list_intent and not run_intent and not write_intent:
                add(
                    5.0,
                    "windows.files",
                    "list",
                    {"path": path},
                    "list enumerate folder directory files contents",
                )
            if read_intent and not run_intent and not write_intent:
                add(
                    4.8,
                    "windows.files",
                    "read",
                    {"path": path},
                    "read inspect view show file contents",
                )

        search_match = re.search(
            r"\b(?:search|research|look\s+up|find)\b"
            r"(?:\s+(?:the\s+)?(?:web|internet|online))?"
            r"(?:\s+(?:for|about|on))?\s+(?P<query>.+)",
            input_text,
            re.IGNORECASE | re.DOTALL,
        )
        if search_match is not None:
            query = search_match.group("query").replace("\x00", "").strip()
            query = re.sub(r"\s+(?:please|thanks?)\s*$", "", query, flags=re.I)
            if query and not self._explicit_absolute_paths(query):
                add(
                    5.2,
                    "web.search",
                    "search",
                    {"query": query[:4_000]},
                    "search research find web internet information",
                )

        if len(urls) == 1:
            url = urls[0]
            fetch_intent = bool(
                re.search(
                    r"\b(?:fetch|download|get|retrieve|inspect|read|summarize)\b",
                    lowered,
                )
            )
            browser_intent = bool(
                re.search(
                    r"\b(?:browse|browser|open|visit|navigate|page|website)\b",
                    lowered,
                )
            )
            add(
                5.8 if fetch_intent else 2.6,
                "web.fetch",
                "fetch",
                {"url": url},
                "fetch retrieve download read web url",
            )
            browser_action = (
                "task"
                if "task" in enabled.get("browser.automation", set())
                else "open"
            )
            browser_steps = self._explicit_browser_steps(input_text, url)
            browser_arguments: Dict[str, Any] = {"url": url}
            if browser_steps:
                browser_arguments["steps"] = browser_steps
            add(
                7.2 if browser_steps else (5.9 if browser_intent else 2.5),
                "browser.automation",
                browser_action,
                browser_arguments,
                "open visit browse navigate website page url",
            )

        if not candidates:
            return None

        input_vector = self.memory.vector_for_text(input_text)
        active_vectors = [
            self.memory.assembly_vectors[assembly_id]
            for assembly_id in assembly_ids
            if assembly_id in self.memory.assembly_vectors
        ]
        active_vector = (
            self.memory.space.bundle(active_vectors)
            if active_vectors
            else None
        )
        ranked: List[
            Tuple[float, str, str, Dict[str, Any]]
        ] = []
        for evidence, tool_id, action, arguments, prototype in candidates:
            prototype_vector = self.memory.vector_for_text(prototype)
            neural_score = 0.22 * self.memory.space.similarity(
                input_vector, prototype_vector
            )
            if active_vector is not None:
                neural_score += 0.08 * self.memory.space.similarity(
                    active_vector, prototype_vector
                )
            ranked.append(
                (evidence + neural_score, tool_id, action, arguments)
            )
        _score, tool_id, action, arguments = sorted(
            ranked,
            key=lambda item: (-item[0], item[1], item[2]),
        )[0]
        return {
            "toolId": tool_id,
            "action": action,
            "arguments": arguments,
        }

    def _select_structured_actions(
        self,
        action_logits: torch.Tensor,
        *,
        schemas: Sequence[Mapping[str, Any]],
        input_text: str,
        assembly_ids: Sequence[str],
        organic_state: Mapping[str, float],
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
        available = {
            str(schema["id"]): set(
                str(value) for value in schema.get("actions", [])
            )
            for schema in schemas
            if str(schema.get("grant", "ask")).strip().lower() != "off"
        }
        tension = float(organic_state.get("computeDemand", 0.0))

        # A blank/random brain has a nearly uniform head. It must learn a
        # decisive distribution before proposing external activity. Safe
        # internal cognition may emerge sooner when unresolved neural tension
        # is high.
        if kind == "talk":
            return scores, actions
        if kind in {"ponder", "learn"}:
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
        if confidence < ACTION_PROPOSAL_CONFIDENCE:
            return scores, actions

        base_arguments: Dict[str, Any] = {
            "assemblyIds": list(assembly_ids),
            "organic": True,
        }
        if kind == "tool":
            selected_tool = self._materialize_generic_tool_action(
                schemas=schemas,
                input_text=input_text,
                assembly_ids=assembly_ids,
            )
            if selected_tool is not None:
                actions.append(
                    {
                        "kind": "tool",
                        "toolId": selected_tool["toolId"],
                        "action": selected_tool["action"],
                        "arguments": {
                            **base_arguments,
                            **selected_tool["arguments"],
                        },
                        "confidence": confidence,
                    }
                )
        elif (
            kind == "imagine"
            and "generate" in available.get("modality.imagine", set())
        ):
            # The recurrent state selects a modality; active assemblies cue the
            # generator without constructing a hidden text prompt.
            modality_index = int(
                abs(float(self.liquid_state.detach().float().sum().item()))
                * 1000
            ) % 3
            modality = ("image", "audio", "video")[modality_index]
            actions.append(
                {
                    "kind": "imagine",
                    "toolId": "modality.imagine",
                    "action": "generate",
                    "arguments": {
                        **base_arguments,
                        "conceptIds": list(assembly_ids),
                        "modality": modality,
                    },
                    "confidence": confidence,
                }
            )
        elif kind == "agent" and "start" in available.get("agent.fork", set()):
            actions.append(
                {
                    "kind": "agent",
                    "toolId": "agent.fork",
                    "action": "start",
                    "arguments": {
                        **base_arguments,
                        "objective": input_text,
                    },
                    "confidence": confidence,
                }
            )
        elif (
            kind == "evolve"
            and self.config.recursive_improvement
            and "propose" in available.get("source.self-modify", set())
        ):
            # An organic action head cannot author a trustworthy source patch:
            # exact paths, complete replacement text, and expected hashes must
            # come through the typed source-edit channel.  Route an edit-free
            # thought into the worker-owned substrate overlay instead.  The
            # current turn has already entered latent replay, so this is a
            # viable isolated experiment rather than an empty Git candidate.
            actions.append(
                {
                    "kind": "evolve",
                    "toolId": "source.self-modify",
                    "action": "propose",
                    "arguments": {
                        **base_arguments,
                        "objective": input_text,
                        "recursive": True,
                        "candidateKind": "substrate",
                        "latentReplay": True,
                    },
                    "confidence": confidence,
                }
            )
        return scores, actions

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
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
        stream_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            raise ValueError("chat input cannot be empty")
        if len(clean) > 1_000_000:
            raise ValueError("chat input is too large")
        requested_generation_tokens = (
            max(1, int(max_new_tokens))
            if max_new_tokens is not None
            else None
        )
        before_checksum = self.parameter_checksum()
        before_parameters = self._parameter_copy()
        normalized_tools = self._normalize_tool_schemas(tool_schemas)

        cue = self.memory.vector_for_text(clean)
        if self.config.vector_symbolic_memory:
            recalled_vector, recalled = self.memory.recall_vector(
                cue, workspace_slots=self.config.working_memory_slots
            )
        else:
            recalled_vector, recalled = cue, []
        recall_audit = dict(self.memory._last_recall_audit)
        experience = self.learn_experience(
            clean,
            kind="question" if clean.rstrip().endswith("?") else "experience",
            source="conversation",
            source_label="chat",
            # Fast substrate/STDP activity happens before the decision. Slow
            # shared-representation learning is committed after generation so
            # the current action uses the previously persisted calibration.
            steps=0,
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
        self.current_context = {
            "tokenCount": int(prompt_ids.shape[1]),
            "tokenHash": prompt_token_hash,
            "recentTokenCount": len(self.recent_token_context),
            "recentTokenHash": self._token_sequence_hash(
                self.recent_token_context
            ),
            "sensorySlots": 0,
            "updatedAt": _iso_now(),
        }
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
        generation_tokens = (
            requested_generation_tokens
            if requested_generation_tokens is not None
            else self.config.generation_token_budget(compute_demand)
        )
        ponder_factors = {
            "liquid": liquid_ponder,
            "novelty": novelty,
            "uncertainty": uncertainty,
            "predictionError": prediction_error,
            "learningProgress": learning_progress,
            "curiosity": curiosity,
        }
        ponder_steps = max(
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
        candidates = []
        # Candidate computation has no fixed "parallel thoughts" or hardware
        # branch ceiling. Neural energy is derived from the current organic
        # state and working workspace, then spent until activity converges or
        # the host reserve reports pressure.
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
                }
            )
            improvement = intrinsic_score - best_score
            score_change = abs(improvement) if math.isfinite(best_score) else float("inf")
            best_score = max(best_score, intrinsic_score)
            normalized_entropy = max(0.0, min(4.0, entropy)) / 4.0
            remaining_neural_energy -= (
                1.0
                + 0.35 * normalized_entropy
                + 0.20 * max(0.0, 1.0 - compute_demand)
            )
            branch += 1

            # Settling is based on measured candidate change, not a count.
            convergence_floor = 0.0025 * (
                1.0 + abs(best_score)
            )
            if (
                len(candidates) > 1
                and score_change <= convergence_floor
            ):
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
        selected_branch = max(
            range(len(candidates)), key=lambda index: candidates[index]["score"]
        )
        selected = candidates[selected_branch]
        generated = selected["tensor"]
        entropies = selected["entropies"]
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
                "branches": branch_count,
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
            )

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

            # Candidate selection remains private neural computation. Replay
            # the chosen deterministic branch once so the visible stream is
            # exactly the committed response rather than a discarded branch.
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
            )
            if not torch.equal(replayed, generated):
                raise RuntimeError(
                    "deterministic selected-branch replay diverged"
                )
            generated = replayed
            entropies = replay_entropies

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
        action_calibration = None
        slow_mutation_requested = bool(
            self.config.online_learning and int(self.config.online_steps) > 0
        )
        slow_mutation_applied = False
        slow_mutation_rolled_back = False
        slow_mutation_failure: Optional[Dict[str, Any]] = None
        slow_mutation_stage = "disabled"
        slow_parameter_checksum_before = self._slow_parameter_checksum()
        slow_parameter_checksum_after = slow_parameter_checksum_before
        if slow_mutation_requested:
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
                pair_training = self._optimize_dialogue_pair(
                    clean, response, cue, steps=1
                )
                # The current turn and optional self-response entered fast
                # neural state before action selection with steps=0. Defer
                # topology growth until the full slow update is available for
                # starter-policy retention to validate atomically.
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
                if self._can_retain_bundled_action_policy():
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
                    action_calibration = self._retain_starter_action_policy(
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
                            "starter action retention rejected the slow mutation"
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
        trace = {
            "id": uuid.uuid4().hex,
            "created_at": _iso_now(),
            "seed": int(seed),
            "input_sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
            "textual_memory_injected": False,
            "long_term_source_text_injected": False,
            "tool_schema_text_injected": False,
            "hidden_prompt_text_expanded": False,
            "prompt_text_expanded": bool(recent_prompt_tokens),
            "prompt_token_count": int(prompt_ids.shape[1]),
            "recent_dialogue_context_injected": bool(recent_prompt_tokens),
            "recent_dialogue_token_count": len(recent_prompt_tokens),
            "recent_dialogue_token_ids_sha256": self._token_sequence_hash(
                recent_prompt_tokens
            ),
            "context_token_evictions": self.counters[
                "context_token_evictions"
            ],
            "working_context_capacity_tokens": self.config.max_seq_len,
            "generation_budget_tokens": generation_tokens,
            "generation_budget_source": (
                "caller"
                if requested_generation_tokens is not None
                else "hardware-and-organic-state"
            ),
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
            "slow_parameter_checksum_before": (
                slow_parameter_checksum_before
            ),
            "slow_parameter_checksum_after": slow_parameter_checksum_after,
            "generation_entropy": (
                sum(entropies) / len(entropies) if entropies else 0.0
            ),
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
                }
                for index, candidate in enumerate(candidates)
            ],
            "selected_branch": selected_branch,
            "steps": [
                {
                    "stage": "encode",
                    "detail": (
                        "Encoded the current turn plus explicit bounded recent "
                        "dialogue at the UTF-8 token boundary."
                    ),
                    "value": "%d tokens" % prompt_ids.shape[1],
                },
                {
                    "stage": "idea-memory",
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
                    "stage": "working-memory",
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
                {
                    "stage": "tool-schema",
                    "detail": (
                        "Encoded enabled tool IDs and actions through the "
                        "neural capability channel; no schema text was added to "
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
                    "detail": (
                        "Updated ternary decoder and idea-consolidation master "
                        "parameters and committed the validated transaction."
                        if slow_mutation_applied
                        else (
                            "Attempted the slow neural update, then restored "
                            "all slow parameters, expert topology, optimizer, "
                            "metaplastic state, and counters after validation "
                            "failed. Fast substrate and working-memory state "
                            "from the turn remain active."
                            if slow_mutation_rolled_back
                            else (
                                "Online slow learning was disabled; no slow "
                                "decoder or consolidation parameter was updated."
                            )
                        )
                    ),
                    "value": (
                        "loss %.6f" % train_loss
                        if slow_mutation_applied
                        else (
                            "rolled back at %s"
                            % (
                                slow_mutation_failure.get("stage", "unknown")
                                if slow_mutation_failure is not None
                                else "unknown"
                            )
                            if slow_mutation_rolled_back
                            else "online_steps=0; no slow parameter update"
                        )
                    ),
                },
                {
                    "stage": "dialogue-learning",
                    "detail": (
                        "Learned and committed the completed human-to-brain "
                        "role-boundary sequence."
                        if pair_training is not None
                        else (
                            "The attempted dialogue update was rolled back with "
                            "the enclosing slow-learning transaction."
                            if slow_mutation_rolled_back
                            else "Online slow dialogue learning was disabled."
                        )
                    ),
                    "value": (
                        "loss %.6f" % pair_training["loss"]
                        if pair_training is not None
                        else (
                            "rolled back"
                            if slow_mutation_rolled_back
                            else "online_steps=0; no dialogue parameter update"
                        )
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
            "actions": proposed_actions,
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
            max_length=self.config.max_seq_len,
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
        payload_size = max(1, self.config.max_seq_len - 2)
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
                def epoch_units():
                    for sample in samples:
                        vector = self.memory.vector_for_text(sample)
                        for window in self.tokenizer.window_tensors(
                            sample,
                            self.device,
                            max_length=self.config.max_seq_len,
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
                    micro_batches = [
                        group[index : index + physical_batch]
                        for index in range(0, len(group), physical_batch)
                    ]
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
        steps = int(steps)
        if steps < 1:
            raise ValueError("consolidation steps must be positive")
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

    def _media_idea(self, source_name: str) -> torch.Tensor:
        vector = self.memory.vector_for_text(source_name or "media experience")
        return self._idea_model_vector(vector).detach()

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
                "audio format needs soundfile or the bundled FFmpeg decoder"
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
                    "MP4/WebM/MOV video needs the bundled FFmpeg decoder; GIF works with Pillow"
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
                    "complete": False,
                },
            }
        optimizer = torch.optim.AdamW(
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
        for target, actual_units in window_stream:
            windows += 1
            processed_units += int(actual_units)
            if int(actual_units) < window_size:
                tail_units = int(actual_units)
            for index in range(steps_per_window):
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
        media_coverage: Dict[str, Any] = {
            "unit": unit,
            "windowSize": window_size,
            "windows": windows,
            "discoveredUnits": processed_units,
            "processedUnits": processed_units,
            "tailUnits": tail_units,
            "complete": True,
        }
        if unit == "samples":
            media_coverage["processedSamples"] = processed_units
            media_coverage["tailSamples"] = tail_units
        elif unit == "frames":
            media_coverage["processedFrames"] = processed_units
            media_coverage["tailFrames"] = tail_units
        return {
            "trained": True,
            "loss": loss_sum / loss_count,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "steps": loss_count,
            "stepsPerWindow": steps_per_window,
            "windows": windows,
            "coverage": media_coverage,
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
        if not reports:
            return {
                "trained": False,
                "loss": 0.0,
                "steps": 0,
                "warnings": [],
                "coverage": self._empty_media_coverage(
                    complete=complete_when_empty
                ),
                "records": [],
            }

        total_loss = 0.0
        total_steps = 0
        warnings: List[str] = []
        by_modality: Dict[str, Dict[str, Any]] = {}
        trained_records = 0
        sanitized_records: List[Dict[str, Any]] = []
        for report in reports:
            kind = str(report["kind"])
            trained = bool(report.get("trained", False))
            if trained:
                trained_records += 1
            steps = max(0, int(report.get("steps", 0)))
            total_loss += float(report.get("loss", 0.0)) * max(1, steps)
            total_steps += steps
            report_warnings = [str(value) for value in report.get("warnings", [])]
            warnings.extend(
                "%s: %s" % (report["name"], warning)
                for warning in report_warnings
            )
            record_coverage = dict(report.get("coverage", {}))
            bucket = by_modality.setdefault(
                kind,
                {
                    "unit": record_coverage.get("unit", "unknown"),
                    "windowSize": int(record_coverage.get("windowSize", 0)),
                    "windows": 0,
                    "discoveredUnits": 0,
                    "processedUnits": 0,
                    "tailUnits": 0,
                    "records": 0,
                    "trainedRecords": 0,
                    "failedRecords": 0,
                    "complete": True,
                },
            )
            bucket["windows"] += int(record_coverage.get("windows", 0))
            bucket["discoveredUnits"] += int(
                record_coverage.get("discoveredUnits", 0)
            )
            bucket["processedUnits"] += int(
                record_coverage.get("processedUnits", 0)
            )
            bucket["tailUnits"] += int(record_coverage.get("tailUnits", 0))
            bucket["records"] += 1
            bucket["trainedRecords"] += 1 if trained else 0
            bucket["failedRecords"] += 0 if trained else 1
            bucket["complete"] = bool(bucket["complete"]) and bool(
                record_coverage.get("complete", False)
            )
            sanitized_records.append(
                {
                    "name": str(report["name"]),
                    "kind": kind,
                    "contentSha256": str(report.get("contentSha256", "")),
                    "provenance": dict(report.get("provenance", {})),
                    "trained": trained,
                    "loss": float(report.get("loss", 0.0)),
                    "steps": steps,
                    "coverage": record_coverage,
                    "warnings": report_warnings,
                }
            )

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
            "records": len(reports),
            "trainedRecords": trained_records,
            "failedRecords": len(reports) - trained_records,
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
            "trained": trained_records > 0,
            "loss": total_loss / max(1, total_steps),
            "steps": total_steps,
            "warnings": warnings,
            "coverage": coverage,
            "records": sanitized_records,
        }

    def ingest(
        self,
        path: Optional[str] = None,
        text: Optional[str] = None,
        name: str = "",
        kind: str = "",
        policy: str = "encode",
        expected_hash: str = "",
        allow_replay: bool = False,
        epoch: int = 0,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if policy not in {"encode", "consolidate", "pretrain", "archive"}:
            raise ValueError("unsupported ingestion policy")
        source_path: Optional[Path] = None
        raw_bytes: Optional[bytes] = None
        extracted = ""
        coverage = DatasetCoverage()
        if path:
            source_path = Path(path).resolve()
            if not source_path.exists() or not source_path.is_file():
                raise FileNotFoundError(str(source_path))
            resolved_kind = dataset_format(source_path, kind)
            source_name = name or source_path.name
            source_bytes = source_path.stat().st_size
            digest = hashlib.sha256()
            with source_path.open("rb") as source_stream:
                for block in iter(lambda: source_stream.read(1024 * 1024), b""):
                    digest.update(block)
            content_hash = digest.hexdigest()
        elif text is not None:
            extracted = str(text).replace("\x00", "")
            raw_bytes = extracted.encode("utf-8")
            resolved_kind = kind or "text"
            source_name = name or "pasted-text"
            source_bytes = len(raw_bytes)
            content_hash = hashlib.sha256(raw_bytes).hexdigest()
        else:
            raise ValueError("ingest requires path or text")
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
        if duplicate is not None and not allow_replay:
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
        loss_total = 0.0
        learned_chunks = 0
        retained_parts: Optional[List[str]] = (
            []
            if self.config.memory_recipe == "total-recall"
            and self.config.retain_source_text
            else None
        )
        media_reports: List[Dict[str, Any]] = []
        if policy != "archive":
            if source_path is not None:
                record_stream = iter_dataset_records(
                    source_path,
                    requested_kind=resolved_kind,
                    coverage=coverage,
                )
            else:
                coverage.discovered_files = 1
                coverage.completed_files = 1
                coverage.discovered_records = 1 if extracted.strip() else 0
                coverage.processed_records = 1 if extracted.strip() else 0
                coverage.processed_bytes = source_bytes
                record_stream = []

            for record in record_stream:
                record_kind = str(getattr(record, "kind", "text"))
                record_name = str(getattr(record, "name", source_name))
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
                        media_reports.append(
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
                            }
                        )
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
                        )
                    except (RuntimeError, ValueError, OSError) as error:
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
                    media_reports.append(
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
                        }
                    )
                    continue

                record_text = str(getattr(record, "text", ""))
                if retained_parts is not None:
                    retained_parts.append(record_text)
                for chunk in self._experience_chunks(record_text):
                    learned = self.learn_experience(
                        chunk,
                        kind="knowledge",
                        source="document",
                        source_label=record_name,
                        steps=2 if policy == "pretrain" else 1,
                        importance=0.65,
                    )
                    loss_total += float(learned["training"]["loss"])
                    learned_chunks += 1
                if progress is not None:
                    progress(
                        min(
                            0.99,
                            coverage.processed_bytes
                            / float(max(1, source_bytes)),
                        ),
                        "Encoding %s (%d records)"
                        % (source_name, coverage.processed_records),
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
                media_reports.append(
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
                    }
                )
            elif source_path is None and extracted.strip():
                if retained_parts is not None:
                    retained_parts.append(extracted)
                for chunk in self._experience_chunks(extracted):
                    learned = self.learn_experience(
                        chunk,
                        kind="knowledge",
                        source="document",
                        source_label=source_name,
                        steps=2 if policy == "pretrain" else 1,
                        importance=0.65,
                    )
                    loss_total += float(learned["training"]["loss"])
                    learned_chunks += 1
        else:
            coverage.discovered_files = 1
            coverage.completed_files = 1
            coverage.processed_bytes = source_bytes
        media_result = self._aggregate_media_reports(
            media_reports,
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
            and len(media_reports) == 1
            and str(media_reports[0].get("kind", "")) == "video"
        ):
            effective_source_kind = "video"
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
            "learned_ideas": len(self.memory.ideas) - before_ideas,
            "learned_concepts": len(self.memory.concepts) - before_concepts,
            "plasticity_events": int(
                self.router.synapses.plasticity_events.item()
            )
            - before_events,
            "raw_text_retained": False,
            "modality_trained": bool(media_result["trained"]),
            "media_coverage": dict(media_result["coverage"]),
            "media_records": list(media_result["records"]),
            "warnings": list(media_result["warnings"]),
            "coverage": coverage.as_dict(),
        }
        if retained_parts:
            source_record["raw_text"] = "\n".join(retained_parts)
            source_record["raw_text_retained"] = True
        if duplicate is None:
            self.training_sources.append(source_record)
        else:
            self.training_sources = [
                source_record
                if source.get("id") == duplicate.get("id")
                else source
                for source in self.training_sources
            ]
        if policy == "consolidate":
            self.consolidate(max(1, min(8, learned_chunks)), progress=progress)
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
                "mediaCoverage": source_record["media_coverage"],
                "warnings": source_record["warnings"],
            },
        )
        return {
            "brainId": self.brain_id,
            "duplicate": False,
            "source": source_record,
            "meanLoss": loss_total / learned_chunks if learned_chunks else 0.0,
            "modalityLoss": float(media_result["loss"]),
            "mediaCoverage": dict(media_result["coverage"]),
            "warnings": list(media_result["warnings"]),
            "coverage": coverage.as_dict(),
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
    def _mp4_bytes(video: torch.Tensor, fps: int = 8) -> bytes:
        """Encode a generated tensor into a browser-viewable H.264 MP4."""

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("MP4 output requires the bundled FFmpeg runtime") from error
        value = video.detach().cpu().float()
        if value.ndim == 5:
            value = value[0]
        if value.ndim != 4 or value.shape[0] != 3:
            raise ValueError("video output must have shape [3, frames, height, width]")
        value = ((value.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        frames = value.permute(1, 2, 3, 0).contiguous()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        frame_rate = max(1, min(60, int(fps)))
        with tempfile.TemporaryDirectory(prefix="omni-video-output-") as temporary:
            output = Path(temporary) / "generated.mp4"
            subprocess.run(
                [
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
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                input=frames.numpy().tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )
            encoded = output.read_bytes()
        if len(encoded) < 12 or encoded[4:8] != b"ftyp":
            raise RuntimeError("FFmpeg did not produce a valid MP4 container")
        return encoded

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
                cue, workspace_slots=self.config.working_memory_slots
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
        preview_callback: Optional[
            Callable[[float, str, bytes], None]
        ] = None,
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
            def encode_preview(
                progress_value: float, tensor: torch.Tensor
            ) -> None:
                if preview_callback is None:
                    return
                if modality == "image":
                    preview_callback(
                        progress_value,
                        "image/png",
                        self._png_bytes(tensor),
                    )
                elif modality == "audio":
                    preview_callback(
                        progress_value,
                        "audio/wav",
                        self._wav_bytes(
                            tensor,
                            sample_rate=int(
                                _finite_number(
                                    (settings or {}).get("sampleRate"),
                                    16000,
                                )
                            ),
                        ),
                    )
                elif modality == "video":
                    preview_callback(
                        progress_value,
                        "image/apng",
                        self._apng_bytes(
                            tensor,
                            fps=int(
                                _finite_number(
                                    (settings or {}).get("fps"), 8
                                )
                            ),
                        ),
                    )

            output = self.modalities.generate(
                modality,
                idea,
                seed=seed,
                preview_callback=(
                    encode_preview if preview_callback is not None else None
                ),
            )
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
                fps = int(_finite_number((settings or {}).get("fps"), 8))
                container_fallback = ""
                try:
                    artifact_bytes = self._mp4_bytes(output, fps=fps)
                    artifact = artifact_dir / (artifact_id + ".mp4")
                    mime_type = "video/mp4"
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
                artifact.write_bytes(artifact_bytes)
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
            if modality == "video":
                result["containerFallback"] = container_fallback
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
        packed = self.export_packed_ternary()
        clean_label = SAFE_NAME.sub("-", label.strip()).strip(".-")[:48] or "snapshot"
        snapshot_id = "%s-%s" % (clean_label, uuid.uuid4().hex[:12])
        destination = self.engine_path / "snapshots" / snapshot_id
        snapshot_files(self.engine_path, destination)
        shutil.copytree(
            self.engine_path / "packed-ternary",
            destination / "packed-ternary",
        )
        checksum = hashlib.sha256(
            (destination / "core.safetensors").read_bytes()
            + (destination / "plasticity.safetensors").read_bytes()
            + str(
                read_json(destination / "brain.json")
                .get("substrate", {})
                .get("persistence", {})
                .get("contentSha256", "")
            ).encode("ascii")
        ).hexdigest()
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
            + int(
                self.router.synapses.effective_weight().ne(0).sum().item()
            ),
            "activeSynapses": int(
                self.router.synapses.effective_weight().ne(0).sum().item()
            ),
            "plasticityEvents": int(
                self.router.synapses.plasticity_events.item()
            ),
            "messages": len(self.messages),
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
            "substrate": str(self.engine_path / "substrate" / "manifest.json"),
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
            ("long_term_threshold", "longTermThreshold", float),
            ("forgetting_rate", "forgettingRate", float),
            ("consolidation_rate", "consolidationRate", float),
        ):
            assign(attribute, key, transform)
        if "learningRate" in raw:
            neural_rate = max(
                1e-5, min(0.02, float(raw["learningRate"]) * 0.02)
            )
            if self.config.learning_rate != neural_rate:
                self.config.learning_rate = neural_rate
                changed.append("learning_rate")
        assign("memory_recipe", "memoryRecipe", str)
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
        self.population_controls_from_config()
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
