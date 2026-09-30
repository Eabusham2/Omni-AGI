"""Configuration and hardware-aware tiny defaults for OmniCortex."""

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional

from .native_architecture import validate_native_architecture


# Stable v1 no longer exposes these beta builder controls. Removed fields are
# accepted-and-discarded by the legacy dictionary loader; a few mandatory
# substrate flags remain internal implementation state but are filtered from
# saved public configuration. AdaptiveBrain enforces those invariants.
_DEPRECATED_BETA_CONTROL_FIELDS = frozenset(
    {
        "ternary_weights",
        "spiking_dynamics",
        "stdp_plasticity",
        "liquid_dynamics",
        "vector_symbolic_memory",
        "consolidation_enabled",
        "metaplasticity",
        "noise",
        "memory_injection",
        "learn_from_own_messages",
        "max_concepts",
        "max_ideas",
        "max_synapses",
        "novelty_drive",
        "coherence_drive",
        "curiosity_drive",
        "parallel_thoughts",
        "growth_policy",
        "max_experts",
    }
)


def safe_rounded_storage_bytes_per_second(value: Any) -> int:
    """Match the desktop's safe rounding for measured byte rates."""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0
    return max(1, min((1 << 53) - 1, math.floor(numeric + 0.5)))


@dataclass
class OmniConfig:
    """Serializable architecture and learning configuration.

    Defaults describe the Personal hardware profile. Tests and constrained
    hosts use :meth:`micro`; the desktop resolves all new builds from an
    operating-system hardware profile.
    """

    name: str = "New OmniCortex"
    seed: int = 7
    vocab_size: int = 261
    # Stable-v1 Personal defaults.  This is temporary neural working context,
    # not long-term memory and not a response-length control.  New desktop
    # builds resolve it from the hardware profiles in ``from_external``.
    max_seq_len: int = 1024
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 160
    dropout: float = 0.0
    idea_dim: int = 64
    vsa_dim: int = 256
    router_neurons: int = 64
    hardware_tier: str = "personal"
    # Every OmniCortex starts from this process's seeded random initialization.
    origin_kind: str = "ground-up"
    # Trusted main-selected Build descriptor. Legacy saved/research shapes
    # omit it and retain their exact explicit tensor dimensions unchanged.
    native_architecture: Optional[Dict[str, Any]] = None
    train_batch_size: int = 2
    gradient_accumulation: int = 2
    gradient_checkpointing: bool = False
    # -1 preserves legacy all-expert softmax. Compatible expansion pins the
    # old pool's normalization and adds independently gated zero residuals.
    expert_routing_baseline_count: int = -1
    # Auto is RAM-first and continuously clamps the physical batch/window to
    # live RAM/accelerator headroom. Manual budgets are optional operational
    # ceilings; they never permit crossing the live safety watermark.
    training_resource_mode: str = "auto"
    # One process-wide envelope shared by model, training, working memory,
    # modalities, caches, and agents. Zero means hardware-derived Auto; a
    # manual value is a percentage of memory left after the OS reserve.
    system_ram_share_percent: float = 0.0
    training_ram_budget_bytes: int = 0
    training_accelerator_budget_bytes: int = 0
    training_scratch_budget_bytes: int = 0
    storage_bytes_per_second: int = 0

    ternary_weights: bool = True
    spiking_dynamics: bool = True
    stdp_plasticity: bool = True
    liquid_dynamics: bool = True
    vector_symbolic_memory: bool = True
    online_learning: bool = True
    consolidation_enabled: bool = True
    metaplasticity: bool = True
    vision_enabled: bool = True
    image_enabled: bool = True
    audio_enabled: bool = True
    video_enabled: bool = True

    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    online_steps: int = 1
    slow_stability_strength: float = 0.025
    slow_importance_decay: float = 0.97
    temperature: float = 0.9
    top_k: int = 40

    membrane_leak: float = 0.88
    firing_threshold: float = 0.55
    stdp_learning_rate: float = 0.035
    stdp_tau_pre: float = 8.0
    stdp_tau_post: float = 8.0
    stdp_a_plus: float = 1.0
    stdp_a_minus: float = 1.05
    metaplasticity_rate: float = 0.025

    liquid_mode: str = "cfc"
    liquid_steps: int = 3
    memory_recipe: str = "adaptive-retention"
    memory_injection: str = "working-memory"
    learn_from_own_messages: bool = True
    retain_source_text: bool = False
    extended_working_memory: bool = False
    recursive_improvement: bool = True
    idle_cognition: bool = True
    working_memory_slots: int = 256
    working_memory_mode: str = "auto"
    memory_offload_bytes: int = 0
    memory_resident_items: int = 256
    memory_offload_slowdown_percent: float = 0.0
    short_term_half_life_minutes: float = 45.0
    # Legacy checkpoint input only. Replay admission is now continuously
    # weighted; this value is not a live memory threshold or public control.
    long_term_threshold: float = 0.62
    forgetting_rate: float = 0.002
    consolidation_rate: float = 0.06

    growth_novelty_threshold: float = 0.92
    growth_patience: int = 3

    image_size: int = 16
    audio_samples: int = 256
    video_frames: int = 4
    modality_channels: int = 16
    device: str = "cpu"
    # Zero selects an adaptive hardware-derived reserve. These are operational
    # safety boundaries, not neural-size or memory-cardinality caps.
    ram_reserve_bytes: int = 0
    disk_reserve_bytes: int = 0
    disk_state_offload: bool = True
    storage_pool_bytes: int = 0
    working_attention_scratch_budget_bytes: Optional[int] = None

    def validate(self) -> None:
        # Stable v1 treats prompt-free recurrence/Ponder as a permanent
        # architectural pathway. Learned neural state can select zero passes,
        # but neither a legacy import nor a direct research constructor may
        # remove the pathway itself.
        self.idle_cognition = True
        self.storage_bytes_per_second = safe_rounded_storage_bytes_per_second(
            self.storage_bytes_per_second
        )
        # Direct research/test constructors may override only the total
        # workspace. The resident portion is a subset, so normalize it before
        # validation; public builds supply a measured value from preflight.
        self.memory_resident_items = max(
            1,
            min(int(self.memory_resident_items), int(self.working_memory_slots)),
        )
        if self.vocab_size < 261:
            raise ValueError("vocab_size must fit bytes and role boundaries (at least 261)")
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for rotary positions")
        if self.idea_dim != self.d_model:
            raise ValueError("idea_dim must currently equal d_model")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len must be at least 8")
        if self.router_neurons <= 1 or self.vsa_dim <= 8:
            raise ValueError("router_neurons and vsa_dim are too small")
        if self.memory_recipe not in {
            "adaptive-retention",
            "human-consolidation",
            "total-recall",
            "synapses-only",
        }:
            raise ValueError("unsupported memory_recipe")
        if self.memory_injection not in {"parameter-only", "working-memory"}:
            raise ValueError("unsupported memory_injection")
        if self.liquid_mode not in {"cfc", "ltc"}:
            raise ValueError("liquid_mode must be cfc or ltc")
        if self.hardware_tier not in {"micro", "personal", "gpu", "workstation"}:
            raise ValueError("unsupported hardware_tier")
        if self.origin_kind != "ground-up":
            raise ValueError("origin_kind must be ground-up")
        if self.image_size < 8 or self.image_size % 4:
            raise ValueError("image_size must be a multiple of four and at least 8")
        if self.video_frames < 2:
            raise ValueError("video_frames must be at least 2")
        if self.working_memory_slots < 1:
            raise ValueError("working-memory slots must be positive")
        if self.working_memory_mode not in {"auto", "extended", "manual"}:
            raise ValueError("working_memory_mode must be auto, extended, or manual")
        if self.memory_offload_bytes < 0:
            raise ValueError("memory_offload_bytes cannot be negative")
        if self.memory_resident_items < 1:
            raise ValueError("memory_resident_items must be positive")
        if not 0.0 <= self.memory_offload_slowdown_percent <= 95.0:
            raise ValueError("memory_offload_slowdown_percent must be in [0, 95]")
        if self.train_batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("training batch size and accumulation must be positive")
        if (type(self.expert_routing_baseline_count) is not int
            or self.expert_routing_baseline_count < -1):
            raise ValueError("expert routing baseline must be -1 or a nonnegative integer")
        if self.training_resource_mode not in {"auto", "manual"}:
            raise ValueError("training_resource_mode must be auto or manual")
        if self.system_ram_share_percent != 0.0 and not (
            30.0 <= self.system_ram_share_percent <= 100.0
        ):
            raise ValueError(
                "system_ram_share_percent must be auto (0) or in [30, 100]"
            )
        if min(
            self.training_ram_budget_bytes,
            self.training_accelerator_budget_bytes,
            self.training_scratch_budget_bytes,
            self.storage_bytes_per_second,
        ) < 0:
            raise ValueError("training resource budgets cannot be negative")
        if self.short_term_half_life_minutes <= 0:
            raise ValueError("short-term half-life must be positive")
        if self.slow_stability_strength < 0:
            raise ValueError("slow_stability_strength cannot be negative")
        if not 0.0 <= self.slow_importance_decay < 1.0:
            raise ValueError("slow_importance_decay must be in [0, 1)")
        if self.ram_reserve_bytes < 0 or self.disk_reserve_bytes < 0:
            raise ValueError("resource reserve bytes cannot be negative")
        if isinstance(self.storage_pool_bytes, bool) or self.storage_pool_bytes < 0:
            raise ValueError("storage pool bytes must be nonnegative")
        if (self.working_attention_scratch_budget_bytes is not None
            and (isinstance(self.working_attention_scratch_budget_bytes, bool)
                 or self.working_attention_scratch_budget_bytes < 0)):
            raise ValueError("working attention scratch bytes must be nonnegative")
        if self.native_architecture is not None:
            descriptor = validate_native_architecture(self.native_architecture)
            from .architecture_migration import validate_compatible_architecture_lineage
            validate_compatible_architecture_lineage(descriptor)
            shape = descriptor["shape"]
            declared = {
                "dModel": self.d_model, "layers": self.n_layers, "feedForward": self.d_ff,
                "nHeads": self.n_heads, "vsaDimensions": self.vsa_dim,
                "routerNeurons": self.router_neurons, "modalityChannels": self.modality_channels,
                "imageSize": self.image_size, "audioSamples": self.audio_samples,
                "videoFrames": self.video_frames, "workingMemoryItems": self.working_memory_slots,
                "workspaceLatents": max(8, self.working_memory_slots // 4),
                "vocabSize": self.vocab_size, "liquidMode": self.liquid_mode,
            }
            if dict(shape) != declared or descriptor["hardwareTier"] != self.hardware_tier:
                raise ValueError("saved native descriptor does not match explicit checkpoint dimensions")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in _DEPRECATED_BETA_CONTROL_FIELDS
            and key != "long_term_threshold"
            and not (key == "native_architecture" and value is None)
            and not (key == "storage_pool_bytes" and value == 0)
            and not (key == "working_attention_scratch_budget_bytes" and value is None)
        }
        if payload.get("memory_recipe") in {"human", "human-consolidation"}:
            payload["memory_recipe"] = "adaptive-retention"
        return payload

    def generation_token_budget(self, cognitive_demand: float = 0.5) -> int:
        """Return a context-independent, state-scaled response budget.

        A larger working window must not make a CPU brain emit thousands of
        tokens on every turn.  The learned recurrent state supplies
        ``cognitive_demand``; hardware tier and the Extended checkbox set the
        base.  Explicit caller-provided generation limits still take
        precedence in :meth:`AdaptiveBrain.chat`.
        """

        base_by_tier = {
            "micro": 96,
            "personal": 192,
            "gpu": 320,
            "workstation": 512,
        }
        demand = max(0.0, min(1.0, float(cognitive_demand)))
        extended_scale = 1.5 if self.extended_working_memory else 1.0
        state_scale = 0.75 + 0.5 * demand
        resolved = round(
            base_by_tier.get(self.hardware_tier, 192)
            * extended_scale
            * state_scale
        )
        return max(1, min(int(self.max_seq_len), max(48, resolved)))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OmniConfig":
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        if values.get("memory_recipe") in {"human", "human-consolidation"}:
            values["memory_recipe"] = "adaptive-retention"
        if "memory_resident_items" not in values:
            values["memory_resident_items"] = min(
                int(values.get("working_memory_slots", cls.working_memory_slots)),
                int(cls.memory_resident_items),
            )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_external(
        cls, raw: Dict[str, Any], *, native_architecture: Optional[Mapping[str, Any]] = None,
    ) -> "OmniConfig":
        """Translate the desktop app's camelCase builder config.

        Stable v1 ignores beta personality sliders and cardinality ceilings.
        Hardware profiling determines the physical recurrent population and
        working workspace; structural assemblies remain resource-governed.

        Persisted/research snake-case dictionaries must use :meth:`from_dict`.
        Treating one of those dictionaries as a public Build request used to
        bypass the versioned hardware profile and could make the worker create
        an undisclosed architecture.  The public boundary therefore never
        switches parsers based on caller-controlled keys.
        """
        tier = str(raw.get("hardwareTier", "personal"))
        if any(key in raw for key in ("nativeArchitecture", "native_architecture", "nativeArchitectureSha256")):
            raise ValueError("native architecture must come from the trusted main command, not renderer config")
        descriptor = (validate_native_architecture(native_architecture)
                      if native_architecture is not None else None)
        if descriptor is not None and descriptor["hardwareTier"] != tier:
            raise ValueError("native descriptor and public hardware tier disagree")
        profiles = {
            "micro": {
                "dimensions": 32,
                "layers": 1,
                "sequence": 256,
                "workspace": 128,
                "router": 24,
                "image": 8,
                "audio": 64,
                "frames": 2,
                "channels": 8,
                "batch": 1,
                "accumulation": 8,
                "checkpointing": True,
            },
            "personal": {
                "dimensions": 64,
                "layers": 2,
                "sequence": 1024,
                "workspace": 256,
                "router": 64,
                "image": 16,
                "audio": 256,
                "frames": 4,
                "channels": 16,
                "batch": 2,
                "accumulation": 4,
                "checkpointing": True,
            },
            "gpu": {
                "dimensions": 96,
                "layers": 4,
                "sequence": 2048,
                "workspace": 512,
                "router": 96,
                "image": 32,
                "audio": 512,
                "frames": 6,
                "channels": 24,
                "batch": 2,
                "accumulation": 8,
                "checkpointing": True,
            },
            "workstation": {
                "dimensions": 128,
                "layers": 6,
                "sequence": 4096,
                "workspace": 1024,
                "router": 128,
                "image": 32,
                "audio": 1024,
                "frames": 8,
                "channels": 32,
                "batch": 1,
                "accumulation": 16,
                "checkpointing": True,
            },
        }
        if tier not in profiles:
            tier = "personal"
        profile = profiles[tier]
        dimensions = int(profile["dimensions"])
        heads = 4 if dimensions <= 64 else 8
        physical_neurons = int(profile["router"])
        external_rate = float(raw.get("learningRate", 0.14))
        system_ram_mode = str(raw.get("systemRamMode", "auto"))
        if system_ram_mode not in {"auto", "manual"}:
            raise ValueError("systemRamMode must be auto or manual")
        system_ram_share_percent = (
            float(raw.get("systemRamSharePercent", 0.0))
            if system_ram_mode == "manual"
            else 0.0
        )
        if system_ram_mode == "manual" and not (
            30.0 <= system_ram_share_percent <= 100.0
        ):
            raise ValueError(
                "manual systemRamSharePercent must be in [30, 100]"
            )
        working_memory_mode = str(
            raw.get(
                "workingMemoryMode",
                "extended" if raw.get("extendedWorkingMemory", False) else "auto",
            )
        )
        extended_working = working_memory_mode == "extended"
        context_tokens = int(profile["sequence"]) * (
            2 if extended_working else 1
        )
        # Stable v1 resource planning may provide the exact live/model-derived
        # active window. Older checkpoints omit it and retain the historical
        # tier/Extended behavior above.
        if "contextWindowTokens" in raw:
            context_tokens = int(raw["contextWindowTokens"])
        workspace_slots = int(profile["workspace"]) * (2 if extended_working else 1)
        # Stable-v1's resource planner is the only public source of a numeric
        # capacity. Old beta sliders lacked workingMemoryMode and remain
        # ignored. Large values are recurrent/paged items, not a dense
        # attention allocation.
        if "workingMemoryMode" in raw:
            workspace_slots = max(1, int(raw.get("workingMemorySlots", workspace_slots)))
        if descriptor is not None:
            resolved = descriptor["shape"]
            if "workingMemorySlots" in raw and int(raw["workingMemorySlots"]) != resolved["workingMemoryItems"]:
                raise ValueError("native descriptor and planned working-memory selection disagree")
            dimensions = int(resolved["dModel"])
            heads = int(resolved["nHeads"])
            physical_neurons = int(resolved["routerNeurons"])
            workspace_slots = int(resolved["workingMemoryItems"])
        values: Dict[str, Any] = {
            "name": str(raw.get("name", "New OmniCortex")),
            "d_model": dimensions,
            "idea_dim": dimensions,
            "n_heads": heads,
            "n_layers": int(profile["layers"]),
            "d_ff": dimensions * 3,
            "max_seq_len": context_tokens,
            "router_neurons": physical_neurons,
            "vsa_dim": max(128, dimensions * 4),
            "hardware_tier": tier,
            # Build always creates the native OmniCortex architecture.
            "origin_kind": "ground-up",
            "train_batch_size": int(profile["batch"]),
            "gradient_accumulation": int(profile["accumulation"]),
            "gradient_checkpointing": bool(profile["checkpointing"]),
            "training_resource_mode": str(
                raw.get("trainingResourceMode", "auto")
            ),
            "system_ram_share_percent": system_ram_share_percent,
            "training_ram_budget_bytes": max(
                0, int(raw.get("trainingRamBudgetBytes", 0))
            ),
            "training_accelerator_budget_bytes": max(
                0, int(raw.get("trainingAcceleratorBudgetBytes", 0))
            ),
            "training_scratch_budget_bytes": max(
                0, int(raw.get("trainingScratchBudgetBytes", 0))
            ),
            "storage_bytes_per_second": safe_rounded_storage_bytes_per_second(
                raw.get("storageBytesPerSecond", 0)
            ),
            "image_size": int(profile["image"]),
            "audio_samples": int(profile["audio"]),
            "video_frames": int(profile["frames"]),
            "modality_channels": int(profile["channels"]),
            "ternary_weights": True,
            "spiking_dynamics": True,
            "stdp_plasticity": True,
            "liquid_dynamics": True,
            "liquid_mode": str(raw.get("liquidMode", "cfc")),
            "vector_symbolic_memory": True,
            "online_learning": bool(raw.get("onlineLearning", True)),
            "consolidation_enabled": True,
            "metaplasticity": True,
            "vision_enabled": bool(raw.get("vision_enabled", True)),
            "image_enabled": bool(raw.get("image_enabled", True)),
            "audio_enabled": bool(raw.get("audio_enabled", True)),
            "video_enabled": bool(raw.get("video_enabled", True)),
            "learning_rate": max(1e-5, min(0.02, external_rate * 0.02)),
            "slow_stability_strength": max(
                0.0, min(10.0, float(raw.get("slowStabilityStrength", 0.025)))
            ),
            "slow_importance_decay": max(
                0.0, min(0.9999, float(raw.get("slowImportanceDecay", 0.97)))
            ),
            "memory_recipe": str(
                raw.get("memoryRecipe", "adaptive-retention")
            ),
            "memory_injection": "working-memory",
            "learn_from_own_messages": True,
            "retain_source_text": bool(raw.get("retainSourceText", False)),
            "extended_working_memory": extended_working,
            "recursive_improvement": bool(
                raw.get("recursiveImprovement", True)
            ),
            # Stable v1 always has the recurrent Ponder capability. Neural
            # state may select zero passes; config cannot remove the pathway.
            "idle_cognition": True,
            "working_memory_slots": workspace_slots,
            "working_memory_mode": working_memory_mode,
            "memory_offload_bytes": max(0, int(raw.get("memoryOffloadBytes", 0))),
            "memory_resident_items": max(
                1,
                min(
                    workspace_slots,
                    int(raw.get("memoryResidentItems", workspace_slots)),
                ),
            ),
            "memory_offload_slowdown_percent": max(
                0.0,
                min(95.0, float(raw.get("memoryOffloadSlowdownPercent", 0.0))),
            ),
            "device": str(raw.get("device", "cpu")),
            "ram_reserve_bytes": max(0, int(raw.get("ramReserveBytes", 0))),
            "disk_reserve_bytes": max(
                0, int(raw.get("diskReserveBytes", 0))
            ),
            "disk_state_offload": True,
            "storage_pool_bytes": max(0, int(raw.get("storagePoolBytes", 0))),
            "working_attention_scratch_budget_bytes": (
                max(0, int(raw["contextOffloadBudgetBytes"]))
                if "contextOffloadBudgetBytes" in raw else None
            ),
        }
        if values["memory_recipe"] in {"human", "human-consolidation"}:
            values["memory_recipe"] = "adaptive-retention"
        if descriptor is not None:
            resolved = descriptor["shape"]
            values.update({
                "native_architecture": descriptor,
                "n_layers": int(resolved["layers"]), "d_ff": int(resolved["feedForward"]),
                "vsa_dim": int(resolved["vsaDimensions"]),
                "image_size": int(resolved["imageSize"]), "audio_samples": int(resolved["audioSamples"]),
                "video_frames": int(resolved["videoFrames"]), "modality_channels": int(resolved["modalityChannels"]),
                "vocab_size": int(resolved["vocabSize"]), "liquid_mode": str(resolved["liquidMode"]),
            })
            if not values["storage_pool_bytes"]:
                values["storage_pool_bytes"] = max(0, int(descriptor["sizing"].get("selectedStoragePoolBytes", 0)))
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def micro(cls, name: str = "Micro OmniCortex", **overrides: Any) -> "OmniConfig":
        # This low-level constructor exists for unit tests and constrained
        # research fixtures. Studio builds use ``from_external``.
        values: Dict[str, Any] = {
            "name": name,
            "max_seq_len": 256,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 64,
            "idea_dim": 32,
            "vsa_dim": 64,
            "router_neurons": 24,
            "working_memory_slots": 128,
            "memory_resident_items": 128,
            "image_size": 8,
            "audio_samples": 64,
            "video_frames": 2,
            "modality_channels": 8,
            "hardware_tier": "micro",
            "train_batch_size": 1,
            "gradient_accumulation": 8,
            "gradient_checkpointing": True,
            "origin_kind": "ground-up",
        }
        values.update(overrides)
        config = cls(**values)
        config.validate()
        return config
