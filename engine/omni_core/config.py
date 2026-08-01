"""Configuration and hardware-aware tiny defaults for OmniCortex."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


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
    origin_kind: str = "blank"
    train_batch_size: int = 2
    gradient_accumulation: int = 2
    gradient_checkpointing: bool = False

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
    memory_recipe: str = "human-consolidation"
    memory_injection: str = "working-memory"
    learn_from_own_messages: bool = True
    retain_source_text: bool = False
    extended_working_memory: bool = False
    recursive_improvement: bool = True
    idle_cognition: bool = True
    replay_capacity: int = 2048
    working_memory_slots: int = 256
    short_term_half_life_minutes: float = 45.0
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

    def validate(self) -> None:
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
        if self.origin_kind not in {"blank", "starter"}:
            raise ValueError("origin_kind must be blank or starter")
        if self.image_size < 8 or self.image_size % 4:
            raise ValueError("image_size must be a multiple of four and at least 8")
        if self.video_frames < 2:
            raise ValueError("video_frames must be at least 2")
        if self.working_memory_slots < 1:
            raise ValueError("working-memory slots must be positive")
        if self.train_batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("training batch size and accumulation must be positive")
        if self.short_term_half_life_minutes <= 0:
            raise ValueError("short-term half-life must be positive")
        if self.slow_stability_strength < 0:
            raise ValueError("slow_stability_strength cannot be negative")
        if not 0.0 <= self.slow_importance_decay < 1.0:
            raise ValueError("slow_importance_decay must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in _DEPRECATED_BETA_CONTROL_FIELDS
        }

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
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        config.validate()
        return config

    @classmethod
    def from_external(cls, raw: Dict[str, Any]) -> "OmniConfig":
        """Translate the desktop app's camelCase builder config.

        Stable v1 ignores beta personality sliders and cardinality ceilings.
        Hardware profiling determines the physical recurrent population and
        working workspace; structural assemblies remain resource-governed.
        """

        if any(key in raw for key in ("d_model", "n_layers", "vsa_dim")):
            return cls.from_dict(raw)
        tier = str(raw.get("hardwareTier", "personal"))
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
        extended_working = bool(raw.get("extendedWorkingMemory", False))
        context_tokens = int(profile["sequence"]) * (
            2 if extended_working else 1
        )
        workspace_slots = int(profile["workspace"]) * (
            2 if extended_working else 1
        )
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
            "origin_kind": str(raw.get("origin_kind", raw.get("origin", "blank"))),
            "train_batch_size": int(profile["batch"]),
            "gradient_accumulation": int(profile["accumulation"]),
            "gradient_checkpointing": bool(profile["checkpointing"]),
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
                raw.get("memoryRecipe", "human-consolidation")
            ),
            "memory_injection": "working-memory",
            "learn_from_own_messages": True,
            "retain_source_text": bool(raw.get("retainSourceText", False)),
            "extended_working_memory": extended_working,
            "recursive_improvement": bool(
                raw.get("recursiveImprovement", True)
            ),
            "idle_cognition": bool(raw.get("idleCognition", True)),
            "working_memory_slots": workspace_slots,
            "device": str(raw.get("device", "cpu")),
        }
        if values["memory_recipe"] == "human":
            values["memory_recipe"] = "human-consolidation"
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def micro(cls, name: str = "Micro OmniCortex", **overrides: Any) -> "OmniConfig":
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
            "image_size": 8,
            "audio_samples": 64,
            "video_frames": 2,
            "modality_channels": 8,
            "hardware_tier": "micro",
            "train_batch_size": 1,
            "gradient_accumulation": 8,
            "gradient_checkpointing": True,
        }
        values.update(overrides)
        config = cls(**values)
        config.validate()
        return config
