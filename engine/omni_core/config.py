"""Configuration and hardware-aware tiny defaults for OmniCortex."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class OmniConfig:
    """Serializable architecture and learning configuration.

    Defaults deliberately describe a tiny model so a blank brain can be
    created and tested on a CPU.  The desktop application may scale these
    values after profiling the Windows machine.
    """

    name: str = "New OmniCortex"
    seed: int = 7
    vocab_size: int = 261
    max_seq_len: int = 128
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
    noise: float = 0.08
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
    memory_injection: str = "parameter-only"
    learn_from_own_messages: bool = True
    retain_source_text: bool = False
    max_concepts: int = 50000
    max_ideas: int = 10000
    replay_capacity: int = 2048
    working_memory_slots: int = 24
    short_term_half_life_minutes: float = 45.0
    long_term_threshold: float = 0.62
    forgetting_rate: float = 0.002
    consolidation_rate: float = 0.06
    max_synapses: int = 1000000
    novelty_drive: float = 0.72
    coherence_drive: float = 0.88
    curiosity_drive: float = 0.58
    parallel_thoughts: int = 3

    growth_policy: str = "elastic"
    max_experts: int = 8
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
        if self.growth_policy not in {"fixed", "elastic", "unbounded"}:
            raise ValueError("unsupported growth_policy")
        if self.hardware_tier not in {"micro", "personal", "gpu", "workstation"}:
            raise ValueError("unsupported hardware_tier")
        if self.origin_kind not in {"blank", "starter"}:
            raise ValueError("origin_kind must be blank or starter")
        if self.image_size < 8 or self.image_size % 4:
            raise ValueError("image_size must be a multiple of four and at least 8")
        if self.video_frames < 2:
            raise ValueError("video_frames must be at least 2")
        if self.working_memory_slots < 1 or self.parallel_thoughts < 1:
            raise ValueError("memory slots and parallel thoughts must be positive")
        if self.train_batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("training batch size and accumulation must be positive")
        if self.short_term_half_life_minutes <= 0:
            raise ValueError("short-term half-life must be positive")
        if self.slow_stability_strength < 0:
            raise ValueError("slow_stability_strength cannot be negative")
        if not 0.0 <= self.slow_importance_decay < 1.0:
            raise ValueError("slow_importance_decay must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OmniConfig":
        allowed = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        config.validate()
        return config

    @classmethod
    def from_external(cls, raw: Dict[str, Any]) -> "OmniConfig":
        """Translate the desktop app's camelCase builder config.

        UI neuron budgets describe conceptual capacity, not a dense recurrent
        matrix.  The physical LIF population therefore scales sublinearly and
        remains bounded by the selected local hardware profile.
        """

        if any(key in raw for key in ("d_model", "n_layers", "vsa_dim")):
            return cls.from_dict(raw)
        neuron_budget = max(64, int(raw.get("initialNeuronBudget", 2048)))
        tier = str(raw.get("hardwareTier", "personal"))
        profiles = {
            "micro": {
                "dimensions": 32,
                "layers": 1,
                "sequence": 64,
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
                "sequence": 128,
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
                "sequence": 256,
                "image": 32,
                "audio": 512,
                "frames": 6,
                "channels": 24,
                "batch": 8,
                "accumulation": 2,
                "checkpointing": True,
            },
            "workstation": {
                "dimensions": 128,
                "layers": 6,
                "sequence": 512,
                "image": 32,
                "audio": 1024,
                "frames": 8,
                "channels": 32,
                "batch": 16,
                "accumulation": 1,
                "checkpointing": False,
            },
        }
        if tier not in profiles:
            tier = "personal"
        profile = profiles[tier]
        dimensions = int(profile["dimensions"])
        heads = 4 if dimensions <= 64 else 8
        physical_neurons = max(24, min(128, int(neuron_budget ** 0.5 * 1.4)))
        external_rate = float(raw.get("learningRate", 0.14))
        values: Dict[str, Any] = {
            "name": str(raw.get("name", "New OmniCortex")),
            "d_model": dimensions,
            "idea_dim": dimensions,
            "n_heads": heads,
            "n_layers": int(profile["layers"]),
            "d_ff": dimensions * 3,
            "max_seq_len": int(profile["sequence"]),
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
            "ternary_weights": bool(raw.get("ternaryWeights", True)),
            "spiking_dynamics": bool(raw.get("spikingDynamics", True)),
            "stdp_plasticity": bool(raw.get("stdpPlasticity", True)),
            "liquid_dynamics": bool(raw.get("liquidDynamics", True)),
            "liquid_mode": str(raw.get("liquidMode", "cfc")),
            "vector_symbolic_memory": bool(
                raw.get("vectorSymbolicMemory", True)
            ),
            "online_learning": bool(raw.get("onlineLearning", True)),
            "consolidation_enabled": bool(raw.get("consolidation", True)),
            "metaplasticity": bool(raw.get("metaplasticity", True)),
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
            "noise": max(0.0, min(1.0, float(raw.get("noise", 0.08)))),
            "firing_threshold": max(
                0.05, min(2.0, float(raw.get("firingThreshold", 0.55)))
            ),
            "membrane_leak": max(
                0.0, min(0.999, float(raw.get("membraneLeak", 0.88)))
            ),
            "stdp_tau_pre": max(1.0, float(raw.get("stdpWindow", 8))),
            "stdp_tau_post": max(1.0, float(raw.get("stdpWindow", 8))),
            "memory_recipe": str(
                raw.get("memoryRecipe", "human-consolidation")
            ),
            "memory_injection": str(
                raw.get("memoryInjection", "parameter-only")
            ),
            "learn_from_own_messages": bool(
                raw.get("learnFromOwnMessages", True)
            ),
            "retain_source_text": bool(raw.get("retainSourceText", False)),
            "max_concepts": int(raw.get("maxConcepts", 50000)),
            "max_synapses": int(raw.get("maxSynapses", 1000000)),
            "growth_policy": str(raw.get("growthPolicy", "elastic")),
            "working_memory_slots": int(raw.get("workingMemorySlots", 24)),
            "short_term_half_life_minutes": float(
                raw.get("shortTermHalfLifeMinutes", 45)
            ),
            "long_term_threshold": float(raw.get("longTermThreshold", 0.62)),
            "forgetting_rate": float(raw.get("forgettingRate", 0.002)),
            "consolidation_rate": float(raw.get("consolidationRate", 0.06)),
            "novelty_drive": float(raw.get("noveltyDrive", 0.72)),
            "coherence_drive": float(raw.get("coherenceDrive", 0.88)),
            "curiosity_drive": float(raw.get("curiosityDrive", 0.58)),
            "parallel_thoughts": int(raw.get("parallelThoughts", 3)),
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
            "max_seq_len": 48,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 64,
            "idea_dim": 32,
            "vsa_dim": 64,
            "router_neurons": 24,
            "image_size": 8,
            "audio_samples": 64,
            "video_frames": 2,
            "modality_channels": 8,
            "max_experts": 3,
            "hardware_tier": "micro",
            "train_batch_size": 1,
            "gradient_accumulation": 8,
            "gradient_checkpointing": True,
        }
        values.update(overrides)
        config = cls(**values)
        config.validate()
        return config
