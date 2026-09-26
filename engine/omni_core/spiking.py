"""Leaky spiking dynamics and local STDP plasticity."""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .model import (
    PackedAdaptiveBitLinear as BitLinear,
    pack_ternary_weight,
    unpack_ternary_weight_rows,
)


class LIFPopulation(nn.Module):
    """Stateful leaky-integrate-and-fire neuron population."""

    def __init__(self, neurons: int, leak: float = 0.88, threshold: float = 0.55):
        super().__init__()
        if not 0.0 <= leak < 1.0:
            raise ValueError("leak must be in [0, 1)")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.neurons = neurons
        self.leak = float(leak)
        self.threshold = float(threshold)
        self.register_buffer("membrane", torch.zeros(neurons))
        self.register_buffer("spike_count", torch.zeros(neurons))

    def reset(self) -> None:
        self.membrane.zero_()
        self.spike_count.zero_()

    def step(
        self, current: torch.Tensor, threshold_offset: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current.ndim == 2:
            if current.shape[0] != 1:
                raise ValueError("persistent LIF state currently supports batch size one")
            current = current[0]
        if current.shape[-1] != self.neurons:
            raise ValueError("current has the wrong neuron dimension")
        threshold = max(0.05, self.threshold + float(threshold_offset))
        self.membrane.mul_(self.leak).add_(current.detach())
        spikes = (self.membrane >= threshold).to(current.dtype)
        self.membrane.sub_(spikes * threshold)
        self.spike_count.add_(spikes)
        # Preserve a differentiable surrogate around the hard threshold.
        soft_spike = torch.sigmoid((current - threshold) * 8.0)
        surrogate = spikes + (soft_spike - soft_spike.detach())
        return surrogate, self.membrane.clone()


class STDPSynapses(nn.Module):
    """Pair-based causal/anti-causal STDP with metaplastic stability.

    The durable connection strength is a packed exact ternary code, never a
    floating shadow weight. Spike timing accumulates only bounded int16
    eligibility before moving a connection by one ternary level. The decoded
    ``weights`` property is an inspection copy, not mutable neural storage.
    """

    def __init__(
        self,
        pre_neurons: int,
        post_neurons: int,
        learning_rate: float = 0.035,
        tau_pre: float = 8.0,
        tau_post: float = 8.0,
        a_plus: float = 1.0,
        a_minus: float = 1.05,
        metaplasticity_rate: float = 0.025,
        weight_limit: float = 1.0,
    ):
        super().__init__()
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.learning_rate = float(learning_rate)
        self.pre_decay = math.exp(-1.0 / max(float(tau_pre), 1e-3))
        self.post_decay = math.exp(-1.0 / max(float(tau_post), 1e-3))
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.metaplasticity_rate = float(metaplasticity_rate)
        self.weight_limit = float(weight_limit)
        self.ternary = True
        if pre_neurons <= 0 or post_neurons <= 0:
            raise ValueError("synapse population dimensions must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("STDP learning rate must be finite and nonnegative")
        self.register_buffer(
            "_packed_weights",
            torch.full(
                (post_neurons, (pre_neurons + 3) // 4),
                0x55,
                dtype=torch.uint8,
            ),
        )
        # Eligibility is subthreshold timing pressure, not a second weight.
        # Its fixed-point range is [-255, 255] after each update.
        self.register_buffer(
            "eligibility_accumulator",
            torch.zeros(post_neurons, pre_neurons, dtype=torch.int16),
        )
        self.register_buffer("stability", torch.zeros(post_neurons, pre_neurons))
        self.register_buffer("pre_trace", torch.zeros(pre_neurons))
        self.register_buffer("post_trace", torch.zeros(post_neurons))
        self.register_buffer("uses", torch.zeros(post_neurons, pre_neurons))
        self.register_buffer("plasticity_events", torch.zeros((), dtype=torch.long))
        self.register_buffer("decay_cycles", torch.zeros((), dtype=torch.long))

    @property
    def weights(self) -> torch.Tensor:
        """Read-only decoded inspection copy for existing trace consumers."""

        return self.effective_weight()

    def authoritative_packed_tensors(self) -> Tuple[torch.Tensor, ...]:
        """The sole persistent synaptic weight bytes for checksum/accounting."""

        self.effective_weight()  # Validate reserved and padding codes.
        return (self._packed_weights,)

    @property
    def logical_ternary_parameter_count(self) -> int:
        return int(self.pre_neurons * self.post_neurons)

    @torch.no_grad()
    def set_effective_weights(self, levels: torch.Tensor) -> None:
        """Replace exact levels for controlled initialization or inspection."""

        if levels.shape != (self.post_neurons, self.pre_neurons):
            raise ValueError("ternary synapse shape is invalid")
        if levels.dtype != torch.int8 or not bool(
            ((levels >= -1) & (levels <= 1)).all()
        ):
            raise ValueError("synapse levels must be exact int8 ternary values")
        self._packed_weights.copy_(
            pack_ternary_weight(levels.to(self._packed_weights.device))
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        legacy_name = prefix + "weights"
        packed_name = prefix + "_packed_weights"
        if legacy_name in state_dict and packed_name in state_dict:
            error_msgs.append(
                prefix + "synapse checkpoint has both packed and floating weights"
            )
        elif legacy_name in state_dict:
            legacy = state_dict.pop(legacy_name)
            if (
                not isinstance(legacy, torch.Tensor)
                or legacy.shape != (self.post_neurons, self.pre_neurons)
                or not legacy.is_floating_point()
                or not bool(torch.isfinite(legacy).all())
            ):
                error_msgs.append(prefix + "legacy STDP weights are invalid")
            else:
                # Native pre-packed checkpoints used this threshold for their
                # effective recurrent synapses. Preserve those exact levels;
                # discard the former dense float master after migration.
                threshold = max(1e-6, self.weight_limit * 0.25)
                levels = torch.where(
                    legacy >= threshold,
                    torch.ones_like(legacy, dtype=torch.int8),
                    torch.where(
                        legacy <= -threshold,
                        -torch.ones_like(legacy, dtype=torch.int8),
                        torch.zeros_like(legacy, dtype=torch.int8),
                    ),
                )
                state_dict[packed_name] = pack_ternary_weight(levels)
                state_dict[prefix + "eligibility_accumulator"] = (
                    torch.zeros_like(self.eligibility_accumulator)
                )
                state_dict[prefix + "decay_cycles"] = torch.zeros_like(
                    self.decay_cycles
                )
        if packed_name not in state_dict:
            error_msgs.append(packed_name + " is required")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )
        if packed_name in state_dict:
            try:
                self.effective_weight()
            except (ValueError, RuntimeError) as error:
                error_msgs.append(str(error))

    def reset_activity(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def effective_weight(self) -> torch.Tensor:
        """Return the exact ternary synapses used by recurrent computation."""

        packed = self._packed_weights
        if packed.dtype != torch.uint8 or packed.shape != (
            self.post_neurons, (self.pre_neurons + 3) // 4
        ):
            raise ValueError("packed STDP synapse shape or dtype is invalid")
        decoded = unpack_ternary_weight_rows(packed, self.pre_neurons)
        # The generic decoder verifies active codes. Enforce canonical zero
        # padding too, so a malformed checkpoint never enters recurrent use.
        if self.pre_neurons % 4:
            last = packed[:, -1]
            for lane in range(self.pre_neurons % 4, 4):
                if bool((((last >> (2 * lane)) & 0x03) != 1).any()):
                    raise ValueError("packed STDP synapse has nonzero row padding")
        return decoded

    def step(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> torch.Tensor:
        pre = pre_spikes.detach().reshape(-1).to(self.pre_trace)
        post = post_spikes.detach().reshape(-1).to(self.post_trace)
        if pre.numel() != self.pre_neurons or post.numel() != self.post_neurons:
            raise ValueError("spike vector dimensions do not match synapses")

        potentiation = self.a_plus * torch.outer(post, self.pre_trace)
        depression = self.a_minus * torch.outer(self.post_trace, pre)
        timing_signal = potentiation - depression
        local_rate = self.learning_rate / (1.0 + self.stability)
        delta = local_rate * timing_signal

        active = timing_signal.ne(0)
        if bool(active.any()):
            previous = self.effective_weight()
            previous_direction = torch.sign(previous)
            update_direction = torch.sign(delta)
            agreement = (previous_direction == update_direction) | (
                previous_direction == 0
            )
            stability_delta = torch.where(
                agreement,
                torch.full_like(self.stability, self.metaplasticity_rate),
                torch.full_like(self.stability, -self.metaplasticity_rate * 0.25),
            )
            self.stability.add_(stability_delta * active).clamp_(0.0, 20.0)
            self.uses.add_(active.to(self.uses))
            # Normalize to one transition per ordinary causal/anti-causal
            # event. We retain fractional timing pressure in a bounded int16
            # trace instead of retaining a full floating synaptic master.
            quantum = max(self.learning_rate, 1e-6)
            increments = (
                (delta / quantum * 256.0)
                .round()
                .clamp(-256, 256)
                .to(torch.int32)
            )
            pressure = (
                self.eligibility_accumulator.to(torch.int32) + increments
            ).clamp(-256, 256)
            transition = torch.where(
                pressure >= 256,
                torch.ones_like(pressure),
                torch.where(
                    pressure <= -256,
                    -torch.ones_like(pressure),
                    torch.zeros_like(pressure),
                ),
            )
            next_levels = (previous.to(torch.int32) + transition).clamp(-1, 1)
            pressure = pressure - transition * 256
            pressure = torch.where(
                next_levels == previous.to(torch.int32),
                torch.zeros_like(pressure),
                pressure,
            )
            self.eligibility_accumulator.copy_(pressure.to(torch.int16))
            if bool((next_levels != previous.to(torch.int32)).any()):
                self.set_effective_weights(next_levels.to(torch.int8))
            self.plasticity_events.add_(int(active.sum().item()))

        self.pre_trace.mul_(self.pre_decay).add_(pre)
        self.post_trace.mul_(self.post_decay).add_(post)
        return delta

    def decay_unused(self, amount: float = 1e-4) -> None:
        amount = max(0.0, min(float(amount), 1.0))
        self.decay_cycles.add_(1)
        if amount:
            # Persistent cycle count gives reproducible, rare discrete decay
            # without a dense floating weight or decay residual. Frequently
            # used links decay more slowly, as in the former continuous rule.
            levels = self.effective_weight()
            positions = torch.arange(
                levels.numel(), device=levels.device, dtype=torch.int64
            ).reshape_as(levels)
            tick = int(self.decay_cycles.item())
            random_codes = (
                positions * 1664525 + tick * 1013904223
            ) & 0xFFFFFFFF
            # MPS has no float64 arithmetic; float32 still resolves the small
            # per-cycle probabilities used for gradual pathway decay.
            draw = random_codes.to(torch.float32) / 4294967296.0
            decay = (levels != 0) & (
                draw < (amount / (1.0 + self.uses)).to(draw)
            )
            if bool(decay.any()):
                next_levels = torch.where(
                    decay, torch.zeros_like(levels), levels
                )
                self.set_effective_weights(next_levels)
        self.stability.mul_(1.0 - amount * 0.1)


class AssociativeSpikingRouter(nn.Module):
    """Maps idea activity through a plastic recurrent LIF population."""

    def __init__(
        self,
        idea_dim: int,
        neurons: int,
        leak: float = 0.88,
        threshold: float = 0.55,
        learning_rate: float = 0.035,
        tau_pre: float = 8.0,
        tau_post: float = 8.0,
        a_plus: float = 1.0,
        a_minus: float = 1.05,
        metaplasticity_rate: float = 0.025,
    ):
        super().__init__()
        self.idea_dim = idea_dim
        self.neurons = neurons
        self.input_projection = BitLinear(idea_dim, neurons, bias=True)
        self.output_projection = BitLinear(neurons, idea_dim, bias=True)
        self.population = LIFPopulation(neurons, leak=leak, threshold=threshold)
        self.synapses = STDPSynapses(
            neurons,
            neurons,
            learning_rate=learning_rate,
            tau_pre=tau_pre,
            tau_post=tau_post,
            a_plus=a_plus,
            a_minus=a_minus,
            metaplasticity_rate=metaplasticity_rate,
        )

    def reset_activity(self) -> None:
        self.population.reset()
        self.synapses.reset_activity()

    def route(
        self,
        idea: torch.Tensor,
        steps: int = 4,
        learn: bool = True,
        threshold_offset: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if idea.ndim == 1:
            idea = idea.unsqueeze(0)
        if idea.shape[0] != 1 or idea.shape[-1] != self.idea_dim:
            raise ValueError("router expects one idea vector")

        projected = torch.sigmoid(self.input_projection(idea))[0]
        previous = torch.zeros_like(projected)
        total_spikes = torch.zeros_like(projected)
        total_update = 0.0
        for _ in range(max(1, int(steps))):
            recurrent = torch.mv(
                self.synapses.effective_weight().to(previous),
                previous.detach(),
            )
            current = projected + 0.35 * recurrent
            spikes, _ = self.population.step(
                current, threshold_offset=threshold_offset
            )
            hard_spikes = (spikes.detach() > 0.5).to(spikes)
            if learn:
                delta = self.synapses.step(previous.detach(), hard_spikes)
                total_update += float(delta.abs().sum().item())
            previous = hard_spikes
            total_spikes = total_spikes + spikes

        activity = total_spikes / float(max(1, int(steps)))
        routed = idea + 0.25 * torch.tanh(
            self.output_projection(activity.unsqueeze(0))
        )
        metrics = {
            "spike_rate": float((total_spikes.detach() > 0).float().mean().item()),
            "spikes": float(total_spikes.detach().sum().item()),
            "stdp_update": total_update,
            "mean_stability": float(self.synapses.stability.mean().item()),
            "active_synapses": float(
                self.synapses.effective_weight().ne(0).sum().item()
            ),
        }
        return routed, metrics

    @torch.no_grad()
    def apply_feedback(
        self, idea: torch.Tensor, direction: int
    ) -> Dict[str, float]:
        """Apply causal or anti-causal local timing to the active assembly.

        This is direct synaptic plasticity, not a scalar reward model. Positive
        feedback replays presynaptic activity before its associated
        postsynaptic activity; negative feedback reverses that timing so the
        same pair-based STDP rule depresses the association.
        """

        if direction not in {-1, 1}:
            raise ValueError("feedback direction must be -1 or +1")
        if idea.ndim == 1:
            idea = idea.unsqueeze(0)
        if idea.shape != (1, self.idea_dim):
            raise ValueError("feedback expects one idea vector")
        projected = torch.sigmoid(self.input_projection(idea))[0]
        threshold = projected.mean()
        pre = (projected >= threshold).to(projected)
        # A deterministic permutation creates associative pre/post pairs while
        # avoiding a self-connection-only update.
        post = torch.roll(pre, shifts=1)
        silence = torch.zeros_like(pre)
        self.synapses.reset_activity()
        before = self.synapses.weights.detach().clone()
        if direction > 0:
            self.synapses.step(pre, silence)
            first_delta = self.synapses.step(silence, post)
        else:
            self.synapses.step(silence, post)
            first_delta = self.synapses.step(pre, silence)
        changed = self.synapses.weights.detach() - before
        return {
            "stdp_update": float(changed.abs().sum().item()),
            "signed_update": float(changed.sum().item()),
            "timing_signal": float(first_delta.abs().sum().item()),
            "active_pairs": float(changed.ne(0).sum().item()),
            "mean_stability": float(self.synapses.stability.mean().item()),
            "plasticity_events": float(
                self.synapses.plasticity_events.item()
            ),
        }
