"""Leaky spiking dynamics and local STDP plasticity."""

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .model import (
    PackedAdaptiveBitLinear as BitLinear,
    pack_ternary_weight,
    unpack_ternary_weight_rows,
)
from .architecture_migration import copy_packed_prefix, copy_tensor_prefix
from .bounded_tensor_io import TRANSFER_BYTES


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

        self._validate_packed()
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
        if legacy_name in state_dict:
            # Loading is not authorization to threshold or quantize learned
            # state. Keep the incompatible archive intact and fail closed.
            error_msgs.append(prefix + "floating learned synapse weights are incompatible with packed-native state")
            return
        if packed_name not in state_dict:
            error_msgs.append(packed_name + " is required")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )
        if packed_name in state_dict:
            try:
                self._validate_packed()
            except (ValueError, RuntimeError) as error:
                error_msgs.append(str(error))

    def reset_activity(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def _validate_packed(self) -> None:
        """Bounded byte/code validation, without an N² decoded shadow."""
        packed = self._packed_weights
        if packed.dtype != torch.uint8 or not packed.is_contiguous() or packed.shape != (
            self.post_neurons, (self.pre_neurons + 3) // 4
        ):
            raise ValueError("packed STDP synapse shape or dtype is invalid")
        step = max(1, TRANSFER_BYTES // 8)
        flat = packed.reshape(-1)
        for start in range(0, flat.numel(), step):
            block = flat[start:start + step]
            if any(bool((((block >> shift) & 3) == 3).any()) for shift in (0, 2, 4, 6)):
                raise ValueError("packed STDP synapse contains reserved ternary codes")
        if self.pre_neurons % 4:
            for start in range(0, self.post_neurons, step):
                last = packed[start:start + step, -1]
                for lane in range(self.pre_neurons % 4, 4):
                    if bool((((last >> (2 * lane)) & 3) != 1).any()):
                        raise ValueError("packed STDP synapse has nonzero row padding")

    def effective_weight(self) -> torch.Tensor:
        """Ephemeral exact recurrent input, not a persistent FP weight master."""
        self._validate_packed()
        return unpack_ternary_weight_rows(self._packed_weights, self.pre_neurons)

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
        self.register_buffer("active_prefix_neurons", torch.tensor(int(neurons), dtype=torch.long))
        self.register_buffer("region_ends", torch.tensor([int(neurons)], dtype=torch.long))
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

    @torch.no_grad()
    def grow_neurons(self, add_neurons: int, *, region_sizes: Optional[Tuple[int, ...]] = None) -> int:
        """Preserve old trits/activity and add an initially dormant region."""
        if isinstance(add_neurons, bool) or not isinstance(add_neurons, int) or add_neurons < 1:
            raise ValueError("router additions must be a positive integer")
        old_neurons, new_neurons = int(self.neurons), int(self.neurons) + add_neurons
        if region_sizes is not None and (any(type(size) is not int or size < 1 for size in region_sizes) or sum(region_sizes) != add_neurons):
            raise ValueError("router region sizes must cover all appended neurons")
        pager = getattr(self.input_projection, "_native_core_pager", None)
        scope = pager.construction(from_checkpoint=False) if pager is not None else nullcontext()
        device = self.population.membrane.device
        with scope:
            incoming = BitLinear(self.idea_dim, new_neurons, bias=True)
            outgoing = BitLinear(new_neurons, self.idea_dim, bias=True)
            for old, new in ((self.input_projection, incoming), (self.output_projection, outgoing)):
                copy_packed_prefix(old._packed_forward_weight, new._packed_forward_weight, old.in_features, new.in_features)
                copy_packed_prefix(old._packed_forward_bias, new._packed_forward_bias, old.out_features, new.out_features)
                for name in ("_packed_forward_scale", "_online_learning_rate", "_row_stability", "_bias_row_stability"):
                    copy_tensor_prefix(getattr(old, name), getattr(new, name))
                new._packed_stability_strength = old._packed_stability_strength
                new._pending_stability_events = old._pending_stability_events
            population = LIFPopulation(new_neurons, leak=self.population.leak, threshold=self.population.threshold).to(device)
            copy_tensor_prefix(self.population.membrane, population.membrane)
            copy_tensor_prefix(self.population.spike_count, population.spike_count)
            old_synapses = self.synapses
            synapses = STDPSynapses(new_neurons, new_neurons,
                learning_rate=old_synapses.learning_rate,
                tau_pre=8.0, tau_post=8.0,
                a_plus=old_synapses.a_plus, a_minus=old_synapses.a_minus,
                metaplasticity_rate=old_synapses.metaplasticity_rate,
                weight_limit=old_synapses.weight_limit).to(device)
            synapses.pre_decay = old_synapses.pre_decay
            synapses.post_decay = old_synapses.post_decay
            copy_packed_prefix(old_synapses._packed_weights, synapses._packed_weights, old_neurons, new_neurons)
            for name in ("eligibility_accumulator", "stability", "uses", "pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
                copy_tensor_prefix(getattr(old_synapses, name), getattr(synapses, name))
            incoming.to(device)
            outgoing.to(device)
        ends = self.region_ends.detach().cpu().tolist()
        running = old_neurons
        for size in region_sizes or (add_neurons,):
            running += size
            ends.append(running)
        self.input_projection, self.output_projection = incoming, outgoing
        self.population, self.synapses = population, synapses
        self.neurons = new_neurons
        self.region_ends = torch.tensor(ends, dtype=torch.long, device=device)
        return new_neurons

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Legacy saved routers mean every existing neuron is active. New
        # control buffers never discard or reset their old nonweight state.
        state_dict = dict(state_dict)
        state_dict.setdefault(prefix + "active_prefix_neurons", torch.tensor(self.neurons, dtype=torch.long))
        ends = state_dict.get(prefix + "region_ends")
        if isinstance(ends, torch.Tensor):
            if ends.ndim != 1 or ends.dtype != torch.long or not ends.numel() or int(ends[0]) < 1 or int(ends[-1]) != self.neurons or bool((ends[1:] <= ends[:-1]).any()):
                error_msgs.append(prefix + "router region geometry is invalid")
            else:
                self.region_ends = torch.empty_like(ends, device=self.population.membrane.device)
        else:
            state_dict[prefix + "region_ends"] = torch.tensor([self.neurons], dtype=torch.long)
        active = state_dict[prefix + "active_prefix_neurons"]
        if active.numel() != 1 or active.dtype != torch.long or not 1 <= int(active) <= self.neurons:
            error_msgs.append(prefix + "router active-prefix geometry is invalid")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def prepare_bounded_state_load(self, specs, prefix: str) -> None:
        spec = specs.get(prefix + "region_ends")
        if spec is not None:
            if len(spec.shape) != 1 or spec.dtype != torch.long or not 1 <= spec.shape[0] <= self.neurons:
                raise ValueError("router region geometry header is invalid")
            self.region_ends = torch.empty(spec.shape, dtype=torch.long, device=self.population.membrane.device)
        active = specs.get(prefix + "active_prefix_neurons")
        if active is not None and (active.shape != () or active.dtype != torch.long):
            raise ValueError("router active-prefix header is invalid")

    def bounded_optional_state_defaults(self, specs, prefix: str):
        """Only the two additive control fields may be absent in legacy files."""
        return {
            name: value for name, value in (
                ("active_prefix_neurons", torch.tensor(self.neurons, dtype=torch.long)),
                ("region_ends", torch.tensor([self.neurons], dtype=torch.long)),
            ) if prefix + name not in specs
        }

    def validate_bounded_state_load(self) -> None:
        """Post-bounded-load control validation; learned bytes are unchanged."""
        active, ends = self.active_prefix_neurons, self.region_ends
        if active.ndim != 0 or active.dtype != torch.long or not 1 <= int(active) <= self.neurons:
            raise ValueError("router active-prefix geometry is invalid")
        if (ends.ndim != 1 or ends.dtype != torch.long or not 1 <= ends.numel() <= self.neurons
            or int(ends[0]) < 1 or int(ends[-1]) != self.neurons
            or bool((ends[1:] <= ends[:-1]).any())):
            raise ValueError("router region geometry is invalid")

    def _validate_packed(self) -> None:
        self.validate_bounded_state_load()

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

        if learn:
            self.active_prefix_neurons.fill_(self.neurons)
        active_neurons = int(self.active_prefix_neurons)

        projected = torch.sigmoid(self.input_projection(idea))[0]
        if active_neurons < self.neurons:
            projected = projected * (torch.arange(self.neurons, device=projected.device) < active_neurons)
        previous = torch.zeros_like(projected)
        total_spikes = torch.zeros_like(projected)
        total_update = 0.0
        for _ in range(max(1, int(steps))):
            recurrent = torch.mv(
                self.synapses.effective_weight().to(previous),
                previous.detach(),
            )
            current = projected + 0.35 * recurrent
            if active_neurons < self.neurons:
                current = current * (torch.arange(self.neurons, device=current.device) < active_neurons)
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
            "spike_rate": float((total_spikes.detach()[:active_neurons] > 0).float().mean().item()),
            "spikes": float(total_spikes.detach().sum().item()),
            "stdp_update": total_update,
            "mean_stability": float(self.synapses.stability[:active_neurons, :active_neurons].mean().item()),
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
        self.active_prefix_neurons.fill_(self.neurons)
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
