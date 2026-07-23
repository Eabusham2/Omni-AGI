"""Leaky spiking dynamics and local STDP plasticity."""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .model import BitLinear


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

    ``weights[post, pre]`` is potentiated when a presynaptic spike precedes a
    postsynaptic spike and depressed when the order is reversed.
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
        self.register_buffer("weights", torch.zeros(post_neurons, pre_neurons))
        self.register_buffer("stability", torch.zeros(post_neurons, pre_neurons))
        self.register_buffer("pre_trace", torch.zeros(pre_neurons))
        self.register_buffer("post_trace", torch.zeros(post_neurons))
        self.register_buffer("uses", torch.zeros(post_neurons, pre_neurons))
        self.register_buffer("plasticity_events", torch.zeros((), dtype=torch.long))

    def reset_activity(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def step(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> torch.Tensor:
        pre = pre_spikes.detach().reshape(-1).to(self.weights)
        post = post_spikes.detach().reshape(-1).to(self.weights)
        if pre.numel() != self.pre_neurons or post.numel() != self.post_neurons:
            raise ValueError("spike vector dimensions do not match synapses")

        potentiation = self.a_plus * torch.outer(post, self.pre_trace)
        depression = self.a_minus * torch.outer(self.post_trace, pre)
        timing_signal = potentiation - depression
        local_rate = self.learning_rate / (1.0 + self.stability)
        delta = local_rate * timing_signal

        active = timing_signal.ne(0)
        if bool(active.any()):
            previous_direction = torch.sign(self.weights)
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
            self.weights.add_(delta).clamp_(-self.weight_limit, self.weight_limit)
            self.plasticity_events.add_(int(active.sum().item()))

        self.pre_trace.mul_(self.pre_decay).add_(pre)
        self.post_trace.mul_(self.post_decay).add_(post)
        return delta

    def decay_unused(self, amount: float = 1e-4) -> None:
        amount = max(0.0, min(float(amount), 1.0))
        use_scale = 1.0 / (1.0 + self.uses)
        self.weights.mul_(1.0 - amount * use_scale)
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
            recurrent = torch.mv(self.synapses.weights, previous.detach())
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
            "active_synapses": float(self.synapses.weights.ne(0).sum().item()),
        }
        return routed, metrics
