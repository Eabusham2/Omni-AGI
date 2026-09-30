"""Leaky spiking dynamics and local STDP plasticity."""

import hashlib
import math
from contextlib import nullcontext
from dataclasses import dataclass
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
from .router_state_paging import (
    MATRIX_FIELDS, RouterMutationJournal, current_router_state_construction,
    iter_row_major_ternary_chunks, pack_tile, release_router_tensor_chunk,
    tile_ranges, unpack_tile, update_historical_tensor_checksum,
)


@dataclass(frozen=True)
class STDPUpdateSummary:
    """Bounded reductions, never an implicit dense N×N derivative tensor."""
    delta_absolute_sum: float = 0.0
    delta_signed_sum: float = 0.0
    changed_absolute_sum: int = 0
    changed_signed_sum: int = 0
    changed_pairs: int = 0
    active_pairs: int = 0


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
        construction = current_router_state_construction()
        self._router_state_pager = construction[0] if construction is not None else None
        self._router_loading_checkpoint = bool(construction is not None and construction[1])
        if pre_neurons <= 0 or post_neurons <= 0:
            raise ValueError("synapse population dimensions must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("STDP learning rate must be finite and nonnegative")
        self.register_buffer(
            "_packed_weights",
            self._router_buffer("_packed_weights", (post_neurons, (pre_neurons + 3) // 4), torch.uint8, 0x55),
        )
        # Eligibility is subthreshold timing pressure, not a second weight.
        # Its fixed-point range is [-255, 255] after each update.
        self.register_buffer(
            "eligibility_accumulator",
            self._router_buffer("eligibility_accumulator", (post_neurons, pre_neurons), torch.int16, 0),
        )
        self.register_buffer("stability", self._router_buffer("stability", (post_neurons, pre_neurons), torch.float32, 0))
        self.register_buffer("pre_trace", torch.zeros(pre_neurons))
        self.register_buffer("post_trace", torch.zeros(post_neurons))
        self.register_buffer("uses", self._router_buffer("uses", (post_neurons, pre_neurons), torch.float32, 0))
        self.register_buffer("plasticity_events", torch.zeros((), dtype=torch.long))
        self.register_buffer("decay_cycles", torch.zeros((), dtype=torch.long))

    def _router_buffer(self, name, shape, dtype, fill):
        pager = getattr(self, "_router_state_pager", None)
        if pager is not None:
            return pager.allocate(self, name, shape, dtype, fill=fill, loading=self._router_loading_checkpoint)
        return torch.full(shape, fill, dtype=dtype)

    def _apply(self, fn, recurse=True):
        if getattr(self, "_router_state_pager", None) is None:
            return super()._apply(fn, recurse=recurse)
        matrices = {name: self._buffers[name] for name in MATRIX_FIELDS}
        try:
            self._buffers.update({name: None for name in matrices})
            return super()._apply(fn, recurse=recurse)
        finally:
            self._buffers.update(matrices)

    def _tile_budget(self):
        pager = getattr(self, "_router_state_pager", None)
        return pager.tile_bytes if pager is not None else TRANSFER_BYTES

    def _operation(self, *, validating=False):
        pager = getattr(self, "_router_state_pager", None)
        return pager.operation(self, validating=validating) if pager is not None else nullcontext()

    def _ram(self, amount, operation):
        pager = getattr(self, "_router_state_pager", None)
        return pager.ram(amount, operation) if pager is not None else nullcontext()

    def _check(self):
        pager = getattr(self, "_router_state_pager", None)
        if pager is not None:
            pager.check()

    def _release_tile(self, r0, r1, c0, c1):
        pager = getattr(self, "_router_state_pager", None)
        if pager is not None:
            pager.release_tile(self, r0, r1, c0, c1)

    def validate_bounded_state_load(self):
        with self._operation(validating=True):
            self._validate_packed()
            for name, length in (("pre_trace", self.pre_neurons), ("post_trace", self.post_neurons)):
                trace = getattr(self, name)
                if trace.shape != (length,) or trace.dtype != torch.float32 or not bool(torch.isfinite(trace).all()) or bool((trace < 0).any()):
                    raise ValueError("router timing vector is invalid")
            for name in ("plasticity_events", "decay_cycles"):
                counter = getattr(self, name)
                if counter.shape != () or counter.dtype != torch.long or int(counter) < 0:
                    raise ValueError("router timing counter is invalid")
            for r0, r1, c0, c1 in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                with self._ram((r1-r0)*(c1-c0)*32, "router control validation"):
                    eligibility = self.eligibility_accumulator[r0:r1, c0:c1]
                    stability, uses = self.stability[r0:r1, c0:c1], self.uses[r0:r1, c0:c1]
                    if bool((eligibility.to(torch.int32).abs() > 255).any()) or not bool(torch.isfinite(stability).all()) or not bool(torch.isfinite(uses).all()) or bool(((stability < 0) | (stability > 20) | (uses < 0)).any()):
                        raise ValueError("router timing/stability/usage controls are invalid")
                self._release_tile(r0, r1, c0, c1)
        pager = getattr(self, "_router_state_pager", None)
        if pager is not None:
            pager.finish_owner_load(self)

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
        if levels.dtype != torch.int8:
            raise ValueError("synapse levels must be exact int8 ternary values")
        with self._operation():
            pager = getattr(self, "_router_state_pager", None)
            journal = RouterMutationJournal(self, pager, ram_bytes=pager.journal_ram_bytes if pager is not None else TRANSFER_BYTES)
            try:
                for r0, r1, c0, c1 in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                    self._check()
                    with self._ram((r1-r0)*(c1-c0)*16, "router explicit ternary replacement tile"):
                        block = levels[r0:r1, c0:c1].detach().to(device="cpu")
                        if not bool(((block >= -1) & (block <= 1)).all()):
                            raise ValueError("synapse levels must be exact int8 ternary values")
                        replacement = pack_tile(block)
                        target = self._packed_weights[r0:r1, c0//4:(c1+3)//4]
                        if not torch.equal(target.detach().cpu(), replacement):
                            journal.capture("_packed_weights", r0, r1, c0//4, (c1+3)//4)
                            target.copy_(replacement.to(self._packed_weights.device))
                    self._release_tile(r0, r1, c0, c1)
            except BaseException:
                journal.rollback()
                raise
            finally:
                journal.close()

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
        step = max(1, min(TRANSFER_BYTES, self._tile_budget()) // 8)
        flat = packed.reshape(-1)
        for start in range(0, flat.numel(), step):
            block = flat[start:start + step]
            with self._ram(block.numel() * 8, "router packed-code validation"):
                if any(bool((((block >> shift) & 3) == 3).any()) for shift in (0, 2, 4, 6)):
                    raise ValueError("packed STDP synapse contains reserved ternary codes")
            release_router_tensor_chunk(packed, start, block.numel())
        if self.pre_neurons % 4:
            for region in tile_ranges(self.post_neurons, 4, self._tile_budget()):
                start, end = region[:2]
                last = packed[start:end, -1]
                with self._ram(last.numel() * 8, "router packed-padding validation"):
                    for lane in range(self.pre_neurons % 4, 4):
                        if bool((((last >> (2 * lane)) & 3) != 1).any()):
                            raise ValueError("packed STDP synapse has nonzero row padding")
                for row in range(start, end):
                    release_router_tensor_chunk(packed, row * packed.shape[1] + packed.shape[1] - 1, 1)

    def effective_weight(self) -> torch.Tensor:
        """Explicit admitted small inspection copy, never the recurrence path."""
        needed = self.pre_neurons * self.post_neurons
        if needed > self._tile_budget():
            raise ValueError("full router inspection exceeds one tile; use iter_effective_tiles/recurrent/count helpers")
        with self._operation(), self._ram(needed * 8, "explicit router inspection"):
            self._validate_packed()
            return unpack_ternary_weight_rows(self._packed_weights, self.pre_neurons)

    def iter_effective_tiles(self):
        with self._operation():
            for region in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                self._check()
                try:
                    with self._ram((region[1]-region[0])*(region[3]-region[2])*16, "router signed tile"):
                        yield region, unpack_tile(self._packed_weights, *region)
                finally:
                    self._release_tile(*region)

    def iter_decoded_weight_chunks(self):
        """Bounded row-major int8 view matching the historical inspection dtype."""
        yield from iter_row_major_ternary_chunks(self)

    def _historical_checksum(self, *, controls: bool) -> str:
        with self._operation():
            digest = hashlib.sha256()
            digest.update(str((self.post_neurons, self.pre_neurons)).encode("ascii"))
            digest.update(str(torch.int8).encode("ascii"))
            for block in self.iter_decoded_weight_chunks():
                digest.update(memoryview(block.view(torch.uint8).numpy()))
            if controls:
                for name in ("stability", "uses", "plasticity_events"):
                    update_historical_tensor_checksum(digest, getattr(self, name), self)
            return digest.hexdigest()

    def checksum_with_controls(self) -> str:
        """Exact old Fresh boundary digest, without decoded N×N weights."""
        return self._historical_checksum(controls=True)

    def decoded_weight_checksum(self) -> str:
        """Exact old feedback weight-only digest, not a packed-byte digest."""
        return self._historical_checksum(controls=False)

    @torch.no_grad()
    def recurrent(self, inputs: torch.Tensor) -> torch.Tensor:
        """Every original dense edge participates; only temporary tiles decode."""
        if inputs.numel() != self.pre_neurons or not bool(torch.isfinite(inputs).all()):
            raise ValueError("router recurrence vector is invalid")
        with self._operation(), self._ram((self.pre_neurons + self.post_neurons) * inputs.element_size() * 3, "router recurrent live vectors"):
            vector = inputs.detach().reshape(-1).to(device="cpu")
            output = torch.zeros(self.post_neurons, dtype=vector.dtype)
            for region, levels in self.iter_effective_tiles():
                r0, r1, c0, c1 = region
                output[r0:r1].add_(torch.mv(levels.to(vector.dtype), vector[c0:c1]))
            return output.to(inputs.device)

    @torch.no_grad()
    def active_synapse_count(self) -> int:
        return sum(int(levels.ne(0).sum()) for _, levels in self.iter_effective_tiles())

    @torch.no_grad()
    def stability_mean(self, prefix: Optional[int] = None) -> float:
        rows = self.post_neurons if prefix is None else min(self.post_neurons, int(prefix))
        columns = self.pre_neurons if prefix is None else min(self.pre_neurons, int(prefix))
        total = 0.0
        with self._operation():
            for r0, r1, c0, c1 in tile_ranges(rows, columns, self._tile_budget()):
                with self._ram((r1-r0)*(c1-c0)*16, "router stability reduction"):
                    total += float(self.stability[r0:r1, c0:c1].detach().to(device="cpu", dtype=torch.float64).sum())
                self._release_tile(r0, r1, c0, c1)
        return total / max(1, rows * columns)

    @torch.no_grad()
    def step(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> STDPUpdateSummary:
        if pre_spikes.numel() != self.pre_neurons or post_spikes.numel() != self.post_neurons:
            raise ValueError("spike vector dimensions do not match synapses")
        pager = getattr(self, "_router_state_pager", None)
        ram_limit = pager.journal_ram_bytes if pager is not None else TRANSFER_BYTES
        with self._operation(), self._ram((self.pre_neurons + self.post_neurons) * 24, "router STDP live vectors"):
            pre = pre_spikes.detach().reshape(-1).to(device="cpu", dtype=self.pre_trace.dtype)
            post = post_spikes.detach().reshape(-1).to(device="cpu", dtype=self.post_trace.dtype)
            old_pre = self.pre_trace.detach().to(device="cpu", copy=True)
            old_post = self.post_trace.detach().to(device="cpu", copy=True)
            if not all(bool(torch.isfinite(value).all()) for value in (pre, post, old_pre, old_post)):
                raise ValueError("router spike/timing vectors must be finite")
            journal = RouterMutationJournal(self, pager, ram_bytes=ram_limit)
            absolute = signed = 0.0
            level_abs = level_signed = changed_count = active_count = 0
            inactive_eligibility_present = False
            try:
                vector_block = max(1, self._tile_budget() // 32)
                for name in ("pre_trace", "post_trace"):
                    for start in range(0, getattr(self, name).numel(), vector_block):
                        journal.capture(name, start, min(getattr(self, name).numel(), start + vector_block))
                journal.capture("plasticity_events")
                for r0, r1, c0, c1 in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                    self._check()
                    with self._ram((r1-r0)*(c1-c0)*96 + 1024, "exact causal/anti-causal STDP tile"):
                        stability = self.stability[r0:r1, c0:c1].detach().to(device="cpu", copy=True)
                        timing = self.a_plus * (post[r0:r1, None] * old_pre[None, c0:c1])
                        timing = timing - self.a_minus * (old_post[r0:r1, None] * pre[None, c0:c1])
                        delta = (self.learning_rate / (1.0 + stability)) * timing
                        absolute += float(delta.abs().to(torch.float64).sum())
                        signed += float(delta.to(torch.float64).sum())
                        active = timing.ne(0)
                        active_count += int(active.sum())
                        if bool(active.any()):
                            previous = unpack_tile(self._packed_weights, r0, r1, c0, c1)
                            uses = self.uses[r0:r1, c0:c1].detach().to(device="cpu", copy=True)
                            eligibility = self.eligibility_accumulator[r0:r1, c0:c1].detach().to(device="cpu", copy=True)
                            agreement = previous.sign().eq(delta.sign()) | previous.eq(0)
                            stability.add_(torch.where(agreement, torch.full_like(stability, self.metaplasticity_rate),
                                torch.full_like(stability, -self.metaplasticity_rate * .25)) * active).clamp_(0., 20.)
                            uses.add_(active.to(uses.dtype))
                            increments = (delta / max(self.learning_rate, 1e-6) * 256.).round().clamp(-256, 256).to(torch.int32)
                            pressure = (eligibility.to(torch.int32) + increments).clamp(-256, 256)
                            transition = torch.where(pressure >= 256, torch.ones_like(pressure),
                                torch.where(pressure <= -256, -torch.ones_like(pressure), torch.zeros_like(pressure)))
                            next_levels = (previous.to(torch.int32) + transition).clamp(-1, 1)
                            pressure = pressure - transition * 256
                            pressure = torch.where(next_levels.eq(previous), 0, pressure).to(torch.int16)
                            difference = next_levels - previous.to(torch.int32)
                            level_abs += int(difference.abs().sum()); level_signed += int(difference.sum())
                            changed_count += int(difference.ne(0).sum())
                            replacements = {"stability": stability, "uses": uses, "eligibility_accumulator": pressure}
                            if bool(difference.ne(0).any()):
                                replacements["_packed_weights"] = pack_tile(next_levels.to(torch.int8))
                            writes = []
                            for name, replacement in replacements.items():
                                a, b = (c0//4, (c1+3)//4) if name == "_packed_weights" else (c0, c1)
                                target = getattr(self, name)[r0:r1, a:b]
                                if not torch.equal(target.detach().cpu(), replacement):
                                    journal.capture(name, r0, r1, a, b)
                                    writes.append((target, replacement))
                            for target, replacement in writes:
                                target.copy_(replacement.to(target.device))
                        elif not inactive_eligibility_present:
                            inactive_eligibility_present = bool(self.eligibility_accumulator[r0:r1, c0:c1].ne(0).any())
                    self._release_tile(r0, r1, c0, c1)
                if active_count and inactive_eligibility_present:
                    # The original whole-matrix rule resets subthreshold
                    # eligibility even in quiet pairs whenever ANY pair is
                    # active. Preserve that cross-tile dependency; an all-
                    # silent step still retains its exact previous controls.
                    for r0, r1, c0, c1 in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                        self._check()
                        with self._ram((r1-r0)*(c1-c0)*16 + 1024, "router quiet-pair eligibility reset"):
                            target = self.eligibility_accumulator[r0:r1, c0:c1]
                            if bool(target.ne(0).any()):
                                journal.capture("eligibility_accumulator", r0, r1, c0, c1)
                                target.zero_()
                        self._release_tile(r0, r1, c0, c1)
                self._check()
                self.pre_trace.mul_(self.pre_decay).add_(pre.to(self.pre_trace.device))
                self.post_trace.mul_(self.post_decay).add_(post.to(self.post_trace.device))
                self.plasticity_events.add_(active_count)
                return STDPUpdateSummary(absolute, signed, level_abs, level_signed, changed_count, active_count)
            except BaseException:
                journal.rollback()
                raise
            finally:
                journal.close()

    @torch.no_grad()
    def decay_unused(self, amount: float = 1e-4) -> None:
        amount = max(0.0, min(float(amount), 1.0))
        pager = getattr(self, "_router_state_pager", None)
        with self._operation():
            journal = RouterMutationJournal(self, pager, ram_bytes=pager.journal_ram_bytes if pager is not None else TRANSFER_BYTES)
            try:
                journal.capture("decay_cycles")
                self.decay_cycles.add_(1)
                tick = int(self.decay_cycles)
                for r0, r1, c0, c1 in tile_ranges(self.post_neurons, self.pre_neurons, self._tile_budget()):
                    self._check()
                    with self._ram((r1-r0)*(c1-c0)*96 + 1024, "router exact decay tile"):
                        if amount:
                            levels = unpack_tile(self._packed_weights, r0, r1, c0, c1)
                            positions = torch.arange(r0, r1, dtype=torch.int64)[:, None] * self.pre_neurons + torch.arange(c0, c1, dtype=torch.int64)[None, :]
                            draw = ((positions * 1664525 + tick * 1013904223) & 0xFFFFFFFF).float() / 4294967296.0
                            uses = self.uses[r0:r1, c0:c1].detach().cpu()
                            decay = levels.ne(0) & (draw < amount / (1.0 + uses))
                            if bool(decay.any()):
                                replacement = pack_tile(torch.where(decay, 0, levels).to(torch.int8))
                                journal.capture("_packed_weights", r0, r1, c0//4, (c1+3)//4)
                                self._packed_weights[r0:r1, c0//4:(c1+3)//4].copy_(replacement.to(self._packed_weights.device))
                            replacement = self.stability[r0:r1, c0:c1].detach().cpu() * (1.0 - amount * .1)
                            journal.capture("stability", r0, r1, c0, c1)
                            self.stability[r0:r1, c0:c1].copy_(replacement.to(self.stability.device))
                    self._release_tile(r0, r1, c0, c1)
            except BaseException:
                journal.rollback(); raise
            finally:
                journal.close()


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
        router_pager = getattr(self.synapses, "_router_state_pager", None)
        if router_pager is not None and router_pager.status()["activeOperations"]:
            raise RuntimeError("router growth requires all operations to finish")
        router_scope = router_pager.construction(loading=False) if router_pager is not None else nullcontext()
        with scope, router_scope:
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
            for name in MATRIX_FIELDS:
                source, target = getattr(old_synapses, name), getattr(synapses, name)
                for r0, r1, c0, c1 in tile_ranges(source.shape[0], source.shape[1], synapses._tile_budget()):
                    with synapses._ram((r1-r0)*(c1-c0)*target.element_size()*3, "exact router growth prefix"):
                        target[r0:r1, c0:c1].copy_(source[r0:r1, c0:c1].to(target.device))
                    if router_pager is not None:
                        for row in range(r0, r1):
                            router_pager.release_chunk(target, (row*target.shape[1]+c0)*target.element_size(), (c1-c0)*target.element_size())
            for name in ("pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
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
        if router_pager is not None:
            router_pager.release_owner(old_synapses)
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
            recurrent = self.synapses.recurrent(previous.detach())
            current = projected + 0.35 * recurrent
            if active_neurons < self.neurons:
                current = current * (torch.arange(self.neurons, device=current.device) < active_neurons)
            spikes, _ = self.population.step(
                current, threshold_offset=threshold_offset
            )
            hard_spikes = (spikes.detach() > 0.5).to(spikes)
            if learn:
                delta = self.synapses.step(previous.detach(), hard_spikes)
                total_update += delta.delta_absolute_sum
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
            "mean_stability": self.synapses.stability_mean(active_neurons),
            "active_synapses": float(self.synapses.active_synapse_count()),
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
        if direction > 0:
            self.synapses.step(pre, silence)
            first_delta = self.synapses.step(silence, post)
        else:
            self.synapses.step(silence, post)
            first_delta = self.synapses.step(pre, silence)
        return {
            "stdp_update": float(first_delta.changed_absolute_sum),
            "signed_update": float(first_delta.changed_signed_sum),
            "timing_signal": first_delta.delta_absolute_sum,
            "active_pairs": float(first_delta.changed_pairs),
            "mean_stability": self.synapses.stability_mean(),
            "plasticity_events": float(
                self.synapses.plasticity_events.item()
            ),
        }
