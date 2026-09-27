"""Ternary decoder-only language model used by an OmniCortex brain."""

import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import OmniConfig


ACTION_KINDS = (
    "talk",
    "tool",
    "imagine",
    "agent",
    "ponder",
    "learn",
    "evolve",
    "stop",
)


PACKED_TERNARY_WEIGHTS_PER_BYTE = 4
PACKED_TERNARY_OUTPUT_BLOCK = 64
PACKED_TERNARY_LEARN_SAMPLE_BLOCK = 256
PACKED_STABILITY_MAX = 15


class _PackedRowMetaplasticity:
    """One byte of resistance per output row, never a dense weight shadow.

    Repeated successful plastic changes make that row harder, but never
    impossible, to change again. The state is checkpointed with the packed
    module and moves/offloads with it. A row-level approximation is deliberate:
    per-synapse FP32 eligibility would exceed the packed model's residency.
    """

    def _init_packed_stability(self, rows: int, has_bias: bool) -> None:
        self.register_buffer(
            "_row_stability", torch.zeros(int(rows), dtype=torch.uint8)
        )
        self.register_buffer(
            "_bias_row_stability",
            torch.zeros(1, dtype=torch.uint8) if has_bias else None,
        )
        self._packed_stability_strength = 0.0
        self._pending_stability_events = 0
        self._row_stability_validated_version = -1
        self._bias_row_stability_validated_version = -1

    def _packed_stability_changed(self) -> bool:
        bias = self._bias_row_stability
        return (
            self._row_stability_validated_version
            != int(self._row_stability._version)
            or (
                bias is not None
                and self._bias_row_stability_validated_version
                != int(bias._version)
            )
        )

    def configure_packed_stability(self, *, enabled: bool, strength: float) -> None:
        value = float(strength)
        if not math.isfinite(value) or value < 0:
            raise ValueError("packed stability strength must be finite and nonnegative")
        # The historical EWC coefficient operated on squared floating weight
        # differences. A fourfold conversion gives its default 0.025 a modest
        # 0.10 per-row resistance without storing a master or hard-freezing.
        self._packed_stability_strength = 4.0 * value if enabled else 0.0

    def drain_packed_stability_events(self) -> int:
        events = int(self._pending_stability_events)
        self._pending_stability_events = 0
        return events

    def packed_stability_status(self) -> Dict[str, object]:
        return {
            "metaplasticityMode": "bounded-uint8-output-row-resistance",
            "metaplasticityEnabled": self._packed_stability_strength > 0.0,
            "metaplasticityCheckpointBytes": int(self._row_stability.numel())
            + (
                0
                if self._bias_row_stability is None
                else int(self._bias_row_stability.numel())
            ),
            "metaplasticityMaximumLevel": PACKED_STABILITY_MAX,
        }

    def _validate_packed_stability(self, rows: int, device: torch.device) -> None:
        resistance = self._row_stability
        if (
            resistance.dtype != torch.uint8
            or resistance.shape != (int(rows),)
            or resistance.device != device
            or bool((resistance > PACKED_STABILITY_MAX).any())
        ):
            raise ValueError("packed output-row stability state is invalid")
        bias = self._bias_row_stability
        if (bias is not None) != bool(getattr(self, "has_bias", False)):
            raise ValueError("packed bias-row stability presence is invalid")
        if bias is not None and (
            bias.dtype != torch.uint8
            or bias.shape != (1,)
            or bias.device != device
            or bool((bias > PACKED_STABILITY_MAX).any())
        ):
            raise ValueError("packed bias-row stability state is invalid")
        self._row_stability_validated_version = int(resistance._version)
        self._bias_row_stability_validated_version = (
            int(bias._version) if bias is not None else -1
        )


def pack_ternary_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack exact ``{-1, 0, +1}`` rows into canonical two-bit codes.

    Codes are row-padded with ternary zero so a row never shares a byte with
    another row. A caller may use the result either as a forward cache or as
    its authoritative, mutable synaptic storage.
    """

    if weight.ndim != 2:
        raise ValueError("packed ternary linear weight must be a matrix")
    if weight.dtype != torch.int8:
        raise ValueError("packed ternary linear weight must use int8 levels")
    if not bool(((weight >= -1) & (weight <= 1)).all()):
        raise ValueError("packed ternary linear weight contains an invalid level")
    rows, columns = weight.shape
    packed_columns = (int(columns) + 3) // 4
    padded_columns = packed_columns * 4
    codes = (weight.to(torch.int16) + 1).to(torch.uint8)
    if padded_columns != int(columns):
        padding = torch.ones(
            (int(rows), padded_columns - int(columns)),
            dtype=torch.uint8,
            device=weight.device,
        )
        codes = torch.cat((codes, padding), dim=1)
    lanes = codes.reshape(int(rows), packed_columns, 4)
    return (
        lanes[:, :, 0]
        | (lanes[:, :, 1] << 2)
        | (lanes[:, :, 2] << 4)
        | (lanes[:, :, 3] << 6)
    ).contiguous()


def unpack_ternary_weight_rows(
    packed: torch.Tensor,
    in_features: int,
    start: int = 0,
    end: Optional[int] = None,
    *,
    validate_reserved: bool = True,
) -> torch.Tensor:
    """Decode one bounded output-row block to int8 for integer matmul.

    Public callers validate reserved packed codes by default. Native runtime
    projections validate their authoritative packed tensor on mutation, then
    omit per-block validation to avoid accelerator-to-host synchronization.
    """

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed ternary weight must be a uint8 matrix")
    stop = int(packed.shape[0]) if end is None else int(end)
    if start < 0 or stop < start or stop > int(packed.shape[0]):
        raise ValueError("packed ternary row range is invalid")
    selected = packed[int(start) : stop]
    lanes = torch.stack(
        tuple((selected >> shift) & 0x03 for shift in (0, 2, 4, 6)),
        dim=-1,
    ).reshape(stop - int(start), -1)
    lanes = lanes[:, : int(in_features)]
    if validate_reserved and bool((lanes == 3).any()):
        raise ValueError("packed ternary weight contains a reserved code")
    return (lanes.to(torch.int8) - 1).contiguous()


@torch.no_grad()
def _apply_packed_gradient_rows(
    packed: torch.Tensor,
    width: int,
    row_start: int,
    gradient: torch.Tensor,
    rate: float,
    scale: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    row_stability: Optional[torch.Tensor] = None,
    stability_strength: float = 0.0,
) -> int:
    """Apply a bounded stochastic step directly to authoritative 2-bit rows."""
    row_end = int(row_start) + int(gradient.shape[0])
    levels = unpack_ternary_weight_rows(
        packed, width, row_start, row_end, validate_reserved=False
    )
    # ``gradient`` is with respect to a physical projection weight, whereas
    # one ternary level changes that weight by ``scale``. Convert a physical
    # SGD-sized update to level units; multiplying by scale here would make
    # wider/smaller-gain layers artificially unable to learn.
    strength = gradient.abs() * (float(rate) / float(scale))
    if row_stability is not None and stability_strength > 0.0:
        if (
            row_stability.dtype != torch.uint8
            or row_stability.ndim != 1
            or row_stability.device != packed.device
            or row_end > row_stability.numel()
        ):
            raise ValueError("packed row stability shape, dtype or device is invalid")
        strength = strength / (
            1.0
            + float(stability_strength)
            * row_stability[row_start:row_end].float().unsqueeze(-1)
        )
    probability = -torch.expm1(-strength)
    draws = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    movement = torch.where(
        (draws < probability) & (gradient < 0) & (levels < 1),
        1,
        torch.where(
            (draws < probability) & (gradient > 0) & (levels > -1),
            -1,
            0,
        ),
    ).to(torch.int8)
    changed = int((movement != 0).sum().item())
    if changed:
        packed[row_start:row_end].copy_(pack_ternary_weight(levels + movement))
        if row_stability is not None and stability_strength > 0.0:
            rows = row_stability[row_start:row_end]
            grew = movement.ne(0).any(dim=1)
            rows.copy_(
                torch.where(
                    grew,
                    (rows.to(torch.int16) + 1).clamp_max(PACKED_STABILITY_MAX)
                    .to(torch.uint8),
                    rows,
                )
            )
    return changed


def _quantize_activation_int8(
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = inputs.float()
    scale = (values.detach().abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(
        torch.finfo(torch.float32).eps
    )
    quantized = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


PACKED_INTEGER_INPUT_BLOCK = 256


def _unsupported_integer_kernel(error: BaseException) -> bool:
    if isinstance(error, NotImplementedError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "not implemented", "not supported", "unsupported",
            "not compiled", "could not run 'aten::",
        )
    )


def _bounded_cpu_integer_mm(
    inputs: torch.Tensor, weight_block: torch.Tensor
) -> torch.Tensor:
    """Fallback one packed output block at a time, never a full weight mirror."""

    weights = weight_block.to(device="cpu", dtype=torch.int32).t().contiguous()
    chunks = []
    for start in range(0, int(inputs.shape[0]), PACKED_INTEGER_INPUT_BLOCK):
        end = min(int(inputs.shape[0]), start + PACKED_INTEGER_INPUT_BLOCK)
        activity = inputs[start:end].to(device="cpu", dtype=torch.int32)
        try:
            chunks.append(activity @ weights)
        except (NotImplementedError, RuntimeError) as error:
            if not _unsupported_integer_kernel(error):
                raise
            # A few CPU builds lack even int32 matmul. Elementwise products
            # and int64 reductions remain exact without a dense float weight.
            rows = weights.t().to(torch.int64)
            chunks.append(
                torch.stack(
                    [
                        (activity.to(torch.int64) * row).sum(dim=1)
                        for row in rows
                    ],
                    dim=1,
                )
            )
    if not chunks:
        return torch.empty(
            (0, int(weight_block.shape[0])),
            dtype=torch.int32,
            device=inputs.device,
        )
    return torch.cat(chunks, dim=0).to(inputs.device)


def _packed_integer_mm(
    inputs: torch.Tensor, weight_block: torch.Tensor
) -> torch.Tensor:
    # CPU _int_mm is present in some PyTorch builds but has no CPU kernel on
    # macOS Intel. Only request it on CUDA; ordinary int32 matmul is exact for
    # the bounded projection widths used by OmniCortex.
    if inputs.device.type == "cuda" and hasattr(torch, "_int_mm"):
        try:
            return torch._int_mm(  # type: ignore[attr-defined]
                inputs, weight_block.t().contiguous()
            )
        except (NotImplementedError, RuntimeError) as error:
            if not _unsupported_integer_kernel(error):
                raise
    try:
        return inputs.to(torch.int32) @ weight_block.to(torch.int32).t()
    except (NotImplementedError, RuntimeError) as error:
        if not _unsupported_integer_kernel(error):
            raise
        # MPS/DirectML builds may lack integer matmul. Transfer only this
        # bounded output block and row chunks, then return to the live device.
        return _bounded_cpu_integer_mm(inputs, weight_block)


def _packed_ternary_forward(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    in_features: int,
    out_features: int,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if inputs.shape[-1] != int(in_features):
        raise ValueError("packed ternary input width is invalid")
    quantized, activation_scale = _quantize_activation_int8(inputs)
    flat = quantized.reshape(-1, int(in_features)).contiguous()
    blocks: List[torch.Tensor] = []
    for start in range(0, int(out_features), PACKED_TERNARY_OUTPUT_BLOCK):
        end = min(int(out_features), start + PACKED_TERNARY_OUTPUT_BLOCK)
        weight_block = unpack_ternary_weight_rows(
            packed,
            int(in_features),
            start,
            end,
            validate_reserved=False,
        )
        blocks.append(_packed_integer_mm(flat, weight_block))
    integer_output = torch.cat(blocks, dim=-1).reshape(
        *inputs.shape[:-1], int(out_features)
    )
    return (
        integer_output.float()
        * activation_scale
        * weight_scale.detach().float()
    ).to(inputs.dtype)


def _packed_ternary_input_gradient(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    in_features: int,
    out_features: int,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    flat_output = grad_output.reshape(-1, int(out_features))
    flat_input = torch.zeros(
        (flat_output.shape[0], int(in_features)),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    scale = weight_scale.detach().to(grad_output.dtype)
    for row in range(int(out_features)):
        levels = unpack_ternary_weight_rows(
            packed,
            int(in_features),
            row,
            row + 1,
            validate_reserved=False,
        )[0]
        positive = torch.nonzero(levels > 0, as_tuple=False).flatten()
        negative = torch.nonzero(levels < 0, as_tuple=False).flatten()
        contribution = flat_output[:, row : row + 1] * scale
        if positive.numel():
            flat_input.index_add_(
                1,
                positive,
                contribution.expand(-1, int(positive.numel())),
            )
        if negative.numel():
            flat_input.index_add_(
                1,
                negative,
                -contribution.expand(-1, int(negative.numel())),
            )
    return flat_input.reshape(*grad_output.shape[:-1], int(in_features))


class _PackedOnlyTernaryLinear(torch.autograd.Function):
    """Autograd for inputs only; synapses are updated by an explicit local rule."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inputs: torch.Tensor,
        packed: torch.Tensor,
        scale: torch.Tensor,
        in_features: int,
        out_features: int,
        module: "PackedAdaptiveBitLinear",
        autograd_trigger: torch.Tensor,
    ) -> torch.Tensor:
        ctx.in_features = int(in_features)
        ctx.out_features = int(out_features)
        ctx.module = module
        # The packed tensor is kept by reference, not as an autograd-saved
        # tensor: this is an online plasticity rule that may mutate it during
        # backward. No dense copy of the weights is retained per graph.
        ctx.packed = packed
        ctx.scale = scale
        ctx.save_for_backward(inputs)
        output = _packed_ternary_forward(
            inputs, packed, ctx.in_features, ctx.out_features, scale
        )
        if module.has_bias:
            output = output + (
                module.effective_bias().to(output.dtype)
                * scale.detach().to(output.dtype)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (inputs,) = ctx.saved_tensors
        grad_input = _packed_ternary_input_gradient(
            grad_output,
            ctx.packed,
            ctx.in_features,
            ctx.out_features,
            ctx.scale,
        )
        if ctx.module.training:
            ctx.module.learn_from_gradient(
                inputs,
                grad_output,
                ctx.module.online_learning_rate,
            )
        return grad_input, None, None, None, None, None, None


class PackedAdaptiveBitLinear(_PackedRowMetaplasticity, nn.Module):
    """A packed-authoritative ternary projection with direct discrete updates.

    It has no dense floating weight parameter, latent master, or per-weight
    optimizer moments. Four synapses occupy one
    byte throughout its lifetime. ``learn_from_gradient`` consumes activations
    and downstream gradients in bounded row blocks and stochastically moves
    each eligible synapse by one exact ternary level. The stochastic update
    keeps subthreshold gradients capable of learning over repeated events
    without retaining a second full-sized weight tensor.

    This is the native cortical projection used by the current decoder and
    associated modalities.
    """

    ternary = True
    ternary_eligible = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        scale: Optional[float] = None,
        online_learning_rate: float = 1.0,
    ) -> None:
        super().__init__()
        if int(in_features) <= 0 or int(out_features) <= 0:
            raise ValueError("packed ternary dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.has_bias = bool(bias)
        initial_scale = (
            math.sqrt(3.0 / (self.in_features + self.out_features))
            if scale is None
            else float(scale)
        )
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError("packed ternary scale must be finite and positive")
        if not math.isfinite(float(online_learning_rate)) or float(online_learning_rate) < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        self.register_buffer(
            "_packed_forward_weight",
            torch.empty(
                (self.out_features, (self.in_features + 3) // 4),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "_packed_forward_bias",
            (
                torch.full(
                    (1, (self.out_features + 3) // 4), 0x55, dtype=torch.uint8
                )
                if self.has_bias
                else None
            ),
        )
        self.register_buffer(
            "_packed_forward_scale",
            torch.tensor(initial_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "_online_learning_rate",
            torch.tensor(float(online_learning_rate), dtype=torch.float32),
        )
        # One scalar makes a standalone first projection participate in normal
        # loss.backward() even when token/sensory inputs have no gradient. It
        # is not an optimizer parameter or a second synaptic weight array.
        self.register_buffer(
            "_autograd_trigger", torch.zeros((), requires_grad=True), persistent=False
        )
        self._init_packed_stability(self.out_features, self.has_bias)
        self._packed_validated_version = -1
        self._bias_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device: Optional[torch.device] = None
        self._online_transaction: Optional[Dict[str, Any]] = None
        # Initialization itself is row-blocked: no whole dense weight is held.
        with torch.no_grad():
            for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
                end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
                levels = torch.randint(
                    -1, 2, (end - start, self.in_features), dtype=torch.int8
                )
                self._packed_forward_weight[start:end].copy_(
                    pack_ternary_weight(levels)
                )
        self._validate_packed()

    def _apply(self, fn):
        if self._online_transaction is not None:
            raise RuntimeError("cannot move packed projection during an online step")
        result = super()._apply(fn)
        self._packed_validated_version = -1
        self._bias_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device = None
        return result

    @property
    def online_learning_rate(self) -> float:
        return float(self._online_learning_rate.item())

    @online_learning_rate.setter
    def online_learning_rate(self, value: float) -> None:
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        self._online_learning_rate.fill_(rate)

    def _validate_packed(self) -> None:
        packed = self._packed_forward_weight
        if packed.dtype != torch.uint8 or packed.shape != (
            self.out_features,
            (self.in_features + 3) // 4,
        ):
            raise ValueError("packed-authoritative synapse shape or dtype is invalid")
        # Validate both active codes and the row-padding codes. A bad external
        # load/mutation must fail closed instead of becoming an effective +2.
        for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
            block = packed[start:end]
            codes = torch.stack(
                tuple((block >> shift) & 0x03 for shift in (0, 2, 4, 6)),
                dim=-1,
            ).reshape(end - start, -1)
            if bool((codes[:, : self.in_features] == 3).any()):
                raise ValueError("packed-authoritative synapse contains a reserved code")
            if bool((codes[:, self.in_features :] != 1).any()):
                raise ValueError("packed-authoritative synapse has nonzero row padding")
        bias = self._packed_forward_bias
        if self.has_bias:
            if bias is None or bias.dtype != torch.uint8 or bias.shape != (
                1, (self.out_features + 3) // 4
            ):
                raise ValueError("packed-authoritative bias shape or dtype is invalid")
            if bias.device != packed.device:
                raise ValueError("packed-authoritative bias and synapses must share a device")
            bias_codes = torch.stack(
                tuple((bias >> shift) & 0x03 for shift in (0, 2, 4, 6)),
                dim=-1,
            ).reshape(1, -1)
            if bool((bias_codes[:, : self.out_features] == 3).any()):
                raise ValueError("packed-authoritative bias contains a reserved code")
            if bool((bias_codes[:, self.out_features :] != 1).any()):
                raise ValueError("packed-authoritative bias has nonzero padding")
            self._bias_validated_version = int(bias._version)
        elif bias is not None:
            raise ValueError("packed-authoritative bias is unexpected")
        scale = self._packed_forward_scale
        if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("packed-authoritative scale must be finite and positive")
        if scale.device != packed.device:
            raise ValueError("packed-authoritative scale and synapses must share a device")
        self._validate_packed_stability(self.out_features, packed.device)
        self._packed_validated_version = int(packed._version)
        self._scale_validated_version = int(scale._version)
        self._validated_device = packed.device

    def packed_forward_weight(self) -> torch.Tensor:
        if (
            self._packed_validated_version != int(self._packed_forward_weight._version)
            or (
                self.has_bias
                and self._packed_forward_bias is not None
                and self._bias_validated_version
                != int(self._packed_forward_bias._version)
            )
            or self._scale_validated_version != int(self._packed_forward_scale._version)
            or self._packed_stability_changed()
            or self._validated_device != self._packed_forward_weight.device
        ):
            self._validate_packed()
        return self._packed_forward_weight

    def effective_weight(self) -> torch.Tensor:
        """Materialize exact levels for inspection/export only, not residency."""
        return unpack_ternary_weight_rows(
            self.packed_forward_weight(), self.in_features
        )

    @torch.no_grad()
    def set_ternary_weight_(self, levels: torch.Tensor) -> None:
        """Install exact packed levels, including for a newly grown expert route."""
        if self._online_transaction is not None:
            raise RuntimeError("cannot replace packed synapses during an online step")
        if levels.dtype != torch.int8 or levels.shape != (
            self.out_features, self.in_features
        ):
            raise ValueError("packed ternary replacement has invalid shape or dtype")
        packed = self.packed_forward_weight()
        packed.copy_(pack_ternary_weight(levels.to(device=packed.device)))
        self._row_stability.zero_()
        self._pending_stability_events = 0
        self._validate_packed()

    def effective_bias(self) -> Optional[torch.Tensor]:
        """Materialize the ternary bias levels only when requested."""
        self.packed_forward_weight()
        if self._packed_forward_bias is None:
            return None
        return unpack_ternary_weight_rows(
            self._packed_forward_bias, self.out_features
        )[0]

    def authoritative_packed_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Persistent synapse bytes, excluding scalar gains and training state."""
        packed = self.packed_forward_weight()
        if self._packed_forward_bias is None:
            return (packed,)
        return packed, self._packed_forward_bias

    @property
    def logical_ternary_parameter_count(self) -> int:
        return self.in_features * self.out_features + (
            self.out_features if self.has_bias else 0
        )

    @property
    def ternary_weight_shape(self) -> Tuple[int, ...]:
        return self.out_features, self.in_features

    @property
    def ternary_bias_shape(self) -> Tuple[int, ...]:
        return (self.out_features,) if self.has_bias else ()

    @torch.no_grad()
    def fill_ternary_(
        self, level: int, *, zero_bias: bool = True
    ) -> "PackedAdaptiveBitLinear":
        """Fill authoritative synapses with one exact level, in row blocks.

        This supports deterministic zeroing of a newly grown expert without
        exposing or retaining a floating ``weight`` parameter. The optional
        packed bias is zeroed by default; callers may preserve it explicitly.
        """
        if isinstance(level, bool) or level not in (-1, 0, 1):
            raise ValueError("packed ternary fill level must be -1, 0, or +1")
        packed = self.packed_forward_weight()
        for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
            levels = torch.full(
                (end - start, self.in_features),
                int(level),
                dtype=torch.int8,
                device=packed.device,
            )
            packed[start:end].copy_(pack_ternary_weight(levels))
        self._row_stability.zero_()
        self._pending_stability_events = 0
        self._packed_validated_version = int(packed._version)
        if zero_bias and self._packed_forward_bias is not None:
            self._packed_forward_bias.fill_(0x55)
            self._bias_row_stability.zero_()
            self._bias_validated_version = int(self._packed_forward_bias._version)
        return self

    def begin_online_step(self, *, max_scratch_synapses: int = 1048576) -> None:
        """Accumulate local gradients until one post-backward packed update.

        The temporary FP32 derivative is bounded and never a resident learned
        weight or optimizer moment. A large cortical matrix fails closed; it
        continues to use its row-blocked immediate update path instead.
        """
        if self._online_transaction is not None:
            raise RuntimeError("packed online step is already active")
        if self.logical_ternary_parameter_count > int(max_scratch_synapses):
            raise ValueError("packed online step exceeds bounded scratch allowance")
        device = self.packed_forward_weight().device
        self._online_transaction = {
            "weight": torch.zeros(
                (self.out_features, self.in_features),
                dtype=torch.float32,
                device=device,
            ),
            "bias": (
                torch.zeros((self.out_features,), dtype=torch.float32, device=device)
                if self.has_bias
                else None
            ),
            "rate": None,
            "calls": 0,
        }

    @torch.no_grad()
    def commit_online_step(self) -> int:
        """Apply one level update from the sum of all uses in a graph."""
        transaction = self._online_transaction
        if transaction is None:
            raise RuntimeError("packed online step is not active")
        self._online_transaction = None
        if not transaction["calls"]:
            return 0
        packed = self.packed_forward_weight()
        rate = float(transaction["rate"])
        changed = 0
        for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
            changed += _apply_packed_gradient_rows(
                packed,
                self.in_features,
                start,
                transaction["weight"][start:end],
                rate,
                self._packed_forward_scale,
                row_stability=self._row_stability,
                stability_strength=self._packed_stability_strength,
            )
        self._packed_validated_version = int(packed._version)
        if self._packed_forward_bias is not None:
            changed += _apply_packed_gradient_rows(
                self._packed_forward_bias,
                self.out_features,
                0,
                transaction["bias"][None, :],
                rate,
                self._packed_forward_scale,
                row_stability=self._bias_row_stability,
                stability_strength=self._packed_stability_strength,
            )
            self._bias_validated_version = int(self._packed_forward_bias._version)
        if changed and self._packed_stability_strength > 0.0:
            self._pending_stability_events += 1
        return changed

    def discard_online_step(self) -> None:
        """Drop temporary derivatives without changing packed synapses."""
        self._online_transaction = None

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Effective scaled bias, reconstructed on demand (not resident)."""
        levels = self.effective_bias()
        if levels is None:
            return None
        return levels.float() * self._packed_forward_scale.float()

    def packed_forward_status(self) -> Dict[str, object]:
        packed = self.packed_forward_weight()
        return {
            "format": "omni-packed-adaptive-ternary-linear",
            "formatVersion": 1,
            "weightLevels": [-1, 0, 1],
            "weightsPerByte": PACKED_TERNARY_WEIGHTS_PER_BYTE,
            "activationBits": 8,
            "packedBytes": int(packed.numel())
            + (0 if self._packed_forward_bias is None else int(self._packed_forward_bias.numel())),
            "hasPackedBias": self.has_bias,
            "authoritativePackedWeight": True,
            "denseForwardWeightMaterialized": False,
            "latentMasterLearningState": False,
            **self.packed_stability_status(),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _PackedOnlyTernaryLinear.apply(
            inputs,
            self.packed_forward_weight(),
            self._packed_forward_scale,
            self.in_features,
            self.out_features,
            self,
            self._autograd_trigger if self.training else self._autograd_trigger.detach(),
        )

    @torch.no_grad()
    def learn_from_gradient(
        self,
        inputs: torch.Tensor,
        grad_output: torch.Tensor,
        learning_rate: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Immediately mutate packed synapses from one local gradient event.

        The caller supplies the input activity and the gradient of loss with
        respect to this layer's output, normally after ``backward()``. A row
        block is the largest transient dense learning tensor. Each level
        changes at most one step toward negative gradient; independent
        Bernoulli draws give small gradients a nonzero chance to act without
        persistent floating shadows. Returns the number of changed synapses.
        """
        rate = float(learning_rate)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if inputs.shape[-1] != self.in_features or grad_output.shape[-1] != self.out_features:
            raise ValueError("packed local-gradient dimensions are invalid")
        activity = inputs.detach().reshape(-1, self.in_features)
        downstream = grad_output.detach().reshape(-1, self.out_features)
        if activity.shape[0] == 0 or activity.shape[0] != downstream.shape[0]:
            raise ValueError("packed local-gradient examples are missing or misaligned")
        if activity.device != self._packed_forward_weight.device or downstream.device != activity.device:
            raise ValueError("packed local-gradient tensors must share the weight device")
        if not bool(torch.isfinite(activity).all()) or not bool(torch.isfinite(downstream).all()):
            raise ValueError("packed local-gradient tensors must be finite")
        if rate == 0.0:
            return 0
        packed = self.packed_forward_weight()
        transaction = self._online_transaction
        if transaction is not None:
            if transaction["rate"] is None:
                transaction["rate"] = rate
            elif transaction["rate"] != rate:
                raise ValueError("packed online step received inconsistent learning rates")
            transaction["calls"] += 1
        changed = 0
        for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
            gradient = torch.zeros(
                (end - start, self.in_features),
                dtype=torch.float32,
                device=activity.device,
            )
            for sample_start in range(
                0, int(activity.shape[0]), PACKED_TERNARY_LEARN_SAMPLE_BLOCK
            ):
                sample_end = min(
                    int(activity.shape[0]),
                    sample_start + PACKED_TERNARY_LEARN_SAMPLE_BLOCK,
                )
                gradient.add_(
                    downstream[sample_start:sample_end, start:end].float().t()
                    @ activity[sample_start:sample_end].float()
                )
            gradient.div_(float(activity.shape[0]))
            if transaction is not None:
                transaction["weight"][start:end].add_(gradient)
                continue
            changed += _apply_packed_gradient_rows(
                packed,
                self.in_features,
                start,
                gradient,
                rate,
                self._packed_forward_scale,
                generator,
                self._row_stability,
                self._packed_stability_strength,
            )
        if self._packed_forward_bias is not None:
            bias_gradient = torch.empty(
                (self.out_features,), dtype=torch.float32, device=activity.device
            )
            for start in range(0, self.out_features, PACKED_TERNARY_OUTPUT_BLOCK):
                end = min(self.out_features, start + PACKED_TERNARY_OUTPUT_BLOCK)
                block_gradient = torch.zeros(
                    (end - start,), dtype=torch.float32, device=activity.device
                )
                for sample_start in range(
                    0, int(activity.shape[0]), PACKED_TERNARY_LEARN_SAMPLE_BLOCK
                ):
                    sample_end = min(
                        int(activity.shape[0]),
                        sample_start + PACKED_TERNARY_LEARN_SAMPLE_BLOCK,
                    )
                    block_gradient.add_(
                        downstream[sample_start:sample_end, start:end]
                        .float()
                        .sum(dim=0)
                    )
                bias_gradient[start:end] = block_gradient / float(activity.shape[0])
            if transaction is not None:
                transaction["bias"].add_(bias_gradient)
                return 0
            changed += _apply_packed_gradient_rows(
                self._packed_forward_bias,
                self.out_features,
                0,
                bias_gradient[None, :],
                rate,
                self._packed_forward_scale,
                generator,
                self._bias_row_stability,
                self._packed_stability_strength,
            )
            self._bias_validated_version = int(self._packed_forward_bias._version)
        self._packed_validated_version = int(packed._version)
        if changed and self._packed_stability_strength > 0.0:
            self._pending_stability_events += 1
        return changed


@contextmanager
def packed_online_step(
    roots: Sequence[nn.Module], *, max_scratch_synapses: int = 1048576
) -> Iterator[None]:
    """Combine repeated small-head uses into one direct packed update.

    One bounded transient derivative array is allocated per participating
    projection. The total logical size is checked before any allocation. An
    exception discards all pending derivatives; no packed code is changed.
    """
    projections: List[PackedAdaptiveBitLinear] = []
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            if isinstance(module, PackedAdaptiveBitLinear) and id(module) not in seen:
                seen.add(id(module))
                projections.append(module)
    if sum(module.logical_ternary_parameter_count for module in projections) > int(
        max_scratch_synapses
    ):
        raise ValueError("packed online step exceeds bounded scratch allowance")
    begun: List[PackedAdaptiveBitLinear] = []
    try:
        for module in projections:
            module.begin_online_step(max_scratch_synapses=max_scratch_synapses)
            begun.append(module)
        yield
    except BaseException:
        for module in begun:
            module.discard_online_step()
        raise
    snapshots = [
        (
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors()),
            module._row_stability.clone(),
            (
                module._bias_row_stability.clone()
                if module._bias_row_stability is not None
                else None
            ),
            int(module._pending_stability_events),
        )
        for module in begun
    ]
    try:
        for module in begun:
            module.commit_online_step()
    except BaseException:
        with torch.no_grad():
            for module, snapshot in zip(begun, snapshots):
                current = (module._packed_forward_weight,)
                if module._packed_forward_bias is not None:
                    current += (module._packed_forward_bias,)
                for tensor, original in zip(
                    current, snapshot[0]
                ):
                    tensor.copy_(original)
                module._row_stability.copy_(snapshot[1])
                if module._bias_row_stability is not None and snapshot[2] is not None:
                    module._bias_row_stability.copy_(snapshot[2])
                module._pending_stability_events = snapshot[3]
                module._validate_packed()
                module.discard_online_step()
        raise


class _PackedOnlyTernaryEmbedding(torch.autograd.Function):
    """Autograd trigger for sparse, direct packed token-row plasticity."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        indices: torch.Tensor,
        packed: torch.Tensor,
        scale: torch.Tensor,
        module: "PackedAdaptiveTernaryEmbedding",
        autograd_trigger: torch.Tensor,
    ) -> torch.Tensor:
        ctx.module = module
        ctx.save_for_backward(indices)
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                (*indices.shape, module.embedding_dim),
                dtype=scale.dtype,
                device=packed.device,
            )
        selected = packed.index_select(0, flat)
        levels = unpack_ternary_weight_rows(
            selected, module.embedding_dim, validate_reserved=False
        )
        return (
            levels.reshape(*indices.shape, module.embedding_dim).float()
            * scale.detach().float()
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (indices,) = ctx.saved_tensors
        if ctx.module.training:
            ctx.module.learn_from_gradient(
                indices, grad_output, ctx.module.online_learning_rate
            )
        return None, None, None, None, None


class PackedAdaptiveTernaryEmbedding(_PackedRowMetaplasticity, nn.Module):
    """Token rows stored only as mutable packed {-1,0,+1} synapses.

    Forward gathers *only touched rows*; backward sorts token ids and updates
    each touched row from its accumulated gradient. No vocabulary-sized dense
    gradient, floating master, or optimizer moments are held. The scale and
    learning-rate scalars are not learned synapses.
    """

    ternary = True
    ternary_eligible = True
    has_bias = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        *,
        scale: float = 0.025,
        online_learning_rate: float = 1.0,
        initial_level: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        if self.num_embeddings <= 0 or self.embedding_dim <= 0:
            raise ValueError("packed embedding dimensions must be positive")
        self.padding_idx = None if padding_idx is None else int(padding_idx)
        if self.padding_idx is not None and not (0 <= self.padding_idx < self.num_embeddings):
            raise ValueError("packed embedding padding_idx is out of range")
        gain = float(scale)
        rate = float(online_learning_rate)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("packed embedding scale must be finite and positive")
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        if initial_level is not None and (
            isinstance(initial_level, bool) or initial_level not in (-1, 0, 1)
        ):
            raise ValueError("packed embedding initial level must be -1, 0, or +1")
        self.register_buffer(
            "_packed_forward_weight",
            torch.empty(
                (self.num_embeddings, (self.embedding_dim + 3) // 4),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "_packed_forward_scale", torch.tensor(gain, dtype=torch.float32)
        )
        self.register_buffer(
            "_online_learning_rate", torch.tensor(rate, dtype=torch.float32)
        )
        self.register_buffer(
            "_autograd_trigger", torch.zeros((), requires_grad=True), persistent=False
        )
        self._init_packed_stability(self.num_embeddings, False)
        self._packed_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device: Optional[torch.device] = None
        with torch.no_grad():
            if initial_level == 0:
                self._packed_forward_weight.fill_(0x55)
            else:
                for start in range(0, self.num_embeddings, PACKED_TERNARY_OUTPUT_BLOCK):
                    end = min(self.num_embeddings, start + PACKED_TERNARY_OUTPUT_BLOCK)
                    levels = (
                        torch.randint(
                            -1, 2, (end - start, self.embedding_dim), dtype=torch.int8
                        )
                        if initial_level is None
                        else torch.full(
                            (end - start, self.embedding_dim),
                            int(initial_level), dtype=torch.int8,
                        )
                    )
                    if self.padding_idx is not None and start <= self.padding_idx < end:
                        levels[self.padding_idx - start].zero_()
                    self._packed_forward_weight[start:end].copy_(
                        pack_ternary_weight(levels)
                    )
        self._validate_packed()

    def _apply(self, fn):
        result = super()._apply(fn)
        self._packed_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device = None
        return result

    @property
    def online_learning_rate(self) -> float:
        return float(self._online_learning_rate.item())

    @online_learning_rate.setter
    def online_learning_rate(self, value: float) -> None:
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        self._online_learning_rate.fill_(rate)

    @property
    def ternary_weight_shape(self) -> Tuple[int, ...]:
        return self.num_embeddings, self.embedding_dim

    @property
    def ternary_bias_shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def logical_ternary_parameter_count(self) -> int:
        return self.num_embeddings * self.embedding_dim

    def authoritative_packed_tensors(self) -> Tuple[torch.Tensor, ...]:
        return (self.packed_forward_weight(),)

    def _validate_packed(self) -> None:
        packed = self._packed_forward_weight
        if packed.dtype != torch.uint8 or packed.shape != (
            self.num_embeddings, (self.embedding_dim + 3) // 4
        ):
            raise ValueError("packed embedding synapse shape or dtype is invalid")
        for start in range(0, self.num_embeddings, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.num_embeddings, start + PACKED_TERNARY_OUTPUT_BLOCK)
            block = packed[start:end]
            codes = torch.stack(
                tuple((block >> shift) & 0x03 for shift in (0, 2, 4, 6)),
                dim=-1,
            ).reshape(end - start, -1)
            if bool((codes[:, : self.embedding_dim] == 3).any()):
                raise ValueError("packed embedding contains a reserved code")
            if bool((codes[:, self.embedding_dim :] != 1).any()):
                raise ValueError("packed embedding has nonzero row padding")
        if self.padding_idx is not None:
            padding = unpack_ternary_weight_rows(
                packed, self.embedding_dim, self.padding_idx, self.padding_idx + 1,
                validate_reserved=False,
            )
            if bool((padding != 0).any()):
                raise ValueError("packed embedding padding row must remain zero")
        scale = self._packed_forward_scale
        if (
            scale.numel() != 1
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
            or scale.device != packed.device
        ):
            raise ValueError("packed embedding scale is invalid")
        self._validate_packed_stability(self.num_embeddings, packed.device)
        self._packed_validated_version = int(packed._version)
        self._scale_validated_version = int(scale._version)
        self._validated_device = packed.device

    def packed_forward_weight(self) -> torch.Tensor:
        if (
            self._packed_validated_version != int(self._packed_forward_weight._version)
            or self._scale_validated_version != int(self._packed_forward_scale._version)
            or self._packed_stability_changed()
            or self._validated_device != self._packed_forward_weight.device
        ):
            self._validate_packed()
        return self._packed_forward_weight

    def effective_weight(self) -> torch.Tensor:
        """Full exact levels for deliberate inspection/export, not forward."""
        return unpack_ternary_weight_rows(
            self.packed_forward_weight(), self.embedding_dim
        )

    @torch.no_grad()
    def fill_ternary_(self, level: int) -> None:
        """Initialize a packed table without creating a floating master."""
        if isinstance(level, bool) or level not in (-1, 0, 1):
            raise ValueError("packed ternary fill level must be -1, 0, or +1")
        packed = self.packed_forward_weight()
        for start in range(0, self.num_embeddings, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.num_embeddings, start + PACKED_TERNARY_OUTPUT_BLOCK)
            values = torch.full(
                (end - start, self.embedding_dim),
                int(level), dtype=torch.int8, device=packed.device,
            )
            if self.padding_idx is not None and start <= self.padding_idx < end:
                values[self.padding_idx - start].zero_()
            packed[start:end].copy_(pack_ternary_weight(values))
        self._row_stability.zero_()
        self._pending_stability_events = 0
        self._validate_packed()

    def packed_forward_status(self) -> Dict[str, object]:
        return {
            "format": "omni-packed-adaptive-ternary-embedding",
            "formatVersion": 1,
            "weightLevels": [-1, 0, 1],
            "weightsPerByte": PACKED_TERNARY_WEIGHTS_PER_BYTE,
            "packedBytes": int(self.packed_forward_weight().numel()),
            "authoritativePackedWeight": True,
            "latentMasterLearningState": False,
            "touchedRowsOnly": True,
            **self.packed_stability_status(),
        }

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("packed embedding indices must be integers")
        if bool((indices < 0).any()) or bool((indices >= self.num_embeddings).any()):
            raise IndexError("packed embedding index is out of range")
        if indices.device != self._packed_forward_weight.device:
            raise ValueError("packed embedding indices and synapses must share a device")
        return _PackedOnlyTernaryEmbedding.apply(
            indices.to(torch.long),
            self.packed_forward_weight(),
            self._packed_forward_scale,
            self,
            self._autograd_trigger if self.training else self._autograd_trigger.detach(),
        )

    @torch.no_grad()
    def learn_from_gradient(
        self,
        indices: torch.Tensor,
        grad_output: torch.Tensor,
        learning_rate: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        rate = float(learning_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if grad_output.shape != (*indices.shape, self.embedding_dim):
            raise ValueError("packed embedding gradient shape is invalid")
        if indices.device != self._packed_forward_weight.device or grad_output.device != indices.device:
            raise ValueError("packed embedding learning tensors are on different devices")
        if not bool(torch.isfinite(grad_output).all()):
            raise ValueError("packed embedding learning gradient must be finite")
        if indices.numel() == 0:
            return 0
        if rate == 0:
            return 0
        packed = self.packed_forward_weight()
        flat_ids = indices.detach().reshape(-1).to(torch.long)
        flat_gradient = grad_output.detach().reshape(-1, self.embedding_dim)
        # Full-width gain tables and other consecutive row visits can update
        # in bounded blocks without sorting or one device round trip per row.
        if (
            self.padding_idx is None
            and flat_ids.numel() == self.num_embeddings
            and torch.equal(
                flat_ids,
                torch.arange(self.num_embeddings, device=flat_ids.device),
            )
        ):
            changed = 0
            for start in range(0, self.num_embeddings, PACKED_TERNARY_OUTPUT_BLOCK):
                end = min(self.num_embeddings, start + PACKED_TERNARY_OUTPUT_BLOCK)
                changed += _apply_packed_gradient_rows(
                    packed,
                    self.embedding_dim,
                    start,
                    flat_gradient[start:end].float(),
                    rate,
                    self._packed_forward_scale,
                    generator,
                    self._row_stability,
                    self._packed_stability_strength,
                )
            self._packed_validated_version = int(packed._version)
            if changed and self._packed_stability_strength > 0.0:
                self._pending_stability_events += 1
            return changed
        sorted_ids, order = torch.sort(flat_ids)
        unique_ids, counts = torch.unique_consecutive(
            sorted_ids, return_counts=True
        )
        changed = 0
        offset = 0
        for token_id, count in zip(unique_ids.tolist(), counts.tolist()):
            next_offset = offset + int(count)
            if token_id != self.padding_idx:
                selected = order[offset:next_offset]
                gradient = flat_gradient.index_select(0, selected).float().sum(
                    dim=0, keepdim=True
                )
                changed += _apply_packed_gradient_rows(
                    packed,
                    self.embedding_dim,
                    int(token_id),
                    gradient,
                    rate,
                    self._packed_forward_scale,
                    generator,
                    self._row_stability,
                    self._packed_stability_strength,
                )
            offset = next_offset
        self._packed_validated_version = int(packed._version)
        if changed and self._packed_stability_strength > 0.0:
            self._pending_stability_events += 1
        return changed


PACKED_CONV_POSITION_BLOCK = 64


def _spatial_tuple(value: object, dimensions: int, label: str) -> Tuple[int, ...]:
    if isinstance(value, int):
        result = (int(value),) * dimensions
    elif isinstance(value, (tuple, list)) and len(value) == dimensions:
        result = tuple(int(item) for item in value)
    else:
        raise ValueError("packed convolution %s is invalid" % label)
    return result


def _unravel_spatial(index: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    values = [0] * len(shape)
    remaining = int(index)
    for dimension in range(len(shape) - 1, -1, -1):
        size = int(shape[dimension])
        values[dimension] = remaining % size
        remaining //= size
    return tuple(values)


def _flat_spatial_index(coordinate: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    value = 0
    for current, size in zip(coordinate, shape):
        value = value * int(size) + int(current)
    return value


def _kernel_coordinates(kernel: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    total = math.prod(kernel)
    return [_unravel_spatial(index, kernel) for index in range(total)]


def _standard_padding_pairs(
    padding: object,
    input_shape: Tuple[int, ...],
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    if isinstance(padding, str):
        if padding == "valid":
            return ((0, 0),) * len(kernel)
        if padding != "same" or any(value != 1 for value in stride):
            raise ValueError("packed convolution supports only valid or stride-1 same padding")
        pairs = []
        for size, width, rate in zip(input_shape, kernel, dilation):
            total = max(0, (int(size) - 1) + rate * (width - 1) + 1 - int(size))
            before = total // 2
            pairs.append((before, total - before))
        return tuple(pairs)
    values = _spatial_tuple(padding, len(kernel), "padding")
    if any(value < 0 for value in values):
        raise ValueError("packed convolution padding cannot be negative")
    return tuple((value, value) for value in values)


def _pad_arguments(pairs: Tuple[Tuple[int, int], ...]) -> List[int]:
    values: List[int] = []
    for before, after in reversed(pairs):
        values.extend((before, after))
    return values


def _pad_for_convolution(
    inputs: torch.Tensor,
    pairs: Tuple[Tuple[int, int], ...],
    mode: str,
) -> torch.Tensor:
    if not any(before or after for before, after in pairs):
        return inputs
    arguments = _pad_arguments(pairs)
    return F.pad(inputs, arguments, mode="constant" if mode == "zeros" else mode)


def _undo_convolution_padding(
    gradient: torch.Tensor,
    input_shape: Tuple[int, ...],
    pairs: Tuple[Tuple[int, int], ...],
    mode: str,
) -> torch.Tensor:
    if not any(before or after for before, after in pairs):
        return gradient
    if mode == "zeros":
        slices = tuple(
            slice(before, before + int(size))
            for size, (before, _after) in zip(input_shape, pairs)
        )
        return gradient[(slice(None), slice(None), *slices)]
    with torch.enable_grad():
        source = torch.zeros(
            (gradient.shape[0], gradient.shape[1], *input_shape),
            dtype=gradient.dtype,
            device=gradient.device,
            requires_grad=True,
        )
        padded = _pad_for_convolution(source, pairs, mode)
        restored = torch.autograd.grad(
            padded,
            source,
            grad_outputs=gradient,
            retain_graph=False,
            create_graph=False,
        )[0]
    return restored


def _standard_output_shape(
    padded_shape: Tuple[int, ...],
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[int, ...]:
    result = tuple(
        (size - rate * (width - 1) - 1) // step + 1
        for size, width, step, rate in zip(
            padded_shape, kernel, stride, dilation
        )
    )
    if any(value < 1 for value in result):
        raise ValueError("packed convolution kernel exceeds its padded input")
    return result


def _transpose_output_shape(
    input_shape: Tuple[int, ...],
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    output_padding: Tuple[int, ...],
) -> Tuple[int, ...]:
    result = tuple(
        (size - 1) * step
        - 2 * pad
        + rate * (width - 1)
        + extra
        + 1
        for size, width, step, pad, rate, extra in zip(
            input_shape,
            kernel,
            stride,
            padding,
            dilation,
            output_padding,
        )
    )
    if any(value < 1 for value in result):
        raise ValueError("packed transposed convolution output is empty")
    return result


def _extract_standard_patches(
    padded: torch.Tensor,
    coordinates: Sequence[Tuple[int, ...]],
    channel_start: int,
    channel_end: int,
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for coordinate in coordinates:
        slices = tuple(
            slice(
                position * step,
                position * step + rate * (width - 1) + 1,
                rate,
            )
            for position, width, step, rate in zip(
                coordinate, kernel, stride, dilation
            )
        )
        values.append(
            padded[(slice(None), slice(channel_start, channel_end), *slices)]
            .reshape(padded.shape[0], -1)
        )
    return torch.stack(values, dim=1)


def _packed_standard_conv_forward(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_shape: Tuple[int, ...],
    scale: torch.Tensor,
    stride: Tuple[int, ...],
    padding: object,
    dilation: Tuple[int, ...],
    groups: int,
    padding_mode: str,
) -> torch.Tensor:
    dimensions = inputs.ndim - 2
    kernel = tuple(int(value) for value in weight_shape[2:])
    pairs = _standard_padding_pairs(
        padding,
        tuple(int(value) for value in inputs.shape[2:]),
        kernel,
        stride,
        dilation,
    )
    padded = _pad_for_convolution(inputs, pairs, padding_mode)
    output_shape = _standard_output_shape(
        tuple(int(value) for value in padded.shape[2:]),
        kernel,
        stride,
        dilation,
    )
    batch = int(inputs.shape[0])
    out_channels = int(weight_shape[0])
    in_per_group = int(weight_shape[1])
    out_per_group = out_channels // int(groups)
    row_width = in_per_group * math.prod(kernel)
    total = math.prod(output_shape)
    output = torch.empty(
        (batch, total, out_channels),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    for start in range(0, total, PACKED_CONV_POSITION_BLOCK):
        end = min(total, start + PACKED_CONV_POSITION_BLOCK)
        coordinates = [
            _unravel_spatial(index, output_shape)
            for index in range(start, end)
        ]
        for group in range(int(groups)):
            channel_start = group * in_per_group
            patches = _extract_standard_patches(
                padded,
                coordinates,
                channel_start,
                channel_start + in_per_group,
                kernel,
                stride,
                dilation,
            )
            row_start = group * out_per_group
            projected = _packed_ternary_forward(
                patches,
                packed[row_start : row_start + out_per_group],
                row_width,
                out_per_group,
                scale,
            )
            output[:, start:end, row_start : row_start + out_per_group] = projected
    permutation = (0, dimensions + 1, *range(1, dimensions + 1))
    return output.reshape(batch, *output_shape, out_channels).permute(permutation)


def _packed_transpose_conv_forward(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_shape: Tuple[int, ...],
    scale: torch.Tensor,
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    output_padding: Tuple[int, ...],
) -> torch.Tensor:
    dimensions = inputs.ndim - 2
    kernel = tuple(int(value) for value in weight_shape[2:])
    input_shape = tuple(int(value) for value in inputs.shape[2:])
    output_shape = _transpose_output_shape(
        input_shape, kernel, stride, padding, dilation, output_padding
    )
    kernel_coordinates = _kernel_coordinates(kernel)
    kernel_count = len(kernel_coordinates)
    batch = int(inputs.shape[0])
    in_channels = int(inputs.shape[1])
    in_per_group = in_channels // int(groups)
    out_per_group = int(weight_shape[1])
    out_channels = out_per_group * int(groups)
    input_total = math.prod(input_shape)
    output_total = math.prod(output_shape)
    flat_input = inputs.reshape(batch, in_channels, input_total).transpose(1, 2)
    flat_output = torch.zeros(
        (batch, out_channels, output_total),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    for start in range(0, input_total, PACKED_CONV_POSITION_BLOCK):
        end = min(input_total, start + PACKED_CONV_POSITION_BLOCK)
        coordinates = [
            _unravel_spatial(index, input_shape)
            for index in range(start, end)
        ]
        for group in range(int(groups)):
            channel_start = group * in_per_group
            row_start = group * out_per_group * kernel_count
            projected = _packed_ternary_forward(
                flat_input[
                    :, start:end, channel_start : channel_start + in_per_group
                ],
                packed[
                    row_start : row_start + out_per_group * kernel_count
                ],
                in_per_group,
                out_per_group * kernel_count,
                scale,
            ).reshape(batch, end - start, out_per_group, kernel_count)
            output_channels = slice(
                group * out_per_group, (group + 1) * out_per_group
            )
            for kernel_index, kernel_coordinate in enumerate(kernel_coordinates):
                valid_positions: List[int] = []
                target_indices: List[int] = []
                for local, coordinate in enumerate(coordinates):
                    target = tuple(
                        position * step - pad + offset * rate
                        for position, step, pad, offset, rate in zip(
                            coordinate,
                            stride,
                            padding,
                            kernel_coordinate,
                            dilation,
                        )
                    )
                    if all(
                        0 <= value < size
                        for value, size in zip(target, output_shape)
                    ):
                        valid_positions.append(local)
                        target_indices.append(
                            _flat_spatial_index(target, output_shape)
                        )
                if not valid_positions:
                    continue
                positions = torch.tensor(
                    valid_positions, dtype=torch.long, device=inputs.device
                )
                targets = torch.tensor(
                    target_indices, dtype=torch.long, device=inputs.device
                )
                contribution = projected[:, positions, :, kernel_index].permute(
                    0, 2, 1
                )
                flat_output[:, output_channels].index_add_(
                    2, targets, contribution
                )
    return flat_output.reshape(batch, out_channels, *output_shape)


def _packed_standard_conv_backward(
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    weight_shape: Tuple[int, ...],
    scale: torch.Tensor,
    stride: Tuple[int, ...],
    padding: object,
    dilation: Tuple[int, ...],
    groups: int,
    padding_mode: str,
    *,
    compute_weight_gradient: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if compute_weight_gradient:
        raise RuntimeError("floating convolution weight gradients are retired")
    kernel = tuple(int(value) for value in weight_shape[2:])
    input_shape = tuple(int(value) for value in inputs.shape[2:])
    pairs = _standard_padding_pairs(
        padding, input_shape, kernel, stride, dilation
    )
    padded = _pad_for_convolution(inputs, pairs, padding_mode)
    output_shape = tuple(int(value) for value in grad_output.shape[2:])
    batch = int(inputs.shape[0])
    out_channels = int(weight_shape[0])
    in_per_group = int(weight_shape[1])
    out_per_group = out_channels // int(groups)
    kernel_count = math.prod(kernel)
    row_width = in_per_group * kernel_count
    total = math.prod(output_shape)
    flat_grad_output = grad_output.reshape(batch, out_channels, total).transpose(1, 2)
    grad_padded = torch.zeros_like(padded)
    for start in range(0, total, PACKED_CONV_POSITION_BLOCK):
        end = min(total, start + PACKED_CONV_POSITION_BLOCK)
        coordinates = [
            _unravel_spatial(index, output_shape)
            for index in range(start, end)
        ]
        for group in range(int(groups)):
            input_start = group * in_per_group
            output_start = group * out_per_group
            output_gradient = flat_grad_output[
                :, start:end, output_start : output_start + out_per_group
            ]
            patch_gradient = _packed_ternary_input_gradient(
                output_gradient,
                packed[output_start : output_start + out_per_group],
                row_width,
                out_per_group,
                scale,
            ).reshape(batch, end - start, in_per_group, *kernel)
            for local, coordinate in enumerate(coordinates):
                slices = tuple(
                    slice(
                        position * step,
                        position * step + rate * (width - 1) + 1,
                        rate,
                    )
                    for position, width, step, rate in zip(
                        coordinate, kernel, stride, dilation
                    )
                )
                grad_padded[
                    (slice(None), slice(input_start, input_start + in_per_group), *slices)
                ].add_(patch_gradient[:, local])
    grad_input = _undo_convolution_padding(
        grad_padded, input_shape, pairs, padding_mode
    )
    return grad_input, None


def _gather_transpose_output_gradient(
    grad_output: torch.Tensor,
    coordinates: Sequence[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    output_channels: slice,
    out_per_group: int,
    kernel_coordinates: Sequence[Tuple[int, ...]],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> torch.Tensor:
    batch = int(grad_output.shape[0])
    output_total = math.prod(output_shape)
    flat = grad_output.reshape(batch, grad_output.shape[1], output_total)
    gathered = torch.zeros(
        (batch, len(coordinates), out_per_group, len(kernel_coordinates)),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    for kernel_index, kernel_coordinate in enumerate(kernel_coordinates):
        valid_positions: List[int] = []
        target_indices: List[int] = []
        for local, coordinate in enumerate(coordinates):
            target = tuple(
                position * step - pad + offset * rate
                for position, step, pad, offset, rate in zip(
                    coordinate,
                    stride,
                    padding,
                    kernel_coordinate,
                    dilation,
                )
            )
            if all(
                0 <= value < size for value, size in zip(target, output_shape)
            ):
                valid_positions.append(local)
                target_indices.append(_flat_spatial_index(target, output_shape))
        if not valid_positions:
            continue
        positions = torch.tensor(
            valid_positions, dtype=torch.long, device=grad_output.device
        )
        targets = torch.tensor(
            target_indices, dtype=torch.long, device=grad_output.device
        )
        gathered[:, positions, :, kernel_index] = flat[
            :, output_channels
        ].index_select(2, targets).permute(0, 2, 1)
    return gathered


def _packed_transpose_conv_backward(
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    weight_shape: Tuple[int, ...],
    scale: torch.Tensor,
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    *,
    compute_weight_gradient: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if compute_weight_gradient:
        raise RuntimeError("floating convolution weight gradients are retired")
    kernel = tuple(int(value) for value in weight_shape[2:])
    kernel_coordinates = _kernel_coordinates(kernel)
    kernel_count = len(kernel_coordinates)
    input_shape = tuple(int(value) for value in inputs.shape[2:])
    output_shape = tuple(int(value) for value in grad_output.shape[2:])
    batch = int(inputs.shape[0])
    in_channels = int(inputs.shape[1])
    in_per_group = in_channels // int(groups)
    out_per_group = int(weight_shape[1])
    input_total = math.prod(input_shape)
    flat_input = inputs.reshape(batch, in_channels, input_total).transpose(1, 2)
    flat_grad_input = torch.zeros_like(flat_input)
    for start in range(0, input_total, PACKED_CONV_POSITION_BLOCK):
        end = min(input_total, start + PACKED_CONV_POSITION_BLOCK)
        coordinates = [
            _unravel_spatial(index, input_shape)
            for index in range(start, end)
        ]
        for group in range(int(groups)):
            input_start = group * in_per_group
            output_channels = slice(
                group * out_per_group, (group + 1) * out_per_group
            )
            gathered = _gather_transpose_output_gradient(
                grad_output,
                coordinates,
                output_shape,
                output_channels,
                out_per_group,
                kernel_coordinates,
                stride,
                padding,
                dilation,
            )
            flattened = gathered.reshape(
                batch, end - start, out_per_group * kernel_count
            )
            row_start = group * out_per_group * kernel_count
            flat_grad_input[
                :, start:end, input_start : input_start + in_per_group
            ] = _packed_ternary_input_gradient(
                flattened,
                packed[
                    row_start : row_start + out_per_group * kernel_count
                ],
                in_per_group,
                out_per_group * kernel_count,
                scale,
            )
    return (
        flat_grad_input.transpose(1, 2).reshape_as(inputs),
        None,
    )


class _PackedOnlyTernaryConvolution(torch.autograd.Function):
    """Packed convolution autograd with online, blockwise ternary plasticity."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inputs: torch.Tensor,
        packed: torch.Tensor,
        scale: torch.Tensor,
        module: "_PackedAdaptiveConvBase",
        autograd_trigger: torch.Tensor,
    ) -> torch.Tensor:
        ctx.module = module
        ctx.packed = packed
        ctx.scale = scale
        ctx.save_for_backward(inputs)
        if module.transposed:
            output = _packed_transpose_conv_forward(
                inputs,
                packed,
                module._weight_shape,
                scale,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.output_padding,
            )
        else:
            output = _packed_standard_conv_forward(
                inputs,
                packed,
                module._weight_shape,
                scale,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.padding_mode,
            )
        if module.has_bias:
            bias = module.effective_bias()
            if bias is None:
                raise RuntimeError("packed convolution bias is missing")
            output = output + (
                bias.to(output.dtype) * scale.detach().to(output.dtype)
            ).view(1, module.out_channels, *([1] * module.dimensions))
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (inputs,) = ctx.saved_tensors
        module = ctx.module
        if module.transposed:
            grad_input, _ = _packed_transpose_conv_backward(
                inputs,
                grad_output,
                ctx.packed,
                module._weight_shape,
                ctx.scale,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                compute_weight_gradient=False,
            )
        else:
            grad_input, _ = _packed_standard_conv_backward(
                inputs,
                grad_output,
                ctx.packed,
                module._weight_shape,
                ctx.scale,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.padding_mode,
                compute_weight_gradient=False,
            )
        if module.training:
            module.learn_from_gradient(
                inputs, grad_output, module.online_learning_rate
            )
        return grad_input, None, None, None, None


class _PackedAdaptiveConvBase(_PackedRowMetaplasticity, nn.Module):
    """Packed-authoritative convolution shared by all spatial dimensions."""

    ternary = True
    ternary_eligible = True
    dimensions = 0
    transposed = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: object,
        stride: object = 1,
        padding: object = 0,
        dilation: object = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        scale: Optional[float] = None,
        online_learning_rate: float = 1.0,
        _output_padding: object = 0,
    ) -> None:
        super().__init__()
        if self.dimensions not in (1, 2, 3):
            raise ValueError("packed convolution dimensions must be 1, 2, or 3")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        if (
            self.in_channels <= 0
            or self.out_channels <= 0
            or self.groups <= 0
            or self.in_channels % self.groups
            or self.out_channels % self.groups
        ):
            raise ValueError("packed convolution has invalid channel groups")
        self.kernel_size = _spatial_tuple(kernel_size, self.dimensions, "kernel_size")
        self.stride = _spatial_tuple(stride, self.dimensions, "stride")
        self.dilation = _spatial_tuple(dilation, self.dimensions, "dilation")
        self.output_padding = _spatial_tuple(
            _output_padding, self.dimensions, "output_padding"
        )
        if any(value <= 0 for value in (*self.kernel_size, *self.stride, *self.dilation)):
            raise ValueError("packed convolution kernel, stride, and dilation must be positive")
        if any(value < 0 for value in self.output_padding):
            raise ValueError("packed convolution output padding cannot be negative")
        if self.transposed:
            self.padding = _spatial_tuple(padding, self.dimensions, "padding")
            if any(value < 0 for value in self.padding):
                raise ValueError("packed convolution padding cannot be negative")
            if padding_mode != "zeros":
                raise ValueError("packed transposed convolution supports zero padding only")
            if any(extra >= max(step, rate) for extra, step, rate in zip(
                self.output_padding, self.stride, self.dilation
            )):
                raise ValueError("packed output_padding must be less than stride or dilation")
        else:
            self.padding = (
                padding
                if isinstance(padding, str)
                else _spatial_tuple(padding, self.dimensions, "padding")
            )
            if self.padding not in ("valid", "same") and isinstance(self.padding, str):
                raise ValueError("packed convolution padding string is invalid")
            if any(self.output_padding):
                raise ValueError("standard convolution cannot have output_padding")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("packed convolution padding_mode is invalid")
        self.padding_mode = padding_mode
        self.has_bias = bool(bias)
        kernel_count = math.prod(self.kernel_size)
        if self.transposed:
            self._weight_shape = (
                self.in_channels,
                self.out_channels // self.groups,
                *self.kernel_size,
            )
            self._matrix_rows = self.out_channels * kernel_count
            self._matrix_width = self.in_channels // self.groups
        else:
            self._weight_shape = (
                self.out_channels,
                self.in_channels // self.groups,
                *self.kernel_size,
            )
            self._matrix_rows = self.out_channels
            self._matrix_width = (self.in_channels // self.groups) * kernel_count
        initial_scale = (
            math.sqrt(3.0 / (self._matrix_rows + self._matrix_width))
            if scale is None
            else float(scale)
        )
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("packed convolution scale must be finite and positive")
        rate = float(online_learning_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        self.register_buffer(
            "_packed_forward_weight",
            torch.empty(
                (self._matrix_rows, (self._matrix_width + 3) // 4),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "_packed_forward_bias",
            (
                torch.full(
                    (1, (self.out_channels + 3) // 4), 0x55, dtype=torch.uint8
                )
                if self.has_bias
                else None
            ),
        )
        self.register_buffer(
            "_packed_forward_scale", torch.tensor(initial_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "_online_learning_rate", torch.tensor(rate, dtype=torch.float32)
        )
        self.register_buffer(
            "_autograd_trigger", torch.zeros((), requires_grad=True), persistent=False
        )
        self._init_packed_stability(self._matrix_rows, self.has_bias)
        self._packed_validated_version = -1
        self._bias_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device: Optional[torch.device] = None
        with torch.no_grad():
            for start in range(0, self._matrix_rows, PACKED_TERNARY_OUTPUT_BLOCK):
                end = min(self._matrix_rows, start + PACKED_TERNARY_OUTPUT_BLOCK)
                levels = torch.randint(
                    -1, 2, (end - start, self._matrix_width), dtype=torch.int8
                )
                self._packed_forward_weight[start:end].copy_(
                    pack_ternary_weight(levels)
                )
        self._validate_packed()

    def _apply(self, fn):
        result = super()._apply(fn)
        self._packed_validated_version = -1
        self._bias_validated_version = -1
        self._scale_validated_version = -1
        self._validated_device = None
        return result

    @property
    def online_learning_rate(self) -> float:
        return float(self._online_learning_rate.item())

    @online_learning_rate.setter
    def online_learning_rate(self, value: float) -> None:
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("online_learning_rate must be finite and nonnegative")
        self._online_learning_rate.fill_(rate)

    def _validate_packed(self) -> None:
        packed = self._packed_forward_weight
        if packed.dtype != torch.uint8 or packed.shape != (
            self._matrix_rows, (self._matrix_width + 3) // 4
        ):
            raise ValueError("packed convolution synapse shape or dtype is invalid")
        for start in range(0, self._matrix_rows, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self._matrix_rows, start + PACKED_TERNARY_OUTPUT_BLOCK)
            codes = torch.stack(
                tuple((packed[start:end] >> shift) & 0x03 for shift in (0, 2, 4, 6)),
                dim=-1,
            ).reshape(end - start, -1)
            if bool((codes[:, : self._matrix_width] == 3).any()):
                raise ValueError("packed convolution contains a reserved synapse code")
            if bool((codes[:, self._matrix_width :] != 1).any()):
                raise ValueError("packed convolution has nonzero row padding")
        bias = self._packed_forward_bias
        if self.has_bias:
            if bias is None or bias.dtype != torch.uint8 or bias.shape != (
                1, (self.out_channels + 3) // 4
            ):
                raise ValueError("packed convolution bias shape or dtype is invalid")
            if bias.device != packed.device:
                raise ValueError("packed convolution bias is on the wrong device")
            codes = torch.stack(
                tuple((bias >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=-1
            ).reshape(1, -1)
            if bool((codes[:, : self.out_channels] == 3).any()):
                raise ValueError("packed convolution bias contains a reserved code")
            if bool((codes[:, self.out_channels :] != 1).any()):
                raise ValueError("packed convolution bias has nonzero padding")
            self._bias_validated_version = int(bias._version)
        elif bias is not None:
            raise ValueError("packed convolution has an unexpected bias")
        scale = self._packed_forward_scale
        if (
            scale.numel() != 1
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
            or scale.device != packed.device
        ):
            raise ValueError("packed convolution scale is invalid")
        self._validate_packed_stability(self._matrix_rows, packed.device)
        self._packed_validated_version = int(packed._version)
        self._scale_validated_version = int(scale._version)
        self._validated_device = packed.device

    def packed_forward_weight(self) -> torch.Tensor:
        if (
            self._packed_validated_version != int(self._packed_forward_weight._version)
            or self._scale_validated_version != int(self._packed_forward_scale._version)
            or self._packed_stability_changed()
            or self._validated_device != self._packed_forward_weight.device
            or (
                self._packed_forward_bias is not None
                and self._bias_validated_version != int(self._packed_forward_bias._version)
            )
        ):
            self._validate_packed()
        return self._packed_forward_weight

    def authoritative_packed_tensors(self) -> Tuple[torch.Tensor, ...]:
        packed = self.packed_forward_weight()
        if self._packed_forward_bias is None:
            return (packed,)
        return packed, self._packed_forward_bias

    @property
    def logical_ternary_parameter_count(self) -> int:
        return math.prod(self._weight_shape) + (
            self.out_channels if self.has_bias else 0
        )

    @property
    def ternary_weight_shape(self) -> Tuple[int, ...]:
        return self._weight_shape

    @property
    def ternary_bias_shape(self) -> Tuple[int, ...]:
        return (self.out_channels,) if self.has_bias else ()

    def effective_weight(self) -> torch.Tensor:
        matrix = unpack_ternary_weight_rows(
            self.packed_forward_weight(), self._matrix_width
        )
        if not self.transposed:
            return matrix.reshape(self._weight_shape)
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        spatial_axes = tuple(range(2, 2 + self.dimensions))
        return (
            matrix.reshape(self.groups, out_per_group, *self.kernel_size, in_per_group)
            .permute(0, self.dimensions + 2, 1, *spatial_axes)
            .reshape(self._weight_shape)
        )

    def effective_bias(self) -> Optional[torch.Tensor]:
        self.packed_forward_weight()
        if self._packed_forward_bias is None:
            return None
        return unpack_ternary_weight_rows(
            self._packed_forward_bias, self.out_channels
        )[0]

    @property
    def bias(self) -> Optional[torch.Tensor]:
        levels = self.effective_bias()
        if levels is None:
            return None
        return levels.float() * self._packed_forward_scale.float()

    def packed_forward_status(self) -> Dict[str, object]:
        packed = self.packed_forward_weight()
        return {
            "format": "omni-packed-adaptive-ternary-convolution",
            "formatVersion": 1,
            "weightLevels": [-1, 0, 1],
            "weightsPerByte": PACKED_TERNARY_WEIGHTS_PER_BYTE,
            "activationBits": 8,
            "packedBytes": int(packed.numel())
            + (0 if self._packed_forward_bias is None else int(self._packed_forward_bias.numel())),
            "hasPackedBias": self.has_bias,
            "authoritativePackedWeight": True,
            "packedForward": True,
            "boundedPositionBlock": PACKED_CONV_POSITION_BLOCK,
            "denseFloatForwardWeightMaterialized": False,
            "latentMasterLearningState": False,
            **self.packed_stability_status(),
        }

    def forward(
        self, inputs: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        if output_size is not None:
            raise ValueError("explicit transposed-convolution output_size is unsupported")
        if inputs.ndim != self.dimensions + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError("packed convolution input shape is invalid")
        return _PackedOnlyTernaryConvolution.apply(
            inputs,
            self.packed_forward_weight(),
            self._packed_forward_scale,
            self,
            self._autograd_trigger if self.training else self._autograd_trigger.detach(),
        )

    @torch.no_grad()
    def learn_from_gradient(
        self,
        inputs: torch.Tensor,
        grad_output: torch.Tensor,
        learning_rate: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        rate = float(learning_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if inputs.ndim != self.dimensions + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError("packed convolution learning input is invalid")
        if (
            grad_output.ndim != inputs.ndim
            or grad_output.shape[0] != inputs.shape[0]
            or grad_output.shape[1] != self.out_channels
            or grad_output.device != inputs.device
            or inputs.device != self._packed_forward_weight.device
        ):
            raise ValueError("packed convolution learning output is invalid")
        if not bool(torch.isfinite(inputs).all()) or not bool(torch.isfinite(grad_output).all()):
            raise ValueError("packed convolution learning activity must be finite")
        if rate == 0:
            return 0
        packed = self.packed_forward_weight()
        batch = int(inputs.shape[0])
        changed = 0
        if self.transposed:
            input_shape = tuple(int(value) for value in inputs.shape[2:])
            expected = _transpose_output_shape(
                input_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
            )
            if tuple(grad_output.shape[2:]) != expected:
                raise ValueError("packed transposed convolution gradient shape is invalid")
            total = math.prod(input_shape)
            kernel_coordinates = _kernel_coordinates(self.kernel_size)
            kernel_count = len(kernel_coordinates)
            in_per_group = self.in_channels // self.groups
            out_per_group = self.out_channels // self.groups
            flat_input = inputs.reshape(batch, self.in_channels, total).transpose(1, 2)
            for group in range(self.groups):
                input_start = group * in_per_group
                output_channels = slice(
                    group * out_per_group, (group + 1) * out_per_group
                )
                group_row_start = group * out_per_group * kernel_count
                for local_start in range(
                    0, out_per_group * kernel_count, PACKED_TERNARY_OUTPUT_BLOCK
                ):
                    local_end = min(
                        out_per_group * kernel_count,
                        local_start + PACKED_TERNARY_OUTPUT_BLOCK,
                    )
                    gradient = torch.zeros(
                        (local_end - local_start, in_per_group),
                        dtype=torch.float32,
                        device=inputs.device,
                    )
                    for position_start in range(0, total, PACKED_CONV_POSITION_BLOCK):
                        position_end = min(total, position_start + PACKED_CONV_POSITION_BLOCK)
                        coordinates = [
                            _unravel_spatial(index, input_shape)
                            for index in range(position_start, position_end)
                        ]
                        gathered = _gather_transpose_output_gradient(
                            grad_output,
                            coordinates,
                            expected,
                            output_channels,
                            out_per_group,
                            kernel_coordinates,
                            self.stride,
                            self.padding,
                            self.dilation,
                        ).reshape(batch * (position_end - position_start), -1)
                        activity = flat_input[
                            :, position_start:position_end,
                            input_start : input_start + in_per_group,
                        ].reshape(-1, in_per_group)
                        gradient.add_(
                            gathered[:, local_start:local_end].float().t()
                            @ activity.float()
                        )
                    gradient.div_(max(1, batch * total))
                    changed += _apply_packed_gradient_rows(
                        packed,
                        in_per_group,
                        group_row_start + local_start,
                        gradient,
                        rate,
                        self._packed_forward_scale,
                        generator,
                        self._row_stability,
                        self._packed_stability_strength,
                    )
        else:
            input_shape = tuple(int(value) for value in inputs.shape[2:])
            pairs = _standard_padding_pairs(
                self.padding, input_shape, self.kernel_size, self.stride, self.dilation
            )
            padded = _pad_for_convolution(inputs, pairs, self.padding_mode)
            output_shape = _standard_output_shape(
                tuple(int(value) for value in padded.shape[2:]),
                self.kernel_size,
                self.stride,
                self.dilation,
            )
            if tuple(grad_output.shape[2:]) != output_shape:
                raise ValueError("packed convolution gradient shape is invalid")
            total = math.prod(output_shape)
            in_per_group = self.in_channels // self.groups
            out_per_group = self.out_channels // self.groups
            flat_grad = grad_output.reshape(
                batch, self.out_channels, total
            ).transpose(1, 2)
            for group in range(self.groups):
                channel_start = group * in_per_group
                row_start = group * out_per_group
                for start in range(row_start, row_start + out_per_group, PACKED_TERNARY_OUTPUT_BLOCK):
                    end = min(row_start + out_per_group, start + PACKED_TERNARY_OUTPUT_BLOCK)
                    gradient = torch.zeros(
                        (end - start, self._matrix_width),
                        dtype=torch.float32,
                        device=inputs.device,
                    )
                    for position_start in range(0, total, PACKED_CONV_POSITION_BLOCK):
                        position_end = min(total, position_start + PACKED_CONV_POSITION_BLOCK)
                        coordinates = [
                            _unravel_spatial(index, output_shape)
                            for index in range(position_start, position_end)
                        ]
                        patches = _extract_standard_patches(
                            padded,
                            coordinates,
                            channel_start,
                            channel_start + in_per_group,
                            self.kernel_size,
                            self.stride,
                            self.dilation,
                        ).reshape(-1, self._matrix_width)
                        downstream = flat_grad[
                            :, position_start:position_end, start:end
                        ].reshape(-1, end - start)
                        gradient.add_(downstream.float().t() @ patches.float())
                    gradient.div_(max(1, batch * total))
                    changed += _apply_packed_gradient_rows(
                        packed,
                        self._matrix_width,
                        start,
                        gradient,
                        rate,
                        self._packed_forward_scale,
                        generator,
                        self._row_stability,
                        self._packed_stability_strength,
                    )
        if self._packed_forward_bias is not None:
            total = math.prod(tuple(int(value) for value in grad_output.shape[2:]))
            flat = grad_output.reshape(batch, self.out_channels, total)
            bias_gradient = torch.zeros(
                (self.out_channels,), dtype=torch.float32, device=inputs.device
            )
            for start in range(0, self.out_channels, PACKED_TERNARY_OUTPUT_BLOCK):
                end = min(self.out_channels, start + PACKED_TERNARY_OUTPUT_BLOCK)
                for position_start in range(0, total, PACKED_CONV_POSITION_BLOCK):
                    position_end = min(total, position_start + PACKED_CONV_POSITION_BLOCK)
                    bias_gradient[start:end].add_(
                        flat[:, start:end, position_start:position_end]
                        .float()
                        .sum(dim=(0, 2))
                    )
            bias_gradient.div_(max(1, batch * total))
            changed += _apply_packed_gradient_rows(
                self._packed_forward_bias,
                self.out_channels,
                0,
                bias_gradient[None, :],
                rate,
                self._packed_forward_scale,
                generator,
                self._bias_row_stability,
                self._packed_stability_strength,
            )
            self._bias_validated_version = int(self._packed_forward_bias._version)
        self._packed_validated_version = int(packed._version)
        if changed and self._packed_stability_strength > 0.0:
            self._pending_stability_events += 1
        return changed


class PackedAdaptiveBitConv1d(_PackedAdaptiveConvBase):
    dimensions = 1


class PackedAdaptiveBitConv2d(_PackedAdaptiveConvBase):
    dimensions = 2


class PackedAdaptiveBitConv3d(_PackedAdaptiveConvBase):
    dimensions = 3


class _PackedAdaptiveTransposeConvBase(_PackedAdaptiveConvBase):
    transposed = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: object,
        stride: object = 1,
        padding: object = 0,
        output_padding: object = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: object = 1,
        padding_mode: str = "zeros",
        *,
        scale: Optional[float] = None,
        online_learning_rate: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            scale=scale,
            online_learning_rate=online_learning_rate,
            _output_padding=output_padding,
        )


class PackedAdaptiveBitConvTranspose1d(_PackedAdaptiveTransposeConvBase):
    dimensions = 1


class PackedAdaptiveBitConvTranspose2d(_PackedAdaptiveTransposeConvBase):
    dimensions = 2


class PackedAdaptiveBitConvTranspose3d(_PackedAdaptiveTransposeConvBase):
    dimensions = 3


PACKED_AUTHORITATIVE_PROJECTION_TYPES = (
    PackedAdaptiveBitLinear,
    _PackedAdaptiveConvBase,
    PackedAdaptiveTernaryEmbedding,
)


TERNARY_PROJECTION_TYPES = PACKED_AUTHORITATIVE_PROJECTION_TYPES


def packed_runtime_status(module: nn.Module) -> Dict[str, object]:
    packed_linear = 0
    packed_convolution = 0
    authoritative_linear = 0
    authoritative_convolution = 0
    authoritative_embedding = 0
    resident_float_master_blockers: List[str] = []
    dense_linear_blockers: List[str] = []
    dense_convolution_blockers: List[str] = []
    dense_embedding_blockers: List[str] = []
    floating_learned_parameter_blockers = [
        name or "<root>"
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and (parameter.is_floating_point() or parameter.is_complex())
    ]
    dense_bf16_linear_materialized = False
    for name, child in module.named_modules():
        if isinstance(child, PackedAdaptiveTernaryEmbedding):
            authoritative_embedding += 1
        elif isinstance(child, PackedAdaptiveBitLinear):
            packed_linear += 1
            authoritative_linear += 1
        elif isinstance(child, _PackedAdaptiveConvBase):
            packed_convolution += 1
            authoritative_convolution += 1
        elif isinstance(child, nn.Linear):
            dense_linear_blockers.append(name or "<root>")
            dense_bf16_linear_materialized |= child.weight.dtype == torch.bfloat16
        elif isinstance(child, nn.Embedding):
            dense_embedding_blockers.append(name or "<root>")
        elif isinstance(
            child,
            (
                nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            ),
        ):
            dense_convolution_blockers.append(name or "<root>")
    return {
        "format": "omni-native-packed-runtime-audit",
        "formatVersion": 1,
        "complete": not dense_linear_blockers
        and not dense_convolution_blockers
        and not dense_embedding_blockers
        and not resident_float_master_blockers
        and not floating_learned_parameter_blockers,
        "weightFormat": "two-bit-packed-ternary",
        "activationFormat": "signed-int8-dynamic-scale",
        "packedBitLinearModules": packed_linear,
        "packedConvolutionModules": packed_convolution,
        "packedAuthoritativeLinearModules": authoritative_linear,
        "packedAuthoritativeConvolutionModules": authoritative_convolution,
        "packedAuthoritativeEmbeddingModules": authoritative_embedding,
        "residentFloatMasterBlockers": resident_float_master_blockers,
        "floatingLearnedParameterBlockers": floating_learned_parameter_blockers,
        "denseLinearBlockers": dense_linear_blockers,
        "denseConvolutionBlockers": dense_convolution_blockers,
        "denseEmbeddingBlockers": dense_embedding_blockers,
        "denseBf16LinearWeightMaterialized": dense_bf16_linear_materialized,
    }


def require_packed_runtime_complete(module: nn.Module) -> Dict[str, object]:
    status = packed_runtime_status(module)
    if status["complete"] is not True:
        raise RuntimeError(
            "final packed runtime is incomplete: "
            + ", ".join(
                [
                    *status["denseLinearBlockers"],
                    *status["denseConvolutionBlockers"],
                    *status["denseEmbeddingBlockers"],
                    *status["residentFloatMasterBlockers"],
                    *status["floatingLearnedParameterBlockers"],
                ]
            )
        )
    return status


class RMSNorm(nn.Module):
    def __init__(self, dimensions: int, eps: float = 1e-6):
        super().__init__()
        # The identity is fixed; the only adaptive channel gain is an exact
        # packed ternary offset. There is no resident FP32 affine master.
        self.scale_delta = PackedAdaptiveTernaryEmbedding(
            1, dimensions, scale=0.125, initial_level=0
        )
        self.register_buffer(
            "_gain_index", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(rms.to(inputs.dtype) + self.eps)
        delta = self.scale_delta(self._gain_index).squeeze(0)
        return normalized * (1.0 + delta.to(dtype=normalized.dtype))


class PackedAdaptiveTernaryScalar(nn.Module):
    """Small packed adaptive correction to a fixed decoder residual gain."""

    def __init__(self, base: float, *, digits: int = 16, digit_scale: float = 0.025):
        super().__init__()
        if not math.isfinite(float(base)):
            raise ValueError("decoder residual gain base must be finite")
        self.base = float(base)
        self.levels = PackedAdaptiveTernaryEmbedding(
            1, digits, scale=digit_scale, initial_level=0
        )
        self.register_buffer("_index", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self) -> torch.Tensor:
        return torch.tanh(self.base + self.levels(self._index).sum())


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = max(1, int(max_seq_len))
        self.base = float(base)
        inverse = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)
        # A selected context is capacity, not an instruction to reserve the
        # complete future workspace.  RoPE tables grow geometrically from the
        # sequence that is actually executing; attention/KV state is likewise
        # produced only for live tokens.  This keeps multi-million-token
        # configurations representable on smaller hosts without pretending the
        # active cortex was shrunk.
        self.register_buffer(
            "cos", torch.empty((0, inverse.numel()), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sin", torch.empty((0, inverse.numel()), dtype=torch.float32),
            persistent=False,
        )

    @property
    def cached_seq_len(self) -> int:
        return int(self.cos.shape[0])

    def configure_max_seq_len(self, max_seq_len: int) -> None:
        """Change capacity without eagerly materializing positional state."""

        value = max(1, int(max_seq_len))
        self.max_seq_len = value
        if self.cached_seq_len > value:
            # Derived buffers contain no learned state.  Trimming on an
            # explicit capacity decrease releases RAM immediately.
            self.cos = self.cos[:value].contiguous()
            self.sin = self.sin[:value].contiguous()

    def _ensure_cache(self, sequence: int, device: torch.device) -> None:
        sequence = int(sequence)
        if sequence > self.max_seq_len:
            raise ValueError(
                "attention sequence exceeds the configured context capacity"
            )
        if sequence <= self.cached_seq_len and self.cos.device == device:
            return
        current = self.cached_seq_len if self.cos.device == device else 0
        target = max(sequence, max(16, current * 2))
        target = min(self.max_seq_len, target)
        positions = torch.arange(target, dtype=torch.float32, device=device)
        inverse = self.inverse_frequency.to(device=device, dtype=torch.float32)
        angles = torch.outer(positions, inverse)
        self.cos = angles.cos()
        self.sin = angles.sin()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = query.shape[-2]
        start = int(position_offset)
        if start < 0:
            raise ValueError("rotary position offset must be non-negative")
        end = start + sequence
        self._ensure_cache(end, query.device)
        cos = self.cos[start:end].to(dtype=query.dtype)[None, None, :, :]
        sin = self.sin[start:end].to(dtype=query.dtype)[None, None, :, :]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            even = value[..., 0::2]
            odd = value[..., 1::2]
            result = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            )
            return result.flatten(-2)

        return rotate(query), rotate(key)


@dataclass
class _AttentionInferenceCache:
    key: torch.Tensor
    value: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = PackedAdaptiveBitLinear(config.d_model, config.d_model * 3)
        self.output = PackedAdaptiveBitLinear(config.d_model, config.d_model)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = config.dropout
        # Bound live long-context score allocation. This preserves exact
        # causal attention while trading speed for a linear peak working set.
        self.query_chunk_tokens = 256

    def _project(
        self,
        hidden: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch, sequence, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query, key = self.rotary(query, key, position_offset=position_offset)
        return query, key, value

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        batch, _, sequence, _ = query.shape
        attended_chunks = []
        key_positions = torch.arange(key.shape[-2], device=query.device)
        scale = math.sqrt(float(self.head_dim))
        for start in range(0, sequence, self.query_chunk_tokens):
            end = min(sequence, start + self.query_chunk_tokens)
            scores = torch.matmul(
                query[..., start:end, :], key.transpose(-2, -1)
            )
            scores = scores / scale
            query_positions = torch.arange(
                position_offset + start,
                position_offset + end,
                device=query.device,
            )
            mask = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(mask[None, None, :, :], -torch.inf)
            probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
            probabilities = F.dropout(
                probabilities, p=self.dropout, training=self.training
            )
            attended_chunks.append(torch.matmul(probabilities, value))
        attended = torch.cat(attended_chunks, dim=-2)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.n_heads * self.head_dim
        )
        return self.output(attended)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._attend(*self._project(hidden))

    def forward_cached(
        self,
        hidden: torch.Tensor,
        cache: Optional[_AttentionInferenceCache] = None,
    ) -> Tuple[torch.Tensor, _AttentionInferenceCache]:
        """Project only new tokens; cached keys already have their RoPE phase."""

        if self.training or torch.is_grad_enabled():
            raise RuntimeError("attention inference cache requires eval and no_grad")
        position_offset = 0 if cache is None else int(cache.key.shape[-2])
        query, key, value = self._project(hidden, position_offset)
        if cache is not None:
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        else:
            # A split-head value view otherwise retains the full QKV storage
            # from prefill even though only V is needed for future tokens.
            value = value.clone()
        attended = self._attend(query, key, value, position_offset)
        return attended, _AttentionInferenceCache(key, value)


class SwiGLU(nn.Module):
    def __init__(self, dimensions: int, hidden_dimensions: int):
        super().__init__()
        self.up = PackedAdaptiveBitLinear(dimensions, hidden_dimensions * 2)
        self.down = PackedAdaptiveBitLinear(hidden_dimensions, dimensions)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(hidden).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class DecoderBlock(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.d_model)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden

    def forward_cached(
        self,
        hidden: torch.Tensor,
        cache: Optional[_AttentionInferenceCache] = None,
    ) -> Tuple[torch.Tensor, _AttentionInferenceCache]:
        attended, cache = self.attention.forward_cached(
            self.attention_norm(hidden), cache
        )
        hidden = hidden + attended
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden, cache


class TernaryExpert(nn.Module):
    """Small growable residual expert selected by an idea prototype."""

    def __init__(self, dimensions: int, hidden_dimensions: int):
        super().__init__()
        self.norm = RMSNorm(dimensions)
        self.network = SwiGLU(dimensions, hidden_dimensions)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(self.norm(hidden))


def _release_mps_workspace_cache(device: torch.device) -> None:
    """Retain MPSGraph cache lifetime across recomputed slot chunks.

    PyTorch may still have work queued when a tensor operation returns. An
    explicit MPS cache clear can then deallocate a graph referenced by that
    queue and abort the process. Resource admission/chunk sizing bounds this
    workspace; unlike CUDA, MPS cache release is therefore deliberately a
    no-op.
    """

    del device


class _SequentialWorkspaceSummary(torch.autograd.Function):
    """Recompute one exact slot chunk at a time during first-order backward."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        workspace: "GlobalWorkspace",
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.workspace = workspace
        ctx.parameter_count = len(parameters)
        ctx.has_attention_mask = attention_mask is not None
        saved_mask = (
            attention_mask
            if attention_mask is not None
            else torch.empty(0, dtype=torch.bool, device=inputs.device)
        )
        ctx.save_for_backward(inputs, saved_mask, *parameters)
        with torch.no_grad():
            return workspace._ordered_summary(
                inputs,
                attention_mask,
                release_mps_cache=True,
            )

    @staticmethod
    @once_differentiable
    def backward(ctx, summary_gradient: torch.Tensor):
        saved = ctx.saved_tensors
        inputs = saved[0]
        saved_mask = saved[1]
        parameters = saved[2:]
        if len(parameters) != ctx.parameter_count:
            raise RuntimeError("workspace summary parameter count changed")
        attention_mask = (
            saved_mask if ctx.has_attention_mask else None
        )
        workspace = ctx.workspace
        input_needed = bool(ctx.needs_input_grad[0])
        parameter_needed = tuple(
            bool(value) for value in ctx.needs_input_grad[3:]
        )
        input_gradient: Optional[torch.Tensor] = None
        parameter_gradients: List[Optional[torch.Tensor]] = [
            None for _ in parameters
        ]
        scaled_summary_gradient = summary_gradient / float(workspace.slots)
        chunk_slots = max(1, int(workspace.query_chunk_slots))
        # Shared packed projections are reused for every slot chunk. Their
        # weights must stay fixed throughout recomputation or later chunks
        # differentiate a different model. A small workspace accumulates one
        # bounded transient derivative and commits after all chunks. Larger
        # projections remain row-streamed online; that approximation is named
        # in operational state rather than claimed to be exact checkpointing.
        shared_synapses = sum(
            module.logical_ternary_parameter_count
            for module in workspace.modules()
            if isinstance(module, PackedAdaptiveBitLinear)
        )
        exact_transaction = shared_synapses <= 65536
        workspace.last_chunk_backward_mode = (
            "exact-bounded-packed-transaction"
            if exact_transaction
            else "online-streaming-approximate"
        )

        def replay_chunk(start: int) -> None:
            nonlocal input_gradient
            replay_inputs = inputs.detach().requires_grad_(input_needed)
            targets: List[torch.Tensor] = []
            target_indexes: List[Tuple[str, int]] = []
            if input_needed:
                targets.append(replay_inputs)
                target_indexes.append(("input", -1))
            for index, (parameter, needed) in enumerate(
                zip(parameters, parameter_needed)
            ):
                if needed:
                    targets.append(parameter)
                    target_indexes.append(("parameter", index))
            # The packed slot table has no nn.Parameter. Listing its scalar
            # autograd trigger ensures the row-local backward executes during
            # this recomputed chunk; no dense table gradient is requested.
            if workspace.latent_table.training:
                targets.append(workspace.latent_table._autograd_trigger)
                target_indexes.append(("packed-latent", -1))

            with torch.enable_grad():
                contribution = workspace._summary_chunk(
                    replay_inputs,
                    attention_mask,
                    start,
                    min(workspace.slots, start + chunk_slots),
                )
                gradients = torch.autograd.grad(
                    contribution,
                    targets,
                    grad_outputs=scaled_summary_gradient,
                    allow_unused=True,
                    create_graph=False,
                )

            for (kind, index), gradient in zip(
                target_indexes, gradients
            ):
                if gradient is None:
                    continue
                detached = gradient.detach()
                if kind == "input":
                    if input_gradient is None:
                        input_gradient = detached.clone()
                    else:
                        input_gradient.add_(detached)
                elif kind == "packed-latent":
                    continue
                elif parameter_gradients[index] is None:
                    parameter_gradients[index] = detached.clone()
                else:
                    parameter_gradients[index].add_(detached)

            del contribution, gradients, replay_inputs, targets
            del target_indexes
            _release_mps_workspace_cache(inputs.device)

        transaction = (
            packed_online_step((workspace,), max_scratch_synapses=65536)
            if exact_transaction
            else nullcontext()
        )
        with transaction:
            for start in range(0, workspace.slots, chunk_slots):
                replay_chunk(start)

        return (
            input_gradient,
            None,
            None,
            *parameter_gradients,
        )


@dataclass
class _WorkspaceInferenceChunk:
    queries: torch.Tensor
    mass: torch.Tensor
    weighted_values: torch.Tensor


class GlobalWorkspace(nn.Module):
    """Bidirectionally distill a complete input into shared latent slots.

    Language generation remains causal at the token boundary. Before decoding
    a user turn, this module lets all tokens compete for a small shared neural
    workspace, providing a whole-input signal rather than reducing an
    experience to only the last token.
    """

    def __init__(self, dimensions: int, slots: int, iterations: int = 2):
        super().__init__()
        self.dimensions = int(dimensions)
        self.slots = max(4, int(slots))
        self.iterations = max(1, int(iterations))
        self.latent_table = PackedAdaptiveTernaryEmbedding(
            self.slots, self.dimensions
        )
        self.query = PackedAdaptiveBitLinear(dimensions, dimensions)
        self.key = PackedAdaptiveBitLinear(dimensions, dimensions)
        self.value = PackedAdaptiveBitLinear(dimensions, dimensions)
        self.update = PackedAdaptiveBitLinear(dimensions, dimensions)
        self.broadcast = PackedAdaptiveBitLinear(dimensions, dimensions)
        self.norm = RMSNorm(dimensions)
        # A workspace slot attends to the complete token sequence independently
        # of every other slot.  Chunking only this query axis therefore keeps
        # the exact objective while bounding the live score/softmax allocation.
        # During training, a custom first-order backward recomputes and frees
        # one complete chunk at a time instead of retaining all chunk graphs.
        # This execution control is deliberately not serialized neural state.
        self.query_chunk_slots = 256
        self.last_chunk_backward_mode = "not-run"

    def _latent_rows(self, start: int, end: int) -> torch.Tensor:
        """Decode only the active slot rows, not the full learned table."""
        indexes = torch.arange(
            int(start), int(end),
            dtype=torch.long,
            device=self.latent_table.packed_forward_weight().device,
        )
        return self.latent_table(indexes)

    @property
    def latents(self) -> torch.Tensor:
        """Transient all-slot view for explicit whole-workspace inspection."""
        return self._latent_rows(0, self.slots)

    def _attend(
        self,
        latent_chunk: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        queries = self.query(self.norm(latent_chunk))
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.dimensions))
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[:, None, :],
                -torch.inf,
            )
        attention = F.softmax(scores.float(), dim=-1).to(values.dtype)
        return latent_chunk + self.update(torch.matmul(attention, values))

    @staticmethod
    def _validated_attention_mask(
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(
                "global workspace expects [batch, sequence, dimensions]"
            )
        if attention_mask is None:
            return None
        if attention_mask.shape != inputs.shape[:2]:
            raise ValueError(
                "global workspace attention_mask must match [batch, sequence]"
            )
        attention_mask = attention_mask.to(
            device=inputs.device,
            dtype=torch.bool,
        )
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError(
                "global workspace attention_mask needs one token per row"
            )
        return attention_mask

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = self._validated_attention_mask(
            inputs, attention_mask
        )
        batch = inputs.shape[0]
        latents = self._latent_rows(0, self.slots).unsqueeze(0).expand(batch, -1, -1)
        keys = self.key(inputs)
        values = self.value(inputs)
        for _ in range(self.iterations):
            latents = self._attend(
                latents, keys, values, attention_mask
            )
        summary = self.broadcast(self.norm(latents)).mean(dim=1)
        return latents, summary

    def causal_summaries(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        _inference_cache: Optional[List[_WorkspaceInferenceChunk]] = None,
    ) -> torch.Tensor:
        """Distill every prefix with the same learned slots used at inference.

        A language loss cannot train against a bidirectional summary of its
        answer: that would expose future target bytes. Fixed learned slot
        queries let each prefix share one cumulative attention calculation,
        so the last position still integrates the complete live prompt while
        every earlier position is strictly causal. The full iterative
        workspace remains available to ``forward`` and ``summarize`` for
        complete-input encoding.
        """

        if _inference_cache is not None and (
            self.training or torch.is_grad_enabled()
        ):
            raise RuntimeError("workspace inference cache requires eval and no_grad")
        attention_mask = self._validated_attention_mask(inputs, attention_mask)
        keys = self.key(inputs)
        values = self.value(inputs)
        summary_sum: Optional[torch.Tensor] = None
        # A small bounded score range avoids overflow/underflow in a prefix
        # cumulative sum without taking a future-dependent normalization max.
        # It is equivalent to ordinary softmax for scores inside the bound.
        for start in range(0, self.slots, max(1, int(self.query_chunk_slots))):
            latent_chunk = self._latent_rows(
                start, min(self.slots, start + max(1, int(self.query_chunk_slots)))
            )
            queries = self.query(self.norm(latent_chunk))
            scores = torch.matmul(
                queries.unsqueeze(0), keys.transpose(-2, -1)
            ) / math.sqrt(float(self.dimensions))
            weights = scores.float().clamp(-16.0, 16.0).exp()
            if attention_mask is not None:
                weights = weights * attention_mask[:, None, :].to(
                    weights.dtype
                )
            prefix_mass = weights.cumsum(dim=-1)
            weighted_values = (
                weights.unsqueeze(-1) * values[:, None].float()
            ).cumsum(dim=2)
            prefix_values = weighted_values / prefix_mass.clamp_min(
                torch.finfo(weights.dtype).tiny
            ).unsqueeze(-1)
            if _inference_cache is not None:
                # Clone terminal statistics: a view would retain the entire
                # [batch, slots, prefix, dimensions] prefill allocation.
                _inference_cache.append(
                    _WorkspaceInferenceChunk(
                        queries,
                        prefix_mass[:, :, -1].clone(),
                        weighted_values[:, :, -1].clone(),
                    )
                )
            del weighted_values
            updated = latent_chunk[None, :, None, :] + self.update(
                prefix_values.to(inputs.dtype)
            )
            contribution = self.broadcast(self.norm(updated)).sum(dim=1)
            summary_sum = (
                contribution
                if summary_sum is None
                else summary_sum + contribution
            )
        if summary_sum is None:
            raise RuntimeError("global workspace has no latent slots")
        return summary_sum / float(self.slots)

    def causal_step(
        self,
        inputs: torch.Tensor,
        cache: List[_WorkspaceInferenceChunk],
    ) -> torch.Tensor:
        """Continue fixed-query prefix attention from its sufficient statistics.

        This cache holds neural attention mass and weighted activations, not
        text, answers, or a retrieval table. It is valid only for an unchanged
        model and an append-only window; the decoder owns that invalidation.
        """

        if self.training or torch.is_grad_enabled():
            raise RuntimeError("workspace inference cache requires eval and no_grad")
        keys = self.key(inputs)
        values = self.value(inputs)
        summary_sum: Optional[torch.Tensor] = None
        start = 0
        for chunk in cache:
            end = start + int(chunk.queries.shape[0])
            latent_chunk = self._latent_rows(start, end)
            scores = torch.matmul(
                chunk.queries.unsqueeze(0), keys.transpose(-2, -1)
            ) / math.sqrt(float(self.dimensions))
            weights = scores.float().clamp(-16.0, 16.0).exp()
            prefix_mass = weights.cumsum(dim=-1) + chunk.mass.unsqueeze(-1)
            weighted_values = (
                weights.unsqueeze(-1) * values[:, None].float()
            ).cumsum(dim=2) + chunk.weighted_values.unsqueeze(2)
            prefix_values = weighted_values / prefix_mass.clamp_min(
                torch.finfo(weights.dtype).tiny
            ).unsqueeze(-1)
            chunk.mass = prefix_mass[:, :, -1].clone()
            chunk.weighted_values = weighted_values[:, :, -1].clone()
            updated = latent_chunk[None, :, None, :] + self.update(
                prefix_values.to(inputs.dtype)
            )
            contribution = self.broadcast(self.norm(updated)).sum(dim=1)
            summary_sum = (
                contribution
                if summary_sum is None
                else summary_sum + contribution
            )
            start = end
        if summary_sum is None or start != self.slots:
            raise ValueError("workspace inference cache has the wrong slot count")
        return summary_sum / float(self.slots)

    def _summary_chunk(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> torch.Tensor:
        keys = self.key(inputs)
        values = self.value(inputs)
        latent_chunk = self._latent_rows(start, end).unsqueeze(0).expand(
            inputs.shape[0], -1, -1
        )
        for _ in range(self.iterations):
            latent_chunk = self._attend(
                latent_chunk, keys, values, attention_mask
            )
        return self.broadcast(self.norm(latent_chunk)).sum(dim=1)

    def _ordered_summary(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        release_mps_cache: bool = False,
    ) -> torch.Tensor:
        keys = self.key(inputs)
        values = self.value(inputs)
        chunk_slots = max(1, int(self.query_chunk_slots))
        summary_sum: Optional[torch.Tensor] = None
        for start in range(0, self.slots, chunk_slots):
            latent_chunk = self._latent_rows(
                start, min(self.slots, start + chunk_slots)
            ).unsqueeze(0).expand(inputs.shape[0], -1, -1)
            for _ in range(self.iterations):
                latent_chunk = self._attend(
                    latent_chunk, keys, values, attention_mask
                )
            contribution = self.broadcast(
                self.norm(latent_chunk)
            ).sum(dim=1)
            summary_sum = (
                contribution
                if summary_sum is None
                else summary_sum + contribution
            )
            if release_mps_cache:
                del contribution, latent_chunk
                _release_mps_workspace_cache(inputs.device)
        if summary_sum is None:
            raise RuntimeError("global workspace has no latent slots")
        return summary_sum / float(self.slots)

    def summarize(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the exact workspace summary without retaining all slots.

        Workspace slots never attend to one another.  A slot chunk can
        therefore complete every recurrent iteration independently, project
        its final latents, and contribute its ordered sum to the same mean as
        :meth:`forward`.  Training uses a first-order custom autograd path that
        retains only the inputs and parameters, then recomputes, differentiates,
        and releases one complete slot chunk at a time during backward.
        """

        attention_mask = self._validated_attention_mask(
            inputs, attention_mask
        )
        chunk_slots = max(1, int(self.query_chunk_slots))
        parameters = tuple(self.parameters())
        sequential_backward = (
            torch.is_grad_enabled()
            and self.slots > chunk_slots
            and (
                inputs.requires_grad
                or any(parameter.requires_grad for parameter in parameters)
            )
        )
        if sequential_backward:
            return _SequentialWorkspaceSummary.apply(
                inputs,
                attention_mask,
                self,
                *parameters,
            )
        return self._ordered_summary(inputs, attention_mask)


class ActionPolicyHead(nn.Module):
    """Neural policy over conversational and embodied action channels."""

    # A shared inverse temperature sharpens a learned class margin without
    # changing the selected class or assigning a particular action to text.
    # This is architecture normalization, not a learned/floating synapse.
    logit_gain = 2.0

    def __init__(self, dimensions: int):
        super().__init__()
        self.hidden_width = int(dimensions) * 4
        self.norm = RMSNorm(dimensions)
        self.hidden = PackedAdaptiveBitLinear(
            dimensions,
            self.hidden_width,
            bias=True,
            online_learning_rate=4.0,
        )
        self.projection = PackedAdaptiveBitLinear(
            self.hidden_width,
            len(ACTION_KINDS),
            bias=True,
            online_learning_rate=4.0,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        return self.logit_gain * self.projection(
            F.silu(self.hidden(self.norm(hidden)))
        )


class ToolRouteHead(nn.Module):
    """Trainable internal-state/candidate compatibility for typed tool routes.

    Uniform feature hashing describes structural route identities, not chat
    utterances on the deployed path. The text query remains checkpointed for
    older diagnostics but is not trained by new ground-up builds. Cross-entropy
    trains the internal query and candidate ternary projections. The registry
    contains schema identities and numeric features, never utterances, answers,
    or nearest-neighbor keys. Unknown routes and an untrained checkpoint fail
    closed.
    """

    feature_width = 1024
    latent_width = 96

    def __init__(self, state_width: int = 64) -> None:
        super().__init__()
        self.query = PackedAdaptiveBitLinear(self.feature_width, self.latent_width, bias=True)
        self.internal_query = PackedAdaptiveBitLinear(state_width, self.latent_width, bias=True)
        self.candidate = PackedAdaptiveBitLinear(self.feature_width, self.latent_width, bias=False)
        self.register_buffer("route_keys", torch.empty((0, 32), dtype=torch.uint8))
        self.register_buffer("route_features", torch.empty((0, self.feature_width)))
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("internal_training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("experience_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("experience_event_keys", torch.empty((0, 32), dtype=torch.uint8))
        self.register_buffer("null_features", self.encode("<no-action>"))

    @classmethod
    def encode(cls, text: str) -> torch.Tensor:
        """Apply the same domain-agnostic token/character map to every input."""
        words = str(text).casefold().split()
        units = ["word:" + word for word in words]
        for word in words:
            bounded = "^" + word + "$"
            for width in (2, 3, 4):
                units.extend(
                    "gram:" + bounded[index:index + width]
                    for index in range(max(0, len(bounded) - width + 1))
                )
        values = torch.zeros(cls.feature_width, dtype=torch.float32)
        for unit in units:
            digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % cls.feature_width
            values[index] += 1.0 if digest[4] & 1 else -1.0
        return F.normalize(values, dim=0)

    @staticmethod
    def identity(tool_id: str, action: str) -> bytes:
        return hashlib.sha256((tool_id + "\x00" + action).encode("utf-8")).digest()

    def route_index(self, tool_id: str, action: str) -> Optional[int]:
        key = torch.tensor(list(self.identity(tool_id, action)), dtype=torch.uint8,
                           device=self.route_keys.device)
        found = (self.route_keys == key).all(dim=-1).nonzero().flatten()
        return int(found[0].item()) if found.numel() else None

    def register_route(self, tool_id: str, action: str) -> int:
        existing = self.route_index(tool_id, action)
        if existing is not None:
            return existing
        key = torch.tensor([list(self.identity(tool_id, action))], dtype=torch.uint8,
                           device=self.route_keys.device)
        # Identity fields are structural schema values, not tool descriptions.
        features = self.encode(tool_id + " " + action).to(self.route_features.device)
        self.route_keys = torch.cat((self.route_keys, key), dim=0)
        self.route_features = torch.cat((self.route_features, features[None]), dim=0)
        return int(self.route_keys.shape[0]) - 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        queries = F.normalize(torch.tanh(self.query(features)), dim=-1)
        candidates = F.normalize(self.candidate(torch.cat(
            (self.null_features[None], self.route_features), dim=0
        )), dim=-1)
        return 16.0 * (queries @ candidates.t())

    def forward_internal(self, neural_state: torch.Tensor) -> torch.Tensor:
        if neural_state.ndim != 2:
            raise ValueError("internal route state must be batched")
        queries = F.normalize(torch.tanh(self.internal_query(neural_state)), dim=-1)
        candidates = F.normalize(self.candidate(torch.cat(
            (self.null_features[None], self.route_features), dim=0
        )), dim=-1)
        return 16.0 * (queries @ candidates.t())

    @torch.no_grad()
    def select(self, text: str, schemas: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {
            "kind": "trained-neural-schema-route",
            "trainingSteps": int(self.training_steps.item()),
            "experienceUpdates": int(self.experience_updates.item()),
            "featureEncoding": "uniform-hashed-word-and-character-ngrams",
            "keywordRouteRules": False,
            "sourceTextLookup": False,
            "selected": None,
        }
        if not evidence["trainingSteps"]:
            return {**evidence, "reason": "untrained-route-head"}
        enabled: Dict[int, Tuple[str, str]] = {}
        for schema in schemas:
            if str(schema.get("grant", "ask")).lower() == "off":
                continue
            for action in schema.get("actions", ()):
                tool_id = str(schema.get("id", ""))
                index = self.route_index(tool_id, str(action))
                if index is not None:
                    enabled[index + 1] = (tool_id, str(action))
        if not enabled:
            return {**evidence, "reason": "no-trained-enabled-candidate"}
        logits = self(self.encode(text)[None].to(self.route_features.device))[0]
        if not bool(torch.isfinite(logits).all()):
            return {**evidence, "reason": "non-finite-route-head"}
        probabilities = logits.softmax(-1)
        winner = int(probabilities.argmax().item())
        confidence = float(probabilities[winner].item())
        evidence.update({
            "confidence": confidence,
            "minimumConfidence": 0.70,
            "noActionProbability": float(probabilities[0].item()),
            "candidates": [
                {"toolId": pair[0], "action": pair[1],
                 "probability": float(probabilities[index].item())}
                for index, pair in enabled.items()
            ],
        })
        # Disabled routes remain probability competitors: removing a capability
        # must never promote a different action merely by renormalizing it.
        if winner not in enabled or confidence < 0.70:
            return {**evidence, "reason": "no-confident-enabled-route"}
        pair = enabled[winner]
        return {**evidence, "selected": {"toolId": pair[0], "action": pair[1]},
                "reason": "trained-weights"}

    @torch.no_grad()
    def select_internal(
        self, neural_state: torch.Tensor, schemas: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """Choose a route from active neurons, not synthesized focus text."""

        evidence: Dict[str, Any] = {
            "kind": "trained-internal-schema-route",
            "trainingSteps": int(self.internal_training_steps.item()),
            "sourceTextLookup": False,
            "hiddenPrompt": False,
            "selected": None,
        }
        if not evidence["trainingSteps"]:
            return {**evidence, "reason": "untrained-internal-route"}
        enabled: Dict[int, Tuple[str, str]] = {}
        for schema in schemas:
            if str(schema.get("grant", "ask")).lower() == "off":
                continue
            for action in schema.get("actions", ()):
                tool_id = str(schema.get("id", ""))
                index = self.route_index(tool_id, str(action))
                if index is not None:
                    enabled[index + 1] = (tool_id, str(action))
        if not enabled:
            return {**evidence, "reason": "no-trained-enabled-candidate"}
        logits = self.forward_internal(neural_state.to(self.route_features.device))[0]
        if not bool(torch.isfinite(logits).all()):
            return {**evidence, "reason": "non-finite-internal-route"}
        probabilities = logits.softmax(-1)
        winner = int(probabilities.argmax().item())
        confidence = float(probabilities[winner].item())
        evidence.update({"confidence": confidence, "minimumConfidence": 0.70})
        if winner not in enabled or confidence < 0.70:
            return {**evidence, "reason": "no-confident-enabled-route"}
        pair = enabled[winner]
        return {
            **evidence,
            "selected": {"toolId": pair[0], "action": pair[1]},
            "reason": "trained-internal-weights",
        }

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The learned schema registry grows only when independent typed training
        # evidence introduces a route. All state is still tensor-checkpointed.
        for name in ("route_keys", "route_features", "experience_event_keys"):
            value = state_dict.get(prefix + name)
            if isinstance(value, torch.Tensor):
                expected_width = self.feature_width if name == "route_features" else 32
                if value.ndim != 2 or value.shape[1] != expected_width:
                    error_msgs.append(prefix + name + " has invalid route registry shape")
                    continue
                setattr(self, name, torch.empty_like(value, device=self.null_features.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class ActionArgumentHead(nn.Module):
    """Ternary, autoregressive typed-argument decoder in the same checkpoint.

    Its condition is an active neural state plus numeric features of one
    structural tool schema. No utterance, assembly label, tool description,
    answer table, or behavioral prompt is read while decoding. The project
    curriculum can teach JSON syntax, but only confirmed host outcomes may
    enable autonomous decoding of real argument values.
    """

    eos_id = 256
    start_id = 257
    token_count = 258
    output_count = 257

    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = int(dimensions)
        self.condition = PackedAdaptiveBitLinear(
            self.dimensions + ToolRouteHead.feature_width,
            self.dimensions,
            bias=True,
        )
        self.token_embedding = PackedAdaptiveTernaryEmbedding(
            self.token_count, self.dimensions
        )
        self.transition = PackedAdaptiveBitLinear(self.dimensions * 2, self.dimensions, bias=True)
        self.output = PackedAdaptiveBitLinear(self.dimensions, self.output_count, bias=True)
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("grounded_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("route_keys", torch.empty((0, 32), dtype=torch.uint8))
        self.register_buffer("grounded_route_updates", torch.empty((0,), dtype=torch.long))

    def register_route(self, tool_id: str, action: str) -> int:
        key = torch.tensor(
            list(ToolRouteHead.identity(tool_id, action)),
            dtype=torch.uint8, device=self.route_keys.device,
        )
        found = (self.route_keys == key).all(-1).nonzero().flatten()
        if found.numel():
            return int(found[0].item())
        self.route_keys = torch.cat((self.route_keys, key[None]), dim=0)
        self.grounded_route_updates = torch.cat((
            self.grounded_route_updates,
            torch.zeros((1,), dtype=torch.long, device=self.route_keys.device),
        ))
        return int(self.route_keys.shape[0]) - 1

    def grounded_for(self, tool_id: str, action: str) -> int:
        key = torch.tensor(
            list(ToolRouteHead.identity(tool_id, action)),
            dtype=torch.uint8, device=self.route_keys.device,
        )
        found = (self.route_keys == key).all(-1).nonzero().flatten()
        return int(self.grounded_route_updates[int(found[0])].item()) if found.numel() else 0

    @staticmethod
    def schema_features(
        tool_id: str, action: str, schema: Mapping[str, Any]
    ) -> Optional[torch.Tensor]:
        """Hash only field names, types, and required flags into numeric input."""

        if schema.get("id") != tool_id or action not in schema.get("actions", ()):
            return None
        if str(schema.get("grant", "ask")).lower() == "off":
            return None
        by_action = schema.get("actionInputSchemas")
        typed = (
            by_action.get(action)
            if isinstance(by_action, Mapping)
            else schema.get("inputSchema")
        )
        if not isinstance(typed, Mapping):
            return None
        properties = typed.get("properties", {})
        required = typed.get("required", ())
        if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
            return None
        fields = [
            [str(name), str(spec.get("type", "unknown")), str(name) in required]
            for name, spec in sorted(properties.items())
            if isinstance(spec, Mapping)
        ]
        structure = json.dumps(
            [tool_id, action, fields], ensure_ascii=False,
            separators=(",", ":"),
        )
        return ToolRouteHead.encode(structure)

    def forward(
        self,
        neural_state: torch.Tensor,
        schema_features: torch.Tensor,
        prefixes: torch.Tensor,
    ) -> torch.Tensor:
        if neural_state.ndim != 2 or neural_state.shape[-1] != self.dimensions:
            raise ValueError("action argument neural state has invalid shape")
        if (
            schema_features.ndim != 2
            or schema_features.shape != (
                neural_state.shape[0], ToolRouteHead.feature_width
            )
        ):
            raise ValueError("action argument schema features have invalid shape")
        if prefixes.ndim != 2 or prefixes.shape[0] != neural_state.shape[0]:
            raise ValueError("action argument prefix has invalid shape")
        state = torch.tanh(
            self.condition(torch.cat((neural_state, schema_features), dim=-1))
        )
        logits = []
        for index in range(prefixes.shape[1]):
            token = self.token_embedding(prefixes[:, index])
            state = torch.tanh(self.transition(torch.cat((state, token), dim=-1)))
            logits.append(self.output(state))
        return torch.stack(logits, dim=1)

    def supervised_loss(
        self,
        neural_state: torch.Tensor,
        schema_features: torch.Tensor,
        arguments: Mapping[str, Any],
    ) -> torch.Tensor:
        """Teacher-force one typed host/curriculum argument object."""

        encoded = json.dumps(
            dict(arguments), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        if not encoded or len(encoded) > 4096:
            raise ValueError("action argument target exceeds the decode budget")
        device = neural_state.device
        prefixes = torch.tensor(
            [[self.start_id, *encoded]], dtype=torch.long, device=device,
        )
        targets = torch.tensor(
            [[*encoded, self.eos_id]], dtype=torch.long, device=device,
        )
        logits = self(neural_state, schema_features, prefixes)
        return F.cross_entropy(
            logits.float().reshape(-1, self.output_count),
            targets.reshape(-1),
        )

    @torch.no_grad()
    def decode(
        self,
        neural_state: torch.Tensor,
        schema_features: torch.Tensor,
        *,
        tool_id: str,
        action: str,
        max_output_bytes: int = 512,
        minimum_mean_probability: float = 0.85,
    ) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {
            "kind": "ternary-neural-argument-decoder",
            "syntaxTrainingSteps": int(self.training_steps.item()),
            "groundedTrainingSteps": int(self.grounded_steps.item()),
            "groundedRouteUpdates": self.grounded_for(tool_id, action),
            "hiddenPrompt": False,
            "sourceTextLookup": False,
            "arguments": None,
        }
        if evidence["groundedRouteUpdates"] < 1:
            return {**evidence, "reason": "untrained-grounded-route"}
        if not bool(torch.isfinite(neural_state).all()) or not bool(
            torch.isfinite(schema_features).all()
        ):
            return {**evidence, "reason": "non-finite-condition"}
        budget = max(1, min(int(max_output_bytes), 4096))
        state = torch.tanh(
            self.condition(torch.cat((neural_state, schema_features), dim=-1))
        )
        token = torch.tensor([self.start_id], dtype=torch.long, device=neural_state.device)
        generated = bytearray()
        probabilities: List[float] = []
        for _ in range(budget):
            state = torch.tanh(
                self.transition(
                    torch.cat((state, self.token_embedding(token)), dim=-1)
                )
            )
            logits = self.output(state).float()[0]
            if not bool(torch.isfinite(logits).all()):
                return {**evidence, "reason": "non-finite-token-logits"}
            distribution = F.softmax(logits, dim=-1)
            token_id = int(distribution.argmax().item())
            probabilities.append(float(distribution[token_id].item()))
            if token_id == self.eos_id:
                break
            generated.append(token_id)
            token = torch.tensor([token_id], dtype=torch.long, device=neural_state.device)
        else:
            return {**evidence, "reason": "decode-budget-exhausted"}
        mean_probability = sum(probabilities) / len(probabilities)
        evidence["meanTokenProbability"] = mean_probability
        if mean_probability < minimum_mean_probability:
            return {**evidence, "reason": "uncertain-argument-decoding"}
        try:
            parsed = json.loads(bytes(generated).decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError):
            return {**evidence, "reason": "invalid-argument-json"}
        if not isinstance(parsed, dict):
            return {**evidence, "reason": "arguments-not-an-object"}

        def synthetic_placeholder(value: Any) -> bool:
            if isinstance(value, str):
                return value.startswith("<") and value.endswith(">")
            if isinstance(value, list):
                return any(synthetic_placeholder(item) for item in value)
            if isinstance(value, dict):
                return any(synthetic_placeholder(item) for item in value.values())
            return False

        if synthetic_placeholder(parsed):
            return {**evidence, "reason": "curriculum-placeholder-rejected"}
        return {**evidence, "arguments": parsed, "reason": "learned-typed-arguments"}

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        keys = state_dict.get(prefix + "route_keys")
        updates = state_dict.get(prefix + "grounded_route_updates")
        if (
            isinstance(keys, torch.Tensor)
            and isinstance(updates, torch.Tensor)
            and (keys.ndim != 2 or updates.ndim != 1
                 or keys.shape[0] != updates.shape[0])
        ):
            error_msgs.append(prefix + "grounded route registry lengths disagree")
        for name, width in (("route_keys", 32), ("grounded_route_updates", None)):
            value = state_dict.get(prefix + name)
            if isinstance(value, torch.Tensor):
                if (value.ndim != (2 if width else 1)
                        or (width and value.shape[1] != width)):
                    error_msgs.append(prefix + name + " has invalid route registry shape")
                    continue
                setattr(self, name, torch.empty_like(value, device=self.training_steps.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


@dataclass
class _DecoderInferenceCache:
    length: int
    signature: Tuple[Any, ...]
    workspace: List[_WorkspaceInferenceChunk]
    attention: List[_AttentionInferenceCache]
    hidden_sum: torch.Tensor
    memory_hidden: Optional[torch.Tensor]


class OmniDecoder(nn.Module):
    """A compact, from-scratch autoregressive decoder.

    It has no system prompt, preference head, reward model, or pretrained
    component.  Internal idea vectors can bias activations without injecting
    remembered source text into the token stream.
    """

    def __init__(self, config: OmniConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = PackedAdaptiveTernaryEmbedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        # Distillation latents scale with the hardware-resolved recurrent
        # workspace. The old fixed 32-latent ceiling made higher tiers claim
        # more working memory without giving whole-input integration more
        # capacity. Extended memory can expand this population; physical tier
        # selection remains the resource guard.
        workspace_slots = max(8, config.working_memory_slots // 4)
        self.global_workspace = GlobalWorkspace(
            config.d_model,
            slots=workspace_slots,
            iterations=max(1, min(4, config.liquid_steps)),
        )
        self.workspace_strength = PackedAdaptiveTernaryScalar(0.12)
        self.memory_projection = PackedAdaptiveBitLinear(config.idea_dim, config.d_model)
        self.memory_strength = PackedAdaptiveTernaryScalar(0.15)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.language_head = PackedAdaptiveBitLinear(config.d_model, config.vocab_size)
        self.action_policy = ActionPolicyHead(config.d_model)
        self.internal_action_policy = ActionPolicyHead(config.d_model)
        self.experts = nn.ModuleList()
        self.expert_prototypes = nn.ModuleList()
        # Operational metadata only; never serialized in the neural state.
        self.last_generation_cache_mode = "not-started"
        self.tool_route_head = ToolRouteHead(config.d_model)
        self.action_argument_head = ActionArgumentHead(config.d_model)

    @property
    def expert_count(self) -> int:
        return len(self.experts)

    def grow_expert(self, prototype: Optional[torch.Tensor] = None) -> int:
        device = self.embedding.packed_forward_weight().device
        expert = TernaryExpert(self.config.d_model, max(16, self.config.d_ff // 2))
        expert.to(device)
        if prototype is None:
            prototype = torch.randn(self.config.d_model, device=device)
        prototype = prototype.detach().reshape(-1).to(device=device, dtype=torch.float32)
        if prototype.numel() != self.config.d_model or not bool(torch.isfinite(prototype).all()):
            raise ValueError("expert prototype must be a finite decoder-width vector")
        # A sparse signed route approximates the supplied novel activation.
        # The packed projection learns directly from downstream routing loss.
        threshold = prototype.abs().mean() * 0.5
        levels = torch.where(
            prototype.abs() >= threshold,
            prototype.sign(),
            torch.zeros_like(prototype),
        ).to(torch.int8)
        route = PackedAdaptiveBitLinear(
            self.config.d_model,
            1,
            scale=math.sqrt(3.0 / (2.0 * self.config.d_model)),
        ).to(device)
        route.set_ternary_weight_(levels.unsqueeze(0))
        self.experts.append(expert)
        self.expert_prototypes.append(route)
        return len(self.experts) - 1

    def _apply_experts(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        pooled_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.experts:
            return hidden, None
        if pooled_hidden is None:
            if attention_mask is None:
                pooled_hidden = hidden.mean(dim=1)
            else:
                weights = attention_mask.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).unsqueeze(-1)
                pooled_hidden = (hidden * weights).sum(dim=1) / weights.sum(
                    dim=1
                ).clamp_min(1.0)
        pooled = F.normalize(pooled_hidden, dim=-1)
        routing_logits = torch.cat(
            [route(pooled) for route in self.expert_prototypes], dim=-1
        )
        routing = F.softmax(routing_logits, dim=-1)
        residuals = torch.stack(
            [expert(hidden) for expert in self.experts], dim=1
        )
        mixed = (residuals * routing[:, :, None, None]).sum(dim=1)
        return hidden + mixed, routing

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_bias: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_global_workspace: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must match input_ids [batch, sequence]"
                )
            attention_mask = attention_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
            if bool(
                (
                    attention_mask[:, 1:]
                    & ~attention_mask[:, :-1]
                ).any()
            ):
                raise ValueError("attention_mask must describe right padding")
        if input_ids.shape[1] > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len :]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.max_seq_len :]
        if attention_mask is not None and not bool(
            attention_mask.any(dim=1).all()
        ):
            raise ValueError("attention_mask needs one token per row")

        embedded = self.embedding(input_ids)
        hidden = embedded
        workspace_latents = None
        # The default language path and token generation both use the same
        # loss-trained, prefix-causal workspace. An explicit True additionally
        # exposes whole-input latents to complete-input callers, but never
        # injects future tokens into the autoregressive logits.
        if use_global_workspace is not False:
            workspace_summary = self.global_workspace.causal_summaries(
                hidden, attention_mask=attention_mask
            )
            hidden = hidden + self.workspace_strength() * (
                workspace_summary
            )
        if use_global_workspace is True:
            workspace_latents, _ = self.global_workspace(
                embedded, attention_mask=attention_mask
            )
        if memory_bias is not None:
            if memory_bias.ndim == 1:
                memory_bias = memory_bias.unsqueeze(0)
            projected = self.memory_projection(memory_bias).unsqueeze(1)
            hidden = hidden + self.memory_strength() * projected

        for block in self.blocks:
            if self.training and self.config.gradient_checkpointing:
                hidden = checkpoint(
                    block,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden = block(hidden)
        hidden, routing = self._apply_experts(
            hidden,
            attention_mask=attention_mask,
        )
        hidden = self.final_norm(hidden)
        logits = self.language_head(hidden)
        action_logits = self.action_policy(hidden)
        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden": hidden,
            "action_logits": action_logits,
        }
        if workspace_latents is not None:
            output["workspace"] = workspace_latents
        if routing is not None:
            output["expert_routing"] = routing
        if labels is not None:
            if labels.shape[1] < 2:
                raise ValueError("labels need at least two tokens")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=0,
            )
            output["loss"] = loss
        return output

    def encode_whole(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the complete-input workspace summary without decoding."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        input_ids = input_ids[:, -self.config.max_seq_len :]
        return self.global_workspace.summarize(self.embedding(input_ids))

    def _generation_signature(
        self, memory_bias: Optional[torch.Tensor]
    ) -> Optional[Tuple[Any, ...]]:
        """Invalidate local inference state after learning or cue replacement."""

        memory_signature = None
        if memory_bias is not None:
            try:
                memory_signature = (id(memory_bias), int(memory_bias._version))
            except RuntimeError:
                # Inference-mode tensors have no mutation counter. Preserve
                # mutable-cue semantics by using full forwards for this call.
                return None
        parameters = tuple(
            (id(value), int(value._version), value.device, value.dtype)
            for value in self.parameters()
        )
        rotary = tuple(
            (
                id(block.attention.rotary.inverse_frequency),
                int(block.attention.rotary.inverse_frequency._version),
            )
            for block in self.blocks
        )
        return (
            parameters,
            rotary,
            memory_signature,
            int(self.config.max_seq_len),
            int(self.global_workspace.slots),
            int(self.global_workspace.query_chunk_slots),
        )

    def _generation_step(
        self,
        window: torch.Tensor,
        memory_bias: Optional[torch.Tensor],
        cache: Optional[_DecoderInferenceCache],
    ) -> Tuple[torch.Tensor, Optional[_DecoderInferenceCache]]:
        """Return next-token logits with an invocation-local eval-only cache.

        Earlier block states are prefix-causal. Growable experts are applied
        after the blocks and route using the complete prefix, so their route
        must be recomputed from the cached pre-expert hidden sum each step.
        This private cache belongs only to ``generate``'s append-only loop;
        it is not a reusable prefix/session cache for unrelated input tensors.

        Evicting a token changes the workspace summary of *every* retained
        token, in addition to rebasing RoPE positions. Dropping one KV entry
        would therefore be wrong: a shifted full window is always prefetched
        again. Model/cue mutations likewise require a fresh prefill.
        """

        if torch.is_grad_enabled():
            raise RuntimeError("decoder inference cache requires no_grad")
        if (
            self.training
            or self.global_workspace.training
            or any(block.attention.training for block in self.blocks)
        ):
            return self.forward(window, memory_bias=memory_bias)["logits"][:, -1], None
        signature = self._generation_signature(memory_bias)
        if signature is None:
            return self.forward(window, memory_bias=memory_bias)["logits"][:, -1], None
        if window.ndim != 2 or window.shape[1] < 1:
            raise ValueError("generation input must be a non-empty [batch, sequence]")
        window = window[:, -self.config.max_seq_len :]
        length = int(window.shape[1])
        if (
            cache is not None
            and (
                cache.signature != signature
                or cache.length + 1 != length
                or cache.hidden_sum.shape[0] != window.shape[0]
            )
        ):
            cache = None
        hidden = self.embedding(window if cache is None else window[:, -1:])
        if cache is None:
            workspace_cache: List[_WorkspaceInferenceChunk] = []
            summary = self.global_workspace.causal_summaries(
                hidden, _inference_cache=workspace_cache
            )
            memory_hidden = None
            if memory_bias is not None:
                cue = memory_bias.unsqueeze(0) if memory_bias.ndim == 1 else memory_bias
                memory_hidden = self.memory_strength() * (
                    self.memory_projection(cue).unsqueeze(1)
                )
        else:
            workspace_cache = cache.workspace
            summary = self.global_workspace.causal_step(hidden, workspace_cache)
            memory_hidden = cache.memory_hidden
        hidden = hidden + self.workspace_strength() * summary
        if memory_hidden is not None:
            hidden = hidden + memory_hidden
        attention_cache: List[_AttentionInferenceCache] = []
        for index, block in enumerate(self.blocks):
            hidden, layer_cache = block.forward_cached(
                hidden, None if cache is None else cache.attention[index]
            )
            attention_cache.append(layer_cache)
        hidden_sum = hidden.sum(dim=1)
        if cache is not None:
            hidden_sum = cache.hidden_sum + hidden_sum
        pooled_hidden = (
            hidden.mean(dim=1) if cache is None else hidden_sum / float(length)
        )
        hidden, _ = self._apply_experts(
            hidden[:, -1:], pooled_hidden=pooled_hidden
        )
        logits = self.language_head(self.final_norm(hidden))[:, -1]
        return logits, _DecoderInferenceCache(
            length,
            signature,
            workspace_cache,
            attention_cache,
            hidden_sum,
            memory_hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        memory_bias: Optional[torch.Tensor] = None,
        max_new_tokens: int = 48,
        temperature: float = 0.9,
        top_k: int = 40,
        noise: float = 0.0,
        seed: int = 0,
        printable_only: bool = True,
        token_callback: Optional[
            Callable[[torch.Tensor, int, float], None]
        ] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        *,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Sample from the decoder with invocation-local inference state.

        Cache arithmetic is mathematically equivalent to a complete-prefix
        forward, but floating-point reduction order can cross A8 rounding
        boundaries. ``use_cache=False`` retains the full-forward numerical
        reference for callers requiring that exact execution path. Neither
        mode changes model weights, sampling policy, or the random generator.
        """

        self.eval()
        self.last_generation_cache_mode = "not-started"
        generated = input_ids
        inference_cache: Optional[_DecoderInferenceCache] = None
        entropies: List[float] = []
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(int(seed))
        cancelled = cancelled or (lambda: False)

        printable = None
        visible = None
        if printable_only:
            byte_values = [9, 10] + list(range(32, 127))
            printable = torch.tensor(
                [value + 3 for value in byte_values],
                dtype=torch.long,
                device=input_ids.device,
            )
            visible = torch.tensor(
                [value + 3 for value in range(33, 127)],
                dtype=torch.long,
                device=input_ids.device,
            )

        for step in range(max(1, int(max_new_tokens))):
            if cancelled():
                break
            window = generated[:, -self.config.max_seq_len :]
            if use_cache:
                logits, inference_cache = self._generation_step(
                    window, memory_bias, inference_cache
                )
                cache_mode = (
                    "incremental-v1"
                    if inference_cache is not None
                    else "full-prefix-fallback"
                )
            else:
                logits = self.forward(window, memory_bias=memory_bias)[
                    "logits"
                ][:, -1]
                cache_mode = "full-prefix-reference"
            if self.last_generation_cache_mode == "not-started":
                self.last_generation_cache_mode = cache_mode
            elif self.last_generation_cache_mode != cache_mode:
                self.last_generation_cache_mode = "mixed"
            if cancelled():
                break
            if noise > 0:
                jitter = torch.randn(
                    logits.shape,
                    generator=generator,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                logits = logits + float(noise) * jitter
            logits = logits / max(float(temperature), 1e-4)

            if printable is not None:
                allowed = visible if step == 0 else printable
                if step > 0:
                    allowed = torch.cat(
                        [
                            printable,
                            torch.tensor(
                                [2], dtype=torch.long, device=input_ids.device
                            ),
                        ]
                    )
                selected = logits.index_select(-1, allowed)
                if top_k > 0 and top_k < selected.shape[-1]:
                    values, indices = torch.topk(selected, top_k, dim=-1)
                    probabilities = F.softmax(values.float(), dim=-1)
                    sample = torch.multinomial(
                        probabilities, 1, generator=generator
                    )
                    token = allowed[indices.gather(-1, sample)]
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-9).log()
                    ).sum(dim=-1)
                else:
                    probabilities = F.softmax(selected.float(), dim=-1)
                    sample = torch.multinomial(
                        probabilities, 1, generator=generator
                    )
                    token = allowed[sample]
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-9).log()
                    ).sum(dim=-1)
            else:
                if top_k > 0 and top_k < logits.shape[-1]:
                    values, indices = torch.topk(logits, top_k, dim=-1)
                    probabilities = F.softmax(values.float(), dim=-1)
                    sample = torch.multinomial(
                        probabilities, 1, generator=generator
                    )
                    token = indices.gather(-1, sample)
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-9).log()
                    ).sum(dim=-1)
                else:
                    probabilities = F.softmax(logits.float(), dim=-1)
                    token = torch.multinomial(
                        probabilities, 1, generator=generator
                    )
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-9).log()
                    ).sum(dim=-1)
            entropies.append(float(entropy.mean().item()))
            generated = torch.cat([generated, token], dim=1)
            if token_callback is not None:
                token_callback(
                    token.detach().cpu(),
                    step,
                    entropies[-1],
                )
            if step > 0 and bool((token == 2).all()):
                break
        return generated, entropies
