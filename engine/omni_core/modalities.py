"""Tiny trainable multimodal perception and imagination baselines.

These modules are intentionally small and randomly initialized.  They prove
the complete learning/generation paths without pretending to provide the
quality of a large pretrained image, audio, or video model.
"""

import math

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import OmniConfig
from .packed_collective_hooks import packed_row_owner
from .media_planning import (
    MediaOutputPlan,
    MediaPlanError,
    MediaResourceDemand,
    MediaResourcePause,
    MediaResourceWatermark,
    axis_positions,
    require_media_resources,
)
from .model import (
    PACKED_TERNARY_OUTPUT_BLOCK,
    PackedAdaptiveBitConv1d as BitConv1d,
    PackedAdaptiveBitConv2d as BitConv2d,
    PackedAdaptiveBitConv3d as BitConv3d,
    PackedAdaptiveBitConvTranspose1d as BitConvTranspose1d,
    PackedAdaptiveBitConvTranspose2d as BitConvTranspose2d,
    PackedAdaptiveBitConvTranspose3d as BitConvTranspose3d,
    PackedAdaptiveBitLinear as BitLinear,
    _apply_packed_gradient_rows,
)


IMAGINATION_MODALITIES = ("image", "audio", "video")


ScaledPreviewCallback = Callable[
    [float, torch.Tensor, Dict[str, Any]], None
]


@dataclass
class ScaledMediaOutput:
    """An actual neural decode plus an explicit non-quality-claiming receipt."""

    tensor: torch.Tensor
    metadata: Dict[str, Any]


class ModalityGenerationCancelled(RuntimeError):
    """Raised between bounded decoder steps when a media job is cancelled."""


def _check_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise ModalityGenerationCancelled("modality generation was cancelled")


def _preview_steps(total_steps: int, maximum_previews: Optional[int]) -> set[int]:
    """Choose evenly spaced real decoder steps, always including the last.

    This changes only how often an already-computed latent is decoded for the
    UI. It never changes diffusion/codec work, tensor shape, output resolution,
    or the final artifact. Keeping this decision inside the decoder also avoids
    doing expensive preview encodes that a low-end machine should not publish.
    """

    total = max(1, int(total_steps))
    maximum = total if maximum_previews is None else max(
        1, min(total, int(maximum_previews))
    )
    if maximum == 1:
        return {total - 1}
    return {
        min(
            total - 1,
            max(0, round(index * (total - 1) / float(maximum - 1))),
        )
        for index in range(maximum)
    }


def _derived_seed(seed: int, *coordinates: int) -> int:
    """Stable independent seed without adding checkpointed parameters."""

    value = int(seed) & 0x7FFFFFFFFFFFFFFF
    for coordinate in coordinates:
        value = (
            value * 6364136223846793005
            + int(coordinate)
            + 1442695040888963407
        ) & 0x7FFFFFFFFFFFFFFF
    return value


def _positioned_idea(
    idea: torch.Tensor,
    *coordinates: float,
) -> torch.Tensor:
    """Bind tile/time position into the existing shared idea representation."""

    dimensions = int(idea.shape[-1])
    indices = torch.arange(
        1,
        dimensions + 1,
        dtype=idea.dtype,
        device=idea.device,
    )
    position = torch.zeros_like(indices)
    for axis, coordinate in enumerate(coordinates):
        frequency = float(axis + 1) * math.pi
        position = position + torch.sin(
            indices * frequency * (float(coordinate) + 1.0)
        )
    position = F.normalize(position.reshape(1, -1), dim=-1)
    if idea.shape[0] != 1:
        position = position.expand(idea.shape[0], -1)
    return F.normalize(0.88 * idea + 0.12 * position, dim=-1)


def _blend_weight(
    length: int,
    overlap: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=dtype)
    edge = min(max(0, overlap), max(0, length // 2))
    if edge:
        ramp = torch.linspace(
            1.0 / float(edge + 1),
            1.0,
            edge,
            device=device,
            dtype=dtype,
        )
        weight[:edge] = ramp
        weight[-edge:] = torch.flip(ramp, dims=(0,))
    return weight


def _ordered_spatial_positions(
    height: int,
    width: int,
    patch: int,
    overlap: int,
) -> list[tuple[int, int]]:
    values = [
        (top, left)
        for top in axis_positions(height, patch, overlap)
        for left in axis_positions(width, patch, overlap)
    ]
    center_y = max(0.0, (height - patch) / 2.0)
    center_x = max(0.0, (width - patch) / 2.0)
    return sorted(
        values,
        key=lambda item: (
            (item[0] - center_y) ** 2 + (item[1] - center_x) ** 2,
            item[0],
            item[1],
        ),
    )


def _should_publish(completed: int, total: int, every: int) -> bool:
    return completed == 1 or completed == total or completed % max(1, every) == 0


class _PackedTernaryTableRead(torch.autograd.Function):
    """Expose a transient table while keeping its learned state packed.

    This is an embedding-style read, not a second floating-point parameter.
    Backward updates the same packed synapses that supplied the forward values.
    """

    @staticmethod
    def forward(ctx, trigger: torch.Tensor, owner: "PackedTernaryTable") -> torch.Tensor:
        ctx.owner = owner
        projection = owner.projection
        device = (projection._native_compute_device
                  if projection._native_core_pager is not None else trigger.device)
        with projection._packed_residency_scope(device):
            levels = projection.effective_weight().transpose(0, 1)
            return levels.float() * projection._packed_forward_scale

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        owner = ctx.owner
        if owner.training:
            owner.learn_from_gradient(gradient)
        return torch.zeros_like(owner.projection._autograd_trigger), None


class PackedTernaryTable(nn.Module):
    """A learned code/position table with one authoritative ternary substrate.

    The wrapped projection owns two-bit packed synapses. A float table exists
    only as an activation for the current forward/backward, never as a learned
    master copy or optimizer moment. This is suitable for VQ codebooks and
    latent positions, whose entries are used like embeddings rather than a
    dense matrix multiplication.
    """

    def __init__(self, rows: int, dimensions: int, *, scale: float) -> None:
        super().__init__()
        self.rows = int(rows)
        self.dimensions = int(dimensions)
        self.projection = BitLinear(
            self.rows, self.dimensions, bias=False, scale=float(scale)
        )

    def forward(self) -> torch.Tensor:
        trigger = self.projection._autograd_trigger
        if not self.training:
            trigger = trigger.detach()
        return _PackedTernaryTableRead.apply(trigger, self)

    @torch.no_grad()
    def learn_from_gradient(self, gradient: torch.Tensor) -> int:
        with self.projection._packed_residency_scope(gradient.device), packed_row_owner(self.projection):
            return self._learn_from_gradient_impl(gradient)

    @torch.no_grad()
    def _learn_from_gradient_impl(self, gradient: torch.Tensor) -> int:
        if tuple(gradient.shape) != (self.rows, self.dimensions):
            raise ValueError("ternary table gradient shape is invalid")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("ternary table gradient must be finite")
        projection = self.projection
        packed = projection.packed_forward_weight()
        if gradient.device != packed.device:
            raise ValueError("ternary table gradient and synapses must share a device")
        rate = projection.online_learning_rate
        if rate == 0.0:
            return 0
        transposed = gradient.detach().float().transpose(0, 1)
        changed = 0
        for start in range(0, self.dimensions, PACKED_TERNARY_OUTPUT_BLOCK):
            end = min(self.dimensions, start + PACKED_TERNARY_OUTPUT_BLOCK)
            changed += _apply_packed_gradient_rows(
                packed,
                self.rows,
                start,
                transposed[start:end],
                rate,
                projection._packed_forward_scale,
                row_stability=projection._row_stability,
                stability_strength=projection._packed_stability_strength,
            )
        projection._packed_validated_version = int(packed._version)
        if changed and projection._packed_stability_strength > 0.0:
            projection._pending_stability_events += 1
        return changed


class VectorQuantizer(nn.Module):
    """Straight-through nearest-neighbour vector quantizer."""

    def __init__(self, codes: int, dimensions: int):
        super().__init__()
        self.codebook = PackedTernaryTable(codes, dimensions, scale=0.08)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim < 3:
            raise ValueError("quantizer expects [batch, channels, ...]")
        batch, channels = latents.shape[:2]
        spatial = latents.shape[2:]
        flat = latents.reshape(batch, channels, -1).transpose(1, 2)
        codebook = self.codebook().to(flat.dtype)
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=-1)[None, None, :]
        )
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, codebook)
        commitment = F.mse_loss(flat, quantized.detach()) + 0.25 * F.mse_loss(
            quantized, flat.detach()
        )
        quantized = flat + (quantized - flat).detach()
        output = quantized.transpose(1, 2).reshape(batch, channels, *spatial)
        return output, commitment


class TinyVisionEncoder(nn.Module):
    def __init__(self, shared_dim: int, channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            BitConv2d(3, channels, 3, stride=2, padding=1),
            nn.SiLU(),
            BitConv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = BitLinear(channels * 2, shared_dim, bias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return F.normalize(self.projection(features), dim=-1)


class TernaryTransformerBlock(nn.Module):
    """Small pre-norm attention block whose projections use BitLinear."""

    def __init__(self, dimensions: int, heads: int = 2):
        super().__init__()
        if dimensions % heads:
            heads = 1
        self.dimensions = dimensions
        self.heads = heads
        self.head_dim = dimensions // heads
        self.norm_attention = nn.LayerNorm(dimensions, elementwise_affine=False)
        self.qkv = BitLinear(dimensions, dimensions * 3, bias=True)
        self.attention_output = BitLinear(dimensions, dimensions, bias=True)
        self.norm_feed_forward = nn.LayerNorm(dimensions, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            BitLinear(dimensions, dimensions * 3, bias=True),
            nn.SiLU(),
            BitLinear(dimensions * 3, dimensions, bias=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        normalized = self.norm_attention(tokens)
        qkv = self.qkv(normalized).view(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention = torch.softmax(
            query @ key.transpose(-1, -2) / self.head_dim**0.5,
            dim=-1,
        )
        mixed = (attention @ value).transpose(1, 2).reshape(
            batch, length, self.dimensions
        )
        tokens = tokens + self.attention_output(mixed)
        return tokens + self.feed_forward(self.norm_feed_forward(tokens))


class TernaryLatentTransformer(nn.Module):
    """Idea- and diffusion-time-conditioned transformer over latent tokens."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        max_tokens: int,
        layers: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.max_tokens = max_tokens
        self.positions = PackedTernaryTable(max_tokens, channels, scale=0.02)
        self.condition = BitLinear(condition_dim, channels, bias=True)
        self.diffusion_time = nn.Sequential(
            BitLinear(1, channels, bias=True),
            nn.SiLU(),
            BitLinear(channels, channels, bias=True),
        )
        self.blocks = nn.ModuleList(
            [TernaryTransformerBlock(channels) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.output = BitLinear(channels, channels, bias=True)

    def forward(
        self,
        latent: torch.Tensor,
        idea: torch.Tensor,
        timestep: float = 0.0,
    ) -> torch.Tensor:
        original_shape = latent.shape
        if latent.ndim < 3:
            raise ValueError("latent transformer expects [batch, channels, ...]")
        batch, channels = latent.shape[:2]
        tokens = latent.reshape(batch, channels, -1).transpose(1, 2)
        if tokens.shape[1] > self.max_tokens:
            raise ValueError("latent token count exceeds transformer capacity")
        time = torch.full(
            (batch, 1),
            float(timestep),
            device=latent.device,
            dtype=latent.dtype,
        )
        tokens = (
            tokens
            + self.positions()[: tokens.shape[1]].to(tokens.dtype)[None, :, :]
            + self.condition(idea)[:, None, :]
            + self.diffusion_time(time)[:, None, :]
        )
        for block in self.blocks:
            tokens = block(tokens)
        output = self.output(self.output_norm(tokens))
        return output.transpose(1, 2).reshape(original_shape)


class LiquidTemporalGate(nn.Module):
    """CfC-like recurrent gate for compressed video frames."""

    def __init__(self, channels: int):
        super().__init__()
        self.proposal = BitLinear(channels * 2, channels, bias=True)
        self.time_constant = BitLinear(channels * 2, channels, bias=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, _, _ = latent.shape
        state = torch.zeros(
            batch, channels, device=latent.device, dtype=latent.dtype
        )
        evolved = []
        for index in range(frames):
            frame = latent[:, :, index]
            observation = frame.mean(dim=(-2, -1))
            joined = torch.cat((observation, state), dim=-1)
            proposal = torch.tanh(self.proposal(joined))
            gate = torch.sigmoid(self.time_constant(joined))
            state = gate * state + (1.0 - gate) * proposal
            evolved.append(
                frame * (0.75 + 0.25 * gate[:, :, None, None])
                + 0.1 * state[:, :, None, None]
            )
        return torch.stack(evolved, dim=2)


class TinyImageImagination(nn.Module):
    """VQ autoencoder with a ternary idea-conditioned latent DiT."""

    def __init__(self, shared_dim: int, image_size: int, channels: int = 16):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_size = image_size // 4
        self.encoder = nn.Sequential(
            BitConv2d(3, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            BitConv2d(channels, channels, 4, stride=2, padding=1),
        )
        self.quantizer = VectorQuantizer(32, channels)
        self.decoder = nn.Sequential(
            BitConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            BitConvTranspose2d(channels, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.idea_projection = BitLinear(shared_dim, channels, bias=True)
        self.encoder_projection = BitLinear(channels, shared_dim, bias=True)
        self.denoiser = TernaryLatentTransformer(
            channels,
            shared_dim,
            self.latent_size * self.latent_size,
        )

    def forward(
        self, images: torch.Tensor, idea: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        latent = self.encoder(images)
        quantized, commitment = self.quantizer(latent)
        timestep = 0.55
        alpha = 1.0 - timestep * 0.72
        noise = torch.randn_like(quantized)
        noisy = alpha**0.5 * quantized + (1.0 - alpha) ** 0.5 * noise
        predicted_noise = self.denoiser(noisy, idea, timestep=timestep)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        denoised = (
            noisy - (1.0 - alpha) ** 0.5 * predicted_noise
        ) / max(alpha**0.5, 1e-4)
        conditioned = (
            0.8 * quantized
            + 0.2 * denoised
            + self.idea_projection(idea)[:, :, None, None]
        )
        reconstructed = self.decoder(conditioned)
        embedding = F.normalize(
            self.encoder_projection(latent.mean(dim=(-2, -1))), dim=-1
        )
        return {
            "reconstruction": reconstructed,
            "embedding": embedding,
            "commitment_loss": commitment,
            "diffusion_loss": diffusion_loss,
            "loss": (
                F.mse_loss(reconstructed, images)
                + 0.1 * commitment
                + 0.05 * diffusion_loss
            ),
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map an image directly into the brain's shared idea space."""

        latent = self.encoder(images)
        return F.normalize(
            self.encoder_projection(latent.mean(dim=(-2, -1))), dim=-1
        )

    def generate(
        self,
        idea: torch.Tensor,
        generator: torch.Generator,
        steps: int = 4,
        preview_callback: Optional[
            Callable[[float, torch.Tensor], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        maximum_previews: Optional[int] = None,
    ) -> torch.Tensor:
        _check_cancelled(cancel_check)
        latent = torch.randn(
            idea.shape[0],
            self.channels,
            self.latent_size,
            self.latent_size,
            generator=generator,
            device=idea.device,
            dtype=idea.dtype,
        )
        condition = self.idea_projection(idea)[:, :, None, None]
        total_steps = max(1, steps)
        preview_steps = _preview_steps(total_steps, maximum_previews)
        for index in range(total_steps):
            _check_cancelled(cancel_check)
            rate = 0.35 / float(index + 1)
            prediction = self.denoiser(
                latent + condition,
                idea,
                timestep=1.0 - index / float(total_steps),
            )
            latent = latent - rate * prediction
            latent = 0.9 * latent + 0.1 * condition
            if preview_callback is not None and index in preview_steps:
                # The visible revision is the actual codebook projection used
                # by this VQ decoder at this diffusion step, not a CSS effect
                # or an interpolated placeholder.
                quantized_preview, _commitment = self.quantizer(latent)
                preview_callback(
                    float(index + 1) / float(total_steps),
                    self.decoder(quantized_preview).clamp(-1.0, 1.0),
                )
            _check_cancelled(cancel_check)
        _check_cancelled(cancel_check)
        quantized, _commitment = self.quantizer(latent)
        return self.decoder(quantized).clamp(-1.0, 1.0)


class TinyAudioCodec(nn.Module):
    """Residual-vector-quantized codec plus idea-conditioned token generator."""

    def __init__(self, shared_dim: int, samples: int, channels: int = 16):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.latent_samples = samples // 4
        self.encoder = nn.Sequential(
            BitConv1d(1, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            BitConv1d(channels, channels, 4, stride=2, padding=1),
        )
        self.quantizer_a = VectorQuantizer(32, channels)
        self.quantizer_b = VectorQuantizer(16, channels)
        self.decoder = nn.Sequential(
            BitConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            BitConvTranspose1d(channels, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.idea_projection = BitLinear(
            shared_dim, channels * self.latent_samples, bias=True
        )
        self.encoder_projection = BitLinear(channels, shared_dim, bias=True)
        self.token_generator = TernaryLatentTransformer(
            channels, shared_dim, self.latent_samples
        )

    def forward(
        self, waveform: torch.Tensor, idea: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        latent = self.encoder(waveform)
        first, loss_a = self.quantizer_a(latent)
        residual, loss_b = self.quantizer_b(latent - first.detach())
        quantized = first + residual
        predicted_tokens = self.token_generator(quantized, idea, timestep=0.0)
        reconstruction = self.decoder(
            quantized
            + self.idea_projection(idea).view_as(quantized) * 0.1
            + predicted_tokens * 0.05
        )
        embedding = F.normalize(
            self.encoder_projection(latent.mean(dim=-1)), dim=-1
        )
        loss = F.mse_loss(reconstruction, waveform) + 0.05 * (loss_a + loss_b)
        return {
            "reconstruction": reconstruction,
            "embedding": embedding,
            "loss": loss,
        }

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map a bounded waveform directly into the shared idea space."""

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        latent = self.encoder(waveform)
        return F.normalize(
            self.encoder_projection(latent.mean(dim=-1)), dim=-1
        )

    def generate(
        self,
        idea: torch.Tensor,
        generator: torch.Generator,
        preview_callback: Optional[
            Callable[[float, torch.Tensor], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        maximum_previews: Optional[int] = None,
    ) -> torch.Tensor:
        _check_cancelled(cancel_check)
        latent = self.idea_projection(idea).view(
            idea.shape[0], self.channels, self.latent_samples
        )
        noise = torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        )
        latent = latent + 0.18 * noise
        total_steps = 3
        preview_steps = _preview_steps(total_steps, maximum_previews)

        def decode_codec(value: torch.Tensor) -> torch.Tensor:
            first, _loss_a = self.quantizer_a(value)
            residual, _loss_b = self.quantizer_b(value - first.detach())
            return self.decoder(first + residual).squeeze(1)

        for index in range(total_steps):
            _check_cancelled(cancel_check)
            predicted = self.token_generator(
                latent, idea, timestep=1.0 - index / float(total_steps)
            )
            latent = 0.8 * latent + 0.2 * predicted
            if preview_callback is not None and index in preview_steps:
                decoded = decode_codec(latent)
                # Publish only the prefix that has become visible at this
                # codec revision. The WAV therefore grows in real duration;
                # silence is never fabricated for samples not yet presented.
                visible_samples = max(
                    1,
                    min(
                        decoded.shape[-1],
                        math.ceil(
                            decoded.shape[-1]
                            * float(index + 1)
                            / float(total_steps)
                        ),
                    ),
                )
                preview_callback(
                    float(index + 1) / float(total_steps),
                    decoded[..., :visible_samples],
                )
            _check_cancelled(cancel_check)
        _check_cancelled(cancel_check)
        return decode_codec(latent)


class TinyVideoImagination(nn.Module):
    """Factorized spatial/temporal compressed video generator."""

    def __init__(
        self,
        shared_dim: int,
        image_size: int,
        frames: int,
        channels: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.frames = frames
        self.channels = channels
        self.latent_size = image_size // 4
        latent_elements = channels * frames * self.latent_size * self.latent_size
        self.idea_projection = BitLinear(shared_dim, latent_elements, bias=True)
        self.temporal = nn.Sequential(
            BitConv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            BitConv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.spatial = nn.Sequential(
            BitConv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            BitConv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.liquid_gate = LiquidTemporalGate(channels)
        self.decoder = nn.Sequential(
            BitConvTranspose3d(
                channels,
                channels,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            BitConvTranspose3d(
                channels,
                3,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.Tanh(),
        )
        self.encoder = nn.Sequential(
            BitConv3d(
                3,
                channels,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            BitConv3d(
                channels,
                channels,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        self.encoder_projection = BitLinear(channels, shared_dim, bias=True)

    def _condition(self, idea: torch.Tensor) -> torch.Tensor:
        return self.idea_projection(idea).view(
            idea.shape[0],
            self.channels,
            self.frames,
            self.latent_size,
            self.latent_size,
        )

    def _evolve(self, latent: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(latent)
        temporal = self.temporal(latent + 0.25 * spatial)
        return self.liquid_gate(latent + 0.25 * spatial + 0.25 * temporal)

    def forward(
        self, video: torch.Tensor, idea: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        latent = self.encoder(video)
        condition = self._condition(idea)
        timestep = 0.6
        alpha = 1.0 - timestep * 0.7
        noise = torch.randn_like(latent)
        noisy = alpha**0.5 * latent + (1.0 - alpha) ** 0.5 * noise
        predicted_noise = self._evolve(noisy + 0.1 * condition)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        denoised = (
            noisy - (1.0 - alpha) ** 0.5 * predicted_noise
        ) / max(alpha**0.5, 1e-4)
        reconstructed = self.decoder(
            0.8 * latent + 0.2 * denoised + 0.05 * condition
        )
        embedding = F.normalize(
            self.encoder_projection(latent.mean(dim=(2, 3, 4))), dim=-1
        )
        return {
            "reconstruction": reconstructed,
            "embedding": embedding,
            "diffusion_loss": diffusion_loss,
            "loss": F.mse_loss(reconstructed, video) + 0.05 * diffusion_loss,
        }

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Map a bounded frame sequence directly into the shared idea space."""

        latent = self.encoder(video)
        return F.normalize(
            self.encoder_projection(latent.mean(dim=(2, 3, 4))), dim=-1
        )

    def generate(
        self,
        idea: torch.Tensor,
        generator: torch.Generator,
        steps: int = 3,
        preview_callback: Optional[
            Callable[[float, torch.Tensor], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        maximum_previews: Optional[int] = None,
    ) -> torch.Tensor:
        _check_cancelled(cancel_check)
        condition = self._condition(idea)
        latent = torch.randn(
            condition.shape,
            generator=generator,
            device=condition.device,
            dtype=condition.dtype,
        )
        total_steps = max(1, steps)
        preview_steps = _preview_steps(total_steps, maximum_previews)
        for index in range(total_steps):
            _check_cancelled(cancel_check)
            timestep = 1.0 - index / float(total_steps)
            predicted_noise = self._evolve(
                latent + (0.05 + 0.1 * timestep) * condition
            )
            rate = 0.32 / float(index + 1)
            latent = latent - rate * predicted_noise + 0.08 * condition
            if preview_callback is not None and index in preview_steps:
                decoded = self.decoder(latent).clamp(-1.0, 1.0)
                visible_frames = max(
                    1,
                    min(
                        decoded.shape[2],
                        math.ceil(
                            decoded.shape[2]
                            * float(index + 1)
                            / float(total_steps)
                        ),
                    ),
                )
                preview_callback(
                    float(index + 1) / float(total_steps),
                    decoded[:, :, :visible_frames],
                )
            _check_cancelled(cancel_check)
        _check_cancelled(cancel_check)
        return self.decoder(latent).clamp(-1.0, 1.0)


class ModalityHub(nn.Module):
    """Shared-concept-space multimodal module collection."""

    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        channels = config.modality_channels
        self.vision = TinyVisionEncoder(config.idea_dim, channels)
        self.image = TinyImageImagination(
            config.idea_dim, config.image_size, channels
        )
        self.audio = TinyAudioCodec(
            config.idea_dim, config.audio_samples, channels
        )
        self.video = TinyVideoImagination(
            config.idea_dim,
            config.image_size,
            config.video_frames,
            channels,
        )
        # This is part of the same checkpointed OmniCortex, not a prompt
        # classifier or a second model. Its forward projection is mandatory
        # ternary and acquires useful routing only from real media training.
        self.imagination_selector = BitLinear(
            config.idea_dim, len(IMAGINATION_MODALITIES), bias=True
        )

    def imagination_logits(self, idea: torch.Tensor) -> torch.Tensor:
        return self.imagination_selector(idea)

    @torch.no_grad()
    def select_imagination(
        self,
        idea: torch.Tensor,
        enabled: Optional[Iterable[str]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Select media through the learned ternary route, masking unavailable packs."""

        logits = self.imagination_logits(idea).detach().float()[0]
        enabled_set = set(enabled or IMAGINATION_MODALITIES)
        masked = logits.clone()
        for index, name in enumerate(IMAGINATION_MODALITIES):
            if name not in enabled_set:
                masked[index] = -torch.inf
        if not bool(torch.isfinite(masked).any()):
            raise ValueError("no imagination modality is enabled")
        probabilities = torch.softmax(masked, dim=-1)
        selected = int(probabilities.argmax().item())
        return IMAGINATION_MODALITIES[selected], {
            name: float(probabilities[index].item())
            for index, name in enumerate(IMAGINATION_MODALITIES)
        }

    def perception_embedding(
        self,
        modality: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Encode live bounded sensory tensors into the shared neural space."""

        if modality == "image":
            if self.config.vision_enabled:
                return self.vision(tensor)
            if self.config.image_enabled:
                return self.image.encode(tensor)
            raise ValueError("image perception is disabled for this brain")
        if modality == "audio":
            if not self.config.audio_enabled:
                raise ValueError("audio perception is disabled for this brain")
            return self.audio.encode(tensor)
        if modality == "video":
            if not self.config.video_enabled:
                raise ValueError("video perception is disabled for this brain")
            return self.video.encode(tensor)
        raise ValueError("perception modality must be image, audio, or video")

    @torch.no_grad()
    def generate(
        self,
        modality: str,
        idea: torch.Tensor,
        seed: int = 0,
        preview_callback: Optional[
            Callable[[float, torch.Tensor], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        maximum_previews: Optional[int] = None,
    ) -> torch.Tensor:
        _check_cancelled(cancel_check)
        self.eval()
        generator = torch.Generator(device=idea.device)
        generator.manual_seed(int(seed))
        if modality == "image":
            return self.image.generate(
                idea,
                generator,
                preview_callback=preview_callback,
                cancel_check=cancel_check,
                maximum_previews=maximum_previews,
            )
        if modality == "audio":
            return self.audio.generate(
                idea,
                generator,
                preview_callback=preview_callback,
                cancel_check=cancel_check,
                maximum_previews=maximum_previews,
            )
        if modality == "video":
            return self.video.generate(
                idea,
                generator,
                preview_callback=preview_callback,
                cancel_check=cancel_check,
                maximum_previews=maximum_previews,
            )
        raise ValueError("modality must be image, audio, or video")

    @staticmethod
    def _scaled_metadata(
        plan: MediaOutputPlan,
        *,
        training_steps: int,
        installed_pack_id: Optional[str],
        composition: str,
    ) -> Dict[str, Any]:
        trained = training_steps > 0 or bool(installed_pack_id)
        return {
            "format": "omni-scaled-media-output-1",
            "modality": plan.modality,
            "plan": plan.as_dict(),
            "hardwareScaled": True,
            "modelDefinedMaximum": None,
            "nativeWindowPreserved": True,
            "legacyCheckpointCompatible": True,
            "composition": composition,
            "trained": trained,
            "trainingEvidence": {
                "steps": training_steps,
                "installedPackId": installed_pack_id,
            },
            "trainingState": (
                "trained-unverified-quality"
                if trained
                else "untrained-diagnostic"
            ),
            "sizeFloorMet": bool(plan.useful_floor_met),
            # Training and sufficient size are necessary but do not prove
            # prompt alignment or perceptual usefulness.
            "semanticQualityClaimed": False,
        }

    @staticmethod
    def _admit_plan(plan: MediaOutputPlan) -> None:
        if plan.admitted:
            return
        raise MediaResourcePause(
            MediaResourceDemand(
                modality=plan.modality,
                stage="plan-admission",
                completed_units=0,
                total_units=plan.work_units,
                working_bytes=plan.estimated_peak_memory_bytes,
                output_bytes=plan.estimated_peak_storage_bytes,
            )
        )

    @torch.no_grad()
    def generate_scaled(
        self,
        plan: MediaOutputPlan,
        idea: torch.Tensor,
        *,
        seed: int = 0,
        training_steps: int = 0,
        installed_pack_id: Optional[str] = None,
        preview_callback: Optional[ScaledPreviewCallback] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        resource_watermark: Optional[MediaResourceWatermark] = None,
    ) -> ScaledMediaOutput:
        """Compose arbitrary admitted output from checkpoint-native units.

        This adds no parameters and leaves the legacy ``generate`` path
        untouched. Every tile/chunk/window is an actual invocation of the
        trained (or truthfully labelled untrained) same-brain decoder.
        """

        if plan.modality not in IMAGINATION_MODALITIES:
            raise MediaPlanError("scaled media modality is invalid")
        if isinstance(training_steps, bool) or int(training_steps) < 0:
            raise MediaPlanError("modality training steps cannot be negative")
        training_steps = int(training_steps)
        trained = training_steps > 0 or bool(installed_pack_id)

        def truthful_preview(
            progress: float,
            tensor: torch.Tensor,
            detail: Dict[str, Any],
        ) -> None:
            if preview_callback is None:
                return
            preview_callback(
                progress,
                tensor,
                {
                    **detail,
                    "hardwareScaled": True,
                    "modelDefinedMaximum": None,
                    "trained": trained,
                    "trainingState": (
                        "trained-unverified-quality"
                        if trained
                        else "untrained-diagnostic"
                    ),
                    "semanticQualityClaimed": False,
                },
            )

        scaled_preview = truthful_preview if preview_callback is not None else None
        self._admit_plan(plan)
        _check_cancelled(cancel_check)
        require_media_resources(
            resource_watermark,
            self._unit_demand(plan, "output-allocation", 0),
        )
        self.eval()
        if plan.modality == "image":
            tensor = self._generate_scaled_image(
                plan,
                idea,
                seed,
                scaled_preview,
                cancel_check,
                resource_watermark,
            )
            composition = "position-conditioned-overlap-tiles"
        elif plan.modality == "audio":
            tensor = self._generate_scaled_audio(
                plan,
                idea,
                seed,
                scaled_preview,
                cancel_check,
                resource_watermark,
            )
            composition = "recurrent-overlap-add-codec-chunks"
        else:
            tensor = self._generate_scaled_video(
                plan,
                idea,
                seed,
                scaled_preview,
                cancel_check,
                resource_watermark,
            )
            composition = "position-conditioned-rolling-video-tile-windows"
        _check_cancelled(cancel_check)
        return ScaledMediaOutput(
            tensor=tensor,
            metadata=self._scaled_metadata(
                plan,
                training_steps=training_steps,
                installed_pack_id=installed_pack_id,
                composition=composition,
            ),
        )

    def _unit_demand(
        self,
        plan: MediaOutputPlan,
        stage: str,
        completed: int,
    ) -> MediaResourceDemand:
        return MediaResourceDemand(
            modality=plan.modality,
            stage=stage,
            completed_units=completed,
            total_units=plan.work_units,
            working_bytes=plan.estimated_peak_memory_bytes,
            output_bytes=plan.estimated_peak_storage_bytes,
        )

    def _generate_scaled_image(
        self,
        plan: MediaOutputPlan,
        idea: torch.Tensor,
        seed: int,
        preview_callback: Optional[ScaledPreviewCallback],
        cancel_check: Optional[Callable[[], bool]],
        resource_watermark: Optional[MediaResourceWatermark],
    ) -> torch.Tensor:
        if plan.width is None or plan.height is None:
            raise MediaPlanError("image plan is missing width or height")
        patch = int(self.image.image_size)
        positions = _ordered_spatial_positions(
            plan.height,
            plan.width,
            patch,
            plan.spatial_overlap,
        )
        if len(positions) != plan.work_units:
            raise MediaPlanError("image plan work-unit count diverged")
        output = torch.zeros(
            idea.shape[0],
            3,
            plan.height,
            plan.width,
            device=idea.device,
            dtype=idea.dtype,
        )
        weights = torch.zeros(
            1,
            1,
            plan.height,
            plan.width,
            device=idea.device,
            dtype=idea.dtype,
        )
        axis_weight = _blend_weight(
            patch,
            plan.spatial_overlap,
            device=idea.device,
            dtype=idea.dtype,
        )
        patch_weight = axis_weight[:, None] * axis_weight[None, :]
        total = len(positions)
        for index, (top, left) in enumerate(positions):
            _check_cancelled(cancel_check)
            require_media_resources(
                resource_watermark,
                self._unit_demand(plan, "image-tile", index),
            )
            y_position = top / float(max(1, plan.height - patch))
            x_position = left / float(max(1, plan.width - patch))
            positioned = _positioned_idea(idea, y_position, x_position)
            generator = torch.Generator(device=idea.device)
            generator.manual_seed(_derived_seed(seed, top, left))
            tile = self.image.generate(
                positioned,
                generator,
                cancel_check=cancel_check,
            )
            region_height = min(patch, plan.height - top)
            region_width = min(patch, plan.width - left)
            region_weight = patch_weight[:region_height, :region_width]
            output[
                :,
                :,
                top : top + region_height,
                left : left + region_width,
            ] += tile[:, :, :region_height, :region_width] * region_weight
            weights[
                :,
                :,
                top : top + region_height,
                left : left + region_width,
            ] += region_weight
            completed = index + 1
            if preview_callback is not None and _should_publish(
                completed,
                total,
                plan.preview_every_units,
            ):
                covered = float((weights > 0).float().mean().item())
                preview_callback(
                    completed / float(total),
                    (output / weights.clamp_min(1e-8)).clamp(-1.0, 1.0).clone(),
                    {
                        "stage": "diffusion-vq-decode",
                        "scaledStage": "position-conditioned-image-tiles",
                        "completedUnits": completed,
                        "totalUnits": total,
                        "completedTiles": completed,
                        "totalTiles": total,
                        "width": plan.width,
                        "height": plan.height,
                        "coveredFraction": covered,
                        "partialCoverage": completed < total,
                        "actualDecoderOutput": True,
                        "spatialResolutionReduced": False,
                        "semanticQualityClaimed": False,
                    },
                )
            _check_cancelled(cancel_check)
        return (output / weights.clamp_min(1e-8)).clamp(-1.0, 1.0)

    def _generate_scaled_audio(
        self,
        plan: MediaOutputPlan,
        idea: torch.Tensor,
        seed: int,
        preview_callback: Optional[ScaledPreviewCallback],
        cancel_check: Optional[Callable[[], bool]],
        resource_watermark: Optional[MediaResourceWatermark],
    ) -> torch.Tensor:
        if plan.total_samples is None:
            raise MediaPlanError("audio plan is missing total samples")
        chunk_samples = int(self.audio.samples)
        positions = axis_positions(
            plan.total_samples,
            chunk_samples,
            plan.temporal_overlap,
        )
        if len(positions) != plan.work_units:
            raise MediaPlanError("audio plan work-unit count diverged")
        output = torch.zeros(
            idea.shape[0],
            plan.total_samples,
            device=idea.device,
            dtype=idea.dtype,
        )
        weights = torch.zeros(
            1,
            plan.total_samples,
            device=idea.device,
            dtype=idea.dtype,
        )
        chunk_weight = _blend_weight(
            chunk_samples,
            plan.temporal_overlap,
            device=idea.device,
            dtype=idea.dtype,
        )
        carry: Optional[torch.Tensor] = None
        total = len(positions)
        stable_stride = chunk_samples - plan.temporal_overlap
        for index, start in enumerate(positions):
            _check_cancelled(cancel_check)
            require_media_resources(
                resource_watermark,
                self._unit_demand(plan, "audio-chunk", index),
            )
            time_position = start / float(max(1, plan.total_samples - chunk_samples))
            positioned = _positioned_idea(idea, time_position)
            if carry is not None:
                positioned = F.normalize(0.82 * positioned + 0.18 * carry, dim=-1)
            generator = torch.Generator(device=idea.device)
            generator.manual_seed(_derived_seed(seed, start))
            chunk = self.audio.generate(
                positioned,
                generator,
                cancel_check=cancel_check,
            )
            carry = self.audio.encode(chunk.unsqueeze(1)).detach()
            length = min(chunk_samples, plan.total_samples - start)
            region_weight = chunk_weight[:length]
            output[:, start : start + length] += chunk[:, :length] * region_weight
            weights[:, start : start + length] += region_weight
            completed = index + 1
            if preview_callback is not None and _should_publish(
                completed,
                total,
                plan.preview_every_units,
            ):
                stable_end = (
                    plan.total_samples
                    if completed == total
                    else min(plan.total_samples, start + stable_stride)
                )
                preview_callback(
                    completed / float(total),
                    (
                        output[:, :stable_end]
                        / weights[:, :stable_end].clamp_min(1e-8)
                    ).clamp(-1.0, 1.0).clone(),
                    {
                        "stage": "codec-waveform",
                        "scaledStage": "recurrent-overlap-audio-chunks",
                        "completedUnits": completed,
                        "totalUnits": total,
                        "completedChunks": completed,
                        "totalChunks": total,
                        "sampleCount": stable_end,
                        "stablePrefixSamples": stable_end,
                        "totalSamples": plan.total_samples,
                        "sampleRate": plan.sample_rate,
                        "durationMs": (
                            stable_end / float(plan.sample_rate) * 1_000.0
                            if plan.sample_rate
                            else None
                        ),
                        "recurrentCarryUsed": index > 0,
                        "actualDecoderOutput": True,
                        "semanticQualityClaimed": False,
                    },
                )
            _check_cancelled(cancel_check)
        return (output / weights.clamp_min(1e-8)).clamp(-1.0, 1.0)

    def _generate_scaled_video(
        self,
        plan: MediaOutputPlan,
        idea: torch.Tensor,
        seed: int,
        preview_callback: Optional[ScaledPreviewCallback],
        cancel_check: Optional[Callable[[], bool]],
        resource_watermark: Optional[MediaResourceWatermark],
    ) -> torch.Tensor:
        if plan.width is None or plan.height is None or plan.total_frames is None:
            raise MediaPlanError("video plan is missing dimensions or frames")
        patch = int(self.video.image_size)
        window_frames = int(self.video.frames)
        spatial = _ordered_spatial_positions(
            plan.height,
            plan.width,
            patch,
            plan.spatial_overlap,
        )
        temporal = axis_positions(
            plan.total_frames,
            window_frames,
            plan.temporal_overlap,
        )
        if len(spatial) * len(temporal) != plan.work_units:
            raise MediaPlanError("video plan work-unit count diverged")
        output = torch.zeros(
            idea.shape[0],
            3,
            plan.total_frames,
            plan.height,
            plan.width,
            device=idea.device,
            dtype=idea.dtype,
        )
        weights = torch.zeros(
            1,
            1,
            plan.total_frames,
            plan.height,
            plan.width,
            device=idea.device,
            dtype=idea.dtype,
        )
        spatial_axis = _blend_weight(
            patch,
            plan.spatial_overlap,
            device=idea.device,
            dtype=idea.dtype,
        )
        spatial_weight = spatial_axis[:, None] * spatial_axis[None, :]
        temporal_weight = _blend_weight(
            window_frames,
            plan.temporal_overlap,
            device=idea.device,
            dtype=idea.dtype,
        )
        completed = 0
        carry: Optional[torch.Tensor] = None
        stable_stride = window_frames - plan.temporal_overlap
        for temporal_index, frame_start in enumerate(temporal):
            window_embeddings = []
            for top, left in spatial:
                _check_cancelled(cancel_check)
                require_media_resources(
                    resource_watermark,
                    self._unit_demand(plan, "video-tile-window", completed),
                )
                time_position = frame_start / float(
                    max(1, plan.total_frames - window_frames)
                )
                y_position = top / float(max(1, plan.height - patch))
                x_position = left / float(max(1, plan.width - patch))
                positioned = _positioned_idea(
                    idea,
                    time_position,
                    y_position,
                    x_position,
                )
                if carry is not None:
                    positioned = F.normalize(
                        0.84 * positioned + 0.16 * carry,
                        dim=-1,
                    )
                generator = torch.Generator(device=idea.device)
                generator.manual_seed(
                    _derived_seed(seed, frame_start, top, left)
                )
                block = self.video.generate(
                    positioned,
                    generator,
                    cancel_check=cancel_check,
                )
                window_embeddings.append(self.video.encode(block).detach())
                frame_count = min(
                    window_frames,
                    plan.total_frames - frame_start,
                )
                region_height = min(patch, plan.height - top)
                region_width = min(patch, plan.width - left)
                weight = (
                    temporal_weight[:frame_count, None, None]
                    * spatial_weight[None, :region_height, :region_width]
                )
                output[
                    :,
                    :,
                    frame_start : frame_start + frame_count,
                    top : top + region_height,
                    left : left + region_width,
                ] += (
                    block[
                        :,
                        :,
                        :frame_count,
                        :region_height,
                        :region_width,
                    ]
                    * weight
                )
                weights[
                    :,
                    :,
                    frame_start : frame_start + frame_count,
                    top : top + region_height,
                    left : left + region_width,
                ] += weight
                completed += 1
                if preview_callback is not None and _should_publish(
                    completed,
                    plan.work_units,
                    plan.preview_every_units,
                ):
                    covered = float((weights > 0).float().mean().item())
                    visible_frame_end = min(
                        plan.total_frames,
                        frame_start + frame_count,
                    )
                    preview_callback(
                        completed / float(plan.work_units),
                        (
                            output[:, :, :visible_frame_end]
                            / weights[:, :, :visible_frame_end].clamp_min(1e-8)
                        ).clamp(-1.0, 1.0).clone(),
                        {
                            "stage": "temporal-frame-timeline",
                            "scaledStage": "rolling-video-tile-windows",
                            "completedUnits": completed,
                            "totalUnits": plan.work_units,
                            "completedTileWindows": completed,
                            "totalTileWindows": plan.work_units,
                            "completedTemporalWindows": temporal_index,
                            "totalTemporalWindows": len(temporal),
                            "coveredFraction": covered,
                            "partialCoverage": completed < plan.work_units,
                            "frameCount": visible_frame_end,
                            "totalFrames": plan.total_frames,
                            "fps": plan.fps,
                            "durationMs": (
                                visible_frame_end / float(plan.fps) * 1_000.0
                                if plan.fps
                                else None
                            ),
                            "width": plan.width,
                            "height": plan.height,
                            "stableFramePrefix": (
                                plan.total_frames
                                if completed == plan.work_units
                                else min(
                                    plan.total_frames,
                                    frame_start + stable_stride,
                                )
                            ),
                            "recurrentCarryUsed": temporal_index > 0,
                            "actualDecoderOutput": True,
                            "semanticQualityClaimed": False,
                        },
                    )
                _check_cancelled(cancel_check)
            if window_embeddings:
                carry = F.normalize(
                    torch.stack(window_embeddings, dim=0).mean(dim=0),
                    dim=-1,
                )
        return (output / weights.clamp_min(1e-8)).clamp(-1.0, 1.0)
