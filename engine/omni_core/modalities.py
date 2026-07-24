"""Tiny trainable multimodal perception and imagination baselines.

These modules are intentionally small and randomly initialized.  They prove
the complete learning/generation paths without pretending to provide the
quality of a large pretrained image, audio, or video model.
"""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import OmniConfig
from .model import BitLinear


class VectorQuantizer(nn.Module):
    """Straight-through nearest-neighbour vector quantizer."""

    def __init__(self, codes: int, dimensions: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codes, dimensions) * 0.08)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim < 3:
            raise ValueError("quantizer expects [batch, channels, ...]")
        batch, channels = latents.shape[:2]
        spatial = latents.shape[2:]
        flat = latents.reshape(batch, channels, -1).transpose(1, 2)
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=-1)[None, None, :]
        )
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
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
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
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
        self.norm_attention = nn.LayerNorm(dimensions)
        self.qkv = BitLinear(dimensions, dimensions * 3, bias=True)
        self.attention_output = BitLinear(dimensions, dimensions, bias=True)
        self.norm_feed_forward = nn.LayerNorm(dimensions)
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
        self.positions = nn.Parameter(
            torch.randn(1, max_tokens, channels) * 0.02
        )
        self.condition = BitLinear(condition_dim, channels, bias=True)
        self.diffusion_time = nn.Sequential(
            BitLinear(1, channels, bias=True),
            nn.SiLU(),
            BitLinear(channels, channels, bias=True),
        )
        self.blocks = nn.ModuleList(
            [TernaryTransformerBlock(channels) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(channels)
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
            + self.positions[:, : tokens.shape[1]].to(tokens.dtype)
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
            nn.Conv2d(3, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
        )
        self.quantizer = VectorQuantizer(32, channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.idea_projection = BitLinear(shared_dim, channels, bias=True)
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
        return {
            "reconstruction": reconstructed,
            "commitment_loss": commitment,
            "diffusion_loss": diffusion_loss,
            "loss": (
                F.mse_loss(reconstructed, images)
                + 0.1 * commitment
                + 0.05 * diffusion_loss
            ),
        }

    def generate(
        self, idea: torch.Tensor, generator: torch.Generator, steps: int = 4
    ) -> torch.Tensor:
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
        for index in range(max(1, steps)):
            rate = 0.35 / float(index + 1)
            prediction = self.denoiser(
                latent + condition,
                idea,
                timestep=1.0 - index / float(max(1, steps)),
            )
            latent = latent - rate * prediction
            latent = 0.9 * latent + 0.1 * condition
        return self.decoder(latent).clamp(-1.0, 1.0)


class TinyAudioCodec(nn.Module):
    """Residual-vector-quantized codec plus idea-conditioned token generator."""

    def __init__(self, shared_dim: int, samples: int, channels: int = 16):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.latent_samples = samples // 4
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
        )
        self.quantizer_a = VectorQuantizer(32, channels)
        self.quantizer_b = VectorQuantizer(16, channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(channels, 1, 4, stride=2, padding=1),
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

    def generate(
        self, idea: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
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
        for index in range(3):
            predicted = self.token_generator(
                latent, idea, timestep=1.0 - index / 3.0
            )
            latent = 0.8 * latent + 0.2 * predicted
        return self.decoder(latent).squeeze(1)


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
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.liquid_gate = LiquidTemporalGate(channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                channels,
                channels,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                channels,
                3,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.Tanh(),
        )
        self.encoder = nn.Sequential(
            nn.Conv3d(
                3,
                channels,
                (1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
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

    def generate(
        self, idea: torch.Tensor, generator: torch.Generator, steps: int = 3
    ) -> torch.Tensor:
        condition = self._condition(idea)
        latent = torch.randn(
            condition.shape,
            generator=generator,
            device=condition.device,
            dtype=condition.dtype,
        )
        total_steps = max(1, steps)
        for index in range(total_steps):
            timestep = 1.0 - index / float(total_steps)
            predicted_noise = self._evolve(
                latent + (0.05 + 0.1 * timestep) * condition
            )
            rate = 0.32 / float(index + 1)
            latent = latent - rate * predicted_noise + 0.08 * condition
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

    @torch.no_grad()
    def generate(
        self, modality: str, idea: torch.Tensor, seed: int = 0
    ) -> torch.Tensor:
        self.eval()
        generator = torch.Generator(device=idea.device)
        generator.manual_seed(int(seed))
        if modality == "image":
            return self.image.generate(idea, generator)
        if modality == "audio":
            return self.audio.generate(idea, generator)
        if modality == "video":
            return self.video.generate(idea, generator)
        raise ValueError("modality must be image, audio, or video")
