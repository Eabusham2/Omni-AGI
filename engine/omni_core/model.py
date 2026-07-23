"""Ternary decoder-only language model used by an OmniCortex brain."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import OmniConfig


def ternary_quantize(weight: torch.Tensor) -> torch.Tensor:
    """Quantize a latent floating weight tensor to scaled {-1, 0, +1}.

    Rounding is used in the forward pass and the normalized latent value in
    the backward pass (the straight-through estimator).  ``BitLinear`` keeps
    the original parameter as its trainable master weight.
    """

    scale = weight.detach().abs().mean().clamp_min(1e-6)
    normalized = weight / scale
    ternary = normalized.round().clamp(-1.0, 1.0)
    straight_through = normalized + (ternary - normalized).detach()
    return straight_through * scale


class BitLinear(nn.Module):
    """Linear projection with floating master weights and ternary forwards."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary = True
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def effective_weight(self) -> torch.Tensor:
        scale = self.weight.detach().abs().mean().clamp_min(1e-6)
        return (self.weight.detach() / scale).round().clamp(-1, 1).to(torch.int8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = ternary_quantize(self.weight) if self.ternary else self.weight
        return F.linear(inputs, weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dimensions: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dimensions))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(rms.to(inputs.dtype) + self.eps)
        return normalized * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inverse = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = query.shape[-2]
        cos = self.cos[:sequence].to(dtype=query.dtype)[None, None, :, :]
        sin = self.sin[:sequence].to(dtype=query.dtype)[None, None, :, :]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            even = value[..., 0::2]
            odd = value[..., 1::2]
            result = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            )
            return result.flatten(-2)

        return rotate(query), rotate(key)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = BitLinear(config.d_model, config.d_model * 3)
        self.output = BitLinear(config.d_model, config.d_model)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, dimensions = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch, sequence, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query, key = self.rotary(query, key)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.head_dim))
        mask = torch.ones(
            sequence, sequence, device=hidden.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], -torch.inf)
        probabilities = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
        probabilities = F.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, dimensions
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, dimensions: int, hidden_dimensions: int):
        super().__init__()
        self.up = BitLinear(dimensions, hidden_dimensions * 2)
        self.down = BitLinear(hidden_dimensions, dimensions)

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


class TernaryExpert(nn.Module):
    """Small growable residual expert selected by an idea prototype."""

    def __init__(self, dimensions: int, hidden_dimensions: int):
        super().__init__()
        self.norm = RMSNorm(dimensions)
        self.network = SwiGLU(dimensions, hidden_dimensions)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(self.norm(hidden))


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
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.memory_projection = BitLinear(config.idea_dim, config.d_model)
        self.memory_strength = nn.Parameter(torch.tensor(0.15))
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.language_head = BitLinear(config.d_model, config.vocab_size)
        self.experts = nn.ModuleList()
        self.expert_prototypes = nn.ParameterList()
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def expert_count(self) -> int:
        return len(self.experts)

    def grow_expert(self, prototype: Optional[torch.Tensor] = None) -> int:
        expert = TernaryExpert(self.config.d_model, max(16, self.config.d_ff // 2))
        expert.to(next(self.parameters()).device)
        self.experts.append(expert)
        if prototype is None:
            prototype = torch.randn(
                self.config.d_model, device=next(self.parameters()).device
            )
        prototype = F.normalize(prototype.detach().reshape(-1), dim=0)
        self.expert_prototypes.append(nn.Parameter(prototype.clone()))
        return len(self.experts) - 1

    def _apply_experts(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.experts:
            return hidden, None
        pooled = F.normalize(hidden.mean(dim=1), dim=-1)
        prototypes = F.normalize(
            torch.stack(list(self.expert_prototypes)), dim=-1
        )
        routing = F.softmax(pooled @ prototypes.t(), dim=-1)
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
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len :]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len :]

        hidden = self.embedding(input_ids)
        if memory_bias is not None:
            if memory_bias.ndim == 1:
                memory_bias = memory_bias.unsqueeze(0)
            projected = self.memory_projection(memory_bias).unsqueeze(1)
            hidden = hidden + torch.tanh(self.memory_strength) * projected

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
        hidden, routing = self._apply_experts(hidden)
        hidden = self.final_norm(hidden)
        logits = self.language_head(hidden)
        output: Dict[str, torch.Tensor] = {"logits": logits, "hidden": hidden}
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
    ) -> Tuple[torch.Tensor, List[float]]:
        self.eval()
        generated = input_ids
        entropies: List[float] = []
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(int(seed))

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
            window = generated[:, -self.config.max_seq_len :]
            logits = self.forward(window, memory_bias=memory_bias)["logits"][:, -1]
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
            if step > 0 and bool((token == 2).all()):
                break
        return generated, entropies
