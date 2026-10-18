"""The existing native single-token sampling law, with no world callbacks.

Generator, home device and UTF-8 boundary state belong to the caller. This
helper does not accept a token, publish text, invoke an action or learn.
"""
import torch
from torch.nn import functional as F


def sample_native_token(logits, generator, *, temperature, top_k, noise, text_boundaries, remaining_tokens):
    if noise > 0:
        jitter = torch.randn(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
        logits = logits + float(noise) * jitter
    logits = logits / max(float(temperature), 1e-4)
    if text_boundaries is not None:
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for row, boundary in enumerate(text_boundaries):
            mask[row, boundary.allowed_token_ids(remaining_tokens)] = True
        logits = logits.masked_fill(~mask, -float("inf"))
    if top_k > 0 and top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, top_k, dim=-1)
        probabilities = F.softmax(values.float(), dim=-1)
        sample = torch.multinomial(probabilities, 1, generator=generator)
        token = indices.gather(-1, sample)
        selected_probability = probabilities.gather(-1, sample)
    else:
        probabilities = F.softmax(logits.float(), dim=-1)
        token = torch.multinomial(probabilities, 1, generator=generator)
        selected_probability = probabilities.gather(-1, token)
    log_probability = float(selected_probability.clamp_min(1e-45).log().mean().item())
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    return token, log_probability, float(entropy.mean().item())
