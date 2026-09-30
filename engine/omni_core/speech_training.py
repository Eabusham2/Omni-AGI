"""Supervision of the same native idea-to-waveform graph used for voice output."""

import torch
from torch.nn import functional as F
from .text_spool import require_parser_resources


class StreamingSpeechResampler:
    """Phase-continuous linear PCM resampling, in admitted native-sized blocks.

    Every source sample is consumed, and output duration is determined by the
    measured source rate. This is signal processing, not a speech generator.
    """

    def __init__(self, source_rate, target_rate=16000, block_samples=4096):
        if any(type(value) is not int or value < 1 for value in (source_rate, target_rate, block_samples)):
            raise ValueError("speech resampling needs measured positive integer rates and block size")
        self.source_rate, self.target_rate, self.block_samples = source_rate, target_rate, block_samples
        self.source_samples = self.output_samples = 0
        self.previous = None

    def push(self, values, *, final=False):
        values = values.detach().float().flatten().cpu()
        if not values.numel() and self.previous is None:
            return
        base = self.source_samples - (1 if self.previous is not None else 0)
        joined = torch.cat((self.previous, values)) if self.previous is not None else values
        self.source_samples += values.numel()
        self.previous = joined[-1:].clone()
        end = ((self.source_samples - 1) * self.target_rate) // self.source_rate + 1
        if final:
            end = (self.source_samples * self.target_rate + self.source_rate - 1) // self.source_rate
        while self.output_samples < end:
            count = min(self.block_samples, end - self.output_samples)
            require_parser_resources("paired speech resampling", ram_bytes=count * 64 + joined.numel() * 4)
            positions = torch.arange(self.output_samples, self.output_samples + count, dtype=torch.int64)
            numerator = positions * self.source_rate
            left = (numerator // self.target_rate - base).clamp(0, joined.numel() - 1)
            right = (left + 1).clamp(max=joined.numel() - 1)
            fraction = (numerator % self.target_rate).float() / float(self.target_rate)
            output = joined[left] * (1.0 - fraction) + joined[right] * fraction
            self.output_samples += count
            yield output


def paired_speech_waveform_loss(codec, idea, target, actual_samples, *, seed):
    """Teach the real generator, not just audio reconstruction or filename embeddings.

    Native windows remain bounded; a tail's padding is never a speech target.
    Paired examples alone do not verify intelligibility or alignment quality.
    """
    if type(actual_samples) is not int or not 0 < actual_samples <= target.shape[-1]:
        raise ValueError("paired speech window has invalid actual sample coverage")
    if target.ndim == 3 and target.shape[1] == 1:
        target = target.squeeze(1)
    tokens = max(1, int(getattr(codec, "latent_samples", target.shape[-1] // 4)))
    blocks = getattr(getattr(codec, "token_generator", None), "blocks", ())
    heads = sum(int(getattr(block, "heads", 1)) for block in blocks)
    parameter_bytes = sum(value.numel() * value.element_size() for value in codec.parameters()) if hasattr(codec, "parameters") else 0
    # Three native diffusion steps retain attention/codec graphs for backward.
    # This is conservative physical admission, not a claimed allocator cap.
    require_parser_resources("paired speech generator supervision", ram_bytes=(
        3 * target.shape[0] * heads * tokens * tokens * target.element_size() * 8
        + target.numel() * target.element_size() * 64 + parameter_bytes * 4))
    generator = torch.Generator(device=idea.device)
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    predicted = codec.generate(idea, generator)
    if predicted.shape != target.shape:
        raise RuntimeError("paired speech generator and waveform target shapes differ")
    return F.mse_loss(predicted[..., :actual_samples], target[..., :actual_samples])
