"""Exact invocation-local native sampling frontier, not text-prefix replay.

The cursor contains activity/RNG/cache state, never another learned model. It
is only captured between iterations, after publishing the completed token
event. The lifecycle owner must persist it before releasing its cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch

from .utf8_text_boundary import Utf8TextBoundary


class NativeGenerationSuspended(Exception):
    def __init__(self, cursor):
        super().__init__("native generation reached its exact suspension boundary")
        self.cursor = cursor


@dataclass
class NativeGenerationCursor:
    generated: torch.Tensor
    input_length: int
    memory_bias: Any
    cache: Any
    entropies: list[float]
    rng_state: torch.Tensor
    text_boundaries: Any
    last_text_boundary: int
    emission_start: int
    token_budget: int
    options: dict[str, Any]
    metadata: dict[str, Any]
    resumed_conditioning: Any = None
    window_start: int = 0

    @classmethod
    def capture(cls, *, generated, input_length, memory_bias, cache, entropies,
                generator, text_boundaries, last_text_boundary, emission_start,
                token_budget, options, decoder, resumed_conditioning=None, window_start=0):
        return cls(generated.detach(), int(input_length), memory_bias, cache,
            list(entropies), generator.get_state().clone(),
            None if text_boundaries is None else [dict(vars(boundary)) for boundary in text_boundaries],
            int(last_text_boundary), int(emission_start), int(token_budget), dict(options),
            {name: getattr(decoder, name) for name in (
                "last_generation_cache_mode", "last_generation_stop_reason",
                "last_generation_tool_observations", "last_generation_sensory_observations",
                "last_generation_observations", "last_generation_neural_state") if hasattr(decoder, name)},
            resumed_conditioning, int(window_start))

    def restore_boundaries(self):
        if self.text_boundaries is None:
            return None
        result = []
        # Derive the small wire-syntax state from its exact existing byte
        # tokens; this executes no neural model, context injection or action.
        for row, recorded in zip(self.generated[:, self.input_length:], self.text_boundaries):
            boundary = Utf8TextBoundary()
            for token in row.detach().cpu().tolist():
                boundary.accept(token)
            if vars(boundary) != recorded:
                raise ValueError("saved UTF-8 state differs from its exact token frontier")
            result.append(boundary)
        return result

    def validate(self, input_ids, options):
        if (self.generated.ndim != 2 or input_ids.ndim != 2 or self.generated.shape[0] != input_ids.shape[0]
                or self.input_length != input_ids.shape[1] or self.generated.shape[1] < self.input_length
                or self.generated.dtype != torch.long or self.generated.device != input_ids.device
                or not torch.equal(self.generated[:, :self.input_length], input_ids)):
            raise ValueError("native continuation has a different input or token frontier")
        expected = {key: value for key, value in self.options.items() if key != "context_capacity"}
        actual = {key: value for key, value in options.items() if key != "context_capacity"}
        minimum_context = min(self.generated.shape[1] - self.window_start, self.options["context_capacity"])
        if expected != actual or options["context_capacity"] < minimum_context:
            raise ValueError("native continuation sampling options changed")
        count = self.generated.shape[1] - self.input_length
        if (len(self.entropies) != count or count > self.token_budget
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in self.entropies)
                or not self.input_length <= self.emission_start <= self.last_text_boundary <= self.generated.shape[1]
                or self.rng_state.dtype != torch.uint8 or self.rng_state.ndim != 1
                or type(self.window_start) is not int or not 0 <= self.window_start < self.generated.shape[1]
                or self.text_boundaries is not None and len(self.text_boundaries) != self.generated.shape[0]):
            raise ValueError("native continuation counters or sampling state are invalid")

    def bind_verified_cache(self, decoder):
        """Caller first verifies the paused learned/checkpoint binding."""
        def bind(owner, cache, cue):
            if cache is None:
                return
            cache.signature = owner._generation_signature(cue)
            if hasattr(cache, "nested"):
                parts = [None] * len(owner.regions) if cue is None else cue.split(owner.widths, dim=-1)
                for region, nested, part in zip(owner.regions, cache.nested, parts):
                    bind(region.decoder, nested, part)
        bind(decoder, self.cache, self.memory_bias)

    def close(self):
        if self.cache is not None:
            self.cache.close()
            self.cache = None
