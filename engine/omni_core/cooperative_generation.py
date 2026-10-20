"""Interleaved same-model inference with private cache and sampling state.

This is cooperative concurrency, not simultaneous GPU batches. A branch never
keeps a ContextVar, pager activation, or cancellation scope entered over a
yield. All decoder metadata is restored after each advance.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch


_ABSENT = object()
_FIELDS = ("last_generation_cache_mode", "last_generation_stop_reason",
           "last_generation_tool_observations", "last_generation_sensory_observations",
           "last_generation_observations", "last_generation_neural_state")


class CooperativeNativeGeneration:
    def __init__(self, decoder, input_ids, *, capture_tokens=False, **options):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.shape[1] > decoder.config.max_seq_len:
            raise ValueError("cooperative generation requires an admitted nonempty context")
        if options.get("token_callback") is not None or options.get("conditioning_callback") is not None:
            raise ValueError("private candidates cannot publish tokens or execute world observations")
        self.decoder, self.input_ids = decoder, input_ids
        self.options = dict(options)
        # Optional private data collector, not an output/world callback. MCP
        # sampling needs complete UTF-8 token boundaries to stop a private
        # completion before returning it for human review. Other branches do
        # not allocate this token list unless requested.
        if type(capture_tokens) is not bool:
            raise ValueError("private token capture must be a boolean")
        self.emitted_token_ids = [] if capture_tokens else None
        if capture_tokens:
            def collect(tokens, _step, _entropy):
                self.emitted_token_ids.extend(tokens.reshape(-1).tolist())
            self.options["token_callback"] = collect
        self.cache_owner = [None]
        self.iterator = decoder._generate_scoped_iter(input_ids, **self.options, _cache_owner=self.cache_owner)
        self.metadata = {"last_generation_cache_mode": "not-started", "last_generation_stop_reason": "not-started",
                         "last_generation_tool_observations": [], "last_generation_sensory_observations": [],
                         "last_generation_observations": []}
        self.metadata_owners = [owner for owner in decoder.modules()
                                if owner is decoder or any(hasattr(owner, name) for name in _FIELDS)]
        self.regional_metadata = {id(owner): ({name: getattr(owner, name) for name in _FIELDS if hasattr(owner, name)}
                                            if owner is not decoder else self.metadata)
                                  for owner in self.metadata_owners}
        self.log_probabilities = []
        self.result = None
        self.done = False
        self.closed = False
        self.admitted = False

    def stop_requested(self):
        return any(callable(self.options.get(name)) and self.options[name]() for name in ("cancelled", "steer_check"))

    @torch.no_grad()
    def advance(self):
        if self.done or self.closed:
            return None
        decoder = self.decoder
        previous = [(owner, {name: getattr(owner, name, _ABSENT) for name in _FIELDS})
                    for owner in self.metadata_owners]
        for owner in self.metadata_owners:
            for name in _FIELDS:
                value = self.regional_metadata[id(owner)].get(name, _ABSENT)
                if value is _ABSENT:
                    if hasattr(owner, name):
                        delattr(owner, name)
                else:
                    setattr(owner, name, value)
        pager = decoder._working_pager()
        scope = pager.activate(self.stop_requested) if pager is not None else nullcontext()
        try:
            with scope:
                if not self.admitted:
                    if pager is not None:
                        count = self.input_ids.numel() + max(1, int(self.options.get("max_new_tokens", 48))) * self.input_ids.shape[0]
                        pager.admit_compute(3 * count * self.input_ids.element_size(), "private branch token reservation", device=self.input_ids.device)
                    self.admitted = True
                try:
                    event = next(self.iterator)
                except StopIteration as stopped:
                    self.result = stopped.value
                    self.done = True
                    return None
                self.log_probabilities.append(float(event["logProbability"]))
                return event
        except BaseException:
            self.close()
            raise
        finally:
            for owner, state in previous:
                self.regional_metadata[id(owner)] = {name: getattr(owner, name) for name in _FIELDS if hasattr(owner, name)}
                for name, value in state.items():
                    if value is _ABSENT:
                        if hasattr(owner, name):
                            delattr(owner, name)
                    else:
                        setattr(owner, name, value)
            self.metadata = self.regional_metadata[id(decoder)]
            if self.done:
                self.close()

    @property
    def stop_reason(self):
        return self.metadata.get("last_generation_stop_reason", "unknown")

    @property
    def mean_log_probability(self):
        count = max(0, int(self.result[0].shape[1] - self.input_ids.shape[1])) if self.result is not None else len(self.log_probabilities)
        values = self.log_probabilities[:count]
        return sum(values) / len(values) if values else None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.iterator.close()
        if self.cache_owner[0] is not None:
            self.cache_owner[0].close()
            self.cache_owner[0] = None
