"""Own packed auxiliary drafting research; deliberately not a runtime toggle.

The shared native embedding/readout stay read-only. This bounded candidate
uses parallel local mixing and a predecessor-conditioned selector inspired by
DFlash2, not its foreign weights or complete diffusion backbone. Verification
uses real sequential prefix distributions because growable expert routing
makes one long forward's earlier logits an invalid general verifier.

Only static, no-noise, unmasked research decoding is supported here. Live
actions, Ponder, UTF-8 publication and cold resume remain on ordinary generate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modalities import PackedTernaryTable
from .model import PackedAdaptiveBitLinear, RMSNorm, SwiGLU


class DraftEvaluationInterrupted(RuntimeError):
    pass


def sampling_distribution(logits, temperature, top_k=0):
    """The actual proposal/target law, including its own top-K support."""
    if (logits.ndim != 1 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or not math.isfinite(temperature) or temperature < 0 or type(top_k) is not int or top_k < 0):
        raise ValueError('draft evaluation sampling inputs are invalid')
    if temperature == 0:
        return F.one_hot(logits.argmax(), logits.numel()).to(dtype=torch.float32)
    scores = logits.float() / temperature
    if 0 < top_k < scores.numel():
        selected, ids = scores.topk(top_k)
        return torch.zeros_like(scores).scatter(0, ids, selected.softmax(-1))
    return scores.softmax(-1)


def _probabilities(value):
    if (value.ndim != 1 or not value.is_floating_point() or not value.numel()
            or not bool(torch.isfinite(value).all()) or bool((value < 0).any())
            or not bool(torch.isfinite(value.sum())) or not bool(value.sum() > 0)):
        raise ValueError('speculative probabilities are invalid')
    return value / value.sum()


def acceptance_and_residual(target, proposal):
    """Leviathan et al. exact acceptance mass and positive residual law.

    Float activities retain ordinary numerical precision; no epsilon changes
    the law and no unmodified-target fallback disguises a zero residual.
    """
    if target.shape != proposal.shape or target.device != proposal.device:
        raise ValueError('proposal and target vocabularies/devices differ')
    p, q = _probabilities(target), _probabilities(proposal)
    overlap = torch.minimum(p, q)
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    return p, q, overlap, None if not bool(total > 0) else residual / total


def verify_proposal(token, target, proposal, generator):
    p, q, _overlap, residual = acceptance_and_residual(target, proposal)
    if type(token) is not int or not 0 <= token < p.numel() or not bool(q[token] > 0):
        raise ValueError('proposed token has no probability under its actual proposal')
    accepted = bool(torch.rand((), generator=generator, device=p.device) * q[token] < p[token])
    if accepted:
        return token, True
    if residual is None:
        raise RuntimeError('a rejection with identical probability laws is impossible')
    return int(torch.multinomial(residual, 1, generator=generator)), False


class NativeDraftRegion(nn.Module):
    """Small packed adapter, sharing the brain's vocabulary and readout.

    No target is registered as a child or held as an alternate model. Every
    learned projection/table here has authoritative signs/integer strengths.
    Token representations enter only as transient shared-embedding activity.
    """
    def __init__(self, dimensions, block_size=4, selector_rank=4, candidate_count=16):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (dimensions, block_size, selector_rank, candidate_count)):
            raise ValueError('native draft geometry must be positive integers')
        self.dimensions, self.block_size = dimensions, block_size
        self.candidate_count = candidate_count
        self.norm = RMSNorm(dimensions)
        self.context = PackedAdaptiveBitLinear(dimensions, dimensions, bias=False)
        self.positions = PackedTernaryTable(block_size, dimensions, scale=1 / math.sqrt(dimensions))
        self.local_kernels = PackedAdaptiveBitLinear(dimensions, 2, bias=False)
        self.local_base = PackedTernaryTable(2, dimensions, scale=.25)
        self.mix = SwiGLU(dimensions, dimensions * 2)
        self.predecessor = PackedAdaptiveBitLinear(dimensions, selector_rank, bias=False)
        self.successor = PackedAdaptiveBitLinear(dimensions, selector_rank, bias=False)
        self.gate = PackedAdaptiveBitLinear(dimensions, selector_rank, bias=False)

    def scratch_bytes(self, vocabulary, count):
        candidates = min(vocabulary, self.candidate_count)
        return count * (vocabulary * 16 + self.dimensions * (candidates * 16 + 96)) + 131072

    def _features(self, target, hidden, anchor, count):
        if (count < 1 or count > self.block_size or hidden.shape != (1, self.dimensions)
                or anchor.shape != (1,) or hidden.device != anchor.device
                or target.training or target.embedding.training or target.language_head.training
                or target.config.d_model != self.dimensions):
            raise ValueError('native draft requires matching read-only target activity')
        with torch.no_grad():
            anchored = target.embedding(anchor).detach()
        positions = torch.arange(count, device=hidden.device)
        initial = self.context(hidden.detach())[:, None] + self.positions(positions)[None]
        previous = torch.cat((anchored[:, None], initial[:, :-1]), dim=1)
        kernels = self.local_kernels(initial).tanh()
        base = self.local_base(torch.arange(2, device=hidden.device))
        local = initial * (base[0] + kernels[..., :1]) + previous * (base[1] + kernels[..., 1:])
        mixed = initial + self.mix(self.norm(local))
        # Eval target projections pass activity gradients but never learn.
        unary = target.language_head(self.norm(mixed))
        return mixed, unary

    def _scores(self, target, hidden, unary, candidates, predecessor):
        with torch.no_grad():
            before = target.embedding(predecessor).detach()
            successors = target.embedding(candidates).detach()
        edge = self.predecessor(before) * self.gate(hidden)
        return unary + (self.successor(successors) * edge[:, None]).sum(-1)

    @torch.no_grad()
    def propose(self, target, hidden, anchor, count, generator, temperature=.7):
        mixed, logits = self._features(target, hidden, anchor, count)
        unary, candidates = logits.topk(min(self.candidate_count, logits.shape[-1]), dim=-1)
        previous, tokens, probabilities = anchor, [], []
        for position in range(count):
            scores = self._scores(target, mixed[:, position], unary[:, position], candidates[:, position], previous)
            law = sampling_distribution(scores[0], temperature)
            index = torch.multinomial(law, 1, generator=generator)
            previous = candidates[:, position].gather(-1, index.reshape(1, 1)).reshape(1)
            tokens.append(previous[0])
            probabilities.append(torch.zeros(logits.shape[-1], dtype=law.dtype, device=law.device)
                .scatter(0, candidates[0, position], law))
        return torch.stack(tokens), torch.stack(probabilities)

    def teacher_loss(self, target, hidden, anchor, tokens):
        """Own-target unary/path imitation only; no reward/preference loop.

        Inclusion of a missing teacher candidate is training-only. Actual
        proposals never insert a verifier token into the candidate support.
        """
        count = tokens.numel()
        mixed, logits = self._features(target, hidden, anchor, count)
        unary, candidates = logits.topk(min(self.candidate_count, logits.shape[-1]), dim=-1)
        losses, previous = [F.cross_entropy(logits[0], tokens)], anchor
        for position in range(count):
            ids = candidates[:, position].clone()
            scores = unary[:, position].clone()
            match = ids[0].eq(tokens[position])
            if not bool(match.any()):
                ids[0, -1] = tokens[position]
                scores[0, -1] = logits[0, position, tokens[position]]
                label = ids.shape[-1] - 1
            else:
                label = int(torch.nonzero(match, as_tuple=False)[0, 0])
            corrected = self._scores(target, mixed[:, position], scores, ids, previous)
            losses.append(F.cross_entropy(corrected, torch.tensor([label], device=tokens.device)))
            previous = tokens[position:position + 1]
        return torch.stack(losses).mean()


@contextmanager
def private_target_metadata(target):
    fields = ('last_generation_cache_mode', 'last_generation_stop_reason',
              'last_generation_tool_observations', 'last_generation_neural_state')
    absent = object()
    originals = [(module, {field: getattr(module, field, absent) for field in fields}) for module in target.modules()]
    try:
        yield
    finally:
        for module, values in originals:
            for field, value in values.items():
                if value is absent:
                    if hasattr(module, field):
                        delattr(module, field)
                else:
                    setattr(module, field, value)


@dataclass
class DraftRound:
    tokens: torch.Tensor
    proposed: int
    accepted: int
    target_calls: int
    terminal: bool


@torch.no_grad()
def evaluate_round(target, auxiliary, prefix, remaining, *, draft_generator, verifier_generator,
                   temperature=.7, top_k=0, memory_bias=None, stop_check=None, admit=None):
    """One private atomic research round; no published callbacks/effects.

    This deliberately does not reuse a live decoder cache. On interruption,
    both RNGs roll back and every private cache closes. A future production
    hook needs a separately versioned verifier frontier/cursor before enabling.
    """
    if (any(module.training for module in target.modules()) or auxiliary.training or prefix.ndim != 2 or prefix.shape[0] != 1
            or prefix.shape[1] < 1 or type(remaining) is not int or remaining < 1
            or prefix.shape[1] + min(auxiliary.block_size + 1, remaining) > target.config.max_seq_len):
        raise ValueError('draft evaluation needs an admitted static eval frontier')
    draft_state, verifier_state = draft_generator.get_state(), verifier_generator.get_state()
    cache = None
    def boundary():
        if stop_check is not None and stop_check():
            raise DraftEvaluationInterrupted('native draft research stopped at a private boundary')
    try:
        boundary()
        if admit is not None:
            admit(auxiliary.scratch_bytes(target.config.vocab_size, min(auxiliary.block_size, remaining)),
                'native auxiliary drafting research')
        with private_target_metadata(target):
            logits, cache = target._generation_step(prefix, memory_bias, None)
            calls = 1
            hidden = target.last_generation_neural_state
            count = min(auxiliary.block_size, remaining)
            boundary()
            proposals, laws = auxiliary.propose(target, hidden, prefix[:, -1], count,
                draft_generator, temperature)
            output, accepted = [], 0
            for position in range(count):
                boundary()
                token, keep = verify_proposal(int(proposals[position]),
                    sampling_distribution(logits[0], temperature, top_k), laws[position], verifier_generator)
                output.append(token)
                accepted += int(keep)
                if token == 2 or not keep or len(output) == remaining:
                    boundary()
                    return DraftRound(torch.tensor(output, device=prefix.device, dtype=prefix.dtype), count,
                        accepted, calls, token == 2)
                current = torch.cat((prefix, torch.tensor([output], device=prefix.device, dtype=prefix.dtype)), dim=1)
                logits, cache = target._generation_step(current, memory_bias, cache)
                calls += 1
            boundary()
            bonus = int(torch.multinomial(sampling_distribution(logits[0], temperature, top_k), 1,
                generator=verifier_generator))
            output.append(bonus)
            return DraftRound(torch.tensor(output, device=prefix.device, dtype=prefix.dtype), count,
                accepted, calls, bonus == 2)
    except BaseException:
        draft_generator.set_state(draft_state)
        verifier_generator.set_state(verifier_state)
        raise
    finally:
        if cache is not None:
            cache.close()
