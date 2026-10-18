"""Connected native regions used by full, multi-parent merges.

Parent modules remain exact named owners of their packed learned state. A
single token stream advances all cortical regions and exchanges causal hidden
activity between them. This is a neural graph, not a collection of chat calls:
there is one sampler, one workspace, and one action result. New bridge weights
start at zero; integration learning can activate them without first rewriting
or averaging any parent's learned connections.

The caller owns graph memories and their persistence. ``recall`` callbacks
must consume/return neural vectors only; source text is not an input here.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .config import OmniConfig
from .liquid import LiquidController
from .modalities import IMAGINATION_MODALITIES, ModalityHub
from .model import (
    ACTION_KINDS, ActionArgumentHead, OmniDecoder, PackedAdaptiveBitLinear,
    ToolRouteHead,
)
from .native_core_paging import current_native_core_pager
from .spiking import AssociativeSpikingRouter
from .working_attention_paging import current_working_attention_pager
from .operation_auto_policy import measured_perception
from .whole_path_scheduler import owned_memory_effect


@dataclass(frozen=True)
class RegionIdentity:
    region_id: str
    parent_id: str
    generation_sha256: str


@dataclass
class _ConnectedInferenceCache:
    length: int
    signature: Any
    workspace: list[list[Any]]
    attention: list[list[Any]]
    temporal: list[Any]
    hidden_sum: list[torch.Tensor]
    nested: list[Any]

    def close(self):
        for caches in (*self.workspace, *self.attention):
            for value in caches:
                if value is not None:
                    value.close()
        for value in self.nested:
            if value is not None:
                value.close()


def _widths(parts: Sequence[int]) -> tuple[int, ...]:
    result = tuple(parts)
    if not result or any(type(value) is not int or value <= 0 for value in result):
        raise ValueError("connected region widths must be positive integers")
    return result


def _admit(byte_count: int, label: str, device: torch.device) -> None:
    pager = current_working_attention_pager()
    if pager is not None:
        pager.check_cancelled()
        pager.admit_compute(int(byte_count), label, device=device)


class RegionExchange(nn.Module):
    """Directed, trainable, packed bridges between all distinct regions.

    Separate maps preserve different widths. There is no full dense N-by-N
    floating matrix. Construction goes through the ordinary packed pager and
    each map participates in its normal bounded gradient path.
    """

    def __init__(self, widths: Sequence[int], *, loading: bool = False):
        super().__init__()
        self.widths = _widths(widths)
        self.bridges = nn.ModuleDict()
        for target, target_width in enumerate(self.widths):
            for source, source_width in enumerate(self.widths):
                if source == target:
                    continue
                bridge = PackedAdaptiveBitLinear(source_width, target_width, bias=False)
                if not loading and not bridge._native_loading_checkpoint:
                    bridge.fill_ternary_(0)
                self.bridges[f"r{source}_to_r{target}"] = bridge

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        if activity.shape[-1] != sum(self.widths):
            raise ValueError("connected workspace width disagrees with its regions")
        _admit(activity.numel() * activity.element_size() * 3,
               "connected-region workspace exchange", activity.device)
        parts = activity.split(self.widths, dim=-1)
        result = []
        scale = 1.0 / math.sqrt(max(1, len(parts) - 1))
        for target, original in enumerate(parts):
            updated = original
            for source, incoming in enumerate(parts):
                if source != target:
                    updated = updated + scale * self.bridges[f"r{source}_to_r{target}"](incoming)
            result.append(updated)
        return torch.cat(result, dim=-1)

    def alignment_loss(self, regional_activity: Sequence[torch.Tensor]) -> torch.Tensor:
        """Train bridges from paired *actual* activity, without source replay.

        Parent activity is a stopped-gradient target; this objective only
        changes new connections. It is integration evidence, not a promise of
        retention, fluency, or a semantic match between independently trained
        dimensions. Those require the separate accepted quality evaluation.
        """
        if len(regional_activity) != len(self.widths):
            raise ValueError("integration activity must include every region")
        losses = []
        for target, expected in enumerate(regional_activity):
            if expected.shape[-1] != self.widths[target]:
                raise ValueError("integration target has the wrong width")
            for source, incoming in enumerate(regional_activity):
                if source == target:
                    continue
                if incoming.shape[:-1] != expected.shape[:-1]:
                    raise ValueError("integration samples are not temporally paired")
                losses.append(F.mse_loss(
                    self.bridges[f"r{source}_to_r{target}"](incoming.detach()),
                    expected.detach(),
                ))
        if not losses:
            raise ValueError("integration requires at least two distinct regions")
        return torch.stack(losses).mean()


class NativeRegion(nn.Module):
    """All trainable owners of one parent, used inside the joint graph."""

    def __init__(self, identity: RegionIdentity, config: OmniConfig, *, connected=None):
        super().__init__()
        config.validate()
        self.identity = identity
        self.config = config
        self.decoder = connected.decoder if connected is not None else OmniDecoder(config)
        self.memory_bridge = PackedAdaptiveBitLinear(config.vsa_dim, config.idea_dim, bias=True)
        self.idea_adapter = nn.Sequential(
            PackedAdaptiveBitLinear(config.idea_dim, config.idea_dim * 2, bias=True),
            nn.SiLU(),
            PackedAdaptiveBitLinear(config.idea_dim * 2, config.idea_dim, bias=True),
        )
        self.router = AssociativeSpikingRouter(
            config.idea_dim, config.router_neurons, leak=config.membrane_leak,
            threshold=config.firing_threshold, learning_rate=config.stdp_learning_rate,
            tau_pre=config.stdp_tau_pre, tau_post=config.stdp_tau_post,
            a_plus=config.stdp_a_plus, a_minus=config.stdp_a_minus,
            metaplasticity_rate=config.metaplasticity_rate,
        )
        self.liquid = LiquidController(config.idea_dim, mode=config.liquid_mode,
                                       solver_steps=config.liquid_steps)
        self.modalities = connected.modalities if connected is not None else ModalityHub(config)
        self.register_buffer("liquid_state", torch.zeros(1, config.idea_dim))
        # New links from current cortical activity into the parent's retained
        # graph. Its original memory bridge and idea adapter remain active.
        pager = current_native_core_pager()
        # This is a newly created connection, not an uninitialized missing
        # parent tensor. A subsequent joint reload overwrites it normally.
        with pager.construction(from_checkpoint=False) if pager is not None else nullcontext():
            self.recall_projection = PackedAdaptiveBitLinear(config.d_model, config.vsa_dim)
        object.__setattr__(self, "_recall", None)
        object.__setattr__(self, "_learn_memory", None)

    def attach_recall(self, callback: Callable[[torch.Tensor], torch.Tensor]) -> None:
        object.__setattr__(self, "_recall", callback)

    def attach_memory_learning(self, callback: Callable[..., Any]) -> None:
        object.__setattr__(self, "_learn_memory", callback)

    @torch.no_grad()
    @owned_memory_effect("connected-region-experience")
    def observe(self, neural, *, event_key: str, importance: float = 0.5):
        if neural.ndim == 1:
            neural = neural.unsqueeze(0)
        if neural.shape != (1, self.config.d_model):
            raise ValueError("regional experience needs one admitted neural state")
        cue = self.recall_projection(neural)[0]
        evidence = self._learn_memory(cue, event_key=event_key, importance=importance) if self._learn_memory is not None else None
        adapted = self.idea_adapter(self.memory_bridge(cue.unsqueeze(0)))
        state, controls = self.liquid(adapted, self.liquid_state)
        self.liquid_state.copy_(state)
        routed, timing = self.router.route(adapted, learn=True,
            threshold_offset=float(controls["threshold_offset"][0]))
        nested = (self.decoder.observe_experience(neural, event_key=event_key, importance=importance)
                  if isinstance(self.decoder, ConnectedOmniDecoder) else None)
        return {"regionId": self.identity.region_id, "eventKey": event_key,
                "memory": evidence, "timing": timing, "activity": routed,
                **({"nestedRegions": nested} if nested is not None else {})}

    def condition(self, neural: torch.Tensor, *, recurrent_state=None, return_state=False):
        """Causal neural-only recall, recurrent and temporal processing.

        Local activity is advanced here without committing persistent timing
        state. The single brain controller commits activity once for an
        admitted experience, not once for each inference prefill/retry.
        """
        if neural.ndim != 3 or neural.shape[-1] != self.config.d_model:
            raise ValueError("regional activity must be [batch, tokens, width]")
        _admit(neural.numel() * neural.element_size() * 10 +
               neural.shape[0] * self.router.neurons * 24,
               "connected regional temporal and recall activity", neural.device)
        results = []
        if recurrent_state is None:
            liquid = self.liquid_state.to(neural).expand(neural.shape[0], -1)
            membrane = self.router.population.membrane.to(neural).unsqueeze(0).expand(neural.shape[0], -1).clone()
            previous = torch.zeros_like(membrane)
        else:
            liquid, membrane, previous = recurrent_state
        for position in range(neural.shape[1]):
            value = neural[:, position]
            if self._recall is not None:
                query = self.recall_projection(value)
                memories = []
                for row in query:
                    recalled = self._recall(row.detach())
                    if not isinstance(recalled, torch.Tensor) or recalled.numel() != self.config.vsa_dim:
                        raise ValueError("regional recall must return its exact neural vector width")
                    memories.append(recalled.to(value))
                value = value + self.memory_bridge(torch.stack(memories))
            else:
                # With an empty graph the trained bridge still processes the
                # activity cue; never substitute a text label or answer key.
                value = value + self.memory_bridge(self.recall_projection(value))
            value = self.idea_adapter(value)
            liquid, controls = self.liquid(value, liquid)
            projected = torch.sigmoid(self.router.input_projection(value))
            recurrent = torch.stack([self.router.synapses.recurrent(row) for row in previous])
            current = projected + 0.35 * recurrent
            membrane = membrane * self.router.population.leak + current
            threshold = torch.clamp(
                self.router.population.threshold + controls["threshold_offset"], min=0.05
            ).unsqueeze(-1)
            hard = (membrane >= threshold).to(value)
            membrane = membrane - hard * threshold
            soft = torch.sigmoid((current - threshold) * 8.0)
            spikes = hard + soft - soft.detach()
            previous = hard.detach()
            routed = value + 0.25 * torch.tanh(self.router.output_projection(spikes))
            results.append(routed + liquid)
        output = torch.stack(results, dim=1)
        return (output, (liquid, membrane, previous)) if return_state else output


class _BorrowedRegionalModule(nn.Module):
    """Call region-owned modules without registering duplicate state paths."""
    def __init__(self, regions: Sequence[NativeRegion], widths: Sequence[int]):
        super().__init__()
        object.__setattr__(self, "regions", tuple(regions))
        self.widths = tuple(widths)


class _JoinedEmbedding(_BorrowedRegionalModule):
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([region.decoder.embedding(ids) for region in self.regions], dim=-1)


class _JoinedNorm(_BorrowedRegionalModule):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([region.decoder.final_norm(part) for region, part in
                          zip(self.regions, value.split(self.widths, dim=-1))], dim=-1)


class _BorrowedSequence(nn.Module):
    def __init__(self, values):
        super().__init__()
        object.__setattr__(self, "values", tuple(values))

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def parameters(self, recurse=True):
        for module in self.values:
            yield from module.parameters(recurse=recurse)


class _JoinedWorkspace(_BorrowedRegionalModule):
    def __init__(self, regions, widths, exchange):
        super().__init__(regions, widths)
        object.__setattr__(self, "exchange", exchange)
        self.slots = sum(r.decoder.global_workspace.slots for r in regions)
        self.query_chunk_slots = min(r.decoder.global_workspace.query_chunk_slots for r in regions)

    def summarize(self, hidden, attention_mask=None):
        parts = [region.decoder.global_workspace.summarize(value, attention_mask=attention_mask)
                 for region, value in zip(self.regions, hidden.split(self.widths, -1))]
        return self.exchange(torch.cat(parts, -1))

    def causal_summaries(self, hidden, attention_mask=None):
        parts = [region.decoder.global_workspace.causal_summaries(value, attention_mask=attention_mask)
                 for region, value in zip(self.regions, hidden.split(self.widths, -1))]
        return self.exchange(torch.cat(parts, -1))

    def forward(self, hidden, attention_mask=None):
        summaries = self.summarize(hidden, attention_mask)
        return summaries.unsqueeze(1), summaries


class _JoinedMemoryProjection(_BorrowedRegionalModule):
    def forward(self, value):
        return torch.cat([region.decoder.memory_projection(part) for region, part in
                          zip(self.regions, value.split(self.widths, -1))], -1)


class _BorrowedGains(_BorrowedRegionalModule):
    def __init__(self, regions, widths, field):
        super().__init__(regions, widths)
        self.field = field

    def forward(self):
        return torch.stack([getattr(r.decoder, self.field)() for r in self.regions]).mean()


class _JoinedArgumentCondition(_BorrowedRegionalModule):
    def forward(self, value):
        state, schema = value.split((sum(self.widths), ToolRouteHead.feature_width), dim=-1)
        return torch.cat([region.decoder.action_argument_head.condition(torch.cat((part, schema), dim=-1))
                          for region, part in zip(self.regions, state.split(self.widths, dim=-1))], dim=-1)


class _JoinedArgumentEmbedding(_BorrowedRegionalModule):
    def forward(self, value):
        return torch.cat([region.decoder.action_argument_head.token_embedding(value) for region in self.regions], dim=-1)


class _JoinedArgumentTransition(_BorrowedRegionalModule):
    def __init__(self, regions, widths):
        super().__init__(regions, widths)
        self.exchange = RegionExchange(widths)

    def forward(self, value):
        state, tokens = value.split(sum(self.widths), dim=-1)
        joined = torch.cat([
            region.decoder.action_argument_head.transition(torch.cat((hidden, token), dim=-1))
            for region, hidden, token in zip(self.regions, state.split(self.widths, dim=-1), tokens.split(self.widths, dim=-1))
        ], dim=-1)
        return self.exchange(joined)


class _JoinedArgumentOutput(_BorrowedRegionalModule):
    def forward(self, value):
        result = None
        for region, part in zip(self.regions, value.split(self.widths, dim=-1)):
            item = region.decoder.action_argument_head.output(part)
            result = item if result is None else result + item
        return result / math.sqrt(len(self.regions))


class _BorrowedLearningHead:
    """Expose complete direct learning/rollback roots without alias saves.

    A root decoder state_dict already owns the original regional heads. A
    direct head transaction instead needs those same owners in its snapshot
    and packed-gradient collector. Prefix-aware snapshots avoid serializing
    the same tensors twice in the root safe-tensor file.
    """
    def train(self, mode=True):
        super().train(mode)
        for head in self._borrowed_learning_heads:
            head.train(mode)
        return self

    def modules(self):
        seen = set()
        for root in (self, *self._borrowed_learning_heads):
            iterator = nn.Module.modules(root) if root is self else root.modules()
            for module in iterator:
                if id(module) not in seen:
                    seen.add(id(module))
                    yield module

    def parameters(self, recurse=True):
        seen = set()
        for root in (self, *self._borrowed_learning_heads):
            iterator = nn.Module.parameters(root, recurse=recurse) if root is self else root.parameters(recurse=recurse)
            for value in iterator:
                if id(value) not in seen:
                    seen.add(id(value))
                    yield value

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        direct = destination is None and not prefix and not args
        result = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if direct:
            for index, head in enumerate(self._borrowed_learning_heads):
                # A nested joined head has borrowed grandchildren too. Ask
                # for its direct learning snapshot, then namespace it; passing
                # a root-save prefix would intentionally exclude those owners.
                for name, value in head.state_dict(keep_vars=keep_vars).items():
                    result[f"borrowed_regions.{index}." + name] = value
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        borrowed = any(key.startswith("borrowed_regions.") for key in state_dict)
        if borrowed:
            for index, head in enumerate(self._borrowed_learning_heads):
                prefix = f"borrowed_regions.{index}."
                values = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
                head.load_state_dict(values, strict=strict, assign=assign)
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith("borrowed_regions.")}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class _JoinedHead(_BorrowedLearningHead, _BorrowedRegionalModule):
    def __init__(self, regions, widths, field):
        super().__init__(regions, widths)
        object.__setattr__(self, "_borrowed_learning_heads", tuple(getattr(r.decoder, field) for r in regions))
        self.field = field

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = None
        for region, part in zip(self.regions, value.split(self.widths, dim=-1)):
            output = getattr(region.decoder, self.field)(part)
            result = output if result is None else result + output
        # Activation reduction only: all original owners remain independent.
        return result / math.sqrt(len(self.regions))


class ConnectedArgumentHead(_BorrowedLearningHead, ActionArgumentHead):
    def __init__(self, regions, widths):
        nn.Module.__init__(self)
        object.__setattr__(self, "_borrowed_learning_heads", tuple(r.decoder.action_argument_head for r in regions))
        self.dimensions = sum(widths)
        self.condition = _JoinedArgumentCondition(regions, widths)
        self.token_embedding = _JoinedArgumentEmbedding(regions, widths)
        self.transition = _JoinedArgumentTransition(regions, widths)
        self.output = _JoinedArgumentOutput(regions, widths)
        self.register_buffer("training_steps", torch.tensor(sum(int(r.decoder.action_argument_head.training_steps) for r in regions)))
        self.register_buffer("grounded_steps", torch.tensor(sum(int(r.decoder.action_argument_head.grounded_steps) for r in regions)))
        keys: dict[bytes, int] = {}
        for region in regions:
            head = region.decoder.action_argument_head
            for key, updates in zip(head.route_keys, head.grounded_route_updates):
                raw = bytes(key.detach().cpu().tolist())
                keys[raw] = max(keys.get(raw, 0), int(updates))
        self.register_buffer("route_keys", torch.tensor([list(k) for k in keys], dtype=torch.uint8).reshape(-1, 32))
        self.register_buffer("grounded_route_updates", torch.tensor(list(keys.values()), dtype=torch.long))


class ConnectedToolRouteHead(_BorrowedLearningHead, ToolRouteHead):
    def __init__(self, regions, widths):
        nn.Module.__init__(self)
        object.__setattr__(self, "_borrowed_learning_heads", tuple(r.decoder.tool_route_head for r in regions))
        object.__setattr__(self, "regions", tuple(regions))
        self.widths = tuple(widths)
        self.internal_query = _BorrowedSequence(r.decoder.tool_route_head.internal_query for r in regions)
        self.internal_query.in_features = sum(widths)
        self.candidate = _BorrowedSequence(r.decoder.tool_route_head.candidate for r in regions)
        keys: dict[bytes, torch.Tensor] = {}
        for region in regions:
            for key, features in zip(region.decoder.tool_route_head.route_keys, region.decoder.tool_route_head.route_features):
                raw = bytes(key.detach().cpu().tolist())
                if raw in keys and not torch.equal(keys[raw], features.cpu()):
                    raise ValueError("same tool identity has inconsistent structural encoding")
                keys[raw] = features.detach().cpu()
        self.register_buffer("route_keys", torch.tensor([list(k) for k in keys], dtype=torch.uint8).reshape(-1, 32))
        self.register_buffer("route_features", torch.stack(list(keys.values())) if keys else torch.empty(0, self.feature_width))
        for name in ("training_steps", "internal_training_steps", "experience_updates"):
            self.register_buffer(name, torch.tensor(sum(int(getattr(r.decoder.tool_route_head, name)) for r in regions)))
        self.register_buffer("experience_event_keys", torch.empty(0, 32, dtype=torch.uint8))
        self.register_buffer("null_features", self.encode("<no-action>"))

    def forward(self, features):
        candidates = torch.cat((self.null_features[None], self.route_features), 0)
        return self.structural_logits(features, candidates)

    def structural_logits(self, features, candidates):
        result = None
        for region in self.regions:
            head = region.decoder.tool_route_head
            if isinstance(head, ConnectedToolRouteHead):
                logits = head.structural_logits(features, candidates)
            else:
                query = F.normalize(torch.tanh(head.query(features)), dim=-1)
                encoded = F.normalize(head.candidate(candidates), dim=-1)
                logits = 16.0 * (query @ encoded.t())
            result = logits if result is None else result + logits
        return result / math.sqrt(len(self.regions))

    def forward_internal(self, neural_state, candidate_features=None):
        if neural_state.ndim != 2 or neural_state.shape[-1] != sum(self.widths):
            raise ValueError("connected tool state has the wrong width")
        features = self.route_features if candidate_features is None else candidate_features
        result = None
        for region, part in zip(self.regions, neural_state.split(self.widths, dim=-1)):
            logits = region.decoder.tool_route_head.forward_internal(part, features)
            result = logits if result is None else result + logits
        return result / math.sqrt(len(self.regions))

    def prepare_bounded_state_load(self, specs, prefix):
        # ToolRouteHead's loader expects a single query owner; the joined
        # head keeps only structural metadata and borrows regional queries.
        for name, width, dtype in (("route_keys", 32, torch.uint8),
                ("route_features", self.feature_width, torch.float32),
                ("experience_event_keys", 32, torch.uint8)):
            spec = specs.get(prefix + name)
            if spec is not None:
                if len(spec.shape) != 2 or spec.shape[1] != width or spec.dtype != dtype:
                    raise ValueError("connected tool registry shape/dtype is invalid")
                _admit(spec.byte_count, "connected tool registry", self.null_features.device)
                setattr(self, name, torch.empty(spec.shape, dtype=dtype, device=self.null_features.device))


class ConnectedOmniDecoder(OmniDecoder):
    """One autoregressive decoder with preserved, cross-connected parents."""

    def __init__(self, config: OmniConfig, regions: Sequence[NativeRegion]):
        nn.Module.__init__(self)
        if not regions:
            raise ValueError("a connected decoder needs at least one preserved region")
        ids = [r.identity.region_id for r in regions]
        if len(set(ids)) != len(ids):
            raise ValueError("connected region IDs must be unique")
        if any(r.config.vocab_size != config.vocab_size for r in regions):
            raise ValueError("connected regions must share native byte token boundaries")
        widths = tuple(r.config.d_model for r in regions)
        if config.d_model != sum(widths) or config.idea_dim != sum(widths):
            raise ValueError("connected root width must equal all preserved regional widths")
        self.config = config
        self.native_core_pager = current_native_core_pager()
        self.regions = nn.ModuleList(regions)
        self.widths = widths
        self.embedding = _JoinedEmbedding(regions, widths)
        self.final_norm = _JoinedNorm(regions, widths)
        self.language_head = _JoinedHead(regions, widths, "language_head")
        self.action_policy = _JoinedHead(regions, widths, "action_policy")
        self.internal_action_policy = _JoinedHead(regions, widths, "internal_action_policy")
        self.tool_route_head = ConnectedToolRouteHead(regions, widths)
        self.action_argument_head = ConnectedArgumentHead(regions, widths)
        self.workspace_exchange = RegionExchange(widths)
        self.cortical_exchange = RegionExchange(widths)
        self.global_workspace = _JoinedWorkspace(regions, widths, self.workspace_exchange)
        self.memory_projection = _JoinedMemoryProjection(regions, widths)
        self.workspace_strength = _BorrowedGains(regions, widths, "workspace_strength")
        self.memory_strength = _BorrowedGains(regions, widths, "memory_strength")
        self.blocks = _BorrowedSequence(block for r in self.regions for block in r.decoder.blocks)
        self.experts = _BorrowedSequence(expert for r in self.regions for expert in r.decoder.experts)
        self.expert_prototypes = _BorrowedSequence(prototype for r in self.regions for prototype in r.decoder.expert_prototypes)
        self.last_generation_cache_mode = "not-started"
        self.last_generation_stop_reason = "not-started"
        self.last_region_activity: tuple[torch.Tensor, ...] = ()
        self._origin_layout = [(len(r.decoder.blocks), r.decoder.expert_count) for r in regions]

    @property
    def expert_count(self):
        return sum(r.decoder.expert_count for r in self.regions)

    def layout(self):
        return {"format": "omni-connected-layout", "formatVersion": 1,
                "regions": [{"regionId": r.identity.region_id,
                             "layers": len(r.decoder.blocks), "experts": r.decoder.expert_count,
                             **({"nested": r.decoder.layout()} if isinstance(r.decoder, ConnectedOmniDecoder) else {})}
                            for r in self.regions]}

    def _refresh_views(self):
        self.blocks = _BorrowedSequence(block for r in self.regions for block in r.decoder.blocks)
        self.experts = _BorrowedSequence(expert for r in self.regions for expert in r.decoder.experts)
        self.expert_prototypes = _BorrowedSequence(prototype for r in self.regions for prototype in r.decoder.expert_prototypes)

    def grow_expert(self, prototype=None):
        pieces = ([None] * len(self.regions) if prototype is None else prototype.reshape(-1).split(self.widths))
        selected = (min(range(len(self.regions)), key=lambda i: self.regions[i].decoder.expert_count)
                    if prototype is None else max(range(len(self.regions)), key=lambda i: float(pieces[i].abs().mean())))
        self.regions[selected].decoder.grow_expert(pieces[selected])
        self._refresh_views()
        return self.expert_count - 1

    def grow_depth(self, add_layers):
        index = max(range(len(self.regions)), key=lambda i: len(self.regions[i].decoder.blocks))
        result = self.regions[index].decoder.grow_depth(add_layers)
        self._refresh_views()
        return result

    def restore_layout(self, layout):
        if not isinstance(layout, Mapping) or layout.get("format") != "omni-connected-layout" or layout.get("formatVersion") != 1:
            raise ValueError("connected topology checkpoint is invalid")
        rows = layout.get("regions")
        if not isinstance(rows, list) or len(rows) != len(self.regions):
            raise ValueError("connected topology lost a parent region")
        for index, (row, region) in enumerate(zip(rows, self.regions)):
            baseline_layers, baseline_experts = self._origin_layout[index]
            if not isinstance(row, Mapping) or row.get("regionId") != region.identity.region_id or any(type(row.get(name)) is not int for name in ("layers", "experts")) or row["layers"] < baseline_layers or row["experts"] < baseline_experts:
                raise ValueError("connected topology cannot remove a preserved parent owner")
        for row, region in zip(rows, self.regions):
            decoder = region.decoder
            if isinstance(decoder, ConnectedOmniDecoder):
                decoder.restore_layout(row.get("nested"))
                if len(decoder.blocks) != row["layers"] or decoder.expert_count != row["experts"]:
                    raise ValueError("nested connected topology counters disagree with its preserved owners")
                continue
            if "nested" in row:
                raise ValueError("ordinary parent cannot be replaced by an undeclared nested region")
            while decoder.expert_count < row["experts"]:
                decoder.grow_expert()
            if len(decoder.blocks) < row["layers"]:
                decoder.grow_depth(row["layers"] - len(decoder.blocks))
            removed = [*list(decoder.blocks)[row["layers"]:], *list(decoder.experts)[row["experts"]:], *list(decoder.expert_prototypes)[row["experts"]:]]
            for owner in removed:
                for child in owner.modules():
                    pager = getattr(child, "_native_core_pager", None)
                    if pager is not None:
                        pager.retire(child)
            decoder.blocks = nn.ModuleList(list(decoder.blocks)[:row["layers"]])
            decoder.experts = nn.ModuleList(list(decoder.experts)[:row["experts"]])
            decoder.expert_prototypes = nn.ModuleList(list(decoder.expert_prototypes)[:row["experts"]])
        self._refresh_views()

    def configure_working_attention(self, pager):
        self.working_attention_pager = pager
        for region in self.regions:
            region.decoder.configure_working_attention(pager)

    def _forward_window_bytes(self, batch, tokens, element_bytes=4):
        return sum(r.decoder._forward_window_bytes(batch, tokens, element_bytes) for r in self.regions) + batch * tokens * self.config.d_model * max(4, element_bytes) * 12

    def _forward_bounded(self, input_ids, memory_bias=None, labels=None,
                         use_global_workspace=None, attention_mask=None):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("connected input must fit the complete selected working window")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or
                bool((attention_mask[:, 1:].bool() & ~attention_mask[:, :-1].bool()).any()) or
                not bool(attention_mask.bool().any(dim=1).all())):
            raise ValueError("connected attention mask must describe nonempty right padding")
        if labels is not None and (labels.shape != input_ids.shape or labels.shape[1] < 2):
            raise ValueError("connected labels must describe the full causal window")
        hidden = self.embedding(input_ids)
        parts = list(hidden.split(self.widths, dim=-1))
        if memory_bias is not None:
            if memory_bias.ndim == 1:
                memory_bias = memory_bias.unsqueeze(0)
            if memory_bias.shape[-1] != self.config.idea_dim:
                raise ValueError("connected memory cue has the wrong width")
        cues = (memory_bias.split(self.widths, dim=-1) if memory_bias is not None else [None] * len(parts))
        prepared = []
        for region, part, cue in zip(self.regions, parts, cues):
            decoder = region.decoder
            nested = isinstance(decoder, ConnectedOmniDecoder)
            if nested:
                # An already merged parent is a connected region in its own
                # right. Do not flatten away its learned inner exchanges,
                # auxiliary pathways, or heterogeneous block widths.
                part = decoder(input_ids, memory_bias=cue,
                    use_global_workspace=use_global_workspace, attention_mask=attention_mask)["hidden"]
            elif use_global_workspace is not False:
                part = part + decoder.workspace_strength() * decoder.global_workspace.causal_summaries(part, attention_mask=attention_mask)
            # Every parent's real temporal/router/graph pathways are used.
            part = part + 0.125 * region.condition(part)
            if cue is not None and not nested:
                part = part + decoder.memory_strength() * decoder.memory_projection(cue).unsqueeze(1)
            prepared.append(part)
        hidden = self.workspace_exchange(torch.cat(prepared, dim=-1))
        for depth in range(max((len(r.decoder.blocks) for r in self.regions if not isinstance(r.decoder, ConnectedOmniDecoder)), default=1)):
            parts = hidden.split(self.widths, dim=-1)
            updated = [region.decoder.blocks[depth](part) if not isinstance(region.decoder, ConnectedOmniDecoder) and depth < len(region.decoder.blocks) else part
                       for region, part in zip(self.regions, parts)]
            hidden = self.cortical_exchange(torch.cat(updated, dim=-1))
        final = []
        for region, part in zip(self.regions, hidden.split(self.widths, dim=-1)):
            value = part
            if not isinstance(region.decoder, ConnectedOmniDecoder):
                value, _ = region.decoder._apply_experts(part, attention_mask=attention_mask)
            final.append(region.decoder.final_norm(value))
        hidden = torch.cat(final, dim=-1)
        self.last_region_activity = tuple(value[:, -1].detach() for value in final)
        output = {"logits": self.language_head(hidden), "hidden": hidden,
                  "action_logits": self.action_policy(hidden)}
        if use_global_workspace is True:
            output["workspace"] = hidden.mean(dim=1, keepdim=True)
        if labels is not None:
            output["loss"] = F.cross_entropy(output["logits"][:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index=0)
        return output

    def _generation_step_bounded(self, window, memory_bias, cache):
        if torch.is_grad_enabled():
            raise RuntimeError("connected generation requires no_grad")
        if window.ndim != 2 or not 1 <= window.shape[1] <= self.config.max_seq_len:
            raise ValueError("connected generation window is invalid")
        signature = self._generation_signature(memory_bias)
        if self.training or signature is None:
            if cache is not None:
                cache.close()
            output = self.forward(window, memory_bias=memory_bias)
            self.last_generation_neural_state = output["hidden"][:, -1].detach().clone()
            return output["logits"][:, -1], None
        length, batch = window.shape[1], window.shape[0]
        if cache is not None and (cache.signature != signature or cache.length + 1 != length or
                cache.hidden_sum[0].shape[0] != batch):
            cache.close()
            cache = None
        first = cache is None
        if cache is None:
            cache = _ConnectedInferenceCache(0, signature,
                [[] for _ in self.regions],
                [[None] * len(region.decoder.blocks) for region in self.regions],
                [None for _ in self.regions],
                [torch.zeros(batch, width, device=window.device) for width in self.widths],
                [None for _ in self.regions])
        cues = [None] * len(self.regions)
        if memory_bias is not None:
            if memory_bias.ndim == 1:
                memory_bias = memory_bias.unsqueeze(0)
            if memory_bias.shape[-1] != sum(self.widths):
                cache.close()
                raise ValueError("connected generation cue has the wrong width")
            cues = list(memory_bias.split(self.widths, -1))
        pager = self._working_pager()
        try:
            # One causal token frontier per step. Region KV stores use the
            # shared WorkingAttentionPager, so long windows spill instead of
            # requiring a complete prefix hidden/logit allocation in RAM.
            for position in range(0 if first else length - 1, length):
                _admit(self._forward_window_bytes(batch, 1), "connected causal token frontier", window.device)
                prepared = []
                for index, (region, cue) in enumerate(zip(self.regions, cues)):
                    decoder = region.decoder
                    nested = isinstance(decoder, ConnectedOmniDecoder)
                    if nested:
                        _, child_cache = decoder._generation_step_bounded(window[:, :position + 1], cue, cache.nested[index])
                        cache.nested[index] = child_cache
                        hidden = decoder.last_generation_neural_state.unsqueeze(1)
                    else:
                        hidden = decoder.embedding(window[:, position:position + 1])
                        if not cache.workspace[index]:
                            summary = decoder.global_workspace.causal_summaries(hidden, _inference_cache=cache.workspace[index])
                        else:
                            summary = decoder.global_workspace.causal_step(hidden, cache.workspace[index])
                        hidden = hidden + decoder.workspace_strength() * summary
                    condition, state = region.condition(hidden, recurrent_state=cache.temporal[index], return_state=True)
                    cache.temporal[index] = state
                    hidden = hidden + 0.125 * condition
                    if cue is not None and not nested:
                        hidden = hidden + decoder.memory_strength() * decoder.memory_projection(cue).unsqueeze(1)
                    prepared.append(hidden)
                hidden = self.workspace_exchange(torch.cat(prepared, -1))
                for depth in range(max((len(region.decoder.blocks) for region in self.regions if not isinstance(region.decoder, ConnectedOmniDecoder)), default=1)):
                    updated = []
                    for index, (region, part) in enumerate(zip(self.regions, hidden.split(self.widths, -1))):
                        if not isinstance(region.decoder, ConnectedOmniDecoder) and depth < len(region.decoder.blocks):
                            part, layer = region.decoder.blocks[depth].forward_cached(part, cache.attention[index][depth])
                            cache.attention[index][depth] = layer
                        updated.append(part)
                    hidden = self.cortical_exchange(torch.cat(updated, -1))
                for index, part in enumerate(hidden.split(self.widths, -1)):
                    cache.hidden_sum[index] = cache.hidden_sum[index] + part[:, 0]
                if pager is not None:
                    pager.note_prefill(1)
                final = []
            for index, (region, part) in enumerate(zip(self.regions, hidden.split(self.widths, -1))):
                if not isinstance(region.decoder, ConnectedOmniDecoder):
                    part, _ = region.decoder._apply_experts(part, pooled_hidden=cache.hidden_sum[index] / float(length))
                final.append(region.decoder.final_norm(part))
            combined = torch.cat(final, -1)
            self.last_generation_neural_state = combined[:, -1].detach().clone()
            self.last_region_activity = tuple(part[:, -1].detach() for part in final)
            cache.length = int(length)
            return self.language_head(combined)[:, -1], cache
        except BaseException:
            cache.close()
            raise

    def encode_whole(self, input_ids):
        return self.forward(input_ids, use_global_workspace=True)["workspace"].mean(dim=1)

    def integration_loss(self):
        if not self.last_region_activity:
            raise ValueError("integration needs a real preceding neural activity sample")
        return self.workspace_exchange.alignment_loss(self.last_region_activity) + self.cortical_exchange.alignment_loss(self.last_region_activity)

    def observe_experience(self, neural_activity, *, event_key: str, importance=0.5):
        """Called once inside the root's admitted-experience transaction."""
        if not event_key or "\0" in event_key:
            raise ValueError("connected experience requires its durable root receipt ID")
        if neural_activity.shape[-1] != self.config.d_model:
            raise ValueError("connected experience has the wrong common width")
        return [region.observe(part, event_key=event_key, importance=importance)
                for region, part in zip(self.regions, neural_activity.split(self.widths, -1))]


class _ConnectedMediaModule(nn.Module):
    def __init__(self, hub, modality):
        super().__init__()
        object.__setattr__(self, "hub", hub)
        self.modality = modality
        self.image_size = hub.config.image_size
        self.samples = hub.config.audio_samples
        self.frames = hub.config.video_frames

    def encode(self, value):
        return self.hub.perception_embedding(self.modality, value)

    def forward(self, value, idea=None):
        if self.modality == "vision":
            return self.hub.perception_embedding("image", value)
        if idea is None:
            raise ValueError("connected modality training requires its shared idea")
        conditioned = self.hub.connected_decoder.workspace_exchange(idea)
        weights = self.hub.selector(conditioned).softmax(-1)
        outputs = []
        for region, part in zip(self.hub.connected_decoder.regions, conditioned.split(self.hub.widths, -1)):
            target = self.hub._resize(value, self.modality, region.config)
            outputs.append(getattr(region.modalities, self.modality)(target, part))
        loss = torch.stack([item["loss"] for item in outputs]).mean()
        reconstruction = None
        for index, item in enumerate(outputs):
            value = self.hub._resize(item["reconstruction"], self.modality, self.hub.config)
            scaled = value * weights[:, index].reshape(-1, *([1] * (value.ndim - 1)))
            reconstruction = scaled if reconstruction is None else reconstruction + scaled
        embedding = self.hub.connected_decoder.workspace_exchange(torch.cat([item["embedding"] for item in outputs], -1))
        return {"loss": loss, "reconstruction": reconstruction, "embedding": embedding}

    def generate(self, idea, generator, **options):
        return self.hub._generate(self.modality, idea, generator, **options)


class ConnectedModalityHub(ModalityHub):
    """Shared imagination/perception across preserved parent sensory regions."""

    def __init__(self, decoder: ConnectedOmniDecoder):
        nn.Module.__init__(self)
        object.__setattr__(self, "connected_decoder", decoder)
        self.config = decoder.config
        self.widths = decoder.widths
        self.selector = PackedAdaptiveBitLinear(self.config.idea_dim, len(decoder.regions), bias=True)
        self.vision = _ConnectedMediaModule(self, "vision")
        self.image = _ConnectedMediaModule(self, "image")
        self.audio = _ConnectedMediaModule(self, "audio")
        self.video = _ConnectedMediaModule(self, "video")

    @staticmethod
    def _resize(value, modality, config):
        target = ((config.audio_samples,) if modality == "audio" else
                  (config.video_frames, config.image_size, config.image_size) if modality == "video" else
                  (config.image_size, config.image_size))
        squeezed = modality == "audio" and value.ndim == 2
        value = value.unsqueeze(1) if squeezed else value
        if tuple(value.shape[2:]) != target:
            value = F.interpolate(value, size=target, mode="nearest")
        return value.squeeze(1) if squeezed else value

    def imagination_logits(self, idea):
        result = None
        for region, part in zip(self.connected_decoder.regions, idea.split(self.widths, -1)):
            value = region.modalities.imagination_logits(part)
            result = value if result is None else result + value
        return result / math.sqrt(len(self.widths))

    @torch.no_grad()
    def select_imagination(self, idea, enabled=None):
        logits = self.imagination_logits(idea)[0].float()
        enabled = set(IMAGINATION_MODALITIES if enabled is None else enabled)
        for i, name in enumerate(IMAGINATION_MODALITIES):
            if name not in enabled:
                logits[i] = -torch.inf
        if not bool(torch.isfinite(logits).any()):
            raise ValueError("no imagination modality is enabled")
        probabilities = logits.softmax(-1)
        return IMAGINATION_MODALITIES[int(probabilities.argmax())], {name: float(probabilities[i]) for i, name in enumerate(IMAGINATION_MODALITIES)}

    @measured_perception
    def perception_embedding(self, modality, tensor):
        values = []
        for region in self.connected_decoder.regions:
            values.append(region.modalities.perception_embedding(modality, self._resize(tensor, modality, region.config)))
        return self.connected_decoder.workspace_exchange(torch.cat(values, -1))

    @torch.no_grad()
    def generate(self, modality, idea, seed=0, preview_callback=None,
                 cancel_check=None, maximum_previews=None):
        from .media_randomness import media_generator
        generator = media_generator(idea.device, seed)
        return self._generate(modality, idea, generator, preview_callback=preview_callback,
                              cancel_check=cancel_check, maximum_previews=maximum_previews)

    def _generate(self, modality, idea, generator, preview_callback=None,
                  cancel_check=None, maximum_previews=None, **options):
        conditioned = self.connected_decoder.workspace_exchange(idea)
        weights = self.selector(conditioned).softmax(-1)
        combined = None
        # Serial resource-admitted execution keeps one output plus one region
        # working set; all regions contribute to one shared modality output.
        for index, (region, part) in enumerate(zip(self.connected_decoder.regions, conditioned.split(self.widths, -1))):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("connected imagination stopped")
            generated = getattr(region.modalities, modality).generate(part, generator,
                cancel_check=cancel_check, maximum_previews=0, **options)
            generated = self._resize(generated, modality, self.config)
            _admit(generated.numel() * generated.element_size() * 3,
                   "connected imagination output", generated.device)
            value = generated * weights[:, index].reshape(-1, *([1] * (generated.ndim - 1)))
            combined = value if combined is None else combined + value
        if preview_callback is not None:
            preview_callback(1.0, combined)
        return combined
