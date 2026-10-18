"""Reviewed MCP sensory sampling on the existing native brain, without learning.

The caller already owns the single neural writer. Server messages are the
only token input: no system/persona text, recent chat, retrieved source text,
foreign model, or external action callback is added. Packed learned owners
are shared read-only. Small recurrent activity, recall inspection and native
generation caches belong to this invocation and are discarded afterwards.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch.nn import functional as F

from .cooperative_generation import CooperativeNativeGeneration
from .parameter_diagnostics import native_parameter_read_only
from .working_attention_paging import WorkingAttentionCancelled
from .mcp_sampling_media import NativeSamplingMedia, media_block, sensory_windows
from .streamed_vsa_activity import activity_reduction_scratch_bytes


FORMAT = "omni-native-mcp-sampling-v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")


class NativeSamplingCancelled(RuntimeError):
    """No private output is delivered when its owning task is interrupted."""


def _text(value: Any, *, identifier=False) -> str:
    if not isinstance(value, str) or "\x00" in value or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("native sampling requires valid Unicode text")
    if identifier and _ID.fullmatch(value) is None:
        raise ValueError("native sampling request ownership is invalid")
    return value


@dataclass(frozen=True)
class NativeSamplingInput:
    brain_id: str
    request_id: str
    turn_id: str
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    temperature: float | None
    stop_sequences: tuple[str, ...]
    input_sha256: str
    prompt_tokens: int
    media: tuple[NativeSamplingMedia, ...] = ()


def parse_approved_sampling(value: Mapping[str, Any]) -> NativeSamplingInput:
    """Authenticate the exact reviewed JSON bytes, not a reserialized float."""
    fields = {"brainId", "requestId", "inputSha256", "approvedInputJson"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("native sampling transport contains unreviewed fields")
    raw, digest = value["approvedInputJson"], value["inputSha256"]
    if not isinstance(raw, str) or not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise ValueError("native sampling approval checksum is invalid")
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise ValueError("native sampling input differs from its reviewed bytes")
    def invalid(_value):
        raise ValueError("native sampling input must be finite JSON")
    approved = json.loads(raw, parse_constant=invalid)
    required = {"format", "brainId", "requestId", "messages", "maxTokens"}
    optional = {"originatingTurnId", "temperature", "stopSequences"}
    if not isinstance(approved, dict) or not required <= set(approved) or set(approved) - required - optional:
        raise ValueError("native sampling approval contains unsupported instructions or fields")
    if approved["format"] != FORMAT:
        raise ValueError("native sampling approval version is unsupported")
    brain_id, request_id = _text(approved["brainId"], identifier=True), _text(approved["requestId"], identifier=True)
    if brain_id != value["brainId"] or request_id != value["requestId"]:
        raise ValueError("native sampling transport changed its approved owner")
    turn = _text(approved["originatingTurnId"], identifier=True) if "originatingTurnId" in approved else ""
    count = approved["maxTokens"]
    if type(count) is not int or not 1 <= count <= (1 << 53) - 1:
        raise ValueError("native sampling maxTokens must be a positive safe integer")
    temperature = approved.get("temperature")
    if temperature is not None and (type(temperature) not in {int, float} or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("native sampling temperature must be finite and nonnegative")
    messages = approved["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("native sampling needs a reviewed message")
    rows, media, prompt_tokens = [], [], 2  # one BOS and the response role marker
    for message_index, row in enumerate(messages):
        if (not isinstance(row, dict) or set(row) != {"role", "content"} or not isinstance(row["role"], str)
                or row["role"] not in {"user", "assistant"}):
            raise ValueError("native MCP sampling requires typed reviewed message blocks")
        blocks = row["content"] if isinstance(row["content"], list) else [row["content"]]
        if not blocks:
            raise ValueError("native MCP sampling message blocks cannot be empty")
        pieces = []
        for block_index, block in enumerate(blocks):
            if isinstance(block, dict) and set(block) == {"type", "text"} and block["type"] == "text":
                pieces.append(_text(block["text"]))
            else:
                media.append(media_block(block, message_index, block_index))
        text = "".join(pieces)
        rows.append((row["role"], text))
        prompt_tokens += 1 + len(text.encode("utf-8")) + int(row["role"] == "assistant")
    if not media and not any(text.strip() for _role, text in rows):
        raise ValueError("empty sampling approval cannot enqueue native work")
    stops = approved.get("stopSequences", [])
    if not isinstance(stops, list) or any(not isinstance(item, str) or not item for item in stops):
        raise ValueError("native sampling stop sequences must be nonempty text")
    return NativeSamplingInput(brain_id, request_id, turn, tuple(rows), count,
        None if temperature is None else float(temperature), tuple(_text(item) for item in stops), digest, prompt_tokens, tuple(media))


def _memory_view(memory, cancelled):
    """Shared authoritative stores, private derived readout/frontier owners."""
    view = copy.copy(memory)
    view._paged_recall_graph = None
    view._last_recall_paged_state = None
    view._paged_idle_selector = None
    guard = memory.growth_guard
    def private_admission(size):
        if cancelled():
            raise NativeSamplingCancelled("MCP sampling owner was interrupted during neural recall")
        return guard(size) if guard is not None else True
    view.growth_guard = private_admission
    return view


def _private_recall(memory, cue, slots):
    """Consume a readout vector, then release only its disposable frontier."""
    try:
        return memory.recall_vector(cue, workspace_slots=slots, record_activity=False)[0]
    finally:
        state = getattr(memory, "_last_recall_paged_state", None)
        if state is not None:
            memory._last_recall_paged_state = None
            state.close()


@contextmanager
def private_sampling_activity(brain, *, cancelled=lambda: False):
    """Swap small activity owners; never copy or rewrite learned weights."""
    from .connected_regions import NativeRegion

    roots = [getattr(brain, name) for name in ("decoder", "memory_bridge", "idea_adapter", "liquid", "router", "modalities")]
    modules = {id(module): module for root in roots for module in root.modules()}
    modes = [(module, module.training) for module in modules.values()]
    owners = [brain, *(module for module in modules.values() if isinstance(module, NativeRegion))]
    controls = []
    for owner in owners:
        if isinstance(getattr(owner, "liquid_state", None), torch.Tensor):
            controls.append((owner, "liquid_state", owner.liquid_state))
        population = getattr(getattr(owner, "router", None), "population", None)
        for name in ("membrane", "spike_count"):
            if isinstance(getattr(population, name, None), torch.Tensor):
                controls.append((population, name, getattr(population, name)))
    previous = [(module, "last_region_activity", module.last_region_activity)
                for module in modules.values() if hasattr(module, "last_region_activity")]
    original_memory = brain.memory
    memories = {id(original_memory): original_memory}
    connected = getattr(brain, "_connected_regions", None)
    if connected is not None:
        memories.update((id(memory), memory) for _name, memory in connected.iter_memories())
    views = {key: _memory_view(memory, cancelled) for key, memory in memories.items()}
    absent = object()
    observers = [(store, getattr(store, "_recall_graph_observer", absent)) for memory in memories.values()
                 for store in (memory.neurons, memory.synapses)]
    callbacks = []
    def install_regional_views(regions):
        if regions is None:
            return
        for region in regions.decoder.regions:
            memory = regions.memories[region.identity.region_id]
            callbacks.append((region, region._recall))
            region.attach_recall(lambda cue, view=views[id(memory)], slots=region.config.working_memory_slots:
                                 _private_recall(view, cue, slots))
        for nested in (regions.nested or {}).values():
            install_regional_views(nested)

    size = 16384 + len(modules) * 256 + len(views) * 2048 + sum(value.numel() * value.element_size() for _owner, _name, value in controls)
    with brain.resource_policy.reserve_ram(size, "MCP private native activity owners"):
        cpu_rng, python_rng = torch.random.get_rng_state(), random.getstate()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        mps_rng = torch.mps.get_rng_state() if brain.device.type == "mps" else None
        try:
            for owner, name, value in controls:
                setattr(owner, name, value.detach().clone())
            for root in roots:
                root.eval()
            brain.memory = views[id(original_memory)]
            install_regional_views(connected)
            with native_parameter_read_only():
                yield
        finally:
            def restore_observer(store, observer):
                if observer is absent:
                    if hasattr(store, "_recall_graph_observer"):
                        del store._recall_graph_observer
                else:
                    store._recall_graph_observer = observer
            # Every owner is restored even if closing a disposable recall
            # database fails. A cleanup failure must not strand private state
            # in the original live turn.
            with ExitStack() as cleanup:
                cleanup.callback(torch.random.set_rng_state, cpu_rng)
                cleanup.callback(random.setstate, python_rng)
                if cuda_rng is not None:
                    cleanup.callback(torch.cuda.set_rng_state_all, cuda_rng)
                if mps_rng is not None:
                    cleanup.callback(torch.mps.set_rng_state, mps_rng)
                for module, mode in modes:
                    cleanup.callback(setattr, module, "training", mode)
                for owner, name, value in [*controls, *previous]:
                    cleanup.callback(setattr, owner, name, value)
                for store, observer in observers:
                    cleanup.callback(restore_observer, store, observer)
                for region, callback in callbacks:
                    cleanup.callback(region.attach_recall, callback)
                cleanup.callback(setattr, brain, "memory", original_memory)
                for view in views.values():
                    cleanup.callback(view.release_paged_recall_scratch)


@torch.no_grad()
def _sample_native_mcp(brain, request: NativeSamplingInput, *, cancelled):
    if request.brain_id != brain.brain_id:
        raise ValueError("MCP sampling cannot substitute another brain identity")
    if request.prompt_tokens > int(brain.config.max_seq_len):
        raise ValueError("MCP sampling input exceeds selected working memory; it was not truncated")
    if cancelled():
        raise NativeSamplingCancelled("MCP sampling owner was interrupted before native work")
    for block in request.media:
        if block.kind == "image" and not (brain.config.vision_enabled or brain.config.image_enabled):
            raise ValueError("this brain's native image perception is disabled")
        if block.kind == "audio" and not brain.config.audio_enabled:
            raise ValueError("this brain's native audio perception is disabled")
    # Native generation can roll its private token window, so maxTokens is not
    # silently capped to context capacity. Reserve the full requested output
    # and refuse honestly if the actual RAM/spill budget cannot admit it.
    byte_count = 32768 + (request.prompt_tokens + request.max_tokens) * 64
    pager = brain.decoder._working_pager()
    # Replace the outer chat's cancellation callback while privately reading
    # native regions. Otherwise a tiled conditioning read can recursively
    # service another sampling request through the main turn's control tick.
    scope = pager.activate(cancelled) if pager is not None else nullcontext()
    with brain.resource_policy.reserve_ram(byte_count, "reviewed MCP native tokens and result"), \
            private_sampling_activity(brain, cancelled=cancelled), scope:
        tokens = [brain.tokenizer.bos_id]
        for role, text in request.messages:
            tokens.append(brain.tokenizer.human_id if role == "user" else brain.tokenizer.brain_id)
            tokens.extend(brain.tokenizer.encode(text))
            if role == "assistant":
                tokens.append(brain.tokenizer.eos_id)
        tokens.append(brain.tokenizer.brain_id)
        prompt = torch.tensor([tokens], dtype=torch.long, device=brain.device)
        # Same learned VSA/graph, liquid, LIF and cortical owners as ordinary
        # generation; this read does not record usage, STDP, episodes or replay.
        with brain.resource_policy.reserve_ram(request.prompt_tokens * 192 + 8192, "native MCP explicit cue labels"):
            labels = brain.memory.extract_concepts("\n".join(text for _role, text in request.messages))
            dimensions = brain.memory.space.dimensions
            row_bytes = dimensions * 256 + 8192
            records = getattr(brain.memory.neuron_vectors, "parallel_records", None)
            if callable(records):
                for label in labels:
                    key = hashlib.sha256(("semantic:" + label).encode("utf-8")).hexdigest()[:24]
                    if key in brain.memory.neuron_vectors:
                        row_bytes = max(row_bytes, dimensions * 256 + len(records(key)) * 8 + 8192)
            cue_bytes = activity_reduction_scratch_bytes(dimensions, len(labels)) + row_bytes
            with brain.resource_policy.reserve_ram(cue_bytes, "native MCP bounded VSA reduction and weighted-row decoding"):
                cue = (brain.memory.vector_for_labels(labels) if labels or any(text.strip() for _role, text in request.messages)
                       else torch.zeros(dimensions))
            del labels
        recalled = _private_recall(brain.memory, cue, brain.config.working_memory_slots)
        idea = brain._idea_model_vector(cue)
        _state, liquid = brain.liquid(idea, state=brain.liquid_state, elapsed=1.0)
        routed, _metrics = brain.router.route(idea, learn=False,
            steps=max(2, int(round(float(liquid["ponder_scale"].mean())))),
            threshold_offset=float(liquid["threshold_offset"].mean()))
        memory_bias = brain.idea_adapter(.65 * routed + .35 * brain._idea_model_vector(recalled))
        media_receipts = []
        recurrent = brain.liquid_state
        for block in request.media:
            receipt = {}
            windows = sensory_windows(brain, block, cancelled, receipt)
            try:
                for target, elapsed in windows:
                    if cancelled():
                        raise NativeSamplingCancelled("MCP sensory sampling was interrupted")
                    # The existing same-brain sensory region is the encoder.
                    # No generated caption, source filename, or base64 byte is
                    # converted to hidden language tokens or a learned episode.
                    embedding = brain.modalities.perception_embedding(block.kind, target)
                    if embedding.ndim != 2 or embedding.shape != (1, brain.config.idea_dim) or not bool(torch.isfinite(embedding).all()):
                        raise ValueError("native MCP perception returned invalid shared sensory activity")
                    sensory = brain.memory_bridge.transpose_projection(embedding)[0].detach().cpu()
                    sensory = F.normalize(sensory, dim=0)
                    recalled_sensory = _private_recall(brain.memory, sensory, brain.config.working_memory_slots)
                    recurrent, controls = brain.liquid(embedding, state=recurrent, elapsed=elapsed)
                    routed_sensory, _ = brain.router.route(embedding + recurrent, learn=False,
                        steps=max(2, int(round(float(controls["ponder_scale"].mean())))),
                        threshold_offset=float(controls["threshold_offset"].mean()))
                    current = brain.idea_adapter(.65 * routed_sensory + .35 * brain._idea_model_vector(recalled_sensory))
                    retention = controls["retention"].reshape(-1, 1).clamp(0, 1)
                    memory_bias = (1 - retention) * memory_bias + retention * current
                    receipt["nativeSensory"] = True
            finally:
                windows.close()
            if not receipt.get("nativeSensory") or not receipt.get("wholeInputTraversed"):
                raise RuntimeError("native MCP sensory input did not finish its actual window traversal")
            media_receipts.append(receipt)
        stopped_sequence = None
        seen_tokens, emitted_characters, stop_tail = 0, 0, ""
        retained_characters = max((len(stop) for stop in request.stop_sequences), default=1) - 1
        session = CooperativeNativeGeneration(brain.decoder, prompt, capture_tokens=bool(request.stop_sequences),
            memory_bias=memory_bias, max_new_tokens=request.max_tokens,
            temperature=brain.config.temperature if request.temperature is None else request.temperature,
            top_k=brain.config.top_k, noise=0.0, seed=int(request.input_sha256[:16], 16) & 0x7FFFFFFF,
            printable_only=True, cancelled=cancelled)
        try:
            while not session.done:
                session.advance()
                if cancelled():
                    raise NativeSamplingCancelled("MCP sampling owner was stopped, steered or unloaded")
                if request.stop_sequences and stopped_sequence is None:
                    # Decode only newly complete scalar bytes. Re-decoding the
                    # entire prefix on every token would make long sampling
                    # quadratic; a suffix of max-stop-length minus one is
                    # sufficient for matches crossing emitted UTF-8 chunks.
                    chunk = brain.tokenizer.decode(session.emitted_token_ids[seen_tokens:])
                    seen_tokens = len(session.emitted_token_ids)
                    current = stop_tail + chunk
                    offset = emitted_characters - len(stop_tail)
                    matches = [(offset + current.find(stop), stop) for stop in request.stop_sequences if stop in current]
                    if matches:
                        stopped_sequence = min(matches, key=lambda item: item[0])
                        break  # exact complete UTF-8 emission boundary
                    emitted_characters += len(chunk)
                    stop_tail = current[-retained_characters:] if retained_characters else ""
            reason = session.stop_reason
            if stopped_sequence is not None:
                output = brain.tokenizer.decode(session.emitted_token_ids)[:stopped_sequence[0]]
                stop_reason = "stopSequence"
            else:
                if session.result is None:
                    raise RuntimeError("native sampling ended without an actual native result")
                output = brain.tokenizer.decode(session.result[0][0, prompt.shape[1]:].tolist())
                if reason == "learned-boundary":
                    stop_reason = "endTurn"
                elif reason == "token-budget":
                    stop_reason = "maxTokens"
                else:
                    raise NativeSamplingCancelled("native MCP sampling did not complete: " + str(reason))
            return {"role": "assistant", "content": {"type": "text", "text": output},
                "model": "OmniCortex/" + brain.brain_id, "stopReason": stop_reason,
                "inputSha256": request.input_sha256, "requestId": request.request_id,
                "brainId": request.brain_id, "privateWorkspace": True, "parametersMutated": False,
                **({"mediaInput": media_receipts} if media_receipts else {})}
        finally:
            session.close()


def sample_native_mcp(brain, request: NativeSamplingInput, *, cancelled: Callable[[], bool] = lambda: False):
    try:
        return _sample_native_mcp(brain, request, cancelled=cancelled)
    except WorkingAttentionCancelled as error:
        if cancelled():
            raise NativeSamplingCancelled("MCP sampling owner was interrupted at a native tile boundary") from error
        raise
    except InterruptedError as error:
        if cancelled():
            raise NativeSamplingCancelled("MCP sensory input was interrupted at a decoder boundary") from error
        raise
