"""Ordered sensory encoding for one ordinary connected-App experience.

No captions, raw-source prompt injection, selector optimizer or foreign model.
Decoded signals are transient activity; only ordinary admission learns them.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, nullcontext

import torch
from torch.nn import functional as F

from .app_observation_audit import AppObservationAuditWriter
from .mcp_sampling_media import IMAGE_MIMES, AUDIO_MIMES, media_block, sensory_windows
from .parameter_diagnostics import native_parameter_read_only
from .working_attention_paging import WorkingAttentionCancelled


def _text(value):
    if not isinstance(value, str):
        raise ValueError("App content text must be a Unicode string")
    value.encode("utf-8")
    if "\0" in value:
        raise ValueError("App content text contains a NUL character")
    return value


def _metadata(block):
    for key in ("annotations", "_meta"):
        if key in block and not isinstance(block[key], dict):
            raise ValueError("App content annotations must be typed metadata")


def task_blocks(data):
    """Every logical block, then optional structured data, in source order."""
    params = data["params"]
    raw = params.get("content", [])
    raw = raw if isinstance(raw, list) else [raw]
    for block in raw:
        if not isinstance(block, dict):
            raise ValueError("App content must be a typed block")
        _metadata(block)
        kind = block.get("type")
        common = {"annotations", "_meta"}
        if kind == "text" and set(block) <= common | {"type", "text"}:
            yield "text", _text(block.get("text")), None
        elif kind in {"image", "audio"} and set(block) <= common | {"type", "data", "mimeType"}:
            yield kind, None, {key: block[key] for key in ("type", "data", "mimeType")}
        elif kind == "resource" and set(block) <= common | {"type", "resource"}:
            resource = block.get("resource")
            if (not isinstance(resource, dict) or set(resource) - {"uri", "mimeType", "_meta", "text", "blob"}
                    or not _text(resource.get("uri")) or ("text" in resource) == ("blob" in resource)):
                raise ValueError("App inline resource has an invalid typed body")
            if "_meta" in resource and not isinstance(resource["_meta"], dict):
                raise ValueError("App inline resource metadata must be a typed object")
            if "mimeType" in resource:
                _text(resource["mimeType"])
            if "text" in resource:
                yield "resource-text", _text(resource["text"]), None
            else:
                mime = _text(resource.get("mimeType"))
                modality = "image" if mime.lower() in IMAGE_MIMES else "audio" if mime.lower() in AUDIO_MIMES else None
                if modality is None:
                    raise ValueError("App binary resource is unsupported; it was not fetched or learned")
                yield modality, None, {"type": modality, "data": resource["blob"], "mimeType": mime}
        elif kind == "resource_link" and set(block) <= {"type", "uri", "name", "title", "description", "mimeType", "size", "icons", "annotations", "_meta"}:
            if not _text(block.get("uri")) or not _text(block.get("name")):
                raise ValueError("App resource reference requires its actual URI and name")
            # This is the literal reviewed reference, not the linked contents.
            yield "resource-reference", json.dumps(block, ensure_ascii=False, separators=(",", ":"), allow_nan=False), None
        else:
            raise ValueError("Unsupported App content block; no block was silently dropped")
    if "structuredContent" in params:
        yield "structured", json.dumps(params["structuredContent"], ensure_ascii=False, separators=(",", ":"), allow_nan=False), None


def needs_content_path(data):
    params = data["params"]
    blocks = params.get("content", [])
    blocks = blocks if isinstance(blocks, list) else [blocks]
    return any(isinstance(block, dict) and (block.get("type") in {"image", "audio", "resource_link"}
        or block.get("type") == "resource" and isinstance(block.get("resource"), dict) and "blob" in block["resource"]) for block in blocks)


@contextmanager
def encode_app_content(brain, approved, *, cancelled=lambda: False):
    """Bounded accumulation + durable current-ACK audit, before any mutation."""
    data = approved.task_data
    policy = brain.resource_policy
    dimensions = int(brain.config.vsa_dim)
    total = correction = None
    text_parts, contributions = [], 0
    spool = AppObservationAuditWriter(brain, approved)
    modes = [(module, module.training) for module in brain.modalities.modules()]
    activity_lease = None

    def check():
        if cancelled():
            raise InterruptedError("ordinary App sensory encoding was cancelled before learning")
        pressure = getattr(policy, "check_active_pressure", None)
        if callable(pressure):
            pressure("ordinary App sensory encoding")

    def add(vector, block_index, window_index, weight=1.0):
        nonlocal total, correction, contributions
        check()
        vector = vector.detach().cpu().float().reshape(-1)
        if vector.numel() != dimensions or not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 1e-8:
            raise ValueError("App input did not produce a finite same-brain neural signal")
        position = brain.memory.space.symbol("app-block:%d:window:%d" % (block_index, window_index))
        bound = brain.memory.space.permute(brain.memory.space.bind(vector, position), block_index + window_index + 1)
        increment = bound * float(weight) - correction
        updated = total + increment
        correction = (updated - total) - increment
        total = updated
        contributions += 1

    try:
        activity_lease = policy.reserve_ram(dimensions * 4 * 12 + 131072, "ordered App sensory activity and audit window")
        total = torch.zeros(dimensions, dtype=torch.float32)
        correction = torch.zeros_like(total)
        pager = getattr(brain, "working_attention_pager", None)
        with torch.no_grad(), native_parameter_read_only(), pager.activate(cancelled) if pager is not None else nullcontext():
            brain.modalities.eval()
            for index, (kind, text, media) in enumerate(task_blocks(data)):
                check()
                row = {"format": "omni-native-app-observation-block", "version": 1, "sourceBrainId": brain.brain_id,
                    "recordSha256": approved.record_sha256, "dataSha256": approved.data_sha256, "blockIndex": index,
                    "kind": kind, "sourceEncoding": "decoded-inline-media" if media is not None else "approved-data-json" if kind in {"structured", "resource-reference"} else "utf8-text",
                    "wholeInputTraversed": True, "nativeSensory": media is not None, "fetched": False}
                if media is None:
                    payload = text.encode("utf-8")
                    if kind in {"structured", "resource-reference"}:
                        row.update(sourceSha256=approved.data_sha256, sourceBytes=approved.data_bytes)
                    else:
                        row.update(sourceSha256=hashlib.sha256(payload).hexdigest(), sourceBytes=len(payload))
                    text_parts.append(text)
                    if text.strip():
                        add(brain.memory.vector_for_text(text), index, 0)
                else:
                    block = media_block(media, 0, index)
                    receipt, encoded = {}, 0
                    windows = sensory_windows(brain, block, cancelled, receipt)
                    try:
                        for tensor, elapsed in windows:
                            check()
                            channels = max(1, int(brain.config.modality_channels))
                            units = int(brain.config.image_size) ** 2 if kind == "image" else int(brain.config.audio_samples)
                            scratch = (units * (channels * 8 + 8) + int(brain.config.idea_dim) * 16 + dimensions * 8) * 4 + 131072
                            with policy.reserve_ram(scratch, "same-brain App sensory forward activity"):
                                embedding = brain.modalities.perception_embedding(kind, tensor)
                                if embedding.shape != (1, brain.config.idea_dim) or not bool(torch.isfinite(embedding).all()):
                                    raise ValueError("App perception returned invalid same-brain shared activity")
                                lifted = brain._sensory_substrate_vector(embedding, kind, receipt["sourceSha256"])
                                add(lifted, index, encoded, max(1e-6, float(elapsed)))
                            encoded += 1
                    finally:
                        windows.close()
                    if receipt.get("wholeInputTraversed") is not True or encoded < 1 or encoded != receipt["decodedWindows"]:
                        raise ValueError("App sensory input did not fully traverse and encode its source windows")
                    receipt["nativeSensory"] = True
                    row.update(sourceSha256=receipt["sourceSha256"], sourceBytes=receipt["sourceBytes"], media=receipt)
                spool.add(row)
            check()
            if contributions < 1 or not bool(torch.isfinite(total).all()) or float(total.norm()) <= 1e-8:
                raise ValueError("App content produced no finite integrated neural experience")
            content = spool.finalize()
            vector = F.normalize(total.reshape(1, -1), dim=-1)[0]
            text_projection = "\n".join(text_parts)
        # End the read-only guard BEFORE actual ordinary neural admission.
        for module, mode in modes:
            module.training = mode
        yield vector, text_projection, content, spool
    except WorkingAttentionCancelled as error:
        if cancelled():
            raise InterruptedError("ordinary App sensory reading stopped before neural admission") from error
        raise
    finally:
        for module, mode in modes:
            module.training = mode
        try:
            spool.close()
        finally:
            if activity_lease is not None:
                activity_lease.release()
