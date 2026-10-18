"""The existing native tile/chunk decoders with disk-backed overlap activity."""
from __future__ import annotations

import math
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import torch
from torch.nn import functional as F

from .media_planning import axis_window_count, require_media_resources


def axis_starts(length, window, overlap):
    count = axis_window_count(length, window, overlap)
    for index in range(count): yield min(index * (window-overlap), max(0, length-window))


def _nearest_start(length, window, overlap):
    count = axis_window_count(length, window, overlap)
    center, stride = max(0.0, (length-window) / 2.0), window-overlap
    middle = max(0, min(count-1, int(center // stride)))
    return min((min(index * stride, max(0, length-window)) for index in {middle, min(count-1, middle+1)}),
               key=lambda value: ((value-center)**2, value))


def spatial_starts(canvas, height, width, patch, overlap):
    """Exact center-first order with bounded SQLite metadata, not N² RAM.

    Publish the first native tile before building the remaining index. The
    primary-key tree provides the original (distance, top, left) ordering and
    never requires an unowned OS temporary sort or a giant coordinate list.
    """
    count = axis_window_count(height, patch, overlap) * axis_window_count(width, patch, overlap)
    policy = canvas.policy; lease = handle = connection = path = identity = None
    try:
        canvas.output.check()
        estimate = count * 128 + 131072
        placed = policy.allocate_spill_file(estimate, "native media spatial-order index", ".activity", cancelled=canvas.output.check)
        if placed is not None:
            lease, path = placed, Path(placed.path); descriptor = placed.open(os.O_RDWR)
        else:
            lease = policy.reserve_spill(estimate, "native media spatial-order index")
            path = canvas.output.path.parent / (uuid.uuid4().hex + ".activity")
            lease.bind_path(path)
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        opened = os.fstat(handle.fileno()); identity = (opened.st_dev, opened.st_ino)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-128"); connection.execute("PRAGMA mmap_size=0")
        connection.execute("CREATE TABLE positions(distance REAL,top INTEGER,left INTEGER,PRIMARY KEY(distance,top,left)) WITHOUT ROWID")
        connection.commit(); lease.commit(path=path, promised_bytes=estimate, retain_open_backing=True)
        first = (_nearest_start(height, patch, overlap), _nearest_start(width, patch, overlap))
        yield first
        if count == 1: return
        center_y, center_x = max(0.0, (height-patch)/2.0), max(0.0, (width-patch)/2.0)
        with policy.reserve_ram(512 * 1024, "bounded native media spatial-order page"):
            batch = []
            for top in axis_starts(height, patch, overlap):
                for left in axis_starts(width, patch, overlap):
                    if (top, left) == first: continue
                    canvas.output.check()
                    batch.append(((top-center_y)**2 + (left-center_x)**2, top, left))
                    if len(batch) == 256:
                        connection.executemany("INSERT INTO positions VALUES(?,?,?)", batch); connection.commit()
                        batch.clear(); lease.refresh_physical()
            if batch: connection.executemany("INSERT INTO positions VALUES(?,?,?)", batch); connection.commit()
            lease.refresh_physical()
            for top, left in connection.execute("SELECT top,left FROM positions ORDER BY distance,top,left"):
                canvas.output.check(); yield top, left
    finally:
        if connection is not None: connection.close()
        if handle is not None: handle.close()
        if path is not None:
            try:
                value = path.lstat()
                if (value.st_dev, value.st_ino) == identity and not path.is_symlink(): path.unlink()
            except FileNotFoundError: pass
        if lease is not None: lease.backing_closed()


def generate_file_canvas(hub, plan, idea, seed, canvas, preview_callback, cancel_check, resource_watermark):
    from .modalities import _blend_weight, _check_cancelled, _derived_seed, _positioned_idea, _should_publish
    from .media_randomness import media_generator
    modality, total, completed = plan.modality, plan.work_units, 0
    def tick():
        _check_cancelled(cancel_check)
        require_media_resources(resource_watermark, hub._unit_demand(plan, modality + "-canvas-unit", completed))
    def emit(view, detail):
        if preview_callback is not None and _should_publish(completed, total, plan.preview_every_units):
            preview_callback(completed / float(total), view, {"completedUnits": completed, "totalUnits": total,
                "actualDecoderOutput": True, "semanticQualityClaimed": False, "activityStorage": "file", **detail})
    if modality == "audio":
        window, carry = int(hub.audio.samples), None
        weight = _blend_weight(window, plan.temporal_overlap, device=idea.device, dtype=idea.dtype)
        for index, start in enumerate(axis_starts(plan.total_samples, window, plan.temporal_overlap)):
            tick(); positioned = _positioned_idea(idea, start / float(max(1, plan.total_samples-window)))
            if carry is not None: positioned = F.normalize(.82 * positioned + .18 * carry, dim=-1)
            chunk = hub.audio.generate(positioned, media_generator(idea.device, _derived_seed(seed, start)), cancel_check=cancel_check)
            carry = hub.audio.encode(chunk.unsqueeze(1)).detach()
            length = min(window, plan.total_samples-start)
            canvas.add((slice(0, idea.shape[0]), slice(start, start+length)), chunk[:, :length], weight[:length])
            completed += 1
            stable = plan.total_samples if completed == total else min(plan.total_samples, start + window-plan.temporal_overlap)
            emit(canvas.view(end=stable), {"stage": "codec-waveform", "scaledStage": "recurrent-overlap-audio-chunks",
                "completedChunks": completed, "totalChunks": total, "sampleCount": stable, "stablePrefixSamples": stable,
                "totalSamples": plan.total_samples, "sampleRate": plan.sample_rate, "durationMs": stable / float(plan.sample_rate) * 1000,
                "recurrentCarryUsed": index > 0})
    else:
        patch = int(hub.image.image_size if modality == "image" else hub.video.image_size)
        spatial_weight = _blend_weight(patch, plan.spatial_overlap, device=idea.device, dtype=idea.dtype)
        spatial_weight = spatial_weight[:, None] * spatial_weight[None, :]
        temporal_window = 1 if modality == "image" else int(hub.video.frames)
        times = (0,) if modality == "image" else axis_starts(plan.total_frames, temporal_window, plan.temporal_overlap)
        temporal_weight = None if modality == "image" else _blend_weight(temporal_window, plan.temporal_overlap, device=idea.device, dtype=idea.dtype)
        carry = None
        for temporal_index, frame in enumerate(times):
            embedding_sum, embedding_count = None, 0
            with closing(spatial_starts(canvas, plan.height, plan.width, patch, plan.spatial_overlap)) as positions:
                for top, left in positions:
                    tick(); coordinates = (top / float(max(1, plan.height-patch)), left / float(max(1, plan.width-patch)))
                    if modality == "video": coordinates = (frame / float(max(1, plan.total_frames-temporal_window)), *coordinates)
                    positioned = _positioned_idea(idea, *coordinates)
                    if carry is not None: positioned = F.normalize(.84 * positioned + .16 * carry, dim=-1)
                    generator = media_generator(idea.device, _derived_seed(seed, *((frame,) if modality == "video" else ()), top, left))
                    unit = getattr(hub, modality).generate(positioned, generator, cancel_check=cancel_check)
                    height, width = min(patch, plan.height-top), min(patch, plan.width-left)
                    region = (slice(0, idea.shape[0]), slice(0, 3))
                    weight = spatial_weight[:height, :width]
                    if modality == "video":
                        frames = min(temporal_window, plan.total_frames-frame)
                        region += (slice(frame, frame+frames),)
                        weight = temporal_weight[:frames, None, None] * weight[None]
                        contribution = unit[:, :, :frames, :height, :width]
                        embedding = hub.video.encode(unit).detach()
                        embedding_sum = embedding if embedding_sum is None else embedding_sum + embedding
                        embedding_count += 1
                    else: contribution = unit[:, :, :height, :width]
                    region += (slice(top, top+height), slice(left, left+width))
                    canvas.add(region, contribution, weight); completed += 1
                    if preview_callback is not None and _should_publish(completed, total, plan.preview_every_units):
                        covered = canvas.covered_fraction()
                        if modality == "image":
                            emit(canvas.view(), {"stage": "diffusion-vq-decode", "scaledStage": "position-conditioned-image-tiles",
                                "completedTiles": completed, "totalTiles": total, "width": plan.width, "height": plan.height,
                                "coveredFraction": covered, "partialCoverage": completed < total, "spatialResolutionReduced": False})
                        else:
                            visible = min(plan.total_frames, frame+frames)
                            stable = plan.total_frames if completed == total else min(plan.total_frames, frame+temporal_window-plan.temporal_overlap)
                            emit(canvas.view(end=visible), {"stage": "temporal-frame-timeline", "scaledStage": "rolling-video-tile-windows",
                                "completedTileWindows": completed, "totalTileWindows": total, "completedTemporalWindows": temporal_index,
                                "totalTemporalWindows": axis_window_count(plan.total_frames, temporal_window, plan.temporal_overlap),
                                "coveredFraction": covered, "partialCoverage": completed < total, "frameCount": visible,
                                "totalFrames": plan.total_frames, "fps": plan.fps, "durationMs": visible/float(plan.fps)*1000,
                                "width": plan.width, "height": plan.height, "stableFramePrefix": stable, "recurrentCarryUsed": temporal_index > 0})
                    _check_cancelled(cancel_check)
            if embedding_sum is not None: carry = F.normalize(embedding_sum / float(embedding_count), dim=-1)
    if completed != total: raise RuntimeError("file-backed media did not traverse its complete planned units")
    return canvas.view()
