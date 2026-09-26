"""Measured, resource-bounded plans for scalable neural media generation.

The persisted ``image_size``, ``audio_samples`` and ``video_frames`` values
describe the native patch/chunk/window learned by a modality pack.  They are
not output ceilings.  This module deliberately has no model-defined maximum:
an explicit request is admitted exactly when its measured peak-memory and
storage demand fits the caller-provided resource envelope, while an automatic
request chooses the largest interactive plan supported by the same evidence.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


USEFUL_IMAGE_EDGE = 256
USEFUL_AUDIO_DURATION_MS = 2_000
USEFUL_VIDEO_EDGE = 128
USEFUL_VIDEO_FRAMES = 16
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_VIDEO_FPS = 8
# Bounded renderer fallback only, never a generation or model-output ceiling.
# Keep in sync with src/shared/mediaTransport.ts.
INLINE_MEDIA_BINARY_BYTES = 512 * 1024


def inline_media_data_url(mime_type: str, payload: bytes) -> Optional[str]:
    """Embed bounded media for native playback without exposing local paths."""

    if len(payload) > INLINE_MEDIA_BINARY_BYTES:
        return None
    return "data:%s;base64,%s" % (
        str(mime_type),
        base64.b64encode(payload).decode("ascii"),
    )


class MediaPlanError(ValueError):
    """Raised when a requested plan is structurally invalid."""


class MediaResourcePause(RuntimeError):
    """Raised before a unit crosses a live resource watermark."""

    def __init__(self, demand: "MediaResourceDemand"):
        super().__init__(
            "%s generation paused at the live resource watermark during %s"
            % (demand.modality, demand.stage)
        )
        self.demand = demand


@dataclass(frozen=True)
class NeuralMediaWindows:
    """Checkpoint-compatible native neural work-unit dimensions."""

    image_patch: int
    audio_chunk_samples: int
    video_window_frames: int
    channels: int

    @classmethod
    def from_config(cls, config: Any) -> "NeuralMediaWindows":
        return cls(
            image_patch=int(config.image_size),
            audio_chunk_samples=int(config.audio_samples),
            video_window_frames=int(config.video_frames),
            channels=int(config.modality_channels),
        ).validated()

    def validated(self) -> "NeuralMediaWindows":
        if self.image_patch < 4 or self.image_patch % 4:
            raise MediaPlanError("image patch must be a positive multiple of four")
        if self.audio_chunk_samples < 4 or self.audio_chunk_samples % 4:
            raise MediaPlanError("audio chunk must be a positive multiple of four")
        if self.video_window_frames < 2:
            raise MediaPlanError("video window must contain at least two frames")
        if self.channels < 1:
            raise MediaPlanError("modality channels must be positive")
        return self


@dataclass(frozen=True)
class MediaGenerationMeasurements:
    """Live post-reserve resources plus a measured native-unit benchmark."""

    available_memory_bytes: int
    available_storage_bytes: int
    image_tile_ms: float
    audio_chunk_ms: float
    video_tile_window_ms: float
    target_latency_ms: float = 30_000.0
    preview_interval_ms: float = 500.0
    # Units that the caller may keep resident together. The benchmark times
    # above must already reflect its execution strategy; the portable core
    # does not invent a parallel speedup.
    parallel_units: int = 1
    source: str = "measured-live-resource-envelope"
    measured_modality: Optional[str] = None

    @classmethod
    def for_modality(
        cls,
        modality: str,
        *,
        available_memory_bytes: int,
        available_storage_bytes: int,
        native_unit_ms: float,
        target_latency_ms: float = 30_000.0,
        preview_interval_ms: float = 500.0,
        parallel_units: int = 1,
        source: str,
    ) -> "MediaGenerationMeasurements":
        if modality not in {"image", "audio", "video"}:
            raise MediaPlanError("media modality must be image, audio, or video")
        # Only the selected field participates in its plan. Mirroring the
        # measured value into the unused fields avoids inventing evidence for
        # another modality while keeping one compact immutable record.
        return cls(
            available_memory_bytes=available_memory_bytes,
            available_storage_bytes=available_storage_bytes,
            image_tile_ms=native_unit_ms,
            audio_chunk_ms=native_unit_ms,
            video_tile_window_ms=native_unit_ms,
            target_latency_ms=target_latency_ms,
            preview_interval_ms=preview_interval_ms,
            parallel_units=parallel_units,
            source=source,
            measured_modality=modality,
        ).validated()

    def validated(self) -> "MediaGenerationMeasurements":
        if self.available_memory_bytes < 1 or self.available_storage_bytes < 1:
            raise MediaPlanError("available media resources must be positive")
        for label, value in (
            ("image tile time", self.image_tile_ms),
            ("audio chunk time", self.audio_chunk_ms),
            ("video tile-window time", self.video_tile_window_ms),
            ("target latency", self.target_latency_ms),
            ("preview interval", self.preview_interval_ms),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise MediaPlanError("%s must be finite and positive" % label)
        if self.parallel_units < 1:
            raise MediaPlanError("parallel media units must be positive")
        if not str(self.source).strip():
            raise MediaPlanError("media measurements require an evidence source")
        if self.measured_modality not in {None, "image", "audio", "video"}:
            raise MediaPlanError("measured media modality is invalid")
        return self


@dataclass(frozen=True)
class MediaOutputRequest:
    """Optional exact output request; omitted fields use measured Auto."""

    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[float] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    fps: int = DEFAULT_VIDEO_FPS

    def explicit_for(self, modality: str) -> bool:
        if modality == "image":
            return self.width is not None or self.height is not None
        if modality == "audio":
            return self.duration_ms is not None
        return (
            self.width is not None
            or self.height is not None
            or self.duration_ms is not None
        )


@dataclass(frozen=True)
class MediaOutputPlan:
    modality: str
    width: Optional[int]
    height: Optional[int]
    sample_rate: Optional[int]
    total_samples: Optional[int]
    duration_ms: Optional[float]
    fps: Optional[int]
    total_frames: Optional[int]
    spatial_overlap: int
    temporal_overlap: int
    work_units: int
    preview_every_units: int
    estimated_previews: int
    estimated_generation_ms: float
    estimated_peak_memory_bytes: int
    estimated_artifact_bytes: int
    estimated_peak_storage_bytes: int
    available_memory_bytes: int
    available_storage_bytes: int
    parallel_units: int
    explicit_request: bool
    admitted: bool
    within_latency_budget: bool
    useful_floor_met: bool
    limiting_resource: Optional[str]
    measurement_source: str
    model_defined_maximum: None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "width": self.width,
            "height": self.height,
            "sampleRate": self.sample_rate,
            "totalSamples": self.total_samples,
            "durationMs": self.duration_ms,
            "fps": self.fps,
            "totalFrames": self.total_frames,
            "spatialOverlap": self.spatial_overlap,
            "temporalOverlap": self.temporal_overlap,
            "workUnits": self.work_units,
            "previewEveryUnits": self.preview_every_units,
            "estimatedPreviews": self.estimated_previews,
            "estimatedGenerationMs": self.estimated_generation_ms,
            "estimatedPeakMemoryBytes": self.estimated_peak_memory_bytes,
            "estimatedArtifactBytes": self.estimated_artifact_bytes,
            "estimatedPeakStorageBytes": self.estimated_peak_storage_bytes,
            "availableMemoryBytes": self.available_memory_bytes,
            "availableStorageBytes": self.available_storage_bytes,
            "parallelUnits": self.parallel_units,
            "explicitRequest": self.explicit_request,
            "admitted": self.admitted,
            "withinLatencyBudget": self.within_latency_budget,
            "minimumSizeFloorMet": self.useful_floor_met,
            "semanticQualityClaimed": False,
            "limitingResource": self.limiting_resource,
            "measurementSource": self.measurement_source,
            "modelDefinedMaximum": None,
        }


@dataclass(frozen=True)
class MediaResourceDemand:
    modality: str
    stage: str
    completed_units: int
    total_units: int
    working_bytes: int
    output_bytes: int


MediaResourceWatermark = Callable[[MediaResourceDemand], Optional[bool]]


def media_resource_headroom(status: Mapping[str, Any]) -> tuple[int, int]:
    """Derive incremental media headroom from one correlated policy sample."""

    available = max(0, int(status.get("availableMemoryBytes", 0) or 0))
    ram_reserve = max(0, int(status.get("ramReserveBytes", 0) or 0))
    available_safe = max(
        0,
        int(status.get("availableSafeRamBytes", available - ram_reserve) or 0),
    )
    system_budget = max(0, int(status.get("systemRamBudgetBytes", 0) or 0))
    process_memory = max(0, int(status.get("processMemoryBytes", 0) or 0))
    process_headroom = max(0, system_budget - process_memory)
    accelerator_free = status.get("acceleratorFreeMemoryBytes")
    candidates = [value for value in (available_safe, process_headroom) if value > 0]
    if isinstance(accelerator_free, int) and accelerator_free > 0:
        candidates.append(accelerator_free)
    memory = min(candidates) if candidates else 1
    disk_free = max(0, int(status.get("diskFreeBytes", 0) or 0))
    disk_reserve = max(0, int(status.get("diskReserveBytes", 0) or 0))
    storage = max(1, disk_free - disk_reserve)
    return memory, storage


def require_media_resources(
    watermark: Optional[MediaResourceWatermark],
    demand: MediaResourceDemand,
) -> None:
    """Fail before work when a live caller reports pressure.

    A callback may raise its own richer resource-pause exception. Returning
    ``False`` uses the portable core exception above; ``None`` and ``True``
    both mean that the measured unit remains admitted.
    """

    if watermark is not None and watermark(demand) is False:
        raise MediaResourcePause(demand)


def axis_positions(length: int, window: int, overlap: int) -> tuple[int, ...]:
    if length < 1 or window < 1:
        raise MediaPlanError("media dimensions must be positive")
    if overlap < 0 or overlap >= window:
        raise MediaPlanError("media overlap must be smaller than its window")
    if length <= window:
        return (0,)
    stride = window - overlap
    positions = list(range(0, max(1, length - window + 1), stride))
    final = length - window
    if positions[-1] != final:
        positions.append(final)
    return tuple(positions)


def axis_window_count(length: int, window: int, overlap: int) -> int:
    """Count work units without materializing attacker-sized position lists."""

    if length < 1 or window < 1:
        raise MediaPlanError("media dimensions must be positive")
    if overlap < 0 or overlap >= window:
        raise MediaPlanError("media overlap must be smaller than its window")
    if length <= window:
        return 1
    stride = window - overlap
    return (length - window + stride - 1) // stride + 1


def _positive_integer(value: Optional[int], label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MediaPlanError("%s must be a positive integer" % label)
    return value


def _positive_float(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MediaPlanError("%s must be finite and positive" % label)
    return number


def _overlap(window: int) -> int:
    return max(1, min(window - 1, window // 8))


def _unit_peak_bytes(modality: str, windows: NeuralMediaWindows) -> int:
    channels = windows.channels
    scalar = 4
    if modality == "image":
        patch = windows.image_patch
        latent_tokens = max(1, (patch // 4) ** 2)
        tensors = channels * patch * patch * scalar * 32
        attention = latent_tokens * latent_tokens * scalar * 4
        return tensors + attention
    if modality == "audio":
        samples = windows.audio_chunk_samples
        latent_tokens = max(1, samples // 4)
        tensors = channels * samples * scalar * 24
        attention = latent_tokens * latent_tokens * scalar * 4
        return tensors + attention
    patch = windows.image_patch
    frames = windows.video_window_frames
    tensors = channels * frames * patch * patch * scalar * 24
    return tensors


def _shape_demand(
    modality: str,
    windows: NeuralMediaWindows,
    width: Optional[int],
    height: Optional[int],
    total_samples: Optional[int],
    total_frames: Optional[int],
    parallel_units: int,
) -> tuple[int, int, int, int, int]:
    spatial_overlap = _overlap(windows.image_patch)
    temporal_window = (
        windows.audio_chunk_samples
        if modality == "audio"
        else windows.video_window_frames
    )
    temporal_overlap = _overlap(temporal_window)
    if modality == "image":
        assert width is not None and height is not None
        work_units = axis_window_count(
            width, windows.image_patch, spatial_overlap
        ) * axis_window_count(height, windows.image_patch, spatial_overlap)
        # Accumulator + blend weights + one immutable preview snapshot.
        output_memory = 7 * width * height * 4
        artifact_bytes = 3 * width * height + height + 4_096
    elif modality == "audio":
        assert total_samples is not None
        work_units = axis_window_count(
            total_samples,
            windows.audio_chunk_samples,
            temporal_overlap,
        )
        output_memory = 3 * total_samples * 4
        artifact_bytes = total_samples * 2 + 44
    else:
        assert width is not None and height is not None and total_frames is not None
        spatial = axis_window_count(
            width, windows.image_patch, spatial_overlap
        ) * axis_window_count(height, windows.image_patch, spatial_overlap)
        temporal = axis_window_count(
            total_frames,
            windows.video_window_frames,
            temporal_overlap,
        )
        work_units = spatial * temporal
        output_memory = 7 * total_frames * width * height * 4
        # Raw RGB is a conservative reservation for either MP4 staging or APNG.
        artifact_bytes = 3 * total_frames * width * height + 64 * 1_024
    peak = output_memory + _unit_peak_bytes(modality, windows) * parallel_units
    return (
        work_units,
        peak,
        artifact_bytes,
        0 if modality == "audio" else spatial_overlap,
        0 if modality == "image" else temporal_overlap,
    )


def _unit_ms(modality: str, measured: MediaGenerationMeasurements) -> float:
    return {
        "image": measured.image_tile_ms,
        "audio": measured.audio_chunk_ms,
        "video": measured.video_tile_window_ms,
    }[modality]


def _preview_count(work_units: int, every: int) -> int:
    points = work_units // every
    if every != 1:
        points += 1  # The first completed unit is always visible.
        if work_units > 1 and work_units % every:
            points += 1  # The final unit is always visible.
    return max(1, points)


def _auto_image(
    windows: NeuralMediaWindows,
    measured: MediaGenerationMeasurements,
) -> tuple[int, int]:
    unit_budget = max(
        1,
        int(measured.target_latency_ms / measured.image_tile_ms),
    )
    high = max(1, int(math.sqrt(unit_budget)))
    low = 1
    selected = 1
    overlap = _overlap(windows.image_patch)
    while low <= high:
        scale = (low + high) // 2
        edge = windows.image_patch * scale
        units = axis_window_count(edge, windows.image_patch, overlap) ** 2
        if units <= unit_budget:
            selected = scale
            low = scale + 1
        else:
            high = scale - 1
    edge = windows.image_patch * selected
    return edge, edge


def _auto_audio(
    windows: NeuralMediaWindows,
    measured: MediaGenerationMeasurements,
) -> int:
    units = max(
        1,
        int(measured.target_latency_ms / measured.audio_chunk_ms),
    )
    overlap = _overlap(windows.audio_chunk_samples)
    return windows.audio_chunk_samples + (units - 1) * (
        windows.audio_chunk_samples - overlap
    )


def _auto_video(
    windows: NeuralMediaWindows,
    measured: MediaGenerationMeasurements,
) -> tuple[int, int, int]:
    unit_budget = max(
        1,
        int(measured.target_latency_ms / measured.video_tile_window_ms),
    )
    spatial_overlap = _overlap(windows.image_patch)
    temporal_overlap = _overlap(windows.video_window_frames)
    floor_units = (
        axis_window_count(
            USEFUL_VIDEO_EDGE,
            windows.image_patch,
            spatial_overlap,
        ) ** 2
        * axis_window_count(
            USEFUL_VIDEO_FRAMES,
            windows.video_window_frames,
            temporal_overlap,
        )
    )
    ratio = max(1e-9, unit_budget / float(max(1, floor_units))) ** (1.0 / 3.0)
    edge = max(
        windows.image_patch,
        int(USEFUL_VIDEO_EDGE * ratio // windows.image_patch) * windows.image_patch,
    )
    frames = max(
        windows.video_window_frames,
        int(round(USEFUL_VIDEO_FRAMES * ratio)),
    )
    while True:
        spatial = axis_window_count(
            edge, windows.image_patch, spatial_overlap
        ) ** 2
        temporal = axis_window_count(
            frames, windows.video_window_frames, temporal_overlap
        )
        if spatial * temporal <= unit_budget:
            return edge, edge, frames
        if edge > windows.image_patch and spatial >= temporal:
            edge -= windows.image_patch
        elif frames > windows.video_window_frames:
            frames -= 1
        else:
            return windows.image_patch, windows.image_patch, windows.video_window_frames


def plan_media_output(
    modality: str,
    windows: NeuralMediaWindows,
    measurements: MediaGenerationMeasurements,
    request: Optional[MediaOutputRequest] = None,
) -> MediaOutputPlan:
    """Resolve an exact or automatic output without a model-size ceiling."""

    if modality not in {"image", "audio", "video"}:
        raise MediaPlanError("media modality must be image, audio, or video")
    windows = windows.validated()
    measurements = measurements.validated()
    if (
        measurements.measured_modality is not None
        and measurements.measured_modality != modality
    ):
        raise MediaPlanError("media benchmark belongs to another modality")
    request = request or MediaOutputRequest()
    width = _positive_integer(request.width, "media width")
    height = _positive_integer(request.height, "media height")
    duration_ms = _positive_float(request.duration_ms, "media duration")
    sample_rate = _positive_integer(request.sample_rate, "sample rate")
    fps = _positive_integer(request.fps, "video FPS")
    assert sample_rate is not None and fps is not None
    explicit = request.explicit_for(modality)

    total_samples: Optional[int] = None
    total_frames: Optional[int] = None
    if modality == "image":
        if width is None and height is None:
            width, height = _auto_image(windows, measurements)
        elif width is None:
            width = height
        elif height is None:
            height = width
        sample_rate = None
        fps = None
        duration_ms = None
    elif modality == "audio":
        if duration_ms is None:
            total_samples = _auto_audio(windows, measurements)
            duration_ms = total_samples / float(sample_rate) * 1_000.0
        else:
            total_samples = max(1, int(math.ceil(duration_ms * sample_rate / 1_000.0)))
        width = height = None
        fps = None
    else:
        if width is None and height is None and duration_ms is None:
            width, height, total_frames = _auto_video(windows, measurements)
            duration_ms = total_frames / float(fps) * 1_000.0
        else:
            if width is None and height is None:
                width = height = USEFUL_VIDEO_EDGE
            elif width is None:
                width = height
            elif height is None:
                height = width
            if duration_ms is None:
                total_frames = USEFUL_VIDEO_FRAMES
                duration_ms = total_frames / float(fps) * 1_000.0
            else:
                total_frames = max(1, int(math.ceil(duration_ms * fps / 1_000.0)))
        sample_rate = None

    parallel = measurements.parallel_units
    while parallel > 1:
        _units, candidate_peak, _artifact, _spatial, _temporal = _shape_demand(
            modality,
            windows,
            width,
            height,
            total_samples,
            total_frames,
            parallel,
        )
        if candidate_peak <= measurements.available_memory_bytes:
            break
        parallel -= 1
    work_units, peak, artifact_bytes, spatial_overlap, temporal_overlap = _shape_demand(
        modality,
        windows,
        width,
        height,
        total_samples,
        total_frames,
        parallel,
    )
    unit_ms = _unit_ms(modality, measurements)
    preview_every = max(1, int(round(measurements.preview_interval_ms / unit_ms)))
    previews = _preview_count(work_units, preview_every)
    peak_storage_bytes = artifact_bytes * (previews + 1)
    # Auto is allowed to scale down to the native neural unit. Explicit
    # dimensions/duration are never silently rewritten: an oversized request
    # remains intact and returns ``admitted=False`` with its limiting resource.
    if not explicit:
        while (
            peak > measurements.available_memory_bytes
            or peak_storage_bytes > measurements.available_storage_bytes
        ):
            changed = False
            if modality == "image" and width is not None and height is not None:
                if width > windows.image_patch or height > windows.image_patch:
                    if width >= height and width > windows.image_patch:
                        width = max(windows.image_patch, width - windows.image_patch)
                    elif height > windows.image_patch:
                        height = max(windows.image_patch, height - windows.image_patch)
                    changed = True
            elif modality == "audio" and total_samples is not None:
                stride = windows.audio_chunk_samples - _overlap(
                    windows.audio_chunk_samples
                )
                if total_samples > windows.audio_chunk_samples:
                    total_samples = max(
                        windows.audio_chunk_samples,
                        total_samples - stride,
                    )
                    assert sample_rate is not None
                    duration_ms = total_samples / float(sample_rate) * 1_000.0
                    changed = True
            elif (
                modality == "video"
                and width is not None
                and height is not None
                and total_frames is not None
            ):
                if width > windows.image_patch or height > windows.image_patch:
                    if width >= height and width > windows.image_patch:
                        width = max(windows.image_patch, width - windows.image_patch)
                    elif height > windows.image_patch:
                        height = max(windows.image_patch, height - windows.image_patch)
                    changed = True
                elif total_frames > windows.video_window_frames:
                    total_frames -= 1
                    assert fps is not None
                    duration_ms = total_frames / float(fps) * 1_000.0
                    changed = True
            if not changed:
                break
            work_units, peak, artifact_bytes, spatial_overlap, temporal_overlap = (
                _shape_demand(
                    modality,
                    windows,
                    width,
                    height,
                    total_samples,
                    total_frames,
                    parallel,
                )
            )
            previews = _preview_count(work_units, preview_every)
            peak_storage_bytes = artifact_bytes * (previews + 1)
    generation_ms = work_units * unit_ms
    memory_ok = peak <= measurements.available_memory_bytes
    storage_ok = peak_storage_bytes <= measurements.available_storage_bytes
    admitted = memory_ok and storage_ok
    limiting = None if admitted else ("memory" if not memory_ok else "storage")
    useful = (
        width is not None
        and height is not None
        and (
            (modality == "image" and width >= USEFUL_IMAGE_EDGE and height >= USEFUL_IMAGE_EDGE)
            or (
                modality == "video"
                and width >= USEFUL_VIDEO_EDGE
                and height >= USEFUL_VIDEO_EDGE
                and (total_frames or 0) >= USEFUL_VIDEO_FRAMES
            )
        )
    ) or (
        modality == "audio"
        and duration_ms is not None
        and duration_ms >= USEFUL_AUDIO_DURATION_MS
    )
    return MediaOutputPlan(
        modality=modality,
        width=width,
        height=height,
        sample_rate=sample_rate,
        total_samples=total_samples,
        duration_ms=duration_ms,
        fps=fps,
        total_frames=total_frames,
        spatial_overlap=spatial_overlap,
        temporal_overlap=temporal_overlap,
        work_units=work_units,
        preview_every_units=preview_every,
        estimated_previews=previews,
        estimated_generation_ms=generation_ms,
        estimated_peak_memory_bytes=peak,
        estimated_artifact_bytes=artifact_bytes,
        estimated_peak_storage_bytes=peak_storage_bytes,
        available_memory_bytes=measurements.available_memory_bytes,
        available_storage_bytes=measurements.available_storage_bytes,
        parallel_units=parallel,
        explicit_request=explicit,
        admitted=admitted,
        within_latency_budget=generation_ms <= measurements.target_latency_ms,
        useful_floor_met=useful,
        limiting_resource=limiting,
        measurement_source=measurements.source,
    )
