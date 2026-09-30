#!/usr/bin/env python3
"""OmniCortex JSON-RPC 2.0 stdio worker.

Stdout is protocol-only.  Diagnostics and tracebacks go to stderr so Electron
can safely parse one JSON response/notification per line.
"""

import base64
import binascii
import copy
import hashlib
import json
import os
import re
import platform
import signal
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError as FutureCancelledError
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import torch
from worker_transport import ProtocolLineReader, serve_worker_stdio

from omni_core import AdaptiveBrain, OmniConfig, __version__
from omni_core.brain import ChatGenerationCancelled, is_allocator_oom_error
from omni_core.brain_lease import BrainLeaseBusy, BrainOwnerLease
from omni_core.modalities import ModalityGenerationCancelled, ModalityHub
from omni_core.media_planning import MediaResourcePause, inline_media_data_url
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.evolution import NeuralEvolutionManager
from omni_core.offload import (
    NeuralStateResourcePause,
    copy_mutable_state_snapshot,
)
from omni_core.persistence import (
    EventLog,
    atomic_write_json,
    copy_substrate_snapshot,
    read_json,
)
from omni_core.substrate_inspection import query_persisted_substrate
from omni_core.text_spool import DatasetResourcePause, require_parser_resources
from omni_core.isolated_module_snapshot import IsolatedModuleSnapshot
from omni_core.slow_state_snapshot import admit_snapshot_metadata
from video_runtime import (
    VideoRuntimeConfigurationCancelled,
)
from codec_gateway import CodecOwner, CodecRuntimeGateway, CodecRuntimeCancelled


PROTOCOL_VERSION = 1
INLINE_GENERATION_TTL_SECONDS = 5 * 60
PREVIEW_CACHE_DIRECTORY = ".preview-cache"
BACKGROUND_IDLE_RETRY_SECONDS = 15 * 60
BACKGROUND_IDLE_MAX_SUBSTRATE_ENTITIES = 50_000
BACKGROUND_IDLE_MAX_SUBSTRATE_SHARDS = 128
BACKGROUND_IDLE_MAX_DENSE_CHECKPOINT_BYTES = 64 * 1024 * 1024
BACKGROUND_IDLE_MAX_METADATA_BYTES = 16 * 1024 * 1024
_STDOUT_LOCK = threading.Lock()

# ``AdaptiveBrain.create`` is idempotent: when a checkpoint already exists it
# loads that persisted config instead of applying the supplied config.  The
# public Build boundary must therefore authenticate both sides of that handoff
# or an old/custom snake-case checkpoint could replace the versioned profile.
_BUILD_CONFIG_IDENTITY_FIELDS = (
    "seed",
    "vocab_size",
    "max_seq_len",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "idea_dim",
    "vsa_dim",
    "router_neurons",
    "hardware_tier",
    "origin_kind",
    "ternary_weights",
    "spiking_dynamics",
    "stdp_plasticity",
    "liquid_dynamics",
    "liquid_mode",
    "liquid_steps",
    "vector_symbolic_memory",
    "working_memory_slots",
    "image_size",
    "audio_samples",
    "video_frames",
    "modality_channels",
    "vision_enabled",
    "image_enabled",
    "audio_enabled",
    "video_enabled",
)


def _preview_extension(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/apng": ".apng",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
    }.get(str(mime_type).lower(), ".media")


def _write_content_addressed_preview(
    directory: Path,
    mime_type: str,
    payload: bytes,
) -> Tuple[str, Path]:
    """Synchronously persist one immutable preview before publishing it.

    The synchronous write is intentional transport backpressure: the decoder
    cannot enqueue another revision while this one is being committed. The
    renderer receives only a main-process lease URL, never this filesystem
    path or the potentially large JSON/base64 payload.
    """

    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / (digest + _preview_extension(mime_type))
    if artifact.is_file():
        if artifact.stat().st_size != len(payload):
            raise RuntimeError("content-addressed preview collision")
        return digest, artifact.resolve()
    temporary = directory / (".%s.%s.tmp" % (digest, uuid.uuid4().hex))
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(artifact))
    finally:
        temporary.unlink(missing_ok=True)
    return digest, artifact.resolve()


def _preview_status(details: Dict[str, Any]) -> str:
    stage = str(details.get("stage", ""))
    completed = max(0, int(details.get("completedUnits", 0)))
    total = max(completed, int(details.get("totalUnits", 0)))
    if stage == "diffusion-vq-decode":
        return "Diffusion/VQ decoder revision %d/%d" % (completed, total)
    if stage == "codec-waveform":
        samples = max(0, int(details.get("sampleCount", 0)))
        total_samples = max(samples, int(details.get("totalSamples", 0)))
        return "Neural codec waveform %d/%d samples" % (
            samples,
            total_samples,
        )
    if stage == "temporal-frame-timeline":
        frames = max(0, int(details.get("frameCount", 0)))
        total_frames = max(frames, int(details.get("totalFrames", 0)))
        return "Temporal decoder timeline %d/%d frames" % (
            frames,
            total_frames,
        )
    return "Decoding the current neural latent"


class RpcFault(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


class DeferredEventLog:
    """Collect background events for main-thread SQLite commit.

    sqlite3 connections are thread-affine by default. Inline imagination uses
    an isolated neural snapshot and records its audit payload here; the worker
    that owns the real brain commits the event only when the corresponding
    typed tool job claims the generated artifact.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[Tuple[str, Dict[str, Any], Optional[str]]] = []

    def append(
        self,
        kind: str,
        payload: Dict[str, Any],
        job_id: Optional[str] = None,
    ) -> str:
        with self._lock:
            self._events.append(
                (str(kind), copy.deepcopy(payload), str(job_id) if job_id else None)
            )
        return uuid.uuid4().hex

    def take(self) -> List[Tuple[str, Dict[str, Any], Optional[str]]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


class IsolatedModalityDecoder(ModalityHub):
    """Private decoder copies for one output and an optional linked medium."""

    def __init__(
        self,
        modality: str,
        module: torch.nn.Module,
        companion_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ):
        # Deliberately skip ModalityHub.__init__: inline generation must copy
        # only the selected trained decoder (and optional audio companion),
        # while inheriting its parameter-free scaled composition routines.
        torch.nn.Module.__init__(self)
        self.modality = modality
        selected_modules = {modality: module, **(companion_modules or {})}
        self._available_modalities = frozenset(selected_modules)
        for name, selected in selected_modules.items():
            setattr(self, name, selected)
            selected.eval()

    @torch.no_grad()
    def generate(
        self,
        modality: str,
        idea: torch.Tensor,
        seed: int = 0,
        preview_callback: Optional[Callable[[float, torch.Tensor], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        maximum_previews: Optional[int] = None,
    ) -> torch.Tensor:
        if modality not in self._available_modalities:
            raise ValueError("isolated modality snapshot does not match request")
        module = getattr(self, modality)
        generator = torch.Generator(device=idea.device)
        generator.manual_seed(int(seed))
        return module.generate(
            idea,
            generator,
            preview_callback=preview_callback,
            cancel_check=cancel_check,
            maximum_previews=maximum_previews,
        )


@dataclass
class ChatSteeringState:
    brain_id: str
    stream_id: str
    requested: threading.Event = field(default_factory=threading.Event)
    successor_id: str = ""
    request_id: str = ""
    claimed: bool = False


@dataclass
class InlineGeneration:
    brain_id: str
    action_id: str
    stream_id: str
    signature: str
    staging_root: Path
    events: DeferredEventLog
    created_at: float = field(default_factory=time.monotonic)
    first_preview: threading.Event = field(default_factory=threading.Event)
    preview_emit_lock: threading.Lock = field(default_factory=threading.Lock)
    future: Optional[Future] = None
    job_id: str = ""
    preview_revision: int = 0
    latest_preview: Optional[Dict[str, Any]] = None
    cancelled: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    cancellation_notified: bool = False
    publication_started: bool = False
    execution_device: str = "unknown"
    authoritative_accelerator_isolated: bool = False
    snapshot_state: Optional[Any] = None


@dataclass
class LiveObservationSessionState:
    session_id: str
    brain_id: str
    modalities: Tuple[str, ...]
    permission: Dict[str, Any]
    retention: str
    tool_schemas: List[Dict[str, Any]]
    capabilities: Dict[str, bool]
    created_at: str
    max_packet_bytes: int
    packets_accepted: int = 0
    bytes_accepted: int = 0
    last_sequence: int = -1
    last_timestamp_ms: float = -1.0


class Worker:
    def __init__(self):
        self.worker_role = str(
            os.environ.get("OMNI_WORKER_ROLE", "neural")
        ).strip().lower()
        if self.worker_role not in {"neural", "inspection"}:
            raise RuntimeError("OMNI_WORKER_ROLE must be neural or inspection")
        self.brains: Dict[str, AdaptiveBrain] = {}
        self._brain_leases: Dict[str, BrainOwnerLease] = {}
        self._brain_lease_guard = threading.RLock()
        self._ever_loaded_brains: set[str] = set()
        self.cancelled_jobs = set()
        self._cooperative_cancel = threading.Event()
        self._active_request_lock = threading.Lock()
        self._active_request: Optional[Tuple[str, str]] = None
        self._artifact_request_owner: Optional[CodecOwner] = None
        self._codec_gateway = CodecRuntimeGateway(self._notify_codec_runtime)
        self._steering_lock = threading.RLock()
        self._chat_steering: Dict[Tuple[str, str], ChatSteeringState] = {}
        self.running = True
        self._inline_lock = threading.RLock()
        self._inline_generations: Dict[Tuple[str, str], InlineGeneration] = {}
        self._inline_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="omni-inline-imagination",
        )
        self._inline_executor_closed = False
        self._observation_lock = threading.RLock()
        self._observation_sessions: Dict[str, LiveObservationSessionState] = {}
        self.methods: Dict[str, Callable[[Dict[str, Any], Optional[str]], Any]] = {
            "health": self.health,
            "hardware_projection_profile": self.hardware_projection_profile,
            "configure_video_runtime": self.configure_video_runtime,
            "create": self.create,
            "load": self.load,
            "reload": self.reload,
            "unload": self.unload,
            "restore_snapshot": self.restore_snapshot,
            "update_config": self.update_config,
            "preview_overlay": self.preview_overlay,
            "merge_overlay": self.merge_overlay,
            "install_modality_pack": self.install_modality_pack,
            "list": self.list_brains,
            "state": self.state,
            "export_state": self.state,
            "query_substrate": self.query_substrate,
            "query_concept_id_view": self.query_concept_id_view,
            "query_cortex": self.query_cortex,
            "cortex_activity": self.cortex_activity,
            "workspace": self.workspace,
            "fresh_attention": self.fresh_attention,
            "feedback": self.feedback,
            "idle_cycle": self.idle_cycle,
            "chat": self.chat,
            "consolidate_chat_learning": self.consolidate_chat_learning,
            "learn_tool_route_outcome": self.learn_tool_route_outcome,
            "chat_receipt": self.chat_receipt,
            "conversation_page": self.conversation_page,
            "train": self.train,
            "ingest": self.ingest,
            "modality_capabilities": self.modality_capabilities,
            "generate_modality": self.generate_modality,
            "generate_neural_speech": self.generate_neural_speech,
            "start_observation": self.start_observation,
            "observe_packet": self.observe_packet,
            "stop_observation": self.stop_observation,
            "cancel_observation": self.cancel_observation,
            "resolve_observation_control": self.resolve_observation_control,
            "evolution.propose": self.evolution_propose,
            "evolution.evaluate": self.evolution_evaluate,
            "evolution.list": self.evolution_list,
            "evolution.promote": self.evolution_promote,
            "evolution.reject": self.evolution_reject,
            "evolution.rollback": self.evolution_rollback,
            # Aliases for transports that reserve dotted method names.
            "evolution_propose": self.evolution_propose,
            "evolution_evaluate": self.evolution_evaluate,
            "evolution_list": self.evolution_list,
            "evolution_promote": self.evolution_promote,
            "evolution_reject": self.evolution_reject,
            "evolution_rollback": self.evolution_rollback,
            "export_ternary": self.export_ternary,
            "checkpoint": self.checkpoint,
            "snapshot": self.snapshot,
            "trace": self.trace,
            "events": self.events,
            "cancel": self.cancel,
            "shutdown": self.shutdown,
        }
        if self.worker_role == "inspection":
            self.methods = {
                "health": self.health,
                "query_substrate": self.query_substrate,
                "query_concept_id_view": self.query_concept_id_view,
                "query_cortex": self.query_cortex,
                "cancel": self.cancel,
                "shutdown": self.shutdown,
            }

    @staticmethod
    def _default_root() -> Path:
        configured = os.environ.get("OMNI_HOME")
        if configured:
            return Path(configured).expanduser().resolve()
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local).resolve() / "OmniAGI" / "brains"
        return Path.home() / ".omni-agi" / "brains"

    @staticmethod
    def _send(message: Dict[str, Any]) -> None:
        serialized = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = (serialized + "\n").encode("utf-8")
        # Chat tokens and background media previews can be emitted by separate
        # threads. Keep each protocol line atomic so Electron never receives
        # interleaved JSON fragments.  A PyInstaller console-less Windows
        # process can expose a cp1252 TextIOWrapper even though its pipe is the
        # UTF-8 JSON-RPC transport.  Write protocol bytes directly whenever the
        # stream exposes its binary buffer; text-only test/embedding streams
        # retain the exact same UTF-8 string semantics through the fallback.
        with _STDOUT_LOCK:
            binary = getattr(sys.stdout, "buffer", None)
            if binary is not None:
                binary.write(payload)
                binary.flush()
            else:
                sys.stdout.write(payload.decode("utf-8"))
                sys.stdout.flush()

    def notify(
        self,
        event_type: str,
        brain_id: str = "",
        job_id: str = "",
        stream_id: str = "",
        sequence: Optional[int] = None,
        action_id: str = "",
        progress: Optional[float] = None,
        message: str = "",
        data: Any = None,
    ) -> None:
        params: Dict[str, Any] = {"type": event_type}
        if brain_id:
            params["brainId"] = brain_id
        if job_id:
            params["jobId"] = job_id
        if stream_id:
            params["streamId"] = stream_id
        if sequence is not None:
            params["sequence"] = max(0, int(sequence))
        if action_id:
            params["actionId"] = action_id
        if progress is not None:
            params["progress"] = max(0.0, min(float(progress), 1.0))
        if message:
            params["message"] = message
        if data is not None:
            params["data"] = data
        self._send({"jsonrpc": "2.0", "method": "event", "params": params})

    def _notify_codec_runtime(self, event_type, owner, data):
        self.notify(event_type, brain_id=owner.brain_id, job_id=owner.job_id,
            stream_id=owner.stream_id, action_id=owner.action_id, data=data)

    def request_cooperative_cancel(self) -> bool:
        with self._active_request_lock:
            active = self._active_request
        if active is None or active[0] not in {
            "load",
            "chat",
            "consolidate_chat_learning",
            "configure_video_runtime",
            "generate_neural_speech",
            "generate_modality",
        }:
            return False
        self._cooperative_cancel.set()
        return True

    def _acknowledge_inline_cancellation(self, record: InlineGeneration) -> None:
        with self._inline_lock:
            if not record.cancelled or not record.finished.is_set() or record.cancellation_notified:
                return
            self._remove_inline_root(record)
            if record.staging_root.exists():
                return
            record.cancellation_notified = True
        self.notify("inline-imagination-cancelled", brain_id=record.brain_id,
                    stream_id=record.stream_id, action_id=record.action_id,
                    data={"acknowledged": True, "cleanupCompleted": True})

    def dispatch_control(self, request: Any) -> Dict[str, Any]:
        if self.worker_role != "neural" or not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or \
                request.get("method") not in {"cancel_inline_generation", "steer_chat", "resolve_codec_runtime", "cancel_artifact_request"} or \
                not isinstance(request.get("id"), (str, int)) or not isinstance(request.get("params"), dict):
            raise RpcFault(-32600, "invalid flags-only worker control request")
        params = request["params"]
        if request["method"] == "cancel_artifact_request":
            with self._active_request_lock:
                owner = getattr(self, "_artifact_request_owner", None)
                if owner is None or params != owner.fields() or self._active_request is None or \
                        self._active_request[1] != owner.request_id or self._active_request[0] not in {"generate_modality", "generate_neural_speech"}:
                    raise RpcFault(-32602, "artifact cancel does not own the active neural request")
                self._cooperative_cancel.set()
            return {"jsonrpc": "2.0", "id": request["id"], "result": {"requested": True, "requestId": owner.request_id}}
        if request["method"] == "resolve_codec_runtime":
            try:
                result = self._codec_gateway.resolve(params)
            except ValueError as error:
                raise RpcFault(-32602, str(error)) from error
            return {"jsonrpc": "2.0", "id": request["id"], "result": result}
        brain_id = self._brain_id(params)
        stream_id = params.get("streamId")
        if request["method"] == "steer_chat":
            successor_id = params.get("successorTurnId")
            if not isinstance(stream_id, str) or not isinstance(successor_id, str) or \
                    not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", stream_id) or \
                    not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", successor_id) or successor_id == stream_id:
                raise RpcFault(-32602, "invalid warm steering ownership")
            with self._steering_lock:
                session = self._chat_steering.get((brain_id, stream_id))
                if session is None:
                    raise RpcFault(-32602, "warm steering does not own the requested live chat")
                session.successor_id = successor_id
                session.requested.set()
            return {"jsonrpc": "2.0", "id": request["id"], "result": {
                "requested": True, "warm": True, "successorTurnId": successor_id}}
        action_id = self._valid_inline_action_id(params.get("neuralActionId"))
        if not action_id or not isinstance(stream_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", stream_id):
            raise RpcFault(-32602, "invalid inline artifact ownership")
        with self._inline_lock:
            record = self._inline_generations.get((brain_id, action_id))
            if record is None or record.stream_id != stream_id:
                raise RpcFault(-32602, "inline artifact is not owned by the requested chat turn")
            if record.publication_started:
                raise RpcFault(-32602, "artifact publication has already started; no inline cancellation was admitted")
            record.cancelled = True
            record.first_preview.set()
            if record.future is not None and record.future.cancel():
                record.finished.set()
        self._acknowledge_inline_cancellation(record)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {
            "requested": True, "acknowledged": record.cancellation_notified}}

    def reserve_chat_steering(self, request: Dict[str, Any]) -> None:
        if self.worker_role != "neural" or request.get("method") != "chat":
            return
        params = request.get("params")
        if request.get("jsonrpc") != "2.0" or not isinstance(params, dict):
            raise RpcFault(-32600, "invalid chat control reservation")
        brain_id = self._brain_id(params)
        request_id = str(request.get("id", ""))
        stream_id = str(params.get("streamId", "")).strip() or request_id
        if not request_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", stream_id):
            raise RpcFault(-32602, "invalid chat steering reservation")
        key = (brain_id, stream_id)
        with self._steering_lock:
            if key in self._chat_steering:
                raise RpcFault(-32602, "chat steering ownership is already reserved")
            self._chat_steering[key] = ChatSteeringState(brain_id, stream_id, request_id=request_id)

    @staticmethod
    def _inline_request(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        modality = str(value.get("modality", "")).strip().lower()
        if modality not in {"image", "audio", "video"}:
            return None
        prompt = value.get("prompt", "")
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            return None
        prompt = prompt[:1_000_000]
        raw_concepts = value.get("conceptIds", [])
        if raw_concepts is None:
            raw_concepts = []
        if not isinstance(raw_concepts, (list, tuple)) or not all(
            isinstance(item, str) for item in raw_concepts
        ):
            return None
        raw_settings = value.get("settings", {})
        if raw_settings is None:
            raw_settings = {}
        if not isinstance(raw_settings, dict):
            return None
        concept_view = value.get("conceptIdView")
        if concept_view is not None:
            from omni_core.concept_id_views import validate_descriptor
            try:
                concept_view = validate_descriptor(concept_view, brain_id=str(concept_view.get("brainId", "")),
                                                   turn_id=value.get("sourceTurnId", ""))
            except (ValueError, TypeError, AttributeError):
                return None
        raw_settings = copy.deepcopy(raw_settings)
        raw_settings.setdefault("outputMode", "auto")
        input_path = value.get("inputPath", "")
        if input_path is None:
            input_path = ""
        if not isinstance(input_path, str):
            return None
        seed = value.get("seed")
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or abs(seed) > 9_007_199_254_740_991
        ):
            return None
        return {
            "modality": modality,
            "prompt": prompt,
            "conceptIds": list(raw_concepts),
            **({"conceptIdView": concept_view, "sourceTurnId": concept_view["turnId"]} if concept_view is not None else {}),
            "inputPath": input_path,
            "settings": raw_settings,
            "seed": seed,
        }

    @classmethod
    def _inline_signature(cls, value: Dict[str, Any]) -> str:
        normalized = cls._inline_request(value)
        if normalized is None:
            return ""
        try:
            return json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _valid_inline_action_id(value: Any) -> str:
        action_id = str(value or "").strip().lower()
        if len(action_id) != 32:
            return ""
        if any(character not in "0123456789abcdef" for character in action_id):
            return ""
        return action_id

    @staticmethod
    def _clear_inline_staging(engine_path: Path) -> None:
        staging = Path(engine_path) / ".inline-imagination"
        if staging.is_symlink() or staging.is_file():
            staging.unlink(missing_ok=True)
        elif staging.is_dir():
            shutil.rmtree(staging)

    @staticmethod
    def _clear_preview_cache(engine_path: Path) -> None:
        cache = Path(engine_path) / PREVIEW_CACHE_DIRECTORY
        if cache.is_symlink() or cache.is_file():
            cache.unlink(missing_ok=True)
        elif cache.is_dir():
            shutil.rmtree(cache)

    @staticmethod
    def _remove_inline_root(record: InlineGeneration) -> None:
        if record.snapshot_state is not None:
            record.snapshot_state.close()
            record.snapshot_state = None
        root = record.staging_root
        try:
            if root.is_symlink() or root.is_file():
                root.unlink(missing_ok=True)
            elif root.is_dir():
                shutil.rmtree(root)
            if root.parent.name == ".inline-imagination":
                root.parent.rmdir()
        except OSError:
            # This is a disposable, app-owned staging cache. A later worker
            # start retries exact-directory cleanup before loading the brain.
            pass

    def _cleanup_expired_inline_generations(self) -> None:
        cutoff = time.monotonic() - INLINE_GENERATION_TTL_SECONDS
        expired: List[InlineGeneration] = []
        with self._inline_lock:
            for key, record in list(self._inline_generations.items()):
                if record.created_at > cutoff:
                    continue
                record.cancelled = True
                if record.future is not None:
                    record.future.cancel()
                expired.append(record)
                self._inline_generations.pop(key, None)
        for record in expired:
            if record.future is None or record.future.done():
                self._remove_inline_root(record)

    def _shutdown_inline_generations(self) -> None:
        if self._inline_executor_closed:
            return
        with self._inline_lock:
            records = list(self._inline_generations.values())
            for record in records:
                record.cancelled = True
                if record.future is not None:
                    record.future.cancel()
        self._inline_executor.shutdown(wait=True, cancel_futures=True)
        self._inline_executor_closed = True
        for record in records:
            self._remove_inline_root(record)
        with self._inline_lock:
            self._inline_generations.clear()

    def _discard_inline_generations(self, brain_id: str) -> None:
        records: List[InlineGeneration] = []
        with self._inline_lock:
            for key, record in list(self._inline_generations.items()):
                if record.brain_id != brain_id:
                    continue
                record.cancelled = True
                if record.future is not None:
                    record.future.cancel()
                records.append(record)
                self._inline_generations.pop(key, None)
        for record in records:
            if record.future is not None:
                try:
                    record.future.result()
                except Exception:
                    pass
            self._remove_inline_root(record)

    def _start_inline_generation(
        self,
        brain: AdaptiveBrain,
        action_id: str,
        stream_id: str,
        action: Dict[str, Any],
        emit_preview: Callable[[InlineGeneration, Dict[str, Any]], None],
        emit_started: Optional[Callable[[InlineGeneration], None]] = None,
    ) -> Optional[InlineGeneration]:
        if self._inline_executor_closed:
            return None
        action_id = self._valid_inline_action_id(action_id)
        arguments = action.get("arguments")
        if (
            not action_id
            or action.get("kind") != "imagine"
            or action.get("toolId") != "modality.imagine"
            or action.get("action") != "generate"
            or not isinstance(arguments, dict)
        ):
            return None
        request = self._inline_request(arguments)
        signature = self._inline_signature(arguments)
        if request is None or not signature:
            return None
        key = (brain.brain_id, action_id)
        with self._inline_lock:
            existing = self._inline_generations.get(key)
            if existing is not None:
                return existing

        # Capture the idea and a private copy of the modality parameters on the
        # chat thread. PyTorch's MPSGraph/cache lifetime is not safe across the
        # authoritative chat thread and an inline decoder thread, even when
        # the modules are distinct Python objects. Move every tensor the
        # background decoder can reach to CPU before submitting its future.
        # CUDA and CPU also use isolated CPU state: no original-device
        # allocation may happen before the shared resource admission.
        staging_root = (
            brain.engine_path / ".inline-imagination" / action_id
        ).resolve()
        staging_parent = (brain.engine_path / ".inline-imagination").resolve()
        record = InlineGeneration(brain_id=brain.brain_id, action_id=action_id,
                                  stream_id=stream_id, signature=signature,
                                  staging_root=staging_root, events=DeferredEventLog())
        gateway = getattr(self, "_codec_gateway", None)
        parent_owner = gateway.current_owner() if gateway is not None else None
        codec_owner = CodecOwner(parent_owner.request_id, brain.brain_id, "", stream_id, action_id) if parent_owner else None
        with self._inline_lock:
            self._inline_generations[key] = record
        def finish() -> None:
            record.finished.set()
            self._acknowledge_inline_cancellation(record)
            if codec_owner:
                self.notify("inline-imagination-finished", brain_id=brain.brain_id,
                    stream_id=stream_id, action_id=action_id, data={"requestId": codec_owner.request_id})
        try:
            if emit_started is not None:
                emit_started(record)
            staging_root.relative_to(staging_parent)
            staging_root.mkdir(parents=True, exist_ok=False)
            with torch.no_grad():
                if record.cancelled:
                    raise ModalityGenerationCancelled("inline imagination was cancelled")
                source_device = torch.device(brain.device)
                isolate_mps = source_device.type != "cpu"
                inline_device = torch.device("cpu")
                snapshot_policy = copy.copy(brain.resource_policy)
                snapshot_policy.include_accelerator_memory = False
                state_minimum = int(brain.liquid_state.numel() * brain.liquid_state.element_size() * 2) + int(brain.config.vsa_dim) * 16 + 131072
                status = snapshot_policy.status(estimated_ram_bytes=state_minimum)
                if status.get("memoryPressure"):
                    raise NeuralStateResourcePause("inline idea and liquid state require an admitted CPU minimum", {**status,
                        "recoverable": True, "paused": True, "minimumResidentBytes": state_minimum,
                        "stage": "inline-neural-cue-snapshot", "neuralWorkerRestartRequired": False})
                concept_ids = request["conceptIds"]
                if request.get("conceptIdView") is not None:
                    from omni_core.concept_id_views import IdView
                    concept_ids = IdView(brain.engine_path, request["conceptIdView"], brain_id=brain.brain_id,
                                         turn_id=request["sourceTurnId"])
                idea = brain._modality_idea(
                    request["prompt"], concept_ids
                ).detach().to(inline_device).clone()
                idea_evidence = brain._modality_idea_evidence(
                    request["prompt"], concept_ids
                )

                selected_names = [request["modality"]]
                if request["modality"] == "video" and self._trained_modality_capabilities(brain)["audioGeneration"]:
                    selected_names.append("audio")
                record.snapshot_state = IsolatedModuleSnapshot(
                    {name: getattr(brain.modalities, name) for name in selected_names},
                    directory=staging_root / "decoder-state", policy=snapshot_policy,
                    cancelled=lambda: record.cancelled)
                modality_snapshot = IsolatedModalityDecoder(
                    request["modality"],
                    record.snapshot_state.roots[request["modality"]],
                    companion_modules={name: record.snapshot_state.roots[name] for name in selected_names[1:]},
                )
                admit_snapshot_metadata(snapshot_policy, {
                    "config": brain.config, "counters": brain.counters,
                    "modality_training": brain.modality_training,
                    "installed_modality_packs": brain.installed_modality_packs,
                    "idea_evidence": idea_evidence,
                }, "inline snapshot control metadata")
                deferred_events = DeferredEventLog()
                snapshot = copy.copy(brain)
                snapshot.config = copy.deepcopy(brain.config)
                snapshot.modalities = modality_snapshot
                snapshot.events = deferred_events
                snapshot.engine_path = staging_root
                snapshot.counters = dict(brain.counters)
                snapshot.modality_training = dict(brain.modality_training)
                snapshot.installed_modality_packs = copy.deepcopy(brain.installed_modality_packs)
                snapshot.device = inline_device
                snapshot.device_backend = "cpu"
                snapshot.liquid_state = brain.liquid_state.detach().to(inline_device).clone()
                snapshot.resource_policy = snapshot_policy
                snapshot._modality_idea = lambda _prompt="", _concept_ids=None: idea.detach().clone()
                snapshot._modality_idea_evidence = lambda _prompt="", _concept_ids=None: copy.deepcopy(idea_evidence)
                record.events = deferred_events
                record.execution_device = str(inline_device)
                record.authoritative_accelerator_isolated = isolate_mps
        except Exception as error:
            self._remove_inline_root(record)
            if record.cancelled:
                finish()
                return record
            record.future = Future()
            record.future.set_exception(error)
            finish()
            return record # typed failed/paused artifact, never silently drop or regenerate it

        def preview(
            generation_progress: float,
            mime_type: str,
            payload: bytes,
            details: Dict[str, Any],
        ) -> None:
            digest, artifact_path = _write_content_addressed_preview(
                record.staging_root / "previews",
                mime_type,
                payload,
            )
            bounded_progress = 0.2 + 0.75 * max(
                0.0, min(float(generation_progress), 1.0)
            )
            status_label = _preview_status(details)
            preview_value: Dict[str, Any] = {
                **copy.deepcopy(details),
                "schemaVersion": 1,
                "producer": "same-brain-decoder",
                "payloadSha256": digest,
                "artifactPath": str(artifact_path),
                "progress": bounded_progress,
                "statusLabel": status_label,
                "mimeType": str(mime_type),
                "executionDevice": record.execution_device,
                "authoritativeAcceleratorIsolated": (
                    record.authoritative_accelerator_isolated
                ),
            }
            embedded_media = inline_media_data_url(mime_type, payload)
            if embedded_media is not None:
                preview_value["dataUrl"] = embedded_media
            # Serialize job binding/replay with new revisions. This preserves
            # monotonic previews when the queued tool request claims a decode
            # at the exact moment a new frame/sample is emitted.
            with record.preview_emit_lock:
                with self._inline_lock:
                    if record.cancelled:
                        return
                    revision = record.preview_revision
                    record.preview_revision += 1
                    preview_value["revision"] = revision
                    record.latest_preview = copy.deepcopy(preview_value)
                emit_preview(record, preview_value)
                record.first_preview.set()

        def generate() -> Dict[str, Any]:
            try:
                with gateway.scope(codec_owner, lambda: record.cancelled) if codec_owner else nullcontext():
                    result = snapshot.generate_modality(
                        modality=request["modality"],
                        prompt=request["prompt"],
                        concept_ids=request["conceptIds"],
                        input_path=request["inputPath"],
                        settings=request["settings"],
                        seed=request["seed"],
                        preview_callback=preview,
                        cancel_check=lambda: record.cancelled,
                    )
                with self._inline_lock:
                    cancelled = record.cancelled
                if cancelled:
                    self._remove_inline_root(record)
                    raise RuntimeError("inline imagination was cancelled")
                result["inlineExecutionDevice"] = record.execution_device
                result["authoritativeAcceleratorIsolated"] = (
                    record.authoritative_accelerator_isolated
                )
                return result
            except Exception:
                self._remove_inline_root(record)
                raise
            finally:
                if record.snapshot_state is not None:
                    record.snapshot_state.close()
                    record.snapshot_state = None
                finish()

        with self._inline_lock:
            if record.cancelled:
                self._remove_inline_root(record)
                finish()
                return record
        try:
            with self._inline_lock:
                record.future = self._inline_executor.submit(generate)
        except Exception as error:
            self._remove_inline_root(record)
            record.future = Future()
            record.future.set_exception(error)
            finish()
            return record
        return record

    def _claim_inline_generation(
        self,
        brain: AdaptiveBrain,
        params: Dict[str, Any],
        job_id: str,
    ) -> Optional[Dict[str, Any]]:
        action_id = self._valid_inline_action_id(params.get("neuralActionId"))
        signature = self._inline_signature(params)
        if not action_id or not signature:
            return None
        key = (brain.brain_id, action_id)
        with self._inline_lock:
            record = self._inline_generations.get(key)
        if record is None:
            return None
        with record.preview_emit_lock:
            with self._inline_lock:
                if record.signature != signature:
                    return None
                if record.cancelled:
                    raise RpcFault(-32800, "this inline artifact was cancelled; it must not be regenerated")
                record.job_id = job_id
                latest_preview = copy.deepcopy(record.latest_preview)
                future = record.future
            if latest_preview is not None:
                self.notify(
                    "modality-preview",
                    brain_id=brain.brain_id,
                    job_id=job_id,
                    action_id=action_id,
                    sequence=int(latest_preview.get("revision", 0)),
                    progress=float(latest_preview.get("progress", 0.0)),
                    message=str(latest_preview.get("statusLabel", "")),
                    data={"preview": latest_preview},
                )
        if future is None:
            return None

        try:
            try:
                result = dict(future.result())
            except FutureCancelledError as error:
                raise ModalityGenerationCancelled("owned inline artifact was cancelled before decoder execution") from error
            with self._inline_lock:
                if record.cancelled:
                    raise RpcFault(-32800, "this inline artifact was cancelled")
                # Cancellation and artifact promotion have one atomic ownership
                # boundary. A control never deletes files during publication.
                record.publication_started = True
            if job_id and job_id in self.cancelled_jobs:
                raise RpcFault(-32800, "job was cancelled")
            raw_path = result.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise RuntimeError("inline imagination produced no artifact path")
            source = Path(raw_path).resolve()
            artifact_root = (record.staging_root / "artifacts").resolve()
            try:
                source.relative_to(artifact_root)
            except ValueError as error:
                raise RuntimeError(
                    "inline imagination artifact escaped its staging area"
                ) from error
            if not source.is_file():
                raise RuntimeError("inline imagination artifact is missing")
            destination_root = (brain.engine_path / "artifacts").resolve()
            destination_root.mkdir(parents=True, exist_ok=True)
            destination = destination_root / source.name
            while destination.exists():
                destination = destination_root / (
                    uuid.uuid4().hex + source.suffix.lower()
                )
            os.replace(str(source), str(destination))
            result["path"] = str(destination)
            result["neuralActionId"] = action_id
            result["generatedDuringChat"] = True

            for kind, payload, deferred_job_id in record.events.take():
                committed = dict(payload)
                if committed.get("outputPath") == raw_path:
                    committed["outputPath"] = str(destination)
                committed["neuralActionId"] = action_id
                committed["generatedDuringChat"] = True
                brain.events.append(
                    kind,
                    committed,
                    job_id=job_id or deferred_job_id,
                )
            return result
        finally:
            with self._inline_lock:
                self._inline_generations.pop(key, None)
            self._remove_inline_root(record)

    @staticmethod
    def _brain_id(params: Dict[str, Any], required: bool = True) -> str:
        value = params.get("brainId") or params.get("brain_id")
        if value is None and required:
            raise RpcFault(-32602, "params.brainId is required")
        return str(value or "")

    def _storage(self, params: Dict[str, Any], brain_id: str) -> Path:
        raw = params.get("storagePath") or params.get("storage_path")
        if raw:
            return Path(str(raw)).expanduser().resolve()
        return self._default_root() / brain_id

    def _owner_lease(
        self, brain_id: str, storage: Path
    ) -> Tuple[BrainOwnerLease, bool]:
        if self.worker_role != "neural":
            raise RpcFault(-32601, "inspection workers cannot own neural state")
        with self._brain_lease_guard:
            current = self._brain_leases.get(brain_id)
            if current is not None:
                if current.storage_path != storage.resolve():
                    raise RpcFault(-32602, "brainId is already leased at another storagePath")
                current.assert_exclusive()
                return current, False
            try:
                lease = BrainOwnerLease(storage).acquire()
            except BrainLeaseBusy as error:
                raise RpcFault(-32009, str(error)) from error
            self._brain_leases[brain_id] = lease
            return lease, True

    def _release_owner_lease(self, brain_id: str) -> None:
        with self._brain_lease_guard:
            lease = self._brain_leases.pop(brain_id, None)
        if lease is not None:
            lease.release()

    def _release_all_owner_leases(self) -> None:
        with self._brain_lease_guard:
            brain_ids = list(self._brain_leases)
        for brain_id in brain_ids:
            self._release_owner_lease(brain_id)

    def _prune_verified_abandoned_live_caches(
        self, brain_id: str, brain: AdaptiveBrain, lease: BrainOwnerLease
    ) -> None:
        """Clean only exact owned derived paths after verified reconstruction.

        This is called only for this worker's first load of the brain, before
        publishing the new object in ``self.brains``. The exclusive lease
        excludes other neural workers; the local checks exclude old objects
        and inline jobs. A missing proof leaves cache bytes untouched.
        """

        rebuilt = getattr(brain, "_verified_paged_rebuild", None)
        if rebuilt is None:
            return
        from omni_core.live_paging_migration import (
            prune_abandoned_live_paging_cache,
        )

        cache_parent = brain.engine_path / "state" / "live-substrate-cache"
        if not cache_parent.is_dir() or cache_parent.is_symlink():
            return
        def candidate_paths():
            yield cache_parent
            for child in cache_parent.iterdir():
                if (
                    child.is_dir() and not child.is_symlink()
                    and len(child.name) == 32
                    and all(character in "0123456789abcdef" for character in child.name)
                ):
                    yield child
        outcomes: List[Dict[str, Any]] = []
        for candidate in candidate_paths():
            for target in ("staging", "live"):
                def no_consumers(path: Path, owned_child: Path = candidate) -> bool:
                    try:
                        lease.assert_exclusive()
                    except (OSError, RuntimeError):
                        return False
                    if (
                        self.brains.get(brain_id) is not None
                        or brain_id in self._ever_loaded_brains
                        or path.parent.resolve() != owned_child.resolve()
                    ):
                        return False
                    with self._inline_lock:
                        if any(
                            item.brain_id == brain_id
                            for item in self._inline_generations.values()
                        ):
                            return False
                    current_cache = getattr(
                        brain, "_live_paging_cache_directory", None
                    )
                    if current_cache is not None and path.is_relative_to(
                        Path(current_cache).resolve()
                    ):
                        return False
                    return True

                try:
                    result = prune_abandoned_live_paging_cache(
                        candidate,
                        brain.engine_path,
                        verified_rebuild=rebuilt,
                        target=target,
                        no_consumers=no_consumers,
                    )
                    if result.get("removed"):
                        outcomes.append({"target": target, "removed": True})
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    # A derived cache is never recovery authority. A failed
                    # exact proof must not block loading the verified brain or
                    # become permission to delete an unrecognized directory.
                    outcomes.append({
                        "target": target, "removed": False,
                        "reason": type(error).__name__,
                    })
        brain._derived_cache_prune_outcomes = outcomes

    def _get(self, params: Dict[str, Any]) -> AdaptiveBrain:
        brain_id = self._brain_id(params)
        storage = self._storage(params, brain_id)
        existing = self.brains.get(brain_id)
        if existing is not None:
            if existing.storage_path != storage:
                raise RpcFault(
                    -32602,
                    "brainId is already loaded from a different storagePath",
                )
            self._owner_lease(brain_id, storage)
            return existing
        # A hard process interruption can leave only disposable inline media
        # staging behind. It is never an authoritative brain artifact and is
        # removed before the persistent checkpoint is opened again.
        # Runtime requests are deliberately load-only.  The desktop repository
        # document becomes visible before neural initialization has
        # finished verification/materialization, so idle cognition (or any
        # other eager runtime caller) can legitimately arrive in that window.
        # Treating its public config as permission to create used to let that
        # request win the race and persist a blank/default cortex before the
        # authoritative builder request arrived.  Only the explicit ``create``
        # RPC below may initialize neural state.
        if not (storage / "engine" / "brain.json").is_file():
            raise RpcFault(
                -32004,
                "brain is not initialized; an explicit create request must "
                "complete before load or runtime operations",
            )
        lease, newly_acquired = self._owner_lease(brain_id, storage)
        try:
            self._clear_inline_staging(storage / "engine")
            self._clear_preview_cache(storage / "engine")
            brain = AdaptiveBrain.load(storage, expected_brain_id=brain_id)
            if newly_acquired and brain_id not in self._ever_loaded_brains:
                self._prune_verified_abandoned_live_caches(
                    brain_id, brain, lease
                )
            lease.assert_exclusive()
        except BaseException:
            if "brain" in locals():
                try:
                    brain.close()
                except Exception:
                    pass
            if newly_acquired:
                self._release_owner_lease(brain_id)
            raise
        self.brains[brain_id] = brain
        self._ever_loaded_brains.add(brain_id)
        return brain

    def _builder_config(
        self, params: Dict[str, Any], raw_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(raw_config)
        if params.get("hardwareTier"):
            merged["hardwareTier"] = str(params["hardwareTier"])
        if params.get("origin"):
            merged["origin_kind"] = str(params["origin"])
        modalities = params.get("modalities")
        if isinstance(modalities, list):
            selected = {str(value) for value in modalities}
            merged["vision_enabled"] = "vision" in selected
            merged["image_enabled"] = "image" in selected
            merged["audio_enabled"] = "audio" in selected
            merged["video_enabled"] = "video" in selected
        tier = str(merged.get("hardwareTier", "personal"))
        if "device" not in merged and tier in {"gpu", "workstation"}:
            if torch.cuda.is_available():
                merged["device"] = "cuda"
            elif self._mps_available():
                merged["device"] = "mps"
            elif self._directml_available():
                merged["device"] = "directml"
        return merged

    @staticmethod
    def _build_config_identity(config: OmniConfig) -> Dict[str, Any]:
        return {
            field_name: getattr(config, field_name)
            for field_name in _BUILD_CONFIG_IDENTITY_FIELDS
        }

    @classmethod
    def _build_config_mismatches(
        cls,
        actual: OmniConfig,
        expected: OmniConfig,
    ) -> List[str]:
        actual_identity = cls._build_config_identity(actual)
        expected_identity = cls._build_config_identity(expected)
        return [
            field_name
            for field_name in _BUILD_CONFIG_IDENTITY_FIELDS
            if actual_identity[field_name] != expected_identity[field_name]
        ]

    @classmethod
    def _preflight_existing_build_config(
        cls,
        storage: Path,
        expected: OmniConfig,
    ) -> None:
        """Reject a valid but non-matching checkpoint before create finalizes it.

        Corrupt/incomplete metadata is left to ``AdaptiveBrain.load`` so its
        existing recovery and validation remain authoritative.  This bounded
        read exists only to keep a valid custom profile from reaching the
        idempotent finalize path.
        """

        metadata_path = storage / "engine" / "brain.json"
        if not metadata_path.is_file():
            return
        try:
            metadata_bytes = int(metadata_path.stat().st_size)
        except OSError:
            return
        if metadata_bytes > BACKGROUND_IDLE_MAX_METADATA_BYTES:
            raise RpcFault(
                -32602,
                "existing brain metadata exceeds the bounded Build preflight",
            )
        try:
            payload = metadata_path.read_bytes()
        except OSError:
            return
        if len(payload) > BACKGROUND_IDLE_MAX_METADATA_BYTES:
            raise RpcFault(
                -32602,
                "existing brain metadata exceeds the bounded Build preflight",
            )
        try:
            metadata = json.loads(payload.decode("utf-8"))
            raw_config = (
                metadata.get("config") if isinstance(metadata, dict) else None
            )
            if not isinstance(raw_config, dict):
                return
            actual = OmniConfig.from_dict(raw_config)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return
        mismatches = cls._build_config_mismatches(actual, expected)
        if mismatches:
            raise RpcFault(
                -32602,
                "existing brain architecture does not match the requested "
                "versioned hardwareTier profile: %s" % ", ".join(mismatches),
            )

    @staticmethod
    def _mps_available() -> bool:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None:
            return False
        try:
            return bool(backend.is_built() and backend.is_available())
        except (AttributeError, RuntimeError):
            return False

    @staticmethod
    def _directml_available() -> bool:
        try:
            import torch_directml

            torch_directml.device()
            return True
        except (ImportError, RuntimeError):
            return False

    def health(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del params, request_id
        if self.worker_role == "inspection":
            return {
                "ready": True,
                "worker": "python-inspection",
                "engineVersion": __version__,
                "protocolVersion": PROTOCOL_VERSION,
                "detail": "Read-only persisted substrate inspection worker is ready.",
                "pythonVersion": platform.python_version(),
                "platform": platform.platform(),
                "operatingSystem": sys.platform,
                "capabilities": {
                    "persistedSubstrateInspection": True,
                    "neuralMutation": False,
                },
                "loadedBrains": 0,
            }
        cuda = torch.cuda.is_available()
        mps = self._mps_available()
        return {
            "ready": True,
            "worker": "python",
            "engineVersion": __version__,
            "protocolVersion": PROTOCOL_VERSION,
            "detail": "Python OmniCortex neural worker is ready.",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "platform": platform.platform(),
            "operatingSystem": sys.platform,
            "capabilities": {
                "cpu": True,
                "cuda": cuda,
                "cudaDevices": torch.cuda.device_count() if cuda else 0,
                "mps": mps,
                "directml": self._directml_available(),
                "distributed": bool(torch.distributed.is_available()),
                "safetensors": True,
                "sqliteEventLog": True,
                "modalities": ["vision", "image", "audio", "video"],
            },
            "loadedBrains": len(self.brains),
        }

    def hardware_projection_profile(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        """Explicit, main-owned hardware measurement; never loads a brain."""
        del request_id
        if self.worker_role != "neural":
            raise RpcFault(-32601, "Projection profiling is not an inspection operation")
        if set(params) - {"device", "ramBudgetBytes", "hardwareTier"}:
            raise RpcFault(-32602, "Projection profiling accepts hardware fields only")
        device = params.get("device", "cpu")
        budget = params.get("ramBudgetBytes")
        tier = params.get("hardwareTier", "personal")
        if not isinstance(device, str) or re.fullmatch(r"cpu|mps|directml|cuda(?::[0-9]+)?", device) is None:
            raise RpcFault(-32602, "Projection profiling device is invalid")
        if type(budget) is not int or not 1 <= budget <= (1 << 53) - 1:
            raise RpcFault(-32602, "Projection profiling requires a measured positive RAM envelope")
        if not isinstance(tier, str) or tier not in {"micro", "personal", "gpu", "workstation"}:
            raise RpcFault(-32602, "Projection profiling tier is invalid")
        with getattr(self, "_inline_lock", nullcontext()):
            if any(record.future is not None and not record.future.done()
                   for record in getattr(self, "_inline_generations", {}).values()):
                return {"available": False, "reason": "hardware-measurement-deferred-active-inline-job"}
        from omni_core.native_compute_profile import profile_native_projection_compute
        from omni_core.offload import ResourcePolicy
        policy = ResourcePolicy(self._default_root(), hardware_tier=tier)

        def reserve(byte_count: int, actual_device: str) -> None:
            status = policy.status(estimated_ram_bytes=byte_count)
            free = status.get("acceleratorFreeMemoryBytes")
            accelerator_reserve = max(256 * 1024 * 1024, int(status.get("acceleratorTotalMemoryBytes") or 0) // 10)
            if (status["memoryPressure"] or int(status.get("projectedProcessMemoryBytes") or 0) > budget
                or actual_device != "cpu" and isinstance(free, int) and free < byte_count + accelerator_reserve):
                raise NeuralStateResourcePause("Hardware projection measurement waits for its admitted memory envelope", status)
        try:
            return profile_native_projection_compute(
                device=device, reserve=reserve, cancelled=self._cooperative_cancel.is_set,
            )
        except InterruptedError as error:
            raise RpcFault(-32800, str(error), {"hardwareMeasurementCancelled": True, "safeBoundary": True}) from error
        except NeuralStateResourcePause as error:
            return {"available": False, "reason": "hardware-measurement-memory-admission", "status": error.status}

    def configure_video_runtime(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        if self.worker_role != "neural":
            raise RpcFault(-32601, "Video runtime selection is not an inspection operation")
        try:
            gateway = getattr(self, "_codec_gateway", None)
            if gateway is None:
                gateway = self._codec_gateway = CodecRuntimeGateway(self._notify_codec_runtime)
            return gateway.configure(params, cancelled=self._cooperative_cancel.is_set)
        except (VideoRuntimeConfigurationCancelled, CodecRuntimeCancelled) as error:
            raise RpcFault(-32800, str(error)) from error
        except (ValueError, OSError) as error:
            raise RpcFault(-32602, str(error)) from error

    @staticmethod
    def _native_builder_metadata(params: Dict[str, Any], raw_config: Dict[str, Any]) -> Dict[str, Any]:
        if "nativeArchitecture" in raw_config:
            from omni_core.native_architecture import validate_native_architecture
            trusted = params.get("nativeArchitecture")
            try:
                if trusted is None:
                    raise ValueError("embedded native architecture needs the trusted top-level descriptor")
                top = validate_native_architecture(trusted)
                embedded = validate_native_architecture(raw_config["nativeArchitecture"])
                if top != embedded:
                    raise ValueError("embedded and trusted native architecture descriptors differ")
            except (ValueError, TypeError, KeyError) as error:
                raise RpcFault(-32602, str(error)) from error
            raw_config = dict(raw_config)
            del raw_config["nativeArchitecture"]
        return raw_config

    def create(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        brain_id = self._brain_id(params, required=False) or uuid.uuid4().hex
        storage = self._storage(params, brain_id)
        raw_config = params.get("config") or {}
        if not isinstance(raw_config, dict):
            raise RpcFault(-32602, "params.config must be an object")
        raw_config = self._native_builder_metadata(params, raw_config)
        # A new Build selects one versioned architecture through hardwareTier.
        # Snake-case tensor-shape fields belong only to persisted checkpoints
        # and explicit research constructors.  Reject them here instead of
        # silently accepting a caller-authored model behind the Studio UI.
        private_architecture_fields = {
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "idea_dim",
            "vsa_dim",
            "router_neurons",
            "hardware_tier",
        }
        supplied_private_fields = sorted(
            private_architecture_fields.intersection(raw_config)
        )
        if supplied_private_fields:
            raise RpcFault(
                -32602,
                "new brain architecture is resolved from the versioned "
                "hardwareTier profile; private fields are not accepted: %s"
                % ", ".join(supplied_private_fields),
            )
        declared_tiers = [
            str(value)
            for value in (
                params.get("hardwareTier"),
                raw_config.get("hardwareTier"),
            )
            if value is not None
        ]
        if any(
            value not in {"micro", "personal", "gpu", "workstation"}
            for value in declared_tiers
        ):
            raise RpcFault(-32602, "new brain hardwareTier is invalid")
        if len(set(declared_tiers)) > 1:
            raise RpcFault(
                -32602,
                "new brain hardwareTier declarations do not match",
            )
        declared_modalities = params.get("modalities")
        if declared_modalities is not None and (
            not isinstance(declared_modalities, list)
            or any(
                not isinstance(value, str)
                or value not in {"vision", "image", "audio", "video"}
                for value in declared_modalities
            )
        ):
            raise RpcFault(-32602, "new brain modalities are invalid")
        # Validate the caller's declared origin before from_external applies
        # its safe new-build defaults.
        declared_origins = [
            value
            for value in (
                params.get("origin"),
                raw_config.get("origin"),
                raw_config.get("origin_kind"),
            )
            if value is not None
        ]
        if any(str(value) != "ground-up" for value in declared_origins):
            raise RpcFault(
                -32602,
                "Build requires the locally initialized OmniCortex configuration",
            )
        if any(
            key in container
            for container in (params, raw_config)
            for key in ("foundationModelId", "foundation_model_id")
        ):
            raise RpcFault(
                -32602,
                "Build does not accept a foundation model",
            )
        config = OmniConfig.from_external(
            self._builder_config(params, raw_config),
            native_architecture=params.get("nativeArchitecture"),
        )
        if config.origin_kind != "ground-up":
            raise RpcFault(
                -32602,
                "Build requires a locally initialized OmniCortex native core",
            )
        self._preflight_existing_build_config(storage, config)
        _lease, newly_acquired = self._owner_lease(brain_id, storage)
        stream_id = str(params.get("streamId", "")).strip()
        build_sequence = 0

        def build_progress(
            phase: str,
            value: float,
            message: str,
            metrics: Dict[str, Any],
        ) -> None:
            nonlocal build_sequence
            payload_metrics = dict(metrics)
            initial_checksum = getattr(build_progress, "initial_checksum", "")
            current_checksum = str(payload_metrics.pop("parameterChecksum", ""))
            if not initial_checksum and current_checksum:
                setattr(build_progress, "initial_checksum", current_checksum)
                initial_checksum = current_checksum
            payload_metrics["parameterChecksumChanged"] = bool(
                initial_checksum
                and current_checksum
                and current_checksum != initial_checksum
            )
            self.notify(
                "build-progress",
                brain_id=brain_id,
                stream_id=stream_id,
                sequence=build_sequence,
                progress=value,
                message=message,
                data={"phase": phase, "metrics": payload_metrics},
            )
            build_sequence += 1

        try:
            self._discard_inline_generations(brain_id)
            self._clear_inline_staging(storage / "engine")
            brain = AdaptiveBrain.create(
                brain_id,
                storage,
                config,
                progress=build_progress if stream_id else None,
                # Only the public worker Build boundary runs the production local
                # curriculum. Low-level test/research construction remains fast.
                initialize_ground_up=True,
            )
        except BaseException:
            if newly_acquired:
                self._release_owner_lease(brain_id)
            raise
        mismatches = self._build_config_mismatches(brain.config, config)
        if mismatches:
            try:
                brain.close()
            except Exception:
                # The identity violation remains the authoritative failure;
                # cleanup cannot make the loaded checkpoint acceptable.
                pass
            if self.brains.get(brain_id) is brain:
                self.brains.pop(brain_id, None)
            if newly_acquired:
                self._release_owner_lease(brain_id)
            raise RpcFault(
                -32602,
                "created brain architecture does not match the requested "
                "versioned hardwareTier profile: %s" % ", ".join(mismatches),
            )
        self.brains[brain_id] = brain
        self._ever_loaded_brains.add(brain_id)
        self.notify(
            "brain-created",
            brain_id=brain_id,
            progress=1.0,
            message="OmniCortex native core initialized and locally trained.",
        )
        return brain.summary()

    def load(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        if self._cooperative_cancel.is_set():
            raise RpcFault(
                -32800,
                "brain load was cancelled after reaching a safe boundary",
                {"cancelled": True, "safeBoundary": True, "warm": True},
            )
        return brain.summary()

    def reload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        storage = self._storage(params, brain_id)
        _lease, _newly_acquired = self._owner_lease(brain_id, storage)
        try:
            self._discard_inline_generations(brain_id)
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
            self._clear_inline_staging(storage / "engine")
            brain = AdaptiveBrain.load(storage, expected_brain_id=brain_id)
        except BaseException:
            if self.brains.get(brain_id) is None:
                self._release_owner_lease(brain_id)
            raise
        self.brains[brain_id] = brain
        self._ever_loaded_brains.add(brain_id)
        return brain.summary()

    def unload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        self._discard_inline_generations(brain_id)
        previous = self.brains.pop(brain_id, None)
        if previous is not None:
            previous.close()
        self._release_owner_lease(brain_id)
        return {"brainId": brain_id, "unloaded": previous is not None}

    def restore_snapshot(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain_id = self._brain_id(params)
        storage = self._storage(params, brain_id)
        raw_snapshot = params.get("snapshotPath")
        if raw_snapshot:
            snapshot = Path(str(raw_snapshot)).expanduser().resolve()
            snapshot_root = (storage / "engine" / "snapshots").resolve()
            try:
                snapshot.relative_to(snapshot_root)
            except ValueError as error:
                raise RpcFault(
                    -32602, "snapshotPath must be inside engine/snapshots"
                ) from error
            for filename in (
                "brain.json",
                "core.safetensors",
                "plasticity.safetensors",
            ):
                source = snapshot / filename
                if not source.is_file():
                    raise RpcFault(-32602, "snapshot is missing %s" % filename)
            self._owner_lease(brain_id, storage)
            engine = storage / "engine"
            try:
                # Validate and materialize every blob referenced by the
                # immutable shard graph before brain.json can commit it.
                copy_substrate_snapshot(snapshot, engine)
                copy_mutable_state_snapshot(snapshot, engine)
            except (OSError, ValueError) as error:
                raise RpcFault(
                    -32602,
                    "snapshot neural state failed validation: %s" % error,
                ) from error
            for filename in ("core.safetensors", "plasticity.safetensors"):
                source = snapshot / filename
                temporary = engine / (filename + ".restore.tmp")
                shutil.copy2(str(source), str(temporary))
                os.replace(str(temporary), str(engine / filename))
            source = snapshot / "brain.json"
            temporary = engine / "brain.json.restore.tmp"
            shutil.copy2(str(source), str(temporary))
            os.replace(str(temporary), str(engine / "brain.json"))
        restored = self.reload(params, request_id)
        brain = self.brains[brain_id]
        brain.export_packed_ternary()
        return {**restored, "packedTernary": brain.packed_ternary_manifest}

    def update_config(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        raw = params.get("config")
        if not isinstance(raw, dict):
            raise RpcFault(-32602, "params.config must be an object")
        return brain.update_config(raw)

    def merge_overlay(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        target_id = str(params.get("targetBrainId", ""))
        source_id = str(params.get("sourceBrainId", ""))
        if not target_id or not source_id:
            raise RpcFault(
                -32602, "targetBrainId and sourceBrainId are required"
            )
        target_params = {
            "brainId": target_id,
            "storagePath": params.get("targetStoragePath"),
        }
        source_params = {
            "brainId": source_id,
            "storagePath": params.get("sourceStoragePath"),
        }
        target = self._get(target_params)
        source = self._get(source_params)
        expected_digest = str(params.get("expectedPreviewDigest", ""))
        if not expected_digest:
            raise RpcFault(
                -32602, "params.expectedPreviewDigest is required"
            )
        try:
            return target.merge_overlay(source, expected_digest)
        except ValueError as error:
            raise RpcFault(-32009, str(error)) from error

    def preview_overlay(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        target_id = str(params.get("targetBrainId", ""))
        source_id = str(params.get("sourceBrainId", ""))
        if not target_id or not source_id:
            raise RpcFault(
                -32602, "targetBrainId and sourceBrainId are required"
            )
        target = self._get(
            {
                "brainId": target_id,
                "storagePath": params.get("targetStoragePath"),
            }
        )
        source = self._get(
            {
                "brainId": source_id,
                "storagePath": params.get("sourceStoragePath"),
            }
        )
        return target.preview_overlay(source)

    def install_modality_pack(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        storage = self._storage(params, brain_id)
        raw_path = params.get("packPath")
        manifest = params.get("manifest")
        if not raw_path or not isinstance(manifest, dict):
            raise RpcFault(-32602, "packPath and manifest are required")
        pack_path = Path(str(raw_path)).expanduser().resolve()
        pack_root = (storage / "packs").resolve()
        try:
            pack_path.relative_to(pack_root)
        except ValueError as error:
            raise RpcFault(
                -32602, "packPath must be inside the brain packs directory"
            ) from error
        if not pack_path.is_file() or pack_path.name != "modality.safetensors":
            raise RpcFault(-32602, "packPath is not a staged modality safetensors file")
        brain = self._get(params)
        return brain.install_modality_pack(pack_path, manifest)

    def list_brains(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del params, request_id
        return {"brains": [brain.summary() for brain in self.brains.values()]}

    def state(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return brain.state(include_events=int(params.get("eventLimit", 20)))

    @staticmethod
    def _trained_modality_capabilities(brain: AdaptiveBrain) -> Dict[str, bool]:
        installed = {
            str(modality)
            for pack in brain.installed_modality_packs
            if isinstance(pack, dict)
            for modality in pack.get("modalities", [])
        }

        def trained(modality: str) -> bool:
            return bool(
                int(brain.modality_training.get(modality, 0)) > 0
                or modality in installed
            )

        visual_input = bool(
            (brain.config.vision_enabled and trained("vision"))
            or (brain.config.image_enabled and trained("image"))
        )
        return {
            "imageNeural": visual_input,
            "audioNeural": bool(
                brain.config.audio_enabled and trained("audio")
            ),
            "videoNeural": bool(
                brain.config.video_enabled and trained("video")
            ),
            "imageGeneration": bool(
                brain.config.image_enabled and trained("image")
            ),
            "audioGeneration": bool(
                brain.config.audio_enabled and trained("audio")
            ),
            "videoGeneration": bool(
                brain.config.video_enabled and trained("video")
            ),
        }

    def modality_capabilities(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        capabilities = self._trained_modality_capabilities(brain)
        # The built-in audio decoder makes general sound. It is not trained as
        # ASR or intelligible TTS, so those capabilities must remain false.
        return {
            "brainId": brain.brain_id,
            "hardwareTier": str(brain.config.hardware_tier),
            "imagePerception": capabilities["imageNeural"],
            "audioPerception": capabilities["audioNeural"],
            "videoPerception": capabilities["videoNeural"],
            "imageGeneration": capabilities["imageGeneration"],
            "audioGeneration": capabilities["audioGeneration"],
            "videoGeneration": capabilities["videoGeneration"],
            "neuralSpeechRecognition": False,
            "neuralSpeechSynthesis": False,
            "audioRegionAvailable": bool(brain.config.audio_enabled),
            "speechPairedExamples": int(brain.modality_training.get("audio_speech_pairs", 0)),
            "speechQuality": ("unverified" if int(brain.modality_training.get("audio_speech_pairs", 0)) > 0 else "needs-speech-training"),
            "synchronizedVideoAudioGeneration": bool(
                capabilities["videoGeneration"]
                and capabilities["audioGeneration"]
            ),
            "sameBrainSubstrate": True,
            "hiddenBehavioralPrompt": False,
            "detail": (
                "Direct audio perception enters the same neural substrate; "
                "platform STT/TTS remain defaults. Own waveform output needs paired speech training; intelligibility is not verified."
            ),
        }

    def query_cortex(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        from omni_core.cortical_inspection import query_committed_cortex
        query = params.get("query", {})
        if not isinstance(query, dict):
            raise RpcFault(-32602, "params.query must be an object")
        brain_id = self._brain_id(params)
        try:
            return query_committed_cortex(self._storage(params, brain_id) / "engine", brain_id, query)
        except (ValueError, OSError, TypeError) as error:
            raise RpcFault(-32602, str(error)) from error

    def cortex_activity(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        loaded = self.brains.get(brain_id)
        if loaded is None:
            return {"available": False, "observed": False, "observation": None,
                    "reason": "brain-not-loaded-no-inspection-model-created"}
        if self._storage(params, brain_id) != loaded.storage_path:
            raise RpcFault(-32602, "cortical activity storage identity mismatch")
        query = params.get("query", {})
        if not isinstance(query, dict) or not isinstance(query.get("enabled", False), bool):
            raise RpcFault(-32602, "invalid cortical activity request")
        module = str(query.get("module", ""))
        if len(module) > 512:
            raise RpcFault(-32602, "cortical activity module is too long")
        start, count = query.get("start", 0), query.get("count", 64)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, count)):
            raise RpcFault(-32602, "invalid cortical activity viewport")
        return loaded.core_pager.observation(module, enabled=query.get("enabled", False), start=start, count=count)

    def query_concept_id_view(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        from omni_core.concept_id_views import IdView
        from omni_core.recall_views import id_page
        brain_id = self._brain_id(params)
        engine = self._storage(params, brain_id) / "engine"
        try:
            metadata = json.loads((engine / "brain.json").read_bytes())
            if metadata.get("brain_id") != brain_id:
                raise ValueError("concept ID query brain identity differs")
            descriptor = params.get("conceptIdView")
            source_owner = descriptor.get("brainId", "") if isinstance(descriptor, dict) else ""
            historical = source_owner != brain_id
            if historical and params.get("historicalInspection") is not True:
                raise ValueError("foreign concept view owner requires explicit historical read-only inspection")
            view = IdView(engine, descriptor, brain_id=source_owner if historical else brain_id,
                          turn_id=params.get("sourceTurnId", ""))
            page, coverage = id_page(view, byte_budget=65536, offset=params.get("offset", 0))
            return {"brainId": brain_id, "ids": page, **coverage,
                    "sourceBrainId": source_owner, "sourceTurnId": params.get("sourceTurnId", ""),
                    "ownership": "owned-historical-file" if historical else "current-brain-owned-file",
                    "historicalInspection": historical, "executionAuthorized": False,
                    "ancestryProvenance": "preserved-original-header; current-owned-copy; no inherited execution permission" if historical else "current-owner"}
        except (ValueError, OSError, TypeError, KeyError) as error:
            raise RpcFault(-32602, str(error)) from error

    def query_substrate(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        query = params.get("query", {})
        if not isinstance(query, dict):
            raise RpcFault(-32602, "params.query must be an object")
        try:
            loaded = self.brains.get(brain_id)
            expected_live = None
            if loaded is not None:
                if self._storage(params, brain_id) != loaded.storage_path:
                    raise RpcFault(
                        -32602,
                        "brainId is already loaded from a different storagePath",
                    )
                self._owner_lease(brain_id, loaded.storage_path)
                expected_live = {
                    "activeGeneration": str(
                        (loaded.memory.persistence_manifest or {}).get(
                            "activeGeneration", ""
                        )
                    ),
                    "stateRevision": int(loaded.memory.state_revision),
                    "counts": {
                        "neurons": len(loaded.memory.neurons),
                        "assemblies": len(loaded.memory.assemblies),
                        "synapses": len(loaded.memory.synapses),
                    },
                    "attentionOverlay": loaded.memory.attention_overlay_metadata(),
                }
            # Inspection is a generation-bound read path. Never call `_get`
            # here: materializing millions of synapses would inflate immutable
            # shards into gigabytes of Python objects and serialize chat behind
            # a read-only map request.
            return query_persisted_substrate(
                self._storage(params, brain_id) / "engine",
                brain_id,
                query,
                expected_live=expected_live,
            )
        except (OSError, TypeError, ValueError) as error:
            raise RpcFault(-32602, str(error)) from error

    def workspace(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        # The renderer's transparent Runtime Card must describe the worker's
        # measured placement, not reconstruct it from desktop configuration.
        return {
            **brain.workspace_snapshot(),
            "runtimeCard": brain.runtime_card(),
        }

    def fresh_attention(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain_id = self._brain_id(params)
        with self._inline_lock:
            inline_count = sum(
                1
                for record in self._inline_generations.values()
                if record.brain_id == brain_id
            )
        self._discard_inline_generations(brain_id)
        with self._observation_lock:
            observation_ids = [
                session_id
                for session_id, session in self._observation_sessions.items()
                if session.brain_id == brain_id
            ]
            for session_id in observation_ids:
                self._observation_sessions.pop(session_id, None)
        brain = self._get(params)
        operation_id = str(params.get("operationId") or request_id or "")
        try:
            result = brain.start_fresh_attention(operation_id)
        except Exception:
            # A pre-commit failure may have cleared transient RAM in this
            # object. Drop it so the next request reloads the last atomic disk
            # generation rather than treating partial in-memory state as live.
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
            raise
        if bool(result.get("pagedCleanupPending")):
            # brain.json already commits an empty page checkpoint. Reload will
            # transactionally remove unreachable rows before any new scratch
            # page can be admitted by this process.
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
        self.notify(
            "brain-mutated",
            brain_id=brain_id,
            progress=1.0,
            message="Fresh attention boundary committed.",
            data={
                "operationId": operation_id,
                "attentionEpoch": result["boundary"]["epoch"],
            },
        )
        return {
            **result,
            "inlineGenerationsCancelled": inline_count,
            "observationSessionsCancelled": len(observation_ids),
        }

    def feedback(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        text = params.get("text")
        direction = params.get("direction")
        if not isinstance(text, str):
            raise RpcFault(-32602, "params.text must be a string")
        if direction not in {"up", "down"}:
            raise RpcFault(-32602, "params.direction must be up or down")
        brain = self._get(params)
        try:
            return brain.feedback(
                text,
                str(direction),
                trace_id=str(params.get("traceId", "")),
                message_id=str(params.get("messageId", "")),
            )
        except Exception as error:
            # STDP, optional cortical learning and working state may have
            # changed before a checkpoint write fails. Keep only the last
            # brain.json-authoritative generation in this worker.
            brain_id = str(brain.brain_id)
            if self.brains.get(brain_id) is brain:
                self.brains.pop(brain_id, None)
                try:
                    brain.close()
                except Exception:
                    pass
                try:
                    self.brains[brain_id] = AdaptiveBrain.load(
                        brain.storage_path, expected_brain_id=brain_id
                    )
                except Exception:
                    # Fail closed: never reinsert the unacknowledged object.
                    pass
            if isinstance(error, ValueError):
                raise RpcFault(-32602, str(error)) from error
            raise

    def idle_cycle(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        try:
            schemas = AdaptiveBrain._normalize_tool_schemas(
                params.get("toolSchemas", [])
            )
            minimum_idle = _number(params.get("minimumIdleSeconds", 45.0))
            if minimum_idle < 0 or minimum_idle > 86_400:
                raise ValueError(
                    "minimumIdleSeconds must be between 0 and 86400"
                )
            deferred = self._background_idle_admission(params)
            if deferred is not None:
                return deferred
            return self._get(params).idle_cycle(
                tool_schemas=schemas,
                minimum_idle_seconds=minimum_idle,
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

    def _background_idle_admission(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Defer optional work that cannot finish inside a bounded envelope.

        This check reads only the last committed scalar/count metadata and runs
        before ``_get``. A cold multi-gigabyte brain therefore cannot occupy
        the serial worker merely because the prompt-free scheduler woke first.
        Foreground load/chat remains authoritative and sees the untouched
        checkpoint.
        """

        brain_id = self._brain_id(params)
        engine_path = self._storage(params, brain_id) / "engine"
        metadata_path = engine_path / "brain.json"
        try:
            metadata_bytes = int(metadata_path.stat().st_size)
            if metadata_bytes > BACKGROUND_IDLE_MAX_METADATA_BYTES:
                return self._deferred_idle_result(
                    brain_id,
                    metadata_bytes=metadata_bytes,
                    substrate_entities=0,
                    substrate_shards=0,
                    dense_checkpoint_bytes=0,
                )
            metadata = json.loads(metadata_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Let the authoritative load path report missing/corrupt state.
            return None
        if (
            not isinstance(metadata, dict)
            or str(metadata.get("brain_id", "")) != brain_id
        ):
            return None
        substrate = metadata.get("substrate", {})
        persistence = (
            substrate.get("persistence", {})
            if isinstance(substrate, dict)
            else {}
        )
        counts = (
            persistence.get("counts", {})
            if isinstance(persistence, dict)
            else {}
        )

        def nonnegative_integer(value: Any) -> int:
            return (
                int(value)
                if isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                else 0
            )

        substrate_entities = sum(
            nonnegative_integer(counts.get(name))
            for name in ("assemblies", "neurons", "synapses")
        )
        substrate_shards = nonnegative_integer(
            persistence.get("shardCount")
            if isinstance(persistence, dict)
            else None
        )
        dense_checkpoint_bytes = 0
        for name in ("core.safetensors", "plasticity.safetensors"):
            try:
                dense_checkpoint_bytes += int(
                    (engine_path / name).stat().st_size
                )
            except OSError:
                pass
        if (
            substrate_entities <= BACKGROUND_IDLE_MAX_SUBSTRATE_ENTITIES
            and substrate_shards <= BACKGROUND_IDLE_MAX_SUBSTRATE_SHARDS
            and dense_checkpoint_bytes
            <= BACKGROUND_IDLE_MAX_DENSE_CHECKPOINT_BYTES
        ):
            return None
        return self._deferred_idle_result(
            brain_id,
            metadata_bytes=metadata_bytes,
            substrate_entities=substrate_entities,
            substrate_shards=substrate_shards,
            dense_checkpoint_bytes=dense_checkpoint_bytes,
        )

    @staticmethod
    def _deferred_idle_result(
        brain_id: str,
        *,
        metadata_bytes: int,
        substrate_entities: int,
        substrate_shards: int,
        dense_checkpoint_bytes: int,
    ) -> Dict[str, Any]:
        return {
            "brainId": str(brain_id),
            "ran": False,
            "reason": "resource-envelope",
            "retryAfterSeconds": BACKGROUND_IDLE_RETRY_SECONDS,
            "actions": [],
            "admission": {
                "policy": "bounded-persisted-idle-v1",
                "metadataBytes": max(0, int(metadata_bytes)),
                "substrateEntities": max(0, int(substrate_entities)),
                "substrateShards": max(0, int(substrate_shards)),
                "denseCheckpointBytes": max(
                    0, int(dense_checkpoint_bytes)
                ),
                "limits": {
                    "metadataBytes": BACKGROUND_IDLE_MAX_METADATA_BYTES,
                    "substrateEntities": (
                        BACKGROUND_IDLE_MAX_SUBSTRATE_ENTITIES
                    ),
                    "substrateShards": BACKGROUND_IDLE_MAX_SUBSTRATE_SHARDS,
                    "denseCheckpointBytes": (
                        BACKGROUND_IDLE_MAX_DENSE_CHECKPOINT_BYTES
                    ),
                },
                "checkpointMutated": False,
                "brainLoadedByRequest": False,
            },
        }

    def _job(
        self,
        params: Dict[str, Any],
        request_id: Optional[str],
        kind: str,
    ):
        brain = self._get(params)
        job_id = str(params.get("jobId") or "")
        if job_id:
            brain.events.append(
                "job-start",
                {"kind": kind, "requestId": request_id},
                job_id=job_id,
            )
        self.notify(
            "job-progress",
            brain_id=brain.brain_id,
            job_id=job_id,
            progress=0.0,
            message="%s started" % kind,
        )

        def progress(
            value: float,
            message: str,
            data: Optional[Dict[str, Any]] = None,
        ) -> None:
            if job_id and job_id in self.cancelled_jobs:
                raise RpcFault(-32800, "job was cancelled")
            self.notify(
                "job-progress",
                brain_id=brain.brain_id,
                job_id=job_id,
                progress=value,
                message=message,
                data=data,
            )

        return brain, job_id, progress

    def _job_complete(
        self,
        brain: AdaptiveBrain,
        job_id: str,
        kind: str,
        result: Dict[str, Any],
    ) -> None:
        self.notify(
            "job-progress",
            brain_id=brain.brain_id,
            job_id=job_id,
            progress=1.0,
            message="%s complete" % kind,
            data={"promoted": result.get("promoted")},
        )
        if job_id:
            brain.events.append(
                "job-complete",
                {"kind": kind, "result": result},
                job_id=job_id,
            )

    def _restore_committed_brain_after_failed_chat(
        self,
        brain: AdaptiveBrain,
        *,
        prior_ledger_head: Optional[Mapping[str, Any]],
        turn_id: str,
        input_sha256: str,
    ) -> bool:
        """Never retain a failed turn's uncommitted fast state in RAM.

        The authoritative brain.json may represent either the prior turn or a
        just-committed turn whose post-save acknowledgement failed. Reloading
        that checkpoint preserves whichever state was actually committed,
        while dropping every in-memory mutation that never reached it.
        """

        brain_id = str(brain.brain_id)
        self._discard_inline_generations(brain_id)
        if self.brains.get(brain_id) is not brain:
            return False
        self.brains.pop(brain_id, None)
        storage = Path(brain.storage_path)
        try:
            brain.close()
        except Exception:
            # The stale object must stay unloaded even if its diagnostics
            # handles cannot be closed cleanly.
            pass
        try:
            committed = read_json(storage / "engine" / "brain.json")
            receipts = committed.get("completed_chat_turns", [])
            turn_committed = bool(
                isinstance(receipts, list)
                and any(
                    isinstance(receipt, Mapping)
                    and receipt.get("turnId") == turn_id
                    and receipt.get("inputSha256") == input_sha256
                    for receipt in receipts
                )
            )
            if not turn_committed and isinstance(prior_ledger_head, Mapping):
                ledger = NeuralConversationLedger(
                    storage / "engine" / "conversation.sqlite3", brain_id
                )
                try:
                    ledger.truncate_after_head(
                        int(prior_ledger_head["headSequence"]),
                        str(prior_ledger_head["headSha256"]),
                    )
                finally:
                    ledger.close()
        except Exception:
            # If the committed state or pre-turn ledger boundary cannot be
            # verified, leave the brain unloaded rather than admitting a
            # potentially orphaned turn into the next request.
            return False
        try:
            restored = AdaptiveBrain.load(
                storage, expected_brain_id=brain_id
            )
        except Exception:
            # A corrupt/unavailable checkpoint is a hard load failure, never
            # permission to continue from an unacknowledged mutable object.
            return False
        self.brains[brain_id] = restored
        return True

    def chat(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        brain_id = self._brain_id(params)
        stream_id = str(params.get("streamId", "")).strip() or str(request_id or uuid.uuid4().hex)
        session = ChatSteeringState(brain_id, stream_id)
        key = (brain_id, stream_id)
        with self._steering_lock:
            reserved = self._chat_steering.get(key)
            if reserved is not None:
                if reserved.claimed or reserved.request_id != str(request_id or ""):
                    raise RpcFault(-32602, "this chat does not own the reserved steering session")
                session = reserved
            else:
                self._chat_steering[key] = session
            session.claimed = True
        try:
            return self._chat(params, request_id, session.requested.is_set)
        finally:
            with self._steering_lock:
                self._chat_steering.pop(key, None)

    def _chat(self, params: Dict[str, Any], request_id: Optional[str],
              steer_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        brain = self._get(params)
        # The desktop's small, atomic switch is authoritative even when this
        # warm worker still holds a checkpoint from before Pause was pressed.
        # The next chat save persists this setting alongside the neural turn.
        if "onlineLearning" in params:
            if not isinstance(params["onlineLearning"], bool):
                raise RpcFault(-32602, "params.onlineLearning must be a boolean")
            brain.config.online_learning = params["onlineLearning"]
        value = params.get("input", params.get("message", params.get("text")))
        if not isinstance(value, str):
            raise RpcFault(-32602, "params.input must be a string")
        try:
            tool_schemas = AdaptiveBrain._normalize_tool_schemas(
                params.get("toolSchemas", [])
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error
        stream_id = str(params.get("streamId", "")).strip()
        if stream_id and (
            len(stream_id) > 128
            or "\x00" in stream_id
            or any(character in "\r\n" for character in stream_id)
        ):
            raise RpcFault(-32602, "params.streamId is invalid")
        sequence = 0
        sequence_lock = threading.Lock()
        inline_records: List[InlineGeneration] = []
        turn_id = stream_id or str(request_id or uuid.uuid4().hex)
        input_sha256 = hashlib.sha256(
            value.replace("\x00", "").strip().encode("utf-8")
        ).hexdigest()
        ledger_head = brain.conversation.summary()
        prior_ledger_head = (
            dict(ledger_head) if isinstance(ledger_head, Mapping) else None
        )
        imagination_grant = next(
            (
                str(schema.get("grant", "ask")).strip().lower()
                for schema in tool_schemas
                if schema.get("id") == "modality.imagine"
                and "generate" in schema.get("actions", [])
            ),
            "off",
        )

        def stream(kind: str, payload: Dict[str, Any]) -> None:
            nonlocal sequence
            if self._cooperative_cancel.is_set():
                raise ChatGenerationCancelled("chat generation was cancelled")
            if not stream_id:
                return
            action: Optional[Dict[str, Any]] = None
            action_id = ""
            with sequence_lock:
                if kind == "token":
                    self.notify(
                        "chat-token",
                        brain_id=brain.brain_id,
                        stream_id=stream_id,
                        sequence=sequence,
                        data={"delta": str(payload.get("delta", ""))},
                    )
                elif kind == "action":
                    raw_action = payload.get("action")
                    action = raw_action if isinstance(raw_action, dict) else None
                    action_id = str(payload.get("actionId", ""))
                    self.notify(
                        "chat-action",
                        brain_id=brain.brain_id,
                        stream_id=stream_id,
                        sequence=sequence,
                        action_id=action_id,
                        data={"action": action},
                    )
                elif kind == "preview":
                    preview_value = payload.get("preview")
                    if not isinstance(preview_value, dict):
                        raise RuntimeError("neural modality preview is invalid")
                    self.notify(
                        "modality-preview",
                        brain_id=brain.brain_id,
                        job_id=str(payload.get("jobId", "")),
                        stream_id=stream_id,
                        sequence=sequence,
                        action_id=str(payload.get("actionId", "")),
                        progress=float(preview_value.get("progress", 0.0)),
                        message=str(preview_value.get("statusLabel", "")),
                        data={"preview": preview_value},
                    )
                elif kind == "phase":
                    if payload != {
                        "phase": "reply-complete-learning",
                        "replyComplete": True,
                        "turnCommitted": False,
                        "learning": True,
                        "saving": True,
                    }:
                        raise RuntimeError(
                            "neural chat phase event is invalid"
                        )
                    self.notify(
                        "chat-phase",
                        brain_id=brain.brain_id,
                        stream_id=stream_id,
                        sequence=sequence,
                        data=dict(payload),
                    )
                elif kind == "inline-started":
                    self.notify("inline-imagination-started", brain_id=brain.brain_id,
                                stream_id=stream_id, sequence=sequence,
                                action_id=str(payload.get("actionId", "")), data={"requestId": str(request_id or "")})
                else:
                    raise RuntimeError("unsupported neural chat stream event")
                sequence += 1

            # The permission-bearing capability schema is authoritative. Ask
            # and Off actions must not begin work before the trusted Electron
            # permission controller approves them.
            if (
                kind == "action"
                and action is not None
                and imagination_grant in {"auto", "full"}
            ):
                record = self._start_inline_generation(
                    brain,
                    action_id,
                    stream_id,
                    action,
                    lambda current, preview: stream(
                        "preview",
                        {
                            "actionId": current.action_id,
                            "jobId": current.job_id,
                            "preview": preview,
                        },
                    ),
                    lambda current: stream("inline-started", {"actionId": current.action_id}),
                )
                if record is not None and record not in inline_records:
                    inline_records.append(record)

        try:
            result = brain.chat(
                value,
                max_new_tokens=(
                    int(params["maxNewTokens"])
                    if params.get("maxNewTokens") is not None
                    else None
                ),
                seed=(
                    int(params["seed"])
                    if params.get("seed") is not None
                    else None
                ),
                tool_schemas=tool_schemas,
                stream_callback=stream if stream_id else None,
                turn_id=turn_id,
                defer_slow_learning=True,
                cancel_check=self._cooperative_cancel.is_set,
                steer_check=steer_check,
                temporary_steering_context=params.get("temporarySteeringContext"),
            )
        except ChatGenerationCancelled as error:
            # The model raises this only at a pre-commit boundary. Preserve
            # the warm brain for Steer; restarting a large loaded model here
            # would repeat the cold-load memory spike.
            for record in inline_records:
                with self._inline_lock:
                    record.cancelled = True
                    if record.future is not None:
                        record.future.cancel()
            raise RpcFault(
                -32800,
                str(error),
                {"cancelled": True, "safeBoundary": True},
            ) from error
        except Exception:
            self._restore_committed_brain_after_failed_chat(
                brain,
                prior_ledger_head=prior_ledger_head,
                turn_id=turn_id,
                input_sha256=input_sha256,
            )
            raise

        if result.get("zeroTokenYield") is True and (result.get("steered") is True or result.get("nativeStopped") is True):
            steered = result.get("steered") is True
            raise RpcFault(-32801 if steered else -32802, "Chat stopped at a safe boundary before visible output.", {
                "brainId": brain.brain_id, "turnId": turn_id,
                "inputSha256": input_sha256, "steered": steered, "nativeStopped": not steered,
                "zeroTokenYield": True, "safeBoundary": True, "warm": True})

        if result.get("noReply") is True:
            if result.get("text") != "" or result.get("turnCommitted") is not True:
                raise RuntimeError("no-reply must be an exact committed zero-text completion")

        # A streamed Auto/Full imagination action normally publishes at least one
        # real decoder preview before the chat RPC resolves. Longer generation
        # continues concurrently and becomes the same typed tool job/artifact.
        for record in ([] if result.get("steered") is True or result.get("nativeStopped") is True else inline_records):
            while not record.first_preview.wait(timeout=0.05):
                future = record.future
                if future is None or future.done():
                    break
        self.notify(
            "brain-mutated",
            brain_id=brain.brain_id,
            progress=1.0,
            message=(
                "Turn committed to fast episodic neural state; slow replay "
                "is queued in the background."
                if result.get("trace", {}).get("slow_learning_job")
                else "Turn committed to fast episodic neural state."
            ),
            data={
                "traceId": result["trace"]["id"],
                "slowLearningJob": result.get("trace", {}).get(
                    "slow_learning_job"
                ),
            },
        )
        return result

    def consolidate_chat_learning(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain = self._get(params)
        if "onlineLearning" in params:
            if not isinstance(params["onlineLearning"], bool):
                raise RpcFault(-32602, "params.onlineLearning must be a boolean")
            brain.config.online_learning = params["onlineLearning"]
        job_id = str(params.get("jobId", "")).strip()
        self.notify(
            "job-progress",
            brain_id=brain.brain_id,
            job_id=job_id or str(request_id or ""),
            progress=0.0,
            message="Replaying retained chat activity into slow parameters.",
        )
        try:
            result = brain.consolidate_pending_chat_learning(
                job_id,
                cancel_check=self._cooperative_cancel.is_set,
            )
        except ChatGenerationCancelled as error:
            raise RpcFault(
                -32800,
                str(error),
                {"cancelled": True, "safeBoundary": True},
            ) from error
        self.notify(
            "brain-mutated",
            brain_id=brain.brain_id,
            job_id=str(result.get("jobId", job_id)),
            progress=1.0,
            message=(
                "Background cortical replay committed."
                if result.get("processed")
                else "No pending cortical replay was available."
            ),
            data=result,
        )
        return result

    def learn_tool_route_outcome(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        """Commit one host-observed typed tool outcome to the route head.

        The original user utterance and the actual successful host invocation
        are supplied as separate typed fields. Generated assistant prose is
        never used as a route target. The brain owns event-id idempotency.
        """

        del request_id
        brain = self._get(params)
        try:
            result = brain.learn_tool_route_experience(
                utterance=str(params.get("utterance", "")),
                tool_id=str(params.get("toolId", "")),
                action=str(params.get("action", "")),
                outcome=str(params.get("outcome", "")),
                source="host-tool-outcome",
                event_id=str(params.get("eventId", "")),
                arguments=(
                    params.get("arguments")
                    if isinstance(params.get("arguments"), dict)
                    else None
                ),
            )
            if bool(result.get("applied")):
                brain.save()
        except Exception:
            # A failed checkpoint must not leave a trained-but-unsaved route
            # head resident. Reload the last atomic state before a retry.
            brain_id = brain.brain_id
            storage = brain.storage_path
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
            self.brains[brain_id] = AdaptiveBrain.load(
                storage, expected_brain_id=brain_id
            )
            raise
        if bool(result.get("applied")):
            self.notify(
                "brain-mutated",
                brain_id=brain.brain_id,
                progress=1.0,
                message="A confirmed tool outcome trained the neural route head.",
                data={"eventId": str(params.get("eventId", "")), **result},
            )
        return {
            "brainId": brain.brain_id,
            "routeLearning": result,
            "metrics": brain.metrics(),
        }

    def chat_receipt(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        """Read one atomically committed turn without loading neural tensors."""

        del request_id
        brain_id = self._brain_id(params)
        turn_id = AdaptiveBrain._validated_chat_turn_id(
            str(params.get("turnId", ""))
        )
        if not turn_id:
            raise RpcFault(-32602, "params.turnId is required")
        input_sha256 = str(params.get("inputSha256", ""))
        if not AdaptiveBrain._sha256_identifier(input_sha256):
            raise RpcFault(-32602, "params.inputSha256 is invalid")
        minimum_inference = params.get("minimumInferenceCount", 0)
        if (
            isinstance(minimum_inference, bool)
            or not isinstance(minimum_inference, int)
            or minimum_inference < 0
        ):
            raise RpcFault(
                -32602, "params.minimumInferenceCount is invalid"
            )
        metadata_path = self._storage(params, brain_id) / "engine" / "brain.json"
        try:
            if metadata_path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("engine metadata exceeds receipt query bound")
            metadata = json.loads(metadata_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise RpcFault(
                -32004, "committed chat state is unavailable"
            ) from error
        if (
            not isinstance(metadata, dict)
            or str(metadata.get("brain_id", "")) != brain_id
        ):
            raise RpcFault(-32004, "committed chat state identity is invalid")
        messages = metadata.get("messages", [])
        traces = metadata.get("traces", [])
        counters = metadata.get("counters", {})
        if (
            not isinstance(messages, list)
            or not isinstance(traces, list)
            or not isinstance(counters, dict)
        ):
            raise RpcFault(-32004, "committed chat state is invalid")
        inference_count = counters.get("inference_count", 0)
        plasticity_events = counters.get("plasticity_events", 0)
        consolidation_cycles = counters.get("consolidation_cycles", 0)
        if (
            isinstance(inference_count, bool)
            or not isinstance(inference_count, int)
            or inference_count < 0
            or isinstance(plasticity_events, bool)
            or not isinstance(plasticity_events, int)
            or plasticity_events < 0
            or isinstance(consolidation_cycles, bool)
            or not isinstance(consolidation_cycles, int)
            or consolidation_cycles < 0
        ):
            raise RpcFault(-32004, "committed chat counters are invalid")
        try:
            receipts = AdaptiveBrain._validated_completed_chat_turns(
                metadata.get("completed_chat_turns")
            )
        except ValueError as error:
            raise RpcFault(
                -32004, "committed chat receipts are invalid"
            ) from error
        receipt = next(
            (
                value
                for value in reversed(receipts)
                if value.get("turnId") == turn_id
                and value.get("inputSha256") == input_sha256
            ),
            None,
        )
        legacy_matched = False
        if receipt is not None:
            if (
                int(receipt["inferenceCount"]) != minimum_inference + 1
                or int(receipt["inferenceCount"]) != inference_count
            ):
                return self._missing_chat_receipt(
                    brain_id, turn_id, input_sha256, inference_count
                )
            human_id = str(receipt["humanMessageId"])
            brain_message_id = str(receipt["brainMessageId"])
            trace_id = str(receipt["traceId"])
            committed_inference = int(receipt["inferenceCount"])
            parameter_checksum = str(receipt["parameterChecksumAfter"])
        else:
            # One-way reconciliation for a turn committed just before this
            # receipt format shipped. Accept only the latest atomic pair and
            # exactly one inference beyond the caller's pre-turn baseline.
            # Once any receipt exists, never reinterpret a current-format
            # commit under another caller-provided turn id.
            if (
                receipts
                or inference_count != minimum_inference + 1
                or len(messages) < 2
            ):
                return self._missing_chat_receipt(
                    brain_id, turn_id, input_sha256, inference_count
                )
            human_candidate = messages[-2]
            brain_candidate = messages[-1]
            trace_candidate = traces[-1] if traces else None
            if (
                not isinstance(human_candidate, dict)
                or not isinstance(brain_candidate, dict)
                or not isinstance(trace_candidate, dict)
                or human_candidate.get("role") != "human"
                or brain_candidate.get("role") != "brain"
                or hashlib.sha256(
                    str(human_candidate.get("content", "")).encode("utf-8")
                ).hexdigest()
                != input_sha256
                or trace_candidate.get("input_sha256") != input_sha256
                or str(trace_candidate.get("created_at", ""))
                != str(brain_candidate.get("created_at", ""))
                or "turn_id" in human_candidate
                or "turn_id" in brain_candidate
                or "turn_id" in trace_candidate
            ):
                return self._missing_chat_receipt(
                    brain_id, turn_id, input_sha256, inference_count
                )
            human_id = str(human_candidate.get("id", ""))
            brain_message_id = str(brain_candidate.get("id", ""))
            trace_id = str(trace_candidate.get("id", ""))
            parameter_checksum = str(
                trace_candidate.get("parameter_checksum_after", "")
            )
            committed_inference = inference_count
            legacy_matched = True

        human_message = next(
            (
                value
                for value in messages
                if isinstance(value, dict) and value.get("id") == human_id
            ),
            None,
        )
        brain_message = next(
            (
                value
                for value in messages
                if isinstance(value, dict)
                and value.get("id") == brain_message_id
            ),
            None,
        )
        trace = next(
            (
                value
                for value in traces
                if isinstance(value, dict) and value.get("id") == trace_id
            ),
            None,
        )
        if receipt is not None and (
            human_message is None or brain_message is None or trace is None
        ):
            ledger = NeuralConversationLedger(
                metadata_path.parent / "conversation.sqlite3",
                brain_id,
            )
            try:
                human_message = ledger.payload_by_id("message", human_id)
                brain_message = ledger.payload_by_id(
                    "message", brain_message_id
                )
                trace = ledger.payload_by_id("trace", trace_id)
            finally:
                ledger.close()
        if (
            not isinstance(human_message, dict)
            or not isinstance(brain_message, dict)
            or not isinstance(trace, dict)
            or human_message.get("role") != "human"
            or brain_message.get("role") != "brain"
            or not human_id
            or not brain_message_id
            or not trace_id
            or not AdaptiveBrain._sha256_identifier(parameter_checksum)
            or trace.get("parameter_checksum_after") != parameter_checksum
            or hashlib.sha256(
                str(human_message.get("content", "")).encode("utf-8")
            ).hexdigest()
            != input_sha256
        ):
            raise RpcFault(-32004, "committed chat receipt references invalid state")
        if not legacy_matched and (
            human_message.get("turn_id") != turn_id
            or brain_message.get("turn_id") != turn_id
            or trace.get("turn_id") != turn_id
        ):
            raise RpcFault(-32004, "committed chat turn binding is invalid")
        from omni_core.chat_steering import validate_no_reply_turn
        try:
            no_reply = validate_no_reply_turn(human_message, brain_message, trace, receipt or {})
        except ValueError as error:
            raise RpcFault(-32004, "committed no-reply evidence is invalid") from error

        def external_message(
            value: Dict[str, Any], *, trace_value: str = ""
        ) -> Dict[str, Any]:
            created_at = str(
                value.get("created_at", value.get("createdAt", ""))
            )
            result = {
                "id": str(value.get("id", "")),
                "role": str(value.get("role", "")),
                "content": str(value.get("content", "")),
                "createdAt": created_at,
            }
            runtime = value.get("runtime")
            if isinstance(runtime, str) and runtime:
                result["runtime"] = runtime
            if trace_value:
                result["traceId"] = trace_value
            epoch = value.get("attention_epoch", value.get("attentionEpoch"))
            if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0:
                result["attentionEpoch"] = epoch
            if value.get("generation_end") is not None:
                result["generation_end"] = value["generation_end"]
            return result

        substrate = metadata.get("substrate", {})
        substrate_persistence = (
            substrate.get("persistence", {})
            if isinstance(substrate, dict)
            else {}
        )
        mutable_state = metadata.get("mutable_state", {})
        return {
            "format": "omni-chat-turn-receipt-query",
            "formatVersion": 1,
            "brainId": brain_id,
            "turnId": turn_id,
            "committed": True,
            "turnCommitted": True,
            "legacyMatched": legacy_matched,
            "inputSha256": input_sha256,
            "humanMessage": external_message(human_message),
            "brainMessage": external_message(
                brain_message, trace_value=trace_id
            ),
            "trace": dict(trace),
            "inferenceCount": committed_inference,
            "plasticityEvents": int(plasticity_events),
            "consolidationCycles": int(consolidation_cycles),
            "parameterChecksumAfter": parameter_checksum,
            "engineUpdatedAt": str(metadata.get("updated_at", "")),
            "substrateGeneration": str(
                substrate_persistence.get("activeGeneration", "")
            ),
            "mutableStateGeneration": str(
                mutable_state.get("activeGeneration", "")
                if isinstance(mutable_state, dict)
                else ""
            ),
            "idempotentCompletion": True,
            "noReply": no_reply,
            **({"generationEnd": receipt["generationEnd"]}
               if receipt is not None and "generationEnd" in receipt else {}),
        }

    @staticmethod
    def _missing_chat_receipt(
        brain_id: str,
        turn_id: str,
        input_sha256: str,
        inference_count: int,
    ) -> Dict[str, Any]:
        return {
            "format": "omni-chat-turn-receipt-query",
            "formatVersion": 1,
            "brainId": brain_id,
            "turnId": turn_id,
            "committed": False,
            "turnCommitted": False,
            "inputSha256": input_sha256,
            "inferenceCount": max(0, int(inference_count)),
        }

    def train(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        texts = params.get("texts")
        if texts is None and isinstance(params.get("text"), str):
            texts = [params["text"]]
        if texts is not None and not (
            isinstance(texts, list)
            and all(isinstance(value, str) for value in texts)
        ):
            raise RpcFault(-32602, "params.texts must be a string array")
        source_ids = params.get("sourceIds")
        if source_ids is not None and not (
            isinstance(source_ids, list)
            and all(isinstance(value, str) for value in source_ids)
        ):
            raise RpcFault(-32602, "params.sourceIds must be a string array")
        if not any(
            isinstance(value, str) and value.replace("\x00", "").strip()
            for value in (texts or [])
        ) and not any(
            isinstance(value, str) and value.strip()
            for value in (source_ids or [])
        ):
            raise RpcFault(
                -32602,
                "training requires non-empty text or explicit retained sourceIds",
            )
        brain, job_id, progress = self._job(params, request_id, "training")
        result = brain.train(
            texts=texts,
            epochs=int(params.get("epochs", params.get("steps", 1))),
            learning_rate=(
                _number(params.get("learningRate"))
                if params.get("learningRate") is not None
                else None
            ),
            source_ids=source_ids,
            progress=progress,
        )
        self._job_complete(brain, job_id, "training", result)
        return result

    def conversation_page(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        before_value = params.get("beforeSequence")
        before = None if before_value is None else int(before_value)
        limit = max(1, min(1000, int(params.get("limit", 100))))
        if before is not None and before < 1:
            raise RpcFault(-32602, "conversation cursor is invalid")
        kinds_value = params.get("kinds", ["message", "action", "trace"])
        if not isinstance(kinds_value, list) or any(
            item not in {"message", "action", "trace"}
            for item in kinds_value
        ):
            raise RpcFault(-32602, "conversation kinds are invalid")
        entries = brain.conversation.page(
            before_sequence=before,
            limit=limit,
            kinds=tuple(str(item) for item in kinds_value),
        )
        summary = brain.conversation.summary()
        first = entries[0]["sequence"] if entries else None
        return {
            "brainId": brain.brain_id,
            "entries": entries,
            "summary": summary,
            "hasOlder": bool(first is not None and first > 1),
            **(
                {"nextBeforeSequence": int(first)}
                if first is not None and first > 1
                else {}
            ),
        }

    def ingest(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        transaction_key = params.get("transactionKey", params.get("idempotencyKey", ""))
        if not isinstance(transaction_key, str) or (
            transaction_key and not re.fullmatch(r"[a-f0-9]{64}", transaction_key)
        ):
            raise RpcFault(-32602, "ingestion transaction key must be a lowercase sha256")
        if (
            params.get("transactionKey") is not None
            and params.get("idempotencyKey") is not None
            and params["transactionKey"] != params["idempotencyKey"]
        ):
            raise RpcFault(-32602, "ingestion idempotency keys disagree")
        brain, job_id, progress = self._job(params, request_id, "ingestion")
        try:
            result = brain.ingest(
                path=str(params["path"]) if params.get("path") else None,
                text=(
                    str(params["text"])
                    if params.get("text") is not None
                    else None
                ),
                name=str(params.get("name", "")),
                kind=str(params.get("kind", "")),
                policy=str(params.get("policy", "encode")),
                expected_hash=str(
                    params.get(
                        "contentHash", params.get("expectedSha256", "")
                    )
                ),
                committed_sqlite_snapshot=bool(
                    params.get("committedSqliteSnapshot", False)
                ),
                allow_replay=bool(params.get("allowReplay", False)),
                epoch=int(params.get("epoch", 0)),
                transaction_key=transaction_key,
                progress=progress,
            )
        except Exception as error:
            # Ingestion mutates fast weights and assemblies as records stream,
            # but the durable cursor is committed only after the whole worker
            # call succeeds. Drop that in-memory object and reload the last
            # atomic checkpoint so retrying the same manifest record cannot
            # inherit a partial application.
            recovered_error: Exception = error
            if (
                not isinstance(
                    error,
                    (NeuralStateResourcePause, DatasetResourcePause),
                )
                and is_allocator_oom_error(error)
            ):
                brain._allocator_oom_count += 1
                brain._release_training_allocator_cache()
                recovered_error = brain._allocator_resource_pause(
                    error,
                    stage="ingestion",
                )
            brain_id = brain.brain_id
            storage = brain.storage_path
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
            restored = AdaptiveBrain.load(
                storage, expected_brain_id=brain_id
            )
            self.brains[brain_id] = restored
            resource_status: Optional[Dict[str, Any]] = None
            if isinstance(
                recovered_error,
                (NeuralStateResourcePause, DatasetResourcePause),
            ):
                resource_status = dict(recovered_error.status)
                resource_status.setdefault("recoverable", True)
            runtime_downgrade: Optional[Dict[str, int]] = None
            if resource_status is not None and bool(
                resource_status.get("allocatorOutOfMemory", False)
            ):
                runtime_downgrade = restored.apply_allocator_oom_downgrade(
                    resource_status
                )
            active_checkpoints = [
                {
                    "transactionId": str(value.get("transactionId", "")),
                    "contentHash": str(value.get("contentHash", "")),
                    "epoch": int(value.get("epoch", 0)),
                    "policy": str(value.get("policy", "")),
                    "committedRecords": int(value.get("committedRecords", 0)),
                    "visitedRecords": int(value.get("visitedRecords", 0)),
                    "commitSequence": int(value.get("commitSequence", 0)),
                }
                for value in restored.ingestion_checkpoints.values()
            ]
            restored.events.append(
                "ingestion-rollback",
                {
                    "jobId": job_id,
                    "reason": (
                        "allocator pause; uncommitted record batch discarded"
                        if resource_status is not None
                        and bool(resource_status.get("allocatorOutOfMemory", False))
                        else "uncommitted record batch discarded"
                    ),
                    "activeCheckpoints": active_checkpoints,
                    "uncommittedLearningRepresentedAsCommitted": False,
                    "resourcePause": resource_status,
                    "runtimeDowngrade": runtime_downgrade,
                },
                job_id=job_id or None,
            )
            self.notify(
                "ingestion-rolled-back",
                brain_id=brain_id,
                job_id=job_id,
                progress=0.0,
                message=(
                    "Training paused and the last atomic checkpoint was restored; "
                    "resume will retry only the uncommitted dataset suffix."
                    if resource_status is not None
                    else "Partial ingestion was discarded; the last atomic "
                    "checkpoint was restored."
                ),
                data=(
                    {
                        "resourcePause": resource_status,
                        "runtimeDowngrade": runtime_downgrade,
                        "activeCheckpoints": active_checkpoints,
                    }
                    if resource_status is not None
                    else None
                ),
            )
            if resource_status is not None:
                if job_id:
                    restored.events.append(
                        "job-paused",
                        {
                            "kind": "ingestion",
                            "resourcePause": resource_status,
                            "runtimeDowngrade": runtime_downgrade,
                        },
                        job_id=job_id,
                    )
                raise RpcFault(
                    -32020,
                    str(recovered_error),
                    {
                        "recoverable": True,
                        "resourcePause": resource_status,
                        "runtimeDowngrade": runtime_downgrade,
                        "activeCheckpoints": active_checkpoints,
                    },
                ) from error
            raise
        self._job_complete(brain, job_id, "ingestion", result)
        return result

    @staticmethod
    def _observation_session_payload(
        session: LiveObservationSessionState,
    ) -> Dict[str, Any]:
        return {
            "id": session.session_id,
            "brainId": session.brain_id,
            "modalities": list(session.modalities),
            "permission": copy.deepcopy(session.permission),
            "retention": session.retention,
            "capabilities": dict(session.capabilities),
            "packetsAccepted": int(session.packets_accepted),
            "bytesAccepted": int(session.bytes_accepted),
            "lastSequence": int(session.last_sequence),
            "maxPacketBytes": int(session.max_packet_bytes),
            "createdAt": session.created_at,
            "rawPacketsStored": False,
            "datasetCoverageCommitted": False,
        }

    def start_observation(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        session_id = str(params.get("sessionId", "")).strip()
        if not session_id or len(session_id) > 128 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in session_id
        ):
            raise RpcFault(-32602, "params.sessionId is invalid")
        raw_modalities = params.get("modalities")
        if (
            not isinstance(raw_modalities, list)
            or not raw_modalities
            or len(raw_modalities) > 3
        ):
            raise RpcFault(-32602, "params.modalities must be a non-empty array")
        modalities = tuple(str(value).strip().lower() for value in raw_modalities)
        if len(set(modalities)) != len(modalities) or any(
            value not in {"image", "audio", "video"} for value in modalities
        ):
            raise RpcFault(-32602, "params.modalities contains an invalid value")
        permission = params.get("permission")
        if not isinstance(permission, dict):
            raise RpcFault(-32602, "params.permission is required")
        source = str(permission.get("source", "")).strip().lower()
        granted_at = str(permission.get("grantedAt", "")).strip()
        device_hash = str(permission.get("deviceIdHash", "")).strip().lower()
        if (
            permission.get("granted") is not True
            or permission.get("scope") != "session"
            or source not in {"camera", "microphone", "screen", "mixed"}
            or not granted_at
            or len(granted_at) > 64
            or (device_hash and (
                len(device_hash) != 64
                or any(character not in "0123456789abcdef" for character in device_hash)
            ))
        ):
            raise RpcFault(-32602, "params.permission is invalid")
        permitted_modalities = {
            "camera": {"image", "video"},
            "microphone": {"audio"},
            "screen": {"image", "video"},
            "mixed": {"image", "audio", "video"},
        }[source]
        if set(modalities).difference(permitted_modalities):
            raise RpcFault(
                -32602,
                "capture permission source does not cover every requested modality",
            )
        retention = str(params.get("retention", "neural")).strip().lower()
        if retention not in {"working", "neural"}:
            raise RpcFault(-32602, "params.retention must be working or neural")
        max_packet_bytes = params.get("maxPacketBytes")
        if (
            isinstance(max_packet_bytes, bool)
            or not isinstance(max_packet_bytes, int)
            or max_packet_bytes < 1
        ):
            raise RpcFault(
                -32602,
                "params.maxPacketBytes must be a positive resource-derived integer",
            )
        available_memory = _available_memory_bytes()
        worker_resource_bound = (
            max(1, available_memory // 12)
            if available_memory is not None
            else max_packet_bytes
        )
        if max_packet_bytes > worker_resource_bound:
            raise RpcFault(
                -32020,
                "live observation packet envelope exceeds current worker resources",
                {
                    "recoverable": True,
                    "requestedPacketBytes": max_packet_bytes,
                    "resourceDerivedPacketBytes": worker_resource_bound,
                },
            )
        try:
            tool_schemas = AdaptiveBrain._normalize_tool_schemas(
                params.get("toolSchemas", [])
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error
        readiness = self._trained_modality_capabilities(brain)
        capabilities = {
            "imageNeural": readiness["imageNeural"],
            "audioNeural": readiness["audioNeural"],
            "videoNeural": readiness["videoNeural"],
        }
        unavailable = [
            modality
            for modality in modalities
            if not capabilities[modality + "Neural"]
        ]
        if unavailable:
            raise RpcFault(
                -32021,
                "live neural path is not trained for: %s" % ", ".join(unavailable),
                {"capabilities": capabilities, "unavailableModalities": unavailable},
            )
        session = LiveObservationSessionState(
            session_id=session_id,
            brain_id=brain.brain_id,
            modalities=modalities,
            permission={
                "source": source,
                "granted": True,
                "scope": "session",
                "grantedAt": granted_at,
                **({"deviceIdHash": device_hash} if device_hash else {}),
            },
            retention=retention,
            tool_schemas=[dict(value) for value in tool_schemas],
            capabilities=capabilities,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            max_packet_bytes=max_packet_bytes,
        )
        with self._observation_lock:
            if session_id in self._observation_sessions:
                raise RpcFault(-32602, "observation session already exists")
            self._observation_sessions[session_id] = session
        return self._observation_session_payload(session)

    def observe_packet(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        session_id = str(params.get("sessionId", "")).strip()
        with self._observation_lock:
            session = self._observation_sessions.get(session_id)
        if session is None:
            raise RpcFault(-32004, "live observation session was not found")
        modality = str(params.get("modality", "")).strip().lower()
        if modality not in session.modalities:
            raise RpcFault(-32602, "packet modality is not enabled for this session")
        sequence = params.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RpcFault(-32602, "packet sequence must be a non-negative integer")
        timestamp_ms = _number(params.get("timestampMs"))
        if timestamp_ms < 0:
            raise RpcFault(-32602, "packet timestampMs must be non-negative")
        mime_type = str(params.get("mimeType", "")).strip().lower()
        encoded = params.get("dataBase64")
        if not isinstance(encoded, str):
            raise RpcFault(-32602, "packet dataBase64 must be a string")
        maximum_encoded = ((session.max_packet_bytes + 2) // 3) * 4
        if len(encoded) > maximum_encoded:
            raise RpcFault(
                -32602,
                "live packet exceeds the %d-byte maximum"
                % session.max_packet_bytes,
            )
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError, TypeError) as error:
            raise RpcFault(-32602, "packet dataBase64 is invalid") from error
        if not payload or len(payload) > session.max_packet_bytes:
            raise RpcFault(
                -32602,
                "live packet must contain 1 to %d bytes"
                % session.max_packet_bytes,
            )
        settings = params.get("settings", {})
        if not isinstance(settings, dict):
            raise RpcFault(-32602, "packet settings must be an object")
        with self._observation_lock:
            current = self._observation_sessions.get(session_id)
            if current is None:
                raise RpcFault(-32004, "live observation session was stopped")
            if sequence <= current.last_sequence:
                raise RpcFault(-32602, "packet sequence must increase monotonically")
            if timestamp_ms < current.last_timestamp_ms:
                raise RpcFault(-32602, "packet timestamp must not move backwards")
        brain = self.brains.get(session.brain_id)
        if brain is None:
            raise RpcFault(-32004, "live observation brain is no longer loaded")
        try:
            observation = brain.observe_live_packet(
                modality=modality,
                mime_type=mime_type,
                payload=payload,
                session_id=session_id,
                sequence=sequence,
                timestamp_ms=timestamp_ms,
                retention=session.retention,
                settings=settings,
                permission_source=str(session.permission["source"]),
            )
            organic = (
                brain.idle_cycle(
                    tool_schemas=session.tool_schemas,
                    minimum_idle_seconds=0.0,
                )
                if session.retention == "neural"
                else {"ran": False, "actions": []}
            )
            if session.retention == "neural" and not bool(organic.get("ran")):
                brain.save()
        except Exception as error:
            if is_allocator_oom_error(error):
                brain._allocator_oom_count += 1
                brain._release_training_allocator_cache()
            brain_id = brain.brain_id
            storage = brain.storage_path
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.close()
            self.brains[brain_id] = AdaptiveBrain.load(
                storage, expected_brain_id=brain_id
            )
            if is_allocator_oom_error(error):
                status = self.brains[brain_id].resource_policy.status()
                status.update(
                    {
                        "recoverable": True,
                        "allocatorOutOfMemory": True,
                        "failureStage": "live-%s-observation" % modality,
                        "packetCommitted": False,
                        "datasetCoverageCommitted": False,
                    }
                )
                raise RpcFault(
                    -32020,
                    "live neural observation paused after allocator exhaustion",
                    {"resourcePause": status},
                ) from error
            raise
        with self._observation_lock:
            current = self._observation_sessions.get(session_id)
            if current is None:
                raise RpcFault(-32004, "live observation session was stopped")
            current.last_sequence = sequence
            current.last_timestamp_ms = timestamp_ms
            current.packets_accepted += 1
            current.bytes_accepted += len(payload)
            session_payload = self._observation_session_payload(current)
        return {
            "session": session_payload,
            "observation": observation,
            "actions": list(organic.get("actions", [])),
            "trace": organic.get("trace"),
        }

    def _finish_observation(
        self, params: Dict[str, Any], *, cancelled: bool
    ) -> Dict[str, Any]:
        session_id = str(params.get("sessionId", "")).strip()
        with self._observation_lock:
            session = self._observation_sessions.pop(session_id, None)
        if session is None:
            raise RpcFault(-32004, "live observation session was not found")
        payload = self._observation_session_payload(session)
        payload["state"] = "cancelled" if cancelled else "stopped"
        payload["rawPacketsStored"] = False
        return payload

    def stop_observation(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        return self._finish_observation(params, cancelled=False)

    def cancel_observation(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        return self._finish_observation(params, cancelled=True)

    def resolve_observation_control(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        session_id = str(params.get("sessionId", "")).strip()
        with self._observation_lock:
            session = self._observation_sessions.get(session_id)
        if session is None:
            raise RpcFault(-32004, "live observation session was not found")
        brain = self._get(params)
        if session.brain_id != brain.brain_id:
            raise RpcFault(-32602, "live observation control brain does not match")
        control = params.get("control")
        if not isinstance(control, dict):
            raise RpcFault(-32602, "params.control is required")
        try:
            result = brain.record_live_observation_control(
                session_id=session_id, control=control
            )
            brain.save()
            return result
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

    def generate_neural_speech(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        text, speech_id = params.get("text"), params.get("speechRequestId")
        if not isinstance(text, str) or not text.strip() or "\x00" in text or \
                not isinstance(speech_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", speech_id):
            raise RpcFault(-32602, "invalid same-brain speech waveform request")
        rate = _number(params.get("rate", 1.0))
        if rate <= 0:
            raise RpcFault(-32602, "speech presentation rate must be positive")
        try:
            require_parser_resources("speech text-to-idea conditioning", ram_bytes=len(text) * 64 + 131072)
        except DatasetResourcePause as error:
            raise RpcFault(-32020, "speech waveform generation paused at physical text allocation admission", {"resourcePause": error.status}) from error
        words = max(1, len(text.split()))
        result = self.generate_modality({**params, "modality": "audio", "prompt": text,
            "inputPath": "", "settings": {"outputMode": "auto", "sampleRate": 16000,
                "durationMs": max(250.0, words * 1000.0 / 3.0)}}, request_id)
        result["speech"] = {"requestId": speech_id, "source": "same-brain-audio-region",
            "textSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "textConditioned": True, "sameBrain": True, "externalModelUsed": False,
            "intelligibilityVerified": False,
            "pairedExamples": int(result.get("speechPairedExamples", 0)),
            "trainingState": "needs-speech-training" if not result.get("speechPairedExamples", 0) else "speech-quality-unverified"}
        return result

    def generate_modality(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain, job_id, progress = self._job(
            params, request_id, "modality-generation"
        )
        try:
            inline_result = self._claim_inline_generation(brain, params, job_id)
        except NeuralStateResourcePause as error:
            raise RpcFault(-32020, "inline artifact snapshot awaits physical resources", {"resourcePause": error.status}) from error
        except ModalityGenerationCancelled as error:
            raise RpcFault(-32800, "job was cancelled", {"modalityCancelled": True, "safeBoundary": True}) from error
        except MediaResourcePause as error:
            demand = error.demand
            status = brain.resource_policy.status(
                estimated_write_bytes=demand.output_bytes,
                estimated_ram_bytes=demand.working_bytes,
            )
            raise RpcFault(
                -32020,
                "modality generation paused at the live resource watermark",
                {
                    "resourcePause": status,
                    "mediaDemand": {
                        "stage": demand.stage,
                        "completedUnits": demand.completed_units,
                        "totalUnits": demand.total_units,
                        "workingBytes": demand.working_bytes,
                        "outputBytes": demand.output_bytes,
                    },
                },
            ) from error
        except Exception as error:
            if not is_allocator_oom_error(error):
                raise
            brain._allocator_oom_count += 1
            brain._release_training_allocator_cache()
            status = brain.resource_policy.status()
            status.update(
                {
                    "recoverable": True,
                    "allocatorOutOfMemory": True,
                    "failureStage": "inline-modality-generation",
                    "artifactCommitted": False,
                    "neuralStateRollbackRequired": False,
                    "retryWithSmallerMediaProfile": True,
                }
            )
            raise RpcFault(
                -32020,
                "modality generation paused after allocator exhaustion",
                {"resourcePause": status},
            ) from error
        if inline_result is not None:
            progress(0.98, "Committing the imagination formed during chat")
            self._job_complete(
                brain,
                job_id,
                "modality-generation",
                inline_result,
            )
            return inline_result
        progress(0.2, "Activating internal idea vectors")
        preview_revision = 0

        def preview(
            generation_progress: float,
            mime_type: str,
            payload: bytes,
            details: Dict[str, Any],
        ) -> None:
            nonlocal preview_revision
            cache_key = hashlib.sha256(
                (job_id or str(request_id or "unbound-preview")).encode("utf-8")
            ).hexdigest()[:32]
            digest, artifact_path = _write_content_addressed_preview(
                brain.engine_path / PREVIEW_CACHE_DIRECTORY / cache_key,
                mime_type,
                payload,
            )
            bounded_progress = 0.2 + 0.75 * max(
                0.0, min(float(generation_progress), 1.0)
            )
            status_label = _preview_status(details)
            preview_value: Dict[str, Any] = {
                **copy.deepcopy(details),
                "schemaVersion": 1,
                "producer": "same-brain-decoder",
                "payloadSha256": digest,
                "artifactPath": str(artifact_path),
                "revision": preview_revision,
                "progress": bounded_progress,
                "statusLabel": status_label,
                "mimeType": mime_type,
            }
            embedded_media = inline_media_data_url(mime_type, payload)
            if embedded_media is not None:
                preview_value["dataUrl"] = embedded_media
            self.notify(
                "modality-preview",
                brain_id=brain.brain_id,
                job_id=job_id,
                sequence=preview_revision,
                progress=bounded_progress,
                message=status_label,
                data={"preview": preview_value},
            )
            preview_revision += 1

        modality = str(params.get("modality", ""))
        try:
            result = brain.generate_modality(
                modality=modality,
                prompt=str(params.get("prompt", "")),
                concept_ids=params.get("conceptIds"),
                concept_id_view=params.get("conceptIdView"),
                source_turn_id=str(params.get("sourceTurnId", "")),
                input_path=str(params.get("inputPath", "")),
                settings=(
                    params.get("settings")
                    if isinstance(params.get("settings"), dict)
                    else {}
                ),
                seed=(
                    int(params["seed"])
                    if params.get("seed") is not None
                    else None
                ),
                preview_callback=preview,
                cancel_check=lambda: bool(
                    job_id and job_id in self.cancelled_jobs
                ) or self._cooperative_cancel.is_set(),
            )
        except ModalityGenerationCancelled as error:
            raise RpcFault(-32800, "job was cancelled", {"modalityCancelled": True, "safeBoundary": True}) from error
        except MediaResourcePause as error:
            demand = error.demand
            status = brain.resource_policy.status(
                estimated_write_bytes=demand.output_bytes,
                estimated_ram_bytes=demand.working_bytes,
            )
            raise RpcFault(
                -32020,
                "modality generation paused at the live resource watermark",
                {
                    "resourcePause": status,
                    "mediaDemand": {
                        "stage": demand.stage,
                        "completedUnits": demand.completed_units,
                        "totalUnits": demand.total_units,
                        "workingBytes": demand.working_bytes,
                        "outputBytes": demand.output_bytes,
                    },
                },
            ) from error
        except Exception as error:
            if not is_allocator_oom_error(error):
                raise
            brain._allocator_oom_count += 1
            brain._release_training_allocator_cache()
            status = brain.resource_policy.status()
            status.update(
                {
                    "recoverable": True,
                    "allocatorOutOfMemory": True,
                    "failureStage": "%s-generation" % (modality or "modality"),
                    "artifactCommitted": False,
                    "neuralStateRollbackRequired": False,
                    "retryWithSmallerMediaProfile": True,
                }
            )
            raise RpcFault(
                -32020,
                "modality generation paused after allocator exhaustion",
                {"resourcePause": status},
            ) from error
        if params.get("speechRequestId"):
            result["speechPairedExamples"] = max(0, int(brain.modality_training.get("audio_speech_pairs", 0)))
        self._job_complete(brain, job_id, "modality-generation", result)
        return result

    def evolution_propose(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain, job_id, progress = self._job(
            params, request_id, "neural-evolution-proposal"
        )
        texts = params.get("texts")
        if texts is None and isinstance(params.get("text"), str):
            texts = [params["text"]]
        if texts is not None and not (
            isinstance(texts, list)
            and all(isinstance(value, str) for value in texts)
        ):
            raise RpcFault(-32602, "params.texts must be a string array")
        source_ids = params.get("sourceIds")
        if source_ids is not None and not (
            isinstance(source_ids, list)
            and all(isinstance(value, str) for value in source_ids)
        ):
            raise RpcFault(-32602, "params.sourceIds must be a string array")
        objectives = params.get("objectives")
        if objectives is not None and not (
            isinstance(objectives, list)
            and all(isinstance(value, str) for value in objectives)
        ):
            raise RpcFault(-32602, "params.objectives must be a string array")
        provenance = params.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise RpcFault(-32602, "params.provenance must be an object")
        architecture = params.get("architectureChange")
        if architecture is not None and not isinstance(architecture, dict):
            raise RpcFault(
                -32602, "params.architectureChange must be an object"
            )
        try:
            result = NeuralEvolutionManager(brain).propose(
                texts=texts,
                source_ids=source_ids,
                epochs=int(params.get("epochs", params.get("steps", 1))),
                learning_rate=(
                    _number(params.get("learningRate"))
                    if params.get("learningRate") is not None
                    else None
                ),
                latent_replay=bool(params.get("latentReplay", False)),
                objectives=objectives,
                provenance=provenance,
                architecture_change=architecture,
                progress=progress,
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error
        self._job_complete(brain, job_id, "neural-evolution-proposal", result)
        return result

    def evolution_evaluate(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        candidate_id = str(params.get("candidateId", ""))
        if not candidate_id:
            raise RpcFault(-32602, "params.candidateId is required")
        try:
            return NeuralEvolutionManager(self._get(params)).evaluate(
                candidate_id
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

    def evolution_list(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        return NeuralEvolutionManager(self._get(params)).list()

    def evolution_promote(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        candidate_id = str(params.get("candidateId", ""))
        if not candidate_id:
            raise RpcFault(-32602, "params.candidateId is required")
        brain = self._get(params)
        try:
            result = NeuralEvolutionManager(brain).promote(candidate_id)
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error
        previous = self.brains.pop(brain.brain_id, None)
        if previous is not None:
            previous.close()
        reloaded = AdaptiveBrain.load(
            brain.storage_path, expected_brain_id=brain.brain_id
        )
        self.brains[brain.brain_id] = reloaded
        result["runtimeCard"] = reloaded.runtime_card()
        return result

    def evolution_reject(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        candidate_id = str(params.get("candidateId", ""))
        if not candidate_id:
            raise RpcFault(-32602, "params.candidateId is required")
        try:
            return NeuralEvolutionManager(self._get(params)).reject(
                candidate_id, str(params.get("reason", ""))
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

    def evolution_rollback(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        candidate_id = str(params.get("candidateId", ""))
        if not candidate_id:
            raise RpcFault(-32602, "params.candidateId is required")
        brain = self._get(params)
        try:
            result = NeuralEvolutionManager(brain).rollback(
                candidate_id, force=bool(params.get("force", False))
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error
        previous = self.brains.pop(brain.brain_id, None)
        if previous is not None:
            previous.close()
        reloaded = AdaptiveBrain.load(
            brain.storage_path, expected_brain_id=brain.brain_id
        )
        self.brains[brain.brain_id] = reloaded
        result["runtimeCard"] = reloaded.runtime_card()
        return result

    def export_ternary(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return brain.export_packed_ternary()

    def snapshot(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return brain.snapshot(str(params.get("label", "snapshot")))

    def checkpoint(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        operation_id = str(params.get("operationId", "")).strip()
        if not operation_id:
            raise RpcFault(-32602, "params.operationId is required")
        try:
            return self._get(params).checkpoint(operation_id)
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

    def trace(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        limit = max(1, min(int(params.get("limit", 50)), 1000))
        return {"brainId": brain.brain_id, "traces": brain.traces[-limit:]}

    def events(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return {
            "brainId": brain.brain_id,
            "integrity": brain.events.integrity(),
            "events": brain.events.recent(int(params.get("limit", 100))),
        }

    def cancel(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        job_id = str(params.get("jobId", ""))
        if not job_id:
            raise RpcFault(-32602, "params.jobId is required")
        self.cancelled_jobs.add(job_id)
        reason = str(params.get("reason", "")).strip() or "Cancelled by operator."
        requested_candidates = params.get("candidateIds", [])
        if not (
            isinstance(requested_candidates, list)
            and all(
                isinstance(value, str)
                and len(value) == 32
                and all(character in "0123456789abcdef" for character in value)
                for value in requested_candidates
            )
        ):
            raise RpcFault(-32602, "params.candidateIds must contain hex identifiers")
        requested_candidate_ids = set(requested_candidates)
        acknowledged_candidate_ids: List[str] = []
        brain_id = self._brain_id(params, required=False)
        storage = self._storage(params, brain_id) if brain_id else None
        if storage is not None:
            candidates_root = storage / "engine" / "candidates"
            if candidates_root.is_dir():
                for directory in candidates_root.iterdir():
                    record_path = directory / "candidate.json"
                    if (
                        directory.is_symlink()
                        or not directory.is_dir()
                        or not record_path.is_file()
                    ):
                        continue
                    try:
                        record = read_json(record_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    candidate_id = str(record.get("id", ""))
                    if (
                        directory.name != candidate_id
                        or len(candidate_id) != 32
                        or any(
                            character not in "0123456789abcdef"
                            for character in candidate_id
                        )
                    ):
                        continue
                    provenance = record.get("provenance", {})
                    provenance_request_id = (
                        str(provenance.get("runtimeRequestId", ""))
                        if isinstance(provenance, dict)
                        else ""
                    )
                    if not (
                        candidate_id in requested_candidate_ids
                        or provenance_request_id == job_id
                    ):
                        continue
                    if record.get("kind") != "neural-evolution":
                        continue
                    if record.get("status") not in {
                        "promoted",
                        "rolled-back",
                    }:
                        atomic_write_json(
                            record_path,
                            {
                                **record,
                                "status": "rejected",
                                "reason": reason,
                                "rejectedAt": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                ),
                                "cancelledJobId": job_id,
                                "cancellationAcknowledged": True,
                            },
                        )
                    acknowledged_candidate_ids.append(candidate_id)
        inline_cancelled = 0
        cleanup: List[InlineGeneration] = []
        with self._inline_lock:
            for key, record in list(self._inline_generations.items()):
                if record.job_id != job_id:
                    continue
                record.cancelled = True
                if record.future is not None:
                    record.future.cancel()
                self._inline_generations.pop(key, None)
                cleanup.append(record)
                inline_cancelled += 1
        for record in cleanup:
            if record.future is None or record.future.done():
                self._remove_inline_root(record)
        event_id = ""
        if storage is not None:
            loaded = self.brains.get(brain_id)
            owns_log = loaded is None
            events = (
                EventLog(storage / "engine" / "events.sqlite3", brain_id)
                if owns_log
                else loaded.events
            )
            try:
                existing = next(
                    (
                        event
                        for event in events.recent(10_000)
                        if event.get("kind") == "job-cancelled"
                        and event.get("jobId") == job_id
                    ),
                    None,
                )
                event_id = (
                    str(existing.get("eventId", ""))
                    if isinstance(existing, dict)
                    else events.append(
                        "job-cancelled",
                        {
                            "kind": str(params.get("kind", "runtime-job")),
                            "reason": reason,
                            "candidateIds": sorted(acknowledged_candidate_ids),
                            "workerPid": os.getpid(),
                            "terminationPrecededAcknowledgement": True,
                        },
                        job_id=job_id,
                    )
                )
            finally:
                if owns_log:
                    events.close()
            self.notify(
                "job-cancelled",
                brain_id=brain_id,
                job_id=job_id,
                progress=0.0,
                message="%s cancelled" % str(params.get("kind", "runtime-job")),
                data={
                    "acknowledged": True,
                    "candidateIds": sorted(acknowledged_candidate_ids),
                },
            )
        return {
            "jobId": job_id,
            "cancelled": True,
            "acknowledged": True,
            "eventId": event_id,
            "candidateIds": sorted(acknowledged_candidate_ids),
            "inlineGenerationsCancelled": inline_cancelled,
        }

    def shutdown(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del params, request_id
        self.running = False
        self._shutdown_inline_generations()
        with self._observation_lock:
            self._observation_sessions.clear()
        for brain in list(self.brains.values()):
            brain.close()
        self.brains.clear()
        self._release_all_owner_leases()
        return {"stopping": True}

    def dispatch(self, request: Any) -> Optional[Dict[str, Any]]:
        self._cleanup_expired_inline_generations()
        if not isinstance(request, dict):
            raise RpcFault(-32600, "request must be a JSON object")
        if request.get("jsonrpc") != "2.0":
            raise RpcFault(-32600, "jsonrpc must equal '2.0'")
        request_id = request.get("id")
        if request_id is not None and not isinstance(request_id, (str, int)):
            raise RpcFault(-32600, "id must be a string or number")
        method = request.get("method")
        if not isinstance(method, str):
            raise RpcFault(-32600, "method must be a string")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise RpcFault(-32602, "params must be an object")
        handler = self.methods.get(method)
        if handler is None:
            raise RpcFault(-32601, "method not found: %s" % method)
        correlated_id = str(request_id) if request_id is not None else ""
        owner = CodecOwner(correlated_id, str(params.get("brainId", "")), str(params.get("jobId", "")),
            str(params.get("streamId", "")), str(params.get("neuralActionId", "")))
        cooperatively_cancellable = method in {
            "hardware_projection_profile",
            "load",
            "chat",
            "consolidate_chat_learning",
            "configure_video_runtime",
            "generate_neural_speech",
            "generate_modality",
        }
        if cooperatively_cancellable:
            self._cooperative_cancel.clear()
            with self._active_request_lock:
                self._active_request = (method, correlated_id)
                self._artifact_request_owner = owner if method in {"generate_modality", "generate_neural_speech"} else None
        try:
            # Serialized neural dispatch is a quiescent migration boundary.
            # State/inspection/export requests do not move execution devices.
            if self.worker_role == "neural" and method in {
                "chat", "consolidate_chat_learning", "learn_tool_route_outcome",
                "train", "ingest", "generate_modality", "generate_neural_speech", "idle_cycle", "feedback",
                "observe_packet", "evolution.evaluate", "evolution_evaluate",
            }:
                brain = self._get(params)
                prepare = getattr(brain, "prepare_native_execution", None)
                if callable(prepare):
                    prepare(operation=method, quiescent=True)
            gateway = getattr(self, "_codec_gateway", None)
            with gateway.scope(owner, lambda: self._cooperative_cancel.is_set() or bool(owner.job_id and owner.job_id in self.cancelled_jobs)) if gateway else nullcontext():
                result = handler(params, correlated_id or None)
        except CodecRuntimeCancelled as error:
            raise RpcFault(-32800, str(error), {"codecRuntimeCancelled": True, "safeBoundary": True}) from error
        finally:
            if cooperatively_cancellable:
                with self._active_request_lock:
                    if self._active_request == (method, correlated_id):
                        self._active_request = None
                        self._artifact_request_owner = None
                self._cooperative_cancel.clear()
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RpcFault(-32602, "numeric parameter is invalid") from error
    if not (number == number and abs(number) != float("inf")):
        raise RpcFault(-32602, "numeric parameter must be finite")
    return number


def _available_memory_bytes() -> Optional[int]:
    """Best-effort live physical-memory probe without a product byte limit."""

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(1, int(status.available_physical))
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    worker = Worker()
    cooperative_signal = (
        getattr(signal, "SIGBREAK", None)
        if os.name == "nt"
        else getattr(signal, "SIGUSR1", None)
    )
    if cooperative_signal is not None:
        signal.signal(
            cooperative_signal,
            lambda _signum, _frame: worker.request_cooperative_cancel(),
        )
    serve_worker_stdio(worker, ProtocolLineReader(sys.stdin.buffer.fileno()), Worker._send,
                       _available_memory_bytes, RpcFault)
    for brain in worker.brains.values():
        try:
            brain.close()
        except Exception:
            pass
    worker._shutdown_inline_generations()
    worker._release_all_owner_leases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
