#!/usr/bin/env python3
"""OmniCortex JSON-RPC 2.0 stdio worker.

Stdout is protocol-only.  Diagnostics and tracebacks go to stderr so Electron
can safely parse one JSON response/notification per line.
"""

import base64
import copy
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import torch

from omni_core import AdaptiveBrain, OmniConfig, __version__
from omni_core.evolution import NeuralEvolutionManager
from omni_core.persistence import copy_substrate_snapshot


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 32 * 1024 * 1024
INLINE_GENERATION_TTL_SECONDS = 5 * 60
_STDOUT_LOCK = threading.Lock()


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


class IsolatedModalityDecoder:
    """A private copy of exactly one selected modality generator."""

    def __init__(self, modality: str, module: torch.nn.Module):
        self.modality = modality
        self.module = module
        self.module.eval()

    @torch.no_grad()
    def generate(
        self,
        modality: str,
        idea: torch.Tensor,
        seed: int = 0,
        preview_callback: Optional[Callable[[float, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        if modality != self.modality:
            raise ValueError("isolated modality snapshot does not match request")
        generator = torch.Generator(device=idea.device)
        generator.manual_seed(int(seed))
        return self.module.generate(
            idea,
            generator,
            preview_callback=preview_callback,
        )


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


class Worker:
    def __init__(self):
        self.brains: Dict[str, AdaptiveBrain] = {}
        self.cancelled_jobs = set()
        self.running = True
        self._inline_lock = threading.RLock()
        self._inline_generations: Dict[Tuple[str, str], InlineGeneration] = {}
        self._inline_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="omni-inline-imagination",
        )
        self._inline_executor_closed = False
        self.methods: Dict[str, Callable[[Dict[str, Any], Optional[str]], Any]] = {
            "health": self.health,
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
            "workspace": self.workspace,
            "feedback": self.feedback,
            "idle_cycle": self.idle_cycle,
            "chat": self.chat,
            "train": self.train,
            "ingest": self.ingest,
            "consolidate": self.consolidate,
            "generate_modality": self.generate_modality,
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
            "snapshot": self.snapshot,
            "trace": self.trace,
            "events": self.events,
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
        # Chat tokens and background media previews can be emitted by separate
        # threads. Keep each protocol line atomic so Electron never receives
        # interleaved JSON fragments.
        with _STDOUT_LOCK:
            sys.stdout.write(serialized + "\n")
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
            "inputPath": input_path,
            "settings": copy.deepcopy(raw_settings),
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
    def _remove_inline_root(record: InlineGeneration) -> None:
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
        # chat thread. Subsequent slow-weight learning may safely continue on
        # the authoritative brain while the snapshot decodes in parallel.
        staging_root = (
            brain.engine_path / ".inline-imagination" / action_id
        ).resolve()
        staging_parent = (brain.engine_path / ".inline-imagination").resolve()
        try:
            staging_root.relative_to(staging_parent)
            staging_root.mkdir(parents=True, exist_ok=False)
            with torch.no_grad():
                idea = brain._modality_idea(
                    request["prompt"], request["conceptIds"]
                ).detach().clone()
                modality_snapshot = IsolatedModalityDecoder(
                    request["modality"],
                    copy.deepcopy(
                        getattr(brain.modalities, request["modality"])
                    ),
                )
        except Exception:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
            return None

        deferred_events = DeferredEventLog()
        snapshot = copy.copy(brain)
        snapshot.modalities = modality_snapshot
        snapshot.events = deferred_events
        snapshot.engine_path = staging_root
        snapshot.counters = dict(brain.counters)
        snapshot.modality_training = dict(brain.modality_training)
        snapshot.installed_modality_packs = copy.deepcopy(
            brain.installed_modality_packs
        )
        snapshot._modality_idea = (
            lambda _prompt="", _concept_ids=None: idea.detach().clone()
        )
        record = InlineGeneration(
            brain_id=brain.brain_id,
            action_id=action_id,
            stream_id=stream_id,
            signature=signature,
            staging_root=staging_root,
            events=deferred_events,
        )

        def preview(
            generation_progress: float,
            mime_type: str,
            payload: bytes,
        ) -> None:
            bounded_progress = 0.2 + 0.75 * max(
                0.0, min(float(generation_progress), 1.0)
            )
            preview_value: Dict[str, Any] = {
                "progress": bounded_progress,
                "statusLabel": "Decoding the current neural latent",
                "mimeType": str(mime_type),
            }
            if len(payload) <= 12 * 1024 * 1024:
                preview_value["dataUrl"] = (
                    "data:%s;base64,%s"
                    % (
                        mime_type,
                        base64.b64encode(payload).decode("ascii"),
                    )
                )
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
                result = snapshot.generate_modality(
                    modality=request["modality"],
                    prompt=request["prompt"],
                    concept_ids=request["conceptIds"],
                    input_path=request["inputPath"],
                    settings=request["settings"],
                    seed=request["seed"],
                    preview_callback=preview,
                )
                with self._inline_lock:
                    cancelled = record.cancelled
                if cancelled:
                    self._remove_inline_root(record)
                    raise RuntimeError("inline imagination was cancelled")
                return result
            except Exception:
                self._remove_inline_root(record)
                raise

        with self._inline_lock:
            self._inline_generations[key] = record
        try:
            record.future = self._inline_executor.submit(generate)
        except Exception:
            with self._inline_lock:
                self._inline_generations.pop(key, None)
            self._remove_inline_root(record)
            return None
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
                if record.signature != signature or record.cancelled:
                    return None
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
            result = dict(future.result())
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
            return existing
        # A hard process interruption can leave only disposable inline media
        # staging behind. It is never an authoritative brain artifact and is
        # removed before the persistent checkpoint is opened again.
        self._clear_inline_staging(storage / "engine")
        if (storage / "engine" / "brain.json").exists():
            brain = AdaptiveBrain.load(storage, expected_brain_id=brain_id)
        else:
            raw_config = params.get("config")
            if not isinstance(raw_config, dict):
                raise RpcFault(
                    -32004,
                    "brain is not initialized; params.config is required",
                )
            brain = AdaptiveBrain.create(
                brain_id,
                storage,
                OmniConfig.from_external(self._builder_config(params, raw_config)),
            )
        self.brains[brain_id] = brain
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
            elif self._directml_available():
                merged["device"] = "directml"
        return merged

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
        cuda = torch.cuda.is_available()
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
                "directml": self._directml_available(),
                "distributed": bool(torch.distributed.is_available()),
                "safetensors": True,
                "sqliteEventLog": True,
                "modalities": ["vision", "image", "audio", "video"],
            },
            "loadedBrains": len(self.brains),
        }

    def create(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        brain_id = self._brain_id(params, required=False) or uuid.uuid4().hex
        storage = self._storage(params, brain_id)
        self._discard_inline_generations(brain_id)
        self._clear_inline_staging(storage / "engine")
        raw_config = params.get("config") or {}
        if not isinstance(raw_config, dict):
            raise RpcFault(-32602, "params.config must be an object")
        config = OmniConfig.from_external(self._builder_config(params, raw_config))
        brain = AdaptiveBrain.create(brain_id, storage, config)
        self.brains[brain_id] = brain
        self.notify(
            "brain-created",
            brain_id=brain_id,
            progress=1.0,
            message=(
                "Bundled trained Omni Starter created."
                if config.origin_kind == "starter"
                else "Blank randomly initialized OmniCortex brain created."
            ),
        )
        return brain.summary()

    def load(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return brain.summary()

    def reload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        self._discard_inline_generations(brain_id)
        previous = self.brains.pop(brain_id, None)
        if previous is not None:
            previous.events.close()
        self._clear_inline_staging(self._storage(params, brain_id) / "engine")
        brain = AdaptiveBrain.load(
            self._storage(params, brain_id), expected_brain_id=brain_id
        )
        self.brains[brain_id] = brain
        return brain.summary()

    def unload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        self._discard_inline_generations(brain_id)
        previous = self.brains.pop(brain_id, None)
        if previous is not None:
            previous.events.close()
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
            engine = storage / "engine"
            try:
                # Validate and materialize every blob referenced by the
                # immutable shard graph before brain.json can commit it.
                copy_substrate_snapshot(snapshot, engine)
            except (OSError, ValueError) as error:
                raise RpcFault(
                    -32602,
                    "snapshot neural substrate failed validation: %s" % error,
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

    def query_substrate(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        query = params.get("query", {})
        if not isinstance(query, dict):
            raise RpcFault(-32602, "params.query must be an object")
        try:
            return brain.query_substrate(query)
        except (TypeError, ValueError) as error:
            raise RpcFault(-32602, str(error)) from error

    def workspace(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del request_id
        return self._get(params).workspace_snapshot()

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
        try:
            return self._get(params).feedback(
                text,
                str(direction),
                trace_id=str(params.get("traceId", "")),
                message_id=str(params.get("messageId", "")),
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

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
            return self._get(params).idle_cycle(
                tool_schemas=schemas,
                minimum_idle_seconds=minimum_idle,
            )
        except ValueError as error:
            raise RpcFault(-32602, str(error)) from error

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

        def progress(value: float, message: str) -> None:
            if job_id and job_id in self.cancelled_jobs:
                raise RpcFault(-32800, "job was cancelled")
            self.notify(
                "job-progress",
                brain_id=brain.brain_id,
                job_id=job_id,
                progress=value,
                message=message,
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

    def chat(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        brain = self._get(params)
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
            )
        except Exception:
            for record in inline_records:
                with self._inline_lock:
                    record.cancelled = True
                    if record.future is not None:
                        record.future.cancel()
            raise

        # A streamed Auto/Full imagination action always publishes at least one
        # real decoder preview before the chat RPC resolves. Longer generation
        # continues concurrently and becomes the same typed tool job/artifact.
        for record in inline_records:
            while not record.first_preview.wait(timeout=0.05):
                future = record.future
                if future is None or future.done():
                    break
        self.notify(
            "brain-mutated",
            brain_id=brain.brain_id,
            progress=1.0,
            message="Turn learned into fast synapses and slow parameters.",
            data={"traceId": result["trace"]["id"]},
        )
        return result

    def train(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        brain, job_id, progress = self._job(params, request_id, "training")
        texts = params.get("texts")
        if texts is None and isinstance(params.get("text"), str):
            texts = [params["text"]]
        if texts is not None and not (
            isinstance(texts, list)
            and all(isinstance(value, str) for value in texts)
        ):
            raise RpcFault(-32602, "params.texts must be a string array")
        result = brain.train(
            texts=texts,
            epochs=int(params.get("epochs", params.get("steps", 1))),
            learning_rate=(
                _number(params.get("learningRate"))
                if params.get("learningRate") is not None
                else None
            ),
            source_ids=params.get("sourceIds"),
            progress=progress,
        )
        self._job_complete(brain, job_id, "training", result)
        return result

    def ingest(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
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
                allow_replay=bool(params.get("allowReplay", False)),
                epoch=int(params.get("epoch", 0)),
                progress=progress,
            )
        except Exception:
            # Ingestion mutates fast weights and assemblies as records stream,
            # but the durable cursor is committed only after the whole worker
            # call succeeds. Drop that in-memory object and reload the last
            # atomic checkpoint so retrying the same manifest record cannot
            # inherit a partial application.
            brain_id = brain.brain_id
            storage = brain.storage_path
            previous = self.brains.pop(brain_id, None)
            if previous is not None:
                previous.events.close()
            restored = AdaptiveBrain.load(
                storage, expected_brain_id=brain_id
            )
            self.brains[brain_id] = restored
            self.notify(
                "ingestion-rolled-back",
                brain_id=brain_id,
                job_id=job_id,
                progress=0.0,
                message=(
                    "Partial ingestion was discarded; the last atomic "
                    "checkpoint was restored."
                ),
            )
            raise
        self._job_complete(brain, job_id, "ingestion", result)
        return result

    def consolidate(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain, job_id, progress = self._job(params, request_id, "consolidation")
        result = brain.consolidate(
            steps=int(params.get("steps", params.get("epochs", 4))),
            progress=progress,
        )
        self._job_complete(brain, job_id, "consolidation", result)
        return result

    def generate_modality(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        brain, job_id, progress = self._job(
            params, request_id, "modality-generation"
        )
        inline_result = self._claim_inline_generation(brain, params, job_id)
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
        ) -> None:
            nonlocal preview_revision
            bounded_progress = 0.2 + 0.75 * max(
                0.0, min(float(generation_progress), 1.0)
            )
            preview_value: Dict[str, Any] = {
                "revision": preview_revision,
                "progress": bounded_progress,
                "statusLabel": "Decoding the current neural latent",
                "mimeType": mime_type,
            }
            if len(payload) <= 12 * 1024 * 1024:
                preview_value["dataUrl"] = (
                    "data:%s;base64,%s"
                    % (
                        mime_type,
                        base64.b64encode(payload).decode("ascii"),
                    )
                )
            self.notify(
                "modality-preview",
                brain_id=brain.brain_id,
                job_id=job_id,
                sequence=preview_revision,
                progress=bounded_progress,
                message="Decoding the current neural latent",
                data={"preview": preview_value},
            )
            preview_revision += 1

        result = brain.generate_modality(
            modality=str(params.get("modality", "")),
            prompt=str(params.get("prompt", "")),
            concept_ids=params.get("conceptIds"),
            input_path=str(params.get("inputPath", "")),
            settings=(
                params.get("settings")
                if isinstance(params.get("settings"), dict)
                else {}
            ),
            seed=int(params["seed"]) if params.get("seed") is not None else None,
            preview_callback=preview,
        )
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
            previous.events.close()
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
            previous.events.close()
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
        return {
            "jobId": job_id,
            "cancelled": True,
            "inlineGenerationsCancelled": inline_cancelled,
        }

    def shutdown(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del params, request_id
        self.running = False
        self._shutdown_inline_generations()
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
        result = handler(params, str(request_id) if request_id is not None else None)
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


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    worker = Worker()
    while worker.running:
        raw = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
            Worker._send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": "request line exceeds protocol limit",
                    },
                }
            )
            break
        request_id: Any = None
        try:
            request = json.loads(raw.decode("utf-8"))
            if isinstance(request, dict):
                request_id = request.get("id")
            response = worker.dispatch(request)
            if response is not None:
                Worker._send(response)
        except json.JSONDecodeError as error:
            Worker._send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "parse error: %s" % error},
                }
            )
        except RpcFault as error:
            payload: Dict[str, Any] = {
                "code": error.code,
                "message": error.message,
            }
            if error.data is not None:
                payload["data"] = error.data
            Worker._send(
                {"jsonrpc": "2.0", "id": request_id, "error": payload}
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            Worker._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "%s: %s"
                        % (error.__class__.__name__, str(error)),
                    },
                }
            )
    for brain in worker.brains.values():
        try:
            brain.events.close()
        except Exception:
            pass
    worker._shutdown_inline_generations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
