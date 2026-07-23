#!/usr/bin/env python3
"""OmniCortex JSON-RPC 2.0 stdio worker.

Stdout is protocol-only.  Diagnostics and tracebacks go to stderr so Electron
can safely parse one JSON response/notification per line.
"""

import json
import os
import platform
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional


WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import torch

from omni_core import AdaptiveBrain, OmniConfig, __version__


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 32 * 1024 * 1024


class RpcFault(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


class Worker:
    def __init__(self):
        self.brains: Dict[str, AdaptiveBrain] = {}
        self.cancelled_jobs = set()
        self.running = True
        self.methods: Dict[str, Callable[[Dict[str, Any], Optional[str]], Any]] = {
            "health": self.health,
            "create": self.create,
            "load": self.load,
            "reload": self.reload,
            "unload": self.unload,
            "restore_snapshot": self.restore_snapshot,
            "update_config": self.update_config,
            "merge_overlay": self.merge_overlay,
            "install_modality_pack": self.install_modality_pack,
            "list": self.list_brains,
            "state": self.state,
            "export_state": self.state,
            "chat": self.chat,
            "train": self.train,
            "ingest": self.ingest,
            "consolidate": self.consolidate,
            "generate_modality": self.generate_modality,
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
        sys.stdout.write(serialized + "\n")
        sys.stdout.flush()

    def notify(
        self,
        event_type: str,
        brain_id: str = "",
        job_id: str = "",
        progress: Optional[float] = None,
        message: str = "",
        data: Any = None,
    ) -> None:
        params: Dict[str, Any] = {"type": event_type}
        if brain_id:
            params["brainId"] = brain_id
        if job_id:
            params["jobId"] = job_id
        if progress is not None:
            params["progress"] = max(0.0, min(float(progress), 1.0))
        if message:
            params["message"] = message
        if data is not None:
            params["data"] = data
        self._send({"jsonrpc": "2.0", "method": "event", "params": params})

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
            message="Randomly initialized OmniCortex brain created.",
        )
        return brain.summary()

    def load(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain = self._get(params)
        return brain.summary()

    def reload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
        previous = self.brains.pop(brain_id, None)
        if previous is not None:
            previous.events.close()
        brain = AdaptiveBrain.load(
            self._storage(params, brain_id), expected_brain_id=brain_id
        )
        self.brains[brain_id] = brain
        return brain.summary()

    def unload(self, params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        del request_id
        brain_id = self._brain_id(params)
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
            for filename in (
                "core.safetensors",
                "plasticity.safetensors",
                "brain.json",
            ):
                source = snapshot / filename
                temporary = engine / (filename + ".restore.tmp")
                shutil.copy2(str(source), str(temporary))
                os.replace(str(temporary), str(engine / filename))
        return self.reload(params, request_id)

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
        return target.merge_overlay(source)

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
        result = brain.chat(
            value,
            max_new_tokens=int(params.get("maxNewTokens", 48)),
            seed=(
                int(params["seed"])
                if params.get("seed") is not None
                else None
            ),
            tool_schemas=tool_schemas,
        )
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
        result = brain.ingest(
            path=str(params["path"]) if params.get("path") else None,
            text=str(params["text"]) if params.get("text") is not None else None,
            name=str(params.get("name", "")),
            kind=str(params.get("kind", "")),
            policy=str(params.get("policy", "encode")),
            expected_hash=str(
                params.get("contentHash", params.get("expectedSha256", ""))
            ),
            progress=progress,
        )
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
        progress(0.2, "Activating internal idea vectors")
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
        )
        self._job_complete(brain, job_id, "modality-generation", result)
        return result

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
        return {"jobId": job_id, "cancelled": True}

    def shutdown(
        self, params: Dict[str, Any], request_id: Optional[str]
    ) -> Dict[str, Any]:
        del params, request_id
        self.running = False
        return {"stopping": True}

    def dispatch(self, request: Any) -> Optional[Dict[str, Any]]:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
