"""Closed runtime/kernel/hardware seal for exact stochastic continuation.

Import/export may preserve this declaration on another machine. Executing its
frontier requires an identical supported runtime, not merely a device string.
No automatic conversion, reseeding or speculative cache replay is performed.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@lru_cache(maxsize=96)
def _file_seal(path, size, mtime_ns, ctime_ns, inode):
    del size, mtime_ns, ctime_ns, inode
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _digest(path):
    info = path.stat()
    return _file_seal(str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)


def _hardware(device):
    body = {"machine": platform.machine(), "processor": platform.processor(),
        "cpuCapability": torch.backends.cpu.get_cpu_capability()}
    if sys.platform == "darwin":
        # Fixed read-only keys only; no shell or user-provided command strings.
        for key in ("hw.model", "machdep.cpu.brand_string", "hw.optional.arm64"):
            result = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, timeout=2, check=False)
            body[key] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    elif sys.platform.startswith("linux"):
        cpu = Path("/proc/cpuinfo")
        if cpu.is_file():
            with cpu.open(encoding="utf-8") as stream:
                for line in stream:
                    name, separator, value = line.partition(":")
                    if separator and name.strip() in {"vendor_id", "model name", "cpu family", "model", "stepping", "flags", "Features", "CPU implementer", "CPU architecture", "CPU part", "CPU revision"}:
                        body.setdefault(name.strip(), value.strip())
                    if not line.strip() and len(body) > 3:
                        break
    elif sys.platform == "win32":
        body["processorIdentifier"] = os.environ.get("PROCESSOR_IDENTIFIER", "unavailable")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        body["accelerator"] = {key: str(getattr(properties, key, "unavailable")) for key in
            ("name", "major", "minor", "total_memory", "multi_processor_count", "uuid")}
    elif device.type not in {"cpu", "mps"}:
        raise ValueError("exact continuation has no hardware compatibility seal for this backend")
    return body


def capture_execution_compatibility(device):
    device = torch.device(device)
    directory = Path(__file__).parent
    source_names = ("model.py", "brain.py", "connected_regions.py", "cooperative_generation.py", "generation_cursor.py", "activity_clock.py", "working_activity.py", "offload.py", "execution_rng.py",
        "packed_connections.py", "packed_native_kernel.py", "kernels/weighted_ternary.cpp", "kernels/weighted_ternary.metal")
    files = {name: _digest(directory / name) for name in source_names}
    native = directory / "native"
    if native.is_dir():
        for path in sorted(native.iterdir()):
            if path.is_file() and path.suffix in {".so", ".dylib", ".dll", ".json"}:
                files["native/" + path.name] = _digest(path)
    torch_root = Path(torch.__file__).parent
    torch_files = {"_C": _digest(Path(torch._C.__file__))}
    library_directory = torch_root / "lib"
    if library_directory.is_dir():
        for path in sorted(library_directory.iterdir()):
            if path.is_file() and path.suffix in {".so", ".dylib", ".dll"}:
                torch_files[path.name] = _digest(path)
    body = {"format": "omni-exact-runtime-compatibility", "version": 1, "backend": str(device),
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(), "byteorder": sys.byteorder},
        "system": {"name": platform.system(), "release": platform.release(), "hardware": _hardware(device)},
        "torch": {"version": str(torch.__version__), "gitVersion": torch.version.git_version,
            "cuda": torch.version.cuda, "build": hashlib.sha256(torch.__config__.show().encode()).hexdigest(), "libraries": torch_files},
        "kernels": files, "threads": {"intraOp": torch.get_num_threads(), "interOp": torch.get_num_interop_threads()},
        "determinism": {"algorithms": torch.are_deterministic_algorithms_enabled(),
            "warnOnly": torch.is_deterministic_algorithms_warn_only_enabled(), "cudnnBenchmark": torch.backends.cudnn.benchmark,
            "cudnnDeterministic": torch.backends.cudnn.deterministic, "matmulTf32": torch.backends.cuda.matmul.allow_tf32}}
    return {"declaration": body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_execution_compatibility(value):
    if (type(value) is not dict or set(value) != {"declaration", "sha256"} or type(value["declaration"]) is not dict
            or value["declaration"].get("format") != "omni-exact-runtime-compatibility" or value["declaration"].get("version") != 1
            or value["sha256"] != hashlib.sha256(_canonical(value["declaration"])).hexdigest()):
        raise ValueError("saved exact runtime compatibility seal is invalid")
    return value


def require_execution_compatibility(value, device):
    if value is None:
        raise ValueError("this saved frontier predates the exact runtime compatibility seal; exact restore is unsupported")
    validate_execution_compatibility(value)
    if value != capture_execution_compatibility(device):
        raise ValueError("exact restore is unsupported on this changed kernel, Python/PyTorch build, device, hardware or threading runtime")


def execution_compatibility_status(manifest):
    """Read-only disclosure; historical activity can be preserved but not guessed."""
    try:
        require_execution_compatibility(manifest.get("runtimeCompatibility"), manifest["backend"])
        payload = manifest.get("payload")
        fields = dict(payload.get("entries", [])) if isinstance(payload, dict) and payload.get("kind") == "map" else {}
        if "activityClock" not in fields:
            raise ValueError("this saved frontier has no sealed paused activity clock; exact restore is unsupported")
        return {"compatible": True}
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        return {"compatible": False, "reason": str(error)}
