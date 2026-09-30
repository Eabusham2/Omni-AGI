"""Main-authenticated standalone video-runtime configuration; no setup execution.

This module intentionally imports no neural runtime. Stdio is a trusted main to
worker boundary; there is no renderer RPC forwarding or remote manifest input.
The main process owns source/license review and verifies all pinned payloads.
The worker additionally checks its fixed cache root, receipt and binary before
changing imageio's executable, without replacing a warm brain or spawning code.
"""

import hashlib
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional


class VideoRuntimeConfigurationCancelled(RuntimeError):
    pass


def _native_header(stream: Any, target: str) -> None:
    header = stream.read(64)
    if len(header) < 20:
        raise ValueError("FFmpeg native executable header is truncated")
    if target.startswith("linux-"):
        machine = 62 if target.endswith("-x64") else 183
        if (
            header[:4] != b"\x7fELF"
            or header[4:6] != b"\x02\x01"
            or struct.unpack_from("<H", header, 18)[0] != machine
        ):
            raise ValueError("FFmpeg is not the reviewed native ELF64 target")
    elif target.startswith("darwin-"):
        machine = 0x01000007 if target.endswith("-x64") else 0x0100000C
        if struct.unpack_from("<II", header)[0:2] != (0xFEEDFACF, machine):
            raise ValueError("FFmpeg is not the reviewed native Mach-O target")
    else:
        if len(header) < 64 or header[:2] != b"MZ":
            raise ValueError("FFmpeg is not a native PE executable")
        offset = struct.unpack_from("<I", header, 60)[0]
        if offset < 64 or offset > 1024 * 1024:
            raise ValueError("FFmpeg PE offset is invalid")
        stream.seek(offset)
        pe = stream.read(6)
        machine = 0x8664 if target.endswith("-x64") else 0xAA64
        if len(pe) != 6 or pe[:4] != b"PE\x00\x00" or struct.unpack_from("<H", pe, 4)[0] != machine:
            raise ValueError("FFmpeg is not the reviewed native PE target")


def configure_verified_video_runtime(
    params: Mapping[str, Any],
    *,
    environment: Optional[MutableMapping[str, str]] = None,
    platform_name: Optional[str] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Reverify only a main-provisioned binary and atomically select its path.

    Sources and build/install archives remain ordinary data; no downloaded
    script, installer, shell command or package manager is run here.
    """
    env = os.environ if environment is None else environment
    platform_name = sys.platform if platform_name is None else platform_name
    if set(params) != {"executablePath", "artifactSha256", "binarySha256", "binarySizeBytes", "target"}:
        raise ValueError("Video runtime configuration contains unsupported fields")
    cache = env.get("OMNI_VIDEO_RUNTIME_CACHE_ROOT", "")
    path_value = params.get("executablePath")
    if not cache or not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("A fixed cache root and absolute video runtime path are required")
    path = Path(path_value)
    root = Path(cache).resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or path.parent.is_symlink() or path != resolved or root not in resolved.parents:
        raise ValueError("Video runtime must be a regular file inside the configured cache")
    artifact_hash = params.get("artifactSha256")
    binary_hash = params.get("binarySha256")
    if not all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in (artifact_hash, binary_hash)):
        raise ValueError("Video runtime requires exact artifact and binary SHA-256 pins")
    if not path.parent.name.endswith("-" + artifact_hash) or path.parent.parent != root:
        raise ValueError("Video runtime path does not match its main-verified artifact directory")
    target = params.get("target")
    if not isinstance(target, str) or target not in {"win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "linux-x64", "linux-arm64"} or not target.startswith(platform_name + "-"):
        raise ValueError("Video runtime target does not match this operating system")
    expected_name = "ffmpeg.exe" if target.startswith("win32-") else "ffmpeg"
    metadata = path.lstat()
    expected_size = params.get("binarySizeBytes")
    if (
        path.name != expected_name
        or not stat.S_ISREG(metadata.st_mode)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 1 <= expected_size <= 1024 ** 3
        or metadata.st_size != expected_size
        or (platform_name != "win32" and not os.access(path, os.X_OK))
    ):
        raise ValueError("Video runtime file is not the expected accessible native executable")
    receipt_path = path.parent / "receipt.json"
    receipt_metadata = receipt_path.lstat()
    if not stat.S_ISREG(receipt_metadata.st_mode) or receipt_metadata.st_size > 512 * 1024:
        raise ValueError("Video runtime receipt is missing or unsafe")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(receipt.get("artifact"), dict) or not isinstance(receipt["artifact"].get("binary"), dict):
        raise ValueError("Video runtime receipt shape is invalid")
    binary = receipt["artifact"]["binary"]
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("artifactSha256") != artifact_hash
        or binary.get("sha256") != binary_hash
        or binary.get("sizeBytes") != expected_size
        or receipt.get("artifact", {}).get("target") != target
    ):
        raise ValueError("Video runtime receipt does not match the main-verified request")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        _native_header(stream, target)
        stream.seek(0)
        while True:
            if cancelled is not None and cancelled():
                raise VideoRuntimeConfigurationCancelled("Video runtime configuration was cancelled")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != binary_hash:
        raise ValueError("Video runtime binary no longer matches its reviewed SHA-256")
    if cancelled is not None and cancelled():
        raise VideoRuntimeConfigurationCancelled("Video runtime configuration was cancelled")
    env["IMAGEIO_FFMPEG_EXE"] = str(path)
    return {"configured": True, "artifactSha256": artifact_hash, "target": target}
