"""Immutable, streamed retention anchors for isolated native evolution.

The wire format is ordinary safetensors. Neither saving nor evaluation stacks
the complete replay corpus or transfers it to the accelerator at once.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from .bounded_tensor_io import BoundedTensorFile, TRANSFER_BYTES


def _admit(policy: Any, ram: int, disk: int = 0) -> None:
    if policy is None:
        return
    status = policy.status(estimated_ram_bytes=max(0, ram), estimated_write_bytes=max(0, disk))
    if status.get("memoryPressure") or status.get("diskPressure"):
        raise RuntimeError("immutable evolution anchors paused at the live resource reserve")


class RetentionAnchorFile:
    def __init__(self, path: Path, *, expected_width: Optional[int] = None):
        self.path = Path(path)
        self.reader = BoundedTensorFile(self.path)
        spec = self.reader.specs.get("retention_anchors")
        if spec is None or len(self.reader.specs) != 1 or spec.dtype != torch.float32 or len(spec.shape) != 2:
            raise ValueError("candidate retention anchor geometry is invalid")
        if expected_width is not None and spec.shape[1] != expected_width:
            raise ValueError("candidate retention anchors changed idea geometry")
        self.shape = spec.shape

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def batches(self, *, max_rows: int = 32, byte_budget: int = TRANSFER_BYTES, policy: Any = None):
        rows, width = self.shape
        row_bytes = width * 4
        if row_bytes < 1 or row_bytes > byte_budget:
            raise RuntimeError("one retention anchor exceeds the live bounded transfer reservation")
        count = max(1, min(max_rows, byte_budget // row_bytes))
        spec = self.reader.specs["retention_anchors"]
        with self.path.open("rb") as handle:
            handle.seek(spec.offset)
            for start in range(0, rows, count):
                size = min(count, rows - start) * row_bytes
                _admit(policy, size * 32)
                payload = bytearray(size)
                if handle.readinto(payload) != size:
                    raise ValueError("immutable retention anchors were truncated")
                yield torch.frombuffer(payload, dtype=torch.float32).reshape(-1, width)


def save_retention_anchors(
    path: Path, rows: Iterable[torch.Tensor], *, count: int, width: int,
    candidate_id: str, policy: Any = None,
) -> RetentionAnchorFile:
    if sys.byteorder != "little":
        raise RuntimeError("native retention anchors require a little-endian host")
    if count < 0 or width < 1:
        raise ValueError("invalid retention anchor dimensions")
    row_bytes = width * 4
    _admit(policy, row_bytes * 3, count * row_bytes + 4096)
    header = {
        "retention_anchors": {"dtype": "F32", "shape": [count, width], "data_offsets": [0, count * row_bytes]},
        "__metadata__": {"format": "omni-evolution-evaluation-anchors", "candidate_id": candidate_id},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            written = 0
            for row in rows:
                if written >= count or row.numel() != width:
                    raise ValueError("retention anchor count/width changed during snapshot")
                _admit(policy, row_bytes * 3)
                block = row.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
                if not bool(torch.isfinite(block).all()):
                    raise ValueError("retention anchor contains a nonfinite value")
                handle.write(memoryview(block.view(torch.uint8).numpy()))
                written += 1
            if written != count:
                raise ValueError("retention anchor count changed during snapshot")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return RetentionAnchorFile(path, expected_width=width)
