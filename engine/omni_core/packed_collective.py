"""Canonical collective packed mutations from bounded transient derivatives.

Ranks stage derivatives, not privately mutated weights. One rank applies each
seeded ternary step, then broadcasts authoritative uint8 rows and stability.
"""

import hashlib
import json
import math
import sqlite3
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from .packed_collective_hooks import packed_derivative_sink
from .text_spool import DatasetResourcePause
from .parameter_diagnostics import packed_diagnostic_write


ROW_BLOCK = 32


class PackedCollectiveController:
    def __init__(self, owners, context, directory, apply_rows, reserve=None):
        self.owners = dict(owners)
        self.names = {id(owner): name for name, owner in self.owners.items()}
        if len(self.names) != len(self.owners):
            raise ValueError("packed collective inventory contains aliases")
        self.context = context
        self.apply_rows = apply_rows
        self.reserve = reserve or (lambda **_: None)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(str(self.directory / "derivatives.sqlite3"))
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=FULL")
        self.database.executescript('''
            CREATE TABLE IF NOT EXISTS derivatives (
              owner TEXT, role TEXT, row_start INTEGER, width INTEGER, rows INTEGER,
              rate REAL, scale REAL, strength REAL, gradient BLOB,
              PRIMARY KEY(owner, role, row_start));
            CREATE TABLE IF NOT EXISTS rollback (
              owner TEXT, role TEXT, row_start INTEGER, packed BLOB, stability BLOB,
              PRIMARY KEY(owner, role, row_start));
        ''')
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.commit()
        self.step_id = ""
        self.pending_events = {}

    def _scope(self, owner):
        return owner._packed_residency_scope(self.context.device) if hasattr(owner, "_packed_residency_scope") else nullcontext()

    def _buffer(self, owner, role):
        return getattr(owner, "_packed_forward_weight" if role == "weight" else "_packed_forward_bias", None)

    def _stability(self, owner, role):
        return getattr(owner, "_row_stability" if role == "weight" else "_bias_row_stability", None)

    def _objects(self, value):
        if not self.context.distributed:
            return [value]
        gathered = [None] * self.context.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def identity(self):
        digest = hashlib.sha256()
        for name, owner in sorted(self.owners.items()):
            with self._scope(owner):
                digest.update(name.encode("utf-8"))
                digest.update(json.dumps({"stabilityStrength": float(getattr(owner, "_packed_stability_strength", 0)),
                    "pendingStabilityEvents": int(getattr(owner, "_pending_stability_events", 0))}, sort_keys=True).encode("ascii"))
                for role in ("weight", "bias"):
                    packed = self._buffer(owner, role)
                    if packed is None:
                        continue
                    if packed.dtype != torch.uint8 or packed.ndim != 2:
                        raise ValueError("packed collective owner has no authoritative uint8 rows")
                    digest.update(json.dumps([role, list(packed.shape)]).encode("ascii"))
                    for start in range(0, packed.shape[0], ROW_BLOCK):
                        digest.update(packed[start:start + ROW_BLOCK].detach().cpu().contiguous().numpy().tobytes())
                    stability = self._stability(owner, role)
                    if stability is not None:
                        digest.update(json.dumps([str(stability.dtype), list(stability.shape)]).encode("ascii"))
                        for start in range(0, stability.numel(), ROW_BLOCK):
                            digest.update(stability[start:start + ROW_BLOCK].detach().cpu().contiguous().numpy().tobytes())
                scale = getattr(owner, "_packed_forward_scale", None)
                if scale is not None:
                    digest.update(scale.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _phase(self, operation):
        value, error = None, None
        try:
            value = operation()
        except BaseException as failure:
            message = "%s: %s" % (type(failure).__name__, str(failure)[:2_000])
            status = getattr(failure, "status", {})
            error = {"message": message, "status": status if isinstance(status, dict) else {},
                "resource": isinstance(failure, (DatasetResourcePause, MemoryError))
                    or isinstance(status, dict) and any(bool(status.get(key)) for key in ("recoverable", "paused", "memoryPressure", "diskPressure"))
                    or any(marker in message.lower() for marker in ("out of memory", "no space left", "disk is full", "cannot allocate memory"))}
        errors = self._objects(error)
        if any(errors):
            selected = next((item for item in errors if item and not item["resource"]), next(item for item in errors if item))
            message = "packed collective phase failed: %s" % selected["message"]
            if selected["resource"]:
                raise DatasetResourcePause(message, selected["status"])
            raise RuntimeError(message)
        return value

    def begin(self, step_id):
        if self.step_id:
            raise RuntimeError("packed collective step is already active")
        if len(set(self._objects(str(step_id)))) != 1:
            raise RuntimeError("packed ranks proposed different logical mutation transactions")
        identities = self._objects(self._phase(self.identity))
        if len(set(identities)) != 1:
            raise RuntimeError("ranks do not share one packed neural identity before forward")
        def clear_scratch():
            self.database.execute("DELETE FROM derivatives")
            self.database.execute("DELETE FROM rollback")
            self.database.commit()
        self._phase(clear_scratch)
        self.step_id = str(step_id)
        self.pending_events = {name: int(getattr(owner, "_pending_stability_events", 0)) for name, owner in self.owners.items()}

    def stage(self, owner, packed, width, row_start, gradient, rate, scale,
              row_stability, stability_strength):
        if not self.step_id or id(owner) not in self.names:
            raise RuntimeError("packed derivative owner is not in the active collective inventory")
        name = self.names[id(owner)]
        role = "weight" if packed is self._buffer(owner, "weight") else "bias" if packed is self._buffer(owner, "bias") else None
        if role is None or gradient.ndim != 2 or gradient.shape[1] != width:
            raise ValueError("packed derivative buffer ownership/geometry is invalid")
        if not math.isfinite(rate) or rate < 0 or not math.isfinite(stability_strength) or stability_strength < 0 or not bool(torch.isfinite(gradient).all()):
            raise ValueError("packed derivative is non-finite")
        scale_value = float(scale.detach().cpu().item()) if isinstance(scale, torch.Tensor) else float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise ValueError("packed derivative scale is invalid")
        end = row_start + int(gradient.shape[0])
        if row_start < 0 or end > packed.shape[0]:
            raise ValueError("packed derivative row range exceeds its owner")
        for block in range(row_start // ROW_BLOCK * ROW_BLOCK, end, ROW_BLOCK):
            rows = min(ROW_BLOCK, int(packed.shape[0]) - block)
            self.reserve(ram_bytes=rows * width * 8, disk_bytes=rows * width * 4)
            found = self.database.execute(
                "SELECT width,rows,rate,scale,strength,gradient FROM derivatives WHERE owner=? AND role=? AND row_start=?",
                (name, role, block)).fetchone()
            if found is None:
                accumulated = torch.zeros((rows, width), dtype=torch.float32)
            else:
                if tuple(found[:5]) != (width, rows, rate, scale_value, stability_strength):
                    raise ValueError("packed derivative geometry/rate changed within one step")
                accumulated = torch.frombuffer(bytearray(found[5]), dtype=torch.float32).reshape(rows, width)
            first, last = max(block, row_start), min(block + rows, end)
            accumulated[first - block:last - block].add_(gradient[first - row_start:last - row_start].detach().to(device="cpu", dtype=torch.float32))
            if not bool(torch.isfinite(accumulated).all()):
                raise ValueError("packed derivative accumulation overflowed")
            self.database.execute("INSERT OR REPLACE INTO derivatives VALUES (?,?,?,?,?,?,?,?,?)",
                (name, role, block, width, rows, rate, scale_value, stability_strength,
                 accumulated.contiguous().numpy().tobytes()))
        self.database.commit()
        return 0

    def commit(self, *, retain_rollback=False):
        if not self.step_id:
            raise RuntimeError("packed collective step is not active")
        last_key = None
        changed_owners = set()
        try:
            while True:
                if last_key is None:
                    row = self.database.execute("SELECT owner,role,row_start,width,rows,rate,scale,strength FROM derivatives ORDER BY owner,role,row_start LIMIT 1").fetchone()
                else:
                    row = self.database.execute("SELECT owner,role,row_start,width,rows,rate,scale,strength FROM derivatives WHERE (owner,role,row_start)>(?,?,?) ORDER BY owner,role,row_start LIMIT 1", last_key).fetchone()
                frontier = self._objects(list(row) if row is not None else None)
                available = [entry for entry in frontier if entry is not None]
                if not available:
                    break
                selected = min(available, key=lambda entry: tuple(entry[:3]))
                key = tuple(selected[:3])
                if any(entry[:3] == selected[:3] and entry != selected for entry in available):
                    raise ValueError("ranks emitted conflicting packed derivative specifications")
                name, role, start, width, rows, rate, scale_value, strength = selected
                owner = self.owners[name]
                device = self.context.device if self.context.backend == "nccl" else torch.device("cpu")
                def prepare_gradient():
                    self.reserve(ram_bytes=rows * width * 8, disk_bytes=rows * width)
                    local = self.database.execute("SELECT gradient FROM derivatives WHERE owner=? AND role=? AND row_start=?", key).fetchone()
                    value = torch.zeros((rows, width), dtype=torch.float32) if local is None else torch.frombuffer(bytearray(local[0]), dtype=torch.float32).reshape(rows, width)
                    return value.to(device)
                gradient = self._phase(prepare_gradient)
                if self.context.distributed:
                    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("collective packed derivative overflowed before mutation")
                with ExitStack() as scopes:
                    def prepare_owner():
                        scopes.enter_context(self._scope(owner))
                        packed = self._buffer(owner, role)
                        stability = self._stability(owner, role)
                        if packed is None or packed.dtype != torch.uint8 or packed.ndim != 2 or start + rows > packed.shape[0] or (width + 3) // 4 != packed.shape[1]:
                            raise ValueError("collective packed topology changed during backward")
                        if stability is not None and (stability.dtype != torch.uint8 or stability.ndim != 1 or stability.numel() != packed.shape[0]):
                            raise ValueError("collective row-resistance topology changed during backward")
                        self.reserve(ram_bytes=rows * packed.shape[1] * 3 + rows * 3,
                                     disk_bytes=rows * packed.shape[1] + rows + 4096)
                        self.database.execute("INSERT OR IGNORE INTO rollback VALUES (?,?,?,?,?)", (
                            name, role, start, packed[start:start + rows].detach().cpu().contiguous().numpy().tobytes(),
                            stability[start:start + rows].detach().cpu().numpy().tobytes() if stability is not None else None))
                        self.database.commit()
                        return packed, stability, packed[start:start + rows].to(device).clone(), (
                            stability[start:start + rows].to(device).clone() if stability is not None else None)
                    packed, stability, block, stable_block = self._phase(prepare_owner)
                    def canonical_update():
                        if self.context.is_rank_zero:
                            seed = int.from_bytes(hashlib.sha256((self.step_id + "\0" + json.dumps(key)).encode()).digest()[:8], "little") % (1 << 63)
                            generator = torch.Generator(device=packed.device).manual_seed(seed)
                            with packed_derivative_sink(None):
                                changed = self.apply_rows(packed, width, start, gradient.to(packed.device), rate,
                                    torch.tensor(scale_value, device=packed.device), generator,
                                    stability, strength)
                            block.copy_(packed[start:start + rows].to(device))
                            if stable_block is not None:
                                stable_block.copy_(stability[start:start + rows].to(device))
                            return int(changed)
                        return 0
                    changed = self._phase(canonical_update)
                    changed_values = self._objects(changed)
                    if self.context.distributed:
                        dist.broadcast(block, src=0)
                        if stability is not None:
                            dist.broadcast(stable_block, src=0)
                    def install_rows():
                        replacement = block.to(packed.device)
                        with packed_diagnostic_write(owner, packed, replacement, start=start * packed.shape[1]):
                            packed[start:start + rows].copy_(replacement)
                        if stability is not None:
                            stability[start:start + rows].copy_(stable_block.to(stability.device))
                        setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
                    self._phase(install_rows)
                    self._phase(scopes.close)
                if any(changed_values):
                    changed_owners.add(name)
                last_key = key
            if len(set(self._objects(self._phase(self.identity)))) != 1:
                raise RuntimeError("collective packed update did not preserve one shared identity")
            for name in changed_owners:
                owner = self.owners[name]
                if float(getattr(owner, "_packed_stability_strength", 0)) > 0:
                    owner._pending_stability_events += 1
            if not retain_rollback:
                self.finalize()
            return len(changed_owners)
        except BaseException:
            self.rollback()
            raise

    def rollback(self):
        for name, role, start, packed_bytes, stable_bytes in self.database.execute("SELECT owner,role,row_start,packed,stability FROM rollback ORDER BY owner,role,row_start"):
            owner = self.owners[name]
            with self._scope(owner):
                packed = self._buffer(owner, role)
                rows = len(packed_bytes) // int(packed.shape[1])
                replacement = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).reshape(rows, packed.shape[1]).to(packed.device)
                with packed_diagnostic_write(owner, packed, replacement, start=start * packed.shape[1]):
                    packed[start:start + rows].copy_(replacement)
                stability = self._stability(owner, role)
                if stable_bytes is not None:
                    stability[start:start + rows].copy_(torch.frombuffer(bytearray(stable_bytes), dtype=torch.uint8).to(stability.device))
                setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.commit()
        self.step_id = ""
        for name, count in self.pending_events.items():
            self.owners[name]._pending_stability_events = count
        self.pending_events = {}

    def finalize(self):
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.commit()
        self.step_id = ""
        self.pending_events = {}

    def close(self):
        self.database.close()
        # These are owned transient derivative/rollback journals, not neural
        # authority. DELETE alone retains free-page gradient bytes and can
        # otherwise pin a giant scratch allocation after every replica refresh.
        for name in ("derivatives.sqlite3", "derivatives.sqlite3-wal", "derivatives.sqlite3-shm"):
            try:
                (self.directory / name).unlink()
            except FileNotFoundError:
                pass
