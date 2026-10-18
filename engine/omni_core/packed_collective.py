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

from .packed_collective_hooks import packed_derivative_sink, packed_row_owner
from .text_spool import DatasetResourcePause
from .parameter_diagnostics import packed_diagnostic_write
from .packed_collective_tiles import PackedColumnCollective
from .native_core_paging import release_native_tensor_chunk


ROW_BLOCK = 32
CONTACT_BATCH = 128


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
            CREATE TABLE IF NOT EXISTS parallel_rollback (
              owner TEXT, role TEXT, row_start INTEGER, sequence INTEGER, payload BLOB,
              PRIMARY KEY(owner,role,row_start,sequence)) WITHOUT ROWID;
        ''')
        if "connections" not in {row[1] for row in self.database.execute("PRAGMA table_info(rollback)")}:
            self.database.execute("ALTER TABLE rollback ADD COLUMN connections BLOB")
        if "parallel" not in {row[1] for row in self.database.execute("PRAGMA table_info(rollback)")}:
            self.database.execute("ALTER TABLE rollback ADD COLUMN parallel BLOB")
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.execute("DELETE FROM parallel_rollback")
        self.column_tiles = PackedColumnCollective(self)
        self.database.commit()
        self.step_id = ""
        self.pending_events = {}
        self.pending_topology = {}

    def _scope(self, owner):
        return owner._packed_residency_scope(self.context.device) if hasattr(owner, "_packed_residency_scope") else nullcontext()

    def _buffer(self, owner, role):
        return getattr(owner, "_packed_forward_weight" if role == "weight" else "_packed_forward_bias", None)

    def _stability(self, owner, role):
        return getattr(owner, "_row_stability" if role == "weight" else "_bias_row_stability", None)

    def _connections(self, owner, role):
        return getattr(owner, "_connection_state" if role == "weight" else "_bias_connection_state", None)

    def _objects(self, value):
        if not self.context.distributed:
            return [value]
        gathered = [None] * self.context.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def identity(self):
        digest = hashlib.sha256()
        def hash_tensor(value):
            flat = value.detach().reshape(-1)
            step = max(1, 4096 // value.element_size())
            for start in range(0, flat.numel(), step):
                count = min(step, flat.numel() - start)
                self.reserve(ram_bytes=count * value.element_size() * 2, disk_bytes=0)
                try:
                    digest.update(flat[start:start + count].cpu().contiguous().numpy().tobytes())
                finally:
                    release_native_tensor_chunk(value, start * value.element_size(), count * value.element_size())
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
                    hash_tensor(packed)
                    stability = self._stability(owner, role)
                    if stability is not None:
                        digest.update(json.dumps([str(stability.dtype), list(stability.shape)]).encode("ascii"))
                        hash_tensor(stability)
                    connections = self._connections(owner, role)
                    if connections is not None:
                        digest.update(json.dumps(["connections", str(connections.dtype), list(connections.shape)]).encode("ascii"))
                        hash_tensor(connections)
                scale = getattr(owner, "_packed_forward_scale", None)
                pool = getattr(owner, "parallel_connections", None)
                structure = getattr(pool, "iter_identity_structure", None)
                if callable(structure):
                    for item in structure():
                        digest.update(json.dumps(item, separators=(",", ":")).encode("ascii"))
                parallel = getattr(owner, "iter_parallel_connection_records", None)
                if callable(parallel):
                    for bias in (False, True):
                        digest.update(b"bias-parallel" if bias else b"weight-parallel")
                        for record in parallel(bias=bias):
                            digest.update(json.dumps(record, separators=(",", ":")).encode("ascii"))
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

    def _contact_batches(self, owner, role, start, rows):
        iterator = getattr(owner, "iter_parallel_connection_records", None)
        if not callable(iterator):
            return
        batch = []
        for record in iterator(bias=role == "bias", start=start, end=start + rows):
            batch.append(record)
            if len(batch) == CONTACT_BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    def _snapshot_contacts(self, owner, name, role, start, rows):
        self.reserve(ram_bytes=CONTACT_BATCH * 1024, disk_bytes=0)
        for sequence, batch in enumerate(self._contact_batches(owner, role, start, rows)):
            payload = json.dumps(batch, separators=(",", ":")).encode("ascii")
            self.reserve(ram_bytes=0, disk_bytes=len(payload) + 8192)
            self.database.execute("INSERT OR IGNORE INTO parallel_rollback VALUES (?,?,?,?,?)",
                (name, role, start, sequence, payload))
        self.database.commit()

    def _install_contact_stream(self, owner, role, start, rows):
        # Only rank zero walks canonical state. Each broadcast is bounded,
        # including for millions of repeated contacts on one output row.
        stream = iter(self._contact_batches(owner, role, start, rows)) if self.context.is_rank_zero else None
        self._phase(lambda: self.reserve(ram_bytes=CONTACT_BATCH * 1024 * (2 + self.context.world_size), disk_bytes=0))
        while True:
            batch = self._phase(lambda: next(stream, None) if stream is not None else None)
            shared = self._objects(batch)[0]
            if shared is None:
                break
            if not self.context.is_rank_zero:
                self._phase(lambda: owner.install_parallel_connection_records_(shared,
                    bias=role == "bias", start=start, end=start + rows))
            else:
                self._phase(lambda: None)

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
            self.database.execute("DELETE FROM parallel_rollback")
            self.column_tiles.clear()
            self.database.commit()
        self._phase(clear_scratch)
        self.step_id = str(step_id)
        self.pending_events = {name: int(getattr(owner, "_pending_stability_events", 0)) for name, owner in self.owners.items()}
        self.pending_topology = {name: owner.capture_packed_topology_boundary() for name, owner in self.owners.items()
            if callable(getattr(owner, "capture_packed_topology_boundary", None))}

    def stage(self, owner, packed, width, row_start, gradient, rate, scale,
              row_stability, stability_strength):
        if getattr(owner, "_packed_column_learning", False):
            return self.stage_tile(owner, packed, width, row_start, 0, gradient, rate, scale, row_stability, stability_strength)
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

    def stage_tile(self, owner, packed, width, row_start, column_start, gradient, rate, scale,
                   row_stability=None, stability_strength=0.):
        return self.column_tiles.stage(owner, packed, width, row_start, column_start, gradient, rate, scale,
                                       row_stability, stability_strength)

    def commit(self, *, retain_rollback=False):
        if not self.step_id:
            raise RuntimeError("packed collective step is not active")
        last_key = None
        changed_owners = set()
        try:
            changed_owners.update(self.column_tiles.commit())
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
                        connections = self._connections(owner, role)
                        if packed is None or packed.dtype != torch.uint8 or packed.ndim != 2 or start + rows > packed.shape[0] or (width + 3) // 4 != packed.shape[1]:
                            raise ValueError("collective packed topology changed during backward")
                        if stability is not None and (stability.dtype != torch.uint8 or stability.ndim != 1 or stability.numel() != packed.shape[0]):
                            raise ValueError("collective row-resistance topology changed during backward")
                        if connections is not None and (connections.dtype != torch.uint8 or tuple(connections.shape) != (packed.shape[0], width)):
                            raise ValueError("collective connection state topology changed during backward")
                        self.reserve(ram_bytes=rows * (packed.shape[1] + (width if connections is not None else 0)) * 3 + rows * 3,
                                     disk_bytes=rows * (packed.shape[1] + (width if connections is not None else 0)) + rows + 4096)
                        self.database.execute("INSERT OR IGNORE INTO rollback VALUES (?,?,?,?,?,?,?)", (
                            name, role, start, packed[start:start + rows].detach().cpu().contiguous().numpy().tobytes(),
                            stability[start:start + rows].detach().cpu().numpy().tobytes() if stability is not None else None,
                            connections[start:start + rows].detach().cpu().contiguous().numpy().tobytes() if connections is not None else None,
                            None))
                        self.database.commit()
                        self._snapshot_contacts(owner, name, role, start, rows)
                        return packed, stability, connections, packed[start:start + rows].to(device).clone(), (
                            stability[start:start + rows].to(device).clone() if stability is not None else None), (
                            connections[start:start + rows].to(device).clone() if connections is not None else None)
                    packed, stability, connections, block, stable_block, connection_block = self._phase(prepare_owner)
                    def canonical_update():
                        if self.context.is_rank_zero:
                            seed = int.from_bytes(hashlib.sha256((self.step_id + "\0" + json.dumps(key)).encode()).digest()[:8], "little") % (1 << 63)
                            generator = torch.Generator(device=packed.device).manual_seed(seed)
                            with packed_derivative_sink(None), packed_row_owner(owner):
                                changed = self.apply_rows(packed, width, start, gradient.to(packed.device), rate,
                                    torch.tensor(scale_value, device=packed.device), generator,
                                    stability, strength)
                            block.copy_(packed[start:start + rows].to(device))
                            if stable_block is not None:
                                stable_block.copy_(stability[start:start + rows].to(device))
                            if connection_block is not None:
                                connection_block.copy_(connections[start:start + rows].to(device))
                            return int(changed)
                        return 0
                    changed = self._phase(canonical_update)
                    changed_values = self._objects(changed)
                    if self.context.distributed:
                        dist.broadcast(block, src=0)
                        if stability is not None:
                            dist.broadcast(stable_block, src=0)
                        if connections is not None:
                            dist.broadcast(connection_block, src=0)
                    def install_rows():
                        replacement = block.to(packed.device)
                        with packed_diagnostic_write(owner, packed, replacement, start=start * packed.shape[1]):
                            packed[start:start + rows].copy_(replacement)
                        if stability is not None:
                            stability[start:start + rows].copy_(stable_block.to(stability.device))
                        if connections is not None:
                            replacement = connection_block.to(connections.device)
                            with packed_diagnostic_write(owner, connections, replacement, start=start * width):
                                connections[start:start + rows].copy_(replacement)
                        setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
                    self._phase(install_rows)
                    self._install_contact_stream(owner, role, start, rows)
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
        self.column_tiles.rollback()
        for name, role, start, packed_bytes, stable_bytes, connection_bytes, parallel_bytes in self.database.execute("SELECT owner,role,row_start,packed,stability,connections,parallel FROM rollback ORDER BY owner,role,row_start"):
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
                if connection_bytes is not None:
                    connections = self._connections(owner, role)
                    replacement = torch.frombuffer(bytearray(connection_bytes), dtype=torch.uint8).reshape(rows, connections.shape[1]).to(connections.device)
                    with packed_diagnostic_write(owner, connections, replacement, start=start * connections.shape[1]):
                        connections[start:start + rows].copy_(replacement)
                if parallel_bytes is not None:
                    owner.restore_parallel_connections_(json.loads(parallel_bytes), bias=role == "bias", start=start, end=start + rows)
                for (payload,) in self.database.execute("SELECT payload FROM parallel_rollback WHERE owner=? AND role=? AND row_start=? ORDER BY sequence", (name, role, start)):
                    owner.install_parallel_connection_records_(json.loads(payload), bias=role == "bias", start=start, end=start + rows)
                setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
        for owner, boundary in self.pending_topology.values():
            owner.restore_topology_boundary(boundary)
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.execute("DELETE FROM parallel_rollback")
        self.column_tiles.clear()
        self.database.commit()
        self.step_id = ""
        for name, count in self.pending_events.items():
            self.owners[name]._pending_stability_events = count
        self.pending_events = {}
        self.pending_topology = {}

    def finalize(self):
        self.database.execute("DELETE FROM derivatives")
        self.database.execute("DELETE FROM rollback")
        self.database.execute("DELETE FROM parallel_rollback")
        self.column_tiles.clear()
        self.database.commit()
        self.step_id = ""
        self.pending_events = {}
        self.pending_topology = {}

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
