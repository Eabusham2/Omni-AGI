"""Bounded column-aware collective derivatives for native modality tables.

Legacy projection row transactions remain separate. Table derivatives use a
canonical one-dimension/64-code journal partition, regardless of local sample
or compute tiles, so ranks with different batching still sum before one step.
"""
import hashlib
import json
import math
from contextlib import ExitStack

import torch
import torch.distributed as dist

from .packed_gradient_tiles import apply_packed_gradient_tile, finish_row_resistance, release_gradient_tile
from .parameter_diagnostics import packed_diagnostic_write


COLUMN_BLOCK = 64


class PackedColumnCollective:
    def __init__(self, controller):
        self.controller = controller
        self.database = controller.database
        self.database.executescript('''
            CREATE TABLE IF NOT EXISTS column_derivatives (
              owner TEXT, role TEXT, row_start INTEGER, column_start INTEGER,
              width INTEGER, columns INTEGER, rate REAL, scale REAL, strength REAL, gradient BLOB,
              PRIMARY KEY(owner,role,row_start,column_start)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS column_rollback (
              owner TEXT, role TEXT, row_start INTEGER, column_start INTEGER,
              columns INTEGER, packed BLOB, connections BLOB,
              PRIMARY KEY(owner,role,row_start,column_start)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS column_rows (
              owner TEXT, role TEXT, row_start INTEGER, stability BLOB, grew INTEGER,
              PRIMARY KEY(owner,role,row_start)) WITHOUT ROWID;
        ''')
        self.clear()

    def clear(self):
        for table in ("column_derivatives", "column_rollback", "column_rows"):
            self.database.execute("DELETE FROM " + table)

    def stage(self, owner, packed, width, row_start, column_start, gradient, rate, scale,
              row_stability, stability_strength):
        c = self.controller
        if not c.step_id or id(owner) not in c.names:
            raise RuntimeError("packed derivative owner is not in the active collective inventory")
        role = "weight" if packed is c._buffer(owner, "weight") else "bias" if packed is c._buffer(owner, "bias") else None
        if (role is None or gradient.ndim != 2 or column_start < 0 or column_start % 4 or row_start < 0
                or column_start + gradient.shape[1] > width or row_start + gradient.shape[0] > packed.shape[0]
                or (width + 3) // 4 != packed.shape[1]
                or not bool(torch.isfinite(gradient).all())):
            raise ValueError("packed derivative tile ownership/geometry/values are invalid")
        scale_value = float(scale)
        if not all(math.isfinite(value) for value in (rate, scale_value, stability_strength)) or rate < 0 or scale_value <= 0 or stability_strength < 0:
            raise ValueError("packed derivative tile rate/scale is invalid")
        name = c.names[id(owner)]
        end = column_start + gradient.shape[1]
        for row in range(row_start, row_start + gradient.shape[0]):
            for column in range(column_start // COLUMN_BLOCK * COLUMN_BLOCK, end, COLUMN_BLOCK):
                columns = min(COLUMN_BLOCK, width - column)
                first, last = max(column, column_start), min(column + columns, end)
                c.reserve(ram_bytes=columns * 16 + 4096, disk_bytes=columns * 4 + 4096)
                key = (name, role, row, column)
                found = self.database.execute("SELECT width,columns,rate,scale,strength,gradient FROM column_derivatives WHERE owner=? AND role=? AND row_start=? AND column_start=?", key).fetchone()
                if found is None:
                    accumulated = torch.zeros((1, columns), dtype=torch.float32)
                else:
                    if tuple(found[:5]) != (width, columns, rate, scale_value, stability_strength):
                        raise ValueError("packed derivative tile specification changed within one step")
                    accumulated = torch.frombuffer(bytearray(found[5]), dtype=torch.float32).reshape(1, columns)
                accumulated[0, first-column:last-column].add_(gradient[row-row_start, first-column_start:last-column_start].detach().to(device="cpu", dtype=torch.float32))
                if not bool(torch.isfinite(accumulated).all()):
                    raise ValueError("packed derivative tile accumulation overflowed")
                self.database.execute("INSERT OR REPLACE INTO column_derivatives VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (*key, width, columns, rate, scale_value, stability_strength, accumulated.numpy().tobytes()))
        self.database.commit()
        return 0

    def commit(self):
        c = self.controller
        last_key = None
        changed_owners = set()
        while True:
            if last_key is None:
                row = self.database.execute("SELECT owner,role,row_start,column_start,width,columns,rate,scale,strength FROM column_derivatives ORDER BY owner,role,row_start,column_start LIMIT 1").fetchone()
            else:
                row = self.database.execute("SELECT owner,role,row_start,column_start,width,columns,rate,scale,strength FROM column_derivatives WHERE (owner,role,row_start,column_start)>(?,?,?,?) ORDER BY owner,role,row_start,column_start LIMIT 1", last_key).fetchone()
            available = [entry for entry in c._objects(list(row) if row is not None else None) if entry is not None]
            if not available:
                break
            selected = min(available, key=lambda entry: tuple(entry[:4]))
            key = tuple(selected[:4])
            if any(entry[:4] == selected[:4] and entry != selected for entry in available):
                raise ValueError("ranks emitted conflicting packed derivative tile specifications")
            name, role, start, column, width, columns, rate, scale, strength = selected
            owner = c.owners[name]
            device = c.context.device if c.context.backend == "nccl" else torch.device("cpu")

            def prepare_gradient():
                c.reserve(ram_bytes=columns * 16 + 4096, disk_bytes=0)
                found = self.database.execute("SELECT gradient FROM column_derivatives WHERE owner=? AND role=? AND row_start=? AND column_start=?", key).fetchone()
                value = torch.zeros((1, columns), dtype=torch.float32) if found is None else torch.frombuffer(bytearray(found[0]), dtype=torch.float32).reshape(1, columns)
                return value.to(device)

            gradient = c._phase(prepare_gradient)
            if c.context.distributed:
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("collective packed derivative tile overflowed before mutation")
            # Opposite rank contributions may cancel exactly. No private
            # update or normalization may precede this global sum.
            if bool(gradient.ne(0).any()):
                with ExitStack() as scopes:
                    c._phase(lambda: scopes.enter_context(c._scope(owner)))
                    packed, connections, stability = c._buffer(owner, role), c._connections(owner, role), c._stability(owner, role)
                    first, last = column // 4, (column + columns + 3) // 4

                    def snapshot():
                        if (connections is None or connections.dtype != torch.uint8 or tuple(connections.shape) != (packed.shape[0], width)
                                or packed.dtype != torch.uint8 or (width + 3) // 4 != packed.shape[1]):
                            raise ValueError("collective packed tile topology changed during backward")
                        c.reserve(ram_bytes=columns * 96 + 4096, disk_bytes=columns * 2 + 8192)
                        old = self.database.execute("SELECT 1 FROM column_rows WHERE owner=? AND role=? AND row_start=?", (name, role, start)).fetchone()
                        if old is None:
                            self.database.execute("INSERT INTO column_rows VALUES (?,?,?,?,0)", (name, role, start,
                                None if stability is None else stability[start:start+1].detach().cpu().numpy().tobytes()))
                            c._snapshot_contacts(owner, name, role, start, 1)
                        self.database.execute("INSERT OR IGNORE INTO column_rollback VALUES (?,?,?,?,?,?,?)", (*key, columns,
                            packed[start:start+1, first:last].detach().cpu().contiguous().numpy().tobytes(),
                            connections[start:start+1, column:column+columns].detach().cpu().contiguous().numpy().tobytes()))
                        self.database.commit()

                    c._phase(snapshot)

                    def update():
                        if not c.context.is_rank_zero:
                            return 0, False
                        seed = int.from_bytes(hashlib.sha256((c.step_id + "\0" + json.dumps(key)).encode()).digest()[:8], "little") % (1 << 63)
                        generator = torch.Generator(device=packed.device).manual_seed(seed)
                        count, grew = apply_packed_gradient_tile(owner, packed, width, start, column, gradient.to(packed.device), rate, scale,
                            generator=generator, row_stability=stability, stability_strength=strength, connection_state=connections)
                        return count, bool(grew[0])

                    count, grew = c._objects(c._phase(update))[0]
                    if count:
                        changed_owners.add(name)
                    block = c._phase(lambda: packed[start:start+1, first:last].detach().to(device).clone())
                    states = c._phase(lambda: connections[start:start+1, column:column+columns].detach().to(device).clone())
                    if c.context.distributed:
                        dist.broadcast(block, src=0)
                        dist.broadcast(states, src=0)

                    def install():
                        replacement = block.to(packed.device)
                        with packed_diagnostic_write(owner, packed, replacement, start=start * packed.shape[1] + first):
                            packed[start:start+1, first:last].copy_(replacement)
                        replacement = states.to(connections.device)
                        with packed_diagnostic_write(owner, connections, replacement, start=start * width + column):
                            connections[start:start+1, column:column+columns].copy_(replacement)
                        self.database.execute("UPDATE column_rows SET grew=MAX(grew,?) WHERE owner=? AND role=? AND row_start=?", (int(grew), name, role, start))
                        self.database.commit()
                        setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
                        release_gradient_tile(packed, connections, start, 1, column, columns)

                    c._phase(install)
                    c._phase(scopes.close)
            last_key = key
        # All tiles used the original shared row resistance; advance it once,
        # then synchronize the actual repeated contacts for each touched row.
        for name, role, start, _old, grew in self.database.execute("SELECT owner,role,row_start,stability,grew FROM column_rows ORDER BY owner,role,row_start"):
            owner = c.owners[name]
            with ExitStack() as scopes:
                c._phase(lambda: scopes.enter_context(c._scope(owner)))
                stability = c._stability(owner, role)
                if stability is not None:
                    c._phase(lambda: finish_row_resistance(owner, stability, start,
                        torch.tensor([bool(grew)], device=stability.device), float(getattr(owner, "_packed_stability_strength", 0))))
                c._install_contact_stream(owner, role, start, 1)
                c._phase(scopes.close)
        return changed_owners

    def rollback(self):
        c = self.controller
        for name, role, start, column, columns, signs, states in self.database.execute("SELECT * FROM column_rollback ORDER BY owner,role,row_start,column_start"):
            owner = c.owners[name]
            with c._scope(owner):
                packed, connections = c._buffer(owner, role), c._connections(owner, role)
                first, last = column // 4, (column + columns + 3) // 4
                replacement = torch.frombuffer(bytearray(signs), dtype=torch.uint8).to(packed.device)
                with packed_diagnostic_write(owner, packed, replacement, start=start * packed.shape[1] + first):
                    packed[start, first:last].copy_(replacement)
                replacement = torch.frombuffer(bytearray(states), dtype=torch.uint8).to(connections.device)
                with packed_diagnostic_write(owner, connections, replacement, start=start * connections.shape[1] + column):
                    connections[start, column:column+columns].copy_(replacement)
                release_gradient_tile(packed, connections, start, 1, column, columns)
                setattr(owner, "_packed_validated_version" if role == "weight" else "_bias_validated_version", int(packed._version))
        for name, role, start, stable, _grew in self.database.execute("SELECT * FROM column_rows ORDER BY owner,role,row_start"):
            owner = c.owners[name]
            with c._scope(owner):
                if stable is not None:
                    stability = c._stability(owner, role)
                    replacement = torch.frombuffer(bytearray(stable), dtype=torch.uint8).to(stability.device)
                    with packed_diagnostic_write(owner, stability, replacement, start=start):
                        stability[start:start+1].copy_(replacement)
                for (payload,) in self.database.execute("SELECT payload FROM parallel_rollback WHERE owner=? AND role=? AND row_start=? ORDER BY sequence", (name, role, start)):
                    owner.install_parallel_connection_records_(json.loads(payload), bias=role == "bias", start=start, end=start+1)
