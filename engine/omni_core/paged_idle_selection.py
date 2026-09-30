"""Exact idle workspace ranking with dirty-group updates and safe bounds.

Every assembly remains eligible/addressable; the only result cardinality is
the existing hardware-derived active workspace. Bounds skip metadata pages
only when no member can outrank the current workspace. Global decay is applied
with the source's exact anchor math, not by silently treating rows unchanged.
The numeric components are expendable scheduling metadata, never neural
weights, recall answers, or recovery authority.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import os
import sqlite3
import tempfile
import weakref
from pathlib import Path

from .authenticated_paged_cache import canonical
from .paged_dirty_journal import source_stamp
from .paged_merkle_kv import SqliteMerkleMap
from .paged_neuron_metadata import PagedNeuronMetadata


_ACTIVE: dict[str, weakref.ReferenceType] = {}


def observe_idle_source_connection(connection: sqlite3.Connection, path: Path) -> None:
    """Source-transaction notifications, including harmless rollback extras.

    All store connections register this no-fail function. With no active
    selector it is a no-op: a later selector builds from actual current rows.
    The process-owned authenticated dirty set cannot omit/forge IDs on disk.
    """

    key = str(path.resolve())
    def changed(identifier):
        reference = _ACTIVE.get(key)
        selector = reference() if reference is not None else None
        if selector is not None:
            selector.queue_change(identifier)
        return 0
    connection.create_function("omni_idle_source_changed", 1, changed)


class PagedIdleSelector:
    PAGE_ROWS = 128

    def __init__(self, memory) -> None:
        self.memory = weakref.ref(memory)
        self.source = memory.assemblies.index
        self.neurons = memory.neurons
        if self.source.path.resolve() != self.neurons.path.resolve() or self.source.path.resolve() != memory.neuron_vectors.path.resolve():
            raise ValueError("indexed idle selection requires the shared three-store paged cache")
        self.source._reserve_disk(128 * 1024, "idle query index initialization")
        self.source._reserve_memory(320 * 1024, "idle query index window")
        self._temporary = tempfile.TemporaryDirectory(prefix=".idle-query-", dir=self.source.path.parent)
        self.path = Path(self._temporary.name) / "priority.sqlite3"
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-256")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("CREATE TABLE components(sequence INTEGER PRIMARY KEY,assembly_id TEXT UNIQUE NOT NULL,group_id INTEGER NOT NULL,payload BLOB NOT NULL,authentication TEXT NOT NULL)")
        self.connection.execute("CREATE INDEX idle_component_group ON components(group_id,sequence)")
        self.connection.execute("CREATE TABLE summaries(group_id INTEGER PRIMARY KEY,payload BLOB NOT NULL,authentication TEXT NOT NULL)")
        SqliteMerkleMap.ensure_schema(self.connection)
        self.merkle = SqliteMerkleMap("idle-priority-summaries")
        self.root = self.merkle.empty()
        self.changes = SqliteMerkleMap("idle-priority-changes")
        self.change_root = self.changes.empty()
        self.invalid = False
        self.secret = os.urandom(32)
        self.closed = False
        self.last_groups_read = self.last_source_rows_read = 0
        self._install_changes()
        self._rebuild()
        _ACTIVE[str(self.source.path.resolve())] = weakref.ref(self)

    def _sign(self, kind, key, payload):
        return hmac.new(self.secret, kind.encode() + b"\0" + str(key).encode() + b"\0" + payload, hashlib.sha256).hexdigest()

    def _checked(self, kind, key, payload, authentication):
        if not isinstance(payload, bytes) or not hmac.compare_digest(self._sign(kind, key, payload), authentication):
            raise ValueError("idle query cache row is unauthenticated")
        return json.loads(payload)

    def _install_changes(self):
        with self.source._transaction(write=True) as connection:
            for table, column in (("assembly_records", "assembly_id"), ("paged_neuron_records", "neuron_id"), ("paged_vector_rows", "vector_id")):
                for event in ("INSERT", "UPDATE", "DELETE"):
                    row = "OLD" if event == "DELETE" else "NEW"
                    connection.execute("DROP TRIGGER IF EXISTS idle_query_%s_%s" % (table, event.lower()))
                    connection.execute(
                        "CREATE TRIGGER idle_query_%s_%s AFTER %s ON %s BEGIN "
                        "SELECT omni_idle_source_changed(%s.%s); END"
                        % (table, event.lower(), event, table, row, column)
                    )

    def queue_change(self, identifier):
        try:
            if self.invalid or self.closed:
                return
            self.source._reserve_disk(65536 + len(identifier) * 4, "idle dirty query metadata")
            root = self.changes.put(self.connection, self.change_root, identifier, hashlib.sha256(identifier.encode()).hexdigest())
            self.connection.commit()
            self.change_root = root
        except Exception:
            # A disposable scheduling index can pause/rebuild, not reject a
            # valid source transaction or become a second memory authority.
            self.invalid = True

    def _group(self, connection, group_id):
        low, high = group_id * self.PAGE_ROWS + 1, (group_id + 1) * self.PAGE_ROWS
        sizes = connection.execute(
            "SELECT COALESCE(SUM(LENGTH(a.record_json)+COALESCE(LENGTH(n.record_json),0)),0) "
            "FROM assembly_records AS a LEFT JOIN paged_neuron_records AS n ON n.neuron_id=a.assembly_id "
            "WHERE a.sequence BETWEEN ? AND ?", (low, high),
        ).fetchone()[0]
        self.source._reserve_memory(4096 + 4 * int(sizes), "idle changed metadata page")
        self.source._reserve_disk(65536 + 4 * int(sizes), "idle changed priority page")
        decay = self.neurons._decay_state(connection)
        records = []
        for row in connection.execute(
            "SELECT a.sequence,a.assembly_id,a.fingerprint,a.record_json,a.record_sha256,"
            "n.sequence,n.neuron_id,n.record_json,n.record_sha256,n.decay_log_anchor,n.decay_sum_anchor,n.decay_zero_anchor "
            "FROM assembly_records AS a LEFT JOIN paged_neuron_records AS n ON n.neuron_id=a.assembly_id "
            "WHERE a.sequence BETWEEN ? AND ? AND EXISTS(SELECT 1 FROM paged_vector_rows AS v WHERE v.vector_id=a.assembly_id) "
            "ORDER BY a.sequence", (low, high),
        ):
            assembly = self.source._decode_record(row[:5])
            node = {}
            if row[5] is not None:
                self.neurons._decode(row[5:], decay)  # Validate actual source checksum/anchors.
                node = json.loads(row[7])
            records.append({"id": assembly["id"], "sequence": row[0],
                            "importance": float(assembly.get("importance", 0.0)),
                            "rehearsals": int(assembly.get("rehearsals", 0)),
                            "activation": float(node.get("activation", 0.0) or 0.0),
                            "uncertainty": float(node.get("uncertainty", 0.5)),
                            "uncertaintyDecays": "uncertainty" in node,
                            "exposures": int(node.get("exposures", 0)),
                            "anchorLog": float(row[9]) if node else decay[1],
                            "anchorSum": float(row[10]) if node else decay[2],
                            "anchorZero": int(row[11]) if node else decay[3]})
        self.last_source_rows_read += len(records)
        self.connection.execute("DELETE FROM components WHERE group_id=?", (group_id,))
        previous = self.merkle.get(self.connection, self.root, str(group_id))
        if not records:
            self.connection.execute("DELETE FROM summaries WHERE group_id=?", (group_id,))
            if previous is not None:
                self.root = self.merkle.delete(self.connection, self.root, str(group_id))
            return
        for record in records:
            payload = canonical(record)
            self.connection.execute("INSERT INTO components VALUES(?,?,?,?,?)", (record["sequence"], record["id"], group_id, payload,
                                    self._sign("component", record["sequence"], payload)))
        dynamic = [record for record in records if record["uncertaintyDecays"]]
        summary = {
            "first": records[0]["sequence"], "count": len(records),
            "base": max(record["importance"] * 0.30 + 1.0 / (1.0 + record["rehearsals"]) * 0.15 for record in records),
            "activation": max((record["activation"] for record in records), default=0.0),
            "anchorLog": min(record["anchorLog"] for record in records),
            "zero": max(record["anchorZero"] for record in records),
            "uncertainty": max((record["uncertainty"] for record in dynamic), default=0.0),
            "uncertaintyStatic": max((record["uncertainty"] for record in records if not record["uncertaintyDecays"]), default=0.0),
            "sum": min((record["anchorSum"] for record in dynamic), default=decay[2]),
            "exposures": min((record["exposures"] for record in dynamic), default=0),
            "componentSha256": hashlib.sha256(b"".join(canonical(record) + b"\n" for record in records)).hexdigest(),
        }
        payload = canonical(summary)
        self.connection.execute("INSERT OR REPLACE INTO summaries VALUES(?,?,?)", (group_id, payload, self._sign("summary", group_id, payload)))
        self.root = self.merkle.put(self.connection, self.root, str(group_id), hashlib.sha256(payload).hexdigest())

    def _rebuild(self):
        self.connection.execute("DELETE FROM components")
        self.connection.execute("DELETE FROM summaries")
        self.root = self.merkle.clear(self.connection)
        with self.source._transaction() as connection:
            stamp = source_stamp(connection)
            groups = connection.execute("SELECT DISTINCT (sequence-1)/? FROM assembly_records ORDER BY sequence", (self.PAGE_ROWS,))
            for (group_id,) in groups:
                self._group(connection, int(group_id))
        self.change_root = self.changes.clear(self.connection)
        self.connection.commit()
        self.stamp = stamp
        self.invalid = False

    def refresh(self):
        self.last_source_rows_read = 0
        if self.invalid:
            self._rebuild()
            return
        try:
            with self.source._transaction() as connection:
                stamp = source_stamp(connection)
                if stamp[::2][:3] != self.stamp[::2][:3]:
                    raise ValueError("idle query source identity changed")
                # Metadata-only global decay has exact compatible epoch math.
                # Any other revision drift needs at least one source-row log.
                if not self.change_root.count and (stamp[1] != self.stamp[1] or stamp[3] != self.stamp[3] or
                                  (stamp[5] != self.stamp[5] and stamp[6:] == self.stamp[6:])):
                    raise ValueError("idle query source drift has no journal records")
                self.connection.execute("CREATE TEMP TABLE IF NOT EXISTS idle_changed_groups(group_id INTEGER PRIMARY KEY)")
                self.connection.execute("DELETE FROM idle_changed_groups")
                for identifier, _digest in self.changes.iter_items(self.connection, self.change_root):
                    old = self.connection.execute("SELECT group_id FROM components WHERE assembly_id=?", (identifier,)).fetchone()
                    current = connection.execute("SELECT (sequence-1)/? FROM assembly_records WHERE assembly_id=?", (self.PAGE_ROWS, identifier)).fetchone()
                    for value in (old, current):
                        if value is not None:
                            self.connection.execute("INSERT OR IGNORE INTO idle_changed_groups VALUES(?)", value)
                for (group_id,) in self.connection.execute("SELECT group_id FROM idle_changed_groups ORDER BY group_id"):
                    self._group(connection, int(group_id))
            self.change_root = self.changes.clear(self.connection)
            self.connection.commit()
            self.stamp = stamp
        except (ValueError, sqlite3.DatabaseError):
            self.connection.rollback()
            self._rebuild()  # Never use a missed/malformed derived change index.

    def select(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("idle workspace capacity must be positive")
        self.refresh()
        memory = self.memory()
        if memory is None:
            raise RuntimeError("idle query source is closed")
        with self.neurons._transaction() as source_connection:
            decay = self.neurons._decay_state(source_connection)
        legacy = bool(memory.attention_legacy_raw_active)
        active_ids = memory.attention_active_neuron_ids
        count = int(self.connection.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])
        if count != self.root.count:
            self._rebuild()
            return self.select(capacity)
        def bound(group_id, payload, authentication):
            summary = self._checked("summary", group_id, payload, authentication)
            if self.merkle.get(self.connection, self.root, str(group_id)) != hashlib.sha256(payload).hexdigest():
                raise ValueError("idle bound root changed")
            activation = 0.0
            if (legacy or active_ids) and summary["zero"] == decay[3]:
                activation = max(0.0, summary["activation"]) * math.exp(min(0.0, decay[1] - summary["anchorLog"]))
            uncertainty = max(summary["uncertaintyStatic"], min(1.0, summary["uncertainty"] +
                              max(0.0, decay[2] - summary["sum"]) / (1.0 + summary["exposures"])))
            # Outward rounding cannot discard a near-equal real candidate.
            upper = summary["base"] + activation * 0.35 + uncertainty * 0.20
            return upper + 8 * math.ulp(upper)
        self.connection.create_function("idle_upper", 3, bound)
        heap = []
        self.last_groups_read = 0
        for group_id, encoded_summary, authentication, upper in self.connection.execute(
            "SELECT group_id,payload,authentication,idle_upper(group_id,payload,authentication) AS upper "
            "FROM summaries ORDER BY upper DESC,group_id"
        ):
            summary = self._checked("summary", group_id, encoded_summary, authentication)
            if len(heap) == capacity and (upper, -summary["first"]) <= heap[0][:2]:
                break
            self.last_groups_read += 1
            observed = 0
            digest = hashlib.sha256()
            for sequence, payload, authentication in self.connection.execute("SELECT sequence,payload,authentication FROM components WHERE group_id=? ORDER BY sequence", (group_id,)):
                record = self._checked("component", sequence, payload, authentication)
                observed += 1
                digest.update(payload + b"\n")
                activation = 0.0
                if record["anchorZero"] == decay[3] and (legacy or record["id"] in active_ids):
                    activation = record["activation"] * math.exp(min(0.0, decay[1] - record["anchorLog"]))
                uncertainty = record["uncertainty"]
                if record["uncertaintyDecays"]:
                    uncertainty = min(1.0, uncertainty + max(0.0, decay[2] - record["anchorSum"]) / (1.0 + record["exposures"]))
                score = activation * 0.35 + record["importance"] * 0.30 + uncertainty * 0.20 + 1.0 / (1.0 + record["rehearsals"]) * 0.15
                item = score, -sequence, record["id"]
                if len(heap) < capacity:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
            if observed != summary["count"] or digest.hexdigest() != summary["componentSha256"]:
                raise ValueError("idle component page coverage changed")
        with self.source._transaction() as connection:
            if source_stamp(connection) != self.stamp:
                raise ValueError("idle selection source changed before publication")
        return [memory.assemblies.get_by_id(identifier) for _score, _sequence, identifier in sorted(heap, reverse=True)]

    def close(self):
        if not self.closed:
            self.closed = True
            key = str(self.source.path.resolve())
            if _ACTIVE.get(key) is not None and _ACTIVE[key]() is self:
                _ACTIVE.pop(key)
            self.connection.close()
            self._temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def select_paged_idle_workspace(memory, capacity: int):
    selector = getattr(memory, "_paged_idle_selector", None)
    if selector is None:
        selector = PagedIdleSelector(memory)
        memory._paged_idle_selector = selector
    return selector.select(capacity)
