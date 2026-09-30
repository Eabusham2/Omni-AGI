"""Exact, incrementally maintained disk graph for recurrent sparse recall.

This expendable execution index is never neural recovery state. It contains
only structural IDs and exact ternary signs from the actual synapse mapping.
Missing endpoints are retained but cannot fire; membership deltas revisit only
incident rows. A missed notification or unsupported source forces a checked
full rebuild, never silently incomplete spreading.
"""

from __future__ import annotations

import sqlite3
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable

from .paged_merkle_kv import SqliteMerkleMap


class PagedRecallGraph:
    def __init__(self, directory: Path, *, reserve: Callable[[int, str], Any]) -> None:
        self.reserve = reserve
        self._require(65536, "paged recall graph initialization")
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=".recurrent-graph-", dir=directory)
        self.path = Path(self._temporary.name) / "graph.sqlite3"
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-256")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute(
            "CREATE TABLE raw_edges(edge_id TEXT PRIMARY KEY,ordinal INTEGER NOT NULL UNIQUE,"
            "source_id TEXT NOT NULL,target_id TEXT NOT NULL,level INTEGER NOT NULL CHECK(level IN (-1,1)),"
            "eligible INTEGER NOT NULL CHECK(eligible IN (0,1))) WITHOUT ROWID"
        )
        self.connection.execute("CREATE INDEX graph_edge_source ON raw_edges(source_id,ordinal)")
        self.connection.execute("CREATE INDEX graph_edge_target ON raw_edges(target_id,ordinal)")
        self.connection.execute(
            "CREATE VIEW edges AS SELECT ordinal,source_id,target_id,level FROM raw_edges WHERE eligible=1"
        )
        self.connection.execute("CREATE TABLE incoming(node_id TEXT PRIMARY KEY,count INTEGER NOT NULL) WITHOUT ROWID")
        SqliteMerkleMap.ensure_schema(self.connection)
        self.pending_map = SqliteMerkleMap("recurrent-graph-changes")
        self.pending_root = self.pending_map.empty()
        for event, old, new in (("INSERT", False, True), ("DELETE", True, False), ("UPDATE", True, True)):
            statements = []
            if old:
                statements.extend((
                    "UPDATE incoming SET count=count-1 WHERE node_id=OLD.target_id AND OLD.eligible=1;",
                    "DELETE FROM incoming WHERE node_id=OLD.target_id AND count=0;",
                ))
            if new:
                statements.append(
                    "INSERT INTO incoming SELECT NEW.target_id,1 WHERE NEW.eligible=1 "
                    "ON CONFLICT(node_id) DO UPDATE SET count=count+1;"
                )
            self.connection.execute(
                "CREATE TRIGGER graph_degree_%s AFTER %s ON raw_edges BEGIN %s END"
                % (event.lower(), event, " ".join(statements))
            )
        self.next_ordinal = 0
        self.eligible_edges = 0
        self.inhibitory_edges = 0
        self.applied_synapse_revision = None
        self.applied_neuron_revision = None
        self.queued_synapse_revision = None
        self.queued_neuron_revision = None
        self.invalid = False
        self.closed = False

    def _require(self, size: int, operation: str) -> None:
        if self.reserve(max(1, int(size)), operation) is False:
            raise RuntimeError("%s paused at resource reserve" % operation)

    def _replace(self, identifier: str, source: str, target: str, level: int, *, node_exists) -> None:
        if not isinstance(identifier, str) or not identifier or type(level) is not int or level not in {-1, 0, 1}:
            raise ValueError("recall graph edge identity/sign is invalid")
        old = self.connection.execute(
            "SELECT ordinal,level,eligible FROM raw_edges WHERE edge_id=?", (identifier,),
        ).fetchone()
        self._require(4096 + 4 * (len(identifier) + len(source) + len(target)), "recall graph edge delta")
        if old is not None:
            self.eligible_edges -= int(old[2])
            self.inhibitory_edges -= int(old[2] and old[1] < 0)
        if not level:
            self.connection.execute("DELETE FROM raw_edges WHERE edge_id=?", (identifier,))
            return
        eligible = int(node_exists(source) and node_exists(target))
        ordinal = old[0] if old is not None else self.next_ordinal
        if old is None:
            self.next_ordinal += 1
        self.connection.execute(
            "INSERT INTO raw_edges VALUES(?,?,?,?,?,?) ON CONFLICT(edge_id) DO UPDATE SET "
            "source_id=excluded.source_id,target_id=excluded.target_id,level=excluded.level,eligible=excluded.eligible",
            (identifier, ordinal, source, target, level, eligible),
        )
        self.eligible_edges += eligible
        self.inhibitory_edges += int(eligible and level < 0)

    def load(self, edges: Iterable[tuple[str, str, str, int]], *, node_exists,
             synapse_revision, neuron_revision) -> None:
        for identifier, source, target, level in edges:
            self._replace(identifier, source, target, level, node_exists=node_exists)
        self.connection.commit()
        self.applied_synapse_revision = synapse_revision
        self.queued_synapse_revision = synapse_revision
        self.applied_neuron_revision = neuron_revision
        self.queued_neuron_revision = neuron_revision

    def queue_change(self, identifier: str | None, revision: int) -> None:
        """A source mutation cannot fail just because its optional index failed."""

        try:
            if not identifier or self.invalid or self.closed or revision != self.queued_synapse_revision + 1:
                self.invalid = True
                return
            self._require(4096 + 4 * len(identifier), "recall graph change journal")
            key = "edge:" + identifier
            root = self.pending_map.put(self.connection, self.pending_root, key,
                                        hashlib.sha256(str(revision).encode()).hexdigest())
            self.connection.commit()
            self.pending_root = root
            self.queued_synapse_revision = revision
        except Exception:
            self.invalid = True

    def queue_membership_change(self, identifiers: tuple[str, ...] | None, revision: int) -> None:
        try:
            if not identifiers or self.invalid or self.closed or revision != self.queued_neuron_revision + 1:
                self.invalid = True
                return
            root = self.pending_root
            for identifier in identifiers:
                self._require(4096 + 4 * len(identifier), "recall graph membership journal")
                root = self.pending_map.put(self.connection, root, "node:" + identifier,
                                           hashlib.sha256(str(revision).encode()).hexdigest())
            self.connection.commit()
            self.pending_root = root
            self.queued_neuron_revision = revision
        except Exception:
            self.invalid = True

    def refresh(self, synapses, neurons) -> bool:
        """Apply complete notified edge/incident-node deltas, else request rebuild."""

        synapse_revision = getattr(synapses, "graph_revision", None)
        neuron_revision = getattr(neurons, "graph_revision", None)
        if self.invalid or synapse_revision is None or neuron_revision is None or (
            synapse_revision != self.queued_synapse_revision or neuron_revision != self.queued_neuron_revision
        ):
            return False
        try:
            for key, _digest in self.pending_map.iter_items(self.connection, self.pending_root):
                kind, _separator, identifier = key.partition(":")
                if kind == "edge":
                    record = synapses.get(identifier)
                    if record is None:
                        self._replace(identifier, "", "", 0, node_exists=lambda _identifier: False)
                    else:
                        from .vsa import NeuralSubstrate
                        self._replace(identifier, str(record["source_id"]), str(record["target_id"]),
                                      NeuralSubstrate.exact_effective_weight(record.get("effective_weight", 0)),
                                      node_exists=lambda value: value in neurons)
                elif kind == "node":
                    for edge_id, source, target, level in self.connection.execute(
                        "SELECT edge_id,source_id,target_id,level FROM raw_edges WHERE source_id=? OR target_id=?",
                        (identifier, identifier),
                    ):
                        self._replace(edge_id, source, target, int(level), node_exists=lambda value: value in neurons)
                else:
                    raise ValueError("recurrent graph delta kind is invalid")
            if synapse_revision != getattr(synapses, "graph_revision", None) or (
                neuron_revision != getattr(neurons, "graph_revision", None)
            ):
                raise ValueError("recurrent graph source changed during delta application")
            root = self.pending_map.clear(self.connection)
            self.connection.commit()
            self.pending_root = root
            self.applied_synapse_revision = synapse_revision
            self.applied_neuron_revision = neuron_revision
            return True
        except Exception:
            # A partially updated execution cache is never queried. The
            # source still owns every valid mutation; its next build is exact.
            self.connection.rollback()
            self.invalid = True
            return False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        self._temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
