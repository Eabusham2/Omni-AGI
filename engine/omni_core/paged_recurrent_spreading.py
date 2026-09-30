"""Exact synchronous recurrent spreading with a disk-backed active frontier.

This is a numerical execution scratch, not learned state or answer storage.
No hop, edge, active-node or top-k limit is introduced. Every eligible signed
ternary contribution participates until the existing contraction settles.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
import json
from pathlib import Path
from typing import Any, Optional


class PagedRecurrentState:
    """Own one expendable SQLite frontier with constant-size Python buffers."""

    def __init__(
        self, directory: Path, *, reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> None:
        self.reserve = reserve
        self._require(65536, "recurrent frontier initialization")
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=".recurrent-frontier-", dir=directory)
        self.path = Path(self._temporary.name) / "frontier.sqlite3"
        self.connection = sqlite3.connect(self.path, uri=True)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-256")
        self.connection.execute("PRAGMA mmap_size=0")
        for table, columns in (
            ("seeds", "node_id TEXT PRIMARY KEY,value REAL NOT NULL"),
            ("activation", "node_id TEXT PRIMARY KEY,value REAL NOT NULL"),
            ("admitted", "node_id TEXT PRIMARY KEY"),
            ("drives", "node_id TEXT PRIMARY KEY,value REAL NOT NULL"),
            ("settled", "node_id TEXT PRIMARY KEY,value REAL NOT NULL"),
            ("incoming", "node_id TEXT PRIMARY KEY,count INTEGER NOT NULL"),
            ("edges", "ordinal INTEGER PRIMARY KEY,source_id TEXT NOT NULL,target_id TEXT NOT NULL,"
                      "level INTEGER NOT NULL CHECK(level IN (-1,1))"),
            ("recalled", "position INTEGER PRIMARY KEY,assembly_id TEXT NOT NULL UNIQUE,payload BLOB NOT NULL"),
        ):
            self.connection.execute("CREATE TABLE %s (%s)" % (table, columns))
        self.connection.execute("CREATE INDEX recurrent_edge_source ON edges(source_id,ordinal)")
        self.rounds = 0
        self.convergence_delta = 0.0
        self.eligible_edges = 0
        self.inhibitory_edges = 0
        self.inhibitory_signals = 0
        self.active_count = 0
        self.seed_count = 0
        self.suppressed_seeds = 0
        self.closed = False
        self.recalled_count = 0

    def _require(self, size: int, operation: str) -> None:
        if self.reserve is not None and self.reserve(max(1, size), operation) is False:
            raise RuntimeError("%s paused at resource reserve" % operation)

    def load(
        self, seeds: Iterable[tuple[str, float]], edges: Iterable[tuple[str, str, int]],
        *, node_exists: Callable[[str], bool],
    ) -> None:
        connection = self.connection
        for identifier, value in seeds:
            if not isinstance(identifier, str) or not identifier or not math.isfinite(float(value)):
                raise ValueError("recurrent seed is invalid")
            self._require(4096 + len(identifier) * 4, "recurrent seed page")
            connection.execute("INSERT INTO seeds VALUES(?,?)", (identifier, float(value)))
            self.seed_count += 1
        for source, target, level in edges:
            if type(level) is not int or level not in {-1, 0, 1}:
                raise ValueError("recurrent contribution is not exactly ternary")
            if level == 0 or not node_exists(source) or not node_exists(target):
                continue
            self._require(4096 + 4 * (len(source) + len(target)), "recurrent edge page")
            connection.execute("INSERT INTO edges VALUES(?,?,?,?)",
                               (self.eligible_edges, source, target, level))
            connection.execute(
                "INSERT INTO incoming VALUES(?,1) ON CONFLICT(node_id) DO UPDATE SET count=count+1",
                (target,),
            )
            self.eligible_edges += 1
            self.inhibitory_edges += int(level < 0)
        connection.execute("INSERT INTO activation SELECT * FROM seeds")
        connection.execute("INSERT INTO admitted SELECT node_id FROM seeds")
        connection.commit()
        self.active_count = self.seed_count

    def use_graph(self, graph: Any) -> None:
        """Share a verified same-revision read-only graph, never copy its corpus."""

        self.connection.execute("DROP TABLE edges")
        self.connection.execute("DROP TABLE incoming")
        self.connection.execute("ATTACH DATABASE ? AS graph_index", (graph.path.as_uri() + "?mode=ro",))
        self.connection.execute("CREATE TEMP VIEW edges AS SELECT * FROM graph_index.edges")
        self.connection.execute("CREATE TEMP VIEW incoming AS SELECT * FROM graph_index.incoming")
        self.eligible_edges = graph.eligible_edges
        self.inhibitory_edges = graph.inhibitory_edges
        self._graph_owner = graph  # Keep its owned scratch alive for this result.

    def load_seeds(self, seeds: Iterable[tuple[str, float]]) -> None:
        for identifier, value in seeds:
            if not isinstance(identifier, str) or not identifier or not math.isfinite(float(value)):
                raise ValueError("recurrent seed is invalid")
            self._require(4096 + len(identifier) * 4, "recurrent seed page")
            self.connection.execute("INSERT INTO seeds VALUES(?,?)", (identifier, float(value)))
            self.seed_count += 1
        self.connection.execute("INSERT INTO activation SELECT * FROM seeds")
        self.connection.execute("INSERT INTO admitted SELECT node_id FROM seeds")
        self.connection.commit()
        self.active_count = self.seed_count

    def settle(self, *, slots: int, adaptive_floor: float) -> None:
        if slots < 1 or not math.isfinite(adaptive_floor) or adaptive_floor <= 0:
            raise ValueError("recurrent pressure configuration is invalid")
        connection = self.connection
        while self.active_count:
            pressure = max(1.0, self.active_count / float(slots))
            floor = max(1e-5, adaptive_floor * (0.02 + 0.03 * math.log2(pressure + 1.0)))
            outgoing = int(connection.execute(
                "SELECT COUNT(*) FROM activation AS a CROSS JOIN edges AS e ON e.source_id=a.node_id "
                "WHERE ABS(a.value)>1e-12"
            ).fetchone()[0])
            self._require(65536 + 256 * (self.active_count + outgoing), "recurrent settling disk round")
            connection.execute("DELETE FROM drives")
            # Edge ordinal retains the exact source topology's order. SQLite
            # performs the sum on disk; no all-frontier Python dict is formed.
            connection.execute(
                "INSERT INTO drives SELECT e.target_id,SUM(a.value*e.level) "
                "FROM activation AS a CROSS JOIN edges AS e ON e.source_id=a.node_id "
                "WHERE ABS(a.value)>1e-12 GROUP BY e.target_id"
            )
            self.inhibitory_signals = int(connection.execute(
                "SELECT COUNT(*) FROM activation AS a CROSS JOIN edges AS e ON e.source_id=a.node_id "
                "WHERE ABS(a.value)>1e-12 AND a.value*e.level<0"
            ).fetchone()[0])
            connection.execute("DELETE FROM settled")
            connection.execute(
                "INSERT INTO settled "
                "SELECT c.node_id,MAX(-1.0,MIN(1.0,COALESCE(s.value,0.0)"
                "+0.52*COALESCE(d.value,0.0)/MAX(1,COALESCE(i.count,0)))) "
                "FROM (SELECT node_id FROM admitted UNION SELECT node_id FROM seeds "
                "UNION SELECT node_id FROM drives) AS c "
                "LEFT JOIN seeds AS s ON s.node_id=c.node_id "
                "LEFT JOIN drives AS d ON d.node_id=c.node_id "
                "LEFT JOIN incoming AS i ON i.node_id=c.node_id "
                "WHERE EXISTS(SELECT 1 FROM admitted AS old WHERE old.node_id=c.node_id) "
                "OR ABS(MAX(-1.0,MIN(1.0,COALESCE(s.value,0.0)"
                "+0.52*COALESCE(d.value,0.0)/MAX(1,COALESCE(i.count,0)))))>=?", (floor,),
            )
            self.convergence_delta = float(connection.execute(
                "SELECT COALESCE(MAX(ABS(COALESCE(s.value,0)-COALESCE(a.value,0))),0) "
                "FROM (SELECT node_id FROM settled UNION SELECT node_id FROM activation) AS c "
                "LEFT JOIN settled AS s ON s.node_id=c.node_id "
                "LEFT JOIN activation AS a ON a.node_id=c.node_id"
            ).fetchone()[0])
            connection.execute("INSERT OR IGNORE INTO admitted SELECT node_id FROM settled")
            connection.execute("DELETE FROM activation")
            connection.execute("INSERT INTO activation SELECT * FROM settled")
            self.active_count = int(connection.execute("SELECT COUNT(*) FROM activation").fetchone()[0])
            connection.commit()
            self.rounds += 1
            if self.convergence_delta <= max(1e-7, floor * 0.001):
                break
        self.suppressed_seeds = int(connection.execute(
            "SELECT COUNT(*) FROM seeds AS s LEFT JOIN activation AS a ON a.node_id=s.node_id "
            "WHERE s.value>=? AND COALESCE(a.value,0)<?", (adaptive_floor, adaptive_floor),
        ).fetchone()[0])

    def iter_activation(self, *, positive_only: bool = False) -> Iterator[tuple[str, float]]:
        condition = "WHERE value>0" if positive_only else ""
        for identifier, value in self.connection.execute(
            "SELECT node_id,value FROM activation %s ORDER BY value DESC,node_id" % condition
        ):
            yield identifier, float(value)

    def activation(self, identifier: str) -> float:
        row = self.connection.execute("SELECT value FROM activation WHERE node_id=?", (identifier,)).fetchone()
        return float(row[0]) if row else 0.0

    def append_recalled(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._require(4096 + 4 * len(encoded), "recurrent recalled metadata page")
        self.connection.execute("INSERT INTO recalled VALUES(?,?,?)",
                                (self.recalled_count, record["assembly_id"], encoded))
        self.recalled_count += 1

    def recalled(self) -> "PagedRecalledAssemblies":
        self.connection.commit()
        return PagedRecalledAssemblies(self)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        self._temporary.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PagedRecalledAssemblies(Sequence[dict[str, Any]]):
    """Unbounded addressable result with bounded per-record decoding."""

    def __init__(self, state: PagedRecurrentState) -> None:
        self.state = state

    def __len__(self) -> int:
        return self.state.recalled_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for (encoded,) in self.state.connection.execute("SELECT payload FROM recalled ORDER BY position"):
            yield json.loads(encoded)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            result = []
            for position in range(start, stop, step):
                result.append(self[position])
            return result
        if type(index) is not int:
            raise TypeError("recalled position must be an integer")
        position = index + len(self) if index < 0 else index
        row = self.state.connection.execute("SELECT payload FROM recalled WHERE position=?", (position,)).fetchone()
        if row is None:
            raise IndexError("recalled position is out of range")
        return json.loads(row[0])
