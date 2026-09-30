"""Exact assembly-only pages over paged structural and neural stores.

The assembly index may share the vector database or use a separate metadata
database. Its own ``page_rows`` method cannot score the neuron store on a
*different* SQLite path. This adapter walks assembly IDs by stable insertion sequence and reads
their one authoritative packed neuron row in bounded windows. Neither an
all-neuron scan nor a second learned assembly vector is introduced.

Both stores must remain at the captured generation across every page and the
caller's final readout. A concurrent edit fails closed rather than mixing
old and new neural state. This is a working-cache read contract, not a claim
that two independent SQLite files are an atomic checkpoint.
"""

from __future__ import annotations

import hashlib
import torch
from dataclasses import dataclass
from typing import Any, Optional

from .paged_assembly_index import PagedAssemblyIndex
from .paged_packed_vectors import PagedPackedVectors, _decoded_row
from .paged_vector_scoring import MAX_PAGE_ROWS, PackedVectorPage, PackedVectorRow
from .paged_store_counts import row_count
from .packed_vsa_vectors import _decode_exact_row


@dataclass(frozen=True)
class _Generation:
    index_store_id: str
    index_revision: int
    assembly_count: int
    assembly_high_water: int
    vector_store_id: str
    vector_revision: int


class PagedAssemblyVectorProvider:
    """Bounded exact scorer input for metadata-only assembly indexes."""

    def __init__(
        self, index: PagedAssemblyIndex, vectors: PagedPackedVectors
    ) -> None:
        if not isinstance(index, PagedAssemblyIndex):
            raise TypeError("assembly scorer requires a paged assembly index")
        if not isinstance(vectors, PagedPackedVectors):
            raise TypeError("assembly scorer requires packed neuron vectors")
        if index._vectors is not None and index._vectors is not vectors:
            raise ValueError("assembly scorer cannot use a second vector authority")
        if (
            index.path.resolve() == vectors.path.resolve()
            and index._vectors is not vectors
        ):
            raise ValueError("shared assembly cache must use its exact vector object")
        self.index = index
        self.vectors = vectors
        self.dimensions = vectors.dimensions
        self._generation: Optional[_Generation] = None
        self._snapshot_id: Optional[str] = None

    def iter_activities(self, *, page_size: int = 128):
        """Exact original float activity math; no metadata/per-ID SQL scan."""

        snapshot = self.current_snapshot()
        cursor = None
        while True:
            page = self.page_rows(snapshot, cursor, page_size)
            for row in page.rows:
                value = _decode_exact_row(row.packed, self.dimensions).to(torch.float32)
                norm = value.norm()
                yield row.assembly_id, value / norm if float(norm) > 0 else value
            if not page.has_more:
                break
            cursor = page.next_cursor
        self.assert_unchanged(snapshot)

    @staticmethod
    def _index_state(connection: Any) -> tuple[str, int, int, int]:
        store_id = PagedAssemblyIndex._store_id(connection)
        revision = PagedAssemblyIndex._index_revision(connection)
        count = row_count(connection, "assembly_records")
        high_water = connection.execute("SELECT COALESCE(MAX(sequence),0) FROM assembly_records").fetchone()[0]
        return store_id, revision, int(count), int(high_water)

    @staticmethod
    def _vector_state(connection: Any) -> tuple[str, int]:
        return (
            PagedPackedVectors.store_id_in_connection(connection),
            PagedPackedVectors.revision_in_connection(connection),
        )

    def _read_generation(self) -> _Generation:
        with self.index._transaction() as connection:
            index_state = self._index_state(connection)
        with self.vectors._transaction() as connection:
            vector_state = self._vector_state(connection)
        return _Generation(*index_state, *vector_state)

    def _assert_revisions(self, generation: _Generation) -> None:
        # Transaction-maintained counters make capture/final checks indexed.
        # Per-page safety needs only monotonic store revisions.
        with self.index._transaction() as connection:
            if (
                self.index._store_id(connection),
                self.index._index_revision(connection),
            ) != (generation.index_store_id, generation.index_revision):
                raise ValueError("assembly/vector snapshot generation drift")
        with self.vectors._transaction() as connection:
            if self._vector_state(connection) != (
                generation.vector_store_id, generation.vector_revision,
            ):
                raise ValueError("assembly/vector snapshot generation drift")

    def current_snapshot(self) -> str:
        first = self._read_generation()
        if first != self._read_generation():
            raise ValueError("assembly/vector snapshot changed while captured")
        digest = hashlib.sha256(repr(first).encode("utf-8")).hexdigest()
        self._generation = first
        self._snapshot_id = digest
        return digest

    def assert_unchanged(self, snapshot_id: str) -> None:
        if snapshot_id != self._snapshot_id or self._generation is None:
            raise ValueError("assembly/vector snapshot identity is invalid")
        if self._read_generation() != self._generation:
            raise ValueError("assembly/vector snapshot generation drift")

    @staticmethod
    def _cursor(snapshot_id: str, after: int, seen: int) -> str:
        body = "%s:%d:%d" % (snapshot_id, after, seen)
        return "%d:%d:%s" % (
            after, seen, hashlib.sha256(body.encode("ascii")).hexdigest()
        )

    @classmethod
    def _decode_cursor(
        cls, cursor: str, snapshot_id: str, generation: _Generation
    ) -> tuple[int, int]:
        if not isinstance(cursor, str) or len(cursor) > 160:
            raise ValueError("assembly vector cursor is invalid")
        fields = cursor.split(":")
        if len(fields) != 3 or not fields[0].isdecimal() or not fields[1].isdecimal():
            raise ValueError("assembly vector cursor is invalid")
        after, seen = int(fields[0]), int(fields[1])
        if (
            after > generation.assembly_high_water
            or seen > generation.assembly_count
            or cursor != cls._cursor(snapshot_id, after, seen)
        ):
            raise ValueError("assembly vector cursor is invalid")
        return after, seen

    def page_rows(
        self, snapshot_id: str, cursor: Optional[str], page_size: int
    ) -> PackedVectorPage:
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= MAX_PAGE_ROWS
        ):
            raise ValueError("assembly vector page size is invalid")
        generation = self._generation
        if generation is None or snapshot_id != self._snapshot_id:
            raise ValueError("assembly/vector snapshot identity is invalid")
        after, seen = (
            self._decode_cursor(cursor, snapshot_id, generation)
            if cursor is not None else (0, 0)
        )

        with self.index._transaction() as connection:
            if (
                self.index._store_id(connection),
                self.index._index_revision(connection),
            ) != (
                generation.index_store_id,
                generation.index_revision,
            ):
                raise ValueError("assembly/vector snapshot generation drift")
            sizes = connection.execute(
                "SELECT sequence,LENGTH(assembly_id) FROM assembly_records "
                "WHERE sequence>? AND sequence<=? "
                "ORDER BY sequence LIMIT ?",
                (after, generation.assembly_high_water, page_size + 1),
            ).fetchall()
            has_more = len(sizes) > page_size
            selected = sizes[:page_size]
            self.index._reserve_memory(
                4096 + 4 * sum(int(size) for _, size in selected)
                + 128 * max(1, len(selected)),
                "assembly scorer ID page",
            )
            last = int(selected[-1][0]) if selected else after
            identifiers = connection.execute(
                "SELECT sequence,assembly_id FROM assembly_records "
                "WHERE sequence>? AND sequence<=? ORDER BY sequence",
                (after, last),
            ).fetchall() if selected else []
            if [row[0] for row in identifiers] != [row[0] for row in selected]:
                raise ValueError("assembly vector page changed during read")

        rows_by_id: dict[str, bytes] = {}
        with self.vectors._transaction() as connection:
            if self._vector_state(connection) != (
                generation.vector_store_id, generation.vector_revision,
            ):
                raise ValueError("assembly/vector snapshot generation drift")
            self.vectors._reserve_memory(
                4096 + len(identifiers) * (4 * self.vectors.row_bytes + 256),
                "assembly scorer packed page",
            )
            for offset in range(0, len(identifiers), 256):
                batch = [str(identifier) for _, identifier in identifiers[offset:offset + 256]]
                placeholders = ",".join("?" for _ in batch)
                fetched = connection.execute(
                    "SELECT sequence,vector_id,packed,updates,row_sha256 "
                    "FROM paged_vector_rows WHERE vector_id IN (%s)" % placeholders,
                    batch,
                ).fetchall()
                for row in fetched:
                    _sequence, identifier, packed, _updates = _decoded_row(
                        row, self.dimensions
                    )
                    rows_by_id[identifier] = packed
                if len(fetched) != len(batch):
                    raise ValueError("assembly lacks its authoritative packed neuron row")

        self._assert_revisions(generation)
        result = tuple(
            PackedVectorRow(int(sequence), str(identifier), rows_by_id[str(identifier)])
            for sequence, identifier in identifiers
        )
        next_seen = seen + len(result)
        if next_seen > generation.assembly_count or (
            not has_more and next_seen != generation.assembly_count
        ):
            raise ValueError("assembly vector snapshot coverage mismatch")
        return PackedVectorPage(
            rows=result,
            next_cursor=self._cursor(snapshot_id, last, next_seen),
            has_more=has_more,
            snapshot_id=snapshot_id,
        )
