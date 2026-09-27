"""A bounded assembly view over :mod:`paged_assembly_index`.

This deliberately is *not* a mutable ``list[dict]`` drop-in. A paged read is
detached from SQLite, so mutating a returned dictionary would silently lose
learned metadata. Public reads are recursively read-only. Existing records
must be changed with ``transaction().edit_by_id(...)``; the transaction keeps
at most its declared row and byte window, commits all changes atomically, and
never evicts an uncommitted record. Successive windows have no total-record
limit. Insertion-order indexes rely on the index's append-only sequence; a
sequence gap fails closed instead of returning the wrong assembly.
"""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Optional

from .paged_assembly_index import (
    AssemblyIndexResourcePause,
    PagedAssemblyBatch,
    PagedAssemblyIndex,
    _packed_payload,
    _record_payload,
)


def _read_only(value: Any) -> Any:
    """Freeze nested JSON too, including provenance maps and ID arrays."""

    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value


class PagedAssemblyView:
    """Append-only, insertion-ordered, read-only view of assembly metadata.

    ``view[index]`` supports integer positions, including negative positions.
    Slices are intentionally unsupported: materializing a large slice would
    defeat paging. Use ``iter_range`` for a bounded-memory range traversal.
    Iteration captures an insertion high-water at its first page; later appends
    are omitted, while updates to not-yet-read rows can still be observed.
    """

    def __init__(self, index: PagedAssemblyIndex, *, page_size: int = 128) -> None:
        if not isinstance(index, PagedAssemblyIndex):
            raise TypeError("paged assembly view requires a PagedAssemblyIndex")
        if type(page_size) is not int or not 1 <= page_size <= 4096:
            raise ValueError("assembly view page size must be between 1 and 4096")
        self.index = index
        self.page_size = page_size

    def _size(self) -> int:
        # The index exposes no deletion. A gap means an external writer has
        # broken the position-to-sequence mapping; never shift positions.
        with self.index._transaction() as connection:
            count, high_water = connection.execute(
                "SELECT COUNT(*),COALESCE(MAX(sequence),0) FROM assembly_records"
            ).fetchone()
        if int(count) != int(high_water):
            raise ValueError("assembly insertion sequence has a gap")
        return int(count)

    def __len__(self) -> int:
        return self._size()

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        if isinstance(position, slice):
            raise TypeError("assembly slices are unbounded; use iter_range")
        if type(position) is not int:
            raise TypeError("assembly position must be an integer")
        count = self._size()
        if position < 0:
            position += count
        if position < 0 or position >= count:
            raise IndexError("assembly position out of range")
        record = self.index.get_by_sequence(position + 1)
        if record is None:
            raise ValueError("assembly insertion sequence has a gap")
        return _read_only(record)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        self._size()
        for page in self.index.iter_pages(self.page_size):
            for record in page.records:
                yield _read_only(record)

    def iter_range(
        self, start: int = 0, stop: Optional[int] = None
    ) -> Iterator[Mapping[str, Any]]:
        """Traverse zero-based ``[start, stop)`` via bounded keyset pages."""

        count = self._size()
        if type(start) is not int or (stop is not None and type(stop) is not int):
            raise TypeError("assembly range positions must be integers")
        first, last, step = slice(start, stop).indices(count)
        assert step == 1
        if first >= last:
            return
        seen = 0
        page = self.index.page_after(
            first, page_size=min(self.page_size, last - first),
            through_sequence=last,
        )
        while True:
            if not page.records:
                raise ValueError("assembly insertion sequence has a gap")
            for record in page.records:
                yield _read_only(record)
                seen += 1
            if not page.has_more:
                break
            page = self.index.page(page_size=self.page_size, cursor=page.cursor)
        if seen != last - first:
            raise ValueError("assembly insertion sequence has a gap")

    def get_by_id(self, assembly_id: str) -> Optional[Mapping[str, Any]]:
        record = self.index.get_by_id(assembly_id)
        return None if record is None else _read_only(record)

    def get_by_fingerprint(self, fingerprint: str) -> Optional[Mapping[str, Any]]:
        record = self.index.get_by_fingerprint(fingerprint)
        return None if record is None else _read_only(record)

    def append(
        self, record: Mapping[str, Any], *, packed_vector: Optional[bytes] = None
    ) -> None:
        """Atomically insert one *new* record (and optional shared vector).

        A duplicate is an error, not a re-exposure update. Exact repeat and
        metadata changes belong in an explicit edit transaction.
        """

        identifier, fingerprint, payload, digest = _record_payload(record)
        vector = (
            _packed_payload(packed_vector, self.index.dimensions)[0]
            if packed_vector is not None else None
        )
        self.index._reserve_memory(
            4096 + 3 * (len(payload) + (len(vector) if vector else 0)),
            "assembly view append buffer",
        )
        revision: Optional[int] = None
        with self.index._transaction(write=True) as connection:
            collision = connection.execute(
                "SELECT 1 FROM assembly_records WHERE assembly_id=? OR fingerprint=? "
                "LIMIT 1",
                (identifier, fingerprint),
            ).fetchone()
            if collision is not None:
                raise ValueError("assembly append requires a new id and fingerprint")
            created, revision = self.index._upsert_in_connection(
                connection, identifier, fingerprint, payload, digest, vector, None
            )
            if not created:
                raise ValueError("assembly append did not create a record")
        if revision is not None and self.index._vectors is not None:
            self.index._vectors.invalidate(identifier, revision)

    def transaction(
        self, *, max_rows: int = 128, max_payload_bytes: int = 8 * 1024 * 1024
    ) -> "PagedAssemblyEdit":
        """Open a bounded atomic overlay for *existing* metadata records."""

        return PagedAssemblyEdit(
            self.index, max_rows=max_rows, max_payload_bytes=max_payload_bytes
        )


class PagedAssemblyEdit:
    """Explicit bounded dirty-record transaction, with no hidden writeback.

    ``edit_by_id`` gives the callback a private mutable copy. The copy is
    validated, privacy-projected and staged immediately after the callback;
    any reference retained by the callback cannot alter the staged value.
    No read method returns a mutable overlay object. A failed edit makes the
    whole transaction abort-only, even if its exception is caught inside the
    ``with`` block. Commit uses the index's bounded batch and reserve hooks.
    """

    def __init__(
        self, index: PagedAssemblyIndex, *, max_rows: int, max_payload_bytes: int
    ) -> None:
        self.index = index
        self._batch: PagedAssemblyBatch = index.batch(
            max_rows=max_rows, max_payload_bytes=max_payload_bytes
        )
        self.max_rows = max_rows
        self.max_payload_bytes = max_payload_bytes
        self._dirty: OrderedDict[str, tuple[bytes, dict[str, Any], int]] = OrderedDict()
        self._used_bytes = 0
        self._active = False
        self._closed = False
        self._aborted = False

    @property
    def pending_count(self) -> int:
        return len(self._dirty)

    def __enter__(self) -> "PagedAssemblyEdit":
        if self._active or self._closed:
            raise RuntimeError("assembly edit transaction cannot be reopened")
        self._batch.__enter__()
        self._active = True
        return self

    def _connection(self) -> sqlite3.Connection:
        if not self._active or self._aborted:
            raise RuntimeError("assembly edit transaction is not active")
        return self._batch._connection_or_raise()

    def _load(self, column: str, key: str) -> Optional[tuple[dict[str, Any], bytes]]:
        try:
            connection = self._connection()
            if not isinstance(key, str) or not key:
                return None
            if column not in {"assembly_id", "fingerprint"}:
                raise ValueError("unsupported assembly lookup column")
            size = connection.execute(
                "SELECT sequence,LENGTH(record_json) FROM assembly_records "
                "WHERE %s=?" % column,
                (key,),
            ).fetchone()
            if size is None:
                return None
            self.index._reserve_memory(
                4096 + 3 * int(size[1]), "assembly edit record read"
            )
            row = connection.execute(
                "SELECT sequence,assembly_id,fingerprint,record_json,record_sha256 "
                "FROM assembly_records WHERE sequence=?",
                (size[0],),
            ).fetchone()
            if row is None:
                raise ValueError("assembly metadata changed during edit")
            record = self.index._decode_record(row)
            staged = self._dirty.get(record["id"])
            return (staged[1], staged[0]) if staged is not None else (record, row[3])
        except BaseException:
            self._aborted = True
            raise

    def get_by_id(self, assembly_id: str) -> Optional[Mapping[str, Any]]:
        loaded = self._load("assembly_id", assembly_id)
        return None if loaded is None else _read_only(loaded[0])

    def get_by_fingerprint(self, fingerprint: str) -> Optional[Mapping[str, Any]]:
        loaded = self._load("fingerprint", fingerprint)
        return None if loaded is None else _read_only(loaded[0])

    def edit_by_id(
        self, assembly_id: str, mutator: Callable[[dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        """Stage one validated in-place metadata change by stable assembly ID."""

        try:
            if not callable(mutator):
                raise TypeError("assembly edit mutator must be callable")
            loaded = self._load("assembly_id", assembly_id)
            if loaded is None:
                raise KeyError(assembly_id)
            current, previous_payload = loaded
            # JSON round-trip detaches nested provenance and member arrays.
            self.index._reserve_memory(
                4096 + 6 * len(previous_payload), "assembly edit working copy"
            )
            working = json.loads(json.dumps(current, ensure_ascii=False))
            result = mutator(working)
            if result is not None:
                raise ValueError("assembly edit mutator must change its copy in place")
            identifier, fingerprint, payload, _digest = _record_payload(working)
            if identifier != current["id"] or fingerprint != current["fingerprint"]:
                raise ValueError("assembly edit cannot change id or fingerprint")
            prior = self._dirty.get(identifier)
            baseline_payload = prior[0] if prior is not None else previous_payload
            prior_charge = prior[2] if prior is not None else 0
            if payload == baseline_payload:
                if prior is not None:
                    del self._dirty[identifier]
                    self._used_bytes -= prior_charge
                return _read_only(json.loads(payload))
            charge = 4 * (len(payload) + len(identifier) + len(fingerprint)) + 128
            if prior is None and len(self._dirty) >= self.max_rows:
                raise AssemblyIndexResourcePause("assembly edit row window reached")
            if self._used_bytes - prior_charge + charge > self.max_payload_bytes:
                raise AssemblyIndexResourcePause("assembly edit byte window reached")
            if charge > prior_charge:
                self.index._reserve_memory(
                    charge - prior_charge, "assembly edit dirty overlay"
                )
            staged = json.loads(payload)
            self._dirty[identifier] = (baseline_payload, staged, charge)
            self._used_bytes = self._used_bytes - prior_charge + charge
            return _read_only(staged)
        except BaseException:
            self._aborted = True
            raise

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        if not self._active:
            raise RuntimeError("assembly edit transaction was not opened")
        self._active = False
        self._closed = True
        try:
            if error_type is not None or self._aborted:
                self._batch.__exit__(error_type or RuntimeError, error, traceback)
                if error_type is None:
                    raise RuntimeError("assembly edit transaction was aborted")
                return False
            try:
                for _baseline, record, _charge in self._dirty.values():
                    self._batch.upsert(record)
            except BaseException as commit_error:
                self._batch.__exit__(
                    type(commit_error), commit_error, commit_error.__traceback__
                )
                raise
            self._batch.__exit__(None, None, None)
            return False
        finally:
            self._dirty.clear()
            self._used_bytes = 0
