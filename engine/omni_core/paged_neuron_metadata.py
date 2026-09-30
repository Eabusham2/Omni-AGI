"""Bounded, checked SQLite working cache for neural neuron metadata.

The committed v3 shard generation remains authoritative. This table is a
derived/live working cache, not a second learned vector store. Returned
records are recursively read-only: mutations must use ``edit_by_id`` or a
bounded batch so updates cannot disappear when a page is evicted.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import closing, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from .paged_store_counts import ensure_row_count, row_count


_FIELDS = frozenset({
    "id", "neuron_id", "label", "region", "activation", "importance",
    "uncertainty", "exposures", "created_at", "last_activated_at", "aliases",
    "memory_strength", "retention_score", "activity_score",
    "plasticity_score", "reinforcement_drive", "unfinished_score",
    "unfinished", "settling_signals", "last_settled_at",
})
_NUMBER_FIELDS = frozenset({
    "activation", "importance", "uncertainty", "created_at",
    "last_activated_at", "memory_strength", "retention_score",
    "activity_score", "plasticity_score", "reinforcement_drive",
    "unfinished_score",
})
_FORMAT = "omni-paged-neuron-metadata"
_VERSION = 2
_ROW_FIELDS = (
    "sequence,neuron_id,record_json,record_sha256,"
    "decay_log_anchor,decay_sum_anchor,decay_zero_anchor"
)


def _row_digest(
    encoded: bytes, anchor_log: float, anchor_sum: float, anchor_zero: int
) -> str:
    return hashlib.sha256(
        encoded + b"\0" + _canonical([anchor_log, anchor_sum, anchor_zero])
    ).hexdigest()


class NeuronMetadataResourcePause(RuntimeError):
    """A configured bounded memory/disk reserve refused a page or edit."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _payload(record: Mapping[str, Any]) -> tuple[str, bytes, str]:
    if not isinstance(record, Mapping):
        raise ValueError("neuron metadata must be a mapping")
    data = dict(record)
    identifier = data.get("id")
    if (
        not isinstance(identifier, str) or not identifier
        or len(identifier) > 512
        or any(ord(char) < 33 or ord(char) == 127 for char in identifier)
        or data.get("neuron_id", identifier) != identifier
        or set(data) - _FIELDS
    ):
        raise ValueError("neuron metadata identity or fields are invalid")
    for name, maximum in (("label", 256), ("region", 64), ("last_settled_at", 128)):
        if name in data and (
            not isinstance(data[name], str) or len(data[name]) > maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in data[name])
        ):
            raise ValueError("neuron metadata text field is unsafe")
    aliases = data.get("aliases", [])
    if (
        not isinstance(aliases, (list, tuple)) or len(aliases) > 32
        or any(
            not isinstance(item, str) or len(item) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in item)
            for item in aliases
        )
    ):
        raise ValueError("neuron aliases are unsafe")
    if isinstance(aliases, tuple):
        data["aliases"] = list(aliases)
    for name in _NUMBER_FIELDS & data.keys():
        value = data[name]
        if (
            isinstance(value, (bool, str, bytes))
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("neuron numeric metadata is invalid")
    if "exposures" in data and (
        type(data["exposures"]) is not int or data["exposures"] < 0
    ):
        raise ValueError("neuron exposures are invalid")
    if "unfinished" in data and type(data["unfinished"]) is not bool:
        raise ValueError("neuron unfinished marker is invalid")
    signals = data.get("settling_signals")
    if signals is not None:
        if (
            not isinstance(signals, Mapping) or len(signals) > 64
            or any(
                not isinstance(key, str) or not key or len(key) > 64
                or isinstance(value, (bool, str, bytes))
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for key, value in signals.items()
            )
        ):
            raise ValueError("neuron settling signals are invalid")
        data["settling_signals"] = dict(signals)
    encoded = _canonical(data)
    if len(encoded) > 64 * 1024:
        raise ValueError("neuron metadata row exceeds bounded record size")
    return identifier, encoded, hashlib.sha256(encoded).hexdigest()


class PagedNeuronMetadata(MutableMapping[str, Mapping[str, Any]]):
    """Exact ID map with bounded pages and no corpus-sized Python mirror."""

    def __init__(
        self,
        path: Path,
        *,
        resource_policy: Optional[Any] = None,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        memory_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> None:
        if resource_policy is not None and disk_reserve is not None:
            raise ValueError("provide either resource_policy or disk_reserve")
        self.path = Path(path)
        self._disk_reserve = (
            resource_policy.require_disk if resource_policy is not None else disk_reserve
        )
        self._memory_reserve = memory_reserve
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                tables = {name for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                has_records = "paged_neuron_records" in tables
                has_meta = "paged_neuron_meta" in tables
                if has_records != has_meta:
                    raise ValueError("paged neuron metadata schema is incomplete")
                if not has_records:
                    self._reserve_disk(128 * 1024, "paged neuron metadata schema")
                    connection.execute(
                        "CREATE TABLE paged_neuron_records ("
                        "sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "neuron_id TEXT NOT NULL UNIQUE,"
                        "record_json BLOB NOT NULL,"
                        "record_sha256 TEXT NOT NULL,"
                        "decay_log_anchor REAL NOT NULL,"
                        "decay_sum_anchor REAL NOT NULL,"
                        "decay_zero_anchor INTEGER NOT NULL)"
                    )
                    connection.execute(
                        "CREATE TABLE paged_neuron_meta ("
                        "key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID"
                    )
                    for key, value in (
                        ("format", _FORMAT), ("version", str(_VERSION)),
                        ("store_id", uuid.uuid4().hex), ("revision", "0"),
                        ("graph_revision", "0"),
                        ("committed_generation_sha256", ""),
                        ("committed_revision", "-1"),
                        ("decay_epoch", "0"),
                        ("decay_log_factor", "0.0"),
                        ("decay_uncertainty_sum", "0.0"),
                        ("decay_zero_generation", "0"),
                    ):
                        connection.execute(
                            "INSERT INTO paged_neuron_meta(key,value) VALUES (?,?)",
                            (key, value),
                        )
                elif self._meta(connection, "format") != _FORMAT or (
                    self._meta(connection, "version") != str(_VERSION)
                ):
                    raise ValueError("paged neuron metadata format is incompatible")
                else:
                    columns = {
                        item[1] for item in connection.execute(
                            "PRAGMA table_info(paged_neuron_records)"
                        )
                    }
                    if not {
                        "sequence", "neuron_id", "record_json", "record_sha256",
                        "decay_log_anchor", "decay_sum_anchor", "decay_zero_anchor",
                    } <= columns:
                        raise ValueError("paged neuron decay-anchor schema is incomplete")
                    self._decay_state(connection)
                ensure_row_count(connection, "paged_neuron_records", self._reserve_disk)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        from .paged_idle_selection import observe_idle_source_connection
        observe_idle_source_connection(connection, self.path)
        return connection

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _reserve(self, callback: Optional[Callable[[int, str], Any]], size: int, operation: str) -> None:
        if callback is not None and callback(max(1, int(size)), operation) is False:
            raise NeuronMetadataResourcePause("%s paused at resource reserve" % operation)

    def _reserve_disk(self, size: int, operation: str) -> None:
        self._reserve(self._disk_reserve, size, operation)

    def _reserve_memory(self, size: int, operation: str) -> None:
        self._reserve(self._memory_reserve, size, operation)

    @staticmethod
    def _meta(connection: sqlite3.Connection, key: str) -> str:
        value = connection.execute(
            "SELECT value FROM paged_neuron_meta WHERE key=?", (key,)
        ).fetchone()
        if value is None or not isinstance(value[0], str):
            raise ValueError("paged neuron metadata identity is missing")
        return value[0]

    @classmethod
    def _revision(cls, connection: sqlite3.Connection) -> int:
        value = cls._meta(connection, "revision")
        if not value.isdecimal() or str(int(value)) != value:
            raise ValueError("paged neuron metadata revision is invalid")
        return int(value)

    @classmethod
    def _decay_state(
        cls, connection: sqlite3.Connection
    ) -> tuple[int, float, float, int]:
        try:
            epoch = int(cls._meta(connection, "decay_epoch"))
            log_factor = float(cls._meta(connection, "decay_log_factor"))
            uncertainty_sum = float(cls._meta(connection, "decay_uncertainty_sum"))
            zero_generation = int(cls._meta(connection, "decay_zero_generation"))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("paged neuron decay epoch is invalid") from error
        if (
            epoch < 0 or zero_generation < 0
            or not math.isfinite(log_factor) or log_factor > 0.0
            or not math.isfinite(uncertainty_sum) or uncertainty_sum < 0.0
        ):
            raise ValueError("paged neuron decay epoch is invalid")
        return epoch, log_factor, uncertainty_sum, zero_generation

    @staticmethod
    def _anchors(state: tuple[int, float, float, int]) -> tuple[float, float, int]:
        return state[1], state[2], state[3]

    @property
    def graph_revision(self) -> int:
        with self._transaction() as connection:
            value = self._meta(connection, "graph_revision")
        if not value.isdecimal() or str(int(value)) != value:
            raise ValueError("paged neuron graph revision is invalid")
        return int(value)

    @classmethod
    def _advance(cls, connection: sqlite3.Connection, *, structural: bool) -> None:
        connection.execute(
            "UPDATE paged_neuron_meta SET value=? WHERE key='revision'",
            (str(cls._revision(connection) + 1),),
        )
        if structural:
            prior = cls._meta(connection, "graph_revision")
            connection.execute(
                "UPDATE paged_neuron_meta SET value=? WHERE key='graph_revision'",
                (str(int(prior) + 1),),
            )

    def _notify_membership(self, identifiers: tuple[str, ...], revision: int) -> None:
        # This is an optional, process-owned execution index, not recovery
        # authority. Its callback catches failure and marks itself invalid;
        # committed source mutations never depend on a cache write succeeding.
        observer = getattr(self, "_recall_graph_observer", None)
        if callable(observer):
            observer(identifiers, revision)

    @staticmethod
    def _decode(
        row: tuple[Any, ...], state: tuple[int, float, float, int]
    ) -> dict[str, Any]:
        sequence, identifier, encoded, digest, anchor_log, anchor_sum, anchor_zero = row
        if (
            type(sequence) is not int or sequence < 1
            or not isinstance(identifier, str) or not identifier
            or not isinstance(encoded, bytes) or not isinstance(digest, str)
            or not isinstance(anchor_log, (float, int))
            or not isinstance(anchor_sum, (float, int))
            or type(anchor_zero) is not int or anchor_zero < 0
            or not math.isfinite(float(anchor_log))
            or not math.isfinite(float(anchor_sum))
            or _row_digest(encoded, float(anchor_log), float(anchor_sum), anchor_zero)
            != digest
        ):
            raise ValueError("paged neuron row checksum or identity mismatch")
        try:
            value = json.loads(encoded)
        except (TypeError, UnicodeError, ValueError) as error:
            raise ValueError("paged neuron row JSON is invalid") from error
        if not isinstance(value, dict) or value.get("id") != identifier:
            raise ValueError("paged neuron row identity mismatch")
        _identifier, validated, _digest = _payload(value)
        if validated != encoded:
            raise ValueError("paged neuron row is not canonical")
        _epoch, global_log, global_sum, global_zero = state
        if (
            anchor_zero > global_zero
            or float(anchor_sum) > global_sum + 1e-9
            or (anchor_zero == global_zero and float(anchor_log) < global_log - 1e-9)
        ):
            raise ValueError("paged neuron decay anchor is ahead of its epoch")
        if anchor_zero != global_zero and "activation" in value:
            value["activation"] = 0.0
        elif "activation" in value:
            value["activation"] = float(value["activation"]) * math.exp(
                min(0.0, global_log - float(anchor_log))
            )
        if "uncertainty" in value:
            value["uncertainty"] = min(
                1.0,
                float(value["uncertainty"])
                + max(0.0, global_sum - float(anchor_sum))
                / (1.0 + int(value.get("exposures", 0))),
            )
        return value

    def decay(self, amount: float) -> int:
        """Advance one durable global decay epoch without touching row pages."""

        if (
            isinstance(amount, (bool, str, bytes))
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or not 0.0 <= float(amount) <= 1.0
        ):
            raise ValueError("paged neuron decay amount must be in [0,1]")
        if amount == 0.0:
            with self._transaction() as connection:
                return self._decay_state(connection)[0]
        self._reserve_disk(64 * 1024, "paged neuron decay epoch")
        with self._transaction(write=True) as connection:
            epoch, log_factor, uncertainty_sum, zero_generation = (
                self._decay_state(connection)
            )
            next_sum = uncertainty_sum + float(amount)
            if not math.isfinite(next_sum):
                raise NeuronMetadataResourcePause("paged neuron decay epoch exhausted")
            if float(amount) == 1.0:
                next_log = 0.0
                zero_generation += 1
            else:
                next_log = log_factor + math.log1p(-float(amount))
            for key, value in (
                ("decay_epoch", str(epoch + 1)),
                ("decay_log_factor", repr(next_log)),
                ("decay_uncertainty_sum", repr(next_sum)),
                ("decay_zero_generation", str(zero_generation)),
            ):
                connection.execute(
                    "UPDATE paged_neuron_meta SET value=? WHERE key=?",
                    (value, key),
                )
            self._advance(connection, structural=False)
            return epoch + 1

    def __len__(self) -> int:
        with self._transaction() as connection:
            return row_count(connection, "paged_neuron_records")

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._transaction() as connection:
            return connection.execute(
                "SELECT 1 FROM paged_neuron_records WHERE neuron_id=?", (key,)
            ).fetchone() is not None

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        with self._transaction() as connection:
            state = self._decay_state(connection)
            size = connection.execute(
                "SELECT sequence,LENGTH(record_json) FROM paged_neuron_records "
                "WHERE neuron_id=?", (key,),
            ).fetchone()
            if size is None:
                raise KeyError(key)
            self._reserve_memory(4096 + 4 * int(size[1]), "paged neuron read")
            row = connection.execute(
                "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                "WHERE sequence=?", (size[0],),
            ).fetchone()
            if row is None:
                raise ValueError("paged neuron record changed during read")
            return _freeze(self._decode(row, state))

    def __setitem__(self, key: str, value: Mapping[str, Any]) -> None:
        identifier, encoded, _digest = _payload(value)
        if key != identifier:
            raise ValueError("paged neuron mapping key differs from record id")
        self._reserve_memory(4096 + 4 * len(encoded), "paged neuron write buffer")
        self._reserve_disk(64 * 1024 + 4 * len(encoded), "paged neuron upsert")
        with self._transaction(write=True) as connection:
            state = self._decay_state(connection)
            anchor_log, anchor_sum, anchor_zero = self._anchors(state)
            digest = _row_digest(encoded, anchor_log, anchor_sum, anchor_zero)
            current = connection.execute(
                "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                "WHERE neuron_id=?", (key,),
            ).fetchone()
            if current is not None:
                self._decode(current, state)
                if current[2] == encoded and current[3] == digest:
                    return
                connection.execute(
                    "UPDATE paged_neuron_records SET record_json=?,record_sha256=?,"
                    "decay_log_anchor=?,decay_sum_anchor=?,decay_zero_anchor=? "
                    "WHERE neuron_id=?",
                    (encoded, digest, anchor_log, anchor_sum, anchor_zero, key),
                )
            else:
                connection.execute(
                    "INSERT INTO paged_neuron_records "
                    "(neuron_id,record_json,record_sha256,decay_log_anchor,"
                    "decay_sum_anchor,decay_zero_anchor) VALUES (?,?,?,?,?,?)",
                    (key, encoded, digest, anchor_log, anchor_sum, anchor_zero),
                )
            self._advance(connection, structural=current is None)
            graph_revision = int(self._meta(connection, "graph_revision"))
        if current is None:
            self._notify_membership((key,), graph_revision)

    def __delitem__(self, key: str) -> None:
        self._reserve_disk(64 * 1024, "paged neuron delete")
        with self._transaction(write=True) as connection:
            state = self._decay_state(connection)
            current = connection.execute(
                "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                "WHERE neuron_id=?", (key,),
            ).fetchone()
            if current is None:
                raise KeyError(key)
            self._decode(current, state)
            connection.execute(
                "DELETE FROM paged_neuron_records WHERE neuron_id=?", (key,)
            )
            self._advance(connection, structural=True)
            graph_revision = int(self._meta(connection, "graph_revision"))
        self._notify_membership((key,), graph_revision)

    def edit_by_id(
        self, key: str, mutator: Callable[[dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        """Commit one detached record mutation atomically, or do nothing."""

        if not callable(mutator):
            raise TypeError("neuron mutator must be callable")
        with self._transaction(write=True) as connection:
            state = self._decay_state(connection)
            current = connection.execute(
                "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                "WHERE neuron_id=?", (key,),
            ).fetchone()
            if current is None:
                raise KeyError(key)
            record = self._decode(current, state)
            mutator(record)
            identifier, encoded, _digest = _payload(record)
            if identifier != key:
                raise ValueError("neuron edit changed its identity")
            self._reserve_memory(4096 + 4 * len(encoded), "paged neuron edit buffer")
            anchor_log, anchor_sum, anchor_zero = self._anchors(state)
            digest = _row_digest(encoded, anchor_log, anchor_sum, anchor_zero)
            if encoded != current[2] or digest != current[3]:
                self._reserve_disk(64 * 1024 + 4 * len(encoded), "paged neuron edit")
                connection.execute(
                    "UPDATE paged_neuron_records SET record_json=?,record_sha256=?,"
                    "decay_log_anchor=?,decay_sum_anchor=?,decay_zero_anchor=? "
                    "WHERE neuron_id=?",
                    (encoded, digest, anchor_log, anchor_sum, anchor_zero, key),
                )
                self._advance(connection, structural=False)
            return _freeze(record)

    def import_page(
        self,
        records: list[Mapping[str, Any]],
        *,
        max_rows: int = 256,
        max_payload_bytes: int = 8 * 1024 * 1024,
    ) -> int:
        """Insert one bounded shard in one WAL transaction, with no overwrite."""

        if (
            not isinstance(records, list)
            or type(max_rows) is not int or not 1 <= max_rows <= 4096
            or type(max_payload_bytes) is not int or max_payload_bytes < 1
            or len(records) > max_rows
        ):
            raise ValueError("paged neuron import window is invalid")
        prepared = [_payload(record) for record in records]
        if len({identifier for identifier, _, _ in prepared}) != len(prepared):
            raise ValueError("paged neuron import repeats an ID")
        charge = sum(4 * len(encoded) + len(identifier) + 128
                     for identifier, encoded, _digest in prepared)
        if charge > max_payload_bytes:
            raise ValueError("paged neuron import exceeds bounded byte window")
        if not prepared:
            return 0
        self._reserve_memory(4096 + charge, "paged neuron import buffer")
        self._reserve_disk(64 * 1024 + 2 * charge, "paged neuron import")
        with self._transaction(write=True) as connection:
            anchor_log, anchor_sum, anchor_zero = self._anchors(
                self._decay_state(connection)
            )
            for identifier, encoded, _digest in prepared:
                if connection.execute(
                    "SELECT 1 FROM paged_neuron_records WHERE neuron_id=?",
                    (identifier,),
                ).fetchone() is not None:
                    raise ValueError("paged neuron import collides with an existing ID")
                connection.execute(
                    "INSERT INTO paged_neuron_records "
                    "(neuron_id,record_json,record_sha256,decay_log_anchor,"
                    "decay_sum_anchor,decay_zero_anchor) VALUES (?,?,?,?,?,?)",
                    (
                        identifier, encoded,
                        _row_digest(encoded, anchor_log, anchor_sum, anchor_zero),
                        anchor_log, anchor_sum, anchor_zero,
                    ),
                )
            self._advance(connection, structural=True)
            graph_revision = int(self._meta(connection, "graph_revision"))
        self._notify_membership(tuple(identifier for identifier, _encoded, _digest in prepared), graph_revision)
        return len(prepared)

    def iter_pages(
        self, page_size: int = 128
    ) -> Iterator[tuple[Mapping[str, Any], ...]]:
        if type(page_size) is not int or not 1 <= page_size <= 4096:
            raise ValueError("paged neuron page size is invalid")
        with self._transaction() as connection:
            store_id = self._meta(connection, "store_id")
            revision = self._revision(connection)
            decay_state = self._decay_state(connection)
            count = row_count(connection, "paged_neuron_records")
            through = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM paged_neuron_records"
            ).fetchone()[0]
        after = 0
        seen = 0
        while True:
            with self._transaction() as connection:
                if (
                    self._meta(connection, "store_id") != store_id
                    or self._revision(connection) != revision
                    or self._decay_state(connection) != decay_state
                ):
                    raise ValueError("paged neuron iteration generation drift")
                sizes = connection.execute(
                    "SELECT sequence,LENGTH(record_json) FROM paged_neuron_records "
                    "WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT ?",
                    (after, through, page_size),
                ).fetchall()
                if not sizes:
                    if seen != count:
                        raise ValueError("paged neuron iteration coverage mismatch")
                    return
                self._reserve_memory(
                    4096 + 4 * sum(int(size) for _, size in sizes),
                    "paged neuron page",
                )
                last = int(sizes[-1][0])
                rows = connection.execute(
                    "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                    "WHERE sequence>? AND sequence<=? ORDER BY sequence",
                    (after, last),
                ).fetchall()
                if [row[0] for row in rows] != [item[0] for item in sizes]:
                    raise ValueError("paged neuron page changed during read")
                page = tuple(_freeze(self._decode(row, decay_state)) for row in rows)
            seen += len(page)
            after = last
            yield page

    def __iter__(self) -> Iterator[str]:
        for page in self.iter_pages():
            for record in page:
                yield str(record["id"])

    def items(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        for page in self.iter_pages():
            for record in page:
                yield str(record["id"]), record

    def values(self) -> Iterator[Mapping[str, Any]]:
        for page in self.iter_pages():
            yield from page

    def activity_metrics(
        self,
        *,
        active_ids: Optional[set[str] | frozenset[str]] = None,
        legacy_raw_active: bool = True,
        threshold: float = 0.1,
        sample_rows: int = 512,
    ) -> dict[str, Any]:
        """Measure chat-facing activity without materializing every neuron.

        Up to ``sample_rows`` indexed, evenly spaced rows are decoded under
        one SQLite read snapshot. The result is exact for a small store and
        explicitly estimated for a larger one. A bounded attention overlay
        can give an exact active count even when mean uncertainty is sampled.
        This is a measurement budget, never a cap on learned neurons.
        """

        if (
            type(sample_rows) is not int or not 1 <= sample_rows <= 4096
            or isinstance(threshold, (bool, str, bytes))
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
            or type(legacy_raw_active) is not bool
            or (not legacy_raw_active and active_ids is None)
            or (
                active_ids is not None
                and not isinstance(active_ids, (set, frozenset))
            )
        ):
            raise ValueError("paged neuron activity measurement options are invalid")
        if active_ids is not None and any(
            not isinstance(identifier, str) for identifier in active_ids
        ):
            raise ValueError("active neuron IDs must be strings")
        with self._transaction() as connection:
            state = self._decay_state(connection)
            revision = self._revision(connection)
            count = row_count(connection, "paged_neuron_records")
            high_water = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM paged_neuron_records"
            ).fetchone()[0]
            count, high_water = int(count), int(high_water)
            if count == 0:
                return {
                    "rowCount": 0,
                    "meanUncertainty": 0.5,
                    "activeCount": 0,
                    "activeFraction": 0.0,
                    "activeCountExact": True,
                    "estimated": False,
                    "sampledRows": 0,
                    "revision": revision,
                    "decayEpoch": state[0],
                }

            sampled = count > sample_rows
            if sampled:
                positions = (
                    1 + (index * high_water) // sample_rows
                    for index in range(sample_rows)
                )
                rows = (
                    connection.execute(
                        "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                        "WHERE sequence>=? ORDER BY sequence LIMIT 1",
                        (position,),
                    ).fetchone()
                    for position in positions
                )
            else:
                rows = connection.execute(
                    "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                    "ORDER BY sequence"
                )
            seen_sequences: set[int] = set()
            uncertainty_total = 0.0
            sampled_active = 0
            sample_count = 0
            for row in rows:
                if row is None or row[0] in seen_sequences:
                    continue
                seen_sequences.add(int(row[0]))
                self._reserve_memory(
                    4096 + 4 * len(row[2]), "paged neuron activity sample"
                )
                record = self._decode(row, state)
                uncertainty_total += float(record.get("uncertainty", 0.5))
                if (
                    (legacy_raw_active or active_ids is None or row[1] in active_ids)
                    and float(record.get("activation", 0.0)) >= threshold
                ):
                    sampled_active += 1
                sample_count += 1
            if sample_count == 0:
                raise ValueError("paged neuron activity sample has no rows")
            if not sampled and sample_count != count:
                raise ValueError("paged neuron activity coverage mismatch")

            exact_active = False
            if not legacy_raw_active and sampled and (
                active_ids is not None and len(active_ids) <= sample_rows
            ):
                active_count = 0
                for identifier in active_ids:
                    row = connection.execute(
                        "SELECT " + _ROW_FIELDS + " FROM paged_neuron_records "
                        "WHERE neuron_id=?", (identifier,),
                    ).fetchone()
                    if row is None:
                        continue
                    self._reserve_memory(
                        4096 + 4 * len(row[2]),
                        "paged neuron active overlay",
                    )
                    if float(self._decode(row, state).get("activation", 0.0)) >= threshold:
                        active_count += 1
                exact_active = True
            elif not sampled:
                active_count = sampled_active
                exact_active = True
            else:
                active_count = min(
                    count,
                    max(0, round(sampled_active * count / sample_count)),
                )
            return {
                "rowCount": count,
                "meanUncertainty": uncertainty_total / sample_count,
                "activeCount": active_count,
                "activeFraction": active_count / count,
                "activeCountExact": exact_active,
                "estimated": sampled or not exact_active,
                "sampledRows": sample_count,
                "revision": revision,
                "decayEpoch": state[0],
            }

    def status(self) -> dict[str, Any]:
        with self._transaction() as connection:
            count = row_count(connection, "paged_neuron_records")
            high_water = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM paged_neuron_records"
            ).fetchone()[0]
            revision = self._revision(connection)
            committed_generation = self._meta(
                connection, "committed_generation_sha256"
            )
            committed_revision = self._meta(connection, "committed_revision")
            decay_state = self._decay_state(connection)
            return {
                "format": _FORMAT,
                "formatVersion": _VERSION,
                "storeId": self._meta(connection, "store_id"),
                "revision": revision,
                "graphRevision": int(self._meta(connection, "graph_revision")),
                "decayEpoch": decay_state[0],
                "rowCount": int(count),
                "highWaterSequence": int(high_water),
                "committedGenerationSha256": committed_generation or None,
                "dirtySinceCommit": committed_revision != str(revision),
            }

    def bind_committed_generation(self, generation_sha256: str) -> None:
        if (
            not isinstance(generation_sha256, str)
            or len(generation_sha256) != 64
            or any(char not in "0123456789abcdef" for char in generation_sha256)
        ):
            raise ValueError("paged neuron committed generation is invalid")
        self._reserve_disk(64 * 1024, "paged neuron generation binding")
        with self._transaction(write=True) as connection:
            for key, value in (
                ("committed_generation_sha256", generation_sha256),
                ("committed_revision", str(self._revision(connection))),
            ):
                connection.execute(
                    "UPDATE paged_neuron_meta SET value=? WHERE key=?",
                    (value, key),
                )
