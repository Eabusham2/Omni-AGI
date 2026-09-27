"""Generation-bound SQLite dirty-ID journal for a shared paged substrate.

SQLite triggers record neuron metadata, packed-row, and assembly mutations in
the *same transaction* as their source row. The base v3 shard membership is
stored on disk, not in a corpus-sized Python map. This is an incremental-save
primitive, not a second checkpoint authority: a caller must still publish a
complete checksummed v3 generation, commit ``brain.json``, and rebase the
journal before treating dirty IDs as clean. Until the writer does that, it
must keep its full-scan fallback or fail closed; it must never silently use a
stale or partially built journal to skip neural records.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from .vsa import NeuralSubstrate


_FORMAT = "omni-paged-substrate-dirty-journal"
_VERSION = 1
_MAX_RECORD_BLOB = 64 * 1024 * 1024
_KINDS = ("neurons", "assemblies")
_TABLES = {
    "neurons": ("paged_neuron_records", "neuron_id"),
    "assemblies": ("assembly_records", "assembly_id"),
    "vectors": ("paged_vector_rows", "vector_id"),
}


class DirtyJournalResourcePause(RuntimeError):
    """The disk reserve refused journal initialization or planning."""


@dataclass(frozen=True)
class DirtyGroup:
    kind: str
    bucket: str
    part: int
    record_ids: tuple[str, ...]


def _reserve(callback: Optional[Callable[[int, str], Any]], size: int, label: str) -> None:
    if callback is not None and callback(max(1, int(size)), label) is False:
        raise DirtyJournalResourcePause("paged dirty journal reached disk reserve")


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _meta(connection: sqlite3.Connection, table: str, key: str) -> str:
    row = connection.execute(
        "SELECT value FROM %s WHERE key=?" % table, (key,)
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        raise ValueError("paged store committed binding is incomplete")
    return row[0]


def _bindings(connection: sqlite3.Connection, generation: str) -> tuple[str, str, str]:
    """Require a clean, committed three-store boundary before installing."""

    for table in (
        "index_metadata", "paged_vector_meta", "paged_neuron_meta",
        "assembly_records", "paged_vector_rows", "paged_neuron_records",
    ):
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is None:
            raise ValueError("dirty journal needs one shared three-store SQLite cache")
    bindings = (
        ("index_metadata", "index_revision", "committed_index_revision"),
        ("paged_vector_meta", "revision", "committed_revision"),
        ("paged_neuron_meta", "revision", "committed_revision"),
    )
    for table, current_key, committed_key in bindings:
        if (
            _meta(connection, table, "committed_generation_sha256") != generation
            or _meta(connection, table, current_key)
            != _meta(connection, table, committed_key)
        ):
            raise ValueError("paged stores are not cleanly bound to prior generation")
    if _meta(connection, "index_metadata", "committed_vector_revision") != _meta(
        connection, "paged_vector_meta", "revision"
    ):
        raise ValueError("assembly/vector committed revisions disagree")
    return (
        _meta(connection, "index_metadata", "store_id"),
        _meta(connection, "paged_vector_meta", "store_id"),
        _meta(connection, "paged_neuron_meta", "store_id"),
    )


def _trigger_sql(name: str, table: str, column: str, kind: str, event: str) -> str:
    old = "OLD" if event == "DELETE" else "NEW"
    statements = [
        "INSERT OR IGNORE INTO substrate_dirty_ids(kind,record_id) "
        "VALUES('%s',%s.%s);" % (kind, old, column)
    ]
    if event == "UPDATE":
        statements.insert(0,
            "INSERT OR IGNORE INTO substrate_dirty_ids(kind,record_id) "
            "VALUES('%s',OLD.%s);" % (kind, column)
        )
    return "CREATE TRIGGER %s AFTER %s ON %s BEGIN %s END" % (
        name, event, table, " ".join(statements)
    )


def _triggers() -> dict[str, str]:
    result: dict[str, str] = {}
    for source, (table, column) in _TABLES.items():
        kind = "neurons" if source in {"neurons", "vectors"} else "assemblies"
        for event in ("INSERT", "UPDATE", "DELETE"):
            name = "substrate_dirty_%s_%s" % (source, event.lower())
            result[name] = _trigger_sql(name, table, column, kind, event)
    result["substrate_dirty_neuron_decay_update"] = (
        "CREATE TRIGGER substrate_dirty_neuron_decay_update "
        "AFTER UPDATE OF value ON paged_neuron_meta "
        "WHEN NEW.key='decay_epoch' AND NEW.value!=OLD.value BEGIN "
        "UPDATE substrate_dirty_meta SET full_neurons_dirty=1 WHERE singleton=1; END"
    )
    result["substrate_dirty_neuron_decay_insert"] = (
        "CREATE TRIGGER substrate_dirty_neuron_decay_insert "
        "AFTER INSERT ON paged_neuron_meta "
        "WHEN NEW.key='decay_epoch' AND NEW.value!='0' BEGIN "
        "UPDATE substrate_dirty_meta SET full_neurons_dirty=1 WHERE singleton=1; END"
    )
    return result


def _check_triggers(connection: sqlite3.Connection) -> None:
    for name, expected in _triggers().items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
        ).fetchone()
        if row is None or row[0] != expected:
            raise ValueError("paged dirty journal trigger is missing or altered")


def _journal_row(connection: sqlite3.Connection) -> Optional[tuple[Any, ...]]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='substrate_dirty_meta'"
    ).fetchone()
    if exists is None:
        return None
    row = connection.execute(
        "SELECT format,version,base_generation,manifest_sha256,state,"
        "records_per_shard,index_store_id,vector_store_id,neuron_store_id,"
        "full_neurons_dirty "
        "FROM substrate_dirty_meta WHERE singleton=1"
    ).fetchone()
    if row is None:
        raise ValueError("paged dirty journal metadata row is missing")
    return row


def _validate_journal(
    connection: sqlite3.Connection, generation: str, manifest_sha: str
) -> tuple[Any, ...]:
    row = _journal_row(connection)
    if (
        row is None or row[0] != _FORMAT or row[1] != _VERSION
        or row[2] != generation or row[3] != manifest_sha
        or row[4] not in {"building", "ready"}
        or type(row[5]) is not int or not 1 <= row[5] <= 512
        or row[9] not in (0, 1)
    ):
        raise ValueError("paged dirty journal is stale or malformed")
    _check_triggers(connection)
    identities = (
        _meta(connection, "index_metadata", "store_id"),
        _meta(connection, "paged_vector_meta", "store_id"),
        _meta(connection, "paged_neuron_meta", "store_id"),
    )
    if tuple(row[6:9]) != identities:
        raise ValueError("paged dirty journal belongs to another cache")
    return row


def _generation(store: Path, pointer: Mapping[str, Any]) -> dict[str, Any]:
    if pointer.get("format") != "omni-substrate-shards" or pointer.get("formatVersion") != 3:
        raise ValueError("dirty journal requires a v3 committed generation")
    generation_id, _references = NeuralSubstrate._generation_references(
        store, dict(pointer)
    )
    path = NeuralSubstrate._safe_store_path(
        store, "generations/%s/manifest.json" % generation_id
    )
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("formatVersion") != 3:
        raise ValueError("dirty journal prior generation is invalid")
    return value


def _record_ids(store: Path, shard: Mapping[str, Any], max_rows: int) -> list[str]:
    spec = shard.get("records")
    if not isinstance(spec, Mapping):
        raise ValueError("dirty journal prior shard lacks records")
    checksum = spec.get("sha256")
    size = spec.get("bytes")
    if (
        not isinstance(checksum, str) or len(checksum) != 64
        or any(char not in "0123456789abcdef" for char in checksum)
        or type(size) is not int or not 0 <= size <= _MAX_RECORD_BLOB
        or spec.get("path") != "blobs/%s.json" % checksum
    ):
        raise ValueError("dirty journal prior record blob is invalid")
    path = NeuralSubstrate._safe_store_path(store, spec["path"])
    if path.is_symlink() or path.stat().st_size != size:
        raise ValueError("dirty journal prior record blob size changed")
    payload = path.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("dirty journal prior record blob checksum mismatch")
    value = json.loads(payload)
    ids = value.get("ids") if isinstance(value, dict) else None
    kind = shard.get("kind")
    bucket = shard.get("bucket")
    if (
        not isinstance(value, dict)
        or value.get("kind") != kind or not isinstance(ids, list)
        or len(ids) != shard.get("count") or not 1 <= len(ids) <= max_rows
        or ids != sorted(set(ids))
        or any(
            not isinstance(item, str)
            or NeuralSubstrate._bucket(kind, item) != bucket
            for item in ids
        )
    ):
        raise ValueError("dirty journal prior record IDs are invalid")
    return ids


def install_generation_journal(
    sqlite_path: Path,
    store_root: Path,
    committed_pointer: Mapping[str, Any],
    *,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
) -> dict[str, Any]:
    """Install/resume same-transaction triggers from a clean v3 boundary.

    Membership is imported one checked shard at a time. During a crash or
    reserve pause the journal remains ``building`` and cannot drive saves;
    triggers still preserve any subsequent mutations for a safe retry.
    """

    path = Path(sqlite_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("dirty journal needs an existing regular shared cache")
    store = Path(store_root).resolve()
    generation = _generation(store, committed_pointer)
    generation_id = str(committed_pointer["activeGeneration"])
    manifest_sha = str(committed_pointer["generationManifestSha256"])
    records_per_shard = generation.get("recordsPerShard")
    if type(records_per_shard) is not int or not 1 <= records_per_shard <= 512:
        raise ValueError("dirty journal requires bounded v3 prior shards")
    with closing(_connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = _journal_row(connection)
            if existing is None:
                identities = _bindings(connection, generation_id)
                _reserve(disk_reserve, 256 * 1024, "dirty journal schema")
                connection.execute(
                    "CREATE TABLE substrate_dirty_meta ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                    "format TEXT NOT NULL,version INTEGER NOT NULL,"
                    "base_generation TEXT NOT NULL,manifest_sha256 TEXT NOT NULL,"
                    "state TEXT NOT NULL,records_per_shard INTEGER NOT NULL,"
                    "index_store_id TEXT NOT NULL,vector_store_id TEXT NOT NULL,"
                    "neuron_store_id TEXT NOT NULL,"
                    "full_neurons_dirty INTEGER NOT NULL DEFAULT 0)"
                )
                connection.execute(
                    "CREATE TABLE substrate_dirty_ids ("
                    "kind TEXT NOT NULL,record_id TEXT NOT NULL,"
                    "PRIMARY KEY(kind,record_id)) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE substrate_base_membership ("
                    "kind TEXT NOT NULL,record_id TEXT NOT NULL,"
                    "bucket TEXT NOT NULL,part INTEGER NOT NULL,"
                    "PRIMARY KEY(kind,record_id)) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE INDEX substrate_base_group ON "
                    "substrate_base_membership(kind,bucket,part,record_id)"
                )
                connection.execute(
                    "INSERT INTO substrate_dirty_meta "
                    "(singleton,format,version,base_generation,manifest_sha256,state,"
                    "records_per_shard,index_store_id,vector_store_id,neuron_store_id) "
                    "VALUES (1,?,?,?,?,?,?,?,?,?)",
                    (_FORMAT, _VERSION, generation_id, manifest_sha, "building",
                     records_per_shard, *identities),
                )
                for statement in _triggers().values():
                    connection.execute(statement)
            else:
                _validate_journal(connection, generation_id, manifest_sha)
                if existing[4] == "ready":
                    connection.commit()
                    return journal_status(path, committed_pointer)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

        observed = {kind: 0 for kind in _KINDS}
        for shard in generation.get("shards", []):
            if not isinstance(shard, Mapping) or shard.get("kind") not in _KINDS:
                continue
            kind = str(shard["kind"])
            bucket = shard.get("bucket")
            part = shard.get("part")
            if (
                not isinstance(bucket, str) or len(bucket) != 1
                or bucket not in "0123456789abcdef"
                or type(part) is not int or part < 0
            ):
                raise ValueError("dirty journal prior shard placement is invalid")
            ids = _record_ids(store, shard, records_per_shard)
            observed[kind] += len(ids)
            _reserve(disk_reserve, 65536 + 512 * sum(len(item) for item in ids),
                     "dirty journal base membership")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT INTO substrate_base_membership(kind,record_id,bucket,part) "
                    "VALUES (?,?,?,?) ON CONFLICT(kind,record_id) DO UPDATE SET "
                    "bucket=excluded.bucket,part=excluded.part",
                    ((kind, item, bucket, part) for item in ids),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        expected = generation.get("counts")
        if not isinstance(expected, Mapping) or any(
            observed[kind] != expected.get(kind) for kind in _KINDS
        ):
            raise ValueError("dirty journal prior shard counts disagree")
        connection.execute("BEGIN IMMEDIATE")
        try:
            _validate_journal(connection, generation_id, manifest_sha)
            for kind in _KINDS:
                count = connection.execute(
                    "SELECT COUNT(*) FROM substrate_base_membership WHERE kind=?", (kind,)
                ).fetchone()[0]
                if count != expected[kind]:
                    raise ValueError("dirty journal base membership is incomplete")
            connection.execute(
                "UPDATE substrate_dirty_meta SET state='ready' WHERE singleton=1"
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return journal_status(path, committed_pointer)


def journal_status(sqlite_path: Path, committed_pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Check the exact trigger/base binding without trusting a stale journal."""

    path = Path(sqlite_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("dirty journal shared cache is missing")
    with closing(_connect(path)) as connection:
        row = _validate_journal(
            connection,
            str(committed_pointer.get("activeGeneration", "")),
            str(committed_pointer.get("generationManifestSha256", "")),
        )
        dirty = connection.execute(
            "SELECT COUNT(*) FROM substrate_dirty_ids"
        ).fetchone()[0]
        return {
            "baseGeneration": row[2],
            "state": row[4],
            "recordsPerShard": row[5],
            "dirtyIds": dirty,
            "fullNeuronsDirty": bool(row[9]),
        }


def iter_dirty_ids(
    sqlite_path: Path, committed_pointer: Mapping[str, Any]
) -> Iterator[tuple[str, str, Optional[str], Optional[int]]]:
    """Stream dirty IDs and their prior placement; new IDs have no placement.

    This is a read-only partial publication API. A writer must explicitly
    assign new IDs to bounded parts and rebase only after ``brain.json`` has
    committed. It must not interpret an absent prior placement as no change.
    """

    path = Path(sqlite_path)
    with closing(_connect(path)) as connection:
        connection.execute("BEGIN")
        try:
            row = _validate_journal(
                connection,
                str(committed_pointer.get("activeGeneration", "")),
                str(committed_pointer.get("generationManifestSha256", "")),
            )
            if row[4] != "ready":
                raise ValueError("dirty journal base membership is still building")
            if row[9]:
                raise ValueError(
                    "neuron decay epoch changed; all neuron shards require publication"
                )
            cursor = connection.execute(
                "SELECT d.kind,d.record_id,m.bucket,m.part FROM substrate_dirty_ids AS d "
                "LEFT JOIN substrate_base_membership AS m "
                "ON m.kind=d.kind AND m.record_id=d.record_id "
                "ORDER BY d.kind,d.record_id"
            )
            for kind, record_id, bucket, part in cursor:
                if kind not in _KINDS or not isinstance(record_id, str) or not record_id:
                    raise ValueError("dirty journal contains an invalid ID")
                yield kind, record_id, bucket, part
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
