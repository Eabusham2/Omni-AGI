"""Generation-bound incremental publication for a shared paged substrate.

SQLite triggers record neuron metadata, packed-row, and assembly mutations in
the *same transaction* as their source row. The base v3 shard membership is
stored on disk, not in a corpus-sized Python map. Plans replace complete stable
bucket/part groups, including deletions and newly assigned IDs. Only an
independently verified ``brain.json`` commit may rebase membership and clear
dirty IDs. SQLite remains an expendable working cache, not checkpoint authority.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from .vsa import NeuralSubstrate
from .paged_store_counts import (
    assembly_vector_count, ensure_assembly_vector_count, ensure_row_count, row_count,
)


_FORMAT = "omni-paged-substrate-dirty-journal"
_VERSION = 2
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
    statements.append(
        "UPDATE substrate_dirty_meta SET mutation_revision=mutation_revision+1 "
        "WHERE singleton=1;"
    )
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
        "WHEN NEW.key IN ('decay_epoch','decay_log_factor',"
        "'decay_uncertainty_sum','decay_zero_generation') AND NEW.value!=OLD.value BEGIN "
        "UPDATE substrate_dirty_meta SET full_neurons_dirty=1,"
        "mutation_revision=mutation_revision+1 WHERE singleton=1; END"
    )
    result["substrate_dirty_neuron_decay_insert"] = (
        "CREATE TRIGGER substrate_dirty_neuron_decay_insert "
        "AFTER INSERT ON paged_neuron_meta "
        "WHEN NEW.key='decay_epoch' AND NEW.value!='0' BEGIN "
        "UPDATE substrate_dirty_meta SET full_neurons_dirty=1,"
        "mutation_revision=mutation_revision+1 WHERE singleton=1; END"
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
    if "mutation_revision" not in {
        item[1] for item in connection.execute("PRAGMA table_info(substrate_dirty_meta)")
    }:
        raise ValueError("paged dirty journal needs a committed-generation rebuild")
    row = connection.execute(
        "SELECT format,version,base_generation,manifest_sha256,state,"
        "records_per_shard,index_store_id,vector_store_id,neuron_store_id,"
        "full_neurons_dirty,mutation_revision,plan_nonce "
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
        or type(row[10]) is not int or row[10] < 0
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
    for table in ("assembly_records", "paged_vector_rows", "paged_neuron_records"):
        row_count(connection, table)  # Missing counters must not initiate a scan here.
        ensure_row_count(connection, table)
    assembly_vector_count(connection)
    ensure_assembly_vector_count(connection)
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
    before = path.stat()
    if path.is_symlink() or before.st_size != size:
        raise ValueError("dirty journal prior record blob size changed")
    payload = path.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("dirty journal prior record blob checksum mismatch")
    from .paged_substrate_writer import remember_verified_blob
    remember_verified_blob(path, checksum, before=before)
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
                    "full_neurons_dirty INTEGER NOT NULL DEFAULT 0,"
                    "mutation_revision INTEGER NOT NULL DEFAULT 0,"
                    "plan_nonce TEXT NOT NULL DEFAULT '')"
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
                    "CREATE TABLE substrate_base_groups (kind TEXT NOT NULL,"
                    "bucket TEXT NOT NULL,part INTEGER NOT NULL,count INTEGER NOT NULL,"
                    "PRIMARY KEY(kind,bucket,part)) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE substrate_plan_membership (kind TEXT NOT NULL,"
                    "record_id TEXT NOT NULL,bucket TEXT,part INTEGER,"
                    "PRIMARY KEY(kind,record_id)) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE INDEX substrate_plan_member_group ON "
                    "substrate_plan_membership(kind,bucket,part,record_id)"
                )
                connection.execute(
                    "CREATE TABLE substrate_plan_groups (kind TEXT NOT NULL,"
                    "bucket TEXT NOT NULL,part INTEGER NOT NULL,count INTEGER NOT NULL,"
                    "PRIMARY KEY(kind,bucket,part)) WITHOUT ROWID"
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
                connection.execute(
                    "INSERT INTO substrate_base_groups(kind,bucket,part,count) "
                    "VALUES (?,?,?,?) ON CONFLICT(kind,bucket,part) DO UPDATE SET "
                    "count=excluded.count",
                    (kind, bucket, part, len(ids)),
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
            "mutationRevision": row[10],
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


def source_stamp(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Indexed identities/revisions/epochs; never enumerate source rows."""

    return tuple(
        _meta(connection, table, key)
        for table, keys in (
            ("index_metadata", ("store_id", "index_revision")),
            ("paged_vector_meta", ("store_id", "revision")),
            ("paged_neuron_meta", (
                "store_id", "revision", "decay_epoch", "decay_log_factor",
                "decay_uncertainty_sum", "decay_zero_generation",
            )),
        )
        for key in keys
    )


def bind_shared_generation(
    connection: sqlite3.Connection, generation: str, expected_stamp: tuple[str, ...]
) -> None:
    """Rebind all three stores in their caller's single SQLite transaction."""

    if source_stamp(connection) != expected_stamp:
        raise ValueError("paged source changed before committed-generation binding")
    for table, values in (
        ("index_metadata", (
            ("committed_generation_sha256", generation),
            ("committed_index_revision", expected_stamp[1]),
            ("committed_vector_revision", expected_stamp[3]),
        )),
        ("paged_vector_meta", (
            ("committed_generation_sha256", generation),
            ("committed_revision", expected_stamp[3]),
        )),
        ("paged_neuron_meta", (
            ("committed_generation_sha256", generation),
            ("committed_revision", expected_stamp[5]),
        )),
    ):
        connection.executemany(
            "UPDATE %s SET value=? WHERE key=?" % table,
            ((value, key) for key, value in values),
        )


class DirtyShardPlan:
    """Disk-backed, revision-checked complete-group publication plan.

    Only membership DELTAS are staged. Existing IDs never move groups; new
    IDs append to a bucket's last bounded part, allocating a new part when
    full. A deletion rewrites its complete old group without repacking any
    other group. Global decay explicitly rewrites every neuron group. No plan
    operation clears the dirty journal before the authoritative brain commit.
    The caller serializes live writes; stamps also fail closed on drift.
    """

    def __init__(
        self, sqlite_path: Path, pointer: Mapping[str, Any],
        *, disk_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> None:
        self.path = Path(sqlite_path)
        self.pointer = dict(pointer)
        self.disk_reserve = disk_reserve
        self.count_delta = {kind: 0 for kind in _KINDS}
        with closing(_connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = _validate_journal(
                    connection, str(pointer.get("activeGeneration", "")),
                    str(pointer.get("generationManifestSha256", "")),
                )
                if row[4] != "ready":
                    raise ValueError("dirty journal base membership is still building")
                self.records_per_shard = int(row[5])
                self.mutation_revision = int(row[10])
                self.plan_nonce = uuid.uuid4().hex
                connection.execute(
                    "UPDATE substrate_dirty_meta SET plan_nonce=? WHERE singleton=1",
                    (self.plan_nonce,),
                )
                self.stamp = source_stamp(connection)
                self.full_neurons_dirty = bool(row[9])
                _reserve(disk_reserve, 65536, "dirty shard plan")
                connection.execute("DELETE FROM substrate_plan_membership")
                connection.execute("DELETE FROM substrate_plan_groups")
                if self.full_neurons_dirty:
                    # This is O(neuron SHARDS), not an all-ID journal materialization.
                    connection.execute(
                        "INSERT INTO substrate_plan_groups(kind,bucket,part,count) "
                        "SELECT kind,bucket,part,count FROM substrate_base_groups "
                        "WHERE kind='neurons'"
                    )
                cursor = connection.execute(
                    "SELECT d.kind,d.record_id,m.bucket,m.part "
                    "FROM substrate_dirty_ids AS d LEFT JOIN substrate_base_membership AS m "
                    "ON m.kind=d.kind AND m.record_id=d.record_id "
                    "ORDER BY d.kind,d.record_id"
                )
                for kind, identifier, bucket, part in cursor:
                    if kind not in _KINDS or not isinstance(identifier, str) or not identifier:
                        raise ValueError("dirty journal contains an invalid ID")
                    table, column = _TABLES[kind]
                    present = connection.execute(
                        "SELECT 1 FROM %s WHERE %s=?" % (table, column),
                        (identifier,),
                    ).fetchone() is not None
                    if kind == "neurons":
                        vector_present = connection.execute(
                            "SELECT 1 FROM paged_vector_rows WHERE vector_id=?", (identifier,)
                        ).fetchone() is not None
                        if present != vector_present:
                            raise ValueError("dirty neuron metadata/packed-row identity differs")
                    if bucket is not None:
                        self._mark_group(connection, kind, bucket, part)
                        if not present:
                            self._change_count(connection, kind, bucket, part, -1)
                            self.count_delta[kind] -= 1
                            connection.execute(
                                "INSERT INTO substrate_plan_membership "
                                "(kind,record_id,bucket,part) VALUES (?,?,NULL,NULL)",
                                (kind, identifier),
                            )
                    elif present:
                        bucket = NeuralSubstrate._bucket(kind, identifier)
                        tails = [connection.execute(
                            "SELECT MAX(part) FROM %s WHERE kind=? AND bucket=?" % table,
                            (kind, bucket),
                        ).fetchone()[0] for table in (
                            "substrate_base_groups", "substrate_plan_groups",
                        )]
                        part = max((int(value) for value in tails if value is not None), default=0)
                        count = self._mark_group(connection, kind, bucket, part)
                        if count >= self.records_per_shard:
                            part += 1
                            self._mark_group(connection, kind, bucket, part)
                        _reserve(disk_reserve, 4096 + 4 * len(identifier),
                                 "dirty shard membership delta")
                        connection.execute(
                            "INSERT INTO substrate_plan_membership "
                            "(kind,record_id,bucket,part) VALUES (?,?,?,?)",
                            (kind, identifier, bucket, part),
                        )
                        self._change_count(connection, kind, bucket, part, 1)
                        self.count_delta[kind] += 1
                if source_stamp(connection) != self.stamp:
                    raise ValueError("paged source changed while planning dirty groups")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _mark_group(
        connection: sqlite3.Connection, kind: str, bucket: str, part: int
    ) -> int:
        connection.execute(
            "INSERT OR IGNORE INTO substrate_plan_groups(kind,bucket,part,count) "
            "SELECT ?,?,?,COALESCE((SELECT count FROM substrate_base_groups "
            "WHERE kind=? AND bucket=? AND part=?),0)",
            (kind, bucket, part, kind, bucket, part),
        )
        return int(connection.execute(
            "SELECT count FROM substrate_plan_groups WHERE kind=? AND bucket=? AND part=?",
            (kind, bucket, part),
        ).fetchone()[0])

    def _change_count(
        self, connection: sqlite3.Connection, kind: str, bucket: str, part: int, delta: int
    ) -> None:
        connection.execute(
            "UPDATE substrate_plan_groups SET count=count+? "
            "WHERE kind=? AND bucket=? AND part=?", (delta, kind, bucket, part),
        )
        count = connection.execute(
            "SELECT count FROM substrate_plan_groups WHERE kind=? AND bucket=? AND part=?",
            (kind, bucket, part),
        ).fetchone()[0]
        if not 0 <= count <= self.records_per_shard:
            raise ValueError("dirty shard planned membership exceeds bounded part")

    def assert_unchanged(self, connection: Optional[sqlite3.Connection] = None) -> None:
        if connection is None:
            with closing(_connect(self.path)) as checked:
                checked.execute("BEGIN")
                self.assert_unchanged(checked)
            return
        row = _validate_journal(
            connection, str(self.pointer["activeGeneration"]),
            str(self.pointer["generationManifestSha256"]),
        )
        if (
            source_stamp(connection) != self.stamp
            or row[10] != self.mutation_revision
            or bool(row[9]) != self.full_neurons_dirty
            or row[11] != self.plan_nonce
        ):
            raise ValueError("paged source/journal revision or decay epoch changed")

    def groups(self) -> Iterator[DirtyGroup]:
        with closing(_connect(self.path)) as connection:
            connection.execute("BEGIN")
            self.assert_unchanged(connection)
            cursor = connection.execute(
                "SELECT kind,bucket,part,count FROM substrate_plan_groups "
                "ORDER BY kind,bucket,part"
            )
            for kind, bucket, part, count in cursor:
                rows = connection.execute(
                    "SELECT m.record_id FROM substrate_base_membership AS m "
                    "LEFT JOIN substrate_plan_membership AS p "
                    "ON p.kind=m.kind AND p.record_id=m.record_id "
                    "WHERE m.kind=? AND m.bucket=? AND m.part=? AND p.record_id IS NULL "
                    "UNION SELECT record_id FROM substrate_plan_membership "
                    "WHERE kind=? AND bucket=? AND part=? ORDER BY record_id LIMIT ?",
                    (kind, bucket, part, kind, bucket, part, self.records_per_shard + 1),
                ).fetchall()
                ids = tuple(value[0] for value in rows)
                if len(ids) != count or len(ids) > self.records_per_shard:
                    raise ValueError("dirty shard complete-group coverage differs")
                yield DirtyGroup(kind, bucket, part, ids)
        self.assert_unchanged()

    def rebase(
        self, committed_pointer: Mapping[str, Any],
        *, verify_brain_commit: Callable[[], bool],
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        additional_rebase: Optional[Callable[[sqlite3.Connection], None]] = None,
    ) -> None:
        """Atomically advance ONLY after the caller verifies brain.json."""

        _reserve(disk_reserve or self.disk_reserve, 65536, "dirty journal committed rebase")
        with closing(_connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self.assert_unchanged(connection)
                if verify_brain_commit() is not True:
                    raise ValueError("brain.json has not committed the dirty shard generation")
                expected_counts = self.pointer["counts"]
                if any(
                    committed_pointer["counts"].get(kind)
                    != expected_counts[kind] + self.count_delta[kind]
                    for kind in _KINDS
                ):
                    raise ValueError("dirty shard committed counts diverge from membership")
                if additional_rebase is not None:
                    additional_rebase(connection)
                connection.execute(
                    "DELETE FROM substrate_base_membership WHERE (kind,record_id) "
                    "IN (SELECT kind,record_id FROM substrate_plan_membership)"
                )
                connection.execute(
                    "INSERT INTO substrate_base_membership(kind,record_id,bucket,part) "
                    "SELECT kind,record_id,bucket,part FROM substrate_plan_membership "
                    "WHERE bucket IS NOT NULL"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO substrate_base_groups(kind,bucket,part,count) "
                    "SELECT kind,bucket,part,count FROM substrate_plan_groups"
                )
                bind_shared_generation(
                    connection, str(committed_pointer["activeGeneration"]), self.stamp
                )
                connection.execute("DELETE FROM substrate_dirty_ids")
                connection.execute("DELETE FROM substrate_plan_membership")
                connection.execute("DELETE FROM substrate_plan_groups")
                connection.execute(
                    "UPDATE substrate_dirty_meta SET base_generation=?,manifest_sha256=?,"
                    "full_neurons_dirty=0 WHERE singleton=1",
                    (committed_pointer["activeGeneration"],
                     committed_pointer["generationManifestSha256"]),
                )
                if verify_brain_commit() is not True:
                    raise ValueError("brain.json changed during dirty journal rebase")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


def rebuild_generation_journal(
    sqlite_path: Path, store_root: Path, committed_pointer: Mapping[str, Any],
    *, expected_stamp: tuple[str, ...], verify_brain_commit: Callable[[], bool],
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
    additional_rebase: Optional[Callable[[sqlite3.Connection], None]] = None,
) -> dict[str, Any]:
    """Initial/full-save rebase or recovery from a stale derived journal.

    This intentionally reads committed shards once. A stale/building journal
    is never trusted for a partial save. Crash recovery rebuilds the working
    cache from the independently committed brain generation before this call.
    """

    _reserve(disk_reserve, 256 * 1024, "dirty journal full rebase")
    with closing(_connect(Path(sqlite_path))) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if verify_brain_commit() is not True:
                raise ValueError("brain.json has not committed the full shard generation")
            bind_shared_generation(
                connection, str(committed_pointer["activeGeneration"]), expected_stamp
            )
            if additional_rebase is not None:
                additional_rebase(connection)
            for name in _triggers():
                connection.execute("DROP TRIGGER IF EXISTS %s" % name)
            for table in (
                "substrate_dirty_meta", "substrate_dirty_ids", "substrate_base_membership",
                "substrate_base_groups", "substrate_plan_membership", "substrate_plan_groups",
            ):
                connection.execute("DROP TABLE IF EXISTS %s" % table)
            if verify_brain_commit() is not True:
                raise ValueError("brain.json changed during dirty journal full rebase")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return install_generation_journal(
        sqlite_path, store_root, committed_pointer, disk_reserve=disk_reserve
    )
