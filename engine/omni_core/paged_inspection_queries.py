"""Load-free, immutable generation query index for sparse brain inspection.

The initial build reads checked shards in bounded windows. Later clicks seek
indexed metadata and decode only the requested records. The immutable SQLite
file is verified cryptographically once per protected process/file identity;
unknown/weak filesystems retain actual cryptographic reads. It never selects
neural recovery state or constructs a brain/model.

A public cache checksum is not proof that cached rows came from the brain.
Only a build from checked canonical source records earns an in-process HMAC
over its generation-bound descriptor. No authentication key is persisted:
the first query in a new process rebuilds the derived generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from .authenticated_paged_cache import cache_session, canonical
from .inspection_resources import current_inspection_resources


_OWNERS: dict[str, "_ProofOwner"] = {}


class _ProofOwner:
    def __init__(self, path: Path, brain_id: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction(write=True) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS index_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO index_metadata VALUES('store_id',?)", (brain_id,))
        self._disk_reserve = None

    @contextmanager
    def _transaction(self, *, write=False):
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _reserve_disk(self, size, operation):
        from .substrate_inspection import INSPECTION_DISK_RESERVE_BYTES
        available = shutil.disk_usage(self.path.parent).free
        if available < INSPECTION_DISK_RESERVE_BYTES + int(size):
            raise RuntimeError("%s paused at inspection disk reserve" % operation)

    @staticmethod
    def _reserve_memory(_size, _operation):
        return None


def _owner(view):
    key = str(view.engine_directory)
    value = _OWNERS.get(key)
    if value is None:
        value = _ProofOwner(view.engine_directory / "state" / "inspection-proofs.sqlite3", view.brain_id)
        _OWNERS[key] = value
    return value


class PagedInspectionIndex:
    def __init__(self, view) -> None:
        self.view = view
        self.admit, self.cancelled = current_inspection_resources()
        self.admit = self.admit or (lambda _size, _label: None)
        self.admit(1024 * 1024, "inspection indexed SQLite/parser scratch")
        self._callback_error = None
        self.directory = view.store / "inspection" / "query-generations" / view.substrate_revision
        self.directory.resolve().relative_to(view.store.resolve())
        self.path = self.directory / "records.sqlite3"
        self.manifest = self.directory / "manifest.json"
        self.session = cache_session(_owner(view))
        self._ensure()

    def check(self):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("substrate inspection cancelled")

    def _validate(self):
        from .paged_substrate_writer import _verified_file
        self.check()
        self.admit(1024 * 1024 + self.manifest.stat().st_size * 8, "inspection index manifest")
        value = json.loads(self.manifest.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("inspection query index descriptor is malformed")
        body = {key: item for key, item in value.items() if key not in {"contentSha256", "authentication"}}
        if (
            value.get("format") != "omni-immutable-substrate-query-index" or value.get("version") != 4
            or value.get("sourceGeneration") != self.view.substrate_revision
            or value.get("sourceManifestSha256") != self.view.generation_sha256
            or value.get("counts") != self.view.counts
            or value.get("contentSha256") != hashlib.sha256(canonical(body)).hexdigest()
            or not self.session.authentic("inspection-source-origin-v4", body, value.get("authentication"))
        ):
            raise ValueError("inspection query index has no checked source provenance")
        with self.session.blob_scope():
            _verified_file(self.path, value["sha256"], value["bytes"])

    def _ensure(self):
        self.check()
        self.admit(1024 * 1024, "inspection indexed SQLite/parser scratch")
        if self.path.is_file() and self.manifest.is_file():
            try:
                self._validate()
                return
            except InterruptedError:
                raise
            except (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError):
                # Never bless untrusted rows merely because their public hash
                # and row count are self-consistent. This cache is expendable.
                pass
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session.reserve_disk(128 * 1024, "inspection index initial files")
        descriptor, name = tempfile.mkstemp(prefix=".query-index-", suffix=".sqlite3", dir=self.directory)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA cache_size=-256")
                connection.execute("PRAGMA temp_store=FILE")
                connection.set_progress_handler(lambda: self._progress(), 1000)
                connection.execute(
                    "CREATE TABLE records(kind TEXT NOT NULL,id TEXT NOT NULL,payload BLOB NOT NULL,"
                    "region TEXT NOT NULL,search_text TEXT NOT NULL,source_id TEXT,target_id TEXT,"
                    "PRIMARY KEY(kind,id)) WITHOUT ROWID"
                )
                connection.execute("CREATE INDEX record_region ON records(kind,region,id)")
                connection.execute("CREATE INDEX record_source ON records(kind,source_id,id)")
                connection.execute("CREATE INDEX record_target ON records(kind,target_id,id)")
                connection.execute("CREATE TABLE assembly_members(assembly_id TEXT,member_kind TEXT,ordinal INTEGER,member_id TEXT,"
                                   "PRIMARY KEY(assembly_id,member_kind,ordinal)) WITHOUT ROWID")
                connection.execute("CREATE TABLE neuron_activity(id TEXT PRIMARY KEY,region TEXT,activation REAL) WITHOUT ROWID")
                observed = {kind: 0 for kind in self.view.counts}
                from .substrate_inspection import _synapse_records
                from .inspection_source_stream import stream_metadata_shard
                for kind in ("neurons", "assemblies"):
                    for shard in self.view.kind_shards(kind):
                        observed[kind] += stream_metadata_shard(self.view, shard, connection, admit=self.admit,
                            reserve_disk=self.session.reserve_disk, check=self.check)
                for shard in self.view.kind_shards("synapses"):
                    self.check()
                    self.admit(1024 * 1024 + int(shard["records"]["bytes"]) * 8 + int(shard["tensors"]["bytes"]) * 8 + int(shard["count"]) * 512,
                               "inspection exact synapse source/tensor shard")
                    for record in _synapse_records(self.view, (shard,), apply_attention=False):
                        self.check()
                        self.session.reserve_disk(4096 + 4 * len(canonical(record)), "inspection indexed synapse")
                        connection.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)",
                                           ("synapses", record["id"], canonical(record), "",
                                            " ".join(str(record.get(key, "")) for key in ("id", "sourceId", "targetId", "kind")).casefold(),
                                            record["sourceId"], record["targetId"]))
                        observed["synapses"] += 1
                if observed != self.view.counts:
                    raise ValueError("inspection index source coverage differs")
                self.session.reserve_disk(max(65536, sum(observed.values()) * 256), "inspection aggregate index projection")
                connection.execute("INSERT INTO neuron_activity SELECT id,COALESCE(json_extract(payload,'$.region'),'semantic'),"
                                   "CAST(COALESCE(json_extract(payload,'$.activation'),0) AS REAL) FROM records WHERE kind='neurons'")
                connection.execute("CREATE INDEX neuron_activity_region ON neuron_activity(region,id)")
                connection.execute("CREATE TABLE region_totals AS SELECT region,count(*) AS total FROM neuron_activity GROUP BY region")
                connection.execute("CREATE UNIQUE INDEX region_totals_id ON region_totals(region)")
                connection.execute("CREATE TABLE raw_bands AS SELECT region,CASE WHEN activation>=0.66 THEN 'high' WHEN activation>=0.2 THEN 'medium' ELSE 'quiet' END AS band,"
                                   "count(*) AS count,sum(activation>=0.1) AS active,sum(activation) AS total,max(activation) AS maximum FROM neuron_activity GROUP BY region,band")
                connection.execute("CREATE TABLE pathway_groups AS SELECT COALESCE(s.region,'unknown') AS source_region,COALESCE(t.region,'unknown') AS target_region,"
                                   "count(*) AS count,sum(json_extract(r.payload,'$.effectiveWeight')<0) AS negative,"
                                   "sum(json_extract(r.payload,'$.effectiveWeight')=0) AS zero_count,sum(json_extract(r.payload,'$.effectiveWeight')>0) AS positive "
                                   "FROM records AS r LEFT JOIN neuron_activity AS s ON s.id=r.source_id LEFT JOIN neuron_activity AS t ON t.id=r.target_id "
                                   "WHERE r.kind='synapses' GROUP BY source_region,target_region")
                connection.execute("CREATE UNIQUE INDEX pathway_groups_id ON pathway_groups(source_region,target_region)")
                connection.commit()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            sha = hashlib.sha256()
            with temporary.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    self.check()
                    sha.update(chunk)
            body = {"format": "omni-immutable-substrate-query-index", "version": 4,
                    "sourceGeneration": self.view.substrate_revision, "sourceManifestSha256": self.view.generation_sha256,
                    "counts": self.view.counts, "sha256": sha.hexdigest(), "bytes": temporary.stat().st_size}
            from .persistence import atomic_write_bytes
            os.replace(temporary, self.path)
            atomic_write_bytes(self.manifest, canonical({
                **body, "contentSha256": hashlib.sha256(canonical(body)).hexdigest(),
                "authentication": self.session.sign("inspection-source-origin-v4", body),
            }))
            self._validate()
        except sqlite3.OperationalError:
            if self._callback_error is not None: raise self._callback_error
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def _progress(self):
        try:
            self.check()
            self.session.reserve_disk(65536, "inspection SQLite traversal reserve")
            return 0
        except BaseException as error:
            self._callback_error = error
            return 1

    @contextmanager
    def connection(self):
        self.check(); self._validate()
        before = self.path.lstat()
        if not stat.S_ISREG(before.st_mode): raise ValueError("inspection query index must remain a regular file")
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        uri = "file:%s?mode=ro&immutable=1" % self.path
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA cache_size=-256")
            connection.execute("PRAGMA temp_store=FILE")
            connection.set_progress_handler(lambda: self._progress(), 1000)
            connection.create_function("omni_casefold", 1, lambda value: str(value).casefold() if value is not None else "", deterministic=True)
            try:
                yield connection
            except sqlite3.OperationalError:
                if self._callback_error is not None: raise self._callback_error
                raise
            finally:
                after = self.path.lstat()
                if not stat.S_ISREG(after.st_mode) or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("inspection query index changed during traversal")

    def page(
        self, kind: str, *, region: str, search: str, connected_to: str,
        offset: int, page_size: int, source_region: str = "", target_region: str = "", activation_band: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        clauses = ["r.kind=?"]
        arguments: list[Any] = [kind]
        if region and kind != "synapses":
            clauses.append("r.region=?")
            arguments.append(region.casefold())
        if search:
            clauses.append("INSTR(r.search_text,?)>0")
            arguments.append(search.casefold())
        if connected_to:
            clauses.append("(r.source_id=? OR r.target_id=?)")
            arguments.extend((connected_to, connected_to))
        if region and kind == "synapses":
            clauses.append("EXISTS(SELECT 1 FROM records AS n WHERE n.kind='neurons' "
                           "AND n.id IN (r.source_id,r.target_id) AND n.region=?)")
            arguments.append(region.casefold())
        for column, selected in (("source_id", source_region), ("target_id", target_region)):
            if selected:
                clauses.append("EXISTS(SELECT 1 FROM neuron_activity AS n WHERE n.id=r.%s AND n.region=?)" % column)
                arguments.append(selected)
        where = " AND ".join(clauses)
        with self.connection() as connection:
            if activation_band:
                connection.execute("CREATE TEMP TABLE exact_active_ids(id TEXT PRIMARY KEY) WITHOUT ROWID")
                if not self.view.attention_legacy_raw_active:
                    for number, identifier in enumerate(self.view.attention_active_neuron_ids):
                        self.check()
                        if number % 128 == 0: self.session.reserve_disk(65536, "inspection exact band attention window")
                        connection.execute("INSERT INTO exact_active_ids VALUES(?)", (identifier,))
                activation = "n.activation" if self.view.attention_legacy_raw_active else "CASE WHEN EXISTS(SELECT 1 FROM exact_active_ids AS a WHERE a.id=n.id) THEN n.activation ELSE 0 END"
                clauses.append("EXISTS(SELECT 1 FROM neuron_activity AS n WHERE n.id=r.id AND CASE WHEN (%s)>=0.66 THEN 'high' WHEN (%s)>=0.2 THEN 'medium' ELSE 'quiet' END=?)" % (activation, activation))
                arguments.append(activation_band)
                where = " AND ".join(clauses)
            matched = int(connection.execute("SELECT COUNT(*) FROM records AS r WHERE " + where, arguments).fetchone()[0])
            page = []
            byte_count = 0
            for identifier, length in connection.execute(
                "SELECT r.id,length(r.payload) FROM records AS r WHERE " + where + " ORDER BY r.id LIMIT ? OFFSET ?",
                (*arguments, page_size, offset),
            ):
                if page and byte_count + length > 16 * 1024 * 1024:
                    break
                self.check(); self.admit(1024 * 1024 + (byte_count + length) * 8, "inspection bounded indexed records")
                payload = connection.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()[0]
                record = json.loads(payload)
                if kind == "assemblies":
                    total = record.get("__inspection_neuron_count", 0) + record.get("__inspection_child_count", 0)
                    # Detail view keeps small complete lists. Larger lists have
                    # a real independent member cursor, not a pretend synapse query.
                    if total <= page_size:
                        members = connection.execute("SELECT member_kind,member_id FROM assembly_members WHERE assembly_id=? ORDER BY member_kind DESC,ordinal", (record["id"],))
                        for member_kind, member_id in members:
                            record["neuron_ids" if member_kind == "neuron" else "child_assembly_ids"].append(member_id)
                page.append(record)
                byte_count += len(payload)
        return page, matched

    def neuron(self, identifier: str):
        with self.connection() as connection:
            row = connection.execute("SELECT length(payload) FROM records WHERE kind='neurons' AND id=?", (identifier,)).fetchone()
            if row is None: return None
            self.admit(1024 * 1024 + int(row[0]) * 8, "inspection exact neuron metadata")
            return json.loads(connection.execute("SELECT payload FROM records WHERE kind='neurons' AND id=?", (identifier,)).fetchone()[0])

    def assembly(self, identifier: str):
        with self.connection() as connection:
            row = connection.execute("SELECT length(payload) FROM records WHERE kind='assemblies' AND id=?", (identifier,)).fetchone()
            if row is None: return None
            self.admit(1024 * 1024 + int(row[0]) * 8, "inspection exact assembly header")
            return json.loads(connection.execute("SELECT payload FROM records WHERE kind='assemblies' AND id=?", (identifier,)).fetchone()[0])

    def members(self, identifier: str, offset: int, page_size: int):
        header = self.assembly(identifier)
        if header is None: raise ValueError("selected assembly is absent from the authentic inspection index")
        total = header["__inspection_neuron_count"] + header["__inspection_child_count"]
        with self.connection() as connection:
            page, held = [], 1024 * 1024
            # Global offset maps directly onto original array ordinals. Two
            # indexed seeks avoid walking all prior members on every page.
            neuron_count = header["__inspection_neuron_count"]
            starts = [("neuron", offset)] if offset < neuron_count else []
            starts.append(("assembly", max(0, offset - neuron_count)))
            for kind, start in starts:
                rows = connection.execute("SELECT ordinal,member_id,length(member_id) FROM assembly_members WHERE assembly_id=? AND member_kind=? "
                    "AND ordinal>=? ORDER BY ordinal LIMIT ?", (identifier, kind, start, page_size - len(page)))
                for ordinal, member_id, length in rows:
                    self.check(); held += int(length) * 8 + 512
                    self.admit(held, "inspection exact member page")
                    page.append({"assemblyId": identifier, "memberKind": kind, "memberId": member_id, "ordinal": ordinal})
                if len(page) == page_size: break
            return header, page, total
