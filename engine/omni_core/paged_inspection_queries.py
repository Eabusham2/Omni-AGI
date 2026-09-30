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
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from .authenticated_paged_cache import cache_session, canonical


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
        available = os.statvfs(self.path.parent).f_bavail * os.statvfs(self.path.parent).f_frsize
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
        self.directory = view.store / "inspection" / "query-generations" / view.substrate_revision
        self.path = self.directory / "records.sqlite3"
        self.manifest = self.directory / "manifest.json"
        self.session = cache_session(_owner(view))
        self._ensure()

    def _validate(self):
        from .paged_substrate_writer import _verified_file
        value = json.loads(self.manifest.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("inspection query index descriptor is malformed")
        body = {key: item for key, item in value.items() if key not in {"contentSha256", "authentication"}}
        if (
            value.get("format") != "omni-immutable-substrate-query-index" or value.get("version") != 2
            or value.get("sourceGeneration") != self.view.substrate_revision
            or value.get("sourceManifestSha256") != self.view.generation_sha256
            or value.get("counts") != self.view.counts
            or value.get("contentSha256") != hashlib.sha256(canonical(body)).hexdigest()
            or not self.session.authentic("inspection-source-origin-v2", body, value.get("authentication"))
        ):
            raise ValueError("inspection query index has no checked source provenance")
        with self.session.blob_scope():
            _verified_file(self.path, value["sha256"], value["bytes"])

    def _ensure(self):
        if self.path.is_file() and self.manifest.is_file():
            try:
                self._validate()
                return
            except (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError):
                # Never bless untrusted rows merely because their public hash
                # and row count are self-consistent. This cache is expendable.
                pass
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".query-index-", suffix=".sqlite3", dir=self.directory)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA cache_size=-256")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute(
                    "CREATE TABLE records(kind TEXT NOT NULL,id TEXT NOT NULL,payload BLOB NOT NULL,"
                    "region TEXT NOT NULL,search_text TEXT NOT NULL,source_id TEXT,target_id TEXT,"
                    "PRIMARY KEY(kind,id)) WITHOUT ROWID"
                )
                connection.execute("CREATE INDEX record_region ON records(kind,region,id)")
                connection.execute("CREATE INDEX record_source ON records(kind,source_id,id)")
                connection.execute("CREATE INDEX record_target ON records(kind,target_id,id)")
                observed = {kind: 0 for kind in self.view.counts}
                from .substrate_inspection import _synapse_records
                for kind in ("neurons", "assemblies"):
                    for shard in self.view.kind_shards(kind):
                        self.session.reserve_disk(65536 + shard["records"]["bytes"] * 4, "inspection indexed shard")
                        for raw in self.view.records(shard)["records"]:
                            if kind == "assemblies" and "source_text" in raw:
                                raw = {key: value for key, value in raw.items() if key != "source_text"}
                                raw["__inspection_retains_source_text"] = True
                            identifier = raw["id"]
                            region = str(raw.get("region", "semantic")) if kind == "neurons" else "assembly"
                            label = str(raw.get("label", raw.get("source_label", raw.get("kind", "assembly"))))
                            search_text = identifier + " " + label + " " + region
                            if kind == "assemblies":
                                search_text += " " + str(raw.get("kind", "assembly"))
                            connection.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)",
                                               (kind, identifier, canonical(raw), region.casefold(),
                                                search_text.casefold(), None, None))
                            observed[kind] += 1
                for record in _synapse_records(self.view, apply_attention=False):
                    self.session.reserve_disk(4096 + 4 * len(canonical(record)), "inspection indexed synapse")
                    connection.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)",
                                       ("synapses", record["id"], canonical(record), "",
                                        " ".join(str(record.get(key, "")) for key in ("id", "sourceId", "targetId", "kind")).casefold(),
                                        record["sourceId"], record["targetId"]))
                    observed["synapses"] += 1
                if observed != self.view.counts:
                    raise ValueError("inspection index source coverage differs")
                connection.commit()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            sha = hashlib.sha256()
            with temporary.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    sha.update(chunk)
            body = {"format": "omni-immutable-substrate-query-index", "version": 2,
                    "sourceGeneration": self.view.substrate_revision, "sourceManifestSha256": self.view.generation_sha256,
                    "counts": self.view.counts, "sha256": sha.hexdigest(), "bytes": temporary.stat().st_size}
            from .persistence import atomic_write_bytes
            os.replace(temporary, self.path)
            atomic_write_bytes(self.manifest, canonical({
                **body, "contentSha256": hashlib.sha256(canonical(body)).hexdigest(),
                "authentication": self.session.sign("inspection-source-origin-v2", body),
            }))
            self._validate()
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def connection(self):
        uri = "file:%s?mode=ro&immutable=1" % self.path
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA cache_size=-256")
            yield connection

    def page(
        self, kind: str, *, region: str, search: str, connected_to: str,
        offset: int, page_size: int,
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
        where = " AND ".join(clauses)
        with self.connection() as connection:
            matched = int(connection.execute("SELECT COUNT(*) FROM records AS r WHERE " + where, arguments).fetchone()[0])
            page = []
            byte_count = 0
            for (payload,) in connection.execute(
                "SELECT r.payload FROM records AS r WHERE " + where + " ORDER BY r.id LIMIT ? OFFSET ?",
                (*arguments, page_size, offset),
            ):
                if page and byte_count + len(payload) > 16 * 1024 * 1024:
                    break
                page.append(json.loads(payload))
                byte_count += len(payload)
        return page, matched

    def neuron(self, identifier: str):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM records WHERE kind='neurons' AND id=?", (identifier,)).fetchone()
            return json.loads(row[0]) if row else None
