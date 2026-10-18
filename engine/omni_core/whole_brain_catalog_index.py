"""Immutable indexed locations into hash-bound catalog shards, not neural data."""
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile

from .whole_brain_catalog import SHARD_BYTES, check_cancel, read_owned_json

_PROOFS = OrderedDict()  # bounded operational FD metadata; no model/cardinality cap


class CatalogIndexWriter:
    def __init__(self, engine, admit, reserve_disk, cancelled):
        self.engine, self.admit, self.reserve_disk, self.cancelled = Path(engine), admit, reserve_disk, cancelled
        self.root = self.engine / "whole-brain-catalog" / "index"
        self.root.mkdir(parents=True, exist_ok=True)
        fd, self.path = tempfile.mkstemp(prefix=".build-", suffix=".sqlite3", dir=self.root)
        os.close(fd)
        self.db = None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.executescript("""PRAGMA journal_mode=DELETE; PRAGMA cache_size=-128; PRAGMA temp_store=FILE;
          CREATE TABLE records(id TEXT PRIMARY KEY,kind TEXT,shard TEXT,ordinal INTEGER,owner_id TEXT,region_id TEXT,
              parent_id TEXT,source_id TEXT,target_id TEXT,rank INTEGER);
          CREATE INDEX records_order ON records(rank,id);
          CREATE INDEX records_owner ON records(owner_id,id);
          CREATE INDEX records_region ON records(region_id,owner_id);
          CREATE INDEX records_parent ON records(parent_id,id);""")
        except BaseException:
            self.close()
            raise

    def add(self, sha, records):
        for ordinal, row in enumerate(records):
            check_cancel(self.cancelled)
            rank = {"region": 1, "owner": 3, "alias": 4, "membership": 5, "inspection": 6, "bridge": 7}[row["kind"]]
            if row.get("regionKind") == "whole-brain": rank = 0
            elif row.get("regionKind") == "connected-parent": rank = 2
            self.reserve_disk(1024 + len(row["id"]) * 4, "whole-brain immutable catalog location index")
            self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)", (row["id"], row["kind"], sha, ordinal,
                row["id"] if row["kind"] == "owner" else row.get("ownerId"), row.get("regionId"), row.get("parentRegionId"),
                row.get("sourceRegionId"), row.get("targetRegionId"), rank))

    def finish(self):
        self.db.commit(); self.db.close(); self.db = None
        sha, size = hashlib.sha256(), 0
        with open(self.path, "rb") as source:
            while True:
                check_cancel(self.cancelled)
                part = source.read(SHARD_BYTES)
                if not part: break
                sha.update(part); size += len(part)
        digest = sha.hexdigest(); target = self.root / (digest + ".sqlite3")
        self.reserve_disk(size, "whole-brain catalog index publication")
        try: os.link(self.path, target)
        except FileExistsError:
            if target.stat().st_size != size:
                raise ValueError("existing immutable catalog index differs")
            with catalog_index(self.engine, {"path": "whole-brain-catalog/index/%s.sqlite3" % digest,
                               "sha256": digest, "bytes": size}, admit=self.admit, cancelled=self.cancelled):
                pass
        self.close()
        return {"path": "whole-brain-catalog/index/%s.sqlite3" % digest, "sha256": digest, "bytes": size}

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None
        Path(self.path).unlink(missing_ok=True)
        Path(self.path + "-journal").unlink(missing_ok=True)


@contextmanager
def catalog_index(engine, descriptor, *, admit, cancelled=None):
    sha = descriptor.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or descriptor.get("path") != "whole-brain-catalog/index/%s.sqlite3" % sha:
        raise ValueError("invalid whole-brain index descriptor")
    path = Path(engine) / descriptor["path"]
    path.resolve().relative_to(Path(engine).resolve())
    before = path.lstat()
    if path.is_symlink() or not path.is_file() or before.st_size != descriptor.get("bytes"):
        raise ValueError("whole-brain index source identity differs")
    admit(512 * 1024, "whole-brain bounded immutable SQLite index reader")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)); database = None
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    signature = identity(before); proof = (str(path.resolve()), *signature, sha)
    try:
        if identity(os.fstat(fd)) != signature: raise ValueError("whole-brain index changed before opening")
        if proof not in _PROOFS:
            actual, remaining = hashlib.sha256(), before.st_size
            while remaining:
                check_cancel(cancelled); part = os.read(fd, min(SHARD_BYTES, remaining))
                if not part: raise ValueError("whole-brain index truncated")
                actual.update(part); remaining -= len(part)
            if os.read(fd, 1) or actual.hexdigest() != sha: raise ValueError("whole-brain index checksum differs")
            _PROOFS[proof] = True
            while len(_PROOFS) > 16: _PROOFS.popitem(last=False)
        else: _PROOFS.move_to_end(proof)
        database = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        database.execute("PRAGMA query_only=ON"); database.execute("PRAGMA cache_size=-128")
        yield CatalogIndexReader(database, Path(engine), admit, cancelled)
        check_cancel(cancelled)
        if identity(os.fstat(fd)) != signature or identity(path.lstat()) != signature: raise ValueError("whole-brain index changed during page")
    finally:
        if database is not None: database.close()
        os.close(fd)


class CatalogIndexReader:
    def __init__(self, database, engine, admit, cancelled):
        self.database, self.engine, self.admit, self.cancelled = database, engine, admit, cancelled
        self.cached_shard, self.cached_body = None, None
        self.held_bytes = 0

    def row(self, location):
        if location is None: return None
        identifier, sha, ordinal = location
        if sha != self.cached_shard:
            self.cached_body, _ = read_owned_json(self.engine, "whole-brain-catalog/blobs/%s.json" % sha,
                admit=lambda size, label: self.admit(size + self.held_bytes, label), cancelled=self.cancelled, expected_sha=sha)
            self.cached_shard = sha
        records = self.cached_body.get("records", [])
        if type(ordinal) is not int or not 0 <= ordinal < len(records) or records[ordinal].get("id") != identifier:
            raise ValueError("whole-brain catalog index location differs from actual shard")
        from .whole_brain_catalog import canonical
        self.held_bytes += len(canonical(records[ordinal])) * 8 + 256
        self.admit(512 * 1024 + self.held_bytes, "whole-brain retained result/anchor metadata")
        return records[ordinal]

    def get(self, identifier):
        return self.row(self.database.execute("SELECT id,shard,ordinal FROM records WHERE id=?", (identifier,)).fetchone())

    def page(self, facet, *, region_id=None, owner_id=None, after=(-1, ""), count=64):
        where, arguments = "1", []
        if facet == "overview": where = "kind IN ('region','bridge')"
        elif facet == "members":
            if not region_id: raise ValueError("region members need an exact current region")
            where = "(id=? OR parent_id=? OR source_id=? OR target_id=? OR owner_id IN (SELECT owner_id FROM records WHERE kind='membership' AND region_id=?))"
            arguments = [region_id] * 5
        elif facet == "owner":
            if not owner_id: raise ValueError("owner detail needs an exact registered owner")
            where, arguments = "(id=? OR owner_id=?)", [owner_id, owner_id]
        elif facet != "all": raise ValueError("invalid whole-brain catalog facet")
        sql = "SELECT id,shard,ordinal,rank FROM records WHERE (" + where + ") AND (rank>? OR (rank=? AND id>?)) ORDER BY rank,id LIMIT ?"
        locations = list(self.database.execute(sql, (*arguments, after[0], after[0], after[1], count + 1)))
        self.admit(512 * 1024 + len(locations) * 256, "whole-brain indexed result/anchor page")
        records = [self.row(row[:3]) for row in locations[:count]]
        anchors, known = [], {row["id"] for row in records}
        for row in records:
            for field in ("parentRegionId", "regionId", "ownerId", "sourceRegionId", "targetRegionId"):
                identifier = row.get(field)
                if identifier and identifier not in known:
                    actual = self.get(identifier)
                    if actual is None or actual.get("kind") not in ("region", "owner"):
                        raise ValueError("whole-brain relation has no actual endpoint anchor")
                    anchors.append(actual); known.add(identifier)
        last = locations[min(count, len(locations)) - 1] if locations else None
        return records, anchors, (last[3], last[0]) if last else after, len(locations) > count
