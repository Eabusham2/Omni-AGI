"""Exact, paged native sparse-assembly transfer; never a cortex/model importer.

Committed readers never instantiate a brain. The loaded-only caller owns the
normal writer and serializes apply; a saved receipt MUST be included by the
caller's ordinary atomic brain metadata publication, not written afterward.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import struct
from contextlib import nullcontext
from pathlib import Path

FORMAT = "omni-sparse-substrate-transfer"
SCOPE = "native-sparse-assemblies"
KINDS = ("neurons", "assemblies", "synapses")
CONTROL = struct.Struct("<bBdddqd")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
SCHEMA = {
    "meta": "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
    "rows": "CREATE TABLE rows(kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,sha TEXT NOT NULL,PRIMARY KEY(kind,id))",
    "expected": "CREATE TABLE expected(kind TEXT NOT NULL,id TEXT NOT NULL,before_sha TEXT,before_body TEXT,PRIMARY KEY(kind,id))",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid sparse transfer SHA256")
    return value


def _id(value):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("invalid sparse transfer identity")
    return value


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _tick(cancelled, reserve, size=0, label="sparse transfer page"):
    if cancelled():
        raise InterruptedError("sparse transfer cancelled at an owned byte/row boundary")
    if size:
        reserve(size, label)


def _file_sha(path, *, cancelled=lambda: False, reserve=lambda *_: None):
    path = Path(path)
    before = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ValueError("sparse custody needs an exact regular file")
    _tick(cancelled, reserve, min(before.st_size, 65536), "sparse custody checksum")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(opened):
            raise ValueError("sparse custody changed before its owned read")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while True:
                _tick(cancelled, reserve)
                chunk = source.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        if identity(opened) != identity(os.fstat(descriptor)) or identity(opened) != identity(path.lstat()):
            raise ValueError("sparse custody changed during its owned read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _guarded_json(path, *, cancelled=lambda: False, reserve=lambda *_: None):
    """Admit exact metadata before allocation/decoding and retain FD identity."""
    path = Path(path); before = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ValueError("sparse metadata is not a regular owned file")
    _tick(cancelled, reserve, before.st_size * 32 + 65536, "sparse metadata parse and owner binding")
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if identity(os.fstat(descriptor)) != identity(before):
            raise ValueError("sparse metadata changed before its opened read")
        raw = bytearray()
        while len(raw) < before.st_size:
            _tick(cancelled, reserve)
            page = os.read(descriptor, min(65536, before.st_size - len(raw)))
            if not page:
                raise ValueError("sparse metadata ended before its declared bytes")
            raw.extend(page)
        if os.read(descriptor, 1) or identity(os.fstat(descriptor)) != identity(before) or identity(path.lstat()) != identity(before):
            raise ValueError("sparse metadata FD/path changed during its read")
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("sparse metadata must be an object")
        return decoded, sha(raw)
    finally:
        os.close(descriptor)


def _open(path, *, create=False):
    path = Path(path)
    if create:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    elif path.is_symlink() or not path.is_file():
        raise ValueError("sparse bundle/plan is missing or unsafe")
    connection = sqlite3.connect(path if create else path.as_uri() + "?mode=ro", uri=not create)
    try:
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA cache_size=-128")
        connection.execute("PRAGMA mmap_size=0")
        if create:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            for declaration in SCHEMA.values():
                connection.execute(declaration)
            connection.commit()
        else:
            normalize = lambda value: "".join(str(value).split()).rstrip(";").upper()
            seen = set()
            for kind, name, declaration in connection.execute("SELECT type,name,sql FROM sqlite_master"):
                if kind == "index" and name.startswith("sqlite_autoindex_") and declaration is None:
                    continue
                if kind != "table" or name not in SCHEMA or normalize(declaration) != normalize(SCHEMA[name]):
                    raise ValueError("sparse custody contains executable or unknown SQLite schema")
                seen.add(name)
            if seen != set(SCHEMA) or connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("sparse custody schema/integrity check failed")
        return connection
    except BaseException:
        connection.close()
        raise


def _row(kind, record, *, payload=None, updates=None, controls=None):
    from .weighted_vsa_row import payload_parts
    record = _plain(record)
    identifier = _id(record.get("id"))
    if kind not in KINDS or "latent_weight" in record:
        raise ValueError("sparse row kind or obsolete learned floating shadow is unsupported")
    body = {"kind": kind, "id": identifier, "recordJson": canonical(record)}
    if kind == "neurons":
        from .paged_substrate_writer import _neuron_record
        _neuron_record(identifier, record)
        if not isinstance(payload, bytes) or type(updates) is not int or not 0 <= updates < 2**64:
            raise ValueError("sparse neuron lacks exact packed payload/counter")
        body.update(packedPayloadBase64=base64.b64encode(payload).decode("ascii"), updateCountDecimal=str(updates))
    elif kind == "assemblies":
        from .paged_assembly_index import _record_payload
        _identifier, _fingerprint, projected, _checksum = _record_payload(record)
        if projected.decode("utf-8") != body["recordJson"]:
            raise ValueError("sparse assembly metadata requires canonical native structural fields")
    elif kind == "synapses":
        if not isinstance(controls, bytes) or len(controls) != CONTROL.size:
            raise ValueError("sparse synapse lacks exact native controls")
        sign, strength, eligibility, plasticity, stability, uses, updated = CONTROL.unpack(controls)
        if sign not in (-1, 0, 1) or not 1 <= strength <= 16 or uses < 0 or not all(math.isfinite(v) for v in (eligibility, plasticity, stability, updated)):
            raise ValueError("sparse synaptic controls are invalid")
        _id(record.get("source_id")); _id(record.get("target_id"))
        body["controlsBase64"] = base64.b64encode(controls).decode("ascii")
    encoded = canonical(body)
    return identifier, encoded, sha(encoded)


def _decode(body, compatibility):
    from .weighted_vsa_row import payload_parts
    row = json.loads(body)
    allowed = {"kind", "id", "recordJson"} | ({"packedPayloadBase64", "updateCountDecimal"} if row.get("kind") == "neurons" else {"controlsBase64"} if row.get("kind") == "synapses" else set())
    if set(row) != allowed or canonical(row) != body:
        raise ValueError("sparse row is not exact canonical typed data")
    record = json.loads(row["recordJson"])
    if canonical(record) != row["recordJson"] or record.get("id") != row["id"]:
        raise ValueError("sparse row metadata identity failed")
    payload = controls = None; updates = None
    if row["kind"] == "neurons":
        payload = base64.b64decode(row["packedPayloadBase64"], validate=True)
        if base64.b64encode(payload).decode() != row["packedPayloadBase64"]:
            raise ValueError("sparse packed bytes are not canonical")
        updates = int(row["updateCountDecimal"])
        if str(updates) != row["updateCountDecimal"]:
            raise ValueError("sparse update counter is not canonical")
        payload_parts(payload, compatibility["dimensions"])
    elif row["kind"] == "synapses":
        controls = base64.b64decode(row["controlsBase64"], validate=True)
    identifier, encoded, checksum = _row(row["kind"], record, payload=payload, updates=updates, controls=controls)
    if encoded != body:
        raise ValueError("sparse row contains alternate byte encodings")
    return row["kind"], identifier, record, payload, updates, controls, checksum


def _checked_compatibility(value):
    if (not isinstance(value, dict) or set(value) != {"dimensions", "seed", "zeroDeadbandHex"}
            or type(value["dimensions"]) is not int or value["dimensions"] < 1
            or type(value["seed"]) is not int or not isinstance(value["zeroDeadbandHex"], str)):
        raise ValueError("sparse native compatibility metadata is invalid")
    deadband = float.fromhex(value["zeroDeadbandHex"])
    if not math.isfinite(deadband) or not 0 <= deadband < 1 or deadband.hex() != value["zeroDeadbandHex"]:
        raise ValueError("sparse native deadband is invalid")
    return value


def _compatibility(vectors):
    return {"dimensions": vectors.dimensions, "seed": vectors.seed, "zeroDeadbandHex": float(vectors.zero_deadband).hex()}


class CommittedSparseSnapshot:
    """One authenticated root sparse generation; no AdaptiveBrain/_get/load."""
    def __init__(self, engine, brain_id, *, cancelled=lambda: False, reserve=lambda *_: None):
        from .substrate_inspection import PersistedSubstrateView
        self.engine, self.brain_id = Path(engine), _id(brain_id)
        self.cancelled, self.reserve = cancelled, reserve
        metadata, self.metadata_sha = _guarded_json(self.engine / "brain.json", cancelled=cancelled, reserve=reserve)
        config = metadata.get("config", {})
        if config.get("origin_kind") != "ground-up" or any(key in config for key in ("foundation_model_id", "foundationModelId")) or metadata.get("neural_sequence_memory") is not None:
            raise ValueError("sparse transfers accept current native state only")
        pointer = metadata.get("substrate", {}).get("persistence", {})
        generation_id = _hash(pointer.get("activeGeneration"))
        relative = "generations/%s/manifest.json" % generation_id
        if pointer.get("generationManifest") != relative:
            raise ValueError("sparse committed generation path is invalid")
        generation_path = self.engine / "substrate" / relative
        admitted_generation, generation_sha = _guarded_json(generation_path, cancelled=cancelled, reserve=reserve)
        if generation_sha != pointer.get("generationManifestSha256"):
            raise ValueError("sparse committed generation bytes changed")
        self.base_footprint = ((self.engine / "brain.json").stat().st_size + generation_path.stat().st_size) * 32 + 65536
        del admitted_generation
        _tick(cancelled, reserve, self.base_footprint, "sparse committed metadata and generation inventory")
        self.view = PersistedSubstrateView.open(self.engine, brain_id)
        if _file_sha(self.engine / "brain.json", cancelled=cancelled, reserve=reserve) != self.metadata_sha:
            raise ValueError("sparse committed root changed while authenticating its generation")
        if self.view.generation.get("formatVersion") != 3:
            raise ValueError("sparse transfers require authoritative native-v3 assembly state")
        self.generation = self.view.generation["contentSha256"]
        self.compatibility = None

    def rows(self):
        from .packed_vsa_vectors import PackedTernaryVectors
        for kind in KINDS:
            for shard in self.view.kind_shards(kind):
                _tick(self.cancelled, self.reserve, self.base_footprint + 16 * sum(int(shard.get(field, {}).get("bytes", 0)) for field in ("records", "tensors")), "sparse committed shard")
                records = self.view.records(shard)["records"]
                if kind == "neurons":
                    descriptor = self.view.records(shard)["packedVectorState"]
                    vectors = PackedTernaryVectors.from_state(descriptor, self.view.tensors(shard))
                    compatibility = _compatibility(vectors)
                    if self.compatibility is not None and compatibility != self.compatibility:
                        raise ValueError("committed sparse vector owners differ")
                    self.compatibility = compatibility
                    for record in records:
                        _tick(self.cancelled, self.reserve)
                        yield kind, *_row(kind, record, payload=vectors.row_payload(record["id"]), updates=vectors.update_count(record["id"]))
                elif kind == "assemblies":
                    for record in records:
                        _tick(self.cancelled, self.reserve); yield kind, *_row(kind, record)
                else:
                    tensors = self.view.tensors(shard)
                    for index, record in enumerate(records):
                        _tick(self.cancelled, self.reserve)
                        controls = CONTROL.pack(int(tensors["effective_weight"][index]), int(tensors["connection_strength"][index]),
                            float(tensors["eligibility"][index]), float(tensors["plasticity"][index]), float(tensors["stability"][index]),
                            int(tensors["uses"][index]), float(tensors["last_updated_at"][index]))
                        yield kind, *_row(kind, record, controls=controls)
        if self.compatibility is None:
            raise ValueError("an empty sparse owner has no transferable packed geometry")


def inspect_sparse_snapshot(snapshot, query=None, *, cancelled=lambda: False, reserve=lambda *_: None):
    query = query or {}
    if not isinstance(query, dict) or set(query) - {"entity", "cursor", "maxBytes", "stateScope"} or query.get("stateScope", "committed") != "committed":
        raise ValueError("only explicit committed sparse inspection is supported")
    entity = query.get("entity", "all")
    if entity not in (*KINDS, "all"):
        raise ValueError("invalid sparse inspection entity")
    maximum = query.get("maxBytes", 65536)
    if type(maximum) is not int or not 1024 <= maximum <= 262144:
        raise ValueError("invalid sparse inspection byte page")
    offset = 0
    if query.get("cursor") is not None:
        if not isinstance(query["cursor"], str) or len(query["cursor"]) > 2048:
            raise ValueError("sparse inspection cursor exceeds its transport boundary")
        try:
            raw = base64.urlsafe_b64decode(query["cursor"] + "=" * (-len(query["cursor"]) % 4)).decode()
            envelope = json.loads(raw); body = envelope["body"]
            if sha(canonical(body)) != envelope["sha256"] or body[:2] != [snapshot.generation, entity] or type(body[2]) is not int or body[2] < 0:
                raise ValueError()
            offset = body[2]
        except Exception as error:
            raise ValueError("sparse inspection cursor is stale/foreign/invalid") from error
    rows = []; used = 0; seen = 0; more = False
    for kind, _identifier, encoded, checksum in snapshot.rows():
        if entity != "all" and kind != entity:
            continue
        if seen < offset:
            seen += 1; continue
        _tick(cancelled, reserve, len(encoded.encode()) * 6 + 4096, "packed sparse inspection page")
        size = len(encoded.encode()) + 96
        if used + size > maximum:
            if not rows:
                raise ValueError("one exact sparse row exceeds this page; use byte-paged export/read")
            more = True; break
        rows.append({**json.loads(encoded), "rowSha256": checksum}); used += size; seen += 1
    result = {"format": FORMAT, "version": 1, "scope": SCOPE, "stateScope": "committed", "brainId": snapshot.brain_id,
        "generationSha256": snapshot.generation, "rows": rows, "hasMore": more}
    if more:
        body = [snapshot.generation, entity, offset + len(rows)]
        result["nextCursor"] = base64.urlsafe_b64encode(canonical({"body": body, "sha256": sha(canonical(body))}).encode()).decode().rstrip("=")
    return result


def _references(kind, record):
    if kind == "assemblies":
        yield "neurons", _id(record.get("assembly_neuron_id", record["id"]))
        for field in ("neuron_ids", "concept_ids", "statistical_anchor_ids"):
            for identifier in record.get(field, []):
                yield "neurons", _id(identifier)
        for identifier in record.get("child_assembly_ids", []):
            yield "assemblies", _id(identifier)
    elif kind == "synapses":
        yield "neurons", _id(record["source_id"])
        yield "neurons", _id(record["target_id"])


def _manifest(connection, *, cancelled=lambda: False, reserve=lambda *_: None):
    size = connection.execute("SELECT LENGTH(CAST(value AS BLOB)) FROM meta WHERE key='manifest'").fetchone()
    if size is None:
        raise ValueError("sparse custody manifest is missing")
    _tick(cancelled, reserve, int(size[0]) * 32 + 65536, "sparse SQLite manifest before parse")
    row = connection.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
    if row is None:
        raise ValueError("sparse custody manifest is missing")
    value = json.loads(row[0])
    if canonical(value) != row[0] or value.get("scope") != SCOPE or value.get("version") != 1:
        raise ValueError("sparse custody manifest is incompatible")
    return value, row[0]


def _iter_rows(connection, *, cancelled=lambda: False, reserve=lambda *_: None, joined=False, kind=None):
    """Read lengths before materializing potentially large imported JSON/bytes."""
    query = "SELECT r.rowid,LENGTH(CAST(r.body AS BLOB))+LENGTH(CAST(r.id AS BLOB))"
    if joined:
        query += "+COALESCE(LENGTH(CAST(e.before_body AS BLOB)),0) FROM rows r JOIN expected e USING(kind,id)"
    else:
        query += " FROM rows r"
    if kind is not None:
        query += " WHERE r.kind=?"
    query += " ORDER BY r.kind,r.id"
    for rowid, size in connection.execute(query, () if kind is None else (kind,)):
        _tick(cancelled, reserve, int(size) * 16 + 65536, "sparse SQLite row/preimage before materialization")
        fields = "r.kind,r.id,r.body,r.sha" + (",e.before_sha,e.before_body" if joined else "")
        sql = "SELECT " + fields + " FROM rows r" + (" JOIN expected e USING(kind,id)" if joined else "") + " WHERE r.rowid=?"
        row = connection.execute(sql, (rowid,)).fetchone()
        if row is None:
            raise ValueError("sparse immutable row disappeared during traversal")
        yield row


def _fold(connection, *, cancelled=lambda: False, reserve=lambda *_: None):
    digest = hashlib.sha256(); count = 0
    for kind, identifier, body, checksum in _iter_rows(connection, cancelled=cancelled, reserve=reserve):
        _tick(cancelled, reserve, len(body.encode()) * 6 + 4096)
        if sha(body) != checksum:
            raise ValueError("sparse row checksum mismatch")
        digest.update(canonical([kind, identifier, checksum]).encode()); digest.update(b"\n"); count += 1
    return count, digest.hexdigest()


def export_sparse_bundle(snapshot, selection, destination, *, cancelled=lambda: False, reserve=lambda *_: None):
    """Stream the complete selected dependency closure through disk indexes."""
    destination = Path(destination); scratch = destination.with_name(destination.name + ".selection")
    source = _open(scratch, create=True); output = None
    try:
        for kind, identifier, body, checksum in snapshot.rows():
            _tick(cancelled, reserve, len(body.encode()) * 6 + 8192, "sparse export staging")
            source.execute("INSERT INTO rows VALUES(?,?,?,?)", (kind, identifier, body, checksum))
        source.execute("CREATE TEMP TABLE selected(kind TEXT,id TEXT,PRIMARY KEY(kind,id))")
        source.execute("CREATE TEMP TABLE visited(kind TEXT,id TEXT,PRIMARY KEY(kind,id))")
        if selection.get("all") is True:
            source.execute("INSERT INTO selected SELECT kind,id FROM rows")
        else:
            if set(selection) - {"all", *KINDS} or selection.get("all") not in (None, False):
                raise ValueError("invalid sparse selection")
            for kind in KINDS:
                identifiers = selection.get(kind, [])
                if not isinstance(identifiers, list):
                    raise ValueError("sparse selections require explicit ID arrays")
                for identifier in identifiers:
                    source.execute("INSERT OR IGNORE INTO selected VALUES(?,?)", (kind, _id(identifier)))
        while True:
            pending = source.execute("SELECT s.kind,s.id FROM selected s LEFT JOIN visited v ON v.kind=s.kind AND v.id=s.id WHERE v.id IS NULL LIMIT 1").fetchone()
            if pending is None:
                break
            _tick(cancelled, reserve)
            row = source.execute("SELECT body FROM rows WHERE kind=? AND id=?", pending).fetchone()
            if row is None:
                raise ValueError("sparse selection/dependency has no exact native row")
            record = json.loads(json.loads(row[0])["recordJson"])
            for kind, identifier in _references(pending[0], record):
                source.execute("INSERT OR IGNORE INTO selected VALUES(?,?)", (kind, identifier))
            source.execute("INSERT INTO visited VALUES(?,?)", pending)
        # Include every contact internal to the selected closure; never invent
        # links from labels, and disclose contacts outside this induced slice.
        boundary = 0
        for identifier, body in source.execute("SELECT id,body FROM rows WHERE kind='synapses'"):
            record = json.loads(json.loads(body)["recordJson"])
            selected = [source.execute("SELECT 1 FROM selected WHERE kind='neurons' AND id=?", (record[field],)).fetchone() is not None for field in ("source_id", "target_id")]
            if all(selected):
                source.execute("INSERT OR IGNORE INTO selected VALUES('synapses',?)", (identifier,))
            elif any(selected):
                boundary += 1
        output = _open(destination, create=True)
        for row in source.execute("SELECT r.kind,r.id,r.body,r.sha FROM rows r JOIN selected s ON r.kind=s.kind AND r.id=s.id ORDER BY r.kind,r.id"):
            _tick(cancelled, reserve, len(row[2].encode()) * 4 + 8192, "selected sparse bundle")
            output.execute("INSERT INTO rows VALUES(?,?,?,?)", row)
        count, digest = _fold(output, cancelled=cancelled, reserve=reserve)
        manifest = {"format": FORMAT, "version": 1, "scope": SCOPE, "sourceBrainId": snapshot.brain_id,
            "sourceGenerationSha256": snapshot.generation, "sourceMetadataSha256": snapshot.metadata_sha,
            "compatibility": snapshot.compatibility, "rows": count, "rowsSha256": digest,
            "closure": "selected-dependencies-and-induced-contacts", "excludedBoundaryContacts": boundary,
            "stateScope": "committed", "cortexStateIncluded": False, "sourceAuthorityIncluded": False}
        output.execute("INSERT INTO meta VALUES('manifest',?)", (canonical(manifest),)); output.commit(); output.close(); output = None
        return {"manifestJson": canonical(manifest), "manifestSha256": sha(canonical(manifest)), "sha256": _file_sha(destination, cancelled=cancelled, reserve=reserve), "bytes": destination.stat().st_size, "rows": count}
    except BaseException:
        if output is not None:
            output.close(); output = None
        destination.unlink(missing_ok=True)
        raise
    finally:
        source.close(); scratch.unlink(missing_ok=True)


def validate_sparse_bundle(path, expected_sha, *, cancelled=lambda: False, reserve=lambda *_: None):
    if _file_sha(path, cancelled=cancelled, reserve=reserve) != _hash(expected_sha):
        raise ValueError("sparse bundle bytes changed")
    connection = _open(path)
    try:
        manifest, _raw = _manifest(connection, cancelled=cancelled, reserve=reserve)
        if manifest.get("format") != FORMAT or manifest.get("stateScope") != "committed" or manifest.get("cortexStateIncluded") is not False or manifest.get("sourceAuthorityIncluded") is not False:
            raise ValueError("bundle is not an attributable native sparse assembly export")
        _hash(manifest.get("sourceGenerationSha256")); _hash(manifest.get("sourceMetadataSha256")); _id(manifest.get("sourceBrainId"))
        _checked_compatibility(manifest.get("compatibility"))
        count, digest = _fold(connection, cancelled=cancelled, reserve=reserve)
        if (count, digest) != (manifest.get("rows"), manifest.get("rowsSha256")):
            raise ValueError("sparse bundle coverage digest differs")
        for kind, identifier, body, _checksum in _iter_rows(connection, cancelled=cancelled, reserve=reserve):
            decoded = _decode(body, manifest["compatibility"])
            if decoded[:2] != (kind, identifier):
                raise ValueError("sparse bundle indexed identity differs")
            for dependency in _references(kind, decoded[2]):
                if connection.execute("SELECT 1 FROM rows WHERE kind=? AND id=?", dependency).fetchone() is None:
                    raise ValueError("sparse bundle has an unresolved native dependency")
        if _file_sha(path, cancelled=cancelled, reserve=reserve) != expected_sha:
            raise ValueError("sparse bundle changed during structural traversal")
        return manifest
    finally:
        connection.close()


def sparse_bundle_descriptor(path, expected_sha, *, cancelled=lambda: False, reserve=lambda *_: None):
    manifest = validate_sparse_bundle(path, expected_sha, cancelled=cancelled, reserve=reserve)
    encoded = canonical(manifest)
    return {"manifestJson": encoded, "manifestSha256": sha(encoded), "sha256": expected_sha,
        "bytes": Path(path).stat().st_size, "rows": manifest["rows"]}


def prepare_sparse_overlay(target_snapshot, bundle, bundle_sha, destination, mode, *, cancelled=lambda: False, reserve=lambda *_: None):
    if mode not in ("append", "replace-exact"):
        raise ValueError("sparse overlay needs an explicit conflict mode")
    manifest = validate_sparse_bundle(bundle, bundle_sha, cancelled=cancelled, reserve=reserve)
    source = _open(bundle); plan = None
    try:
        plan = _open(destination, create=True)
        plan.execute("CREATE TEMP TABLE desired_fingerprints(fingerprint TEXT PRIMARY KEY,id TEXT NOT NULL)")
        fingerprint_conflicts = 0
        for row in _iter_rows(source, cancelled=cancelled, reserve=reserve):
            plan.execute("INSERT INTO rows VALUES(?,?,?,?)", row)
            plan.execute("INSERT INTO expected VALUES(?,?,NULL,NULL)", row[:2])
            if row[0] == "assemblies":
                record = json.loads(json.loads(row[2])["recordJson"])
                plan.execute("INSERT INTO desired_fingerprints VALUES(?,?)", (record["fingerprint"], row[1]))
        for kind, identifier, body, checksum in target_snapshot.rows():
            _tick(cancelled, reserve, len(body.encode()) * 4 + 4096)
            plan.execute("UPDATE expected SET before_sha=?,before_body=? WHERE kind=? AND id=?", (checksum, body, kind, identifier))
            if kind == "assemblies":
                record = json.loads(json.loads(body)["recordJson"])
                desired = plan.execute("SELECT id FROM desired_fingerprints WHERE fingerprint=?", (record["fingerprint"],)).fetchone()
                if desired is not None and desired[0] != identifier:
                    fingerprint_conflicts += 1
        if manifest["compatibility"] != target_snapshot.compatibility:
            raise ValueError("sparse overlay geometry/seed/deadband is incompatible; use full native-region merge instead")
        preimages = hashlib.sha256(); changes = hashlib.sha256(); additions = replacements = identical = conflicts = 0
        for kind, identifier, checksum, before in plan.execute("SELECT r.kind,r.id,r.sha,e.before_sha FROM rows r JOIN expected e USING(kind,id) ORDER BY r.kind,r.id"):
            preimages.update(canonical([kind, identifier, before]).encode()); preimages.update(b"\n")
            if before is None:
                additions += 1
            elif before == checksum:
                identical += 1
            elif mode == "replace-exact":
                replacements += 1
            else:
                conflicts += 1
            changes.update(canonical([kind, identifier, before, checksum]).encode()); changes.update(b"\n")
        result = {"format": "omni-sparse-substrate-plan", "version": 1, "scope": SCOPE,
            "brainId": target_snapshot.brain_id, "bundleSha256": bundle_sha, "source": manifest,
            "mode": mode, "targetGenerationSha256": target_snapshot.generation,
            "selectedPreimagesSha256": preimages.hexdigest(), "changesSha256": changes.hexdigest(),
            "rows": additions + replacements + identical + conflicts, "additions": additions, "replacements": replacements,
            "identical": identical, "conflicts": conflicts + fingerprint_conflicts,
            "fingerprintConflicts": fingerprint_conflicts, "unrelatedTargetUpdatesPreserved": True}
        if _file_sha(bundle, cancelled=cancelled, reserve=reserve) != bundle_sha:
            raise ValueError("source bundle changed during preview")
        plan.execute("INSERT INTO meta VALUES('manifest',?)", (canonical(result),)); plan.commit(); plan.close(); plan = None
        return {**result, "manifestJson": canonical(result), "manifestSha256": sha(canonical(result)), "sha256": _file_sha(destination, cancelled=cancelled, reserve=reserve), "bytes": Path(destination).stat().st_size}
    except BaseException:
        if plan is not None:
            plan.close(); plan = None
        Path(destination).unlink(missing_ok=True)
        raise
    finally:
        source.close()


def prepare_sparse_rollback(target_snapshot, original_plan, original_plan_sha, receipt_json, receipt_sha, destination,
                            *, cancelled=lambda: False, reserve=lambda *_: None):
    """Reviewed inverse for replacements, never blind removal of added nodes.

    This version cannot authenticate EVERY live working/scratch reference to an
    introduced node from a committed reader. Such removals remain conflicts;
    an apparently empty adjacency list is not a safe deletion proof.
    """
    validate_sparse_transfer_head({"receiptJson": receipt_json, "receiptSha256": receipt_sha})
    receipt = json.loads(receipt_json)
    if receipt["brainId"] != target_snapshot.brain_id or receipt["planSha256"] != original_plan_sha:
        raise ValueError("inverse preview requires this owner's exact applied receipt")
    if _file_sha(original_plan, cancelled=cancelled, reserve=reserve) != _hash(original_plan_sha):
        raise ValueError("original sparse plan changed before inverse review")
    source = _open(original_plan); plan = None
    try:
        original, _raw = _manifest(source, cancelled=cancelled, reserve=reserve)
        if original.get("brainId") != target_snapshot.brain_id or original.get("changesSha256") != receipt["changesSha256"] or original.get("selectedPreimagesSha256") != receipt["selectedPreimagesSha256"]:
            raise ValueError("original sparse plan differs from its applied receipt")
        plan = _open(destination, create=True)
        unsafe_removals = 0
        for kind, identifier, after_body, after_sha, before_sha, before_body in _iter_rows(source, joined=True, cancelled=cancelled, reserve=reserve):
            _tick(cancelled, reserve, len(after_body.encode()) * 8 + (len(before_body.encode()) * 8 if before_body else 0) + 8192, "sparse inverse preimage")
            if sha(after_body) != after_sha or before_body is not None and sha(before_body) != before_sha:
                raise ValueError("original sparse before/after image is corrupt")
            if before_body is None:
                unsafe_removals += 1
                continue
            plan.execute("INSERT INTO rows VALUES(?,?,?,?)", (kind, identifier, before_body, before_sha))
            plan.execute("INSERT INTO expected VALUES(?,?,?,?)", (kind, identifier, after_sha, after_body))
        plan.execute("CREATE TEMP TABLE changed(kind TEXT,id TEXT,PRIMARY KEY(kind,id))")
        plan.execute("CREATE TEMP TABLE seen(kind TEXT,id TEXT,PRIMARY KEY(kind,id))")
        for kind, identifier, body, checksum in target_snapshot.rows():
            current = plan.execute("SELECT before_sha FROM expected WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if current is not None:
                if current[0] != checksum:
                    plan.execute("INSERT OR IGNORE INTO changed VALUES(?,?)", (kind, identifier))
                plan.execute("INSERT INTO seen VALUES(?,?)", (kind, identifier))
                plan.execute("UPDATE expected SET before_body=? WHERE kind=? AND id=?", (body, kind, identifier))
        plan.execute("INSERT OR IGNORE INTO changed SELECT e.kind,e.id FROM expected e LEFT JOIN seen s USING(kind,id) WHERE s.id IS NULL")
        if target_snapshot.compatibility != original["source"]["compatibility"]:
            raise SparseOverlayStale("sparse inverse geometry changed")
        preimages = hashlib.sha256(); changes = hashlib.sha256(); replacements = identical = count = 0
        for kind, identifier, checksum, expected in plan.execute("SELECT r.kind,r.id,r.sha,e.before_sha FROM rows r JOIN expected e USING(kind,id) ORDER BY r.kind,r.id"):
            preimages.update(canonical([kind, identifier, expected]).encode()); preimages.update(b"\n")
            changes.update(canonical([kind, identifier, expected, checksum]).encode()); changes.update(b"\n")
            count += 1
            if checksum == expected:
                identical += 1
            else:
                replacements += 1
        result = {"format": "omni-sparse-substrate-plan", "version": 1, "scope": SCOPE,
            "brainId": target_snapshot.brain_id, "bundleSha256": original["bundleSha256"], "source": original["source"],
            "operation": "rollback", "rollbackOf": {"planSha256": original_plan_sha, "receiptSha256": receipt_sha, "jobId": receipt["jobId"]},
            "mode": "replace-exact", "targetGenerationSha256": target_snapshot.generation,
            "selectedPreimagesSha256": preimages.hexdigest(), "changesSha256": changes.hexdigest(), "rows": count,
            "additions": 0, "replacements": replacements, "identical": identical, "conflicts": unsafe_removals + plan.execute("SELECT COUNT(*) FROM changed").fetchone()[0],
            "unprovenRemovalRows": unsafe_removals, "removalPolicy": "refuse-without-full-live-dependency-and-active-reference-proof",
            "unrelatedTargetUpdatesPreserved": True}
        plan.execute("INSERT INTO meta VALUES('manifest',?)", (canonical(result),)); plan.commit(); plan.close(); plan = None
        return {**result, "manifestJson": canonical(result), "manifestSha256": sha(canonical(result)), "sha256": _file_sha(destination, cancelled=cancelled, reserve=reserve), "bytes": Path(destination).stat().st_size}
    except BaseException:
        if plan is not None:
            plan.close(); plan = None
        Path(destination).unlink(missing_ok=True)
        raise
    finally:
        source.close()


def validate_sparse_transfer_head(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"receiptJson", "receiptSha256"} or sha(value.get("receiptJson", "")) != _hash(value.get("receiptSha256")):
        raise ValueError("native sparse receipt head is invalid")
    body = json.loads(value["receiptJson"])
    fields = {"format", "version", "scope", "brainId", "jobId", "requestSha256", "planSha256", "bundleSha256",
        "selectedPreimagesSha256", "changesSha256", "appliedRows", "committedAt", "weightsAveraged", "cortexParametersChanged", "originChanged"}
    if (not isinstance(body, dict) or set(body) != fields or body.get("format") != "omni-sparse-substrate-applied" or body.get("version") != 1 or body.get("scope") != SCOPE
            or any(body.get(field) is not False for field in ("weightsAveraged", "cortexParametersChanged", "originChanged"))
            or not isinstance(body.get("committedAt"), str)):
        raise ValueError("native sparse receipt head scope is invalid")
    for field in ("brainId", "jobId"):
        _id(body.get(field))
    for field in ("requestSha256", "planSha256", "bundleSha256", "selectedPreimagesSha256", "changesSha256"):
        _hash(body.get(field))
    if canonical(body) != value["receiptJson"] or type(body.get("appliedRows")) is not int or body["appliedRows"] < 0:
        raise ValueError("native sparse receipt body is invalid")
    return dict(value)


def live_sparse_row(memory, kind, identifier):
    if kind == "neurons":
        if identifier not in memory.neurons:
            return None
        return _row(kind, memory.neurons[identifier], payload=memory.neuron_vectors.row_payload(identifier), updates=memory.neuron_vectors.update_count(identifier))
    if kind == "assemblies":
        getter = getattr(memory.assemblies, "get_by_id", None)
        record = getter(identifier) if callable(getter) else next((item for item in memory.assemblies if item["id"] == identifier), None)
        return None if record is None else _row(kind, record)
    record = memory.synapses.get(identifier)
    if record is None:
        return None
    fields = {"effective_weight", "connection_strength", "eligibility", "plasticity", "stability", "uses", "last_updated_at"}
    controls = CONTROL.pack(int(record.get("effective_weight", 0)), int(record.get("connection_strength", 1)), float(record.get("eligibility", 0)),
        float(record.get("plasticity", 1)), float(record.get("stability", 0)), int(record.get("uses", 0)), float(record.get("last_updated_at", 0)))
    return _row(kind, {key: value for key, value in record.items() if key not in fields}, controls=controls)


def _install(memory, decoded):
    kind, identifier, record, payload, updates, controls, _checksum = decoded
    if kind == "neurons":
        vectors = memory.neuron_vectors
        if hasattr(vectors, "write_packed_in_connection"):
            with vectors._transaction(write=True) as connection:
                vectors.write_packed_in_connection(connection, identifier, payload, dimensions=vectors.dimensions, updates=updates, reserve_disk=vectors._disk_reserve)
                revision = vectors.revision_in_connection(connection)
            vectors.invalidate(identifier, revision)
        else:
            if identifier in vectors:
                del vectors[identifier]
            vectors.append_row_payload(identifier, payload, updates)
        memory.neurons[identifier] = record
    elif kind == "assemblies":
        index = getattr(memory.assemblies, "index", None)
        if callable(getattr(index, "upsert", None)):
            index.upsert(record)
        else:
            index = next((i for i, item in enumerate(memory.assemblies) if item["id"] == identifier), None)
            if index is None:
                memory.assemblies.append(record)
            else:
                memory.assemblies[index] = record
    else:
        sign, strength, eligibility, plasticity, stability, uses, updated = CONTROL.unpack(controls)
        memory.synapses[identifier] = {**record, "effective_weight": sign, "connection_strength": strength,
            "eligibility": eligibility, "plasticity": plasticity, "stability": stability, "uses": uses, "last_updated_at": updated}


class SparseOverlayStale(ValueError):
    pass


class SparseOverlayCommitError(RuntimeError):
    def __init__(self, message, committed):
        super().__init__(message)
        self.committed, self.reload_required = committed, True


def apply_sparse_overlay(brain, plan_path, params, *, publication, mutation_scope=nullcontext, cancelled=lambda: False, reserve=lambda *_: None):
    """Loaded-only owner supplied by Worker; never acquire/load another brain.

    publication(head) MUST stage head in normal brain._metadata and call save.
    Caller holds host FIFO and a complete mutation scope (including regional
    stores). Failures fence the RAM owner, which the Worker must release.
    """
    from datetime import datetime, timezone
    from .connected_execution import require_joint_mutation_allowed
    require_joint_mutation_allowed(brain, operation="sparse_overlay")
    if getattr(brain, "_connected_recovery_required", False):
        raise SparseOverlayCommitError("native sparse owner requires explicit reload", False)
    plan_sha = _hash(params["planSha256"])
    if _file_sha(plan_path, cancelled=cancelled, reserve=reserve) != plan_sha:
        raise ValueError("reviewed sparse plan bytes changed")
    previous = validate_sparse_transfer_head(getattr(brain, "sparse_transfer_head", None))
    if previous is not None:
        old = json.loads(previous["receiptJson"])
        if old["brainId"] == brain.brain_id and old["jobId"] == params["jobId"]:
            if old["requestSha256"] != params["requestSha256"] or old["planSha256"] != plan_sha:
                raise ValueError("sparse import request conflicts with its committed receipt")
            return {**old, **previous, "duplicate": True, "generationSha256": brain.memory.persistence_manifest["activeGeneration"]}
        if params.get("acknowledgedReceiptSha256") != previous["receiptSha256"]:
            raise ValueError("archive the previous native sparse receipt before another apply")
    elif params.get("acknowledgedReceiptSha256") is not None:
        raise ValueError("sparse archival acknowledgement has no current native head")
    connection = _open(plan_path)
    mutated = saving = False
    try:
        plan, _raw = _manifest(connection, cancelled=cancelled, reserve=reserve)
        if plan.get("format") != "omni-sparse-substrate-plan" or plan.get("brainId") != brain.brain_id or plan.get("conflicts") != 0:
            raise ValueError("sparse plan is foreign, incompatible or has unresolved conflicts")
        if _compatibility(brain.memory.neuron_vectors) != plan["source"]["compatibility"]:
            raise SparseOverlayStale("native sparse geometry changed after review")
        preimages = hashlib.sha256(); changes = hashlib.sha256(); count = 0
        for kind, identifier, body, checksum, before, before_body in _iter_rows(connection, joined=True, cancelled=cancelled, reserve=reserve):
            _tick(cancelled, reserve, len(body.encode()) * 8 + 8192, "reviewed sparse preimage")
            decoded = _decode(body, plan["source"]["compatibility"])
            if decoded[:2] != (kind, identifier) or decoded[-1] != checksum or (before_body is not None and sha(before_body) != before):
                raise ValueError("sparse plan row/preimage binding is corrupt")
            current = live_sparse_row(brain.memory, kind, identifier)
            if (None if current is None else current[-1]) != before:
                raise SparseOverlayStale("selected sparse row/dependency changed; review a fresh plan")
            if kind == "assemblies":
                getter = getattr(brain.memory.assemblies, "get_by_fingerprint", None)
                record = decoded[2]
                existing = getter(record["fingerprint"]) if callable(getter) else next((item for item in brain.memory.assemblies if item["fingerprint"] == record["fingerprint"]), None)
                if existing is not None and existing["id"] != identifier:
                    raise SparseOverlayStale("an assembly fingerprint acquired another native owner; review a fresh plan")
            preimages.update(canonical([kind, identifier, before]).encode()); preimages.update(b"\n")
            changes.update(canonical([kind, identifier, before, checksum]).encode()); changes.update(b"\n"); count += 1
        if (preimages.hexdigest(), changes.hexdigest(), count) != (plan["selectedPreimagesSha256"], plan["changesSha256"], plan["rows"]):
            raise ValueError("reviewed sparse plan digest or coverage differs")
        if _file_sha(plan_path, cancelled=cancelled, reserve=reserve) != plan_sha:
            raise ValueError("reviewed sparse plan changed during preimage validation")
        with mutation_scope() as transaction:
            # Vectors/metadata first, then assembly aliases, then contacts.
            applied = 0
            for kind in KINDS:
                for _kind, _identifier, body, checksum, before, _before_body in _iter_rows(connection, joined=True, kind=kind, cancelled=cancelled, reserve=reserve):
                    _tick(cancelled, reserve, len(body.encode()) * 8 + 8192, "native sparse row apply")
                    if before == checksum:
                        continue
                    mutated = True; _install(brain.memory, _decode(body, plan["source"]["compatibility"])); applied += 1
            brain.memory.state_revision += 1
            brain.memory.invalidate_assembly_index(); brain.memory.release_paged_recall_scratch()
            brain.memory._recall_graph_cache = None; brain.memory._weighted_representation_cache = None
            body = {"format": "omni-sparse-substrate-applied", "version": 1, "scope": SCOPE, "brainId": brain.brain_id,
                "jobId": _id(params["jobId"]), "requestSha256": _hash(params["requestSha256"]), "planSha256": plan_sha,
                "bundleSha256": plan["bundleSha256"], "selectedPreimagesSha256": plan["selectedPreimagesSha256"], "changesSha256": plan["changesSha256"],
                "appliedRows": applied, "committedAt": datetime.now(timezone.utc).isoformat(),
                "weightsAveraged": False, "cortexParametersChanged": False, "originChanged": False}
            encoded = canonical(body); head = {"receiptJson": encoded, "receiptSha256": sha(encoded)}
            validate_sparse_transfer_head(head); _tick(cancelled, reserve)
            if transaction is not None and callable(getattr(transaction, "prepare_publication", None)):
                transaction.prepare_publication()
            saving = True; publication(head)
            saved, _saved_sha = _guarded_json(brain.engine_path / "brain.json", cancelled=lambda: False, reserve=reserve)
            if saved.get("brain_id") != brain.brain_id or validate_sparse_transfer_head(saved.get("sparse_transfer_head")) != head:
                raise ValueError("ordinary atomic save did not bind the sparse applied receipt")
            return {**body, **head, "generationSha256": saved["substrate"]["persistence"]["activeGeneration"], "duplicate": False}
    except BaseException as error:
        if mutated or saving:
            brain._connected_recovery_required = True
            committed = False
            if saving:
                try:
                    saved, _saved_sha = _guarded_json(brain.engine_path / "brain.json", cancelled=lambda: False, reserve=reserve)
                    current = validate_sparse_transfer_head(saved.get("sparse_transfer_head"))
                    committed = bool(current and json.loads(current["receiptJson"])["jobId"] == params["jobId"] and json.loads(current["receiptJson"])["planSha256"] == plan_sha)
                except Exception:
                    committed = None
            raise SparseOverlayCommitError(str(error), committed) from error
        raise
    finally:
        connection.close()
