"""Checked streaming metadata indexing; assembly member arrays stay on disk.

The immutable source FD is hashed before parsing and identity-checked after.
Only display metadata is retained. Exact ordered neuron/child memberships are
inserted incrementally, including duplicates, rather than decoding a giant
assembly object/list in Python. This is a derived inspector, not learned state.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat

from .authenticated_paged_cache import canonical
from .text_spool import BoundedCharacters

_HEADERS = frozenset({"id", "assembly_neuron_id", "label", "region", "activation", "importance", "uncertainty",
    "exposures", "created_at", "last_activated_at", "kind", "source", "source_label", "confidence", "rehearsals", "last_recalled_at"})


class _JSON:
    def __init__(self, source, admit, check):
        self.chars = BoundedCharacters(source)
        self.admit, self.check = admit, check
        self.held = 512 * 1024
        self.admit(self.held, "inspection streaming JSON scratch")

    def scalar(self, first, *, keep=True):
        if first in "[{" or not first:
            raise ValueError("inspection metadata requires a scalar")
        parts, size = ([first] if keep else []), 1
        if first == '"':
            escaped, unicode_left = False, 0
            while True:
                value = self.chars.get()
                if not value:
                    raise ValueError("unterminated inspection JSON string")
                size += 1
                if keep:
                    if size % 4096 == 0:
                        self.check(); self.admit(self.held + size * 16, "inspection scalar metadata")
                    parts.append(value)
                elif size % 4096 == 0:
                    self.check()
                if unicode_left:
                    if value not in "0123456789abcdefABCDEF": raise ValueError("invalid JSON unicode escape")
                    unicode_left -= 1
                elif escaped:
                    if value not in '"\\/bfnrtu': raise ValueError("invalid JSON escape")
                    unicode_left = 4 if value == "u" else 0
                    escaped = False
                elif value == "\\": escaped = True
                elif value == '"': break
                elif ord(value) < 32: raise ValueError("invalid JSON string control character")
        else:
            # JSON number/boolean/null tokens are small in authentic metadata;
            # large numeric literals still receive actual admission, not a cap.
            while True:
                value = self.chars.get()
                if not value or value in ",]} \t\r\n":
                    if value: self.chars.unread(value)
                    break
                parts.append(value); size += 1
                if size % 4096 == 0:
                    self.check(); self.admit(self.held + size * 16, "inspection numeric metadata")
            if not keep:
                parts.insert(0, first)
        if not keep and first == '"': return None
        self.admit(self.held + size * 16, "inspection scalar metadata")
        value = json.loads("".join(parts), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON constant")))
        return value if keep else None

    def array(self, first, consume):
        if first != "[": raise ValueError("inspection source array is invalid")
        following, ordinal = self.chars.nonspace(), 0
        if following == "]": return 0
        while True:
            self.check(); consume(following, ordinal); ordinal += 1
            following = self.chars.nonspace()
            if following == "]": return ordinal
            if following != ",": raise ValueError("inspection JSON array separator is invalid")
            following = self.chars.nonspace()

    def mapping(self, first, consume):
        if first != "{": raise ValueError("inspection source object is invalid")
        following, known = self.chars.nonspace(), set()
        if following == "}": return
        while True:
            self.check()
            if following != '"': raise ValueError("inspection JSON object key is invalid")
            key = self.scalar(following)
            if key in known: raise ValueError("inspection JSON has duplicate keys")
            known.add(key); self.admit(self.held + len(known) * 512, "inspection object keys")
            if self.chars.nonspace() != ":": raise ValueError("inspection JSON object colon is invalid")
            consume(key, self.chars.nonspace())
            following = self.chars.nonspace()
            if following == "}": return
            if following != ",": raise ValueError("inspection JSON object separator is invalid")
            following = self.chars.nonspace()

    def skip(self, first):
        if first == "{": self.mapping(first, lambda _, value: self.skip(value))
        elif first == "[": self.array(first, lambda value, _: self.skip(value))
        else: self.scalar(first, keep=False)


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def stream_metadata_shard(view, shard, connection, *, admit, reserve_disk, check):
    """Insert bounded headers and exact member rows from one authentic shard."""
    spec = shard.get("records", {})
    sha, size = spec.get("sha256"), spec.get("bytes")
    if (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            or type(size) is not int or size < 0 or spec.get("path") != "blobs/%s.json" % sha):
        raise ValueError("inspection streaming source descriptor is invalid")
    path = view.store / spec["path"]
    root = view.store.resolve()
    path.parent.resolve().relative_to(root)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode): raise ValueError("inspection source must be a regular owned file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as binary:
        identity = _identity(os.fstat(binary.fileno()))
        if identity != _identity(before) or identity[2] != size:
            raise ValueError("inspection source file identity differs")
        admit(512 * 1024, "inspection source hash and parser scratch")
        digest, observed = hashlib.sha256(), 0
        for piece in iter(lambda: binary.read(64 * 1024), b""):
            check(); observed += len(piece)
            if observed > size: raise ValueError("inspection source grew beyond declared bytes")
            digest.update(piece)
        if observed != size or digest.hexdigest() != sha: raise ValueError("inspection source checksum differs")
        binary.seek(0)
        source = io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline="")
        parser = _JSON(source, admit, check)
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS checked_shard_ids(field TEXT,ordinal INTEGER,id TEXT,PRIMARY KEY(field,ordinal)) WITHOUT ROWID")
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS checked_members(kind TEXT,ordinal INTEGER,id TEXT,PRIMARY KEY(kind,ordinal)) WITHOUT ROWID")
        connection.execute("DELETE FROM checked_shard_ids")
        kind, root_fields, records = shard["kind"], {}, 0
        charged = 0

        def charge(amount):
            nonlocal charged
            if amount > charged:
                reserve_disk(max(64 * 1024, amount - charged), "inspection streaming index rows")
                charged = amount + 64 * 1024

        def identifier(first):
            value = parser.scalar(first)
            if not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 33 or ord(c) == 127 for c in value):
                raise ValueError("inspection source member identity is invalid")
            return value

        def record(first, ordinal):
            nonlocal records
            parser.held = 512 * 1024
            header, counts = {}, {"neuron": 0, "assembly": 0}
            connection.execute("DELETE FROM checked_members")

            def field(name, value):
                if name in ("neuron_ids", "child_assembly_ids") and kind == "assemblies":
                    member_kind = "neuron" if name == "neuron_ids" else "assembly"
                    def member(token, index):
                        member_id = identifier(token)
                        charge(connection.total_changes * 1024 + len(member_id) * 8)
                        connection.execute("INSERT INTO checked_members VALUES(?,?,?)", (member_kind, index, member_id))
                    counts[member_kind] = parser.array(value, member)
                elif name in _HEADERS:
                    header[name] = parser.scalar(value)
                    parser.held = 512 * 1024 + sum(len(str(item)) * 16 for item in header.values())
                elif name == "aliases" and kind == "neurons":
                    aliases = []
                    def alias(token, _):
                        aliases.append(identifier(token)); admit(parser.held + sum(len(item) * 16 for item in aliases), "inspection aliases")
                    parser.array(value, alias); header[name] = aliases
                else:
                    if name == "source_text": header["__inspection_retains_source_text"] = True
                    parser.skip(value)
            parser.mapping(first, field)
            actual_id = header.get("id")
            if not isinstance(actual_id, str) or not actual_id or len(actual_id) > 512 or any(ord(c) < 33 or ord(c) == 127 for c in actual_id):
                raise ValueError("inspection source record identity is invalid")
            if kind == "assemblies":
                header.update(neuron_ids=[], child_assembly_ids=[],
                    __inspection_neuron_count=counts["neuron"], __inspection_child_count=counts["assembly"])
                if header.get("assembly_neuron_id") is not None and header["assembly_neuron_id"] != actual_id:
                    raise ValueError("inspection assembly no longer aliases its actual neuron row")
                member_bytes = connection.execute("SELECT COALESCE(sum(length(id)),0) FROM checked_members").fetchone()[0]
                reserve_disk(65536 + (counts["neuron"] + counts["assembly"]) * 128 + member_bytes * 4,
                             "inspection complete streamed membership publication")
                connection.execute("INSERT INTO assembly_members SELECT ?,kind,ordinal,id FROM checked_members", (actual_id,))
            region = str(header.get("region", "semantic")) if kind == "neurons" else "assembly"
            label = str(header.get("label", header.get("source_label", header.get("kind", "assembly"))))
            encoded = canonical(header)
            admit(parser.held + len(encoded) * 8, "inspection bounded record header")
            charge(connection.total_changes * 1024 + len(encoded) * 8)
            connection.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (kind, actual_id, encoded, region.casefold(),
                (actual_id + " " + label + " " + region + " " + str(header.get("kind", ""))).casefold(), None, None))
            connection.execute("INSERT INTO checked_shard_ids VALUES('records',?,?)", (ordinal, actual_id))
            records += 1

        def top(name, value):
            if name == "records": parser.array(value, record)
            elif name in ("ids", "vectorIds"):
                root_fields[name] = True
                def saved_id(token, ordinal):
                    actual_id = identifier(token); charge(connection.total_changes * 1024 + len(actual_id) * 8)
                    connection.execute("INSERT INTO checked_shard_ids VALUES(?,?,?)", (name, ordinal, actual_id))
                parser.array(value, saved_id)
            elif name == "packedVectorState" and kind == "neurons":
                root_fields[name] = True
                def packed_field(field, token):
                    if field == "ids":
                        root_fields["packedIds"] = True
                        def packed_id(value, ordinal):
                            actual_id = identifier(value); charge(connection.total_changes * 1024 + len(actual_id) * 8)
                            connection.execute("INSERT INTO checked_shard_ids VALUES('packedIds',?,?)", (ordinal, actual_id))
                        parser.array(token, packed_id)
                    else: parser.skip(token)
                parser.mapping(value, packed_field)
            elif name in ("kind", "vectorStorage"):
                root_fields[name] = parser.scalar(value)
            else: parser.skip(value)
        parser.mapping(parser.chars.nonspace(), top)
        if parser.chars.nonspace(): raise ValueError("inspection source has a trailing JSON value")
        if root_fields.get("kind") != kind or records != shard["count"]:
            raise ValueError("inspection streamed shard coverage differs")
        if int(view.generation.get("formatVersion", 0)) == 3:
            for name in ("ids", "vectorIds") + (("packedIds",) if kind == "neurons" else ()):
                mismatch = connection.execute("SELECT 1 FROM checked_shard_ids AS r LEFT JOIN checked_shard_ids AS i ON i.field=? AND i.ordinal=r.ordinal WHERE r.field='records' AND (i.id IS NULL OR i.id<>r.id) LIMIT 1", (name,)).fetchone()
                count = connection.execute("SELECT count(*) FROM checked_shard_ids WHERE field=?", (name,)).fetchone()[0]
                if not root_fields.get(name) or mismatch or count != records: raise ValueError("inspection streamed source IDs differ")
            if kind == "neurons" and (not root_fields.get("packedVectorState") or not isinstance(shard.get("tensors"), dict)):
                raise ValueError("inspection neuron source has no packed-vector identity")
            if kind == "assemblies" and (root_fields.get("vectorStorage") != "shared-neuron-packed" or shard.get("tensors") is not None):
                raise ValueError("inspection assembly source storage differs")
        check()
        if _identity(os.fstat(binary.fileno())) != identity or _identity(path.lstat()) != identity:
            raise ValueError("inspection source changed during streamed indexing")
        source.detach()
        return records
