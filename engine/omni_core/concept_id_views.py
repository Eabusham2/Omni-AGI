"""Immutable structural ID argument views, never text or neural authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from collections.abc import Sequence

from .authenticated_paged_cache import canonical


FORMAT = "omni-structural-concept-id-view"
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_FIELDS = {"format", "version", "brainId", "turnId", "path", "sha256", "bytes", "count"}


def validate_descriptor(value, *, brain_id: str, turn_id: str):
    if not isinstance(value, dict) or set(value) != _FIELDS or (
        value.get("format") != FORMAT or type(value.get("version")) is not int or value.get("version") != 1
        or value.get("brainId") != brain_id or value.get("turnId") != turn_id
        or not isinstance(brain_id, str) or not isinstance(turn_id, str)
        or not _OWNER.fullmatch(brain_id) or not _OWNER.fullmatch(turn_id)
        or not isinstance(value.get("sha256"), str) or not _SHA.fullmatch(value["sha256"])
        or type(value.get("bytes")) is not int or value["bytes"] < 0
        or type(value.get("count")) is not int or value["count"] < 0
        or value.get("path") != "state/concept-id-views/%s.jsonl" % value["sha256"]
    ):
        raise ValueError("concept ID view descriptor/brain/turn ownership is invalid")
    return dict(value)


def _id(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or any(char.isspace() or ord(char) < 33 or ord(char) == 127 for char in value):
        raise ValueError("concept ID view contains a non-structural identifier")
    return value


def _directory(engine: Path):
    root = Path(engine).resolve()
    for path in (root / "state", root / "state" / "concept-id-views"):
        if path.is_symlink():
            raise ValueError("concept ID view directory must be owned, not a symlink")
    return root, root / "state" / "concept-id-views"


def publish_id_view(engine: Path, identifiers: Sequence[str], *, brain_id: str, turn_id: str, reserve=None):
    if not _OWNER.fullmatch(brain_id) or not _OWNER.fullmatch(turn_id):
        raise ValueError("concept ID view needs strict brain/turn ownership")
    cache = getattr(identifiers, "_published_concept_view", None)
    if cache is not None and cache[0] == (str(Path(engine).resolve()), brain_id, turn_id):
        return dict(cache[1])
    root, directory = _directory(engine)
    if reserve is not None and reserve(65536, "concept ID view initialization") is False:
        raise RuntimeError("concept ID view paused at resource reserve")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".id-view-", dir=directory)
    temporary = Path(temporary_name)
    count = len(identifiers)
    header = {"format": FORMAT, "version": 1, "brainId": brain_id, "turnId": turn_id, "count": count}
    digest = hashlib.sha256()
    size = seen = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            block = canonical(header) + b"\n"
            handle.write(block)
            digest.update(block)
            size += len(block)
            for identifier in identifiers:
                block = canonical(_id(identifier)) + b"\n"
                if reserve is not None and reserve(4096 + len(block), "concept ID argument view") is False:
                    raise RuntimeError("concept ID view paused at disk reserve")
                handle.write(block)
                digest.update(block)
                size += len(block)
                seen += 1
            if seen != count:
                raise ValueError("concept ID view source coverage changed")
            handle.flush()
            os.fsync(handle.fileno())
        sha = digest.hexdigest()
        destination = directory / (sha + ".jsonl")
        if destination.exists():
            checked = {**header, "path": "state/concept-id-views/%s.jsonl" % sha, "sha256": sha, "bytes": size}
            IdView(root, checked, brain_id=brain_id, turn_id=turn_id)
        else:
            os.replace(temporary, destination)
        value = {**header, "path": "state/concept-id-views/%s.jsonl" % sha, "sha256": sha, "bytes": size}
        try:
            identifiers._published_concept_view = ((str(root), brain_id, turn_id), value)
        except AttributeError:
            pass
        return value
    finally:
        temporary.unlink(missing_ok=True)


class IdView(Sequence[str]):
    """Two bounded passes: authenticate complete bytes, then stream every ID."""

    def __init__(self, engine: Path, descriptor, *, brain_id: str, turn_id: str):
        self.descriptor = validate_descriptor(descriptor, brain_id=brain_id, turn_id=turn_id)
        root, directory = _directory(engine)
        self.path = root / self.descriptor["path"]
        if self.path.parent != directory or self.path.is_symlink() or not self.path.is_file():
            raise ValueError("concept ID view source is missing or unsafe")
        before = self.path.stat()
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        after = self.path.stat()
        self.identity = self._identity(after)
        if self._identity(before) != self.identity or after.st_size != self.descriptor["bytes"] or digest.hexdigest() != self.descriptor["sha256"]:
            raise ValueError("concept ID view checksum/identity mismatch")
        # Verify the complete structural grammar/coverage, not a prefix only.
        seen = sum(1 for _identifier in self)
        if seen != self.descriptor["count"]:
            raise ValueError("concept ID view coverage differs")

    @staticmethod
    def _identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def __len__(self):
        return self.descriptor["count"]

    def __iter__(self):
        if self._identity(self.path.stat()) != self.identity:
            raise ValueError("immutable concept ID view changed")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            header_line = handle.readline(4096)
            digest.update(header_line)
            header = json.loads(header_line)
            expected = {key: self.descriptor[key] for key in ("format", "version", "brainId", "turnId", "count")}
            if header != expected or header_line != canonical(header) + b"\n":
                raise ValueError("concept ID view source ownership/header differs")
            seen = 0
            while True:
                line = handle.readline(4096)
                if not line:
                    break
                digest.update(line)
                value = _id(json.loads(line))
                if line != canonical(value) + b"\n":
                    raise ValueError("concept ID view is not canonical")
                seen += 1
                if seen > len(self):
                    raise ValueError("concept ID view exceeds declared coverage")
                yield value
        if seen != len(self) or digest.hexdigest() != self.descriptor["sha256"] or self._identity(self.path.stat()) != self.identity:
            raise ValueError("concept ID view changed during complete traversal")

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [value for position, value in enumerate(self) if start <= position < stop and (position - start) % step == 0]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("concept ID view position is out of range")
        for position, value in enumerate(self):
            if position == index:
                return value
        raise ValueError("concept ID view coverage changed")


def unique_ids(identifiers, directory: Path, *, reserve=None):
    """Keep exact first occurrence order without a corpus-sized Python set."""

    if getattr(identifiers, "unique", False):
        yield from identifiers
        return
    if directory.is_symlink():
        raise ValueError("unique concept scratch directory must be owned, not a symlink")
    if reserve is not None and reserve(65536, "unique concept scratch") is False:
        raise RuntimeError("unique concept scratch paused at resource reserve")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unique-concepts-", dir=directory) as folder:
        connection = sqlite3.connect(Path(folder) / "ids.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-256")
            connection.execute("CREATE TABLE ids(identifier TEXT PRIMARY KEY) WITHOUT ROWID")
            for value in identifiers:
                identifier = _id(str(value))
                if reserve is not None and reserve(4096 + len(identifier) * 4, "unique concept argument") is False:
                    raise RuntimeError("unique concept scratch paused at resource reserve")
                inserted = connection.execute("INSERT OR IGNORE INTO ids VALUES(?)", (identifier,)).rowcount
                if inserted:
                    yield identifier
        finally:
            connection.close()
