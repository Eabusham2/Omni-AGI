"""One replayable current-ACK audit, never an unbounded native history."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import nullcontext
from pathlib import Path

ZERO = "0" * 64
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_COUNTS = ("blockCount", "textBlocks", "resourceTextBlocks", "referenceBlocks", "structuredBlocks", "imageBlocks",
           "audioBlocks", "decodedWindows", "imageFrames", "audioModelSamples", "sourceMediaBytes")
_KINDS = {"text": "textBlocks", "resource-text": "resourceTextBlocks", "resource-reference": "referenceBlocks",
          "structured": "structuredBlocks", "image": "imageBlocks", "audio": "audioBlocks"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def count(value, minimum=0):
    if type(value) is not int or not minimum <= value <= (1 << 53) - 1:
        raise ValueError("App content count is invalid")
    return value


def sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("App content hash is invalid")
    return value


def order_hash(previous, row):
    body = json.dumps([row[key] for key in ("blockIndex", "kind", "sourceEncoding", "sourceSha256", "sourceBytes")],
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((sha(previous) + "\0" + body).encode("utf-8")).hexdigest()


def validate_content(value, *, record_sha=None, data_sha=None):
    if not isinstance(value, dict) or set(value) != {"summary", "audit"}:
        raise ValueError("Native App content receipt is invalid")
    summary, audit = value["summary"], value["audit"]
    if (not isinstance(summary, dict) or set(summary) != {"format", "version", *_COUNTS,
            "orderedInputSha256", "wholeInputTraversed", "nativeSensory"}
            or summary["format"] != "omni-native-app-content-summary" or type(summary["version"]) is not int
            or summary["version"] != 1 or summary["wholeInputTraversed"] is not True):
        raise ValueError("Native App content coverage is invalid")
    for key in _COUNTS:
        count(summary[key])
    if (summary["blockCount"] < 1 or summary["blockCount"] != sum(summary[key] for key in _KINDS.values())
            or summary["structuredBlocks"] > 1 or type(summary["nativeSensory"]) is not bool
            or summary["nativeSensory"] != (summary["imageBlocks"] + summary["audioBlocks"] > 0)
            or summary["decodedWindows"] < summary["imageBlocks"] + summary["audioBlocks"]
            or summary["imageFrames"] < summary["imageBlocks"]
            or summary["imageFrames"] > summary["decodedWindows"]
            or summary["audioModelSamples"] < summary["audioBlocks"]
            or summary["nativeSensory"] and summary["sourceMediaBytes"] < 1
            or not summary["nativeSensory"] and any(summary[key] for key in ("decodedWindows", "imageFrames", "audioModelSamples", "sourceMediaBytes"))):
        raise ValueError("Native App content coverage counts contradict its actual input kinds")
    sha(summary["orderedInputSha256"])
    if (not isinstance(audit, dict) or set(audit) != {"format", "version", "path", "sha256", "bytes", "rows", "sourceBrainId",
            "recordSha256", "dataSha256", "orderedInputSha256"} or audit["format"] != "omni-native-app-observation-audit"
            or type(audit["version"]) is not int or audit["version"] != 1
            or not isinstance(audit["sourceBrainId"], str) or not _ID.fullmatch(audit["sourceBrainId"])):
        raise ValueError("Native App current audit descriptor is invalid")
    for key in ("sha256", "recordSha256", "dataSha256", "orderedInputSha256"):
        sha(audit[key])
    if (audit["path"] != "app-observation-audits/" + audit["sha256"] + ".jsonl"
            or count(audit["bytes"], 1) < audit["rows"] or count(audit["rows"], 1) != summary["blockCount"]
            or audit["orderedInputSha256"] != summary["orderedInputSha256"]
            or record_sha is not None and audit["recordSha256"] != record_sha
            or data_sha is not None and audit["dataSha256"] != data_sha):
        raise ValueError("Native App current audit lost its exact content/coverage binding")
    return copy.deepcopy(value)


def content_hash_values(content):
    content = validate_content(content)
    summary, audit = content["summary"], content["audit"]
    return [[summary[key] for key in ("format", "version", *_COUNTS, "orderedInputSha256", "wholeInputTraversed", "nativeSensory")],
            [audit[key] for key in ("format", "version", "path", "sha256", "bytes", "rows", "sourceBrainId", "recordSha256", "dataSha256", "orderedInputSha256")]]


def validate_block(row):
    required = {"format", "version", "sourceBrainId", "recordSha256", "dataSha256", "blockIndex", "kind", "sourceEncoding",
                "sourceSha256", "sourceBytes", "wholeInputTraversed", "nativeSensory", "fetched"}
    if (not isinstance(row, dict) or set(row) - {"media"} != required or row["format"] != "omni-native-app-observation-block"
            or type(row["version"]) is not int or row["version"] != 1 or row["kind"] not in _KINDS
            or not isinstance(row["sourceBrainId"], str) or not _ID.fullmatch(row["sourceBrainId"])
            or row["wholeInputTraversed"] is not True or row["fetched"] is not False):
        raise ValueError("Native App measured block receipt is invalid")
    for key in ("recordSha256", "dataSha256", "sourceSha256"):
        sha(row[key])
    count(row["blockIndex"]); count(row["sourceBytes"])
    sensory = row["kind"] in {"image", "audio"}
    encoding = "decoded-inline-media" if sensory else "approved-data-json" if row["kind"] in {"structured", "resource-reference"} else "utf8-text"
    if row["sourceEncoding"] != encoding or row["nativeSensory"] is not sensory:
        raise ValueError("Native App block representation does not match its measured source kind")
    if sensory:
        media = row.get("media")
        optional = {"sourceFrames", "sourceSampleFrames", "sourceSampleRate", "modelSamples", "modelSampleRate"}
        required_media = {"messageIndex", "blockIndex", "kind", "mimeType", "sourceBytes", "sourceSha256", "decodedWindows", "wholeInputTraversed", "nativeSensory"}
        if (not isinstance(media, dict) or set(media) - optional != required_media or media["messageIndex"] != 0
                or media["blockIndex"] != row["blockIndex"] or media["kind"] != row["kind"]
                or media["sourceBytes"] != row["sourceBytes"] or media["sourceSha256"] != row["sourceSha256"]
                or not isinstance(media["mimeType"], str) or media["wholeInputTraversed"] is not True or media["nativeSensory"] is not True):
            raise ValueError("Native App media receipt lost its actual decoded source")
        count(media["decodedWindows"], 1)
        count(media["messageIndex"]); count(media["blockIndex"]); count(media["sourceBytes"], 1)
        for key in optional.intersection(media):
            count(media[key], 1)
        if row["kind"] == "image" and media.get("sourceFrames") != media["decodedWindows"]:
            raise ValueError("Native App image frame coverage differs from encoded windows")
        if row["kind"] == "audio" and (not media.get("modelSamples") or media.get("modelSampleRate") != 16000):
            raise ValueError("Native App audio receipt lacks actual native samples")
    elif "media" in row:
        raise ValueError("Text/reference App data cannot claim sensory decoding")
    return row


def safe_audit_path(engine, descriptor):
    engine = Path(engine)
    directory = engine / "app-observation-audits"
    if descriptor.get("path") != "app-observation-audits/" + sha(descriptor.get("sha256")) + ".jsonl":
        raise ValueError("Native App audit path is not its exact owned content address")
    target = engine / descriptor["path"]
    if engine.is_symlink() or not engine.is_dir() or directory.is_symlink() or not directory.is_dir() or target.is_symlink():
        raise ValueError("Native App audit cannot follow a symlink or missing private directory")
    return target


def verify_audit_file(engine, descriptor, *, policy=None, return_identity=False):
    path = safe_audit_path(engine, descriptor)
    lease = getattr(policy, "reserve_ram", None)
    with lease(65536 + 8192, "current App audit integrity window") if callable(lease) else nullcontext():
        descriptor_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor_fd, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != descriptor["bytes"]:
                raise ValueError("Native App audit file size/type changed")
            digest = hashlib.sha256()
            for piece in iter(lambda: source.read(65536), b""):
                digest.update(piece)
            after, current = os.fstat(source.fileno()), path.lstat()
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if (identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                    or digest.hexdigest() != descriptor["sha256"]):
                raise ValueError("Native App current audit bytes or source inode changed")
    return (path, identity) if return_identity else path


class AppObservationAuditWriter:
    def __init__(self, brain, approved):
        self.brain, self.approved = brain, approved
        self.directory = Path(brain.engine_path) / "app-observation-audits"
        if Path(brain.engine_path).is_symlink() or self.directory.is_symlink():
            raise ValueError("Native App audit directory is unsafe")
        self.directory.mkdir(mode=0o700, exist_ok=True)
        stat_dir = self.directory.stat()
        self.directory_identity = (stat_dir.st_dev, stat_dir.st_ino)
        self.path = self.directory / (".audit-" + uuid.uuid4().hex + ".tmp")
        brain.resource_policy.require_disk(65536, "current App audit staging")
        self.handle = os.fdopen(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600), "wb")
        opened = os.fstat(self.handle.fileno())
        self.identity = (opened.st_dev, opened.st_ino)
        self.digest, self.bytes, self.rows = hashlib.sha256(), 0, 0
        self.owned, self.committed = True, False
        self.summary = {"format": "omni-native-app-content-summary", "version": 1,
            **{key: 0 for key in _COUNTS}, "orderedInputSha256": ZERO, "wholeInputTraversed": True, "nativeSensory": False}

    def add(self, row):
        row = validate_block(row)
        if (row["blockIndex"] != self.rows or row["sourceBrainId"] != self.brain.brain_id
                or row["recordSha256"] != self.approved.record_sha256 or row["dataSha256"] != self.approved.data_sha256):
            raise ValueError("Native App audit rows must preserve exact reviewed FIFO ownership")
        raw = canonical(row) + b"\n"
        self.brain.resource_policy.require_disk(self.bytes + len(raw) + 65536, "current App audit rows")
        if self.handle.write(raw) != len(raw):
            raise OSError("Native App audit had a short write")
        self.digest.update(raw); self.bytes += len(raw); self.rows += 1
        self.summary["blockCount"] += 1
        self.summary[_KINDS[row["kind"]]] += 1
        self.summary["orderedInputSha256"] = order_hash(self.summary["orderedInputSha256"], row)
        if row["nativeSensory"]:
            media = row["media"]
            self.summary["nativeSensory"] = True
            self.summary["decodedWindows"] += media["decodedWindows"]
            self.summary["sourceMediaBytes"] += media["sourceBytes"]
            self.summary["imageFrames"] += media.get("sourceFrames", 0)
            self.summary["audioModelSamples"] += media.get("modelSamples", 0)

    def finalize(self):
        self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close(); self.handle = None
        directory = self.directory.lstat()
        if not stat.S_ISDIR(directory.st_mode) or (directory.st_dev, directory.st_ino) != self.directory_identity:
            raise RuntimeError("Native App audit staging directory changed")
        digest = self.digest.hexdigest()
        audit = {"format": "omni-native-app-observation-audit", "version": 1,
            "path": "app-observation-audits/" + digest + ".jsonl", "sha256": digest, "bytes": self.bytes, "rows": self.rows,
            "sourceBrainId": self.brain.brain_id, "recordSha256": self.approved.record_sha256,
            "dataSha256": self.approved.data_sha256, "orderedInputSha256": self.summary["orderedInputSha256"]}
        target = self.directory / (digest + ".jsonl")
        try:
            os.link(self.path, target, follow_symlinks=False)
        except FileExistsError:
            verify_audit_file(self.brain.engine_path, audit, policy=self.brain.resource_policy)
            self._unlink_owned(); self.path, self.owned = target, False
        else:
            self._unlink_owned(); self.path = target
        from .joint_generation import _fsync_directory
        _fsync_directory(self.directory)
        return validate_content({"summary": self.summary, "audit": audit})

    def _unlink_owned(self):
        if not self.owned:
            return
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self.identity:
            raise RuntimeError("Native App audit staging was replaced; replacement was not deleted")
        self.path.unlink()

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None
        if not self.committed:
            self._unlink_owned()


def app_observation_audit_snapshot_files(engine, metadata, *, policy=None):
    from .app_observation import validate_frontier_head
    head = validate_frontier_head(metadata.get("app_observation_frontier"))
    if head is None or head.get("content") is None:
        return {}
    descriptor = head["content"]["audit"]
    return {descriptor["path"]: verify_audit_file(engine, descriptor, policy=policy)}


def copy_app_observation_audit_snapshot(source, destination, metadata, *, policy=None):
    for relative, path in app_observation_audit_snapshot_files(source, metadata, policy=policy).items():
        target = Path(destination) / relative
        if policy is not None:
            policy.require_disk(path.stat().st_size + 65536, "current App ACK audit snapshot")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.is_symlink() or target.parent.is_symlink():
            raise ValueError("App ACK audit snapshot target is unsafe")
        descriptor = metadata["app_observation_frontier"]["content"]["audit"]
        if target.exists():
            verify_audit_file(destination, descriptor, policy=policy)
            continue
        with path.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=65536)
            writer.flush(); os.fsync(writer.fileno())
    app_observation_audit_snapshot_files(destination, metadata, policy=policy)


def prune_archived_previous_audit(brain, previous, candidate, archived_sha):
    old = (previous or {}).get("content")
    if not old or old["audit"]["sha256"] != archived_sha:
        return
    current = candidate.get("content")
    if current and current["audit"]["sha256"] == archived_sha:
        return
    path, identity = verify_audit_file(brain.engine_path, old["audit"], policy=brain.resource_policy, return_identity=True)
    current_inode = path.lstat()
    if (current_inode.st_dev, current_inode.st_ino, current_inode.st_size, current_inode.st_mtime_ns) != identity:
        raise RuntimeError("Archived App audit inode changed before cleanup")
    path.unlink()
    from .joint_generation import _fsync_directory
    _fsync_directory(path.parent)
