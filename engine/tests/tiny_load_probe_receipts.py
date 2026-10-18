"""Standard-library-only custody observations for the opt-in tiny Load probe.

These helpers never import a neural runtime or open the preserved source SQLite
databases. Only an independent private copy may be opened for logical SQL reads.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
import re
import shutil
import sqlite3
import stat
from urllib.parse import quote


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_receipt(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tree_snapshot(root, *, include=None):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("fixture snapshot needs its explicit non-symlink directory")
    output = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("fixture custody cannot follow a directory symlink")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if include is not None and not include(relative):
                continue
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("fixture custody accepts regular files only")
            digest = hashlib.sha256()
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError("fixture source changed before its exact file read")
                for block in iter(lambda: stream.read(65536), b""):
                    digest.update(block)
                after = os.fstat(stream.fileno())
            current = path.lstat()
            identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            if identity(before) != identity(after) or identity(after) != identity(current):
                raise ValueError("fixture source changed during its exact file read")
            output[relative] = {
                "sha256": digest.hexdigest(), "bytes": after.st_size,
                "device": after.st_dev, "inode": after.st_ino,
                "mtimeNs": after.st_mtime_ns, "ctimeNs": after.st_ctime_ns,
                "links": after.st_nlink,
            }
    return dict(sorted(output.items()))


def content_snapshot(snapshot):
    return {name: {"sha256": item["sha256"], "bytes": item["bytes"]}
            for name, item in snapshot.items()}


def prepare_private_copy(source_root, root, *, minimum_free_disk_bytes=0):
    source_root, root = Path(source_root), Path(root)
    for folder in (source_root, root):
        if (not folder.is_absolute() or folder.is_symlink() or not folder.is_dir()
                or not folder.name.startswith("omni-native-cold-load-")):
            raise ValueError("Load-only custody needs explicit generated fixture directories")
    source_root, root = source_root.resolve(), root.resolve()
    if root == source_root or source_root in root.parents or root in source_root.parents or any(root.iterdir()):
        raise ValueError("Load-only output must be new, empty and outside the preserved source")
    before = tree_snapshot(source_root)
    if shutil.disk_usage(root).free - sum(item["bytes"] for item in before.values()) < minimum_free_disk_bytes:
        raise ValueError("private copy would cross the declared physical free-disk floor")
    source_brain = source_root / "brain"
    if source_brain.is_symlink() or not (source_brain / "engine" / "brain.json").is_file():
        raise ValueError("Load-only source must contain an already-created native checkpoint")
    metadata = json.loads((source_brain / "engine" / "brain.json").read_text())
    config = metadata.get("config", {})
    if (config.get("origin_kind") != "ground-up" or metadata.get("brain_id") != "tiny-load"
            or (config.get("d_model"), config.get("n_layers"), config.get("router_neurons")) != (8, 1, 4)
            or any(metadata.get("counters", {}).get(key) != 0
                   for key in ("training_steps", "inference_count", "experiences"))
            or metadata.get("ground_up_training_manifest") is not None
            or metadata.get("connected_regions") is not None):
        raise ValueError("Load-only source is not the declared tiny untrained native fixture")
    # shutil copies into separate files; never preserve source hardlinks or
    # use mutable source/clone shared inodes as runtime destinations.
    shutil.copytree(source_brain, root / "brain", copy_function=shutil.copy2)
    copied = tree_snapshot(root / "brain")
    source_files = {name.removeprefix("brain/"): item for name, item in before.items() if name.startswith("brain/")}
    if content_snapshot(copied) != content_snapshot(source_files):
        raise ValueError("private copy differs from preserved checkpoint bytes")
    if any(item["links"] != 1 or (item["device"], item["inode"]) ==
           (source_files[name]["device"], source_files[name]["inode"])
           for name, item in copied.items()):
        raise ValueError("Load-only copy must have independent non-hardlinked file ownership")
    if tree_snapshot(source_root) != before:
        raise ValueError("preserved source changed during copy")
    receipt = {"sourceRoot": str(source_root), "sourceTreeBefore": before,
               "privateFilesBefore": copied, "privateContentMatchesSource": True,
               "privateFilesIndependent": True, "sourceUntouchedDuringCopy": True,
               "originDirectoryPresent": (source_brain / "engine" / "origin").is_dir(),
               "scope": "already-created-random-native-Load-only-not-trained-origin-or-connected-frontier"}
    write_receipt(root / "copy-receipt.json", receipt)
    return receipt


def sqlite_snapshot(private_brain):
    """Bounded-row complete logical contents; never call on preserved source."""
    output = {}
    for relative in ("engine/events.sqlite3", "engine/conversation.sqlite3",
                     "engine/state/replay.sqlite3", "engine/state/working-memory.sqlite3"):
        path = Path(private_brain) / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("expected copied SQLite owner is absent or unsafe")
        with closing(sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True)) as connection:
            tables = {}
            schema = list(connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"))
            for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                escaped = str(name).replace('"', '""')
                cursor = connection.execute('SELECT * FROM "' + escaped + '" ORDER BY rowid')
                digest, count = hashlib.sha256(), 0
                while True:
                    rows = cursor.fetchmany(32)
                    if not rows:
                        break
                    for row in rows:
                        values = [{"blobSha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                                  if isinstance(value, bytes) else value for value in row]
                        encoded = canonical(values)
                        digest.update(len(encoded).to_bytes(8, "little")); digest.update(encoded)
                        count += 1
                tables[str(name)] = {"rows": count, "sha256": digest.hexdigest()}
            output[relative] = {"schema": schema, "userVersion": connection.execute("PRAGMA user_version").fetchone()[0],
                                "applicationId": connection.execute("PRAGMA application_id").fetchone()[0], "tables": tables}
    return output


def strict_private_files(snapshot):
    # SQLite may checkpoint a WAL or change physical headers on open/close.
    # Its schema/every logical row are checked separately, not ignored.
    sqlite_names = {"engine/events.sqlite3", "engine/conversation.sqlite3",
                    "engine/state/replay.sqlite3", "engine/state/working-memory.sqlite3"}
    return content_snapshot({name: item for name, item in snapshot.items()
                             if not any(name == database + suffix for database in sqlite_names
                                        for suffix in ("", "-shm", "-wal"))
                             and re.match(r"engine/state/live-substrate-cache/[a-f0-9]{32}/", name) is None})


def runtime_source_snapshot():
    engine = Path(__file__).resolve().parents[1]
    def included(name):
        return (name in {"worker.py", "tests/probe_tiny_load_summary.py", "tests/tiny_load_probe_receipts.py"}
                or name.startswith("omni_core/") and Path(name).suffix in {".py", ".cpp", ".h", ".metal"})
    return content_snapshot(tree_snapshot(engine, include=included))
