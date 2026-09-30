"""Incremental structural membership checksum for a derived forward index.

An insertion-order-independent compressed Patricia trie has domain-separated
SHA256 leaves and branch hashes. Each changed ID visits at most 256 path nodes
and adds at most a leaf and a branch, never a corpus-sized Python map. SQLite
triggers stage membership deltas with the source mutation. This remains a
derived forward-index binding, not learned state or recovery authority.
"""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from pathlib import Path
from collections.abc import Iterable
from typing import Callable, Optional

from .paged_store_counts import row_count


ALGORITHM = "sha256-patricia-id-set-v1"
_EMPTY = hashlib.sha256(b"omni-forward-membership-empty-v1").hexdigest()


def _key(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("assembly membership ID is invalid")
    return "".join(format(value, "08b") for value in hashlib.sha256(
        b"omni-forward-membership-key-v1\0" + identifier.encode("utf-8")
    ).digest())


def _leaf_sha(identifier: str) -> str:
    return hashlib.sha256(
        b"omni-forward-membership-leaf-v1\0" + identifier.encode("utf-8")
    ).hexdigest()


def _branch_sha(prefix: str, left_sha: str, right_sha: str, count: int) -> str:
    return hashlib.sha256(
        b"omni-forward-membership-branch-v1\0" + prefix.encode("ascii")
        + b"\0" + str(count).encode("ascii") + b"\0"
        + bytes.fromhex(left_sha) + bytes.fromhex(right_sha)
    ).hexdigest()


def _node(connection: sqlite3.Connection, prefix: str) -> tuple:
    row = connection.execute(
        "SELECT prefix,assembly_id,left_prefix,right_prefix,count,sha256 "
        "FROM assembly_membership_nodes WHERE prefix=?", (prefix,),
    ).fetchone()
    if row is None or (
        not isinstance(row[0], str) or len(row[0]) > 256
        or any(bit not in "01" for bit in row[0])
        or type(row[4]) is not int or row[4] < 1
        or not isinstance(row[5], str) or len(row[5]) != 64
        or any(char not in "0123456789abcdef" for char in row[5])
    ):
        raise ValueError("assembly membership Merkle node is invalid")
    if row[1] is not None:
        if (
            _key(row[1]) != prefix or row[2] is not None or row[3] is not None
            or row[4] != 1 or row[5] != _leaf_sha(row[1])
        ):
            raise ValueError("assembly membership Merkle leaf is corrupt")
    elif (
        not isinstance(row[2], str) or not isinstance(row[3], str)
        or not row[2].startswith(prefix + "0") or not row[3].startswith(prefix + "1")
        or len(row[2]) > 256 or len(row[3]) > 256
    ):
        raise ValueError("assembly membership Merkle branch is corrupt")
    return row


def _put_branch(connection: sqlite3.Connection, prefix: str, left: str, right: str) -> None:
    lhs, rhs = _node(connection, left), _node(connection, right)
    if not left.startswith(prefix + "0") or not right.startswith(prefix + "1"):
        raise ValueError("assembly membership Merkle branch placement is invalid")
    count = lhs[4] + rhs[4]
    connection.execute(
        "INSERT OR REPLACE INTO assembly_membership_nodes "
        "(prefix,assembly_id,left_prefix,right_prefix,count,sha256) VALUES(?,NULL,?,?,?,?)",
        (prefix, left, right, count, _branch_sha(prefix, lhs[5], rhs[5], count)),
    )


def _checked_branch(connection: sqlite3.Connection, node: tuple) -> None:
    lhs, rhs = _node(connection, node[2]), _node(connection, node[3])
    if (
        node[4] != lhs[4] + rhs[4]
        or node[5] != _branch_sha(node[0], lhs[5], rhs[5], node[4])
    ):
        raise ValueError("assembly membership Merkle branch checksum mismatch")


def _attach(connection: sqlite3.Connection, path: list[tuple], replacement: str) -> str:
    # Strictly increasing prefixes bound this path to 256, independent of N.
    for parent, side in reversed(path):
        _put_branch(connection, parent[0],
                    replacement if side == 0 else parent[2],
                    replacement if side == 1 else parent[3])
        replacement = parent[0]
    return replacement


def _change(connection: sqlite3.Connection, root: str, identifier: str, delta: int) -> str:
    key = _key(identifier)
    path: list[tuple] = []
    position = root
    while position != "empty":
        node = _node(connection, position)
        if node[1] is None:
            _checked_branch(connection, node)
        common = 0
        while common < len(position) and key[common] == position[common]:
            common += 1
        if common != len(position):
            break
        if node[1] is not None:
            break
        side = int(key[len(position)])
        path.append((node, side))
        position = node[2 + side]
    if delta == 1:
        if position != "empty" and position == key:
            raise ValueError("assembly membership Merkle duplicate or hash collision")
        connection.execute(
            "INSERT INTO assembly_membership_nodes "
            "(prefix,assembly_id,left_prefix,right_prefix,count,sha256) VALUES(?,?,NULL,NULL,1,?)",
            (key, identifier, _leaf_sha(identifier)),
        )
        if position == "empty":
            return key
        split = position[:common]
        left, right = (key, position) if key[common] == "0" else (position, key)
        _put_branch(connection, split, left, right)
        return _attach(connection, path, split)
    if (
        delta != -1 or position == "empty" or position != key
        or node[1] != identifier
    ):
        raise ValueError("assembly membership Merkle deletion has no exact leaf")
    connection.execute("DELETE FROM assembly_membership_nodes WHERE prefix=?", (key,))
    if not path:
        return "empty"
    parent, side = path.pop()
    replacement = parent[3 if side == 0 else 2]
    connection.execute("DELETE FROM assembly_membership_nodes WHERE prefix=?", (parent[0],))
    return _attach(connection, path, replacement)


def _triggers() -> dict[str, str]:
    return {
        "paged_membership_insert":
            "CREATE TRIGGER paged_membership_insert AFTER INSERT ON assembly_records BEGIN "
            "INSERT INTO assembly_membership_deltas(assembly_id,delta) VALUES(NEW.assembly_id,1); END",
        "paged_membership_delete":
            "CREATE TRIGGER paged_membership_delete AFTER DELETE ON assembly_records BEGIN "
            "INSERT INTO assembly_membership_deltas(assembly_id,delta) VALUES(OLD.assembly_id,-1); END",
        "paged_membership_update":
            "CREATE TRIGGER paged_membership_update AFTER UPDATE OF assembly_id ON assembly_records "
            "WHEN OLD.assembly_id!=NEW.assembly_id BEGIN "
            "INSERT INTO assembly_membership_deltas(assembly_id,delta) VALUES(OLD.assembly_id,-1); "
            "INSERT INTO assembly_membership_deltas(assembly_id,delta) VALUES(NEW.assembly_id,1); END",
    }


def ensure_membership_checksum(
    connection: sqlite3.Connection, reserve_disk: Optional[Callable[[int, str], None]] = None,
) -> None:
    existing = connection.execute(
        "SELECT value FROM index_metadata WHERE key='assembly_membership_root'"
    ).fetchone()
    expected = _triggers()
    if existing is None:
        if reserve_disk is not None:
            reserve_disk(65536, "assembly membership checksum migration")
        connection.execute(
            "CREATE TABLE assembly_membership_nodes (prefix TEXT PRIMARY KEY,assembly_id TEXT,"
            "left_prefix TEXT,right_prefix TEXT,count INTEGER NOT NULL,sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE assembly_membership_deltas (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
            "assembly_id TEXT NOT NULL,delta INTEGER NOT NULL CHECK(delta IN (-1,1)))"
        )
        root = "empty"
        for (identifier,) in connection.execute("SELECT assembly_id FROM assembly_records"):
            if reserve_disk is not None:
                reserve_disk(8192, "assembly membership Merkle migration node")
            root = _change(connection, root, identifier, 1)
        connection.execute(
            "INSERT INTO index_metadata(key,value) VALUES('assembly_membership_root',?)", (root,),
        )
        for statement in expected.values():
            connection.execute(statement)
    else:
        for name, statement in expected.items():
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,),
            ).fetchone()
            if row is None or row[0] != statement:
                raise ValueError("assembly membership checksum trigger is missing or altered")


def membership_checksum(
    connection: sqlite3.Connection, reserve_disk: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Consume bounded-memory membership deltas in the caller's transaction."""

    row = connection.execute(
        "SELECT value FROM index_metadata WHERE key='assembly_membership_root'"
    ).fetchone()
    if (
        row is None or not isinstance(row[0], str)
        or (row[0] != "empty" and (len(row[0]) > 256 or any(bit not in "01" for bit in row[0])))
    ):
        raise ValueError("assembly membership checksum is malformed")
    root = row[0]
    if root != "empty":
        base = _node(connection, root)
        if base[1] is None:
            _checked_branch(connection, base)
    changed = False
    for identifier, delta in connection.execute(
        "SELECT assembly_id,delta FROM assembly_membership_deltas ORDER BY sequence"
    ):
        if delta not in (-1, 1):
            raise ValueError("assembly membership checksum delta is malformed")
        if reserve_disk is not None:
            reserve_disk(8192 + 512 * len(identifier), "assembly membership Merkle delta")
        root = _change(connection, root, identifier, delta)
        changed = True
    if changed:
        if reserve_disk is not None:
            reserve_disk(65536, "assembly membership checksum delta commit")
        connection.execute(
            "UPDATE index_metadata SET value=? WHERE key='assembly_membership_root'", (root,),
        )
        connection.execute("DELETE FROM assembly_membership_deltas")
    count = row_count(connection, "assembly_records")
    if root == "empty":
        if count != 0:
            raise ValueError("assembly membership Merkle root count differs")
        return _EMPTY
    node = _node(connection, root)
    if node[1] is None:
        _checked_branch(connection, node)
    if count != node[4]:
        raise ValueError("assembly membership Merkle root count differs")
    return node[5]


def checksum_for_ids(values: Iterable[str]) -> str:
    """One bounded-RAM legacy/resident verification without a resident trie.

    Paged live checkpoints use the transaction-held cache method instead.
    This scratch database is expendable and is never recovery authority.
    """

    with tempfile.TemporaryDirectory(prefix="omni-membership-proof-") as folder:
        connection = sqlite3.connect(Path(folder) / "proof.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-1024")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE assembly_membership_nodes (prefix TEXT PRIMARY KEY,assembly_id TEXT,"
                "left_prefix TEXT,right_prefix TEXT,count INTEGER NOT NULL,sha256 TEXT NOT NULL) WITHOUT ROWID"
            )
            root = "empty"
            for identifier in values:
                root = _change(connection, root, identifier, 1)
            return _EMPTY if root == "empty" else _node(connection, root)[5]
        finally:
            connection.close()
