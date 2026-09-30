"""Bounded-path authenticated Patricia map for expendable SQLite indexes.

Keys and value digests are structural cache metadata, never learned vectors
or answers. Roots are insertion-order independent. A caller must anchor the
root digest to independently verified data/authentication; this module does
not promote a database or a root supplied by that database to authority.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Iterator, Optional


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class MerkleRoot:
    prefix: str
    sha256: str
    count: int


class SqliteMerkleMap:
    """Each operation buffers at most a 256-node path, not all indexed keys."""

    def __init__(self, namespace: str) -> None:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 128:
            raise ValueError("Merkle cache namespace is invalid")
        self.namespace = namespace
        self.domain = b"omni-derived-merkle-kv-v1\0" + namespace.encode("utf-8") + b"\0"

    @staticmethod
    def ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS authenticated_merkle_nodes ("
            "namespace TEXT NOT NULL,prefix TEXT NOT NULL,item_key TEXT,value_sha256 TEXT,"
            "left_prefix TEXT,right_prefix TEXT,count INTEGER NOT NULL,node_sha256 TEXT NOT NULL,"
            "PRIMARY KEY(namespace,prefix)) WITHOUT ROWID"
        )

    def empty(self) -> MerkleRoot:
        return MerkleRoot("empty", _sha(self.domain + b"empty"), 0)

    def clear(self, connection: sqlite3.Connection) -> MerkleRoot:
        connection.execute("DELETE FROM authenticated_merkle_nodes WHERE namespace=?",
                           (self.namespace,))
        return self.empty()

    def _key(self, item: str) -> str:
        if not isinstance(item, str) or not item or len(item.encode("utf-8")) > 4096:
            raise ValueError("Merkle cache key is invalid")
        return "".join(format(value, "08b") for value in hashlib.sha256(
            self.domain + b"key\0" + item.encode("utf-8")
        ).digest())

    def _leaf_sha(self, item: str, value: str) -> str:
        if not _hex(value):
            raise ValueError("Merkle cache value digest is invalid")
        return _sha(self.domain + b"leaf\0" + item.encode("utf-8") + b"\0" + bytes.fromhex(value))

    def _branch_sha(self, prefix: str, count: int, left: str, right: str) -> str:
        return _sha(self.domain + b"branch\0" + prefix.encode("ascii") + b"\0"
                    + str(count).encode("ascii") + b"\0" + bytes.fromhex(left) + bytes.fromhex(right))

    def _node(self, connection: sqlite3.Connection, prefix: str) -> tuple:
        row = connection.execute(
            "SELECT prefix,item_key,value_sha256,left_prefix,right_prefix,count,node_sha256 "
            "FROM authenticated_merkle_nodes WHERE namespace=? AND prefix=?",
            (self.namespace, prefix),
        ).fetchone()
        if row is None or (
            not isinstance(row[0], str) or len(row[0]) > 256
            or any(bit not in "01" for bit in row[0])
            or type(row[5]) is not int or row[5] < 1 or not _hex(row[6])
        ):
            raise ValueError("authenticated Merkle cache node is missing or invalid")
        if row[1] is not None:
            if (
                self._key(row[1]) != prefix or not _hex(row[2])
                or row[3] is not None or row[4] is not None or row[5] != 1
                or row[6] != self._leaf_sha(row[1], row[2])
            ):
                raise ValueError("authenticated Merkle cache leaf is corrupt")
        elif (
            row[2] is not None or not isinstance(row[3], str) or not isinstance(row[4], str)
            or not row[3].startswith(prefix + "0") or not row[4].startswith(prefix + "1")
            or len(row[3]) > 256 or len(row[4]) > 256
        ):
            raise ValueError("authenticated Merkle cache branch is corrupt")
        return row

    def _check_branch(self, connection: sqlite3.Connection, node: tuple) -> None:
        lhs, rhs = self._node(connection, node[3]), self._node(connection, node[4])
        if (
            node[5] != lhs[5] + rhs[5]
            or node[6] != self._branch_sha(node[0], node[5], lhs[6], rhs[6])
        ):
            raise ValueError("authenticated Merkle cache branch checksum mismatch")

    def _root(self, connection: sqlite3.Connection, prefix: str) -> MerkleRoot:
        if prefix == "empty":
            return self.empty()
        node = self._node(connection, prefix)
        if node[1] is None:
            self._check_branch(connection, node)
        return MerkleRoot(prefix, node[6], node[5])

    def validate_root(self, connection: sqlite3.Connection, root: MerkleRoot) -> None:
        if self._root(connection, root.prefix) != root:
            raise ValueError("authenticated Merkle cache root changed")

    def get(self, connection: sqlite3.Connection, root: MerkleRoot, item: str) -> Optional[str]:
        self.validate_root(connection, root)
        key = self._key(item)
        prefix, expected = root.prefix, root.sha256
        while prefix != "empty":
            node = self._node(connection, prefix)
            if node[6] != expected:
                raise ValueError("authenticated Merkle cache path checksum mismatch")
            if node[1] is None:
                self._check_branch(connection, node)
            if not key.startswith(prefix):
                return None  # Authenticated non-membership, not an unchecked SQL miss.
            if node[1] is not None:
                if key == prefix and node[1] != item:
                    raise ValueError("Merkle cache key hash collision")
                return node[2] if node[1] == item else None
            side = int(key[len(prefix)])
            prefix = node[3 + side]
            expected = self._node(connection, prefix)[6]
        return None

    def iter_items(self, connection: sqlite3.Connection, root: MerkleRoot) -> Iterator[tuple[str, str]]:
        """Traverse every authenticated leaf using at most one bounded path.

        An SQL scan cannot prove that a row was not omitted. This traversal
        follows the independently anchored root and checks every branch and
        leaf, so deletion/replacement cannot be mistaken for an empty delta.
        """

        self.validate_root(connection, root)
        stack = [] if root.prefix == "empty" else [(root.prefix, root.sha256)]
        seen = 0
        while stack:
            prefix, expected = stack.pop()
            node = self._node(connection, prefix)
            if node[6] != expected:
                raise ValueError("authenticated Merkle cache traversal checksum mismatch")
            if node[1] is not None:
                seen += 1
                yield node[1], node[2]
            else:
                self._check_branch(connection, node)
                lhs, rhs = self._node(connection, node[3]), self._node(connection, node[4])
                stack.extend(((rhs[0], rhs[6]), (lhs[0], lhs[6])))
            if len(stack) > 256:
                raise ValueError("authenticated Merkle path is invalid")
        if seen != root.count:
            raise ValueError("authenticated Merkle traversal coverage mismatch")

    def _put_branch(self, connection: sqlite3.Connection, prefix: str, left: str, right: str) -> None:
        lhs, rhs = self._node(connection, left), self._node(connection, right)
        if not left.startswith(prefix + "0") or not right.startswith(prefix + "1"):
            raise ValueError("Merkle cache branch placement is invalid")
        count = lhs[5] + rhs[5]
        connection.execute(
            "INSERT OR REPLACE INTO authenticated_merkle_nodes "
            "(namespace,prefix,item_key,value_sha256,left_prefix,right_prefix,count,node_sha256) "
            "VALUES(?,?,NULL,NULL,?,?,?,?)",
            (self.namespace, prefix, left, right, count,
             self._branch_sha(prefix, count, lhs[6], rhs[6])),
        )

    def _attach(self, connection: sqlite3.Connection, path: list[tuple], replacement: str) -> str:
        for parent, side in reversed(path):
            self._put_branch(connection, parent[0], replacement if side == 0 else parent[3],
                             replacement if side == 1 else parent[4])
            replacement = parent[0]
        return replacement

    def put(
        self, connection: sqlite3.Connection, root: MerkleRoot, item: str, value: str,
    ) -> MerkleRoot:
        previous = self.get(connection, root, item)
        if previous == value:
            return root
        if previous is not None:
            root = self.delete(connection, root, item)
        key = self._key(item)
        path: list[tuple] = []
        position = root.prefix
        common = 0
        while position != "empty":
            node = self._node(connection, position)
            if node[1] is None:
                self._check_branch(connection, node)
            common = 0
            while common < len(position) and key[common] == position[common]:
                common += 1
            if common != len(position) or node[1] is not None:
                break
            side = int(key[len(position)])
            path.append((node, side))
            position = node[3 + side]
        if position == key:
            raise ValueError("Merkle cache duplicate or key hash collision")
        connection.execute(
            "INSERT INTO authenticated_merkle_nodes "
            "(namespace,prefix,item_key,value_sha256,left_prefix,right_prefix,count,node_sha256) "
            "VALUES(?,?,?,?,NULL,NULL,1,?)",
            (self.namespace, key, item, value, self._leaf_sha(item, value)),
        )
        if position == "empty":
            return self._root(connection, key)
        split = position[:common]
        left, right = (key, position) if key[common] == "0" else (position, key)
        self._put_branch(connection, split, left, right)
        return self._root(connection, self._attach(connection, path, split))

    def delete(self, connection: sqlite3.Connection, root: MerkleRoot, item: str) -> MerkleRoot:
        if self.get(connection, root, item) is None:
            raise ValueError("Merkle cache deletion has no exact member")
        key = self._key(item)
        path: list[tuple] = []
        position = root.prefix
        while position != key:
            node = self._node(connection, position)
            self._check_branch(connection, node)
            side = int(key[len(position)])
            path.append((node, side))
            position = node[3 + side]
        connection.execute(
            "DELETE FROM authenticated_merkle_nodes WHERE namespace=? AND prefix=?",
            (self.namespace, key),
        )
        if not path:
            return self.empty()
        parent, side = path.pop()
        replacement = parent[4 if side == 0 else 3]
        connection.execute(
            "DELETE FROM authenticated_merkle_nodes WHERE namespace=? AND prefix=?",
            (self.namespace, parent[0]),
        )
        return self._root(connection, self._attach(connection, path, replacement))
