"""Reconstructible, transaction-maintained weighted vector payload totals."""
from __future__ import annotations

import sqlite3


_KEYS = ("vector_payload_bytes", "vector_weighted_rows", "vector_repeated_contacts")


def ensure_vector_counts(connection: sqlite3.Connection, dimensions: int, reserve=None):
    base = (dimensions + 3) // 4
    weighted_base = base + dimensions + 12
    expressions = ("LENGTH({side}packed)",
        "CASE WHEN LENGTH({side}packed)>=" + str(weighted_base) + " THEN 1 ELSE 0 END",
        "CASE WHEN LENGTH({side}packed)>=" + str(weighted_base) + " THEN (LENGTH({side}packed)-" + str(weighted_base) + ")/10 ELSE 0 END")
    expected = {}
    for event in ("INSERT", "DELETE", "UPDATE"):
        name = "weighted_vector_count_" + event.lower()
        updates = []
        for key, expression in zip(_KEYS, expressions):
            delta = ("+ (" + expression.format(side="NEW.") + ")" if event in {"INSERT", "UPDATE"} else "")
            delta += ("- (" + expression.format(side="OLD.") + ")" if event in {"DELETE", "UPDATE"} else "")
            updates.append("UPDATE paged_vector_meta SET value=CAST(value AS INTEGER)" + delta + " WHERE key='" + key + "';")
        expected[name] = "CREATE TRIGGER " + name + " AFTER " + event + (" OF packed" if event == "UPDATE" else "") + " ON paged_vector_rows BEGIN " + " ".join(updates) + " END"
    existing = dict(connection.execute("SELECT key,value FROM paged_vector_meta WHERE key IN (?,?,?)", _KEYS))
    if not existing:
        if reserve is not None:
            reserve(65536, "weighted vector count migration")
        totals = connection.execute("SELECT " + ",".join("COALESCE(SUM(" + expression.format(side="") + "),0)" for expression in expressions) + " FROM paged_vector_rows").fetchone()
        connection.executemany("INSERT INTO paged_vector_meta(key,value) VALUES (?,?)", zip(_KEYS, map(str, totals)))
        for statement in expected.values():
            connection.execute(statement)
    elif set(existing) != set(_KEYS):
        raise ValueError("weighted vector counters are incomplete")
    for name, statement in expected.items():
        row = connection.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
        if row is None or row[0] != statement:
            raise ValueError("weighted vector count trigger is missing or altered")
    vector_counts(connection)


def vector_counts(connection: sqlite3.Connection) -> tuple[int, int, int]:
    values = dict(connection.execute("SELECT key,value FROM paged_vector_meta WHERE key IN (?,?,?)", _KEYS))
    if set(values) != set(_KEYS) or any(not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value for value in values.values()):
        raise ValueError("weighted vector counter is invalid")
    return tuple(int(values[key]) for key in _KEYS)
