"""Transaction-maintained counts for paged caches, not neural-state limits.

COUNT(*) over a large SQLite B-tree is still a corpus scan. These counters
cost one indexed metadata update per row INSERT/DELETE and roll back with the
source mutation. An old cache is counted once while installing exact checked
triggers; subsequent status/length queries read only a metadata row.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, Optional


_STORES = {
    "assembly_records": "index_metadata",
    "paged_vector_rows": "paged_vector_meta",
    "paged_neuron_records": "paged_neuron_meta",
}


def ensure_row_count(
    connection: sqlite3.Connection, table: str,
    reserve_disk: Optional[Callable[[int, str], None]] = None,
) -> None:
    meta = _STORES[table]
    row = connection.execute(
        "SELECT value FROM %s WHERE key='row_count'" % meta
    ).fetchone()
    expected = {
        "paged_count_%s_%s" % (table, event.lower()):
        "CREATE TRIGGER paged_count_%s_%s AFTER %s ON %s BEGIN "
        "UPDATE %s SET value=CAST(value AS INTEGER)%s1 WHERE key='row_count'; END"
        % (table, event.lower(), event, table, meta, operator)
        for event, operator in (("INSERT", "+"), ("DELETE", "-"))
    }
    if row is None:
        if reserve_disk is not None:
            reserve_disk(65536, "paged row-count migration")
        count = connection.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
        connection.execute(
            "INSERT INTO %s(key,value) VALUES('row_count',?)" % meta, (str(count),)
        )
        for name, statement in expected.items():
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone() is not None:
                raise ValueError("paged row-count trigger has no counter")
            connection.execute(statement)
    else:
        for name, statement in expected.items():
            observed = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone()
            if observed is None or observed[0] != statement:
                raise ValueError("paged row-count trigger is missing or altered")
    row_count(connection, table)


def row_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(
        "SELECT value FROM %s WHERE key='row_count'" % _STORES[table]
    ).fetchone()
    if (
        row is None or not isinstance(row[0], str) or not row[0].isdecimal()
        or str(int(row[0])) != row[0]
    ):
        raise ValueError("paged row counter is invalid")
    return int(row[0])


def ensure_assembly_vector_count(
    connection: sqlite3.Connection,
    reserve_disk: Optional[Callable[[int, str], None]] = None,
) -> None:
    """Count the exact indexed assembly/vector join without rescanning it."""

    expected: dict[str, str] = {}
    for table, column, other, other_column in (
        ("assembly_records", "assembly_id", "paged_vector_rows", "vector_id"),
        ("paged_vector_rows", "vector_id", "assembly_records", "assembly_id"),
    ):
        for event, operator, side in (("INSERT", "+", "NEW"), ("DELETE", "-", "OLD")):
            name = "paged_link_count_%s_%s" % (table, event.lower())
            expected[name] = (
                "CREATE TRIGGER %s AFTER %s ON %s WHEN EXISTS "
                "(SELECT 1 FROM %s WHERE %s=%s.%s) BEGIN "
                "UPDATE index_metadata SET value=CAST(value AS INTEGER)%s1 "
                "WHERE key='assembly_vector_count'; END"
                % (name, event, table, other, other_column, side, column, operator)
            )
        name = "paged_link_count_%s_update" % table
        expected[name] = (
            "CREATE TRIGGER %s AFTER UPDATE OF %s ON %s WHEN OLD.%s!=NEW.%s BEGIN "
            "UPDATE index_metadata SET value=CAST(value AS INTEGER)"
            "-(SELECT COUNT(*) FROM %s WHERE %s=OLD.%s)"
            "+(SELECT COUNT(*) FROM %s WHERE %s=NEW.%s) "
            "WHERE key='assembly_vector_count'; END"
            % (name, column, table, column, column, other, other_column, column,
               other, other_column, column)
        )
    existing = connection.execute(
        "SELECT value FROM index_metadata WHERE key='assembly_vector_count'"
    ).fetchone()
    if existing is None:
        if reserve_disk is not None:
            reserve_disk(65536, "paged linked-row-count migration")
        count = connection.execute(
            "SELECT COUNT(*) FROM assembly_records AS a "
            "JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO index_metadata(key,value) VALUES('assembly_vector_count',?)",
            (str(count),),
        )
        for statement in expected.values():
            connection.execute(statement)
    else:
        for name, statement in expected.items():
            observed = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone()
            if observed is None or observed[0] != statement:
                raise ValueError("paged linked-row-count trigger is missing or altered")
    assembly_vector_count(connection)


def assembly_vector_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM index_metadata WHERE key='assembly_vector_count'"
    ).fetchone()
    if (
        row is None or not isinstance(row[0], str) or not row[0].isdecimal()
        or str(int(row[0])) != row[0]
    ):
        raise ValueError("paged linked-row counter is invalid")
    return int(row[0])
