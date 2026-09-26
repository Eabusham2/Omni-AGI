"""Append-only, hash-chained paging for neural messages, actions, and traces."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ZERO_HASH = "0" * 64


class NeuralConversationLedger:
    FORMAT = "omni-neural-conversation-ledger"
    VERSION = 1

    def __init__(self, path: Path, brain_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.brain_id = str(brain_id)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entries(
              sequence INTEGER PRIMARY KEY,
              entry_key TEXT NOT NULL UNIQUE,
              kind TEXT NOT NULL CHECK(kind IN ('message','action','trace')),
              created_at TEXT NOT NULL,
              attention_epoch INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              payload_sha256 TEXT NOT NULL,
              previous_sha256 TEXT NOT NULL,
              row_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS entries_kind_sequence
              ON entries(kind, sequence DESC);
            CREATE INDEX IF NOT EXISTS entries_epoch_sequence
              ON entries(attention_epoch, sequence DESC);
            """
        )
        stored = self.connection.execute(
            "SELECT value FROM meta WHERE key='brain_id'"
        ).fetchone()
        if stored is not None and str(stored[0]) != self.brain_id:
            raise ValueError("neural conversation ledger belongs to another brain")
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('brain_id',?)",
            (self.brain_id,),
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('format',?)",
            (self.FORMAT,),
        )
        self.connection.commit()

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def _payload_sha(cls, value: Mapping[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(value).encode("utf-8")).hexdigest()

    @classmethod
    def _row_sha(cls, row: Mapping[str, Any]) -> str:
        body = {
            "sequence": int(row["sequence"]),
            "entryKey": str(row["entry_key"]),
            "kind": str(row["kind"]),
            "createdAt": str(row["created_at"]),
            "attentionEpoch": int(row["attention_epoch"]),
            "payloadSha256": str(row["payload_sha256"]),
            "previousSha256": str(row["previous_sha256"]),
        }
        return hashlib.sha256(cls._canonical(body).encode("utf-8")).hexdigest()

    @staticmethod
    def _key(kind: str, value: Mapping[str, Any]) -> str:
        identifier = str(value.get("id", "")).strip()
        if not identifier:
            identifier = hashlib.sha256(
                NeuralConversationLedger._canonical(value).encode("utf-8")
            ).hexdigest()
        if kind == "action":
            return "action:%s:%s:%s" % (
                identifier,
                str(value.get("state", "unknown")),
                str(value.get("updatedAt", value.get("updated_at", ""))),
            )
        return "%s:%s" % (kind, identifier)

    def append(self, kind: str, values: Iterable[Mapping[str, Any]]) -> None:
        if kind not in {"message", "action", "trace"}:
            raise ValueError("unsupported neural conversation ledger kind")
        last = self.connection.execute(
            "SELECT sequence,row_sha256 FROM entries ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = int(last[0]) if last else 0
        previous = str(last[1]) if last else ZERO_HASH
        with self.connection:
            for raw in values:
                value = dict(raw)
                key = self._key(kind, value)
                payload_json = self._canonical(value)
                payload_sha256 = hashlib.sha256(
                    payload_json.encode("utf-8")
                ).hexdigest()
                existing = self.connection.execute(
                    "SELECT payload_sha256 FROM entries WHERE entry_key=?",
                    (key,),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != payload_sha256:
                        raise ValueError("neural conversation idempotency conflict")
                    continue
                sequence += 1
                epoch = value.get("attention_epoch", value.get("attentionEpoch", 0))
                if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                    raise ValueError("neural conversation attention epoch is invalid")
                created_at = str(value.get("created_at", value.get("createdAt", "")))
                row = {
                    "sequence": sequence,
                    "entry_key": key,
                    "kind": kind,
                    "created_at": created_at,
                    "attention_epoch": epoch,
                    "payload_sha256": payload_sha256,
                    "previous_sha256": previous,
                }
                digest = self._row_sha(row)
                self.connection.execute(
                    """INSERT INTO entries(
                      sequence,entry_key,kind,created_at,attention_epoch,
                      payload_json,payload_sha256,previous_sha256,row_sha256
                    ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        sequence,
                        key,
                        kind,
                        created_at,
                        epoch,
                        payload_json,
                        payload_sha256,
                        previous,
                        digest,
                    ),
                )
                previous = digest

    def backfill(
        self,
        messages: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
        actions: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        combined = [
            ("message", index, value)
            for index, value in enumerate(messages)
        ] + [
            ("trace", len(messages) + index, value)
            for index, value in enumerate(traces)
        ] + [
            ("action", len(messages) + len(traces) + index, value)
            for index, value in enumerate(actions)
        ]
        combined.sort(
            key=lambda item: (
                str(item[2].get("created_at", item[2].get("createdAt", ""))),
                item[1],
            )
        )
        for kind, _index, value in combined:
            self.append(kind, [value])

    def counts(self, attention_epoch: Optional[int] = None) -> Dict[str, int]:
        """Return authoritative ledger counts, optionally for one attention epoch."""

        clause = "" if attention_epoch is None else " WHERE attention_epoch=?"
        arguments: tuple[Any, ...] = (
            () if attention_epoch is None else (int(attention_epoch),)
        )
        values = self.connection.execute(
            """SELECT COUNT(*),
              SUM(CASE WHEN kind='message' THEN 1 ELSE 0 END),
              SUM(CASE WHEN kind='action' THEN 1 ELSE 0 END),
              SUM(CASE WHEN kind='trace' THEN 1 ELSE 0 END)
              FROM entries%s""" % clause,
            arguments,
        ).fetchone()
        return {
            "totalEntries": int(values[0] or 0),
            "messageCount": int(values[1] or 0),
            "actionCount": int(values[2] or 0),
            "traceCount": int(values[3] or 0),
        }

    def summary(self) -> Dict[str, Any]:
        counts = self.counts()
        maximum_epoch = self.connection.execute(
            "SELECT MAX(attention_epoch) FROM entries"
        ).fetchone()
        head = self.connection.execute(
            "SELECT sequence,row_sha256 FROM entries ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return {
            "format": self.FORMAT,
            "formatVersion": self.VERSION,
            **counts,
            "attentionEpoch": int(maximum_epoch[0] or 0),
            "headSequence": int(head[0]) if head else 0,
            "headSha256": str(head[1]) if head else ZERO_HASH,
        }

    def page(
        self,
        *,
        before_sequence: Optional[int] = None,
        limit: int = 100,
        kinds: Sequence[str] = ("message", "action", "trace"),
        attention_epoch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        selected = tuple(kind for kind in kinds if kind in {"message", "action", "trace"})
        if not selected:
            return []
        bounded = max(1, min(1000, int(limit)))
        before = int(before_sequence) if before_sequence is not None else (1 << 63) - 1
        placeholders = ",".join("?" for _ in selected)
        arguments: List[Any] = [before, *selected]
        epoch_clause = ""
        if attention_epoch is not None:
            epoch_clause = " AND attention_epoch=?"
            arguments.append(int(attention_epoch))
        arguments.append(bounded)
        rows = self.connection.execute(
            "SELECT sequence,kind,payload_json,payload_sha256,row_sha256 "
            "FROM entries WHERE sequence<? AND kind IN (%s)%s "
            "ORDER BY sequence DESC LIMIT ?" % (placeholders, epoch_clause),
            arguments,
        ).fetchall()
        output = []
        for sequence, kind, payload_json, payload_sha256, row_sha256 in reversed(rows):
            if hashlib.sha256(str(payload_json).encode("utf-8")).hexdigest() != payload_sha256:
                raise ValueError("neural conversation payload checksum failed")
            output.append(
                {
                    "sequence": int(sequence),
                    "kind": str(kind),
                    "payload": json.loads(str(payload_json)),
                    "payloadSha256": str(payload_sha256),
                    "rowSha256": str(row_sha256),
                }
            )
        return output

    def recent_payloads(
        self, kind: str, limit: int, attention_epoch: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return [
            dict(entry["payload"])
            for entry in self.page(
                limit=limit,
                kinds=(kind,),
                attention_epoch=attention_epoch,
            )
        ]

    def payload_by_id(self, kind: str, identifier: str) -> Optional[Dict[str, Any]]:
        key = "%s:%s" % (kind, str(identifier))
        row = self.connection.execute(
            "SELECT payload_json,payload_sha256 FROM entries WHERE entry_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload_json, payload_sha256 = str(row[0]), str(row[1])
        if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != payload_sha256:
            raise ValueError("neural conversation payload checksum failed")
        value = json.loads(payload_json)
        return dict(value) if isinstance(value, dict) else None

    def truncate_after_head(self, sequence: int, head_sha256: str) -> int:
        """Drop only rows appended after a verified pre-turn ledger head.

        The caller must first establish that the failed chat turn has no
        committed brain.json receipt. Its captured head preserves all earlier
        messages, traces, and host action rows even when brain.json's older
        summary legitimately lags standalone action events.
        """

        sequence = int(sequence)
        if sequence < 0:
            raise ValueError("conversation rollback head is invalid")
        expected_sha = str(head_sha256)
        if sequence == 0:
            if expected_sha != ZERO_HASH:
                raise ValueError("conversation rollback head hash is invalid")
        else:
            saved = self.connection.execute(
                "SELECT row_sha256 FROM entries WHERE sequence=?", (sequence,)
            ).fetchone()
            if saved is None or str(saved[0]) != expected_sha:
                raise ValueError("conversation rollback head does not match")
        current = self.connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM entries"
        ).fetchone()
        current_sequence = int(current[0] or 0)
        if current_sequence < sequence:
            raise ValueError("conversation rollback head is ahead of the ledger")
        with self.connection:
            self.connection.execute(
                "DELETE FROM entries WHERE sequence>?", (sequence,)
            )
        return current_sequence - sequence

    def integrity(self) -> Dict[str, Any]:
        previous = ZERO_HASH
        expected = 1
        for row in self.connection.execute(
            """SELECT sequence,entry_key,kind,created_at,attention_epoch,
              payload_json,payload_sha256,previous_sha256,row_sha256
              FROM entries ORDER BY sequence"""
        ):
            record = {
                "sequence": int(row[0]),
                "entry_key": str(row[1]),
                "kind": str(row[2]),
                "created_at": str(row[3]),
                "attention_epoch": int(row[4]),
                "payload_sha256": str(row[6]),
                "previous_sha256": str(row[7]),
            }
            if (
                record["sequence"] != expected
                or record["previous_sha256"] != previous
                or hashlib.sha256(str(row[5]).encode("utf-8")).hexdigest()
                != record["payload_sha256"]
                or self._row_sha(record) != str(row[8])
            ):
                raise ValueError("neural conversation ledger integrity failed")
            previous = str(row[8])
            expected += 1
        return self.summary()

    def close(self) -> None:
        self.connection.close()
