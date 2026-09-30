"""Lazy structural recall IDs and byte-budgeted diagnostic pages.

These views carry only actual assembly IDs from a completed readout. They are
not an answer/learning store. Full neural consumers iterate the complete view;
JSON diagnostics receive a bounded page with exact total/continuation data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from itertools import islice


class RecalledIdSequence(Sequence[str]):
    unique = True  # Completed recall records already have unique assembly IDs.
    def __init__(self, recalled, *, prefix: str = "") -> None:
        self.recalled = recalled
        self.prefix = prefix
        self.state = getattr(recalled, "state", None)
        self.excluded_position = None
        if prefix:
            if self.state is not None:
                row = self.state.connection.execute("SELECT position FROM recalled WHERE assembly_id=?", (prefix,)).fetchone()
                self.excluded_position = int(row[0]) if row else None
            else:
                self.excluded_position = next((index for index, row in enumerate(recalled)
                                              if str(row.get("idea_id", row.get("assembly_id", ""))) == prefix), None)

    def __len__(self):
        return len(self.recalled) + int(bool(self.prefix)) - int(self.excluded_position is not None)

    def __iter__(self) -> Iterator[str]:
        if self.prefix:
            yield self.prefix
        if self.state is not None:
            for (identifier,) in self.state.connection.execute("SELECT assembly_id FROM recalled ORDER BY position"):
                if identifier != self.prefix:
                    yield identifier
        else:
            for row in self.recalled:
                identifier = str(row.get("idea_id", row.get("assembly_id", "")))
                if identifier != self.prefix:
                    yield identifier

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[position] for position in range(start, stop, step)]
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError("recall ID position is out of range")
        if self.prefix:
            if position == 0:
                return self.prefix
            position -= 1
        if self.excluded_position is not None and position >= self.excluded_position:
            position += 1
        if self.state is not None:
            row = self.state.connection.execute("SELECT assembly_id FROM recalled WHERE position=?", (position,)).fetchone()
            if row is None:
                raise ValueError("recall ID view coverage changed")
            return row[0]
        row = self.recalled[position]
        return str(row.get("idea_id", row.get("assembly_id", "")))


def id_page(identifiers: Sequence[str], *, byte_budget: int, offset: int = 0):
    if type(byte_budget) is not int or byte_budget < 128 or type(offset) is not int or offset < 0:
        raise ValueError("recall diagnostic page budget/cursor is invalid")
    page = []
    size = 2
    for identifier in islice(identifiers, offset, None):
        encoded = len(json.dumps(identifier, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        charge = encoded + int(bool(page))
        if size + charge > byte_budget:
            break
        page.append(identifier)
        size += charge
    next_offset = offset + len(page)
    total = len(identifiers)
    return page, {"totalCount": total, "returnedCount": len(page), "offset": offset,
                  "nextOffset": next_offset if next_offset < total else None,
                  "viewTruncated": next_offset < total, "byteBudget": byte_budget,
                  "coverage": "diagnostic-page-only" if next_offset < total or offset else "complete",
                  "neuralReadoutCoverage": (
                      "all-eligible-assemblies" if isinstance(identifiers, RecalledIdSequence)
                      else "not-inferred-from-structural-ids"
                  )}
