"""Resumable neural replay windows; cursors contain positions, never answers.

Each target byte is visited once per pass with one-token causal overlap. A
window may become smaller after resource pressure without changing coverage.
Only the authenticated observed experience supplies its content.
"""

from typing import Any, Dict, Mapping, Tuple, List


def initial_replay_cursor() -> Dict[str, Any]:
    return {"phase": "bos", "character": 0, "byte": 0}


def validate_replay_cursor(value: Any) -> Dict[str, Any]:
    if value is None:
        return initial_replay_cursor()
    if (not isinstance(value, Mapping) or set(value) != {"phase", "character", "byte"}
        or value.get("phase") not in {"bos", "text", "eos", "done"}
        or type(value.get("character")) is not int or value["character"] < 0
        or type(value.get("byte")) is not int or not 0 <= value["byte"] <= 3):
        raise ValueError("online replay cursor is invalid")
    return dict(value)


def replay_window(text: str, cursor: Any, maximum: int, tokenizer: Any) -> Tuple[List[int], Dict[str, Any], bool]:
    """Produce one bounded window without encoding the entire source again."""
    if type(maximum) is not int or maximum < 2:
        raise ValueError("online replay window requires at least two tokens")
    position = validate_replay_cursor(cursor)
    if position["character"] > len(text):
        raise ValueError("online replay cursor exceeds its bound experience")
    values, overlap = [], dict(position)
    while len(values) < maximum and position["phase"] != "done":
        overlap = dict(position)
        phase = position["phase"]
        if phase == "bos":
            if position["character"] or position["byte"]:
                raise ValueError("BOS replay cursor must begin the experience")
            values.append(int(tokenizer.bos_id))
            position["phase"] = "text" if text else "eos"
        elif phase == "text":
            index, offset = position["character"], position["byte"]
            if index >= len(text):
                raise ValueError("text replay cursor exceeds the experience")
            payload = text[index].encode("utf-8")
            if offset >= len(payload):
                raise ValueError("replay byte offset is invalid for its character")
            values.append(payload[offset] + int(tokenizer.byte_offset))
            offset += 1
            if offset == len(payload):
                position["character"], position["byte"] = index + 1, 0
                if index + 1 == len(text):
                    position["phase"] = "eos"
            else:
                position["byte"] = offset
        elif phase == "eos":
            if position["character"] != len(text) or position["byte"]:
                raise ValueError("EOS replay cursor must end the experience")
            values.append(int(tokenizer.eos_id))
            position["phase"] = "done"
    complete = position["phase"] == "done"
    return values, position if complete else overlap, complete


def rehearsal_due(cadence: Mapping[str, Any], *, committed_records: int,
                  previous_records: int, committed_windows: int,
                  active_window: bool, last_middle_wave: int,
                  periodic_due: bool) -> bool:
    """Use record midpoint where known and real in-record progress otherwise."""
    mode = cadence.get("mode")
    if mode == "indefinite-periodic":
        return bool(periodic_due)
    if mode != "finite-midpoint" or last_middle_wave:
        return False
    middle = cadence.get("middleWave")
    if middle is None:
        # A giant record has multiple committed windows although its row count
        # is one. Rehearse after real progress, before the record's final tail.
        return active_window and committed_windows > 0
    stride = max(1, int(cadence["checkpointRecords"]))
    return committed_records > previous_records and committed_records // stride >= int(middle)
