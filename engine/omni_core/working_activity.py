"""Cold working-page recency in an owner's paused activity coordinates.

Wall audit timestamps are never shifted. Older rows are interpreted from one
persisted, last-committed checkpoint baseline, not from file/copy time. Unknown
historical age contributes no invented recency advantage.
"""
from __future__ import annotations

from datetime import datetime
import math
from typing import Mapping


ACTIVITY_FIELD = "activityUpdatedAt"
BASELINE_FORMAT = "omni-cold-working-activity"


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid cold working " + label)
    return float(value)


def checkpoint_wall_time(value):
    """Only an authenticated saved provenance date can seed old-page age."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        stamp = parsed.timestamp()
        return stamp if math.isfinite(stamp) and stamp >= 0 else None
    except (ValueError, OverflowError, OSError):
        return None


def checked_activity_baseline(activity_at, wall_at):
    activity_at = _finite(activity_at, "baseline activity")
    if activity_at < 0:
        raise ValueError("cold working baseline activity cannot be negative")
    if wall_at is not None:
        wall_at = _finite(wall_at, "checkpoint wall timestamp")
        if wall_at < 0:
            raise ValueError("cold working checkpoint wall timestamp cannot be negative")
    return {"activityAt": activity_at, "wallAt": wall_at}


def checkpoint_activity_time(metadata):
    from .activity_clock import ActivityClock
    state = metadata.get("activity_clock")
    parsed = ActivityClock.from_state(state, legacy_markers=tuple(metadata.get(name, 0.0)
        for name in ("last_activity_decay", "last_idle_cycle_at", "last_idle_visible_action_at")))
    # A saved running flag does not authorize charging hydration/copy time.
    return parsed.now() if state is None else float(state["elapsed"])


def read_activity_baseline(connection):
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='working_activity_clock'").fetchone() is None:
        return None
    if connection.execute("SELECT COUNT(*) FROM working_activity_clock").fetchone() != (1,):
        raise ValueError("cold working activity baseline is missing or duplicated")
    row = connection.execute("SELECT singleton,format,version,activity_at,wall_at FROM working_activity_clock").fetchone()
    if row is None or row[:3] != (1, BASELINE_FORMAT, 1):
        raise ValueError("cold working activity baseline format is invalid")
    return checked_activity_baseline(row[3], row[4])


def page_activity_at(metadata: Mapping, updated_at, baseline):
    if "activityRecencyUnknown" in metadata:
        if metadata["activityRecencyUnknown"] is not True or ACTIVITY_FIELD in metadata:
            raise ValueError("cold working unknown recency declaration is invalid")
        return None
    if ACTIVITY_FIELD in metadata:
        # Translation into a newly born owner's coordinate can be negative.
        return _finite(metadata[ACTIVITY_FIELD], "page activity timestamp")
    if baseline is None or baseline["wallAt"] is None:
        return None
    wall = _finite(updated_at, "page audit timestamp")
    return baseline["activityAt"] - max(0.0, baseline["wallAt"] - wall)


def rebase_page_activity(metadata: Mapping, *, source_now, source_at, target_now):
    """Carry known age, including zero age after a future/rollback stamp."""
    source_now = _finite(source_now, "source activity coordinate")
    target_now = _finite(target_now, "target activity coordinate")
    output = dict(metadata)
    if source_at is None:
        output.pop(ACTIVITY_FIELD, None)
        output["activityRecencyUnknown"] = True
    else:
        output[ACTIVITY_FIELD] = target_now - max(0.0, source_now - _finite(source_at, "source page timestamp"))
        output.pop("activityRecencyUnknown", None)
    return output
