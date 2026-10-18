"""Compact sequential region schedules; no tensor/model/hardware work on import.

Schedules are hypotheses. Only a current complete-operation controller's exact
output/state gate and authenticated whole-path samples may select one.
"""
from __future__ import annotations

import copy
import re

from .whole_path_execution import digest

MIXED_MODE = "native-mixed-sequential"
BASE_FIELDS = {"planSha256", "executionDevice", "deviceId", "memoryGroupId", "backendMode",
               "asyncDevices", "supported", "fallbackAcknowledged"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE = re.compile(r"cpu|mps|cuda:[0-9]+\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,255}\Z")


def _base(value):
    if (not isinstance(value, dict) or set(value) != BASE_FIELDS
            or not _SHA.fullmatch(str(value.get("planSha256", "")))
            or not isinstance(value["executionDevice"], str) or not _DEVICE.fullmatch(value["executionDevice"])
            or any(not isinstance(value[name], str) or not _ID.fullmatch(value[name]) for name in ("deviceId", "memoryGroupId"))
            or type(value["supported"]) is not bool or type(value["fallbackAcknowledged"]) is not bool
            or not isinstance(value["asyncDevices"], list)
            or any(not isinstance(item, str) or item == "cpu" or not _DEVICE.fullmatch(item) for item in value["asyncDevices"])
            or value["asyncDevices"] != sorted(set(value["asyncDevices"]))):
        raise ValueError("mixed schedule needs exact reviewed base-device descriptors")
    mode, device = value["backendMode"], value["executionDevice"]
    if (mode == "native-cpu" and (device != "cpu" or value["fallbackAcknowledged"])
            or mode == "native-metal" and (device != "mps" or "mps" not in value["asyncDevices"] or value["fallbackAcknowledged"])
            or mode == "explicit-cuda-cpu-tile" and (not device.startswith("cuda:") or device not in value["asyncDevices"] or not value["fallbackAcknowledged"])
            or mode not in {"native-cpu", "native-metal", "explicit-cuda-cpu-tile"}
            or digest({key: item for key, item in value.items() if key != "planSha256"}) != value["planSha256"]):
        raise ValueError("mixed schedule base cannot invent an accelerator or fallback authority")
    return copy.deepcopy(value)


def checked_mixed_candidate(value, *, bases=None, registry_sha=None, region_count=None):
    if not isinstance(value, dict) or set(value) != BASE_FIELDS | {"regionSchedule"} or value.get("backendMode") != MIXED_MODE:
        raise ValueError("invalid mixed-region candidate descriptor")
    schedule = value["regionSchedule"]
    if (not isinstance(schedule, dict) or set(schedule) != {"version", "registrySha256", "regionCount", "outerCandidate", "segments"}
            or type(schedule["version"]) is not int or schedule["version"] != 1
            or not isinstance(schedule["registrySha256"], str) or not _SHA.fullmatch(schedule["registrySha256"])
            or type(schedule["regionCount"]) is not int or not 1 <= schedule["regionCount"] <= (1 << 53) - 1
            or not isinstance(schedule["segments"], list) or not schedule["segments"]
            or registry_sha is not None and registry_sha != schedule["registrySha256"]
            or region_count is not None and region_count != schedule["regionCount"]):
        raise ValueError("mixed schedule registry/ranges are invalid or stale")
    outer, routes, end = _base(schedule["outerCandidate"]), [], 0
    for row in schedule["segments"]:
        if (not isinstance(row, dict) or set(row) != {"first", "count", "candidate"}
                or type(row["first"]) is not int or type(row["count"]) is not int or row["first"] < end or row["count"] < 1
                or row["first"] + row["count"] > schedule["regionCount"]):
            raise ValueError("mixed schedule regions overlap or exceed the actual registry")
        route = _base(row["candidate"])
        if route["executionDevice"] == outer["executionDevice"]:
            raise ValueError("mixed schedule contains a redundant ordinary-device segment")
        routes.append(route); end = row["first"] + row["count"]
    used = [outer, *routes]
    if bases is not None and any(not any(base == row for base in bases) for row in used):
        raise ValueError("mixed schedule includes a device absent from the exact current review")
    expected = {**outer, "backendMode": MIXED_MODE, "regionSchedule": schedule,
                "asyncDevices": sorted({name for row in used for name in row["asyncDevices"]}),
                "fallbackAcknowledged": any(row["fallbackAcknowledged"] for row in used),
                "supported": all(row["supported"] for row in used)}
    expected.pop("planSha256")
    expected["planSha256"] = digest(expected)
    if value != expected:
        raise ValueError("mixed candidate hash, home, participants or fallback semantics differ")
    return copy.deepcopy(value)


def mixed_candidate(outer, segments, registry_sha, region_count):
    used = [outer, *(row["candidate"] for row in segments)]
    value = {**outer, "backendMode": MIXED_MODE,
             "regionSchedule": {"version": 1, "registrySha256": registry_sha, "regionCount": region_count,
                                "outerCandidate": outer, "segments": segments},
             "asyncDevices": sorted({name for row in used for name in row["asyncDevices"]}),
             "fallbackAcknowledged": any(row["fallbackAcknowledged"] for row in used),
             "supported": all(row["supported"] for row in used)}
    value.pop("planSha256"); value["planSha256"] = digest(value)
    return checked_mixed_candidate(value)


class MixedRegionCandidatePool:
    """Linear span hypotheses, generated one at a time under caller admission.

    No Cartesian assignment product, rate guess or fixed region-count ceiling.
    A single extra partition can use several actually reviewed device targets.
    """
    def __init__(self, current, bases, registry_sha, region_count):
        if (not isinstance(registry_sha, str) or not _SHA.fullmatch(registry_sha)
                or type(region_count) is not int or not 1 <= region_count <= (1 << 53) - 1
                or not isinstance(bases, list) or current not in bases):
            raise ValueError("mixed pool needs the actual current review and positive registry geometry")
        self.current = _base(current)
        self.bases = [self.current, *[_base(row) for row in bases if row != current]]
        if len({row["executionDevice"] for row in bases}) != len(bases):
            raise ValueError("mixed pool review repeats an execution interface")
        if any(row["supported"] is not True for row in self.bases): raise ValueError("unsupported mixed base")
        self.targets = [row for row in self.bases if row["executionDevice"] != self.current["executionDevice"]]
        self.registry_sha, self.region_count = registry_sha, region_count
        self.has_mixed = bool(self.targets and region_count)
    def __len__(self):
        if not self.has_mixed: return len(self.bases)
        return len(self.bases) + len(self.targets) * (2 * self.region_count - 1) + int(len(self.targets) > 1 and self.region_count > 1)
    def __iter__(self):
        yield from self.bases
        if not self.has_mixed: return
        n = self.region_count
        for target in self.targets:
            for count in range(1, n + 1):
                yield mixed_candidate(self.current, [{"first": 0, "count": count, "candidate": target}], self.registry_sha, n)
                if count < n:
                    yield mixed_candidate(self.current, [{"first": n - count, "count": count, "candidate": target}], self.registry_sha, n)
        if len(self.targets) > 1 and n > 1:
            targets, first, segments = self.targets[:n], 0, []
            for index, target in enumerate(targets):
                count = n // len(targets) + int(index < n % len(targets))
                segments.append({"first": first, "count": count, "candidate": target}); first += count
            yield mixed_candidate(self.current, segments, self.registry_sha, n)
    def resolve(self, plan_sha):
        for row in self:
            if row["planSha256"] == plan_sha: return row
        raise ValueError("mixed selection was not issued by this actual operation's candidate pool")
