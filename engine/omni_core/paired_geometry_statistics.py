"""Bounded source-free paired native-loss evidence, not invented quality.

Record-level paired signs use a conservative one-sided Hoeffding bound under
the explicitly stated independent-unit assumption. Ties are preservation, not
benefit. Corpus identity/count/weight pairing is exact and stream verified.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import uuid


FORMAT = "omni-paired-native-losses"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class PairedLossWriter:
    def __init__(self, path, policy, expected_records):
        self.path, self.count, self.expected = Path(path), 0, int(expected_records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = self.path.with_name("." + self.path.name + "." + uuid.uuid4().hex + ".next")
        self.lease = policy.reserve_spill(((self.expected * 256 + 4096 + 65535) // 65536) * 65536,
            "protected paired heldout loss observations")
        self.lease.bind_path(self.stage)
        try: self.stream = self.stage.open("xb")
        except BaseException:
            self.lease.release()
            raise
        self.stream.write((json.dumps({"format": FORMAT, "formatVersion": 1, "records": self.expected}, separators=(",", ":")) + "\n").encode())

    def add(self, category, key, loss, weight):
        if (category not in {"token", "modality", "tool"} or not isinstance(key, str) or len(key) != 64
            or any(char not in "0123456789abcdef" for char in key) or isinstance(loss, bool)
            or not math.isfinite(loss) or loss < 0 or type(weight) is not int or weight < 1
            or self.count >= self.expected):
            raise ValueError("paired loss observation is invalid or exceeds its declared corpus")
        encoded = (json.dumps({"category": category, "key": key, "loss": float(loss), "weight": weight},
            sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(encoded) > 256: raise ValueError("paired loss wire record exceeds its fixed schema")
        self.stream.write(encoded); self.count += 1

    def finish(self):
        if self.count != self.expected: raise ValueError("paired loss corpus is incomplete")
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        self.lease.commit(path=self.stage)
        os.replace(self.stage, self.path)
        self.lease.commit(path=self.path)
        return {"format": FORMAT, "formatVersion": 1, "records": self.count,
            "sha256": file_sha(self.path), "path": self.path.name}

    def abort(self):
        if not self.stream.closed: self.stream.close()
        self.stage.unlink(missing_ok=True)
        self.lease.release()


def _rows(path, descriptor):
    if (descriptor.get("format") != FORMAT or descriptor.get("formatVersion") != 1
        or descriptor.get("path") != Path(path).name or file_sha(path) != descriptor.get("sha256")):
        raise ValueError("paired heldout loss corpus identity changed")
    with Path(path).open("rb") as source:
        header = json.loads(source.readline(4096))
        if header != {"format": FORMAT, "formatVersion": 1, "records": descriptor["records"]}:
            raise ValueError("paired loss header does not bind its declared record count")
        count = 0
        for raw in source:
            if len(raw) > 256: raise ValueError("paired loss row is not bounded by its wire schema")
            value = json.loads(raw)
            if set(value) != {"category", "key", "loss", "weight"}: raise ValueError("paired loss row schema changed")
            if (value["category"] not in {"token", "modality", "tool"}
                or not isinstance(value["key"], str) or len(value["key"]) != 64
                or any(char not in "0123456789abcdef" for char in value["key"])
                or type(value["weight"]) is not int or value["weight"] < 1
                or isinstance(value["loss"], bool) or not isinstance(value["loss"], (int, float))
                or not math.isfinite(value["loss"]) or value["loss"] < 0):
                raise ValueError("paired loss values are invalid/nonfinite")
            count += 1
            yield value
        if count != descriptor["records"]: raise ValueError("paired loss count changed")


def paired_improvement_statistics(baseline_path, candidate_path, baseline, candidate, *, objectives, alpha=.05):
    objective_domains = {"language-prediction": "token", "modality-reconstruction": "modality",
        "typed-tool-prediction": "tool"}
    agreed = {objective_domains[name] for name in objectives if name in objective_domains}
    if not agreed: raise ValueError("statistical geometry promotion needs an agreed measured objective")
    if baseline.get("records") != candidate.get("records"): raise ValueError("paired corpus counts differ")
    aggregates = {name: {"wins": 0, "losses": 0, "ties": 0, "weight": 0, "baselineSum": 0., "candidateSum": 0.}
        for name in ("token", "modality", "tool")}
    old, new = iter(_rows(baseline_path, baseline)), iter(_rows(candidate_path, candidate))
    for _ in range(baseline["records"]):
        a, b = next(old), next(new)
        if (a["category"], a["key"], a["weight"]) != (b["category"], b["key"], b["weight"]):
            raise ValueError("paired loss sources/order/label weights differ")
        value = aggregates[a["category"]]
        delta = float(a["loss"]) - float(b["loss"])
        if not math.isfinite(delta): raise ValueError("paired loss delta is nonfinite")
        value["wins" if delta > 0 else "losses" if delta < 0 else "ties"] += 1
        value["weight"] += a["weight"]
        value["baselineSum"] += a["loss"] * a["weight"]
        value["candidateSum"] += b["loss"] * b["weight"]
    if next(old, None) is not None or next(new, None) is not None: raise ValueError("paired loss corpus has extra observations")
    threshold, supported, preserved = alpha / len(agreed), [], True
    result = {}
    for category, value in aggregates.items():
        if value["weight"] < 1: raise ValueError("paired heldout category is empty")
        before, after = value["baselineSum"] / value["weight"], value["candidateSum"] / value["weight"]
        n = value["wins"] + value["losses"]
        margin = value["wins"] / n - .5 if n else 0.
        bound = math.exp(max(-744., -2 * n * margin * margin)) if margin > 0 else 1.
        nonregressed = after <= before + 1e-12 * max(1., abs(before))
        benefit = category in agreed and after < before and bound <= threshold
        preserved = preserved and nonregressed
        if benefit: supported.append(category)
        result[category] = {**value, "baselineLoss": before, "candidateLoss": after, "nonRegression": nonregressed,
            "pValueUpperBound": bound, "alpha": threshold, "supportedBenefit": benefit,
            "relativeGain": (before - after) / max(abs(before), 1e-12)}
    return {"method": "paired-record-sign-hoeffding-one-sided", "familyWiseAlpha": alpha,
        "independenceAssumption": "registered logical records are independent statistical units; not established by a checksum",
        "allRegisteredPairs": True, "supportedObjectiveDomains": sorted(supported), "domains": result,
        "preservationPassed": preserved, "passed": preserved and bool(supported)}
