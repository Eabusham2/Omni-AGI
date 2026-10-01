"""Independent saved-instance registration for real geometry holdout data.

Hashes bind a transaction to registered files; they are not signatures and
cannot authenticate a party rewriting every authoritative file. No model,
training, external data download, or ordinary-chat readiness check lives here.
"""

from pathlib import Path
import hashlib
import json
import os
import re
import uuid

from .persistence import read_json, atomic_write_json
from .text_spool import bounded_json_sha256


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def portable_geometry_reference(relative):
    """Keep logical legacy references/hash intact; use one safe physical alias."""
    if not isinstance(relative, str) or "\\" in relative or "\x00" in relative:
        raise ValueError("geometry reference is invalid")
    match = re.fullmatch(r"evaluation/data/([0-9a-f]{32})(\.[^/\\\x00]+)?", relative)
    if match is None:
        if relative == "evaluation/geometry-holdouts.json": return relative
        raise ValueError("geometry data reference is not a generated owned filename")
    suffix = match.group(2) or ""
    if not suffix or re.fullmatch(r"\.[A-Za-z0-9_-]{1,128}", suffix): return relative
    return "evaluation/data/" + match.group(1) + ".legacy-" + hashlib.sha256(suffix.encode("utf-8")).hexdigest()


def _registered_file(engine, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("geometry holdout reference is invalid")
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts) or path.parts[0] != "evaluation":
        raise ValueError("geometry holdout references must remain inside saved-instance evaluation data")
    value = engine
    for part in path.parts:
        value = value / part
        if value.is_symlink(): raise ValueError("geometry holdout reference traverses a symlink")
    if not value.exists() and relative.startswith("evaluation/data/"):
        alias = portable_geometry_reference(relative)
        value = engine / alias
        if value.is_symlink(): raise ValueError("geometry alias is a symlink")
    if not value.is_file() or value.stat().st_size < 1:
        raise ValueError("registered geometry holdout data is unavailable or empty")
    return value


def _file_sha(path, cancelled=None):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if cancelled is not None and cancelled(): raise InterruptedError("registered holdout verification cancelled")
            digest.update(block)
    return digest.hexdigest()


def load_registered_geometry_holdouts(engine_path, *, cancelled=None):
    engine = Path(engine_path).resolve()
    metadata = read_json(engine / "brain.json")
    reference = metadata.get("geometry_holdout_registration")
    if (not isinstance(reference, dict) or set(reference) != {"format", "formatVersion", "manifestPath", "manifestSha256"}
        or reference["format"] != "omni-geometry-holdout-registration" or type(reference["formatVersion"]) is not int
        or reference["formatVersion"] != 1 or reference["manifestPath"] != "evaluation/geometry-holdouts.json"
        or not _sha(reference["manifestSha256"])):
        raise ValueError("geometry requires a protected saved-instance real-data holdout registration")
    path = _registered_file(engine, reference["manifestPath"])
    if _file_sha(path, cancelled) != reference["manifestSha256"]:
        raise ValueError("registered geometry holdout manifest hash changed")
    value = read_json(path)
    if (not isinstance(value, dict) or set(value) != {"format", "formatVersion", "categories"}
        or value["format"] != "omni-registered-geometry-holdouts" or type(value["formatVersion"]) is not int
        or value["formatVersion"] != 1 or not isinstance(value["categories"], dict)
        or set(value["categories"]) != {"token", "modality", "tool"}):
        raise ValueError("registered geometry holdout category schema is invalid")
    seen, categories = set(), {}
    for category, files in value["categories"].items():
        if not isinstance(files, list) or not files:
            raise ValueError("geometry held-out " + category + " data is unavailable")
        for item in files:
            expected = {"path", "sha256", "records"} | ({"kind", "conditionText"} if isinstance(item, dict) and category == "modality" and "kind" in item else set())
            if isinstance(item, dict) and "sourceName" in item: expected.add("sourceName")
            if (not isinstance(item, dict) or set(item) != expected
                or not _sha(item["sha256"]) or type(item["records"]) is not int or item["records"] < 1
                or not isinstance(item["path"], str) or item["path"] in seen):
                raise ValueError("registered geometry holdout data declaration is invalid/repeated")
            if "sourceName" in item and (not isinstance(item["sourceName"], str) or "\x00" in item["sourceName"]):
                raise ValueError("geometry source provenance is invalid")
            portable_geometry_reference(item["path"])
            source = _registered_file(engine, item["path"])
            if source == path or _file_sha(source, cancelled) != item["sha256"]:
                raise ValueError("registered geometry held-out data hash changed")
            seen.add(item["path"])
        categories[category] = {"files": files, "sourceSha256": bounded_json_sha256(files),
            "examples": sum(item["records"] for item in files)}
    return {"format": value["format"], "formatVersion": 1,
        "benchmarkSha256": reference["manifestSha256"], "categories": categories,
        "root": str(engine), "registeredDataVerified": True}


def register_geometry_holdouts(brain, declarations, *, cancelled=None):
    """Explicit selected real data only; no generated fallback or hidden split."""
    if not isinstance(declarations, dict) or set(declarations) != {"token", "modality", "tool"}:
        raise ValueError("geometry holdouts require token/modality/tool source declarations")
    engine, categories, owned, leases = Path(brain.engine_path), {}, [], []
    evaluation = engine / "evaluation"
    for directory in (evaluation, evaluation / "data"):
        if directory.is_symlink(): raise ValueError("geometry owned directory is a symlink")
        directory.mkdir(parents=True, exist_ok=True)
    manifest = evaluation / "geometry-holdouts.json"
    if manifest.is_symlink(): raise ValueError("geometry manifest is a symlink")
    previous_manifest = manifest.read_bytes() if manifest.is_file() else None
    previous = getattr(brain, "geometry_holdout_registration", None)
    reference, installed = None, False
    try:
        for category, files in declarations.items():
            if not isinstance(files, list) or not files: raise ValueError("geometry holdout category is empty")
            selected = []
            for entry in files:
                fields = {"path", "kind", "conditionText"} if category == "modality" else {"path", "records"}
                if not isinstance(entry, dict) or set(entry) != fields: raise ValueError("geometry source declaration schema is invalid")
                path = Path(entry["path"])
                if path.is_symlink() or not path.is_file() or not path.stat().st_size: raise ValueError("geometry source is unavailable")
                if category != "modality" and (type(entry["records"]) is not int or not 1 <= entry["records"] <= (1 << 53) - 1): raise ValueError("geometry source count must be positive")
                if category == "modality" and (entry["kind"] not in {"image", "audio", "video"} or not isinstance(entry["conditionText"], str) or not entry["conditionText"].strip() or "\x00" in entry["conditionText"]):
                    raise ValueError("geometry media needs explicit kind and literal condition text")
                suffix = path.suffix
                if category != "modality": suffix = ".jsonl"
                elif not re.fullmatch(r"\.[A-Za-z0-9_-]{1,128}", suffix):
                    with path.open("rb") as handle: magic = handle.read(12)
                    suffix = ".gif" if magic.startswith((b"GIF87a", b"GIF89a")) else (
                        ".webp" if magic.startswith(b"RIFF") and magic[8:12] == b"WEBP" else "." + entry["kind"])
                target = evaluation / "data" / (uuid.uuid4().hex + suffix)
                reserve = brain.resource_policy.reserve_spill(((path.stat().st_size + 65535) // 65536) * 65536, "owned geometry holdout copy")
                leases.append(reserve); reserve.bind_path(target)
                before = _file_sha(path, cancelled)
                with path.open("rb") as source, target.open("xb") as output:
                    owned.append(target)
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        if cancelled is not None and cancelled(): raise InterruptedError("geometry registration cancelled")
                        output.write(block)
                    output.flush(); os.fsync(output.fileno())
                if _file_sha(target, cancelled) != before or _file_sha(path, cancelled) != before:
                    raise ValueError("selected geometry source changed during registration")
                reserve.commit(path=target)
                item = {"path": target.relative_to(engine).as_posix(), "sha256": before,
                    "sourceName": path.name, "records": 1 if category == "modality" else entry["records"]}
                if category == "modality": item.update(kind=entry["kind"], conditionText=entry["conditionText"])
                selected.append(item)
            categories[category] = selected
        body = {"format": "omni-registered-geometry-holdouts", "formatVersion": 1, "categories": categories}
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        staged = evaluation / ("." + uuid.uuid4().hex + ".registration.next")
        manifest_lease = brain.resource_policy.reserve_spill(((len(payload) + len(previous_manifest or b"") + 65535) // 65536) * 65536, "owned geometry holdout manifest")
        leases.append(manifest_lease); manifest_lease.bind_path(staged)
        with staged.open("xb") as output:
            owned.append(staged)
            output.write(payload); output.flush(); os.fsync(output.fileno())
        reference = {"format": "omni-geometry-holdout-registration", "formatVersion": 1,
            "manifestPath": "evaluation/geometry-holdouts.json", "manifestSha256": _file_sha(staged)}
        os.replace(staged, manifest); installed = True
        brain.geometry_holdout_registration = reference
        brain.save()
        if read_json(engine / "brain.json").get("geometry_holdout_registration") != reference:
            raise ValueError("saved instance did not persist its protected holdout registration")
        manifest_lease.commit(path=manifest)
        return load_registered_geometry_holdouts(engine, cancelled=cancelled)
    except BaseException:
        # Do not roll back a registration already committed by the save.
        committed = reference is not None and read_json(engine / "brain.json").get("geometry_holdout_registration") == reference
        if not committed:
            brain.geometry_holdout_registration = previous
            if installed:
                if previous_manifest is None: manifest.unlink(missing_ok=True)
                else:
                    restoration = evaluation / ("." + uuid.uuid4().hex + ".restore.next")
                    with restoration.open("xb") as output:
                        output.write(previous_manifest); output.flush(); os.fsync(output.fileno())
                    os.replace(restoration, manifest)
                    # Retain the restored original manifest's physical charge.
                    leases[-1].commit(path=manifest)
            for path in reversed(owned):
                try: path.unlink(missing_ok=True)
                except OSError: pass  # An extant backing keeps its ledger charge.
            for lease in reversed(leases): lease.release()
        elif installed:
            # A postcommit exception leaves all published content owned/charged.
            leases[-1].commit(path=manifest)
        raise


def registered_geometry_snapshot_files(engine):
    """Portable physical paths while immutable logical refs stay unchanged."""
    engine = Path(engine)
    if read_json(engine / "brain.json").get("geometry_holdout_registration") is None: return {}
    declared = load_registered_geometry_holdouts(engine)
    result = {"evaluation/geometry-holdouts.json": engine / "evaluation/geometry-holdouts.json"}
    for category in declared["categories"].values():
        for item in category["files"]:
            relative = portable_geometry_reference(item["path"])
            if relative in result: raise ValueError("geometry portable reference is repeated")
            result[relative] = _registered_file(engine, item["path"])
    return result
