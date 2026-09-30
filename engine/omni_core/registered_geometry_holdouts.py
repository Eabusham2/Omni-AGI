"""Independent saved-instance registration for real geometry holdout data.

Hashes bind a transaction to registered files; they are not signatures and
cannot authenticate a party rewriting every authoritative file. No model,
training, external data download, or ordinary-chat readiness check lives here.
"""

from pathlib import Path
import hashlib
import uuid

from .persistence import read_json, atomic_write_json
from .text_spool import bounded_json_sha256


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


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
            expected = {"path", "sha256", "records"} | ({"kind", "conditionText"} if category == "modality" and "kind" in item else set())
            if (not isinstance(item, dict) or set(item) != expected
                or not _sha(item["sha256"]) or type(item["records"]) is not int or item["records"] < 1
                or not isinstance(item["path"], str) or item["path"] in seen):
                raise ValueError("registered geometry holdout data declaration is invalid/repeated")
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
    engine, categories = Path(brain.engine_path), {}
    for category, files in declarations.items():
        if not isinstance(files, list) or not files: raise ValueError("geometry holdout category is empty")
        selected = []
        for entry in files:
            fields = {"path", "kind", "conditionText"} if category == "modality" else {"path", "records"}
            if not isinstance(entry, dict) or set(entry) != fields: raise ValueError("geometry source declaration schema is invalid")
            path = Path(entry["path"])
            if path.is_symlink() or not path.is_file() or not path.stat().st_size: raise ValueError("geometry source is unavailable")
            if category != "modality" and (type(entry["records"]) is not int or entry["records"] < 1): raise ValueError("geometry source count must be positive")
            if category == "modality" and (entry["kind"] not in {"image", "audio", "video"} or not isinstance(entry["conditionText"], str) or not entry["conditionText"].strip() or "\x00" in entry["conditionText"]):
                raise ValueError("geometry media needs explicit kind and literal condition text")
            brain.resource_policy.require_disk(path.stat().st_size * 2 + 65536, "registered geometry holdout data")
            target = engine / "evaluation" / "data" / (uuid.uuid4().hex + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = _file_sha(path, cancelled)
            with path.open("rb") as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    if cancelled is not None and cancelled(): raise InterruptedError("geometry registration cancelled")
                    output.write(block)
            if _file_sha(target, cancelled) != before or _file_sha(path, cancelled) != before:
                raise ValueError("selected geometry source changed during registration")
            item = {"path": target.relative_to(engine).as_posix(), "sha256": before,
                "records": 1 if category == "modality" else entry["records"]}
            if category == "modality": item.update(kind=entry["kind"], conditionText=entry["conditionText"])
            selected.append(item)
        categories[category] = selected
    manifest = engine / "evaluation" / "geometry-holdouts.json"
    previous_manifest = read_json(manifest) if manifest.is_file() else None
    atomic_write_json(manifest, {"format": "omni-registered-geometry-holdouts", "formatVersion": 1, "categories": categories})
    reference = {"format": "omni-geometry-holdout-registration", "formatVersion": 1,
        "manifestPath": "evaluation/geometry-holdouts.json", "manifestSha256": _file_sha(manifest)}
    previous = getattr(brain, "geometry_holdout_registration", None)
    brain.geometry_holdout_registration = reference
    try:
        brain.save()
        if read_json(engine / "brain.json").get("geometry_holdout_registration") != reference:
            raise ValueError("saved instance did not persist its protected holdout registration")
    except BaseException:
        # Do not roll back a registration already committed by the save.
        if read_json(engine / "brain.json").get("geometry_holdout_registration") != reference:
            brain.geometry_holdout_registration = previous
            if previous_manifest is not None:
                atomic_write_json(manifest, previous_manifest)
        raise
    return load_registered_geometry_holdouts(engine, cancelled=cancelled)
