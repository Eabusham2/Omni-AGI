"""Read-only custody qualification for an ALREADY initialized native fixture.

Only Python's standard library is imported. This is not a native Load, tensor
decoder, training receipt generator, origin repair or readiness override.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat


HASH = re.compile(r"^[a-f0-9]{64}$")
ORIGIN_FILES = (
    "brain.json", "core.safetensors", "plasticity.safetensors",
    "substrate/manifest.json", "state/manifest.json",
    "packed-ternary/manifest.json", "packed-ternary/manifest.sha256",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("fixture custody requires regular files: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return {"bytes": info.st_size, "sha256": result.hexdigest()}


def inside(root, relative):
    if not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("fixture declaration contains an unsafe relative path")
    result = root
    for part in relative.split("/"):
        result = result / part
        if result.is_symlink():
            raise ValueError("fixture dependency is a symlink")
    if root.resolve() not in result.resolve().parents:
        raise ValueError("fixture dependency escaped its declared root")
    return result


def read_json(path):
    digest(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("fixture metadata/manifest is not an object")
    return value


def content_hash(value):
    body = dict(value)
    claimed = body.pop("contentSha256", None)
    if not isinstance(claimed, str) or not HASH.fullmatch(claimed) or hashlib.sha256(canonical(body)).hexdigest() != claimed:
        raise ValueError("fixture manifest content hash is invalid")


def learned_closure(engine, metadata):
    """Check selected committed generations and EVERY declared blob, boundedly."""
    files = {"brain.json": digest(engine / "brain.json")}
    generations = {}
    for directory, pointer in (("state", metadata.get("mutable_state")),
                               ("substrate", (metadata.get("substrate") or {}).get("persistence"))):
        if not isinstance(pointer, dict) or not isinstance(pointer.get("activeGeneration"), str) or not HASH.fullmatch(pointer["activeGeneration"]):
            raise ValueError("fixture learned generation pointer is missing")
        relative = "generations/" + pointer["activeGeneration"] + "/manifest.json"
        if pointer.get("generationManifest") != relative or pointer.get("contentSha256") != pointer["activeGeneration"]:
            raise ValueError("fixture learned generation declaration changed")
        path = inside(engine / directory, relative)
        actual = digest(path)
        if actual["sha256"] != pointer.get("generationManifestSha256"):
            raise ValueError("fixture learned generation manifest SHA differs")
        manifest = read_json(path)
        content_hash(manifest)
        if manifest.get("contentSha256") != pointer["activeGeneration"]:
            raise ValueError("fixture learned generation identity differs")
        files[directory + "/" + relative] = actual
        generations[directory] = manifest
        if directory == "state":
            if manifest.get("format") != "omni-mutable-state" or manifest.get("formatVersion") != 1 or not isinstance(manifest.get("roles"), dict):
                raise ValueError("fixture mutable state format is invalid")
            specs = list(manifest["roles"].values())
        else:
            if manifest.get("format") != "omni-substrate-shards" or manifest.get("formatVersion") != 3 or not isinstance(manifest.get("shards"), list):
                raise ValueError("fixture substrate format is invalid")
            specs = [entry[role] for entry in manifest["shards"] for role in ("records", "tensors") if entry.get(role) is not None]
        for spec in specs:
            if not isinstance(spec, dict) or not isinstance(spec.get("sha256"), str) or not HASH.fullmatch(spec["sha256"]):
                raise ValueError("fixture learned payload declaration is invalid")
            payload = inside(engine / directory, spec.get("path"))
            actual = digest(payload)
            if actual != {"bytes": spec.get("bytes"), "sha256": spec["sha256"]}:
                raise ValueError("fixture learned payload bytes/SHA differ: " + str(payload))
            files[directory + "/" + spec["path"]] = actual
    for name in ("core", "plasticity"):
        path = engine / (name + ".safetensors")
        actual = digest(path)
        spec = generations["state"]["roles"].get(name)
        if not isinstance(spec, dict) or actual["sha256"] != spec.get("sha256"):
            raise ValueError("fixture active neural file differs from its committed generation")
        files[path.name] = actual
    return files


def qualify_initialized_source(source):
    source = Path(source).resolve()
    engine = source / "engine"
    report = {"format": "omni-inherited-fixture-source-qualification", "version": 1,
              "source": str(source), "qualified": False, "neuralExecution": False,
              "currentLearnedClosureVerified": False, "missingOriginFiles": [], "problems": []}
    try:
        metadata = read_json(engine / "brain.json")
        if metadata.get("brain_id") != source.name or (metadata.get("config") or {}).get("origin_kind") != "ground-up" or metadata.get("connected_regions") is not None:
            raise ValueError("fixture source must be its own single initialized native owner")
        report["brainId"] = metadata["brain_id"]
        report["configuration"] = metadata["config"]
        report["counters"] = metadata.get("counters")
        origin = engine / "origin"
        missing = [relative for relative in ORIGIN_FILES if not (origin / relative).is_file() or (origin / relative).is_symlink()]
        report["missingOriginFiles"] = missing
        report["currentFiles"] = learned_closure(engine, metadata)
        report["currentLearnedClosureVerified"] = True
        if missing:
            raise ValueError("code-required immutable native origin is incomplete; a current checkpoint cannot substitute")
        saved = read_json(origin / "brain.json")
        manifest = saved.get("ground_up_training_manifest")
        if not isinstance(manifest, dict) or manifest.get("format") != "omni-ground-up-curriculum-manifest" or manifest.get("formatVersion") != 3 or manifest.get("baseFrozen") is not False:
            raise ValueError("fixture immutable origin has no supported real training manifest")
        content_hash(manifest)
        receipt = manifest.get("trainingReceipt")
        if not isinstance(receipt, dict):
            raise ValueError("fixture immutable origin has no real training receipt")
        content_hash(receipt)
        report["originFiles"] = learned_closure(origin, saved)
        packed_path = origin / "packed-ternary/manifest.json"
        packed = read_json(packed_path)
        packed_hash = digest(packed_path)
        if (origin / "packed-ternary/manifest.sha256").read_text(encoding="ascii").strip() != packed_hash["sha256"]:
            raise ValueError("fixture origin packed manifest sidecar differs")
        content_hash(packed)
        packed_metadata = packed.get("metadata")
        expected = {"brainId": saved.get("brain_id"), "originKind": "ground-up", "baseFrozen": False,
                    "groundUpCurriculumSha256": manifest.get("sha256"),
                    "groundUpTrainingManifestSha256": manifest["contentSha256"],
                    "groundUpTrainingReceiptSha256": receipt["contentSha256"],
                    "parameterChecksum": receipt.get("parameterChecksumAfter")}
        if not isinstance(packed_metadata, dict) or any(packed_metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("fixture packed origin does not bind its actual training receipt")
        for spec in packed.get("shards", []):
            if not isinstance(spec, dict) or not isinstance(spec.get("file"), str) or "/" in spec["file"]:
                raise ValueError("fixture packed shard declaration is invalid")
            actual = digest(inside(origin / "packed-ternary", spec["file"]))
            if actual != {"bytes": spec.get("byteLength"), "sha256": spec.get("sha256")}:
                raise ValueError("fixture packed origin payload differs")
            report["originFiles"]["packed-ternary/" + spec["file"]] = actual
        report["originFiles"]["packed-ternary/manifest.json"] = packed_hash
        report["originFiles"]["packed-ternary/manifest.sha256"] = digest(origin / "packed-ternary/manifest.sha256")
        report["qualified"] = True
        report["nativeSemanticVerification"] = "still-required-through-real-Worker"
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["problems"].append(str(error))
    return report


def copy_qualified_initialized_source(source, target):
    report = qualify_initialized_source(source)
    if report["qualified"] is not True:
        raise ValueError("native source cannot be copied as a qualified fixture: " + "; ".join(report["problems"]))
    target = Path(target).resolve()
    if not target.parent.name.startswith("omni-connected-current-seal-") or target.name != report["brainId"] or target.exists():
        raise ValueError("fixture destination is not a new private owner path")
    shutil.copytree(Path(source) / "engine", target / "engine", symlinks=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="existing native instance root, not an engine/origin or arbitrary metadata file")
    arguments = parser.parse_args()
    report = qualify_initialized_source(arguments.source)
    # Summary is bounded. Exact payload hashes remain available to the private
    # host preparer; stdout is not an unbounded copy of the source tree.
    summary = {key: value for key, value in report.items() if key not in {"currentFiles", "originFiles", "configuration"}}
    summary["currentFileCount"] = len(report.get("currentFiles", {}))
    summary["originFileCount"] = len(report.get("originFiles", {}))
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if report["qualified"] else 2)
