"""Main-channel-owned temporary modality output; never a model argument."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import secrets


@dataclass(frozen=True)
class SpontaneousOutputLease:
    brain_id: str
    job_id: str
    lease_id: str
    directory: Path


class SpontaneousOutputLeases:
    def __init__(self):
        self._leases = {}
        self._consumed = set()

    def authorize(self, engine: Path, brain_id: str, job_id: str, origin: dict):
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", job_id) or set(origin) != {"kind", "authority", "actionEventId"} or origin.get("kind") != "spontaneous" or origin.get("authority") != "main-organic-action" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", str(origin.get("actionEventId", ""))):
            raise ValueError("spontaneous output needs actual main organic job ownership")
        engine = Path(engine)
        if engine.is_symlink() or not engine.is_dir():
            raise ValueError("spontaneous output requires its existing native owner")
        parent = engine / ".spontaneous-imagination"
        parent.mkdir(exist_ok=True)
        if parent.is_symlink() or parent.resolve().parent != engine.resolve():
            raise ValueError("unsafe spontaneous output parent")
        if (brain_id, job_id) in self._leases:
            raise ValueError("spontaneous job already owns an output lease")
        lease_id = secrets.token_hex(32)
        directory = parent / lease_id
        directory.mkdir(mode=0o700)
        value = SpontaneousOutputLease(brain_id, job_id, lease_id, directory)
        (directory / "owner.json").write_text(json.dumps({"format": "omni-spontaneous-output-owner", "version": 1, "brainId": brain_id, "jobId": job_id, "leaseId": lease_id, "origin": origin}), encoding="utf-8")
        self._leases[(brain_id, job_id)] = value
        return {"brainId": brain_id, "jobId": job_id, "leaseId": lease_id, "retention": "ephemeral-spontaneous"}

    def consume(self, brain_id: str, job_id: str, lease_id: str):
        value = self._leases.get((brain_id, job_id))
        if value is None or value.lease_id != lease_id or (brain_id, job_id) in self._consumed:
            raise ValueError("spontaneous output lease is absent, changed or already consumed")
        if value.directory.is_symlink() or not value.directory.is_dir() or list(value.directory.iterdir()) != [value.directory / "owner.json"]:
            raise ValueError("spontaneous output scratch is no longer fresh")
        self._check_marker(value)
        self._consumed.add((brain_id, job_id))
        return value.directory

    def discard(self, brain_id: str, job_id: str, lease_id: str):
        value = self._leases.get((brain_id, job_id))
        if value is None or value.lease_id != lease_id:
            raise ValueError("scratch cleanup has no exact spontaneous output owner")
        if value.directory.is_symlink():
            raise ValueError("scratch owner became a link; cleanup refused")
        if not value.directory.exists():
            self._leases.pop((brain_id, job_id))
            self._consumed.discard((brain_id, job_id))
            return {"brainId": brain_id, "jobId": job_id, "discarded": True, "alreadyAbsent": True}
        checked_output_directory(value.directory.parent.parent, value.directory)
        self._check_marker(value)
        children = list(value.directory.iterdir())
        for child in children:
            if child.is_symlink() or not child.is_file() or child.name != "owner.json" and not re.fullmatch(r"(?:[a-f0-9]{32}|[a-f0-9]{64})\.(?:png|wav|mp4)", child.name):
                raise ValueError("scratch cleanup found an unowned payload; preserving evidence")
        for child in children:
            child.unlink()
        value.directory.rmdir()
        self._leases.pop((brain_id, job_id))
        self._consumed.discard((brain_id, job_id))
        return {"brainId": brain_id, "jobId": job_id, "discarded": True}

    @staticmethod
    def _check_marker(value):
        marker = value.directory / "owner.json"
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 65536:
            raise ValueError("spontaneous output ownership marker changed")
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("format") != "omni-spontaneous-output-owner" or data.get("version") != 1 or data.get("brainId") != value.brain_id or data.get("jobId") != value.job_id or data.get("leaseId") != value.lease_id:
            raise ValueError("spontaneous output identity changed")

    def discard_brain(self, brain_id):
        unresolved = []
        for owner, job in list(self._leases):
            if owner != brain_id:
                continue
            value = self._leases[(owner, job)]
            try:
                self.discard(owner, job, value.lease_id)
            except (OSError, ValueError):
                unresolved.append(job)
        return {"complete": not unresolved, "preservedUnverifiedJobIds": unresolved}


def checked_output_directory(engine: Path, owned: Path = None):
    if owned is None:
        return engine / "artifacts"
    owned = Path(owned)
    if engine.is_symlink() or (engine / ".spontaneous-imagination").is_symlink() or owned.is_symlink() or not re.fullmatch(r"[a-f0-9]{64}", owned.name) or owned.resolve().parent != (engine / ".spontaneous-imagination").resolve():
        raise ValueError("modality output directory is not its authenticated fresh job scratch")
    marker = owned / "owner.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("spontaneous output owner is unavailable")
    return owned
