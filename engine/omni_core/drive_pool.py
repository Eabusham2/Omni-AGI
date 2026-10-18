"""Authorized session-owned ephemeral spill files across physical drive pools.

Imported configuration paths are preferences, never authority. The desktop
picker records local owner grants in the shared ledger; this module only reads
them. Authoritative checkpoints keep their existing instance-local paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import threading
import uuid
from pathlib import Path

from .shared_resource_ledger import SharedAllocationLease, SharedQuotaPause


def selection_signature(selection):
    body = {key: selection[key] for key in ("deviceId", "enabled", "mode")}
    body["percentBits"] = struct.pack(">d", float(selection["percent"])).hex()
    return "native-spill-selection-1:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _directory_identity(path):
    value = Path(path).lstat()
    if (not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode)
            or int(getattr(value, "st_file_attributes", 0)) & 0x400):
        raise ValueError("spill root must remain a real directory")
    return str(value.st_dev), str(value.st_ino)


class PlacedSpillLease:
    def __init__(self, manager, grant, directory, identity, path, quota, promised):
        self.manager, self.grant = manager, grant
        self.directory, self.identity, self.path = directory, identity, path
        self.quota, self.promised_bytes = quota, promised
        self.allocated = False
        self._held = False
        self._physical = None

    def owns_path(self, path):
        if Path(path) != self.path or self.path.parent != self.directory:
            return False
        self.manager.verify_grant(self.grant)
        return _directory_identity(self.directory) == self.identity

    def bind_path(self, path):
        if not self.owns_path(path):
            raise ValueError("spill lease cannot bind a caller-selected path")
        self.quota.bind_path(path)
        return self

    def open(self, flags=os.O_RDWR):
        if not self.owns_path(self.path):
            raise ValueError("spill directory identity changed before allocation")
        flags |= os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        if os.open in os.supports_dir_fd:
            parent = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                observed = os.fstat(parent)
                if (str(observed.st_dev), str(observed.st_ino)) != self.identity:
                    raise ValueError("spill directory identity changed before open")
                return os.open(self.path.name, flags, 0o600, dir_fd=parent)
            finally:
                os.close(parent)
        return os.open(self.path, flags, 0o600)

    def commit(self, actual_allocated_bytes=None, *, path=None, promised_bytes=0, retain_open_backing=False):
        self.bind_path(self.path if path is None else path)
        self.quota.commit(actual_allocated_bytes, path=self.path, promised_bytes=promised_bytes,
                          retain_open_backing=retain_open_backing)
        self.allocated = True
        self._held = retain_open_backing
        value = self.path.stat()
        self._physical = int(value.st_blocks) * 512 if hasattr(value, "st_blocks") else value.st_size
        return self

    def refresh_physical(self):
        value = self.path.stat()
        physical = int(value.st_blocks) * 512 if hasattr(value, "st_blocks") else value.st_size
        if physical != self._physical:
            self.commit(path=self.path, promised_bytes=value.st_size, retain_open_backing=self._held)

    def release(self):
        self.quota.release()
        if not self.path.exists():
            try:
                if _directory_identity(self.directory) == self.identity:
                    self.directory.rmdir()
            except OSError:
                pass

    def backing_closed(self):
        self.quota.backing_closed()
        self.release()


class DrivePoolManager:
    def __init__(self, policy, settings, *, disk_usage=shutil.disk_usage):
        self.policy, self.settings, self.disk_usage = policy, settings, disk_usage
        self._session = uuid.uuid4().hex
        self._directories = {}
        self._lock = threading.RLock()
        self._cleanup_done = False

    def cleanup_stale(self, choices, cancelled=None):
        """Retire only registered generated files from demonstrably dead jobs."""
        by_id = {grant["pool_id"]: grant for grant, _selection in choices}
        ledger = self.policy.shared_ledger
        cursor = ""
        while True:
            rows = ledger.drive_cleanup_candidates(self.policy.shared_resource_owner_id, after_token=cursor)
            if not rows:
                return
            for record in rows:
                if cancelled is not None:
                    cancelled()
                cursor = record["token"]
                grant = by_id.get(record["pool_id"])
                if grant is None or ledger.drive_backing_has_live_owner(record["token"]):
                    continue
                self.verify_grant(grant)
                path = Path(record["path"])
                if (path.parent.parent != Path(grant["backing_path"])
                        or re.fullmatch(r"[a-f0-9]{32}", path.parent.name) is None
                        or re.fullmatch(r"[a-f0-9]{32}\.(?:packed-page|router-page|activity|saved-activation|packed-rollback|mcp-media)", path.name) is None):
                    continue
                try:
                    _directory_identity(path.parent)
                    value = path.lstat()
                    identity = "%d:%d" % (value.st_dev, value.st_ino)
                    if not stat.S_ISREG(value.st_mode) or (record["identity"] is not None and identity != record["identity"]):
                        continue
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    continue  # Detached/OS-held storage remains charged.
                SharedAllocationLease(ledger, record["token"], "spill").backing_closed()
                try:
                    path.parent.rmdir()
                except OSError:
                    pass

    @staticmethod
    def verify_grant(grant):
        root, backing = Path(grant["path"]), Path(grant["backing_path"])
        if (not root.is_absolute() or backing.parent != root
                or not backing.name.startswith(".omni-spill-")):
            raise ValueError("invalid local spill-root grant")
        if _directory_identity(root) != (grant["device"], grant["inode"]):
            raise ValueError("selected drive root changed; select it again")
        if _directory_identity(backing) != (grant["backing_device"], grant["backing_inode"]):
            raise ValueError("owned drive backing changed; select the drive again")

    def _selected(self):
        roots = self.settings["storageDirectories"]
        if not self.settings["allowStorageSpill"]:
            raise SharedQuotaPause("storage spill is disabled", {"spillDisabled": True})
        if not roots:
            return []
        grants = {value["path"]: value for value in self.policy.shared_ledger.authorized_drive_pools(self.policy.shared_resource_owner_id)}
        result = []
        for root in roots:
            # Do not resolve a path provided by imported metadata into new
            # authority. It must match the exact canonical path from the picker.
            grant = grants.get(root)
            if grant is None:
                raise SharedQuotaPause("select the requested spill directories on this device first", {"storageAuthorizationRequired": True})
            self.verify_grant(grant)
            identifier = "storage:" + hashlib.sha256(root.encode()).hexdigest()[:24]
            choice = next((item for item in self.settings["allocations"] if item["deviceId"] == identifier),
                          {"deviceId": identifier, "enabled": True, "mode": "auto", "percent": 90, "autoIncrease": False})
            if choice["enabled"]:
                result.append((grant, choice))
        # Rates are observed by main, not imported from model settings.
        return sorted(result, key=lambda item: (-int(item[0]["rate"]), item[0]["pool_id"]))

    def allocate(self, byte_count, operation, suffix, *, cancelled=None):
        if type(byte_count) is not int or byte_count < 0 or suffix not in {".packed-page", ".router-page", ".activity", ".saved-activation", ".packed-rollback", ".mcp-media"}:
            raise ValueError("invalid internal spill allocation")
        with self._lock:
            choices = self._selected()
            if not self.settings["storageDirectories"]:
                return None  # Preserve the existing single-drive path exactly.
            if not self._cleanup_done:
                self.cleanup_stale(choices, cancelled)
                self._cleanup_done = True
            failures = []
            for grant, selection in choices:
                if cancelled is not None:
                    cancelled()
                self.verify_grant(grant)
                usage = self.disk_usage(grant["path"])
                reserve = max(int(self.policy.configured_disk_reserve), self.policy._adaptive_disk_reserve(int(usage.total)))
                try:
                    unit = max(4096, int(os.statvfs(grant["path"]).f_frsize))
                except (AttributeError, OSError):
                    unit = 65536
                promised = math.ceil(byte_count / unit) * unit
                proposed = int(max(0, usage.free - reserve) * selection["percent"] / 100)
                # Enabling automatic growth alone is not a manual resize and
                # cannot bypass the separate storage-auto-increase switch.
                signature = selection_signature(selection)
                try:
                    quota = self.policy.shared_ledger.reserve_drive(self.policy.shared_resource_owner_id,
                        grant["pool_id"], promised, proposed_ceiling=proposed, selection_signature=signature,
                        auto_increase=selection["autoIncrease"] and self.settings["autoIncreaseStorage"],
                        free_bytes=int(usage.free), reserve_bytes=reserve)
                except SharedQuotaPause as error:
                    failures.append(error.status)
                    continue
                lease = None
                try:
                    directory = Path(grant["backing_path"]) / self._session
                    if grant["pool_id"] not in self._directories:
                        directory.mkdir(mode=0o700, exist_ok=False)
                        self._directories[grant["pool_id"]] = _directory_identity(directory)
                    elif not directory.exists():
                        directory.mkdir(mode=0o700, exist_ok=False)
                        self._directories[grant["pool_id"]] = _directory_identity(directory)
                    identity = self._directories[grant["pool_id"]]
                    if _directory_identity(directory) != identity:
                        raise ValueError("session spill directory was replaced")
                    path = directory / (uuid.uuid4().hex + suffix)
                    lease = PlacedSpillLease(self, grant, directory, identity, path, quota, promised)
                    lease.bind_path(path)
                    if cancelled is not None:
                        cancelled()
                    return lease
                except BaseException:
                    if lease is not None:
                        lease.release()
                    else:
                        quota.release()
                    raise
            raise SharedQuotaPause(operation + " awaits a selected drive pool", {"drivePlacementFailures": failures, "sourceDiscarded": False})
