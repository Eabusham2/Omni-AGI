"""Cross-process owner lease for one app-managed brain directory.

Only neural workers acquire this exclusive lease. A read-only inspection
worker never opens the mutable live paging cache. The lock file lives in the
brain's parent, not in the deletable brain directory, and is never unlinked:
recreating a locked inode could let two workers believe they own one brain.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Optional


class BrainLeaseBusy(RuntimeError):
    """A separate neural worker already owns this brain."""


class BrainOwnerLease:
    """One nonblocking OS lock held for a worker's brain lifetime."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        identity = hashlib.sha256(
            str(self.storage_path).encode("utf-8")
        ).hexdigest()
        self.directory = self.storage_path.parent / ".omni-brain-leases"
        self.path = self.directory / (identity + ".lock")
        self._descriptor: Optional[int] = None
        self._identity: Optional[tuple[int, int]] = None

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> "BrainOwnerLease":
        if self._descriptor is not None:
            self.assert_exclusive()
            return self
        if self.directory.is_symlink():
            raise ValueError("brain lease directory must not be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("brain lease file must not be a symlink")
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = os.open(self.path, flags)
        try:
            details = os.fstat(descriptor)
            visible = os.stat(self.path, follow_symlinks=False)
            if (
                not stat.S_ISREG(details.st_mode)
                or not stat.S_ISREG(visible.st_mode)
                or (visible.st_dev, visible.st_ino)
                != (details.st_dev, details.st_ino)
                or details.st_nlink != 1
            ):
                raise ValueError("brain lease is not an owned regular file")
            if details.st_size == 0:
                os.write(descriptor, b"1")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise BrainLeaseBusy("brain is active in another neural worker") from error
            else:
                import fcntl

                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise BrainLeaseBusy("brain is active in another neural worker") from error
            self._descriptor = descriptor
            self._identity = (details.st_dev, details.st_ino)
            self.assert_exclusive()
            return self
        except BaseException:
            if self._descriptor is None:
                os.close(descriptor)
            else:
                self.release()
            raise

    def assert_exclusive(self) -> None:
        descriptor = self._descriptor
        identity = self._identity
        if descriptor is None or identity is None:
            raise RuntimeError("brain owner lease is not held")
        details = os.fstat(descriptor)
        visible = os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(visible.st_mode)
            or (details.st_dev, details.st_ino) != identity
            or (visible.st_dev, visible.st_ino) != identity
            or visible.st_nlink != 1
        ):
            raise RuntimeError("brain owner lease file changed while locked")

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        self._identity = None
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "BrainOwnerLease":
        return self.acquire()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()
