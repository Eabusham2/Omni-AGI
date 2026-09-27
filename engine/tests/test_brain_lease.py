"""Cross-process lease tests without loading or training a model."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omni_core.brain_lease import BrainOwnerLease


ENGINE = Path(__file__).resolve().parents[1]


_CHILD = """
import sys
from pathlib import Path
from omni_core.brain_lease import BrainLeaseBusy, BrainOwnerLease
try:
    with BrainOwnerLease(Path(sys.argv[1])):
        sys.exit(0)
except BrainLeaseBusy:
    sys.exit(3)
"""


class BrainLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(prefix="omni-brain-lease-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _child_exit(self, storage: Path) -> int:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, str(storage)],
            check=False, capture_output=True, text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    part for part in (str(ENGINE), os.environ.get("PYTHONPATH", ""))
                    if part
                ),
            },
        )
        self.assertEqual(result.stderr, "")
        return result.returncode

    def test_same_brain_is_exclusive_across_processes_and_released(self) -> None:
        storage = self.root / "brain-a"
        lease = BrainOwnerLease(storage).acquire()
        first_inode = lease.path.stat().st_ino
        self.assertEqual(self._child_exit(storage), 3)
        lease.assert_exclusive()
        lease.release()
        self.assertEqual(self._child_exit(storage), 0)
        self.assertEqual(lease.path.stat().st_ino, first_inode)

    def test_distinct_brains_can_be_owned_simultaneously(self) -> None:
        with BrainOwnerLease(self.root / "brain-a") as owner:
            self.assertEqual(self._child_exit(self.root / "brain-b"), 0)
            owner.assert_exclusive()

    def test_unheld_lease_never_claims_exclusivity(self) -> None:
        lease = BrainOwnerLease(self.root / "brain-a")
        with self.assertRaises(RuntimeError):
            lease.assert_exclusive()


if __name__ == "__main__":
    unittest.main()
