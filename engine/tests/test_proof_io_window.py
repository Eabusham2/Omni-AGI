import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_core.authenticated_paged_cache import AuthenticatedCacheSession, cache_session
from omni_core.paged_assembly_index import PagedAssemblyIndex


class ProofIOWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-proof-window-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.owner = PagedAssemblyIndex(self.root / "cache.sqlite3")
        self.session = cache_session(self.owner)

    def test_window_preserves_authentication_identity_and_independent_commits(self):
        path = self.root / "source.bin"
        identity = (1, 2, 3, 4, 5)
        with self.session.proof_io_window():
            self.assertTrue(self.session.remember_blob(path, "a" * 64, identity))
            self.assertEqual(self.session.lookup_blob(path, identity), "a" * 64)
            self.assertIsNone(self.session.lookup_blob(path, (1, 2, 3, 4, 6)))
            with self.owner._transaction() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM authenticated_blob_proofs").fetchone()[0], 1)
        self.assertIsNone(AuthenticatedCacheSession(self.owner).lookup_blob(path, identity))

    def test_tampered_proof_does_not_become_authority_while_connection_is_open(self):
        path = self.root / "source.bin"
        identity = (1, 2, 3, 4, 5)
        with self.session.proof_io_window():
            self.session.remember_blob(path, "a" * 64, identity)
            with self.owner._transaction(write=True) as connection:
                connection.execute("UPDATE authenticated_blob_proofs SET sha256=?", ("b" * 64,))
            self.assertIsNone(self.session.lookup_blob(path, identity))

    def test_exception_closes_window_before_database_publication(self):
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with self.session.proof_io_window():
                self.session.remember_blob(self.root / "source.bin", "a" * 64, (1, 2, 3, 4, 5))
                raise RuntimeError("fixture")
        # Last connection closing permits checkpoint/publication; no read
        # snapshot or helper handle remains after a rejected verification.
        self.assertFalse(Path(str(self.owner.path) + "-wal").exists())
        destination = self.root / "published.sqlite3"
        self.owner.path.rename(destination)
        self.assertTrue(destination.is_file())

    def test_refused_optional_window_leaves_original_verification_path_available(self):
        calls = []
        with patch.object(self.session, "reserve_disk", side_effect=RuntimeError("no window headroom")):
            with self.session.proof_io_window():
                calls.append("verified read still runs")
        self.assertEqual(calls, ["verified read still runs"])
        self.assertTrue(self.session.remember_blob(self.root / "source.bin", "a" * 64, (1, 2, 3, 4, 5)))


if __name__ == "__main__":
    unittest.main()
