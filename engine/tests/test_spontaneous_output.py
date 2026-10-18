import tempfile
import unittest
from pathlib import Path
from spontaneous_output import SpontaneousOutputLeases, checked_output_directory


class SpontaneousOutputOwnershipTests(unittest.TestCase):
    def test_fresh_single_use_job_owner_never_changes_existing_user_outputs(self):
        with tempfile.TemporaryDirectory(prefix="omni-spontaneous-owner-") as root:
            engine = Path(root) / "engine"
            (engine / "artifacts").mkdir(parents=True)
            original = engine / "artifacts" / "old-user.png"
            original.write_bytes(b"old user bytes")
            leases = SpontaneousOutputLeases()
            ack = leases.authorize(engine, "brain", "job", {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "action"})
            with self.assertRaises(ValueError):
                leases.consume("other", "job", ack["leaseId"])
            directory = leases.consume("brain", "job", ack["leaseId"])
            self.assertEqual(checked_output_directory(engine, directory), directory)
            self.assertEqual(checked_output_directory(engine), engine / "artifacts")
            with self.assertRaises(ValueError):
                leases.consume("brain", "job", ack["leaseId"])
            (directory / ("a" * 64 + ".png")).write_bytes(b"only newly owned synthetic output")
            self.assertTrue(leases.discard("brain", "job", ack["leaseId"])["discarded"])
            self.assertEqual(original.read_bytes(), b"old user bytes")

    def test_no_renderer_flag_keyword_or_unclassified_source_can_authorize_scratch(self):
        with tempfile.TemporaryDirectory(prefix="omni-spontaneous-refusal-") as root:
            engine = Path(root)
            for origin in [{"kind": "spontaneous"}, {"kind": "user-requested", "authority": "main-ui", "requestId": "request"}, {"kind": "unclassified", "authority": "native-brain-action"}, {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "action", "prompt": "spontaneous"}]:
                with self.assertRaises(ValueError):
                    SpontaneousOutputLeases().authorize(engine, "brain", "job", origin)

    def test_unsafe_scratch_and_unknown_cleanup_payload_are_preserved_not_recursively_deleted(self):
        with tempfile.TemporaryDirectory(prefix="omni-spontaneous-safe-") as root:
            engine = Path(root)
            leases = SpontaneousOutputLeases()
            ack = leases.authorize(engine, "brain", "job", {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "action"})
            directory = leases.consume("brain", "job", ack["leaseId"])
            unexpected = directory / "user-file.txt"
            unexpected.write_text("preserve")
            with self.assertRaises(ValueError):
                leases.discard("brain", "job", ack["leaseId"])
            self.assertTrue(unexpected.exists())
            with self.assertRaises(ValueError):
                checked_output_directory(engine, engine / "artifacts")

    def test_unload_cleanup_only_removes_acknowledged_job_scratch_and_reports_unknown_evidence(self):
        with tempfile.TemporaryDirectory(prefix="omni-spontaneous-unload-") as root:
            engine = Path(root)
            leases = SpontaneousOutputLeases()
            one = leases.authorize(engine, "brain", "one", {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "a"})
            two = leases.authorize(engine, "brain", "two", {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "b"})
            first = leases.consume("brain", "one", one["leaseId"])
            second = leases.consume("brain", "two", two["leaseId"])
            (first / ("d" * 64 + ".png")).write_bytes(b"synthetic ephemeral output")
            (second / "unexpected.txt").write_text("preserve")
            receipt = leases.discard_brain("brain")
            self.assertFalse(first.exists())
            self.assertTrue((second / "unexpected.txt").exists())
            self.assertEqual(receipt, {"complete": False, "preservedUnverifiedJobIds": ["two"]})

    def test_modified_owner_marker_cannot_authorize_writes_or_remove_evidence(self):
        with tempfile.TemporaryDirectory(prefix="omni-spontaneous-marker-") as root:
            engine = Path(root)
            leases = SpontaneousOutputLeases()
            ack = leases.authorize(engine, "brain", "job", {"kind": "spontaneous", "authority": "main-organic-action", "actionEventId": "a"})
            directory = engine / ".spontaneous-imagination" / ack["leaseId"]
            (directory / "owner.json").write_text('{}')
            with self.assertRaises(ValueError):
                leases.consume("brain", "job", ack["leaseId"])
            self.assertTrue((directory / "owner.json").exists())


if __name__ == "__main__":
    unittest.main()
