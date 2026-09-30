"""Pure lock/parser fixtures; never install dependencies or build a worker."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
INSTALLER_PATH = REPOSITORY / "scripts" / "install-engine-lock.py"
SPEC = importlib.util.spec_from_file_location("omni_engine_lock_fixture", INSTALLER_PATH)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class EngineSchemaLockContractFixtures(unittest.TestCase):
    def test_all_reviewed_schema_pins_and_wheel_hashes_match_every_target(self):
        INSTALLER.verify_all_locks()
        self.assertEqual(sum(len(values) for values in INSTALLER.TARGET_PACKAGES.values()), 159)
        for target in INSTALLER.TARGET_PACKAGES:
            with self.subTest(target=target):
                pins = INSTALLER.parse_lock(INSTALLER.lock_path(target))
                for name in (*INSTALLER.SCHEMA_UNIVERSAL_WHEEL_HASHES, "rpds-py"):
                    self.assertEqual(pins[name], INSTALLER.COMMON_PINS[name])

    def test_substituted_universal_schema_hash_is_rejected_even_if_well_formed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macos-arm64-py311.lock"
            source = INSTALLER.lock_path("macos-arm64").read_text(encoding="utf8")
            path.write_text(source.replace(INSTALLER.SCHEMA_UNIVERSAL_WHEEL_HASHES["jsonschema"], "0" * 64), encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "reviewed macos-arm64 schema wheel for jsonschema"):
                INSTALLER.parse_lock(path)

    def test_other_architecture_rpds_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows-x86_64-py311.lock"
            source = INSTALLER.lock_path("windows-x86_64").read_text(encoding="utf8")
            path.write_text(source.replace(
                INSTALLER.SCHEMA_RPDS_WHEEL_HASHES["windows-x86_64"],
                INSTALLER.SCHEMA_RPDS_WHEEL_HASHES["macos-arm64"],
            ), encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "reviewed windows-x86_64 schema wheel for rpds-py"):
                INSTALLER.parse_lock(path)

    def test_unreviewed_optional_format_dependency_is_not_part_of_worker_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linux-x86_64-py311.lock"
            source = INSTALLER.lock_path("linux-x86_64").read_text(encoding="utf8")
            path.write_text(source + "fqdn==1.5.1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "package closure changed"):
                INSTALLER.parse_lock(path)


if __name__ == "__main__":
    unittest.main()
