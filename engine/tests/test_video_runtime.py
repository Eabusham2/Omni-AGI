"""Pure file/header/RPC-boundary tests: no torch, worker, model or executable run."""

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_runtime import (
    VideoRuntimeConfigurationCancelled,
    configure_verified_video_runtime,
)


def native_bytes(target):
    data = bytearray(128)
    if target.startswith("linux-"):
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", data, 18, 62 if target.endswith("-x64") else 183)
    elif target.startswith("darwin-"):
        struct.pack_into("<II", data, 0, 0xFEEDFACF, 0x01000007 if target.endswith("-x64") else 0x0100000C)
    else:
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 60, 80)
        data[80:84] = b"PE\x00\x00"
        struct.pack_into("<H", data, 84, 0x8664 if target.endswith("-x64") else 0xAA64)
    return bytes(data)


class VideoRuntimeConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-video-runtime-rpc-test-")
        self.root = Path(self.temporary.name).resolve()
        self.environment = {"OMNI_VIDEO_RUNTIME_CACHE_ROOT": str(self.root)}

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, target="linux-x64"):
        artifact_hash = "a" * 64
        directory = self.root / ("fixture-" + target + "-" + artifact_hash)
        directory.mkdir()
        path = directory / ("ffmpeg.exe" if target.startswith("win32-") else "ffmpeg")
        payload = native_bytes(target)
        path.write_bytes(payload)
        path.chmod(0o700)
        binary_hash = hashlib.sha256(payload).hexdigest()
        (directory / "receipt.json").write_text(json.dumps({
            "schemaVersion": 1, "artifactSha256": artifact_hash,
            "artifact": {"target": target, "binary": {"sha256": binary_hash, "sizeBytes": len(payload)}}
        }), encoding="utf-8")
        return {
            "executablePath": str(path), "artifactSha256": artifact_hash,
            "binarySha256": binary_hash, "binarySizeBytes": len(payload), "target": target
        }

    def configure(self, params, **kwargs):
        return configure_verified_video_runtime(params, environment=self.environment, platform_name=params.get("target", "linux-x64").split("-", 1)[0], **kwargs)

    def test_selects_all_six_native_targets_without_executing_them(self):
        for target in ("win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "linux-x64", "linux-arm64"):
            with self.subTest(target=target):
                params = self.fixture(target)
                self.assertTrue(self.configure(params)["configured"])
                self.assertEqual(self.environment["IMAGEIO_FFMPEG_EXE"], params["executablePath"])

    def test_rejects_missing_fixed_cache_root_or_extra_env_commands(self):
        params = self.fixture()
        with self.assertRaises(ValueError):
            configure_verified_video_runtime(params, environment={}, platform_name="linux")
        params["installCommand"] = "sh arbitrary.sh"
        with self.assertRaises(ValueError):
            self.configure(params)
        self.assertNotIn("IMAGEIO_FFMPEG_EXE", self.environment)

    def test_rejects_external_paths_and_mismatched_artifact_directories(self):
        params = self.fixture()
        params["artifactSha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.configure(params)
        params["executablePath"] = str(self.root.parent / "ffmpeg")
        with self.assertRaises((ValueError, OSError)):
            self.configure(params)

    def test_rejects_changed_hash_wrong_size_and_changed_receipt(self):
        params = self.fixture()
        path = Path(params["executablePath"])
        path.write_bytes(b"x" * params["binarySizeBytes"])
        with self.assertRaises(ValueError):
            self.configure(params)
        path.write_bytes(native_bytes("linux-x64"))
        params["binarySizeBytes"] += 1
        with self.assertRaises(ValueError):
            self.configure(params)
        params["binarySizeBytes"] -= 1
        (path.parent / "receipt.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.configure(params)
        self.assertNotIn("IMAGEIO_FFMPEG_EXE", self.environment)

    def test_rejects_a_script_even_when_its_fixture_hash_matches(self):
        params = self.fixture()
        path = Path(params["executablePath"])
        payload = b"#!/bin/sh\necho no\n" + b"x" * 64
        path.write_bytes(payload)
        params["binarySha256"] = hashlib.sha256(payload).hexdigest()
        params["binarySizeBytes"] = len(payload)
        (path.parent / "receipt.json").write_text(json.dumps({
            "schemaVersion": 1, "artifactSha256": params["artifactSha256"],
            "artifact": {"target": params["target"], "binary": {"sha256": params["binarySha256"], "sizeBytes": len(payload)}}
        }), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.configure(params)

    @unittest.skipIf(os.name == "nt", "Windows symlink creation may need privilege")
    def test_rejects_executable_and_receipt_symlinks(self):
        params = self.fixture()
        path = Path(params["executablePath"])
        moved = path.parent / "renamed-native"
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaises(ValueError):
            self.configure(params)
        path.unlink()
        moved.rename(path)
        receipt = path.parent / "receipt.json"
        saved = path.parent / "saved-receipt.json"
        receipt.rename(saved)
        receipt.symlink_to(saved)
        with self.assertRaises(ValueError):
            self.configure(params)

    def test_cooperative_stop_preserves_existing_selection(self):
        params = self.fixture()
        self.environment["IMAGEIO_FFMPEG_EXE"] = "/fixture/prior-selection"
        with self.assertRaises(VideoRuntimeConfigurationCancelled):
            self.configure(params, cancelled=lambda: True)
        self.assertEqual(self.environment["IMAGEIO_FFMPEG_EXE"], "/fixture/prior-selection")

    def test_rejects_cross_os_target_before_setting_environment(self):
        params = self.fixture("linux-x64")
        with self.assertRaises(ValueError):
            configure_verified_video_runtime(params, environment=self.environment, platform_name="darwin")
        self.assertNotIn("IMAGEIO_FFMPEG_EXE", self.environment)


if __name__ == "__main__":
    unittest.main()
