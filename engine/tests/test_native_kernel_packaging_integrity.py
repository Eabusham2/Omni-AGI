"""New tiny build/loader contracts; no Torch, model, worker or training import."""
import ast
import copy
import ctypes
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("owned_build_native_integrity", ROOT / "scripts/native_kernel_integrity.py")
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)


def dylib(*, signed=False, relocated=False, cpu=0x100000C):
    def segment(name, fileoff, filesize, sections=b""):
        return struct.pack("<II16sQQQQiiII", 0x19, 72 + len(sections), name, 0, 4096,
            fileoff, filesize, 7, 5, len(sections) // 80, 0) + sections
    section = struct.pack("<16s16sQQIIIIIIII", b"__text", b"__TEXT", 0, 32, 512, 2, 0, 0, 0, 0, 0, 0)
    name = b"@rpath/omni_packed_kernel.dylib" if relocated else b"/compiler/omni_packed_kernel.dylib"
    size = (24 + len(name) + 1 + 7) // 8 * 8
    identity = struct.pack("<IIIIII", 0xD, size, 24, 0, 65536, 65536) + name + b"\0" * (size - 24 - len(name))
    signature_size = 48 if signed else 32
    commands = [segment(b"__TEXT", 0, 544, section), segment(b"__LINKEDIT", 544, 64 + signature_size), identity]
    if relocated:
        name = b"@loader_path/../.."
        size = (12 + len(name) + 1 + 7) // 8 * 8
        commands.append(struct.pack("<III", 0x8000001C, size, 12) + name + b"\0" * (size - 12 - len(name)))
    commands.append(struct.pack("<IIII", 0x1D, 16, 608, signature_size))
    body = b"".join(commands)
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cpu, 0, 6, len(commands), len(body), 0, 0)
    raw = bytearray(608 + signature_size)
    raw[:32], raw[32:32 + len(body)] = header, body
    raw[512:608] = bytes(range(96))
    raw[608:] = bytes([0xBB if signed else 0xAA]) * signature_size
    return bytes(raw)


class NativeKernelPackagingIntegrityCases(unittest.TestCase):
    def fixture(self):
        temporary = TemporaryDirectory(prefix="omni-native-packaging-contract-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        module = root / "omni_core"
        (module / "native").mkdir(parents=True)
        (module / "kernels").mkdir()
        source = module / "kernels/weighted_ternary.cpp"
        source.write_text("// inert compiler source fixture\n")
        library = module / "native/omni_packed_kernel.dylib"
        library.write_bytes(dylib())
        payload = I.macho_payload(library)
        manifest = {"abi": 1, "platform": "darwin", "architecture": "arm64", "file": library.name,
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "machoPayloadVersion": 1, "machoPayloadSha256": payload["sha256"]}
        (module / "native/manifest.json").write_text(json.dumps(manifest))
        return root, module, library, manifest

    def test_signature_and_allowed_relocation_keep_compiled_payload_but_reseal_exact_whole_file(self):
        root, module, library, manifest = self.fixture()
        self.assertEqual(I.checked_compiled_manifest(module), manifest)
        library.write_bytes(dylib(signed=True, relocated=True))
        self.assertNotEqual(hashlib.sha256(library.read_bytes()).hexdigest(), manifest["sha256"])
        self.assertEqual(I.macho_payload(library)["sha256"], manifest["machoPayloadSha256"])
        report = I.finalize_manifest(module, root, compiled_manifest=manifest)
        saved = json.loads((module / "native/manifest.json").read_text())
        self.assertTrue(report["compiledPayloadUnchanged"])
        self.assertEqual(saved["sha256"], hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertEqual({key: value for key, value in saved.items() if key != "sha256"},
            {key: value for key, value in manifest.items() if key != "sha256"})

    def test_machine_code_or_forged_package_payload_is_not_resealed(self):
        for forged in (False, True):
            with self.subTest(forged=forged):
                root, module, library, manifest = self.fixture()
                raw = bytearray(dylib(signed=True, relocated=True)); raw[520] ^= 1
                library.write_bytes(raw)
                manifest_path = module / "native/manifest.json"
                if forged:
                    manifest_path.write_text(json.dumps({**manifest, "machoPayloadSha256": I.macho_payload(library)["sha256"]}))
                before = manifest_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "payload|compiler-issued"):
                    I.finalize_manifest(module, root, compiled_manifest=manifest)
                self.assertEqual(manifest_path.read_bytes(), before)

    def test_changed_library_between_payload_proof_and_publication_is_not_resealed(self):
        root, module, library, manifest = self.fixture()
        before = (module / "native/manifest.json").read_bytes()
        original, calls, actual_library = Path.read_bytes, [0], library.resolve()
        def read(path):
            result = original(path)
            if path.resolve() == actual_library:
                calls[0] += 1
                if calls[0] == 2:
                    altered = bytearray(result); altered[521] ^= 1
                    return bytes(altered)
            return result
        with patch.object(Path, "read_bytes", read), self.assertRaisesRegex(ValueError, "changed during"):
            I.finalize_manifest(module, root, compiled_manifest=manifest)
        self.assertEqual((module / "native/manifest.json").read_bytes(), before)

    def test_source_ABI_architecture_and_compiler_manifest_custody_are_mandatory(self):
        for change in ("source", "abi", "architecture", "baseline"):
            with self.subTest(change=change):
                root, module, library, manifest = self.fixture()
                if change == "source": (module / "kernels/weighted_ternary.cpp").write_text("// changed source\n")
                elif change == "baseline":
                    library.write_bytes(dylib(signed=True))
                    with self.assertRaisesRegex(ValueError, "compiler-issued"):
                        I.checked_compiled_manifest(module)
                    continue
                else:
                    altered = {**manifest, change: 2 if change == "abi" else "x86_64"}
                    (module / "native/manifest.json").write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "custody|compiler-issued"):
                    I.finalize_manifest(module, root, compiled_manifest=manifest)

    def test_actual_runtime_loader_still_requires_final_exact_hash_and_abi_without_Torch(self):
        root, module, library, manifest = self.fixture()
        source = ast.parse((ROOT / "engine/omni_core/packed_native_kernel.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_library")
        class Function:
            restype = argtypes = None
            def __call__(self): return 1
        loaded = SimpleNamespace(**{name: Function() for name in ("omni_packed_kernel_abi", "omni_weighted_ternary_forward_i8",
            "omni_weighted_ternary_input_gradient_f32", "omni_weighted_ternary_forward_f32", "omni_weighted_ternary_forward_f64")})
        calls = []
        namespace = {"__file__": str(module / "packed_native_kernel.py"), "Path": Path, "json": json,
            "hashlib": hashlib, "sys": SimpleNamespace(platform="darwin"), "lru_cache": functools.lru_cache,
            "ctypes": SimpleNamespace(CDLL=lambda path: calls.append(path) or loaded,
                c_int=ctypes.c_int, c_void_p=ctypes.c_void_p, c_int64=ctypes.c_int64)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual-native-loader", "exec"), namespace)
        library.write_bytes(dylib(signed=True, relocated=True))
        with self.assertRaisesRegex(RuntimeError, "integrity verification failed"): namespace["_library"]()
        self.assertEqual(calls, [])
        I.finalize_manifest(module, root, compiled_manifest=manifest)
        self.assertIs(namespace["_library"](), loaded)
        namespace["_library"].cache_clear()
        raw = bytearray(library.read_bytes()); raw[521] ^= 1; library.write_bytes(raw)
        with self.assertRaisesRegex(RuntimeError, "integrity verification failed"): namespace["_library"]()
        self.assertEqual(len(calls), 1)

    def test_unexpected_rpaths_headers_and_foreign_file_roots_are_refused(self):
        root, module, library, manifest = self.fixture()
        for raw in (b"short", dylib(cpu=1), dylib(relocated=True).replace(b"@loader_path/../..", b"@loader_path/evil!")):
            library.write_bytes(raw)
            with self.assertRaises(ValueError): I.macho_payload(library)
        library.write_bytes(dylib())
        with self.assertRaises(ValueError): I.finalize_manifest(module, root / "foreign", compiled_manifest=manifest)
        if os.name != "nt":
            outside = root.parent / (root.name + "-outside")
            outside.write_bytes(dylib())
            self.addCleanup(lambda: outside.unlink(missing_ok=True))
            library.unlink(); library.symlink_to(outside)
            with self.assertRaises(ValueError): I.finalize_manifest(module, root, compiled_manifest=manifest)

    def test_build_seals_after_vendor_processing_and_never_weakens_runtime_packed_requirement(self):
        build = (ROOT / "scripts/build-engine-posix.sh").read_text()
        self.assertLess(build.index('-m PyInstaller'), build.index('finalize-native-kernel-manifest.py'))
        self.assertLess(build.index('finalize-native-kernel-manifest.py'), build.index('verify-packaged-compliance.mjs'))
        compiler = (ROOT / "scripts/build-native-kernels.py").read_text()
        self.assertIn("machoPayloadSha256=payload[\"sha256\"]", compiler)
        loader = (ROOT / "engine/omni_core/packed_native_kernel.py").read_text()
        self.assertIn('manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()', loader)
        worker = (ROOT / "engine/worker.py").read_text()
        self.assertIn('"ready": packed_cpu', worker)
        self.assertIn("packed_cpu = native_packed_kernel_available()", worker)


if __name__ == "__main__": unittest.main()
