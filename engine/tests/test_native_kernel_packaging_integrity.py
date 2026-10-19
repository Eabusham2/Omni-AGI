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
from unittest.mock import Mock, patch

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
    def native_builder(self):
        spec = importlib.util.spec_from_file_location("owned_native_kernel_builder", ROOT / "scripts/build-native-kernels.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_windows_developer_setup_preserves_cmd_quotes_and_selects_actual_target_tools_without_building(self):
        builder = self.native_builder()
        for architecture, target, component in (("x86_64", "x64", "x86.x64"), ("arm64", "arm64", "ARM64")):
            with self.subTest(target=target):
                calls = []
                processor = r"C:\Windows\System32\cmd.exe"
                environment = {"ComSpec": processor, "ProgramFiles(x86)": r"C:\Program Files (x86)", "Path": "original"}
                def run(command, **options):
                    calls.append((command, options))
                    if isinstance(command, list):
                        self.assertIn("Microsoft.VisualStudio.Component.VC.Tools." + component, command)
                        self.assertIn("-requires", command)
                        return SimpleNamespace(stdout=r"C:\Program Files\Microsoft Visual Studio\18\Enterprise" + "\n")
                    # Model cmd's documented /s outer-quote removal, not CRT
                    # argv decoding. No actual cmd/compiler/device is started.
                    self.assertIsInstance(command, str)
                    shell_program = command.split(" /s /c ", 1)[1]
                    self.assertEqual(shell_program[1:-1], 'call "%OMNI_KERNEL_VSDEVCMD%" -no_logo -arch=' + target + ' >nul && set')
                    self.assertTrue(shell_program.startswith('"') and shell_program.endswith('"'))
                    self.assertNotIn('\\"', shell_program)
                    self.assertEqual(options["executable"], processor)
                    self.assertEqual(options["encoding"], "utf-16-le")
                    self.assertTrue(options["capture_output"] and options["check"])
                    self.assertEqual(options["timeout"], 60)
                    return SimpleNamespace(stdout="Path=owned-msvc-path\r\nVSCMD_ARG_TGT_ARCH=" + target
                        + "\r\nOMNI_KERNEL_VSDEVCMD=owned-private-helper\r\n")
                with patch.dict(builder.os.environ, environment, clear=True), patch.object(builder.Path, "is_file", return_value=True), \
                     patch.object(builder.shutil, "which", side_effect=lambda name, path=None: "owned-cl.exe" if name == "cl" and path == "owned-msvc-path" else None), \
                     patch.object(builder.subprocess, "run", side_effect=run):
                    configured = builder.windows_msvc_environment(architecture)
                self.assertEqual(len(calls), 2)
                self.assertEqual(configured["VSCMD_ARG_TGT_ARCH"], target)
                self.assertEqual(configured["PATH"], "owned-msvc-path")
                self.assertNotIn("OMNI_KERNEL_VSDEVCMD", configured)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            builder.windows_developer_command("cmd.exe", "unreviewed-target")

    def test_windows_failed_developer_setup_does_not_publish_private_environment_or_continue_to_compile(self):
        builder = self.native_builder()
        discovery = SimpleNamespace(stdout="owned-installed-visual-studio\n")
        failure = builder.subprocess.CalledProcessError(1, "owned-cmd", output="PRIVATE_SIGNING=OWNED_TEST_SECRET", stderr="OWNED_TEST_SECRET")
        with patch.dict(builder.os.environ, {"ComSpec": "cmd.exe", "Path": "original"}, clear=True), \
             patch.object(builder.Path, "is_file", return_value=True), patch.object(builder.shutil, "which", return_value=None), \
             patch.object(builder.subprocess, "run", side_effect=[discovery, failure]) as run:
            with self.assertRaisesRegex(RuntimeError, "initialization failed for x64.*exit 1") as raised:
                builder.windows_msvc_environment("x86_64")
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("OWNED_TEST_SECRET", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_windows_owned_kernel_cleanup_retries_only_transient_locks_after_child_verification(self):
        builder = self.native_builder()
        locked = PermissionError("owned DLL is temporarily locked"); locked.winerror = 5
        cleanup = Mock(side_effect=[locked, locked, None])
        temporary = SimpleNamespace(cleanup=cleanup)
        with patch.object(builder, "sys", SimpleNamespace(platform="win32")), patch.object(builder.time, "sleep") as sleep:
            builder.cleanup_owned_kernel_directory(temporary)
        self.assertEqual(cleanup.call_count, 3)
        self.assertEqual([row.args[0] for row in sleep.call_args_list], [0.1, 0.2])
        source = (ROOT / "scripts/build-native-kernels.py").read_text()
        body = ast.parse(source)
        main = next(node for node in body.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        guarded = next(node for node in main.body if isinstance(node, ast.Try))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "verify_library" for node in ast.walk(guarded)))
        self.assertEqual(guarded.finalbody[0].value.func.id, "cleanup_owned_kernel_directory")

    def test_owned_kernel_cleanup_expiry_and_non_windows_permission_failures_remain_fatal(self):
        builder = self.native_builder()
        for platform_name, code, attempts in (("win32", 5, 21), ("win32", 32, 21), ("win32", 999, 1), ("darwin", 5, 1)):
            with self.subTest(platform=platform_name, code=code):
                locked = PermissionError("owned lock must not be ignored"); locked.winerror = code
                cleanup = Mock(side_effect=locked)
                with patch.object(builder, "sys", SimpleNamespace(platform=platform_name)), patch.object(builder.time, "sleep") as sleep:
                    with self.assertRaises(PermissionError) as raised:
                        builder.cleanup_owned_kernel_directory(SimpleNamespace(cleanup=cleanup))
                self.assertIs(raised.exception, locked)
                self.assertEqual(cleanup.call_count, attempts)
                self.assertEqual(sleep.call_count, attempts - 1)
                self.assertLessEqual(sum(row.args[0] for row in sleep.call_args_list), 10)

    def test_windows_worker_runtime_fault_hook_is_early_stdlib_only_and_diagnostic(self):
        hook = ROOT / "scripts/pyinstaller-worker-fault-hook.py"
        source = hook.read_text()
        body = ast.parse(source)
        self.assertEqual([node.names[0].name for node in body.body if isinstance(node, ast.Import)], ["faulthandler", "sys"])
        self.assertFalse(any(isinstance(node, ast.ImportFrom) for node in ast.walk(body)))
        for stderr, failure in ((object(), None), (None, None), (object(), OSError("owned closed stderr"))):
            with self.subTest(stderr_present=stderr is not None, diagnostic_failure=failure is not None):
                enable = Mock(side_effect=failure)
                def owned_import(name, *args, **kwargs):
                    if name == "faulthandler": return SimpleNamespace(enable=enable)
                    if name == "sys": return SimpleNamespace(stderr=stderr)
                    self.fail("Unexpected worker hook dependency: " + name)
                exec(compile(body, str(hook), "exec"), {"__builtins__": {"__import__": owned_import,
                    "AttributeError": AttributeError, "OSError": OSError, "RuntimeError": RuntimeError, "ValueError": ValueError}})
                if stderr is None: enable.assert_not_called()
                else: enable.assert_called_once_with(file=stderr, all_threads=False)
        build = (ROOT / "scripts/build-engine.ps1").read_text()
        runtime_hook = '--runtime-hook (Join-Path $RepoRoot "scripts\\pyinstaller-worker-fault-hook.py")'
        self.assertLess(build.index("-m PyInstaller"), build.index(runtime_hook))
        self.assertLess(build.index(runtime_hook), build.index("--paths"))

    def test_windows_worker_closed_stdout_diagnostic_is_owned_bounded_and_preserves_failure(self):
        smoke = (ROOT / "scripts/smoke-engine.ps1").read_text()
        self.assertLess(smoke.index("$Worker.StandardError.ReadToEndAsync()"), smoke.index("function Invoke-WorkerRpc"))
        diagnostic = smoke.split("function Get-WorkerExitDiagnostic {", 1)[1].split("function Invoke-WorkerRpc", 1)[0]
        self.assertIn("$Worker.HasExited", diagnostic)
        self.assertIn("$Worker.WaitForExit(5000)", diagnostic)
        self.assertIn("$WorkerErrorReadTask.Wait(1000)", diagnostic)
        self.assertIn(".Substring($DiagnosticError.Length - 12000)", diagnostic)
        self.assertIn("[string]$Worker.ExitCode", diagnostic)
        self.assertIn("[Convert]::ToString([int]$Worker.ExitCode, 16)", diagnostic)
        self.assertIn(".PadLeft(8, '0')", diagnostic)
        for field in ("workerExited=", "exitCode=", "exitHex=", "stderrComplete="):
            self.assertIn(field, diagnostic)
        self.assertNotIn("Environment", diagnostic)
        self.assertNotIn("Get-Process", diagnostic)
        self.assertNotIn("Kill", diagnostic)
        self.assertIn('throw "Packaged worker closed stdout during $Method. $(Get-WorkerExitDiagnostic)"', smoke)
        self.assertIn('throw "Packaged worker did not exit cleanly. $(Get-WorkerExitDiagnostic)"', smoke)
        self.assertIn('$Health.worker -ne "python"', smoke)
        self.assertIn('$Health.protocolVersion -ne 1', smoke)

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
