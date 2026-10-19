"""Build the small portable packed kernel; no Torch headers or model involved."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def runtime_architecture():
    # An x64 Python process on Windows ARM64 must load an x64 DLL. The
    # operating-system machine name can describe the ARM64 host instead.
    if sys.platform == "win32":
        value = sysconfig.get_platform().lower()
        if value == "win-amd64":
            return "x86_64"
        if value == "win-arm64":
            return "arm64"
        raise RuntimeError("The native worker requires 64-bit x64 or ARM64 Python.")
    value = platform.machine().lower()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError("The native worker requires an x64 or ARM64 runtime.")


def windows_developer_command(command_processor, target):
    if target not in {"x64", "arm64"}:
        raise ValueError("Unsupported native Windows compiler target.")
    # cmd.exe parses the /c remainder itself. Passing this program as an argv
    # item applies CRT backslash-quote escaping, which cmd does not understand.
    # /s strips only these outer quotes and preserves the quoted batch path.
    return (subprocess.list2cmdline([command_processor]) + ' /d /u /s /c "call '
        '"%OMNI_KERNEL_VSDEVCMD%" -no_logo -arch=' + target + ' >nul && set"')


def windows_msvc_environment(architecture):
    target = "x64" if architecture == "x86_64" else "arm64"
    environment = {name.upper(): value for name, value in os.environ.items()}
    if environment.get("VSCMD_ARG_TGT_ARCH", "").lower() == target and shutil.which("cl", path=environment.get("PATH")):
        return environment
    program_files = environment.get("PROGRAMFILES(X86)", environment.get("PROGRAMFILES", ""))
    installed = Path(program_files) / "Microsoft Visual Studio/Installer/vswhere.exe"
    locator = str(installed) if installed.is_file() else shutil.which("vswhere")
    if not locator:
        raise RuntimeError("The Windows worker build needs Visual Studio C++ Build Tools (vswhere and VsDevCmd.bat).")
    component = "Microsoft.VisualStudio.Component.VC.Tools." + ("x86.x64" if target == "x64" else "ARM64")
    discovery = subprocess.run([locator, "-latest", "-products", "*", "-requires", component, "-property", "installationPath", "-utf8"],
        check=True, capture_output=True, encoding="utf-8", timeout=30)
    installation = discovery.stdout.strip()
    developer_command = Path(installation) / "Common7/Tools/VsDevCmd.bat"
    if not installation or not developer_command.is_file():
        raise RuntimeError("Visual Studio C++ Build Tools did not provide VsDevCmd.bat.")
    environment["OMNI_KERNEL_VSDEVCMD"] = str(developer_command)
    environment["VSCMD_SKIP_SENDTELEMETRY"] = "1"
    # Capture the developer environment privately; never print its contents,
    # which also contain the caller's signing variables. Select the Python
    # process target explicitly instead of trusting the runner's default PATH.
    command_processor = environment.get("COMSPEC", "cmd.exe")
    try:
        initialized = subprocess.run(windows_developer_command(command_processor, target), executable=command_processor,
            env=environment, check=True, capture_output=True, encoding="utf-16-le", errors="replace", timeout=60)
    except subprocess.CalledProcessError as error:
        # VsDevCmd's captured environment can contain signing credentials.
        # Keep diagnostics stage/target-specific without printing its streams.
        raise RuntimeError("Visual Studio developer environment initialization failed for " + target
            + " (exit " + str(error.returncode) + "); check the requested C++ toolchain installation.") from None
    environment = {}
    for line in initialized.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            environment[name.upper()] = value
    if environment.get("VSCMD_ARG_TGT_ARCH", "").lower() != target or not shutil.which("cl", path=environment.get("PATH")):
        raise RuntimeError("Visual Studio did not initialize the requested Python worker architecture.")
    environment.pop("OMNI_KERNEL_VSDEVCMD", None)
    return environment


def verify_library(path, environment):
    # A fresh stdlib-only process checks loader dependencies, process bitness,
    # ABI and exports, then exits so Windows releases the DLL before staging.
    # No model, neural forward, tensor library or training is involved.
    probe = (
        "import ctypes,sys; library=ctypes.CDLL(sys.argv[1]); "
        "library.omni_packed_kernel_abi.restype=ctypes.c_int; "
        "assert library.omni_packed_kernel_abi()==1, 'packed kernel ABI mismatch'; "
        "[getattr(library,name) for name in "
        "('omni_weighted_ternary_forward_i8','omni_weighted_ternary_input_gradient_f32',"
        "'omni_weighted_ternary_forward_f32','omni_weighted_ternary_forward_f64')]"
    )
    subprocess.run([sys.executable, "-I", "-c", probe, str(path)], env=environment, check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "engine/omni_core/kernels/weighted_ternary.cpp"
    destination = args.output_directory or root / "engine/omni_core/native"
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    name = "omni_packed_kernel" + suffix
    architecture = runtime_architecture()
    explicit_compiler = os.environ.get("CXX")
    compiler_environment = dict(os.environ)
    if sys.platform == "win32" and (not explicit_compiler or Path(explicit_compiler).name.lower() in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}):
        compiler_environment = windows_msvc_environment(architecture)
        compiler = explicit_compiler or shutil.which("cl", path=compiler_environment.get("PATH"))
    else:
        compiler = explicit_compiler or next((shutil.which(item) for item in ("clang++", "g++") if shutil.which(item)), None)
    if not compiler:
        raise RuntimeError("A C++17 compiler is required when building the native worker package.")
    with tempfile.TemporaryDirectory(prefix="omni-kernel-build-") as temporary:
        output = Path(temporary) / name
        if Path(compiler).name.lower() in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}:
            command = [compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", "/MT", "/LD", str(source), "/link", "/OUT:" + str(output)]
            if sys.platform == "win32":
                command.append("/MACHINE:" + ("X64" if architecture == "x86_64" else "ARM64"))
        else:
            command = [compiler, "-O3", "-std=c++17", "-shared", str(source), "-o", str(output)]
            if sys.platform != "win32":
                command.insert(1, "-fPIC")
        subprocess.run(command, cwd=temporary, env=compiler_environment, check=True, timeout=120)
        verify_library(output, compiler_environment)
        staged = destination / (name + ".next")
        shutil.copy2(output, staged)
        staged.replace(destination / name)
    manifest = {"abi": 1, "platform": sys.platform, "architecture": architecture,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(), "file": name}
    if sys.platform == "darwin":
        spec = importlib.util.spec_from_file_location("omni_build_native_integrity", Path(__file__).with_name("native_kernel_integrity.py"))
        integrity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integrity)
        payload = integrity.macho_payload(destination / name)
        if payload["architecture"] != architecture:
            raise ValueError("native packed kernel compiled for the wrong architecture")
        manifest.update(machoPayloadVersion=1, machoPayloadSha256=payload["sha256"])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
