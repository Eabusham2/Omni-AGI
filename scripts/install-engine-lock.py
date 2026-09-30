#!/usr/bin/env python3
"""Install or verify the reviewed, wheel-only Omni engine release lock."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_DIRECTORY = REPOSITORY / "engine" / "locks"
LOCK_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9+.!_-]*) "
    r"--hash=sha256:(?P<hash>[0-9a-f]{64})$"
)
INPUT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9+.!_-]*"
    r"(?:; .+)?$"
)

AUDITED_PACKAGES = {
    "attrs", "jsonschema", "jsonschema-specifications", "referencing", "rpds-py",
    "altgraph",
    "cffi",
    "filelock",
    "fsspec",
    "ijson",
    "imageio",
    "imageio-ffmpeg",
    "jinja2",
    "macholib",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "pefile",
    "pillow",
    "psutil",
    "pyarrow",
    "pycparser",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pypdf",
    "pywin32-ctypes",
    "safetensors",
    "setuptools",
    "soundfile",
    "sympy",
    "torch",
    "typing-extensions",
}

TARGET_PINS = {
    "linux-aarch64": {
        "torch": "2.10.0+cpu",
        "numpy": "2.2.6",
        "pyarrow": "22.0.0",
    },
    "linux-x86_64": {
        "torch": "2.10.0+cpu",
        "numpy": "2.2.6",
        "pyarrow": "22.0.0",
    },
    "macos-arm64": {
        "torch": "2.10.0",
        "numpy": "2.2.6",
        "pyarrow": "22.0.0",
    },
    "macos-x86_64": {
        "torch": "2.2.2",
        "numpy": "1.26.4",
        "pyarrow": "17.0.0",
    },
    # The Windows ARM64 Electron shell intentionally packages this x64 worker
    # under emulation until the whole binary dependency closure ships ARM64 wheels.
    "windows-x86_64": {
        "torch": "2.10.0+cpu",
        "numpy": "2.2.6",
        "pyarrow": "22.0.0",
    },
}

# setup-python's reviewed manifest supplies newer security-only CPython archives
# for Linux, while 3.11.9 is its final macOS/Windows installer build.
TARGET_PYTHON_VERSIONS = {
    "linux-aarch64": (3, 11, 16),
    "linux-x86_64": (3, 11, 16),
    "macos-arm64": (3, 11, 9),
    "macos-x86_64": (3, 11, 9),
    "windows-x86_64": (3, 11, 9),
}

COMMON_PINS = {
    "attrs": "26.1.0",
    "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "0.30.0",
    "altgraph": "0.17.5",
    "cffi": "2.1.1",
    "filelock": "3.29.0",
    "fsspec": "2026.7.0",
    "ijson": "3.4.0.post0",
    "imageio": "2.37.2",
    "imageio-ffmpeg": "0.6.0",
    "jinja2": "3.1.6",
    "macholib": "1.16.4",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.4.2",
    "packaging": "26.3",
    "pefile": "2024.8.26",
    "pillow": "12.3.0",
    "psutil": "7.2.2",
    "pycparser": "3.0",
    "pyinstaller": "6.21.0",
    "pyinstaller-hooks-contrib": "2026.6",
    "pypdf": "6.6.2",
    "pywin32-ctypes": "0.2.3",
    "safetensors": "0.8.0",
    "setuptools": "84.0.0",
    "soundfile": "0.13.1",
    "sympy": "1.14.0",
    "typing-extensions": "4.15.0",
}

# Reviewed against each exact release's official PyPI JSON metadata. All four
# pure packages select their py3-none-any wheel; rpds-py selects the target's
# cp311-cp311 compiled wheel, not an sdist or another architecture's binary.
# This offline check prevents a well-formed but substituted lock hash from
# silently changing the native schema-validator dependency closure.
SCHEMA_UNIVERSAL_WHEEL_HASHES = {
    "attrs": "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309",
    "jsonschema": "d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce",
    "jsonschema-specifications": "98802fee3a11ee76ecaca44429fda8a41bff98b00a0f2838151b113f210cc6fe",
    "referencing": "381329a9f99628c9069361716891d34ad94af76e461dcb0335825aecc7692231",
}
SCHEMA_RPDS_WHEEL_HASHES = {
    "linux-aarch64": "422c3cb9856d80b09d30d2eb255d0754b23e090034e1deb4083f8004bd0761e4",
    "linux-x86_64": "33f559f3104504506a44bb666b93a33f5d33133765b0c216a5bf2f1e1503af89",
    "macos-arm64": "dc4f992dfe1e2bc3ebc7444f6c7051b4bc13cd8e33e43511e8ffd13bf407010d",
    "macos-x86_64": "a2bffea6a4ca9f01b3f8e548302470306689684e61602aa3d141e34da06cf425",
    "windows-x86_64": "a51033ff701fca756439d641c0ad09a41d9242fa69121c7d8769604a0a629825",
}

BASE_PACKAGES = {
    "attrs", "jsonschema", "jsonschema-specifications", "referencing", "rpds-py",
    "altgraph",
    "cffi",
    "filelock",
    "fsspec",
    "ijson",
    "imageio",
    "imageio-ffmpeg",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "pillow",
    "psutil",
    "pyarrow",
    "pycparser",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pypdf",
    "safetensors",
    "setuptools",
    "soundfile",
    "sympy",
    "torch",
    "typing-extensions",
}
TARGET_PACKAGES = {
    "linux-aarch64": BASE_PACKAGES,
    "linux-x86_64": BASE_PACKAGES,
    "macos-arm64": BASE_PACKAGES | {"macholib"},
    "macos-x86_64": BASE_PACKAGES | {"macholib"},
    "windows-x86_64": BASE_PACKAGES | {"pefile", "pywin32-ctypes"},
}

RUNTIME_INPUT_PACKAGES = {
    "attrs", "jsonschema", "jsonschema-specifications", "referencing", "rpds-py",
    "ijson",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "pillow",
    "psutil",
    "pyarrow",
    "pypdf",
    "safetensors",
    "soundfile",
    "torch",
}
BUILD_INPUT_PACKAGES = {"pyinstaller"}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def current_target() -> str:
    system = sys.platform
    machine = platform.machine().lower()
    if machine in {"amd64", "x64", "x86-64"}:
        machine = "x86_64"
    elif machine == "arm64" and system.startswith("linux"):
        machine = "aarch64"

    if system == "darwin" and machine in {"arm64", "x86_64"}:
        return f"macos-{machine}"
    if system.startswith("linux") and machine in {"aarch64", "x86_64"}:
        return f"linux-{machine}"
    if system == "win32" and machine == "x86_64":
        return "windows-x86_64"
    if system == "win32" and machine in {"arm64", "aarch64"}:
        raise RuntimeError(
            "Native Windows ARM64 is not a locked worker target; use the release "
            "matrix's x64 CPython worker under Windows emulation."
        )
    raise RuntimeError(f"Unsupported engine lock target: {system}/{machine}")


def lock_path(target: str) -> Path:
    if target not in TARGET_PINS:
        raise RuntimeError(f"Unknown engine lock target: {target}")
    return LOCK_DIRECTORY / f"{target}-py311.lock"


def parse_lock(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"Engine lock is missing: {path}")
    options: set[str] = set()
    pins: dict[str, str] = {}
    wheel_hashes: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            options.add(line)
            continue
        match = LOCK_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"{path}:{line_number} is not one exact pin with one SHA-256 wheel hash."
            )
        name = canonical_name(match.group("name"))
        if name in pins:
            raise RuntimeError(f"{path}:{line_number} duplicates {name}.")
        pins[name] = match.group("version")
        wheel_hashes[name] = match.group("hash")

    required_options = {
        "--require-hashes",
        "--only-binary=:all:",
        "--index-url https://pypi.org/simple",
    }
    if not required_options.issubset(options):
        missing = sorted(required_options - options)
        raise RuntimeError(f"{path} is missing lock options: {', '.join(missing)}")
    pytorch_index = "--extra-index-url https://download.pytorch.org/whl/cpu"
    if target_from_path(path).startswith(("linux-", "windows-")):
        if pytorch_index not in options:
            raise RuntimeError(f"{path} is missing the reviewed PyTorch CPU wheel index.")
    elif pytorch_index in options:
        raise RuntimeError(f"{path} must resolve its native macOS Torch wheel from PyPI.")

    target = target_from_path(path)
    expected_packages = TARGET_PACKAGES[target]
    if set(pins) != expected_packages:
        missing = sorted(expected_packages - set(pins))
        unexpected = sorted(set(pins) - expected_packages)
        raise RuntimeError(
            f"{path} package closure changed; missing={missing}, unexpected={unexpected}"
        )
    expected = {
        name: COMMON_PINS[name]
        for name in expected_packages
        if name not in TARGET_PINS[target]
    }
    expected.update(TARGET_PINS[target])
    for name, version in expected.items():
        if pins.get(name) != version:
            raise RuntimeError(
                f"{path} must pin {name}=={version}, found {pins.get(name)!r}."
            )
    reviewed_schema_hashes = {
        **SCHEMA_UNIVERSAL_WHEEL_HASHES,
        "rpds-py": SCHEMA_RPDS_WHEEL_HASHES[target],
    }
    for name, digest in reviewed_schema_hashes.items():
        if wheel_hashes.get(name) != digest:
            raise RuntimeError(f"{path} must select the reviewed {target} schema wheel for {name}.")
    return pins


def target_from_path(path: Path) -> str:
    suffix = "-py311.lock"
    if not path.name.endswith(suffix):
        raise RuntimeError(f"Unexpected engine lock filename: {path.name}")
    return path.name[: -len(suffix)]


def verify_input_pins() -> None:
    inputs = {
        REPOSITORY / "engine" / "requirements.txt": RUNTIME_INPUT_PACKAGES,
        REPOSITORY / "engine" / "requirements-build.txt": BUILD_INPUT_PACKAGES,
    }
    for path, expected_names in inputs.items():
        names: set[str] = set()
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if INPUT_PATTERN.fullmatch(line) is None:
                raise RuntimeError(f"{path}:{line_number} is not an exact reviewed pin: {line}")
            names.add(canonical_name(line.split("==", 1)[0]))
        if names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            raise RuntimeError(
                f"{path} direct package set changed; missing={missing}, unexpected={unexpected}"
            )


def verify_all_locks() -> None:
    verify_input_pins()
    target_union = set().union(*TARGET_PACKAGES.values())
    if target_union != AUDITED_PACKAGES:
        raise RuntimeError("Internal audited engine package inventory is inconsistent.")
    for target in sorted(TARGET_PINS):
        parse_lock(lock_path(target))


def require_release_python(target: str) -> None:
    expected_version = TARGET_PYTHON_VERSIONS[target]
    if sys.version_info[:3] != expected_version:
        expected = ".".join(map(str, expected_version))
        actual = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            f"Release worker locks require CPython {expected}; {sys.executable} is {actual}."
        )


def verify_installed(path: Path) -> None:
    pins = parse_lock(path)
    mismatches: list[str] = []
    for name, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing (expected {expected})")
            continue
        if actual != expected:
            mismatches.append(f"{name}: {actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("Installed engine lock mismatch:\n  " + "\n  ".join(mismatches))


def install(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            str(path),
        ],
        check=True,
    )
    verify_installed(path)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--print-lock", action="store_true")
    actions.add_argument("--verify-locks", action="store_true")
    actions.add_argument("--verify-only", action="store_true")
    parser.add_argument("--target", choices=sorted(TARGET_PINS))
    arguments = parser.parse_args()

    if arguments.verify_locks:
        verify_all_locks()
        pin_count = sum(len(packages) for packages in TARGET_PACKAGES.values())
        print(f"Verified {len(TARGET_PINS)} engine locks ({pin_count} target package pins).")
        return 0

    target = arguments.target or current_target()
    require_release_python(target)
    path = lock_path(target)
    parse_lock(path)
    if arguments.print_lock:
        print(path)
    elif arguments.verify_only:
        verify_installed(path)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    else:
        install(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"engine lock error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
