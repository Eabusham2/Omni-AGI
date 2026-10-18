"""Seal final packaged macOS bytes after a checked build transformation."""
import argparse
import json
from pathlib import Path
import sys

import importlib.util
_SPEC = importlib.util.spec_from_file_location("omni_build_native_integrity", Path(__file__).with_name("native_kernel_integrity.py"))
_INTEGRITY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INTEGRITY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--app-root", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin" or bool(args.worker_root) == bool(args.app_root):
        raise ValueError("select exactly one actual macOS worker or app root")
    if args.worker_root:
        root = args.worker_root.resolve()
        modules = [root / "_internal/omni_core"]
    else:
        root = args.app_root.resolve()
        if root.suffix != ".app":
            raise ValueError("native kernel finalization requires its exact app bundle")
        resources = root / "Contents/Resources"
        modules = [resources / "engine/omni_core", resources / "engine-runtime/_internal/omni_core"]
    compiled = _INTEGRITY.checked_compiled_manifest(Path(__file__).resolve().parents[1] / "engine/omni_core")
    print(json.dumps({"nativeKernels": [_INTEGRITY.finalize_manifest(module, root, compiled_manifest=compiled) for module in modules]}))


if __name__ == "__main__": main()
