"""Fixed, stdlib-only source contracts for package CI, never brain discovery."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "engine/tests/test_native_compute_tensor_bytes.py",
    "engine/tests/test_manual_regional_execution.py",
    "engine/tests/test_manual_regional_boundaries.py",
    "engine/tests/test_whole_path_memory_scheduler.py",
    "engine/tests/test_reply_execution_planner.py",
    "engine/tests/test_mixed_region_schedule.py",
    "engine/tests/test_outer_learning_pipeline.py",
    "engine/tests/test_native_kernel_packaging_integrity.py",
    "engine/tests/test_retained_context_capacity.py",
)
FORBIDDEN = ("torch", "tensorflow", "jax", "mlx", "transformers", "safetensors", "omni_core")


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise RuntimeError("Bounded CI source contracts cannot import a neural runtime: " + fullname)
        return None


def manifest():
    return {"format": "omni-python-source-contracts", "version": 1,
            "scope": "fixed-stdlib-constructor-free-source-contracts",
            "fullTestDiscovery": False, "neuralRuntimeImportsAllowed": False,
            "brainCreated": False, "trainingPerformed": False,
            "files": [{"path": relative, "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()}
                      for relative in FILES]}


def main():
    if sys.argv[1:] not in ([], ["--list"]):
        raise ValueError("Use the fixed contract selection or --list; arbitrary test paths are not accepted.")
    evidence = manifest()
    if sys.argv[1:] == ["--list"]:
        print(json.dumps(evidence, sort_keys=True))
        return 0
    sys.meta_path.insert(0, NoNeuralImports())
    # This is an in-process import guard, not an OS sandbox. The fixed reviewed
    # fixtures use inert owners/extracted methods; it is not permission to add
    # subprocess-based neural tests to this lane or claim neural capability.
    suite = unittest.TestSuite()
    for index, relative in enumerate(FILES):
        spec = importlib.util.spec_from_file_location("owned_ci_contract_" + str(index), ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    evidence.update(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                    skipped=len(result.skipped), passed=result.wasSuccessful())
    print(json.dumps(evidence, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
