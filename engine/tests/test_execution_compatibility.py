"""Runtime compatibility is strict; this suite constructs no neural owner."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from omni_core.execution_compatibility import capture_execution_compatibility, require_execution_compatibility, validate_execution_compatibility


def sealed(body):
    return {"declaration": body, "sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


class ExecutionCompatibilityTests(unittest.TestCase):
    def test_current_cpu_hardware_kernel_build_and_threads_have_a_reproducible_seal(self):
        first = capture_execution_compatibility("cpu")
        self.assertEqual(first, capture_execution_compatibility("cpu"))
        self.assertIs(validate_execution_compatibility(first), first)
        self.assertIn("weighted_ternary.cpp", " ".join(first["declaration"]["kernels"]))
        self.assertIn("libraries", first["declaration"]["torch"])
        with patch("omni_core.execution_compatibility.capture_execution_compatibility", return_value=first):
            require_execution_compatibility(first, "cpu")
            for field in ("system", "torch", "kernels", "threads"):
                changed = copy.deepcopy(first["declaration"]); changed[field] = {"changed": True}
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    require_execution_compatibility(sealed(changed), "cpu")

    def test_old_or_corrupted_frontier_does_not_guess_an_exact_runtime(self):
        with self.assertRaisesRegex(ValueError, "predates"):
            require_execution_compatibility(None, "cpu")
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_execution_compatibility({"declaration": {"format": "omni-exact-runtime-compatibility", "version": 1}, "sha256": "0" * 64})


if __name__ == "__main__": unittest.main()
