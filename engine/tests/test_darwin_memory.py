import ctypes
import sys
import unittest
from unittest.mock import patch

from omni_core.darwin_memory import VMStatistics64Rev0, available_bytes, memory_reading
from omni_core.offload import _mac_memory


class DarwinMemoryTests(unittest.TestCase):
    def test_statistics_prefix_matches_rev0_layout(self):
        self.assertEqual(ctypes.sizeof(VMStatistics64Rev0), 96)
        self.assertEqual(VMStatistics64Rev0.purgeable_count.offset, 88)
        self.assertEqual(VMStatistics64Rev0.speculative_count.offset, 92)

    def test_bounded_same_policy_without_speculative_double_count(self):
        stats = VMStatistics64Rev0(free_count=10, inactive_count=20,
                                  purgeable_count=2, speculative_count=8)
        self.assertEqual(available_bytes(100_000, 1024, stats), 32 * 1024)
        self.assertEqual(available_bytes(100_000, 1024, stats, 50), 50_000)
        self.assertEqual(available_bytes(100_000, 1024, stats, 101), 32 * 1024)
        self.assertEqual(available_bytes(1024, 1024, stats), 1024)
        with self.assertRaises(ValueError):
            available_bytes(0, 1024, stats)

    def test_api_failure_uses_existing_measured_fallback(self):
        with patch("omni_core.darwin_memory.memory_reading", side_effect=OSError("unsupported")), \
             patch("omni_core.offload._mac_memory_command_fallback", return_value=(100, 20)) as fallback:
            self.assertEqual(_mac_memory(), (100, 20))
            fallback.assert_called_once()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin ABI")
    def test_actual_darwin_path_launches_no_commands(self):
        with patch("subprocess.check_output", side_effect=AssertionError("memory hot path forked")):
            total, available = memory_reading()
            self.assertGreater(total, 0)
            self.assertGreaterEqual(available, 0)
            self.assertLessEqual(available, total)
            self.assertEqual(_mac_memory()[0], total)


if __name__ == "__main__":
    unittest.main()
