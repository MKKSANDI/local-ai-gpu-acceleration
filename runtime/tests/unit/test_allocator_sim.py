from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.allocator_sim import ALIGNMENT, Arena, align_up, simulate  # noqa: E402


class AllocatorSimulationTests(unittest.TestCase):
    def test_align_up(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), ALIGNMENT)
        self.assertEqual(align_up(ALIGNMENT), ALIGNMENT)
        self.assertEqual(align_up(ALIGNMENT + 1), ALIGNMENT * 2)

    def test_free_block_reuse(self) -> None:
        arena = Arena("workspace", 4096)
        first = arena.allocate("first", 512)
        self.assertIsNotNone(first)
        arena.free("first")
        second = arena.allocate("second", 256)
        self.assertIsNotNone(second)
        self.assertEqual(second.offset, first.offset)
        self.assertEqual(arena.reuse_hits, 1)

    def test_overflow_failure_is_counted(self) -> None:
        arena = Arena("dma", 1024)
        self.assertIsNotNone(arena.allocate("a", 1024))
        self.assertIsNone(arena.allocate("b", 1))
        self.assertEqual(arena.failed_allocations, 1)

    def test_simulation_reports_arena_metrics(self) -> None:
        result = simulate(seed=3, steps=32, weight_mib=16, kv_mib=64, workspace_mib=32, dma_mib=8)
        self.assertEqual(result["path"], "allocator_sim")
        self.assertIn("workspace", result["arenas"])
        self.assertGreater(result["total_high_water_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
