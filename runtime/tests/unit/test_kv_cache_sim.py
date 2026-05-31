from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.kv_cache_sim import (  # noqa: E402
    PageArena,
    PrefixEntry,
    maybe_reclaim_prefixes,
    pages_for_tokens,
    simulate,
)


class KvCacheSimulationTests(unittest.TestCase):
    def test_pages_for_tokens_rounds_up(self) -> None:
        self.assertEqual(pages_for_tokens(1, 16), 1)
        self.assertEqual(pages_for_tokens(16, 16), 1)
        self.assertEqual(pages_for_tokens(17, 16), 2)

    def test_arena_reuses_released_pages(self) -> None:
        arena = PageArena(page_count=2)
        first = arena.reserve(2, "a")
        self.assertIsNotNone(first)
        self.assertIsNone(arena.reserve(1, "b"))
        arena.release([first[0]])
        second = arena.reserve(1, "c")
        self.assertEqual(second, [first[0]])

    def test_simulation_hits_prefix_cache_and_drains(self) -> None:
        result = simulate(
            page_count=256,
            page_tokens=16,
            requests=96,
            max_active=8,
            prefix_pool=4,
            keep_prefixes=3,
            seed=11,
        )
        self.assertGreater(result["admitted"], 0)
        self.assertGreater(result["prefix_hits"], 0)
        self.assertGreaterEqual(result["high_water_pages"], result["used_pages_after_drain"])

    def test_small_arena_rejects_under_pressure(self) -> None:
        result = simulate(
            page_count=16,
            page_tokens=16,
            requests=96,
            max_active=16,
            prefix_pool=16,
            keep_prefixes=16,
            seed=3,
        )
        self.assertGreater(result["rejected"], 0)
        self.assertGreater(result["allocation_failures"], 0)

    def test_prefix_reclaim_respects_keep_limit(self) -> None:
        arena = PageArena(page_count=8)
        prefixes = {}
        for index in range(4):
            pages = arena.reserve(1, f"shared:prefix-{index}")
            self.assertIsNotNone(pages)
            prefixes[f"prefix-{index}"] = PrefixEntry(
                key=f"prefix-{index}",
                tokens=16,
                pages=pages,
                hits=index,
                last_used_step=index,
            )
        reclaimed = maybe_reclaim_prefixes(arena, prefixes, active=[], keep_prefixes=2)
        self.assertEqual(reclaimed, 2)
        self.assertEqual(len(prefixes), 2)


if __name__ == "__main__":
    unittest.main()
