from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.policy_sim import (  # noqa: E402
    ModelEstimate,
    RuntimeBudget,
    decide_balanced,
    decide_overflow,
    decide_strict,
)


class PolicySimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = RuntimeBudget(
            free_vram_bytes=12 * 1024**3,
            wddm_guard_bytes=512 * 1024**2,
            workspace_bytes=512 * 1024**2,
            kv_bytes=768 * 1024**2,
            safety_margin_bytes=256 * 1024**2,
        )

    def test_strict_admits_fitting_model(self) -> None:
        model = ModelEstimate("small.gguf", 4 * 1024**3, "Q4")
        decision = decide_strict(model, self.budget)
        self.assertTrue(decision["admit"])
        self.assertFalse(decision["host_spill_allowed"])
        self.assertEqual(decision["gpu_residency_fraction"], 1.0)

    def test_strict_rejects_oversized_model(self) -> None:
        model = ModelEstimate("large.gguf", 20 * 1024**3, "Q4")
        decision = decide_strict(model, self.budget)
        self.assertFalse(decision["admit"])
        self.assertFalse(decision["host_spill_allowed"])

    def test_balanced_requires_meaningful_gpu_fraction(self) -> None:
        model = ModelEstimate("huge.gguf", 80 * 1024**3, "Q8")
        decision = decide_balanced(model, self.budget, min_gpu_fraction=0.45)
        self.assertFalse(decision["admit"])

    def test_overflow_is_explicit_spill(self) -> None:
        model = ModelEstimate("large.gguf", 20 * 1024**3, "Q4")
        decision = decide_overflow(model, self.budget)
        self.assertTrue(decision["admit"])
        self.assertTrue(decision["host_spill_allowed"])
        self.assertGreater(decision["host_weight_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
