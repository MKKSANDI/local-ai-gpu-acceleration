from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.offload_plan import (  # noqa: E402
    ModelRow,
    WeightBudget,
    estimate_ngl,
    profile_model,
    profile_tensor_model,
)


class OffloadPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = WeightBudget(
            free_vram_bytes=12 * 1024**3,
            wddm_guard_bytes=512 * 1024**2,
            kv_reserve_bytes=768 * 1024**2,
            workspace_bytes=512 * 1024**2,
            safety_margin_bytes=256 * 1024**2,
        )

    def test_estimated_ngl_tracks_resident_fraction(self) -> None:
        self.assertEqual(estimate_ngl(5 * 1024**3, 10 * 1024**3, 48), 24)
        self.assertEqual(estimate_ngl(20 * 1024**3, 10 * 1024**3, 48), 48)
        self.assertEqual(estimate_ngl(0, 10 * 1024**3, 48), 0)

    def test_strict_profile_admits_full_fit(self) -> None:
        model = ModelRow("small.gguf", "Q4", 4 * 1024**3, False)
        profiles = profile_model(
            model,
            self.budget,
            layer_count=48,
            min_balanced_fraction=0.45,
            latency_extra_guard_bytes=1024**3,
        )
        strict = next(row for row in profiles if row["profile"] == "strict_full_resident")
        self.assertTrue(strict["admit"])
        self.assertEqual(strict["estimated_ngl"], 48)
        self.assertEqual(strict["resident_weight_bytes"], model.size_bytes)

    def test_large_model_is_overflow_only_when_fraction_is_low(self) -> None:
        model = ModelRow("q8.gguf", "Q8_K_P", 40 * 1024**3, False)
        profiles = profile_model(
            model,
            self.budget,
            layer_count=48,
            min_balanced_fraction=0.45,
            latency_extra_guard_bytes=1024**3,
        )
        strict = next(row for row in profiles if row["profile"] == "strict_full_resident")
        balanced = next(row for row in profiles if row["profile"] == "balanced_max_resident")
        overflow = next(row for row in profiles if row["profile"] == "overflow_explicit")
        self.assertFalse(strict["admit"])
        self.assertFalse(balanced["admit"])
        self.assertTrue(overflow["admit"])
        self.assertTrue(overflow["host_spill_allowed"])
        self.assertGreater(overflow["host_weight_bytes"], overflow["resident_weight_bytes"])

    def test_tensor_profile_uses_complete_layer_estimate(self) -> None:
        model = ModelRow("target.gguf", "IQ2_M", 1200, "E:\\models\\target.gguf")
        tensor_plan = {
            "tensor_span_bytes": 1000,
            "layer_count": 40,
            "strict_full_resident": False,
            "resident_weight_bytes_estimate": 810,
            "resident_layers_estimate": 32,
        }
        latency_plan = {
            "tensor_span_bytes": 1000,
            "layer_count": 40,
            "strict_full_resident": False,
            "resident_weight_bytes_estimate": 700,
            "resident_layers_estimate": 28,
        }
        profiles = profile_tensor_model(
            model,
            tensor_plan,
            latency_plan,
            min_balanced_fraction=0.45,
        )
        balanced = next(row for row in profiles if row["profile"] == "balanced_max_resident")
        latency = next(row for row in profiles if row["profile"] == "latency_guard")
        self.assertEqual(balanced["estimated_ngl"], 32)
        self.assertEqual(latency["estimated_ngl"], 28)
        self.assertEqual(balanced["layer_count_basis"], "tensor")
        self.assertAlmostEqual(balanced["gpu_residency_fraction"], 0.81)


if __name__ == "__main__":
    unittest.main()
