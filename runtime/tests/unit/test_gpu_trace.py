from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.run_with_gpu_trace import (  # noqa: E402
    parse_nvidia_smi_row,
    sanitize_label,
    strip_delimiter,
    summarize_samples,
)


class GpuTraceTests(unittest.TestCase):
    def test_parse_nvidia_smi_row(self) -> None:
        sample = parse_nvidia_smi_row(
            "2026/05/28 22:14:05.123, NVIDIA GeForce RTX 3060, 9, 512, 11776, 42.50, 47"
        )
        self.assertEqual(sample["name"], "NVIDIA GeForce RTX 3060")
        self.assertEqual(sample["utilization_gpu_percent"], 9)
        self.assertEqual(sample["memory_used_mib"], 512)
        self.assertEqual(sample["memory_free_mib"], 11776)
        self.assertEqual(sample["power_draw_w"], 42.5)
        self.assertEqual(sample["temperature_gpu_c"], 47)

    def test_parse_nvidia_smi_na_values(self) -> None:
        sample = parse_nvidia_smi_row(
            "2026/05/28 22:14:05.123, NVIDIA GeForce RTX 3060, N/A, 512, 11776, N/A, 47"
        )
        self.assertIsNone(sample["utilization_gpu_percent"])
        self.assertIsNone(sample["power_draw_w"])

    def test_summary_ignores_missing_values(self) -> None:
        samples = [
            parse_nvidia_smi_row(
                "2026/05/28 22:14:05.123, NVIDIA GeForce RTX 3060, 9, 512, 11776, 42.50, 47"
            ),
            parse_nvidia_smi_row(
                "2026/05/28 22:14:05.373, NVIDIA GeForce RTX 3060, N/A, 768, 11520, N/A, 48"
            ),
        ]
        summary = summarize_samples(samples)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["max_memory_used_mib"], 768.0)
        self.assertEqual(summary["min_memory_free_mib"], 11520.0)
        self.assertEqual(summary["max_gpu_utilization_percent"], 9.0)
        self.assertEqual(summary["max_power_draw_w"], 42.5)

    def test_command_delimiter_and_label(self) -> None:
        self.assertEqual(strip_delimiter(["--", "python", "--version"]), ["python", "--version"])
        self.assertEqual(sanitize_label("llama smoke: q4/km"), "llama_smoke_q4_km")
        self.assertEqual(sanitize_label(""), "gpu_trace")


if __name__ == "__main__":
    unittest.main()
