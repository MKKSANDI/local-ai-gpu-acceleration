from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.compare_target_runs import (  # noqa: E402
    best_decode_row,
    build_rows,
    latest_by_ngl,
    ngl_from_path,
)


class CompareTargetRunsTests(unittest.TestCase):
    def test_extracts_ngl_from_path(self) -> None:
        self.assertEqual(ngl_from_path(Path("target_quant_pipeline_IQ2_M_ngl31.json")), 31)
        self.assertEqual(ngl_from_path(Path("target-iq2m-ngl28-controlled_1.json")), 28)
        self.assertIsNone(ngl_from_path(Path("target_quant_pipeline_IQ2_M.json")))

    def test_latest_by_ngl_groups_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "target-iq2m-ngl31-controlled_old.json"
            new = root / "target-iq2m-ngl31-controlled_new.json"
            old.write_text("{}", encoding="utf-8")
            new.write_text("{}", encoding="utf-8")
            os.utime(old, (100.0, 100.0))
            os.utime(new, (200.0, 200.0))
            latest = latest_by_ngl([old, new])
            self.assertEqual(latest[31], new)

    def test_build_rows_joins_run_and_trace_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "benchmarks"
            traces = root / "traces"
            bench.mkdir()
            traces.mkdir()
            (bench / "target_quant_pipeline_IQ2_M_ngl28.json").write_text(
                json.dumps(
                    {
                        "status": "llamacpp_run_complete",
                        "ngl_source": "override",
                        "tensor_plan": {
                            "resident_layers_estimate": 31,
                            "gpu_residency_fraction_estimate": 0.788,
                        },
                        "llamacpp_metrics": {
                            "decode_tokens_per_second": 17.41,
                            "prompt_tokens_per_second": 4.46,
                            "graphs_reused": 30,
                            "load_time_ms": 206564.28,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (traces / "target-iq2m-ngl28-controlled_1.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "avg_gpu_utilization_percent": 29.9,
                            "max_memory_used_mib": 9009,
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = build_rows(root, "IQ2_M", "target-iq2m")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ngl"], 28)
            self.assertEqual(rows[0]["decode_tokens_per_second"], 17.41)
            self.assertEqual(rows[0]["max_memory_used_mib"], 9009)

    def test_best_decode_row(self) -> None:
        best = best_decode_row(
            [
                {"ngl": 28, "decode_tokens_per_second": 17.41},
                {"ngl": 32, "decode_tokens_per_second": 20.86},
            ]
        )
        self.assertEqual(best["ngl"], 32)


if __name__ == "__main__":
    unittest.main()
