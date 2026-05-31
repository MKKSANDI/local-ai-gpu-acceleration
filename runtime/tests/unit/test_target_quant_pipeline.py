from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.target_quant_pipeline import (  # noqa: E402
    llama_metric_summary,
    llama_model_run,
    parse_harness_object,
    resolve_ngl,
    sanitize,
    select_model,
)


class TargetQuantPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = [
            {"filename": "model-Q4.gguf", "quant": "Q4", "size_bytes": 400},
            {"filename": "model-IQ2.gguf", "quant": "IQ2_M", "size_bytes": 200},
            {"filename": "readme.md", "quant": "README", "size_bytes": 1},
        ]

    def test_selects_smallest_gguf_by_default(self) -> None:
        self.assertEqual(select_model(self.models, quant=None, filename=None)["quant"], "IQ2_M")

    def test_selects_by_quant(self) -> None:
        self.assertEqual(select_model(self.models, quant="Q4", filename=None)["filename"], "model-Q4.gguf")

    def test_selects_by_filename(self) -> None:
        self.assertEqual(
            select_model(self.models, quant=None, filename="model-IQ2.gguf")["quant"],
            "IQ2_M",
        )

    def test_sanitize(self) -> None:
        self.assertEqual(sanitize("IQ2/M target"), "IQ2_M_target")

    def test_parse_harness_skips_help_json_examples(self) -> None:
        output = (
            "help text with an example {\"key1\":\"value1\"}\n"
            "{\"path\":\"llamacpp_harness\",\"status\":\"model_run_complete\","
            "\"runs\":[{\"returncode\":0,\"wall_ms\":42.0,"
            "\"metrics\":{\"decode_tokens_per_second\":9.7,\"graphs_reused\":30}}]}\n"
            "json=E:\\benchmarks\\llamacpp_harness.json\n"
        )
        parsed = parse_harness_object(output)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["status"], "model_run_complete")

    def test_metric_summary_uses_model_run_metrics(self) -> None:
        harness = {
            "path": "llamacpp_harness",
            "status": "model_run_complete",
            "runs": [
                {"command": ["llama-cli.exe", "--help"], "returncode": 0},
                {
                    "command": ["llama-completion.exe", "-m", "model.gguf"],
                    "returncode": 0,
                    "wall_ms": 1000.0,
                    "metrics": {
                        "prompt_tokens": 7,
                        "prompt_tokens_per_second": 3.26,
                        "decode_runs": 31,
                        "decode_tokens_per_second": 9.7,
                        "graphs_reused": 30,
                        "load_time_ms": 215615.11,
                        "use_graphs": True,
                    },
                },
            ],
        }
        self.assertEqual(llama_model_run(harness)["command"][0], "llama-completion.exe")
        summary = llama_metric_summary(harness)
        self.assertEqual(summary["decode_tokens_per_second"], 9.7)
        self.assertEqual(summary["graphs_reused"], 30)

    def test_resolve_ngl_prefers_override(self) -> None:
        tensor_plan = {"resident_layers_estimate": 32}
        self.assertEqual(resolve_ngl(tensor_plan, 28), (28, "override"))
        self.assertEqual(resolve_ngl(tensor_plan, None), (32, "tensor_plan"))
        self.assertEqual(resolve_ngl(None, None), (0, "none"))

    def test_resolve_ngl_rejects_negative_override(self) -> None:
        with self.assertRaises(ValueError):
            resolve_ngl({"resident_layers_estimate": 32}, -1)


if __name__ == "__main__":
    unittest.main()
