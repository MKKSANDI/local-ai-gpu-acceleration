from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_pack_plan import (  # noqa: E402
    build_pack_plan,
    pack_action,
    summarize_roles,
    summarize_types,
    tensor_role,
)


class GGufPackPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tensors = [
            {"name": "token_embd.weight", "type": "IQ3_S", "span_bytes": 110, "dimensions": [256]},
            {"name": "blk.0.attn_q.weight", "type": "Q8_0", "span_bytes": 34, "dimensions": [32]},
            {"name": "blk.0.ffn_down_exps.weight", "type": "IQ2_S", "span_bytes": 82, "dimensions": [256]},
            {"name": "output.weight", "type": "Q5_K", "span_bytes": 176, "dimensions": [256]},
        ]

    def test_tensor_role_strips_layer_prefix_and_weight_suffix(self) -> None:
        self.assertEqual(tensor_role("blk.12.attn_q.weight"), "attn_q.weight")
        self.assertEqual(tensor_role("token_embd.weight"), "token_embd")

    def test_pack_action_classifies_known_types(self) -> None:
        self.assertEqual(
            pack_action("IQ2_S"),
            "iq2s_raw_resident_block_packer_plus_dequant_probe",
        )
        self.assertEqual(
            pack_action("IQ3_S"),
            "iq3s_raw_resident_block_packer_plus_dequant_probe",
        )
        self.assertEqual(pack_action("Q4_K"), "q4k_resident_payload_metadata_packer")
        self.assertEqual(pack_action("Q5_K"), "q5k_raw_resident_block_packer_plus_dequant_probe")
        self.assertEqual(pack_action("UNKNOWN"), "unknown_source_type")

    def test_type_summary_accumulates_bytes(self) -> None:
        summary = summarize_types(self.tensors)
        by_type = {row["type"]: row for row in summary}
        self.assertEqual(by_type["IQ2_S"]["span_bytes"], 82)
        self.assertEqual(
            by_type["Q5_K"]["pack_action"],
            "q5k_raw_resident_block_packer_plus_dequant_probe",
        )

    def test_role_summary_accumulates_roles(self) -> None:
        summary = summarize_roles(self.tensors)
        by_role = {row["role"]: row for row in summary}
        self.assertEqual(by_role["ffn_down_exps.weight"]["span_bytes"], 82)
        self.assertEqual(by_role["token_embd"]["types"], {"IQ3_S": 1})

    def test_build_pack_plan_reports_loader_gap(self) -> None:
        plan = build_pack_plan(
            {
                "model_path": "model.gguf",
                "metadata": {"architecture": "qwen35moe", "tensor_count": 4},
                "layer_count": 1,
                "tensor_span_mib": 1.0,
                "strict_full_resident": False,
                "resident_layers_estimate": 1,
                "tensors": self.tensors,
            }
        )
        self.assertEqual(plan["architecture"], "qwen35moe")
        self.assertIn("Q4_K tensors have an offline resident", plan["loader_gap"])
        self.assertIn("IQ2_S tensors now have a raw resident block packer", plan["loader_gap"])
        self.assertIn("IQ3_S tensors now have a raw resident block packer", plan["loader_gap"])
        self.assertIn("Q5_K has a raw resident block packer plus CUDA dequant", plan["loader_gap"])
        self.assertEqual(plan["block_validation_summary"]["tensor_count"], 4)
        self.assertEqual(plan["block_validation_summary"]["valid_tensors"], 4)


if __name__ == "__main__":
    unittest.main()
