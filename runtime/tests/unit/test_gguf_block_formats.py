from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_block_formats import (  # noqa: E402
    expected_data_bytes,
    validate_tensor_storage,
    validation_summary,
)


class GGufBlockFormatsTests(unittest.TestCase):
    def test_expected_data_bytes_for_target_formats(self) -> None:
        self.assertEqual(expected_data_bytes("Q4_K", [2048, 8192]), 9_437_184)
        self.assertEqual(expected_data_bytes("Q5_K", [2048, 248320]), 349_634_560)
        self.assertEqual(expected_data_bytes("IQ2_S", [2048, 512, 256]), 85_983_232)
        self.assertEqual(expected_data_bytes("IQ3_S", [512, 2048, 256]), 115_343_360)
        self.assertEqual(expected_data_bytes("F32", [2048]), 8192)

    def test_validate_tensor_storage_allows_alignment_padding(self) -> None:
        row = validate_tensor_storage(
            {
                "name": "blk.0.attn_qkv.weight",
                "type": "Q4_K",
                "dimensions": [2048, 8192],
                "span_bytes": 9_437_184 + 16,
            },
            alignment=32,
        )
        self.assertTrue(row["valid"])
        self.assertEqual(row["padding_bytes"], 16)

    def test_validate_tensor_storage_rejects_bad_span(self) -> None:
        row = validate_tensor_storage(
            {
                "name": "bad",
                "type": "IQ2_S",
                "dimensions": [256],
                "span_bytes": 1,
            },
            alignment=32,
        )
        self.assertFalse(row["valid"])
        self.assertIn("smaller", row["reason"])

    def test_validation_summary_counts_rows(self) -> None:
        rows = [
            validate_tensor_storage(
                {"name": "a", "type": "F32", "dimensions": [4], "span_bytes": 16},
                alignment=32,
            ),
            validate_tensor_storage(
                {"name": "b", "type": "UNKNOWN", "dimensions": [4], "span_bytes": 16},
                alignment=32,
            ),
        ]
        summary = validation_summary(rows)
        self.assertEqual(summary["tensor_count"], 2)
        self.assertEqual(summary["supported_tensors"], 1)
        self.assertEqual(summary["valid_tensors"], 1)
        self.assertEqual(summary["invalid_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
