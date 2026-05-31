from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_tensor_plan import tensor_layer, tensor_spans  # noqa: E402


class GgufTensorPlanTests(unittest.TestCase):
    def test_tensor_layer_parsing(self) -> None:
        self.assertEqual(tensor_layer("blk.12.attn_q.weight"), 12)
        self.assertIsNone(tensor_layer("token_embd.weight"))

    def test_tensor_spans_from_offsets(self) -> None:
        metadata = {
            "tensor_data_offset": 128,
            "tensors": [
                {"name": "token_embd.weight", "offset": 0, "type": "F16", "dimensions": [8, 16]},
                {"name": "blk.0.attn_q.weight", "offset": 256, "type": "F16", "dimensions": [8, 8]},
                {"name": "blk.1.attn_q.weight", "offset": 384, "type": "F16", "dimensions": [8, 8]},
            ],
        }
        spans = tensor_spans(metadata, file_size=640)
        self.assertEqual([row["span_bytes"] for row in spans], [256, 128, 128])
        self.assertIsNone(spans[0]["layer"])
        self.assertEqual(spans[1]["layer"], 0)
        self.assertEqual(spans[2]["absolute_offset"], 512)


if __name__ == "__main__":
    unittest.main()
