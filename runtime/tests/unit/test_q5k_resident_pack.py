from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.q5k_resident_pack import (  # noqa: E402
    pack_tensor,
    q5k_tensor_shape,
    tensor_role,
)


class Q5KResidentPackTests(unittest.TestCase):
    def test_q5k_shape_uses_256_value_blocks(self) -> None:
        values, blocks = q5k_tensor_shape(
            {
                "name": "output.weight",
                "type": "Q5_K",
                "dimensions": [2048, 248320],
            }
        )
        self.assertEqual(values, 508_559_360)
        self.assertEqual(blocks, 1_986_560)

    def test_role_parser_handles_output_weight(self) -> None:
        self.assertEqual(tensor_role("output.weight"), (None, "output"))
        self.assertEqual(tensor_role("blk.1.some.weight"), (1, "some.weight"))

    def test_pack_tensor_writes_raw_blocks_and_manifest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "tiny.gguf"
            payload = bytes((index % 251 for index in range(176 * 2)))
            prefix = b"header"
            model.write_bytes(prefix + payload)
            out_dir = root / "pack"
            out_dir.mkdir()
            tensor = {
                "name": "output.weight",
                "type": "Q5_K",
                "dimensions": [256, 2],
                "absolute_offset": len(prefix),
                "span_bytes": len(payload),
            }

            row = pack_tensor(model=model, tensor=tensor, out_dir=out_dir)

            payload_path = Path(str(row["payload_path"]))
            self.assertEqual(payload_path.read_bytes(), payload)
            self.assertEqual(row["block_count"], 2)
            self.assertEqual(row["block_type_size"], 176)
            self.assertEqual(row["payload_checksum"], sum(payload))
            self.assertEqual(row["source_checksum"], sum(payload))
            self.assertEqual(row["runtime_kernel"], "q5_k_raw_pending")
            json.dumps(row)


if __name__ == "__main__":
    unittest.main()
