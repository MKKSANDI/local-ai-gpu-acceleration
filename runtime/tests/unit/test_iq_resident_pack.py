import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.bench.iq_resident_pack import iq_tensor_shape, pack_tensor, select_tensors, tensor_role


class IQResidentPackTests(unittest.TestCase):
    def test_iq2s_shape_uses_256_value_blocks(self) -> None:
        values, blocks = iq_tensor_shape(
            {
                "name": "blk.0.ffn_gate_exps.weight",
                "type": "IQ2_S",
                "dimensions": [2048, 512, 128],
            }
        )
        self.assertEqual(values, 134217728)
        self.assertEqual(blocks, 524288)

    def test_iq3s_shape_uses_same_block_width(self) -> None:
        values, blocks = iq_tensor_shape(
            {
                "name": "blk.0.some.weight",
                "type": "IQ3_S",
                "dimensions": [2048, 512],
            }
        )
        self.assertEqual(values, 1048576)
        self.assertEqual(blocks, 4096)

    def test_role_parser_preserves_layer_and_weight_suffix(self) -> None:
        self.assertEqual(
            tensor_role("blk.12.ffn_gate_exps.weight"),
            (12, "ffn_gate_exps.weight"),
        )
        self.assertEqual(tensor_role("output.weight"), (None, "output"))

    def test_select_tensors_filters_layer_before_limit(self) -> None:
        tensors = [
            {"name": "blk.0.attn_gate.weight"},
            {"name": "blk.0.ffn_gate_exps.weight"},
            {"name": "blk.1.attn_gate.weight"},
            {"name": "output.weight"},
        ]

        selected = select_tensors(tensors, layer=0, limit=1)

        self.assertEqual([tensor["name"] for tensor in selected], ["blk.0.attn_gate.weight"])

    def test_select_tensors_respects_requested_names(self) -> None:
        tensors = [
            {"name": "blk.0.attn_gate.weight"},
            {"name": "blk.0.ffn_gate_exps.weight"},
        ]

        selected = select_tensors(tensors, requested={"blk.0.ffn_gate_exps.weight"})

        self.assertEqual([tensor["name"] for tensor in selected], ["blk.0.ffn_gate_exps.weight"])

    def test_pack_tensor_marks_iq3s_runtime_probe_kernel(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(bytes(range(110)))
            out_dir = root / "pack"
            out_dir.mkdir()

            row = pack_tensor(
                model=model,
                tensor={
                    "name": "blk.0.ffn_down_shexp.weight",
                    "type": "IQ3_S",
                    "dimensions": [256],
                    "absolute_offset": 0,
                    "span_bytes": 110,
                },
                out_dir=out_dir,
            )

            self.assertEqual(row["runtime_kernel"], "iq3s_raw_block_probe")
            self.assertEqual(row["block_type_size"], 110)


if __name__ == "__main__":
    unittest.main()
