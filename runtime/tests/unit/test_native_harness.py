from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.bench.native_harness import (  # noqa: E402
    default_target_model,
    default_target_q4_payload,
    extract_json_object,
    latest_iq2s_pack_manifest,
    latest_iq3s_pack_manifest,
    latest_q4k_pack_manifest,
    latest_q5k_pack_manifest,
    q4k_validation_manifest,
    sanitize_label,
)


class NativeHarnessTests(unittest.TestCase):
    def test_extract_json_object_from_console_output(self) -> None:
        parsed = extract_json_object(
            'prefix\n{\n  "path": "native_cuda_graph_synthetic_decode",\n'
            '  "steps": 256,\n  "steps_per_second": 4446.56\n}\nsuffix'
        )
        self.assertEqual(parsed["path"], "native_cuda_graph_synthetic_decode")
        self.assertEqual(parsed["steps"], 256)

    def test_extract_json_rejects_missing_object(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("not json")

    def test_sanitize_label(self) -> None:
        self.assertEqual(sanitize_label("stress 8192x8192"), "stress_8192x8192")
        self.assertEqual(sanitize_label(""), "native")

    def test_default_target_model_is_on_external_storage_root(self) -> None:
        path = default_target_model(Path(r"E:\AI project"))
        self.assertIn("HauhauCS__", str(path))
        self.assertEqual(path.name, "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf")

    def test_default_target_q4_payload_is_external_tmp(self) -> None:
        path = default_target_q4_payload(Path(r"E:\AI project"))
        self.assertEqual(path.parent, Path(r"E:\AI project\tmp"))
        self.assertEqual(path.name, "blk0_attn_qkv_q4_payload.bin")

    def test_q4k_manifest_helpers_find_expected_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_root = (
                root
                / "packs"
                / "q4k"
                / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M"
            )
            (pack_root / "100").mkdir(parents=True)
            (pack_root / "200").mkdir()
            (pack_root / "validation_reps").mkdir()
            (pack_root / "100" / "manifest.json").write_text("{}", encoding="utf-8")
            (pack_root / "200" / "manifest.json").write_text("{}", encoding="utf-8")
            (pack_root / "validation_reps" / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_q4k_pack_manifest(root), pack_root / "200" / "manifest.json")
            self.assertEqual(q4k_validation_manifest(root), pack_root / "validation_reps" / "manifest.json")

    def test_iq2s_manifest_helper_finds_latest_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_root = (
                root
                / "packs"
                / "iq"
                / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M"
                / "iq2_s"
            )
            (pack_root / "100").mkdir(parents=True)
            (pack_root / "300").mkdir()
            (pack_root / "100" / "manifest.json").write_text("{}", encoding="utf-8")
            (pack_root / "300" / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_iq2s_pack_manifest(root), pack_root / "300" / "manifest.json")

    def test_iq3s_manifest_helper_finds_latest_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_root = (
                root
                / "packs"
                / "iq"
                / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M"
                / "iq3_s"
            )
            (pack_root / "100").mkdir(parents=True)
            (pack_root / "350").mkdir()
            (pack_root / "100" / "manifest.json").write_text("{}", encoding="utf-8")
            (pack_root / "350" / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_iq3s_pack_manifest(root), pack_root / "350" / "manifest.json")

    def test_q5k_manifest_helper_finds_latest_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_root = (
                root
                / "packs"
                / "q5k"
                / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M"
            )
            (pack_root / "100").mkdir(parents=True)
            (pack_root / "250").mkdir()
            (pack_root / "100" / "manifest.json").write_text("{}", encoding="utf-8")
            (pack_root / "250" / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_q5k_pack_manifest(root), pack_root / "250" / "manifest.json")


if __name__ == "__main__":
    unittest.main()
