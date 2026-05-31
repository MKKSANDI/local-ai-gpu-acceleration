import struct
import unittest

from tools.bench.q4k_resident_pack import (
    Q4K_BYTES,
    Q4K_PAYLOAD_BYTES,
    auto_policy,
    build_metadata,
    extract_payload,
    metadata_bytes_per_block,
    metadata_format,
    scale_min_k4,
)


def half(value: float) -> bytes:
    return struct.pack("<e", value)


def sample_block() -> bytes:
    block = bytearray(Q4K_BYTES)
    block[0:2] = half(0.5)
    block[2:4] = half(0.25)
    block[4:16] = bytes([1, 2, 3, 4, 5, 6, 7, 8, 0x40, 0x80, 0xC0, 0x00])
    for index in range(Q4K_PAYLOAD_BYTES):
        block[16 + index] = index & 0xFF
    return bytes(block)


class Q4KResidentPackTests(unittest.TestCase):
    def test_scale_min_unpacking_matches_ggml_k4_layout(self) -> None:
        scales = sample_block()[4:16]
        self.assertEqual(scale_min_k4(0, scales), (1, 5))
        self.assertEqual(scale_min_k4(3, scales), (4, 8))
        self.assertEqual(scale_min_k4(4, scales), (0, 4))
        self.assertEqual(scale_min_k4(5, scales), (0, 8))
        self.assertEqual(scale_min_k4(6, scales), (0, 12))

    def test_payload_extraction_strips_q4k_metadata(self) -> None:
        block = sample_block()
        payload = extract_payload(block)
        self.assertEqual(len(payload), Q4K_PAYLOAD_BYTES)
        self.assertEqual(payload[:4], bytes([0, 1, 2, 3]))
        self.assertEqual(payload[-1], 127)

    def test_split_predecoded_metadata_products_are_fp32(self) -> None:
        metadata = build_metadata(sample_block(), "split-predecoded")
        self.assertEqual(len(metadata), metadata_bytes_per_block("split-predecoded"))
        first_scale, first_min = struct.unpack("<ff", metadata[:8])
        self.assertEqual(first_scale, 0.5)
        self.assertEqual(first_min, 1.25)

    def test_compact_metadata_preserves_raw_d_and_unpacked_scales(self) -> None:
        block = sample_block()
        metadata = build_metadata(block, "split-compact")
        self.assertEqual(len(metadata), metadata_bytes_per_block("split-compact"))
        self.assertEqual(metadata[:4], block[:4])
        self.assertEqual(metadata[4:8], bytes([1, 2, 3, 4]))
        self.assertEqual(metadata[12:16], bytes([5, 6, 7, 8]))

    def test_native_metadata_preserves_original_sixteen_bytes(self) -> None:
        block = sample_block()
        metadata = build_metadata(block, "split-native")
        self.assertEqual(len(metadata), metadata_bytes_per_block("split-native"))
        self.assertEqual(metadata, block[:16])
        self.assertEqual(metadata_format("split-native"), "native16")

    def test_auto_policy_is_shape_aware(self) -> None:
        self.assertEqual(auto_policy(rows=512, cols=2048, block_count=4096), "split-compact")
        self.assertEqual(
            auto_policy(rows=8192, cols=2048, block_count=65536),
            "split-predecoded",
        )


if __name__ == "__main__":
    unittest.main()
