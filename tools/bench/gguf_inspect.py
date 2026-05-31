from __future__ import annotations

import argparse
import io
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
}


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass
class Reader:
    fh: BinaryIO

    def read_exact(self, count: int) -> bytes:
        data = self.fh.read(count)
        if len(data) != count:
            raise EOFError("unexpected end of GGUF file")
        return data

    def u8(self) -> int:
        return self.read_exact(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read_exact(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read_exact(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read_exact(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read_exact(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read_exact(length).decode("utf-8", errors="replace")


def read_value(reader: Reader, value_type: int, array_limit: int, full_arrays: bool):
    if value_type == 0:
        return reader.u8()
    if value_type == 1:
        return reader.i8()
    if value_type == 2:
        return reader.u16()
    if value_type == 3:
        return reader.i16()
    if value_type == 4:
        return reader.u32()
    if value_type == 5:
        return reader.i32()
    if value_type == 6:
        return reader.f32()
    if value_type == 7:
        return bool(reader.u8())
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        item_type = reader.u32()
        length = reader.u64()
        values = []
        for index in range(length):
            value = read_value(reader, item_type, array_limit, full_arrays)
            if full_arrays or index < array_limit:
                values.append(value)
        return {
            "array_type": GGUF_VALUE_TYPES.get(item_type, f"unknown:{item_type}"),
            "length": length,
            "values_sampled": len(values),
            "values": values,
            "omitted": 0 if full_arrays else max(0, length - len(values)),
        }
    if value_type == 10:
        return reader.u64()
    if value_type == 11:
        return reader.i64()
    if value_type == 12:
        return reader.f64()
    raise ValueError(f"unsupported GGUF metadata value type: {value_type}")


def inspect_stream(
    fh: BinaryIO,
    tensor_limit: int | None = None,
    array_limit: int = 16,
    full_arrays: bool = False,
) -> dict[str, object]:
    reader = Reader(fh)
    magic = reader.read_exact(4)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")

    version = reader.u32()
    tensor_count = reader.u64()
    metadata_count = reader.u64()

    metadata = {}
    metadata_types = {}
    for _ in range(metadata_count):
        key = reader.string()
        value_type = reader.u32()
        metadata_types[key] = GGUF_VALUE_TYPES.get(value_type, f"unknown:{value_type}")
        metadata[key] = read_value(reader, value_type, array_limit, full_arrays)

    tensors = []
    max_tensors = tensor_count if tensor_limit is None else min(tensor_count, tensor_limit)
    for index in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        dims = [reader.u64() for _ in range(n_dims)]
        tensor_type = reader.u32()
        offset = reader.u64()
        if index < max_tensors:
            tensors.append(
                {
                    "name": name,
                    "dimensions": dims,
                    "type_id": tensor_type,
                    "type": GGML_TYPES.get(tensor_type, f"unknown:{tensor_type}"),
                    "offset": offset,
                }
            )

    tensor_info_end_offset = fh.tell()
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        alignment = 32
    tensor_data_offset = align_up(tensor_info_end_offset, alignment)

    return {
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "tensor_info_end_offset": tensor_info_end_offset,
        "tensor_data_offset": tensor_data_offset,
        "tensor_data_alignment": alignment,
        "metadata": metadata,
        "metadata_types": metadata_types,
        "tensors_sampled": len(tensors),
        "tensors": tensors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_self_test_blob() -> bytes:
    out = io.BytesIO()
    out.write(b"GGUF")
    out.write(struct.pack("<IQQ", 3, 1, 2))
    write_string(out, "general.architecture")
    out.write(struct.pack("<I", 8))
    write_string(out, "test")
    write_string(out, "general.context_length")
    out.write(struct.pack("<I", 4))
    out.write(struct.pack("<I", 2048))
    write_string(out, "blk.0.weight")
    out.write(struct.pack("<I", 2))
    out.write(struct.pack("<QQ", 16, 32))
    out.write(struct.pack("<IQ", 0, 0))
    return out.getvalue()


def write_string(out: io.BytesIO, value: str) -> None:
    data = value.encode("utf-8")
    out.write(struct.pack("<Q", len(data)))
    out.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GGUF metadata without loading tensors.")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--tensor-limit", type=int, default=32)
    parser.add_argument("--array-limit", type=int, default=16)
    parser.add_argument("--full-arrays", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = inspect_stream(
            io.BytesIO(build_self_test_blob()),
            tensor_limit=args.tensor_limit,
            array_limit=args.array_limit,
            full_arrays=args.full_arrays,
        )
        print(json.dumps(result, indent=2))
        return 0

    if not args.path:
        raise SystemExit("path is required unless --self-test is used")

    path = Path(args.path)
    with path.open("rb") as fh:
        result = inspect_stream(
            fh,
            tensor_limit=args.tensor_limit,
            array_limit=args.array_limit,
            full_arrays=args.full_arrays,
        )
    result["path"] = str(path)
    result["size_bytes"] = path.stat().st_size

    out_path = Path(args.output) if args.output else path.with_suffix(path.suffix + ".metadata.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
