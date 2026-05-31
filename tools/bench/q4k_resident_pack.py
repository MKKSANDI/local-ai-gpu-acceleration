from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.native_harness import (  # noqa: E402
    DEFAULT_REPO_SAFE,
    DEFAULT_TARGET,
    sanitize_label,
)


Q4K_VALUES = 256
Q4K_BYTES = 144
Q4K_PAYLOAD_BYTES = 128
Q4K_META_BYTES = 16


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def default_model(root: Path) -> Path:
    return root / "models" / DEFAULT_REPO_SAFE / DEFAULT_TARGET


def default_tensor_plan(root: Path) -> Path:
    return root / "benchmarks" / f"gguf_tensor_plan_{DEFAULT_TARGET.removesuffix('.gguf')}.json"


def checksum(data: bytes | bytearray) -> int:
    return int(sum(data))


def fp16_to_float(raw: bytes) -> float:
    return float(struct.unpack("<e", raw)[0])


def float_to_fp16(value: float) -> bytes:
    return struct.pack("<e", value)


def scale_min_k4(index: int, packed: bytes | bytearray | memoryview) -> tuple[int, int]:
    if index < 4:
        return packed[index] & 63, packed[index + 4] & 63
    scale = (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4)
    min_value = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)
    return scale, min_value


def q4k_shape(tensor: dict[str, object]) -> tuple[int, int, int]:
    dims = [int(value) for value in tensor.get("dimensions") or []]
    if len(dims) != 2:
        raise ValueError(f"{tensor.get('name')} is not a 2D Q4_K tensor")
    cols, rows = dims[0], dims[1]
    if cols <= 0 or rows <= 0 or cols % Q4K_VALUES != 0:
        raise ValueError(f"{tensor.get('name')} has unsupported Q4_K dimensions: {dims}")
    return rows, cols, rows * (cols // Q4K_VALUES)


def q4k_expected_bytes(tensor: dict[str, object]) -> int:
    _, _, block_count = q4k_shape(tensor)
    return block_count * Q4K_BYTES


def auto_policy(rows: int, cols: int, block_count: int) -> str:
    del rows, cols
    if block_count <= 4096:
        return "split-compact"
    return "split-predecoded"


def metadata_format(policy: str) -> str:
    return {
        "split-predecoded": "fp32-products",
        "split-half": "fp16-products",
        "split-compact": "compact-u8",
        "split-native": "native16",
    }[policy]


def metadata_bytes_per_block(policy: str) -> int:
    return {
        "split-predecoded": 8 * 2 * 4,
        "split-half": 8 * 2 * 2,
        "split-compact": 20,
        "split-native": 16,
    }[policy]


def build_metadata(blocks: bytes, policy: str) -> bytes:
    out = bytearray()
    for offset in range(0, len(blocks), Q4K_BYTES):
        block = memoryview(blocks)[offset : offset + Q4K_BYTES]
        if policy == "split-native":
            out.extend(block[:Q4K_META_BYTES])
            continue

        d_raw = bytes(block[0:2])
        dmin_raw = bytes(block[2:4])
        scales = block[4:16]
        if policy == "split-compact":
            out.extend(d_raw)
            out.extend(dmin_raw)
            unpacked_scales = []
            unpacked_mins = []
            for index in range(8):
                scale, min_value = scale_min_k4(index, scales)
                unpacked_scales.append(scale)
                unpacked_mins.append(min_value)
            out.extend(unpacked_scales)
            out.extend(unpacked_mins)
            continue

        d = fp16_to_float(d_raw)
        dmin = fp16_to_float(dmin_raw)
        for index in range(8):
            scale, min_value = scale_min_k4(index, scales)
            scale_product = d * float(scale)
            min_product = dmin * float(min_value)
            if policy == "split-predecoded":
                out.extend(struct.pack("<f", scale_product))
                out.extend(struct.pack("<f", min_product))
            elif policy == "split-half":
                out.extend(float_to_fp16(scale_product))
                out.extend(float_to_fp16(min_product))
            else:
                raise ValueError(f"unknown Q4_K metadata policy: {policy}")
    return bytes(out)


def extract_payload(blocks: bytes) -> bytes:
    out = bytearray()
    for offset in range(0, len(blocks), Q4K_BYTES):
        out.extend(blocks[offset + Q4K_META_BYTES : offset + Q4K_BYTES])
    return bytes(out)


def load_tensor_plan(path: Path) -> list[dict[str, object]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    tensors = plan.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError(f"{path} does not contain tensor rows")
    return [tensor for tensor in tensors if tensor.get("type") == "Q4_K"]


def read_tensor_blocks(model: Path, tensor: dict[str, object]) -> bytes:
    absolute_offset = int(tensor["absolute_offset"])
    expected = q4k_expected_bytes(tensor)
    span = int(tensor.get("span_bytes") or 0)
    if span < expected:
        raise ValueError(f"{tensor['name']} span is smaller than expected Q4_K bytes")
    with model.open("rb") as handle:
        handle.seek(absolute_offset)
        data = handle.read(expected)
    if len(data) != expected:
        raise ValueError(f"short read for {tensor['name']}: expected {expected}, got {len(data)}")
    return data


def pack_tensor(
    *,
    model: Path,
    tensor: dict[str, object],
    out_dir: Path,
    policy_arg: str,
    write_blocks: bool,
) -> dict[str, object]:
    rows, cols, block_count = q4k_shape(tensor)
    policy = auto_policy(rows, cols, block_count) if policy_arg == "auto" else policy_arg
    blocks = read_tensor_blocks(model, tensor)
    payload = extract_payload(blocks)
    metadata = build_metadata(blocks, policy)

    if len(payload) != block_count * Q4K_PAYLOAD_BYTES:
        raise ValueError(f"{tensor['name']} payload length mismatch")
    expected_meta = block_count * metadata_bytes_per_block(policy)
    if len(metadata) != expected_meta:
        raise ValueError(f"{tensor['name']} metadata length mismatch")

    label = sanitize_label(str(tensor["name"]))
    payload_path = out_dir / f"{label}.q4k_payload.bin"
    meta_path = out_dir / f"{label}.q4k_meta.{metadata_format(policy)}.bin"
    payload_path.write_bytes(payload)
    meta_path.write_bytes(metadata)

    blocks_path = None
    if write_blocks:
        blocks_path = out_dir / f"{label}.q4k_blocks.bin"
        blocks_path.write_bytes(blocks)

    return {
        "name": tensor["name"],
        "dimensions": tensor.get("dimensions"),
        "rows": rows,
        "cols": cols,
        "block_count": block_count,
        "source_offset": tensor["absolute_offset"],
        "source_bytes": len(blocks),
        "source_checksum": checksum(blocks),
        "policy": policy,
        "metadata_format": metadata_format(policy),
        "payload_path": str(payload_path),
        "payload_bytes": len(payload),
        "payload_checksum": checksum(payload),
        "metadata_path": str(meta_path),
        "metadata_bytes": len(metadata),
        "metadata_checksum": checksum(metadata),
        "blocks_path": "" if blocks_path is None else str(blocks_path),
        "resident_bytes": len(payload) + len(metadata),
        "compression_ratio_vs_fp16": (rows * cols * 2) / float(len(payload) + len(metadata)),
        "runtime_kernel": policy,
    }


def write_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Q4_K Resident Pack",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Model: `{manifest['model']}`",
        f"Output dir: `{manifest['output_dir']}`",
        f"Policy: `{manifest['policy']}`",
        f"Tensor count: `{manifest['tensor_count']}`",
        f"Resident bytes: `{manifest['resident_bytes']}`",
        f"Payload bytes: `{manifest['payload_bytes']}`",
        f"Metadata bytes: `{manifest['metadata_bytes']}`",
        "",
        "| tensor | dims | policy | payload B | meta B | resident B | checksum |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for tensor in manifest["tensors"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(tensor["name"]),
                    str(tensor["dimensions"]),
                    str(tensor["policy"]),
                    str(tensor["payload_bytes"]),
                    str(tensor["metadata_bytes"]),
                    str(tensor["resident_bytes"]),
                    str(tensor["source_checksum"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack target GGUF Q4_K tensors into resident payload/metadata files.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tensor-plan", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--policy",
        choices=["auto", "split-predecoded", "split-half", "split-compact", "split-native"],
        default="auto",
    )
    parser.add_argument("--tensor", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-blocks", action="store_true")
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    model = Path(args.model) if args.model else default_model(root)
    tensor_plan = Path(args.tensor_plan) if args.tensor_plan else default_tensor_plan(root)
    stamp = int(time.time())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "packs" / "q4k" / model.stem / str(stamp)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_tensor_plan(tensor_plan)
    if args.tensor:
        requested = set(args.tensor)
        tensors = [tensor for tensor in tensors if str(tensor.get("name")) in requested]
    if args.limit is not None:
        tensors = tensors[: args.limit]
    if not tensors:
        raise SystemExit("no Q4_K tensors selected")

    packed = [
        pack_tensor(
            model=model,
            tensor=tensor,
            out_dir=out_dir,
            policy_arg=args.policy,
            write_blocks=args.write_blocks,
        )
        for tensor in tensors
    ]
    manifest = {
        "path": "q4k_resident_pack",
        "generated_at": generated_at,
        "model": str(model),
        "tensor_plan": str(tensor_plan),
        "output_dir": str(out_dir),
        "policy": args.policy,
        "write_blocks": args.write_blocks,
        "tensor_count": len(packed),
        "source_bytes": sum(int(row["source_bytes"]) for row in packed),
        "payload_bytes": sum(int(row["payload_bytes"]) for row in packed),
        "metadata_bytes": sum(int(row["metadata_bytes"]) for row in packed),
        "resident_bytes": sum(int(row["resident_bytes"]) for row in packed),
        "tensors": packed,
    }
    json_path = out_dir / "manifest.json"
    md_path = out_dir / "manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(md_path, manifest)
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(f"tensors={manifest['tensor_count']}")
    print(f"resident_bytes={manifest['resident_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
