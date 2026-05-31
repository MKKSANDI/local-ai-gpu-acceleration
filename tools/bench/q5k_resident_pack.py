from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_block_formats import (  # noqa: E402
    BLOCK_FORMATS,
    element_count,
    expected_data_bytes,
)
from tools.bench.native_harness import (  # noqa: E402
    DEFAULT_REPO_SAFE,
    DEFAULT_TARGET,
    sanitize_label,
)


SUPPORTED_TYPES = ("Q5_K",)
LAYER_RE = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<role>.+)$")


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def default_model(root: Path) -> Path:
    return root / "models" / DEFAULT_REPO_SAFE / DEFAULT_TARGET


def default_tensor_plan(root: Path) -> Path:
    return root / "benchmarks" / f"gguf_tensor_plan_{DEFAULT_TARGET.removesuffix('.gguf')}.json"


def checksum(data: bytes | bytearray) -> int:
    return int(sum(data))


def tensor_role(name: str) -> tuple[int | None, str]:
    match = LAYER_RE.match(name)
    if match:
        return int(match.group("layer")), match.group("role")
    if name.endswith(".weight"):
        return None, name.removesuffix(".weight")
    return None, name


def load_tensor_plan(path: Path) -> list[dict[str, object]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    tensors = plan.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError(f"{path} does not contain tensor rows")
    return [tensor for tensor in tensors if str(tensor.get("type")) == "Q5_K"]


def q5k_tensor_shape(tensor: dict[str, object]) -> tuple[int, int]:
    tensor_type = str(tensor.get("type"))
    fmt = BLOCK_FORMATS.get(tensor_type)
    if fmt is None or tensor_type not in SUPPORTED_TYPES:
        raise ValueError(f"{tensor.get('name')} has unsupported Q5_K type: {tensor_type}")
    values = element_count(tensor.get("dimensions"))
    if values % fmt.block_size != 0:
        raise ValueError(
            f"{tensor.get('name')} element count {values} is not divisible by {fmt.block_size}"
        )
    return values, values // fmt.block_size


def read_tensor_bytes(model: Path, tensor: dict[str, object]) -> bytes:
    expected = expected_data_bytes("Q5_K", tensor.get("dimensions"))
    if expected is None:
        raise ValueError(f"{tensor['name']} has unsupported tensor type {tensor.get('type')}")
    span = int(tensor.get("span_bytes") or 0)
    if span < expected:
        raise ValueError(f"{tensor['name']} span is smaller than expected Q5_K bytes")
    absolute_offset = int(tensor["absolute_offset"])
    with model.open("rb") as handle:
        handle.seek(absolute_offset)
        data = handle.read(expected)
    if len(data) != expected:
        raise ValueError(f"short read for {tensor['name']}: expected {expected}, got {len(data)}")
    return data


def pack_tensor(*, model: Path, tensor: dict[str, object], out_dir: Path) -> dict[str, object]:
    values, block_count = q5k_tensor_shape(tensor)
    fmt = BLOCK_FORMATS["Q5_K"]
    payload = read_tensor_bytes(model, tensor)
    label = sanitize_label(str(tensor["name"]))
    payload_path = out_dir / f"{label}.q5_k_raw_blocks.bin"
    payload_path.write_bytes(payload)
    layer, role = tensor_role(str(tensor["name"]))
    return {
        "name": tensor["name"],
        "type": "Q5_K",
        "layer": layer,
        "role": role,
        "dimensions": tensor.get("dimensions"),
        "element_count": values,
        "block_size": fmt.block_size,
        "block_type_size": fmt.type_size,
        "block_count": block_count,
        "source_offset": tensor["absolute_offset"],
        "source_bytes": len(payload),
        "source_checksum": checksum(payload),
        "payload_path": str(payload_path),
        "payload_bytes": len(payload),
        "payload_checksum": checksum(payload),
        "resident_bytes": len(payload),
        "policy": "raw-blocks",
        "runtime_kernel": "q5_k_raw_pending",
    }


def write_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Q5_K Resident Pack",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Model: `{manifest['model']}`",
        f"Tensor plan: `{manifest['tensor_plan']}`",
        f"Output dir: `{manifest['output_dir']}`",
        f"Tensor count: `{manifest['tensor_count']}`",
        f"Resident bytes: `{manifest['resident_bytes']}`",
        "",
        "| tensor | dims | blocks | resident B | checksum |",
        "|---|---|---:|---:|---:|",
    ]
    for tensor in manifest["tensors"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(tensor["name"]),
                    str(tensor["dimensions"]),
                    str(tensor["block_count"]),
                    str(tensor["resident_bytes"]),
                    str(tensor["source_checksum"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack target GGUF Q5_K tensors into raw resident block files."
    )
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tensor-plan", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tensor", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    model = Path(args.model) if args.model else default_model(root)
    tensor_plan = Path(args.tensor_plan) if args.tensor_plan else default_tensor_plan(root)
    stamp = int(time.time())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "packs" / "q5k" / model.stem / str(stamp)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_tensor_plan(tensor_plan)
    if args.tensor:
        requested = set(args.tensor)
        tensors = [tensor for tensor in tensors if str(tensor.get("name")) in requested]
    if args.limit is not None:
        tensors = tensors[: args.limit]
    if not tensors:
        raise SystemExit("no Q5_K tensors selected")

    packed = [pack_tensor(model=model, tensor=tensor, out_dir=out_dir) for tensor in tensors]
    manifest = {
        "path": "q5k_resident_pack",
        "generated_at": generated_at,
        "model": str(model),
        "tensor_plan": str(tensor_plan),
        "output_dir": str(out_dir),
        "types": ["Q5_K"],
        "tensor_count": len(packed),
        "source_bytes": sum(int(row["source_bytes"]) for row in packed),
        "payload_bytes": sum(int(row["payload_bytes"]) for row in packed),
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
