from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_inspect import inspect_stream


LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def mib(value: int | float) -> float:
    return value / (1024**2)


def gib(value: int | float) -> float:
    return value / (1024**3)


def query_gpu_free_bytes() -> int | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    first = raw.splitlines()[0].strip() if raw else ""
    return int(float(first)) * 1024 * 1024 if first else None


def tensor_layer(name: str) -> int | None:
    match = LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def tensor_spans(metadata: dict[str, object], file_size: int) -> list[dict[str, object]]:
    tensors = metadata["tensors"]
    if not isinstance(tensors, list):
        raise ValueError("GGUF metadata did not contain tensor list")
    data_offset = int(metadata["tensor_data_offset"])
    sorted_tensors = sorted(tensors, key=lambda item: int(item["offset"]))
    rows: list[dict[str, object]] = []
    for index, tensor in enumerate(sorted_tensors):
        offset = int(tensor["offset"])
        next_offset = (
            int(sorted_tensors[index + 1]["offset"])
            if index + 1 < len(sorted_tensors)
            else file_size - data_offset
        )
        span_bytes = max(0, next_offset - offset)
        name = str(tensor["name"])
        layer = tensor_layer(name)
        rows.append(
            {
                "name": name,
                "layer": layer,
                "span_bytes": span_bytes,
                "span_mib": mib(span_bytes),
                "offset": offset,
                "absolute_offset": data_offset + offset,
                "type": tensor.get("type"),
                "type_id": tensor.get("type_id"),
                "dimensions": tensor.get("dimensions"),
            }
        )
    return rows


def build_plan(
    path: Path,
    *,
    free_vram_bytes: int,
    wddm_guard_mib: int,
    kv_reserve_mib: int,
    workspace_mib: int,
    safety_margin_mib: int,
) -> dict[str, object]:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        metadata = inspect_stream(fh, tensor_limit=None, array_limit=4, full_arrays=False)
    tensors = tensor_spans(metadata, file_size)

    non_layer_bytes = sum(int(row["span_bytes"]) for row in tensors if row["layer"] is None)
    tensor_span_bytes = sum(int(row["span_bytes"]) for row in tensors)
    layer_bytes: dict[int, int] = defaultdict(int)
    layer_tensor_counts: dict[int, int] = defaultdict(int)
    for row in tensors:
        layer = row["layer"]
        if layer is None:
            continue
        layer_bytes[layer] += int(row["span_bytes"])
        layer_tensor_counts[layer] += 1

    layers = [
        {
            "layer": layer,
            "tensor_count": layer_tensor_counts[layer],
            "span_bytes": layer_bytes[layer],
            "span_mib": mib(layer_bytes[layer]),
        }
        for layer in sorted(layer_bytes)
    ]

    usable_weight_bytes = max(
        0,
        free_vram_bytes
        - wddm_guard_mib * 1024**2
        - kv_reserve_mib * 1024**2
        - workspace_mib * 1024**2
        - safety_margin_mib * 1024**2,
    )
    remaining = max(0, usable_weight_bytes - non_layer_bytes)
    resident_layers = 0
    resident_layer_bytes = 0
    for layer in layers:
        size = int(layer["span_bytes"])
        if resident_layer_bytes + size > remaining:
            break
        resident_layer_bytes += size
        resident_layers += 1

    resident_weight_bytes = min(non_layer_bytes + resident_layer_bytes, tensor_span_bytes)
    return {
        "path": "gguf_tensor_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(path),
        "model_size_bytes": file_size,
        "model_size_mib": mib(file_size),
        "metadata": {
            "architecture": metadata.get("metadata", {}).get("general.architecture"),
            "name": metadata.get("metadata", {}).get("general.name"),
            "tensor_count": metadata.get("tensor_count"),
            "metadata_count": metadata.get("metadata_count"),
            "tensor_data_offset": metadata.get("tensor_data_offset"),
            "tensor_data_alignment": metadata.get("tensor_data_alignment"),
        },
        "budget": {
            "free_vram_bytes": free_vram_bytes,
            "free_vram_gib": gib(free_vram_bytes),
            "wddm_guard_mib": wddm_guard_mib,
            "kv_reserve_mib": kv_reserve_mib,
            "workspace_mib": workspace_mib,
            "safety_margin_mib": safety_margin_mib,
            "usable_weight_bytes": usable_weight_bytes,
            "usable_weight_mib": mib(usable_weight_bytes),
        },
        "non_layer_bytes": non_layer_bytes,
        "non_layer_mib": mib(non_layer_bytes),
        "tensor_span_bytes": tensor_span_bytes,
        "tensor_span_mib": mib(tensor_span_bytes),
        "layer_count": len(layers),
        "resident_layers_estimate": resident_layers,
        "resident_weight_bytes_estimate": resident_weight_bytes,
        "resident_weight_mib_estimate": mib(resident_weight_bytes),
        "gpu_residency_fraction_estimate": resident_weight_bytes / tensor_span_bytes
        if tensor_span_bytes
        else 0.0,
        "strict_full_resident": tensor_span_bytes <= usable_weight_bytes,
        "layers": layers,
        "largest_tensors": sorted(tensors, key=lambda row: int(row["span_bytes"]), reverse=True)[:16],
        "tensors": tensors,
    }


def write_markdown(path: Path, plan: dict[str, object], tensor_limit: int) -> None:
    lines = [
        "# GGUF Tensor Plan",
        "",
        f"Generated: `{plan['generated_at']}`",
        "",
        f"- Model: `{plan['model_path']}`",
        f"- Size: `{plan['model_size_mib']:.2f} MiB`",
        f"- Architecture: `{plan['metadata'].get('architecture')}`",
        f"- Tensor count: `{plan['metadata'].get('tensor_count')}`",
        f"- Tensor span bytes: `{plan['tensor_span_mib']:.2f} MiB`",
        f"- Layer count detected: `{plan['layer_count']}`",
        f"- Non-layer bytes: `{plan['non_layer_mib']:.2f} MiB`",
        f"- Usable weight budget: `{plan['budget']['usable_weight_mib']:.2f} MiB`",
        f"- Strict full resident: `{plan['strict_full_resident']}`",
        f"- Resident layers estimate: `{plan['resident_layers_estimate']}`",
        f"- Resident weight estimate: `{plan['resident_weight_mib_estimate']:.2f} MiB`",
        f"- GPU residency estimate: `{plan['gpu_residency_fraction_estimate']:.3f}`",
        "",
        "## Layers",
        "",
        "| layer | tensors | span MiB |",
        "|---:|---:|---:|",
    ]
    for layer in plan["layers"]:
        lines.append(f"| {layer['layer']} | {layer['tensor_count']} | {layer['span_mib']:.4f} |")

    lines.extend(
        [
            "",
            "## Largest Tensors",
            "",
            "| name | layer | type | span MiB | dimensions |",
            "|---|---:|---|---:|---|",
        ]
    )
    for tensor in plan["largest_tensors"][:tensor_limit]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(tensor["name"]),
                    "" if tensor["layer"] is None else str(tensor["layer"]),
                    str(tensor["type"]),
                    f"{tensor['span_mib']:.4f}",
                    str(tensor["dimensions"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tensor/layer residency plan for a local GGUF.")
    parser.add_argument("model", nargs="?")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--kv-reserve-mib", type=int, default=768)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--free-vram-mib", type=int, default=None)
    parser.add_argument("--largest-tensor-limit", type=int, default=16)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    default_model = (
        root
        / "models"
        / "smoke"
        / "aladar__llama-2-tiny-random-GGUF"
        / "llama-2-tiny-random.gguf"
    )
    model = Path(args.model) if args.model else default_model
    free_vram_bytes = (
        args.free_vram_mib * 1024**2
        if args.free_vram_mib is not None
        else query_gpu_free_bytes()
    )
    if free_vram_bytes is None:
        raise SystemExit("could not determine free VRAM; pass --free-vram-mib")

    plan = build_plan(
        model,
        free_vram_bytes=free_vram_bytes,
        wddm_guard_mib=args.wddm_guard_mib,
        kv_reserve_mib=args.kv_reserve_mib,
        workspace_mib=args.workspace_mib,
        safety_margin_mib=args.safety_margin_mib,
    )

    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.stem)
    out_json = Path(args.output_json) if args.output_json else bench_dir / f"gguf_tensor_plan_{safe_name}.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / f"gguf_tensor_plan_{safe_name}.md"
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_markdown(out_md, plan, tensor_limit=args.largest_tensor_limit)

    print(f"json={out_json}")
    print(f"markdown={out_md}")
    print(
        "resident_layers="
        f"{plan['resident_layers_estimate']}/{plan['layer_count']} "
        f"strict={plan['strict_full_resident']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
