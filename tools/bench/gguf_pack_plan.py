from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_block_formats import validate_tensor_storage, validation_summary  # noqa: E402
from tools.bench.gguf_tensor_plan import build_plan, query_gpu_free_bytes  # noqa: E402


DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_TARGET = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
LAYER_RE = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<role>.+)$")

PACK_ACTIONS = {
    "F32": "convert_or_keep_metadata_scalar",
    "F16": "copy_or_pack_fp16",
    "Q8_0": "needs_q8_block_reader",
    "Q5_K": "q5k_raw_resident_block_packer_plus_dequant_probe",
    "Q4_K": "q4k_resident_payload_metadata_packer",
    "Q3_K": "needs_k_quant_block_reader",
    "Q2_K": "needs_k_quant_block_reader",
    "IQ4_XS": "needs_iq_block_reader",
    "IQ4_NL": "needs_iq_block_reader",
    "IQ3_S": "iq3s_raw_resident_block_packer_plus_dequant_probe",
    "IQ2_S": "iq2s_raw_resident_block_packer_plus_dequant_probe",
    "IQ2_M": "needs_iq_block_reader",
}


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def mib(value: int | float) -> float:
    return value / (1024**2)


def default_model_path(root: Path) -> Path:
    return root / "models" / DEFAULT_REPO_SAFE / DEFAULT_TARGET


def tensor_role(name: str) -> str:
    match = LAYER_RE.match(name)
    if match:
        return match.group("role")
    if name.endswith(".weight"):
        return name.removesuffix(".weight")
    return name


def pack_action(tensor_type: object) -> str:
    return PACK_ACTIONS.get(str(tensor_type), "unknown_source_type")


def summarize_types(tensors: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    bytes_by_type: defaultdict[str, int] = defaultdict(int)
    for tensor in tensors:
        tensor_type = str(tensor.get("type"))
        counts[tensor_type] += 1
        bytes_by_type[tensor_type] += int(tensor.get("span_bytes") or 0)
    return [
        {
            "type": tensor_type,
            "tensor_count": counts[tensor_type],
            "span_bytes": bytes_by_type[tensor_type],
            "span_mib": mib(bytes_by_type[tensor_type]),
            "pack_action": pack_action(tensor_type),
        }
        for tensor_type in sorted(counts)
    ]


def summarize_roles(tensors: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    bytes_by_role: defaultdict[str, int] = defaultdict(int)
    types_by_role: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for tensor in tensors:
        role = tensor_role(str(tensor.get("name")))
        tensor_type = str(tensor.get("type"))
        counts[role] += 1
        bytes_by_role[role] += int(tensor.get("span_bytes") or 0)
        types_by_role[role][tensor_type] += 1
    return [
        {
            "role": role,
            "tensor_count": counts[role],
            "span_bytes": bytes_by_role[role],
            "span_mib": mib(bytes_by_role[role]),
            "types": dict(sorted(types_by_role[role].items())),
        }
        for role in sorted(counts)
    ]


def build_pack_plan(tensor_plan: dict[str, object]) -> dict[str, object]:
    tensors = tensor_plan.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("tensor plan does not contain full tensor rows")
    type_summary = summarize_types(tensors)
    role_summary = summarize_roles(tensors)
    unknown_types = [
        row["type"] for row in type_summary if row["pack_action"] == "unknown_source_type"
    ]
    alignment = int(tensor_plan["metadata"].get("tensor_data_alignment") or 32)
    block_validation = [
        validate_tensor_storage(tensor, alignment=alignment) for tensor in tensors
    ]
    block_summary = validation_summary(block_validation)
    return {
        "path": "gguf_pack_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": tensor_plan["model_path"],
        "architecture": tensor_plan["metadata"].get("architecture"),
        "tensor_count": tensor_plan["metadata"].get("tensor_count"),
        "layer_count": tensor_plan["layer_count"],
        "tensor_span_mib": tensor_plan["tensor_span_mib"],
        "strict_full_resident": tensor_plan["strict_full_resident"],
        "resident_layers_estimate": tensor_plan["resident_layers_estimate"],
        "type_summary": type_summary,
        "role_summary": role_summary,
        "unknown_types": unknown_types,
        "block_validation_summary": block_summary,
        "invalid_block_tensors": [
            row for row in block_validation if not row.get("valid")
        ][:32],
        "loader_gap": (
            "Q4_K tensors have an offline resident payload/metadata packer and native manifest consumption; "
            "IQ2_S tensors now have a raw resident block packer plus CUDA dequant matvec probe; "
            "IQ3_S tensors now have a raw resident block packer plus CUDA dequant matvec probe; "
            "Q5_K has a raw resident block packer plus CUDA dequant matvec probe. "
            "The remaining execution gap is full layer-role composition"
        ),
    }


def write_markdown(path: Path, plan: dict[str, object]) -> None:
    lines = [
        "# GGUF Pack Plan",
        "",
        f"Generated: `{plan['generated_at']}`",
        f"Model: `{plan['model_path']}`",
        f"Architecture: `{plan['architecture']}`",
        f"Tensor count: `{plan['tensor_count']}`",
        f"Layer count: `{plan['layer_count']}`",
        f"Tensor span: `{plan['tensor_span_mib']:.2f} MiB`",
        f"Strict full resident: `{plan['strict_full_resident']}`",
        f"Resident layers estimate: `{plan['resident_layers_estimate']}`",
        "",
        "## Block Validation",
        "",
        f"- Supported tensors: `{plan['block_validation_summary']['supported_tensors']}/{plan['block_validation_summary']['tensor_count']}`",
        f"- Valid tensor spans: `{plan['block_validation_summary']['valid_tensors']}/{plan['block_validation_summary']['tensor_count']}`",
        f"- Invalid tensors: `{plan['block_validation_summary']['invalid_tensors']}`",
        f"- Expected data bytes: `{plan['block_validation_summary']['expected_data_bytes']}`",
        f"- Padding bytes: `{plan['block_validation_summary']['padding_bytes']}`",
        f"- Max padding bytes: `{plan['block_validation_summary']['max_padding_bytes']}`",
        f"- Families: `{json.dumps(plan['block_validation_summary']['families'], sort_keys=True)}`",
        "",
        f"Loader gap: {plan['loader_gap']}",
        "",
        "## Source Types",
        "",
        "| type | tensors | span MiB | pack action |",
        "|---|---:|---:|---|",
    ]
    for row in plan["type_summary"]:
        lines.append(
            f"| {row['type']} | {row['tensor_count']} | {row['span_mib']:.2f} | {row['pack_action']} |"
        )
    lines.extend(
        [
            "",
            "## Tensor Roles",
            "",
            "| role | tensors | span MiB | source types |",
            "|---|---:|---:|---|",
        ]
    )
    for row in sorted(plan["role_summary"], key=lambda item: float(item["span_mib"]), reverse=True):
        lines.append(
            f"| {row['role']} | {row['tensor_count']} | {row['span_mib']:.2f} | `{json.dumps(row['types'], sort_keys=True)}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline native-loader pack plan from a GGUF.")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--kv-reserve-mib", type=int, default=768)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else default_model_path(root)
    free_vram = query_gpu_free_bytes() or 0
    tensor_plan = build_plan(
        model_path,
        free_vram_bytes=free_vram,
        wddm_guard_mib=args.wddm_guard_mib,
        kv_reserve_mib=args.kv_reserve_mib,
        workspace_mib=args.workspace_mib,
        safety_margin_mib=args.safety_margin_mib,
    )
    pack_plan = build_pack_plan(tensor_plan)

    stem = model_path.stem
    out_json = Path(args.output_json) if args.output_json else bench_dir / f"gguf_pack_plan_{stem}.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / f"gguf_pack_plan_{stem}.md"
    out_json.write_text(json.dumps(pack_plan, indent=2), encoding="utf-8")
    write_markdown(out_md, pack_plan)
    print(f"json={out_json}")
    print(f"markdown={out_md}")
    print(f"types={len(pack_plan['type_summary'])}")
    print(f"roles={len(pack_plan['role_summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
