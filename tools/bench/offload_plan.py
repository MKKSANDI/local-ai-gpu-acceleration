from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_tensor_plan import build_plan as build_tensor_plan


DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"


@dataclass(frozen=True)
class WeightBudget:
    free_vram_bytes: int
    wddm_guard_bytes: int
    kv_reserve_bytes: int
    workspace_bytes: int
    safety_margin_bytes: int

    @property
    def strict_weight_bytes(self) -> int:
        reserved = (
            self.wddm_guard_bytes
            + self.kv_reserve_bytes
            + self.workspace_bytes
            + self.safety_margin_bytes
        )
        return max(0, self.free_vram_bytes - reserved)

    def with_extra_guard(self, extra_guard_bytes: int) -> "WeightBudget":
        return WeightBudget(
            free_vram_bytes=self.free_vram_bytes,
            wddm_guard_bytes=self.wddm_guard_bytes + extra_guard_bytes,
            kv_reserve_bytes=self.kv_reserve_bytes,
            workspace_bytes=self.workspace_bytes,
            safety_margin_bytes=self.safety_margin_bytes,
        )


@dataclass(frozen=True)
class ModelRow:
    filename: str
    quant: str
    size_bytes: int
    local_path: str | None


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def gib(value: int | float) -> float:
    return value / (1024**3)


def mib(value: int | float) -> float:
    return value / (1024**2)


def quote(value: str) -> str:
    return f'"{value}"'


def estimate_ngl(resident_bytes: int, model_size_bytes: int, layer_count: int) -> int:
    if model_size_bytes <= 0 or layer_count <= 0:
        return 0
    fraction = min(1.0, max(0.0, resident_bytes / model_size_bytes))
    return min(layer_count, max(0, math.floor(fraction * layer_count)))


def profile_model(
    model: ModelRow,
    budget: WeightBudget,
    *,
    layer_count: int,
    min_balanced_fraction: float,
    latency_extra_guard_bytes: int,
) -> list[dict[str, object]]:
    strict_bytes = min(model.size_bytes, budget.strict_weight_bytes)
    strict_fit = model.size_bytes <= budget.strict_weight_bytes
    balanced_fraction = strict_bytes / model.size_bytes if model.size_bytes else 0.0

    latency_budget = budget.with_extra_guard(latency_extra_guard_bytes)
    latency_bytes = min(model.size_bytes, latency_budget.strict_weight_bytes)
    latency_fraction = latency_bytes / model.size_bytes if model.size_bytes else 0.0

    profiles = [
        {
            "profile": "strict_full_resident",
            "policy": "strict",
            "admit": strict_fit,
            "host_spill_allowed": False,
            "resident_weight_bytes": model.size_bytes if strict_fit else 0,
            "host_weight_bytes": 0 if strict_fit else model.size_bytes,
            "gpu_residency_fraction": 1.0 if strict_fit else 0.0,
            "estimated_ngl": layer_count if strict_fit else 0,
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "off", "--no-mmap"],
            "reason": "full weight residency fits" if strict_fit else "reject: full weight residency does not fit",
        },
        {
            "profile": "balanced_max_resident",
            "policy": "balanced",
            "admit": balanced_fraction >= min_balanced_fraction,
            "host_spill_allowed": False,
            "resident_weight_bytes": strict_bytes if balanced_fraction >= min_balanced_fraction else 0,
            "host_weight_bytes": model.size_bytes - strict_bytes
            if balanced_fraction >= min_balanced_fraction
            else model.size_bytes,
            "gpu_residency_fraction": balanced_fraction if balanced_fraction >= min_balanced_fraction else 0.0,
            "estimated_ngl": estimate_ngl(strict_bytes, model.size_bytes, layer_count)
            if balanced_fraction >= min_balanced_fraction
            else 0,
            "ctx_size": 2048,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1024"],
            "reason": "partial offload meets balanced GPU residency threshold"
            if balanced_fraction >= min_balanced_fraction
            else "reject: partial offload would be too RAM dominated",
        },
        {
            "profile": "latency_guard",
            "policy": "balanced",
            "admit": latency_fraction >= min_balanced_fraction,
            "host_spill_allowed": False,
            "resident_weight_bytes": latency_bytes if latency_fraction >= min_balanced_fraction else 0,
            "host_weight_bytes": model.size_bytes - latency_bytes
            if latency_fraction >= min_balanced_fraction
            else model.size_bytes,
            "gpu_residency_fraction": latency_fraction if latency_fraction >= min_balanced_fraction else 0.0,
            "estimated_ngl": estimate_ngl(latency_bytes, model.size_bytes, layer_count)
            if latency_fraction >= min_balanced_fraction
            else 0,
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "2048"],
            "reason": "partial offload fits with extra desktop latency guard"
            if latency_fraction >= min_balanced_fraction
            else "reject: extra guard leaves too little GPU residency",
        },
        {
            "profile": "overflow_explicit",
            "policy": "overflow",
            "admit": strict_bytes > 0,
            "host_spill_allowed": True,
            "resident_weight_bytes": strict_bytes,
            "host_weight_bytes": model.size_bytes - strict_bytes,
            "gpu_residency_fraction": balanced_fraction,
            "estimated_ngl": estimate_ngl(strict_bytes, model.size_bytes, layer_count),
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1536", "--cpu-moe"],
            "reason": "explicit overflow mode; RAM dependence is expected",
        },
    ]

    for profile in profiles:
        profile["filename"] = model.filename
        profile["quant"] = model.quant
        profile["model_size_gib"] = gib(model.size_bytes)
        profile["resident_weight_gib"] = gib(int(profile["resident_weight_bytes"]))
        profile["host_weight_gib"] = gib(int(profile["host_weight_bytes"]))
    return profiles


def profile_tensor_model(
    model: ModelRow,
    tensor_plan: dict[str, object],
    latency_tensor_plan: dict[str, object],
    *,
    min_balanced_fraction: float,
) -> list[dict[str, object]]:
    weight_bytes = int(tensor_plan["tensor_span_bytes"])
    layer_count = int(tensor_plan["layer_count"])
    strict_fit = bool(tensor_plan["strict_full_resident"])

    balanced_bytes = min(weight_bytes, int(tensor_plan["resident_weight_bytes_estimate"]))
    balanced_fraction = balanced_bytes / weight_bytes if weight_bytes else 0.0
    balanced_layers = int(tensor_plan["resident_layers_estimate"])

    latency_bytes = min(weight_bytes, int(latency_tensor_plan["resident_weight_bytes_estimate"]))
    latency_fraction = latency_bytes / weight_bytes if weight_bytes else 0.0
    latency_layers = int(latency_tensor_plan["resident_layers_estimate"])

    profiles = [
        {
            "profile": "strict_full_resident",
            "policy": "strict",
            "admit": strict_fit,
            "host_spill_allowed": False,
            "resident_weight_bytes": weight_bytes if strict_fit else 0,
            "host_weight_bytes": 0 if strict_fit else weight_bytes,
            "gpu_residency_fraction": 1.0 if strict_fit else 0.0,
            "estimated_ngl": layer_count if strict_fit else 0,
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "off", "--no-mmap"],
            "reason": "full tensor span residency fits"
            if strict_fit
            else "reject: full tensor span residency does not fit",
        },
        {
            "profile": "balanced_max_resident",
            "policy": "balanced",
            "admit": balanced_fraction >= min_balanced_fraction,
            "host_spill_allowed": False,
            "resident_weight_bytes": balanced_bytes if balanced_fraction >= min_balanced_fraction else 0,
            "host_weight_bytes": weight_bytes - balanced_bytes
            if balanced_fraction >= min_balanced_fraction
            else weight_bytes,
            "gpu_residency_fraction": balanced_fraction if balanced_fraction >= min_balanced_fraction else 0.0,
            "estimated_ngl": balanced_layers if balanced_fraction >= min_balanced_fraction else 0,
            "ctx_size": 2048,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1024"],
            "reason": "tensor-derived partial offload meets balanced GPU residency threshold"
            if balanced_fraction >= min_balanced_fraction
            else "reject: tensor-derived partial offload would be too RAM dominated",
        },
        {
            "profile": "latency_guard",
            "policy": "balanced",
            "admit": latency_fraction >= min_balanced_fraction,
            "host_spill_allowed": False,
            "resident_weight_bytes": latency_bytes if latency_fraction >= min_balanced_fraction else 0,
            "host_weight_bytes": weight_bytes - latency_bytes
            if latency_fraction >= min_balanced_fraction
            else weight_bytes,
            "gpu_residency_fraction": latency_fraction if latency_fraction >= min_balanced_fraction else 0.0,
            "estimated_ngl": latency_layers if latency_fraction >= min_balanced_fraction else 0,
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "2048"],
            "reason": "tensor-derived partial offload fits with extra desktop latency guard"
            if latency_fraction >= min_balanced_fraction
            else "reject: extra guard leaves too little tensor-derived GPU residency",
        },
        {
            "profile": "overflow_explicit",
            "policy": "overflow",
            "admit": balanced_bytes > 0,
            "host_spill_allowed": True,
            "resident_weight_bytes": balanced_bytes,
            "host_weight_bytes": weight_bytes - balanced_bytes,
            "gpu_residency_fraction": balanced_fraction,
            "estimated_ngl": balanced_layers,
            "ctx_size": 1024,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1536", "--cpu-moe"],
            "reason": "explicit overflow mode using tensor-derived residency; RAM dependence is expected",
        },
    ]

    for profile in profiles:
        profile["filename"] = model.filename
        profile["quant"] = model.quant
        profile["model_size_gib"] = gib(model.size_bytes)
        profile["weight_span_gib"] = gib(weight_bytes)
        profile["resident_weight_gib"] = gib(int(profile["resident_weight_bytes"]))
        profile["host_weight_gib"] = gib(int(profile["host_weight_bytes"]))
        profile["layer_count"] = layer_count
        profile["layer_count_basis"] = "tensor"
        profile["local_model_path"] = model.local_path
    return profiles


def build_llamacpp_command(
    exe: Path,
    model_path: Path,
    profile: dict[str, object],
    *,
    prompt: str,
    n_predict: int,
) -> str:
    args = [
        quote(str(exe)),
        "-m",
        quote(str(model_path)),
        "-p",
        quote(prompt),
        "-n",
        str(n_predict),
        "-c",
        str(profile["ctx_size"]),
        "-ngl",
        str(profile["estimated_ngl"]),
        "-ctk",
        str(profile["cache_type_k"]),
        "-ctv",
        str(profile["cache_type_v"]),
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
    ]
    args.extend(str(flag) for flag in profile["extra_flags"])
    return " ".join(args)


def load_models(plan: dict[str, object]) -> list[ModelRow]:
    rows = []
    for item in plan["models"]:
        rows.append(
            ModelRow(
                filename=str(item["filename"]),
                quant=str(item["quant"]),
                size_bytes=int(item["size_bytes"]),
                local_path=str(item["local"]) if item.get("local") else None,
            )
        )
    return rows


def local_tensor_profiles(
    model: ModelRow,
    *,
    free_vram_bytes: int,
    wddm_guard_mib: int,
    kv_reserve_mib: int,
    workspace_mib: int,
    safety_margin_mib: int,
    latency_extra_guard_mib: int,
    min_balanced_fraction: float,
) -> list[dict[str, object]] | None:
    if not model.local_path:
        return None
    path = Path(model.local_path)
    if not path.exists():
        return None
    tensor_plan = build_tensor_plan(
        path,
        free_vram_bytes=free_vram_bytes,
        wddm_guard_mib=wddm_guard_mib,
        kv_reserve_mib=kv_reserve_mib,
        workspace_mib=workspace_mib,
        safety_margin_mib=safety_margin_mib,
    )
    latency_tensor_plan = build_tensor_plan(
        path,
        free_vram_bytes=free_vram_bytes,
        wddm_guard_mib=wddm_guard_mib,
        kv_reserve_mib=kv_reserve_mib,
        workspace_mib=workspace_mib,
        safety_margin_mib=safety_margin_mib + latency_extra_guard_mib,
    )
    return profile_tensor_model(
        model,
        tensor_plan,
        latency_tensor_plan,
        min_balanced_fraction=min_balanced_fraction,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build target GGUF offload and residency profiles.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model-plan", default=None)
    parser.add_argument("--assumed-layer-count", type=int, default=48)
    parser.add_argument("--balanced-min-gpu-fraction", type=float, default=0.45)
    parser.add_argument("--latency-extra-guard-mib", type=int, default=1024)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--prompt", default="Write one sentence about GPU residency.")
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    plan_path = Path(args.model_plan) if args.model_plan else bench_dir / "model_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    models = load_models(plan)

    budget = WeightBudget(
        free_vram_bytes=int(plan["gpu"]["memory_free_bytes"]),
        wddm_guard_bytes=int(plan["wddm_guard_mib"]) * 1024**2,
        kv_reserve_bytes=int(plan["kv_reserve_mib"]) * 1024**2,
        workspace_bytes=int(plan["workspace_mib"]) * 1024**2,
        safety_margin_bytes=args.safety_margin_mib * 1024**2,
    )
    llama_exe = root / "third_party" / "llama.cpp" / "llama-completion.exe"
    model_dir = root / "models" / DEFAULT_REPO_SAFE

    rows = []
    for model in models:
        model_path = model_dir / model.filename
        profiles = local_tensor_profiles(
            model,
            free_vram_bytes=budget.free_vram_bytes,
            wddm_guard_mib=int(plan["wddm_guard_mib"]),
            kv_reserve_mib=int(plan["kv_reserve_mib"]),
            workspace_mib=int(plan["workspace_mib"]),
            safety_margin_mib=args.safety_margin_mib,
            latency_extra_guard_mib=args.latency_extra_guard_mib,
            min_balanced_fraction=args.balanced_min_gpu_fraction,
        )
        if profiles is None:
            profiles = profile_model(
                model,
                budget,
                layer_count=args.assumed_layer_count,
                min_balanced_fraction=args.balanced_min_gpu_fraction,
                latency_extra_guard_bytes=args.latency_extra_guard_mib * 1024**2,
            )
            for profile in profiles:
                profile["weight_span_gib"] = profile["model_size_gib"]
                profile["layer_count"] = args.assumed_layer_count
                profile["layer_count_basis"] = "assumed"
                profile["local_model_path"] = model.local_path
        for profile in profiles:
            profile["command"] = build_llamacpp_command(
                llama_exe,
                model_path,
                profile,
                prompt=args.prompt,
                n_predict=args.n_predict,
            )
            rows.append(profile)

    target = next(
        (
            row
            for row in rows
            if row["filename"] == plan["target"]["filename"]
            and row["profile"] == "overflow_explicit"
        ),
        None,
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path),
        "repo": plan["repo"],
        "assumed_layer_count": args.assumed_layer_count,
        "balanced_min_gpu_fraction": args.balanced_min_gpu_fraction,
        "latency_extra_guard_mib": args.latency_extra_guard_mib,
        "safety_margin_mib": args.safety_margin_mib,
        "budget": {
            "free_vram_gib": gib(budget.free_vram_bytes),
            "strict_weight_budget_gib": gib(budget.strict_weight_bytes),
            "latency_guard_weight_budget_gib": gib(
                budget.with_extra_guard(args.latency_extra_guard_mib * 1024**2).strict_weight_bytes
            ),
        },
        "target_overflow_profile": target,
        "note": (
            "Local GGUF rows use tensor-derived layer spans and complete-layer residency estimates. "
            "Rows without a local file still use the assumed layer count."
        ),
        "rows": rows,
    }

    out_json = Path(args.output_json) if args.output_json else bench_dir / "offload_plan.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / "offload_plan.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Offload Plan",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        f"- Source plan: `{plan_path}`",
        f"- Assumed layer count for non-local rows: `{args.assumed_layer_count}`",
        f"- Strict weight budget: `{result['budget']['strict_weight_budget_gib']:.2f} GiB`",
        f"- Latency-guard weight budget: `{result['budget']['latency_guard_weight_budget_gib']:.2f} GiB`",
        f"- Note: {result['note']}",
        "",
        "| quant | basis | size GiB | profile | admit | est ngl | layers | GPU GiB | host GiB | GPU fraction | spill | reason |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["quant"]),
                    str(row["layer_count_basis"]),
                    f"{row['model_size_gib']:.2f}",
                    str(row["profile"]),
                    str(row["admit"]),
                    str(row["estimated_ngl"]),
                    str(row["layer_count"]),
                    f"{row['resident_weight_gib']:.2f}",
                    f"{row['host_weight_gib']:.2f}",
                    f"{row['gpu_residency_fraction']:.2f}",
                    str(row["host_spill_allowed"]),
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"json={out_json}")
    print(f"markdown={out_md}")
    if target:
        print(
            "target overflow estimate: "
            f"ngl={target['estimated_ngl']} "
            f"gpu={target['resident_weight_gib']:.2f}GiB "
            f"host={target['host_weight_gib']:.2f}GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
