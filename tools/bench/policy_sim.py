from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RuntimeBudget:
    free_vram_bytes: int
    wddm_guard_bytes: int
    workspace_bytes: int
    kv_bytes: int
    safety_margin_bytes: int

    @property
    def usable_weight_bytes(self) -> int:
        used = self.wddm_guard_bytes + self.workspace_bytes + self.kv_bytes + self.safety_margin_bytes
        return max(0, self.free_vram_bytes - used)


@dataclass(frozen=True)
class ModelEstimate:
    filename: str
    size_bytes: int
    quant: str


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def gib(value: int | float) -> float:
    return value / (1024**3)


def decide_strict(model: ModelEstimate, budget: RuntimeBudget) -> dict[str, object]:
    fits = model.size_bytes <= budget.usable_weight_bytes
    return {
        "policy": "strict",
        "admit": fits,
        "host_spill_allowed": False,
        "gpu_weight_bytes": model.size_bytes if fits else 0,
        "host_weight_bytes": 0 if fits else model.size_bytes,
        "gpu_residency_fraction": 1.0 if fits else 0.0,
        "reason": "fits strict VRAM budget" if fits else "reject: full weight residency does not fit",
    }


def decide_balanced(model: ModelEstimate, budget: RuntimeBudget, min_gpu_fraction: float = 0.45) -> dict[str, object]:
    if model.size_bytes <= budget.usable_weight_bytes:
        return {
            "policy": "balanced",
            "admit": True,
            "host_spill_allowed": False,
            "gpu_weight_bytes": model.size_bytes,
            "host_weight_bytes": 0,
            "gpu_residency_fraction": 1.0,
            "reason": "fits without partial offload",
        }

    gpu_bytes = budget.usable_weight_bytes
    fraction = gpu_bytes / model.size_bytes if model.size_bytes else 0.0
    admit = fraction >= min_gpu_fraction
    return {
        "policy": "balanced",
        "admit": admit,
        "host_spill_allowed": False,
        "gpu_weight_bytes": gpu_bytes if admit else 0,
        "host_weight_bytes": model.size_bytes - gpu_bytes if admit else model.size_bytes,
        "gpu_residency_fraction": fraction if admit else 0.0,
        "reason": (
            "partial offload meets minimum GPU residency"
            if admit
            else "reject: partial offload would be too CPU/RAM dominated"
        ),
    }


def decide_overflow(model: ModelEstimate, budget: RuntimeBudget) -> dict[str, object]:
    gpu_bytes = min(model.size_bytes, budget.usable_weight_bytes)
    return {
        "policy": "overflow",
        "admit": True,
        "host_spill_allowed": True,
        "gpu_weight_bytes": gpu_bytes,
        "host_weight_bytes": model.size_bytes - gpu_bytes,
        "gpu_residency_fraction": gpu_bytes / model.size_bytes if model.size_bytes else 0.0,
        "reason": "explicit overflow: RAM dependence is accepted",
    }


def simulate_model(model: ModelEstimate, budget: RuntimeBudget, min_gpu_fraction: float) -> list[dict[str, object]]:
    decisions = [
        decide_strict(model, budget),
        decide_balanced(model, budget, min_gpu_fraction=min_gpu_fraction),
        decide_overflow(model, budget),
    ]
    for decision in decisions:
        decision["filename"] = model.filename
        decision["quant"] = model.quant
        decision["model_size_gib"] = gib(model.size_bytes)
        decision["gpu_weight_gib"] = gib(int(decision["gpu_weight_bytes"]))
        decision["host_weight_gib"] = gib(int(decision["host_weight_bytes"]))
    return decisions


def load_models_from_plan(plan_path: Path) -> tuple[dict[str, object], list[ModelEstimate]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    models = [
        ModelEstimate(
            filename=str(row["filename"]),
            size_bytes=int(row["size_bytes"]),
            quant=str(row["quant"]),
        )
        for row in plan["models"]
    ]
    return plan, models


def self_test() -> None:
    budget = RuntimeBudget(
        free_vram_bytes=12 * 1024**3,
        wddm_guard_bytes=512 * 1024**2,
        workspace_bytes=512 * 1024**2,
        kv_bytes=768 * 1024**2,
        safety_margin_bytes=256 * 1024**2,
    )
    small = ModelEstimate("small.gguf", 4 * 1024**3, "Q4")
    large = ModelEstimate("large.gguf", 20 * 1024**3, "Q4")
    assert decide_strict(small, budget)["admit"] is True
    assert decide_strict(large, budget)["admit"] is False
    assert decide_overflow(large, budget)["admit"] is True
    assert decide_overflow(large, budget)["host_spill_allowed"] is True
    assert decide_balanced(large, budget, min_gpu_fraction=0.45)["admit"] is True


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate strict/balanced/overflow residency policy.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model-plan", default=None)
    parser.add_argument("--free-vram-gib", type=float, default=None)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--kv-mib", type=int, default=768)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--balanced-min-gpu-fraction", type=float, default=0.45)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("policy_sim self-test passed")
        return 0

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    plan_path = Path(args.model_plan) if args.model_plan else bench_dir / "model_plan.json"
    plan, models = load_models_from_plan(plan_path)

    free_vram = (
        int(args.free_vram_gib * 1024**3)
        if args.free_vram_gib is not None
        else int(plan["gpu"]["memory_free_bytes"])
    )
    budget = RuntimeBudget(
        free_vram_bytes=free_vram,
        wddm_guard_bytes=args.wddm_guard_mib * 1024**2,
        workspace_bytes=args.workspace_mib * 1024**2,
        kv_bytes=args.kv_mib * 1024**2,
        safety_margin_bytes=args.safety_margin_mib * 1024**2,
    )

    rows = []
    for model in models:
        rows.extend(simulate_model(model, budget, args.balanced_min_gpu_fraction))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path),
        "budget": {
            "free_vram_gib": gib(budget.free_vram_bytes),
            "wddm_guard_mib": args.wddm_guard_mib,
            "workspace_mib": args.workspace_mib,
            "kv_mib": args.kv_mib,
            "safety_margin_mib": args.safety_margin_mib,
            "usable_weight_gib": gib(budget.usable_weight_bytes),
        },
        "balanced_min_gpu_fraction": args.balanced_min_gpu_fraction,
        "rows": rows,
    }

    out_json = Path(args.output_json) if args.output_json else bench_dir / "policy_sim.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / "policy_sim.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Residency Policy Simulation",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        f"- Source plan: `{plan_path}`",
        f"- Free VRAM: `{result['budget']['free_vram_gib']:.2f} GiB`",
        f"- Usable weight budget: `{result['budget']['usable_weight_gib']:.2f} GiB`",
        f"- Balanced minimum GPU fraction: `{args.balanced_min_gpu_fraction:.2f}`",
        "",
        "| quant | size GiB | policy | admit | GPU GiB | host GiB | GPU fraction | spill | reason |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["quant"]),
                    f"{row['model_size_gib']:.2f}",
                    str(row["policy"]),
                    str(row["admit"]),
                    f"{row['gpu_weight_gib']:.2f}",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
