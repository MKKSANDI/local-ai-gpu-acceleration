from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def quote(value: str) -> str:
    return f'"{value}"'


def build_command(exe: str, model_path: str, profile: dict[str, object]) -> str:
    args = [
        quote(exe),
        "-m",
        quote(model_path),
        "-p",
        quote("Write one sentence about GPU residency."),
        "-n",
        str(profile["n_predict"]),
        "-c",
        str(profile["ctx_size"]),
        "-ngl",
        str(profile["ngl"]),
        "-ctk",
        str(profile["cache_type_k"]),
        "-ctv",
        str(profile["cache_type_v"]),
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
    ]
    for flag in profile["extra_flags"]:
        args.append(str(flag))
    return " ".join(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate llama.cpp target-model execution matrix.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model-plan", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    plan_path = Path(args.model_plan) if args.model_plan else bench_dir / "model_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    llama_root = root / "third_party" / "llama.cpp"
    exe = str(llama_root / "llama-completion.exe")
    model_dir = root / "models" / DEFAULT_REPO_SAFE

    profiles = [
        {
            "name": "strict_probe",
            "intent": "Prove whether full GPU residency is possible; failure is acceptable evidence.",
            "ngl": "all",
            "ctx_size": 1024,
            "n_predict": 16,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "off", "--no-mmap"],
            "policy": "strict",
        },
        {
            "name": "balanced_fit",
            "intent": "Let llama.cpp fit layers to available VRAM while keeping KV cache smaller.",
            "ngl": "auto",
            "ctx_size": 2048,
            "n_predict": 64,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1024"],
            "policy": "balanced",
        },
        {
            "name": "latency_guard",
            "intent": "Use a larger VRAM guard and shorter context to reduce desktop pressure.",
            "ngl": "auto",
            "ctx_size": 1024,
            "n_predict": 64,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "2048"],
            "policy": "balanced",
        },
        {
            "name": "overflow_moe_cpu",
            "intent": "Emergency large-model continuation with MoE weights on CPU; expected RAM dependence.",
            "ngl": "auto",
            "ctx_size": 1024,
            "n_predict": 32,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "extra_flags": ["-fit", "on", "-fit-target", "1536", "--cpu-moe"],
            "policy": "overflow",
        },
    ]

    rows = []
    for model in plan["models"]:
        filename = model["filename"]
        model_path = str(model_dir / filename)
        for profile in profiles:
            rows.append(
                {
                    "filename": filename,
                    "quant": model["quant"],
                    "size_gib": model["size_gib"],
                    "local": bool(model["local"]),
                    "strict_weight_fit_with_reserves": model[
                        "strict_weight_fit_with_reserves"
                    ],
                    "profile": profile["name"],
                    "policy": profile["policy"],
                    "intent": profile["intent"],
                    "command": build_command(exe, model_path, profile),
                }
            )

    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path),
        "llama_completion": exe,
        "repo": plan["repo"],
        "usable_vram_for_weights_gib": plan["usable_vram_for_weights_gib"],
        "note": (
            "Profiles are command plans, not proof of fit. Use strict_probe to collect failure evidence, "
            "balanced_fit for first practical target-repo runs, and overflow_moe_cpu only when explicit RAM "
            "dependence is acceptable."
        ),
        "rows": rows,
    }

    out_json = Path(args.output_json) if args.output_json else bench_dir / "llamacpp_matrix.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / "llamacpp_matrix.md"
    out_json.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    lines = [
        "# llama.cpp Execution Matrix",
        "",
        f"Generated: `{matrix['generated_at']}`",
        "",
        f"- Source plan: `{plan_path}`",
        f"- Usable strict weight budget: `{plan['usable_vram_for_weights_gib']:.2f} GiB`",
        f"- Note: {matrix['note']}",
        "",
        "| quant | size GiB | local | strict fit | profile | policy | command |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["quant"],
                    f"{row['size_gib']:.2f}",
                    str(row["local"]),
                    str(row["strict_weight_fit_with_reserves"]),
                    row["profile"],
                    row["policy"],
                    f"`{row['command']}`",
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
