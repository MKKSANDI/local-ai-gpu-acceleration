from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


NGL_RE = re.compile(r"ngl(?P<ngl>\d+)", re.IGNORECASE)


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def metric(data: dict[str, object], key: str) -> object | None:
    direct = data.get("llamacpp_metrics")
    if isinstance(direct, dict) and key in direct:
        return direct[key]
    run = data.get("llamacpp_run")
    if isinstance(run, dict):
        metrics = run.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            return metrics[key]
    return None


def ngl_from_path(path: Path) -> int | None:
    match = NGL_RE.search(path.stem)
    return int(match.group("ngl")) if match else None


def latest_by_ngl(paths: list[Path]) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    for path in paths:
        ngl = ngl_from_path(path)
        if ngl is None:
            continue
        previous = latest.get(ngl)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest[ngl] = path
    return latest


def load_trace_summary(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    return summary if isinstance(summary, dict) else {}


def build_rows(root: Path, quant: str, trace_label_prefix: str) -> list[dict[str, object]]:
    bench_dir = root / "benchmarks"
    trace_dir = root / "traces"
    run_paths = latest_by_ngl(list(bench_dir.glob(f"target_quant_pipeline_{quant}_ngl*.json")))
    trace_paths = latest_by_ngl(list(trace_dir.glob(f"{trace_label_prefix}-ngl*-controlled_*.json")))

    rows: list[dict[str, object]] = []
    for ngl in sorted(run_paths):
        run_path = run_paths[ngl]
        data = json.loads(run_path.read_text(encoding="utf-8"))
        trace_path = trace_paths.get(ngl)
        summary = load_trace_summary(trace_path)
        tensor_plan = data.get("tensor_plan")
        if not isinstance(tensor_plan, dict):
            tensor_plan = {}
        rows.append(
            {
                "ngl": ngl,
                "run_json": str(run_path),
                "trace_json": str(trace_path) if trace_path else None,
                "status": data.get("status"),
                "ngl_source": data.get("ngl_source"),
                "planned_tensor_ngl": tensor_plan.get("resident_layers_estimate"),
                "tensor_residency_fraction": tensor_plan.get("gpu_residency_fraction_estimate"),
                "decode_tokens_per_second": metric(data, "decode_tokens_per_second"),
                "prompt_tokens_per_second": metric(data, "prompt_tokens_per_second"),
                "graphs_reused": metric(data, "graphs_reused"),
                "load_time_ms": metric(data, "load_time_ms"),
                "harness_wall_ms": metric(data, "wall_ms"),
                "trace_wall_ms": summary.get("wall_ms"),
                "avg_gpu_utilization_percent": summary.get("avg_gpu_utilization_percent"),
                "max_gpu_utilization_percent": summary.get("max_gpu_utilization_percent"),
                "max_memory_used_mib": summary.get("max_memory_used_mib"),
                "min_memory_free_mib": summary.get("min_memory_free_mib"),
                "max_power_draw_w": summary.get("max_power_draw_w"),
                "max_temperature_gpu_c": summary.get("max_temperature_gpu_c"),
            }
        )
    return rows


def best_decode_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    numeric = [
        row
        for row in rows
        if isinstance(row.get("decode_tokens_per_second"), int | float)
    ]
    return max(numeric, key=lambda row: float(row["decode_tokens_per_second"])) if numeric else None


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, object]) -> None:
    rows = result["rows"]
    best = result.get("best_decode")
    lines = [
        "# Target NGL Comparison",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Storage root: `{result['storage_root']}`",
        f"Quant: `{result['quant']}`",
        "",
    ]
    if isinstance(best, dict):
        lines.extend(
            [
                f"Best decode row: `-ngl {best['ngl']}` at `{best['decode_tokens_per_second']}` tok/s.",
                "",
            ]
        )
    lines.extend(
        [
            "| ngl | status | decode tok/s | prompt tok/s | graphs | load ms | max VRAM MiB | min free MiB | avg GPU % | max GPU % | max power W | max temp C | tensor plan ngl | tensor residency |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ngl"]),
                    str(row["status"]),
                    fmt(row["decode_tokens_per_second"], 2),
                    fmt(row["prompt_tokens_per_second"], 2),
                    str(row["graphs_reused"] or ""),
                    fmt(row["load_time_ms"], 2),
                    fmt(row["max_memory_used_mib"], 0),
                    fmt(row["min_memory_free_mib"], 0),
                    fmt(row["avg_gpu_utilization_percent"], 3),
                    fmt(row["max_gpu_utilization_percent"], 0),
                    fmt(row["max_power_draw_w"], 2),
                    fmt(row["max_temperature_gpu_c"], 0),
                    str(row["planned_tensor_ngl"] or ""),
                    fmt(row["tensor_residency_fraction"], 3),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare controlled target quant runs by -ngl.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--quant", default="IQ2_M")
    parser.add_argument("--trace-label-prefix", default="target-iq2m")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    rows = build_rows(root, args.quant, args.trace_label_prefix)
    result: dict[str, object] = {
        "path": "target_ngl_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(root),
        "quant": args.quant,
        "trace_label_prefix": args.trace_label_prefix,
        "rows": rows,
        "best_decode": best_decode_row(rows),
    }
    out_json = Path(args.output_json) if args.output_json else bench_dir / f"target_ngl_comparison_{args.quant}.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / f"target_ngl_comparison_{args.quant}.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(f"json={out_json}")
    print(f"markdown={out_md}")
    print(f"rows={len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
