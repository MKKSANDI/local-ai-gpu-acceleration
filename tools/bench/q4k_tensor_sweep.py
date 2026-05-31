from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.native_harness import (  # noqa: E402
    DEFAULT_BUILD,
    DEFAULT_REPO_SAFE,
    DEFAULT_TARGET,
    run_command,
    sanitize_label,
    storage_root,
    write_metric_artifact,
)


def default_model(root: Path) -> Path:
    return root / "models" / DEFAULT_REPO_SAFE / DEFAULT_TARGET


def default_tensor_plan(root: Path) -> Path:
    return root / "benchmarks" / f"gguf_tensor_plan_{DEFAULT_TARGET.removesuffix('.gguf')}.json"


def default_build_dir(root: Path) -> Path:
    return root / DEFAULT_BUILD


def select_representative_q4k_tensors(plan_path: Path) -> list[dict[str, object]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tensors = plan.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError(f"{plan_path} does not contain tensor rows")
    selected: dict[tuple[int, ...], dict[str, object]] = {}
    for tensor in tensors:
        if tensor.get("type") != "Q4_K":
            continue
        dims = tuple(int(value) for value in tensor.get("dimensions") or [])
        if len(dims) != 2:
            continue
        selected.setdefault(dims, tensor)
    return [selected[key] for key in sorted(selected, key=lambda item: (item[1], item[0]))]


def variant_args(
    variant: str,
    *,
    blocks_path: Path,
    payload_path: Path,
) -> list[str]:
    if variant == "packed_vec4x4":
        return ["--weights-file", str(payload_path), "--packed-kernel", "vec4x4"]
    if variant == "q4k_vec4x4":
        return ["--q4k-blocks-file", str(blocks_path), "--q4k-kernel", "vec4x4"]
    if variant == "q4k_predecoded":
        return ["--q4k-blocks-file", str(blocks_path), "--q4k-kernel", "predecoded"]
    if variant == "q4k_split_predecoded":
        return [
            "--q4k-blocks-file",
            str(blocks_path),
            "--q4k-payload-file",
            str(payload_path),
            "--q4k-kernel",
            "split-predecoded",
        ]
    if variant == "q4k_split_half":
        return [
            "--q4k-blocks-file",
            str(blocks_path),
            "--q4k-payload-file",
            str(payload_path),
            "--q4k-kernel",
            "split-half",
        ]
    if variant == "q4k_split_compact":
        return [
            "--q4k-blocks-file",
            str(blocks_path),
            "--q4k-payload-file",
            str(payload_path),
            "--q4k-kernel",
            "split-compact",
        ]
    if variant == "q4k_split_native":
        return [
            "--q4k-blocks-file",
            str(blocks_path),
            "--q4k-payload-file",
            str(payload_path),
            "--q4k-kernel",
            "split-native",
        ]
    raise ValueError(f"unknown variant: {variant}")


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Q4_K Tensor Sweep",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Model: `{result['model']}`",
        f"Build dir: `{result['build_dir']}`",
        f"Steps: `{result['steps']}`",
        f"KV MiB: `{result['kv_mib']}`",
        "",
        "| tensor | dims | variant | ref | max err | steps/s | p50 ms | p95 ms | p99 ms | meta B | meta fmt | artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in result["runs"]:
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("tensor_name", "")),
                    str(run.get("dimensions", "")),
                    str(run.get("variant", "")),
                    str(metrics.get("reference_passed", "")),
                    fmt(metrics.get("max_logit_abs_error"), 6),
                    fmt(metrics.get("steps_per_second"), 2),
                    fmt(metrics.get("p50_ms")),
                    fmt(metrics.get("p95_ms")),
                    fmt(metrics.get("p99_ms")),
                    str(metrics.get("q4k_predecoded_meta_bytes", "")),
                    str(metrics.get("q4k_predecoded_meta_format", "")),
                    str(run.get("artifact", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Q4_K kernel variants across target GGUF tensors.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tensor-plan", default=None)
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--kv-mib", type=int, default=64)
    parser.add_argument("--active-pages", type=int, default=512)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--tensor", action="append", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    tmp_dir = root / "tmp" / "q4k_sweep"
    bench_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model = Path(args.model) if args.model else default_model(root)
    tensor_plan = Path(args.tensor_plan) if args.tensor_plan else default_tensor_plan(root)
    build_dir = Path(args.build_dir) if args.build_dir else default_build_dir(root)
    probe_exe = build_dir / "runtime_gguf_probe.exe"
    fused_exe = build_dir / "runtime_fused_bench.exe"
    stamp = int(time.time())
    generated_at = datetime.now(timezone.utc).isoformat()

    tensors = select_representative_q4k_tensors(tensor_plan)
    if args.tensor:
        requested = set(args.tensor)
        tensors = [tensor for tensor in tensors if str(tensor.get("name")) in requested]
    if not tensors:
        raise SystemExit("no Q4_K tensors selected")

    variants = [
        "packed_vec4x4",
        "q4k_vec4x4",
        "q4k_predecoded",
        "q4k_split_predecoded",
        "q4k_split_half",
        "q4k_split_compact",
        "q4k_split_native",
    ]
    reference_variants = [
        "q4k_vec4x4",
        "q4k_split_predecoded",
        "q4k_split_half",
        "q4k_split_compact",
        "q4k_split_native",
    ]

    runs: list[dict[str, object]] = []
    for tensor in tensors:
        tensor_name = str(tensor["name"])
        dims = [int(value) for value in tensor["dimensions"]]
        cols, rows = dims[0], dims[1]
        label = sanitize_label(tensor_name)
        payload_path = tmp_dir / f"{label}_payload.bin"
        blocks_path = tmp_dir / f"{label}_blocks.bin"
        probe_args = [
            "--model",
            str(model),
            "--q4k-tensor",
            tensor_name,
            "--dump-q4k-payload",
            str(payload_path),
            "--dump-q4k-blocks",
            str(blocks_path),
        ]
        probe_run = {
            "name": f"probe_{label}",
            "tensor_name": tensor_name,
            "dimensions": dims,
            "variant": "probe",
            "exe": str(probe_exe),
            "args": probe_args,
        }
        probe_run.update(run_command([str(probe_exe), *probe_args], timeout=args.timeout))
        probe_run["status"] = "ok" if probe_run["returncode"] == 0 and "metrics" in probe_run else "failed"
        artifact = write_metric_artifact(bench_dir, probe_run, stamp=stamp, generated_at=generated_at)
        if artifact:
            probe_run["artifact"] = str(artifact)
        runs.append(probe_run)
        if probe_run["status"] != "ok":
            continue

        for variant in variants:
            bench_args = [
                "--label",
                f"{label}_{variant}",
                "--steps",
                str(args.steps),
                "--rows",
                str(rows),
                "--cols",
                str(cols),
                "--kv-mib",
                str(args.kv_mib),
                "--active-pages",
                str(args.active_pages),
                *variant_args(variant, blocks_path=blocks_path, payload_path=payload_path),
            ]
            run = {
                "name": f"{label}_{variant}",
                "tensor_name": tensor_name,
                "dimensions": dims,
                "variant": variant,
                "exe": str(fused_exe),
                "args": bench_args,
            }
            run.update(run_command([str(fused_exe), *bench_args], timeout=args.timeout))
            run["status"] = "ok" if run["returncode"] == 0 and "metrics" in run else "failed"
            artifact = write_metric_artifact(bench_dir, run, stamp=stamp, generated_at=generated_at)
            if artifact:
                run["artifact"] = str(artifact)
            runs.append(run)

        if args.reference:
            for variant in reference_variants:
                check_args = [
                    "--label",
                    f"{label}_{variant}_reference",
                    "--steps",
                    "1",
                    "--rows",
                    str(rows),
                    "--cols",
                    str(cols),
                    "--kv-mib",
                    str(args.kv_mib),
                    "--active-pages",
                    str(args.active_pages),
                    *variant_args(variant, blocks_path=blocks_path, payload_path=payload_path),
                    "--check-only",
                ]
                run = {
                    "name": f"{label}_{variant}_reference",
                    "tensor_name": tensor_name,
                    "dimensions": dims,
                    "variant": f"{variant}_reference",
                    "exe": str(fused_exe),
                    "args": check_args,
                }
                run.update(run_command([str(fused_exe), *check_args], timeout=args.timeout))
                run["status"] = "ok" if run["returncode"] == 0 and "metrics" in run else "failed"
                artifact = write_metric_artifact(bench_dir, run, stamp=stamp, generated_at=generated_at)
                if artifact:
                    run["artifact"] = str(artifact)
                runs.append(run)

    result = {
        "path": "q4k_tensor_sweep",
        "generated_at": generated_at,
        "model": str(model),
        "tensor_plan": str(tensor_plan),
        "build_dir": str(build_dir),
        "steps": args.steps,
        "kv_mib": args.kv_mib,
        "active_pages": args.active_pages,
        "runs": runs,
    }
    json_path = bench_dir / f"q4k_tensor_sweep_{stamp}.json"
    md_path = bench_dir / f"q4k_tensor_sweep_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(md_path, result)
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0 if all(run.get("status") == "ok" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
