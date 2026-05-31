from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], timeout: int) -> dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "output_tail": proc.stdout[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable RTXLLM benchmark/planning suite.")
    parser.add_argument("--storage-root", default=r"E:\AI project")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--include-resident-pack", action="store_true")
    parser.add_argument("--include-llamacpp-smoke", action="store_true")
    parser.add_argument("--trace-llamacpp-smoke", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    root = Path(args.storage_root)
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    commands = [
        [py, "tools/bench/model_plan.py", "--storage-root", args.storage_root],
        [py, "tools/bench/policy_sim.py", "--storage-root", args.storage_root],
        [py, "tools/bench/offload_plan.py", "--storage-root", args.storage_root],
        [py, "tools/bench/gguf_tensor_plan.py", "--storage-root", args.storage_root],
        [py, "tools/bench/target_quant_pipeline.py", "--storage-root", args.storage_root],
        [py, "tools/bench/allocator_sim.py", "--storage-root", args.storage_root],
        [py, "tools/bench/kv_cache_sim.py", "--storage-root", args.storage_root],
        [py, "tools/bench/llamacpp_matrix.py", "--storage-root", args.storage_root],
    ]

    if args.include_gpu and not args.plan_only:
        commands.extend(
            [
                [
                    py,
                    "tools/bench/numba_residency_bench.py",
                    "--storage-root",
                    args.storage_root,
                    "--steps",
                    "128",
                    "--weights-mib",
                    "256",
                    "--workspace-mib",
                    "64",
                    "--kv-pages",
                    "256",
                ],
                [
                    py,
                    "tools/bench/cupy_graph_bench.py",
                    "--storage-root",
                    args.storage_root,
                    "--steps",
                    "256",
                    "--weights-mib",
                    "256",
                    "--workspace-mib",
                    "64",
                    "--kv-pages",
                    "256",
                ],
            ]
        )

    if args.include_native and not args.plan_only:
        commands.extend(
            [
                [py, "tools/bench/native_harness.py", "--storage-root", args.storage_root],
                [
                    py,
                    "tools/bench/q4k_tensor_sweep.py",
                    "--storage-root",
                    args.storage_root,
                    "--timeout",
                    str(args.timeout),
                    "--reference",
                ],
            ]
        )

    if args.include_resident_pack and not args.plan_only:
        commands.extend(
            [
                [py, "tools/bench/q4k_resident_pack.py", "--storage-root", args.storage_root],
                [py, "tools/bench/q5k_resident_pack.py", "--storage-root", args.storage_root],
            ]
        )

    if args.include_llamacpp_smoke and not args.plan_only:
        smoke_model = (
            root
            / "models"
            / "smoke"
            / "aladar__llama-2-tiny-random-GGUF"
            / "llama-2-tiny-random.gguf"
        )
        smoke_command = [
            py,
            "tools/bench/llamacpp_harness.py",
            "--storage-root",
            args.storage_root,
            "--model",
            str(smoke_model),
            "--ctx-size",
            "256",
            "--n-predict",
            "16",
        ]
        if args.trace_llamacpp_smoke:
            smoke_command = [
                py,
                "tools/bench/run_with_gpu_trace.py",
                "--storage-root",
                args.storage_root,
                "--label",
                "llamacpp-smoke",
                "--",
                *smoke_command,
            ]
        commands.append(smoke_command)

    commands.append([py, "tools/bench/summarize_benchmarks.py", "--storage-root", args.storage_root])

    runs = [run(command, timeout=args.timeout) for command in commands]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": args.storage_root,
        "plan_only": args.plan_only,
        "include_gpu": args.include_gpu,
        "include_native": args.include_native,
        "include_resident_pack": args.include_resident_pack,
        "include_llamacpp_smoke": args.include_llamacpp_smoke,
        "trace_llamacpp_smoke": args.trace_llamacpp_smoke,
        "runs": runs,
    }

    out_path = bench_dir / f"bench_suite_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"json={out_path}")
    return 0 if all(run_result["returncode"] == 0 for run_result in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
