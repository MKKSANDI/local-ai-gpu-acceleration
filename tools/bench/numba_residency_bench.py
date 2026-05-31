from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import cuda


@cuda.jit
def init_u8_kernel(values):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for pos in range(i, values.size, stride):
        values[pos] = (pos * 131 + 17) & 0xFF


@cuda.jit
def init_f32_kernel(values):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for pos in range(i, values.size, stride):
        values[pos] = 0.0


@cuda.jit
def synthetic_decode_kernel(weights, kv, token, page_words, active_pages, step, touch_stride):
    tid = cuda.grid(1)
    grid_stride = cuda.gridsize(1) * touch_stride

    acc = 0
    for pos in range(tid, weights.size, grid_stride):
        acc += weights[pos]

    if tid < active_pages:
        kv_pos = tid * page_words + (step % page_words)
        kv[kv_pos] = kv[kv_pos] + np.float32((acc & 0xFF) * 0.000001)

    if tid == 0:
        token[0] = np.uint32((step + acc) % 32000)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def mib(value: int) -> int:
    return value * 1024 * 1024


def query_gpu() -> dict[str, object]:
    query = "name,memory.total,memory.free,driver_version,compute_cap"
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [part.strip() for part in raw.split(",")]
    return {
        "name": parts[0],
        "memory_total_bytes": int(parts[1]) * 1024 * 1024,
        "memory_free_bytes": int(parts[2]) * 1024 * 1024,
        "driver_version": parts[3],
        "compute_cap": parts[4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic CUDA residency benchmark for RTXLLM.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--weights-mib", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--kv-pages", type=int, default=256)
    parser.add_argument("--page-kib", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--touch-stride", type=int, default=64)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")

    gpu_before = query_gpu()
    root = storage_root(args.storage_root)
    out_dir = root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    free_before = int(gpu_before["memory_free_bytes"])
    total = int(gpu_before["memory_total_bytes"])

    weight_bytes = mib(args.weights_mib)
    workspace_bytes = mib(args.workspace_mib)
    page_bytes = args.page_kib * 1024
    kv_bytes = args.kv_pages * page_bytes
    prompt_bytes = args.prompt_tokens * np.dtype(np.uint32).itemsize
    token_bytes = np.dtype(np.uint32).itemsize
    requested = weight_bytes + kv_bytes + workspace_bytes + prompt_bytes + token_bytes
    usable = max(0, free_before - mib(args.wddm_guard_mib))

    if requested > usable:
        result = {
            "path": "numba_cuda_synthetic_decode",
            "policy": "strict",
            "admitted": False,
            "reason": "strict VRAM admission rejected synthetic workload",
            "requested_bytes": requested,
            "usable_bytes": usable,
            "free_before_bytes": free_before,
            "total_bytes": total,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path = out_dir / f"numba_residency_reject_{int(time.time())}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"json={out_path}")
        return 2

    weights = cuda.device_array(weight_bytes, dtype=np.uint8)
    kv = cuda.device_array(kv_bytes // 4, dtype=np.float32)
    workspace = cuda.device_array(workspace_bytes // 4, dtype=np.float32)
    device_prompt = cuda.device_array(args.prompt_tokens, dtype=np.uint32)
    device_token = cuda.device_array(1, dtype=np.uint32)

    host_prompt = cuda.pinned_array(args.prompt_tokens, dtype=np.uint32)
    host_prompt[:] = np.arange(args.prompt_tokens, dtype=np.uint32)
    host_token = cuda.pinned_array(1, dtype=np.uint32)

    cuda.to_device(host_prompt, to=device_prompt)

    init_u8_kernel[args.blocks, args.threads](weights)
    init_f32_kernel[args.blocks, args.threads](kv)
    init_f32_kernel[args.blocks, args.threads](workspace)
    synthetic_decode_kernel[args.blocks, args.threads](
        weights, kv, device_token, page_bytes // 4, args.kv_pages, 0, args.touch_stride
    )
    device_token.copy_to_host(host_token)
    cuda.synchronize()

    gpu_after_alloc = query_gpu()
    free_after_alloc = int(gpu_after_alloc["memory_free_bytes"])

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for step in range(args.steps):
        t0 = time.perf_counter()
        synthetic_decode_kernel[args.blocks, args.threads](
            weights,
            kv,
            device_token,
            page_bytes // 4,
            args.kv_pages,
            step,
            args.touch_stride,
        )
        device_token.copy_to_host(host_token)
        cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_ms = (time.perf_counter() - start) * 1000.0

    gpu_after = query_gpu()
    free_after = int(gpu_after["memory_free_bytes"])
    allocated_estimate = free_before - free_after_alloc
    steps_per_second = args.steps * 1000.0 / wall_ms if wall_ms > 0 else 0.0
    touched_weight_bytes_per_step = weight_bytes // max(1, args.touch_stride)

    result = {
        "path": "numba_cuda_synthetic_decode",
        "policy": "strict",
        "admitted": True,
        "device": gpu_before["name"],
        "driver_version": gpu_before["driver_version"],
        "compute_capability": gpu_before["compute_cap"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": args.steps,
        "wall_ms": wall_ms,
        "steps_per_second": steps_per_second,
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "weights_mib": args.weights_mib,
        "workspace_mib": args.workspace_mib,
        "kv_pages": args.kv_pages,
        "page_kib": args.page_kib,
        "touch_stride": args.touch_stride,
        "requested_bytes": requested,
        "allocated_estimate_bytes": allocated_estimate,
        "free_before_bytes": free_before,
        "free_after_alloc_bytes": free_after_alloc,
        "free_after_bytes": free_after,
        "total_bytes": total,
        "h2d_bytes": prompt_bytes,
        "d2h_bytes": token_bytes * (args.steps + 1),
        "graph_replay_rate": 0.0,
        "graph_note": "Numba path does not expose CUDA Graph capture; native runtime owns this work.",
        "touched_weight_bytes_per_step": touched_weight_bytes_per_step,
        "cpu_copy_back_bytes_per_step": token_bytes,
        "last_token": int(host_token[0]),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"numba_residency_{timestamp}.json"
    md_path = out_dir / f"numba_residency_{timestamp}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Numba CUDA Residency Benchmark",
                "",
                f"- Path: `{result['path']}`",
                f"- Device: `{result['device']}`",
                f"- Steps/sec: `{steps_per_second:.2f}`",
                f"- p50/p95/p99 ms: `{result['p50_ms']:.3f}` / `{result['p95_ms']:.3f}` / `{result['p99_ms']:.3f}`",
                f"- Requested MiB: `{requested / (1024**2):.2f}`",
                f"- Allocated estimate MiB: `{allocated_estimate / (1024**2):.2f}`",
                f"- H2D bytes: `{result['h2d_bytes']}`",
                f"- D2H bytes: `{result['d2h_bytes']}`",
                f"- Graph replay rate: `{result['graph_replay_rate']}`",
                "",
                f"JSON: `{json_path}`",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
