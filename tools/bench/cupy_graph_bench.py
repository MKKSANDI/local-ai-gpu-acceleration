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

import cupy as cp
import numpy as np


KERNELS = r"""
extern "C" __global__ void init_u8(unsigned char* values, unsigned long long n) {
  const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = gridDim.x * blockDim.x;
  for (unsigned long long pos = tid; pos < n; pos += stride) {
    values[pos] = (unsigned char)((pos * 131ull + 17ull) & 0xffull);
  }
}

extern "C" __global__ void init_f32(float* values, unsigned long long n) {
  const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = gridDim.x * blockDim.x;
  for (unsigned long long pos = tid; pos < n; pos += stride) {
    values[pos] = 0.0f;
  }
}

extern "C" __global__ void synthetic_decode_graph(
    const unsigned char* weights,
    float* kv,
    unsigned int* token,
    unsigned long long weight_bytes,
    unsigned long long kv_words,
    unsigned int page_words,
    unsigned int active_pages,
    int touch_stride) {
  const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      gridDim.x * blockDim.x * (unsigned long long)touch_stride;

  unsigned int acc = 0;
  for (unsigned long long pos = tid; pos < weight_bytes; pos += stride) {
    acc += (unsigned int)weights[pos];
  }

  if (tid < active_pages) {
    const unsigned long long page_base = tid * (unsigned long long)page_words;
    const unsigned long long offset = (token[0] + tid) % page_words;
    const unsigned long long kv_pos = (page_base + offset) % kv_words;
    kv[kv_pos] = kv[kv_pos] + (float)(acc & 0xffu) * 0.000001f;
  }

  if (tid == 0) {
    token[0] = (token[0] + 1u + acc) % 32000u;
  }
}
"""


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


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def mib(value: int) -> int:
    return value * 1024 * 1024


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="CuPy CUDA Graph decode benchmark.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--weights-mib", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--kv-pages", type=int, default=256)
    parser.add_argument("--page-kib", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--touch-stride", type=int, default=64)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    out_dir = root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_before = query_gpu()
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
            "path": "cupy_cuda_graph_synthetic_decode",
            "policy": "strict",
            "admitted": False,
            "reason": "strict VRAM admission rejected synthetic workload",
            "requested_bytes": requested,
            "usable_bytes": usable,
            "free_before_bytes": free_before,
            "total_bytes": total,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path = out_dir / f"cupy_graph_reject_{int(time.time())}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"json={out_path}")
        return 2

    mem_pool = cp.cuda.MemoryPool()
    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_allocator(mem_pool.malloc)
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)

    init_u8 = cp.RawKernel(KERNELS, "init_u8")
    init_f32 = cp.RawKernel(KERNELS, "init_f32")
    decode = cp.RawKernel(KERNELS, "synthetic_decode_graph")

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        weights = cp.empty((weight_bytes,), dtype=cp.uint8)
        kv = cp.empty((kv_bytes // 4,), dtype=cp.float32)
        workspace = cp.empty((workspace_bytes // 4,), dtype=cp.float32)
        token = cp.zeros((1,), dtype=cp.uint32)
        prompt_host = np.arange(args.prompt_tokens, dtype=np.uint32)
        prompt = cp.asarray(prompt_host)

        init_u8((args.blocks,), (args.threads,), (weights, np.uint64(weight_bytes)))
        init_f32((args.blocks,), (args.threads,), (kv, np.uint64(kv.size)))
        init_f32((args.blocks,), (args.threads,), (workspace, np.uint64(workspace.size)))
        decode(
            (args.blocks,),
            (args.threads,),
            (
                weights,
                kv,
                token,
                np.uint64(weight_bytes),
                np.uint64(kv.size),
                np.uint32(page_bytes // 4),
                np.uint32(args.kv_pages),
                np.int32(args.touch_stride),
            ),
        )
    stream.synchronize()
    gpu_after_alloc = query_gpu()
    free_after_alloc = int(gpu_after_alloc["memory_free_bytes"])

    graph = None
    capture_ms = 0.0
    upload_ms = 0.0
    if not args.no_graph:
        t0 = time.perf_counter()
        stream.begin_capture()
        decode(
            (args.blocks,),
            (args.threads,),
            (
                weights,
                kv,
                token,
                np.uint64(weight_bytes),
                np.uint64(kv.size),
                np.uint32(page_bytes // 4),
                np.uint32(args.kv_pages),
                np.int32(args.touch_stride),
            ),
        )
        graph = stream.end_capture()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        graph.upload(stream)
        stream.synchronize()
        upload_ms = (time.perf_counter() - t1) * 1000.0

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for _ in range(args.steps):
        t0 = time.perf_counter()
        if graph is None:
            decode(
                (args.blocks,),
                (args.threads,),
                (
                    weights,
                    kv,
                    token,
                    np.uint64(weight_bytes),
                    np.uint64(kv.size),
                    np.uint32(page_bytes // 4),
                    np.uint32(args.kv_pages),
                    np.int32(args.touch_stride),
                ),
            )
        else:
            graph.launch(stream)
        stream.synchronize()
        last_token = int(token.get()[0])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_ms = (time.perf_counter() - start) * 1000.0

    gpu_after = query_gpu()
    free_after = int(gpu_after["memory_free_bytes"])
    allocated_estimate = free_before - free_after_alloc
    steps_per_second = args.steps * 1000.0 / wall_ms if wall_ms > 0 else 0.0

    result = {
        "path": "cupy_cuda_graph_synthetic_decode"
        if graph is not None
        else "cupy_cuda_nograph_synthetic_decode",
        "policy": "strict",
        "admitted": True,
        "device": gpu_before["name"],
        "driver_version": gpu_before["driver_version"],
        "compute_capability": gpu_before["compute_cap"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": args.steps,
        "wall_ms": wall_ms,
        "steps_per_second": steps_per_second,
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "capture_ms": capture_ms,
        "graph_upload_ms": upload_ms,
        "weights_mib": args.weights_mib,
        "workspace_mib": args.workspace_mib,
        "kv_pages": args.kv_pages,
        "page_kib": args.page_kib,
        "touch_stride": args.touch_stride,
        "requested_bytes": requested,
        "allocated_estimate_bytes": allocated_estimate,
        "cupy_pool_used_bytes": mem_pool.used_bytes(),
        "cupy_pool_total_bytes": mem_pool.total_bytes(),
        "free_before_bytes": free_before,
        "free_after_alloc_bytes": free_after_alloc,
        "free_after_bytes": free_after,
        "total_bytes": total,
        "h2d_bytes": prompt_bytes,
        "d2h_bytes": token_bytes * args.steps,
        "graph_replay_rate": 1.0 if graph is not None else 0.0,
        "cpu_copy_back_bytes_per_step": token_bytes,
        "last_token": last_token,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "graph" if graph is not None else "nograph"
    json_path = out_dir / f"cupy_{suffix}_{timestamp}.json"
    md_path = out_dir / f"cupy_{suffix}_{timestamp}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# CuPy CUDA Graph Benchmark",
                "",
                f"- Path: `{result['path']}`",
                f"- Device: `{result['device']}`",
                f"- Steps/sec: `{steps_per_second:.2f}`",
                f"- p50/p95/p99 ms: `{result['p50_ms']:.3f}` / `{result['p95_ms']:.3f}` / `{result['p99_ms']:.3f}`",
                f"- Capture/upload ms: `{capture_ms:.3f}` / `{upload_ms:.3f}`",
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
