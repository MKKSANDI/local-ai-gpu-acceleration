# Profiling

The profiler split is:

- Nsight Systems for timelines, launch gaps, allocation calls, H2D/D2H copies, CPU wake-ups, and WDDM stalls.
- Nsight Compute for kernel occupancy, memory throughput, instruction mix, Tensor Core activity, and register/shared-memory pressure.

## First Questions

For every benchmark run, answer:

- Did any large D2H copy happen?
- Did H2D occur only at request boundary?
- Did allocated bytes stay within predicted budget?
- Did inter-token latency jitter widen under WDDM?
- Are kernel launches dominating?

## Trace Output

Traces should go to:

```text
E:\AI project\traces
```

The benchmark JSON records enough paths and parameters to pair raw results with later Nsight captures.

## GPU Trace Wrapper

Use `run_with_gpu_trace.py` when a benchmark needs a lightweight GPU utilization, VRAM, power, and temperature envelope:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label trace-smoke -- $env:RTXLLM_PYTHON --version
```

For the llama.cpp smoke model:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_bench_suite.py --include-llamacpp-smoke --trace-llamacpp-smoke
```

For the native CUDA Graph runner:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label native-graph-bench -- "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_graph_bench.exe"
```

For the native fused Q4 runner:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label native-fused-bench -- "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe"
```

For the larger fused Q4 stress shape:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label native-fused-stress -- "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label stress_8192x8192 --steps 64 --rows 8192 --cols 8192 --kv-mib 64 --active-pages 512
```

For the fused Q4 CPU-reference check:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label native-fused-reference-check -- "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label reference_64x128 --rows 64 --cols 128 --steps 4 --kv-mib 1 --page-words 256 --active-pages 16 --check-only
```

The wrapper writes one JSON file and one Markdown summary to `E:\AI project\traces`. It samples `nvidia-smi` while the child process runs, records the child return code and wall time, and keeps only a bounded stdout/stderr tail so long benchmark logs do not bloat trace files.
