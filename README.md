# Local AI GPU Acceleration Runtime

Local AI GPU Acceleration Runtime is a Windows-first research and engineering project for running local language-model workloads efficiently on consumer NVIDIA GPUs. The current development target is an RTX 3060 desktop card with 12 GB of VRAM, CUDA compute capability 8.6, and the normal Windows WDDM driver model.

The project focuses on a practical problem: local inference performance is usually limited by memory residency, host/device traffic, launch overhead, and scheduler behavior before it is limited by raw arithmetic throughput. This repository builds the runtime pieces needed to keep useful model state resident on the GPU, replay stable CUDA Graph workloads, measure memory pressure explicitly, and reject hidden host-memory fallback instead of letting the driver or operating system turn a GPU workload into a slow paging workload.

The codebase is not a complete chat application yet. It is the low-level runtime foundation: allocators, manifest loaders, GGUF planning tools, CUDA kernels, graph replay benchmarks, mixed-quant tensor execution paths, strict admission telemetry, and reference-checked benchmark harnesses. The current benchmark path already executes real tensors from the target model family in resident CUDA Graph workloads and validates GPU output against CPU reference calculations.

## Project Goals

The primary goal is to build a local inference runtime that treats the GPU as the main execution environment rather than as a partial accelerator attached to a CPU-controlled loop.

Core goals:

- Keep model weights and hot runtime state in VRAM whenever possible.
- Make host memory fallback explicit and measurable instead of silent.
- Use CUDA Graph replay to reduce per-token launch overhead.
- Use fixed arenas and allocation plans instead of allocation during steady-state decode.
- Track KV, workspace, graph, DMA, and WDDM guard pressure before admitting work.
- Keep steady-state host/device traffic close to token-sized transfers.
- Validate each GPU path against deterministic CPU reference gates.
- Use benchmark evidence to choose the next subsystem to optimize.

The long-term stress target is a model file that is intentionally too large for full residency on a 12 GB GPU. That target forces the runtime design to handle residency, streaming, rejection policy, and scheduling pressure directly rather than assuming every tensor can always remain resident.

## Current Status

The repository currently contains:

- A native C++/CUDA runtime skeleton under `runtime/`.
- A startup-style VRAM superallocator and strict admission model.
- Page-style KV cache primitives and telemetry.
- CUDA Graph benchmark runners for synthetic decode, packed Q4 matvec, Q4_K tensor sequences, IQ tensor probes, and mixed-format layer slices.
- GGUF inspection, tensor planning, resident pack planning, and model download helpers.
- Resident packers for Q4_K, IQ2_S, IQ3_S, and Q5_K tensor formats.
- Native CUDA readers and matvec probes for Q4_K, IQ2_S, IQ3_S, and Q5_K payloads.
- A mixed Q4_K/IQ2_S/IQ3_S/Q5_K layer-slice graph runner.
- A reusable layer role and phase planner for attention, FFN, SSM, and output stages.
- Strict layer-plan validation with CPU reference gates.
- Attention, FFN, SSM, and output phase experiments inside the mixed runner.
- Benchmark summary generation with graph, memory, timing, and phase telemetry.

The most recent full native harness passed all configured rows:

```text
E:\AI project\benchmarks\native_harness_1780126797.md
88/88 rows passed
```

The unit test suite also passed:

```text
68 tests passed
```

## What This Is Not Yet

This repository is not yet a production local LLM server. The current work does not include a full tokenizer/chat loop, full model execution for every layer of the large target file, or an end-user UI. The present milestone is lower-level: prove the runtime can load real target-family tensors into GPU-resident arenas, execute mixed quantization formats through captured CUDA Graphs, keep CPU reference gates in place, and expose enough memory/timing telemetry to guide the next kernel and scheduler work.

The remaining major gap is turning the validated mixed layer-slice path into a complete transformer-layer executor and then into a full model runtime with a strict residency policy.

## Hardware Target

Primary target:

```text
Operating system: Windows 11
GPU: NVIDIA RTX 3060 desktop
VRAM: 12 GB GDDR6
CUDA capability: 8.6
Driver model: WDDM
```

The RTX 3060 is a useful development target because it has enough VRAM to run meaningful local models, but not enough to hide poor residency decisions. That makes it a good forcing function for admission control, memory accounting, graph replay, and explicit offload policy.

The runtime is written with Windows constraints in mind:

- WDDM residency behavior matters.
- TDR-safe kernel durations matter.
- Host/device transfer costs matter.
- Silent system-memory fallback is treated as a failure mode for performance-oriented paths.
- Build and benchmark scripts assume PowerShell and Visual Studio CUDA tooling.

Linux remains a useful comparison target later, but the first-class product target is a Windows desktop GPU.

## Storage Policy

Large files are deliberately kept out of the repository. The default local storage root is:

```text
E:\AI project
```

Expected local directories:

```text
E:\AI project\models
E:\AI project\hf-cache
E:\AI project\build
E:\AI project\benchmarks
E:\AI project\traces
E:\AI project\tmp
E:\AI project\third_party
E:\AI project\packs
E:\AI project\venvs
```

The repository should stay source-only. Model files, GGUFs, generated resident packs, traces, benchmark outputs, build products, and third-party runtime downloads belong under the storage root. The `.gitignore` is configured to keep common model and build artifacts out of Git.

Before downloading models or running large benchmarks, load the environment:

```powershell
. .\scripts\env.ps1
```

## Target Model Family

The long-term stress model is:

```text
Repository: HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
File: Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
```

That file is far larger than the 12 GB VRAM budget, so it is not expected to become fully resident on the RTX 3060. It is the stress case for the runtime's future streaming, scheduling, and strict no-hidden-spill behavior.

The current working validation path uses the smaller `IQ2_M` file from the same model repository. That file provides real GGUF metadata, real quantized tensor layouts, and representative target-family tensor shapes while keeping iteration practical on the current hardware.

Current local workflow:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --download
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --run-llamacpp --ctx-size 1024 --n-predict 32 --timeout 1200
```

The first comparison baseline uses llama.cpp. Controlled traced rows compare different `-ngl` values and record decode throughput plus sampled GPU memory behavior.

## Architecture Overview

The runtime is organized around five ideas:

1. Explicit residency planning.
2. Fixed GPU allocation arenas.
3. Page-managed KV and workspace state.
4. Captured CUDA Graph replay.
5. Reference-checked benchmark iteration.

At a high level:

```text
GGUF or resident pack metadata
  -> tensor and residency planning
  -> strict admission estimate
  -> VRAM superallocator plan
  -> setup-time H2D upload
  -> CUDA Graph capture
  -> CUDA Graph replay
  -> GPU-side token update
  -> token-sized D2H result
  -> JSON benchmark telemetry
```

The project keeps the admission path and execution path close together. A benchmark row reports not only throughput and latency, but also the memory budget used to admit the run: resident weights, predicted and allocated KV, workspace, DMA, graph bytes, WDDM guard, usable bytes, required bytes, and any over-budget amount.

## Repository Layout

```text
runtime/
  core/
    allocator/      VRAM superallocator and arena planning
    graphs/         CUDA Graph bucket accounting
    kv/             page-style KV cache model
    scheduler/      admission control, layer plans, executor contracts
  kernels/
    decode/         CUDA kernels for decode-shaped benchmark paths
  apps/
    benchmark_runner/
    graph_benchmark_runner/
    fused_benchmark_runner/
    q4k_sequence_benchmark_runner/
    iq_probe/
    iq_matvec_probe/
    iq_sequence_benchmark_runner/
    mixed_sequence_benchmark_runner/
    q5k_probe/
    q5k_matvec_probe/
    gguf_probe/
  tests/
    unit/

tools/
  bench/            model planning, packers, benchmark harnesses, summaries

scripts/
  env.ps1           storage-root and Python environment setup
  build_native_vs.ps1
  setup_llamacpp.ps1

docs/
  architecture.md
  benchmarking.md
  model-target.md
  profiling.md
  windows-notes.md
  iteration-log.md

config/
  runtime.example.toml
```

## Runtime Components

### VRAM superallocator

The allocator is designed around a startup reservation model. The current native paths estimate required GPU memory before running, allocate fixed regions, and avoid allocation inside the timed replay loop.

Tracked memory categories include:

- resident weight bytes
- predicted KV bytes
- allocated KV bytes
- phase workspace bytes
- auxiliary workspace bytes
- DMA staging bytes
- CUDA Graph accounting bytes
- WDDM guard bytes
- required bytes
- usable bytes
- over-budget bytes

### KV model

The current benchmark paths use page-style KV updates and active-page parameters to model decode behavior. The next stages are deeper page-table ownership, prefix reuse, quantized KV tiers, and stricter multi-request admission behavior.

### CUDA Graph execution

The native runners capture graph-shaped workloads and replay them for timing. This keeps launch overhead visible and gives a stable place to measure graph memory pressure. The mixed runner also reports graph bucket estimates so future decode bucket selection can be admitted before capture.

### Layer role and phase planning

The scheduler layer classifies tensors into a reusable phase contract:

- attention
- FFN
- SSM
- output

The strict role-plan mode validates phase order, known roles, contiguous segments, expected graph work, feedback edges, phase-local workspaces, auxiliary slices, and failure cases. This is currently covered by `runtime_layer_plan_selftest.exe`.

### Mixed-format layer-slice runner

`runtime_mixed_sequence_bench.exe` is the most important current native path. It loads representative Q4_K, IQ2_S, IQ3_S, and Q5_K tensors, validates them against the source GGUF tensor plan, places them into resident arenas, captures the mixed execution path into a CUDA Graph, and checks the result against a CPU reference path.

Supported phase experiments include:

- `--ffn-phase-mode gated-silu`
- `--attention-phase-mode qkv-scratch`
- `--attention-phase-mode qkv-reduce`
- `--attention-phase-mode qkv-window`
- `--attention-phase-mode qkv-head-window`
- `--attention-phase-mode qkv-head-dim-window`
- `--attention-phase-mode qkv-head-tile-window`
- `--attention-phase-mode qkv-head-group-window`
- `--attention-phase-mode qkv-head-group-rope-window`
- `--attention-phase-mode qkv-head-group-fused`
- `--ssm-phase-mode recurrent-state`
- `--ssm-phase-mode scan-scratch`
- `--ssm-phase-mode selective-scan`
- `--ssm-phase-mode source-parameterized`
- `--ssm-phase-mode source-parameterized-fused`
- `--output-phase-mode final-token`

These modes are benchmark tools, not final model-quality implementations. Their purpose is to evolve the runtime contract toward real attention, FFN, SSM, and output behavior while keeping residency, graph replay, and CPU reference validation in place.

## Quantization and Tensor Formats

The current local pipeline works with several GGUF quantization families:

- Q4_K resident payload and metadata packs
- IQ2_S raw resident blocks
- IQ3_S raw resident blocks
- Q5_K raw resident blocks for the output projection path

The tools can inspect GGUF tensor tables, plan resident formats, create source-verified resident packs, and feed native CUDA readers or matvec probes. Current representative packs live under `E:\AI project\packs`, not in this repository.

Important tools:

- `tools/bench/gguf_inspect.py`
- `tools/bench/gguf_tensor_plan.py`
- `tools/bench/gguf_pack_plan.py`
- `tools/bench/q4k_resident_pack.py`
- `tools/bench/iq_resident_pack.py`
- `tools/bench/q5k_resident_pack.py`
- `tools/bench/native_harness.py`
- `tools/bench/summarize_benchmarks.py`

## Latest Benchmark Snapshot

Latest full harness:

```text
E:\AI project\benchmarks\native_harness_1780126797.md
```

Summary:

```text
Full native harness: 88/88 rows passed
Python unit tests: 68/68 passed
```

Recent mixed all-phase source-fused row:

```text
artifact: native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused_1780126797.json
reference_passed: true
strict_layer_plan_passed: true
token check: 62/62
max logit/KV error: 9.52184e-06 / 9.52184e-06
actual kernels per graph replay: 28
admission workspace bytes: 213760
admission graph bytes: 192512
sequence steps/sec: 2719.42
p50/p95/p99: 0.366 / 0.374 / 0.374 ms
```

Recent standalone source-parameterized fused SSM row:

```text
artifact: native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused_1780126797.json
reference_passed: true
strict_layer_plan_passed: true
token check: 8133/8133
max logit/KV error: 5.48363e-06 / 5.48363e-06
actual kernels per graph replay: 25
SSM kernels per graph replay: 1
admission workspace bytes: 99072
admission graph bytes: 178432
sequence steps/sec: 2855.92
p50/p95/p99: 0.349 / 0.355 / 0.355 ms
```

The source-parameterized fused SSM path folds source-parameter cache writes into the RMSNorm feedback edge. That keeps the source-parameterized behavior while returning the graph-kernel and graph-byte envelope to the simpler scan-scratch path.

## Phase Timing Telemetry

The mixed runner now emits top-level phase timing fields:

- `phase_timing_attention_p50_ms`
- `phase_timing_attention_p95_ms`
- `phase_timing_ffn_p50_ms`
- `phase_timing_ffn_p95_ms`
- `phase_timing_ssm_p50_ms`
- `phase_timing_ssm_p95_ms`
- `phase_timing_output_p50_ms`
- `phase_timing_output_p95_ms`
- `phase_timing_slowest_p95_phase`
- `phase_timing_slowest_p95_ms`
- `graph_minus_phase_timing_sum_p95_ms`

These values come from separate post-graph phase replays. They should be read as hot-spot telemetry, not as additive graph critical-path timing.

Latest all-phase source-fused isolated p95 values:

```text
attention: 0.302 ms
FFN:       0.453 ms
SSM:       0.208 ms
output:    0.046 ms
```

The current isolated hot spot is FFN. The next performance pass should decide whether to improve FFN phase math first or to separately investigate graph-level jitter.

## Build Requirements

Expected Windows toolchain:

- Windows 11
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- Visual Studio Build Tools with C++ support
- CMake
- Python 3.10 virtual environment for benchmark tooling
- PowerShell

The native build helper searches for the Visual Studio developer environment and builds into the storage root:

```powershell
.\scripts\build_native_vs.ps1
```

On the current development machine, native build products are placed under:

```text
E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release
```

## Quick Start

Prepare the environment:

```powershell
. .\scripts\env.ps1
.\scripts\check_toolchain.ps1
```

Run planning tools:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\model_manifest.py
& $env:RTXLLM_PYTHON .\tools\bench\model_plan.py
& $env:RTXLLM_PYTHON .\tools\bench\policy_sim.py
& $env:RTXLLM_PYTHON .\tools\bench\offload_plan.py
& $env:RTXLLM_PYTHON .\tools\bench\allocator_sim.py
& $env:RTXLLM_PYTHON .\tools\bench\kv_cache_sim.py
```

Build native executables:

```powershell
.\scripts\build_native_vs.ps1
```

Run the native harness:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\native_harness.py --storage-root "E:\AI project" --timeout 180
```

Regenerate the benchmark summary:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\summarize_benchmarks.py --storage-root "E:\AI project"
```

Run unit tests:

```powershell
& $env:RTXLLM_PYTHON -m unittest discover -s runtime\tests\unit
```

Run lightweight syntax checks:

```powershell
& $env:RTXLLM_PYTHON -m py_compile tools\bench\native_harness.py tools\bench\summarize_benchmarks.py
```

## llama.cpp Comparison Path

The repository includes helper scripts for installing a Windows CUDA llama.cpp release under the external storage root and running comparison smoke tests.

Install:

```powershell
. .\scripts\env.ps1
.\scripts\setup_llamacpp.ps1
```

Run comparison harness:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\llamacpp_harness.py
```

Run controlled target comparisons:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\llamacpp_matrix.py
& $env:RTXLLM_PYTHON .\tools\bench\compare_target_runs.py
```

## Benchmark Output

Benchmark tools write machine-readable JSON and Markdown summaries under:

```text
E:\AI project\benchmarks
```

The summary generator produces:

```text
E:\AI project\benchmarks\benchmark_summary.md
```

The summary table includes fields for:

- correctness gates
- observed tokens
- max logit error
- max KV error
- graph replay rate
- graph kernel counts
- per-format tensor counts
- workspace bytes
- graph admission bytes
- H2D/D2H traffic
- p50/p95/p99 latency
- phase timing p95
- slowest phase
- resident and allocated memory

## Development Notes

The runtime is intentionally narrow at this stage. The current target is not broad model compatibility; it is building a reliable resident execution path with strong measurement. That means some benchmark modes are deliberately small or synthetic, but they exercise runtime invariants that are required for the full path:

- stable GPU memory ownership
- fixed graph replay shape
- setup-only bulk upload
- token-sized steady-state download
- strict reference checking
- explicit admission accounting
- measured phase behavior

The most useful next engineering tasks are:

- Convert the mixed layer slice into a complete transformer-layer executor.
- Replace benchmark phase primitives with production-quality attention, FFN, and SSM kernels.
- Add quantized and paged KV variants for longer context tests.
- Add stricter no-spill runtime policy checks.
- Build a full-model execution path over the smaller target-family GGUF first.
- Develop a streaming policy for the much larger stress target.
- Add Nsight Systems and Nsight Compute capture recipes for the native runners.
- Add a clean command-line local inference shell after the core executor is stable.

## Documentation

Detailed project notes live in:

- `docs/architecture.md`
- `docs/benchmarking.md`
- `docs/model-target.md`
- `docs/windows-notes.md`
- `docs/profiling.md`
- `docs/iteration-log.md`

The iteration log is intentionally detailed. It records benchmark artifacts, rows added to the harness, validation checks, and measured regressions or improvements across runtime experiments.

## License

No license file has been selected yet. Treat the code as source-available until a license is added.
