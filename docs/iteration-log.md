# Iteration Log

## 2026-05-28 Bootstrap

Machine facts observed:

- GPU: NVIDIA GeForce RTX 3060, 12 GiB VRAM, compute capability 8.6.
- Driver: 596.21.
- Mode: WDDM, watchdog enabled.
- External storage: `E:\AI project`, about 119 GiB free at bootstrap.
- Native CUDA compiler: CUDA 12.0 and 13.0 installed.
- Native C++ blocker: `cl.exe` is not visible; Visual Studio C++ build tools need to be enabled before C++/CUDA targets can compile.

Repository work completed:

- Created source-only repo scaffold.
- Routed models, Hugging Face cache, builds, traces, temporary files, and benchmark output to `E:\AI project`.
- Added native C++/CUDA skeleton for allocator, admission policy, KV pages, graph bucket wrapper, synthetic decode kernel, and benchmark executable.
- Added Python 3.10 venv setup under `E:\AI project\venvs\rtxllm-py310`.
- Added Hugging Face manifest tooling for the target model.
- Added first executable GPU benchmark using Numba CUDA.
- Added CuPy CUDA Graph benchmark and benchmark summary generator.
- Added model fit planner and single-file external-storage downloader.
- Installed official llama.cpp Windows CUDA binaries under `E:\AI project\third_party\llama.cpp`.
- Added llama.cpp comparison harness.

Target model metadata:

- Repo: `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`.
- File: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`.
- API size: `40.61 GiB`.
- Manifest: `E:\AI project\benchmarks\manifest_HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive.json`.
- Hugging Face CLI dry-run confirms one target file totaling `43.6G`; no model bytes downloaded yet.
- Model plan: `E:\AI project\benchmarks\model_plan.md`.
- Current plan result: no target-repo GGUF fits strict full weight residency once WDDM, KV, and workspace reserves are applied. Even `IQ2_M` is `10.86 GiB`, while current usable strict weight budget is `9.04 GiB`.
- llama.cpp target execution matrix: `E:\AI project\benchmarks\llamacpp_matrix.md`.
- Matrix profiles generated for each target-repo quant: `strict_probe`, `balanced_fit`, `latency_guard`, and `overflow_moe_cpu`.

First successful GPU benchmark:

- Result: `E:\AI project\benchmarks\numba_residency_20260528_215115.json`.
- Workload: 256 MiB synthetic weights, 64 MiB workspace, 256 KV pages, 128 decode steps.
- Steps/sec: `5454.44`.
- p50/p95/p99 step latency: `0.137 ms / 0.455 ms / 0.675 ms`.
- D2H bytes: `516`.
- H2D bytes: `2048`.
- Graph replay rate: `0.0` because Numba does not expose CUDA Graph capture here.

First CUDA Graph benchmark:

- Graph result: `E:\AI project\benchmarks\cupy_graph_20260528_215732.json`.
- No-graph comparison: `E:\AI project\benchmarks\cupy_nograph_20260528_215741.json`.
- Workload: 256 MiB synthetic weights, 64 MiB workspace, 256 KV pages, 256 decode steps.
- Graph steps/sec: `43861.15`.
- No-graph steps/sec: `8335.45`.
- Graph p50/p95/p99 step latency: `0.019 ms / 0.032 ms / 0.046 ms`.
- No-graph p50/p95/p99 step latency: `0.106 ms / 0.248 ms / 0.356 ms`.
- Capture/upload time: `0.191 ms / 0.021 ms`.
- This confirms the local benchmark harness can measure launch-overhead reduction from graph replay before the native C++ path is available.
- Summary report: `E:\AI project\benchmarks\benchmark_summary.md`.

llama.cpp baseline:

- Installed release: `ggml-org/llama.cpp` tag `b9371`, published `2026-05-27T23:45:52Z`.
- Release URL: `https://github.com/ggml-org/llama.cpp/releases/tag/b9371`.
- Installed assets: `llama-b9371-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip`.
- Install manifest: `E:\AI project\third_party\llama.cpp\rtxllm-llamacpp-install.json`.
- Harness result: `E:\AI project\benchmarks\llamacpp_harness_1779972164.json`.
- Current harness status: `model_missing`; `llama-cli.exe --help` runs successfully, but no local GGUF has been downloaded yet.
- Smoke GGUF downloaded: `E:\AI project\models\smoke\aladar__llama-2-tiny-random-GGUF\llama-2-tiny-random.gguf`.
- Smoke metadata: `E:\AI project\benchmarks\smoke_aladar_llama2_tiny_random_metadata.json`.
- Negative smoke result: `E:\AI project\models\smoke\tiny-random-llama-Q4_K_M-GGUF\tiny-random-llama.Q4_K_M.gguf` parses as GGUF, but llama.cpp rejects it because one quantized tensor row is not a valid block multiple.
- Harness hardening after smoke tests: close stdin, force `--no-conversation`, `--single-turn`, `--simple-io`, decode output as UTF-8, and mark non-zero model return codes as `model_run_failed`.
- Successful smoke run: `E:\AI project\benchmarks\llamacpp_harness_1779972844.json`.
- Smoke llama.cpp metrics: prompt eval `11173.18 tok/s`, decode `3227.89 tok/s`, graphs reused `14`, load time `140.26 ms`, `USE_GRAPHS = 1`.
- The harness now prefers `llama-completion.exe` for non-interactive model runs and keeps `llama-cli.exe` for tool availability checks.

GGUF inspector:

- Added `tools/bench/gguf_inspect.py`.
- Self-test parses an in-memory GGUF v3 header.
- Real smoke model inspection extracts architecture, context length, vocabulary arrays with samples only, tensor names, tensor dimensions, tensor types, and offsets without loading tensor data.

Strict admission rejection:

- Result: `E:\AI project\benchmarks\numba_residency_reject_1779970890.json`.
- Requested synthetic weight scale: 42,000 MiB plus workspace/KV.
- Required bytes: `44,610,619,396`.
- Usable bytes after guard: `11,183,063,040`.
- Outcome: rejected before allocation, matching strict no-silent-spill policy.

Observed toolchain issue:

- Default Python 3.13 Numba stack can detect CUDA but fails during kernel/context execution with a CUDA context access violation.
- Python 3.10 venv works and is now preferred via `scripts/env.ps1`.
- Native CMake configure initially failed because `nvcc` could not find `cl.exe` from a plain PowerShell session; Visual C++ tools were later found under `D:\Visual build tools`.
- The Ninja preset also failed after `vcvars64.bat` because CMake reused/picked WinLibs `windres.exe`/`ld.exe` in the CUDA try-compile. The working native path is the Visual Studio generator preset.

## 2026-05-28 Native Graph Path

Repository work completed:

- Added `runtime/kernels/decode/decode_kernels.h` for native kernel declarations.
- Added stateful synthetic decode kernel entry point `rtxllm_launch_synthetic_decode_stateful`.
- Added native executable target `runtime_graph_bench`.
- `runtime_graph_bench` reserves the same style of VRAM arenas as the baseline, captures a decode kernel plus token-sized D2H copy into a CUDA Graph, uploads the graph, replays it for 256 steps, and emits JSON metrics: replay latency percentiles, capture/upload time, graph replay rate, D2H bytes, and allocated bytes.
- Added `tools/bench/llamacpp_matrix.py` to generate strict, balanced, and overflow llama.cpp command profiles for target-repo GGUF tests.
- Added `tools/bench/policy_sim.py` to simulate strict, balanced, and overflow residency policy decisions against the target model plan.
- Added `runtime/tests/unit/test_policy_sim.py` for policy fault-injection unit coverage.
- Added `tools/bench/run_bench_suite.py` as a repeatable planning/GPU/smoke orchestrator.
- Added `tools/bench/kv_cache_sim.py` as a deterministic fixed-page KV cache simulation with prefix refcounts, active sequence private pages, reclaim, pressure rejection, and utilization metrics.
- Added `runtime/tests/unit/test_kv_cache_sim.py` to cover page rounding, page reuse, prefix reclaim limits, prefix cache hits, and pressure rejection.
- Added `tools/bench/allocator_sim.py` as a deterministic VRAM arena simulation for immutable weights, KV pages, transient workspace, and DMA rings.
- Added `runtime/tests/unit/test_allocator_sim.py` to cover alignment, free-block reuse, overflow failure accounting, and arena metrics.
- Added `tools/bench/run_with_gpu_trace.py` as a reusable command wrapper that samples `nvidia-smi` GPU utilization, VRAM, power, and temperature while a benchmark process runs.
- Added `runtime/tests/unit/test_gpu_trace.py` to cover nvidia-smi row parsing, missing telemetry values, summary generation, and command delimiter handling.
- Added `--trace-llamacpp-smoke` to `tools/bench/run_bench_suite.py` so the llama.cpp smoke path can be executed under the GPU trace wrapper.
- Added `tools/bench/offload_plan.py` to turn the target model manifest and current VRAM budget into estimated strict, balanced, latency-guard, and overflow execution profiles.
- Added `runtime/tests/unit/test_offload_plan.py` to cover resident-layer estimation, strict admission, and overflow-only large-model behavior.
- `run_bench_suite.py` now refreshes `kv_cache_sim` during plan-only runs.
- `run_bench_suite.py` now refreshes `allocator_sim` during plan-only runs.
- `run_bench_suite.py` now refreshes `offload_plan` during plan-only runs.
- Plan-only suite result: `E:\AI project\benchmarks\bench_suite_1779973641.json`.
- Latest plan-only suite result: `E:\AI project\benchmarks\bench_suite_1779974201.json`.
- Policy simulation result: `E:\AI project\benchmarks\policy_sim.md`.
- Current policy boundary: strict rejects every target-repo GGUF; balanced admits `IQ2_M`, `Q2_K_P`, `IQ3_M`, `IQ4_XS`, `Q3_K_P`, and `IQ4_NL` under a 45% minimum GPU-residency threshold; `Q4_K_M` and larger are overflow-only under that threshold. `Q8_K_P` is estimated at about `22%` GPU weight residency in overflow mode.
- Allocator simulation artifact: `E:\AI project\benchmarks\allocator_sim.md`.
- Latest allocator simulation: `512` steps, `0` rejected allocations, total high-water fraction `0.412`; arena high-water: weights `256.00 MiB`, KV `81.00 MiB`, workspace `41.82 MiB`, DMA `3.95 MiB`.
- KV simulation artifact: `E:\AI project\benchmarks\kv_cache_sim.md`.
- Latest KV simulation: `512` requests, `512` admitted, `0` rejected, prefix hit rate `0.719`, high-water utilization `0.097`, mean utilization `0.075`, reclaimed prefixes `132`, retained hot-prefix pages after active drain `271`.
- Benchmark summary now includes allocator high-water/reject columns and KV prefix-hit/high-water columns.
- First GPU trace smoke result: `E:\AI project\traces\trace-smoke_20260528_224057.md`.
- Trace smoke command: project Python `--version`.
- Trace smoke result: return code `0`, wall time `82.133 ms`, `2` GPU samples, max VRAM used `1057 MiB`, min free VRAM `11054 MiB`, max GPU utilization `39%`.
- Suite-level traced llama.cpp smoke result: `E:\AI project\traces\llamacpp-smoke_20260528_224232.md`.
- Traced llama.cpp smoke telemetry: return code `0`, wall time `885.812 ms`, `4` GPU samples, max VRAM used `1191 MiB`, min free VRAM `10920 MiB`, max GPU utilization `33%`.
- Wrapped llama.cpp smoke metrics: prompt eval `6435.01 tok/s`, decode `2875.77 tok/s`, graphs reused `14`, load time `151.87 ms`, `USE_GRAPHS = 1`.
- Suite result with traced llama.cpp smoke: `E:\AI project\benchmarks\bench_suite_1779973953.json`.
- Offload plan artifact: `E:\AI project\benchmarks\offload_plan.md`.
- Current Q8_K_P overflow estimate with assumed `48` layers and `256 MiB` safety margin: `-ngl 10`, `8.80 GiB` GPU-resident weights, `31.81 GiB` host-resident weights, GPU residency fraction `0.22`.

Validation status:

- Source-level Python tooling validation still passes.
- Policy, allocator, KV, GPU trace, and offload planner unit tests pass: `20` tests.
- CMake presets are still discoverable.
- At this checkpoint, native C++/CUDA compilation was still blocked until `cl.exe` was made visible to `nvcc`.

## 2026-05-28 Native Build Unblocked

Repository work completed:

- Added `scripts/build_native_vs.ps1`, which locates `vcvars64.bat` via `vswhere` or known install paths and runs the Visual Studio CMake preset inside the MSVC developer environment.
- Updated `scripts/check_toolchain.ps1` to report the discovered Visual C++ install and point to the native build helper when `cl.exe` is not in the current shell.
- Fixed native CUDA build issues:
  - disabled unnecessary CUDA separable compilation for the simple synthetic kernel library;
  - fixed `cudaHostAlloc` typed pointer conversion in `runtime_graph_bench`;
  - updated `cudaGraphInstantiate` to the CUDA 12/13 three-argument signature.
- Added `tools/bench/native_harness.py` to run `runtime_bench.exe` and `runtime_graph_bench.exe`, parse their JSON output, and write benchmark artifacts under `E:\AI project\benchmarks`.
- Added `runtime/tests/unit/test_native_harness.py`.
- Added `--include-native` to `tools/bench/run_bench_suite.py`.

Native build evidence:

- Configure command through `vcvars64.bat` and `windows-vs2022-cuda86-release` succeeded.
- Build command through `scripts/build_native_vs.ps1 -SkipConfigure` succeeded.
- Built executables:
  - `E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_bench.exe`
  - `E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_graph_bench.exe`

Native benchmark evidence:

- Direct `runtime_bench.exe`: `128` steps, `1934.10` steps/s, `419430400` allocated bytes.
- Direct `runtime_graph_bench.exe`: `256` steps, `4446.56` steps/s, p50/p95/p99 `0.2612/0.5675/0.7058 ms`, graph replay rate `1.0`, D2H bytes `1024`.
- Traced native graph result: `E:\AI project\traces\native-graph-bench_20260528_225227.md`.
- Traced native graph telemetry: return code `0`, wrapper wall time `248.803 ms`, `2` GPU samples, max VRAM used `1054 MiB`, min free VRAM `11057 MiB`, max GPU utilization `26%`.
- Native harness result: `E:\AI project\benchmarks\native_harness_1779974719.md`.
- Latest native harness metrics: launch-per-step native baseline `4464.71` steps/s; graph native baseline `9235.74` steps/s, p50/p95/p99 `0.089/0.249/0.537 ms`, graph replay rate `1.0`, D2H bytes `1024`.
- Suite result with native harness: `E:\AI project\benchmarks\bench_suite_1779974720.json`.

Validation status:

- Native build helper works when invoked through this session's approved `cmd /d` execution path; direct PowerShell invocation inside the sandbox cannot write to `E:\AI project`, but the script is intended for the user's normal shell.
- Python tooling validation passes.
- Policy, allocator, KV, GPU trace, offload planner, and native harness unit tests pass: `22` tests.
- Native C++/CUDA compilation is no longer blocked for the Visual Studio preset.

Next bottlenecks after the warp-broadcast pass:

- Replace synthetic byte-touch kernel with fused dequant/matmul and KV page access patterns.
- Replace estimated offload layer counts with tensor-level GGUF inspection after a target-repo quant is downloaded.
- Run llama.cpp against a target-repo quant once one is downloaded to `E:\AI project\models`.
- Pin or repair the Ninja preset if a faster single-config generator is still useful; the Visual Studio preset is the working path.

## 2026-05-28 Native Fused Q4 Path

Repository work completed:

- Added `rtxllm_launch_q4_matvec_decode` to the native CUDA kernel library.
- The new kernel path reads packed Q4 weights, dequantizes inside the matvec loop, writes logits, updates page-style KV slots, and runs GPU-side argmax/token update before the token-sized D2H copy.
- Added native executable `runtime_fused_bench.exe`.
- Added `runtime/apps/fused_benchmark_runner/main.cpp`.
- Added `runtime_fused_bench` to `CMakeLists.txt`.
- Added `runtime_fused_bench.exe` to `tools/bench/native_harness.py`.
- Extended benchmark summary output with allocated MiB, setup H2D MiB, and Q4 values per step.

Native fused benchmark evidence:

- Build command through `scripts/build_native_vs.ps1` succeeded.
- Built executable: `E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe`.
- Direct fused run: `128` steps, `5909.76` steps/s, p50/p95/p99 `0.1395/0.3134/0.5914 ms`, graph replay rate `1.0`, `8,388,608` Q4 values per step, setup H2D `4,218,884` bytes, steady-state H2D `0`, D2H `512` bytes.
- Native harness result: `E:\AI project\benchmarks\native_harness_1779975144.md`.
- Latest harness fused metrics: `6369.90` steps/s, p50/p95/p99 `0.132/0.303/0.581 ms`, graph replay rate `1.0`, allocated `21.03 MiB`, setup H2D `4.02 MiB`, D2H `512` bytes.
- Trace result: `E:\AI project\traces\native-fused-bench_20260528_230149.md`.
- Traced fused telemetry: return code `0`, wrapper wall time `265.115 ms`, `2` GPU samples, max VRAM used `1053 MiB`, min free VRAM `11058 MiB`, max GPU utilization `34%`.
- Suite result with native fused harness: `E:\AI project\benchmarks\bench_suite_1779975145.json`.

Validation status:

- Python tooling validation passes.
- Native build passes through the Visual Studio preset.
- Native fused benchmark runs directly, under the harness, and under the GPU trace wrapper.
- Policy, allocator, KV, GPU trace, offload planner, and native harness unit tests pass: `22` tests.

Next bottlenecks:

- Replace the scalar Q4 matvec kernel with a tiled/shared-memory or Tensor Core-friendly path.
- Add tensor-level GGUF planning once a target-repo quant is downloaded.
- Add a native loader path that imports packed weight metadata instead of generated host buffers.

## 2026-05-28 Parameterized Fused Stress Profiles

Repository work completed:

- Added command-line shape controls to `runtime_fused_bench.exe`: `--label`, `--steps`, `--rows`, `--cols`, `--kv-mib`, `--page-words`, `--active-pages`, and `--wddm-guard-mib`.
- Added even-column validation for packed Q4 layout.
- Updated `tools/bench/native_harness.py` to run two fused profiles in one pass:
  - `runtime_fused_bench_default`: `2048x4096`, `128` steps;
  - `runtime_fused_bench_stress`: `8192x8192`, `64` steps, `64 MiB` KV arena, `512` active pages.
- Updated native harness artifact names so multiple runs with the same benchmark path no longer overwrite each other.
- Updated benchmark summary output with a `label` column.

Stress benchmark evidence:

- Direct stress run: `64` steps, `1334.55` steps/s, p50/p95/p99 `0.6277/1.3634/1.4607 ms`, graph replay rate `1.0`, `67,108,864` Q4 values per step, allocated `101,810,176` bytes, setup H2D `33,619,972` bytes, steady-state H2D `0`, D2H `256` bytes.
- Native harness result: `E:\AI project\benchmarks\native_harness_1779975375.md`.
- Harness stress artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_stress_1779975375.json`.
- Traced stress result: `E:\AI project\traces\native-fused-stress_20260528_230615.md`.
- Traced stress telemetry: return code `0`, wrapper wall time `388.567 ms`, `3` GPU samples, max VRAM used `1510 MiB`, min free VRAM `10601 MiB`, max GPU utilization `57%`.
- Latest suite result with labeled default/stress fused rows: `E:\AI project\benchmarks\bench_suite_1779975393.json`.
- Latest summary rows include `default` and `stress_8192x8192` labels; latest stress row reports `1337.36` steps/s, p50/p95/p99 `0.628/1.479/1.771 ms`, allocated `97.09 MiB`, setup H2D `32.06 MiB`, and D2H `256` bytes.

Validation status:

- Native build passes through the Visual Studio preset.
- Fused stress profile runs directly, through native harness, through GPU trace wrapper, and through `run_bench_suite.py --include-native`.
- Python tooling validation passes.
- Unit tests pass: `23` tests.

Next bottlenecks:

- Replace the scalar Q4 matvec reduction with a tiled/vectorized implementation and compare default/stress profiles.
- Add tensor-level GGUF import planning and a target-repo quant download/run path.

## 2026-05-28 Packed Q4 Byte-Load Optimization

Repository work completed:

- Replaced the fused Q4 matvec inner loop's scalar nibble path with a `packed2` path that loads one packed byte and dequants both Q4 nibbles in the same iteration.
- Added `kernel_variant: "packed2"` to `runtime_fused_bench.exe` JSON output.
- Added `q4_packed_bytes_per_step` to fused benchmark JSON output.
- Updated `tools/bench/summarize_benchmarks.py` with `variant` and `Q4 packed B/step` columns.

Before/after evidence:

- Previous comparable latest default fused row: `5922.99` steps/s, p50/p95/p99 `0.1345/0.380/0.6186 ms`.
- New packed2 default harness row: `8852.31` steps/s, p50/p95/p99 `0.094/0.194/0.550 ms`.
- Previous comparable latest stress row: `1337.36` steps/s, p50/p95/p99 `0.6278/1.479/1.7712 ms`.
- New packed2 stress harness row: `2493.67` steps/s, p50/p95/p99 `0.355/0.909/1.072 ms`.
- Direct packed2 stress run: `2178.45` steps/s, p50/p95/p99 `0.3835/0.8176/1.3119 ms`.
- Traced packed2 stress result: `E:\AI project\traces\native-fused-stress-packed2_20260528_230948.md`.
- Traced packed2 stress telemetry: return code `0`, wrapper wall time `317.718 ms`, `2` GPU samples, max VRAM used `1084 MiB`, min free VRAM `11027 MiB`, max GPU utilization `53%`.
- Native harness result: `E:\AI project\benchmarks\native_harness_1779975588.md`.
- Latest suite result: `E:\AI project\benchmarks\bench_suite_1779975612.json`.

Validation status:

- Native build passes through the Visual Studio preset.
- Packed2 default and stress profiles run directly.
- Native harness, GPU trace wrapper, and suite refresh pass.
- Python tooling validation passes.
- Unit tests pass: `23` tests.

Next bottlenecks:

- Move from per-row block reductions to a tiled/vectorized kernel with better activation reuse and less reduction overhead.
- Add correctness checks against a CPU reference for small Q4 matvec shapes.
- Add tensor-level GGUF import planning and a target-repo quant download/run path.

## 2026-05-28 Native Q4 Reference Check

Repository work completed:

- Added `--check-reference` and `--check-only` to `runtime_fused_bench.exe`.
- The reference check runs the CUDA packed2 Q4 matvec path on a deterministic small shape, copies logits/KV/token back, computes a CPU reference, and compares logits, KV writes, and token selection.
- Added reference metrics to fused JSON output: `reference_checked`, `reference_passed`, `max_logit_abs_error`, `max_kv_abs_error`, mismatch counts, expected token, and observed token.
- Added `runtime_fused_reference_check` to `tools/bench/native_harness.py`.
- Updated native harness Markdown with a `check` column.
- Updated benchmark summary with `ref` and `max logit err` columns.

Reference evidence:

- Direct command:
  `runtime_fused_bench.exe --label reference_64x128 --rows 64 --cols 128 --steps 4 --kv-mib 1 --page-words 256 --active-pages 16 --check-only`
- Direct result: `reference_passed=true`, max logit abs error `3.27826e-07`, max KV abs error `3.12924e-07`, `0` logit mismatches, `0` KV mismatches, token expected/observed `50/50`.
- Native harness result: `E:\AI project\benchmarks\native_harness_1779975876.md`.
- Reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_reference_check_1779975876.json`.
- Trace artifact: `E:\AI project\traces\native-fused-reference-check_20260528_231436.md`.
- Latest suite result with reference row: `E:\AI project\benchmarks\bench_suite_1779975895.json`.
- Latest summary row: `native_cuda_q4_matvec_reference_check`, label `reference_64x128`, variant `packed2`, `ref=True`.

Validation status:

- Native build passes through the Visual Studio preset.
- Direct reference check passes.
- Native harness, GPU trace wrapper, and suite refresh pass.
- Python tooling validation passes.
- Unit tests pass: `23` tests.

Next bottlenecks:

- Make future kernel variants prove correctness through the reference check before comparing performance.
- Move from per-row block reductions to a tiled/vectorized kernel with better activation reuse and less reduction overhead.
- Add tensor-level GGUF import planning and a target-repo quant download/run path.

## 2026-05-28 GGUF Tensor Residency Planner

Repository work completed:

- Enhanced `tools/bench/gguf_inspect.py` to expose tensor info end offset, tensor data offset, and tensor data alignment.
- Added `tools/bench/gguf_tensor_plan.py`.
- The tensor planner reads local GGUF metadata, derives tensor spans from tensor offsets and file size, groups `blk.N.*` tensors into layers, separates non-layer tensors, and estimates complete-layer residency under the current strict VRAM budget.
- Added `runtime/tests/unit/test_gguf_tensor_plan.py`.
- Added `gguf_tensor_plan.py` to `tools/bench/run_bench_suite.py` plan-only refreshes.

Local smoke GGUF evidence:

- Model: `E:\AI project\models\smoke\aladar__llama-2-tiny-random-GGUF\llama-2-tiny-random.gguf`.
- Tensor plan: `E:\AI project\benchmarks\gguf_tensor_plan_llama-2-tiny-random.md`.
- JSON plan: `E:\AI project\benchmarks\gguf_tensor_plan_llama-2-tiny-random.json`.
- Detected architecture: `llama`.
- Tensor count: `12`.
- Tensor data offset: `724416`.
- Layer count detected: `1`.
- Non-layer tensor span: `0.98 MiB`.
- Tensor span total: `0.98 MiB`.
- Strict full resident: `True`.
- Resident layers estimate: `1/1`.
- Tensor-span GPU residency estimate: `1.000`.
- Largest tensor spans: `token_embd.weight` and `output.weight`, each `0.4883 MiB`.
- Latest plan-only suite including tensor planner: `E:\AI project\benchmarks\bench_suite_1779976271.json`.

Validation status:

- `gguf_tensor_plan.py` runs against the local smoke GGUF.
- Plan-only suite refresh passes.
- Python tooling validation passes.
- Unit tests pass: `25` tests.

Next bottlenecks:

- Run `gguf_tensor_plan.py` against a downloaded target-repo quant and replace assumed layer counts in `offload_plan.py` with tensor-derived complete-layer estimates.
- Add a target quant download/run path that starts with the smallest practical target-repo GGUF.

## 2026-05-28 Target IQ2_M Download And Baseline Run

Repository work completed:

- Added `tools/bench/target_quant_pipeline.py`.
- The target pipeline selects a target-repo GGUF quant, downloads it to `E:\AI project\models`, derives a local tensor/layer residency plan, builds a planned llama.cpp command, and can run or import a llama.cpp harness result.
- Added `runtime/tests/unit/test_target_quant_pipeline.py`.
- Hardened target pipeline parsing so llama.cpp help text containing JSON examples does not hide the real `llamacpp_harness` JSON metrics.
- Added `--llamacpp-harness-json` to regenerate a clean target report from an existing harness artifact without repeating the long model load.

Downloaded target quant:

- Repo: `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`.
- File: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf`.
- Local path: `E:\AI project\models\HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf`.
- Model plan now marks `IQ2_M` as local.

Tensor-plan evidence:

- Target report: `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M.md`.
- Direct tensor plan: `E:\AI project\benchmarks\gguf_tensor_plan_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.md`.
- Architecture: `qwen35moe`.
- Tensor count: `733`.
- Detected layer count: `40`.
- Tensor span: `11108.63 MiB`.
- Non-layer span: `541.84 MiB`.
- Strict full resident: `False`.
- Complete-layer estimate varies with free VRAM snapshot: first run planned `31/40`; latest refreshed report planned `32/40`.

First target-repo llama.cpp baseline:

- Harness artifact: `E:\AI project\benchmarks\llamacpp_harness_1779978087.json`.
- Actual model command used `-ngl 31`, `-c 1024`, and `-n 32`.
- Result: return code `0`.
- Load time: `215615.11 ms`.
- Prompt eval: `7` tokens, `3.26 tok/s`.
- Decode: `31` runs, `9.70 tok/s`.
- CUDA graphs: `USE_GRAPHS = 1`, graphs reused `30`.
- Warning observed: `n_ctx_seq (1024) < n_ctx_train (262144)`, expected for this short-context first run.

Validation status:

- Target pipeline unit tests pass.
- The refreshed target report imports the existing harness metrics cleanly and distinguishes the current planned `-ngl` from the actual imported run command.

Offload planner update:

- `tools/bench/offload_plan.py` now uses tensor-derived layer spans for local GGUF rows.
- Refreshed artifact: `E:\AI project\benchmarks\offload_plan.md`.
- `IQ2_M` now reports `basis=tensor`, `40` layers, balanced `-ngl 32`, latency-guard `-ngl 28`, and strict full residency rejected.
- Not-yet-downloaded rows still use the assumed `48` layer count.

Traced `-ngl 32` target run:

- Trace artifact: `E:\AI project\traces\target-iq2m-ngl32_20260529_000028.md`.
- Benchmark artifact refreshed: `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M.md`.
- Harness artifact: `E:\AI project\benchmarks\llamacpp_harness_1779978848.json`.
- Result: return code `0`.
- Actual model command used `-ngl 32`, `-c 1024`, and `-n 32`.
- Load time: `205488.47 ms`.
- Prompt eval: `7` tokens, `4.28 tok/s`.
- Decode: `31` runs, `22.15 tok/s`.
- CUDA graphs: `USE_GRAPHS = 1`, graphs reused `30`.
- Trace envelope: wall `220559.799 ms`, `708` samples, average GPU utilization `29.569%`, max GPU utilization `99%`, max VRAM used `10065 MiB`, min free VRAM `2046 MiB`, max power `55.110 W`, max temp `52 C`.
- Compared with the earlier `-ngl 31` run, decode improved from `9.70` to `22.15` tok/s. This comparison is directional rather than isolated because the second run also had different cache/warm-state conditions and trace sampling.
- `tools/bench/summarize_benchmarks.py` now reads target-pipeline `llamacpp_metrics`, so `target_quant_pipeline_IQ2_M.json` appears in `E:\AI project\benchmarks\benchmark_summary.md`.

Controlled `-ngl` comparison:

- Added `--ngl` override to `tools/bench/target_quant_pipeline.py`.
- Added `tools/bench/compare_target_runs.py` to join target-pipeline llama.cpp metrics with GPU trace summaries.
- Comparison artifact: `E:\AI project\benchmarks\target_ngl_comparison_IQ2_M.md`.
- Controlled trace artifacts:
  - `E:\AI project\traces\target-iq2m-ngl28-controlled_20260529_000708.md`;
  - `E:\AI project\traces\target-iq2m-ngl31-controlled_20260529_001058.md`;
  - `E:\AI project\traces\target-iq2m-ngl32-controlled_20260529_001449.md`.
- Controlled benchmark artifacts:
  - `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M_ngl28.md`;
  - `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M_ngl31.md`;
  - `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M_ngl32.md`.
- Controlled results:
  - `-ngl 28`: `17.41` decode tok/s, max VRAM `9009 MiB`, min free VRAM `3102 MiB`;
  - `-ngl 31`: `19.60` decode tok/s, max VRAM `9801 MiB`, min free VRAM `2310 MiB`;
  - `-ngl 32`: `20.86` decode tok/s, max VRAM `10072 MiB`, min free VRAM `2039 MiB`.
- Current decision: keep `-ngl 32` as the best short-run target for `IQ2_M`, with `-ngl 28` preserved as the lower-VRAM latency-guard profile.

Native loader pack plan:

- Added `tools/bench/gguf_pack_plan.py`.
- Added `runtime/tests/unit/test_gguf_pack_plan.py`.
- Pack-plan artifact: `E:\AI project\benchmarks\gguf_pack_plan_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.md`.
- Target `IQ2_M` source types:
  - `IQ2_S`: `375` tensors, `9841.20 MiB`;
  - `IQ3_S`: `16` tensors, `573.63 MiB`;
  - `Q4_K`: `40` tensors, `275.62 MiB`;
  - `Q5_K`: `1` tensor, `333.44 MiB`;
  - `F32`: `301` tensors, `84.74 MiB`.
- Largest tensor roles:
  - `ffn_down_exps.weight`: `3364.00 MiB`;
  - `ffn_gate_exps.weight`: `3280.00 MiB`;
  - `ffn_up_exps.weight`: `3280.00 MiB`;
  - `output`: `333.44 MiB`;
  - `attn_qkv.weight`: `270.00 MiB`.
- Loader gap identified: native runtime currently has synthetic packed-Q4 kernels only; GGUF IQ/K-quant block readers and conversion packers are required before real target weights can feed the hot path.

Native GGUF probe:

- Added `runtime/apps/gguf_probe/main.cpp`.
- Added `runtime_gguf_probe.exe` to `CMakeLists.txt`.
- Added block-size validation for `F32`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, `IQ2_S`, and `IQ3_S`.
- Added `tools/bench/gguf_block_formats.py` to keep Python pack-plan block validation aligned with the native probe.
- `tools/bench/gguf_pack_plan.py` now validates source block spans: target result is `733/733` valid tensors, `0` invalid, `0` padding bytes.
- Native build through `scripts/build_native_vs.ps1 -SkipConfigure` succeeds and produces `E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_gguf_probe.exe`.
- Standalone native probe artifact: `E:\AI project\benchmarks\native_gguf_probe_IQ2_M.json`.
- Harness native probe artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1779980538.json`.
- Native probe result: version `3`, architecture `qwen35moe`, tensor count `733`, supported tensors `733`, valid tensors `733`, invalid tensors `0`, expected data bytes `11648246272`, padding bytes `0`.
- `tools/bench/native_harness.py` now runs `runtime_gguf_probe_iq2m`.
- `tools/bench/summarize_benchmarks.py` now includes `valid tensors` and `invalid tensors` columns.

Native `Q4_K` data smoke:

- Extended `runtime_gguf_probe.exe` to seek into the GGUF tensor-data section and read the first real `Q4_K` block.
- Tensor selected: `blk.0.attn_qkv.weight`.
- Tensor dimensions: `[2048, 8192]`.
- Tensor absolute offset: `581848704`.
- Tensor span: `9437184` bytes.
- Block count: `65536`.
- First block size: `144` bytes.
- Packed Q4 payload: `128` bytes / `256` values.
- Q4 nibble range: `0..15`.
- Scale-byte checksum: `2384`.
- Packed Q4 checksum: `16789`.
- Harness artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1779980843.json`.
- Harness Markdown: `E:\AI project\benchmarks\native_harness_1779980843.md`.
- `tools/bench/summarize_benchmarks.py` now includes `q4k smoke` and `q4k checksum` columns.

Full `Q4_K` payload staging into CUDA graph matvec:

- Extended `runtime_gguf_probe.exe` with `--q4k-tensor` and `--dump-q4k-payload`.
- Dumped `blk.0.attn_qkv.weight` packed Q4 payload to `E:\AI project\tmp\blk0_attn_qkv_q4_payload.bin`.
- Payload bytes: `8388608`.
- Full payload checksum: `1055786949`.
- Extended `runtime_fused_bench.exe` with `--weights-file`.
- `runtime_fused_bench.exe` now reports `weight_source`, `weights_file`, and `host_weight_checksum`.
- Extended `tools/bench/native_harness.py` to run:
  - target GGUF probe with payload dump;
  - synthetic fused stress profile;
  - real target Q4 payload fused profile.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779981156.md`.
- Latest target Q4 fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779981156.json`.
- Target Q4 fused result:
  - shape `2048x8192`;
  - steps `64`;
  - weight source `file`;
  - host checksum `1055786949`;
  - q4 values per step `16777216`;
  - packed bytes per step `8388608`;
  - setup H2D `8429572` bytes;
  - steady-state H2D `0`;
  - D2H `256` bytes;
  - graph replay rate `1.0`;
  - steps/sec `5886.03`;
  - p50/p95/p99 `0.1425/0.3387/0.5747 ms`.
- `tools/bench/summarize_benchmarks.py` now includes `weight source` and `weight checksum` columns.

Native `Q4_K` first-block dequant reference:

- Added CPU-side FP16 conversion and GGML-compatible `Q4_K` scale/min extraction to `runtime_gguf_probe.exe`.
- The probe now decodes the first `blk.0.attn_qkv.weight` Q4_K block into reference stats while preserving the packed-payload dump used by `runtime_fused_bench.exe`.
- Standalone artifact: `E:\AI project\benchmarks\native_gguf_probe_IQ2_M_q4k_decode.json`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779981723.md`.
- Latest native probe harness artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1779981723.json`.
- First-block decode facts:
  - `d=8.50558e-05`;
  - `dmin=0.000560284`;
  - decoded scale checksum `408`;
  - decoded min checksum `413`;
  - dequant range `-0.0352979..0.0490355`;
  - dequant absmax `0.0490355`;
  - full payload checksum remains `1055786949`.
- Latest target Q4 fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779981723.json`.
- Target Q4 fused result:
  - shape `2048x8192`;
  - steps `64`;
  - weight source `file`;
  - host checksum `1055786949`;
  - graph replay rate `1.0`;
  - steps/sec `5805.25`;
  - p50/p95/p99 `0.1382/0.3698/0.5844 ms`;
  - setup H2D `8429572` bytes;
  - steady-state H2D `0`;
  - D2H `256` bytes.
- `tools/bench/native_harness.py` now reports `q4k dequant` and `q4k absmax`.
- `tools/bench/summarize_benchmarks.py` now reports `q4k dequant`, `q4k absmax`, and decoded scale/min checksums.

Native full-block `Q4_K` CUDA path:

- Extended `runtime_gguf_probe.exe` with `--dump-q4k-blocks`.
- Full block stream path: `E:\AI project\tmp\blk0_attn_qkv_q4_blocks.bin`.
- Full block stream bytes: `9437184`.
- Full block stream checksum: `1217417794`.
- Added `rtxllm_launch_q4k_matvec_decode`, a CUDA graph-compatible kernel path that reads full GGML Q4_K blocks and decodes FP16 scale/min metadata on device.
- Extended `runtime_fused_bench.exe` with `--q4k-blocks-file`.
- Extended `tools/bench/native_harness.py` to run `target_q4k_blocks_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779982126.md`.
- Latest packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779982126.json`.
- Latest full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779982126.json`.
- Packed-payload target result:
  - checksum `1055786949`;
  - steps/sec `5943.43`;
  - p50/p95/p99 `0.1353/0.3425/0.5983 ms`.
- Full-block Q4_K target result:
  - checksum `1217417794`;
  - graph replay rate `1.0`;
  - setup H2D `9469956` bytes;
  - D2H `256` bytes;
  - Q4_K block bytes per step `9437184`;
  - steps/sec `2555.60`;
  - p50/p95/p99 `0.3404/0.6395/1.0454 ms`.
- Interpretation: the real metadata path is about `2.33x` slower than the raw packed-payload ceiling on this shape, which gives a concrete optimization target for fusing scale/min extraction and reducing redundant block metadata work.

Full-block `Q4_K` CPU-reference gate:

- Removed the `--check-reference` limitation for `--q4k-blocks-file`.
- Added host-side FP16 decode, Q4_K scale/min extraction, and full-block CPU matvec reference inside `runtime_fused_bench.exe`.
- Extended `tools/bench/native_harness.py` with `runtime_fused_q4k_blocks_reference_check`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779982404.md`.
- Latest full-block reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_reference_check_1779982404.json`.
- Full-block reference result:
  - `reference_passed=true`;
  - tolerance `0.005`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.27177e-06`;
  - logit mismatches `0`;
  - KV mismatches `0`;
  - token expected/observed `1236/1236`;
  - source checksum `1217417794`.
- Latest packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779982404.json`.
- Latest full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779982404.json`.
- Packed-payload target result:
  - steps/sec `6195.67`;
  - p50/p95/p99 `0.1347/0.3332/0.4435 ms`.
- Full-block Q4_K target result:
  - steps/sec `2536.51`;
  - p50/p95/p99 `0.3372/0.7262/0.8970 ms`.
- The reference check confirms the slower full-block path is a correctness-preserving real Q4_K metadata path, not just a byte-consumption benchmark.

Shared-scale `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_shared`.
- Added `runtime_fused_bench.exe --q4k-kernel direct|shared`.
- The shared variant decodes the eight scale/min pairs for each 256-value Q4_K block once into shared memory, then reuses them for the 128 packed bytes in that block.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_shared_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_shared_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779982691.md`.
- Direct full-block reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_reference_check_1779982691.json`.
- Shared full-block reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_shared_reference_check_1779982691.json`.
- Both direct and shared reference checks passed:
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.27177e-06`;
  - token expected/observed `1236/1236`.
- Latest packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779982691.json`.
- Latest direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779982691.json`.
- Latest shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779982691.json`.
- Target results:
  - packed-payload ceiling: `6031.08` steps/sec, p50/p95/p99 `0.1341/0.3218/0.6247 ms`;
  - direct full-block Q4_K: `2524.86` steps/sec, p50/p95/p99 `0.3432/0.6392/0.8985 ms`;
  - shared full-block Q4_K: `3423.39` steps/sec, p50/p95/p99 `0.2470/0.4739/0.7006 ms`.
- Interpretation: shared scale/min staging improves the real full-block Q4_K path by about `35.6%` over the direct metadata decode path, while preserving the same CPU-reference result.

Warp `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_warp`.
- Extended `runtime_fused_bench.exe --q4k-kernel` to support `direct`, `shared`, and `warp`.
- The warp variant assigns one warp to one output row, processes four packed-byte groups per Q4_K block per lane, and uses warp shuffle reduction instead of a block-wide shared-memory reduction.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_warp_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_warp_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779982987.md`.
- Warp full-block reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_warp_reference_check_1779982987.json`.
- Warp reference result:
  - `reference_passed=true`;
  - max logit abs error `7.62939e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`.
- Latest packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779982987.json`.
- Latest direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779982987.json`.
- Latest shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779982987.json`.
- Latest warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779982987.json`.
- Target results:
  - packed-payload ceiling: `5868.01` steps/sec, p50/p95/p99 `0.1453/0.3198/0.6020 ms`;
  - direct full-block Q4_K: `2562.30` steps/sec, p50/p95/p99 `0.3422/0.5986/0.7817 ms`;
  - shared full-block Q4_K: `2851.17` steps/sec, p50/p95/p99 `0.2583/0.8407/1.3941 ms`;
  - warp full-block Q4_K: `3154.51` steps/sec, p50/p95/p99 `0.2355/0.6988/1.4172 ms`.
- Interpretation: the warp variant is now the best full-block Q4_K path in the latest harness, about `23.1%` faster than direct and `10.6%` faster than shared by steps/sec. Its p99 is worse than direct, so jitter still needs attention.

Warp-broadcast `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_warp_broadcast`.
- Extended `runtime_fused_bench.exe --q4k-kernel` to support `direct`, `shared`, `warp`, and `broadcast`.
- The broadcast variant keeps one warp per row, but only lanes `0..7` decode the eight Q4_K scale/min pairs for each block and then broadcasts those values with warp shuffle instructions.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_broadcast_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_broadcast_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779983348.md`.
- Warp-broadcast reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_broadcast_reference_check_1779983348.json`.
- Warp-broadcast reference result:
  - `reference_passed=true`;
  - max logit abs error `7.62939e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`.
- Latest packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779983348.json`.
- Latest direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779983348.json`.
- Latest shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779983348.json`.
- Latest warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779983348.json`.
- Latest warp-broadcast full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_broadcast_blk0_1779983348.json`.
- Target results:
  - packed-payload ceiling: `6069.57` steps/sec, p50/p95/p99 `0.1321/0.3543/0.6090 ms`;
  - direct full-block Q4_K: `2586.58` steps/sec, p50/p95/p99 `0.3380/0.6278/0.7851 ms`;
  - shared full-block Q4_K: `3370.00` steps/sec, p50/p95/p99 `0.2510/0.4981/0.6950 ms`;
  - warp full-block Q4_K: `3629.50` steps/sec, p50/p95/p99 `0.2348/0.4544/0.6696 ms`;
  - warp-broadcast full-block Q4_K: `4434.71` steps/sec, p50/p95/p99 `0.1931/0.3670/0.6578 ms`.
- Interpretation: warp-broadcast is the current best real Q4_K path, about `72.8%` faster than direct and `22.2%` faster than plain warp by steps/sec in the latest harness, while also improving p95/p99 over the other full-block variants.

Next bottlenecks:

- Add a longer decode/context profile for `IQ2_M` so the comparison is less dominated by load time and tiny prompt effects.
- Keep optimizing the full-block Q4_K kernel: next candidates are multi-warp rows, wider output tiling, and comparing p95/p99 stability over longer replay runs.

Warp-broadcast-vec4 `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4`.
- Extended `runtime_fused_bench.exe --q4k-kernel` to support `direct`, `shared`, `warp`, `broadcast`, and `vec4`.
- The first vec4 attempt reduced Q4 payload load instruction count but did not beat broadcast in a same-run 64-step harness, because the activation reads became the next bottleneck.
- The promoted vec4 version keeps lanes mapped to four-byte Q4 chunks and also uses aligned `float4` activation loads for both halves of each Q4_K group.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_vec4_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_vec4_blk0`.
- The target Q4_K benchmark rows now use `128` graph replays instead of `64` so p95/p99 comparisons are less dominated by short-run jitter.
- Vec4 comparison harness: `E:\AI project\benchmarks\native_harness_1779983896.md`.
- Warp-broadcast-vec4 reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_vec4_reference_check_1779983896.json`.
- Warp-broadcast-vec4 reference result:
  - `reference_passed=true`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`.
- Vec4 packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779983896.json`.
- Vec4 direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779983896.json`.
- Vec4 shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779983896.json`.
- Vec4 warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779983896.json`.
- Vec4 warp-broadcast full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_broadcast_blk0_1779983896.json`.
- Vec4 warp-broadcast-vec4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4_blk0_1779983896.json`.
- 128-step target results:
  - packed-payload ceiling: `5671.87` steps/sec, p50/p95/p99 `0.1374/0.3787/0.5945 ms`;
  - direct full-block Q4_K: `2517.46` steps/sec, p50/p95/p99 `0.3420/0.7593/0.7910 ms`;
  - shared full-block Q4_K: `3401.60` steps/sec, p50/p95/p99 `0.2532/0.4808/0.7081 ms`;
  - warp full-block Q4_K: `3601.92` steps/sec, p50/p95/p99 `0.2375/0.4596/0.6886 ms`;
  - warp-broadcast full-block Q4_K: `4159.95` steps/sec, p50/p95/p99 `0.2014/0.4215/0.6441 ms`;
  - warp-broadcast-vec4 full-block Q4_K: `4669.59` steps/sec, p50/p95/p99 `0.1808/0.3854/0.6125 ms`.
- Interpretation at this point: vec4 became the best one-warp real full-block Q4_K path, about `12.3%` faster than broadcast and `85.5%` faster than direct by steps/sec, while also improving p50/p95/p99 over broadcast in the same 128-step run. It was superseded by the two-warp row split below.

Warp-broadcast-vec4x2 `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x2`.
- Extended `runtime_fused_bench.exe --q4k-kernel` to support `direct`, `shared`, `warp`, `broadcast`, `vec4`, and `vec4x2`.
- The vec4x2 variant assigns two warps to one output row, splits Q4_K block indices across those warps, keeps the vec4 packed-weight and `float4` activation loads, and combines the two warp partial sums in shared memory before writing logits/KV.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_vec4x2_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_vec4x2_blk0`.
- Vec4x2 comparison harness: `E:\AI project\benchmarks\native_harness_1779984219.md`.
- Warp-broadcast-vec4x2 reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_vec4x2_reference_check_1779984219.json`.
- Warp-broadcast-vec4x2 reference result:
  - `reference_passed=true`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`.
- Vec4x2 packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779984219.json`.
- Vec4x2 direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779984219.json`.
- Vec4x2 shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779984219.json`.
- Vec4x2 warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779984219.json`.
- Vec4x2 warp-broadcast full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_broadcast_blk0_1779984219.json`.
- Vec4x2 warp-broadcast-vec4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4_blk0_1779984219.json`.
- Vec4x2 warp-broadcast-vec4x2 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x2_blk0_1779984219.json`.
- 128-step target results:
  - packed-payload ceiling: `5804.67` steps/sec, p50/p95/p99 `0.1377/0.3438/0.5558 ms`;
  - direct full-block Q4_K: `2494.88` steps/sec, p50/p95/p99 `0.3435/0.7494/0.8058 ms`;
  - shared full-block Q4_K: `3302.16` steps/sec, p50/p95/p99 `0.2552/0.5536/0.6832 ms`;
  - warp full-block Q4_K: `3650.46` steps/sec, p50/p95/p99 `0.2294/0.4782/0.6790 ms`;
  - warp-broadcast full-block Q4_K: `4253.89` steps/sec, p50/p95/p99 `0.1974/0.4051/0.6516 ms`;
  - warp-broadcast-vec4 full-block Q4_K: `4736.34` steps/sec, p50/p95/p99 `0.1799/0.3789/0.5158 ms`;
  - warp-broadcast-vec4x2 full-block Q4_K: `5991.58` steps/sec, p50/p95/p99 `0.1395/0.3307/0.4674 ms`.
- Interpretation at this point: vec4x2 became the best real full-block Q4_K path in that harness, about `40.9%` faster than broadcast, `26.5%` faster than one-warp vec4, and `140.2%` faster than direct by steps/sec. It was superseded by the four-warp row split below.

Warp-broadcast-vec4x4 `Q4_K` kernel variant:

- Added `rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x4`.
- Extended `runtime_fused_bench.exe --q4k-kernel` to support `direct`, `shared`, `warp`, `broadcast`, `vec4`, `vec4x2`, and `vec4x4`.
- The vec4x4 variant assigns four warps to one output row, splits Q4_K block indices across the warps, keeps the vec4 packed-weight and `float4` activation loads, and combines four warp partial sums in shared memory before writing logits/KV.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_vec4x4_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_vec4x4_blk0`.
- Vec4x4 comparison harness: `E:\AI project\benchmarks\native_harness_1779984549.md`.
- Vec4x4 comparison reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_vec4x4_reference_check_1779984549.json`.
- Warp-broadcast-vec4x4 reference result:
  - `reference_passed=true`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`.
- Vec4x4 comparison packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779984549.json`.
- Vec4x4 comparison direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779984549.json`.
- Vec4x4 comparison shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779984549.json`.
- Vec4x4 comparison warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779984549.json`.
- Vec4x4 comparison warp-broadcast full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_broadcast_blk0_1779984549.json`.
- Vec4x4 comparison warp-broadcast-vec4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4_blk0_1779984549.json`.
- Vec4x4 comparison warp-broadcast-vec4x2 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x2_blk0_1779984549.json`.
- Vec4x4 comparison warp-broadcast-vec4x4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x4_blk0_1779984549.json`.
- 128-step target results:
  - packed-payload ceiling: `5837.58` steps/sec, p50/p95/p99 `0.1361/0.3779/0.6012 ms`;
  - direct full-block Q4_K: `2537.72` steps/sec, p50/p95/p99 `0.3377/0.6655/0.9793 ms`;
  - shared full-block Q4_K: `3328.92` steps/sec, p50/p95/p99 `0.2526/0.5575/0.8468 ms`;
  - warp full-block Q4_K: `3534.84` steps/sec, p50/p95/p99 `0.2375/0.5187/0.7281 ms`;
  - warp-broadcast full-block Q4_K: `4374.05` steps/sec, p50/p95/p99 `0.1906/0.3973/0.7896 ms`;
  - warp-broadcast-vec4 full-block Q4_K: `4731.09` steps/sec, p50/p95/p99 `0.1791/0.3981/0.6274 ms`;
  - warp-broadcast-vec4x2 full-block Q4_K: `5649.37` steps/sec, p50/p95/p99 `0.1393/0.3835/0.5403 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6178.89` steps/sec, p50/p95/p99 `0.1275/0.3389/0.5409 ms`.
- Interpretation at this point: vec4x4 became the best real full-block Q4_K path for the `2048x8192` target block, about `41.3%` faster than broadcast, `9.4%` faster than vec4x2, and `143.5%` faster than direct by steps/sec in that harness. It temporarily exceeded the packed-payload row because the packed-payload path was still the older packed2 one-block-per-row kernel; that comparison is superseded by the packed-payload vec4x4 ceiling below.

Next bottlenecks:

- Compare one, two, and four warps per row across larger `cols` buckets to find the break-even shape.
- Move from isolated `blk.0.attn_qkv.weight` benchmarking toward layer-sequence execution with real tensor-role scheduling.

Packed-payload vec4x4 ceiling:

- Added `rtxllm_launch_q4_matvec_decode_vec4x4` for the raw packed-Q4 payload path.
- Extended `runtime_fused_bench.exe --packed-kernel` to support `block` and `vec4x4`.
- The packed vec4x4 variant assigns four warps to one output row, uses one aligned `uint32_t` packed-Q4 load per lane iteration, uses two aligned `float4` activation loads, and reduces four warp partials through shared memory.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_packed_vec4x4_reference_check`;
  - `runtime_fused_bench_target_q4k_payload_vec4x4_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779984954.md`.
- Packed-payload vec4x4 reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_packed_vec4x4_reference_check_1779984954.json`.
- Packed-payload vec4x4 reference result:
  - `reference_passed=true`;
  - max logit abs error `5.24521e-05`;
  - max KV abs error `3.62396e-05`;
  - token expected/observed `288/288`.
- Latest default packed-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blk0_1779984954.json`.
- Latest packed-payload vec4x4 target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_payload_vec4x4_blk0_1779984954.json`.
- Latest direct full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_blk0_1779984954.json`.
- Latest shared full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_shared_blk0_1779984954.json`.
- Latest warp full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_warp_blk0_1779984954.json`.
- Latest warp-broadcast full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_broadcast_blk0_1779984954.json`.
- Latest warp-broadcast-vec4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4_blk0_1779984954.json`.
- Latest warp-broadcast-vec4x2 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x2_blk0_1779984954.json`.
- Latest warp-broadcast-vec4x4 full-block target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x4_blk0_1779984954.json`.
- 128-step target results:
  - default packed-payload ceiling: `5795.95` steps/sec, p50/p95/p99 `0.1386/0.3531/0.6140 ms`;
  - packed-payload vec4x4 ceiling: `9298.95` steps/sec, p50/p95/p99 `0.0837/0.2919/0.3967 ms`;
  - direct full-block Q4_K: `2196.07` steps/sec, p50/p95/p99 `0.3570/1.0146/1.0941 ms`;
  - shared full-block Q4_K: `2902.30` steps/sec, p50/p95/p99 `0.2654/0.6460/0.6745 ms`;
  - warp full-block Q4_K: `3123.64` steps/sec, p50/p95/p99 `0.2482/0.6223/0.6924 ms`;
  - warp-broadcast full-block Q4_K: `3781.99` steps/sec, p50/p95/p99 `0.2073/0.5442/0.6383 ms`;
  - warp-broadcast-vec4 full-block Q4_K: `3967.64` steps/sec, p50/p95/p99 `0.1946/0.5285/0.5865 ms`;
  - warp-broadcast-vec4x2 full-block Q4_K: `5688.69` steps/sec, p50/p95/p99 `0.1439/0.3521/0.4878 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6214.01` steps/sec, p50/p95/p99 `0.1264/0.3205/0.5662 ms`.
- Interpretation: packed vec4x4 raises the comparable packed-payload ceiling by about `60.4%` over default packed2 and shows the best full-block Q4_K path reaches about `66.8%` of the packed vec4x4 ceiling. The remaining gap is now real Q4_K metadata/dequant overhead rather than an unfair packed-payload baseline.

Next bottlenecks:

- Reduce Q4_K metadata/dequant overhead, likely by predecoded metadata experiments or a tighter metadata broadcast path.
- Sweep one, two, and four warps per row across larger `cols` buckets to find the break-even shapes.
- Move from isolated `blk.0.attn_qkv.weight` benchmarking toward layer-sequence execution with real tensor-role scheduling.

Predecoded `Q4_K` metadata experiment:

- Added `rtxllm_launch_q4k_matvec_decode_predecoded_vec4x4`.
- Extended `runtime_fused_bench.exe --q4k-kernel` with `predecoded`.
- The predecoded variant keeps the full 144-byte Q4_K blocks resident for payload reads, but uploads a setup-time table of eight `(scale, min)` float pairs per Q4_K block. This isolates in-kernel FP16 scale/min decode and bitfield unpacking from the remaining full-block payload stride.
- Added `q4k_predecoded_meta_bytes` to benchmark JSON output.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_blocks_predecoded_reference_check`;
  - `runtime_fused_bench_target_q4k_blocks_predecoded_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779985471.md`.
- Predecoded reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_blocks_predecoded_reference_check_1779985471.json`.
- Predecoded reference result:
  - `reference_passed=true`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`;
  - metadata bytes `4194304`.
- Latest predecoded target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_predecoded_blk0_1779985471.json`.
- Same-run 128-step target results:
  - default packed-payload ceiling: `5960.70` steps/sec, p50/p95/p99 `0.1374/0.3236/0.5871 ms`;
  - packed-payload vec4x4 ceiling: `9875.48` steps/sec, p50/p95/p99 `0.0816/0.2442/0.5409 ms`;
  - direct full-block Q4_K: `2491.02` steps/sec, p50/p95/p99 `0.3440/0.7993/0.9275 ms`;
  - shared full-block Q4_K: `3350.36` steps/sec, p50/p95/p99 `0.2530/0.5596/0.7957 ms`;
  - warp full-block Q4_K: `3677.65` steps/sec, p50/p95/p99 `0.2320/0.4930/0.6680 ms`;
  - warp-broadcast full-block Q4_K: `4264.59` steps/sec, p50/p95/p99 `0.1957/0.4535/0.6673 ms`;
  - warp-broadcast-vec4 full-block Q4_K: `4495.83` steps/sec, p50/p95/p99 `0.1832/0.3934/0.7117 ms`;
  - warp-broadcast-vec4x2 full-block Q4_K: `5773.72` steps/sec, p50/p95/p99 `0.1403/0.3326/0.5857 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6659.94` steps/sec, p50/p95/p99 `0.1247/0.3166/0.4685 ms`;
  - predecoded-metadata full-block Q4_K: `8245.62` steps/sec, p50/p95/p99 `0.1019/0.2695/0.4329 ms`.
- Interpretation: predecoding Q4_K metadata improves the four-warp full-block path by about `23.8%` and reaches about `83.5%` of the packed vec4x4 ceiling in this harness. That proves scale/min decode and bitfield unpacking are a major remaining hot-path cost, while the remaining gap is likely full-block payload stride, extra metadata bandwidth, and row-reduction/scheduling overhead.

Next bottlenecks:

- Convert the predecoded experiment into an offline packed internal layout that stores payload and metadata separately, then benchmark it against both full-block Q4_K and raw packed payload.
- Sweep the predecoded path across row/column buckets to see whether the 4 MiB metadata side table remains worthwhile at different layer shapes.
- Start a layer-sequence runner that schedules multiple real tensor roles instead of only isolated `blk.0.attn_qkv.weight`.

Split-payload predecoded `Q4_K` internal-layout experiment:

- Added `rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4`.
- Extended `runtime_fused_bench.exe` with `--q4k-payload-file` and `--q4k-kernel split-predecoded`.
- This variant uses the full Q4_K blocks only on the CPU side for reference and metadata predecode, but uploads the staged 128-byte Q4 payload stream plus the 4 MiB predecoded `(scale, min)` side table as the runtime weight representation.
- Added JSON fields for `q4k_blocks_file`, `q4k_payload_file`, and `q4k_source_checksum`, so the benchmark distinguishes uploaded payload checksum from source block checksum.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_split_predecoded_reference_check`;
  - `runtime_fused_bench_target_q4k_split_predecoded_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779985908.md`.
- Split-payload reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_predecoded_reference_check_1779985908.json`.
- Split-payload reference result:
  - `reference_passed=true`;
  - max logit abs error `7.39098e-06`;
  - max KV abs error `7.03335e-06`;
  - token expected/observed `1236/1236`;
  - payload checksum `1055786949`;
  - source checksum `1217417794`;
  - metadata bytes `4194304`.
- Latest split-payload target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_predecoded_blk0_1779985908.json`.
- Same-run 128-step target results:
  - default packed-payload ceiling: `5716.51` steps/sec, p50/p95/p99 `0.1386/0.3707/0.5980 ms`;
  - packed-payload vec4x4 ceiling: `9201.42` steps/sec, p50/p95/p99 `0.0877/0.2676/0.4520 ms`;
  - direct full-block Q4_K: `2544.48` steps/sec, p50/p95/p99 `0.3385/0.7846/0.8538 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6610.14` steps/sec, p50/p95/p99 `0.1237/0.3078/0.5998 ms`;
  - predecoded-metadata full-block Q4_K: `8372.96` steps/sec, p50/p95/p99 `0.0994/0.2831/0.5352 ms`;
  - split-payload predecoded Q4_K: `8680.61` steps/sec, p50/p95/p99 `0.0946/0.2658/0.4130 ms`.
- Interpretation: split-payload predecode is about `3.7%` faster than full-block predecode in the same harness, about `31.3%` faster than in-kernel vec4x4 Q4_K, and about `94.3%` of the packed vec4x4 ceiling. The full-block 144-byte stride is now a smaller cost than metadata decode, but the internal split layout still improves p50/p95/p99 and reduces setup residency by one MiB for this tensor.

Next bottlenecks:

- Reduce the metadata side table footprint from float pairs toward packed half/scale encodings and measure whether lower bandwidth beats conversion cost.
- Add shape sweeps for split-predecoded across representative `attn_qkv`, `ffn_gate`, `ffn_up`, and `ffn_down` dimensions.
- Start layer-sequence execution so the scheduler and graph path see multiple real tensors instead of one isolated matvec.

Split-payload half-metadata `Q4_K` experiment:

- Added `rtxllm_launch_q4k_matvec_decode_split_half_vec4x4`.
- Extended `runtime_fused_bench.exe --q4k-kernel` with `split-half`.
- Added `q4k_predecoded_meta_format` JSON output so FP32 and FP16 metadata runs are distinguishable in benchmark artifacts.
- This variant keeps the staged 128-byte Q4 payload stream from `split-predecoded`, but stores the eight `(scale, min)` pairs per Q4_K block as FP16 half values instead of FP32 values.
- The split-half reference gate uses an explicit lossy tolerance of `0.0125`; the original Q4_K block semantics use FP16 `d/dmin` values multiplied by integer scale/min values in FP32, so precomputing those products into FP16 is not exact.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_split_half_reference_check`;
  - `runtime_fused_bench_target_q4k_split_half_blk0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779986416.md`.
- Split-half reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_half_reference_check_1779986416.json`.
- Split-half reference result:
  - `reference_passed=true`;
  - tolerance `0.0125`;
  - max logit abs error `0.0090878`;
  - max KV abs error `0.0090878`;
  - token expected/observed `1236/1236`;
  - payload checksum `1055786949`;
  - source checksum `1217417794`;
  - metadata bytes `2097152`;
  - metadata format `fp16`.
- Latest split-half target artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_half_blk0_1779986416.json`.
- Same-run 128-step target results:
  - packed-payload vec4x4 ceiling: `9910.88` steps/sec, p50/p95/p99 `0.083/0.245/0.609 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6609.28` steps/sec, p50/p95/p99 `0.128/0.297/0.549 ms`;
  - predecoded-metadata full-block Q4_K: `8333.01` steps/sec, p50/p95/p99 `0.099/0.277/0.390 ms`;
  - split-payload predecoded Q4_K: `8769.05` steps/sec, p50/p95/p99 `0.096/0.259/0.483 ms`;
  - split-payload half-metadata Q4_K: `8547.64` steps/sec, p50/p95/p99 `0.097/0.257/0.367 ms`.
- Interpretation: split-half cuts setup metadata from `4194304` to `2097152` bytes, but is about `2.5%` slower than FP32 split-predecoded in the same harness. The FP32 split-predecoded path remains the speed winner for this tensor; split-half is a residency knob and a signal that half conversion cost can outweigh metadata bandwidth savings.

Next bottlenecks:

- Try a compact exact metadata layout that preserves Q4_K semantics without the 4 MiB FP32 side table.
- Test `half2` or packed half-pair loads to see whether split-half can recover the conversion overhead.
- Add shape sweeps for split-predecoded and split-half across representative tensor roles before deciding which metadata format belongs in the resident packer.

Exact compact metadata `Q4_K` experiments:

- Added `rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4`.
- Added `rtxllm_launch_q4k_matvec_decode_split_native_vec4x4`.
- Extended `runtime_fused_bench.exe --q4k-kernel` with `split-compact` and `split-native`.
- `split-compact` stores the staged 128-byte Q4 payload plus a 20-byte-per-block metadata stream: FP16 `d/dmin`, eight unpacked scale bytes, and eight unpacked min bytes.
- `split-native` stores the staged 128-byte Q4 payload plus the original 16-byte Q4_K metadata stream, separating metadata from payload but preserving the original scale/min bit packing.
- Extended `tools/bench/native_harness.py` with:
  - `runtime_fused_q4k_split_compact_reference_check`;
  - `runtime_fused_q4k_split_native_reference_check`;
  - `runtime_fused_bench_target_q4k_split_compact_blk0`;
  - `runtime_fused_bench_target_q4k_split_native_blk0`.
- Harness for this experiment: `E:\AI project\benchmarks\native_harness_1779987166.md`.
- Compact/native reference results:
  - `split-compact`: `reference_passed=true`, tolerance `0.005`, max logit/KV error `7.39098e-06/7.03335e-06`, metadata bytes `1310720`, metadata format `compact-u8`;
  - `split-native`: `reference_passed=true`, tolerance `0.005`, max logit/KV error `7.39098e-06/7.03335e-06`, metadata bytes `1048576`, metadata format `native16`.
- Same-run 128-step target results:
  - packed-payload vec4x4 ceiling: `9461.58` steps/sec, p50/p95/p99 `0.085/0.239/0.503 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `6606.18` steps/sec, p50/p95/p99 `0.127/0.317/0.561 ms`;
  - predecoded-metadata full-block Q4_K: `7610.26` steps/sec, p50/p95/p99 `0.105/0.298/0.424 ms`;
  - split-payload predecoded Q4_K: `8114.67` steps/sec, p50/p95/p99 `0.095/0.284/0.406 ms`;
  - split-payload half-metadata Q4_K: `8503.29` steps/sec, p50/p95/p99 `0.094/0.282/0.521 ms`;
  - split-payload compact metadata Q4_K: `6533.18` steps/sec, p50/p95/p99 `0.125/0.317/0.548 ms`;
  - split-payload native metadata Q4_K: `6549.22` steps/sec, p50/p95/p99 `0.125/0.314/0.545 ms`.
- Interpretation: exact compact metadata saves residency (`1.25 MiB` and `1 MiB`) and passes the strict reference gate, but it is not a speed win. It performs roughly like the in-kernel four-warp full-block path, which shows the remaining cost is hot-path Q4_K metadata reconstruction, not mainly the 144-byte GGML block stride. The speed path remains precomputed products or another exact representation that avoids per-token scale/min reconstruction.

Next bottlenecks:

- Prototype an exact fixed-point or packed-FP32-sidecar scheme that avoids hot-path `d/dmin * scale/min` reconstruction without returning to a full 4 MiB FP32 table.
- Run shape sweeps before promoting any metadata layout into the offline resident packer.
- Start a layer-sequence runner so metadata decisions are measured across multiple tensor roles, not only `blk.0.attn_qkv.weight`.

Allocator reset ordering and Q4_K shape sweep:

- Added reference-debug telemetry to `runtime_fused_bench.exe`: max-error row/index plus expected/observed logit values.
- Found that the failing `blk.3.attn_v.weight` references were not a Q4_K decode issue. The superpool reset used default-stream `cudaMemset`, then uploads and kernels ran on a non-blocking stream, so reset could race and zero/corrupt benchmark buffers.
- Fixed `VramSuperallocator::reset()` by synchronizing after the reset memset. This is a prototype correctness fix; the production allocator should grow a stream-ordered reset API instead of relying on a global sync.
- Fixed `argmax_token_kernel` tie-breaking so GPU token selection matches the CPU reference's lowest-index maximum.
- Added `tools/bench/q4k_tensor_sweep.py` to select representative Q4_K target tensors from the GGUF tensor plan, dump their payload/full-block streams, run Q4_K kernel variants, and optionally run strict reference gates.
- Latest sweep: `E:\AI project\benchmarks\q4k_tensor_sweep_1779988479.md`.
- All references passed:
  - `blk.3.attn_v.weight` max exact error around `1e-06`; split-half max error `0.003033` under its explicit `0.0125` lossy tolerance.
  - `blk.0.attn_qkv.weight` max exact error around `4e-06`; split-half max error `0.007908` under its explicit `0.0125` lossy tolerance.
- Same-run `blk.3.attn_v.weight` `[2048, 512]` results:
  - packed-payload vec4x4: `14130.70` steps/sec, p50/p95/p99 `0.055/0.227/0.326 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `14453.30` steps/sec, p50/p95/p99 `0.055/0.212/0.314 ms`;
  - predecoded-metadata full-block Q4_K: `12733.30` steps/sec, p50/p95/p99 `0.058/0.232/0.408 ms`;
  - split-payload predecoded Q4_K: `13639.70` steps/sec, p50/p95/p99 `0.051/0.238/0.517 ms`;
  - split-payload half-metadata Q4_K: `14869.00` steps/sec, p50/p95/p99 `0.056/0.183/0.271 ms`;
  - split-payload compact metadata Q4_K: `13950.80` steps/sec, p50/p95/p99 `0.053/0.226/0.379 ms`;
  - split-payload native metadata Q4_K: `13874.00` steps/sec, p50/p95/p99 `0.057/0.201/0.301 ms`.
- Same-run `blk.0.attn_qkv.weight` `[2048, 8192]` results:
  - packed-payload vec4x4: `8448.62` steps/sec, p50/p95/p99 `0.100/0.237/0.552 ms`;
  - warp-broadcast-vec4x4 full-block Q4_K: `5661.91` steps/sec, p50/p95/p99 `0.142/0.362/0.609 ms`;
  - predecoded-metadata full-block Q4_K: `7218.59` steps/sec, p50/p95/p99 `0.110/0.319/0.435 ms`;
  - split-payload predecoded Q4_K: `7382.76` steps/sec, p50/p95/p99 `0.109/0.291/0.544 ms`;
  - split-payload half-metadata Q4_K: `7419.90` steps/sec, p50/p95/p99 `0.109/0.309/0.552 ms`;
  - split-payload compact metadata Q4_K: `5818.87` steps/sec, p50/p95/p99 `0.141/0.345/0.470 ms`;
  - split-payload native metadata Q4_K: `5542.83` steps/sec, p50/p95/p99 `0.142/0.376/0.478 ms`.
- Interpretation: the internal Q4_K packer should be shape-aware. The large QKV tensor still favors precomputed products, while the small V tensor is latency dominated and does not benefit the same way from the larger FP32 side table.

Follow-up implementation:

- Added `VramSuperallocator::reset_async(cudaStream_t)`.
- Moved `runtime_bench.exe`, `runtime_graph_bench.exe`, and `runtime_fused_bench.exe` to create their non-blocking stream first, then enqueue the superpool reset on that same stream.
- Kept the synchronous `reset()` API for conservative callers, but the native benchmark path no longer pays a global device sync for setup ordering.
- Added `tools/bench/q4k_tensor_sweep.py --reference` to `run_bench_suite.py --include-native`, so the standard native suite now proves both the broad native harness and representative target-GGUF Q4_K shapes.
- Integrated suite artifact: `E:\AI project\benchmarks\bench_suite_1779988874.json`.
- Latest native harness from the suite: `E:\AI project\benchmarks\native_harness_1779988860.md`.
- Latest Q4_K tensor sweep from the suite: `E:\AI project\benchmarks\q4k_tensor_sweep_1779988867.md`.

Next bottlenecks:

- Start an offline Q4_K resident-packer prototype with per-shape metadata policy.
- Add stream-ordered arena-level reset/clear operations so future callers can avoid clearing the entire superpool.
- Add longer soak runs for the Q4_K tensor sweep once the resident packer starts writing internal layout files.

Q4_K resident packer prototype:

- Added `tools/bench/q4k_resident_pack.py`.
- Added `runtime/tests/unit/test_q4k_resident_pack.py` for K4 scale/min unpacking, Q4_K payload extraction, FP32 product metadata, exact compact metadata, native metadata, and auto policy selection.
- Added `--include-resident-pack` to `tools/bench/run_bench_suite.py`.
- Updated `tools/bench/gguf_pack_plan.py` so Q4_K now reports `q4k_resident_payload_metadata_packer` instead of remaining a generic K-quant reader gap.
- Full target `IQ2_M` Q4_K resident pack:
  - manifest: `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.md`;
  - tensors: `40`;
  - source bytes: `289013760`;
  - payload bytes: `256901120`;
  - metadata bytes: `126648320`;
  - resident bytes: `383549440`.
- Current `auto` policy:
  - large `attn_qkv` tensors `[2048, 8192]`: `split-predecoded`, payload `8388608` bytes plus FP32 product metadata `4194304` bytes;
  - small `attn_v` tensors `[2048, 512]`: `split-compact`, payload `524288` bytes plus exact compact metadata `81920` bytes.
- Representative validation pack:
  - manifest: `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\validation_reps\manifest.md`;
  - includes packer-written payload and full-block files for `blk.0.attn_qkv.weight` and `blk.3.attn_v.weight`.
- Native CUDA reference gates against packer-created bytes:
  - `blk.0.attn_qkv.weight` with `split-predecoded`: `reference_passed=true`, max logit/KV abs error `3.93391e-06/1.3113e-06`, token expected/observed `4560/4560`, source checksum `1217417794`, payload checksum `1055786949`, metadata bytes `4194304`;
  - `blk.3.attn_v.weight` with `split-compact`: `reference_passed=true`, max logit/KV abs error `1.19209e-06/1.19209e-06`, token expected/observed `324/324`, source checksum `75259501`, payload checksum `65934804`, metadata bytes `81920`.

Native manifest consumption:

- Extended `runtime_fused_bench.exe` with:
  - `--q4k-manifest <manifest.json>`;
  - `--q4k-tensor <tensor name>`;
  - `--q4k-meta-file <metadata.bin>`.
- The fused benchmark now reads a selected manifest tensor, validates payload and metadata byte checksums, uploads the packed payload plus metadata file directly, and dispatches the manifest's runtime kernel policy.
- If a validation manifest includes `blocks_path`, `--check-only` still uses those source blocks for a CPU Q4_K reference.
- Full pack manifests omit source blocks, so those rows now exercise the intended resident payload + metadata path rather than regenerating metadata from GGML blocks.
- Updated `tools/bench/native_harness.py` to detect the latest full Q4_K pack manifest and the `validation_reps` manifest, then add manifest-backed reference and benchmark rows automatically.
- Native build through the VS CUDA preset succeeded after this change.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779989802.md`.
- Manifest-backed reference artifacts:
  - `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_manifest_blk0_reference_check_1779989802.json`;
  - `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_manifest_blk3_reference_check_1779989802.json`.
- Manifest-backed full-pack benchmark artifacts:
  - `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_q4k_manifest_blk0_1779989802.json`;
  - `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_q4k_manifest_blk3_1779989802.json`.
- Latest full-pack manifest rows:
  - `blk.0.attn_qkv.weight`: `q4k_split_predecoded_vec4x4`, `16` steps, `9451.24` graph replays/sec, setup H2D `12591108` bytes, payload checksum `1055786949`, metadata checksum `343001347`;
  - `blk.3.attn_v.weight`: `q4k_split_compact_vec4x4`, `16` steps, `9307.20` graph replays/sec, setup H2D `614404` bytes, payload checksum `65934804`, metadata checksum `3895573`.

Native Q4_K sequence graph:

- Added reusable native loader module:
  - `runtime/loaders/q4k_manifest.h`;
  - `runtime/loaders/q4k_manifest.cpp`.
- Added executable:
  - `runtime/apps/q4k_sequence_benchmark_runner/main.cpp`;
  - CMake target `runtime_q4k_sequence_bench`.
- The sequence runner:
  - loads a Q4_K resident pack manifest;
  - selects named tensors or a manifest prefix via `--limit`;
  - validates payload and metadata checksums;
  - admits the combined payload plus metadata residency in strict mode;
  - uploads all selected payloads into one weights arena and all metadata into one metadata arena;
  - captures the selected tensor sequence into one CUDA Graph;
  - replays the graph with no steady-state H2D traffic and one token-sized D2H copy per replay.
- Native build succeeded and produced `E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_q4k_sequence_bench.exe`.
- Direct smoke run, `--limit 4 --steps 16`:
  - tensors: `4`;
  - resident bytes: `38354944`;
  - allocated bytes: `106553344`;
  - setup H2D: `38363140`;
  - graph kernels per replay: `4`;
  - sequence steps/sec: `3139.84`;
  - tensor launches/sec: `12559.40`.
- Direct full-pack run, `--limit 40 --steps 16`:
  - tensors: `40`;
  - resident bytes: `383549440`;
  - allocated bytes: `451747840`;
  - setup H2D: `383557636`;
  - graph kernels per replay: `40`;
  - sequence steps/sec: `398.72`;
  - tensor launches/sec: `15948.60`;
  - p50/p95/p99 replay latency: `2.152/3.303/3.303 ms`.
- The native harness now includes `runtime_q4k_sequence_bench_full40_chained` and `runtime_q4k_sequence_bench_full40_nochain`.
- Activation-chaining harness artifacts before source-plan validation:
  - chained: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_chained_1779990534.json`;
  - no-chain comparison: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_nochain_1779990534.json`.
- Chained harness sequence row before source-plan validation:
  - sequence steps/sec: `390.20`;
  - tensor launches/sec: `15608.10`;
  - graph kernels/sec: `31216.20`;
  - graph kernels per replay: `80`;
  - activation-feedback kernels per replay: `40`;
  - p50/p95/p99 replay latency: `2.281/3.301/3.301 ms`;
  - resident bytes: `383549440`;
  - setup H2D bytes: `383557636`;
  - D2H bytes: `64`.
- No-chain comparison row before source-plan validation:
  - sequence steps/sec: `396.16`;
  - tensor launches/sec: `15846.30`;
  - graph kernels per replay: `40`;
  - p50/p95/p99 replay latency: `2.247/3.322/3.322 ms`.

Q4_K sequence activation chaining:

- Added `rtxllm_launch_activation_feedback` to the native CUDA kernel library.
- The feedback kernel maps each tensor's logits back into the shared activation buffer for the next selected manifest tensor, so the captured graph now has a device-side data dependency between resident tensor stages instead of reusing one static activation vector.
- Added manifest tensor metadata parsing for layer id and role (`attn_qkv.weight`, `attn_v.weight`) in the reusable Q4_K manifest loader.
- `runtime_q4k_sequence_bench.exe` now reports `chain_activation`, `activation_feedbacks_per_graph`, `kernel_launches_per_graph`, `graph_kernel_launches_per_second`, `layer_count_observed`, and `role_counts`.
- The full sequence observes `40` layers with role counts `30` `attn_qkv.weight` and `10` `attn_v.weight`.
- Updated benchmark summarization to include sequence tensor count, observed layers, graph kernels, feedback kernels, chaining status, and total graph kernels/sec.

Q4_K sequence source-plan validation:

- Extended the reusable Q4_K manifest loader with manifest dimensions, source offset/source byte fields, and the pack manifest's `tensor_plan` path.
- Added native source tensor-plan validation for selected Q4_K sequence tensors before CUDA allocation or launch.
- Validation checks that each selected manifest tensor exists in the source GGUF tensor plan, has type `Q4_K`, has matching dimensions/rows/cols, and matches source span/absolute offset when present.
- Added `--tensor-plan PATH` to override the manifest's source plan and `--no-source-plan-validation` for explicit bypasses.
- Negative validation smoke: using the smoke-model tensor plan against `blk.0.attn_qkv.weight` fails before allocation with `tensor plan is missing manifest tensor: blk.0.attn_qkv.weight`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779990982.md`.
- Latest chained sequence artifact: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_chained_1779990982.json`.
- Latest no-chain sequence artifact: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_nochain_1779990982.json`.
- Latest chained sequence row:
  - source-plan tensors checked: `40`;
  - sequence steps/sec: `386.53`;
  - tensor launches/sec: `15461.00`;
  - graph kernels/sec: `30922.10`;
  - graph kernels per replay: `80`;
  - activation-feedback kernels per replay: `40`;
  - p50/p95/p99 replay latency: `2.079/3.412/3.412 ms`;
  - resident bytes: `383549440`;
  - setup H2D bytes: `383557636`;
  - D2H bytes: `64`.

IQ raw resident packer:

- Added `tools/bench/iq_resident_pack.py`.
- The packer supports `IQ2_S` and `IQ3_S` source tensors using the GGUF tensor plan and existing block-format metadata.
- It copies raw IQ block streams from the target GGUF to external-storage resident payload files and records dimensions, element count, block count, source offset, source bytes, payload bytes, and checksums.
- This is a loader boundary only; runtime CUDA dequant kernels and IQ manifest consumption are still pending.
- Added `runtime/tests/unit/test_iq_resident_pack.py`.
- Updated `gguf_pack_plan.py` so `IQ2_S`/`IQ3_S` are classified as `iq_raw_resident_block_packer_pending_kernel` instead of a completely missing IQ reader.
- First IQ2_S raw-pack smoke:
  - manifest: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1779991189\manifest.md`;
  - tensors: `2`;
  - resident bytes: `88670208`;
  - selected tensors: `blk.0.attn_gate.weight` and `blk.0.ffn_gate_exps.weight`;
  - source checksums: `329856869` and `6172290754`.

IQ2_S CUDA block-reader probe:

- Added `runtime/loaders/iq_manifest.{h,cpp}` for IQ raw-pack manifest loading and tensor selection.
- Added `rtxllm_launch_iq2s_probe` and `runtime/apps/iq_probe`.
- The probe validates the GGML `IQ2_S` raw-block layout at the CUDA boundary by scanning resident 82-byte blocks and comparing GPU-observed payload checksum, first-block scale bits, `qs`, `qh`, `scales`, and block count against a CPU reference.
- Added `runtime_iq_probe.exe` to CMake and `tools/bench/native_harness.py`, with latest-pack discovery under `E:\AI project\packs\iq\...\iq2_s`.
- Fixed the IQ manifest string-array parser so the generated `types` array is accepted by the native loader.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779991826.md`.
- Latest IQ2_S probe artifacts:
  - `blk.0.attn_gate.weight`: `E:\AI project\benchmarks\native_iq2s_probe_runtime_iq2s_probe_attn_gate_1779991826.json`, `reference_passed=true`, payload/checksum `2686976/329856869`, `1041.44` graph replays/sec, p50/p95/p99 `0.779/1.489/1.489 ms`;
  - `blk.0.ffn_gate_exps.weight`: `E:\AI project\benchmarks\native_iq2s_probe_runtime_iq2s_probe_ffn_gate_exps_1779991826.json`, `reference_passed=true`, payload/checksum `85983232/6172290754`, `823.11` graph replays/sec, p50/p95/p99 `0.948/1.881/1.881 ms`.
- Harness validation after this step: `60` Python unit tests passed, native build passed, and `41/41` native harness rows completed successfully.

IQ2_S dequantized matvec probe:

- Pulled the upstream GGML IQ2_S decode formula from `ggml-org/llama.cpp` and generated `runtime/kernels/decode/iq2s_kgrid_values.inl` from the `kgrid_2bit_1024` table.
- Added `rtxllm_launch_iq2s_matvec_probe`, which reconstructs IQ2_S grid magnitudes, sign bits, FP16 block scale, and packed scale nibbles inside a CUDA row matvec.
- Added `runtime/apps/iq_matvec_probe`, which derives rows/cols from the IQ manifest dimensions, uploads the raw resident payload plus a deterministic activation vector, captures the IQ2_S matvec and logits D2H copy into a CUDA Graph, and compares logits against a CPU reference.
- Added `runtime_iq_matvec_probe.exe` to CMake and `tools/bench/native_harness.py`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779995581.md`.
- Latest IQ2_S matvec artifacts:
  - `blk.0.attn_gate.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_matvec_attn_gate_1779995581.json`, `reference_passed=true`, shape `4096x2048`, checked rows `64`, max logit error `1.19209e-07`, `17448.20` graph replays/sec, p50/p95/p99 `0.053/0.070/0.070 ms`;
  - `blk.0.ffn_gate_exps.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_matvec_ffn_gate_exps_1779995581.json`, `reference_passed=true`, derived shape `131072x2048`, checked rows `8`, max logit error `9.31323e-09`, `13838.40` graph replays/sec, p50/p95/p99 `0.059/0.135/0.135 ms`.
- Harness validation after this step: native build passed, `60` Python unit tests passed, and `43/43` native harness rows completed successfully.

IQ2_S decode-style KV/token probe:

- Added `rtxllm_launch_iq2s_matvec_decode_probe`, which reuses the IQ2_S resident dequant matvec kernel, writes page-style KV entries, and runs GPU argmax token update.
- Updated `runtime/apps/iq_matvec_probe` so the captured CUDA Graph performs IQ2_S dequant matvec + KV write + token update + token-sized D2H only. Full logits and KV are copied after the timed graph replay only for CPU reference validation.
- Extended the CPU reference to simulate repeated graph replays over persistent token state and compare logits, KV arena contents, and final token.
- Updated `tools/bench/native_harness.py` rows to `runtime_iq2s_decode_attn_gate` and `runtime_iq2s_decode_ffn_gate_exps`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779996157.md`.
- Latest IQ2_S decode artifacts:
  - `blk.0.attn_gate.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_decode_attn_gate_1779996157.json`, `reference_passed=true`, shape `4096x2048`, checked rows `64`, max logit/KV error `1.19209e-07`, token expected/observed `104/104`, `20186.70` graph replays/sec, p50/p95/p99 `0.046/0.069/0.069 ms`, timed D2H `32` bytes over `8` replays;
  - `blk.0.ffn_gate_exps.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_decode_ffn_gate_exps_1779996157.json`, `reference_passed=true`, derived shape `131072x2048`, checked rows `8`, max logit/KV error `9.31323e-09`, token expected/observed `48/48`, `10244.60` graph replays/sec, p50/p95/p99 `0.044/0.284/0.284 ms`, timed D2H `32` bytes over `8` replays.
- Validation after this step: native build passed, direct IQ2_S decode probes passed, `60` Python unit tests passed, `43/43` native harness rows completed successfully, CMake preset listing passed, and benchmark summary generation succeeded.

IQ2_S source-plan-validated sequence graph:

- Added native source tensor-plan validation for IQ manifest tensors. The gate checks selected tensor name, type, dimensions, source byte span, and source offset against the GGUF tensor plan before CUDA allocation.
- Added executable:
  - `runtime/apps/iq_sequence_benchmark_runner/main.cpp`;
  - CMake target `runtime_iq_sequence_bench`.
- The IQ sequence runner:
  - loads selected raw IQ2_S manifest tensors into one resident weights arena;
  - validates payload checksums;
  - admits the combined residency request under strict VRAM policy;
  - captures a sequence graph with one IQ2_S dequant/KV/token matvec per selected tensor;
  - optionally inserts GPU activation-feedback kernels between tensor stages;
  - copies back only one token-sized value per graph replay.
- Added native harness rows:
  - `runtime_iq2s_sequence_limit2_chained`;
  - `runtime_iq2s_sequence_limit2_nochain`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779997277.md`.
- Latest IQ2_S sequence artifacts:
  - chained: `E:\AI project\benchmarks\native_iq2s_sequence_graph_runtime_iq2s_sequence_limit2_chained_1779997277.json`, `2` tensors, source-plan checks `2/2`, roles `attn_gate.weight` and `ffn_gate_exps.weight`, resident bytes `88670208`, setup H2D `88678404`, steady-state H2D `0`, D2H `32`, graph kernels/replay `4`, activation feedbacks/replay `2`, logical values/replay `262144`, `15807.20` sequence steps/sec, p50/p95/p99 `0.059/0.074/0.074 ms`;
  - no-chain: `E:\AI project\benchmarks\native_iq2s_sequence_graph_runtime_iq2s_sequence_limit2_nochain_1779997277.json`, source-plan checks `2/2`, graph kernels/replay `2`, steady-state H2D `0`, D2H `32`, `16132.30` sequence steps/sec, p50/p95/p99 `0.056/0.082/0.082 ms`.
- Validation after this step: native build passed, direct IQ2_S sequence runs passed, `45/45` native harness rows completed successfully, `60` Python unit tests passed, CMake preset listing passed, and benchmark summary generation succeeded.

Mixed Q4_K/IQ2_S layer-slice sequence graph:

- Added executable:
  - `runtime/apps/mixed_sequence_benchmark_runner/main.cpp`;
  - CMake target `runtime_mixed_sequence_bench`.
- The mixed runner:
  - loads the latest Q4_K resident-pack manifest plus the IQ2_S raw resident-pack manifest;
  - selects Q4_K and IQ2_S tensors for one layer, defaulting to layer `0`;
  - validates both manifest families against the same source GGUF tensor plan before CUDA allocation;
  - admits the combined Q4_K payload, Q4_K metadata, IQ2_S payload, KV, workspace, and DMA residency in strict mode;
  - uploads Q4_K and IQ2_S payloads into separate arenas inside one superallocator plan;
  - captures one CUDA Graph containing Q4_K and IQ2_S decode-style matvec kernels, optional activation-feedback kernels, GPU token updates, and one token-sized D2H copy per replay.
- Added native harness rows:
  - `runtime_mixed_layer0_chained`;
  - `runtime_mixed_layer0_nochain`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1779998901.md`.
- Latest mixed artifacts:
  - chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_sequence_graph_runtime_mixed_layer0_chained_1779998901.json`, `3` tensors, `3/3` source-plan checks, roles `attn_qkv.weight`, `attn_gate.weight`, and `ffn_gate_exps.weight`, resident bytes `101253120`, allocated bytes `110731264`, setup H2D `101261316`, steady-state H2D `0`, D2H `32`, graph kernels/replay `6`, activation feedbacks/replay `3`, logical values/replay `17039360`, `7698.97` sequence steps/sec, p50/p95/p99 `0.126/0.155/0.155 ms`;
  - no-chain: `E:\AI project\benchmarks\native_mixed_q4k_iq_sequence_graph_runtime_mixed_layer0_nochain_1779998901.json`, `3` tensors, `3/3` source-plan checks, graph kernels/replay `3`, steady-state H2D `0`, D2H `32`, `7193.60` sequence steps/sec, p50/p95/p99 `0.122/0.223/0.223 ms`.
- Validation after this step: native build passed, direct mixed runs passed, `47/47` native harness rows completed successfully, `60` Python unit tests passed, benchmark scripts compiled, CMake preset listing passed, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across more layer roles and batches.

Q5_K raw resident pack and native probe:

- Added `tools/bench/q5k_resident_pack.py`.
- Added unit coverage in `runtime/tests/unit/test_q5k_resident_pack.py`.
- Updated `tools/bench/gguf_pack_plan.py` so `Q5_K` reported the raw-block-only state before the later dequant probe landed.
- Added `rtxllm_launch_q5k_probe`, a CUDA resident block-reader probe that checks full payload checksum plus first-block Q5_K `d`, `dmin`, scales, `qh`, `qs`, and block count.
- Added executable:
  - `runtime/apps/q5k_probe/main.cpp`;
  - CMake target `runtime_q5k_probe`.
- Updated `tools/bench/native_harness.py` to detect the latest Q5_K pack manifest and run `runtime_q5k_probe_output`.
- Q5_K pack:
  - manifest: `E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.md`;
  - tensor: `output.weight`;
  - dimensions: `[2048, 248320]`;
  - blocks: `1986560`;
  - resident bytes: `349634560`;
  - checksum: `45716023722`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780002898.md`.
- Latest Q5_K probe artifact: `E:\AI project\benchmarks\native_q5k_probe_runtime_q5k_probe_output_1780002898.json`.
- Latest Q5_K probe row:
  - `reference_passed=true`;
  - GPU payload checksum: `45716023722`;
  - first-block fields matched CPU reference: `d=863`, `dmin=5054`, scales sum `2463`, `qh` sum `4291`, `qs` sum `16347`;
  - setup H2D: `349634560`;
  - steady-state H2D: `0`;
  - D2H: `448`;
  - graph replays/sec: `257.66`;
  - p50/p95/p99: `4.071/4.840/4.840 ms`.
- Validation after this step: native build passed, direct Q5_K probe passed, `48/48` native harness rows completed successfully, `63` Python unit tests passed, benchmark scripts compiled, CMake preset listing passed, GGUF pack plan regenerated, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across more layer roles and batches.

Q5_K dequantized matvec probe:

- Added `rtxllm_launch_q5k_matvec_decode_probe`, a CUDA Graph-safe Q5_K decode-style matvec path that mirrors GGML Q5_K scale/min and high-bit layout, writes page-style KV entries, and updates the token on GPU.
- Added executable:
  - `runtime/apps/q5k_matvec_probe/main.cpp`;
  - CMake target `runtime_q5k_matvec_probe`.
- Updated `tools/bench/native_harness.py` to run `runtime_q5k_decode_output` when the latest Q5_K pack manifest exists.
- Updated `tools/bench/gguf_pack_plan.py` so `Q5_K` now reports `q5k_raw_resident_block_packer_plus_dequant_probe`.
- Latest native harness at this step: `E:\AI project\benchmarks\native_harness_1780008122.md`.
- Q5_K dequant artifact at this step: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780008122.json`.
- Latest Q5_K dequant row:
  - `reference_passed=true`;
  - tensor: `output.weight`, shape `248320x2048`, checked rows `64`;
  - logical values/replay: `131072`;
  - max logit/KV error: `7.15256e-07`;
  - token expected/observed: `480/480`;
  - setup H2D: `349642752`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph replays/sec: `13175.20`;
  - p50/p95/p99: `0.048/0.230/0.230 ms`.
- Validation after this step: native build passed, direct Q5_K dequant probe passed, `49/49` native harness rows completed successfully, `63` Python unit tests passed, benchmark scripts compiled, and CMake preset listing passed.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across more layer roles and batches.
- Start integrating Q5_K output projection into a composed execution path instead of probing it alone.

Mixed Q4_K/IQ2_S/Q5_K output projection graph:

- Extended `runtime_mixed_sequence_bench.exe` with optional `--q5k-manifest` and `--q5k-tensor` arguments.
- Added a Q5_K stage kind backed by its own resident raw-payload arena.
- The mixed runner now validates Q4_K, IQ2_S, and Q5_K selected tensors against the same source GGUF tensor plan before launch, admits their combined residency request in strict mode, uploads each format into separate arenas, and captures one graph with Q4_K, IQ2_S, and Q5_K decode-style matvec stages.
- Added native harness rows:
  - `runtime_mixed_layer0_output_chained`;
  - `runtime_mixed_layer0_output_nochain`.
- Native harness at this step: `E:\AI project\benchmarks\native_harness_1780008569.md`.
- Mixed output artifacts at this step:
  - chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_output_chained_1780008569.json`;
  - no-chain: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_output_nochain_1780008569.json`.
- Latest chained mixed output row:
  - tensors: `4` (`blk.0.attn_qkv.weight`, `blk.0.attn_gate.weight`, `blk.0.ffn_gate_exps.weight`, `output.weight`);
  - source-plan checks: `4/4`;
  - resident bytes: `450887680`;
  - allocated bytes: `460365824`;
  - setup H2D: `450895876`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph kernels/replay: `8`;
  - activation feedbacks/replay: `4`;
  - logical values/replay: `17170432`;
  - sequence steps/sec: `5934.28`;
  - p50/p95/p99: `0.156/0.233/0.233 ms`.
- Validation after this step: native build passed, direct mixed-output graph passed, `51/51` native harness rows completed successfully, `63` Python unit tests passed, benchmark scripts compiled, CMake preset listing passed, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across more layer roles and batches.

Source-offset stage ordering for mixed graphs:

- Updated `runtime_mixed_sequence_bench.exe` so selected layer-local stages are ordered by their GGUF source offsets, while global stages such as Q5_K `output.weight` remain at the tail.
- Added JSON telemetry:
  - `stage_order_policy`;
  - `stage_source_offsets`;
  - ordered `tensors` list.
- Added `latest_q5k_pack_manifest` unit coverage in `runtime/tests/unit/test_native_harness.py`.
- Direct ordered mixed output run confirmed:
  - policy: `source_offset_with_global_tail`;
  - order: `blk.0.attn_gate.weight -> blk.0.attn_qkv.weight -> blk.0.ffn_gate_exps.weight -> output.weight`;
  - offsets: `579153536, 581848704, 707079808, 10989184`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780008907.md`.
- Latest ordered mixed output artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_output_chained_1780008907.json`.
- Latest ordered mixed output row:
  - tensors: `4`;
  - source-plan checks: `4/4`;
  - resident bytes: `450887680`;
  - allocated bytes: `460365824`;
  - setup H2D: `450895876`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph kernels/replay: `8`;
  - activation feedbacks/replay: `4`;
  - sequence steps/sec: `6543.43`;
  - p50/p95/p99: `0.146/0.182/0.182 ms`.
- Validation after this step: native build passed, direct ordered mixed-output graph passed, `51/51` native harness rows completed successfully, `64` Python unit tests passed, benchmark scripts compiled, CMake preset listing passed, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across more layer roles and batches.
- Add source-plan-driven selection for more layer roles as their raw/dequant kernels come online.

Layer-0 IQ2_S pack and broader mixed graph:

- Added `--layer N` filtering to `tools/bench/iq_resident_pack.py`.
- Added unit coverage for layered IQ tensor selection.
- Generated a new layer-0 IQ2_S resident pack:
  - manifest: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.md`;
  - tensors: `8`;
  - resident bytes: `178054144`.
- Direct broader mixed output run passed with:
  - tensors: `10`;
  - source-plan checks: `10/10`;
  - resident bytes: `540271616`;
  - graph kernels/replay: `20`;
  - order: `blk.0.attn_gate.weight -> blk.0.attn_qkv.weight -> blk.0.ffn_gate_exps.weight -> blk.0.ffn_gate_shexp.weight -> blk.0.ffn_up_exps.weight -> blk.0.ffn_up_shexp.weight -> blk.0.ssm_alpha.weight -> blk.0.ssm_beta.weight -> blk.0.ssm_out.weight -> output.weight`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780009298.md`.
- Latest broader mixed output artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_output_chained_1780009298.json`.
- Latest broader mixed output row:
  - resident bytes: `540271616`;
  - allocated bytes: `549757952`;
  - setup H2D: `540288004`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph kernels/replay: `20`;
  - activation feedbacks/replay: `10`;
  - sequence steps/sec: `4171.88`;
  - p50/p95/p99: `0.227/0.282/0.282 ms`.
- Validation after this step: direct broader mixed-output graph passed, `51/51` native harness rows completed successfully, `66` Python unit tests passed, benchmark scripts compiled, CMake preset listing passed, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Add Q4_K/IQ2_S/IQ3_S coverage for missing layer roles, especially Q4_K `attn_v` and IQ3_S `ffn_down` paths.
- Extend mixed execution across more layers once the role coverage is broader.

IQ3_S raw resident probe:

- Added an IQ3_S CUDA raw block-reader path to `runtime_iq_probe.exe`.
- The probe validates the 110-byte GGML IQ3_S block layout:
  - `d`: bytes `0..1`;
  - `qs`: `64` bytes;
  - `qh`: `8` bytes;
  - `signs`: `32` bytes;
  - scale nibbles: `4` bytes.
- Updated `tools/bench/iq_resident_pack.py` so new IQ manifests record `iq2s_raw_block_probe` and `iq3s_raw_block_probe` runtime labels instead of a pending-kernel marker.
- Added latest-pack discovery and native harness rows for `iq3_s` packs.
- Generated layer-0 IQ3_S resident pack:
  - manifest: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.md`;
  - tensors: `2`;
  - resident bytes: `115793920`;
  - tensors: `blk.0.ffn_down_exps.weight`, `blk.0.ffn_down_shexp.weight`.
- Direct IQ3_S probe runs passed for both tensors.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780009976.md`.
- Latest IQ3_S artifacts:
  - `E:\AI project\benchmarks\native_iq3s_probe_runtime_iq3s_probe_ffn_down_exps_1780009976.json`;
  - `E:\AI project\benchmarks\native_iq3s_probe_runtime_iq3s_probe_ffn_down_shexp_1780009976.json`.
- Latest `ffn_down_exps` row:
  - `reference_passed=true`;
  - payload/checksum: `115343360/11653039204`;
  - block count: `1048576`;
  - setup H2D: `115343360`;
  - steady-state H2D: `0`;
  - timed D2H: `448`;
  - graph replays/sec: `737.58`;
  - p50/p95/p99: `1.221/1.875/1.875 ms`.
- Latest `ffn_down_shexp` row:
  - `reference_passed=true`;
  - payload/checksum: `450560/53149533`;
  - block count: `4096`;
  - setup H2D: `450560`;
  - steady-state H2D: `0`;
  - timed D2H: `448`;
  - graph replays/sec: `2224.94`;
  - p50/p95/p99: `0.376/0.648/0.648 ms`.
- Validation after this step: native build passed, direct IQ3_S probes passed, `53/53` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, GGUF pack plan regenerated, and benchmark summary generation succeeded.

Next bottlenecks:

- Implement IQ3_S dequantized matvec/KV/token probe using the upstream `iq3s_grid` semantics.
- Fold IQ3_S down-projection roles into the mixed layer graph once the dequant path exists.
- Replace feedback-only activation handoff with real layer-role composition.

IQ3_S dequantized matvec probe:

- Added `runtime/kernels/decode/iq3s_grid_values.inl`, generated from upstream GGML `IQ3_S.grid_hex` into packed 4-value words.
- Added `rtxllm_launch_iq3s_matvec_decode_probe`.
- Extended `runtime_iq_matvec_probe.exe` so the same app supports both `IQ2_S` and `IQ3_S`.
- The IQ3_S CUDA and CPU reference path now dequantizes:
  - 2-byte block scale;
  - 64 selector bytes;
  - 8 high-bit bytes;
  - 32 sign bytes;
  - 4 scale-nibble bytes.
- Added native harness rows:
  - `runtime_iq3s_decode_ffn_down_exps`;
  - `runtime_iq3s_decode_ffn_down_shexp`.
- Direct IQ3_S decode probe runs passed for both layer-0 down-projection tensors.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780010641.md`.
- Latest IQ3_S dequant artifacts:
  - `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_exps_1780010641.json`;
  - `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_shexp_1780010641.json`.
- Latest `ffn_down_exps` dequant row:
  - `reference_passed=true`;
  - shape: `524288x512`;
  - checked rows: `8`;
  - logical values/replay: `4096`;
  - max logit/KV error: `1.01281e-08`;
  - token expected/observed: `56/56`;
  - setup H2D: `115345408`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph replays/sec: `8343.76`;
  - p50/p95/p99: `0.061/0.376/0.376 ms`.
- Latest `ffn_down_shexp` dequant row:
  - `reference_passed=true`;
  - shape: `2048x512`;
  - checked rows: `64`;
  - logical values/replay: `32768`;
  - max logit/KV error: `1.19209e-07`;
  - token expected/observed: `160/160`;
  - setup H2D: `452608`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph replays/sec: `17482.50`;
  - p50/p95/p99: `0.055/0.064/0.064 ms`.
- Validation after this step: native build passed, direct IQ3_S dequant probes passed, `55/55` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.
- External pack-plan regeneration was intentionally not repeated after the approval request was interrupted; the repository classifier is updated, while the existing `E:\AI project\benchmarks\gguf_pack_plan_*.json/md` may still carry the previous IQ3_S label until explicitly regenerated.

Next bottlenecks:

- Fold IQ3_S down-projection roles into a mixed layer graph.
- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across later layers where `attn_v`/`attn_output` role coverage is broader.

Mixed graph IQ3_S integration:

- Extended `runtime_mixed_sequence_bench.exe` with `--extra-iq-manifest` so one graph can load the existing IQ2_S layer pack and the IQ3_S down-projection pack together.
- Added IQ3_S stage dispatch in the mixed runner through `rtxllm_launch_iq3s_matvec_decode_probe`.
- The mixed runner now validates combined Q4_K, IQ2_S, IQ3_S, and Q5_K selections against the source GGUF tensor plan before launch.
- Added native harness rows:
  - `runtime_mixed_layer0_iq3_output_chained`;
  - `runtime_mixed_layer0_iq3_output_nochain`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780012929.md`.
- Latest mixed IQ3 artifacts:
  - `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_chained_1780012929.json`;
  - `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_nochain_1780012929.json`.
- Latest chained row:
  - tensors: `12`;
  - source-plan checks: `12/12`;
  - type counts: `IQ2_S=8`, `IQ3_S=2`, `Q4_K=1`, `Q5_K=1`;
  - resident bytes: `656065536`;
  - allocated bytes: `665551872`;
  - setup H2D: `656081924`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph kernels/replay: `24`;
  - activation feedback kernels/replay: `12`;
  - logical values/replay: `18022400`;
  - sequence steps/sec: `3251.37`;
  - p50/p95/p99: `0.228/0.532/0.532 ms`.
- Latest no-chain row:
  - tensors: `12`;
  - source-plan checks: `12/12`;
  - graph kernels/replay: `12`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - sequence steps/sec: `3248.34`.
- Validation after this step: native build passed, direct mixed IQ3 smoke passed, `57/57` native harness rows completed successfully, `68` Python unit tests passed, and benchmark scripts compiled.
- External pack-plan regeneration remains intentionally skipped after the interrupted approval request; repo-side classifier code is updated, while existing external `gguf_pack_plan_*.json/md` artifacts may still need explicit regeneration.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across later layers where `attn_v`/`attn_output` role coverage is broader.
- Add stricter mixed-graph reference checks over staged activation handoffs before widening tensor coverage.

Mixed graph CPU reference gate:

- Added `--check-reference` to `runtime_mixed_sequence_bench.exe`.
- The reference path resets token/KV/logits/activation, runs one staged mixed graph launch, then computes the same Q4_K/IQ2_S/IQ3_S/Q5_K matvec sequence, page-KV writes, token updates, and activation-feedback handoff on CPU.
- The runner now emits `reference_checked`, `reference_passed`, final-stage rows, token expected/observed, max final-logit error, max KV error, and mismatch counts. A checked reference failure returns a non-zero exit code.
- Added native harness row `runtime_mixed_layer0_iq3_output_reference_chained`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780013632.md`.
- Latest reference artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_chained_1780013632.json`.
- Reference row result:
  - `reference_passed=true`;
  - tensors: `12`;
  - source-plan checks: `12/12`;
  - final stage: `output.weight`;
  - final rows: `64`;
  - token expected/observed: `7671/7671`;
  - max final-logit error: `1.30385e-08`;
  - max KV error: `1.19209e-07`;
  - logit/KV mismatches: `0/0`;
  - resident bytes: `656065536`;
  - setup H2D: `656081924`;
  - steady-state H2D: `0`.
- Latest mixed IQ3 chained benchmark row: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_chained_1780013632.json`, `4371.35` sequence steps/sec, p50/p95/p99 `0.225/0.243/0.243 ms`.
- Validation after this step: native build passed, direct mixed reference smoke passed, `58/58` native harness rows completed successfully, and benchmark scripts compiled.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Extend mixed execution across later layers where Q4_K, IQ2_S, IQ3_S, and Q5_K coverage can prove more of the actual model graph.
- Promote the reference gate into any future widened mixed graph before treating its timings as meaningful.

Layer-1 mixed graph widening:

- Generated layer-specific IQ resident packs under external storage so the default layer-0 pack discovery remains stable:
  - IQ2_S layer 1: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201\manifest.md`, `8` tensors, `178054144` resident bytes;
  - IQ3_S layer 1: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201\manifest.md`, `2` tensors, `115793920` resident bytes.
- Added `latest_iq_layer_pack_manifest(...)` to `tools/bench/native_harness.py`.
- Added native harness rows:
  - `runtime_mixed_layer1_iq3_output_chained`;
  - `runtime_mixed_layer1_iq3_output_reference_chained`.
- Focused layer-1 reference smoke passed before the full harness:
  - `reference_passed=true`;
  - tensors: `12`;
  - source-plan checks: `12/12`;
  - final stage: `output.weight`;
  - token expected/observed: `4359/4359`;
  - max final-logit error: `1.02445e-08`;
  - max KV error: `6.14673e-08`;
  - steady-state H2D: `0`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780028031.md`.
- Latest layer-1 chained artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_chained_1780028031.json`.
- Latest layer-1 reference artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_chained_1780028031.json`.
- Layer-1 chained row:
  - tensors: `12`;
  - source-plan checks: `12/12`;
  - type counts: `IQ2_S=8`, `IQ3_S=2`, `Q4_K=1`, `Q5_K=1`;
  - resident bytes: `656065536`;
  - allocated bytes: `665551872`;
  - setup H2D: `656081924`;
  - steady-state H2D: `0`;
  - timed D2H: `32`;
  - graph kernels/replay: `24`;
  - activation feedback kernels/replay: `12`;
  - sequence steps/sec: `3758.52`;
  - p50/p95/p99: `0.226/0.421/0.421 ms`.
- Layer-1 reference row:
  - `reference_passed=true`;
  - token expected/observed: `4359/4359`;
  - max final-logit error: `1.02445e-08`;
  - max KV error: `6.14673e-08`;
  - logit/KV mismatches: `0/0`;
  - sequence steps/sec: `4571.43`;
  - p50/p95/p99: `0.214/0.231/0.231 ms`.
- Validation after this step: direct layer-1 mixed reference smoke passed, `60/60` native harness rows completed successfully, `68` Python unit tests passed, and benchmark scripts compiled.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition.
- Add staged per-role reference checks before widening from representative output slices to a fuller layer executor.
- Expand later-layer pack coverage without disrupting the stable layer-0 benchmark discovery path.

Mixed graph per-stage reference telemetry:

- Refactored `runtime_mixed_sequence_bench.exe` so the checked reference path can launch each mixed stage individually outside the timed graph, copy that stage's logits/token before activation feedback, then continue the same device-side activation handoff.
- The timed CUDA Graph path remains unchanged; the extra stage snapshots are only produced under `--check-reference`.
- Added `reference_stage_count`, `reference_stage_mismatches`, and `reference_stage_results` to checked mixed-run JSON.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780028725.md`.
- Latest layer-0 reference artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_chained_1780028725.json`.
- Latest layer-1 reference artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_chained_1780028725.json`.
- Layer-0 reference result:
  - `reference_passed=true`;
  - `reference_stage_count=12`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - max final-logit error: `1.30385e-08`;
  - max KV error: `1.19209e-07`.
- Layer-1 reference result:
  - `reference_passed=true`;
  - `reference_stage_count=12`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - max final-logit error: `1.02445e-08`;
  - max KV error: `6.14673e-08`.
- Layer-1 stage results all passed individually, including roles `attn_gate.weight`, `attn_qkv.weight`, `ffn_down_exps.weight`, `ffn_down_shexp.weight`, `ffn_gate_exps.weight`, `ffn_gate_shexp.weight`, `ffn_up_exps.weight`, `ffn_up_shexp.weight`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_out.weight`, and `output`.
- Validation after this step: native build passed, direct layer-1 stage-check smoke passed, `60/60` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace feedback-only activation handoff with real layer-role composition using the new per-stage reference snapshots as the guardrail.
- Add a layer-role executor skeleton that names attention, SSM, FFN, and output-projection phases explicitly before changing math semantics.
- Keep the reference gate active on both layer 0 and layer 1 for any widened dataflow.

Mixed graph explicit role-plan ordering:

- Added `--stage-order source-offset|role-plan` to `runtime_mixed_sequence_bench.exe`.
- Default ordering remains `source_offset_with_global_tail`.
- At this checkpoint, `role-plan` sorted layer-local stages by explicit model-role rank (`attn_gate`, `attn_qkv`, down-first FFN rank, SSM, then global output tail) instead of relying only on source offsets. This was later corrected to gate/up/down FFN order.
- Added JSON telemetry:
  - `stage_order`;
  - `stage_order_policy`;
  - `stage_role_plan_unknown_roles`;
  - `stage_roles`.
- Added native harness rows:
  - `runtime_mixed_layer0_iq3_output_reference_roleplan`;
  - `runtime_mixed_layer1_iq3_output_reference_roleplan`.
- Focused layer-1 role-plan smoke passed:
  - `reference_passed=true`;
  - `stage_order_policy=role_plan_with_global_tail`;
  - `stage_role_plan_unknown_roles=0`;
  - `reference_stage_count=12`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - max final-logit error: `1.02445e-08`;
  - max KV error: `6.14673e-08`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780029216.md`.
- Latest layer-0 role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780029216.json`.
- Latest layer-1 role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780029216.json`.
- Layer-0 role-plan result:
  - `reference_passed=true`;
  - `stage_role_plan_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - max final-logit error: `1.30385e-08`;
  - max KV error: `1.19209e-07`;
  - sequence steps/sec: `3527.03`;
  - p50/p95/p99: `0.231/0.450/0.450 ms`.
- Layer-1 role-plan result:
  - `reference_passed=true`;
  - `stage_role_plan_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - max final-logit error: `1.02445e-08`;
  - max KV error: `6.14673e-08`;
  - sequence steps/sec: `4440.50`;
  - p50/p95/p99: `0.224/0.235/0.235 ms`.
- Validation after this step: native build passed, direct layer-1 role-plan smoke passed, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Convert the explicit role plan into a real layer-role executor instead of only sorting independent decode-style matvec stages.
- Keep both source-offset and role-plan reference rows until the layer executor no longer depends on GGUF source-file order.
- Add role-phase metrics once attention, SSM, FFN, and output-projection phases become distinct graph regions.

Mixed graph phase-plan telemetry:

- Added coarse mixed-stage phase classification in `runtime_mixed_sequence_bench.exe`:
  - `attention` for `attn_*` roles;
  - `ffn` for `ffn_*` roles;
  - `ssm` for `ssm_*` roles;
  - `output` for `output` / `output.weight`.
- Added JSON telemetry:
  - `stage_phase_unknown_roles`;
  - `stage_phase_counts`;
  - `stage_phases`;
  - `stage_phase_segments`.
- Focused layer-1 role-plan phase smoke passed:
  - `reference_passed=true`;
  - `stage_order_policy=role_plan_with_global_tail`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - phase counts: `attention=2`, `ffn=6`, `ssm=3`, `output=1`;
  - phase segments: `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780029689.md`.
- Latest layer-0 role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780029689.json`.
- Latest layer-1 role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780029689.json`.
- Layer-0 role-plan result:
  - `reference_passed=true`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - phase counts: `attention=2`, `ffn=6`, `ssm=3`, `output=1`;
  - phase segments: `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`;
  - sequence steps/sec: `3365.87`;
  - p50/p95/p99: `0.229/0.494/0.494 ms`.
- Layer-1 role-plan result:
  - `reference_passed=true`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - phase counts: `attention=2`, `ffn=6`, `ssm=3`, `output=1`;
  - phase segments: `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`;
  - sequence steps/sec: `4424.78`;
  - p50/p95/p99: `0.225/0.226/0.226 ms`.
- Validation after this step: native build passed, focused layer-1 phase smoke passed, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Start replacing the independent phase segments with actual attention/SSM/FFN dataflow while keeping stage snapshots as guardrails.
- Add per-phase timing once phases become separately capturable or separately replayable graph regions.
- Extend the phase-plan checks to reject unknown roles in strict layer-executor mode rather than only reporting them.

Strict layer-plan admission gate:

- Added `--strict-layer-plan` to `runtime_mixed_sequence_bench.exe`.
- The strict gate requires:
  - `--stage-order role-plan`;
  - no unknown role-plan roles;
  - no unknown execution phases;
  - contiguous phase order `attention -> ffn -> ssm -> output`.
- Added JSON telemetry:
  - `strict_layer_plan`;
  - `strict_layer_plan_passed`;
  - `layer_plan_checked`;
  - `layer_plan_passed`;
  - `layer_plan_message`;
  - `layer_plan_expected_phases`;
  - `layer_plan_observed_phases`.
- Updated the layer-0 and layer-1 role-plan native harness rows to run with `--strict-layer-plan`.
- Focused validation:
  - positive layer-1 strict role-plan smoke passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `layer_plan_passed=true`, `reference_stage_mismatches=0`, and token `4359/4359`;
  - negative source-offset strict smoke failed early with `strict layer plan requires --stage-order role-plan`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780030171.md`.
- Latest layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780030171.json`.
- Latest layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780030171.json`.
- Layer-0 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan=true`;
  - `strict_layer_plan_passed=true`;
  - `layer_plan_message=layer phase plan validated`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - sequence steps/sec: `4191.55`;
  - p50/p95/p99: `0.231/0.260/0.260 ms`.
- Layer-1 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan=true`;
  - `strict_layer_plan_passed=true`;
  - `layer_plan_message=layer phase plan validated`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - sequence steps/sec: `4223.42`;
  - p50/p95/p99: `0.238/0.249/0.249 ms`.
- Validation after this step: native build passed, positive and negative strict-layer focused smokes behaved as expected, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Promote the strict layer plan into a reusable core scheduler/executor data structure instead of keeping the whole contract local to the benchmark runner.
- Start substituting real phase dataflow for the feedback-only phase sequence under the strict plan gate.
- Add strict-plan failure fixtures for unknown roles and phase-order mistakes once synthetic manifests can exercise them cheaply.

Core layer-plan scheduler contract:

- Promoted the mixed runner's role/phase validation into `runtime/core/scheduler/layer_plan.h` and `runtime/core/scheduler/layer_plan.cpp`.
- Added the core layer-plan implementation to `rtxllm_core` in `CMakeLists.txt`.
- The reusable contract now owns:
  - layer role ranking;
  - role-to-phase classification;
  - unknown role/phase counts;
  - phase-count and contiguous phase-segment construction;
  - strict phase-plan validation for `attention -> ffn -> ssm -> output`.
- Refactored `runtime_mixed_sequence_bench.exe` so it adapts local stages into `rtxllm::LayerStageDescriptor` and uses the core API for role sorting, telemetry, and strict admission instead of maintaining benchmark-local copies.
- Focused validation:
  - positive layer-1 strict role-plan smoke passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `layer_plan_passed=true`, `reference_stage_mismatches=0`, observed phases `attention,ffn,ssm,output`, and token `4359/4359`;
  - negative source-offset strict smoke failed early with `strict layer plan requires --stage-order role-plan`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780030754.md`.
- Latest layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780030754.json`.
- Latest layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780030754.json`.
- Layer-0 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan=true`;
  - `strict_layer_plan_passed=true`;
  - `layer_plan_message=layer phase plan validated`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - sequence steps/sec: `3325.57`;
  - p50/p95/p99: `0.240/0.458/0.458 ms`.
- Layer-1 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan=true`;
  - `strict_layer_plan_passed=true`;
  - `layer_plan_message=layer phase plan validated`;
  - `stage_role_plan_unknown_roles=0`;
  - `stage_phase_unknown_roles=0`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - sequence steps/sec: `4467.78`;
  - p50/p95/p99: `0.222/0.228/0.228 ms`.
- Validation after this step: native build passed, positive and negative strict-layer focused smokes behaved as expected, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Start substituting real phase dataflow for the feedback-only phase sequence under the core strict-plan gate.
- Add cheap strict-plan failure fixtures for unknown roles and phase-order mistakes.
- Move per-phase executor timing into the shared scheduler/executor contract once attention, FFN, SSM, and output become separately capturable units.

Core phase execution plan:

- Added `LayerPhaseExecution` and `LayerExecutionPlan` to `runtime/core/scheduler/layer_plan.h`.
- Added `build_layer_execution_plan(...)` to `runtime/core/scheduler/layer_plan.cpp`.
- The core execution plan maps validated phase segments into per-phase graph work:
  - stage count;
  - phase count;
  - matvec kernels per graph replay;
  - activation-feedback kernels per graph replay;
  - total kernels per graph replay.
- Refactored `runtime_mixed_sequence_bench.exe` so `activation_feedbacks_per_graph`, `kernel_launches_per_graph`, `tensor_launches_per_second`, and `graph_kernel_launches_per_second` come from the core execution plan instead of raw `stages.size()` arithmetic.
- Added JSON telemetry:
  - `layer_execution_phase_count`;
  - `layer_execution_stage_count`;
  - `layer_execution_matvec_kernels_per_graph`;
  - `layer_execution_activation_feedbacks_per_graph`;
  - `layer_execution_kernel_launches_per_graph`;
  - `layer_execution_phases`.
- Focused validation:
  - positive layer-1 strict role-plan smoke passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `layer_plan_passed=true`, `reference_stage_mismatches=0`, token `4359/4359`, `layer_execution_phase_count=4`, `layer_execution_stage_count=12`, and `layer_execution_kernel_launches_per_graph=24`;
  - core phase work breakdown was `attention=4`, `ffn=12`, `ssm=6`, `output=2` graph kernels;
  - negative source-offset strict smoke still failed early with `strict layer plan requires --stage-order role-plan`.
- Latest native harness: `E:\AI project\benchmarks\native_harness_1780031376.md`.
- Latest layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780031376.json`.
- Latest layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780031376.json`.
- Layer-0 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan_passed=true`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - execution phases: `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`;
  - graph kernels by phase: `4/12/6/2`;
  - sequence steps/sec: `3241.75`;
  - p50/p95/p99: `0.242/0.508/0.508 ms`.
- Layer-1 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan_passed=true`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - execution phases: `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`;
  - graph kernels by phase: `4/12/6/2`;
  - sequence steps/sec: `4014.05`;
  - p50/p95/p99: `0.227/0.285/0.285 ms`.
- Validation after this step: native build passed, focused positive/negative strict smokes passed, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Attach CUDA event timing or separately replayable graph buckets to the core phase execution plan.
- Start replacing feedback-only handoff with real phase dataflow under the same `LayerExecutionPlan` envelope.
- Add strict-plan failure fixtures for unknown-role and phase-order mistakes.

Mixed graph phase timing:

- Added phase-level CUDA event timing to `runtime_mixed_sequence_bench.exe`.
- The captured CUDA Graph benchmark path is unchanged and still measures graph replay wall time separately.
- Phase timing uses a post-graph CUDA-event phase replay over the same `LayerExecutionPlan` segments, so `layer_execution_phases` now includes:
  - `timing_samples`;
  - `timing_mean_ms`;
  - `timing_p50_ms`;
  - `timing_p95_ms`;
  - `timing_p99_ms`.
- Added JSON telemetry:
  - `layer_execution_timing_measured=true`;
  - `layer_execution_timing_method=cuda_events_phase_replay`.
- Focused validation:
  - positive layer-1 strict role-plan smoke passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `layer_plan_passed=true`, `reference_stage_mismatches=0`, and token `4359/4359`;
  - phase timing p50/p95 from the focused run: attention `0.111/0.128 ms`, FFN `0.207/0.411 ms`, SSM `0.125/0.408 ms`, output `0.044/0.066 ms`;
  - negative source-offset strict smoke still failed early with `strict layer plan requires --stage-order role-plan`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780031980.md`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780031980.json`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780031980.json`.
- Layer-0 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan_passed=true`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `7671/7671`;
  - graph replay p50/p95/p99: `0.241/0.275/0.275 ms`;
  - phase timing p50 attention/FFN/SSM/output: `0.125/0.186/0.053/0.038 ms`;
  - phase timing p95 attention/FFN/SSM/output: `0.134/0.679/0.101/1.507 ms`.
- Layer-1 strict role-plan result:
  - `reference_passed=true`;
  - `strict_layer_plan_passed=true`;
  - `reference_stage_mismatches=0`;
  - token expected/observed: `4359/4359`;
  - graph replay p50/p95/p99: `0.236/0.440/0.440 ms`;
  - phase timing p50 attention/FFN/SSM/output: `0.085/0.123/0.052/0.043 ms`;
  - phase timing p95 attention/FFN/SSM/output: `0.119/0.242/0.116/0.059 ms`.
- Validation after this step: native build passed, focused positive/negative strict smokes passed, `62/62` native harness rows completed successfully, `68` Python unit tests passed, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Use the per-phase timing to decide which phase gets the first real dataflow replacement.
- Add strict-plan failure fixtures for unknown-role and phase-order mistakes.
- Start replacing feedback-only handoff with actual attention/FFN/SSM dataflow while keeping the phase timing and stage reference gates active.

Strict layer-plan native self-test:

- Added `runtime/apps/layer_plan_selftest/main.cpp` and the `runtime_layer_plan_selftest.exe` CMake target.
- Added the self-test as the first row in `tools/bench/native_harness.py`.
- Native fixtures now cover the valid role plan, chained and unchained `LayerExecutionPlan` graph-work accounting, empty-plan rejection, unknown role rejection, unknown execution-phase rejection, phase-order rejection, segment-count rejection, and empty-segment rejection.
- Direct self-test passed: `reference_passed=true`, `cases_passed=9/9`.
- Previous native harness: `E:\AI project\benchmarks\native_harness_1780032683.md`.
- Harness result: `63/63` rows passed, including `runtime_layer_plan_selftest`.
- Previous layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780032683.json`.
- Previous layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780032683.json`, `reference_passed=true`, token `7671/7671`, CUDA-event phase timing p50 attention/FFN/SSM/output `0.097/0.140/0.090/0.026 ms`.
- Previous layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780032683.json`, `reference_passed=true`, token `4359/4359`, CUDA-event phase timing p50 attention/FFN/SSM/output `0.127/0.177/0.132/0.029 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed, full native harness passed, `68` Python unit tests passed outside the sandbox, and benchmark scripts compiled.

Next bottlenecks:

- Use the self-test as the cheap guard before changing the core layer-plan API.
- Start replacing feedback-only handoff with real phase dataflow under the same strict-plan and phase-timing envelope.
- Promote the phase execution plan from benchmark accounting into the next layer-executor skeleton.

Core layer executor replay skeleton:

- Added `runtime/core/scheduler/layer_executor.h`.
- The executor validates `LayerExecutionPlan` bounds and kernel-count consistency, then replays full plans or individual phases through stage and activation-feedback callbacks.
- Refactored `runtime_mixed_sequence_bench.exe` so graph warmup, graph capture, and CUDA-event phase timing use the core executor replay path instead of benchmark-local loops.
- Added JSON telemetry:
  - `layer_executor_contract=core_scheduler_layer_executor`;
  - warmup callback counts;
  - graph-capture callback counts.
- Expanded `runtime_layer_plan_selftest.exe` from `9` to `12` cases with full executor replay, FFN phase replay, and out-of-range phase rejection.
- Direct self-test passed: `reference_passed=true`, `cases_passed=12/12`.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, token `4359/4359`, and warmup/capture callback counts `4/12/12/24`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780033336.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780033336.json`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780033336.json`, `reference_passed=true`, token `7671/7671`, executor warmup/capture callback counts `4/12/12/24`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780033336.json`, `reference_passed=true`, token `4359/4359`, executor warmup/capture callback counts `4/12/12/24`.
- Validation after this step: native build passed, direct layer-plan self-test passed, focused strict role-plan mixed run passed, full native harness passed, and benchmark summary generation succeeded.

Checkpoint next bottlenecks:

- Replace the executor's benchmark callbacks with named phase callbacks for attention, FFN, SSM, and output.
- Keep the core executor counters as an invariant while changing phase math.
- Start moving feedback-only activation handoff toward real phase-local dataflow.

Named phase executor dispatch:

- Extended `runtime/core/scheduler/layer_executor.h` with named attention, FFN, SSM, and output phase callback counters.
- Added `replay_layer_execution_plan_by_phase(...)`, which dispatches each validated `LayerPhaseExecution` to its named phase callback and validates returned callback counts.
- Refactored `runtime_mixed_sequence_bench.exe` so the full graph replay path uses named phase callbacks before each callback reuses the current stage-level benchmark kernels.
- Added JSON telemetry for named warmup and capture phase callbacks:
  - `layer_executor_*_attention_phase_callbacks`;
  - `layer_executor_*_ffn_phase_callbacks`;
  - `layer_executor_*_ssm_phase_callbacks`;
  - `layer_executor_*_output_phase_callbacks`.
- Expanded `runtime_layer_plan_selftest.exe` from `12` to `13` cases with `executor_named_phase_dispatch_passed`.
- Direct self-test passed: `reference_passed=true`, `cases_passed=13/13`.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, token `4359/4359`, warmup named phase callbacks `1/1/1/1`, and capture named phase callbacks `1/1/1/1`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780033744.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780033744.json`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780033744.json`, `reference_passed=true`, token `7671/7671`, named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780033744.json`, `reference_passed=true`, token `4359/4359`, named warmup/capture phase callbacks `1/1/1/1`.
- Validation after this step: native build passed, direct layer-plan self-test passed, focused strict role-plan mixed run passed, full native harness passed, benchmark scripts compiled, Python unit tests passed `68/68`, and benchmark summary generation succeeded.

Next bottlenecks:

- Introduce phase-local state descriptors so attention/FFN/SSM/output callbacks no longer share one anonymous activation/logits workspace contract.
- Keep named phase callback counters stable while moving real phase math into the executor callbacks.
- Replace feedback-only activation handoff with phase-local dataflow one phase at a time.

Phase-local workspace descriptors:

- Added `LayerStageWorkspaceDescriptor`, `LayerPhaseWorkspaceDescriptor`, and `LayerExecutorWorkspacePlan` to `runtime/core/scheduler/layer_executor.h`.
- Added `build_layer_executor_workspace_plan(...)`, which validates the existing `LayerExecutionPlan`, rejects zero-sized stage workspaces, and derives per-phase `max_rows`, `max_cols`, and logical values/replay from the same stage sequence used by graph capture.
- Wired `runtime_mixed_sequence_bench.exe` to build the workspace plan before graph execution and emit:
  - `layer_executor_workspace_plan=phase_local_workspace_descriptors`;
  - `layer_executor_workspace_phase_count`;
  - `layer_executor_workspace_max_rows`;
  - `layer_executor_workspace_max_cols`;
  - `layer_executor_workspace_logical_values_per_graph`;
  - `layer_executor_phase_workspaces`.
- Expanded `runtime_layer_plan_selftest.exe` from `13` to `15` cases with `workspace_plan_passed` and `workspace_empty_stage_rejected`.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, token `4359/4359`, workspace plan `phase_local_workspace_descriptors`, four phase workspaces, global envelope `8192x4096`, logical values/replay `18022400`, and named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780034267.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780034267.json`, `cases_passed=15/15`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780034267.json`, `reference_passed=true`, token `7671/7671`, workspace plan `phase_local_workspace_descriptors`, workspace envelope `8192x4096`, logical values/replay `18022400`, named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780034267.json`, `reference_passed=true`, token `4359/4359`, workspace plan `phase_local_workspace_descriptors`, workspace envelope `8192x4096`, logical values/replay `18022400`, named warmup/capture phase callbacks `1/1/1/1`.
- Validation after this step: native build passed, direct layer-plan self-test passed, focused strict role-plan mixed run passed, full native harness passed, Python unit tests passed `68/68`, benchmark scripts compiled, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace the descriptor-only phase workspace contract with real phase-local buffers for attention, FFN, SSM, and output callbacks.
- Keep the descriptor envelope stable while moving feedback-only activation handoff into real phase dataflow.
- Add allocator telemetry for phase workspace high-water use once phase-local buffers become live allocations.

Phase-local activation/logits allocation:

- Added `LayerPhaseWorkspaceAllocation` and `LayerExecutorWorkspaceAllocationPlan` to `runtime/core/scheduler/layer_executor.h`.
- Added `build_layer_executor_workspace_allocation_plan(...)`, which converts phase-local row/column descriptors into aligned activation and logits slices.
- Added `find_layer_workspace_phase_index_for_stage(...)` so stage launches can resolve their owning phase workspace without benchmark-local phase math.
- Refactored `runtime_mixed_sequence_bench.exe` so each phase now launches against its own activation/logits slice instead of one shared anonymous activation/logits pair.
- Activation feedback now writes from the current phase logits slice into the next stage's phase activation slice, preserving the reference-checked staged handoff while making phase-local buffers real.
- Added telemetry:
  - `layer_executor_workspace_allocation_plan=phase_local_activation_logits`;
  - `layer_executor_workspace_bytes`;
  - `layer_executor_workspace_activation_bytes`;
  - `layer_executor_workspace_logits_bytes`;
  - per-phase `activation_offset`, `activation_bytes`, `logits_offset`, `logits_bytes`, and `total_bytes`.
- Expanded `runtime_layer_plan_selftest.exe` from `15` to `16` cases with `workspace_allocation_plan_passed`.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, token `4359/4359`, allocation plan `phase_local_activation_logits`, `74496` workspace bytes, activation/logits bytes `40960/33536`, and named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780035511.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780035511.json`, `cases_passed=16/16`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780035511.json`, `reference_passed=true`, token `7671/7671`, allocation plan `phase_local_activation_logits`, workspace bytes `74496`, named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780035511.json`, `reference_passed=true`, token `4359/4359`, allocation plan `phase_local_activation_logits`, workspace bytes `74496`, named warmup/capture phase callbacks `1/1/1/1`.
- Validation after this step: native build passed, direct layer-plan self-test passed, focused strict role-plan mixed run passed, full native harness passed, Python unit tests passed `68/68`, and benchmark summary generation succeeded.

Next bottlenecks:

- Move the phase-local runtime pointers out of the mixed benchmark and into a reusable layer-executor runtime state object.
- Start replacing feedback-only phase handoff with real attention or FFN dataflow while keeping the same allocation plan and reference gates.
- Add allocator high-water telemetry for phase workspace slices and graph capture workspace pressure.

Phase-local runtime binding:

- Added `LayerPhaseWorkspaceRuntime` and `LayerExecutorWorkspaceRuntime` to `runtime/core/scheduler/layer_executor.h`.
- Added `bind_layer_executor_workspace_runtime(...)`, which validates descriptor/allocation phase counts, allocation bounds, nonzero activation/logits slices, and a non-null workspace base before returning typed runtime pointers for each phase.
- Added `find_layer_workspace_runtime_phase_index_for_stage(...)` so benchmark stage launches resolve their runtime phase slices through the reusable executor binding instead of benchmark-local workspace math.
- Refactored `runtime_mixed_sequence_bench.exe` to bind the phase-local allocation plan once, then use `LayerExecutorWorkspaceRuntime::phases` for initial activation uploads, matvec launch pointers, logits handoff, and phase-local telemetry.
- Expanded `runtime_layer_plan_selftest.exe` from `16` to `17` cases with `workspace_runtime_binding_passed`, including pointer-offset checks for the attention and output phase activation/logits slices.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, `reference_stage_mismatches=0`, runtime-bound allocation plan `phase_local_activation_logits`, `74496` workspace bytes, activation/logits bytes `40960/33536`, steady-state H2D `0`, D2H `16`, and named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780036018.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780036018.json`, `cases_passed=17/17`, `workspace_runtime_binding_passed=true`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780036018.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `7671/7671`, runtime-bound allocation bytes `74496`, activation/logits bytes `40960/33536`, `reference_stage_mismatches=0`, and named warmup/capture phase callbacks `1/1/1/1`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780036018.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, runtime-bound allocation bytes `74496`, activation/logits bytes `40960/33536`, `reference_stage_mismatches=0`, and named warmup/capture phase callbacks `1/1/1/1`.
- Validation after this step: native build passed, direct layer-plan self-test passed `17/17`, focused strict role-plan mixed run passed, full native harness passed `63/63`, Python unit tests passed `68/68` outside the sandbox, and benchmark summary generation succeeded.

Next bottlenecks:

- Add allocator high-water telemetry for phase workspace slices and graph capture workspace pressure.
- Start replacing feedback-only phase handoff with real attention or FFN dataflow while keeping the same runtime-bound allocation plan and reference gates.
- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.

Workspace high-water and graph pressure telemetry:

- Added `LayerPhaseWorkspaceUsage` and `LayerExecutorWorkspaceUsage` to `runtime/core/scheduler/layer_executor.h`.
- Added `measure_layer_executor_workspace_usage(...)`, which validates runtime workspace pointers, derives activation/logits logical high-water bytes from the bound phase descriptors, verifies the high-water demand fits the allocation, and reports slack bytes.
- Refactored `runtime_mixed_sequence_bench.exe` to emit:
  - `layer_executor_workspace_usage_plan=phase_local_logical_high_water`;
  - total activation/logits/combined high-water bytes;
  - total activation/logits/combined slack bytes;
  - per-phase high-water and slack bytes in `layer_executor_phase_workspaces`.
- Added CUDA memory snapshots around graph capture, instantiate, upload, and destroy. The mixed runner now reports graph capture, instantiate, upload, live, and destroy-recovery memory-pressure deltas next to the existing capture/upload timings.
- Expanded `runtime_layer_plan_selftest.exe` from `17` to `18` cases with `workspace_usage_passed`.
- Focused layer-1 strict role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, `reference_stage_mismatches=0`, workspace high-water bytes `74496`, activation/logits high-water bytes `40960/33536`, workspace slack `0`, graph live device pressure `0`, steady-state H2D `0`, and D2H `16`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780036696.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780036696.json`, `cases_passed=18/18`, `workspace_runtime_binding_passed=true`, `workspace_usage_passed=true`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780036696.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `7671/7671`, workspace high-water bytes `74496`, activation/logits high-water bytes `40960/33536`, workspace slack `0`, graph live device pressure `0`, and `reference_stage_mismatches=0`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780036696.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, workspace high-water bytes `74496`, activation/logits high-water bytes `40960/33536`, workspace slack `0`, graph live device pressure `0`, and `reference_stage_mismatches=0`.
- Validation after this step: native build passed, direct layer-plan self-test passed `18/18`, focused strict role-plan mixed run passed, full native harness passed `63/63`, Python unit tests passed `68/68` outside the sandbox, and benchmark summary generation succeeded.

Next bottlenecks:

- Start replacing feedback-only phase handoff with real attention or FFN dataflow while keeping the runtime-bound allocation and high-water telemetry stable.
- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.

Strict admission memory breakdowns:

- Extended `runtime/core/scheduler/admission.h` with separated request-estimate fields for `kv_cache_bytes`, `dma_bytes`, `graph_bytes`, and `workspace_high_water_bytes`.
- Added `AdmissionBreakdown` telemetry for resident weights, predicted/current KV, workspace allocation, workspace high-water, workspace slack, DMA, graph, WDDM guard, usable bytes, required bytes, and over-budget bytes.
- Refactored `decide_admission(...)` so strict rejection reasons include required, usable, and over-budget bytes while preserving the no-silent-spill policy.
- Added a plan/allocation overload of `measure_layer_executor_workspace_usage(...)` so admission can report workspace high-water before runtime pointers are bound.
- Refactored `runtime_mixed_sequence_bench.exe` to feed separated resident weight, KV, workspace, DMA, and workspace high-water estimates into admission, emit `admission_*` JSON fields for both accepted and rejected runs, and assert the runtime-bound workspace usage still matches the admission-time estimate.
- Expanded `runtime_layer_plan_selftest.exe` from `18` to `20` cases with `admission_breakdown_passed` and `admission_over_budget_passed`.
- Focused accepted layer-1 strict role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, admission required/usable bytes `665577216/11259609088`, resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`, workspace high-water/slack bytes `74496/0`, and over-budget bytes `0`.
- Focused forced-rejection run with an oversized WDDM guard rejected before allocation and reported required/usable/over-budget bytes `665577216/0/665577216` with the same memory breakdown.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780037320.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780037320.json`, `cases_passed=20/20`, `admission_breakdown_passed=true`, `admission_over_budget_passed=true`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780037320.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `7671/7671`, admission required/usable bytes `665577216/11259609088`, resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`, workspace high-water/slack bytes `74496/0`, graph live device pressure `0`, and `reference_stage_mismatches=0`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780037320.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, admission required/usable bytes `665577216/11259609088`, resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`, workspace high-water/slack bytes `74496/0`, graph live device pressure `0`, and `reference_stage_mismatches=0`.
- Validation after this step: native build passed, direct layer-plan self-test passed `20/20`, focused accepted strict role-plan mixed run passed, forced strict rejection reported the expected over-budget breakdown, full native harness passed `63/63`, Python unit tests passed `68/68` outside the sandbox, and benchmark summary generation succeeded.

Next bottlenecks:

- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

Gated FFN SiLU handoff:

- Added `rtxllm_launch_rmsnorm_feedback_with_silu_cache(...)` and `rtxllm_launch_gated_silu_rmsnorm_feedback(...)` to the native decode kernel library.
- Added `--feedback-mode phase-aware-ffn-gated-silu` to `runtime_mixed_sequence_bench.exe`.
- The gated mode uses the gate/up/down role order: it RMS-normalizes the `ffn_gate_shexp -> ffn_up_exps` edge, writes normal activation for the next stage, caches the SiLU gate output in an extra FFN workspace slice, then RMS-normalizes `ffn_up_shexp` and multiplies it by the cached gate before `ffn_down_exps`.
- Admission now includes the extra gated-FFN workspace demand. The layer-1 gated row reports admission workspace bytes `82688`, core layer-executor workspace bytes `74496`, FFN gate-cache workspace bytes `8192`, and workspace high-water bytes `82688`.
- Added a full native harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_gated`.
- Focused gated layer-1 strict role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_gated_silu`, gate-cache/gated-combine edges `1/1`, token `8074/8074`, max logit/KV error `1.10567e-05/1.10567e-05`, `2842.12` sequence steps/sec, and p50/p95/p99 `0.342/0.373/0.373 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780040972.md`.
- Harness result: `65/65` rows passed.
- Checkpoint gated FFN SiLU layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_gated_1780040972.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, gate-cache/gated-combine edges `1/1`, token `8074/8074`, `1625.16` sequence steps/sec, and p50/p95/p99 `0.399/1.009/1.009 ms`.
- Checkpoint default RMSNorm layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780040972.json`, `reference_passed=true`, token `8133/8133`, and p50/p95/p99 `0.362/0.729/0.729 ms`.
- Checkpoint phase-aware FFN SiLU layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_phaseaware_1780040972.json`, `reference_passed=true`, token `8022/8022`, and p50/p95/p99 `0.519/0.711/0.711 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `22/22`, focused gated layer-1 role-plan run passed, full native harness passed `65/65`, benchmark summary generation succeeded, and Python unit tests passed `68/68`.

Next bottlenecks:

- Replace the handoff-level gated approximation with a real FFN phase primitive that keeps gate and up projection inputs independent before the elementwise product.
- Move the gated FFN cache shape into the core layer-executor workspace contract if the next primitive needs more than one auxiliary phase buffer.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

Phase-local activation-feedback dataflow:

- Added `LayerActivationFeedbackEdge` and `LayerExecutorDataflowPlan` to `runtime/core/scheduler/layer_executor.h`.
- Added `build_layer_executor_dataflow_plan(...)`, which derives source stage, target stage, source/target phase indices, source/target phases, source row count, target column count, phase-crossing markers, and wrap markers from the validated layer execution plan plus runtime-bound phase workspaces.
- Added `find_layer_activation_feedback_edge(...)` so callers no longer implement local next-stage routing policy.
- Refactored `runtime_mixed_sequence_bench.exe` so CUDA activation-feedback launches and the CPU reference path both use the core executor dataflow plan instead of benchmark-local `(index + 1) % stages.size()` logic.
- Added mixed-run telemetry:
  - `layer_executor_dataflow_plan=phase_local_activation_feedback_edges`;
  - `layer_executor_activation_feedback_edges`;
  - `layer_executor_phase_crossing_feedback_edges`;
  - `layer_executor_wrap_feedback_edges`;
  - `layer_executor_activation_feedback_plan`.
- Expanded `runtime_layer_plan_selftest.exe` from `20` to `22` cases with `dataflow_plan_passed` and `unchained_dataflow_passed`.
- Focused layer-1 strict role-plan dataflow run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, `reference_stage_mismatches=0`, feedback edges/phase-crossing/wrap edges `12/4/1`, admission required/usable bytes `665577216/11259609088`, and resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780038042.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780038042.json`, `cases_passed=22/22`, `dataflow_plan_passed=true`, `unchained_dataflow_passed=true`, `admission_breakdown_passed=true`, and `admission_over_budget_passed=true`.
- Checkpoint layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780038042.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `7671/7671`, feedback edges/phase-crossing/wrap edges `12/4/1`, `reference_stage_mismatches=0`, `4271.22` sequence steps/sec, and p50/p95/p99 `0.231/0.239/0.239 ms`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780038042.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `4359/4359`, feedback edges/phase-crossing/wrap edges `12/4/1`, `reference_stage_mismatches=0`, `4367.77` sequence steps/sec, and p50/p95/p99 `0.232/0.235/0.235 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `22/22`, focused layer-1 strict role-plan dataflow run passed, full native harness passed `63/63`, benchmark summary generation succeeded, and Python unit tests passed `68/68` outside the sandbox.

Next bottlenecks:

- Replace the synthetic activation-feedback transform with the first real phase math primitive while keeping the same dataflow edge plan and reference gates.
- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

RMSNorm phase handoff:

- Added `rtxllm_launch_rmsnorm_feedback(...)` to the native decode kernel library. The CUDA kernel normalizes each source-logit row group with an RMSNorm-style scale and writes the result into the next phase activation slice chosen by the shared dataflow edge plan.
- Kept the prior synthetic activation-feedback kernel as an explicit comparison mode and changed `runtime_mixed_sequence_bench.exe` to default to `--feedback-mode rmsnorm`.
- Added `FeedbackMode`, `--feedback-mode synthetic|rmsnorm`, CPU reference selection, and JSON `feedback_mode` telemetry to the mixed runner. The graph launch path and CPU reference path now use the same `LayerActivationFeedbackEdge` plus selected feedback mode.
- Focused layer-1 RMSNorm run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `8068/8068`, `reference_stage_mismatches=0`, max logit/KV error `5.32717e-06/5.32717e-06`, feedback edges/phase-crossing/wrap edges `12/4/1`, admission required/usable bytes `665577216/11259609088`, resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`, `2413.27` sequence steps/sec over the focused four-step run, and p50/p95/p99 `0.375/0.523/0.523 ms`.
- Focused synthetic comparison still passed with `reference_passed=true`, token `4359/4359`, `reference_stage_mismatches=0`, feedback edges/phase-crossing/wrap edges `12/4/1`, and p50/p95/p99 `0.270/0.276/0.276 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780038688.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780038688.json`, `cases_passed=22/22`, `dataflow_plan_passed=true`, `unchained_dataflow_passed=true`, `admission_breakdown_passed=true`, and `admission_over_budget_passed=true`.
- Checkpoint layer-0 RMSNorm strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780038688.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `5201/5201`, `reference_stage_mismatches=0`, max logit/KV error `3.57628e-06/3.57628e-06`, feedback edges/phase-crossing/wrap edges `12/4/1`, `2432.35` sequence steps/sec, and p50/p95/p99 `0.363/0.544/0.544 ms`.
- Checkpoint layer-1 RMSNorm strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780038688.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `8068/8068`, `reference_stage_mismatches=0`, max logit/KV error `5.32717e-06/5.32717e-06`, feedback edges/phase-crossing/wrap edges `12/4/1`, `2809.58` sequence steps/sec, and p50/p95/p99 `0.356/0.364/0.364 ms`.
- Checkpoint Q5_K dequant artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780038688.json`, `reference_passed=true`, `12857.60` graph replays/sec, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, and p50/p95/p99 `0.068/0.134/0.134 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `22/22`, focused RMSNorm strict role-plan mixed run passed, focused synthetic comparison run passed, full native harness passed `63/63`, benchmark summary generation succeeded, and Python unit tests passed `68/68` outside the sandbox.

Next bottlenecks:

- Replace the RMSNorm-style phase handoff with the first real attention or FFN phase primitive while preserving the same dataflow edge plan and reference gates.
- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

Phase-aware FFN SiLU handoff:

- Added `rtxllm_launch_silu_rmsnorm_feedback(...)` to the native decode kernel library. The CUDA kernel reuses the RMSNorm-style scale, then applies `SiLU(x) = x * sigmoid(x)` before writing into the next activation slice.
- Added `--feedback-mode phase-aware-ffn-silu` to `runtime_mixed_sequence_bench.exe`. The mode keeps RMSNorm feedback for attention, SSM, output, and phase-crossing edges, and uses SiLU-normalized feedback only when both the source and target stages are inside the FFN phase.
- Added CPU reference coverage and JSON telemetry for `phase_aware_ffn_silu_feedback_edges`.
- Focused layer-1 phase-aware strict role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_silu`, `phase_aware_ffn_silu_feedback_edges=5`, token `8047/8047`, `reference_stage_mismatches=0`, max logit/KV error `1.59144e-05/1.59144e-05`, feedback edges/phase-crossing/wrap edges `12/4/1`, admission required/usable bytes `665577216/11259609088`, resident/KV/workspace/DMA bytes `656065536/8388608/74496/1048576`, `2801.71` sequence steps/sec, and p50/p95/p99 `0.351/0.368/0.368 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780039500.md`.
- Harness result: `63/63` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780039500.json`, `cases_passed=22/22`.
- Checkpoint default RMSNorm layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780039500.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `phase_aware_ffn_silu_feedback_edges=0`, token `5201/5201`, `2509.10` sequence steps/sec, and p50/p95/p99 `0.343/0.566/0.566 ms`.
- Checkpoint default RMSNorm layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780039500.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `phase_aware_ffn_silu_feedback_edges=0`, token `8068/8068`, `2834.67` sequence steps/sec, and p50/p95/p99 `0.351/0.359/0.359 ms`.
- Checkpoint Q5_K dequant artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780039500.json`, `reference_passed=true`, `17524.60` graph replays/sec, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, and p50/p95/p99 `0.054/0.076/0.076 ms`.
- Validation after this step: native build passed, focused phase-aware FFN SiLU strict role-plan mixed run passed, full native harness passed `63/63`, and benchmark summary generation succeeded.

Next bottlenecks:

- Replace the FFN handoff approximation with a real gated FFN phase primitive that combines gate/up/down projections explicitly instead of chaining one projection at a time.
- Generalize the runtime workspace binding away from `float*` if the next phase introduces FP16/BF16 activation storage.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

Gate/up/down FFN role order:

- Changed `runtime/core/scheduler/layer_plan.cpp` so role-plan FFN ranks are semantic gate/up/down instead of the previous down-first order.
- Updated `runtime/apps/layer_plan_selftest/main.cpp` fixtures and phase workspace expectations to match the new FFN role order.
- Added the layer-1 `--feedback-mode phase-aware-ffn-silu` strict role-plan row to `tools/bench/native_harness.py`, so the opt-in FFN SiLU handoff is now tracked by the full native harness instead of only focused manual runs.
- Focused layer-1 RMSNorm role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `8133/8133`, max logit/KV error `7.09295e-06/7.09295e-06`, and p50/p95/p99 `0.3538/0.3823/0.3823 ms`.
- Focused layer-1 phase-aware role-plan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_silu`, `phase_aware_ffn_silu_feedback_edges=5`, token `8022/8022`, max logit/KV error `1.58548e-05/1.58548e-05`, and p50/p95/p99 `0.3554/0.6694/0.6694 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780040122.md`.
- Harness result: `64/64` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780040122.json`, `cases_passed=22/22`.
- Checkpoint default RMSNorm layer-0 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780040122.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `5199/5199`, `2801.71` sequence steps/sec, and p50/p95/p99 `0.354/0.364/0.364 ms`.
- Checkpoint default RMSNorm layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780040122.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token `8133/8133`, `2810.96` sequence steps/sec, and p50/p95/p99 `0.347/0.377/0.377 ms`.
- Checkpoint phase-aware FFN SiLU layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_phaseaware_1780040122.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `phase_aware_ffn_silu_feedback_edges=5`, token `8022/8022`, `2875.84` sequence steps/sec, and p50/p95/p99 `0.339/0.367/0.367 ms`.
- Checkpoint Q5_K dequant artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780040122.json`, `reference_passed=true`, `18306.60` graph replays/sec, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, and p50/p95/p99 `0.050/0.075/0.075 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `22/22`, focused layer-1 RMSNorm and phase-aware role-plan runs passed, full native harness passed `64/64`, benchmark summary generation succeeded, and Python unit tests passed `68/68` outside the sandbox.

Next bottlenecks:

- Replace the FFN handoff approximation with a real gated FFN phase primitive that consumes gate and up projections together before down projection.
- Keep role-plan, workspace, strict-admission, and CPU-reference gates stable while moving from feedback approximations to actual phase math.
- Add graph-bucket admission rows once decode/prefill bucket capture starts reporting nonzero graph workspace pressure.

Gated FFN phase primitive:

- Added `--ffn-phase-mode gated-silu` to `runtime_mixed_sequence_bench.exe` as a separate opt-in phase mode from `--feedback-mode`.
- Added CUDA cache/product kernels for phase-local FFN execution: cache normalized `SiLU(gate)` branch output, cache normalized `up` branch output, and write the gated product into the down-projection activation.
- The gated phase mode requires a validated role-plan order and chained activation. It keeps gate/up projections on the same FFN input activation, then writes the product only before the down stage.
- Added an FFN up-cache workspace slice beside the existing gate cache. The layer-1 gated phase row reports admission workspace bytes `90880`, core layer-executor workspace bytes `74496`, FFN gate/up cache workspace bytes `8192/8192`, and workspace high-water bytes `90880`.
- Added a full native harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_ffnphase`.
- Focused layer-1 gated FFN phase run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, gate/up/product phase edges `2/2/1`, token `8119/8119`, max logit/KV error `7.83987e-06/7.83987e-06`, `2813.53` sequence steps/sec, and p50/p95/p99 `0.354/0.358/0.358 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780042500.md`.
- Harness result: `66/66` rows passed.
- Checkpoint gated FFN phase layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ffnphase_1780042500.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8119/8119`, `2809.19` sequence steps/sec, and p50/p95/p99 `0.354/0.362/0.362 ms`.
- Validation after this step: native build passed, focused gated FFN phase layer-1 role-plan run passed, full native harness passed `66/66`, benchmark summary generation succeeded, and Python unit tests passed `68/68`.

Core auxiliary workspace promotion:

- Added `LayerAuxWorkspaceDescriptor`, `LayerAuxWorkspaceAllocation`, `LayerAuxWorkspaceRuntime`, and `LayerAuxWorkspaceUsage` to `runtime/core/scheduler/layer_executor.h`.
- Added core helper `build_gated_ffn_aux_workspace_descriptors(...)` and runtime lookup for named auxiliary slices.
- The mixed runner now asks the core layer executor to allocate and bind `ffn_gate_cache` and `ffn_up_cache` instead of computing benchmark-local offsets after the phase workspace plan.
- Telemetry now reports `layer_executor_workspace_phase_bytes`, `layer_executor_aux_workspace_count`, `layer_executor_aux_workspace_bytes`, `layer_executor_workspace_auxiliary_high_water_bytes`, `layer_executor_workspace_auxiliary_slack_bytes`, and the `layer_executor_aux_workspaces` array.
- Expanded `runtime_layer_plan_selftest.exe` from `22` to `23` cases with `aux_workspace_allocation_passed=true`, proving the gate/up cache offsets, sizes, runtime bindings, and high-water usage.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780043198.md`.
- Harness result: `66/66` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780043198.json`, `cases_passed=23/23`, `aux_workspace_allocation_passed=true`.
- Checkpoint gated FFN phase layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ffnphase_1780043198.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8119/8119`, layer-executor phase/aux workspace bytes `74496/16384`, admission workspace bytes `90880`, `2319.11` sequence steps/sec, and p50/p95/p99 `0.354/0.661/0.661 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `23/23`, full native harness passed `66/66`, benchmark summary generation succeeded, and Python unit tests passed `68/68`.

Generic phase scratch workspace contract:

- Added `build_layer_phase_scratch_workspace_descriptors(...)` to `runtime/core/scheduler/layer_executor.h` so attention, FFN, SSM, and output kernels can request named phase-local scratch slices without benchmark-local offset math.
- Added `--phase-scratch NAME:PHASE:VALUES` to `runtime_mixed_sequence_bench.exe`; the harness now carries a layer-1 strict role-plan row with `attention_qkv_scratch:attention:8192` and `ssm_state_scratch:ssm:4096`.
- Expanded `runtime_layer_plan_selftest.exe` from `23` to `24` cases with `phase_scratch_workspace_passed=true`, proving generic scratch descriptor validation, allocation order, runtime binding, and high-water accounting.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780043986.md`.
- Harness result: `67/67` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780043986.json`, `cases_passed=24/24`, `aux_workspace_allocation_passed=true`, `phase_scratch_workspace_passed=true`.
- Checkpoint phase-scratch layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_scratch_1780043986.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, layer-executor phase/aux workspace bytes `74496/49152`, `phase_scratch_workspace_bytes=49152`, admission workspace bytes `123648`, `2780.67` sequence steps/sec, and p50/p95/p99 `0.353/0.375/0.375 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `24/24`, focused phase-scratch row passed, full native harness passed `67/67`, benchmark summary generation succeeded, and Python unit tests passed `68/68` after running with escalated temp-file permissions because the sandbox blocked Python-created temporary children.

Captured phase scratch digest kernel:

- Added `rtxllm_launch_phase_scratch_digest(...)` to the decode CUDA kernel library so phase-local scratch slices are read and written by a deterministic GPU kernel.
- The mixed runner now binds generic phase scratch runtimes from the core executor and launches the digest kernel after matching phase callbacks during warmup, CUDA Graph capture/replay, and post-graph phase timing.
- Telemetry now reports `phase_scratch_digest_enabled`, `phase_scratch_kernel_launches_per_graph`, `actual_kernel_launches_per_graph`, `layer_execution_actual_kernel_launches_per_graph`, and per-phase scratch/actual kernel counts.
- Checkpoint harness artifact set: `1780045144`.
- Harness result: `67/67` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780045144.json`, `cases_passed=24/24`, `aux_workspace_allocation_passed=true`, `phase_scratch_workspace_passed=true`.
- Checkpoint captured phase-scratch layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_scratch_1780045144.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `phase_scratch_digest_enabled=true`, `phase_scratch_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, layer-executor phase/aux workspace bytes `74496/49152`, `phase_scratch_workspace_bytes=49152`, admission workspace bytes `123648`, `2271.31` sequence steps/sec, and p50/p95/p99 `0.360/0.670/0.670 ms`.
- Per-phase scratch digest launch counts are attention `1`, FFN `0`, SSM `1`, output `0`, giving actual per-phase kernel counts `5/12/7/2`.
- Validation after this step: native build passed, direct layer-plan self-test passed `24/24`, focused captured phase-scratch row passed, full native harness passed `67/67`, benchmark summary generation succeeded, and Python unit tests passed `68/68` with escalated temp-file permissions.

Next bottlenecks:

- Replace the remaining down/SSM/output handoff primitives with phase math that preserves independent branch inputs where the real model needs them.
- Replace the scratch digest placeholder with real attention/SSM phase math while keeping the captured phase-local scratch contract.
- Replace graph-bucket admission estimates with measured graph pool pressure once decode/prefill bucket capture owns real graph workspaces.

Core CUDA graph-bucket admission estimate:

- Added `CudaGraphBucketDescriptor`, `CudaGraphBucketAllocation`, and `CudaGraphBucketPlan` to `runtime/core/graphs/cuda_graph_bucket.h`.
- Added `estimate_cuda_graph_bucket_allocation(...)` and `build_cuda_graph_bucket_plan(...)`, with aligned metadata bytes, workspace guard bytes, total bytes, plan offsets, and max-bucket bytes.
- Updated `CudaGraphBucket::instantiate` to the CUDA 13-compatible `cudaGraphInstantiate(&exec_, graph_, 0)` call shape.
- Expanded `runtime_layer_plan_selftest.exe` from `24` to `25` cases with `graph_bucket_plan_passed=true`.
- The mixed runner now builds a core graph-bucket plan from actual per-graph kernel count and layer workspace high-water before strict admission, charges the result as `admission_graph_bytes`, and emits `graph_bucket_count`, `graph_bucket_estimated_bytes`, `graph_bucket_max_estimated_bytes`, and a per-bucket `graph_buckets` array.
- Added `--graph-bucket-count`, `--graph-batch-bucket`, and `--graph-query-bucket` to `runtime_mixed_sequence_bench.exe`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_graphbucket`, which models three decode buckets (`q1/q2/q3`) for the layer-1 strict role-plan reference path.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780051778.md`.
- Harness result: `68/68` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780051778.json`, `cases_passed=25/25`, `graph_bucket_plan_passed=true`, `phase_scratch_workspace_passed=true`, and `admission_breakdown_passed=true`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780051778.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `7.09295e-06/7.09295e-06`, `admission_graph_bytes=173824`, `2781.45` sequence steps/sec, and p50/p95/p99 `0.358/0.364/0.364 ms`.
- Checkpoint captured phase-scratch artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_scratch_1780051778.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `phase_scratch_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, admission workspace/graph bytes `123648/182784`, `2614.72` sequence steps/sec, and p50/p95/p99 `0.371/0.410/0.410 ms`.
- Checkpoint graph-bucket artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_graphbucket_1780051778.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `graph_bucket_count=3`, `graph_bucket_estimated_bytes=523008`, `graph_bucket_max_estimated_bytes=174848`, `admission_graph_bytes=523008`, `actual_kernel_launches_per_graph=24`, `1231.07` sequence steps/sec, and p50/p95/p99 `0.372/1.909/1.909 ms`.
- Checkpoint Q5_K dequant artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780051778.json`, `reference_passed=true`, token `480/480`, `16420.40` graph replays/sec, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, and p50/p95/p99 `0.059/0.070/0.070 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `25/25`, focused graph-bucket mixed run passed, full native harness passed `68/68`, benchmark summary generation succeeded, and Python unit tests passed `68/68` with escalated temp-file permissions.

Next bottlenecks:

- Replace the remaining down/SSM/output handoff primitives with phase math that preserves independent branch inputs where the real model needs them.
- Replace the scratch digest placeholder with real attention/SSM phase math while keeping the captured phase-local scratch contract.
- Tie graph-bucket estimates to measured graph-pool pressure once real decode/prefill bucket capture owns persistent workspaces.

SSM recurrent state phase primitive:

- Added `build_ssm_recurrent_aux_workspace_descriptors(...)` to `runtime/core/scheduler/layer_executor.h`, producing a core-owned `ssm_recurrent_state` auxiliary slice tied to the SSM phase and sized from the output phase envelope.
- Added `rtxllm_launch_ssm_recurrent_state(...)` to the decode CUDA kernel library. The kernel updates persistent SSM state from previous state, SSM logits, and output activation, then feeds the activation consumed by the output stage.
- Added `--ssm-phase-mode passthrough|recurrent-state` to `runtime_mixed_sequence_bench.exe`. Recurrent-state mode requires chained activation, strict role-plan stage order, and at least one SSM role; it adds one captured SSM kernel per graph replay and reports `ssm_phase_kernel_launches_per_graph`, `ssm_recurrent_state_workspace_bytes`, and `ssm_recurrent_state_values`.
- Expanded `runtime_layer_plan_selftest.exe` from `25` to `26` cases with `ssm_recurrent_workspace_passed=true`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmphase`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780057719.md`.
- Harness result: `69/69` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780057719.json`, `cases_passed=26/26`, `ssm_recurrent_workspace_passed=true`, `graph_bucket_plan_passed=true`, `phase_scratch_workspace_passed=true`, and `admission_breakdown_passed=true`.
- Checkpoint SSM phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmphase_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=recurrent_state`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `ssm_recurrent_state_workspace_bytes=8192`, admission workspace/graph bytes `82688/178176`, `2571.52` sequence steps/sec, and p50/p95/p99 `0.367/0.449/0.449 ms`.
- Checkpoint layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `7.09295e-06/7.09295e-06`, `admission_graph_bytes=173824`, `2763.96` sequence steps/sec, and p50/p95/p99 `0.353/0.382/0.382 ms`.
- Checkpoint graph-bucket artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_graphbucket_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `graph_bucket_count=3`, `graph_bucket_estimated_bytes=523008`, `graph_bucket_max_estimated_bytes=174848`, `admission_graph_bytes=523008`, `actual_kernel_launches_per_graph=24`, `2714.44` sequence steps/sec, and p50/p95/p99 `0.364/0.377/0.377 ms`.
- Checkpoint Q5_K dequant artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780057719.json`, `reference_passed=true`, token `480/480`, `16785.60` graph replays/sec, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, and p50/p95/p99 `0.055/0.078/0.078 ms`.
- Validation after this step: native build passed, direct layer-plan self-test passed `26/26`, focused SSM phase row passed, full native harness passed `69/69`, benchmark summary generation succeeded, and Python unit tests passed `68/68` with escalated temp-file permissions.

Next bottlenecks:

- Replace the remaining attention/output approximations with real phase math while preserving the strict workspace and graph-bucket contracts.
- Replace the scratch digest placeholder with real attention/SSM scratch consumers under the same captured phase-local scratch contract.
- Tie graph-bucket estimates to measured graph-pool pressure once real decode/prefill bucket capture owns persistent workspaces.

Output final-token phase primitive:

- Added `rtxllm_launch_output_token_sample(...)` to the decode CUDA kernel library. The kernel reduces output logits, selects the final token on GPU, and writes only the token-sized result for CPU detokenization.
- Added `--output-phase-mode passthrough|final-token` to `runtime_mixed_sequence_bench.exe`. Final-token mode requires chained activation plus a validated role-plan order, runs after the output phase callback during warmup, graph capture/replay, reference execution, and phase timing, and reports `output_phase_kernel_launches_per_graph`, `output_phase_final_token_enabled`, `output_phase_vocab_size`, and per-phase output kernel counts.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_outputphase`.
- Focused output phase run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `output_phase_mode=final_token`, token `63/63`, `output_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, max logit/KV error `7.09295e-06/7.09295e-06`, `2736.54` sequence steps/sec, and p50/p95/p99 `0.361/0.370/0.370 ms`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780063102.md`.
- Harness result: `70/70` rows passed.
- Checkpoint output phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_outputphase_1780063102.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `output_phase_mode=final_token`, token `63/63`, `output_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `admission_graph_bytes=177920`, `2815.91` sequence steps/sec, and p50/p95/p99 `0.354/0.356/0.356 ms`.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780063102.json`, `cases_passed=26/26`, `ssm_recurrent_workspace_passed=true`, `graph_bucket_plan_passed=true`, and `phase_scratch_workspace_passed=true`.
- Validation after this step: native build passed, direct layer-plan self-test passed `26/26`, focused output phase row passed, full native harness passed `70/70`, benchmark summary was regenerated and verified with the new `output kernels` column, and Python unit tests passed `68/68` with escalated temp-file permissions.

Next bottlenecks:

- Replace the remaining attention/SSM approximations with real phase math while preserving the strict workspace and graph-bucket contracts.
- Replace the scratch digest placeholder with real attention/SSM scratch consumers under the same captured phase-local scratch contract.
- Tie graph-bucket estimates to measured graph-pool pressure once real decode/prefill bucket capture owns persistent workspaces.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780063102.md`.
- Harness result: `70/70` rows passed.
- Output final-token layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_outputphase_1780063102.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `63/63`, `output_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `admission_graph_bytes=177920`, `2815.91` sequence steps/sec, and p50/p95/p99 `0.354/0.356/0.356 ms`.
- Validation after that checkpoint: native build passed, direct layer-plan self-test passed `26/26`, focused output phase row passed, full native harness passed `70/70`, benchmark summary regenerated and verified, and Python unit tests passed `68/68`.

Attention QKV scratch phase primitive:

- Added `rtxllm_launch_attention_qkv_scratch(...)` to the decode CUDA kernel library. The kernel writes deterministic Q/K/V-style scratch values into a core-owned attention auxiliary slice and applies a small attention residual to the FFN activation through the existing attention-to-FFN dataflow edge.
- Added `--attention-phase-mode passthrough|qkv-scratch` to `runtime_mixed_sequence_bench.exe`. QKV scratch mode requires chained activation plus a validated role-plan order, reserves `attention_qkv_state`, adds one captured attention kernel per graph replay, mirrors the update in the CPU reference path, and reports `attention_phase_kernel_launches_per_graph`, `attention_qkv_scratch_workspace_bytes`, `attention_qkv_scratch_values`, and per-phase attention kernel counts.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionphase`.
- Focused attention phase run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_scratch`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, max logit/KV error `4.64916e-06/4.64916e-06`, `admission_graph_bytes=179456`, and no reference mismatches.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780064700.md`.
- Harness result: `71/71` rows passed.
- Checkpoint attention phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionphase_1780064700.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_scratch`, token `8133/8133`, max logit/KV error `4.64916e-06/4.64916e-06`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=98304`, admission workspace/graph bytes `172800/179456`, `2789.40` sequence steps/sec, and p50/p95/p99 `0.357/0.362/0.362 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md` with new `attention kernels` and `attention qkv B` columns and verified at `60/60` columns for the header, separator, and new attention row.
- Validation after this step: native build passed, direct layer-plan self-test passed `26/26`, focused attention phase row passed, full native harness passed `71/71`, benchmark summary regenerated and verified, and Python unit tests passed `68/68` with escalated temp-file permissions.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780064700.md`.
- Harness result: `71/71` rows passed.
- Attention QKV scratch layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionphase_1780064700.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=179456`, `2789.40` sequence steps/sec, and p50/p95/p99 `0.357/0.362/0.362 ms`.
- Validation after that checkpoint: native build passed, direct layer-plan self-test passed `26/26`, focused attention phase row passed, full native harness passed `71/71`, benchmark summary regenerated and verified, and Python unit tests passed `68/68`.

SSM scan scratch phase primitive:

- Added `ssm_scan_scratch` support to `build_ssm_recurrent_aux_workspace_descriptors(...)` in `runtime/core/scheduler/layer_executor.h`. Scan-scratch mode now reserves the existing `ssm_recurrent_state` slice plus a core-owned `ssm_scan_scratch` auxiliary slice tied to the SSM phase envelope.
- Added `rtxllm_launch_ssm_scan_scratch(...)` to the decode CUDA kernel library. The kernel consumes SSM logits, recurrent state, scan scratch, and output activation, then updates scan scratch, recurrent state, and activation inside captured graph replay.
- Extended `runtime_mixed_sequence_bench.exe` with `--ssm-phase-mode scan-scratch`. The mode requires chained activation, strict role-plan stage order, an SSM-to-output feedback edge, and SSM roles; it adds one captured SSM phase kernel per graph replay and reports `ssm_phase_scan_scratch_enabled`, `ssm_scan_scratch_workspace_bytes`, `ssm_scan_scratch_values`, and `ssm_scan_scratch_scale`.
- Expanded `runtime_layer_plan_selftest.exe` from `26` to `27` cases with `ssm_scan_workspace_passed=true`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan`.
- Focused SSM scan run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=scan_scratch`, token `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `ssm_recurrent_state_workspace_bytes=8192`, `ssm_scan_scratch_workspace_bytes=16384`, `admission_workspace_bytes=99072`, `admission_graph_bytes=178432`, and no reference mismatches.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780066554.md`.
- Harness result: `72/72` rows passed.
- Checkpoint layer-plan self-test artifact: `E:\AI project\benchmarks\native_layer_plan_selftest_runtime_layer_plan_selftest_1780066554.json`, `cases_passed=27/27`, `ssm_scan_workspace_passed=true`, `ssm_recurrent_workspace_passed=true`, `graph_bucket_plan_passed=true`, `phase_scratch_workspace_passed=true`, and `admission_breakdown_passed=true`.
- Checkpoint SSM scan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan_1780066554.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=scan_scratch`, token `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, recurrent/scan workspace bytes `8192/16384`, admission workspace/graph bytes `99072/178432`, `2861.03` sequence steps/sec, and p50/p95/p99 `0.350/0.356/0.356 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md` with the new `ssm scan B` column and verified at `61/61` columns for the header, separator, and new SSM scan row.
- Validation after this step: native build passed, direct layer-plan self-test passed `27/27`, focused SSM scan row passed, full native harness passed `72/72`, benchmark summary regenerated and verified, and Python unit tests passed `68/68` with escalated temp-file permissions.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780066554.md`.
- Harness result: `72/72` rows passed.
- SSM scan-scratch layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan_1780066554.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `ssm_scan_scratch_workspace_bytes=16384`, `admission_graph_bytes=178432`, `2861.03` sequence steps/sec, and p50/p95/p99 `0.350/0.356/0.356 ms`.
- Validation after that checkpoint: native build passed, direct layer-plan self-test passed `27/27`, focused SSM scan row passed, full native harness passed `72/72`, benchmark summary regenerated and verified, and Python unit tests passed `68/68`.

Integrated all-phase strict role-plan row:

- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase`.
- The row combines `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-scratch`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token` in one strict role-plan graph.
- Focused all-phase run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, max logit/KV error `5.62146e-06/5.62146e-06`, phase kernels attention/SSM/output `1/1/1`, `actual_kernel_launches_per_graph=27`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/188416`, and no reference mismatches.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780067039.md`.
- Harness result: `73/73` rows passed.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780067039.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_scratch`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `5.62146e-06/5.62146e-06`, phase kernels attention/SSM/output `1/1/1`, `actual_kernel_launches_per_graph=27`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/188416`, `2327.21` sequence steps/sec, and p50/p95/p99 `0.381/0.572/0.572 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the all-phase row is present and the header, separator, and row all verify at `61/61` columns.
- Validation after this step: Python script compilation passed, Python unit tests passed `68/68` with temp files under `E:\AI project`, full native harness passed `73/73`, and benchmark summary regenerated and verified.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780067039.md`.
- Harness result: `73/73` rows passed.
- Integrated all-phase strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780067039.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=27`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=188416`, `2327.21` sequence steps/sec, and p50/p95/p99 `0.381/0.572/0.572 ms`.
- Validation after that checkpoint: Python script compilation passed, Python unit tests passed `68/68`, full native harness passed `73/73`, and benchmark summary regenerated and verified.

Attention QKV reduce phase primitive:

- Added `rtxllm_launch_attention_qkv_reduce(...)` to the decode CUDA kernel library. The kernel consumes the Q/K/V scratch rows written by `rtxllm_launch_attention_qkv_scratch`, computes a bounded score-derived residual from scratch Q/K/V values, and applies the weighted residual to the FFN activation before FFN stages consume it.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce`. Reduce mode keeps the existing scratch writer, adds the reduce kernel as a second captured attention phase kernel, mirrors the same math in the CPU reference path, and reports `attention_phase_qkv_reduce_enabled` plus `attention_qkv_reduce_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionreduce`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-reduce`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Focused attention reduce run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `8.58307e-06/8.58307e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, admission workspace/graph bytes `172800/183552`, and no reference mismatches.
- Focused updated all-phase run passed with `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, max logit/KV error `7.16746e-06/7.16746e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, and no reference mismatches.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780067869.md`.
- Harness result: `74/74` rows passed.
- Checkpoint attention reduce artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionreduce_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_reduce`, `attention_phase_qkv_reduce_enabled=true`, token `8133/8133`, max logit/KV error `8.58307e-06/8.58307e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, admission workspace/graph bytes `172800/183552`, `2684.38` sequence steps/sec, and p50/p95/p99 `0.371/0.385/0.385 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_reduce`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `7.16746e-06/7.16746e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2373.89` sequence steps/sec, and p50/p95/p99 `0.419/0.425/0.425 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention reduce and updated all-phase rows are present and the header, separator, and both rows all verify at `61/61` columns.
- Validation after this step: native build passed, focused attention reduce row passed, focused updated all-phase row passed, Python script compilation passed, Python unit tests passed `68/68` with temp files under `E:\AI project`, full native harness passed `74/74`, and benchmark summary regenerated and verified.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780067869.md`.
- Harness result: `74/74` rows passed.
- Attention QKV reduce layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionreduce_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2684.38` sequence steps/sec, and p50/p95/p99 `0.371/0.385/0.385 ms`.
- Integrated all-phase strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2373.89` sequence steps/sec, and p50/p95/p99 `0.419/0.425/0.425 ms`.
- Validation after that checkpoint: native build passed, focused attention reduce and all-phase rows passed, Python script compilation passed, Python unit tests passed `68/68`, full native harness passed `74/74`, and benchmark summary regenerated and verified.

Attention QKV window phase primitive:

- Added `rtxllm_launch_attention_qkv_window(...)` to the decode CUDA kernel library. The kernel consumes the Q/K/V scratch rows written by `rtxllm_launch_attention_qkv_scratch`, computes a small stable softmax window over Q/K scores, aggregates V scratch rows into a context residual, and applies that residual to the FFN activation before FFN stages consume it.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce|qkv-window`. Window mode keeps the existing scratch writer, adds the window kernel as a second captured attention phase kernel, mirrors the same softmax-window math in the CPU reference path, and reports `attention_phase_qkv_window_enabled`, `attention_qkv_window_size`, `attention_qkv_window_softmax_scale`, and `attention_qkv_window_residual_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionwindow`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-window`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780068872.md`.
- Harness result: `75/75` rows passed.
- Checkpoint attention window artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionwindow_1780068872.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_window`, `attention_phase_qkv_window_enabled=true`, token `8133/8133`, max logit/KV error `1.01924e-05/1.01924e-05`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, admission workspace/graph bytes `172800/183552`, `2594.37` sequence steps/sec, and p50/p95/p99 `0.373/0.400/0.400 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780068872.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_window`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.29832e-06/9.29832e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2538.72` sequence steps/sec, and p50/p95/p99 `0.389/0.413/0.413 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention window and updated all-phase rows are present and the header, separator, and both rows all verify at `62/62` pipe counts.
- Validation after this step: native build passed, Python script compilation passed, full native harness passed `75/75`, benchmark summary regenerated and verified, and the layer-plan self-test still passed `27/27`.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780068872.md`.
- Harness result: `75/75` rows passed.
- Attention QKV window layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionwindow_1780068872.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2594.37` sequence steps/sec, and p50/p95/p99 `0.373/0.400/0.400 ms`.
- Integrated all-phase strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780068872.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2538.72` sequence steps/sec, and p50/p95/p99 `0.389/0.413/0.413 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `75/75`, benchmark summary regenerated and verified, and the layer-plan self-test passed `27/27`.

Attention QKV head-window phase primitive:

- Added `rtxllm_launch_attention_qkv_head_window(...)` to the decode CUDA kernel library. The kernel consumes the Q/K/V scratch rows written by `rtxllm_launch_attention_qkv_scratch`, interprets them as contiguous per-head blocks, computes a stable local softmax window inside the selected head, aggregates V rows without crossing head blocks, and applies that residual before FFN stages consume the activation.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce|qkv-window|qkv-head-window`. Head-window mode keeps the existing scratch writer, adds the head-local window kernel as the second captured attention phase kernel, mirrors the same head/block math in the CPU reference path, and reports `attention_phase_qkv_head_window_enabled`, `attention_qkv_head_window_head_count`, `attention_qkv_head_window_size`, `attention_qkv_head_window_softmax_scale`, and `attention_qkv_head_window_residual_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadwindow`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-head-window`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780069653.md`.
- Harness result: `76/76` rows passed.
- Checkpoint attention head-window artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_window`, `attention_phase_qkv_head_window_enabled=true`, token `8133/8133`, max logit/KV error `8.53837e-06/8.53837e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/window `8/4`, admission workspace/graph bytes `172800/183552`, `2691.61` sequence steps/sec, and p50/p95/p99 `0.373/0.382/0.382 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_window`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `5.36442e-06/5.36442e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2719.42` sequence steps/sec, and p50/p95/p99 `0.363/0.378/0.378 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention head-window and updated all-phase rows are present in the summary.
- Validation after this step: native build passed, Python script compilation passed, full native harness passed `76/76`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780069653.md`.
- Harness result: `76/76` rows passed.
- Attention QKV head-window layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2691.61` sequence steps/sec, and p50/p95/p99 `0.373/0.382/0.382 ms`.
- Integrated all-phase strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2719.42` sequence steps/sec, and p50/p95/p99 `0.363/0.378/0.378 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `76/76`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Attention QKV head-dim-window phase primitive:

- Added `rtxllm_launch_attention_qkv_head_dim_window(...)` to the decode CUDA kernel library. The kernel consumes the Q/K/V scratch rows written by `rtxllm_launch_attention_qkv_scratch`, groups them as head/context/head-dimension blocks, scores each candidate key with an 8-lane Q/K dot product, computes a stable softmax over a local window, aggregates the matching V lane, and applies that residual before FFN stages consume the activation.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce|qkv-window|qkv-head-window|qkv-head-dim-window`. Head-dim mode keeps the existing scratch writer, adds the dot-product head-dim kernel as the second captured attention phase kernel, mirrors the same math in the CPU reference path, and reports `attention_phase_qkv_head_dim_window_enabled`, `attention_qkv_head_dim_window_head_count`, `attention_qkv_head_dim_window_head_dim`, `attention_qkv_head_dim_window_size`, `attention_qkv_head_dim_window_softmax_scale`, and `attention_qkv_head_dim_window_residual_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheaddim`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-head-dim-window`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780070384.md`.
- Harness result: `77/77` rows passed.
- Checkpoint attention head-dim artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheaddim_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_dim_window`, `attention_phase_qkv_head_dim_window_enabled=true`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/head-dim/window `8/8/4`, admission workspace/graph bytes `172800/183552`, `2562.30` sequence steps/sec, and p50/p95/p99 `0.381/0.408/0.408 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_dim_window`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2036.87` sequence steps/sec, and p50/p95/p99 `0.477/0.573/0.573 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention head-dim and updated all-phase rows are present in the summary.
- Validation after this step: native build passed, Python script compilation passed, full native harness passed `77/77`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Checkpoint summary at that stage:

- Native harness at that stage: `E:\AI project\benchmarks\native_harness_1780070384.md`.
- Harness result: `77/77` rows passed.
- Attention QKV head-dim layer-1 strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheaddim_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2562.30` sequence steps/sec, and p50/p95/p99 `0.381/0.408/0.408 ms`.
- Integrated all-phase strict role-plan artifact at that stage: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2036.87` sequence steps/sec, and p50/p95/p99 `0.477/0.573/0.573 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `77/77`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Attention QKV head-tile-window phase primitive:

- Added `rtxllm_launch_attention_qkv_head_tile_window(...)` to the decode CUDA kernel library. The kernel consumes the Q/K/V scratch rows written by `rtxllm_launch_attention_qkv_scratch`, maps one 32-thread block to each head/context tile, computes the Q/K score and softmax weights once per tile, and writes all `8` V lanes from those shared weights.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce|qkv-window|qkv-head-window|qkv-head-dim-window|qkv-head-tile-window`. Tile mode keeps the existing scratch writer, adds the tiled head-dim kernel as the second captured attention phase kernel, mirrors the same math in the CPU reference path, and reports `attention_phase_qkv_head_tile_window_enabled`, `attention_qkv_head_tile_window_head_count`, `attention_qkv_head_tile_window_head_dim`, `attention_qkv_head_tile_window_size`, `attention_qkv_head_tile_window_softmax_scale`, and `attention_qkv_head_tile_window_residual_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadtile`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-head-tile-window`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780071159.md`.
- Harness result: `78/78` rows passed.
- Checkpoint attention head-tile artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadtile_1780071159.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_tile_window`, `attention_phase_qkv_head_tile_window_enabled=true`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/head-dim/window `8/8/4`, admission workspace/graph bytes `172800/183552`, `2396.50` sequence steps/sec, and p50/p95/p99 `0.363/0.554/0.554 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780071159.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_tile_window`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2013.49` sequence steps/sec, and p50/p95/p99 `0.369/0.709/0.709 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention head-tile and updated all-phase rows are present in the summary.
- Validation after this step: native build passed, Python script compilation passed, full native harness passed `78/78`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Checkpoint summary at that stage:

- Stage native harness: `E:\AI project\benchmarks\native_harness_1780071159.md`.
- Harness result: `78/78` rows passed.
- Stage attention QKV head-tile layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadtile_1780071159.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2396.50` sequence steps/sec, and p50/p95/p99 `0.363/0.554/0.554 ms`.
- Stage integrated all-phase strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780071159.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2013.49` sequence steps/sec, and p50/p95/p99 `0.369/0.709/0.709 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `78/78`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Attention QKV head-group-window phase primitive:

- Added `rtxllm_launch_attention_qkv_head_group_window(...)` to the decode CUDA kernel library. The kernel consumes the same Q/K/V scratch rows, groups `4` query contexts per head block, computes one Q/K score window per query context, and writes all `8` V lanes from shared score/weight values.
- Extended `--attention-phase-mode` to `passthrough|qkv-scratch|qkv-reduce|qkv-window|qkv-head-window|qkv-head-dim-window|qkv-head-tile-window|qkv-head-group-window`. Group mode keeps the existing scratch writer, adds the grouped head-dim kernel as the second captured attention phase kernel, mirrors the same math in the CPU reference path, and reports `attention_phase_qkv_head_group_window_enabled`, `attention_qkv_head_group_window_head_count`, `attention_qkv_head_group_window_head_dim`, `attention_qkv_head_group_window_size`, `attention_qkv_head_group_window_contexts_per_block`, `attention_qkv_head_group_window_softmax_scale`, and `attention_qkv_head_group_window_residual_scale`.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup`.
- Updated the integrated all-phase row to combine `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-head-group-window`, `--ssm-phase-mode scan-scratch`, and `--output-phase-mode final-token`.
- Checkpoint native harness: `E:\AI project\benchmarks\native_harness_1780071919.md`.
- Harness result: `79/79` rows passed.
- Checkpoint attention head-group artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_window`, `attention_phase_qkv_head_group_window_enabled=true`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/head-dim/window/contexts-per-block `8/8/4/4`, admission workspace/graph bytes `172800/183552`, `2762.05` sequence steps/sec, and p50/p95/p99 `0.360/0.370/0.370 ms`.
- Checkpoint all-phase artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2736.35` sequence steps/sec, and p50/p95/p99 `0.361/0.372/0.372 ms`.
- Benchmark summary was regenerated at `E:\AI project\benchmarks\benchmark_summary.md`; the attention head-group and updated all-phase rows are present in the summary.
- Validation after this step: native build passed, Python script compilation passed, full native harness passed `79/79`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Checkpoint summary at that stage:

- Stage native harness: `E:\AI project\benchmarks\native_harness_1780071919.md`.
- Harness result: `79/79` rows passed.
- Stage attention QKV head-group layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2762.05` sequence steps/sec, and p50/p95/p99 `0.360/0.370/0.370 ms`.
- Stage integrated all-phase strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_graph_bytes=192512`, `2736.35` sequence steps/sec, and p50/p95/p99 `0.361/0.372/0.372 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `79/79`, benchmark summary regenerated, and the layer-plan self-test passed `27/27`.

Attention QKV head-group-fused phase primitive:

- Added `rtxllm_launch_attention_qkv_head_group_fused(...)` to the decode CUDA kernel library. The kernel computes Q/K/V projections directly from logits, groups `4` query contexts per head block, applies the scratch-writer-equivalent activation residual without allocating `attention_qkv_state`, computes a small stable Q/K softmax window per grouped query context, aggregates projected V values, and updates the FFN activation in one captured kernel.
- Extended `--attention-phase-mode` to include `qkv-head-group-fused`. Fused mode does not allocate the QKV scratch auxiliary, reports `attention_phase_qkv_head_group_fused_enabled`, and uses a dedicated CPU reference path because it intentionally removes scratch persistence/aging from the grouped-scratch comparison.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadfused`.
- Tested an integrated all-phase fused variant in harness `1780072830`; it passed correctness but regressed to `1754.85` sequence steps/sec with p50/p95/p99 `0.384/0.997/0.997 ms`, so the all-phase row was returned to `qkv-head-group-window`.
- Stage native harness: `E:\AI project\benchmarks\native_harness_1780073010.md`.
- Harness result: `80/80` rows passed.
- Stage attention QKV head-group-fused layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadfused_1780073010.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=0`, `admission_workspace_bytes=74496`, `admission_graph_bytes=177920`, `1879.79` sequence steps/sec, and p50/p95/p99 `0.537/0.644/0.644 ms`.
- Stage attention QKV head-group-window comparison artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780073010.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `1936.86` sequence steps/sec, and p50/p95/p99 `0.375/0.945/0.945 ms`.
- Stage integrated all-phase strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780073010.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `attention_phase_mode=qkv_head_group_window`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2671.65` sequence steps/sec, and p50/p95/p99 `0.375/0.380/0.380 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `80/80`, benchmark summary regenerated, and the all-phase row kept the grouped-scratch attention path because the fused direct comparison saved workspace but lost measured throughput and p50 latency.

Attention QKV head-group-rope-window phase primitive:

- Added `rtxllm_launch_attention_qkv_head_group_rope_window(...)` to the decode CUDA kernel library. The kernel keeps the resident QKV scratch writer and grouped query-context launch shape, rotates Q/K pairs by query/key context with a RoPE-style theta scale before computing the local softmax window, aggregates V lanes from the same scratch arena, and updates the FFN activation without adding auxiliary workspace.
- Extended `--attention-phase-mode` to include `qkv-head-group-rope-window`. RoPE mode keeps the two-kernel grouped-scratch envelope, reports `attention_phase_qkv_head_group_rope_window_enabled` plus theta/head/window metadata, mirrors the same pair-rotation math in the CPU reference gate, and gives a direct comparison against raw grouped-scratch and direct fused attention.
- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadrope`.
- Stage native harness: `E:\AI project\benchmarks\native_harness_1780074053.md`.
- Harness result: `81/81` rows passed.
- Stage attention QKV head-group-rope-window layer-1 strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadrope_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_rope_window`, token `8133/8133`, max logit/KV error `7.51019e-06/7.51019e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_workspace_bytes=172800`, `admission_graph_bytes=183552`, `2595.04` sequence steps/sec, and p50/p95/p99 `0.375/0.405/0.405 ms`.
- Stage attention QKV head-group-window comparison artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `admission_graph_bytes=183552`, `2696.87` sequence steps/sec, and p50/p95/p99 `0.375/0.382/0.382 ms`.
- Stage attention QKV head-group-fused comparison artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadfused_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=0`, `admission_graph_bytes=177920`, `1533.27` sequence steps/sec, and p50/p95/p99 `0.598/0.910/0.910 ms`.
- Stage integrated all-phase strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `attention_phase_mode=qkv_head_group_window`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `1414.88` sequence steps/sec, and p50/p95/p99 `0.527/1.137/1.137 ms` in that noisy full-harness pass.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `81/81`, benchmark summary regenerated, and summary rows for `attentionheadgroup_1780074053`, `attentionheadrope_1780074053`, `attentionheadfused_1780074053`, and `allphase_1780074053` are present in `E:\AI project\benchmarks\benchmark_summary.md`.

Integrated RoPE all-phase comparison:

- Added the full-harness row `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_rope`.
- The new row keeps the same layer-1 strict role-plan, gated FFN phase primitive, SSM scan-scratch primitive, and output final-token primitive as the existing integrated all-phase row, but swaps attention from `qkv-head-group-window` to `qkv-head-group-rope-window`.
- Stage native harness: `E:\AI project\benchmarks\native_harness_1780074658.md`.
- Harness result: `82/82` rows passed.
- Stage integrated all-phase grouped strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780074658.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_window`, token `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2681.14` sequence steps/sec, and p50/p95/p99 `0.369/0.378/0.378 ms`.
- Stage integrated all-phase RoPE strict role-plan artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_rope_1780074658.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_rope_window`, token `62/62`, max logit/KV error `9.57027e-06/9.57027e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2598.58` sequence steps/sec, and p50/p95/p99 `0.377/0.406/0.406 ms`.
- Validation after that checkpoint: Python script compilation passed, full native harness passed `82/82`, benchmark summary regenerated, and summary rows for `attentionheadgroup_1780074658`, `attentionheadrope_1780074658`, `allphase_1780074658`, and `allphase_rope_1780074658` are present in `E:\AI project\benchmarks\benchmark_summary.md`.

Selective SSM phase primitive:

- Added `rtxllm_launch_ssm_selective_scan(...)` to the decode CUDA kernel library. The kernel keeps the existing recurrent-state plus scan-scratch workspace contract, but updates the SSM-output activation through a sigmoid-gated candidate, scan, and recurrent-state mix instead of the simpler scan-scratch recurrence.
- Extended `--ssm-phase-mode` to `passthrough|recurrent-state|scan-scratch|selective-scan`. Selective mode shares `ssm_recurrent_state` and `ssm_scan_scratch`, reports `ssm_phase_selective_scan_enabled`, and emits its gate scales alongside the existing scan telemetry.
- Added CPU reference coverage for selective scan and full-harness rows `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmselective` and `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_selective`.
- Stage native harness: `E:\AI project\benchmarks\native_harness_1780123258.md`.
- Harness result: `84/84` rows passed.
- Stage standalone SSM scan comparison artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan_1780123258.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, recurrent/scan workspace bytes `8192/16384`, `actual_kernel_launches_per_graph=25`, `admission_workspace_bytes=99072`, `admission_graph_bytes=178432`, `2742.73` sequence steps/sec, and p50/p95/p99 `0.363/0.368/0.368 ms`.
- Stage standalone SSM selective artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmselective_1780123258.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `6.3777e-06/6.3777e-06`, same recurrent/scan workspace bytes `8192/16384`, `actual_kernel_launches_per_graph=25`, `admission_workspace_bytes=99072`, `admission_graph_bytes=178432`, `2557.54` sequence steps/sec, and p50/p95/p99 `0.345/0.524/0.524 ms`.
- Stage integrated all-phase grouped artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780123258.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `ssm_phase_mode=scan_scratch`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2693.60` sequence steps/sec, and p50/p95/p99 `0.365/0.382/0.382 ms`.
- Stage integrated all-phase selective artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_selective_1780123258.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `ssm_phase_mode=selective_scan`, max logit/KV error `9.10461e-06/9.10461e-06`, the same `28` actual kernels, the same auxiliary/workspace/graph bytes `5/139264` and `213760/192512`, `2741.42` sequence steps/sec, and p50/p95/p99 `0.362/0.370/0.370 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `84/84`, benchmark summary regenerated, Python unit tests passed `68/68`, and summary rows for `ssmscan_1780123258`, `ssmselective_1780123258`, `allphase_1780123258`, `allphase_rope_1780123258`, and `allphase_selective_1780123258` are present in `E:\AI project\benchmarks\benchmark_summary.md`.

Source-parameterized SSM phase primitive:

- Added `rtxllm_launch_ssm_source_parameter_cache(...)` and `rtxllm_launch_ssm_source_parameterized_scan(...)` to the decode CUDA kernel library. The new mode keeps the existing recurrent-state plus scan-scratch workspace envelope, splits scan scratch into alpha/beta parameter cache halves, caches source-derived alpha and beta from the SSM alpha/beta stages, and updates scan scratch, recurrent state, and output activation from source-derived parameters in the final SSM/output edge.
- Extended `--ssm-phase-mode` to `passthrough|recurrent-state|scan-scratch|selective-scan|source-parameterized`. Source-parameterized mode reports `ssm_phase_source_parameterized_enabled`, reuses recurrent/scan workspace bytes `8192/16384`, and charges three SSM kernels per graph: two parameter-cache kernels plus one parameterized scan kernel.
- Added CPU reference coverage and full-harness rows `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsource` and `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_source`.
- Stage native harness: `E:\AI project\benchmarks\native_harness_1780124538.md`.
- Harness result: `86/86` rows passed.
- Stage standalone SSM source-parameterized artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsource_1780124538.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, recurrent/scan workspace bytes `8192/16384`, `actual_kernel_launches_per_graph=27`, `admission_workspace_bytes=99072`, `admission_graph_bytes=186624`, `2724.61` sequence steps/sec, and p50/p95/p99 `0.368/0.372/0.372 ms`.
- Stage integrated all-phase source-parameterized artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_source_1780124538.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized`, `output_phase_mode=final_token`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/3/1`, `actual_kernel_launches_per_graph=30`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=200704`, `2689.98` sequence steps/sec, and p50/p95/p99 `0.369/0.375/0.375 ms`.
- Validation after that checkpoint: native build passed, Python script compilation passed, full native harness passed `86/86`, benchmark summary regenerated, and summary rows for `ssmsource_1780124538` and `allphase_source_1780124538` are present in `E:\AI project\benchmarks\benchmark_summary.md`.

Fused source-parameterized SSM cache primitive:

- Added `rtxllm_launch_ssm_rmsnorm_feedback_parameter_cache(...)` to fold source-parameter alpha/beta cache writes into the existing RMSNorm feedback edge. This keeps the source-parameterized CPU reference semantics but removes the two separate cache kernels from graph replay when `--ssm-phase-mode source-parameterized-fused` is selected.
- Extended `--ssm-phase-mode` to `passthrough|recurrent-state|scan-scratch|selective-scan|source-parameterized|source-parameterized-fused`. Fused source mode requires `rmsnorm` feedback, reports `ssm_phase_source_parameterized_fused_enabled`, and keeps the recurrent/scan workspace bytes `8192/16384`.
- Added full-harness rows `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused` and `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused`.
- Current native harness: `E:\AI project\benchmarks\native_harness_1780125807.md`.
- Harness result: `88/88` rows passed.
- Current standalone fused source-parameterized artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, recurrent/scan workspace bytes `8192/16384`, `actual_kernel_launches_per_graph=25`, `ssm_phase_kernel_launches_per_graph=1`, `admission_workspace_bytes=99072`, `admission_graph_bytes=178432`, `2546.96` sequence steps/sec, and p50/p95/p99 `0.365/0.447/0.447 ms`.
- Current integrated all-phase fused source artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized_fused`, `output_phase_mode=final_token`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2041.02` sequence steps/sec, and p50/p95/p99 `0.366/0.715/0.715 ms`.
- Validation after the current checkpoint: native build passed, full native harness passed `88/88`, benchmark summary regenerated, and summary rows for `ssmsourcefused_1780125807` and `allphase_sourcefused_1780125807` are present in `E:\AI project\benchmarks\benchmark_summary.md`.

Top-level phase timing telemetry:

- Added top-level phase timing aggregates to `runtime_mixed_sequence_bench.exe`: per-phase mean/p50/p95/p99/sample counts, p50/p95/p99 phase sums, graph-minus-phase residual fields, and the slowest isolated p95 phase.
- Extended `tools/bench/summarize_benchmarks.py` with summary columns for `phase_timing_sum_p95_ms`, `graph_minus_phase_timing_sum_p95_ms`, `phase_timing_slowest_p95_phase`, and `phase_timing_slowest_p95_ms`.
- Current native harness: `E:\AI project\benchmarks\native_harness_1780126797.md`.
- Harness result: `88/88` rows passed.
- Current standalone fused source-parameterized artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, recurrent/scan workspace bytes `8192/16384`, `actual_kernel_launches_per_graph=25`, `ssm_phase_kernel_launches_per_graph=1`, `admission_workspace_bytes=99072`, `admission_graph_bytes=178432`, `2855.92` sequence steps/sec, p50/p95/p99 `0.349/0.355/0.355 ms`, and isolated slowest phase p95 `output=0.273 ms`.
- Current integrated all-phase fused source artifact: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized_fused`, `output_phase_mode=final_token`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, `admission_workspace_bytes=213760`, `admission_graph_bytes=192512`, `2719.42` sequence steps/sec, p50/p95/p99 `0.366/0.374/0.374 ms`, isolated phase p95 attention/FFN/SSM/output `0.302/0.453/0.208/0.046 ms`, and slowest isolated phase `ffn=0.453 ms`.
- Interpretation: phase p95 sums can exceed graph p95 because the phase timings come from separate post-graph replays. Use them to choose the next hot phase, not as additive critical-path timing. The current evidence points at FFN as the slowest isolated phase in the all-phase source-fused path while the whole graph remains clean at sub-`0.4 ms` p95 in this harness pass.
- Validation after the current checkpoint: native build passed, full native harness passed `88/88`, benchmark summary regenerated, and summary rows for `ssmsourcefused_1780126797` and `allphase_sourcefused_1780126797` are present in `E:\AI project\benchmarks\benchmark_summary.md`.
