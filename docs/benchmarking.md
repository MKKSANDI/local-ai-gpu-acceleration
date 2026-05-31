# Benchmarking

Benchmarks write JSON and Markdown reports to `E:\AI project\benchmarks`.

## Required Metrics

- tokens/sec or synthetic steps/sec;
- p50/p95/p99 inter-step latency;
- allocated GPU bytes and high-water estimate;
- H2D bytes;
- D2H bytes;
- GPU free/total memory before and after;
- graph replay rate;
- benchmark path name;
- residency policy;
- WDDM/watchdog state when available.

## First Benchmark

Run:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\numba_residency_bench.py --steps 128 --weights-mib 256 --kv-pages 256
```

This benchmark is synthetic. It exercises CUDA allocation, pinned request staging, decode-like kernel launches, per-token D2H copies, and KV page writes.

## CUDA Graph Benchmark

Run:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\cupy_graph_bench.py --steps 256 --weights-mib 256 --kv-pages 256
```

This benchmark captures one decode-like kernel into a CUDA Graph, uploads the graph, replays it for each synthetic decode step, and copies back only a token-sized value per step.

## Native Benchmark

Build the native C++/CUDA executables through the Visual Studio developer environment:

```powershell
. .\scripts\env.ps1
.\scripts\build_native_vs.ps1
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_bench.exe"
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_graph_bench.exe"
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe"
```

All native build outputs are configured for `E:\AI project\build`.

`runtime_bench` is the launch-per-step native synthetic decode baseline. `runtime_graph_bench` captures a stateful decode kernel plus token-sized D2H copy into a CUDA Graph, uploads it, and replays it for the synthetic decode loop. It is the native counterpart to the current CuPy CUDA Graph benchmark.

`runtime_fused_bench` captures a packed Q4 dequant + matvec + KV page update + GPU argmax token path into a CUDA Graph. It reports Q4 values per replay, packed Q4 bytes per replay, setup H2D bytes, steady-state H2D bytes, token-sized D2H bytes, and graph latency percentiles. The default packed kernel variant is `packed2`, which dequants two Q4 values per packed byte load; `--packed-kernel vec4x4` selects the four-warp row-split packed-payload ceiling. For full Q4_K blocks, `--q4k-kernel predecoded` uploads a setup-time scale/min table and measures the remaining payload-stride path without in-kernel scale/min decode. `--q4k-kernel split-predecoded` reads the staged 128-byte payload stream with the same metadata table, approximating the runtime's internal packed layout. `--q4k-kernel split-half` keeps that split payload but stores predecoded metadata as half pairs, so its reference gate uses an explicit lossy FP16 tolerance. `--q4k-kernel split-compact` and `--q4k-kernel split-native` are exact compact metadata probes that trade speed for lower residency.

It supports shape controls for stress profiles:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label stress_8192x8192 --steps 128 --rows 8192 --cols 8192 --kv-mib 64 --active-pages 512
```

Run the deterministic CPU-reference correctness check:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label reference_64x128 --rows 64 --cols 128 --steps 4 --kv-mib 1 --page-words 256 --active-pages 16 --check-only
```

To write parsed native benchmark artifacts:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\native_harness.py
```

The latest native harness writes per-executable JSON artifacts such as `native_cuda_synthetic_decode_*.json`, `native_cuda_graph_synthetic_decode_*.json`, and labeled `native_cuda_graph_q4_matvec_decode_*.json` files for the default and stress fused profiles. Current full-harness result: `E:\AI project\benchmarks\native_harness_1780125807.md` / `.json`, `88/88` rows passed. The first row is `runtime_layer_plan_selftest.exe`, which passed `27/27` native layer-plan, workspace, allocation, runtime binding, phase-local dataflow, workspace usage, auxiliary FFN cache workspace, SSM recurrent-state workspace, SSM scan-scratch workspace, generic phase-scratch workspace, graph-bucket admission estimate, strict-admission breakdown, over-budget rejection, and executor contract cases before the CUDA/model-backed rows ran, including `workspace_runtime_binding_passed=true`, `dataflow_plan_passed=true`, `unchained_dataflow_passed=true`, `workspace_usage_passed=true`, `aux_workspace_allocation_passed=true`, `ssm_recurrent_workspace_passed=true`, `ssm_scan_workspace_passed=true`, `phase_scratch_workspace_passed=true`, `graph_bucket_plan_passed=true`, `admission_breakdown_passed=true`, and `admission_over_budget_passed=true`. The strict role-plan mixed rows expose `feedback_mode=rmsnorm`, `ffn_phase_mode`, `attention_phase_mode`, `ssm_phase_mode`, `output_phase_mode`, `layer_executor_workspace_plan=phase_local_workspace_descriptors`, `layer_executor_workspace_allocation_plan=phase_local_activation_logits_auxiliary`, `layer_executor_workspace_usage_plan=phase_local_logical_high_water`, `layer_executor_dataflow_plan=phase_local_activation_feedback_edges`, runtime-bound phase-local workspace allocations, named auxiliary workspaces, high-water/slack bytes, CUDA Graph memory-pressure snapshots, strict-admission resident/KV/workspace/DMA/guard breakdowns, graph bucket admission bytes, and named attention/FFN/SSM/output warmup and capture counts of `1/1/1/1`. The role-plan FFN order is now gate/up/down, and the harness includes opt-in `--feedback-mode phase-aware-ffn-silu`, `--feedback-mode phase-aware-ffn-gated-silu`, `--ffn-phase-mode gated-silu`, `--attention-phase-mode qkv-scratch|qkv-reduce|qkv-window|qkv-head-window|qkv-head-dim-window|qkv-head-tile-window|qkv-head-group-window|qkv-head-group-rope-window|qkv-head-group-fused`, `--phase-scratch NAME:PHASE:VALUES`, multi-bucket graph admission rows, `--ssm-phase-mode recurrent-state`, `--ssm-phase-mode scan-scratch`, `--ssm-phase-mode selective-scan`, `--ssm-phase-mode source-parameterized`, `--ssm-phase-mode source-parameterized-fused`, `--output-phase-mode final-token`, integrated grouped/RoPE all-phase rows, and integrated selective/source-parameterized/fused SSM all-phase rows; the attention scratch row launches `rtxllm_launch_attention_qkv_scratch`, the attention reduce row launches both `rtxllm_launch_attention_qkv_scratch` and `rtxllm_launch_attention_qkv_reduce`, the attention window row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_window`, the attention head-window row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_head_window`, the attention head-dim row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_head_dim_window`, the attention head-tile row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_head_tile_window`, the attention head-group row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_head_group_window`, the attention head-rope row launches `rtxllm_launch_attention_qkv_scratch` plus `rtxllm_launch_attention_qkv_head_group_rope_window`, the attention head-fused row launches only `rtxllm_launch_attention_qkv_head_group_fused`, phase-scratch rows launch captured `rtxllm_launch_phase_scratch_digest` kernels after matching phases, the SSM recurrent row launches `rtxllm_launch_ssm_recurrent_state`, the SSM scan row launches `rtxllm_launch_ssm_scan_scratch`, the SSM selective row launches `rtxllm_launch_ssm_selective_scan`, the SSM source-parameterized row launches `rtxllm_launch_ssm_source_parameter_cache` plus `rtxllm_launch_ssm_source_parameterized_scan`, the fused source-parameterized row launches `rtxllm_launch_ssm_rmsnorm_feedback_parameter_cache` plus `rtxllm_launch_ssm_source_parameterized_scan`, the output row launches `rtxllm_launch_output_token_sample`, and the integrated rows run gated FFN plus grouped or RoPE grouped attention, SSM scan/selective/source-parameterized/fused, and output phase kernels during graph replay.

Current SSM comparison rows from harness `1780125807`:

- `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan`: `reference_passed=true`, token `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, `2653.22` sequence steps/sec, p50/p95/p99 `0.366/0.407/0.407 ms`, recurrent/scan workspace bytes `8192/16384`, graph bytes `178432`;
- `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmselective`: `reference_passed=true`, token `8133/8133`, max logit/KV error `6.3777e-06/6.3777e-06`, `2810.96` sequence steps/sec, p50/p95/p99 `0.356/0.358/0.358 ms`, same recurrent/scan workspace bytes and graph bytes;
- `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsource`: `reference_passed=true`, `ssm_phase_mode=source_parameterized`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `2227.92` sequence steps/sec, p50/p95/p99 `0.359/0.709/0.709 ms`, recurrent/scan workspace bytes `8192/16384`, `3` SSM kernels per graph, and graph bytes `186624`;
- `runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused`: `reference_passed=true`, `ssm_phase_mode=source_parameterized_fused`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `2546.96` sequence steps/sec, p50/p95/p99 `0.365/0.447/0.447 ms`, the same recurrent/scan workspace bytes, `1` SSM kernel per graph, and graph bytes `178432`;
- `runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused`: `reference_passed=true`, token `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, `28` actual kernels per graph, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2041.02` sequence steps/sec, and p50/p95/p99 `0.366/0.715/0.715 ms`.

## llama.cpp Comparison
## llama.cpp Comparison
## llama.cpp Comparison

Install prebuilt llama.cpp CUDA binaries to external storage:

```powershell
. .\scripts\env.ps1
.\scripts\setup_llamacpp.ps1
```

Record tool presence, and run a model if a GGUF has been downloaded:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\llamacpp_harness.py
```

The harness writes JSON under `E:\AI project\benchmarks`.

Generate a target-repo command matrix:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\llamacpp_matrix.py
```

Select/download/plan the first target-repo quant and optionally run the llama.cpp baseline:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --download
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --run-llamacpp --ctx-size 1024 --n-predict 32 --timeout 1200
```

The current target artifact is `E:\AI project\benchmarks\target_quant_pipeline_IQ2_M.md`. The first successful target-repo llama.cpp baseline used `IQ2_M`, `-ngl 31`, `-c 1024`, `-n 32`, and measured `9.70` decode tok/s with CUDA graphs reused `30` times.

The latest traced target run used tensor-derived `-ngl 32` and measured `22.15` decode tok/s:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --storage-root "E:\AI project" --label target-iq2m-ngl32 -- $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --storage-root "E:\AI project" --run-llamacpp --ctx-size 1024 --n-predict 32 --timeout 1200
```

Trace artifact: `E:\AI project\traces\target-iq2m-ngl32_20260529_000028.md`.

Generate a comparison report after controlled target runs:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\compare_target_runs.py --storage-root "E:\AI project"
```

Current comparison: `E:\AI project\benchmarks\target_ngl_comparison_IQ2_M.md`. The current controlled rows are `-ngl 28` at `17.41` tok/s, `-ngl 31` at `19.60` tok/s, and `-ngl 32` at `20.86` tok/s.

Generate the target GGUF native-loader pack plan:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\gguf_pack_plan.py --storage-root "E:\AI project"
```

Artifact: `E:\AI project\benchmarks\gguf_pack_plan_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.md`.

Generate the offline Q4_K resident pack:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\q4k_resident_pack.py --storage-root "E:\AI project"
```

Current pack artifact: `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.md`.

The current pack covers all `40` Q4_K tensors in the target `IQ2_M` GGUF:

- payload bytes: `256901120`;
- metadata bytes: `126648320`;
- resident bytes: `383549440`;
- large `attn_qkv` shape `[2048, 8192]`: `split-predecoded` payload plus FP32 product metadata;
- small `attn_v` shape `[2048, 512]`: `split-compact` payload plus exact compact metadata.

Generate a raw IQ resident pack for the next IQ kernel work:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\iq_resident_pack.py --storage-root "E:\AI project" --type IQ2_S --limit 2
& $env:RTXLLM_PYTHON .\tools\bench\iq_resident_pack.py --storage-root "E:\AI project" --type IQ2_S --layer 0
& $env:RTXLLM_PYTHON .\tools\bench\iq_resident_pack.py --storage-root "E:\AI project" --type IQ3_S --layer 0
& $env:RTXLLM_PYTHON .\tools\bench\iq_resident_pack.py --storage-root "E:\AI project" --type IQ2_S --layer 1 --output-dir "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201"
& $env:RTXLLM_PYTHON .\tools\bench\iq_resident_pack.py --storage-root "E:\AI project" --type IQ3_S --layer 1 --output-dir "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201"
```

Latest layer-0 IQ2_S raw-pack artifact: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.md`.

- selected tensors: `8`;
- resident bytes: `178054144`;
- representative tensors: `blk.0.attn_gate.weight`, `blk.0.ffn_gate_exps.weight`, `blk.0.ffn_up_exps.weight`, and `blk.0.ssm_out.weight`;
- policy: raw IQ block copy with source offsets, dimensions, block counts, and checksums recorded for IQ CUDA kernels.

Latest layer-0 IQ3_S raw-pack artifact: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.md`.

- selected tensors: `2`;
- resident bytes: `115793920`;
- tensors: `blk.0.ffn_down_exps.weight` and `blk.0.ffn_down_shexp.weight`;
- policy: raw IQ3_S block copy with source offsets, dimensions, block counts, checksums, and `iq3s_raw_block_probe` labels recorded for the native probe. The same payloads are now consumed by the IQ3_S dequant matvec path.

Latest layer-1 IQ raw-pack artifacts:

- IQ2_S: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201\manifest.md`, selected tensors `8`, resident bytes `178054144`;
- IQ3_S: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201\manifest.md`, selected tensors `2`, resident bytes `115793920`.

Run CUDA-side IQ raw resident block-reader probes:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_iq_probe.exe" --label iq2s_probe_ffn_gate_exps --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --tensor blk.0.ffn_gate_exps.weight --steps 8
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_iq_probe.exe" --label iq3s_probe_ffn_down_exps --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.json" --tensor blk.0.ffn_down_exps.weight --steps 8
```

Latest IQ raw probe artifacts:

- IQ2_S `blk.0.attn_gate.weight`: `E:\AI project\benchmarks\native_iq2s_probe_runtime_iq2s_probe_attn_gate_1780013632.json`, `reference_passed=true`, payload/checksum `2686976/329856869`;
- IQ2_S `blk.0.ffn_gate_exps.weight`: `E:\AI project\benchmarks\native_iq2s_probe_runtime_iq2s_probe_ffn_gate_exps_1780013632.json`, `reference_passed=true`, payload/checksum `85983232/6172290754`;
- IQ3_S `blk.0.ffn_down_exps.weight`: `E:\AI project\benchmarks\native_iq3s_probe_runtime_iq3s_probe_ffn_down_exps_1780013632.json`, `reference_passed=true`, payload/checksum `115343360/11653039204`;
- IQ3_S `blk.0.ffn_down_shexp.weight`: `E:\AI project\benchmarks\native_iq3s_probe_runtime_iq3s_probe_ffn_down_shexp_1780013632.json`, `reference_passed=true`, payload/checksum `450560/53149533`;
- all rows use setup-only H2D and no steady-state H2D. IQ2_S copies 48 bytes D2H per replay; IQ3_S copies 56 bytes D2H per replay for the seven observed checksum/field values.

Run the CUDA-side IQ decode-style matvec probes:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_iq_matvec_probe.exe" --label iq2s_decode_attn_gate --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --tensor blk.0.attn_gate.weight --rows-limit 64 --steps 8 --kv-mib 1 --page-words 256 --active-pages 64
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_iq_matvec_probe.exe" --label iq3s_decode_ffn_down_shexp --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.json" --tensor blk.0.ffn_down_shexp.weight --rows-limit 64 --steps 8 --kv-mib 1 --page-words 256 --active-pages 64
```

Latest IQ decode-style matvec artifacts:

- IQ2_S `blk.0.attn_gate.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_decode_attn_gate_1780013632.json`, `reference_passed=true`, shape `4096x2048`, checked rows `64`, logical values/replay `131072`, max logit/KV error `1.19209e-07`, token expected/observed `104/104`;
- IQ2_S `blk.0.ffn_gate_exps.weight`: `E:\AI project\benchmarks\native_iq2s_matvec_probe_runtime_iq2s_decode_ffn_gate_exps_1780013632.json`, `reference_passed=true`, derived shape `131072x2048`, checked rows `8`, logical values/replay `16384`, max logit/KV error `9.31323e-09`, token expected/observed `48/48`;
- IQ3_S `blk.0.ffn_down_exps.weight`: `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_exps_1780013632.json`, `reference_passed=true`, shape `524288x512`, checked rows `8`, logical values/replay `4096`, max logit/KV error `1.01281e-08`, token expected/observed `56/56`, `20790.00` replays/sec, p50/p95/p99 `0.045/0.062/0.062 ms`;
- IQ3_S `blk.0.ffn_down_shexp.weight`: `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_shexp_1780013632.json`, `reference_passed=true`, shape `2048x512`, checked rows `64`, logical values/replay `32768`, max logit/KV error `1.19209e-07`, token expected/observed `160/160`, `10268.30` replays/sec, p50/p95/p99 `0.052/0.261/0.261 ms`;
- all rows use setup-only H2D, no steady-state H2D, page-style KV writes, GPU token update, CPU/GPU logits+KV+token reference checking, and only `32` timed D2H bytes over `8` graph replays.

Run the multi-tensor IQ2_S sequence benchmark:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_iq_sequence_bench.exe" --label iq2s_sequence_limit2_chained --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --limit 2 --rows-limit 64 --steps 8 --kv-mib 1 --page-words 256 --active-pages 64
```

Latest IQ2_S sequence artifacts:

- chained: `E:\AI project\benchmarks\native_iq2s_sequence_graph_runtime_iq2s_sequence_limit2_chained_1780013632.json`, source-plan checks `2/2`, resident bytes `88670208`, graph kernels/replay `4`, activation feedbacks/replay `2`, logical values/replay `262144`, setup H2D `88678404`, steady-state H2D `0`, D2H `32`;
- no-chain: `E:\AI project\benchmarks\native_iq2s_sequence_graph_runtime_iq2s_sequence_limit2_nochain_1780013632.json`, source-plan checks `2/2`, graph kernels/replay `2`, steady-state H2D `0`, D2H `32`.

Generate and probe the Q5_K raw resident pack:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\q5k_resident_pack.py --storage-root "E:\AI project"
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_q5k_probe.exe" --label q5k_probe_output --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --tensor output.weight --steps 8
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_q5k_matvec_probe.exe" --label q5k_decode_output --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --tensor output.weight --rows-limit 64 --steps 8 --kv-mib 8 --page-words 256 --active-pages 64
```

Latest Q5_K artifacts:

- pack: `E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.md`, tensor `output.weight`, dimensions `[2048, 248320]`, blocks `1986560`, resident bytes `349634560`, checksum `45716023722`;
- raw probe: `E:\AI project\benchmarks\native_q5k_probe_runtime_q5k_probe_output_1780063102.json`, `reference_passed=true`, GPU payload checksum `45716023722`, first-block fields matched (`d=863`, `dmin=5054`, scale sum `2463`, high-bit sum `4291`, low-bit sum `16347`), setup H2D `349634560`;
- dequant matvec: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780063102.json`, `reference_passed=true`, shape `248320x2048`, checked rows `64`, logical values/replay `131072`, setup H2D `349642752`, D2H `32`, token `480/480`, `18294.10` graph replays/sec, p50/p95/p99 `0.054/0.065/0.065 ms`.

Run the mixed Q4_K/IQ layer-slice benchmark, optionally appending IQ3_S down-projection tensors and Q5_K `output.weight`:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_mixed_sequence_bench.exe" --label mixed_layer0_chained --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --layer 0 --rows-limit 64 --steps 8 --kv-mib 8 --page-words 256 --active-pages 64
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_mixed_sequence_bench.exe" --label mixed_layer0_output_chained --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --q5k-tensor output.weight --layer 0 --rows-limit 64 --steps 8 --kv-mib 8 --page-words 256 --active-pages 64
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_mixed_sequence_bench.exe" --label mixed_layer0_iq3_output_chained --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.json" --extra-iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.json" --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --q5k-tensor output.weight --layer 0 --rows-limit 64 --steps 8 --kv-mib 8 --page-words 256 --active-pages 64
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_mixed_sequence_bench.exe" --label mixed_layer1_iq3_output_reference_chained --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201\manifest.json" --extra-iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201\manifest.json" --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --q5k-tensor output.weight --layer 1 --rows-limit 64 --steps 4 --kv-mib 8 --page-words 256 --active-pages 64 --check-reference
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_mixed_sequence_bench.exe" --label mixed_layer1_iq3_output_reference_roleplan --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201\manifest.json" --extra-iq-manifest "E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201\manifest.json" --q5k-manifest "E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.json" --q5k-tensor output.weight --layer 1 --rows-limit 64 --steps 4 --kv-mib 8 --page-words 256 --active-pages 64 --check-reference --stage-order role-plan --strict-layer-plan
```

Checkpoint mixed artifacts from harness `1780031980`:

- Q4_K/IQ2_S chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_sequence_graph_runtime_mixed_layer0_chained_1780031980.json`, tensors `9`, source-plan checks `9/9`, steady-state H2D `0`;
- Q4_K/IQ2_S/Q5_K chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_output_chained_1780031980.json`, tensors `10`, source-plan checks `10/10`, order policy `source_offset_with_global_tail`, steady-state H2D `0`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-0 chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_chained_1780031980.json`, tensors `12`, source-plan checks `12/12`, IQ manifests `2`, type counts `IQ2_S=8`, `IQ3_S=2`, `Q4_K=1`, `Q5_K=1`, phase counts `attention=2`, `ffn=6`, `ssm=3`, `output=1`, resident bytes `656065536`, graph kernels/replay `24`, activation feedbacks/replay `12`, setup H2D `656081924`, steady-state H2D `0`, D2H `32`, `4123.71` sequence steps/sec, p50/p95/p99 `0.227/0.332/0.332 ms`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-0 CPU reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_chained_1780031980.json`, `reference_passed=true`, `stage_order_policy=source_offset_with_global_tail`, `layer_plan_passed=true`, `reference_stage_count=12`, `reference_stage_mismatches=0`, final stage `output.weight`, final rows `64`, token expected/observed `7671/7671`, max final-logit error `1.30385e-08`, max KV error `1.19209e-07`, source-plan checks `12/12`, steady-state H2D `0`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-0 strict role-plan CPU reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780031980.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `stage_order_policy=role_plan_with_global_tail`, `stage_role_plan_unknown_roles=0`, `stage_phase_unknown_roles=0`, `reference_stage_mismatches=0`, token expected/observed `7671/7671`, phase segments `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`, `layer_execution_phase_count=4`, graph kernels by phase `4/12/6/2`, CUDA-event phase timing p50 attention/FFN/SSM/output `0.125/0.186/0.053/0.038 ms`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-1 chained: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_chained_1780031980.json`, tensors `12`, source-plan checks `12/12`, resident bytes `656065536`, graph kernels/replay `24`, activation feedbacks/replay `12`, setup H2D `656081924`, steady-state H2D `0`, D2H `32`, `4163.63` sequence steps/sec, p50/p95/p99 `0.239/0.260/0.260 ms`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-1 CPU reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_chained_1780031980.json`, `reference_passed=true`, `stage_order_policy=source_offset_with_global_tail`, `layer_plan_passed=true`, `reference_stage_count=12`, `reference_stage_mismatches=0`, final stage `output.weight`, final rows `64`, token expected/observed `4359/4359`, max final-logit error `1.02445e-08`, max KV error `6.14673e-08`, source-plan checks `12/12`, steady-state H2D `0`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-1 strict role-plan CPU reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780031980.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `stage_order_policy=role_plan_with_global_tail`, `stage_role_plan_unknown_roles=0`, `stage_phase_unknown_roles=0`, `reference_stage_mismatches=0`, token expected/observed `4359/4359`, phase segments `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`, `layer_execution_phase_count=4`, graph kernels by phase `4/12/6/2`, CUDA-event phase timing p50 attention/FFN/SSM/output `0.085/0.123/0.052/0.043 ms`;
- Q4_K/IQ2_S/IQ3_S/Q5_K layer-0 no-chain: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_nochain_1780031980.json`, tensors `12`, source-plan checks `12/12`, graph kernels/replay `12`, steady-state H2D `0`, D2H `32`.

Latest strict-admission, RMSNorm dataflow, phase-aware FFN SiLU, gated FFN SiLU, FFN phase-primitive, attention QKV scratch/reduce/window/head-window/head-dim-window/head-tile-window/head-group-window/head-group-rope-window/head-group-fused, captured phase-scratch, graph-bucket admission, SSM recurrent-state, SSM scan-scratch/selective/source-parameterized/fused, output final-token, and integrated all-phase mixed artifacts through harness `1780125807`:

- layer-0 strict role-plan reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token expected/observed `5199/5199`, `reference_stage_mismatches=0`, max logit/KV error `7.689e-06/7.689e-06`, admission required/usable bytes `665751040/11259609088`, resident/KV/workspace/DMA/graph bytes `656065536/8388608/74496/1048576/173824`, workspace high-water/slack bytes `74496/0`, feedback edges/phase-crossing/wrap `12/4/1`, phase-aware FFN SiLU edges `0`, admission over-budget bytes `0`, `2653.40` sequence steps/sec, and p50/p95/p99 `0.362/0.429/0.429 ms`;
- layer-1 strict role-plan reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, token expected/observed `8133/8133`, `reference_stage_mismatches=0`, max logit/KV error `7.09295e-06/7.09295e-06`, admission required/usable bytes `665751040/11259609088`, resident/KV/workspace/DMA/graph bytes `656065536/8388608/74496/1048576/173824`, workspace high-water/slack bytes `74496/0`, feedback edges/phase-crossing/wrap `12/4/1`, phase-aware FFN SiLU edges `0`, admission over-budget bytes `0`, `2763.96` sequence steps/sec, and p50/p95/p99 `0.353/0.382/0.382 ms`;
- layer-1 captured phase-scratch role-plan reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_scratch_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, scratch descriptors `attention_qkv_scratch:attention:8192` and `ssm_state_scratch:ssm:4096`, token expected/observed `8133/8133`, max logit/KV error `7.09295e-06/7.09295e-06`, layer-executor phase/aux workspace bytes `74496/49152`, `phase_scratch_workspace_bytes=49152`, `phase_scratch_digest_enabled=true`, `phase_scratch_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, admission workspace/graph bytes `123648/182784`, `2704.71` sequence steps/sec, and p50/p95/p99 `0.363/0.393/0.393 ms`;
- layer-1 graph-bucket admission reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_graphbucket_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token expected/observed `8133/8133`, `graph_bucket_count=3`, `graph_bucket_estimated_bytes=523008`, `graph_bucket_max_estimated_bytes=174848`, bucket totals `173824/174336/174848`, `admission_graph_bytes=523008`, `actual_kernel_launches_per_graph=24`, `2714.44` sequence steps/sec, and p50/p95/p99 `0.364/0.377/0.377 ms`;
- layer-1 attention QKV scratch phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionphase_1780064700.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_scratch`, token expected/observed `8133/8133`, max logit/KV error `4.64916e-06/4.64916e-06`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, admission workspace/graph bytes `172800/179456`, `2789.40` sequence steps/sec, and p50/p95/p99 `0.357/0.362/0.362 ms`;
- layer-1 attention QKV reduce phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionreduce_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_reduce`, `attention_phase_qkv_reduce_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `8.58307e-06/8.58307e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, admission workspace/graph bytes `172800/183552`, `2684.38` sequence steps/sec, and p50/p95/p99 `0.371/0.385/0.385 ms`;
- layer-1 attention QKV window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_window`, `attention_phase_qkv_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `1.01924e-05/1.01924e-05`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, admission workspace/graph bytes `172800/183552`, `2636.96` sequence steps/sec, and p50/p95/p99 `0.379/0.391/0.391 ms`;
- layer-1 attention QKV head-window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_window`, `attention_phase_qkv_head_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `8.53837e-06/8.53837e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/window `8/4`, admission workspace/graph bytes `172800/183552`, `2691.61` sequence steps/sec, and p50/p95/p99 `0.373/0.382/0.382 ms`;
- layer-1 attention QKV head-dim-window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheaddim_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_dim_window`, `attention_phase_qkv_head_dim_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window `8/8/4`, admission workspace/graph bytes `172800/183552`, `2562.30` sequence steps/sec, and p50/p95/p99 `0.381/0.408/0.408 ms`;
- layer-1 attention QKV head-tile-window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadtile_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_tile_window`, `attention_phase_qkv_head_tile_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window `8/8/4`, admission workspace/graph bytes `172800/183552`, `2699.06` sequence steps/sec, and p50/p95/p99 `0.363/0.391/0.391 ms`;
- layer-1 attention QKV head-group-window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_window`, `attention_phase_qkv_head_group_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window/contexts-per-block `8/8/4/4`, admission workspace/graph bytes `172800/183552`, `2755.77` sequence steps/sec, and p50/p95/p99 `0.362/0.366/0.366 ms`;
- layer-1 attention QKV head-group-rope-window phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadrope_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_rope_window`, `attention_phase_qkv_head_group_rope_window_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `7.51019e-06/7.51019e-06`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/head-dim/window/contexts-per-block `8/8/4/4`, RoPE theta scale `0.03125`, admission workspace/graph bytes `172800/183552`, `2760.33` sequence steps/sec, and p50/p95/p99 `0.361/0.367/0.367 ms`;
- layer-1 attention QKV head-group-fused phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadfused_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_fused`, `attention_phase_qkv_head_group_fused_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=0`, head/head-dim/window/contexts-per-block `8/8/4/4`, admission workspace/graph bytes `74496/177920`, `1533.27` sequence steps/sec, and p50/p95/p99 `0.598/0.910/0.910 ms`;
- layer-1 SSM recurrent-state phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmphase_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=recurrent_state`, token expected/observed `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `ssm_recurrent_state_workspace_bytes=8192`, admission workspace/graph bytes `82688/178176`, `2571.52` sequence steps/sec, and p50/p95/p99 `0.367/0.449/0.449 ms`;
- layer-1 SSM scan-scratch phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=scan_scratch`, token expected/observed `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, recurrent/scan workspace bytes `8192/16384`, `ssm_scan_scratch_values=4096`, admission workspace/graph bytes `99072/178432`, `2743.30` sequence steps/sec, and p50/p95/p99 `0.364/0.370/0.370 ms`;
- layer-1 SSM selective-scan phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmselective_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=selective_scan`, token expected/observed `8133/8133`, max logit/KV error `6.3777e-06/6.3777e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, same recurrent/scan workspace bytes `8192/16384`, admission workspace/graph bytes `99072/178432`, `2802.49` sequence steps/sec, and p50/p95/p99 `0.351/0.374/0.374 ms`;
- layer-1 SSM source-parameterized phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsource_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=source_parameterized`, `ssm_phase_source_parameterized_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `ssm_phase_kernel_launches_per_graph=3`, `actual_kernel_launches_per_graph=27`, recurrent/scan workspace bytes `8192/16384`, admission workspace/graph bytes `99072/186624`, `2430.28` sequence steps/sec, and p50/p95/p99 `0.363/0.549/0.549 ms`;
- layer-1 SSM source-parameterized-fused phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=source_parameterized_fused`, `ssm_phase_source_parameterized_fused_enabled=true`, token expected/observed `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, recurrent/scan workspace bytes `8192/16384`, admission workspace/graph bytes `99072/178432`, `2855.92` sequence steps/sec, and p50/p95/p99 `0.349/0.355/0.355 ms`;
- layer-1 output final-token phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_outputphase_1780063102.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `output_phase_mode=final_token`, token expected/observed `63/63`, max logit/KV error `7.09295e-06/7.09295e-06`, `output_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, admission graph bytes `177920`, `2815.91` sequence steps/sec, and p50/p95/p99 `0.354/0.356/0.356 ms`;
- layer-1 phase-aware FFN SiLU role-plan reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_phaseaware_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_silu`, token expected/observed `8022/8022`, `reference_stage_mismatches=0`, max logit/KV error `1.58548e-05/1.58548e-05`, feedback edges/phase-crossing/wrap `12/4/1`, phase-aware FFN SiLU edges `5`, admission graph bytes `173824`, `1974.53` sequence steps/sec, and p50/p95/p99 `0.377/0.797/0.797 ms`;
- layer-1 gated FFN SiLU role-plan reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_gated_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_gated_silu`, token expected/observed `8074/8074`, `reference_stage_mismatches=0`, max logit/KV error `1.10567e-05/1.10567e-05`, feedback edges/phase-crossing/wrap `12/4/1`, gate-cache/gated-combine edges `1/1`, FFN gate-cache workspace bytes `8192`, layer-executor auxiliary workspace bytes `8192`, admission workspace/graph bytes `82688/174080`, `2838.69` sequence steps/sec, and p50/p95/p99 `0.348/0.358/0.358 ms`;
- layer-1 gated FFN phase primitive reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ffnphase_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `ffn_phase_mode=gated_silu`, token expected/observed `8119/8119`, `reference_stage_mismatches=0`, max logit/KV error `7.83987e-06/7.83987e-06`, gate/up/product phase edges `2/2/1`, FFN gate/up cache workspace bytes `8192/8192`, layer-executor phase/aux workspace bytes `74496/16384`, admission workspace/graph bytes `90880/174080`, `2491.75` sequence steps/sec, and p50/p95/p99 `0.385/0.440/0.440 ms`;
- layer-1 integrated all-phase reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `attention_phase_qkv_head_group_window_enabled=true`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token expected/observed `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2742.92` sequence steps/sec, p50/p95/p99 `0.364/0.366/0.366 ms`, and isolated phase p95 attention/FFN/SSM/output `0.254/0.308/0.107/0.052 ms`;
- layer-1 integrated all-phase RoPE reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_rope_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_rope_window`, `attention_phase_qkv_head_group_rope_window_enabled=true`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token expected/observed `62/62`, max logit/KV error `9.57027e-06/9.57027e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2674.87` sequence steps/sec, and p50/p95/p99 `0.370/0.383/0.383 ms`;
- layer-1 integrated all-phase selective-SSM reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_selective_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=selective_scan`, `output_phase_mode=final_token`, token expected/observed `62/62`, max logit/KV error `9.10461e-06/9.10461e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2429.99` sequence steps/sec, and p50/p95/p99 `0.364/0.540/0.540 ms`;
- layer-1 integrated all-phase source-parameterized SSM reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_source_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized`, `output_phase_mode=final_token`, token expected/observed `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/3/1`, `actual_kernel_launches_per_graph=30`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/200704`, `2714.81` sequence steps/sec, and p50/p95/p99 `0.363/0.374/0.374 ms`;
- layer-1 integrated all-phase source-parameterized-fused SSM reference gate: `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized_fused`, `output_phase_mode=final_token`, token expected/observed `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2719.42` sequence steps/sec, p50/p95/p99 `0.366/0.374/0.374 ms`, and isolated phase p95 attention/FFN/SSM/output `0.302/0.453/0.208/0.046 ms`;
- forced strict rejection smoke: `runtime_mixed_sequence_bench.exe` with an oversized WDDM guard rejected before allocation and reported required/usable/over-budget bytes `665577216/0/665577216` with the same weight/KV/workspace/DMA breakdown.

The strict role-plan rows now validate through `runtime/core/scheduler/layer_plan.*`, `runtime/core/scheduler/layer_executor.h`, and `runtime/core/graphs/cuda_graph_bucket.h`, so the same role ranking, phase classification, phase segmentation, admission rule, graph-work accounting, graph-bucket estimate, activation-feedback edge routing, feedback-mode dispatch, FFN phase-mode dispatch, attention phase-mode dispatch, SSM phase-mode dispatch, output phase-mode dispatch, auxiliary/scratch workspace allocation, CUDA-event phase timing, and strict memory breakdown are available to future layer executors. `--feedback-mode synthetic` remains available as a direct comparison path. `--feedback-mode phase-aware-ffn-silu` is now part of the full harness as a layer-1 strict role-plan row, with the same `12/4/1` dataflow edge counts and `5` FFN-internal SiLU feedback edges. `--feedback-mode phase-aware-ffn-gated-silu` is the first gate/up/down-aware handoff: it caches the SiLU gate output and combines it with the up output before down projection. `--ffn-phase-mode gated-silu` moves that behavior into a phase-local primitive: gate/up branch matvecs retain the same input activation, branch outputs are cached separately in core-owned `ffn_gate_cache`/`ffn_up_cache` auxiliary workspace slices, and only the product writes the activation consumed by down projection. `--attention-phase-mode qkv-scratch` reserves core-owned attention QKV scratch, adds one captured attention kernel through `rtxllm_launch_attention_qkv_scratch`, updates the FFN activation, and is mirrored in the CPU reference gate. `--attention-phase-mode qkv-reduce` keeps that scratch writer, then launches `rtxllm_launch_attention_qkv_reduce` to consume Q/K/V scratch rows and apply a bounded weighted residual before FFN consumption. `--attention-phase-mode qkv-window` keeps the same scratch writer, then launches `rtxllm_launch_attention_qkv_window` to compute a small stable Q/K softmax window and V-context residual inside graph replay. `--attention-phase-mode qkv-head-window` keeps the same two-kernel envelope but launches `rtxllm_launch_attention_qkv_head_window`, which interprets scratch rows as `8` contiguous head blocks and computes a local softmax window within the selected head before V aggregation. `--attention-phase-mode qkv-head-dim-window` extends that envelope with `rtxllm_launch_attention_qkv_head_dim_window`, grouping scratch as head/context/head-dim blocks and scoring each key through an 8-lane Q/K dot product before V-lane aggregation. `--attention-phase-mode qkv-head-tile-window` keeps the same math but computes the score/weight window once per head/context tile in `rtxllm_launch_attention_qkv_head_tile_window`. `--attention-phase-mode qkv-head-group-window` keeps the same score sharing while grouping four query contexts per head block in `rtxllm_launch_attention_qkv_head_group_window`. `--attention-phase-mode qkv-head-group-rope-window` keeps the grouped-scratch envelope and applies a RoPE-style Q/K pair rotation by query/key context before the local softmax, making the attention primitive more position-aware without adding workspace. `--attention-phase-mode qkv-head-group-fused` removes the QKV scratch auxiliary and collapses the attention primitive into one captured kernel, but current evidence shows worse p50 and throughput, so the integrated all-phase row keeps the grouped-scratch path. `--phase-scratch NAME:PHASE:VALUES` reserves core-owned per-phase scratch slices and exercises them with `rtxllm_launch_phase_scratch_digest` during graph capture/replay; the current attention/SSM row adds one scratch digest kernel to attention and one to SSM for `26` actual graph kernel launches. `--graph-bucket-count` models multiple captured decode buckets and charges their estimated metadata plus workspace guard bytes before allocation. `--ssm-phase-mode recurrent-state` is the first SSM-local phase primitive: it allocates `ssm_recurrent_state`, adds one captured SSM kernel through `rtxllm_launch_ssm_recurrent_state`, updates the output activation, and is mirrored in the CPU reference gate. `--ssm-phase-mode scan-scratch` extends that path with a core-owned `ssm_scan_scratch` auxiliary slice and `rtxllm_launch_ssm_scan_scratch`, updating scan scratch, recurrent state, and output activation inside captured graph replay. `--ssm-phase-mode selective-scan` keeps the same recurrent/scan workspace envelope and launches `rtxllm_launch_ssm_selective_scan`, adding sigmoid-gated candidate, scan, and recurrent-state mixing inside the same one-kernel SSM phase slot. `--output-phase-mode final-token` adds one captured output sampler kernel through `rtxllm_launch_output_token_sample`, keeps final token selection on GPU, and is mirrored in the CPU reference gate. The integrated all-phase rows prove these phase-local primitives can share one strict role-plan graph, one workspace allocation plan, and one CPU reference path without losing token/logit/KV agreement.

The `phase_timing_*` top-level JSON fields and benchmark-summary columns expose separate post-graph phase p50/p95/p99 aggregates, the slowest isolated phase, and graph-minus-phase residuals. These phase p95 values are not additive critical-path timing because they come from separate phase replays, but they give a stable hot-spot signal beside pure captured-graph wall-time.

For CUDA reference validation against packer-created bytes, write representative block streams:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\q4k_resident_pack.py --storage-root "E:\AI project" --output-dir "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\validation_reps" --write-blocks --tensor blk.0.attn_qkv.weight --tensor blk.3.attn_v.weight
```

The representative validation pack is `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\validation_reps\manifest.md`. The native CUDA reference gates passed for:

- `blk.0.attn_qkv.weight`, `split-predecoded`, max logit/KV abs error `3.93391e-06/1.3113e-06`, token expected/observed `4560/4560`;
- `blk.3.attn_v.weight`, `split-compact`, max logit/KV abs error `1.19209e-06/1.19209e-06`, token expected/observed `324/324`.

The native fused benchmark can consume these manifests directly:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label manifest_full_blk0 --steps 16 --kv-mib 64 --active-pages 512 --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --q4k-tensor blk.0.attn_qkv.weight
```

Detailed manifest-backed native harness from `1780031980`: `E:\AI project\benchmarks\native_harness_1780031980.md`.

- manifest reference `blk.0.attn_qkv.weight`: `reference_passed=true`, token `4560/4560`, artifact `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_manifest_blk0_reference_check_1780031980.json`;
- manifest reference `blk.3.attn_v.weight`: `reference_passed=true`, token `324/324`, artifact `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_manifest_blk3_reference_check_1780031980.json`;
- full manifest `blk.0.attn_qkv.weight`: `8637.91` graph replays/sec over `16` steps, setup H2D `12591108` bytes, artifact `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_q4k_manifest_blk0_1780031980.json`;
- full manifest `blk.3.attn_v.weight`: `9685.23` graph replays/sec over `16` steps, setup H2D `614404` bytes, artifact `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_q4k_manifest_blk3_1780031980.json`.

Run the multi-tensor Q4_K sequence benchmark:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_q4k_sequence_bench.exe" --label q4k_sequence_full40_chained --q4k-manifest "E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.json" --limit 40 --steps 16 --kv-mib 64 --active-pages 512
```

By default the sequence runner inserts one GPU activation-feedback kernel after each Q4_K matvec, so the graph carries a device-side activation dependency between resident tensors. Use `--no-chain-activation` to run the earlier static-activation comparison.

Detailed chained sequence artifact from `1780057719`: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_chained_1780057719.json`.

Detailed no-chain comparison artifact from `1780057719`: `E:\AI project\benchmarks\native_q4k_manifest_sequence_graph_runtime_q4k_sequence_bench_full40_nochain_1780057719.json`.

- selected tensors: `40`;
- source-plan validation: `40/40` selected tensors checked against `E:\AI project\benchmarks\gguf_tensor_plan_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.json`;
- observed layers: `40`;
- role counts: `30` `attn_qkv.weight`, `10` `attn_v.weight`;
- graph kernels per replay: `80` in chained mode, `40` without activation chaining;
- activation feedback kernels per replay: `40` in chained mode;
- resident bytes: `383549440`;
- allocated bytes: `451747840`;
- setup H2D bytes: `383557636`;
- steady-state H2D bytes: `0`;
- D2H bytes over `16` replays: `64`;
- chained sequence steps/sec: `70.95`;
- chained tensor launches/sec inside graph replay: `2838.17`;
- chained p50/p95/p99 replay latency: `1.935/70.376/70.376 ms`;
- no-chain sequence steps/sec: `475.20`;
- no-chain tensor launches/sec inside graph replay: `19007.90`;
- no-chain p50/p95/p99 replay latency: `1.907/2.710/2.710 ms`.

The native harness now also runs `runtime_gguf_probe.exe` against the local target GGUF:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\native_harness.py --storage-root "E:\AI project" --timeout 180
```

Latest native harness artifacts: `E:\AI project\benchmarks\native_harness_1780126797.md` and `E:\AI project\benchmarks\native_harness_1780126797.json`.

Latest native probe harness artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1780013632.json`.

The native harness table includes the `q4k smoke`, `q4k dequant`, `q4k checksum`, and `q4k absmax` columns. Current target result: `q4k smoke=True`, `q4k dequant=True`, checksum `16789`, decoded scale/min checksums `408/413`, dequant absmax `0.0490355`, and full-block checksum `1217417794`.

The native harness also dumps the full `blk.0.attn_qkv.weight` Q4 payload and runs `runtime_fused_bench` against it:

- payload: `E:\AI project\tmp\blk0_attn_qkv_q4_payload.bin`;
- full blocks: `E:\AI project\tmp\blk0_attn_qkv_q4_blocks.bin`;
- packed-payload vec4x4 fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_payload_vec4x4_blk0_1779988860.json`;
- warp-broadcast-vec4x4 full-block fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_vec4x4_blk0_1779988860.json`;
- predecoded-metadata full-block fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_blocks_predecoded_blk0_1779988860.json`;
- split-payload predecoded fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_predecoded_blk0_1779988860.json`;
- split-payload half-metadata fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_half_blk0_1779988860.json`;
- split-payload compact metadata fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_compact_blk0_1779988860.json`;
- split-payload native metadata fused artifact: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_native_blk0_1779988860.json`;
- split-payload compact metadata reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_compact_reference_check_1779988860.json`;
- split-payload native metadata reference artifact: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_native_reference_check_1779988860.json`;
- packed-payload vec4x4 result: `9574.32` graph replays/sec, `p50/p95/p99=0.082/0.266/0.512 ms`, checksum `1055786949`;
- warp-broadcast-vec4x4 full-block Q4_K result: `5984.50` graph replays/sec, `p50/p95/p99=0.130/0.335/0.460 ms`, checksum `1217417794`;
- predecoded-metadata full-block Q4_K result: `8065.33` graph replays/sec, `p50/p95/p99=0.100/0.279/0.554 ms`, checksum `1217417794`, setup metadata `4194304` bytes;
- split-payload predecoded Q4_K result: `7946.86` graph replays/sec, `p50/p95/p99=0.097/0.287/0.436 ms`, payload checksum `1055786949`, source checksum `1217417794`, setup metadata `4194304` bytes;
- split-payload half-metadata Q4_K result: `8437.32` graph replays/sec, `p50/p95/p99=0.093/0.279/0.440 ms`, payload checksum `1055786949`, source checksum `1217417794`, setup metadata `2097152` bytes;
- split-payload compact metadata Q4_K result: `5539.61` graph replays/sec, `p50/p95/p99=0.139/0.359/0.476 ms`, payload checksum `1055786949`, source checksum `1217417794`, setup metadata `1310720` bytes;
- split-payload native metadata Q4_K result: `6587.41` graph replays/sec, `p50/p95/p99=0.127/0.314/0.453 ms`, payload checksum `1055786949`, source checksum `1217417794`, setup metadata `1048576` bytes;
- references: compact and native metadata passed with strict tolerance `0.005`, max logit/KV abs error around `7.4e-06/7.0e-06`, and token expected/observed `1236/1236`. The half-metadata row uses tolerance `0.0125` because its metadata is intentionally lossy FP16.

Representative Q4_K tensor sweep:

- sweep report: `E:\AI project\benchmarks\q4k_tensor_sweep_1779988867.md`;
- JSON: `E:\AI project\benchmarks\q4k_tensor_sweep_1779988867.json`;
- `blk.3.attn_v.weight` `[2048, 512]`: packed vec4x4 `15735.30`, full-block vec4x4 `15136.30`, split-half `14823.60`, split-native `14857.50`, all reference gates passed;
- `blk.0.attn_qkv.weight` `[2048, 8192]`: packed vec4x4 `7654.59`, full-block vec4x4 `5899.43`, split-predecoded `7042.99`, split-half `7057.75`, split-native `5612.46`, all reference gates passed;
- interpretation: precomputed product metadata is still the speed path for the large QKV tensor, while the smaller V tensor is launch/latency dominated enough that exact compact layouts are competitive and the FP16 side table leads this run. The sweep is now a required check before promoting a Q4_K internal layout into the packer.

Run the repeatable suite:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_bench_suite.py --plan-only
& $env:RTXLLM_PYTHON .\tools\bench\run_bench_suite.py --include-gpu --include-native --include-llamacpp-smoke
& $env:RTXLLM_PYTHON .\tools\bench\run_bench_suite.py --include-native --include-resident-pack
& $env:RTXLLM_PYTHON .\tools\bench\run_bench_suite.py --include-llamacpp-smoke --trace-llamacpp-smoke
```

The first command refreshes planning artifacts only. The second includes CUDA synthetic benchmarks, native C++/CUDA benchmarks, and the tiny llama.cpp smoke model. The third runs native benchmarks plus the Q4_K resident packer. The fourth runs the llama.cpp smoke path under the GPU trace wrapper and writes sampled telemetry to `E:\AI project\traces`.

Plan-only runs refresh:

- model fit plan;
- residency policy simulation;
- offload profiles, using tensor-derived layers for local GGUFs and assumed layers for missing GGUFs;
- local GGUF tensor/layer residency plan;
- target quant selection and local tensor plan;
- allocator simulation;
- KV cache simulation;
- llama.cpp command matrix;
- benchmark summary.

## GPU Trace Envelope

To trace any command directly:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\run_with_gpu_trace.py --label my-run -- <command> <args>
```

Each trace captures child wall time, return code, output tail, sampled GPU utilization, VRAM used/free, power draw, and GPU temperature.
