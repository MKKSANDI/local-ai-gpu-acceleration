# Primary Model Target

Repository:

```text
HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
```

Primary stress file:

```text
Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
```

The Hugging Face file listing currently reports the Q8_K_P GGUF at about 43.6 GB. On a 12 GB RTX 3060 this cannot be a fully resident weight configuration. It is therefore the long-term residency and streaming stress test, not the first fully resident inference target.

Near-term validation should use smaller quants from the same repo for comparative behavior:

- `IQ2_M`: smallest practical smoke target;
- `Q2_K_P` or `IQ3_M`: early streaming/residency tests;
- `Q4_K_M` or `Q4_K_P`: realistic quality/performance midpoint;
- `Q8_K_P`: final stress target.

The runtime must keep strict mode honest: if a configuration cannot fit, it reports that before generation instead of relying on driver or OS paging.

Generate the current fit plan with:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\model_plan.py
```

The plan writes `E:\AI project\benchmarks\model_plan.json` and `E:\AI project\benchmarks\model_plan.md`.

Generate llama.cpp command profiles for strict, balanced, and overflow target-repo tests:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\llamacpp_matrix.py
```

The matrix writes `E:\AI project\benchmarks\llamacpp_matrix.json` and `E:\AI project\benchmarks\llamacpp_matrix.md`.

Simulate residency policy decisions against the target-repo model plan:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\policy_sim.py
```

The simulator writes `E:\AI project\benchmarks\policy_sim.json` and `E:\AI project\benchmarks\policy_sim.md`.

Generate offload profiles and `-ngl` values:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\offload_plan.py
```

The planner writes `E:\AI project\benchmarks\offload_plan.json` and `E:\AI project\benchmarks\offload_plan.md`. Local GGUF rows use tensor-derived layer spans; not-yet-downloaded rows still use the assumed layer count.

Inspect a downloaded GGUF header without loading tensor data:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\gguf_inspect.py "E:\AI project\models\...\model.gguf"
```

Generate a tensor/layer residency plan for a downloaded GGUF:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\gguf_tensor_plan.py "E:\AI project\models\...\model.gguf"
```

For now the default invocation plans the local smoke GGUF. Once a target-repo quant is downloaded, this replaces assumed layer counts with real tensor spans and `blk.N.*` layer grouping.

## Target Quant Pipeline

Select the smallest target-repo GGUF, download it to external storage, derive a tensor/layer residency plan, and optionally run the llama.cpp baseline:

```powershell
. .\scripts\env.ps1
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --download
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --run-llamacpp --ctx-size 1024 --n-predict 32 --timeout 1200
```

The current downloaded target quant is:

```text
E:\AI project\models\HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
```

Current tensor-plan facts for `IQ2_M`:

- architecture: `qwen35moe`;
- tensor count: `733`;
- layer count: `40`;
- tensor span: about `11108.63 MiB`;
- strict full residency: `False`;
- current complete-layer estimate: about `31-32/40`, depending on free VRAM at planning time.

First llama.cpp target-repo run:

- command used `-ngl 31`, `-c 1024`, `-n 32`;
- prompt eval: `3.26 tok/s`;
- decode: `9.70 tok/s`;
- graphs reused: `30`;
- load time: `215615.11 ms`;
- artifact: `E:\AI project\benchmarks\llamacpp_harness_1779978087.json`.

Latest traced target-repo run:

- command used tensor-derived `-ngl 32`, `-c 1024`, `-n 32`;
- prompt eval: `4.28 tok/s`;
- decode: `22.15 tok/s`;
- graphs reused: `30`;
- load time: `205488.47 ms`;
- trace artifact: `E:\AI project\traces\target-iq2m-ngl32_20260529_000028.md`;
- max sampled VRAM: `10065 MiB`;
- min sampled free VRAM: `2046 MiB`;
- max sampled GPU utilization: `99%`;
- average sampled GPU utilization over the full load+run envelope: `29.569%`.

Controlled `-ngl` comparison:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\compare_target_runs.py --storage-root "E:\AI project"
```

Current comparison artifact: `E:\AI project\benchmarks\target_ngl_comparison_IQ2_M.md`.

| ngl | decode tok/s | max VRAM MiB | min free MiB |
|---:|---:|---:|---:|
| 28 | 17.41 | 9009 | 3102 |
| 31 | 19.60 | 9801 | 2310 |
| 32 | 20.86 | 10072 | 2039 |

The controlled result keeps `-ngl 32` as the best current offload point for this short `IQ2_M` run, while `-ngl 28` is the lower-VRAM latency-guard point.

## Native Loader Pack Plan

Generate the source tensor role/type plan for the local target GGUF:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\gguf_pack_plan.py --storage-root "E:\AI project"
```

Current artifact: `E:\AI project\benchmarks\gguf_pack_plan_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.md`.

Source type summary for `IQ2_M`:

| type | tensors | span MiB | required native action |
|---|---:|---:|---|
| IQ2_S | 375 | 9841.20 | raw resident block packer plus CUDA block-reader, decode-style dequant/KV/token probes, source-plan-validated IQ sequence graph, and mixed Q4_K/IQ2_S/IQ3_S/Q5_K output-slice graph |
| IQ3_S | 16 | 573.63 | raw resident block packer plus CUDA block-reader, decode-style dequant/KV/token probes, and mixed Q4_K/IQ2_S/IQ3_S/Q5_K output-slice graph |
| Q4_K | 40 | 275.62 | resident payload/metadata packer |
| Q5_K | 1 | 333.44 | raw resident block packer plus CUDA block-reader and dequantized matvec probes |
| F32 | 301 | 84.74 | scalar metadata/parameter conversion |

The Q4_K source tensors now have an offline resident packer, and the native fused benchmark can consume individual manifest tensors directly. `IQ2_S`/`IQ3_S` now have raw resident block packers that record source-verified payload manifests, CUDA block-reader validation, and decode-style dequant/KV/token probes. `IQ2_S` also has a source-plan-validated IQ sequence graph over selected resident tensors, and both IQ families can now be composed with Q4_K and Q5_K in the mixed output-slice graph. `Q5_K` now has a raw resident packer, CUDA block-reader probe, decode-style dequant/KV/token probe, and optional composed-graph integration for `output.weight`. The custom native runtime still needs a real transformer-layer executor.

Native GGUF probe:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_gguf_probe.exe" --model "E:\AI project\models\HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
```

Current native harness artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1779988860.json`.

Native probe result:

- version: `3`;
- architecture: `qwen35moe`;
- tensor count: `733`;
- supported tensors: `733`;
- valid tensors: `733`;
- invalid tensors: `0`;
- expected data bytes: `11648246272`;
- padding bytes: `0`.

Native `Q4_K` data smoke:

- tensor: `blk.0.attn_qkv.weight`;
- dimensions: `[2048, 8192]`;
- absolute offset: `581848704`;
- span bytes: `9437184`;
- block count: `65536`;
- first block read: `144` bytes;
- packed Q4 payload bytes: `128`;
- Q4 values: `256`;
- nibble range: `0..15`;
- packed payload checksum: `16789`;
- FP16 superblock scales: `d=8.50558e-05`, `dmin=0.000560284`;
- decoded scale/min checksums: `408/413`;
- decoded dequant range: `-0.0352979..0.0490355`;
- decoded dequant absmax: `0.0490355`;
- result: `ok`.

Full `Q4_K` payload staging:

- payload path: `E:\AI project\tmp\blk0_attn_qkv_q4_payload.bin`;
- payload bytes: `8388608`;
- full payload checksum: `1055786949`;
- full block path: `E:\AI project\tmp\blk0_attn_qkv_q4_blocks.bin`;
- full block bytes: `9437184`;
- full block checksum: `1217417794`;
- harness probe artifact: `E:\AI project\benchmarks\native_gguf_probe_runtime_gguf_probe_iq2m_1779988860.json`.

The fused CUDA graph benchmark can now consume the staged target bytes:

```powershell
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_fused_bench.exe" --label target_q4k_blk0 --rows 2048 --cols 8192 --steps 128 --kv-mib 64 --active-pages 512 --weights-file "E:\AI project\tmp\blk0_attn_qkv_q4_payload.bin"
```

Latest comparison harness: `E:\AI project\benchmarks\native_harness_1779988860.md`.

Latest compact metadata artifacts:

- split-payload compact metadata: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_compact_blk0_1779988860.json`;
- split-payload native metadata: `E:\AI project\benchmarks\native_cuda_graph_q4_matvec_decode_runtime_fused_bench_target_q4k_split_native_blk0_1779988860.json`;
- split-payload compact metadata reference: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_compact_reference_check_1779988860.json`;
- split-payload native metadata reference: `E:\AI project\benchmarks\native_cuda_q4_matvec_reference_check_runtime_fused_q4k_split_native_reference_check_1779988860.json`.

Current default packed-payload result:

- packed payload weight source: `file`;
- host weight checksum: `1055786949`;
- q4 values per step: `16777216`;
- q4 packed bytes per step: `8388608`;
- graph replay rate: `1.0`;
- steps/sec: `5854.29`;
- p50/p95/p99: `0.140/0.328/0.620 ms`.

Current packed-payload vec4x4 result:

- packed kernel: `vec4x4`;
- weight source: `file`;
- host weight checksum: `1055786949`;
- q4 values per step: `16777216`;
- q4 packed bytes per step: `8388608`;
- graph replay rate: `1.0`;
- steps/sec: `9574.32`;
- p50/p95/p99: `0.082/0.266/0.512 ms`.

Current direct full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `2567.99`;
- p50/p95/p99: `0.336/0.643/0.905 ms`.

Current shared-scale full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `3312.86`;
- p50/p95/p99: `0.255/0.515/0.686 ms`.

Current warp full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `3352.54`;
- p50/p95/p99: `0.236/0.532/0.957 ms`.

Current warp-broadcast full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `4327.76`;
- p50/p95/p99: `0.194/0.407/0.630 ms`.

Current warp-broadcast-vec4 full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `4714.67`;
- p50/p95/p99: `0.180/0.383/0.611 ms`.

Current warp-broadcast-vec4x2 full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `5837.76`;
- p50/p95/p99: `0.139/0.349/0.581 ms`.

Current warp-broadcast-vec4x4 full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- graph replay rate: `1.0`;
- steps/sec: `5984.50`;
- p50/p95/p99: `0.130/0.335/0.460 ms`.

Current predecoded-metadata full-block `Q4_K` result:

- weight source: `q4k_blocks_file`;
- host weight checksum: `1217417794`;
- Q4_K block bytes per step: `9437184`;
- predecoded metadata bytes: `4194304`;
- graph replay rate: `1.0`;
- steps/sec: `8065.33`;
- p50/p95/p99: `0.100/0.279/0.554 ms`.

Current split-payload predecoded `Q4_K` result:

- weight source: `q4k_split_payload_file`;
- host payload checksum: `1055786949`;
- source full-block checksum: `1217417794`;
- Q4 packed bytes per step: `8388608`;
- Q4_K source block bytes per step: `9437184`;
- predecoded metadata bytes: `4194304`;
- graph replay rate: `1.0`;
- steps/sec: `7946.86`;
- p50/p95/p99: `0.097/0.287/0.436 ms`.

Current split-payload half-metadata `Q4_K` result:

- weight source: `q4k_split_payload_file`;
- host payload checksum: `1055786949`;
- source full-block checksum: `1217417794`;
- Q4 packed bytes per step: `8388608`;
- Q4_K source block bytes per step: `9437184`;
- predecoded metadata bytes: `2097152`;
- predecoded metadata format: `fp16`;
- graph replay rate: `1.0`;
- steps/sec: `8437.32`;
- p50/p95/p99: `0.093/0.279/0.440 ms`.

Current split-payload compact metadata `Q4_K` result:

- weight source: `q4k_split_payload_file`;
- host payload checksum: `1055786949`;
- source full-block checksum: `1217417794`;
- Q4 packed bytes per step: `8388608`;
- Q4_K source block bytes per step: `9437184`;
- predecoded metadata bytes: `1310720`;
- predecoded metadata format: `compact-u8`;
- graph replay rate: `1.0`;
- steps/sec: `5539.61`;
- p50/p95/p99: `0.139/0.359/0.476 ms`.

Current split-payload native metadata `Q4_K` result:

- weight source: `q4k_split_payload_file`;
- host payload checksum: `1055786949`;
- source full-block checksum: `1217417794`;
- Q4 packed bytes per step: `8388608`;
- Q4_K source block bytes per step: `9437184`;
- predecoded metadata bytes: `1048576`;
- predecoded metadata format: `native16`;
- graph replay rate: `1.0`;
- steps/sec: `6587.41`;
- p50/p95/p99: `0.127/0.314/0.453 ms`.

Packed-payload vec4x4 reference result:

- reference passed: `true`;
- tolerance: `0.005`;
- max logit abs error: `5.24521e-05`;
- max KV abs error: `3.62396e-05`;
- logit/KV mismatches: `0/0`;
- token expected/observed: `288/288`.

Full-block `Q4_K` reference result:

- reference passed: `true`;
- tolerance: `0.005`;
- max logit abs error: `7.39098e-06`;
- max KV abs error: `7.27177e-06`;
- logit/KV mismatches: `0/0`;
- token expected/observed: `1236/1236`.

The warp reference row has max logit/KV error `7.62939e-06/7.03335e-06` and the same `1236/1236` token agreement.

The warp-broadcast reference row has max logit/KV error `7.62939e-06/7.03335e-06` and the same `1236/1236` token agreement.

The warp-broadcast-vec4 reference row has max logit/KV error `7.39098e-06/7.03335e-06` and the same `1236/1236` token agreement.

The warp-broadcast-vec4x2 reference row has max logit/KV error `7.39098e-06/7.03335e-06` and the same `1236/1236` token agreement.

The warp-broadcast-vec4x4 reference row has max logit/KV error `7.39098e-06/7.03335e-06` and the same `1236/1236` token agreement.

The predecoded-metadata reference row has max logit/KV error `7.39098e-06/7.03335e-06`, metadata bytes `4194304`, and the same `1236/1236` token agreement.

The split-payload predecoded reference row has max logit/KV error `7.39098e-06/7.03335e-06`, payload checksum `1055786949`, source checksum `1217417794`, metadata bytes `4194304`, and the same `1236/1236` token agreement.

The split-payload half-metadata reference row has max logit/KV error `0.0090878/0.0090878`, payload checksum `1055786949`, source checksum `1217417794`, metadata bytes `2097152`, explicit lossy tolerance `0.0125`, and the same `1236/1236` token agreement.

The split-payload compact metadata reference row has max logit/KV error `7.39098e-06/7.03335e-06`, payload checksum `1055786949`, source checksum `1217417794`, metadata bytes `1310720`, metadata format `compact-u8`, and the same `1236/1236` token agreement.

The split-payload native metadata reference row has max logit/KV error `7.39098e-06/7.03335e-06`, payload checksum `1055786949`, source checksum `1217417794`, metadata bytes `1048576`, metadata format `native16`, and the same `1236/1236` token agreement.

Representative target-GGUF Q4_K tensor sweep:

- sweep report: `E:\AI project\benchmarks\q4k_tensor_sweep_1779988867.md`;
- selected tensors: `blk.3.attn_v.weight` with dimensions `[2048, 512]`, and `blk.0.attn_qkv.weight` with dimensions `[2048, 8192]`;
- all Q4_K reference gates passed after fixing allocator reset ordering and GPU argmax tie-breaking;
- `blk.3.attn_v.weight` results: packed vec4x4 `15735.30` steps/sec, full-block vec4x4 `15136.30`, split-predecoded `14855.90`, split-half `14823.60`, split-compact `15297.10`, split-native `14857.50`;
- `blk.0.attn_qkv.weight` results: packed vec4x4 `7654.59` steps/sec, full-block vec4x4 `5899.43`, split-predecoded `7042.99`, split-half `7057.75`, split-compact `5900.82`, split-native `5612.46`;
- interpretation: large QKV still benefits from precomputed products; small V is less sensitive to metadata reconstruction and is dominated by launch/latency overhead. The resident packer should keep shape-specific metadata policy rather than forcing one Q4_K layout globally.

Q4_K resident pack:

- pack manifest: `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1779989132\manifest.md`;
- tensors packed: `40/40` Q4_K tensors;
- source bytes: `289013760`;
- payload bytes: `256901120`;
- metadata bytes: `126648320`;
- resident bytes: `383549440`;
- policy: `auto`, with large `attn_qkv` tensors using `split-predecoded` and small `attn_v` tensors using `split-compact`;
- representative validation pack: `E:\AI project\packs\q4k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\validation_reps\manifest.md`;
- validation: `blk.0.attn_qkv.weight` passed the `split-predecoded` CUDA reference gate with token `4560/4560`, and `blk.3.attn_v.weight` passed the `split-compact` CUDA reference gate with token `324/324`.
- native manifest consumption: `runtime_fused_bench.exe --q4k-manifest ... --q4k-tensor ...`;
- native multi-tensor sequence: `runtime_q4k_sequence_bench.exe --q4k-manifest ... --limit 40`;
- latest manifest-backed harness: `E:\AI project\benchmarks\native_harness_1780126797.md`;
- latest full-manifest rows from the current harness keep the per-tensor manifest reference gates passing.
- latest full Q4_K sequence chained row: `40` tensors, `40/40` source tensor-plan checks, `40` observed layers, `80` kernels per graph replay (`40` Q4_K matvec plus `40` activation-feedback kernels), resident bytes `383549440`, allocated bytes `451747840`, setup H2D `383557636`, steady-state H2D `0`, D2H `64` bytes over `16` graph replays, `70.95` sequence steps/sec, `2838.17` tensor launches/sec, and p50/p95/p99 `1.935/70.376/70.376 ms`.
- latest full Q4_K sequence no-chain comparison: `40` tensors, `40/40` source tensor-plan checks, `40` kernels per graph replay, `475.20` sequence steps/sec, `19007.90` tensor launches/sec, and p50/p95/p99 `1.907/2.710/2.710 ms`.

IQ2_S raw resident and matvec probes:

- pack manifest: `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s\1780009251\manifest.md`;
- CUDA probe harness: `E:\AI project\benchmarks\native_harness_1780064700.md`;
- raw block-reader rows still validate payload checksums for `blk.0.attn_gate.weight` and `blk.0.ffn_gate_exps.weight`;
- `blk.0.attn_gate.weight` decode matvec: derived shape `4096x2048`, checked rows `64`, logical values/replay `131072`, max logit/KV error `1.19209e-07`, token `104/104`, `19323.70` graph replays/sec, timed D2H `32` bytes over `8` replays;
- `blk.0.ffn_gate_exps.weight` decode matvec: derived shape `131072x2048`, checked rows `8`, logical values/replay `16384`, max logit/KV error `9.31323e-09`, token `48/48`, `17746.20` graph replays/sec, timed D2H `32` bytes over `8` replays;
- IQ2_S sequence graph: `2` tensors, `2/2` source-plan checks, roles `attn_gate.weight` and `ffn_gate_exps.weight`, resident bytes `88670208`, logical values/replay `262144`, setup H2D `88678404`, steady-state H2D `0`, D2H `32`, chained `13934.90` sequence steps/sec with `4` kernels per graph replay;
- IQ3_S raw block probe: layer-0 `ffn_down_exps` and `ffn_down_shexp` resident pack `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s\1780009953\manifest.md`, `115793920` resident bytes, both raw CUDA Graph probes passed full payload and first-block `d/qs/qh/signs/scales` validation;
- IQ3_S dequant matvec: `ffn_down_exps` artifact `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_exps_1780057719.json`, checked `8` rows of shape `524288x512`, max logit/KV error `1.01281e-08`, token `56/56`, `21697.90` graph replays/sec; `ffn_down_shexp` artifact `E:\AI project\benchmarks\native_iq3s_matvec_probe_runtime_iq3s_decode_ffn_down_shexp_1780057719.json`, checked `64` rows of shape `2048x512`, max logit/KV error `1.19209e-07`, token `160/160`, `19445.80` graph replays/sec;
- mixed Q4_K/IQ2_S/IQ3_S/Q5_K layer-0 output graph: `12` tensors (`8` layer-0 IQ2_S tensors, `2` layer-0 IQ3_S down-projection tensors, `blk.0.attn_qkv.weight`, and `output.weight`), order policy `source_offset_with_global_tail`, `12/12` source-plan checks, resident bytes `656065536`, allocated bytes `665577216`, logical values/replay `18022400`, setup H2D `656106500`, steady-state H2D `0`, D2H `32`, phase counts `attention=2`, `ffn=6`, `ssm=3`, `output=1`, chained `2796.23` sequence steps/sec with `24` kernels per graph replay;
- layer-0 mixed graph CPU reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_chained_1780031980.json`, `reference_passed=true`, `stage_order_policy=source_offset_with_global_tail`, `layer_plan_passed=true`, `reference_stage_count=12`, `reference_stage_mismatches=0`, final stage `output.weight`, token `7671/7671`, max final-logit error `1.30385e-08`, max KV error `1.19209e-07`;
- layer-0 strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer0_iq3_output_reference_roleplan_1780057719.json`, `reference_passed=true`, `stage_order_policy=role_plan_with_global_tail`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `stage_role_plan_unknown_roles=0`, `stage_phase_unknown_roles=0`, `reference_stage_mismatches=0`, token `5199/5199`, max logit/KV error `7.689e-06/7.689e-06`, phase segments `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`, core execution phases `4/12/6/2` graph kernels, `layer_executor_workspace_plan=phase_local_workspace_descriptors`, `layer_executor_workspace_allocation_plan=phase_local_activation_logits_auxiliary`, `layer_executor_workspace_usage_plan=phase_local_logical_high_water`, `layer_executor_dataflow_plan=phase_local_activation_feedback_edges`, runtime-bound workspace bytes `74496` split into phase/auxiliary bytes `74496/0`, activation/logits bytes `40960/33536`, high-water bytes `74496`, slack bytes `0`, graph live device pressure `0`, admission required/usable bytes `665751040/11259609088`, resident/KV/workspace/DMA/graph bytes `656065536/8388608/74496/1048576/173824`, feedback edges/phase-crossing/wrap edges `12/4/1`, phase-aware FFN SiLU edges `0`, admission over-budget bytes `0`, workspace envelope `8192x4096`, logical values/replay `18022400`, named warmup/capture phase callbacks `1/1/1/1`, `2653.40` sequence steps/sec, and p50/p95/p99 `0.362/0.429/0.429 ms`;
- layer-1 IQ resident packs: IQ2_S `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq2_s_layer1\1780027201\manifest.md`, IQ3_S `E:\AI project\packs\iq\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\iq3_s_layer1\1780027201\manifest.md`;
- layer-1 mixed graph CPU reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_chained_1780031980.json`, `reference_passed=true`, `stage_order_policy=source_offset_with_global_tail`, `layer_plan_passed=true`, `reference_stage_count=12`, `reference_stage_mismatches=0`, final stage `output.weight`, token `4359/4359`, max final-logit error `1.02445e-08`, max KV error `6.14673e-08`, resident bytes `656065536`, steady-state H2D `0`;
- layer-1 strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_1780057719.json`, `reference_passed=true`, `stage_order_policy=role_plan_with_global_tail`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `stage_role_plan_unknown_roles=0`, `stage_phase_unknown_roles=0`, `reference_stage_mismatches=0`, token `8133/8133`, resident bytes `656065536`, setup-only H2D, max logit/KV error `7.09295e-06/7.09295e-06`, phase segments `attention:0+2`, `ffn:2+6`, `ssm:8+3`, `output:11+1`, core execution phases `4/12/6/2` graph kernels, `layer_executor_workspace_plan=phase_local_workspace_descriptors`, `layer_executor_workspace_allocation_plan=phase_local_activation_logits_auxiliary`, `layer_executor_workspace_usage_plan=phase_local_logical_high_water`, `layer_executor_dataflow_plan=phase_local_activation_feedback_edges`, runtime-bound workspace bytes `74496` split into phase/auxiliary bytes `74496/0`, activation/logits bytes `40960/33536`, high-water bytes `74496`, slack bytes `0`, graph live device pressure `0`, admission required/usable bytes `665751040/11259609088`, resident/KV/workspace/DMA/graph bytes `656065536/8388608/74496/1048576/173824`, feedback edges/phase-crossing/wrap edges `12/4/1`, phase-aware FFN SiLU edges `0`, admission over-budget bytes `0`, workspace envelope `8192x4096`, logical values/replay `18022400`, named warmup/capture phase callbacks `1/1/1/1`, `2763.96` sequence steps/sec, and p50/p95/p99 `0.353/0.382/0.382 ms`;
- layer-1 captured phase-scratch reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_scratch_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, scratch descriptors `attention_qkv_scratch:attention:8192` and `ssm_state_scratch:ssm:4096`, token `8133/8133`, max logit/KV error `7.09295e-06/7.09295e-06`, layer-executor phase/aux workspace bytes `74496/49152`, `phase_scratch_workspace_bytes=49152`, `phase_scratch_digest_enabled=true`, `phase_scratch_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, admission workspace/graph bytes `123648/182784`, `2704.71` sequence steps/sec, and p50/p95/p99 `0.363/0.393/0.393 ms`;
- layer-1 graph-bucket reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_graphbucket_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, token `8133/8133`, `graph_bucket_count=3`, `graph_bucket_estimated_bytes=523008`, `graph_bucket_max_estimated_bytes=174848`, bucket totals `173824/174336/174848`, `admission_graph_bytes=523008`, `actual_kernel_launches_per_graph=24`, `2714.44` sequence steps/sec, and p50/p95/p99 `0.364/0.377/0.377 ms`;
- layer-1 attention QKV scratch phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionphase_1780064700.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_scratch`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, token `8133/8133`, max logit/KV error `4.64916e-06/4.64916e-06`, admission workspace/graph bytes `172800/179456`, `2789.40` sequence steps/sec, and p50/p95/p99 `0.357/0.362/0.362 ms`;
- layer-1 attention QKV reduce phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionreduce_1780067869.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_reduce`, `attention_phase_qkv_reduce_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, token `8133/8133`, max logit/KV error `8.58307e-06/8.58307e-06`, admission workspace/graph bytes `172800/183552`, `2684.38` sequence steps/sec, and p50/p95/p99 `0.371/0.385/0.385 ms`;
- layer-1 attention QKV window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_window`, `attention_phase_qkv_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, token `8133/8133`, max logit/KV error `1.01924e-05/1.01924e-05`, admission workspace/graph bytes `172800/183552`, `2636.96` sequence steps/sec, and p50/p95/p99 `0.379/0.391/0.391 ms`;
- layer-1 attention QKV head-window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadwindow_1780069653.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_window`, `attention_phase_qkv_head_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/window `8/4`, token `8133/8133`, max logit/KV error `8.53837e-06/8.53837e-06`, admission workspace/graph bytes `172800/183552`, `2691.61` sequence steps/sec, and p50/p95/p99 `0.373/0.382/0.382 ms`;
- layer-1 attention QKV head-dim-window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheaddim_1780070384.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_dim_window`, `attention_phase_qkv_head_dim_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window `8/8/4`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, admission workspace/graph bytes `172800/183552`, `2562.30` sequence steps/sec, and p50/p95/p99 `0.381/0.408/0.408 ms`;
- layer-1 attention QKV head-tile-window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadtile_1780071919.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_tile_window`, `attention_phase_qkv_head_tile_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window `8/8/4`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, admission workspace/graph bytes `172800/183552`, `2699.06` sequence steps/sec, and p50/p95/p99 `0.363/0.391/0.391 ms`;
- layer-1 attention QKV head-group-window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadgroup_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_window`, `attention_phase_qkv_head_group_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, `attention_qkv_scratch_values=24576`, head/head-dim/window/contexts-per-block `8/8/4/4`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, admission workspace/graph bytes `172800/183552`, `2755.77` sequence steps/sec, and p50/p95/p99 `0.362/0.366/0.366 ms`;
- layer-1 attention QKV head-group-rope-window phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadrope_1780125807.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_rope_window`, `attention_phase_qkv_head_group_rope_window_enabled=true`, `attention_phase_kernel_launches_per_graph=2`, `actual_kernel_launches_per_graph=26`, `attention_qkv_scratch_workspace_bytes=98304`, head/head-dim/window/contexts-per-block `8/8/4/4`, RoPE theta scale `0.03125`, token `8133/8133`, max logit/KV error `7.51019e-06/7.51019e-06`, admission workspace/graph bytes `172800/183552`, `2760.33` sequence steps/sec, and p50/p95/p99 `0.361/0.367/0.367 ms`;
- layer-1 attention QKV head-group-fused phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_attentionheadfused_1780074053.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `attention_phase_mode=qkv_head_group_fused`, `attention_phase_qkv_head_group_fused_enabled=true`, `attention_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `attention_qkv_scratch_workspace_bytes=0`, head/head-dim/window/contexts-per-block `8/8/4/4`, token `8133/8133`, max logit/KV error `6.4373e-06/6.4373e-06`, admission workspace/graph bytes `74496/177920`, `1533.27` sequence steps/sec, and p50/p95/p99 `0.598/0.910/0.910 ms`;
- layer-1 SSM recurrent-state phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmphase_1780063102.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=recurrent_state`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, `ssm_recurrent_state_workspace_bytes=8192`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, admission workspace/graph bytes `82688/178176`, `2058.46` sequence steps/sec, and p50/p95/p99 `0.378/0.649/0.649 ms`;
- layer-1 SSM scan-scratch phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmscan_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=scan_scratch`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, recurrent/scan workspace bytes `8192/16384`, `ssm_scan_scratch_values=4096`, token `8133/8133`, max logit/KV error `4.88758e-06/4.88758e-06`, admission workspace/graph bytes `99072/178432`, `2743.30` sequence steps/sec, and p50/p95/p99 `0.364/0.370/0.370 ms`;
- layer-1 SSM selective-scan phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmselective_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=selective_scan`, `ssm_phase_selective_scan_enabled=true`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, same recurrent/scan workspace bytes `8192/16384`, token `8133/8133`, max logit/KV error `6.3777e-06/6.3777e-06`, admission workspace/graph bytes `99072/178432`, `2802.49` sequence steps/sec, and p50/p95/p99 `0.351/0.374/0.374 ms`;
- layer-1 SSM source-parameterized phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsource_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=source_parameterized`, `ssm_phase_source_parameterized_enabled=true`, `ssm_phase_kernel_launches_per_graph=3`, `actual_kernel_launches_per_graph=27`, recurrent/scan workspace bytes `8192/16384`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, admission workspace/graph bytes `99072/186624`, `2430.28` sequence steps/sec, and p50/p95/p99 `0.363/0.549/0.549 ms`;
- layer-1 SSM source-parameterized-fused phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ssmsourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ssm_phase_mode=source_parameterized_fused`, `ssm_phase_source_parameterized_fused_enabled=true`, `ssm_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, recurrent/scan workspace bytes `8192/16384`, token `8133/8133`, max logit/KV error `5.48363e-06/5.48363e-06`, admission workspace/graph bytes `99072/178432`, `2855.92` sequence steps/sec, and p50/p95/p99 `0.349/0.355/0.355 ms`;
- layer-1 output final-token phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_outputphase_1780063102.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `output_phase_mode=final_token`, `output_phase_kernel_launches_per_graph=1`, `actual_kernel_launches_per_graph=25`, token `63/63`, max logit/KV error `7.09295e-06/7.09295e-06`, admission graph bytes `177920`, `2815.91` sequence steps/sec, and p50/p95/p99 `0.354/0.356/0.356 ms`;
- layer-1 phase-aware FFN SiLU reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_phaseaware_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_silu`, `phase_aware_ffn_silu_feedback_edges=5`, token `8022/8022`, max logit/KV error `1.58548e-05/1.58548e-05`, `1974.53` sequence steps/sec, and p50/p95/p99 `0.377/0.797/0.797 ms`;
- layer-1 gated FFN SiLU reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_gated_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=phase_aware_ffn_gated_silu`, gate-cache/gated-combine edges `1/1`, FFN gate-cache workspace bytes `8192`, layer-executor auxiliary workspace bytes `8192`, token `8074/8074`, max logit/KV error `1.10567e-05/1.10567e-05`, `2838.69` sequence steps/sec, and p50/p95/p99 `0.348/0.358/0.358 ms`;
- layer-1 gated FFN phase primitive reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_ffnphase_1780057719.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `feedback_mode=rmsnorm`, `ffn_phase_mode=gated_silu`, gate/up/product phase edges `2/2/1`, FFN gate/up cache workspace bytes `8192/8192`, layer-executor phase/aux workspace bytes `74496/16384`, token `8119/8119`, max logit/KV error `7.83987e-06/7.83987e-06`, `2491.75` sequence steps/sec, and p50/p95/p99 `0.385/0.440/0.440 ms`;
- layer-1 integrated all-phase strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `attention_phase_qkv_head_group_window_enabled=true`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.20892e-06/9.20892e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2742.92` sequence steps/sec, p50/p95/p99 `0.364/0.366/0.366 ms`, and isolated phase p95 attention/FFN/SSM/output `0.254/0.308/0.107/0.052 ms`;
- layer-1 integrated all-phase RoPE strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_rope_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_rope_window`, `attention_phase_qkv_head_group_rope_window_enabled=true`, `ssm_phase_mode=scan_scratch`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.57027e-06/9.57027e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2674.87` sequence steps/sec, and p50/p95/p99 `0.370/0.383/0.383 ms`;
- layer-1 integrated all-phase selective-SSM strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_selective_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=selective_scan`, `ssm_phase_selective_scan_enabled=true`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.10461e-06/9.10461e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2429.99` sequence steps/sec, and p50/p95/p99 `0.364/0.540/0.540 ms`;
- layer-1 integrated all-phase source-parameterized SSM strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_source_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized`, `ssm_phase_source_parameterized_enabled=true`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/3/1`, `actual_kernel_launches_per_graph=30`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/200704`, `2714.81` sequence steps/sec, and p50/p95/p99 `0.363/0.374/0.374 ms`;
- layer-1 integrated all-phase source-parameterized-fused SSM strict role-plan reference gate: artifact `E:\AI project\benchmarks\native_mixed_q4k_iq_q5k_sequence_graph_runtime_mixed_layer1_iq3_output_reference_roleplan_allphase_sourcefused_1780126797.json`, `reference_passed=true`, `strict_layer_plan_passed=true`, `ffn_phase_mode=gated_silu`, `attention_phase_mode=qkv_head_group_window`, `ssm_phase_mode=source_parameterized_fused`, `ssm_phase_source_parameterized_fused_enabled=true`, `output_phase_mode=final_token`, token `62/62`, max logit/KV error `9.52184e-06/9.52184e-06`, phase kernels attention/SSM/output `2/1/1`, `actual_kernel_launches_per_graph=28`, auxiliary workspace count/bytes `5/139264`, admission workspace/graph bytes `213760/192512`, `2719.42` sequence steps/sec, p50/p95/p99 `0.366/0.374/0.374 ms`, and isolated phase p95 attention/FFN/SSM/output `0.302/0.453/0.208/0.046 ms`;
- interpretation: raw IQ2_S/IQ3_S block residency, CUDA-side block layout access, IQ2_S/IQ3_S dequantized matvec correctness, page-style KV writes, GPU token update, source-plan-validated IQ sequence scheduling, and mixed Q4_K/IQ2_S/IQ3_S/Q5_K graph capture/reference checking are now validated across two adjacent model layers. The layer role/phase contract is shared through `runtime/core/scheduler/layer_plan.*`, graph work accounting uses the core phase execution plan, phase-local workspace envelopes, aligned activation/logits slices, named auxiliary FFN cache slices, attention QKV scratch/reduce/window/head-window/head-dim-window/head-tile-window/head-group-window/head-group-rope-window slices, the direct fused head-group comparison path, SSM recurrent-state, scan-scratch, selective-scan, and source-parameterized/fused slices, runtime-bound phase/aux pointers, activation-feedback edge routing, high-water workspace accounting, and core CUDA graph bucket admission estimates are derived before strict admission; the mixed runner now also snapshots CUDA graph capture/instantiate/upload memory pressure. The default handoff is an RMSNorm-style CUDA primitive over the same dataflow edge plan, with the older synthetic transform kept only as an explicit comparison mode. The opt-in `phase-aware-ffn-silu` mode keeps RMSNorm outside FFN while applying SiLU to normalized logits across FFN-internal feedback edges. The opt-in `phase-aware-ffn-gated-silu` mode is the first gate/up/down-aware handoff. The newer `ffn_phase_mode=gated_silu` path keeps gate/up projections on the same FFN input, caches normalized `SiLU(gate)` and normalized `up` separately in core-owned `ffn_gate_cache`/`ffn_up_cache` auxiliary slices, and writes the product before down projection. The newer `attention_phase_mode=qkv_scratch` path keeps attention QKV scratch resident, launches a captured attention kernel, and updates the FFN activation before the FFN phase consumes it; `attention_phase_mode=qkv_reduce` consumes those Q/K/V scratch rows in a second captured kernel and applies a bounded weighted residual; `attention_phase_mode=qkv_window` consumes Q/K/V scratch rows in a second captured kernel with a small stable softmax window and V-context residual; `attention_phase_mode=qkv_head_window` keeps the same two-kernel envelope but constrains the softmax/V aggregation to contiguous per-head scratch blocks; `attention_phase_mode=qkv_head_dim_window` keeps the same envelope while adding explicit head dimensions and 8-lane Q/K dot-product scoring before V-lane aggregation; `attention_phase_mode=qkv_head_tile_window` computes the same score/weight window once per head/context tile; `attention_phase_mode=qkv_head_group_window` groups four query contexts per head block, preserving score sharing; `attention_phase_mode=qkv_head_group_rope_window` keeps that grouped scratch envelope and adds a RoPE-style Q/K pair rotation by query/key context before softmax, now validated both standalone and inside the integrated all-phase graph; `attention_phase_mode=qkv_head_group_fused` removes the QKV scratch auxiliary and collapses the comparison primitive into one captured kernel, but the latest measured row loses throughput and p50 latency, so the integrated row keeps grouped-scratch attention. The newer `ssm_phase_mode=recurrent_state` path keeps a phase-local `ssm_recurrent_state` buffer and updates output activation with `rtxllm_launch_ssm_recurrent_state`; `ssm_phase_mode=scan_scratch` adds `ssm_scan_scratch` and updates scan scratch, recurrent state, and output activation with `rtxllm_launch_ssm_scan_scratch`; `ssm_phase_mode=selective_scan` keeps the same recurrent/scan workspace envelope while adding sigmoid-gated candidate, scan, and state mixing with `rtxllm_launch_ssm_selective_scan`; `ssm_phase_mode=source_parameterized` reuses that envelope, caches alpha/beta source parameters in the scan scratch, and then updates scan scratch, recurrent state, and output activation with `rtxllm_launch_ssm_source_parameterized_scan`; `ssm_phase_mode=source_parameterized_fused` folds the alpha/beta cache writes into RMSNorm feedback with `rtxllm_launch_ssm_rmsnorm_feedback_parameter_cache`. The newer `output_phase_mode=final_token` path samples the final token from output logits on GPU inside graph replay. The integrated all-phase rows now prove gated FFN, grouped or RoPE grouped attention, SSM scan scratch, selective scan, or source-parameterized/fused scan, and output final-token can share one strict role-plan graph, five auxiliary slices, and one CPU reference path. The next bottleneck is reducing the remaining noisy all-phase tail latency now that fused source-parameter caching preserves the scan-scratch graph envelope.

Phase telemetry note: latest harness `1780126797` adds top-level `phase_timing_*`, `phase_timing_slowest_p95_*`, and `graph_minus_phase_timing_sum_*` JSON fields. The all-phase source-fused row reports graph p95 `0.374 ms` while isolated phase p95s are attention/FFN/SSM/output `0.302/0.453/0.208/0.046 ms`, so those phase values should guide hot-spot selection but should not be summed as graph critical-path time.

Q5_K raw resident and dequant probes:

- pack manifest: `E:\AI project\packs\q5k\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M\1780002833\manifest.md`;
- raw CUDA probe artifact: `E:\AI project\benchmarks\native_q5k_probe_runtime_q5k_probe_output_1780063102.json`;
- dequant matvec artifact: `E:\AI project\benchmarks\native_q5k_matvec_probe_runtime_q5k_decode_output_1780063102.json`;
- tensor: `output.weight`, dimensions `[2048, 248320]`, blocks `1986560`, resident bytes `349634560`;
- raw validation: `reference_passed=true`, GPU payload checksum `45716023722`, first-block `d`, `dmin`, scales, `qh`, `qs`, and block-count fields all matched CPU reference;
- dequant validation: checked `64` output rows and `131072` logical values per graph replay, max logit/KV error `7.15256e-07`, setup H2D `349642752`, steady-state H2D `0`, D2H `32`, `18294.10` graph replays/sec, and p50/p95/p99 `0.054/0.065/0.065 ms`;
- interpretation: the final target source type now has source-verified resident packing, native CUDA block-reader proof, a decode-style Q5_K dequantized matvec probe, and optional integration into the mixed output-slice graph. Remaining work is a real layer executor rather than standalone projection probes.

To regenerate the clean target report from an existing llama.cpp harness artifact without repeating the model load:

```powershell
& $env:RTXLLM_PYTHON .\tools\bench\target_quant_pipeline.py --llamacpp-harness-json "E:\AI project\benchmarks\llamacpp_harness_1779978087.json"
```
