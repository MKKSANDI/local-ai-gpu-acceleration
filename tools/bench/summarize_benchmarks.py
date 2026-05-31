from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON files.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    rows = []
    for path in sorted(bench_dir.glob("*.json"), key=lambda item: item.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("magic") == "GGUF":
            continue
        if path.name.startswith("manifest_") or path.name == "model_plan.json":
            continue
        if "path" not in data:
            continue
        rows.append(
            # q4k_smoke is emitted by the native GGUF probe.
            # Keep these fields flat so the global report can show loader progress.
            {
                "file": path.name,
                "path": data.get("path"),
                "label": data.get("label", ""),
                "kernel_variant": data.get("kernel_variant", ""),
                "reference_passed": data.get("reference_passed", ""),
                "max_logit_abs_error": data.get("max_logit_abs_error"),
                "status": data.get("status", ""),
                "admitted": data.get("admitted", ""),
                "steps": data.get("steps", ""),
                "steps_per_second": data.get("steps_per_second"),
                "tensor_count": data.get("tensor_count"),
                "source_plan_validated": data.get("source_plan_validated"),
                "source_plan_tensors_checked": data.get("source_plan_tensors_checked"),
                "kernel_launches_per_graph": data.get("kernel_launches_per_graph"),
                "actual_kernel_launches_per_graph": data.get(
                    "actual_kernel_launches_per_graph"
                ),
                "attention_phase_kernel_launches_per_graph": data.get(
                    "attention_phase_kernel_launches_per_graph"
                ),
                "activation_feedbacks_per_graph": data.get("activation_feedbacks_per_graph"),
                "chain_activation": data.get("chain_activation"),
                "tensor_launches_per_second": data.get("tensor_launches_per_second"),
                "graph_kernel_launches_per_second": data.get("graph_kernel_launches_per_second"),
                "layer_count_observed": data.get("layer_count_observed"),
                "p50_ms": data.get("p50_ms"),
                "p95_ms": data.get("p95_ms"),
                "p99_ms": data.get("p99_ms"),
                "phase_timing_sum_p95_ms": data.get("phase_timing_sum_p95_ms"),
                "graph_minus_phase_timing_sum_p95_ms": data.get(
                    "graph_minus_phase_timing_sum_p95_ms"
                ),
                "phase_timing_slowest_p95_phase": data.get(
                    "phase_timing_slowest_p95_phase",
                    "",
                ),
                "phase_timing_slowest_p95_ms": data.get(
                    "phase_timing_slowest_p95_ms"
                ),
                "graph_replay_rate": data.get("graph_replay_rate"),
                "llama_decode_tps": llama_metric(data, "decode_tokens_per_second"),
                "llama_graphs_reused": llama_metric(data, "graphs_reused"),
                "kv_prefix_hit_rate": data.get("prefix_hit_rate"),
                "kv_high_water": data.get("high_water_utilization"),
                "allocator_high_water": data.get("total_high_water_fraction"),
                "allocator_rejects": data.get("rejected_allocations"),
                "valid_tensors": data.get("valid_tensors"),
                "invalid_tensors": data.get("invalid_tensors"),
                "q4k_smoke_valid": q4k_metric(data, "valid"),
                "q4k_dequant_valid": q4k_metric(data, "dequant_valid"),
                "q4k_packed_checksum": q4k_metric(data, "packed_q4_checksum"),
                "q4k_dequant_absmax": q4k_metric(data, "dequant_absmax"),
                "q4k_decoded_scale_checksum": q4k_metric(data, "decoded_scale_checksum"),
                "q4k_decoded_min_checksum": q4k_metric(data, "decoded_min_checksum"),
                "q4k_full_blocks_checksum": q4k_metric(data, "full_blocks_checksum"),
                "weight_source": data.get("weight_source"),
                "host_weight_checksum": data.get("host_weight_checksum"),
                "allocated_mib": data.get("allocated_bytes", 0) / (1024**2)
                if data.get("allocated_bytes")
                else None,
                "setup_h2d_mib": data.get("setup_h2d_bytes", 0) / (1024**2)
                if data.get("setup_h2d_bytes")
                else None,
                "resident_mib": data.get("resident_bytes", 0) / (1024**2)
                if data.get("resident_bytes")
                else None,
                "q4_values_per_step": data.get("q4_values_per_step"),
                "q4_packed_bytes_per_step": data.get("q4_packed_bytes_per_step"),
                "q4k_block_bytes_per_step": data.get("q4k_block_bytes_per_step"),
                "q4k_predecoded_meta_bytes": data.get("q4k_predecoded_meta_bytes"),
                "q4k_predecoded_meta_format": data.get("q4k_predecoded_meta_format", ""),
                "requested_mib": data.get("requested_bytes", 0) / (1024**2)
                if data.get("requested_bytes")
                else None,
                "admission_graph_bytes": data.get("admission_graph_bytes"),
                "attention_qkv_scratch_workspace_bytes": data.get(
                    "attention_qkv_scratch_workspace_bytes"
                ),
                "ssm_phase_kernel_launches_per_graph": data.get(
                    "ssm_phase_kernel_launches_per_graph"
                ),
                "output_phase_kernel_launches_per_graph": data.get(
                    "output_phase_kernel_launches_per_graph"
                ),
                "ssm_recurrent_state_workspace_bytes": data.get(
                    "ssm_recurrent_state_workspace_bytes"
                ),
                "ssm_scan_scratch_workspace_bytes": data.get(
                    "ssm_scan_scratch_workspace_bytes"
                ),
                "graph_bucket_count": data.get("graph_bucket_count"),
                "graph_bucket_estimated_bytes": data.get("graph_bucket_estimated_bytes"),
                "graph_live_device_pressure_bytes": data.get(
                    "graph_live_device_pressure_bytes"
                ),
                "d2h_bytes": data.get("d2h_bytes", ""),
            }
        )

    out_path = Path(args.output) if args.output else bench_dir / "benchmark_summary.md"
    header = "| file | path | label | variant | ref | max logit err | status | admitted | steps | steps/s | tensors | source plan | plan tensors | layers | graph kernels | actual graph kernels | attention kernels | ssm kernels | output kernels | feedbacks | chained | tensor launches/s | graph kernels/s | p50 ms | p95 ms | p99 ms | phase sum p95 ms | graph-phase p95 ms | slowest phase | slowest phase p95 ms | graph | llama decode tok/s | llama graphs | KV hit | KV high-water | alloc high-water | alloc rejects | valid tensors | invalid tensors | q4k smoke | q4k dequant | q4k checksum | q4k absmax | q4k scale sum | q4k min sum | q4k block checksum | weight source | weight checksum | allocated MiB | resident MiB | setup H2D MiB | Q4 values/step | Q4 packed B/step | Q4_K block B/step | Q4_K meta B | Q4_K meta fmt | requested MiB | admission graph B | attention qkv B | ssm state B | ssm scan B | graph bucket B | graph buckets | graph live B | D2H bytes |"
    lines = [
        "# Benchmark Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        header,
        "|" + "|".join("---" for _ in range(header.count("|") - 1)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["file"],
                    str(row["path"]),
                    str(row["label"]),
                    str(row["kernel_variant"]),
                    str(row["reference_passed"]),
                    fmt(row["max_logit_abs_error"], 6),
                    str(row["status"]),
                    str(row["admitted"]),
                    str(row["steps"]),
                    fmt(row["steps_per_second"], 2),
                    str(row["tensor_count"] if row["tensor_count"] is not None else ""),
                    str(
                        row["source_plan_validated"]
                        if row["source_plan_validated"] is not None
                        else ""
                    ),
                    str(
                        row["source_plan_tensors_checked"]
                        if row["source_plan_tensors_checked"] is not None
                        else ""
                    ),
                    str(
                        row["layer_count_observed"]
                        if row["layer_count_observed"] is not None
                        else ""
                    ),
                    str(
                        row["kernel_launches_per_graph"]
                        if row["kernel_launches_per_graph"] is not None
                        else ""
                    ),
                    str(
                        row["actual_kernel_launches_per_graph"]
                        if row["actual_kernel_launches_per_graph"] is not None
                        else ""
                    ),
                    str(
                        row["attention_phase_kernel_launches_per_graph"]
                        if row["attention_phase_kernel_launches_per_graph"] is not None
                        else ""
                    ),
                    str(
                        row["ssm_phase_kernel_launches_per_graph"]
                        if row["ssm_phase_kernel_launches_per_graph"] is not None
                        else ""
                    ),
                    str(
                        row["output_phase_kernel_launches_per_graph"]
                        if row["output_phase_kernel_launches_per_graph"] is not None
                        else ""
                    ),
                    str(
                        row["activation_feedbacks_per_graph"]
                        if row["activation_feedbacks_per_graph"] is not None
                        else ""
                    ),
                    str(row["chain_activation"] if row["chain_activation"] is not None else ""),
                    fmt(row["tensor_launches_per_second"], 2),
                    fmt(row["graph_kernel_launches_per_second"], 2),
                    fmt(row["p50_ms"]),
                    fmt(row["p95_ms"]),
                    fmt(row["p99_ms"]),
                    fmt(row["phase_timing_sum_p95_ms"]),
                    fmt(row["graph_minus_phase_timing_sum_p95_ms"]),
                    str(row["phase_timing_slowest_p95_phase"] or ""),
                    fmt(row["phase_timing_slowest_p95_ms"]),
                    fmt(row["graph_replay_rate"], 2),
                    fmt(row["llama_decode_tps"], 2),
                    str(row["llama_graphs_reused"] or ""),
                    fmt(row["kv_prefix_hit_rate"], 3),
                    fmt(row["kv_high_water"], 3),
                    fmt(row["allocator_high_water"], 3),
                    str(row["allocator_rejects"] if row["allocator_rejects"] is not None else ""),
                    str(row["valid_tensors"] if row["valid_tensors"] is not None else ""),
                    str(row["invalid_tensors"] if row["invalid_tensors"] is not None else ""),
                    str(row["q4k_smoke_valid"] if row["q4k_smoke_valid"] is not None else ""),
                    str(row["q4k_dequant_valid"] if row["q4k_dequant_valid"] is not None else ""),
                    str(row["q4k_packed_checksum"] if row["q4k_packed_checksum"] is not None else ""),
                    fmt(row["q4k_dequant_absmax"], 6),
                    str(
                        row["q4k_decoded_scale_checksum"]
                        if row["q4k_decoded_scale_checksum"] is not None
                        else ""
                    ),
                    str(
                        row["q4k_decoded_min_checksum"]
                        if row["q4k_decoded_min_checksum"] is not None
                        else ""
                    ),
                    str(
                        row["q4k_full_blocks_checksum"]
                        if row["q4k_full_blocks_checksum"] is not None
                        else ""
                    ),
                    str(row["weight_source"] or ""),
                    str(row["host_weight_checksum"] if row["host_weight_checksum"] is not None else ""),
                    fmt(row["allocated_mib"], 2),
                    fmt(row["resident_mib"], 2),
                    fmt(row["setup_h2d_mib"], 2),
                    str(row["q4_values_per_step"] or ""),
                    str(row["q4_packed_bytes_per_step"] or ""),
                    str(row["q4k_block_bytes_per_step"] or ""),
                    str(row["q4k_predecoded_meta_bytes"] or ""),
                    str(row["q4k_predecoded_meta_format"] or ""),
                    fmt(row["requested_mib"], 2),
                    str(
                        row["admission_graph_bytes"]
                        if row["admission_graph_bytes"] is not None
                        else ""
                    ),
                    str(
                        row["attention_qkv_scratch_workspace_bytes"]
                        if row["attention_qkv_scratch_workspace_bytes"] is not None
                        else ""
                    ),
                    str(
                        row["ssm_recurrent_state_workspace_bytes"]
                        if row["ssm_recurrent_state_workspace_bytes"] is not None
                        else ""
                    ),
                    str(
                        row["ssm_scan_scratch_workspace_bytes"]
                        if row["ssm_scan_scratch_workspace_bytes"] is not None
                        else ""
                    ),
                    str(
                        row["graph_bucket_estimated_bytes"]
                        if row["graph_bucket_estimated_bytes"] is not None
                        else ""
                    ),
                    str(
                        row["graph_bucket_count"]
                        if row["graph_bucket_count"] is not None
                        else ""
                    ),
                    str(
                        row["graph_live_device_pressure_bytes"]
                        if row["graph_live_device_pressure_bytes"] is not None
                        else ""
                    ),
                    str(row["d2h_bytes"]),
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    return 0


def llama_metric(data: dict[str, object], key: str):
    direct = data.get("llamacpp_metrics")
    if isinstance(direct, dict) and key in direct:
        return direct[key]
    run = data.get("llamacpp_run")
    if isinstance(run, dict):
        metrics = run.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            return metrics[key]
    runs = data.get("runs")
    if not isinstance(runs, list):
        return None
    for run in reversed(runs):
        if not isinstance(run, dict):
            continue
        metrics = run.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            return metrics[key]
    return None


def q4k_metric(data: dict[str, object], key: str):
    smoke = data.get("q4k_smoke")
    if isinstance(smoke, dict) and key in smoke:
        return smoke[key]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
