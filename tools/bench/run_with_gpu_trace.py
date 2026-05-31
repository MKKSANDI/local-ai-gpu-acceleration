from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


GPU_QUERY = [
    "timestamp",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.free",
    "power.draw",
    "temperature.gpu",
]


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def sanitize_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "gpu_trace"


def strip_delimiter(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def to_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_nvidia_smi_row(row: str) -> dict[str, object]:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != len(GPU_QUERY):
        raise ValueError(f"expected {len(GPU_QUERY)} fields from nvidia-smi, got {len(parts)}")
    return {
        "timestamp": parts[0],
        "name": parts[1],
        "utilization_gpu_percent": to_int(parts[2]),
        "memory_used_mib": to_int(parts[3]),
        "memory_free_mib": to_int(parts[4]),
        "power_draw_w": to_float(parts[5]),
        "temperature_gpu_c": to_int(parts[6]),
    }


def sample_gpu() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY)}",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace")
    return [parse_nvidia_smi_row(line) for line in raw.splitlines() if line.strip()]


def summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    def numeric_values(key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), int | float)
        ]

    gpu_util = numeric_values("utilization_gpu_percent")
    memory_used = numeric_values("memory_used_mib")
    memory_free = numeric_values("memory_free_mib")
    power = numeric_values("power_draw_w")
    temp = numeric_values("temperature_gpu_c")
    names = sorted({str(sample.get("name")) for sample in samples if sample.get("name")})
    return {
        "sample_count": len(samples),
        "gpu_names": names,
        "avg_gpu_utilization_percent": sum(gpu_util) / len(gpu_util) if gpu_util else None,
        "max_gpu_utilization_percent": max(gpu_util) if gpu_util else None,
        "max_memory_used_mib": max(memory_used) if memory_used else None,
        "min_memory_free_mib": min(memory_free) if memory_free else None,
        "max_power_draw_w": max(power) if power else None,
        "max_temperature_gpu_c": max(temp) if temp else None,
    }


def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def tail(value: str, limit: int) -> str:
    return value[-limit:] if len(value) > limit else value


def run_traced(command: list[str], interval_s: float, timeout_s: int, output_tail: int) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    sample_errors: list[str] = []
    stop = threading.Event()

    def record_sample() -> None:
        try:
            samples.extend(sample_gpu())
        except Exception as exc:  # noqa: BLE001 - telemetry should not kill the workload.
            if len(sample_errors) < 8:
                sample_errors.append(str(exc))

    def sampler() -> None:
        while not stop.wait(interval_s):
            record_sample()

    record_sample()
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    thread = threading.Thread(target=sampler, name="gpu-trace-sampler", daemon=True)
    thread.start()

    timed_out = False
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        output, _ = proc.communicate()
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval_s * 2.0))
        record_sample()

    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "command": command,
        "command_line": command_line(command),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_ms": wall_ms,
        "sample_interval_s": interval_s,
        "samples": samples,
        "summary": summarize_samples(samples),
        "sample_errors": sample_errors,
        "output_tail": tail(output, output_tail),
    }


def write_markdown(path: Path, result: dict[str, object], json_path: Path) -> None:
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# GPU Trace",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Label: `{result['label']}`",
        f"Command: `{result['command_line']}`",
        f"JSON: `{json_path}`",
        "",
        "| return code | timed out | wall ms | samples | avg GPU % | max GPU % | max VRAM MiB | min free VRAM MiB | max power W | max temp C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                str(result.get("returncode")),
                str(result.get("timed_out")),
                f"{float(result.get('wall_ms', 0.0)):.3f}",
                str(summary.get("sample_count", "")),
                fmt(summary.get("avg_gpu_utilization_percent")),
                fmt(summary.get("max_gpu_utilization_percent")),
                fmt(summary.get("max_memory_used_mib")),
                fmt(summary.get("min_memory_free_mib")),
                fmt(summary.get("max_power_draw_w")),
                fmt(summary.get("max_temperature_gpu_c")),
            ]
        )
        + " |",
        "",
        "## Output Tail",
        "",
        "```text",
        str(result.get("output_tail", "")).rstrip(),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def self_test() -> int:
    rows = [
        "2026/05/28 22:14:05.123, NVIDIA GeForce RTX 3060, 9, 512, 11776, 42.50, 47",
        "2026/05/28 22:14:05.373, NVIDIA GeForce RTX 3060, N/A, 768, 11520, N/A, 48",
    ]
    samples = [parse_nvidia_smi_row(row) for row in rows]
    result = {
        "path": "gpu_trace_self_test",
        "samples": samples,
        "summary": summarize_samples(samples),
    }
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command while sampling nvidia-smi GPU telemetry."
    )
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-tail", type=int, default=8000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    command = strip_delimiter(args.command)
    if not command:
        parser.error("command is required; pass it after --")

    root = storage_root(args.storage_root)
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    label = sanitize_label(args.label or Path(command[0]).stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = trace_dir / f"{label}_{stamp}.json"
    md_path = trace_dir / f"{label}_{stamp}.md"

    traced = run_traced(command, interval_s=args.interval, timeout_s=args.timeout, output_tail=args.output_tail)
    result = {
        "path": "gpu_trace",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "storage_root": str(root),
        "json": str(json_path),
        "markdown": str(md_path),
        **traced,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(md_path, result, json_path=json_path)
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 1 if result["returncode"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
