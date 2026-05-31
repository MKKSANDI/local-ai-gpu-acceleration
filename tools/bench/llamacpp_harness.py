from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def find_exe(root: Path, names: list[str]) -> Path | None:
    install_root = root / "third_party" / "llama.cpp"
    for name in names:
        matches = list(install_root.rglob(name))
        if matches:
            return matches[0]
    return None


def run_command(command: list[str], timeout: int) -> dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    result = {
        "command": command,
        "returncode": proc.returncode,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "output_tail": proc.stdout[-6000:],
    }
    metrics = parse_llamacpp_metrics(proc.stdout)
    if metrics:
        result["metrics"] = metrics
    return result


def parse_llamacpp_metrics(output: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    prompt = re.search(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens.*?([\d.]+) tokens per second", output)
    if prompt:
        metrics["prompt_tokens"] = int(prompt.group(1))
        metrics["prompt_tokens_per_second"] = float(prompt.group(2))

    decode = re.search(r"eval time =\s+[\d.]+ ms /\s+(\d+) runs\s+.*?([\d.]+) tokens per second", output)
    if decode:
        metrics["decode_runs"] = int(decode.group(1))
        metrics["decode_tokens_per_second"] = float(decode.group(2))

    graphs = re.search(r"graphs reused =\s+(\d+)", output)
    if graphs:
        metrics["graphs_reused"] = int(graphs.group(1))

    load_time = re.search(r"load time =\s+([\d.]+) ms", output)
    if load_time:
        metrics["load_time_ms"] = float(load_time.group(1))

    if "USE_GRAPHS = 1" in output:
        metrics["use_graphs"] = True
    elif "USE_GRAPHS = 0" in output:
        metrics["use_graphs"] = False

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="llama.cpp presence and benchmark harness.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--prompt", default="Write one sentence about GPU residency.")
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    llama_cli = find_exe(root, ["llama-cli.exe", "main.exe"])
    llama_completion = find_exe(root, ["llama-completion.exe"])
    llama_bench = find_exe(root, ["llama-bench.exe", "llama-batched-bench.exe"])
    model = Path(args.model) if args.model else None
    if model is None:
        model_dir = root / "models" / DEFAULT_REPO_SAFE
        local_ggufs = sorted(model_dir.glob("*.gguf"), key=lambda path: path.stat().st_size)
        model = local_ggufs[0] if local_ggufs else None

    result: dict[str, object] = {
        "path": "llamacpp_harness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(root),
        "llama_cli": str(llama_cli) if llama_cli else None,
        "llama_completion": str(llama_completion) if llama_completion else None,
        "llama_bench": str(llama_bench) if llama_bench else None,
        "model": str(model) if model else None,
        "status": "tool_missing"
        if llama_cli is None and llama_completion is None and llama_bench is None
        else "tool_present",
        "runs": [],
    }

    if args.dry_run:
        result["status"] = "dry_run"
    elif llama_cli is not None or llama_completion is not None:
        model_exe = llama_completion or llama_cli
        help_exe = llama_cli or llama_completion
        result["runs"].append(run_command([str(help_exe), "--help"], timeout=60))
        if model and model.exists():
            command = [
                str(model_exe),
                "-m",
                str(model),
                "-ngl",
                str(args.ngl),
                "-c",
                str(args.ctx_size),
                "-n",
                str(args.n_predict),
                "-p",
                args.prompt,
                "--no-display-prompt",
                "--no-conversation",
                "--single-turn",
                "--simple-io",
            ]
            model_run = run_command(command, timeout=args.timeout)
            result["runs"].append(model_run)
            result["status"] = (
                "model_run_complete" if model_run["returncode"] == 0 else "model_run_failed"
            )
        else:
            result["status"] = "model_missing"
    elif llama_bench is not None:
        result["runs"].append(run_command([str(llama_bench), "--help"], timeout=60))

    out_path = bench_dir / f"llamacpp_harness_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"json={out_path}")
    return 0 if result["status"] != "tool_missing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
