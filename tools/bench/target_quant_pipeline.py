from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench.gguf_tensor_plan import build_plan, query_gpu_free_bytes  # noqa: E402


DEFAULT_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def mib(value: int | float) -> float:
    return value / (1024**2)


def gib(value: int | float) -> float:
    return value / (1024**3)


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-") or "target"


def select_model(
    models: list[dict[str, object]],
    *,
    quant: str | None,
    filename: str | None,
) -> dict[str, object]:
    ggufs = [row for row in models if str(row.get("filename", "")).endswith(".gguf")]
    if filename:
        for row in ggufs:
            if row.get("filename") == filename:
                return row
        raise ValueError(f"filename not found in model plan: {filename}")
    if quant:
        for row in ggufs:
            if row.get("quant") == quant:
                return row
        raise ValueError(f"quant not found in model plan: {quant}")
    if not ggufs:
        raise ValueError("model plan contains no GGUF rows")
    return sorted(ggufs, key=lambda row: int(row["size_bytes"]))[0]


def local_model_path(root: Path, filename: str) -> Path:
    return root / "models" / DEFAULT_REPO_SAFE / filename


def quote(value: str) -> str:
    return f'"{value}"'


def command_text(value: object) -> str:
    if isinstance(value, list):
        return subprocess.list2cmdline([str(part) for part in value])
    return str(value)


def llama_command(root: Path, model_path: Path, ngl: int, ctx_size: int, n_predict: int) -> str:
    exe = root / "third_party" / "llama.cpp" / "llama-completion.exe"
    return " ".join(
        [
            quote(str(exe)),
            "-m",
            quote(str(model_path)),
            "-p",
            quote("Write one sentence about GPU residency."),
            "-n",
            str(n_predict),
            "-c",
            str(ctx_size),
            "-ngl",
            str(ngl),
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--no-display-prompt",
            "--single-turn",
            "--simple-io",
            "-fit",
            "on",
            "-fit-target",
            "1024",
        ]
    )


def download_model(repo: str, filename: str, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hf-cache" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(root / "hf-cache" / "xet"))
    local_dir = root / "models" / DEFAULT_REPO_SAFE
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=local_dir,
            resume_download=True,
        )
    )


def run_llamacpp_harness(
    root: Path,
    model_path: Path,
    *,
    ngl: int,
    ctx_size: int,
    n_predict: int,
    timeout: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        "tools/bench/llamacpp_harness.py",
        "--storage-root",
        str(root),
        "--model",
        str(model_path),
        "--ngl",
        str(ngl),
        "--ctx-size",
        str(ctx_size),
        "--n-predict",
        str(n_predict),
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout + 60,
    )
    result: dict[str, object] = {
        "command": command,
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-4000:],
    }
    parsed = parse_harness_object(proc.stdout)
    if parsed is not None:
        result["harness"] = parsed
        result["metrics"] = llama_metric_summary(parsed)
        model_run = llama_model_run(parsed)
        if model_run is not None:
            result["model_command"] = model_run.get("command")
            result["model_returncode"] = model_run.get("returncode")
            result["model_wall_ms"] = model_run.get("wall_ms")
            result["output_tail"] = str(model_run.get("output_tail", ""))[-4000:]
    return result


def iter_json_objects(output: str):
    decoder = json.JSONDecoder()
    start = output.find("{")
    while start >= 0:
        try:
            parsed, end = decoder.raw_decode(output[start:])
        except json.JSONDecodeError:
            start = output.find("{", start + 1)
            continue
        if isinstance(parsed, dict):
            yield parsed
        start = output.find("{", start + max(end, 1))


def parse_harness_object(output: str) -> dict[str, object] | None:
    for parsed in iter_json_objects(output):
        if parsed.get("path") == "llamacpp_harness" and isinstance(parsed.get("runs"), list):
            return parsed
    return None


def llama_model_run(harness: dict[str, object]) -> dict[str, object] | None:
    runs = harness.get("runs")
    if not isinstance(runs, list):
        return None
    for run in reversed(runs):
        if isinstance(run, dict) and isinstance(run.get("metrics"), dict):
            return run
    for run in reversed(runs):
        if not isinstance(run, dict):
            continue
        command = run.get("command")
        if isinstance(command, list) and any("llama-completion" in str(part) for part in command):
            return run
    return None


def llama_metric_summary(harness: dict[str, object]) -> dict[str, object]:
    run = llama_model_run(harness)
    if run is None:
        return {}
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        "status": harness.get("status"),
        "returncode": run.get("returncode"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "prompt_tokens_per_second": metrics.get("prompt_tokens_per_second"),
        "decode_runs": metrics.get("decode_runs"),
        "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
        "graphs_reused": metrics.get("graphs_reused"),
        "load_time_ms": metrics.get("load_time_ms"),
        "use_graphs": metrics.get("use_graphs"),
        "wall_ms": run.get("wall_ms"),
    }


def resolve_ngl(tensor_plan: dict[str, object] | None, override: int | None) -> tuple[int, str]:
    if override is not None:
        if override < 0:
            raise ValueError("--ngl must be non-negative")
        return override, "override"
    if isinstance(tensor_plan, dict):
        return int(tensor_plan["resident_layers_estimate"]), "tensor_plan"
    return 0, "none"


def load_llamacpp_harness_artifact(path: Path) -> dict[str, object]:
    harness = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(harness, dict):
        raise ValueError(f"not a llama.cpp harness JSON object: {path}")
    model_run = llama_model_run(harness)
    metrics = llama_metric_summary(harness)
    result: dict[str, object] = {
        "source_json": str(path),
        "returncode": model_run.get("returncode") if model_run else None,
        "model_command": model_run.get("command") if model_run else None,
        "model_returncode": model_run.get("returncode") if model_run else None,
        "model_wall_ms": model_run.get("wall_ms") if model_run else None,
        "output_tail": str(model_run.get("output_tail", ""))[-4000:] if model_run else "",
        "harness": harness,
        "metrics": metrics,
    }
    return result


def write_markdown(path: Path, result: dict[str, object]) -> None:
    selected = result["selected"]
    lines = [
        "# Target Quant Pipeline",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        f"- Repo: `{result['repo']}`",
        f"- Selected quant: `{selected['quant']}`",
        f"- Selected file: `{selected['filename']}`",
        f"- Size: `{selected['size_gib']:.2f} GiB`",
        f"- Local path: `{result['local_path']}`",
        f"- Local exists: `{result['local_exists']}`",
        f"- Download attempted: `{result['download_attempted']}`",
        f"- Downloaded: `{result['downloaded']}`",
        f"- Status: `{result['status']}`",
        f"- Planned NGL: `{result['planned_ngl']}`",
        f"- NGL source: `{result['ngl_source']}`",
        "",
    ]
    tensor_plan = result.get("tensor_plan")
    if isinstance(tensor_plan, dict):
        lines.extend(
            [
                "## Tensor Plan",
                "",
                f"- Strict full resident: `{tensor_plan['strict_full_resident']}`",
                f"- Layer count: `{tensor_plan['layer_count']}`",
                f"- Resident layers estimate: `{tensor_plan['resident_layers_estimate']}`",
                f"- Tensor-span residency estimate: `{tensor_plan['gpu_residency_fraction_estimate']:.3f}`",
                f"- Tensor span: `{tensor_plan['tensor_span_mib']:.2f} MiB`",
                "",
            ]
        )
    if result.get("llama_command"):
        lines.extend(["## Planned llama.cpp Command", "", f"```powershell\n{result['llama_command']}\n```", ""])
    if result.get("llamacpp_run"):
        run = result["llamacpp_run"]
        if not isinstance(run, dict):
            run = {}
        metrics = run.get("metrics")
        lines.extend(["## llama.cpp Run", ""])
        if run.get("source_json"):
            lines.append(f"- Harness JSON: `{run.get('source_json')}`")
        lines.append(f"- Return code: `{run.get('returncode')}`")
        if isinstance(metrics, dict) and metrics:
            lines.extend(
                [
                    f"- Decode tokens/sec: `{metrics.get('decode_tokens_per_second')}`",
                    f"- Decode runs: `{metrics.get('decode_runs')}`",
                    f"- Prompt tokens/sec: `{metrics.get('prompt_tokens_per_second')}`",
                    f"- Prompt tokens: `{metrics.get('prompt_tokens')}`",
                    f"- Graphs reused: `{metrics.get('graphs_reused')}`",
                    f"- Load time ms: `{metrics.get('load_time_ms')}`",
                    f"- USE_GRAPHS: `{metrics.get('use_graphs')}`",
                    f"- Harness wall ms: `{metrics.get('wall_ms')}`",
                ]
            )
        if run.get("model_command"):
            lines.extend(
                [
                    "",
                    "### Actual Model Command",
                    "",
                    f"```powershell\n{command_text(run.get('model_command'))}\n```",
                ]
            )
        lines.extend(
            [
                "",
                "```text",
                str(run.get("output_tail", "")).rstrip(),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select/download/plan/run a target-repo GGUF quant.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--model-plan", default=None)
    parser.add_argument("--quant", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run-llamacpp", action="store_true")
    parser.add_argument("--llamacpp-harness-json", default=None)
    parser.add_argument("--ngl", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--kv-reserve-mib", type=int, default=768)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.model_plan) if args.model_plan else bench_dir / "model_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = select_model(plan["models"], quant=args.quant, filename=args.filename)
    filename = str(selected["filename"])
    local_path = local_model_path(root, filename)

    status = "local_ready" if local_path.exists() else "local_missing"
    downloaded = False
    if args.download and not local_path.exists():
        downloaded_path = download_model(str(plan.get("repo", DEFAULT_REPO)), filename, root)
        downloaded = downloaded_path.exists()
        local_path = downloaded_path
        status = "downloaded" if downloaded else "download_failed"

    tensor_plan = None
    if local_path.exists():
        free_vram = query_gpu_free_bytes()
        if free_vram is not None:
            tensor_plan = build_plan(
                local_path,
                free_vram_bytes=free_vram,
                wddm_guard_mib=args.wddm_guard_mib,
                kv_reserve_mib=args.kv_reserve_mib,
                workspace_mib=args.workspace_mib,
                safety_margin_mib=args.safety_margin_mib,
            )
        status = "planned"

    ngl, ngl_source = resolve_ngl(tensor_plan, args.ngl)
    command = llama_command(root, local_path, ngl=ngl, ctx_size=args.ctx_size, n_predict=args.n_predict)
    llamacpp_run = None
    if args.run_llamacpp and local_path.exists():
        llamacpp_run = run_llamacpp_harness(
            root,
            local_path,
            ngl=ngl,
            ctx_size=args.ctx_size,
            n_predict=args.n_predict,
            timeout=args.timeout,
        )
        status = "llamacpp_run_complete" if llamacpp_run["returncode"] == 0 else "llamacpp_run_failed"
    elif args.llamacpp_harness_json:
        llamacpp_run = load_llamacpp_harness_artifact(Path(args.llamacpp_harness_json))
        status = "llamacpp_metrics_imported" if llamacpp_run.get("metrics") else "llamacpp_imported"

    result: dict[str, object] = {
        "path": "target_quant_pipeline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": plan.get("repo", DEFAULT_REPO),
        "source_plan": str(plan_path),
        "selected": selected,
        "storage_root": str(root),
        "local_path": str(local_path),
        "local_exists": local_path.exists(),
        "download_attempted": args.download,
        "downloaded": downloaded,
        "status": status,
        "planned_ngl": ngl,
        "ngl_source": ngl_source,
        "llama_command": command,
        "tensor_plan": tensor_plan,
        "llamacpp_run": llamacpp_run,
        "llamacpp_metrics": llamacpp_run.get("metrics") if isinstance(llamacpp_run, dict) else None,
    }
    safe = sanitize(str(selected["quant"]))
    out_json = Path(args.output_json) if args.output_json else bench_dir / f"target_quant_pipeline_{safe}.json"
    out_md = Path(args.output_md) if args.output_md else bench_dir / f"target_quant_pipeline_{safe}.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(f"json={out_json}")
    print(f"markdown={out_md}")
    print(f"status={status}")
    print(f"selected={selected['filename']}")
    return 0 if status not in {"download_failed", "llamacpp_run_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
