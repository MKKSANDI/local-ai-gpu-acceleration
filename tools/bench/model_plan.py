from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_MANIFEST = f"manifest_{DEFAULT_REPO_SAFE}.json"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def query_gpu() -> dict[str, object]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total_mib, free_mib, driver, compute_cap = [part.strip() for part in raw.split(",")]
    return {
        "name": name,
        "memory_total_bytes": int(total_mib) * 1024 * 1024,
        "memory_free_bytes": int(free_mib) * 1024 * 1024,
        "driver_version": driver,
        "compute_cap": compute_cap,
    }


def existing_model_path(model_dir: Path, filename: str) -> Path | None:
    direct = model_dir / filename
    if direct.exists():
        return direct
    matches = list(model_dir.rglob(filename))
    return matches[0] if matches else None


def mib(value: int | float) -> float:
    return value / (1024**2)


def gib(value: int | float) -> float:
    return value / (1024**3)


def classify(filename: str) -> str:
    stem = Path(filename).stem
    suffix = stem.split("-")[-1]
    if suffix in {"M", "P", "NL", "XS"} and len(stem.split("-")) >= 2:
        prev = stem.split("-")[-2]
        return f"{prev}_{suffix}"
    return suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan GGUF downloads and strict VRAM fit.")
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--wddm-guard-mib", type=int, default=512)
    parser.add_argument("--kv-reserve-mib", type=int, default=768)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--target", default="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf")
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    bench_dir = root / "benchmarks"
    model_dir = root / "models" / DEFAULT_REPO_SAFE
    manifest_path = Path(args.manifest) if args.manifest else bench_dir / DEFAULT_MANIFEST

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpu = query_gpu()
    free_disk_bytes = shutil.disk_usage(root).free

    usable_vram = max(
        0,
        int(gpu["memory_free_bytes"])
        - args.wddm_guard_mib * 1024 * 1024
        - args.kv_reserve_mib * 1024 * 1024
        - args.workspace_mib * 1024 * 1024,
    )

    rows = []
    for item in manifest["files"]:
        name = item["rfilename"]
        if not name.endswith(".gguf") or name.startswith("mmproj-"):
            continue
        size = int(item.get("size") or 0)
        local = existing_model_path(model_dir, name)
        strict_fit = size <= usable_vram
        disk_fit = size <= free_disk_bytes
        rows.append(
            {
                "filename": name,
                "quant": classify(name),
                "size_bytes": size,
                "size_gib": gib(size),
                "local": str(local) if local else None,
                "disk_fit_now": disk_fit,
                "strict_weight_fit_with_reserves": strict_fit,
                "sha256": (item.get("lfs") or {}).get("sha256"),
            }
        )

    rows.sort(key=lambda row: row["size_bytes"])
    target_row = next((row for row in rows if row["filename"] == args.target), None)
    plan = {
        "repo": manifest["repo"],
        "sha": manifest["sha"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(root),
        "model_dir": str(model_dir),
        "free_disk_bytes": free_disk_bytes,
        "free_disk_gib": gib(free_disk_bytes),
        "gpu": gpu,
        "wddm_guard_mib": args.wddm_guard_mib,
        "kv_reserve_mib": args.kv_reserve_mib,
        "workspace_mib": args.workspace_mib,
        "usable_vram_for_weights_bytes": usable_vram,
        "usable_vram_for_weights_gib": gib(usable_vram),
        "target": target_row,
        "models": rows,
        "recommendation": (
            "No target-repo GGUF fits strict full weight residency with current reserves; "
            "use partial offload or a smaller model family for fully resident tests."
            if not any(row["strict_weight_fit_with_reserves"] for row in rows)
            else "At least one target-repo GGUF fits strict weight residency with current reserves."
        ),
    }

    out_json = bench_dir / "model_plan.json"
    out_md = bench_dir / "model_plan.md"
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    lines = [
        "# Model Plan",
        "",
        f"Generated: `{plan['generated_at']}`",
        "",
        f"- Repo: `{plan['repo']}`",
        f"- Storage root: `{root}`",
        f"- Free disk: `{plan['free_disk_gib']:.2f} GiB`",
        f"- GPU: `{gpu['name']}`",
        f"- Free VRAM: `{gib(int(gpu['memory_free_bytes'])):.2f} GiB`",
        f"- Usable VRAM for weights after reserves: `{plan['usable_vram_for_weights_gib']:.2f} GiB`",
        f"- Recommendation: {plan['recommendation']}",
        "",
        "| quant | size GiB | disk fit | strict weight fit | local | file |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["quant"],
                    f"{row['size_gib']:.2f}",
                    str(row["disk_fit_now"]),
                    str(row["strict_weight_fit_with_reserves"]),
                    "yes" if row["local"] else "no",
                    row["filename"],
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"json={out_json}")
    print(f"markdown={out_md}")
    print(plan["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
