from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_FILE = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Hugging Face model file metadata.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--target-file", default=DEFAULT_FILE)
    parser.add_argument("--storage-root", default=None)
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    hf_home = root / "hf-cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    out_dir = root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)
    files = []
    target = None
    for sibling in info.siblings:
        item = {
            "rfilename": sibling.rfilename,
            "size": getattr(sibling, "size", None),
            "blob_id": getattr(sibling, "blob_id", None),
            "lfs": getattr(sibling, "lfs", None),
        }
        files.append(item)
        if sibling.rfilename == args.target_file:
            target = item

    manifest = {
        "repo": args.repo,
        "sha": info.sha,
        "private": info.private,
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags,
        "target_file": args.target_file,
        "target": target,
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    safe_repo = args.repo.replace("/", "__")
    out_path = out_dir / f"manifest_{safe_repo}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if target is None:
        print(f"target file not found: {args.target_file}")
        print(out_path)
        return 2

    size = target.get("size")
    gib = size / (1024**3) if size else None
    print(f"repo: {args.repo}")
    print(f"sha: {info.sha}")
    print(f"target: {args.target_file}")
    if gib is not None:
        print(f"target_size_gib: {gib:.2f}")
    print(f"manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
