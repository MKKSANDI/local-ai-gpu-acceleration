from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_REPO_SAFE = "HauhauCS__Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"


def storage_root(value: str | None) -> Path:
    return Path(value or os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one model file to external storage.")
    parser.add_argument("filename")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = storage_root(args.storage_root)
    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hf-cache" / "hub"))

    local_dir = root / "models" / DEFAULT_REPO_SAFE
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"repo={args.repo}")
        print(f"filename={args.filename}")
        print(f"local_dir={local_dir}")
        print("dry_run=true")
        return 0

    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=local_dir,
        resume_download=True,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
