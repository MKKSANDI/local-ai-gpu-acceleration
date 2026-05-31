from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Page:
    id: int
    owner: str | None = None
    ref_count: int = 0


@dataclass
class PageArena:
    page_count: int
    pages: list[Page] = field(init=False)
    free_ids: list[int] = field(init=False)
    high_water: int = 0
    allocation_failures: int = 0

    def __post_init__(self) -> None:
        self.pages = [Page(id=index) for index in range(self.page_count)]
        self.free_ids = list(range(self.page_count - 1, -1, -1))

    @property
    def used_pages(self) -> int:
        return self.page_count - len(self.free_ids)

    @property
    def utilization(self) -> float:
        return self.used_pages / self.page_count if self.page_count else 0.0

    def reserve(self, count: int, owner: str) -> list[int] | None:
        if count > len(self.free_ids):
            self.allocation_failures += 1
            return None
        ids = [self.free_ids.pop() for _ in range(count)]
        for page_id in ids:
            page = self.pages[page_id]
            page.owner = owner
            page.ref_count = 1
        self.high_water = max(self.high_water, self.used_pages)
        return ids

    def retain(self, page_ids: list[int]) -> None:
        for page_id in page_ids:
            self.pages[page_id].ref_count += 1

    def release(self, page_ids: list[int]) -> None:
        for page_id in page_ids:
            page = self.pages[page_id]
            if page.ref_count <= 0:
                raise RuntimeError(f"release of unreferenced page {page_id}")
            page.ref_count -= 1
            if page.ref_count == 0:
                page.owner = None
                self.free_ids.append(page_id)


@dataclass
class PrefixEntry:
    key: str
    tokens: int
    pages: list[int]
    hits: int = 0
    last_used_step: int = 0


@dataclass
class ActiveSequence:
    id: str
    prefix_key: str
    private_pages: list[int]
    shared_pages: list[int]
    remaining_steps: int


def pages_for_tokens(tokens: int, page_tokens: int) -> int:
    return max(1, math.ceil(tokens / page_tokens))


def maybe_reclaim_prefixes(
    arena: PageArena,
    prefixes: dict[str, PrefixEntry],
    active: list[ActiveSequence],
    keep_prefixes: int,
) -> int:
    active_prefixes = {seq.prefix_key for seq in active}
    reclaimable = [
        entry
        for entry in prefixes.values()
        if entry.key not in active_prefixes
    ]
    reclaimable.sort(key=lambda entry: (entry.hits, entry.last_used_step))
    reclaimed = 0
    while len(prefixes) > keep_prefixes and reclaimable:
        entry = reclaimable.pop(0)
        arena.release(entry.pages)
        prefixes.pop(entry.key, None)
        reclaimed += 1
    return reclaimed


def simulate(
    *,
    page_count: int,
    page_tokens: int,
    requests: int,
    max_active: int,
    prefix_pool: int,
    keep_prefixes: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    arena = PageArena(page_count=page_count)
    prefixes: dict[str, PrefixEntry] = {}
    active: list[ActiveSequence] = []

    admitted = 0
    rejected = 0
    prefix_hits = 0
    prefix_misses = 0
    reclaimed_prefixes = 0
    completed = 0
    utilization_samples: list[float] = []

    for step in range(requests):
        still_active: list[ActiveSequence] = []
        for seq in active:
            seq.remaining_steps -= 1
            if seq.remaining_steps <= 0:
                arena.release(seq.private_pages)
                arena.release(seq.shared_pages)
                completed += 1
            else:
                still_active.append(seq)
        active = still_active

        reclaimed_prefixes += maybe_reclaim_prefixes(arena, prefixes, active, keep_prefixes)

        if len(active) >= max_active:
            utilization_samples.append(arena.utilization)
            continue

        prefix_key = f"prefix-{rng.randrange(prefix_pool)}"
        prompt_tokens = rng.randrange(page_tokens, page_tokens * 48)
        new_tokens = rng.randrange(page_tokens, page_tokens * 12)
        lifetime = rng.randrange(2, 12)

        if prefix_key in prefixes:
            prefix = prefixes[prefix_key]
            arena.retain(prefix.pages)
            prefix.hits += 1
            prefix.last_used_step = step
            shared_pages = list(prefix.pages)
            prefix_hits += 1
        else:
            prefix_pages_needed = pages_for_tokens(prompt_tokens, page_tokens)
            pages = arena.reserve(prefix_pages_needed, f"shared:{prefix_key}")
            if pages is None:
                rejected += 1
                utilization_samples.append(arena.utilization)
                continue
            prefix = PrefixEntry(
                key=prefix_key,
                tokens=prompt_tokens,
                pages=pages,
                last_used_step=step,
            )
            prefixes[prefix_key] = prefix
            arena.retain(prefix.pages)
            shared_pages = list(prefix.pages)
            prefix_misses += 1

        private_pages_needed = pages_for_tokens(new_tokens, page_tokens)
        private_pages = arena.reserve(private_pages_needed, f"seq:{step}")
        if private_pages is None:
            arena.release(shared_pages)
            rejected += 1
            utilization_samples.append(arena.utilization)
            continue

        active.append(
            ActiveSequence(
                id=f"seq-{step}",
                prefix_key=prefix_key,
                private_pages=private_pages,
                shared_pages=shared_pages,
                remaining_steps=lifetime,
            )
        )
        admitted += 1
        utilization_samples.append(arena.utilization)

    for seq in active:
        arena.release(seq.private_pages)
        arena.release(seq.shared_pages)
        completed += 1
    active.clear()

    utilization_mean = (
        sum(utilization_samples) / len(utilization_samples)
        if utilization_samples
        else 0.0
    )
    prefix_total = prefix_hits + prefix_misses
    hot_prefix_pages_after_drain = arena.used_pages
    result = {
        "path": "kv_cache_sim",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "page_count": page_count,
        "page_tokens": page_tokens,
        "requests": requests,
        "max_active": max_active,
        "prefix_pool": prefix_pool,
        "keep_prefixes": keep_prefixes,
        "admitted": admitted,
        "rejected": rejected,
        "completed": completed,
        "prefix_hits": prefix_hits,
        "prefix_misses": prefix_misses,
        "prefix_hit_rate": prefix_hits / prefix_total if prefix_total else 0.0,
        "prefixes_remaining": len(prefixes),
        "reclaimed_prefixes": reclaimed_prefixes,
        "allocation_failures": arena.allocation_failures,
        "high_water_pages": arena.high_water,
        "high_water_utilization": arena.high_water / page_count if page_count else 0.0,
        "mean_utilization": utilization_mean,
        "hot_prefix_pages_after_drain": hot_prefix_pages_after_drain,
        "used_pages_after_drain": hot_prefix_pages_after_drain,
    }
    return result


def write_reports(result: dict[str, object], out_json: Path, out_md: Path) -> None:
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# KV Cache Simulation",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        f"- Requests: `{result['requests']}`",
        f"- Admitted/rejected: `{result['admitted']}` / `{result['rejected']}`",
        f"- Prefix hit rate: `{result['prefix_hit_rate']:.3f}`",
        f"- High-water pages: `{result['high_water_pages']}` / `{result['page_count']}`",
        f"- High-water utilization: `{result['high_water_utilization']:.3f}`",
        f"- Mean utilization: `{result['mean_utilization']:.3f}`",
        f"- Allocation failures: `{result['allocation_failures']}`",
        f"- Reclaimed prefixes: `{result['reclaimed_prefixes']}`",
        f"- Retained hot-prefix pages after active drain: `{result['hot_prefix_pages_after_drain']}`",
        "",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    result = simulate(
        page_count=256,
        page_tokens=16,
        requests=80,
        max_active=8,
        prefix_pool=4,
        keep_prefixes=3,
        seed=7,
    )
    assert result["admitted"] > 0
    assert result["prefix_hits"] > 0
    assert result["used_pages_after_drain"] <= result["high_water_pages"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic paged KV cache simulation.")
    parser.add_argument("--storage-root", default=os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))
    parser.add_argument("--page-count", type=int, default=4096)
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--max-active", type=int, default=24)
    parser.add_argument("--prefix-pool", type=int, default=16)
    parser.add_argument("--keep-prefixes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("kv_cache_sim self-test passed")
        return 0

    result = simulate(
        page_count=args.page_count,
        page_tokens=args.page_tokens,
        requests=args.requests,
        max_active=args.max_active,
        prefix_pool=args.prefix_pool,
        keep_prefixes=args.keep_prefixes,
        seed=args.seed,
    )

    bench_dir = Path(args.storage_root) / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "kv_cache_sim.json"
    out_md = bench_dir / "kv_cache_sim.md"
    write_reports(result, out_json, out_md)
    print(json.dumps(result, indent=2))
    print(f"json={out_json}")
    print(f"markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
