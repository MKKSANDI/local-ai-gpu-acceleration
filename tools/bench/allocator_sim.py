from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ALIGNMENT = 256


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    mask = alignment - 1
    return (value + mask) & ~mask


@dataclass
class Allocation:
    id: str
    offset: int
    size: int
    lifetime: int | None


@dataclass
class Arena:
    name: str
    size: int
    bump_offset: int = 0
    active: dict[str, Allocation] = field(default_factory=dict)
    free_list: list[tuple[int, int]] = field(default_factory=list)
    high_water: int = 0
    failed_allocations: int = 0
    total_allocations: int = 0
    reuse_hits: int = 0

    def allocate(self, allocation_id: str, size: int, lifetime: int | None = None) -> Allocation | None:
        size = align_up(size)
        self.total_allocations += 1

        for index, (offset, free_size) in enumerate(self.free_list):
            aligned_offset = align_up(offset)
            padding = aligned_offset - offset
            if free_size >= padding + size:
                self.free_list.pop(index)
                tail_start = aligned_offset + size
                tail_size = (offset + free_size) - tail_start
                if tail_size > 0:
                    self.free_list.append((tail_start, tail_size))
                if padding > 0:
                    self.free_list.append((offset, padding))
                allocation = Allocation(allocation_id, aligned_offset, size, lifetime)
                self.active[allocation_id] = allocation
                self.high_water = max(self.high_water, self.used_bytes)
                self.reuse_hits += 1
                return allocation

        aligned = align_up(self.bump_offset)
        if aligned + size > self.size:
            self.failed_allocations += 1
            return None
        allocation = Allocation(allocation_id, aligned, size, lifetime)
        self.active[allocation_id] = allocation
        self.bump_offset = aligned + size
        self.high_water = max(self.high_water, self.used_bytes)
        return allocation

    def free(self, allocation_id: str) -> None:
        allocation = self.active.pop(allocation_id)
        self.free_list.append((allocation.offset, allocation.size))
        self.coalesce()

    def tick(self) -> None:
        expired = []
        for allocation in self.active.values():
            if allocation.lifetime is None:
                continue
            allocation.lifetime -= 1
            if allocation.lifetime <= 0:
                expired.append(allocation.id)
        for allocation_id in expired:
            self.free(allocation_id)

    @property
    def used_bytes(self) -> int:
        return sum(allocation.size for allocation in self.active.values())

    @property
    def reserved_bytes(self) -> int:
        return self.bump_offset

    @property
    def free_bytes(self) -> int:
        return self.size - self.used_bytes

    @property
    def largest_free_block(self) -> int:
        free_blocks = [size for _, size in self.free_list]
        tail = self.size - align_up(self.bump_offset)
        free_blocks.append(max(0, tail))
        return max(free_blocks) if free_blocks else 0

    @property
    def fragmentation(self) -> float:
        if self.free_bytes == 0:
            return 0.0
        return 1.0 - (self.largest_free_block / self.free_bytes)

    def coalesce(self) -> None:
        blocks = sorted(self.free_list)
        merged: list[tuple[int, int]] = []
        for offset, size in blocks:
            if not merged:
                merged.append((offset, size))
                continue
            prev_offset, prev_size = merged[-1]
            if prev_offset + prev_size == offset:
                merged[-1] = (prev_offset, prev_size + size)
            else:
                merged.append((offset, size))
        self.free_list = merged

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size_bytes": self.size,
            "used_bytes": self.used_bytes,
            "reserved_bytes": self.reserved_bytes,
            "high_water_bytes": self.high_water,
            "active_allocations": len(self.active),
            "free_blocks": len(self.free_list),
            "largest_free_block_bytes": self.largest_free_block,
            "fragmentation": self.fragmentation,
            "failed_allocations": self.failed_allocations,
            "total_allocations": self.total_allocations,
            "reuse_hits": self.reuse_hits,
        }


def mib(value: int | float) -> int:
    return int(value * 1024 * 1024)


def simulate(
    *,
    seed: int,
    steps: int,
    weight_mib: int,
    kv_mib: int,
    workspace_mib: int,
    dma_mib: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    arenas = {
        "weights": Arena("weights", mib(weight_mib)),
        "kv_pages": Arena("kv_pages", mib(kv_mib)),
        "workspace": Arena("workspace", mib(workspace_mib)),
        "dma": Arena("dma", mib(dma_mib)),
    }

    events: list[dict[str, object]] = []
    weights = arenas["weights"].allocate("model.weights", mib(weight_mib) - align_up(4096), lifetime=None)
    if weights is None:
        raise RuntimeError("immutable weight arena could not be initialized")

    rejected = 0
    for step in range(steps):
        for arena in arenas.values():
            arena.tick()

        kv_pages = rng.randrange(1, 10)
        kv_size = kv_pages * mib(1)
        if arenas["kv_pages"].allocate(f"seq.{step}.kv", kv_size, lifetime=rng.randrange(4, 18)) is None:
            rejected += 1
            events.append({"step": step, "arena": "kv_pages", "event": "reject", "bytes": kv_size})

        scratch_count = rng.randrange(1, 5)
        for index in range(scratch_count):
            scratch_size = rng.randrange(mib(1), mib(12))
            if arenas["workspace"].allocate(
                f"step.{step}.scratch.{index}", scratch_size, lifetime=1
            ) is None:
                rejected += 1
                events.append(
                    {"step": step, "arena": "workspace", "event": "reject", "bytes": scratch_size}
                )

        dma_size = rng.randrange(256 * 1024, mib(2))
        if arenas["dma"].allocate(f"step.{step}.dma", dma_size, lifetime=2) is None:
            rejected += 1
            events.append({"step": step, "arena": "dma", "event": "reject", "bytes": dma_size})

    for arena in arenas.values():
        for _ in range(32):
            arena.tick()

    snapshots = {name: arena.snapshot() for name, arena in arenas.items()}
    total_size = sum(arena.size for arena in arenas.values())
    total_high_water = sum(arena.high_water for arena in arenas.values())
    result = {
        "path": "allocator_sim",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "steps": steps,
        "alignment": ALIGNMENT,
        "rejected_allocations": rejected,
        "arena_count": len(arenas),
        "total_size_bytes": total_size,
        "total_high_water_bytes": total_high_water,
        "total_high_water_fraction": total_high_water / total_size if total_size else 0.0,
        "arenas": snapshots,
        "events_sample": events[:32],
    }
    return result


def write_reports(result: dict[str, object], out_json: Path, out_md: Path) -> None:
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Allocator Simulation",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        f"- Steps: `{result['steps']}`",
        f"- Rejected allocations: `{result['rejected_allocations']}`",
        f"- Total high-water fraction: `{result['total_high_water_fraction']:.3f}`",
        "",
        "| arena | size MiB | high-water MiB | failed | reuse hits | fragmentation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arena in result["arenas"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(arena["name"]),
                    f"{arena['size_bytes'] / (1024**2):.2f}",
                    f"{arena['high_water_bytes'] / (1024**2):.2f}",
                    str(arena["failed_allocations"]),
                    str(arena["reuse_hits"]),
                    f"{arena['fragmentation']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", f"JSON: `{out_json}`"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    arena = Arena("test", 1024)
    a = arena.allocate("a", 200, lifetime=None)
    b = arena.allocate("b", 200, lifetime=None)
    assert a is not None and b is not None
    assert a.offset % ALIGNMENT == 0
    arena.free("a")
    c = arena.allocate("c", 128, lifetime=None)
    assert c is not None
    assert c.offset == a.offset
    result = simulate(seed=1, steps=16, weight_mib=16, kv_mib=64, workspace_mib=32, dma_mib=8)
    assert result["arenas"]["weights"]["failed_allocations"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate fixed VRAM arena allocation behavior.")
    parser.add_argument("--storage-root", default=os.environ.get("RTXLLM_STORAGE_ROOT", r"E:\AI project"))
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--weight-mib", type=int, default=256)
    parser.add_argument("--kv-mib", type=int, default=512)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--dma-mib", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("allocator_sim self-test passed")
        return 0

    result = simulate(
        seed=args.seed,
        steps=args.steps,
        weight_mib=args.weight_mib,
        kv_mib=args.kv_mib,
        workspace_mib=args.workspace_mib,
        dma_mib=args.dma_mib,
    )
    bench_dir = Path(args.storage_root) / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "allocator_sim.json"
    out_md = bench_dir / "allocator_sim.md"
    write_reports(result, out_json, out_md)
    print(json.dumps(result, indent=2))
    print(f"json={out_json}")
    print(f"markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
