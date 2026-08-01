#!/usr/bin/env python3
"""Benchmarks for MajorTomGrid.generate_grid_cells.

Mirrors the AOIs used by majortom-rs/benches/grid_bench.rs so the two
implementations can be compared directly:

- bigSouthampton  — small rectangular AOI (~0.04° × 0.025°)
- large_maryland  — 1° × 1° box (exercises the heavy path)

Usage (from the repo root, with the package installed / editable):

    uv run python benches/grid_bench.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

from shapely.geometry import Polygon

from majortom_eg.MajorTom import MajorTomGrid


def big_southampton() -> Polygon:
    return Polygon(
        [
            (-76.35673421721803, 39.55614384974018),
            (-76.35673421721803, 39.53123810591927),
            (-76.3131967920373, 39.53123810591927),
            (-76.3131967920373, 39.55614384974018),
            (-76.35673421721803, 39.55614384974018),
        ]
    )


def large_maryland() -> Polygon:
    """~1° × 1° box — same footprint as the Rust large_* benches."""
    return Polygon(
        [
            (-77.5, 39.0),
            (-77.5, 40.0),
            (-76.5, 40.0),
            (-76.5, 39.0),
            (-77.5, 39.0),
        ]
    )


def _bench(name: str, fn: Callable[[], object], *, rounds: int, warmup: int) -> None:
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    result = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)

    count = len(result) if hasattr(result, "__len__") else "n/a"
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(
        f"{name:40s}  n={count:<8}  "
        f"mean={_fmt(mean):>10}  ±{_fmt(stdev):>9}  "
        f"(min={_fmt(min(samples))}, max={_fmt(max(samples))}, rounds={rounds})"
    )


def _fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.1f}µs"


def main() -> None:
    southampton = big_southampton()
    maryland = large_maryland()

    print("majortom_eg generate_grid_cells benchmarks")
    print("(AOIs match majortom-rs/benches/grid_bench.rs)\n")

    grid_overlap = MajorTomGrid(d=320, overlap=True)
    grid_no = MajorTomGrid(d=320, overlap=False)

    _bench(
        "generate_grid_cells_overlap",
        lambda: list(grid_overlap.generate_grid_cells(southampton)),
        rounds=50,
        warmup=5,
    )
    _bench(
        "generate_grid_cells_no_overlap",
        lambda: list(grid_no.generate_grid_cells(southampton)),
        rounds=50,
        warmup=5,
    )
    _bench(
        "generate_grid_cells_large_overlap",
        lambda: list(grid_overlap.generate_grid_cells(maryland)),
        rounds=5,
        warmup=1,
    )
    _bench(
        "generate_grid_cells_large_no_overlap",
        lambda: list(grid_no.generate_grid_cells(maryland)),
        rounds=5,
        warmup=1,
    )

    # cell_from_id — same id family as the Rust bench (Southampton AOI).
    cells = list(grid_overlap.generate_grid_cells(southampton))
    sample_id = cells[0].id()

    def lookup() -> object:
        return grid_overlap.cell_from_id(sample_id)

    _bench(f"cell_from_id ({sample_id})", lookup, rounds=5000, warmup=100)


if __name__ == "__main__":
    main()
