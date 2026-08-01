# majortom

[![PyPI - Version](https://img.shields.io/pypi/v/majortom_eg.svg)](https://pypi.org/project/majortom_eg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/majortom_eg.svg)](https://pypi.org/project/majortom_eg)

-----

An implementation of the ESA Major Tom Grid.

The hot path (`generate_grid_cells`, `cell_from_id`, `migrate_cell_id`) is
implemented in Rust via [`majortom-rs`](https://github.com/earth-genome/majortom-rs)
and exposed through a PyO3 extension (`majortom_eg._native`). The public Python
API (shapely polygons, generators) is unchanged.

## Installation

Pre-built wheels (manylinux / macOS / Windows):

```console
pip install majortom_eg
```

From a source checkout (requires a Rust toolchain). Clone or symlink
[`majortom-rs`](https://github.com/earth-genome/majortom-rs) to
`vendor/majortom-rs`:

```console
mkdir -p vendor
ln -sfn ../../majortom-rs vendor/majortom-rs   # sibling checkout under ../majortom-rs
uv sync
uv run maturin develop --release
```
## Usage

```python
import shapely.geometry
from shapely.io import to_geojson
from majortom_eg import MajorTomGrid, GridCell

# generate an overlapping grid with cells of 320m square
grid = MajorTomGrid(d=320, overlap=True)

# polygon 1/10 of a degree square
my_aoi = shapely.geometry.Polygon(((0., 0.), (0., 0.1), (0.1, 0.1), (0.1, 0.), (0., 0.)))

# iterate of cells returned from generator
for cell in grid.generate_grid_cells(my_aoi):
    # do something with cells
    print(f'cell id is {cell.id()}')
    print(f'cell geom is {to_geojson(cell.geom)}')
```

## Performance

Benchmarks in `benches/grid_bench.py` (same AOIs as `majortom-rs`), after
switching the core to Rust (shapely polygons are built lazily on `.geom` access):

| AOI | Cells (overlap) | Pure Python (before) | Rust-backed Python |
|-----|----------------:|---------------------:|-------------------:|
| Southampton | 225 | ~7.1 ms | **~0.15 ms** (~45×) |
| 1° × 1° Maryland | 187,626 | ~5.7 s | **~135 ms** (~40×) |
| `cell_from_id` | — | ~315 µs | **~0.5 µs** |

Raw Rust (`majortom-rs` criterion) is still faster (~12 ms for the 1° case) because
it never crosses the FFI / Python object boundary; the wrap is the right default
for Python callers.

```console
uv run python benches/grid_bench.py
```

## License

`majortom` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
