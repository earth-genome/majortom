# majortom

[![PyPI - Version](https://img.shields.io/pypi/v/majortom_eg.svg)](https://pypi.org/project/majortom_eg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/majortom_eg.svg)](https://pypi.org/project/majortom_eg)

-----
An implementation of the ESA Major Tom Grid 



## Installation

```console
pip install majortom_eg
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

## License

`majortom` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

