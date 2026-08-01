"""Python facade over the Rust Major TOM grid implementation."""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

try:
    from majortom_eg import _native
except ImportError as exc:  # pragma: no cover - install/dev error path
    raise ImportError(
        "majortom_eg requires its Rust extension module (_native). "
        "Install a published wheel, or build from source with "
        "`maturin develop` / `pip install .` (Rust toolchain required)."
    ) from exc

EARTH_RADIUS = 6378137


def _exterior_coords(geometry: BaseGeometry) -> List[Tuple[float, float]]:
    """Extract a closed exterior ring as `(lon, lat)` pairs for the Rust core."""
    if geometry.geom_type == "Polygon":
        poly = geometry
    elif geometry.geom_type == "MultiPolygon":
        if len(geometry.geoms) != 1:
            raise TypeError(
                "generate_grid_cells expects a single Polygon "
                f"(got MultiPolygon with {len(geometry.geoms)} parts)"
            )
        poly = geometry.geoms[0]
    else:
        raise TypeError(
            f"unsupported geometry type for generate_grid_cells: {geometry.geom_type}"
        )
    return [(float(x), float(y)) for x, y in poly.exterior.coords]


def _polygon_from_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Polygon:
    return Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )


def _cell_from_tuple(record: Tuple[str, bool, float, float, float, float]) -> "GridCell":
    cell_id, is_primary, min_lon, min_lat, max_lon, max_lat = record
    return GridCell.from_native(cell_id, is_primary, min_lon, min_lat, max_lon, max_lat)


class GridCell:
    """A single Major TOM grid cell (shapely polygon + geohash id)."""

    __slots__ = ("_geom", "_id", "is_primary", "_bbox")

    def __init__(
        self,
        geom: Polygon,
        is_primary: bool = True,
        _id: Optional[str] = None,
    ):
        self._geom: Optional[Polygon] = geom
        self.is_primary = is_primary
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        if _id is not None:
            self._id = _id
        else:
            centroid = geom.centroid
            self._id = _native.encode_geohash(centroid.y, centroid.x, 11)

    @classmethod
    def from_native(
        cls,
        cell_id: str,
        is_primary: bool,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> "GridCell":
        """Build a cell from the Rust FFI record without constructing shapely yet."""
        cell = object.__new__(cls)
        cell._geom = None
        cell._id = cell_id
        cell.is_primary = is_primary
        cell._bbox = (min_lon, min_lat, max_lon, max_lat)
        return cell

    @property
    def geom(self) -> Polygon:
        if self._geom is None:
            assert self._bbox is not None
            self._geom = _polygon_from_bbox(*self._bbox)
        return self._geom

    @geom.setter
    def geom(self, value: Polygon) -> None:
        self._geom = value
        self._bbox = None

    def id(self) -> str:
        return self._id


class MajorTomGrid:
    """Equal-area Major TOM grid backed by the Rust `majortom` crate."""

    def __init__(self, d: int = 320, overlap: bool = True):
        if d <= 0:
            raise ValueError("Grid spacing must be positive")
        self._native = _native.MajorTomGrid(d, overlap)
        self.D = d
        self.earth_radius = EARTH_RADIUS
        self.overlap = overlap
        self.row_count = self._native.row_count
        self.lat_spacing = self._native.lat_spacing
        self._lat_offset = self._native.lat_offset

    def get_lat_spacing(self) -> float:
        return self.lat_spacing

    def get_row_lat(self, row_idx: int) -> float:
        return self._native.row_lat(int(row_idx))

    def get_lon_spacing(self, lat: float) -> float:
        return self._native.lon_spacing(float(lat))

    def get_lon_offset(self, lon_spacing: float) -> float:
        return self._native.lon_offset(float(lon_spacing))

    def get_col_lon(self, col_idx: int, lon_spacing: float, lon_offset: float) -> float:
        return self._native.col_lon(int(col_idx), float(lon_spacing), float(lon_offset))

    def generate_grid_cells(self, polygon: BaseGeometry) -> Iterator[GridCell]:
        coords = _exterior_coords(polygon)
        for record in self._native.generate_grid_cells(coords):
            yield _cell_from_tuple(record)

    def cell_from_id(self, cell_id: str) -> GridCell:
        return _cell_from_tuple(self._native.cell_from_id(cell_id))

    def migrate_cell_id(self, old_id: str) -> GridCell:
        """Map a cell ID from a prior grid version to the current grid."""
        return _cell_from_tuple(self._native.migrate_cell_id(old_id))
