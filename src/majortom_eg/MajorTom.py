import numpy as np
import shapely.geometry
from shapely.geometry import Polygon
from geolib import geohash
from shapely.geometry.geo import box


class GridCell:

    def __init__(self, geom: shapely.geometry.Polygon):
        self.geom = geom

    def id(self) -> str:
        return geohash.encode(self.geom.centroid.y, self.geom.centroid.x, 11)


class MajorTomGrid:
    def __init__(self, d: int = 320, overlap=True):
        self.D = d  # grid spacing in meters
        self.earth_radius = 6378137  # Earth's radius in meters (WGS84 ellipsoid)
        self.overlap = overlap
        self.row_count = np.ceil(np.pi * self.earth_radius / self.D)
        self.lat_spacing = 180 / self.row_count

    def get_row_lat(self, row_idx):
        return -90 + row_idx * self.lat_spacing

    def get_lon_spacing(self, lat):
        lat_rad = np.radians(lat)
        circumference = 2 * np.pi * self.earth_radius * np.cos(lat_rad)
        n_cols = int(np.ceil(circumference / self.D))
        return 360 / n_cols

    def generate_grid_cells(self, polygon):
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        # use bounds for intersection detection for performance reasons
        bnds = box(*polygon.bounds)
        start_row = max(0, int((min_lat + 90) / self.lat_spacing))
        end_row = min(self.row_count, int((max_lat + 90) / self.lat_spacing) + 1)
        tiles = []

        for row_idx in range(start_row, end_row):
            lat = self.get_row_lat(row_idx)
            lon_spacing = self.get_lon_spacing(lat)
            half_lat_spacing = self.lat_spacing / 2
            half_lon_spacing = lon_spacing / 2

            start_col = max(0, int((min_lon + 180) / lon_spacing))
            end_col = min(int(360 / lon_spacing), int((max_lon + 180) / lon_spacing) + 1)

            for col_idx in range(start_col, end_col):
                lon = -180 + col_idx * lon_spacing
                # Create the primary grid cell polygon
                primary_cell_polygon = Polygon([
                    [lon, lat],
                    [lon + lon_spacing, lat],
                    [lon + lon_spacing, lat + self.lat_spacing],
                    [lon, lat + self.lat_spacing]
                ])
                if primary_cell_polygon.intersects(bnds):
                    yield GridCell(primary_cell_polygon)
                    # Create overlapping tiles if desired
                if self.overlap:
                    # East overlapping tile
                    east_overlap_cell = Polygon([
                        [lon + half_lon_spacing, lat],
                        [lon + lon_spacing + half_lon_spacing, lat],
                        [lon + lon_spacing + half_lon_spacing, lat + self.lat_spacing],
                        [lon + half_lon_spacing, lat + self.lat_spacing]
                    ])
                    if east_overlap_cell.intersects(bnds):
                        yield GridCell(east_overlap_cell)
                    # South overlapping tile
                    south_overlap_cell = Polygon([
                        [lon, lat - half_lat_spacing],
                        [lon + lon_spacing, lat - half_lat_spacing],
                        [lon + lon_spacing, lat + self.lat_spacing - half_lat_spacing],
                        [lon, lat + self.lat_spacing - half_lat_spacing]
                    ])
                    if south_overlap_cell.intersects(bnds):
                        yield GridCell(south_overlap_cell)

        return tiles

    def cell_from_id(self, cell_id:str) -> GridCell:
        if len(cell_id) > 11:
            cell_id = cell_id[:10]
        bounds = geohash.bounds(cell_id)
        p = box(bounds.sw[1],bounds.sw[0],bounds.ne[1],bounds.ne[0])
        candidates = self.generate_grid_cells(p)
        for candidate in candidates:
            if candidate.id() == cell_id:
                return candidate
        raise Exception(f"Can't find cell with id {cell_id}")