import numpy as np
from shapely.geometry import Polygon


class MajorTomGrid:
    def __init__(self, d: int = 320, overlap=True):
        self.D = d  # grid spacing in meters
        self.earth_radius = 6378137  # Earth's radius in meters (WGS84 ellipsoid)
        self.overlap = overlap

    def lat_spacing(self, n_rows):
        return 180 / n_rows

    def row_count(self):
        return int(np.ceil(np.pi * self.earth_radius / self.D))

    def get_row_lat(self, row_idx):
        n_rows = self.row_count()
        return -90 + row_idx * self.lat_spacing(n_rows)

    def get_lon_spacing(self, lat):
        lat_rad = np.radians(lat)
        circumference = 2 * np.pi * self.earth_radius * np.cos(lat_rad)
        n_cols = int(np.ceil(circumference / self.D))
        return 360 / n_cols

    def tile_polygon(self, polygon):
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        n_rows = self.row_count()
        start_row = max(0, int((min_lat + 90) / self.lat_spacing(n_rows)))
        end_row = min(n_rows, int((max_lat + 90) / self.lat_spacing(n_rows)) + 1)
        tiles = []

        for row_idx in range(start_row, end_row):
            lat = self.get_row_lat(row_idx)
            lon_spacing = self.get_lon_spacing(lat)
            half_lat_spacing = self.lat_spacing(n_rows) / 2
            half_lon_spacing = lon_spacing / 2

            start_col = max(0, int((min_lon + 180) / lon_spacing))
            end_col = min(int(360 / lon_spacing), int((max_lon + 180) / lon_spacing) + 1)

            for col_idx in range(start_col, end_col):
                lon = -180 + col_idx * lon_spacing
                # Create the primary grid cell polygon
                primary_cell_polygon = Polygon([
                    [lon, lat],
                    [lon + lon_spacing, lat],
                    [lon + lon_spacing, lat + self.lat_spacing(n_rows)],
                    [lon, lat + self.lat_spacing(n_rows)]
                ])
                if primary_cell_polygon.intersects(polygon):
                    yield primary_cell_polygon
                    # Create overlapping tiles if desired
                    if self.overlap:
                        # East overlapping tile
                        east_overlap_cell = Polygon([
                            [lon + half_lon_spacing, lat],
                            [lon + lon_spacing + half_lon_spacing, lat],
                            [lon + lon_spacing + half_lon_spacing, lat + self.lat_spacing(n_rows)],
                            [lon + half_lon_spacing, lat + self.lat_spacing(n_rows)]
                        ])
                        if east_overlap_cell.intersects(polygon):
                            yield east_overlap_cell
                        # South overlapping tile
                        south_overlap_cell = Polygon([
                            [lon, lat - half_lat_spacing],
                            [lon + lon_spacing, lat - half_lat_spacing],
                            [lon + lon_spacing, lat + self.lat_spacing(n_rows) - half_lat_spacing],
                            [lon, lat + self.lat_spacing(n_rows) - half_lat_spacing]
                        ])
                        if south_overlap_cell.intersects(polygon):
                            yield south_overlap_cell

        return tiles

    def count_polygon(self, polygon):
        count = 0
        for _ in self.tile_polygon(polygon):
            count += 1
        return count
