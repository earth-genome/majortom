import numpy as np
import shapely.geometry
from shapely.geometry import Polygon
from geolib import geohash
from shapely.geometry.geo import box


class GridCell:

    def __init__(self, geom: shapely.geometry.Polygon, is_primary: bool = True):
        self.geom = geom
        self.is_primary = is_primary

    def id(self) -> str:
        return geohash.encode(self.geom.centroid.y, self.geom.centroid.x, 11)


class MajorTomGrid:
    def __init__(self, d: int = 320, overlap=True):
        if d <= 0:
            raise ValueError("Grid spacing must be positive")
        self.D = d
        self.earth_radius = 6378137
        self.overlap = overlap
        self.row_count = max(2, np.ceil(np.pi * self.earth_radius / self.D))
        self.lat_spacing = self.get_lat_spacing()  # Calculate lat_spacing once

    def get_lat_spacing(self):
        return min(180 / self.row_count, 89)

    def get_row_lat(self, row_idx):
        return -90 + row_idx * self.lat_spacing

    def get_lon_spacing(self, lat):
        lat_rad = np.radians(min(max(lat, -89), 89))
        circumference = 2 * np.pi * self.earth_radius * np.cos(lat_rad)
        n_cols = int(np.ceil(circumference / self.D))
        return 360 / max(n_cols, 1)

    def generate_grid_cells(self, polygon):
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        # Handle date line crossing
        if min_lon > max_lon:
            max_lon += 360

        # --- Key Change 1: More Precise Row/Col Ranges ---
        start_row = int(np.floor((min_lat + 90) / self.lat_spacing))
        end_row = int(np.ceil((max_lat + 90) / self.lat_spacing))
        
        # Adjust row range
        while self.get_row_lat(start_row) > min_lat + 1e-10:
            start_row -= 1
        while self.get_row_lat(end_row) < max_lat - 1e-10:
            end_row += 1

        for row_idx in range(start_row, end_row + 1):
            lat = self.get_row_lat(row_idx)
            lon_spacing = self.get_lon_spacing(lat)

            start_col = int(np.floor((min_lon + 180) / lon_spacing))
            end_col = int(np.ceil((max_lon + 180) / lon_spacing))
            
            # Adjust column range
            while -180 + start_col * lon_spacing > min_lon + 1e-10:
                start_col -= 1
            while -180 + end_col * lon_spacing < max_lon - 1e-10:
                end_col += 1

            for col_idx in range(start_col, end_col + 1):
                lon = -180 + col_idx * lon_spacing
                # Create the primary grid cell polygon
                primary_cell_polygon = Polygon([
                    [lon, lat],
                    [lon + lon_spacing, lat],
                    [lon + lon_spacing, lat + self.lat_spacing],
                    [lon, lat + self.lat_spacing]
                ])

                if self.overlap:
                    # Create overlapping cell with 50% overlap
                    overlap_lon = lon + lon_spacing/2
                    overlap_lat = lat + self.lat_spacing/2
                    overlap_cell_polygon = Polygon([
                        [overlap_lon, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat + self.lat_spacing],
                        [overlap_lon, overlap_lat + self.lat_spacing]
                    ])
                    
                    if primary_cell_polygon.intersects(polygon):
                        yield GridCell(primary_cell_polygon, is_primary=True)
                    if overlap_cell_polygon.intersects(polygon):
                        yield GridCell(overlap_cell_polygon, is_primary=False)
                else:
                    if primary_cell_polygon.intersects(polygon):
                        yield GridCell(primary_cell_polygon, is_primary=True)


    def cell_from_id(self, cell_id: str, buffer=False) -> GridCell:
        if len(cell_id)!= 11:
            raise ValueError("Cell ID must be exactly 11 characters")

        bounds = geohash.bounds(cell_id)
        p = box(bounds.sw[1],bounds.sw[0],bounds.ne[1],bounds.ne[0])

        if buffer:
            buffer_size = 0.0001 * self.D
            p = p.buffer(buffer_size)

        candidates = list(self.generate_grid_cells(p))
        for candidate in candidates:
            if candidate.id() == cell_id:
                return candidate

        if not buffer:
            return self.cell_from_id(cell_id, True)

        raise ValueError(f"No cell found with ID {cell_id}")