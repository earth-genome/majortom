use geo::{Coord, LineString, Polygon};
use majortom::{
    decode_geohash as rs_decode_geohash, encode_geohash as rs_encode_geohash, GridError,
    MajorTomGrid as RustGrid,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn grid_error(err: GridError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn polygon_from_coords(coords: Vec<(f64, f64)>) -> PyResult<Polygon<f64>> {
    if coords.len() < 3 {
        return Err(PyValueError::new_err(
            "polygon exterior must have at least 3 coordinates",
        ));
    }
    let mut ring: Vec<Coord<f64>> = coords
        .into_iter()
        .map(|(x, y)| Coord { x, y })
        .collect();
    if let (Some(first), Some(last)) = (ring.first().copied(), ring.last().copied()) {
        if first != last {
            ring.push(first);
        }
    }
    Ok(Polygon::new(LineString::new(ring), vec![]))
}

/// Compact cell record: `(id, is_primary, min_lon, min_lat, max_lon, max_lat)`.
type CellTuple = (String, bool, f64, f64, f64, f64);

fn cell_tuple(cell: &majortom::GridCell) -> CellTuple {
    let (min_lon, min_lat, max_lon, max_lat) = cell.bbox();
    (
        cell.id().to_string(),
        cell.is_primary,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
}

#[pyclass(name = "MajorTomGrid", module = "majortom_eg._native")]
struct PyMajorTomGrid {
    inner: RustGrid,
    d: u64,
}

#[pymethods]
impl PyMajorTomGrid {
    #[new]
    #[pyo3(signature = (d = 320, overlap = true))]
    fn new(d: u64, overlap: bool) -> PyResult<Self> {
        let inner = RustGrid::new(d, overlap).map_err(grid_error)?;
        Ok(Self { inner, d })
    }

    #[getter]
    fn d(&self) -> u64 {
        self.d
    }

    #[getter]
    fn overlap(&self) -> bool {
        self.inner.overlap()
    }

    #[getter]
    fn row_count(&self) -> i64 {
        self.inner.row_count()
    }

    #[getter]
    fn lat_spacing(&self) -> f64 {
        self.inner.lat_spacing()
    }

    #[getter]
    fn lat_offset(&self) -> f64 {
        self.inner.lat_offset()
    }

    fn row_lat(&self, row_idx: i64) -> f64 {
        self.inner.row_lat(row_idx)
    }

    fn lon_spacing(&self, lat: f64) -> f64 {
        self.inner.lon_spacing(lat)
    }

    fn lon_offset(&self, lon_spacing: f64) -> f64 {
        self.inner.lon_offset(lon_spacing)
    }

    fn col_lon(&self, col_idx: i64, lon_spacing: f64, lon_offset: f64) -> f64 {
        self.inner.col_lon(col_idx, lon_spacing, lon_offset)
    }

    /// Generate intersecting cells from an exterior ring of `(lon, lat)` pairs.
    fn generate_grid_cells(&self, coords: Vec<(f64, f64)>) -> PyResult<Vec<CellTuple>> {
        let polygon = polygon_from_coords(coords)?;
        let cells = self.inner.generate_grid_cells(&polygon);
        Ok(cells.iter().map(cell_tuple).collect())
    }

    fn cell_from_id(&self, cell_id: &str) -> PyResult<CellTuple> {
        let cell = self.inner.cell_from_id(cell_id).map_err(grid_error)?;
        Ok(cell_tuple(&cell))
    }

    fn migrate_cell_id(&self, old_id: &str) -> PyResult<CellTuple> {
        let cell = self.inner.migrate_cell_id(old_id).map_err(grid_error)?;
        Ok(cell_tuple(&cell))
    }
}

#[pyfunction]
#[pyo3(signature = (lat, lon, precision = 11))]
fn encode_geohash(lat: f64, lon: f64, precision: usize) -> String {
    rs_encode_geohash(lat, lon, precision)
}

#[pyfunction]
fn decode_geohash(hash: &str) -> PyResult<(f64, f64)> {
    rs_decode_geohash(hash).map_err(grid_error)
}

#[pymodule]
#[pyo3(name = "_native")]
fn majortom_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMajorTomGrid>()?;
    m.add_function(wrap_pyfunction!(encode_geohash, m)?)?;
    m.add_function(wrap_pyfunction!(decode_geohash, m)?)?;
    Ok(())
}
