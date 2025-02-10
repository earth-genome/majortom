from shapely.geometry import box
import geopandas as gpd
import matplotlib.pyplot as plt
from majortom_eg.MajorTom import MajorTomGrid
import numpy as np
from collections import defaultdict
import logging
import sys
import psutil
from datetime import datetime
import random
from shapely.geometry import Point
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Constants
OVERLAP_THRESHOLD = 1e-12
AREA_THRESHOLD = 1e-10
BUFFER_DISTANCE_FACTOR = 1e-6
EDGE_TOLERANCE_FACTOR = 1e-5
MERIDIAN_TOLERANCE = 0.1
MIN_PERCENTAGE_DISPLAY = 0.1

def test_overlap_consistency(grid, test_bounds):
    """Calculate exact coverage distribution by overlaying cell geometries"""
    
    # Get all cells that intersect with test bounds
    cells = list(grid.generate_grid_cells(test_bounds))
    
    # Create GeoDataFrame with all cells
    cell_gdf = gpd.GeoDataFrame(geometry=[cell.geom for cell in cells])
    cell_gdf['coverage'] = 1
    
    # Dissolve with sum to get coverage counts
    coverage = cell_gdf.overlay(cell_gdf, how='union')
    coverage['coverage'] = coverage['coverage_1'].fillna(0) + coverage['coverage_2'].fillna(0)
    
    # Round coverage to nearest integer and filter tiny geometries
    coverage['coverage'] = np.round(coverage['coverage'])
    coverage = coverage[coverage.area > 1e-10]  # Filter out tiny slivers
    
    # Clip to test bounds
    coverage = coverage.clip(test_bounds)
    
    return coverage

def detect_cell_overlap(cell_info, other_cell_info, size):
    cell_geom, cell_centroid = cell_info['geometry'], cell_info['centroid']
    other_geom, other_centroid = other_cell_info['geometry'], other_cell_info['centroid']
    
    buffer_distance = size * BUFFER_DISTANCE_FACTOR
    edge_tolerance = size * EDGE_TOLERANCE_FACTOR
    
    # Check for geometric overlap
    buffered_cell = cell_geom.buffer(buffer_distance)
    buffered_other = other_geom.buffer(buffer_distance)
    
    if buffered_cell.intersects(buffered_other):
        intersection = cell_geom.intersection(other_geom)
        if intersection.area > OVERLAP_THRESHOLD or cell_geom.touches(other_geom):
            return True
            
    # Check for meridian cases
    near_meridian = (abs(abs(cell_centroid.x) - 90) < MERIDIAN_TOLERANCE or 
                    abs(abs(other_centroid.x) - 90) < MERIDIAN_TOLERANCE)
    
    if near_meridian:
        cell_bounds = cell_geom.bounds
        other_bounds = other_geom.bounds
        lat_overlap = (min(cell_bounds[3], other_bounds[3]) - 
                      max(cell_bounds[1], other_bounds[1])) > -edge_tolerance
        lon_diff = abs(abs(cell_centroid.x) - abs(other_centroid.x))
        if lat_overlap and lon_diff < edge_tolerance:
            return True
            
    return False

def analyze_overlaps(gdf, size):
    spatial_index = gdf.sindex
    overlap_analysis = {
        'percentages': [],
        'neighbor_counts': [],
        'overlapping_pairs': [],
        'suspicious_cases': []
    }
    
    for idx, row in gdf.iterrows():
        cell_geom = row['geometry']
        possible_matches_idx = list(spatial_index.intersection(cell_geom.bounds))
        possible_matches = gdf.iloc[possible_matches_idx]
        
        neighbors = []
        for other_idx, other_row in possible_matches.iterrows():
            if idx == other_idx or row['is_primary'] == other_row['is_primary']:
                continue
                
            if detect_cell_overlap(row, other_row, size):
                overlap_analysis['overlapping_pairs'].append((
                    (row['centroid'].x, row['centroid'].y),
                    (other_row['centroid'].x, other_row['centroid'].y)
                ))
                neighbors.append(other_idx)
        
        overlap_analysis['neighbor_counts'].append(len(neighbors))
    
    return overlap_analysis

def plot_grid_layout(ax, gdf, test_bounds):
    gpd.GeoDataFrame(geometry=[test_bounds]).plot(
        ax=ax, edgecolor='green', facecolor='none', 
        linewidth=2, linestyle='--'
    )
    gdf[gdf['is_primary']].plot(
        ax=ax, alpha=0.3, edgecolor='blue', 
        facecolor='blue', label='Primary'
    )
    gdf[~gdf['is_primary']].plot(
        ax=ax, alpha=0.3, edgecolor='black', 
        facecolor='yellow', label='Secondary'
    )
    ax.legend()
    ax.set_title('Grid Cell Layout (Blue: Primary, Yellow: Secondary)')

def calculate_pairwise_overlap_percentages(gdf, size):
    """Calculate overlap percentages between each pair of overlapping cells."""
    overlap_percentages = []
    spatial_index = gdf.sindex
    
    for idx, row in gdf.iterrows():
        cell_geom = row['geometry']
        cell_area = cell_geom.area
        
        possible_matches_idx = list(spatial_index.intersection(cell_geom.bounds))
        possible_matches = gdf.iloc[possible_matches_idx]
        
        for other_idx, other_row in possible_matches.iterrows():
            if idx >= other_idx:  # Skip self and avoid counting pairs twice
                continue
            
            if detect_cell_overlap(row, other_row, size):
                intersection = cell_geom.intersection(other_row['geometry'])
                intersection_area = intersection.area
                
                # Calculate overlap percentage relative to both cells
                percentage_of_cell = (intersection_area / cell_area) * 100
                percentage_of_other = (intersection_area / other_row['geometry'].area) * 100
                
                # Use the larger percentage to identify significant overlaps
                max_percentage = max(percentage_of_cell, percentage_of_other)
                overlap_percentages.append(max_percentage)
    
    return overlap_percentages

def plot_coverage_distribution(ax, gdf, size):
    """Plot histogram of pairwise cell overlap percentages."""
    overlap_percentages = calculate_pairwise_overlap_percentages(gdf, size)
    
    # Create histogram with percentage bins
    bins = np.linspace(0, 100, 21)  # 5% intervals
    
    ax.hist(overlap_percentages, bins=bins,
            edgecolor='black', alpha=0.7)
    
    ax.set_title('Pairwise Cell Overlap Distribution')
    ax.set_xlabel('Maximum Overlap Percentage Between Cell Pairs')
    ax.set_ylabel('Number of Cell Pairs')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_overlap = np.mean(overlap_percentages)
    ax.axvline(mean_overlap, color='red', linestyle='--', 
               label=f'Mean: {mean_overlap:.1f}%')
    
    # Add warning threshold line at 90%
    ax.axvline(90, color='orange', linestyle='--',
               label='Warning Threshold (90%)')
    
    ax.legend()

def investigate_grid_alignment(size: int = 160*2550):
    """
    Investigate grid alignment and overlap issues
    
    Args:
        size: Cell size in meters
    """
    start_time = datetime.now()
    logger.info(f"Starting grid alignment investigation with cell size {size}m")
    
    # Create a test area with the shape of CONUS
    CONUS = gpd.read_file('CONUS.geojson')
    test_bounds = CONUS.geometry.iloc[0]
    #test_bounds = box(-124.8, 24.3, -66.8, 49.3)
    logger.info(f"Created test bounds: {test_bounds.bounds}")
    
    # Generate cells using MajorTomGrid directly
    logger.info("Generating grid cells...")
    grid = MajorTomGrid(d=size, overlap=True)
    cells = list(grid.generate_grid_cells(test_bounds))
    logger.info(f"Generated {len(cells)} cells")
    
    # Convert cells to GeoDataFrame for analysis
    logger.info("Converting cells to GeoDataFrame...")
    cells_data = []
    primary_cells = []
    for i, cell in enumerate(cells):
        is_primary = getattr(cell, 'is_primary', False)
        cells_data.append({
            'tile_id': cell.id(),
            'geometry': cell.geom,
            'cell_id': i,
            'centroid': cell.geom.centroid,
            'is_primary': is_primary
        })
        if is_primary:
            primary_cells.append(cell.geom)
    
    gdf = gpd.GeoDataFrame(cells_data)
    logger.info(f"Created GeoDataFrame with {len(gdf)} cells ({len(primary_cells)} primary)")
    
    # Test primary grid coverage
    logger.info("Testing primary grid coverage...")

    # Calculate overlap percentages
    logger.info("Analyzing overlaps...")
    overlap_analysis = analyze_overlaps(gdf, size)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Grid Layout with primary/secondary cells
    logger.info("Creating grid layout plot...")
    plot_grid_layout(ax1, gdf, test_bounds)
    
    # Test coverage distribution
    logger.info("Analyzing coverage distribution...")
    coverage = test_overlap_consistency(grid, test_bounds)
    
    # Log value distribution
    logger.info(f"Coverage value range: {coverage['coverage'].min():.2f} to {coverage['coverage'].max():.2f}")
    
    # Plot 2: Coverage Distribution
    logger.info("Creating coverage distribution plot...")
    plot_coverage_distribution(ax2, gdf, size)
    
    # Plot 3: Number of Overlapping Neighbors
    logger.info("Creating neighbor count plot...")
    neighbor_counts = overlap_analysis['neighbor_counts']
    mean_neighbors = np.mean(neighbor_counts)

    # Create histogram with improved formatting
    ax3.hist(neighbor_counts, 
             bins=range(min(neighbor_counts), max(neighbor_counts) + 2, 1),
             edgecolor='black', 
             facecolor='skyblue',
             alpha=0.7,
             align='left')

    # Add mean line
    ax3.axvline(mean_neighbors, color='red', linestyle='--', 
                label=f'Mean: {mean_neighbors:.1f}')

    # Improve formatting
    ax3.set_title('Distribution of Overlapping Neighbors per Cell', pad=10)
    ax3.set_xlabel('Number of Overlapping Neighbors')
    ax3.set_ylabel('Number of Cells')

    # Add grid for better readability
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Set integer ticks on x-axis
    ax3.set_xticks(range(min(neighbor_counts), max(neighbor_counts) + 1))

    # Add legend only if we have the mean line
    ax3.legend()
    
    # Plot 4: Cell Centers and Connections
    logger.info("Creating cell centers plot...")
    
    # Extract coordinates for primary and secondary cells
    primary_centers = [(point.x, point.y) for point in gdf[gdf['is_primary']]['centroid']]
    secondary_centers = [(point.x, point.y) for point in gdf[~gdf['is_primary']]['centroid']]
    
    # Plot primary cell centers in blue
    ax4.scatter([x for x,y in primary_centers], [y for x,y in primary_centers], 
                color='blue', zorder=2, label='Primary')
    
    # Plot secondary cell centers in yellow
    ax4.scatter([x for x,y in secondary_centers], [y for x,y in secondary_centers], 
                color='yellow', edgecolor='black', zorder=2, label='Secondary')
    
    overlapping_pairs = overlap_analysis['overlapping_pairs']
    suspicious_cases = overlap_analysis['suspicious_cases']
    
    # Plot connections
    for (x1, y1), (x2, y2) in overlapping_pairs:
        ax4.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, zorder=1)
    
    # Log suspicious cases
    if suspicious_cases:
        logger.warning(f"Found {len(suspicious_cases)} suspicious cases where expected overlaps were not detected")
        for case in suspicious_cases[:5]:
            logger.warning(f"Cells {case[0]}-{case[1]}: distance={case[2]:.2f}")
    
    ax4.legend()
    ax4.set_title('Cell Centers and Connections\n(Lines show overlapping cells)')
    
    # Extract all x and y coordinates from both primary and secondary centers
    all_x = [x for x,y in primary_centers + secondary_centers]
    all_y = [y for x,y in primary_centers + secondary_centers]
    
    # Ensure proper axis limits
    ax4.set_xlim([min(all_x) - 0.05, max(all_x) + 0.05])
    ax4.set_ylim([min(all_y) - 0.05, max(all_y) + 0.05])
    
    plt.tight_layout()
    #plt.show()
    plt.savefig("grid_alignment.png")
    # Log overlap statistics
    logger.info(f"Found {len(overlapping_pairs)} overlapping cell pairs")
    
    end_time = datetime.now()
    logger.info(f"Analysis completed in {end_time - start_time}")

if __name__ == "__main__":
    investigate_grid_alignment()