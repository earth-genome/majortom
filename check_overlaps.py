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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

def investigate_grid_alignment(size: int = 2*2550):
    """
    Investigate grid alignment and overlap issues
    
    Args:
        size: Cell size in meters
    """
    start_time = datetime.now()
    logger.info(f"Starting grid alignment investigation with cell size {size}m")
    log_memory_usage()
    
    # Create a test area (slightly larger than 4x4 cells)
    test_bounds = box(-122.3, 37, -122, 37.3)
    logger.info(f"Created test bounds: {test_bounds.bounds}")
    
    # Generate cells using MajorTomGrid directly
    logger.info("Generating grid cells...")
    grid = MajorTomGrid(d=size, overlap=True)
    cells = list(grid.generate_grid_cells(test_bounds))
    logger.info(f"Generated {len(cells)} cells")
    log_memory_usage()
    
    # Convert cells to GeoDataFrame for analysis
    logger.info("Converting cells to GeoDataFrame...")
    cells_data = []
    for i, cell in enumerate(cells):
        if i % 10 == 0:
            logger.info(f"Processing cell {i+1}/{len(cells)}")
        cells_data.append({
            'tile_id': cell.id(),
            'geometry': cell.geom,
            'cell_id': i,
            'centroid': cell.geom.centroid
        })
    
    gdf = gpd.GeoDataFrame(cells_data)
    logger.info(f"Created GeoDataFrame with {len(gdf)} cells")
    log_memory_usage()
    
    # Analyze overlaps between cells
    logger.info("Starting overlap analysis...")
    overlap_analysis = defaultdict(list)
    problem_cells = []
    neighbor_counts = []
    
    total_cells = len(gdf)
    for i, cell1 in gdf.iterrows():
        if i % 5 == 0:
            logger.info(f"Analyzing overlaps for cell {i+1}/{total_cells}")
            log_memory_usage()
        
        # Find all overlapping cells
        cell_overlaps = []
        for j, cell2 in gdf.iterrows():
            if i != j and cell1.geometry.intersects(cell2.geometry):
                overlap_area = cell1.geometry.intersection(cell2.geometry).area
                overlap_percentage = (overlap_area / cell1.geometry.area) * 100
                
                if overlap_percentage > 1:
                    dx = cell2.centroid.x - cell1.centroid.x
                    dy = cell2.centroid.y - cell1.centroid.y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    cell_overlaps.append({
                        'neighbor_id': cell2.tile_id,
                        'overlap_percentage': overlap_percentage,
                        'distance': distance,
                        'dx': dx,
                        'dy': dy
                    })
        
        neighbor_counts.append(len(cell_overlaps))
        
        # Analyze overlap pattern
        issues = []
        cell_overlaps.sort(key=lambda x: x['distance'])
        
        for overlap in cell_overlaps:
            overlap_analysis['percentages'].append(overlap['overlap_percentage'])
            if not ((40 <= overlap['overlap_percentage'] <= 60) or
                    (15 <= overlap['overlap_percentage'] <= 35)):
                logger.warning(f"Abnormal overlap: {overlap['overlap_percentage']:.1f}% between {cell1.tile_id} and {overlap['neighbor_id']}")
                issues.append(f"Overlap of {overlap['overlap_percentage']:.1f}% with {overlap['neighbor_id']}")
        
        if issues:
            problem_cells.append((cell1.tile_id, issues))
    
    logger.info("Creating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Grid Layout with middle cell highlighted
    logger.info("Creating grid layout plot...")
    gdf.iloc[1:].plot(ax=ax1, alpha=0.1, edgecolor='black')
    gdf.iloc[len(gdf)//2:len(gdf)//2+1].plot(ax=ax1, alpha=0.1, edgecolor='red', linewidth=4)
    ax1.set_title('Grid Cell Layout (Middle Cell in Red)')
    
    # Plot 2: Overlap Percentage Distribution
    logger.info("Creating overlap distribution plot...")
    percentages = overlap_analysis['percentages']
    ax2.hist(percentages, bins=25, range=(0, 50), edgecolor='black')
    ax2.axvline(x=50, color='r', linestyle='--', linewidth=2, label='Expected 50% (same row/column)')
    ax2.axvline(x=25, color='g', linestyle='--', linewidth=2, label='Expected 25% (different row & column)')
    ax2.set_title('Overlap Percentage Distribution')
    ax2.set_xlabel('Overlap Percentage')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 55)
    ax2.legend(loc='lower left')
    
    # Plot 3: Number of Neighbors Distribution
    logger.info("Creating neighbor count distribution plot...")
    ax3.hist(neighbor_counts, bins=range(max(neighbor_counts)+2), align='right', rwidth=0.8)
    ax3.axvline(x=9, color='r', linestyle='--', label='Expected (middle cells)')
    ax3.axvline(x=6, color='g', linestyle='--', label='Expected (edge cells)')
    ax3.axvline(x=4, color='b', linestyle='--', label='Expected (corner cells)')
    
    ax3.set_title('Number of Overlapping Neighbors')
    ax3.set_xlabel('Number of Neighbors')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # Plot 4: Cell Centers and Connections
    logger.info("Creating cell centers and connections plot...")
    for i, cell1 in gdf.iterrows():
        ax4.scatter(cell1.centroid.x, cell1.centroid.y, c='blue', s=50)
        for j, cell2 in gdf.iterrows():
            if i < j and cell1.geometry.intersects(cell2.geometry):
                ax4.plot([cell1.centroid.x, cell2.centroid.x],
                        [cell1.centroid.y, cell2.centroid.y],
                        'k-', alpha=0.2)
    ax4.set_title('Cell Centers and Connections')
    
    plt.tight_layout()
    logger.info("Saving visualization to grid_alignment.png...")
    plt.savefig('grid_alignment.png')
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Investigation completed in {duration:.1f} seconds")
    log_memory_usage()

if __name__ == "__main__":
    investigate_grid_alignment()