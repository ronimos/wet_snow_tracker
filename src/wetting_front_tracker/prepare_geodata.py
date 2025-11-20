"""
prepare_geodata.py
==================

Geospatial data preparation for the Wetting Front Tracker application.

This module handles all geospatial preprocessing, including:
1. DEM acquisition from OpenTopography API with retry logic
2. Aspect classification and polygon splitting
3. Data cleaning and validation
4. Linking polygons to SNOWPACK model files

Key Dependencies:
- geopandas for vector data manipulation
- rioxarray and rasterio for raster data processing
- requests for API communication with retry logic
- scipy for spatial indexing (k-d tree)

Author: Ron Simenhois
Last Updated: October 12, 2025
"""

import itertools
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import rioxarray
from numba import njit
from rasterio.features import shapes
from rasterio.merge import merge
from scipy.spatial import cKDTree
from shapely import union_all
from shapely.geometry import LinearRing, Polygon, mapping, shape
from tqdm import tqdm

from .param_config import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DEM Download Settings
MAX_DEM_AREA_KM2 = 10000.0  # Maximum area per tile
MAX_RETRIES = 3  # Maximum number of download retry attempts
RETRY_DELAY = 2.0  # Initial delay between retries (seconds)
REQUEST_TIMEOUT = 120  # HTTP request timeout (seconds)

# Aspect Classification
ASPECT_BINS = {
    "N": (315, 45),
    "E": (45, 135),
    "S": (135, 225),
    "W": (225, 315),
}

# Polygon Filtering
MIN_POLYGON_AREA_M2 = 1600.0  # 40m x 40m minimum
MIN_AREA_RATIO = 0.1  # 10% of original polygon minimum

# Polygon Clustering
CLUSTER_BUFFER_DISTANCE = 0.1  # Degrees for clustering nearby polygons


# ---------------------------------------------------------------------------
# Error Handling and Retry Logic
# ---------------------------------------------------------------------------

class DEMDownloadError(Exception):
    """Raised when DEM download fails after all retries."""
    pass


class GeodataValidationError(Exception):
    """Raised when geodata validation fails."""
    pass


def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = RETRY_DELAY,
    backoff_factor: float = 2.0
):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        
    Returns:
        Function result if successful
        
    Raises:
        Last exception if all retries fail
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        raise last_exception
    
    return wrapper


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_geodataframe(gdf: gpd.GeoDataFrame, required_cols: Optional[List[str]] = None) -> None:
    """
    Validates a GeoDataFrame has required columns and valid geometries.
    
    Args:
        gdf: GeoDataFrame to validate
        required_cols: List of required column names
        
    Raises:
        GeodataValidationError: If validation fails
    """
    if gdf.empty:
        raise GeodataValidationError("GeoDataFrame is empty")
    
    if required_cols:
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise GeodataValidationError(f"Missing required columns: {missing_cols}")
    
    if not gdf.geometry.is_valid.all():
        invalid_count = (~gdf.geometry.is_valid).sum()
        logger.warning(f"Found {invalid_count} invalid geometries. Will attempt to repair.")


def validate_api_key(api_key: str) -> None:
    """
    Validates that an API key has been properly configured.
    
    Args:
        api_key: The API key to validate
        
    Raises:
        ValueError: If API key is not set
    """
    if api_key == "YOUR_API_KEY_HERE" or not api_key:
        raise ValueError(
            "OpenTopography API key not set. "
            "Set OPENTOPO_API_KEY in your .env file or environment variables."
        )


# ---------------------------------------------------------------------------
# Polygon Smoothing (Chaikin Algorithm)
# ---------------------------------------------------------------------------

@njit
def _chaikin_iteration(coords: np.ndarray, ratio: float = 0.25) -> np.ndarray:
    """
    Performs a single iteration of Chaikin's corner-cutting algorithm.

    This Numba-jitted function efficiently calculates new vertices for one
    smoothing pass. It replaces each vertex with two new vertices.

    Args:
        coords: Array of (x, y) coordinates
        ratio: Ratio for cutting corners (typically 0.25)

    Returns:
        New array of coordinates after smoothing
    """
    new_coords = np.zeros((len(coords) * 2 - 2, 2))
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        
        q1_x = x1 + (x2 - x1) * ratio
        q1_y = y1 + (y2 - y1) * ratio
        q2_x = x1 + (x2 - x1) * (1 - ratio)
        q2_y = y1 + (y2 - y1) * (1 - ratio)
        
        new_coords[2 * i] = (q1_x, q1_y)
        new_coords[2 * i + 1] = (q2_x, q2_y)
    
    return new_coords


def chaikin_smooth(geometry: Polygon, iterations: int = 5) -> Polygon:
    """
    Applies Chaikin's corner-cutting algorithm to smooth a polygon.

    Args:
        geometry: Input Shapely Polygon to smooth
        iterations: Number of smoothing iterations

    Returns:
        Smoothed Shapely Polygon
    """
    if not isinstance(geometry, Polygon) or geometry.is_empty:
        return geometry

    # Smooth exterior
    exterior_coords = np.array(geometry.exterior.coords)
    for _ in range(iterations):
        exterior_coords = _chaikin_iteration(exterior_coords)
    
    smoothed_exterior = LinearRing(
        np.vstack([exterior_coords, exterior_coords[0]])
    )

    # Smooth interiors
    smoothed_interiors = []
    for interior in geometry.interiors:
        interior_coords = np.array(interior.coords)
        for _ in range(iterations):
            interior_coords = _chaikin_iteration(interior_coords)
        smoothed_interiors.append(
            LinearRing(np.vstack([interior_coords, interior_coords[0]]))
        )

    return Polygon(smoothed_exterior, smoothed_interiors)


# ---------------------------------------------------------------------------
# DEM Download and Management
# ---------------------------------------------------------------------------

def _calculate_tiles(bounds: Tuple[float, float, float, float]) -> List[Tuple]:
    """
    Splits a large bounding box into smaller tiles under max area constraint.

    Args:
        bounds: Bounding box (west, south, east, north)

    Returns:
        List of bounding box tuples for tiles
    """
    west, south, east, north = bounds
    
    # Calculate approximate dimensions in km
    lat_dist = (north - south) * 111
    lon_dist = (east - west) * 111 * math.cos(math.radians((north + south) / 2))
    
    area = lat_dist * lon_dist
    if area <= MAX_DEM_AREA_KM2:
        return [bounds]

    # Calculate split factors
    split_factor = math.sqrt(area / MAX_DEM_AREA_KM2)
    total_dist = lat_dist + lon_dist
    n_lat_splits = max(2, math.ceil(split_factor * (lat_dist / total_dist)))
    n_lon_splits = max(2, math.ceil(split_factor * (lon_dist / total_dist)))
    
    # Generate tiles
    lat_step = (north - south) / n_lat_splits
    lon_step = (east - west) / n_lon_splits
    
    tiles = [
        (
            west + j * lon_step,
            south + i * lat_step,
            west + (j + 1) * lon_step,
            south + (i + 1) * lat_step
        )
        for i, j in itertools.product(range(n_lat_splits), range(n_lon_splits))
    ]
    
    logger.info(
        f"Split bounding box into {n_lon_splits}x{n_lat_splits} grid "
        f"({len(tiles)} tiles)"
    )
    return tiles


@retry_with_backoff
def _download_tile(
    api_key: str,
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    dataset_config: Dict
) -> None:
    """
    Downloads a single DEM tile from OpenTopography API with retry logic.

    Args:
        api_key: OpenTopography API key
        bounds: Bounding box (west, south, east, north)
        output_path: Local file path to save the GeoTIFF
        dataset_config: Dictionary with dataset configuration

    Raises:
        requests.RequestException: If download fails after retries
        DEMDownloadError: If response is not successful
    """
    west, south, east, north = bounds
    
    params = {
        dataset_config['param_name']: dataset_config['name'],
        'south': south,
        'north': north,
        'west': west,
        'east': east,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }

    logger.debug(f"Downloading tile: bounds={bounds}")
    
    response = requests.get(
        dataset_config['api_endpoint'],
        params=params,
        timeout=REQUEST_TIMEOUT
    )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        logger.debug(f"Successfully downloaded to {output_path}")
    else:
        error_msg = (
            f"Failed to download tile for bounds {bounds}. "
            f"Status: {response.status_code}, Response: {response.text[:200]}"
        )
        logger.error(error_msg)
        raise DEMDownloadError(error_msg)


def _mosaic_tiles(tile_paths: List[Path], output_path: Path) -> None:
    """
    Merges multiple DEM GeoTIFF tiles into a single seamless raster.

    Args:
        tile_paths: List of paths to GeoTIFF tiles
        output_path: Path for the final merged GeoTIFF

    Raises:
        IOError: If mosaic operation fails
    """
    logger.info(f"Mosaicking {len(tile_paths)} tiles into {output_path}...")
    
    try:
        sources = [rasterio.open(p) for p in tile_paths]
        mosaic, out_trans = merge(sources)
        
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Clean up sources
        for src in sources:
            src.close()
        
        logger.info("Mosaicking complete")
    
    except Exception as e:
        logger.error(f"Failed to mosaic tiles: {e}", exc_info=True)
        raise


def _identify_polygon_clusters(polygons_gdf: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
    """
    Groups nearby polygons into clusters for efficient DEM downloading.

    Args:
        polygons_gdf: GeoDataFrame containing polygons

    Returns:
        List of GeoDataFrames, one per cluster
    """
    logger.info("Identifying polygon clusters for optimized downloads...")
    
    polygons_gdf = polygons_gdf.reset_index(drop=True)
    sindex = polygons_gdf.sindex
    visited_indices = set()
    clusters = []
    
    for index, polygon in polygons_gdf.iterrows():
        if index in visited_indices:
            continue
        
        # Find nearby polygons
        buffered = polygon.geometry.buffer(CLUSTER_BUFFER_DISTANCE)
        possible_matches_raw = list(sindex.intersection(buffered.bounds))
        possible_matches_idx = [int(i) for i in possible_matches_raw]
        possible_matches = polygons_gdf.iloc[possible_matches_idx]
        
        # Get actual intersecting polygons
        cluster_gdf = possible_matches[possible_matches.intersects(buffered)]
        
        visited_indices.update(cluster_gdf.index)
        clusters.append(cluster_gdf)
    
    logger.info(f"Identified {len(clusters)} polygon clusters")
    return clusters


def download_dem_for_polygons(
    polygons_gdf: gpd.GeoDataFrame,
    api_key: str,
    output_path: Path
) -> None:
    """
    Strategically downloads and mosaics DEM tiles for polygon areas.

    This function optimizes downloads by clustering nearby polygons and only
    downloading DEM tiles for those specific areas.

    Args:
        polygons_gdf: GeoDataFrame containing polygons of interest
        api_key: OpenTopography API key
        output_path: Path for the final mosaicked DEM

    Raises:
        ValueError: If API key is invalid
        DEMDownloadError: If downloads fail
        FileNotFoundError: If no tiles are downloaded
    """
    validate_api_key(api_key)
    
    # Identify clusters
    clusters = _identify_polygon_clusters(polygons_gdf)
    
    tile_paths = []
    temp_dir = output_path.parent / "dem_tiles"
    temp_dir.mkdir(exist_ok=True)

    # Process each cluster
    for i, cluster in enumerate(clusters):
        cluster_bounds = tuple(cluster.total_bounds)
        
        # Select appropriate DEM dataset
        centroid_lon = (cluster_bounds[0] + cluster_bounds[2]) / 2
        centroid_lat = (cluster_bounds[1] + cluster_bounds[3]) / 2
        
        selected_dem = config.dem.get_dataset_for_location(centroid_lon, centroid_lat)
        if not selected_dem:
            logger.warning(f"No DEM dataset found for cluster {i}")
            continue
        
        logger.info(f"Using {selected_dem.name} for cluster {i + 1}/{len(clusters)}")
        
        # Get tiles for this cluster
        tiles_for_cluster = _calculate_tiles(cluster_bounds)
        
        # Download each tile
        for j, tile_bounds in enumerate(tiles_for_cluster):
            tile_path = temp_dir / f"cluster_{i}_tile_{j}.tif"
            
            if tile_path.exists():
                logger.debug(f"Tile already exists: {tile_path}")
            else:
                logger.info(
                    f"Downloading cluster {i + 1}, tile {j + 1}/{len(tiles_for_cluster)}..."
                )
                
                dataset_config = {
                    'name': selected_dem.name,
                    'api_endpoint': selected_dem.api_endpoint,
                    'param_name': selected_dem.param_name
                }
                
                try:
                    _download_tile(api_key, tile_bounds, tile_path, dataset_config)
                except Exception as e:
                    logger.error(f"Failed to download tile after retries: {e}")
                    raise DEMDownloadError(f"Failed to download tile: {e}")
            
            tile_paths.append(tile_path)
    
    if not tile_paths:
        raise FileNotFoundError("No DEM tiles were downloaded")
    
    # Mosaic or rename single tile
    if len(tile_paths) > 1:
        _mosaic_tiles(tile_paths, output_path)
    else:
        logger.info("Single tile, renaming instead of mosaicking")
        tile_paths[0].rename(output_path)
    
    logger.info(f"DEM saved to {output_path}")


# ---------------------------------------------------------------------------
# Polygon Filtering and Cleaning
# ---------------------------------------------------------------------------

def _filter_small_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float = MIN_POLYGON_AREA_M2,
    min_area_ratio: float = MIN_AREA_RATIO
) -> gpd.GeoDataFrame:
    """
    Removes small sliver polygons based on absolute and relative area.

    Args:
        gdf: GeoDataFrame to filter (must have 'original_area' column)
        min_area_m2: Minimum absolute area in square meters
        min_area_ratio: Minimum ratio relative to original polygon

    Returns:
        Filtered GeoDataFrame
    """
    if 'original_area' not in gdf.columns:
        logger.warning("Missing 'original_area' column. Skipping ratio filter.")
        return gdf

    initial_count = len(gdf)
    
    # Calculate current area
    if gdf.crs and gdf.crs.is_geographic:
        logger.debug("Reprojecting to calculate area accurately")
        gdf['current_area'] = gdf.to_crs("EPSG:3857").geometry.area
    else:
        gdf['current_area'] = gdf.geometry.area
    
    # Filter conditions
    area_condition = gdf['current_area'] >= min_area_m2
    ratio_condition = (gdf['current_area'] / gdf['original_area']) >= min_area_ratio
    
    gdf_filtered = gdf[area_condition & ratio_condition].copy()
    
    filtered_count = initial_count - len(gdf_filtered)
    logger.info(f"Filtered out {filtered_count} small polygons")
    
    return gdf_filtered.drop(columns=['current_area'])


def _filter_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filters to keep only Polygon and MultiPolygon geometries.

    Args:
        gdf: GeoDataFrame to filter

    Returns:
        Filtered GeoDataFrame with only valid polygon types
    """
    initial_count = len(gdf)
    valid_types = ['Polygon', 'MultiPolygon']
    gdf_filtered = gdf[gdf.geometry.geom_type.isin(valid_types)].copy()
    
    removed_count = initial_count - len(gdf_filtered)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} non-polygonal geometries")
    
    return gdf_filtered


def _postprocess_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Applies final cleaning and repairing to geometries.

    Args:
        gdf: GeoDataFrame to process

    Returns:
        Processed GeoDataFrame with repaired geometries
    """
    logger.info("Repairing invalid geometries...")
    gdf['geometry'] = gdf.geometry.buffer(0)
    
    # Clean up temporary columns
    columns_to_drop = ['original_area']
    gdf = gdf.drop(
        columns=[col for col in columns_to_drop if col in gdf.columns],
        errors='ignore'
    )
    
    return gdf


# ---------------------------------------------------------------------------
# Aspect Classification
# ---------------------------------------------------------------------------

def _calculate_aspect_from_dem(
    dem_path: Path,
    polygons_gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, Any, gpd.GeoDataFrame]:
    """
    Clips DEM and calculates terrain aspect.

    Args:
        dem_path: Path to the DEM GeoTIFF file
        polygons_gdf: GeoDataFrame with polygons to define clip area

    Returns:
        Tuple of (aspect_array, clipped_dem, reprojected_polygons)
    """
    logger.info("Opening DEM...")
    dem_datasets = rioxarray.open_rasterio(
        dem_path,
        chunks={'x': 2048, 'y': 2048}
    )
    
    # Handle list or single dataset
    dem_rds = dem_datasets[0] if isinstance(dem_datasets, list) else dem_datasets
    
    # Reproject polygons to DEM CRS
    polygons_in_dem_crs = polygons_gdf.to_crs(dem_rds.rio.crs)
    
    logger.info("Clipping DEM to polygon boundaries...")
    clipped_dem = dem_rds.rio.clip(
        polygons_in_dem_crs.geometry.apply(mapping),
        dem_rds.rio.crs,
        from_disk=True
    )
    
    # Calculate gradient and aspect
    logger.info("Calculating aspect...")
    elevation = clipped_dem.squeeze().values
    gy, gx = np.gradient(elevation)
    aspect_rad = np.arctan2(gy, -gx)
    
    # Convert to degrees (0° = North)
    aspect_deg_trig = np.degrees(aspect_rad)
    aspect_deg_cart = (90.0 - aspect_deg_trig + 360) % 360
    
    return aspect_deg_cart, clipped_dem, polygons_in_dem_crs


def _build_aspect_mask(
    aspect_raster: np.ndarray,
    lower: float,
    upper: float,
    aspect_name: str
) -> np.ndarray:
    """
    Creates a boolean mask for a given aspect bin.

    Args:
        aspect_raster: Raster of aspect values
        lower: Lower bound of aspect bin (degrees)
        upper: Upper bound of aspect bin (degrees)
        aspect_name: Name of the aspect (e.g., "N")

    Returns:
        Boolean array where True indicates pixels in the aspect bin
    """
    if aspect_name == "N":
        # North wraps around 0°
        return (aspect_raster > lower) | (aspect_raster <= upper)
    return (aspect_raster > lower) & (aspect_raster <= upper)


def _vectorize_aspect(mask: np.ndarray, clipped_dem: Any) -> List:
    """
    Converts a raster mask into shapely geometries.

    Args:
        mask: Boolean mask to vectorize
        clipped_dem: Clipped DEM for transform info

    Returns:
        List of Shapely geometry objects
    """
    aspect_shapes = shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=clipped_dem.rio.transform()
    )
    return [shape(s) for s, v in aspect_shapes if v == 1]


def _extract_aspect_polygons(
    aspect_raster: np.ndarray,
    clipped_dem: Any,
    polygons_in_dem_crs: gpd.GeoDataFrame
) -> Optional[gpd.GeoDataFrame]:
    """
    Extracts polygons for each aspect bin and intersects with source polygons.

    This function vectorizes the aspect raster into four aspect unions (N, E, S, W),
    then intersects each source polygon with these unions, preserving attributes.

    Args:
        aspect_raster: 2D array of aspect values
        clipped_dem: Clipped DEM DataArray
        polygons_in_dem_crs: Source polygons in DEM CRS

    Returns:
        GeoDataFrame with polygons split by aspect, or None if empty
    """
    # Vectorize entire raster once for efficiency
    all_aspect_geoms = {}
    
    for aspect_name, (lower, upper) in ASPECT_BINS.items():
        mask = _build_aspect_mask(aspect_raster, lower, upper, aspect_name)
        aspect_geoms = _vectorize_aspect(mask, clipped_dem)
        
        if aspect_geoms:
            all_aspect_geoms[aspect_name] = union_all(aspect_geoms)
    
    if not all_aspect_geoms:
        logger.warning("No aspect geometries created")
        return None

    # Intersect each source polygon with aspect unions
    final_polygons = []
    
    for _, source_poly_row in tqdm(
        polygons_in_dem_crs.iterrows(),
        total=len(polygons_in_dem_crs),
        desc="Splitting by Aspect"
    ):
        source_geom = source_poly_row.geometry
        
        for aspect_name, aspect_union_geom in all_aspect_geoms.items():
            intersected = source_geom.intersection(aspect_union_geom)

            if intersected.is_empty:
                continue

            # Handle Polygon or MultiPolygon results
            geoms = (
                intersected.geoms if hasattr(intersected, 'geoms')
                else [intersected]
            )

            # Create new features with source attributes
            for poly in geoms:
                if not poly.is_empty:
                    properties = source_poly_row.to_dict()
                    properties['geometry'] = poly
                    properties['aspect'] = aspect_name
                    final_polygons.append(properties)

    if not final_polygons:
        logger.warning("No polygons generated after aspect intersection")
        return None

    final_gdf = gpd.GeoDataFrame(final_polygons, crs=clipped_dem.rio.crs)
    return _filter_valid_geometries(final_gdf)


def _process_aspect_polygons(
    aspect_raster: np.ndarray,
    clipped_dem: Any,
    polygons_in_dem_crs: gpd.GeoDataFrame
) -> Optional[gpd.GeoDataFrame]:
    """
    Orchestrates aspect polygon extraction, filtering, and cleaning.

    Args:
        aspect_raster: 2D array of aspect values
        clipped_dem: Clipped DEM DataArray
        polygons_in_dem_crs: Source polygons in DEM CRS

    Returns:
        Processed GeoDataFrame in WGS84, or None if empty
    """
    # Store original areas for filtering
    if polygons_in_dem_crs.crs.is_geographic:
        polygons_in_dem_crs['original_area'] = (
            polygons_in_dem_crs.to_crs("EPSG:3857").geometry.area
        )
    else:
        polygons_in_dem_crs['original_area'] = polygons_in_dem_crs.geometry.area

    # Extract aspect polygons
    aspect_gdf = _extract_aspect_polygons(
        aspect_raster,
        clipped_dem,
        polygons_in_dem_crs
    )

    if aspect_gdf is None or aspect_gdf.empty:
        return None

    # Filter small slivers
    filtered_gdf = _filter_small_polygons(aspect_gdf)
    
    # Post-process geometries
    final_gdf = _postprocess_geometries(filtered_gdf)

    return final_gdf.to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# Main Processing Functions
# ---------------------------------------------------------------------------

def prepare_aspect_polygons(
    input_geojson: Path,
    output_geojson: Path,
    force_update: bool = False
) -> None:
    """
    Orchestrates the workflow to split input polygons by terrain aspect.

    This is the main public function for aspect classification. It manages
    DEM download, aspect calculation, and polygon processing.

    Args:
        input_geojson: Path to input GeoJSON file of polygons
        output_geojson: Path to save output aspect-classified GeoJSON
        force_update: If True, re-runs even if output exists

    Raises:
        FileNotFoundError: If input file doesn't exist
        GeodataValidationError: If input data is invalid
    """
    if output_geojson.exists() and not force_update:
        logger.info(f"Aspect-classified GeoJSON already exists: {output_geojson}")
        return

    # Load and validate input
    if not input_geojson.exists():
        raise FileNotFoundError(f"Input file not found: {input_geojson}")
    
    logger.info(f"Loading polygons from {input_geojson}")
    polygons_gdf = gpd.read_file(input_geojson)
    
    try:
        validate_geodataframe(polygons_gdf)
    except GeodataValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise

    # Ensure DEM exists
    dem_path = config.paths.dem_tif
    if not dem_path.exists():
        logger.info("DEM file not found. Downloading strategically...")
        bounds_gdf_wgs84 = polygons_gdf.to_crs("EPSG:4326")
        download_dem_for_polygons(
            bounds_gdf_wgs84,
            config.api.opentopo_api_key,
            dem_path
        )

    # Calculate aspect and process polygons
    aspect_raster, clipped_dem, polygons_in_dem_crs = _calculate_aspect_from_dem(
        dem_path,
        polygons_gdf
    )
    
    final_gdf = _process_aspect_polygons(
        aspect_raster,
        clipped_dem,
        polygons_in_dem_crs
    )

    # Save results
    if final_gdf is not None and not final_gdf.empty:
        logger.info(
            f"Saving {len(final_gdf)} aspect-classified polygons to {output_geojson}"
        )
        final_gdf.to_file(output_geojson, driver='GeoJSON')
    else:
        logger.warning("No polygons generated after aspect classification")


def _convert_aspect_to_cardinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts aspect degrees to cardinal directions, or keeps cardinal aspects as-is.
    
    Handles two input formats:
    1. Numeric degrees (0-360) → converts to N, E, S, W
    2. Already cardinal (N, E, S, W, flat, Flat) → standardizes capitalization

    Args:
        df: DataFrame with 'aspect' column in degrees or cardinal directions

    Returns:
        DataFrame with 'aspect' converted to N, E, S, W, or Flat
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if aspects are already cardinal (not numeric)
    # Try to convert first value to numeric to test
    test_val = str(df['aspect'].iloc[0]).strip().upper()
    is_already_cardinal = test_val in ['N', 'E', 'S', 'W', 'FLAT']
    
    if is_already_cardinal:
        # Aspects are already cardinal - just standardize capitalization
        # Handle 'flat' vs 'Flat'
        df['aspect'] = df['aspect'].astype(str).str.strip().str.upper()
        df.loc[df['aspect'] == 'FLAT', 'aspect'] = 'Flat'
        return df
    
    # Otherwise, convert numeric degrees to cardinal
    is_flat = df['aspect'] == 'Flat'
    df.loc[is_flat, 'aspect_cardinal'] = 'Flat'

    numeric_aspects = pd.to_numeric(df.loc[~is_flat, 'aspect'], errors='coerce')
    
    bins = [0, 45, 135, 225, 315, 360]
    labels = ["N", "E", "S", "W", "N"]
    
    categorized_aspects = pd.cut(
        numeric_aspects,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=False
    )
    
    df.loc[~is_flat, 'aspect_cardinal'] = categorized_aspects
    df['aspect'] = df['aspect_cardinal']
    
    return df



def link_polygons_to_pro_files(
    polygons_path: Path,
    locations_path: Path,
    output_path: Path
) -> None:
    """
    Finds the most relevant .pro file for each aspect polygon.

    This function matches each polygon to the nearest SNOWPACK model location
    with the same cardinal aspect using a k-d tree for efficiency.

    Args:
        polygons_path: Path to aspect-classified polygons GeoJSON
        locations_path: Path to CSV of SNOWPACK model locations
        output_path: Path to save final GeoJSON with linked .pro files

    Raises:
        FileNotFoundError: If input files don't exist
    """
    if not polygons_path.exists():
        raise FileNotFoundError(
            f"Aspect polygon file not found at {polygons_path}. "
            "Run aspect preparation first."
        )

    logger.info("Linking polygons to closest .pro files by aspect...")
    
    # Load data
    polygons_gdf = gpd.read_file(polygons_path)
    locations_df = pd.read_csv(locations_path)
    
    # Convert aspects to cardinal directions
    locations_df = _convert_aspect_to_cardinal(locations_df)

    # Create GeoDataFrame from locations
    locations_gdf = gpd.GeoDataFrame(
        locations_df,
        geometry=gpd.points_from_xy(
            locations_df.longitude,
            locations_df.latitude
        ),
        crs="EPSG:4326"
    )

    # Project to metric CRS for distance calculations
    projected_crs = "EPSG:3587"
    polygons_proj = polygons_gdf.to_crs(projected_crs)
    locations_proj = locations_gdf.to_crs(projected_crs)

    # Calculate centroids
    polygons_proj['centroid'] = polygons_proj.geometry.centroid
    polygons_gdf['pro_file_path'] = None

    # Match by aspect
    for aspect_name, group in tqdm(
        polygons_proj.groupby('aspect'),
        desc="Matching Aspects"
    ):
        aspect_locations = locations_proj[locations_proj['aspect'] == aspect_name]

        if aspect_locations.empty:
            logger.warning(f"No .pro files found for aspect '{aspect_name}'")
            continue

        # Build k-d tree for this aspect
        location_coords = np.array([
            geom.coords[0] for geom in aspect_locations.geometry
        ])
        tree = cKDTree(location_coords)

        # Find nearest location for each polygon
        polygon_coords = np.array([
            geom.coords[0] for geom in group['centroid']
        ])
        
        _, indices = tree.query(polygon_coords, k=1)
        matched_paths = aspect_locations.iloc[indices]['path'].values
        
        polygons_gdf.loc[group.index, 'pro_file_path'] = matched_paths

    # Remove unmatched polygons
    unmatched_count = polygons_gdf['pro_file_path'].isna().sum()
    if unmatched_count > 0:
        logger.warning(
            f"{unmatched_count} polygons could not be matched and will be removed"
        )
        polygons_gdf.dropna(subset=['pro_file_path'], inplace=True)

    # Save results
    polygons_gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(polygons_gdf)} linked polygons to {output_path}")


def generate_pro_file_manifest(
    linked_polygons_path: Path,
    manifest_path: Path
) -> set:
    """
    Creates a manifest of unique .pro files from linked polygons.

    Args:
        linked_polygons_path: Path to linked polygons GeoJSON
        manifest_path: Path to write manifest file

    Returns:
        Set of unique .pro file paths
    """
    logger.info("Generating .pro file manifest...")
    
    if not linked_polygons_path.exists():
        logger.error(f"File not found: {linked_polygons_path}")
        return set()

    gdf = gpd.read_file(linked_polygons_path)
    unique_paths = set(gdf['pro_file_path'].dropna().unique())

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(sorted(list(unique_paths)), f, indent=4)
    
    logger.info(
        f"Manifest created with {len(unique_paths)} unique .pro files "
        f"at {manifest_path}"
    )
    return unique_paths


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        prepare_aspect_polygons(
            input_geojson=config.paths.input_polygons,
            output_geojson=config.paths.aspect_polygons
        )
        
        link_polygons_to_pro_files(
            polygons_path=config.paths.aspect_polygons,
            locations_path=config.paths.snowpack_locations_csv,
            output_path=config.paths.linked_polygons
        )

        generate_pro_file_manifest(
            linked_polygons_path=config.paths.linked_polygons,
            manifest_path=config.paths.pro_file_manifest
        )
        
        logger.info("Geodata preparation complete!")
    
    except Exception as e:
        logger.error(f"Geodata preparation failed: {e}", exc_info=True)
        raise