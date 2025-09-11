"""
prepare_geodata.py
==================

This module handles all geospatial data preparation for the Wetting Front Tracker
application. Its primary purpose is to take a GeoJSON file of input polygons 
(e.g., avalanche paths) and process it into a final, analysis-ready GeoJSON file.

The workflow orchestrated by this module includes:
1.  **DEM Acquisition:** Strategically downloads Digital Elevation Model (DEM) data 
    from the OpenTopography API. It optimizes downloads by identifying clusters of 
    polygons and fetching only the necessary DEM tiles, which are then mosaicked 
    into a single raster file.
2.  **Aspect Classification:** Uses the DEM to calculate the terrain aspect for 
    the area covered by the input polygons. It then splits each polygon into 
    sub-polygons based on the four cardinal aspects: North, East, South, and West.
3.  **Data Cleaning:** Filters out small, insignificant "sliver" polygons that can be 
    generated during the splitting process. It also repairs and validates 
    geometries to ensure they are well-formed.
4.  **Data Linking:** Links each aspect-classified polygon to the most relevant 
    SNOWPACK (.pro) model output file. This matching is performed based on spatial 
    proximity and matching terrain aspect, ensuring that each polygon is associated 
    with the most representative snowpack simulation.
5.  **Manifest Generation:** Creates a manifest file listing all the unique .pro 
    files required for the subsequent analysis steps.

This module is designed to be run as a preliminary step before the main snowpack 
analysis. The final output, `linked_aspect_polygons.geojson`, serves as the primary 
input for the analysis phase.

Key Dependencies:
- geopandas for vector data manipulation
- rioxarray and rasterio for raster data processing
- requests for API communication
- scipy for spatial indexing (k-d tree)
- numpy for numerical operations
"""
import itertools
import math
import logging
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree  # type: ignore   
import rioxarray
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, LinearRing, mapping
from shapely import union_all
from numba import njit
from tqdm import tqdm
from typing import Optional

from .param_config import (OPENTOPO_API_KEY,
                           ASPECT_POLYGONS_GEOJSON, DEM_DATASETS, DEM_TIF, 
                           INPUT_POLYGONS_GEOJSON, LINKED_POLYGONS_GEOJSON,
                           PRO_FILE_MANIFEST, SNOWPACK_LOCATIONS_CSV)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


@njit
def _chaikin_iteration(coords, ratio=0.25):
    """
    Performs a single iteration of Chaikin's corner-cutting algorithm.

    This Numba-jitted function efficiently calculates the new vertices for one
    smoothing pass. It replaces each vertex with two new vertices, one at a
    `ratio` along the incoming segment and one at `1 - ratio` along the same
    segment.

    Args:
        coords (np.ndarray): An array of (x, y) coordinates for a linestring.
        ratio (float): The ratio for cutting corners, typically 0.25.

    Returns:
        np.ndarray: A new array of coordinates after one smoothing iteration.
    """
    new_coords = np.zeros((len(coords) * 2 - 2, 2))
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        
        q1_x, q1_y = x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        q2_x, q2_y = x1 + (x2 - x1) * (1 - ratio), y1 + (y2 - y1) * (1 - ratio)
        
        new_coords[2*i] = (q1_x, q1_y)
        new_coords[2*i+1] = (q2_x, q2_y)
        
    return new_coords

def chaikin_smooth(geometry, iterations=5):
    """
    Applies Chaikin's corner-cutting algorithm to smooth a polygon.

    This function iteratively applies the corner-cutting algorithm to the
    exterior and any interior rings of a polygon, resulting in a smoother,
    more organic shape.

    Args:
        geometry (Polygon): The input Shapely Polygon to smooth.
        iterations (int): The number of smoothing iterations to perform.

    Returns:
        Polygon: The smoothed Shapely Polygon. Returns the original geometry
                 if it's not a valid polygon.
    """
    if not isinstance(geometry, Polygon) or geometry.is_empty:
        return geometry

    exterior_coords = np.array(geometry.exterior.coords)
    for _ in range(iterations):
        exterior_coords = _chaikin_iteration(exterior_coords)
    
    smoothed_exterior = LinearRing(np.vstack([exterior_coords, exterior_coords[0]]))

    smoothed_interiors = []
    for interior in geometry.interiors:
        interior_coords = np.array(interior.coords)
        for _ in range(iterations):
            interior_coords = _chaikin_iteration(interior_coords)
        smoothed_interiors.append(LinearRing(np.vstack([interior_coords, interior_coords[0]])))

    return Polygon(smoothed_exterior, smoothed_interiors)


def _calculate_tiles(bounds: tuple, max_area_km2: float = 10000.0) -> list[tuple]:
    """
    Splits a large bounding box into a grid of smaller tiles under a max area.

    This function is used to break down a large DEM request into smaller chunks
    that are compliant with API limits. It calculates the required number of
    splits in latitude and longitude to ensure each tile is smaller than
    `max_area_km2`.

    Args:
        bounds (tuple): A tuple representing the bounding box (west, south,
                        east, north).
        max_area_km2 (float): The maximum desired area for each tile in square km.

    Returns:
        list[tuple]: A list of bounding box tuples for the generated tiles.
    """
    west, south, east, north = bounds
    lat_dist = (north - south) * 111
    lon_dist = (east - west) * 111 * math.cos(math.radians((north + south) / 2))
    
    area = lat_dist * lon_dist
    if area <= max_area_km2:
        return [bounds]

    split_factor = math.sqrt(area / max_area_km2)
    n_lat_splits = max(2, math.ceil(split_factor * (lat_dist / (lat_dist + lon_dist))))
    n_lon_splits = max(2, math.ceil(split_factor * (lon_dist / (lat_dist + lon_dist))))
    
    lat_step = (north - south) / n_lat_splits
    lon_step = (east - west) / n_lon_splits
    
    tiles = [
        (
            west + j * lon_step, south + i * lat_step,
            west + (j + 1) * lon_step, south + (i + 1) * lat_step
        )
        for i, j in itertools.product(range(n_lat_splits), range(n_lon_splits))
    ]
            
    logging.info(f"Bounding box split into a {n_lon_splits}x{n_lat_splits} grid ({len(tiles)} total tiles).")
    return tiles


def _download_tile(api_key: str, bounds: tuple, output_path: Path, dataset: dict) -> bool:
    """
    Downloads a single DEM tile from the OpenTopography API.

    Args:
        api_key (str): The OpenTopography API key.
        bounds (tuple): The bounding box (west, south, east, north) for the tile.
        output_path (Path): The local file path to save the downloaded GeoTIFF.
        dataset (dict): A dictionary containing configuration for the DEM
                        dataset, including its name and API endpoint.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    base_url = dataset['api_endpoint']
    west, south, east, north = bounds
    params = {dataset['param_name']: dataset['name'], 'south': south, 'north': north,
              'west': west, 'east': east, 'outputFormat': 'GTiff', 'API_Key': api_key}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        logging.error(f"Failed to download tile for bounds {bounds}. Status: {response.status_code}")
        logging.error(f"Response: {response.text}")
        return False


def _mosaic_tiles(tile_paths: list[Path], output_path: Path):
    """
    Merges multiple DEM GeoTIFF tiles into a single, seamless raster file.

    Args:
        tile_paths (list[Path]): A list of paths to the GeoTIFF tiles.
        output_path (Path): The path for the final merged GeoTIFF file.
    """
    logging.info(f"Mosaicking {len(tile_paths)} tiles into {output_path}...")
    sources = [rasterio.open(p) for p in tile_paths]
    mosaic, out_trans = merge(sources)
    
    out_meta = sources[0].meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                     "width": mosaic.shape[2], "transform": out_trans})
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    for src in sources:
        src.close()
    logging.info("Mosaicking complete.")

def _select_dem_dataset(bounds: tuple) -> dict:
    """
    Selects the best DEM dataset from the config based on the bounding box.

    This function iterates through available DEM datasets defined in the config
    and selects the most appropriate one based on whether the centroid of the
    bounding box falls within the dataset's coverage area. It falls back to a
    global dataset if no specific regional dataset matches.

    Args:
        bounds (tuple): The bounding box (west, south, east, north) of interest.

    Returns:
        dict: The configuration dictionary for the selected DEM dataset.
    """
    west, south, east, north = bounds
    centroid_lon, centroid_lat = (west + east) / 2, (south + north) / 2

    for dem in DEM_DATASETS:
        bb = dem['bounds']
        if bb[0] <= centroid_lon <= bb[2] and bb[1] <= centroid_lat <= bb[3]:
            logging.info(f"Selected DEM dataset: {dem['name']} for location ({centroid_lat:.2f}, {centroid_lon:.2f})")
            return dem
    
    logging.warning("No specific DEM found for location, using global fallback.")
    return DEM_DATASETS[-1]


def download_dem_for_polygons(polygons_gdf: gpd.GeoDataFrame, api_key: str, output_path: Path):
    """
    Strategically downloads and mosaics DEM tiles that intersect with polygons.

    This function optimizes the DEM download process. It first groups nearby
    polygons into clusters, calculates a bounding box for each cluster, and then
    downloads and mosaics the DEM tiles needed to cover those specific areas,
    avoiding unnecessary downloads.

    Args:
        polygons_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the polygons of interest.
        api_key (str): The OpenTopography API key.
        output_path (Path): The file path for the final mosaicked DEM.

    Raises:
        ValueError: If the API key has not been set.
        ConnectionError: If a DEM tile fails to download.
        FileNotFoundError: If no DEM tiles are downloaded.
    """
    if api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set your OPENTOPO_API_KEY in param_config.py")

    logging.info("Identifying polygon clusters to create optimized download boxes...")
    polygons_gdf = polygons_gdf.reset_index(drop=True)
    sindex = polygons_gdf.sindex
    visited_indices = set()
    clusters = []
    for index, polygon in polygons_gdf.iterrows():
        if index in visited_indices:
            continue
        
        buffer_distance = 0.1
        possible_matches_index_raw = list(sindex.intersection(polygon.geometry.buffer(buffer_distance).bounds))
        possible_matches_index = [int(i) for i in possible_matches_index_raw]
        possible_matches = polygons_gdf.iloc[possible_matches_index]
        cluster_gdf = possible_matches[possible_matches.intersects(polygon.geometry.buffer(buffer_distance))]
        
        visited_indices.update(cluster_gdf.index)
        clusters.append(cluster_gdf)

    logging.info(f"Identified {len(clusters)} polygon clusters.")

    tile_paths = []
    temp_dir = output_path.parent / "dem_tiles"
    temp_dir.mkdir(exist_ok=True)

    for i, cluster in enumerate(clusters):
        cluster_bounds = tuple(cluster.total_bounds)
        selected_dem = _select_dem_dataset(cluster_bounds)
        
        tiles_for_cluster = _calculate_tiles(cluster_bounds)
        
        for j, tile_bounds in enumerate(tiles_for_cluster):
            tile_path = temp_dir / f"cluster_{i}_tile_{j}.tif"
            if not tile_path.exists():
                logging.info(f"Downloading DEM for cluster {i+1}, tile {j+1}/{len(tiles_for_cluster)}...")
                if not _download_tile(api_key, tile_bounds, tile_path, selected_dem):
                    raise ConnectionError("Failed to download one or more DEM tiles.")
            tile_paths.append(tile_path)
            
    if not tile_paths:
        raise FileNotFoundError("No DEM tiles were downloaded.")
    elif len(tile_paths) > 1:
        _mosaic_tiles(tile_paths, output_path)
    else:
        tile_paths[0].rename(output_path)


def _filter_small_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float,
    min_area_ratio: float
) -> gpd.GeoDataFrame:
    """
    Removes small sliver polygons based on absolute area and relative area.

    This helper function cleans up the results of a split or intersection
    operation by removing polygons that are either smaller than a fixed
    area (`min_area_m2`) or smaller than a certain percentage (`min_area_ratio`)
    of their original parent polygon's area.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to filter. It must contain an
                                'original_area' column for ratio filtering.
        min_area_m2 (float): The minimum absolute area in square meters to keep a polygon.
        min_area_ratio (float): The minimum area ratio relative to the original
                                polygon to keep a polygon.

    Returns:
        gpd.GeoDataFrame: The filtered GeoDataFrame.
    """
    if 'original_area' not in gdf.columns:
        logging.warning("Missing 'original_area' column. Cannot perform ratio-based filtering.")
        return gdf

    initial_count = len(gdf)
    
    # The CRS should already be projected from previous steps, allowing for area calculation.
    if gdf.crs and gdf.crs.is_geographic:
        logging.warning("Reprojecting to calculate area accurately.")
        gdf['current_area'] = gdf.to_crs("EPSG:3857").geometry.area
    else:
        gdf['current_area'] = gdf.geometry.area
    
    # Conditions for KEEPING a polygon:
    # 1. Its area is greater than the absolute minimum.
    area_condition = gdf['current_area'] >= min_area_m2
    # 2. Its area is greater than the minimum percentage of its original parent polygon.
    ratio_condition = (gdf['current_area'] / gdf['original_area']) >= min_area_ratio

    # Keep polygons that meet EITHER condition.
    gdf_to_keep = gdf[area_condition & ratio_condition].copy()
    
    final_count = len(gdf_to_keep)
    logging.info(f"Filtered out {initial_count - final_count} small polygons based on area thresholds.")
    
    return gdf_to_keep.drop(columns=['current_area'])


def _ensure_dem_exists(polygons_gdf: gpd.GeoDataFrame, dem_path: Path):
    """
    Checks if a mosaicked DEM file exists and triggers a download if it does not.

    Args:
        polygons_gdf (gpd.GeoDataFrame): The GeoDataFrame of polygons that require a DEM.
        dem_path (Path): The expected path to the DEM file.
    """
    if not dem_path.exists():
        logging.info("DEM file not found. Downloading strategically...")
        bounds_gdf_wgs84 = polygons_gdf.to_crs("EPSG:4326")
        download_dem_for_polygons(bounds_gdf_wgs84, OPENTOPO_API_KEY, dem_path)

def _calculate_aspect_from_dem(dem_path: Path, polygons_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Clips a DEM to the extent of input polygons and calculates terrain aspect.

    This function opens the main DEM, clips it to the bounding box of the
    input polygons for efficiency, calculates the aspect (direction of steepest
    slope) for each pixel, and converts it to geographic degrees (0=North).

    Args:
        dem_path (Path): The path to the DEM GeoTIFF file.
        polygons_gdf (gpd.GeoDataFrame): GeoDataFrame with polygons to define the clip area.

    Returns:
        tuple: A tuple containing:
               - aspect_deg_cart (np.ndarray): The aspect raster in degrees.
               - clipped_dem (xr.DataArray): The clipped DEM raster.
               - polygons_in_dem_crs (gpd.GeoDataFrame): The input polygons
                 reprojected to the DEM's CRS.
    """
    logging.info("Opening DEM...")
    dem_datasets  = rioxarray.open_rasterio(dem_path, chunks={'x': 2048, 'y': 2048})
    
    # If open_rasterio returned a list, pick the first Dataset
    dem_rds = dem_datasets[0] if isinstance(dem_datasets, list) else dem_datasets
    
    # Ensure polygons are in the same CRS as the DEM
    polygons_in_dem_crs = polygons_gdf.to_crs(dem_rds.rio.crs)
    
    logging.info("Clipping DEM to polygon boundaries...")
    clipped_dem = dem_rds.rio.clip(
        polygons_in_dem_crs.geometry.apply(mapping),
        dem_rds.rio.crs,
        from_disk=True
    )
    
    # Calculate gradient and aspect
    elevation = clipped_dem.squeeze().values
    gy, gx = np.gradient(elevation)
    aspect_rad = np.arctan2(gy, -gx)
    
    # Convert to degrees (0° = North)
    aspect_deg_trig = np.degrees(aspect_rad)
    aspect_deg_cart = (90.0 - aspect_deg_trig + 360) % 360
    
    return aspect_deg_cart, clipped_dem, polygons_in_dem_crs

def _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorizes an aspect raster, intersects, filters, and cleans the result.

    This function takes the raw aspect raster and orchestrates the process of
    converting it into clean, aspect-classified vector polygons, ensuring they
    are filtered and repaired.

    Args:
        aspect_raster (np.ndarray): The 2D numpy array of aspect values.
        clipped_dem: The clipped DEM DataArray from rioxarray.
        polygons_in_dem_crs (gpd.GeoDataFrame): The source polygons in the DEM's CRS.

    Returns:
        Optional[gpd.GeoDataFrame]: A GeoDataFrame of the final, processed,
                                    aspect-classified polygons in WGS84, or
                                    None if no polygons are generated.
    """
    if polygons_in_dem_crs.crs.is_geographic:
        polygons_in_dem_crs['original_area'] = polygons_in_dem_crs.to_crs("EPSG:3857").geometry.area
    else:
        polygons_in_dem_crs['original_area'] = polygons_in_dem_crs.geometry.area

    # This function now correctly splits polygons by aspect while retaining attributes.
    aspect_gdf = _extract_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs)

    if aspect_gdf is None or aspect_gdf.empty:
        return None

    # Filter out the small sliver polygons created during the aspect split.
    filtered_gdf = _filter_small_polygons(
        aspect_gdf, 
        min_area_m2=1600.0, 
        min_area_ratio=0.1
    )
    
    # Post-process the final, filtered geometries.
    final_gdf = _postprocess_geometries(filtered_gdf)

    return final_gdf.to_crs("EPSG:4326")


def prepare_aspect_polygons(input_geojson: Path, 
                            output_geojson: Path,
                            force_update: bool = False):
    """
    Orchestrates the workflow to split input polygons by terrain aspect.

    This is a main public function for the module. It checks if the output file
    already exists, and if not (or if `force_update` is True), it manages the
    process of downloading the DEM, calculating aspect, and processing the
    polygons.

    Args:
        input_geojson (Path): Path to the input GeoJSON file of polygons.
        output_geojson (Path): Path to save the output aspect-classified GeoJSON.
        force_update (bool): If True, re-runs the process even if the output
                             file exists.
    """
    if output_geojson.exists() and not force_update:
        logging.info(f"Aspect-classified GeoJSON already exists: {output_geojson}")
        return

    polygons_gdf = gpd.read_file(input_geojson)
    dem_path = DEM_TIF

    _ensure_dem_exists(polygons_gdf, dem_path)
    aspect_raster, clipped_dem, polygons_in_dem_crs = _calculate_aspect_from_dem(dem_path, polygons_gdf)
    final_gdf = _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs)

    if final_gdf is not None and not final_gdf.empty:
        logging.info(f"Saving {len(final_gdf)} aspect-classified polygons to: {output_geojson}")
        final_gdf.to_file(output_geojson, driver='GeoJSON')
    else:
        logging.warning("No new polygons were generated after aspect classification.")
        
def _extract_aspect_polygons(aspect_raster: np.ndarray, 
                             clipped_dem, 
                             polygons_in_dem_crs: gpd.GeoDataFrame
) -> Optional[gpd.GeoDataFrame]:
    """
    Extracts polygons for each aspect bin and intersects them with source polygons.

    This function first vectorizes the entire aspect raster into four large
    multi-polygons (one for each cardinal direction). Then, it iterates through
    each individual source polygon and intersects it with these aspect unions.
    This approach preserves the original attributes of each source polygon for
    its newly created child polygons.

    Args:
        aspect_raster (np.ndarray): The 2D numpy array of aspect values.
        clipped_dem: The clipped DEM DataArray from rioxarray.
        polygons_in_dem_crs (gpd.GeoDataFrame): The source polygons.

    Returns:
        Optional[gpd.GeoDataFrame]: A new GeoDataFrame containing polygons split
                                    by aspect, or None if no valid polygons result.
    """
    aspect_bins = {
        "N": (315, 45),
        "E": (45, 135),
        "S": (135, 225),
        "W": (225, 315),
    }

    # Vectorize the entire aspect raster once for efficiency
    all_aspect_geoms = {}
    for aspect_name, (lower, upper) in aspect_bins.items():
        mask = _build_aspect_mask(aspect_raster, lower, upper, aspect_name)
        if aspect_geoms := _vectorize_aspect(mask, clipped_dem):
            all_aspect_geoms[aspect_name] = union_all(aspect_geoms)

    final_polygons = []
    # Iterate over each source polygon to process it individually
    for _, source_poly_row in tqdm(polygons_in_dem_crs.iterrows(), total=len(polygons_in_dem_crs), desc="Splitting by Aspect"):
        source_geom = source_poly_row.geometry
        
        for aspect_name, aspect_union_geom in all_aspect_geoms.items():
            # Intersect the source polygon with the union of all polygons for that aspect
            intersected = source_geom.intersection(aspect_union_geom)

            if intersected.is_empty:
                continue

            # Handle both Polygon and MultiPolygon results
            geoms = intersected.geoms if hasattr(intersected, 'geoms') else [intersected]

            # Create new features, carrying over attributes from the source polygon
            for poly in geoms:
                if not poly.is_empty:
                    properties = source_poly_row.to_dict()
                    properties['geometry'] = poly
                    properties['aspect'] = aspect_name
                    final_polygons.append(properties)

    if not final_polygons:
        return None

    # Create the final GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final_polygons, crs=clipped_dem.rio.crs)
    return _filter_valid_geometries(final_gdf)


def _build_aspect_mask(aspect_raster, 
                       lower: float, 
                       upper: float, 
                       aspect_name: str
) -> np.ndarray:
    """
    Creates a boolean mask for a given aspect bin.

    Handles the wrap-around case for North, which spans from 315 to 45 degrees.

    Args:
        aspect_raster (np.ndarray): The raster of aspect values.
        lower (float): The lower bound of the aspect bin in degrees.
        upper (float): The upper bound of the aspect bin in degrees.
        aspect_name (str): The name of the aspect (e.g., "N").

    Returns:
        np.ndarray: A boolean numpy array where True indicates a pixel is
                    within the aspect bin.
    """
    if aspect_name == "N":
        return (aspect_raster > lower) | (aspect_raster <= upper)
    return (aspect_raster > lower) & (aspect_raster <= upper)


def _vectorize_aspect(mask: np.ndarray, clipped_dem) -> list:
    """
    Converts a raster mask into a list of shapely geometries.

    Args:
        mask (np.ndarray): The boolean mask to vectorize.
        clipped_dem: The clipped DEM DataArray, used for its transform info.

    Returns:
        list: A list of Shapely geometry objects derived from the mask.
    """
    aspect_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=clipped_dem.rio.transform())
    return [shape(s) for s, v in aspect_shapes if v == 1]


def _filter_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filters a GeoDataFrame to keep only Polygon and MultiPolygon geometries.

    Intersection and other geometric operations can sometimes produce
    undesirable geometry types like points or lines. This function removes them.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to filter.

    Returns:
        gpd.GeoDataFrame: The filtered GeoDataFrame containing only valid polygons.
    """
    initial_count = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    removed_count = initial_count - len(gdf)
    if removed_count > 0:
        logging.info(f"Removed {removed_count} non-polygonal geometries (e.g., points or lines).")
    return gdf


def _postprocess_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Applies final cleaning and repairing steps to geometries.

    This function runs a buffer(0) operation, a common and effective trick to
    fix invalid geometries (like self-intersections) that may have been created
    during previous steps. It also cleans up temporary columns.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to process.

    Returns:
        gpd.GeoDataFrame: The processed GeoDataFrame with repaired geometries.
    """    
    # Removed for now
    #logging.info("Smoothing polygon corners...")
    #tqdm.pandas(desc="Smoothing Polygons")
    #gdf['geometry'] = gdf['geometry'].progress_apply(lambda geom: chaikin_smooth(geom))

    #logging.info("Simplifying geometries to reduce file size...")
    #gdf['geometry'] = gdf.simplify(tolerance=0.001, preserve_topology=True)

    logging.info("Repairing any invalid geometries...")
    gdf['geometry'] = gdf.geometry.buffer(0)

    # Clean up columns that are no longer relevant after filtering.
    columns_to_drop = ['original_area']
    gdf = gdf.drop(columns=[col for col in columns_to_drop if col in gdf.columns], errors='ignore')
    
    return gdf


def _convert_deg_to_cardinal_from_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame column of aspect degrees to cardinal directions.

    This utility is used to categorize the aspects of the SNOWPACK model
    locations into the same N, E, S, W bins used for the polygons.

    Args:
        df (pd.DataFrame): A DataFrame with an 'aspect' column in degrees.

    Returns:
        pd.DataFrame: The DataFrame with an added 'aspect_cardinal' column
                      and the 'aspect' column updated to the new cardinal values.
    """
    is_flat = df['aspect'] == 'Flat'
    df.loc[is_flat, 'aspect_cardinal'] = 'Flat'

    numeric_aspects = pd.to_numeric(df.loc[~is_flat, 'aspect'], errors='coerce')
    
    bins = [0, 45, 135, 225, 315, 360]
    labels = ["N_part2", "E", "S", "W", "N"]
    
    categorized_aspects = pd.cut(numeric_aspects, bins=bins, labels=labels, right=False, include_lowest=True)
    categorized_aspects = categorized_aspects.replace("N_part2", "N")
    
    df.loc[~is_flat, 'aspect_cardinal'] = categorized_aspects
    df['aspect'] = df['aspect_cardinal']
    return df

def link_polygons_to_pro_files(polygons_path: Path, locations_path: Path, output_path: Path):
    """
    Finds the most relevant .pro file for each aspect polygon.

    This function matches each processed polygon to the nearest SNOWPACK model
    location that has the same cardinal aspect. It uses a k-d tree for
    efficient nearest-neighbor searching.

    Args:
        polygons_path (Path): Path to the aspect-classified polygons GeoJSON.
        locations_path (Path): Path to the CSV of SNOWPACK model locations.
        output_path (Path): Path to save the final GeoJSON with linked .pro files.
    """
    if not polygons_path.exists():
        logging.error(f"Aspect polygon file not found at {polygons_path}. Run aspect preparation first.")
        return

    logging.info("Linking polygons to closest .pro files by aspect...")
    polygons_gdf = gpd.read_file(polygons_path)
    locations_df = pd.read_csv(locations_path)

    locations_df = _convert_deg_to_cardinal_from_map(locations_df)

    locations_gdf = gpd.GeoDataFrame(
        locations_df,
        geometry=gpd.points_from_xy(locations_df.longitude, locations_df.latitude),
        crs="EPSG:4326"
    )

    projected_crs = "EPSG:3587"
    polygons_proj = polygons_gdf.to_crs(projected_crs)
    locations_proj = locations_gdf.to_crs(projected_crs)

    polygons_proj['centroid'] = polygons_proj.geometry.centroid
    polygons_gdf['pro_file_path'] = None

    for aspect_name, group in tqdm(polygons_proj.groupby('aspect'), desc="Matching Aspects"):
        aspect_locations = locations_proj[locations_proj['aspect'] == aspect_name]

        if aspect_locations.empty:
            logging.warning(f"No .pro files found for aspect '{aspect_name}'. Skipping.")
            continue

        location_coords = np.array([geom.coords[0] for geom in aspect_locations.geometry])
        tree = cKDTree(location_coords)

        polygon_coords = np.array([geom.coords[0] for geom in group['centroid']])
        
        _, indices = tree.query(polygon_coords, k=1)

        matched_paths = aspect_locations.iloc[indices]['path'].values

        polygons_gdf.loc[group.index, 'pro_file_path'] = matched_paths

    unmatched_count = polygons_gdf['pro_file_path'].isna().sum()
    if unmatched_count > 0:
        logging.warning(f"{unmatched_count} polygons could not be matched to a .pro file and will be removed.")
        polygons_gdf.dropna(subset=['pro_file_path'], inplace=True)

    polygons_gdf.to_file(output_path, driver='GeoJSON')
    logging.info(f"Saved {len(polygons_gdf)} linked polygons to {output_path}")

def generate_pro_file_manifest(linked_polygons_path: Path, manifest_path: Path) -> set:
    """
    Reads the linked polygons GeoJSON and creates a manifest of unique .pro files.

    This function generates a simple text file that lists every unique .pro file
    path required for the subsequent analysis. This can be useful for data
    staging or validation.

    Args:
        linked_polygons_path (Path): The path to the final linked polygons GeoJSON.
        manifest_path (Path): The path to write the output manifest.txt file.

    Returns:
        set: A set of the unique .pro file paths.
    """
    logging.info("Generating .pro file manifest...")
    if not linked_polygons_path.exists():
        logging.error(f"Cannot generate manifest, file not found: {linked_polygons_path}")
        return set()

    gdf = gpd.read_file(linked_polygons_path)
    unique_paths = set(gdf['pro_file_path'].dropna().unique())

    with open(manifest_path, 'w') as f:
        for path in sorted(list(unique_paths)):
            f.write(f"{path}\n")
    
    logging.info(f"Manifest created with {len(unique_paths)} unique .pro files at: {manifest_path}")
    return unique_paths

if __name__ == '__main__':
    
    if OPENTOPO_API_KEY == "YOUR_API_KEY_HERE":
        logging.error("Please set your OPENTOPO_API_KEY in param_config.py")
    else:
        prepare_aspect_polygons(
            input_geojson=INPUT_POLYGONS_GEOJSON,
            output_geojson=ASPECT_POLYGONS_GEOJSON
        )
        
        link_polygons_to_pro_files(
            polygons_path=ASPECT_POLYGONS_GEOJSON,
            locations_path=SNOWPACK_LOCATIONS_CSV,
            output_path=LINKED_POLYGONS_GEOJSON
        )

        generate_pro_file_manifest(
            linked_polygons_path=LINKED_POLYGONS_GEOJSON,
            manifest_path=PRO_FILE_MANIFEST
        )

