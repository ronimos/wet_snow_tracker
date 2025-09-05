import itertools
import math
import logging
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree   
import rioxarray
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon, Polygon, LinearRing, mapping, box
from numba import njit
from tqdm import tqdm
from typing import Optional

from .param_config import (OPENTOPO_API_KEY,
                           ASPECT_POLYGONS_GEOJSON, DEV,
                           DEM_TIF, INPUT_PATH, DEM_DATASETS)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@njit
def _chaikin_iteration(coords, ratio=0.25):
    """Performs a single iteration of Chaikin's corner-cutting algorithm."""
    new_coords = np.zeros((len(coords) * 2 - 2, 2))
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        
        q1_x, q1_y = x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        q2_x, q2_y = x1 + (x2 - x1) * (1 - ratio), y1 + (y2 - y1) * (1 - ratio)
        
        new_coords[2*i] = (q1_x, q1_y)
        new_coords[2*i+1] = (q2_x, q2_y)
        
    return new_coords

def chaikin_smooth(geometry, iterations=2):
    """Applies Chaikin's corner-cutting algorithm to smooth a polygon."""
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
    """Splits a large bounding box into a grid of smaller tiles."""
    west, south, east, north = bounds
    lat_dist = (north - south) * 111
    lon_dist = (east - west) * 111 * math.cos(math.radians((north + south) / 2))
    
    area = lat_dist * lon_dist
    if area <= max_area_km2:
        return [bounds]

    split_factor = math.sqrt(area / max_area_km2)
    # Ensure at least 2 splits if area is slightly over
    n_lat_splits = max(2, math.ceil(split_factor * (lat_dist / (lat_dist + lon_dist))))
    n_lon_splits = max(2, math.ceil(split_factor * (lon_dist / (lat_dist + lon_dist))))
    
    lat_step = (north - south) / n_lat_splits
    lon_step = (east - west) / n_lon_splits
    
    tiles = [
        (
            west + j * lon_step,        # west coordinate
            south + i * lat_step,       # south coordinate
            west + (j + 1) * lon_step,  # east coordinate
            south + (i + 1) * lat_step  # north coordinate
        )
        for i, j in itertools.product(range(n_lat_splits), range(n_lon_splits))
    ]
            
    logging.info(f"Bounding box split into a {n_lon_splits}x{n_lat_splits} grid ({len(tiles)} total tiles).")
    return tiles

def _get_dem_config(cluster_bounds: tuple) -> Optional[dict]:
    """
    Selects the best DEM dataset for a given bounding box.

    Iterates through the DEM_DATASETS defined in the config and returns the first
    one that completely contains the cluster's bounds.

    Args:
        cluster_bounds (tuple): The (west, south, east, north) bounds of a polygon cluster.

    Returns:
        dict | None: The configuration dictionary for the best-fit DEM, or None.
    """
    cl_w, cl_s, cl_e, cl_n = cluster_bounds
    for config in DEM_DATASETS:
        ds_w, ds_s, ds_e, ds_n = config['bounds']
        if ds_w <= cl_w and ds_s <= cl_s and ds_e >= cl_e and ds_n >= cl_n:
            logging.info(f"Selected DEM dataset: {config['name']}")
            return config
    return None

def _download_tile(api_key: str, bounds: tuple, output_path: Path, dem_config: dict) -> bool:
    """
    Downloads a single DEM tile using a specific dataset configuration.
    """
    base_url = dem_config['api_endpoint']
    # The API parameter for the dataset name is different for global vs. USGS endpoints
    dataset_name_key = 'datasetName' if 'usgsdem' in base_url else 'demtype'
    
    west, south, east, north = bounds
    params = {
        dataset_name_key: dem_config['datasetName'],
        'south': south, 'north': north,
        'west': west, 'east': east,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }

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
    """Merges multiple DEM GeoTIFF tiles into a single raster file."""
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


def download_dem_for_polygons(polygons_gdf: gpd.GeoDataFrame, api_key: str, output_path: Path):
    """
    Strategically downloads DEMs by finding clusters of polygons.
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
        raw_indices = list(sindex.intersection(polygon.geometry.buffer(buffer_distance).bounds))
        possible_matches_index = [int(i) for i in raw_indices]
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
        
        dem_config = _get_dem_config(cluster_bounds)
        if not dem_config:
            logging.warning(f"No suitable DEM dataset found for cluster {i+1} at bounds {cluster_bounds}. Skipping.")
            continue

        tiles_for_cluster = _calculate_tiles(cluster_bounds)
        
        for j, tile_bounds in enumerate(tiles_for_cluster):
            tile_path = temp_dir / f"cluster_{i}_tile_{j}.tif"
            if not tile_path.exists():
                logging.info(f"Downloading DEM for cluster {i+1}, tile {j+1} using {dem_config['name']}...")
                if not _download_tile(api_key, tile_bounds, tile_path, dem_config):
                    raise ConnectionError("Failed to download one or more DEM tiles.")
            tile_paths.append(tile_path)
            
    if not tile_paths:
        raise FileNotFoundError("No DEM tiles were downloaded.")
    elif len(tile_paths) > 1:
        _mosaic_tiles(tile_paths, output_path)
    else:
            tile_paths[0].rename(output_path)


def _ensure_dem_exists(polygons_gdf: gpd.GeoDataFrame, dem_path: Path):
    """
    Checks if a mosaicked DEM file exists and triggers a download if it does not.

    This function serves as a data-fetching gateway, ensuring that the necessary
    DEM is available locally before any processing begins.

    Args:
        polygons_gdf (gpd.GeoDataFrame): The GeoDataFrame of input polygons, used
                                         to determine the required DEM bounds.
        dem_path (Path): The target path for the final, mosaicked DEM file.
    """
    if not dem_path.exists():
        logging.info("DEM file not found. Downloading strategically...")
        bounds_gdf_wgs84 = polygons_gdf.to_crs("EPSG:4326")
        download_dem_for_polygons(bounds_gdf_wgs84, OPENTOPO_API_KEY, dem_path)

def _calculate_aspect_from_dem(dem_path: Path, polygons_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Clips a DEM to the extent of input polygons and calculates the terrain aspect.

    This function performs the primary raster processing by loading the DEM,
    clipping it to the area of interest, and calculating the aspect for every
    pixel within that area.

    Args:
        dem_path (Path): The path to the local DEM file.
        polygons_gdf (gpd.GeoDataFrame): The polygons used to define the clip area.

    Returns:
        tuple: A tuple containing:
            - aspect (np.ndarray): The calculated aspect raster.
            - clipped_dem (rioxarray.DataArray): The clipped DEM raster object.
            - polygons_in_dem_crs (gpd.GeoDataFrame): The input polygons reprojected
              to match the DEM's coordinate reference system.
    """
    logging.info("Processing DEM to calculate aspect...")
    with rioxarray.open_rasterio(dem_path, chunks={'x': 2048, 'y': 2048}) as dem_rds:
        polygons_in_dem_crs = polygons_gdf.to_crs(dem_rds.rio.crs)
        
        logging.info("Clipping DEM to polygon boundaries...")
        clipped_dem = dem_rds.rio.clip(polygons_in_dem_crs.geometry.apply(mapping), dem_rds.rio.crs, from_disk=True)
        
        elevation = clipped_dem.squeeze().values
        gy, gx = np.gradient(elevation)
        aspect_rad = np.arctan2(-gy, gx)
        aspect_deg = np.degrees(aspect_rad)
        aspect = (aspect_deg + 360) % 360
        
        return aspect, clipped_dem, polygons_in_dem_crs

def _merge_small_polygons(gdf: gpd.GeoDataFrame, min_area_m2: float) -> gpd.GeoDataFrame:
    """
    Finds polygons smaller than a threshold and merges them with their largest neighbor.

    This function re-projects the data to a projected CRS to ensure accurate
    area calculations in square meters.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame with polygons to clean.
        min_area_m2 (float): The minimum area in square meters to keep a polygon.

    Returns:
        gpd.GeoDataFrame: The cleaned GeoDataFrame with small polygons merged.
    """
    # Ensure the GeoDataFrame has a CRS before proceeding
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS defined for merging operations.")
    
    # Store the original CRS to return the data in the same format
    original_crs = gdf.crs

    # Reproject to a projected CRS suitable for area calculations (e.g., Web Mercator)
    gdf_proj = gdf.to_crs("EPSG:3857")
    
    # Identify polygons that are smaller than the threshold using the projected data
    small_polygons_indices = gdf_proj[gdf_proj.geometry.area < min_area_m2].index
    
    logging.info(f"Found {len(small_polygons_indices)} polygons smaller than {min_area_m2} m² to merge.")

    # Loop through the indices of the small polygons
    for index in tqdm(small_polygons_indices, desc="Merging Small Polygons"):
        # Ensure the polygon still exists (it might have been merged already)
        if index not in gdf_proj.index:
            continue
            
        small_poly_geom = gdf_proj.loc[index, 'geometry']
        
        # Find all polygons that touch the small one
        possible_neighbors = gdf_proj[gdf_proj.geometry.touches(small_poly_geom)]
        
        if not possible_neighbors.empty:
            # Find the largest neighbor by area
            largest_neighbor_index = possible_neighbors.geometry.area.idxmax()
            largest_neighbor_geom = gdf_proj.loc[largest_neighbor_index, 'geometry']
            
            # Merge the small polygon into the largest one
            merged_geom = gpd.GeoSeries([largest_neighbor_geom, small_poly_geom]).union_all()
            
            # Update the largest neighbor's geometry and drop the small one
            gdf_proj.loc[largest_neighbor_index, 'geometry'] = merged_geom
            gdf_proj.drop(index, inplace=True)
            
    # Reproject the final, cleaned dataframe back to the original CRS
    return gdf_proj.to_crs(original_crs)

def _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs) -> gpd.GeoDataFrame | None:
    """
    Vectorizes an aspect raster, intersects it with polygons, and smooths the result.

    This function handles the conversion from raster data (the aspect grid) to
    vector data (polygons). It creates polygons for each aspect class, intersects
    them with the original polygon boundaries, and applies a smoothing algorithm.

    Args:
        aspect_raster (np.ndarray): The grid of aspect values.
        clipped_dem (rioxarray.DataArray): The clipped DEM, used for transform info.
        polygons_in_dem_crs (gpd.GeoDataFrame): The original polygons in the DEM's CRS.

    Returns:
        gpd.GeoDataFrame | None: A final GeoDataFrame with smoothed,
                                 aspect-classified polygons, or None if no
                                 polygons were generated.
    """
    new_polygons = []
    original_geom_unary = polygons_in_dem_crs.unary_union
    aspect_bins = {
        "N": (315, 45), "E": (45, 135),
        "S": (135, 225), "W": (225, 315),
    }

    for aspect_name, (lower, upper) in tqdm(aspect_bins.items(), desc="Processing Aspects"):
        mask = (aspect_raster > lower) | (aspect_raster <= upper) if aspect_name == "N" else (aspect_raster > lower) & (aspect_raster <= upper)
        
        aspect_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=clipped_dem.rio.transform())
        aspect_multipolygon = MultiPolygon([shape(s) for s, v in aspect_shapes if v == 1])
        final_geom = original_geom_unary.intersection(aspect_multipolygon)
        
        if not final_geom.is_empty:
            geoms = final_geom.geoms if final_geom.geom_type == 'MultiPolygon' else [final_geom]
            new_polygons.extend({'geometry': poly, 'aspect': aspect_name} for poly in geoms)

    if not new_polygons:
        return None

    temp_gdf = gpd.GeoDataFrame(new_polygons, crs=clipped_dem.rio.crs)
    
    # The CRS must be projected (in meters) to calculate area correctly
    temp_gdf = _merge_small_polygons(temp_gdf, min_area_m2=400.0)
    
    logging.info("Smoothing polygon corners...")
    tqdm.pandas(desc="Smoothing Polygons")
    temp_gdf['geometry'] = temp_gdf['geometry'].progress_apply(lambda geom: chaikin_smooth(geom, iterations=2))

    logging.info("Simplifying geometries...")
    temp_gdf['geometry'] = temp_gdf.simplify(tolerance=0.00005, preserve_topology=True)
    
    return temp_gdf.to_crs("EPSG:4326")

def link_polygons_to_pro_files(polygons_path: Path, locations_path: Path, output_path: Path):
    """
    Finds the most relevant .pro file for each aspect polygon.

    The most relevant file is the one closest to the polygon's center that
    shares the same aspect classification.

    Args:
        polygons_path (Path): Path to the aspect-classified GeoJSON polygons.
        locations_path (Path): Path to the CSV of .pro file locations and metadata.
        output_path (Path): Path to save the new GeoJSON with linked file paths.
    """
    if not polygons_path.exists():
        logging.error(f"Aspect polygon file not found at {polygons_path}. Run aspect preparation first.")
        return

    logging.info("Linking polygons to closest .pro files by aspect...")
    polygons_gdf = gpd.read_file(polygons_path)
    locations_df = pd.read_csv(locations_path)

    # 1. Define the direct mapping from degrees to cardinal directions
    aspect_map = {
        0.0: "N", 45.0: "E", 90.0: "E", 135.0: "E",
        180.0: "S", 225.0: "W", 270.0: "W", 315.0: "W",
        "Flat": "Flat"
    }
        # 2. First, handle the non-numeric 'Flat' aspect values
    is_flat = locations_df['aspect'] == 'Flat'
    
    # 3. On the remaining numeric values, convert to float and map
    numeric_aspects = pd.to_numeric(locations_df.loc[~is_flat, 'aspect'], errors='coerce')
    locations_df.loc[~is_flat, 'aspect'] = numeric_aspects.map(aspect_map)

    # Convert locations CSV to a GeoDataFrame
    locations_gdf = gpd.GeoDataFrame(
        locations_df,
        geometry=gpd.points_from_xy(locations_df.longitude, locations_df.latitude),
        crs="EPSG:4326"
    )

    # Use a projected CRS for accurate distance calculations
    projected_crs = "EPSG:3857"
    polygons_proj = polygons_gdf.to_crs(projected_crs)
    locations_proj = locations_gdf.to_crs(projected_crs)

    polygons_proj['centroid'] = polygons_proj.geometry.centroid
    polygons_gdf['pro_file_path'] = None

    # Group by aspect and find the nearest neighbor for each group
    for aspect_name, group in tqdm(polygons_proj.groupby('aspect'), desc="Matching Aspects"):
        aspect_locations = locations_proj[locations_proj['aspect'] == aspect_name]

        if aspect_locations.empty:
            logging.warning(f"No .pro files found for aspect '{aspect_name}'. Skipping.")
            continue

        # Build a KD-Tree for fast nearest-neighbor search
        location_coords = np.array([geom.coords[0] for geom in aspect_locations.geometry])
        tree = cKDTree(location_coords)

        polygon_coords = np.array([geom.coords[0] for geom in group['centroid']])
        
        # Query the tree to find the index of the nearest location for each polygon
        _, indices = tree.query(polygon_coords, k=1)

        # Get the corresponding .pro file paths using the indices
        matched_paths = aspect_locations.iloc[indices]['path'].values

        # Assign the matched paths back to the original GeoDataFrame
        polygons_gdf.loc[group.index, 'pro_file_path'] = matched_paths

    # Remove polygons that couldn't be matched
    unmatched_count = polygons_gdf['pro_file_path'].isna().sum()
    if unmatched_count > 0:
        logging.warning(f"{unmatched_count} polygons could not be matched to a .pro file and will be removed.")
        polygons_gdf.dropna(subset=['pro_file_path'], inplace=True)

    polygons_gdf.to_file(output_path, driver='GeoJSON')
    logging.info(f"Saved {len(polygons_gdf)} linked polygons to {output_path}")


def generate_pro_file_manifest(linked_polygons_path: Path, manifest_path: Path) -> set:
    """
    Reads the linked polygons GeoJSON and creates a manifest of unique .pro files.

    Args:
        linked_polygons_path (Path): Path to the GeoJSON with linked .pro files.
        manifest_path (Path): Path to save the output manifest text file.

    Returns:
        set: A set of unique .pro file paths required for the analysis.
    """
    logging.info("Generating .pro file manifest...")
    if not linked_polygons_path.exists():
        logging.error(f"Cannot generate manifest, file not found: {linked_polygons_path}")
        return set()

    gdf = gpd.read_file(linked_polygons_path)
    unique_paths = set(gdf['pro_file_path'].dropna().unique())
    if DEV:
        unique_paths = {p.replace('/ssd/snowpack/output/2024-newhs', str(INPUT_PATH)) for p in unique_paths}    
    with open(manifest_path, 'w') as f:
        for path in sorted(list(unique_paths)):
            f.write(f"{path}\n")
    
    logging.info(f"Manifest created with {len(unique_paths)} unique .pro files at: {manifest_path}")
    return unique_paths

# --- MAIN ORCHESTRATION FUNCTION ---

def prepare_aspect_polygons(input_geojson: Path, output_geojson: Path):
    """
    Orchestrates the workflow to split input polygons by terrain aspect.
    """
    if not output_geojson.exists():
        logging.info(f"Aspect-classified GeoJSON not found. Preparing new file: {output_geojson}")
        polygons_gdf = gpd.read_file(input_geojson)
        dem_path = DEM_TIF

        # Step 1: Ensure DEM exists
        _ensure_dem_exists(polygons_gdf, dem_path)

        # Step 2: Calculate aspect raster from the DEM
        aspect_raster, clipped_dem, polygons_in_dem_crs = _calculate_aspect_from_dem(dem_path, polygons_gdf)

        # Step 3: Vectorize raster, intersect, and smooth polygons
        final_gdf = _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs)

        # Step 4: Save final results
        if final_gdf is not None and not final_gdf.empty:
            logging.info(f"Saving {len(final_gdf)} aspect-classified polygons to: {ASPECT_POLYGONS_GEOJSON}")
            final_gdf.to_file(ASPECT_POLYGONS_GEOJSON, driver='GeoJSON')
                    # Run the new linking step
