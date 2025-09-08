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
from shapely import union_all
from numba import njit
from tqdm import tqdm
from typing import Optional

from .param_config import (OPENTOPO_API_KEY,
                           ASPECT_POLYGONS_GEOJSON, DEM_DATASETS, DEM_TIF, 
                           INPUT_POLYGONS_GEOJSON, LINKED_POLYGONS_GEOJSON,
                           PRO_FILE_MANIFEST, SNOWPACK_LOCATIONS_CSV)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@njit
def _chaikin_iteration(coords, ratio=0.25):
    """
    Performs a single iteration of Chaikin's corner-cutting algorithm.
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
    Strategically downloads and mosaics only the DEM tiles that intersect with polygons.
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
    Removes small sliver polygons based on absolute area and relative area percentage.
    """
    if 'original_area' not in gdf.columns:
        logging.warning("Missing 'original_area' column. Cannot perform ratio-based filtering.")
        return gdf

    initial_count = len(gdf)
    
    # The CRS should already be projected from previous steps, allowing for area calculation.
    if gdf.crs.is_geographic:
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
    """
    if not dem_path.exists():
        logging.info("DEM file not found. Downloading strategically...")
        bounds_gdf_wgs84 = polygons_gdf.to_crs("EPSG:4326")
        download_dem_for_polygons(bounds_gdf_wgs84, OPENTOPO_API_KEY, dem_path)

def _calculate_aspect_from_dem(dem_path: Path, polygons_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Clips a DEM to the extent of input polygons and calculates the terrain aspect.
    """
    logging.info("Processing DEM to calculate aspect...")
    with rioxarray.open_rasterio(dem_path, chunks={'x': 2048, 'y': 2048}) as dem_rds:
        polygons_in_dem_crs = polygons_gdf.to_crs(dem_rds.rio.crs)
        
        logging.info("Clipping DEM to polygon boundaries...")
        clipped_dem = dem_rds.rio.clip(polygons_in_dem_crs.geometry.apply(mapping), dem_rds.rio.crs, from_disk=True)
        
        elevation = clipped_dem.squeeze().values
        gy, gx = np.gradient(elevation)
        aspect_rad = np.arctan2(gy, -gx)
        
        aspect_deg_trig = np.degrees(aspect_rad)
        aspect_deg_cart = (90.0 - aspect_deg_trig + 360) % 360
        
        return aspect_deg_cart, clipped_dem, polygons_in_dem_crs

def _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorizes an aspect raster, intersects, filters, and cleans the result.
    """
    # Calculate the area of the original polygons for ratio-based filtering.
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


def prepare_aspect_polygons(input_geojson: Path, output_geojson: Path):
    """
    Orchestrates the workflow to split input polygons by terrain aspect.
    """
    if output_geojson.exists():
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
        
def _extract_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs) -> Optional[gpd.GeoDataFrame]:
    """
    Extracts polygons for each aspect bin, intersecting them with each source polygon
    individually to prevent creating overlapping features.
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


def _build_aspect_mask(aspect_raster, lower: float, upper: float, aspect_name: str) -> np.ndarray:
    """Create a boolean mask for the aspect bin."""
    if aspect_name == "N":
        return (aspect_raster > lower) | (aspect_raster <= upper)
    return (aspect_raster > lower) & (aspect_raster <= upper)


def _vectorize_aspect(mask: np.ndarray, clipped_dem) -> list:
    """Convert raster mask into shapely geometries."""
    aspect_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=clipped_dem.rio.transform())
    return [shape(s) for s, v in aspect_shapes if v == 1]


def _filter_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only Polygon and MultiPolygon geometries."""
    initial_count = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    removed_count = initial_count - len(gdf)
    if removed_count > 0:
        logging.info(f"Removed {removed_count} non-polygonal geometries (e.g., points or lines).")
    return gdf


def _postprocess_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Smooth, simplify, and repair geometries."""
    
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

