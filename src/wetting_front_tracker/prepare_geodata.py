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

def chaikin_smooth(geometry, iterations=2):
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
    base_url = dataset['url']
    west, south, east, north = bounds
    params = {'datasetName': dataset['name'], 'south': south, 'north': north,
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
        bb = dem['bbox']
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


def _merge_small_polygons(gdf: gpd.GeoDataFrame, min_area_m2: float, min_area_ratio: float) -> gpd.GeoDataFrame:
    """
    Finds and merges small/insignificant polygons with their largest neighbor.
    """
    if 'original_id' not in gdf.columns or 'original_area' not in gdf.columns:
        logging.warning("Missing 'original_id' or 'original_area' columns. Cannot perform advanced merge.")
        return gdf

    processed_gdf = gdf.copy()
    processed_gdf['current_area'] = processed_gdf.geometry.area
    
    area_condition = processed_gdf['current_area'] < min_area_m2
    ratio_condition = (processed_gdf['current_area'] / processed_gdf['original_area']) < min_area_ratio
    
    small_polygons_indices = processed_gdf[area_condition | ratio_condition].index
    
    logging.info(f"Found {len(small_polygons_indices)} small/insignificant polygons to merge.")

    for index in tqdm(small_polygons_indices, desc="Merging Small Polygons"):
        if index not in processed_gdf.index:
            continue
            
        small_poly_row = processed_gdf.loc[index]
        small_poly_geom = small_poly_row.geometry
        original_id = small_poly_row.original_id
        
        possible_neighbors = processed_gdf[
            (processed_gdf.index != index) &
            (processed_gdf.geometry.touches(small_poly_geom)) &
            (processed_gdf['original_id'] == original_id)
        ]
        
        if not possible_neighbors.empty:
            largest_neighbor_index = possible_neighbors.geometry.area.idxmax()
            largest_neighbor_geom = processed_gdf.loc[largest_neighbor_index, 'geometry']
            
            merged_geom = gpd.GeoSeries([largest_neighbor_geom, small_poly_geom]).unary_union
            
            processed_gdf.loc[largest_neighbor_index, 'geometry'] = merged_geom
            processed_gdf.drop(index, inplace=True)
            
    return processed_gdf.drop(columns=['current_area'])


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
        aspect_rad = np.arctan2(-gy, gx)
        aspect_deg = np.degrees(aspect_rad)
        aspect = (aspect_deg + 360) % 360
        
        return aspect, clipped_dem, polygons_in_dem_crs

def _process_aspect_polygons(aspect_raster, clipped_dem, polygons_in_dem_crs) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorizes an aspect raster, intersects it with polygons, and cleans the result.
    """
    polygons_in_dem_crs['original_id'] = range(len(polygons_in_dem_crs))
    polygons_in_dem_crs['original_area'] = polygons_in_dem_crs.geometry.area

    all_aspect_polys = []
    aspect_bins = {
        "N": (315, 45), "E": (45, 135),
        "S": (135, 225), "W": (225, 315),
    }

    # Use the faster gpd.union_all()
    original_geom_unary = gpd.union_all(polygons_in_dem_crs.geometry)

    for aspect_name, (lower, upper) in tqdm(aspect_bins.items(), desc="Processing Aspects"):
        mask = (aspect_raster > lower) | (aspect_raster <= upper) if aspect_name == "N" else (aspect_raster > lower) & (aspect_raster <= upper)
        
        aspect_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=clipped_dem.rio.transform())
        aspect_geoms = [shape(s) for s, v in aspect_shapes if v == 1]
        if not aspect_geoms:
            continue
        
        # Use the faster shapely.union_all()
        aspect_multipolygon = union_all(aspect_geoms)

        final_geom = original_geom_unary.intersection(aspect_multipolygon)
        
        if not final_geom.is_empty:
            geoms = final_geom.geoms if final_geom.geom_type == 'MultiPolygon' else [final_geom]
            for poly in geoms:
                all_aspect_polys.append({
                    'geometry': poly, 
                    'aspect': aspect_name,
                })

    if not all_aspect_polys:
        return None
        
    aspect_gdf = gpd.GeoDataFrame(all_aspect_polys, crs=clipped_dem.rio.crs)
    final_gdf = gpd.sjoin(aspect_gdf, polygons_in_dem_crs, how="inner", predicate="intersects")
    
    final_gdf = _merge_small_polygons(final_gdf, min_area_m2=400.0, min_area_ratio=0.20)
    
    logging.info("Smoothing polygon corners...")
    tqdm.pandas(desc="Smoothing Polygons")
    final_gdf['geometry'] = final_gdf['geometry'].progress_apply(lambda geom: chaikin_smooth(geom, iterations=2))
    
    # --- FIX: Add a zero-buffer operation to repair invalid geometries ---
    logging.info("Repairing any invalid geometries...")
    final_gdf['geometry'] = final_gdf.geometry.buffer(0)

    logging.info("Simplifying geometries to reduce file size...")
    final_gdf['geometry'] = final_gdf.simplify(tolerance=10, preserve_topology=True)

    final_gdf = final_gdf.drop(columns=['original_id', 'original_area', 'index_right'])

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

    projected_crs = "EPSG:3857"
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

