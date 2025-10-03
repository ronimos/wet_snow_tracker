"""
main.py
=======

This script serves as the main entry point and orchestrator for the Wetting 
Front Tracker application. It manages the end-to-end workflow, from command-line
argument parsing and geospatial data preparation to the parallelized analysis of
snowpack files and the final generation of a summary map.

Workflow Overview:
------------------
1.  **Initialization:** The script begins by parsing command-line arguments, which
    allow the user to specify a central analysis date, force data regeneration,
    or define a custom time window for the analysis. It establishes the primary
    time window (e.g., 7 days before and 72 hours after the central date).

2.  **Geodata Preparation (Conditional):** If the `--regenerate-data` flag is used,
    or if essential processed geodata files are missing, it triggers the
    `prepare_geodata` module. This step downloads DEMs, splits input avalanche
    path polygons by terrain aspect, and links each resulting polygon to its
    most relevant SNOWPACK (.pro) model output file.

3.  **Task Generation:** It reads the `linked_aspect_polygons.geojson` file, which
    contains the geometries and the path to the corresponding .pro file for each
    polygon to be analyzed. It creates a list of tasks, with each task
    containing the necessary information to process one polygon.

4.  **Parallel Snowpack Analysis:** Using Python's `multiprocessing` library, the
    script distributes the analysis tasks across all available CPU cores. For
    each polygon, a worker process:
    a. Reads the linked .pro file into a `SnowpackProfile` object.
    b. Calculates a time series of key snowpack metrics (e.g., weak layer
       height, wetting front depth, total snow depth).
    c. Applies a persistence logic to track the primary weak layer (LOC)
       through melt events.
    d. Calculates the final `time_to_loc` metric: the time (in hours) for the
       wetting front to reach the weak layer relative to the central date.
    e. Generates a static Matplotlib plot and an interactive Plotly plot for
       the analysis time window.

5.  **Aggregation and Final Visualization:** After all worker processes are complete,
    the main script collects the results. It merges the analysis results (like
    `time_to_loc`) back into the GeoDataFrame and calls the `plotting` module to
    create the final `summary_map.html`. This map displays all polygons,
    color-coded by their risk level, with tooltips and links to the detailed plots.

Usage:
------
- To run with default settings:
  `python -m src.wetting_front_tracker.main`
- To specify a central date:
  `python -m src.wetting_front_tracker.main --date YYYY-MM-DD`
- To force regeneration of all geodata:
  `python -m src.wetting_front_tracker.main --regenerate-data`
"""
import argparse
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import json

from .param_config import (ASSETS_PATH, INPUT_POLYGONS_GEOJSON,
                           INPUT_POLYGONS_GEOJSON_TEST, ASPECT_POLYGONS_GEOJSON,
                           LINKED_POLYGONS_GEOJSON, PRO_FILES_BASE_PATH,
                           PRO_FILES_SOURCE, REMOTE_PRO_FILES_URL, RESULTS_PATH,
                           SNOWPACK_LOCATIONS_CSV, USE_TEST_DATA,
                           PRO_FILE_MANIFEST,)
from .plotting import (create_folium_map, plot_summary_matplotlib, 
                       plot_summary_plotly)
from .prepare_geodata import (link_polygons_to_pro_files,
                              prepare_aspect_polygons)
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                get_total_snow_depth, lwc_above_weak,
                                wet_front_lwc, find_wet_slab_loc_bottom_half)


def generate_pro_file_manifest(base_path: Path, manifest_path: Path):
    """Recursively scans a directory for .pro files and saves their paths to a manifest file.

    The manifest is a JSON object that maps a simple filename (e.g., "station.pro") 
    to its full, absolute path. This allows for quick lookups without needing to 
    re-scan the entire filesystem on every run.

    Args:
        base_path (Path): The root directory to start the recursive scan from.
        manifest_path (Path): The full path where the output JSON manifest 
                              file will be saved.
    """
    logging.info(f"Scanning for .pro files under {base_path}...")
    # Use rglob to find all files ending with .pro in all subdirectories
    pro_files = list(base_path.rglob('*.pro'))
    
    # Create a dictionary of {filename: /full/path/to/file.pro}
    manifest = {file.name: str(file.resolve()) for file in pro_files}
    
    # Write the dictionary to the specified JSON file
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    logging.info(f"Pro file manifest with {len(manifest)} entries saved to {manifest_path}")
    
    
def ensure_pro_file_is_local(file_name: str, local_input_path: Path, remote_base_url: str, central_date: datetime):
    """
    Checks if a .pro file exists locally and is fresh. If not, downloads it.
    This is a placeholder for your actual download logic (e.g., from S3, HTTP).
    """
    local_file_path = local_input_path / file_name
    
    # 1. Check if file exists and is fresh (less than 12 hours old)
    if local_file_path.exists():
        mod_time_ts = os.path.getmtime(local_file_path)
        mod_time_dt = datetime.fromtimestamp(mod_time_ts, tz=timezone.utc)
        central_date_utc = central_date.replace(tzinfo=timezone.utc)
        if (central_date_utc - mod_time_dt) < timedelta(hours=12):
            logging.debug(f"'{file_name}' is fresh. Skipping download.")
            return  # File is fresh, no need to download

    # 2. If we reach here, the file is either missing or stale, so download it.
    logging.info(f"Downloading '{file_name}'...")
    remote_file_url = f"{remote_base_url.rstrip('/')}/{file_name}"
    
    # --- !!! ADD YOUR DOWNLOAD LOGIC HERE !!! ---
    # Example for S3 using boto3:
    # import boto3
    # s3 = boto3.client('s3')
    # bucket_name = "my-bucket"
    # object_key = f"pro-files/{file_name}"
    # s3.download_file(bucket_name, object_key, str(local_file_path))

    # Example for HTTP using requests:
    # import requests
    # r = requests.get(remote_file_url, stream=True)
    # if r.status_code == 200:
    #     with open(local_file_path, 'wb') as f:
    #         for chunk in r.iter_content(chunk_size=8192):
    #             f.write(chunk)
    # else:
    #     logging.error(f"Failed to download {file_name}. Status: {r.status_code}")
    
    # For now, we'll just log a placeholder message.
    logging.warning(f"Placeholder: Pretending to download from {remote_file_url} to {local_file_path}")
    # You would create a dummy file for testing if needed
    # local_file_path.touch()
    
    
def _initialize_and_validate_profile(pro_file_path: Path, aspect: str) -> tuple[SnowpackProfile | None, str | None]:
    """
    Initializes a SnowpackProfile object and validates its data.

    This helper function loads a .pro file, creates a unique file stem for
    output files based on the file name and aspect, and checks if the profile
    contains valid, timestamped data.

    Args:
        pro_file_path: The path to the .pro input file.
        aspect: The aspect of the polygon being processed (e.g., 'N', 'E').

    Returns:
        A tuple containing the initialized SnowpackProfile object and a unique
        file stem, or (None, None) if the profile data is invalid or missing.
    """
    profile = SnowpackProfile(pro_file_path)
    file_stem = f"{pro_file_path.stem}_{aspect}"
    profile.metadata['aspect'] = aspect

    if profile.data is None or 'timestamp' not in profile.data.coords:
        logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
        return None, None
    return profile, file_stem

def _unpack_and_prepare_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpacks tuple columns from the summary and ensures correct data types.

    The initial summary from `get_profile_summary` may contain columns where
    each cell is a tuple (e.g., (value, height)). This function unpacks these
    tuples into separate columns and ensures that key columns used in
    calculations are converted to a numeric type.

    Args:
        summary_df: The raw summary DataFrame from the profile analysis.

    Returns:
        A prepared DataFrame with unpacked columns and appropriate numeric
        data types, ready for further analysis.
    """
    rename_map = {
        "weak_layer_value": "weak_layer_gs_diff",
        # "wet_front_lwc_value": "wet_front_lwc_value", # This is redundant
    }
    summary_df.rename(columns=rename_map, inplace=True)

    numeric_cols = ['weak_layer_height', 'wet_front_lwc_height', 'hs']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    
    return summary_df

def _persist_loc_height(summary_df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """
    Identifies the primary weak layer just before the most recent melt event
    and carries its height forward dynamically.

    This function finds the start of the most recent melt event relative to a
    reference date, locks onto the last known weak layer before it, and then
    tracks that layer. If a new weak layer is detected at a higher elevation,
    the lock is updated to that new, higher layer.

    Args:
        summary_df: The prepared summary DataFrame.
        reference_date: The central date for the analysis (e.g., today).

    Returns:
        A DataFrame with `weak_layer_height` adjusted for persistence.
    """
    if 'weak_layer_height' not in summary_df.columns or 'wet_front_lwc_height' not in summary_df.columns:
        return summary_df

    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]

    relevant_start_times = all_start_times[all_start_times <= reference_date]

    if relevant_start_times.empty:
        return summary_df
    
    trigger_time = relevant_start_times[-1]

    lookback_window_end = trigger_time
    lookback_window_start = lookback_window_end - timedelta(days=2)
    pre_melt_df = summary_df.loc[lookback_window_start:lookback_window_end]
    
    # FIX: Check if the filtered DataFrame is empty before accessing iloc[-1]
    valid_pre_melt_locs = pre_melt_df['weak_layer_height'].dropna()
    initial_lock_height = np.nan if valid_pre_melt_locs.empty else valid_pre_melt_locs.iloc[-1]

    if pd.isna(initial_lock_height):
        return summary_df
        
    persisted_loc = summary_df['weak_layer_height'].copy()
    wet_season_mask: pd.Series = summary_df.index >= trigger_time
    
    wet_loc_series: pd.Series = summary_df.loc[wet_season_mask, 'weak_layer_height']
    anchored_series = pd.concat([pd.Series([initial_lock_height]), wet_loc_series.reset_index(drop=True)])
    
    running_max_loc = anchored_series.cummax().iloc[1:].values
    
    persisted_loc.loc[wet_season_mask] = running_max_loc

    persisted_loc_filled = persisted_loc.ffill()

    persisted_loc_filled[summary_df['hs'] < persisted_loc_filled] = np.nan

    summary_df['weak_layer_height'] = persisted_loc_filled
    
    return summary_df

def process_single_profile(pro_file_path: Path, 
                           aspect: str, 
                           start_date_arg: str | None = None, 
                           end_date_arg: str | None = None, 
                           central_date_arg: datetime | None = None,
                           assets_path: Path | None = None) -> dict[str, Any] | None:
    """
    Handles the full analysis workflow for a single polygon and its linked .pro file.
    
    This is the core analysis function that is parallelized. It orchestrates
    the reading of a snowpack file, running various analyses on it, applying
    the LOC persistence logic, generating plots, and calculating the final
    `time_to_loc` metric.

    Args:
        pro_file_path: The path to the .pro input file.
        aspect: The aspect ('N', 'E', 'S', 'W', 'Flat') of the polygon being
                processed.
        start_date_arg: The start date for the analysis window, used for
                        the Matplotlib plot's visible range.
        end_date_arg: The end date for the analysis window.
        central_date_arg: The central reference date for the `time_to_loc`
                          calculation and for the vertical line on the plot.
        assets_path: The directory where output plots should be saved.

    Returns:
        A dictionary containing results for the final summary map (station name,
        file_stem, time_to_loc), or None if processing fails.
    """
    try:
        profile, file_stem = _initialize_and_validate_profile(pro_file_path, aspect)
        if not profile or not profile.data or not file_stem:
            return None
        
        if central_date_arg:
            min_date_in_data = central_date_arg - timedelta(days=7)
            max_date_in_data = central_date_arg + timedelta(hours=72)
        else:
            # Use the full time range from the data for the analysis
            min_date_in_data = pd.to_datetime(profile.data.timestamp.values[0])
            max_date_in_data = pd.to_datetime(profile.data.timestamp.values[-1])

        # MODIFICATION: Use the high-resolution summary function
        raw_summary = profile.get_full_timeseries_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, 
                "weak_layer": find_wet_slab_loc_bottom_half,
                "wet_front_lwc": wet_front_lwc,
                "highest_wet_point": get_highest_wet_point,
                "lwc_above_weak": lambda df: lwc_above_weak(df, find_wet_slab_loc_bottom_half)
            },
            start_date=str(min_date_in_data),
            end_date=str(max_date_in_data),
        ).copy()
        
        if raw_summary.empty:
            return None

        prepared_summary = _unpack_and_prepare_summary(raw_summary)
        
        # Apply the robust persistence logic
        referance_date = central_date_arg or datetime.now()
        summary_full = _persist_loc_height(prepared_summary, referance_date)

        # Generate plots and calculate final metrics
        
        # --- Data Slicing for Plots ---
        # Explicit boolean masking for robust filtering.
        if start_date_arg is None or end_date_arg is None:
            logging.error("Analysis window start/end dates are missing. Cannot create plots.")
            return None # Can't proceed without a valid window
        
        start_dt, end_dt = pd.to_datetime(start_date_arg), pd.to_datetime(end_date_arg)
        
        # Data for line plots (daily summary)
        is_in_window = (summary_full.index >= start_dt) & (summary_full.index <= end_dt)
        summary_for_plot = summary_full[is_in_window]

        # Data for LWC colormesh (potentially higher temporal resolution)
        # Select only the needed variables for efficiency
        full_season_plot_data = profile.data[['lwc', 'height']]
        is_in_lwc_window = (full_season_plot_data.timestamp >= start_dt) & (full_season_plot_data.timestamp <= end_dt)
        lwc_data_for_plot = full_season_plot_data.sel(timestamp=is_in_lwc_window)

        station_metadata = profile.metadata
        del profile


        if not summary_for_plot.empty:
            plot_summary_matplotlib(summary_for_plot, file_stem, station_metadata, lwc_data_for_plot, central_date_arg, assets_path)
            plot_summary_plotly(summary_full, file_stem, station_metadata, central_date_arg, assets_path)
            
        else:
            logging.warning(
                f"No snowpack data found for {file_stem} in the analysis window "
                f"({start_date_arg} to {end_date_arg}). Plots will be skipped."
            )

        time_to_loc = find_time_to_loc(summary_full, reference_date=referance_date)

        return {
            "station_name": station_metadata.get('stationName', file_stem),
            "file_stem": file_stem,
            "time_to_loc": time_to_loc,
            "central_date_str": central_date_arg.strftime('%Y-%m-%d %H:%M') if central_date_arg else None
        }

    except Exception as e:
        logging.error(f"Error processing {pro_file_path.name} for aspect {aspect}: {e}", exc_info=True)
        return None

def worker_wrapper(task_tuple: tuple) -> dict[str, Any] | None:
    """
    Wrapper function to enable multiprocessing by unpacking arguments.

    This function simply unpacks a tuple of arguments and passes them to the
    main `process_single_profile` function. It is used as the target for the
    multiprocessing pool.

    Args:
        task_tuple: A tuple containing the arguments required by
                    `process_single_profile`.

    Returns:
        The result dictionary from `process_single_profile`, or None if an
        error occurred.
    """
    return process_single_profile(*task_tuple)

def _get_closest_synoptic_time(reference_time: datetime) -> datetime:
    """
    Finds the closest standard synoptic time (00, 06, 12, 18 UTC) to a given datetime.

    This ensures that the analysis is centered on a standard meteorological
    reporting time, providing consistency.

    Args:
        reference_time (datetime): The input time (e.g., current time or a
                                     user-specified time).

    Returns:
        datetime: The datetime object representing the closest synoptic time.
    """
    base_date = reference_time.date()
    # Create candidate times on the same day as the reference time
    candidates = [
        datetime.combine(base_date, datetime.min.time()).replace(hour=h)
        for h in [0, 6, 12, 18]
    ]
    # To be thorough, also check the last synoptic time of the previous day
    # and the first of the next day.
    candidates.insert(0, candidates[0] - timedelta(hours=6))
    candidates.append(candidates[1] + timedelta(days=1))

    # Find the candidate with the minimum absolute time difference
    return min(candidates, key=lambda dt: abs(reference_time - dt))

def parse_args() -> argparse.Namespace:
    """
    Sets up and parses command-line arguments for the script.

    Defines arguments for controlling the analysis, such as forcing data
    regeneration and setting the analysis time window.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force regeneration of all processed data."
    )
    parser.add_argument(
        "-d", "--date", dest="central_date",
        help="Central date and time for analysis (e.g., 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'). "
             "Rounds to the closest synoptic time (00, 06, 12, 18).",
             default="2025-05-09 12:00"  # Default to a future date for demonstration
    )
    parser.add_argument("-s", "--start", dest="start_date", 
                        help="Start date for analysis (overrides default window)."
    )
    parser.add_argument("-e", "--end", dest="end_date", 
                        help="End date for analysis (overrides default window)."
    )
    parser.add_argument("-i", "--input-dir", dest="input_dir", 
                        type=Path, default=None, help="Override default base directory for .pro files."
    )
    parser.add_argument("-o", "--output-dir", dest="output_dir", 
                        type=Path, default=None, help="Override default directory for the final map."
    )
    parser.add_argument("-a", "--assets-dir", dest="assets_dir", 
                        type=Path, default=None, help="Override default directory for plot assets."
    )
    
    return parser.parse_args()


def main():
    """
    Main orchestrator for the entire analysis and mapping workflow.

    This function handles argument parsing, date setup, geodata preparation,
    and the parallel processing of snowpack files before generating the
    final summary map.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="wetting_front_tracker.log",
        filemode="w"   # overwrite each run; use "a" to append
    )    
    
    args = parse_args()
    # --- Path Configuration ---
    input_path = args.input_dir or PRO_FILES_BASE_PATH
    output_path = args.output_dir or RESULTS_PATH
    assets_path = args.assets_dir or ASSETS_PATH
    output_path.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)
    summary_map_path = output_path / "summary_map.html"
    logging.info(f"Input .pro directory: {input_path}")
    logging.info(f"Output map directory: {output_path}")
    logging.info(f"Plot assets directory: {assets_path}")

    # --- Date Handling (Single Day) ---
    if args.central_date:
        try:
            # First, try parsing the full date and time
            initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d %H:%M')
        except ValueError:
            try:
                # If that fails, try parsing with date only
                initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid date format for '{args.central_date}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")
                return
    else:
        # If no date is provided, use the current time
        initial_ref_time = datetime.now()

    central_date = _get_closest_synoptic_time(initial_ref_time)
    start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Processing for central date: {central_date.strftime('%Y-%m-%d %H:%M')}")

    # --- Geodata Preparation ---
    input_geojson = INPUT_POLYGONS_GEOJSON_TEST if USE_TEST_DATA else INPUT_POLYGONS_GEOJSON
    if args.regenerate_data or not LINKED_POLYGONS_GEOJSON.exists():
        logging.info("Regenerating all processed data and pro file manifest...")
        
        # First, generate the manifest so the geo-linking can use it if needed
        generate_pro_file_manifest(input_path, PRO_FILE_MANIFEST)
        
        prepare_aspect_polygons(input_geojson, ASPECT_POLYGONS_GEOJSON, args.regenerate_data)
        link_polygons_to_pro_files(ASPECT_POLYGONS_GEOJSON, SNOWPACK_LOCATIONS_CSV, LINKED_POLYGONS_GEOJSON)

    try:
        linked_gdf = gpd.read_file(LINKED_POLYGONS_GEOJSON)
        with open(PRO_FILE_MANIFEST, 'r') as f:
            pro_file_manifest = json.load(f)
        logging.info(f"Successfully loaded pro file manifest with {len(pro_file_manifest)} entries.")
    except FileNotFoundError:
        logging.error(f"Manifest file not found at {PRO_FILE_MANIFEST}. Please run with --regenerate-data.")
        return

    # --- NEW: Conditional Data Download Step ---
    if PRO_FILES_SOURCE == 'remote':
        logging.info("Checking for remote .pro files to download...")
        # Create a unique set of filenames to check
        required_files = {Path(p).name for p in linked_gdf['pro_file_path'].unique()}
        for file_name in tqdm(required_files, desc="Updating data files"):
            ensure_pro_file_is_local(file_name, input_path, REMOTE_PRO_FILES_URL, central_date)
    else:
        logging.info("Skipping remote file check. PRO_FILES_SOURCE is 'local'.")

    # --- Task Generation (Single Day) ---
    tasks = []
    for poly in linked_gdf.itertuples(index=False):
# Get the filename from the geodataframe
        file_name = Path(str(poly.pro_file_path)).name
        
        # Look up the full path from our manifest dictionary
        if full_pro_path_str := pro_file_manifest.get(file_name):
            effective_path = Path(full_pro_path_str)
            tasks.append((effective_path, poly.aspect, start_date, end_date, central_date, assets_path))
        else:
            logging.warning(f"File '{file_name}' from geojson not found in the manifest. Skipping.")
    if not tasks:
        logging.warning("No tasks were generated for processing. Exiting.")
        return

    # --- Parallel Processing ---
    logging.info(f"Starting parallel processing on {len(tasks)} polygons...")
    cpu_cores = os.cpu_count()
    worker_count = int(max(1, cpu_cores / 4 )) if cpu_cores else 1 
    with multiprocessing.Pool(processes=worker_count) as pool:
        results = list(tqdm(pool.map(worker_wrapper, tasks, chunksize=1), total=len(tasks)))

    # --- Aggregation and Final Map ---
    if valid_results := [res for res in results if res is not None]:
        results_df = pd.DataFrame(valid_results)
        final_gdf = gpd.GeoDataFrame(pd.concat([
            linked_gdf.reset_index(drop=True),
            results_df.reset_index(drop=True)
        ], axis=1))
        
        logging.info("All polygons processed. Creating summary map...")
        create_folium_map(final_gdf, summary_map_path, central_date, assets_path )
    else:
        logging.info("No valid results were generated. Skipping map creation.")


if __name__ == "__main__":
    main()

