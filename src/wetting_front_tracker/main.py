"""
main.py

This script serves as the main entry point for the Wetting Front Tracker application.
It orchestrates the entire workflow, from preparing geospatial data to analyzing
snowpack files and generating a final summary map.

The workflow consists of these main steps:
1.  **Geodata Preparation (Optional):** If the `--regenerate-data` flag is used
    or if processed files are missing, the script will:
    a. Download and mosaic Digital Elevation Model (DEM) tiles.
    b. Split input avalanche path polygons by terrain aspect (N, E, S, W).
    c. Filter out small, insignificant polygon fragments.
    d. Link the final polygons to the most relevant SNOWPACK (.pro) model output
       files based on location and aspect.
    e. Generate a manifest of all .pro files required for the analysis.

2.  **Snowpack Analysis:** The script reads the manifest of .pro files and processes
    each one in parallel to speed up the analysis. For each file, it:
    a. Extracts snowpack properties over time (e.g., total depth, weak layer height).
    b. Tracks the depth of the wetting front (where Liquid Water Content > 3%).
    c. Calculates the `time_to_loc`: the time (in hours) for the wetting front
       to reach the identified weak layer, relative to a central analysis date.

3.  **Visualization:** For each processed .pro file, the script generates:
    a. An interactive HTML plot (Plotly) showing the full time-series analysis.
    b. A static PNG plot (Matplotlib) showing a specific time window.
    Finally, it creates a single summary map (`summary_map.html`) that displays all
    the processed avalanche polygons, colored by their `time_to_loc` to indicate risk.

Usage:
    To run the full analysis with default settings (last 7 days to next 72 hours):
    $ python -m src.wetting_front_tracker.main

    To run for a specific central date:
    $ python -m src.wetting_front_tracker.main --date YYYY-MM-DD

    To force regeneration of all geospatial data:
    $ python -m src.wetting_front_tracker.main --regenerate-data
"""
import argparse
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

from .param_config import (
    ASPECT_POLYGONS_GEOJSON, INPUT_POLYGONS_GEOJSON, LINKED_POLYGONS_GEOJSON,
    PRO_FILE_MANIFEST, SNOWPACK_LOCATIONS_CSV, SUMMARY_MAP_HTML,
    IS_DEV_ENVIRONMENT, PRO_FILES_BASE_PATH_PROD, PRO_FILES_BASE_PATH_DEV
)
from .plotting import create_folium_map, plot_summary_matplotlib, plot_summary_plotly
from .prepare_geodata import (generate_pro_file_manifest,
                              link_polygons_to_pro_files,
                              prepare_aspect_polygons)
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                get_total_snow_depth,
                                largest_fc_dh_gs_diff_bottom_half,
                                lwc_above_weak, wet_front_form, wet_front_lwc)


def process_single_profile(pro_file_path: Path, start_date_arg: str | None = None, end_date_arg: str | None = None, central_date_arg: datetime | None = None) -> dict[str, Any] | None:
    """
    Handles the full analysis workflow for a single SNOWPACK .pro file.

    This function reads a .pro file, calculates a time-series summary of key
    snowpack metrics, persists the weak layer height through wetting events,
    generates plots, and calculates the final time-to-loc for map coloring.

    Args:
        pro_file_path: The path to the .pro input file.
        start_date_arg: The start date for the analysis window (for plotting).
        end_date_arg: The end date for the analysis window (for plotting).
        central_date_arg: The central reference date for `time_to_loc` calculation.

    Returns:
        A dictionary containing the results needed for the summary map
        (e.g., station name, coordinates, time_to_loc), or None if processing fails.
    """
    try:
        profile = SnowpackProfile(pro_file_path)
        file_stem = pro_file_path.stem
        station_name = profile.metadata.get('stationName', file_stem)

        if profile.data is None or 'timestamp' not in profile.data.coords:
            logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
            return None

        time_coords = pd.to_datetime(profile.data.timestamp.values)
        min_date_in_data = time_coords.min()
        max_date_in_data = time_coords.max()

        def calculate_lwc_at_interface(df: pd.DataFrame):
            return lwc_above_weak(df, weak_layer_func=largest_fc_dh_gs_diff_bottom_half)

        # 1. Get the summary for the FULL date range for Plotly and analysis
        summary_full = profile.get_profile_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_by_grain": wet_front_form, "wet_front_by_lwc": wet_front_lwc,
                "lwc_at_interface": calculate_lwc_at_interface, "highest_wet_point": get_highest_wet_point,
            },
            start_date=min_date_in_data.strftime('%Y-%m-%d'),
            end_date=max_date_in_data.strftime('%Y-%m-%d'),
        ).copy()

        if summary_full.empty:
            return None

        # Unpack tuple columns for both plots
        tuple_cols = {
            "weak_layer": ['weak_layer_gs_diff', 'weak_layer_height'],
            "wet_front_by_grain": ['wet_front_grain_type', 'wet_front_grain_height'],
            "wet_front_by_lwc": ['wet_front_lwc_value', 'wet_front_lwc_height'],
            "lwc_at_interface": ['interface_lwc_value', 'interface_lwc_height'],
        }
        for col, new_cols in tuple_cols.items():
            if col in summary_full and summary_full[col].notna().any():
                unpacked = pd.DataFrame(summary_full[col].dropna().tolist(), index=summary_full[col].dropna().index)
                unpacked.columns = new_cols
                summary_full = summary_full.join(unpacked)

        # Ensure relevant columns are numeric before performing calculations.
        numeric_cols = ['weak_layer_height', 'wet_front_lwc_height', 'hs']
        for col in numeric_cols:
            if col in summary_full.columns:
                summary_full[col] = pd.to_numeric(summary_full[col], errors='coerce')

        # --- Logic to Persist LOC Height ---
        for i in range(1, len(summary_full)):
            prev_idx = summary_full.index[i - 1]
            curr_idx = summary_full.index[i]

            previous_loc_height = summary_full.loc[prev_idx, 'weak_layer_height']
            current_wet_front = summary_full.loc[curr_idx, 'wet_front_lwc_height']
            current_hs = summary_full.loc[curr_idx, 'hs']

            if pd.notna(previous_loc_height) and pd.notna(current_wet_front):
                if current_wet_front <= previous_loc_height:
                    if pd.notna(current_hs) and current_hs >= previous_loc_height:
                        summary_full.loc[curr_idx, 'weak_layer_height'] = previous_loc_height
                    else:
                        summary_full.loc[curr_idx, 'weak_layer_height'] = np.nan

        # 2. Generate the interactive Plotly plot using the full data range
        plot_summary_plotly(summary_full, file_stem, profile.metadata)

        # 3. Filter the full summary to get the data for the Matplotlib plot's time window
        summary_matplotlib = summary_full.loc[start_date_arg:end_date_arg]

        # 4. Generate the static Matplotlib plot for the specific time window
        if not summary_matplotlib.empty:
            plot_summary_matplotlib(summary_matplotlib, file_stem, profile.metadata, central_date=central_date_arg)
        else:
            logging.warning(f"No data available in the Matplotlib date range for {file_stem}")

        # 5. Perform final analysis on the full summary to get the most accurate result
        time_to_loc = find_time_to_loc(summary_full, reference_date=central_date_arg)

        return {
            "station_name": station_name,
            "file_stem": file_stem,
            "lat": float(profile.metadata.get('latitude', 0)),
            "lon": float(profile.metadata.get('longitude', 0)),
            "time_to_loc": time_to_loc,
        }

    except Exception as e:
        logging.error(f"Error processing {pro_file_path.name}: {e}", exc_info=True)
        return None

def worker_wrapper(task_tuple: tuple) -> dict[str, Any] | None:
    """
    A wrapper function to enable multiprocessing.

    It unpacks a tuple of arguments and calls the main `process_single_profile`
    function. It also attaches the original (production) file path to the
    result dictionary for later use in mapping.

    Args:
        task_tuple: A tuple containing the effective file path to process,
                    the original production file path, start date, end date,
                    and central date.

    Returns:
        The result dictionary from `process_single_profile`, augmented with
        the `pro_file_path`, or None if processing failed.
    """
    effective_path, original_path, start_date, end_date, central_date = task_tuple
    result = process_single_profile(effective_path, start_date, end_date, central_date)
    if result:
        result['pro_file_path'] = original_path
    return result

def parse_args() -> argparse.Namespace:
    """
    Sets up and parses command-line arguments for the script.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force regeneration of all processed data, including the manifest."
    )
    parser.add_argument(
        "-d", "--date", dest="central_date",
        help="Central date for analysis (YYYY-MM-DD). Overrides start/end."
    )
    parser.add_argument("-s", "--start", dest="start_date", help="Start date for analysis (YYYY-MM-DD).")
    parser.add_argument("-e", "--end", dest="end_date", help="End date for analysis (YYYY-MM-DD).")

    return parser.parse_args()


def main():
    """
    Main orchestrator for the entire analysis and mapping workflow.

    This function handles argument parsing, date setup, geodata preparation,
    and the parallel processing of snowpack files before generating the
    final summary map.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()

    # --- Date Handling ---
    now = datetime.now()
    central_date = None

    if args.central_date:
        try:
            central_date = datetime.strptime(args.central_date, '%Y-%m-%d')
            start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Using central date {args.central_date}. Analysis window: {start_date} to {end_date}")
        except ValueError:
            logging.error("Invalid date format for --date. Please use YYYY-MM-DD.")
            return
    else:
        start_date = args.start_date or (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = args.end_date or (now + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
        central_date = now # Default to now for time_to_loc if no central date is given
        logging.info(f"Using analysis window: {start_date} to {end_date}")


    if args.regenerate_data or not PRO_FILE_MANIFEST.exists():
        logging.info("Regenerating all processed data and manifest...")
        try:
            prepare_aspect_polygons(INPUT_POLYGONS_GEOJSON, ASPECT_POLYGONS_GEOJSON, args.regenerate_data)
            link_polygons_to_pro_files(ASPECT_POLYGONS_GEOJSON, SNOWPACK_LOCATIONS_CSV, LINKED_POLYGONS_GEOJSON)
            generate_pro_file_manifest(LINKED_POLYGONS_GEOJSON, PRO_FILE_MANIFEST)
        except Exception as e:
            logging.error(f"Failed during geodata preparation. Aborting. Error: {e}", exc_info=True)
            return

    logging.info(f"Loading .pro file list from manifest: {PRO_FILE_MANIFEST}")
    try:
        with open(PRO_FILE_MANIFEST, 'r') as f:
            production_paths = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logging.error(f"Manifest file not found at {PRO_FILE_MANIFEST}. Run with --regenerate-data to create it.")
        return

    if not production_paths:
        logging.warning("Manifest is empty. No .pro files to process.")
        return

    tasks = []
    if IS_DEV_ENVIRONMENT:
        logging.info("Development environment detected. Remapping .pro file paths.")
        prod_base = Path(PRO_FILES_BASE_PATH_PROD)
        for prod_path_str in production_paths:
            prod_path = Path(prod_path_str)
            relative_part = prod_path.relative_to(prod_base)
            effective_path = PRO_FILES_BASE_PATH_DEV / relative_part
            tasks.append((effective_path, prod_path_str, start_date, end_date, central_date))
    else:
        tasks.extend(
            (Path(prod_path_str), prod_path_str, start_date, end_date, central_date)
            for prod_path_str in production_paths
        )

    logging.info(f"Starting parallel processing on {os.cpu_count()} cores...")
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.map(worker_wrapper, tasks), total=len(tasks)))

    if all_results := [res for res in results if res is not None]:
        logging.info("All files processed. Creating summary map...")
        create_folium_map(all_results, SUMMARY_MAP_HTML, geojson_path=LINKED_POLYGONS_GEOJSON)
    else:
        logging.info("No valid results were generated, skipping map creation.")


if __name__ == "__main__":
    main()

