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

2.  **Snowpack Analysis:** The script now reads the `linked_aspect_polygons.geojson`
    file and processes each polygon individually. For each polygon, it:
    a. Extracts snowpack properties over time from the linked .pro file.
    b. Tracks the depth of the wetting front (where Liquid Water Content > 3%).
    c. Calculates the `time_to_loc`: the time (in hours) for the wetting front
       to reach the identified weak layer, relative to a central analysis date.

3.  **Visualization:** For each processed polygon, the script generates:
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
import geopandas as gpd
import numpy as np
from tqdm import tqdm

from .param_config import (
    ASPECT_POLYGONS_GEOJSON, INPUT_POLYGONS_GEOJSON, LINKED_POLYGONS_GEOJSON,
    SUMMARY_MAP_HTML, IS_DEV_ENVIRONMENT, PRO_FILES_BASE_PATH_PROD, 
    PRO_FILES_BASE_PATH_DEV, SNOWPACK_LOCATIONS_CSV
)
from .plotting import (create_folium_map, plot_summary_matplotlib, 
                       plot_summary_plotly)
from .prepare_geodata import (link_polygons_to_pro_files,
                              prepare_aspect_polygons)
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                get_total_snow_depth, lwc_above_weak,
                                largest_fc_dh_gs_diff_bottom_half, wet_front_lwc)


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
    tuple_cols = {
        "weak_layer": ['weak_layer_gs_diff', 'weak_layer_height'],
        "wet_front_lwc": ['wet_front_lwc_value', 'wet_front_lwc_height'],
    }
    for col, new_cols in tuple_cols.items():
        if col in summary_df and summary_df[col].notna().any():
            unpacked = pd.DataFrame(summary_df[col].dropna().tolist(), index=summary_df[col].dropna().index)
            unpacked.columns = new_cols
            summary_df = summary_df.join(unpacked)

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

    # 1. Identify the start times of all melt events in the entire dataset.
    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]

    # 2. Find the most recent melt event that started AT OR BEFORE the reference date.
    relevant_start_times = all_start_times[all_start_times <= reference_date]

    if relevant_start_times.empty:
        # No melt event started by the reference date, so no persistence is applied.
        return summary_df
    
    trigger_time = relevant_start_times[-1]  # This is the start of the most recent melt event.

    # 3. Find the LOC to lock onto just before this trigger time.
    lookback_window_end = trigger_time
    lookback_window_start = lookback_window_end - timedelta(days=2)
    pre_melt_df = summary_df.loc[lookback_window_start:lookback_window_end]
    
    initial_lock_height = pre_melt_df['weak_layer_height'].dropna().iloc[-1] if pre_melt_df['weak_layer_height'].notna().any() else np.nan

    if pd.isna(initial_lock_height):
        # A melt event is starting, but no weak layer was found right before it.
        return summary_df
        
    # 4. Apply persistence logic from the trigger time onwards.
    persisted_loc = summary_df['weak_layer_height'].copy()
    wet_season_mask = summary_df.index >= trigger_time
    
    # Create a series of LOCs for the wet season, anchored by our determined lock height.
    wet_loc_series = summary_df.loc[wet_season_mask, 'weak_layer_height']
    anchored_series = pd.concat([pd.Series([initial_lock_height]), wet_loc_series.reset_index(drop=True)])
    
    # The running maximum handles the "or higher if it moved up" logic.
    running_max_loc = anchored_series.cummax().iloc[1:].values
    
    persisted_loc.loc[wet_season_mask] = running_max_loc

    # 5. Forward-fill to track the layer even on days it's not detected.
    persisted_loc_filled = persisted_loc.ffill()

    # 6. The layer is eliminated when total snow depth (hs) is below its height.
    persisted_loc_filled[summary_df['hs'] < persisted_loc_filled] = np.nan

    # 7. Overwrite the original column.
    summary_df['weak_layer_height'] = persisted_loc_filled
    
    return summary_df

def process_single_profile(pro_file_path: Path, aspect: str, start_date_arg: str | None = None, end_date_arg: str | None = None, central_date_arg: datetime | None = None) -> dict[str, Any] | None:
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

    Returns:
        A dictionary containing results for the final summary map (station name,
        file_stem, time_to_loc), or None if processing fails.
    """
    try:
        profile, file_stem = _initialize_and_validate_profile(pro_file_path, aspect)
        if not profile:
            return None
        
        time_coords = pd.to_datetime(profile.data.timestamp.values)
        min_date_in_data, max_date_in_data = time_coords.min(), time_coords.max()

        raw_summary = profile.get_profile_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, 
                "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_lwc": wet_front_lwc,
                "highest_wet_point": get_highest_wet_point,
                "lwc_above_weak": lambda df: lwc_above_weak(df, largest_fc_dh_gs_diff_bottom_half)
            },
            start_date=min_date_in_data.strftime('%Y-%m-%d'),
            end_date=max_date_in_data.strftime('%Y-%m-%d'),
        ).copy()

        if raw_summary.empty:
            return None

        prepared_summary = _unpack_and_prepare_summary(raw_summary)
        
        # Apply the robust persistence logic
        summary_full = _persist_loc_height(prepared_summary, central_date_arg)

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


        if not summary_for_plot.empty:
            plot_summary_matplotlib(summary_for_plot, file_stem, profile.metadata, lwc_data_for_plot, central_date=central_date_arg)
            plot_summary_plotly(summary_for_plot, file_stem, profile.metadata)
            
        else:
            logging.warning(
                f"No snowpack data found for {file_stem} in the analysis window "
                f"({start_date_arg} to {end_date_arg}). Plots will be skipped."
            )

        time_to_loc = find_time_to_loc(summary_full, reference_date=central_date_arg)

        return {
            "station_name": profile.metadata.get('stationName', file_stem),
            "file_stem": file_stem,
            "time_to_loc": time_to_loc,
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

def parse_args() -> argparse.Namespace:
    """
    Sets up and parses command-line arguments for the script.

    Defines arguments for controlling the analysis, such as forcing data
    regeneration and setting the analysis time window.

    Returns:
        An argparse.Namespace object containing the parsed command-line arguments.
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
        help="Central date for analysis (YYYY-MM-DD). Overrides start/end."
    )
    parser.add_argument("-s", "--start", dest="start_date", help="Start date for analysis.")
    parser.add_argument("-e", "--end", dest="end_date", help="End date for analysis.")
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
    if args.central_date:
        try:
            central_date = datetime.strptime(args.central_date, '%Y-%m-%d')
            start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            logging.error("Invalid date format. Use YYYY-MM-DD.")
            return
    else:
        start_date = args.start_date or (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = args.end_date or (now + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
        central_date = now

    if args.regenerate_data or not LINKED_POLYGONS_GEOJSON.exists():
        logging.info("Regenerating all processed data...")
        try:
            prepare_aspect_polygons(INPUT_POLYGONS_GEOJSON, ASPECT_POLYGONS_GEOJSON, args.regenerate_data)
            link_polygons_to_pro_files(ASPECT_POLYGONS_GEOJSON, SNOWPACK_LOCATIONS_CSV, LINKED_POLYGONS_GEOJSON)
        except Exception as e:
            logging.error(f"Failed during geodata preparation: {e}", exc_info=True)
            return

    try:
        linked_gdf = gpd.read_file(LINKED_POLYGONS_GEOJSON)
    except Exception as e:
        logging.error(f"Could not read {LINKED_POLYGONS_GEOJSON}. Run with --regenerate-data.", exc_info=True)
        return

    # --- Task Generation ---
    tasks = []
    prod_base = Path(PRO_FILES_BASE_PATH_PROD)
    for poly in linked_gdf.itertuples(index=False):
        original_path = poly.pro_file_path
        aspect = poly.aspect
        effective_path = Path(original_path)
        if IS_DEV_ENVIRONMENT:
            try:
                relative_part = effective_path.relative_to(prod_base)
                effective_path = PRO_FILES_BASE_PATH_DEV / relative_part
            except ValueError:
                logging.warning(f"Could not remap dev path for {original_path}, skipping.")
                continue
        
        tasks.append((effective_path, aspect, start_date, end_date, central_date))

    logging.info(f"Starting parallel processing on {len(tasks)} polygons...")
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.map(worker_wrapper, tasks), total=len(tasks)))

    valid_results = [res for res in results if res is not None]
    if valid_results:
        logging.info("All polygons processed. Creating summary map...")
        results_df = pd.DataFrame(valid_results)
        # The order is preserved by pool.map, so we can concatenate horizontally
        final_gdf = pd.concat([
            linked_gdf.reset_index(drop=True), 
            results_df.reset_index(drop=True)
        ], axis=1)
        create_folium_map(final_gdf, SUMMARY_MAP_HTML)
    else:
        logging.info("No valid results were generated, skipping map creation.")

if __name__ == "__main__":
    main()

