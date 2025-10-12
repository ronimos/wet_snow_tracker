"""
main.py
=======

Main entry point and orchestrator for the Wetting Front Tracker application.

This script manages the end-to-end workflow:
1. Parse command-line arguments and setup configuration
2. Prepare geospatial data (conditionally)
3. Generate analysis tasks for each polygon
4. Execute parallel snowpack analysis
5. Aggregate results and create summary map

Usage:
    python -m src.wetting_front_tracker.main
    python -m src.wetting_front_tracker.main --date 2025-05-09
    python -m src.wetting_front_tracker.main --regenerate-data
"""

import argparse
import json
import logging
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from .param_config import config
from .plotting import create_folium_map, plot_summary_matplotlib, plot_summary_plotly
from .prepare_geodata import (
    link_polygons_to_pro_files,
    prepare_aspect_polygons,
)
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (
    find_time_to_loc,
    find_wet_slab_loc_bottom_half,
    get_highest_wet_point,
    get_total_snow_depth,
    lwc_above_weak,
    wet_front_lwc,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("wetting_front_tracker.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File Manifest Management
# ---------------------------------------------------------------------------

def generate_pro_file_manifest(base_path: Path, manifest_path: Path) -> None:
    """
    Recursively scans a directory for .pro files and saves paths to a manifest.

    The manifest is a JSON object mapping filenames to their absolute paths,
    allowing for quick lookups without filesystem scanning on every run.

    Args:
        base_path: The root directory to start the recursive scan from
        manifest_path: The full path where the JSON manifest will be saved
    """
    logger.info(f"Scanning for .pro files under {base_path}...")
    
    try:
        pro_files = list(base_path.rglob('*.pro'))
        manifest = {file.name: str(file.resolve()) for file in pro_files}
        
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        logger.info(f"Pro file manifest with {len(manifest)} entries saved to {manifest_path}")
    
    except Exception as e:
        logger.error(f"Failed to generate pro file manifest: {e}", exc_info=True)
        raise


def ensure_pro_file_is_local(
    file_name: str,
    local_input_path: Path,
    remote_base_url: str,
    central_date: datetime
) -> None:
    """
    Checks if a .pro file exists locally and is fresh. Downloads if needed.
    
    A file is considered "fresh" if it's less than 12 hours old relative to
    the central analysis date.

    Args:
        file_name: Name of the .pro file to check
        local_input_path: Local directory where files should be stored
        remote_base_url: Base URL for downloading remote files
        central_date: The central date for the analysis (for freshness check)
        
    Note:
        This is a placeholder. Implement actual download logic based on your
        data source (S3, HTTP, etc.)
    """
    local_file_path = local_input_path / file_name
    
    # Check if file exists and is fresh (less than 12 hours old)
    if local_file_path.exists():
        mod_time_ts = os.path.getmtime(local_file_path)
        mod_time_dt = datetime.fromtimestamp(mod_time_ts, tz=timezone.utc)
        central_date_utc = central_date.replace(tzinfo=timezone.utc)
        
        if (central_date_utc - mod_time_dt) < timedelta(hours=12):
            logger.debug(f"'{file_name}' is fresh. Skipping download.")
            return

    # File is missing or stale, download it
    logger.info(f"Downloading '{file_name}'...")
    remote_file_url = f"{remote_base_url.rstrip('/')}/{file_name}"
    
    # TODO: Implement your download logic here
    # Examples:
    # - S3: boto3.client('s3').download_file(bucket, key, str(local_file_path))
    # - HTTP: requests.get(url, stream=True) and save chunks
    
    logger.warning(f"Placeholder: Would download from {remote_file_url} to {local_file_path}")


# ---------------------------------------------------------------------------
# Profile Processing - Helper Functions
# ---------------------------------------------------------------------------

def _initialize_profile(
    pro_file_path: Path,
    aspect: str
) -> tuple[Optional[SnowpackProfile], Optional[str]]:
    """
    Initializes a SnowpackProfile and validates its data.

    Args:
        pro_file_path: Path to the .pro input file
        aspect: The aspect of the polygon ('N', 'E', 'S', 'W', 'Flat')

    Returns:
        A tuple of (profile, file_stem) or (None, None) if invalid
    """
    try:
        profile = SnowpackProfile(pro_file_path)
        file_stem = f"{pro_file_path.stem}_{aspect}"
        profile.metadata['aspect'] = aspect

        if profile.data is None:
            logger.warning(f"No data loaded from '{pro_file_path.name}'. Skipping.")
            return None, None
        
        if 'timestamp' not in profile.data.coords:
            logger.warning(f"No timestamps in '{pro_file_path.name}'. Skipping.")
            return None, None
        
        return profile, file_stem
    
    except FileNotFoundError:
        logger.error(f"File not found: {pro_file_path}")
        return None, None
    
    except Exception as e:
        logger.error(f"Failed to initialize profile from {pro_file_path}: {e}", exc_info=True)
        return None, None


def _calculate_analysis_window(
    profile: SnowpackProfile,
    central_date: Optional[datetime]
) -> tuple[datetime, datetime]:
    """
    Calculates the analysis window for the profile.

    Args:
        profile: The SnowpackProfile object
        central_date: Optional central date for the analysis

    Returns:
        A tuple of (min_date, max_date) for the analysis window
    """
    if central_date:
        min_date = central_date - timedelta(days=7)
        max_date = central_date + timedelta(hours=72)
    else:
        # Use the full time range from the data
        min_date = pd.to_datetime(profile.data.timestamp.values[0])
        max_date = pd.to_datetime(profile.data.timestamp.values[-1])
    
    return min_date, max_date


def _calculate_summary(
    profile: SnowpackProfile,
    min_date: datetime,
    max_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Calculates the full time series summary for a profile.

    Args:
        profile: The SnowpackProfile object
        min_date: Start date for the analysis
        max_date: End date for the analysis

    Returns:
        A DataFrame with the summary, or None if calculation fails
    """
    try:
        parameters_to_calculate = {
            "hs": get_total_snow_depth,
            "weak_layer": find_wet_slab_loc_bottom_half,
            "wet_front_lwc": wet_front_lwc,
            "highest_wet_point": get_highest_wet_point,
            "lwc_above_weak": lambda df: lwc_above_weak(df, find_wet_slab_loc_bottom_half)
        }
        
        raw_summary = profile.get_full_timeseries_summary(
            parameters_to_calculate=parameters_to_calculate,
            start_date=str(min_date),
            end_date=str(max_date),
        )
        
        if raw_summary.empty:
            logger.warning("Summary calculation returned empty DataFrame")
            return None
        
        return raw_summary.copy()
    
    except Exception as e:
        logger.error(f"Failed to calculate summary: {e}", exc_info=True)
        return None


def _prepare_summary_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpacks tuple columns and ensures correct data types.

    Args:
        summary_df: The raw summary DataFrame

    Returns:
        Prepared DataFrame with unpacked columns and numeric types
    """
    # Rename columns for clarity
    rename_map = {
        "weak_layer_value": "weak_layer_gs_diff",
    }
    summary_df.rename(columns=rename_map, inplace=True)

    # Ensure numeric types for key columns
    numeric_cols = ['weak_layer_height', 'wet_front_lwc_height', 'hs']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    
    return summary_df


def _persist_loc_height(
    summary_df: pd.DataFrame,
    reference_date: datetime
) -> pd.DataFrame:
    """
    Identifies and tracks the primary weak layer through a melt event.

    This function finds the start of the most recent melt event, locks onto
    the last known weak layer before it, and tracks that layer forward. If a
    new weak layer appears at a higher elevation, the lock updates to track it.

    Args:
        summary_df: The prepared summary DataFrame
        reference_date: The central date for the analysis

    Returns:
        DataFrame with adjusted weak_layer_height for persistence
    """
    required_cols = ['weak_layer_height', 'wet_front_lwc_height']
    if not all(col in summary_df.columns for col in required_cols):
        logger.warning(f"Cannot persist LOC: missing required columns")
        return summary_df

    # Find melt event starts
    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]

    # Find the most recent event start before reference date
    relevant_start_times = all_start_times[all_start_times <= reference_date]

    if relevant_start_times.empty:
        logger.info("No melt event found before reference date")
        return summary_df
    
    trigger_time = relevant_start_times[-1]
    logger.info(f"Melt event detected at {trigger_time}")

    # Look back 2 days before the melt event to find the weak layer
    lookback_window_end = trigger_time
    lookback_window_start = lookback_window_end - timedelta(days=2)
    pre_melt_df = summary_df.loc[lookback_window_start:lookback_window_end]
    
    # Find the last valid weak layer before the melt
    valid_pre_melt_locs = pre_melt_df['weak_layer_height'].dropna()
    
    if valid_pre_melt_locs.empty:
        logger.warning(f"No valid weak layers found in lookback window before {trigger_time}")
        return summary_df
    
    initial_lock_height = valid_pre_melt_locs.iloc[-1]
    logger.info(f"Locked onto weak layer at {initial_lock_height}m")

    # Apply persistence logic from the melt event forward
    persisted_loc = summary_df['weak_layer_height'].copy()
    wet_season_mask = summary_df.index >= trigger_time
    
    wet_loc_series = summary_df.loc[wet_season_mask, 'weak_layer_height']
    
    # Anchor with the initial lock height and compute running maximum
    anchored_series = pd.concat([
        pd.Series([initial_lock_height]),
        wet_loc_series.reset_index(drop=True)
    ])
    
    running_max_loc = anchored_series.cummax().iloc[1:].values
    persisted_loc.loc[wet_season_mask] = running_max_loc

    # Forward fill and validate against total snow depth
    persisted_loc_filled = persisted_loc.ffill()
    persisted_loc_filled[summary_df['hs'] < persisted_loc_filled] = np.nan

    summary_df['weak_layer_height'] = persisted_loc_filled
    
    return summary_df


def _slice_data_for_plots(
    summary_full: pd.DataFrame,
    lwc_data_full: Any,
    start_date: str,
    end_date: str
) -> tuple[pd.DataFrame, Any]:
    """
    Slices data to the analysis window for plotting.

    Args:
        summary_full: Full time series summary DataFrame
        lwc_data_full: Full LWC data from xarray
        start_date: Start date string
        end_date: End date string

    Returns:
        Tuple of (sliced_summary, sliced_lwc_data)
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Slice summary DataFrame
    is_in_window = (summary_full.index >= start_dt) & (summary_full.index <= end_dt)
    summary_for_plot = summary_full[is_in_window]

    # Slice LWC data
    is_in_lwc_window = (
        (lwc_data_full.timestamp >= start_dt) &
        (lwc_data_full.timestamp <= end_dt)
    )
    lwc_data_for_plot = lwc_data_full.sel(timestamp=is_in_lwc_window)

    return summary_for_plot, lwc_data_for_plot


def _generate_plots(
    summary_for_plot: pd.DataFrame,
    summary_full: pd.DataFrame,
    lwc_data_for_plot: Any,
    file_stem: str,
    station_metadata: dict,
    central_date: Optional[datetime],
    assets_path: Path
) -> None:
    """
    Generates both static and interactive plots for the profile.

    Args:
        summary_for_plot: Summary data sliced to plot window
        summary_full: Full summary data (for Plotly)
        lwc_data_for_plot: LWC data sliced to plot window
        file_stem: Base filename for outputs
        station_metadata: Station metadata dictionary
        central_date: Central date for vertical line on plots
        assets_path: Directory to save plots
    """
    if summary_for_plot.empty:
        logger.warning(f"No data in plot window for {file_stem}. Skipping plots.")
        return

    try:
        plot_summary_matplotlib(
            summary_for_plot,
            file_stem,
            station_metadata,
            lwc_data_for_plot,
            central_date,
            assets_path
        )
        logger.debug(f"Created Matplotlib plot for {file_stem}")
    except Exception as e:
        logger.error(f"Failed to create Matplotlib plot for {file_stem}: {e}")

    try:
        plot_summary_plotly(
            summary_full,
            file_stem,
            station_metadata,
            central_date,
            assets_path
        )
        logger.debug(f"Created Plotly plot for {file_stem}")
    except Exception as e:
        logger.error(f"Failed to create Plotly plot for {file_stem}: {e}")


def _build_result_dict(
    summary_full: pd.DataFrame,
    file_stem: str,
    station_metadata: dict,
    reference_date: datetime
) -> dict[str, Any]:
    """
    Builds the final result dictionary for a processed profile.

    Args:
        summary_full: The complete summary DataFrame
        file_stem: The file stem identifier
        station_metadata: Station metadata dictionary
        reference_date: Reference date for time_to_loc calculation

    Returns:
        Dictionary with results for the summary map
    """
    time_to_loc = find_time_to_loc(summary_full, reference_date=reference_date)

    return {
        "station_name": station_metadata.get('stationName', file_stem),
        "file_stem": file_stem,
        "time_to_loc": time_to_loc,
        "central_date_str": reference_date.strftime('%Y-%m-%d %H:%M')
    }


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def process_single_profile(
    pro_file_path: Path,
    aspect: str,
    start_date_arg: Optional[str] = None,
    end_date_arg: Optional[str] = None,
    central_date_arg: Optional[datetime] = None,
    assets_path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """
    Handles the full analysis workflow for a single polygon and its .pro file.
    
    This function is parallelized across multiple CPU cores. It orchestrates:
    1. Profile initialization and validation
    2. Time series summary calculation
    3. LOC persistence logic
    4. Plot generation
    5. Final metric calculation

    Args:
        pro_file_path: Path to the .pro input file
        aspect: The aspect of the polygon ('N', 'E', 'S', 'W', 'Flat')
        start_date_arg: Start date for the analysis window (plot range)
        end_date_arg: End date for the analysis window (plot range)
        central_date_arg: Central reference date for calculations
        assets_path: Directory where output plots should be saved

    Returns:
        Dictionary containing results for the summary map, or None if processing fails
    """
    try:
        # Step 1: Initialize and validate profile
        profile, file_stem = _initialize_profile(pro_file_path, aspect)
        if not profile or not file_stem:
            return None
        
        # Step 2: Calculate analysis window
        min_date, max_date = _calculate_analysis_window(profile, central_date_arg)
        
        # Step 3: Calculate summary
        raw_summary = _calculate_summary(profile, min_date, max_date)
        if raw_summary is None:
            return None

        # Step 4: Prepare summary columns
        prepared_summary = _prepare_summary_columns(raw_summary)
        
        # Step 5: Apply LOC persistence logic
        reference_date = central_date_arg or datetime.now()
        summary_full = _persist_loc_height(prepared_summary, reference_date)

        # Step 6: Prepare data for plotting
        if start_date_arg is None or end_date_arg is None:
            logger.error(f"Analysis window dates missing for {file_stem}. Cannot create plots.")
            return None
        
        # Get LWC data before deleting profile
        lwc_data_full = profile.data[['lwc', 'height']]
        station_metadata = profile.metadata
        del profile  # Free memory

        summary_for_plot, lwc_data_for_plot = _slice_data_for_plots(
            summary_full,
            lwc_data_full,
            start_date_arg,
            end_date_arg
        )

        # Step 7: Generate plots
        _generate_plots(
            summary_for_plot,
            summary_full,
            lwc_data_for_plot,
            file_stem,
            station_metadata,
            central_date_arg,
            assets_path or config.paths.assets_path
        )

        # Step 8: Build and return result
        return _build_result_dict(summary_full, file_stem, station_metadata, reference_date)

    except Exception as e:
        logger.error(
            f"Error processing {pro_file_path.name} for aspect {aspect}: {e}",
            exc_info=True
        )
        return None


def worker_wrapper(task_tuple: tuple) -> Optional[dict[str, Any]]:
    """
    Wrapper function to enable multiprocessing by unpacking arguments.

    Args:
        task_tuple: Tuple of arguments for process_single_profile

    Returns:
        Result dictionary from process_single_profile, or None on error
    """
    return process_single_profile(*task_tuple)


# ---------------------------------------------------------------------------
# Date and Time Utilities
# ---------------------------------------------------------------------------

def _get_closest_synoptic_time(reference_time: datetime) -> datetime:
    """
    Finds the closest standard synoptic time (00, 06, 12, 18 UTC).

    Args:
        reference_time: The input time

    Returns:
        Datetime object representing the closest synoptic time
    """
    base_date = reference_time.date()
    
    # Create candidate times on the same day
    candidates = [
        datetime.combine(base_date, datetime.min.time()).replace(hour=h)
        for h in [0, 6, 12, 18]
    ]
    
    # Add previous and next day edge candidates
    candidates.insert(0, candidates[0] - timedelta(hours=6))
    candidates.append(candidates[-1] + timedelta(hours=6))

    # Find the closest candidate
    return min(candidates, key=lambda dt: abs(reference_time - dt))


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Sets up and parses command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of all processed data."
    )
    
    parser.add_argument(
        "-d", "--date",
        dest="central_date",
        help="Central date and time for analysis (e.g., 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'). "
             "Rounds to the closest synoptic time (00, 06, 12, 18).",
        default="2025-05-09 12:00"
    )
    
    parser.add_argument(
        "-s", "--start",
        dest="start_date",
        help="Start date for analysis (overrides default window)."
    )
    
    parser.add_argument(
        "-e", "--end",
        dest="end_date",
        help="End date for analysis (overrides default window)."
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        dest="input_dir",
        type=Path,
        default=None,
        help="Override default base directory for .pro files."
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Override default directory for the final map."
    )
    
    parser.add_argument(
        "-a", "--assets-dir",
        dest="assets_dir",
        type=Path,
        default=None,
        help="Override default directory for plot assets."
    )
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    """
    Main orchestrator for the entire analysis and mapping workflow.
    
    Handles:
    1. Argument parsing and configuration
    2. Date setup and validation
    3. Geodata preparation (conditional)
    4. Parallel processing of snowpack files
    5. Final summary map generation
    """
    args = parse_args()
    
    # Path configuration
    input_path = args.input_dir or config.paths.input_path
    output_path = args.output_dir or config.paths.results_path
    assets_path = args.assets_dir or config.paths.assets_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)
    summary_map_path = output_path / "summary_map.html"
    
    logger.info(f"Input .pro directory: {input_path}")
    logger.info(f"Output map directory: {output_path}")
    logger.info(f"Plot assets directory: {assets_path}")

    # Date handling
    try:
        # Try parsing with time first, then date only
        try:
            initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d %H:%M')
        except ValueError:
            initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d')
    except ValueError:
        logger.error(
            f"Invalid date format: '{args.central_date}'. "
            "Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'."
        )
        return

    central_date = _get_closest_synoptic_time(initial_ref_time)
    start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Processing for central date: {central_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Analysis window: {start_date} to {end_date}")

    # Geodata preparation
    input_geojson = config.get_input_polygons_path()
    linked_polygons_path = config.paths.linked_polygons
    manifest_path = config.paths.pro_file_manifest
    
    if args.regenerate_data or not linked_polygons_path.exists():
        logger.info("Regenerating processed data and pro file manifest...")
        
        # Generate manifest first
        generate_pro_file_manifest(input_path, manifest_path)
        
        # Prepare geodata
        prepare_aspect_polygons(
            input_geojson,
            config.paths.aspect_polygons,
            args.regenerate_data
        )
        
        link_polygons_to_pro_files(
            config.paths.aspect_polygons,
            config.paths.snowpack_locations_csv,
            linked_polygons_path
        )

    # Load linked polygons and manifest
    try:
        linked_gdf = gpd.read_file(linked_polygons_path)
        with open(manifest_path, 'r') as f:
            pro_file_manifest = json.load(f)
        logger.info(f"Loaded manifest with {len(pro_file_manifest)} .pro files")
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Run with --regenerate-data.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid manifest file: {e}. Run with --regenerate-data.")
        return

    # Conditional data download
    if config.data_source.is_remote:
        logger.info("Checking for remote .pro files to download...")
        required_files = {Path(p).name for p in linked_gdf['pro_file_path'].unique()}
        
        for file_name in tqdm(required_files, desc="Updating data files"):
            ensure_pro_file_is_local(
                file_name,
                input_path,
                config.data_source.remote_url,
                central_date
            )
    else:
        logger.info("Using local .pro files (PRO_FILES_SOURCE is 'local')")

    # Task generation
    tasks = []
    for poly in linked_gdf.itertuples(index=False):
        file_name = Path(str(poly.pro_file_path)).name
        
        if full_pro_path_str := pro_file_manifest.get(file_name):
            effective_path = Path(full_pro_path_str)
            tasks.append((
                effective_path,
                poly.aspect,
                start_date,
                end_date,
                central_date,
                assets_path
            ))
        else:
            logger.warning(f"File '{file_name}' not found in manifest. Skipping.")
    
    if not tasks:
        logger.error("No tasks were generated for processing. Exiting.")
        return

    logger.info(f"Generated {len(tasks)} analysis tasks")

    # Parallel processing
    cpu_cores = os.cpu_count() or 1
    worker_count = max(1, int(cpu_cores / 4))
    
    logger.info(f"Starting parallel processing with {worker_count} workers...")
    
    with multiprocessing.Pool(processes=worker_count) as pool:
        results = list(
            tqdm(
                pool.imap(worker_wrapper, tasks, chunksize=1),
                total=len(tasks),
                desc="Processing profiles"
            )
        )

    # Aggregation and final map
    valid_results = [res for res in results if res is not None]
    
    if not valid_results:
        logger.warning("No valid results were generated. Skipping map creation.")
        return
    
    logger.info(f"Successfully processed {len(valid_results)}/{len(tasks)} profiles")
    
    results_df = pd.DataFrame(valid_results)
    final_gdf = gpd.GeoDataFrame(
        pd.concat([
            linked_gdf.reset_index(drop=True),
            results_df.reset_index(drop=True)
        ], axis=1)
    )
    
    logger.info("Creating summary map...")
    create_folium_map(final_gdf, summary_map_path, central_date, assets_path)
    logger.info(f"Summary map saved to {summary_map_path}")


if __name__ == "__main__":
    main()