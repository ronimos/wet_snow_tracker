import argparse
import logging
import multiprocessing
import os
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
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


def process_single_profile(pro_file_path: Path, start_date_arg: str | None = None, end_date_arg: str | None = None, plotter: str = 'matplotlib') -> dict[str, Any] | None:
    """
    Handles the analysis workflow for a single file and returns its results.
    """
    try:
        profile = SnowpackProfile(pro_file_path)
        file_stem = pro_file_path.stem
        station_name = profile.metadata.get('stationName', file_stem)

        if profile.data is None or 'timestamp' not in profile.data.coords:
            logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
            return None

        time_coords = pd.to_datetime(profile.data.timestamp.values)
        min_date_in_data = time_coords.min().strftime('%Y-%m-%d')
        max_date_in_data = time_coords.max().strftime('%Y-%m-%d')
        start_date = start_date_arg or min_date_in_data
        end_date = end_date_arg or max_date_in_data

        def calculate_lwc_at_interface(df: pd.DataFrame):
            return lwc_above_weak(df, weak_layer_func=largest_fc_dh_gs_diff_bottom_half)

        summary = profile.get_profile_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_by_grain": wet_front_form, "wet_front_by_lwc": wet_front_lwc,
                "lwc_at_interface": calculate_lwc_at_interface, "highest_wet_point": get_highest_wet_point,
            },
            start_date=start_date, end_date=end_date,
        )

        if summary.empty:
            return None
        
        tuple_cols = {
            "weak_layer": ['weak_layer_gs_diff', 'weak_layer_height'],
            "wet_front_by_grain": ['wet_front_grain_type', 'wet_front_grain_height'],
            "wet_front_by_lwc": ['wet_front_lwc_value', 'wet_front_lwc_height'],
            "lwc_at_interface": ['interface_lwc_value', 'interface_lwc_height'],
        }
        for col, new_cols in tuple_cols.items():
            if col in summary and summary[col].notna().any():
                unpacked = pd.DataFrame(summary[col].dropna().tolist(), index=summary[col].dropna().index)
                unpacked.columns = new_cols
                summary = summary.join(unpacked)
        
        if plotter == 'plotly':
            plot_summary_plotly(summary, file_stem, profile.metadata)
        else:
            plot_summary_matplotlib(summary, file_stem, profile.metadata)

        time_to_loc = find_time_to_loc(summary)
        
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
    A wrapper function for the multiprocessing pool. It calls the main processing
    function and attaches the original production file path to the result.
    """
    effective_path, original_path, start_date, end_date, plotter = task_tuple
    result = process_single_profile(effective_path, start_date, end_date, plotter)
    if result:
        result['pro_file_path'] = original_path
    return result

def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force regeneration of all processed data, including the manifest."
    )
    parser.add_argument("-s", "--start", dest="start_date", help="Start date for analysis (YYYY-MM-DD).")
    parser.add_argument("-e", "--end", dest="end_date", help="End date for analysis (YYYY-MM-DD).")
    parser.add_argument(
        "--plotter", choices=['matplotlib', 'plotly'], default='matplotlib',
        help="The plotting library to use for detailed charts."
    )

    return parser.parse_args()


def main():
    """Main orchestrator for the analysis and mapping workflow."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()

    if args.regenerate_data or not PRO_FILE_MANIFEST.exists():
        logging.info("Regenerating all processed data and manifest...")
        try:
            prepare_aspect_polygons(INPUT_POLYGONS_GEOJSON, ASPECT_POLYGONS_GEOJSON)
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
        for prod_path_str in production_paths:
            # Use PurePosixPath to correctly handle the Linux-style path string
            prod_path = PurePosixPath(prod_path_str)
            prod_base = PurePosixPath(PRO_FILES_BASE_PATH_PROD)
            # Get the path part relative to the production base
            relative_part = prod_path.relative_to(prod_base)
            # Join this relative part to the local development base Path object
            effective_path = PRO_FILES_BASE_PATH_DEV / relative_part
            tasks.append((effective_path, prod_path_str, args.start_date, args.end_date, args.plotter))
    else:
            tasks.extend(
                (
                    Path(prod_path_str),
                    prod_path_str,
                    args.start_date,
                    args.end_date,
                    args.plotter,
                )
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

