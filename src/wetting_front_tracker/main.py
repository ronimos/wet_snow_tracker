"""
main.py
=======

This script is the primary entry point for the Wetting Front Tracker, a tool designed
for analyzing wet snow conditions in snowpack data simulated by SNOWPACK. It
processes `.pro` files to identify and track metrics related to wet slab
avalanche formation.

Workflow
--------
The script follows a structured workflow:
1.  **Argument Parsing**: It first parses command-line arguments to determine the
    input path (file or directory), an optional date range for analysis, and the
    preferred plotting library (`matplotlib` or `plotly`).
2.  **File Discovery**: It identifies all target `.pro` files. If the input path is
    a single file, it processes that file. If it's a directory, it discovers
    and processes every `.pro` file within it.
3.  **Data Loading & Analysis**: For each file, it loads the data, computes a daily
    time-series summary of key snowpack parameters, and observes when the
    simulated wetting front reaches the weak layer.
4.  **Output Generation**: For each file, it produces:
    - A summary table logged to the console.
    - A detailed plot (PNG or HTML) saved to the `results` directory.
5.  **Summary Map**: After processing all files, it generates a single interactive
    Folium map (`summary_map.html`), visualizing the risk level for all locations.

Command-Line Usage
------------------
# Analyze a single file with the default Matplotlib plotter
python -m wetting_front_tracker.main --path /path/to/your/file.pro

# Analyze all .pro files in a directory using the interactive Plotly plotter
python -m wetting_front_tracker.main --path /path/to/directory/ --plotter plotly
"""
import argparse
import logging
import multiprocessing
import os
import pandas as pd
from pathlib import Path
from typing import Any
from tqdm import tqdm

# Local application imports
from .param_config import (
    DATA_PATH, SUMMARY_MAP_HTML, INPUT_POLYGONS_GEOJSON, 
    ASPECT_POLYGONS_GEOJSON, LINKED_POLYGONS_GEOJSON,
    SNOWPACK_LOCATIONS_CSV, PRO_FILE_MANIFEST, DEV, INPUT_PATH,
    get_png_path, get_html_path, 
)
# Import the new function
from .prepare_geodata import prepare_aspect_polygons, generate_pro_file_manifest, link_polygons_to_pro_files
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (
    largest_fc_dh_gs_diff_bottom_half,
    wet_front_form,
    wet_front_lwc,
    lwc_above_weak,
    find_time_to_loc,
    get_total_snow_depth,
    get_highest_wet_point,
)
# Import the new plotting module
from .plotting import (
    plot_summary_matplotlib,
    plot_summary_plotly,
    create_folium_map,
)


def process_single_profile(pro_file_path: Path, start_date_arg: str | None = None, end_date_arg: str | None = None, plotter: str = 'matplotlib') -> dict[str, Any] | None:
    """
    Handles the complete analysis workflow for a single file and returns its results.

    This function loads, analyzes, logs, and plots data for one .pro file. It
    returns a dictionary containing key results needed for the summary map.

    Args:
        pro_file_path (Path): Path to the .pro file.
        start_date_arg (str | None): Optional start date for analysis (YYYY-MM-DD).
        end_date_arg (str | None): Optional end date for analysis (YYYY-MM-DD).
        plotter (str): The plotting engine to use ('matplotlib' or 'plotly').

    Returns:
        dict[str, Any] | None: A dictionary with summary results for mapping,
                                or None if processing fails.
    """
    try:
        profile = SnowpackProfile(str(pro_file_path))
        station_name = profile.metadata.get('stationName', pro_file_path.stem)

        if profile.data is None or 'timestamp' not in profile.data.coords:
            logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
            return None

        time_coords = pd.to_datetime(profile.data.timestamp.values)
        min_date_in_data = time_coords.min().strftime('%Y-%m-%d')
        max_date_in_data = time_coords.max().strftime('%Y-%m-%d')
        start_date = start_date_arg or min_date_in_data
        end_date = end_date_arg or max_date_in_data

        def calculate_lwc_at_interface(df: pd.DataFrame):
            """Wrapper to call lwc_above_weak with a specific weak_layer_func."""
            return lwc_above_weak(
                df, weak_layer_func=largest_fc_dh_gs_diff_bottom_half
            )

        summary = profile.get_profile_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth,
                "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_by_grain": wet_front_form,
                "wet_front_by_lwc": wet_front_lwc,
                "lwc_at_interface": calculate_lwc_at_interface,
                "highest_wet_point": get_highest_wet_point,
            },
            start_date=start_date,
            end_date=end_date,
        )

        if summary.empty:
            logging.info("No data found for the specified date range in %s.", pro_file_path.name)
            return None
        
        # Unpack tuple columns for logging and plotting
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
        
        # Log results to console
        #logging.info("Analysis summary:\n%s", summary.to_string())
        
        # Generate detailed plot
        #if plotter == 'plotly':
        plot_summary_plotly(summary, pro_file_path, **profile.metadata)
        #else:
        plot_summary_matplotlib(summary, pro_file_path, **profile.metadata)

        # Calculate final metric for the map
        time_to_loc = find_time_to_loc(summary)
        
        # Return key results for the summary map
        return {
            "station_name": station_name,
            "lat": float(profile.metadata.get('latitude', 0)),
            "lon": float(profile.metadata.get('longitude', 0)),
            "time_to_loc": time_to_loc,
            "png_path": get_png_path(station_name),
            "html_path": get_html_path(station_name),
        }

    except Exception:
        logging.error("An unexpected error occurred while processing '%s'.", pro_file_path.name, exc_info=True)
        return None


def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p", "--path",
        dest="input_path",
        help="Path to the input .pro file or a directory containing .pro files.",
        default=str(DATA_PATH / 'input')
    )
    parser.add_argument(
        "-s", "--start",
        dest="start_date",
        help="Start date for analysis (YYYY-MM-DD).",
        default="2025-04-05"
    )
    parser.add_argument(
        "-e", "--end",
        dest="end_date",
        help="End date for analysis (YYYY-MM-DD).",
        default="2025-05-10"
    )
    parser.add_argument(
        "--plotter",
        choices=['matplotlib', 'plotly'],
        default='matplotlib',
        help="The plotting library to use for detailed charts."
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of all processed data, including the manifest."
    )
    return parser.parse_args()


def main():
    """Main orchestrator for the analysis and mapping workflow."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = parse_args()

    # If the manifest doesn't exist or --regenerate-data is used, run the full prep pipeline.
    if args.regenerate_data or not PRO_FILE_MANIFEST.exists():
        logging.info("Regenerating all processed data and manifest...")
        try:
            prepare_aspect_polygons(INPUT_POLYGONS_GEOJSON, ASPECT_POLYGONS_GEOJSON)
            link_polygons_to_pro_files(
                ASPECT_POLYGONS_GEOJSON,
                SNOWPACK_LOCATIONS_CSV,
                LINKED_POLYGONS_GEOJSON
            )
            generate_pro_file_manifest(LINKED_POLYGONS_GEOJSON, PRO_FILE_MANIFEST)
        except Exception as e:
            logging.error(f"Failed during geodata preparation. Aborting. Error: {e}")
            return

    # --- Load the list of files to process directly from the manifest ---
    logging.info(f"Loading .pro file list from manifest: {PRO_FILE_MANIFEST}")
    try:
        with open(PRO_FILE_MANIFEST, 'r') as f:
            files_to_process = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logging.error(f"Manifest file not found at {PRO_FILE_MANIFEST}. "
                      "Run with --regenerate-data to create it.")
        return

    if not files_to_process:
        logging.warning("Manifest is empty. No .pro files to process.")
        return

    # Run analysis ONLY on the files in the manifest
    # Prepare the arguments for each parallel task
    if DEV:
        files_to_process = {p.replace('/ssd/snowpack/output/2024-newhs', str(INPUT_PATH)) for p in files_to_process}
    tasks = [
        (Path(pro_file), args.start_date, args.end_date, args.plotter)
        for pro_file in files_to_process
    ]

    # Use a process pool to execute the tasks in parallel
    # os.cpu_count() uses all available cores for maximum speed
    logging.info(f"Starting parallel processing on {os.cpu_count()} cores...")
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # Use starmap to pass multiple arguments to the function
        # Wrap with tqdm for a progress bar
        results = list(tqdm(pool.starmap(process_single_profile, tasks), total=len(tasks)))

    # Combine tasks with results before filtering ---
    combined = zip(tasks, results)
    all_results = []
    for task, res in combined:
        if res is not None:
            res['pro_file_path'] = str(task[0]) # task[0] is the Path object
            all_results.append(res)
            
    # After processing, create the final summary map
    if all_results:
        logging.info("All files processed. Creating summary map...")
        create_folium_map(all_results, SUMMARY_MAP_HTML, geojson_path=LINKED_POLYGONS_GEOJSON)
    else:
        logging.info("No valid results were generated, skipping map creation.")
    create_folium_map(all_results, SUMMARY_MAP_HTML, geojson_path=LINKED_POLYGONS_GEOJSON)
    
if __name__ == "__main__":
    main()