"""
main.py
=======

This script serves as the main entry point for running the wet snow tracker
analysis. It utilizes the SnowpackProfile class to read and process SNOWPACK
.pro files and applies a series of custom analysis functions from the
`wet_snow_tracker` module to derive insights about snowpack stability.

The script is designed to be a complete, end-to-end analysis tool that:
1.  Reads snowpack data, utilizing a fast NetCDF cache if available.
2.  Defines a set of analysis parameters focused on wet slab avalanche conditions.
3.  Computes a daily time-series summary of these parameters over a specified
    date range.
4.  Processes and formats the results into a human-readable table printed to
    the console.
5.  Generates and displays a comprehensive plot visualizing the daily evolution
    of key snowpack metrics.

To run the analysis, execute the script from the command line, providing the
path to a .pro file. Optional start and end dates can be specified.

Command-Line Usage:
    # Run analysis for the entire time series in a file
    python main.py /path/to/your/file.pro

    # Run analysis for a specific date range
    python main.py /path/to/your/file.pro --start 2025-02-01 --end 2025-03-15

Dependencies:
- pandas
- matplotlib
- Local: wet_snow_tracker.snowpack_reader, wet_snow_tracker.wet_front_tracker

Authors: Itai and Ron
Last Updated: August 26, 2025
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse # Added for command-line argument parsing


# Local application imports
from wet_snow_tracker.snowpack_reader import SnowpackProfile
from wet_snow_tracker.wet_front_tracker import (
    largest_fc_dh_gs_diff_bottom_half,
    wet_front_form,
    wet_front_lwc,
    lwc_above_weak,
)


def get_total_snow_depth(df: pd.DataFrame) -> float:
    """Calculates the total snow depth (HS) for a single daily profile.

    Args:
        df (pd.DataFrame): A DataFrame representing one day's snow profile,
                           containing a 'height' column.

    Returns:
        float: The maximum height in the profile, representing the total
               snow depth in centimeters. Returns 0 if the DataFrame is empty.
    """
    if df.empty or "height" not in df:
        return 0
    return df['height'].max()


def get_highest_wet_point(df: pd.DataFrame):
    """
    Finds the highest point of a layer meeting "wet" conditions.

    This function determines the upper boundary of the wet snow region for a
    given day's profile. A layer is considered "wet" if its grain type is a
    wet form (SNOWPACK codes 770-779) OR its liquid water content (LWC) is
    greater than 3%. This value is used in the plot to shade the full
    vertical extent of the wet snow.

    Args:
        df (pd.DataFrame): A DataFrame for a single day's snow profile.

    Returns:
        float or None: The height of the highest wet layer, or None if no
                       layers meet the criteria.
    """
    if df.empty or "grain_type" not in df or "lwc" not in df:
        return None

    mask = ((df['grain_type'] >= 770) & (df['grain_type'] < 780)) | (df['lwc'] > 0.03)
    wet_layers = df[mask]

    if wet_layers.empty:
        return None

    return float(wet_layers['height'].max())


def plot_summary(df: pd.DataFrame, file_path: str, **kwargs):
    """
    Generates and displays a plot summarizing the snowpack analysis over time.

    This function visualizes the daily evolution of key snowpack metrics:
    - Total Snow Depth (HS): A blue line with markers.
    - Weak Layer Height (LOC): A solid black line.
    - Deepest Wet Front: A solid red line indicating the lowest point of
      significant water penetration (LWC > 3%).
    - Wet Layer Extent: A shaded cyan area showing the full vertical region of
      the snowpack that meets "wet" conditions.

    Args:
        df (pd.DataFrame): The summary DataFrame containing the daily analysis
                           results.
        file_path (str): The path to the source .pro file, used for context.
        **kwargs: A dictionary of station metadata (e.g., stationName,
                  altitude, latitude) used to create a detailed plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    if 'hs' in df.columns:
        ax.plot(df.index, df['hs'], label='Total Snow Depth (HS)', color='blue', marker='o', linewidth=2)
    if 'weak_layer_height' in df.columns:
        ax.plot(df.index, df['weak_layer_height'], label='Weak Layer Height (LOC)', color='black')
    if 'wet_front_lwc_height' in df.columns:
        ax.plot(df.index, df['wet_front_lwc_height'], label='Deepest Wet Front (LWC > 3%)', color='red')
    
    if 'wet_front_lwc_height' in df.columns and 'highest_wet_point' in df.columns:
        ax.fill_between(
            df.index,
            df['wet_front_lwc_height'],
            df['highest_wet_point'],
            where=df['wet_front_lwc_height'].notna(),
            color='cyan',
            alpha=0.7,
            interpolate=True,
            label='Wet Layer Extent'
        )
    
    location_id = kwargs.get('stationName', "N/A")
    location = (kwargs.get("latitude"), kwargs.get('longitude'))
    elevation = kwargs.get("altitude")
    aspect = kwargs.get("slopeAzi")
    title = f"Wet front tracking, Station id: {location_id}, location: {location}, elevation: {elevation}m, aspect: {aspect}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    order = ['Total Snow Depth (HS)', 'Wet Layer Extent', 'Deepest Wet Front (LWC > 3%)', 'Weak Layer Height (LOC)']
    ax.legend([handles[labels.index(key)] for key in order if key in labels],
              [key for key in order if key in labels])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def main():
    """Main function to orchestrate the entire snowpack analysis workflow."""
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on a SNOWPACK .pro file."
    )
    parser.add_argument(
        "-f", "--pro_file",
        help="Path to the input .pro file.",
        default="/Users/ronsi/Documents/Projects/central_asia_snowpack/data/snowpack/17602_res.pro"
    )
    parser.add_argument(
        "-s", "--start",
        dest="start_date",
        help="Start date for analysis (YYYY-MM-DD). Defaults to the beginning of the time series."
    )
    parser.add_argument(
        "-e", "--end",
        dest="end_date",
        help="End date for analysis (YYYY-MM-DD). Defaults to the end of the time series."
    )
    args = parser.parse_args()

    # --- 2. Load Profile and Handle File Path ---
    pro_file_path = args.pro_file
    if not Path(pro_file_path).exists():
        raise FileNotFoundError(
            f"Error: The file was not found at '{pro_file_path}'. "
            "Please provide a valid path."
        )

    try:
        profile = SnowpackProfile(pro_file_path)

        # --- 3. Determine Date Range (with defaults) ---
        if profile.data is not None and 'timestamp' in profile.data.coords:
            time_coords = pd.to_datetime(profile.data.timestamp.values)
            min_date_in_data = time_coords.min().strftime('%Y-%m-%d')
            max_date_in_data = time_coords.max().strftime('%Y-%m-%d')

            start_date = args.start_date if args.start_date else min_date_in_data
            end_date = args.end_date if args.end_date else max_date_in_data
        else:
            # Fallback if data is empty or malformed
            start_date = args.start_date
            end_date = args.end_date
            
        # --- 4. Define Analysis Calculations ---
        lwc_above_bottom_weak = lambda df: lwc_above_weak(
            df, weak_layer_func=largest_fc_dh_gs_diff_bottom_half
        )

        summary = profile.get_profile_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth,
                "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_by_grain": wet_front_form,
                "wet_front_by_lwc": wet_front_lwc,
                "lwc_at_interface": lwc_above_bottom_weak,
                "highest_wet_point": get_highest_wet_point,
            },
            start_date=start_date,
            end_date=end_date,
        )

        # --- 5. Process and Display Results ---
        if not summary.empty:
            print("--- Snowpack Analysis Summary (Bottom Half Focus) ---")
            print(f"File: {pro_file_path}")
            print(f"Date Range: {start_date} to {end_date}")
            print("-" * 50)

            tuple_cols = {
                "weak_layer": ['weak_layer_gs_diff', 'weak_layer_height'],
                "wet_front_by_grain": ['wet_front_grain_type', 'wet_front_grain_height'],
                "wet_front_by_lwc": ['wet_front_lwc_value', 'wet_front_lwc_height'],
                "lwc_at_interface": ['interface_lwc_value', 'interface_lwc_height'],
            }

            for col, new_cols in tuple_cols.items():
                if col in summary and summary[col].notna().any():
                    unpacked_df = pd.DataFrame(summary[col].dropna().tolist(), index=summary[col].dropna().index)
                    unpacked_df.columns = new_cols
                    summary = summary.join(unpacked_df)

            display_columns = [
                'hs', 'weak_layer_height', 'weak_layer_gs_diff',
                'wet_front_lwc_height', 'highest_wet_point', 'interface_lwc_value'
            ]

            cols_to_show = [col for col in display_columns if col in summary.columns]
            print(summary[cols_to_show].round(3))

            # --- 6. Generate and Display Plot ---
            print("\nGenerating plot...")
            plot_summary(summary, pro_file_path, **profile.metadata)

        else:
            print("No data found for the specified date range.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # This construct ensures the main() function is called only when the
    # script is executed directly, allowing it to be a reusable module.

    main()