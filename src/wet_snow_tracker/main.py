"""
main.py
=======

This script serves as the main entry point for running the wet snow tracker
analysis. It utilizes the SnowpackProfile class to read and process SNOWPACK
.pro files and applies a series of custom analysis functions from the
`wet_snow_tracker` module to derive insights about snowpack stability.

The primary workflow is to:
1.  Load a specified .pro file.
2.  Apply functions to find weak layers in the bottom half of the snowpack.
3.  Track the progression of wet fronts.
4.  Determine the liquid water content (LWC) at the interface above the weak layer.
5.  Print a consolidated daily summary of these metrics to the console.

This script is designed to be executed directly from the command line.

Authors: Itai and Ron
Last Updated: August 21, 2025
"""

import pandas as pd
from pathlib import Path

# Local application imports
from wet_snow_tracker.snowpack_reader import SnowpackProfile
from wet_snow_tracker.wet_front_tracker import (
    largest_fc_dh_gs_diff_bottom_half,
    wet_front_form,
    wet_front_lwc,
    lwc_above_weak,
)


start_date = "2025-03-20"
end_date = "2025-04-01"


def main():
    """
    Main function to orchestrate the snowpack analysis.

    This function defines the target .pro file and the date range for the
    analysis. It initializes a SnowpackProfile object, runs the summary
    calculations with a focus on weak layers in the bottom half of the
    snowpack, and prints a formatted, daily summary of the results.

    The function includes robust error handling for missing files and gracefully
    handles cases where analysis functions return no results for a given day.

    Raises:
        FileNotFoundError: If the specified `pro_file_path` does not exist.
    """
    pro_file_path = "data/088314.pro"  # <-- IMPORTANT: Make sure this path is correct

    if not Path(pro_file_path).exists():
        raise FileNotFoundError(
            f"Error: The file was not found at '{pro_file_path}'. "
            "Please update the 'pro_file_path' variable with the correct location."
        )

    try:
        # --- 1. Initialize the SnowpackProfile ---
        profile = SnowpackProfile(pro_file_path)

        # --- 2. Define and run the analysis calculations ---
        # Use a lambda function to configure lwc_above_weak to use the
        # weak layer function that focuses on the bottom half of the snowpack.
        lwc_above_bottom_weak = lambda df: lwc_above_weak(
            df, weak_layer_func=largest_fc_dh_gs_diff_bottom_half
        )

        summary = profile.get_profile_summary(
            parameters_to_calculate={
                "weak_layer": largest_fc_dh_gs_diff_bottom_half,
                "wet_front_by_grain": wet_front_form,
                "wet_front_by_lwc": wet_front_lwc,
                "lwc_at_interface": lwc_above_bottom_weak,
            },
            start_date=start_date,#"2025-03-20",
            end_date=end_date,#"2025-04-01",
        )

        # --- 3. Process and display the results ---
        if not summary.empty:
            print("--- Snowpack Analysis Summary (Bottom Half Focus) ---")
            print(f"File: {pro_file_path}")
            print(f"Date Range: {start_date} to {end_date}")
            print("-" * 50)

            # Robustly unpack tuple results into separate columns
            if summary['weak_layer'].notna().any():
                summary[['weak_layer_gs_diff', 'weak_layer_height']] = summary['weak_layer'].apply(pd.Series)
            else:
                summary[['weak_layer_gs_diff', 'weak_layer_height']] = pd.NA

            if summary['wet_front_by_grain'].notna().any():
                summary[['wet_front_grain_type', 'wet_front_grain_height']] = summary['wet_front_by_grain'].apply(pd.Series)
            else:
                summary[['wet_front_grain_type', 'wet_front_grain_height']] = pd.NA

            if summary['wet_front_by_lwc'].notna().any():
                summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = summary['wet_front_by_lwc'].apply(pd.Series)
            else:
                summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.NA
            
            if summary['lwc_at_interface'].notna().any():
                summary[['interface_lwc_value', 'interface_lwc_height']] = summary['lwc_at_interface'].apply(pd.Series)
            else:
                summary[['interface_lwc_value', 'interface_lwc_height']] = pd.NA

            # Define columns for the final display
            display_columns = [
                'weak_layer_height',
                'weak_layer_gs_diff',
                'wet_front_grain_height',
                'wet_front_lwc_height',
                'interface_lwc_value'
            ]
            
            # Filter for columns that actually exist to prevent errors
            cols_to_show = [col for col in display_columns if col in summary.columns]
            print(summary[cols_to_show].round(3))
            
        else:
            print("No data found for the specified date range.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # This block ensures the main function is called only when the script
    # is executed directly.
    main()