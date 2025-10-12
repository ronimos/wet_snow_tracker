#!/bin/bash

# --- Go to Project Root ---
# This block finds the directory where the script is located and cds into it.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

echo "Running analysis from: $(pwd)"

# --- Configuration ---
VENV_PATH="./.venv"

# --- Environment Variables for the Script ---
export PRO_FILES_SOURCE="local" # Set to 'local' for this run
# export REMOTE_PRO_FILES_URL="s3://my-snowpack-data/pro-files/"
# Set the input and output directories to the https://nwp.mtnweather.info/ locations
export PRO_FILES_INPUT_DIR="/ssd/snowpack/fcst/2025"
export WFT_RESULTS_OUTPUT_DIR="/home/www/html/ron/wetting_front"
export WFT_ASSETS_OUTPUT_DIR="/home/www/html/ron/wetting_front/plot_assets"

# --- Dynamic Date Configuration ---
# This command gets today's date in YYYY-MM-DD format and appends the time.
TODAY_AT_NOON="$(date +'%Y-%m-%d') 12:00"

# --- Execution ---
PYTHON_EXECUTABLE="$VENV_PATH/Scripts/python.exe" # For Windows Git Bash/WSL
# PYTHON_EXECUTABLE="$VENV_PATH/bin/python" # For Linux/macOS

# Run the main module, passing the dynamically generated date as an argument.
echo "Running analysis for date: $TODAY_AT_NOON"
"$PYTHON_EXECUTABLE" -m src.wetting_front_tracker.main --date "$TODAY_AT_NOON"

echo "Script finished."