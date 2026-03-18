#!/bin/bash

# --- Go to Project Root ---
# This block finds the directory where the script is located and cds into it.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

echo "Running analysis from: $(pwd)"

# --- Configuration ---
source .venv/bin/activate

# --- Environment Variables for the Script ---
export PRO_FILES_SOURCE="local" # Set to 'local' for this run
# export REMOTE_PRO_FILES_URL="s3://my-snowpack-data/pro-files/"
export PRO_FILES_INPUT_DIR="/ssd/snowpack/fcst/2025"
export WFT_RESULTS_OUTPUT_DIR="/home/www/html/ron/wetting_front"
export WFT_ASSETS_OUTPUT_DIR="/home/www/html/ron/wetting_front/plot_assets"

# --- Dynamic Date Configuration ---
# This command gets today's date in YYYY-MM-DD format and appends the time.
TODAY_AT_NOON="$(date +'%Y-%m-%d') 12:00"

# --- Execution ---
# 1. Create a log directory if it doesn't exist
mkdir -p logs

# 2. Define a log file with a timestamp
LOG_FILE="logs/run_$(date +'%Y-%m-%d_%H-%M-%S').log"

# 3. Run the main module, redirecting all output (>> and 2>&1) to the log file
echo "Running analysis for date: $TODAY_AT_NOON. See log for details: $LOG_FILE"
python -m src.wetting_front_tracker.main --date "$TODAY_AT_NOON" >> "$LOG_FILE" 2>&1

echo "Script finished."