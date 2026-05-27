#!/bin/bash
set -e

# --- Configuration ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

source .venv/bin/activate

INPUT_DIR="/home/snowpack/runs/output/2024"
RUN_DATE="2026-03-19"
OUTPUT_DIR="results/ml_training/${RUN_DATE//-/}"
DATASET="$OUTPUT_DIR/ml_training_dataset.csv"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ml_train_${RUN_DATE//-/}_$(date +'%H%M%S').log"

echo "ML Training Pipeline — $RUN_DATE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="

# --- Phase 1: Collect training data ---
echo "[$(date +'%H:%M:%S')] Starting data collection..."
python -m src.wetting_front_tracker.main \
    --collect-ml-data \
    -i "$INPUT_DIR" \
    -d "$RUN_DATE" \
    --ml-training-output "$OUTPUT_DIR" \
    >> "$LOG_FILE" 2>&1

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Data collection failed — $DATASET not found. Check $LOG_FILE"
    exit 1
fi

ROWS=$(wc -l < "$DATASET")
echo "[$(date +'%H:%M:%S')] Data collection complete: $((ROWS - 1)) examples"

# --- Phase 2: Train model ---
echo "[$(date +'%H:%M:%S')] Starting model training..."
python -m src.wetting_front_tracker.main \
    --train-ml-model \
    --ml-training-data "$DATASET" \
    --ml-training-output "$OUTPUT_DIR" \
    --promote-model \
    >> "$LOG_FILE" 2>&1

echo "[$(date +'%H:%M:%S')] Done. Log: $LOG_FILE"

