# Wetting Front Tracker

The Wetting Front Tracker is a Python-based application designed to analyze snowpack data and assess the risk of wet slab avalanches. It processes SNOWPACK `.pro` files, identifies key features related to wet snow instability, and generates an interactive summary map for geographic risk assessment.

## Features

  - **Geospatial Preparation**: Downloads and processes DEMs, classifies avalanche paths by aspect, and links them to the most relevant snowpack model outputs.
  - **Hardware-Adaptive Analysis**: Leverages NVIDIA GPUs via `cupy` for accelerated calculations if available, with a seamless fallback to `numpy` for CPU-only execution.
  - **Wet Slab LOC Detection**: Implements a physically-based model for identifying the Layer of Concern (LOC) based on the formation of a capillary barrier.
  - **Interactive Visualization**: Generates detailed static (Matplotlib) and interactive (Plotly) plots for each analysis, aggregated into a final Folium summary map.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ronimos/wet_snow_tracker
    cd wetting_front_tracker
    ```
2.  **Create the virtual environment and install dependencies:**
    ```bash
    uv sync
    ```
3.  **GPU Support (Optional)**: If you have an NVIDIA GPU and the appropriate CUDA Toolkit installed, you can enable GPU acceleration by adding `cupy` to your `pyproject.toml` or `requirements.txt` and re-running `uv sync`.

## Data Setup

Before running the application, you must provide the primary input geodataset.

  * Place your main avalanche path boundaries file in the following location: `data/reference/Paths.geojson`.

## Configuration

The application is configured using an `.env` file in the project's root directory. Create a file named `.env` and add the following variables.

```bash
# .env.example

# --- Required for DEM downloads ---
# Get your free key from https://portal.opentopography.org/myopentopo
OPENTOPO_API_KEY="YOUR_API_KEY_HERE"

# --- Data Source Configuration ---
PRO_FILES_SOURCE="local"
REMOTE_PRO_FILES_URL="s3://your-bucket/pro-files/"

# --- Path Configuration (Overrides) ---
PRO_FILES_INPUT_DIR="./your-local/pro-files"
WFT_RESULTS_OUTPUT_DIR="path/to/where to save the map"
WFT_ASSETS_OUTPUT_DIR= WFT_ASSETS_OUTPUT_DIR / plot_assets"
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENTOPO_API_KEY` | **Required for initial setup.** Your API key for the [OpenTopography](https://portal.opentopography.org/myopentopo) service. The script uses this to automatically download DEMs if they aren't found locally. | `YOUR_API_KEY_HERE` |
| `PRO_FILES_SOURCE` | Sets the source for `.pro` files. Can be `local` or `remote`. | `local` |
| `REMOTE_PRO_FILES_URL`| The base URL to download `.pro` files from if `PRO_FILES_SOURCE` is `remote`. | `""` |
| `PRO_FILES_INPUT_DIR` | The default local directory to read `.pro` files from. | `./data/input` |
| `WFT_RESULTS_OUTPUT_DIR` | The default directory to save the final `summary_map.html`.| `./results` |
| `WFT_ASSETS_OUTPUT_DIR`| The default directory to save all plot assets (PNGs, HTML plots).| `./results/plot_assets` |

## Usage

Before running the commands manually, activate your virtual environment:

```bash
source .venv/bin/activate
```

### Initial Data Preparation

The first time you run the application, or any time the input `Paths.geojson` changes, you must run it with the `--regenerate-data` flag.

This step performs all necessary geospatial processing and creates a manifest of your `.pro` files for faster subsequent runs. If a processed DEM is not found in `data/processed/`, the script will automatically download the necessary elevation data from OpenTopography using your API key.

```bash
python -m src.wetting_front_tracker.main --regenerate-data
```

### Standard Run

For daily analysis, run the script with the desired date. It will use the pre-processed data and manifest to run quickly.

```bash
python -m src.wetting_front_tracker.main --date "2025-05-09"
```

### Development & Testing

For faster setup and iteration, you can use a small subset of test data. Open the `src/wetting_front_tracker/param_config.py` file and set the following flag:

  * **`USE_TEST_DATA = True`**: This will make the script use `Paths_test.geojson` instead of the full `Paths.geojson` file, dramatically speeding up the data preparation and analysis steps.

### Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--regenerate-data`| Force regeneration of all processed geodata and the `.pro` file manifest. | `False` |
| `-d`, `--date` | The central date for the analysis (e.g., `'YYYY-MM-DD'`). | Today's date |
| `-i`, `--input-dir` | Override the directory for `.pro` files. | Value from `.env` or `./data/input` |
| `-o`, `--output-dir` | Override the directory for the final summary map. | Value from `.env` or `./results` |
| `-a`, `--assets-dir` | Override the directory for plot assets. | Value from `.env` or `./results/plot_assets` |

### Using the Shell Script

A pre-configured shell script, `run_analysis.sh`, is provided for convenience, especially for automated runs (e.g., via cron). It sets environment variables, activates the virtual environment, and runs the main script with today's date.

1.  **Make it executable:**
    ```bash
    chmod 754 run_analysis.sh
    ```
2.  **Run it:**
    ```bash
    ./run_analysis.sh
    ```