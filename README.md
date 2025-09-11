# Wetting Front Tracker for Snowpack Analysis
## Overview
This application automates the analysis of wet slab avalanche potential by processing SNOWPACK model output files (.pro), correlating them with specific terrain features, and visualizing the risk over time.

It identifies potential weak layers within the snowpack, tracks the penetration of liquid water, and calculates the time it takes for the wetting front to reach these critical layers.

The final output is an interactive map that provides a geographic overview of wet slab risk for a given set of avalanche paths or polygons.

Example of the final summary_map.html output

## Features
- **Automated Geodata Preparation:** Downloads and mosaics DEM data from the OpenTopography API.

- **Terrain Analysis:** Splits polygons (e.g., avalanche paths) into sub-polygons by aspect (N, E, S, W).

- **Intelligent Data Linking:** Matches aspect polygons to the most relevant SNOWPACK `.pro` files.

- **Parallel Processing:** Leverages all CPU cores to analyze hundreds of snowpack profiles efficiently.

- **Advanced Snowpack Metrics:**

  - Calculates snow depth

  - Identifies weak layers (faceted crystals/depth hoar)

  - Tracks wetting front depth using Liquid Water Content (LWC)

  - Applies persistence logic to track weak layers through melt events

- Rich Visualization:

  - Static Matplotlib PNG plots

  - Interactive Plotly HTML plots

  - Final Folium map (`summary_map.html`) with color-coded polygons and interactive popups

## Project Structure
```
wetting-front-tracker/
├── data/
│   ├── reference/         <-- Input files (Paths.geojson, snowpack metadata)
│   ├── processed/         <-- Intermediate geodata
│   └── input/             <-- Sample .pro files (for dev/testing)
│
├── results/
│   ├── summary_map.html   <-- Final interactive map
│   └── plot_assets/       <-- Generated plots (PNG + HTML)
│
├── src/
│   └── wetting_front_tracker/
│       ├── main.py        <-- Main CLI script
│       ├── param_config.py  <-- User config file (API keys, paths, etc.)
│       └── ... (modules)
│
├── .env                   <-- For storing API keys (not committed)
├── pyproject.toml         <-- Dependencies
└── README.md
```
## Setup and Installation
### Prerequisites
- Python 3.9+

- `pip`

## Steps
```
# Clone the repository
git clone <your-repository-url>
cd wetting-front-tracker

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Linux/macOS
.venv\Scripts\activate      # on Windows

# Install the project and its dependencies
pip install .
```
## Configuration
### Step 1: Input Data Files
Place the following files inside the `data/reference/` directory.

`Paths.geojson`
A standard GeoJSON file containing `Polygon` or `MultiPolygon` features representing your areas of interest (e.g., avalanche paths).

- CRS: WGS84 (EPSG:4326) is recommended.

- Required Properties: Each feature's `properties` object must contain:

  - `pathName`: A unique string identifier for the path (e.g., "Main Gully").

`snowpack_locations_with_metadata.csv`
This CSV file links each SNOWPACK model output file to its geographic context.

- **Required Columns:**

  - `latitude`: The latitude of the model point (decimal degrees, WGS84).

  - `longitude`: The longitude of the model point (decimal degrees, WGS84).

  - `aspect`: The aspect of the slope at the model point. This can be in degrees (0-360) or the string "Flat".

  - `path`: The absolute file path to the corresponding `.pro` file as it exists on the production machine where the analysis will run.

### Step 2: Environment and Path Configuration
#### OpenTopography API Key
This is required for downloading Digital Elevation Models (DEMs).

- **Preferred Method (**`.env` **file):**

  1. Create a file named `.env` in the project's root directory.

  2. Add the following line to the file, replacing `your_key_here` with your actual key:

    ```
    OPENTOPO_API_KEY="your_key_here"
    ```

  - **Alternative Method:**

    - Open `src/wetting_front_tracker/param_config.py` and hardcode your key by replacing `"YOUR_API_KEY_HERE"`.

### SNOWPACK File Paths
Open `src/wetting_front_tracker/param_config.py` and configure the base paths for your `.pro` files.

- `PRO_FILES_BASE_PATH_PROD`: The base directory for `.pro` files on your production machine (e.g., a Linux server). This path should match the base used in your `snowpack_locations_with_metadata.csv`.

- `PRO_FILES_BASE_PATH_DEV`: The directory where you store local `.pro` files for development and testing (defaults to `data/input/`). The script uses this to find local copies when run on a development machine (like Windows).

## How It Works
The application follows a multi-stage pipeline:

1. Geospatial Pre-processing (`prepare_geodata.py`): The input polygons are first split by cardinal aspect (N, E, S, W) using a Digital Elevation Model (DEM). Each of these new, smaller polygons is then spatially matched to its nearest SNOWPACK model output location that shares the same aspect. This creates the primary analysis-ready file: `linked_aspect_polygons.geojson`.

2. Snowpack Data Reading (`snowpack_reader.py`): The SnowpackProfile class efficiently parses the `.pro` files. It uses `xarray` to hold the data and can leverage `CuPy` for GPU-accelerated calculations if a compatible GPU is detected.

3. Analysis (`wet_front_tracker.py`): A series of analysis functions are applied to each day's profile to identify weak layers and track the wetting front. The key output is the `time_to_loc` metric, which is calculated for a specific reference date.

4. Visualization (`plotting.py`): The results are rendered into static PNGs, interactive HTML plots, and the final Folium summary map.

## Running the Analysis
The project is installed as a command-line script named `main`.
```
# Standard run using the current date and time
main

# Run for a specific date (time defaults to noon)
main --date "2025-03-15"

# Run for a specific date and time (will be rounded to the closest synoptic time)
main --date "2025-03-15 18:00"

# Force regeneration of DEMs, aspect polygons, and linked files
main --regenerate-data
```
## Understanding the Output
- `results/summary_map.html`: This is the main interactive output.

  - **Color-Coded Polygons:** Represent the `time_to_loc` (time for the wetting front to reach the weak layer), indicating risk level.

  - **Tooltip:** Hover over a polygon to see its name, aspect, analysis date, and a thumbnail of its plot.

  - **Popup:** Click on a polygon to get a link to the detailed interactive chart.

- `results/plot_assets/`: Contains the individual plots generated for each analysis.

  - `*_wetting_front.png`: A static, high-resolution PNG plot created with Matplotlib.

  - `*_wetting_front.html`: A fully interactive HTML plot created with Plotly.

## Key Dependencies
- **Geospatial:** `geopandas`, `rasterio`, `rioxarray`, `shapely`

- **Data Handling:** `pandas`, `xarray`, `numpy`

- **Visualization:** `folium`, `matplotlib`, `plotly`

- **Performance:** `numba`, `cupy` (optional, for GPU)

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and open a pull request.

## License
This project is licensed under the MIT License.

*README last updated: September 11, 2025*