# Wetting Front Tracker

A geospatial analysis tool for assessing **wet slab avalanche risk** by processing **SNOWPACK simulation files** and terrain data.

This project provides a complete pipeline to analyze snowpack stability across a region. It automatically downloads and processes Digital Elevation Models (DEMs), classifies terrain by aspect, links avalanche paths to the most relevant snowpack simulations, and generates a final interactive risk map.

---

## Core Features

### Automated Geospatial Data Preparation
- Downloads high-resolution DEMs from **OpenTopography**, fetching only the tiles that intersect input polygons.
- Classifies terrain into cardinal aspects (North, East, South, West).
- Generates smoothed GeoJSON files, splitting large polygons by aspect.

### Intelligent Data Linking
- Reads a manifest of available **SNOWPACK (`.pro`) simulations**.
- For each aspect-classified polygon, finds the closest `.pro` file with a matching aspect.
- Links terrain polygons directly to snowpack simulations.

### Advanced Snowpack Analysis
- Uses **multiprocessing** to analyze hundreds of `.pro` files in parallel.
- Computes daily time series of:
  - Snow depth (HS)
  - Weak layer location (LOC)
  - Wetting front depth
- Tracks when (in hours) the wetting front reaches the weak layer.

### Rich Visualization
- Generates:
  - Static **PNG plots** (Matplotlib)
  - Interactive **HTML plots** (Plotly)
- Builds an **interactive Folium summary map**:
  - Polygons colored by risk level (High, Watch, Low).
  - Tooltips display plots on hover.
  - Popups link to interactive HTML simulations.

---

## 📂 Project Structure

```
wetting_front_tracker/
│
├── .env                  # Stores your API key
├── .gitignore
├── pyproject.toml        # Dependencies and configuration
├── README.md             # This file
│
└── src/
    └── wetting_front_tracker/
        ├── __init__.py
        ├── param_config.py         # Config and file paths
        ├── prepare_geodata.py      # DEMs and aspect classification
        ├── extract_pro_metadata.py # Generates locations CSV
        ├── snowpack_reader.py      # Reads .pro files
        ├── wet_front_tracker.py    # Core snowpack analysis
        ├── plotting.py             # Plotting and mapping
        └── main.py                 # Orchestrates the workflow
```

---

## ⚙️ Setup and Installation

### 1. Environment Setup
This project uses **[uv](https://github.com/astral-sh/uv)** for environment management.

```bash
# Create the virtual environment
uv venv

# Activate the environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip sync pyproject.toml
```

### 2. API Key Configuration
- Create a free account at [OpenTopography](https://opentopography.org).
- Copy your API key from **My Account**.
- Add it to a `.env` file in the project root:

```bash
OPENTOPO_API_KEY="YOUR_ACTUAL_API_KEY"
```

### 3. Input Data
Place inputs under `data/reference/`:
- **Polygons GeoJSON**: Large polygons for avalanche paths (default: `HighwayPaths.geojson`).
- **Snowpack Locations CSV**: Metadata for `.pro` simulations (default: `snowpack_locations_with_metadata.csv`).
  - Generate with `extract_pro_metadata.py`.

---

## ▶️ Running the Analysis

### Step 1: Full Pipeline
First run downloads DEMs, prepares geodata, and runs the analysis:

```bash
uv run main
```

This will:
- Prepare geodata and classify aspects.
- Generate a manifest (`pro_file_manifest.txt`) of `.pro` files.
- Run snowpack analysis in parallel.
- Produce plots and the summary map.

### Step 2: Subsequent Runs
Later runs skip preprocessing for faster execution:

```bash
uv run main
```

### Forcing Data Regeneration
If you update polygons or DEMs:

```bash
uv run main -- --regenerate-data
```

*(Note the `--` separates uv args from script args.)*

---

## 📊 Outputs

- **Processed Data** (`data/processed/`):
  - `dem.tif` — Mosaicked DEM
  - `aspect_polygons.geojson` — Aspect-classified polygons
  - `linked_aspect_polygons.geojson` — Final polygons with `pro_file_path`
  - `pro_file_manifest.txt` — List of `.pro` files analyzed

- **Results** (`results/`):
  - Daily plots (PNGs + HTMLs)
  - `summary_map.html` — Final interactive Folium map

---

## 🗺 Example Output
An interactive map where avalanche paths are shaded by wet slab risk. Hover to preview plots, click to open interactive simulations.

---

## 📜 License
[MIT License](LICENSE) – free to use and modify.

---
