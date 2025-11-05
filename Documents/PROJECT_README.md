# Wetting Front Tracker

A comprehensive geospatial analysis tool for tracking and visualizing wet snow avalanche risk by monitoring wetting front progression in alpine snowpacks.

## Overview

The Wetting Front Tracker analyzes SNOWPACK model output to identify and track wet snow conditions that can lead to avalanche hazards. The application processes snowpack profile data, identifies weak layers (Layer of Concern - LOC), tracks the progression of liquid water through the snowpack, and generates interactive maps showing risk levels across multiple avalanche paths.

### Key Features

- **Automated Weak Layer Detection**: Identifies structural weaknesses (faceted crystals, depth hoar) in snowpack profiles
- **Wetting Front Tracking**: Monitors the downward progression of liquid water through the snowpack
- **Time-to-LOC Calculation**: Predicts when the wetting front will reach critical weak layers
- **Water Content Analysis**: Calculates average free water content above weak layers (NEW)
- **Multi-Level Visualization**: 
  - Static high-resolution plots (PNG)
  - Interactive time-series plots (HTML/Plotly)
  - Geospatial risk maps (Folium)
- **Parallel Processing**: Efficient multi-core processing for large-scale analyses
- **GPU Acceleration**: Optional CUDA support for faster calculations
- **Geospatial Integration**: Automatic DEM acquisition and aspect-based polygon splitting

## Project Structure

```
wetting_front_tracker/
├── src/
│   └── wetting_front_tracker/
│       ├── main.py                    # Main orchestrator
│       ├── snowpack_reader.py         # SNOWPACK .pro file parser
│       ├── wet_front_tracker.py       # Analysis algorithms
│       ├── plotting.py                # Visualization engine
│       ├── prepare_geodata.py         # Geospatial preprocessing
│       ├── param_config.py            # Configuration parameters
│       └── util.py                    # Utility functions
├── data/
│   ├── input/                         # SNOWPACK .pro files
│   ├── reference/                     # Metadata CSV, polygons
│   └── results/                       # Output plots and maps
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- GDAL libraries (for geospatial operations)
- Optional: CUDA-capable GPU for acceleration

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd wetting_front_tracker
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```
numpy
pandas
xarray
geopandas
rasterio
rioxarray
shapely
scipy
matplotlib
plotly
folium
branca
Pillow
tqdm
requests
numba
```

**Optional (for GPU acceleration):**
```bash
pip install cupy-cuda12x  # Replace with your CUDA version
```

## Data Requirements

### Input Data

#### 1. SNOWPACK Profile Files (.pro)
- Format: SNOWPACK model output files
- Location: `data/input/*.pro`
- Required metadata in header:
  - `Latitude=`
  - `Longitude=`
  - `Altitude=`
  - `StationName=`
  - `SlopeAngle=`
  - `SlopeAzi=`
- Required data columns:
  - Timestamp (0500)
  - Height (0501)
  - Density (0502)
  - Temperature (0503)
  - Liquid Water Content (0506)
  - Grain type (0507)
  - Grain size (0512)

#### 2. Avalanche Path Polygons (Optional)
- Format: Shapefile or GeoJSON
- Location: `data/reference/avalanche_paths.shp`
- If not provided, will be generated from SNOWPACK locations

### Reference Data

The application generates:
- `snowpack_locations_with_metadata.csv` - Station metadata
- `pro_files_manifest.json` - File index for quick lookup
- Aspect-split polygons with linked .pro files

## Usage

### Quick Start

Run analysis with default settings:
```bash
python -m src.wetting_front_tracker.main
```

### Command Line Options

```bash
python -m src.wetting_front_tracker.main [OPTIONS]
```

**Available Options:**
- `--date YYYY-MM-DD` - Analysis reference date (default: today)
- `--start-date YYYY-MM-DD` - Plot window start date
- `--end-date YYYY-MM-DD` - Plot window end date
- `--regenerate-data` - Force regeneration of geospatial data
- `--output-dir PATH` - Custom output directory
- `--workers N` - Number of parallel workers (default: CPU count)

**Examples:**

```bash
# Analyze specific date
python -m src.wetting_front_tracker.main --date 2025-05-15

# Custom date range for plots
python -m src.wetting_front_tracker.main \
    --date 2025-05-15 \
    --start-date 2025-05-01 \
    --end-date 2025-05-31

# Regenerate geospatial data
python -m src.wetting_front_tracker.main --regenerate-data

# Use specific number of workers
python -m src.wetting_front_tracker.main --workers 8
```

### Configuration

Edit `param_config.py` to customize:

```python
# Analysis parameters
ANALYSIS_DAYS_BEFORE = 7      # Days to look back
ANALYSIS_DAYS_AFTER = 7       # Days to look ahead

# Thresholds
LWC_THRESHOLD_PERCENT = 4.0   # Wetting front threshold (%)
MIN_GS_DIFFERENCE = 0.5       # Grain size difference (mm)

# Paths
RESULTS_PATH = Path("data/results")
INPUT_PATH = Path("data/input")
REFERENCE_PATH = Path("data/reference")

# Processing
MAX_WORKERS = None            # None = use all CPUs
```

## Workflow

### 1. Data Preparation Phase

```
Input .pro files
    ↓
Parse metadata (util.py)
    ↓
Create metadata CSV
    ↓
Generate/load polygons
    ↓
Fetch DEMs from OpenTopography
    ↓
Split polygons by aspect
    ↓
Link polygons to .pro files
```

### 2. Analysis Phase (Parallel)

For each polygon:
```
Load SNOWPACK profile
    ↓
Calculate time series:
    - Total snow depth
    - Weak layer location
    - Wetting front position
    - Water content metrics
    ↓
Apply LOC persistence logic
    ↓
Calculate time-to-LOC
    ↓
Calculate avg LWC above LOC
    ↓
Generate plots (PNG + HTML)
    ↓
Return results
```

### 3. Visualization Phase

```
Aggregate results
    ↓
Create GeoDataFrame
    ↓
Apply color coding:
    - Red: avg LWC > 3%
    - Yellow: avg LWC 1-3%
    - Time-based colors otherwise
    ↓
Generate summary map
    ↓
Save to results directory
```

## Output Files

### Directory Structure

```
data/results/
├── summary_map.html              # Interactive overview map
├── map_data.geojson             # Map data (for external use)
└── plot_assets/
    ├── polygon_001_N.png        # High-res static plots
    ├── polygon_001_N_thumb.png  # Thumbnails for map
    ├── polygon_001_N.html       # Interactive plots
    └── ...
```

### Summary Map (`summary_map.html`)

Interactive Folium map with:
- **Polygon colors** indicating risk level
- **Tooltips** with thumbnail previews
- **Popups** linking to detailed interactive plots
- **Multiple basemaps** (Street, Topo, Satellite)
- **Legend** explaining color codes
- **Persistent view** (remembers zoom/pan between sessions)

**Color Legend:**
- 🔴 **Red**: Avg free water content > 3% above LOC (HIGH RISK)
- 🟡 **Yellow**: Avg free water content 1-3% above LOC (ELEVATED RISK)
- 🟥 **Dark Red**: Wetting front reaches LOC in 0-24h (IMMINENT)
- 🟧 **Orange**: 24-48h until LOC reached
- 🟨 **Yellow**: 48-72h until LOC reached
- 🔵 **Light Blue**: LOC reached 24-48h ago
- 🟦 **Dark Blue**: LOC reached 48-72h ago
- ⚪ **Gray**: No data or other

### Static Plots (PNG)

High-resolution (300 DPI) plots showing:
- **Top panel**: Time series of:
  - Snow depth (HS)
  - Weak layer height (LOC)
  - Wetting front position
  - Highest wet point
- **Bottom panel**: Heatmap of liquid water content vs height vs time

### Interactive Plots (HTML)

Plotly-based interactive versions with:
- Zoom/pan capabilities
- Hover information
- Linked to external snowpack viewer
- Embedded in clean HTML template

## Analysis Metrics

### Time-to-LOC
Hours until wetting front reaches the weak layer, measured from reference date:
- Positive values: Future (e.g., +24h = 24 hours from now)
- Negative values: Past (e.g., -12h = 12 hours ago)
- NaN: Wetting front doesn't reach LOC during analysis window

### Average LWC Above LOC (NEW)
Mean liquid water content of all snowpack layers above the weak layer:
- Calculated as percentage (0-100%)
- Updated at each time step
- Used for priority-based polygon coloring
- Indicates potential for wet slab avalanche

### Weak Layer Detection
Identifies Layer of Concern (LOC) using:
- Grain type classification (faceted crystals, depth hoar)
- Grain size differences between layers
- Structural weaknesses in bottom half of snowpack

### Wetting Front Tracking
Monitors downward progression of liquid water:
- Threshold: 4% volumetric water content (configurable)
- Tracks the lowest point in snowpack exceeding threshold
- Identifies active melt events

## Advanced Features

### LOC Persistence Logic

The weak layer can appear/disappear in SNOWPACK output due to grain metamorphism. The tracker:
1. Identifies the start of the most recent melt event
2. Locks onto the last known weak layer before melt began
3. Tracks that layer forward in time
4. Updates if a higher weak layer appears (real structural change)

This prevents false negatives from weak layer "disappearance" in model output.

### GPU Acceleration

If CuPy is installed and CUDA GPU is available:
- Automatically uses GPU for array operations
- Significantly faster for large datasets
- Falls back to NumPy/CPU if GPU unavailable
- Status logged at startup

### Parallel Processing

- Multi-core processing using Python multiprocessing
- Progress bar shows real-time status
- Configurable worker count
- Automatic load balancing

### DEM Acquisition

Automatically fetches elevation data from OpenTopography API:
- 1-arcsec resolution (SRTM Global)
- Automatic retry logic for failed requests
- Merges multiple tiles if needed
- Caches downloaded DEMs

## Troubleshooting

### Common Issues

**1. No .pro files found**
- Check `data/input/` directory exists
- Verify .pro files are in correct location
- Check file permissions

**2. GDAL/Rasterio errors**
- Install GDAL system libraries first
- On Ubuntu: `sudo apt-get install gdal-bin libgdal-dev`
- On macOS: `brew install gdal`
- On Windows: Use OSGeo4W installer

**3. Memory errors with large datasets**
- Reduce number of workers: `--workers 4`
- Process smaller date ranges
- Close other applications

**4. All polygons showing gray**
- Verify .pro files contain valid data
- Check date range covers available data
- Review logs for parsing errors

**5. Import errors**
- Ensure all files in correct directory structure
- Check relative import paths match structure
- Verify all dependencies installed

### Debug Mode

Enable detailed logging:
```python
# In main.py or any module
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### For Large Datasets

```python
# param_config.py
MAX_WORKERS = 16              # Use more cores
MATPLOTLIB_DPI = 150          # Reduce plot resolution
THUMBNAIL_MAX_SIZE = (600, 400)  # Smaller thumbnails
```

### For Limited Memory

```python
MAX_WORKERS = 4               # Fewer parallel processes
# Process in batches by modifying date ranges
```

### For Faster Processing

```python
# Install CuPy for GPU acceleration
pip install cupy-cuda12x

# Use more workers
python -m src.wetting_front_tracker.main --workers 32
```

## Scientific Background

### Wet Snow Avalanches

Wet snow avalanches occur when liquid water infiltrates the snowpack and reaches a structural weak layer, reducing strength and triggering failure. This application helps forecast these events by:

1. **Identifying vulnerable structure**: Detecting weak layers (facets, depth hoar)
2. **Monitoring water infiltration**: Tracking wetting front progression
3. **Predicting failure timing**: Calculating when water reaches weak layers
4. **Assessing current risk**: Quantifying water content above weak layers

### SNOWPACK Model

This tool processes output from the SNOWPACK model, a physics-based snowpack evolution model that simulates:
- Heat transfer
- Water percolation
- Snow metamorphism
- Layer formation and settling

Learn more: https://models.slf.ch/p/snowpack/

## Data Sources

- **SNOWPACK profiles**: Numerical weather prediction model output
- **DEM data**: OpenTopography (SRTM 1-arcsec)
- **Avalanche paths**: User-provided or generated from station locations

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Authors

- **Ron Simenhois** - Primary developer
- **Itai** - Analysis algorithms

## Acknowledgments

- SNOWPACK model: WSL Institute for Snow and Avalanche Research SLF
- OpenTopography for DEM data
- Mountain Weather Information for visualization infrastructure

## Citation

If you use this tool in your research, please cite:

```
[Add citation information here]
```

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [your contact information]

## Version History

- **v1.1** (November 2025) - Added average LWC above LOC coloring
- **v1.0** (October 2025) - Initial release

---

**Last Updated:** November 2025
