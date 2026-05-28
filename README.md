# Wetting Front Tracker

An avalanche forecasting system to help forecasters predicting wet slab avalanche risk by analyzing liquid water infiltration into snowpack layers. The system combines SNOWPACK model output with machine learning to identify when and where water penetration reaches critical weak layers.

## Overview

The Wetting Front Tracker addresses a critical challenge in avalanche forecasting: **predicting when liquid water infiltration will destabilize snowpack layers**. Unlike dry slab avalanches that depend primarily on loading and weak layer structure, wet slab avalanches occur when liquid water (from melt or rain) penetrates to structural weaknesses, reducing shear strength and triggering failure.

### Key Capabilities

- **Multi-criteria risk assessment**: Combines wetting front depth, Layer of Concern (LOC) detection, and liquid water content analysis
- **Time-based forecasting**: Predicts when wetting front will reach critical layers (0-72 hour window)
- **Machine learning LOC detection**: Trained model identifies weak layers with 98.97% ROC-AUC
- **Interactive visualization**: Generates detailed plots and risk maps for operational decision-making
- **Geospatial analysis**: Processes aspect-specific polygons for terrain-aware forecasting

## Scientific Background

### Wet Slab Avalanche Mechanism

Wet slab avalanches occur through a well-understood physical process:

1. **Liquid water introduction**: Surface melt or rain introduces liquid water into snowpack
2. **Wetting front propagation**: Water percolates downward through snow layers
3. **Interface stress**: Water accumulates at layer boundaries, particularly at capillary barriers
4. **Strength reduction**: Liquid water reduces grain-to-grain bonds and cohesion
5. **Failure initiation**: When shear stress exceeds reduced strength, slab releases

### Critical Thresholds

The system uses physically-motivated thresholds derived from snow science literature:

- **Wetting Front (4% LWC)**: Volumetric liquid water content ≥ 4% indicates significant weakening (Colbeck, 1982; Techel & Pielmeier, 2011; Baggi & Schweizer, 2009)
- **Mean LWC (3%)**: Sustained 3% LWC above weak layers indicates dangerous saturation (Mitterer et al., 2011; Mitterer & Schweizer, 2013)
- **Layer of Concern**: Structural weaknesses where failure is likely (facets, depth hoar, capillary barriers) (Schweizer et al., 2003; Baggi & Schweizer, 2009)

### Detection Challenges

Traditional avalanche forecasting relies on point observations and manual snowpack assessment. The Wetting Front Tracker overcomes key limitations:

- **Temporal coverage**: Hourly SNOWPACK model input vs. daily observations
- **Spatial coverage**: Automated processing of hundreds of locations
- **Objective criteria**: ML-based LOC detection removes observer bias
- **Forward-looking**: Forecasts when conditions will become critical

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Input Data                           │
│  • SNOWPACK .pro files (time series profiles)            │
│  • Polygon geometries (aspect-specific terrain units)    │
│  • Digital Elevation Model (for aspect calculation)      │
└────────────────────┬─────────────────────────────────────┘
                     │
                     v
┌──────────────────────────────────────────────────────────┐
│                 Data Processing Layer                    │
│  • Profile reading via xsnow (snowpack_reader.py)        │
│  • Geospatial polygon linking (prepare_geodata.py)       │
│  • Time series extraction                                │
└────────────────────┬─────────────────────────────────────┘
                     │
                     v
┌──────────────────────────────────────────────────────────┐
│                  Analysis Engine                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Wetting Front Detection (wet_front_tracker.py)    │  │
│  │  • LWC threshold analysis (4%)                     │  │
│  │  • Deepest wet layer identification                │  │
│  │  • Mean LWC calculation (3% threshold)             │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  LOC Detection (ml_loc_detector.py)                │  │
│  │  • ML-based prediction (XGBoost/LightGBM)          │  │
│  │  • Rule-based fallback (capillary barriers)        │  │
│  │  • Hybrid mode with confidence scoring             │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Temporal Analysis (find_time_to_loc)              │  │
│  │  • Wetting event isolation                         │  │
│  │  • Penetration timing calculation                  │  │
│  │  • Multi-candidate worst-case selection            │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Risk Synthesis                                    │  │
│  │  • Time-to-LOC priority ranking                    │  │
│  │  • Mean LWC threshold evaluation                   │  │
│  │  • Color code assignment                           │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     v
┌──────────────────────────────────────────────────────────┐
│                 Visualization Layer                      │
│  • Matplotlib plots (wetting front evolution)            │
│  • Plotly interactive plots (zoomable time series)       │
│  • Folium maps (geospatial risk visualization)           │
└──────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- SNOWPACK model output files (.pro format)
- Sufficient disk space for time series data

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd wetting_front_tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure paths (create .env file)
cat > .env << EOF
PRO_FILES_SOURCE=local
PRO_FILES_INPUT_DIR=/path/to/snowpack/output
WFT_RESULTS_OUTPUT_DIR=/path/to/results
EOF

# Run analysis
python -m src.wetting_front_tracker.main --loc-mode ml_only --date "2025-04-06 18:00"
```

### Dependencies

Core libraries:
- `xsnow`: SNOWPACK `.pro` file I/O (replaces the legacy custom parser)
- `xarray`: Multi-dimensional data handling
- `pandas`, `numpy`: Data analysis
- `geopandas`, `shapely`: Geospatial operations
- `scikit-learn`, `xgboost`, `lightgbm`: Machine learning
- `matplotlib`, `plotly`, `folium`: Visualization

## Usage

### Basic Analysis

```bash
# Run with ML LOC detection (recommended)
python -m src.wetting_front_tracker.main \
    --loc-mode ml_only \
    --date "2025-04-06 18:00"

# Run with specific model and threshold
python -m src.wetting_front_tracker.main \
    --loc-mode ml_only \
    --ml-model-path assets/models/production \
    --ml-threshold 0.5

# Run with diagnostics
python -m src.wetting_front_tracker.main \
    --loc-mode ml_only \
    --enable-diagnostics
```

### LOC Detection Modes

**Rule-based** (default, no ML required):
```bash
python -m src.wetting_front_tracker.main --loc-mode rule_based
```
- Detects capillary barriers (small grains over large FC/DH)
- Identifies structural weaknesses (large over small, non-wet)
- Fast but may miss complex patterns

**ML-only** (requires trained model):
```bash
python -m src.wetting_front_tracker.main --loc-mode ml_only
```
- Uses trained XGBoost/LightGBM model
- 98.97% ROC-AUC performance
- Detects subtle patterns in layer structure

**Hybrid** (ML with rule-based fallback):
```bash
python -m src.wetting_front_tracker.main --loc-mode hybrid
```
- Tries ML first, falls back to rules if low confidence
- Best of both approaches
- Recommended for operational use

### Training New ML Models

```bash
# Step 1: Collect training data
python -m src.wetting_front_tracker.main \
    --collect-ml-data \
    --output-dir results/training_data

# Step 2: Train models
python -m src.wetting_front_tracker.main \
    --train-ml-model \
    --ml-training-data results/training_data/features_dataset.parquet \
    --ml-training-output results/trained_models \
    --ml-training-models xgboost lightgbm random_forest

# Step 3: Promote to production (optional)
python -m src.wetting_front_tracker.main \
    --train-ml-model \
    --ml-training-data results/training_data/features_dataset.parquet \
    --promote-model
```

## Output Products

### 1. Wetting Front Plots (PNG)

**Location**: `results/plot_assets/{station}_{aspect}_wetting_front.png`

Static plots showing:
- Total snow depth (HS) time series
- Wetting front depth (LWC ≥ 4% threshold)
- LOC height(s) with confidence
- Liquid water content heatmap
- Central analysis date marker

**Use case**: Detailed forensic analysis, documentation

### 2. Interactive Plots (HTML)

**Location**: `results/plot_assets/{station}_{aspect}_wetting_front.html`

Plotly-based interactive visualization:
- Zoom, pan, hover for details
- Multiple LOC candidates visible
- Full time series exploration
- Exportable as PNG from browser

**Use case**: Exploratory analysis, presentations

### 3. Risk Map (HTML)

**Location**: `results/summary_map.html`

Folium geospatial map with:
- Color-coded polygons (risk level)
- Interactive tooltips (thumbnail plots)
- Clickable popups (links to detailed plots)
- Multiple base layers (street, topo, satellite)
- Map state persistence

**Use case**: Operational briefings, spatial decision-making

### Color Code Interpretation

| Color | Time Range | Mean LWC | Interpretation |
|-------|-----------|----------|----------------|
| **Dark Red** | 0-24h | - | Front will reach LOC within 24h (IMMINENT) |
| **Orange** | 24-48h | OR ≥3% | Front 24-48h out OR high saturation |
| **Yellow** | 48-72h | - | Front 48-72h out (DEVELOPING) |
| **Red** | -24-0h | - | Front reached LOC recently (ACTIVE) |
| **Light Blue** | -48--24h | - | Front reached 24-48h ago (WANING) |
| **Dark Blue** | -72--48h | - | Front reached 48-72h ago (OLD) |
| **Gray** | N/A | <3% | No risk detected or insufficient data |

## Configuration

### Environment Variables (.env)

```bash
# Data source
PRO_FILES_SOURCE=local                    # 'local' or 'remote'
PRO_FILES_INPUT_DIR=/data/snowpack        # Input directory
REMOTE_PRO_FILES_URL=https://...          # Remote URL (if remote source)

# Output paths
WFT_RESULTS_OUTPUT_DIR=/results           # Results directory
WFT_ASSETS_OUTPUT_DIR=/results/assets     # Plot assets subdirectory

# ML configuration
ML_ENABLED=true                           # Enable ML detection
ML_MODEL_PATH=/path/to/model              # Model directory
ML_PROBABILITY_THRESHOLD=0.5              # Detection threshold
LOC_DETECTION_MODE=ml_only                # 'rule_based', 'ml_only', 'hybrid'

# API keys
OPENTOPO_API_KEY=your_key_here            # OpenTopography (for DEM downloads)
```

### Parameter Configuration (param_config.py)

```python
# LWC thresholds
LWC_THRESHOLD_PERCENT = 4.0      # Wetting front (%)
LWC_THRESHOLD_WET_LAYER = 3.0    # Wet layer detection (%)
MEAN_LWC_THRESHOLD = 3.0         # Mean LWC coloring (%)

# Grain type codes (SNOWPACK)
FC_DH_MIN_CODE = 400             # Faceted crystals min
FC_DH_MAX_CODE = 600             # Depth hoar max
WET_GRAIN_MIN_CODE = 770         # Wet forms min
WET_GRAIN_MAX_CODE = 780         # Wet forms max

# Detection parameters
MIN_GS_DIFFERENCE = 0.5          # Grain size threshold (mm)
```

## Algorithm Details

### Wetting Front Detection

**Function**: `wet_front_lwc(df) -> (lwc, height)`

Identifies the deepest penetration of liquid water:

```python
# Pseudo-code
for each layer in profile (top to bottom):
    if layer.lwc >= 0.04:  # 4% threshold
        wetting_front_height = layer.height
        break
return wetting_front_height
```

**Rationale**: 4% LWC threshold based on research showing significant strength loss at this saturation level.

### Layer of Concern (LOC) Detection

#### Rule-Based Method

**Primary**: Capillary barrier detection
```python
# Small grains over large weak grains = water pooling
if (gs_difference < 0) and (lower_layer is FC/DH):
    return LOC_height
```

**Fallback**: Structural weakness
```python
# Large over small, non-wet top = stress concentration
if (gs_difference > 0) and (upper_layer is not wet):
    return LOC_height
```

#### ML-Based Method

**Features** (per layer, with 24h lookback):
- Interface stress ratio (most important)
- Density contrast
- Temperature gradient
- Grain size difference
- Layer thickness
- Hardness difference
- Time-lagged values

**Model**: Ensemble of XGBoost and LightGBM
- Training: ~50,000 labeled interfaces
- Performance: 98.97% ROC-AUC
- Output: Probability scores for each interface

### Time-to-LOC Calculation

**Function**: `find_time_to_loc(summary_df, reference_date) -> hours`

Multi-step temporal analysis:

1. **Event isolation**: Identify distinct wetting events (separated by dry periods)
2. **Relevant event**: Select event active at reference date
3. **Penetration detection**: Find when `wetting_front_height ≤ LOC_height`
4. **Time calculation**: Hours from reference date to penetration

```python
# Simplified logic
wetting_events = identify_events(summary_df)
current_event = get_event_at(reference_date)

for timestamp in current_event:
    if wetting_front[timestamp] <= LOC_height[timestamp]:
        penetration_time = timestamp
        break

hours = (penetration_time - reference_date).total_seconds() / 3600
return hours
```

**Edge cases**:
- No wetting event: returns NaN (gray polygon)
- Front stalls above LOC: returns NaN
- Multiple LOCs: uses worst-case (earliest) penetration

### Mean LWC Criterion

**Function**: `mean_lwc_above_reference(df, reference_height) -> percentage`

Calculates average liquid water content above a reference:

```python
# Use LOC height if available, else ground (0m)
reference = LOC_height if LOC_detected else 0.0

layers_above = profile[profile.height > reference]
mean_lwc = layers_above.lwc.mean() * 100  # Convert to percentage

if mean_lwc >= 3.0:
    risk_flag = True  # Trigger orange coloring
```

**Rationale**: Sustained 3% LWC indicates dangerous saturation even without discrete wetting front. Captures gradual wetting scenarios missed by 4% threshold.

### Risk Prioritization

When multiple criteria indicate risk, the system uses priority logic:

1. **Mean LWC ≥ 3%**: Orange (highest priority)
2. **Time-to-LOC 0-24h**: Dark red
3. **Time-to-LOC 24-48h**: Orange
4. **Time-to-LOC 48-72h**: Yellow
5. **Time-to-LOC recent past**: Red/Blue
6. **No data**: Gray (lowest priority)

Multiple LOC candidates: worst-case (earliest) time used.

## Performance & Scalability

### Computational Performance

- **Single profile**: ~2-10 seconds
- **Typical run (100 polygons)**: 5-15 minutes
- **Multiprocessing**: Scales with CPU cores (default: cores/4)

### Memory Requirements

- **Per profile**: ~50-200 MB RAM
- **Large runs**: ~2-8 GB RAM
- **Disk**: ~1 MB per polygon (plots + data)

### Optimization Tips

```bash
# Increase workers (if RAM available)
# Edit main.py:
worker_count = int(max(1, cpu_cores / 2))  # More aggressive

# Reduce output resolution
# Edit plotting.py:
MATPLOTLIB_DPI = 150  # Lower DPI (vs 300)

# Skip interactive plots
# Comment out in process_single_profile():
# plot_summary_plotly(...)
```

## Troubleshooting

### Common Issues

**1. Gray polygons (no coloring)**

Diagnosis:
```bash
python -m src.wetting_front_tracker.main \
    --loc-mode ml_only \
    --enable-diagnostics
```

Check diagnostic output:
- LOC detection rate low → Adjust `--ml-threshold` (try 0.3)
- No wetting detected → Check if profiles have LWC data
- Front doesn't reach LOC → Front may be stalling (capillary barrier)

**2. "Cannot unpack NoneType" errors**

Fixed in latest version. Ensure using updated `wet_front_tracker.py` with:
```python
try:
    result = weak_layer_func(df)
    if result is None:
        return None, None
    _, weak_layer_height = result
except (TypeError, ValueError):
    return None, None
```

**3. ML model not found**

```bash
# Check model path
ls -la assets/models/production/model.joblib

# Specify explicitly
python -m src.wetting_front_tracker.main \
    --ml-model-path /full/path/to/model
```

**4. Profile read errors**

Check SNOWPACK file format:
```bash
# Should have [STATION_PARAMETERS] and [DATA] sections
head -50 your_file.pro
```

Ensure `.pro` extension and proper formatting.

### Validation Tools

**Compare detection methods**:
```bash
python compare_loc_detection.py \
    /path/to/file.pro \
    --ml-model assets/models/production \
    --threshold 0.5
```

**Test mean LWC calculation**:
```bash
python test_mean_lwc.py
```

## Development

### Project Structure

```
wetting_front_tracker/
├── src/
│   └── wetting_front_tracker/
│       ├── main.py                    # Main orchestrator
│       ├── wet_front_tracker.py       # Core analysis functions
│       ├── ml_loc_detector.py         # ML detection
│       ├── snowpack_reader.py         # Data I/O (xsnow-backed)
│       ├── plotting.py                # Visualization
│       ├── param_config.py            # Configuration
│       ├── diagnostic_wrapper.py      # Debugging tools
│       └── assets/
│           └── models/                # Trained ML models
├── data/
│   ├── input/                         # SNOWPACK .pro files
│   ├── reference/                     # Polygons, DEMs
│   └── processed/                     # Intermediate data
├── results/                           # Output products
├── tests/                             # Unit tests
└── docs/                              # Documentation
```

### Adding New Features

1. **New analysis metric**: Add to `wet_front_tracker.py`
2. **New visualization**: Update `plotting.py`
3. **New LOC method**: Extend `ml_loc_detector.py`
4. **New data source**: Modify `snowpack_reader.py` (uses `xsnow.read()` internally)

### Testing

```bash
# Run unit tests
pytest tests/

# Test specific module
pytest tests/test_wet_front_tracker.py

# Run with coverage
pytest --cov=src/wetting_front_tracker tests/
```

## References

### Key Scientific Publications

#### Wet Slab Avalanche Mechanics

- **Baggi, S. & Schweizer, J. (2009)**. "Characteristics of wet-snow avalanche activity: 20 years of observations from a high alpine valley (Dischma, Switzerland)." *Natural Hazards*, 50, 97-108. [https://doi.org/10.1007/s11069-008-9322-7](https://doi.org/10.1007/s11069-008-9322-7)
  - Key finding: Wet slab instability strongly depends on snowpack properties including isothermal state and capillary barriers

- **Conway, H. & Raymond, C.F. (1993)**. "Snow stability during rain." *Journal of Glaciology*, 39(133), 635-642. [https://doi.org/10.1017/s0022143000016531](https://doi.org/10.1017/s0022143000016531)
  - Documented mechanisms of strength loss during rain-on-snow events

- **Armstrong, R.L. (1976)**. "Wet snow avalanches." In *Avalanche Release and Snow Characteristics*, San Juan Mountains, Colorado. INSTAAR Occasional Paper No. 19, 67-82.
  - Foundational research on wet avalanche formation

#### Liquid Water Content & Snow Strength

- **Colbeck, S.C. (1982)**. "An overview of seasonal snow metamorphism." *Reviews of Geophysics*, 20(1), 45-61. [https://doi.org/10.1029/RG020i001p00045](https://doi.org/10.1029/RG020i001p00045)
  - Established relationship between LWC and snow strength reduction

- **Techel, F., Pielmeier, C., et al. (2011)**. "Point observations of liquid water content in wet snow." *The Cryosphere*, 5, 405-418. [https://doi.org/10.5194/tc-5-405-2011](https://doi.org/10.5194/tc-5-405-2011)
  - Field measurements of LWC with Snow Fork and Denoth meter

- **Mitterer, C., Hirashima, H., & Schweizer, J. (2011)**. "Wet-snow instabilities: comparison of measured and modelled liquid water content and snow stratigraphy." *Annals of Glaciology*, 52(58), 201-208.
  - Compared measured vs. modeled LWC in wet snow instabilities
  - Established 3% LWC threshold for wet slab risk assessment

- **Wever, N., Würzer, S., Fierz, C., & Lehning, M. (2016)**. "Simulating ice layer formation under the presence of preferential flow in layered snowpacks." *The Cryosphere*, 10, 2731-2744. [https://doi.org/10.5194/tc-10-2731-2016](https://doi.org/10.5194/tc-10-2731-2016)
  - Analysis of liquid water flow through layered snow

#### SNOWPACK Model

- **Bartelt, P. & Lehning, M. (2002)**. "A physical SNOWPACK model for the Swiss avalanche warning Part I: Numerical model." *Cold Regions Science and Technology*, 35(3), 123-145. [https://doi.org/10.1016/S0165-232X(02)00074-5](https://doi.org/10.1016/S0165-232X(02)00074-5)
  - Core SNOWPACK model documentation

- **Lehning, M., Bartelt, P., Brown, R.L., & Fierz, C. (2002a)**. "A physical SNOWPACK model for the Swiss avalanche warning Part II: Snow microstructure." *Cold Regions Science and Technology*, 35(3), 147-167.
  - Microstructural modeling and constitutive laws

- **Lehning, M., Bartelt, P., Brown, R.L., & Fierz, C. (2002b)**. "A physical SNOWPACK model for the Swiss avalanche warning Part III: Meteorological forcing, thin layer formation and evaluation." *Cold Regions Science and Technology*, 35(3), 169-184.
  - Operational implementation and validation

- **Wever, N., Schmid, L., Heilig, A., Eisen, O., Fierz, C., & Lehning, M. (2015)**. "Verification of the multi-layer SNOWPACK model with different water transport schemes." *The Cryosphere*, 9, 2271-2293. [https://doi.org/10.5194/tc-9-2271-2015](https://doi.org/10.5194/tc-9-2271-2015)
  - Advanced water transport modeling in SNOWPACK

#### Water Flow & Capillary Barriers

- **Hirashima, H., Yamaguchi, S., Sato, A., & Lehning, M. (2010)**. "Numerical modeling of liquid water movement through layered snow based on new measurements of the water retention curve." *Cold Regions Science and Technology*, 64(2), 94-103. [https://doi.org/10.1016/j.coldregions.2010.09.003](https://doi.org/10.1016/j.coldregions.2010.09.003)
  - Improved modeling of water retention and capillary barriers

- **Waldner, P.A., Schneebeli, M., Schultze-Zimmermann, U., & Flühler, H. (2004)**. "Effect of snow structure on water flow and solute transport." *Hydrological Processes*, 18(7), 1271-1290.
  - Documented capillary barrier effects in layered snow

- **Avanzi, F., Hirashima, H., Yamaguchi, S., Katsushima, T., & De Michele, C. (2016)**. "Observations of capillary barriers and preferential flow in layered snow during cold laboratory experiments." *The Cryosphere*, 10, 2013-2026. [https://doi.org/10.5194/tc-10-2013-2016](https://doi.org/10.5194/tc-10-2013-2016)
  - Laboratory observations of capillary barrier formation

#### Machine Learning in Avalanche Forecasting

- **Schirmer, M., Lehning, M., & Schweizer, J. (2009)**. "Statistical forecasting of regional avalanche danger using simulated snow-cover data." *Journal of Glaciology*, 55(193), 761-768.
  - Early ML applications to avalanche forecasting

- **Dreier, L., Mitterer, C., Feick, S., Harvey, S., & Schweizer, J. (2016)**. "Relating meteorological parameters to glide-snow avalanche activity." *Cold Regions Science and Technology*, 128, 57-68.
  - Classification trees for avalanche prediction

- **Gavaldà, J., Moner, I., & Bacardit, M. (2013)**. "Integrating advanced data analysis and machine learning into automatic avalanche detection systems." *Proceedings ISSW 2013*, 452-457.
  - Machine learning for automated avalanche detection

### Operational Avalanche Resources

- **Colorado Avalanche Information Center (CAIC)**. Wet Slab Problem Type. [https://avalanche.state.co.us/wet-slab](https://avalanche.state.co.us/wet-slab)
- **American Avalanche Association**. Avalanche Encyclopedia - Wet Slab. [https://avalanche.org/avalanche-encyclopedia/](https://avalanche.org/avalanche-encyclopedia/)
- **SLF/WSL Institut für Schnee- und Lawinenforschung**. Avalanche Types. [https://www.slf.ch/en/avalanches/](https://www.slf.ch/en/avalanches/)
- **Varsom (Norwegian Avalanche Warning Service)**. Wet Snow Avalanche Problems. [https://www.varsom.no/en/avalanches/](https://www.varsom.no/en/avalanches/)

### Software & Models

- **SNOWPACK Model**. Official documentation and source code. [https://models.slf.ch/p/snowpack/](https://models.slf.ch/p/snowpack/)
- **MeteoIO Library**. Meteorological data processing for SNOWPACK. [https://models.slf.ch/p/meteoio/](https://models.slf.ch/p/meteoio/)

## Citation

If you use this system in research or operations, please cite:

```
Simenhois, R. (2025). Wetting Front Tracker: A Machine Learning System for 
Wet Slab Avalanche Forecasting. Version 2.0.0. 
[Software/Report - details to be added]
```

Additionally, please cite relevant scientific works listed in the References section above that inform the methodology.

## License

[License information to be added]

## Contact

- **Author**: Ron Simenhois
- **Email**: [email to be added]
- **Issues**: [GitHub issues URL]

## Acknowledgments

- SNOWPACK model development team
- Avalanche research community
- Field validation partners

## Version History

**Current Version**: 2.0.0

Major releases:
- **2.0.0**: ML-based LOC detection, mean LWC criterion
- **1.5.0**: Multi-candidate LOC support, hybrid detection
- **1.0.0**: Initial release with rule-based detection

See CHANGELOG.md for detailed version history.
