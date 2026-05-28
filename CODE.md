# Code Description — Wetting Front Tracker

This document describes how the code is organized, how it runs, and how data flows through the system from raw SNOWPACK files to avalanche risk outputs.

---

## Table of Contents

1. [System Purpose](#system-purpose)
2. [Entry Points](#entry-points)
3. [File Descriptions](#file-descriptions)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [ML Sub-System](#ml-sub-system)
6. [Key Data Structures](#key-data-structures)

---

## System Purpose

The Wetting Front Tracker ingests SNOWPACK `.pro` model output files (one per simulation location), detects when liquid water infiltration is approaching a Layer of Concern (LOC), and outputs per-location risk estimates with a 0–72 hour forecast window. Risk is expressed as a color-coded time-to-LOC and rendered as PNG plots, interactive Plotly HTML, and a Folium map.

---

## Entry Points

### Normal analysis (`main.py`)

```bash
python -m wetting_front_tracker.main \
    --date "2025-04-06 18:00" \
    --loc-mode hybrid
```

The `main()` function is the top-level orchestrator. It:
1. Parses CLI arguments (`parse_args`)
2. Resolves the central date to the nearest synoptic time (00, 06, 12, or 18 UTC)
3. Builds the LOC detector(s) based on `--loc-mode`
4. Loads pre-linked geodata or regenerates it via `prepare_geodata.py`
5. Dispatches one task per polygon to a `multiprocessing.Pool`
6. Collects results into a GeoDataFrame and calls `create_folium_map`

### ML data collection

```bash
python -m wetting_front_tracker.main --collect-ml-data
```

Redirects to `run_ml_data_collection()`, which calls `ml_data_collection/collect_ml_data.py`.

### ML model training

```bash
python -m wetting_front_tracker.main --train-ml-model \
    --ml-training-data data/ml_training/dataset.csv \
    --promote-model
```

Redirects to `run_ml_training()`, which calls `ml_training/train_fit_pipeline.py`. The `--promote-model` flag copies the best model to `assets/models/production/`.

---

## File Descriptions

### Core package — `src/wetting_front_tracker/`

#### `main.py`
The orchestrator. Owns the CLI (`parse_args`), the multiprocessing dispatch loop, and the two training workflow helpers. The key per-location function is `process_single_profile`, which runs both detectors (rule-based and ML) in a single pass and returns a result dict.

Key classes:
- `LocResultStandardizer` — wraps any LOC detection callable to normalize its return value to `list[tuple[height, probability]]`. Defined at module level so it is picklable by `multiprocessing`.
- `MLDetectorCallable` — lazy-loading wrapper around `MLLocDetector`. Validates the model path at construction time (raises `FileNotFoundError` on bad config) but loads the model only on first call, avoiding duplicate loads across worker processes.

Key functions:
- `process_single_profile` — full single-location pipeline: load profile → compute timeseries metrics → run rule and ML detectors → determine worst-case `time_to_loc` across all LOC candidates → call plotting functions → return result dict.
- `get_loc_detection_function` — factory that returns the appropriate callable for `rule_based`, `ml_only`, or `hybrid` mode.
- `_get_closest_synoptic_time` — rounds a datetime to the nearest 00/06/12/18 UTC.
- `worker_wrapper` — top-level function passed to `pool.map`; unpacks the task tuple and calls `process_single_profile`.

---

#### `snowpack_reader.py`
Reads SNOWPACK `.pro` files and exposes a clean xarray Dataset. Internally delegates all file I/O to `xsnow.read()`, then adapts the resulting 5-D dataset to the 2-D `(timestamp, layer_index)` structure expected by the rest of the pipeline.

Post-read, two derived variables are computed and appended to the dataset:
- `depth` — distance from snow surface to each layer
- `rc_flat` — a load-corrected grain radius proxy used as a structural strength indicator; GPU-accelerated via CuPy when available

Key class: `SnowpackProfile`
- `__init__` / `_read_profile` — calls `xsnow.read`, squeezes singleton dims (`location`, `slope`, `realization`), renames `time→timestamp` and `layer→layer_index`
- `get_full_timeseries_summary(parameters_to_calculate, start_date, end_date)` — iterates over each timestamp, extracts that day's layer DataFrame, and calls each supplied function against it, returning a `(timestamp, metric)` DataFrame
- `metadata` dict — `stationName`, `latitude`, `longitude`, `altitude`

---

#### `wet_front_tracker.py`
Pure analysis functions. Each function takes a single-timestep layer DataFrame (one row per snow layer, ordered bottom-to-top) and returns a scalar or tuple.

Key functions:
- `wet_front_lwc(df)` — finds the deepest layer with LWC ≥ 4%; returns its height. This is the wetting front position.
- `get_highest_wet_point(df)` — returns the height of the uppermost wet layer.
- `get_total_snow_depth(df)` — total snowpack height (HS).
- `mean_lwc_above_reference(df, ref_height)` — mean LWC in the snowpack above a reference height; used to detect dangerous saturation (≥ 3% triggers orange risk).
- `find_wet_slab_loc(df)` — identifies the LOC as a capillary barrier: a fine-over-coarse grain-size transition (signed grain size difference < −0.5 mm) above a FC/DH layer. Returns `(signed_gs_diff, height)`.
- `find_wet_slab_loc_bottom_half(df)` — same logic but restricted to the bottom half of the snowpack (reduces false positives from surface crusts).
- `find_time_to_loc(summary_df, reference_date)` — given a timeseries summary, extrapolates the wetting front trajectory to estimate when it will reach the LOC height. Returns hours (negative = already passed, positive = future).

Physical thresholds (constants in this module):
| Constant | Value | Meaning |
|---|---|---|
| `LWC_THRESHOLD` | 0.04 (4%) | Wetting front presence |
| `LWC_THRESHOLD_WET_LAYER` | 0.03 (3%) | Mean saturation danger |
| `MIN_GS_DIFFERENCE` | 0.5 mm | Capillary barrier grain size contrast |
| `FC_DH_MIN/MAX_CODE` | 400–600 | SNOWPACK grain type codes for facets/depth hoar |

---

#### `ml_loc_detector.py`
ML-based LOC detection. Loads a trained model (XGBoost or LightGBM, saved with `joblib`) and predicts which layer interfaces are likely to stall the wetting front.

Key class: `MLLocDetector`
- `_load_model` — loads `model.joblib`, optional `scaler.joblib`, `imputer.joblib`, and `feature_names.json` from the model directory
- `find_ml_loc(df, top_n)` — extracts per-layer features from the profile DataFrame, runs the model, filters by probability threshold, and returns the top-N candidates as `list[tuple[height, probability]]`

`create_hybrid_loc_detector(model_path, ...)` — returns a callable that tries the ML detector first and falls back to `find_wet_slab_loc` if the ML output is empty or the model fails.

---

#### `prepare_geodata.py`
Geospatial preprocessing. Runs once (or when `--regenerate-data` is passed) and produces two persistent files: `aspect_polygons.gpkg` and `linked_polygons.gpkg`.

Key functions:
- `prepare_aspect_polygons(input_polygons, output_path, force)` — splits input avalanche terrain polygons by slope aspect (N/NE/E/SE/S/SW/W/NW) using DEM-derived aspect rasters; downloads DEM tiles from OpenTopography if needed
- `link_polygons_to_pro_files(aspect_polygons, locations_csv, output_path)` — spatially joins each aspect polygon to the nearest SNOWPACK simulation location using a k-d tree, then writes the linked GeoPackage
- `generate_pro_file_manifest(base_path, manifest_path)` — walks the input directory and writes a `{filename: full_path}` JSON used at runtime to resolve filenames to absolute paths

---

#### `plotting.py`
All visualization. Outputs are written to `assets_path` per location and a single summary map.

Key functions:
- `plot_summary_matplotlib(summary_df, file_stem, metadata, lwc_data, central_date, assets_path, ml_loc_df)` — 3-panel static PNG: (1) snow height + wetting front depth over time, (2) LWC heatmap, (3) LOC height comparison (rule vs ML when both available)
- `plot_summary_plotly(...)` — equivalent interactive HTML with hover tooltips
- `create_folium_map(gdf, output_path, central_date, assets_path)` — Folium map with one polygon per location, colored by `time_to_loc` risk level, popup showing the PNG thumbnail

Risk color mapping (also defined in `param_config.py`):
| Color | Condition |
|---|---|
| Dark red | Front reaches LOC within 24 h |
| Orange | 24–48 h, or mean LWC above LOC ≥ 3% |
| Yellow | 48–72 h |
| Red | Front already reached LOC recently |
| Light/dark blue | Waning / old event |
| Gray | No data or no risk |

---

#### `param_config.py`
All configuration constants and path resolution, loaded once at import time.

Key dataclasses:
- `PathConfig` — all file paths derived from project root; auto-detected from the location of `param_config.py` itself
- `DataSourceConfig` — `is_remote` flag, `remote_url` for downloading `.pro` files on-demand
- `MLModelConfig` — `enabled`, `model_path`, `probability_threshold`, `use_ml_primary`
- `AppConfig` — aggregates all of the above

Module-level singletons `config`, `ML_CONFIG`, and `LOC_DETECTION_MODE` are imported by every other module.

---

#### `diagnostic_wrapper.py`
Optional debugging layer. When `--enable-diagnostics` is passed, it monkey-patches the core analysis functions to log call counts, timing, and output statistics per function. Not loaded in normal operation.

---

#### `util.py`
Shared utility functions (date parsing, string normalization, etc.) used across modules.

---

#### `generate_locations_csv.py`
One-off script to generate the `snowpack_locations.csv` from a set of `.pro` files. Not part of the normal runtime; run manually when the station network changes.

---

### Sub-package — `ml_data_collection/`

Collects labeled training examples from historical `.pro` files by detecting wetting front stall events and extracting per-layer features.

#### `collect_ml_data.py`
Top-level script for the data collection pipeline. Iterates over `.pro` files, calls `StallDetector` to find stall events, calls `LayerFeatureExtractor` to generate feature rows, and writes the combined dataset to CSV.

#### `stall_detector.py`
Detects when the wetting front stalls at a layer interface. Tracks stalling by SNOWPACK `element_ID` (not height) to maintain identity across time steps even as layers settle.

Key class: `StallDetector`
- Uses `extract_wetting_front_timeseries` to get the wetting front position at each timestamp
- A stall is defined as the front remaining at or near the same layer ID for a configurable number of consecutive hours
- Returns `StallEvent` records with: `layer_id`, `stall_start`, `stall_end`, `ids_above`, `ids_below`

#### `feature_extractor.py`
Extracts the feature vector for each candidate layer interface at the timestamp just before a stall event begins (the last dry state). Features include grain type, grain size, density, LWC, rc_flat, layer thickness, and relative position in the snowpack.

Key class: `LayerFeatureExtractor`

---

### Sub-package — `ml_training/`

Trains, evaluates, and saves LOC detection models from the dataset produced by `ml_data_collection`.

#### `model_trainer.py`
Core training infrastructure.

Key classes:
- `FeatureSelector` — removes low-variance and highly correlated features before training
- `ModelTrainer` — wraps scikit-learn/XGBoost/LightGBM models with a unified `fit` / `predict_proba` / `save_model` / `load_model` interface; optionally tunes hyperparameters with grid search
- `ModelAnalyzer` — post-training analysis: SHAP values, permutation importance, calibration

#### `train_fit_pipeline.py`
Orchestrates the full training run: loads dataset → feature selection → train/val split → trains each configured model → compares ROC-AUC → saves the best model

#### `train_stall_predictor.py`
CLI entry point for the training workflow (equivalent to `main.py --train-ml-model`).

#### `predict_stall.py`
Standalone prediction script for debugging; loads a saved model and runs it on a single `.pro` file.

---

## End-to-End Data Flow

```
SNOWPACK .pro files          Terrain polygons (.gpkg)    DEM (.tif)
        │                           │                        │
        ▼                           ▼                        │
snowpack_reader.py          prepare_geodata.py ◄────────────┘
SnowpackProfile                     │
(timestamp × layer_index)           │ aspect_polygons.gpkg
        │                           │ linked_polygons.gpkg
        │                           │
        └──────────┬────────────────┘
                   │
                   ▼
           main.py — process_single_profile()
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
  wet_front_tracker.py   ml_loc_detector.py
  (rule-based LOC)       (ML LOC, optional)
         │                    │
         └─────────┬──────────┘
                   │
           find_time_to_loc()
           → time_to_loc (hours)
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
  plotting.py           result dict
  PNG + HTML            (per polygon)
                   │
                   ▼
         GeoDataFrame (all polygons)
                   │
                   ▼
         create_folium_map()
         → summary_map.html
```

### Step-by-step walkthrough

1. **Geodata preparation** (once, or `--regenerate-data`): `prepare_geodata.py` downloads DEM tiles, computes slope aspect, splits terrain polygons by aspect, and spatially links each polygon to its nearest SNOWPACK station. Results are cached as GeoPackage files.

2. **Date resolution**: The user-supplied date is snapped to the nearest synoptic hour. The analysis window is `[central_date − 7 days, central_date + 72 hours]`.

3. **Task construction**: Each row in `linked_polygons.gpkg` becomes one task tuple `(pro_file, aspect, start, end, central_date, assets_path, loc_detector_rule, loc_detector_ml)`.

4. **Parallel dispatch**: Tasks are distributed across `cpu_count / 4` worker processes via `multiprocessing.Pool.map`.

5. **Per-location analysis** (`process_single_profile`):
   a. `SnowpackProfile` reads the `.pro` file via `xsnow`, computes `depth` and `rc_flat`.
   b. `get_full_timeseries_summary` iterates over each timestamp and applies: `get_total_snow_depth`, `wet_front_lwc`, `get_highest_wet_point`, `loc_detector_rule`, and (if available) `loc_detector_ml`.
   c. The raw timeseries summary is unpacked: multi-candidate LOC heights are expanded into separate columns (`weak_layer_height_0`, `weak_layer_height_1`, …).
   d. LOC height is persisted forward in time (`_persist_loc_height`) so a layer remains flagged even after it melts out.
   e. `find_time_to_loc` extrapolates the wetting front trajectory to each LOC candidate. `_get_worst_case_time` returns the soonest (most dangerous) result across all candidates.
   f. Plotting functions write per-location PNG and HTML files.

6. **Risk aggregation**: All per-location result dicts are assembled into a GeoDataFrame. `create_folium_map` colors each polygon by its `time_to_loc` and writes `summary_map.html`.

---

## ML Sub-System

The ML path runs in parallel with the rule-based detector inside `process_single_profile` — both detectors consume the same timeseries summary in a single pass. The ML result is used for plot comparison and can be promoted to primary detection via `--loc-mode ml_only`.

**Training pipeline** (run separately before deploying ML detection):

```
Historical .pro files
        │
        ▼
stall_detector.py       ← finds when/where wetting front stalled
        │
feature_extractor.py    ← extracts per-layer features at pre-stall state
        │
collect_ml_data.py      ← writes dataset.csv  (label: stalled=1/0)
        │
model_trainer.py        ← trains XGBoost / LightGBM / RF, tunes, evaluates
        │
train_fit_pipeline.py   ← selects best model, saves to timestamped directory
        │
main.py --promote-model ← copies to assets/models/production/
```

The saved model directory contains: `model.joblib`, `scaler.joblib`, `imputer.joblib`, `feature_names.json`, and evaluation plots (SHAP summary, permutation importance, ROC curve).

---

## Key Data Structures

### `SnowpackProfile.data` (xarray Dataset)
Dimensions: `(timestamp, layer_index)`

Key variables: `height`, `depth`, `lwc`, `grain_type`, `grain_size`, `grain_size_diff`, `density`, `shear_strength`, `rc_flat`, `element_id`

### Per-timestep layer DataFrame (passed to analysis functions)
One row per snow layer, ordered bottom-to-top. Columns are the xarray variables flattened for that timestamp. Passed to every function in `wet_front_tracker.py` and `ml_loc_detector.py`.

### Timeseries summary DataFrame (output of `get_full_timeseries_summary`)
Index: `timestamp`. Columns: one per metric returned by each supplied function (`hs`, `wet_front_lwc`, `highest_wet_point`, `weak_layer_rule_height_0`, `weak_layer_rule_prob_0`, …).

### Result dict (output of `process_single_profile`)
```python
{
    "station_name": str,
    "file_stem": str,
    "time_to_loc": float,          # hours (rule-based, worst-case)
    "time_to_loc_ml": float | None, # hours (ML, worst-case)
    "mean_lwc_above_loc": float,
    "highest_wet_point": float,
    "hs": float,
}
```
This dict is merged with the polygon geometry to form the final GeoDataFrame row.
