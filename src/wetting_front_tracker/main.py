"""
main.py
=======

This script serves as the main entry point and orchestrator for the Wetting 
Front Tracker application.
"""
import argparse
import logging
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timedelta, timezone

import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import json

try:
    from .param_config import config, MLModelConfig, ML_CONFIG, LOC_DETECTION_MODE
except ImportError:
    from param_config import config, MLModelConfig, ML_CONFIG, LOC_DETECTION_MODE
try:
    from .prepare_geodata import (link_polygons_to_pro_files,
                                prepare_aspect_polygons)
except ImportError:
    from prepare_geodata import (link_polygons_to_pro_files,
                                prepare_aspect_polygons)
try:
    from .snowpack_reader import SnowpackProfile
except ImportError:
    from snowpack_reader import SnowpackProfile
try:
    from .wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                    get_total_snow_depth, lwc_above_weak,
                                    wet_front_lwc, find_wet_slab_loc_bottom_half)
except ImportError:
    from wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                    get_total_snow_depth, lwc_above_weak,
                                    wet_front_lwc, find_wet_slab_loc_bottom_half)
try:
    from .plotting import (plot_summary_matplotlib,
                           plot_summary_plotly,
                           create_folium_map)
except ImportError:
    from plotting import (plot_summary_matplotlib,
                           plot_summary_plotly,
                           create_folium_map)
try:
    from .ml_loc_detector import (
        MLLocDetector, 
        create_hybrid_loc_detector
    )
except ImportError:
    from ml_loc_detector import (
        MLLocDetector, 
        create_hybrid_loc_detector
    )   

try:
    from diagnostic_wrapper import enable_diagnostics
    enable_diagnostics()
except ImportError:
        try:
            from .diagnostic_wrapper import enable_diagnostics
            enable_diagnostics()
        except ImportError:
            pass
    
logger = logging.getLogger(__name__)

def get_loc_detection_function(mode: str, ml_config: MLModelConfig, top_n: int = 5):
    """Select LOC detection function."""
    try:    
        from .wet_front_tracker import find_wet_slab_loc
    except ImportError:
        from wet_front_tracker import find_wet_slab_loc
    
    # Helper to wrap single-return functions into list-return
    def list_wrapper(func):
        def wrapper(df):
            try:
                res = func(df)
            except Exception:
                return []
            
            if res is None: return []
            
            if isinstance(res, dict):
                # Handle Height directly
                if 'loc_height' in res: 
                    return [(res['loc_height'], 1.0)]
                # Handle Depth -> Height conversion
                if 'loc_depth' in res and 'height' in df.columns:
                    hs = df['height'].max()
                    if pd.notna(hs):
                        return [(hs - res['loc_depth'], 1.0)]
            
            if isinstance(res, tuple): return [res]
            
            # Fallback for scalar return
            try:
                return [(float(res), 1.0)]
            except:
                return []
        return wrapper

    if mode == "rule_based":
        logger.info("Using rule-based LOC detection")
        return list_wrapper(find_wet_slab_loc)
    
    elif mode == "ml_only":
        if not ml_config.enabled or ml_config.model_path is None:
            return list_wrapper(find_wet_slab_loc)
        
        logger.info(f"Using ML-only LOC detection")
        detector = MLLocDetector(ml_config.model_path, ml_config.probability_threshold)
        # Return a lambda that calls find_ml_loc with top_n
        return lambda df: detector.find_ml_loc(df, top_n=top_n)
    
    elif mode == "hybrid":
        if not ml_config.enabled or ml_config.model_path is None:
            return list_wrapper(find_wet_slab_loc)
        
        logger.info(f"Using hybrid LOC detection")
        return create_hybrid_loc_detector(
            model_path=ml_config.model_path,
            use_ml_primary=ml_config.use_ml_primary,
            ml_threshold=ml_config.probability_threshold,
            rule_based_fallback=find_wet_slab_loc,
            top_n=top_n
        )
    else:
        return list_wrapper(find_wet_slab_loc)
    

def generate_pro_file_manifest(base_path: Path, manifest_path: Path):
    """Recursively scans a directory for .pro files and saves their paths."""
    logging.info(f"Scanning for .pro files under {base_path}...")
    pro_files = list(base_path.rglob('*.pro'))
    manifest = {file.name: str(file.resolve()) for file in pro_files}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    logging.info(f"Pro file manifest with {len(manifest)} entries saved to {manifest_path}")
    
    
def ensure_pro_file_is_local(file_name: str, local_input_path: Path, remote_base_url: str, central_date: datetime):
    """Checks if a .pro file exists locally and is fresh."""
    local_file_path = local_input_path / file_name
    
    if local_file_path.exists():
        mod_time_ts = os.path.getmtime(local_file_path)
        mod_time_dt = datetime.fromtimestamp(mod_time_ts, tz=timezone.utc)
        central_date_utc = central_date.replace(tzinfo=timezone.utc)
        if (central_date_utc - mod_time_dt) < timedelta(hours=12):
            logging.debug(f"'{file_name}' is fresh. Skipping download.")
            return

    logging.info(f"Downloading '{file_name}'...")
    remote_file_url = f"{remote_base_url.rstrip('/')}/{file_name}"
    logging.warning(f"Placeholder: Pretending to download from {remote_file_url} to {local_file_path}")
    
    
def _initialize_and_validate_profile(pro_file_path: Path, aspect: str) -> tuple[SnowpackProfile | None, str | None]:
    """Initializes a SnowpackProfile object and validates its data."""
    profile = SnowpackProfile(pro_file_path)
    file_stem = f"{pro_file_path.stem}_{aspect}"
    profile.metadata['aspect'] = aspect

    if profile.data is None or 'timestamp' not in profile.data.coords:
        logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
        return None, None
    return profile, file_stem

def _unpack_and_prepare_summary(summary_df: pd.DataFrame, max_locs: int = 3) -> pd.DataFrame:
    """
    Unpacks summary columns, handling multiple LOC candidates.
    Creates columns: weak_layer_height, weak_layer_prob (for best),
    plus weak_layer_height_1, weak_layer_prob_1, etc.
    """
# 1. Expand Multiple LOCs
    if 'weak_layer' in summary_df.columns:
        
        def expand_locs(row):
            locs = row['weak_layer']
            # Handle list of tuples vs single value/NaN
            if not isinstance(locs, list): 
                return pd.Series([np.nan] * (max_locs * 2))
            
            vals = []
            for i in range(max_locs):
                if i < len(locs):
                    # Append Height, Prob
                    vals.extend([locs[i][0], locs[i][1]])
                else:
                    vals.extend([np.nan, np.nan])
            return pd.Series(vals)

        # Create column names
        cols = []
        for i in range(max_locs):
            cols.extend([f'weak_layer_height_{i}', f'weak_layer_prob_{i}'])
            
        expanded = summary_df.apply(expand_locs, axis=1)
        expanded.columns = cols
        
        # Only concat expanded columns, do not pre-initialize them
        summary_df = pd.concat([summary_df, expanded], axis=1)
        
        # Alias index 0 to the standard name for backward compatibility
        # This assumes expanded columns are now present and unique
        if f'weak_layer_height_0' in summary_df.columns:
            summary_df['weak_layer_height'] = summary_df['weak_layer_height_0']
            summary_df['weak_layer_prob'] = summary_df['weak_layer_prob_0']

    # 2. Standard Unpacking
    rename_map = {"wet_front_lwc_value": "wet_front_lwc_val"} # Just to clean up
    summary_df.rename(columns=rename_map, inplace=True)

    numeric_cols = ['wet_front_lwc_height', 'hs']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
            
    return summary_df

def _persist_loc_height(summary_df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """Persists the LOC height through the melt event."""
    if 'weak_layer_height' not in summary_df.columns or 'wet_front_lwc_height' not in summary_df.columns:
        return summary_df

    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]

    relevant_start_times = all_start_times[all_start_times <= reference_date]

    if relevant_start_times.empty:
        return summary_df
    
    trigger_time = relevant_start_times[-1]

    lookback_window_end = trigger_time
    lookback_window_start = lookback_window_end - timedelta(days=2)
    pre_melt_df = summary_df.loc[lookback_window_start:lookback_window_end]
    
    valid_pre_melt_locs = pre_melt_df['weak_layer_height'].dropna()
    initial_lock_height = np.nan if valid_pre_melt_locs.empty else valid_pre_melt_locs.iloc[-1]

    if pd.isna(initial_lock_height):
        return summary_df
        
    persisted_loc = summary_df['weak_layer_height'].copy()
    wet_season_mask: pd.Series = summary_df.index >= trigger_time
    
    wet_loc_series: pd.Series = summary_df.loc[wet_season_mask, 'weak_layer_height']
    anchored_series = pd.concat([pd.Series([initial_lock_height]), wet_loc_series.reset_index(drop=True)])
    
    running_max_loc = anchored_series.cummax().iloc[1:].values
    
    persisted_loc.loc[wet_season_mask] = running_max_loc
    persisted_loc_filled = persisted_loc.ffill()
    persisted_loc_filled[summary_df['hs'] < persisted_loc_filled] = np.nan
    summary_df['weak_layer_height'] = persisted_loc_filled
    
    return summary_df

def _get_worst_case_time(times: list[float]) -> float | None:
    """
    Selects the most critical time_to_loc from a list of candidates.
    Priority order (Highest Risk to Lowest):
    1. Imminent (0 to 24h)
    2. Recent (-24 to 0h)
    3. Near Future (24 to 48h)
    4. Moderate Future (48 to 72h)
    5. Past Near (-48 to -24h)
    6. Past Far (-72 to -48h)
    7. Everything else (Safe/Unknown)
    """
    valid_times = [t for t in times if pd.notna(t)]
    if not valid_times:
        return None

    # Priority buckets (min_inc, max_exc)
    # Lower index = Higher Priority
    buckets = [
        (0, 24),        # Imminent (Dark Red)
        (-24, 0),       # Recent (Red)
        (24, 48),       # Near (Orange)
        (48, 72),       # Moderate (Yellow)
        (-48, -24),     # Past Near (Light Blue)
        (-72, -48),     # Past Far (Dark Blue)
    ]

    for min_t, max_t in buckets:
        for t in valid_times:
            if min_t <= t < max_t:
                return t
                
    # If no times fell into risk buckets, return the minimum positive time 
    # (soonest future), or max negative (most recent past)
    positive_times = [t for t in valid_times if t >= 0]
    if positive_times:
        return min(positive_times)
    
    return max(valid_times)


def process_single_profile(pro_file_path: Path, 
                           aspect: str, 
                           start_date_arg: str | None = None, 
                           end_date_arg: str | None = None, 
                           central_date_arg: datetime | None = None,
                           assets_path: Path | None = None,
                           loc_detector: Callable | None = None) -> dict[str, Any] | None:
    """Handles the full analysis workflow for a single polygon."""
    try:
        if loc_detector is None:
            loc_detector = find_wet_slab_loc_bottom_half
            
        profile, file_stem = _initialize_and_validate_profile(pro_file_path, aspect)
        if not profile or not profile.data or not file_stem:
            return None
        
        if central_date_arg:
            min_date_in_data = central_date_arg - timedelta(days=7)
            max_date_in_data = central_date_arg + timedelta(hours=72)
        else:
            min_date_in_data = pd.to_datetime(profile.data.timestamp.values[0])
            max_date_in_data = pd.to_datetime(profile.data.timestamp.values[-1])

        # Unwrap list-based LOC results for lwc_above_weak
        # lwc_above_weak expects a function returning (height, prob) or None
        def loc_adapter(df):
            res = loc_detector(df)
            return res[0] if res and len(res) > 0 else None

        raw_summary = profile.get_full_timeseries_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, 
                "weak_layer": loc_detector, # Returns list of tuples
                "wet_front_lwc": wet_front_lwc,
                "highest_wet_point": get_highest_wet_point,
                "lwc_above_weak": lambda df: lwc_above_weak(df, loc_adapter) # Uses adapter
            },
            start_date=str(min_date_in_data),
            end_date=str(max_date_in_data),
        ).copy()
        
        if raw_summary.empty:
            return None

        prepared_summary = _unpack_and_prepare_summary(raw_summary)
        referance_date = central_date_arg or datetime.now()
        
        # Persist LOC height (Applied primarily to the 'weak_layer_height' alias)
        # NOTE: Ideally we would persist ALL candidates, but for now we persist the primary
        # to ensure the main logic holds. The Multi-LOC logic below uses raw unpacked values.
        summary_full = _persist_loc_height(prepared_summary, referance_date)

        if start_date_arg is None or end_date_arg is None:
            logging.error("Analysis window start/end dates are missing.")
            return None
        
        # --- Plotting Logic ---
        start_dt, end_dt = pd.to_datetime(start_date_arg), pd.to_datetime(end_date_arg)
        is_in_window = (summary_full.index >= start_dt) & (summary_full.index <= end_dt)
        summary_for_plot = summary_full[is_in_window]

        full_season_plot_data = profile.data[['lwc', 'height']]
        is_in_lwc_window = (full_season_plot_data.timestamp >= start_dt) & (full_season_plot_data.timestamp <= end_dt)
        lwc_data_for_plot = full_season_plot_data.sel(timestamp=is_in_lwc_window)

        station_metadata = profile.metadata
        del profile

        if not summary_for_plot.empty:
            plot_summary_matplotlib(summary_for_plot, file_stem, station_metadata, lwc_data_for_plot, central_date_arg, assets_path)
            plot_summary_plotly(summary_full, file_stem, station_metadata, central_date_arg, assets_path)
        else:
            logging.warning(f"No snowpack data found for {file_stem} in window.")

        # --- UPDATED: Multi-Candidate Time Calculation ---
        candidate_times = []
        
        # 1. Check Primary (Legacy column)
        t0 = find_time_to_loc(summary_full, reference_date=referance_date)
        candidate_times.append(t0)
        
        # 2. Check Alternatives (weak_layer_height_1, _2, etc.)
        # We need to iterate columns because find_time_to_loc hardcodes 'weak_layer_height'
        loc_cols = [c for c in summary_full.columns if c.startswith('weak_layer_height_') and c != 'weak_layer_height_0']
        
        for col in loc_cols:
            # Create a temporary view where this candidate is the "weak_layer_height"
            temp_df = summary_full.copy()
            temp_df['weak_layer_height'] = temp_df[col]
            # Apply persistence to this candidate too (optional but recommended)
            temp_df = _persist_loc_height(temp_df, referance_date)
            
            t_alt = find_time_to_loc(temp_df, reference_date=referance_date)
            candidate_times.append(t_alt)
            
        # 3. Pick Worst Case
        worst_case_time = _get_worst_case_time(candidate_times)

        return {
            "station_name": station_metadata.get('stationName', file_stem),
            "file_stem": file_stem,
            "time_to_loc": worst_case_time,  # Use the worst-case time for the map
            "central_date_str": central_date_arg.strftime('%Y-%m-%d %H:%M') if central_date_arg else None
        }

    except Exception as e:
        logging.error(f"Error processing {pro_file_path.name} for aspect {aspect}: {e}", exc_info=True)
        return None
    
    
def worker_wrapper(task_tuple: tuple) -> dict[str, Any] | None:
    """Wrapper function for multiprocessing."""
    return process_single_profile(*task_tuple)

def _get_closest_synoptic_time(reference_time: datetime) -> datetime:
    """Finds the closest standard synoptic time (00, 06, 12, 18 UTC)."""
    base_date = reference_time.date()
    candidates = [
        datetime.combine(base_date, datetime.min.time()).replace(hour=h)
        for h in [0, 6, 12, 18]
    ]
    candidates.insert(0, candidates[0] - timedelta(hours=6))
    candidates.append(candidates[1] + timedelta(days=1))
    return min(candidates, key=lambda dt: abs(reference_time - dt))


def run_ml_data_collection(input_path: Path, output_dir: Path, central_date: datetime) -> Path | None:
    """Run ML training data collection workflow."""
    try:
        from .ml_data_collection.stall_detector import StallDetector, StallDetectionConfig
        from .ml_data_collection.feature_extractor import LayerFeatureExtractor, FeatureExtractionConfig
    except ImportError:
        logging.error("ML data collection modules not found.")
        return None
    
    logging.info("=" * 80)
    logging.info("ML TRAINING DATA COLLECTION")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stall_config = StallDetectionConfig(
        min_duration_hours=12.0,
        max_duration_hours=240.0,
        min_lwc_threshold=0.04,
        feature_lookback_hours=24.0
    )
    
    feature_config = FeatureExtractionConfig(
        use_dynamic_lookback=True,
        lwc_threshold_pct=1.0,
        max_lookback_hours=72.0,
        fallback_lookback_hours=24.0
    )
    
    pro_files = list(input_path.rglob('*.pro'))
    logging.info(f"Found {len(pro_files)} .pro files")
    
    detector = StallDetector(stall_config)
    extractor = LayerFeatureExtractor(feature_config)
    all_features = []
    
    for pro_file in tqdm(pro_files, desc="Processing files"):
        try:
            from .snowpack_reader import SnowpackProfile
            profile = SnowpackProfile(pro_file)
            stalls = detector.detect_stalls(profile)
            for stall in stalls:
                features = extractor.extract_features_for_stall(profile, stall)
                if features is not None:
                    all_features.append(features)
        except Exception as e:
            logging.warning(f"Error processing {pro_file.name}: {e}")
            continue
    
    if not all_features:
        logging.error("No training data collected!")
        return None
    
    df = pd.DataFrame(all_features)
    output_file = output_dir / f"ml_training_dataset_{central_date.strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    
    logging.info(f"Collected {len(df)} examples. Saved to {output_file}")
    return output_file


def run_ml_training(
    training_data_path: Path,
    output_dir: Path,
    models_to_train: list,
    tune_hyperparameters: bool = True,
    compute_shap: bool = True
) -> Path:
    """Run ML model training workflow."""
    try:
        from .ml_training.model_trainer import ModelTrainer, ModelConfig
        from .ml_training.model_trainer import FeatureImportanceAnalyzer
        from .ml_training.model_trainer import plot_model_comparison, plot_feature_importance, plot_shap_summary
    except ImportError:
        logging.error("ML training modules not found.")
        return None
    
    logging.info("=" * 80)
    logging.info("ML MODEL TRAINING")
    logging.info(f"Data: {training_data_path}")
    
    # Load training data
    df = pd.read_csv(training_data_path)
    
    metadata_cols = [
        'event_id', 'pro_file', 'start_time', 'end_time',
        'stall_layer_id', 'layer_above_id', 'layer_below_id',
        'feature_extraction_time', 'lookback_hours', 'station_name',
        'duration_hours', 'confidence', 'n_data_points', 'is_ongoing',
        'lookback_method', 'above_lwc_at_extraction', 'below_lwc_at_extraction',
    ]
    
    irrelevant_cols = [
        'distance_from_stall_m', 'example_type', 'requested_lookback_hours'
    ]
    
    target_col = 'target' if 'target' in df.columns else 'stalled'
    feature_cols = [c for c in df.columns
                   if c not in metadata_cols and c not in irrelevant_cols and c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    config = ModelConfig(
        models_to_train=models_to_train,
        tune_hyperparameters=tune_hyperparameters,
        tuning_method='random',
        n_iter_random=30,
        cv_folds=5,
        use_time_series_cv=True,
        compute_shap_values=compute_shap,
        scale_features=True,
        remove_low_variance=True,
        remove_correlated=True
    )
    
    trainer = ModelTrainer(config)
    trainer.fit(X, y)
    
    # Feature importance
    if compute_shap:
        analyzer = FeatureImportanceAnalyzer(trainer.best_model_, trainer.X_train, trainer.feature_names_)
        importance_results = analyzer.analyze_all(
            trainer.X_val, trainer.y_val, compute_shap=True, shap_sample_size=min(200, len(trainer.X_val))
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / 'trained_model'
    trainer.save_model(model_dir)
    
    # Save artifacts
    try:
        plot_model_comparison(
            trainer.results_,
            save_path=output_dir / 'model_comparison.png'
        )
        logging.info(f"Saved: {output_dir / 'model_comparison.png'}")
        
        if compute_shap and 'shap' in importance_results:
            plot_feature_importance(
                importance_results,
                top_n=30,
                save_path=output_dir / 'feature_importance.png'
            )
            logging.info(f"Saved: {output_dir / 'feature_importance.png'}")
            
            # UPDATED: Use the matching 'shap_data' from the results
            if 'shap_values' in importance_results and 'shap_data' in importance_results:
                plot_shap_summary(
                    importance_results['shap_values'],
                    importance_results['shap_data'],  # <--- USE THIS instead of trainer.X_val.head()
                    save_path=output_dir / 'shap_summary.png'
                )
                logging.info(f"Saved: {output_dir / 'shap_summary.png'}")
            else:
                logging.warning("SHAP values present but shap_data missing. Cannot plot summary.")

    except Exception as e:
        logging.warning(f"Could not save all plots: {e}")


def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--regenerate-data", action="store_true", help="Force regeneration of all processed data.")
    parser.add_argument("-d", "--date", dest="central_date", default="2025-05-09 12:00", help="Central date and time.")
    parser.add_argument("-s", "--start", dest="start_date", help="Start date.")
    parser.add_argument("-e", "--end", dest="end_date", help="End date.")
    parser.add_argument("-i", "--input-dir", dest="input_dir", type=Path, default=None, help="Override input dir.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=Path, default=None, help="Override output dir.")
    parser.add_argument("-a", "--assets-dir", dest="assets_dir", type=Path, default=None, help="Override plot assets dir.")
    
    # ML Args
    parser.add_argument("--loc-mode", dest="loc_mode", choices=["rule_based", "ml_only", "hybrid"], default=None, help="LOC detection mode.")
    parser.add_argument("--ml-model-path", dest="ml_model_path", type=Path, default=None, help="Override ML model path.")
    parser.add_argument("--ml-models-dir", dest="ml_models_dir", type=Path, default=config.paths.models_path, help="Directory to search for production models.")
    parser.add_argument("--ml-threshold", dest="ml_threshold", type=float, default=None, help="ML probability threshold.")
    
    # Training Args
    parser.add_argument("--collect-ml-data", action="store_true", help="Collect ML training data.")
    parser.add_argument("--train-ml-model", action="store_true", help="Train a new ML model.")
    parser.add_argument("--ml-training-data", dest="ml_training_data", type=Path, default=None, help="Path to ML dataset.")
    parser.add_argument("--ml-training-output", dest="ml_training_output", type=Path, default=None, help="Output dir for trained model.")
    parser.add_argument("--ml-training-models", dest="ml_training_models", nargs='+', default=['xgboost', 'lightgbm', 'random_forest'], help="Models to train.")
    parser.add_argument("--no-ml-tune", action="store_true", help="Skip hyperparameter tuning.")
    parser.add_argument("--no-ml-shap", action="store_true", help="Skip SHAP analysis.")
    parser.add_argument("--promote-model", action="store_true", help="If training succeeds, copy the model to assets/models/production.")
    
    return parser.parse_args()


def main():
    """Main orchestrator for the entire analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="wetting_front_tracker.log",
        filemode="w"
    )    
    
    args = parse_args()
    
    # --- Handle ML Training Workflows ---
    if args.collect_ml_data:
        input_path = args.input_dir or config.paths.input_path
        output_dir = args.ml_training_output or (config.paths.results_path / 'trained_models' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        central_date = datetime.strptime(args.central_date.split()[0], '%Y-%m-%d') if args.central_date else datetime.now()
        
        dataset_path = run_ml_data_collection(input_path, output_dir, central_date)
        if dataset_path:
            logging.info(f"ML data collection complete: {dataset_path}")
        return
    
    if args.train_ml_model:
        if not args.ml_training_data or not args.ml_training_data.exists():
            logging.error("--ml-training-data is required and must exist.")
            return
            
        output_dir = args.ml_training_output or (config.paths.results_path / 'trained_models' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        model_dir = run_ml_training(
            args.ml_training_data,
            output_dir,
            args.ml_training_models,
            tune_hyperparameters=not args.no_ml_tune,
            compute_shap=not args.no_ml_shap
        )
        
        # Promotion Logic
        if model_dir and args.promote_model:
            target_dir = config.paths.models_path / "production"
            logging.info(f"Promoting model to: {target_dir}")
            
            if target_dir.exists():
                logging.warning(f"Overwriting existing production model at {target_dir}")
                shutil.rmtree(target_dir)
            
            shutil.copytree(model_dir, target_dir)
            logging.info("Model promotion complete.")
            
        return

    # --- Normal Analysis Workflow ---
    input_path = args.input_dir or config.paths.input_path
    output_path = args.output_dir or config.paths.results_path
    assets_path = args.assets_dir or config.paths.plot_assets_path
    output_path.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)

    # Date handling
    try:
        initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d %H:%M')
    except ValueError:
        initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d')

    central_date = _get_closest_synoptic_time(initial_ref_time)
    start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')

    # LOC Configuration
    loc_mode = args.loc_mode or LOC_DETECTION_MODE
    ml_config = ML_CONFIG
    
    # If user overrides model path, use it
    if args.ml_model_path:
        ml_config.model_path = args.ml_model_path
        ml_config.enabled = True
    elif not ml_config.model_path and config.paths.models_path.exists():
        # Fallback to checking internal assets if no path found yet
        for item in config.paths.models_path.iterdir():
            if item.is_dir() and (item / "model.joblib").exists():
                ml_config.model_path = item
                ml_config.enabled = True
                break

    if args.ml_threshold is not None:
        ml_config.probability_threshold = args.ml_threshold

    loc_detector = get_loc_detection_function(loc_mode, ml_config)

    # Geodata & Processing
    if args.regenerate_data or not config.paths.linked_polygons.exists():
        generate_pro_file_manifest(input_path, config.paths.pro_file_manifest)
        prepare_aspect_polygons(config.get_input_polygons_path(), config.paths.aspect_polygons, args.regenerate_data)
        link_polygons_to_pro_files(config.paths.aspect_polygons, config.paths.snowpack_locations_csv, config.paths.linked_polygons)

    try:
        linked_gdf = gpd.read_file(config.paths.linked_polygons)
        with open(config.paths.pro_file_manifest, 'r') as f:
            pro_file_manifest = json.load(f)
    except FileNotFoundError:
        logging.error("Geodata not found. Run with --regenerate-data.")
        return

    if config.data_source.is_remote:
        required_files = {Path(p).name for p in linked_gdf['pro_file_path'].unique()}
        for file_name in tqdm(required_files, desc="Updating files"):
            ensure_pro_file_is_local(file_name, input_path, config.data_source.remote_url, central_date)

    tasks = []
    for poly in linked_gdf.itertuples(index=False):
        file_name = Path(str(poly.pro_file_path)).name
        if full_pro_path_str := pro_file_manifest.get(file_name):
            tasks.append((Path(full_pro_path_str), poly.aspect, start_date, end_date, central_date, assets_path, loc_detector))

    if not tasks:
        return

    logging.info(f"Starting analysis on {len(tasks)} polygons...")
    cpu_cores = os.cpu_count()
    worker_count = int(max(1, cpu_cores / 4 )) if cpu_cores else 1 
    with multiprocessing.Pool(processes=worker_count) as pool:
        results = list(tqdm(pool.map(worker_wrapper, tasks, chunksize=1), total=len(tasks)))

    if valid_results := [res for res in results if res is not None]:
        results_df = pd.DataFrame(valid_results)
        final_gdf = gpd.GeoDataFrame(pd.concat([linked_gdf.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1))
        create_folium_map(final_gdf, config.paths.summary_map_html, central_date, assets_path )
    else:
        logging.info("No valid results generated.")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
"""
main.py
=======

This script serves as the main entry point and orchestrator for the Wetting 
Front Tracker application. It manages the end-to-end workflow, from command-line
argument parsing and geospatial data preparation to the parallelized analysis of
snowpack files and the final generation of a summary map.

Workflow Overview:
------------------
1.  **Initialization:** The script begins by parsing command-line arguments, which
    allow the user to specify a central analysis date, force data regeneration,
    or define a custom time window for the analysis. It establishes the primary
    time window (e.g., 7 days before and 72 hours after the central date).

2.  **Geodata Preparation (Conditional):** If the `--regenerate-data` flag is used,
    or if essential processed geodata files are missing, it triggers the
    `prepare_geodata` module. This step downloads DEMs, splits input avalanche
    path polygons by terrain aspect, and links each resulting polygon to its
    most relevant SNOWPACK (.pro) model output file.

3.  **Task Generation:** It reads the `linked_aspect_polygons.geojson` file, which
    contains the geometries and the path to the corresponding .pro file for each
    polygon to be analyzed. It creates a list of tasks, with each task
    containing the necessary information to process one polygon.

4.  **Parallel Snowpack Analysis:** Using Python's `multiprocessing` library, the
    script distributes the analysis tasks across all available CPU cores. For
    each polygon, a worker process:
    a. Reads the linked .pro file into a `SnowpackProfile` object.
    b. Calculates a time series of key snowpack metrics (e.g., weak layer
       height, wetting front depth, total snow depth).
    c. Applies a persistence logic to track the primary weak layer (LOC)
       through melt events.
    d. Calculates the final `time_to_loc` metric: the time (in hours) for the
       wetting front to reach the weak layer relative to the central date.
    e. Generates a static Matplotlib plot and an interactive Plotly plot for
       the analysis time window.

5.  **Aggregation and Final Visualization:** After all worker processes are complete,
    the main script collects the results. It merges the analysis results (like
    `time_to_loc`) back into the GeoDataFrame and calls the `plotting` module to
    create the final `summary_map.html`. This map displays all polygons,
    color-coded by their risk level, with tooltips and links to the detailed plots.

Usage:
------
- To run with default settings:
  `python -m src.wetting_front_tracker.main`
- To specify a central date:
  `python -m src.wetting_front_tracker.main --date YYYY-MM-DD`
- To force regeneration of all geodata:
  `python -m src.wetting_front_tracker.main --regenerate-data`
"""
import argparse
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timedelta, timezone

import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import json

from .param_config import config, MLModelConfig, ML_CONFIG, LOC_DETECTION_MODE

from .plotting import (create_folium_map, plot_summary_matplotlib, 
                       plot_summary_plotly)
from .prepare_geodata import (link_polygons_to_pro_files,
                              prepare_aspect_polygons)
from .snowpack_reader import SnowpackProfile
from .wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                get_total_snow_depth, lwc_above_weak,
                                wet_front_lwc, find_wet_slab_loc_bottom_half)

from .ml_loc_detector import (
    MLLocDetector, 
    create_hybrid_loc_detector
)

# Initialize logger
logger = logging.getLogger(__name__)

def get_loc_detection_function(mode: str, ml_config: MLModelConfig):
    """
    Select LOC detection function based on configuration.
    
    Args:
        mode: Detection mode ("rule_based", "ml_only", "hybrid")
        ml_config: ML model configuration
    
    Returns:
        Callable LOC detection function
    """
    from .wet_front_tracker import find_wet_slab_loc
    
    if mode == "rule_based":
        logger.info("Using rule-based LOC detection")
        return find_wet_slab_loc
    
    elif mode == "ml_only":
        if not ml_config.enabled or ml_config.model_path is None:
            logger.error("ML-only mode selected but ML not configured")
            logger.info("Falling back to rule-based detection")
            return find_wet_slab_loc
        
        logger.info(f"Using ML-only LOC detection from {ml_config.model_path}")
        detector = MLLocDetector(
            ml_config.model_path,
            ml_config.probability_threshold
        )
        return detector.find_ml_loc
    
    elif mode == "hybrid":
        if not ml_config.enabled or ml_config.model_path is None:
            logger.warning("Hybrid mode selected but ML not configured")
            logger.info("Using rule-based detection only")
            return find_wet_slab_loc
        
        logger.info(f"Using hybrid LOC detection (ML + rule-based)")
        return create_hybrid_loc_detector(
            model_path=ml_config.model_path,
            use_ml_primary=ml_config.use_ml_primary,
            ml_threshold=ml_config.probability_threshold,
            rule_based_fallback=find_wet_slab_loc
        )
    
    else:
        logger.error(f"Unknown LOC detection mode: {mode}")
        logger.info("Falling back to rule-based detection")
        return find_wet_slab_loc


def generate_pro_file_manifest(base_path: Path, manifest_path: Path):
    """Recursively scans a directory for .pro files and saves their paths to a manifest file.

    The manifest is a JSON object that maps a simple filename (e.g., "station.pro") 
    to its full, absolute path. This allows for quick lookups without needing to 
    re-scan the entire filesystem on every run.

    Args:
        base_path (Path): The root directory to start the recursive scan from.
        manifest_path (Path): The full path where the output JSON manifest 
                              file will be saved.
    """
    logging.info(f"Scanning for .pro files under {base_path}...")
    # Use rglob to find all files ending with .pro in all subdirectories
    pro_files = list(base_path.rglob('*.pro'))
    
    # Create a dictionary of {filename: /full/path/to/file.pro}
    manifest = {file.name: str(file.resolve()) for file in pro_files}
    
    # Write the dictionary to the specified JSON file
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    logging.info(f"Pro file manifest with {len(manifest)} entries saved to {manifest_path}")
    
    
def ensure_pro_file_is_local(file_name: str, local_input_path: Path, remote_base_url: str, central_date: datetime):
    """
    Checks if a .pro file exists locally and is fresh. If not, downloads it.
    This is a placeholder for your actual download logic (e.g., from S3, HTTP).
    """
    local_file_path = local_input_path / file_name
    
    # 1. Check if file exists and is fresh (less than 12 hours old)
    if local_file_path.exists():
        mod_time_ts = os.path.getmtime(local_file_path)
        mod_time_dt = datetime.fromtimestamp(mod_time_ts, tz=timezone.utc)
        central_date_utc = central_date.replace(tzinfo=timezone.utc)
        if (central_date_utc - mod_time_dt) < timedelta(hours=12):
            logging.debug(f"'{file_name}' is fresh. Skipping download.")
            return  # File is fresh, no need to download

    # 2. If we reach here, the file is either missing or stale, so download it.
    logging.info(f"Downloading '{file_name}'...")
    remote_file_url = f"{remote_base_url.rstrip('/')}/{file_name}"
    
    # --- !!! ADD YOUR DOWNLOAD LOGIC HERE !!! ---
    # Example for S3 using boto3:
    # import boto3
    # s3 = boto3.client('s3')
    # bucket_name = "my-bucket"
    # object_key = f"pro-files/{file_name}"
    # s3.download_file(bucket_name, object_key, str(local_file_path))

    # Example for HTTP using requests:
    # import requests
    # r = requests.get(remote_file_url, stream=True)
    # if r.status_code == 200:
    #     with open(local_file_path, 'wb') as f:
    #         for chunk in r.iter_content(chunk_size=8192):
    #             f.write(chunk)
    # else:
    #     logging.error(f"Failed to download {file_name}. Status: {r.status_code}")
    
    # For now, we'll just log a placeholder message.
    logging.warning(f"Placeholder: Pretending to download from {remote_file_url} to {local_file_path}")
    # You would create a dummy file for testing if needed
    # local_file_path.touch()
    
    
def _initialize_and_validate_profile(pro_file_path: Path, aspect: str) -> tuple[SnowpackProfile | None, str | None]:
    """
    Initializes a SnowpackProfile object and validates its data.

    This helper function loads a .pro file, creates a unique file stem for
    output files based on the file name and aspect, and checks if the profile
    contains valid, timestamped data.

    Args:
        pro_file_path: The path to the .pro input file.
        aspect: The aspect of the polygon being processed (e.g., 'N', 'E').

    Returns:
        A tuple containing the initialized SnowpackProfile object and a unique
        file stem, or (None, None) if the profile data is invalid or missing.
    """
    profile = SnowpackProfile(pro_file_path)
    file_stem = f"{pro_file_path.stem}_{aspect}"
    profile.metadata['aspect'] = aspect

    if profile.data is None or 'timestamp' not in profile.data.coords:
        logging.warning("No valid data or timestamps in '%s'. Skipping.", pro_file_path.name)
        return None, None
    return profile, file_stem

def _unpack_and_prepare_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpacks tuple columns from the summary and ensures correct data types.

    The initial summary from `get_profile_summary` may contain columns where
    each cell is a tuple (e.g., (value, height)). This function unpacks these
    tuples into separate columns and ensures that key columns used in
    calculations are converted to a numeric type.

    Args:
        summary_df: The raw summary DataFrame from the profile analysis.

    Returns:
        A prepared DataFrame with unpacked columns and appropriate numeric
        data types, ready for further analysis.
    """
    rename_map = {
        "weak_layer_value": "weak_layer_gs_diff",
        # "wet_front_lwc_value": "wet_front_lwc_value", # This is redundant
    }
    summary_df.rename(columns=rename_map, inplace=True)

    numeric_cols = ['weak_layer_height', 'wet_front_lwc_height', 'hs']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    
    return summary_df

def _persist_loc_height(summary_df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """
    Identifies the primary weak layer just before the most recent melt event
    and carries its height forward dynamically.

    This function finds the start of the most recent melt event relative to a
    reference date, locks onto the last known weak layer before it, and then
    tracks that layer. If a new weak layer is detected at a higher elevation,
    the lock is updated to that new, higher layer.

    Args:
        summary_df: The prepared summary DataFrame.
        reference_date: The central date for the analysis (e.g., today).

    Returns:
        A DataFrame with `weak_layer_height` adjusted for persistence.
    """
    if 'weak_layer_height' not in summary_df.columns or 'wet_front_lwc_height' not in summary_df.columns:
        return summary_df

    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]

    relevant_start_times = all_start_times[all_start_times <= reference_date]

    if relevant_start_times.empty:
        return summary_df
    
    trigger_time = relevant_start_times[-1]

    lookback_window_end = trigger_time
    lookback_window_start = lookback_window_end - timedelta(days=2)
    pre_melt_df = summary_df.loc[lookback_window_start:lookback_window_end]
    
    # FIX: Check if the filtered DataFrame is empty before accessing iloc[-1]
    valid_pre_melt_locs = pre_melt_df['weak_layer_height'].dropna()
    initial_lock_height = np.nan if valid_pre_melt_locs.empty else valid_pre_melt_locs.iloc[-1]

    if pd.isna(initial_lock_height):
        return summary_df
        
    persisted_loc = summary_df['weak_layer_height'].copy()
    wet_season_mask: pd.Series = summary_df.index >= trigger_time
    
    wet_loc_series: pd.Series = summary_df.loc[wet_season_mask, 'weak_layer_height']
    anchored_series = pd.concat([pd.Series([initial_lock_height]), wet_loc_series.reset_index(drop=True)])
    
    running_max_loc = anchored_series.cummax().iloc[1:].values
    
    persisted_loc.loc[wet_season_mask] = running_max_loc

    persisted_loc_filled = persisted_loc.ffill()

    persisted_loc_filled[summary_df['hs'] < persisted_loc_filled] = np.nan

    summary_df['weak_layer_height'] = persisted_loc_filled
    
    return summary_df

def process_single_profile(pro_file_path: Path, 
                           aspect: str, 
                           start_date_arg: str | None = None, 
                           end_date_arg: str | None = None, 
                           central_date_arg: datetime | None = None,
                           assets_path: Path | None = None,
                           loc_detector: Callable | None = None) -> dict[str, Any] | None:
    """
    Handles the full analysis workflow for a single polygon and its linked .pro file.
    
    This is the core analysis function that is parallelized. It orchestrates
    the reading of a snowpack file, running various analyses on it, applying
    the LOC persistence logic, generating plots, and calculating the final
    `time_to_loc` metric.

    Args:
        pro_file_path: The path to the .pro input file.
        aspect: The aspect ('N', 'E', 'S', 'W', 'Flat') of the polygon being
                processed.
        start_date_arg: The start date for the analysis window, used for
                        the Matplotlib plot's visible range.
        end_date_arg: The end date for the analysis window.
        central_date_arg: The central reference date for the `time_to_loc`
                          calculation and for the vertical line on the plot.
        assets_path: The directory where output plots should be saved.
        loc_detector: Callable for LOC detection. If None, uses rule-based method.

    Returns:
        A dictionary containing results for the final summary map (station name,
        file_stem, time_to_loc), or None if processing fails.
    """
    try:
        # Use provided loc_detector or fall back to rule-based
        if loc_detector is None:
            loc_detector = find_wet_slab_loc_bottom_half
            
        profile, file_stem = _initialize_and_validate_profile(pro_file_path, aspect)
        if not profile or not profile.data or not file_stem:
            return None
        
        if central_date_arg:
            min_date_in_data = central_date_arg - timedelta(days=7)
            max_date_in_data = central_date_arg + timedelta(hours=72)
        else:
            # Use the full time range from the data for the analysis
            min_date_in_data = pd.to_datetime(profile.data.timestamp.values[0])
            max_date_in_data = pd.to_datetime(profile.data.timestamp.values[-1])

        # MODIFICATION: Use the high-resolution summary function with configurable LOC detector
        raw_summary = profile.get_full_timeseries_summary(
            parameters_to_calculate={
                "hs": get_total_snow_depth, 
                "weak_layer": loc_detector,
                "wet_front_lwc": wet_front_lwc,
                "highest_wet_point": get_highest_wet_point,
                "lwc_above_weak": lambda df: lwc_above_weak(df, loc_detector)
            },
            start_date=str(min_date_in_data),
            end_date=str(max_date_in_data),
        ).copy()
        
        if raw_summary.empty:
            return None

        prepared_summary = _unpack_and_prepare_summary(raw_summary)
        
        # Apply the robust persistence logic
        referance_date = central_date_arg or datetime.now()
        summary_full = _persist_loc_height(prepared_summary, referance_date)

        # Generate plots and calculate final metrics
        
        # --- Data Slicing for Plots ---
        # Explicit boolean masking for robust filtering.
        if start_date_arg is None or end_date_arg is None:
            logging.error("Analysis window start/end dates are missing. Cannot create plots.")
            return None # Can't proceed without a valid window
        
        start_dt, end_dt = pd.to_datetime(start_date_arg), pd.to_datetime(end_date_arg)
        
        # Data for line plots (daily summary)
        is_in_window = (summary_full.index >= start_dt) & (summary_full.index <= end_dt)
        summary_for_plot = summary_full[is_in_window]

        # Data for LWC colormesh (potentially higher temporal resolution)
        # Select only the needed variables for efficiency
        full_season_plot_data = profile.data[['lwc', 'height']]
        is_in_lwc_window = (full_season_plot_data.timestamp >= start_dt) & (full_season_plot_data.timestamp <= end_dt)
        lwc_data_for_plot = full_season_plot_data.sel(timestamp=is_in_lwc_window)

        station_metadata = profile.metadata
        del profile


        if not summary_for_plot.empty:
            plot_summary_matplotlib(summary_for_plot, file_stem, station_metadata, lwc_data_for_plot, central_date_arg, assets_path)
            plot_summary_plotly(summary_full, file_stem, station_metadata, central_date_arg, assets_path)
            
        else:
            logging.warning(
                f"No snowpack data found for {file_stem} in the analysis window "
                f"({start_date_arg} to {end_date_arg}). Plots will be skipped."
            )

        time_to_loc = find_time_to_loc(summary_full, reference_date=referance_date)

        return {
            "station_name": station_metadata.get('stationName', file_stem),
            "file_stem": file_stem,
            "time_to_loc": time_to_loc,
            "central_date_str": central_date_arg.strftime('%Y-%m-%d %H:%M') if central_date_arg else None
        }

    except Exception as e:
        logging.error(f"Error processing {pro_file_path.name} for aspect {aspect}: {e}", exc_info=True)
        return None

def worker_wrapper(task_tuple: tuple) -> dict[str, Any] | None:
    """
    Wrapper function to enable multiprocessing by unpacking arguments.

    This function simply unpacks a tuple of arguments and passes them to the
    main `process_single_profile` function. It is used as the target for the
    multiprocessing pool.

    Args:
        task_tuple: A tuple containing the arguments required by
                    `process_single_profile`.

    Returns:
        The result dictionary from `process_single_profile`, or None if an
        error occurred.
    """
    return process_single_profile(*task_tuple)

def _get_closest_synoptic_time(reference_time: datetime) -> datetime:
    """
    Finds the closest standard synoptic time (00, 06, 12, 18 UTC) to a given datetime.

    This ensures that the analysis is centered on a standard meteorological
    reporting time, providing consistency.

    Args:
        reference_time (datetime): The input time (e.g., current time or a
                                     user-specified time).

    Returns:
        datetime: The datetime object representing the closest synoptic time.
    """
    base_date = reference_time.date()
    # Create candidate times on the same day as the reference time
    candidates = [
        datetime.combine(base_date, datetime.min.time()).replace(hour=h)
        for h in [0, 6, 12, 18]
    ]
    # To be thorough, also check the last synoptic time of the previous day
    # and the first of the next day.
    candidates.insert(0, candidates[0] - timedelta(hours=6))
    candidates.append(candidates[1] + timedelta(days=1))

    # Find the candidate with the minimum absolute time difference
    return min(candidates, key=lambda dt: abs(reference_time - dt))


def run_ml_data_collection(input_path: Path, output_dir: Path, central_date: datetime) -> Path | None:
    """
    Run ML training data collection workflow.
    
    Args:
        input_path: Directory containing .pro files
        output_dir: Output directory for training data
        central_date: Central date for analysis
        
    Returns:
        Path to generated training dataset CSV
    """
    try:
        from .ml_data_collection.stall_detector import StallDetector, StallDetectionConfig
        from .ml_data_collection.feature_extractor import LayerFeatureExtractor, FeatureExtractionConfig
    except ImportError:
        logging.error("ML data collection modules not found. Please ensure ml_data_collection package is installed.")
        return None
    
    logging.info("=" * 80)
    logging.info("ML TRAINING DATA COLLECTION")
    logging.info("=" * 80)
    logging.info(f"Input directory: {input_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Central date: {central_date}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure data collection
    stall_config = StallDetectionConfig(
        min_duration_hours=12.0,
        max_duration_hours=240.0,
        min_lwc_threshold=0.04,
        feature_lookback_hours=24.0
    )
    
    feature_config = FeatureExtractionConfig(
        use_dynamic_lookback=True,
        lwc_threshold_pct=1.0,
        max_lookback_hours=72.0,
        fallback_lookback_hours=24.0
    )
    
    # Find all .pro files
    pro_files = list(input_path.rglob('*.pro'))
    logging.info(f"Found {len(pro_files)} .pro files")
    
    # Collect training data
    detector = StallDetector(stall_config)
    extractor = LayerFeatureExtractor(feature_config)
    
    all_features = []
    
    for pro_file in tqdm(pro_files, desc="Processing files"):
        try:
            from .snowpack_reader import SnowpackProfile
            profile = SnowpackProfile(pro_file)
            
            # Detect stalls
            stalls = detector.detect_stalls(profile)
            
            # Extract features for each stall
            for stall in stalls:
                features = extractor.extract_features_for_stall(profile, stall)
                if features is not None:
                    all_features.append(features)
                    
        except Exception as e:
            logging.warning(f"Error processing {pro_file.name}: {e}")
            continue
    
    if not all_features:
        logging.error("No training data collected!")
        return None
    
    # Combine and save
    df = pd.DataFrame(all_features)
    output_file = output_dir / f"ml_training_dataset_{central_date.strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    
    logging.info(f"Collected {len(df)} training examples")
    logging.info(f"Training data saved to: {output_file}")
    
    return output_file


def run_ml_training(
    training_data_path: Path,
    output_dir: Path,
    models_to_train: list,
    tune_hyperparameters: bool = True,
    compute_shap: bool = True
) -> Path:
    """
    Run ML model training workflow.
    
    Args:
        training_data_path: Path to training dataset CSV
        output_dir: Output directory for trained model
        models_to_train: List of model names to train
        tune_hyperparameters: Whether to tune hyperparameters
        compute_shap: Whether to compute SHAP values
        
    Returns:
        Path to trained model directory
    """
    try:
        from .ml_training.model_trainer import ModelTrainer, ModelConfig
        from .ml_training.model_trainer import FeatureImportanceAnalyzer
        from .ml_training.model_trainer import plot_model_comparison, plot_feature_importance, plot_shap_summary
    except ImportError:
        logging.error("ML training modules not found. Please ensure ml_training package is installed.")
        return None
    
    logging.info("=" * 80)
    logging.info("ML MODEL TRAINING")
    logging.info("=" * 80)
    logging.info(f"Training data: {training_data_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Models: {models_to_train}")
    logging.info(f"Hyperparameter tuning: {tune_hyperparameters}")
    logging.info(f"SHAP analysis: {compute_shap}")
    
    # Load training data
    logging.info("Loading training data...")
    df = pd.read_csv(training_data_path)
    logging.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Identify metadata and target columns
    metadata_cols = [
        'event_id', 'pro_file', 'start_time', 'end_time',
        'stall_layer_id', 'layer_above_id', 'layer_below_id',
        'feature_extraction_time', 'lookback_hours', 'station_name',
        'duration_hours', 'confidence', 'n_data_points', 'is_ongoing',
        'lookback_method', 'above_lwc_at_extraction', 'below_lwc_at_extraction',
    ]
    
    irrelevant_cols = [
        'above_absorbed_shortwave', 'above_age', 'above_coordination_number',
        'above_critical_cut_length', 'above_element_ID', 'above_inverse_texture_index',
        'above_soil_volume_fraction', 'above_ssi', 'above_stability_sdef',
        'above_stability_sk38', 'above_stability_sn38', 'above_thermal_conductivity',
        'below_absorbed_shortwave', 'below_age', 'below_coordination_number',
        'below_critical_cut_length', 'below_element_ID', 'below_inverse_texture_index',
        'below_soil_volume_fraction', 'below_ssi', 'below_stability_sdef',
        'below_stability_sk38', 'below_stability_sn38', 'below_thermal_conductivity',
        'distance_from_stall_m', 'example_type', 'interface_coordination_number_diff',
        'interface_ssi_diff', 'interface_ssi_ratio', 'interface_stability_sdef_diff',
        'interface_stability_sdef_ratio', 'interface_stability_sk38_diff',
        'interface_stability_sk38_ratio', 'interface_stability_sn38_diff',
        'interface_stability_sn38_ratio', 'requested_lookback_hours'
    ]
    
    target_col = 'target' if 'target' in df.columns else 'stalled'
    
    # Extract features and target
    feature_cols = [c for c in df.columns
                   if c not in metadata_cols and c not in irrelevant_cols and c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logging.info(f"Features: {X.shape[1]}")
    logging.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Configure training
    config = ModelConfig(
        models_to_train=models_to_train,
        tune_hyperparameters=tune_hyperparameters,
        tuning_method='random',
        n_iter_random=30,
        cv_folds=5,
        use_time_series_cv=True,
        compute_shap_values=compute_shap,
        scale_features=True,
        remove_low_variance=True,
        remove_correlated=True
    )
    
    # Train models
    logging.info("Training models (this may take several minutes)...")
    trainer = ModelTrainer(config)
    trainer.fit(X, y)
    
    logging.info(f"Best model: {trainer.best_model_name_}")
    logging.info(f"Validation ROC-AUC: {trainer.results_[trainer.best_model_name_]['roc_auc']:.4f}")
    
    # Test set evaluation
    test_key = f"{trainer.best_model_name_}_test"
    if test_key in trainer.results_:
        test_res = trainer.results_[test_key]
        logging.info("Test set results:")
        logging.info(f"  Accuracy:  {test_res['accuracy']:.4f}")
        logging.info(f"  Precision: {test_res['precision']:.4f}")
        logging.info(f"  Recall:    {test_res['recall']:.4f}")
        logging.info(f"  F1 Score:  {test_res['f1']:.4f}")
        if 'roc_auc' in test_res:
            logging.info(f"  ROC-AUC:   {test_res['roc_auc']:.4f}")
    
    # Feature importance analysis
    if compute_shap:
        logging.info("Analyzing feature importance...")
        analyzer = FeatureImportanceAnalyzer(
            trainer.best_model_,
            trainer.X_train,
            trainer.feature_names_
        )
        
        importance_results = analyzer.analyze_all(
            trainer.X_val,
            trainer.y_val,
            compute_shap=True,
            shap_sample_size=min(200, len(trainer.X_val))
        )
        
        if 'shap' in importance_results:
            logging.info("Top 10 features (SHAP):")
            for i, (feat, score) in enumerate(importance_results['shap'].head(10).items(), 1):
                logging.info(f"  {i:2d}. {feat:40s} {score:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model
    model_dir = output_dir / 'trained_model'
    trainer.save_model(model_dir)
    logging.info(f"Model saved to: {model_dir}")
    
    # Save plots
    try:
        plot_model_comparison(
            trainer.results_,
            save_path=output_dir / 'model_comparison.png'
        )
        logging.info(f"Saved: {output_dir / 'model_comparison.png'}")
        
        if compute_shap and 'shap' in importance_results:
            plot_feature_importance(
                importance_results,
                top_n=30,
                save_path=output_dir / 'feature_importance.png'
            )
            logging.info(f"Saved: {output_dir / 'feature_importance.png'}")
            
            if 'shap_values' in importance_results:
                plot_shap_summary(
                    importance_results['shap_values'],
                    trainer.X_val.head(200),
                    save_path=output_dir / 'shap_summary.png'
                )
                logging.info(f"Saved: {output_dir / 'shap_summary.png'}")
    except Exception as e:
        logging.warning(f"Could not save all plots: {e}")
    
    # Save feature rankings
    if compute_shap and 'shap' in importance_results:
        importance_df = pd.DataFrame({
            'feature': trainer.feature_names_,
            'shap_importance': importance_results['shap'].reindex(trainer.feature_names_).fillna(0)
        }).sort_values('shap_importance', ascending=False)
        
        importance_df.to_csv(output_dir / 'feature_rankings.csv', index=False)
        logging.info(f"Saved: {output_dir / 'feature_rankings.csv'}")
    
    # Save results summary
    results_summary = []
    for model_name, res in trainer.results_.items():
        if 'test' in model_name:
            continue
        results_summary.append({
            'model': model_name,
            'accuracy': res.get('accuracy', np.nan),
            'precision': res.get('precision', np.nan),
            'recall': res.get('recall', np.nan),
            'f1': res.get('f1', np.nan),
            'roc_auc': res.get('roc_auc', np.nan)
        })
    
    results_df = pd.DataFrame(results_summary).sort_values('roc_auc', ascending=False)
    results_df.to_csv(output_dir / 'model_results.csv', index=False)
    logging.info(f"Saved: {output_dir / 'model_results.csv'}")
    
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"All results saved to: {output_dir}")
    
    return model_dir


def parse_args() -> argparse.Namespace:
    """
    Sets up and parses command-line arguments for the script.

    Defines arguments for controlling the analysis, such as forcing data
    regeneration and setting the analysis time window.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force regeneration of all processed data."
    )
    parser.add_argument(
        "-d", "--date", dest="central_date",
        help="Central date and time for analysis (e.g., 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'). "
             "Rounds to the closest synoptic time (00, 06, 12, 18).",
             default="2025-05-09 12:00"  # Default to a future date for demonstration
    )
    parser.add_argument("-s", "--start", dest="start_date", 
                        help="Start date for analysis (overrides default window)."
    )
    parser.add_argument("-e", "--end", dest="end_date", 
                        help="End date for analysis (overrides default window)."
    )
    parser.add_argument("-i", "--input-dir", dest="input_dir", 
                        type=Path, default=None, 
                        help=f"Override default base directory for .pro files. Default: {config.paths.input_path}"
    )
    parser.add_argument("-o", "--output-dir", dest="output_dir", 
                        type=Path, default=None, 
                        help=f"Override default directory for the final map. Default: {config.paths.results_path}"
    )
    parser.add_argument("-a", "--assets-dir", dest="assets_dir", 
                        type=Path, default=None, 
                        help=f"Override default directory for plot assets. Default: {config.paths.assets_path}"
    )
    parser.add_argument(
        "--loc-mode", 
        dest="loc_mode",
        choices=["rule_based", "ml_only", "hybrid"],
        default=None,
        help=(
            "LOC detection mode: 'rule_based' (traditional capillary barrier), "
            "'ml_only' (ML predictions only), or 'hybrid' (ML with rule-based fallback). "
            "Defaults to LOC_DETECTION_MODE from config if not specified."
        )
    )
    parser.add_argument(
        "--ml-model-path",
        dest="ml_model_path",
        type=Path,
        default=None,
        help="Path to trained ML model directory (overrides config)"
    )
    parser.add_argument(
        "--ml-threshold",
        dest="ml_threshold",
        type=float,
        default=None,
        help="ML probability threshold for LOC detection (default: 0.5)"
    )
    
    # ML Training and Data Collection Flags
    parser.add_argument(
        "--collect-ml-data",
        action="store_true",
        help="Collect ML training data from .pro files and exit"
    )
    parser.add_argument(
        "--train-ml-model",
        action="store_true",
        help="Train a new ML model and exit"
    )
    parser.add_argument(
        "--ml-training-data",
        dest="ml_training_data",
        type=Path,
        default=None,
        help="Path to ML training dataset CSV (for --train-ml-model)"
    )
    parser.add_argument(
        "--ml-training-output",
        dest="ml_training_output",
        type=Path,
        default=None,
        help="Output directory for trained model (default: results/trained_models/TIMESTAMP)"
    )
    parser.add_argument(
        "--ml-training-models",
        dest="ml_training_models",
        nargs='+',
        default=['xgboost', 'lightgbm', 'random_forest'],
        help="Models to train (space-separated)"
    )
    parser.add_argument(
        "--no-ml-tune",
        action="store_true",
        help="Skip hyperparameter tuning during ML training (faster)"
    )
    parser.add_argument(
        "--no-ml-shap",
        action="store_true",
        help="Skip SHAP analysis during ML training (faster)"
    )
    
    return parser.parse_args()


def main():
    """
    Main orchestrator for the entire analysis and mapping workflow.

    This function handles argument parsing, date setup, geodata preparation,
    and the parallel processing of snowpack files before generating the
    final summary map.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="wetting_front_tracker.log",
        filemode="w"   # overwrite each run; use "a" to append
    )    
    
    args = parse_args()
    
    # --- Handle ML Training Workflows (Early Exit) ---
    # These workflows run independently and exit before the main analysis
    
    if args.collect_ml_data:
        logging.info("ML data collection mode activated")
        input_path = args.input_dir or config.paths.input_path
        
        # Determine output directory
        if args.ml_training_output:
            output_dir = args.ml_training_output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = config.paths.results_path / 'trained_models' / timestamp

        # Use central date or current date
        if args.central_date:
            try:
                central_date = datetime.strptime(args.central_date, '%Y-%m-%d %H:%M')
            except ValueError:
                try:
                    central_date = datetime.strptime(args.central_date, '%Y-%m-%d')
                except ValueError:
                    central_date = datetime.now()
        else:
            central_date = datetime.now()
        
        # Run data collection
        dataset_path = run_ml_data_collection(input_path, output_dir, central_date)
        
        if dataset_path:
            logging.info(f"\nML training data collection complete!")
            logging.info(f"Dataset: {dataset_path}")
            logging.info(f"\nNext step: Train a model with:")
            logging.info(f"  python -m src.wetting_front_tracker.main --train-ml-model --ml-training-data {dataset_path}")
        else:
            logging.error("ML data collection failed!")
        
        return  # Exit after data collection
    
    if args.train_ml_model:
        logging.info("ML model training mode activated")
        
        # Validate training data path
        if not args.ml_training_data:
            logging.error("--ml-training-data is required for --train-ml-model")
            logging.error("Usage: --train-ml-model --ml-training-data path/to/dataset.csv")
            return
        
        if not args.ml_training_data.exists():
            logging.error(f"Training data not found: {args.ml_training_data}")
            return
        
        # Determine output directory
        if args.ml_training_output:
            output_dir = args.ml_training_output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = config.paths.results_path / 'trained_models' / timestamp
        
        # Run training
        model_dir = run_ml_training(
            args.ml_training_data,
            output_dir,
            args.ml_training_models,
            tune_hyperparameters=not args.no_ml_tune,
            compute_shap=not args.no_ml_shap
        )
        
        if model_dir:
            logging.info(f"\nML model training complete!")
            logging.info(f"Model saved to: {model_dir}")
            logging.info(f"\nNext step: Use the model with:")
            logging.info(f"  python -m src.wetting_front_tracker.main \\")
            logging.info(f"    --date 2025-05-09 \\")
            logging.info(f"    --loc-mode hybrid \\")
            logging.info(f"    --ml-model-path {model_dir}")
        else:
            logging.error("ML model training failed!")
        
        return  # Exit after training
    
    # --- Normal Analysis Workflow Continues Below ---
    
    # --- Path Configuration ---
    input_path = args.input_dir or config.paths.input_path
    output_path = args.output_dir or config.paths.results_path
    assets_path = args.assets_dir or config.paths.assets_path
    output_path.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)
    summary_map_path = config.paths.summary_map_html
    logging.info(f"Input .pro directory: {input_path}")
    logging.info(f"Output map directory: {output_path}")
    logging.info(f"Plot assets directory: {assets_path}")

    # --- Date Handling (Single Day) ---
    if args.central_date:
        try:
            # First, try parsing the full date and time
            initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d %H:%M')
        except ValueError:
            try:
                # If that fails, try parsing with date only
                initial_ref_time = datetime.strptime(args.central_date, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid date format for '{args.central_date}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")
                return
    else:
        # If no date is provided, use the current time
        initial_ref_time = datetime.now()

    central_date = _get_closest_synoptic_time(initial_ref_time)
    start_date = (central_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    end_date = (central_date + timedelta(hours=72)).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Processing for central date: {central_date.strftime('%Y-%m-%d %H:%M')}")

    # --- LOC Detection Configuration ---
    # Determine LOC detection mode from args or config
    loc_mode = args.loc_mode or LOC_DETECTION_MODE
    
    # Update ML config based on command-line arguments
    ml_config = ML_CONFIG
    if args.ml_model_path:
        ml_config.model_path = args.ml_model_path
        ml_config.enabled = True
        logging.info(f"Using ML model from command-line argument: {args.ml_model_path}")
    if args.ml_threshold is not None:
        ml_config.probability_threshold = args.ml_threshold
        logging.info(f"Using ML threshold from command-line: {args.ml_threshold}")
    
    # Get the appropriate LOC detection function
    loc_detector = get_loc_detection_function(loc_mode, ml_config)
    logging.info(f"LOC detection mode: {loc_mode}")
    
    # Log ML configuration details if ML is being used
    if loc_mode in ["ml_only", "hybrid"]:
        if ml_config.enabled and ml_config.model_path:
            logging.info(f"ML model path: {ml_config.model_path}")
            logging.info(f"ML probability threshold: {ml_config.probability_threshold}")
        else:
            logging.warning("ML mode selected but ML not properly configured, using rule-based fallback")

    # --- Geodata Preparation ---
    input_geojson = config.get_input_polygons_path()
    if args.regenerate_data or not config.paths.linked_polygons.exists():
        logging.info("Regenerating processed data...")
        generate_pro_file_manifest(input_path, config.paths.pro_file_manifest)
        prepare_aspect_polygons(input_geojson, config.paths.aspect_polygons, args.regenerate_data)
        link_polygons_to_pro_files(config.paths.aspect_polygons, config.paths.snowpack_locations_csv, config.paths.linked_polygons)

    try:
        linked_gdf = gpd.read_file(config.paths.linked_polygons)
        with open(config.paths.pro_file_manifest, 'r') as f:
            pro_file_manifest = json.load(f)
    except FileNotFoundError:
        logging.error(f"Manifest file not found at {config.paths.pro_file_manifest}. Please run with --regenerate-data.")
        return

    # --- NEW: Conditional Data Download Step ---
    if config.data_source.is_remote:
        logging.info("Checking for remote .pro files to download...")
        # Create a unique set of filenames to check
        required_files = {Path(p).name for p in linked_gdf['pro_file_path'].unique()}
        for file_name in tqdm(required_files, desc="Updating data files"):
            ensure_pro_file_is_local(file_name, input_path, config.data_source.remote_url, central_date)
    else:
        logging.info("Skipping remote file check. PRO_FILES_SOURCE is 'local'.")

    # --- Task Generation (Single Day) ---
    tasks = []
    for poly in linked_gdf.itertuples(index=False):
    # Get the filename from the geodataframe
        file_name = Path(str(poly.pro_file_path)).name
        
        # Look up the full path from our manifest dictionary
        if full_pro_path_str := pro_file_manifest.get(file_name):
            effective_path = Path(full_pro_path_str)
            # Include loc_detector in task tuple
            tasks.append((effective_path, poly.aspect, start_date, end_date, central_date, assets_path, loc_detector))
        else:
            logging.warning(f"File '{file_name}' from geojson not found in the manifest. Skipping.")
    if not tasks:
        logging.warning("No tasks were generated for processing. Exiting.")
        return

    # --- Parallel Processing ---
    logging.info(f"Starting parallel processing on {len(tasks)} polygons...")
    cpu_cores = os.cpu_count()
    worker_count = int(max(1, cpu_cores / 4 )) if cpu_cores else 1 
    with multiprocessing.Pool(processes=worker_count) as pool:
        results = list(tqdm(pool.map(worker_wrapper, tasks, chunksize=1), total=len(tasks)))

    # --- Aggregation and Final Map ---
    if valid_results := [res for res in results if res is not None]:
        results_df = pd.DataFrame(valid_results)
        final_gdf = gpd.GeoDataFrame(pd.concat([
            linked_gdf.reset_index(drop=True),
            results_df.reset_index(drop=True)
        ], axis=1))
        
        logging.info("All polygons processed. Creating summary map...")
        create_folium_map(final_gdf, summary_map_path, central_date, assets_path )
    else:
        logging.info("No valid results were generated. Skipping map creation.")


if __name__ == "__main__":
    main()

=======
    main()
>>>>>>> 6343557 (implementin ml LOC detection)
