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
                                    wet_front_lwc, find_wet_slab_loc,
                                    find_wet_slab_loc_bottom_half,
                                    mean_lwc_above_reference)
except ImportError:
    from wet_front_tracker import (find_time_to_loc, get_highest_wet_point,
                                    get_total_snow_depth, lwc_above_weak,
                                    wet_front_lwc, find_wet_slab_loc,
                                    find_wet_slab_loc_bottom_half,
                                    mean_lwc_above_reference)
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

# Diagnostic wrapper is loaded conditionally based on --enable-diagnostics flag
# (see parse_args and main function)

logger = logging.getLogger(__name__)


# --- Picklable Wrapper Classes (Fix for Multiprocessing Error) ---

class LocResultStandardizer:
    """
    Wraps a detection function to ensure it always returns a list of (height, prob) tuples.
    Defined at module level to ensure it can be pickled by multiprocessing.
    """
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        try:
            res = self.func(df)
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
        
        if isinstance(res, list): return res

        # Fallback for scalar return
        try:
            return [(float(res), 1.0)]
        except:
            return []

class MLDetectorCallable:
    """
    Callable wrapper for ML-only detection.
    Loads the detector lazily on the worker process to ensure picklability and reduce overhead.
    """
    def __init__(self, model_path: Path, probability_threshold: float, top_n: int):
        self.model_path = model_path
        self.probability_threshold = probability_threshold
        self.top_n = top_n
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            try:
                from .ml_loc_detector import MLLocDetector
            except ImportError:
                from ml_loc_detector import MLLocDetector
            self._detector = MLLocDetector(self.model_path, self.probability_threshold)
        return self._detector

    def __call__(self, df):
        return self.detector.find_ml_loc(df, top_n=self.top_n)


def get_loc_detection_function(mode: str, ml_config: MLModelConfig, top_n: int = 5):
    """Select LOC detection function."""
    try:    
        from .wet_front_tracker import find_wet_slab_loc
    except ImportError:
        from wet_front_tracker import find_wet_slab_loc
    
    if mode == "rule_based":
        logger.info("Using rule-based LOC detection")
        return LocResultStandardizer(find_wet_slab_loc)
    
    elif mode == "ml_only":
        if not ml_config.enabled or ml_config.model_path is None:
            logger.warning("ML config disabled or path missing, falling back to rule-based.")
            return LocResultStandardizer(find_wet_slab_loc)
        
        logger.info(f"Using ML-only LOC detection")
        return MLDetectorCallable(ml_config.model_path, ml_config.probability_threshold, top_n)
    
    elif mode == "hybrid":
        if not ml_config.enabled or ml_config.model_path is None:
            logger.warning("ML config disabled or path missing, falling back to rule-based.")
            return LocResultStandardizer(find_wet_slab_loc)
        
        logger.info(f"Using hybrid LOC detection")
        return create_hybrid_loc_detector(
            model_path=ml_config.model_path,
            use_ml_primary=ml_config.use_ml_primary,
            ml_threshold=ml_config.probability_threshold,
            rule_based_fallback=find_wet_slab_loc,
            top_n=top_n
        )
    else:
        return LocResultStandardizer(find_wet_slab_loc)
    

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
                           loc_detector_rule: Callable | None = None,
                           loc_detector_ml: Callable | None = None) -> dict[str, Any] | None:
    """
    Handles the full analysis workflow for a single polygon.
    
    Runs both rule-based and ML LOC detectors (when available) in a single
    pass through the profile, producing independent time_to_loc values for
    each. Shared metrics (HS, wet front, highest wet point) are computed once.
    """
    try:
        if loc_detector_rule is None:
            loc_detector_rule = LocResultStandardizer(find_wet_slab_loc_bottom_half)
            
        profile, file_stem = _initialize_and_validate_profile(pro_file_path, aspect)
        if not profile or not profile.data or not file_stem:
            return None
        
        if central_date_arg:
            min_date_in_data = central_date_arg - timedelta(days=7)
            max_date_in_data = central_date_arg + timedelta(hours=72)
        else:
            min_date_in_data = pd.to_datetime(profile.data.timestamp.values[0])
            max_date_in_data = pd.to_datetime(profile.data.timestamp.values[-1])

        # --- Build parameter dict: shared metrics + both detectors ---
        params = {
            "hs": get_total_snow_depth, 
            "wet_front_lwc": wet_front_lwc,
            "highest_wet_point": get_highest_wet_point,
            "weak_layer_rule": loc_detector_rule,
        }
        if loc_detector_ml is not None:
            params["weak_layer_ml"] = loc_detector_ml

        raw_summary = profile.get_full_timeseries_summary(
            parameters_to_calculate=params,
            start_date=str(min_date_in_data),
            end_date=str(max_date_in_data),
        ).copy()
        
        if raw_summary.empty:
            return None

        referance_date = central_date_arg or datetime.now()

        # --- Process rule-based LOC ---
        rule_summary = _unpack_and_prepare_summary(raw_summary.rename(
            columns={c: c.replace('weak_layer_rule', 'weak_layer') 
                     for c in raw_summary.columns if 'weak_layer_rule' in c}
        ).copy())
        rule_summary = _persist_loc_height(rule_summary, referance_date)
        
        # Multi-candidate worst-case for rule-based
        rule_times = [find_time_to_loc(rule_summary, reference_date=referance_date)]
        for col in [c for c in rule_summary.columns 
                    if c.startswith('weak_layer_height_') and c != 'weak_layer_height_0']:
            temp_df = rule_summary.copy()
            temp_df['weak_layer_height'] = temp_df[col]
            temp_df = _persist_loc_height(temp_df, referance_date)
            rule_times.append(find_time_to_loc(temp_df, reference_date=referance_date))
        time_to_loc_rule = _get_worst_case_time(rule_times)

        # --- Process ML LOC (if available) ---
        time_to_loc_ml = None
        ml_summary = None
        if loc_detector_ml is not None and 'weak_layer_ml' in raw_summary.columns:
            ml_summary = _unpack_and_prepare_summary(raw_summary.rename(
                columns={c: c.replace('weak_layer_ml', 'weak_layer') 
                         for c in raw_summary.columns if 'weak_layer_ml' in c}
            ).copy())
            ml_summary = _persist_loc_height(ml_summary, referance_date)
            
            ml_times = [find_time_to_loc(ml_summary, reference_date=referance_date)]
            for col in [c for c in ml_summary.columns 
                        if c.startswith('weak_layer_height_') and c != 'weak_layer_height_0']:
                temp_df = ml_summary.copy()
                temp_df['weak_layer_height'] = temp_df[col]
                temp_df = _persist_loc_height(temp_df, referance_date)
                ml_times.append(find_time_to_loc(temp_df, reference_date=referance_date))
            time_to_loc_ml = _get_worst_case_time(ml_times)

        # --- Plotting (use rule-based summary for the plots) ---
        if start_date_arg is None or end_date_arg is None:
            logging.error("Analysis window start/end dates are missing.")
            return None
        
        start_dt, end_dt = pd.to_datetime(start_date_arg), pd.to_datetime(end_date_arg)
        is_in_window = (rule_summary.index >= start_dt) & (rule_summary.index <= end_dt)
        summary_for_plot = rule_summary[is_in_window]

        full_season_plot_data = profile.data[['lwc', 'height']]
        is_in_lwc_window = (full_season_plot_data.timestamp >= start_dt) & (full_season_plot_data.timestamp <= end_dt)
        lwc_data_for_plot = full_season_plot_data.sel(timestamp=is_in_lwc_window)

        station_metadata = profile.metadata
        del profile

        if not summary_for_plot.empty:
            # Prepare ML summary for plot window (if available)
            ml_for_plot = None
            if ml_summary is not None:
                ml_in_window = (ml_summary.index >= start_dt) & (ml_summary.index <= end_dt)
                ml_for_plot = ml_summary[ml_in_window] if ml_in_window.any() else None

            plot_summary_matplotlib(summary_for_plot, file_stem, station_metadata, 
                                   lwc_data_for_plot, central_date_arg, assets_path,
                                   ml_loc_df=ml_for_plot)
            plot_summary_plotly(rule_summary, file_stem, station_metadata, 
                               central_date_arg, assets_path,
                               ml_loc_df=ml_summary)

        return {
            "station_name": station_metadata.get('stationName', file_stem),
            "file_stem": file_stem,
            "time_to_loc_rule": time_to_loc_rule,
            "time_to_loc_ml": time_to_loc_ml,
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
        lookback_hours=24.0,
        extract_all_parameters=True,
        compute_differences=True,
        compute_ratios=True,
        compute_gradients=True
    )
    
    pro_files = list(input_path.rglob('*.pro'))
    logging.info(f"Found {len(pro_files)} .pro files")
    
    detector = StallDetector(stall_config)
    extractor = LayerFeatureExtractor(feature_config)
    all_features = []
    
    # Import data collection implementation
    # We reuse the logic from collect_ml_data.py by calling it here would be ideal
    # But for now, let's use a simplified version or the one imported
    try:
        from .ml_data_collection.collect_ml_data import collect_ml_data, MLDataCollectionConfig
        
        collection_config = MLDataCollectionConfig()
        collection_config.output_dir = output_dir
        collection_config.stall_config = stall_config
        collection_config.feature_config = feature_config
        # Ensure negative sampling is on by default
        collection_config.negative_sampling_strategy = 'combined' 
        
        stall_df, features_df = collect_ml_data(input_path, output_dir, collection_config)
        
        if features_df.empty:
            return None
            
        output_file = output_dir / "ml_training_dataset.csv"
        return output_file
        
    except ImportError:
        # Fallback if collect_ml_data isn't importable
        logging.warning("Using fallback collection loop (might lack negative sampling)")
        for pro_file in tqdm(pro_files, desc="Processing files"):
            try:
                from .snowpack_reader import SnowpackProfile
                from .wet_front_tracker import wet_front_lwc
                
                profile = SnowpackProfile(str(pro_file))
                
                # Needed for stall detector
                summary = profile.get_full_timeseries_summary(
                    parameters_to_calculate={"wet_front_lwc": wet_front_lwc}
                )
                if 'wet_front_lwc' in summary.columns:
                     summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                        summary['wet_front_lwc'].tolist(), index=summary.index)
                
                wetting_front = summary['wet_front_lwc_height']
                station_name = pro_file.stem
                
                stalls = detector.find_stalls(profile, wetting_front, station_name)
                
                for stall in stalls:
                    # This only collects POSITIVE examples
                    features = extractor.extract_features_for_interface(
                        profile, 
                        stall.start_time - timedelta(hours=24),
                        stall.stall_layer_id,
                        stall.layer_above_id,
                        stall.layer_below_id
                    )
                    if features:
                        row = stall.to_dict()
                        row.update(features)
                        row['stalled'] = 1
                        all_features.append(row)
                        
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
    
    return model_dir


def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run wet snow tracker analysis on SNOWPACK .pro files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--regenerate-data", action="store_true", help="Force regeneration of all processed data.")
    parser.add_argument("-d", "--date", dest="central_date", default="2025-04-06 18:00", help="Central date and time.")
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
    
    # Diagnostic Args
    parser.add_argument("--enable-diagnostics", action="store_true", help="Enable diagnostic wrapper for debugging (wraps functions to track statistics).")
    
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
    
    # --- Enable Diagnostics if requested ---
    if args.enable_diagnostics:
        try:
            import diagnostic_wrapper
            diagnostic_wrapper.enable_diagnostics()
            logging.info("Diagnostic wrapper enabled")
        except ImportError:
            logging.warning("Could not import diagnostic_wrapper - diagnostics disabled")
    
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
    
    # --loc-mode rule_based disables ML even if a model is available
    if args.loc_mode == "rule_based":
        ml_config.enabled = False

    if args.ml_threshold is not None:
        ml_config.probability_threshold = args.ml_threshold

    # --- Build both detectors ---
    loc_detector_rule = LocResultStandardizer(find_wet_slab_loc_bottom_half)
    logging.info("Rule-based LOC detector: capillary barrier (bottom half)")

    loc_detector_ml = None
    if ml_config.enabled and ml_config.model_path is not None:
        loc_detector_ml = MLDetectorCallable(
            ml_config.model_path, ml_config.probability_threshold, top_n=5
        )
        logging.info(f"ML LOC detector enabled (threshold={ml_config.probability_threshold})")
    else:
        logging.info("ML LOC detector: disabled (no model path or config)")

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
            tasks.append((Path(full_pro_path_str), poly.aspect, start_date, end_date, 
                         central_date, assets_path, loc_detector_rule, loc_detector_ml))

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
        create_folium_map(final_gdf, config.paths.summary_map_html, central_date, assets_path)
    else:
        logging.info("No valid results generated.")


if __name__ == "__main__":
    main()
    