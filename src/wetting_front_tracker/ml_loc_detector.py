"""
ml_loc_detector.py
==================

ML-based Layer of Concern (LOC) detection for wet slab avalanches.
Updated to return multiple LOC candidates (Top-N).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr

try:
    from .ml_training.model_trainer import ModelTrainer
except ImportError:
    from wetting_front_tracker.ml_training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class MLLocDetector:
    """
    Machine learning-based LOC detector.
    """
    
    def __init__(
        self, 
        model_path: Path,
        probability_threshold: float = 0.5
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_path = model_path
        self.probability_threshold = probability_threshold
        
        # Load Model
        logger.info(f"Loading trained model from {model_path}")
        self.trainer = ModelTrainer.load_model(model_path)
        self.feature_names = self.trainer.feature_names_
        
        logger.info(f"Initialized MLLocDetector (Threshold: {probability_threshold})")

    def detect_timeseries(self, profile_data: xr.Dataset, top_n: int = 3) -> pd.DataFrame:
        """Run detection on an entire Xarray timeseries."""
        if profile_data is None or profile_data.timestamp.size == 0:
            return pd.DataFrame()

        results = []
        timestamps = profile_data.timestamp.values
        
        logger.info(f"Running ML detection on {len(timestamps)} timesteps...")
        
        for ts in tqdm(timestamps, desc="Detecting LOC"):
            try:
                # Slice xarray for specific timestamp
                ds_slice = profile_data.sel(timestamp=ts)
                
                if hasattr(ds_slice, 'as_numpy'):
                    ds_slice = ds_slice.as_numpy()
                elif hasattr(ds_slice, 'compute'):
                    ds_slice = ds_slice.compute()
                
                df_slice = ds_slice.to_dataframe().reset_index()
                df_clean = df_slice.dropna(subset=['height'])
                
                if df_clean.empty:
                    continue
                
                # Run detection on this snapshot
                # Returns list of tuples [(height, prob), ...]
                locs = self.find_ml_loc(df_clean, top_n=top_n)
                
                if locs:
                    # Flatten for DataFrame: loc_0_height, loc_0_prob, etc.
                    row = {'timestamp': ts}
                    for i, (h, p) in enumerate(locs):
                        row[f'loc_height_{i}'] = h
                        row[f'loc_prob_{i}'] = p
                    results.append(row)
                    
            except Exception as e:
                logger.debug(f"Detection failed at {ts}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results).set_index('timestamp').sort_index()

    def find_ml_loc(
        self,
        df: pd.DataFrame,
        top_n: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Find the top N most likely Layers of Concern.
        
        Returns:
            List of (height, probability) tuples, sorted by probability descending.
            Empty list if no LOCs found.
        """
        predictions_df = self.predict_all_interfaces(df)
        
        if predictions_df.empty:
            return []
        
        # Filter for candidates above threshold
        candidates = predictions_df[predictions_df['stall_probability'] >= self.probability_threshold]
        
        if candidates.empty:
            return []
        
        # Sort by probability (descending) and take top N
        candidates = candidates.sort_values('stall_probability', ascending=False).head(top_n)
        
        # Convert to list of tuples
        results = []
        for _, row in candidates.iterrows():
            results.append((float(row['below_height']), float(row['stall_probability'])))
            
        return results

    def predict_all_interfaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features and probabilities for all interfaces."""
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        df = df.sort_values('height').reset_index(drop=True)
        interface_rows = []
        
        for i in range(len(df) - 1):
            below = df.iloc[i]
            above = df.iloc[i+1]
            features = self._calculate_interface_features(below, above)
            features['interface_index'] = i
            features['below_height'] = below.get('height', np.nan)
            features['above_height'] = above.get('height', np.nan)
            interface_rows.append(features)
            
        if not interface_rows:
            return pd.DataFrame()
            
        features_df = pd.DataFrame(interface_rows)
        
        # Align with Model Features
        X = pd.DataFrame(index=features_df.index)
        for col in self.feature_names:
            if col in features_df.columns:
                X[col] = features_df[col]
            else:
                X[col] = np.nan

        # Predict
        probs = self.trainer.predict_proba(X)[:, 1]
        
        results = features_df[['interface_index', 'below_height', 'above_height']].copy()
        results['stall_probability'] = probs
        
        return results

    def _calculate_interface_features(self, below: pd.Series, above: pd.Series) -> Dict[str, float]:
        """Calculate structural features for the current interface."""
        # (Same feature extraction logic as before - kept for brevity)
        f = {}
        params = [
            'density', 'temperature', 'lwc', 'sphericity', 'bond_size', 
            'grain_size', 'grain_type', 'optical_grain_size', 'stress', 
            'viscosity', 'viscous_deformation_rate', 'hand_hardness', 
            'shear_strength', 'temperature_gradient', 'grain_size_difference', 
            'hardness_difference', 'ice_volume_fraction'
        ]
        for p in params:
            f[f'above_{p}'] = float(above.get(p, np.nan))
            f[f'below_{p}'] = float(below.get(p, np.nan))

        h_above = float(above.get('height', np.nan))
        h_below = float(below.get('height', np.nan))
        f['stall_height'] = (h_above + h_below) / 2.0
        dist = abs(h_above - h_below)
        f['interface_layer_distance'] = dist

        for p in params:
            val_a = f[f'above_{p}']
            val_b = f[f'below_{p}']
            if pd.notna(val_a) and pd.notna(val_b):
                f[f'interface_{p}_diff'] = val_a - val_b
            else:
                f[f'interface_{p}_diff'] = np.nan

        ratio_params = [
            'lwc', 'bond_size', 'grain_size', 'optical_grain_size', 
            'stress', 'viscosity', 'viscous_deformation_rate', 'shear_strength'
        ]
        for p in ratio_params:
            val_a = f[f'above_{p}']
            val_b = f[f'below_{p}']
            if pd.notna(val_a) and pd.notna(val_b) and val_b != 0:
                f[f'interface_{p}_ratio'] = val_a / val_b
            else:
                f[f'interface_{p}_ratio'] = np.nan

        grad_params = ['density', 'temperature', 'lwc', 'grain_size']
        if pd.notna(dist) and dist > 0:
            for p in grad_params:
                diff = f.get(f'interface_{p}_diff', np.nan)
                if pd.notna(diff):
                    f[f'interface_{p}_gradient'] = diff / dist
                else:
                    f[f'interface_{p}_gradient'] = np.nan
        else:
            for p in grad_params:
                f[f'interface_{p}_gradient'] = np.nan
        return f


# ---------------------------------------------------------------------------
# Hybrid Detector (Updated for List Output)
# ---------------------------------------------------------------------------

class HybridLocDetector:
    """
    Callable class for hybrid detection returning multiple LOCs.
    """
    def __init__(
        self,
        model_path: Optional[Path],
        use_ml_primary: bool,
        ml_threshold: float,
        rule_based_fallback: Optional[Callable],
        top_n: int = 3
    ):
        self.use_ml_primary = use_ml_primary
        self.rule_based_fallback = rule_based_fallback
        self.top_n = top_n
        self.ml_detector = None
        
        if model_path is not None and model_path.exists():
            try:
                self.ml_detector = MLLocDetector(model_path, ml_threshold)
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}. Using rule-based only.")

    def __call__(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Returns list of (height, score) tuples."""
        # 1. Try Primary Method
        if self.use_ml_primary and self.ml_detector:
            locs = self.ml_detector.find_ml_loc(df, top_n=self.top_n)
            if locs: return locs
        elif not self.use_ml_primary and self.rule_based_fallback:
            locs = self._run_rule_based(df)
            if locs: return locs

        # 2. Try Fallback Method
        if not self.use_ml_primary and self.ml_detector:
            locs = self.ml_detector.find_ml_loc(df, top_n=self.top_n)
            if locs: return locs
        elif self.use_ml_primary and self.rule_based_fallback:
            locs = self._run_rule_based(df)
            if locs: return locs
        
        return []

    def _run_rule_based(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Normalize rule-based output to list format, handling depth->height conversion."""
        try:
            res = self.rule_based_fallback(df)
        except Exception:
            return []

        if isinstance(res, dict):
            # Case A: Explicit Height (Ideal)
            if 'loc_height' in res:
                return [(float(res['loc_height']), 1.0)]
            
            # Case B: Depth (Convert to Height)
            # Height = Total Snow Depth (HS) - Depth
            if 'loc_depth' in res and 'height' in df.columns:
                hs = df['height'].max()
                if pd.notna(hs):
                    return [(hs - float(res['loc_depth']), 1.0)]
                
        elif isinstance(res, tuple):
            return [res]
            
        elif res is not None:
            # Case C: Scalar result (Assume it is Height if > 0 and typical snow depth)
            try:
                val = float(res)
                return [(val, 1.0)]
            except:
                pass
                
        return []

def create_hybrid_loc_detector(
    model_path: Optional[Path] = None,
    use_ml_primary: bool = True,
    ml_threshold: float = 0.5,
    rule_based_fallback: Callable = None,
    top_n: int = 3
) -> Callable:
    return HybridLocDetector(
        model_path=model_path,
        use_ml_primary=use_ml_primary,
        ml_threshold=ml_threshold,
        rule_based_fallback=rule_based_fallback,
        top_n=top_n
    )

def find_ml_loc_simple(df: pd.DataFrame, model_path: Path, top_n: int = 3) -> List[Tuple[float, float]]:
    detector = MLLocDetector(model_path)
    return detector.find_ml_loc(df, top_n=top_n)