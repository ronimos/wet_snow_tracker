"""
ml_loc_detector.py
==================

ML-based Layer of Concern (LOC) detection for wet slab avalanches.

This module extracts features from a *current* snowpack profile (snapshot)
and uses a trained model to predict if an interface has the structural 
characteristics of a wetting front stall.

Usage:
    # Initialize
    detector = MLLocDetector(Path("results/model/trained_model"))
    
    # 1. Predict on a single profile (Pandas DataFrame)
    loc_height, prob = detector.find_ml_loc(df)
    
    # 2. Predict on a full time series (Xarray)
    results_df = detector.detect_timeseries(profile.data)

Author: Ron Simenhois
Updated: November 2025
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr

from .ml_training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class MLLocDetector:
    """
    Machine learning-based LOC detector.
    
    Calculates interface properties (gradients, differences, ratios) for the
    current profile state and predicts stall probability using a trained XGBoost model.
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

    def detect_timeseries(self, profile_data: xr.Dataset) -> pd.DataFrame:
        """
        Run detection on an entire Xarray timeseries.
        """
        if profile_data is None or profile_data.timestamp.size == 0:
            return pd.DataFrame()

        results = []
        timestamps = profile_data.timestamp.values
        
        logger.info(f"Running ML detection on {len(timestamps)} timesteps...")
        
        for ts in tqdm(timestamps, desc="Detecting LOC"):
            try:
                # Slice xarray for specific timestamp
                ds_slice = profile_data.sel(timestamp=ts)
                
                # Handle GPU/CuPy -> Numpy conversion
                if hasattr(ds_slice, 'as_numpy'):
                    ds_slice = ds_slice.as_numpy()
                elif hasattr(ds_slice, 'compute'):
                    ds_slice = ds_slice.compute()
                
                # Convert to DataFrame and clean
                # reset_index() ensures we have columns like 'height', 'density', etc.
                df_slice = ds_slice.to_dataframe().reset_index()
                df_clean = df_slice.dropna(subset=['height'])
                
                if df_clean.empty:
                    continue
                
                # Run detection on this snapshot
                loc_height, prob = self.find_ml_loc(df_clean)
                
                if loc_height is not None:
                    results.append({
                        'timestamp': ts,
                        'loc_height': loc_height,
                        'stall_probability': prob
                    })
                    
            except Exception as e:
                logger.debug(f"Detection failed at {ts}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results).set_index('timestamp').sort_index()

    def find_ml_loc(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Find the most likely Layer of Concern in a single profile dataframe.
        
        Args:
            df: Pandas DataFrame representing one moment in time. 
                Must contain columns: height, density, grain_size, etc.
                
        Returns:
            (loc_height, probability) or (None, None)
        """
        # 1. Generate predictions for every interface in the profile
        predictions_df = self.predict_all_interfaces(df)
        
        if predictions_df.empty:
            return None, None
        
        # 2. Filter for LOC candidates
        candidates = predictions_df[predictions_df['is_loc']]
        
        if candidates.empty:
            return None, None
        
        # 3. Return the best candidate
        # We return 'below_height' because the "LOC" is typically the layer 
        # *beneath* the stalling interface (the bed surface).
        best = candidates.iloc[0]
        return float(best['below_height']), float(best['stall_probability'])

    def predict_all_interfaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features and probabilities for all interfaces in the DataFrame.
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        # Ensure sorted by height (bottom to top)
        df = df.sort_values('height').reset_index(drop=True)
        
        # 1. Extract features for every interface
        interface_rows = []
        
        for i in range(len(df) - 1):
            below = df.iloc[i]
            above = df.iloc[i+1]
            
            # Calculate the specific features the model needs
            features = self._calculate_interface_features(below, above)
            
            # Metadata for results
            features['interface_index'] = i
            features['below_height'] = below.get('height', np.nan)
            features['above_height'] = above.get('height', np.nan)
            
            interface_rows.append(features)
            
        if not interface_rows:
            return pd.DataFrame()
            
        features_df = pd.DataFrame(interface_rows)
        
        # 2. Align with Model Features
        # Create a matrix X with exact columns the model expects
        X = pd.DataFrame(index=features_df.index)
        for col in self.feature_names:
            if col in features_df.columns:
                X[col] = features_df[col]
            else:
                # If input profile lacks a parameter (e.g. 'viscosity'), fill NaN
                X[col] = np.nan

        # 3. Predict
        probs = self.trainer.predict_proba(X)[:, 1]
        predictions = self.trainer.predict(X)
        
        # 4. Format Results
        results = features_df[['interface_index', 'below_height', 'above_height']].copy()
        results['stall_probability'] = probs
        results['prediction'] = predictions
        
        # Identify LOCs (Probability > Threshold)
        # If multiple exceed threshold, pick the highest probability
        results['is_loc'] = False
        if (results['stall_probability'] >= self.probability_threshold).any():
            best_idx = results.loc[results['stall_probability'] >= self.probability_threshold, 'stall_probability'].idxmax()
            results.loc[best_idx, 'is_loc'] = True
            
        return results

    def _calculate_interface_features(self, below: pd.Series, above: pd.Series) -> Dict[str, float]:
        """
        Calculate structural features for the current interface.
        Matches the logic used in training feature extraction.
        """
        f = {}
        
        # --- A. Basic Properties (Above & Below) ---
        # We extract every parameter present in the dataframe
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

        # --- B. Interface Geometry ---
        h_above = float(above.get('height', np.nan))
        h_below = float(below.get('height', np.nan))
        
        # For prediction, 'stall_height' is the current interface height
        f['stall_height'] = (h_above + h_below) / 2.0
        dist = abs(h_above - h_below)
        f['interface_layer_distance'] = dist

        # --- C. Differences (Above - Below) ---
        # Based on feature_names.json
        for p in params:
            val_a = f[f'above_{p}']
            val_b = f[f'below_{p}']
            if pd.notna(val_a) and pd.notna(val_b):
                f[f'interface_{p}_diff'] = val_a - val_b
            else:
                f[f'interface_{p}_diff'] = np.nan

        # --- D. Ratios (Above / Below) ---
        # Based on feature_names.json
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

        # --- E. Gradients (Diff / Distance) ---
        # Based on feature_names.json
        grad_params = ['density', 'temperature', 'lwc', 'grain_size']
        if pd.notna(dist) and dist > 0:
            for p in grad_params:
                diff = f.get(f'interface_{p}_diff', np.nan)
                if pd.notna(diff):
                    f[f'interface_{p}_gradient'] = diff / dist
                else:
                    f[f'interface_{p}_gradient'] = np.nan
                
        return f


# ---------------------------------------------------------------------------
# Factory and Helper Functions (Required by main.py)
# ---------------------------------------------------------------------------

def create_hybrid_loc_detector(
    model_path: Optional[Path] = None,
    use_ml_primary: bool = True,
    ml_threshold: float = 0.5,
    rule_based_fallback: Callable = None
) -> Callable:
    """
    Create a hybrid LOC detection function combining ML and rule-based approaches.
    
    Args:
        model_path: Path to trained model (None = use rule-based only)
        use_ml_primary: If True, try ML first, else use rule-based first
        ml_threshold: Probability threshold for ML predictions
        rule_based_fallback: Rule-based detection function (e.g., find_wet_slab_loc)
    
    Returns:
        Function with signature: (df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]
    """
    # Initialize ML detector if model path provided
    ml_detector = None
    if model_path is not None and model_path.exists():
        try:
            ml_detector = MLLocDetector(model_path, ml_threshold)
            logger.info(f"Hybrid detector initialized with ML model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}. Using rule-based only.")
    
    def hybrid_detect(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Hybrid LOC detection combining ML and rule-based approaches.
        """
        # Primary method
        if use_ml_primary and ml_detector is not None:
            height, score = ml_detector.find_ml_loc(df)
            if height is not None:
                return height, score
        elif rule_based_fallback is not None:
            # Rule based functions might return a dict or a tuple depending on implementation
            # Standardizing on tuple (height, score) if possible, or parsing the dict
            res = rule_based_fallback(df)
            if isinstance(res, dict) and 'loc_depth' in res:
                # Convert depth to height if needed, or just return what we have
                # Assuming rule_based returns {loc_height: ..., score: ...}
                return res.get('loc_height'), 1.0 # Dummy score for rule-based
            elif isinstance(res, tuple):
                return res
            elif res is not None:
                # Assume it returned a height directly
                return res, 1.0
        
        # Fallback method (Swap priority)
        if not use_ml_primary and ml_detector is not None:
            height, score = ml_detector.find_ml_loc(df)
            if height is not None:
                return height, score
        elif use_ml_primary and rule_based_fallback is not None:
             # Same rule-based handling as above
            res = rule_based_fallback(df)
            if isinstance(res, dict) and 'loc_height' in res:
                return res.get('loc_height'), 1.0
            elif isinstance(res, tuple):
                return res
        
        return None, None
    
    return hybrid_detect


def find_ml_loc_simple(df: pd.DataFrame, model_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Wrapper for quick one-off detection."""
    detector = MLLocDetector(model_path)
    return detector.find_ml_loc(df)