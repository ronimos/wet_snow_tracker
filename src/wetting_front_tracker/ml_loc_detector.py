"""
ml_loc_detector.py
===================

Machine Learning-based detection of Layer of Concern (LOC) depths/heights.

FINAL VERSION: Properly aligned with trained model features.

Author: Ron Simenhois
Created: November 2025
Last Updated: November 2025
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable, Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class MLLocDetector:
    """
    Machine Learning-based Layer of Concern (LOC) Detector.
    
    Loads a trained model and predicts stall probabilities for layer interfaces.
    Feature extraction is aligned with the training pipeline.
    """

    def __init__(
        self, 
        model_path: Union[str, Path], 
        probability_threshold: float = 0.5
    ):
        """
        Initialize the ML-based LOC detector.

        Args:
            model_path: Path to directory containing trained model artifacts
            probability_threshold: Minimum probability to classify as LOC (0.0-1.0)
        """
        if not 0 <= probability_threshold <= 1:
            raise ValueError(f"probability_threshold must be in [0,1], got {probability_threshold}")
            
        self.model_path = Path(model_path)
        self.threshold = probability_threshold
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.imputer = None
        
        self._load_model()

    def _load_model(self) -> None:
        """Load model artifacts from disk."""
        try:
            if self.model_path.is_file():
                # Direct path to model file
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                
                # Try to load artifacts from same directory
                parent_dir = self.model_path.parent
                self._load_artifacts(parent_dir)
                    
            else:
                # Directory structure
                model_file = self.model_path / "model.joblib"
                
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found at {model_file}")
                
                self.model = joblib.load(model_file)
                logger.info(f"Loaded model from {model_file}")
                
                self._load_artifacts(self.model_path)
                    
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None
            raise

    def _load_artifacts(self, artifact_dir: Path) -> None:
        """Load supporting artifacts (scaler, imputer, feature names)."""
        
        # Try loading feature_names.json first, then .joblib
        feat_json = artifact_dir / "feature_names.json"
        feat_joblib = artifact_dir / "feature_names.joblib"
        
        if feat_json.exists():
            with open(feat_json, 'r') as f:
                self.feature_names = json.load(f)
            logger.debug(f"Loaded {len(self.feature_names)} feature names from JSON")
        elif feat_joblib.exists():
            self.feature_names = joblib.load(feat_joblib)
            logger.debug(f"Loaded {len(self.feature_names)} feature names from joblib")
        
        # Load scaler
        scaler_file = artifact_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.debug("Loaded feature scaler")
        
        # Load imputer
        imputer_file = artifact_dir / "imputer.joblib"
        if imputer_file.exists():
            self.imputer = joblib.load(imputer_file)
            logger.debug("Loaded feature imputer")

    def _extract_interface_features(
        self, 
        df: pd.DataFrame
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Extract features from layer interfaces matching training pipeline.
        
        Args:
            df: DataFrame with one row per layer, sorted by height
            
        Returns:
            Tuple of (feature_dicts, interface_heights)
        """
        interfaces = []
        interface_heights = []
        
        # Ensure sorted by height (ground to surface)
        df = df.sort_values('height', ascending=True).reset_index(drop=True)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'height']
        
        if not feature_cols:
            logger.warning("No numeric feature columns found")
            return [], []
        
        logger.debug(f"Extracting features from {len(feature_cols)} columns")

        # Process each pair of adjacent layers
        for i in range(len(df) - 1):
            below = df.iloc[i]
            above = df.iloc[i + 1]
            
            feat = {}
            
            # Stall height (interface midpoint)
            interface_height = (above['height'] + below['height']) / 2.0
            feat['stall_height'] = interface_height
            
            # Layer distance
            feat['interface_layer_distance'] = above['height'] - below['height']
            
            # Extract above/below features for each column
            for col in feature_cols:
                val_above = float(above[col]) if pd.notna(above[col]) else np.nan
                val_below = float(below[col]) if pd.notna(below[col]) else np.nan
                
                # Above and below raw values
                feat[f'above_{col}'] = val_above
                feat[f'below_{col}'] = val_below
                
                # Differences
                if not (np.isnan(val_above) or np.isnan(val_below)):
                    feat[f'interface_{col}_diff'] = val_above - val_below
                else:
                    feat[f'interface_{col}_diff'] = np.nan
                
                # Ratios (avoid division by zero)
                if not np.isnan(val_below) and val_below != 0:
                    feat[f'interface_{col}_ratio'] = val_above / val_below
                else:
                    feat[f'interface_{col}_ratio'] = np.nan
                
                # Gradients (per meter)
                layer_distance = feat['interface_layer_distance']
                if layer_distance > 0 and not (np.isnan(val_above) or np.isnan(val_below)):
                    feat[f'interface_{col}_gradient'] = (val_above - val_below) / layer_distance
                else:
                    feat[f'interface_{col}_gradient'] = np.nan
            
            interfaces.append(feat)
            interface_heights.append(interface_height)

        logger.debug(f"Extracted {len(interfaces)} interfaces")
        return interfaces, interface_heights

    def find_ml_loc(
        self, 
        df: pd.DataFrame, 
        top_n: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Identify potential Layers of Concern using ML.
        
        Args:
            df: DataFrame representing snowpack profile at single timestamp
            top_n: Maximum number of candidates to return
            
        Returns:
            List of (height_m, probability) tuples
        """
        if self.model is None or df.empty or len(df) < 2:
            return []

        try:
            # 1. Extract interface features
            interfaces, interface_heights = self._extract_interface_features(df)
            
            if not interfaces:
                return []

            # 2. Prepare features DataFrame
            X_pred = pd.DataFrame(interfaces)
            
            # 3. Align with training features
            if self.feature_names is not None:
                # Add missing columns with NaN
                missing_cols = set(self.feature_names) - set(X_pred.columns)
                for col in missing_cols:
                    X_pred[col] = np.nan
                
                # Select and reorder columns to match training
                X_pred = X_pred[self.feature_names]
                
                # Apply imputer if available
                if self.imputer is not None:
                    X_imputed = self.imputer.transform(X_pred)
                    X_pred = pd.DataFrame(X_imputed, columns=self.feature_names)
                
                # Apply scaler if available
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X_pred)
                    X_pred = pd.DataFrame(X_scaled, columns=self.feature_names)
            
            # 4. Predict probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_pred)[:, 1]
            else:
                probs = self.model.predict(X_pred)

            # 5. Filter and rank candidates
            candidates = []
            for height, prob in zip(interface_heights, probs):
                if prob >= self.threshold:
                    candidates.append((height, float(prob)))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            if candidates:
                logger.debug(f"Found {len(candidates)} LOC candidates above threshold {self.threshold}")
            
            return candidates[:top_n]

        except Exception as e:
            logger.warning(f"Error during ML LOC inference: {e}", exc_info=True)
            return []

    def detect_timeseries(
        self,
        xr_data: xr.Dataset,
        top_n: int = 5,
        return_all_candidates: bool = False
    ) -> pd.DataFrame:
        """
        Detect LOCs across all timestamps in a profile timeseries.
        
        Args:
            xr_data: xarray Dataset with dimensions (timestamp, height)
            top_n: For each timestamp, consider top N candidates
            return_all_candidates: If True, return all; if False, only top per timestamp
                
        Returns:
            DataFrame with columns: [timestamp, loc_height, stall_probability, rank]
        """
        if self.model is None:
            logger.error("Model not loaded")
            return pd.DataFrame()
            
        try:
            results = []
            timestamps = pd.to_datetime(xr_data.timestamp.values)
            
            logger.info(f"Processing {len(timestamps)} timestamps")
            
            for ts in timestamps:
                # Extract profile at this timestamp
                profile_data = xr_data.sel(timestamp=ts)
                
                # Convert to DataFrame
                if 'height' in profile_data.dims:
                    df = profile_data.to_dataframe().reset_index()
                else:
                    df = profile_data.to_dataframe().reset_index()
                
                # CRITICAL: Drop timestamp column if present
                if 'timestamp' in df.columns:
                    df = df.drop(columns=['timestamp'])
                
                # Remove NaN-only rows
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df = df.dropna(how='all', subset=numeric_cols)
                
                if df.empty or len(df) < 2:
                    continue
                
                # Detect LOCs
                candidates = self.find_ml_loc(df, top_n=top_n)
                
                # Store results
                for rank, (height, prob) in enumerate(candidates, start=1):
                    results.append({
                        'timestamp': ts,
                        'loc_height': height,
                        'stall_probability': prob,
                        'rank': rank
                    })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            if results_df.empty:
                logger.warning("No LOCs detected in timeseries")
                return results_df
            
            # Set timestamp as index
            results_df = results_df.set_index('timestamp')
            
            # Filter to top candidate only if requested
            if not return_all_candidates:
                results_df = results_df[results_df['rank'] == 1]
            
            logger.info(f"Detected LOCs at {len(results_df.index.unique())} timestamps")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in timeseries detection: {e}", exc_info=True)
            return pd.DataFrame()


class HybridLocDetector:
    """Hybrid detector combining ML and rule-based approaches."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        use_ml_primary: bool,
        ml_threshold: float,
        rule_based_fallback: Optional[Callable],
        top_n: int
    ):
        """Initialize hybrid detector."""
        self.model_path = model_path
        self.use_ml_primary = use_ml_primary
        self.ml_threshold = ml_threshold
        self.rule_based_fallback = rule_based_fallback
        self.top_n = top_n
        self._ml_detector = None

    @property
    def ml_detector(self) -> MLLocDetector:
        """Lazy-load ML detector."""
        if self._ml_detector is None:
            self._ml_detector = MLLocDetector(self.model_path, self.ml_threshold)
        return self._ml_detector

    def __call__(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Detect LOCs using hybrid approach."""
        candidates = []
        
        # Try ML first
        if self.use_ml_primary or self.rule_based_fallback is None:
            try:
                candidates = self.ml_detector.find_ml_loc(df, self.top_n)
            except Exception as e:
                logger.warning(f"ML detection failed: {e}")
        
        # Fallback to rule-based
        if not candidates and self.rule_based_fallback is not None:
            try:
                res = self.rule_based_fallback(df)
                
                if isinstance(res, list):
                    candidates = res
                elif isinstance(res, tuple):
                    candidates = [res]
                elif isinstance(res, dict):
                    if 'loc_height' in res:
                        candidates = [(res['loc_height'], 1.0)]
                    elif 'loc_depth' in res and 'height' in df.columns:
                        hs = df['height'].max()
                        if pd.notna(hs):
                            candidates = [(hs - res['loc_depth'], 1.0)]
                elif res is not None:
                    try:
                        candidates = [(float(res), 1.0)]
                    except (TypeError, ValueError):
                        pass
                        
            except Exception as e:
                logger.warning(f"Rule-based fallback failed: {e}")
                    
        return candidates


def create_hybrid_loc_detector(
    model_path: Union[str, Path],
    use_ml_primary: bool = True,
    ml_threshold: float = 0.5,
    rule_based_fallback: Optional[Callable] = None,
    top_n: int = 5
) -> HybridLocDetector:
    """Factory function for hybrid detector."""
    return HybridLocDetector(
        model_path=model_path,
        use_ml_primary=use_ml_primary,
        ml_threshold=ml_threshold,
        rule_based_fallback=rule_based_fallback,
        top_n=top_n
    )