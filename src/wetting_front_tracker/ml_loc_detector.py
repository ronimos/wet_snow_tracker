"""
ml_loc_detector.py
===================

Provides ML-based detection of layer of concern (LOC) depths/heights.
Used by the main workflow for inference.

Author: Ron Simenhois
Created: November 2025
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import parameter config for feature mapping
try:
    from wetting_front_tracker.param_config import SNOWPACK_PARAMETERS
except ImportError:
    # Fallback or mock if necessary
    SNOWPACK_PARAMETERS = {}

class MLLocDetector:
    """
    Machine Learning based Layer of Concern Detector.
    Loads a trained model and predicts stall probabilities for layer interfaces.
    """

    def __init__(self, model_path: Union[str, Path], probability_threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            model_path: Directory containing the trained model artifacts (model.joblib, feature_names.joblib)
            probability_threshold: Probability above which an interface is considered a LOC
        """
        self.model_path = Path(model_path)
        self.threshold = probability_threshold
        self.model = None
        self.feature_names = None
        self.scaler = None
        
        self._load_model()

    def _load_model(self):
        """Load model artifacts."""
        try:
            # Support both direct file path and directory path
            if self.model_path.is_file():
                # If pointing directly to a joblib file
                self.model = joblib.load(self.model_path)
                # Assume feature names are in the same dir
                feat_path = self.model_path.parent / "feature_names.joblib"
                if feat_path.exists():
                    self.feature_names = joblib.load(feat_path)
            else:
                # Standard directory structure from ModelTrainer.save_model
                model_file = self.model_path / "model.joblib"
                features_file = self.model_path / "feature_names.joblib"
                scaler_file = self.model_path / "scaler.joblib"
                
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found at {model_file}")
                
                self.model = joblib.load(model_file)
                
                if features_file.exists():
                    self.feature_names = joblib.load(features_file)
                
                if scaler_file.exists():
                    self.scaler = joblib.load(scaler_file)
                    
            logger.info(f"Loaded ML model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None

    def find_ml_loc(self, df: pd.DataFrame, top_n: int = 5) -> List[Tuple[float, float]]:
        """
        Identify potential layers of concern in a profile dataframe using the ML model.

        Args:
            df: DataFrame representing a snowpack profile at a single timestamp.
            top_n: Maximum number of candidates to return.

        Returns:
            List of tuples (height_m, probability) sorted by probability desc.
        """
        if self.model is None or df.empty or len(df) < 2:
            return []

        try:
            # 1. Generate interfaces and features
            # Note: This mimics logic in feature_extractor/collect_ml_data 
            # but operates on a single instantaneous dataframe (no lookback features)
            interfaces = []
            interface_indices = []
            
            # Sort by height ensures we process layers in order
            df = df.sort_values('height', ascending=True).reset_index(drop=True)

            for i in range(len(df) - 1):
                below = df.iloc[i]
                above = df.iloc[i+1]
                
                # Basic features matching what LayerFeatureExtractor produces
                feat = {}
                
                # Extract 'above' and 'below' params
                for col in df.columns:
                    # Simple mapping if column names match SNOWPACK params
                    val_a = float(above[col]) if pd.notna(above[col]) else 0.0
                    val_b = float(below[col]) if pd.notna(below[col]) else 0.0
                    
                    # Clean column name if needed (assuming simple names like 'density', 'temp')
                    # In a full implementation, this should use SNOWPACK_PARAMETERS mapping
                    feat[f'above_{col}'] = val_a
                    feat[f'below_{col}'] = val_b
                    feat[f'interface_{col}_diff'] = val_a - val_b
                    if val_b != 0:
                        feat[f'interface_{col}_ratio'] = val_a / val_b
                    else:
                        feat[f'interface_{col}_ratio'] = 0.0

                # Calculate interface height
                interface_height = (above['height'] + below['height']) / 2.0
                feat['stall_height'] = interface_height # Sometimes used as feature
                
                interfaces.append(feat)
                interface_indices.append(interface_height)

            if not interfaces:
                return []

            # 2. Prepare DataFrame for prediction
            X_pred = pd.DataFrame(interfaces)
            
            # 3. Align with model features
            if self.feature_names is not None:
                # Add missing columns with 0
                missing_cols = set(self.feature_names) - set(X_pred.columns)
                for c in missing_cols:
                    X_pred[c] = 0.0
                
                # Drop extra columns
                X_pred = X_pred[self.feature_names]
                
                # Scale if scaler exists
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X_pred)
                    X_pred = pd.DataFrame(X_scaled, columns=self.feature_names)
            
            # 4. Predict
            if hasattr(self.model, "predict_proba"):
                # Get probability of class 1 (Stall)
                probs = self.model.predict_proba(X_pred)[:, 1]
            else:
                probs = self.model.predict(X_pred)

            # 5. Filter and format results
            candidates = []
            for height, prob in zip(interface_indices, probs):
                if prob >= self.threshold:
                    candidates.append((height, float(prob)))
            
            # Sort by probability descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates[:top_n]

        except Exception as e:
            logger.warning(f"Error during ML inference: {e}")
            return []


class HybridLocDetector:
    """
    Hybrid detector strategy that uses both ML and rule-based fallback.
    Defined as a class to be picklable for multiprocessing.
    """
    def __init__(
        self,
        model_path: Union[str, Path],
        use_ml_primary: bool,
        ml_threshold: float,
        rule_based_fallback: Optional[Callable],
        top_n: int
    ):
        self.model_path = model_path
        self.use_ml_primary = use_ml_primary
        self.ml_threshold = ml_threshold
        self.rule_based_fallback = rule_based_fallback
        self.top_n = top_n
        # Don't load model here to avoid large object serialization during pickle
        self._ml_detector = None

    @property
    def ml_detector(self):
        if self._ml_detector is None:
            self._ml_detector = MLLocDetector(self.model_path, self.ml_threshold)
        return self._ml_detector

    def __call__(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        candidates = []
        
        # Attempt ML detection
        if self.use_ml_primary or self.rule_based_fallback is None:
            candidates = self.ml_detector.find_ml_loc(df, self.top_n)
        
        # Fallback logic
        if not candidates and self.rule_based_fallback is not None:
            # Rule based usually returns list of tuples or single tuple
            res = self.rule_based_fallback(df)
            if isinstance(res, list):
                candidates = res
            elif isinstance(res, tuple):
                candidates = [res]
            elif res is not None:
                # Try to parse single result
                try:
                    candidates = [(float(res), 1.0)]
                except:
                    pass
                    
        return candidates


def create_hybrid_loc_detector(
    model_path: Union[str, Path],
    use_ml_primary: bool = True,
    ml_threshold: float = 0.5,
    rule_based_fallback: Optional[Callable] = None,
    top_n: int = 5
) -> Callable:
    """
    Factory function to create a hybrid detector strategy.
    Returns a picklable class instance.
    """
    return HybridLocDetector(
        model_path=model_path,
        use_ml_primary=use_ml_primary,
        ml_threshold=ml_threshold,
        rule_based_fallback=rule_based_fallback,
        top_n=top_n
    )