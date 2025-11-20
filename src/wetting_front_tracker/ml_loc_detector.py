"""
ml_loc_detector.py
==================

ML-based Layer of Concern (LOC) detection for wet slab avalanches.

This module provides an alternative to rule-based LOC detection by using
a trained machine learning model to predict wetting front stalls at layer
interfaces. It extracts features from snowpack profiles and uses the trained
model to identify likely stall locations.

Usage:
    # Initialize detector with trained model
    detector = MLLocDetector(model_path=Path("results/trained_model"))
    
    # Find LOC in a daily profile
    loc_height, stall_probability = detector.find_ml_loc(profile_df)
    
    # Or get predictions for all interfaces
    predictions_df = detector.predict_all_interfaces(profile_df)

Author: Ron Simenhois
Created: November 2025
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

from .ml_training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class MLLocDetector:
    """
    Machine learning-based LOC detector.
    
    Uses a trained XGBoost model to predict wetting front stall probability
    at each layer interface in the snowpack.
    
    Attributes:
        trainer: Loaded ModelTrainer with trained model
        model_path: Path to trained model directory
        probability_threshold: Minimum probability to consider as LOC (default 0.5)
        feature_names: List of feature names required by the model
    """
    
    def __init__(
        self, 
        model_path: Path,
        probability_threshold: float = 0.5
    ):
        """
        Initialize the ML LOC detector.
        
        Args:
            model_path: Path to directory containing trained model
            probability_threshold: Minimum stall probability to consider as LOC
        
        Raises:
            FileNotFoundError: If model not found at model_path
            ValueError: If threshold not in [0, 1]
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not 0 <= probability_threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {probability_threshold}")
        
        self.model_path = model_path
        self.probability_threshold = probability_threshold
        
        # Load the trained model
        logger.info(f"Loading trained model from {model_path}")
        self.trainer = ModelTrainer.load_model(model_path)
        self.feature_names = self.trainer.feature_names_
        
        logger.info(
            f"ML LOC detector initialized with {len(self.feature_names)} features "
            f"and threshold={probability_threshold}"
        )
    
    def _extract_interface_features(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Extract features for all interfaces in a snowpack profile.
        
        This creates a feature vector for each layer interface that can
        be fed to the ML model for stall prediction.
        
        Args:
            df: DataFrame with snowpack layer data (single timestamp)
            lookback_hours: Hours of history to include in features
        
        Returns:
            DataFrame with one row per interface and columns matching
            the model's required features
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame()
        
        # Required columns for interface analysis
        required_cols = [
            'height', 'density', 'temperature', 'lwc', 
            'grain_size', 'grain_type'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Sort by height to ensure proper layer ordering
        df_sorted = df.sort_values('height').reset_index(drop=True)
        
        # Initialize list to store interface features
        interface_features = []
        
        # Process each interface (between consecutive layers)
        for i in range(len(df_sorted) - 1):
            below = df_sorted.iloc[i]
            above = df_sorted.iloc[i + 1]
            
            # Create feature dictionary for this interface
            features = self._create_interface_feature_dict(below, above)
            
            # Add metadata
            features['interface_index'] = i
            features['below_height'] = below['height']
            features['above_height'] = above['height']
            features['lookback_hours'] = lookback_hours
            
            interface_features.append(features)
        
        return pd.DataFrame(interface_features)
    
    def _create_interface_feature_dict(
        self,
        below: pd.Series,
        above: pd.Series
    ) -> dict:
        """
        Create feature dictionary for a single interface.
        
        Extracts features from the layer below and above an interface,
        plus interface properties like differences and ratios.
        
        Args:
            below: Series with properties of layer below interface
            above: Series with properties of layer above interface
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Layer properties (below)
        below_features = [
            'height', 'density', 'temperature', 'lwc',
            'grain_size', 'grain_type'
        ]
        for feat in below_features:
            if feat in below.index:
                features[f'below_{feat}'] = below[feat]
        
        # Layer properties (above)
        above_features = [
            'height', 'density', 'temperature', 'lwc',
            'grain_size', 'grain_type'
        ]
        for feat in above_features:
            if feat in above.index:
                features[f'above_{feat}'] = above[feat]
        
        # Interface differences
        for feat in ['density', 'temperature', 'lwc', 'grain_size']:
            if feat in above.index and feat in below.index:
                features[f'interface_{feat}_diff'] = above[feat] - below[feat]
        
        # Interface ratios (avoid division by zero)
        for feat in ['density', 'grain_size']:
            if feat in above.index and feat in below.index:
                if below[feat] > 0:
                    features[f'interface_{feat}_ratio'] = above[feat] / below[feat]
                else:
                    features[f'interface_{feat}_ratio'] = np.nan
        
        # Interface gradients (per unit height)
        height_diff = above['height'] - below['height']
        if height_diff > 0:
            for feat in ['density', 'temperature', 'lwc', 'grain_size']:
                if feat in above.index and feat in below.index:
                    value_diff = above[feat] - below[feat]
                    features[f'interface_{feat}_gradient'] = value_diff / height_diff
        
        # Special features
        # Capillary barrier indicator (smaller grains above larger grains)
        if 'grain_size' in above.index and 'grain_size' in below.index:
            features['is_capillary_barrier'] = (
                above['grain_size'] < below['grain_size']
            )
        
        # Weak layer indicator (FC/DH below)
        if 'grain_type' in below.index:
            features['below_is_fc_dh'] = (
                400 <= below['grain_type'] < 600
            )
        
        return features
    
    def _prepare_features_for_model(
        self,
        interface_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare interface features to match model's expected input.
        
        Ensures all required features are present and in the correct order.
        Missing features are filled with NaN and will be handled by the model.
        
        Based on SHAP importance analysis:
        - Critical features (>1.0): interface_layer_distance, interface_stress_ratio, above_stress
        - High importance (0.5-1.0): lwc features, temperature gradients
        - Moderate (0.3-0.5): grain_size, bond_size, sphericity, viscosity
        
        Args:
            interface_df: DataFrame with extracted interface features
        
        Returns:
            DataFrame with features in correct order for model
        """
        # Create DataFrame with all required features
        model_features = pd.DataFrame(index=interface_df.index)
        
        # Track missing features by importance tier
        critical_missing = []  # SHAP > 1.0
        high_missing = []      # SHAP 0.5-1.0
        moderate_missing = []  # SHAP 0.3-0.5
        
        for feature_name in self.feature_names:
            if feature_name in interface_df.columns:
                model_features[feature_name] = interface_df[feature_name]
            else:
                # Feature not available - use NaN
                # XGBoost handles missing values internally
                model_features[feature_name] = np.nan
                
                # Classify missing feature by importance
                if any(f in feature_name for f in ['stress_ratio', 'above_stress', 'layer_distance']):
                    critical_missing.append(feature_name)
                elif any(f in feature_name for f in ['lwc', 'temperature_gradient']):
                    high_missing.append(feature_name)
                elif any(f in feature_name for f in ['grain_size', 'bond_size', 'sphericity', 'viscosity']):
                    moderate_missing.append(feature_name)
                else:
                    logger.debug(f"Low-importance feature '{feature_name}' not found")
        
        # Warn about missing important features
        if critical_missing:
            logger.warning(
                f"CRITICAL features missing (SHAP>1.0): {critical_missing}\n"
                f"Model performance will be significantly degraded.\n"
                f"Ensure 'stress' and 'height' are extracted from SNOWPACK data."
            )
        
        if high_missing:
            logger.warning(
                f"High-importance features missing (SHAP 0.5-1.0): {high_missing}\n"
                f"Model performance may be reduced by ~10-15%."
            )
        
        if moderate_missing:
            logger.info(
                f"Moderate-importance features missing (SHAP 0.3-0.5): {len(moderate_missing)} features\n"
                f"Performance impact: ~5-10% reduction (acceptable)"
            )
        
        return model_features
    
    def predict_all_interfaces(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Predict stall probability for all interfaces in a profile.
        
        Args:
            df: DataFrame with snowpack layer data
            lookback_hours: Hours of history to consider
        
        Returns:
            DataFrame with predictions for each interface:
                - interface_index: Index of interface (0 = bottom)
                - below_height: Height of layer below interface (m)
                - above_height: Height of layer above interface (m)
                - prediction: Binary prediction (0=no stall, 1=stall)
                - stall_probability: Probability of stall [0, 1]
                - is_loc: Whether this interface is the predicted LOC
        """
        # Extract features for all interfaces
        interface_df = self._extract_interface_features(df, lookback_hours)
        
        if interface_df.empty:
            logger.warning("No valid interfaces found in profile")
            return pd.DataFrame()
        
        # Prepare features for model
        X = self._prepare_features_for_model(interface_df)
        
        # Make predictions
        predictions = self.trainer.predict(X)
        probabilities = self.trainer.predict_proba(X)
        
        # Add predictions to results
        results = interface_df[
            ['interface_index', 'below_height', 'above_height']
        ].copy()
        results['prediction'] = predictions
        results['stall_probability'] = probabilities[:, 1]  # Prob of class 1 (stall)
        
        # Identify the most likely LOC (highest stall probability above threshold)
        above_threshold = results['stall_probability'] >= self.probability_threshold
        if above_threshold.any():
            loc_idx = results.loc[above_threshold, 'stall_probability'].idxmax()
            results['is_loc'] = False
            results.loc[loc_idx, 'is_loc'] = True
        else:
            results['is_loc'] = False
        
        return results
    
    def find_ml_loc(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 24
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Find the Layer of Concern using ML predictions.
        
        This is the main interface that matches the signature of the
        rule-based LOC detection functions. It returns the height and
        stall probability of the most likely LOC.
        
        Args:
            df: DataFrame with snowpack layer data (single timestamp)
            lookback_hours: Hours of history to consider
        
        Returns:
            Tuple of (loc_height, stall_probability) or (None, None) if
            no LOC is detected above the threshold
        
        Examples:
            >>> detector = MLLocDetector(model_path)
            >>> loc_height, prob = detector.find_ml_loc(profile_df)
            >>> if loc_height is not None:
            ...     print(f"LOC at {loc_height}m with probability {prob:.2f}")
        """
        # Get predictions for all interfaces
        predictions_df = self.predict_all_interfaces(df, lookback_hours)
        
        if predictions_df.empty:
            return None, None
        
        # Find the predicted LOC
        loc_rows = predictions_df[predictions_df['is_loc']]
        
        if loc_rows.empty:
            logger.debug("No LOC detected above threshold")
            return None, None
        
        # Return the height of the layer below the LOC interface
        # (since the LOC is the weak layer that water might stall above)
        loc = loc_rows.iloc[0]
        loc_height = float(loc['below_height'])
        stall_probability = float(loc['stall_probability'])
        
        logger.debug(
            f"ML LOC detected at {loc_height:.2f}m "
            f"(probability={stall_probability:.3f})"
        )
        
        return loc_height, stall_probability


def create_hybrid_loc_detector(
    model_path: Optional[Path] = None,
    use_ml_primary: bool = True,
    ml_threshold: float = 0.5,
    rule_based_fallback: callable = None
) -> callable:
    """
    Create a hybrid LOC detection function combining ML and rule-based approaches.
    
    This factory function creates a LOC detection function that:
    1. Tries ML prediction first (if model available and use_ml_primary=True)
    2. Falls back to rule-based if ML fails or returns None
    3. Can be used as a drop-in replacement for rule-based functions
    
    Args:
        model_path: Path to trained model (None = use rule-based only)
        use_ml_primary: If True, try ML first, else use rule-based first
        ml_threshold: Probability threshold for ML predictions
        rule_based_fallback: Rule-based detection function (e.g., find_wet_slab_loc)
    
    Returns:
        Function with signature: (df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]
    
    Examples:
        >>> from wet_front_tracker import find_wet_slab_loc
        >>> 
        >>> # Create hybrid detector
        >>> hybrid_loc = create_hybrid_loc_detector(
        ...     model_path=Path("results/trained_model"),
        ...     rule_based_fallback=find_wet_slab_loc
        ... )
        >>> 
        >>> # Use it like any LOC detection function
        >>> loc_height, score = hybrid_loc(profile_df)
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
        
        Args:
            df: DataFrame with snowpack layer data
        
        Returns:
            Tuple of (height, score) where score is stall probability (ML)
            or gs_difference (rule-based)
        """
        # Primary method
        if use_ml_primary and ml_detector is not None:
            height, score = ml_detector.find_ml_loc(df)
            if height is not None:
                return height, score
        elif rule_based_fallback is not None:
            height, score = rule_based_fallback(df)
            if height is not None:
                return height, score
        
        # Fallback method
        if not use_ml_primary and ml_detector is not None:
            height, score = ml_detector.find_ml_loc(df)
            if height is not None:
                return height, score
        elif use_ml_primary and rule_based_fallback is not None:
            height, score = rule_based_fallback(df)
            if height is not None:
                return height, score
        
        # Both methods failed
        return None, None
    
    return hybrid_detect


# Convenience function for quick ML-based LOC detection
def find_ml_loc_simple(
    df: pd.DataFrame,
    model_path: Path,
    threshold: float = 0.5
) -> Tuple[Optional[float], Optional[float]]:
    """
    Simple wrapper for one-off ML LOC detection without persistent detector.
    
    Note: This creates a new detector each time. For repeated use,
    create a persistent MLLocDetector instance instead.
    
    Args:
        df: DataFrame with snowpack layer data
        model_path: Path to trained model
        threshold: Probability threshold
    
    Returns:
        Tuple of (loc_height, stall_probability) or (None, None)
    """
    detector = MLLocDetector(model_path, threshold)
    return detector.find_ml_loc(df)