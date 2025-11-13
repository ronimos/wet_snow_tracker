"""
feature_extractor.py
=======================

Extracts comprehensive SNOWPACK layer parameters for ML training.

This version:
- Extracts features from specific layers by ID (not height ranges)
- Collects data 24 hours BEFORE stalling occurs
- Extracts ALL available SNOWPACK parameters
- Computes interface differences and ratios

Author: Ron Simenhois
Created: November 2025
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import parameter configuration
try:
    from wetting_front_tracker.param_config import (
        SNOWPACK_PARAMETERS,
        get_parameters_for_differences,
        get_parameters_for_ratios,
        get_column_name
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from wetting_front_tracker.param_config import (
        SNOWPACK_PARAMETERS,
        get_parameters_for_differences,
        get_parameters_for_ratios,
        get_column_name
    )

# Import SnowpackProfile with proper type checking support
if TYPE_CHECKING:
    # For type checkers, always import the real type
    from ..snowpack_reader import SnowpackProfile as SnowpackProfileType
else:
    # At runtime, try to import and fall back gracefully
    try:
        from ..snowpack_reader import SnowpackProfile as SnowpackProfileType
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from ..snowpack_reader import SnowpackProfile as SnowpackProfileType
        except ImportError:
            logger.warning("Could not import SnowpackProfile - will fail at runtime if used")
            SnowpackProfileType = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    # Lookback time for feature extraction
    lookback_hours: float = 24.0
    
    # Which parameter groups to extract
    extract_all_parameters: bool = True
    
    # Fallback options if exact time not available
    max_time_tolerance_hours: float = 2.0
    
    # Whether to compute derived features
    compute_differences: bool = True
    compute_ratios: bool = True
    compute_gradients: bool = True


# ---------------------------------------------------------------------------
# Layer Feature Extraction
# ---------------------------------------------------------------------------

class LayerFeatureExtractor:
    """
    Extracts features from specific SNOWPACK layers by ID.
    
    This class handles:
    - Finding layers at specific times
    - Extracting all available SNOWPACK parameters
    - Computing interface properties (differences, ratios, gradients)
    - Handling missing data gracefully
    """
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """
        Initialize extractor with configuration.
        
        Args:
            config: Feature extraction parameters
        """
        self.config = config or FeatureExtractionConfig()
    
    def extract_features_for_interface(
        self,
        profile: "SnowpackProfileType",
        feature_time: datetime,
        stall_layer_id: int,  # Kept for compatibility, but unused
        layer_above_id: int,
        layer_below_id: int
    ) -> Dict[str, float]:
        """
        Extract all features for a specific interface at a specific time.
        
        Args:
            profile: SnowpackProfile object
            feature_time: When to extract features (e.g., 24h before stall)
            stall_layer_id: Unused, kept for compatibility with caller.
            layer_above_id: Layer ID above the interface
            layer_below_id: Layer ID below the interface
            
        Returns:
            Dictionary of feature_name: value pairs
        """
        features = {}
        
        # Get profile at feature extraction time
        profile_df, actual_time = self._get_profile_at_time(
            profile, feature_time
        )
        
        if profile_df is None or profile_df.empty:
            logger.warning(f"No profile data available at {feature_time}")
            return features

        # Check if actual_time is None
        if actual_time is None:
            logger.warning(f"No valid timestamp returned for {feature_time}")
            return features
        
        # Record actual lookback time
        # Note: stall_time is not available here, so we record actual time.
        # The caller (collect_ml_data) will compute the actual lookback.
        features['feature_extraction_time'] = actual_time.isoformat()
        
        # Extract layer above features
        layer_above = self._get_layer_by_id(profile_df, layer_above_id)
        if layer_above is not None:
            above_features = self._extract_layer_parameters(
                layer_above, prefix='above'
            )
            features.update(above_features)
        else:
            logger.warning(f"Layer above (ID={layer_above_id}) not found at {actual_time}")
        
        # Extract layer below features
        layer_below = self._get_layer_by_id(profile_df, layer_below_id)
        if layer_below is not None:
            below_features = self._extract_layer_parameters(
                layer_below, prefix='below'
            )
            features.update(below_features)
        else:
            logger.warning(f"Layer below (ID={layer_below_id}) not found at {actual_time}")
        
        # Compute interface features if both layers available
        if layer_above is not None and layer_below is not None:
            interface_features = self._compute_interface_features(
                layer_above, layer_below
            )
            features.update(interface_features)
        
        # Add lookback hours (relative to feature_time, not stall_time)
        # The caller computes the *actual* lookback from stall time
        features['requested_lookback_hours'] = self.config.lookback_hours
        
        return features
    
    def _get_profile_at_time(
        self,
        profile: "SnowpackProfileType",
        target_time: datetime
    ) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
        """
        Get profile DataFrame at a specific time.
        
        Args:
            profile: SnowpackProfile object
            target_time: Desired time
            
        Returns:
            Tuple of (profile_df, actual_time)
            Returns (None, None) if time not available
        """
        try:
            # Try to get profile at exact time (with nearest neighbor)
            # Convert tolerance to numpy timedelta64 for xarray compatibility
            tolerance_td = np.timedelta64(
                int(self.config.max_time_tolerance_hours * 3600), 's'
            )
            profile_at_time = profile.data.sel(
                timestamp=target_time,
                method='nearest',
                tolerance=tolerance_td # type: ignore[arg-type]
            )
            
            # Get actual timestamp
            actual_time = pd.Timestamp(profile_at_time.timestamp.values)
            
            # Convert to DataFrame
            profile_df = profile_at_time.to_dataframe().reset_index()
            
            return profile_df, actual_time
            
        except Exception as e:
            logger.error(f"Error getting profile at {target_time}: {e}")
            return None, None
    
    def _get_layer_by_id(
        self,
        profile_df: pd.DataFrame,
        layer_id: int
    ) -> Optional[pd.Series]:
        """
        Extract a specific layer by its element_ID.
        
        Args:
            profile_df: DataFrame with layer data
            layer_id: Element ID to find
            
        Returns:
            Series representing the layer, or None if not found
        """
        if 'element_ID' not in profile_df.columns:
            logger.warning("No element_ID column in profile")
            return None
        
        # Find layer with matching ID
        layer_mask = profile_df['element_ID'] == layer_id
        matching_layers = profile_df[layer_mask]
        
        if len(matching_layers) == 0:
            logger.debug(f"Layer ID {layer_id} not found in profile")
            return None
        
        if len(matching_layers) > 1:
            logger.warning(f"Multiple layers with ID {layer_id}, using first")
        
        return matching_layers.iloc[0]
    
    def _extract_layer_parameters(
        self,
        layer: pd.Series,
        prefix: str = 'layer'
    ) -> Dict[str, float]:
        """
        Extract all available SNOWPACK parameters from a layer.
        
        Args:
            layer: Series representing a single layer
            prefix: Prefix for feature names ('above', 'below', etc.)
            
        Returns:
            Dictionary of parameter features
        """
        features = {}
        
        # Extract each parameter
        for code, param_def in SNOWPACK_PARAMETERS.items():
            col_name = param_def.column_name
            
            if col_name in layer.index:
                value = layer[col_name]
                
                # Ensure we have a scalar value (handle Series edge cases)
                if isinstance(value, pd.Series):
                    value = value.item()
                
                # Handle NaN and infinite values
                if pd.isna(value) or np.isinf(value):
                    value = np.nan
                else:
                    value = float(value)
                
                feature_name = f'{prefix}_{param_def.name}'
                features[feature_name] = value
            else:
                # Parameter not available in this profile
                feature_name = f'{prefix}_{param_def.name}'
                features[feature_name] = np.nan
        
        return features
    
    def _compute_interface_features(
        self,
        layer_above: pd.Series,
        layer_below: pd.Series
    ) -> Dict[str, float]:
        """
        Compute interface features from two adjacent layers.
        
        Computes:
        - Differences (above - below)
        - Ratios (above / below)
        - Gradients (diff / distance)
        
        Args:
            layer_above: Series for layer above interface
            layer_below: Series for layer below interface
            
        Returns:
            Dictionary of interface features
        """
        features = {}
        
        # Get height difference for gradient calculations
        height_diff = None
        if 'height' in layer_above.index and 'height' in layer_below.index:
            h_above = layer_above['height']
            if isinstance(h_above, pd.Series):
                h_above = h_above.item()
            h_above = float(h_above)
            h_below = layer_below['height']
            if isinstance(h_below, pd.Series):
                h_below = h_below.item()
            h_below = float(h_below)
            height_diff = abs(h_above - h_below)
            features['interface_layer_distance'] = height_diff
        
        # Compute differences
        if self.config.compute_differences:
            diff_params = get_parameters_for_differences()
            for code in diff_params:
                param_def = SNOWPACK_PARAMETERS[code]
                col_name = param_def.column_name
                
                if col_name in layer_above.index and col_name in layer_below.index:
                    val_above = layer_above[col_name]
                    if isinstance(val_above, pd.Series):
                        val_above = val_above.item()
                    val_below = layer_below[col_name]
                    if isinstance(val_below, pd.Series):
                        val_below = val_below.item()
                    if pd.notna(val_above) and pd.notna(val_below):
                        diff = float(val_above) - float(val_below)
                        features[f'interface_{param_def.name}_diff'] = diff
                    else:
                        features[f'interface_{param_def.name}_diff'] = np.nan
        
        # Compute ratios
        if self.config.compute_ratios:
            ratio_params = get_parameters_for_ratios()
            for code in ratio_params:
                param_def = SNOWPACK_PARAMETERS[code]
                col_name = param_def.column_name
                
                if col_name in layer_above.index and col_name in layer_below.index:
                    val_above = layer_above[col_name] 
                    if isinstance(val_above, pd.Series):
                        val_above = val_above.item()
                    val_below = layer_below[col_name]
                    if isinstance(val_below, pd.Series):
                        val_below = val_below.item()
                    
                    if pd.notna(val_above) and pd.notna(val_below):
                        val_above = float(val_above)
                        val_below = float(val_below)
                        
                        # Avoid division by zero
                        if val_below != 0:
                            ratio = val_above / val_below
                            features[f'interface_{param_def.name}_ratio'] = ratio
                        else:
                            features[f'interface_{param_def.name}_ratio'] = np.nan
                    else:
                        features[f'interface_{param_def.name}_ratio'] = np.nan
        
        # Compute gradients for key parameters
        if self.config.compute_gradients and height_diff is not None and height_diff > 0:
            gradient_params = ['0502', '0503', '0506', '0512']  # density, temp, lwc, grain_size
            
            for code in gradient_params:
                if code in SNOWPACK_PARAMETERS:
                    param_def = SNOWPACK_PARAMETERS[code]
                    col_name = param_def.column_name
                    
                    if col_name in layer_above.index and col_name in layer_below.index:
                        val_above = layer_above[col_name]
                        if isinstance(val_above, pd.Series):
                            val_above = val_above.item()
                        val_below = layer_below[col_name]
                        if isinstance(val_below, pd.Series):
                            val_below = val_below.item()    
                        
                        if pd.notna(val_above) and pd.notna(val_below):
                            diff = float(val_above) - float(val_below)
                            gradient = diff / height_diff
                            features[f'interface_{param_def.name}_gradient'] = gradient
                        else:
                            features[f'interface_{param_def.name}_gradient'] = np.nan
        
        return features


# ---------------------------------------------------------------------------
# Batch Feature Extraction
# ---------------------------------------------------------------------------

def extract_features_for_stall_events(
    stall_events: List[Dict[str, Any]],
    extractor: LayerFeatureExtractor
) -> pd.DataFrame:
    """
    Extract features for a list of stall events.
    
    Args:
        stall_events: List of stall event dictionaries
        extractor: LayerFeatureExtractor instance
        
    Returns:
        DataFrame with stall events and their features
    """
    if SnowpackProfileType is None:
        raise ImportError("SnowpackProfile not available")
    
    all_features = []
    
    for event in stall_events:
        try:
            # Load profile
            profile = SnowpackProfileType(str(event['pro_file']))
            
            # Extract features
            features = extractor.extract_features_for_interface(
                profile,
                event['start_time'],
                event['stall_layer_id'],
                event['layer_above_id'],
                event['layer_below_id']
            )
            
            # Combine with event metadata
            event_features = {**event, **features}
            all_features.append(event_features)
            
        except Exception as e:
            logger.error(f"Error extracting features for {event['event_id']}: {e}")
            continue
    
    if not all_features:
        logger.warning("No features extracted")
        return pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)
    logger.info(f"Extracted features for {len(features_df)} events")
    
    return features_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate extracted features and return quality metrics.
    
    Args:
        features_df: DataFrame with extracted features
        
    Returns:
        Dictionary of validation metrics
    """
    metrics = {}
    
    # Count total features
    feature_columns = [
        col for col in features_df.columns
        if col.startswith(('above_', 'below_', 'interface_'))
    ]
    metrics['total_features'] = len(feature_columns)
    
    # Missing data analysis
    missing_counts = features_df[feature_columns].isnull().sum()
    metrics['features_with_missing'] = (missing_counts > 0).sum()
    metrics['mean_missing_pct'] = (missing_counts / len(features_df) * 100).mean()
    
    # Identify features with high missing rates
    high_missing = missing_counts[missing_counts > len(features_df) * 0.1]
    metrics['features_over_10pct_missing'] = len(high_missing)
    
    # Check lookback times
    if 'lookback_hours' in features_df.columns:
        metrics['mean_lookback_hours'] = features_df['lookback_hours'].mean()
        metrics['min_lookback_hours'] = features_df['lookback_hours'].min()
        metrics['max_lookback_hours'] = features_df['lookback_hours'].max()
    
    # Check for constant features (no variance)
    constant_features = []
    for col in feature_columns:
        if features_df[col].notna().sum() > 1:  # Need at least 2 non-null values
            if features_df[col].std(ddof=0) == 0:
                constant_features.append(col)
    metrics['constant_features'] = len(constant_features)
    
    return metrics


def print_feature_summary(features_df: pd.DataFrame):
    """Print summary of extracted features."""
    print("=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal examples: {len(features_df)}")
    if 'stalled' in features_df.columns:
        pos_count = (features_df['stalled'] == 1).sum()
        neg_count = (features_df['stalled'] == 0).sum()
        print(f"  Positive examples (stall=1): {pos_count}")
        print(f"  Negative examples (stall=0): {neg_count}")
    
    # Count feature types
    above_features = [col for col in features_df.columns if col.startswith('above_')]
    below_features = [col for col in features_df.columns if col.startswith('below_')]
    interface_features = [col for col in features_df.columns if col.startswith('interface_')]
    
    print(f"\nFeature counts:")
    print(f"  Above layer:     {len(above_features)}")
    print(f"  Below layer:     {len(below_features)}")
    print(f"  Interface:       {len(interface_features)}")
    print(f"  Total features:  {len(above_features) + len(below_features) + len(interface_features)}")
    
    # Validation metrics
    metrics = validate_features(features_df)
    
    print(f"\nData quality:")
    print(f"  Features with missing data:     {metrics['features_with_missing']}")
    print(f"  Mean missing rate:              {metrics['mean_missing_pct']:.1f}%")
    print(f"  Features >10% missing:          {metrics['features_over_10pct_missing']}")
    print(f"  Constant features (no variance): {metrics['constant_features']}")
    
    if 'mean_lookback_hours' in metrics:
        print(f"\nLookback times:")
        print(f"  Mean:    {metrics['mean_lookback_hours']:.1f} hours")
        print(f"  Min:     {metrics['min_lookback_hours']:.1f} hours")
        print(f"  Max:     {metrics['max_lookback_hours']:.1f} hours")
    
    print("=" * 80)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example of how to use the extractor
    
    logging.basicConfig(level=logging.INFO)
    
    print("LayerFeatureExtractor - Layer ID-based feature extraction")
    print("=" * 80)
    print("Extracts all SNOWPACK parameters from specific layers")
    print("24 hours before wetting front stalling occurs.")
    print("=" * 80)