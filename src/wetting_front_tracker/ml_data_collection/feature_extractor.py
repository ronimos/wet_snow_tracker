"""
feature_extractor.py
=======================

Extracts comprehensive SNOWPACK layer parameters for ML training.

This version:
- Extracts features from specific layers by ID (not height ranges)
- NEW: Uses dynamic lookback based on LWC threshold (when both layers were dry)
- Collects all available SNOWPACK parameters
- Computes interface differences and ratios

Author: Ron Simenhois
Created: November 2025
Updated: November 2025 (Dynamic LWC-based lookback)
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

import xsnow


def _squeeze(ds: xsnow.xsnowDataset):
    """Squeeze singleton dims from a single-file xsnowDataset → (time, layer)."""
    return ds.data.squeeze(['location', 'slope', 'realization'], drop=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    # NEW: Dynamic lookback based on LWC threshold
    use_dynamic_lookback: bool = True
    lwc_threshold_pct: float = 1.0      # LWC threshold in percent (1% = 0.01)
    max_lookback_hours: float = 72.0    # Don't look back more than 72h
    min_lookback_hours: float = 1.0     # At least 1 hour of history
    
    # Fallback to fixed lookback if dynamic fails
    fallback_lookback_hours: float = 24.0
    
    # DEPRECATED: Use fallback_lookback_hours instead
    lookback_hours: float = 24.0  # Kept for backwards compatibility
    
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
    - NEW: Dynamic lookback based on LWC threshold
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
    
    def find_last_dry_timestamp(
        self,
        ds: xsnow.xsnowDataset,
        stall_time: datetime,
        layer_above_id: int,
        layer_below_id: int
    ) -> Optional[pd.Timestamp]:
        """
        Find the last timestamp where both layers had LWC < threshold.

        Args:
            ds: xsnowDataset for the station
            stall_time: When the stall event started
            layer_above_id: Layer ID above the interface
            layer_below_id: Layer ID below the interface

        Returns:
            Timestamp when both layers were last dry, or None if not found
        """
        lwc_threshold = self.config.lwc_threshold_pct / 100.0

        all_times = pd.to_datetime(ds.data.time.values)
        
        # Filter to times before stall, within max lookback window
        min_time = stall_time - timedelta(hours=self.config.max_lookback_hours)
        valid_times = all_times[(all_times < stall_time) & (all_times >= min_time)]
        
        if len(valid_times) == 0:
            logger.warning(f"No valid timestamps found before {stall_time}")
            return None
        
        # Sort in reverse chronological order (most recent first)
        valid_times = sorted(valid_times, reverse=True)
        
        data = _squeeze(ds)
        for timestamp in valid_times:
            try:
                profile_at_time = data.sel(time=timestamp, method='nearest')
                profile_df = profile_at_time.to_dataframe().reset_index()
                
                # Find the layers by ID
                layer_above = profile_df[profile_df['element_ID'] == layer_above_id]
                layer_below = profile_df[profile_df['element_ID'] == layer_below_id]
                
                if layer_above.empty or layer_below.empty:
                    # Layers not found at this time, skip
                    continue
                
                # Check LWC for both layers
                lwc_col = self._get_lwc_column(profile_df)
                if lwc_col is None:
                    logger.warning("Could not find LWC column in profile")
                    return None
                
                lwc_above = layer_above[lwc_col].iloc[0]
                lwc_below = layer_below[lwc_col].iloc[0]
                
                # Check if both are below threshold
                if pd.notna(lwc_above) and pd.notna(lwc_below):
                    if lwc_above < lwc_threshold and lwc_below < lwc_threshold:
                        # Found it! Both layers are dry
                        logger.debug(
                            f"Found dry state at {timestamp}: "
                            f"above_lwc={lwc_above:.4f}, below_lwc={lwc_below:.4f}"
                        )
                        
                        # Verify minimum lookback
                        lookback_hours = (stall_time - timestamp).total_seconds() / 3600
                        if lookback_hours < self.config.min_lookback_hours:
                            # Too recent, keep searching
                            continue
                        
                        return pd.Timestamp(timestamp)
                        
            except Exception as e:
                logger.debug(f"Error checking timestamp {timestamp}: {e}")
                continue
        
        # No dry state found within the window
        logger.warning(
            f"No timestamp found where both layers had LWC < {self.config.lwc_threshold_pct}% "
            f"within {self.config.max_lookback_hours}h before {stall_time}"
        )
        return None
    
    def _get_lwc_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the LWC column name in the DataFrame.
        
        Different SNOWPACK outputs may use different names.
        """
        possible_names = [
            'lwc',
            'liquid_water_content',
            'theta_water',
            'volumetric_liquid_water_content',
            'LWC'
        ]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # Try to find it from SNOWPACK_PARAMETERS
        # Code 0506 is typically LWC
        try:
            if '0506' in SNOWPACK_PARAMETERS:
                col_name = SNOWPACK_PARAMETERS['0506'].column_name
                if col_name in df.columns:
                    return col_name
        except Exception:
            pass
        
        return None
    
    def extract_features_for_interface(
        self,
        ds: xsnow.xsnowDataset,
        stall_time: datetime,
        stall_layer_id: int,
        layer_above_id: int,
        layer_below_id: int,
        requested_feature_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Extract all features for a specific interface at a specific time.
        
        UPDATED: Now uses dynamic lookback based on LWC threshold.
        
        Args:
            profile: SnowpackProfile object
            stall_time: When the stall event started (used to determine lookback)
            stall_layer_id: Unused, kept for compatibility with caller.
            layer_above_id: Layer ID above the interface
            layer_below_id: Layer ID below the interface
            requested_feature_time: Optional explicit feature time (overrides dynamic)
            
        Returns:
            Dictionary of feature_name: value pairs
        """
        features = {}
        
        # Determine feature extraction time
        if requested_feature_time is not None:
            # Use explicitly provided time
            feature_time = requested_feature_time
            features['lookback_method'] = 'explicit'
        elif self.config.use_dynamic_lookback:
            feature_time = self.find_last_dry_timestamp(
                ds, stall_time, layer_above_id, layer_below_id
            )
            
            if feature_time is None:
                # Fallback to fixed lookback
                logger.warning(
                    f"Dynamic lookback failed, using fallback of "
                    f"{self.config.fallback_lookback_hours}h"
                )
                feature_time = stall_time - timedelta(
                    hours=self.config.fallback_lookback_hours
                )
                features['lookback_method'] = 'fallback_fixed'
            else:
                features['lookback_method'] = 'dynamic_lwc'
        else:
            # Use fixed lookback (old behavior)
            feature_time = stall_time - timedelta(
                hours=self.config.fallback_lookback_hours
            )
            features['lookback_method'] = 'fixed'
        
        profile_df, actual_time = self._get_profile_at_time(ds, feature_time)
        
        if profile_df is None or profile_df.empty:
            logger.warning(f"No profile data available at {feature_time}")
            return features

        # Check if actual_time is None
        if actual_time is None:
            logger.warning(f"No valid timestamp returned for {feature_time}")
            return features
        
        # Record actual extraction time
        features['feature_extraction_time'] = actual_time.isoformat()
        
        # Calculate actual lookback hours
        actual_lookback_hours = (stall_time - actual_time).total_seconds() / 3600
        features['lookback_hours'] = actual_lookback_hours
        
        # Extract layer above features
        layer_above = self._get_layer_by_id(profile_df, layer_above_id)
        if layer_above is not None:
            above_features = self._extract_layer_parameters(
                layer_above, prefix='above'
            )
            features.update(above_features)
            
            # NEW: Record LWC at extraction time for verification
            lwc_col = self._get_lwc_column(profile_df)
            if lwc_col and lwc_col in layer_above.index:
                features['above_lwc_at_extraction'] = float(layer_above[lwc_col])
        else:
            logger.warning(f"Layer above (ID={layer_above_id}) not found at {actual_time}")
        
        # Extract layer below features
        layer_below = self._get_layer_by_id(profile_df, layer_below_id)
        if layer_below is not None:
            below_features = self._extract_layer_parameters(
                layer_below, prefix='below'
            )
            features.update(below_features)
            
            # NEW: Record LWC at extraction time for verification
            lwc_col = self._get_lwc_column(profile_df)
            if lwc_col and lwc_col in layer_below.index:
                features['below_lwc_at_extraction'] = float(layer_below[lwc_col])
        else:
            logger.warning(f"Layer below (ID={layer_below_id}) not found at {actual_time}")
        
        # Compute interface features if both layers available
        if layer_above is not None and layer_below is not None:
            interface_features = self._compute_interface_features(
                layer_above, layer_below
            )
            features.update(interface_features)
        
        # Add requested lookback hours (for reference)
        features['requested_lookback_hours'] = self.config.fallback_lookback_hours
        
        return features
    
    def _get_profile_at_time(
        self,
        ds: xsnow.xsnowDataset,
        target_time: datetime
    ) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
        """
        Get profile DataFrame at a specific time.

        Args:
            ds: xsnowDataset for the station
            target_time: Desired time

        Returns:
            Tuple of (profile_df, actual_time); (None, None) on failure
        """
        try:
            tolerance_td = np.timedelta64(
                int(self.config.max_time_tolerance_hours * 3600), 's'
            )
            data = _squeeze(ds)
            profile_at_time = data.sel(
                time=target_time,
                method='nearest',
                tolerance=tolerance_td,  # type: ignore[arg-type]
            )
            actual_time = pd.Timestamp(profile_at_time.time.values)
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
    all_features = []

    for event in stall_events:
        try:
            ds = xsnow.read(str(event['pro_file']), lazy=False)
            if ds is None or ds.data.time.size == 0:
                continue
            
            features = extractor.extract_features_for_interface(
                ds,
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
    if 'target' in features_df.columns:
        pos_count = (features_df['target'] == 1).sum()
        neg_count = (features_df['target'] == 0).sum()
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
    
    # NEW: Show lookback method breakdown if available
    if 'lookback_method' in features_df.columns:
        print(f"\nLookback methods:")
        method_counts = features_df['lookback_method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(features_df) * 100
            print(f"  {method}: {count} ({pct:.1f}%)")
    
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
    print("using dynamic LWC-based lookback (when both layers were dry).")
    print("=" * 80)