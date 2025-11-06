"""
feature_extractor.py
====================

Extracts features from snowpack interfaces for ML training.

For each stall event, extracts characteristics of the layers above,
at, and below the interface where the wetting front stalled.

Author: [Your name]
Created: November 2025
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    # Layer thickness for averaging (meters)
    above_thickness: float = 0.20      # 20cm above interface
    below_thickness: float = 0.20      # 20cm below interface
    interface_thickness: float = 0.04  # ±2cm at interface
    
    # Feature categories to extract
    extract_density: bool = True
    extract_temperature: bool = True
    extract_grain: bool = True
    extract_lwc: bool = True
    extract_hardness: bool = True
    extract_structural: bool = True


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class InterfaceFeatureExtractor:
    """
    Extracts comprehensive features from snowpack interfaces.
    
    For a given height and time, extracts:
    - Statistics from layers above the interface
    - Statistics from layers at the interface (gradient)
    - Statistics from layers below the interface
    - Contextual snowpack information
    """
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """
        Initialize extractor with configuration.
        
        Args:
            config: Feature extraction parameters
        """
        self.config = config or FeatureExtractionConfig()
    
    def extract_all_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float,
        timestamp: datetime
    ) -> Dict[str, float]:
        """
        Extract all features for a given interface.
        
        Args:
            profile_df: DataFrame with single timestep profile data
                       Columns: height, density, temperature, grain_size, 
                               grain_type, lwc, hardness, etc.
            interface_height: Height (m) of the interface
            timestamp: Time of the profile
            
        Returns:
            Dictionary of feature_name: value pairs
        """
        features = {}
        
        # Validate input
        if profile_df.empty:
            logger.warning("Empty profile DataFrame")
            return features
        
        # Extract height information
        features.update(self._extract_height_features(
            profile_df, interface_height
        ))
        
        # Extract above-interface features
        above_features = self._extract_above_features(
            profile_df, interface_height
        )
        features.update({f'above_{k}': v for k, v in above_features.items()})
        
        # Extract interface features (gradients)
        interface_features = self._extract_interface_features(
            profile_df, interface_height
        )
        features.update({f'interface_{k}': v for k, v in interface_features.items()})
        
        # Extract below-interface features
        below_features = self._extract_below_features(
            profile_df, interface_height
        )
        features.update({f'below_{k}': v for k, v in below_features.items()})
        
        # Extract contextual features
        context_features = self._extract_context_features(
            profile_df, interface_height, timestamp
        )
        features.update({f'context_{k}': v for k, v in context_features.items()})
        
        # Extract computed features
        computed_features = self._compute_derived_features(
            above_features, interface_features, below_features
        )
        features.update({f'computed_{k}': v for k, v in computed_features.items()})
        
        return features
    
    def _get_layer_subset(
        self,
        profile_df: pd.DataFrame,
        height_min: float,
        height_max: float
    ) -> pd.DataFrame:
        """
        Extract layers within a height range.
        
        Args:
            profile_df: Full profile
            height_min: Bottom of range
            height_max: Top of range
            
        Returns:
            Subset of profile within range
        """
        if 'height' not in profile_df.columns:
            logger.error("No 'height' column in profile")
            return pd.DataFrame()
        
        mask = (profile_df['height'] >= height_min) & (profile_df['height'] <= height_max)
        return profile_df[mask].copy()
    
    def _extract_height_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float
    ) -> Dict[str, float]:
        """Extract height-related features."""
        features = {}
        
        if 'height' not in profile_df.columns:
            return features
        
        total_depth = profile_df['height'].max()
        
        features['absolute_height'] = interface_height
        features['relative_height'] = interface_height / total_depth if total_depth > 0 else 0
        features['distance_from_ground'] = interface_height
        features['distance_from_surface'] = total_depth - interface_height
        features['in_bottom_half'] = 1.0 if interface_height < (total_depth / 2) else 0.0
        
        # Count layers
        features['layers_below'] = len(profile_df[profile_df['height'] < interface_height])
        features['layers_above'] = len(profile_df[profile_df['height'] > interface_height])
        features['total_layers'] = len(profile_df)
        
        return features
    
    def _extract_above_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float
    ) -> Dict[str, float]:
        """Extract features from layers above the interface."""
        features = {}
        
        # Get layers above
        height_max = interface_height + self.config.above_thickness
        above_df = self._get_layer_subset(
            profile_df, interface_height, height_max
        )
        
        if above_df.empty:
            logger.debug(f"No layers above interface at {interface_height}m")
            return self._get_default_features()
        
        # Density features
        if self.config.extract_density and 'density' in above_df.columns:
            features['density_mean'] = above_df['density'].mean()
            features['density_std'] = above_df['density'].std()
            features['density_min'] = above_df['density'].min()
            features['density_max'] = above_df['density'].max()
        
        # Temperature features
        if self.config.extract_temperature and 'temperature' in above_df.columns:
            features['temperature_mean'] = above_df['temperature'].mean()
            features['temperature_std'] = above_df['temperature'].std()
        
        # Grain features
        if self.config.extract_grain:
            if 'grain_size' in above_df.columns:
                features['grain_size_mean'] = above_df['grain_size'].mean()
                features['grain_size_std'] = above_df['grain_size'].std()
            
            if 'grain_type' in above_df.columns:
                features['grain_type_mode'] = above_df['grain_type'].mode().iloc[0] if not above_df['grain_type'].mode().empty else np.nan
                features['has_facets'] = float(any((above_df['grain_type'] >= 400) & (above_df['grain_type'] < 600)))
        
        # LWC features
        if self.config.extract_lwc and 'lwc' in above_df.columns:
            features['lwc_mean'] = above_df['lwc'].mean()
            features['lwc_max'] = above_df['lwc'].max()
            features['lwc_std'] = above_df['lwc'].std()
        
        # Hardness features
        if self.config.extract_hardness and 'hardness' in above_df.columns:
            features['hardness_mean'] = above_df['hardness'].mean()
        
        # Structural features
        if self.config.extract_structural:
            if 'bond_size' in above_df.columns:
                features['bond_size_mean'] = above_df['bond_size'].mean()
            if 'coord_number' in above_df.columns:
                features['coord_number_mean'] = above_df['coord_number'].mean()
        
        return features
    
    def _extract_interface_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float
    ) -> Dict[str, float]:
        """Extract gradient features at the interface."""
        features = {}
        
        # Get layers at interface (±2cm)
        height_min = interface_height - self.config.interface_thickness / 2
        height_max = interface_height + self.config.interface_thickness / 2
        interface_df = self._get_layer_subset(
            profile_df, height_min, height_max
        )
        
        if len(interface_df) < 2:
            logger.debug(f"Insufficient layers at interface {interface_height}m")
            return self._get_default_features()
        
        # Calculate gradients (difference between top and bottom)
        top_layer = interface_df.iloc[-1]
        bottom_layer = interface_df.iloc[0]
        
        if 'density' in interface_df.columns:
            features['density_gradient'] = top_layer['density'] - bottom_layer['density']
        
        if 'temperature' in interface_df.columns:
            features['temperature_gradient'] = top_layer['temperature'] - bottom_layer['temperature']
        
        if 'grain_size' in interface_df.columns:
            features['grain_size_diff'] = top_layer['grain_size'] - bottom_layer['grain_size']
        
        if 'hardness' in interface_df.columns:
            features['hardness_diff'] = top_layer['hardness'] - bottom_layer['hardness']
        
        # Grain type transition
        if 'grain_type' in interface_df.columns:
            features['grain_type_above'] = top_layer['grain_type']
            features['grain_type_below'] = bottom_layer['grain_type']
            features['grain_type_change'] = abs(top_layer['grain_type'] - bottom_layer['grain_type'])
        
        return features
    
    def _extract_below_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float
    ) -> Dict[str, float]:
        """Extract features from layers below the interface."""
        features = {}
        
        # Get layers below
        height_min = interface_height - self.config.below_thickness
        below_df = self._get_layer_subset(
            profile_df, height_min, interface_height
        )
        
        if below_df.empty:
            logger.debug(f"No layers below interface at {interface_height}m")
            return self._get_default_features()
        
        # Similar to above features
        if self.config.extract_density and 'density' in below_df.columns:
            features['density_mean'] = below_df['density'].mean()
            features['density_std'] = below_df['density'].std()
        
        if self.config.extract_temperature and 'temperature' in below_df.columns:
            features['temperature_mean'] = below_df['temperature'].mean()
            features['temperature_std'] = below_df['temperature'].std()
        
        if self.config.extract_grain:
            if 'grain_size' in below_df.columns:
                features['grain_size_mean'] = below_df['grain_size'].mean()
                features['grain_size_std'] = below_df['grain_size'].std()
            
            if 'grain_type' in below_df.columns:
                features['grain_type_mode'] = below_df['grain_type'].mode().iloc[0] if not below_df['grain_type'].mode().empty else np.nan
                features['has_facets'] = float(any((below_df['grain_type'] >= 400) & (below_df['grain_type'] < 600)))
        
        if self.config.extract_lwc and 'lwc' in below_df.columns:
            features['lwc_mean'] = below_df['lwc'].mean()
        
        if self.config.extract_hardness and 'hardness' in below_df.columns:
            features['hardness_mean'] = below_df['hardness'].mean()
        
        return features
    
    def _extract_context_features(
        self,
        profile_df: pd.DataFrame,
        interface_height: float,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract contextual snowpack features."""
        features = {}
        
        if 'height' not in profile_df.columns:
            return features
        
        # Total snow depth
        features['total_snow_depth'] = profile_df['height'].max()
        
        # Number of layers
        features['n_layers_total'] = len(profile_df)
        
        # Temporal features
        features['day_of_year'] = timestamp.timetuple().tm_yday
        features['hour_of_day'] = timestamp.hour
        
        return features
    
    def _compute_derived_features(
        self,
        above_features: Dict[str, float],
        interface_features: Dict[str, float],
        below_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute derived features from basic measurements."""
        features = {}
        
        # Density contrast
        if 'density_mean' in above_features and 'density_mean' in below_features:
            above_density = above_features['density_mean']
            below_density = below_features['density_mean']
            if below_density > 0:
                features['density_contrast'] = abs(above_density - below_density) / below_density
            else:
                features['density_contrast'] = 0.0
        
        # Temperature inversion
        if 'temperature_mean' in above_features and 'temperature_mean' in below_features:
            features['temperature_inversion'] = (
                below_features['temperature_mean'] - above_features['temperature_mean']
            )
        
        # Grain size contrast
        if 'grain_size_mean' in above_features and 'grain_size_mean' in below_features:
            above_gs = above_features['grain_size_mean']
            below_gs = below_features['grain_size_mean']
            if below_gs > 0:
                features['grain_size_ratio'] = above_gs / below_gs
            else:
                features['grain_size_ratio'] = 1.0
        
        # Structural weakness index
        if 'grain_size_diff' in interface_features and 'grain_size_mean' in above_features:
            gs_diff = interface_features['grain_size_diff']
            gs_above = above_features['grain_size_mean']
            if gs_above > 0:
                features['structural_weakness'] = gs_diff / gs_above
            else:
                features['structural_weakness'] = 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return dictionary of NaN values for missing features."""
        return {
            'density_mean': np.nan,
            'density_std': np.nan,
            'temperature_mean': np.nan,
            'grain_size_mean': np.nan,
            'lwc_mean': np.nan,
            'hardness_mean': np.nan
        }


# ---------------------------------------------------------------------------
# Batch Feature Extraction
# ---------------------------------------------------------------------------

def extract_features_for_stall_events(
    stall_events: List[dict],
    get_profile_func: callable,
    extractor: InterfaceFeatureExtractor
) -> pd.DataFrame:
    """
    Extract features for a list of stall events.
    
    Args:
        stall_events: List of stall event dictionaries
        get_profile_func: Function to get profile at specific time
                         Signature: (pro_file: Path, timestamp: datetime) -> pd.DataFrame
        extractor: InterfaceFeatureExtractor instance
        
    Returns:
        DataFrame with stall events and their features
    """
    all_features = []
    
    for event in stall_events:
        try:
            # Get profile at stall time
            profile_df = get_profile_func(
                event['pro_file'],
                event['start_time']
            )
            
            if profile_df is None or profile_df.empty:
                logger.warning(f"No profile data for event {event['event_id']}")
                continue
            
            # Extract features
            features = extractor.extract_all_features(
                profile_df,
                event['stall_height'],
                event['start_time']
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
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example of how to use the extractor
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample profile
    profile_df = pd.DataFrame({
        'height': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'density': [350, 320, 310, 300, 280, 250, 240, 230, 220, 210, 200],
        'temperature': [270, 270.5, 271, 271.5, 272, 272.5, 273, 273.5, 274, 274.5, 275],
        'grain_size': [0.5, 0.6, 0.7, 0.8, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        'grain_type': [200, 200, 200, 200, 450, 200, 200, 200, 200, 200, 200],
        'lwc': [0.0, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0],
        'hardness': [3, 3, 3, 2, 1, 2, 3, 3, 4, 4, 4]
    })
    
    # Create extractor
    extractor = InterfaceFeatureExtractor()
    
    # Extract features at 0.8m (where grain type changes)
    features = extractor.extract_all_features(
        profile_df,
        interface_height=0.8,
        timestamp=datetime(2025, 5, 15, 12, 0)
    )
    
    print("\nExtracted features:")
    for key, value in sorted(features.items()):
        if pd.notna(value):
            print(f"{key:30s}: {value:.3f}")
