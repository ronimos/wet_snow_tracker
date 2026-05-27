"""
stall_detector.py
====================

Detects wetting front stall events and tracks them by layer ID.

This version:
- Tracks stalling by SNOWPACK element_ID instead of height
- Records which layer IDs are above and below the stalling interface
- Enables lookback to collect features before stalling occurs

Author: Ron Simenhois
Created: November 2025
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import from main project
try:
    from wetting_front_tracker.snowpack_reader import SnowpackProfile
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc
except ImportError:
    # This block will be hit if the path wasn't added above
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from wetting_front_tracker.snowpack_reader import SnowpackProfile
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StallDetectionConfig:
    """Configuration parameters for stall detection."""
    
    # Stall definition
    min_duration_hours: float = 12.0      # Minimum stall duration
    height_tolerance_m: float = 0.05      # ±5cm height tolerance
    min_lwc_threshold: float = 0.04       # 4% LWC for wetting front
    
    # Quality control
    min_data_points: int = 3              # Minimum points to confirm stall
    max_gap_hours: float = 6.0            # Maximum time gap in data
    
    # Physical constraints
    min_wetting_front_height: float = 0.05    # 5 cm minimum height
    min_snow_height: float = 0.25             # 25 cm minimum total snowpack
    max_duration_hours: float = 240.0         # 10 days max
    
    # Feature extraction
    feature_lookback_hours: float = 24.0      # How far back to extract features


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StallEvent:
    """
    Represents a single wetting front stall event with layer ID tracking.
    
    Attributes:
        event_id: Unique identifier for this event
        station_name: Name of the monitoring station
        pro_file: Path to source .pro file
        start_time: When stall began
        end_time: When stall ended (or data ended)
        stall_height: Height (m) where wetting front stalled
        stall_layer_id: Element ID of layer where stall occurred
        layer_above_id: Element ID of layer above the interface
        layer_below_id: Element ID of layer below the interface
        duration_hours: How long the stall lasted
        confidence: Confidence score (0-1) for this event
        n_data_points: Number of timesteps confirming stall
        height_std: Standard deviation of height during stall
        is_ongoing: Whether stall is still happening at end of data
    """
    event_id: str
    station_name: str
    pro_file: Path
    start_time: datetime
    end_time: datetime
    stall_height: float
    stall_layer_id: int
    layer_above_id: int
    layer_below_id: int
    duration_hours: float
    confidence: float
    n_data_points: int
    height_std: float
    is_ongoing: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame storage."""
        return {
            'event_id': self.event_id,
            'station_name': self.station_name,
            'pro_file': str(self.pro_file),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'stall_height': self.stall_height,
            'stall_layer_id': self.stall_layer_id,
            'layer_above_id': self.layer_above_id,
            'layer_below_id': self.layer_below_id,
            'duration_hours': self.duration_hours,
            'confidence': self.confidence,
            'n_data_points': self.n_data_points,
            'height_std': self.height_std,
            'is_ongoing': self.is_ongoing
        }


# ---------------------------------------------------------------------------
# Layer ID Utilities
# ---------------------------------------------------------------------------

def get_layer_at_height(
    profile_df: pd.DataFrame,
    target_height: float,
    tolerance: float = 0.05
) -> Optional[int]:
    """
    Find the layer ID (element_ID) at a specific height.
    
    Args:
        profile_df: DataFrame with 'height' and 'element_ID' columns
        target_height: Height (m) to search for
        tolerance: Search tolerance (m)
        
    Returns:
        Element ID at that height, or None if not found
    """
    if 'height' not in profile_df.columns or 'element_ID' not in profile_df.columns:
        logger.warning("Missing required columns: height or element_ID")
        return None
    
    # Find layers within tolerance of target height
    mask = np.abs(profile_df['height'] - target_height) <= tolerance
    matches = profile_df[mask]
    
    if len(matches) == 0:
        logger.debug(f"No layer found at height {target_height:.3f}m ±{tolerance}m")
        return None
    
    # If multiple matches, take the closest one
    closest_idx = (matches['height'] - target_height).abs().idxmin()
    layer_id_val = matches.loc[closest_idx, 'element_ID']
    if isinstance(layer_id_val, pd.Series):
        layer_id_val = layer_id_val.item()
    layer_id = int(layer_id_val)
    
    return layer_id


def get_adjacent_layers(
    profile_df: pd.DataFrame,
    layer_id: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the layer IDs immediately above and below a given layer.
    
    Args:
        profile_df: DataFrame with 'height' and 'element_ID' columns
        layer_id: Element ID of the reference layer
        
    Returns:
        Tuple of (layer_above_id, layer_below_id)
        Either can be None if layer is at boundary
    """
    if 'height' not in profile_df.columns or 'element_ID' not in profile_df.columns:
        return None, None
    
    # Find the reference layer
    ref_layer = profile_df[profile_df['element_ID'] == layer_id]
    
    if len(ref_layer) == 0:
        logger.warning(f"Layer ID {layer_id} not found in profile")
        return None, None
    
    ref_height = float(ref_layer['height'].iloc[0])
    
    # Find layer above (next higher height)
    above_layers = profile_df[profile_df['height'] > ref_height]
    layer_above_id = None
    if len(above_layers) > 0:
        # Get layer with minimum height among those above
        closest_above = above_layers.loc[above_layers['height'].idxmin()]
        layer_above_id = closest_above['element_ID']
        if isinstance(layer_above_id, pd.Series):
            layer_above_id = layer_above_id.item()
        layer_above_id = int(layer_above_id)
    
    # Find layer below (next lower height)
    below_layers = profile_df[profile_df['height'] < ref_height]
    layer_below_id = None
    if len(below_layers) > 0:
        # Get layer with maximum height among those below
        closest_below = below_layers.loc[below_layers['height'].idxmax()]
        layer_below_id = closest_below['element_ID']
        if isinstance(layer_below_id, pd.Series):
            layer_below_id = layer_below_id.item()
        layer_below_id = int(layer_below_id)
    
    return layer_above_id, layer_below_id


def verify_layer_id_persistence(
    profile: SnowpackProfile,
    layer_id: int,
    start_time: datetime,
    end_time: datetime
) -> bool:
    """
    Verify that a layer ID persists across a time range.
    
    Args:
        profile: SnowpackProfile object
        layer_id: Element ID to check
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        True if layer ID exists throughout the time range
    """
    try:
        # Select time range
        time_slice = profile.data.sel(timestamp=slice(start_time, end_time))
        
        # Check if layer_id appears in each timestamp
        for timestamp in time_slice.timestamp.values:
            profile_at_time = time_slice.sel(timestamp=timestamp)
            if 'element_ID' in profile_at_time:
                layer_ids = profile_at_time['element_ID'].values
                if layer_id not in layer_ids:
                    logger.debug(f"Layer {layer_id} missing at {timestamp}")
                    return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error verifying layer ID persistence: {e}")
        return False


# ---------------------------------------------------------------------------
# Stall Detection with Layer ID Tracking
# ---------------------------------------------------------------------------

class StallDetector:
    """
    Detects wetting front stall events and tracks them by layer ID.
    
    Key features:
    - Identifies which layer ID the wetting front stalled at
    - Records layer IDs above and below the interface
    - Enables feature extraction from specific layers
    """
    
    def __init__(self, config: Optional[StallDetectionConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection parameters (uses defaults if None)
        """
        self.config = config or StallDetectionConfig()
        self._event_counter = 0
    
    def find_stalls(
        self,
        profile: SnowpackProfile,
        wetting_front_timeseries: pd.Series,
        station_name: str
    ) -> List[StallEvent]:
        """
        Find all stall events with layer ID tracking.
        
        Args:
            profile: SnowpackProfile object (needed for layer ID lookup)
            wetting_front_timeseries: Series with datetime index and heights (meters)
            station_name: Station identifier
            
        Returns:
            List of detected StallEvent objects
        """
        if wetting_front_timeseries.empty:
            logger.warning(f"Empty time series for {station_name}")
            return []
        
        # Remove NaN values
        valid_data = wetting_front_timeseries.dropna()
        
        if len(valid_data) < self.config.min_data_points:
            logger.debug(f"Insufficient data points for {station_name}")
            return []
        
        # Check if snowpack is deep enough
        max_height = float(valid_data.max())
        if max_height < self.config.min_snow_height:
            logger.info(f"Snowpack too shallow for {station_name}")
            return []
        
        # Find stable periods (height-based detection)
        stable_periods = self._find_stable_periods(valid_data)
        
        # Filter by duration and quality
        stall_periods = self._filter_stalls(stable_periods, valid_data)
        
        # Convert to StallEvent objects with layer IDs
        events = []
        for period in stall_periods:
            event = self._create_stall_event(
                period,
                profile,
                station_name,
                profile.filename
            )
            if event is not None:
                events.append(event)
        
        logger.info(f"Found {len(events)} stall events for {station_name}")
        return events
    
    def _find_stable_periods(self, timeseries: pd.Series) -> List[Dict[str, Any]]:
        """
        Find periods where wetting front height is stable.
        
        Args:
            timeseries: Valid (non-NaN) wetting front time series
            
        Returns:
            List of dictionaries describing stable periods
        """
        stable_periods = []
        
        if len(timeseries) < 2:
            return stable_periods
        
        # Start first potential period
        period_start = timeseries.index[0]
        period_heights = [timeseries.iloc[0]]
        
        for i in range(1, len(timeseries)):
            current_time = timeseries.index[i]
            current_height = timeseries.iloc[i]
            prev_time = timeseries.index[i-1]
            
            # Check time gap
            time_diff = pd.Timestamp(current_time) - pd.Timestamp(prev_time) # type: ignore[arg-type]
            time_gap = time_diff.total_seconds() / 3600
            if time_gap > self.config.max_gap_hours:
                
                # Gap too large - end current period and start new one
                if len(period_heights) >= self.config.min_data_points:
                    stable_periods.append({
                        'start': period_start,
                        'end': timeseries.index[i-1],
                        'heights': period_heights.copy()
                    })
                period_start = current_time
                period_heights = [current_height]
                continue
            
            # Check if height is within tolerance of period mean
            period_mean = np.mean(period_heights)
            if abs(current_height - period_mean) <= self.config.height_tolerance_m:
                # Still stable
                period_heights.append(current_height)
            else:
                # Height deviated - check if we should save this period
                if len(period_heights) >= self.config.min_data_points:
                    stable_periods.append({
                        'start': period_start,
                        'end': timeseries.index[i-1],
                        'heights': period_heights.copy()
                    })
                # Start new period
                period_start = current_time
                period_heights = [current_height]
        
        # Save final period if valid
        if len(period_heights) >= self.config.min_data_points:
            stable_periods.append({
                'start': period_start,
                'end': timeseries.index[-1],
                'heights': period_heights.copy()
            })
        
        return stable_periods
    
    def _filter_stalls(
        self,
        stable_periods: List[Dict[str, Any]],
        timeseries: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Filter stable periods to identify actual stalls.
        
        Args:
            stable_periods: List of stable period dictionaries
            timeseries: Full time series
            
        Returns:
            List of stall period dictionaries
        """
        stalls = []
        
        for period in stable_periods:
            # Calculate duration
            duration = (period['end'] - period['start']).total_seconds() / 3600
            
            # Check duration constraints
            if duration < self.config.min_duration_hours:
                continue
            if duration > self.config.max_duration_hours:
                continue
            
            # Calculate statistics
            heights = period['heights']
            mean_height = float(np.mean(heights))
            std_height = float(np.std(heights))
            
            # Check height constraints
            if mean_height < self.config.min_wetting_front_height:
                continue
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                duration, len(heights), std_height
            )
            
            # Check if stall is ongoing (ends at last data point)
            is_ongoing = (period['end'] == timeseries.index[-1])
            
            # Add to stalls
            stalls.append({
                'start': period['start'],
                'end': period['end'],
                'height': mean_height,
                'height_std': std_height,
                'duration': duration,
                'n_points': len(heights),
                'confidence': confidence,
                'is_ongoing': is_ongoing
            })
        
        return stalls
    
    def _calculate_confidence(
        self,
        duration_hours: float,
        n_points: int,
        height_std: float
    ) -> float:
        """Calculate confidence score for a stall event."""
        # Duration component (0-0.4)
        duration_score = min(0.4, duration_hours / 48.0)
        
        # Data density component (0-0.3)
        density_score = min(0.3, n_points / 20.0)
        
        # Stability component (0-0.3)
        stability_score = max(0, 0.3 - (height_std / self.config.height_tolerance_m) * 0.1)
        
        return min(1.0, duration_score + density_score + stability_score)
    
    def _create_stall_event(
        self,
        stall_period: dict,
        profile: SnowpackProfile,
        station_name: str,
        pro_file: Path
    ) -> Optional[StallEvent]:
        """
        Create StallEvent with layer ID information.
        
        Args:
            stall_period: Dictionary describing the stall
            profile: SnowpackProfile for layer lookup
            station_name: Station name
            pro_file: Source file path
            
        Returns:
            StallEvent object or None if layer ID lookup fails
        """
        try:
            # Get profile at start of stall
            profile_at_stall = profile.data.sel(
                timestamp=stall_period['start'],
                method='nearest'
            )
            profile_df = profile_at_stall.to_dataframe().reset_index()
            
            # Find layer ID at stall height
            stall_height = stall_period['height']
            stall_layer_id = get_layer_at_height(
                profile_df,
                stall_height,
                self.config.height_tolerance_m
            )
            
            if stall_layer_id is None:
                # logger.warning(f"Could not find layer ID at height {stall_height:.3f}m")
                return None
            
            # Find adjacent layers
            layer_above_id, layer_below_id = get_adjacent_layers(
                profile_df,
                stall_layer_id
            )
            
            if layer_above_id is None or layer_below_id is None:
                #logger.warning(f"Could not find adjacent layers for layer {stall_layer_id}")
                return None
            
            # Create event
            self._event_counter += 1
            event_id = f"{station_name}_stall_{self._event_counter:04d}"
            
            event = StallEvent(
                event_id=event_id,
                station_name=station_name,
                pro_file=pro_file,
                start_time=stall_period['start'],
                end_time=stall_period['end'],
                stall_height=stall_height,
                stall_layer_id=stall_layer_id,
                layer_above_id=layer_above_id,
                layer_below_id=layer_below_id,
                duration_hours=stall_period['duration'],
                confidence=stall_period['confidence'],
                n_data_points=stall_period['n_points'],
                height_std=stall_period['height_std'],
                is_ongoing=stall_period['is_ongoing']
            )
            
            logger.debug(f"Created event {event_id}: "
                        f"layer {stall_layer_id} at {stall_height:.3f}m, "
                        f"above={layer_above_id}, below={layer_below_id}")
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating stall event: {e}")
            return None


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def extract_wetting_front_timeseries(summary: pd.DataFrame) -> pd.Series:
    """
    Extract wetting front height time series from summary DataFrame.
    
    Args:
        summary: Summary DataFrame with wet_front_lwc_height column
        
    Returns:
        Series with datetime index and heights in meters
    """
    if 'wet_front_lwc_height' not in summary.columns:
        logger.warning("No wet_front_lwc_height column in summary")
        return pd.Series(dtype=float)
    
    # Extract height column
    wetting_front = summary['wet_front_lwc_height'].copy()
    
    # Units already in meters from SNOWPACK
    
    return wetting_front


def detect_stalls(
    pro_file: Path,
    config: Optional[StallDetectionConfig] = None
) -> List[StallEvent]:
    """
    Convenience function to detect stalls in a .pro file.
    
    Args:
        pro_file: Path to .pro file
        config: Detection configuration
        
    Returns:
        List of StallEvent objects
    """
    if SnowpackProfile is None or wet_front_lwc is None:
        raise ImportError("SnowpackProfile or wet_front_lwc not available")
    
    # Load profile
    profile = SnowpackProfile(str(pro_file))
    
    # Calculate wetting front
    parameters_to_calculate = {
        "wet_front_lwc": wet_front_lwc
    }
    summary = profile.get_full_timeseries_summary(
        parameters_to_calculate=parameters_to_calculate
    )
    
    # Unpack tuple column
    if 'wet_front_lwc' in summary.columns:
        summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
            summary['wet_front_lwc'].tolist(),
            index=summary.index
        )
    
    # Extract wetting front
    wetting_front = extract_wetting_front_timeseries(summary)
    
    # Detect stalls
    detector = StallDetector(config)
    station_name = pro_file.stem
    events = detector.find_stalls(profile, wetting_front, station_name)
    
    return events


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    print("StallDetector - Layer ID-based tracking")
    print("=" * 80)
    print("This module detects wetting front stalls and tracks them by layer ID.")
    print("Use detect_stalls(pro_file) for quick detection.")
    print("=" * 80)