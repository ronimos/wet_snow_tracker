"""
stall_detector.py
=================

Detects wetting front stall events in SNOWPACK profiles.

A "stall" is defined as when the wetting front remains at approximately
the same height for an extended period (e.g., 12+ hours), indicating
it has encountered a layer that impedes water infiltration.

Author: [Your name]
Created: November 2025
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import from main project - handle both installed and local development
try:
    from ..snowpack_reader import SnowpackProfile
    from ..wet_front_tracker import wet_front_lwc
except ImportError:
    # For standalone testing, add parent directories to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from snowpack_reader import SnowpackProfile
        from wet_front_tracker import wet_front_lwc
    except ImportError:
        logger.warning("Could not import SnowpackProfile - using test mode only")
        SnowpackProfile = None
        wet_front_lwc = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StallDetectionConfig:
    """Configuration parameters for stall detection."""
    
    # Stall definition
    min_duration_hours: float = 12.0      # Minimum stall duration
    height_tolerance_m: float = 0.05      # ±5cm height tolerance (in meters)
    min_lwc_threshold: float = 0.04       # 4% LWC for wetting front
    
    # Quality control
    min_data_points: int = 3              # Minimum points to confirm stall
    max_gap_hours: float = 6.0            # Maximum time gap in data
    
    # Analysis window
    lookback_hours: float = 48.0          # How far back to look for stalls
    
    # IMPORTANT: SNOWPACK height data is in METERS, not centimeters
    # This was verified - the 'height' column in SNOWPACK output is already in meters
    # If your data is in cm, you need to convert it first!


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StallEvent:
    """
    Represents a single wetting front stall event.
    
    Attributes:
        event_id: Unique identifier for this event
        station_name: Name of the monitoring station
        pro_file: Path to source .pro file
        start_time: When stall began
        end_time: When stall ended (or data ended)
        stall_height: Height (m) where wetting front stalled
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
            'duration_hours': self.duration_hours,
            'confidence': self.confidence,
            'n_data_points': self.n_data_points,
            'height_std': self.height_std,
            'is_ongoing': self.is_ongoing
        }


# ---------------------------------------------------------------------------
# Stall Detection
# ---------------------------------------------------------------------------

class StallDetector:
    """
    Detects wetting front stall events in SNOWPACK time series.
    
    Algorithm:
    1. Track wetting front position over time
    2. Identify periods where front height is stable (±tolerance)
    3. Filter for events lasting minimum duration
    4. Calculate confidence scores
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
        wetting_front_timeseries: pd.Series,
        station_name: str,
        pro_file: Path
    ) -> List[StallEvent]:
        """
        Find all stall events in a wetting front time series.
        
        Args:
            wetting_front_timeseries: Series with datetime index and heights (in METERS)
            station_name: Station identifier
            pro_file: Source .pro file path
            
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
        
        # Validate units - heights should be in reasonable range for meters
        max_height = float(valid_data.max())
        min_height = float(valid_data.min())
        
        if max_height > 20:
            logger.error(f"Height values too large (max={max_height:.1f}m) - likely still in cm!")
            logger.error("Use extract_wetting_front_timeseries() to auto-convert units")
            return []
        
        if max_height < 0.1 and min_height >= 0:
            logger.warning(f"Height values very small (max={max_height:.3f}m) - check units")
        
        logger.debug(f"Height range: {min_height:.2f}m to {max_height:.2f}m")
        
        # Find stable periods
        stable_periods = self._find_stable_periods(valid_data)
        
        # Convert to stall events
        stall_events = []
        for period_start, period_end, mean_height in stable_periods:
            event = self._create_stall_event(
                period_start, period_end, mean_height,
                valid_data, station_name, pro_file
            )
            
            if self._validate_stall(event):
                stall_events.append(event)
        
        logger.info(f"Found {len(stall_events)} stall events in {station_name}")
        return stall_events
    
    def _find_stable_periods(
        self,
        timeseries: pd.Series
    ) -> List[Tuple[datetime, datetime, float]]:
        """
        Identify periods where height is stable within tolerance.
        
        Args:
            timeseries: Time series of wetting front heights
            
        Returns:
            List of (start_time, end_time, mean_height) tuples
        """
        stable_periods = []
        current_period_start = None
        current_period_indices = []
        
        timestamps = timeseries.index.to_list()
        heights = timeseries.values
        
        # Detect units and convert if necessary
        # If heights are in cm (typical range 0-500), convert to meters
        # If already in meters (typical range 0-5), keep as is
        max_height = np.max(heights[~np.isnan(heights)])
        if max_height > 50:  # Likely in cm
            logger.info(f"Detected heights in cm (max: {max_height:.1f}), converting to meters")
            heights = np.divide(heights, 100.0)
            height_units = "cm (converted to m)"
        else:
            logger.debug(f"Heights appear to be in meters (max: {max_height:.2f})")
            height_units = "m"
        
        # Now heights are in meters, tolerance is in meters
        tolerance_m = self.config.height_tolerance_m
        
        for i in range(len(heights)):
            if current_period_start is None:
                # Start new period
                current_period_start = timestamps[i]
                current_period_indices = [i]
            else:
                # Check if this point is within tolerance of period mean
                period_heights = heights[current_period_indices]
                period_mean = float(np.mean(period_heights))
                
                if abs(float(heights[i]) - period_mean) <= tolerance_m:
                    # Point is within tolerance - extend period
                    current_period_indices.append(i)
                else:
                    # Point breaks the stability - evaluate current period
                    if len(current_period_indices) >= self.config.min_data_points:
                        period_end = timestamps[current_period_indices[-1]]
                        period_mean = float(np.mean(heights[current_period_indices]))
                        
                        duration = float((period_end - current_period_start).total_seconds() / 3600.0)
                        if duration >= self.config.min_duration_hours:
                            stable_periods.append((
                                current_period_start,
                                period_end,
                                period_mean
                            ))
                    
                    # Start new period
                    current_period_start = timestamps[i]
                    current_period_indices = [i]
        
        # Handle final period
        if current_period_indices and len(current_period_indices) >= self.config.min_data_points:
            period_end = timestamps[current_period_indices[-1]]
            period_mean = float(np.mean(heights[current_period_indices]))
            
            duration = float((period_end - current_period_start).total_seconds() / 3600.0)
            if duration >= self.config.min_duration_hours:
                stable_periods.append((
                    current_period_start,
                    period_end,
                    period_mean
                ))
        
        logger.debug(f"Found {len(stable_periods)} stable periods (units: {height_units}, tolerance: ±{tolerance_m}m)")
        return stable_periods
    
    def _create_stall_event(
        self,
        start_time: datetime,
        end_time: datetime,
        mean_height: float,
        full_timeseries: pd.Series,
        station_name: str,
        pro_file: Path
    ) -> StallEvent:
        """
        Create a StallEvent object from a stable period.
        
        Args:
            start_time: Period start
            end_time: Period end
            mean_height: Average height during period
            full_timeseries: Complete time series (for context)
            station_name: Station identifier
            pro_file: Source file
            
        Returns:
            StallEvent object
        """
        # Extract data for this period
        period_data = full_timeseries.loc[start_time:end_time]
        
        # Calculate metrics - explicit float conversions for type checker
        duration_hours = float((end_time - start_time).total_seconds() / 3600.0)
        height_std = float(period_data.std()) if len(period_data) > 0 else 0.0
        n_points = int(len(period_data))
        
        # Check if stall is ongoing (at end of available data)
        is_ongoing = (end_time == full_timeseries.index[-1])
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            duration_hours, height_std, n_points
        )
        
        # Generate unique ID
        self._event_counter += 1
        event_id = f"SE_{self._event_counter:06d}"
        
        return StallEvent(
            event_id=event_id,
            station_name=station_name,
            pro_file=pro_file,
            start_time=start_time,
            end_time=end_time,
            stall_height=float(mean_height),
            duration_hours=duration_hours,
            confidence=confidence,
            n_data_points=n_points,
            height_std=height_std,
            is_ongoing=is_ongoing
        )
    
    def _calculate_confidence(
        self,
        duration_hours: float,
        height_std: float,
        n_points: int
    ) -> float:
        """
        Calculate confidence score for a stall event.
        
        Higher confidence for:
        - Longer duration
        - Lower height variability
        - More data points
        
        Args:
            duration_hours: Stall duration
            height_std: Standard deviation of heights
            n_points: Number of data points
            
        Returns:
            Confidence score (0-1)
        """
        # Duration score: sigmoid centered at 18 hours
        # Convert to float to satisfy type checker
        duration_score = float(1.0 / (1.0 + np.exp(-(float(duration_hours) - 18.0) / 6.0)))
        
        # Stability score: inverse of normalized std
        max_std = float(self.config.height_tolerance_m)
        stability_score = 1.0 - min(float(height_std) / max_std, 1.0)
        
        # Data quality score: sigmoid centered at 10 points
        quality_score = float(1.0 / (1.0 + np.exp(-(float(n_points) - 10.0) / 3.0)))
        
        # Weighted average
        confidence = (
            0.4 * duration_score +
            0.4 * stability_score +
            0.2 * quality_score
        )
        
        return float(confidence)
    
    def _validate_stall(self, event: StallEvent) -> bool:
        """
        Validate that a stall event meets quality criteria.
        
        Args:
            event: StallEvent to validate
            
        Returns:
            True if event is valid
        """
        # Basic checks
        if event.duration_hours < self.config.min_duration_hours:
            return False
        
        if event.n_data_points < self.config.min_data_points:
            return False
        
        if event.confidence < 0.3:  # Minimum confidence threshold
            return False
        
        if event.stall_height < 0:  # Physical constraint
            return False
        
        return True


# ---------------------------------------------------------------------------
# Wetting Front Tracking
# ---------------------------------------------------------------------------

def extract_wetting_front_timeseries(
    summary_df: pd.DataFrame,
    lwc_threshold: float = 0.04
) -> pd.Series:
    """
    Extract wetting front position time series from summary DataFrame.
    
    Args:
        summary_df: Summary DataFrame with wet_front_lwc_height column
        lwc_threshold: LWC threshold for defining wetting front
        
    Returns:
        Series with datetime index and wetting front heights (in METERS)
    """
    if 'wet_front_lwc_height' not in summary_df.columns:
        logger.warning("No wet_front_lwc_height column in summary")
        return pd.Series(dtype=float)
    
    # Extract height column
    wetting_front = summary_df['wet_front_lwc_height'].copy()
    
    # Ensure datetime index
    if not isinstance(wetting_front.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, attempting conversion")
        try:
            wetting_front.index = pd.to_datetime(wetting_front.index)
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}")
            return pd.Series(dtype=float)
    
    # Check units and convert if necessary
    # SNOWPACK height should be in meters, but some outputs may be in cm
    valid_heights = wetting_front.dropna()
    if not valid_heights.empty:
        max_height = float(valid_heights.max())
        # If max height > 20, likely in centimeters (assuming snowpack < 20m)
        if max_height > 20:
            logger.warning(f"Height values appear to be in cm (max={max_height:.1f}), converting to meters")
            wetting_front = wetting_front / 100.0
    
    return wetting_front


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def detect_stalls_batch(
    pro_files: List[Path],
    detector: StallDetector,
    process_summary_func: callable
) -> pd.DataFrame:
    """
    Detect stalls across multiple .pro files.
    
    Args:
        pro_files: List of .pro file paths
        detector: StallDetector instance
        process_summary_func: Function to get summary from .pro file
                              Signature: (pro_file: Path) -> pd.DataFrame
        
    Returns:
        DataFrame with all detected stall events
    """
    all_events = []
    
    for pro_file in pro_files:
        try:
            logger.info(f"Processing {pro_file.name}")
            
            # Get summary data
            summary_df = process_summary_func(pro_file)
            
            if summary_df is None or summary_df.empty:
                logger.warning(f"No summary data for {pro_file.name}")
                continue
            
            # Extract wetting front
            wetting_front = extract_wetting_front_timeseries(summary_df)
            
            if wetting_front.empty:
                logger.debug(f"No wetting front data in {pro_file.name}")
                continue
            
            # Detect stalls
            station_name = pro_file.stem
            events = detector.find_stalls(wetting_front, station_name, pro_file)
            
            all_events.extend([e.to_dict() for e in events])
            
        except Exception as e:
            logger.error(f"Error processing {pro_file.name}: {e}", exc_info=True)
            continue
    
    if not all_events:
        logger.warning("No stall events found in any files")
        return pd.DataFrame()
    
    # Convert to DataFrame
    events_df = pd.DataFrame(all_events)
    logger.info(f"Total stall events found: {len(events_df)}")
    
    return events_df


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example usage: Detect stalls in a real .pro file.
    
    Usage:
        python stall_detector.py path/to/file.pro
        python stall_detector.py path/to/file.pro --min-duration 8 --tolerance 0.10
    """
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Detect wetting front stalls in SNOWPACK profiles')
    parser.add_argument('pro_file', type=Path, help='Path to .pro file')
    parser.add_argument('--min-duration', type=float, default=12.0, 
                       help='Minimum stall duration (hours, default: 12.0)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                       help='Height tolerance (meters, default: 0.05)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Check if file exists
    if not args.pro_file.exists():
        logger.error(f"File not found: {args.pro_file}")
        sys.exit(1)
    
    # Check if we can import required modules
    if SnowpackProfile is None or wet_front_lwc is None:
        logger.error("Cannot import SnowpackProfile or wet_front_lwc")
        logger.error("Make sure you're running from the project directory")
        logger.error("Or install the package: pip install -e .")
        sys.exit(1)
    
    try:
        # Load profile
        logger.info(f"Loading profile: {args.pro_file}")
        profile = SnowpackProfile(str(args.pro_file))
        logger.info(f"  Station: {profile.metadata.get('stationName', 'Unknown')}")
        logger.info(f"  Location: {profile.metadata.get('latitude')}, {profile.metadata.get('longitude')}")
        
        # Calculate summary with wetting front
        logger.info("Calculating wetting front time series...")
        parameters_to_calculate = {
            "wet_front_lwc": wet_front_lwc
        }
        
        # Only pass date parameters if they're specified
        summary_kwargs = {
            'parameters_to_calculate': parameters_to_calculate
        }
        
        if args.start_date:
            summary_kwargs['start_date'] = args.start_date
        if args.end_date:
            summary_kwargs['end_date'] = args.end_date
        
        summary = profile.get_full_timeseries_summary(parameters_to_calculate=summary_kwargs) # type: ignore[call-arg]
        
        if summary.empty:
            logger.error("Summary calculation returned empty DataFrame")
            sys.exit(1)
        
        logger.info(f"  Time range: {summary.index[0]} to {summary.index[-1]}")
        logger.info(f"  Data points: {len(summary)}")
        
        # Unpack wet_front_lwc tuple column
        if 'wet_front_lwc' in summary.columns:
            summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                summary['wet_front_lwc'].tolist(),
                index=summary.index
            )
        
        # Extract wetting front time series
        wetting_front = extract_wetting_front_timeseries(summary)
        
        if wetting_front.empty:
            logger.warning("No wetting front detected in profile")
            logger.warning("This may indicate:")
            logger.warning("  - No liquid water in snowpack during analysis period")
            logger.warning("  - Dry snow conditions")
            logger.warning("  - Early/late season data")
            sys.exit(0)
        
        # Count non-null values
        n_wet_points = wetting_front.notna().sum()
        logger.info(f"  Wetting front detected: {n_wet_points}/{len(wetting_front)} timesteps")
        
        if n_wet_points == 0:
            logger.warning("Wetting front height is null at all timesteps")
            sys.exit(0)
        
        # Display wetting front statistics
        wet_data = wetting_front.dropna()
        logger.info(f"  Height range: {wet_data.min():.2f}m to {wet_data.max():.2f}m")
        logger.info(f"  Mean height: {wet_data.mean():.2f}m")
        
        # Create detector
        config = StallDetectionConfig(
            min_duration_hours=args.min_duration,
            height_tolerance_m=args.tolerance
        )
        detector = StallDetector(config)
        
        # Detect stalls
        logger.info(f"\nDetecting stalls (min duration: {args.min_duration}h, tolerance: ±{args.tolerance}m)...")
        station_name = args.pro_file.stem
        events = detector.find_stalls(wetting_front, station_name, args.pro_file)
        
        # Display results
        print("\n" + "="*80)
        print(f"STALL DETECTION RESULTS: {args.pro_file.name}")
        print("="*80)
        
        if not events:
            print("\n❌ No stall events detected")
            print("\nPossible reasons:")
            print("  - Wetting front moved continuously (no impedance)")
            print("  - Stalls were shorter than minimum duration")
            print("  - Height variations exceeded tolerance")
            print("\nTry adjusting parameters:")
            print(f"  python {Path(__file__).name} {args.pro_file} --min-duration 6 --tolerance 0.10")
        else:
            print(f"\n✓ Found {len(events)} stall event(s):\n")
            
            for i, event in enumerate(events, 1):
                print(f"{'─'*80}")
                print(f"Event {i}: {event.event_id}")
                print(f"{'─'*80}")
                print(f"  Station:      {event.station_name}")
                print(f"  Height:       {event.stall_height:.2f} m")
                print(f"  Duration:     {event.duration_hours:.1f} hours")
                print(f"  Start:        {event.start_time}")
                print(f"  End:          {event.end_time}")
                print(f"  Confidence:   {event.confidence:.2f}")
                print(f"  Height std:   {event.height_std:.3f} m")
                print(f"  Data points:  {event.n_data_points}")
                print(f"  Ongoing:      {'Yes' if event.is_ongoing else 'No'}")
                print()
            
            # Summary statistics
            durations = [e.duration_hours for e in events]
            heights = [e.stall_height for e in events]
            confidences = [e.confidence for e in events]
            
            print(f"{'═'*80}")
            print("SUMMARY STATISTICS")
            print(f"{'═'*80}")
            print(f"  Total stalls:        {len(events)}")
            print(f"  Duration (hours):")
            print(f"    Mean:              {np.mean(durations):.1f}")
            print(f"    Median:            {np.median(durations):.1f}")
            print(f"    Range:             {min(durations):.1f} - {max(durations):.1f}")
            print(f"  Height (m):")
            print(f"    Mean:              {np.mean(heights):.2f}")
            print(f"    Range:             {min(heights):.2f} - {max(heights):.2f}")
            print(f"  Confidence:")
            print(f"    Mean:              {np.mean(confidences):.2f}")
            print(f"    Range:             {min(confidences):.2f} - {max(confidences):.2f}")
            print(f"{'═'*80}\n")
            
            # Export option
            print("💾 To export results:")
            print(f"   python -c \"from stall_detector import *; import pandas as pd; \"\\")
            print(f"             \"events = detect_stalls('{args.pro_file}'); \"\\")
            print(f"             \"pd.DataFrame([e.to_dict() for e in events]).to_csv('stalls.csv', index=False)\"")
        
        print()
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        sys.exit(1)