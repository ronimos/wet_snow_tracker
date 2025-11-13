"""
example_integration.py
======================

Example of integrating physical validation into ML data collection pipeline.

This shows how to use PhysicalConditionValidator to ensure only valid
stall events are included in the ML training dataset.

Author: Ron Simon
Created: November 2025
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

# Import your existing modules
from stall_detector import (
    StallDetector, StallDetectionConfig, StallEvent,
    extract_wetting_front_timeseries
)
from snowpack_reader import SnowpackProfile
from wet_front_tracker import wet_front_lwc
from feature_extractor import InterfaceFeatureExtractor

# Import validation
from test_stall_events import PhysicalConditionValidator

logger = logging.getLogger(__name__)


# ===========================================================================
# Validated Data Collection
# ===========================================================================

def collect_validated_ml_data(
    pro_files: List[Path],
    output_dir: Path,
    require_all_conditions: bool = True
) -> pd.DataFrame:
    """
    Collect ML training data with physical validation.
    
    Only includes stall events that pass physical condition checks.
    
    Args:
        pro_files: List of .pro files to process
        output_dir: Where to save results
        require_all_conditions: If True, all conditions must be valid
                              If False, just log warnings for invalid conditions
        
    Returns:
        DataFrame with validated stall events and features
    """
    # Initialize components
    detector_config = StallDetectionConfig(
        min_duration_hours=12.0,
        max_duration_hours=240.0,
        height_tolerance_m=0.05
    )
    detector = StallDetector(detector_config)
    validator = PhysicalConditionValidator()
    feature_extractor = InterfaceFeatureExtractor()
    
    all_valid_events = []
    validation_stats = {
        'files_processed': 0,
        'events_detected': 0,
        'events_valid': 0,
        'events_invalid': 0,
        'failures': {
            'lwc': 0,
            'temperature': 0,
            'grain_type': 0,
            'density': 0
        }
    }
    
    logger.info(f"Processing {len(pro_files)} files...")
    
    for pro_file in pro_files:
        try:
            # Load profile
            logger.info(f"Loading {pro_file.name}...")
            profile = SnowpackProfile(str(pro_file))
            
            # Calculate wetting front
            parameters_to_calculate = {"wet_front_lwc": wet_front_lwc}
            summary = profile.get_full_timeseries_summary(
                parameters_to_calculate=parameters_to_calculate
            )
            
            if summary.empty:
                logger.warning(f"Empty summary for {pro_file.name}")
                continue
            
            # Unpack wet_front_lwc
            if 'wet_front_lwc' in summary.columns:
                summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                    summary['wet_front_lwc'].tolist(),
                    index=summary.index
                )
            
            # Extract wetting front time series
            wetting_front = extract_wetting_front_timeseries(summary)
            
            if wetting_front.empty or wetting_front.notna().sum() == 0:
                logger.info(f"No wetting front in {pro_file.name}")
                continue
            
            # Detect stalls
            station_name = profile.metadata.get('stationName', pro_file.stem)
            events = detector.find_stalls(wetting_front, station_name, pro_file)
            
            validation_stats['files_processed'] += 1
            validation_stats['events_detected'] += len(events)
            
            logger.info(f"  Found {len(events)} stall event(s)")
            
            # Validate each event
            for event in events:
                try:
                    # Check all conditions at event start
                    results = validator.check_all_conditions(
                        profile.data,
                        event.stall_height,
                        event.start_time
                    )
                    
                    # Check if all conditions are valid
                    all_valid = results['all_valid'][0]
                    
                    if not all_valid:
                        # Log which conditions failed
                        validation_stats['events_invalid'] += 1
                        
                        lwc_valid, lwc_val = results['lwc']
                        temp_valid, temp_val = results['temperature']
                        grain_valid, grain_val = results['grain_type']
                        density_valid, density_val = results['density']
                        
                        failure_msg = []
                        
                        if not lwc_valid:
                            validation_stats['failures']['lwc'] += 1
                            failure_msg.append(f"LWC={lwc_val:.3f}")
                        if not temp_valid:
                            validation_stats['failures']['temperature'] += 1
                            temp_c = temp_val - 273.15 if not pd.isna(temp_val) else float('nan')
                            failure_msg.append(f"T={temp_c:.1f}°C")
                        if not grain_valid:
                            validation_stats['failures']['grain_type'] += 1
                            failure_msg.append(f"grain={grain_val}")
                        if not density_valid:
                            validation_stats['failures']['density'] += 1
                            failure_msg.append(f"ρ={density_val:.0f}")
                        
                        logger.warning(
                            f"  Event {event.event_id} INVALID: {', '.join(failure_msg)}"
                        )
                        
                        if require_all_conditions:
                            continue  # Skip this event
                    
                    # Event passed validation
                    validation_stats['events_valid'] += 1
                    logger.info(f"  Event {event.event_id} ✓ VALID")
                    
                    # Extract features (your existing feature extraction)
                    # Get profile at stall start time
                    profile_at_time = profile.data.sel(
                        time=event.start_time,
                        method='nearest'
                    )
                    
                    # Convert to DataFrame for feature extraction
                    profile_df = pd.DataFrame({
                        'height': profile_at_time.height.values
                    })
                    
                    for var in ['density', 'temperature', 'grain_size', 
                               'grain_type', 'lwc']:
                        if var in profile_at_time:
                            profile_df[var] = profile_at_time[var].values
                    
                    # Extract features
                    features = feature_extractor.extract_all_features(
                        profile_df,
                        event.stall_height,
                        event.start_time
                    )
                    
                    # Combine event metadata with features
                    event_data = {
                        **event.to_dict(),
                        **features,
                        'validated': True,
                        'validation_lwc': lwc_val if 'lwc_val' in locals() else None,
                        'validation_temp': temp_val if 'temp_val' in locals() else None
                    }
                    
                    all_valid_events.append(event_data)
                    
                except Exception as e:
                    logger.error(f"Error validating event {event.event_id}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing {pro_file.name}: {e}")
            continue
    
    # Create DataFrame
    if not all_valid_events:
        logger.warning("No valid events found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_valid_events)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'validated_stall_events.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"\nSaved {len(df)} validated events to {output_file}")
    
    # Print validation statistics
    print("\n" + "="*70)
    print("VALIDATION STATISTICS")
    print("="*70)
    print(f"Files processed: {validation_stats['files_processed']}")
    print(f"Events detected: {validation_stats['events_detected']}")
    print(f"Events VALID: {validation_stats['events_valid']} "
          f"({100*validation_stats['events_valid']/validation_stats['events_detected']:.1f}%)")
    print(f"Events INVALID: {validation_stats['events_invalid']} "
          f"({100*validation_stats['events_invalid']/validation_stats['events_detected']:.1f}%)")
    
    print("\nCondition failures:")
    total_failures = sum(validation_stats['failures'].values())
    for condition, count in validation_stats['failures'].items():
        if total_failures > 0:
            pct = 100 * count / total_failures
            print(f"  {condition}: {count} ({pct:.1f}%)")
    
    print("="*70)
    
    return df


# ===========================================================================
# Example Usage
# ===========================================================================

def main():
    """Example usage of validated data collection."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Define input/output
    data_dir = Path('data/input')
    output_dir = Path('data/ml_training')
    
    # Get all .pro files
    pro_files = list(data_dir.glob('*.pro'))
    
    if not pro_files:
        print(f"No .pro files found in {data_dir}")
        print("Add some .pro files to test the validation!")
        return
    
    print(f"Found {len(pro_files)} .pro files")
    
    # Collect validated data
    df = collect_validated_ml_data(
        pro_files,
        output_dir,
        require_all_conditions=True  # Strict validation
    )
    
    if not df.empty:
        print(f"\n✅ Successfully collected {len(df)} validated stall events!")
        print(f"\nSummary statistics:")
        print(f"  Duration (hours):")
        print(f"    Mean: {df['duration_hours'].mean():.1f}")
        print(f"    Range: {df['duration_hours'].min():.1f} - {df['duration_hours'].max():.1f}")
        print(f"  Height (m):")
        print(f"    Mean: {df['stall_height'].mean():.2f}")
        print(f"    Range: {df['stall_height'].min():.2f} - {df['stall_height'].max():.2f}")
        print(f"  Confidence:")
        print(f"    Mean: {df['confidence'].mean():.2f}")
    else:
        print("\n⚠️ No valid events found. Check your data!")


if __name__ == '__main__':
    main()
