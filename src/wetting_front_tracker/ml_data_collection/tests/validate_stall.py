"""
validate_stall.py
=================

Quick validation script for checking physical validity of detected stall events.

Usage:
    python validate_stall.py path/to/file.pro
    python validate_stall.py path/to/file.pro --verbose

Author: Ron Simon
Created: November 2025
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import List

import pandas as pd

# Import from project
try:
    from stall_detector import (
        StallDetector, StallDetectionConfig, StallEvent,
        extract_wetting_front_timeseries
    )
    from snowpack_reader import SnowpackProfile
    from wet_front_tracker import wet_front_lwc
    from test_stall_events import PhysicalConditionValidator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure you're running from the project directory")
    print("Or install the package: pip install -e .")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ===========================================================================
# Validation Functions
# ===========================================================================

def validate_stall_event(
    event: StallEvent,
    profile: SnowpackProfile,
    validator: PhysicalConditionValidator,
    verbose: bool = False
) -> dict:
    """
    Validate a single stall event.
    
    Args:
        event: StallEvent to validate
        profile: SnowpackProfile containing the data
        validator: PhysicalConditionValidator instance
        verbose: Print detailed information
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'event_id': event.event_id,
        'valid': True,
        'issues': []
    }
    
    if verbose:
        print(f"\nEvent {event.event_id}: {event.station_name}")
        print(f"  Duration: {event.duration_hours:.1f} hours ({event.duration_hours/24:.1f} days)")
        print(f"  Height: {event.stall_height:.2f}m")
        print(f"  Start: {event.start_time}")
        print(f"  End: {event.end_time}")
    
    # Check start conditions
    if verbose:
        print(f"\n  Checking START conditions...")
    
    start_results = validator.check_all_conditions(
        profile.data,
        event.stall_height,
        event.start_time
    )
    
    if not start_results['all_valid'][0]:
        results['valid'] = False
        
        # Check which conditions failed
        lwc_valid, lwc_val = start_results['lwc']
        temp_valid, temp_val = start_results['temperature']
        grain_valid, grain_val = start_results['grain_type']
        density_valid, density_val = start_results['density']
        
        if not lwc_valid:
            issue = f"Start: LWC={lwc_val:.3f} < 0.04"
            results['issues'].append(issue)
            if verbose:
                print(f"    ✗ {issue}")
        
        if not temp_valid:
            temp_c = temp_val - 273.15 if not pd.isna(temp_val) else np.nan
            issue = f"Start: T={temp_c:.1f}°C outside valid range"
            results['issues'].append(issue)
            if verbose:
                print(f"    ✗ {issue}")
        
        if not grain_valid:
            issue = f"Start: Grain type={grain_val} invalid"
            results['issues'].append(issue)
            if verbose:
                print(f"    ✗ {issue}")
        
        if not density_valid:
            issue = f"Start: Density={density_val:.0f} kg/m³ outside valid range"
            results['issues'].append(issue)
            if verbose:
                print(f"    ✗ {issue}")
    else:
        if verbose:
            print(f"    ✓ All start conditions met")
            lwc_val = start_results['lwc'][1]
            temp_val = start_results['temperature'][1]
            temp_c = temp_val - 273.15
            grain_val = start_results['grain_type'][1]
            density_val = start_results['density'][1]
            print(f"      LWC: {lwc_val:.3f}")
            print(f"      T: {temp_c:.1f}°C")
            print(f"      Grain: {grain_val}")
            print(f"      Density: {density_val:.0f} kg/m³")
    
    # Check end conditions (if not ongoing)
    if not event.is_ongoing:
        if verbose:
            print(f"\n  Checking END conditions...")
        
        end_results = validator.check_all_conditions(
            profile.data,
            event.stall_height,
            event.end_time
        )
        
        if end_results['all_valid'][0]:
            # This is suspicious - why did it end if all conditions still met?
            issue = "End: All conditions still met (suspicious)"
            results['issues'].append(issue)
            if verbose:
                print(f"    ⚠ {issue}")
        else:
            if verbose:
                print(f"    ✓ At least one condition violated (correct end)")
                
                # Show which conditions were violated
                lwc_valid, lwc_val = end_results['lwc']
                temp_valid, temp_val = end_results['temperature']
                grain_valid, grain_val = end_results['grain_type']
                
                if not lwc_valid:
                    print(f"      LWC dropped: {lwc_val:.3f} < 0.04")
                if not temp_valid:
                    temp_c = temp_val - 273.15 if not pd.isna(temp_val) else np.nan
                    print(f"      Temperature: {temp_c:.1f}°C (refreezing)")
                if not grain_valid:
                    print(f"      Grain type: {grain_val} (frozen)")
    
    # Check continuity (sample mid-event)
    if verbose:
        print(f"\n  Checking CONTINUITY...")
    
    mid_time = event.start_time + timedelta(hours=event.duration_hours/2)
    mid_results = validator.check_all_conditions(
        profile.data,
        event.stall_height,
        mid_time
    )
    
    if not mid_results['all_valid'][0]:
        issue = "Mid-event: Conditions violated during stall"
        results['issues'].append(issue)
        if verbose:
            print(f"    ✗ {issue}")
    else:
        if verbose:
            print(f"    ✓ Conditions maintained mid-event")
    
    # Overall assessment
    if verbose:
        if results['valid'] and not results['issues']:
            print(f"\n  Overall: ✓ VALID")
        else:
            print(f"\n  Overall: ✗ INVALID")
            if results['issues']:
                print(f"  Issues:")
                for issue in results['issues']:
                    print(f"    - {issue}")
    
    return results


def validate_file(
    pro_file: Path,
    verbose: bool = False,
    detector_config: StallDetectionConfig = None
) -> dict:
    """
    Validate all stall events in a .pro file.
    
    Args:
        pro_file: Path to .pro file
        verbose: Print detailed information
        detector_config: Detector configuration
        
    Returns:
        Dictionary with validation summary
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"VALIDATING: {pro_file.name}")
        print(f"{'='*80}")
    
    # Load profile
    try:
        profile = SnowpackProfile(str(pro_file))
    except Exception as e:
        print(f"Error loading profile: {e}")
        return {'error': str(e)}
    
    # Calculate wetting front
    try:
        parameters_to_calculate = {"wet_front_lwc": wet_front_lwc}
        summary = profile.get_full_timeseries_summary(
            parameters_to_calculate=parameters_to_calculate
        )
        
        if summary.empty:
            print("No data in profile")
            return {'error': 'empty_profile'}
        
        # Unpack wet_front_lwc tuple column
        if 'wet_front_lwc' in summary.columns:
            summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                summary['wet_front_lwc'].tolist(),
                index=summary.index
            )
        
        wetting_front = extract_wetting_front_timeseries(summary)
        
        if wetting_front.empty or wetting_front.notna().sum() == 0:
            if verbose:
                print("\nNo wetting front detected")
            return {'n_events': 0, 'n_valid': 0, 'n_invalid': 0}
    
    except Exception as e:
        print(f"Error calculating wetting front: {e}")
        return {'error': str(e)}
    
    # Detect stalls
    config = detector_config or StallDetectionConfig()
    detector = StallDetector(config)
    
    try:
        events = detector.find_stalls(
            wetting_front,
            profile.metadata.get('stationName', pro_file.stem),
            pro_file
        )
    except Exception as e:
        print(f"Error detecting stalls: {e}")
        return {'error': str(e)}
    
    if not events:
        if verbose:
            print("\nNo stall events detected")
        return {'n_events': 0, 'n_valid': 0, 'n_invalid': 0}
    
    if verbose:
        print(f"\nFound {len(events)} stall event(s)")
    
    # Validate each event
    validator = PhysicalConditionValidator()
    validation_results = []
    
    for event in events:
        result = validate_stall_event(event, profile, validator, verbose)
        validation_results.append(result)
    
    # Summary
    n_valid = sum(1 for r in validation_results if r['valid'] and not r['issues'])
    n_invalid = len(validation_results) - n_valid
    
    summary = {
        'n_events': len(events),
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'events': validation_results
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"  Total events: {summary['n_events']}")
        print(f"  Valid: {summary['n_valid']} ({100*n_valid/len(events):.0f}%)")
        print(f"  Invalid: {summary['n_invalid']} ({100*n_invalid/len(events):.0f}%)")
        
        if n_invalid > 0:
            print(f"\n  Invalid events:")
            for result in validation_results:
                if not result['valid'] or result['issues']:
                    print(f"    - {result['event_id']}: {', '.join(result['issues'])}")
        
        print(f"{'='*80}\n")
    
    return summary


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Validate physical conditions of detected stall events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  python validate_stall.py data/input/138764_res.pro
  
  # Verbose output with details
  python validate_stall.py data/input/138764_res.pro --verbose
  
  # Adjust detection parameters
  python validate_stall.py data/input/138764_res.pro --min-duration 8
  
  # Batch validate all files
  for file in data/input/*.pro; do
      python validate_stall.py "$file"
  done
        """
    )
    
    parser.add_argument(
        'pro_file',
        type=Path,
        help='Path to .pro file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed validation information'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=12.0,
        help='Minimum stall duration (hours, default: 12.0)'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=240.0,
        help='Maximum stall duration (hours, default: 240.0)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='Height tolerance (meters, default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Check file exists
    if not args.pro_file.exists():
        print(f"Error: File not found: {args.pro_file}")
        sys.exit(1)
    
    # Create detector config
    config = StallDetectionConfig(
        min_duration_hours=args.min_duration,
        max_duration_hours=args.max_duration,
        height_tolerance_m=args.tolerance
    )
    
    # Validate
    try:
        results = validate_file(
            args.pro_file,
            verbose=args.verbose,
            detector_config=config
        )
        
        if 'error' in results:
            print(f"Validation failed: {results['error']}")
            sys.exit(1)
        
        # Exit with error if any invalid events
        if results.get('n_invalid', 0) > 0:
            sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
