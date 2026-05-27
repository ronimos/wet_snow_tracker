"""
diagnostic_wrapper.py
====================

Automatic diagnostic instrumentation for the Wetting Front Tracker application.

This module provides non-invasive diagnostics by wrapping key functions to track:
- Profile processing success/failure rates
- Time-to-LOC calculation statistics
- NaN value detection and causes
- Overall pipeline health metrics

The diagnostics are enabled automatically when this module is imported and 
enable_diagnostics() is called. At program exit, a comprehensive summary is printed.

Usage:
    # In main.py, add near the top after imports:
    try:
        import diagnostic_wrapper
        diagnostic_wrapper.enable_diagnostics()
    except ImportError:
        pass  # Gracefully handle if not available

Standalone Usage:
    # Diagnose a single DataFrame
    from diagnostic_wrapper import diagnose_summary_df
    
    summary_df = get_full_timeseries_summary(profile)
    diagnose_summary_df(summary_df, name="MyProfile")

Key Features:
    - Zero-modification integration (uses function wrapping)
    - Real-time statistics tracking
    - Automatic summary at exit
    - Standalone diagnostic functions
    - Colored output for better visibility

Author: Ron Simenhois
Created: November 2025
Last Updated: November 2025
"""

import logging
import sys
import atexit
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistics Tracking
# ---------------------------------------------------------------------------

# Global statistics dictionary
_stats: Dict[str, int] = {
    'total_profiles': 0,
    'successful_profiles': 0,
    'failed_profiles': 0,
    'loc_detected': 0,
    'no_loc_detected': 0,
    'valid_time_to_loc': 0,
    'nan_time_to_loc': 0,
    'wetting_detected': 0,
    'no_wetting': 0
}

# Flag to track if diagnostics are enabled
_diagnostics_enabled = False


# ---------------------------------------------------------------------------
# Core Diagnostic Functions
# ---------------------------------------------------------------------------

def enable_diagnostics() -> None:
    """
    Enable comprehensive diagnostics by wrapping key functions.
    
    This function should be called once at application startup. It will:
    1. Find the main module
    2. Wrap critical functions with diagnostic instrumentation
    3. Register cleanup handlers
    
    The wrapped functions will automatically track statistics and log
    diagnostic information without modifying their behavior.
    
    Example:
        >>> import diagnostic_wrapper
        >>> diagnostic_wrapper.enable_diagnostics()
        >>> # Rest of your application code...
        
    Note:
        This must be called after the main module is imported but before
        the main workflow begins. If the main module can't be found,
        diagnostics will fail gracefully with a warning.
    """
    global _diagnostics_enabled
    
    if _diagnostics_enabled:
        logger.warning("Diagnostics already enabled")
        return
        
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC MODE ENABLED")
    logger.info("=" * 80)
    
    # Try to wrap functions in the main module
    if 'main' in sys.modules:
        main_module = sys.modules['main']
        
        # Wrap process_single_profile if it exists
        if hasattr(main_module, 'process_single_profile'):
            original_func = main_module.process_single_profile
            wrapped_func = wrap_process_single_profile(original_func)
            main_module.process_single_profile = wrapped_func
            logger.info("✓ Wrapped process_single_profile")
        else:
            logger.debug("process_single_profile not found in main module")
        
        # Wrap find_time_to_loc if we can find it
        try:
            # Try importing from wet_front_tracker module
            if 'wet_front_tracker' in sys.modules:
                wft_module = sys.modules['wet_front_tracker']
                if hasattr(wft_module, 'find_time_to_loc'):
                    original_func = wft_module.find_time_to_loc
                    wrapped_func = wrap_find_time_to_loc(original_func)
                    wft_module.find_time_to_loc = wrapped_func
                    # Also update in main if it imported it
                    if hasattr(main_module, 'find_time_to_loc'):
                        main_module.find_time_to_loc = wrapped_func
                    logger.info("✓ Wrapped find_time_to_loc")
        except Exception as e:
            logger.debug(f"Could not wrap find_time_to_loc: {e}")
    else:
        logger.warning("Main module not found - diagnostics may be limited")
    
    _diagnostics_enabled = True
    logger.info("=" * 80)


def wrap_process_single_profile(original_func: Callable) -> Callable:
    """
    Create a diagnostic wrapper for the process_single_profile function.
    
    This wrapper:
    - Tracks success/failure statistics
    - Logs profile processing events
    - Checks time_to_loc validity
    - Handles exceptions gracefully
    
    Args:
        original_func: The original process_single_profile function
        
    Returns:
        Wrapped function with same signature but added diagnostics
    """
    
    @wraps(original_func)
    def wrapper(*args, **kwargs) -> Optional[Dict[str, Any]]:
        """Wrapped process_single_profile with diagnostics."""
        _stats['total_profiles'] += 1
        
        # Extract identifiers for logging
        pro_file_path = args[0] if args else kwargs.get('pro_file_path')
        aspect = args[1] if len(args) > 1 else kwargs.get('aspect', 'unknown')
        
        file_id = (
            f"{pro_file_path.stem}_{aspect}" 
            if pro_file_path and hasattr(pro_file_path, 'stem')
            else "unknown"
        )
        
        logger.debug(f"\n{'─' * 80}")
        logger.debug(f"Processing: {file_id}")
        
        try:
            # Call original function
            result = original_func(*args, **kwargs)
            
            # Check result validity
            if result is None:
                _stats['failed_profiles'] += 1
                logger.warning(f"✗ {file_id}: Returned None")
                return None
            
            _stats['successful_profiles'] += 1
            
            # Check time_to_loc value
            time_to_loc = result.get('time_to_loc')
            if pd.isna(time_to_loc):
                _stats['nan_time_to_loc'] += 1
                logger.warning(f"⚠️  {file_id}: time_to_loc is NaN")
            else:
                _stats['valid_time_to_loc'] += 1
                logger.info(f"✓ {file_id}: time_to_loc = {time_to_loc:.2f} hours")
            
            return result
            
        except Exception as e:
            _stats['failed_profiles'] += 1
            logger.error(f"✗ {file_id}: Exception - {e}")
            raise
    
    return wrapper


def wrap_find_time_to_loc(original_func: Callable) -> Callable:
    """
    Create a diagnostic wrapper for the find_time_to_loc function.
    
    This wrapper:
    - Diagnoses why NaN values are returned
    - Checks for missing LOC detections
    - Checks for missing wetting data
    - Validates wet front penetration depth
    
    Args:
        original_func: The original find_time_to_loc function
        
    Returns:
        Wrapped function with same signature but added diagnostics
    """
    
    @wraps(original_func)
    def wrapper(
        summary_df: pd.DataFrame, 
        reference_date: Optional[Any] = None
    ) -> Optional[float]:
        """Wrapped find_time_to_loc with diagnostics."""
        
        # Call original function
        result = original_func(summary_df, reference_date)
        
        # Diagnose if NaN
        if pd.isna(result):
            logger.debug("  find_time_to_loc returned NaN:")
            
            # Check for LOC detection
            if 'weak_layer_height' not in summary_df.columns:
                logger.debug("    ✗ Missing 'weak_layer_height' column")
            elif summary_df['weak_layer_height'].isna().all():
                logger.debug("    ✗ All weak_layer_height values are NaN (no LOC detected)")
                _stats['no_loc_detected'] += 1
            else:
                num_loc = summary_df['weak_layer_height'].notna().sum()
                logger.debug(f"    ✓ Found {num_loc} LOC detections")
                _stats['loc_detected'] += 1
            
            # Check for wetting detection
            if 'wet_front_lwc_height' not in summary_df.columns:
                logger.debug("    ✗ Missing 'wet_front_lwc_height' column")
            elif summary_df['wet_front_lwc_height'].isna().all():
                logger.debug("    ✗ No wetting detected (all wet_front_lwc_height are NaN)")
                _stats['no_wetting'] += 1
            else:
                num_wet = summary_df['wet_front_lwc_height'].notna().sum()
                max_penetration = summary_df['wet_front_lwc_height'].max()
                logger.debug(f"    ✓ Found {num_wet} wet timesteps")
                logger.debug(f"    ✓ Max wet penetration: {max_penetration:.2f}m")
                _stats['wetting_detected'] += 1
                
                # Check if wet front reaches LOC
                if summary_df['weak_layer_height'].notna().any():
                    loc_depth = summary_df['weak_layer_height'].dropna().iloc[-1]
                    if max_penetration < loc_depth:
                        logger.debug(
                            f"    ⚠️  Wet front ({max_penetration:.2f}m) "
                            f"does not reach LOC ({loc_depth:.2f}m)"
                        )
        else:
            logger.debug(f"  find_time_to_loc returned: {result:.2f} hours")
        
        return result
    
    return wrapper


def print_summary() -> None:
    """
    Print a comprehensive diagnostic summary.
    
    This function is automatically called at program exit via atexit.
    It displays:
    - Total profiles processed
    - Success/failure rates
    - Time-to-LOC statistics
    - Warning messages for common issues
    
    The output uses box-drawing characters and color-coding (via logging)
    for improved readability.
    """
    if not _diagnostics_enabled:
        return
        
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    # Profile Processing Statistics
    print(f"\nProfile Processing:")
    print(f"  Total attempted: {_stats['total_profiles']}")
    print(f"  Successful: {_stats['successful_profiles']}")
    print(f"  Failed: {_stats['failed_profiles']}")
    
    if _stats['total_profiles'] > 0:
        pct_success = 100 * _stats['successful_profiles'] / _stats['total_profiles']
        print(f"  Success rate: {pct_success:.1f}%")
    
    # Time to LOC Statistics
    print(f"\ntime_to_loc Results:")
    print(f"  Valid values: {_stats['valid_time_to_loc']}")
    print(f"  NaN values: {_stats['nan_time_to_loc']}")
    
    if _stats['successful_profiles'] > 0:
        pct_valid = 100 * _stats['valid_time_to_loc'] / _stats['successful_profiles']
        pct_nan = 100 * _stats['nan_time_to_loc'] / _stats['successful_profiles']
        print(f"  Valid: {pct_valid:.1f}%")
        print(f"  NaN: {pct_nan:.1f}%")
        
        # Warning for high NaN rate
        if pct_nan > 50:
            print(f"\n{'⚠️ ' * 20}")
            print("WARNING: Majority of profiles have NaN time_to_loc!")
            print("This will cause gray polygons in the map.")
            print("Common causes:")
            print("  1. No LOC detected in profiles")
            print("  2. No wetting detected in profiles")
            print("  3. Wet front doesn't reach LOC depth")
            print("See diagnostic output above for specific failures.")
            print(f"{'⚠️ ' * 20}")
    
    # Additional statistics
    if _stats['loc_detected'] > 0 or _stats['no_loc_detected'] > 0:
        total_loc_attempts = _stats['loc_detected'] + _stats['no_loc_detected']
        pct_loc = 100 * _stats['loc_detected'] / total_loc_attempts
        print(f"\nLOC Detection:")
        print(f"  LOC found: {_stats['loc_detected']} ({pct_loc:.1f}%)")
        print(f"  No LOC: {_stats['no_loc_detected']}")
    
    if _stats['wetting_detected'] > 0 or _stats['no_wetting'] > 0:
        total_wet_attempts = _stats['wetting_detected'] + _stats['no_wetting']
        pct_wet = 100 * _stats['wetting_detected'] / total_wet_attempts
        print(f"\nWetting Detection:")
        print(f"  Wetting found: {_stats['wetting_detected']} ({pct_wet:.1f}%)")
        print(f"  No wetting: {_stats['no_wetting']}")
    
    print("=" * 80 + "\n")


# Register cleanup to print summary at exit
atexit.register(print_summary)


# ---------------------------------------------------------------------------
# Standalone Diagnostic Functions
# ---------------------------------------------------------------------------

def diagnose_summary_df(
    summary_df: pd.DataFrame, 
    name: str = "profile"
) -> None:
    """
    Diagnose a summary DataFrame to identify data quality issues.
    
    This function performs comprehensive checks on a summary DataFrame,
    looking for common issues like missing data, low fill rates, and
    problematic column values.
    
    Args:
        summary_df: DataFrame from get_full_timeseries_summary() or similar
        name: Identifier for this profile (used in output)
        
    Example:
        >>> from wet_front_tracker import get_full_timeseries_summary
        >>> summary = get_full_timeseries_summary(profile)
        >>> diagnose_summary_df(summary, name="BerthoudPass_N")
        
        DIAGNOSING: BerthoudPass_N
        ────────────────────────────────────────
        DataFrame shape: (500, 15)
        Columns: ['timestamp', 'hs', 'weak_layer_height', ...]
        
        Key columns status:
          ✓ Snow depth           (hs)                  :  500/ 500 (100.0%)
          ⚠ LOC height          (weak_layer_height)   :  234/ 500 ( 46.8%)
             → Majority are NaN - check data quality
          ...
    """
    print(f"\n{'─' * 80}")
    print(f"DIAGNOSING: {name}")
    print(f"{'─' * 80}")
    
    print(f"\nDataFrame shape: {summary_df.shape}")
    print(f"Columns: {summary_df.columns.tolist()}")
    
    # Define key columns to check
    key_cols = {
        'hs': 'Snow depth',
        'weak_layer_height': 'LOC height', 
        'weak_layer': 'LOC (raw)',
        'wet_front_lwc_height': 'Wetting front',
        'wet_front_lwc': 'Wetting front (raw)',
        'lwc_above_weak': 'LWC above LOC'
    }
    
    print(f"\nKey columns status:")
    for col, desc in key_cols.items():
        if col in summary_df.columns:
            series = summary_df[col]
            non_null = series.notna().sum()
            total = len(series)
            pct = 100 * non_null / total if total > 0 else 0
            
            # Choose status symbol based on fill rate
            if pct > 50:
                status = "✓"
            elif pct > 0:
                status = "⚠️"
            else:
                status = "✗"
            
            print(f"  {status} {desc:20s} ({col:25s}): {non_null:4d}/{total:4d} ({pct:5.1f}%)")
            
            # Add diagnostic messages
            if pct == 0:
                print(f"     → All values are NaN!")
            elif pct < 50:
                print(f"     → Majority are NaN - check data quality")
        else:
            print(f"  ✗ {desc:20s} ({col:25s}): MISSING")
    
    # Check for multi-LOC columns (weak_layer_height_0, weak_layer_height_1, etc.)
    weak_layer_cols = [c for c in summary_df.columns if c.startswith('weak_layer_height_')]
    if weak_layer_cols:
        print(f"\nMulti-LOC columns found: {len(weak_layer_cols)}")
        for col in weak_layer_cols:
            non_null = summary_df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
    
    print(f"{'─' * 80}\n")


def reset_statistics() -> None:
    """
    Reset all diagnostic statistics to zero.
    
    Useful for testing or when processing multiple independent datasets.
    
    Example:
        >>> # Process first dataset
        >>> process_profiles(dataset1)
        >>> print_summary()
        >>> 
        >>> # Reset for second dataset
        >>> reset_statistics()
        >>> process_profiles(dataset2)
        >>> print_summary()
    """
    global _stats
    for key in _stats:
        _stats[key] = 0
    logger.info("Diagnostic statistics reset")


def get_statistics() -> Dict[str, int]:
    """
    Get current diagnostic statistics.
    
    Returns:
        Dictionary with all tracked statistics
        
    Example:
        >>> stats = get_statistics()
        >>> print(f"Success rate: {stats['successful_profiles'] / stats['total_profiles']:.2%}")
    """
    return _stats.copy()


# ---------------------------------------------------------------------------
# Module Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Test the diagnostic wrapper with mock data.
    """
    print("Testing diagnostic_wrapper.py")
    print("=" * 80)
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
        'hs': np.random.rand(10) * 2,
        'weak_layer_height': [np.nan] * 5 + list(np.random.rand(5)),
        'wet_front_lwc_height': np.random.rand(10) * 1.5
    })
    
    # Test standalone diagnostic function
    diagnose_summary_df(test_df, name="test_profile")
    
    # Test statistics
    print("\nCurrent statistics:")
    for key, value in get_statistics().items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")