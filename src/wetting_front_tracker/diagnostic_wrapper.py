"""
diagnostic_wrapper.py
====================

Import this at the top of main.py to add automatic diagnostics without modifying code.

Usage in main.py:
    # Add this near the top of main.py after imports
    try:
        import diagnostic_wrapper
        diagnostic_wrapper.enable_diagnostics()
    except ImportError:
        pass
"""

import logging
import pandas as pd
import numpy as np
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

# Statistics tracking
_stats = {
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


def enable_diagnostics():
    """
    Enable comprehensive diagnostics by wrapping key functions.
    Call this once at the start of main().
    """
    logger.info("="*80)
    logger.info("DIAGNOSTIC MODE ENABLED")
    logger.info("="*80)
    
    # We'll wrap the process_single_profile function
    import sys
    if 'main' in sys.modules:
        main_module = sys.modules['main']
        
        # Wrap process_single_profile
        if hasattr(main_module, 'process_single_profile'):
            original_func = main_module.process_single_profile
            wrapped_func = wrap_process_single_profile(original_func)
            main_module.process_single_profile = wrapped_func
            logger.info("✓ Wrapped process_single_profile")
        
        # Wrap find_time_to_loc
        if hasattr(main_module, 'find_time_to_loc'):
            from . import wet_front_tracker
            original_func = wet_front_tracker.find_time_to_loc
            wrapped_func = wrap_find_time_to_loc(original_func)
            wet_front_tracker.find_time_to_loc = wrapped_func
            main_module.find_time_to_loc = wrapped_func
            logger.info("✓ Wrapped find_time_to_loc")
    
    logger.info("="*80)


def wrap_process_single_profile(original_func):
    """Wraps process_single_profile to add diagnostics."""
    
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        _stats['total_profiles'] += 1
        
        pro_file_path = args[0] if args else kwargs.get('pro_file_path')
        aspect = args[1] if len(args) > 1 else kwargs.get('aspect', 'unknown')
        
        file_id = f"{pro_file_path.stem}_{aspect}" if pro_file_path else "unknown"
        
        logger.debug(f"\n{'─'*80}")
        logger.debug(f"Processing: {file_id}")
        
        try:
            result = original_func(*args, **kwargs)
            
            if result is None:
                _stats['failed_profiles'] += 1
                logger.warning(f"❌ {file_id}: Returned None")
                return None
            
            _stats['successful_profiles'] += 1
            
            # Check time_to_loc
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
            logger.error(f"❌ {file_id}: Exception - {e}")
            raise
    
    return wrapper


def wrap_find_time_to_loc(original_func):
    """Wraps find_time_to_loc to diagnose why it returns NaN."""
    
    @wraps(original_func)
    def wrapper(summary_df, reference_date=None):
        result = original_func(summary_df, reference_date)
        
        # Diagnose if NaN
        if pd.isna(result):
            logger.debug("  find_time_to_loc returned NaN:")
            
            # Check for required columns
            if 'weak_layer_height' not in summary_df.columns:
                logger.debug("    ❌ Missing 'weak_layer_height' column")
            elif summary_df['weak_layer_height'].isna().all():
                logger.debug("    ❌ All weak_layer_height values are NaN (no LOC detected)")
            else:
                num_loc = summary_df['weak_layer_height'].notna().sum()
                logger.debug(f"    ✓ Found {num_loc} LOC detections")
            
            if 'wet_front_lwc_height' not in summary_df.columns:
                logger.debug("    ❌ Missing 'wet_front_lwc_height' column")
            elif summary_df['wet_front_lwc_height'].isna().all():
                logger.debug("    ❌ No wetting detected (all wet_front_lwc_height are NaN)")
            else:
                num_wet = summary_df['wet_front_lwc_height'].notna().sum()
                max_penetration = summary_df['wet_front_lwc_height'].max()
                logger.debug(f"    ✓ Found {num_wet} wet timesteps")
                logger.debug(f"    ✓ Max wet penetration: {max_penetration:.2f}m")
                
                # Check if wet front reaches LOC
                if summary_df['weak_layer_height'].notna().any():
                    loc_depth = summary_df['weak_layer_height'].dropna().iloc[-1]
                    if max_penetration < loc_depth:
                        logger.debug(f"    ⚠️  Wet front ({max_penetration:.2f}m) does not reach LOC ({loc_depth:.2f}m)")
        else:
            logger.debug(f"  find_time_to_loc returned: {result:.2f} hours")
        
        return result
    
    return wrapper


def print_summary():
    """Print diagnostic summary."""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    print(f"\nProfile Processing:")
    print(f"  Total attempted: {_stats['total_profiles']}")
    print(f"  Successful: {_stats['successful_profiles']}")
    print(f"  Failed: {_stats['failed_profiles']}")
    
    if _stats['successful_profiles'] > 0:
        pct_success = 100 * _stats['successful_profiles'] / _stats['total_profiles']
        print(f"  Success rate: {pct_success:.1f}%")
    
    print(f"\ntime_to_loc Results:")
    print(f"  Valid values: {_stats['valid_time_to_loc']}")
    print(f"  NaN values: {_stats['nan_time_to_loc']}")
    
    if _stats['successful_profiles'] > 0:
        pct_valid = 100 * _stats['valid_time_to_loc'] / _stats['successful_profiles']
        pct_nan = 100 * _stats['nan_time_to_loc'] / _stats['successful_profiles']
        print(f"  Valid: {pct_valid:.1f}%")
        print(f"  NaN: {pct_nan:.1f}%")
        
        if pct_nan > 50:
            print(f"\n{'⚠️ '*20}")
            print("WARNING: Majority of profiles have NaN time_to_loc!")
            print("This will cause gray polygons in the map.")
            print("See DEBUGGING_GUIDE.md for troubleshooting steps.")
            print(f"{'⚠️ '*20}")
    
    print("="*80 + "\n")


# Register cleanup to print summary at exit
import atexit
atexit.register(print_summary)


# Standalone diagnostic function that can be called directly
def diagnose_summary_df(summary_df: pd.DataFrame, name: str = "profile"):
    """
    Diagnose a summary DataFrame to identify data issues.
    
    Args:
        summary_df: The summary DataFrame from get_full_timeseries_summary()
        name: Identifier for logging
    """
    print(f"\n{'─'*80}")
    print(f"DIAGNOSING: {name}")
    print(f"{'─'*80}")
    
    print(f"\nDataFrame shape: {summary_df.shape}")
    print(f"Columns: {summary_df.columns.tolist()}")
    
    # Check key columns
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
            
            status = "✓" if pct > 50 else "⚠️" if pct > 0 else "❌"
            print(f"  {status} {desc:20s} ({col:25s}): {non_null:4d}/{total:4d} ({pct:5.1f}%)")
            
            if pct == 0:
                print(f"     → All values are NaN!")
            elif pct < 50:
                print(f"     → Majority are NaN - check data quality")
        else:
            print(f"  ❌ {desc:20s} ({col:25s}): MISSING")
    
    # Check weak_layer expansion
    weak_layer_cols = [c for c in summary_df.columns if c.startswith('weak_layer_height_')]
    if weak_layer_cols:
        print(f"\nMulti-LOC columns found: {len(weak_layer_cols)}")
        for col in weak_layer_cols:
            non_null = summary_df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
    
    print(f"{'─'*80}\n")
