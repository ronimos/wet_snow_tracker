"""
debug_loc_detection.py
======================

Diagnostic script to identify why LOCs aren't being detected.

Usage:
    python debug_loc_detection.py --pro-file path/to/file.pro --mode rule_based
    python debug_loc_detection.py --pro-file path/to/file.pro --mode ml_only --model-path path/to/model
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import your modules
from snowpack_reader import SnowpackProfile
from wet_front_tracker import (
    find_wet_slab_loc,
    find_wet_slab_loc_bottom_half,
    get_total_snow_depth,
    wet_front_lwc,
    find_time_to_loc
)
from ml_loc_detector import MLLocDetector, create_hybrid_loc_detector

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_single_timestamp(df: pd.DataFrame, timestamp: datetime, detector_func, detector_name: str):
    """Diagnose LOC detection for a single timestamp."""
    print(f"\n{'='*80}")
    print(f"TIMESTAMP: {timestamp}")
    print(f"Detector: {detector_name}")
    print(f"{'='*80}")
    
    # Check basic data availability
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['height', 'grain_type', 'grain_size', 'lwc']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️  MISSING COLUMNS: {missing}")
        return None
    
    print(f"\nSnow depth (hs): {df['height'].max():.2f} m")
    print(f"Number of layers: {len(df)}")
    
    # Check grain types
    if 'grain_type' in df.columns:
        unique_types = df['grain_type'].unique()
        print(f"Grain types present: {sorted(unique_types)}")
        
        # Check for FC/DH (400-600)
        fc_dh_mask = (df['grain_type'] >= 400) & (df['grain_type'] < 600)
        num_fc_dh = fc_dh_mask.sum()
        print(f"Faceted/Depth Hoar layers: {num_fc_dh}")
        if num_fc_dh > 0:
            print(f"  Heights: {df.loc[fc_dh_mask, 'height'].values}")
    
    # Check grain size differences
    if 'grain_size' in df.columns:
        df_sorted = df.sort_values('height').reset_index(drop=True)
        df_sorted['gs_diff'] = df_sorted['grain_size'].diff()
        
        print(f"\nGrain size range: {df['grain_size'].min():.3f} - {df['grain_size'].max():.3f} mm")
        print(f"Max positive gs_diff: {df_sorted['gs_diff'].max():.3f} mm")
        print(f"Max negative gs_diff: {df_sorted['gs_diff'].min():.3f} mm")
        
        # Show largest differences
        large_positive = df_sorted[df_sorted['gs_diff'] > 0.5]
        large_negative = df_sorted[df_sorted['gs_diff'] < -0.5]
        
        if not large_positive.empty:
            print(f"\nLarge positive gs_diff (coarse over fine):")
            for _, row in large_positive.head(3).iterrows():
                print(f"  Height {row['height']:.2f}m: diff={row['gs_diff']:.3f} mm")
        
        if not large_negative.empty:
            print(f"\nLarge negative gs_diff (fine over coarse - capillary barrier):")
            for _, row in large_negative.head(3).iterrows():
                print(f"  Height {row['height']:.2f}m: diff={row['gs_diff']:.3f} mm, grain_type={row.get('grain_type', 'N/A')}")
    
    # Check LWC
    if 'lwc' in df.columns:
        wet_layers = df[df['lwc'] > 0.03]
        print(f"\nWet layers (LWC > 3%): {len(wet_layers)}")
        if not wet_layers.empty:
            print(f"  Max LWC: {df['lwc'].max():.3f} (at height {df.loc[df['lwc'].idxmax(), 'height']:.2f}m)")
            print(f"  Wet region: {wet_layers['height'].min():.2f} - {wet_layers['height'].max():.2f} m")
    
    # Try LOC detection
    print(f"\n{'─'*80}")
    print("RUNNING LOC DETECTION...")
    print(f"{'─'*80}")
    
    try:
        result = detector_func(df)
        
        if result is None:
            print("❌ Result: None")
            return None
        
        if isinstance(result, list):
            if not result:
                print("❌ Result: Empty list []")
                return None
            
            print(f"✅ Found {len(result)} LOC candidate(s):")
            for i, (height, prob) in enumerate(result):
                print(f"   {i+1}. Height: {height:.2f} m, Probability: {prob:.3f}")
                
                # Find the layer at this height
                closest_idx = (df['height'] - height).abs().idxmin()
                layer = df.loc[closest_idx]
                print(f"      Layer details:")
                print(f"        Grain type: {layer.get('grain_type', 'N/A')}")
                print(f"        Grain size: {layer.get('grain_size', 'N/A'):.3f} mm")
                print(f"        LWC: {layer.get('lwc', 'N/A'):.3f}")
            return result
        
        elif isinstance(result, tuple) and len(result) == 2:
            height, value = result
            if pd.notna(height):
                print(f"✅ Found LOC at height: {height:.2f} m (value: {value:.3f})")
                return [(height, 1.0)]
            else:
                print("❌ Result: (None, None)")
                return None
        
        else:
            print(f"⚠️  Unexpected result type: {type(result)}")
            print(f"   Result: {result}")
            return None
    
    except Exception as e:
        print(f"❌ ERROR during detection: {e}")
        logger.exception("Detection failed")
        return None


def main():
    parser = argparse.ArgumentParser(description='Debug LOC detection')
    parser.add_argument('--pro-file', type=Path, required=True, help='Path to .pro file')
    parser.add_argument('--mode', choices=['rule_based', 'ml_only', 'hybrid'], default='rule_based',
                       help='LOC detection mode')
    parser.add_argument('--model-path', type=Path, help='Path to ML model (for ml_only/hybrid)')
    parser.add_argument('--date', help='Specific date to analyze (YYYY-MM-DD)')
    parser.add_argument('--show-all-dates', action='store_true', 
                       help='Show results for all dates (not just one)')
    
    args = parser.parse_args()
    
    if not args.pro_file.exists():
        logger.error(f"File not found: {args.pro_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"LOC DETECTION DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"File: {args.pro_file}")
    print(f"Mode: {args.mode}")
    
    # Load profile
    profile = SnowpackProfile(args.pro_file)
    
    if profile.data is None:
        logger.error("Failed to load profile data")
        return
    
    print(f"\nProfile loaded successfully")
    print(f"Date range: {pd.to_datetime(profile.data.timestamp.values[0])} to "
          f"{pd.to_datetime(profile.data.timestamp.values[-1])}")
    print(f"Number of timesteps: {len(profile.data.timestamp)}")
    
    # Setup detector
    if args.mode == 'rule_based':
        detector = find_wet_slab_loc
        detector_name = "Rule-based (find_wet_slab_loc)"
        
        def wrapped_detector(df):
            result = detector(df)
            if result is None or (isinstance(result, tuple) and result[0] is None):
                return []
            if isinstance(result, tuple):
                return [result]
            return result
        
        detector = wrapped_detector
        
    elif args.mode == 'ml_only':
        if not args.model_path or not args.model_path.exists():
            logger.error("--model-path required for ml_only mode")
            return
        
        ml_detector = MLLocDetector(args.model_path, probability_threshold=0.5)
        detector = lambda df: ml_detector.find_ml_loc(df, top_n=3)
        detector_name = f"ML Only ({args.model_path.name})"
        
    elif args.mode == 'hybrid':
        if not args.model_path or not args.model_path.exists():
            logger.error("--model-path required for hybrid mode")
            return
        
        detector = create_hybrid_loc_detector(
            model_path=args.model_path,
            use_ml_primary=True,
            ml_threshold=0.5,
            rule_based_fallback=find_wet_slab_loc,
            top_n=3
        )
        detector_name = f"Hybrid ({args.model_path.name})"
    
    # Select dates to analyze
    if args.date:
        target_date = pd.to_datetime(args.date)
        available_dates = pd.to_datetime(profile.data.timestamp.values)
        closest_idx = np.abs(available_dates - target_date).argmin()
        timestamps = [available_dates[closest_idx]]
    else:
        timestamps = pd.to_datetime(profile.data.timestamp.values)
        if not args.show_all_dates:
            # Just show a few representative dates
            n_dates = len(timestamps)
            indices = [0, n_dates//4, n_dates//2, 3*n_dates//4, -1]
            timestamps = [timestamps[i] for i in indices]
    
    # Analyze each timestamp
    results_summary = []
    
    for ts in timestamps:
        # Extract profile for this timestamp
        ds_slice = profile.data.sel(timestamp=ts)
        
        # Convert to DataFrame
        if hasattr(ds_slice, 'compute'):
            ds_slice = ds_slice.compute()
        
        df = ds_slice.to_dataframe().reset_index()
        df = df.dropna(subset=['height'])
        
        if df.empty:
            logger.warning(f"No data for {ts}")
            continue
        
        # Ensure grain_size_difference column exists
        if 'grain_size_difference' in df.columns:
            df['gs_difference'] = df['grain_size_difference']
        elif 'grain_size' in df.columns:
            df_sorted = df.sort_values('height')
            df['gs_difference'] = df_sorted['grain_size'].diff().values
        
        # Run diagnostics
        result = diagnose_single_timestamp(df, ts, detector, detector_name)
        
        results_summary.append({
            'timestamp': ts,
            'found_loc': result is not None and len(result) > 0,
            'num_locs': len(result) if result else 0,
            'snow_depth': df['height'].max()
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(results_summary)
    print(f"\nAnalyzed {len(summary_df)} timesteps")
    print(f"LOCs found: {summary_df['found_loc'].sum()} ({100*summary_df['found_loc'].mean():.1f}%)")
    print(f"No LOCs: {(~summary_df['found_loc']).sum()} ({100*(~summary_df['found_loc']).mean():.1f}%)")
    
    if not summary_df['found_loc'].any():
        print(f"\n{'⚠️ '*20}")
        print("NO LOCs DETECTED IN ANY TIMESTEP!")
        print("Possible issues:")
        print("  1. No faceted crystals (FC) or depth hoar (DH) layers present")
        print("  2. Grain size differences too small (< 0.5mm threshold)")
        print("  3. No capillary barriers (small-over-large interfaces)")
        print("  4. ML model threshold too high (if using ML mode)")
        print("  5. Missing required columns in data")
        print(f"{'⚠️ '*20}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
