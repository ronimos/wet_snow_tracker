"""
collect_ml_data.py (COMPLETE UPDATED VERSION)
===============================================

Main script for collecting ML training data from wetting front stalls.

UPDATED: Now uses dynamic lookback based on LWC threshold instead of fixed 24h.
Features are extracted from the last timestamp where both layers had LWC < 1%.

Usage:
    python collect_ml_data.py --input data/input --output data/ml_training

Author: Ron Simenhois
Created: November 2025
Updated: November 2025 (Dynamic LWC-based lookback)
"""
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Handle imports
try:
    from wetting_front_tracker.ml_data_collection.stall_detector import (
        StallDetector,
        StallDetectionConfig,
        extract_wetting_front_timeseries
    )
    from wetting_front_tracker.ml_data_collection.feature_extractor import (
        LayerFeatureExtractor,
        FeatureExtractionConfig,
        print_feature_summary
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from wetting_front_tracker.ml_data_collection.stall_detector import (
        StallDetector,
        StallDetectionConfig,
        extract_wetting_front_timeseries
    )
    from wetting_front_tracker.ml_data_collection.feature_extractor import (
        LayerFeatureExtractor,
        FeatureExtractionConfig,
        print_feature_summary
    )
    
try:
    from wetting_front_tracker.snowpack_reader import get_full_timeseries_summary
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from wetting_front_tracker.snowpack_reader import get_full_timeseries_summary
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc

import xsnow
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MLDataCollectionConfig:
    """Configuration for ML data collection pipeline."""
    
    def __init__(self):
        # Detection parameters
        self.stall_config = StallDetectionConfig(
            min_duration_hours=12.0,
            max_duration_hours=240.0,
            height_tolerance_m=0.05,
            min_lwc_threshold=0.04,
            min_wetting_front_height=0.05,
            min_snow_height=0.25,
            feature_lookback_hours=24.0  # Now used as fallback
        )
        
        # UPDATED: Feature extraction with dynamic lookback
        self.feature_config = FeatureExtractionConfig(
            # NEW: Dynamic lookback settings
            use_dynamic_lookback=True,          # Enable dynamic lookback
            lwc_threshold_pct=1.0,              # Both layers must have LWC < 1%
            max_lookback_hours=72.0,            # Don't look back more than 72h
            min_lookback_hours=1.0,             # At least 1h before stall
            fallback_lookback_hours=24.0,       # Fallback if dynamic fails
            
            # Unchanged
            extract_all_parameters=True,
            max_time_tolerance_hours=2.0,
            compute_differences=True,
            compute_ratios=True,
            compute_gradients=True
        )
        
        # Balanced dataset configuration
        self.negative_sampling_strategy: str = 'combined'
        self.negatives_per_positive: int = 2
        self.nearby_distance_m: float = 0.05
        self.random_negative_distance_m: float = 0.10
        
        # Analysis parameters
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        
        # Output configuration
        self.output_dir = Path('data/ml_training')
        self.save_intermediate = True
        
        # Processing limits
        self.max_files: Optional[int] = None
        
        # Checkpointing
        self.checkpoint_interval: int = 500  # Save every N files/events


# ---------------------------------------------------------------------------
# Integration Functions
# ---------------------------------------------------------------------------

def get_summary_from_pro_file(
    pro_file: Path,
    config: MLDataCollectionConfig
) -> pd.DataFrame:
    """Get summary DataFrame from a .pro file."""
    try:
        ds = xsnow.read(str(pro_file), lazy=False)
        if ds is None or ds.data.time.size == 0:
            return pd.DataFrame()

        summary = get_full_timeseries_summary(
            ds,
            parameters_to_calculate={"wet_front_lwc": wet_front_lwc},
            start_date=config.start_date,
            end_date=config.end_date,
        )

        if 'wet_front_lwc' in summary.columns:
            summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                summary['wet_front_lwc'].tolist(),
                index=summary.index
            )

        return summary

    except Exception as e:
        logger.error(f"Error getting summary for {pro_file.name}: {e}")
        return pd.DataFrame()


def get_all_interfaces(profile_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Get all layer interfaces from a profile DataFrame.
    """
    if 'height' not in profile_df.columns or 'element_ID' not in profile_df.columns:
        logger.warning("Profile missing height or element_ID, cannot get interfaces")
        return []
        
    if profile_df.empty:
        return []
        
    profile_df = profile_df.sort_values(by='height').reset_index()
    
    interfaces = []
    for i in range(len(profile_df) - 1):
        layer_below = profile_df.iloc[i]
        layer_above = profile_df.iloc[i+1]
        try:
            interfaces.append({
                'above_id': int(layer_above['element_ID']),
                'below_id': int(layer_below['element_ID']),
                'interface_height': (layer_above['height'] + layer_below['height']) / 2.0
            })
        except Exception as e:
            continue        
    return interfaces


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def collect_ml_data(
    input_dir: Path,
    output_dir: Path,
    config: MLDataCollectionConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline for collecting ML training data with layer ID tracking.
    
    UPDATED: Now uses dynamic lookback based on LWC threshold.
    
    Args:
        input_dir: Directory containing .pro files
        output_dir: Directory to save results
        config: Configuration object
        
    Returns:
        Tuple of (stall_events_df, features_df)
    """
    logger.info("=" * 80)
    logger.info("ML Data Collection Pipeline")
    logger.info("Layer ID-Based Tracking with Dynamic LWC Lookback")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .pro files (recursive search)
    all_pro_files = sorted(input_dir.rglob('*_res.pro'))
    
    # Apply file limit if specified
    if config.max_files is not None:
        pro_files = all_pro_files[:config.max_files]
        logger.info(f"Processing {len(pro_files)} of {len(all_pro_files)} .pro files (limit={config.max_files})")
    else:
        pro_files = all_pro_files
        logger.info(f"Found {len(pro_files)} .pro files in {input_dir}")
    
    if not pro_files:
        logger.error("No .pro files found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Initialize detectors
    stall_detector = StallDetector(config.stall_config)
    feature_extractor = LayerFeatureExtractor(config.feature_config)
    
    # Phase 1: Detect stall events with layer IDs
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Detecting Stall Events (Layer ID Tracking)")
    logger.info("=" * 80)
    
    all_stall_events = []
    stall_checkpoint_path = output_dir / 'checkpoint_stall_events.csv'
    processed_files_path = output_dir / 'checkpoint_processed_files.txt'
    
    # Resume from checkpoint if available
    processed_files: set[str] = set()
    if stall_checkpoint_path.exists() and processed_files_path.exists():
        existing_df = pd.read_csv(stall_checkpoint_path)
        all_stall_events = existing_df.to_dict('records')
        processed_files = set(processed_files_path.read_text().strip().split('\n'))
        logger.info(f"Resuming Phase 1 from checkpoint: {len(all_stall_events)} events, "
                    f"{len(processed_files)} files already processed")
    
    files_since_checkpoint = 0
    for pro_file in tqdm(pro_files, desc="Detecting stalls"):
        if str(pro_file) in processed_files:
            continue
        
        try:
            ds = xsnow.read(str(pro_file), lazy=False)
            if ds is None or ds.data.time.size == 0:
                processed_files.add(str(pro_file))
                files_since_checkpoint += 1
                continue

            summary_df = get_summary_from_pro_file(pro_file, config)

            if summary_df is None or summary_df.empty:
                logger.debug(f"No summary for {pro_file.name}")
                processed_files.add(str(pro_file))
                files_since_checkpoint += 1
                continue
            
            wetting_front = extract_wetting_front_timeseries(summary_df)
            
            if wetting_front.empty or wetting_front.notna().sum() == 0:
                logger.debug(f"No wetting front for {pro_file.name}")
                processed_files.add(str(pro_file))
                files_since_checkpoint += 1
                continue
            
            station_name = pro_file.stem
            events = stall_detector.find_stalls(ds, wetting_front, station_name, pro_file)
            
            all_stall_events.extend([e.to_dict() for e in events])
            processed_files.add(str(pro_file))
            files_since_checkpoint += 1
            
        except Exception as e:
            logger.error(f"Error processing {pro_file.name}: {e}")
            processed_files.add(str(pro_file))
            files_since_checkpoint += 1
        
        # Periodic checkpoint (runs after both success and failure)
        if files_since_checkpoint >= config.checkpoint_interval:
            pd.DataFrame(all_stall_events).to_csv(stall_checkpoint_path, index=False)
            processed_files_path.write_text('\n'.join(processed_files))
            logger.info(f"Checkpoint: {len(processed_files)} files processed, "
                        f"{len(all_stall_events)} stall events found")
            files_since_checkpoint = 0
    
    # Final save of Phase 1
    pd.DataFrame(all_stall_events).to_csv(stall_checkpoint_path, index=False)
    processed_files_path.write_text('\n'.join(processed_files))
    
    # Convert to DataFrame
    if not all_stall_events:
        logger.error("No stall events detected!")
        return pd.DataFrame(), pd.DataFrame()
    
    stall_events_df = pd.DataFrame(all_stall_events)
    
    # Ensure datetime columns are parsed (they become strings after CSV round-trip)
    for col in ['start_time', 'end_time']:
        if col in stall_events_df.columns:
            stall_events_df[col] = pd.to_datetime(stall_events_df[col])
    
    logger.info(f"\nDetected {len(stall_events_df)} stall events")
    
    # Save intermediate results
    if config.save_intermediate:
        stall_events_path = output_dir / 'stall_events.csv'
        stall_events_df.to_csv(stall_events_path, index=False)
        logger.info(f"Saved stall events to {stall_events_path}")
    
    # Print statistics
    logger.info("\nStall Event Statistics:")
    logger.info(f"  Events detected:     {len(stall_events_df)}")
    logger.info(f"  Unique stations:     {stall_events_df['station_name'].nunique()}")
    logger.info(f"  Mean duration:       {stall_events_df['duration_hours'].mean():.1f} hours")
    logger.info(f"  Median duration:     {stall_events_df['duration_hours'].median():.1f} hours")
    logger.info(f"  Mean height:         {stall_events_df['stall_height'].mean():.2f} m")
    logger.info(f"  Mean confidence:     {stall_events_df['confidence'].mean():.2f}")
    logger.info(f"  Unique layers:       {stall_events_df['stall_layer_id'].nunique()}")
    
    # Phase 2: Extract features (Positive and Negative Examples)
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Extracting Features (Dynamic LWC-Based Lookback)")
    logger.info(f"Strategy: {config.negative_sampling_strategy}, Ratio (Pos:Neg): 1:{config.negatives_per_positive}")
    logger.info(f"Lookback: Dynamic (LWC < {config.feature_config.lwc_threshold_pct}%), "
                f"max={config.feature_config.max_lookback_hours}h, "
                f"fallback={config.feature_config.fallback_lookback_hours}h")
    logger.info("=" * 80)
    
    all_features = []
    failed_extractions = 0
    lookback_stats = {'dynamic_lwc': 0, 'fallback_fixed': 0, 'fixed': 0, 'explicit': 0}
    
    features_checkpoint_path = output_dir / 'checkpoint_features.csv'
    processed_events_path = output_dir / 'checkpoint_processed_events.txt'
    
    # Resume from checkpoint if available
    processed_event_ids: set[str] = set()
    if features_checkpoint_path.exists() and processed_events_path.exists():
        existing_features = pd.read_csv(features_checkpoint_path)
        all_features = existing_features.to_dict('records')
        processed_event_ids = set(processed_events_path.read_text().strip().split('\n'))
        logger.info(f"Resuming Phase 2 from checkpoint: {len(all_features)} examples, "
                    f"{len(processed_event_ids)} events already processed")
    
    events_since_checkpoint = 0
    for event in tqdm(stall_events_df.to_dict('records'),
                     desc="Extracting features"):
        event_id = str(event.get('event_id', ''))
        if event_id in processed_event_ids:
            continue
        
        try:
            ds = xsnow.read(str(event['pro_file']), lazy=False)
            if ds is None or ds.data.time.size == 0:
                logger.warning(f"Invalid profile from: {event['pro_file']}")
                failed_extractions += 1
                continue

            stall_interface_ids = {
                'above_id': event['layer_above_id'], 
                'below_id': event['layer_below_id']
            }
            stall_height = event['stall_height']

            # --- 1. Extract Positive Example ---
            # NEW: Pass stall_time directly - extractor finds optimal lookback
            positive_features = feature_extractor.extract_features_for_interface(
                ds,
                event['start_time'],  # Stall start time
                event['stall_layer_id'],
                event['layer_above_id'],
                event['layer_below_id']
            )
            
            if not positive_features:
                logger.warning(f"No features extracted for positive event {event['event_id']}")
                failed_extractions += 1
                continue
            
            # Get actual feature extraction time
            actual_feature_time_str = positive_features.get('feature_extraction_time')
            if actual_feature_time_str is None:
                logger.warning(f"No valid feature time for {event['event_id']}, skipping")
                failed_extractions += 1
                continue
            
            actual_feature_time = pd.to_datetime(actual_feature_time_str)
            
            # Get lookback info (now computed by extractor)
            actual_lookback = positive_features.get(
                'lookback_hours', 
                (event['start_time'] - actual_feature_time).total_seconds() / 3600
            )
            lookback_method = positive_features.get('lookback_method', 'unknown')
            
            # Track lookback method statistics
            if lookback_method in lookback_stats:
                lookback_stats[lookback_method] += 1
            
            logger.debug(
                f"Event {event['event_id']}: lookback={actual_lookback:.1f}h, "
                f"method={lookback_method}"
            )
            
            # Add positive example
            event_with_features = {
                **event, 
                **positive_features,
                'target': 1,
                'example_type': 'positive_stall',
                'distance_from_stall_m': 0.0,
                'lookback_hours': actual_lookback
            }
            all_features.append(event_with_features)

            # --- 2. Extract Negative Examples ---
            if config.negative_sampling_strategy == 'none':
                continue

            # Get profile at the actual feature extraction time
            try:
                data = ds.data.squeeze(['location', 'slope', 'realization'], drop=True)
                profile_at_time = data.sel(time=actual_feature_time, method='nearest')
                profile_df = profile_at_time.to_dataframe().reset_index()
            except Exception as e:
                logger.warning(f"Could not get profile at {actual_feature_time} for negatives: {e}")
                continue

            all_interfaces = get_all_interfaces(profile_df)
            if not all_interfaces:
                continue

            # Find potential negative interfaces
            nearby_interfaces = [
                i for i in all_interfaces 
                if abs(i['interface_height'] - stall_height) <= config.nearby_distance_m
                and (i['above_id'] != stall_interface_ids['above_id'] 
                     or i['below_id'] != stall_interface_ids['below_id'])
            ]
            random_interfaces = [
                i for i in all_interfaces
                if abs(i['interface_height'] - stall_height) > config.random_negative_distance_m
            ]
            
            np.random.shuffle(nearby_interfaces)
            np.random.shuffle(random_interfaces)
            
            negative_samples = []
            
            # Select samples based on strategy
            if config.negative_sampling_strategy == 'combined':
                samples_to_add = min(len(nearby_interfaces), config.negatives_per_positive)
                negative_samples.extend(nearby_interfaces[:samples_to_add])
                
                remaining_needed = config.negatives_per_positive - len(negative_samples)
                if remaining_needed > 0:
                    samples_to_add = min(len(random_interfaces), remaining_needed)
                    negative_samples.extend(random_interfaces[:samples_to_add])
            
            elif config.negative_sampling_strategy == 'nearby':
                samples_to_add = min(len(nearby_interfaces), config.negatives_per_positive)
                negative_samples.extend(nearby_interfaces[:samples_to_add])
            
            elif config.negative_sampling_strategy == 'random':
                samples_to_add = min(len(random_interfaces), config.negatives_per_positive)
                negative_samples.extend(random_interfaces[:samples_to_add])

            # Extract features for selected negative samples
            for neg_interface in negative_samples:
                # Use the SAME feature extraction time as positive example
                neg_features = feature_extractor.extract_features_for_interface(
                    ds,
                    event['start_time'],  # Reference stall time
                    0,  # Dummy stall_layer_id
                    neg_interface['above_id'],
                    neg_interface['below_id'],
                    requested_feature_time=actual_feature_time  # Use same time as positive
                )
                
                if not neg_features:
                    logger.warning(f"Failed to extract features for negative sample")
                    continue
                
                neg_event = {
                    **event, 
                    **neg_features,
                    'target': 0,
                    'distance_from_stall_m': neg_interface['interface_height'] - stall_height,
                    'lookback_hours': actual_lookback,
                    'layer_above_id': neg_interface['above_id'],
                    'layer_below_id': neg_interface['below_id'],
                    'stall_height': neg_interface['interface_height'],
                    'stall_layer_id': 0,
                }
                
                # Assign type
                if abs(neg_event['distance_from_stall_m']) <= config.nearby_distance_m:
                    neg_event['example_type'] = 'negative_nearby'
                else:
                    neg_event['example_type'] = 'negative_random'

                all_features.append(neg_event)
            
            processed_event_ids.add(event_id)
            events_since_checkpoint += 1
            
        except Exception as e:
            logger.error(f"Error extracting features for {event['event_id']}: {e}")
            failed_extractions += 1
            processed_event_ids.add(event_id)
            events_since_checkpoint += 1
        
        # Periodic checkpoint (runs after both success and failure)
        if events_since_checkpoint >= config.checkpoint_interval:
            pd.DataFrame(all_features).to_csv(features_checkpoint_path, index=False)
            processed_events_path.write_text('\n'.join(processed_event_ids))
            logger.info(f"Checkpoint: {len(processed_event_ids)} events processed, "
                        f"{len(all_features)} total examples")
            events_since_checkpoint = 0
            continue
    
    # Final checkpoint save
    if all_features:
        pd.DataFrame(all_features).to_csv(features_checkpoint_path, index=False)
        processed_events_path.write_text('\n'.join(processed_event_ids))
    
    # Convert to DataFrame
    if not all_features:
        logger.error("No features extracted!")
        return stall_events_df, pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)
    
    logger.info(f"\nFeature Extraction Results:")
    logger.info(f"  Positive stall events: {len(stall_events_df)}")
    logger.info(f"  Total examples:        {len(features_df)}")
    logger.info(f"  Failed extractions:    {failed_extractions}")
    
    if 'target' in features_df.columns:
        pos_count = (features_df['target'] == 1).sum()
        neg_count = (features_df['target'] == 0).sum()
        logger.info(f"    Positive (target=1):  {pos_count}")
        logger.info(f"    Negative (target=0):  {neg_count}")
        if pos_count > 0:
            ratio = neg_count / pos_count
            logger.info(f"    Actual P:N ratio:      1 : {ratio:.1f}")
    
    # NEW: Show lookback method statistics
    logger.info(f"\n  Lookback Method Statistics (positives only):")
    total_pos = sum(lookback_stats.values())
    for method, count in lookback_stats.items():
        if count > 0:
            pct = count / total_pos * 100 if total_pos > 0 else 0
            logger.info(f"    {method}: {count} ({pct:.1f}%)")
    
    # NEW: Show lookback time distribution
    if 'lookback_hours' in features_df.columns:
        pos_lookbacks = features_df[features_df['target'] == 1]['lookback_hours']
        logger.info(f"\n  Lookback Time Distribution (positives):")
        logger.info(f"    Mean:   {pos_lookbacks.mean():.1f}h")
        logger.info(f"    Median: {pos_lookbacks.median():.1f}h")
        logger.info(f"    Std:    {pos_lookbacks.std():.1f}h")
        logger.info(f"    Min:    {pos_lookbacks.min():.1f}h")
        logger.info(f"    Max:    {pos_lookbacks.max():.1f}h")
    
    # Print feature summary
    print_feature_summary(features_df)
    
    # Save features
    features_path = output_dir / 'ml_training_dataset.csv'
    features_df.to_csv(features_path, index=False)
    logger.info(f"\nSaved training dataset to {features_path}")
    
    # Generate summary report
    generate_summary_report(stall_events_df, features_df, output_dir)
    
    return stall_events_df, features_df


def generate_summary_report(
    stall_events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Generate summary statistics and save to file.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)
    
    report = []
    report.append("ML Training Data Collection Summary")
    report.append("Layer ID-Based Tracking with Dynamic LWC Lookback")
    report.append("=" * 80)
    report.append(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # Stall events statistics
    report.append("STALL EVENTS (Positive Examples):")
    report.append(f"  Total positive events detected: {len(stall_events_df)}")
    report.append(f"  Unique stations: {stall_events_df['station_name'].nunique()}")
    report.append(f"  Unique stall layers: {stall_events_df['stall_layer_id'].nunique()}")
    report.append(f"  Duration (hours):")
    report.append(f"    Mean: {stall_events_df['duration_hours'].mean():.1f}")
    report.append(f"    Median: {stall_events_df['duration_hours'].median():.1f}")
    report.append(f"    Min: {stall_events_df['duration_hours'].min():.1f}")
    report.append(f"    Max: {stall_events_df['duration_hours'].max():.1f}")
    report.append(f"  Height (m):")
    report.append(f"    Mean: {stall_events_df['stall_height'].mean():.2f}")
    report.append(f"    Median: {stall_events_df['stall_height'].median():.2f}")
    report.append("")
    
    # Features statistics
    if not features_df.empty:
        report.append("FULL DATASET (Positive + Negative):")
        report.append(f"  Total examples: {len(features_df)}")
        pos_count = (features_df['target'] == 1).sum()
        neg_count = (features_df['target'] == 0).sum()
        report.append(f"    Positive (target=1):  {pos_count}")
        report.append(f"    Negative (target=0):  {neg_count}")
        if pos_count > 0:
            ratio = neg_count / pos_count
            report.append(f"    Actual P:N ratio:      1 : {ratio:.1f}")
        
        if 'example_type' in features_df.columns:
            report.append(f"  Example type breakdown:")
            for etype, count in features_df['example_type'].value_counts().items():
                report.append(f"    {etype}: {count}")
            
        report.append(f"  Extraction success rate (pos): {pos_count / len(stall_events_df) * 100:.1f}%")
        
        # Count feature types
        feature_cols = [col for col in features_df.columns 
                       if col.startswith(('above_', 'below_', 'interface_'))]
        above_cols = [col for col in feature_cols if col.startswith('above_')]
        below_cols = [col for col in feature_cols if col.startswith('below_')]
        interface_cols = [col for col in feature_cols if col.startswith('interface_')]
        
        report.append(f"  Feature counts:")
        report.append(f"    Above layer: {len(above_cols)}")
        report.append(f"    Below layer: {len(below_cols)}")
        report.append(f"    Interface: {len(interface_cols)}")
        report.append(f"    Total: {len(feature_cols)}")
        
        # NEW: Lookback time statistics
        if 'lookback_hours' in features_df.columns:
            report.append(f"  Lookback times (hours):")
            report.append(f"    Mean: {features_df['lookback_hours'].mean():.1f}")
            report.append(f"    Median: {features_df['lookback_hours'].median():.1f}")
            report.append(f"    Std: {features_df['lookback_hours'].std():.1f}")
            report.append(f"    Min: {features_df['lookback_hours'].min():.1f}")
            report.append(f"    Max: {features_df['lookback_hours'].max():.1f}")
        
        # NEW: Lookback method breakdown
        if 'lookback_method' in features_df.columns:
            report.append(f"  Lookback method breakdown:")
            method_counts = features_df['lookback_method'].value_counts()
            for method, count in method_counts.items():
                pct = count / len(features_df) * 100
                report.append(f"    {method}: {count} ({pct:.1f}%)")
        
        # NEW: LWC at extraction time verification
        if 'above_lwc_at_extraction' in features_df.columns:
            above_lwc = features_df['above_lwc_at_extraction']
            below_lwc = features_df['below_lwc_at_extraction']
            report.append(f"  LWC at extraction time:")
            report.append(f"    Above layer: mean={above_lwc.mean():.4f}, max={above_lwc.max():.4f}")
            report.append(f"    Below layer: mean={below_lwc.mean():.4f}, max={below_lwc.max():.4f}")
            
            # Check how many actually met the threshold
            both_dry = ((above_lwc < 0.01) & (below_lwc < 0.01)).sum()
            report.append(f"    Both layers < 1%: {both_dry}/{len(features_df)} ({both_dry/len(features_df)*100:.1f}%)")
        
        # Missing data
        missing_pct = features_df[feature_cols].isnull().sum() / len(features_df) * 100
        report.append(f"  Missing data:")
        report.append(f"    Mean missing rate: {missing_pct.mean():.1f}%")
        high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
        if not high_missing.empty:
            report.append(f"    Features with >10% missing:")
            for feat, pct in high_missing.head(10).items():
                report.append(f"      {feat}: {pct:.1f}%")
        else:
            report.append(f"    All features have <10% missing data ✓")
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    logger.info("\n" + report_text)
    
    report_path = output_dir / 'collection_summary.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"\nSaved summary report to {report_path}")


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect ML training data from wetting front stalls (Dynamic LWC Lookback)'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/input'),
        help='Input directory with .pro files (default: data/input)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/ml_training'),
        help='Output directory for results (default: data/ml_training)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=float,
        default=12.0,
        help='Minimum stall duration in hours (default: 12.0)'
    )
    
    parser.add_argument(
        '--lookback-hours',
        type=float,
        default=24.0,
        help='Fallback lookback hours if dynamic fails (default: 24.0)'
    )
    
    # NEW: Dynamic lookback arguments
    parser.add_argument(
        '--lwc-threshold',
        type=float,
        default=1.0,
        help='LWC threshold (percent) for dynamic lookback (default: 1.0%%)'
    )
    
    parser.add_argument(
        '--max-lookback',
        type=float,
        default=72.0,
        help='Maximum lookback hours for dynamic search (default: 72.0)'
    )
    
    parser.add_argument(
        '--min-lookback',
        type=float,
        default=1.0,
        help='Minimum lookback hours (default: 1.0)'
    )
    
    parser.add_argument(
        '--disable-dynamic-lookback',
        action='store_true',
        help='Disable dynamic lookback, use fixed lookback instead'
    )
    
    parser.add_argument(
        '--height-tolerance',
        type=float,
        default=0.05,
        help='Height tolerance in meters (default: 0.05)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (default: all)',
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = MLDataCollectionConfig()
    config.stall_config.min_duration_hours = args.min_duration
    config.stall_config.height_tolerance_m = args.height_tolerance
    config.stall_config.feature_lookback_hours = args.lookback_hours
    
    # NEW: Configure dynamic lookback
    config.feature_config.use_dynamic_lookback = not args.disable_dynamic_lookback
    config.feature_config.lwc_threshold_pct = args.lwc_threshold
    config.feature_config.max_lookback_hours = args.max_lookback
    config.feature_config.min_lookback_hours = args.min_lookback
    config.feature_config.fallback_lookback_hours = args.lookback_hours
    
    config.start_date = args.start_date
    config.end_date = args.end_date
    config.output_dir = args.output
    config.max_files = args.max_files
    
    logger.info("\nConfiguration:")
    logger.info(f"  Input directory: {args.input}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Min stall duration: {args.min_duration}h")
    
    # NEW: Show dynamic lookback configuration
    if config.feature_config.use_dynamic_lookback:
        logger.info(f"  Feature lookback: DYNAMIC (LWC < {config.feature_config.lwc_threshold_pct}%)")
        logger.info(f"    Max lookback: {config.feature_config.max_lookback_hours}h")
        logger.info(f"    Min lookback: {config.feature_config.min_lookback_hours}h")
        logger.info(f"    Fallback: {config.feature_config.fallback_lookback_hours}h")
    else:
        logger.info(f"  Feature lookback: FIXED ({config.feature_config.fallback_lookback_hours}h)")
    
    logger.info(f"  Height tolerance: {args.height_tolerance}m")
    logger.info(f"  Negative sampling: {config.negative_sampling_strategy}")
    logger.info(f"  P:N ratio (target): 1:{config.negatives_per_positive}")
    if args.max_files:
        logger.info(f"  Max files: {args.max_files}")
    logger.info("")
    
    # Run collection
    try:
        stall_events_df, features_df = collect_ml_data(
            args.input,
            args.output,
            config
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("COLLECTION COMPLETE ✓")
        logger.info("=" * 80)
        logger.info(f"Stall events detected: {len(stall_events_df)}")
        logger.info(f"Total examples generated: {len(features_df)}")
        logger.info(f"Output directory: {args.output}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())