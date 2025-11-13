"""
collect_ml_data.py
=====================

Main script for collecting ML training data from wetting front stalls.

This version:
- Uses layer ID-based stall tracking
- Extracts features 24 hours before stalling
- Collects all available SNOWPACK parameters
- Generates a balanced training dataset with positive and negative examples

Usage:
    python collect_ml_data.py --input data/input --output data/ml_training

Author: Ron Simenhois
Created: November 2025
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
# Use absolute imports based on your package structure
# This assumes your 'src' directory is in the Python path
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
    # Add a single fallback for running the script directly
    # This adds the 'src' directory to the path
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
    
# Try to import from main project
try:
    from wetting_front_tracker.snowpack_reader import SnowpackProfile
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc
except ImportError:
    # This block will be hit if the path wasn't added above
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from wetting_front_tracker.snowpack_reader import SnowpackProfile
    from wetting_front_tracker.wet_front_tracker import wet_front_lwc
    
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
            feature_lookback_hours=24.0
        )
        
        # Feature extraction parameters
        self.feature_config = FeatureExtractionConfig(
            lookback_hours=24.0,
            extract_all_parameters=True,
            compute_differences=True,
            compute_ratios=True,
            compute_gradients=True
        )
        
        # --- NEW: Balanced dataset configuration ---
        # 'combined': Use nearby and random negatives
        # 'nearby': Use only nearby negatives
        # 'random': Use only random negatives
        # 'none': Only collect positive examples
        self.negative_sampling_strategy: str = 'combined'
        
        # Target number of negative examples per positive example
        self.negatives_per_positive: int = 2  # 1:2 ratio
        
        # Distance (meters) to define a "nearby" interface for negative sampling
        self.nearby_distance_m: float = 0.05  # 5 cm
        
        # Min distance (meters) to define a "random" interface (avoids stall region)
        self.random_negative_distance_m: float = 0.10  # 10 cm
        # --- End of new config ---
        
        # Analysis parameters
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        
        # Output configuration
        self.output_dir = Path('data/ml_training')
        self.save_intermediate = True
        
        # Processing limits
        self.max_files: Optional[int] = None  # None = process all


# ---------------------------------------------------------------------------
# Integration Functions
# ---------------------------------------------------------------------------

def get_summary_from_pro_file(
    pro_file: Path,
    config: MLDataCollectionConfig
) -> pd.DataFrame:
    """
    Get summary DataFrame from a .pro file.
    
    Args:
        pro_file: Path to .pro file
        config: Configuration
        
    Returns:
        Summary DataFrame with wet_front_lwc_height column
    """
    try:
        # Load profile
        profile = SnowpackProfile(str(pro_file))
        
        # Calculate summary
        parameters_to_calculate = {
            "wet_front_lwc": wet_front_lwc
        }
        
        # Build kwargs conditionally
        summary_kwargs: dict[str, Any] = {
            'parameters_to_calculate': parameters_to_calculate
        }
        if config.start_date:
            summary_kwargs['start_date'] = config.start_date
        if config.end_date:
            summary_kwargs['end_date'] = config.end_date
        
        summary = profile.get_full_timeseries_summary(**summary_kwargs)
        
        # Unpack tuple columns if needed
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
    
    Args:
        profile_df: DataFrame of a profile at a single timestamp
        
    Returns:
        List of interface dicts, e.g.,
        [{'above_id': id_A, 'below_id': id_B, 'interface_height': h}, ...]
    """
    if 'height' not in profile_df.columns or 'element_ID' not in profile_df.columns:
        logger.warning("Profile missing height or element_ID, cannot get interfaces")
        return []
        
    if profile_df.empty:
        return []
        
    # Ensure profile is sorted by height
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
            # logger.warning(f"Error getting interface between layers {i} and {i+1}: {e}")
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
    
    Args:
        input_dir: Directory containing .pro files
        output_dir: Directory to save results
        config: Configuration object
        
    Returns:
        Tuple of (stall_events_df, features_df)
    """
    logger.info("=" * 80)
    logger.info("ML Data Collection Pipeline")
    logger.info("Layer ID-Based Tracking with Balanced Negative Sampling")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .pro files
    all_pro_files = list(input_dir.glob('*.pro'))
    
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
    
    for pro_file in tqdm(pro_files, desc="Detecting stalls"):
        try:
            # Load profile
            profile = SnowpackProfile(str(pro_file))
            
            # Get summary data
            summary_df = get_summary_from_pro_file(pro_file, config)
            
            if summary_df is None or summary_df.empty:
                logger.debug(f"No summary for {pro_file.name}")
                continue
            
            # Extract wetting front time series
            wetting_front = extract_wetting_front_timeseries(summary_df)
            
            if wetting_front.empty or wetting_front.notna().sum() == 0:
                logger.debug(f"No wetting front for {pro_file.name}")
                continue
            
            # Detect stalls with layer IDs
            station_name = pro_file.stem
            events = stall_detector.find_stalls(
                profile,
                wetting_front,
                station_name
            )
            
            all_stall_events.extend([e.to_dict() for e in events])
            
        except Exception as e:
            logger.error(f"Error processing {pro_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    if not all_stall_events:
        logger.error("No stall events detected!")
        return pd.DataFrame(), pd.DataFrame()
    
    stall_events_df = pd.DataFrame(all_stall_events)
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
    logger.info("PHASE 2: Extracting Features (Balanced Dataset)")
    logger.info(f"Strategy: {config.negative_sampling_strategy}, Ratio (Pos:Neg): 1:{config.negatives_per_positive}")
    logger.info("=" * 80)
    
    all_features = []
    failed_extractions = 0
    
    for event in tqdm(stall_events_df.to_dict('records'),
                     desc="Extracting features"):
        try:
            # Load profile
            profile = SnowpackProfile(str(event['pro_file']))
            
            # Define feature time (e.g., 24h before stall)
            feature_time = event['start_time'] - timedelta(
                hours=config.feature_config.lookback_hours
            )
            stall_interface_ids = {
                'above_id': event['layer_above_id'], 
                'below_id': event['layer_below_id']
            }
            stall_height = event['stall_height']

            # --- 1. Extract Positive Example ---
            positive_features = feature_extractor.extract_features_for_interface(
                profile,
                feature_time,
                event['stall_layer_id'],
                event['layer_above_id'],
                event['layer_below_id']
            )
            
            if not positive_features:
                logger.warning(f"No features extracted for positive event {event['event_id']}")
                failed_extractions += 1
                continue
            
            # Get actual time for negative sampling
            actual_feature_time_str = positive_features.get('feature_extraction_time')
            if actual_feature_time_str is None:
                logger.warning(f"No valid feature time for {event['event_id']}, skipping negatives")
                continue
            
            actual_feature_time = pd.to_datetime(actual_feature_time_str)
            actual_lookback = (event['start_time'] - actual_feature_time).total_seconds() / 3600
            
            # Add positive example
            event_with_features = {
                **event, 
                **positive_features,
                'stalled': 1,
                'example_type': 'positive_stall',
                'distance_from_stall_m': 0.0,
                'lookback_hours': actual_lookback # Override with actual
            }
            all_features.append(event_with_features)

            # --- 2. Extract Negative Examples ---
            if config.negative_sampling_strategy == 'none':
                continue

            # Get profile at the exact feature time
            try:
                profile_at_time = profile.data.sel(
                    timestamp=actual_feature_time, method='nearest'
                )
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
                # Prioritize nearby, then fill with random
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
                neg_features = feature_extractor.extract_features_for_interface(
                    profile,
                    feature_time, # Use same requested time
                    0, # Dummy stall_layer_id
                    neg_interface['above_id'],
                    neg_interface['below_id']
                )
                
                if not neg_features:
                    logger.warning(f"Failed to extract features for negative sample")
                    continue
                
                neg_event = {
                    **event, 
                    **neg_features,
                    'stalled': 0,
                    'distance_from_stall_m': neg_interface['interface_height'] - stall_height,
                    'lookback_hours': actual_lookback, # Use same actual lookback
                    # Overwrite key fields to reflect this interface
                    'layer_above_id': neg_interface['above_id'],
                    'layer_below_id': neg_interface['below_id'],
                    'stall_height': neg_interface['interface_height'], # Use interface height
                    'stall_layer_id': 0, # N/A
                }
                
                # Assign type
                if abs(neg_event['distance_from_stall_m']) <= config.nearby_distance_m:
                    neg_event['example_type'] = 'negative_nearby'
                else:
                    neg_event['example_type'] = 'negative_random'

                all_features.append(neg_event)
            
        except Exception as e:
            logger.error(f"Error extracting features for {event['event_id']}: {e}")
            failed_extractions += 1
            continue
    
    # Convert to DataFrame
    if not all_features:
        logger.error("No features extracted!")
        return stall_events_df, pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)
    
    logger.info(f"\nFeature Extraction Results:")
    logger.info(f"  Positive stall events: {len(stall_events_df)}")
    logger.info(f"  Total examples:        {len(features_df)}")
    if 'stalled' in features_df.columns:
        pos_count = (features_df['stalled'] == 1).sum()
        neg_count = (features_df['stalled'] == 0).sum()
        logger.info(f"    Positive (stalled=1):  {pos_count}")
        logger.info(f"    Negative (stalled=0):  {neg_count}")
        if pos_count > 0:
            ratio = neg_count / pos_count
            logger.info(f"    Actual P:N ratio:      1 : {ratio:.1f}")
            
    
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
    
    Args:
        stall_events_df: Stall events DataFrame
        features_df: Features DataFrame
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)
    
    report = []
    report.append("ML Training Data Collection Summary")
    report.append("Layer ID-Based Tracking with Balanced Negative Sampling")
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
        pos_count = (features_df['stalled'] == 1).sum()
        neg_count = (features_df['stalled'] == 0).sum()
        report.append(f"    Positive (stalled=1):  {pos_count}")
        report.append(f"    Negative (stalled=0):  {neg_count}")
        if pos_count > 0:
            ratio = neg_count / pos_count
            report.append(f"    Actual P:N ratio:      1 : {ratio:.1f}")
        
        if 'example_type' in features_df.columns:
            report.append(f"  Example type breakdown:")
            report.append(f"    {features_df['example_type'].value_counts().to_string()}")
            
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
        
        # Lookback times
        if 'lookback_hours' in features_df.columns:
            report.append(f"  Lookback times (hours):")
            report.append(f"    Mean: {features_df['lookback_hours'].mean():.1f}")
            report.append(f"    Min: {features_df['lookback_hours'].min():.1f}")
            report.append(f"    Max: {features_df['lookback_hours'].max():.1f}")
        
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
        description='Collect ML training data from wetting front stalls (Layer ID-based)'
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
        help='Hours to look back for feature extraction (default: 24.0)'
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
        # default=10
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
    config.feature_config.lookback_hours = args.lookback_hours
    config.start_date = args.start_date
    config.end_date = args.end_date
    config.output_dir = args.output
    config.max_files = args.max_files
    
    logger.info("\nConfiguration:")
    logger.info(f"  Input directory: {args.input}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Min stall duration: {args.min_duration}h")
    logger.info(f"  Feature lookback: {args.lookback_hours}h")
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