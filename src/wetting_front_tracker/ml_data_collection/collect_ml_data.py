"""
collect_ml_data.py
==================

Main script for collecting ML training data from wetting front stalls.

This script:
1. Processes all .pro files
2. Detects wetting front stall events
3. Extracts interface features at stall locations
4. Saves training dataset to CSV

Usage:
    python collect_ml_data.py --input data/input --output data/ml_training

Author: [Your name]
Created: November 2025
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Imports from your existing code
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from .stall_detector import (
        StallDetector,
        StallDetectionConfig,
        extract_wetting_front_timeseries
    )
    from .feature_extractor import (
        InterfaceFeatureExtractor,
        FeatureExtractionConfig
    )
except ImportError:
    from stall_detector import (
        StallDetector,
        StallDetectionConfig,
        extract_wetting_front_timeseries
    )
    from feature_extractor import (
        InterfaceFeatureExtractor,
        FeatureExtractionConfig
    )
try:
    from ..snowpack_reader import SnowpackProfile
    from ..wet_front_tracker import wet_front_lwc
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from snowpack_reader import SnowpackProfile
    from wet_front_tracker import wet_front_lwc
    

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
            height_tolerance_m=0.05,
            min_lwc_threshold=0.04
        )
        
        # Feature extraction parameters
        self.feature_config = FeatureExtractionConfig(
            above_thickness=0.20,
            below_thickness=0.20,
            interface_thickness=0.04
        )
        
        # Analysis parameters
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        
        # Output configuration
        self.output_dir = Path('data/ml_training')
        self.save_intermediate = True  # Save stall events before features


# ---------------------------------------------------------------------------
# Integration Functions
# ---------------------------------------------------------------------------

def get_summary_from_pro_file(pro_file: Path, config: MLDataCollectionConfig) -> pd.DataFrame:
    """
    Get summary DataFrame from a .pro file.
    
    This is a placeholder - you'll need to integrate with your actual code.
    
    Args:
        pro_file: Path to .pro file
        config: Configuration
        
    Returns:
        Summary DataFrame with wet_front_lwc_height column
    """
    # TODO: Replace with your actual implementation
    
    profile = SnowpackProfile(str(pro_file))
    summary = profile.get_full_timeseries_summary(
        parameters_to_calculate={
            "wet_front_lwc": wet_front_lwc
        },
        start_date=config.start_date,
        end_date=config.end_date
    )
    return summary
    


def get_profile_at_time(pro_file: Path, timestamp: datetime) -> pd.DataFrame:
    """
    Get profile DataFrame at a specific timestamp.
    
    This is a placeholder - you'll need to integrate with your actual code.
    
    Args:
        pro_file: Path to .pro file
        timestamp: Time to extract profile
        
    Returns:
        DataFrame with height, density, temperature, etc.
    """
    # TODO: Replace with your actual implementation
    # from src.wetting_front_tracker.snowpack_reader import SnowpackProfile
    
    # profile = SnowpackProfile(str(pro_file))
    # single_profile = profile.data.sel(timestamp=timestamp, method='nearest')
    # return single_profile.to_dataframe()
    
    logger.warning("Using placeholder profile function - integrate with your code!")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def collect_ml_data(
    input_dir: Path,
    output_dir: Path,
    config: MLDataCollectionConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline for collecting ML training data.
    
    Args:
        input_dir: Directory containing .pro files
        output_dir: Directory to save results
        config: Configuration object
        
    Returns:
        Tuple of (stall_events_df, features_df)
    """
    logger.info("=" * 80)
    logger.info("ML Data Collection Pipeline")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .pro files
    pro_files = list(input_dir.glob('*.pro'))
    logger.info(f"Found {len(pro_files)} .pro files in {input_dir}")
    
    if not pro_files:
        logger.error("No .pro files found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Initialize detectors
    stall_detector = StallDetector(config.stall_config)
    feature_extractor = InterfaceFeatureExtractor(config.feature_config)
    
    # Phase 1: Detect stall events
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Detecting Stall Events")
    logger.info("=" * 80)
    
    all_stall_events = []
    
    for pro_file in tqdm(pro_files, desc="Detecting stalls"):
        try:
            # Get summary data
            summary_df = get_summary_from_pro_file(pro_file, config)
            
            if summary_df is None or summary_df.empty:
                continue
            
            # Extract wetting front
            wetting_front = extract_wetting_front_timeseries(summary_df)
            
            if wetting_front.empty:
                continue
            
            # Detect stalls
            station_name = pro_file.stem
            events = stall_detector.find_stalls(
                wetting_front,
                station_name,
                pro_file
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
    logger.info(f"  Mean duration: {stall_events_df['duration_hours'].mean():.1f} hours")
    logger.info(f"  Median duration: {stall_events_df['duration_hours'].median():.1f} hours")
    logger.info(f"  Mean height: {stall_events_df['stall_height'].mean():.2f} m")
    logger.info(f"  Mean confidence: {stall_events_df['confidence'].mean():.2f}")
    
    # Phase 2: Extract features
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Extracting Features")
    logger.info("=" * 80)
    
    all_features = []
    
    for event in tqdm(stall_events_df.to_dict('records'), 
                         total=len(stall_events_df),
                         desc="Extracting features"):
        try:
            # Get profile at stall time
            profile_df = get_profile_at_time(
                Path(event['pro_file']),
                event['start_time']
            )
            
            if profile_df is None or profile_df.empty:
                logger.warning(f"No profile for event {event['event_id']}")
                continue
            
            # Extract features
            features = feature_extractor.extract_all_features(
                profile_df,
                event['stall_height'],
                event['start_time']
            )
            
            # Combine with event metadata
            event_with_features = {**event, **features}
            all_features.append(event_with_features)
            
        except Exception as e:
            logger.error(f"Error extracting features for {event['event_id']}: {e}")
            continue
    
    # Convert to DataFrame
    if not all_features:
        logger.error("No features extracted!")
        return stall_events_df, pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)
    logger.info(f"\nExtracted features for {len(features_df)} events")
    logger.info(f"Total features per event: {len(features_df.columns) - len(stall_events_df.columns)}")
    
    # Save features
    features_path = output_dir / 'ml_training_dataset.csv'
    features_df.to_csv(features_path, index=False)
    logger.info(f"Saved training dataset to {features_path}")
    
    # Generate summary statistics
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
    report.append("=" * 80)
    report.append(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # Stall events statistics
    report.append("STALL EVENTS:")
    report.append(f"  Total events detected: {len(stall_events_df)}")
    report.append(f"  Unique stations: {stall_events_df['station_name'].nunique()}")
    report.append(f"  Duration (hours):")
    report.append(f"    Mean: {stall_events_df['duration_hours'].mean():.1f}")
    report.append(f"    Median: {stall_events_df['duration_hours'].median():.1f}")
    report.append(f"    Min: {stall_events_df['duration_hours'].min():.1f}")
    report.append(f"    Max: {stall_events_df['duration_hours'].max():.1f}")
    report.append(f"  Height (m):")
    report.append(f"    Mean: {stall_events_df['stall_height'].mean():.2f}")
    report.append(f"    Median: {stall_events_df['stall_height'].median():.2f}")
    report.append(f"  Confidence:")
    report.append(f"    Mean: {stall_events_df['confidence'].mean():.2f}")
    report.append(f"    Median: {stall_events_df['confidence'].median():.2f}")
    report.append("")
    
    # Features statistics
    if not features_df.empty:
        report.append("FEATURES:")
        report.append(f"  Events with features: {len(features_df)}")
        report.append(f"  Total feature columns: {len(features_df.columns) - len(stall_events_df.columns)}")
        report.append(f"  Missing data:")
        missing_pct = features_df.isnull().sum() / len(features_df) * 100
        high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
        if not high_missing.empty:
            report.append(f"    Features with >5% missing:")
            for feat, pct in high_missing.items():
                report.append(f"      {feat}: {pct:.1f}%")
        else:
            report.append(f"    All features have <5% missing data")
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
        description='Collect ML training data from wetting front stalls'
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = MLDataCollectionConfig()
    config.stall_config.min_duration_hours = args.min_duration
    config.stall_config.height_tolerance_m = args.height_tolerance
    config.start_date = args.start_date
    config.end_date = args.end_date
    config.output_dir = args.output
    
    # Run collection
    try:
        stall_events_df, features_df = collect_ml_data(
            args.input,
            args.output,
            config
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Stall events: {len(stall_events_df)}")
        logger.info(f"Events with features: {len(features_df)}")
        logger.info(f"Output directory: {args.output}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
