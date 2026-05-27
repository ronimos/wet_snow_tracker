"""
predict_stall.py
================

Standalone script for making predictions using a trained wetting front stall model.

Usage:
    python predict_stall.py --model results/20251117_120000/trained_model --data new_data.csv --output predictions.csv
    
Author: Ron Simenhois
Created: November 2025
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load data for prediction.
    
    Args:
        data_path: Path to CSV file with features
        
    Returns:
        DataFrame with features
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Prepare features for prediction, ensuring all required features are present.
    
    Args:
        df: Input dataframe
        feature_names: List of feature names required by the model
        
    Returns:
        DataFrame with only the required features in the correct order
    """
    # Check if all required features are present
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        raise ValueError(f"Input data is missing required features: {missing_features}")
    
    # Select only the required features in the correct order
    X = df[feature_names].copy()
    
    logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
    return X


def make_predictions(
    model_dir: Path,
    data_path: Path,
    output_path: Path,
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Load model and make predictions on new data.
    
    Args:
        model_dir: Directory containing trained model
        data_path: Path to CSV file with features
        output_path: Path to save predictions
        include_metadata: Whether to include metadata columns in output
        
    Returns:
        DataFrame with predictions
    """
    # Load the trained model
    logger.info("=" * 80)
    logger.info("LOADING TRAINED MODEL")
    logger.info("=" * 80)
    trainer = ModelTrainer.load_model(model_dir)
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    df = load_data(data_path)
    
    # Identify metadata columns to preserve
    metadata_cols = []
    if include_metadata:
        potential_metadata = [
            'event_id', 'pro_file', 'start_time', 'end_time',
            'stall_layer_id', 'layer_above_id', 'layer_below_id',
            'feature_extraction_time', 'lookback_hours', 'station_name',
            'duration_hours', 'confidence', 'n_data_points', 'is_ongoing',
            'lookback_method', 'above_lwc_at_extraction', 'below_lwc_at_extraction',
            'distance_from_stall_m', 'example_type', 'requested_lookback_hours'
        ]
        metadata_cols = [col for col in potential_metadata if col in df.columns]
    
    # Store metadata if present
    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame()
    
    # Prepare features
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 80)
    X = prepare_features(df, trainer.feature_names_)
    
    # Make predictions
    logger.info("\n" + "=" * 80)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 80)
    
    predictions = trainer.predict(X)
    probabilities = trainer.predict_proba(X)
    
    logger.info(f"Generated predictions for {len(predictions)} samples")
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'probability_no_stall': probabilities[:, 0],
        'probability_stall': probabilities[:, 1]
    })
    
    # Add metadata columns if present
    if not metadata_df.empty:
        results = pd.concat([metadata_df.reset_index(drop=True), 
                           results.reset_index(drop=True)], axis=1)
    
    # Add prediction summary statistics
    n_stall = (predictions == 1).sum()
    n_no_stall = (predictions == 0).sum()
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Predicted stall:    {n_stall} ({100*n_stall/len(predictions):.1f}%)")
    logger.info(f"  Predicted no stall: {n_no_stall} ({100*n_no_stall/len(predictions):.1f}%)")
    logger.info(f"  Mean stall probability: {probabilities[:, 1].mean():.3f}")
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return results


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Make predictions using trained wetting front stall model'
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model directory (containing model.joblib, etc.)'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to CSV file with features for prediction'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('predictions.csv'),
        help='Output path for predictions CSV'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not include metadata columns in output'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        logger.error(f"Model directory not found: {args.model}")
        return
    
    if not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        return
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("WETTING FRONT STALL PREDICTION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Include metadata: {not args.no_metadata}")
    logger.info("=" * 80)
    
    # Make predictions
    try:
        results = make_predictions(
            args.model,
            args.data,
            args.output,
            include_metadata=not args.no_metadata
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output}")
        logger.info("\nFirst few predictions:")
        logger.info(results.head(10).to_string())
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()