"""
train_stall_predictor.py
=========================

Example script for training wetting front stall prediction models.

Usage:
    python train_stall_predictor.py --data ml_training_dataset.csv --output results/
    
Author: Ron Simenhois
Created: November 2025
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_trainer import (
    ModelTrainer,
    ModelConfig,
    FeatureSelector,
    FeatureImportanceAnalyzer,
    plot_model_comparison,
    plot_feature_importance,
    plot_shap_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: Path) -> tuple:
    """
    Load and prepare training data.
    
    Args:
        data_path: Path to CSV file with training data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Identify metadata columns (don't use for training)
    metadata_cols = [
        'event_id',
        'pro_file',
        'start_time',
        'end_time',
        'stall_layer_id',
        'layer_above_id',
        'layer_below_id',
        'feature_extraction_time',
        'lookback_hours'
    ]
    
    # Identify target column
    # Assuming binary classification: stalled (1) vs not stalled (0)
    # You may need to create this from your data
    if 'target' in df.columns:
        target_col = 'target'
    elif 'stalled' in df.columns:
        target_col = 'stalled'
    else:
        # Create target from stall duration if needed
        logger.warning("No target column found - you may need to create one")
        target_col = None
    
    # Extract features, target, and metadata
    metadata_cols_present = [c for c in metadata_cols if c in df.columns]
    metadata = df[metadata_cols_present].copy()
    
    feature_cols = [c for c in df.columns 
                   if c not in metadata_cols_present and c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col else None
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribution: {y.value_counts().to_dict() if y is not None else 'N/A'}")
    
    return X, y, metadata


def save_results(
    trainer: ModelTrainer,
    analyzer: FeatureImportanceAnalyzer,
    output_dir: Path
):
    """Save all results and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model comparison
    logger.info("Saving model comparison plot...")
    plot_model_comparison(
        trainer.results_,
        save_path=output_dir / 'model_comparison.png'
    )
    
    # Save feature importance
    logger.info("Saving feature importance plots...")
    if analyzer.importance_scores_:
        plot_feature_importance(
            analyzer.importance_scores_,
            top_n=30,
            save_path=output_dir / 'feature_importance.png'
        )
    
    # Save SHAP plots
    if 'shap_values' in analyzer.importance_scores_:
        logger.info("Saving SHAP plots...")
        
        # Summary plot
        plot_shap_summary(
            analyzer.importance_scores_['shap_values'],
            trainer.X_val.head(200),
            save_path=output_dir / 'shap_summary.png'
        )
        
        # Waterfall plot for a specific prediction
        try:
            import shap
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=analyzer.importance_scores_['shap_values'][0],
                    base_values=analyzer.importance_scores_['shap_explainer'].expected_value,
                    data=trainer.X_val.iloc[0].values,
                    feature_names=trainer.feature_names_
                ),
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_dir / 'shap_waterfall_example.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create SHAP waterfall plot: {e}")
    
    # Save feature importance rankings to CSV
    importance_df = pd.DataFrame()
    
    if 'builtin' in analyzer.importance_scores_:
        importance_df['builtin_importance'] = analyzer.importance_scores_['builtin']
    
    if 'permutation' in analyzer.importance_scores_:
        perm = analyzer.importance_scores_['permutation'].set_index('feature')
        importance_df['permutation_importance'] = perm['importance_mean']
    
    if 'shap' in analyzer.importance_scores_:
        importance_df['shap_importance'] = analyzer.importance_scores_['shap']
    
    importance_df = importance_df.fillna(0).sort_values(
        'shap_importance' if 'shap_importance' in importance_df.columns else importance_df.columns[0],
        ascending=False
    )
    importance_df.to_csv(output_dir / 'feature_importance_rankings.csv')
    logger.info(f"Saved feature rankings to {output_dir / 'feature_importance_rankings.csv'}")
    
    # Save model results summary
    results_summary = []
    for model_name, res in trainer.results_.items():
        if 'test' in model_name:
            continue
        results_summary.append({
            'model': model_name,
            'accuracy': res.get('accuracy', np.nan),
            'precision': res.get('precision', np.nan),
            'recall': res.get('recall', np.nan),
            'f1': res.get('f1', np.nan),
            'roc_auc': res.get('roc_auc', np.nan)
        })
    
    results_df = pd.DataFrame(results_summary).sort_values('roc_auc', ascending=False)
    results_df.to_csv(output_dir / 'model_results_summary.csv', index=False)
    logger.info(f"Saved model results to {output_dir / 'model_results_summary.csv'}")
    
    # Save best model test results
    test_key = f"{trainer.best_model_name_}_test"
    if test_key in trainer.results_:
        test_results = trainer.results_[test_key]
        with open(output_dir / 'best_model_test_results.txt', 'w') as f:
            f.write(f"Best Model: {trainer.best_model_name_}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Test Set Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy:  {test_results['accuracy']:.4f}\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall:    {test_results['recall']:.4f}\n")
            f.write(f"F1 Score:  {test_results['f1']:.4f}\n")
            if 'roc_auc' in test_results:
                f.write(f"ROC-AUC:   {test_results['roc_auc']:.4f}\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(test_results['confusion_matrix']) + "\n\n")
            f.write("-" * 80 + "\n")
            f.write("Classification Report:\n")
            f.write(test_results['classification_report'])
        
        logger.info(f"Saved test results to {output_dir / 'best_model_test_results.txt'}")
    
    # Save selected features
    if trainer.feature_names_:
        with open(output_dir / 'selected_features.txt', 'w') as f:
            f.write("Selected Features\n")
            f.write("=" * 80 + "\n\n")
            for i, feature in enumerate(trainer.feature_names_, 1):
                f.write(f"{i:3d}. {feature}\n")
        logger.info(f"Saved feature list to {output_dir / 'selected_features.txt'}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train wetting front stall prediction models'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting'],
        help='Models to train'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (faster)'
    )
    parser.add_argument(
        '--no-shap',
        action='store_true',
        help='Skip SHAP analysis (faster)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("WETTING FRONT STALL PREDICTION - MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Hyperparameter tuning: {not args.no_tune}")
    logger.info(f"SHAP analysis: {not args.no_shap}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info("=" * 80)
    
    # Load data
    X, y, metadata = load_and_prepare_data(args.data)
    
    if y is None:
        logger.error("No target variable found! Please ensure your data has a 'target' or 'stalled' column.")
        return
    
    # Configure training
    config = ModelConfig(
        models_to_train=args.models,
        tune_hyperparameters=not args.no_tune,
        compute_shap_values=not args.no_shap,
        cv_folds=args.cv_folds,
        use_time_series_cv=True  # Important for temporal data!
    )
    
    # Train models
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODELS")
    logger.info("=" * 80)
    
    trainer = ModelTrainer(config)
    trainer.fit(X, y)
    
    # Feature importance analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYZING FEATURE IMPORTANCE")
    logger.info("=" * 80)
    
    analyzer = FeatureImportanceAnalyzer(
        trainer.best_model_,
        trainer.X_train,
        trainer.feature_names_
    )
    
    importance_results = analyzer.analyze_all(
        trainer.X_val,
        trainer.y_val,
        compute_shap=not args.no_shap,
        shap_sample_size=min(200, len(trainer.X_val))
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Best model: {trainer.best_model_name_}")
    
    test_key = f"{trainer.best_model_name_}_test"
    if test_key in trainer.results_:
        test_res = trainer.results_[test_key]
        logger.info(f"Test accuracy: {test_res['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {test_res.get('roc_auc', 'N/A')}")
    
    # Top 10 features
    if 'shap' in importance_results:
        logger.info("\nTop 10 features (SHAP):")
        for i, (feat, score) in enumerate(importance_results['shap'].head(10).items(), 1):
            logger.info(f"  {i:2d}. {feat:40s} {score:.4f}")
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    save_results(trainer, analyzer, args.output)
    
    logger.info(f"\nAll results saved to: {args.output}")
    logger.info("\nFiles created:")
    logger.info("  - model_comparison.png")
    logger.info("  - feature_importance.png")
    logger.info("  - feature_importance_rankings.csv")
    logger.info("  - model_results_summary.csv")
    logger.info("  - best_model_test_results.txt")
    logger.info("  - selected_features.txt")
    if not args.no_shap:
        logger.info("  - shap_summary.png")
        logger.info("  - shap_waterfall_example.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
