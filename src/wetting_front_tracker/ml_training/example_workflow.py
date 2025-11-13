"""
example_workflow.py
====================

Complete end-to-end example of the ML training workflow.
This shows how all the pieces fit together.

Author: Ron Simenhois
Created: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import our ML module
from wetting_front_tracker.ml_training.model_trainer import (
    ModelTrainer,
    ModelConfig,
    FeatureSelector,
    FeatureImportanceAnalyzer,
    plot_model_comparison,
    plot_feature_importance,
    plot_shap_summary
)

def main():
    """Complete ML workflow example."""
    
    print("=" * 80)
    print("WETTING FRONT STALL PREDICTION - COMPLETE WORKFLOW")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Load Your Data
    # =========================================================================
    print("\n[STEP 1] Loading data...")
    
    # Replace with your actual data file
    df = pd.read_csv('data/ml_training/ml_training_dataset.csv')
    
    print(f"  Loaded {len(df)} samples")
    print(f"  Total columns: {len(df.columns)}")
    
    # =========================================================================
    # STEP 2: Prepare Features and Target
    # =========================================================================
    print("\n[STEP 2] Preparing features and target...")
    
    # Metadata columns (don't use for training)
    metadata_cols = [
        'event_id', 'pro_file', 'start_time', 'end_time',
        'stall_layer_id', 'layer_above_id', 'layer_below_id',
        'feature_extraction_time', 'lookback_hours'
    ]
    
    # Target column
    target_col = 'target'  # Or 'stalled', depending on your data
    
    # Extract features
    feature_cols = [c for c in df.columns 
                   if c not in metadata_cols and c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # =========================================================================
    # STEP 3: Statistical Feature Filtering (Optional but Recommended)
    # =========================================================================
    print("\n[STEP 3] Statistical feature filtering...")
    
    selector = FeatureSelector(
        variance_threshold=0.01,
        correlation_threshold=0.95
    )
    
    # Remove low variance features
    X_filtered = selector.remove_low_variance(X, y)
    print(f"  After variance filter: {X_filtered.shape[1]} features "
          f"({X.shape[1] - X_filtered.shape[1]} removed)")
    
    # Remove highly correlated features
    X_filtered = selector.remove_correlated(X_filtered, y)
    print(f"  After correlation filter: {X_filtered.shape[1]} features "
          f"({X.shape[1] - X_filtered.shape[1]} total removed)")
    
    X = X_filtered
    
    # =========================================================================
    # STEP 4: Configure Training
    # =========================================================================
    print("\n[STEP 4] Configuring training...")
    
    config = ModelConfig(
        # Which models to train
        models_to_train=[
            'random_forest',
            'xgboost',
            'lightgbm',
            'gradient_boosting'
        ],
        
        # Hyperparameter tuning
        tune_hyperparameters=True,
        tuning_method='random',  # 'random' or 'grid'
        n_iter_random=30,  # Number of random search iterations
        
        # Cross-validation
        cv_folds=5,
        use_time_series_cv=True,  # Important for temporal data!
        
        # Data splitting
        test_size=0.2,
        validation_size=0.2,
        
        # Feature importance
        compute_shap_values=True,
        
        # Preprocessing
        scale_features=True,
        remove_low_variance=False,  # Already done manually
        remove_correlated=False     # Already done manually
    )
    
    print(f"  Models: {config.models_to_train}")
    print(f"  Hyperparameter tuning: {config.tune_hyperparameters}")
    print(f"  SHAP analysis: {config.compute_shap_values}")
    
    # =========================================================================
    # STEP 5: Train Models
    # =========================================================================
    print("\n[STEP 5] Training models...")
    print("  This may take several minutes...")
    
    trainer = ModelTrainer(config)
    trainer.fit(X, y)
    
    print(f"\n  Best model: {trainer.best_model_name_}")
    print(f"  Validation ROC-AUC: {trainer.results_[trainer.best_model_name_]['roc_auc']:.4f}")
    
    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\n[STEP 6] Test set evaluation...")
    
    test_key = f"{trainer.best_model_name_}_test"
    if test_key in trainer.results_:
        test_res = trainer.results_[test_key]
        print(f"  Accuracy:  {test_res['accuracy']:.4f}")
        print(f"  Precision: {test_res['precision']:.4f}")
        print(f"  Recall:    {test_res['recall']:.4f}")
        print(f"  F1 Score:  {test_res['f1']:.4f}")
        if 'roc_auc' in test_res:
            print(f"  ROC-AUC:   {test_res['roc_auc']:.4f}")
        
        print("\n  Confusion Matrix:")
        print(test_res['confusion_matrix'])
    
    # =========================================================================
    # STEP 7: Feature Importance Analysis
    # =========================================================================
    print("\n[STEP 7] Analyzing feature importance...")
    
    analyzer = FeatureImportanceAnalyzer(
        trainer.best_model_,
        trainer.X_train,
        trainer.feature_names_
    )
    
    # Compute all importance methods
    importance_results = analyzer.analyze_all(
        trainer.X_val,
        trainer.y_val,
        compute_shap=config.compute_shap_values,
        shap_sample_size=min(200, len(trainer.X_val))
    )
    
    # Show top 10 features by SHAP
    if 'shap' in importance_results:
        print("\n  Top 10 features (SHAP):")
        for i, (feat, score) in enumerate(importance_results['shap'].head(10).items(), 1):
            print(f"    {i:2d}. {feat:45s} {score:.4f}")
    
    # =========================================================================
    # STEP 8: Feature Selection (Optional)
    # =========================================================================
    print("\n[STEP 8] Feature selection based on SHAP...")
    
    if 'shap' in importance_results:
        # Strategy 1: Top N features
        top_n = 30
        top_features = importance_results['shap'].head(top_n).index.tolist()
        
        print(f"  Selected top {top_n} features by SHAP importance")
        
        # Strategy 2: Add domain-critical features
        # (Features you KNOW matter for avalanche physics)
        critical_features = [
            'interface_lwc_diff',
            'interface_density_diff',
            'interface_temperature_gradient',
            'interface_grain_size_ratio',
            'above_lwc',
            'below_density'
        ]
        
        # Combine: SHAP top features + critical physics features
        critical_present = [f for f in critical_features if f in X.columns]
        final_features = list(set(top_features + critical_present))
        
        print(f"  Added {len(critical_present)} domain-critical features")
        print(f"  Final feature count: {len(final_features)}")
        
        # Retrain with selected features
        print("\n  Retraining with selected features...")
        X_selected = X[final_features]
        
        trainer_refined = ModelTrainer(config)
        trainer_refined.fit(X_selected, y)
        
        print(f"\n  Performance comparison:")
        print(f"    All features ({X.shape[1]}):      "
              f"ROC-AUC = {trainer.results_[trainer.best_model_name_]['roc_auc']:.4f}")
        print(f"    Selected ({len(final_features)}): "
              f"ROC-AUC = {trainer_refined.results_[trainer_refined.best_model_name_]['roc_auc']:.4f}")
    
    # =========================================================================
    # STEP 9: Save Results
    # =========================================================================
    print("\n[STEP 9] Saving results...")
    
    output_dir = Path('results') / 'example_workflow'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot model comparison
    plot_model_comparison(
        trainer.results_,
        save_path=output_dir / 'model_comparison.png'
    )
    print(f"  Saved: {output_dir / 'model_comparison.png'}")
    
    # Plot feature importance
    plot_feature_importance(
        importance_results,
        top_n=30,
        save_path=output_dir / 'feature_importance.png'
    )
    print(f"  Saved: {output_dir / 'feature_importance.png'}")
    
    # SHAP summary plot
    if 'shap_values' in importance_results:
        plot_shap_summary(
            importance_results['shap_values'],
            trainer.X_val.head(200),
            save_path=output_dir / 'shap_summary.png'
        )
        print(f"  Saved: {output_dir / 'shap_summary.png'}")
    
    # Save feature importance rankings
    importance_df = pd.DataFrame({
        'feature': trainer.feature_names_,
        'shap_importance': importance_results['shap'].reindex(trainer.feature_names_).fillna(0)
    }).sort_values('shap_importance', ascending=False)
    
    importance_df.to_csv(output_dir / 'feature_rankings.csv', index=False)
    print(f"  Saved: {output_dir / 'feature_rankings.csv'}")
    
    # Save model results
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
    results_df.to_csv(output_dir / 'model_results.csv', index=False)
    print(f"  Saved: {output_dir / 'model_results.csv'}")
    
    # =========================================================================
    # STEP 10: Make Predictions on New Data (Example)
    # =========================================================================
    print("\n[STEP 10] Example: Making predictions on new data...")
    
    # Simulate new data (in practice, load from a file)
    new_data = X.sample(n=5, random_state=42)
    
    # Get predictions
    predictions = trainer.predict(new_data)
    probabilities = trainer.predict_proba(new_data)
    
    print("\n  Example predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"    Sample {i+1}: Prediction={pred}, "
              f"P(no_stall)={prob[0]:.3f}, P(stall)={prob[1]:.3f}")
    
    # =========================================================================
    # DONE!
    # =========================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review feature importance plots")
    print("  2. Verify important features make physical sense")
    print("  3. Select final feature set")
    print("  4. Retrain and deploy to production")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
