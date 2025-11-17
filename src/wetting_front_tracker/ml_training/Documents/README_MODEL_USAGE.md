# Wetting Front Stall Prediction - Model Training and Prediction

This package provides a complete machine learning pipeline for training and deploying wetting front stall prediction models.

## Overview

The package now includes functionality to:
1. **Train** models on historical data with hyperparameter tuning
2. **Save** trained models for future use
3. **Load** saved models and make predictions on new data

## Files

### Core Modules
- **`model_trainer.py`**: Core ML training module with model saving/loading capabilities
  - `ModelTrainer` class: Train, tune, and compare multiple ML models
  - `FeatureSelector` class: Statistical feature selection
  - `FeatureImportanceAnalyzer` class: SHAP and permutation importance analysis
  - Model save/load functionality

### Training Scripts
- **`train_stall_predictor.py`**: Main training script with command-line interface
- **`train_fit_pipeline.py`**: Complete end-to-end workflow example

### Prediction Script
- **`predict_stall.py`**: Standalone script for making predictions with trained models

## Quick Start

### 1. Training a Model

Train a model on your dataset:

```bash
python train_stall_predictor.py \
    --data data/ml_training/ml_training_dataset.csv \
    --output results/my_model \
    --models random_forest xgboost lightgbm \
    --cv-folds 5
```

This will:
- Train multiple models (Random Forest, XGBoost, LightGBM)
- Perform hyperparameter tuning
- Compute SHAP feature importance
- Save the best model to `results/my_model/trained_model/`
- Generate evaluation plots and feature importance rankings

#### Training Options

```bash
python train_stall_predictor.py --help

Options:
  --data PATH           Path to training data CSV (required)
  --output PATH         Output directory for results
  --models [...]        Models to train (default: random_forest xgboost lightgbm gradient_boosting)
  --no-tune             Skip hyperparameter tuning (faster)
  --no-shap             Skip SHAP analysis (faster)
  --cv-folds N          Number of cross-validation folds (default: 5)
```

### 2. Making Predictions

Use a trained model to make predictions on new data:

```bash
python predict_stall.py \
    --model results/my_model/trained_model \
    --data new_data.csv \
    --output predictions.csv
```

This will:
- Load the trained model
- Validate that new data has all required features
- Make predictions (class and probabilities)
- Save results to `predictions.csv`

#### Prediction Options

```bash
python predict_stall.py --help

Options:
  --model PATH          Path to trained model directory (required)
  --data PATH           Path to CSV file with features for prediction (required)
  --output PATH         Output path for predictions CSV (default: predictions.csv)
  --no-metadata         Do not include metadata columns in output
```

## Saved Model Structure

When you train a model, it saves the following files:

```
results/my_model/trained_model/
├── model.joblib              # Trained scikit-learn/XGBoost/LightGBM model
├── scaler.joblib             # StandardScaler (if feature scaling was used)
├── feature_names.json        # List of features required for prediction
├── model_config.json         # Training configuration
└── metadata.json             # Model metadata (type, parameters, etc.)
```

## Required Features

The model requires specific features to make predictions. These features are:
- Automatically determined during training
- Saved in `feature_names.json`
- Validated when loading new data for prediction

If your new data is missing required features, the prediction script will raise an error showing which features are missing.

## Workflow Examples

### Example 1: Quick Training and Prediction

```bash
# Train a model
python train_stall_predictor.py \
    --data training_data.csv \
    --output results/quick_test \
    --no-tune \
    --no-shap

# Make predictions
python predict_stall.py \
    --model results/quick_test/trained_model \
    --data test_data.csv \
    --output test_predictions.csv
```

### Example 2: Full Training with All Features

```bash
# Train with full hyperparameter tuning and SHAP analysis
python train_stall_predictor.py \
    --data training_data.csv \
    --output results/full_model \
    --models random_forest xgboost lightgbm gradient_boosting \
    --cv-folds 5

# This will take longer but produce the best model
```

### Example 3: Using the Pipeline Script

```bash
# Run the complete end-to-end pipeline
python train_fit_pipeline.py
```

This script demonstrates:
- Data loading and preparation
- Feature filtering
- Model configuration
- Training with hyperparameter tuning
- Feature importance analysis
- Model saving
- Making predictions

## Output Files

### Training Outputs

After training, you'll find these files in your output directory:

```
results/my_model/
├── trained_model/                    # Saved model (for predictions)
├── model_comparison.png              # Performance comparison across models
├── feature_importance.png            # Feature importance from multiple methods
├── feature_importance_rankings.csv   # Detailed feature rankings
├── model_results_summary.csv         # Performance metrics for all models
├── best_model_test_results.txt       # Detailed test set results
├── selected_features.txt             # List of features used
├── shap_summary.png                  # SHAP summary plot
└── shap_waterfall_example.png        # SHAP waterfall for one prediction
```

### Prediction Outputs

The prediction output CSV includes:

```csv
prediction,probability_no_stall,probability_stall
0,0.82,0.18
1,0.35,0.65
...
```

Optional metadata columns (event_id, pro_file, etc.) are included by default if present in the input data.

## Feature Requirements

### Input Data Format

Your input data should be a CSV file with:
- Feature columns: Numeric values for model inputs
- Metadata columns (optional): event_id, pro_file, timestamps, etc.
- Target column (training only): Binary label (0 or 1)

### Feature Names

The model expects specific feature names that were present during training. Common features include:

- Interface features: `interface_lwc_diff`, `interface_density_diff`, `interface_temperature_gradient`
- Layer properties: `above_lwc`, `above_density`, `below_density`, `below_temperature`
- Grain characteristics: `above_grain_size`, `below_grain_size`, `interface_grain_size_ratio`
- And many more...

The exact list is saved in `trained_model/feature_names.json`.

## Programmatic Usage

### Training in Python

```python
from pathlib import Path
import pandas as pd
from model_trainer import ModelTrainer, ModelConfig

# Load data
df = pd.read_csv('training_data.csv')
X = df[feature_columns]
y = df['target']

# Configure training
config = ModelConfig(
    models_to_train=['random_forest', 'xgboost'],
    tune_hyperparameters=True,
    compute_shap_values=True,
    cv_folds=5
)

# Train models
trainer = ModelTrainer(config)
trainer.fit(X, y)

# Save model
trainer.save_model('models/my_model')
```

### Making Predictions in Python

```python
from model_trainer import ModelTrainer
import pandas as pd

# Load trained model
trainer = ModelTrainer.load_model('models/my_model')

# Load new data
df_new = pd.read_csv('new_data.csv')

# Make predictions
predictions = trainer.predict(df_new)
probabilities = trainer.predict_proba(df_new)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Best Practices

1. **Feature Engineering**: Ensure your new data uses the same feature engineering as training data
2. **Missing Values**: Handle missing values before prediction (model expects complete data)
3. **Feature Order**: Don't worry about column order - the model handles this automatically
4. **Validation**: Always validate predictions on a held-out test set before deployment
5. **Model Updates**: Retrain models periodically with new data to maintain performance

## Troubleshooting

### "Missing required features" Error

If you see this error, your new data is missing features that were present during training. Check:
1. Feature names in `trained_model/feature_names.json`
2. Column names in your new data
3. Feature engineering pipeline consistency

### "NaN values detected" Warning

The model automatically handles missing values via median imputation. However, it's better to handle missing values explicitly in your data pipeline.

### Model Performance Issues

If predictions seem poor:
1. Check feature distribution in new data vs. training data
2. Verify that metadata/irrelevant columns aren't being used as features
3. Consider retraining with more recent data
4. Review feature importance to ensure physically meaningful features are important

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm shap matplotlib seaborn joblib
```

## Support

For questions or issues with model training and prediction, check:
1. Feature names match between training and prediction data
2. Model directory contains all required files (model.joblib, feature_names.json, etc.)
3. Input data format matches training data format

## License

See LICENSE file for details.
