# Quick Reference: Model Training and Prediction Workflow

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                          │
└─────────────────────────────────────────────────────────────────┘

    ml_training_dataset.csv
            │
            ▼
    ┌──────────────────┐
    │ train_stall_     │
    │ predictor.py     │  (or train_fit_pipeline.py)
    └──────────────────┘
            │
            │ Uses ModelTrainer.fit()
            │
            ▼
    ┌──────────────────┐
    │  Trained Model   │
    └──────────────────┘
            │
            │ Calls save_model()
            ▼
    ┌───────────────────────────────────────┐
    │  results/my_model/trained_model/      │
    │  ├── model.joblib                     │
    │  ├── scaler.joblib                    │
    │  ├── feature_names.json               │
    │  ├── model_config.json                │
    │  └── metadata.json                    │
    └───────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                       PREDICTION PHASE                          │
└─────────────────────────────────────────────────────────────────┘

    new_data.csv          trained_model/
         │                      │
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
            ┌──────────────────┐
            │  predict_stall.py│
            └──────────────────┘
                    │
                    │ Uses ModelTrainer.load_model()
                    │ Calls predict() / predict_proba()
                    │
                    ▼
            ┌──────────────────┐
            │ predictions.csv  │
            │                  │
            │ prediction, ...  │
            │ 0, 0.82, 0.18   │
            │ 1, 0.35, 0.65   │
            └──────────────────┘
```

## Quick Command Reference

### 1. Training (Full Features)
```bash
# Train with all models and SHAP analysis
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/production_model \
    --models random_forest xgboost lightgbm gradient_boosting \
    --cv-folds 5

# Time: ~10-30 minutes depending on data size
```

### 2. Training (Fast)
```bash
# Quick training for testing
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/quick_test \
    --models random_forest \
    --no-tune \
    --no-shap

# Time: ~1-5 minutes
```

### 3. Making Predictions
```bash
# Use trained model on new data
python predict_stall.py \
    --model results/production_model/trained_model \
    --data new_observations.csv \
    --output predictions.csv

# Time: Seconds to minutes
```

### 4. Prediction Without Metadata
```bash
# Get only predictions, no metadata columns
python predict_stall.py \
    --model results/production_model/trained_model \
    --data new_observations.csv \
    --output predictions.csv \
    --no-metadata
```

## Python API Quick Reference

### Training
```python
import pandas as pd
from model_trainer import ModelTrainer, ModelConfig

# Load and prepare data
df = pd.read_csv('ml_training_dataset.csv')
X = df[feature_columns]
y = df['target']

# Configure and train
config = ModelConfig(
    models_to_train=['random_forest', 'xgboost'],
    tune_hyperparameters=True,
    compute_shap_values=True
)

trainer = ModelTrainer(config)
trainer.fit(X, y)

# Save for later use
trainer.save_model('models/my_model')
```

### Prediction
```python
import pandas as pd
from model_trainer import ModelTrainer

# Load saved model
trainer = ModelTrainer.load_model('models/my_model')

# Load new data
df_new = pd.read_csv('new_data.csv')

# Make predictions
predictions = trainer.predict(df_new)           # Returns: [0, 1, 0, 1, ...]
probabilities = trainer.predict_proba(df_new)   # Returns: [[0.8, 0.2], [0.3, 0.7], ...]

# Access individual components
print(f"Predicted class: {predictions[0]}")
print(f"P(no stall): {probabilities[0][0]:.3f}")
print(f"P(stall): {probabilities[0][1]:.3f}")
```

## File Checklist

### Before Training
- [ ] `ml_training_dataset.csv` with features and target column
- [ ] `train_stall_predictor.py` script
- [ ] `model_trainer.py` module

### After Training
- [ ] `results/my_model/trained_model/` directory
  - [ ] `model.joblib`
  - [ ] `scaler.joblib` (if scaling was used)
  - [ ] `feature_names.json`
  - [ ] `model_config.json`
  - [ ] `metadata.json`
- [ ] Evaluation plots and CSVs

### Before Prediction
- [ ] Trained model directory (from training)
- [ ] `new_data.csv` with same features as training
- [ ] `predict_stall.py` script
- [ ] `model_trainer.py` module

### After Prediction
- [ ] `predictions.csv` with predictions and probabilities

## Troubleshooting Quick Fixes

### Error: "Missing required features"
```bash
# Check what features are required
cat results/my_model/trained_model/feature_names.json

# Check what features you have
head -1 your_data.csv | tr ',' '\n'
```

### Error: "Directory does not exist"
```bash
# Verify model directory exists
ls results/my_model/trained_model/

# Should show: model.joblib, feature_names.json, etc.
```

### Want to see model details
```bash
# Check model metadata
cat results/my_model/trained_model/metadata.json

# Check training configuration
cat results/my_model/trained_model/model_config.json

# Check required features
cat results/my_model/trained_model/feature_names.json
```

## Feature Validation

The model automatically validates that new data contains all required features:

```python
# This happens automatically in predict_stall.py
required_features = json.load('trained_model/feature_names.json')
actual_features = new_data.columns.tolist()
missing = set(required_features) - set(actual_features)

if missing:
    raise ValueError(f"Missing features: {missing}")
```

## Best Practices

### ✅ DO
- Save models after training for reuse
- Validate predictions on test data before deployment
- Document which model version is in production
- Keep training data and model versions synchronized
- Check feature importance to ensure physical sense

### ❌ DON'T
- Use different feature engineering for prediction vs training
- Modify feature names between training and prediction
- Skip validation on new data
- Deploy without testing on held-out data

## Performance Tips

### For Faster Training
1. Use `--no-tune` to skip hyperparameter search
2. Use `--no-shap` to skip SHAP analysis
3. Reduce `--cv-folds` (e.g., 3 instead of 5)
4. Train fewer models (e.g., just `--models xgboost`)

### For Better Models
1. Use full hyperparameter tuning (default)
2. Include SHAP analysis (default)
3. Use more CV folds: `--cv-folds 10`
4. Train all models for comparison

### For Faster Predictions
- Predictions are already fast (seconds)
- For batch processing, load model once and predict multiple times

## Common Workflows

### Development Cycle
```bash
# 1. Quick test
python train_stall_predictor.py --data data.csv --output test --no-tune --no-shap

# 2. Validate predictions
python predict_stall.py --model test/trained_model --data validation.csv

# 3. Full training if results look good
python train_stall_predictor.py --data data.csv --output production
```

### Production Deployment
```bash
# 1. Train final model
python train_stall_predictor.py --data full_dataset.csv --output production_v1

# 2. Test on held-out data
python predict_stall.py --model production_v1/trained_model --data test_set.csv

# 3. Deploy if performance is acceptable
# 4. Use for operational predictions
python predict_stall.py --model production_v1/trained_model --data daily_data.csv
```

### Model Comparison
```bash
# Train multiple versions
python train_stall_predictor.py --data data.csv --output model_v1
python train_stall_predictor.py --data data_updated.csv --output model_v2

# Compare predictions
python predict_stall.py --model model_v1/trained_model --data test.csv --output pred_v1.csv
python predict_stall.py --model model_v2/trained_model --data test.csv --output pred_v2.csv

# Analyze differences
# ... (use pandas to compare pred_v1.csv and pred_v2.csv)
```

## Integration Examples

### As Part of a Pipeline
```python
# pipeline.py
from model_trainer import ModelTrainer
import pandas as pd

# Load model once at startup
MODEL = ModelTrainer.load_model('models/production')

def process_new_observation(data_dict):
    """Process a single new observation."""
    df = pd.DataFrame([data_dict])
    prediction = MODEL.predict(df)[0]
    probabilities = MODEL.predict_proba(df)[0]
    
    return {
        'prediction': int(prediction),
        'probability_no_stall': float(probabilities[0]),
        'probability_stall': float(probabilities[1])
    }

# Use in your application
result = process_new_observation({
    'above_lwc': 0.15,
    'below_density': 350.0,
    # ... all required features
})
```

### Batch Processing
```python
# batch_process.py
from model_trainer import ModelTrainer
import pandas as pd
from pathlib import Path

def process_directory(model_path, data_dir, output_dir):
    """Process all CSV files in a directory."""
    model = ModelTrainer.load_model(model_path)
    
    for csv_file in Path(data_dir).glob('*.csv'):
        df = pd.read_csv(csv_file)
        predictions = model.predict_proba(df)
        
        results = pd.DataFrame({
            'prediction': model.predict(df),
            'prob_no_stall': predictions[:, 0],
            'prob_stall': predictions[:, 1]
        })
        
        output_file = Path(output_dir) / f"{csv_file.stem}_predictions.csv"
        results.to_csv(output_file, index=False)
        print(f"Processed: {csv_file} -> {output_file}")

# Run batch processing
process_directory('models/production', 'data/incoming', 'data/predictions')
```

## Version Control

Consider tracking these in git:
- `model_trainer.py`
- `train_stall_predictor.py`
- `predict_stall.py`
- Training scripts and notebooks

Consider NOT tracking (too large):
- `results/*/trained_model/*.joblib`
- Large datasets

Instead, use a model registry or artifact storage for trained models.

## Summary

| Task | Script | Time | Output |
|------|--------|------|--------|
| Train model (fast) | `train_stall_predictor.py --no-tune` | 1-5 min | trained_model/ + plots |
| Train model (full) | `train_stall_predictor.py` | 10-30 min | trained_model/ + plots |
| Make predictions | `predict_stall.py` | seconds | predictions.csv |
| Load in Python | `ModelTrainer.load_model()` | instant | ModelTrainer object |

---

**Need more help?** See `README_MODEL_USAGE.md` for comprehensive documentation.
