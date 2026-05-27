# ML Prediction Module - Quick Start Guide

## Overview

You now have a complete ML pipeline for predicting wetting front stalls! Here's everything included:

### Files Created

1. **`model_trainer.py`** ⭐ - Core ML module
   - Multiple model training (RF, XGBoost, LightGBM, etc.)
   - Hyperparameter tuning
   - Feature selection (statistical, permutation, SHAP)
   - Cross-validation
   - Comprehensive evaluation

2. **`train_stall_predictor.py`** - Ready-to-use training script
   - Command-line interface
   - Automated pipeline
   - Results visualization
   - Report generation

3. **`FEATURE_SELECTION_GUIDE.md`** - Detailed feature selection strategies
   - Answers your SHAP question
   - Multi-stage approach
   - Best practices

4. **`requirements_ml.txt`** - Python dependencies

---

## Installation

```bash
# Install dependencies
pip install -r requirements_ml.txt

# Verify SHAP is installed (important!)
python -c "import shap; print('SHAP version:', shap.__version__)"
```

---

## Quick Start

### Step 1: Prepare Your Data

Your CSV should have:
- Feature columns (from feature_extractor.py)
- Target column (`target` or `stalled`): 0 = no stall, 1 = stall
- Optional metadata columns: `event_id`, `start_time`, etc.

Example structure:
```
event_id,start_time,above_lwc,below_density,interface_lwc_diff,...,target
evt_001,2024-01-15,0.05,320,0.02,...,1
evt_002,2024-01-16,0.01,280,-0.01,...,0
```

### Step 2: Train Models

```bash
# Basic training (with defaults)
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/

# Custom configuration
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/experiment_001 \
    --models random_forest xgboost lightgbm \
    --cv-folds 5

# Fast mode (no tuning, no SHAP - good for testing)
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/quick_test \
    --no-tune \
    --no-shap
```

### Step 3: Review Results

After training, check the output directory:

```
results/
├── model_comparison.png              # Compare all models
├── feature_importance.png            # Importance from multiple methods
├── feature_importance_rankings.csv   # Detailed feature rankings
├── model_results_summary.csv         # Performance metrics
├── best_model_test_results.txt       # Final test set evaluation
├── selected_features.txt             # Features used
├── shap_summary.png                  # SHAP summary plot
└── shap_waterfall_example.png        # Example prediction explanation
```

---

## Advanced Usage

### Custom Training Pipeline

```python
from model_trainer import ModelTrainer, ModelConfig
import pandas as pd

# Load data
df = pd.read_csv('ml_training_dataset.csv')
X = df.drop(columns=['target', 'event_id'])
y = df['target']

# Configure training
config = ModelConfig(
    models_to_train=['random_forest', 'xgboost', 'lightgbm'],
    tune_hyperparameters=True,
    compute_shap_values=True,
    cv_folds=5,
    use_time_series_cv=True  # Important for time series data!
)

# Train
trainer = ModelTrainer(config)
trainer.fit(X, y)

# Get best model
print(f"Best model: {trainer.best_model_name_}")
print(f"Test ROC-AUC: {trainer.results_[f'{trainer.best_model_name_}_test']['roc_auc']:.4f}")

# Make predictions
new_data = pd.read_csv('new_profiles.csv')
predictions = trainer.predict(new_data)
probabilities = trainer.predict_proba(new_data)
```

### Feature Selection

```python
from model_trainer import FeatureSelector, FeatureImportanceAnalyzer

# Statistical filtering
selector = FeatureSelector(
    variance_threshold=0.01,
    correlation_threshold=0.95
)

X_filtered = selector.remove_low_variance(X)
X_filtered = selector.remove_correlated(X_filtered)

# SHAP analysis
analyzer = FeatureImportanceAnalyzer(
    trainer.best_model_,
    trainer.X_train,
    trainer.feature_names_
)

importance = analyzer.analyze_all(
    trainer.X_val,
    trainer.y_val,
    compute_shap=True
)

# Get top features
top_features = importance['shap'].head(30).index.tolist()
X_top = X[top_features]

# Retrain with selected features
trainer_refined = ModelTrainer()
trainer_refined.fit(X_top, y)
```

### SHAP Visualizations

```python
import shap
import matplotlib.pyplot as plt

# Get SHAP values
explainer, shap_values = analyzer.get_shap_values(
    X_sample=trainer.X_val.head(200)
)

# Summary plot (global importance)
shap.summary_plot(shap_values, trainer.X_val.head(200))

# Bar plot (just magnitudes)
shap.summary_plot(shap_values, trainer.X_val.head(200), plot_type='bar')

# Dependence plot (how feature affects prediction)
shap.dependence_plot(
    'interface_lwc_diff',
    shap_values,
    trainer.X_val.head(200)
)

# Force plot (explain single prediction)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    trainer.X_val.iloc[0]
)

# Waterfall plot (another way to explain single prediction)
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=trainer.X_val.iloc[0].values,
        feature_names=trainer.feature_names_
    )
)
```

---

## Answer to Your Question: SHAP for Feature Selection

**Short answer: YES! SHAP values are excellent for thinning down features.**

**But use a multi-stage approach:**

1. **Stage 1: Statistical** (seconds)
   - Remove low variance features
   - Remove highly correlated features
   
2. **Stage 2: Domain Knowledge** (your expertise)
   - Keep physics-meaningful features
   - Consider rare but important events
   
3. **Stage 3: Model-Agnostic** (minutes)
   - Permutation importance
   - Mutual information
   
4. **Stage 4: SHAP** ⭐ (minutes to hours)
   - Model-specific importance
   - Feature interactions
   - Interpretability for stakeholders

**Why SHAP is perfect for avalanche prediction:**
- ✅ Shows HOW features contribute (not just IF)
- ✅ Reveals physical interactions (temp + LWC effects)
- ✅ Explains individual predictions to forecasters
- ✅ Publication-quality visualizations
- ✅ Scientifically rigorous (game theory based)

See `FEATURE_SELECTION_GUIDE.md` for detailed explanation and examples!

---

## Workflow Recommendation

### Day 1: Baseline
```bash
# Quick baseline with all features
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/baseline \
    --no-tune
```

### Day 2: Full Training
```bash
# Full training with hyperparameter tuning
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/full_training \
    --models random_forest xgboost lightgbm gradient_boosting
```

### Day 3: Feature Selection
```python
# Analyze feature importance (see examples above)
# Select top features based on SHAP + domain knowledge
# Create reduced feature set
```

### Day 4: Final Model
```bash
# Retrain with selected features
python train_stall_predictor.py \
    --data ml_training_dataset_selected.csv \
    --output results/final_model
```

---

## Tips & Best Practices

### 1. Time Series Cross-Validation
Your data is temporal - use `use_time_series_cv=True` (default):
```python
config = ModelConfig(use_time_series_cv=True)
```

### 2. Class Imbalance
If you have more non-stall events than stall events:
```python
# Models will handle this via class_weight='balanced'
# Already included in hyperparameter search spaces
```

### 3. SHAP Computation Time
- Start with small sample (100-200 examples)
- Increase if you have time/resources
- Tree models (RF, XGBoost) are fastest

### 4. Feature Scaling
- Automatically handled by `ModelTrainer`
- Only affects logistic regression, SVM
- Tree models don't need scaling

### 5. Missing Values
- Automatically imputed with median
- Or handle before training if you prefer

### 6. Save Models
```python
import joblib

# Save best model
joblib.dump(trainer.best_model_, 'best_model.pkl')

# Save scaler
joblib.dump(trainer.scaler_, 'scaler.pkl')

# Load later
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
```

---

## Expected Performance

For avalanche stall prediction, expect:
- **Baseline accuracy:** 70-85% (depending on class balance)
- **ROC-AUC:** 0.75-0.90 (good models)
- **Recall (stall detection):** Optimize for this! (safety critical)

**Note:** Predicting rare events (stalls) is challenging. Focus on:
1. High recall (catch all stalls, even if false alarms)
2. Interpretable features (forecasters need to trust it)
3. Physical plausibility (does the model make sense?)

---

## Troubleshooting

### "SHAP not available"
```bash
pip install shap
```

### "XGBoost not found"
```bash
pip install xgboost
```

### "Memory error during SHAP"
- Reduce sample size: `shap_sample_size=50`
- Use TreeExplainer (faster than KernelExplainer)
- Process in batches

### "Training too slow"
- Use `--no-tune` flag
- Reduce `--cv-folds` to 3
- Train fewer models
- Use smaller `n_iter_random`

---

## Next Steps

1. ✅ Collect training data (feature_extractor.py)
2. ✅ Train baseline models
3. ✅ Analyze feature importance with SHAP
4. ✅ Select optimal features
5. ✅ Train final model
6. ⬜ Deploy to operational system
7. ⬜ Monitor and retrain as needed

---

## Questions?

Key concepts explained:
- **ROC-AUC:** Area under receiver operating characteristic curve (0.5=random, 1.0=perfect)
- **Precision:** Of predicted stalls, how many were real? (avoid false alarms)
- **Recall:** Of real stalls, how many did we catch? (safety critical!)
- **F1:** Harmonic mean of precision and recall

For avalanche forecasting: **Optimize for RECALL** (catch all stalls!)

---

**Ready to predict wetting front stalls! 🎿⛷️**
