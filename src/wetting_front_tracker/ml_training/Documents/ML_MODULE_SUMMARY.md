# ML Prediction Module - Complete Package

## 🎯 Your Question Answered

**"Would SHAP values be the best way to thin down uncontributing features?"**

**Answer: YES!** SHAP values are excellent for feature selection, especially for your avalanche prediction application where interpretability is critical. However, I recommend a **4-stage approach** for best results:

1. **Statistical filtering** (fast) → Remove noise
2. **Domain knowledge** (your expertise) → Keep physics
3. **Model-agnostic importance** (medium) → Initial ranking  
4. **SHAP values** (comprehensive) → Deep understanding + final selection ⭐

See `FEATURE_SELECTION_GUIDE.md` for complete details!

---

## 📦 What You're Getting

### Core ML Module
**`model_trainer.py`** (1,000+ lines)
- ✅ Multiple model training & comparison
  - Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, Extra Trees
- ✅ Hyperparameter tuning (Grid/Random search)
- ✅ Feature selection (variance, correlation, mutual info, permutation, SHAP)
- ✅ Time-series cross-validation (critical for temporal data!)
- ✅ Comprehensive evaluation metrics
- ✅ SHAP integration for interpretability
- ✅ Visualization functions

**Classes:**
- `ModelConfig` - Configuration dataclass
- `FeatureSelector` - Statistical feature selection
- `ModelTrainer` - Main training pipeline
- `FeatureImportanceAnalyzer` - Multi-method importance analysis

### Ready-to-Use Script
**`train_stall_predictor.py`**
- Command-line interface
- Automated training pipeline
- Results generation & visualization
- Report creation

**Usage:**
```bash
python train_stall_predictor.py --data dataset.csv --output results/
```

### Documentation
1. **`QUICKSTART.md`** - Get started in 5 minutes
2. **`FEATURE_SELECTION_GUIDE.md`** - Detailed strategy guide
3. **`requirements_ml.txt`** - Dependencies

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_ml.txt
```

### 2. Prepare Your Data
Your CSV needs:
- Feature columns (from your `feature_extractor.py` output)
- Target column: `target` or `stalled` (0 = no stall, 1 = stall)
- Optional metadata: `event_id`, `start_time`, etc.

### 3. Train Models
```bash
# Basic training
python train_stall_predictor.py --data ml_training_dataset.csv --output results/

# With custom options
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/experiment_001 \
    --models random_forest xgboost lightgbm \
    --cv-folds 5
```

### 4. Review Results
Check `results/` for:
- Model performance comparison
- Feature importance rankings (including SHAP!)
- SHAP visualizations
- Test set evaluation
- Selected features list

---

## 🎓 Feature Selection: The Complete Answer

### Why SHAP is Perfect for You

**1. Interpretability for Stakeholders** 🎯
```python
# SHAP shows: "High LWC gradient increases stall risk by 30%"
# Not just: "LWC gradient is important"
```

**2. Feature Interactions** 🔗
```python
# Snow physics has interactions:
# - Warm temp + high LWC = critical
# - Cold temp + high LWC = less critical
# SHAP reveals these!
```

**3. Individual Predictions** 🔍
```python
# Explain to forecasters:
# "This stall was predicted because of:
#  1. LWC gradient = 0.08 (↑ risk +25%)
#  2. Density contrast = 80 kg/m³ (↑ risk +15%)
#  3. Temperature = -2°C (↑ risk +10%)"
```

**4. Scientific Credibility** 📊
- Based on game theory (Shapley values)
- Model-agnostic
- Mathematically rigorous
- Publication-ready

### Recommended Workflow

**Stage 1: Statistical (Seconds)**
```python
# Remove zero-variance features
# Remove highly correlated (r > 0.95)
# ↓ Reduces noise, speeds up later stages
```

**Stage 2: Domain Knowledge (Your Expertise)**
```python
# Always keep:
# - LWC gradients
# - Density contrasts
# - Temperature gradients
# - Grain size ratios
# ↓ Preserve physics even if low variance
```

**Stage 3: Permutation Importance (Minutes)**
```python
# Model-agnostic ranking
# Quick initial importance
# ↓ Identifies obviously unimportant features
```

**Stage 4: SHAP Analysis (Hours)**
```python
# Detailed importance WITH interactions
# Individual prediction explanations
# Publication-quality visualizations
# ↓ Final feature selection + interpretability
```

---

## 📊 Expected Outputs

### Model Comparison
```
Model               Accuracy  Precision  Recall  F1     ROC-AUC
-----------------------------------------------------------------
XGBoost             0.83      0.78       0.88    0.82   0.89
Random Forest       0.81      0.75       0.85    0.80   0.87
LightGBM            0.82      0.77       0.86    0.81   0.88
Gradient Boosting   0.80      0.73       0.84    0.78   0.85
Logistic Regression 0.75      0.68       0.79    0.73   0.80
```

### Feature Importance (Top 10)
```
Rank  Feature                    SHAP    Permutation  Built-in
-----------------------------------------------------------------
1     interface_lwc_diff         0.085   0.092        0.088
2     interface_density_diff     0.072   0.068        0.075
3     above_lwc                  0.065   0.071        0.062
4     interface_temp_gradient    0.058   0.055        0.059
5     interface_grain_size_ratio 0.052   0.049        0.054
...
```

### SHAP Visualizations
- Summary plot (global importance + distribution)
- Bar plot (importance magnitudes)
- Dependence plots (feature effects)
- Waterfall plots (individual predictions)
- Force plots (prediction explanations)

---

## 💡 Key Features of This Module

### 1. Time-Series Aware
```python
# Uses TimeSeriesSplit for CV
# Prevents data leakage from future → past
# Critical for operational forecasting!
```

### 2. Class Imbalance Handling
```python
# Automatically tunes class_weight='balanced'
# Optimizes for recall (catch all stalls)
# Important for rare event prediction
```

### 3. Comprehensive Evaluation
```python
# Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
# Confusion matrices
# Classification reports
# Validation + Test set evaluation
```

### 4. Production Ready
```python
# Save/load models with joblib
# Preprocessing pipeline included
# Reproducible with random seeds
```

### 5. Hyperparameter Tuning
```python
# Random or Grid search
# Extensive parameter grids for each model
# Cross-validated selection
```

---

## 🔧 Advanced Usage Examples

### Custom Feature Selection
```python
from model_trainer import ModelTrainer, FeatureImportanceAnalyzer
import pandas as pd

# Train model
trainer = ModelTrainer()
trainer.fit(X, y)

# Analyze importance
analyzer = FeatureImportanceAnalyzer(
    trainer.best_model_,
    trainer.X_train,
    trainer.feature_names_
)

# Get SHAP values
explainer, shap_values = analyzer.get_shap_values()
shap_importance = analyzer.get_shap_feature_importance(
    shap_values,
    trainer.X_val
)

# Select top 30 features
top_30 = shap_importance.head(30).index.tolist()

# Add domain-critical features
critical = ['interface_lwc_diff', 'interface_density_diff']
final_features = list(set(top_30 + critical))

# Retrain with selected features
X_selected = X[final_features]
trainer_final = ModelTrainer()
trainer_final.fit(X_selected, y)

print(f"Features reduced: {len(X.columns)} → {len(final_features)}")
print(f"Performance: {trainer_final.results_[trainer_final.best_model_name_]['roc_auc']:.4f}")
```

### Compare Multiple Feature Sets
```python
feature_sets = {
    'all_features': X.columns.tolist(),
    'top_50_shap': shap_importance.head(50).index.tolist(),
    'top_30_shap': shap_importance.head(30).index.tolist(),
    'domain_critical': critical_physics_features,
    'shap_plus_domain': list(set(shap_importance.head(25).index.tolist() + critical_physics_features))
}

results = {}
for name, features in feature_sets.items():
    trainer = ModelTrainer()
    trainer.fit(X[features], y)
    results[name] = {
        'n_features': len(features),
        'roc_auc': trainer.results_[trainer.best_model_name_]['roc_auc'],
        'recall': trainer.results_[trainer.best_model_name_]['recall']
    }

# Compare
comparison_df = pd.DataFrame(results).T
print(comparison_df.sort_values('roc_auc', ascending=False))
```

---

## 🎯 For Your Avalanche Application

### Optimize for Recall (Safety Critical!)
```python
# Stall detection is safety-critical
# Better to have false alarms than miss real stalls

# Models automatically tune for class_weight='balanced'
# But you can also set custom thresholds:

# Instead of 0.5 probability threshold:
predictions = (probabilities[:, 1] > 0.3).astype(int)  # Lower threshold = higher recall
```

### Interpretability Matters
```python
# Forecasters need to TRUST the model
# SHAP provides this trust:

# For each prediction, show:
# 1. Which features drove the prediction
# 2. How much each contributed
# 3. Whether it aligns with physics

# This builds confidence and adoption!
```

### Physics Validation
```python
# After feature selection, verify physics:
# - Do important features make sense?
# - Are interactions physically plausible?
# - Can you explain to non-ML avalanche experts?

# If not, investigate:
# - Data quality issues
# - Feature engineering problems
# - Model selection issues
```

---

## 📁 Files Included

```
/mnt/user-data/outputs/
├── model_trainer.py              ⭐ Core ML module (1000+ lines)
├── train_stall_predictor.py      ⭐ Ready-to-use training script
├── QUICKSTART.md                 📖 5-minute getting started
├── FEATURE_SELECTION_GUIDE.md    📖 Detailed strategy guide
├── requirements_ml.txt           📦 Python dependencies
└── ML_MODULE_SUMMARY.md          📋 This file
```

---

## 🎓 Learning Resources

### Understanding SHAP
- Original paper: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
- SHAP GitHub: https://github.com/slundberg/shap
- Tutorial: https://shap.readthedocs.io/

### Model Selection
- For tabular data like yours: Tree-based models (RF, XGBoost, LightGBM) typically best
- XGBoost often wins in competitions
- LightGBM faster for large datasets
- Random Forest most interpretable

### Avalanche Forecasting with ML
- Your work is cutting-edge!
- Few ML applications in operational avalanche forecasting
- Interpretability is KEY for adoption
- SHAP will help bridge ML ↔ forecasters

---

## ✅ What You Can Do Now

1. **Install dependencies**
   ```bash
   pip install -r requirements_ml.txt
   ```

2. **Prepare your data**
   - Run your `feature_extractor.py` on stall events
   - Create target variable (0=no stall, 1=stall)
   - Save as CSV

3. **Train baseline**
   ```bash
   python train_stall_predictor.py --data dataset.csv --output results/baseline --no-tune
   ```

4. **Analyze features with SHAP**
   ```bash
   python train_stall_predictor.py --data dataset.csv --output results/full_analysis
   ```

5. **Select best features**
   - Review SHAP plots
   - Apply domain knowledge
   - Create reduced feature set

6. **Train final model**
   ```bash
   python train_stall_predictor.py --data dataset_selected.csv --output results/final
   ```

---

## 🤝 Summary

You asked: **"Would SHAP values be the best way to thin down uncontributing features?"**

My answer: **Yes, with a 4-stage approach:**

| Stage | Method | Time | Purpose |
|-------|--------|------|---------|
| 1 | Statistical | Seconds | Remove noise |
| 2 | Domain | Minutes | Keep physics |
| 3 | Permutation | Minutes | Initial ranking |
| 4 | **SHAP** ⭐ | Hours | Deep understanding |

**Why SHAP is perfect for you:**
- ✅ Interpretability for operational forecasting
- ✅ Reveals feature interactions (physics!)
- ✅ Explains individual predictions
- ✅ Publication-ready visualizations
- ✅ Builds trust with stakeholders

**Everything is ready to use!** 🚀

---

**Questions? Just ask! Ready to predict wetting front stalls! 🎿⛷️❄️**
