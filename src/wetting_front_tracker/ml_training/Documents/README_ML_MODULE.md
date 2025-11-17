# 🎯 ML Prediction Module for Wetting Front Stall Detection

**Complete machine learning pipeline for predicting wetting front stalls in avalanche forecasting**

---

## 📋 Quick Answer to Your Question

**Q: "Would SHAP values be the best way to thin down uncontributing features?"**

**A: YES!** SHAP values are excellent, especially for your avalanche application where interpretability is critical. However, use a **4-stage approach**:

1. **Statistical** (seconds) → Remove noise
2. **Domain knowledge** (your expertise) → Keep physics  
3. **Model-agnostic** (minutes) → Initial ranking
4. **SHAP** ⭐ (hours) → Deep understanding + final selection

See `FEATURE_SELECTION_GUIDE.md` for complete details!

---

## 📦 What's Included

### 🔧 Core Module (31 KB)
**`model_trainer.py`**
- Multiple model training (Random Forest, XGBoost, LightGBM, etc.)
- Hyperparameter tuning (Random/Grid search)
- Feature selection (statistical, permutation, SHAP)
- Time-series cross-validation
- Comprehensive evaluation
- SHAP integration

### 🚀 Ready-to-Use Script (12 KB)
**`train_stall_predictor.py`**
- Command-line interface
- Automated training pipeline
- Results visualization
- Report generation

### 📖 Documentation (34 KB total)
1. **`QUICKSTART.md`** (10 KB) - Get started in 5 minutes
2. **`FEATURE_SELECTION_GUIDE.md`** (12 KB) - Detailed strategy
3. **`ML_MODULE_SUMMARY.md`** (12 KB) - Complete overview

### 💡 Examples (12 KB)
**`example_workflow.py`** - End-to-end example showing:
- Data loading
- Statistical filtering
- Model training
- Feature selection with SHAP
- Results visualization

### 📦 Dependencies
**`requirements_ml.txt`** - All Python packages needed

---

## 🚀 Quick Start

### 1️⃣ Install
```bash
pip install -r requirements_ml.txt
```

### 2️⃣ Prepare Data
Your CSV needs:
- Feature columns (from `feature_extractor.py`)
- Target column: `target` (0=no stall, 1=stall)

### 3️⃣ Train
```bash
python train_stall_predictor.py \
    --data ml_training_dataset.csv \
    --output results/
```

### 4️⃣ Review Results
Check `results/` for:
- Model performance comparison
- Feature importance (including SHAP!)
- Test set evaluation
- SHAP visualizations

---

## 📚 File Guide

| File | Size | Purpose |
|------|------|---------|
| **model_trainer.py** | 31 KB | Core ML module with all functionality |
| **train_stall_predictor.py** | 12 KB | Command-line training script |
| **example_workflow.py** | 12 KB | Complete usage example |
| **QUICKSTART.md** | 10 KB | Get started quickly |
| **FEATURE_SELECTION_GUIDE.md** | 12 KB | Detailed SHAP strategy guide |
| **ML_MODULE_SUMMARY.md** | 12 KB | Complete package overview |
| **requirements_ml.txt** | <1 KB | Python dependencies |

**Total:** ~90 KB of production-ready code + documentation

---

## 🎓 Key Features

### ✅ Multiple Models
- Random Forest
- XGBoost  
- LightGBM
- Gradient Boosting
- Logistic Regression
- Extra Trees

### ✅ Feature Selection
- Variance filtering
- Correlation filtering
- Mutual information
- Permutation importance
- **SHAP values** ⭐

### ✅ Proper Validation
- Time-series cross-validation
- Stratified splitting
- Separate test set
- Class imbalance handling

### ✅ Interpretability
- SHAP summary plots
- SHAP dependence plots
- SHAP waterfall plots
- Feature interaction analysis

### ✅ Production Ready
- Save/load models
- Preprocessing pipeline
- Reproducible results
- Comprehensive logging

---

## 💡 Example Usage

### Basic Training
```bash
python train_stall_predictor.py --data dataset.csv --output results/
```

### Custom Configuration
```bash
python train_stall_predictor.py \
    --data dataset.csv \
    --output results/experiment_001 \
    --models random_forest xgboost lightgbm \
    --cv-folds 5
```

### Fast Mode (Testing)
```bash
python train_stall_predictor.py \
    --data dataset.csv \
    --output results/quick \
    --no-tune \
    --no-shap
```

### Python API
```python
from model_trainer import ModelTrainer
import pandas as pd

# Load data
df = pd.read_csv('dataset.csv')
X = df.drop(columns=['target'])
y = df['target']

# Train
trainer = ModelTrainer()
trainer.fit(X, y)

# Predict
predictions = trainer.predict(new_data)
probabilities = trainer.predict_proba(new_data)
```

---

## 🎯 Why SHAP for Your Application?

### 1. **Interpretability for Stakeholders** 🎯
```python
# SHAP shows HOW features contribute:
"High LWC gradient (0.08 vs 0.02) increases stall risk by 25%"

# Not just:
"LWC gradient is important"
```

### 2. **Feature Interactions** 🔗
```python
# Snow physics has interactions:
# - Warm temp + high LWC = critical
# - Cold temp + high LWC = less critical
# SHAP reveals these automatically!
```

### 3. **Individual Predictions** 🔍
```python
# Explain to forecasters:
"This stall was predicted because:
 1. LWC gradient = 0.08 (↑ risk +25%)
 2. Density contrast = 80 (↑ risk +15%)  
 3. Temperature = -2°C (↑ risk +10%)"
```

### 4. **Scientific Credibility** 📊
- Based on game theory (Shapley values)
- Model-agnostic
- Mathematically rigorous
- Publication-ready

---

## 📊 Expected Performance

For avalanche stall prediction:
- **Baseline accuracy:** 70-85%
- **ROC-AUC:** 0.75-0.90 (good models)
- **Focus:** High recall (catch all stalls!)

**Note:** Optimize for **recall** (safety critical) rather than precision.

---

## 🗺️ Recommended Workflow

### Day 1: Baseline
```bash
python train_stall_predictor.py --data dataset.csv --output baseline --no-tune
```

### Day 2: Full Training
```bash
python train_stall_predictor.py --data dataset.csv --output full_training
```

### Day 3: Feature Analysis
- Review SHAP plots
- Apply domain knowledge
- Select optimal features

### Day 4: Final Model
```bash
python train_stall_predictor.py --data dataset_selected.csv --output final_model
```

---

## 📖 Documentation Quick Links

- **New user?** Start with `QUICKSTART.md`
- **Want SHAP details?** Read `FEATURE_SELECTION_GUIDE.md`
- **Need complete overview?** Check `ML_MODULE_SUMMARY.md`
- **Want code example?** Run `example_workflow.py`

---

## 🔧 Advanced Features

### Model Comparison
Compare multiple models automatically:
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Cross-validation scores
- Confusion matrices
- Classification reports

### Feature Importance Methods
- Built-in (tree-based models)
- Permutation importance
- SHAP values (global)
- SHAP values (local/individual)
- Mutual information scores

### Hyperparameter Tuning
- Random search (faster)
- Grid search (exhaustive)
- Cross-validated selection
- Extensive parameter grids

### Time-Series Handling
- TimeSeriesSplit cross-validation
- Prevents future→past leakage
- Critical for operational forecasting

---

## ✅ Validation Checklist

Before deploying your model:

- [ ] Features make physical sense
- [ ] SHAP explanations align with snow physics
- [ ] High recall on test set (catch all stalls)
- [ ] Model generalizes to new seasons/locations
- [ ] Forecasters understand the predictions
- [ ] False alarm rate is acceptable
- [ ] Documentation is complete

---

## 🤝 Support

### Common Issues

**"SHAP not installed"**
```bash
pip install shap
```

**"Training too slow"**
- Use `--no-tune` flag
- Reduce `--cv-folds`
- Train fewer models

**"Memory error during SHAP"**
- Reduce sample size
- Use TreeExplainer (faster)
- Process in batches

### Tips

1. **Start small:** Test with `--no-tune --no-shap` first
2. **Use time-series CV:** Critical for temporal data
3. **Optimize recall:** Safety matters more than precision
4. **Trust SHAP:** It reveals true feature importance + interactions
5. **Keep physics:** Always include domain-critical features

---

## 🎓 Learning More

### SHAP Resources
- SHAP documentation: https://shap.readthedocs.io/
- Original paper: Lundberg & Lee (2017)
- GitHub: https://github.com/slundberg/shap

### Machine Learning for Avalanches
- Your work is cutting-edge!
- Few operational ML systems in avalanche forecasting
- Interpretability is KEY for adoption
- This module helps bridge that gap

---

## 📝 Summary

You asked about SHAP for feature selection. **Answer: YES!**

This complete ML module provides:
- ✅ Multiple model training & comparison
- ✅ Statistical feature filtering
- ✅ Permutation importance
- ✅ **SHAP values with visualizations** ⭐
- ✅ Time-series cross-validation
- ✅ Production-ready code
- ✅ Comprehensive documentation

**Everything is ready to use!** 🚀

---

## 📁 All Files

View all files at: `/mnt/user-data/outputs/`

```
model_trainer.py              31 KB  Core ML module
train_stall_predictor.py      12 KB  Training script  
example_workflow.py           12 KB  Complete example
QUICKSTART.md                 10 KB  Quick start guide
FEATURE_SELECTION_GUIDE.md    12 KB  SHAP strategy
ML_MODULE_SUMMARY.md          12 KB  Overview
requirements_ml.txt          <1 KB   Dependencies
README.md                     10 KB  This file
```

**Total: ~100 KB of production-ready ML code + docs**

---

**Ready to predict wetting front stalls! 🎿⛷️❄️**

Questions? Just ask!
