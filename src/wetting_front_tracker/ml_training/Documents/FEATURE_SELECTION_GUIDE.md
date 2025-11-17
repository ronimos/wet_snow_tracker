# Feature Selection Strategy Guide

## Your Question: "Would SHAP values be the best way to thin down uncontributing features?"

**Short Answer:** SHAP values are EXCELLENT for your use case, but I recommend a **multi-stage approach** for best results.

---

## Recommended Feature Selection Pipeline

### Stage 1: Statistical Filtering (Fast, Foundational)
**Time: Seconds**

```python
# Remove features with:
1. Zero or near-zero variance
2. High correlation (>0.95) with other features
3. Too many missing values (>50%)
```

**Why First:**
- Eliminates obvious noise
- Reduces computation for later stages
- No model fitting required

**Example:**
```python
# Remove features with variance < 0.01
# Remove one of each highly correlated pair (r > 0.95)
```

---

### Stage 2: Domain Knowledge (Your Expertise)
**Time: Minutes (thinking time)**

```python
# Keep features you KNOW matter for avalanche physics:
- Liquid water content (LWC) gradients
- Density contrasts at interfaces
- Grain size transitions
- Temperature gradients
- Layer thickness ratios
```

**Why Important:**
- You understand snow physics
- Some features may be important but rare (low variance)
- Prevents removing physically meaningful features

---

### Stage 3: Model-Agnostic Importance (Medium Cost)
**Time: Minutes**

```python
# Permutation Importance or Mutual Information
# - Works with any model
# - Good for initial ranking
# - Fast compared to SHAP
```

**Use This To:**
- Get initial ranking of remaining features
- Identify obviously unimportant features
- Inform which features to investigate with SHAP

---

### Stage 4: SHAP Values (High Value, More Expensive) ⭐
**Time: Minutes to hours depending on dataset size**

```python
# SHAP provides:
1. Feature importance WITH direction of effect
2. Interaction effects between features
3. Individual prediction explanations
4. Publication-quality visualizations
```

**Why SHAP is Perfect for You:**

#### 1. **Interpretability for Stakeholders**
```python
# You can show avalanche forecasters:
"When LWC gradient > 0.05, stall probability increases by 30%"
"Dense layer below + large grain size above = high risk"
```

#### 2. **Feature Interactions**
SHAP shows when features work together:
```python
# Example:
# Temperature alone: small effect
# LWC alone: medium effect  
# Temperature + LWC together: LARGE effect (wet snow at warm temps)
```

#### 3. **Local vs Global Importance**
```python
# Global: "This feature matters across all predictions"
# Local: "This feature caused THIS specific stall event"
```

#### 4. **Scientific Credibility**
- Based on game theory (Shapley values)
- Model-agnostic (works with any model)
- Mathematically rigorous
- Widely accepted in research

---

## Complete Feature Selection Example

```python
from model_trainer import ModelTrainer, FeatureSelector, FeatureImportanceAnalyzer
import pandas as pd

# Load your data
df = pd.read_csv('ml_training_dataset.csv')
X = df.drop(columns=['target', 'event_id', 'timestamp', ...])  # Features only
y = df['target']  # 0 = no stall, 1 = stall

print(f"Starting features: {X.shape[1]}")

# ============================================================================
# STAGE 1: Statistical Filtering
# ============================================================================
selector = FeatureSelector(
    variance_threshold=0.01,
    correlation_threshold=0.95
)

# Remove low variance
X = selector.remove_low_variance(X)
print(f"After variance filter: {X.shape[1]} features")

# Remove correlated
X = selector.remove_correlated(X)
print(f"After correlation filter: {X.shape[1]} features")

# ============================================================================
# STAGE 2: Domain Knowledge Filtering
# ============================================================================
# Keep critical avalanche features
critical_features = [
    'interface_lwc_diff',
    'interface_density_diff', 
    'interface_grain_size_ratio',
    'interface_temperature_gradient',
    'above_lwc',
    'below_density',
    # ... add features you KNOW matter
]

# Ensure critical features are retained
# (Don't drop them even if they seem unimportant statistically)
always_keep = [f for f in critical_features if f in X.columns]
print(f"Protecting {len(always_keep)} domain-critical features")

# ============================================================================
# STAGE 3: Train Model & Get Initial Importance
# ============================================================================
trainer = ModelTrainer()
trainer.fit(X, y)

# Get built-in importance (fast)
analyzer = FeatureImportanceAnalyzer(
    trainer.best_model_,
    trainer.X_train,
    trainer.feature_names_
)

builtin_importance = analyzer.get_builtin_importance()
print("\nTop 10 features (built-in):")
print(builtin_importance.head(10))

# ============================================================================
# STAGE 4: SHAP Analysis (DETAILED)
# ============================================================================
import shap
import matplotlib.pyplot as plt

# Compute SHAP values
explainer, shap_values = analyzer.get_shap_values(
    X_sample=trainer.X_val.head(200),  # Sample for speed
    sample_size=200
)

# Get SHAP-based importance
shap_importance = analyzer.get_shap_feature_importance(
    shap_values,
    trainer.X_val.head(200)
)

print("\nTop 10 features (SHAP):")
print(shap_importance.head(10))

# ============================================================================
# VISUALIZATION: Feature Importance Comparison
# ============================================================================

# 1. Summary plot (shows feature importance + direction)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    trainer.X_val.head(200),
    show=False
)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300)

# 2. Bar plot (just importance magnitude)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    trainer.X_val.head(200),
    plot_type='bar',
    show=False
)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=300)

# 3. Dependence plots (show how feature values affect predictions)
# Pick your top 3 features
top_features = shap_importance.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, feature in enumerate(top_features):
    plt.sca(axes[idx])
    shap.dependence_plot(
        feature,
        shap_values,
        trainer.X_val.head(200),
        show=False
    )
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=300)

# ============================================================================
# DECISION: Select Final Features
# ============================================================================

# Strategy 1: Keep top N by SHAP importance
top_n = 30
selected_features_shap = shap_importance.head(top_n).index.tolist()

# Strategy 2: Keep features above importance threshold
threshold = 0.01  # Adjust based on your data
selected_features_threshold = shap_importance[
    shap_importance > threshold
].index.tolist()

# Strategy 3: Ensemble - features important in multiple methods
from collections import Counter

all_top_features = (
    builtin_importance.head(30).index.tolist() +
    shap_importance.head(30).index.tolist()
)
feature_counts = Counter(all_top_features)

# Keep features that appear in both methods
selected_features_ensemble = [
    f for f, count in feature_counts.items() if count >= 2
]

# Add back critical domain features
final_features = list(set(
    selected_features_ensemble + always_keep
))

print(f"\n{'='*80}")
print(f"FINAL FEATURE SELECTION:")
print(f"{'='*80}")
print(f"Started with: {len(X.columns)} features")
print(f"Selected: {len(final_features)} features")
print(f"Reduction: {(1 - len(final_features)/len(X.columns))*100:.1f}%")
print(f"\nFinal features: {final_features}")

# ============================================================================
# RETRAIN with Selected Features
# ============================================================================
X_selected = X[final_features]
trainer_final = ModelTrainer()
trainer_final.fit(X_selected, y)

print(f"\n{'='*80}")
print(f"PERFORMANCE COMPARISON:")
print(f"{'='*80}")
print(f"All features ({len(X.columns)}): ROC-AUC = {trainer.results_[trainer.best_model_name_]['roc_auc']:.4f}")
print(f"Selected features ({len(final_features)}): ROC-AUC = {trainer_final.results_[trainer_final.best_model_name_]['roc_auc']:.4f}")
```

---

## When to Use Each Method

### Use Built-in Feature Importance When:
- ✅ Quick initial analysis
- ✅ Tree-based models (RF, XGBoost, LightGBM)
- ✅ Need fast iteration

### Use Permutation Importance When:
- ✅ Model-agnostic importance needed
- ✅ Want to verify built-in importance
- ✅ Non-tree models

### Use SHAP Values When: ⭐
- ✅ **Need interpretability for stakeholders** (you!)
- ✅ **Want to understand feature interactions** (physics!)
- ✅ **Need publication-quality explanations**
- ✅ **Working with regulators/forecasters who need transparency**
- ✅ Have computational resources (minutes to hours)
- ✅ Want to explain specific predictions

---

## SHAP Advantages for Avalanche Prediction

### 1. **Physical Interpretability**
```python
# SHAP shows HOW features contribute:
"High LWC gradient (0.05 vs 0.02) increases stall risk by 25%"

# Not just: "LWC gradient is important"
```

### 2. **Interaction Effects**
```python
# Snow science often has interactions:
# - Warm temperature + high LWC = critical
# - Cold temperature + high LWC = less critical
# SHAP reveals these!
```

### 3. **Trust from Forecasters**
```python
# Show forecasters:
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
# "This stall was predicted because of: X, Y, Z"
```

### 4. **Model Debugging**
```python
# If model predicts wrong:
# SHAP shows WHICH features caused the error
# Helps you improve features or get more data
```

---

## Recommended Workflow for You

```python
# Day 1: Statistical + Initial Model
1. Remove low variance, high correlation
2. Train baseline models
3. Get built-in importance

# Day 2: Domain Refinement  
4. Review features with snow science lens
5. Ensure critical physics features kept
6. Remove obviously irrelevant features

# Day 3: SHAP Deep Dive
7. Compute SHAP values (may take time)
8. Generate visualizations
9. Understand feature interactions
10. Select final feature set

# Day 4: Final Model
11. Retrain with selected features
12. Validate performance maintained/improved
13. Create documentation for forecasters
```

---

## Key Takeaways

1. **Yes, SHAP values are excellent for feature selection** - especially for your physics-based avalanche application

2. **But use a multi-stage approach:**
   - Statistics first (fast, removes noise)
   - Domain knowledge second (preserve physics)
   - SHAP last (deep understanding)

3. **SHAP is worth the computational cost because:**
   - Interpretability is CRITICAL for operational avalanche forecasting
   - You need to explain predictions to stakeholders
   - Physics-based features have interactions
   - Scientific credibility matters

4. **Don't just select features - understand them:**
   - Use SHAP plots to verify physics makes sense
   - Check for surprising interactions
   - Document findings for forecasters

---

## Tools Provided

The `model_trainer.py` module includes:
- ✅ `FeatureSelector` - statistical filtering
- ✅ `FeatureImportanceAnalyzer` - permutation + SHAP
- ✅ Visualization functions for all methods
- ✅ Complete pipeline integration

You're ready to go! 🚀
