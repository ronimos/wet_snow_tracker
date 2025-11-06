# ML Branch Implementation Guide

## Quick Start

### 1. Create the Branch

```bash
# From your main branch
git checkout -b feature/ml-weak-layer-detection

# Create directory structure
mkdir -p src/wetting_front_tracker/ml_data_collection
mkdir -p data/ml_training
```

### 2. Copy Files

Copy these files into your project:

```bash
# Core ML modules
cp stall_detector.py src/wetting_front_tracker/ml_data_collection/
cp feature_extractor.py src/wetting_front_tracker/ml_data_collection/
cp collect_ml_data.py src/wetting_front_tracker/ml_data_collection/

# Create __init__.py
touch src/wetting_front_tracker/ml_data_collection/__init__.py
```

### 3. Integration Steps

**File: `src/wetting_front_tracker/ml_data_collection/__init__.py`**
```python
"""ML data collection package."""

from .stall_detector import StallDetector, StallEvent, StallDetectionConfig
from .feature_extractor import InterfaceFeatureExtractor, FeatureExtractionConfig

__all__ = [
    'StallDetector',
    'StallEvent',
    'StallDetectionConfig',
    'InterfaceFeatureExtractor',
    'FeatureExtractionConfig'
]
```

### 4. Update collect_ml_data.py

Replace the placeholder functions with real implementations:

**File: `src/wetting_front_tracker/ml_data_collection/collect_ml_data.py`**

```python
# Add these imports at the top
from ..snowpack_reader import SnowpackProfile
from ..wet_front_tracker import wet_front_lwc, find_wet_slab_loc_bottom_half

# Replace get_summary_from_pro_file()
def get_summary_from_pro_file(pro_file: Path, config: MLDataCollectionConfig) -> pd.DataFrame:
    """Get summary DataFrame from a .pro file."""
    try:
        # Load profile
        profile = SnowpackProfile(str(pro_file))
        
        # Calculate summary
        parameters_to_calculate = {
            "wet_front_lwc": wet_front_lwc,
            "weak_layer": find_wet_slab_loc_bottom_half
        }
        
        summary = profile.get_full_timeseries_summary(
            parameters_to_calculate=parameters_to_calculate,
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        # Unpack tuple columns if needed
        if 'wet_front_lwc' in summary.columns:
            summary[['wet_front_lwc_value', 'wet_front_lwc_height']] = pd.DataFrame(
                summary['wet_front_lwc'].tolist(), 
                index=summary.index
            )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting summary for {pro_file.name}: {e}")
        return pd.DataFrame()


# Replace get_profile_at_time()
def get_profile_at_time(pro_file: Path, timestamp: datetime) -> pd.DataFrame:
    """Get profile DataFrame at a specific timestamp."""
    try:
        profile = SnowpackProfile(str(pro_file))
        
        # Select data at specific time
        profile_at_time = profile.data.sel(
            timestamp=timestamp, 
            method='nearest'
        )
        
        # Convert to DataFrame
        profile_df = profile_at_time.to_dataframe().reset_index()
        
        return profile_df
        
    except Exception as e:
        logger.error(f"Error getting profile at {timestamp} for {pro_file.name}: {e}")
        return pd.DataFrame()
```

### 5. Test on Sample Data

```bash
# Run on a small subset first
python -m src.wetting_front_tracker.ml_data_collection.collect_ml_data \
    --input data/input \
    --output data/ml_training/test_run \
    --min-duration 12 \
    --start-date 2025-05-01 \
    --end-date 2025-05-31
```

## Expected Output

After running, you should have:

```
data/ml_training/
├── stall_events.csv              # All detected stall events
├── ml_training_dataset.csv       # Events with extracted features
├── collection_summary.txt        # Statistics report
└── ml_data_collection.log       # Detailed log
```

## Understanding the Output

### stall_events.csv

```csv
event_id,station_name,pro_file,start_time,end_time,stall_height,duration_hours,confidence,n_data_points,height_std,is_ongoing
SE_000001,Station_N,file.pro,2025-05-15T12:00,2025-05-16T06:00,1.25,18.0,0.87,12,0.023,False
```

**Key columns:**
- `event_id`: Unique identifier
- `stall_height`: Where the wetting front stalled (meters)
- `duration_hours`: How long it stalled
- `confidence`: Quality score (0-1)

### ml_training_dataset.csv

Includes all stall event columns PLUS 50+ feature columns:

**Height features:**
- `absolute_height`, `relative_height`, `in_bottom_half`

**Above interface features:**
- `above_density_mean`, `above_temperature_mean`, `above_grain_size_mean`
- `above_lwc_mean`, `above_has_facets`

**Interface features (gradients):**
- `interface_density_gradient`, `interface_temperature_gradient`
- `interface_grain_size_diff`, `interface_grain_type_change`

**Below interface features:**
- `below_density_mean`, `below_temperature_mean`, `below_grain_size_mean`

**Computed features:**
- `computed_density_contrast`, `computed_temperature_inversion`
- `computed_structural_weakness`

## Validation Steps

### 1. Check Data Quality

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/ml_training/ml_training_dataset.csv')

# Basic checks
print(f"Total events: {len(df)}")
print(f"Features: {len(df.columns)}")
print(f"\nMissing data:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Distribution checks
print(f"\nDuration distribution:")
print(df['duration_hours'].describe())

print(f"\nHeight distribution:")
print(df['stall_height'].describe())
```

### 2. Visualize Stalls

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/ml_training/ml_training_dataset.csv')

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Duration
axes[0,0].hist(df['duration_hours'], bins=30, edgecolor='black')
axes[0,0].set_xlabel('Duration (hours)')
axes[0,0].set_title('Stall Duration Distribution')

# Height
axes[0,1].hist(df['stall_height'], bins=30, edgecolor='black')
axes[0,1].set_xlabel('Height (m)')
axes[0,1].set_title('Stall Height Distribution')

# Confidence
axes[1,0].hist(df['confidence'], bins=20, edgecolor='black')
axes[1,0].set_xlabel('Confidence')
axes[1,0].set_title('Event Confidence Distribution')

# Height vs Duration
axes[1,1].scatter(df['stall_height'], df['duration_hours'], 
                  alpha=0.5, c=df['confidence'], cmap='viridis')
axes[1,1].set_xlabel('Stall Height (m)')
axes[1,1].set_ylabel('Duration (hours)')
axes[1,1].set_title('Height vs Duration (color = confidence)')
plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])

plt.tight_layout()
plt.savefig('data/ml_training/stall_distributions.png', dpi=150)
print("Saved visualization to stall_distributions.png")
```

### 3. Compare to Current LOC Rules

```python
# Add column to compare with current weak layer detection
df['current_loc_detected'] = False

for idx, row in df.iterrows():
    # Load profile and check if current rules find weak layer at stall height
    # This requires integration with your existing code
    pass

# Analyze overlap
overlap = df['current_loc_detected'].sum()
print(f"Stalls where current rules found weak layer: {overlap}/{len(df)} ({overlap/len(df)*100:.1f}%)")
```

## Troubleshooting

### Issue: No stall events detected

**Possible causes:**
1. LWC threshold too high
2. Duration requirement too long
3. No wetting events in date range

**Solutions:**
```bash
# Lower the requirements
python -m src.wetting_front_tracker.ml_data_collection.collect_ml_data \
    --min-duration 6 \
    --height-tolerance 0.10
```

### Issue: Many events but no features

**Possible causes:**
1. `get_profile_at_time()` not returning data
2. Missing columns in profile DataFrame
3. Timestamp alignment issues

**Solutions:**
- Add debug logging in `get_profile_at_time()`
- Check that profile has required columns: `height`, `density`, `temperature`, etc.
- Verify timestamp format matches

### Issue: High missing data percentage

**Expected:** Some features will have missing data (e.g., if hardness not in SNOWPACK output)

**Acceptable:** < 20% missing for critical features

**Solutions:**
- Document which features have high missing rates
- Consider feature imputation for ML phase
- Or exclude those features from model

## Next Steps After Data Collection

### Step 1: Exploratory Data Analysis (EDA)

Create a Jupyter notebook:

```python
# notebook: explore_stall_data.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/ml_training/ml_training_dataset.csv')

# 1. Basic statistics
print(df.describe())

# 2. Feature correlations
feature_cols = [col for col in df.columns if col.startswith(('above_', 'below_', 'interface_', 'computed_'))]
corr_matrix = df[feature_cols].corr()

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=150)

# 3. Feature importance (manual inspection)
# Which features differ most between long vs short stalls?
long_stalls = df[df['duration_hours'] > 24]
short_stalls = df[df['duration_hours'] < 18]

for col in feature_cols[:10]:  # Top 10 features
    if pd.notna(df[col]).sum() > 10:
        print(f"\n{col}:")
        print(f"  Long stalls: {long_stalls[col].mean():.3f}")
        print(f"  Short stalls: {short_stalls[col].mean():.3f}")
```

### Step 2: Add Labels

Decide on your labeling strategy:

**Option A: Current rule-based (bootstrap)**
```python
# Label based on whether current rules found weak layer
df['is_weak_layer'] = df['current_loc_detected'].astype(int)
```

**Option B: Duration-based (hypothesis)**
```python
# Hypothesis: longer stalls = weaker layers
df['is_weak_layer'] = (df['duration_hours'] > 18).astype(int)
```

**Option C: Multi-class (more nuanced)**
```python
# Classify into categories
df['weakness_class'] = pd.cut(
    df['duration_hours'],
    bins=[0, 12, 18, 24, np.inf],
    labels=['moderate', 'significant', 'strong', 'critical']
)
```

### Step 3: Feature Engineering

```python
# Create additional features
df['height_ratio'] = df['stall_height'] / df['context_total_snow_depth']

# Interaction features
df['density_temp_interaction'] = (
    df['computed_density_contrast'] * 
    df['computed_temperature_inversion']
)

# Polynomial features for key metrics
df['grain_size_diff_squared'] = df['interface_grain_size_diff'] ** 2
```

### Step 4: Prepare for ML

```python
# Save final dataset
df.to_csv('data/ml_training/final_training_dataset.csv', index=False)

# Create train/test split
from sklearn.model_selection import train_test_split

# Temporal split (recommended)
df = df.sort_values('start_time')
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

train_df.to_csv('data/ml_training/train.csv', index=False)
test_df.to_csv('data/ml_training/test.csv', index=False)

print(f"Train: {len(train_df)} events")
print(f"Test: {len(test_df)} events")
```

## Success Metrics (Phase 1)

Track these metrics:

- [ ] **Coverage:** ≥ 100 unique stations
- [ ] **Sample size:** ≥ 500 stall events
- [ ] **Feature count:** ≥ 40 features per event
- [ ] **Data quality:** < 10% missing for key features
- [ ] **Duration range:** Events from 12h to 72h+
- [ ] **Height range:** Events across full snowpack depth
- [ ] **Documentation:** All features documented
- [ ] **Validation:** EDA notebook completed

## Timeline

**Week 1:**
- Day 1-2: Integration with existing code
- Day 3-4: Test on sample data
- Day 5: Debug and refine

**Week 2:**
- Day 1-3: Full dataset collection
- Day 4-5: Data quality checks

**Week 3:**
- Day 1-3: Exploratory data analysis
- Day 4-5: Feature engineering

**Week 4:**
- Day 1-2: Labeling strategy
- Day 3-4: Documentation
- Day 5: Phase 1 completion report

## Common Questions

### Q: How many events do I need?
**A:** Aim for 500+ events minimum. More is better. If you have <200, results may not generalize well.

### Q: What if I have very few stalls?
**A:** Lower the duration threshold (try 6 hours instead of 12) or increase the height tolerance (±10cm instead of ±5cm).

### Q: Should I include failed stalls (wetting front continues)?
**A:** Yes! These are important negative examples. The wetting front passed through, so the interface wasn't weak enough to stop it.

### Q: How do I handle missing features?
**A:** Document which features have high missing rates. In Phase 2, you can either:
- Impute missing values (mean, median, model-based)
- Use models that handle missing data (some tree-based methods)
- Exclude features with >30% missing

### Q: What about seasonal effects?
**A:** Great observation! Consider adding:
- Month of year
- Day of season (relative to Oct 1)
- Cumulative precipitation/melt
- Snow age

## Resources

**Helpful files:**
- `ML_STRATEGY.md` - Overall strategy and phases
- `stall_detector.py` - Core detection algorithm
- `feature_extractor.py` - Feature engineering
- `collect_ml_data.py` - Main pipeline

**Next phase:**
- See `ML_STRATEGY.md` Phase 2 for model development

## Commit Strategy

```bash
# Commit incrementally as you build
git add src/wetting_front_tracker/ml_data_collection/
git commit -m "feat: add ML data collection modules"

# After successful test run
git add data/ml_training/
git commit -m "data: add test ML training dataset"

# After full collection
git commit -m "data: add complete ML training dataset (500+ events)"

# Create PR when ready
git push origin feature/ml-weak-layer-detection
```

## Getting Help

If you encounter issues:

1. **Check logs:** `ml_data_collection.log`
2. **Add debug prints:** In `get_summary_from_pro_file()` and `get_profile_at_time()`
3. **Test single file:** Process one .pro file manually to verify integration
4. **Visualize intermediate results:** Plot wetting front time series to verify detection

---

Good luck with Phase 1! This is an exciting approach to improving weak layer detection. 🚀
