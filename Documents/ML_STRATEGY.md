# ML Data Collection Strategy - Wetting Front Stall Detection

## Project Goal

Replace rule-based LOC (Layer of Concern) detection with a machine learning model trained on actual wetting front behavior. The hypothesis: layers where the wetting front stalls for 12+ hours are the true weak layers.

## Rationale

**Current approach:** Rule-based detection using:
- Grain type codes (facets, depth hoar)
- Grain size differences
- Position in snowpack

**New approach:** Physics-based learning from:
- Actual water movement behavior
- Real stall events in the field
- Snowpack characteristics at stall interfaces

**Why this is better:**
- Learns from actual physical behavior
- May capture weak layers missed by rules
- Can identify non-obvious interfaces
- Data-driven rather than assumption-driven

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: Data Collection                   │
│                                                          │
│  .pro files → Stall Detection → Feature Extraction →    │
│  Training Dataset (CSV)                                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         PHASE 2: Model Development (Future)             │
│                                                          │
│  Training Data → ML Model Training → Model Validation   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         PHASE 3: Model Deployment (Future)              │
│                                                          │
│  Trained Model → Replace find_wet_slab_loc_bottom_half  │
└─────────────────────────────────────────────────────────┘
```

## Phase 1: Data Collection (Current Focus)

### Step 1: Define a Stall Event

**Definition:**
A wetting front "stall" occurs when:
1. The wetting front is at height H at time T
2. The wetting front remains within ±5cm of height H for ≥12 hours
3. The wetting front then continues downward OR the event ends

**Parameters (configurable):**
```python
STALL_DURATION_HOURS = 12      # Minimum stall duration
STALL_HEIGHT_TOLERANCE_CM = 5  # Height change tolerance
MIN_LWC_THRESHOLD = 0.04       # 4% - wetting front definition
```

### Step 2: What Data to Collect

For each stall event, collect:

**Temporal features:**
- Stall start time
- Stall end time (or end of available data)
- Duration of stall (hours)
- Time since last precipitation/melt event

**Location features:**
- Absolute height of stall (meters above ground)
- Relative height (% of total snow depth)
- Height of nearest weak layer above
- Height of nearest weak layer below

**Interface characteristics (at stall height):**
- **Above the interface (0-20cm above):**
  - Average density
  - Average temperature
  - Average grain size
  - Grain type distribution
  - Average LWC
  - Hardness
  - Ice content
  
- **At the interface (±2cm):**
  - Density gradient
  - Temperature gradient
  - Grain size difference
  - Grain type codes (both sides)
  - Hardness difference
  - Bond size
  - Coordination number
  
- **Below the interface (0-20cm below):**
  - Average density
  - Average temperature
  - Average grain size
  - Grain type distribution
  - Ice content
  - Hardness

**Snowpack context:**
- Total snow depth at stall time
- Number of layers in snowpack
- Position in bottom half? (boolean)
- Distance from ground (cm)
- Layers between interface and ground

**Weather/forcing (if available):**
- Air temperature at stall time
- Solar radiation
- Recent precipitation amount
- Wind speed

**Outcome (label for ML):**
- Was this interface identified as LOC by current rules? (boolean)
- Did an avalanche occur? (if known)
- Classification: [true_weak_layer, false_positive, uncertain]

### Step 3: Data Structure

**Primary table: `stall_events.csv`**
```csv
event_id,station_name,pro_file,stall_date,stall_height,duration_hours,...
SE_001,Station_N,file.pro,2025-05-15T12:00,1.25,18.5,...
SE_002,Station_E,file.pro,2025-05-16T06:00,0.85,14.2,...
```

**Supplementary table: `interface_features.csv`**
```csv
event_id,feature_name,value,layer_position
SE_001,density_above_avg,280.5,above
SE_001,density_below_avg,320.8,below
SE_001,grain_size_diff,0.85,interface
```

**Metadata table: `collection_metadata.csv`**
```csv
collection_date,n_files_processed,n_stalls_found,version
2025-11-05,150,342,1.0
```

## Phase 1 Implementation Plan

### Module Structure

```
src/wetting_front_tracker/
├── ml_data_collection/
│   ├── __init__.py
│   ├── stall_detector.py           # Find stall events
│   ├── feature_extractor.py        # Extract interface features
│   ├── data_aggregator.py          # Combine into training dataset
│   └── config.py                   # ML-specific configuration
├── main.py                         # Existing
└── ...                             # Other existing modules
```

### Development Steps

**Week 1: Stall Detection**
1. Create `stall_detector.py`
2. Implement wetting front tracking algorithm
3. Detect stall events (12+ hours, ±5cm)
4. Validate on sample files
5. Log statistics (stalls per file, duration distribution)

**Week 2: Feature Extraction**
1. Create `feature_extractor.py`
2. Extract all interface characteristics
3. Handle edge cases (bottom of snowpack, missing data)
4. Create feature engineering pipeline
5. Generate feature documentation

**Week 3: Data Aggregation**
1. Create `data_aggregator.py`
2. Combine stall events with features
3. Generate CSV datasets
4. Add data quality checks
5. Create visualization scripts

**Week 4: Validation & Documentation**
1. Process full dataset
2. Analyze collected data
3. Document feature importance (manual inspection)
4. Create data exploration notebook
5. Write ML roadmap document

### Code Architecture

**Key Classes:**

```python
class StallEvent:
    """Represents a single wetting front stall event."""
    event_id: str
    station_name: str
    pro_file: Path
    start_time: datetime
    end_time: datetime
    stall_height: float
    duration_hours: float
    features: Dict[str, float]

class StallDetector:
    """Detects stall events in snowpack profiles."""
    def find_stalls(self, profile: SnowpackProfile) -> List[StallEvent]
    def validate_stall(self, event: StallEvent) -> bool

class InterfaceFeatureExtractor:
    """Extracts features from snowpack interfaces."""
    def extract_all_features(self, profile: SnowpackProfile, 
                            height: float, time: datetime) -> Dict[str, float]
    def extract_above_features(self, ...) -> Dict[str, float]
    def extract_interface_features(self, ...) -> Dict[str, float]
    def extract_below_features(self, ...) -> Dict[str, float]

class DataAggregator:
    """Aggregates stall events into ML-ready datasets."""
    def collect_from_multiple_files(self, pro_files: List[Path]) -> pd.DataFrame
    def save_dataset(self, df: pd.DataFrame, output_path: Path)
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict
```

## Phase 2: Model Development (Future)

### ML Approach Options

**Option 1: Binary Classification**
- Input: Interface features (50-100 features)
- Output: Is this a weak layer? [Yes/No]
- Models to try: Random Forest, XGBoost, Neural Network

**Option 2: Stall Duration Prediction**
- Input: Interface features
- Output: Expected stall duration (hours)
- Use: Longer stalls → weaker layers

**Option 3: Multi-class Classification**
- Input: Interface features
- Output: [strong_interface, moderate_interface, weak_interface, critical_interface]

**Recommended: Start with Option 1**

### Feature Engineering Ideas

**Computed features:**
```python
# Density contrast
density_contrast = abs(density_above - density_below) / density_above

# Thermal inversion strength
thermal_inversion = temperature_below - temperature_above

# Structural weakness index
structural_weakness = grain_size_diff / grain_size_above

# Ice-over-weak index
ice_over_facets = (ice_content_above > 0.5) and (grain_type_below in WEAK_TYPES)

# Depth factor
depth_factor = stall_height / total_snow_depth
```

### Model Validation Strategy

**Train/Test Split:**
- Temporal: Train on 2022-2024, test on 2025
- Spatial: Train on stations 1-100, test on 101-150
- Random: 80/20 split with stratification

**Evaluation Metrics:**
- Precision: % of predicted weak layers that are actually weak
- Recall: % of actual weak layers that we found
- F1 Score: Harmonic mean of precision and recall
- ROC-AUC: Overall model discrimination ability

**Success Criteria:**
- Better recall than rule-based method (find more weak layers)
- Acceptable precision (not too many false positives)
- Generalizes to new stations/seasons

## Phase 3: Deployment (Future)

### Integration with Existing Code

**Replace this:**
```python
# In main.py
parameters_to_calculate = {
    "weak_layer": find_wet_slab_loc_bottom_half,  # OLD
    ...
}
```

**With this:**
```python
# In main.py
from .ml_models import MLWeakLayerDetector

ml_detector = MLWeakLayerDetector.load('models/weak_layer_v1.pkl')

parameters_to_calculate = {
    "weak_layer": ml_detector.predict_weak_layer,  # NEW
    ...
}
```

### Deployment Checklist

- [ ] Model performance exceeds rule-based method
- [ ] Model generalizes to unseen data
- [ ] Inference speed acceptable (<1s per profile)
- [ ] Model versioning system in place
- [ ] Fallback to rule-based if model fails
- [ ] Monitoring for model drift
- [ ] Documentation updated

## Data Quality Considerations

### Challenges

**1. Imbalanced Data**
- Most interfaces are NOT weak layers
- Stalls may be rare in dataset
- Solution: Oversample stall events, use class weights

**2. Missing Data**
- Not all SNOWPACK parameters always available
- Solution: Feature imputation, robust features

**3. Label Noise**
- Current rule-based method may mislabel some events
- Solution: Manual review of subset, multi-labeler agreement

**4. Temporal Autocorrelation**
- Same snowpack at T and T+1 hour are very similar
- Solution: Careful train/test split, time-based CV

### Data Collection Best Practices

**Do:**
- ✅ Collect from diverse conditions (wet/dry cycles, different elevations)
- ✅ Include negative examples (non-stall interfaces)
- ✅ Document data quality issues
- ✅ Version your dataset
- ✅ Create data exploration visualizations

**Don't:**
- ❌ Only collect from successful stall events
- ❌ Ignore missing data patterns
- ❌ Mix training and test data temporally
- ❌ Skip data quality checks
- ❌ Forget to document assumptions

## Recommended Tools

### Data Collection
```python
pandas        # Data manipulation
numpy         # Numerical operations
xarray        # Multi-dimensional data (already using)
```

### Data Exploration
```python
matplotlib    # Plotting (already using)
seaborn       # Statistical visualization
plotly        # Interactive plots (already using)
jupyter       # Notebooks for exploration
```

### ML Development (Phase 2)
```python
scikit-learn  # ML models, preprocessing
xgboost       # Gradient boosting
imbalanced-learn  # Handle class imbalance
shap          # Model interpretability
```

### Model Deployment (Phase 3)
```python
joblib        # Model serialization
mlflow        # Experiment tracking
pydantic      # Input validation
```

## Success Metrics

### Phase 1 (Data Collection)
- [ ] 500+ stall events collected
- [ ] 50+ features per event
- [ ] Data from 100+ unique .pro files
- [ ] <5% missing data per feature
- [ ] Documentation complete

### Phase 2 (Model Development)
- [ ] Model recall ≥ 90% (find most weak layers)
- [ ] Model precision ≥ 70% (limit false positives)
- [ ] Generalizes to new seasons/stations
- [ ] Inference time <1 second per profile

### Phase 3 (Deployment)
- [ ] Integrated into production pipeline
- [ ] A/B test shows improvement over rules
- [ ] No performance degradation
- [ ] Monitoring dashboard active

## Timeline Estimate

**Phase 1 (Data Collection): 4 weeks**
- Week 1: Stall detection algorithm
- Week 2: Feature extraction pipeline
- Week 3: Data aggregation and storage
- Week 4: Validation and documentation

**Phase 2 (Model Development): 6-8 weeks**
- Weeks 1-2: Data exploration and preprocessing
- Weeks 3-4: Model training and tuning
- Weeks 5-6: Validation and testing
- Weeks 7-8: Documentation and refinement

**Phase 3 (Deployment): 2-4 weeks**
- Weeks 1-2: Integration and testing
- Weeks 3-4: Monitoring setup and documentation

**Total: 12-16 weeks** for complete ML pipeline

## Next Steps

1. **Create branch:**
   ```bash
   git checkout -b feature/ml-weak-layer-detection
   ```

2. **Set up structure:**
   ```bash
   mkdir -p src/wetting_front_tracker/ml_data_collection
   mkdir -p data/ml_training
   ```

3. **Start with stall detector:**
   - Implement basic algorithm
   - Test on 5-10 .pro files
   - Verify stall detection logic

4. **Iterate:**
   - Review results
   - Refine parameters
   - Expand to more files

## Questions to Answer During Phase 1

- [ ] What % of .pro files contain stall events?
- [ ] What is the typical stall duration distribution?
- [ ] Do stalls correlate with current LOC rules?
- [ ] Which features show strongest differences between stall/no-stall?
- [ ] Are there temporal patterns in stalls?
- [ ] Do certain aspects/elevations have more stalls?
- [ ] What % of stalls are false positives (non-critical interfaces)?

## Documentation to Create

- [ ] Stall detection algorithm specification
- [ ] Feature dictionary (all 50+ features defined)
- [ ] Data collection report (statistics, visualizations)
- [ ] Data quality assessment
- [ ] Exploratory data analysis notebook
- [ ] ML roadmap document (Phases 2-3 details)

---

This strategy gives you a solid foundation for developing a physics-informed ML model. Start with Phase 1 data collection, and we can refine the approach as you learn from the data!
