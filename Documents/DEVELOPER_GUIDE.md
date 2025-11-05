# Developer Guide - Wetting Front Tracker

Guide for developers who want to understand, modify, or extend the Wetting Front Tracker codebase.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Organization](#code-organization)
3. [Key Design Patterns](#key-design-patterns)
4. [Adding New Features](#adding-new-features)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Debugging Tips](#debugging-tips)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────┐
│  Command Line   │
│   Arguments     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   main.py       │ ◄──── Orchestrates entire workflow
│   - Parse args  │
│   - Setup data  │
│   - Process     │
│   - Visualize   │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────────┐
    │         │            │              │
    ▼         ▼            ▼              ▼
┌─────┐  ┌──────┐    ┌──────────┐   ┌──────────┐
│util │  │geodat│    │snowpack  │   │plotting  │
│.py  │  │a.py  │    │_reader.py│   │.py       │
└─────┘  └──────┘    └──────────┘   └──────────┘
         │                │              │
         │                ▼              │
         │         ┌──────────────┐      │
         │         │wet_front_    │      │
         └────────►│tracker.py    │◄─────┘
                   │- Analysis    │
                   │  functions   │
                   └──────────────┘
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | Orchestration, parallel processing | `process_single_profile()`, `worker_wrapper()` |
| `snowpack_reader.py` | .pro file parsing, data management | `SnowpackProfile` class |
| `wet_front_tracker.py` | Analysis algorithms | `find_time_to_loc()`, `avg_lwc_above_weak()` |
| `plotting.py` | Visualization generation | `plot_summary_matplotlib()`, `create_folium_map()` |
| `prepare_geodata.py` | Geospatial preprocessing | `prepare_aspect_polygons()`, `link_polygons_to_pro_files()` |
| `util.py` | Helper utilities | `parse_pro_file()` |
| `param_config.py` | Configuration management | `Config` class |

---

## Code Organization

### Directory Structure

```
src/wetting_front_tracker/
├── __init__.py
├── main.py                 # Entry point
├── snowpack_reader.py      # Data I/O (900+ lines)
├── wet_front_tracker.py    # Analysis (650+ lines)
├── plotting.py             # Visualization (1200+ lines)
├── prepare_geodata.py      # Geospatial (1000+ lines)
├── param_config.py         # Configuration (300+ lines)
└── util.py                 # Utilities (150+ lines)
```

### Import Dependencies

```
main.py
  ├── param_config
  ├── plotting
  ├── prepare_geodata
  ├── snowpack_reader
  └── wet_front_tracker

plotting.py
  └── param_config

prepare_geodata.py
  └── param_config

snowpack_reader.py
  └── (no internal dependencies)

wet_front_tracker.py
  └── (no internal dependencies)
```

### Key Classes

#### SnowpackProfile

```python
class SnowpackProfile:
    """
    Main interface for SNOWPACK data.
    
    Attributes:
        data: xarray.Dataset with all snowpack variables
        metadata: dict with station information
        
    Methods:
        get_full_timeseries_summary() - Calculate metrics over time
        slice_data() - Filter by date range
        calculate_rc_flat() - Compute stability indices
    """
```

**Usage Pattern:**
```python
profile = SnowpackProfile("path/to/file.pro", aspect="N")
summary = profile.get_full_timeseries_summary(
    parameters_to_calculate={
        "metric_name": calculation_function
    }
)
```

---

## Key Design Patterns

### 1. Plugin-Based Analysis

**Pattern:** Analysis functions are passed as parameters

```python
# Define analysis functions
def my_metric(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Calculate custom metric from daily profile."""
    # Your analysis here
    return value, height

# Register with calculator
parameters_to_calculate = {
    "my_metric": my_metric
}

# Automatic calculation across all timesteps
summary = profile.get_full_timeseries_summary(
    parameters_to_calculate=parameters_to_calculate
)
```

**Why:** Allows easy extension without modifying core code.

### 2. Functional Analysis Functions

**Pattern:** Stateless functions that operate on DataFrames

```python
def analyze_layer(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Analyze single timestep.
    
    Args:
        df: DataFrame with columns [height, lwc, density, grain_type, ...]
        
    Returns:
        Tuple of (value, height) or (None, None)
    """
    # Validate
    if not _validate_dataframe(df, required_cols):
        return None, None
    
    # Calculate
    result = df.loc[condition, 'column'].values[0]
    height = df.loc[condition, 'height'].values[0]
    
    return float(result), float(height)
```

**Benefits:**
- Easy to test
- Composable
- Parallelizable

### 3. Configuration Object Pattern

**Pattern:** Centralized configuration with dataclasses

```python
@dataclass
class Paths:
    base_path: Path
    input_path: Path
    # ...

class Config:
    def __init__(self):
        self.paths = Paths(...)
        self.analysis = AnalysisParams()
        
# Global instance
config = Config()
```

**Access:**
```python
from .param_config import config
results_path = config.paths.results_path
```

### 4. Parallel Processing with Worker Functions

**Pattern:** Wrapper function for multiprocessing

```python
def process_single_profile(pro_file_path, aspect, ...):
    """Main processing function."""
    # Do work
    return result

def worker_wrapper(task_tuple):
    """Unpack arguments for multiprocessing."""
    return process_single_profile(*task_tuple)

# Use with Pool
with Pool(processes=n) as pool:
    results = pool.map(worker_wrapper, tasks)
```

**Why:** Python multiprocessing requires pickleable functions.

---

## Adding New Features

### Example: Add a New Analysis Metric

**Goal:** Calculate "days since last significant snowfall"

#### Step 1: Create Analysis Function

```python
# In wet_front_tracker.py

def days_since_snowfall(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate days since last snowfall > 10cm.
    
    Args:
        df: Single day's profile DataFrame
        
    Returns:
        Days since last snowfall, or None if not applicable
    """
    required_cols = ['height', 'density', 'timestamp']
    if not _validate_dataframe(df, required_cols):
        return None
    
    # Get snow surface height
    surface_height = df['height'].max()
    
    # Look for recent layers with low density (new snow)
    new_snow = df[df['density'] < 150]  # Fresh snow threshold
    
    if new_snow.empty:
        return None
    
    # Calculate age of top new snow layer
    top_new_snow = new_snow.nlargest(1, 'height')
    age_hours = (df['timestamp'].iloc[0] - top_new_snow['timestamp'].iloc[0]).total_seconds() / 3600
    
    return float(age_hours / 24.0)  # Convert to days
```

#### Step 2: Register in main.py

```python
# In _calculate_summary()

from .wet_front_tracker import days_since_snowfall  # Add import

parameters_to_calculate = {
    "hs": get_total_snow_depth,
    "weak_layer": find_wet_slab_loc_bottom_half,
    "days_since_snow": days_since_snowfall,  # Add this line
    # ... other metrics
}
```

#### Step 3: Use in Results

```python
# In _build_result_dict()

def _build_result_dict(summary_full, file_stem, station_metadata, reference_date):
    time_to_loc = find_time_to_loc(summary_full, reference_date)
    
    # Extract new metric
    days_since_snow = None
    if reference_date in summary_full.index and 'days_since_snow' in summary_full.columns:
        days_since_snow = summary_full.loc[reference_date, 'days_since_snow']
    
    return {
        "station_name": station_metadata.get('stationName'),
        "file_stem": file_stem,
        "time_to_loc": time_to_loc,
        "days_since_snow": days_since_snow,  # Add this
        # ... other fields
    }
```

#### Step 4: Add to Visualization (Optional)

```python
# In plotting.py - generate_tooltip_html()

def generate_tooltip_html(row: pd.Series, assets_dir: Path) -> str:
    html = (
        f"<b>{row.get('pathName')}</b><br>"
        f"Days since snow: {row.get('days_since_snow', 'N/A')}<br>"
        # ... rest of tooltip
    )
    return html
```

### Example: Add a New Color Scheme

**Goal:** Color by elevation instead of time/LWC

#### Step 1: Create Color Function

```python
# In plotting.py

def get_elevation_color(altitude: float) -> str:
    """
    Color polygons by elevation.
    
    Args:
        altitude: Elevation in meters
        
    Returns:
        Color name
    """
    if altitude < 2500:
        return 'green'
    elif altitude < 3000:
        return 'yellow'
    elif altitude < 3500:
        return 'orange'
    else:
        return 'red'
```

#### Step 2: Apply in Map Creation

```python
# In create_folium_map()

# Option A: Replace existing coloring
final_gdf['color'] = final_gdf['altitude'].apply(get_elevation_color)

# Option B: Create toggle between color schemes
if config.map.color_by == 'elevation':
    final_gdf['color'] = final_gdf['altitude'].apply(get_elevation_color)
else:
    final_gdf['color'] = final_gdf.apply(
        lambda row: get_polygon_color(row['time_to_loc'], row.get('avg_lwc_above_loc')),
        axis=1
    )
```

#### Step 3: Update Legend

```python
def create_elevation_legend_html() -> str:
    return """
     <b>Elevation</b><br>
     <i style="background:green"></i> < 2500m<br>
     <i style="background:yellow"></i> 2500-3000m<br>
     <i style="background:orange"></i> 3000-3500m<br>
     <i style="background:red"></i> > 3500m
    """
```

---

## Testing Guidelines

### Unit Testing Framework

```python
# tests/test_wet_front_tracker.py

import pytest
import pandas as pd
from src.wetting_front_tracker.wet_front_tracker import avg_lwc_above_weak

def test_avg_lwc_above_weak_basic():
    """Test basic LWC calculation."""
    # Create test data
    df = pd.DataFrame({
        'height': [0.0, 0.5, 1.0, 1.5, 2.0],
        'lwc': [0.00, 0.01, 0.02, 0.03, 0.04],
        'grain_type': [200, 200, 400, 200, 200]
    })
    
    def mock_weak_layer(df):
        return None, 1.0  # Weak layer at 1.0m
    
    # Calculate
    result = avg_lwc_above_weak(df, mock_weak_layer)
    
    # Verify
    assert result is not None
    assert 2.0 < result < 4.0  # Average of 2%, 3%, 4% = 3%

def test_avg_lwc_no_weak_layer():
    """Test behavior when no weak layer found."""
    df = pd.DataFrame({
        'height': [0.0, 0.5, 1.0],
        'lwc': [0.00, 0.01, 0.02],
        'grain_type': [200, 200, 200]
    })
    
    def mock_weak_layer(df):
        return None, None  # No weak layer
    
    result = avg_lwc_above_weak(df, mock_weak_layer)
    assert result is None
```

### Integration Testing

```python
# tests/test_integration.py

def test_full_pipeline():
    """Test complete workflow with sample data."""
    from src.wetting_front_tracker.main import process_single_profile
    
    # Setup
    test_pro = Path("tests/fixtures/test_station.pro")
    
    # Execute
    result = process_single_profile(
        pro_file_path=test_pro,
        aspect='N',
        central_date_arg=datetime(2025, 5, 15)
    )
    
    # Verify
    assert result is not None
    assert 'time_to_loc' in result
    assert 'avg_lwc_above_loc' in result
    assert result['station_name'] == 'TestStation'
```

### Manual Testing Checklist

- [ ] Single .pro file processes successfully
- [ ] Parallel processing works with multiple files
- [ ] Map displays with correct colors
- [ ] Plots generated and accessible
- [ ] Edge cases handled (no weak layer, no water, etc.)
- [ ] Different date ranges work correctly
- [ ] Configuration changes take effect
- [ ] Memory usage reasonable for dataset size

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile a function
cProfile.run('process_single_profile(...)', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time consumers
```

### Common Bottlenecks

**1. .pro File Parsing**

**Problem:** Slow I/O for large files  
**Solution:** Use chunking or memory mapping

```python
# Before
with open(file_path) as f:
    data = f.read()

# After (faster)
with open(file_path, 'rb') as f:
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
```

**2. DataFrame Operations**

**Problem:** Slow iteration  
**Solution:** Vectorize operations

```python
# Before (slow)
for idx, row in df.iterrows():
    result.append(calculate(row))

# After (fast)
result = df.apply(calculate, axis=1)
# Or even better
result = vectorized_calculate(df['column1'], df['column2'])
```

**3. Repeated Calculations**

**Problem:** Same calc multiple times  
**Solution:** Cache results

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param):
    # Heavy computation
    return result
```

### GPU Acceleration

```python
# In snowpack_reader.py

# Automatic GPU detection
try:
    import cupy as xp
    GPU_AVAILABLE = True
except:
    import numpy as xp
    GPU_AVAILABLE = False

# Use xp for array operations
result = xp.mean(array)  # Uses GPU if available, CPU otherwise
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# In any module
import logging
logger = logging.getLogger(__name__)

# Use liberally
logger.debug(f"Processing {file_path}")
logger.info(f"Found {len(results)} results")
logger.warning(f"Missing data for {station}")
logger.error(f"Failed to process: {e}")
```

### Inspect Intermediate Data

```python
# Add checkpoints in main.py

def process_single_profile(...):
    # ... processing ...
    
    # Save intermediate data for debugging
    if config.debug_mode:
        summary_full.to_csv(f"debug/{file_stem}_summary.csv")
        
    # ... continue ...
```

### Common Issues

**Issue: "Index out of bounds"**
```python
# Always check before indexing
if not df.empty and idx in df.index:
    value = df.loc[idx, 'column']
```

**Issue: "NaN in calculation"**
```python
# Check for NaN explicitly
if pd.notna(value):
    result = calculate(value)
```

**Issue: "Memory leak in parallel processing"**
```python
# Explicitly delete large objects
del profile
gc.collect()
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

---

## Code Style Guidelines

### Naming Conventions

- **Functions**: `snake_case()` - `calculate_metric()`
- **Classes**: `PascalCase` - `SnowpackProfile`
- **Constants**: `UPPER_SNAKE_CASE` - `LWC_THRESHOLD`
- **Private**: `_leading_underscore()` - `_validate_input()`

### Documentation

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of what function does.
    
    Longer explanation if needed. Can span
    multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When condition occurs
        
    Examples:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### Type Hints

```python
from typing import Optional, Tuple, List, Dict

def process(
    data: pd.DataFrame,
    threshold: float = 0.04
) -> Tuple[Optional[float], Optional[float]]:
    """Use type hints for clarity."""
    pass
```

---

## Contributing Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make changes with tests**
   ```bash
   # Edit code
   # Add tests
   pytest tests/
   ```

3. **Document changes**
   ```bash
   # Update docstrings
   # Update README if needed
   # Add to CHANGELOG
   ```

4. **Submit pull request**
   - Clear description
   - Link to issue if applicable
   - Include test results

---

## Resources

- **xarray docs**: https://docs.xarray.dev/
- **geopandas docs**: https://geopandas.org/
- **SNOWPACK model**: https://models.slf.ch/p/snowpack/
- **Plotly docs**: https://plotly.com/python/
- **Folium docs**: https://python-visualization.github.io/folium/

---

**Questions?** Open an issue or contact the development team.

**Last Updated:** November 2025
