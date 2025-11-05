# Quick Reference - Key Code Changes

## New Analysis Function (wet_front_tracker.py)

```python
def avg_lwc_above_weak(
    df: pd.DataFrame, 
    weak_layer_func: Callable[[pd.DataFrame], Tuple[Optional[float], Optional[float]]]
) -> Optional[float]:
    """
    Calculates the average LWC (as percentage) of all layers above the weak layer.
    
    Returns:
        Average LWC above the weak layer as a percentage (0-100),
        or None if not found or no layers exist above the weak layer.
    """
    required_cols = ['lwc', 'height']
    if not _validate_dataframe(df, required_cols):
        return None

    _, weak_layer_height = weak_layer_func(df)
    if weak_layer_height is None:
        return None

    # Find all layers that are physically above the weak layer
    layers_above = df[df['height'] > weak_layer_height]
    if layers_above.empty:
        return None

    # Calculate the mean LWC and convert to percentage
    avg_lwc = layers_above['lwc'].mean()
    return float(avg_lwc * 100.0) if pd.notna(avg_lwc) else None
```

## New Coloring Function (plotting.py)

```python
def get_polygon_color(time_to_loc: float, avg_lwc_above_loc: Optional[float] = None) -> str:
    """
    Determines polygon color based on time_to_loc value and average LWC above LOC.
    
    Priority:
    1. If avg_lwc_above_loc > 3%, return red
    2. If avg_lwc_above_loc between 1-3%, return yellow  
    3. Otherwise, use time_to_loc coloring
    """
    # Check LWC-based coloring first (higher priority)
    if avg_lwc_above_loc is not None and pd.notna(avg_lwc_above_loc):
        if avg_lwc_above_loc > 3.0:
            return 'red'
        elif avg_lwc_above_loc >= 1.0:
            return 'yellow'
    
    # Fall back to time_to_loc coloring
    if pd.isna(time_to_loc):
        return TIME_TO_LOC_COLORS['unknown'][0]
    
    time = float(time_to_loc)
    
    for category, (color, min_time, max_time) in TIME_TO_LOC_COLORS.items():
        if min_time is not None and max_time is not None:
            if min_time <= time < max_time:
                return color
    
    return TIME_TO_LOC_COLORS['unknown'][0]
```

## Updated Analysis Parameters (main.py)

```python
parameters_to_calculate = {
    "hs": get_total_snow_depth,
    "weak_layer": find_wet_slab_loc_bottom_half,
    "wet_front_lwc": wet_front_lwc,
    "highest_wet_point": get_highest_wet_point,
    "lwc_above_weak": lambda df: lwc_above_weak(df, find_wet_slab_loc_bottom_half),
    "avg_lwc_above_loc": lambda df: avg_lwc_above_weak(df, find_wet_slab_loc_bottom_half)  # NEW
}
```

## Updated Result Dictionary (main.py)

```python
def _build_result_dict(summary_full, file_stem, station_metadata, reference_date):
    time_to_loc = find_time_to_loc(summary_full, reference_date=reference_date)
    
    # NEW: Get avg_lwc_above_loc at the reference date
    avg_lwc_above_loc = None
    if reference_date in summary_full.index and 'avg_lwc_above_loc' in summary_full.columns:
        avg_lwc_value = summary_full.loc[reference_date, 'avg_lwc_above_loc']
        if pd.notna(avg_lwc_value):
            avg_lwc_above_loc = float(avg_lwc_value)

    return {
        "station_name": station_metadata.get('stationName', file_stem),
        "file_stem": file_stem,
        "time_to_loc": time_to_loc,
        "avg_lwc_above_loc": avg_lwc_above_loc,  # NEW
        "central_date_str": reference_date.strftime('%Y-%m-%d %H:%M')
    }
```

## Updated Map Color Assignment (plotting.py)

```python
# OLD:
final_gdf['color'] = final_gdf['time_to_loc'].apply(get_time_to_loc_color)

# NEW:
final_gdf['color'] = final_gdf.apply(
    lambda row: get_polygon_color(
        row['time_to_loc'], 
        row.get('avg_lwc_above_loc')
    ),
    axis=1
)
```

## Updated Legend (plotting.py)

```python
def create_map_legend_html() -> str:
    return """
     <b>Time for Wetting Front to Reach LOC</b><br>
     <i style="background:darkred"></i> 0 to 24h (Imminent)<br>
     <i style="background:orange"></i> 24 to 48h<br>
     <i style="background:yellow"></i> 48 to 72h<br>
     <hr style='border-top: 1px solid grey; margin-top: 5px; margin-bottom: 5px;'>
     <i style="background:red"></i> -24 to 0h (Recent)<br>
     <i style="background:lightblue"></i> -48 to -24h (Past)<br>
     <i style="background:darkblue"></i> -72 to -48h (Past)<br>
     <hr style='border-top: 1px solid grey; margin-top: 5px; margin-bottom: 5px;'>
     <b>Avg Free Water Content Above LOC</b><br>
     <i style="background:yellow"></i> 1-3% (Elevated)<br>
     <i style="background:red"></i> >3% (High Risk)<br>
     <hr style='border-top: 1px solid grey; margin-top: 5px; margin-bottom: 5px;'>
     <i style="background:gray"></i> Other / No Data
"""
```

## Import Changes (main.py)

```python
# ADD this import:
from .wet_front_tracker import (
    find_time_to_loc,
    find_wet_slab_loc_bottom_half,
    get_highest_wet_point,
    get_total_snow_depth,
    lwc_above_weak,
    avg_lwc_above_weak,  # NEW
    wet_front_lwc,
)
```
