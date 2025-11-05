"""
wet_snow_tracker.py
===================

This module provides a suite of custom analysis functions designed to extend the
capabilities of the SnowpackProfile class. It focuses on identifying and
tracking key features related to wet snow slab avalanches.

These functions are intended to be used as plug-ins with the
`get_profile_summary()` method of the SnowpackProfile class, allowing for a
powerful and flexible daily time-series analysis of snowpack stability factors.

Authors: Itai and Ron
Last Updated: October 12, 2025
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SNOWPACK grain type codes
FC_DH_MIN_CODE = 400  # Faceted crystals minimum code
FC_DH_MAX_CODE = 600  # Faceted crystals/depth hoar maximum code
WET_GRAIN_MIN_CODE = 770  # Wet grain forms minimum code
WET_GRAIN_MAX_CODE = 780  # Wet grain forms maximum code

# Thresholds
MIN_GS_DIFFERENCE = 0.5  # Minimum grain size difference (mm)
LWC_THRESHOLD_PERCENT = 4.0  # LWC threshold as percentage
LWC_THRESHOLD = LWC_THRESHOLD_PERCENT / 100.0  # 4% = 0.04 volumetric
LWC_THRESHOLD_WET_LAYER = 0.03  # 3% threshold for wet layer detection

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """
    Validates that a DataFrame has required columns and is not empty.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return False
    
    return True


def _is_fc_or_dh(grain_type: float) -> bool:
    """Check if grain type is faceted crystal (FC) or depth hoar (DH)."""
    return FC_DH_MIN_CODE <= grain_type < FC_DH_MAX_CODE

def _is_wet_grain(grain_type: float) -> bool:
    """Check if grain type is a wet grain form."""
    return WET_GRAIN_MIN_CODE <= grain_type < WET_GRAIN_MAX_CODE



# ---------------------------------------------------------------------------
# Weak Layer Detection
# ---------------------------------------------------------------------------

def largest_fc_dh_gs_diff(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the faceted (FC) or depth hoar (DH) layer with the largest positive
    grain size difference relative to the layer below it.

    This metric is a proxy for a potential weak layer. A large, positive
    `gs_difference` indicates that larger, weaker faceted grains are sitting on
    top of a layer of smaller grains, which can form a stark structural weakness.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'grain_type', 'gs_difference', and 'height' columns.

    Returns:
        A tuple containing (gs_difference, height) of the most prominent FC/DH 
        weak layer, or (None, None) if no suitable layer is found.
        
    Examples:
        >>> weak_gs_diff, weak_height = largest_fc_dh_gs_diff(profile_df)
        >>> if weak_height is not None:
        ...     print(f"Weak layer at {weak_height}m with gs_diff={weak_gs_diff}")
    """
    required_cols = ['grain_type', 'gs_difference', 'height']
    if not _validate_dataframe(df, required_cols):
        return None, None

    # Filter for FC and DH grain types with positive grain size difference
    mask_type = ((df['grain_type'] >= FC_DH_MIN_CODE) & 
                 (df['grain_type'] < FC_DH_MAX_CODE))
    mask_diff = df['gs_difference'] > MIN_GS_DIFFERENCE
    candidates = df[mask_type & mask_diff]

    if candidates.empty:
        return None, None

    # Find the layer with the maximum grain size difference
    best = candidates.loc[candidates['gs_difference'].idxmax()]
    return float(best['gs_difference']), float(best['height'])


def find_wet_slab_loc(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the LOC (layer of concern) for a wet slab avalanche.

    Primary Method: Identifies LOC based on capillary barrier - an interface where 
    smaller grains sit on top of larger, weak grains (FC or DH), which can lead 
    to water pooling.

    Fallback Method: If no capillary barrier is found, identifies the interface 
    with the largest grain size difference where larger grains sit on top of 
    smaller grains, and the top layer is NOT composed of wet grains.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'grain_type', 'gs_difference', and 'height' columns.

    Returns:
        A tuple containing (gs_difference, height) of the LOC, or (None, None) 
        if no suitable layer is found.
        
    Notes:
        - Primary: Negative gs_difference (small over large = capillary barrier)
        - Fallback: Positive gs_difference (large over small, non-wet top layer)
    """
    required_cols = ['grain_type', 'gs_difference', 'height']
    if not _validate_dataframe(df, required_cols) or len(df) < 2:
        return None, None

    # CRITICAL FIX: Capillary barrier = small grains over large grains
    # This means NEGATIVE grain size difference (upper layer smaller than lower)
    capillary_interfaces = df[df['gs_difference'] < -MIN_GS_DIFFERENCE].copy()
    
    if capillary_interfaces.empty:
        # No capillary barrier found - try fallback method
        return _find_largest_gs_diff_non_wet_top(df)

    # Identify the layer below each interface (the potential LOC)
    lower_layer_indices = capillary_interfaces.index - 1
    
    # Ensure indices are valid
    valid_indices = lower_layer_indices[lower_layer_indices >= df.index.min()]
    if valid_indices.empty:
        # No valid layer indices - try fallback method
        return _find_largest_gs_diff_non_wet_top(df)
        
    loc_candidates = df.loc[valid_indices].copy()

    # The LOC must be faceted crystals (FC) or depth hoar (DH)
    mask_type = ((loc_candidates['grain_type'] >= FC_DH_MIN_CODE) & 
                 (loc_candidates['grain_type'] < FC_DH_MAX_CODE))
    final_candidates = loc_candidates[mask_type].copy()

    if final_candidates.empty:
        # No FC/DH layers found at capillary barriers - try fallback method
        return _find_largest_gs_diff_non_wet_top(df)

    # Get the gs_difference from the layer above each candidate
    corresponding_upper_indices = final_candidates.index + 1
    
    # Ensure upper indices are valid
    valid_upper = corresponding_upper_indices[
        corresponding_upper_indices.isin(df.index)
    ]
    
    if valid_upper.empty:
        # No valid upper layer indices - try fallback method
        return _find_largest_gs_diff_non_wet_top(df)
    
    final_candidates['gs_difference_interface'] = df.loc[
        valid_upper, 'gs_difference'
    ].values

    # Select the LOC with the most significant capillary barrier (most negative)
    best_loc = final_candidates.loc[
        final_candidates['gs_difference_interface'].idxmin()
    ]
    
    return float(best_loc['gs_difference_interface']), float(best_loc['height'])


def _find_largest_gs_diff_non_wet_top(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback function: Finds the interface with the largest grain size difference
    where the top layer has larger grains and is NOT a wet grain.
    
    This is used when no capillary barrier (LOC) is found. It identifies structural
    weaknesses where large grains sit on top of smaller grains, as long as the
    top layer hasn't been wetted yet.
    
    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'grain_type', 'gs_difference', and 'height' columns.
    
    Returns:
        A tuple containing (gs_difference, height) of the interface,
        or (None, None) if no suitable interface is found.
    """
    required_cols = ['grain_type', 'gs_difference', 'height']
    if not _validate_dataframe(df, required_cols) or len(df) < 2:
        return None, None
    
    # Find interfaces with positive grain size difference (larger on top)
    positive_interfaces = df[df['gs_difference'] > MIN_GS_DIFFERENCE].copy()
    
    if positive_interfaces.empty:
        return None, None
    
    # Filter out interfaces where the top layer is wet grains
    # The top layer is at the current index (since gs_difference is comparing with layer below)
    mask_not_wet = ~positive_interfaces['grain_type'].apply(_is_wet_grain)
    non_wet_interfaces = positive_interfaces[mask_not_wet]
    
    if non_wet_interfaces.empty:
        return None, None
    
    # Select the interface with the largest grain size difference
    best_interface = non_wet_interfaces.loc[
        non_wet_interfaces['gs_difference'].idxmax()
    ]
    
    return float(best_interface['gs_difference']), float(best_interface['height'])


def find_wet_slab_loc_bottom_half(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the wet slab LOC only within the bottom half of the snowpack.
    
    Weak layers in the lower part of the snowpack are often more critical
    as they can potentially fail under the load of a larger slab.
    
    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain a 'height' column.

    Returns:
        A tuple containing (gs_difference, height) of the LOC within the 
        bottom half, or (None, None) if no suitable layer is found.
    """
    if not _validate_dataframe(df, ['height']):
        return None, None

    total_depth = df['height'].max()
    mid_point = total_depth / 2

    # Filter to only the lower half
    bottom_half_df = df[df['height'] <= mid_point].copy()

    return find_wet_slab_loc(bottom_half_df)


def largest_fc_dh_gs_diff_bottom_half(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the FC/DH layer with the largest grain size difference, but only
    within the BOTTOM HALF of the snowpack.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain a 'height' column.

    Returns:
        A tuple containing (gs_difference, height) of the target layer within 
        the bottom half, or (None, None) if no suitable layer is found.
    """
    if not _validate_dataframe(df, ['height']):
        return None, None

    total_depth = df['height'].max()
    mid_point = total_depth / 2

    # Filter to only the lower half
    bottom_half_df = df[df['height'] <= mid_point].copy()

    return largest_fc_dh_gs_diff(bottom_half_df)


# ---------------------------------------------------------------------------
# Wet Front Detection
# ---------------------------------------------------------------------------

def wet_front_form(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the deepest layer where the grain morphology indicates wet forms.

    This function identifies the penetration depth of the wetting front by
    looking for specific grain types (SNOWPACK codes 770-779) that represent
    wet or melting snow.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'grain_type' and 'height' columns.

    Returns:
        A tuple containing (grain_type, height) of the deepest wet grain form,
        or (None, None) if not found.
    """
    required_cols = ['grain_type', 'height']
    if not _validate_dataframe(df, required_cols):
        return None, None

    # Filter for wet grain forms
    mask = ((df['grain_type'] >= WET_GRAIN_MIN_CODE) & 
            (df['grain_type'] < WET_GRAIN_MAX_CODE))
    candidates = df[mask]

    if candidates.empty:
        return None, None

    # The deepest layer is the one with the minimum height
    deepest = candidates.loc[candidates['height'].idxmin()]
    return float(deepest['grain_type']), float(deepest['height'])


def wet_front_lwc(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the deepest layer where liquid water content (LWC) exceeds 4%.

    This provides a quantitative method for tracking the wetting front. A
    volumetric LWC of 4% indicates that the snow is becoming significantly 
    wet, which can lead to a rapid loss of strength.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'lwc' and 'height' columns.

    Returns:
        A tuple containing (lwc, height) of the deepest sufficiently wet layer,
        or (None, None) if not found.
        
    Notes:
        SNOWPACK stores LWC as a volumetric fraction (e.g., 0.04 for 4%).
    """
    required_cols = ['lwc', 'height']
    if not _validate_dataframe(df, required_cols):
        return None, None

    mask = df['lwc'] >= LWC_THRESHOLD
    candidates = df[mask]

    if candidates.empty:
        return None, None

    # The deepest layer is the one with the minimum height
    deepest_idx = candidates['height'].idxmin()
    deepest = candidates.loc[deepest_idx]
    return float(deepest['lwc']), float(deepest['height'])


# ---------------------------------------------------------------------------
# LWC Above Weak Layer
# ---------------------------------------------------------------------------

def lwc_above_weak(
    df: pd.DataFrame, 
    weak_layer_func: Callable[[pd.DataFrame], Tuple[Optional[float], Optional[float]]]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the LWC at the interface directly above a specified weak layer.

    This function first identifies the weak layer using the provided function,
    then finds the layer immediately on top of it and returns its LWC and height.

    Args:
        df: A single day's snow profile DataFrame.
        weak_layer_func: A function that returns (value, height) of the weak layer.

    Returns:
        A tuple containing (lwc, height) of the layer above the weak layer,
        or (None, None) if not found.
        
    Examples:
        >>> lwc, height = lwc_above_weak(df, find_wet_slab_loc_bottom_half)
    """
    required_cols = ['lwc', 'height']
    if not _validate_dataframe(df, required_cols):
        return None, None

    _, weak_layer_height = weak_layer_func(df)

    if weak_layer_height is None:
        return None, None

    # Find all layers that are physically above the weak layer
    layers_above = df[df['height'] > weak_layer_height]

    if layers_above.empty:
        return None, None

    # Find the layer with the minimum height (the one right on top)
    interface_layer_idx = layers_above['height'].idxmin()
    
    # Return the LWC and height from that layer
    interface_layer = df.loc[interface_layer_idx]
    return float(interface_layer['lwc']), float(interface_layer['height'])


def avg_lwc_above_weak(
    df: pd.DataFrame, 
    weak_layer_func: Callable[[pd.DataFrame], Tuple[Optional[float], Optional[float]]]
) -> Optional[float]:
    """
    Calculates the average LWC (as percentage) of all layers above the weak layer.

    This function identifies the weak layer using the provided function,
    then calculates the mean LWC of all layers above it, converting to percentage.

    Args:
        df: A single day's snow profile DataFrame.
        weak_layer_func: A function that returns (value, height) of the weak layer.

    Returns:
        Average LWC above the weak layer as a percentage (0-100),
        or None if not found or no layers exist above the weak layer.
        
    Examples:
        >>> avg_lwc_pct = avg_lwc_above_weak(df, find_wet_slab_loc_bottom_half)
        >>> if avg_lwc_pct and avg_lwc_pct > 3.0:
        ...     print("High water content above LOC")
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
    # LWC in the data is volumetric (0.0 to 1.0), so multiply by 100 for percentage
    avg_lwc = layers_above['lwc'].mean()
    return float(avg_lwc * 100.0) if pd.notna(avg_lwc) else None


# ---------------------------------------------------------------------------
# Time Series Analysis
# ---------------------------------------------------------------------------

def find_time_to_loc(
    summary_df: pd.DataFrame, 
    reference_date: datetime
) -> float:
    """
    Calculates the time (in hours) for the wetting front to reach the weak layer,
    measured from a specific reference date, considering only the current event.

    Args:
        summary_df: A pandas DataFrame with a DateTimeIndex and columns:
                    'wet_front_lwc_height' and 'weak_layer_height'.
        reference_date: The central date for the analysis.

    Returns:
        Time in hours from reference date until wetting front reaches weak layer,
        or np.nan if it doesn't happen during the current event.
    """
    required_cols = ['wet_front_lwc_height', 'weak_layer_height']
    if summary_df is None or summary_df.empty:
        return np.nan
    
    if not all(col in summary_df.columns for col in required_cols):
        logger.warning(f"Missing required columns: {required_cols}")
        return np.nan

    # --- Isolate the current wetting event ---
    # Find the start of all wetting events
    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]
    
    # Find the most recent event start time relative to the reference date
    relevant_start_times = all_start_times[all_start_times <= reference_date]
    if relevant_start_times.empty:
        return np.nan  # No wetting event has started yet
    
    current_event_start_time = relevant_start_times[-1]
    
    # Filter to only look at data from this event forward
    event_df = summary_df.loc[current_event_start_time:].copy()

    # --- Find when the front reaches the LOC ---
    penetration_df = event_df[
        event_df['wet_front_lwc_height'].notna() &
        event_df['weak_layer_height'].notna() &
        (event_df['wet_front_lwc_height'] <= event_df['weak_layer_height'])
    ]

    if penetration_df.empty:
        return np.nan  # The front does not reach the LOC during this event

    # Find the first time the penetration happens in this event
    first_penetration_time = penetration_df.index[0]
    
    # Calculate the difference from the reference date and return in hours
    time_diff_seconds = (first_penetration_time - reference_date).total_seconds()
    
    return float(time_diff_seconds) / 3600.0


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_total_snow_depth(df: pd.DataFrame) -> float:
    """
    Calculates the total snow depth (HS) for a single daily profile.

    This is a simple helper function designed to be passed to `get_profile_summary`.

    Args:
        df: A DataFrame for a single day's snow profile.

    Returns:
        The maximum height value in the profile, or 0.0 if empty.
    """
    if df.empty or 'height' not in df.columns:
        return 0.0
    
    return float(df['height'].max())


def get_highest_wet_point(df: pd.DataFrame) -> Optional[float]:
    """
    Finds the height of the uppermost 'wet' layer in a daily profile.

    A layer is considered 'wet' if its grain type indicates melt forms or if
    its LWC is above a 3% threshold. This function identifies the top of the
    wet snow region in the snowpack.

    Args:
        df: A DataFrame for a single day's snow profile.

    Returns:
        The height of the highest wet layer, or None if no wet layers are found.
    """
    required_cols = ['grain_type', 'lwc', 'height']
    if not _validate_dataframe(df, required_cols):
        return None
    
    # Define wet layer criteria
    is_wet_grain = ((df['grain_type'] >= WET_GRAIN_MIN_CODE) & 
                    (df['grain_type'] < WET_GRAIN_MAX_CODE))
    is_wet_lwc = df['lwc'] > LWC_THRESHOLD_WET_LAYER
    
    wet_layers = df[is_wet_grain | is_wet_lwc]
    
    if wet_layers.empty:
        return None
    
    return float(wet_layers['height'].max())

# ---------------------------------------------------------------------------
# Temporal Persistence
# ---------------------------------------------------------------------------

def apply_loc_temporal_carryforward(
    weak_layer_series: pd.Series,
    method: str = 'forward'
) -> pd.Series:
    """
    Apply temporal carry-forward to LOC (Layer of Concern) heights.
    
    When no LOC is detected at a given timestep (None/NaN), this function
    carries forward the height from the most recent timestep where an LOC
    was detected.
    
    This is the third tier of LOC detection:
    1. Primary: Capillary barrier (small over large FC/DH)
    2. Fallback: Structural weakness (large over small, non-wet top)
    3. Temporal: Carry forward last detected LOC
    
    Args:
        weak_layer_series: Pandas Series with index as timestamps and values 
                          as LOC heights. NaN/None indicates no LOC detected.
        method: 'forward' for forward-fill only (default), or 'both' for 
                forward and backward fill
    
    Returns:
        Series with NaN values filled using the specified method
        
    Examples:
        >>> dates = pd.date_range('2025-03-01', periods=5, freq='D')
        >>> heights = pd.Series([1.2, np.nan, np.nan, 1.5, np.nan], index=dates)
        >>> filled = apply_loc_temporal_carryforward(heights)
        >>> print(filled)
        2025-03-01    1.2  # Original detection
        2025-03-02    1.2  # Carried forward from 03-01
        2025-03-03    1.2  # Carried forward from 03-01
        2025-03-04    1.5  # New detection
        2025-03-05    1.5  # Carried forward from 03-04
        
    Notes:
        - Only fills forward by default (doesn't back-fill)
        - Preserves original detections
        - Useful for maintaining LOC awareness when detection temporarily fails
        - Should be applied AFTER primary and fallback detection methods
        - Consider validation against total snow depth after filling
    """
    if weak_layer_series.empty:
        return weak_layer_series
    
    filled_series = weak_layer_series.copy()
    
    if method == 'forward':
        # Forward fill only (carry last known value forward)
        filled_series = filled_series.ffill()
    elif method == 'both':
        # Fill forward then backward (useful if first values are NaN)
        filled_series = filled_series.ffill().bfill()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'forward' or 'both'.")
    
    return filled_series


def apply_loc_temporal_carryforward_with_validation(
    weak_layer_series: pd.Series,
    snow_depth_series: pd.Series,
    method: str = 'forward'
) -> pd.Series:
    """
    Apply temporal carry-forward to LOC heights with snow depth validation.
    
    This is an enhanced version that ensures carried-forward LOC heights
    don't exceed the current snow depth (which would be physically impossible).
    
    Args:
        weak_layer_series: Pandas Series of LOC heights (NaN = no detection)
        snow_depth_series: Pandas Series of total snow depth (hs)
        method: 'forward' or 'both' for fill direction
    
    Returns:
        Series with filled LOC heights, validated against snow depth
        
    Examples:
        >>> dates = pd.date_range('2025-03-01', periods=4, freq='D')
        >>> heights = pd.Series([1.2, np.nan, np.nan, 1.5], index=dates)
        >>> snow_depth = pd.Series([2.0, 1.8, 1.0, 2.5], index=dates)
        >>> filled = apply_loc_temporal_carryforward_with_validation(
        ...     heights, snow_depth
        ... )
        >>> print(filled)
        2025-03-01    1.2   # Original
        2025-03-02    1.2   # Carried forward (1.2 < 1.8 ✓)
        2025-03-03    NaN   # Carried forward 1.2 > 1.0 ✗ (invalidated)
        2025-03-04    1.5   # New detection
    
    Notes:
        - Invalidates carried-forward values that exceed snow depth
        - Prevents physically impossible scenarios
        - Important for profiles with significant melt or settling
    """
    if weak_layer_series.empty or snow_depth_series.empty:
        return weak_layer_series
    
    # Ensure both series have the same index
    if not weak_layer_series.index.equals(snow_depth_series.index):
        logger.warning(
            "weak_layer_series and snow_depth_series have different indices. "
            "Reindexing to match."
        )
        snow_depth_series = snow_depth_series.reindex(weak_layer_series.index)
    
    # Apply temporal carry-forward
    filled_series = apply_loc_temporal_carryforward(weak_layer_series, method=method)
    
    # Validate against snow depth
    invalid_mask = filled_series > snow_depth_series
    if invalid_mask.any():
        num_invalid = invalid_mask.sum()
        logger.info(
            f"Invalidated {num_invalid} carried-forward LOC heights that "
            f"exceeded snow depth"
        )
        filled_series[invalid_mask] = np.nan
    
    return filled_series
