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
    Finds the LOC (layer of concern) for a wet slab avalanche based on a 
    capillary barrier.

    This function identifies the LOC by finding an interface where a layer of
    smaller grains sits on top of a layer of larger, weak grains (FC or DH).
    This creates a capillary barrier that can lead to water pooling.

    Args:
        df: A DataFrame representing a single day's snow profile.
            Must contain 'grain_type', 'gs_difference', and 'height' columns.

    Returns:
        A tuple containing (gs_difference, height) of the LOC (the lower layer),
        or (None, None) if no suitable layer is found.
        
    Notes:
        The gs_difference represents the interface characteristic, where negative
        values indicate small grains over large grains (capillary barrier).
    """
    required_cols = ['grain_type', 'gs_difference', 'height']
    if not _validate_dataframe(df, required_cols) or len(df) < 2:
        return None, None

    # CRITICAL FIX: Capillary barrier = small grains over large grains
    # This means NEGATIVE grain size difference (upper layer smaller than lower)
    capillary_interfaces = df[df['gs_difference'] < -MIN_GS_DIFFERENCE].copy()
    
    if capillary_interfaces.empty:
        return None, None

    # Identify the layer below each interface (the potential LOC)
    lower_layer_indices = capillary_interfaces.index - 1
    
    # Ensure indices are valid
    valid_indices = lower_layer_indices[lower_layer_indices >= df.index.min()]
    if valid_indices.empty:
        return None, None
        
    loc_candidates = df.loc[valid_indices].copy()

    # The LOC must be faceted crystals (FC) or depth hoar (DH)
    mask_type = ((loc_candidates['grain_type'] >= FC_DH_MIN_CODE) & 
                 (loc_candidates['grain_type'] < FC_DH_MAX_CODE))
    final_candidates = loc_candidates[mask_type].copy()

    if final_candidates.empty:
        return None, None

    # Get the gs_difference from the layer above each candidate
    corresponding_upper_indices = final_candidates.index + 1
    
    # Ensure upper indices are valid
    valid_upper = corresponding_upper_indices[
        corresponding_upper_indices.isin(df.index)
    ]
    
    if valid_upper.empty:
        return None, None
    
    final_candidates['gs_difference_interface'] = df.loc[
        valid_upper, 'gs_difference'
    ].values

    # Select the LOC with the most significant capillary barrier (most negative)
    best_loc = final_candidates.loc[
        final_candidates['gs_difference_interface'].idxmin()
    ]
    
    return float(best_loc['gs_difference_interface']), float(best_loc['height'])


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