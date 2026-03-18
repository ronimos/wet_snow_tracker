"""
wet_snow_tracker.py
===================

This module provides a suite of custom analysis functions designed to extend the
capabilities of the SnowpackProfile class. It focuses on identifying and
tracking key features related to wet snow slab avalanches, a type of avalanche
that occurs when liquid water weakens the bonds within the snowpack,
particularly at the interface between a cohesive slab and an underlying weak
layer.

These functions are intended to be used as plug-ins with the
`get_profile_summary()` method of the SnowpackProfile class, allowing for a
powerful and flexible daily time-series analysis of snowpack stability factors.
Each function is designed to analyze a single day's snow profile (represented
as a pandas DataFrame) and return a specific metric of interest.

Key Functions:
- `largest_fc_dh_gs_diff`: Finds the most prominent weak layer of faceted
  crystals (FC) or depth hoar (DH) based on grain size difference.
- `largest_fc_dh_gs_diff_bottom_half`: Restricts the weak layer search to the
  more critical lower half of the snowpack.
- `wet_front_form`: Tracks the water penetration front based on the first
  appearance of wet grain morphologies.
- `wet_front_lwc`: Tracks the water penetration front based on a liquid water
  content (LWC) threshold.
- `lwc_above_weak`: A critical function that combines weak layer detection with
  LWC analysis to check for water pooling above the weak layer—a key indicator
  of instability.

Authors: Itai and Ron
Last Updated: August 25, 2025
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weak Layer Detection
# ---------------------------------------------------------------------------

def largest_fc_dh_gs_diff(df: pd.DataFrame):
    """
    Finds the faceted (FC) or depth hoar (DH) layer with the largest grain size
    contrast relative to the layer below it.

    A large positive signed difference (grain_size[i] - grain_size[i-1]) at an
    FC/DH layer indicates coarser weak grains sitting on finer grains — a
    structural weakness. This function computes the signed difference from raw
    grain sizes since SNOWPACK field 0602 is unsigned.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.
                           Must contain 'grain_type', 'grain_size', and 'height'.

    Returns:
        tuple: (signed_gs_diff, height) of the most prominent FC/DH weak layer,
               or (None, None) if no suitable layer is found.
    """
    if df.empty or "grain_type" not in df or "grain_size" not in df:
        return None, None

    # Compute signed grain size difference: positive = coarsening upward
    signed_diff = df['grain_size'].diff()

    mask_type = (df['grain_type'] >= 400) & (df['grain_type'] < 600)
    mask_diff = signed_diff > 0.5
    candidates = df[mask_type & mask_diff].copy()

    if candidates.empty:
        return None, None

    candidates['signed_gs_diff'] = signed_diff.loc[candidates.index]
    best = candidates.loc[candidates['signed_gs_diff'].idxmax()]
    return float(best['signed_gs_diff'].item()), float(best['height'].item())

# In wet_front_tracker.py

def find_wet_slab_loc(df: pd.DataFrame):
    """
    Finds the LOC for a wet slab avalanche based on a capillary barrier.

    Identifies interfaces where a layer of smaller grains sits on top of a
    layer of larger, weak grains (FC or DH). This fine-over-coarse transition
    creates a capillary barrier that can lead to water pooling.

    The signed grain size difference is computed from raw grain_size values
    since SNOWPACK field 0602 (gs_difference) is unsigned.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.
                           Must contain 'grain_type', 'grain_size', and 'height'.

    Returns:
        tuple: (signed_gs_diff, height) of the LOC (the lower, coarser layer),
               or (None, None) if no suitable layer is found.
    """
    if df.empty or "grain_type" not in df or "grain_size" not in df or len(df) < 2:
        return None, None

    # Signed difference: gs[i] - gs[i-1], layers ordered bottom-to-top.
    # Negative = fining upward = fine-over-coarse = capillary barrier.
    signed_diff = df['grain_size'].diff()

    # 1. Find interfaces with significant capillary barrier (fining upward)
    capillary_mask = signed_diff < -0.5
    capillary_interfaces = df[capillary_mask]
    if capillary_interfaces.empty:
        return None, None

    # 2. The LOC candidate is the layer below each interface (the coarse layer)
    lower_layer_indices = capillary_interfaces.index - 1
    valid_indices = lower_layer_indices[lower_layer_indices >= df.index.min()]
    if valid_indices.empty:
        return None, None

    loc_candidates = df.loc[valid_indices].copy()

    # 3. The LOC must be FC or DH
    mask_type = (loc_candidates['grain_type'] >= 400) & (loc_candidates['grain_type'] < 600)
    final_candidates = loc_candidates[mask_type].copy()

    if final_candidates.empty:
        return None, None

    # 4. Look up the signed diff at the layer above each candidate (the interface)
    upper_indices = final_candidates.index + 1
    final_candidates['signed_gs_diff'] = signed_diff.loc[upper_indices].to_numpy()

    # Pick the strongest capillary barrier (most negative)
    best_loc = final_candidates.loc[final_candidates['signed_gs_diff'].idxmin()]

    return float(best_loc['signed_gs_diff'].item()), float(best_loc['height'].item())


def find_wet_slab_loc_bottom_half(df: pd.DataFrame):
    """
    Wrapper to find the wet slab LOC only within the bottom half of the snowpack.
    """
    if df.empty or "height" not in df:
        return None, None

    total_depth = df['height'].max()
    mid_point = total_depth / 2

    # Filter the DataFrame to only include layers in the lower half.
    bottom_half_df = df[df['height'] <= mid_point].copy()

    # Reuse the new wet slab LOC detection logic on the filtered data.
    return find_wet_slab_loc(bottom_half_df)


def largest_fc_dh_gs_diff_bottom_half(df: pd.DataFrame):
    """
    Finds the FC/DH layer with the largest grain size difference, but only
    within the BOTTOM HALF of the snowpack.

    This is a focused version of the `largest_fc_dh_gs_diff` function. Weak
    layers in the lower part of the snowpack are often considered more critical
    as they can potentially fail under the load of a larger slab, leading to
    more significant avalanches. This function filters the search to only
    consider these more dangerous layers.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.
                           Must contain a 'height' column.

    Returns:
        tuple or None: A tuple containing (`gs_difference`, `height`) of the
                       target layer within the bottom half, or None if no
                       suitable layer is found there.
    """
    if df.empty or "height" not in df:
        return None

    total_depth = df['height'].max()
    mid_point = total_depth / 2

    # Filter the DataFrame to only include layers in the lower half.
    bottom_half_df = df[df['height'] <= mid_point]

    # Reuse the original weak layer detection logic on the filtered data.
    return largest_fc_dh_gs_diff(bottom_half_df)

# ---------------------------------------------------------------------------
# Wet Front Detection
# ---------------------------------------------------------------------------

def wet_front_form(df: pd.DataFrame):
    """
    Finds the deepest layer where the grain morphology indicates wet forms.

    This function identifies the penetration depth of the wetting front by
    looking for specific grain types (SNOWPACK codes 770-779) that represent
    wet or melting snow. The deepest such layer corresponds to the furthest
    point the water has percolated into the snowpack from the surface.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.

    Returns:
        tuple or None: A tuple containing (`grain_type`, `height`) of the
                       deepest wet grain form, or None if not found.
    """
    if df.empty or "grain_type" not in df:
        return None, None

    # SNOWPACK grain codes for wet forms range from 770 to 779.
    mask = (df['grain_type'] >= 770) & (df['grain_type'] < 780)
    candidates = df[mask]

    if candidates.empty:
        return None, None

    # The deepest layer is the one with the minimum height.
    deepest = candidates.loc[candidates['height'].idxmin()]
    return deepest['grain_type'].astype(float), deepest['height'].astype(float)

def wet_front_lwc(df: pd.DataFrame):
    """
    Finds the deepest layer where liquid water content (LWC) exceeds 4%.

    This provides a quantitative method for tracking the wetting front. A
    volumetric LWC of 4% is a common threshold indicating that the snow is
    becoming significantly wet, which can lead to a rapid loss of strength.
    The function finds the deepest layer meeting this criterion.

    Note: SNOWPACK stores LWC as a volumetric fraction (e.g., 0.04 for 4%).

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.

    Returns:
        tuple or None: A tuple containing (`lwc`, `height`) of the deepest
                       sufficiently wet layer, or None if not found.
    """
    if df.empty or "lwc" not in df:
        return None, None

    mask = df['lwc'] >= 0.04
    candidates = df[mask]

    if candidates.empty:
        return None, None

    # The deepest layer is the one with the minimum height.
    deepest_idx = candidates['height'].idxmin()
    deepest = candidates.loc[deepest_idx]
    return deepest['lwc'].astype(float), deepest['height'].astype(float)

# ---------------------------------------------------------------------------
# LWC Above Weak Layer
# ---------------------------------------------------------------------------

def lwc_above_weak(df: pd.DataFrame, weak_layer_func: Callable) -> tuple[float | None, float | None]:
    """
    Calculates the LWC at the interface directly above a specified weak layer.

    This function first identifies the weak layer using the provided function,
    then finds the layer immediately on top of it and returns its LWC and height.

    Args:
        df (pd.DataFrame): A single day's snow profile.
        weak_layer_func (callable): A function that returns the properties
                                    of the weak layer.

    Returns:
        tuple[float | None, float | None]: A tuple containing the LWC and height
                                           of the layer above the weak layer,
                                           or (None, None) if not found.
    """
    gs_diff, weak_layer_height = weak_layer_func(df)

    if weak_layer_height is None:
        return None, None

    # Find all layers that are physically above the weak layer
    layers_above = df[df['height'] > weak_layer_height]

    if layers_above.empty:
        return None, None

    # From those layers, find the one with the minimum height (the one right on top)
    interface_layer_idx = layers_above['height'].idxmin()
    
    # Return the specific LWC and height values (scalars) from that single layer
    return df.loc[interface_layer_idx]['lwc'], df.loc[interface_layer_idx]['height']


# In wet_front_tracker.py

def find_time_to_loc(summary_df: pd.DataFrame, reference_date: datetime) -> float | None:
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
    if summary_df is None or summary_df.empty or 'wet_front_lwc_height' not in summary_df or 'weak_layer_height' not in summary_df:
        return np.nan

    # --- NEW LOGIC: Isolate the current wetting event ---
    # 1. Find the start of all wetting events (where wet front appears).
    is_wet = summary_df['wet_front_lwc_height'].notna()
    event_starts = is_wet & ~is_wet.shift(1, fill_value=False)
    all_start_times = summary_df.index[event_starts]
    
    # 2. Find the most recent event start time relative to the reference date.
    relevant_start_times = all_start_times[all_start_times <= reference_date]
    if relevant_start_times.empty:
        return np.nan # No wetting event has started yet.
    
    current_event_start_time = relevant_start_times[-1]
    
    # 3. Filter the dataframe to only look at data from this event forward.
    event_df = summary_df.loc[current_event_start_time:].copy()

    # --- Original logic, now applied to the filtered event_df ---
    penetration_df = event_df[
        event_df['wet_front_lwc_height'].notna() &
        event_df['weak_layer_height'].notna() &
        (event_df['wet_front_lwc_height'] <= event_df['weak_layer_height'])
    ]

    if penetration_df.empty:
        return np.nan # The front does not reach the LOC during this event.

    # Find the first time the penetration happens in this event
    first_penetration_time = penetration_df.index[0]
    
    # Calculate the difference from the reference date and return in hours
    time_diff_seconds = (first_penetration_time - reference_date).total_seconds()
    
    return float(time_diff_seconds) / 3600.0

def get_total_snow_depth(df: pd.DataFrame) -> float:
    """
    Calculates the total snow depth (HS) for a single daily profile.

    This is a simple helper function designed to be passed to `get_profile_summary`.

    Args:
        df (pd.DataFrame): A DataFrame for a single day's snow profile.

    Returns:
        float: The maximum height value in the profile, or 0 if empty.
    """
    return 0 if df.empty or "height" not in df else df['height'].max()


def get_highest_wet_point(df: pd.DataFrame) -> float | None:
    """
    Finds the height of the uppermost 'wet' layer in a daily profile.

    A layer is considered 'wet' if its grain type indicates melt forms or if
    its LWC is above a 3% threshold. This function identifies the top of the
    wet snow region in the snowpack.

    Args:
        df (pd.DataFrame): A DataFrame for a single day's snow profile.

    Returns:
        Optional[float]: The height of the highest wet layer, or None if no
                         wet layers are found.
    """
    if df.empty or "grain_type" not in df or "lwc" not in df:
        return None
    mask = ((df['grain_type'] >= 770) & (df['grain_type'] < 780)) | (df['lwc'] > 0.03)
    wet_layers = df[mask]
    return None if wet_layers.empty else float(wet_layers['height'].max())
