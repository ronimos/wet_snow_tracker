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

import pandas as pd
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weak Layer Detection
# ---------------------------------------------------------------------------

def largest_fc_dh_gs_diff(df: pd.DataFrame):
    """
    Finds the faceted (FC) or depth hoar (DH) layer with the largest positive
    grain size difference relative to the layer below it.

    This metric is a proxy for a potential weak layer. A large, positive
    `gs_difference` indicates that larger, weaker faceted grains are sitting on
    top of a layer of smaller grains, which can form a stark structural weakness.
    This function searches the entire snowpack profile.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.
                           It must contain 'grain_type', 'gs_difference', and
                           'height' columns.

    Returns:
        tuple or None: A tuple containing (`gs_difference`, `height`) of the most
                       prominent FC/DH weak layer, or None if no suitable
                       layer is found.
    """
    if df.empty or "grain_type" not in df or "gs_difference" not in df:
        return None, None

    # SNOWPACK grain codes for FC and DH range from 400 to 599.
    mask_type = ((df['grain_type'] >= 400) & (df['grain_type'] < 600))
    # A positive difference indicates larger grains in the upper layer.
    mask_diff = df['gs_difference'] > 0
    candidates = df[mask_type & mask_diff]

    if candidates.empty:
        return None, None

    # Find the layer with the maximum grain size difference among candidates.
    best = candidates.loc[candidates['gs_difference'].idxmax()]
    return best['gs_difference'].astype(float), best['height'].astype(float)

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
        return None

    # SNOWPACK grain codes for wet forms range from 770 to 779.
    mask = (df['grain_type'] >= 770) & (df['grain_type'] < 780)
    candidates = df[mask]

    if candidates.empty:
        return None

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
        return None

    mask = df['lwc'] >= 0.04
    candidates = df[mask]

    if candidates.empty:
        return None

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

def find_time_to_loc(summary_df: pd.DataFrame) -> float | None:
    """
    Finds the time in hours until the wet front reaches the weak layer.

    This function observes the simulation data to find the first timestamp where
    the wet front height is at or above the weak layer height.

    Args:
        summary_df (pd.DataFrame): The daily summary data for a station, indexed
                                   by timestamp.

    Returns:
        float | None: The time in hours from the start of the analysis.
                      Returns -1.0 if the front has already passed at the start.
                      Returns None if the front never reaches the layer.
    """
    # Ensure the necessary columns exist
    if 'wet_front_lwc_height' not in summary_df or 'weak_layer_height' not in summary_df:
        return None

    # Drop any rows where the values are missing to ensure a clean comparison
    valid_df = summary_df[['wet_front_lwc_height', 'weak_layer_height']].dropna()
    if valid_df.empty:
        return None

    # Check if the front has already passed at the very first timestep
    if valid_df['wet_front_lwc_height'].iloc[0] >= valid_df['weak_layer_height'].iloc[0]:
        return -1.0  # Special value for "already passed"

    # Find all future times when the front is at or past the weak layer
    intersection_times = valid_df[valid_df['wet_front_lwc_height'] >= valid_df['weak_layer_height']]

    # If it never reaches, return None
    if intersection_times.empty:
        return None

    # Get the timestamp of the first intersection event
    first_intersection_time = intersection_times.index[0]
    start_time = valid_df.index[0]
    
    # Calculate the time difference in hours
    return (first_intersection_time - start_time).total_seconds() / 3600


def get_total_snow_depth(df: pd.DataFrame) -> float:
    """Calculates the total snow depth (HS) for a single daily profile."""
    return 0 if df.empty or "height" not in df else df['height'].max()


def get_highest_wet_point(df: pd.DataFrame) -> float | None:
    """Finds the height of the uppermost 'wet' layer in a daily profile."""
    if df.empty or "grain_type" not in df or "lwc" not in df:
        return None
    mask = ((df['grain_type'] >= 770) & (df['grain_type'] < 780)) | (df['lwc'] > 0.03)
    wet_layers = df[mask]
    return None if wet_layers.empty else float(wet_layers['height'].max())