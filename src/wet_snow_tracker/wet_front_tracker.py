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
from .snowpack_reader import SnowpackProfile

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
        return None

    # SNOWPACK grain codes for FC and DH range from 400 to 599.
    mask_type = ((df['grain_type'] >= 400) & (df['grain_type'] < 600))
    # A positive difference indicates larger grains in the upper layer.
    mask_diff = df['gs_difference'] > 0
    candidates = df[mask_type & mask_diff]

    if candidates.empty:
        return None

    # Find the layer with the maximum grain size difference among candidates.
    best = candidates.loc[candidates['gs_difference'].idxmax()]
    return float(best['gs_difference']), float(best['height'])

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
    return int(deepest['grain_type']), float(deepest['height'])

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
    deepest = candidates.loc[candidates['height'].idxmin()]
    return float(deepest['lwc']), float(deepest['height'])

# ---------------------------------------------------------------------------
# LWC Above Weak Layer
# ---------------------------------------------------------------------------

def lwc_above_weak(df: pd.DataFrame, weak_layer_func=largest_fc_dh_gs_diff):
    """
    Checks the layer immediately above a detected weak layer for high LWC.

    This function simulates a critical process in wet slab avalanche formation:
    percolating water being impeded by a layer boundary (like a crust or a
    change in density at a weak layer), causing water to pool. This pooling
    dramatically lubricates the weak layer and reduces its shear strength.

    The function is modular: it accepts another function (`weak_layer_func`)
    to first identify the weak layer of interest before checking for water above it.

    Args:
        df (pd.DataFrame): The daily snow profile data.
        weak_layer_func (callable, optional): The function to use for finding
            the weak layer. Defaults to `largest_fc_dh_gs_diff`, which searches
            the entire snowpack. In practice, this is often swapped with
            `largest_fc_dh_gs_diff_bottom_half`.

    Returns:
        tuple or None: A tuple containing (`lwc`, `height`) of the layer
                       immediately above the weak layer if its LWC is > 3%,
                       otherwise None. Returns None if no weak layer is found.
    """
    # Step 1: Find the weak layer using the provided function.
    weak_layer_result = weak_layer_func(df)

    if not weak_layer_result:
        return None # No weak layer found, so no interface to check.

    # The height of the weak layer is the second element in the returned tuple.
    weak_layer_height = weak_layer_result[1]

    # Step 2: Isolate the single layer immediately above the weak layer.
    # We find all layers with greater height, sort them by height, and take the first one.
    above = df[df['height'] > weak_layer_height].sort_values("height").head(1)

    if above.empty or "lwc" not in above:
        return None # No layer exists above the weak layer.

    # Step 3: Check if the LWC in that layer exceeds the 3% threshold.
    lwc_val = above["lwc"].iloc[0]
    if lwc_val > 0.03:
        # If it's wet, return its LWC and height.
        return float(lwc_val), float(above["height"].iloc[0])

    # If the layer is not wet enough, return None.
    return None