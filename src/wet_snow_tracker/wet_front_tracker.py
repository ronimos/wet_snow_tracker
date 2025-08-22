"""
wet_snow_tracker.py
===================

This module provides a suite of custom analysis functions designed to extend the
capabilities of the SnowpackProfile class. It focuses on identifying and tracking
key features related to wet snow slab avalanches, including the detection of
weak layers, the progression of wet fronts, and the presence of liquid water
content (LWC) at critical interfaces.

These functions are intended to be used with the `get_profile_summary()` method
of the SnowpackProfile class, allowing for daily time-series analysis of
snowpack stability factors.

Key Functions:
- largest_fc_dh_gs_diff: Finds the most prominent weak layer based on grain size.
- largest_fc_dh_gs_diff_bottom_half: Restricts the weak layer search to the
  more critical lower half of the snowpack.
- wet_front_form: Tracks the wet front based on wet grain morphologies.
- wet_front_lwc: Tracks the wet front based on a liquid water content threshold.
- lwc_above_weak: Calculates the liquid water content in the layer directly
  above a detected weak layer.

Authors: Itai and Ron
Last Updated: August 21, 2025
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
    Find the FC/DH layer (grain_type 400–599) with the largest positive
    grain size difference (larger grain in the bottom layer).

    This function searches the entire snowpack profile.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile,
                           containing columns like 'grain_type', 'gs_difference',
                           and 'height'.

    Returns:
        tuple or None: A tuple containing (gs_difference, height) of the target
                       layer, or None if no suitable layer is found.
    """
    if df.empty or "grain_type" not in df or "gs_difference" not in df:
        return None
    
    mask_type = ((df['grain_type'] >= 400) & (df['grain_type'] < 600))
    mask_diff = df['gs_difference'] > 0
    candidates = df[mask_type & mask_diff]

    if candidates.empty:
        return None
    
    best = candidates.loc[candidates['gs_difference'].idxmax()]
    return float(best['gs_difference']), float(best['height'])

def largest_fc_dh_gs_diff_bottom_half(df: pd.DataFrame):
    """
    Find the FC/DH layer with the largest grain size difference located in the
    BOTTOM HALF of the snowpack.

    Weak layers in the lower part of the snowpack are often more critical for
    slab avalanche formation. This function filters the search to only consider
    these layers.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.
                           Must contain a 'height' column.

    Returns:
        tuple or None: A tuple containing (gs_difference, height) of the target
                       layer within the bottom half, or None if no suitable
                       layer is found there.
    """
    if df.empty or "height" not in df:
        return None
    
    total_depth = df['height'].max()
    mid_point = total_depth / 2
    
    # Filter for layers in the bottom half of the snowpack
    bottom_half_df = df[df['height'] <= mid_point]
    
    # Reuse the original logic on the filtered dataframe
    return largest_fc_dh_gs_diff(bottom_half_df)

# ---------------------------------------------------------------------------
# Wet Front Detection
# ---------------------------------------------------------------------------

def wet_front_form(df: pd.DataFrame):
    """
    Find the deepest layer where grain_type indicates a wet form (770–779).

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.

    Returns:
        tuple or None: A tuple containing (grain_type, height) of the deepest
                       wet form, or None if not found.
    """
    if df.empty or "grain_type" not in df:
        return None
    
    mask = (df['grain_type'] >= 770) & (df['grain_type'] < 780)
    candidates = df[mask]
    
    if candidates.empty:
        return None
    
    deepest = candidates.loc[candidates['height'].idxmin()]
    return int(deepest['grain_type']), float(deepest['height'])

def wet_front_lwc(df: pd.DataFrame):
    """
    Find the deepest layer where liquid water content (lwc) > 3%.

    SNOWPACK stores lwc as a fraction, so the threshold used is 0.03.

    Args:
        df (pd.DataFrame): A DataFrame representing a single day's snow profile.

    Returns:
        tuple or None: A tuple containing (lwc, height) of the deepest wet
                       layer, or None if not found.
    """
    if df.empty or "lwc" not in df:
        return None
    
    mask = df['lwc'] > 0.03
    candidates = df[mask]
    
    if candidates.empty:
        return None
    
    deepest = candidates.loc[candidates['height'].idxmin()]
    return float(deepest['lwc']), float(deepest['height'])

# ---------------------------------------------------------------------------
# LWC Above Weak Layer
# ---------------------------------------------------------------------------

def lwc_above_weak(df: pd.DataFrame, weak_layer_func=largest_fc_dh_gs_diff):
    """
    Check the layer immediately above a specified weak layer for high LWC.

    This function first identifies a weak layer using a provided function, then
    checks if the liquid water content in the layer directly above it exceeds 3%.

    Args:
        df (pd.DataFrame): The daily snow profile data.
        weak_layer_func (callable, optional): The function to use for finding
            the weak layer. Defaults to `largest_fc_dh_gs_diff`, which searches
            the entire snowpack.

    Returns:
        tuple or None: A tuple containing (lwc, height) of the layer above the
                       weak layer if it's wet, otherwise None.
    """
    weak_layer_result = weak_layer_func(df)
    
    if not weak_layer_result:
        return None

    # The height of the weak layer is the second element in the tuple
    weak_layer_height = weak_layer_result[1]
    
    # Find the layer immediately above the weak layer
    above = df[df['height'] > weak_layer_height].sort_values("height").head(1)
    
    if above.empty or "lwc" not in above:
        return None

    lwc_val = above["lwc"].iloc[0]
    if lwc_val > 0.03:
        return float(lwc_val), float(above["height"].iloc[0])
        
    return None