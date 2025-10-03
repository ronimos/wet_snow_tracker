"""
find_stalled_fronts.py
======================

This script processes SNOWPACK (.pro) files to generate a dataset suitable for
training a machine learning model to predict the stalling of a wetting front.

Instead of only identifying long stalls, this script traces the wetting front
through the snowpack over time. It records the detailed snowpack properties of
every layer interface the front crosses. For each interface, it records a target
variable, `stall_duration_hours`, which is 0 if the front passed through, or the
duration in hours if the front stalled at that interface.

Workflow:
---------
1.  **Manifest Reading**: Reads a list of .pro files to be processed.
2.  **Iterative Analysis**: For each .pro file, it traces the wetting front's
    deepest penetration point over time.
3.  **Event Detection**: Whenever the front advances past one or more layers, it
    treats each newly crossed interface as an "event".
4.  **Feature Extraction**: For each event, it captures a comprehensive set of
    snowpack variables (e.g., grain size, density, temperature) for the layers
    immediately above and below the interface.
5.  **Target Variable Generation**: It pre-calculates all stall periods and uses
    this information to assign the `stall_duration_hours` for each event.
6.  **Output Generation**: The final, feature-rich dataset is saved to a single
    CSV file, ready for use in an ML pipeline.

Usage:
------
Run this script from the root of the project:
`python special_analysis/find_stalled_fronts.py`
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

# --- Debugging Flag ---
DEBUG_MODE = True
DEBUG_FILE_LIMIT = 5
# ---

# --- Path Correction ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
# ---

from wetting_front_tracker.snowpack_reader import SnowpackProfile, GPU_AVAILABLE
from wetting_front_tracker.wet_front_tracker import wet_front_lwc, get_total_snow_depth
from wetting_front_tracker.param_config import (
    PRO_FILE_MANIFEST, IS_DEV_ENVIRONMENT, PRO_FILES_BASE_PATH_PROD,
    PRO_FILES_BASE_PATH_DEV, SNOWPACK_LOCATIONS_CSV, DATA_PATH
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_manifest_from_directory(manifest_path: Path) -> bool:
    """Generates a manifest file by scanning for .pro files."""
    search_dir = DATA_PATH / "input"
    if not search_dir.exists(): return False
    
    pro_files = list(search_dir.rglob("*.pro"))
    if not pro_files: return False

    manifest_paths = []
    for p in pro_files:
        try:
            relative_part = p.relative_to(PRO_FILES_BASE_PATH_DEV)
            prod_path = str(PRO_FILES_BASE_PATH_PROD / relative_part)
            manifest_paths.append(prod_path.replace('\\', '/'))
        except ValueError:
            manifest_paths.append(str(p).replace('\\', '/'))

    with open(manifest_path, 'w') as f:
        for path in sorted(manifest_paths): f.write(f"{path}\n")
    
    logging.info(f"Generated manifest with {len(pro_files)} files at: {manifest_path}")
    return True

def get_summary_series(profile: SnowpackProfile, parameter: str, function: callable) -> pd.Series:
    """Calculates a time series for a given parameter at full temporal resolution."""
    if profile.data is None or profile.data.timestamp.size == 0:
        return pd.Series(dtype=float)

    start = pd.to_datetime(profile.data.timestamp.values[0])
    end = pd.to_datetime(profile.data.timestamp.values[-1])
    
    summary = profile.get_full_timeseries_summary(
        parameters_to_calculate={parameter: function},
        start_date=str(start),
        end_date=str(end),
    )
    
    # This logic correctly finds the relevant data series whether the source 
    # function returned a single value or a tuple that was unpacked.
    height_col = f"{parameter}_height"
    value_col = f"{parameter}_value"

    if height_col in summary.columns:
        return summary[height_col]
    elif value_col in summary.columns:
        return summary[value_col]
    elif parameter in summary.columns:
        return summary[parameter]
    else:
        return pd.Series(dtype=float)


def find_stalled_periods(wet_front_series: pd.Series, hs_series: pd.Series, min_duration_hours: int = 12) -> Dict[Tuple[pd.Timestamp, float], float]:
    """
    Identifies all stall periods and returns them as a lookup dictionary.
    The dictionary maps (start_time, height) to stall_duration.
    """
    stall_lookup = {}
    # Bridge small gaps, but not entire dry periods
    series = wet_front_series.ffill(limit=6)
    
    # Identify blocks of consecutive equal values
    blocks = (series.diff() != 0).cumsum()
    
    for _, group in series.groupby(blocks):
        if group.isnull().all() or len(group) < 2: continue

        start_time, end_time = group.index[0], group.index[-1]
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Check for minimum duration
        if duration >= min_duration_hours:
            # Physical plausibility check
            hs_during_stall = hs_series.loc[start_time:end_time]
            stalled_height = group.iloc[0]
            if not hs_during_stall.empty and (hs_during_stall < stalled_height).any():
                continue # Skip if snow depth drops below the stall height

            stall_lookup[(start_time, stalled_height)] = duration
            
    return stall_lookup

def get_interface_layers(profile_df: pd.DataFrame, height: float) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Finds the layers directly above and below a specific height."""
    if profile_df.empty: return None, None
    below_candidates = profile_df[profile_df['height'] <= height]
    layer_below = None if below_candidates.empty else below_candidates.loc[below_candidates['height'].idxmax()]
    above_candidates = profile_df[profile_df['height'] > height]
    layer_above = None if above_candidates.empty else above_candidates.loc[above_candidates['height'].idxmin()]
    
    return layer_below, layer_above

def analyze_interface_properties(profile: SnowpackProfile, timestamp: pd.Timestamp, interface_height: float, stall_lookup: Dict) -> Optional[Dict[str, Any]]:
    """
    Extracts detailed snowpack properties for a single interface crossing.
    """
    if profile.data is None: return None

    try:
        profile_xr = profile.data.sel(timestamp=timestamp, method='nearest')
        
        if GPU_AVAILABLE:
            profile_df = profile_xr.as_numpy().to_dataframe().reset_index().dropna(subset=['height'])
        else:
            profile_df = profile_xr.to_dataframe().reset_index().dropna(subset=['height'])
        
        if profile_df.empty: return None

    except (KeyError, IndexError):
        return None

    layer_below, layer_above = get_interface_layers(profile_df, interface_height)
    if layer_below is None or layer_above is None: return None

    # Determine the target variable: stall duration (defaults to 0 for pass-through)
    stall_duration = stall_lookup.get((timestamp, interface_height), 0.0)

    result = {
        "timestamp": timestamp,
        "interface_height_m": interface_height,
        "stall_duration_hours": stall_duration,
        "total_snow_depth_m": profile_df['height'].max(),
        "latitude": profile.metadata.get('latitude'),
        "longitude": profile.metadata.get('longitude'),
        "elevation": profile.metadata.get('altitude'),
        "aspect": profile.metadata.get('aspect'),
    }
    
    # Add all properties from the layers below and above the interface
    for col in layer_below.index:
        if col not in ['timestamp', 'layer_index']:
            result[f'below_{col}'] = layer_below[col]
    for col in layer_above.index:
        if col not in ['timestamp', 'layer_index']:
            result[f'above_{col}'] = layer_above[col]
    
    return result

def extract_wetting_front_events(profile: SnowpackProfile, stall_lookup: Dict) -> List[Dict[str, Any]]:
    """
    Traces the wetting front and records data for every interface it crosses.
    """
    if profile.data is None or 'height' not in profile.data: return []

    all_events = []
    
    # Get all unique layer boundary heights that ever exist in the profile
    # This is more robust than checking at a single timestamp
    all_heights_xr = profile.data['height'].dropna(dim='layer_index', how='all')
    if 'layer_index' not in all_heights_xr.dims: return [] # No data
    
    if GPU_AVAILABLE:
        height_values_cpu = all_heights_xr.to_numpy()
    else:
        height_values_cpu = all_heights_xr.values
       
    unique_layer_heights = np.unique(height_values_cpu[~np.isnan(height_values_cpu)])

    wet_front_series = get_summary_series(profile, "wet_front", wet_front_lwc).dropna()
    if wet_front_series.empty: return []

    # Track the deepest point the front has reached so far
    deepest_front_so_far = float('inf')

    for timestamp, current_depth in wet_front_series.items():
        # Check if the front has advanced to a new deepest point
        if current_depth < deepest_front_so_far:
            # Identify all layer interfaces the front just passed through
            interfaces_crossed = unique_layer_heights[
                (unique_layer_heights > current_depth) & (unique_layer_heights < deepest_front_so_far)
            ]
            
            # For each newly crossed interface, generate a data row
            for interface_height in sorted(interfaces_crossed, reverse=True):
                event_data = analyze_interface_properties(profile, timestamp, interface_height, stall_lookup)
                if event_data:
                    all_events.append(event_data)
            
            # Update the deepest point
            deepest_front_so_far = current_depth
    
    return all_events

def get_aspect_from_metadata(metadata: Dict[str, Any]) -> str:
    """Converts aspect from degrees in metadata to a cardinal direction."""
    aspect_deg = pd.to_numeric(metadata.get('slopeAzi'), errors='coerce')
    if pd.isna(aspect_deg):
        return "Flat"
    
    if (aspect_deg > 315) or (aspect_deg <= 45): return "N"
    if aspect_deg <= 135: return "E"
    if aspect_deg <= 225: return "S"
    if aspect_deg <= 315: return "W"
    return "Flat"

def main():
    """Main function to orchestrate the data generation."""
    generate_manifest_from_directory(PRO_FILE_MANIFEST)
    if not PRO_FILE_MANIFEST.exists():
        logging.error(f"Manifest file not found: {PRO_FILE_MANIFEST}")
        return

    with open(PRO_FILE_MANIFEST, 'r') as f:
        pro_files = [line.strip() for line in f.readlines()]
        
    if DEBUG_MODE:
        pro_files = pro_files[:DEBUG_FILE_LIMIT]
        logging.info(f"--- DEBUG MODE ON: Processing only {len(pro_files)} files. ---")

    all_results = []
    
    logging.info(f"Starting analysis on {len(pro_files)} .pro files...")
    for pro_file_str in tqdm(pro_files, desc="Generating ML Data"):
        pro_file_path = Path(pro_file_str)
        
        effective_path = pro_file_path
        if IS_DEV_ENVIRONMENT:
            try:
                # Adjust path for local development
                relative_part = pro_file_path.relative_to(Path(PRO_FILES_BASE_PATH_PROD))
                effective_path = PRO_FILES_BASE_PATH_DEV / relative_part
            except ValueError:
                logging.warning(f"Could not remap dev path for {pro_file_path}")
                continue
        
        if not effective_path.exists(): continue

        try:
            profile = SnowpackProfile(effective_path)
            # Add cardinal aspect to metadata for consistent feature generation
            profile.metadata['aspect'] = get_aspect_from_metadata(profile.metadata)
            if profile.data is None: continue

            # CORRECTED: Call the robust, high-resolution function for both series
            wet_front_series = get_summary_series(profile, "wet_front", wet_front_lwc)
            hs_series = get_summary_series(profile, "hs", get_total_snow_depth)
            
            if wet_front_series.empty or hs_series.empty: continue
            
            # Pre-calculate all stall events to use as a lookup for labeling
            stall_lookup = find_stalled_periods(wet_front_series, hs_series)

            # Generate the feature data for all interface crossings
            event_data = extract_wetting_front_events(profile, stall_lookup)
            all_results.extend(event_data)
            
        except Exception as e:
            logging.error(f"Failed to process {effective_path}: {e}", exc_info=False)

    if not all_results:
        logging.info("Analysis complete. No wetting front events found.")
        return

    # Save results to a dedicated folder
    results_path = project_root / "special_analysis_results"
    results_path.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    
    # Define a new, descriptive output file name
    output_path = results_path / "front_stall_snowpack_data.csv"
    results_df.to_csv(output_path, index=False)
    
    logging.info(f"Analysis complete. Found {len(all_results)} interface crossing events.")
    logging.info(f"ML dataset saved to: {output_path}")

if __name__ == "__main__":
    main()

