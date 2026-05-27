"""
test_plot_loc_detection.py
===========================

Visual regression tests for LOC (Layer of Concern) detection methods.

This module creates comparison plots showing:
1. Seasonal evolution of LOC detection (ML vs rule-based)
2. Timeline of detection events
3. Multi-file comparison plots

The plots help validate that both detection methods are working correctly
and identify any discrepancies or improvements from the ML approach.

Requirements:
    - Real .pro files in fixtures/real_data/
    - Trained ML model (optional - tests will skip if unavailable)
    - pytest for test orchestration

Usage:
    # Run all tests
    pytest test_plot_loc_detection.py -v
    
    # Run specific test
    pytest test_plot_loc_detection.py::TestPlotLOCDetection::test_plot_season_comparison
    
    # Generate plots without pytest
    python test_plot_loc_detection.py

Author: Ron Simenhois
Created: November 2025
Last Updated: November 2025
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# Updated imports for the new architecture
from wetting_front_tracker.ml_loc_detector import MLLocDetector
from wetting_front_tracker.snowpack_reader import SnowpackProfile
from wetting_front_tracker.wet_front_tracker import find_wet_slab_loc
from wetting_front_tracker.main import _get_closest_synoptic_time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_data"
OUTPUT_DIR = Path(__file__).parent / "output_plots"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir() -> Path:
    """
    Create output directory for plots.
    
    Returns:
        Path to output directory
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture
def real_pro_files() -> List[Path]:
    """
    Get all real .pro files from fixtures directory.
    
    Returns:
        List of Path objects to .pro files
        
    Raises:
        pytest.skip: If fixtures directory doesn't exist or no files found
    """
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")
    
    pro_files = list(FIXTURES_DIR.glob("*.pro"))
    if not pro_files:
        pytest.skip(f"No .pro files found in {FIXTURES_DIR}")
    
    return pro_files


@pytest.fixture
def sample_pro_file(real_pro_files: List[Path]) -> Path:
    """
    Get first .pro file for testing.
    
    Args:
        real_pro_files: List of available .pro files
        
    Returns:
        Path to first .pro file
    """
    return real_pro_files[0]


@pytest.fixture
def trained_model_path() -> Optional[Path]:
    """
    Locate the trained model directory.
    
    Checks multiple common locations for trained models.
    
    Returns:
        Path to model directory, or None if not found
    """
    # Common paths where the model might be stored
    potential_paths = [
        Path("src/wetting_front_tracker/assets/models"),
        Path("results/trained_models/latest/trained_model"),
        Path("results/model/trained_model"),
        Path("models/trained"),
        Path("trained_models"),
        Path("src/wetting_front_tracker/tests/fixtures/model")
    ]
    
    for path in potential_paths:
        if path.exists() and (path / "model.joblib").exists():
            return path
    
    return None


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------

class TestPlotLOCDetection:
    """Visual tests for LOC detection comparison."""
    
    def test_plot_season_comparison(
        self,
        sample_pro_file: Path,
        output_dir: Path,
        trained_model_path: Optional[Path]
    ) -> None:
        """
        Generate seasonal plot comparing rule-based and ML LOC detection.
        
        This test creates a comprehensive visualization showing:
        - Snowpack height evolution
        - Rule-based LOC detections (iterative window approach)
        - ML LOC detections at multiple probability thresholds
        
        Args:
            sample_pro_file: Path to test .pro file
            output_dir: Directory to save output plots
            trained_model_path: Path to trained model (or None if unavailable)
        """
        # Load profile
        profile = SnowpackProfile(sample_pro_file)
        
        if profile.data is None:
            pytest.skip("Could not parse .pro file")
        
        # Extract data
        timestamps = pd.to_datetime(profile.data.timestamp.values)
        
        # Get snowpack surface height (max height at each timestamp)
        heights_raw = profile.data.height.values
        if heights_raw.ndim == 2:
            heights = np.nanmax(heights_raw, axis=1)
        else:
            heights = heights_raw
        
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        print(f"\nProcessing: {sample_pro_file.name}")
        print(f"Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # =================================================================
        # 1. Collect Rule-Based Detections (Iterative Window Approach)
        # =================================================================
        rule_based_points = []
        test_dates = pd.date_range(start_time, end_time, freq='7D')
        
        print(f"Rule-based detection: Testing {len(test_dates)} windows...")
        for test_date in test_dates:
            central_date = _get_closest_synoptic_time(test_date)
            
            # Window for rule-based logic
            w_start = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            w_end = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if we have data in this window
            window_mask = (timestamps >= w_start) & (timestamps <= w_end)
            if not window_mask.any():
                continue
            
            try:
                window_data = profile.data.sel(timestamp=slice(w_start, w_end))
                if len(window_data.timestamp) == 0:
                    continue
                
                # Apply rule-based detection
                loc_result = find_wet_slab_loc(window_data, min_depth_below_surface=0.20)
                
                # Extract height from result
                if loc_result is not None:
                    if 'loc_height' in loc_result:
                        h = loc_result['loc_height']
                        rule_based_points.append({
                            'time': central_date,
                            'height': h
                        })
                    elif 'loc_depth' in loc_result:
                        # Convert depth to height if needed
                        # Get snow surface height at this time
                        idx = np.argmin(np.abs(timestamps - central_date))
                        surface_height = heights[idx]
                        h = surface_height - loc_result['loc_depth']
                        rule_based_points.append({
                            'time': central_date,
                            'height': h
                        })
            except Exception as e:
                # Silently skip windows with errors
                pass
        
        print(f"  Found {len(rule_based_points)} rule-based detections")
        
        # =================================================================
        # 2. Collect ML Detections (Batch Approach using detect_timeseries)
        # =================================================================
        ml_results = {}
        ml_thresholds = [0.3, 0.5, 0.7, 0.9]

        if trained_model_path:
            
            print(f"ML detection using model: {trained_model_path}")
            for threshold in ml_thresholds:
                try:
                    # Instantiate detector with current threshold
                    detector = MLLocDetector(
                        model_path=trained_model_path,
                        probability_threshold=threshold
                    )
                    
                    # Use efficient batch processing method
                    # Returns DataFrame with index=timestamp, columns=[loc_height, stall_probability, rank]
                    df_results = detector.detect_timeseries(
                        xr_data=profile.data,
                        top_n=1,  # Only highest probability LOC per timestamp
                        return_all_candidates=False
                    )
                    
                    if not df_results.empty:
                        ml_results[f'ml_{threshold}'] = df_results
                        print(f"  Threshold {threshold}: {len(df_results)} detections")
                        
                except Exception as e:
                    print(f"  ML detection failed for threshold {threshold}: {e}")
        else:
            print("Skipping ML detection (no model found)")

        # =================================================================
        # 3. Create the Comparison Plot
        # =================================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot snowpack height as baseline
        ax.plot(timestamps, heights, 'k-', linewidth=1, alpha=0.3, 
               label='Snowpack Height')
        ax.fill_between(timestamps, 0, heights, alpha=0.1, color='gray')
        
        # Plot Rule-Based detections
        if rule_based_points:
            r_times = [d['time'] for d in rule_based_points]
            r_heights = [d['height'] for d in rule_based_points]
            ax.scatter(r_times, r_heights,
                      color='red', s=100, marker='o',
                      alpha=0.8, label='Rule-based LOC', zorder=5)
        
        # Plot ML Detections at different thresholds
        ml_styles = {
            'ml_0.3': ('pink', 'x', '0.3'),
            'ml_0.5': ('orange', 'o', '0.5'),
            'ml_0.7': ('purple', 's', '0.7'),
            'ml_0.9': ('blue', '^', '0.9'),
        }
        
        for key, (color, marker, thresh_label) in ml_styles.items():
            if key in ml_results:
                df = ml_results[key]
                # df.index is timestamp, df['loc_height'] is the Y value
                ax.scatter(df.index, df['loc_height'],
                          c=color, s=80, marker=marker,
                          alpha=0.9, label=f'ML (p≥{thresh_label})', zorder=4)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title(
            f'LOC Detection Comparison - {sample_pro_file.stem}\n'
            f'Period: {start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}',
            fontsize=14, fontweight='bold'
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f"{sample_pro_file.stem}_loc_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Plot saved: {output_file}")
        assert output_file.exists(), "Output file was not created"

    def test_plot_multiple_files(
        self,
        real_pro_files: List[Path],
        output_dir: Path
    ) -> None:
        """
        Generate comparison plots for multiple .pro files.
        
        This creates simple height plots for the first 3 files to verify
        data loading and basic plotting functionality.
        
        Args:
            real_pro_files: List of available .pro files
            output_dir: Directory to save output plots
        """
        plots_created = 0
        
        for pro_file in real_pro_files[:3]:  # Limit to first 3 files
            profile = SnowpackProfile(pro_file)
            if profile.data is None:
                continue
            
            timestamps = pd.to_datetime(profile.data.timestamp.values)
            heights_raw = profile.data.height.values
            
            if heights_raw.ndim == 2:
                heights = np.nanmax(heights_raw, axis=1)
            else:
                heights = heights_raw
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, heights, 'k-', linewidth=1.5)
            ax.fill_between(timestamps, 0, heights, alpha=0.2, color='skyblue')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Height (m)')
            ax.set_title(f'Snowpack Height - {pro_file.stem}')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save
            output_file = output_dir / f"{pro_file.stem}_height.png"
            plt.savefig(output_file, dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"Created: {output_file}")
            plots_created += 1
        
        assert plots_created > 0, "No plots were created"

    def test_plot_detection_timeline(
        self,
        sample_pro_file: Path,
        output_dir: Path
    ) -> None:
        """
        Create a timeline showing when LOC detections occur (Rule-based).
        
        This creates a simple temporal visualization of detection events,
        useful for understanding detection frequency and patterns.
        
        Args:
            sample_pro_file: Path to test .pro file
            output_dir: Directory to save output plots
        """
        profile = SnowpackProfile(sample_pro_file)
        if profile.data is None:
            pytest.skip("Could not parse .pro file")
        
        timestamps = pd.to_datetime(profile.data.timestamp.values)
        detection_times = []
        
        # Sample every 3 days to avoid excessive computation
        test_dates = pd.date_range(timestamps[0], timestamps[-1], freq='3D')
        
        for test_date in test_dates:
            central_date = _get_closest_synoptic_time(test_date)
            w_start = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            w_end = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                window_data = profile.data.sel(timestamp=slice(w_start, w_end))
                if len(window_data.timestamp) == 0:
                    continue
                
                loc_result = find_wet_slab_loc(window_data)
                # Check if valid result exists
                if loc_result is not None:
                    detection_times.append(central_date)
            except Exception:
                pass
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(14, 4))
        if detection_times:
            ax.scatter(detection_times, [1] * len(detection_times),
                      s=100, c='red', alpha=0.6, marker='|')
        
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.set_xlabel('Date')
        ax.set_title(f'LOC Detection Timeline (Rule-Based) - {sample_pro_file.stem}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f"{sample_pro_file.stem}_timeline.png"
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"Created timeline: {output_file}")
        assert output_file.exists()


# ---------------------------------------------------------------------------
# Main - Run without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run tests standalone without pytest.
    
    This is useful for quick validation during development.
    """
    print("Running LOC detection plotting tests...")
    print("=" * 80)
    
    # Setup
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get .pro files
    if not FIXTURES_DIR.exists():
        print(f"Error: Fixtures directory not found at {FIXTURES_DIR}")
        exit(1)
    
    pro_files = list(FIXTURES_DIR.glob("*.pro"))
    if not pro_files:
        print(f"Error: No .pro files found in {FIXTURES_DIR}")
        exit(1)
    
    print(f"Found {len(pro_files)} .pro files")
    
    # Find model
    model_path = None
    potential_paths = [
        Path("src/wetting_front_tracker/assets/models/v1"),
        Path("results/trained_models/latest/trained_model"),
    ]
    
    for path in potential_paths:
        if path.exists() and (path / "model.joblib").exists():
            model_path = path
            print(f"Found model at: {model_path}")
            break
    
    if not model_path:
        print("Warning: No trained model found - ML tests will be skipped")
    
    # Run tests
    tester = TestPlotLOCDetection()
    
    print("\n1. Creating seasonal comparison plot...")
    tester.test_plot_season_comparison(pro_files[0], output_dir, model_path)
    
    print("\n2. Creating height plots...")
    tester.test_plot_multiple_files(pro_files, output_dir)
    
    print("\n3. Creating detection timeline...")
    tester.test_plot_detection_timeline(pro_files[0], output_dir)
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print(f"Plots saved to: {output_dir}")