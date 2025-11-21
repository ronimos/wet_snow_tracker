# test_plot_loc_detection.py

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Updated Imports for the new architecture
from wetting_front_tracker.ml_loc_detector import MLLocDetector
from wetting_front_tracker.snowpack_reader import SnowpackProfile
from wetting_front_tracker.wet_front_tracker import find_wet_slab_loc
from wetting_front_tracker.main import _get_closest_synoptic_time


# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_data"
OUTPUT_DIR = Path(__file__).parent / "output_plots"


@pytest.fixture
def output_dir():
    """Create output directory for plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture
def real_pro_files():
    """Get all real .pro files."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")
    
    pro_files = list(FIXTURES_DIR.glob("*.pro"))
    if not pro_files:
        pytest.skip(f"No .pro files found in {FIXTURES_DIR}")
    
    return pro_files


@pytest.fixture
def sample_pro_file(real_pro_files):
    """Get first .pro file for testing."""
    return real_pro_files[0]


@pytest.fixture
def trained_model_path():
    """Locate the trained model directory."""
    # Common paths where the model might be stored
    potential_paths = [
        Path("results/model/trained_model"),
        Path("models/trained"),
        Path("trained_models"),
        Path("src/wetting_front_tracker/tests/fixtures/model") 
    ]
    
    for path in potential_paths:
        if path.exists() and (path / "model.joblib").exists():
            return path
            
    return None


class TestPlotLOCDetection:
    """Visual tests for LOC detection comparison."""
    
    def test_plot_season_comparison(self, sample_pro_file, output_dir, trained_model_path):
        """
        Generate seasonal plot comparing rule-based and ML LOC detection.
        Refactored to use MLLocDetector class.
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
        
        # 1. Collect Rule-Based Detections (Iterative Window Approach)
        rule_based_points = []
        test_dates = pd.date_range(start_time, end_time, freq='7D')
        
        for test_date in test_dates:
            central_date = _get_closest_synoptic_time(test_date)
            # Window for rule-based logic
            w_start = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            w_end = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if we have data in this window
            window_mask = (timestamps >= w_start) & (timestamps <= w_end)
            if not window_mask.any():
                continue
                
            window_data = profile.data.sel(timestamp=slice(w_start, w_end))
            if len(window_data.timestamp) == 0:
                continue
                
            try:
                loc_result = find_wet_slab_loc(window_data)
                # Assuming find_wet_slab_loc returns a dictionary with 'loc_depth' or similar
                # Adjust key based on actual return of find_wet_slab_loc
                if loc_result is not None:
                    # Note: Check if your rule-based function returns depth (from surface)
                    # or height (from ground). The ML detector returns HEIGHT.
                    # Assuming here we plot Height.
                    if 'loc_height' in loc_result:
                        h = loc_result['loc_height']
                        rule_based_points.append({'time': central_date, 'height': h})
                    elif 'loc_depth' in loc_result:
                        # If it returns depth, you might need to convert: surface - depth
                        pass 
            except Exception:
                pass

        # 2. Collect ML Detections (Batch Approach using MLLocDetector)
        ml_results = {}
        ml_thresholds = [0.3, 0.5, 0.7, 0.9]

        if trained_model_path:
            print(f"Using ML model at: {trained_model_path}")
            
            for threshold in ml_thresholds:
                try:
                    # Instantiate the new class-based detector
                    detector = MLLocDetector(
                        model_path=trained_model_path,
                        probability_threshold=threshold
                    )
                    
                    # Use the efficient batch processing method
                    # Returns DataFrame with index=timestamp, cols=[loc_height, stall_probability]
                    df_results = detector.detect_timeseries(profile.data)
                    
                    if not df_results.empty:
                        ml_results[f'ml_{threshold}'] = df_results
                        
                except Exception as e:
                    print(f"ML detection failed for threshold {threshold}: {e}")
        else:
            print("Skipping ML detection (no model found)")

        # 3. Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot snowpack height
        ax.plot(timestamps, heights, 'k-', linewidth=1, alpha=0.3, label='Snowpack Height')
        ax.fill_between(timestamps, 0, heights, alpha=0.1, color='gray')
        
        # Plot Rule-Based
        if rule_based_points:
            r_times = [d['time'] for d in rule_based_points]
            r_heights = [d['height'] for d in rule_based_points]
            ax.scatter(r_times, r_heights, 
                      color='red', s=100, marker='o', 
                      alpha=0.8, label='Rule-based LOC', zorder=5)
        
        # Plot ML Detections
        ml_colors = {
            'ml_0.3': ('blue', 0.2, '0.3'),
            'ml_0.5': ('blue', 0.4, '0.5'),
            'ml_0.7': ('blue', 0.6, '0.7'),
            'ml_0.9': ('blue', 0.9, '0.9')
        }
        
        for key, (color, alpha, thresh_label) in ml_colors.items():
            if key in ml_results:
                df = ml_results[key]
                # df.index is timestamp, df['loc_height'] is the Y value
                ax.scatter(df.index, df['loc_height'],
                          color=color, s=80, marker='x',
                          alpha=alpha, label=f'ML (p>{thresh_label})', zorder=4)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title(f'LOC Detection Comparison - {sample_pro_file.stem}\n'
                    f'Period: {start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}',
                    fontsize=14, fontweight='bold')
        
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
        assert output_file.exists()

    def test_plot_multiple_files(self, real_pro_files, output_dir):
        """Generate comparison plots for multiple .pro files (Basic Height Check)."""
        plots_created = 0
        
        for pro_file in real_pro_files[:3]:
            profile = SnowpackProfile(pro_file)
            if profile.data is None: continue
            
            timestamps = pd.to_datetime(profile.data.timestamp.values)
            heights_raw = profile.data.height.values
            if heights_raw.ndim == 2:
                heights = np.nanmax(heights_raw, axis=1)
            else:
                heights = heights_raw
            
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
            
            output_file = output_dir / f"{pro_file.stem}_height.png"
            plt.savefig(output_file, dpi=120, bbox_inches='tight')
            plt.close()
            plots_created += 1
        
        assert plots_created > 0

    def test_plot_detection_timeline(self, sample_pro_file, output_dir):
        """Create a timeline showing when LOC detections occur (Rule-based)."""
        profile = SnowpackProfile(sample_pro_file)
        if profile.data is None: pytest.skip("Could not parse")
        
        timestamps = pd.to_datetime(profile.data.timestamp.values)
        detection_times = []
        
        # Using a sampling frequency to avoid checking every single hour
        test_dates = pd.date_range(timestamps[0], timestamps[-1], freq='3D')
        
        for test_date in test_dates:
            central_date = _get_closest_synoptic_time(test_date)
            w_start = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            w_end = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
            
            window_data = profile.data.sel(timestamp=slice(w_start, w_end))
            if len(window_data.timestamp) == 0: continue
            
            try:
                loc_result = find_wet_slab_loc(window_data)
                # Check if valid result exists
                if loc_result is not None:
                    detection_times.append(central_date)
            except Exception:
                pass
        
        fig, ax = plt.subplots(figsize=(14, 4))
        if detection_times:
            ax.scatter(detection_times, [1]*len(detection_times),
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
        
        output_file = output_dir / f"{sample_pro_file.stem}_timeline.png"
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        assert output_file.exists()