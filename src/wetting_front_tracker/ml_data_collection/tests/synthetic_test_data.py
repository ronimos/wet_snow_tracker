"""
synthetic_test_data.py
======================

Generates synthetic snowpack profiles for testing stall detection.

Creates realistic test data with known properties to validate detection logic
without requiring large collections of real .pro files.

Author: Ron Simon
Created: November 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, Tuple


class SyntheticProfileGenerator:
    """Generates synthetic snowpack profiles with controlled properties."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def create_stall_scenario(
        self,
        scenario: str = "perfect_stall"
    ) -> Tuple[xr.Dataset, Dict]:
        """
        Create a test scenario with expected results.
        
        Args:
            scenario: Type of scenario to create
            
        Returns:
            (profile_dataset, expected_results)
        """
        if scenario == "perfect_stall":
            return self._create_perfect_stall()
        elif scenario == "refreezing":
            return self._create_refreezing_stall()
        elif scenario == "lwc_drop":
            return self._create_lwc_drop_stall()
        elif scenario == "no_barrier":
            return self._create_no_barrier()
        elif scenario == "soil_moisture":
            return self._create_soil_moisture()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _create_perfect_stall(self) -> Tuple[xr.Dataset, Dict]:
        """
        Create a perfect stall event:
        - Clear capillary barrier at 0.35m
        - Sustained LWC >= 4% above barrier
        - Stable height for 24 hours
        - Temperature >= 0°C throughout
        """
        # Time series: 48 hours, hourly data
        times = pd.date_range('2025-05-01', periods=48, freq='h')
        
        # Heights: Create a stable wetting front at 0.35m for hours 12-36
        heights = np.full(48, np.nan)
        
        # Wetting front arrives
        heights[0:6] = np.nan  # No front yet
        heights[6:12] = np.linspace(0.10, 0.35, 6)  # Front descends
        heights[12:36] = 0.35 + self.rng.normal(0, 0.01, 24)  # Stalls (24h)
        heights[36:48] = np.linspace(0.35, 0.20, 12)  # Continues down (refreezes)
        
        # Create full snowpack profile
        n_layers = 15
        layer_heights = np.linspace(0, 1.5, n_layers)  # 0 to 1.5m
        
        # Create realistic properties
        # Fine snow above 0.35m, coarse below (capillary barrier)
        grain_sizes = np.where(
            layer_heights > 0.35,
            0.5 + self.rng.normal(0, 0.05, n_layers),  # Fine above
            1.2 + self.rng.normal(0, 0.1, n_layers)    # Coarse below
        )
        
        # Density increases with depth
        densities = 200 + (layer_heights * 150) + self.rng.normal(0, 20, n_layers)
        densities = np.clip(densities, 150, 500)
        
        # Temperature: warm (melting conditions)
        temperatures = 273.15 + self.rng.normal(0.5, 0.2, n_layers)
        
        # LWC: High above barrier during stall period
        lwc = np.zeros((len(times), n_layers))
        for i, t in enumerate(times):
            if 12 <= i < 36:  # During stall
                # High LWC above 0.35m, low below
                lwc[i, :] = np.where(
                    layer_heights > 0.35,
                    0.05 + self.rng.normal(0, 0.01, n_layers),  # 5% above
                    0.01 + self.rng.normal(0, 0.005, n_layers)  # 1% below
                )
            else:
                # Low LWC everywhere else
                lwc[i, :] = self.rng.normal(0.01, 0.005, n_layers)
        
        lwc = np.clip(lwc, 0, 0.15)
        
        # Grain types: 770 for wetting front, other values for non-wetting
        grain_types = np.where(
            layer_heights > 0.35,
            770,  # Wetting front grain type
            200   # Non-wetting front
        ).astype(int)
        
        # Create xarray dataset
        profile = xr.Dataset({
            'density': (['time', 'height'], np.tile(densities, (len(times), 1))),
            'temperature': (['time', 'height'], np.tile(temperatures, (len(times), 1))),
            'grain_size': (['time', 'height'], np.tile(grain_sizes, (len(times), 1))),
            'grain_type': (['time', 'height'], np.tile(grain_types, (len(times), 1))),
            'lwc': (['time', 'height'], lwc),
        }, coords={
            'time': times,
            'height': layer_heights
        })
        
        expected = {
            'should_detect': True,
            'stall_height': 0.35,
            'duration_hours': 24,
            'start_time': times[12],
            'end_time': times[36],
            'min_lwc': 0.04,
            'min_temp': 0.0,
            'scenario': 'perfect_stall'
        }
        
        return profile, expected
    
    def _create_refreezing_stall(self) -> Tuple[xr.Dataset, Dict]:
        """
        Stall that ends due to refreezing.
        Temperature drops below 0°C, ending the stall.
        """
        profile, expected = self._create_perfect_stall()
        
        # Modify temperature to drop below freezing after hour 30
        times_after_30 = slice(30, None)
        profile['temperature'][times_after_30, :] = 273.15 - \
            np.linspace(0, 2, len(profile.time[times_after_30]))[:, np.newaxis]
        
        # Grain types change from wetting front (770) to frozen (not 770)
        profile['grain_type'][times_after_30, :] = 200  # No longer wetting front
        
        expected.update({
            'scenario': 'refreezing',
            'end_reason': 'temperature_drop',
            'duration_hours': 18  # Ends at hour 30 instead of 36
        })
        
        return profile, expected
    
    def _create_lwc_drop_stall(self) -> Tuple[xr.Dataset, Dict]:
        """
        Stall that ends due to LWC dropping below threshold.
        Water drains away, ending the stall.
        """
        profile, expected = self._create_perfect_stall()
        
        # LWC drops below threshold after hour 28
        times_after_28 = slice(28, None)
        profile['lwc'][times_after_28, :] = 0.02  # Below 4% threshold
        
        expected.update({
            'scenario': 'lwc_drop',
            'end_reason': 'lwc_decrease',
            'duration_hours': 16  # Ends at hour 28 instead of 36
        })
        
        return profile, expected
    
    def _create_no_barrier(self) -> Tuple[xr.Dataset, Dict]:
        """
        No capillary barrier - wetting front descends continuously.
        Should NOT detect a stall.
        """
        times = pd.date_range('2025-05-01', periods=48, freq='h')
        
        # Wetting front descends continuously (no stall)
        heights = np.linspace(1.0, 0.1, 48)  # Steady descent
        
        n_layers = 15
        layer_heights = np.linspace(0, 1.5, n_layers)
        
        # Uniform properties (no barrier)
        grain_sizes = np.full(n_layers, 0.8) + self.rng.normal(0, 0.05, n_layers)
        densities = 250 + self.rng.normal(0, 20, n_layers)
        temperatures = 273.15 + self.rng.normal(0.5, 0.1, n_layers)
        grain_types = np.full(n_layers, 770, dtype=int)  # Wetting front grain type
        
        # LWC follows wetting front
        lwc = np.zeros((len(times), n_layers))
        for i, h in enumerate(heights):
            lwc[i, :] = np.where(
                layer_heights >= h,
                0.05,  # Wet above
                0.01   # Dry below
            )
        
        profile = xr.Dataset({
            'density': (['time', 'height'], np.tile(densities, (len(times), 1))),
            'temperature': (['time', 'height'], np.tile(temperatures, (len(times), 1))),
            'grain_size': (['time', 'height'], np.tile(grain_sizes, (len(times), 1))),
            'grain_type': (['time', 'height'], np.tile(grain_types, (len(times), 1))),
            'lwc': (['time', 'height'], lwc),
        }, coords={
            'time': times,
            'height': layer_heights
        })
        
        expected = {
            'should_detect': False,
            'scenario': 'no_barrier',
            'reason': 'continuous_descent'
        }
        
        return profile, expected
    
    def _create_soil_moisture(self) -> Tuple[xr.Dataset, Dict]:
        """
        False positive: soil moisture detection.
        Very long "stall" near ground - should be filtered out.
        """
        profile, expected = self._create_perfect_stall()
        
        # Move stall very close to ground
        expected.update({
            'should_detect': False,
            'stall_height': 0.03,  # Too close to ground
            'scenario': 'soil_moisture',
            'reason': 'too_close_to_ground'
        })
        
        return profile, expected


# ===========================================================================
# Testing Utilities
# ===========================================================================

def create_profile_at_time(
    full_profile: xr.Dataset,
    timestamp: datetime
) -> pd.DataFrame:
    """
    Extract a single timestep as a DataFrame.
    
    Args:
        full_profile: Full xarray dataset
        timestamp: Time to extract
        
    Returns:
        DataFrame with height and all variables
    """
    profile_at_time = full_profile.sel(time=timestamp, method='nearest')
    
    df = pd.DataFrame({
        'height': profile_at_time.height.values
    })
    
    for var in ['density', 'temperature', 'grain_size', 'grain_type', 'lwc']:
        if var in profile_at_time:
            df[var] = profile_at_time[var].values
    
    return df


# ===========================================================================
# Example Usage
# ===========================================================================

if __name__ == '__main__':
    """Demonstrate synthetic data generation."""
    
    print("="*70)
    print("SYNTHETIC TEST DATA GENERATOR")
    print("="*70)
    
    generator = SyntheticProfileGenerator()
    
    scenarios = [
        'perfect_stall',
        'refreezing',
        'lwc_drop',
        'no_barrier',
        'soil_moisture'
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario.upper().replace('_', ' ')}")
        print("-" * 70)
        
        profile, expected = generator.create_stall_scenario(scenario)
        
        print(f"Time points: {len(profile.time)}")
        print(f"Layers: {len(profile.height)}")
        print(f"Height range: {profile.height.min().values:.2f}m - "
              f"{profile.height.max().values:.2f}m")
        print(f"Should detect: {expected.get('should_detect', 'N/A')}")
        
        if expected.get('should_detect'):
            print(f"Expected stall height: {expected.get('stall_height'):.2f}m")
            print(f"Expected duration: {expected.get('duration_hours')}h")
        else:
            print(f"Reason for no detection: {expected.get('reason', 'N/A')}")
        
        # Show LWC stats
        lwc_mean = profile['lwc'].mean().values
        lwc_max = profile['lwc'].max().values
        print(f"LWC range: {lwc_mean:.3f} (mean) - {lwc_max:.3f} (max)")
    
    print("\n" + "="*70)
    print("Synthetic data ready for testing!")
    print("="*70)
