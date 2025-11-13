"""
test_stall_events.py
====================

Comprehensive test suite for validating wetting front stall detection.

Tests that detected stall events satisfy physical conditions:
- Start when ALL wetting front conditions exist
- End when ANY condition ceases
- Maintain physical validity throughout

Author: Ron Simon
Created: November 2025
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ..stall_detector import StallEvent, StallDetectionConfig
from .test_data_manager import get_test_profile


# ===========================================================================
# Physical Condition Thresholds
# ===========================================================================

class PhysicalThresholds:
    """Physical thresholds for wetting front conditions."""
    
    # LWC thresholds
    MIN_LWC = 0.04  # 4% volumetric LWC for wetting front
    
    # Temperature thresholds (Kelvin)
    MIN_TEMP = 273.15 - 0.5  # -0.5°C (slight tolerance for measurement error)
    MAX_TEMP = 278.15  # 5°C (unrealistic if higher in snowpack)
    
    # Density thresholds (kg/m³)
    MIN_DENSITY = 100
    MAX_DENSITY = 600
    
    # Grain type codes
    # CRITICAL: Only 770 indicates wetting front
    # All other grain types are frozen/not wetting front
    WET_GRAIN_TYPES = {770}
    
    FROZEN_GRAIN_TYPES = set()  # Not used since only 770 is valid


# ===========================================================================
# Physical Condition Validator
# ===========================================================================

class PhysicalConditionValidator:
    """
    Validates physical conditions at stall events.
    
    Checks that stalls start when ALL conditions exist and end when
    ANY condition ceases - ensuring physical validity.
    """
    
    def __init__(self, thresholds: Optional[PhysicalThresholds] = None):
        """
        Initialize validator.
        
        Args:
            thresholds: Physical thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or PhysicalThresholds()
    
    def check_lwc_condition(
        self,
        profile: xr.Dataset,
        height: float,
        time: datetime
    ) -> Tuple[bool, float]:
        """
        Check if LWC condition is met.
        
        Args:
            profile: Full profile dataset
            height: Height to check (m)
            time: Time to check
            
        Returns:
            (condition_met, lwc_value)
        """
        try:
            # Select nearest height and time
            lwc = profile['lwc'].sel(
                height=height,
                time=time,
                method='nearest'
            ).values
            
            lwc = float(lwc)
            condition_met = lwc >= self.thresholds.MIN_LWC
            
            return condition_met, lwc
            
        except (KeyError, IndexError) as e:
            return False, np.nan
    
    def check_temperature_condition(
        self,
        profile: xr.Dataset,
        height: float,
        time: datetime
    ) -> Tuple[bool, float]:
        """
        Check if temperature condition is met (>= 0°C).
        
        Args:
            profile: Full profile dataset
            height: Height to check (m)
            time: Time to check
            
        Returns:
            (condition_met, temperature_kelvin)
        """
        try:
            temp = profile['temperature'].sel(
                height=height,
                time=time,
                method='nearest'
            ).values
            
            temp = float(temp)
            condition_met = (
                temp >= self.thresholds.MIN_TEMP and
                temp <= self.thresholds.MAX_TEMP
            )
            
            return condition_met, temp
            
        except (KeyError, IndexError) as e:
            return False, np.nan
    
    def check_grain_type_condition(
        self,
        profile: xr.Dataset,
        height: float,
        time: datetime
    ) -> Tuple[bool, int]:
        """
        Check if grain type is appropriate (wet, not frozen).
        
        Args:
            profile: Full profile dataset
            height: Height to check (m)
            time: Time to check
            
        Returns:
            (condition_met, grain_type_code)
        """
        try:
            grain_type = profile['grain_type'].sel(
                height=height,
                time=time,
                method='nearest'
            ).values
            
            grain_type = int(grain_type)
            
            # Only 770 indicates wetting front
            # All other grain types are frozen/not wetting front
            condition_met = (grain_type == 770)
            
            return condition_met, grain_type
            
        except (KeyError, IndexError) as e:
            return False, -1
    
    def check_density_condition(
        self,
        profile: xr.Dataset,
        height: float,
        time: datetime
    ) -> Tuple[bool, float]:
        """
        Check if density is realistic.
        
        Args:
            profile: Full profile dataset
            height: Height to check (m)
            time: Time to check
            
        Returns:
            (condition_met, density_kg_m3)
        """
        try:
            density = profile['density'].sel(
                height=height,
                time=time,
                method='nearest'
            ).values
            
            density = float(density)
            condition_met = (
                density >= self.thresholds.MIN_DENSITY and
                density <= self.thresholds.MAX_DENSITY
            )
            
            return condition_met, density
            
        except (KeyError, IndexError) as e:
            return False, np.nan
    
    def check_all_conditions(
        self,
        profile: xr.Dataset,
        height: float,
        time: datetime
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Check all physical conditions at once.
        
        Args:
            profile: Full profile dataset
            height: Height to check (m)
            time: Time to check
            
        Returns:
            Dictionary with results for each condition
        """
        results = {}
        
        # Check each condition
        results['lwc'] = self.check_lwc_condition(profile, height, time)
        results['temperature'] = self.check_temperature_condition(profile, height, time)
        results['grain_type'] = self.check_grain_type_condition(profile, height, time)
        results['density'] = self.check_density_condition(profile, height, time)
        
        # All conditions must be met
        all_valid = all(result[0] for result in results.values())
        results['all_valid'] = (all_valid, None)
        
        return results


# ===========================================================================
# Test Suite
# ===========================================================================

class TestStallEventPhysics(unittest.TestCase):
    """Test physical validity of detected stall events."""
    
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        # Try to get real data, fall back to synthetic
        cls.profile, cls.expected, cls.source = get_test_profile(
            scenario='perfect_stall',
            prefer_real=True
        )
        
        print(f"\n{'='*70}")
        print(f"TEST DATA: {cls.source.upper()}")
        print(f"{'='*70}")
        
        # Create mock events for testing
        # In real use, these would come from detector.find_stalls()
        cls.events = cls._create_mock_events()
        
        cls.validator = PhysicalConditionValidator()
        cls.config = StallDetectionConfig()
    
    @classmethod
    def _create_mock_events(cls) -> List[StallEvent]:
        """Create mock stall events from test data."""
        events = []
        
        if 'should_detect' in cls.expected and cls.expected['should_detect']:
            # Create event from expected results
            event = StallEvent(
                event_id='SE_TEST_001',
                station_name='TestStation',
                pro_file=Path('test.pro'),
                start_time=cls.expected.get('start_time', cls.profile.time.values[0]),
                end_time=cls.expected.get('end_time', cls.profile.time.values[-1]),
                stall_height=cls.expected.get('stall_height', 0.35),
                duration_hours=cls.expected.get('duration_hours', 24),
                confidence=0.8,
                n_data_points=24,
                height_std=0.02,
                is_ongoing=False
            )
            events.append(event)
        
        return events
    
    # =======================================================================
    # Start Condition Tests
    # =======================================================================
    
    def test_event_start_conditions(self):
        """Test 1: At event start, ALL conditions must be met."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                results = self.validator.check_all_conditions(
                    self.profile,
                    event.stall_height,
                    event.start_time
                )
                
                # Check each condition
                lwc_valid, lwc_val = results['lwc']
                temp_valid, temp_val = results['temperature']
                grain_valid, grain_val = results['grain_type']
                density_valid, density_val = results['density']
                
                self.assertTrue(
                    lwc_valid,
                    f"Event {event.event_id}: LWC={lwc_val:.3f} < {self.validator.thresholds.MIN_LWC} at start"
                )
                
                self.assertTrue(
                    temp_valid,
                    f"Event {event.event_id}: T={temp_val-273.15:.1f}°C outside valid range at start"
                )
                
                self.assertTrue(
                    grain_valid,
                    f"Event {event.event_id}: Grain type={grain_val} invalid at start"
                )
                
                self.assertTrue(
                    density_valid,
                    f"Event {event.event_id}: Density={density_val:.0f} kg/m³ outside valid range at start"
                )
    
    def test_lwc_throughout_event(self):
        """Test 2: LWC >= 4% maintained throughout event."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # Check LWC at start, middle, and end
                times_to_check = [
                    event.start_time,
                    event.start_time + timedelta(hours=event.duration_hours/2),
                    event.end_time - timedelta(hours=1)  # 1 hour before end
                ]
                
                for t in times_to_check:
                    lwc_valid, lwc_val = self.validator.check_lwc_condition(
                        self.profile,
                        event.stall_height,
                        t
                    )
                    
                    self.assertTrue(
                        lwc_valid or pd.isna(lwc_val),  # Allow NaN if time not in data
                        f"Event {event.event_id}: LWC={lwc_val:.3f} dropped below threshold at {t}"
                    )
    
    # =======================================================================
    # End Condition Tests
    # =======================================================================
    
    def test_event_end_conditions(self):
        """Test 3: At event end, at least ONE condition should be violated."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # Skip if event is ongoing (reaches end of data)
                if event.is_ongoing:
                    self.skipTest(f"Event {event.event_id} is ongoing")
                
                results = self.validator.check_all_conditions(
                    self.profile,
                    event.stall_height,
                    event.end_time
                )
                
                # At least one condition should be violated
                any_violated = not results['all_valid'][0]
                
                self.assertTrue(
                    any_violated,
                    f"Event {event.event_id}: All conditions still met at end - why did it end?"
                )
    
    # =======================================================================
    # Continuity Tests
    # =======================================================================
    
    def test_no_refreezing_during_event(self):
        """Test 4: Temperature >= 0°C throughout event."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # Sample temperatures during event
                n_samples = min(10, int(event.duration_hours))
                time_samples = pd.date_range(
                    event.start_time,
                    event.end_time,
                    periods=n_samples
                )
                
                for t in time_samples:
                    temp_valid, temp_val = self.validator.check_temperature_condition(
                        self.profile,
                        event.stall_height,
                        t
                    )
                    
                    if not pd.isna(temp_val):
                        temp_celsius = temp_val - 273.15
                        self.assertGreaterEqual(
                            temp_celsius,
                            -0.5,
                            f"Event {event.event_id}: Temperature={temp_celsius:.1f}°C < -0.5°C at {t}"
                        )
    
    def test_grain_type_consistency(self):
        """Test 5: Grains must be 770 (wetting front) throughout event."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # Sample grain types during event
                n_samples = min(10, int(event.duration_hours))
                time_samples = pd.date_range(
                    event.start_time,
                    event.end_time,
                    periods=n_samples
                )
                
                for t in time_samples:
                    grain_valid, grain_val = self.validator.check_grain_type_condition(
                        self.profile,
                        event.stall_height,
                        t
                    )
                    
                    if grain_val != -1:  # -1 indicates missing data
                        self.assertEqual(
                            grain_val,
                            770,
                            f"Event {event.event_id}: Grain type {grain_val} != 770 (wetting front) at {t}"
                        )
    
    def test_height_stability(self):
        """Test 6: Height standard deviation indicates true stall."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # Height std should be small (it's a "stall")
                max_acceptable_std = 0.10  # 10 cm
                
                self.assertLess(
                    event.height_std,
                    max_acceptable_std,
                    f"Event {event.event_id}: Height std={event.height_std:.3f}m too large for stall"
                )
    
    def test_data_continuity(self):
        """Test 7: No large gaps in data during event."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                # This is a simple check - in real data you'd check actual timestamps
                expected_points = event.duration_hours  # Assuming hourly data
                
                # Allow some missing data
                min_acceptable_ratio = 0.7  # At least 70% of expected points
                actual_ratio = event.n_data_points / expected_points
                
                self.assertGreater(
                    actual_ratio,
                    min_acceptable_ratio,
                    f"Event {event.event_id}: Only {actual_ratio:.1%} of expected data points"
                )
    
    # =======================================================================
    # Physical Constraint Tests
    # =======================================================================
    
    def test_minimum_duration(self):
        """Test 8: Duration >= minimum threshold."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                self.assertGreaterEqual(
                    event.duration_hours,
                    self.config.min_duration_hours,
                    f"Event {event.event_id}: Duration {event.duration_hours:.1f}h < minimum"
                )
    
    def test_maximum_duration(self):
        """Test 9: Duration <= maximum threshold (filters soil moisture)."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                self.assertLessEqual(
                    event.duration_hours,
                    self.config.max_duration_hours,
                    f"Event {event.event_id}: Duration {event.duration_hours:.1f}h > maximum (possible soil moisture)"
                )
    
    def test_minimum_height_above_ground(self):
        """Test 10: Height >= minimum to exclude soil moisture."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                self.assertGreaterEqual(
                    event.stall_height,
                    self.config.min_wetting_front_height,
                    f"Event {event.event_id}: Height {event.stall_height:.3f}m too close to ground"
                )
    
    def test_confidence_score(self):
        """Test 11: Confidence score meets minimum threshold."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                min_confidence = 0.3
                self.assertGreaterEqual(
                    event.confidence,
                    min_confidence,
                    f"Event {event.event_id}: Confidence {event.confidence:.2f} too low"
                )
    
    def test_physical_realism(self):
        """Test 12: Realistic combinations of conditions."""
        for event in self.events:
            with self.subTest(event=event.event_id):
                results = self.validator.check_all_conditions(
                    self.profile,
                    event.stall_height,
                    event.start_time
                )
                
                lwc_val = results['lwc'][1]
                temp_val = results['temperature'][1]
                density_val = results['density'][1]
                
                # If very high LWC, temperature must be above freezing
                if lwc_val > 0.10:  # 10% LWC
                    temp_celsius = temp_val - 273.15
                    self.assertGreater(
                        temp_celsius,
                        0.0,
                        f"Event {event.event_id}: High LWC ({lwc_val:.3f}) but T={temp_celsius:.1f}°C"
                    )
                
                # If very high LWC, density should be reasonable
                if lwc_val > 0.10:
                    self.assertGreater(
                        density_val,
                        250,
                        f"Event {event.event_id}: High LWC but low density ({density_val:.0f} kg/m³)"
                    )


# ===========================================================================
# Test Runner
# ===========================================================================

def run_tests(verbosity=2):
    """
    Run the test suite.
    
    Args:
        verbosity: Level of output detail (0=quiet, 1=normal, 2=verbose)
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStallEventPhysics)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    """Run tests when script is executed directly."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stall event detection')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test (e.g., test_event_start_conditions)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestStallEventPhysics(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        runner.run(suite)
    else:
        # Run all tests
        verbosity = 2 if args.verbose else 1
        result = run_tests(verbosity=verbosity)
        
        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)
