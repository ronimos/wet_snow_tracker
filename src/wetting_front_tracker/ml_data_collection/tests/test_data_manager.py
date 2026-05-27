"""
test_data_manager.py
====================

Manages test data from both real .pro files and synthetic generation.

This allows tests to run with:
1. Synthetic data (fast, controlled, always available)
2. Real data (when available, for integration testing)

Author: Ron Simon
Created: November 2025
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import xarray as xr

from synthetic_test_data import SyntheticProfileGenerator


class TestDataManager:
    """Manages test data sources."""
    
    def __init__(self, test_data_dir: Optional[Path] = None):
        """
        Initialize test data manager.
        
        Args:
            test_data_dir: Directory containing real .pro files for testing
        """
        self.test_data_dir = test_data_dir or Path('tests/data')
        self.synthetic_generator = SyntheticProfileGenerator()
        
        # Track available real files
        self.real_files = self._find_real_files()
    
    def _find_real_files(self) -> Dict[str, Path]:
        """Find available real test files."""
        files = {}
        
        if not self.test_data_dir.exists():
            return files
        
        # Look for specific test files
        test_file_patterns = {
            'perfect_stall': 'test_stall_perfect.pro',
            'refreezing': 'test_stall_refreezing.pro',
            'lwc_drop': 'test_stall_lwc_drop.pro',
            'soil': 'test_stall_soil.pro',
            'no_stall': 'test_no_stall.pro',
            'general': 'test_stall.pro'  # General test file
        }
        
        for key, pattern in test_file_patterns.items():
            file_path = self.test_data_dir / pattern
            if file_path.exists():
                files[key] = file_path
        
        # Also look for any .pro files
        for pro_file in self.test_data_dir.glob('*.pro'):
            if pro_file.stem not in files:
                files[pro_file.stem] = pro_file
        
        return files
    
    def has_real_data(self, scenario: Optional[str] = None) -> bool:
        """
        Check if real test data is available.
        
        Args:
            scenario: Specific scenario name, or None for any
            
        Returns:
            True if real data is available
        """
        if scenario:
            return scenario in self.real_files
        return len(self.real_files) > 0
    
    def get_test_profile(
        self,
        scenario: str = "perfect_stall",
        prefer_real: bool = True
    ) -> Tuple[xr.Dataset, Dict, str]:
        """
        Get a test profile (real or synthetic).
        
        Args:
            scenario: Type of test scenario
            prefer_real: Use real data if available
            
        Returns:
            (profile_dataset, expected_results, source_type)
            where source_type is 'real' or 'synthetic'
        """
        # Try to use real data if preferred and available
        if prefer_real and self.has_real_data(scenario):
            return self._load_real_profile(scenario)
        
        # Fall back to synthetic
        return self._generate_synthetic_profile(scenario)
    
    def _load_real_profile(
        self,
        scenario: str
    ) -> Tuple[xr.Dataset, Dict, str]:
        """Load a real .pro file."""
        from src.wetting_front_tracker.snowpack_reader import SnowpackProfile
        
        pro_file = self.real_files[scenario]
        
        try:
            profile = SnowpackProfile(str(pro_file))
            
            # Expected results need to be determined from actual analysis
            # For now, just mark as unknown
            expected = {
                'source': 'real',
                'file': str(pro_file),
                'needs_analysis': True
            }
            
            return profile.data, expected, 'real'
            
        except Exception as e:
            print(f"Warning: Could not load {pro_file}: {e}")
            print("Falling back to synthetic data")
            return self._generate_synthetic_profile(scenario)
    
    def _generate_synthetic_profile(
        self,
        scenario: str
    ) -> Tuple[xr.Dataset, Dict, str]:
        """Generate synthetic profile."""
        profile, expected = self.synthetic_generator.create_stall_scenario(scenario)
        expected['source'] = 'synthetic'
        return profile, expected, 'synthetic'
    
    def get_test_file_path(self, scenario: str = 'general') -> Optional[Path]:
        """
        Get path to real test file if available.
        
        Args:
            scenario: Which test file to get
            
        Returns:
            Path to file, or None if not available
        """
        return self.real_files.get(scenario)
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """
        List all available test data.
        
        Returns:
            Dictionary with 'real' and 'synthetic' lists
        """
        synthetic_scenarios = [
            'perfect_stall',
            'refreezing',
            'lwc_drop',
            'no_barrier',
            'soil_moisture'
        ]
        
        return {
            'real': list(self.real_files.keys()),
            'synthetic': synthetic_scenarios
        }
    
    def setup_test_data_directory(self):
        """Create test data directory structure."""
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme = self.test_data_dir / 'README.md'
        if not readme.exists():
            readme.write_text("""# Test Data

This directory contains .pro files for testing stall detection.

## Recommended Test Files

Create these by copying from your actual data:

- `test_stall_perfect.pro` - Clean stall with good conditions
- `test_stall_refreezing.pro` - Stall that refreezes
- `test_stall_lwc_drop.pro` - Stall where LWC drops
- `test_stall_soil.pro` - False positive (soil moisture)
- `test_no_stall.pro` - Profile with no stall

## How to Create Test Files

```bash
# Find good examples
python -m src.wetting_front_tracker.ml_data_collection.stall_detector \\
    data/input/YOUR_FILE.pro

# If it has the conditions you want, copy it
cp data/input/YOUR_FILE.pro tests/data/test_stall_perfect.pro
```

## Fallback

If real files aren't available, tests will use synthetic data automatically.
""")
        
        return self.test_data_dir


# ===========================================================================
# Convenience Functions
# ===========================================================================

_global_manager: Optional[TestDataManager] = None


def get_test_data_manager(test_data_dir: Optional[Path] = None) -> TestDataManager:
    """Get global test data manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = TestDataManager(test_data_dir)
    return _global_manager


def get_test_profile(
    scenario: str = "perfect_stall",
    prefer_real: bool = True
) -> Tuple[xr.Dataset, Dict, str]:
    """
    Convenience function to get test profile.
    
    Args:
        scenario: Type of test scenario
        prefer_real: Use real data if available
        
    Returns:
        (profile, expected_results, source_type)
    """
    manager = get_test_data_manager()
    return manager.get_test_profile(scenario, prefer_real)


def has_real_test_data(scenario: Optional[str] = None) -> bool:
    """Check if real test data is available."""
    manager = get_test_data_manager()
    return manager.has_real_data(scenario)


# ===========================================================================
# Example Usage
# ===========================================================================

if __name__ == '__main__':
    """Demonstrate test data management."""
    
    print("="*70)
    print("TEST DATA MANAGER")
    print("="*70)
    
    # Initialize manager
    manager = TestDataManager(Path('tests/data'))
    
    # Setup directory
    print("\n1. Setting up test data directory...")
    test_dir = manager.setup_test_data_directory()
    print(f"   Created: {test_dir}")
    
    # List available data
    print("\n2. Available test data:")
    available = manager.list_available_data()
    print(f"   Real files: {len(available['real'])}")
    for name in available['real']:
        print(f"     - {name}")
    print(f"   Synthetic scenarios: {len(available['synthetic'])}")
    for name in available['synthetic']:
        print(f"     - {name}")
    
    # Get test profiles
    print("\n3. Loading test profiles:")
    scenarios = ['perfect_stall', 'refreezing', 'soil_moisture']
    
    for scenario in scenarios:
        profile, expected, source = manager.get_test_profile(scenario)
        print(f"\n   {scenario}:")
        print(f"     Source: {source}")
        print(f"     Layers: {len(profile['height'])}")
        if 'should_detect' in expected:
            print(f"     Should detect: {expected['should_detect']}")
    
    # Check real data availability
    print("\n4. Real data availability:")
    print(f"   Has any real data: {manager.has_real_data()}")
    print(f"   Has perfect_stall: {manager.has_real_data('perfect_stall')}")
    print(f"   Has refreezing: {manager.has_real_data('refreezing')}")
    
    if not manager.has_real_data():
        print("\n   ðŸ’¡ TIP: Add .pro files to tests/data/ for integration testing")
        print("   Tests will use synthetic data automatically until then.")
    
    print("\n" + "="*70)
    print("Test data manager ready!")
    print("="*70)