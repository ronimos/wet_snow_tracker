"""Example test file to verify test setup works."""
import pytest
import pandas as pd
from src.wetting_front_tracker.wet_front_tracker import get_total_snow_depth


class TestExample:
    """Example test class."""
    
    def test_basic_assertion(self):
        """Test that basic assertions work."""
        assert 1 + 1 == 2
    
    def test_with_fixture(self, sample_profile_df):
        """Test using a fixture from conftest.py."""
        assert not sample_profile_df.empty
        assert 'height' in sample_profile_df.columns
    
    def test_snow_depth_calculation(self):
        """Test actual function from the codebase."""
        df = pd.DataFrame({'height': [0, 10, 20, 30, 40, 50]})
        depth = get_total_snow_depth(df)
        assert depth == 50.0
