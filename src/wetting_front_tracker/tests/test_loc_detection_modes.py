"""
test_loc_detection_modes.py
============================

Focused tests for LOC (Layer of Concern) detection with real SNOWPACK data.

Tests different interface detection modes:
- rule_based: Traditional capillary barrier detection
- ml_only: ML-based detection
- hybrid: ML with rule-based fallback

Place real .pro files in: src/wetting_front_tracker/tests/fixtures/real_data/
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from wetting_front_tracker.main import (
    process_single_profile,
    get_loc_detection_function,
    _get_closest_synoptic_time
)
from wetting_front_tracker.param_config import MLModelConfig
from wetting_front_tracker.snowpack_reader import SnowpackProfile


# Path to real data fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_data"


@pytest.fixture
def real_pro_files():
    """Get all real .pro files from fixtures directory."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")
    
    pro_files = list(FIXTURES_DIR.glob("*.pro"))
    if not pro_files:
        pytest.skip(f"No .pro files found in {FIXTURES_DIR}")
    
    return pro_files


@pytest.fixture
def sample_real_pro(real_pro_files):
    """Get first real .pro file for testing."""
    return real_pro_files[0]


class TestLOCDetectionModes:
    """Test different LOC detection modes on real data."""
    
    def test_rule_based_mode(self, sample_real_pro, tmp_path):
        """Test rule-based LOC detection on real SNOWPACK file."""
        profile = SnowpackProfile(sample_real_pro)
        
        if profile.data is None:
            pytest.skip("Could not parse .pro file")
        
        # Get a valid date from the file
        last_timestamp = pd.to_datetime(profile.data.timestamp.values[-1])
        central_date = _get_closest_synoptic_time(last_timestamp)
        start_date = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Test rule-based detection
        result = process_single_profile(
            sample_real_pro,
            "N",
            start_date,
            end_date,
            central_date,
            tmp_path,
            loc_detector=None  # Uses rule-based
        )
        
        # Should complete without error (result can be None or dict)
        assert result is None or isinstance(result, dict)
    
    def test_ml_only_mode_with_model(self, sample_real_pro, tmp_path):
        """Test ML-only mode if model available."""
        # Check if trained model exists
        model_paths = [
            Path("models/trained"),
            Path("trained_models"),
            tmp_path / "mock_model"
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists() and (path / "model_config.json").exists():
                model_path = path
                break
        
        if model_path is None:
            pytest.skip("No trained ML model available")
        
        config = MLModelConfig(
            enabled=True,
            model_path=model_path,
            probability_threshold=0.5
        )
        
        loc_detector = get_loc_detection_function("ml_only", config)
        
        profile = SnowpackProfile(sample_real_pro)
        if profile.data is None:
            pytest.skip("Could not parse .pro file")
        
        last_timestamp = pd.to_datetime(profile.data.timestamp.values[-1])
        central_date = _get_closest_synoptic_time(last_timestamp)
        start_date = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
        
        result = process_single_profile(
            sample_real_pro,
            "N",
            start_date,
            end_date,
            central_date,
            tmp_path,
            loc_detector=loc_detector
        )
        
        assert result is None or isinstance(result, dict)
    
    def test_hybrid_mode(self, sample_real_pro, tmp_path):
        """Test hybrid mode (ML with rule-based fallback)."""
        config = MLModelConfig(
            enabled=True,
            model_path=tmp_path / "nonexistent",  # Will fallback
            probability_threshold=0.5
        )
        
        # Hybrid mode should fallback to rule-based if ML fails
        loc_detector = get_loc_detection_function("hybrid", config)
        
        profile = SnowpackProfile(sample_real_pro)
        if profile.data is None:
            pytest.skip("Could not parse .pro file")
        
        last_timestamp = pd.to_datetime(profile.data.timestamp.values[-1])
        central_date = _get_closest_synoptic_time(last_timestamp)
        start_date = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
        
        result = process_single_profile(
            sample_real_pro,
            "N",
            start_date,
            end_date,
            central_date,
            tmp_path,
            loc_detector=loc_detector
        )
        
        assert result is None or isinstance(result, dict)
    
    def test_multiple_files(self, real_pro_files, tmp_path):
        """Test rule-based detection on all available .pro files."""
        results = []
        
        for pro_file in real_pro_files[:5]:  # Test first 5 files
            profile = SnowpackProfile(pro_file)
            
            if profile.data is None:
                continue
            
            last_timestamp = pd.to_datetime(profile.data.timestamp.values[-1])
            central_date = _get_closest_synoptic_time(last_timestamp)
            start_date = (central_date - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            end_date = (central_date + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
            
            result = process_single_profile(
                pro_file,
                "N",
                start_date,
                end_date,
                central_date,
                tmp_path,
                loc_detector=None
            )
            
            results.append({
                'file': pro_file.name,
                'processed': True,
                'found_stall': result is not None
            })
        
        # Should process at least one file
        assert len(results) > 0
        assert all(r['processed'] for r in results)


class TestRealDataValidation:
    """Quick validation of real data files."""
    
    def test_files_exist(self, real_pro_files):
        """Test that .pro files exist."""
        assert len(real_pro_files) > 0
    
    def test_files_loadable(self, real_pro_files):
        """Test that .pro files can be loaded."""
        loaded = 0
        
        for pro_file in real_pro_files:
            try:
                profile = SnowpackProfile(pro_file)
                if profile.data is not None:
                    loaded += 1
            except Exception:
                pass
        
        assert loaded > 0, "No .pro files could be loaded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])