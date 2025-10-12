"""Shared test fixtures for all tests."""
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dates():
    """Provides a range of test dates."""
    start = datetime(2025, 5, 1)
    return [start + timedelta(days=i) for i in range(10)]


@pytest.fixture
def sample_profile_df():
    """Creates a sample snow profile DataFrame."""
    return pd.DataFrame({
        'height': [0, 10, 20, 30, 40, 50],
        'grain_type': [300, 400, 450, 500, 300, 200],
        'gs_difference': [0.2, 0.8, -0.6, 0.3, -0.4, 0.1],
        'lwc': [0.0, 0.02, 0.05, 0.08, 0.03, 0.01],
        'density': [200, 220, 250, 280, 240, 210],
        'grain_size': [0.5, 0.8, 1.2, 1.5, 0.9, 0.6],
        'shear_strength': [1.0, 1.2, 0.8, 0.6, 1.1, 1.3]
    })


@pytest.fixture
def sample_metadata():
    """Provides sample station metadata."""
    return {
        'stationName': 'Test Station',
        'latitude': 39.5,
        'longitude': -105.5,
        'altitude': 3500,
        'slopeAngle': 35,
        'slopeAzi': 180,
        'aspect': 'N'
    }
