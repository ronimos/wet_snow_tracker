#!/bin/bash
# ============================================================================
# Setup Test Environment for Wetting Front Tracker
# ============================================================================
set -e

echo "🚀 Setting up test environment for Wetting Front Tracker..."
echo ""

# ============================================================================
# 1. Check Prerequisites
# ============================================================================

echo "📋 Checking prerequisites..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo ""
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi
echo "✓ uv is installed"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found. Are you in the project root?"
    exit 1
fi
echo "✓ Found pyproject.toml"

# ============================================================================
# 2. Create Virtual Environment
# ============================================================================

echo ""
echo "📦 Setting up virtual environment..."

if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists at .venv"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing .venv..."
        rm -rf .venv
        uv venv
        echo "✓ Recreated virtual environment"
    else
        echo "✓ Using existing virtual environment"
    fi
else
    uv venv
    echo "✓ Created virtual environment"
fi

# ============================================================================
# 3. Install Dependencies
# ============================================================================

echo ""
echo "📥 Installing dependencies..."

# Activate virtual environment
source .venv/bin/activate

# Install package with test dependencies
echo "Installing package with test dependencies..."
uv pip install -e ".[test]"

echo "✓ Installed all dependencies"

# ============================================================================
# 4. Create Test Directory Structure
# ============================================================================

echo ""
echo "📁 Creating test directory structure..."

mkdir -p tests/{unit,integration,fixtures,test_data}
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/fixtures/__init__.py

echo "✓ Created test directories"

# ============================================================================
# 5. Create Sample Test Data Files
# ============================================================================

echo ""
echo "📄 Creating sample test data..."

# Create test polygons GeoJSON
cat > tests/test_data/test_polygons.geojson << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-105.5, 39.5],
          [-105.4, 39.5],
          [-105.4, 39.6],
          [-105.5, 39.6],
          [-105.5, 39.5]
        ]]
      },
      "properties": {
        "pathName": "Test Path 1",
        "id": "test_001"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-105.6, 39.6],
          [-105.5, 39.6],
          [-105.5, 39.7],
          [-105.6, 39.7],
          [-105.6, 39.6]
        ]]
      },
      "properties": {
        "pathName": "Test Path 2",
        "id": "test_002"
      }
    }
  ]
}
EOF

# Create test locations CSV
cat > tests/test_data/test_locations.csv << 'EOF'
latitude,longitude,aspect,altitude,stationName,path
39.52,-105.47,N,3500,TestStation_N,test_station.pro
39.52,-105.47,E,3500,TestStation_E,test_station.pro
39.52,-105.47,S,3500,TestStation_S,test_station.pro
39.52,-105.47,W,3500,TestStation_W,test_station.pro
39.65,-105.55,N,3600,TestStation2_N,test_station2.pro
39.65,-105.55,E,3600,TestStation2_E,test_station2.pro
39.65,-105.55,S,3600,TestStation2_S,test_station2.pro
39.65,-105.55,W,3600,TestStation2_W,test_station2.pro
EOF

echo "✓ Created test data files"

# ============================================================================
# 6. Create Example Test Files (if they don't exist)
# ============================================================================

echo ""
echo "📝 Creating example test files..."

if [ ! -f "tests/conftest.py" ]; then
    cat > tests/conftest.py << 'EOF'
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
EOF
    echo "✓ Created tests/conftest.py"
fi

if [ ! -f "tests/unit/test_example.py" ]; then
    cat > tests/unit/test_example.py << 'EOF'
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
EOF
    echo "✓ Created tests/unit/test_example.py"
fi

# ============================================================================
# 7. Verify Installation
# ============================================================================

echo ""
echo "🔍 Verifying installation..."

# Check if pytest is available
if ! pytest --version &> /dev/null; then
    echo "❌ pytest not found. Installation may have failed."
    exit 1
fi
echo "✓ pytest is installed"

# Run a quick test to verify everything works
echo ""
echo "🧪 Running a quick test..."
if pytest tests/unit/test_example.py -v --tb=short; then
    echo "✓ Test run successful!"
else
    echo "⚠️  Some tests failed, but setup is complete"
fi

# ============================================================================
# 8. Print Success Message and Next Steps
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║   ✅ Test Environment Setup Complete!                         ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 What was created:"
echo "   ✓ Virtual environment (.venv)"
echo "   ✓ Test directory structure (tests/)"
echo "   ✓ Sample test data files"
echo "   ✓ Example test files"
echo "   ✓ All test dependencies installed"
echo ""
echo "🚀 Quick Start:"
echo ""
echo "   1. Activate the virtual environment:"
echo "      source .venv/bin/activate"
echo ""
echo "   2. Run all tests:"
echo "      pytest"
echo ""
echo "   3. Run with coverage:"
echo "      pytest --cov"
echo ""
echo "   4. Run unit tests only:"
echo "      pytest tests/unit -v"
echo ""
echo "   5. Run fast tests (skip slow ones):"
echo "      pytest -m 'not slow'"
echo ""
echo "📚 Additional Commands:"
echo ""
echo "   Using make (if you have Makefile):"
echo "     make test              # Run all tests"
echo "     make test-cov          # With coverage"
echo "     make test-unit         # Unit tests only"
echo "     make help              # See all commands"
echo ""
echo "   Using just (if installed):"
echo "     just test              # Run all tests"
echo "     just test-cov          # With coverage"
echo "     just                   # See all commands"
echo ""
echo "📊 View Coverage Report:"
echo "   pytest --cov --cov-report=html"
echo "   open htmlcov/index.html"
echo ""
echo "💡 Pro Tips:"
echo "   • Use 'pytest -x' to stop on first failure"
echo "   • Use 'pytest --lf' to run last failed tests"
echo "   • Use 'pytest -n auto' for parallel execution"
echo "   • Use 'pytest -k pattern' to run specific tests"
echo ""
echo "Happy testing! 🧪"
echo ""