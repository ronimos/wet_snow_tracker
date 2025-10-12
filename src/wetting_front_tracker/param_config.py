"""
param_config.py
===============

Configuration management for the Wetting Front Tracker application.

This module provides centralized configuration using dataclasses, environment
variables, and validated path management. It replaces the previous approach
of using global variables with a more structured and testable system.

Usage:
    from param_config import config, SnowpackConstants
    
    # Access paths
    print(config.paths.results_path)
    
    # Access data source settings
    if config.data_source.is_remote:
        download_from(config.data_source.remote_url)
    
    # Access SNOWPACK constants
    print(SnowpackConstants.GRAIN_TYPE_CODE[4])
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """Configuration for all file paths used in the application."""
    
    # Root directories
    project_root: Path
    data_path: Path
    results_path: Path
    assets_path: Path
    
    # Data subdirectories
    reference_path: Path
    processed_data_path: Path
    input_path: Path
    
    # Specific files
    input_polygons: Path
    input_polygons_test: Path
    snowpack_locations_csv: Path
    snowpack_viewer_locations_csv: Path
    dem_tif: Path
    aspect_polygons: Path
    linked_polygons: Path
    summary_map_html: Path
    pro_file_manifest: Path
    
    @classmethod
    def from_project_root(cls, project_root: Optional[Path] = None) -> 'PathConfig':
        """
        Initialize paths from the project root directory.
        
        Args:
            project_root: The root directory of the project. If None, auto-detects
                         from the location of this config file.
        """
        if project_root is None:
            # Auto-detect: this file is in src/wetting_front_tracker/
            config_file = Path(__file__).resolve()
            project_root = config_file.parent.parent.parent
        
        project_root = Path(project_root)
        
        # Core directories
        data_path = project_root / 'data'
        reference_path = data_path / 'reference'
        processed_data_path = data_path / 'processed'
        
        # Input path from environment or default
        input_path = Path(os.getenv(
            "PRO_FILES_INPUT_DIR",
            default=str(data_path / "input")
        ))
        
        # Results path from environment or default
        results_path = Path(os.getenv(
            "WFT_RESULTS_OUTPUT_DIR",
            default=str(project_root / 'results')
        ))
        
        # Assets path from environment or default
        assets_subfolder = "plot_assets"
        assets_path = Path(os.getenv(
            "WFT_ASSETS_OUTPUT_DIR",
            default=str(results_path / assets_subfolder)
        ))
        
        return cls(
            project_root=project_root,
            data_path=data_path,
            results_path=results_path,
            assets_path=assets_path,
            reference_path=reference_path,
            processed_data_path=processed_data_path,
            input_path=input_path,
            # Specific files
            input_polygons=reference_path / 'Paths.geojson',
            input_polygons_test=reference_path / 'Paths_test.geojson',
            snowpack_locations_csv=reference_path / 'snowpack_locations_with_metadata.csv',
            snowpack_viewer_locations_csv=reference_path / 'snowpack_viewer_locations.csv',
            dem_tif=processed_data_path / 'dem.tif',
            aspect_polygons=processed_data_path / 'aspect_polygons.geojson',
            linked_polygons=processed_data_path / 'linked_aspect_polygons.geojson',
            summary_map_html=results_path / "summary_map.html",
            pro_file_manifest=processed_data_path / "pro_file_manifest.json",
        )
    
    def ensure_directories_exist(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_path,
            self.results_path,
            self.reference_path,
            self.processed_data_path,
            self.assets_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_png_path(self, file_stem: str) -> Path:
        """Generate the output path for a Matplotlib PNG plot."""
        return self.assets_path / f"{file_stem}_wetting_front.png"
    
    def get_html_path(self, file_stem: str) -> Path:
        """Generate the output path for a Plotly HTML plot."""
        return self.assets_path / f"{file_stem}_wetting_front.html"


# ---------------------------------------------------------------------------
# Data Source Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataSourceConfig:
    """Configuration for data source and fetching behavior."""
    
    source: str  # 'local' or 'remote'
    remote_url: str
    use_test_data: bool
    
    @property
    def is_remote(self) -> bool:
        """Check if data source is configured as remote."""
        return self.source.lower() == 'remote'
    
    @property
    def is_local(self) -> bool:
        """Check if data source is configured as local."""
        return self.source.lower() == 'local'
    
    @classmethod
    def from_env(cls) -> 'DataSourceConfig':
        """Load data source configuration from environment variables."""
        return cls(
            source=os.getenv("PRO_FILES_SOURCE", "local").lower(),
            remote_url=os.getenv(
                "REMOTE_PRO_FILES_URL",
                "https://nwp.mtnweather.info/ron/ssd/snowpack/output/"
            ),
            use_test_data=os.getenv("USE_TEST_DATA", "false").lower() == "true",
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        if self.source not in ('local', 'remote'):
            raise ValueError(
                f"PRO_FILES_SOURCE must be 'local' or 'remote', got '{self.source}'"
            )
        
        if self.is_remote and not self.remote_url:
            raise ValueError(
                "REMOTE_PRO_FILES_URL must be set when PRO_FILES_SOURCE is 'remote'"
            )


# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    
    opentopo_api_key: str
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Load API configuration from environment variables."""
        return cls(
            opentopo_api_key=os.getenv("OPENTOPO_API_KEY", "YOUR_API_KEY_HERE")
        )
    
    def validate(self) -> None:
        """Validate the API configuration."""
        if self.opentopo_api_key == "YOUR_API_KEY_HERE":
            logger.warning(
                "OpenTopography API key not set. DEM downloads will fail. "
                "Set OPENTOPO_API_KEY in your .env file."
            )


# ---------------------------------------------------------------------------
# DEM Configuration
# ---------------------------------------------------------------------------

@dataclass
class DEMDataset:
    """Configuration for a DEM dataset."""
    name: str
    api_endpoint: str
    param_name: str
    bounds: tuple[float, float, float, float]  # (west, south, east, north)


@dataclass
class DEMConfig:
    """Configuration for DEM datasets."""
    
    datasets: List[DEMDataset] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> 'DEMConfig':
        """Create default DEM configuration with standard datasets."""
        return cls(datasets=[
            DEMDataset(
                name="USGS10m",
                api_endpoint="https://portal.opentopography.org/API/usgsdem",
                param_name="demtype",
                bounds=(-124.73, 24.96, -66.95, 49.37),  # Contiguous US
            ),
            DEMDataset(
                name="SRTMGL1",
                api_endpoint="https://portal.opentopography.org/API/globaldem",
                param_name="demtype",
                bounds=(-180, -90, 180, 90),  # Global
            ),
        ])
    
    def get_dataset_for_location(
        self, 
        longitude: float, 
        latitude: float
    ) -> Optional[DEMDataset]:
        """
        Select the best DEM dataset for a given location.
        
        Args:
            longitude: The longitude of the location
            latitude: The latitude of the location
            
        Returns:
            The most appropriate DEMDataset, or None if no dataset covers the location
        """
        for dataset in self.datasets:
            west, south, east, north = dataset.bounds
            if west <= longitude <= east and south <= latitude <= north:
                return dataset
        
        # Fallback to global dataset (should be last in list)
        return self.datasets[-1] if self.datasets else None


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------

@dataclass
class WFTConfig:
    """Main configuration class for Wetting Front Tracker."""
    
    paths: PathConfig
    data_source: DataSourceConfig
    api: APIConfig
    dem: DEMConfig
    
    # Environment detection
    is_dev_environment: bool = field(default_factory=lambda: os.name == 'nt')
    
    @classmethod
    def load(cls) -> 'WFTConfig':
        """
        Load configuration from environment variables and defaults.
        
        Returns:
            A fully initialized WFTConfig instance
        """
        paths = PathConfig.from_project_root()
        data_source = DataSourceConfig.from_env()
        api = APIConfig.from_env()
        dem = DEMConfig.default()
        
        config = cls(
            paths=paths,
            data_source=data_source,
            api=api,
            dem=dem,
        )
        
        # Validate and initialize
        config.validate()
        config.paths.ensure_directories_exist()
        
        return config
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        self.data_source.validate()
        self.api.validate()
    
    def get_input_polygons_path(self) -> Path:
        """Get the appropriate input polygons path based on test mode."""
        if self.data_source.use_test_data:
            return self.paths.input_polygons_test
        return self.paths.input_polygons


# ---------------------------------------------------------------------------
# SNOWPACK Constants (Static Lookup Tables)
# ---------------------------------------------------------------------------

class SnowpackConstants:
    """
    Static constants and lookup tables for SNOWPACK data.
    
    These are class-level constants that don't change and don't need
    to be part of the main configuration.
    """
    
    # Hand hardness conversion
    HAND_HARDNESS_TO_NUMERIC = {
        'F': 1, 'F+': 1.5, '4F-': 1.5, '4F': 2, '4F+': 2.5,
        '1F-': 2.5, '1F': 3, '1F+': 3.5, 'P-': 3.5, 'P': 4,
        'P+': 4.5, 'K-': 4.5, 'K': 5, 'K+': 5.5, 'I': 6
    }
    
    NUMERIC_TO_HAND_HARDNESS = {
        v: k for k, v in HAND_HARDNESS_TO_NUMERIC.items()
    }
    
    # Grain type codes (full names)
    GRAIN_TYPE_CODE = {
        1: 'Precipitation particules (PP)',
        2: 'Decomposing fragmented PP (DF)',
        3: 'Rounded grains (RG)',
        4: 'Faceted crystals (FC)',
        5: 'Depth hoar (DH)',
        6: 'Surface hoar (SH)',
        7: 'Melt forms (MF)',
        8: 'Ice formations (IF)',
        9: 'Rounding faceted particules (FCxr)'
    }
    
    # Grain type codes (short names)
    GRAIN_TYPE_CODE_SHORT = {
        1: 'PP', 2: 'DF', 3: 'RG', 4: 'FC', 5: 'DH',
        6: 'SH', 7: 'MF', 8: 'IF', 9: 'FCxr'
    }
    
    # Grain type name to ID
    GRAIN_TYPE_NAME_TO_ID = {
        '': 0, 'PP': 1, 'DF': 2, 'RG': 3, 'FC': 4, 'DH': 5,
        'SH': 6, 'MF': 7, 'FCxr': 8, 'MFcr': 9
    }
    
    # Colors for plotting (by ID)
    GRAIN_TYPE_COLORS_BY_ID = {
        1: 'lime', 2: 'darkgreen', 3: 'pink', 4: 'lightblue',
        5: 'blue', 6: 'magenta', 7: 'crimson', 8: 'crimson',
        9: 'skyblue'
    }
    
    # Colors for plotting (by name)
    GRAIN_TYPE_COLORS_BY_NAME = {
        'PP': 'lime', 'DF': 'darkgreen', 'RG': 'pink',
        'FC': 'lightblue', 'DH': 'blue', 'SH': 'magenta',
        'MF': 'crimson', 'FCxr': 'crimson', 'MFcr': 'crimson',
        '': 'whitesmoke'
    }
    
    # RGB colors for Plotly
    GRAIN_TYPE_RGB = {
        'PP': 'rgb(0, 255, 0)', 'DF': 'rgb(34, 139, 34)',
        'RG': 'rgb(255, 182, 193)', 'FC': 'rgb(173, 216, 230)',
        'DH': 'rgb(0, 0, 255)', 'SH': 'rgb(255, 0, 255)',
        'MF': 'rgb(255, 0, 0)', 'IF': 'rgb(255, 0, 0)',
        'FCxr': 'rgb(0, 255, 255)', '': 'rgb(200, 200, 200)'
    }
    
    # Grain type similarity table
    @staticmethod
    def get_grain_type_similarity_table() -> pd.DataFrame:
        """
        Get the grain type similarity table as a pandas DataFrame.
        
        Returns:
            DataFrame with grain type similarities
        """
        data = np.array([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.5, 0.8, 1.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.4, 0.0],
            [0.5, 0.5, 0.8, 1.0, 0.4, 0.1, 0.0, 0.0, 0.5, 0.0],
            [0.5, 0.2, 0.4, 0.4, 1.0, 0.5, 0.3, 0.0, 0.6, 0.0],
            [0.5, 0.0, 0.0, 0.1, 0.5, 1.0, 0.9, 0.0, 0.4, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.3, 0.9, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2],
            [0.5, 0.2, 0.4, 0.5, 0.6, 0.4, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0]
        ])
        
        index_names = ['', 'PP', 'DF', 'RG', 'FC', 'DH', 'SH', 'MF', 'FCxr', 'MFcr']
        
        return pd.DataFrame(
            data,
            columns=index_names,
            index=index_names
        )
    
    # Test grading scores
    TEST_GRADING_SCORE = {
        'ECTP': 1,
        'ECTN': 2
    }


# ---------------------------------------------------------------------------
# Singleton Configuration Instance
# ---------------------------------------------------------------------------

# Create the global configuration instance
config = WFTConfig.load()

# Backward compatibility: expose commonly used paths at module level
PATHS = config.paths
DATA_PATH = config.paths.data_path
RESULTS_PATH = config.paths.results_path
ASSETS_PATH = config.paths.assets_path
PRO_FILES_BASE_PATH = config.paths.input_path

# Backward compatibility: specific files
INPUT_POLYGONS_GEOJSON = config.paths.input_polygons
INPUT_POLYGONS_GEOJSON_TEST = config.paths.input_polygons_test
LINKED_POLYGONS_GEOJSON = config.paths.linked_polygons
PRO_FILE_MANIFEST = config.paths.pro_file_manifest
SNOWPACK_LOCATIONS_CSV = config.paths.snowpack_locations_csv

# Backward compatibility: other settings
USE_TEST_DATA = config.data_source.use_test_data
OPENTOPO_API_KEY = config.api.opentopo_api_key


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_png_path(file_stem: str, assets_dir: Optional[Path] = None) -> Path:
    """
    Generate the output path for a Matplotlib PNG plot.
    
    Args:
        file_stem: The base name for the file
        assets_dir: Optional override for the assets directory
        
    Returns:
        Path to the PNG file
    """
    if assets_dir is None:
        assets_dir = config.paths.assets_path
    return Path(assets_dir) / f"{file_stem}_wetting_front.png"


def get_html_path(file_stem: str, assets_dir: Optional[Path] = None) -> Path:
    """
    Generate the output path for a Plotly HTML plot.
    
    Args:
        file_stem: The base name for the file
        assets_dir: Optional override for the assets directory
        
    Returns:
        Path to the HTML file
    """
    if assets_dir is None:
        assets_dir = config.paths.assets_path
    return Path(assets_dir) / f"{file_stem}_wetting_front.html"


# ---------------------------------------------------------------------------
# Module Initialization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test configuration loading
    print("Wetting Front Tracker Configuration")
    print("=" * 50)
    print(f"\nProject Root: {config.paths.project_root}")
    print(f"Data Source: {config.data_source.source}")
    print(f"Is Remote: {config.data_source.is_remote}")
    print(f"Use Test Data: {config.data_source.use_test_data}")
    print(f"Results Path: {config.paths.results_path}")
    print(f"\nAPI Key Set: {config.api.opentopo_api_key != 'YOUR_API_KEY_HERE'}")
    print(f"Available DEM Datasets: {len(config.dem.datasets)}")
    
    # Test DEM dataset selection
    test_location = (-105.5, 39.5)  # Colorado
    dataset = config.dem.get_dataset_for_location(*test_location)
    if dataset:
        print(f"\nDEM for Colorado: {dataset.name}")