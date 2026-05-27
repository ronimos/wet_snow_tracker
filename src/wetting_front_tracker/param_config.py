"""
param_config.py
===============

Configuration management for the Wetting Front Tracker application.
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
    src_root: Path      # src/wetting_front_tracker/
    data_path: Path
    results_path: Path
    plot_assets_path: Path
    
    # New Asset directories
    internal_assets_path: Path # src/wetting_front_tracker/assets/
    models_path: Path          # src/wetting_front_tracker/assets/models/
    
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
        """
        # Auto-detect source root (where this file lives)
        config_file = Path(__file__).resolve()
        src_root = config_file.parent
        
        if project_root is None:
            project_root = src_root.parent.parent
        
        project_root = Path(project_root)
        
        # Core directories
        data_path = project_root / 'data'
        reference_path = data_path / 'reference'
        processed_data_path = data_path / 'processed'
        
        # Internal Assets (Models, Templates, etc.)
        internal_assets_path = src_root / 'assets'
        models_path = internal_assets_path / 'models'
        
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
        
        # Plot Assets path (External/Output)
        plot_assets_subfolder = "plot_assets"
        plot_assets_path = Path(os.getenv(
            "WFT_ASSETS_OUTPUT_DIR",
            default=str(results_path / plot_assets_subfolder)
        ))
        
        return cls(
            project_root=project_root,
            src_root=src_root,
            data_path=data_path,
            results_path=results_path,
            plot_assets_path=plot_assets_path,
            internal_assets_path=internal_assets_path,
            models_path=models_path,
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
            self.plot_assets_path,
            self.internal_assets_path,
            self.models_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # logger.debug(f"Ensured directory exists: {directory}")
    
    def get_png_path(self, file_stem: str) -> Path:
        """Generate the output path for a Matplotlib PNG plot."""
        return self.plot_assets_path / f"{file_stem}_wetting_front.png"
    
    def get_html_path(self, file_stem: str) -> Path:
        """Generate the output path for a Plotly HTML plot."""
        return self.plot_assets_path / f"{file_stem}_wetting_front.html"


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
        return self.source.lower() == 'remote'
    
    @property
    def is_local(self) -> bool:
        return self.source.lower() == 'local'
    
    @classmethod
    def from_env(cls) -> 'DataSourceConfig':
        return cls(
            source=os.getenv("PRO_FILES_SOURCE", "local").lower(),
            remote_url=os.getenv(
                "REMOTE_PRO_FILES_URL",
                "https://nwp.mtnweather.info/ron/ssd/snowpack/output/"
            ),
            use_test_data=os.getenv("USE_TEST_DATA", "false").lower() == "true",
        )
    
    def validate(self) -> None:
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
        return cls(
            opentopo_api_key=os.getenv("OPENTOPO_API_KEY", "YOUR_API_KEY_HERE")
        )
    
    def validate(self) -> None:
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
    name: str
    api_endpoint: str
    param_name: str
    bounds: tuple[float, float, float, float]  # (west, south, east, north)


@dataclass
class DEMConfig:
    datasets: List[DEMDataset] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> 'DEMConfig':
        return cls(datasets=[
            DEMDataset(
                name="USGS10m",
                api_endpoint="https://portal.opentopography.org/API/usgsdem",
                param_name="demtype",
                bounds=(-124.73, 24.96, -66.95, 49.37),
            ),
            DEMDataset(
                name="SRTMGL1",
                api_endpoint="https://portal.opentopography.org/API/globaldem",
                param_name="demtype",
                bounds=(-180, -90, 180, 90),
            ),
        ])
    
    def get_dataset_for_location(
        self, 
        longitude: float, 
        latitude: float
    ) -> Optional[DEMDataset]:
        for dataset in self.datasets:
            west, south, east, north = dataset.bounds
            if west <= longitude <= east and south <= latitude <= north:
                return dataset
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
    
    is_dev_environment: bool = field(default_factory=lambda: os.name == 'nt')
    
    @classmethod
    def load(cls) -> 'WFTConfig':
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
        
        config.validate()
        config.paths.ensure_directories_exist()
        return config
    
    def validate(self) -> None:
        self.data_source.validate()
        self.api.validate()
    
    def get_input_polygons_path(self) -> Path:
        if self.data_source.use_test_data:
            return self.paths.input_polygons_test
        return self.paths.input_polygons


# ---------------------------------------------------------------------------
# SNOWPACK Constants (Static Lookup Tables)
# ---------------------------------------------------------------------------

class SnowpackConstants:
    """Static constants and lookup tables for SNOWPACK data."""
    
    HAND_HARDNESS_TO_NUMERIC = {
        'F': 1, 'F+': 1.5, '4F-': 1.5, '4F': 2, '4F+': 2.5,
        '1F-': 2.5, '1F': 3, '1F+': 3.5, 'P-': 3.5, 'P': 4,
        'P+': 4.5, 'K-': 4.5, 'K': 5, 'K+': 5.5, 'I': 6
    }
    
    NUMERIC_TO_HAND_HARDNESS = {
        v: k for k, v in HAND_HARDNESS_TO_NUMERIC.items()
    }
    
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
    
    GRAIN_TYPE_CODE_SHORT = {
        1: 'PP', 2: 'DF', 3: 'RG', 4: 'FC', 5: 'DH',
        6: 'SH', 7: 'MF', 8: 'IF', 9: 'FCxr'
    }
    
    GRAIN_TYPE_NAME_TO_ID = {
        '': 0, 'PP': 1, 'DF': 2, 'RG': 3, 'FC': 4, 'DH': 5,
        'SH': 6, 'MF': 7, 'FCxr': 8, 'MFcr': 9
    }
    
    GRAIN_TYPE_COLORS_BY_ID = {
        1: 'lime', 2: 'darkgreen', 3: 'pink', 4: 'lightblue',
        5: 'blue', 6: 'magenta', 7: 'crimson', 8: 'crimson',
        9: 'skyblue'
    }
    
    GRAIN_TYPE_COLORS_BY_NAME = {
        'PP': 'lime', 'DF': 'darkgreen', 'RG': 'pink',
        'FC': 'lightblue', 'DH': 'blue', 'SH': 'magenta',
        'MF': 'crimson', 'FCxr': 'crimson', 'MFcr': 'crimson',
        '': 'whitesmoke'
    }
    
    GRAIN_TYPE_RGB = {
        'PP': 'rgb(0, 255, 0)', 'DF': 'rgb(34, 139, 34)',
        'RG': 'rgb(255, 182, 193)', 'FC': 'rgb(173, 216, 230)',
        'DH': 'rgb(0, 0, 255)', 'SH': 'rgb(255, 0, 255)',
        'MF': 'rgb(255, 0, 0)', 'IF': 'rgb(255, 0, 0)',
        'FCxr': 'rgb(0, 255, 255)', '': 'rgb(200, 200, 200)'
    }
    
    @staticmethod
    def get_grain_type_similarity_table() -> pd.DataFrame:
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
        return pd.DataFrame(data, columns=index_names, index=index_names)
    
    TEST_GRADING_SCORE = {
        'ECTP': 1,
        'ECTN': 2
    }


# ---------------------------------------------------------------------------
# SNOWPACK Parameter Definitions (for ML feature extraction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnowpackParameterDef:
    """Definition of a SNOWPACK output parameter."""
    code: str          # SNOWPACK field code (e.g., '0502')
    column_name: str   # Column name in the DataFrame (matches snowpack_reader.PARAM_CODES)
    name: str          # Human-readable name used in ML feature naming
    unit: str = ''
    compute_diff: bool = False   # Include in interface difference features
    compute_ratio: bool = False  # Include in interface ratio features


SNOWPACK_PARAMETERS: Dict[str, SnowpackParameterDef] = {
    '0501': SnowpackParameterDef('0501', 'height', 'height', 'cm'),
    '0502': SnowpackParameterDef('0502', 'density', 'density', 'kg/m³', compute_diff=True, compute_ratio=True),
    '0503': SnowpackParameterDef('0503', 'temperature', 'temperature', '°C', compute_diff=True),
    '0506': SnowpackParameterDef('0506', 'lwc', 'lwc', '%vol', compute_diff=True, compute_ratio=True),
    '0508': SnowpackParameterDef('0508', 'dendricity', 'dendricity', '', compute_diff=True),
    '0509': SnowpackParameterDef('0509', 'sphericity', 'sphericity', '', compute_diff=True),
    '0510': SnowpackParameterDef('0510', 'coord_number', 'coord_number', ''),
    '0511': SnowpackParameterDef('0511', 'bond_size', 'bond_size', 'mm', compute_diff=True, compute_ratio=True),
    '0512': SnowpackParameterDef('0512', 'grain_size', 'grain_size', 'mm', compute_diff=True, compute_ratio=True),
    '0513': SnowpackParameterDef('0513', 'grain_type', 'grain_type', ''),
    '0515': SnowpackParameterDef('0515', 'ice_content', 'ice_content', '%', compute_diff=True),
    '0516': SnowpackParameterDef('0516', 'air_content', 'air_content', '%', compute_diff=True),
    '0517': SnowpackParameterDef('0517', 'stress', 'stress', 'kPa', compute_diff=True, compute_ratio=True),
    '0518': SnowpackParameterDef('0518', 'viscosity', 'viscosity', 'GPa·s'),
    '0520': SnowpackParameterDef('0520', 'temperature_gradient', 'temperature_gradient', 'K/m', compute_diff=True),
    '0523': SnowpackParameterDef('0523', 'viscous_deformation_rate', 'viscous_deformation_rate', '1e-6/s'),
    '0531': SnowpackParameterDef('0531', 'stab_deformation_rate', 'stab_deformation_rate', ''),
    '0532': SnowpackParameterDef('0532', 'sn38', 'sn38', ''),
    '0533': SnowpackParameterDef('0533', 'sk38', 'sk38', ''),
    '0534': SnowpackParameterDef('0534', 'hand_hardness', 'hand_hardness', '', compute_diff=True),
    '0535': SnowpackParameterDef('0535', 'opt_equ_grain_size', 'opt_equ_grain_size', 'mm', compute_diff=True, compute_ratio=True),
    '0601': SnowpackParameterDef('0601', 'shear_strength', 'shear_strength', 'kPa', compute_diff=True, compute_ratio=True),
    '0602': SnowpackParameterDef('0602', 'gs_difference', 'gs_difference', 'mm'),
    '0603': SnowpackParameterDef('0603', 'hardness_difference', 'hardness_difference', ''),
    '0604': SnowpackParameterDef('0604', 'ssi', 'ssi', '', compute_diff=True),
}


def get_parameters_for_differences() -> List[str]:
    """Return SNOWPACK codes for parameters where interface differences are meaningful."""
    return [code for code, p in SNOWPACK_PARAMETERS.items() if p.compute_diff]


def get_parameters_for_ratios() -> List[str]:
    """Return SNOWPACK codes for parameters where interface ratios are meaningful."""
    return [code for code, p in SNOWPACK_PARAMETERS.items() if p.compute_ratio]


def get_column_name(code: str) -> Optional[str]:
    """Map a SNOWPACK field code to its DataFrame column name."""
    param = SNOWPACK_PARAMETERS.get(code)
    return param.column_name if param else None


# ---------------------------------------------------------------------------
# Singleton Configuration Instance
# ---------------------------------------------------------------------------

config = WFTConfig.load()

# Backward compatibility
PATHS = config.paths
DATA_PATH = config.paths.data_path
RESULTS_PATH = config.paths.results_path
ASSETS_PATH = config.paths.plot_assets_path
PRO_FILES_BASE_PATH = config.paths.input_path

INPUT_POLYGONS_GEOJSON = config.paths.input_polygons
INPUT_POLYGONS_GEOJSON_TEST = config.paths.input_polygons_test
LINKED_POLYGONS_GEOJSON = config.paths.linked_polygons
PRO_FILE_MANIFEST = config.paths.pro_file_manifest
SNOWPACK_LOCATIONS_CSV = config.paths.snowpack_locations_csv
USE_TEST_DATA = config.data_source.use_test_data
OPENTOPO_API_KEY = config.api.opentopo_api_key


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_png_path(file_stem: str, assets_dir: Optional[Path] = None) -> Path:
    if assets_dir is None:
        assets_dir = config.paths.plot_assets_path
    return Path(assets_dir) / f"{file_stem}_wetting_front.png"

def get_html_path(file_stem: str, assets_dir: Optional[Path] = None) -> Path:
    if assets_dir is None:
        assets_dir = config.paths.plot_assets_path
    return Path(assets_dir) / f"{file_stem}_wetting_front.html"


# ---------------------------------------------------------------------------
# ML Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class MLModelConfig:
    """Configuration for ML-based LOC detection."""
    enabled: bool = False
    model_path: Optional[Path] = None
    use_ml_primary: bool = True
    probability_threshold: float = 0.5
    lookback_hours: int = 24

# --- ML Path Resolution Strategy ---
# 1. Check Env Variable
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", None)
_ml_model_path = None

if ML_MODEL_PATH:
    _ml_model_path = Path(ML_MODEL_PATH)
else:
    # 2. Check Internal Assets (Production Model)
    # Look for any subfolder in assets/models
    if config.paths.models_path.exists():
        # Just pick the first valid model folder we find
        # In a real scenario, you might look for "latest" or "v1" explicitly
        for item in config.paths.models_path.iterdir():
            if item.is_dir() and (item / "model.joblib").exists():
                _ml_model_path = item
                break

    # 3. Fallback to Results (Experimental Model) if Internal not found
    if _ml_model_path is None:
        experimental_path = RESULTS_PATH / "trained_models"
        if experimental_path.exists():
             # Find latest timestamped folder
            runs = sorted([d for d in experimental_path.iterdir() if d.is_dir()], reverse=True)
            if runs and (runs[0] / "trained_model").exists():
                _ml_model_path = runs[0] / "trained_model"

ML_CONFIG = MLModelConfig(
    enabled=False,
    model_path=_ml_model_path,
    use_ml_primary=True,
    probability_threshold=float(os.getenv("ML_PROBABILITY_THRESHOLD", "0.5")),
    lookback_hours=int(os.getenv("ML_LOOKBACK_HOURS", "24"))
)

if os.getenv("ML_ENABLED", "false").lower() == "true":
    ML_CONFIG.enabled = True
    
# Auto-enable if we found a valid path
if _ml_model_path and _ml_model_path.exists():
    ML_CONFIG.enabled = True

LOC_DETECTION_MODE = os.getenv("LOC_DETECTION_MODE", "hybrid")  # 'rule-based', 'ml-only', 'hybrid'

# ---------------------------------------------------------------------------
# Feature Requirements for ML
# ---------------------------------------------------------------------------
# (Unchanged - keeping definitions for SHAP features)
ML_REQUIRED_FEATURES = [
    'height', 'density', 'stress', 'lwc', 'temperature', 'temperature_gradient', 'grain_size', 'grain_type'
]
ML_OPTIMAL_FEATURES = ML_REQUIRED_FEATURES + [
    'viscosity', 'shear_strength', 'bond_size', 'sphericity', 'optical_grain_size', 
    'grain_size_difference', 'hand_hardness', 'ice_content', 'hardness_difference', 'viscous_deformation_rate'
]

if __name__ == "__main__":
    print("Wetting Front Tracker Configuration")
    print("=" * 50)
    print(f"Models Path: {config.paths.models_path}")
    print(f"Detected Model: {_ml_model_path}")