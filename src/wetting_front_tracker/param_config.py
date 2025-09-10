from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# --- ENVIRONMENT CONFIGURATION ---
# Detect if running on Windows (development) or another OS (production)
IS_DEV_ENVIRONMENT = os.name == 'nt'

# --- TESTING SUITE FLAG ---
# Set this to True to use a small subset of polygons for faster debugging.
# Set it to False to run the full analysis.
USE_TEST_DATA = True 

# --- PATH DEFINITIONS ---
# Use a more robust method to define paths relative to this config file
CONFIG_FILE_PATH = Path(__file__).resolve()
SRC_PATH = CONFIG_FILE_PATH.parent.parent
PROJECT_ROOT = SRC_PATH.parent

# Core Directories
DATA_PATH = PROJECT_ROOT / 'data'
RESULTS_PATH = PROJECT_ROOT / 'results'
REFERENCE_PATH = DATA_PATH / 'reference'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'

# --- .pro File Base Paths ---
# Store the production path as a raw string to prevent OS-specific interpretation
PRO_FILES_BASE_PATH_PROD = "/ssd/snowpack/output/2024-newhs/"
PRO_FILES_BASE_PATH_DEV = DATA_PATH / "input" 


# Centralized File Paths
INPUT_POLYGONS_GEOJSON = REFERENCE_PATH / 'Paths.geojson'
INPUT_POLYGONS_GEOJSON_TEST = REFERENCE_PATH / 'Paths_test.geojson' # New test file path
SNOWPACK_LOCATIONS_CSV = REFERENCE_PATH / 'snowpack_locations_with_metadata.csv'
SNOWPACK_VIEWER_LOCATIONS_CSV = REFERENCE_PATH / 'snowpack_viewer_locations.csv'
DEM_TIF = PROCESSED_DATA_PATH / 'dem.tif'
ASPECT_POLYGONS_GEOJSON = PROCESSED_DATA_PATH / 'aspect_polygons.geojson'
LINKED_POLYGONS_GEOJSON = PROCESSED_DATA_PATH / 'linked_aspect_polygons.geojson'
SUMMARY_MAP_HTML = RESULTS_PATH / "summary_map.html"
PRO_FILE_MANIFEST = PROCESSED_DATA_PATH / "pro_file_manifest.txt"

# Directory Creation
for path in [DATA_PATH, RESULTS_PATH, REFERENCE_PATH, PROCESSED_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# --- API KEY CONFIGURATION ---
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "YOUR_API_KEY_HERE")

# --- DEM DATASET SELECTION ---
DEM_DATASETS = [
    {
        "name": "USGS10m",
        "url": "https://portal.opentopography.org/API/usgsdem",
        "bbox": [-124.73, 24.96, -66.95, 49.37],  # Contiguous US
    },
    {
        "name": "SRTMGL1",
        "url": "https://portal.opentopography.org/API/globaldem",
        "bbox": [-180, -90, 180, 90], # Global
    }
]


# --- Functions to generate standardized output paths ---
def get_png_path(file_stem: str) -> Path:
    """Generates the output path for the Matplotlib PNG plot."""
    return RESULTS_PATH / f"{file_stem}_wetting_front.png"

def get_html_path(file_stem: str) -> Path:
    """Generates the output path for the Plotly HTML plot."""
    return RESULTS_PATH / f"{file_stem}_wetting_front.html"


# --- SNOWPACK PARAMETERS ---
HAND_HARD_2_NUMERIC = {'F': 1, 'F+': 1.5, '4F-': 1.5, '4F': 2, '4F+': 2.5,
                       '1F-': 2.5, '1F': 3, '1F+': 3.5, 'P-': 3.5, 'P': 4,
                       'P+': 4.5, 'K-': 4.5, 'K': 5, 'K+': 5.5, 'I': 6}

HAND_HARNESS = {v: k for k, v in HAND_HARD_2_NUMERIC.items()}

GRAIN_TYPE_CODE = {1: 'Precipitation particules (PP)',
                   2: 'Decomposing fragmented PP (DF)',
                   3: 'Rounded grains (RG)',
                   4: 'Faceted crystals (FC)',
                   5: 'Depth hoar (DH)',
                   6: 'Surface hoar (SH)',
                   7: 'Melt forms (MF)',
                   8: 'Ice formations (IF)',
                   9: 'Rounding faceted particules (FCxr)'}

# --- RESTORED PARAMETERS ---
GRAIN_TYPE_CODE_S = {1: 'PP',
                     2: 'DF',
                     3: 'RG',
                     4: 'FC',
                     5: 'DH',
                     6: 'SH',
                     7: 'MF',
                     8: 'IF',
                     9: 'FCxr'}

GRAIN_TYPE_COLORS_BY_ID = {1: 'lime',
                           2: 'darkgreen',
                           3: 'pink',
                           4: 'lightblue',
                           5: 'blue',
                           6: 'magenta',
                           7: 'crimson',
                           8: 'crimson',
                           9: 'skyblue'}

GRAIN_TYPE_INDEX_ID = {''    : 0,
                       'PP'  : 1,
                       'DF'  : 2,
                       'RG'  : 3,
                       'FC'  : 4,
                       'DH'  : 5,
                       'SH'  : 6,
                       'MF'  : 7,
                       'FCxr': 8,
                       'MFcr': 9}

GRAIN_TYPE_COLORS_BY_NAME = {'PP': 'lime',
                             'DF': 'darkgreen',
                             'RG': 'pink',
                             'FC': 'lightblue',
                             'DH': 'blue',
                             'SH': 'magenta',
                             'MF': 'crimson',
                             'FCxr': 'crimson',
                             'MFcr': 'crimson',
                             ''    : 'whitesmoke'}

GRAIN_TYPE_NAME_TO_COLOR = {'PP'  : 'rbg(0, 255, 0)',
                            'DF'  : 'rbg(34, 139, 34)',
                            'RG'  : 'rbg(255, 182, 193)',
                            'FC'  : 'rbg(173, 216, 230)',
                            'DH'  : 'rbg(0, 0, 255)',
                            'SH'  : 'rbg(255, 0, 255)',
                            'MF'  : 'rbg(255, 0, 0)',
                            'IF'  : 'rbg(255, 0, 0)',
                            'FCxr': 'rbg(0, 255, 255)',
                            ''    : 'rgb(200,200,200)'}

grain_type_similaty_table_data = np.array(
    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
     [0.5, 1. , 0.8, 0.5, 0.2, 0. , 0. , 0. , 0.2, 0. ],
     [0.5, 0.8, 1. , 0.8, 0.4, 0. , 0. , 0. , 0.4, 0. ],
     [0.5, 0.5, 0.8, 1. , 0.4, 0.1, 0. , 0. , 0.5, 0. ],
     [0.5, 0.2, 0.4, 0.4, 1. , 0.5, 0.3, 0. , 0.6, 0. ],
     [0.5, 0. , 0. , 0.1, 0.5, 1. , 0.9, 0. , 0.4, 0. ],
     [0.5, 0. , 0. , 0. , 0.3, 0.9, 1. , 0. , 0. , 0. ],
     [0.5, 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0.2],
     [0.5, 0.2, 0.4, 0.5, 0.6, 0.4, 0. , 0. , 1. , 0. ],
     [0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 1. ]])

grain_type_similaty_table = pd.DataFrame(grain_type_similaty_table_data, 
                                         columns=list(GRAIN_TYPE_INDEX_ID.keys()), 
                                         index=list(GRAIN_TYPE_INDEX_ID.keys()))

test_grading_score = {'ECTP': 1,
                      'ECTN': 2}
