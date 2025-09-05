# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:56:19 2024

@author: Avalanche
"""
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import pandas as pd

HAND_HARD_2_NUMERIC = {'F'  : 1,
                       'F+' : 1.5,
                       '4F-': 1.5,
                       '4F' : 2,
                       '4F+': 2.5,
                       '1F-': 2.5,
                       '1F' : 3,
                       '1F+': 3.5,
                       'P-' : 3.5,
                       'P'  : 4,
                       'P+' : 4.5,
                       'K-' : 4.5,
                       'K'  : 5,
                       'K+' : 5.5,
                       'I'  : 6}

HAND_HARNESS =  {v:k for k,v in HAND_HARD_2_NUMERIC.items()}

# SNOWPILOT conversion:
GRAIN_TYPE_CODE = {1: 'Precipitation particules (PP)',
                   2: 'Decomposing fragmented PP (DF)',
                   3: 'Rounded grains (RG)',
                   4: 'Faceted crystals (FC)',
                   5: 'Depth hoar (DH)',
                   6: 'Surface hoar (SH)',
                   7: 'Melt forms (MF)',
                   8: 'Ice formations (IF)',
                   9: 'Rounding faceted particules (FCxr)'
                   }


GRAIN_TYPE_CODE_S = {1: 'PP',
                     2: 'DF',
                     3: 'RG',
                     4: 'FC',
                     5: 'DH',
                     6: 'SH',
                     7: 'MF',
                     8: 'IF',
                     9: 'FCxr'
                     }

#{k: v[v.find('(')+1:-1] for k, v in GRAIN_TYPE_CODE.items()}

# SnowPilot conversion:
GRAIN_TYPE_COLORS_BY_ID = {1: 'lime',
                           2: 'darkgreen',
                           3: 'pink',
                           4: 'lightblue',
                           5: 'blue',
                           6: 'magenta',
                           7: 'crimson',
                           8: 'crimson',
                           9: 'skyblue'
                           }

GRAIN_TYPE_INDEX_ID = {''    : 0,
                       'PP'  : 1,
                       'DF'  : 2,
                       'RG'  : 3,
                       'FC'  : 4,
                       'DH'  : 5,
                       'SH'  : 6,
                       'MF'  : 7,
                       'FCxr': 8,
                       'MFcr': 9,
                       }

GRAIN_TYPE_COLORS_BY_NAME = {'PP': 'lime',
                             'DF': 'darkgreen',
                             'RG': 'pink',
                             'FC': 'lightblue',
                             'DH': 'blue',
                             'SH': 'magenta',
                             'MF': 'crimson',
                             'FCxr': 'crimson',
                             'MFcr': 'crimson',
                             ''    : 'whitesmoke'
                             }

GRAIN_TYPE_COLORS_BY_NAME = {'PP': 'lime',
                             'DF': 'darkgreen',
                             'RG': 'pink',
                             'FC': 'lightblue',
                             'DH': 'blue',
                             'SH': 'magenta',
                             'MF': 'crimson',
                             'FCxr': 'crimson',
                             'MFcr': 'crimson',
                             ''    : 'whitesmoke'
                             }


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



grain_type_similaty_table = np.array(
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


grain_type_similaty_table = pd.DataFrame(grain_type_similaty_table, 
                                         columns=list(GRAIN_TYPE_INDEX_ID.keys()), 
                                         index=list(GRAIN_TYPE_INDEX_ID.keys()))

test_grading_score = {'ECTP': 1,
                      'ECTN': 2}

# --- NEW: LOAD ENVIRONMENT VARIABLES ---

DEV = True

# This will load the variables from the .env file in your project root
load_dotenv()

# Define base path relative to the current working directory
PROJECT_ROOT = Path.cwd()

# Core Directories
DATA_PATH = (PROJECT_ROOT / 'data').resolve()
INPUT_PATH = DATA_PATH / 'input'
REFERENCE_PATH = DATA_PATH / 'reference'
RESULTS_PATH = (PROJECT_ROOT / 'results').resolve()
PROCESSED_DATA_PATH = DATA_PATH / 'processed' # For DEMs and new GeoJSONs

# Ensure all directories exist, creating them if necessary
DATA_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# --- NEW: Centralized File Paths ---
# Input Files
INPUT_POLYGONS_GEOJSON = REFERENCE_PATH / 'HighwayPaths.geojson'
SNOWPACK_LOCATIONS_CSV = REFERENCE_PATH / 'snowpack_locations_with_metadata.csv'

# Processed / Intermediate Files
DEM_TIF = PROCESSED_DATA_PATH / 'dem.tif'
ASPECT_POLYGONS_GEOJSON = PROCESSED_DATA_PATH / 'aspect_polygons.geojson'

# Final Output Files
LINKED_POLYGONS_GEOJSON = PROCESSED_DATA_PATH / 'linked_aspect_polygons.geojson'
SUMMARY_MAP_HTML = RESULTS_PATH / "summary_map.html"
PRO_FILE_MANIFEST = PROCESSED_DATA_PATH / "pro_file_manifest.txt"

# --- Directory Creation ---
for path in [DATA_PATH, RESULTS_PATH, REFERENCE_PATH, PROCESSED_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)
    
DEM_DATASETS = [
    {
        'name': 'USGS 10m (CONUS)',
        'datasetName': 'USGS10m',
        'api_endpoint': 'https://portal.opentopography.org/API/usgsdem',
        # Bounding box for the Contiguous United States
        'bounds': (-125.0, 24.0, -66.0, 50.0)
    },
    # You could add other regional datasets here, e.g., for Alaska or New Zealand.
    {
        'name': 'SRTM GL1 (Global Fallback)',
        'datasetName': 'SRTMGL1',
        'api_endpoint': 'https://portal.opentopography.org/API/globaldem',
        # Global bounding box
        'bounds': (-180.0, -90.0, 180.0, 90.0)
    }
]

# --- API KEY CONFIGURATION ---
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "YOUR_API_KEY_HERE")


# --- Functions to generate standardized output paths ---
# (These functions remain unchanged)
def get_png_path(station_name: str) -> Path:
    """Generates the output path for the Matplotlib PNG plot."""
    return RESULTS_PATH / f"{station_name}_wetting_front.png"

def get_html_path(station_name: str) -> Path:
    """Generates the output path for the Plotly HTML plot."""
    return RESULTS_PATH / f"{station_name}_wetting_front.html"