import glob
import argparse # <-- Import argparse
from pathlib import Path
import logging
import pandas as pd

# --- CONFIGURATION ---
# The default path is now set in the argument parser
OUTPUT_FILENAME = "simulation_locations_with_metadata.csv"

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_metadata_from_pro(file_path: Path) -> dict | None:
    """
    Reads the header of a .pro file to extract key metadata fields.
    """
    metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(30):
                line = f.readline()
                if not line or line.strip() == "[DATA]":
                    break
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()

            slope_angle = float(metadata.get('SlopeAngle', -1))
            if slope_angle == 0.00:
                metadata['aspect'] = 'Flat'
            else:
                metadata['aspect'] = metadata.get('slopeAzi', 'N/A')
            
            if 'Latitude' in metadata and 'Longitude' in metadata and 'Altitude' in metadata:
                return metadata
            else:
                logging.warning(f"Missing required metadata in {file_path}")
                return None
    except Exception as e:
        logging.error(f"Could not read or parse file {file_path}: {e}")
        return None

def find_simulation_locations(search_path: Path, output_file: Path):
    """
    Finds all .pro files, extracts metadata, and saves the results to a CSV.
    """
    logging.info(f"Searching for .pro files in subdirectories of: {search_path}")
    
    search_pattern = str(search_path / "*/*/*.pro")
    pro_files = [r'C:\Users\Avalanche\Documents\projects\wetting_front_tracker\data\input\103250_res.pro']#glob.glob(search_pattern)

    if not pro_files:
        logging.warning("No .pro files found.")
        return

    logging.info(f"Found {len(pro_files)} total .pro files. Processing all of them...")
    
    all_metadata = []
    for file in pro_files:
        file_path = Path(file)
        
        
        if metadata := extract_metadata_from_pro(file_path):
            all_metadata.append({
                'latitude': metadata['latitude'],
                'longitude': metadata['longitude'],
                'elevation': metadata['altitude'],
                'aspect': metadata['aspect'],
                'path': str(file_path)
            })

    if not all_metadata:
        logging.warning("Could not extract valid metadata from any files.")
        return

    df = pd.DataFrame(all_metadata)
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved {len(df)} records to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write output CSV file: {e}")


if __name__ == '__main__':
    # --- NEW: Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Extract metadata from SNOWPACK .pro files in a directory structure."
    )
    parser.add_argument(
        "-d", "--directory",
        dest="search_directory",
        help="The top-level directory to search for .pro files.",
        default="data/input/"#"/ssd/snowpack/output/2024-newhs/"
    )
    args = parser.parse_args()
    
    # Use the directory from the command-line arguments
    search_dir = Path(args.search_directory)
    output_path = Path(OUTPUT_FILENAME)
    
    find_simulation_locations(search_dir, output_path)