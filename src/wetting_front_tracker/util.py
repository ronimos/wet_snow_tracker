import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

def get_aspect_from_azi(azi: float):
    """Converts slope azimuth to a cardinal direction."""
    if azi == 0.00:
        return 'N'
    elif azi == 90.00:
        return 'E'
    elif azi == 180.00:
        return 'S'
    elif azi == 270.00:
        return 'W'
    else:
        # Handle cases that are not perfectly cardinal
        return f"Azi_{azi}"

def parse_pro_file(file_path: Path):
    """
    Parses a single .pro file to extract metadata and generates a single
    entry with the correct aspect based on SlopeAngle and SlopeAzi.

    Args:
        file_path: The Path object pointing to the .pro file.

    Returns:
        A list containing a single dictionary for the final DataFrame row,
        or an empty list if parsing fails.
    """
    metadata = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read the header section of the file, assuming it's within the first 30 lines
            for _ in range(30):
                line = f.readline()
                if not line or line.strip() == "[DATA]":
                    break
                
                # Use regex to find key-value pairs like "key = value"
                match = re.match(r"^\s*(\w+)\s*=\s*(.*)", line)
                if match:
                    key, value = match.groups()
                    # Store keys in lowercase to handle case-insensitivity
                    metadata[key.strip().lower()] = value.strip()

        # Check if essential keys were found (using lowercase)
        required_keys = ['latitude', 'longitude', 'altitude', 'stationname', 'slopeangle', 'slopeazi']
        if not all(key in metadata for key in required_keys):
            print(f"Warning: Skipping {file_path.name}. Missing one of the required keys: {required_keys}")
            return []

        lat = float(metadata['latitude'])
        lon = float(metadata['longitude'])
        alt = int(metadata['altitude'])
        station_name = metadata['stationname']
        slope_angle = float(metadata['slopeangle'])
        slope_azi = float(metadata['slopeazi'])

        # Determine aspect based on the new logic
        if slope_angle == 0.00:
            aspect = 'flat'
        else:
            aspect = get_aspect_from_azi(slope_azi)

        # Create a single record for this file
        record = {
            'latitude': lat,
            'longitude': lon,
            'aspect': aspect,
            'altitude': alt,
            'stationName': station_name, # Use the name directly from the file
            'path': file_path.name
        }
        
        return [record] # Return as a list containing the single dictionary
            
    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return []

def main():
    """
    Main function to find .pro files, process them in parallel,
    and save the results to a CSV file.
    """
    try:
        # 1. Define input and output directories
        pro_dir = Path(__file__).resolve().parent.parent.parent / "data" / "input"
        output_dir = pro_dir.parent / "reference"
        output_path = output_dir / "snowpack_locations_with_metadata.csv"

        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Find all .pro files in the directory
        pro_files = list(pro_dir.glob("*.pro"))
        if not pro_files:
            print(f"No .pro files found in {pro_dir}. Exiting.")
            return

        print(f"Found {len(pro_files)} .pro files to process.")

        # 3. Process files in parallel with a progress bar
        all_records = []
        num_processes = min(cpu_count(), len(pro_files))
        
        print(f"Starting processing with {num_processes} worker(s)...")
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered to get results as they complete, which works well with tqdm
            results_iterator = pool.imap_unordered(parse_pro_file, pro_files)
            
            # Wrap the iterator with tqdm for the progress bar
            for records in tqdm(results_iterator, total=len(pro_files), desc="Processing files"):
                if records:
                    all_records.extend(records)

        if not all_records:
            print("No data was successfully parsed from the files. Output file will not be created.")
            return
            
        # 4. Create DataFrame and save to CSV
        df = pd.DataFrame(all_records)
        
        # Reorder columns to the desired specification
        column_order = ['latitude', 'longitude', 'aspect', 'altitude', 'stationName', 'path']
        df = df[column_order]
        
        df.to_csv(output_path, index=False)

        print(f"\nProcessing complete.")
        print(f"DataFrame successfully created with {len(df)} rows.")
        print(f"Data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The input directory does not exist at '{pro_dir}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()

