import geopandas as gpd
from pathlib import Path

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = PROJECT_ROOT / 'data' / 'reference'
INPUT_FILE = REFERENCE_PATH / 'HighwayPaths.geojson'
OUTPUT_FILE = REFERENCE_PATH / 'HighwayPaths_test.geojson'

def create_test_suite_data(num_polygons: int = 10):
    """
    Reads the main polygon GeoJSON file, takes a small subset of polygons,
    and saves them to a new file for testing purposes.
    """
    print("Creating a smaller test dataset for faster debugging...")

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        print("Please ensure the original 'HighwayPaths.geojson' exists.")
        return

    try:
        # Read the full GeoJSON file
        gdf = gpd.read_file(INPUT_FILE)

        if len(gdf) < num_polygons:
            print(f"Warning: The source file has fewer than {num_polygons} polygons. Using all {len(gdf)}.")
            test_gdf = gdf
        else:
            # Take the first N polygons
            test_gdf = gdf.head(num_polygons)

        # Save the subset to a new file
        test_gdf.to_file(OUTPUT_FILE, driver='GeoJSON')

        print(f"Successfully created test file with {len(test_gdf)} polygons.")
        print(f"Test file saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_test_suite_data()