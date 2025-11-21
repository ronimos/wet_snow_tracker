"""
inspect_map_data.py
==================

Inspects the GeoDataFrame that's passed to create_folium_map to identify
why polygons are showing as gray (no data).

Usage:
    # If you have a saved map_data.geojson
    python inspect_map_data.py --geojson results/map_data.geojson
    
    # If you want to intercept during actual run
    # Add this code to main.py before create_folium_map():
    # import inspect_map_data
    # inspect_map_data.inspect_gdf(final_gdf)
"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


def inspect_gdf(gdf: gpd.GeoDataFrame, output_path: Path = None):
    """Detailed inspection of GeoDataFrame for mapping."""
    
    print(f"\n{'='*80}")
    print("GEODATAFRAME INSPECTION")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\nShape: {gdf.shape}")
    print(f"CRS: {gdf.crs}")
    print(f"\nColumns: {gdf.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['time_to_loc', 'geometry']
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        print(f"\n⚠️  MISSING REQUIRED COLUMNS: {missing}")
    
    # Inspect time_to_loc column
    if 'time_to_loc' in gdf.columns:
        print(f"\n{'─'*80}")
        print("TIME_TO_LOC ANALYSIS")
        print(f"{'─'*80}")
        
        ttl = gdf['time_to_loc']
        
        print(f"\nTotal polygons: {len(ttl)}")
        print(f"Non-null values: {ttl.notna().sum()} ({100*ttl.notna().mean():.1f}%)")
        print(f"Null values: {ttl.isna().sum()} ({100*ttl.isna().mean():.1f}%)")
        
        if ttl.notna().any():
            print(f"\nNon-null statistics:")
            print(f"  Min: {ttl.min():.2f} hours")
            print(f"  Max: {ttl.max():.2f} hours")
            print(f"  Mean: {ttl.mean():.2f} hours")
            print(f"  Median: {ttl.median():.2f} hours")
            
            # Categorize by risk level
            imminent = ((ttl >= 0) & (ttl < 24)).sum()
            near = ((ttl >= 24) & (ttl < 48)).sum()
            moderate = ((ttl >= 48) & (ttl < 72)).sum()
            recent = ((ttl >= -24) & (ttl < 0)).sum()
            past_near = ((ttl >= -48) & (ttl < -24)).sum()
            past_far = ((ttl >= -72) & (ttl < -48)).sum()
            other = ((ttl < -72) | (ttl >= 72)).sum()
            
            print(f"\nRisk categories:")
            print(f"  Imminent (0-24h):    {imminent:4d} ({100*imminent/len(gdf):.1f}%) - DARK RED")
            print(f"  Near (24-48h):       {near:4d} ({100*near/len(gdf):.1f}%) - ORANGE")
            print(f"  Moderate (48-72h):   {moderate:4d} ({100*moderate/len(gdf):.1f}%) - YELLOW")
            print(f"  Recent (-24-0h):     {recent:4d} ({100*recent/len(gdf):.1f}%) - RED")
            print(f"  Past Near (-48--24h):{past_near:4d} ({100*past_near/len(gdf):.1f}%) - LIGHT BLUE")
            print(f"  Past Far (-72--48h): {past_far:4d} ({100*past_far/len(gdf):.1f}%) - DARK BLUE")
            print(f"  Other:               {other:4d} ({100*other/len(gdf):.1f}%)")
            print(f"  No data (NaN):       {ttl.isna().sum():4d} ({100*ttl.isna().mean():.1f}%) - GRAY")
        else:
            print("\n⚠️  ALL VALUES ARE NULL!")
            print("This explains why all polygons are gray.")
    
    # Check other relevant columns
    relevant_cols = ['station_name', 'file_stem', 'aspect', 'pathName']
    present_cols = [col for col in relevant_cols if col in gdf.columns]
    
    if present_cols:
        print(f"\n{'─'*80}")
        print("OTHER COLUMNS")
        print(f"{'─'*80}")
        
        for col in present_cols:
            null_count = gdf[col].isna().sum()
            print(f"\n{col}:")
            print(f"  Non-null: {len(gdf) - null_count}")
            print(f"  Null: {null_count}")
            if null_count < len(gdf):
                print(f"  Sample values: {gdf[col].dropna().head(3).tolist()}")
    
    # Show sample rows
    print(f"\n{'─'*80}")
    print("SAMPLE ROWS")
    print(f"{'─'*80}")
    
    display_cols = ['station_name', 'file_stem', 'time_to_loc', 'aspect']
    display_cols = [col for col in display_cols if col in gdf.columns]
    
    if display_cols:
        print("\nFirst 10 rows:")
        print(gdf[display_cols].head(10).to_string())
        
        # Show rows with non-null time_to_loc
        if 'time_to_loc' in gdf.columns and gdf['time_to_loc'].notna().any():
            print("\nRows with valid time_to_loc (first 5):")
            print(gdf[gdf['time_to_loc'].notna()][display_cols].head(5).to_string())
        
        # Show rows with null time_to_loc
        if 'time_to_loc' in gdf.columns and gdf['time_to_loc'].isna().any():
            print("\nRows with NULL time_to_loc (first 5):")
            print(gdf[gdf['time_to_loc'].isna()][display_cols].head(5).to_string())
    
    # Diagnose the issue
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")
    
    if 'time_to_loc' not in gdf.columns:
        print("\n❌ PROBLEM: 'time_to_loc' column is missing!")
        print("   This should be added by process_single_profile() in main.py")
        print("   Check that process_single_profile() is returning a dict with 'time_to_loc' key")
        
    elif gdf['time_to_loc'].isna().all():
        print("\n❌ PROBLEM: All time_to_loc values are NaN!")
        print("\nPossible causes:")
        print("  1. LOC detection is failing for all profiles")
        print("     → Run debug_loc_detection.py on a sample .pro file")
        print("  2. find_time_to_loc() is returning NaN")
        print("     → Check that weak_layer_height and wet_front_lwc_height are being calculated")
        print("  3. No wetting events detected")
        print("     → Check if LWC data is present in profiles")
        print("  4. Reference date is outside the data range")
        print("     → Check central_date parameter")
        
    elif gdf['time_to_loc'].isna().mean() > 0.5:
        pct_null = 100 * gdf['time_to_loc'].isna().mean()
        print(f"\n⚠️  WARNING: {pct_null:.1f}% of polygons have no data")
        print("   This is causing most polygons to be gray")
        print("\nCheck the failed polygons:")
        if 'file_stem' in gdf.columns:
            failed = gdf[gdf['time_to_loc'].isna()]['file_stem'].tolist()
            print(f"   Failed file_stems (first 10): {failed[:10]}")
        
    else:
        print("\n✅ Data looks reasonable!")
        pct_valid = 100 * gdf['time_to_loc'].notna().mean()
        print(f"   {pct_valid:.1f}% of polygons have valid time_to_loc values")
    
    # Save detailed report if requested
    if output_path:
        report_path = output_path / "gdf_inspection_report.txt"
        with open(report_path, 'w') as f:
            # Write summary statistics
            f.write("GEODATAFRAME INSPECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total polygons: {len(gdf)}\n")
            if 'time_to_loc' in gdf.columns:
                f.write(f"Valid time_to_loc: {gdf['time_to_loc'].notna().sum()}\n")
                f.write(f"Null time_to_loc: {gdf['time_to_loc'].isna().sum()}\n\n")
                
                # Write full list of null cases
                if gdf['time_to_loc'].isna().any():
                    f.write("Polygons with no data:\n")
                    f.write("-" * 80 + "\n")
                    null_cols = ['station_name', 'file_stem', 'aspect'] 
                    null_cols = [c for c in null_cols if c in gdf.columns]
                    null_df = gdf[gdf['time_to_loc'].isna()][null_cols]
                    f.write(null_df.to_string())
        
        print(f"\nDetailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Inspect GeoDataFrame for map visualization')
    parser.add_argument('--geojson', type=Path, help='Path to map_data.geojson')
    parser.add_argument('--output', type=Path, help='Directory to save inspection report')
    
    args = parser.parse_args()
    
    if args.geojson:
        if not args.geojson.exists():
            print(f"Error: File not found: {args.geojson}")
            return
        
        print(f"Loading GeoDataFrame from {args.geojson}...")
        gdf = gpd.read_file(args.geojson)
        inspect_gdf(gdf, args.output)
    else:
        print("No GeoJSON file provided. Use --geojson to specify file.")
        print("\nAlternatively, add this to your main.py before create_folium_map():")
        print("    from inspect_map_data import inspect_gdf")
        print("    inspect_gdf(final_gdf, config.paths.results_path)")


if __name__ == "__main__":
    main()
