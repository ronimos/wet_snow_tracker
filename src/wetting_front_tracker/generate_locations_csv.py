"""
generate_locations_csv.py

Scans a directory tree for SNOWPACK .pro files and generates a locations CSV
by parsing station metadata from file headers. No need to load full profiles.

Usage:
    python generate_locations_csv.py /ssd/snowpack/fcst/2025 -o snowpack_locations.csv
    python generate_locations_csv.py /ssd/snowpack/fcst/2025 --glob "zone*/*/*_res.pro"
"""
import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_pro_header(filepath: Path) -> Optional[dict]:
    """
    Reads a .pro file header and extracts station metadata.

    Reads lines until [HEADER] is encountered, parsing key=value pairs
    from the [STATION_PARAMETERS] block.

    Returns dict with keys: latitude, longitude, elevation, aspect, path
    or None if the header can't be parsed.
    """
    metadata = {}
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line == '[HEADER]':
                    break
                if '=' in line:
                    key, _, val = line.partition('=')
                    metadata[key.strip()] = val.strip()
    except Exception as e:
        logging.warning(f"Failed to read {filepath}: {e}")
        return None

    required = ('Latitude', 'Longitude', 'Altitude', 'SlopeAngle', 'SlopeAzi')
    if not all(k in metadata for k in required):
        logging.warning(f"Missing header fields in {filepath}: "
                        f"found {list(metadata.keys())}")
        return None

    try:
        slope_angle = float(metadata['SlopeAngle'])
        slope_azi = float(metadata['SlopeAzi'])
    except ValueError:
        logging.warning(f"Non-numeric slope values in {filepath}")
        return None

    # Flat = both SlopeAngle and SlopeAzi are 0
    if slope_angle == 0.0 and slope_azi == 0.0:
        aspect = 'Flat'
    else:
        aspect = f"{slope_azi:.2f}"

    return {
        'latitude': metadata['Latitude'],
        'longitude': metadata['Longitude'],
        'elevation': float(metadata['Altitude']),
        'aspect': aspect,
        'path': str(filepath),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SNOWPACK locations CSV from .pro file headers."
    )
    parser.add_argument('base_dir', type=Path,
                        help="Root directory to scan for .pro files.")
    parser.add_argument('-o', '--output', type=Path,
                        default=Path(f'{DATA_DIR}/reference/snowpack_locations_with_metadata.csv'),
                        help="Output CSV path (default: snowpack_locations.csv)")
    parser.add_argument('--glob', type=str, default='**/*_res.pro',
                        help="Glob pattern relative to base_dir "
                             "(default: '**/*_res.pro')")
    args = parser.parse_args()

    if not args.base_dir.is_dir():
        logging.error(f"Directory not found: {args.base_dir}")
        return

    pro_files = sorted(args.base_dir.glob(args.glob))
    logging.info(f"Found {len(pro_files)} .pro files matching '{args.glob}' "
                 f"under {args.base_dir}")

    if not pro_files:
        return

    rows = []
    failed = 0
    for fp in pro_files:
        result = parse_pro_header(fp)
        if result:
            rows.append(result)
        else:
            failed += 1

    logging.info(f"Parsed {len(rows)} files successfully, {failed} failed.")

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['latitude', 'longitude',
                                               'elevation', 'aspect', 'path'])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()

