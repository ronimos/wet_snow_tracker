# Quick Start Guide - Wetting Front Tracker

Get up and running with the Wetting Front Tracker in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] GDAL libraries installed on your system
- [ ] SNOWPACK .pro files ready
- [ ] 10+ GB disk space (for DEMs and results)

## Installation Steps

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-dev
```

**macOS (with Homebrew):**
```bash
brew install gdal
```

**Windows:**
- Download and install OSGeo4W from https://trac.osgeo.org/osgeo4w/
- Add GDAL to PATH

### 2. Set Up Python Environment

```bash
# Clone or navigate to project directory
cd wetting_front_tracker

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Data

```bash
# Create directory structure
mkdir -p data/input
mkdir -p data/reference
mkdir -p data/results

# Copy your SNOWPACK .pro files to data/input/
cp /path/to/your/*.pro data/input/
```

### 4. Verify .pro File Format

Each .pro file should have a header like:
```
[HEADER]
StationName=MyStation
Latitude=40.123
Longitude=-105.456
Altitude=3500
SlopeAngle=35.0
SlopeAzi=180.0

[DATA]
0500,0501,0502,0503,0506,0507,0512,...
[timestamp],[height],[density],[temp],[lwc],...
```

## First Run

### Default Analysis (Today's Date)

```bash
python -m src.wetting_front_tracker.main
```

This will:
1. ✅ Parse all .pro files in `data/input/`
2. ✅ Create metadata CSV
3. ✅ Generate/download geospatial data
4. ✅ Analyze snowpack profiles in parallel
5. ✅ Generate plots and maps
6. ✅ Save results to `data/results/`

**Expected output:**
```
INFO: Found 150 .pro files to process
INFO: Scanning for .pro files...
INFO: Processing profiles: 100%|████████| 150/150
INFO: Successfully processed 148/150 profiles
INFO: Summary map saved to: data/results/summary_map.html
```

### Custom Date Analysis

```bash
python -m src.wetting_front_tracker.main --date 2025-05-15
```

### View Results

Open the map in your browser:
```bash
# Linux/macOS
open data/results/summary_map.html

# Windows
start data/results/summary_map.html
```

## Understanding Your First Results

### The Summary Map

**What you'll see:**
- Colored polygons representing avalanche paths
- Each color indicates risk level

**Interact with the map:**
1. **Hover** over a polygon → See thumbnail of analysis plot
2. **Click** a polygon → Open popup with link to detailed plot
3. **Change basemap** → Use layer control (top-right)
4. **Zoom/pan** → Your view is saved automatically

### Color Interpretation

| Color | Meaning | Action |
|-------|---------|--------|
| 🔴 Red | Water content >3% above LOC | **High Risk** - Monitor closely |
| 🟡 Yellow | Water content 1-3% OR reaches LOC in 48-72h | **Elevated Risk** - Watch |
| 🟧 Orange | LOC reached in 24-48h | **Moderate Risk** |
| 🟥 Dark Red | LOC reached in 0-24h | **Imminent Risk** |
| 🔵 Blue shades | LOC already reached (past) | **Recent Activity** |
| ⚪ Gray | No data or no concern | **Low Risk** |

### Detailed Plots

Click "Open Interactive Plot" in any popup to see:
- **Time series** of snow depth, weak layer position, wetting front
- **Heatmap** of water content through the snowpack
- **Interactive** zoom, pan, and hover for exact values

## Common First-Time Issues

### Issue: "No .pro files found"

**Solution:**
```bash
# Check your files are in the right place
ls data/input/*.pro

# If empty, copy files there
cp /source/path/*.pro data/input/
```

### Issue: "GDAL not found"

**Solution:**
- Make sure GDAL is installed (see step 1)
- Try: `gdalinfo --version`
- If still failing, reinstall with conda: `conda install -c conda-forge gdal`

### Issue: "Memory Error"

**Solution:**
```bash
# Use fewer parallel workers
python -m src.wetting_front_tracker.main --workers 4

# Or process fewer files at once
# Move some .pro files to a different directory temporarily
```

### Issue: "All polygons are gray"

**Possible causes:**
1. Date range doesn't match available data
2. No weak layers detected in profiles
3. No liquid water in snowpack (too early in season)

**Solution:**
```bash
# Try a different date during spring melt
python -m src.wetting_front_tracker.main --date 2025-05-01

# Check your .pro files have data in that time range
```

## Next Steps

### 1. Customize Configuration

Edit `src/wetting_front_tracker/param_config.py`:

```python
# Change analysis window
ANALYSIS_DAYS_BEFORE = 14  # Look back 2 weeks
ANALYSIS_DAYS_AFTER = 7    # Look ahead 1 week

# Adjust thresholds
LWC_THRESHOLD_PERCENT = 3.0  # More sensitive wetting front
MIN_GS_DIFFERENCE = 0.3      # More sensitive weak layer
```

### 2. Process Multiple Dates

Create a bash script:
```bash
#!/bin/bash
for date in 2025-05-{01..31}; do
    python -m src.wetting_front_tracker.main --date $date
    mv data/results/summary_map.html data/results/map_$date.html
done
```

### 3. Use Your Own Polygons

Instead of auto-generated polygons, provide your avalanche path shapefile:
```bash
cp my_avalanche_paths.shp data/reference/avalanche_paths.shp
# Also copy .shx, .dbf, .prj files

# Run with regenerate flag
python -m src.wetting_front_tracker.main --regenerate-data
```

### 4. Integrate with Workflow

```python
# In your Python script
from src.wetting_front_tracker.main import main as run_tracker

# Run programmatically
run_tracker(
    date='2025-05-15',
    start_date='2025-05-01',
    end_date='2025-05-31'
)
```

## Performance Tips

### For Faster Processing

```bash
# Use all CPU cores
python -m src.wetting_front_tracker.main --workers $(nproc)

# Or install GPU support (if you have NVIDIA GPU)
pip install cupy-cuda12x
# Will automatically use GPU when available
```

### For Lower Resource Usage

```python
# In param_config.py
MATPLOTLIB_DPI = 150          # Lower resolution plots
THUMBNAIL_MAX_SIZE = (600, 400)  # Smaller thumbnails
```

## Getting Help

### Check Logs

```bash
# View detailed logs
cat wetting_front_tracker.log

# Watch in real-time
tail -f wetting_front_tracker.log
```

### Enable Debug Mode

```python
# In main.py, change logging level
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO
    format="%(asctime)s [%(levelname)s] %(message)s"
)
```

### Test Individual Components

```python
# Test file parsing
python -m src.wetting_front_tracker.util

# Test geospatial processing (after setting up)
from src.wetting_front_tracker.prepare_geodata import prepare_aspect_polygons
```

## Example Workflow

A typical daily operational workflow:

```bash
#!/bin/bash
# daily_analysis.sh

# Activate environment
source venv/bin/activate

# Run analysis for today
python -m src.wetting_front_tracker.main

# Copy results to web server
scp data/results/summary_map.html user@server:/var/www/html/avalanche/

# Create backup
tar -czf results_$(date +%Y%m%d).tar.gz data/results/

# Send notification
echo "Analysis complete for $(date)" | mail -s "Wetting Front Update" team@example.com
```

## Validation Checklist

Before trusting results, verify:

- [ ] .pro files cover the correct date range
- [ ] Metadata parsed correctly (check `snowpack_locations_with_metadata.csv`)
- [ ] Plots show reasonable snow depths
- [ ] Weak layers identified where expected
- [ ] Colors on map match plot conditions
- [ ] At least 90% of polygons processed successfully

## Success Indicators

You're ready to use the tool operationally when:

✅ **Processing completes** without errors  
✅ **Map displays** with colored polygons  
✅ **Plots show** realistic snowpack structure  
✅ **Colors make sense** relative to weather/conditions  
✅ **Performance is acceptable** for your workflow  
✅ **Results are reproducible** on repeated runs  

## Resources

- **Full Documentation**: See PROJECT_README.md
- **Code Changes**: See CHANGES_SUMMARY.md for recent updates
- **Examples**: See COLORING_EXAMPLES.md
- **Code Reference**: See CODE_REFERENCE.md

---

**Need more help?** Open an issue or contact the development team.

**Ready to dive deeper?** Check out PROJECT_README.md for comprehensive documentation.
