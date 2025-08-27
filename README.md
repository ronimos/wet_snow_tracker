# Wet Snow Tracker 💧❄️
A hardware-adaptive analysis tool for tracking wet snow slab avalanche conditions using SNOWPACK `.pro` files. This project provides a complete, end-to-end workflow from high-speed data parsing to daily stability analysis and comprehensive visualization.

At its core, the project is built on a powerful `SnowpackProfile` class that reads `.pro` files into `xarray` datasets, automatically leveraging GPU acceleration with `cupy` if available. Building on this, a library of specialized functions analyzes critical factors for wet slab instability, such as the location of weak layers and the penetration depth of liquid water.

The main script ties everything together, producing a daily time-series summary and a clear visual plot that tracks the evolution of the snowpack over a given period.

## The Problem: Wet Snow Avalanches
Wet slab avalanches occur when liquid water from melt or rain percolates into the snowpack. This water can pool on top of less permeable layers (like crusts or existing weak layers), dramatically reducing the shear strength of the snow and creating dangerous instability. This tool is designed to identify and track the key ingredients for this type of avalanche problem: a persistent weak layer, the presence of liquid water, and the interaction between them.

## Key Features
- **Hardware Acceleration 🚀:** Automatically detects and uses an NVIDIA GPU for intensive numerical calculations, falling back seamlessly to the CPU if a GPU is not available.

- **High-Speed Caching:** The first time a .pro file is read, it's parsed and saved as a NetCDF (.nc) file. Subsequent loads read the cached file, reducing data loading times from minutes to milliseconds.

- **Specialized Analysis Library:** Includes a suite of functions specifically designed to identify features relevant to wet snow stability:

  - Locating the most prominent faceted (FC/DH) weak layers, with a focus on the more critical bottom half of the snowpack.

  - Tracking the wetting front penetration based on both grain morphology and a quantitative liquid water content (LWC) threshold.

  - Checking for dangerous conditions where LWC is high in the layer immediately above a detected weak layer.

- **End-to-End Workflow:** The main script provides a complete, runnable example that loads data, computes a daily summary of the metrics above, prints a table, and generates a final plot.

- **Comprehensive Visualization:** The output plot provides an intuitive, at-a-glance summary of the snowpack's evolution, showing total snow depth (HS), weak layer location (LOC), and the wet front, with a shaded region to clearly indicate the wet portion of the snowpack.

## Installation
This project is packaged and can be installed using uv or pip.

1. Clone the repository:

```Bash

git clone https://github.com/ronimos/snowpack.git
cd snowpack
```
2. **Install the project and its dependencies:**
The project dependencies are listed in pyproject.toml.

```Bash

# Install the package in editable mode
uv pip install -e .
```
3. **(Optional) Install CuPy for GPU Acceleration:**
If you have an NVIDIA GPU and the CUDA Toolkit, install `cupy` to enable GPU support. For example, for CUDA 12.x:

```Bash

uv pip install cupy-cuda12x
```

## Usage
The script is run directly from the command line, requiring the path to a `.pro` file. You can optionally specify a start and end date for the analysis.

### Basic Usage

To analyze the entire time series within a file, provide the file path as an argument. The script will automatically use the earliest and latest dates in the data. Use `--pro_file` (`-f`).

```Bash

python main.py -f /path/to/your/data.pro
```

### Specifying a Date Range
To focus on a specific period, use the `--start` (`-s`) and `--end` (`-e`) flags in `YYYY-MM-DD` format.

```Bash

python main.py --pro_file /path/to/your/data.pro --start 2025-02-15 --end 2025-03-20
```

This command executes the primary workflow defined in `main.py`. It will:

1. Reade the `.pro` file.

2. Run the daily stability calculations for the configured date range.

3. Print a summary table to your console.

4. Display the summary plot.

## Project Components
- `snowpack_reader.py`: The core data engine. This module contains the `SnowpackProfile` class responsible for parsing `.pro` files, handling the GPU/CPU backend, caching, and providing the main data analysis methods (`slice`, `get_profile_summary`, etc.).

- `wet_snow_tracker.py`: The scientific analysis library. This module provides the specialized functions for identifying weak layers and tracking water, which are designed to be plugged into the `get_profile_summary` method.

- `main.py`: The main executable script. This file serves as the primary entry point and demonstrates how to tie the reader and analysis functions together to perform a complete, end-to-end wet snow stability analysis.

- `pyproject.toml`: The project configuration file. It defines the project's dependencies, metadata, and the `run-analysis` command-line script.