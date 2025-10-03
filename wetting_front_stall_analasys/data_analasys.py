"""
analyze_stall_data.py
=====================

This script reads the CSV output from the `find_stalled_fronts.py` analysis
and generates a series of plots to visualize the data. It aims to uncover
relationships between the duration of wetting front stalls and various snowpack
properties like terrain aspect, grain size, and liquid water content.

The script saves each generated plot as a PNG file in a dedicated 'plots'
subdirectory within the 'special_analysis_results' folder.

Workflow:
---------
1.  **Load Data**: Reads the `stalled_wetting_fronts_analysis.csv` file.
2.  **Pre-process Data**: Cleans and transforms data for plotting. This includes
    calculating the average LWC in the slab from the LWC profile lists.
3.  **Generate Plots**: Creates and saves several plots:
    - A boxplot showing the distribution of stall durations for each aspect.
    - A series of scatter plots (faceted by aspect) showing the relationship
      between the grain size at the interface and the stall duration.
    - A series of scatter plots (faceted by aspect) showing the relationship
      between the average LWC of the slab and the stall duration.

Usage:
------
Run this script from the root of the project after `find_stalled_fronts.py`
has been successfully run:
`python special_analysis/analyze_stall_data.py`
"""
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# --- Configuration ---
analysis_project_root = Path(__file__).resolve().parent
INPUT_CSV = analysis_project_root / "front_stall_snowpack_data" / "stalled_wetting_fronts_data.csv"
OUTPUT_DIR = analysis_project_root / "plots"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares the DataFrame for plotting.
    - Converts string representations of lists back into actual lists.
    - Calculates average slab LWC.
    """
    logging.info("Preprocessing data...")
    # Safely evaluate string-formatted lists into actual lists
    for col in ['slab_lwc_profile_start', 'slab_lwc_profile_end']:
        # Ensure the column exists and handle potential errors
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else None
            )

    # Calculate average LWC for the slab
    if 'slab_lwc_profile_start' in df.columns:
        df['avg_slab_lwc_start'] = df['slab_lwc_profile_start'].apply(
            lambda x: sum(x) / len(x) if x and isinstance(x, list) else None
        )
        
    # Ensure aspect is treated as a categorical variable for plotting
    if 'aspect' in df.columns:
        df['aspect'] = df['aspect'].astype('category')
        
    logging.info("Preprocessing complete.")
    return df

def plot_duration_by_aspect(df: pd.DataFrame, output_dir: Path):
    """
    Creates and saves a boxplot of stall durations for each aspect.
    """
    logging.info("Generating plot: Stall Duration by Aspect...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='aspect', y='duration_hours', order=['N', 'E', 'S', 'W', 'Flat'])
    
    plt.title('Distribution of Wetting Front Stall Durations by Aspect')
    plt.xlabel('Aspect')
    plt.ylabel('Stall Duration (hours)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = output_dir / "stall_duration_by_aspect.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def plot_grain_size_vs_duration(df: pd.DataFrame, output_dir: Path):
    """
    Creates a faceted scatter plot of grain size vs. stall duration for each aspect.
    """
    logging.info("Generating plot: Grain Size vs. Stall Duration...")
    # We use the grain size of the layer *below* the stall as it often forms the barrier
    if 'below_grain_size' not in df.columns:
        logging.warning("Column 'below_grain_size' not found. Skipping plot.")
        return

    g = sns.lmplot(
        data=df,
        x='below_grain_size',
        y='duration_hours',
        col='aspect',
        col_wrap=3,
        col_order=['N', 'E', 'S', 'W', 'Flat'],
        height=4,
        scatter_kws={'alpha': 0.6}
    )
    
    g.fig.suptitle('Stall Duration vs. Grain Size of Layer Below Interface', y=1.03)
    g.set_axis_labels('Grain Size Below (mm)', 'Stall Duration (hours)')
    
    output_path = output_dir / "grain_size_vs_duration.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def plot_lwc_vs_duration(df: pd.DataFrame, output_dir: Path):
    """
    Creates a faceted scatter plot of average slab LWC vs. stall duration.
    """
    logging.info("Generating plot: Average Slab LWC vs. Stall Duration...")
    if 'avg_slab_lwc_start' not in df.columns:
        logging.warning("Column 'avg_slab_lwc_start' not found. Skipping plot.")
        return

    g = sns.lmplot(
        data=df,
        x='avg_slab_lwc_start',
        y='duration_hours',
        col='aspect',
        col_wrap=3,
        col_order=['N', 'E', 'S', 'W', 'Flat'],
        height=4,
        scatter_kws={'alpha': 0.6}
    )
    
    g.fig.suptitle('Stall Duration vs. Average LWC of Slab at Start of Stall', y=1.03)
    g.set_axis_labels('Average Slab LWC (volumetric %)', 'Stall Duration (hours)')
    
    output_path = output_dir / "avg_lwc_vs_duration.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def main():
    """
    Main function to load data and generate all plots.
    """
    if not INPUT_CSV.exists():
        logging.error(f"Input data file not found at: {INPUT_CSV}")
        logging.error("Please run `find_stalled_fronts.py` first to generate the data.")
        return

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Plots will be saved to: {OUTPUT_DIR}")

    df = pd.read_csv(INPUT_CSV)
    df = preprocess_data(df)

    # Generate and save all plots
    plot_duration_by_aspect(df, OUTPUT_DIR)
    plot_grain_size_vs_duration(df, OUTPUT_DIR)
    plot_lwc_vs_duration(df, OUTPUT_DIR)
    
    logging.info("All plots have been generated successfully.")

if __name__ == "__main__":
    main()
