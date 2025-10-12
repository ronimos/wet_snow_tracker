"""
plotting.py
===========

Visualization functions for the Wetting Front Tracker application.

This module creates:
- Static plots (Matplotlib PNG) for detailed analysis
- Interactive plots (Plotly HTML) with zoom/pan capabilities
- Summary map (Folium) showing risk levels across all polygons

Author: Ron Simenhois
Last Updated: October 12, 2025
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from branca.element import Element, MacroElement, Template
from folium import GeoJson, GeoJsonPopup, GeoJsonTooltip
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image

from .param_config import ASSETS_SUBFOLDER_NAME, RESULTS_PATH, get_html_path, get_png_path

matplotlib.use('Agg')

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thumbnail settings
THUMBNAIL_MAX_SIZE = (800, 534)
THUMBNAIL_OPTIMIZE = True

# Plot dimensions
MATPLOTLIB_FIGSIZE = (14, 8)
MATPLOTLIB_DPI = 300

# LWC colormap settings
LWC_COLORMAP_COLORS = ["white", "blue", "orange", "red"]
LWC_VMIN = 0
LWC_VMAX = 500
LWC_COLORBAR_TICKS = [0, 100, 200, 300, 400, 500]
LWC_COLORBAR_LABELS = ['0', '1', '2', '3', '4', '5+']

# Time to LOC color scheme (hours)
TIME_TO_LOC_COLORS = {
    'imminent': ('darkred', 0, 24),       # 0-24h
    'near': ('orange', 24, 48),           # 24-48h  
    'moderate': ('yellow', 48, 72),       # 48-72h
    'recent': ('red', -24, 0),            # -24-0h
    'past_near': ('lightblue', -48, -24), # -48--24h
    'past_far': ('darkblue', -72, -48),   # -72--48h
    'unknown': ('gray', None, None)       # No data
}

# Aspect mapping
ASPECT_MAP = {
    'N': 'north',
    'E': 'east',
    'S': 'south',
    'W': 'west',
    'Flat': 'flat'
}

# External visualization URL
SNOWPACK_VIEWER_BASE_URL = "https://nwp.mtnweather.info/snowpack/spvizll.php"

# Map settings
MAP_DEFAULT_LOCATION = [40, -105]
MAP_DEFAULT_ZOOM = 8


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

class PlottingError(Exception):
    """Raised when plotting operations fail."""
    pass


def validate_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> None:
    """
    Validates a DataFrame has required columns and data.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Raises:
        PlottingError: If validation fails
    """
    if df.empty:
        raise PlottingError("DataFrame is empty")
    
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise PlottingError(f"Missing required columns: {missing}")


def validate_metadata(metadata: Dict[str, Any], required_keys: Optional[List[str]] = None) -> None:
    """
    Validates metadata dictionary has required keys.
    
    Args:
        metadata: Metadata dictionary to validate
        required_keys: List of required keys
        
    Raises:
        PlottingError: If validation fails
    """
    if not metadata:
        raise PlottingError("Metadata dictionary is empty")
    
    if required_keys:
        missing = [key for key in required_keys if key not in metadata]
        if missing:
            logger.warning(f"Missing metadata keys: {missing}")


# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------

def create_thumbnail(
    source_path: Path,
    output_path: Path,
    max_size: Tuple[int, int] = THUMBNAIL_MAX_SIZE
) -> bool:
    """
    Creates a web-optimized PNG thumbnail from a larger image.

    Args:
        source_path: Path to the source PNG image
        output_path: Path to save the thumbnail
        max_size: Maximum (width, height) for the thumbnail

    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(source_path) as img:
            img.thumbnail(max_size)
            img.save(output_path, "PNG", optimize=THUMBNAIL_OPTIMIZE)
        logger.debug(f"Created thumbnail: {output_path}")
        return True
    
    except FileNotFoundError:
        logger.warning(f"Source image not found: {source_path}")
        return False
    
    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")
        return False


# ---------------------------------------------------------------------------
# URL Generation
# ---------------------------------------------------------------------------

def generate_snowpack_viewer_url(
    metadata: Dict[str, Any],
    central_date: Optional[datetime] = None
) -> Optional[str]:
    """
    Constructs URL for the external snowpack visualization tool.

    Args:
        metadata: Station metadata (must include lat, lon, aspect)
        central_date: Reference date for determining the season

    Returns:
        Formatted URL string, or None if metadata is insufficient
    """
    lat = metadata.get('latitude')
    lon = metadata.get('longitude')
    aspect = metadata.get('aspect')
    
    if not all([lat, lon, aspect]):
        logger.debug("Insufficient metadata for viewer URL generation")
        return None
    
    # Determine season (Oct-Sep)
    date = central_date or datetime.now()
    season = date.year - 1 if date.month < 10 else date.year
    
    # Convert aspect to word format
    aspect_word = ASPECT_MAP.get(aspect, 'flat') if isinstance(aspect, str) else 'flat'
    
    return (
        f"{SNOWPACK_VIEWER_BASE_URL}?"
        f"lat={lat}&lon={lon}&aspect={aspect_word}&season={season}"
    )


# ---------------------------------------------------------------------------
# HTML Template Generation
# ---------------------------------------------------------------------------

def generate_html_template(
    plotly_fig_html: str,
    metadata: Dict[str, Any],
    central_date: Optional[datetime] = None
) -> str:
    """
    Embeds a Plotly figure and metadata into a full HTML page template.

    Args:
        plotly_fig_html: HTML string of the Plotly figure
        metadata: Profile metadata for titles and links
        central_date: Reference date for the analysis

    Returns:
        Complete HTML page content as string
    """
    station_name = metadata.get('stationName', 'Unknown Station')
    
    # Format date strings
    date_title = ""
    date_header = ""
    if central_date:
        date_str = central_date.strftime('%Y-%m-%d %H:%M UTC')
        date_title = f" | Analysis for {date_str}"
        date_header = f"Analysis for: {date_str}"
    
    # Generate external viewer link
    viewer_url = generate_snowpack_viewer_url(metadata, central_date)
    if viewer_url:
        viewer_section = f'''
        <h2>Snowpack Visualization</h2>
        <iframe class="iframe-container" src="{viewer_url}" 
                title="Snowpack Visualization"></iframe>
        '''
    else:
        viewer_section = "<h2>Snowpack Visualization Link Not Available</h2>"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wetting Front Analysis: {station_name}{date_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", 
                         Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header-container {{
            padding: 15px 25px;
            background-color: #343a40;
            color: white;
            border-bottom: 4px solid #007bff;
        }}
        .header-container h1 {{
            margin: 0;
            font-size: 1.6em;
        }}
        .iframe-container {{
            width: 100%;
            height: 700px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }}
        .content-section {{
            padding: 20px;
        }}
        h2 {{
            text-align: center;
            color: #007bff;
            margin-top: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header-container">
        <h1>Wetting Front Height & Layers of Concern: {station_name}</h1>
        <p style="margin:0; font-size: 1.1em; color: #ddd;">{date_header}</p>
    </div>
    <div class="content-section">
        {viewer_section}
    </div>
    <div class="content-section">
        <h2>Wetting Front Analysis</h2>
        {plotly_fig_html}
    </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Matplotlib Plotting - LWC Colormesh
# ---------------------------------------------------------------------------

def plot_lwc_colormesh(
    ax: Axes,
    lwc_data: xr.Dataset
) -> Optional[QuadMesh]:
    """
    Plots Liquid Water Content (LWC) data as a colormesh.

    Args:
        ax: Matplotlib axes object
        lwc_data: xarray Dataset with 'lwc' and 'height' variables

    Returns:
        Matplotlib QuadMesh object, or None if plotting fails
    """
    try:
        # Extract data (handle both numpy and cupy arrays)
        try:
            lwc_values = lwc_data['lwc'].to_numpy().T * 100
            height_values_raw = lwc_data['height'].to_numpy().T
            timestamps_num = mdates.date2num(lwc_data.timestamp.to_numpy())
        except AttributeError:
            lwc_values = lwc_data['lwc'].values.T * 100
            height_values_raw = lwc_data['height'].values.T
            timestamps_num = mdates.date2num(lwc_data.timestamp.values)

        layer_indices = lwc_data.layer_index.values
        X, _ = np.meshgrid(timestamps_num, layer_indices)
        
        # Fill missing height values
        df_heights = pd.DataFrame(height_values_raw, dtype=np.float64)
        df_heights_filled = df_heights.ffill(axis=0).bfill(axis=0)
        
        if df_heights_filled.isnull().values.any():
            logger.warning("Cannot plot LWC: persistent NaN height values")
            return None

        height_values = df_heights_filled.to_numpy()
        lwc_masked = np.ma.masked_where(df_heights.isnull().to_numpy(), lwc_values)

        # Create colormap and plot
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom",
            LWC_COLORMAP_COLORS
        )
        norm = mcolors.Normalize(vmin=LWC_VMIN, vmax=LWC_VMAX)
        
        return ax.pcolormesh(
            X,
            height_values,
            lwc_masked,
            cmap=cmap,
            norm=norm,
            shading="gouraud",
            zorder=1
        )
    
    except Exception as e:
        logger.error(f"Failed to plot LWC colormesh: {e}")
        return None


def clip_colormesh_to_wet_area(
    ax: Axes,
    colormesh: QuadMesh,
    df: pd.DataFrame
) -> None:
    """
    Creates a clipping path to show LWC only within detected wet layer.

    Args:
        ax: Matplotlib axes object
        colormesh: Colormesh object to clip
        df: Summary DataFrame with boundary data
    """
    try:
        # Resample to hourly for smooth boundaries
        df_resampled = df.asfreq('h')
        x_dense = mdates.date2num(df_resampled.index)
        
        # Interpolate boundaries
        cy_series = df_resampled['highest_wet_point'].interpolate(
            method='linear'
        ).bfill().ffill()
        sy_series = df_resampled['wet_front_lwc_height'].interpolate(
            method='linear'
        ).bfill().ffill()

        # Validate data
        if cy_series.isnull().all() or sy_series.isnull().all():
            logger.debug("Insufficient boundary data for clipping")
            return

        cy = cy_series.to_numpy(dtype=float)
        sy = sy_series.to_numpy(dtype=float)
        
        # Create clipping path
        verts = np.concatenate([
            np.column_stack([x_dense, cy]),
            np.column_stack([x_dense[::-1], sy[::-1]])
        ])
        path = MplPath(verts)
        patch = PathPatch(
            path,
            transform=ax.transData,
            facecolor='none',
            edgecolor='none'
        )
        colormesh.set_clip_path(patch)
        ax.add_patch(patch)
    
    except Exception as e:
        logger.warning(f"Failed to clip colormesh: {e}")


# ---------------------------------------------------------------------------
# Matplotlib Plotting - Line Series
# ---------------------------------------------------------------------------

def plot_line_series(
    ax: Axes,
    df: pd.DataFrame,
    central_date: Optional[datetime] = None
) -> None:
    """
    Plots primary data series (HS, LOC, Wet Front) on the axes.

    Args:
        ax: Matplotlib axes object
        df: Summary DataFrame with data series
        central_date: Reference date, plotted as vertical line
    """
    # Total snow depth
    if 'hs' in df.columns:
        ax.plot(
            df.index,
            df['hs'],
            label='Total Snow Depth (HS)',
            color='navy',
            marker='.',
            linestyle='-',
            zorder=10
        )
    
    # Weak layer height
    if 'weak_layer_height' in df.columns:
        ax.plot(
            df.index,
            df['weak_layer_height'],
            label='Weak Layer Height (LOC)',
            color='black',
            zorder=10
        )
    
    # Wet front (plot in segments to avoid connecting across gaps)
    if 'wet_front_lwc_height' in df.columns:
        wet_front = df['wet_front_lwc_height']
        is_valid = wet_front.notna()
        
        # Find segment boundaries
        starts = wet_front.index[
            is_valid & ~is_valid.shift(1, fill_value=False).astype(bool)
        ]
        ends = wet_front.index[
            is_valid & ~is_valid.shift(-1, fill_value=False).astype(bool)
        ]
        
        # Plot each segment
        for i, (start, end) in enumerate(zip(starts, ends)):
            segment = wet_front.loc[start:end]
            label = 'Deepest Wet Front (LWC > 3%)' if i == 0 else None
            ax.plot(
                segment.index.to_numpy(),
                np.asarray(segment, dtype=float),
                color='red',
                zorder=10,
                label=label
            )
    
    # Central date vertical line
    if central_date:
        date_num = float(mdates.date2num(central_date))
        ax.axvline(
            x=date_num,
            color='purple',
            linestyle='--',
            linewidth=2,
            label='Central Date',
            zorder=11
        )
        ax.text(
            date_num,
            ax.get_ylim()[0],
            central_date.strftime('%Y-%m-%d'),
            rotation=90,
            verticalalignment='bottom',
            color='purple',
            fontsize=10
        )


def configure_plot_aesthetics(
    fig: Figure,
    ax: Axes,
    metadata: Dict[str, Any],
    colormesh: Optional[QuadMesh] = None,
    central_date: Optional[datetime] = None
) -> None:
    """
    Configures plot titles, labels, grid, legend, and axes formatting.

    Args:
        fig: Matplotlib Figure object
        ax: Matplotlib Axes object
        metadata: Metadata for plot title
        colormesh: Colormesh object for colorbar (if applicable)
        central_date: Central analysis date
    """
    # Build title
    location = (metadata.get("latitude"), metadata.get('longitude'))
    elevation = metadata.get("altitude")
    aspect = metadata.get("slopeAzi", "N/A")
    
    date_info = ""
    if central_date:
        date_info = f"Analysis for: {central_date.strftime('%Y-%m-%d %H:%M UTC')}"
    
    title = (
        f"Wetting Front Tracking: {metadata.get('stationName', 'N/A')}\n"
        f"Loc: {location}, Elev: {elevation}m, Aspect: {aspect}\n"
        f"{date_info}"
    )
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add colorbar if colormesh exists
    if colormesh:
        cbar = fig.colorbar(
            colormesh,
            ax=ax,
            label='Liquid Water Content (%)',
            extend='max'
        )
        cbar.set_ticks(LWC_COLORBAR_TICKS)
        cbar.set_ticklabels(LWC_COLORBAR_LABELS)

    # Configure legend with specific order
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        'Total Snow Depth (HS)',
        'Deepest Wet Front (LWC > 3%)',
        'Weak Layer Height (LOC)',
        'Central Date'
    ]
    
    ordered_handles = []
    ordered_labels = []
    for key in desired_order:
        if key in labels:
            idx = labels.index(key)
            ordered_handles.append(handles[idx])
            ordered_labels.append(key)
    
    if ordered_handles:
        ax.legend(ordered_handles, ordered_labels)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    fig.autofmt_xdate()
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Main Plotting Functions - Matplotlib
# ---------------------------------------------------------------------------

def plot_summary_matplotlib(
    df: pd.DataFrame,
    file_stem: str,
    metadata: Dict[str, Any],
    lwc_plot_data: Optional[xr.Dataset] = None,
    central_date: Optional[datetime] = None,
    assets_dir: Optional[Path] = None
) -> bool:
    """
    Generates and saves a static PNG plot of snowpack analysis.

    Args:
        df: Summary DataFrame with daily analysis results
        file_stem: Unique identifier for output file naming
        metadata: Metadata about the snowpack profile
        lwc_plot_data: Full-resolution LWC data for colormesh
        central_date: Reference date for the analysis
        assets_dir: Directory for saving plot assets

    Returns:
        True if successful, False otherwise
    """
    try:
        validate_dataframe(df)
        validate_metadata(metadata)
        
        if assets_dir is None:
            logger.warning("Assets directory not provided. Cannot save plot.")
            return False
        
        # Create figure
        fig, ax = plt.subplots(figsize=MATPLOTLIB_FIGSIZE)
        
        # Plot LWC colormesh (if available)
        colormesh = None
        if lwc_plot_data is not None and 'lwc' in lwc_plot_data:
            if not lwc_plot_data['lwc'].isnull().to_numpy().all():
                colormesh = plot_lwc_colormesh(ax, lwc_plot_data)
                if colormesh:
                    clip_colormesh_to_wet_area(ax, colormesh, df)
        
        # Plot line series
        plot_line_series(ax, df, central_date)
        
        # Configure aesthetics
        configure_plot_aesthetics(fig, ax, metadata, colormesh, central_date)
        
        # Save figure
        output_path = get_png_path(file_stem, assets_dir)
        plt.savefig(output_path, dpi=MATPLOTLIB_DPI)
        plt.close(fig)
        
        logger.debug(f"Saved Matplotlib plot: {output_path}")
        return True
    
    except PlottingError as e:
        logger.error(f"Validation error in plot_summary_matplotlib: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Failed to create Matplotlib plot: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main Plotting Functions - Plotly
# ---------------------------------------------------------------------------

def create_plotly_figure(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    central_date: Optional[datetime] = None
) -> go.Figure:
    """
    Creates an interactive Plotly figure from summary data.

    Args:
        df: Summary DataFrame with analysis results
        metadata: Profile metadata
        central_date: Central analysis date

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Plot filled area for wet layer (in segments)
    if 'wet_front_lwc_height' in df.columns and 'highest_wet_point' in df.columns:
        is_valid = (
            df['wet_front_lwc_height'].notna() &
            df['highest_wet_point'].notna()
        )
        starts = df.index[is_valid & ~is_valid.shift(1, fill_value=False)]
        ends = df.index[is_valid & ~is_valid.shift(-1, fill_value=False)]
        
        for start, end in zip(starts, ends):
            segment = df.loc[start:end]
            if len(segment) > 1:
                x_coords = segment.index.tolist() + segment.index.tolist()[::-1]
                y_coords = (
                    segment['highest_wet_point'].tolist() +
                    segment['wet_front_lwc_height'].tolist()[::-1]
                )
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor='rgba(0, 200, 200, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # Add legend entry for wet area
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color='rgba(0, 200, 200, 0.4)', size=10),
        name='Wet Layer Extent'
    ))

    # Plot line series
    if 'hs' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['hs'],
            name='Total Snow Depth (HS)',
            mode='lines+markers',
            line=dict(color='darkblue')
        ))
    
    if 'weak_layer_height' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['weak_layer_height'],
            name='Weak Layer Height (LOC)',
            mode='lines',
            line=dict(color='black', width=2)
        ))
    
    if 'wet_front_lwc_height' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['wet_front_lwc_height'],
            name='Deepest Wet Front (LWC > 3%)',
            mode='lines',
            line=dict(color='red', width=2),
            connectgaps=False
        ))
    
    # Configure layout
    date_str = ""
    if central_date:
        date_str = f" | Analysis for {central_date.strftime('%Y-%m-%d %H:%M UTC')}"
    
    title = f"Wetting Front Analysis for {metadata.get('stationName', 'N/A')}{date_str}"
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Height (cm)',
        legend_title_text='Metrics',
        template='plotly_white',
    )
    
    # Set y-axis range
    if 'hs' in df.columns and df['hs'].notna().any():
        fig.update_yaxes(range=[0, df['hs'].max() * 1.1])
    
    return fig


def plot_summary_plotly(
    df: pd.DataFrame,
    file_stem: str,
    metadata: Dict[str, Any],
    central_date: Optional[datetime] = None,
    assets_dir: Optional[Path] = None
) -> bool:
    """
    Generates an interactive HTML page with a Plotly plot.

    Args:
        df: Summary DataFrame with analysis results
        file_stem: Unique identifier for output file naming
        metadata: Metadata about the snowpack profile
        central_date: Central analysis date
        assets_dir: Directory for saving plot assets

    Returns:
        True if successful, False otherwise
    """
    try:
        validate_dataframe(df)
        validate_metadata(metadata)
        
        if assets_dir is None:
            logger.warning("Assets directory not provided. Cannot save plot.")
            return False
        
        # Create Plotly figure
        fig = create_plotly_figure(df, metadata, central_date)
        
        # Generate full HTML page
        plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        full_html = generate_html_template(plotly_html, metadata, central_date)
        
        # Save HTML file
        output_path = get_html_path(file_stem, assets_dir)
        with open(output_path, 'w') as f:
            f.write(full_html)
        
        logger.debug(f"Saved Plotly plot: {output_path}")
        return True
    
    except PlottingError as e:
        logger.error(f"Validation error in plot_summary_plotly: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Failed to create Plotly plot: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Folium Map - Color Coding
# ---------------------------------------------------------------------------

def get_time_to_loc_color(time_to_loc: float) -> str:
    """
    Determines polygon color based on time_to_loc value.

    Args:
        time_to_loc: Time in hours for wetting front to reach LOC

    Returns:
        Color name as string
    """
    if pd.isna(time_to_loc):
        return TIME_TO_LOC_COLORS['unknown'][0]
    
    time = float(time_to_loc)
    
    for category, (color, min_time, max_time) in TIME_TO_LOC_COLORS.items():
        if min_time is not None and max_time is not None:
            if min_time <= time < max_time:
                return color
    
    return TIME_TO_LOC_COLORS['unknown'][0]


# ---------------------------------------------------------------------------
# Folium Map - HTML Generation
# ---------------------------------------------------------------------------

def generate_tooltip_html(row: pd.Series, assets_dir: Path) -> str:
    """
    Generates HTML content for map tooltip.

    Args:
        row: GeoDataFrame row with polygon data
        assets_dir: Directory containing plot assets

    Returns:
        HTML string for tooltip
    """
    if pd.isna(row.get('file_stem')):
        return ""
    
    # Create thumbnail
    png_path = get_png_path(row['file_stem'], assets_dir)
    thumb_path = png_path.parent / f"{png_path.stem}_thumb.png"
    create_thumbnail(png_path, thumb_path)
    
    # Build HTML
    image_path = f"{ASSETS_SUBFOLDER_NAME}/{thumb_path.name}"
    
    date_str = ""
    if pd.notna(row.get('central_date_str')):
        date_str = f"Analysis Date: {row['central_date_str']}<br>"
    
    return (
        f"<b>{row.get('pathName', 'Unknown Path')}</b><br>"
        f"{date_str}"
        f"Aspect: {row.get('aspect', 'N/A')}<br>"
        f'<img src="{image_path}" width="800">'
    )


def generate_popup_html(row: pd.Series, assets_dir: Path) -> str:
    """
    Generates HTML content for map popup.

    Args:
        row: GeoDataFrame row with polygon data
        assets_dir: Directory containing plot assets

    Returns:
        HTML string for popup
    """
    if pd.isna(row.get('file_stem')):
        return ""
    
    html_path = get_html_path(row['file_stem'], assets_dir)
    link_path = f"{ASSETS_SUBFOLDER_NAME}/{html_path.name}"
    
    date_str = ""
    if pd.notna(row.get('central_date_str')):
        date_str = f"Analysis Date: {row['central_date_str']}<br>"
    
    return (
        f"<b>{row.get('station_name', 'Unknown Station')}</b><br>"
        f"{date_str}"
        f'<a href="{link_path}" target="_blank">Open Interactive Plot</a>'
    )


# ---------------------------------------------------------------------------
# Folium Map - Legend and Title
# ---------------------------------------------------------------------------

def create_map_legend_html() -> str:
    """
    Creates HTML for the map legend.

    Returns:
        HTML string for legend
    """
    return """
     <b>Time for Wetting Front to Reach LOC</b><br>
     <i style="background:darkred"></i> 0 to 24h (Imminent)<br>
     <i style="background:orange"></i> 24 to 48h<br>
     <i style="background:yellow"></i> 48 to 72h<br>
     <hr style='border-top: 1px solid grey; margin-top: 5px; margin-bottom: 5px;'>
     <i style="background:red"></i> -24 to 0h (Recent)<br>
     <i style="background:lightblue"></i> -48 to -24h (Past)<br>
     <i style="background:darkblue"></i> -72 to -48h (Past)<br>
     <hr style='border-top: 1px solid grey; margin-top: 5px; margin-bottom: 5px;'>
     <i style="background:gray"></i> Other / No Data
"""


def create_map_title_html(central_date: Optional[datetime]) -> str:
    """
    Creates HTML for the map title.

    Args:
        central_date: Central analysis date

    Returns:
        HTML string for title
    """
    date_str = ""
    if central_date:
        date_str = central_date.strftime('%Y-%m-%d %H:%M')
    
    return f'<h3 align="center" style="font-size:16px"><b>Wetting Front Analysis | {date_str}</b></h3>'


def create_map_persistence_javascript(map_name: str) -> str:
    """
    Creates JavaScript for saving/restoring map state in localStorage.

    Args:
        map_name: The Folium map variable name

    Returns:
        JavaScript code as string
    """
    return f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        const DEBUG_MODE = false;
        var mapObj = window.{map_name};
        if (!mapObj) return;

        function log(message) {{
            if (DEBUG_MODE) console.log(message);
        }}

        // Restore last view
        var lastView = localStorage.getItem('preferredView');
        if (lastView) {{
            try {{
                var view = JSON.parse(lastView);
                mapObj.setView(view.center, view.zoom);
            }} catch(e) {{}}
        }}

        // Restore last basemap
        setTimeout(function() {{
            var lastBase = localStorage.getItem('preferredBaseLayer');
            log("Restoring basemap: " + lastBase);

            var found = false;
            var layerControl = document.querySelector('.leaflet-control-layers-base');
            if (layerControl) {{
                var labels = layerControl.querySelectorAll('label');
                labels.forEach(function(label) {{
                    var layerName = label.textContent.trim();
                    if (layerName === lastBase) {{
                        var input = label.querySelector('input[type=radio]');
                        if (input && !input.checked) {{
                            input.click();
                            found = true;
                            log("Restored basemap: " + layerName);
                        }}
                    }}
                }});
            }}
        }}, 750);

        // Save on base layer change
        mapObj.on('baselayerchange', function(e) {{
            localStorage.setItem('preferredBaseLayer', e.name);
            log("Saved basemap: " + e.name);
        }});

        // Save view on move/zoom
        function saveView() {{
            var center = mapObj.getCenter();
            var zoom = mapObj.getZoom();
            localStorage.setItem('preferredView', JSON.stringify({{
                center: [center.lat, center.lng],
                zoom: zoom
            }}));
        }}
        mapObj.on('moveend', saveView);
        mapObj.on('zoomend', saveView);
    }});
    </script>
    """


# ---------------------------------------------------------------------------
# Folium Map - Main Function
# ---------------------------------------------------------------------------

def create_folium_map(
    final_gdf: gpd.GeoDataFrame,
    map_output_path: Path,
    central_date: datetime,
    assets_dir: Path
) -> bool:
    """
    Creates a Folium summary map with polygons colored by risk level.

    Args:
        final_gdf: GeoDataFrame with polygon geometries and analysis results
        map_output_path: Path to save the summary_map.html file
        central_date: Central analysis date for the map title
        assets_dir: Directory where plot assets are stored

    Returns:
        True if successful, False otherwise
    """
    try:
        if final_gdf.empty:
            logger.warning("GeoDataFrame is empty. Cannot create map.")
            return False
        
        # Repair geometries
        final_gdf['geometry'] = final_gdf.geometry.buffer(0)
        
        # Calculate areas
        gdf_proj = final_gdf.to_crs("EPSG:3857")
        final_gdf['area_sq_meters'] = gdf_proj.geometry.area
        
        # Add color, tooltip, and popup columns
        final_gdf['color'] = final_gdf['time_to_loc'].apply(get_time_to_loc_color)
        final_gdf['tooltip'] = final_gdf.apply(
            lambda row: generate_tooltip_html(row, assets_dir),
            axis=1
        )
        final_gdf['popup'] = final_gdf.apply(
            lambda row: generate_popup_html(row, assets_dir),
            axis=1
        )
        
        # Save map data
        map_data_path = RESULTS_PATH / "map_data.geojson"
        final_gdf.to_file(map_data_path, driver='GeoJSON')
        
        # Calculate map center
        map_center = final_gdf.to_crs("EPSG:4269").unary_union.centroid
        center_coords = (
            [map_center.y, map_center.x]
            if map_center
            else MAP_DEFAULT_LOCATION
        )
        
        # Create map
        m = folium.Map(location=center_coords, zoom_start=MAP_DEFAULT_ZOOM)
        
        # Add tile layers
        folium.TileLayer('OpenStreetMap', name='Street View').add_to(m)
        folium.TileLayer('OpenTopoMap', name='Topographic').add_to(m)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite'
        ).add_to(m)
        
        # Add GeoJSON layer
        def style_function(x):
            return {
                "fillColor": x['properties']['color'],
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6
            }
        
        gjson = GeoJson(
            str(map_data_path.resolve()),
            style_function=style_function,
            name='Avalanche Path Risk',
            tooltip=GeoJsonTooltip(
                fields=['tooltip'],
                aliases=[''],
                localize=True,
                sticky=True
            ),
            popup=GeoJsonPopup(
                fields=['popup'],
                aliases=[''],
                localize=True
            )
        )
        gjson.add_to(m)
        
        # Add title and legend
        title_html = create_map_title_html(central_date)
        legend_html = create_map_legend_html()
        
        template = f"""
        {{% macro script(this, kwargs) %}}
            var title = L.control({{position: 'topright'}});
            title.onAdd = function (map) {{
                var div = L.DomUtil.create('div', 'info');
                div.innerHTML = `{title_html}`;
                return div;
            }};
            title.addTo({m.get_name()});

            var legend = L.control({{position: 'bottomleft'}});
            legend.onAdd = function (map) {{
                var div = L.DomUtil.create('div', 'info legend');
                div.innerHTML = `{legend_html}`;
                return div;
            }};
            legend.addTo({m.get_name()});

            var style = document.createElement('style');
            style.innerHTML = `
                .legend {{
                    line-height: 20px;
                    color: #333;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 5px;
                    border: 2px solid #aaa;
                }}
                .legend i {{
                    width: 18px;
                    height: 18px;
                    float: left;
                    margin-right: 8px;
                    opacity: 0.9;
                }}
            `;
            document.head.appendChild(style);
        {{% endmacro %}}
        """
        
        macro = MacroElement()
        macro._template = Template(template)
        m.get_root().add_child(macro)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add persistence JavaScript
        persistence_js = create_map_persistence_javascript(m.get_name())
        m.get_root().html.add_child(Element(persistence_js))  # type: ignore
        
        # Save map
        m.save(str(map_output_path))
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        logger.info(f"Summary map saved to: {map_output_path} at {timestamp}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create Folium map: {e}", exc_info=True)
        return False