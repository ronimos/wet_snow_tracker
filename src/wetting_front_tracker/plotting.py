"""
plotting.py
===========

This module contains all functions for creating visual outputs for the Wetting
Front Tracker application. It is responsible for generating static and interactive
plots for individual snowpack analyses, as well as the final summary map that
aggregates the results for all processed polygons.

Key Responsibilities:
---------------------
- **Static Plots (Matplotlib):** Creates detailed, static PNG plots for each
  individual snowpack profile analysis. These plots show total snow depth,
  weak layer location, and the progression of the wetting front over time,
  overlaid on a colormesh of Liquid Water Content (LWC).
- **Interactive Plots (Plotly):** Generates standalone HTML files containing
  interactive Plotly charts. These charts provide the same information as the
- **Summary Map (Folium):** Produces a final, interactive HTML map that displays
  all processed avalanche path polygons. Polygons are color-coded based on the
  calculated `time_to_loc` metric, providing a geographic overview of wet slab
  avalanche risk. The map includes popups with links to the detailed interactive
  plots for each polygon.
- **Helper Functions:** Includes utilities for creating plot thumbnails, generating
  URLs for external visualization tools, and embedding plots within a standardized
  HTML template.

This module acts as the visualization layer of the application, transforming the
numerical results from the analysis into easily interpretable charts and maps.
"""
from datetime import datetime
import folium
from folium import GeoJson, GeoJsonTooltip, GeoJsonPopup
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import QuadMesh
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from PIL import Image
from typing import Any, Dict, Optional
from branca.element import Template, MacroElement # type: ignore

from .param_config import get_png_path, get_html_path, RESULTS_PATH, ASSETS_SUBFOLDER_NAME  

def _create_thumbnail(png_path: Path, thumb_path: Path, max_size: tuple[int, int] = (800, 534)) -> None:
    """
    Creates a smaller, web-optimized PNG thumbnail from a larger image.

    This is used to generate quick-loading previews for the Folium map tooltips.

    Args:
        png_path (Path): The path to the source PNG image.
        thumb_path (Path): The path to save the generated thumbnail.
        max_size (tuple[int, int]): The maximum (width, height) of the thumbnail.
    """
    try:
        with Image.open(png_path) as img:
            img.thumbnail(max_size)
            img.save(thumb_path, "PNG", optimize=True)
    except FileNotFoundError:
        logging.warning("Could not create thumbnail, source image not found at %s", png_path)

def _generate_snowpack_viewer_url(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Constructs the URL for the external snowpack profile visualization tool.

    Args:
        metadata (Dict[str, Any]): Dictionary of station metadata, must include
                                  'latitude', 'longitude', and 'aspect'.

    Returns:
        Optional[str]: The formatted URL string, or None if essential
                       metadata is missing.
    """
    lat = metadata.get('latitude')
    lon = metadata.get('longitude')
    aspect = metadata.get('aspect')
    season = datetime.now().year
    season  = season if datetime.now().month >= 10 else season - 1

    if not all([lat, lon, aspect]):
        logging.warning("Missing lat, lon, or aspect in metadata; cannot generate viewer URL.")
        return None

    aspect_map = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west', 'Flat': 'flat'}
    aspect_word = aspect_map.get(aspect, 'flat') if isinstance(aspect, str) else 'flat'

    return f"https://nwp.mtnweather.info/snowpack/spvizll.php?lat={lat}&lon={lon}&aspect={aspect_word}&season={season}"


def _generate_html_from_template(plotly_fig_html: str, metadata: Dict[str, Any], central_date: Optional[datetime]) -> str:
    """
    Embeds a Plotly figure and metadata into a full HTML page template.

    This function creates a complete, standalone HTML file that includes the
    interactive Plotly chart and an iframe linking to an external snowpack
    visualization tool, providing a comprehensive view for a single analysis.

    Args:
        plotly_fig_html (str): The HTML string of the Plotly figure (without the
                               full page structure).
        metadata (Dict[str, Any]): The metadata dictionary for the profile, used
                                   to populate titles and links.
        central_date (Optional[datetime]): The reference date for the analysis,
                                           displayed in the title if provided.

    Returns:
        str: A string containing the full HTML page content.
    """
    station_name = metadata.get('stationName', 'N/A')
    date_str_title = f" | Analysis for {central_date.strftime('%Y-%m-%d %H:%M UTC')}" if central_date else ""
    date_str_header = f"Analysis for: {central_date.strftime('%Y-%m-%d %H:%M UTC')}" if central_date else ""
    snowpack_viewer_link = _generate_snowpack_viewer_url(metadata)

    snowpack_viz_html = ""
    if snowpack_viewer_link:
        snowpack_viz_html = f'''
        <h2>Snowpack Visualization</h2>
        <iframe class="iframe-container" src="{snowpack_viewer_link}" title="Snowpack Visualization"></iframe>
        '''
    else:
        snowpack_viz_html = "<h2>Snowpack Visualization Link Not Available</h2>"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wetting Front Analysis: {station_name}{date_str_title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #333; }}
        .container {{ max-width: 1400px; margin: 20px auto; background: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; }}
        .header-container {{ padding: 15px 25px; background-color: #343a40; color: white; border-bottom: 4px solid #007bff; }}
        .header-container h1 {{ margin: 0; font-size: 1.6em; }}
        .iframe-container {{ width: 100%; height: 700px; border: 1px solid #dee2e6; border-radius: 4px; }}
        .content-section {{ padding: 20px; }}
        h2 {{ text-align: center; color: #007bff; margin-top: 10px; margin-bottom: 20px; font-weight: 500; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header-container">
        <h1>Wetting Front Height & Layers of Concern: {station_name}</h1>
        <p style="margin:0; font-size: 1.1em; color: #ddd;">{date_str_header}</p>
    </div>
    <div class="content-section">
        {snowpack_viz_html}
    </div>
    <div class="content-section">
        <h2>Wetting Front Analysis</h2>
        {plotly_fig_html}
    </div>
</div>
</body>
</html>
"""
# --- Matplotlib Plotting Helpers ---

def _plot_lwc_colormesh(ax: Axes, lwc_data: xr.Dataset) -> Optional[QuadMesh]:
    """
    Prepares Liquid Water Content (LWC) data and plots it as a colormesh.

    This function transforms the LWC data from the xarray Dataset into a format
    suitable for Matplotlib's `pcolormesh`, creating a visual representation
    of water content throughout the snowpack over time.

    Args:
        ax (Axes): The Matplotlib axes object on which to plot.
        lwc_data (xr.Dataset): An xarray Dataset containing 'lwc' and 'height'
                               variables with 'timestamp' and 'layer_index'
                               coordinates.

    Returns:
        Optional[QuadMesh]: The Matplotlib QuadMesh object that was plotted,
                            or None if the data was unsuitable for plotting.
    """
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
    
    df_heights = pd.DataFrame(height_values_raw, dtype=np.float64)
    df_heights_filled = df_heights.ffill(axis=0).bfill(axis=0)
    
    if df_heights_filled.isnull().values.any():
        logging.warning("Skipping LWC colormesh due to persistent NaN height values.")
        return None

    height_values = df_heights_filled.to_numpy()
    lwc_masked = np.ma.masked_where(df_heights.isnull().to_numpy(), lwc_values)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", "blue", "orange", "red"])
    norm = mcolors.Normalize(vmin=0, vmax=500)
    
    return ax.pcolormesh(X, height_values, lwc_masked, cmap=cmap, norm=norm, shading="gouraud", zorder=1)

def _clip_colormesh_to_wet_area(ax: Axes, c: QuadMesh, df: pd.DataFrame) -> None:
    """
    Creates a clipping path to show LWC only within the detected wet layer.

    This function uses the 'highest_wet_point' and 'wet_front_lwc_height'
    series to define the upper and lower boundaries of the wet snow region,
    then applies this path as a clip to the LWC colormesh.

    Args:
        ax (Axes): The Matplotlib axes object containing the plot.
        c (QuadMesh): The colormesh object to be clipped.
        df (pd.DataFrame): The summary DataFrame containing the boundary data.
    """
    df_resampled = df.asfreq('h')
    x_dense: np.ndarray = mdates.date2num(df_resampled.index)
    
    # Get the clipping boundaries by interpolating only the specific numeric series
    cy_series = df_resampled['highest_wet_point'].interpolate(method='linear').bfill().ffill()
    sy_series = df_resampled['wet_front_lwc_height'].interpolate(method='linear').bfill().ffill()

    # Guard clause: If there's no valid data in the clipping series, do not attempt to clip.
    if cy_series.isnull().all() or sy_series.isnull().all():
        logging.info("Cannot clip colormesh, not enough valid boundary data.")
        return

    cy: np.ndarray = cy_series.to_numpy(dtype=float)
    sy: np.ndarray = sy_series.to_numpy(dtype=float)    
    
    verts = np.concatenate([np.column_stack([x_dense, cy]), np.column_stack([x_dense[::-1], sy[::-1]])])
    path = MplPath(verts)
    patch = PathPatch(path, transform=ax.transData, facecolor='none', ec='none')
    c.set_clip_path(patch)
    ax.add_patch(patch)

def _plot_line_series(ax: Axes, df: pd.DataFrame, central_date: Optional[datetime]) -> None:
    """
    Plots the primary data series (HS, LOC, Wet Front) on the given axes.

    Args:
        ax (Axes): The Matplotlib axes object on which to plot.
        df (pd.DataFrame): The summary DataFrame containing the data series.
        central_date (Optional[datetime]): The reference date for the analysis,
                                            plotted as a vertical line.
    """
    if 'hs' in df.columns:
        ax.plot(df.index, df['hs'], label='Total Snow Depth (HS)', color='navy', marker='.', linestyle='-', zorder=10)
    if 'weak_layer_height' in df.columns:
        ax.plot(df.index, df['weak_layer_height'], label='Weak Layer Height (LOC)', color='black', zorder=10)
    
    if 'wet_front_lwc_height' in df.columns:
        wet_front_series = df['wet_front_lwc_height']
        is_valid = wet_front_series.notna()
        starts = wet_front_series.index[is_valid & ~is_valid.shift(1, fill_value=False).astype(bool)]
        ends = wet_front_series.index[is_valid & ~is_valid.shift(-1, fill_value=False).astype(bool)]
        for i, (start, end) in enumerate(zip(starts, ends)):
            segment = wet_front_series.loc[start:end]
            label = 'Deepest Wet Front (LWC > 3%)' if i == 0 else ""
            ax.plot(segment.index.to_numpy(), 
                    np.asarray(segment, dtype=float), 
                    color='red', 
                    zorder=10, 
                    label=label)
    
    if central_date:
        ax.axvline(x=float(mdates.date2num(central_date)), 
                   color='purple', 
                   linestyle='--', 
                   linewidth=2, 
                   label='Central Date', 
                   zorder=11)
        ax.text(float(mdates.date2num(central_date)), 
                ax.get_ylim()[0], central_date.strftime('%Y-%m-%d'),
                rotation=90, 
                verticalalignment='bottom', 
                color='purple', fontsize=10)

def _configure_plot_aesthetics(fig: Figure, 
                               ax: Axes, 
                               metadata: Dict[str, Any], 
                               c: Optional[QuadMesh], 
                               central_date: Optional[datetime]
) -> None:
    """
    Configures plot titles, labels, grid, legend, and axes formatting.

    Args:
        fig (Figure): The Matplotlib Figure object.
        ax (Axes): The Matplotlib Axes object.
        metadata (Dict[str, Any]): Metadata for populating the plot title.
        c (Optional[QuadMesh]): The colormesh object, used to add a colorbar.
        central_date (Optional[datetime]): The central analysis date.
    """
    location = (metadata.get("latitude"), metadata.get('longitude'))
    elevation = metadata.get("altitude")
    aspect = metadata.get("slopeAzi", "N/A")
    date_info = f"Analysis for: {central_date.strftime('%Y-%m-%d %H:%M UTC')}" if central_date else ""
    
    title = (f"Wetting Front Tracking: {metadata.get('stationName', 'N/A')}\n"
             f"Loc: {location}, Elev: {elevation}m, Aspect: {aspect}\n"
             f"{date_info}")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a colorbar if the colormesh was plotted
    if c:
        cbar = fig.colorbar(c, ax=ax, label='Liquid Water Content (%)', extend='max')
        cbar.set_ticks([0, 100, 200, 300, 400, 500])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5+'])

    handles, labels = ax.get_legend_handles_labels()
    order = ['Total Snow Depth (HS)', 'Deepest Wet Front (LWC > 3%)', 'Weak Layer Height (LOC)', 'Central Date']
    ax.legend([handles[labels.index(key)] for key in order if key in labels],
              [key for key in order if key in labels])

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    fig.autofmt_xdate()
    plt.tight_layout()

# --- Main Plotting Functions ---

def plot_summary_matplotlib(df: pd.DataFrame, file_stem: str, 
                            metadata: dict[str, Any], 
                            lwc_plot_data: xr.Dataset | None = None, 
                            central_date: datetime | None = None) -> None:
    """ 
    Generates and saves a static PNG plot of the snowpack analysis.

    This function orchestrates the entire Matplotlib plotting process by calling
    the various helper functions to plot the LWC colormesh, line series, and
    configure the final plot aesthetics before saving it to a file.

    Args:
        df (pd.DataFrame): The summary DataFrame with daily analysis results.
        file_stem (str): A unique identifier used to name the output file.
        metadata (Dict[str, Any]): Metadata about the snowpack profile.
        lwc_plot_data (Optional[xr.Dataset]): The full-resolution LWC data for
                                              the colormesh background.
        central_date (Optional[datetime]): The reference date for the analysis,
                                            shown as a vertical line on the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    c = None
    # Guard clause for LWC data
    if lwc_plot_data is not None and 'lwc' in lwc_plot_data and not lwc_plot_data['lwc'].isnull().to_numpy().all():
        c = _plot_lwc_colormesh(ax, lwc_plot_data)
        if c:
            _clip_colormesh_to_wet_area(ax, c, df)

    _plot_line_series(ax, df, central_date)
    _configure_plot_aesthetics(fig, ax, metadata, c, central_date)
    
    plt.savefig(get_png_path(file_stem), dpi=300)
    plt.close(fig)
        
def plot_summary_plotly(df: pd.DataFrame, 
                        file_stem: str, 
                        metadata: dict[str, Any], 
                        central_date: Optional[datetime] = None
) -> None:
    """
    Generates an interactive HTML page containing a Plotly plot of the analysis.

    This function creates a Plotly figure with similar data as the Matplotlib
    version but with interactive features like zoom, pan, and hover tooltips.
    The resulting figure is then embedded into a full HTML page using a template.

    Args:
        df (pd.DataFrame): The summary DataFrame with daily analysis results.
        file_stem (str): A unique identifier used to name the output file.
        metadata (Dict[str, Any]): Metadata about the snowpack profile.
        central_date (Optional[datetime]): The central analysis date for the title.
    """
    fig = go.Figure()

    # --- Filled Area for Wet Layer ---
    if 'wet_front_lwc_height' in df.columns and 'highest_wet_point' in df.columns:
        # Plot in segments to avoid connecting across NaN gaps
        is_valid = df['wet_front_lwc_height'].notna() & df['highest_wet_point'].notna()
        starts = df.index[is_valid & ~is_valid.shift(1, fill_value=False)]
        ends = df.index[is_valid & ~is_valid.shift(-1, fill_value=False)]
        
        for start, end in zip(starts, ends):
            segment = df.loc[start:end]
            if len(segment) > 1:
                x_coords = segment.index.tolist() + segment.index.tolist()[::-1]
                y_coords = segment['highest_wet_point'].tolist() + segment['wet_front_lwc_height'].tolist()[::-1]
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor='rgba(0, 200, 200, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # --- Line and Marker Plots ---
    # Add a dummy trace for the wet area legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='rgba(0, 200, 200, 0.4)', size=10),
                             name='Wet Layer Extent'))

    if 'hs' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['hs'], name='Total Snow Depth (HS)', mode='lines+markers', line=dict(color='darkblue')))
    if 'weak_layer_height' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['weak_layer_height'], name='Weak Layer Height (LOC)', mode='lines', line=dict(color='black', width=2)))
    if 'wet_front_lwc_height' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['wet_front_lwc_height'], name='Deepest Wet Front (LWC > 3%)', mode='lines', line=dict(color='red', width=2), connectgaps=False)) # connectgaps=False
    
    date_str = f" | Analysis for {central_date.strftime('%Y-%m-%d %H:%M UTC')}" if central_date else ""
    plotly_title = f"Wetting Front Analysis for {metadata.get('stationName', 'N/A')}{date_str}"
    fig.update_layout(
        title=plotly_title,
        xaxis_title='Date',
        yaxis_title='Height (cm)',
        legend_title_text='Metrics',
        template='plotly_white',
    )
    if 'hs' in df and df['hs'].notna().any():
        fig.update_yaxes(range=[0, df['hs'].max() * 1.1])
    
    full_html = _generate_html_from_template(fig.to_html(full_html=False, include_plotlyjs='cdn'), metadata, central_date)
    
    with open(get_html_path(file_stem), 'w') as f:
        f.write(full_html)


def create_folium_map(final_gdf: gpd.GeoDataFrame, map_output_path: Path, central_date: datetime) -> None:
    """
    Creates a Folium summary map with polygons colored by risk level.

    This function takes the final GeoDataFrame, which contains all analysis
    results, and generates an interactive HTML map. Polygons are styled based
    on the 'time_to_loc' value. The map includes tooltips that show a thumbnail
    of the static plot and popups that link to the full interactive plot.

    Args:
        final_gdf (gpd.GeoDataFrame): The final GeoDataFrame containing polygon
                                      geometries and all associated analysis results.
        map_output_path (Path): The path to save the final summary_map.html file.
        central_date (datetime): The central analysis date for the map title.
    """
    if final_gdf.empty:
        logging.warning("GeoDataFrame is empty. Cannot create map.")
        return
        
    final_gdf['geometry'] = final_gdf.geometry.buffer(0)
    
    # Ensure CRS is projected for accurate area calculation
    gdf_proj = final_gdf.to_crs("EPSG:3857")
    final_gdf['area_sq_meters'] = gdf_proj.geometry.area

    def get_color(time_to_loc):
        if pd.isna(time_to_loc): 
            return 'gray'
        time = float(time_to_loc)
        if time < -48:
            return 'green'
        elif -48 <= time < -24:
            return 'yellow'
        elif -24 <= time <= 0:
            return 'purple'
        elif 0 < time <= 24: 
            return 'red'
        elif 24 < time <= 72: 
            return 'orange'
        else: 
            return 'gray'

    def get_tooltip_html(row):
        if pd.isna(row['file_stem']): 
            return ""
        png_path = get_png_path(row['file_stem'])
        thumb_path = png_path.parent / f"{png_path.stem}_thumb.png"
        _create_thumbnail(png_path, thumb_path)
        image_path = f"{ASSETS_SUBFOLDER_NAME}/{thumb_path.name}"
        date_str = f"Analysis Date: {row['central_date_str']}<br>" if pd.notna(row['central_date_str']) else ""
        return (f"<b>{row['pathName']}</b><br>"
                f"{date_str}"
                f"Aspect: {row['aspect']}<br>"
                f'<img src="{image_path}" width="800">')

    def get_popup_html(row):
        if pd.isna(row['file_stem']): 
            return ""
        html_path = get_html_path(row['file_stem'])
        link_path = f"{ASSETS_SUBFOLDER_NAME}/{html_path.name}"
        date_str = f"Analysis Date: {row['central_date_str']}<br>" if pd.notna(row['central_date_str']) else ""
        return (f"<b>{row['station_name']}</b><br>"
                f"{date_str}"
                f'<a href="{link_path}" target="_blank">Open Interactive Plot</a>')
    
    final_gdf['color'] = final_gdf['time_to_loc'].apply(get_color)
    final_gdf['tooltip'] = final_gdf.apply(get_tooltip_html, axis=1)
    final_gdf['popup'] = final_gdf.apply(get_popup_html, axis=1)
    
    map_data_path = RESULTS_PATH / "map_data.geojson"
    final_gdf.to_file(map_data_path, driver='GeoJSON')

    map_center = final_gdf.to_crs("EPSG:4269").unary_union.centroid
    m = folium.Map(location=[map_center.y, map_center.x] if map_center else [40, -105], zoom_start=8)

    folium.TileLayer('OpenTopoMap', name='Topographic').add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Satellite').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street View').add_to(m)

    def style_function(x): 
        return {"fillColor": x['properties']['color'], "color": "black", "weight": 1, "fillOpacity": 0.6}
    
    gjson = GeoJson(
        str(map_data_path.resolve()),
        style_function=style_function,
        name='Avalanche Path Risk',
        tooltip=GeoJsonTooltip(fields=['tooltip'], aliases=[''], localize=True, sticky=True),
        popup=GeoJsonPopup(fields=['popup'], aliases=[''], localize=True)
    )
    gjson.add_to(m)
    
    date_str = central_date.strftime('%Y-%m-%d %H:%M') if central_date else ""
    title_html = f'<h3 align="center" style="font-size:16px"><b>Wetting Front Analysis | {date_str}</b></h3>'
    legend_html = """
        <b>Expected  Wetting Front Time at LOC</b><br>
        <i style="background:red"></i> 0 to 24h<br>
        <i style="background:orange"></i> 24 to 72h<br>
        <i style="background:purple"></i> -24 to 0h (Recent)<br>
        <i style="background:yellow"></i> -48 to -24h (Past)<br>
        <i style="background:green"></i> &gt; 48h ago (Past)<br>
        <i style="background:gray"></i> Other / No Data
    """

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

    folium.LayerControl().add_to(m)    
    # --- Save map ---
    m.save(str(map_output_path))
    logging.info(f"Summary map saved to: {map_output_path}")
