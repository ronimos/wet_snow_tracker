"""
plotting.py

This module contains functions for creating all visual outputs for the Wetting
Front Tracker application. It handles the generation of static Matplotlib plots,
interactive Plotly plots embedded in full HTML pages, and the final Folium
summary map.
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
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from PIL import Image
from typing import Any

from .param_config import get_png_path, get_html_path, RESULTS_PATH 

def _create_thumbnail(png_path: Path, thumb_path: Path, max_size: tuple = (800, 534)):
    """ Creates a smaller, web-optimized PNG thumbnail from a larger image. """
    try:
        with Image.open(png_path) as img:
            img.thumbnail(max_size)
            img.save(thumb_path, "PNG", optimize=True)
    except FileNotFoundError:
        logging.warning("Could not create thumbnail, source image not found at %s", png_path)

def _generate_snowpack_viewer_url(metadata: dict[str, Any]) -> str | None:
    """
    Constructs the URL for the snowpack visualization based on metadata.

    Args:
        metadata: Dictionary of metadata for the station, must include 'latitude',
                  'longitude', and 'aspect'.

    Returns:
        The formatted URL string, or None if essential metadata is missing.
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


def _generate_html_from_template(plotly_fig_html: str, metadata: dict[str, Any]) -> str:
    """
    Embeds a Plotly figure and metadata into a full HTML page template that
    mirrors the reference layout.
    """
    station_name = metadata.get('stationName', 'N/A')
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
        <title>Wetting Front Analysis: {station_name}</title>
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

def plot_summary_matplotlib(df: pd.DataFrame, file_stem: str, metadata: dict[str, Any], lwc_data: xr.Dataset, central_date: datetime | None = None):
    """ Generates a static PNG plot of the snowpack analysis. """
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- LWC Colormesh background ---
    if lwc_data is not None and 'lwc' in lwc_data and 'height' in lwc_data:
        try:
            # Use to_numpy() for safe conversion from xarray to numpy
            lwc_values = lwc_data['lwc'].to_numpy() * 100
            height_values = lwc_data['height'].to_numpy()
            time_values = lwc_data['timestamp'].to_numpy()
        except AttributeError:
            # Fallback for older xarray or if it's already numpy
            lwc_values = lwc_data['lwc'].values * 100
            height_values = lwc_data['height'].values
            time_values = lwc_data['timestamp'].values

        # Filter out timestamps that have no layers at all to avoid errors
        valid_time_indices = ~np.all(np.isnan(height_values), axis=1)
        if np.any(valid_time_indices):
            lwc_values = lwc_values[valid_time_indices]
            height_values = height_values[valid_time_indices]
            time_values = time_values[valid_time_indices]

            # --- FIX for pcolormesh non-finite values error ---
            # pcolormesh requires finite coordinates. We fill NaNs in the height
            # array (Y coordinates) and mask the corresponding LWC values (C).
            original_height_nan_mask = np.isnan(height_values)
            
            # Use pandas to ffill/bfill along the layer axis (axis=1)
            df_heights = pd.DataFrame(height_values, dtype=np.float64)
            df_heights_filled = df_heights.ffill(axis=1).bfill(axis=1)
            
            # Proceed only if filling was successful (no all-NaN rows remain)
            if not df_heights_filled.isnull().values.any():
                height_values_filled = df_heights_filled.to_numpy()

                x_coords = mdates.date2num(time_values)
                X = np.tile(x_coords, (lwc_values.shape[1], 1)).T

                # Create a final mask for LWC where either LWC or original height was NaN
                combined_mask = original_height_nan_mask | np.isnan(lwc_values)
                lwc_masked = np.ma.masked_where(combined_mask, lwc_values)

                # Custom colormap: white -> blue -> red -> black. Normalize from 0 to 6+
                cmap = mcolors.LinearSegmentedColormap.from_list("custom_lwc", ["white", "blue", "orange", "red"])
                norm = mcolors.Normalize(vmin=0, vmax=500)

                # Plot the colormesh using the filled heights and masked LWC data
                c = ax.pcolormesh(X, height_values_filled, lwc_masked, cmap=cmap, norm=norm, shading="gouraud", zorder=1)
                
                # --- Clip colormesh to region between curves ---
                clip_df = df[['wet_front_lwc_height', 'highest_wet_point']].dropna()
                if not clip_df.empty and len(clip_df) > 1:
                    x_clip = mdates.date2num(clip_df.index)
                    cy = clip_df['highest_wet_point'].values
                    sy = clip_df['wet_front_lwc_height'].values
                    
                    verts = np.concatenate([np.column_stack([x_clip, cy]), np.column_stack([x_clip[::-1], sy[::-1]])])
                    path = MplPath(verts)
                    patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
                    ax.add_patch(patch)
                    c.set_clip_path(patch)
            else:
                 logging.warning(f"Skipping LWC colormesh for {file_stem} due to persistent NaN height values.")

    if 'hs' in df.columns:
        ax.plot(df.index, df['hs'], label='Total Snow Depth (HS)', color='darkblue', marker='.', linestyle='-', zorder=2)
    if 'weak_layer_height' in df.columns:
        ax.plot(df.index, df['weak_layer_height'], label='Weak Layer Height (LOC)', color='black', linewidth=2, zorder=2)
    if 'wet_front_lwc_height' in df.columns:
        ax.plot(df.index, df['wet_front_lwc_height'], label='Deepest Wet Front (LWC > 3%)', color='red', linewidth=2, zorder=2)

    if 'wet_front_lwc_height' in df.columns:
        # Plot in segments to avoid connecting across NaN gaps
        wet_front_series = df['wet_front_lwc_height']
        is_valid = wet_front_series.notna()
        # Find start and end points of continuous data segments
        starts = df.index[is_valid & ~is_valid.shift(1, fill_value=False)]
        ends = df.index[is_valid & ~is_valid.shift(-1, fill_value=False)]
        
        # Plot each segment individually
        for start, end in zip(starts, ends):
            segment = wet_front_series.loc[start:end]
            ax.plot(segment.index, segment.values, color='red', linewidth=2, zorder=2)
        
        # Add a dummy plot with a label for the legend
        ax.plot([], [], color='red', linewidth=2, label='Deepest Wet Front (LWC > 3%)')
    
    if central_date:
        ax.axvline(x=central_date, color='purple', linestyle='--', linewidth=2, label='Central Date', zorder=2)
        ax.text(central_date, plt.ylim()[0], central_date.strftime('%Y-%m-%d'),
                rotation=90, verticalalignment='bottom', color='purple', fontsize=10)

    location = (metadata.get("latitude"), metadata.get('longitude'))
    elevation = metadata.get("altitude")
    aspect = metadata.get("slopeAzi", "N/A")
    title = f"Wetting Front Tracking\nLocation: {location}, Elevation: {elevation}m, Aspect: {aspect}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a colorbar if the colormesh was plotted
    if 'c' in locals():
        cbar = fig.colorbar(c, ax=ax, label="Liquid Water Content (%)", extend='max')
        cbar.set_ticks([0, 100, 200, 300, 400, 500])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5+'])

    handles, labels = ax.get_legend_handles_labels()
    order = ['Total Snow Depth (HS)', 'Deepest Wet Front (LWC > 3%)', 'Weak Layer Height (LOC)', 'Central Date']
    ax.legend([handles[labels.index(key)] for key in order if key in labels],
              [key for key in order if key in labels])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(get_png_path(file_stem), dpi=300)
    plt.close(fig)   
    
def plot_summary_plotly(df: pd.DataFrame, file_stem: str, metadata: dict[str, Any]):
    """ Generates an interactive HTML page containing a Plotly plot. """
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
    
    plotly_title = f"Wetting Front Analysis for {metadata.get('stationName', 'N/A')}"
    fig.update_layout(
        title=plotly_title,
        xaxis_title='Date',
        yaxis_title='Height (cm)',
        legend_title_text='Metrics',
        template='plotly_white',
    )
    if 'hs' in df and df['hs'].notna().any():
        fig.update_yaxes(range=[0, df['hs'].max() * 1.1])
    
    full_html = _generate_html_from_template(fig.to_html(full_html=False, include_plotlyjs='cdn'), metadata)
    
    with open(get_html_path(file_stem), 'w') as f:
        f.write(full_html)


def create_folium_map(final_gdf: gpd.GeoDataFrame, map_output_path: Path):
    """ Creates a Folium map with polygons colored by risk and detailed tooltips. """
    if final_gdf.empty:
        logging.warning("GeoDataFrame is empty. Cannot create map.")
        return
        
    final_gdf['geometry'] = final_gdf.geometry.buffer(0)
    
    # Ensure CRS is projected for accurate area calculation
    gdf_proj = final_gdf.to_crs("EPSG:3857")
    final_gdf['area_sq_meters'] = gdf_proj.geometry.area

    def get_color(time_to_loc):
        if pd.isna(time_to_loc): return 'gray'
        time = float(time_to_loc)
        if -24 <= time <= 0: return 'purple'
        elif 0 < time <= 24: return 'red'
        elif 24 < time <= 72: return 'orange'
        else: return 'gray'

    def get_tooltip_html(row):
        if pd.isna(row['file_stem']): return ""
        png_path = get_png_path(row['file_stem'])
        thumb_path = png_path.parent / f"{png_path.stem}_thumb.png"
        _create_thumbnail(png_path, thumb_path)
        area_str = f"{row['area_sq_meters']:,.0f} m²"
        return (f"<b>{row['pathName']}</b><br>"
                f"Aspect: {row['aspect']}<br>"
                f"Area: {area_str}<br>"
                f'<img src="{thumb_path.name}" width="400">')

    def get_popup_html(row):
        if pd.isna(row['file_stem']): return ""
        html_path = get_html_path(row['file_stem'])
        return (f"<b>{row['station_name']}</b><br>"
                f'<a href="{html_path.name}" target="_blank">Open Interactive Plot</a>')

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
    folium.LayerControl().add_to(m)
    m.save(str(map_output_path))
    logging.info(f"Summary map saved to: {map_output_path}")

