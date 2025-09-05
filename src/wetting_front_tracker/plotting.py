# In src/wetting_front_tracker/plotting.py

import base64
import json
import folium
from folium import GeoJson, GeoJsonTooltip, GeoJsonPopup 
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')   # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from PIL import Image
from typing import Any

# Import our new config functions
from .param_config import get_png_path, get_html_path

def _create_thumbnail(png_path: Path, thumb_path: Path, max_size: tuple = (400, 267)):
    """Creates a smaller, web-optimized thumbnail from a larger PNG."""
    if thumb_path.exists():
        return # Don't recreate if it already exists
    try:
        with Image.open(png_path) as img:
            img.thumbnail(max_size)
            img.save(thumb_path, "PNG", optimize=True)
    except FileNotFoundError:
        logging.warning("Could not create thumbnail, source image not found at %s", png_path)

def plot_summary_matplotlib(df: pd.DataFrame, file_stem: str, metadata: dict[str, Any]):
    """Generates a static plot using Matplotlib and saves it as a PNG file."""
    fig, ax = plt.subplots(figsize=(14, 8))

    if 'hs' in df.columns:
        ax.plot(df.index, df['hs'], label='Total Snow Depth (HS)', color='blue', marker='o', linewidth=2)
    if 'weak_layer_height' in df.columns:
        ax.plot(df.index, df['weak_layer_height'], label='Weak Layer Height (LOC)', color='black')
    if 'wet_front_lwc_height' in df.columns:
        ax.plot(df.index, df['wet_front_lwc_height'], label='Deepest Wet Front (LWC > 3%)', color='red')

    if 'wet_front_lwc_height' in df.columns and 'highest_wet_point' in df.columns:
        ax.fill_between(
            df.index,
            df['wet_front_lwc_height'],
            df['highest_wet_point'],
            where=df['wet_front_lwc_height'].notna().tolist(),
            color='cyan', alpha=0.7, interpolate=True, label='Wet Layer Extent'
        )

    location = (metadata.get("latitude"), metadata.get('longitude'))
    elevation = metadata.get("altitude")
    aspect = "Flat" if metadata.get("slopeAngle") == "0.00" else metadata.get("slopeAzi")
    title = f"Wetting Front Tracking\nLocation: {location}, Elevation: {elevation}m, Aspect: {aspect}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    order = ['Total Snow Depth (HS)', 'Wet Layer Extent', 'Deepest Wet Front (LWC > 3%)', 'Weak Layer Height (LOC)']
    ax.legend([handles[labels.index(key)] for key in order if key in labels],
              [key for key in order if key in labels])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()

    plt.tight_layout()
    output_filename = get_png_path(file_stem)
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)


def plot_summary_plotly(df: pd.DataFrame, file_stem: str, metadata: dict[str, Any]):
    """Generates an interactive plot using Plotly and saves it as an HTML file."""
    fig = go.Figure()

    if 'wet_front_lwc_height' in df.columns and 'highest_wet_point' in df.columns:
        # This logic creates the shaded area for wet layer extent
        valid_data = df['wet_front_lwc_height'].notna()
        starts = df.index[valid_data & ~valid_data.shift(1).fillna(False)]
        ends = df.index[valid_data & ~valid_data.shift(-1).fillna(False)]

        for start_date, end_date in zip(starts, ends):
            block_df = df.loc[start_date:end_date]
            
            if len(block_df) > 1:
                fig.add_trace(go.Scatter(
                    x=block_df.index.tolist() + block_df.index.tolist()[::-1],
                    y=block_df['highest_wet_point'].tolist() + block_df['wet_front_lwc_height'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(0, 200, 200, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=False
                ))
            elif len(block_df) == 1:
                fig.add_trace(go.Scatter(
                    x=[block_df.index[0], block_df.index[0]],
                    y=[block_df['wet_front_lwc_height'].iloc[0], block_df['highest_wet_point'].iloc[0]],
                    mode='lines', line=dict(color='rgba(0, 200, 200, 0.5)', width=4),
                    hoverinfo='skip', showlegend=False
                ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='rgba(0, 200, 200, 0.4)', size=10),
        name='Wet Layer Extent'
    ))

    if 'hs' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['hs'], name='Total Snow Depth (HS)', mode='lines+markers', line=dict(color='blue')))
    if 'weak_layer_height' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['weak_layer_height'], name='Weak Layer Height (LOC)', mode='lines', line=dict(color='black')))
    if 'wet_front_lwc_height' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['wet_front_lwc_height'], name='Deepest Wet Front (LWC > 3%)', mode='lines', line=dict(color='red')))

    location = (metadata.get("latitude"), metadata.get('longitude'))
    elevation = metadata.get("altitude")
    aspect = "Flat" if metadata.get("slopeAngle") == "0.00" else metadata.get("slopeAzi")
    
    # Create the base title string, exactly like in Matplotlib
    base_title = f"Wetting Front Tracking\nLocation: {location}, Elevation: {elevation}m, Aspect: {aspect}"
    
    # Convert the newline character to an HTML break tag for Plotly
    plotly_title = base_title.replace('\n', '<br>')

    fig.update_layout(
        title=plotly_title,
        xaxis_title='Date',
        yaxis_title='Height (cm)',
        legend_title_text='Metrics',
        template='plotly_white'
    )
    output_filename = get_html_path(file_stem)
    fig.write_html(output_filename)

def _load_and_clean_geojson(geojson_path: Path) -> dict | None:
    """Loads and filters a GeoJSON file to remove features with null geometries."""
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        if 'features' in data and isinstance(data['features'], list):
            # Keep only features that have a valid, non-null geometry
            valid_features = [
                feature for feature in data['features'] if feature.get('geometry')
            ]
            data['features'] = valid_features
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error("Failed to read or parse GeoJSON file at %s: %s", geojson_path, e)
        return None
    
def create_folium_map(results_list: list, map_output_path: Path, geojson_path: Path | None = None):
    """
    Creates a Folium map where aspect polygons are colored by risk level.

    Instead of circle markers, this function styles the GeoJSON polygons
    directly based on the analysis results from their linked .pro files.
    Tooltips show a plot image, and popups link to an interactive plot.

    Args:
        results_list (list): A list of dictionaries, one for each analyzed .pro file.
        map_output_path (Path): The path to save the final HTML map file.
        geojson_path (Path | None): Path to the GeoJSON file containing polygons
                                    linked to .pro file paths.
    """
    if not geojson_path or not geojson_path.exists():
        logging.error("Linked GeoJSON file not found. Cannot create map.")
        return

    polygons_gdf = gpd.read_file(geojson_path)
    if polygons_gdf.empty:
        logging.warning("GeoJSON is empty. Cannot create map.")
        return
        
    results_lookup = {res['pro_file_path']: res for res in results_list}

    # --- NEW: Helper functions for vectorized operations ---
    def get_tooltip_html(pro_path):
        result = results_lookup.get(pro_path)
        if not result: 
            return ""
        
        file_stem = result['file_stem']
        png_path = get_png_path(file_stem)
        thumb_path = png_path.parent / f"{png_path.stem}_thumb.png"
        _create_thumbnail(png_path, thumb_path)
        
        if thumb_path.exists():
            encoded = base64.b64encode(open(thumb_path, 'rb').read()).decode()
            return f'<img src="data:image/png;base64,{encoded}" width="400">'
        return ""

    def get_popup_html(pro_path):
        result = results_lookup.get(pro_path)
        if not result: 
            return ""

        file_stem = result['file_stem']
        html_path = get_html_path(file_stem)
        if html_path.exists():
            return (f"<b>{result['station_name']}</b><br>"
                    f'<a href="./{html_path.name}" target="_blank">Open Interactive Plot</a>')
        return ""

    # --- Use .map() to create the new columns efficiently ---
    polygons_gdf['tooltip_html'] = polygons_gdf['pro_file_path'].map(get_tooltip_html)
    polygons_gdf['popup_html'] = polygons_gdf['pro_file_path'].map(get_popup_html)

    map_center = polygons_gdf.to_crs("EPSG:4326").union_all().centroid
    m = folium.Map(location=[map_center.y, map_center.x] if map_center else [40, -105], zoom_start=8)

    folium.TileLayer('OpenTopoMap', name='Topographic').add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Satellite').add_to(m)

    def style_function(feature):
        pro_path = feature['properties']['pro_file_path']
        result = results_lookup.get(pro_path)
        
        color = "gray"
        if result:
            time_to_loc = result.get('time_to_loc')
            if time_to_loc is None or time_to_loc > 72 or time_to_loc < 0:
                color = 'gray'
            elif time_to_loc <= 24:
                color = 'red'
            else:
                color = 'orange'
        
        return {"fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.6}

    gjson = GeoJson(
        polygons_gdf,
        style_function=style_function,
        name='Avalanche Path Risk',
        tooltip=GeoJsonTooltip(fields=['tooltip_html'], aliases=['']),
        popup=GeoJsonPopup(fields=['popup_html'], aliases=[''])
    )

    gjson.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(map_output_path))
    logging.info("Summary map saved to: %s", map_output_path)