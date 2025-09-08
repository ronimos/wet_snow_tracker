import base64
import json
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
import plotly.graph_objects as go
from PIL import Image
from typing import Any

from .param_config import get_png_path, get_html_path, RESULTS_PATH

def _create_thumbnail(png_path: Path, thumb_path: Path, max_size: tuple = (800, 534)):
    """Creates a smaller, web-optimized thumbnail from a larger PNG."""
    if thumb_path.exists():
        return
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
        valid_data = df['wet_front_lwc_height'].notna()
        
        # --- FIX for FutureWarning: Use nullable boolean type to avoid downcasting ---
        # Convert to nullable boolean, which uses pd.NA instead of np.nan
        valid_data_nullable = valid_data.astype('boolean')
        
        # Now, .shift() introduces pd.NA, and .fillna() works without downcasting
        shifted_starts = valid_data_nullable.shift(1).fillna(False)
        shifted_ends = valid_data_nullable.shift(-1).fillna(False)

        starts = df.index[valid_data & ~shifted_starts]
        ends = df.index[valid_data & ~shifted_ends]
        
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
    base_title = f"Wetting Front Tracking\nLocation: {location}, Elevation: {elevation}m, Aspect: {aspect}"
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


def create_folium_map(results_list: list, map_output_path: Path, geojson_path: Path | None = None):
    """
    Creates a fast-loading Folium map by externalizing the GeoJSON data and using
    relative links for tooltips.
    """
    if not geojson_path or not geojson_path.exists():
        logging.error("Linked GeoJSON file not found. Cannot create map.")
        return

    polygons_gdf = gpd.read_file(geojson_path)
    if polygons_gdf.empty:
        logging.warning("GeoJSON is empty. Cannot create map.")
        return
        
    polygons_gdf['geometry'] = polygons_gdf.geometry.buffer(0)
    
    results_df = pd.DataFrame(results_list)
    merged_gdf = polygons_gdf.merge(results_df, on='pro_file_path', how='left')

    # --- VECTORIZED PROPERTY CREATION ---
    def get_color(time_to_loc):
        if pd.isna(time_to_loc) or time_to_loc > 72 or time_to_loc < 0: return 'gray'
        elif time_to_loc <= 24: return 'red'
        else: 
            return 'orange'

    def get_tooltip_html(row):
        """Builds an HTML string with polygon info and a plot thumbnail."""
        # Start with the text information
        info_html = (
            f"<b>Path Name:</b> {row.get('pathName', 'N/A')}<br>"
            f"<b>Aspect:</b> {row.get('aspect', 'N/A')}<br>"
        )
        
        # Add the plot image if a corresponding result exists
        if pd.notna(row['file_stem']):
            png_path = get_png_path(row['file_stem'])
            thumb_path = png_path.parent / f"{png_path.stem}_thumb.png"
            _create_thumbnail(png_path, thumb_path)
            # Use a relative path for fast loading
            info_html += f'<br><img src="{thumb_path.name}" width="400">'
            
        return info_html
    
    def get_popup_html(row):
        if pd.isna(row['file_stem']): return ""
        html_path = get_html_path(row['file_stem'])
        return (f"<b>{row['station_name']}</b><br>"
                f'<a href="{html_path.name}" target="_blank">Open Interactive Plot</a>')

    merged_gdf['color'] = merged_gdf['time_to_loc'].apply(get_color)
    merged_gdf['tooltip'] = merged_gdf.apply(get_tooltip_html, axis=1)
    merged_gdf['popup'] = merged_gdf.apply(get_popup_html, axis=1)
    
    # --- EXTERNALIZE GEOJSON ---
    map_data_path = RESULTS_PATH / "map_data.geojson"
    merged_gdf.to_file(map_data_path, driver='GeoJSON')
    logging.info(f"Map data saved to {map_data_path}")

    # --- CREATE MAP ---
    map_center = merged_gdf.to_crs("EPSG:4269").unary_union.centroid
    m = folium.Map(location=[map_center.y, map_center.x] if map_center else [40, -105], zoom_start=8)

    folium.TileLayer('OpenTopoMap', name='Topographic').add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Satellite').add_to(m)

    def style_function(x): 
        return {
        "fillColor": x['properties']['color'],
        "color": "black", "weight": 1, "fillOpacity": 0.6
    }
    
    gjson = GeoJson(
        str(map_data_path.resolve()),
        style_function=style_function,
        name='Avalanche Path Risk',
        tooltip=GeoJsonTooltip(fields=['tooltip'], aliases=[''], localize=True, sticky=False),
        popup=GeoJsonPopup(fields=['popup'], aliases=[''], localize=True)
    )

    gjson.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(map_output_path))
    logging.info(f"Summary map saved to: {map_output_path}. It will load data from map_data.geojson.")

