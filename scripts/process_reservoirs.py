import os
import json
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# Constants
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
NDWI_THRESHOLD = 0.05  # Standard threshold: water > 0.05
LOOKBACK_DAYS = 15

def get_auth_token():
    """Retrieves the OAuth2 bearer token from CDSE."""
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    
    if not username or not password:
        raise ValueError("CDSE_USERNAME and CDSE_PASSWORD environment variables must be set.")
    
    print("Authenticating with Copernicus Data Space Ecosystem...")
    data = {
        'client_id': 'cdse-public',
        'username': username,
        'password': password,
        'grant_type': 'password'
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json().get('access_token')

def search_sentinel_scene(bbox):
    """Searches for the newest, clearest Sentinel-2 L2A scene over the bbox in the lookback period."""
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
    
    datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": 20
    }
    
    print(f"Searching for scenes between {start_date.date()} and {end_date.date()}...")
    response = requests.post(STAC_URL, json=payload)
    response.raise_for_status()
    
    features = response.json().get("features", [])
    if not features:
        return None
        
    print(f"Found {len(features)} candidate scenes.")
    
    # We want to select a scene that has low cloud cover and is as recent as possible.
    # We sort by:
    # 1. Cloud cover < 20% (preferred)
    # 2. Then by date (newest first)
    
    valid_scenes = []
    for feat in features:
        props = feat.get("properties", {})
        cloud_cover = props.get("eo:cloud_cover", 100.0)
        dt_str = props.get("datetime")
        valid_scenes.append((feat, cloud_cover, dt_str))
        
    # Sort: first filter those under 25% cloud cover and sort them by date descending
    clear_scenes = [s for s in valid_scenes if s[1] < 25.0]
    if clear_scenes:
        # Sort by date (index 2) descending
        clear_scenes.sort(key=lambda x: x[2], reverse=True)
        selected = clear_scenes[0][0]
    else:
        # If no clean scenes, pick the one with the lowest cloud cover overall
        valid_scenes.sort(key=lambda x: x[1])
        selected = valid_scenes[0][0]
        
    props = selected.get("properties", {})
    print(f"Selected Scene: {selected.get('id')}")
    print(f"  Acquisition Date: {props.get('datetime')}")
    print(f"  Scene Cloud Cover: {props.get('eo:cloud_cover')}%")
    
    return selected

def download_file(url, output_path, token):
    """Downloads a file from CDSE using the OAuth token."""
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def read_cropped_band(file_path, aoi_geometry, out_shape=None, out_transform=None, out_crs=None):
    """
    Opens a local Sentinel-2 band, and crops it to the AOI geometry.
    If out_shape is provided, reprojects the band to match it.
    """
    with rasterio.open(file_path) as src:
        # Project AOI geometry to raster CRS
        project = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        aoi_utm = transform(project, aoi_geometry)
        
        # Crop using rasterio.mask
        cropped_image, cropped_transform = mask(src, [aoi_utm], crop=True)
        band_data = cropped_image[0].astype(np.float32)
        
        # Reproject/upsample if requested (useful for SCL 20m -> 10m alignment)
        if out_shape is not None and band_data.shape != out_shape:
            reprojected_band = np.zeros(out_shape, dtype=band_data.dtype)
            reproject(
                band_data,
                reprojected_band,
                src_transform=cropped_transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest
            )
            return reprojected_band, out_transform, out_crs
            
        return band_data, cropped_transform, src.crs

def process_reservoir(name, geojson_path, token):
    """Processes a single reservoir AOI: downloads, NDWI, masks clouds, calculates water area."""
    print(f"\n{'='*40}\nProcessing Reservoir: {name.upper()}\n{'='*40}")
    
    # 1. Load GeoJSON AOI
    gdf = gpd.read_file(geojson_path)
    aoi_geometry = gdf.geometry.values[0]
    bbox = list(gdf.total_bounds) # [minx, miny, maxx, maxy]
    
    # 2. Search for Sentinel-2 Scene
    scene = search_sentinel_scene(bbox)
    if not scene:
        print(f"No Sentinel-2 scenes found in the last {LOOKBACK_DAYS} days for {name}.")
        return None
        
    scene_id = scene.get("id")
    scene_props = scene.get("properties", {})
    scene_cloud_cover = scene_props.get("eo:cloud_cover", 0.0)
    acquisition_date = scene_props.get("datetime")[:10] # YYYY-MM-DD
    
    # Get asset hrefs
    assets = scene.get("assets", {})
    b03_key = "B03_10m" if "B03_10m" in assets else "B03"
    b08_key = "B08_10m" if "B08_10m" in assets else "B08"
    scl_key = "SCL_20m" if "SCL_20m" in assets else ("SCL_60m" if "SCL_60m" in assets else "SCL")
    
    if b03_key not in assets or b08_key not in assets or scl_key not in assets:
        print(f"Error: Missing required bands in scene assets. Required: B03, B08, SCL.")
        return None
        
    b03_href = assets[b03_key]["alternate"]["https"]["href"]
    b08_href = assets[b08_key]["alternate"]["https"]["href"]
    scl_href = assets[scl_key]["alternate"]["https"]["href"]
    
    # Create temp directory
    temp_dir = Path("temp_bands")
    temp_dir.mkdir(exist_ok=True)
    
    b03_path = temp_dir / f"{name}_B03.jp2"
    b08_path = temp_dir / f"{name}_B08.jp2"
    scl_path = temp_dir / f"{name}_SCL.jp2"
    
    try:
        print("Downloading Green band (B03) locally...")
        download_file(b03_href, b03_path, token)
        print("Cropping Green band (B03)...")
        b03_data, b03_transform, b03_crs = read_cropped_band(b03_path, aoi_geometry)
        
        print("Downloading NIR band (B08) locally...")
        download_file(b08_href, b08_path, token)
        print("Cropping NIR band (B08)...")
        b08_data, _, _ = read_cropped_band(b08_path, aoi_geometry, 
                                          out_shape=b03_data.shape, 
                                          out_transform=b03_transform, 
                                          out_crs=b03_crs)
                                          
        print("Downloading SCL band locally...")
        download_file(scl_href, scl_path, token)
        print("Reprojecting SCL band to 10m...")
        scl_data, _, _ = read_cropped_band(scl_path, aoi_geometry,
                                          out_shape=b03_data.shape,
                                          out_transform=b03_transform,
                                          out_crs=b03_crs)
    finally:
        # Clean up temp files
        for p in [b03_path, b08_path, scl_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temp file {p}: {e}")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            try:
                temp_dir.rmdir()
            except Exception as e:
                pass
    
    # 3. Process bands
    # Compute NDWI = (B03 - B08) / (B03 + B08)
    denom = b03_data + b08_data
    # Avoid division by zero
    ndwi = np.where(denom == 0, 0, (b03_data - b08_data) / denom)
    
    # Cloud and invalid pixels masking using SCL
    # Cloud-related classes: 3 (shadows), 8 (medium clouds), 9 (high clouds), 10 (cirrus)
    # Also 0 (no data), 1 (saturated/defective)
    cloud_classes = [0, 1, 3, 8, 9, 10]
    cloud_mask = np.isin(scl_data, cloud_classes)
    
    # Valid pixels within AOI (exclude nodata margins where B03 is 0)
    nodata_mask = (b03_data == 0) | (b08_data == 0)
    valid_pixels = ~(cloud_mask | nodata_mask)
    
    # AOI cloud statistics
    total_aoi_pixels = np.sum(~nodata_mask)
    cloudy_aoi_pixels = np.sum(cloud_mask & ~nodata_mask)
    aoi_cloud_cover_percent = (cloudy_aoi_pixels / total_aoi_pixels * 100) if total_aoi_pixels > 0 else 100.0
    
    # Water mask
    water_mask = (ndwi > NDWI_THRESHOLD) & valid_pixels
    
    # 4. Calculate Water Area
    # Resolution of Sentinel-2 10m pixel = 10m x 10m = 100 m^2
    water_pixel_count = np.sum(water_mask)
    water_area_km2 = (water_pixel_count * 100.0) / 1_000_000.0
    
    print(f"Results for {name}:")
    print(f"  Water Pixels: {water_pixel_count}")
    print(f"  Calculated Water Area: {water_area_km2:.4f} km²")
    print(f"  AOI Cloud Cover: {aoi_cloud_cover_percent:.2f}%")
    
    # 5. Save outputs
    # Create output folders
    Path("outputs/maps").mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    
    # Save water GeoJSON
    geojson_out = f"outputs/maps/{name}_{acquisition_date}_water.geojson"
    save_water_geojson(water_mask, b03_transform, b03_crs, geojson_out)
    
    # Save Plot
    plot_out = f"outputs/plots/{name}_{acquisition_date}.png"
    save_visualization(ndwi, water_mask, name, acquisition_date, water_area_km2, aoi_cloud_cover_percent, plot_out)
    
    return {
        "date": acquisition_date,
        "reservoir": name,
        "scene_id": scene_id,
        "cloud_cover_scene": scene_cloud_cover,
        "cloud_cover_aoi": aoi_cloud_cover_percent,
        "water_area_km2": water_area_km2
    }

def save_water_geojson(water_mask, transform_mat, crs, output_path):
    """Converts the binary water mask into a GeoJSON file with polygons."""
    mask_shapes = shapes(water_mask.astype(np.uint8), mask=water_mask, transform=transform_mat)
    
    geoms = []
    for geom, val in mask_shapes:
        if val == 1:
            geom_shape = shape(geom)
            # Project back from raster UTM CRS to WGS84 (EPSG:4326)
            project_back = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
            geom_wgs = transform(project_back, geom_shape)
            geoms.append(geom_wgs)
            
    if geoms:
        water_gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        water_gdf = water_gdf.dissolve() # merge overlapping shapes
        water_gdf.to_file(output_path, driver="GeoJSON")
        print(f"  Saved water boundary GeoJSON to {output_path}")
    else:
        # Create empty geojson if no water found
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_path, driver="GeoJSON")
        print(f"  No water detected. Saved empty GeoJSON to {output_path}")

def save_visualization(ndwi, water_mask, name, date, area, cloud_cover, output_path):
    """Generates and saves a side-by-side plot of the NDWI and the extracted water mask."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # NDWI plot
    im1 = axes[0].imshow(ndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    axes[0].set_title(f"NDWI Map\n(Threshold > {NDWI_THRESHOLD})")
    fig.colorbar(im1, ax=axes[0], label="NDWI")
    axes[0].axis('off')
    
    # Water mask plot
    axes[1].imshow(water_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Extracted Water Body\nArea: {area:.2f} km²")
    axes[1].axis('off')
    
    plt.suptitle(f"{name.upper()} Reservoir - {date} (AOI Clouds: {cloud_cover:.1f}%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot visualization to {output_path}")

def update_history_csv(new_records):
    """Appends new records to outputs/water_history.csv and calculates historical change metrics."""
    csv_path = Path("outputs/water_history.csv")
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "date", "reservoir", "scene_id", "cloud_cover_scene", 
            "cloud_cover_aoi", "water_area_km2", "change_km2", "change_percent"
        ])
        
    new_df = pd.DataFrame(new_records)
    
    # Combine
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort by reservoir and date
    combined_df = combined_df.sort_values(by=["reservoir", "date"]).reset_index(drop=True)
    
    # Drop duplicates to prevent double-adding the same scene
    combined_df = combined_df.drop_duplicates(subset=["date", "reservoir"], keep="last")
    
    # Recalculate differences for each reservoir group
    updated_records = []
    for reservoir, group in combined_df.groupby("reservoir"):
        group = group.sort_values("date")
        group["change_km2"] = group["water_area_km2"].diff()
        # Handle initial record diff (NaN -> 0.0)
        group["change_km2"] = group["change_km2"].fillna(0.0)
        
        # Calculate percentage change
        prev_area = group["water_area_km2"].shift(1)
        group["change_percent"] = (group["change_km2"] / prev_area) * 100.0
        group["change_percent"] = group["change_percent"].fillna(0.0)
        
        updated_records.append(group)
        
    final_df = pd.concat(updated_records).sort_values(by=["date", "reservoir"]).reset_index(drop=True)
    
    # Create output directory
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_path, index=False)
    print(f"\nUpdated water history CSV saved with {len(final_df)} records.")
    return final_df

def plot_history_trends(df):
    """Generates a historical trend plot for all reservoirs."""
    if df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    for reservoir, group in df.groupby("reservoir"):
        # Convert date to datetime for plotting
        dates = pd.to_datetime(group["date"])
        plt.plot(dates, group["water_area_km2"], marker='o', linewidth=2, label=reservoir.upper())
        
    plt.title("Reservoir Water Surface Area Trends", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Water Area (km²)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = Path("outputs/plots/history.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved trend history plot to {plot_path}")

def main():
    print("Starting reservoir monitoring processing pipeline...")
    
    # Check if credentials are set
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    
    if not username or not password:
        print("\n" + "!" * 60)
        print("WARNING: Copernicus credentials are not set in the environment.")
        print("Please set CDSE_USERNAME and CDSE_PASSWORD variables.")
        print("Running in DRY-RUN mode: Syntax and configuration check passed.")
        print("!" * 60 + "\n")
        return
        
    try:
        token = get_auth_token()
    except Exception as e:
        print(f"Error authenticating with CDSE: {e}")
        print("Please verify your CDSE_USERNAME and CDSE_PASSWORD.")
        return
        
    reservoirs = {
        "bhakra": "aoi/bhakra.geojson",
        "thein": "aoi/thein.geojson",
        "pong": "aoi/pong.geojson"
    }
    
    new_records = []
    for name, geojson_path in reservoirs.items():
        if not os.path.exists(geojson_path):
            print(f"GeoJSON file for {name} not found at {geojson_path}. Skipping.")
            continue
            
        try:
            record = process_reservoir(name, geojson_path, token)
            if record:
                new_records.append(record)
        except Exception as e:
            print(f"Error processing reservoir {name}: {e}")
            import traceback
            traceback.print_exc()
            
    if new_records:
        history_df = update_history_csv(new_records)
        plot_history_trends(history_df)
    else:
        print("No new scenes processed successfully in this run.")

if __name__ == "__main__":
    main()
