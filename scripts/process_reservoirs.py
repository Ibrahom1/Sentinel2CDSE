import os
import json
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj
from dotenv import load_dotenv
from scipy import ndimage
import shutil

# Load local .env file if it exists
load_dotenv()

# Constants
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
NDWI_THRESHOLD = 0.05  # Standard threshold: water > 0.05
LOOKBACK_DAYS = 15
MIN_RESERVOIR_PIXELS = 500  # Minimum cluster size to be considered a reservoir (~0.05 km²)


# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# STAC Search
# ──────────────────────────────────────────────

def search_sentinel_scene(bbox):
    """Searches for the newest, clearest Sentinel-2 L2A scene over the bbox."""
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

    valid_scenes = []
    for feat in features:
        props = feat.get("properties", {})
        cloud_cover = props.get("eo:cloud_cover", 100.0)
        dt_str = props.get("datetime")
        valid_scenes.append((feat, cloud_cover, dt_str))

    # Prefer scenes under 25% cloud, sorted by newest first
    clear_scenes = [s for s in valid_scenes if s[1] < 25.0]
    if clear_scenes:
        clear_scenes.sort(key=lambda x: x[2], reverse=True)
        selected = clear_scenes[0][0]
    else:
        valid_scenes.sort(key=lambda x: x[1])
        selected = valid_scenes[0][0]

    props = selected.get("properties", {})
    print(f"Selected Scene: {selected.get('id')}")
    print(f"  Acquisition Date: {props.get('datetime')}")
    print(f"  Scene Cloud Cover: {props.get('eo:cloud_cover')}%")

    return selected


# ──────────────────────────────────────────────
# File download & tile stacking
# ──────────────────────────────────────────────

def download_file(url, output_path, token):
    """Downloads a file from CDSE using the OAuth token."""
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def stack_scene_if_missing(scene, token):
    """Downloads B02, B03, B04, B08 for a scene and stacks them into a single GeoTIFF."""
    scene_id = scene.get("id")
    output_dir = Path("temp_stack")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{scene_id}.tif"

    if output_path.exists():
        print(f"Full-tile stacked TIFF for {scene_id} already exists in cache.")
        return output_path

    assets = scene.get("assets", {})
    b02_key = "B02_10m" if "B02_10m" in assets else "B02"
    b03_key = "B03_10m" if "B03_10m" in assets else "B03"
    b04_key = "B04_10m" if "B04_10m" in assets else "B04"
    b08_key = "B08_10m" if "B08_10m" in assets else "B08"

    if any(k not in assets for k in [b02_key, b03_key, b04_key, b08_key]):
        print("Error: Missing 10m bands (B02, B03, B04, B08) in scene assets.")
        return None

    temp_dir = Path("temp_bands")
    temp_dir.mkdir(exist_ok=True)

    band_paths = {
        "B02": temp_dir / f"{scene_id}_B02.jp2",
        "B03": temp_dir / f"{scene_id}_B03.jp2",
        "B04": temp_dir / f"{scene_id}_B04.jp2",
        "B08": temp_dir / f"{scene_id}_B08.jp2",
    }

    try:
        print(f"Downloading bands for full-tile stack of {scene_id}...")
        for band_name, band_key in [("Blue B02", b02_key), ("Green B03", b03_key),
                                     ("Red B04", b04_key), ("NIR B08", b08_key)]:
            print(f"  Downloading {band_name}...")
            download_file(assets[band_key]["alternate"]["https"]["href"],
                          band_paths[band_key.split("_")[0]], token)

        print("Creating compressed 4-band GeoTIFF (deflate)...")
        with rasterio.open(band_paths["B02"]) as src:
            meta = src.meta.copy()

        meta.update(count=4, driver='GTiff', compress='deflate', predictor=2, zlevel=6)

        with rasterio.open(output_path, 'w', **meta) as dst:
            for idx, bname in enumerate(["B02", "B03", "B04", "B08"], start=1):
                print(f"  Writing band {idx}/4 to TIFF...")
                with rasterio.open(band_paths[bname]) as src_band:
                    dst.write(src_band.read(1), idx)

        print(f"Saved stacked TIFF to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating stacked TIFF for {scene_id}: {e}")
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return None
    finally:
        for p in band_paths.values():
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            try:
                temp_dir.rmdir()
            except Exception:
                pass


# ──────────────────────────────────────────────
# Band reading & cropping
# ──────────────────────────────────────────────

def read_cropped_band(file_path, aoi_geometry, band_idx=1,
                      out_shape=None, out_transform=None, out_crs=None):
    """Opens a local raster, crops a specific band to the AOI geometry."""
    with rasterio.open(file_path) as src:
        project = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        aoi_utm = transform(project, aoi_geometry)

        cropped_image, cropped_transform = mask(src, [aoi_utm], crop=True)
        band_data = cropped_image[band_idx - 1].astype(np.float32)

        if out_shape is not None and band_data.shape != out_shape:
            reprojected_band = np.zeros(out_shape, dtype=band_data.dtype)
            reproject(
                band_data, reprojected_band,
                src_transform=cropped_transform, src_crs=src.crs,
                dst_transform=out_transform, dst_crs=out_crs,
                resampling=Resampling.nearest
            )
            return reprojected_band, out_transform, out_crs

        return band_data, cropped_transform, src.crs


# ──────────────────────────────────────────────
# Reservoir detection (largest connected water cluster)
# ──────────────────────────────────────────────

def detect_reservoir_cluster(water_mask):
    """
    Uses connected-component labeling to find the largest contiguous
    water body (the reservoir) among all detected water pixels.
    Returns:
        reservoir_mask  – boolean array, True only for the reservoir cluster
        cluster_label   – integer label of the reservoir cluster
        num_clusters    – total number of water clusters found
    """
    # Label connected components (8-connectivity)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_array, num_clusters = ndimage.label(water_mask.astype(np.int32), structure=structure)

    if num_clusters == 0:
        return np.zeros_like(water_mask, dtype=bool), 0, 0

    # Find the largest cluster
    cluster_sizes = ndimage.sum(water_mask, labeled_array, range(1, num_clusters + 1))
    largest_label = np.argmax(cluster_sizes) + 1  # labels are 1-indexed
    largest_size = int(cluster_sizes[largest_label - 1])

    print(f"  Connected-component analysis: {num_clusters} water clusters found")
    print(f"  Largest cluster: label={largest_label}, pixels={largest_size} ({largest_size * 100 / 1e6:.4f} km²)")

    if largest_size < MIN_RESERVOIR_PIXELS:
        print(f"  WARNING: Largest cluster ({largest_size} px) is below minimum threshold ({MIN_RESERVOIR_PIXELS} px)")

    reservoir_mask = (labeled_array == largest_label)
    return reservoir_mask, largest_label, num_clusters


def compute_tight_bbox(reservoir_mask, raster_transform, raster_crs):
    """
    Computes the tight bounding box around the reservoir pixels
    in both pixel coordinates and geographic (WGS84) coordinates.
    Returns:
        bbox_geom_wgs84  – shapely box in EPSG:4326
        bbox_geom_utm    – shapely box in the raster CRS
        row_slice, col_slice – numpy slices for the tight region
    """
    rows, cols = np.where(reservoir_mask)
    if len(rows) == 0:
        return None, None, None, None

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Add a small buffer (20 pixels = 200m at 10m resolution)
    buffer_px = 20
    min_row = max(0, min_row - buffer_px)
    max_row = min(reservoir_mask.shape[0] - 1, max_row + buffer_px)
    min_col = max(0, min_col - buffer_px)
    max_col = min(reservoir_mask.shape[1] - 1, max_col + buffer_px)

    # Convert pixel corners to map coordinates using the affine transform
    # Top-left corner of min_row, min_col
    x_min, y_max = raster_transform * (min_col, min_row)
    # Bottom-right corner of max_row, max_col
    x_max, y_min = raster_transform * (max_col + 1, max_row + 1)

    bbox_utm = box(x_min, y_min, x_max, y_max)

    # Project to WGS84
    project_back = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True).transform
    bbox_wgs84 = transform(project_back, bbox_utm)

    row_slice = slice(min_row, max_row + 1)
    col_slice = slice(min_col, max_col + 1)

    return bbox_wgs84, bbox_utm, row_slice, col_slice


# ──────────────────────────────────────────────
# Vector export
# ──────────────────────────────────────────────

def save_reservoir_vector(reservoir_mask, raster_transform, raster_crs, output_path):
    """Polygonizes the reservoir mask and saves as a dissolved GeoJSON vector."""
    mask_shapes = shapes(reservoir_mask.astype(np.uint8), mask=reservoir_mask, transform=raster_transform)

    geoms = []
    for geom, val in mask_shapes:
        if val == 1:
            geom_shape = shape(geom)
            project_back = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True).transform
            geom_wgs = transform(project_back, geom_shape)
            geoms.append(geom_wgs)

    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        gdf = gdf.dissolve()
        gdf["feature_type"] = "reservoir_water"
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"  Saved reservoir vector to {output_path}")
    else:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_path, driver="GeoJSON")
        print(f"  No reservoir detected. Saved empty vector to {output_path}")


def save_bbox_vector(bbox_geom, output_path):
    """Saves the tight bounding box as a GeoJSON."""
    if bbox_geom is None:
        return
    gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
    gdf["feature_type"] = "reservoir_bbox"
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"  Saved tight bbox vector to {output_path}")


def save_all_water_geojson(water_mask, raster_transform, raster_crs, output_path):
    """Polygonizes the full water mask (all water bodies) and saves as GeoJSON."""
    mask_shapes = shapes(water_mask.astype(np.uint8), mask=water_mask, transform=raster_transform)

    geoms = []
    for geom, val in mask_shapes:
        if val == 1:
            geom_shape = shape(geom)
            project_back = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True).transform
            geom_wgs = transform(project_back, geom_shape)
            geoms.append(geom_wgs)

    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        gdf = gdf.dissolve()
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"  Saved all-water GeoJSON to {output_path}")
    else:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_path, driver="GeoJSON")
        print(f"  No water detected. Saved empty GeoJSON to {output_path}")


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def save_visualization(ndwi, water_mask, reservoir_mask, tight_bbox_slice,
                       name, date, reservoir_area, total_area, cloud_cover, output_path):
    """Generates a 3-panel plot: NDWI | All Water | Reservoir with tight bbox."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: NDWI
    im1 = axes[0].imshow(ndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    axes[0].set_title(f"NDWI Map\n(Threshold > {NDWI_THRESHOLD})")
    fig.colorbar(im1, ax=axes[0], label="NDWI", shrink=0.8)
    axes[0].axis('off')

    # Panel 2: All water bodies
    axes[1].imshow(water_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"All Water Bodies\nTotal: {total_area:.2f} km²")
    axes[1].axis('off')

    # Panel 3: Reservoir only with tight bbox
    # Show NDWI in background, overlay reservoir in blue
    axes[2].imshow(ndwi, cmap="gray_r", vmin=-0.3, vmax=0.3, alpha=0.4)
    reservoir_display = np.ma.masked_where(~reservoir_mask, reservoir_mask.astype(float))
    axes[2].imshow(reservoir_display, cmap="Blues", vmin=0, vmax=1, alpha=0.8)

    # Draw tight bounding box rectangle
    if tight_bbox_slice is not None:
        row_sl, col_sl = tight_bbox_slice
        rect = mpatches.Rectangle(
            (col_sl.start, row_sl.start),
            col_sl.stop - col_sl.start,
            row_sl.stop - row_sl.start,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        axes[2].add_patch(rect)

    axes[2].set_title(f"Detected Reservoir\nArea: {reservoir_area:.2f} km²")
    axes[2].axis('off')

    plt.suptitle(
        f"{name.upper()} Reservoir — {date}  |  AOI Clouds: {cloud_cover:.1f}%",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ──────────────────────────────────────────────
# Core reservoir processing
# ──────────────────────────────────────────────

def process_reservoir(name, geojson_path, token):
    """Full pipeline for one reservoir: download → detect → bbox → area → export."""
    print(f"\n{'='*50}\nProcessing Reservoir: {name.upper()}\n{'='*50}")

    # 1. Load AOI
    gdf = gpd.read_file(geojson_path)
    aoi_geometry = gdf.geometry.values[0]
    bbox = list(gdf.total_bounds)

    # 2. Search for scene
    scene = search_sentinel_scene(bbox)
    if not scene:
        print(f"No Sentinel-2 scenes found in the last {LOOKBACK_DAYS} days for {name}.")
        return None

    scene_id = scene.get("id")
    scene_props = scene.get("properties", {})
    scene_cloud_cover = scene_props.get("eo:cloud_cover", 0.0)
    acquisition_date = scene_props.get("datetime")[:10]

    # 3. Stack full tile
    stacked_tiff_path = stack_scene_if_missing(scene, token)
    if not stacked_tiff_path:
        print(f"Error: Could not stack TIFF for scene {scene_id}.")
        return None

    # Copy full tile to raw_tiles for artifact
    raw_tiles_dir = Path("raw_tiles")
    raw_tiles_dir.mkdir(exist_ok=True)
    reservoir_tiff_path = raw_tiles_dir / f"{name}_{acquisition_date}_full.tif"
    shutil.copy(stacked_tiff_path, reservoir_tiff_path)
    print(f"  Saved full tile TIFF to {reservoir_tiff_path}")

    # 4. Download SCL
    assets = scene.get("assets", {})
    scl_key = "SCL_20m" if "SCL_20m" in assets else ("SCL_60m" if "SCL_60m" in assets else "SCL")
    if scl_key not in assets:
        print("Error: Missing SCL band in scene assets.")
        return None
    scl_href = assets[scl_key]["alternate"]["https"]["href"]

    temp_dir = Path("temp_bands")
    temp_dir.mkdir(exist_ok=True)
    scl_path = temp_dir / f"{name}_SCL.jp2"

    try:
        # 5. Read bands cropped to broad AOI
        print("Cropping Green band (B03) from stacked TIFF...")
        b03_data, b03_transform, b03_crs = read_cropped_band(
            reservoir_tiff_path, aoi_geometry, band_idx=2)

        print("Cropping NIR band (B08) from stacked TIFF...")
        b08_data, _, _ = read_cropped_band(
            reservoir_tiff_path, aoi_geometry, band_idx=4,
            out_shape=b03_data.shape, out_transform=b03_transform, out_crs=b03_crs)

        print("Downloading SCL band locally...")
        download_file(scl_href, scl_path, token)
        print("Reprojecting SCL band to 10m...")
        scl_data, _, _ = read_cropped_band(
            scl_path, aoi_geometry, band_idx=1,
            out_shape=b03_data.shape, out_transform=b03_transform, out_crs=b03_crs)
    finally:
        if scl_path.exists():
            try:
                scl_path.unlink()
            except Exception:
                pass
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            try:
                temp_dir.rmdir()
            except Exception:
                pass

    # 6. Calculate NDWI
    denom = b03_data + b08_data
    ndwi = np.where(denom == 0, 0, (b03_data - b08_data) / denom)

    # 7. Cloud masking
    cloud_classes = [0, 1, 3, 8, 9, 10]
    cloud_mask = np.isin(scl_data, cloud_classes)
    nodata_mask = (b03_data == 0) | (b08_data == 0)
    valid_pixels = ~(cloud_mask | nodata_mask)

    total_aoi_pixels = np.sum(~nodata_mask)
    cloudy_aoi_pixels = np.sum(cloud_mask & ~nodata_mask)
    aoi_cloud_cover_pct = (cloudy_aoi_pixels / total_aoi_pixels * 100) if total_aoi_pixels > 0 else 100.0

    # 8. Full water mask (all water in AOI)
    water_mask = (ndwi > NDWI_THRESHOLD) & valid_pixels
    total_water_pixels = int(np.sum(water_mask))
    total_water_area_km2 = total_water_pixels * 100.0 / 1_000_000.0

    print(f"  Total water in AOI: {total_water_pixels} pixels = {total_water_area_km2:.4f} km²")

    # 9. Detect the reservoir (largest connected water cluster)
    print("  Running reservoir detection (connected-component analysis)...")
    reservoir_mask, cluster_label, num_clusters = detect_reservoir_cluster(water_mask)

    reservoir_pixels = int(np.sum(reservoir_mask))
    reservoir_area_km2 = reservoir_pixels * 100.0 / 1_000_000.0

    # 10. Compute tight bounding box around the reservoir
    bbox_wgs84, bbox_utm, row_slice, col_slice = compute_tight_bbox(
        reservoir_mask, b03_transform, b03_crs)

    if bbox_wgs84 is not None:
        bounds = bbox_wgs84.bounds
        print(f"  Tight bbox (WGS84): W={bounds[0]:.4f} S={bounds[1]:.4f} E={bounds[2]:.4f} N={bounds[3]:.4f}")

    print(f"\nResults for {name.upper()}:")
    print(f"  Reservoir Water Area: {reservoir_area_km2:.4f} km²")
    print(f"  Total Water in AOI:   {total_water_area_km2:.4f} km²")
    print(f"  Water Clusters Found: {num_clusters}")
    print(f"  AOI Cloud Cover:      {aoi_cloud_cover_pct:.2f}%")

    # 11. Save outputs
    Path("outputs/maps").mkdir(parents=True, exist_ok=True)
    Path("outputs/vectors").mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    # Reservoir water boundary vector (dissolved polygon)
    save_reservoir_vector(
        reservoir_mask, b03_transform, b03_crs,
        f"outputs/vectors/{name}_{acquisition_date}_reservoir.geojson")

    # Tight bounding box vector
    save_bbox_vector(
        bbox_wgs84,
        f"outputs/vectors/{name}_{acquisition_date}_bbox.geojson")

    # All water bodies in AOI (for reference)
    save_all_water_geojson(
        water_mask, b03_transform, b03_crs,
        f"outputs/maps/{name}_{acquisition_date}_water.geojson")

    # Visualization
    tight_bbox_slice = (row_slice, col_slice) if row_slice is not None else None
    save_visualization(
        ndwi, water_mask, reservoir_mask, tight_bbox_slice,
        name, acquisition_date, reservoir_area_km2, total_water_area_km2,
        aoi_cloud_cover_pct,
        f"outputs/plots/{name}_{acquisition_date}.png")

    return {
        "date": acquisition_date,
        "reservoir": name,
        "scene_id": scene_id,
        "cloud_cover_scene": scene_cloud_cover,
        "cloud_cover_aoi": round(aoi_cloud_cover_pct, 2),
        "reservoir_area_km2": round(reservoir_area_km2, 4),
        "total_water_area_km2": round(total_water_area_km2, 4),
        "water_clusters": num_clusters
    }


# ──────────────────────────────────────────────
# History tracking
# ──────────────────────────────────────────────

def update_history_csv(new_records):
    """Appends new records and recalculates change metrics."""
    csv_path = Path("outputs/water_history.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "date", "reservoir", "scene_id", "cloud_cover_scene",
            "cloud_cover_aoi", "reservoir_area_km2", "total_water_area_km2",
            "water_clusters", "change_km2", "change_percent"
        ])

    new_df = pd.DataFrame(new_records)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=["reservoir", "date"]).reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset=["date", "reservoir"], keep="last")

    updated = []
    for reservoir, group in combined_df.groupby("reservoir"):
        group = group.sort_values("date")
        group["change_km2"] = group["reservoir_area_km2"].diff().fillna(0.0)
        prev = group["reservoir_area_km2"].shift(1)
        group["change_percent"] = ((group["change_km2"] / prev) * 100.0).fillna(0.0)
        updated.append(group)

    final_df = pd.concat(updated).sort_values(by=["date", "reservoir"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_path, index=False)
    print(f"\nUpdated water history CSV with {len(final_df)} records.")
    return final_df


def plot_history_trends(df):
    """Generates a historical trend plot for all reservoirs."""
    if df.empty:
        return

    plt.figure(figsize=(10, 6))

    for reservoir, group in df.groupby("reservoir"):
        dates = pd.to_datetime(group["date"])
        plt.plot(dates, group["reservoir_area_km2"], marker='o', linewidth=2, label=reservoir.upper())

    plt.title("Reservoir Water Surface Area Trends", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Reservoir Water Area (km²)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = Path("outputs/plots/history.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved trend history plot to {plot_path}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    print("Starting reservoir monitoring processing pipeline...")

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        print("\n" + "!" * 60)
        print("WARNING: Copernicus credentials are not set.")
        print("Please set CDSE_USERNAME and CDSE_PASSWORD variables.")
        print("Running in DRY-RUN mode: Syntax check passed.")
        print("!" * 60 + "\n")
        return

    try:
        token = get_auth_token()
    except Exception as e:
        print(f"Error authenticating: {e}")
        return

    reservoirs = {
        "bhakra": "aoi/bhakra.geojson",
        "thein": "aoi/thein.geojson",
        "pong": "aoi/pong.geojson"
    }

    new_records = []
    for name, geojson_path in reservoirs.items():
        if not os.path.exists(geojson_path):
            print(f"GeoJSON for {name} not found at {geojson_path}. Skipping.")
            continue

        try:
            record = process_reservoir(name, geojson_path, token)
            if record:
                new_records.append(record)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    if new_records:
        history_df = update_history_csv(new_records)
        plot_history_trends(history_df)
    else:
        print("No scenes processed successfully.")

    # Clean up temp_stack
    temp_stack_dir = Path("temp_stack")
    if temp_stack_dir.exists():
        try:
            shutil.rmtree(temp_stack_dir)
            print("Cleaned up temporary stacking cache.")
        except Exception as e:
            print(f"Warning: Could not remove temp_stack: {e}")


if __name__ == "__main__":
    main()
