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
NDWI_THRESHOLD_DEFAULT = 0.05  # Fallback threshold when Otsu fails
MNDWI_THRESHOLD_DEFAULT = 0.0  # MNDWI threshold (water > 0)
LOOKBACK_DAYS = 30  # Extended from 15 to have more scene options
MIN_RESERVOIR_PIXELS = 500  # Minimum cluster size to be considered a reservoir (~0.05 km²)
MAX_SCENES_TO_TRY = 3  # Maximum number of scenes to try if detection fails
MIN_EXPECTED_AREA_KM2 = 1.0  # If detected area is below this, try next scene
AOI_CLOUD_COVER_SKIP = 70.0  # Skip scene if AOI cloud cover exceeds this %


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
# STAC Search (returns multiple candidates)
# ──────────────────────────────────────────────

def search_sentinel_scenes(bbox):
    """
    Searches for Sentinel-2 L2A scenes over the bbox.
    Returns a list of candidate scenes sorted by preference
    (low cloud cover, then most recent).
    """
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)

    datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": 30
    }

    print(f"Searching for scenes between {start_date.date()} and {end_date.date()}...")
    response = requests.post(STAC_URL, json=payload)
    response.raise_for_status()

    features = response.json().get("features", [])
    if not features:
        return []

    print(f"Found {len(features)} candidate scenes.")

    valid_scenes = []
    for feat in features:
        props = feat.get("properties", {})
        cloud_cover = props.get("eo:cloud_cover", 100.0)
        dt_str = props.get("datetime")
        valid_scenes.append((feat, cloud_cover, dt_str))

    # Sort by: scenes under 40% cloud first (sorted by newest), then rest by cloud cover
    clear_scenes = [s for s in valid_scenes if s[1] < 40.0]
    cloudy_scenes = [s for s in valid_scenes if s[1] >= 40.0]

    clear_scenes.sort(key=lambda x: x[2], reverse=True)  # newest first
    cloudy_scenes.sort(key=lambda x: x[1])  # least cloudy first

    sorted_scenes = clear_scenes + cloudy_scenes

    for i, (scene, cc, dt) in enumerate(sorted_scenes[:5]):
        props = scene.get("properties", {})
        print(f"  Candidate {i+1}: {scene.get('id')} | Date: {dt[:10]} | Cloud: {cc:.1f}%")

    return [s[0] for s in sorted_scenes]


# ──────────────────────────────────────────────
# File download & tile stacking (5 bands: B02, B03, B04, B08, B11)
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
    """
    Downloads B02, B03, B04, B08, B11 for a scene and stacks them
    into a single 5-band GeoTIFF.
    Band order: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR-20m)
    """
    scene_id = scene.get("id")
    output_dir = Path("temp_stack")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{scene_id}.tif"

    if output_path.exists():
        # Verify it has 5 bands; if old 4-band cached file, re-download
        try:
            with rasterio.open(output_path) as src:
                if src.count >= 5:
                    print(f"Full-tile stacked TIFF (5 bands) for {scene_id} already exists in cache.")
                    return output_path
                else:
                    print(f"Cached TIFF has {src.count} bands (need 5). Re-downloading...")
                    output_path.unlink()
        except Exception:
            output_path.unlink(missing_ok=True)

    assets = scene.get("assets", {})

    # 10m bands
    b02_key = "B02_10m" if "B02_10m" in assets else "B02"
    b03_key = "B03_10m" if "B03_10m" in assets else "B03"
    b04_key = "B04_10m" if "B04_10m" in assets else "B04"
    b08_key = "B08_10m" if "B08_10m" in assets else "B08"

    # 20m SWIR band
    b11_key = "B11_20m" if "B11_20m" in assets else "B11"

    required_keys = [b02_key, b03_key, b04_key, b08_key, b11_key]
    if any(k not in assets for k in required_keys):
        missing = [k for k in required_keys if k not in assets]
        print(f"Error: Missing bands in scene assets: {missing}")
        print(f"  Available assets: {list(assets.keys())}")
        return None

    temp_dir = Path("temp_bands")
    temp_dir.mkdir(exist_ok=True)

    band_paths = {
        "B02": temp_dir / f"{scene_id}_B02.jp2",
        "B03": temp_dir / f"{scene_id}_B03.jp2",
        "B04": temp_dir / f"{scene_id}_B04.jp2",
        "B08": temp_dir / f"{scene_id}_B08.jp2",
        "B11": temp_dir / f"{scene_id}_B11.jp2",
    }

    try:
        print(f"Downloading bands for full-tile stack of {scene_id}...")
        for band_name, band_key in [("Blue B02", b02_key), ("Green B03", b03_key),
                                     ("Red B04", b04_key), ("NIR B08", b08_key),
                                     ("SWIR B11", b11_key)]:
            print(f"  Downloading {band_name}...")
            download_file(assets[band_key]["alternate"]["https"]["href"],
                          band_paths[band_key.split("_")[0]], token)

        # Use B02 (10m) as the reference grid
        print("Creating compressed 5-band GeoTIFF (deflate)...")
        with rasterio.open(band_paths["B02"]) as src:
            meta = src.meta.copy()
            ref_shape = (src.height, src.width)
            ref_transform = src.transform
            ref_crs = src.crs

        meta.update(count=5, driver='GTiff', compress='deflate', predictor=2, zlevel=6)

        with rasterio.open(output_path, 'w', **meta) as dst:
            for idx, bname in enumerate(["B02", "B03", "B04", "B08", "B11"], start=1):
                print(f"  Writing band {idx}/5 ({bname}) to TIFF...")
                with rasterio.open(band_paths[bname]) as src_band:
                    band_data = src_band.read(1)

                    # B11 is 20m — resample to 10m if shape doesn't match
                    if band_data.shape != ref_shape:
                        print(f"    Resampling {bname} from {band_data.shape} to {ref_shape}...")
                        resampled = np.zeros(ref_shape, dtype=band_data.dtype)
                        reproject(
                            band_data, resampled,
                            src_transform=src_band.transform, src_crs=src_band.crs,
                            dst_transform=ref_transform, dst_crs=ref_crs,
                            resampling=Resampling.bilinear
                        )
                        dst.write(resampled, idx)
                    else:
                        dst.write(band_data, idx)

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
# Adaptive thresholding (Otsu's method)
# ──────────────────────────────────────────────

def otsu_threshold(data, valid_mask, n_bins=256):
    """
    Computes Otsu's optimal threshold for bimodal separation
    of water vs non-water pixels in an index image (NDWI/MNDWI).
    
    Returns:
        threshold: float — the optimal split point
        success: bool — whether a valid bimodal split was found
    """
    # Get valid data points
    valid_data = data[valid_mask]
    if len(valid_data) < 100:
        return NDWI_THRESHOLD_DEFAULT, False

    # Clip to reasonable range and create histogram
    data_min, data_max = np.percentile(valid_data, [1, 99])
    valid_data_clipped = np.clip(valid_data, data_min, data_max)

    hist, bin_edges = np.histogram(valid_data_clipped, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu's method: maximize inter-class variance
    total = hist.sum()
    if total == 0:
        return NDWI_THRESHOLD_DEFAULT, False

    sum_total = np.sum(bin_centers * hist)
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = NDWI_THRESHOLD_DEFAULT

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = bin_centers[i]

    # Sanity checks on the threshold
    # If Otsu puts the threshold outside a reasonable range, fall back
    if best_threshold < -0.3 or best_threshold > 0.5:
        print(f"    Otsu threshold ({best_threshold:.3f}) outside reasonable range, using default")
        return NDWI_THRESHOLD_DEFAULT, False

    return best_threshold, True


# ──────────────────────────────────────────────
# Combined water detection (NDWI + MNDWI + adaptive threshold)
# ──────────────────────────────────────────────

def detect_water_combined(ndwi, mndwi, valid_mask):
    """
    Multi-index water detection using NDWI and MNDWI with adaptive thresholding.
    
    Strategy:
    1. Compute adaptive thresholds for both NDWI and MNDWI using Otsu's method
    2. A pixel is water if MNDWI > threshold (MNDWI is the primary index, more robust)
    3. NDWI is used as a secondary confirmation and to catch edge cases
    4. Combined mask: pixel is water if (MNDWI detects it) OR (NDWI detects it)
    
    Returns:
        water_mask: boolean array
        ndwi_threshold: float — threshold used for NDWI
        mndwi_threshold: float — threshold used for MNDWI
        detection_method: str — description of method used
    """
    # Compute adaptive thresholds
    ndwi_thresh, ndwi_otsu_ok = otsu_threshold(ndwi, valid_mask)
    mndwi_thresh, mndwi_otsu_ok = otsu_threshold(mndwi, valid_mask)

    # Ensure MNDWI threshold is not too aggressive
    # MNDWI > 0 is generally water; allow Otsu to refine but clamp
    if mndwi_otsu_ok:
        mndwi_thresh = max(mndwi_thresh, -0.1)  # Don't go too negative
    else:
        mndwi_thresh = MNDWI_THRESHOLD_DEFAULT

    if ndwi_otsu_ok:
        ndwi_thresh = max(ndwi_thresh, -0.05)  # Don't go too negative
    else:
        ndwi_thresh = NDWI_THRESHOLD_DEFAULT

    method_parts = []

    # Primary: MNDWI (more robust against haze/atmosphere)
    mndwi_water = (mndwi > mndwi_thresh) & valid_mask
    method_parts.append(f"MNDWI>{mndwi_thresh:.3f}({'Otsu' if mndwi_otsu_ok else 'fixed'})")

    # Secondary: NDWI
    ndwi_water = (ndwi > ndwi_thresh) & valid_mask
    method_parts.append(f"NDWI>{ndwi_thresh:.3f}({'Otsu' if ndwi_otsu_ok else 'fixed'})")

    # Combined: Union of both detections
    # MNDWI is primary (catches water that NDWI misses due to haze)
    # NDWI catches some edge water that MNDWI might miss
    water_mask = mndwi_water | ndwi_water

    detection_method = " | ".join(method_parts)

    mndwi_count = int(np.sum(mndwi_water))
    ndwi_count = int(np.sum(ndwi_water))
    combined_count = int(np.sum(water_mask))
    print(f"    MNDWI water pixels: {mndwi_count}")
    print(f"    NDWI water pixels:  {ndwi_count}")
    print(f"    Combined (union):   {combined_count}")
    print(f"    Method: {detection_method}")

    return water_mask, ndwi_thresh, mndwi_thresh, detection_method


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
    """Polygonizes the reservoir mask and saves as a dissolved Shapefile."""
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
        gdf["feat_type"] = "reservoir"
        gdf.to_file(output_path)
        print(f"  Saved reservoir vector to {output_path}")
    else:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_path)
        print(f"  No reservoir detected. Saved empty vector to {output_path}")


def save_bbox_vector(bbox_geom, output_path):
    """Saves the tight bounding box as a Shapefile."""
    if bbox_geom is None:
        return
    gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
    gdf["feat_type"] = "res_bbox"
    gdf.to_file(output_path)
    print(f"  Saved tight bbox vector to {output_path}")


def save_all_water_shapefile(water_mask, raster_transform, raster_crs, output_path):
    """Polygonizes the full water mask (all water bodies) and saves as Shapefile."""
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
        gdf.to_file(output_path)
        print(f"  Saved all-water shapefile to {output_path}")
    else:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_path)
        print(f"  No water detected. Saved empty shapefile to {output_path}")


# ──────────────────────────────────────────────
# Quality scoring
# ──────────────────────────────────────────────

def compute_quality_score(aoi_cloud_pct, reservoir_area_km2, detection_method):
    """
    Computes a quality confidence score (0-100) for the detection result.
    
    Factors:
    - Cloud cover penalty (lower cloud = higher score)
    - Area plausibility (very small area = lower confidence)
    - Detection method (Otsu success = higher confidence)
    """
    # Start at 100, apply penalties
    score = 100.0

    # Cloud cover penalty: lose up to 50 points
    score -= min(50.0, aoi_cloud_pct * 0.7)

    # Area penalty: if very small area detected, reduce confidence
    if reservoir_area_km2 < 0.1:
        score -= 30.0
    elif reservoir_area_km2 < 1.0:
        score -= 15.0
    elif reservoir_area_km2 < 5.0:
        score -= 5.0

    # Method bonus: Otsu success means better separation
    if "Otsu" in detection_method:
        score += 5.0

    return max(0.0, min(100.0, score))


# ──────────────────────────────────────────────
# Visualization (updated with cloud mask overlay)
# ──────────────────────────────────────────────

def save_visualization(ndwi, mndwi, water_mask, reservoir_mask, cloud_mask,
                       tight_bbox_slice, name, date, reservoir_area, total_area,
                       cloud_cover, quality_score, detection_method, output_path):
    """Generates a multi-panel plot showing NDWI, MNDWI, and extracted water."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: MNDWI (primary detection index)
    im1 = axes[0].imshow(mndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    # Overlay cloud mask as semi-transparent gray
    if cloud_mask is not None and np.any(cloud_mask):
        cloud_overlay = np.ma.masked_where(~cloud_mask, np.ones_like(cloud_mask, dtype=float))
        axes[0].imshow(cloud_overlay, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    axes[0].set_title(f"MNDWI Map (Green-SWIR)\n(Primary Water Index)", fontsize=11)
    fig.colorbar(im1, ax=axes[0], label="MNDWI", shrink=0.8)
    axes[0].axis('off')

    # Panel 2: NDWI for comparison
    im2 = axes[1].imshow(ndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    if cloud_mask is not None and np.any(cloud_mask):
        cloud_overlay = np.ma.masked_where(~cloud_mask, np.ones_like(cloud_mask, dtype=float))
        axes[1].imshow(cloud_overlay, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"NDWI Map (Green-NIR)\n(Secondary Index)", fontsize=11)
    fig.colorbar(im2, ax=axes[1], label="NDWI", shrink=0.8)
    axes[1].axis('off')

    # Panel 3: Extracted water body
    reservoir_display = np.ma.masked_where(~reservoir_mask, reservoir_mask.astype(float))
    # Light blue background
    background = np.ones((*reservoir_mask.shape, 3)) * np.array([0.92, 0.95, 1.0])
    axes[2].imshow(background)
    axes[2].imshow(reservoir_display, cmap="Blues", vmin=0, vmax=1, alpha=0.9)

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

    axes[2].set_title(f"Extracted Water Body\nArea: {reservoir_area:.2f} km²", fontsize=11)
    axes[2].axis('off')

    # Quality indicator color
    if quality_score >= 70:
        quality_color = "green"
    elif quality_score >= 40:
        quality_color = "orange"
    else:
        quality_color = "red"

    quality_text = f"Quality: {quality_score:.0f}%"

    plt.suptitle(
        f"{name.upper()} Reservoir - {date} (AOI Clouds: {cloud_cover:.1f}%)  |  {quality_text}\n"
        f"Detection: {detection_method}",
        fontsize=13, fontweight='bold',
        color="black"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ──────────────────────────────────────────────
# Core reservoir processing (single scene attempt)
# ──────────────────────────────────────────────

def try_process_scene(name, aoi_geometry, scene, token):
    """
    Attempts to process a single scene for reservoir detection.
    Returns the result dict or None if the scene is unsuitable.
    """
    scene_id = scene.get("id")
    scene_props = scene.get("properties", {})
    scene_cloud_cover = scene_props.get("eo:cloud_cover", 0.0)
    acquisition_date = scene_props.get("datetime")[:10]

    print(f"\n  --- Trying scene: {scene_id} (Cloud: {scene_cloud_cover:.1f}%) ---")

    # 1. Stack full tile (5 bands)
    stacked_tiff_path = stack_scene_if_missing(scene, token)
    if not stacked_tiff_path:
        print(f"  Error: Could not stack TIFF for scene {scene_id}.")
        return None

    # Copy full tile to raw_tiles for artifact
    raw_tiles_dir = Path("raw_tiles")
    raw_tiles_dir.mkdir(exist_ok=True)
    reservoir_tiff_path = raw_tiles_dir / f"{name}_{acquisition_date}_full.tif"
    shutil.copy(stacked_tiff_path, reservoir_tiff_path)
    print(f"  Saved full tile TIFF to {reservoir_tiff_path}")

    # 2. Download SCL
    assets = scene.get("assets", {})
    scl_key = "SCL_20m" if "SCL_20m" in assets else ("SCL_60m" if "SCL_60m" in assets else "SCL")
    if scl_key not in assets:
        print("  Error: Missing SCL band in scene assets.")
        return None
    scl_href = assets[scl_key]["alternate"]["https"]["href"]

    temp_dir = Path("temp_bands")
    temp_dir.mkdir(exist_ok=True)
    scl_path = temp_dir / f"{name}_SCL.jp2"

    try:
        # 3. Read bands cropped to broad AOI
        print("  Cropping Green band (B03) from stacked TIFF...")
        b03_data, b03_transform, b03_crs = read_cropped_band(
            reservoir_tiff_path, aoi_geometry, band_idx=2)

        print("  Cropping NIR band (B08) from stacked TIFF...")
        b08_data, _, _ = read_cropped_band(
            reservoir_tiff_path, aoi_geometry, band_idx=4,
            out_shape=b03_data.shape, out_transform=b03_transform, out_crs=b03_crs)

        print("  Cropping SWIR band (B11) from stacked TIFF...")
        b11_data, _, _ = read_cropped_band(
            reservoir_tiff_path, aoi_geometry, band_idx=5,
            out_shape=b03_data.shape, out_transform=b03_transform, out_crs=b03_crs)

        print("  Downloading SCL band locally...")
        download_file(scl_href, scl_path, token)
        print("  Reprojecting SCL band to 10m...")
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

    # 4. Calculate water indices
    # NDWI = (Green - NIR) / (Green + NIR)
    denom_ndwi = b03_data + b08_data
    ndwi = np.where(denom_ndwi == 0, 0, (b03_data - b08_data) / denom_ndwi)

    # MNDWI = (Green - SWIR) / (Green + SWIR)  — more robust for water detection
    denom_mndwi = b03_data + b11_data
    mndwi = np.where(denom_mndwi == 0, 0, (b03_data - b11_data) / denom_mndwi)

    # 5. Enhanced cloud masking
    # SCL classes:
    #   0 = No data, 1 = Saturated/Defective, 2 = Cast Shadows (dark), 3 = Cloud Shadow
    #   4 = Vegetation, 5 = Bare Soil, 6 = Water, 7 = Unclassified
    #   8 = Cloud Medium Prob, 9 = Cloud High Prob, 10 = Thin Cirrus, 11 = Snow/Ice
    cloud_shadow_classes = [0, 1, 3, 8, 9, 10]  # Bad pixels / clouds / cirrus
    cloud_mask = np.isin(scl_data, cloud_shadow_classes)
    nodata_mask = (b03_data == 0) | (b08_data == 0) | (b11_data == 0)
    valid_pixels = ~(cloud_mask | nodata_mask)

    total_aoi_pixels = np.sum(~nodata_mask)
    cloudy_aoi_pixels = np.sum(cloud_mask & ~nodata_mask)
    aoi_cloud_cover_pct = (cloudy_aoi_pixels / total_aoi_pixels * 100) if total_aoi_pixels > 0 else 100.0

    print(f"  AOI cloud cover: {aoi_cloud_cover_pct:.1f}%")

    # Skip scene if too cloudy
    if aoi_cloud_cover_pct > AOI_CLOUD_COVER_SKIP:
        print(f"  SKIPPING: AOI cloud cover ({aoi_cloud_cover_pct:.1f}%) exceeds {AOI_CLOUD_COVER_SKIP}% limit")
        return None

    # 6. Combined multi-index water detection
    print("  Running combined water detection (NDWI + MNDWI + adaptive thresholds)...")
    water_mask, ndwi_thresh, mndwi_thresh, detection_method = detect_water_combined(
        ndwi, mndwi, valid_pixels)

    total_water_pixels = int(np.sum(water_mask))
    total_water_area_km2 = total_water_pixels * 100.0 / 1_000_000.0

    print(f"  Total water in AOI: {total_water_pixels} pixels = {total_water_area_km2:.4f} km²")

    # 7. Detect the reservoir (largest connected water cluster)
    print("  Running reservoir detection (connected-component analysis)...")
    reservoir_mask, cluster_label, num_clusters = detect_reservoir_cluster(water_mask)

    reservoir_pixels = int(np.sum(reservoir_mask))
    reservoir_area_km2 = reservoir_pixels * 100.0 / 1_000_000.0

    # 8. Quality score
    quality_score = compute_quality_score(aoi_cloud_cover_pct, reservoir_area_km2, detection_method)

    # 9. Compute tight bounding box around the reservoir
    bbox_wgs84, bbox_utm, row_slice, col_slice = compute_tight_bbox(
        reservoir_mask, b03_transform, b03_crs)

    if bbox_wgs84 is not None:
        bounds = bbox_wgs84.bounds
        print(f"  Tight bbox (WGS84): W={bounds[0]:.4f} S={bounds[1]:.4f} E={bounds[2]:.4f} N={bounds[3]:.4f}")

    print(f"\n  Results for {name.upper()} (scene {scene_id}):")
    print(f"    Reservoir Water Area: {reservoir_area_km2:.4f} km²")
    print(f"    Total Water in AOI:   {total_water_area_km2:.4f} km²")
    print(f"    Water Clusters Found: {num_clusters}")
    print(f"    AOI Cloud Cover:      {aoi_cloud_cover_pct:.2f}%")
    print(f"    Quality Score:        {quality_score:.0f}%")
    print(f"    Detection Method:     {detection_method}")

    return {
        "scene_id": scene_id,
        "acquisition_date": acquisition_date,
        "scene_cloud_cover": scene_cloud_cover,
        "aoi_cloud_cover_pct": aoi_cloud_cover_pct,
        "reservoir_area_km2": reservoir_area_km2,
        "total_water_area_km2": total_water_area_km2,
        "num_clusters": num_clusters,
        "quality_score": quality_score,
        "detection_method": detection_method,
        # Data arrays for saving outputs
        "ndwi": ndwi,
        "mndwi": mndwi,
        "water_mask": water_mask,
        "reservoir_mask": reservoir_mask,
        "cloud_mask": cloud_mask,
        "b03_transform": b03_transform,
        "b03_crs": b03_crs,
        "bbox_wgs84": bbox_wgs84,
        "row_slice": row_slice,
        "col_slice": col_slice,
    }


# ──────────────────────────────────────────────
# Main reservoir processing with multi-scene fallback
# ──────────────────────────────────────────────

def process_reservoir(name, geojson_path, token):
    """
    Full pipeline for one reservoir with multi-scene fallback:
    Tries multiple scenes if the first one yields poor results.
    """
    print(f"\n{'='*60}")
    print(f"Processing Reservoir: {name.upper()}")
    print(f"{'='*60}")

    # 1. Load AOI
    gdf = gpd.read_file(geojson_path)
    aoi_geometry = gdf.geometry.values[0]
    bbox = list(gdf.total_bounds)

    # 2. Search for scenes (returns multiple candidates)
    scenes = search_sentinel_scenes(bbox)
    if not scenes:
        print(f"No Sentinel-2 scenes found in the last {LOOKBACK_DAYS} days for {name}.")
        return None

    # 3. Try scenes until we get a good result or exhaust candidates
    best_result = None
    scenes_to_try = min(len(scenes), MAX_SCENES_TO_TRY)

    for attempt_idx in range(scenes_to_try):
        scene = scenes[attempt_idx]
        scene_id = scene.get("id")

        print(f"\n  ╔══ Attempt {attempt_idx + 1}/{scenes_to_try} ══╗")
        try:
            result = try_process_scene(name, aoi_geometry, scene, token)
        except Exception as e:
            print(f"  Error processing scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if result is None:
            continue

        # Keep the best result so far (highest quality score)
        if best_result is None or result["quality_score"] > best_result["quality_score"]:
            best_result = result

        # If result is good enough, stop trying more scenes
        if result["reservoir_area_km2"] >= MIN_EXPECTED_AREA_KM2 and result["quality_score"] >= 60:
            print(f"\n  ✓ Good detection achieved (area={result['reservoir_area_km2']:.2f} km², "
                  f"quality={result['quality_score']:.0f}%). Using this scene.")
            break
        else:
            print(f"\n  ⚠ Low detection (area={result['reservoir_area_km2']:.2f} km², "
                  f"quality={result['quality_score']:.0f}%). Trying next scene...")

    if best_result is None:
        print(f"\nFailed to detect water for {name.upper()} in any of {scenes_to_try} scenes.")
        return None

    # 4. Save outputs using the best result
    acquisition_date = best_result["acquisition_date"]
    scene_id = best_result["scene_id"]
    reservoir_area_km2 = best_result["reservoir_area_km2"]
    total_water_area_km2 = best_result["total_water_area_km2"]
    aoi_cloud_cover_pct = best_result["aoi_cloud_cover_pct"]
    quality_score = best_result["quality_score"]
    detection_method = best_result["detection_method"]

    ndwi = best_result["ndwi"]
    mndwi = best_result["mndwi"]
    water_mask = best_result["water_mask"]
    reservoir_mask = best_result["reservoir_mask"]
    cloud_mask = best_result["cloud_mask"]
    b03_transform = best_result["b03_transform"]
    b03_crs = best_result["b03_crs"]
    bbox_wgs84 = best_result["bbox_wgs84"]
    row_slice = best_result["row_slice"]
    col_slice = best_result["col_slice"]

    print(f"\n{'─'*40}")
    print(f"FINAL Results for {name.upper()}:")
    print(f"  Scene:              {scene_id}")
    print(f"  Reservoir Area:     {reservoir_area_km2:.4f} km²")
    print(f"  Total Water in AOI: {total_water_area_km2:.4f} km²")
    print(f"  AOI Cloud Cover:    {aoi_cloud_cover_pct:.2f}%")
    print(f"  Quality Score:      {quality_score:.0f}%")
    print(f"  Detection Method:   {detection_method}")
    print(f"{'─'*40}")

    # 5. Save outputs
    Path("outputs/maps").mkdir(parents=True, exist_ok=True)
    Path("outputs/vectors").mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    # Reservoir water boundary vector (dissolved polygon)
    save_reservoir_vector(
        reservoir_mask, b03_transform, b03_crs,
        f"outputs/vectors/{name}_{acquisition_date}_reservoir.shp")

    # Tight bounding box vector
    save_bbox_vector(
        bbox_wgs84,
        f"outputs/vectors/{name}_{acquisition_date}_bbox.shp")

    # All water bodies in AOI (for reference)
    save_all_water_shapefile(
        water_mask, b03_transform, b03_crs,
        f"outputs/maps/{name}_{acquisition_date}_water.shp")

    # Visualization
    tight_bbox_slice = (row_slice, col_slice) if row_slice is not None else None
    save_visualization(
        ndwi, mndwi, water_mask, reservoir_mask, cloud_mask,
        tight_bbox_slice, name, acquisition_date,
        reservoir_area_km2, total_water_area_km2,
        aoi_cloud_cover_pct, quality_score, detection_method,
        f"outputs/plots/{name}_{acquisition_date}.png")

    return {
        "date": acquisition_date,
        "reservoir": name,
        "scene_id": scene_id,
        "cloud_cover_scene": best_result["scene_cloud_cover"],
        "cloud_cover_aoi": round(aoi_cloud_cover_pct, 2),
        "reservoir_area_km2": round(reservoir_area_km2, 4),
        "total_water_area_km2": round(total_water_area_km2, 4),
        "water_clusters": best_result["num_clusters"],
        "quality_score": round(quality_score, 1),
        "detection_method": detection_method,
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
            "water_clusters", "quality_score", "detection_method",
            "change_km2", "change_percent"
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
    print(f"Configuration:")
    print(f"  LOOKBACK_DAYS:          {LOOKBACK_DAYS}")
    print(f"  MAX_SCENES_TO_TRY:      {MAX_SCENES_TO_TRY}")
    print(f"  MIN_EXPECTED_AREA_KM2:  {MIN_EXPECTED_AREA_KM2}")
    print(f"  AOI_CLOUD_COVER_SKIP:   {AOI_CLOUD_COVER_SKIP}%")
    print(f"  NDWI_THRESHOLD_DEFAULT: {NDWI_THRESHOLD_DEFAULT}")
    print(f"  MNDWI_THRESHOLD_DEFAULT:{MNDWI_THRESHOLD_DEFAULT}")
    print()

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
