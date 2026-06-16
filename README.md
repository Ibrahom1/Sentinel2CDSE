# Sentinel-2 Reservoir Water Monitoring Pipeline

An automated Python pipeline that uses Sentinel-2 L2A satellite imagery to monitor water surface area changes in reservoirs. This system runs automatically every 7 days using GitHub Actions, streaming only the required bands for specific areas of interest (AOIs) without full product downloads.

## Reservoirs Monitored
1. **Bhakra Reservoir** (`aoi/bhakra.geojson`)
2. **Thein Reservoir** (Ranjit Sagar) (`aoi/thein.geojson`)
3. **Pong Reservoir** (`aoi/pong.geojson`)

---

## Project Structure

```
reservoir-monitor/
├── .github/
│   └── workflows/
│       └── monitor.yml         # GitHub Actions schedule & run config
├── aoi/
│   ├── bhakra.geojson          # Bounding box polygon for Bhakra
│   ├── pong.geojson            # Bounding box polygon for Pong
│   └── thein.geojson           # Bounding box polygon for Thein
├── scripts/
│   └── process_reservoirs.py   # Core processing pipeline script
├── outputs/
│   ├── water_history.csv       # Historical log of water areas & changes
│   ├── maps/                   # Date-specific GeoJSON water boundary maps
│   └── plots/                  # Side-by-side NDWI/Mask pngs and history trends
├── requirements.txt            # Python library dependencies
└── README.md                   # Setup and execution guide (this file)
```

---

## Setup Instructions

### 1. Register a Copernicus Data Space Ecosystem (CDSE) Account
If you don't already have one, register for a free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/).

### 2. Configure GitHub Secrets (for Automatic Execution)
To enable GitHub Actions to run the pipeline automatically, you must add your credentials as Repository Secrets:
1. On GitHub, navigate to your repository.
2. Go to **Settings** -> **Secrets and variables** -> **Actions**.
3. Click **New repository secret**.
4. Add the following secrets:
   - **`CDSE_USERNAME`**: Your CDSE account email/username.
   - **`CDSE_PASSWORD`**: Your CDSE account password.

### 3. Local Execution & Testing
To run the monitoring pipeline on your local computer:

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your credentials:**
   Create a file named `.env` in the project root directory and add your CDSE credentials:
   ```env
   CDSE_USERNAME=your_copernicus_username_or_email
   CDSE_PASSWORD=your_copernicus_password
   ```

3. **Run the pipeline script:**
   ```bash
   python scripts/process_reservoirs.py
   ```
   *Note: If no `.env` file is present, the script runs in a safe "dry-run" mode to verify configuration and structure.*

---

## Technical Details

### 1. Data Access (STAC API & Window Streaming)
Rather than downloading large Sentinel-2 SAFE archives (typically 700 MB to 1.2 GB), this pipeline:
- Searches the **CDSE STAC API** (`https://catalogue.dataspace.copernicus.eu/stac/search`) for the newest, clearest Sentinel-2 L2A scene covering each reservoir.
- Connects directly to the JPEG2000 band assets via HTTPS using the CDSE OAuth token.
- Uses **Rasterio's `/vsicurl/` virtual file system** to stream only the pixels within the reservoir's bounding box. This reduces bandwidth usage from gigabytes to just a few megabytes per run.

### 2. Analysis & Indexing
- **Resolution:** Bands are analyzed at a native spatial resolution of **10 meters** (1 pixel = 100 m²).
- **Index (NDWI):** 
  $$\text{NDWI} = \frac{\text{Green} (B03) - \text{NIR} (B08)}{\text{Green} (B03) + \text{NIR} (B08)}$$
  Water pixels are identified where $\text{NDWI} > 0.05$.
- **Cloud & Shadow Masking:** The **Scene Classification Layer (SCL)** band (20m native resolution, nearest-neighbor upsampled to 10m) is used to detect and mask out clouds (classes 8, 9, 10) and cloud shadows (class 3).
- **AOI Cloud Percentage:** The script calculates the exact percentage of cloud cover over the reservoir AOI. This helps you identify if a sudden drop in water area is due to cloud cover obstructing the view rather than a physical change in reservoir level.

### 3. Generated Outputs
On each successful run:
- **`outputs/water_history.csv`**: A structured table containing columns: `date`, `reservoir`, `scene_id`, `cloud_cover_scene`, `cloud_cover_aoi`, `water_area_km2`, `change_km2`, `change_percent`. It automatically computes the change from the last observation.
- **`outputs/maps/{reservoir}_{date}_water.geojson`**: A vector polygon file of the extracted water body. You can drag and drop this directly into QGIS or ArcGIS to visualize the water boundary.
- **`outputs/plots/{reservoir}_{date}.png`**: A side-by-side verification plot showing the NDWI gradient on the left and the binary water mask on the right.
- **`outputs/plots/history.png`**: A combined historical trend chart showing the surface area changes over time for all monitored reservoirs.
