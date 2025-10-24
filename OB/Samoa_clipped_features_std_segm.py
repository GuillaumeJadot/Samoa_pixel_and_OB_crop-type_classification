
#%%
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import glob
import pathlib as Path
import rasterio
from rasterio import features
import otbApplication
import rasterio
from rasterio.mask import mask
# --- Paths ---
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
img_temp_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002.tif"
feature_tif = img_temp_tif.replace(".tif", "_features.tif")
aoi_clip_file = f'{computer_path}aoi_clip_VHR.shp'
clipped_tif = feature_tif.replace(".tif", "_clipped.tif")

# Output
# Input (clipped) and output (standardized)
# std_tif = clipped_tif.replace(".tif", "_std.tif")
std_tif = feature_tif.replace(".tif", "_std.tif")

# with rasterio.open(clipped_tif) as src:
with rasterio.open(feature_tif) as src:
    data = src.read().astype(np.float32)
    profile = src.profile.copy()
    nodata = src.nodata if src.nodata is not None else -9999

# Create mask for valid data (ignore NoData)
valid_mask = np.all(data != nodata, axis=0)

# Initialize output array
data_std = np.full_like(data, nodata, dtype=np.float32)

# --- Standardize only first 4 bands (R, G, B, NIR) ---
# Compute z-score per band
# for i in range(data.shape[0]):
for i in range(4):
    band = data[i]
    valid_pixels = band[valid_mask]
    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)
    print(f"Band {i+1}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Apply standardization only on valid pixels
    standardized = (band - mean_val) / std_val
    band_std = np.where(valid_mask, standardized, nodata)
    data_std[i] = band_std

# --- Keep NDVI & NDWI unchanged ---
data_std[4] = data[4]  # NDVI
data_std[5] = data[5]  # NDWI
print("Bands 5 (NDVI) and 6 (NDWI) kept unstandardized.")


# Update metadata
profile.update(dtype="float32", nodata=nodata)

# Save standardized cube
with rasterio.open(std_tif, "w", **profile) as dst:
    dst.write(data_std)

print(f"\n Standardized 6-band cube saved:\n{std_tif}")