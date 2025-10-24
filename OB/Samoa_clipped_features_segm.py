
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

# Output
clipped_tif = feature_tif.replace(".tif", "_clipped.tif")

#%%

# --- Load AOI polygon ---
aoi = gpd.read_file(aoi_clip_file)
aoi = aoi.to_crs(rasterio.open(feature_tif).crs)  # ensure CRS match
# Convert AOI to GeoJSON-like mapping for rasterio
aoi_geom = [aoi.geometry.unary_union.__geo_interface__]


#%%
# --- Clip the feature raster ---
with rasterio.open(feature_tif) as src:
    out_image, out_transform = mask(src, aoi_geom, crop=True)
    out_meta = src.meta.copy()


# Update metadata
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})


#%%
# --- Save clipped 6-band feature cube ---
with rasterio.open(clipped_tif, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"Clipped feature cube saved:\n{clipped_tif}")
print(f"  Bands: {out_image.shape[0]}, Shape: {out_image.shape[1:]}")
