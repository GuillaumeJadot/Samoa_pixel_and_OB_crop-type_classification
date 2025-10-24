#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import geopandas as gpd
import rasterio
from rasterio import features
from tqdm import tqdm
import numpy as np

# -------------------------------
# User configuration
# -------------------------------
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
segments_shp = f"{computer_path}segments/all_VHR1/04-26-2025_Ortho_4Band-002_features_std.shp"
reference_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002_features_std.tif"  # same grid as features
out_tif = segments_shp.replace(".shp", ".tif")

# -------------------------------
# Rasterization
# -------------------------------
print(f"Reading reference raster: {reference_tif}")
with rasterio.open(reference_tif) as ref:
    meta = ref.meta.copy()
    transform = ref.transform
    crs = ref.crs
    shape = (ref.height, ref.width)

print(f"Reading segmentation polygons: {segments_shp}")
gdf = gpd.read_file(segments_shp)
if "seg_id" not in gdf.columns:
    # Try common column names
    possible = [c for c in gdf.columns if "id" in c.lower() or "seg" in c.lower()]
    if not possible:
        raise ValueError("Could not find a segment ID column in the shapefile.")
    seg_col = possible[0]
    print(f"[INFO] Using '{seg_col}' as segment ID column.")
else:
    seg_col = "seg_id"

# Ensure IDs are integers
gdf[seg_col] = gdf[seg_col].astype(int)

# Create generator of (geometry, value) pairs
shapes_iter = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf[seg_col]))

# Output metadata
meta.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": "int32",
    "compress": "lzw"
})

print(f"Rasterizing {len(gdf)} polygons to match {reference_tif} grid...")
with rasterio.open(out_tif, "w", **meta) as dst:
    seg_raster = features.rasterize(
        shapes=shapes_iter,
        out_shape=shape,
        transform=transform,
        fill=0,            # background = 0
        dtype="int32",
        all_touched=False, # conservative rasterization (set True for full coverage)
        default_value=1
    )
    dst.write(seg_raster, 1)

print(f" Saved rasterized segmentation: {out_tif}")
