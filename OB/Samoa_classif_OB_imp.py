#%%
import os
import numpy as np
import rasterio
import os, numpy as np, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm

# --- Input paths ---
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
in_tif  = f"{computer_path}classif_VHR_OB/VHR11_object_classif_all.tif"
ndvi_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002_ndvi.tif"
ndwi_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002_ndwi.tif"
nodata = -9999

# --- Output path ---
out_tif = in_tif.replace(".tif", "_masked.tif")

# --- Parameters ---
mask_ndvi_class  = 8111     # class for low NDVI
mask_ndvi_thresh = 0.48

mask_ndwi_class  = 9111     # class for high NDWI (water)
mask_ndwi_thresh = 0.11

# --- Streamed processing ---
with rasterio.open(in_tif) as class_src, \
     rasterio.open(ndvi_tif) as ndvi_src, \
     rasterio.open(ndwi_tif) as ndwi_src:
    
    profile = class_src.profile
    profile.update(dtype=rasterio.int32, count=1, nodata=nodata)

    # Virtual NDVI & NDWI aligned to classification grid
    with WarpedVRT(
        ndvi_src,
        crs=class_src.crs,
        transform=class_src.transform,
        width=class_src.width,
        height=class_src.height,
        resampling=Resampling.bilinear,
    ) as ndvi_vrt, WarpedVRT(
        ndwi_src,
        crs=class_src.crs,
        transform=class_src.transform,
        width=class_src.width,
        height=class_src.height,
        resampling=Resampling.bilinear,
    ) as ndwi_vrt, rasterio.open(out_tif, "w", **profile) as dst:

        total_masked_ndvi = 0
        total_masked_ndwi = 0

        for _, window in tqdm(class_src.block_windows(1), desc="Applying NDVI/NDWI masks"):

            cls_block = class_src.read(1, window=window)
            ndvi_block = ndvi_vrt.read(1, window=window)
            ndwi_block = ndwi_vrt.read(1, window=window)

            out_block = cls_block.copy()
        
             # --- NDVI rule ---
            mask_ndvi = ndvi_block <= mask_ndvi_thresh
            total_masked_ndvi += np.count_nonzero(mask_ndvi)
            out_block[mask_ndvi] = mask_ndvi_class

            # --- NDWI rule (overwrites if both match) ---
            mask_ndwi = ndwi_block > mask_ndwi_thresh
            total_masked_ndwi += np.count_nonzero(mask_ndwi)
            out_block[mask_ndwi] = mask_ndwi_class

            # Preserve nodata
            out_block[cls_block == nodata] = nodata

            dst.write(out_block.astype(np.int32), 1, window=window)

print(f" Masked classification saved: {out_tif}")
print(f"   Pixels set to {mask_ndvi_class} (NDVI ≤ {mask_ndvi_thresh}): {total_masked_ndvi}")
print(f"   Pixels set to {mask_ndwi_class} (NDWI > {mask_ndwi_thresh}): {total_masked_ndwi}")