import os
import numpy as np
import geopandas as gpd
import pandas as pd
import glob
import pathlib as Path
import rasterio
from rasterio import features
import otbApplication
from IPython.display import display
from matplotlib import pyplot as plt
from collections import defaultdict
print("libraires imported")

#%%

# =====================
# PATHS
# =====================

#in
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
img_temp_tif = glob.glob(f'{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002.tif')[0]
print(f'Raster template file : {img_temp_tif}')
gdf_file = f'{computer_path}survey_windshield_cleaned/04262025/gdf_all_buffered_04262025_V2'
gdf = gpd.read_file(gdf_file)

#out
insitu_tif = f'{computer_path}classif_VHR/insitu_raster/VHR11_insitu_V2.tif'
Path.Path(insitu_tif).parent.mkdir(parents=True, exist_ok=True)
gdf_file2 = f'{computer_path}survey_windshield_cleaned/04262025/gdf2_all_buffered_04262025_V2'


#%%
# ==============================
# ADD POLYGON IDs
# ==============================

# Add polygon IDs (having id and label will help to splitting train/test later based on pixels count)

# if not os.path.exists(gdf_file2):
#     gdf = gdf.reset_index(drop=True)
#     gdf["poly_id"] = gdf.index.astype(int) + 1  # <-- start at 1, not 0 (avoid taking pixel == 0 further)
#     gdf.to_file(gdf_file2, driver="GPKG")
#     print(f"Saved updated GDF with poly_id → {gdf_file2}")
# else:
#     print("GPKG with poly_id already exists, skipping.")
#     gdf = gpd.read_file(gdf_file2)



gdf = gdf.reset_index(drop=True)
gdf["poly_id"] = gdf.index.astype(int) + 1  # <-- start at 1, not 0 (avoid taking pixel == 0 further)
gdf.to_file(gdf_file2, driver="GPKG")
print(f"Saved updated GDF with poly_id → {gdf_file2}")


#%%

# =========================
# RASTERIZATION
# =========================

crop_tif = insitu_tif.replace(".tif", "_crop.tif")
poly_tif = insitu_tif.replace(".tif", "_poly.tif")

if not os.path.exists(crop_tif):
    # Initialize Rasterization application
    r_crop = otbApplication.Registry.CreateApplication("Rasterization")
    # Set input vector data
    r_crop.SetParameterString("in", gdf_file2)
    # Template raster — defines projection, resolution, and extent
    r_crop.SetParameterString("im", img_temp_tif)
    # Rasterization mode: use an attribute field to burn values
    r_crop.SetParameterString("mode", "attribute")
    r_crop.SetParameterString("mode.attribute.field", "crop_code") ##take the crop_code and poly_id fields, poly_id will be used for the split
    # Set output raster
    r_crop.SetParameterString("out", crop_tif)
    # (Optional) Data type — default is float32; you can force int16
    r_crop.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int32)
    # Execution
    r_crop.ExecuteAndWriteOutput()
    print("Rasterization (crop_code) done.")
else:
    print("Crop raster already exists, skipping rasterization.")



# --- Rasterize poly_id ---
if not os.path.exists(poly_tif):
    r_poly = otbApplication.Registry.CreateApplication("Rasterization")
    r_poly.SetParameterString("in", gdf_file2)
    r_poly.SetParameterString("im", img_temp_tif)
    r_poly.SetParameterString("mode", "attribute")
    r_poly.SetParameterString("mode.attribute.field", "poly_id")
    r_poly.SetParameterString("out", poly_tif)
    r_poly.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int32)
    r_poly.ExecuteAndWriteOutput()
    print("Rasterization (poly_id) done.")
else:
    print("Poly raster already exists, skipping rasterization.")


# --- Concatenate the two rasters into a single one ---
if not os.path.exists(insitu_tif):
    concat = otbApplication.Registry.CreateApplication("ConcatenateImages")
    concat.SetParameterStringList("il", [crop_tif, poly_tif])
    concat.SetParameterString("out", insitu_tif)
    concat.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int32)
    concat.ExecuteAndWriteOutput()
    print("Concatenation done.")
else:
    print("Combined in-situ raster already exists, skipping concatenation.")

#%%

# ======================================================
# LOAD RASTER & COMPUTE PIXEL COUNTS
# ======================================================


# Loading tiff insitu
# with rasterio.open(insitu_tif) as src:
#     insitu_arr = src.read(1)
#     print(f"Shape: {insitu_arr.shape}")
#     print(f"CRS: {src.crs}")
#     print(f"Bounds: {src.bounds}")

# # Count pixels per crop_code
# unique, counts = np.unique(insitu_arr, return_counts=True)
# pix_count_df = pd.DataFrame(zip(unique, counts), columns=['crop_code', 'pix_count'])
# pix_count_df = pix_count_df.astype({'crop_code': 'str', 'pix_count': 'int64'})
# display(pix_count_df)

# --- Read in-situ raster once ---

# Count pixels per polygon (poly_id) instead of crop_code (avoid biais due to large polygon)
with rasterio.open(insitu_tif) as src:
    crop_arr = src.read(1)
    poly_arr = src.read(2)
    profile = src.profile

# --- Compute pixel counts per polygon ---
unique, counts = np.unique(poly_arr[poly_arr > 0], return_counts=True)
pix_count_df = pd.DataFrame({'poly_id': unique, 'pix_count': counts})
gdf = gdf.merge(pix_count_df, on='poly_id', how='left')

# Merge pixel counts into the original GeoDataFrame
# Ensure same dtype for merge key
# gdf['crop_code'] = gdf['crop_code'].astype(int)
# pix_count_df['crop_code'] = pix_count_df['crop_code'].astype(int)
# gdf = gdf.merge(pix_count_df, on='crop_code', how='left')
# display(gdf.head())

print(f'There are {len(gdf)} polygons')

#%%

# ======================================================
# SPLIT CAL / VAL
# ======================================================

# --- Split polygons into calibration / validation ---
pc_cal = 75

# Determine pixel targets for calibration
gdf['crop_pix'] = gdf.groupby('crop_code')['pix_count'].transform('sum')
crop_targets = (
    gdf[['crop_code', 'crop_pix']].drop_duplicates().set_index('crop_code') * (pc_cal / 100)).to_dict()['crop_pix']

# Shuffle DataFrame to randomize selection
gdf = gdf.sample(frac=1, random_state=42).reset_index(drop=True)

# Assign polygons to calibration or validation
training_pixels = defaultdict(int)
cal_idx, val_idx = [], []

for idx, row in gdf.iterrows():
    crop_id = row['crop_code']
    crop_pix = row['pix_count']
    # ensures each crop class’s pixel proportion is roughly 75/25, not just number of polygons
    if training_pixels[crop_id] + crop_pix <= crop_targets[crop_id]:
        # Add to calibration if within the target
        training_pixels[crop_id] += crop_pix
        cal_idx.append(idx)
    else:
        # Otherwise, add to validation
        val_idx.append(idx)

# Create calibration and validation GeoDataFrames
cal_gdf = gdf.loc[cal_idx].reset_index(drop=True)
val_gdf = gdf.loc[val_idx].reset_index(drop=True)

#%%
# --- Write GeoPackages ---
cal_gpkg = f"{computer_path}classif_VHR/insitu_raster/VHR11_labels_cal.gpkg"
val_gpkg = f"{computer_path}classif_VHR/insitu_raster/VHR11_labels_val.gpkg"

if not os.path.exists(cal_gpkg) or not os.path.exists(val_gpkg):
    cal_gdf.to_file(cal_gpkg, driver='GPKG')
    val_gdf.to_file(val_gpkg, driver='GPKG')
    print("Calibration and validation GPKG saved.")
else:
    print("Cal/Val GPKG already exist, skipping save.")

print(f"Calibration: {len(cal_gdf)} polygons")
print(f"Validation: {len(val_gdf)} polygons")

#%% 
# Sanity chcek 

# Load saved calibration and validation shapefiles and print unique crop_code counts
# cal_gdf_file = insitu_tif.replace('.tif', '_cal.shp')
# val_gdf_file = insitu_tif.replace('.tif', '_val.shp')

# cal_gdf = gpd.read_file(cal_gdf_file)
# val_gdf = gpd.read_file(val_gdf_file)

print(f'Calibration polygons: {len(cal_gdf)}')
print(f'Validation polygons: {len(val_gdf)}')

print(f'Calibration unique crop_code count: {cal_gdf["crop_code"].nunique()}')
print(f'Validation unique crop_code count: {val_gdf["crop_code"].nunique()}')

print('Calibration crop_codes:', sorted(cal_gdf['crop_code'].unique()))
print('Validation crop_codes:', sorted(val_gdf['crop_code'].unique()))

def summarize_split(gdf, name):
    dist = gdf['crop_code'].value_counts().sort_index()
    print(f"\n{name} set distribution:")
    print(dist)

summarize_split(cal_gdf, "Calibration")
summarize_split(val_gdf, "Validation")


#%%

# --- Build raster subsets ---

# ======================================================
# CREATE CAL / VAL TIFS
# ======================================================


nodata_val = -9999  # (value outside crop_code range)

cal_tif = insitu_tif.replace(".tif", "_cal.tif")
val_tif = insitu_tif.replace(".tif", "_val.tif")

# if Overlap between cal/val: 0 : good
cal_ids = cal_gdf["poly_id"].dropna().astype(np.int64).values
val_ids = val_gdf["poly_id"].dropna().astype(np.int64).values
# exclude background ID 0
cal_ids = cal_ids[cal_ids > 0]
val_ids = val_ids[val_ids > 0]
print(f"Unique poly_id in calibration: {len(np.unique(cal_ids))}")
print(f"Unique poly_id in validation: {len(np.unique(val_ids))}")
print(f"Overlap between cal/val: {len(np.intersect1d(cal_ids, val_ids))}")

# --- Extra sanity check ---
if len(val_ids) == 0 or val_ids.dtype != np.int64:
    raise ValueError(f"Validation poly_id array looks wrong: dtype={val_ids.dtype}, len={len(val_ids)}")


# Create binary masks using polygon IDs
# cal_mask = np.isin(poly_arr, cal_gdf["poly_id"].values)
# val_mask = np.isin(poly_arr, val_gdf["poly_id"].values)

# Apply mask to get raster subsets
# cal_raster = np.where(cal_mask, crop_arr, 0)
# val_raster = np.where(val_mask, crop_arr, 0)

cal_mask = np.isin(poly_arr.astype(np.int64), cal_ids)
val_mask = np.isin(poly_arr.astype(np.int64), val_ids)

# sanity check
print(f"Pixels in calibration mask: {cal_mask.sum()}")
print(f"Pixels in validation mask: {val_mask.sum()}")

# --- Make sure masks do not overlap ---
overlap_pixels = np.logical_and(cal_mask, val_mask).sum()
print(f"Overlap pixels (should be 0): {overlap_pixels}")

cal_raster = np.where(cal_mask, crop_arr, nodata_val)
val_raster = np.where(val_mask, crop_arr, nodata_val)
profile.update(dtype="int16", count=1, nodata=nodata_val)

print(f"Calibration total valid pixels: {(cal_raster != nodata_val).sum()}")
print(f"Validation total valid pixels: {(val_raster != nodata_val).sum()}")


# Save results

if not os.path.exists(cal_tif):
    with rasterio.open(cal_tif, "w", **profile) as dst:
        dst.write(cal_raster.astype(np.int16), 1)
    print("Calibration raster saved.")
else:
    print("Calibration raster already exists, skipping.")

if not os.path.exists(val_tif):
    with rasterio.open(val_tif, "w", **profile) as dst:
        dst.write(val_raster.astype(np.int16), 1)
    print("Validation raster saved.")
else:
    print("Validation raster already exists, skipping.")
