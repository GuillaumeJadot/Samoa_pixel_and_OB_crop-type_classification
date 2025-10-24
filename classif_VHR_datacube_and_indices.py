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


computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
img_temp_tif = glob.glob(f'{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002.tif')[0]
print(f'Raster template file : {img_temp_tif}')

# list_src_arr = []
# with rasterio.open(img_temp_tif, "r") as src:
#     print(src.meta)
    
#     # Iterate over each band and append to list
#     for i in range(1, src.count + 1):  # Bands in rasterio are 1-indexed
#         band_data = src.read(i)
#         list_src_arr.append(band_data)
# print(f"Number of bands loaded: {len(list_src_arr)}")
# print(f'Number of features : {len(list_src_arr)}')

def build_feature_cube(img_temp_tif):
    with rasterio.open(img_temp_tif, "r") as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        profile = src.meta.copy()
    
     # Mask no-data (if reflectance = 0)
    nodata_val = -9999.0  # same as in in-situ raster
    mask = (red == 0) & (green == 0) & (blue == 0) & (nir == 0)
    red[mask] = nodata_val
    green[mask] = nodata_val
    blue[mask] = nodata_val
    nir[mask] = nodata_val

    ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), nodata_val)
    ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), nodata_val)

    profile.update(dtype="float32", count=6, nodata=nodata_val)

    feature_tif = img_temp_tif.replace(".tif", "_features.tif")

    with rasterio.open(feature_tif, "w", **profile) as dst:
        for i, arr in enumerate([red, green, blue, nir, ndvi, ndwi], start=1):
            dst.write(arr.astype(np.float32), i)
            names = ["Red", "Green", "Blue", "NIR", "NDVI", "NDWI"]
            dst.set_band_description(i, names[i - 1])
    print(f"Feature cube saved: {feature_tif}")
    print("Bands: Red, Green, Blue, NIR, NDVI, NDWI")

    return feature_tif, 6

# feature_tif = build_feature_cube(img_temp_tif)
feature_tif, n_features = build_feature_cube(img_temp_tif)
print(f"Feature TIF ready with {n_features} layers.")