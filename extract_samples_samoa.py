
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from rasterio.windows import from_bounds
from joblib import dump, load
import joblib
import xgboost as xgb
from tabulate import tabulate
from sklearn.metrics import f1_score
import h5py


# --- Paths ---
#in
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
img_temp_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002.tif"
feature_tif = img_temp_tif.replace(".tif", "_features.tif")
#out
insitu_tif = f'{computer_path}classif_VHR/insitu_raster/VHR11_insitu_V2_val.tif'
output_hdf5 = f"{computer_path}classif_VHR/hdf5/VHR11_features_V2_val.h5"
# Make sure output folder exists
Path.Path(output_hdf5).parent.mkdir(parents=True, exist_ok=True)

nodata = -9999

# =========================
# Function to extract data
# =========================

def extract_features_to_hdf5(feature_tif, insitu_tif, output_hdf5, nodata=0):
    """
    Extracts feature vectors only where in-situ labels exist
    and saves them as an HDF5 dataset.
    """
    print(f"\n Loading features from: {feature_tif}")
    print(f" Using in-situ labels from: {insitu_tif}")

    with rasterio.open(feature_tif) as src_feat, rasterio.open(insitu_tif) as src_lab:
        # Sanity check: ensure same dimensions / alignment
        if (src_feat.width != src_lab.width) or (src_feat.height != src_lab.height):
            raise ValueError("Feature cube and in-situ raster do not match in dimensions.")
        if src_feat.crs != src_lab.crs:
            raise ValueError("Feature and in-situ rasters have different CRS.")

        # Read data
        features = np.stack([src_feat.read(i).astype(np.float32) for i in range(1, src_feat.count + 1)], axis=-1)
        labels = src_lab.read(1).astype(np.int32)

        features[np.isnan(features)] = nodata

    # Mask valid samples (ignore nodata)
    mask = labels != nodata
    print(f" Valid pixels: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.2f}%)")

    # Extract features and labels only where mask is True
    X = features[mask, :]  # shape: (n_valid_pixels, n_features)
    y = labels[mask]       # shape: (n_valid_pixels,)

    print(f"Extracted feature matrix: {X.shape}, labels: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    # Save to HDF5
    with h5py.File(output_hdf5, "w") as hf:
        hf.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        hf.create_dataset("y", data=y, compression="gzip", compression_opts=4)

    print(f" Saved HDF5 file: {output_hdf5}")
    print(f"   - Dataset 'X' shape: {X.shape}")
    print(f"   - Dataset 'y' shape: {y.shape}")

# =========================
# Run the extraction
# =========================
extract_features_to_hdf5(feature_tif, insitu_tif, output_hdf5, nodata)