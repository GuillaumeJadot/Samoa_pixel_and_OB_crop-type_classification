#%%
import glob, os, time, math
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
from rasterio.windows import Window
import psutil
import gc

# --- Paths ---

computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
img_temp_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002.tif"
feature_tif = img_temp_tif.replace(".tif", "_features.tif")

model_path = f"{computer_path}classif_VHR/model/VHR11_classifV3_V2.joblib"
# scaler_path = model_path.replace(".joblib", "_scaler.joblib")
encoder_path = model_path.replace(".joblib", "_encoder.joblib")

classif_tif_XGB = f"{computer_path}classif_VHR/XGB_classifV3_VHR11.tif"
nodata = -9999

#%%
# --- Load model, scaler, and encoder ---
model = load(model_path)
# scaler = load(scaler_path)
label_encoder = load(encoder_path)
# with rasterio.open(feature_tif) as src_feat:
#     feat_arr = np.dstack([src_feat.read(i).astype(np.float32)
#                           for i in range(1, src_feat.count + 1)])

available_ram_gb = psutil.virtual_memory().available / 1e9
if available_ram_gb < 16:
    model.set_params(n_jobs=1)
elif available_ram_gb < 32:
    model.set_params(n_jobs=4)
else:
    model.set_params(n_jobs=-1)

#%%
# --- Open input raster ---
with rasterio.open(feature_tif) as src:
    profile = src.profile.copy()
    profile.update(dtype="int32", count=1, nodata=nodata)
    ncols, nrows = src.width, src.height
    block_size = 1024  # tile size (adjust if you have more/less RAM)

    print(f"Starting tile-based prediction on {ncols}x{nrows} raster ({src.count} features).")
    start_time = time.time()

    with rasterio.open(classif_tif_XGB, "w", **profile) as dst:
        for row_off in range(0, nrows, block_size):
            for col_off in range(0, ncols, block_size):
                # Define window
                width = min(block_size, ncols - col_off)
                height = min(block_size, nrows - row_off)
                window = Window(col_off, row_off, width, height)

                # Read 6-band tile
                data = np.stack(
                    [src.read(i, window=window).astype(np.float32) for i in range(1, src.count + 1)],
                    axis=-1
                )

                # Mask and reshape
                no_data_mask = np.isnan(data).any(axis=2) | (data == nodata).any(axis=2)
                data = np.nan_to_num(data, nan=nodata)
                X = data.reshape(-1, data.shape[2])

                # Predict only valid pixels
                valid_mask = ~no_data_mask.ravel()
                preds_encoded = np.full(X.shape[0], nodata, dtype=np.int32)
                X_valid = None

                try:
                    if valid_mask.any():
                        # X_valid = scaler.transform(X[valid_mask])
                        X_valid = X[valid_mask]  # If no scaler is used
                        preds_encoded[valid_mask] = model.predict(X_valid)
                except Exception as e:
                    print(f"Prediction error at window col_off={col_off}, row_off={row_off}: {e}")

                preds_encoded = preds_encoded.reshape(height, width)

                # Decode to original crop codes
                preds_decoded = np.full_like(preds_encoded, nodata)
                valid_predictions = preds_encoded != nodata
                if valid_predictions.any():
                    preds_decoded[valid_predictions] = label_encoder.inverse_transform(
                        preds_encoded[valid_predictions]
                    )

                # Write to output file
                dst.write(preds_decoded.astype(np.int32), 1, window=window)

                # --- Free memory manually ---
                # del data, X, X_valid, preds_encoded, preds_decoded, no_data_mask, valid_mask, valid_predictions
                for var in ["data", "X", "X_valid", "preds_encoded", "preds_decoded",
                            "no_data_mask", "valid_mask", "valid_predictions"]:
                    if var in locals():
                        del locals()[var]
                gc.collect()

            # Optional: show progress
            progress = (row_off / nrows) * 100
            print(f"Processed rows {row_off}/{nrows} ({progress:.1f}%)")

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n Full-scene prediction complete in {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
    print(f"Saved classification map → {classif_tif_XGB}")
