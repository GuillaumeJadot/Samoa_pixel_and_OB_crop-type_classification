import os, numpy as np, h5py, rasterio
from joblib import load
from tqdm import tqdm

# -- input paths ---
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
# segments_tif = f"{computer_path}segments/04-26-2025_Ortho_4Band-002_features_clipped_std_V4.tif"
segments_tif = f'{computer_path}segments/all_VHR1/04-26-2025_Ortho_4Band-002_features_std.tif'
features_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_features_all.h5"
model_path = f"{computer_path}classif_VHR_OB/model/xgb_object_model_all.joblib"
encoder_path = model_path.replace(".joblib", "_encoder.joblib")
#out
out_tif = f"{computer_path}classif_VHR_OB/VHR11_object_classif_all.tif"
nodata = -9999
chunk = 50_000   # adjust for RAM


# --- Load model and encoder ---
clf = load(model_path)
le = load(encoder_path)

# --- Predict in chunks (no full X_obj in RAM) ---
with h5py.File(features_h5, "r") as hf:
    X_ds = hf["X_obj"]
    seg_ids = hf["seg_ids"][:]
    nobj, nfeat = X_ds.shape
    print(f"Predicting {nobj} objects with {nfeat} features...")

    y_pred_codes = np.empty(nobj, dtype=np.int32)

    for i in tqdm(range(0, nobj, chunk), desc="Predicting"):
        X_chunk = X_ds[i:i+chunk]
        y_chunk = clf.predict(X_chunk)
        y_pred_codes[i:i+chunk] = le.inverse_transform(y_chunk)

# Build lightweight lookup
lut = dict(zip(seg_ids.tolist(), y_pred_codes.tolist()))



# --- Stream raster writing (no full arrays) ---
with rasterio.open(segments_tif) as src:
    profile = src.profile
    profile.update(dtype=rasterio.int32, count=1, nodata=nodata)
    with rasterio.open(out_tif, "w", **profile) as dst:
        for _, window in tqdm(src.block_windows(1), desc="Writing map"):
            seg_block = src.read(1, window=window)
            out_block = np.full(seg_block.shape, nodata, dtype=np.int32)
            mask = seg_block > 0
            if np.any(mask):
                seg_vals = seg_block[mask]
                # Vectorized lookup via np.frompyfunc avoids building huge flat arrays
                get_pred = np.frompyfunc(lambda sid: lut.get(int(sid), nodata), 1, 1)
                out_vals = get_pred(seg_vals).astype(np.int32)
                out_block[mask] = out_vals
            dst.write(out_block, 1, window=window)

print(f" Saved object-based classification map:\n{out_tif}")


# # Predict
# y_pred_all = clf.predict(X_obj)
# y_pred_codes = le.inverse_transform(y_pred_all)

# # Raster backfilling
# with rasterio.open(segments_tif) as src:
#     seg = src.read(1)
#     profile = src.profile
#     profile.update(dtype=rasterio.int32, count=1, nodata=nodata)
#     pred_map = np.full(seg.shape, fill_value=nodata, dtype=np.int32)
#     lut = dict(zip(seg_ids, y_pred_codes))
#     mask_valid = seg > 0
#     seg_flat = seg[mask_valid]
#     pred_flat = np.vectorize(lut.get)(seg_flat)
#     pred_map[mask_valid] = pred_flat

# with rasterio.open(out_tif, "w", **profile) as dst:
#     dst.write(pred_map, 1)

# print(f" Saved object-based classification map:\n{out_tif}")
