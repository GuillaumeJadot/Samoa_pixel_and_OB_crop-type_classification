#%%
import os, numpy as np, h5py, rasterio
from collections import Counter, defaultdict
import geopandas as gpd
import pandas as pd
import glob
import pathlib as Path
import rasterio
from rasterio import features
import otbApplication
import rasterio
from rasterio.mask import mask
from collections import defaultdict
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import fiona
from shapely.geometry import shape, mapping
from rasterio.features import shapes
import numpy as np, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from collections import defaultdict

# --- Paths ---
# --- input path ---
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
# segments_tif = f"{computer_path}segments/04-26-2025_Ortho_4Band-002_features_clipped_std_V4.tif"
segments_tif = f'{computer_path}segments/all_VHR1/04-26-2025_Ortho_4Band-002_features_std.tif'
in_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_features_all.h5"

dataset_type = "val"  # or "val"
label_tif = f"{computer_path}classif_VHR/insitu_raster/VHR11_insitu_V2_{dataset_type}.tif"

# --- output path ---
# out_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_training.h5"
out_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_all_{dataset_type}.h5"

nodata = -9999
purity_thresh = 0.9

#%%
# Load object features
# Read seg IDs list from HDF5
with h5py.File(in_h5, "r") as hf:
    seg_ids = hf["seg_ids"][:]           # small
    X_ds = hf["X_obj"]                   # keep as on-disk dataset (no [:])

# Dense index mapping for segment ids
seg_ids_sorted = np.asarray(seg_ids, dtype=np.int64)
seg_id_to_idx = {int(s): i for i, s in enumerate(seg_ids_sorted)}
N = seg_ids_sorted.size


# First pass: collect present class codes
class_set = set()

with rasterio.open(segments_tif) as seg_src, rasterio.open(label_tif) as lab_src:
    # Virtual warped view: lab in segmentation grid/res/crs
    with WarpedVRT(
        lab_src,
        crs=seg_src.crs,
        transform=seg_src.transform,
        width=seg_src.width,
        height=seg_src.height,
        resampling=Resampling.nearest,
        nodata=nodata
    ) as lab_vrt:

        # First pass over windows: collect present class codes (to size the table)
        for _, window in seg_src.block_windows(1):
            seg_tile = seg_src.read(1, window=window)
            lab_tile = lab_vrt.read(1, window=window)
            valid = (seg_tile > 0) & (lab_tile != nodata)
            if not np.any(valid): 
                continue
            cls_vals = np.unique(lab_tile[valid])
            class_set.update(map(int, cls_vals))

# Compact class codes
classes = np.array(sorted(class_set), dtype=np.int32)
K = classes.size
class_to_idx = {int(c): i for i, c in enumerate(classes)}

# Segment×class counts (int32 is enough; grow to int64 if you expect very large counts)
SC = np.zeros((N, K), dtype=np.int32)

# Second pass: fill counts with vectorized binning
with rasterio.open(segments_tif) as seg_src, rasterio.open(label_tif) as lab_src:
    with WarpedVRT(
        lab_src,
        crs=seg_src.crs,
        transform=seg_src.transform,
        width=seg_src.width,
        height=seg_src.height,
        resampling=Resampling.nearest,
        nodata=nodata
    ) as lab_vrt:

        for _, window in seg_src.block_windows(1):
            seg_tile = seg_src.read(1, window=window)
            lab_tile = lab_vrt.read(1, window=window)
            valid = (seg_tile > 0) & (lab_tile != nodata)
            if not np.any(valid): 
                continue

            s = seg_tile[valid].astype(np.int64, copy=False)
            c = lab_tile[valid].astype(np.int64, copy=False)

            # map to dense indices
            # (vectorized map via searchsorted requires sorted keys; dict lookup also works)
            seg_idx = np.fromiter((seg_id_to_idx.get(int(v), -1) for v in s), count=s.size, dtype=np.int64)
            keep = seg_idx >= 0
            if not np.any(keep): 
                continue
            seg_idx = seg_idx[keep]
            c = c[keep]
            cls_idx = np.fromiter((class_to_idx[int(v)] for v in c), count=c.size, dtype=np.int64)

            # 2D bincount via 1D trick
            flat = seg_idx * K + cls_idx
            bc = np.bincount(flat, minlength=N * K)
            SC += bc.reshape(N, K).astype(SC.dtype, copy=False)

# Majority + purity per segment (no Python Counter)
tot = SC.sum(axis=1)
maj_idx = SC.argmax(axis=1)
maj_cnt = SC[np.arange(N), maj_idx]
pure = (tot > 0) & (maj_cnt / np.maximum(tot, 1) >= purity_thresh)

y_obj = np.full(N, -1, dtype=np.int32)  # -1 = unlabeled/impure
y_obj[pure] = classes[maj_idx[pure]]

mask = y_obj >= 0
sid_train = seg_ids_sorted[mask]
y_train_obj = y_obj[mask]

#%%
# Copy selected rows to a new HDF5 in chunks
chunk = 50_000  # tune for your system
idxs = np.flatnonzero(mask)
with h5py.File(in_h5, "r") as hf_in, h5py.File(out_h5, "w") as hf_out:
    X_in = hf_in["X_obj"]
    nfeat = X_in.shape[1]
    X_out = hf_out.create_dataset(
        "X_train_obj", shape=(idxs.size, nfeat), dtype=X_in.dtype, chunks=True, compression="gzip"
    )
    y_out = hf_out.create_dataset("y_train_obj", data=y_train_obj, compression="gzip")
    sid_out = hf_out.create_dataset("seg_ids_train", data=sid_train, compression="gzip")

    for i in range(0, idxs.size, chunk):
        sl = idxs[i:i+chunk]
        X_out[i:i+sl.size] = X_in[sl]   # fancy index; h5py reads sparsely

print("Saved:", out_h5)


#%%

# Only keep labeled segments in the mask to avoid extracting everything
with rasterio.open(segments_tif) as src:
    seg = src.read(1)
    transform = src.transform
    crs = src.crs
mask_seg = np.isin(seg, sid_train)

schema = {
    "geometry": "Polygon",
    "properties": {"seg_id": "int64", "class_code": "int32"},
}
out_shp = out_h5.replace(".h5", "_segments.shp")

# LUT from segment id → class
lut = dict(zip(sid_train.tolist(), y_train_obj.tolist()))

with fiona.open(out_shp, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as sink:
    for geom, value in shapes(seg, mask=mask_seg, transform=transform):
        seg_id = int(value)
        cls = lut.get(seg_id)
        if cls is None:
            continue
        sink.write({
            "geometry": geom,               # already GeoJSON-like mapping
            "properties": {"seg_id": seg_id, "class_code": int(cls)}
        })
print(f"Saved labeled segments shapefile: {out_shp}")
