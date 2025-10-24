#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, h5py, rasterio 
from collections import defaultdict 
import geopandas as gpd 
import pandas as pd 
import glob 
import pathlib as Path 
from rasterio  import features 
from rasterio.mask import mask 
import scipy.stats as stats 
from tqdm import tqdm

# -------------------------------
# User paths (adjust as needed)
# -------------------------------
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/" 
# segments_shp = f"{computer_path}segments/04-26-2025_Ortho_4Band-002_features_clipped_std_V4.shp" 
segments_shp = f'{computer_path}segments/all_VHR1/04-26-2025_Ortho_4Band-002_features_std.shp' 
segments_tif = segments_shp.replace(".shp", ".tif") 
# feature_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002_features_clipped_std.tif" 
feature_tif = f"{computer_path}VHR/VHR1/04-26-2025_Ortho_4Band-002_features_std.tif" 
# out 
out_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_features_all.h5" 
out_dir = os.path.dirname(out_h5) or "." 
os.makedirs(out_dir, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------

def iter_windows(src, band=1):
    """
    Iterate over windows that match the raster's internal tiling (fast path).
    Falls back to a single full-window if the dataset is stripped.
    """
    try:
        # Streams block by block if tiled
        for ji, window in src.block_windows(band):
            yield window
    except Exception:
        # Fallback: one full window
        from rasterio.windows import Window
        yield Window(0, 0, src.width, src.height)

def check_alignment(seg_src, feat_src):
    """Basic sanity checks to prevent silent misalignment."""
    if seg_src.width != feat_src.width or seg_src.height != feat_src.height:
        raise ValueError(
            f"Shape mismatch: segments {seg_src.width}x{seg_src.height} vs "
            f"features {feat_src.width}x{feat_src.height}"
        )
    if seg_src.transform != feat_src.transform:
        # Not strictly required if arrays are same size and we only use array indexing,
        # but warn because it usually indicates a geospatial misalignment.
        print("[WARN] Geo-transform differs between rasters. "
              "Arrays will be matched by pixel index, not georeference.")

def safe_float32(x):
    """Cast to float32 without copying if possible."""
    return x.astype(np.float32, copy=False)

# -------------------------------
# Main
# -------------------------------
def main(segments_tif, feature_tif, out_h5):
    with rasterio.open(segments_tif) as seg_src, rasterio.open(feature_tif) as feat_src:
        check_alignment(seg_src, feat_src)
        B = feat_src.count

        # ---------------------------
        # PASS 1: collect all seg IDs
        # ---------------------------
        seg_id_set = set()
        pbar = tqdm(iter_windows(seg_src, band=1), desc="Pass 1/2: scanning IDs", unit="win")
        for window in pbar:
            seg_tile = seg_src.read(1, window=window)
            # Collect positive IDs only
            ids = np.unique(seg_tile[seg_tile > 0])
            if ids.size:
                seg_id_set.update(map(int, ids))

        if not seg_id_set:
            raise ValueError("No positive segment IDs found.")

        seg_ids = np.array(sorted(seg_id_set), dtype=np.int64)   # shape (N,)
        N = seg_ids.size
        print(f"Total segments: {N}, Bands: {B}")

        # Build a compact index mapping (ID -> [0..N-1]) without allocating a huge dense map
        # We'll use searchsorted on the sorted seg_ids for each tile:
        # idx = np.searchsorted(seg_ids, seg_flat) and rely on exact match.

        # --------------------------------------------
        # Allocate accumulators in compact (B, N) form
        # --------------------------------------------
        sum_all  = np.zeros((B, N), dtype=np.float64)
        sum2_all = np.zeros((B, N), dtype=np.float64)
        min_all  = np.full((B, N), np.inf, dtype=np.float64)
        max_all  = np.full((B, N), -np.inf, dtype=np.float64)
        count_all = np.zeros(N, dtype=np.int64)  # counts are per-segment (not per-band)

        # --------------------------------
        # PASS 2: stream tiles & accumulate
        # --------------------------------
        pbar = tqdm(iter_windows(seg_src, band=1), desc="Pass 2/2: accumulating", unit="win")
        for window in pbar:
            seg_tile = seg_src.read(1, window=window)
            valid = seg_tile > 0
            if not np.any(valid):
                continue

            # Flatten valid segment IDs in this tile
            seg_flat = seg_tile[valid].astype(np.int64, copy=False)

            # Map to compact indices (vectorized binary search)
            seg_idx = np.searchsorted(seg_ids, seg_flat)
            # Safety check (should always be true)
            if not np.all(seg_idx < N) or not np.all(seg_ids[seg_idx] == seg_flat):
                raise RuntimeError("Found a segment ID in pass 2 that wasn't seen in pass 1.")

            # Update counts ONCE per pixel (not per band)
            # bincount over compact indices -> length N
            counts_tile = np.bincount(seg_idx, minlength=N)
            count_all += counts_tile

            # Read feature tile (B, h, w) once; keep as float32 for memory, compute in float64 accumulators
            feat_tile = feat_src.read(window=window)
            if feat_tile.dtype != np.float32:
                feat_tile = safe_float32(feat_tile)

            # For each band, accumulate sum, sum2, min, max in compact arrays
            # We avoid Python loops over pixels by using np.add.at and reduction-at ops.
            for b in range(B):
                vals = feat_tile[b][valid]
                # Sums
                np.add.at(sum_all[b], seg_idx, vals)
                np.add.at(sum2_all[b], seg_idx, vals * vals)
                # Min/Max
                np.minimum.at(min_all[b], seg_idx, vals)
                np.maximum.at(max_all[b], seg_idx, vals)

        # --------------------------------
        # Finalize statistics
        # --------------------------------
        # Prevent division by zero
        denom = np.maximum(count_all, 1).astype(np.float64)
        mean_all = sum_all / denom  # (B, N)
        var_all = np.maximum(sum2_all / denom - mean_all**2, 0.0)
        std_all = np.sqrt(var_all)  # (B, N)

        # Keep only segments that actually occurred (count>0)
        keep = count_all > 0
        seg_ids_kept = seg_ids[keep]
        if not np.all(keep):
            mean_all = mean_all[:, keep]
            std_all  = std_all[:, keep]
            min_all  = min_all[:, keep]
            max_all  = max_all[:, keep]

        # Stack features to (N_kept, 4*B) in the same order: mean, std, min, max
        X_obj = np.hstack([
            mean_all.T.astype(np.float32, copy=False),
            std_all.T.astype(np.float32, copy=False),
            min_all.T.astype(np.float32, copy=False),
            max_all.T.astype(np.float32, copy=False),
        ])

        print(f"Feature matrix shape: {X_obj.shape}")
        print(f"Each segment now has {X_obj.shape[1]} features (mean, std, min, max × {B} bands).")

        # --------------------------------
        # Save HDF5 (compressed)
        # --------------------------------
        with h5py.File(out_h5, "w") as hf:
            hf.create_dataset("X_obj", data=X_obj, compression="gzip")
            hf.create_dataset("seg_ids", data=seg_ids_kept.astype(np.int64), compression="gzip")

        print(f"Saved object features: {out_h5}")
        print(f"  Segments: {len(seg_ids_kept)}, Features: {X_obj.shape[1]}")

if __name__ == "__main__":
    main(segments_tif, feature_tif, out_h5)