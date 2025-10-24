#  Samoa Land Cover Classification Workflow

This repository contains all scripts used for both **pixel-based** and **object-based (OB)** land cover classification of the Very High Resolution (VHR) imagery for **Samoa (2025)**, as part of the **Sen4Stat** project.

The workflow is modular and memory-friendly, using **HDF5** for intermediate storage and **streamed raster I/O** to handle large images efficiently.

---

## Project Structure


---

##  Environment

All scripts use:
- **Python ≥ 3.9**
- Libraries: `numpy`, `rasterio`, `geopandas`, `h5py`, `scikit-learn`, `xgboost`, `joblib`, `tqdm`, `scipy`

---

## Workflow Overview

The workflow is divided into two independent pipelines:

1. **Pixel-based classification**
2. **Object-based (OB) classification**

Both pipelines start from the same VHR data cube and in-situ training data but differ in how they aggregate features (pixel vs segment).

---

## Pixel-Based Classification

### Scripts

| Script | Purpose |
|--------|----------|
| **`S4S_insitu_ipynb`** | Jupyter notebook for cleaning and exploratory analysis of the received in-situ dataset. |
| **`classif_VHR_rasterization.py`** | Rasterizes the in-situ shapefile into a labeled raster matching the VHR grid. Splits it into calibration and validation rasters. |
| **`classif_VHR_datacube_and_indices.py`** | Builds a multi-band data cube (R, G, B, NIR) and computes **NDVI** and **NDWI**, producing a standardized spectral stack. |
| **`Extract_samples_samoa.py`** | Extracts training samples only where in-situ data are available and stores them in an `.h5` file (for memory efficiency). |
| **`classif_VHR_cal.py`** | Trains a pixel-based classifier (e.g., RandomForest or XGBoost). |
| **`classif_VHR_val.py`** | Validates the trained model using the validation set and computes performance metrics. |
| **`classif_VHR_pred.py`** | Applies the trained model to predict **pixel-wise classes** across the full ROI using streamed raster I/O. |
| *(optional)* **`datacube_standardization.py`** | Standardizes the datacube (band-wise normalization). Can also reuse the OB standardization script. |

---

##  Object-Based (OB) Classification

### Scripts

| Script | Purpose |
|--------|----------|
| **`Samoa_classif_OB_rasterize_segments.py`** | Rasterizes the segmentation shapefile (`.shp`) into a raster (`.tif`) where each pixel stores a segment ID. |
| **`Samoa_clipped_features.py`** | Optionally clips the segmentation and corresponding VHR data to a smaller ROI for testing. |
| **`Samoa_clipped_features_std_segm.py`** | Standardizes the 4-band VHR imagery (R, G, B, NIR). Can be reused in the pixel-based workflow. |
| **`Samoa_classif_OB_object_stats.py`** | Computes per-segment statistics (**mean, std, min, max**) for each band and stores them efficiently in an `.h5` file (`object_features_all.h5`). The script streams data tile-by-tile to stay memory-friendly. |
| **`Samoa_classif_ob_objects_labels.py`** | Assigns labels to each segment based on the in-situ raster using a **majority vote with purity filtering**. Creates calibration/validation `.h5` datasets for training. |
| **`Samoa_Classif_OB_val.py`** | Trains and validates an **XGBoost object-based model** using the segment features and assigned labels. |
| **`Samoa_classif_OB_pred.py`** | Applies the trained model to all segments in the ROI, producing a segment-based classification raster. Prediction is done in **chunks** for low memory usage. |
| **`Samoa_classif_OB_imp.py`** | Post-processes the OB classification map using NDVI and NDWI masks: <br> - **NDVI  0.48 - class 8111 (built-up / baresoil) **<br>  **NDWI > 0.11 -  class 9111 (water)**<br> Fully streamed and memory-safe. |

---

##  Data Flow Summary


---

##  Outputs

| Type | Example File | Description |
|------|---------------|-------------|
| **Pixel features** | `classif_VHR_OB/hdf5/pixel_features_cal.h5` | Pixel-level training data. |
| **Object features** | `classif_VHR_OB/hdf5/object_features_all.h5` | Per-segment statistics (mean, std, min, max). |
| **Training samples (object)** | `classif_VHR_OB/hdf5/object_all_cal.h5` | Object-level calibration dataset. |
| **Model files** | `classif_VHR_OB/model/xgb_object_model_V3.joblib` | Trained XGBoost model and label encoder. |
| **Predicted map (object)** | `classif_VHR_OB/VHR11_object_classif_all.tif` | Raw object-based classification map. |
| **Post-processed map** | `classif_VHR_OB/VHR11_object_classif_all_masked.tif` | Map after NDVI/NDWI-based masking (water & vegetation refinement). |

---

##  Notes

- All heavy computations (feature extraction, object stats, predictions) are done **tile-by-tile** with `rasterio.block_windows()` -  works on standard laptops.
- The `.h5` format stores compressed intermediate arrays, making it easy to load chunks without exceeding memory.
- The same NDVI and NDWI layers are used consistently for both pixel-based and object-based workflows.
- The OB workflow can be easily extended with additional indices or features (e.g., texture, entropy, GLCM).

---

##  Recommended Execution Order

**Pixel-based:**
1. `S4S_insitu_ipynb`
2. `classif_VHR_rasterization.py`
3. `classif_VHR_datacube_and_indices.py`
4. `Extract_samples_samoa.py`
5. `classif_VHR_cal.py`
6. `classif_VHR_val.py`
7. `classif_VHR_pred.py`

**Object-based:**
1. `Samoa_classif_OB_rasterize_segments.py`
2. `Samoa_clipped_features_std_segm.py`
3. `Samoa_classif_OB_object_stats.py`
4. `Samoa_classif_ob_objects_labels.py`
5. `Samoa_Classif_OB_val.py`
6. `Samoa_classif_OB_pred.py`
7. `Samoa_classif_OB_imp.py`

---

##  Tips

- Use **chunked HDF5 I/O** (`h5py`) for scalable training data.
- Keep consistent CRS and resolution across all rasters.
- When in doubt, use **WarpedVRT** to align rasters on the fly.
- All scripts are written to handle **large datasets** safely -  memory use typically stays below **500 MB** even on 40kx40k rasters.

---

**Author:** *Guillaume Jadot*  
**Project:** *Sen4Stat - Samoa 2025*  
**Last updated:** *October 2025*

