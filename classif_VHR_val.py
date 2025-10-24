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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

# =========================
# Paths and setup
# =========================
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
val_hdf5 = f"{computer_path}classif_VHR/hdf5/VHR11_features_V2_val.h5"
model_path = f"{computer_path}classif_VHR/model/VHR11_classifV3_V2.joblib"
encoder_path = model_path.replace(".joblib", "_encoder.joblib")

#out
cm_out = model_path.replace(".joblib", "_confusion_matrix_val.csv")
report_out = model_path.replace(".joblib", "_val_report.txt")

#%%
# =========================
# Load HDF5 data
# =========================

with h5py.File(val_hdf5, 'r') as h5f:
    X = h5f['X'][:]
    y_test = h5f['y'][:]
print(f"Loaded validation data: X={X.shape}, y={y_test.shape}")
print(f"Unique labels: {np.unique(y_test)}")

if y_test is None:
    print("No 'y' dataset found in HDF5.")
else:
    arr = np.asarray(y_test).ravel()
    unique, counts = np.unique(arr, return_counts=True)
    print("Counts:", dict(zip(unique.tolist(), counts.tolist())))

#%%

# =========================
# Load model and encoder
# =========================

model = load(model_path)
label_encoder = load(encoder_path)
print(" Model and encoder loaded.")

# =========================
# Encode labels
# =========================

y_test_encoded = label_encoder.transform(y_test)
class_names = label_encoder.classes_
print(f"Encoded {len(class_names)} unique classes.")



#%%
# =========================
# Predictions
# =========================

predictions_encoded = model.predict(X)
predictions = label_encoder.inverse_transform(predictions_encoded)

# =========================
# Metrics
# =========================

# Evaluate
accuracy = accuracy_score(y_test_encoded, predictions_encoded)
f1_macro = f1_score(y_test_encoded, predictions_encoded, average='macro')
bal_acc = balanced_accuracy_score(y_test_encoded, predictions_encoded)
kappa = cohen_kappa_score(y_test_encoded, predictions_encoded)
print(f"\n--- Validation Summary ---")
print(f"Overall accuracy:     {accuracy:.3f}")
print(f"Balanced accuracy:    {bal_acc:.3f}")
print(f"Macro F1-score:       {f1_macro:.3f}")
print(f"Cohen's Kappa:        {kappa:.3f}")

# =========================
# Detailed per-class report
# =========================
class_names = label_encoder.classes_.astype(str)
report = classification_report(y_test_encoded, predictions_encoded, target_names=class_names)
print("\nClassification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test_encoded, predictions_encoded)
print(f"Confusion matrix shape: {cm.shape}")


# Save results
np.savetxt(cm_out, cm, fmt="%d", delimiter=",", header=",".join(class_names))
with open(report_out, "w") as f:
    f.write(report)

print(f"\nSaved confusion matrix → {cm_out}")
print(f"Saved classification report → {report_out}")

# =========================
# Missing class check
# =========================
classes_all = sorted(np.unique(y_test_encoded))
classes_pred = sorted(np.unique(predictions_encoded))
classes_missing = set(classes_all) - set(classes_pred)

if classes_missing:
    missing_names = [label_encoder.inverse_transform([c])[0] for c in classes_missing]
    print(f"\nMissing predicted classes: {missing_names}")
else:
    print("\nAll validation classes predicted at least once.")


# # 6. Optional: map numeric crop codes to human-readable names
# class_data = {
#     1111: 'Taro', 1121: 'Taamu', 1192: 'Other Crops',
#     2132: 'Mix Crops (banana, taro, coconut, etc)', 2133: 'Banana',
#     2416: 'Breadfruit', 2417: 'Coconut', 2419: 'Cocoa',
#     3113: 'Grassland', 5111: 'Shrubland', 6999: 'Forest',
#     7211: 'Baresoil', 8111: 'Built-up', 9111: 'Water',
#     9121: 'Seashore', 9214: 'Clouds', 9311: 'Cloud shadows',
# }
