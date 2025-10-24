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

# =========================
# Paths and parameters
# =========================
#in
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
cal_hdf5 = f"{computer_path}classif_VHR/hdf5/VHR11_features_V2_cal.h5"
#out
model_path = f"{computer_path}classif_VHR/model/VHR11_classifV3_V2.joblib"
Path.Path(model_path).parent.mkdir(parents=True, exist_ok=True)
encoder_path = model_path.replace(".joblib", "_encoder.joblib")
metrics_path = model_path.replace(".joblib", "_metrics_training.csv")

nodata = -9999
#%%

# =========================
# Load HDF5 data
# =========================

with h5py.File(cal_hdf5, 'r') as h5f:
    X = h5f['X'][:]
    labels = h5f['y'][:]
print(f"Loaded training data: X={X.shape}, y={labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

if labels is None:
    print("No 'y' dataset found in HDF5.")
else:
    arr = np.asarray(labels).ravel()
    unique, counts = np.unique(arr, return_counts=True)
    print("Counts:", dict(zip(unique.tolist(), counts.tolist())))

#%%
# =========================
# Encode labels
# =========================

label_encoder = LabelEncoder()
train_labels= label_encoder.fit_transform(labels)
class_names = label_encoder.classes_
print(f"Encoded {len(class_names)} unique classes.")

# =========================
# Handle class imbalance
# =========================

sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)
print(" Computed class-balanced sample weights.")

#%%

# =========================
# Train XGBoost
# =========================

if os.path.exists(model_path): #and os.path.exists(scaler_path):
    print("Model and scaler already exist. Loading...")
    model = load(model_path)
    print(f'Model loaded: {model}')
    scaler = load(model_path.replace(".joblib", "_scaler.joblib"))
    label_encoder = load(model_path.replace(".joblib", "_encoder.joblib"))
else:
    print("Training the model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,  # Base was 0.05
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        objective='multi:softprob', # 'multi:softmax', # get class probabilities, not just labels
        num_class=len(np.unique(train_labels)),  # Adjust according to your unique classes
        use_label_encoder=False,
        eval_metric='mlogloss', 
        tree_method="hist",          # faster, memory-efficient
        reg_lambda=1.0,              # L2 regularization to reduce overfitting
    )
    # model.fit(X, labels)
    print("Training XGBoost with class-balanced weights...")
    model.fit(X, train_labels, sample_weight=sample_weights)
    print("Model trained successfully")
    # Save everything
    dump(model, model_path)
    dump(label_encoder, model_path.replace(".joblib", "_encoder.joblib"))
    print("Model and encoder saved.")

# =========================
# Evaluate performance
# =========================

print("\n Evaluating on training data (for diagnostics)...")
preds_encoded = model.predict(X)
bal_acc = balanced_accuracy_score(train_labels, preds_encoded)
kappa = cohen_kappa_score(train_labels, preds_encoded)

# Decode predictions back to original label codes
preds = label_encoder.inverse_transform(preds_encoded)

# Classification report per class
report = classification_report(train_labels, preds_encoded, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df["balanced_accuracy"] = bal_acc
report_df["cohen_kappa"] = kappa

# Confusion matrix
cm = confusion_matrix(train_labels, preds_encoded)

print(f"\nBalanced Accuracy: {bal_acc:.3f}")
print(f"Cohen’s Kappa: {kappa:.3f}")

# Save metrics
report_df.to_csv(metrics_path, index=True)
print(f"Saved classification report to: {metrics_path}")

# =========================
# Optional: Display summary
# =========================
print("\nTop-level summary:")
print(report_df[["precision", "recall", "f1-score"]].head(10))

# =========================
# Confusion matrix summary
# =========================
print("\nConfusion Matrix (first 10 classes):")
print(cm[:10, :10])