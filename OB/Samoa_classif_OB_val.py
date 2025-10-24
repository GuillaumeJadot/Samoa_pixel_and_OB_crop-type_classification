import os
import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb
from joblib import dump, load

# --- Paths ---
computer_path = "/export/projects/Sen4Stat/COUNTRIES/Samoa/2025/"
train_h5 = f"{computer_path}classif_VHR_OB/hdf5/object_all_cal.h5"
val_h5   = f"{computer_path}classif_VHR_OB/hdf5/object_all_val.h5"

# -- out 
model_path   = f"{computer_path}classif_VHR_OB/model/xgb_object_model_all.joblib"
encoder_path = model_path.replace(".joblib", "_encoder.joblib")
metrics_path = model_path.replace(".joblib", "_metrics.csv")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# --- Load data ---
with h5py.File(train_h5, "r") as hf:
    X_train_obj = hf["X_train_obj"][:]
    y_train_obj = hf["y_train_obj"][:]

with h5py.File(val_h5, "r") as hf:
    X_val_obj = hf["X_train_obj"][:]
    y_val_obj = hf["y_train_obj"][:]

print(f" Training objects: {X_train_obj.shape}, Validation: {X_val_obj.shape}")

# === Encode labels on TRAIN ONLY ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_obj)

# Diagnostics
uniq_train = np.unique(y_train_enc)
print(f"Train unique encoded labels: {uniq_train} (count={len(uniq_train)})")
print(f"Original train classes (le.classes_): {le.classes_} (count={len(le.classes_)})")

# Map VAL labels into the train encoding
# Samples whose labels were never seen in training will map to -1 and be filtered for evaluation.
val_label_to_enc = {cls: i for i, cls in enumerate(le.classes_)}
y_val_enc = np.array([val_label_to_enc.get(lbl, -1) for lbl in y_val_obj], dtype=np.int32)

num_unseen_in_val = np.sum(y_val_enc == -1)
if num_unseen_in_val > 0:
    unseen_vals = np.unique(y_val_obj[y_val_enc == -1])
    print(f" Validation contains {num_unseen_in_val} samples from unseen classes: {unseen_vals}")
else:
    print("Validation has no unseen-in-train classes.")

# Keep only validation rows with known (seen-in-train) classes for scoring
val_keep = (y_val_enc != -1)
X_val_known = X_val_obj[val_keep]
y_val_known = y_val_enc[val_keep]
print(f"Validation kept for scoring: {X_val_known.shape[0]} / {X_val_obj.shape[0]}")

# Class weights on train
w = compute_sample_weight(class_weight='balanced', y=y_train_enc)

# === Model ===
if not os.path.exists(model_path):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=0,
    )

    print(" Training XGBoost model on object features...")
    clf.fit(X_train_obj, y_train_enc, sample_weight=w)
    print(" Model trained successfully.")
    # Save model + encoder
    dump(clf, model_path)
    dump(le, encoder_path)
    print(f"\n Model saved to: {model_path}")
    print(f" Encoder saved to: {encoder_path}")
else:
    print(f" Loading existing model from: {model_path}")
    # clf = xgb.XGBClassifier()
    clf = load(model_path)

# === Evaluate (only on known classes) ===
if X_val_known.shape[0] > 0:
    y_pred_known = clf.predict(X_val_known)

    bal_acc = balanced_accuracy_score(y_val_known, y_pred_known)
    kappa   = cohen_kappa_score(y_val_known, y_pred_known)
    report  = classification_report(
        y_val_known, y_pred_known,
        target_names=le.classes_.astype(str),  # names align with encoded order
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df["balanced_accuracy"] = bal_acc
    report_df["cohen_kappa"] = kappa

    print("\n Object-level validation (seen classes only):")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"Cohen’s Kappa: {kappa:.3f}")
    print(report_df[["precision", "recall", "f1-score"]].head(10))

    cm = confusion_matrix(y_val_known, y_pred_known)
    print("\nConfusion matrix:")
    print(cm)

    # Save metrics
    # Save confusion matrix with class labels (rows = true, cols = pred)
    labels_present = np.unique(np.concatenate([y_val_known, y_pred_known]))
    class_names = le.classes_[labels_present].astype(str)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(metrics_path.replace(".csv", "_confusion_matrix.csv"), index=True)
    print(f" Confusion matrix saved to: {metrics_path.replace('.csv', '_confusion_matrix.csv')}")
    report_df.to_csv(metrics_path, index=True)
    print(f" Metrics saved to: {metrics_path}")
else:
    print("\n  No validation samples with seen-in-train classes; skipping metrics.")
