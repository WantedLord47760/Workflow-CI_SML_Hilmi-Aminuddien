"""
modelling.py  (untuk MLflow Project / CI)
-----------------------------------------
Script training yang digunakan oleh MLflow Project dan GitHub Actions CI.
Menerima argumen dari command line (entry point MLProject).

Penulis : Hilmi Aminuddien
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================
parser = argparse.ArgumentParser(description="Train Heart Disease model")
parser.add_argument("--n_estimators",      type=int,   default=100)
parser.add_argument("--max_depth",         type=int,   default=10)
parser.add_argument("--min_samples_split", type=int,   default=2)
parser.add_argument("--min_samples_leaf",  type=int,   default=1)
parser.add_argument("--random_state",      type=int,   default=42)
args = parser.parse_args()

# ==============================================================================
# LOAD DATA
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "heart_preprocessing")

print("[INFO] Memuat data preprocessing...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

# ==============================================================================
# TRAINING + MLFLOW LOGGING
# ==============================================================================
mlflow.set_experiment("Heart-Disease-CI")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("random_state",      args.random_state)

    # Training
    model = RandomForestClassifier(
        n_estimators      = args.n_estimators,
        max_depth         = args.max_depth,
        min_samples_split = args.min_samples_split,
        min_samples_leaf  = args.min_samples_leaf,
        random_state      = args.random_state,
        n_jobs            = -1
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"[HASIL] Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")
    print("[SUKSES] Model berhasil disimpan ke MLflow.")
