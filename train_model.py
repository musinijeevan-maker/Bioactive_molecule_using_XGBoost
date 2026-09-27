"""
train_model.py
==============
ML Training Pipeline for Bioactive Molecule Prediction

Experimental Design (following the base paper):
  - 70% Development set (10-fold Stratified CV + full training)
  - 30% Final Holdout set (untouched evaluation)

Models evaluated:
  1. XGBoost
  2. Random Forest
  3. Linear SVM
  4. Naive Bayes
  5. RBF Network (RBFSampler + SGDClassifier)

Selection:
  Highest Holdout F1-score (Holdout ROC-AUC as tie-breaker).
  Automatically saves best_model.pkl and metadata.json.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import xgboost as xgb

DATA_PATH = "CHEMBL230_Preprocessed_Data.csv"
MODEL_SAVE_DIR = os.path.join("backend", "model")


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Convert SMILES string into a 2048-bit Morgan circular fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


def evaluate_metrics(y_true, y_pred, y_proba=None):
    """Calculate Accuracy, Precision, Recall, Specificity, F1, and ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4),
    }


def run_10fold_cv(model_factory, X_dev, y_dev, skf):
    """
    Run 10-fold stratified CV sequentially.
    Safe and predictable on all platforms (avoids Windows multiprocessing deadlocks).
    """
    f1_list = []
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, X_val = X_dev[train_idx], X_dev[val_idx]
        y_tr, y_val = y_dev[train_idx], y_dev[val_idx]

        clf = model_factory()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        f1_list.append(f1_score(y_val, preds, zero_division=0))

    return float(np.mean(f1_list)), float(np.std(f1_list))


def main():
    print("=" * 65, flush=True)
    print("BIOACTIVE MOLECULE PREDICTOR - MACHINE LEARNING PIPELINE", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Data
    print("\n[1/6] Loading dataset from:", DATA_PATH, flush=True)
    df = pd.read_csv(DATA_PATH)
    print(f"      Rows: {len(df)}, Columns: {list(df.columns)}", flush=True)
    print(f"      Target distribution: {df['bioactivity_class'].value_counts().to_dict()}", flush=True)

    # 2. Molecular Feature Generation
    print("\n[2/6] Generating Morgan Fingerprints (RDKit, radius=2, 2048 bits)...", flush=True)
    fingerprints = []
    valid_indices = []
    for idx, row in df.iterrows():
        fp = smiles_to_fingerprint(str(row["smiles"]))
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)

    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    X = np.array(fingerprints)
    label_map = {"active": 1, "inactive": 0}
    y = df_valid["bioactivity_class"].map(label_map).values
    print(f"      Valid molecules: {len(X)} | Feature Matrix X shape: {X.shape}", flush=True)
    print(f"      Class counts: Active (1)={np.sum(y==1)}, Inactive (0)={np.sum(y==0)}", flush=True)

    # 3. 70% Development / 30% Holdout Split
    print("\n[3/6] Splitting data: 70% Development (for 10-fold CV) / 30% Holdout...", flush=True)
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"      Development set: {len(X_dev)} samples", flush=True)
    print(f"      Holdout set    : {len(X_hold)} samples", flush=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Define model factories for 10-fold CV and final fit
    models = {
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.15,
            eval_metric="logloss", random_state=42, verbosity=0
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Linear SVM": lambda: Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ]),
        "Naive Bayes": lambda: GaussianNB(),
        "RBF Network": lambda: Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("rbf_sampler", RBFSampler(gamma=0.01, n_components=500, random_state=42)),
            ("clf", SGDClassifier(loss="log_loss", max_iter=1000, random_state=42))
        ])
    }

    # 4. Train, 10-Fold CV & Evaluate Holdout
    print("\n[4/6] Executing 10-fold Cross-Validation & Final Evaluation on Holdout...", flush=True)
    results = []
    trained_models = {}

    for name, factory in models.items():
        print(f"\n  --> Evaluating: {name}", flush=True)

        # 10-fold CV on Development set
        cv_mean, cv_std = run_10fold_cv(factory, X_dev, y_dev, skf)
        print(f"      Mean 10-Fold CV F1: {cv_mean:.4f} (+/- {cv_std:.4f})", flush=True)

        # Train on full 70% Development set
        final_model = factory()
        final_model.fit(X_dev, y_dev)
        trained_models[name] = final_model

        # Evaluate on untouched 30% Holdout set
        y_pred = final_model.predict(X_hold)
        if hasattr(final_model, "predict_proba"):
            y_proba = final_model.predict_proba(X_hold)[:, 1]
        elif hasattr(final_model, "decision_function"):
            scores = final_model.decision_function(X_hold)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_proba = None

        metrics = evaluate_metrics(y_hold, y_pred, y_proba)
        metrics["model"] = name
        metrics["cv_f1_mean"] = round(cv_mean, 4)
        metrics["cv_f1_std"] = round(cv_std, 4)
        results.append(metrics)

        print(f"      Holdout Accuracy : {metrics['accuracy']:.4f}", flush=True)
        print(f"      Holdout F1-Score : {metrics['f1']:.4f}", flush=True)
        print(f"      Holdout ROC-AUC  : {metrics['auc']:.4f}", flush=True)

    # 5. Model Comparison Table
    print("\n" + "=" * 65, flush=True)
    print("STEP 5 - MODEL COMPARISON TABLE (Final Holdout Evaluation)", flush=True)
    print("=" * 65, flush=True)

    res_df = pd.DataFrame(results)
    table_cols = ["model", "cv_f1_mean", "accuracy", "precision", "recall", "specificity", "f1", "auc"]
    res_df = res_df[table_cols]
    res_df.columns = ["Model", "Mean CV F1", "Accuracy", "Precision", "Recall", "Specificity", "Holdout F1", "ROC-AUC"]
    print(res_df.to_string(index=False), flush=True)

    # 6. Automatic Best Model Selection
    print("\n" + "=" * 65, flush=True)
    print("STEP 6 - AUTOMATIC BEST MODEL SELECTION", flush=True)
    print("=" * 65, flush=True)
    print("Selection Criteria: Highest Holdout F1-Score -> Tie-breaker: Highest ROC-AUC", flush=True)

    sorted_df = res_df.sort_values(by=["Holdout F1", "ROC-AUC"], ascending=[False, False]).reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    best_name = best_row["Model"]

    print("\n" + "*" * 40, flush=True)
    print(f"*** WINNING MODEL: {best_name} ***", flush=True)
    print(f"    Holdout F1-Score : {best_row['Holdout F1']:.4f}", flush=True)
    print(f"    Holdout ROC-AUC  : {best_row['ROC-AUC']:.4f}", flush=True)
    print(f"    Holdout Accuracy : {best_row['Accuracy']:.4f}", flush=True)
    print(f"    Mean 10-Fold CV  : {best_row['Mean CV F1']:.4f}", flush=True)
    print("*" * 40, flush=True)

    # Save best model and metadata
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_file = os.path.join(MODEL_SAVE_DIR, "best_model.pkl")
    joblib.dump(trained_models[best_name], model_file)
    print(f"\nSaved best model to: {model_file}", flush=True)

    metadata = {
        "best_model_name": best_name,
        "label_mapping": label_map,
        "reverse_mapping": {"1": "active", "0": "inactive"},
        "fingerprint_radius": 2,
        "fingerprint_bits": 2048,
        "split": "70% development / 30% holdout",
        "cv_folds": 10,
        "selection_metric": "Holdout F1-score",
        "best_metrics": {
            "cv_f1_mean": float(best_row["Mean CV F1"]),
            "holdout_f1": float(best_row["Holdout F1"]),
            "accuracy": float(best_row["Accuracy"]),
            "auc": float(best_row["ROC-AUC"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
            "specificity": float(best_row["Specificity"])
        },
        "all_results": res_df.to_dict(orient="records")
    }

    meta_file = os.path.join(MODEL_SAVE_DIR, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_file}", flush=True)
    print("\nPipeline execution complete!\n", flush=True)


if __name__ == "__main__":
    main()
