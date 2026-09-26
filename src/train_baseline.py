"""
train_baseline.py: Trains Logistic Regression baseline model, tunes threshold, and evaluates F0.5.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from src.config import DATA_DIR, MODELS_DIR, EXPERIMENTS_CSV
from src.evaluate import evaluate_macro_f05

def train_baseline_model(
    pair_features_csv: str = None,
    save_model: bool = True
):
    if pair_features_csv is None:
        pair_features_csv = os.path.join(DATA_DIR, "processed", "pair_features.csv")

    print(f"Loading pair features from {pair_features_csv}...")
    df = pd.read_csv(pair_features_csv)
    print(f"Loaded {len(df)} total pairs.")

    feature_cols = [
        c for c in df.columns
        if c not in ["source1_id", "candidate_id", "candidate_source", "split_group", "label"]
    ]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    train_df = df[df["split_group"] == "train"].copy()
    val_df = df[df["split_group"] == "val"].copy()

    print(f"Train pairs: {len(train_df)} (Positives: {(train_df['label'] == 1).sum()})")
    print(f"Val pairs:   {len(val_df)} (Positives: {(val_df['label'] == 1).sum()})")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training Logistic Regression baseline...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    val_df["pred_prob"] = val_probs

    # Build ground truth mapping for validation S1 entities
    val_gt = defaultdict(set)
    for _, row in val_df.iterrows():
        if row["label"] == 1:
            val_gt[row["source1_id"]].add(row["candidate_id"])
        else:
            if row["source1_id"] not in val_gt:
                val_gt[row["source1_id"]] = set()

    # Threshold tuning sweep on validation set
    print("\nTuning probability threshold for F0.5...")
    best_thresh = 0.50
    best_macro_f05 = -1.0
    best_metrics = {}

    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for th in thresholds:
        preds = defaultdict(set)
        for _, row in val_df.iterrows():
            if row["pred_prob"] >= th:
                preds[row["source1_id"]].add(row["candidate_id"])

        metrics = evaluate_macro_f05(val_gt, preds)
        print(f"Threshold: {th:.2f} | Macro F0.5: {metrics['macro_f05']:.4f} | Prec: {metrics['macro_precision']:.4f} | Rec: {metrics['macro_recall']:.4f} | TP: {metrics['total_tp']} | FP: {metrics['total_fp']} | FN: {metrics['total_fn']}")

        if metrics["macro_f05"] > best_macro_f05:
            best_macro_f05 = metrics["macro_f05"]
            best_thresh = th
            best_metrics = metrics

    print(f"\n=== BASELINE RESULTS ===")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Validation Macro F0.5:     {best_metrics['macro_f05']:.4f}")
    print(f"Validation Macro Precision: {best_metrics['macro_precision']:.4f}")
    print(f"Validation Macro Recall:    {best_metrics['macro_recall']:.4f}")
    print(f"Validation Micro F0.5:     {best_metrics['micro_f05']:.4f}")

    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "baseline_logistic.joblib")
        joblib.dump({
            "model": clf,
            "scaler": scaler,
            "features": feature_cols,
            "threshold": best_thresh,
            "metrics": best_metrics
        }, model_path)
        print(f"Saved baseline model to {model_path}")

    # Track in experiments CSV
    os.makedirs(os.path.dirname(EXPERIMENTS_CSV), exist_ok=True)
    exp_record = {
        "experiment": "baseline_logistic",
        "features": f"{len(feature_cols)}_engineered_features",
        "model": "LogisticRegression",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "precision": f"{best_metrics['macro_precision']:.4f}",
        "recall": f"{best_metrics['macro_recall']:.4f}",
        "f0_5": f"{best_metrics['macro_f05']:.4f}",
        "threshold": f"{best_thresh:.2f}",
        "notes": "Standardized pairwise features with threshold sweep"
    }
    
    write_header = not os.path.exists(EXPERIMENTS_CSV)
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = pd.DataFrame([exp_record])
        writer.to_csv(f, header=write_header, index=False)

    return clf, scaler, best_thresh, best_metrics

if __name__ == "__main__":
    train_baseline_model()
