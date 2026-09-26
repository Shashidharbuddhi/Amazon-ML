"""
error_analysis.py: Detailed false positive and false negative error analysis.
"""
import os
import joblib
import pandas as pd
import numpy as np

from src.config import DATA_DIR, MODELS_DIR

def run_error_analysis(
    model_path: str = None,
    pair_features_csv: str = None,
    num_samples: int = 10
):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "tree_model_lgbm.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "baseline_logistic.joblib")

    if pair_features_csv is None:
        pair_features_csv = os.path.join(DATA_DIR, "processed", "pair_features.csv")

    print(f"Loading artifact from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    threshold = artifact["threshold"]
    scaler = artifact.get("scaler", None)

    print(f"Loading pair features from {pair_features_csv}...")
    df = pd.read_csv(pair_features_csv)
    val_df = df[df["split_group"] == "val"].copy()

    X_val = val_df[feature_cols].values
    if scaler is not None:
        X_val = scaler.transform(X_val)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_df["pred_prob"] = val_probs
    val_df["pred_label"] = (val_probs >= threshold).astype(int)

    # Classify into TP, FP, TN, FN
    val_df["category"] = "TN"
    val_df.loc[(val_df["label"] == 1) & (val_df["pred_label"] == 1), "category"] = "TP"
    val_df.loc[(val_df["label"] == 0) & (val_df["pred_label"] == 1), "category"] = "FP"
    val_df.loc[(val_df["label"] == 1) & (val_df["pred_label"] == 0), "category"] = "FN"

    counts = val_df["category"].value_counts().to_dict()
    print(f"\nValidation Confusion Counts at threshold {threshold:.2f}:")
    print(counts)

    # Inspect False Positives (high probability but label 0)
    fp_df = val_df[val_df["category"] == "FP"].sort_values("pred_prob", ascending=False)
    print(f"\n=== TOP FALSE POSITIVES (Model predicted Match, Ground Truth = No Match) ===")
    print(f"Total FP: {len(fp_df)}")
    cols_to_show = ["source1_id", "candidate_id", "pred_prob", "name_ratio", "name_token_set_ratio", "address_ratio", "country_match"]
    if not fp_df.empty:
        print(fp_df[cols_to_show].head(num_samples).to_string(index=False))

    # Inspect False Negatives (low probability but label 1)
    fn_df = val_df[val_df["category"] == "FN"].sort_values("pred_prob", ascending=True)
    print(f"\n=== TOP FALSE NEGATIVES (Model predicted No Match, Ground Truth = Match) ===")
    print(f"Total FN: {len(fn_df)}")
    if not fn_df.empty:
        print(fn_df[cols_to_show].head(num_samples).to_string(index=False))

    # Inspect True Positives
    tp_df = val_df[val_df["category"] == "TP"].sort_values("pred_prob", ascending=False)
    print(f"\n=== SAMPLE TRUE POSITIVES ===")
    if not tp_df.empty:
        print(tp_df[cols_to_show].head(5).to_string(index=False))

if __name__ == "__main__":
    run_error_analysis()
