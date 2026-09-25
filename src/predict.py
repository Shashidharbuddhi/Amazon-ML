"""
predict.py: Candidate generation and pairwise ML scoring for test set.
Generates output/matching_results.tsv and output/candidate_pairs.tsv adhering to official specifications.
"""
import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.config import (
    BASE_DIR, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, MODELS_DIR
)
from src.schema import DELIM
from src.normalization import (
    normalize_name, normalize_address, normalize_country, extract_numeric_tokens
)
from src.features import build_pair_features
from src.blocking import extract_blocking_keys

def run_prediction_pipeline(
    model_path: str = None,
    threshold: float = None,
    chunk_size: int = 100000
):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "tree_model_lgbm.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "baseline_logistic.joblib")

    print(f"Loading trained matcher artifact from: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    scaler = artifact.get("scaler", None)
    
    if threshold is None:
        threshold = artifact.get("threshold", 0.70)
    print(f"Using classification threshold: {threshold:.2f}")

    os.makedirs(os.path.dirname(MATCHING_RESULTS_PATH), exist_ok=True)

    # 1. Read Test S1 IDs in original sequence so output matches order
    print("Reading Test Source 1 entities...")
    s1_ids_ordered = []
    s1_by_country = defaultdict(list)
    with open(TEST_SOURCE1, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = normalize_country(parts[3] if len(parts) > 3 else "")
            s1_ids_ordered.append(s1_id)
            s1_by_country[country].append({
                "id": s1_id,
                "name": name,
                "addr": addr,
                "country": country,
                "norm_name": normalize_name(name),
                "norm_addr": normalize_address(addr),
            })

    total_s1 = len(s1_ids_ordered)
    print(f"Total Test S1 entities: {total_s1} across countries: {list(s1_by_country.keys())}")

    # Results dictionary {s1_id: set_of_matched_ids} and candidates dictionary {s1_id: set_of_candidate_ids}
    final_matches = defaultdict(list)
    final_candidates = defaultdict(list)

    # Process country-by-country to keep memory footprint under control
    for country, s1_records in s1_by_country.items():
        print(f"\n==========================================")
        print(f"Processing Country: {country} ({len(s1_records)} S1 entities)...")

        # Build candidate blocking index for this country from Test S2 and Test S3
        print(f"Building blocking index for {country} from Test S2 & S3...")
        block_index = defaultdict(list)
        cand_store = {}

        # Scan Test S2 for this country
        with open(TEST_SOURCE2, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split(DELIM)
                c_val = normalize_country(parts[3] if len(parts) > 3 else "")
                if c_val != country:
                    continue
                cand_id = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                norm_n = normalize_name(name)
                norm_a = normalize_address(addr)

                cand_store[cand_id] = (name, addr, c_val, norm_n, norm_a)
                keys = extract_blocking_keys(norm_n, norm_a, c_val)
                for k_type, k_val in keys:
                    block_index[k_val].append(cand_id)

        # Scan Test S3 for this country
        with open(TEST_SOURCE3, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split(DELIM)
                c_val = normalize_country(parts[3] if len(parts) > 3 else "")
                if c_val != country:
                    continue
                cand_id = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                norm_n = normalize_name(name)
                norm_a = normalize_address(addr)

                cand_store[cand_id] = (name, addr, c_val, norm_n, norm_a)
                keys = extract_blocking_keys(norm_n, norm_a, c_val)
                for k_type, k_val in keys:
                    block_index[k_val].append(cand_id)

        print(f"Index built: {len(cand_store)} candidates, {len(block_index)} blocking keys.")

        # Score candidates for each S1 entity in this country
        batch_pairs = []
        batch_s1_ids = []
        batch_cand_ids = []
        
        matched_count = 0
        total_cand_count = 0

        for i, s1_rec in enumerate(s1_records):
            s1_id = s1_rec["id"]
            keys = extract_blocking_keys(s1_rec["norm_name"], s1_rec["norm_addr"], country)
            
            # Retrieve unique candidates
            cand_id_set = set()
            for k_type, k_val in keys:
                if k_val in block_index:
                    for cid in block_index[k_val][:20]:
                        cand_id_set.add(cid)
                        if len(cand_id_set) >= 30:
                            break
                if len(cand_id_set) >= 30:
                    break

            if not cand_id_set:
                continue

            for cid in cand_id_set:
                c_data = cand_store.get(cid)
                if not c_data:
                    continue
                c_name, c_addr, c_country, c_norm_n, c_norm_a = c_data
                feats = build_pair_features(
                    s1_rec["name"], s1_rec["addr"], s1_rec["country"],
                    c_name, c_addr, c_country,
                    s1_norm_name=s1_rec["norm_name"], s1_norm_addr=s1_rec["norm_addr"],
                    cand_norm_name=c_norm_n, cand_norm_addr=c_norm_a
                )
                feat_vec = [feats[col] for col in feature_cols]
                batch_pairs.append(feat_vec)
                batch_s1_ids.append(s1_id)
                batch_cand_ids.append(cid)

            # Score in batches to maximize vectorization speed
            if len(batch_pairs) >= 50000 or i == len(s1_records) - 1:
                if batch_pairs:
                    X_arr = np.array(batch_pairs, dtype=np.float32)
                    if scaler is not None:
                        X_arr = scaler.transform(X_arr)
                    probs = model.predict_proba(X_arr)[:, 1]

                    for s1_curr, cid_curr, p_val in zip(batch_s1_ids, batch_cand_ids, probs):
                        final_candidates[s1_curr].append(cid_curr)
                        total_cand_count += 1
                        if p_val >= threshold:
                            final_matches[s1_curr].append(cid_curr)
                            matched_count += 1

                    batch_pairs = []
                    batch_s1_ids = []
                    batch_cand_ids = []

        print(f"Completed {country}: {total_cand_count} candidates scored, {matched_count} matches identified.")
        del block_index
        del cand_store
        gc.collect()

    # Write output files
    print("\nWriting final submission files...")
    print(f"Writing matching results to {MATCHING_RESULTS_PATH}...")
    with open(MATCHING_RESULTS_PATH, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids_ordered:
            m_list = final_matches.get(s1_id, [])
            # Deduplicate while preserving order
            unique_m = list(dict.fromkeys(m_list))
            f_match.write(f"{s1_id}\t{','.join(unique_m)}\n")

    print(f"Writing candidate pairs to {CANDIDATE_PAIRS_PATH}...")
    with open(CANDIDATE_PAIRS_PATH, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_ordered:
            c_list = final_candidates.get(s1_id, [])
            unique_c = list(dict.fromkeys(c_list))
            f_cand.write(f"{s1_id}\t{','.join(unique_c)}\n")

    print("\nPrediction generation complete!")

if __name__ == "__main__":
    run_prediction_pipeline()
