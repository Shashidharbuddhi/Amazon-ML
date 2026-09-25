"""
build_training_pairs.py: Constructs labeled training and validation pairs with realistic hard negatives.
"""
import os
import random
import csv
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    DATA_DIR
)
from src.schema import DELIM
from src.normalization import (
    normalize_name, normalize_address, normalize_country
)
from src.features import build_pair_features
from src.split_data import get_entity_split_group

def load_ground_truth(path: str, max_records: int = None) -> Dict[str, Set[str]]:
    """Load ground truth mapping {s1_id: set(matched_s2_or_s3_ids)}."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            parts = line.rstrip("\n").split(DELIM)
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if matched_str.strip():
                gt[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}
            else:
                gt[s1_id] = set()
    return gt

def build_training_dataset(
    num_s1_samples: int = 40000,
    neg_to_pos_ratio: float = 1.5,
    seed: int = 42,
    output_csv: str = None
):
    """
    Build labeled pairwise training dataset with ground truth positives and mined hard negatives.
    Uses grouped splitting (by S1 ID) to prevent data leakage.
    """
    random.seed(seed)
    print(f"Loading ground truth for training pair construction...")
    all_gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    # Sample S1 IDs that have matches plus some singletons
    s1_with_matches = [s1 for s1, matches in all_gt.items() if len(matches) > 0]
    singletons = [s1 for s1, matches in all_gt.items() if len(matches) == 0]
    
    random.shuffle(s1_with_matches)
    random.shuffle(singletons)
    
    # Take 95% matching S1s and 5% singletons (matching actual dataset distribution)
    n_pos_s1 = int(num_s1_samples * 0.95)
    n_zero_s1 = num_s1_samples - n_pos_s1
    selected_s1 = set(s1_with_matches[:n_pos_s1] + singletons[:n_zero_s1])
    
    print(f"Selected {len(selected_s1)} Source 1 entities ({n_pos_s1} with matches, {n_zero_s1} singletons).")

    # Collect needed positive candidate IDs
    needed_s2 = set()
    needed_s3 = set()
    for s1 in selected_s1:
        for mid in all_gt.get(s1, set()):
            if mid.startswith("S2-"):
                needed_s2.add(mid)
            elif mid.startswith("S3-"):
                needed_s3.add(mid)

    print(f"Needed positive candidates: {len(needed_s2)} S2, {len(needed_s3)} S3.")

    # Read selected S1 records
    print("Reading selected S1 records...")
    s1_records = {}
    with open(TRAIN_SOURCE1, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            s1_id = parts[0]
            if s1_id in selected_s1:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                s1_records[s1_id] = {
                    "name": name,
                    "addr": addr,
                    "country": country,
                    "norm_name": normalize_name(name),
                    "norm_addr": normalize_address(addr),
                    "country_norm": normalize_country(country),
                }

    # Build token index on S1 for hard negative mining
    print("Building token index on S1 for hard negative mining...")
    token_to_s1 = defaultdict(list)
    for s1_id, rec in s1_records.items():
        # First word of name and numeric tokens from address
        tokens = rec["norm_name"].split()[:2]
        for t in tokens:
            if len(t) >= 4:
                token_to_s1[t].append(s1_id)

    # Read needed S2 records + mine hard negatives
    s2_records = {}
    hard_negatives = defaultdict(set)
    print("Reading S2 records and mining hard negatives...")
    with open(TRAIN_SOURCE2, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            s2_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            
            is_needed = s2_id in needed_s2
            norm_name = None
            
            # Check for hard negative overlap if we still need negatives
            if not is_needed:
                first_word = name.strip().split()[0].lower() if name.strip() else ""
                if first_word and len(first_word) >= 4 and first_word in token_to_s1:
                    candidates_s1 = token_to_s1[first_word]
                    for s1_id in candidates_s1[:3]:
                        if len(hard_negatives[s1_id]) < 4 and s2_id not in all_gt.get(s1_id, set()):
                            hard_negatives[s1_id].add(s2_id)
                            is_needed = True
                            break

            if is_needed:
                s2_records[s2_id] = {
                    "name": name,
                    "addr": addr,
                    "country": country,
                    "norm_name": normalize_name(name),
                    "norm_addr": normalize_address(addr),
                    "country_norm": normalize_country(country),
                }

    # Read needed S3 records + mine hard negatives
    s3_records = {}
    print("Reading S3 records and mining hard negatives...")
    with open(TRAIN_SOURCE3, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            s3_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            
            is_needed = s3_id in needed_s3
            if not is_needed:
                first_word = name.strip().split()[0].lower() if name.strip() else ""
                if first_word and len(first_word) >= 4 and first_word in token_to_s1:
                    candidates_s1 = token_to_s1[first_word]
                    for s1_id in candidates_s1[:3]:
                        if len(hard_negatives[s1_id]) < 4 and s3_id not in all_gt.get(s1_id, set()):
                            hard_negatives[s1_id].add(s3_id)
                            is_needed = True
                            break

            if is_needed:
                s3_records[s3_id] = {
                    "name": name,
                    "addr": addr,
                    "country": country,
                    "norm_name": normalize_name(name),
                    "norm_addr": normalize_address(addr),
                    "country_norm": normalize_country(country),
                }

    print(f"Loaded {len(s1_records)} S1, {len(s2_records)} S2, {len(s3_records)} S3 records.")

    # Now construct labeled pairs
    print("Constructing labeled feature pairs...")
    if output_csv is None:
        os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
        output_csv = os.path.join(DATA_DIR, "processed", "pair_features.csv")

    fieldnames = [
        "source1_id", "candidate_id", "candidate_source", "split_group", "label"
    ]
    
    # Sample feature dict to get feature column names
    sample_feat = build_pair_features("test", "test", "US", "test", "test", "US")
    feat_cols = list(sample_feat.keys())
    fieldnames.extend(feat_cols)

    total_pairs = 0
    total_pos = 0
    total_neg = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for s1_id, s1_rec in s1_records.items():
            split_group = get_entity_split_group(s1_id)
            true_matches = all_gt.get(s1_id, set())

            # 1. Positives
            for mid in true_matches:
                cand_rec = s2_records.get(mid) if mid.startswith("S2-") else s3_records.get(mid)
                if cand_rec is None:
                    continue
                feats = build_pair_features(
                    s1_rec["name"], s1_rec["addr"], s1_rec["country"],
                    cand_rec["name"], cand_rec["addr"], cand_rec["country"],
                    s1_norm_name=s1_rec["norm_name"], s1_norm_addr=s1_rec["norm_addr"],
                    cand_norm_name=cand_rec["norm_name"], cand_norm_addr=cand_rec["norm_addr"]
                )
                row = {
                    "source1_id": s1_id,
                    "candidate_id": mid,
                    "candidate_source": "source2" if mid.startswith("S2-") else "source3",
                    "split_group": split_group,
                    "label": 1
                }
                row.update(feats)
                writer.writerow(row)
                total_pos += 1
                total_pairs += 1

            # 2. Hard Negatives
            cand_neg_ids = hard_negatives.get(s1_id, set())
            for nid in cand_neg_ids:
                cand_rec = s2_records.get(nid) if nid.startswith("S2-") else s3_records.get(nid)
                if cand_rec is None:
                    continue
                feats = build_pair_features(
                    s1_rec["name"], s1_rec["addr"], s1_rec["country"],
                    cand_rec["name"], cand_rec["addr"], cand_rec["country"],
                    s1_norm_name=s1_rec["norm_name"], s1_norm_addr=s1_rec["norm_addr"],
                    cand_norm_name=cand_rec["norm_name"], cand_norm_addr=cand_rec["norm_addr"]
                )
                row = {
                    "source1_id": s1_id,
                    "candidate_id": nid,
                    "candidate_source": "source2" if nid.startswith("S2-") else "source3",
                    "split_group": split_group,
                    "label": 0
                }
                row.update(feats)
                writer.writerow(row)
                total_neg += 1
                total_pairs += 1

    print(f"\nFinished building training pairs!")
    print(f"Total pairs written: {total_pairs} (Positives: {total_pos}, Negatives: {total_neg}, Pos rate: {total_pos/max(1, total_pairs):.2%})")
    print(f"Output saved to: {output_csv}")
    return output_csv

if __name__ == "__main__":
    build_training_dataset(num_s1_samples=40000)
