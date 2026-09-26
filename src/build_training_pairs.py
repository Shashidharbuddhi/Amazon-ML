"""
build_training_pairs.py: Fast, memory-conscious training pair construction using Person 1 candidate generation logic.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from src.blocking import CandidatePair, country_name_signature_key, exact_country_address_key
from src.config import DATA_DIR
from src.data_loader import EntityRecord, discover_split, iter_entities, iter_ground_truth
from src.features import build_pair_features
from src.split_data import get_entity_split_group


def load_ground_truth_dict(path: Path) -> dict[str, set[str]]:
    """Load ground truth mapping {s1_id: set(matched_ids)}."""
    gt: dict[str, set[str]] = {}
    for rec in iter_ground_truth(path):
        gt[rec.source1_entity_id] = set(rec.matched_entity_ids)
    return gt


def build_training_dataset(
    data_root: str | Path | None = None,
    num_s1_samples: int = 15000,
    seed: int = 42,
    output_csv: str | Path | None = None,
) -> str:
    """Build labeled pairwise training dataset matching Person 1's multi-pass candidate generator logic."""
    random.seed(seed)
    root = Path(data_root) if data_root else Path(DATA_DIR)
    paths = discover_split(root, "train")

    print(f"Loading ground truth from {paths.ground_truth}...")
    assert paths.ground_truth is not None
    all_gt = load_ground_truth_dict(paths.ground_truth)

    print("Sampling S1 entities...")
    s1_with_matches = [s1 for s1, matches in all_gt.items() if len(matches) > 0]
    singletons = [s1 for s1, matches in all_gt.items() if len(matches) == 0]

    random.shuffle(s1_with_matches)
    random.shuffle(singletons)

    n_pos_s1 = int(num_s1_samples * 0.95)
    n_zero_s1 = num_s1_samples - n_pos_s1
    selected_s1_ids = set(s1_with_matches[:n_pos_s1] + singletons[:n_zero_s1])

    print(f"Selected {len(selected_s1_ids):,} S1 entities ({n_pos_s1} with matches, {n_zero_s1} singletons).")

    print("Indexing sampled S1 entity records and blocking keys...")
    s1_dict: dict[str, EntityRecord] = {}
    key_to_s1: dict[tuple[str, object], list[str]] = defaultdict(list)

    for r in iter_entities(paths.source1, "S1"):
        if r.entity_id in selected_s1_ids:
            s1_dict[r.entity_id] = r
            k_addr = exact_country_address_key(r)
            if k_addr is not None:
                key_to_s1[("exact_country_address", k_addr)].append(r.entity_id)
            k_sig = country_name_signature_key(r)
            if k_sig is not None:
                key_to_s1[("country_name_signature", k_sig)].append(r.entity_id)

    print(f"Indexed {len(s1_dict):,} S1 entities into {len(key_to_s1):,} blocking keys.")

    candidate_pairs_map: dict[tuple[str, str], CandidatePair] = {}
    cand_dict: dict[str, EntityRecord] = {}

    print("Streaming S2 and S3 target records to match blocking keys...")
    for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
        for r in iter_entities(target_path, target_source):
            matched_s1_ids: set[str] = set()

            k_addr = exact_country_address_key(r)
            if k_addr is not None and ("exact_country_address", k_addr) in key_to_s1:
                matched_s1_ids.update(key_to_s1[("exact_country_address", k_addr)])

            k_sig = country_name_signature_key(r)
            if k_sig is not None and ("country_name_signature", k_sig) in key_to_s1:
                matched_s1_ids.update(key_to_s1[("country_name_signature", k_sig)])

            if matched_s1_ids:
                cand_dict[r.entity_id] = r
                for s1_id in matched_s1_ids:
                    pair_key = (s1_id, r.entity_id)
                    if pair_key not in candidate_pairs_map:
                        candidate_pairs_map[pair_key] = CandidatePair(s1_id, r.entity_id, target_source)

    print(f"Generated {len(candidate_pairs_map):,} unique candidate pairs for sampled S1 entities.")

    if output_csv is None:
        out_dir = root / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / "pair_features.csv"
    else:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["source1_id", "candidate_id", "candidate_source", "split_group", "label"]
    sample_feat = build_pair_features("test", "test", "US", "test", "test", "US")
    feat_cols = list(sample_feat.keys())
    fieldnames.extend(feat_cols)

    total_pairs = 0
    total_pos = 0
    total_neg = 0

    print("Computing pairwise features and saving dataset...")
    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for (s1_id, cand_id), pair in candidate_pairs_map.items():
            s1_rec = s1_dict.get(s1_id)
            cand_rec = cand_dict.get(cand_id)

            if not s1_rec or not cand_rec:
                continue

            split_group = get_entity_split_group(s1_id)
            true_matches = all_gt.get(s1_id, set())
            is_match = 1 if cand_id in true_matches else 0

            feats = build_pair_features(
                s1_rec.business_name,
                s1_rec.business_address,
                s1_rec.country,
                cand_rec.business_name,
                cand_rec.business_address,
                cand_rec.country,
            )

            row = {
                "source1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": pair.candidate_source,
                "split_group": split_group,
                "label": is_match,
            }
            row.update(feats)
            writer.writerow(row)

            total_pairs += 1
            if is_match:
                total_pos += 1
            else:
                total_neg += 1

    print(f"\nFinished building training pairs!")
    print(
        f"Total pairs written: {total_pairs:,} (Positives: {total_pos:,}, Negatives: {total_neg:,}, Pos rate: {total_pos / max(1, total_pairs):.2%})"
    )
    print(f"Output saved to: {output_csv}")
    return str(output_csv)


if __name__ == "__main__":
    build_training_dataset(num_s1_samples=15000)
