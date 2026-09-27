"""
evaluate_end_to_end.py: Realistic, leakage-free end-to-end evaluation against ORIGINAL ground truth.
Counts candidate generation misses as actual false negatives.
Evaluates end-to-end Macro F0.5, Macro Precision, and Macro Recall across threshold sweeps.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Sequence

import joblib
import numpy as np

from src.blocking import (
    KeyFunction,
    country_name_signature_key,
    exact_country_address_key,
    feature_normalized_name_key,
    compact_name_key,
    country_address_signature_key,
    country_longest_token_addr_num_key,
    country_first_2_name_tokens_key,
)
from src.candidate_generation import CandidateGenerationConfig
from src.config import DATA_DIR, MODELS_DIR
from src.data_loader import EntityRecord, iter_entities, iter_ground_truth, discover_split
from src.evaluate import compute_entity_f05
from src.features import build_pair_features
from src.split_data import get_entity_split_group


def get_file_chunks(path: Path, num_chunks: int) -> list[tuple[int, int]]:
    total_size = path.stat().st_size
    chunk_size = total_size // num_chunks
    chunks = []
    with path.open("rb") as f:
        start = 0
        for i in range(num_chunks):
            if i == num_chunks - 1:
                end = total_size
            else:
                target = (i + 1) * chunk_size
                f.seek(target)
                f.readline()
                end = f.tell()
            chunks.append((start, end))
            start = end
            if start >= total_size:
                break
    return chunks


def _scan_chunk_with_records(
    path_str: str,
    target_source: str,
    byte_range: tuple[int, int],
    needed_keys_by_pass: list[set[Hashable]],
    pass_callables: list[Callable[[EntityRecord], Hashable | None]],
) -> tuple[list[dict[Hashable, list[str]]], dict[str, EntityRecord]]:
    start_byte, end_byte = byte_range
    num_passes = len(pass_callables)
    local_postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in range(num_passes)]
    local_records: dict[str, EntityRecord] = {}

    with open(path_str, "rb") as f:
        f.seek(start_byte)
        if start_byte == 0:
            f.readline()
        
        while f.tell() < end_byte:
            line_bytes = f.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            
            rec = EntityRecord(
                entity_id=parts[0],
                business_name=parts[1],
                business_address=parts[2],
                country=parts[3],
                source=target_source,
            )
            
            matched_any = False
            for p_idx in range(num_passes):
                k = pass_callables[p_idx](rec)
                if k is not None and k in needed_keys_by_pass[p_idx]:
                    local_postings[p_idx][k].append(parts[0])
                    matched_any = True
            
            if matched_any:
                local_records[parts[0]] = rec

    return [dict(p) for p in local_postings], local_records


def run_end_to_end_validation(
    val_s1_limit: int = 10000,
    model_path: str | Path | None = None,
    config: CandidateGenerationConfig | None = None,
    num_workers: int = 8,
    data_root: Path | None = None,
) -> dict:
    root = data_root or Path(DATA_DIR)
    paths = discover_split(root, "train")
    assert paths.ground_truth is not None

    cfg = config or CandidateGenerationConfig()
    passes = cfg.passes
    max_posting_list_size = cfg.max_posting_list_size

    # Load model
    if model_path is None:
        model_path = Path(MODELS_DIR) / "tree_model_lgbm.joblib"
    else:
        model_path = Path(model_path)

    print(f"Loading model artifact from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    scaler = artifact.get("scaler", None)

    # 1. Select held-out validation S1 entities
    print(f"Selecting held-out validation S1 entities (group == 'val')...")
    val_s1_records: dict[str, EntityRecord] = {}
    val_s1_countries: dict[str, str] = {}

    for rec in iter_entities(paths.source1, "S1"):
        if get_entity_split_group(rec.entity_id) == "val":
            val_s1_records[rec.entity_id] = rec
            val_s1_countries[rec.entity_id] = (rec.country or "UNKNOWN").strip().upper()
            if len(val_s1_records) >= val_s1_limit:
                break

    print(f"Loaded {len(val_s1_records):,} held-out validation S1 entities.")

    # 2. Load original ground truth for these validation S1 entities
    print("Loading original ground truth for validation S1s...")
    all_val_gt: dict[str, set[str]] = {s1_id: set() for s1_id in val_s1_records}
    true_by_source = Counter()
    true_by_country = Counter()

    for rec in iter_ground_truth(paths.ground_truth):
        if rec.source1_entity_id in all_val_gt:
            targets = set(rec.matched_entity_ids)
            all_val_gt[rec.source1_entity_id] = targets
            c = val_s1_countries[rec.source1_entity_id]
            for t in targets:
                src = t.split("-")[0]
                true_by_source[src] += 1
                true_by_country[c] += 1

    total_true_links = sum(true_by_source.values())
    singletons_count = sum(1 for targets in all_val_gt.values() if len(targets) == 0)
    print(f"Validation GT: {total_true_links:,} true links (S2: {true_by_source['S2']:,}, S3: {true_by_source['S3']:,})")
    print(f"Singletons (0 true links): {singletons_count:,} ({singletons_count / len(val_s1_records):.2%})")

    # 3. Generate candidates using production blocker
    print(f"\nGenerating candidates using {len(passes)} passes (cap={max_posting_list_size})...")
    needed_keys_by_pass: list[set[Hashable]] = [set() for _ in passes]
    keys_by_s1: dict[str, list[list[Hashable]]] = {
        s1_id: [[] for _ in passes] for s1_id in val_s1_records
    }
    pass_callables = [fn for _, fn in passes]

    for p_idx, (_, key_fn) in enumerate(passes):
        for s1_id, rec in val_s1_records.items():
            k = key_fn(rec)
            if k is not None:
                needed_keys_by_pass[p_idx].add(k)
                keys_by_s1[s1_id][p_idx].append(k)

    target_records_pool: dict[str, EntityRecord] = {}
    candidates_by_s1: dict[str, set[str]] = {s1_id: set() for s1_id in val_s1_records}
    t_start = time.time()

    for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
        print(f"Scanning {target_source} with {num_workers} parallel workers...")
        chunks = get_file_chunks(target_path, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _scan_chunk_with_records,
                    str(target_path),
                    target_source,
                    chunk,
                    needed_keys_by_pass,
                    pass_callables,
                )
                for chunk in chunks
            ]
            results = [f.result() for f in futures]

        merged_postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in passes]
        for postings_list, rec_dict in results:
            target_records_pool.update(rec_dict)
            for p_idx, p_dict in enumerate(postings_list):
                for k, ids in p_dict.items():
                    merged_postings[p_idx][k].extend(ids)

        capped_postings: list[dict[Hashable, list[str]]] = [{} for _ in passes]
        for p_idx, p_dict in enumerate(merged_postings):
            for k, ids in p_dict.items():
                if len(ids) < max_posting_list_size:
                    capped_postings[p_idx][k] = ids

        for s1_id, s1_pass_keys in keys_by_s1.items():
            s1_cands = candidates_by_s1[s1_id]
            for p_idx, keys in enumerate(s1_pass_keys):
                for k in keys:
                    c_ids = capped_postings[p_idx].get(k)
                    if c_ids:
                        s1_cands.update(c_ids)

    # 4. Measure candidate recall
    captured_true_by_source = Counter()
    captured_true_by_country = Counter()
    total_candidates = sum(len(cands) for cands in candidates_by_s1.values())
    cands_per_s1 = [len(cands) for cands in candidates_by_s1.values()]
    zero_cands = sum(1 for c in cands_per_s1 if c == 0)

    captured_true_total = 0
    for s1_id, cands in candidates_by_s1.items():
        true_targets = all_val_gt.get(s1_id, set())
        matched = cands & true_targets
        captured_true_total += len(matched)
        c = val_s1_countries[s1_id]
        for m in matched:
            src = m.split("-")[0]
            captured_true_by_source[src] += 1
            captured_true_by_country[c] += 1

    cand_recall = captured_true_total / total_true_links if total_true_links else 0.0
    cand_recall_s2 = captured_true_by_source["S2"] / true_by_source["S2"] if true_by_source["S2"] else 0.0
    cand_recall_s3 = captured_true_by_source["S3"] / true_by_source["S3"] if true_by_source["S3"] else 0.0

    print(f"\n--- CANDIDATE GENERATION RECALL ---")
    print(f"Overall Candidate Recall: {cand_recall:.4%} ({captured_true_total:,}/{total_true_links:,})")
    print(f"S2 Candidate Recall:      {cand_recall_s2:.4%} ({captured_true_by_source['S2']:,}/{true_by_source['S2']:,})")
    print(f"S3 Candidate Recall:      {cand_recall_s3:.4%} ({captured_true_by_source['S3']:,}/{true_by_source['S3']:,})")
    print(f"Total Candidates:         {total_candidates:,} ({total_candidates / len(val_s1_records):.2f}/S1)")
    print(f"Zero-candidate S1s:       {zero_cands:,} ({zero_cands / len(val_s1_records):.2%})")

    # 5. Compute pairwise features and model predictions
    print(f"\nComputing pairwise features for {total_candidates:,} candidate pairs...")
    t_feat = time.time()
    
    # Batch feature computation
    batch_features = []
    pair_index = []  # list of (s1_id, cand_id)

    for s1_id, cands in candidates_by_s1.items():
        s1_rec = val_s1_records[s1_id]
        for cand_id in cands:
            cand_rec = target_records_pool[cand_id]
            feats = build_pair_features(
                s1_rec.business_name,
                s1_rec.business_address,
                s1_rec.country,
                cand_rec.business_name,
                cand_rec.business_address,
                cand_rec.country,
            )
            batch_features.append([feats[col] for col in feature_cols])
            pair_index.append((s1_id, cand_id))

    print(f"Computed features in {time.time() - t_feat:.2f}s.")

    # Model inference
    print("Scoring candidates with classifier...")
    X_val = np.asarray(batch_features, dtype=np.float32)
    if scaler is not None:
        X_val = scaler.transform(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    # Organize predictions by S1
    cand_probs_by_s1: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (s1_id, cand_id), prob in zip(pair_index, probs):
        cand_probs_by_s1[s1_id].append((cand_id, float(prob)))

    # 6. Threshold sweep for true end-to-end Macro F0.5
    print("\n--- THRESHOLD SWEEP FOR END-TO-END MACRO F0.5 ---")
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    best_th = 0.60
    best_f05 = -1.0
    best_results = {}
    table_rows = []

    for th in thresholds:
        precisions = []
        recalls = []
        f05_scores = []
        
        tp_total = 0
        fp_total = 0
        fn_total = 0
        predicted_links_count = 0

        # Breakdowns by country and source
        country_tp = Counter()
        country_fp = Counter()
        country_fn = Counter()
        source_tp = Counter()
        source_fp = Counter()
        source_fn = Counter()

        for s1_id, true_targets in all_val_gt.items():
            cands = cand_probs_by_s1.get(s1_id, [])
            pred_set = {cid for cid, p in cands if p >= th}
            predicted_links_count += len(pred_set)

            p, r, f05 = compute_entity_f05(true_targets, pred_set)
            precisions.append(p)
            recalls.append(r)
            f05_scores.append(f05)

            tp = len(true_targets & pred_set)
            fp = len(pred_set - true_targets)
            fn = len(true_targets - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            c = val_s1_countries[s1_id]
            country_tp[c] += tp
            country_fp[c] += fp
            country_fn[c] += fn

            for mid in (true_targets & pred_set):
                source_tp[mid.split("-")[0]] += 1
            for mid in (pred_set - true_targets):
                source_fp[mid.split("-")[0]] += 1
            for mid in (true_targets - pred_set):
                source_fn[mid.split("-")[0]] += 1

        macro_p = float(np.mean(precisions))
        macro_r = float(np.mean(recalls))
        macro_f05 = float(np.mean(f05_scores))

        row_info = {
            "threshold": th,
            "macro_f05": macro_f05,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "predicted_links": predicted_links_count,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "country_breakdown": {
                c: {
                    "tp": country_tp[c],
                    "fp": country_fp[c],
                    "fn": country_fn[c],
                    "precision": country_tp[c] / (country_tp[c] + country_fp[c]) if (country_tp[c] + country_fp[c]) else 0.0,
                    "recall": country_tp[c] / (country_tp[c] + country_fn[c]) if (country_tp[c] + country_fn[c]) else 0.0,
                }
                for c in ["INDIA", "US"]
            },
            "source_breakdown": {
                s: {
                    "tp": source_tp[s],
                    "fp": source_fp[s],
                    "fn": source_fn[s],
                    "precision": source_tp[s] / (source_tp[s] + source_fp[s]) if (source_tp[s] + source_fp[s]) else 0.0,
                    "recall": source_tp[s] / (source_tp[s] + source_fn[s]) if (source_tp[s] + source_fn[s]) else 0.0,
                }
                for s in ["S2", "S3"]
            }
        }
        table_rows.append(row_info)

        print(
            f"Th: {th:.2f} | Macro F0.5: {macro_f05:.4f} | Prec: {macro_p:.4f} | Rec: {macro_r:.4f} | Pred Links: {predicted_links_count:,} | TP: {tp_total:,} | FP: {fp_total:,} | FN: {fn_total:,}"
        )

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_th = th
            best_results = row_info

    print(f"\n==================== BEST END-TO-END VALIDATION RESULT ====================")
    print(f"Optimal Threshold:       {best_th:.2f}")
    print(f"End-to-End Macro F0.5:   {best_results['macro_f05']:.4f}")
    print(f"End-to-End Precision:    {best_results['macro_precision']:.4f}")
    print(f"End-to-End Recall:       {best_results['macro_recall']:.4f}")
    print(f"Candidate Recall:        {cand_recall:.4%}")
    print(f"Predicted Links:         {best_results['predicted_links']:,}")
    print(f"India Precision/Recall:  {best_results['country_breakdown']['INDIA']['precision']:.4f} / {best_results['country_breakdown']['INDIA']['recall']:.4f}")
    print(f"US Precision/Recall:     {best_results['country_breakdown']['US']['precision']:.4f} / {best_results['country_breakdown']['US']['recall']:.4f}")
    print(f"S2 Precision/Recall:     {best_results['source_breakdown']['S2']['precision']:.4f} / {best_results['source_breakdown']['S2']['recall']:.4f}")
    print(f"S3 Precision/Recall:     {best_results['source_breakdown']['S3']['precision']:.4f} / {best_results['source_breakdown']['S3']['recall']:.4f}")
    print(f"===========================================================================\n")

    return {
        "candidate_recall": cand_recall,
        "best_threshold": best_th,
        "best_results": best_results,
        "sweep_table": table_rows,
    }


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    model = sys.argv[2] if len(sys.argv) > 2 else None
    run_end_to_end_validation(val_s1_limit=limit, model_path=model)
