"""
benchmark_candidate_recall.py: Measure candidate recall against ground truth on a deterministic representative S1 sample.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Hashable, Iterable, Sequence

from src.blocking import (
    KeyFunction,
    country_name_signature_key,
    exact_country_address_key,
    exact_country_name_key,
    feature_normalized_name_key,
)
from src.config import DATA_DIR
from src.data_loader import EntityRecord, iter_entities, iter_ground_truth, discover_split


def evaluate_candidate_passes(
    s1_sample_size: int = 50000,
    passes: Sequence[tuple[str, KeyFunction]] = (),
    max_posting_list_size: int = 100,
    data_root: Path | None = None,
) -> dict:
    root = data_root or Path(DATA_DIR)
    paths = discover_split(root, "train")
    assert paths.ground_truth is not None

    print(f"Loading first {s1_sample_size:,} S1 entities from {paths.source1}...")
    s1_records: dict[str, EntityRecord] = {}
    s1_countries: dict[str, str] = {}
    for idx, rec in enumerate(iter_entities(paths.source1, "S1")):
        if idx >= s1_sample_size:
            break
        s1_records[rec.entity_id] = rec
        s1_countries[rec.entity_id] = (rec.country or "UNKNOWN").strip().upper()

    print(f"Loaded {len(s1_records):,} S1 entities.")

    # Load ground truth for these S1s
    print(f"Loading ground truth for sampled S1 entities...")
    gt_for_s1: dict[str, set[str]] = defaultdict(set)
    total_true_by_source = Counter()
    total_true_by_country = Counter()
    for rec in iter_ground_truth(paths.ground_truth):
        if rec.source1_entity_id in s1_records:
            targets = set(rec.matched_entity_ids)
            gt_for_s1[rec.source1_entity_id] = targets
            c = s1_countries[rec.source1_entity_id]
            for t in targets:
                src = t.split("-")[0]
                total_true_by_source[src] += 1
                total_true_by_country[c] += 1

    total_true = sum(total_true_by_source.values())
    print(f"Total true links in sample: {total_true:,} (S2: {total_true_by_source['S2']:,}, S3: {total_true_by_source['S3']:,})")
    for c, cnt in sorted(total_true_by_country.items()):
        print(f"  Country {c}: {cnt:,} true links")

    # Index S1 keys for the given passes
    print(f"Extracting blocking keys for {len(passes)} passes...")
    sampled_s1_by_pass_and_key: list[dict[Hashable, list[str]]] = [
        defaultdict(list) for _ in passes
    ]
    keys_by_s1: dict[str, list[list[Hashable]]] = {
        s1_id: [[] for _ in passes] for s1_id in s1_records
    }

    for pass_idx, (rule_name, key_fn) in enumerate(passes):
        for s1_id, rec in s1_records.items():
            k = key_fn(rec)
            if k is not None:
                sampled_s1_by_pass_and_key[pass_idx][k].append(s1_id)
                keys_by_s1[s1_id][pass_idx].append(k)

    candidates_by_s1: dict[str, set[str]] = {s1_id: set() for s1_id in s1_records}
    found_true_by_source = Counter()
    found_true_by_country = Counter()

    # Stream targets
    t_start = time.time()
    for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
        print(f"\nProcessing {target_source} ({target_path})...")
        t0 = time.time()
        # Postings for keys in sampled S1
        postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in passes]
        blocked: list[set[Hashable]] = [set() for _ in passes]

        record_count = 0
        for rec in iter_entities(target_path, target_source):
            record_count += 1
            for pass_idx, (_, key_fn) in enumerate(passes):
                k = key_fn(rec)
                if k is None or k in blocked[pass_idx]:
                    continue
                if k not in sampled_s1_by_pass_and_key[pass_idx]:
                    continue
                plist = postings[pass_idx][k]
                if len(plist) >= max_posting_list_size:
                    del postings[pass_idx][k]
                    blocked[pass_idx].add(k)
                    continue
                plist.append(rec.entity_id)

        print(f"Scanned {record_count:,} {target_source} records in {time.time() - t0:.2f}s.")

        # Match candidates
        for s1_id, s1_pass_keys in keys_by_s1.items():
            s1_cands = candidates_by_s1[s1_id]
            for pass_idx, keys in enumerate(s1_pass_keys):
                for k in keys:
                    for cand_id in postings[pass_idx].get(k, ()):
                        s1_cands.add(cand_id)

    # Compute metrics
    total_candidates = sum(len(cands) for cands in candidates_by_s1.values())
    cands_per_s1 = [len(cands) for cands in candidates_by_s1.values()]
    zero_cands_s1 = sum(1 for c in cands_per_s1 if c == 0)

    # Calculate recall
    found_true_total = 0
    for s1_id, cands in candidates_by_s1.items():
        true_targets = gt_for_s1.get(s1_id, set())
        matched = cands & true_targets
        found_true_total += len(matched)
        c = s1_countries[s1_id]
        for m in matched:
            src = m.split("-")[0]
            found_true_by_source[src] += 1
            found_true_by_country[c] += 1

    overall_recall = found_true_total / total_true if total_true else 0.0
    s2_recall = found_true_by_source["S2"] / total_true_by_source["S2"] if total_true_by_source["S2"] else 0.0
    s3_recall = found_true_by_source["S3"] / total_true_by_source["S3"] if total_true_by_source["S3"] else 0.0

    print(f"\n==================== CANDIDATE EVALUATION RESULTS ====================")
    print(f"Passes evaluated: {[p[0] for p in passes]}")
    print(f"Posting list cap: {max_posting_list_size}")
    print(f"Time taken: {time.time() - t_start:.2f}s")
    print(f"Total S1 evaluated:            {len(s1_records):,}")
    print(f"Original true links:           {total_true:,}")
    print(f"Captured true links:           {found_true_total:,}")
    print(f"Missed true links:             {total_true - found_true_total:,}")
    print(f"Overall candidate recall:      {overall_recall:.4%}")
    print(f"S2 candidate recall:           {s2_recall:.4%}")
    print(f"S3 candidate recall:           {s3_recall:.4%}")
    print(f"Total candidate pairs:         {total_candidates:,}")
    print(f"Average candidates / S1:       {total_candidates / len(s1_records):.3f}")
    print(f"Median candidates / S1:        {statistics.median(cands_per_s1):.1f}")
    print(f"Max candidates / S1:           {max(cands_per_s1) if cands_per_s1 else 0}")
    print(f"Zero-candidate S1s:            {zero_cands_s1:,} ({zero_cands_s1 / len(s1_records):.3%})")

    print("\n--- Recall by Country ---")
    for c, total_c in sorted(total_true_by_country.items()):
        found_c = found_true_by_country[c]
        rec_c = found_c / total_c if total_c else 0.0
        print(f"  {c:10s}: {found_c:,} / {total_c:,} ({rec_c:.4%})")

    possible_pairs = len(s1_records) * (5034616 + 5285603)
    reduction_ratio = 1.0 - (total_candidates / possible_pairs)
    print(f"Reduction ratio:               {reduction_ratio:.8f}")
    print(f"======================================================================\n")

    return {
        "passes": [p[0] for p in passes],
        "cap": max_posting_list_size,
        "s1_count": len(s1_records),
        "total_true": total_true,
        "found_true": found_true_total,
        "overall_recall": overall_recall,
        "s2_recall": s2_recall,
        "s3_recall": s3_recall,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": total_candidates / len(s1_records),
        "median_candidates_per_s1": statistics.median(cands_per_s1),
        "max_candidates_per_s1": max(cands_per_s1) if cands_per_s1 else 0,
        "zero_candidate_s1_count": zero_cands_s1,
        "zero_candidate_s1_rate": zero_cands_s1 / len(s1_records),
        "recall_by_country": {
            c: found_true_by_country[c] / total_true_by_country[c] if total_true_by_country[c] else 0.0
            for c in total_true_by_country
        },
        "reduction_ratio": reduction_ratio,
    }


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    if mode == "baseline":
        # Rule 1 & Rule 2
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
        )
    elif mode == "three_rule":
        # Rule 1, Rule 2, Rule 3
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    evaluate_candidate_passes(50000, passes, max_posting_list_size=100)
