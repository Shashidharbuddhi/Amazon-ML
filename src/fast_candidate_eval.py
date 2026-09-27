"""
fast_candidate_eval.py: Fast parallel evaluation of candidate-generation blocking passes.
Uses chunked multi-process scanning across all CPU cores to evaluate 50k S1s in ~30 seconds.
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Sequence

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
from src.config import DATA_DIR
from src.data_loader import EntityRecord, iter_entities, iter_ground_truth, discover_split


def get_file_chunks(path: Path, num_chunks: int) -> list[tuple[int, int]]:
    """Split a file into byte ranges aligned to line breaks."""
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
                f.readline()  # advance to next newline
                end = f.tell()
            chunks.append((start, end))
            start = end
            if start >= total_size:
                break
    return chunks


def _scan_chunk(
    path_str: str,
    target_source: str,
    byte_range: tuple[int, int],
    needed_keys_by_pass: list[set[Hashable]],
    pass_callables: list[Callable[[EntityRecord], Hashable | None]],
) -> list[dict[Hashable, list[str]]]:
    """Scan one byte chunk of a TSV file, returning postings for needed keys."""
    start_byte, end_byte = byte_range
    num_passes = len(pass_callables)
    local_postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in range(num_passes)]

    with open(path_str, "rb") as f:
        f.seek(start_byte)
        if start_byte == 0:
            # skip header
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
            
            for p_idx in range(num_passes):
                k = pass_callables[p_idx](rec)
                if k is not None and k in needed_keys_by_pass[p_idx]:
                    local_postings[p_idx][k].append(parts[0])

    return [dict(p) for p in local_postings]


def evaluate_blocking_pipeline(
    s1_sample_size: int = 50000,
    passes: Sequence[tuple[str, KeyFunction]] = (),
    max_posting_list_size: int = 100,
    num_workers: int = 8,
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

    # Extract S1 keys for all passes
    print(f"Extracting blocking keys for {len(passes)} passes across S1...")
    needed_keys_by_pass: list[set[Hashable]] = [set() for _ in passes]
    keys_by_s1: dict[str, list[list[Hashable]]] = {
        s1_id: [[] for _ in passes] for s1_id in s1_records
    }
    pass_callables = [fn for _, fn in passes]

    for p_idx, (_, key_fn) in enumerate(passes):
        for s1_id, rec in s1_records.items():
            k = key_fn(rec)
            if k is not None:
                needed_keys_by_pass[p_idx].add(k)
                keys_by_s1[s1_id][p_idx].append(k)

    candidates_by_s1: dict[str, set[str]] = {s1_id: set() for s1_id in s1_records}
    found_true_by_source = Counter()
    found_true_by_country = Counter()

    t_start = time.time()
    
    # Process S2 and S3 using worker pool
    for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
        print(f"\nProcessing {target_source} ({target_path}) with {num_workers} parallel workers...")
        t0 = time.time()
        chunks = get_file_chunks(target_path, num_workers)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _scan_chunk,
                    str(target_path),
                    target_source,
                    chunk,
                    needed_keys_by_pass,
                    pass_callables,
                )
                for chunk in chunks
            ]
            chunk_results = [f.result() for f in futures]
        
        # Merge postings across chunks
        merged_postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in passes]
        for c_res in chunk_results:
            for p_idx, p_dict in enumerate(c_res):
                for k, ids in p_dict.items():
                    merged_postings[p_idx][k].extend(ids)
        
        # Apply posting list cap
        capped_postings: list[dict[Hashable, list[str]]] = [{} for _ in passes]
        blocked_keys_count = 0
        total_keys_count = 0
        for p_idx, p_dict in enumerate(merged_postings):
            total_keys_count += len(p_dict)
            for k, ids in p_dict.items():
                if len(ids) >= max_posting_list_size:
                    blocked_keys_count += 1
                else:
                    capped_postings[p_idx][k] = ids

        print(f"Scanned & merged {target_source} in {time.time() - t0:.2f}s. Keys matched: {total_keys_count:,}, Dropped by cap={max_posting_list_size}: {blocked_keys_count:,}")

        # Match candidates into S1s
        for s1_id, s1_pass_keys in keys_by_s1.items():
            s1_cands = candidates_by_s1[s1_id]
            for p_idx, keys in enumerate(s1_pass_keys):
                for k in keys:
                    c_ids = capped_postings[p_idx].get(k)
                    if c_ids:
                        s1_cands.update(c_ids)

    # Compute metrics
    total_candidates = sum(len(cands) for cands in candidates_by_s1.values())
    cands_per_s1 = [len(cands) for cands in candidates_by_s1.values()]
    zero_cands_s1 = sum(1 for c in cands_per_s1 if c == 0)

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

    print(f"\n==================== FAST CANDIDATE EVALUATION RESULTS ====================")
    print(f"Passes evaluated: {[p[0] for p in passes]}")
    print(f"Posting list cap: {max_posting_list_size}")
    print(f"Total time taken:              {time.time() - t_start:.2f}s")
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
    print(f"===========================================================================\n")

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
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
        )
    elif mode == "three_rule":
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
        )
    elif mode == "four_rule":
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
            ("compact_name", compact_name_key),
        )
    elif mode == "five_rule":
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
            ("compact_name", compact_name_key),
            ("country_address_signature", country_address_signature_key),
        )
    elif mode == "six_rule":
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
            ("compact_name", compact_name_key),
            ("country_address_signature", country_address_signature_key),
            ("country_longest_token_addr_num", country_longest_token_addr_num_key),
        )
    elif mode == "seven_rule":
        passes = (
            ("exact_country_address", exact_country_address_key),
            ("country_name_signature", country_name_signature_key),
            ("feature_normalized_name", feature_normalized_name_key),
            ("compact_name", compact_name_key),
            ("country_address_signature", country_address_signature_key),
            ("country_longest_token_addr_num", country_longest_token_addr_num_key),
            ("country_first_2_name_tokens", country_first_2_name_tokens_key),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    evaluate_blocking_pipeline(50000, passes, max_posting_list_size=100, num_workers=8)
