"""
build_training_pairs.py: Fast, memory-conscious training pair construction using Person 1 candidate generation logic.
Uses chunked multi-processing to scan targets across CPU cores.
"""
from __future__ import annotations

import csv
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable

from src.blocking import CandidatePair
from src.candidate_generation import CandidateGenerationConfig
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


def _scan_chunk_for_training(
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
            
            matched = False
            for p_idx in range(num_passes):
                k = pass_callables[p_idx](rec)
                if k is not None and k in needed_keys_by_pass[p_idx]:
                    local_postings[p_idx][k].append(parts[0])
                    matched = True
            
            if matched:
                local_records[parts[0]] = rec

    return [dict(p) for p in local_postings], local_records


def build_training_dataset(
    data_root: str | Path | None = None,
    num_s1_samples: int = 20000,
    seed: int = 42,
    output_csv: str | Path | None = None,
    num_workers: int = 8,
) -> str:
    """Build labeled pairwise training dataset matching production candidate generator logic."""
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

    print("Indexing sampled S1 entity records...")
    s1_dict: dict[str, EntityRecord] = {}
    for r in iter_entities(paths.source1, "S1"):
        if r.entity_id in selected_s1_ids:
            s1_dict[r.entity_id] = r

    print(f"Indexed {len(s1_dict):,} sampled S1 entities.")

    blocking_config = CandidateGenerationConfig()
    passes = blocking_config.passes
    max_posting_list_size = blocking_config.max_posting_list_size
    pass_callables = [fn for _, fn in passes]

    # Precompute S1 keys
    needed_keys_by_pass: list[set[Hashable]] = [set() for _ in passes]
    keys_by_s1: dict[str, list[list[Hashable]]] = {
        s1_id: [[] for _ in passes] for s1_id in s1_dict
    }

    for p_idx, (_, key_fn) in enumerate(passes):
        for s1_id, rec in s1_dict.items():
            k = key_fn(rec)
            if k is not None:
                needed_keys_by_pass[p_idx].add(k)
                keys_by_s1[s1_id][p_idx].append(k)

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
    by_source = Counter()
    by_country = Counter()
    pos_by_source = Counter()

    print(f"\nScanning targets and computing pairwise features with {num_workers} parallel workers...")
    t0_all = time.time()

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
            t0 = time.time()
            print(f"Processing {target_source} ({target_path})...")
            chunks = get_file_chunks(target_path, num_workers)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _scan_chunk_for_training,
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
            target_records_pool: dict[str, EntityRecord] = {}
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

            print(f"Scanned {target_source} in {time.time() - t0:.2f}s. Computing pairwise features...")

            source_pairs = 0
            for s1_id, s1_pass_keys in keys_by_s1.items():
                s1_rec = s1_dict[s1_id]
                split_group = get_entity_split_group(s1_id)
                true_matches = all_gt.get(s1_id, set())

                seen_cands = set()
                for p_idx, keys in enumerate(s1_pass_keys):
                    for k in keys:
                        c_ids = capped_postings[p_idx].get(k, ())
                        for cid in c_ids:
                            if cid in seen_cands:
                                continue
                            seen_cands.add(cid)

                            cand_rec = target_records_pool[cid]
                            is_match = 1 if cid in true_matches else 0

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
                                "candidate_id": cid,
                                "candidate_source": target_source,
                                "split_group": split_group,
                                "label": is_match,
                            }
                            row.update(feats)
                            writer.writerow(row)

                            total_pairs += 1
                            source_pairs += 1
                            by_source[target_source] += 1
                            c = (s1_rec.country or "UNKNOWN").strip().upper()
                            by_country[c] += 1

                            if is_match:
                                total_pos += 1
                                pos_by_source[target_source] += 1
                            else:
                                total_neg += 1

            print(f"  {target_source} pairs generated: {source_pairs:,}")

    print(f"\n==================== TRAINING PAIRS GENERATION COMPLETE ====================")
    print(f"Total time taken:             {time.time() - t0_all:.2f}s")
    print(f"Total S1 entities:            {len(s1_dict):,}")
    print(f"Total candidate pairs:        {total_pairs:,}")
    print(f"Candidates per S1:            {total_pairs / len(s1_dict):.2f}")
    print(f"Positives:                    {total_pos:,} ({total_pos / max(1, total_pairs):.2%})")
    print(f"Negatives:                    {total_neg:,} ({total_neg / max(1, total_pairs):.2%})")
    print("\n--- Breakdown by Target Source ---")
    for src in sorted(by_source):
        print(f"  {src}: {by_source[src]:,} pairs (Positives: {pos_by_source[src]:,}, Pos rate: {pos_by_source[src] / max(1, by_source[src]):.2%})")
    print("\n--- Breakdown by Country ---")
    for c in sorted(by_country):
        print(f"  {c}: {by_country[c]:,} pairs")
    print(f"Output saved to:              {output_csv}")
    print(f"============================================================================\n")

    return str(output_csv)


if __name__ == "__main__":
    import sys
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    build_training_dataset(num_s1_samples=samples, num_workers=8)
