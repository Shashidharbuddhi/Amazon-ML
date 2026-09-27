"""
predict.py: Integrated End-to-End High-Performance Inference Entry Point.
Uses 7-pass guarded blocking for candidate generation, pairwise feature engineering,
and the trained LightGBM matcher to generate official output/candidate_pairs.tsv and output/matching_results.tsv.
Optimized for multi-core parallelism and strict memory capping via streaming micro-batches and auto-resumption.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Hashable

import joblib
import numpy as np

from src.candidate_generation import CandidateGenerationConfig
from src.config import DATA_DIR, MODELS_DIR, OUTPUT_DIR
from src.data_loader import EntityRecord, discover_split, iter_entities
from src.features import build_pair_features


def get_file_chunks(path: Path, num_chunks: int) -> list[tuple[int, int]]:
    """Split a file into byte ranges aligned to line breaks."""
    total_size = path.stat().st_size
    if total_size == 0:
        return []
    if num_chunks <= 1 or total_size < 1024 * 1024:
        return [(0, total_size)]
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


def _scan_target_chunk_for_predict(
    path_str: str,
    target_source: str,
    byte_range: tuple[int, int],
    needed_keys_by_pass: list[set[Hashable]],
    pass_callables: list[Callable[[EntityRecord], Hashable | None]],
) -> tuple[list[dict[Hashable, list[str]]], dict[str, tuple[str, str, str]]]:
    """Worker task: scan a slice of target file and return postings + records for needed keys."""
    start_byte, end_byte = byte_range
    num_passes = len(pass_callables)
    local_postings: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in range(num_passes)]
    local_records: dict[str, tuple[str, str, str]] = {}

    with open(path_str, "rb") as f:
        f.seek(start_byte)
        if start_byte == 0:
            f.readline()  # skip header

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
                local_records[parts[0]] = (parts[1], parts[2], parts[3])

    return [dict(p) for p in local_postings], local_records


def _extract_worker(batch_args: tuple[list[tuple[str, str, str, str, str, str]], list[str]]) -> np.ndarray:
    """Worker task: compute pairwise features directly into a float32 numpy array."""
    batch_tuples, feature_cols = batch_args
    n_rows = len(batch_tuples)
    n_cols = len(feature_cols)
    mat = np.empty((n_rows, n_cols), dtype=np.float32)
    for i, (s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country) in enumerate(batch_tuples):
        feats = build_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country)
        for j, col in enumerate(feature_cols):
            mat[i, j] = feats[col]
    return mat


def check_existing_progress(cand_path: Path, match_path: Path) -> int:
    """Check how many S1 lines have already been written to output files to resume cleanly."""
    if not cand_path.exists() or not match_path.exists():
        return 0
    if cand_path.stat().st_size == 0 or match_path.stat().st_size == 0:
        return 0

    with cand_path.open("r", encoding="utf-8") as fc, match_path.open("r", encoding="utf-8") as fm:
        cand_lines = sum(1 for _ in fc) - 1
        match_lines = sum(1 for _ in fm) - 1

    if cand_lines > 0 and cand_lines == match_lines:
        return cand_lines
    return 0


def run_prediction_pipeline(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    threshold: float | None = None,
    max_posting_list_size: int = 100,
    batch_size: int = 50000,
    chunk_size: int = 150000,
    num_workers: int = 8,
    limit: int | None = None,
) -> tuple[int, int, int]:
    root = Path(data_root) if data_root else Path(DATA_DIR)
    out_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = out_dir / "candidate_pairs.tsv"
    match_path = out_dir / "matching_results.tsv"

    if model_path is None:
        model_path = Path(MODELS_DIR) / "tree_model_lgbm.joblib"
        if not model_path.exists():
            model_path = Path(MODELS_DIR) / "baseline_logistic.joblib"
    else:
        model_path = Path(model_path)

    print(f"Loading trained matcher artifact from: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    scaler = artifact.get("scaler", None)

    if threshold is None:
        threshold = artifact.get("threshold", 0.55)
    print(f"Using classification threshold: {threshold:.2f}")

    print("Discovering test split files...")
    paths = discover_split(root, "test")

    blocking_cfg = CandidateGenerationConfig(max_posting_list_size=max_posting_list_size)
    passes = blocking_cfg.passes
    pass_callables = [fn for _, fn in passes]

    # Check resumption status
    completed_s1 = check_existing_progress(cand_path, match_path)
    if completed_s1 > 0 and (limit is None or completed_s1 < limit):
        print(f"RESUMING INFERENCE: Detected {completed_s1:,} already-processed S1 records. Appending new results.")
        f_cand = cand_path.open("a", encoding="utf-8", newline="")
        f_match = match_path.open("a", encoding="utf-8", newline="")
    else:
        completed_s1 = 0
        f_cand = cand_path.open("w", encoding="utf-8", newline="")
        f_match = match_path.open("w", encoding="utf-8", newline="")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.flush()
        f_match.flush()

    total_candidates_count = 0
    total_matches_count = 0
    zero_matches_count = 0
    total_s1_processed = completed_s1

    s1_stream = iter_entities(paths.source1, "S1")
    # Skip already completed S1 entities
    if completed_s1 > 0:
        for _ in range(completed_s1):
            next(s1_stream, None)

    exhausted = False
    chunk_index = completed_s1 // chunk_size

    try:
        while not exhausted:
            # 1. Load one chunk of S1 entities
            s1_chunk_records: dict[str, tuple[str, str, str]] = {}
            s1_chunk_order: list[str] = []

            for rec in s1_stream:
                s1_chunk_records[rec.entity_id] = (rec.business_name, rec.business_address, rec.country)
                s1_chunk_order.append(rec.entity_id)
                if limit is not None and total_s1_processed + len(s1_chunk_order) >= limit:
                    exhausted = True
                    break
                if len(s1_chunk_order) >= chunk_size:
                    break
            else:
                exhausted = True

            if not s1_chunk_order:
                break

            chunk_index += 1
            chunk_s1_count = len(s1_chunk_order)
            total_s1_processed += chunk_s1_count
            print(f"\n--- Processing Chunk {chunk_index} ({chunk_s1_count:,} S1 entities, Total so far: {total_s1_processed:,}) ---")

            # 2. Extract keys for S1 entities in this chunk
            needed_keys_by_pass: list[set[Hashable]] = [set() for _ in passes]
            keys_by_s1: dict[str, list[list[Hashable]]] = {
                s1_id: [[] for _ in passes] for s1_id in s1_chunk_order
            }

            for p_idx, (_, key_fn) in enumerate(passes):
                for s1_id in s1_chunk_order:
                    name, addr, country = s1_chunk_records[s1_id]
                    mock_rec = EntityRecord(s1_id, name, addr, country, "S1")
                    k = key_fn(mock_rec)
                    if k is not None:
                        needed_keys_by_pass[p_idx].add(k)
                        keys_by_s1[s1_id][p_idx].append(k)

            target_records_pool: dict[str, tuple[str, str, str]] = {}
            candidates_by_s1: dict[str, list[str]] = {s1_id: [] for s1_id in s1_chunk_order}

            # 3. Parallel scan S2 and S3 for candidates
            for target_source, target_path in (("S2", paths.source2), ("S3", paths.source3)):
                t0 = time.time()
                chunks = get_file_chunks(target_path, num_workers)
                with ProcessPoolExecutor(max_workers=min(num_workers, max(1, len(chunks)))) as executor:
                    futures = [
                        executor.submit(
                            _scan_target_chunk_for_predict,
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

                for s1_id in s1_chunk_order:
                    s1_pass_keys = keys_by_s1[s1_id]
                    cand_list = candidates_by_s1[s1_id]
                    for p_idx, keys in enumerate(s1_pass_keys):
                        for k in keys:
                            c_ids = capped_postings[p_idx].get(k)
                            if c_ids:
                                cand_list.extend(c_ids)

                print(f"Scanned {target_source} in {time.time() - t0:.2f}s.")

            # Deduplicate candidate lists preserving order
            total_chunk_candidates = 0
            for s1_id in s1_chunk_order:
                unique_cands = list(dict.fromkeys(candidates_by_s1[s1_id]))
                candidates_by_s1[s1_id] = unique_cands
                total_chunk_candidates += len(unique_cands)

            print(f"Chunk candidates: {total_chunk_candidates:,} ({total_chunk_candidates / chunk_s1_count:.2f}/S1)")

            # 4. Stream feature extraction and scoring in memory-capped micro-batches
            matches_by_s1: dict[str, list[str]] = {s1_id: [] for s1_id in s1_chunk_order}
            t_score_start = time.time()
            sub_batch_size = 40000
            worker_chunk_size = 5000

            curr_batch_tuples: list[tuple[str, str, str, str, str, str]] = []
            curr_batch_pairs: list[tuple[str, str]] = []

            def score_micro_batch(batch_tuples, batch_pairs):
                if not batch_tuples:
                    return
                if len(batch_tuples) <= worker_chunk_size:
                    X = _extract_worker((batch_tuples, feature_cols))
                else:
                    sub_chunks = [
                        (batch_tuples[i:i + worker_chunk_size], feature_cols)
                        for i in range(0, len(batch_tuples), worker_chunk_size)
                    ]
                    with ProcessPoolExecutor(max_workers=min(num_workers, len(sub_chunks))) as executor:
                        matrices = list(executor.map(_extract_worker, sub_chunks))
                    X = np.vstack(matrices)

                if scaler is not None:
                    X = scaler.transform(X)
                probabilities = model.predict_proba(X)[:, 1]

                for (s1_id, cand_id), prob in zip(batch_pairs, probabilities):
                    if prob >= threshold:
                        matches_by_s1[s1_id].append(cand_id)

            scored_pairs_count = 0
            for s1_id in s1_chunk_order:
                s1_name, s1_addr, s1_country = s1_chunk_records[s1_id]
                for cand_id in candidates_by_s1[s1_id]:
                    cand_name, cand_addr, cand_country = target_records_pool[cand_id]
                    curr_batch_tuples.append((s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country))
                    curr_batch_pairs.append((s1_id, cand_id))
                    if len(curr_batch_tuples) >= sub_batch_size:
                        scored_pairs_count += len(curr_batch_tuples)
                        score_micro_batch(curr_batch_tuples, curr_batch_pairs)
                        curr_batch_tuples.clear()
                        curr_batch_pairs.clear()

            if curr_batch_tuples:
                scored_pairs_count += len(curr_batch_tuples)
                score_micro_batch(curr_batch_tuples, curr_batch_pairs)
                curr_batch_tuples.clear()
                curr_batch_pairs.clear()

            print(f"Scored {scored_pairs_count:,} pairs in {time.time() - t_score_start:.2f}s.")

            # 5. Write chunk results immediately to disk
            chunk_matches = 0
            for s1_id in s1_chunk_order:
                cands_str = ",".join(candidates_by_s1[s1_id])
                matches_str = ",".join(matches_by_s1[s1_id])
                f_cand.write(f"{s1_id}\t{cands_str}\n")
                f_match.write(f"{s1_id}\t{matches_str}\n")
                if matches_str:
                    chunk_matches += len(matches_by_s1[s1_id])
                else:
                    zero_matches_count += 1

            f_cand.flush()
            f_match.flush()
            os.fsync(f_cand.fileno())
            os.fsync(f_match.fileno())

            total_candidates_count += total_chunk_candidates
            total_matches_count += chunk_matches
            print(f"Chunk output flushed to disk. Predicted matches: {chunk_matches:,}")

            # Explicit garbage collection between chunks
            del s1_chunk_records, s1_chunk_order, needed_keys_by_pass, keys_by_s1
            del target_records_pool, candidates_by_s1, matches_by_s1
            gc.collect()

    finally:
        f_cand.close()
        f_match.close()

    print(f"\n==================== PREDICTION COMPLETE ====================")
    print(f"Total S1 entities processed: {total_s1_processed:,}")
    print(f"Total candidate pairs:       {total_candidates_count:,}")
    print(f"Total matches predicted:     {total_matches_count:,}")
    print(f"S1 with 0 predicted matches: {zero_matches_count:,}")
    print(f"Candidate pairs file:        {cand_path}")
    print(f"Matching results file:       {match_path}")
    print(f"=============================================================")

    return total_candidates_count, total_matches_count, zero_matches_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete entity resolution inference pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-posting-list-size", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=150000, help="Number of S1 entities to process per memory chunk.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional testing convenience: only process the first N Source 1 entities.")
    parser.add_argument("--sample", type=int, default=None, help="Alias for --limit.")
    parser.add_argument("--max-rows", type=int, default=None, help="Alias for --limit.")
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else args.sample
    if limit is None:
        limit = args.max_rows

    run_prediction_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        max_posting_list_size=args.max_posting_list_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        limit=limit,
    )


if __name__ == "__main__":
    main()
