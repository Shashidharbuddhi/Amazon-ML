"""
predict.py: Integrated End-to-End Inference Entry Point.
Uses Person 1 multi-pass blocking for candidate generation, Person 2 pairwise feature engineering,
and the trained matching model to generate official output/candidate_pairs.tsv and output/matching_results.tsv.
"""
from __future__ import annotations

import argparse
import sqlite3
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from src.blocking import (
    KeyFunction,
    country_name_signature_key,
    exact_country_address_key,
    iter_multipass_candidates_for_targets,
)
from src.config import DATA_DIR, MODELS_DIR, OUTPUT_DIR
from src.data_loader import EntityRecord, discover_split, iter_entities
from src.features import build_pair_features


def run_prediction_pipeline(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    threshold: float | None = None,
    max_posting_list_size: int = 100,
    batch_size: int = 50000,
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
        threshold = artifact.get("threshold", 0.60)
    print(f"Using classification threshold: {threshold:.2f}")

    print("Discovering test split files...")
    paths = discover_split(root, "test")

    print("Counting test Source 1 entities...")
    s1_count = sum(1 for idx, _ in enumerate(iter_entities(paths.source1, "S1")) if limit is None or idx < limit)
    print(f"Processing {s1_count:,} S1 entities while holding one target source index at a time.")

    passes: tuple[tuple[str, KeyFunction], ...] = (
        ("exact_country_address", exact_country_address_key),
        ("country_name_signature", country_name_signature_key),
    )

    def s1_factory():
        for idx, record in enumerate(iter_entities(paths.source1, "S1")):
            if limit is not None and idx >= limit:
                break
            yield record

    def score_target(target_path: Path, target_source: str, scratch_path: Path, cache_path: Path) -> tuple[int, int]:
        """Score one source, writing one ordered compact row per S1 to scratch."""
        candidate_count = match_count = 0
        source1_by_id = {record.entity_id: record for record in s1_factory()}
        connection = sqlite3.connect(cache_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE targets (id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT)")

        def cached_targets():
            pending_rows = []
            for record in iter_entities(target_path, target_source):
                pending_rows.append((record.entity_id, record.business_name, record.business_address, record.country))
                if len(pending_rows) >= 10000:
                    connection.executemany("INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", pending_rows)
                    pending_rows.clear()
                yield record
            if pending_rows:
                connection.executemany("INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", pending_rows)
            connection.commit()

        candidate_iter = iter_multipass_candidates_for_targets(
            s1_factory, cached_targets() if target_source == "S2" else (),
            cached_targets() if target_source == "S3" else (), passes,
            max_posting_list_size=max_posting_list_size,
        )
        pending: list[tuple[str, list[str]]] = []
        batch_features: list[list[float]] = []
        batch_pairs: list[tuple[str, str]] = []

        def flush(handle) -> tuple[int, int]:
            nonlocal candidate_count, match_count
            probabilities = np.empty(0, dtype=np.float32)
            if batch_features:
                values = np.asarray(batch_features, dtype=np.float32)
                if scaler is not None:
                    values = scaler.transform(values)
                probabilities = model.predict_proba(values)[:, 1]
            match_by_s1: dict[str, list[str]] = {}
            for (s1_id, candidate_id), probability in zip(batch_pairs, probabilities):
                if probability >= threshold:
                    match_by_s1.setdefault(s1_id, []).append(candidate_id)
                    match_count += 1
            for s1_id, candidate_ids in pending:
                unique_candidates = list(dict.fromkeys(candidate_ids))
                unique_matches = list(dict.fromkeys(match_by_s1.get(s1_id, [])))
                candidate_count += len(unique_candidates)
                handle.write(f"{s1_id}\t{','.join(unique_candidates)}\t{','.join(unique_matches)}\n")
            pending.clear()
            batch_features.clear()
            batch_pairs.clear()
            return candidate_count, match_count

        with scratch_path.open("w", encoding="utf-8", newline="") as scratch:
            for s1_id, group in groupby(candidate_iter, key=lambda item: item.source1_entity_id):
                group_rows = list(group)
                candidate_ids: list[str] = []
                for pair in group_rows:
                    candidate_ids.append(pair.candidate_entity_id)
                    target_row = connection.execute(
                        "SELECT name, address, country FROM targets WHERE id = ?", (pair.candidate_entity_id,)
                    ).fetchone()
                    if target_row is None:
                        raise RuntimeError(f"Missing candidate record {pair.candidate_entity_id}")
                    source1_record = source1_by_id[s1_id]
                    feats = build_pair_features(
                        source1_record.business_name, source1_record.business_address, source1_record.country,
                        target_row[0], target_row[1], target_row[2],
                    )
                    batch_features.append([feats[column] for column in feature_cols])
                    batch_pairs.append((s1_id, pair.candidate_entity_id))
                pending.append((s1_id, candidate_ids))
                if len(batch_features) >= batch_size:
                    flush(scratch)
            flush(scratch)
        connection.close()
        return candidate_count, match_count

    with TemporaryDirectory(prefix="entity_resolution_") as temporary_dir:
        temporary = Path(temporary_dir)
        print("Indexing and scoring S2 candidates with Person 1 blocking...")
        s2_candidates, s2_matches = score_target(paths.source2, "S2", temporary / "s2.tsv", temporary / "s2.sqlite")
        print("Indexing and scoring S3 candidates with Person 1 blocking...")
        s3_candidates, s3_matches = score_target(paths.source3, "S3", temporary / "s3.tsv", temporary / "s3.sqlite")

        print(f"\nWriting candidate pairs to {cand_path} and matching results to {match_path}...")
        zero_matches_count = 0
        with (temporary / "s2.tsv").open(encoding="utf-8") as s2_rows, \
             (temporary / "s3.tsv").open(encoding="utf-8") as s3_rows, \
             cand_path.open("w", encoding="utf-8", newline="") as f_cand, \
             match_path.open("w", encoding="utf-8", newline="") as f_match:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            s2_next = next(s2_rows, None)
            s3_next = next(s3_rows, None)
            for source1_record in s1_factory():
                s1_id = source1_record.entity_id
                candidate_ids: list[str] = []
                match_ids: list[str] = []
                for source_rows, source_next in ((s2_rows, s2_next), (s3_rows, s3_next)):
                    if source_next is not None:
                        row_id, candidates, matches = source_next.rstrip("\n").split("\t")
                        if row_id == s1_id:
                            candidate_ids.extend(filter(None, candidates.split(",")))
                            match_ids.extend(filter(None, matches.split(",")))
                            source_next = next(source_rows, None)
                    if source_rows is s2_rows:
                        s2_next = source_next
                    else:
                        s3_next = source_next
                candidate_text = ",".join(dict.fromkeys(candidate_ids))
                match_text = ",".join(dict.fromkeys(match_ids))
                f_cand.write(f"{s1_id}\t{candidate_text}\n")
                f_match.write(f"{s1_id}\t{match_text}\n")
                if not match_text:
                    zero_matches_count += 1

    total_candidates_count = s2_candidates + s3_candidates
    total_matches_count = s2_matches + s3_matches
    print(f"\nCompleted candidate scoring:")
    print(f"  Total S1 entities: {s1_count:,}")
    print(f"  Total candidates generated: {total_candidates_count:,}")
    print(f"  Total matches predicted: {total_matches_count:,}")
    print(f"S1 entities with 0 predicted matches: {zero_matches_count:,}")

    return total_candidates_count, total_matches_count, zero_matches_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete entity resolution inference pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-posting-list-size", type=int, default=100)
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
        limit=limit,
    )


if __name__ == "__main__":
    main()
