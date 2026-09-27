"""Training-only candidate-set evaluation; labels never influence blocking."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.blocking import (
    CandidatePair,
    KeyFunction,
    country_name_signature_key,
    exact_country_address_key,
    feature_normalized_name_key,
    exact_country_name_key,
    iter_multipass_candidates_for_targets,
    iter_keyed_candidates_for_targets,
)
from src.data_loader import GroundTruthRecord, iter_entities, iter_ground_truth, discover_split


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    source1_entities: int
    source2_entities: int
    source3_entities: int
    candidate_pairs: int
    average_candidates_per_source1: float
    median_candidates_per_source1: float
    maximum_candidates_per_source1: int
    true_matches: int
    true_matches_found: int
    candidate_recall: float
    source2_true_matches: int
    source2_matches_found: int
    source2_recall: float
    source3_true_matches: int
    source3_matches_found: int
    source3_recall: float
    source1_with_true_matches_lost: int
    possible_pairs: int
    reduction_ratio: float


def _recall(found: int, total: int) -> float:
    return found / total if total else 1.0


def evaluate_candidates(
    candidates: Iterable[CandidatePair],
    ground_truth: Iterable[GroundTruthRecord],
    *,
    source1_entities: int,
    source2_entities: int,
    source3_entities: int,
    progress_every: int | None = None,
) -> CandidateEvaluation:
    """Evaluate a unique candidate stream against training labels.

    Ground truth is held as mutable per-S1 target sets. A found target is
    removed, so a duplicate candidate cannot inflate recall and no labels are
    passed to candidate-generation code.
    """
    remaining: dict[str, set[str]] = {}
    total_by_source: Counter[str] = Counter()
    for record in ground_truth:
        if record.source1_entity_id in remaining:
            raise ValueError(f"Duplicate ground-truth S1 ID: {record.source1_entity_id}")
        targets = set(record.matched_entity_ids)
        remaining[record.source1_entity_id] = targets
        for target in targets:
            prefix = target.partition("-")[0]
            if prefix not in {"S2", "S3"}:
                raise ValueError(f"Invalid target ID in ground truth: {target}")
            total_by_source[prefix] += 1

    candidates_by_s1: Counter[str] = Counter()
    found_by_source: Counter[str] = Counter()
    candidate_pairs = 0
    for pair in candidates:
        candidate_pairs += 1
        if progress_every and candidate_pairs % progress_every == 0:
            print(f"candidate_pairs_processed={candidate_pairs:,}", file=sys.stderr, flush=True)
        candidates_by_s1[pair.source1_entity_id] += 1
        targets = remaining.get(pair.source1_entity_id)
        if targets is not None and pair.candidate_entity_id in targets:
            targets.remove(pair.candidate_entity_id)
            found_by_source[pair.candidate_source] += 1

    counts = list(candidates_by_s1.values())
    counts.extend([0] * (source1_entities - len(counts)))
    if len(counts) != source1_entities:
        raise ValueError("Candidate stream contains more distinct S1 IDs than supplied source1_entities")
    total_true = total_by_source["S2"] + total_by_source["S3"]
    total_found = found_by_source["S2"] + found_by_source["S3"]
    lost_s1 = sum(1 for targets in remaining.values() if targets)
    possible_pairs = source1_entities * (source2_entities + source3_entities)
    return CandidateEvaluation(
        source1_entities=source1_entities,
        source2_entities=source2_entities,
        source3_entities=source3_entities,
        candidate_pairs=candidate_pairs,
        average_candidates_per_source1=round(candidate_pairs / source1_entities, 6) if source1_entities else 0.0,
        median_candidates_per_source1=statistics.median(counts) if counts else 0.0,
        maximum_candidates_per_source1=max(counts, default=0),
        true_matches=total_true,
        true_matches_found=total_found,
        candidate_recall=round(_recall(total_found, total_true), 8),
        source2_true_matches=total_by_source["S2"],
        source2_matches_found=found_by_source["S2"],
        source2_recall=round(_recall(found_by_source["S2"], total_by_source["S2"]), 8),
        source3_true_matches=total_by_source["S3"],
        source3_matches_found=found_by_source["S3"],
        source3_recall=round(_recall(found_by_source["S3"], total_by_source["S3"]), 8),
        source1_with_true_matches_lost=lost_s1,
        possible_pairs=possible_pairs,
        reduction_ratio=round(1 - candidate_pairs / possible_pairs, 8) if possible_pairs else 0.0,
    )


def _count(records: Iterable[object]) -> int:
    return sum(1 for _ in records)


RULES: dict[str, KeyFunction] = {
    "exact_country_name": exact_country_name_key,
    "exact_country_address": exact_country_address_key,
    "country_name_signature": country_name_signature_key,
}

CONSERVATIVE_MULTIPASS: tuple[tuple[str, KeyFunction], ...] = (
    ("exact_country_address", exact_country_address_key),
    ("country_name_signature", country_name_signature_key),
    ("feature_normalized_name", feature_normalized_name_key),
)


def evaluate_rule_train(data_root: Path, rule: str, *, progress: bool = False) -> CandidateEvaluation:
    """Run one deterministic blocking rule and evaluate it on training labels."""
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule!r}; choose from {sorted(RULES)}")
    paths = discover_split(data_root, "train")
    assert paths.ground_truth is not None
    if progress:
        print("counting entity records", file=sys.stderr, flush=True)
    source1_count = _count(iter_entities(paths.source1, "S1"))
    source2_count = _count(iter_entities(paths.source2, "S2"))
    source3_count = _count(iter_entities(paths.source3, "S3"))
    if progress:
        print(f"loading ground truth and evaluating {rule}", file=sys.stderr, flush=True)
    candidates = iter_keyed_candidates_for_targets(
        lambda: iter_entities(paths.source1, "S1"),
        iter_entities(paths.source2, "S2"),
        iter_entities(paths.source3, "S3"),
        RULES[rule],
        rule,
        on_target_index_ready=(
            lambda source, keys: print(f"{source} index keys={keys:,}", file=sys.stderr, flush=True)
        )
        if progress
        else None,
    )
    return evaluate_candidates(
        candidates,
        iter_ground_truth(paths.ground_truth),
        source1_entities=source1_count,
        source2_entities=source2_count,
        source3_entities=source3_count,
        progress_every=1_000_000 if progress else None,
    )


def evaluate_exact_name_train(data_root: Path, *, progress: bool = False) -> CandidateEvaluation:
    """Compatibility wrapper for baseline evaluation."""
    return evaluate_rule_train(data_root, "exact_country_name", progress=progress)


def evaluate_multipass_train(
    data_root: Path, *, max_posting_list_size: int = 100, progress: bool = False
) -> CandidateEvaluation:
    """Evaluate the conservative address/name-signature union on training data."""
    paths = discover_split(data_root, "train")
    assert paths.ground_truth is not None
    if progress:
        print("counting entity records", file=sys.stderr, flush=True)
    source1_count = _count(iter_entities(paths.source1, "S1"))
    source2_count = _count(iter_entities(paths.source2, "S2"))
    source3_count = _count(iter_entities(paths.source3, "S3"))
    if progress:
        print(
            f"loading ground truth and evaluating conservative_multipass cap={max_posting_list_size}",
            file=sys.stderr,
            flush=True,
        )
    candidates = iter_multipass_candidates_for_targets(
        lambda: iter_entities(paths.source1, "S1"),
        iter_entities(paths.source2, "S2"),
        iter_entities(paths.source3, "S3"),
        CONSERVATIVE_MULTIPASS,
        max_posting_list_size=max_posting_list_size,
        on_target_index_ready=(
            lambda source, rule, kept, raw: print(
                f"{source} {rule} keys={kept:,}/{raw:,}", file=sys.stderr, flush=True
            )
        )
        if progress
        else None,
    )
    return evaluate_candidates(
        candidates,
        iter_ground_truth(paths.ground_truth),
        source1_entities=source1_count,
        source2_entities=source2_count,
        source3_entities=source3_count,
        progress_every=1_000_000 if progress else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a blocking rule on training labels.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--rule",
        choices=[*sorted(RULES), "conservative_multipass"],
        default="exact_country_name",
    )
    parser.add_argument("--max-posting-list-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("output/exact_name_evaluation.json"))
    args = parser.parse_args()
    if args.rule == "conservative_multipass":
        report = asdict(
            evaluate_multipass_train(
                args.data_root,
                max_posting_list_size=args.max_posting_list_size,
                progress=True,
            )
        )
    else:
        report = asdict(evaluate_rule_train(args.data_root, args.rule, progress=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
