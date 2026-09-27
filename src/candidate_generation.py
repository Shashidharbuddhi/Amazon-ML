"""Final candidate-pair generation for downstream matching."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from src.blocking import (
    CandidatePair,
    KeyFunction,
    country_name_signature_key,
    exact_country_address_key,
    feature_normalized_name_key,
    iter_multipass_candidates_for_targets,
)
from src.data_loader import EntityRecord, iter_entities, discover_split


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Deterministic blocking configuration for production candidate output."""

    passes: tuple[tuple[str, KeyFunction], ...] = (
        ("exact_country_address", exact_country_address_key),
        ("country_name_signature", country_name_signature_key),
        ("feature_normalized_name", feature_normalized_name_key),
    )
    max_posting_list_size: int = 100

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("At least one blocking pass is required")
        if self.max_posting_list_size < 1:
            raise ValueError("max_posting_list_size must be positive")


def generate_candidates(
    source1_factory: Callable[[], Iterable[EntityRecord]],
    source2_records: Iterable[EntityRecord],
    source3_records: Iterable[EntityRecord],
    config: CandidateGenerationConfig | None = None,
) -> Iterable[CandidatePair]:
    """Return a deterministic iterator of deduplicated S1-to-target pairs.

    ``source1_factory`` must return a fresh iterable for each target source.
    Target iterables are consumed once, so callers can stream them from disk.
    """
    selected = config or CandidateGenerationConfig()
    return iter_multipass_candidates_for_targets(
        source1_factory,
        source2_records,
        source3_records,
        selected.passes,
        max_posting_list_size=selected.max_posting_list_size,
    )


def write_candidate_pairs(
    candidates: Iterable[CandidatePair],
    output: TextIO,
    s1_order: Iterable[str] | None = None,
) -> int:
    """Write the official candidate TSV schema (source1_entity_id \\t candidate_entity_ids).

    Returns total number of candidate pairs written.
    """
    s1_candidates: dict[str, list[str]] = defaultdict(list)
    total_pairs = 0
    s1_seen_order: list[str] = []

    for pair in candidates:
        if pair.source1_entity_id not in s1_candidates:
            s1_seen_order.append(pair.source1_entity_id)
        s1_candidates[pair.source1_entity_id].append(pair.candidate_entity_id)
        total_pairs += 1

    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(("source1_entity_id", "candidate_entity_ids"))

    target_s1_order = list(s1_order) if s1_order is not None else s1_seen_order

    for s1_id in target_s1_order:
        cands = s1_candidates.get(s1_id, [])
        # Deduplicate candidates per S1 while preserving order
        unique_cands = list(dict.fromkeys(cands))
        writer.writerow((s1_id, ",".join(unique_cands)))

    return total_pairs


def generate_train_or_test_file(
    data_root: Path,
    output_path: Path,
    *,
    split: Literal["train", "test"] = "test",
    max_posting_list_size: int = 100,
) -> int:
    """Generate a candidate TSV from one supplied data split."""
    paths = discover_split(data_root, split)
    s1_ids = [r.entity_id for r in iter_entities(paths.source1, "S1")]
    candidates = generate_candidates(
        lambda: iter_entities(paths.source1, "S1"),
        iter_entities(paths.source2, "S2"),
        iter_entities(paths.source3, "S3"),
        CandidateGenerationConfig(max_posting_list_size=max_posting_list_size),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        return write_candidate_pairs(candidates, handle, s1_order=s1_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic candidate pairs.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--max-posting-list-size", type=int, default=100)
    args = parser.parse_args()
    count = generate_train_or_test_file(
        args.data_root,
        args.output,
        split=args.split,
        max_posting_list_size=args.max_posting_list_size,
    )
    print(f"candidate_pairs={count:,}")


if __name__ == "__main__":
    main()