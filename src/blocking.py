"""Explainable, memory-conscious blocking rules for entity-resolution candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from src.data_loader import EntityRecord, EntitySource
from src.preprocessing import normalize_country, normalize_name

BlockKey = tuple[str, str]


@dataclass(frozen=True, slots=True, order=True)
class CandidatePair:
    """An allowed S1-to-S2/S3 candidate pair, retaining its target source."""

    source1_entity_id: str
    candidate_entity_id: str
    candidate_source: EntitySource
    rule: str = "exact_country_name"


def exact_country_name_key(record: EntityRecord) -> BlockKey | None:
    """Return the baseline key, rejecting empty fields that would over-block."""
    country = normalize_country(record.country)
    name = normalize_name(record.business_name)
    if not country or not name:
        return None
    return country, name


def build_exact_name_index(records: Iterable[EntityRecord]) -> dict[BlockKey, list[str]]:
    """Build an inverted index without retaining full source records.

    Posting lists preserve input order, ensuring repeatable candidate ordering.
    """
    index: dict[BlockKey, list[str]] = defaultdict(list)
    for record in records:
        if record.source == "S1":
            raise ValueError("An exact-name target index must contain only S2 or S3 entities")
        key = exact_country_name_key(record)
        if key is not None:
            posting_list = index[key]
            # Source IDs are expected to be unique, but retain this defensive
            # guard so malformed input cannot create duplicate candidate pairs.
            if record.entity_id not in posting_list:
                posting_list.append(record.entity_id)
    return dict(index)


def iter_exact_name_candidates(
    source1_records: Iterable[EntityRecord],
    target_index: dict[BlockKey, list[str]],
    target_source: EntitySource,
) -> Iterator[CandidatePair]:
    """Yield deterministic S1-target pairs for a previously built target index."""
    if target_source == "S1":
        raise ValueError("Candidate target source must be S2 or S3")
    for record in source1_records:
        if record.source != "S1":
            raise ValueError("Candidate left side must contain only S1 entities")
        key = exact_country_name_key(record)
        if key is None:
            continue
        for candidate_id in target_index.get(key, ()):
            if not candidate_id.startswith(f"{target_source}-"):
                raise ValueError("Target index contains an entity from a different source")
            yield CandidatePair(record.entity_id, candidate_id, target_source)


def iter_exact_name_candidates_for_targets(
    source1_factory: Callable[[], Iterable[EntityRecord]],
    source2_records: Iterable[EntityRecord],
    source3_records: Iterable[EntityRecord],
) -> Iterator[CandidatePair]:
    """Generate baseline candidates while holding only one target index at once.

    `source1_factory` must return a fresh iterable because S1 is streamed once
    per target source. This bounds peak index memory to one of S2 or S3.
    """
    for target_source, target_records in (("S2", source2_records), ("S3", source3_records)):
        index = build_exact_name_index(target_records)
        yield from iter_exact_name_candidates(source1_factory(), index, target_source)
