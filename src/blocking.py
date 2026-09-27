"""Explainable, memory-conscious blocking rules for entity-resolution candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass

import re
from src.data_loader import EntityRecord, EntitySource
from src.preprocessing import normalize_country, normalize_name
from src.preprocessing import normalize_address, name_tokens, address_tokens
from src.normalization import normalize_name as feature_normalize_name, extract_numeric_tokens

BlockKey = tuple[str, str]
KeyFunction = Callable[[EntityRecord], Hashable | None]


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


def exact_country_address_key(record: EntityRecord) -> BlockKey | None:
    """Return a country/address key, excluding blank normalized addresses."""
    country = normalize_country(record.country)
    address = normalize_address(record.business_address)
    if not country or not address:
        return None
    return country, address


def country_name_signature_key(record: EntityRecord) -> tuple[str, tuple[str, ...]] | None:
    """Return an order-insensitive full-name signature for token reordering."""
    country = normalize_country(record.country)
    tokens = name_tokens(record.business_name)
    if not country or not tokens:
        return None
    return country, tuple(sorted(set(tokens)))


def feature_normalized_name_key(record: EntityRecord) -> BlockKey | None:
    """Return the feature-module country/name key used by the matcher."""
    country = normalize_country(record.country)
    name = feature_normalize_name(record.business_name)
    if not country or not name:
        return None
    return country, name


_DOMAIN_REGEX = re.compile(
    r"\b(www\.)?([a-z0-9\-]+)\.(com|org|net|in|co|us|biz|info|io|gov|edu|co\.in)\b",
    re.IGNORECASE,
)
_WHITESPACE_COLLAPSE = re.compile(r"\s+")


def compact_name_key(record: EntityRecord) -> BlockKey | None:
    """Full name with web domain extensions and whitespace stripped."""
    country = normalize_country(record.country)
    if not country or not record.business_name:
        return None
    name = _DOMAIN_REGEX.sub(r"\2", record.business_name.lower())
    name = re.sub(r"\.(?=\s|[a-zA-Z]|$)", "", name)
    norm = feature_normalize_name(name)
    compact = _WHITESPACE_COLLAPSE.sub("", norm)
    if not compact or len(compact) < 3:
        return None
    return country, compact


def country_address_signature_key(record: EntityRecord) -> tuple[str, tuple[str, ...]] | None:
    """Order-insensitive complete address token signature."""
    country = normalize_country(record.country)
    tokens = address_tokens(record.business_address)
    if not country or not tokens:
        return None
    return country, tuple(sorted(set(tokens)))


def country_longest_token_addr_num_key(record: EntityRecord) -> tuple[str, str, str] | None:
    """Country + longest distinctive business name token (>=4 chars) + address number."""
    country = normalize_country(record.country)
    if not country or not record.business_name or not record.business_address:
        return None
    tokens = [t for t in name_tokens(record.business_name) if len(t) >= 4]
    if not tokens:
        tokens = [t for t in name_tokens(record.business_name) if len(t) >= 3]
    if not tokens:
        return None
    longest = sorted(tokens, key=lambda t: (-len(t), t))[0]
    nums = extract_numeric_tokens(record.business_address)
    if not nums:
        return None
    primary_num = sorted(nums, key=lambda x: (-len(x), x))[0]
    return country, longest, primary_num


def country_first_2_name_tokens_key(record: EntityRecord) -> tuple[str, str, str] | None:
    """Country + first 2 non-trivial name tokens (minimum 2 chars each)."""
    country = normalize_country(record.country)
    if not country or not record.business_name:
        return None
    tokens = [t for t in name_tokens(record.business_name) if len(t) >= 2]
    if len(tokens) < 2:
        return None
    return country, tokens[0], tokens[1]


def build_inverted_index(
    records: Iterable[EntityRecord], key_function: KeyFunction = exact_country_name_key
) -> dict[Hashable, list[str]]:
    """Build an inverted index without retaining full source records.

    Posting lists preserve input order, ensuring repeatable candidate ordering.
    """
    index: dict[Hashable, list[str]] = defaultdict(list)
    seen_entity_ids: set[str] = set()
    for record in records:
        if record.source == "S1":
            raise ValueError("An exact-name target index must contain only S2 or S3 entities")
        if record.entity_id in seen_entity_ids:
            continue
        seen_entity_ids.add(record.entity_id)
        key = key_function(record)
        if key is not None:
            index[key].append(record.entity_id)
    return dict(index)


def build_exact_name_index(records: Iterable[EntityRecord]) -> dict[BlockKey, list[str]]:
    """Compatibility wrapper for the baseline country/name index."""
    return build_inverted_index(records, exact_country_name_key)  # type: ignore[return-value]


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


def iter_keyed_candidates(
    source1_records: Iterable[EntityRecord],
    target_index: dict[Hashable, list[str]],
    target_source: EntitySource,
    key_function: KeyFunction,
    rule: str,
) -> Iterator[CandidatePair]:
    """Yield candidates for a supplied deterministic blocking-key function."""
    if target_source == "S1":
        raise ValueError("Candidate target source must be S2 or S3")
    for record in source1_records:
        if record.source != "S1":
            raise ValueError("Candidate left side must contain only S1 entities")
        key = key_function(record)
        if key is None:
            continue
        for candidate_id in target_index.get(key, ()):
            if not candidate_id.startswith(f"{target_source}-"):
                raise ValueError("Target index contains an entity from a different source")
            yield CandidatePair(record.entity_id, candidate_id, target_source, rule)


def iter_exact_name_candidates_for_targets(
    source1_factory: Callable[[], Iterable[EntityRecord]],
    source2_records: Iterable[EntityRecord],
    source3_records: Iterable[EntityRecord],
    on_target_index_ready: Callable[[EntitySource, int], None] | None = None,
) -> Iterator[CandidatePair]:
    """Generate baseline candidates while holding only one target index at once.

    `source1_factory` must return a fresh iterable because S1 is streamed once
    per target source. This bounds peak index memory to one of S2 or S3.
    """
    for target_source, target_records in (("S2", source2_records), ("S3", source3_records)):
        index = build_exact_name_index(target_records)
        if on_target_index_ready is not None:
            on_target_index_ready(target_source, len(index))
        yield from iter_exact_name_candidates(source1_factory(), index, target_source)


def iter_keyed_candidates_for_targets(
    source1_factory: Callable[[], Iterable[EntityRecord]],
    source2_records: Iterable[EntityRecord],
    source3_records: Iterable[EntityRecord],
    key_function: KeyFunction,
    rule: str,
    on_target_index_ready: Callable[[EntitySource, int], None] | None = None,
) -> Iterator[CandidatePair]:
    """Generate a single blocking rule while indexing S2 and S3 separately."""
    for target_source, target_records in (("S2", source2_records), ("S3", source3_records)):
        index = build_inverted_index(target_records, key_function)
        if on_target_index_ready is not None:
            on_target_index_ready(target_source, len(index))
        yield from iter_keyed_candidates(source1_factory(), index, target_source, key_function, rule)


def iter_multipass_candidates_for_targets(
    source1_factory: Callable[[], Iterable[EntityRecord]],
    source2_records: Iterable[EntityRecord],
    source3_records: Iterable[EntityRecord],
    passes: tuple[tuple[str, KeyFunction], ...],
    *,
    max_posting_list_size: int,
    on_target_index_ready: Callable[[EntitySource, str, int, int], None] | None = None,
) -> Iterator[CandidatePair]:
    """Yield the deduplicated union of guarded blocking passes.

    A pass is disabled for a target key when its posting list exceeds
    ``max_posting_list_size``. This bounds the candidates contributed by each
    pass while retaining the useful small and distinctive blocks.
    """
    if max_posting_list_size < 1:
        raise ValueError("max_posting_list_size must be positive")
    if not passes:
        return

    for target_source, target_records in (("S2", source2_records), ("S3", source3_records)):
        raw_indexes: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in passes]
        blocked_keys: list[set[Hashable]] = [set() for _ in passes]
        seen_entity_ids: set[str] = set()
        for record in target_records:
            if record.source == "S1":
                raise ValueError("A target index must contain only S2 or S3 entities")
            if record.entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(record.entity_id)
            for pass_number, (index, (_, key_function)) in enumerate(zip(raw_indexes, passes)):
                key = key_function(record)
                if key is None or key in blocked_keys[pass_number]:
                    continue
                posting_list = index[key]
                if len(posting_list) >= max_posting_list_size:
                    del index[key]
                    blocked_keys[pass_number].add(key)
                    continue
                posting_list.append(record.entity_id)

        indexes: list[tuple[str, KeyFunction, dict[Hashable, list[str]]]] = []
        for pass_number, ((rule, key_function), raw_index) in enumerate(zip(passes, raw_indexes)):
            indexes.append((rule, key_function, raw_index))
            if on_target_index_ready is not None:
                on_target_index_ready(
                    target_source,
                    rule,
                    len(raw_index),
                    len(raw_index) + len(blocked_keys[pass_number]),
                )

        for record in source1_factory():
            if record.source != "S1":
                raise ValueError("Candidate left side must contain only S1 entities")
            emitted: set[str] = set()
            for rule, key_function, index in indexes:
                key = key_function(record)
                if key is None:
                    continue
                for candidate_id in index.get(key, ()):
                    if candidate_id in emitted:
                        continue
                    if not candidate_id.startswith(f"{target_source}-"):
                        raise ValueError("Target index contains an entity from a different source")
                    emitted.add(candidate_id)
                    yield CandidatePair(record.entity_id, candidate_id, target_source, rule)


def iter_multipass_candidate_records_for_target(
    source1_records: Iterable[EntityRecord],
    target_records: Iterable[EntityRecord],
    target_source: EntitySource,
    passes: tuple[tuple[str, KeyFunction], ...],
    *,
    max_posting_list_size: int,
) -> Iterator[tuple[CandidatePair, EntityRecord, EntityRecord]]:
    """Yield Person 1 candidates with their target record for one target source.

    This is the record-aware counterpart of
    :func:`iter_multipass_candidates_for_targets`.  It uses exactly the same
    keys, posting-list guard, ordering, and per-S1 de-duplication, but retains
    a single target source's raw records so a downstream matcher can score
    candidates without materializing both S2 and S3 in memory.
    """
    if target_source == "S1":
        raise ValueError("Candidate target source must be S2 or S3")
    if max_posting_list_size < 1:
        raise ValueError("max_posting_list_size must be positive")

    raw_indexes: list[dict[Hashable, list[str]]] = [defaultdict(list) for _ in passes]
    blocked_keys: list[set[Hashable]] = [set() for _ in passes]
    records_by_id: dict[str, EntityRecord] = {}

    for record in target_records:
        if record.source != target_source:
            raise ValueError("Target records do not match target_source")
        if record.entity_id in records_by_id:
            continue
        records_by_id[record.entity_id] = record
        for pass_number, (index, (_, key_function)) in enumerate(zip(raw_indexes, passes)):
            key = key_function(record)
            if key is None or key in blocked_keys[pass_number]:
                continue
            posting_list = index[key]
            if len(posting_list) >= max_posting_list_size:
                del index[key]
                blocked_keys[pass_number].add(key)
                continue
            posting_list.append(record.entity_id)

    for source1_record in source1_records:
        if source1_record.source != "S1":
            raise ValueError("Candidate left side must contain only S1 entities")
        emitted: set[str] = set()
        for rule, key_function, index in (
            (rule, key_function, index)
            for (rule, key_function), index in zip(passes, raw_indexes)
        ):
            key = key_function(source1_record)
            if key is None:
                continue
            for candidate_id in index.get(key, ()):
                if candidate_id in emitted:
                    continue
                emitted.add(candidate_id)
                yield (
                    CandidatePair(source1_record.entity_id, candidate_id, target_source, rule),
                    source1_record,
                    records_by_id[candidate_id],
                )
