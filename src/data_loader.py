"""Deterministic, streaming readers for the supplied challenge TSV files.

Records are intentionally represented as strings: entity identifiers must never
be coerced to numbers, and original business text remains available to both
blocking and the downstream matching stage.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

EntitySource = Literal["S1", "S2", "S3"]
ENTITY_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


@dataclass(frozen=True, slots=True)
class EntityRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: EntitySource


@dataclass(frozen=True, slots=True)
class GroundTruthRecord:
    source1_entity_id: str
    matched_entity_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitPaths:
    source1: Path
    source2: Path
    source3: Path
    ground_truth: Path | None = None


def _validate_columns(fieldnames: list[str] | None, expected: tuple[str, ...], path: Path) -> None:
    if fieldnames != list(expected):
        raise ValueError(f"Unexpected columns in {path}: expected {expected}, got {fieldnames}")


def _source_from_id(entity_id: str, path: Path) -> EntitySource:
    prefix = entity_id.partition("-")[0]
    if prefix not in {"S1", "S2", "S3"}:
        raise ValueError(f"Cannot infer source from entity_id {entity_id!r} in {path}")
    return prefix  # type: ignore[return-value]


def _open_dict_reader(path: Path) -> tuple[object, csv.DictReader]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t", restkey="__extra_columns__")
    return handle, reader


def iter_entities(path: Path, expected_source: EntitySource | None = None) -> Iterator[EntityRecord]:
    """Yield entity records one at a time, validating schema and source labels."""
    handle, reader = _open_dict_reader(path)
    try:
        _validate_columns(reader.fieldnames, ENTITY_COLUMNS, path)
        source = expected_source
        for line_number, row in enumerate(reader, start=2):
            if row.get("__extra_columns__"):
                raise ValueError(f"Extra TSV columns at {path}:{line_number}")
            values = tuple(row[column] if row[column] is not None else "" for column in ENTITY_COLUMNS)
            entity_id, business_name, business_address, country = values
            row_source = _source_from_id(entity_id, path)
            if source is None:
                source = row_source
            if row_source != source:
                raise ValueError(f"Mixed entity sources in {path}:{line_number}")
            yield EntityRecord(entity_id, business_name, business_address, country, source)
    finally:
        handle.close()


def iter_ground_truth(path: Path) -> Iterator[GroundTruthRecord]:
    """Yield labels for evaluation only; blank match cells represent no match."""
    handle, reader = _open_dict_reader(path)
    try:
        _validate_columns(reader.fieldnames, GROUND_TRUTH_COLUMNS, path)
        for line_number, row in enumerate(reader, start=2):
            if row.get("__extra_columns__"):
                raise ValueError(f"Extra TSV columns at {path}:{line_number}")
            source1_id = row["source1_entity_id"] or ""
            if _source_from_id(source1_id, path) != "S1":
                raise ValueError(f"Expected S1 ID at {path}:{line_number}")
            raw_matches = row["matched_entity_ids"] or ""
            matches = tuple(identifier for identifier in raw_matches.split(",") if identifier)
            if any(_source_from_id(identifier, path) == "S1" for identifier in matches):
                raise ValueError(f"Ground truth includes S1 match at {path}:{line_number}")
            yield GroundTruthRecord(source1_id, matches)
    finally:
        handle.close()


def discover_split(data_root: Path, split: Literal["train", "test"]) -> SplitPaths:
    """Discover source files by their TSV schema and IDs, not column position."""
    directory = data_root / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing split directory: {directory}")
    entity_paths: dict[EntitySource, Path] = {}
    ground_truth: Path | None = None
    for path in sorted(directory.glob("*.tsv")):
        handle, reader = _open_dict_reader(path)
        try:
            if reader.fieldnames == list(ENTITY_COLUMNS):
                first_row = next(reader, None)
                if first_row is None:
                    raise ValueError(f"Entity TSV is empty: {path}")
                source = _source_from_id(first_row["entity_id"] or "", path)
                if source in entity_paths:
                    raise ValueError(f"Multiple files found for {split} {source}")
                entity_paths[source] = path
            elif reader.fieldnames == list(GROUND_TRUTH_COLUMNS):
                if ground_truth is not None:
                    raise ValueError(f"Multiple ground-truth files found for {split}")
                ground_truth = path
        finally:
            handle.close()
    missing = {"S1", "S2", "S3"} - set(entity_paths)
    if missing:
        raise FileNotFoundError(f"Missing entity sources in {directory}: {sorted(missing)}")
    if split == "train" and ground_truth is None:
        raise FileNotFoundError(f"Missing ground truth in {directory}")
    return SplitPaths(entity_paths["S1"], entity_paths["S2"], entity_paths["S3"], ground_truth)
