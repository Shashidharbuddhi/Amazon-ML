"""Streaming dataset profiling for the Amazon ML entity-resolution TSV files.

The audit deliberately reads one row at a time.  It reports exact row/missing
counts and bounded-memory estimates for high-cardinality text duplicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable

ENTITY_COLUMNS = ("entity_id", "business_name", "business_address", "country")


class HyperLogLog:
    """Small deterministic cardinality estimator used to avoid retaining text."""

    def __init__(self, precision: int = 14) -> None:
        if not 4 <= precision <= 18:
            raise ValueError("precision must be between 4 and 18")
        self.precision = precision
        self.registers = bytearray(1 << precision)

    def add(self, value: str) -> None:
        # crc32 is sufficient for an audit estimate and is substantially faster
        # than a cryptographic digest on multi-gigabyte input.
        digest = zlib.crc32(value.encode("utf-8"))
        index = digest >> (32 - self.precision)
        remainder = (digest << self.precision) & ((1 << 32) - 1)
        rank = 32 - remainder.bit_length() + 1 if remainder else 33 - self.precision
        self.registers[index] = max(self.registers[index], rank)

    def estimate(self) -> int:
        count = len(self.registers)
        harmonic = sum(2.0 ** -register for register in self.registers)
        alpha = 0.7213 / (1 + 1.079 / count)
        raw = alpha * count * count / harmonic
        zeroes = self.registers.count(0)
        if raw <= 2.5 * count and zeroes:
            raw = count * math.log(count / zeroes)
        return round(raw)


def _text_length(value: str) -> int:
    return len(value) if value else 0


def profile_tsv(path: Path) -> dict[str, object]:
    """Return a bounded-memory profile for a challenge TSV file."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        missing = Counter()
        countries = Counter()
        lengths = {
            "business_name": {"count": 0, "sum": 0, "min": None, "max": 0},
            "business_address": {"count": 0, "sum": 0, "min": None, "max": 0},
        }
        distinct = {column: HyperLogLog() for column in ENTITY_COLUMNS if column in columns}
        rows = 0
        for row in reader:
            rows += 1
            for column in columns:
                value = (row.get(column) or "").strip()
                if not value:
                    missing[column] += 1
                if column in distinct and value:
                    distinct[column].add(value)
            if "country" in columns:
                country = (row.get("country") or "").strip()
                if country:
                    countries[country] += 1
            for column in lengths:
                if column in columns:
                    length = _text_length((row.get(column) or "").strip())
                    stats = lengths[column]
                    stats["count"] += 1
                    stats["sum"] += length
                    stats["min"] = length if stats["min"] is None else min(stats["min"], length)
                    stats["max"] = max(stats["max"], length)

    def length_summary(stats: dict[str, int | None]) -> dict[str, float | int]:
        count = stats["count"]
        if not count:
            return {"min": 0, "mean": 0.0, "max": 0}
        return {
            "min": int(stats["min"] or 0),
            "mean": round(int(stats["sum"] or 0) / count, 2),
            "max": int(stats["max"] or 0),
        }

    estimated_distinct = {column: estimator.estimate() for column, estimator in distinct.items()}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
        "missing_values": dict(sorted(missing.items())),
        "country_distribution": dict(sorted(countries.items())),
        "text_lengths": {column: length_summary(stats) for column, stats in lengths.items()},
        "estimated_distinct_nonempty": estimated_distinct,
        "estimated_duplicate_nonempty": {
            column: max(0, rows - missing.get(column, 0) - estimate)
            for column, estimate in estimated_distinct.items()
        },
        "notes": "Distinct/duplicate text values are HyperLogLog estimates; row and missing counts are exact.",
    }


def discover_tsvs(data_root: Path) -> Iterable[Path]:
    return sorted(data_root.rglob("*.tsv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile challenge TSVs without DataFrame loading.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/data_audit.json"))
    args = parser.parse_args()
    files = list(discover_tsvs(args.data_root))
    if not files:
        raise SystemExit(f"No TSV files found below {args.data_root}")
    report = {"files": [profile_tsv(path) for path in files]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
