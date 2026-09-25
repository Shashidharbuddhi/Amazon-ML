"""
inspect_data.py: Inspect dataset files, column names, data types, missingness, and ground truth statistics.
"""
import os
import sys
from collections import Counter
from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
)
from src.schema import DELIM

def inspect_tsv(path, name, max_rows=100000):
    print(f"\n==========================================")
    print(f"File: {name} ({os.path.basename(path)})")
    print(f"Path: {path}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND!")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    total_rows = 0
    missing_counts = Counter()
    countries = Counter()
    id_prefixes = Counter()
    first_row = None

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        header = [c.strip() for c in header_line.rstrip("\n").split(DELIM)]
        print(f"Columns ({len(header)}): {header}")

        for i, line in enumerate(f):
            total_rows += 1
            parts = line.rstrip("\n").split(DELIM)
            if i == 0:
                first_row = dict(zip(header, parts))
            
            # Check ID prefix if entity_id in header
            if "entity_id" in header:
                idx = header.index("entity_id")
                if idx < len(parts) and "-" in parts[idx]:
                    id_prefixes[parts[idx].split("-")[0]] += 1
            elif "source1_entity_id" in header:
                idx = header.index("source1_entity_id")
                if idx < len(parts) and "-" in parts[idx]:
                    id_prefixes[parts[idx].split("-")[0]] += 1

            if "country" in header:
                idx = header.index("country")
                c_val = parts[idx] if idx < len(parts) else ""
                countries[c_val] += 1

            for col_idx, col_name in enumerate(header):
                val = parts[col_idx] if col_idx < len(parts) else ""
                if not val.strip():
                    missing_counts[col_name] += 1

    print(f"Total Rows: {total_rows}")
    print(f"ID Prefixes: {dict(id_prefixes)}")
    if countries:
        print(f"Country Distribution: {dict(countries)}")
    print(f"Missing Values: {dict(missing_counts)}")
    print(f"Sample First Row: {first_row}")
    return total_rows, header

def inspect_ground_truth(path):
    print(f"\n==========================================")
    print(f"Inspecting Ground Truth: {path}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND!")
        return

    total_s1 = 0
    zero_matches = 0
    match_counts = Counter()
    s2_matches = 0
    s3_matches = 0
    total_matches = 0

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        print(f"Ground Truth Header: {header}")
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\n").split(DELIM)
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if not matched_str.strip():
                zero_matches += 1
                match_counts[0] += 1
            else:
                m_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                n = len(m_ids)
                match_counts[n] += 1
                total_matches += n
                for mid in m_ids:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1

    print(f"Total S1 entities in Ground Truth: {total_s1}")
    print(f"Zero-match (Singletons): {zero_matches} ({zero_matches / total_s1 * 100:.2f}%)")
    print(f"Entities with >= 1 match: {total_s1 - zero_matches} ({(total_s1 - zero_matches) / total_s1 * 100:.2f}%)")
    print(f"Total matches across all S1: {total_matches} (S2: {s2_matches}, S3: {s3_matches})")
    print(f"Average matches per non-singleton S1: {total_matches / max(1, total_s1 - zero_matches):.2f}")
    print(f"Match count frequency distribution: {sorted(match_counts.items())[:15]}")

def main():
    print("=== INSPECTING TRAINING FILES ===")
    inspect_tsv(TRAIN_SOURCE1, "Train Source 1")
    inspect_tsv(TRAIN_SOURCE2, "Train Source 2")
    inspect_tsv(TRAIN_SOURCE3, "Train Source 3")
    inspect_ground_truth(TRAIN_GROUND_TRUTH)

    print("\n=== INSPECTING TEST FILES ===")
    inspect_tsv(TEST_SOURCE1, "Test Source 1")
    inspect_tsv(TEST_SOURCE2, "Test Source 2")
    inspect_tsv(TEST_SOURCE3, "Test Source 3")

if __name__ == "__main__":
    main()
