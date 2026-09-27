"""
test_blocking_rules.py: Fast exploration of incremental blocking rules on a representative sample of 5,000 S1s.
"""
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
from src.data_loader import iter_entities, iter_ground_truth
from src.preprocessing import normalize_country, normalize_address as prep_norm_addr, normalize_name as prep_norm_name, name_tokens, address_tokens
from src.normalization import normalize_name as feat_norm_name, normalize_address as feat_norm_addr, extract_numeric_tokens

DOMAIN_REGEX = re.compile(r"\b(www\.)?([a-z0-9\-]+)\.(com|org|net|in|co|us|biz|info|io|gov|edu|co\.in)\b", re.IGNORECASE)

def compact_name(name: str) -> str:
    if not name:
        return ""
    # strip domain
    name_clean = DOMAIN_REGEX.sub(r"\2", name.lower())
    # feature normalize
    norm = feat_norm_name(name_clean)
    # remove all spaces
    return re.sub(r"\s+", "", norm)

def address_signature_key(record) -> tuple | None:
    c = normalize_country(record.country)
    tokens = address_tokens(record.business_address)
    if not c or not tokens:
        return None
    return (c, tuple(sorted(set(tokens))))

def feat_address_key(record) -> tuple | None:
    c = normalize_country(record.country)
    a = feat_norm_addr(record.business_address)
    if not c or not a:
        return None
    return (c, a)

def compact_name_key(record) -> tuple | None:
    c = normalize_country(record.country)
    n = compact_name(record.business_name)
    if not c or not n or len(n) < 3:
        return None
    return (c, n)

def name_token_and_address_num_key(record) -> tuple | None:
    c = normalize_country(record.country)
    tokens = [t for t in name_tokens(record.business_name) if len(t) > 2]
    nums = extract_numeric_tokens(record.business_address)
    if not c or not tokens or not nums:
        return None
    # Use first significant name token and lowest/most informative number
    first_token = tokens[0]
    first_num = sorted(nums)[0]
    return (c, first_token, first_num)

def address_num_and_street_key(record) -> tuple | None:
    c = normalize_country(record.country)
    nums = extract_numeric_tokens(record.business_address)
    tokens = [t for t in address_tokens(record.business_address) if t not in nums and len(t) > 2]
    if not c or not nums or not tokens:
        return None
    # first number + first street token
    return (c, sorted(nums)[0], tokens[0])


def test_rules_on_sample(s1_sample_size: int = 5000):
    print(f"Loading {s1_sample_size} S1 records...")
    s1_dict = {}
    with open("dataset/train/train_source1.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for idx, row in enumerate(r):
            if idx >= s1_sample_size: break
            s1_dict[row[0]] = {
                "id": row[0], "name": row[1], "addr": row[2], "country": row[3]
            }

    gt = defaultdict(set)
    with open("dataset/train/train_ground_truth.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in s1_dict:
                gt[row[0]] = set(filter(None, row[1].split(",")))

    total_true = sum(len(v) for v in gt.values())
    print(f"Total true links for {len(s1_dict)} S1s: {total_true:,}")

    # Build S1 keys for candidate rules
    from dataclasses import dataclass
    @dataclass
    class DummyRecord:
        entity_id: str
        business_name: str
        business_address: str
        country: str
        source: str = "S1"

    s1_recs = {
        sid: DummyRecord(sid, d["name"], d["addr"], d["country"])
        for sid, d in s1_dict.items()
    }

    from src.blocking import exact_country_address_key, country_name_signature_key, feature_normalized_name_key

    rules = [
        ("exact_country_address", exact_country_address_key),
        ("country_name_signature", country_name_signature_key),
        ("feature_normalized_name", feature_normalized_name_key),
        ("address_signature", address_signature_key),
        ("feat_address", feat_address_key),
        ("compact_name", compact_name_key),
        ("name_token_addr_num", name_token_and_address_num_key),
        ("addr_num_street", address_num_and_street_key),
    ]

    s1_keys_by_rule = {name: defaultdict(list) for name, _ in rules}
    for sid, rec in s1_recs.items():
        for name, fn in rules:
            k = fn(rec)
            if k is not None:
                s1_keys_by_rule[name][k].append(sid)

    # Now stream targets and measure recall and candidate counts per rule
    postings_by_rule = {name: defaultdict(list) for name, _ in rules}
    blocked_by_rule = {name: set() for name, _ in rules}
    CAP = 100

    print("Scanning S2 and S3 targets...")
    for target_path, target_source in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
        with open(target_path) as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                tid, name, addr, country = row[0], row[1], row[2], row[3]
                rec = DummyRecord(tid, name, addr, country, target_source)
                for rname, fn in rules:
                    k = fn(rec)
                    if k is None or k in blocked_by_rule[rname]:
                        continue
                    if k not in s1_keys_by_rule[rname]:
                        continue
                    plist = postings_by_rule[rname][k]
                    if len(plist) >= CAP:
                        del postings_by_rule[rname][k]
                        blocked_by_rule[rname].add(k)
                        continue
                    plist.append(tid)

    # Measure performance of each rule individually
    print("\n--- INDIVIDUAL RULE PERFORMANCE (on 5,000 S1 sample) ---")
    rule_cands = {}
    for rname, _ in rules:
        cands_by_s1 = defaultdict(set)
        for k, sids in s1_keys_by_rule[rname].items():
            targets = postings_by_rule[rname].get(k, ())
            for sid in sids:
                cands_by_s1[sid].update(targets)
        
        rule_cands[rname] = cands_by_s1
        tot_cands = sum(len(c) for c in cands_by_s1.values())
        found_true = sum(len(cands_by_s1[sid] & gt[sid]) for sid in s1_dict)
        print(f"Rule: {rname:25s} | Recall: {found_true / total_true:7.2%} ({found_true:,}/{total_true:,}) | Candidates: {tot_cands:,} ({tot_cands / len(s1_dict):.1f}/S1)")

    # Measure cumulative unions
    print("\n--- CUMULATIVE INCREMENTAL PERFORMANCE ---")
    current_union = defaultdict(set)
    for rname in [
        "exact_country_address",
        "country_name_signature",
        "feature_normalized_name",
        "address_signature",
        "feat_address",
        "compact_name",
        "name_token_addr_num",
        "addr_num_street",
    ]:
        for sid, cset in rule_cands[rname].items():
            current_union[sid].update(cset)
        tot_cands = sum(len(c) for c in current_union.values())
        found_true = sum(len(current_union[sid] & gt[sid]) for sid in s1_dict)
        zero_s1 = sum(1 for sid in s1_dict if len(current_union[sid]) == 0)
        print(f"+ {rname:23s} | Total Recall: {found_true / total_true:7.2%} ({found_true:,}/{total_true:,}) | Total Cands: {tot_cands:,} ({tot_cands / len(s1_dict):.1f}/S1) | Zero-cand S1: {zero_s1}")

if __name__ == "__main__":
    test_rules_on_sample(5000)
