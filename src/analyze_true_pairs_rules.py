"""
analyze_true_pairs_rules.py: Evaluate true-link recall for proposed candidate-generation rules directly on true pairs.
Runs in seconds and gives exact recall breakdown by country and source.
"""
import csv
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from src.preprocessing import normalize_country, normalize_address as prep_norm_addr, normalize_name as prep_norm_name, name_tokens, address_tokens
from src.normalization import normalize_name as feat_norm_name, normalize_address as feat_norm_addr, extract_numeric_tokens

@dataclass
class Rec:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str

DOMAIN_REGEX = re.compile(r"\b(www\.)?([a-z0-9\-]+)\.(com|org|net|in|co|us|biz|info|io|gov|edu|co\.in)\b", re.IGNORECASE)

def compact_name(name: str) -> str:
    if not name:
        return ""
    name_clean = DOMAIN_REGEX.sub(r"\2", name.lower())
    norm = feat_norm_name(name_clean)
    return re.sub(r"\s+", "", norm)

def strip_dots_norm_name(name: str) -> str:
    if not name:
        return ""
    # strip periods in abbreviations like L.L.C. or Pvt. Ltd. before legal suffix removal
    cleaned = re.sub(r"\.(?=\s|[a-zA-Z]|$)", "", name)
    return feat_norm_name(cleaned)

def address_signature_key(r: Rec):
    c = normalize_country(r.country)
    toks = address_tokens(r.business_address)
    if not c or not toks:
        return None
    return (c, tuple(sorted(set(toks))))

def feat_address_key(r: Rec):
    c = normalize_country(r.country)
    a = feat_norm_addr(r.business_address)
    if not c or not a:
        return None
    return (c, a)

def compact_name_key(r: Rec):
    c = normalize_country(r.country)
    n = compact_name(r.business_name)
    if not c or not n or len(n) < 3:
        return None
    return (c, n)

def fixed_feature_name_key(r: Rec):
    c = normalize_country(r.country)
    n = strip_dots_norm_name(r.business_name)
    if not c or not n:
        return None
    return (c, n)

def name_token_and_address_num_key(r: Rec):
    c = normalize_country(r.country)
    toks = [t for t in name_tokens(r.business_name) if len(t) > 2]
    nums = extract_numeric_tokens(r.business_address)
    if not c or not toks or not nums:
        return None
    return (c, toks[0], sorted(nums)[0])

def address_num_and_street_key(r: Rec):
    c = normalize_country(r.country)
    nums = extract_numeric_tokens(r.business_address)
    toks = [t for t in address_tokens(r.business_address) if t not in nums and len(t) > 2]
    if not c or not nums or not toks:
        return None
    return (c, sorted(nums)[0], toks[0])

def first_2_name_tokens_key(r: Rec):
    c = normalize_country(r.country)
    toks = [t for t in name_tokens(r.business_name) if len(t) > 1]
    if not c or len(toks) < 2:
        return None
    return (c, toks[0], toks[1])

def zipcode_and_first_name_token_key(r: Rec):
    c = normalize_country(r.country)
    toks = [t for t in name_tokens(r.business_name) if len(t) > 2]
    nums = extract_numeric_tokens(r.business_address)
    # Check for 5-digit US zip or 6-digit Indian PIN
    zipcodes = [n for n in nums if len(n) in (5, 6)]
    if not c or not toks or not zipcodes:
        return None
    return (c, zipcodes[0], toks[0])


def main():
    print("Loading 10,000 S1 sample and their true targets...")
    s1_dict = {}
    with open("dataset/train/train_source1.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for idx, row in enumerate(r):
            if idx >= 10000: break
            s1_dict[row[0]] = Rec(row[0], row[1], row[2], row[3], "S1")

    gt = defaultdict(set)
    with open("dataset/train/train_ground_truth.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in s1_dict:
                gt[row[0]] = set(filter(None, row[1].split(",")))

    needed_targets = {t for mids in gt.values() for t in mids}
    target_dict = {}
    for path, src in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
        with open(path) as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                if row[0] in needed_targets:
                    target_dict[row[0]] = Rec(row[0], row[1], row[2], row[3], src)
                    if len(target_dict) == len(needed_targets):
                        break

    true_pairs = []
    for s1_id, mids in gt.items():
        s1 = s1_dict[s1_id]
        for mid in mids:
            if mid in target_dict:
                true_pairs.append((s1, target_dict[mid]))

    total_true = len(true_pairs)
    print(f"Loaded {total_true:,} true pairs for {len(s1_dict):,} S1 entities.")

    from src.blocking import exact_country_address_key, country_name_signature_key, feature_normalized_name_key

    all_rules = [
        ("exact_country_address (R1)", exact_country_address_key),
        ("country_name_signature (R2)", country_name_signature_key),
        ("feature_normalized_name (R3)", feature_normalized_name_key),
        ("fixed_feature_name (strip dots)", fixed_feature_name_key),
        ("address_signature", address_signature_key),
        ("feat_address (abbr expanded)", feat_address_key),
        ("compact_name (domain/spaces)", compact_name_key),
        ("name_token_addr_num", name_token_and_address_num_key),
        ("addr_num_street", address_num_and_street_key),
        ("first_2_name_tokens", first_2_name_tokens_key),
        ("zipcode_first_name_token", zipcode_and_first_name_token_key),
    ]

    print("\n--- INDIVIDUAL TRUE-LINK RECALL ---")
    rule_matches = {}
    for rname, fn in all_rules:
        matched = 0
        s2_matched = 0
        s3_matched = 0
        s2_total = sum(1 for _, t in true_pairs if t.source == "S2")
        s3_total = sum(1 for _, t in true_pairs if t.source == "S3")
        pair_match_set = set()
        for idx, (s1, tgt) in enumerate(true_pairs):
            k1 = fn(s1)
            k2 = fn(tgt)
            if k1 is not None and k1 == k2:
                matched += 1
                pair_match_set.add(idx)
                if tgt.source == "S2": s2_matched += 1
                else: s3_matched += 1
        rule_matches[rname] = pair_match_set
        print(f"Rule: {rname:35s} | Recall: {matched / total_true:7.2%} ({matched:,}/{total_true:,}) | S2: {s2_matched/s2_total:6.2%} | S3: {s3_matched/s3_total:6.2%}")

    print("\n--- CUMULATIVE RECALL COMBINATIONS ---")
    current_union = set()
    for rname in [
        "exact_country_address (R1)",
        "country_name_signature (R2)",
        "feature_normalized_name (R3)",
        "fixed_feature_name (strip dots)",
        "compact_name (domain/spaces)",
        "address_signature",
        "feat_address (abbr expanded)",
        "name_token_addr_num",
        "first_2_name_tokens",
        "addr_num_street",
        "zipcode_first_name_token",
    ]:
        current_union.update(rule_matches[rname])
        print(f"+ {rname:35s} -> Cumulative Recall: {len(current_union) / total_true:7.2%} ({len(current_union):,}/{total_true:,})")

if __name__ == "__main__":
    main()
