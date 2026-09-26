"""
features.py: Pairwise feature computation between Source 1 record and candidate (Source 2 / 3) record.
"""
from typing import Dict, Any, Set
from rapidfuzz import fuzz
from src.normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_numeric_tokens
)

def compute_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection) / float(union) if union > 0 else 0.0

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def build_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
    s1_norm_name: str = None,
    s1_norm_addr: str = None,
    cand_norm_name: str = None,
    cand_norm_addr: str = None,
) -> Dict[str, float]:
    """
    Build rich pairwise similarity features for a Source 1 and candidate pair.
    Allows passing pre-normalized strings for fast caching.
    """
    # Normalize if not pre-normalized
    norm_n1 = s1_norm_name if s1_norm_name is not None else normalize_name(s1_name)
    norm_n2 = cand_norm_name if cand_norm_name is not None else normalize_name(cand_name)
    
    norm_a1 = s1_norm_addr if s1_norm_addr is not None else normalize_address(s1_addr)
    norm_a2 = cand_norm_addr if cand_norm_addr is not None else normalize_address(cand_addr)
    
    c1 = normalize_country(s1_country)
    c2 = normalize_country(cand_country)

    feats = {}

    # --- Missingness Features ---
    name1_empty = (len(norm_n1) == 0)
    name2_empty = (len(norm_n2) == 0)
    addr1_empty = (len(norm_a1) == 0)
    addr2_empty = (len(norm_a2) == 0)
    
    feats["name_missing"] = 1.0 if (name1_empty or name2_empty) else 0.0
    feats["address_missing"] = 1.0 if (addr1_empty or addr2_empty) else 0.0
    feats["country_missing"] = 1.0 if (not c1 or not c2) else 0.0

    # --- Country Feature ---
    feats["country_match"] = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    # --- Name Features ---
    if name1_empty or name2_empty:
        feats["name_exact"] = 0.0
        feats["name_ratio"] = 0.0
        feats["name_wratio"] = 0.0
        feats["name_token_set_ratio"] = 0.0
        feats["name_token_sort_ratio"] = 0.0
        feats["name_token_jaccard"] = 0.0
        feats["name_char_jaccard"] = 0.0
        feats["name_length_sim"] = 0.0
        feats["name_first_token_match"] = 0.0
    else:
        feats["name_exact"] = 1.0 if norm_n1 == norm_n2 else 0.0
        feats["name_ratio"] = fuzz.ratio(norm_n1, norm_n2) / 100.0
        feats["name_wratio"] = fuzz.WRatio(norm_n1, norm_n2) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
        
        tokens1 = set(norm_n1.split())
        tokens2 = set(norm_n2.split())
        feats["name_token_jaccard"] = compute_jaccard(tokens1, tokens2)
        
        ngrams1 = get_char_ngrams(norm_n1, 3)
        ngrams2 = get_char_ngrams(norm_n2, 3)
        feats["name_char_jaccard"] = compute_jaccard(ngrams1, ngrams2)
        
        len1, len2 = len(norm_n1), len(norm_n2)
        feats["name_length_sim"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
        
        t1_first = norm_n1.split()[0] if norm_n1.split() else ""
        t2_first = norm_n2.split()[0] if norm_n2.split() else ""
        feats["name_first_token_match"] = 1.0 if (t1_first and t1_first == t2_first) else 0.0

    # --- Address Features ---
    if addr1_empty or addr2_empty:
        feats["address_exact"] = 0.0
        feats["address_ratio"] = 0.0
        feats["address_token_set_ratio"] = 0.0
        feats["address_token_sort_ratio"] = 0.0
        feats["address_token_jaccard"] = 0.0
        feats["address_char_jaccard"] = 0.0
        feats["address_length_sim"] = 0.0
        feats["address_numeric_overlap"] = 0.0
        feats["address_numeric_mismatch"] = 0.0
    else:
        feats["address_exact"] = 1.0 if norm_a1 == norm_a2 else 0.0
        feats["address_ratio"] = fuzz.ratio(norm_a1, norm_a2) / 100.0
        feats["address_token_set_ratio"] = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        feats["address_token_sort_ratio"] = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        
        addr_tokens1 = set(norm_a1.split())
        addr_tokens2 = set(norm_a2.split())
        feats["address_token_jaccard"] = compute_jaccard(addr_tokens1, addr_tokens2)
        
        addr_ngrams1 = get_char_ngrams(norm_a1, 3)
        addr_ngrams2 = get_char_ngrams(norm_a2, 3)
        feats["address_char_jaccard"] = compute_jaccard(addr_ngrams1, addr_ngrams2)
        
        alen1, alen2 = len(norm_a1), len(norm_a2)
        feats["address_length_sim"] = min(alen1, alen2) / max(alen1, alen2) if max(alen1, alen2) > 0 else 0.0
        
        # Numeric tokens in addresses (building numbers, zip codes)
        nums1 = extract_numeric_tokens(norm_a1)
        nums2 = extract_numeric_tokens(norm_a2)
        if nums1 and nums2:
            feats["address_numeric_overlap"] = compute_jaccard(nums1, nums2)
            feats["address_numeric_mismatch"] = 1.0 if len(nums1 & nums2) == 0 else 0.0
        else:
            feats["address_numeric_overlap"] = 0.0
            feats["address_numeric_mismatch"] = 0.0

    # --- Interaction Features ---
    feats["name_addr_avg"] = (feats["name_ratio"] + feats["address_ratio"]) / 2.0
    feats["name_high_addr_high"] = feats["name_ratio"] * feats["address_ratio"]
    feats["strong_name_and_addr"] = 1.0 if (feats["name_ratio"] >= 0.85 and feats["address_ratio"] >= 0.70) else 0.0
    feats["country_and_name_agreement"] = feats["country_match"] * feats["name_ratio"]

    return feats
