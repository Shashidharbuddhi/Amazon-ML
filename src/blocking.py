"""
blocking.py: Multi-key high-recall candidate generation (Person 1 candidate interface).
Blocks records across countries, normalized names, name prefixes, and address signatures.
"""
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from src.normalization import (
    normalize_name, normalize_address, normalize_country, extract_numeric_tokens
)

def extract_blocking_keys(norm_name: str, norm_addr: str, country: str) -> List[Tuple[str, str]]:
    """
    Generate multiple complementary blocking keys for an entity.
    Returns list of (key_type, key_value) tuples.
    All keys are country-scoped.
    """
    keys = []
    c = normalize_country(country)
    if not c:
        return keys

    # Key 1: Exact normalized name + country
    if norm_name and len(norm_name) >= 3:
        keys.append(("name_exact", f"{c}|{norm_name}"))

    # Key 2: First 2 tokens of business name
    tokens = norm_name.split()
    if len(tokens) >= 2:
        two_tok = f"{tokens[0]}_{tokens[1]}"
        if len(two_tok) >= 5:
            keys.append(("name_2tok", f"{c}|{two_tok}"))
    elif len(tokens) == 1 and len(tokens[0]) >= 4:
        keys.append(("name_1tok", f"{c}|{tokens[0]}"))

    # Key 3: First token (>=4 chars) + first address number
    numbers = extract_numeric_tokens(norm_addr)
    if tokens and len(tokens[0]) >= 4 and numbers:
        sorted_nums = sorted(numbers)
        keys.append(("tok_num", f"{c}|{tokens[0]}_{sorted_nums[0]}"))

    # Key 4: Full normalized address (catches transliterated names)
    if norm_addr and len(norm_addr) >= 10:
        keys.append(("addr_exact", f"{c}|{norm_addr}"))

    return keys
