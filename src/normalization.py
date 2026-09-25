"""
normalization.py: Text normalization utilities for names, addresses, and country codes.
"""
import re
import unicodedata

# Common legal suffixes to normalize or strip from business names
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|"
    r"ltd|limited|inc|incorporated|corp|corporation|llc|l\.l\.c\.|"
    r"llp|l\.l\.p\.|co|company|holdings|group|enterprise|enterprises|"
    r"sa|sarl|gmbh|spa|bv|nv"
    r")\b",
    re.IGNORECASE
)

# Common street/address abbreviations
ADDRESS_ABBR_MAP = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bp\.?o\.?\s*box\b": "pobox",
    r"\bno\.?\b": "number",
}
ADDRESS_ABBR_COMPILED = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR_MAP.items()]

def normalize_text(text: str) -> str:
    """General unicode and whitespace normalization."""
    if not text or not isinstance(text, str):
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFKD", text)
    # Remove accents/diacritics while keeping base chars
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    return text

def normalize_name(name: str, strip_legal_suffix: bool = True) -> str:
    """
    Normalize business name:
    - Unicode normalize & lowercase
    - Replace '&' with 'and'
    - Replace punctuation with space
    - Optionally remove legal entity suffixes (e.g. Pvt Ltd, Inc)
    - Clean whitespace
    """
    if not name or not isinstance(name, str):
        return ""
    
    text = normalize_text(name)
    text = text.replace("&", " and ")
    
    # Replace punctuation characters with space, keeping alphanumeric and basic word chars
    text = re.sub(r"[^\w\s]", " ", text)
    
    if strip_legal_suffix:
        text = LEGAL_SUFFIXES_REGEX.sub(" ", text)
        
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_address(address: str) -> str:
    """
    Normalize address:
    - Unicode normalize & lowercase
    - Replace common abbreviations (rd -> road, st -> street, etc.)
    - Clean punctuation while preserving alphanumeric tokens (including street numbers and postal codes)
    - Collapse whitespace
    """
    if not address or not isinstance(address, str):
        return ""
    
    text = normalize_text(address)
    
    # Expand abbreviations
    for pattern, repl in ADDRESS_ABBR_COMPILED:
        text = pattern.sub(repl, text)
        
    # Remove punctuation except alphanumeric and space
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_country(country: str) -> str:
    """
    Normalize country label:
    - Uppercase, stripped, open-set string label.
    """
    if not country or not isinstance(country, str):
        return ""
    return country.strip().upper()

def extract_numeric_tokens(text: str) -> set:
    """
    Extract all numeric tokens from text (e.g. building numbers, postal codes, unit numbers).
    Useful for address verification and distinguishing '14 MG Road' from '41 MG Road'.
    """
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))
