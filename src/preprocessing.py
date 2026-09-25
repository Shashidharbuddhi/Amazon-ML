"""Deterministic normalization for blocking keys, without altering raw fields."""

from __future__ import annotations

import re
import unicodedata

# These are deliberately conservative: they are removed only as *trailing*
# complete tokens, never from the middle of a name.
LEGAL_SUFFIX_TOKENS = frozenset(
    {
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "llp",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "pvt",
        "private",
        "sa",
        "sas",
        "sarl",
        "gmbh",
        "ag",
        "bv",
        "nv",
    }
)

_WHITESPACE = re.compile(r"\s+")


def _replace_punctuation(value: str) -> str:
    """Use Unicode categories so combining marks in Indic scripts survive."""
    return "".join(
        character
        if character.isspace() or unicodedata.category(character)[0] in {"L", "M", "N"}
        else " "
        for character in value
    )


def normalize_text(value: str | None) -> str:
    """Normalize Unicode, case, punctuation and whitespace into stable tokens.

    This deliberately does not ASCII-fold: removing combining marks can corrupt
    meaningful Indic and other non-Latin text. Unicode word characters remain.
    """
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    # Apostrophes usually occur within a token (for example Orelee's), where a
    # separator would weaken exact-name blocking more than a deletion.
    value = value.replace("'", "").replace("’", "")
    value = _replace_punctuation(value)
    return _WHITESPACE.sub(" ", value).strip()


def normalize_country(value: str | None) -> str:
    """Produce a generic country comparison key without country-specific maps."""
    return normalize_text(value)


def normalize_name(value: str | None, *, remove_legal_suffixes: bool = True) -> str:
    """Normalize a business name and optionally remove trailing legal suffixes."""
    normalized = normalize_text(value)
    if not normalized or not remove_legal_suffixes:
        return normalized
    tokens = normalized.split()
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(value: str | None) -> str:
    """Normalize an address while retaining building numbers and alphanumeric IDs."""
    return normalize_text(value)


def name_tokens(value: str | None) -> tuple[str, ...]:
    """Return normalized, non-empty name tokens for later blocking rules."""
    normalized = normalize_name(value)
    return tuple(normalized.split()) if normalized else ()


def address_tokens(value: str | None) -> tuple[str, ...]:
    """Return normalized, non-empty address tokens for later blocking rules."""
    normalized = normalize_address(value)
    return tuple(normalized.split()) if normalized else ()
