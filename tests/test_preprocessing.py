from src.preprocessing import address_tokens, name_tokens, normalize_address, normalize_country, normalize_name


def test_name_normalizes_case_punctuation_and_legal_suffixes():
    assert normalize_name("ABC Pvt. Ltd.") == "abc"
    assert normalize_name("ABC Private Limited") == "abc"
    assert normalize_name("ABC PVT. LTD.") == "abc"


def test_name_does_not_remove_meaningful_internal_legal_word():
    assert normalize_name("The Company Store") == "the company store"
    assert normalize_name("Limited Editions") == "limited editions"


def test_unicode_and_ampersand_are_deterministic():
    assert normalize_name("Ｃafé & Bar, Inc.") == "café and bar"
    assert normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_address_retains_numbers_and_alphanumeric_units():
    assert normalize_address("No. 10, Enkay Square #448A / Phase-V") == "no 10 enkay square 448a phase v"
    assert address_tokens("  12-B Main St. ") == ("12", "b", "main", "st")


def test_empty_values_and_country_normalization():
    assert normalize_name(None) == ""
    assert normalize_address("") == ""
    assert normalize_country(" France ") == "france"
    assert name_tokens("Orelee's Barbershop") == ("orelees", "barbershop")
