from src.blocking import (
    build_exact_name_index,
    exact_country_name_key,
    iter_exact_name_candidates,
    iter_exact_name_candidates_for_targets,
)
from src.data_loader import EntityRecord


def entity(entity_id, name, address, country):
    return EntityRecord(entity_id, name, address, country, entity_id.split("-", 1)[0])


def test_exact_key_uses_normalized_country_and_name():
    assert exact_country_name_key(entity("S1-1", "ABC Pvt. Ltd.", "x", " France ")) == ("france", "abc")
    assert exact_country_name_key(entity("S1-2", "", "x", "France")) is None


def test_exact_blocker_emits_only_matching_s1_s2_pairs_in_order():
    source1 = [
        entity("S1-1", "ABC Private Limited", "one", "France"),
        entity("S1-2", "No Match", "two", "France"),
    ]
    source2 = [
        entity("S2-9", "ABC PVT. LTD.", "other", "France"),
        entity("S2-8", "ABC", "other", "France"),
    ]
    index = build_exact_name_index(source2)
    pairs = list(iter_exact_name_candidates(source1, index, "S2"))
    assert [(pair.source1_entity_id, pair.candidate_entity_id) for pair in pairs] == [
        ("S1-1", "S2-9"),
        ("S1-1", "S2-8"),
    ]
    assert all(pair.candidate_source == "S2" for pair in pairs)


def test_targets_are_processed_separately_without_duplicate_pairs():
    source1 = [entity("S1-1", "Cafe", "x", "US")]
    source2 = [entity("S2-1", "Cafe", "x", "US")]
    source3 = [entity("S3-1", "Cafe", "x", "US")]
    pairs = list(iter_exact_name_candidates_for_targets(lambda: iter(source1), source2, source3))
    assert [(pair.source1_entity_id, pair.candidate_entity_id) for pair in pairs] == [
        ("S1-1", "S2-1"),
        ("S1-1", "S3-1"),
    ]


def test_repeated_target_entity_does_not_duplicate_a_candidate_pair():
    targets = [entity("S2-1", "Cafe", "x", "US"), entity("S2-1", "Cafe", "x", "US")]
    pairs = list(iter_exact_name_candidates([entity("S1-1", "Cafe", "x", "US")], build_exact_name_index(targets), "S2"))
    assert len(pairs) == 1
