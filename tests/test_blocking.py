from src.blocking import (
    build_inverted_index,
    build_exact_name_index,
    country_name_signature_key,
    exact_country_address_key,
    exact_country_name_key,
    feature_normalized_name_key,
    iter_exact_name_candidates,
    iter_keyed_candidates,
    iter_keyed_candidates_for_targets,
    iter_exact_name_candidates_for_targets,
    iter_multipass_candidates_for_targets,
)
from src.data_loader import EntityRecord


def entity(entity_id, name, address, country):
    return EntityRecord(entity_id, name, address, country, entity_id.split("-", 1)[0])


def test_exact_key_uses_normalized_country_and_name():
    assert exact_country_name_key(entity("S1-1", "ABC Pvt. Ltd.", "x", " France ")) == ("france", "abc")
    assert exact_country_name_key(entity("S1-2", "", "x", "France")) is None


def test_address_and_name_signature_keys_are_conservative():
    assert exact_country_address_key(entity("S1-1", "x", "No. 10 / 448A", "France")) == ("france", "no 10 448a")
    assert exact_country_address_key(entity("S1-2", "x", "", "France")) is None
    assert country_name_signature_key(entity("S1-3", "Bar & Café", "x", "France")) == ("france", ("and", "bar", "café"))


def test_feature_normalized_name_key_uses_feature_normalization_and_country():
    assert feature_normalized_name_key(
        entity("S2-1", "ABC Pvt. Ltd.", "x", " France ")
    ) == ("france", "abc")
    assert feature_normalized_name_key(entity("S2-2", "ABC", "x", "Germany")) != (
        "france", "abc"
    )


def test_signature_blocking_matches_reordered_names():
    source1 = [entity("S1-1", "Bar Cafe", "x", "France")]
    targets = [entity("S2-1", "Cafe Bar", "x", "France")]
    index = build_inverted_index(targets, country_name_signature_key)
    pairs = list(iter_keyed_candidates(source1, index, "S2", country_name_signature_key, "name_signature"))
    assert [(pair.candidate_entity_id, pair.rule) for pair in pairs] == [("S2-1", "name_signature")]


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


def test_keyed_targets_preserve_rule_name():
    record_s1 = entity("S1-1", "Cafe Bar", "x", "US")
    pairs = list(
        iter_keyed_candidates_for_targets(
            lambda: iter([record_s1]),
            [entity("S2-1", "Bar Cafe", "x", "US")],
            [],
            country_name_signature_key,
            "name_signature",
        )
    )
    assert [pair.rule for pair in pairs] == ["name_signature"]


def test_multipass_union_deduplicates_overlapping_passes():
    source1 = [entity("S1-1", "Cafe Bar", "10 Main Street", "US")]
    source2 = [entity("S2-1", "Bar Cafe", "10 Main Street", "US")]
    pairs = list(
        iter_multipass_candidates_for_targets(
            lambda: iter(source1),
            source2,
            [],
            (("address", exact_country_address_key), ("name", country_name_signature_key)),
            max_posting_list_size=10,
        )
    )
    assert [(pair.candidate_entity_id, pair.rule) for pair in pairs] == [("S2-1", "address")]


def test_multipass_skips_oversized_posting_lists():
    source1 = [entity("S1-1", "Cafe", "10 Main Street", "US")]
    source2 = [
        entity("S2-1", "Cafe", "10 Main Street", "US"),
        entity("S2-2", "Cafe", "10 Main Street", "US"),
    ]
    pairs = list(
        iter_multipass_candidates_for_targets(
            lambda: iter(source1),
            source2,
            [],
            (("address", exact_country_address_key),),
            max_posting_list_size=1,
        )
    )
    assert pairs == []


def test_multipass_union_includes_feature_normalized_name_and_deduplicates():
    source1 = [entity("S1-1", "ABC Private Limited", "other", "US")]
    source2 = [entity("S2-1", "ABC Pvt. Ltd.", "other", "US")]
    pairs = list(
        iter_multipass_candidates_for_targets(
            lambda: iter(source1),
            source2,
            [],
            (
                ("address", exact_country_address_key),
                ("signature", country_name_signature_key),
                ("feature_name", feature_normalized_name_key),
            ),
            max_posting_list_size=100,
        )
    )
    assert [(pair.candidate_entity_id, pair.candidate_source) for pair in pairs] == [("S2-1", "S2")]
