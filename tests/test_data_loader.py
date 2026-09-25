from pathlib import Path

import pytest

from src.data_loader import discover_split, iter_entities, iter_ground_truth


def write_tsv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_entity_loader_preserves_strings_and_missing_values(tmp_path):
    path = tmp_path / "entities.tsv"
    write_tsv(
        path,
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\t  Café  \t\tFrance\n",
    )
    records = list(iter_entities(path, "S2"))
    assert records[0].entity_id == "S2-001"
    assert records[0].business_name == "  Café  "
    assert records[0].business_address == ""
    assert records[0].country == "France"


def test_entity_loader_rejects_mixed_sources(tmp_path):
    path = tmp_path / "entities.tsv"
    write_tsv(
        path,
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tOne\tAddress\tUS\nS2-2\tTwo\tAddress\tUS\n",
    )
    with pytest.raises(ValueError, match="Mixed entity sources"):
        list(iter_entities(path))


def test_ground_truth_handles_no_match_and_multiple_matches(tmp_path):
    path = tmp_path / "labels.tsv"
    write_tsv(
        path,
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-2,S3-3\nS1-2\t\n",
    )
    assert list(iter_ground_truth(path))[0].matched_entity_ids == ("S2-2", "S3-3")
    assert list(iter_ground_truth(path))[1].matched_entity_ids == ()


def test_discover_split_uses_schema_and_entity_prefixes(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    entity_header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for source in ("S1", "S2", "S3"):
        write_tsv(train / f"arbitrary_{source}.tsv", entity_header + f"{source}-1\tName\tAddress\tUS\n")
    write_tsv(train / "labels.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n")
    paths = discover_split(tmp_path, "train")
    assert paths.source1.name == "arbitrary_S1.tsv"
    assert paths.ground_truth and paths.ground_truth.name == "labels.tsv"
