import json

from src.audit import profile_tsv


def test_profile_tsv_reports_exact_rows_and_missing_values(tmp_path):
    data = tmp_path / "source.tsv"
    data.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tCafe\t1 Main St\tUS\n"
        "S1-2\tCafe\t\tUS\n",
        encoding="utf-8",
    )
    report = profile_tsv(data)
    assert report["rows"] == 2
    assert report["missing_values"] == {"business_address": 1}
    assert report["country_distribution"] == {"US": 2}
    assert report["columns"] == ["entity_id", "business_name", "business_address", "country"]
    json.dumps(report)
