from __future__ import annotations

import joblib
import numpy as np

from src.predict import run_prediction_pipeline


class AlwaysMatchModel:
    def predict_proba(self, values):
        return np.asarray([[0.0, 1.0] for _ in values])


def _write_entities(path, rows):
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_prediction_streams_sources_and_writes_official_outputs(tmp_path):
    test_dir = tmp_path / "data" / "test"
    test_dir.mkdir(parents=True)
    _write_entities(test_dir / "test_source1.tsv", [
        "S1-1\tAcme Ltd\t1 Main St\tUS",
        "S1-2\tNo Match\t2 Side St\tUS",
    ])
    _write_entities(test_dir / "test_source2.tsv", ["S2-1\tAcme\t1 Main St\tUS"])
    _write_entities(test_dir / "test_source3.tsv", ["S3-1\tAcme\t1 Main St\tUS"])
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": AlwaysMatchModel(), "features": ["country_match"], "threshold": 0.5}, model_path)

    candidates, matches, empty = run_prediction_pipeline(
        data_root=tmp_path / "data", output_dir=tmp_path / "output", model_path=model_path,
        batch_size=1,
    )

    assert (candidates, matches, empty) == (2, 2, 1)
    assert (tmp_path / "output" / "candidate_pairs.tsv").read_text(encoding="utf-8") == (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
    )
    assert (tmp_path / "output" / "matching_results.tsv").read_text(encoding="utf-8") == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
    )
