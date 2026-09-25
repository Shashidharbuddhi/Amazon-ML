from src.blocking import CandidatePair
from src.data_loader import GroundTruthRecord
from src.evaluation import evaluate_candidates


def test_evaluation_counts_multiple_matches_and_no_match_entities():
    truth = [
        GroundTruthRecord("S1-1", ("S2-1", "S3-1")),
        GroundTruthRecord("S1-2", ()),
        GroundTruthRecord("S1-3", ("S2-2",)),
    ]
    candidates = [
        CandidatePair("S1-1", "S2-1", "S2"),
        CandidatePair("S1-1", "S2-1", "S2"),  # must not inflate recall
        CandidatePair("S1-1", "S3-9", "S3"),
    ]
    result = evaluate_candidates(candidates, truth, source1_entities=3, source2_entities=2, source3_entities=2)
    assert result.candidate_pairs == 3
    assert result.true_matches == 3
    assert result.true_matches_found == 1
    assert result.candidate_recall == 0.33333333
    assert result.source2_recall == 1 / 2
    assert result.source3_recall == 0.0
    assert result.source1_with_true_matches_lost == 2
    assert result.median_candidates_per_source1 == 1
    assert result.reduction_ratio == 0.75
