from io import StringIO

from src.candidate_generation import write_candidate_pairs
from src.data_loader import EntityRecord
from src.blocking import CandidatePair


def test_candidate_writer_emits_exact_two_column_tsv():
    output = StringIO()
    count = write_candidate_pairs(
        iter(
            [
                CandidatePair("S1-1", "S2-1", "S2"),
                CandidatePair("S1-1", "S3-1", "S3"),
            ]
        ),
        output,
    )
    assert count == 2
    assert output.getvalue() == (
        "source1_entity_id\tcandidate_entity_id\n"
        "S1-1\tS2-1\n"
        "S1-1\tS3-1\n"
    )