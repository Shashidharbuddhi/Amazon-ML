from io import StringIO

from src.candidate_generation import write_candidate_pairs
from src.blocking import CandidatePair


def test_candidate_writer_emits_official_schema():
    output = StringIO()
    count = write_candidate_pairs(
        [
            CandidatePair("S1-1", "S2-1", "S2"),
            CandidatePair("S1-1", "S3-1", "S3"),
        ],
        output,
        s1_order=["S1-1", "S1-2"],
    )
    assert count == 2
    assert output.getvalue() == (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
    )