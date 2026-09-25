"""
Schema definitions and constants for Amazon ML Challenge 2026.
"""

DELIM = "\t"

# Source file columns
ENTITY_ID_COL = "entity_id"
BUSINESS_NAME_COL = "business_name"
BUSINESS_ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

SOURCE_COLUMNS = [ENTITY_ID_COL, BUSINESS_NAME_COL, BUSINESS_ADDRESS_COL, COUNTRY_COL]

# Ground truth columns
GT_SOURCE1_COL = "source1_entity_id"
GT_MATCHED_COL = "matched_entity_ids"
GT_COLUMNS = [GT_SOURCE1_COL, GT_MATCHED_COL]

# Candidate file columns (for Person 1 output / blocking)
CAND_SOURCE1_COL = "source1_entity_id"
CAND_MATCHED_COL = "candidate_entity_ids"
CANDIDATE_COLUMNS = [CAND_SOURCE1_COL, CAND_MATCHED_COL]

# Submission file columns (Person 2 final output)
SUBMISSION_SOURCE1_COL = "source1_entity_id"
SUBMISSION_MATCHED_COL = "matched_entity_ids"
SUBMISSION_COLUMNS = [SUBMISSION_SOURCE1_COL, SUBMISSION_MATCHED_COL]
