# Multi-pass blocking selection

The exact country/name baseline recalled only 38.67% of training matches.
Before introducing token retrieval, Phase 5 evaluates two conservative rules
that have bounded posting lists by construction:

1. `country + normalized full address` — captures entities whose complete
   address survives source noise, while ignoring blank addresses.
2. `country + sorted unique normalized full-name tokens` — captures reordered
   names such as `Cafe Bar` / `Bar Cafe` without matching on one generic token.

Neither rule uses labels. Both preserve country separation and generate only
S1–S2/S1–S3 pairs. Their independent and union recall/candidate counts must be
measured before any more permissive rare-token rule is introduced.

## Guarded union implementation

The implementation evaluates the two passes as a deduplicated union. Each
target source is indexed in one scan, and a key is discarded when its posting
list exceeds the configured cap. This prevents a common address or reordered
name signature from creating a large candidate burst while preserving smaller
blocks from the same pass.

Run the training evaluation with the default cap of 100 IDs per key:

```sh
python -m src.evaluation --rule conservative_multipass \
   --output output/conservative_multipass_evaluation.json
```

Use `--max-posting-list-size` to measure another cap. Compare the resulting
recall and candidate volume with the baseline report before selecting a final
blocker.

The generated file has exactly two tab-separated columns:
`source1_entity_id` and `candidate_entity_id`. Candidate metadata such as the
blocking rule is retained only in the in-process `CandidatePair` objects and is
not written to the downstream file.
