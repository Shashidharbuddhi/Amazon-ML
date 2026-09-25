# Baseline exact-name blocking

The first candidate rule is deliberately simple and measurable:

```
(normalize_country(country), normalize_name(business_name))
```

Source 2 and Source 3 each build a separate inverted index from this key to
their entity IDs. For each source-1 entity, the matching posting list emits
only S1–S2 or S1–S3 candidates. Empty normalized countries or names produce no
key, preventing an accidental massive empty-value block.

Indexes retain only keys and IDs, not full records. They are built and consumed
one target source at a time, then released before indexing the other target
source. Candidate and posting-list order follows the source TSV order, making
the rule reproducible without a global candidate set.

This is a baseline, not the selected final blocker. Phase 4 will measure its
training-ground-truth recall and candidate reduction before adding any rule.
