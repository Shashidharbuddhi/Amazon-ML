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

## Measured training baseline

Measured on the supplied training split on 2026-09-26:

| Metric | Result |
| --- | ---: |
| Candidate pairs | 69,818,635 |
| Average / median / maximum candidates per S1 | 31.64 / 3 / 1,241 |
| Overall candidate recall | 38.67% |
| S2 recall | 39.39% |
| S3 recall | 37.99% |
| S1 records with at least one lost true match | 1,916,318 |
| Theoretical-pair reduction | 99.999693% |
| Runtime / peak resident memory | 9m 37s / 4.51 GB |

The very high reduction ratio is not sufficient: more than 61% of labelled
matches are lost. Also, 69.8M candidates exceed a comfortable downstream
feature-computation budget. The next rules must improve recall while controlling
large posting lists; they will be evaluated with the same training-only method.
