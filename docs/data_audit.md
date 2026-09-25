# Phase 0 data audit

Audited 2026-09-25 from the supplied TSVs. There are no archives: `data/` is
already expanded and occupies 2.4 GB. All entity files use the schema
`entity_id`, `business_name`, `business_address`, `country`; ground truth uses
`source1_entity_id`, `matched_entity_ids` (a comma-delimited list).

| Split/source | Rows | Countries | Missing address |
| --- | ---: | --- | ---: |
| train source 1 | 2,206,821 | India 883,188; US 1,323,633 | 0 |
| train source 2 | 5,034,616 | India 2,017,799; US 3,016,817 | 168,967 |
| train source 3 | 5,285,603 | India 2,115,547; US 3,170,056 | 175,916 |
| train ground truth | 2,206,821 | n/a | 123,247 empty match lists |
| test source 1 | 1,732,544 | France 259,452; India 809,986; US 663,106 | 0 |
| test source 2 | 4,887,273 | France 703,378; India 2,312,565; US 1,871,330 | 129,408 |
| test source 3 | 5,082,316 | France 731,615; India 2,405,000; US 1,945,701 | 136,098 |

All entity IDs were checked by external sort and have zero duplicates within
their respective file. `source1_entity_id` also has zero duplicates in ground
truth. Exact duplicate business-name/address counts are intentionally deferred
to the streaming audit report because retaining every unique text value would
be needlessly memory-intensive.

Observed text ranges support normalizing both native-script and Latin text:
train source 1 name/address length is 3–105 / 11–256 characters; train source
2 is 2–104 / 20–249; train source 3 is 2–123 / 0–240. France exists only in
test, so country logic must be generic.

## Memory implications

The raw TSVs total 2.4 GB. Loading source tables simultaneously into pandas
would materially amplify their footprint and is inappropriate on an 8 GB host.
`src.audit` processes one row at a time, retaining only country counters,
fixed-size HyperLogLog registers, and scalar length statistics (about 17 MB
maximum resident memory in a measured source-1 run). Candidate generation will
therefore use streamed source reads and compact indexes rather than Cartesian
joins.

## Reproducing the audit

```bash
python -m src.audit --data-root data --output output/data_audit.json
```

This command produces exact row/missing/country/length summaries and bounded-
memory estimates of duplicate non-empty values. It may take several minutes on
the full dataset; no dataset contents are written to Git.
