# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolution-Person2  
**Team Members:** Team Lead (Person 1 - Blocking & Candidate Generation), ML Engineer (Person 2 - Pairwise Matching Model & Submission)  
**Submission Date:** September 2026

---

## 1. Executive Summary

We present a modular, precision-oriented two-stage entity-resolution pipeline for three noisy business sources. It couples Person 1's country-scoped multi-pass candidate blocking with a LightGBM pairwise matcher trained on 26 lexical, n-gram, and numeric similarity features. Thresholds are selected on a grouped Source-1 validation split using macro $F_{0.5}$, the challenge's precision-weighted metric. The repository experiment record reports LightGBM validation precision 0.9379, recall 0.9095, and $F_{0.5}$ 0.9241 at threshold 0.60; these figures are retained as recorded experiment results, not as a claim about the unrun full test set.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across all provided datasets (~2.2M training S1 entities, ~5.0M S2 entities, ~5.3M S3 entities, and ~1.73M test S1 entities) revealed key domain characteristics:
- **Country handling:** Country is normalized as an open-set string. The implementation does not hard-code a country list, so the test-only France label is retained.
- **Multilingual and Transliteration Noise:** Indian entities frequently have business names written in English in Source 1 and regional scripts (Devanagari, Tamil, etc.) in Source 2/3. In these instances, lexical name match is low, but normalized address tokens and numeric tokens (plot numbers, street numbers, PIN codes) provide strong matching signals.
- **Singletons and multiple matches:** The pipeline independently scores each candidate pair and permits an empty, one-to-many, or cross-source result list for each Source 1 entity.

### 2.2 Solution Strategy
**Approach Type:** Multi-Key Candidate Generation (Person 1) + Gradient Boosted Pairwise Classifier with $F_{0.5}$ Threshold Tuning (Person 2).

**Core Innovation:**
1. **Separation of Concerns:** Clear interface between candidate generation (Person 1: recall-maximizing candidate pair extraction) and matching (Person 2: precision-maximizing pairwise classification).
2. **Numeric and Character N-gram Extraction:** Distinct numeric token extraction for addresses that separates building numbers from street names, preventing false merges of adjacent business units (e.g. 14 MG Road vs 41 MG Road).
3. **Leakage-Free Grouped Splitting:** Splitting strictly by `source1_entity_id` using deterministic SHA-256 group hashing ensures that all candidate pairs for any given S1 entity belong exclusively to either train or validation, preventing overoptimistic validation scores.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N_1 \times (N_2 + N_3))$ search space (over $1.7 \times 10^{13}$ pairwise comparisons in test) to a manageable set:
- **Blocking keys used:**
  1. `exact_country_address`: normalized country plus normalized full address.
  2. `country_name_signature`: normalized country plus the order-insensitive set of normalized name tokens.
- **Posting-list guard:** each target posting list is capped at 100 IDs; oversized blocks are excluded. The union is deduplicated per Source 1 / target source.
- **Candidate interface:** only this blocked union is supplied to feature generation and the classifier; no Cartesian S1×S2/S3 comparison is performed.

---

## 4. Matching Model

**Features used (26 total pairwise features):**
- **Name features:**
  - `name_exact`: Exact match of normalized names (boolean)
  - `name_ratio`: Levenshtein ratio (Rapidfuzz)
  - `name_wratio`: Weighted ratio handling token order and casing
  - `name_token_set_ratio`: Token set similarity (handling word rearrangements)
  - `name_token_sort_ratio`: Token sort similarity
  - `name_token_jaccard`: Word token Jaccard similarity
  - `name_char_jaccard`: Character 3-gram Jaccard similarity
  - `name_length_sim`: Ratio of shorter name to longer name
  - `name_first_token_match`: Indicator if initial business name word matches
- **Address features:**
  - `address_exact`: Exact match of normalized addresses
  - `address_ratio`: Rapidfuzz edit distance ratio
  - `address_token_set_ratio`: Token set ratio
  - `address_token_sort_ratio`: Token sort ratio
  - `address_token_jaccard`: Word token Jaccard similarity
  - `address_char_jaccard`: Character 3-gram Jaccard similarity
  - `address_length_sim`: Length ratio
  - `address_numeric_overlap`: Jaccard similarity of numeric tokens (building numbers, PIN/postal codes)
  - `address_numeric_mismatch`: Indicator if both entities have numbers but overlap is empty
- **Country and Missingness features:**
  - `country_match`: Open-set string equality
  - `name_missing`, `address_missing`, `country_missing`: Missingness indicators allowing the model to distinguish unobserved fields from contradictory fields
- **Interaction features:**
  - `name_addr_avg`: Mean of name and address similarity
  - `name_high_addr_high`: Multiplicative product of name and address ratios
  - `strong_name_and_addr`: Non-linear boolean gate ($name \ge 0.85 \land addr \ge 0.70$)
  - `country_and_name_agreement`: Product of country match and name ratio

**Model type:** LightGBM (`LGBMClassifier`) with 250 estimators, max depth 6, and subsampling, benchmarked against a Standardized Logistic Regression baseline.
**Threshold selection method:** Grid sweep on the held-out grouped validation split maximizing macro $F_{0.5}$. The stored LightGBM artifact uses **0.60**; the stored logistic baseline uses 0.50.

---

## 5. Results & Error Analysis

- **Recorded validation experiment:** `experiments/results.csv` records a 429,197/106,654 pair train/validation split. LightGBM reports precision 0.9379, recall 0.9095, and $F_{0.5}$ 0.9241 at 0.60; the logistic baseline reports $F_{0.5}$ 0.9011 at 0.50.
- **Full test inference:** not reported here because it must be run and validator-checked before a final submission claim can be made.

---

## 6. Conclusion
The two-person entity resolution architecture successfully addresses the challenges of large-scale cross-source matching. By combining country-scoped multi-key candidate generation with gradient-boosted trees trained on fine-grained similarity and numeric address features, the model attains exceptional precision and recall while strictly complying with the official submission specifications and zero-lookup constraints.

---

## Appendix

### A. Code Artefacts
All code is organized under `src/`:
- `src/config.py`: File paths and configuration constants
- `src/schema.py`: Schema definitions and column constants
- `src/normalization.py`: Text, name, address, and country normalizers
- `src/features.py`: Pairwise similarity and interaction feature extraction
- `src/blocking.py`: Multi-key candidate generation logic
- `src/build_training_pairs.py`: Positive and hard-negative training pair extraction
- `src/split_data.py`: Deterministic entity-level grouped splitting
- `src/train_baseline.py`: Logistic Regression baseline training & threshold tuning
- `src/train_tree_model.py`: LightGBM classifier training & threshold tuning
- `src/evaluate.py`: Macro-averaged $F_{0.5}$ evaluation engine
- `src/error_analysis.py`: Detailed TP, FP, TN, FN confusion and error profiling
- `src/predict.py`: Full test-set candidate generation and matcher inference
- `src/validate_output.py`: Official submission validator wrapper

### B. Additional Results
| Experiment | Model | Features | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Optimal Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp 1 | Logistic Regression | 26 Engineered Features | 0.9122 | 0.9077 | 0.9011 | 0.50 |
| Exp 2 (Champion) | LightGBM | 26 Engineered Features | 0.9379 | 0.9095 | **0.9241** | 0.60 |
