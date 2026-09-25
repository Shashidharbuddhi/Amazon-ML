# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolution-Person2  
**Team Members:** Team Lead (Person 1 - Blocking & Candidate Generation), ML Engineer (Person 2 - Pairwise Matching Model & Submission)  
**Submission Date:** September 2026

---

## 1. Executive Summary

We present a modular, high-precision two-stage entity resolution pipeline designed for cross-source enterprise identity linkage across three noisy sources (Source 1 reference, Source 2, and Source 3). The pipeline couples multi-key country-scoped candidate blocking with a gradient-boosted pairwise matching model (LightGBM) trained on 26 engineered phonetic, lexical, n-gram, and numeric similarity features. Because the official challenge metric is macro-averaged $F_{0.5}$ (weighting precision 2× over recall to heavily penalize erroneous company merges), we optimize decision thresholds directly on validation $F_{0.5}$ using a leakage-free, group-split methodology. Our LightGBM matcher achieves a validation macro $F_{0.5}$ score of **0.9907** (macro precision 0.9906, macro recall 0.9959), significantly outperforming the Logistic Regression baseline (0.9865).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across all provided datasets (~2.2M training S1 entities, ~5.0M S2 entities, ~5.3M S3 entities, and ~1.73M test S1 entities) revealed key domain characteristics:
- **Strict Geographic Partitioning:** Ground-truth analysis over 50,000 positive pairs demonstrated **0% cross-country matching**. All true matches occur within the same country label.
- **Open-Set Country Distribution:** The training set contains `US` and `India`, whereas the test set additionally introduces `France` (259,452 S1 records, 703,378 S2 records, 731,615 S3 records). Country matching was modeled as generic string equality rather than fixed categorical encoding.
- **Multilingual and Transliteration Noise:** Indian entities frequently have business names written in English in Source 1 and regional scripts (Devanagari, Tamil, etc.) in Source 2/3. In these instances, lexical name match is low, but normalized address tokens and numeric tokens (plot numbers, street numbers, PIN codes) provide strong matching signals.
- **Singleton Entities:** Training ground truth revealed that 5.58% of Source 1 entities have zero matching records (singletons). For these entities, any false positive prediction drops the entity's macro $F_{0.5}$ score from 1.0 to 0.0, highlighting the necessity of conservative, precision-focused decision thresholds.
- **Non-singleton Entity Cardinality:** Non-singleton S1 entities match an average of 3.67 records across Source 2 and Source 3, with match counts ranging up to 11. Therefore, our model independently scores each candidate pair rather than enforcing rigid 1-to-1 or top-1 constraints.

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
  1. `name_exact`: Country + full normalized business name (with legal suffixes removed)
  2. `name_2tok`: Country + first two significant tokens of the business name
  3. `name_1tok`: Country + first token (length $\ge 4$)
  4. `tok_num`: Country + first name token + first address numeric token
  5. `addr_exact`: Country + full normalized address (capturing transliterated company names sharing identical premises)
- **Candidate pairs generated:** Evaluated up to top-30 candidates per Source 1 entity, ensuring high candidate recall while keeping inference memory under 4 GB.
- **How true matches were preserved:** Inverted index lookups across multiple complementary keys ensure that if a name is misspelled or abbreviated, the address or token-prefix key still captures the candidate.

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
**Threshold selection method:** Grid sweep over candidate thresholds $[0.30, 0.95]$ on the held-out validation set maximizing the macro-averaged $F_{0.5}$ metric. Optimal threshold identified: **0.60** for LightGBM (0.55 for Logistic Regression).

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (macro validation):**
  - **Baseline (Logistic Regression):** Macro $F_{0.5} = \mathbf{0.9865}$ (Precision: 0.9874, Recall: 0.9909)
  - **Champion (LightGBM):** Macro $F_{0.5} = \mathbf{0.9907}$ (Precision: 0.9906, Recall: 0.9959)
  - Micro $F_{0.5}$: **0.9917** on 42,483 validation pairs.
- **Top Feature Importances:**
  1. `address_char_jaccard` (802 splits)
  2. `address_length_sim` (749 splits)
  3. `name_token_jaccard` (717 splits)
  4. `address_token_set_ratio` (688 splits)
  5. `address_token_jaccard` (552 splits)
  6. `name_char_jaccard` (436 splits)
  7. `address_numeric_overlap` (357 splits)
- **Common false positives (wrong merges):** Co-located businesses in the same commercial complex (high address similarity) or distinct franchises/branches with identical brand names but differing unit numbers. Increasing the threshold to $\ge 0.60$ effectively filters these ambiguous pairs.
- **Common false negatives (missed matches):** Records where the address is missing (`address_missing = 1`) and the business name is transliterated into a regional non-Latin script without English phonetic overlap.

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
| Exp 1 | Logistic Regression | 26 Engineered Features | 0.9874 | 0.9909 | 0.9865 | 0.55 |
| Exp 2 (Champion) | LightGBM | 26 Engineered Features | 0.9906 | 0.9959 | **0.9907** | 0.60 |
