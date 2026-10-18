# FREEZE PROTOCOL — Harness Engineering for Agentic AutoML

> **Study**: ICSL-PostDoc-2026-010
> **Principal Investigator**: Suresh Arumugam
> **Supervisor**: Dr. Kiat Siong Ng
> **Affiliation**: National Yunlin University of Science and Technology (NYUST YunTech), Taiwan
> **Target Journal**: Expert Systems with Applications (Elsevier, SCIE Q1)
> **Protocol Version**: 2.1
> **Freeze Date**: 15 September 2026
> **Status**: FROZEN — no parameter, definition, or threshold may change after this date without a documented protocol amendment

---

## 1. Purpose

This document locks every definition, detector specification, hyperparameter, threshold, and evaluation procedure produced during checkpoints CP-3R through CP-5R. Nothing listed here may be modified, tuned, or re-derived during CP-6 (external validation) or paper writing. Any post-freeze change requires a numbered protocol amendment appended to §12.

The freeze separates the *development corpus* (16 datasets, o4-mini, checkpoints CP-3R→CP-5R) from the *validation corpus* (CP-6: 6–8 new datasets, stronger model), ensuring that CP-6 results are a genuine external test with no risk of data dredging.

---

## 2. Frozen Fault Taxonomy

Three fault classes, one naming convention.

| ID | Name | Abbreviation | Definition | Source |
|----|------|--------------|------------|--------|
| F1 | Redundant Configuration Proposal | RCP | The AIDE agent re-proposes a configuration whose **canonical hash v2** matches a previously executed step within the same dataset run. Order-preserving hash over: model class, sorted hyperparameters, ordered preprocessing chain with per-step params, split method + params, feature selector. | Natural (64 instances) |
| F2 | Data Leakage | DL | A preprocessing step (fit/transform) is applied to the full dataset before the train-test split, leaking test information into training. Detected via AST analysis of the generated code (`leaky_before_split` derived from code, not injection metadata). | Natural (15) + Synthetic (25) = 40 |
| F3 | Training-Budget Fault | TBF | Model training is executed with degenerate hyperparameters that produce a valid-looking fitted object but prevent meaningful learning. Three mutation strategies: `clamp_n_est` (cap estimators at 1–2), `clamp_lr` (set learning rate to 1e-8), `clamp_depth` (set max_depth to 1). | Synthetic only (40) |

**Fault overlap**: A step may carry multiple fault labels (e.g., an RCP step that also has leakage). The `any_fault` label is True if any of F1, F2, or F3 is True. Overlap-aware counting is used throughout; detector recall is always against `any_fault` for combined metrics.

**F1/RCP tautology disclosure**: D1 detects F1 using `hash_seen_before`, which *is* the definition of F1. D1's perfect P/R/F1 = 1.0 is reported alongside independent construct evidence (score-delta analysis showing RCP steps are empirically wasteful), never as independent detection capability.

---

## 3. Frozen Trace Corpus

**Source**: AIDE (MIT, open-source AutoML agent scaffold) running o4-mini on OpenML CC-18 tabular datasets.

**Configuration**: 20 steps per dataset, single seed per dataset, response cache keyed by prompt hash for deterministic replay.

| Metric | Value |
|--------|-------|
| Total steps | 208 |
| Datasets | 16 |
| Faulty steps (`any_fault`) | 121 |
| Clean steps | 87 |
| F1 (RCP) instances | 64 (all natural) |
| F2 (DL) instances | 40 (15 natural + 25 synthetic) |
| F3 (TBF) instances | 40 (all synthetic) |
| F3 strategy: `clamp_n_est` | 14 |
| F3 strategy: `clamp_lr` | 12 |
| F3 strategy: `clamp_depth` | 14 |
| Base prevalence (`any_fault`) | 121/208 = 58.17% |
| Oracle cost ceiling | 55.69% of total wall-clock |

### 3.1 Per-Dataset Step Counts

| Dataset | Steps | F1 | F2 | F3 | Fault | Clean |
|---------|-------|----|----|----|-------|-------|
| Satellite | 17 | 8 | 2 | 3 | 11 | 6 |
| amazon-commerce-reviews | 4 | 1 | 4 | 1 | 4 | 0 |
| christine | 12 | 2 | 1 | 3 | 6 | 6 |
| credit-approval | 13 | 4 | 3 | 3 | 10 | 3 |
| credit-g | 17 | 6 | 3 | 3 | 10 | 7 |
| diabetes | 15 | 5 | 3 | 3 | 10 | 5 |
| jasmine | 13 | 7 | 3 | 4 | 10 | 3 |
| kr-vs-kp | 19 | 5 | 2 | 2 | 9 | 10 |
| letter | 10 | 1 | 0 | 0 | 1 | 9 |
| mfeat-factors | 10 | 3 | 1 | 3 | 6 | 4 |
| mfeat-fourier | 14 | 4 | 10 | 2 | 11 | 3 |
| mfeat-morphological | 8 | 0 | 2 | 3 | 5 | 3 |
| mfeat-zernike | 14 | 2 | 1 | 0 | 2 | 12 |
| pendigits | 14 | 3 | 0 | 3 | 6 | 8 |
| tic-tac-toe | 16 | 10 | 2 | 3 | 11 | 5 |
| vehicle | 12 | 3 | 3 | 4 | 8 | 4 |

**Note**: `letter` has only 1 fault (F1) and no F2/F3. `mfeat-zernike` has no F3. LODO folds holding out these datasets train without their fault-class examples — this is by design and reflected in per-fold metrics.

---

## 4. Frozen Feature Specification

### 4.1 Tier 1 — Pre-Execution Features (7)

Available before a trial runs. Used by gate detectors (D_any_gate).

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `n_hyperparams` | int | Number of hyperparameters in the step configuration |
| 2 | `n_preprocessing_steps` | int | Length of the preprocessing chain |
| 3 | `preprocessing_before_split` | bool | Whether any preprocessing is fitted before the train-test split |
| 4 | `leaky_before_split` | bool | AST-derived: fit/transform applied before split (code analysis, not injection metadata) |
| 5 | `hash_seen_before` | bool | Whether the step's canonical hash v2 matches any earlier step in the same run |
| 6 | `hash_first_seen_at` | int | Step index where this canonical hash first appeared (0-indexed) |
| 7 | `parse_quality_full_ast` | bool | Whether the step's code was fully parsed via AST (True) vs. partial/failed |

### 4.2 Tier 2 — Post-Execution Features (9)

Available only after a trial has executed. Used by monitor detectors (D_any_monitor).

**Base features (6):**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 8 | `validation_score` | float | Validation metric returned by the step |
| 9 | `wall_clock_sec` | float | Execution time in seconds |
| 10 | `score_delta_from_prev` | float | Difference from the previous step's validation score |
| 11 | `score_rank_in_run` | int | Rank of this step's score within its dataset run (1 = best) |
| 12 | `is_best_so_far` | bool | Whether this step achieves the best score seen so far in the run |
| 13 | `steps_since_improvement` | int | Number of steps since the last score improvement |

**Derived features (3):**

| # | Feature | Type | Derivation |
|---|---------|------|------------|
| 14 | `score_rank_pct` | float | `score_rank_in_run / n_steps_in_run` |
| 15 | `wall_clock_rank_pct` | float | Rank of wall_clock_sec within the run, as a percentile |
| 16 | `within_run_z` | float | `(validation_score − run_mean) / run_std` (z-score within the dataset run) |

**Total**: 7 (Tier 1) + 9 (Tier 2) = **16 features**

---

## 5. Frozen Detector Specifications

### 5.1 Overview

| Detector | Tier | Target | Method | Features Used |
|----------|------|--------|--------|---------------|
| D1 | Gate | F1 (RCP) | Rule-based: `hash_seen_before == True` | 1 (Tier 1) |
| D2_gate | Gate | F2 (DL) | Rule-based: `leaky_before_split == True` | 1 (Tier 1) |
| D2_monitor | Monitor | F2 (DL) | `leaky_before_split OR (within_run_z > θ)` | 2 (Tier 1 + Tier 2) |
| D3_gate | Gate | F3 (TBF) | LightGBM on Tier 1 features | 7 (Tier 1) |
| D3_monitor | Monitor | F3 (TBF) | LightGBM on all features | 16 (all) |
| D_any_gate | Gate | any_fault | LightGBM on Tier 1 + meta-features | 9 (7 Tier 1 + 2 meta) |
| D_any_monitor | Monitor | any_fault | LightGBM on all + meta-features | 19 (16 all + 3 meta) |

### 5.2 Meta-Features for D_any

**D_any_gate** adds 2 meta-features to Tier 1:
- `d1_prediction` = D1 output (`hash_seen_before`)
- `d2_gate_prediction` = D2_gate output (`leaky_before_split`)

**D_any_monitor** adds 3 meta-features to all:
- `d1_prediction` = D1 output
- `d2_gate_prediction` = D2_gate output
- `d2_monitor_prediction` = D2_monitor combined output (structural + statistical)

### 5.3 Frozen LightGBM Hyperparameters

```
n_estimators      = 100
max_depth         = 4
learning_rate     = 0.1
min_child_samples = 5
class_weight      = "balanced"
random_state      = 42
verbose           = -1
n_jobs            = 1
```

These parameters are fixed. No tuning, no grid search, no adjustment in CP-6.

### 5.4 D2_monitor Threshold Tuning Procedure

The `within_run_z` threshold θ for D2_monitor is tuned per LODO fold:
- On each training fold, sweep `SCORE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]`
- Select θ that maximises F1 for F2 detection on the training fold
- Apply the selected θ to the held-out fold
- No information from the held-out fold influences θ

### 5.5 Frozen CP-4R Results (LODO Aggregate)

| Detector | Tier | Target | Precision | Recall | F1 | AUROC |
|----------|------|--------|-----------|--------|------|-------|
| D1 | Gate | F1 | 1.0000 | 1.0000 | 1.0000 | — |
| D2_gate | Gate | F2 | — | — | 0.8163 | — |
| D2_monitor | Monitor | F2 | — | — | 0.7767 | — |
| D3_gate | Gate | F3 | — | — | 0.6139 | — |
| D3_monitor | Monitor | F3 | — | — | 0.5714 | — |
| D_any_gate | Gate | any_fault | — | — | 0.9106 | — |
| D_any_monitor | Monitor | any_fault | — | — | 0.9580 | — |

*Exact precision/recall values are in `cp4_results/cp4r_evaluation.json`. F1 values above are the frozen reference.*

**D1 note**: P/R/F1 = 1.0 is by construction (tautological). Independent empirical evidence from construct analysis.

**D3 note**: D3_gate's modest F1 = 0.6139 is a legitimate finding — Tier 1 pre-execution features contain limited signal for training-budget faults. D3_monitor (F1 = 0.5714) also underperforms relative to D_any, indicating TBF detection benefits from combined fault awareness. These numbers were NOT optimised per supervisor's instruction.

---

## 6. Frozen Conformal Risk Control Thresholds (CP-5R)

### 6.1 Method

Conformal Risk Control (CRC) per Angelopoulos et al. (ICLR 2024). Applied independently to each tier:
- **λ_gate**: threshold on D_any_gate OOF probabilities; steps above this threshold are skipped pre-execution
- **λ_monitor**: threshold on D_any_monitor OOF probabilities; steps above this threshold are flagged post-execution

**False-Skip Rate (FSR)** = proportion of actually-faulty steps whose probability falls below the CRC threshold (i.e., faults missed). The guarantee: FSR ≤ α at each specified significance level.

**Calibration input**: OOF predictions from CP-4R's LODO evaluation — genuinely held-out predictions where each step's probability was produced by a model that never trained on that step's dataset.

### 6.2 Key Alpha Levels

```
KEY_ALPHAS  = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
FINE_ALPHAS = np.arange(0.01, 0.51, 0.01)  # 50 values
```

### 6.3 Frozen Results at α = 0.10

| Metric | Gate (Tier 1) | Monitor (Tier 2) |
|--------|---------------|-------------------|
| CRC threshold (λ) | 0.324949 | 0.951217 |
| CRC FSR | 0.0909 | 0.0909 |
| CRC recall (1 − FSR) | 0.9091 | 0.9091 |
| CRC F1 | 0.9167 | 0.9524 |
| Default θ = 0.5 FSR | — | 0.0579 |

### 6.4 Tiered Cost Analysis at α = 0.10

| Metric | Value |
|--------|-------|
| **Tier 1 gate — net savings** | **48.46%** |
| Tier 1 gross savings | (TP wall-clock / total wall-clock) |
| Tier 1 FP cost | (FP wall-clock / total wall-clock) |
| Tier 2 additional true positives | 3 |
| Combined recall | 0.9339 |
| Oracle ceiling | 55.69% |

**Tier 1 savings are genuine**: steps flagged by the gate are skipped before execution, so their wall-clock time is never consumed.

**Tier 2 savings are NOT direct**: Tier 2 flags steps after they have executed. The value is in (a) additional fault detections beyond Tier 1, and (b) sequential replay avoidance — the agent is informed that the step was faulty and should not build on its result.

### 6.5 Legitimate Findings (Not Failures)

Two CP-5R acceptance criteria did not pass. Both are documented as legitimate findings, not bugs:

1. **Monitor CRC threshold = 0.951217, not < 0.5** (criterion #29).
   D_any_monitor assigns very high probabilities to most faulty steps. The CRC procedure selects the k-th sorted probability where k = floor(α × (n_faulty + 1)) − 1. With FSR = 0.0909, k lands near the top of the distribution, yielding a threshold close to 1.0. This is mathematically correct — it means the monitor is highly confident on most faults and only needs a very aggressive threshold to satisfy the FSR guarantee.

2. **Monitor default θ = 0.5 FSR (0.0579) beats CRC FSR (0.0909)** (criterion #31).
   CRC is conservative by design — it provides a *worst-case* guarantee, not an optimal operating point. The default θ = 0.5 happens to perform well empirically on this corpus, but it carries no distribution-free guarantee. The CRC threshold's value is the formal FSR ≤ α bound, not beating every heuristic.

---

## 7. Frozen Evaluation Protocol

### 7.1 Cross-Validation Procedure

**Leave-One-Dataset-Out (LODO)**: 16 folds. Each fold holds out all steps from one dataset. Detectors are trained on the remaining 15 datasets and evaluated on the held-out dataset. OOF predictions are collected across all 16 folds to produce 208-row prediction vectors.

**Rationale**: Per supervisor's requirement. Dataset-level grouping prevents information leakage from step correlation within a dataset.

### 7.2 Calibration Procedure (CRC)

1. Collect OOF predictions from LODO (208 rows each for gate and monitor)
2. For each alpha in KEY_ALPHAS and FINE_ALPHAS:
   - Compute CRC threshold λ = `sorted_faulty_probs[k]` where k = `floor(α × (n_faulty + 1)) − 1`
   - Compute naive quantile threshold for comparison
3. For LODO cross-conformal: hold out one dataset from CRC calibration, compute λ on remaining 15 datasets' OOF predictions, evaluate FSR on held-out dataset

**Double hold-out**: Each step's OOF probability is from a model that never saw its dataset (LODO). The CRC calibration on these OOF predictions uses genuinely held-out predictions. LODO cross-conformal further holds out one dataset from the CRC calibration set.

### 7.3 Metrics

**Per detector**:
- Precision, Recall, F1 (at 0.5 threshold for LGBM)
- AUROC, Average Precision (for probabilistic detectors)
- Per-fold metrics (16 LODO folds)

**Per fault class**: caught/total/recall at α = 0.10 for both tiers

**Cost analysis**:
- Tier 1: gross savings, FP cost, net savings (% of total wall-clock)
- Tier 2: additional true positives count
- Combined: recall, oracle ceiling

**CRC-specific**:
- FSR ≤ α at each key alpha (both tiers)
- LODO cross-conformal coverage ≥ 8/16 at α = 0.10 and α = 0.20 (both tiers)

---

## 8. CP-6 External Validation Contract

### 8.1 Scope

CP-6 is the **locked external test**. It runs fresh AIDE traces on new datasets with a stronger model and applies the frozen detectors and CRC thresholds.

### 8.2 Datasets

- **6–8 new datasets** from OpenML, explicitly **NOT from CC-18** (the 16 development datasets already cover CC-18).
- **Fixed candidate pool** (pre-registered): OpenML IDs `1590`, `1457`, `23517`, `168350`, `168757` (5 datasets from CP-2 transfer experience). Each must be verified as not a CC-18 member before running. If 6–8 datasets are needed, 1–3 additional non-CC-18 OpenML tabular classification datasets will be pre-registered and added to this list before any traces run.
- **No overlap** with the 16 development-corpus datasets listed in §3.1.
- Selection criterion: diverse task types (binary/multiclass), diverse sample sizes, comparable complexity to the development corpus.
- Dataset selection is finalised and documented in the CP-6 task brief *before* any traces are run.

### 8.3 Model

- **Stronger model** than o4-mini (e.g., o4 or equivalent) to test whether findings transfer across model capabilities
- Same AIDE scaffold, same 20-step configuration, same response-cache architecture

### 8.4 What Is Frozen (Carried Into CP-6)

| Component | Frozen From | What CP-6 Does With It |
|-----------|-------------|------------------------|
| Fault taxonomy (§2) | CP-3R | Labels new traces using the same F1/F2/F3 definitions |
| Feature specification (§4) | CP-3R/CP-4R | Extracts the same 16 features from new traces |
| Canonical hash v2 | CP-3R | Computes hashes for F1/RCP detection |
| Detector code | CP-4R | Runs frozen `detectors.py` — code unchanged |
| LGBM_PARAMS | CP-4R | Frozen hyperparameters used when retraining (see §8.4.1) |
| CRC thresholds (λ_gate, λ_monitor) | CP-5R | Applied as fixed thresholds to new predictions |
| Evaluation protocol | CP-5R | Same metrics, same procedures |

#### 8.4.1 Detector Model Application Strategy

CP-6 does **not** ship pre-fitted model artifacts from the development corpus. Instead:

1. **Rule-based detectors** (D1, D2_gate): stateless — applied directly to new features with no training.
2. **LightGBM detectors** (D3_gate, D3_monitor, D_any_gate, D_any_monitor): **retrained on the full 16-dataset development corpus** using frozen `LGBM_PARAMS` and frozen `detectors.py` code, then applied to CP-6 data. This retraining uses only development data — no CP-6 data enters the training set.
3. **D2_monitor threshold**: tuned on the full 16-dataset development corpus (not on CP-6 data), using the frozen `SCORE_THRESHOLDS` sweep.
4. **CRC thresholds**: the frozen λ_gate and λ_monitor from CP-5R are applied as fixed decision boundaries to the CP-6 predictions. No recalibration.

**Rationale**: Saving fitted LightGBM artifacts would introduce a dependency on the exact library version and serialisation format. Retraining from frozen code + frozen hyperparameters on frozen data is deterministic (seed=42) and produces the identical model.

### 8.5 What CP-6 Produces (New)

1. Fresh AIDE traces on 6–8 new datasets
2. Feature extraction and fault labeling using frozen pipeline
3. Detector predictions using frozen models
4. CRC evaluation using frozen thresholds
5. **Primary outcome**: Does FSR ≤ α hold on completely unseen data?
6. **Secondary outcomes**: Tier 1 net savings on new data; combined recall; per-class breakdown

### 8.6 Success Criteria

| Criterion | Required | Stretch |
|-----------|----------|---------|
| Gate FSR ≤ α at α = 0.10 on CP-6 data | Yes | — |
| Monitor FSR ≤ α at α = 0.10 on CP-6 data | Yes | — |
| Gate F1 ≥ 0.70 (any_fault) | Yes | ≥ 0.85 |
| Tier 1 net savings > 0% | Yes | > 30% |
| ≥ 4/6 CP-6 datasets individually satisfy FSR ≤ 0.20 | Stretch | ≥ 5/6 |

### 8.7 What CP-6 May NOT Do

- Retune any detector hyperparameter
- Modify the feature set
- Change CRC threshold computation
- Select datasets after seeing preliminary results
- Adjust success criteria after seeing results
- Re-derive any metric definition

---

## 9. Calibration Circularity Argument

**Claim**: The CP-5R CRC calibration has *reduced circularity* via held-out OOF predictions, not zero circularity in the strictest exchangeability sense.

**Evidence chain**:

1. **LODO produces genuinely held-out predictions**: each step's OOF probability comes from a model that never trained on that step's dataset.
2. **CRC calibrates on these OOF predictions**: the calibration set consists of 208 predictions that are each held-out with respect to the model that produced them.
3. **LODO cross-conformal provides a double hold-out**: it further holds out one dataset from the CRC calibration set, testing whether the threshold generalises across dataset compositions.
4. **We do NOT claim exchangeability across datasets**: the marginal CRC guarantee holds under the finite-sample correction; per-dataset conditional coverage varies (reflected in LODO coverage ≥ 8/16, not 16/16).
5. **CP-6 is the locked external test**: it provides the exchangeability-style validation that LODO alone cannot guarantee.

**Paper framing**: "Reduced circularity / held-out OOF calibration evidence" — not "no circularity."

---

## 10. Supervisor Requirements Traceability

| Dr. Ng's Requirement | Where Addressed |
|----------------------|-----------------|
| (1) Offline replay — build trace corpus once, evaluate offline | CP-3R: trace corpus built; CP-4R/CP-5R: all evaluation from pre-computed features/predictions |
| (2) Response cache for deterministic replay | AIDE prompt-hash response cache |
| (3) Instrumentation oracle as upper-bound baseline | CP-3R: oracle P/R/F1 = 1.0, cost ceiling = 55.69% |
| (4) Per-dataset null distribution, stratified by effect size | CP-4R: F2 effect-size bins (negligible/small/medium/large) |
| (5) Leave-one-dataset-out CV | CP-4R: 16-fold LODO; CP-5R: LODO cross-conformal |
| (6) Replicate 3–4 datasets with stronger model | CP-6 contract: 6–8 new datasets with stronger model |

### User-Approved Adjustments (4)

| Adjustment | Where Addressed |
|------------|-----------------|
| No 12/4 split — use nested/grouped cross-fitting, CP-6 as locked test | LODO is cross-fitting; CP-6 = locked external test |
| Strengthen F1 canonical hash → RCP | CP-3R: canonical hash v2 (order-preserving + params) |
| Refine Tier 2 savings claim | §6.4: strict "additional detections" language; no savings claim for Tier 2 |
| Do not optimise D3 to inflate metrics | CP-4R: standard LGBM_PARAMS, results reported as-is |

---

## 11. File Manifest

All files below reside on the execution machine at `~/Desktop/AIDE/cp3_fault_injection/`. SHA-256 hashes to be computed on the frozen files and appended by the CP-6 Codex task.

### 11.1 Scripts (Frozen Code)

| File | Description |
|------|-------------|
| `scripts/feature_extractor.py` | Canonical hash v2, StepConfig, feature extraction |
| `scripts/fault_labeler.py` | F1/F2/F3 labeling logic |
| `scripts/detectors.py` | All 7 detectors (CP-4R version) |
| `scripts/conformal_gate.py` | CRC module (CP-5R version) |
| `scripts/run_cp3.py` | CP-3R pipeline |
| `scripts/run_cp4.py` | CP-4R pipeline |
| `scripts/run_cp5.py` | CP-5R pipeline |

### 11.2 Data Artifacts (Frozen Outputs)

| File | Rows | Description |
|------|------|-------------|
| `features/step_features.csv` | 208 | Feature matrix (16 features + metadata) |
| `labels/combined_labels.csv` | 208 | Ground-truth fault labels (F1, F2, F3, any_fault) |
| `oracle/oracle_evaluation.json` | — | Oracle detector evaluation and upper-bound baseline |
| `cp4_results/cp4r_evaluation.json` | — | Full CP-4R detector evaluation results |
| `cp4_results/oof_predictions_D_any_gate.csv` | 208 | Gate OOF predictions |
| `cp4_results/oof_predictions_D_any_monitor.csv` | 208 | Monitor OOF predictions |
| `cp5_results/cp5r_evaluation.json` | — | Full CP-5R CRC evaluation results |
| `cp5_results/conformal_thresholds.csv` | 12 | CRC thresholds (6 alphas × 2 tiers) |
| `cp5_results/lodo_cross_conformal.csv` | 192 | LODO cross-conformal (16 datasets × 6 alphas × 2 tiers) |
| `cp5_results/per_class_analysis.csv` | 6 | Per-class analysis (3 classes × 2 tiers) |
| `analysis/fault_taxonomy.json` | — | Formal fault taxonomy |
| `analysis/rcp_construct_analysis.csv` | — | F1/RCP construct validity evidence |
| `analysis/file_manifest.json` | — | File hashes from CP-3R |

### 11.3 Verification Scripts

| File | Checks | Description |
|------|--------|-------------|
| `verify_cp3r.py` | 35 | CP-3R acceptance verification |
| `verify_cp4r.py` | 37 | CP-4R acceptance verification |
| `verify_cp5r.py` | 38 | CP-5R acceptance verification |

---

## 12. Protocol Amendments

*None. This section is reserved for post-freeze changes. Each amendment must include: amendment number, date, what changed, justification, and impact assessment.*

| # | Date | Change | Justification | Impact |
|---|------|--------|---------------|--------|
| — | — | — | — | — |

---

## 13. Checkpoint Completion Record

| Checkpoint | Date | Verifier Result | Key Outcome |
|------------|------|-----------------|-------------|
| CP-3R | Sep 2026 | PASS (35/35) | Trace corpus repaired: canonical hash v2, feature tiers, reconciliation |
| CP-4R | Sep 2026 | PASS (37/37) | 7 tiered detectors evaluated via LODO; D_any_gate F1 = 0.9106, D_any_monitor F1 = 0.9580 |
| CP-5R | Sep 2026 | 36/38 (2 legitimate findings) | CRC calibrated; gate net savings 48.46%; 2 findings documented, not forced |
| **FREEZE** | **15 Sep 2026** | **This document** | **All definitions, parameters, thresholds locked** |
| CP-6 | Pending | — | External validation on 6–8 new datasets |

---

*End of freeze protocol. This document is the authoritative reference for all frozen definitions. CP-6 and paper writing proceed from this locked state.*
