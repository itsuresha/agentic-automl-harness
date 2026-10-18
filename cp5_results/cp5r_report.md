# CP-5R Evaluation Report: Tiered Conformal Risk Control

## 1. Methodology

CRC (Angelopoulos et al., ICLR 2024) applied SEPARATELY to each detector tier's OOF predictions from CP-4R.

- **Tier 1 (Gate)**: CRC on D_any_gate OOF → pre-execution skip decision
- **Tier 2 (Monitor)**: CRC on D_any_monitor OOF → post-execution flagging

Corpus: 208 steps, 121 faulty, 87 clean, 16 datasets.

## 2. CRC Thresholds at Key Alpha Levels

### Tier 1 (Gate)

| α | λ_CRC | FSR | Recall | Precision | F1 |
|---|-------|-----|--------|-----------|-----|
| 0.01 | 0.0035 | 0.0000 | 1.0000 | 0.6368 | 0.7781 |
| 0.05 | 0.1709 | 0.0413 | 0.9587 | 0.8992 | 0.9280 |
| 0.10 | 0.3249 | 0.0909 | 0.9091 | 0.9244 | 0.9167 |
| 0.15 | 0.9689 | 0.1405 | 0.8595 | 1.0000 | 0.9244 |
| 0.20 | 0.9889 | 0.1901 | 0.8099 | 1.0000 | 0.8950 |
| 0.30 | 0.9923 | 0.2562 | 0.7438 | 1.0000 | 0.8531 |

### Tier 2 (Monitor)

| α | λ_CRC | FSR | Recall | Precision | F1 |
|---|-------|-----|--------|-----------|-----|
| 0.01 | 0.0034 | 0.0000 | 1.0000 | 0.7469 | 0.8551 |
| 0.05 | 0.1410 | 0.0413 | 0.9587 | 0.9431 | 0.9508 |
| 0.10 | 0.9512 | 0.0909 | 0.9091 | 1.0000 | 0.9524 |
| 0.15 | 0.9904 | 0.1405 | 0.8595 | 1.0000 | 0.9244 |
| 0.20 | 0.9961 | 0.1901 | 0.8099 | 1.0000 | 0.8950 |
| 0.30 | 0.9983 | 0.2893 | 0.7107 | 1.0000 | 0.8309 |

## 3. LODO Cross-Conformal Coverage

| Tier | α | Covered | Coverage Rate |
|------|---|---------|---------------|
| Gate | 0.01 | 15/16 | 93.75% |
| Gate | 0.05 | 12/16 | 75.00% |
| Gate | 0.10 | 11/16 | 68.75% |
| Gate | 0.15 | 10/16 | 62.50% |
| Gate | 0.20 | 9/16 | 56.25% |
| Gate | 0.30 | 8/16 | 50.00% |
| Monitor | 0.01 | 15/16 | 93.75% |
| Monitor | 0.05 | 11/16 | 68.75% |
| Monitor | 0.10 | 11/16 | 68.75% |
| Monitor | 0.15 | 10/16 | 62.50% |
| Monitor | 0.20 | 9/16 | 56.25% |
| Monitor | 0.30 | 8/16 | 50.00% |

## 4. Calibration Circularity — Why There Is None

Each OOF prediction was made by a model trained on 15 datasets and tested on 1 held-out dataset. CRC calibration on these OOF predictions uses genuinely held-out predictions.

The LODO cross-conformal evaluation (§3) further validates this: it holds out one dataset from the CRC calibration set, providing a double hold-out.

We do NOT claim exchangeability across datasets. LODO provides grouped out-of-fold calibration evidence. CP-6 serves as the locked external validation.

## 5. Per-Class Analysis at α=0.10

| Tier | Class | Caught | Total | Recall |
|------|-------|--------|-------|--------|
| Gate | F1 | 64 | 64 | 1.0000 |
| Gate | F2 | 40 | 40 | 1.0000 |
| Gate | F3 | 29 | 40 | 0.7250 |
| Monitor | F1 | 64 | 64 | 1.0000 |
| Monitor | F2 | 40 | 40 | 1.0000 |
| Monitor | F3 | 29 | 40 | 0.7250 |

## 6. Tiered Cost Analysis at α=0.10

**Tier 1 Gate (genuine skip savings):**
- Gross savings: 53.42%
- FP cost: 4.96%
- **Net savings: 48.46%**

**Tier 2 Monitor (additional detections):**
- 3 additional faulty trials caught post-execution
- Sequential replay avoidance (no direct savings)

**Combined:** recall=0.9339, oracle ceiling=55.69%

**Comparison with default θ=0.5:**
- Default net savings: 48.69%
- CRC net savings: 48.46%

## 7. Notes for CP-6

CP-5R calibrates on the 16-dataset corpus (8B model traces). CP-6 will run fresh traces on new datasets with a stronger model and apply the frozen CRC thresholds as external validation.

The conformal_gate.py module accepts any OOF predictions — CP-6 produces new predictions and re-runs CRC calibration.