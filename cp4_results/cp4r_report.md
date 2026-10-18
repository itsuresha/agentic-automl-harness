# CP-4R Evaluation Report: Tiered Practical Detectors

All detectors use 16-fold leave-one-dataset-out cross-validation.
Tier 1 detectors are pre-execution gates. Tier 2 detectors are post-execution monitors.

## Detector Results

| Detector | Tier | Target | P | R | F1 | AUROC |
|---|---|---|---:|---:|---:|---:|
| D1 | gate | F1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| D2_gate | gate | F2 | 0.6897 | 1.0000 | 0.8163 | 0.9464 |
| D2_monitor | all | F2 | 0.6349 | 1.0000 | 0.7767 | 0.9345 |
| D3_gate | gate | F3 | 0.5082 | 0.7750 | 0.6139 | 0.8351 |
| D3_monitor | all | F3 | 0.5455 | 0.6000 | 0.5714 | 0.8580 |
| D_any_gate | gate | any_fault | 0.9386 | 0.8843 | 0.9106 | 0.9758 |
| D_any_monitor | all | any_fault | 0.9744 | 0.9421 | 0.9580 | 0.9878 |

## D1 Tautology Disclosure

D1 uses hash_seen_before to detect F1/RCP. This feature is the definition of F1: a step is labeled F1 iff its canonical hash was seen in an earlier step. D1's perfect P/R/F1 is therefore by construction.

Independent CP-3R construct summary:

```json
{
  "n_rcp_steps_analyzed": 46,
  "n_rcp_steps_with_donor": 46,
  "n_rcp_total": 64,
  "mean_abs_delta": 0.04033963569820349,
  "median_abs_delta": 0.0026000000000000467,
  "std_delta": 0.15972233741352648,
  "pct_within_001": 41.30434782608695,
  "pct_within_005": 69.56521739130434,
  "pct_improved": 41.30434782608695,
  "pct_matched": 41.30434782608695,
  "pct_degraded": 17.391304347826086
}
```

## Savings Analysis

Tier 1 genuine skip savings: 52.99%.
Tier 2 additional detections: 8.

## CP-5R Inputs

- `oof_predictions_D_any_gate.csv`
- `oof_predictions_D_any_monitor.csv`
