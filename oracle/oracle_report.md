# CP-3 Oracle Detector - Summary

**Corpus**: 16 datasets, 320 steps (208 non-buggy, 112 buggy)
**Seed**: 42 | **Runtime**: 7804.54s | **Re-execution**: enabled

## Labels

| Class | Natural | Synthetic | Total |
|-------|---------|-----------|-------|
| F1 stale memory | 64 | 0 | 64 |
| F2 data leakage | 15 | 25 | 40 |
| F3 partial fit | 0 | 40 | 40 |

Faulty steps: **121** of 208 non-buggy
(**58.2%**); clean: 87.

## Oracle performance

Precision / recall / F1 = **1.0** for every class, by construction. This is the
ceiling: no CP-4 detector can exceed it.

## Compute-savings ceiling

A perfect gate skipping every faulty trial would save:

- **47.50%** of non-buggy wall clock
  (7055s of 14852s)
- ~1,133,789 tokens, ~$2.14 API cost (apportioned by wall clock)
- 121 of 208 trials gated

## Signal strength

- **F2 score inflation** (synthetic, viable only): mean 0.022592000000000004, n=25 (excluded 0 pipeline crashes)
- **F3 score degradation** (synthetic, viable only): mean -0.07297567567567566, n=37 (excluded 3 pipeline crashes)

## Catastrophic-Failure Filter

The original viability thresholds are unchanged. Per-strategy means exclude non-viable results;
counts include all successfully executed mutations. Assignment cycles per model class, not by outcome.

Per-strategy breakdown: {"clamp_depth": {"attempted": 16, "crashed": 0, "mean_delta": -0.043924999999999985, "n_measured": 12, "succeeded": 12}, "clamp_iter": {"attempted": 1, "crashed": 0, "mean_delta": -0.0004999999999999449, "n_measured": 1, "succeeded": 1}, "clamp_lr": {"attempted": 16, "crashed": 3, "mean_delta": -0.09159166666666667, "n_measured": 12, "succeeded": 15}, "clamp_n_est": {"attempted": 16, "crashed": 0, "mean_delta": -0.08944999999999999, "n_measured": 12, "succeeded": 12}}
