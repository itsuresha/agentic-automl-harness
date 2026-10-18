# CP-1 and CP-2 execution report

Updated: 2026-09-14T03:03:07+00:00

## CP-1 — completed (zero API cost)

5 outer stratified folds × 3 inner stratified folds. Thresholds are selected on inner folds, then the median is applied to the outer test fold. Encoders are fitted on training data only. SD uses ddof=0.

| Ensemble | Nested macro F1 (mean ± SD) | Default 0.5 mean | 0.7203 − nested mean |
|---|---:|---:|---:|
| brief | 0.701850 ± 0.029049 | 0.694449 | 0.018450 |
| saved_best | 0.705194 ± 0.016463 | 0.688979 | 0.015106 |

The brief specifies depths 3/4/5/6/7, 300 trees, and extra regularization. The actual saved best uses depths 3/5/7/9/11 and 500 trees. Both were evaluated; the saved-best result is the closer validation of the original trial. The brief-profile gap also includes a model change.

These scores correct direct threshold leakage. The dataset was already used to choose the AIDE model, so this does not erase earlier adaptive model-selection bias or create an untouched test set. Fold SD is not a confidence interval.

CP-1 runtime: 29.209 seconds. All fold indices, thresholds, and 1,000 outer predictions per profile are saved in [validation_report.json](../validation_report.json).

## CP-2 — corpus status

Completed datasets: **16/17**. Known saved-run cost estimate: **$3.5711383**. Recorded run time: **10.0127 hours** (excluding downloads, pauses, and setup).

Cost includes the imported Credit-G run; preparing data and constructing its cache caused no new API charges. Future run estimates account for cached input when returned by the provider. Missing/unsettled request usage is not an exact bill.

| Dataset | ID | Group | Status | Best reported F1 | Error/status reason |
|---|---:|---|---|---:|---|
| credit-g | 31 | cc18 | SUCCESS | 0.7203 | none |
| diabetes | 37 | cc18 | SUCCESS | 0.7687 | none |
| tic-tac-toe | 50 | cc18 | SUCCESS | 1.0 | none |
| vehicle | 54 | cc18 | SUCCESS | 0.7596 | none |
| credit-approval | 29 | cc18 | SUCCESS | 0.8972 | none |
| kr-vs-kp | 3 | cc18 | SUCCESS | 0.9966 | none |
| mfeat-factors | 12 | cc18 | SUCCESS | 0.9645 | none |
| mfeat-fourier | 14 | cc18 | SUCCESS | 0.8437 | none |
| mfeat-morphological | 18 | cc18 | SUCCESS | 0.7409 | none |
| mfeat-zernike | 22 | cc18 | SUCCESS | 0.789 | none |
| pendigits | 32 | cc18 | SUCCESS | 0.9942 | none |
| letter | 6 | cc18 | SUCCESS | 0.9711 | none |
| jasmine | 41143 | transfer | SUCCESS | 0.8225 | none |
| amazon-commerce-reviews | 1457 | transfer | SUCCESS | 0.7220848014730368 | none |
| Satellite | 40900 | transfer | SUCCESS | 0.9142 | none |
| micro-mass | 1515 | transfer | FAILED | 0.8795 | APIConnectionError: Connection error. |
| christine | 41142 | transfer | SUCCESS | 0.756 | none |

Transfer cohort verification: **PASSED**, checked 2026-09-14T03:03:02+00:00. See [membership evidence and replacements](transfer_cohort_verification.json). The active transfers must be outside CC-18 suite 99 and members of AutoML Benchmark classification suite 271. This is rechecked before every batch; failure blocks all paid execution.

Requested 1590 (adult) and 23517 (numerai28.6) were rejected as CC-18 members; 168350 and 168757 returned Unknown dataset. Retained 1457 (amazon-commerce-reviews). Replacements: 1590 → 41143 (jasmine), 23517 → 40900 (Satellite), 168350 → 1515 (micro-mass), 168757 → 41142 (christine). Previous transfer directories remain preserved but are excluded from the active cohort summaries. Outside-suite membership alone is not evidence of domain independence.

Cache status: 16 exact datasets; 1 approximate historical datasets. Credit-G's 40 prompt/response records were recovered from Rich console output. Original whitespace, full request settings and precise per-call UTC times cannot be recovered. These entries are explicitly rejected by exact replay lookup.

## Files and continuation

- [Dataset inventory](dataset_inventory.json)
- [Transfer cohort verification](transfer_cohort_verification.json)
- [Corpus summary CSV](trace_corpus_summary.csv)
- [Cache status](cache_summary.json)
- [Offline preflight log](cp-preflight.log)
- [Runner instructions](../CP1_CP2_README.md)

Offline regression preflight: **PASSED**. Tests use mocked responses, not paid API calls. The one-node visualization still emits the upstream divide-by-zero layout warning; it does not fail the worker. Full test output is retained in the linked log.

Run the remaining paid corpus from the terminal session holding the replacement key:

```bash
cd /Users/suresha/Desktop/AIDE
.venv/bin/python scripts/batch_runner.py --run --approve-cost
```

The default estimated-cost stop threshold is $6 for newly captured calls across attempts. It is checked before each request; an in-flight request can exceed it. It is not a provider billing cap. Each dataset has 20 AIDE steps and 900-second candidate timeouts, with 30 seconds between datasets. The brief's ~$4 and ~3.5 hours are estimates, not guaranteed bounds.
