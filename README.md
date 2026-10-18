# Harness Engineering for Agentic AutoML

Reproducibility package for the paper *Harness Engineering for Agentic AutoML: An
Explainable Step-Level Monitor for Memory, Configuration and Tool Execution Faults in
LLM-Driven Pipeline Optimization*.

A conformal trial-gating watcher that decides, before each AutoML trial runs, whether to
execute or skip it, with a distribution-free guarantee on the false-skip rate. It uses
only the validation scores, canonical configuration hashes and execution metadata that
the agent already produces, so it needs no extra model call.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Feature extraction, fault labelling, the seven detectors, the CRC gate, and the CP-3R to CP-6 pipeline drivers |
| `features/` | `step_features.csv` (208 steps x 16 features), tier map, feature metadata |
| `labels/` | `combined_labels.csv`, `injection_manifest.json` with per-injection score deltas |
| `cp4_results/` | Leave-one-dataset-out detector evaluation and out-of-fold predictions |
| `cp5_results/` | CRC thresholds at six alpha levels, LODO cross-conformal results, per-class analysis |
| `cp6_results/` | External validation on six held-out datasets with a stronger backbone model |
| `analysis/` | RCP construct analysis, F2/F3 score deltas, fault taxonomy, natural fault rates |
| `oracle/` | Instrumentation oracle (upper-bound baseline) and its evaluation |
| `paper/` | Figure and table generation script, generated figures, LaTeX tables |
| `FREEZE_PROTOCOL.md` | The locked definitions, detectors, thresholds and evaluation procedure |
| `freeze_manifest.json` | SHA-256 hashes of every frozen artifact |

## The frozen protocol

Every detector parameter, feature definition and threshold was locked in
`FREEZE_PROTOCOL.md` v2.1 **before** the external validation was run. The constants that
matter:

```
LAMBDA_GATE    = 0.324949     # CRC threshold, Tier 1, alpha = 0.10
LAMBDA_MONITOR = 0.951217     # CRC threshold, Tier 2, alpha = 0.10
ALPHA          = 0.10
LGBM_PARAMS    = n_estimators=100, max_depth=4, learning_rate=0.1,
                 min_child_samples=5, class_weight=balanced, random_state=42
```

No value above may change without a numbered amendment to the protocol. Verify the
artifacts still match their recorded hashes before trusting a re-run:

```bash
python scripts/verify_freeze.py        # if present; otherwise see freeze_manifest.json
```

> **Note on the figure generator.** `paper/generate_paper_figures.py` recalculated the
> development-corpus CRC sweep instead of reading the frozen values, which the protocol
> forbids. Its CP-6 outputs are correct; its development-corpus outputs are not. The
> figures in `paper/figures/` prefixed `fig02`, `fig05`, `fig06`, `fig07` and `fig17`
> were regenerated directly from `cp5_results/conformal_thresholds.csv`,
> `cp5_results/lodo_cross_conformal.csv` and `labels/injection_manifest.json`. Fix the
> generator before re-running it.

## Reproducing the results

Offline reproduction needs no API key and no GPU. The trace corpus is pre-computed, so
every detector result can be recreated at zero marginal cost.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-cp.txt

python scripts/run_cp4.py --output-dir .     # detectors + LODO cross-validation
python scripts/run_cp5.py --output-dir .     # conformal risk control calibration
python scripts/run_cp6.py --output-dir .     # external validation, frozen thresholds

python verify_cp4r.py .                      # 37 checks
python verify_cp5r.py .                      # 38 checks, 2 documented findings
python verify_cp6.py  .                      # 32 checks
```

Regenerating the raw AIDE traces instead of using the supplied corpus requires an OpenAI
API key and costs roughly $15 in total.

## Headline results

Development corpus: 208 non-buggy steps across 16 OpenML CC-18 datasets.
External validation: 89 steps across 6 held-out datasets, stronger backbone model.

| | Development | External (CP-6) |
|---|---|---|
| Gate false-skip rate at alpha = 0.10 | 0.0909 | 0.0741 |
| Gate recall | 0.9091 | 0.9259 |
| Gate F1 | 0.9167 | 0.8475 |
| Net wall-clock savings | 48.46% | 39.64% |
| Oracle ceiling | 55.69% | 55.05% |

Reported as found, not as hoped: the Tier 2 monitor misses its false-skip bound on the
external corpus (0.1111 against a 0.10 target), LODO cross-conformal coverage is 11/16
at alpha = 0.10 rather than the marginal guarantee's nominal rate, and one external
dataset (`ada`) returns negative net savings of -11.8%.

## Data sources

Datasets are drawn from OpenML CC-18 and from the AutoML Benchmark suite 271, all under
open licences. No proprietary or personal data is used.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The vendored AIDE scaffold is MIT licensed and is the property of
its authors.
