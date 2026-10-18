# Freeze Audit — CP3R/CP4R/CP5R

Date: 2026-09-15

Protocol: FREEZE_PROTOCOL.md version 2.1, project copy corrected to list the actual oracle artifact `oracle/oracle_evaluation.json`.

## Status

FROZEN. The approved freeze protocol has been copied into the project, corrected for the oracle artifact filename, and core frozen artifacts have been hashed in `freeze_manifest.json`.

## Verification Results

- CP3R: 35/35 PASS
- CP4R: 37/37 PASS
- CP5R: 36/38, with the two monitor-tier findings documented in the protocol and `cp5_results/cp5r_run_audit.md`.

## Frozen CP5R α=0.10 Thresholds

- Gate threshold: 0.324949; FSR: 0.0909
- Monitor threshold: 0.951217; FSR: 0.0909

## Frozen Corpus

- Steps: 208
- Faulty: 121
- Clean: 87
- Datasets: 16

## Notes

- CP6 must use non-CC18 OpenML datasets and verify candidate IDs before running.
- Tier 1 is the only direct compute-savings claim. Tier 2 is post-execution monitoring/additional detection.
- No API calls were made during this freeze run.
