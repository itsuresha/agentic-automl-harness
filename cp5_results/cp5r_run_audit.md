# CP-5R Run Audit

Date: 2026-09-15

## Status

CP-5R pipeline completed successfully and produced the expected output files. The provided acceptance verifier reported 36/38 checks passing.

## Verifier failures

1. Monitor CRC threshold < 0.5 at alpha=0.10 failed: lambda_monitor=0.951217.
2. Monitor CRC FSR <= default FSR at alpha=0.10 failed: CRC FSR=0.0909 vs default FSR=0.0579.

## Interpretation

These failures are not runtime failures. They show that the CP5R brief's monitor-tier acceptance expectations are too strong for the observed CP4R monitor probability distribution. The monitor CRC threshold still satisfies the requested alpha=0.10 FSR target, but the default monitor threshold theta=0.5 is more aggressive and has lower FSR on this internal OOF sample.

## Key alpha=0.10 results

- Gate CRC: lambda=0.324949, FSR=0.0909, recall=0.9091, F1=0.9167.
- Monitor CRC: lambda=0.951217, FSR=0.0909, recall=0.9091, F1=0.9524.
- Gate default theta=0.5: FSR=0.1157, recall=0.8843, F1=0.9106.
- Monitor default theta=0.5: FSR=0.0579, recall=0.9421, F1=0.9580.
- Tier 1 net savings at alpha=0.10: 48.46%.
- Tier 2 additional true positives at alpha=0.10: 3.
- Combined recall at alpha=0.10: 0.9339.

## Recommendation

Do not force the verifier to pass by weakening the evidence. For publication, report CP5R transparently: Tier 1 supports the compute-savings claim; Tier 2 supports post-execution monitoring, with alpha=0.05 or default theta=0.5 appearing stronger than CRC alpha=0.10 for the monitor on this internal sample.
