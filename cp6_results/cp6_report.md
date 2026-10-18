# CP-6 External Validation Report

**Freeze protocol**: v2.1 (15 Sep 2026)
**CP-6 corpus**: 89 steps, 54 faulty, 6 datasets
**CRC α**: 0.1

## 1. Primary Results

| Metric | Gate (Tier 1) | Monitor (Tier 2) |
|--------|---------------|-------------------|
| λ (frozen) | 0.324949 | 0.951217 |
| FSR | 0.0741 | 0.1111 |
| Recall | 0.9259 | 0.8889 |
| Precision | 0.7812 | 1.0000 |
| F1 | 0.8475 | 0.9412 |

**Combined recall**: 0.9259 (gate + 0 monitor additional)

## 2. Tier 1 Savings

- Gross savings: 53.80%
- FP cost: 14.16%
- **Net savings: 39.64%**
- Oracle ceiling: 55.05%

## 3. Success Criteria

| Criterion | Required | Actual | Pass |
|-----------|----------|--------|------|
| Gate FSR ≤ 0.10 | Yes | 0.0741 | ✓ |
| Monitor FSR ≤ 0.10 | Yes | 0.1111 | ✗ |
| Gate F1 ≥ 0.70 | Yes | 0.8475 | ✓ |
| Net savings > 0% | Yes | 39.64% | ✓ |
| ≥4 datasets gate FSR ≤ 0.20 | Stretch | 6/6 | ✓ |

## 4. Per-Dataset Breakdown

| Dataset | Steps | Faulty | Gate FSR | Monitor FSR | Net Savings |
|---------|-------|--------|----------|-------------|------------|
| ada | 17 | 11 | 0.1818 | 0.1818 | -11.76% |
| fabert | 17 | 12 | 0.0000 | 0.0000 | 74.88% |
| philippine | 13 | 10 | 0.0000 | 0.1000 | 82.92% |
| sylvine | 17 | 12 | 0.1667 | 0.2500 | 48.94% |
| wine-quality-white | 12 | 5 | 0.0000 | 0.0000 | 62.51% |
| yeast | 13 | 4 | 0.0000 | 0.0000 | 19.29% |

## 5. Per-Class Analysis at α=0.10

| Tier | Class | Caught | Total | Recall |
|------|-------|--------|-------|--------|
| Gate | F1 | 31 | 33 | 0.9394 |
| Gate | F2 | 12 | 12 | 1.0000 |
| Gate | F3 | 14 | 18 | 0.7778 |
| Monitor | F1 | 30 | 33 | 0.9091 |
| Monitor | F2 | 12 | 12 | 1.0000 |
| Monitor | F3 | 13 | 18 | 0.7222 |

## 6. Individual Detector Comparison

| Detector | Target | P | R | F1 |
|----------|--------|---|---|-----|
| D1 | F1 | 1.0000 | 0.8182 | 0.9000 |
| D2_gate | F2 | 0.6000 | 1.0000 | 0.7500 |
| D2_monitor | F2 | 0.5217 | 1.0000 | 0.6857 |
| D3_gate | F3 | 0.4074 | 0.6111 | 0.4889 |
| D3_monitor | F3 | 0.7222 | 0.7222 | 0.7222 |
| D_any_gate | any_fault | 0.8136 | 0.8889 | 0.8496 |
| D_any_monitor | any_fault | 0.9286 | 0.9630 | 0.9455 |

---

**Overall: 4/5 success criteria met.**