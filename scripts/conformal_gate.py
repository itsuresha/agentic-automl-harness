"""CP-5R: Tiered Conformal Risk Control gating layer.

Redesign of CP-5 to match CP-4R's tiered detector architecture:
  - Tier 1 (gate): CRC on D_any_gate OOF → pre-execution skip
  - Tier 2 (monitor): CRC on D_any_monitor OOF → post-execution flag

Fixes audit finding #4: calibration circularity resolved via LODO
double hold-out argument (§3.5 of spec).

Does NOT import from detectors.py — self-contained.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# ── Key alpha levels ─────────────────────────────────────────────────
KEY_ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
FINE_ALPHAS = np.arange(0.01, 0.51, 0.01).tolist()


# ── CRC Threshold ────────────────────────────────────────────────────
def crc_threshold(faulty_probs: np.ndarray, alpha: float) -> float:
    """Conformal Risk Control threshold (Angelopoulos et al., ICLR 2024).

    λ̂ = sup{λ : R̂(λ) ≤ α}
    where R̂(λ) = (1 + #{faulty with prob < λ}) / (n_f + 1)

    Equivalent: k_max = floor(α(n_f + 1) − 1), λ̂ = sorted[k_max].
    """
    n_f = len(faulty_probs)
    if n_f == 0:
        return 0.0
    sorted_probs = np.sort(faulty_probs)
    k_max = int(np.floor(alpha * (n_f + 1) - 1))
    if k_max < 0:
        return 0.0
    k_max = min(k_max, n_f - 1)
    return float(sorted_probs[k_max])


def naive_quantile_threshold(faulty_probs: np.ndarray,
                              alpha: float) -> float:
    """Naive quantile threshold (no +1 correction, no formal guarantee)."""
    n_f = len(faulty_probs)
    if n_f == 0:
        return 0.0
    sorted_probs = np.sort(faulty_probs)
    k = int(np.floor(alpha * n_f))
    k = min(k, n_f - 1)
    return float(sorted_probs[k])


# ── Gate/Flag Evaluation ─────────────────────────────────────────────
def evaluate_gate(y_true: np.ndarray, y_prob: np.ndarray,
                  threshold: float) -> dict:
    """Evaluate gating/flagging at a given threshold.

    Flag step if y_prob >= threshold.
    """
    flagged = (y_prob >= threshold).astype(int)

    tp = int(((flagged == 1) & (y_true == 1)).sum())
    fp = int(((flagged == 1) & (y_true == 0)).sum())
    fn = int(((flagged == 0) & (y_true == 1)).sum())
    tn = int(((flagged == 0) & (y_true == 0)).sum())

    n_faulty = int(y_true.sum())
    n_clean = int(len(y_true) - n_faulty)

    recall = tp / max(n_faulty, 1)
    fsr = 1.0 - recall  # false-skip-rate
    precision = tp / max(tp + fp, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-10)
          if (precision + recall) > 0 else 0.0)

    return {
        "threshold": round(threshold, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_faulty": n_faulty, "n_clean": n_clean,
        "n_flagged": tp + fp,
        "recall": round(recall, 4),
        "fsr": round(fsr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "flagged": flagged,
    }


# ── Alpha Sweep ──────────────────────────────────────────────────────
def sweep_alpha(y_true: np.ndarray, y_prob: np.ndarray,
                alphas: list, tier_name: str) -> list:
    """Sweep CRC + naive thresholds across alpha levels for one tier."""
    faulty_probs = y_prob[y_true == 1]
    results = []

    for alpha in alphas:
        crc_t = crc_threshold(faulty_probs, alpha)
        naive_t = naive_quantile_threshold(faulty_probs, alpha)

        crc_eval = evaluate_gate(y_true, y_prob, crc_t)
        naive_eval = evaluate_gate(y_true, y_prob, naive_t)

        results.append({
            "tier": tier_name,
            "alpha": round(alpha, 4),
            "crc_threshold": round(crc_t, 6),
            "naive_threshold": round(naive_t, 6),
            "crc_fsr": crc_eval["fsr"],
            "crc_recall": crc_eval["recall"],
            "crc_precision": crc_eval["precision"],
            "crc_f1": crc_eval["f1"],
            "crc_n_flagged": crc_eval["n_flagged"],
            "naive_fsr": naive_eval["fsr"],
            "naive_recall": naive_eval["recall"],
            "naive_precision": naive_eval["precision"],
            "naive_f1": naive_eval["f1"],
            "naive_n_flagged": naive_eval["n_flagged"],
        })

    return results


# ── LODO Cross-Conformal ─────────────────────────────────────────────
def lodo_cross_conformal(datasets: np.ndarray, y_true: np.ndarray,
                          y_prob: np.ndarray, alphas: list,
                          tier_name: str) -> list:
    """Leave-one-dataset-out cross-conformal evaluation for one tier.

    For each held-out dataset:
      1. Calibrate CRC threshold on the OTHER 15 datasets' faulty probs
      2. Apply to the held-out dataset
      3. Check if FSR ≤ α (coverage)
    """
    unique_datasets = sorted(np.unique(datasets))
    results = []

    for alpha in alphas:
        for ds in unique_datasets:
            test_mask = datasets == ds
            train_mask = ~test_mask

            # Calibrate on training datasets
            train_faulty_probs = y_prob[train_mask & (y_true == 1)]
            lodo_threshold = crc_threshold(train_faulty_probs, alpha)

            # Evaluate on held-out dataset
            test_y_true = y_true[test_mask]
            test_y_prob = y_prob[test_mask]
            n_test_faulty = int(test_y_true.sum())

            if n_test_faulty > 0:
                flagged = (test_y_prob >= lodo_threshold).astype(int)
                test_tp = int(
                    ((flagged == 1) & (test_y_true == 1)).sum())
                test_fn = int(
                    ((flagged == 0) & (test_y_true == 1)).sum())
                test_fp = int(
                    ((flagged == 1) & (test_y_true == 0)).sum())
                test_recall = test_tp / n_test_faulty
                test_fsr = 1.0 - test_recall
                covered = test_fsr <= alpha
            else:
                test_tp = test_fn = test_fp = 0
                test_recall = 1.0
                test_fsr = 0.0
                covered = True

            results.append({
                "tier": tier_name,
                "alpha": round(alpha, 4),
                "dataset": ds,
                "n_test": int(test_mask.sum()),
                "n_test_faulty": n_test_faulty,
                "lodo_threshold": round(lodo_threshold, 6),
                "tp": test_tp,
                "fn": test_fn,
                "fp": test_fp,
                "recall": round(test_recall, 4),
                "fsr": round(test_fsr, 4),
                "covered": covered,
            })

    return results


# ── Per-Class Analysis ───────────────────────────────────────────────
def per_class_analysis(df: pd.DataFrame, flagged: np.ndarray,
                       tier_name: str) -> list:
    """Per fault class (F1, F2, F3) analysis at a given threshold."""
    results = []
    for cls in ["F1", "F2", "F3"]:
        cls_mask = df[cls].values.astype(int) == 1
        n_cls = int(cls_mask.sum())
        if n_cls > 0:
            caught = int((flagged[cls_mask] == 1).sum())
            missed = n_cls - caught
            recall = caught / n_cls
        else:
            caught = missed = 0
            recall = 1.0

        results.append({
            "tier": tier_name,
            "fault_class": cls,
            "n_total": n_cls,
            "caught": caught,
            "missed": missed,
            "recall": round(recall, 4),
        })

    return results


# ── Tiered Cost Analysis ─────────────────────────────────────────────
def tiered_cost_analysis(
    y_true: np.ndarray,
    wall_clock: np.ndarray,
    gate_flagged: np.ndarray,
    monitor_flagged: np.ndarray,
) -> dict:
    """Per-tier cost analysis.

    Tier 1 (gate): genuine skip savings — trial never runs.
    Tier 2 (monitor): additional detections — trial already ran, no savings.
    """
    total_wc = float(wall_clock.sum())

    # Tier 1: genuine skip savings
    gate_tp = (gate_flagged == 1) & (y_true == 1)
    gate_fp = (gate_flagged == 1) & (y_true == 0)
    t1_gross = float(wall_clock[gate_tp].sum())
    t1_fp_cost = float(wall_clock[gate_fp].sum())
    t1_net = t1_gross - t1_fp_cost

    # Tier 2: additional detections (caught by monitor, missed by gate)
    monitor_only_tp = ((monitor_flagged == 1) & (y_true == 1)
                       & (gate_flagged == 0))
    t2_additional_tp = int(monitor_only_tp.sum())
    t2_additional_wc = float(wall_clock[monitor_only_tp].sum())

    # Combined detection
    either_flagged_tp = (
        ((gate_flagged == 1) | (monitor_flagged == 1))
        & (y_true == 1)
    )
    combined_tp = int(either_flagged_tp.sum())
    combined_recall = combined_tp / max(int(y_true.sum()), 1)

    # Oracle
    oracle_wc = float(wall_clock[y_true == 1].sum())

    return {
        "tier1_gate": {
            "description": (
                "Genuine skip savings — flagged trials never execute"
            ),
            "true_positives": int(gate_tp.sum()),
            "false_positives": int(gate_fp.sum()),
            "gross_savings_sec": round(t1_gross, 2),
            "gross_savings_pct": round(t1_gross / total_wc * 100, 2),
            "fp_cost_sec": round(t1_fp_cost, 2),
            "fp_cost_pct": round(t1_fp_cost / total_wc * 100, 2),
            "net_savings_sec": round(t1_net, 2),
            "net_savings_pct": round(t1_net / total_wc * 100, 2),
        },
        "tier2_monitor_additional": {
            "description": (
                "Additional faulty trials caught post-execution. "
                "Trial already ran — no direct wall-clock savings. "
                "Prevents cascade errors (sequential replay avoidance)."
            ),
            "additional_true_positives": t2_additional_tp,
            "wall_clock_of_additional_sec": round(t2_additional_wc, 2),
        },
        "combined_detection": {
            "total_true_positives": combined_tp,
            "total_faulty": int(y_true.sum()),
            "combined_recall": round(combined_recall, 4),
            "tier1_net_savings_pct": round(t1_net / total_wc * 100, 2),
            "oracle_ceiling_pct": round(oracle_wc / total_wc * 100, 2),
        },
        "total_wall_clock_sec": round(total_wc, 2),
    }


# ── Default Baseline ─────────────────────────────────────────────────
def default_baseline(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float = 0.5) -> dict:
    """Default θ=0.5 baseline for comparison."""
    result = evaluate_gate(y_true, y_prob, threshold)
    return {
        "threshold": threshold,
        "fsr": result["fsr"],
        "recall": result["recall"],
        "precision": result["precision"],
        "f1": result["f1"],
        "n_flagged": result["n_flagged"],
        "tp": result["tp"],
        "fp": result["fp"],
    }
