"""CP-6 External Validation — Evaluation Pipeline.

Usage:
    cd ~/Desktop/AIDE/cp3_fault_injection
    ../.venv/bin/python scripts/run_cp6.py --output-dir .

Phases:
    0. Pre-flight: verify frozen artifacts exist + SHA-256 integrity
    1. Load dev corpus (features + labels)
    2. Load CP-6 corpus (features + labels from Phase C)
    3. Retrain detectors on full 16-dataset dev corpus
    4. Apply detectors to CP-6 data
    5. Apply frozen CRC thresholds
    6. Compute all metrics (FSR, F1, savings, per-dataset, per-class)
    7. Generate reports

Prerequisites:
    - Phases A–C complete (cp6_features.csv and cp6_labels.csv exist)
    - All frozen artifacts in place (features/, labels/, scripts/)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier


# ═══════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS — from FREEZE_PROTOCOL.md v2.1
# ═══════════════════════════════════════════════════════════════════

LAMBDA_GATE    = 0.324949
LAMBDA_MONITOR = 0.951217
ALPHA          = 0.10

LGBM_PARAMS = {
    "n_estimators":      100,
    "max_depth":         4,
    "learning_rate":     0.1,
    "min_child_samples": 5,
    "class_weight":      "balanced",
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            1,
}

TIER1_FEATURES = [
    "n_hyperparams", "n_preprocessing_steps",
    "preprocessing_before_split", "leaky_before_split",
    "hash_seen_before", "hash_first_seen_at",
    "parse_quality_full_ast",
]

TIER2_FEATURES_BASE = [
    "validation_score", "wall_clock_sec", "score_delta_from_prev",
    "score_rank_in_run", "is_best_so_far", "steps_since_improvement",
]

TIER2_DERIVED = [
    "score_rank_pct", "wall_clock_rank_pct", "within_run_z",
]

ALL_FEATURES = TIER1_FEATURES + TIER2_FEATURES_BASE + TIER2_DERIVED

SCORE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

KEY_ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


# ═══════════════════════════════════════════════════════════════════
# DERIVED FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns expected by the evaluation pipeline.

    The frozen feature extractor produces `parse_quality` as a string
    (full_ast | partial_ast | regex_fallback | unparseable).  The
    evaluation pipeline needs a binary indicator and three rank/z-score
    features that are computed here from the base columns.
    """
    # parse_quality_full_ast: binary indicator (1 = full AST parse)
    if "parse_quality" in df.columns and "parse_quality_full_ast" not in df.columns:
        df["parse_quality_full_ast"] = (df["parse_quality"] == "full_ast").astype(int)

    # score_rank_pct: per-dataset percentile rank of validation_score
    if "validation_score" in df.columns and "score_rank_pct" not in df.columns:
        df["score_rank_pct"] = df.groupby("dataset")["validation_score"].rank(pct=True)

    # wall_clock_rank_pct: per-dataset percentile rank of wall_clock_sec
    if "wall_clock_sec" in df.columns and "wall_clock_rank_pct" not in df.columns:
        df["wall_clock_rank_pct"] = df.groupby("dataset")["wall_clock_sec"].rank(pct=True)

    # within_run_z: within-dataset z-score of validation_score
    if "validation_score" in df.columns and "within_run_z" not in df.columns:
        grp_mean = df.groupby("dataset")["validation_score"].transform("mean")
        grp_std  = df.groupby("dataset")["validation_score"].transform("std")
        grp_std  = grp_std.fillna(0).replace(0, 1e-10)  # single-step datasets
        df["within_run_z"] = (df["validation_score"] - grp_mean) / grp_std

    return df


# ═══════════════════════════════════════════════════════════════════
# PHASE 0: PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════

def phase0_preflight(base_dir: Path) -> bool:
    """Verify frozen artifacts exist and match SHA-256 manifest."""
    print("\n[0/7] Pre-flight: verifying frozen artifacts...")

    required = [
        "features/step_features.csv",
        "labels/combined_labels.csv",
        "scripts/detectors.py",
        "scripts/conformal_gate.py",
        "scripts/feature_extractor.py",
        "scripts/fault_labeler.py",
    ]

    ok = True
    for rel in required:
        p = base_dir / rel
        if not p.exists():
            print(f"  MISSING: {rel}")
            ok = False
        else:
            print(f"  OK: {rel}")

    # Verify SHA-256 if manifest exists
    manifest_path = base_dir / "freeze_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Handle both formats: flat {path: hash} dict or
        # nested {"frozen_files": [{path, hash}, ...]} list
        hash_pairs = {}
        if isinstance(manifest, dict) and "frozen_files" in manifest:
            for entry in manifest["frozen_files"]:
                if isinstance(entry, dict) and "path" in entry and "hash" in entry:
                    hash_pairs[entry["path"]] = entry["hash"]
                elif isinstance(entry, dict) and "path" in entry and "sha256" in entry:
                    hash_pairs[entry["path"]] = entry["sha256"]
        elif isinstance(manifest, dict):
            # Flat {path: hash} format
            hash_pairs = {k: v for k, v in manifest.items()
                          if isinstance(v, str) and len(v) == 64}
        n_checked = 0
        n_mismatch = 0
        for rel_path, expected_hash in hash_pairs.items():
            p = base_dir / rel_path
            if p.exists():
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != expected_hash:
                    print(f"  HASH MISMATCH: {rel_path}")
                    n_mismatch += 1
                n_checked += 1
        print(f"  SHA-256 verified: {n_checked} files, "
              f"{n_mismatch} mismatches")
        if n_mismatch > 0:
            ok = False
    else:
        print("  WARNING: freeze_manifest.json not found — "
              "skipping SHA-256 check")

    return ok


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: LOAD DEV CORPUS
# ═══════════════════════════════════════════════════════════════════

def phase1_load_dev(base_dir: Path):
    """Load frozen dev corpus features and labels."""
    print("\n[1/7] Loading dev corpus...")

    dev_features = pd.read_csv(base_dir / "features" / "step_features.csv")
    dev_labels = pd.read_csv(base_dir / "labels" / "combined_labels.csv")

    # Normalise step column
    for df in [dev_features, dev_labels]:
        if "step_id" in df.columns and "step_idx" not in df.columns:
            df.rename(columns={"step_id": "step_idx"}, inplace=True)

    dev_features = derive_features(dev_features)

    assert len(dev_features) == 208, \
        f"Dev features: expected 208, got {len(dev_features)}"
    assert len(dev_labels) == 208, \
        f"Dev labels: expected 208, got {len(dev_labels)}"

    n_faulty = int(dev_labels["any_fault"].astype(int).sum())
    n_clean = len(dev_labels) - n_faulty
    n_ds = dev_features["dataset"].nunique()
    print(f"  208 steps, {n_faulty} faulty, {n_clean} clean, "
          f"{n_ds} datasets")

    return dev_features, dev_labels


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: LOAD CP-6 CORPUS
# ═══════════════════════════════════════════════════════════════════

def phase2_load_cp6(cp6_dir: Path):
    """Load CP-6 features and labels produced by Phase C."""
    print("\n[2/7] Loading CP-6 corpus...")

    feat_path = cp6_dir / "cp6_features.csv"
    label_path = cp6_dir / "cp6_labels.csv"

    if not feat_path.exists():
        print(f"  FATAL: {feat_path} not found. Run Phase C first.")
        sys.exit(1)
    if not label_path.exists():
        print(f"  FATAL: {label_path} not found. Run Phase C first.")
        sys.exit(1)

    cp6_features = pd.read_csv(feat_path)
    cp6_labels = pd.read_csv(label_path)

    # Normalise step column
    for df in [cp6_features, cp6_labels]:
        if "step_id" in df.columns and "step_idx" not in df.columns:
            df.rename(columns={"step_id": "step_idx"}, inplace=True)

    assert len(cp6_features) == len(cp6_labels), \
        f"Row mismatch: features={len(cp6_features)}, " \
        f"labels={len(cp6_labels)}"

    cp6_features = derive_features(cp6_features)

    # Verify feature columns
    for feat in ALL_FEATURES:
        assert feat in cp6_features.columns, \
            f"Missing feature column: {feat}"

    # Verify label columns
    for col in ["F1", "F2", "F3", "any_fault"]:
        assert col in cp6_labels.columns, \
            f"Missing label column: {col}"

    n_steps = len(cp6_features)
    n_faulty = int(cp6_labels["any_fault"].astype(int).sum())
    n_clean = n_steps - n_faulty
    n_ds = cp6_features["dataset"].nunique()
    ds_names = sorted(cp6_features["dataset"].unique())

    print(f"  {n_steps} steps, {n_faulty} faulty, {n_clean} clean, "
          f"{n_ds} datasets")
    print(f"  Datasets: {', '.join(ds_names)}")

    # Per-class counts
    for cls in ["F1", "F2", "F3"]:
        n = int(cp6_labels[cls].astype(int).sum())
        print(f"  {cls}: {n}")

    return cp6_features, cp6_labels


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: RETRAIN DETECTORS ON FULL DEV CORPUS
# ═══════════════════════════════════════════════════════════════════

def phase3_retrain_detectors(dev_features, dev_labels):
    """Retrain all detectors on the full 16-dataset dev corpus.

    Rule-based detectors (D1, D2_gate) need no training.
    D2_monitor threshold is tuned on full dev corpus.
    LightGBM detectors are trained on full dev corpus (not LODO).
    """
    print("\n[3/7] Retraining detectors on full dev corpus...")

    models = {}

    # --- D1: rule-based (hash_seen_before) ---
    # No training needed. Applied stateless.
    print("  D1 (rule-based): no training needed")

    # --- D2_gate: rule-based (leaky_before_split) ---
    # No training needed. Applied stateless.
    print("  D2_gate (rule-based): no training needed")

    # --- D2_monitor: tune within_run_z threshold ---
    f2_labels = dev_labels["F2"].astype(int).values
    best_theta, best_f1 = 0.5, 0.0
    for theta in SCORE_THRESHOLDS:
        pred = (dev_features["leaky_before_split"].astype(bool).values
                | (dev_features["within_run_z"].values > theta))
        tp = int((pred & (f2_labels == 1)).sum())
        fp = int((pred & (f2_labels == 0)).sum())
        fn = int((~pred & (f2_labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / max(prec + rec, 1e-10)
              if (prec + rec) > 0 else 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_theta = theta
    models["d2_theta"] = best_theta
    print(f"  D2_monitor: θ={best_theta} (F1={best_f1:.4f} on dev)")

    # --- Prepare feature matrices ---
    X_tier1 = dev_features[TIER1_FEATURES].values.astype(float)
    X_all = dev_features[ALL_FEATURES].values.astype(float)
    y_any = dev_labels["any_fault"].astype(int).values
    y_f3 = dev_labels["F3"].astype(int).values

    # --- D3_gate: LightGBM on Tier 1 → F3 ---
    d3_gate = LGBMClassifier(**LGBM_PARAMS)
    d3_gate.fit(X_tier1, y_f3)
    models["d3_gate"] = d3_gate
    print(f"  D3_gate: trained on {len(X_tier1)} samples, "
          f"{y_f3.sum()} positive")

    # --- D3_monitor: LightGBM on all features → F3 ---
    d3_monitor = LGBMClassifier(**LGBM_PARAMS)
    d3_monitor.fit(X_all, y_f3)
    models["d3_monitor"] = d3_monitor
    print(f"  D3_monitor: trained on {len(X_all)} samples")

    # --- Meta-features for D_any ---
    d1_pred = dev_features["hash_seen_before"].astype(int).values
    d2_gate_pred = dev_features["leaky_before_split"].astype(int).values
    d2_monitor_pred = (
        dev_features["leaky_before_split"].astype(bool).values
        | (dev_features["within_run_z"].values > best_theta)
    ).astype(int)

    # --- D_any_gate: Tier 1 + 2 meta → any_fault ---
    X_any_gate = np.column_stack([X_tier1, d1_pred, d2_gate_pred])
    d_any_gate = LGBMClassifier(**LGBM_PARAMS)
    d_any_gate.fit(X_any_gate, y_any)
    models["d_any_gate"] = d_any_gate
    print(f"  D_any_gate: {X_any_gate.shape[1]} features "
          f"(7 tier1 + 2 meta)")

    # --- D_any_monitor: all + 3 meta → any_fault ---
    X_any_monitor = np.column_stack([
        X_all, d1_pred, d2_gate_pred, d2_monitor_pred
    ])
    d_any_monitor = LGBMClassifier(**LGBM_PARAMS)
    d_any_monitor.fit(X_any_monitor, y_any)
    models["d_any_monitor"] = d_any_monitor
    print(f"  D_any_monitor: {X_any_monitor.shape[1]} features "
          f"(16 all + 3 meta)")

    return models


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: APPLY DETECTORS TO CP-6 DATA
# ═══════════════════════════════════════════════════════════════════

def phase4_apply_detectors(cp6_features, models):
    """Apply retrained detectors to CP-6 feature matrix."""
    print("\n[4/7] Applying detectors to CP-6 data...")

    preds = {}

    # --- D1 (rule-based) ---
    preds["d1"] = cp6_features["hash_seen_before"].astype(int).values

    # --- D2_gate (rule-based) ---
    preds["d2_gate"] = cp6_features["leaky_before_split"].astype(int).values

    # --- D2_monitor ---
    theta = models["d2_theta"]
    preds["d2_monitor"] = (
        cp6_features["leaky_before_split"].astype(bool).values
        | (cp6_features["within_run_z"].values > theta)
    ).astype(int)

    # --- D3_gate ---
    X_tier1 = cp6_features[TIER1_FEATURES].values.astype(float)
    d3_gate_prob = models["d3_gate"].predict_proba(X_tier1)[:, 1]
    preds["d3_gate_prob"] = d3_gate_prob
    preds["d3_gate"] = (d3_gate_prob >= 0.5).astype(int)

    # --- D3_monitor ---
    X_all = cp6_features[ALL_FEATURES].values.astype(float)
    d3_monitor_prob = models["d3_monitor"].predict_proba(X_all)[:, 1]
    preds["d3_monitor_prob"] = d3_monitor_prob
    preds["d3_monitor"] = (d3_monitor_prob >= 0.5).astype(int)

    # --- D_any_gate (Tier 1 + 2 meta) ---
    X_any_gate = np.column_stack([
        X_tier1, preds["d1"], preds["d2_gate"]
    ])
    d_any_gate_prob = models["d_any_gate"].predict_proba(X_any_gate)[:, 1]
    preds["d_any_gate_prob"] = d_any_gate_prob

    # --- D_any_monitor (all + 3 meta) ---
    X_any_monitor = np.column_stack([
        X_all, preds["d1"], preds["d2_gate"], preds["d2_monitor"]
    ])
    d_any_monitor_prob = models["d_any_monitor"].predict_proba(
        X_any_monitor)[:, 1]
    preds["d_any_monitor_prob"] = d_any_monitor_prob

    print(f"  D_any_gate probs: min={d_any_gate_prob.min():.4f}, "
          f"max={d_any_gate_prob.max():.4f}, "
          f"mean={d_any_gate_prob.mean():.4f}")
    print(f"  D_any_monitor probs: min={d_any_monitor_prob.min():.4f}, "
          f"max={d_any_monitor_prob.max():.4f}, "
          f"mean={d_any_monitor_prob.mean():.4f}")

    return preds


# ═══════════════════════════════════════════════════════════════════
# PHASE 5: APPLY FROZEN CRC THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

def phase5_crc_evaluation(preds, cp6_labels, cp6_features):
    """Apply frozen CRC thresholds and compute all metrics."""
    print("\n[5/7] Applying frozen CRC thresholds...")
    print(f"  λ_gate    = {LAMBDA_GATE}")
    print(f"  λ_monitor = {LAMBDA_MONITOR}")

    gate_prob = preds["d_any_gate_prob"]
    monitor_prob = preds["d_any_monitor_prob"]
    y_true = cp6_labels["any_fault"].astype(int).values
    n_faulty = int(y_true.sum())
    n_total = len(y_true)

    # --- Gate tier (Tier 1) ---
    gate_flagged = (gate_prob >= LAMBDA_GATE).astype(int)
    gate_tp = int(((gate_flagged == 1) & (y_true == 1)).sum())
    gate_fp = int(((gate_flagged == 1) & (y_true == 0)).sum())
    gate_fn = int(((gate_flagged == 0) & (y_true == 1)).sum())
    gate_tn = int(((gate_flagged == 0) & (y_true == 0)).sum())

    gate_fsr = gate_fn / max(n_faulty, 1)
    gate_recall = 1.0 - gate_fsr
    gate_precision = gate_tp / max(gate_tp + gate_fp, 1)
    gate_f1 = (2 * gate_precision * gate_recall
               / max(gate_precision + gate_recall, 1e-10)
               if (gate_precision + gate_recall) > 0 else 0.0)

    print(f"  Gate: FSR={gate_fsr:.4f}, recall={gate_recall:.4f}, "
          f"precision={gate_precision:.4f}, F1={gate_f1:.4f}")
    print(f"        TP={gate_tp} FP={gate_fp} "
          f"FN={gate_fn} TN={gate_tn}")

    # --- Monitor tier (Tier 2) ---
    monitor_flagged = (monitor_prob >= LAMBDA_MONITOR).astype(int)
    monitor_tp = int(((monitor_flagged == 1) & (y_true == 1)).sum())
    monitor_fp = int(((monitor_flagged == 1) & (y_true == 0)).sum())
    monitor_fn = int(((monitor_flagged == 0) & (y_true == 1)).sum())
    monitor_tn = int(((monitor_flagged == 0) & (y_true == 0)).sum())

    monitor_fsr = monitor_fn / max(n_faulty, 1)
    monitor_recall = 1.0 - monitor_fsr
    monitor_precision = monitor_tp / max(monitor_tp + monitor_fp, 1)
    monitor_f1 = (2 * monitor_precision * monitor_recall
                  / max(monitor_precision + monitor_recall, 1e-10)
                  if (monitor_precision + monitor_recall) > 0 else 0.0)

    print(f"  Monitor: FSR={monitor_fsr:.4f}, "
          f"recall={monitor_recall:.4f}, "
          f"precision={monitor_precision:.4f}, F1={monitor_f1:.4f}")

    # --- Combined detection (gate + monitor additional) ---
    gate_missed_faulty = (gate_flagged == 0) & (y_true == 1)
    monitor_catches_after_gate = gate_missed_faulty & (monitor_flagged == 1)
    additional_tp = int(monitor_catches_after_gate.sum())
    combined_recall = (gate_tp + additional_tp) / max(n_faulty, 1)

    print(f"  Combined: recall={combined_recall:.4f} "
          f"(gate={gate_tp} + monitor_additional={additional_tp})")

    # --- Tier 1 savings ---
    wall_clock = cp6_features["wall_clock_sec"].values
    total_wc = float(wall_clock.sum())

    tp_wc = float(wall_clock[(gate_flagged == 1) & (y_true == 1)].sum())
    fp_wc = float(wall_clock[(gate_flagged == 1) & (y_true == 0)].sum())
    net_savings = tp_wc - fp_wc
    oracle_wc = float(wall_clock[y_true == 1].sum())

    print(f"  Tier 1 savings: gross={tp_wc/total_wc*100:.2f}%, "
          f"FP_cost={fp_wc/total_wc*100:.2f}%, "
          f"net={net_savings/total_wc*100:.2f}%")
    print(f"  Oracle ceiling: {oracle_wc/total_wc*100:.2f}%")

    # --- Per-class analysis ---
    per_class = {}
    for tier_name, flagged in [("gate", gate_flagged),
                                ("monitor", monitor_flagged)]:
        per_class[tier_name] = []
        for cls in ["F1", "F2", "F3"]:
            cls_mask = cp6_labels[cls].astype(int).values == 1
            n_cls = int(cls_mask.sum())
            if n_cls > 0:
                caught = int((flagged[cls_mask] == 1).sum())
                cls_recall = caught / n_cls
            else:
                caught = 0
                cls_recall = 1.0
            per_class[tier_name].append({
                "fault_class": cls,
                "n_total": n_cls,
                "caught": caught,
                "missed": n_cls - caught,
                "recall": round(cls_recall, 4),
            })

    # --- Per-dataset analysis ---
    datasets = cp6_features["dataset"].values
    unique_ds = sorted(np.unique(datasets))
    per_dataset = []

    for ds in unique_ds:
        mask = datasets == ds
        ds_y = y_true[mask]
        ds_n_faulty = int(ds_y.sum())
        ds_n_total = int(mask.sum())

        # Gate
        ds_gate = gate_flagged[mask]
        ds_gate_fn = int(((ds_gate == 0) & (ds_y == 1)).sum())
        ds_gate_fsr = ds_gate_fn / max(ds_n_faulty, 1)
        ds_gate_tp = int(((ds_gate == 1) & (ds_y == 1)).sum())
        ds_gate_fp = int(((ds_gate == 1) & (ds_y == 0)).sum())

        # Monitor
        ds_monitor = monitor_flagged[mask]
        ds_monitor_fn = int(((ds_monitor == 0) & (ds_y == 1)).sum())
        ds_monitor_fsr = ds_monitor_fn / max(ds_n_faulty, 1)

        # Savings
        ds_wc = wall_clock[mask]
        ds_total_wc = float(ds_wc.sum())
        ds_tp_wc = float(ds_wc[(ds_gate == 1) & (ds_y == 1)].sum())
        ds_fp_wc = float(ds_wc[(ds_gate == 1) & (ds_y == 0)].sum())
        ds_net_savings = (ds_tp_wc - ds_fp_wc) / max(ds_total_wc, 1e-10)

        per_dataset.append({
            "dataset": ds,
            "n_steps": ds_n_total,
            "n_faulty": ds_n_faulty,
            "gate_fsr": round(ds_gate_fsr, 4),
            "gate_tp": ds_gate_tp,
            "gate_fp": ds_gate_fp,
            "gate_fn": ds_gate_fn,
            "monitor_fsr": round(ds_monitor_fsr, 4),
            "net_savings_pct": round(ds_net_savings * 100, 2),
        })

    # Gate FSR ≤ 0.20 per dataset
    n_gate_covered = sum(1 for d in per_dataset
                         if d["gate_fsr"] <= 0.20)
    print(f"  Per-dataset gate FSR ≤ 0.20: "
          f"{n_gate_covered}/{len(per_dataset)}")

    return {
        "gate": {
            "lambda": LAMBDA_GATE,
            "fsr": round(gate_fsr, 4),
            "recall": round(gate_recall, 4),
            "precision": round(gate_precision, 4),
            "f1": round(gate_f1, 4),
            "tp": gate_tp, "fp": gate_fp,
            "fn": gate_fn, "tn": gate_tn,
        },
        "monitor": {
            "lambda": LAMBDA_MONITOR,
            "fsr": round(monitor_fsr, 4),
            "recall": round(monitor_recall, 4),
            "precision": round(monitor_precision, 4),
            "f1": round(monitor_f1, 4),
            "tp": monitor_tp, "fp": monitor_fp,
            "fn": monitor_fn, "tn": monitor_tn,
        },
        "combined_recall": round(combined_recall, 4),
        "additional_monitor_tp": additional_tp,
        "tier1_savings": {
            "gross_savings_pct": round(tp_wc / total_wc * 100, 2),
            "fp_cost_pct": round(fp_wc / total_wc * 100, 2),
            "net_savings_pct": round(net_savings / total_wc * 100, 2),
            "oracle_ceiling_pct": round(oracle_wc / total_wc * 100, 2),
        },
        "per_class": per_class,
        "per_dataset": per_dataset,
        "n_faulty": n_faulty,
        "n_total": n_total,
        "n_datasets": len(unique_ds),
        "n_gate_covered_020": n_gate_covered,
        "gate_flagged": gate_flagged,
        "monitor_flagged": monitor_flagged,
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE 6: INDIVIDUAL DETECTOR PERFORMANCE ON CP-6
# ═══════════════════════════════════════════════════════════════════

def phase6_detector_comparison(preds, cp6_labels):
    """Evaluate each of the 7 detectors independently on CP-6 data."""
    print("\n[6/7] Individual detector performance on CP-6...")

    y_any = cp6_labels["any_fault"].astype(int).values
    y_f1 = cp6_labels["F1"].astype(int).values
    y_f2 = cp6_labels["F2"].astype(int).values
    y_f3 = cp6_labels["F3"].astype(int).values

    rows = []

    def _eval(name, pred, target):
        tp = int(((pred == 1) & (target == 1)).sum())
        fp = int(((pred == 1) & (target == 0)).sum())
        fn = int(((pred == 0) & (target == 1)).sum())
        n_pos = int(target.sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / max(prec + rec, 1e-10)
              if (prec + rec) > 0 else 0.0)
        return {
            "detector": name, "target": "F1" if "D1" in name
                else "F2" if "D2" in name
                else "F3" if "D3" in name else "any_fault",
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
            "n_positive": n_pos,
        }

    rows.append(_eval("D1", preds["d1"], y_f1))
    rows.append(_eval("D2_gate", preds["d2_gate"], y_f2))
    rows.append(_eval("D2_monitor", preds["d2_monitor"], y_f2))
    rows.append(_eval("D3_gate", preds["d3_gate"], y_f3))
    rows.append(_eval("D3_monitor", preds["d3_monitor"], y_f3))
    rows.append(_eval("D_any_gate",
                       (preds["d_any_gate_prob"] >= 0.5).astype(int),
                       y_any))
    rows.append(_eval("D_any_monitor",
                       (preds["d_any_monitor_prob"] >= 0.5).astype(int),
                       y_any))

    for r in rows:
        print(f"  {r['detector']:16s} → {r['target']:10s}: "
              f"P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f}")

    return rows


# ═══════════════════════════════════════════════════════════════════
# PHASE 7: SAVE REPORTS
# ═══════════════════════════════════════════════════════════════════

def phase7_save(cp6_dir, cp6_features, cp6_labels, preds,
                crc_results, detector_results):
    """Save all CP-6 evaluation outputs."""
    print("\n[7/7] Saving outputs...")

    # --- cp6_gate_predictions.csv ---
    gate_df = pd.DataFrame({
        "dataset": cp6_features["dataset"],
        "step_idx": cp6_features["step_idx"],
        "y_true": cp6_labels["any_fault"].astype(int),
        "y_prob": preds["d_any_gate_prob"],
    })
    gate_df.to_csv(cp6_dir / "cp6_gate_predictions.csv", index=False)
    print(f"  Saved cp6_gate_predictions.csv ({len(gate_df)} rows)")

    # --- cp6_monitor_predictions.csv ---
    monitor_df = pd.DataFrame({
        "dataset": cp6_features["dataset"],
        "step_idx": cp6_features["step_idx"],
        "y_true": cp6_labels["any_fault"].astype(int),
        "y_prob": preds["d_any_monitor_prob"],
    })
    monitor_df.to_csv(cp6_dir / "cp6_monitor_predictions.csv",
                      index=False)
    print(f"  Saved cp6_monitor_predictions.csv ({len(monitor_df)} rows)")

    # --- cp6_per_dataset.csv ---
    pd.DataFrame(crc_results["per_dataset"]).to_csv(
        cp6_dir / "cp6_per_dataset.csv", index=False)
    print("  Saved cp6_per_dataset.csv")

    # --- cp6_per_class.csv ---
    pc_rows = []
    for tier in ["gate", "monitor"]:
        for row in crc_results["per_class"][tier]:
            row_copy = dict(row)
            row_copy["tier"] = tier
            pc_rows.append(row_copy)
    pd.DataFrame(pc_rows).to_csv(
        cp6_dir / "cp6_per_class.csv", index=False)
    print("  Saved cp6_per_class.csv")

    # --- cp6_detector_comparison.csv ---
    pd.DataFrame(detector_results).to_csv(
        cp6_dir / "cp6_detector_comparison.csv", index=False)
    print("  Saved cp6_detector_comparison.csv")

    # --- cp6_evaluation.json (main results) ---
    # Remove numpy arrays before JSON serialization
    results_json = {
        "meta": {
            "checkpoint": "CP-6",
            "freeze_protocol": "v2.1 (15 Sep 2026)",
            "alpha": ALPHA,
            "lambda_gate": LAMBDA_GATE,
            "lambda_monitor": LAMBDA_MONITOR,
            "n_cp6_steps": crc_results["n_total"],
            "n_cp6_faulty": crc_results["n_faulty"],
            "n_cp6_datasets": crc_results["n_datasets"],
            "lgbm_params": LGBM_PARAMS,
        },
        "crc_evaluation": {
            "gate": crc_results["gate"],
            "monitor": crc_results["monitor"],
        },
        "combined_recall": crc_results["combined_recall"],
        "additional_monitor_tp": crc_results["additional_monitor_tp"],
        "tier1_savings": crc_results["tier1_savings"],
        "per_class": crc_results["per_class"],
        "per_dataset": crc_results["per_dataset"],
        "n_gate_covered_020": crc_results["n_gate_covered_020"],
        "detector_comparison": detector_results,
        "success_criteria": {
            "gate_fsr_le_010": crc_results["gate"]["fsr"] <= 0.10,
            "monitor_fsr_le_010": crc_results["monitor"]["fsr"] <= 0.10,
            "gate_f1_ge_070": crc_results["gate"]["f1"] >= 0.70,
            "tier1_net_savings_gt_0": (
                crc_results["tier1_savings"]["net_savings_pct"] > 0),
            "stretch_gate_coverage_020": (
                crc_results["n_gate_covered_020"]
                >= min(4, crc_results["n_datasets"])),
        },
    }

    with open(cp6_dir / "cp6_evaluation.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print("  Saved cp6_evaluation.json")

    # --- Summary report ---
    _write_report(results_json, cp6_dir / "cp6_report.md")
    print("  Saved cp6_report.md")


def _write_report(R, path: Path):
    """Write CP-6 evaluation report."""
    crc = R["crc_evaluation"]
    sav = R["tier1_savings"]
    meta = R["meta"]
    sc = R["success_criteria"]

    lines = [
        "# CP-6 External Validation Report",
        "",
        f"**Freeze protocol**: {meta['freeze_protocol']}",
        f"**CP-6 corpus**: {meta['n_cp6_steps']} steps, "
        f"{meta['n_cp6_faulty']} faulty, "
        f"{meta['n_cp6_datasets']} datasets",
        f"**CRC α**: {meta['alpha']}",
        "",
        "## 1. Primary Results",
        "",
        "| Metric | Gate (Tier 1) | Monitor (Tier 2) |",
        "|--------|---------------|-------------------|",
        f"| λ (frozen) | {crc['gate']['lambda']} | "
        f"{crc['monitor']['lambda']} |",
        f"| FSR | {crc['gate']['fsr']:.4f} | "
        f"{crc['monitor']['fsr']:.4f} |",
        f"| Recall | {crc['gate']['recall']:.4f} | "
        f"{crc['monitor']['recall']:.4f} |",
        f"| Precision | {crc['gate']['precision']:.4f} | "
        f"{crc['monitor']['precision']:.4f} |",
        f"| F1 | {crc['gate']['f1']:.4f} | "
        f"{crc['monitor']['f1']:.4f} |",
        "",
        f"**Combined recall**: {R['combined_recall']:.4f} "
        f"(gate + {R['additional_monitor_tp']} monitor additional)",
        "",
        "## 2. Tier 1 Savings",
        "",
        f"- Gross savings: {sav['gross_savings_pct']:.2f}%",
        f"- FP cost: {sav['fp_cost_pct']:.2f}%",
        f"- **Net savings: {sav['net_savings_pct']:.2f}%**",
        f"- Oracle ceiling: {sav['oracle_ceiling_pct']:.2f}%",
        "",
        "## 3. Success Criteria",
        "",
        "| Criterion | Required | Actual | Pass |",
        "|-----------|----------|--------|------|",
        f"| Gate FSR ≤ 0.10 | Yes | {crc['gate']['fsr']:.4f} | "
        f"{'✓' if sc['gate_fsr_le_010'] else '✗'} |",
        f"| Monitor FSR ≤ 0.10 | Yes | {crc['monitor']['fsr']:.4f} | "
        f"{'✓' if sc['monitor_fsr_le_010'] else '✗'} |",
        f"| Gate F1 ≥ 0.70 | Yes | {crc['gate']['f1']:.4f} | "
        f"{'✓' if sc['gate_f1_ge_070'] else '✗'} |",
        f"| Net savings > 0% | Yes | {sav['net_savings_pct']:.2f}% | "
        f"{'✓' if sc['tier1_net_savings_gt_0'] else '✗'} |",
        f"| ≥4 datasets gate FSR ≤ 0.20 | Stretch | "
        f"{R['n_gate_covered_020']}/{meta['n_cp6_datasets']} | "
        f"{'✓' if sc['stretch_gate_coverage_020'] else '—'} |",
        "",
        "## 4. Per-Dataset Breakdown",
        "",
        "| Dataset | Steps | Faulty | Gate FSR | Monitor FSR | "
        "Net Savings |",
        "|---------|-------|--------|----------|-------------|"
        "------------|",
    ]

    for d in R["per_dataset"]:
        lines.append(
            f"| {d['dataset']} | {d['n_steps']} | {d['n_faulty']} | "
            f"{d['gate_fsr']:.4f} | {d['monitor_fsr']:.4f} | "
            f"{d['net_savings_pct']:.2f}% |")

    lines += [
        "",
        "## 5. Per-Class Analysis at α=0.10",
        "",
        "| Tier | Class | Caught | Total | Recall |",
        "|------|-------|--------|-------|--------|",
    ]
    for tier in ["gate", "monitor"]:
        for row in R["per_class"][tier]:
            lines.append(
                f"| {tier.title()} | {row['fault_class']} | "
                f"{row['caught']} | {row['n_total']} | "
                f"{row['recall']:.4f} |")

    lines += [
        "",
        "## 6. Individual Detector Comparison",
        "",
        "| Detector | Target | P | R | F1 |",
        "|----------|--------|---|---|-----|",
    ]
    for d in R["detector_comparison"]:
        lines.append(
            f"| {d['detector']} | {d['target']} | "
            f"{d['precision']:.4f} | {d['recall']:.4f} | "
            f"{d['f1']:.4f} |")

    n_pass = sum(1 for v in sc.values() if v is True)
    n_total = len(sc)
    lines += [
        "",
        f"---",
        f"",
        f"**Overall: {n_pass}/{n_total} success criteria met.**",
    ]

    path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CP-6: External Validation Evaluation Pipeline")
    parser.add_argument("--output-dir", required=True,
                        help="Base dir (has features/, labels/, scripts/)")
    args = parser.parse_args()

    base = Path(args.output_dir)
    cp6_dir = base / "cp6_results"
    cp6_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CP-6: External Validation — Evaluation Pipeline")
    print("=" * 60)
    print(f"Frozen CRC: λ_gate={LAMBDA_GATE}, λ_monitor={LAMBDA_MONITOR}")
    print(f"Target α={ALPHA}")

    # Phase 0
    if not phase0_preflight(base):
        print("\nFATAL: Pre-flight failed. Fix issues above.")
        sys.exit(1)

    # Phase 1
    dev_features, dev_labels = phase1_load_dev(base)

    # Phase 2
    cp6_features, cp6_labels = phase2_load_cp6(cp6_dir)

    # Phase 3
    models = phase3_retrain_detectors(dev_features, dev_labels)

    # Phase 4
    preds = phase4_apply_detectors(cp6_features, models)

    # Phase 5
    crc_results = phase5_crc_evaluation(preds, cp6_labels, cp6_features)

    # Phase 6
    detector_results = phase6_detector_comparison(preds, cp6_labels)

    # Phase 7
    phase7_save(cp6_dir, cp6_features, cp6_labels, preds,
                crc_results, detector_results)

    print("\n" + "=" * 60)
    print("CP-6 EVALUATION COMPLETE")
    print("=" * 60)
    sc = {
        "gate_fsr": crc_results["gate"]["fsr"] <= 0.10,
        "monitor_fsr": crc_results["monitor"]["fsr"] <= 0.10,
        "gate_f1": crc_results["gate"]["f1"] >= 0.70,
        "net_savings": crc_results["tier1_savings"]["net_savings_pct"] > 0,
    }
    for name, ok in sc.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
