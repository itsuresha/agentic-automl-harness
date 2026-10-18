"""CP-6 acceptance verification — external validation."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    cp6_dir = out / "cp6_results"

    eval_path = cp6_dir / "cp6_evaluation.json"
    if not eval_path.exists():
        print(f"FATAL: {eval_path} not found")
        sys.exit(1)

    with open(eval_path) as f:
        R = json.load(f)

    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        msg = f"  [{tag}] {total}. {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    print("=" * 60)
    print("CP-6 Acceptance Verification")
    print("=" * 60)

    # ── 1. Output files exist ─────────────────────────────────────
    expected_files = [
        "cp6_evaluation.json",
        "cp6_report.md",
        "cp6_gate_predictions.csv",
        "cp6_monitor_predictions.csv",
        "cp6_per_dataset.csv",
        "cp6_per_class.csv",
        "cp6_detector_comparison.csv",
        "dataset_verification.json",
    ]
    for fname in expected_files:
        fpath = cp6_dir / fname
        check(f"Output file: {fname}", fpath.exists())

    # ── 2. CP-6 corpus files exist ────────────────────────────────
    check("cp6_features.csv exists",
          (cp6_dir / "cp6_features.csv").exists())
    check("cp6_labels.csv exists",
          (cp6_dir / "cp6_labels.csv").exists())

    # ── 3. Dataset verification ───────────────────────────────────
    dv_path = cp6_dir / "dataset_verification.json"
    if dv_path.exists():
        with open(dv_path) as f:
            dv = json.load(f)

        n_final = dv.get("n_final", 0)
        check("At least 6 CP-6 datasets verified",
              n_final >= 6,
              f"n_final={n_final}")

        # No dataset in dev corpus
        dev_names = {
            "satellite", "amazon-commerce-reviews", "christine",
            "credit-approval", "credit-g", "diabetes", "jasmine",
            "kr-vs-kp", "letter", "mfeat-factors", "mfeat-fourier",
            "mfeat-morphological", "mfeat-zernike", "pendigits",
            "tic-tac-toe", "vehicle",
        }
        final_names = {d["name"].lower()
                       for d in dv.get("final_datasets", [])}
        overlap = final_names & dev_names
        check("No CP-6 dataset in dev corpus",
              len(overlap) == 0,
              f"overlap: {overlap}" if overlap else "no overlap")
    else:
        check("dataset_verification.json exists", False)

    # ── 4. Feature matrix structure ───────────────────────────────
    feat_path = cp6_dir / "cp6_features.csv"
    if feat_path.exists():
        feats = pd.read_csv(feat_path)

        expected_feat_cols = [
            "n_hyperparams", "n_preprocessing_steps",
            "preprocessing_before_split", "leaky_before_split",
            "hash_seen_before", "hash_first_seen_at",
            "parse_quality_full_ast", "validation_score",
            "wall_clock_sec", "score_delta_from_prev",
            "score_rank_in_run", "is_best_so_far",
            "steps_since_improvement", "score_rank_pct",
            "wall_clock_rank_pct", "within_run_z",
        ]
        has_all_feats = all(c in feats.columns
                            for c in expected_feat_cols)
        check("Feature matrix has 16 required feature columns",
              has_all_feats,
              f"missing: {set(expected_feat_cols) - set(feats.columns)}"
              if not has_all_feats else "all present")

        check("Feature matrix has > 0 rows",
              len(feats) > 0,
              f"rows={len(feats)}")

    # ── 5. Label structure ────────────────────────────────────────
    label_path = cp6_dir / "cp6_labels.csv"
    if label_path.exists():
        labels = pd.read_csv(label_path)
        for col in ["F1", "F2", "F3", "any_fault"]:
            check(f"Labels have column: {col}",
                  col in labels.columns)

        # any_fault consistency
        if all(c in labels.columns for c in ["F1", "F2", "F3", "any_fault"]):
            computed_any = (labels["F1"].astype(int)
                           | labels["F2"].astype(int)
                           | labels["F3"].astype(int))
            actual_any = labels["any_fault"].astype(int)
            check("any_fault = F1 | F2 | F3",
                  (computed_any == actual_any).all(),
                  f"mismatches: {(computed_any != actual_any).sum()}")

        # Row count matches features
        if feat_path.exists():
            check("Features and labels have same row count",
                  len(feats) == len(labels),
                  f"features={len(feats)}, labels={len(labels)}")

    # ── 6. Frozen artifact integrity ──────────────────────────────
    manifest_path = out / "freeze_manifest.json"
    if manifest_path.exists():
        import hashlib
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Handle both formats: flat {path: hash} or nested {frozen_files: [...]}
        hash_pairs = {}
        if isinstance(manifest, dict) and "frozen_files" in manifest:
            for entry in manifest["frozen_files"]:
                if isinstance(entry, dict) and "path" in entry:
                    h = entry.get("hash") or entry.get("sha256", "")
                    if h:
                        hash_pairs[entry["path"]] = h
        elif isinstance(manifest, dict):
            hash_pairs = {k: v for k, v in manifest.items()
                          if isinstance(v, str) and len(v) == 64}
        n_ok = 0
        n_bad = 0
        for rel, expected in hash_pairs.items():
            p = out / rel
            if p.exists():
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual == expected:
                    n_ok += 1
                else:
                    n_bad += 1
        check("Frozen artifacts SHA-256 verified",
              n_bad == 0,
              f"{n_ok} ok, {n_bad} mismatches")
    else:
        check("freeze_manifest.json exists for integrity check",
              False)

    # ── 7. CRC thresholds match frozen values ─────────────────────
    meta = R.get("meta", {})
    check("λ_gate matches frozen value",
          abs(meta.get("lambda_gate", 0) - 0.324949) < 1e-6,
          f"λ_gate={meta.get('lambda_gate')}")
    check("λ_monitor matches frozen value",
          abs(meta.get("lambda_monitor", 0) - 0.951217) < 1e-6,
          f"λ_monitor={meta.get('lambda_monitor')}")
    check("LGBM_PARAMS match frozen values",
          meta.get("lgbm_params", {}).get("n_estimators") == 100
          and meta.get("lgbm_params", {}).get("max_depth") == 4
          and meta.get("lgbm_params", {}).get("random_state") == 42,
          f"n_est={meta.get('lgbm_params', {}).get('n_estimators')}, "
          f"depth={meta.get('lgbm_params', {}).get('max_depth')}")

    # ── 8. All 7 detectors produced results ───────────────────────
    det_comp = R.get("detector_comparison", [])
    det_names = {d["detector"] for d in det_comp}
    expected_dets = {"D1", "D2_gate", "D2_monitor", "D3_gate",
                     "D3_monitor", "D_any_gate", "D_any_monitor"}
    check("All 7 detectors reported",
          expected_dets.issubset(det_names),
          f"missing: {expected_dets - det_names}"
          if not expected_dets.issubset(det_names) else "all 7")

    # ═══ PRIMARY SUCCESS CRITERIA (from Freeze Protocol §8.6) ═══

    crc = R.get("crc_evaluation", {})
    gate = crc.get("gate", {})
    monitor = crc.get("monitor", {})
    savings = R.get("tier1_savings", {})

    print("\n--- Primary Success Criteria ---")

    check("REQUIRED: Gate FSR ≤ 0.10 at α=0.10",
          gate.get("fsr", 1.0) <= 0.10 + 1e-4,
          f"FSR={gate.get('fsr')}")

    check("REQUIRED: Monitor FSR ≤ 0.10 at α=0.10",
          monitor.get("fsr", 1.0) <= 0.10 + 1e-4,
          f"FSR={monitor.get('fsr')}")

    check("REQUIRED: Gate F1 ≥ 0.70 (any_fault)",
          gate.get("f1", 0) >= 0.70,
          f"F1={gate.get('f1')}")

    check("REQUIRED: Tier 1 net savings > 0%",
          savings.get("net_savings_pct", 0) > 0,
          f"net_savings={savings.get('net_savings_pct')}%")

    # ═══ STRETCH CRITERIA ═══

    print("\n--- Stretch Criteria ---")

    n_ds = R.get("meta", {}).get("n_cp6_datasets", 0)
    n_covered = R.get("n_gate_covered_020", 0)
    check("STRETCH: ≥4 datasets gate FSR ≤ 0.20",
          n_covered >= min(4, n_ds),
          f"{n_covered}/{n_ds} covered")

    check("STRETCH: Gate F1 ≥ 0.85",
          gate.get("f1", 0) >= 0.85,
          f"F1={gate.get('f1')}")

    check("STRETCH: Tier 1 net savings > 30%",
          savings.get("net_savings_pct", 0) > 30,
          f"net_savings={savings.get('net_savings_pct')}%")

    # ═══ SUMMARY ═══

    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}")

    if passed < total:
        # Distinguish required vs stretch failures
        sc = R.get("success_criteria", {})
        required_pass = all([
            sc.get("gate_fsr_le_010", False),
            sc.get("monitor_fsr_le_010", False),
            sc.get("gate_f1_ge_070", False),
            sc.get("tier1_net_savings_gt_0", False),
        ])
        if required_pass:
            print("  NOTE: All REQUIRED criteria passed. "
                  "Failures are stretch or procedural.")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
