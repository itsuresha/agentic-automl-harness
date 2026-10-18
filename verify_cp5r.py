"""CP-5R acceptance verification — tiered conformal risk control."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    cp5_dir = out / "cp5_results"

    eval_path = cp5_dir / "cp5r_evaluation.json"
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
    print("CP-5R Acceptance Verification")
    print("=" * 60)

    # ── Output files ──────────────────────────────────────────────
    expected_files = [
        "cp5r_evaluation.json",
        "cp5r_report.md",
        "conformal_thresholds.csv",
        "lodo_cross_conformal.csv",
        "per_class_analysis.csv",
        "gated_decisions.csv",
        "fsr_efficiency_curves.png",
    ]
    for fname in expected_files:
        fpath = cp5_dir / fname
        check(f"Output file: {fname}", fpath.exists())

    # ── gated_decisions ───────────────────────────────────────────
    gd_path = cp5_dir / "gated_decisions.csv"
    if gd_path.exists():
        gd = pd.read_csv(gd_path)
        check("gated_decisions has 208 rows",
              len(gd) == 208,
              f"got {len(gd)}")

        required_cols = {
            "dataset", "step_idx", "y_true",
            "gate_y_prob", "monitor_y_prob",
            "gate_crc_flag", "monitor_crc_flag",
            "gate_naive_flag", "monitor_naive_flag",
            "gate_default_flag", "monitor_default_flag",
        }
        has_cols = required_cols.issubset(set(gd.columns))
        check("gated_decisions has required columns",
              has_cols,
              f"missing: {required_cols - set(gd.columns)}" if not has_cols
              else f"{len(gd.columns)} columns")

    # ── per_class_analysis ────────────────────────────────────────
    pc_path = cp5_dir / "per_class_analysis.csv"
    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        has_classes = set(pc["fault_class"].unique()) >= {"F1", "F2", "F3"}
        has_tiers = set(pc["tier"].unique()) >= {"gate", "monitor"}
        check("per_class has F1, F2, F3 for both tiers",
              has_classes and has_tiers,
              f"classes={sorted(pc['fault_class'].unique())}, "
              f"tiers={sorted(pc['tier'].unique())}")
        check("per_class has 6 rows (3 classes × 2 tiers)",
              len(pc) == 6,
              f"got {len(pc)}")

    # ── conformal_thresholds ──────────────────────────────────────
    ct_path = cp5_dir / "conformal_thresholds.csv"
    if ct_path.exists():
        ct = pd.read_csv(ct_path)
        check("conformal_thresholds has 12 rows (6 alphas × 2 tiers)",
              len(ct) == 12,
              f"got {len(ct)}")

    # ── lodo_cross_conformal ──────────────────────────────────────
    lodo_path = cp5_dir / "lodo_cross_conformal.csv"
    if lodo_path.exists():
        lodo = pd.read_csv(lodo_path)
        check("lodo_cross_conformal has 192 rows (16×6×2)",
              len(lodo) == 192,
              f"got {len(lodo)}")

    # ── CRC FSR guarantees (gate tier) ────────────────────────────
    gate_sweep = R.get("sweep", {}).get("gate", [])

    for alpha_check in [0.05, 0.10, 0.20]:
        row = next((r for r in gate_sweep
                     if abs(r["alpha"] - alpha_check) < 0.001), None)
        if row:
            check(f"Gate CRC FSR ≤ {alpha_check:.2f} at α={alpha_check:.2f}",
                  row["crc_fsr"] <= alpha_check + 0.001,
                  f"FSR={row['crc_fsr']}")

    # CRC FSR guarantees (monitor tier)
    monitor_sweep = R.get("sweep", {}).get("monitor", [])

    for alpha_check in [0.05, 0.10, 0.20]:
        row = next((r for r in monitor_sweep
                     if abs(r["alpha"] - alpha_check) < 0.001), None)
        if row:
            check(f"Monitor CRC FSR ≤ {alpha_check:.2f} at α={alpha_check:.2f}",
                  row["crc_fsr"] <= alpha_check + 0.001,
                  f"FSR={row['crc_fsr']}")

    # ── Recall checks ─────────────────────────────────────────────
    gate_a10 = next((r for r in gate_sweep
                      if abs(r["alpha"] - 0.10) < 0.001), None)
    monitor_a10 = next((r for r in monitor_sweep
                         if abs(r["alpha"] - 0.10) < 0.001), None)

    if gate_a10:
        check("Gate recall ≥ 0.85 at α=0.10",
              gate_a10["crc_recall"] >= 0.85,
              f"recall={gate_a10['crc_recall']}")

    if monitor_a10:
        check("Monitor recall ≥ 0.85 at α=0.10",
              monitor_a10["crc_recall"] >= 0.85,
              f"recall={monitor_a10['crc_recall']}")

    # ── Net savings > 0% at α=0.10 ───────────────────────────────
    cost = R.get("cost_alpha_0.10", {})
    t1 = cost.get("tier1_gate", {})
    check("Tier 1 net savings > 0% at α=0.10",
          t1.get("net_savings_pct", 0) > 0,
          f"net={t1.get('net_savings_pct')}%")

    # ── CRC ≤ naive at all key alphas ─────────────────────────────
    gate_mono = all(r["crc_threshold"] <= r["naive_threshold"] + 1e-6
                    for r in gate_sweep)
    monitor_mono = all(r["crc_threshold"] <= r["naive_threshold"] + 1e-6
                       for r in monitor_sweep)
    check("Gate CRC threshold ≤ naive at all alphas", gate_mono)
    check("Monitor CRC threshold ≤ naive at all alphas", monitor_mono)

    # ── Threshold monotonicity ────────────────────────────────────
    gate_thresholds = [r["crc_threshold"] for r in gate_sweep]
    monitor_thresholds = [r["crc_threshold"] for r in monitor_sweep]

    gate_nondec = all(gate_thresholds[i] <= gate_thresholds[i+1] + 1e-6
                      for i in range(len(gate_thresholds)-1))
    monitor_nondec = all(monitor_thresholds[i] <= monitor_thresholds[i+1] + 1e-6
                         for i in range(len(monitor_thresholds)-1))
    check("Gate CRC thresholds non-decreasing", gate_nondec)
    check("Monitor CRC thresholds non-decreasing", monitor_nondec)

    # ── FSR monotonicity ──────────────────────────────────────────
    gate_fsrs = [r["crc_fsr"] for r in gate_sweep]
    monitor_fsrs = [r["crc_fsr"] for r in monitor_sweep]

    gate_fsr_nondec = all(gate_fsrs[i] <= gate_fsrs[i+1] + 0.001
                          for i in range(len(gate_fsrs)-1))
    monitor_fsr_nondec = all(monitor_fsrs[i] <= monitor_fsrs[i+1] + 0.001
                             for i in range(len(monitor_fsrs)-1))
    check("Gate FSR non-decreasing across alphas", gate_fsr_nondec)
    check("Monitor FSR non-decreasing across alphas", monitor_fsr_nondec)

    # ── CRC threshold < 0.5 at α=0.10 ────────────────────────────
    if gate_a10:
        check("Gate CRC threshold < 0.5 at α=0.10",
              gate_a10["crc_threshold"] < 0.5,
              f"λ={gate_a10['crc_threshold']}")
    if monitor_a10:
        check("Monitor CRC threshold < 0.5 at α=0.10",
              monitor_a10["crc_threshold"] < 0.5,
              f"λ={monitor_a10['crc_threshold']}")

    # ── CRC FSR < default FSR at α=0.10 ──────────────────────────
    gate_default = R.get("default_baseline", {}).get("gate", {})
    monitor_default = R.get("default_baseline", {}).get("monitor", {})

    if gate_a10 and gate_default:
        check("Gate CRC FSR ≤ default FSR at α=0.10",
              gate_a10["crc_fsr"] <= gate_default.get("fsr", 1.0) + 0.001,
              f"CRC={gate_a10['crc_fsr']}, default={gate_default.get('fsr')}")

    if monitor_a10 and monitor_default:
        check("Monitor CRC FSR ≤ default FSR at α=0.10",
              monitor_a10["crc_fsr"] <= monitor_default.get("fsr", 1.0) + 0.001,
              f"CRC={monitor_a10['crc_fsr']}, default={monitor_default.get('fsr')}")

    # ── LODO coverage ─────────────────────────────────────────────
    gate_lodo = R.get("lodo", {}).get("gate", [])
    monitor_lodo = R.get("lodo", {}).get("monitor", [])

    for tier_name, lodo in [("Gate", gate_lodo), ("Monitor", monitor_lodo)]:
        for alpha, min_covered in [(0.10, 8), (0.20, 8)]:
            rows = [r for r in lodo if abs(r["alpha"] - alpha) < 0.001]
            n_covered = sum(1 for r in rows if r["covered"])
            check(f"{tier_name} LODO ≥ {min_covered}/16 covered at α={alpha:.2f}",
                  n_covered >= min_covered,
                  f"covered={n_covered}/16")

    # ── Monitor recall ≥ gate recall at α=0.10 ───────────────────
    if gate_a10 and monitor_a10:
        check("Monitor recall ≥ gate recall at α=0.10",
              monitor_a10["crc_recall"] >= gate_a10["crc_recall"] - 0.001,
              f"monitor={monitor_a10['crc_recall']}, "
              f"gate={gate_a10['crc_recall']}")

    # ── Tier 2 additional detections reported ─────────────────────
    t2 = cost.get("tier2_monitor_additional", {})
    check("Tier 2 additional detections reported",
          "additional_true_positives" in t2,
          f"additional_tp={t2.get('additional_true_positives')}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
