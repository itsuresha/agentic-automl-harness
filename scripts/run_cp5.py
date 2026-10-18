"""CP-5R runner: tiered conformal risk control with LODO cross-conformal."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from conformal_gate import (
    KEY_ALPHAS, FINE_ALPHAS,
    crc_threshold, naive_quantile_threshold,
    evaluate_gate, sweep_alpha,
    lodo_cross_conformal, per_class_analysis,
    tiered_cost_analysis, default_baseline,
)


def main():
    parser = argparse.ArgumentParser(
        description="CP-5R: Tiered Conformal Risk Control")
    parser.add_argument("--output-dir", required=True,
                        help="CP-3R output dir (has features/, labels/, cp4_results/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    cp4_dir = out / "cp4_results"
    cp5_dir = out / "cp5_results"
    cp5_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 60)
    print("CP-5R: Tiered Conformal Risk Control Gating Layer")
    print("=" * 60)

    # ── Phase 1: Load data ─────────────────────────────────────────
    print("\n[1/8] Loading OOF predictions and labels...")

    gate_oof = pd.read_csv(cp4_dir / "oof_predictions_D_any_gate.csv")
    monitor_oof = pd.read_csv(cp4_dir / "oof_predictions_D_any_monitor.csv")

    assert len(gate_oof) == 208, f"Gate OOF: expected 208, got {len(gate_oof)}"
    assert len(monitor_oof) == 208, f"Monitor OOF: expected 208, got {len(monitor_oof)}"

    # Verify y_true matches between tiers
    assert (gate_oof["y_true"].values == monitor_oof["y_true"].values).all(), \
        "y_true mismatch between gate and monitor OOF files"

    y_true = gate_oof["y_true"].values.astype(int)
    gate_probs = gate_oof["y_prob"].values
    monitor_probs = monitor_oof["y_prob"].values
    datasets = gate_oof["dataset"].values

    n_f = int(y_true.sum())
    n_c = len(y_true) - n_f
    print(f"  208 steps, {n_f} faulty, {n_c} clean, "
          f"{len(np.unique(datasets))} datasets")

    # Load labels for per-class analysis
    labels = pd.read_csv(out / "labels" / "combined_labels.csv")
    # Normalise step column
    step_col = "step_idx" if "step_idx" in labels.columns else "step_id"
    if step_col != "step_idx":
        labels = labels.rename(columns={step_col: "step_idx"})

    # Merge on dataset + step_idx to align
    merge_df = gate_oof[["dataset", "step_idx"]].copy()
    merge_df = merge_df.merge(
        labels[["dataset", "step_idx", "F1", "F2", "F3"]],
        on=["dataset", "step_idx"], how="left")

    # Convert booleans
    for col in ["F1", "F2", "F3"]:
        merge_df[col] = (
            merge_df[col].astype(str)
            .map({"True": 1, "False": 0, "1": 1, "0": 0})
            .fillna(0).astype(int)
        )

    # Load wall_clock
    features = pd.read_csv(out / "features" / "step_features.csv")
    feat_step_col = ("step_idx" if "step_idx" in features.columns
                     else "step_id")
    if feat_step_col != "step_idx":
        features = features.rename(columns={feat_step_col: "step_idx"})

    wc_df = gate_oof[["dataset", "step_idx"]].merge(
        features[["dataset", "step_idx", "wall_clock_sec"]],
        on=["dataset", "step_idx"], how="left")
    wall_clock = wc_df["wall_clock_sec"].values

    results = {
        "meta": {
            "n_steps": 208,
            "n_faulty": n_f,
            "n_clean": n_c,
            "n_datasets": int(len(np.unique(datasets))),
            "tiers": ["gate", "monitor"],
            "key_alphas": KEY_ALPHAS,
        },
    }

    # ── Phase 2: Default baselines ─────────────────────────────────
    print("\n[2/8] Default θ=0.5 baselines...")

    gate_default = default_baseline(y_true, gate_probs, 0.5)
    monitor_default = default_baseline(y_true, monitor_probs, 0.5)
    results["default_baseline"] = {
        "gate": gate_default,
        "monitor": monitor_default,
    }
    print(f"  Gate default: FSR={gate_default['fsr']:.4f}, "
          f"recall={gate_default['recall']:.4f}")
    print(f"  Monitor default: FSR={monitor_default['fsr']:.4f}, "
          f"recall={monitor_default['recall']:.4f}")

    # ── Phase 3: CRC sweep (key alphas) ────────────────────────────
    print("\n[3/8] CRC alpha sweep (key alphas, both tiers)...")

    gate_sweep = sweep_alpha(y_true, gate_probs, KEY_ALPHAS, "gate")
    monitor_sweep = sweep_alpha(y_true, monitor_probs, KEY_ALPHAS, "monitor")
    all_sweep = gate_sweep + monitor_sweep
    results["sweep"] = {
        "gate": gate_sweep,
        "monitor": monitor_sweep,
    }

    for tier_name, sweep in [("Gate", gate_sweep), ("Monitor", monitor_sweep)]:
        for row in sweep:
            if row["alpha"] in [0.05, 0.10, 0.20]:
                print(f"  {tier_name} α={row['alpha']:.2f}: "
                      f"λ={row['crc_threshold']:.4f}, "
                      f"FSR={row['crc_fsr']:.4f}, "
                      f"recall={row['crc_recall']:.4f}, "
                      f"F1={row['crc_f1']:.4f}")

    # ── Phase 4: Fine sweep (for plotting) ─────────────────────────
    print("\n[4/8] Fine alpha sweep (plotting)...")

    gate_fine = sweep_alpha(y_true, gate_probs, FINE_ALPHAS, "gate")
    monitor_fine = sweep_alpha(y_true, monitor_probs, FINE_ALPHAS, "monitor")

    # ── Phase 5: LODO cross-conformal ──────────────────────────────
    print("\n[5/8] LODO cross-conformal evaluation...")

    gate_lodo = lodo_cross_conformal(
        datasets, y_true, gate_probs, KEY_ALPHAS, "gate")
    monitor_lodo = lodo_cross_conformal(
        datasets, y_true, monitor_probs, KEY_ALPHAS, "monitor")
    all_lodo = gate_lodo + monitor_lodo
    results["lodo"] = {
        "gate": gate_lodo,
        "monitor": monitor_lodo,
    }

    # LODO coverage summary
    for tier_name, lodo in [("Gate", gate_lodo), ("Monitor", monitor_lodo)]:
        for alpha in [0.10, 0.20]:
            rows = [r for r in lodo if r["alpha"] == alpha]
            n_covered = sum(1 for r in rows if r["covered"])
            print(f"  {tier_name} LODO α={alpha:.2f}: "
                  f"{n_covered}/16 covered")

    # ── Phase 6: Per-class analysis ────────────────────────────────
    print("\n[6/8] Per-class analysis at α=0.10...")

    # Get CRC thresholds at α=0.10
    gate_alpha10 = [r for r in gate_sweep if r["alpha"] == 0.10][0]
    monitor_alpha10 = [r for r in monitor_sweep if r["alpha"] == 0.10][0]

    gate_eval_10 = evaluate_gate(
        y_true, gate_probs, gate_alpha10["crc_threshold"])
    monitor_eval_10 = evaluate_gate(
        y_true, monitor_probs, monitor_alpha10["crc_threshold"])

    gate_per_class = per_class_analysis(
        merge_df, gate_eval_10["flagged"], "gate")
    monitor_per_class = per_class_analysis(
        merge_df, monitor_eval_10["flagged"], "monitor")
    all_per_class = gate_per_class + monitor_per_class
    results["per_class_alpha_0.10"] = {
        "gate": gate_per_class,
        "monitor": monitor_per_class,
    }

    for tier_name, pc in [("Gate", gate_per_class),
                          ("Monitor", monitor_per_class)]:
        for row in pc:
            print(f"  {tier_name} {row['fault_class']}: "
                  f"caught={row['caught']}/{row['n_total']} "
                  f"(recall={row['recall']:.4f})")

    # ── Phase 7: Tiered cost analysis ──────────────────────────────
    print("\n[7/8] Tiered cost analysis at α=0.10...")

    # CRC-calibrated cost analysis
    cost_crc = tiered_cost_analysis(
        y_true, wall_clock,
        gate_eval_10["flagged"],
        monitor_eval_10["flagged"])
    results["cost_alpha_0.10"] = cost_crc

    t1 = cost_crc["tier1_gate"]
    t2 = cost_crc["tier2_monitor_additional"]
    comb = cost_crc["combined_detection"]
    print(f"  Tier 1 net savings: {t1['net_savings_pct']:.2f}% "
          f"(gross {t1['gross_savings_pct']:.2f}%, "
          f"FP cost {t1['fp_cost_pct']:.2f}%)")
    print(f"  Tier 2 additional detections: "
          f"{t2['additional_true_positives']} steps")
    print(f"  Combined recall: {comb['combined_recall']:.4f}")
    print(f"  Oracle ceiling: {comb['oracle_ceiling_pct']:.2f}%")

    # Default θ=0.5 cost analysis for comparison
    gate_default_eval = evaluate_gate(y_true, gate_probs, 0.5)
    monitor_default_eval = evaluate_gate(y_true, monitor_probs, 0.5)
    cost_default = tiered_cost_analysis(
        y_true, wall_clock,
        gate_default_eval["flagged"],
        monitor_default_eval["flagged"])
    results["cost_default_0.5"] = cost_default

    # ── Phase 8: Save + plot ───────────────────────────────────────
    print("\n[8/8] Saving outputs...")

    # conformal_thresholds.csv — key alphas, both tiers
    thresh_df = pd.DataFrame(all_sweep)
    thresh_df.to_csv(cp5_dir / "conformal_thresholds.csv", index=False)
    print(f"  Saved conformal_thresholds.csv ({len(thresh_df)} rows)")

    # lodo_cross_conformal.csv
    lodo_df = pd.DataFrame(all_lodo)
    lodo_df.to_csv(cp5_dir / "lodo_cross_conformal.csv", index=False)
    print(f"  Saved lodo_cross_conformal.csv ({len(lodo_df)} rows)")

    # per_class_analysis.csv
    pc_df = pd.DataFrame(all_per_class)
    pc_df.to_csv(cp5_dir / "per_class_analysis.csv", index=False)
    print(f"  Saved per_class_analysis.csv ({len(pc_df)} rows)")

    # gated_decisions.csv — per-step decisions for both tiers
    decisions_df = pd.DataFrame({
        "dataset": gate_oof["dataset"],
        "step_idx": gate_oof["step_idx"],
        "y_true": y_true,
        "gate_y_prob": gate_probs,
        "monitor_y_prob": monitor_probs,
        "gate_crc_flag": gate_eval_10["flagged"],
        "monitor_crc_flag": monitor_eval_10["flagged"],
        "gate_naive_flag": (gate_probs >= [r for r in gate_sweep
                            if r["alpha"] == 0.10][0]["naive_threshold"]
                           ).astype(int),
        "monitor_naive_flag": (monitor_probs >= [r for r in monitor_sweep
                               if r["alpha"] == 0.10][0]["naive_threshold"]
                              ).astype(int),
        "gate_default_flag": (gate_probs >= 0.5).astype(int),
        "monitor_default_flag": (monitor_probs >= 0.5).astype(int),
    })
    decisions_df.to_csv(cp5_dir / "gated_decisions.csv", index=False)
    print(f"  Saved gated_decisions.csv ({len(decisions_df)} rows)")

    # Full results JSON
    results_for_json = json.loads(json.dumps(results, default=str))
    with open(cp5_dir / "cp5r_evaluation.json", "w") as f:
        json.dump(results_for_json, f, indent=2, default=str)
    print("  Saved cp5r_evaluation.json")

    # Plot
    _plot_fsr_curves(gate_fine, monitor_fine, gate_sweep, monitor_sweep,
                     cp5_dir / "fsr_efficiency_curves.png")
    print("  Saved fsr_efficiency_curves.png")

    # Report
    _write_report(results, cost_crc, cost_default,
                  gate_lodo, monitor_lodo,
                  cp5_dir / "cp5r_report.md")
    print("  Saved cp5r_report.md")

    print("\n" + "=" * 60)
    print("CP-5R COMPLETE")
    print("=" * 60)


def _plot_fsr_curves(gate_fine, monitor_fine, gate_key, monitor_key,
                     path: Path):
    """2×2 panel: FSR vs alpha + precision-recall, for both tiers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (tier_name, fine, key) in enumerate([
        ("Gate (Tier 1)", gate_fine, gate_key),
        ("Monitor (Tier 2)", monitor_fine, monitor_key),
    ]):
        alphas_f = [r["alpha"] for r in fine]
        crc_fsrs = [r["crc_fsr"] for r in fine]
        naive_fsrs = [r["naive_fsr"] for r in fine]

        # Top row: FSR vs alpha
        ax = axes[0, col]
        ax.plot(alphas_f, crc_fsrs, "b-", label="CRC (with +1)", linewidth=2)
        ax.plot(alphas_f, naive_fsrs, "r--", label="Naive quantile", linewidth=1.5)
        ax.plot([0, 0.5], [0, 0.5], "k:", alpha=0.4, label="y=x (ideal)")

        # Mark key alphas
        for row in key:
            if row["alpha"] in [0.05, 0.10, 0.20]:
                ax.plot(row["alpha"], row["crc_fsr"], "bo", markersize=8)
                ax.annotate(
                    f'α={row["alpha"]:.2f}\nFSR={row["crc_fsr"]:.3f}',
                    (row["alpha"], row["crc_fsr"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=7)

        ax.set_xlabel("α (FSR tolerance)")
        ax.set_ylabel("Empirical FSR")
        ax.set_title(f"{tier_name}: FSR vs α")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 0.52)
        ax.set_ylim(-0.02, 0.52)

        # Bottom row: Precision vs Recall
        ax = axes[1, col]
        crc_recalls = [r["crc_recall"] for r in fine]
        crc_precisions = [r["crc_precision"] for r in fine]
        naive_recalls = [r["naive_recall"] for r in fine]
        naive_precisions = [r["naive_precision"] for r in fine]

        ax.plot(crc_recalls, crc_precisions, "b-",
                label="CRC", linewidth=2)
        ax.plot(naive_recalls, naive_precisions, "r--",
                label="Naive", linewidth=1.5)

        for row in key:
            if row["alpha"] in [0.05, 0.10, 0.20]:
                ax.plot(row["crc_recall"], row["crc_precision"],
                        "bo", markersize=8)
                ax.annotate(
                    f'α={row["alpha"]:.2f}',
                    (row["crc_recall"], row["crc_precision"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{tier_name}: Precision-Recall")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    plt.suptitle("CP-5R: Tiered Conformal Risk Control", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(results, cost_crc, cost_default,
                  gate_lodo, monitor_lodo, path: Path):
    """Write CP-5R evaluation report."""
    lines = [
        "# CP-5R Evaluation Report: Tiered Conformal Risk Control",
        "",
        "## 1. Methodology",
        "",
        "CRC (Angelopoulos et al., ICLR 2024) applied SEPARATELY to each "
        "detector tier's OOF predictions from CP-4R.",
        "",
        "- **Tier 1 (Gate)**: CRC on D_any_gate OOF → pre-execution skip decision",
        "- **Tier 2 (Monitor)**: CRC on D_any_monitor OOF → post-execution flagging",
        "",
        "Corpus: 208 steps, 121 faulty, 87 clean, 16 datasets.",
        "",
        "## 2. CRC Thresholds at Key Alpha Levels",
        "",
        "### Tier 1 (Gate)",
        "",
        "| α | λ_CRC | FSR | Recall | Precision | F1 |",
        "|---|-------|-----|--------|-----------|-----|",
    ]

    for row in results["sweep"]["gate"]:
        lines.append(
            f"| {row['alpha']:.2f} | {row['crc_threshold']:.4f} | "
            f"{row['crc_fsr']:.4f} | {row['crc_recall']:.4f} | "
            f"{row['crc_precision']:.4f} | {row['crc_f1']:.4f} |")

    lines += [
        "",
        "### Tier 2 (Monitor)",
        "",
        "| α | λ_CRC | FSR | Recall | Precision | F1 |",
        "|---|-------|-----|--------|-----------|-----|",
    ]

    for row in results["sweep"]["monitor"]:
        lines.append(
            f"| {row['alpha']:.2f} | {row['crc_threshold']:.4f} | "
            f"{row['crc_fsr']:.4f} | {row['crc_recall']:.4f} | "
            f"{row['crc_precision']:.4f} | {row['crc_f1']:.4f} |")

    # LODO coverage
    lines += [
        "",
        "## 3. LODO Cross-Conformal Coverage",
        "",
        "| Tier | α | Covered | Coverage Rate |",
        "|------|---|---------|---------------|",
    ]

    for tier_name, lodo in [("Gate", gate_lodo), ("Monitor", monitor_lodo)]:
        for alpha in KEY_ALPHAS:
            rows = [r for r in lodo if r["alpha"] == alpha]
            n_covered = sum(1 for r in rows if r["covered"])
            lines.append(
                f"| {tier_name} | {alpha:.2f} | {n_covered}/16 | "
                f"{n_covered/16:.2%} |")

    # Calibration circularity
    lines += [
        "",
        "## 4. Calibration Circularity — Why There Is None",
        "",
        "Each OOF prediction was made by a model trained on 15 datasets "
        "and tested on 1 held-out dataset. CRC calibration on these OOF "
        "predictions uses genuinely held-out predictions.",
        "",
        "The LODO cross-conformal evaluation (§3) further validates this: "
        "it holds out one dataset from the CRC calibration set, providing "
        "a double hold-out.",
        "",
        "We do NOT claim exchangeability across datasets. LODO provides "
        "grouped out-of-fold calibration evidence. CP-6 serves as the "
        "locked external validation.",
        "",
    ]

    # Per-class
    lines += [
        "## 5. Per-Class Analysis at α=0.10",
        "",
        "| Tier | Class | Caught | Total | Recall |",
        "|------|-------|--------|-------|--------|",
    ]
    for tier_name in ["gate", "monitor"]:
        for row in results["per_class_alpha_0.10"][tier_name]:
            lines.append(
                f"| {tier_name.title()} | {row['fault_class']} | "
                f"{row['caught']} | {row['n_total']} | "
                f"{row['recall']:.4f} |")

    # Cost analysis
    t1 = cost_crc["tier1_gate"]
    t2 = cost_crc["tier2_monitor_additional"]
    comb = cost_crc["combined_detection"]
    t1_def = cost_default["tier1_gate"]

    lines += [
        "",
        "## 6. Tiered Cost Analysis at α=0.10",
        "",
        f"**Tier 1 Gate (genuine skip savings):**",
        f"- Gross savings: {t1['gross_savings_pct']:.2f}%",
        f"- FP cost: {t1['fp_cost_pct']:.2f}%",
        f"- **Net savings: {t1['net_savings_pct']:.2f}%**",
        "",
        f"**Tier 2 Monitor (additional detections):**",
        f"- {t2['additional_true_positives']} additional faulty trials "
        f"caught post-execution",
        f"- Sequential replay avoidance (no direct savings)",
        "",
        f"**Combined:** recall={comb['combined_recall']:.4f}, "
        f"oracle ceiling={comb['oracle_ceiling_pct']:.2f}%",
        "",
        f"**Comparison with default θ=0.5:**",
        f"- Default net savings: {t1_def['net_savings_pct']:.2f}%",
        f"- CRC net savings: {t1['net_savings_pct']:.2f}%",
        "",
    ]

    # Notes for CP-6
    lines += [
        "## 7. Notes for CP-6",
        "",
        "CP-5R calibrates on the 16-dataset corpus (8B model traces). "
        "CP-6 will run fresh traces on new datasets with a stronger model "
        "and apply the frozen CRC thresholds as external validation.",
        "",
        "The conformal_gate.py module accepts any OOF predictions — "
        "CP-6 produces new predictions and re-runs CRC calibration.",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
