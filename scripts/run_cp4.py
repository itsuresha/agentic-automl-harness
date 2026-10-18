"""CP-4R runner: tiered practical detectors with LODO-CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from detectors import (  # noqa: E402
    d1_tautology_disclosure,
    f2_effect_size_analysis,
    feature_importance_analysis,
    load_data,
    lodo_evaluate,
    per_target_random_baseline,
    savings_analysis,
    score_threshold_baseline,
    verify_tier_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CP-4R: tiered practical detectors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out = Path(args.output_dir)
    cp4_dir = out / "cp4_results"
    cp4_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CP-4R: Tiered Feature-Based Practical Detectors")
    print("=" * 60)

    print("\n[1/8] Loading data and verifying corpus integrity...")
    df = load_data(out)
    verify_tier_map(out)
    n = len(df)
    results: dict[str, object] = {
        "corpus": {
            "n_steps": n,
            "n_datasets": int(df["dataset"].nunique()),
            "F1": int(df["F1"].sum()),
            "F2": int(df["F2"].sum()),
            "F3": int(df["F3"].sum()),
            "any_fault": int(df["any_fault"].sum()),
            "clean": int(n - df["any_fault"].sum()),
        }
    }
    print(
        f"  {n} steps from {results['corpus']['n_datasets']} datasets | "
        f"F1={results['corpus']['F1']} F2={results['corpus']['F2']} "
        f"F3={results['corpus']['F3']} any={results['corpus']['any_fault']} "
        f"clean={results['corpus']['clean']}"
    )

    manifest_path = out / "labels" / "injection_manifest.json"

    print("\n[2/8] D1 (RCP, Tier 1 definitional rule)...")
    results["D1"] = lodo_evaluate(df, "F1", detector_type="rule_f1", tier="gate")
    _print_aggregate(results["D1"])
    results["D1_tautology_disclosure"] = d1_tautology_disclosure(out)
    print(f"  Tautology disclosure: {results['D1_tautology_disclosure']['status']}")

    print("\n[3/8] D2 (DL, gate + monitor)...")
    results["D2_gate"] = lodo_evaluate(df, "F2", detector_type="gate_f2", tier="gate")
    print("  D2_gate:   ", end="")
    _print_aggregate(results["D2_gate"])
    results["D2_monitor"] = lodo_evaluate(df, "F2", detector_type="monitor_f2", tier="all")
    print("  D2_monitor:", end=" ")
    _print_aggregate(results["D2_monitor"])
    results["F2_effect_sizes"] = f2_effect_size_analysis(df, out / "analysis" / "f2_score_deltas.csv")

    print("\n[4/8] D3 (TBF, gate + monitor)...")
    results["D3_gate"] = lodo_evaluate(
        df, "F3", detector_type="learned", tier="gate", manifest_path=manifest_path
    )
    print("  D3_gate:   ", end="")
    _print_aggregate(results["D3_gate"])
    results["D3_monitor"] = lodo_evaluate(
        df, "F3", detector_type="learned", tier="all", manifest_path=manifest_path
    )
    print("  D3_monitor:", end=" ")
    _print_aggregate(results["D3_monitor"])
    for strategy, info in results["D3_monitor"].get("per_strategy_recall", {}).items():
        print(f"    {strategy}: recall={info['recall']:.4f} support={info['support']}")

    print("\n[5/8] D_any (combined, gate + monitor)...")
    results["D_any_gate"] = lodo_evaluate(df, "any_fault", detector_type="learned", tier="gate")
    print("  D_any_gate:   ", end="")
    _print_aggregate(results["D_any_gate"])
    results["D_any_monitor"] = lodo_evaluate(df, "any_fault", detector_type="learned", tier="all")
    print("  D_any_monitor:", end=" ")
    _print_aggregate(results["D_any_monitor"])

    print("\n[6/8] Baselines...")
    results["baseline_score_threshold"] = score_threshold_baseline(df)
    results["baseline_random"] = per_target_random_baseline(df)
    results["baseline_oracle"] = _oracle_baseline(out)
    results["baseline_ungated"] = {
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "cost_savings_pct": 0.0,
        "description": "No detector; all steps accepted.",
    }
    print(f"  random any_fault expected F1={results['baseline_random']['any_fault']['expected_f1']:.4f}")

    print("\n[7/8] Feature importance and savings...")
    results["feature_importance"] = {
        "D3_gate": feature_importance_analysis(df, "F3", tier="gate"),
        "D3_monitor": feature_importance_analysis(df, "F3", tier="all"),
        "D_any_gate": feature_importance_analysis(df, "any_fault", tier="gate", include_d_any_meta=True),
        "D_any_monitor": feature_importance_analysis(df, "any_fault", tier="all", include_d_any_meta=True),
    }
    gate_preds = np.array(results["D_any_gate"]["oof_y_pred"])
    monitor_preds = np.array(results["D_any_monitor"]["oof_y_pred"])
    results["savings_analysis"] = savings_analysis(df, gate_preds, monitor_preds)
    s = results["savings_analysis"]
    print(
        f"  Tier 1 genuine savings: {s['tier1_gate']['savings_pct']:.2f}% | "
        f"Tier 2 additional TP: {s['tier2_monitor_additional']['additional_true_positives']}"
    )

    print("\n[8/8] Exporting artifacts...")
    _save_outputs(results, df, cp4_dir)
    print("\nCP-4R COMPLETE")
    print(f"Results written to {cp4_dir}")
    return 0


def _print_aggregate(result: dict) -> None:
    agg = result["aggregate"]
    print(f"P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f}")


def _oracle_baseline(out: Path) -> dict:
    oracle = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "source": "cp3r_summary.json"}
    path = out / "cp3r_summary.json"
    if path.exists():
        cp3r = json.loads(path.read_text())
        oracle["cost_savings_pct"] = cp3r.get("oracle", {}).get("cost_savings_pct")
    return oracle


def _save_outputs(results: dict, df: pd.DataFrame, cp4_dir: Path) -> None:
    for name in ["D_any_gate", "D_any_monitor"]:
        pd.DataFrame(
            {
                "dataset": df["dataset"],
                "step_idx": df["step_idx"],
                "y_true": df["any_fault"],
                "y_prob": results[name]["oof_y_prob"],
            }
        ).to_csv(cp4_dir / f"oof_predictions_{name}.csv", index=False)

    serializable = json.loads(json.dumps(results, default=str))
    for key in ["D1", "D2_gate", "D2_monitor", "D3_gate", "D3_monitor", "D_any_gate", "D_any_monitor"]:
        serializable[key].pop("oof_y_pred", None)
        serializable[key].pop("oof_y_prob", None)
        serializable[key].pop("oof_predictions", None)
    (cp4_dir / "cp4r_evaluation.json").write_text(json.dumps(serializable, indent=2))
    (cp4_dir / "savings_analysis.json").write_text(json.dumps(results["savings_analysis"], indent=2))
    (cp4_dir / "tautology_disclosure.json").write_text(
        json.dumps(results["D1_tautology_disclosure"], indent=2, default=str)
    )

    for key in ["D1", "D2_gate", "D2_monitor", "D3_gate", "D3_monitor", "D_any_gate", "D_any_monitor"]:
        pd.DataFrame(results[key]["per_fold"]).to_csv(cp4_dir / f"lodo_folds_{key}.csv", index=False)

    summary = _summary_table(results)
    summary.to_csv(cp4_dir / "detector_comparison.csv", index=False)
    _plot_comparison(summary, cp4_dir / "detector_comparison.png")
    _write_report(results, cp4_dir / "cp4r_report.md")


def _summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for label, key, tier in [
        ("D1 (RCP rule)", "D1", "Gate"),
        ("D2_gate (DL structural)", "D2_gate", "Gate"),
        ("D2_monitor (DL stat+struct)", "D2_monitor", "Monitor"),
        ("D3_gate (TBF Tier1)", "D3_gate", "Gate"),
        ("D3_monitor (TBF all)", "D3_monitor", "Monitor"),
        ("D_any_gate (Tier1+meta)", "D_any_gate", "Gate"),
        ("D_any_monitor (all+meta)", "D_any_monitor", "Monitor"),
    ]:
        agg = results[key]["aggregate"]
        rows.append(
            {
                "detector": label,
                "tier": tier,
                "target": results[key]["target"],
                "precision": agg["precision"],
                "recall": agg["recall"],
                "f1": agg["f1"],
                "auroc": agg.get("auroc"),
            }
        )
    best_t, best = max(results["baseline_score_threshold"].items(), key=lambda item: item[1]["f1"])
    rows.append(
        {
            "detector": f"Score threshold ({best_t})",
            "tier": "Monitor",
            "target": "any_fault",
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "auroc": None,
        }
    )
    for target, info in results["baseline_random"].items():
        rows.append(
            {
                "detector": f"Random ({target})",
                "tier": "-",
                "target": target,
                "precision": info["expected_precision"],
                "recall": info["expected_recall"],
                "f1": info["expected_f1"],
                "auroc": None,
            }
        )
    rows.append(
        {
            "detector": "Oracle (CP-3R)",
            "tier": "-",
            "target": "any_fault",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "auroc": 1.0,
        }
    )
    return pd.DataFrame(rows)


def _plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary[summary["target"].isin(["any_fault", "F1"])].copy()
    labels = plot_df["detector"].tolist()
    values = plot_df["f1"].tolist()
    colors = []
    for label in labels:
        if "Oracle" in label:
            colors.append("#2ecc71")
        elif "Random" in label or "Score" in label:
            colors.append("#95a5a6")
        elif "gate" in label.lower() or "Gate" in label:
            colors.append("#3498db")
        elif "monitor" in label.lower():
            colors.append("#e67e22")
        else:
            colors.append("#9b59b6")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("CP-4R: Tiered Detector Comparison (LODO-CV)")
    ax.set_xlim(0, 1.1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(results: dict, path: Path) -> None:
    lines = [
        "# CP-4R Evaluation Report: Tiered Practical Detectors",
        "",
        "All detectors use 16-fold leave-one-dataset-out cross-validation.",
        "Tier 1 detectors are pre-execution gates. Tier 2 detectors are post-execution monitors.",
        "",
        "## Detector Results",
        "",
        "| Detector | Tier | Target | P | R | F1 | AUROC |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("D1", "D1"),
        ("D2_gate", "D2_gate"),
        ("D2_monitor", "D2_monitor"),
        ("D3_gate", "D3_gate"),
        ("D3_monitor", "D3_monitor"),
        ("D_any_gate", "D_any_gate"),
        ("D_any_monitor", "D_any_monitor"),
    ]:
        agg = results[key]["aggregate"]
        auroc = f"{agg['auroc']:.4f}" if agg.get("auroc") is not None else "-"
        lines.append(
            f"| {label} | {results[key]['tier']} | {results[key]['target']} | "
            f"{agg['precision']:.4f} | {agg['recall']:.4f} | {agg['f1']:.4f} | {auroc} |"
        )
    lines.extend(
        [
            "",
            "## D1 Tautology Disclosure",
            "",
            results["D1_tautology_disclosure"]["explanation"],
            "",
            "Independent CP-3R construct summary:",
            "",
            "```json",
            json.dumps(
                results["D1_tautology_disclosure"]
                .get("independent_evidence", {})
                .get("construct_summary", {}),
                indent=2,
            ),
            "```",
            "",
            "## Savings Analysis",
            "",
            f"Tier 1 genuine skip savings: {results['savings_analysis']['tier1_gate']['savings_pct']:.2f}%.",
            f"Tier 2 additional detections: {results['savings_analysis']['tier2_monitor_additional']['additional_true_positives']}.",
            "",
            "## CP-5R Inputs",
            "",
            "- `oof_predictions_D_any_gate.csv`",
            "- `oof_predictions_D_any_monitor.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
