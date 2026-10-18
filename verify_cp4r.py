"""CP-4R acceptance verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    cp4_dir = out / "cp4_results"
    eval_path = cp4_dir / "cp4r_evaluation.json"
    if not eval_path.exists():
        print(f"FATAL: {eval_path} not found")
        return 1

    results = json.loads(eval_path.read_text())
    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        suffix = f" - {detail}" if detail else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {total}. {name}{suffix}")

    print("=" * 60)
    print("CP-4R Acceptance Verification")
    print("=" * 60)

    corpus = results.get("corpus", {})
    check("Corpus has 208 steps", corpus.get("n_steps") == 208, f"got {corpus.get('n_steps')}")
    check("Corpus has 16 datasets", corpus.get("n_datasets") == 16, f"got {corpus.get('n_datasets')}")
    check(
        "Fault counts match CP-3R",
        corpus.get("F1") == 64
        and corpus.get("F2") == 40
        and corpus.get("F3") == 40
        and corpus.get("any_fault") == 121
        and corpus.get("clean") == 87,
        str(corpus),
    )

    d1 = results.get("D1", {}).get("aggregate", {})
    check("D1 P=1.0 for F1/RCP", d1.get("precision") == 1.0, f"got {d1.get('precision')}")
    check("D1 R=1.0 for F1/RCP", d1.get("recall") == 1.0, f"got {d1.get('recall')}")
    taut = results.get("D1_tautology_disclosure", {})
    check("D1 tautology disclosure acknowledged", taut.get("status") == "acknowledged", f"status={taut.get('status')}")
    check("D1 tautology has construct evidence", "independent_evidence" in taut, str(list(taut.keys())))

    for key in ["D2_gate", "D2_monitor", "D3_gate", "D3_monitor", "D_any_gate", "D_any_monitor"]:
        check(f"{key} has 16 LODO folds", len(results.get(key, {}).get("per_fold", [])) == 16)

    rand = results.get("baseline_random", {})
    check(
        "D2_monitor F1 > random(F2)",
        results.get("D2_monitor", {}).get("aggregate", {}).get("f1", 0)
        > rand.get("F2", {}).get("expected_f1", 0),
    )
    effects = results.get("F2_effect_sizes", {})
    bins = [k for k in effects if k not in {"undetectable_region", "ground_truth_deltas"}]
    check("F2 effect-size has >=4 bins", len(bins) >= 4, str(bins))
    check("F2 undetectable region documented", "undetectable_region" in effects)

    check(
        "D3_monitor F1 > random(F3)",
        results.get("D3_monitor", {}).get("aggregate", {}).get("f1", 0)
        > rand.get("F3", {}).get("expected_f1", 0),
    )
    strategies = results.get("D3_monitor", {}).get("per_strategy_recall", {})
    primary = {k: v for k, v in strategies.items() if k in {"clamp_n_est", "clamp_lr", "clamp_depth"}}
    check("F3 per-strategy recall >=3 primary strategies", len(primary) >= 3, str(strategies))
    check("All primary F3 strategies recall > 0", all(v.get("recall", 0) > 0 for v in primary.values()), str(primary))

    any_rand = rand.get("any_fault", {}).get("expected_f1", 0)
    dany_gate = results.get("D_any_gate", {}).get("aggregate", {})
    dany_monitor = results.get("D_any_monitor", {}).get("aggregate", {})
    check("D_any_gate F1 > random(any_fault)", dany_gate.get("f1", 0) > any_rand)
    check("D_any_monitor F1 > random(any_fault)", dany_monitor.get("f1", 0) > any_rand)
    check(
        "D_any_monitor F1 >= D_any_gate F1",
        dany_monitor.get("f1", 0) >= dany_gate.get("f1", 0),
        f"monitor={dany_monitor.get('f1')} gate={dany_gate.get('f1')}",
    )
    thresholds = results.get("baseline_score_threshold", {})
    best_threshold = max((v["f1"] for v in thresholds.values()), default=0)
    check("D_any_monitor F1 > best score-threshold", dany_monitor.get("f1", 0) > best_threshold)

    savings = results.get("savings_analysis", {})
    check("Tier 1 savings reported", "tier1_gate" in savings)
    check("Tier 2 detections reported separately", "tier2_monitor_additional" in savings)

    fi = results.get("feature_importance", {})
    for key in ["D3_gate", "D3_monitor", "D_any_gate", "D_any_monitor"]:
        check(f"Feature importance for {key}", key in fi and "importances" in fi.get(key, {}))

    for name in ["D_any_gate", "D_any_monitor"]:
        path = cp4_dir / f"oof_predictions_{name}.csv"
        check(f"OOF file exists for {name}", path.exists())
        if path.exists():
            oof = pd.read_csv(path)
            check(f"OOF {name} has required columns", list(oof.columns) == ["dataset", "step_idx", "y_true", "y_prob"])
            check(f"OOF {name} has 208 rows", len(oof) == 208, f"got {len(oof)}")
        else:
            check(f"OOF {name} has required columns", False, "missing")
            check(f"OOF {name} has 208 rows", False, "missing")

    expected_files = [
        "cp4r_evaluation.json",
        "cp4r_report.md",
        "detector_comparison.csv",
        "detector_comparison.png",
        "savings_analysis.json",
        "tautology_disclosure.json",
        "oof_predictions_D_any_gate.csv",
        "oof_predictions_D_any_monitor.csv",
        "lodo_folds_D1.csv",
        "lodo_folds_D2_gate.csv",
        "lodo_folds_D2_monitor.csv",
        "lodo_folds_D3_gate.csv",
        "lodo_folds_D3_monitor.csv",
        "lodo_folds_D_any_gate.csv",
        "lodo_folds_D_any_monitor.csv",
    ]
    missing = [name for name in expected_files if not (cp4_dir / name).exists()]
    check("All 15 CP4R output files exist", not missing, f"missing={missing}")
    check("Oracle baseline documented", "baseline_oracle" in results)

    (cp4_dir / "verify_cp4r_results.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "total": total,
                "failed": total - passed,
            },
            indent=2,
        )
    )
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
