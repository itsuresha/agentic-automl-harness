"""Additional CP-3b checks; the original 15-criterion verifier is untouched."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
BACKUP = OUT.parent / "cp3_fault_injection_backup_v1"
sys.path.insert(0, str(OUT / "scripts"))
from fault_labeler import label_f3
from feature_extractor import extract_config


def main():
    records = json.loads((OUT / "labels/injection_manifest.json").read_text())
    old = json.loads((BACKUP / "labels/injection_manifest.json").read_text())
    summary = json.loads((OUT / "cp3_summary.json").read_text())
    f3 = [r for r in records if r["fault_class"] == "F3" and not r["is_natural"]]
    counts = Counter(r.get("injection_details", {}).get("mutation_strategy", "unknown") for r in f3)
    stats = summary["synthetic_injections"]["F3"]["per_strategy"]
    label_failures = []
    for r in f3:
        step = SimpleNamespace(is_buggy=False, validation_score=r["injected_score"], execution_output="")
        labels, reasons = label_f3([step], [extract_config(r["mutated_code"])])
        if labels != [True]:
            label_failures.append([r["dataset"], r["step_idx"], reasons])
    def negative(name):
        value = stats[name]["mean_delta"]
        return value is not None and value < 0
    checks = {
        "16_three_strategies": len(set(counts) - {"unknown"}) >= 3,
        "17_maximum_share_60_percent": bool(f3) and max(counts.values()) / len(f3) <= .60,
        "18_learning_rate_negative_mean": negative("clamp_lr"),
        "19_depth_negative_mean": negative("clamp_depth"),
        "20_all_strategy_fields": bool(f3) and "unknown" not in counts,
        "21_independent_labeler_recognizes_all_F3": bool(f3) and not label_failures,
    }
    # Execution duration is expected to vary; every other F1/F2 record must match.
    def stable(rs):
        return [{k:v for k,v in r.items() if k != "reexec_wall_clock_sec"}
                for r in rs if r["fault_class"] in {"F1", "F2"}]
    regression = {
        "F1_F2_records_unchanged_except_runtime": stable(records) == stable(old),
        "original_verifier_unchanged": (OUT / "verify_acceptance.py").read_bytes() == (BACKUP / "verify_acceptance.py").read_bytes(),
        "features_unchanged": (OUT / "features/step_features.csv").read_bytes() == (BACKUP / "features/step_features.csv").read_bytes(),
    }
    evidence = dict(checks=checks, regression_checks=regression, strategy_counts=dict(counts),
                    per_strategy=stats, independent_labeler_failures=label_failures,
                    original_verifier_sha256=hashlib.sha256((OUT / "verify_acceptance.py").read_bytes()).hexdigest(),
                    api_cost_usd=0)
    passed = all(checks.values()) and all(regression.values())
    evidence["status"] = "PASSED" if passed else "FAILED"
    (OUT / "analysis/diversification_acceptance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    text = ["# CP-3b F3 diversification", "", f"Status: {evidence['status']}. API cost: $0.", "",
            "| Strategy | Successful | Non-viable | Viable mean delta |", "|---|---:|---:|---:|"]
    for name, result in stats.items():
        text.append(f"| {name} | {result['succeeded']} | {result['crashed']} | {result['mean_delta']} |")
    text += ["", "## Incremental acceptance and regression", ""]
    text += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in {**checks, **regression}.items()]
    text += ["", "The original 15-criterion output is in execution_logs/acceptance-15.log. Model class availability determines eligible strategies; clamp_iter is supported but need not be selected in this corpus.",
             "", "The oracle's perfect score is by construction, not learned-detector performance. Criterion 21 additionally tests the actual labeler against each mutated source. Viability thresholds are unchanged; non-viable runs remain visible and are excluded only from delta means.",
             "", "The brief's aide_runs and checkpoint .venv paths do not exist here. The run uses ../traces/raw, ../results/trace_corpus_summary.csv and ../.venv, matching the original 16-dataset/320-step feature inventory."]
    (OUT / "CP3B_REPORT.md").write_text("\n".join(text) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
