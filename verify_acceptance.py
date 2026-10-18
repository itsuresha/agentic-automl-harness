"""Check CP-3 output against the 15 acceptance criteria in the task brief.

Lives outside scripts/ so that criterion 13 ("no hardcoded paths in scripts/")
stays meaningful. Paths are supplied on the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--python-exe", default=sys.executable)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    results: list[tuple[int, str, bool, str]] = []

    def check(n: int, desc: str, ok: bool, detail: str = "") -> None:
        results.append((n, desc, ok, detail))

    summary = json.loads((out / "cp3_summary.json").read_text())
    ev = json.loads((out / "oracle" / "oracle_evaluation.json").read_text())

    # 1 - all 16 datasets, 20 steps each
    from trace_loader import load_traces
    traces, _ = load_traces(Path(args.corpus_dir), Path(args.summary_csv), [])
    ok1 = len(traces) == 16 and all(len(v) == 20 for v in traces.values())
    check(1, "load_traces: 16 datasets x 20 steps", ok1,
          f"{len(traces)} datasets, steps={sorted({len(v) for v in traces.values()})}")

    # 2 - config extraction >= 85%
    rate = summary["feature_extraction"]["parse_success_rate"]
    check(2, "config extraction >= 85%", rate >= 0.85, f"parse_success_rate={rate}")

    # 3 - canonical_hash deterministic
    from feature_extractor import canonical_hash, extract_config
    cfgs = [extract_config(s.code) for v in traces.values() for s in v if not s.is_buggy]
    ok3 = all(canonical_hash(c) == canonical_hash(c) for c in cfgs)
    check(3, "canonical_hash deterministic", ok3, f"{len(cfgs)} configs")

    # 4 - semantic equivalence
    from feature_extractor import StepConfig
    a = StepConfig(model_class="sklearn.ensemble.RandomForestClassifier",
                   hyperparameters={"n_estimators": 100, "max_depth": None},
                   preprocessing_chain=["PCA", "StandardScaler"])
    b = StepConfig(model_class="RandomForestClassifier",
                   hyperparameters={"max_depth": "auto", "n_estimators": 100.0},
                   preprocessing_chain=["StandardScaler", "PCA"])
    check(4, "semantic equivalence (kwarg order / float / None-auto / chain order)",
          canonical_hash(a) == canonical_hash(b))

    # 5 - natural F1 >= 1
    f1n = summary["natural_faults"]["F1"]["count"]
    check(5, "natural F1 >= 1", f1n >= 1, f"F1 natural={f1n}")

    # 6 - Pipeline guard: CV-wrapped pipeline not labelled F2
    probe = (
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import cross_val_score\n"
        "pipe = make_pipeline(StandardScaler(), RandomForestClassifier())\n"
        "scores = cross_val_score(pipe, X, y, cv=5)\n"
    )
    pc = extract_config(probe)
    check(6, "F2 respects Pipeline/cross_val guard", not pc.leaky_before_split,
          f"leaky_before_split={pc.leaky_before_split}")

    # 7 - combined faulty >= 50
    tot = summary["combined"]["total_labeled_faulty"]
    check(7, "combined labelled faulty >= 50", tot >= 50, f"{tot} faulty")

    # 8 - F2 inflation (viable injections only, excluding pipeline crashes)
    f2 = ev["f2_score_inflation"]
    n_crashed = f2.get("n_crashed", 0)
    ok8 = f2["n_measured"] > 0 and f2["mean_delta"] is not None and f2["mean_delta"] > 0
    detail8 = f"mean={f2['mean_delta']:.6f} n={f2['n_measured']}"
    if n_crashed:
        detail8 += f" (excluded {n_crashed} pipeline crashes)"
    check(8, "F2 synthetic mean delta > 0 (inflation, viable only)", ok8, detail8)

    # 9 - F3 degradation
    f3 = ev["f3_score_degradation"]
    ok9 = f3["n_measured"] > 0 and f3["mean_delta"] is not None and f3["mean_delta"] < 0
    check(9, "F3 synthetic mean delta < 0 (degradation)", ok9,
          f"mean={f3['mean_delta']} n={f3['n_measured']}")

    # 10 - oracle perfect
    pc_ = ev["per_class"]
    ok10 = all(pc_[c]["precision"] == 1.0 and pc_[c]["recall"] == 1.0 and pc_[c]["f1"] == 1.0
               for c in ("F1", "F2", "F3"))
    check(10, "oracle per-class P/R/F1 == 1.0", ok10)

    # 11 - cost savings finite positive
    sp = ev["cost_savings"]["savings_pct"]
    check(11, "cost_savings_pct finite and positive", isinstance(sp, (int, float)) and sp > 0,
          f"{sp:.2f}%")

    # 12 - all artifacts exist and non-empty
    expected = [
        "features/step_features.csv", "features/step_configs.json", "features/extraction_log.txt",
        "labels/natural_labels.json", "labels/injection_manifest.json", "labels/combined_labels.csv",
        "oracle/oracle_detector.py", "oracle/oracle_evaluation.json", "oracle/oracle_report.md",
        "analysis/natural_fault_rates.csv", "analysis/config_hash_collisions.csv",
        "analysis/f2_score_deltas.csv", "analysis/f3_score_deltas.csv",
        "analysis/fault_distribution.png",
        "scripts/run_cp3.py", "scripts/trace_loader.py", "scripts/feature_extractor.py",
        "scripts/fault_labeler.py", "scripts/fault_injector.py", "scripts/reexec_harness.py",
        "scripts/oracle.py", "cp3_summary.json",
    ]
    missing = [p for p in expected if not (out / p).exists() or (out / p).stat().st_size == 0]
    check(12, "all artifacts present and non-empty", not missing, f"missing/empty: {missing}")

    # 13 - no hardcoded absolute paths in scripts/
    offenders = []
    for py in (out / "scripts").glob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r'["\'](?:/Users/|/home/|[A-Z]:\\\\)', line):
                offenders.append(f"{py.name}:{i}")
    check(13, "no hardcoded absolute paths in scripts/", not offenders, f"{offenders}")

    # 14 - reproducibility (re-run with same seed, skip re-exec, compare labels+features)
    tmp = out.parent / "_cp3_repro_check"
    subprocess.run(
        [args.python_exe, str(out / "scripts" / "run_cp3.py"),
         "--corpus-dir", args.corpus_dir, "--summary-csv", args.summary_csv,
         "--output-dir", str(tmp), "--project-root", args.project_root,
         "--skip-reexec", "--seed", "42"],
        capture_output=True, check=False,
    )
    subprocess.run(
        [args.python_exe, str(out / "scripts" / "run_cp3.py"),
         "--corpus-dir", args.corpus_dir, "--summary-csv", args.summary_csv,
         "--output-dir", str(tmp) + "_b", "--project-root", args.project_root,
         "--skip-reexec", "--seed", "42"],
        capture_output=True, check=False,
    )
    same = True
    detail14 = ""
    for rel in ("features/step_features.csv", "labels/combined_labels.csv"):
        pa, pb = tmp / rel, Path(str(tmp) + "_b") / rel
        if not pa.exists() or not pb.exists():
            same = False; detail14 = f"{rel} missing"; break
        if pa.read_bytes() != pb.read_bytes():
            same = False; detail14 = f"{rel} differs"; break
    check(14, "reproducible with --seed 42", same, detail14 or "features+labels byte-identical")

    # 15 - summary schema
    need = {"corpus", "feature_extraction", "natural_faults", "synthetic_injections",
            "combined", "oracle", "runtime_sec"}
    check(15, "cp3_summary.json valid and complete", need.issubset(summary.keys()),
          f"missing={need - set(summary.keys())}")

    # -- report -------------------------------------------------------------
    width = max(len(d) for _, d, _, _ in results)
    npass = sum(1 for _, _, ok, _ in results if ok)
    print(f"\nCP-3 acceptance criteria: {npass}/{len(results)} passed\n")
    for n, desc, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  {n:>2}. [{mark}] {desc:<{width}}  {detail}")
    print()
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
