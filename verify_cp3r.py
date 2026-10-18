"""verify_cp3r.py - unified CP-3R acceptance checks."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def _load_json(path: Path):
    return json.loads(path.read_text())


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _canonical_hash_preserves_order(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "canonical_hash":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "sorted":
                    segment = ast.get_source_segment(path.read_text(), sub) or ""
                    if "preprocessing_chain" in segment or "prep" in segment:
                        return False
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    scripts = out / "scripts"
    sys.path.insert(0, str(scripts))

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    features_path = out / "features" / "step_features.csv"
    labels_path = out / "labels" / "combined_labels.csv"
    manifest_path = out / "labels" / "injection_manifest.json"
    oracle_path = out / "oracle" / "oracle_evaluation.json"
    summary_path = out / "cp3r_summary.json"

    features = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
    labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    manifest = _load_json(manifest_path) if manifest_path.exists() else []
    oracle = _load_json(oracle_path) if oracle_path.exists() else {}
    summary = _load_json(summary_path) if summary_path.exists() else {}

    f1 = _bool_series(labels["F1"]) if "F1" in labels else pd.Series(dtype=bool)
    f2 = _bool_series(labels["F2"]) if "F2" in labels else pd.Series(dtype=bool)
    f3 = _bool_series(labels["F3"]) if "F3" in labels else pd.Series(dtype=bool)
    any_fault = _bool_series(labels["any_fault"]) if "any_fault" in labels else pd.Series(dtype=bool)

    # Original CP-3 checks, adapted to the current artifact names.
    check("01_step_features_exists_ge_200", features_path.exists() and len(features) >= 200, f"rows={len(features)}")
    check("02_combined_labels_same_row_count", labels_path.exists() and len(labels) == len(features), f"labels={len(labels)} features={len(features)}")
    check("03_oracle_evaluation_exists", oracle_path.exists(), str(oracle_path.relative_to(out)) if oracle_path.exists() else "missing")
    check("04_F1_count_ge_30", int(f1.sum()) >= 30, f"F1={int(f1.sum())}")
    check("05_F2_count_ge_35", int(f2.sum()) >= 35, f"F2={int(f2.sum())}")
    check("06_F3_count_ge_35", int(f3.sum()) >= 35, f"F3={int(f3.sum())}")
    check("07_total_faulty_ge_100", int(any_fault.sum()) >= 100, f"faulty={int(any_fault.sum())}")
    check("08_clean_steps_ge_70", int((~any_fault).sum()) >= 70, f"clean={int((~any_fault).sum())}")
    check("09_datasets_covered_16", features.get("dataset", pd.Series(dtype=str)).nunique() == 16, f"datasets={features.get('dataset', pd.Series(dtype=str)).nunique()}")
    required_cols = {
        "validation_score", "wall_clock_sec", "canonical_hash", "model_class",
        "n_hyperparams", "hash_seen_before", "preprocessing_before_split",
    }
    check("10_required_feature_columns", required_cols.issubset(features.columns), f"missing={sorted(required_cols - set(features.columns))}")
    check("11_no_nan_fault_labels", not labels[[c for c in ["F1", "F2", "F3"] if c in labels]].isna().any().any(), "")
    per_class = oracle.get("per_class", {})
    check("12_oracle_recall_1", all(per_class.get(c, {}).get("recall") == 1.0 for c in ("F1", "F2", "F3")), str({c: per_class.get(c, {}).get("recall") for c in ("F1", "F2", "F3")}))
    check("13_oracle_precision_1", all(per_class.get(c, {}).get("precision") == 1.0 for c in ("F1", "F2", "F3")), str({c: per_class.get(c, {}).get("precision") for c in ("F1", "F2", "F3")}))
    check("14_injection_manifest_exists", manifest_path.exists() and isinstance(manifest, list), f"type={type(manifest).__name__} len={len(manifest) if isinstance(manifest, list) else 'na'}")
    check("15_cp3r_summary_exists", summary_path.exists(), str(summary_path.relative_to(out)) if summary_path.exists() else "missing")

    # F3 diversification checks.
    f3_records = [r for r in manifest if r.get("fault_class") == "F3" and not r.get("is_natural")]
    strategies = Counter(r.get("injection_details", {}).get("mutation_strategy", "unknown") for r in f3_records)
    check("16_F3_total_eq_40", len(f3_records) == 40, f"F3 synthetic={len(f3_records)}")
    check("17_F3_ge_2_strategies", len(set(strategies) - {"unknown"}) >= 2, str(dict(strategies)))
    primary_strategies = {"clamp_n_est", "clamp_lr", "clamp_depth"}
    primary_ok = all(strategies.get(name, 0) >= 10 for name in primary_strategies)
    check("18_primary_F3_strategies_ge_10", primary_ok, str(dict(strategies)))
    check("19_clamp_n_est_exists", strategies.get("clamp_n_est", 0) > 0, str(dict(strategies)))
    lr_ok = any(
        r.get("injection_details", {}).get("mutation_strategy") == "clamp_lr"
        and float(r.get("injection_details", {}).get("clamped_value", 0) or 0) >= 50
        for r in f3_records
    )
    check("20_clamp_lr_exists_ge_50", lr_ok, str(dict(strategies)))
    check("21_clamp_depth_exists", strategies.get("clamp_depth", 0) > 0, str(dict(strategies)))

    # CP-3R checks.
    fe_path = scripts / "feature_extractor.py"
    check("R1_canonical_hash_preserves_preprocessing_order", _canonical_hash_preserves_order(fe_path), "")
    from feature_extractor import FEATURE_TIERS, StepConfig
    annotations = getattr(StepConfig, "__dataclass_fields__", {})
    check("R2_StepConfig_preprocessing_params", "preprocessing_params" in annotations, "")
    check("R3_StepConfig_split_method", "split_method" in annotations, "")
    check("R4_StepConfig_feature_selector", "feature_selector" in annotations, "")
    tier_path = out / "features" / "feature_tier_map.json"
    tiers = _load_json(tier_path) if tier_path.exists() else {}
    check("R5_feature_tier_map_exists", tier_path.exists(), "")
    check("R6_feature_tier_assignments_match", tiers.get("tiers") == FEATURE_TIERS, "")
    synthetic_f2 = {
        (r["dataset"], int(r["step_idx"]))
        for r in manifest
        if r.get("fault_class") == "F2" and not r.get("is_natural")
    }
    feature_lookup = {
        (row["dataset"], int(row["step_idx"])): row
        for _, row in features.iterrows()
    }
    f2_values = [bool(feature_lookup[k]["leaky_before_split"]) for k in synthetic_f2 if k in feature_lookup]
    check("R7_synthetic_F2_code_leakage_feature_true", len(f2_values) == len(synthetic_f2) and all(f2_values), f"true={sum(f2_values)} total={len(synthetic_f2)}")
    fe_text = fe_path.read_text()
    no_manifest_reads = not re.search(r"injection_manifest|injection_results", fe_text)
    check("R8_feature_extractor_no_injection_metadata_refs", no_manifest_reads, "")
    rcp_analysis = out / "analysis" / "rcp_construct_analysis.csv"
    rcp_summary = out / "analysis" / "rcp_construct_summary.json"
    check("R9_rcp_construct_analysis_exists", rcp_analysis.exists(), "")
    rcp_summary_obj = _load_json(rcp_summary) if rcp_summary.exists() else {}
    check("R10_rcp_construct_summary_mean_abs_delta", rcp_summary.exists() and "mean_abs_delta" in rcp_summary_obj, "")
    taxonomy = _load_json(out / "analysis" / "fault_taxonomy.json") if (out / "analysis" / "fault_taxonomy.json").exists() else {}
    tax_ok = all(
        c in taxonomy.get("taxonomy", {}) and "name" in taxonomy["taxonomy"][c] and "definition" in taxonomy["taxonomy"][c]
        for c in ("F1", "F2", "F3")
    )
    check("R11_fault_taxonomy_complete", tax_ok, "")
    file_manifest = _load_json(out / "analysis" / "file_manifest.json") if (out / "analysis" / "file_manifest.json").exists() else {}
    check("R12_file_manifest_ge_10_entries", len(file_manifest) >= 10, f"entries={len(file_manifest)}")
    validation_sample = _load_json(out / "analysis" / "validation_sample.json") if (out / "analysis" / "validation_sample.json").exists() else []
    check("R13_validation_sample_20_entries", isinstance(validation_sample, list) and len(validation_sample) == 20, f"entries={len(validation_sample) if isinstance(validation_sample, list) else 'na'}")
    metadata = _load_json(out / "features" / "feature_metadata.json") if (out / "features" / "feature_metadata.json").exists() else {}
    check("R14_feature_metadata_hash_v2", metadata.get("canonical_hash_version") == "v2", str(metadata.get("canonical_hash_version")))

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"CP-3R Verification: {passed}/{total} PASSED")
    print(f"{'=' * 60}")
    if int(f1.sum()) < 50:
        print(f"WARNING: F1 natural count under v2 is below 50: {int(f1.sum())}")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    (out / "verify_cp3r_results.json").write_text(json.dumps({
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "checks": [
            {"name": name, "passed": ok, "detail": detail}
            for name, ok, detail in results
        ],
    }, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
