"""CP-3: Fault Injection & Oracle Detector.

Usage:
    python run_cp3.py --corpus-dir <path> --summary-csv <path> --output-dir <path>

All work is local: no API calls, no GPU, no network. Synthetic F2/F3 faults
are re-executed against the dataset CSVs already on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import itertools
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_injector import inject_leakage, inject_partial_fit, get_applicable_strategies, F3_STRATEGIES  # noqa: E402
from fault_labeler import ENSEMBLE_MODELS, ITERATIVE_MODELS, TREE_MODELS, label_dataset  # noqa: E402
from feature_extractor import FEATURE_TIERS, HASH_VERSION, _is_leaky_fit_from_code, canonical_hash, extract_config  # noqa: E402
from oracle import InjectionRecord, OracleDetector, compute_cost_savings  # noqa: E402
from reexec_harness import is_viable_injection, reexecute_step         # noqa: E402
from trace_loader import load_traces                                   # noqa: E402

FAULT_CLASSES = ("F1", "F2", "F3")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# --------------------------------------------------------------------------
def build_features(traces, configs) -> list[dict]:
    """Per-step feature rows for every non-buggy step."""
    rows: list[dict] = []
    for ds, steps in traces.items():
        seen_hash: dict[str, int] = {}
        best = None
        last_improve = 0
        prev_score = None
        scores_so_far: list[float] = []

        for s in steps:
            if s.is_buggy:
                continue
            cfg = configs[(ds, s.step_idx)]
            h = canonical_hash(cfg) if cfg else ""
            score = s.validation_score

            if score is not None:
                scores_so_far.append(score)
                rank = sorted(scores_so_far, reverse=True).index(score) + 1
                is_best = best is None or score > best
                if is_best:
                    best = score
                    last_improve = s.step_idx
            else:
                rank = -1
                is_best = False

            rows.append({
                "dataset": ds,
                "step_idx": s.step_idx,
                "canonical_hash": h,
                "model_class": cfg.model_class if cfg else "",
                "n_hyperparams": len(cfg.hyperparameters) if cfg else 0,
                "n_preprocessing_steps": len(cfg.preprocessing_chain) if cfg else 0,
                "preprocessing_before_split": bool(cfg.preprocessing_before_split) if cfg else False,
                "leaky_before_split": bool(cfg.leaky_before_split) if cfg else False,
                "validation_score": score,
                "score_delta_from_prev": (score - prev_score) if (score is not None and prev_score is not None) else None,
                "score_rank_in_run": rank,
                "is_best_so_far": is_best,
                "steps_since_improvement": s.step_idx - last_improve,
                "hash_seen_before": h in seen_hash,
                "hash_first_seen_at": seen_hash.get(h),
                "parse_quality": cfg.parse_quality if cfg else "unparseable",
                "wall_clock_sec": s.wall_clock_sec,
                "feature_tier": 2,
            })

            if h and h not in seen_hash:
                seen_hash[h] = s.step_idx
            if score is not None:
                prev_score = score
    return rows


def reconcile_features_after_injection(feature_rows: list[dict], records: list[InjectionRecord]) -> list[dict]:
    """Refresh code-derived features from mutated code for synthetic F2/F3 rows."""
    by_key = {
        (r.dataset, r.step_idx): r
        for r in records
        if not r.is_natural and r.fault_class in {"F2", "F3"} and r.mutated_code
    }
    for row in feature_rows:
        rec = by_key.get((row["dataset"], row["step_idx"]))
        if rec is None:
            continue
        cfg = extract_config(rec.mutated_code or "")
        row["canonical_hash"] = canonical_hash(cfg)
        row["model_class"] = cfg.model_class
        row["n_hyperparams"] = len(cfg.hyperparameters)
        row["n_preprocessing_steps"] = len(cfg.preprocessing_chain)
        row["preprocessing_before_split"] = bool(cfg.preprocessing_before_split)
        row["leaky_before_split"] = bool(_is_leaky_fit_from_code(rec.mutated_code or ""))
        row["parse_quality"] = cfg.parse_quality
        if rec.injected_score is not None:
            row["validation_score"] = rec.injected_score
        if rec.reexec_wall_clock_sec is not None:
            row["wall_clock_sec"] = rec.reexec_wall_clock_sec
    return recompute_sequential_features(feature_rows)


def recompute_sequential_features(feature_rows: list[dict]) -> list[dict]:
    """Recompute all within-run features after mutated-code reconciliation."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["dataset"]].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step_idx"]))
        seen_hashes: dict[str, int] = {}
        best_score = -float("inf")
        steps_since_best = 0
        prev_score = None
        scores: list[tuple[int, float]] = []

        for pos, row in enumerate(rows):
            h = row["canonical_hash"]
            if h in seen_hashes:
                row["hash_seen_before"] = True
                row["hash_first_seen_at"] = seen_hashes[h]
            else:
                row["hash_seen_before"] = False
                row["hash_first_seen_at"] = None
                seen_hashes[h] = int(row["step_idx"])

            score = row["validation_score"]
            if score is not None:
                row["score_delta_from_prev"] = (score - prev_score) if prev_score is not None else None
                if score > best_score:
                    best_score = score
                    steps_since_best = 0
                    row["is_best_so_far"] = True
                else:
                    steps_since_best += 1
                    row["is_best_so_far"] = False
                row["steps_since_improvement"] = steps_since_best
                scores.append((int(row["step_idx"]), float(score)))
                prev_score = score
            else:
                row["score_delta_from_prev"] = None
                row["score_rank_in_run"] = -1
                row["is_best_so_far"] = False
                row["steps_since_improvement"] = steps_since_best

        ranked = {step_idx: rank + 1 for rank, (step_idx, _) in enumerate(
            sorted(scores, key=lambda item: item[1], reverse=True)
        )}
        for row in rows:
            row["score_rank_in_run"] = ranked.get(int(row["step_idx"]), -1)
            row["feature_tier"] = 2
    return feature_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_feature_tier_map(out: Path) -> None:
    payload = {
        "_description": "Feature tier classification for temporal-causality compliance",
        "_tier_1": "Pre-execution: available from code, config, and history before running the step",
        "_tier_2": "Post-execution: requires running the step to observe",
        "tiers": FEATURE_TIERS,
        "derived": {
            "score_rank_pct": {"tier": 2, "source": "score_rank_in_run"},
            "wall_clock_rank_pct": {"tier": 2, "source": "wall_clock_sec"},
            "within_run_z": {"tier": 2, "source": "validation_score"},
        },
    }
    (out / "features" / "feature_tier_map.json").write_text(json.dumps(payload, indent=2))


def write_fault_taxonomy(out: Path, combined: list[dict], records: list[InjectionRecord]) -> None:
    f3_counts = defaultdict(int)
    for r in records:
        if r.fault_class == "F3" and not r.is_natural:
            f3_counts[r.injection_details.get("mutation_strategy", "unknown")] += 1
    payload = {
        "version": "1.0",
        "taxonomy": {
            "F1": {
                "name": "Redundant Configuration Proposal",
                "abbreviation": "RCP",
                "definition": "A step whose canonical configuration hash matches an earlier step in the same AIDE run.",
                "origin": "natural",
                "detection_basis": "Pre-execution (Tier 1): canonical hash comparison against run history.",
                "subtypes": None,
                "count_in_corpus": sum(1 for r in combined if r["F1"]),
            },
            "F2": {
                "name": "Data Leakage",
                "abbreviation": "DL",
                "definition": "Preprocessing or target-aware transformation is fitted before the train-test split.",
                "origin": "natural + synthetic",
                "detection_basis": "Pre-execution (Tier 1): AST analysis of code for fit-before-split pattern.",
                "subtypes": {
                    "natural": "Detected via code analysis in original generated code.",
                    "synthetic": "Injected by inserting target-aware leakage before split.",
                },
                "count_in_corpus": {
                    "natural": sum(1 for r in records if r.fault_class == "F2" and r.is_natural),
                    "synthetic": sum(1 for r in records if r.fault_class == "F2" and not r.is_natural),
                    "total": sum(1 for r in combined if r["F2"]),
                },
            },
            "F3": {
                "name": "Training-Budget Fault",
                "abbreviation": "TBF",
                "definition": "Training hyperparameters are clamped to degenerate values while execution still succeeds.",
                "origin": "synthetic only",
                "detection_basis": "Mixed: config inspection (Tier 1) plus anomalous score/time evidence (Tier 2).",
                "subtypes": dict(f3_counts),
                "count_in_corpus": {**dict(f3_counts), "total": sum(1 for r in combined if r["F3"])},
            },
        },
        "column_convention": "Internal column names remain F1, F2, F3 for code compatibility.",
    }
    (out / "analysis" / "fault_taxonomy.json").write_text(json.dumps(payload, indent=2))


def write_construct_analysis(out: Path, feature_rows: list[dict], combined: list[dict]) -> dict:
    labels = {(r["dataset"], r["step_idx"]): r for r in combined}
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        by_dataset[row["dataset"]].append(row)

    rows: list[dict] = []
    total_rcp = 0
    for dataset, ds_rows in by_dataset.items():
        ds_rows.sort(key=lambda r: int(r["step_idx"]))
        first_seen: dict[str, dict] = {}
        for row in ds_rows:
            label = labels[(row["dataset"], row["step_idx"])]
            h = row["canonical_hash"]
            if label["F1"]:
                total_rcp += 1
                donor = first_seen.get(h)
                if donor is not None:
                    delta = (row["validation_score"] or 0.0) - (donor["validation_score"] or 0.0)
                    rows.append({
                        "dataset": dataset,
                        "rcp_step_idx": row["step_idx"],
                        "donor_step_idx": donor["step_idx"],
                        "canonical_hash": h,
                        "rcp_score": row["validation_score"],
                        "donor_score": donor["validation_score"],
                        "score_delta": delta,
                        "outcome": "improved" if delta > 0.001 else "degraded" if delta < -0.001 else "matched",
                    })
            if h and h not in first_seen:
                first_seen[h] = row

    if rows:
        write_csv(out / "analysis" / "rcp_construct_analysis.csv", rows)
        deltas = [float(r["score_delta"]) for r in rows]
        abs_deltas = [abs(d) for d in deltas]
        summary = {
            "n_rcp_steps_analyzed": len(rows),
            "n_rcp_steps_with_donor": len(rows),
            "n_rcp_total": total_rcp,
            "mean_abs_delta": float(np.mean(abs_deltas)),
            "median_abs_delta": float(np.median(abs_deltas)),
            "std_delta": float(np.std(deltas)),
            "pct_within_001": float(np.mean([d <= 0.001 for d in abs_deltas]) * 100),
            "pct_within_005": float(np.mean([d <= 0.005 for d in abs_deltas]) * 100),
            "pct_improved": float(np.mean([d > 0.001 for d in deltas]) * 100),
            "pct_matched": float(np.mean([abs(d) <= 0.001 for d in deltas]) * 100),
            "pct_degraded": float(np.mean([d < -0.001 for d in deltas]) * 100),
        }
    else:
        (out / "analysis" / "rcp_construct_analysis.csv").write_text(
            "dataset,rcp_step_idx,donor_step_idx,canonical_hash,rcp_score,donor_score,score_delta,outcome\n"
        )
        summary = {
            "n_rcp_steps_analyzed": 0,
            "n_rcp_total": total_rcp,
            "mean_abs_delta": None,
            "note": "No RCP steps had donors under v2 hash.",
        }
    (out / "analysis" / "rcp_construct_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_validation_sample(out: Path, feature_rows: list[dict], combined: list[dict], traces: dict, seed: int) -> None:
    rng = random.Random(seed)
    features = {(r["dataset"], r["step_idx"]): r for r in feature_rows}
    code = {(ds, s.step_idx): s.code for ds, steps in traces.items() for s in steps}
    samples: list[dict] = []

    def add_rows(pool: list[dict], label_name: str) -> None:
        rng.shuffle(pool)
        for row in pool[:5]:
            feat = features[(row["dataset"], row["step_idx"])]
            src = code.get((row["dataset"], row["step_idx"]), "")
            samples.append({
                "dataset": row["dataset"],
                "step_idx": row["step_idx"],
                "fault_class": label_name,
                "canonical_hash": feat["canonical_hash"],
                "validation_score": feat["validation_score"],
                "code_excerpt": "\n".join(src.splitlines()[:50]),
                "label_justification": _justify_label(feat, label_name),
            })

    for fault in FAULT_CLASSES:
        add_rows([r for r in combined if r[fault]], fault)
    add_rows([r for r in combined if not r["any_fault"]], "clean")
    (out / "analysis" / "validation_sample.json").write_text(json.dumps(samples, indent=2))


def _justify_label(row: dict, fault_class: str) -> str:
    if fault_class == "F1":
        return f"hash_seen_before={row['hash_seen_before']}; canonical_hash={row['canonical_hash']}"
    if fault_class == "F2":
        return (
            f"leaky_before_split={row['leaky_before_split']}; "
            f"preprocessing_before_split={row['preprocessing_before_split']}"
        )
    if fault_class == "F3":
        return (
            f"training-budget fault label; wall_clock_sec={row['wall_clock_sec']}; "
            f"validation_score={row['validation_score']}"
        )
    return "No fault criteria met."


def write_manifest(out: Path) -> None:
    manifest = {}
    target = out / "analysis" / "file_manifest.json"
    for f in sorted(out.rglob("*")):
        rel = f.relative_to(out)
        if not f.is_file() or f == target:
            continue
        if (
            "_backup_" in f.name
            or "_backup_" in str(rel)
            or rel.parts[0] in {"reexec_runs", "__pycache__"}
            or rel.parts[0] in {"cp4_results", "cp5_results"}
        ):
            continue
        manifest[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
    target.write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------
def pick_targets(traces, configs, natural, already, fault_class, n_needed, rng):
    """Choose clean steps to receive a synthetic fault, spread across datasets.

    Never selects a buggy step, a step already carrying a natural fault of the
    same class, or a step that already received any synthetic injection.
    """
    by_ds: dict[str, list] = defaultdict(list)
    for ds, steps in traces.items():
        for s in steps:
            if s.is_buggy or s.validation_score is None:
                continue
            key = (ds, s.step_idx)
            if fault_class in natural.get(key, set()):
                continue
            if key in already:
                continue
            cfg = configs[key]
            if cfg is None or not cfg.model_class:
                continue
            if fault_class == "F1" and s.step_idx < 3:
                continue
            if fault_class == "F3":
                low = cfg.model_class.lower()
                if low not in ENSEMBLE_MODELS and low not in ITERATIVE_MODELS and low not in TREE_MODELS:
                    continue
            by_ds[ds].append(s)

    for ds in by_ds:
        rng.shuffle(by_ds[ds])

    # round-robin across datasets so injections spread evenly
    picked: list[tuple[str, object]] = []
    datasets = sorted(by_ds)
    rng.shuffle(datasets)
    i = 0
    while len(picked) < n_needed and any(by_ds[d] for d in datasets):
        d = datasets[i % len(datasets)]
        if by_ds[d]:
            picked.append((d, by_ds[d].pop()))
        i += 1
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="CP-3: Fault Injection & Oracle Detector")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=None,
                        help="Root holding data/ for re-execution (default: corpus-dir's grandparent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-reexec", action="store_true")
    parser.add_argument("--f1-target", type=int, default=60)
    parser.add_argument("--f2-target", type=int, default=40)
    parser.add_argument("--f3-target", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    corpus = Path(args.corpus_dir).resolve()
    summary = Path(args.summary_csv).resolve()
    out = Path(args.output_dir).resolve()
    root = Path(args.project_root).resolve() if args.project_root else corpus.parent.parent
    previous_summary = None
    if (out / "cp3_summary.json").exists():
        try:
            previous_summary = json.loads((out / "cp3_summary.json").read_text())
        except json.JSONDecodeError:
            previous_summary = None

    for sub in ("features", "labels", "oracle", "analysis", "scripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    # -- Phase 1: load ------------------------------------------------------
    print("[1/6] loading traces...")
    traces, meta = load_traces(corpus, summary, log)
    n_steps = sum(len(v) for v in traces.values())
    n_bug = sum(1 for v in traces.values() for s in v if s.is_buggy)
    print(f"      {len(traces)} datasets, {n_steps} steps, {n_steps - n_bug} non-buggy")

    # -- Phase 2: features --------------------------------------------------
    print("[2/6] extracting configs...")
    configs: dict[tuple[str, int], object] = {}
    for ds, steps in traces.items():
        for s in steps:
            if s.is_buggy:
                configs[(ds, s.step_idx)] = None
                continue
            cfg = extract_config(s.code)
            configs[(ds, s.step_idx)] = cfg
            for note in cfg.notes:
                log.append(f"PARSE {ds} step {s.step_idx} [{cfg.parse_quality}]: {note}")

    feature_rows = build_features(traces, configs)
    quality = defaultdict(int)
    for r in feature_rows:
        quality[r["parse_quality"]] += 1
    print(f"      {len(feature_rows)} feature rows | {dict(quality)}")

    with open(out / "features" / "step_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        w.writeheader()
        w.writerows(feature_rows)

    (out / "features" / "feature_metadata.json").write_text(json.dumps({
        "canonical_hash_version": HASH_VERSION,
        "feature_tiers": FEATURE_TIERS,
        "row_count": len(feature_rows),
    }, indent=2))
    write_feature_tier_map(out)

    with open(out / "features" / "step_configs.json", "w") as fh:
        json.dump({
            f"{ds}::{idx}": (None if c is None else {
                "model_class": c.model_class,
                "hyperparameters": _jsonable(c.hyperparameters),
                "preprocessing_chain": c.preprocessing_chain,
                "preprocessing_params": _jsonable(c.preprocessing_params),
                "split_position": c.split_position,
                "split_use_position": c.split_use_position,
                "split_method": c.split_method,
                "split_params": _jsonable(c.split_params),
                "feature_selector": c.feature_selector,
                "feature_var": c.feature_var,
                "target_var": c.target_var,
                "first_fit_position": c.first_fit_position,
                "preprocessing_before_split": c.preprocessing_before_split,
                "leaky_before_split": c.leaky_before_split,
                "parse_quality": c.parse_quality,
                "canonical_hash": canonical_hash(c),
                "notes": c.notes,
            })
            for (ds, idx), c in configs.items()
        }, fh, indent=2)

    # -- Phase 3: natural labels -------------------------------------------
    print("[3/6] natural labelling...")
    natural: dict[tuple[str, int], set[str]] = defaultdict(set)
    natural_json: dict[str, dict[str, list[str]]] = {}
    collisions_rows: list[dict] = []
    rates_rows: list[dict] = []

    for ds, steps in traces.items():
        cfg_list = [configs[(ds, s.step_idx)] for s in steps]
        res = label_dataset(steps, cfg_list)
        per_step: dict[str, list[str]] = {}
        counts = {c: 0 for c in FAULT_CLASSES}

        for i, s in enumerate(steps):
            if s.is_buggy:
                continue
            hits = [c for c in FAULT_CLASSES if res[c][i]]
            for c in hits:
                natural[(ds, s.step_idx)].add(c)
                counts[c] += 1
            if hits:
                per_step[str(s.step_idx)] = hits

        for step_i, donor_j, h in res["f1_collisions"]:
            collisions_rows.append({"dataset": ds, "step_i": step_i, "step_j": donor_j, "hash": h})

        natural_json[ds] = per_step
        n_ok = sum(1 for s in steps if not s.is_buggy)
        rates_rows.append({
            "dataset": ds, "n_steps": len(steps), "n_non_buggy": n_ok,
            **{f"n_{c}": counts[c] for c in FAULT_CLASSES},
            **{f"rate_{c}": round(counts[c] / n_ok, 4) if n_ok else 0.0 for c in FAULT_CLASSES},
        })

    nat_counts = {c: sum(1 for v in natural.values() if c in v) for c in FAULT_CLASSES}
    print(f"      natural: {nat_counts}")

    (out / "labels" / "natural_labels.json").write_text(json.dumps(natural_json, indent=2))
    with open(out / "analysis" / "config_hash_collisions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "step_i", "step_j", "hash"])
        w.writeheader(); w.writerows(collisions_rows)
    with open(out / "analysis" / "natural_fault_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rates_rows[0].keys()))
        w.writeheader(); w.writerows(rates_rows)

    # -- Phase 4: synthetic injection --------------------------------------
    print("[4/6] synthetic injection...")
    records: list[InjectionRecord] = []
    for (ds, idx), classes in natural.items():
        s = next(x for x in traces[ds] if x.step_idx == idx)
        for c in sorted(classes):
            records.append(InjectionRecord(
                dataset=ds, openml_id=int(meta[ds]["openml_id"]), step_idx=idx,
                fault_class=c, is_natural=True,
                original_score=s.validation_score,
                original_code_hash=sha256(s.code),
                injection_details={"source": "natural"},
            ))

    targets = {"F1": args.f1_target, "F2": args.f2_target, "F3": args.f3_target}
    attempted = {c: 0 for c in FAULT_CLASSES}
    succeeded = {c: 0 for c in FAULT_CLASSES}
    crashed = {c: 0 for c in FAULT_CLASSES}
    already: set[tuple[str, int]] = set()
    strategy_cycles = {}
    strategy_attempts = defaultdict(int)

    for cls in FAULT_CLASSES:
        if cls == "F1":
            log.append("INJECT F1: CP-3R keeps F1 natural-only; no synthetic F1 added")
            print(f"      F1: natural {nat_counts[cls]}, CP-3R adds no synthetic F1")
            continue
        need = max(0, targets[cls] - nat_counts[cls])
        if need == 0:
            log.append(f"INJECT {cls}: natural count {nat_counts[cls]} already meets target {targets[cls]}")
            print(f"      {cls}: natural {nat_counts[cls]} >= target {targets[cls]}, no injection needed")
            continue

        # over-provision candidates so failures can be replaced
        cands = pick_targets(traces, configs, natural, already, cls, need * 4, rng)
        print(f"      {cls}: need {need}, {len(cands)} candidates")

        for ds, s in cands:
            if succeeded[cls] >= need:
                break
            key = (ds, s.step_idx)
            if key in already:
                continue
            cfg = configs[key]
            attempted[cls] += 1

            if cls == "F1":
                donors = [
                    x for x in traces[ds]
                    if not x.is_buggy and x.step_idx < s.step_idx and configs[(ds, x.step_idx)]
                ]
                if not donors:
                    log.append(f"SKIP F1 {ds}:{s.step_idx}: no earlier donor")
                    continue
                donor = rng.choice(donors)
                dcfg = configs[(ds, donor.step_idx)]
                records.append(InjectionRecord(
                    dataset=ds, openml_id=int(meta[ds]["openml_id"]), step_idx=s.step_idx,
                    fault_class="F1", is_natural=False,
                    original_score=s.validation_score,
                    injected_score=donor.validation_score,
                    score_delta=None,
                    original_code_hash=sha256(s.code),
                    mutated_code=None,
                    injection_details={
                        "donor_step_idx": donor.step_idx,
                        "canonical_hash": canonical_hash(dcfg),
                        "donor_score": donor.validation_score,
                    },
                    reexec_success=None,
                ))
                already.add(key); succeeded["F1"] += 1
                continue

            if cls == "F3":
                model = cfg.model_class.lower()
                if model not in strategy_cycles:
                    strategy_cycles[model] = itertools.cycle(get_applicable_strategies(model))
                strategy = next(strategy_cycles[model])["name"]
                strategy_attempts[strategy] += 1
                mutated, details = inject_partial_fit(s.code, cfg, strategy=strategy)
            else:
                mutated, details = inject_leakage(s.code, cfg)
            if mutated is None:
                log.append(f"SKIP {cls} {ds}:{s.step_idx}: {details.get('skip_reason')}")
                continue

            if args.skip_reexec:
                records.append(InjectionRecord(
                    dataset=ds, openml_id=int(meta[ds]["openml_id"]), step_idx=s.step_idx,
                    fault_class=cls, is_natural=False,
                    original_score=s.validation_score, injected_score=None, score_delta=None,
                    original_code_hash=sha256(s.code), mutated_code=mutated,
                    injection_details=details, reexec_success=None,
                ))
                already.add(key); succeeded[cls] += 1
                continue

            # Baseline: re-run the UNMUTATED code through the same harness.
            # Comparing against the journal score would fold any difference
            # between AIDE's metric capture and ours into the delta; measuring
            # both sides here makes the delta a pure effect of the mutation.
            base = reexecute_step(s.code, ds, meta[ds].get("group", ""), root, args.timeout, args.python_exe, corpus_dir=corpus)
            if not base["success"]:
                log.append(f"BASELINE-FAIL {cls} {ds}:{s.step_idx}: {base['error']}")
                continue

            r = reexecute_step(mutated, ds, meta[ds].get("group", ""), root, args.timeout, args.python_exe, corpus_dir=corpus)
            if not r["success"]:
                log.append(f"REEXEC-FAIL {cls} {ds}:{s.step_idx}: {r['error']}")
                continue

            # --- Catastrophic-failure filter --------------------------------
            # If the injected code produced a near-random or below-random
            # score, the injection broke the pipeline rather than introducing
            # a subtle fault.  The step IS faulty (the mutation happened) but
            # the delta is not a valid measurement of the fault's effect.
            viable, crash_reason = is_viable_injection(base["score"], r["score"])
            if not viable:
                crashed[cls] += 1
                log.append(
                    f"CRASH {cls} {ds}:{s.step_idx}: {crash_reason} "
                    f"(baseline={base['score']:.4f}, injected={r['score']:.4f})"
                )
                records.append(InjectionRecord(
                    dataset=ds, openml_id=int(meta[ds]["openml_id"]), step_idx=s.step_idx,
                    fault_class=cls, is_natural=False,
                    original_score=s.validation_score,
                    baseline_reexec_score=base["score"],
                    injected_score=r["score"],
                    score_delta=r["score"] - base["score"],
                    original_code_hash=sha256(s.code), mutated_code=mutated,
                    injection_details={**details, "crash_reason": crash_reason},
                    reexec_success=True,
                    reexec_viable=False,
                    reexec_wall_clock_sec=(r["wall_clock_sec"] + base["wall_clock_sec"]),
                ))
                # Still counts as a succeeded injection (the step IS faulty)
                # but the delta is excluded from inflation/degradation stats.
                already.add(key); succeeded[cls] += 1
                print(f"        {cls} {ds}:{s.step_idx} CRASH delta={r['score'] - base['score']:+.4f} [{crash_reason}]")
                continue

            delta = r["score"] - base["score"]
            records.append(InjectionRecord(
                dataset=ds, openml_id=int(meta[ds]["openml_id"]), step_idx=s.step_idx,
                fault_class=cls, is_natural=False,
                original_score=s.validation_score,
                baseline_reexec_score=base["score"],
                injected_score=r["score"], score_delta=delta,
                original_code_hash=sha256(s.code), mutated_code=mutated,
                injection_details=details, reexec_success=True,
                reexec_viable=True,
                reexec_wall_clock_sec=(r["wall_clock_sec"] + base["wall_clock_sec"]),
            ))
            already.add(key); succeeded[cls] += 1
            print(f"        {cls} {ds}:{s.step_idx} delta={delta:+.4f}" if delta is not None
                  else f"        {cls} {ds}:{s.step_idx}")

    print(f"      attempted={dict(attempted)} succeeded={dict(succeeded)} crashed={dict(crashed)}")

    (out / "labels" / "injection_manifest.json").write_text(
        json.dumps([r.to_dict() for r in records], indent=2)
    )

    print("[5/9] reconciling mutated-code features...")
    feature_rows = reconcile_features_after_injection(feature_rows, records)
    write_csv(out / "features" / "step_features.csv", feature_rows)

    for cls in ("F2", "F3"):
        rows = [
            {"dataset": r.dataset, "step_idx": r.step_idx,
             "original_score": r.original_score,
             "baseline_reexec_score": r.baseline_reexec_score,
             "injected_score": r.injected_score, "delta": r.score_delta,
             "viable": r.reexec_viable,
             **({"mutation_strategy": r.injection_details["mutation_strategy"]} if cls == "F3" else {})}
            for r in records if r.fault_class == cls and not r.is_natural and r.score_delta is not None
        ]
        with open(out / "analysis" / f"{cls.lower()}_score_deltas.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "step_idx", "original_score",
                                               "baseline_reexec_score", "injected_score",
                                               "delta", "viable"] + (["mutation_strategy"] if cls == "F3" else []))
            w.writeheader(); w.writerows(rows)

    # -- Phase 6: oracle ----------------------------------------------------
    print("[6/9] building oracle...")
    detector = OracleDetector.from_manifest(records)
    all_steps = [(ds, s.step_idx) for ds, steps in traces.items() for s in steps if not s.is_buggy]
    ev = detector.evaluate(all_steps)
    savings = compute_cost_savings(traces, detector._labels, meta)

    def delta_stats(cls: str, viable_only: bool = True) -> dict:
        """Compute score-delta statistics for a fault class.

        When viable_only=True (default), exclude catastrophic pipeline
        failures so the statistic measures the fault's actual effect.
        """
        ds_ = [
            r.score_delta for r in records
            if r.fault_class == cls
            and not r.is_natural
            and r.score_delta is not None
            and (not viable_only or r.reexec_viable is not False)
        ]
        n_crashed = sum(
            1 for r in records
            if r.fault_class == cls
            and not r.is_natural
            and r.reexec_viable is False
        )
        if not ds_:
            return {"mean_delta": None, "median_delta": None, "max_delta": None,
                    "min_delta": None, "n_measured": 0, "n_crashed": n_crashed}
        sd = sorted(ds_)
        return {
            "mean_delta": sum(ds_) / len(ds_),
            "median_delta": sd[len(sd) // 2],
            "max_delta": max(ds_), "min_delta": min(ds_), "n_measured": len(ds_),
            "n_crashed": n_crashed,
        }

    label_sources = {
        c: {
            "natural": sum(1 for r in records if r.fault_class == c and r.is_natural),
            "synthetic": sum(1 for r in records if r.fault_class == c and not r.is_natural),
        } for c in FAULT_CLASSES
    }
    per_strategy = {}
    for strategy in F3_STRATEGIES:
        name = strategy["name"]
        rs = [r for r in records if r.fault_class == "F3" and not r.is_natural
              and r.injection_details.get("mutation_strategy") == name]
        deltas = [r.score_delta for r in rs if r.score_delta is not None and r.reexec_viable is not False]
        per_strategy[name] = {"attempted": strategy_attempts[name], "succeeded": len(rs),
                              "crashed": sum(r.reexec_viable is False for r in rs),
                              "n_measured": len(deltas),
                              "mean_delta": sum(deltas)/len(deltas) if deltas else None}

    evaluation = {
        **ev,
        "cost_savings": savings,
        "label_sources": label_sources,
        "f2_score_inflation": delta_stats("F2"),
        "f3_score_degradation": delta_stats("F3"),
    }
    (out / "oracle" / "oracle_evaluation.json").write_text(json.dumps(evaluation, indent=2))
    shutil.copy2(Path(__file__).parent / "oracle.py", out / "oracle" / "oracle_detector.py")

    # -- Phase 7: combined labels, chart, summary ---------------------------
    print("[7/9] writing artifacts...")
    combined: list[dict] = []
    syn_map: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in records:
        if not r.is_natural:
            syn_map[(r.dataset, r.step_idx)].add(r.fault_class)

    for ds, idx in all_steps:
        nat = natural.get((ds, idx), set())
        syn = syn_map.get((ds, idx), set())
        allc = nat | syn
        source = "clean" if not allc else (
            "natural" if not syn else ("synthetic" if not nat else "both")
        )
        combined.append({
            "dataset": ds, "step_idx": idx,
            **{c: (c in allc) for c in FAULT_CLASSES},
            "any_fault": bool(allc), "source": source,
        })
    with open(out / "labels" / "combined_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
        w.writeheader(); w.writerows(combined)

    construct_summary = write_construct_analysis(out, feature_rows, combined)
    write_fault_taxonomy(out, combined, records)
    write_validation_sample(out, feature_rows, combined, traces, args.seed)

    chart_written = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dss = sorted(traces)
        counts = {c: [sum(1 for r in combined if r["dataset"] == d and r[c]) for d in dss]
                  for c in FAULT_CLASSES}
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = [0] * len(dss)
        for c, colour in zip(FAULT_CLASSES, ("#4C72B0", "#DD8452", "#55A868")):
            ax.bar(dss, counts[c], bottom=bottom, label=c, color=colour)
            bottom = [b + v for b, v in zip(bottom, counts[c])]
        ax.set_ylabel("labelled faulty steps")
        ax.set_title("CP-3 fault distribution by dataset (natural + synthetic)")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(out / "analysis" / "fault_distribution.png", dpi=120)
        plt.close(fig)
        chart_written = True
    except Exception as exc:  # noqa: BLE001
        log.append(f"CHART skipped: {type(exc).__name__}: {exc}")

    n_ok_total = len(all_steps)
    parse_total = sum(quality.values())
    summary_obj = {
        "corpus": {
            "datasets": len(traces), "total_steps": n_steps,
            "non_buggy_steps": n_steps - n_bug, "buggy_steps": n_bug,
        },
        "feature_extraction": {
            "canonical_hash_version": HASH_VERSION,
            "parsed_full_ast": quality.get("full_ast", 0),
            "parsed_partial_ast": quality.get("partial_ast", 0),
            "parsed_regex_fallback": quality.get("regex_fallback", 0),
            "unparseable": quality.get("unparseable", 0),
            "parse_success_rate": round(
                (parse_total - quality.get("unparseable", 0)) / parse_total, 4) if parse_total else 0.0,
        },
        "natural_faults": {
            **{c: {"count": nat_counts[c], "rate": round(nat_counts[c] / n_ok_total, 4)}
               for c in FAULT_CLASSES},
            "any": {"count": len(natural), "rate": round(len(natural) / n_ok_total, 4)},
        },
        "synthetic_injections": {
            "F1": {"attempted": attempted["F1"], "succeeded": succeeded["F1"], "cp3r_policy": "natural_only"},
            "F2": {"attempted": attempted["F2"], "succeeded": succeeded["F2"],
                   "crashed": crashed["F2"],
                   "mean_score_delta": delta_stats("F2")["mean_delta"]},
            "F3": {"attempted": attempted["F3"], "succeeded": succeeded["F3"],
                   "crashed": crashed["F3"],
                   "mean_score_delta": delta_stats("F3")["mean_delta"],
                   "per_strategy": per_strategy},
        },
        "combined": {
            "total_labeled_faulty": sum(1 for r in combined if r["any_fault"]),
            "total_clean": sum(1 for r in combined if not r["any_fault"]),
            "combined_fault_rate": round(
                sum(1 for r in combined if r["any_fault"]) / n_ok_total, 4),
        },
        "oracle": {
            "cost_savings_pct": round(savings["savings_pct"], 4),
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        },
        "chart_written": chart_written,
        "seed": args.seed,
        "skip_reexec": args.skip_reexec,
        "runtime_sec": round(time.time() - t0, 2),
    }
    (out / "cp3_summary.json").write_text(json.dumps(summary_obj, indent=2))

    cp3r_summary = {
        "checkpoint": "CP-3R",
        "status": "completed",
        "seed": args.seed,
        "canonical_hash_version": HASH_VERSION,
        "repo_adaptations": {
            "corpus_dir": str(corpus),
            "summary_csv": str(summary),
            "labels_file": "labels/combined_labels.csv",
            "manifest_schema": "flat_list",
        },
        "f1_count_comparison": {
            "pre_cp3r_natural_f1": (
                previous_summary.get("natural_faults", {}).get("F1", {}).get("count")
                if previous_summary else None
            ),
            "cp3r_v2_natural_f1": nat_counts["F1"],
            "synthetic_f1_added": 0,
        },
        "fault_counts": {
            "natural": label_sources,
            "combined": {
                "F1": sum(1 for r in combined if r["F1"]),
                "F2": sum(1 for r in combined if r["F2"]),
                "F3": sum(1 for r in combined if r["F3"]),
                "any_fault": sum(1 for r in combined if r["any_fault"]),
                "clean": sum(1 for r in combined if not r["any_fault"]),
            },
        },
        "construct_analysis": construct_summary,
        "artifacts": {
            "feature_tier_map": "features/feature_tier_map.json",
            "feature_metadata": "features/feature_metadata.json",
            "fault_taxonomy": "analysis/fault_taxonomy.json",
            "rcp_construct_summary": "analysis/rcp_construct_summary.json",
            "validation_sample": "analysis/validation_sample.json",
            "file_manifest": "analysis/file_manifest.json",
        },
        "runtime_sec": summary_obj["runtime_sec"],
        "api_cost_usd": 0,
    }
    (out / "cp3r_summary.json").write_text(json.dumps(cp3r_summary, indent=2))

    # human-readable oracle report
    cs = savings
    f2_stats = delta_stats("F2")
    f3_stats = delta_stats("F3")
    report = f"""# CP-3 Oracle Detector - Summary

**Corpus**: {len(traces)} datasets, {n_steps} steps ({n_steps - n_bug} non-buggy, {n_bug} buggy)
**Seed**: {args.seed} | **Runtime**: {summary_obj['runtime_sec']}s | **Re-execution**: {'skipped' if args.skip_reexec else 'enabled'}

## Labels

| Class | Natural | Synthetic | Total |
|-------|---------|-----------|-------|
| F1 stale memory | {label_sources['F1']['natural']} | {label_sources['F1']['synthetic']} | {sum(label_sources['F1'].values())} |
| F2 data leakage | {label_sources['F2']['natural']} | {label_sources['F2']['synthetic']} | {sum(label_sources['F2'].values())} |
| F3 partial fit | {label_sources['F3']['natural']} | {label_sources['F3']['synthetic']} | {sum(label_sources['F3'].values())} |

Faulty steps: **{summary_obj['combined']['total_labeled_faulty']}** of {n_ok_total} non-buggy
(**{summary_obj['combined']['combined_fault_rate']:.1%}**); clean: {summary_obj['combined']['total_clean']}.

## Oracle performance

Precision / recall / F1 = **1.0** for every class, by construction. This is the
ceiling: no CP-4 detector can exceed it.

## Compute-savings ceiling

A perfect gate skipping every faulty trial would save:

- **{cs['savings_pct']:.2f}%** of non-buggy wall clock
  ({cs['faulty_wall_clock_sec']:.0f}s of {cs['total_wall_clock_sec']:.0f}s)
- ~{cs['total_tokens_saved']:,} tokens, ~${cs['total_cost_saved_usd']:.2f} API cost (apportioned by wall clock)
- {cs['n_faulty_steps']} of {cs['n_total_steps']} trials gated

## Signal strength

- **F2 score inflation** (synthetic, viable only): mean {f2_stats['mean_delta']}, n={f2_stats['n_measured']} (excluded {f2_stats['n_crashed']} pipeline crashes)
- **F3 score degradation** (synthetic, viable only): mean {f3_stats['mean_delta']}, n={f3_stats['n_measured']} (excluded {f3_stats['n_crashed']} pipeline crashes)

## Catastrophic-Failure Filter

The original viability thresholds are unchanged. Per-strategy means exclude non-viable results;
counts include all successfully executed mutations. Assignment cycles per model class, not by outcome.

Per-strategy breakdown: {json.dumps(per_strategy, sort_keys=True)}
"""
    (out / "oracle" / "oracle_report.md").write_text(report)

    (out / "features" / "extraction_log.txt").write_text("\n".join(log) + "\n")

    print("[8/9] writing file manifest...")
    write_manifest(out)

    print(f"\ndone in {summary_obj['runtime_sec']}s -> {out}")
    print(f"  faulty {summary_obj['combined']['total_labeled_faulty']}/{n_ok_total} "
          f"({summary_obj['combined']['combined_fault_rate']:.1%}) | "
          f"savings ceiling {savings['savings_pct']:.2f}%")
    if any(crashed[c] for c in FAULT_CLASSES):
        print(f"  pipeline crashes (excluded from delta stats): {dict(crashed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
