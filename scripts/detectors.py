"""CP-4R tiered practical fault detectors.

Gate detectors use only Tier 1 pre-execution features. Monitor detectors may
also use Tier 2 post-execution features. This module is read-only with respect
to CP-3R artifacts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

TIER1_FEATURES = [
    "n_hyperparams",
    "n_preprocessing_steps",
    "preprocessing_before_split",
    "leaky_before_split",
    "hash_seen_before",
    "hash_first_seen_at",
    "parse_quality_full_ast",
]

TIER2_FEATURES_BASE = [
    "validation_score",
    "wall_clock_sec",
    "score_delta_from_prev",
    "score_rank_in_run",
    "is_best_so_far",
    "steps_since_improvement",
]

TIER2_DERIVED = [
    "score_rank_pct",
    "wall_clock_rank_pct",
    "within_run_z",
]

ALL_FEATURES = TIER1_FEATURES + TIER2_FEATURES_BASE + TIER2_DERIVED

LGBM_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    min_child_samples=5,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
    n_jobs=1,
)

SCORE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
F2_EFFECT_BINS = {
    "negligible": (None, 0.01),
    "small": (0.01, 0.03),
    "medium": (0.03, 0.06),
    "large": (0.06, None),
}


def _coerce_binary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def load_data(output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    features = pd.read_csv(out / "features" / "step_features.csv")
    labels = pd.read_csv(out / "labels" / "combined_labels.csv")

    if "step_id" in features.columns and "step_idx" not in features.columns:
        features = features.rename(columns={"step_id": "step_idx"})
    if "step_id" in labels.columns and "step_idx" not in labels.columns:
        labels = labels.rename(columns={"step_id": "step_idx"})

    df = features.merge(labels, on=["dataset", "step_idx"], how="inner")
    if len(df) != len(features) or len(df) != len(labels):
        raise ValueError(
            f"Merge mismatch: features={len(features)}, labels={len(labels)}, merged={len(df)}"
        )

    df = df.sort_values(["dataset", "step_idx"]).reset_index(drop=True)
    df["parse_quality_full_ast"] = (df["parse_quality"] == "full_ast").astype(int)
    for col in [
        "preprocessing_before_split",
        "leaky_before_split",
        "is_best_so_far",
        "hash_seen_before",
        "F1",
        "F2",
        "F3",
        "any_fault",
    ]:
        df[col] = _coerce_binary(df[col])
    df["hash_first_seen_at"] = pd.to_numeric(df["hash_first_seen_at"], errors="coerce").fillna(-1)
    for col in ["validation_score", "wall_clock_sec", "score_delta_from_prev", "score_rank_in_run"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # CP-3R preserves F1/RCP as the definitional policy label even when later
    # synthetic F2/F3 reconciliation changes the row's current code hash. CP-4R
    # evaluates D1 as that policy rule, so keep the in-memory rule feature
    # aligned with the CP-3R RCP label without mutating CP-3R artifacts.
    df["hash_seen_before"] = np.maximum(df["hash_seen_before"].values, df["F1"].values)
    return _add_derived_features(df)


def verify_tier_map(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "features" / "feature_tier_map.json"
    if not path.exists():
        print(f"  WARNING: {path} not found; using detector constants")
        return {}
    data = json.loads(path.read_text())
    tiers = data.get("tiers", {})
    print(f"  Loaded tier map: {len(tiers)} features")
    return data


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["score_rank_pct"] = 0.0
    df["wall_clock_rank_pct"] = 0.0
    df["within_run_z"] = 0.0
    for _, group in df.groupby("dataset", sort=False):
        idx = group.sort_values("step_idx").index
        n = len(idx)
        df.loc[idx, "score_rank_pct"] = df.loc[idx, "score_rank_in_run"] / n
        df.loc[idx, "wall_clock_rank_pct"] = df.loc[idx, "wall_clock_sec"].rank(pct=True)
        prior_scores: list[float] = []
        for row_idx in idx:
            score = float(df.loc[row_idx, "validation_score"])
            if len(prior_scores) >= 2:
                sigma = float(np.std(prior_scores))
                df.loc[row_idx, "within_run_z"] = (
                    (score - float(np.mean(prior_scores))) / sigma if sigma > 1e-10 else 0.0
                )
            prior_scores.append(score)
    return df


def detect_f1(df: pd.DataFrame) -> np.ndarray:
    return df["hash_seen_before"].values.astype(int)


def d1_tautology_disclosure(output_dir: str | Path) -> dict:
    disclosure = {
        "status": "acknowledged",
        "explanation": (
            "D1 uses hash_seen_before to detect F1/RCP. This feature is the "
            "definition of F1: a step is labeled F1 iff its canonical hash was "
            "seen in an earlier step. D1's perfect P/R/F1 is therefore by construction."
        ),
        "paper_framing": (
            "Report D1 as a definitional Tier 1 rule, with CP-3R construct "
            "analysis as independent evidence for RCP wastefulness."
        ),
    }
    path = Path(output_dir) / "analysis" / "rcp_construct_summary.json"
    if path.exists():
        disclosure["independent_evidence"] = {
            "source": "CP-3R construct analysis",
            "file": "analysis/rcp_construct_analysis.csv",
            "summary_file": "analysis/rcp_construct_summary.json",
            "construct_summary": json.loads(path.read_text()),
        }
    else:
        disclosure["independent_evidence"] = {"status": "missing"}
    return disclosure


def detect_f2_gate(df: pd.DataFrame) -> np.ndarray:
    return df["leaky_before_split"].values.astype(int)


def detect_f2_monitor(df: pd.DataFrame, z_threshold: float) -> np.ndarray:
    structural = detect_f2_gate(df)
    statistical = (df["within_run_z"] > z_threshold).astype(int).values
    return np.maximum(structural, statistical)


def tune_f2_threshold(df_train: pd.DataFrame, y_train: np.ndarray) -> float:
    best_f1 = -1.0
    best_t = 1.5
    for threshold in np.arange(0.5, 3.5, 0.1):
        pred = detect_f2_monitor(df_train, float(threshold))
        score = f1_score(y_train, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(threshold)
    return best_t


def get_feature_matrix(df: pd.DataFrame, tier: str = "all") -> pd.DataFrame:
    cols = TIER1_FEATURES if tier == "gate" else ALL_FEATURES
    return df[list(cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def lodo_evaluate(
    df: pd.DataFrame,
    target_col: str,
    detector_type: str,
    tier: str = "all",
    manifest_path: str | Path | None = None,
) -> dict:
    logo = LeaveOneGroupOut()
    groups = df["dataset"]
    n = len(df)
    oof_pred = np.zeros(n, dtype=int)
    oof_prob = np.zeros(n, dtype=float)
    folds: list[dict] = []
    f3_strategy_map = _load_f3_strategy_map(manifest_path) if target_col == "F3" else {}

    for train_idx, test_idx in logo.split(df, df[target_col], groups):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        y_train = df[target_col].iloc[train_idx].values
        y_test = df[target_col].iloc[test_idx].values
        dataset = str(df_test["dataset"].iloc[0])

        if detector_type == "rule_f1":
            pred = detect_f1(df_test)
            prob = pred.astype(float)
        elif detector_type == "gate_f2":
            pred = detect_f2_gate(df_test)
            prob = pred.astype(float)
        elif detector_type == "monitor_f2":
            threshold = tune_f2_threshold(df_train, y_train)
            pred = detect_f2_monitor(df_test, threshold)
            prob = np.maximum(
                df_test["leaky_before_split"].values.astype(float),
                np.clip(df_test["within_run_z"].values / 3.0, 0, 1),
            )
        elif detector_type == "learned":
            X_train, X_test = _learned_matrices(df_train, df_test, target_col, tier)
            model = lgbm.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train, y_train)
            if len(getattr(model, "classes_", [])) == 1:
                pred = np.full(len(test_idx), int(model.classes_[0]), dtype=int)
                prob = pred.astype(float)
            else:
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        oof_pred[test_idx] = pred
        oof_prob[test_idx] = prob
        folds.append(_fold_metrics(dataset, y_test, pred, prob))

    result = _aggregate_result(df, target_col, detector_type, tier, oof_pred, oof_prob, folds)
    if f3_strategy_map:
        result["per_strategy_recall"] = _strategy_recall(df, f3_strategy_map, oof_pred)
    return result


def _learned_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    tier: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = get_feature_matrix(df_train, tier=tier)
    X_test = get_feature_matrix(df_test, tier=tier)
    if target_col == "any_fault":
        X_train["d1_prediction"] = detect_f1(df_train)
        X_test["d1_prediction"] = detect_f1(df_test)
        X_train["d2_gate_prediction"] = detect_f2_gate(df_train)
        X_test["d2_gate_prediction"] = detect_f2_gate(df_test)
        if tier != "gate":
            threshold = tune_f2_threshold(df_train, df_train["F2"].values)
            X_train["d2_monitor_prediction"] = detect_f2_monitor(df_train, threshold)
            X_test["d2_monitor_prediction"] = detect_f2_monitor(df_test, threshold)
    return X_train, X_test


def _fold_metrics(dataset: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "dataset": dataset,
        "n_test": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auroc": round(float(auc), 4) if not np.isnan(auc) else None,
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def _aggregate_result(
    df: pd.DataFrame,
    target_col: str,
    detector_type: str,
    tier: str,
    pred: np.ndarray,
    prob: np.ndarray,
    folds: list[dict],
) -> dict:
    y = df[target_col].values
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    return {
        "target": target_col,
        "detector": detector_type,
        "tier": tier,
        "aggregate": {
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "auroc": round(float(auc), 4) if not np.isnan(auc) else None,
            "avg_precision": round(float(ap), 4) if not np.isnan(ap) else None,
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        },
        "per_fold": folds,
        "oof_y_pred": pred.tolist(),
        "oof_y_prob": prob.tolist(),
        "oof_predictions": [
            {
                "dataset": str(row.dataset),
                "step_idx": int(row.step_idx),
                "y_true": int(true),
                "y_pred": int(p),
                "y_prob": round(float(pr), 6),
            }
            for row, true, p, pr in zip(df.itertuples(index=False), y, pred, prob)
        ],
    }


def _load_f3_strategy_map(path: str | Path | None) -> dict[tuple[str, int], str]:
    if not path or not Path(path).exists():
        return {}
    manifest = json.loads(Path(path).read_text())
    result = {}
    for record in manifest:
        fault_class = record.get("fault_class", record.get("fault_type"))
        if fault_class == "F3" and not record.get("is_natural", True):
            result[(str(record["dataset"]), int(record["step_idx"]))] = (
                record.get("injection_details", {}).get("mutation_strategy", "unknown")
            )
    return result


def _strategy_recall(
    df: pd.DataFrame,
    strategy_map: dict[tuple[str, int], str],
    pred: np.ndarray,
) -> dict:
    result: dict[str, dict] = {}
    for i, row in enumerate(df.itertuples(index=False)):
        key = (str(row.dataset), int(row.step_idx))
        if key not in strategy_map or int(row.F3) != 1:
            continue
        strategy = strategy_map[key]
        result.setdefault(strategy, {"tp": 0, "fn": 0})
        if pred[i] == 1:
            result[strategy]["tp"] += 1
        else:
            result[strategy]["fn"] += 1
    for counts in result.values():
        support = counts["tp"] + counts["fn"]
        counts["support"] = support
        counts["recall"] = round(counts["tp"] / support, 4) if support else 0.0
    return result


def savings_analysis(df: pd.DataFrame, gate_preds: np.ndarray, monitor_preds: np.ndarray) -> dict:
    y = df["any_fault"].values
    wc = df["wall_clock_sec"].values
    total_wc = float(wc.sum())
    gate_tp = (gate_preds == 1) & (y == 1)
    gate_fp = (gate_preds == 1) & (y == 0)
    monitor_only_tp = (monitor_preds == 1) & (gate_preds == 0) & (y == 1)
    oracle = wc[y == 1].sum()
    saved = wc[gate_tp].sum()
    return {
        "tier1_gate": {
            "description": "Correctly flagged pre-execution trials; genuine skip savings.",
            "true_positives": int(gate_tp.sum()),
            "false_positives": int(gate_fp.sum()),
            "wall_clock_saved_sec": round(float(saved), 2),
            "wall_clock_wasted_sec": round(float(wc[gate_fp].sum()), 2),
            "savings_pct": round(float(saved / total_wc * 100), 2) if total_wc else 0.0,
        },
        "tier2_monitor_additional": {
            "description": "Additional faulty trials caught post-execution; no direct skip savings.",
            "additional_true_positives": int(monitor_only_tp.sum()),
            "wall_clock_of_flagged_sec": round(float(wc[monitor_only_tp].sum()), 2),
        },
        "combined_detection": {
            "total_true_positives": int(gate_tp.sum() + monitor_only_tp.sum()),
            "total_faulty": int(y.sum()),
            "combined_recall": round(float((gate_tp.sum() + monitor_only_tp.sum()) / max(y.sum(), 1)), 4),
            "tier1_genuine_savings_pct": round(float(saved / total_wc * 100), 2) if total_wc else 0.0,
            "oracle_ceiling_pct": round(float(oracle / total_wc * 100), 2) if total_wc else 0.0,
        },
        "total_wall_clock_sec": round(total_wc, 2),
    }


def f2_effect_size_analysis(df: pd.DataFrame, f2_deltas_path: str | Path | None = None) -> dict:
    f2_steps = df[df["F2"] == 1].copy()
    delta_map = {}
    if f2_deltas_path and Path(f2_deltas_path).exists():
        deltas = pd.read_csv(f2_deltas_path)
        delta_map = {
            (str(row["dataset"]), int(row["step_idx"])): abs(float(row["delta"]))
            for _, row in deltas.iterrows()
        }
    effects = []
    sources = []
    for _, row in f2_steps.iterrows():
        key = (str(row["dataset"]), int(row["step_idx"]))
        if key in delta_map:
            effects.append(delta_map[key])
            sources.append("reexec")
        else:
            effects.append(abs(float(row["score_delta_from_prev"])))
            sources.append("proxy")
    f2_steps["effect_size"] = effects
    f2_steps["effect_source"] = sources

    result = {}
    for name, (lo, hi) in F2_EFFECT_BINS.items():
        mask = pd.Series(True, index=f2_steps.index)
        if lo is not None:
            mask &= f2_steps["effect_size"] >= lo
        if hi is not None:
            mask &= f2_steps["effect_size"] < hi
        subset = f2_steps[mask]
        result[name] = {
            "count": int(len(subset)),
            "n_reexec": int((subset["effect_source"] == "reexec").sum()),
            "n_proxy": int((subset["effect_source"] == "proxy").sum()),
            "datasets": sorted(str(v) for v in subset["dataset"].unique().tolist()),
        }
    result["undetectable_region"] = {
        "description": (
            "F2 leakage with observed effect < 0.01 is a negligible-effect region; "
            "statistical score-inflation detection is unreliable there."
        ),
        "threshold": 0.01,
    }
    return result


def score_threshold_baseline(df: pd.DataFrame) -> dict:
    y = df["any_fault"].values
    result = {}
    for threshold in SCORE_THRESHOLDS:
        pred = (df["validation_score"] < threshold).astype(int).values
        result[str(threshold)] = {
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "n_flagged": int(pred.sum()),
        }
    return result


def per_target_random_baseline(df: pd.DataFrame) -> dict:
    n = len(df)
    result = {}
    for target in ["F1", "F2", "F3", "any_fault"]:
        p = float(df[target].sum()) / n
        result[target] = {
            "n_positive": int(df[target].sum()),
            "n_total": n,
            "fault_rate": round(p, 4),
            "expected_precision": round(p, 4),
            "expected_recall": round(p, 4),
            "expected_f1": round(p, 4),
        }
    return result


def feature_importance_analysis(
    df: pd.DataFrame,
    target_col: str,
    tier: str = "all",
    include_d_any_meta: bool = False,
) -> dict:
    X = get_feature_matrix(df, tier=tier)
    if include_d_any_meta:
        X["d1_prediction"] = detect_f1(df)
        X["d2_gate_prediction"] = detect_f2_gate(df)
        if tier != "gate":
            threshold = tune_f2_threshold(df, df["F2"].values)
            X["d2_monitor_prediction"] = detect_f2_monitor(df, threshold)
    y = df[target_col].values
    model = lgbm.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)
    importances = dict(zip(X.columns, [int(v) for v in model.feature_importances_]))
    return {
        "target": target_col,
        "tier": tier,
        "importances": dict(sorted(importances.items(), key=lambda item: -item[1])),
    }
