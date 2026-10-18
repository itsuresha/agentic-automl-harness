"""Construct analysis for F1/RCP (Redundant Configuration Proposal)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def run_construct_analysis(features_path: Path, labels_path: Path, output_dir: Path):
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    df = features.merge(labels, on=["dataset", "step_idx"])

    results = []
    total_rcp = int(df["F1"].sum())
    for dataset, group in df.sort_values(["dataset", "step_idx"]).groupby("dataset"):
        first_seen = {}
        for _, row in group.iterrows():
            h = row["canonical_hash"]
            if bool(row["F1"]):
                donor = first_seen.get(h)
                if donor is not None:
                    score_delta = row["validation_score"] - donor["validation_score"]
                    results.append({
                        "dataset": dataset,
                        "rcp_step_idx": int(row["step_idx"]),
                        "donor_step_idx": int(donor["step_idx"]),
                        "canonical_hash": h,
                        "rcp_score": row["validation_score"],
                        "donor_score": donor["validation_score"],
                        "score_delta": score_delta,
                        "outcome": (
                            "improved" if score_delta > 0.001
                            else "degraded" if score_delta < -0.001
                            else "matched"
                        ),
                    })
            if h not in first_seen:
                first_seen[h] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "dataset", "rcp_step_idx", "donor_step_idx", "canonical_hash",
            "rcp_score", "donor_score", "score_delta", "outcome",
        ])
    results_df.to_csv(output_dir / "rcp_construct_analysis.csv", index=False)

    if len(results_df) > 0:
        deltas = results_df["score_delta"].to_numpy()
        summary = {
            "n_rcp_steps_analyzed": int(len(results_df)),
            "n_rcp_steps_with_donor": int(len(results_df)),
            "n_rcp_total": total_rcp,
            "mean_abs_delta": float(np.mean(np.abs(deltas))),
            "median_abs_delta": float(np.median(np.abs(deltas))),
            "std_delta": float(np.std(deltas)),
            "pct_within_001": float(np.mean(np.abs(deltas) <= 0.001) * 100),
            "pct_within_005": float(np.mean(np.abs(deltas) <= 0.005) * 100),
            "pct_improved": float(np.mean(deltas > 0.001) * 100),
            "pct_matched": float(np.mean(np.abs(deltas) <= 0.001) * 100),
            "pct_degraded": float(np.mean(deltas < -0.001) * 100),
        }
    else:
        summary = {
            "n_rcp_steps_analyzed": 0,
            "n_rcp_total": total_rcp,
            "mean_abs_delta": None,
            "note": "No RCP steps had donors under v2 hash.",
        }

    (output_dir / "rcp_construct_summary.json").write_text(json.dumps(summary, indent=2))
    return results_df, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    _, summary = run_construct_analysis(
        Path(args.features), Path(args.labels), Path(args.output_dir)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
