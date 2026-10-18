"""CP-3 oracle detector.

The perfect (upper-bound) detector: it reads ground-truth labels, so its
precision/recall/F1 are 1.0 by construction. Its value is the ceiling it
establishes - notably the fraction of compute a perfect gate would save,
which bounds what any practical CP-4/CP-5 detector can achieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

FAULT_CLASSES = ("F1", "F2", "F3")


@dataclass
class InjectionRecord:
    dataset: str
    openml_id: int
    step_idx: int
    fault_class: str
    is_natural: bool
    original_score: float | None = None          # as recorded in the AIDE journal
    baseline_reexec_score: float | None = None   # unmutated code re-run in THIS harness
    injected_score: float | None = None
    score_delta: float | None = None
    original_code_hash: str = ""
    mutated_code: str | None = None
    injection_details: dict = field(default_factory=dict)
    reexec_success: bool | None = None
    reexec_viable: bool | None = None            # False = catastrophic pipeline crash
    reexec_wall_clock_sec: float | None = None

    def to_dict(self, include_code: bool = True) -> dict[str, Any]:
        d = {
            "dataset": self.dataset,
            "openml_id": self.openml_id,
            "step_idx": self.step_idx,
            "fault_class": self.fault_class,
            "is_natural": self.is_natural,
            "original_score": self.original_score,
            "baseline_reexec_score": self.baseline_reexec_score,
            "injected_score": self.injected_score,
            "score_delta": self.score_delta,
            "original_code_hash": self.original_code_hash,
            "injection_details": self.injection_details,
            "reexec_success": self.reexec_success,
            "reexec_viable": self.reexec_viable,
            "reexec_wall_clock_sec": self.reexec_wall_clock_sec,
        }
        if include_code:
            d["mutated_code"] = self.mutated_code
        return d


class OracleDetector:
    """Perfect detector using ground-truth fault labels."""

    def __init__(self, labels: dict[tuple[str, int], set[str]]):
        self._labels = labels

    @classmethod
    def from_manifest(cls, records: Iterable[InjectionRecord]) -> "OracleDetector":
        labels: dict[tuple[str, int], set[str]] = {}
        for rec in records:
            labels.setdefault((rec.dataset, rec.step_idx), set()).add(rec.fault_class)
        return cls(labels)

    def predict(self, dataset: str, step_idx: int) -> dict[str, bool]:
        faults = self._labels.get((dataset, step_idx), set())
        return {c: c in faults for c in FAULT_CLASSES}

    def predict_any(self, dataset: str, step_idx: int) -> bool:
        return (dataset, step_idx) in self._labels

    def predict_all(self, steps: list[tuple[str, int]]) -> list[dict[str, bool]]:
        return [self.predict(d, i) for d, i in steps]

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, all_steps: list[tuple[str, int]]) -> dict:
        """Per-class and aggregate metrics over the given (non-buggy) steps.

        All values are 1.0 by construction; computing them from the same
        predict() path the practical detectors will use confirms the label
        plumbing is sound rather than assuming it.
        """
        per_class: dict[str, dict] = {}
        for cls in FAULT_CLASSES:
            tp = fp = fn = 0
            for ds, idx in all_steps:
                truth = cls in self._labels.get((ds, idx), set())
                pred = self.predict(ds, idx)[cls]
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif truth and not pred:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 1.0
            per_class[cls] = {
                "precision": prec, "recall": rec, "f1": f1, "support": tp + fn,
            }

        n_faulty = sum(1 for ds, i in all_steps if self.predict_any(ds, i))
        n_clean = len(all_steps) - n_faulty
        return {
            "per_class": per_class,
            "any_fault": {
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "n_faulty": n_faulty,
                "n_clean": n_clean,
                "fault_rate": n_faulty / len(all_steps) if all_steps else 0.0,
            },
        }


def compute_cost_savings(traces: dict, labels: dict[tuple[str, int], set[str]], meta: dict) -> dict:
    """Compute what a perfect gate would have saved by skipping faulty trials."""
    total_wall = faulty_wall = 0.0
    total_steps = faulty_steps = 0

    for dataset, step_list in traces.items():
        for step in step_list:
            if step.is_buggy:
                continue
            w = step.wall_clock_sec or 0.0
            total_wall += w
            total_steps += 1
            if (dataset, step.step_idx) in labels:
                faulty_wall += w
                faulty_steps += 1

    pct = (faulty_wall / total_wall * 100) if total_wall > 0 else 0.0

    # tokens and cost are only reported per dataset, so apportion them by the
    # share of non-buggy wall clock that the faulty steps account for
    tokens_saved = 0.0
    cost_saved = 0.0
    for dataset, step_list in traces.items():
        row = meta.get(dataset, {})
        ds_wall = sum(s.wall_clock_sec or 0.0 for s in step_list if not s.is_buggy)
        ds_faulty = sum(
            s.wall_clock_sec or 0.0
            for s in step_list
            if not s.is_buggy and (dataset, s.step_idx) in labels
        )
        if ds_wall <= 0:
            continue
        frac = ds_faulty / ds_wall
        tokens_saved += frac * float(row.get("total_tokens", 0) or 0)
        cost_saved += frac * float(row.get("api_cost_usd", 0) or 0)

    return {
        "total_wall_clock_sec": total_wall,
        "faulty_wall_clock_sec": faulty_wall,
        "savings_pct": pct,
        "n_total_steps": total_steps,
        "n_faulty_steps": faulty_steps,
        "total_tokens_saved": int(tokens_saved),
        "total_cost_saved_usd": round(cost_saved, 4),
    }
