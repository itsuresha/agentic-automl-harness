"""CP-3 trace loading.

Discovers and parses AIDE journals into flat StepRecord lists.
Schema-agnostic: field names are resolved through a mapping heuristic so
that differing AIDE versions load without code changes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- field-name candidates, highest priority first -------------------------
KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "code": ("code", "solution", "generated_code", "source"),
    "is_buggy": ("is_buggy", "buggy", "has_error"),
    "validation_score": ("metric", "score", "validation_metric", "val_score", "best_metric"),
    "execution_output": ("_term_out", "exec_output", "execution_result", "stdout", "output", "result"),
    "feedback": ("analysis", "feedback", "reflection"),
    "wall_clock_sec": ("exec_time", "wall_clock_sec", "elapsed"),
    "step_idx": ("step", "step_idx", "index"),
}

JOURNAL_CANDIDATES = ("journal.json", "tree_log.json", "log.json", "experiment.json")


@dataclass
class StepRecord:
    dataset: str
    openml_id: int
    step_idx: int
    code: str
    is_buggy: bool
    validation_score: float | None
    execution_output: str
    feedback: str
    wall_clock_sec: float | None
    parse_notes: list[str] = field(default_factory=list)


def discover_journal(dataset_dir: Path) -> Path:
    """Find the AIDE journal file in a dataset directory."""
    for name in JOURNAL_CANDIDATES:
        p = dataset_dir / name
        if p.exists():
            return p
    jsons = sorted(dataset_dir.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
    if jsons:
        return jsons[0]
    raise FileNotFoundError(f"No journal found in {dataset_dir}")


def _first_present(node: dict, field_name: str) -> tuple[Any, str | None]:
    """Return (value, key_used) for the first candidate key present in node."""
    for key in KEY_CANDIDATES[field_name]:
        if key in node:
            return node[key], key
    return None, None


def _coerce_score(raw: Any) -> float | None:
    """AIDE stores metric as {'value': float|None, 'maximize': bool|None}."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        val = raw.get("value")
        return float(val) if isinstance(val, (int, float)) else None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _coerce_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "".join(str(x) for x in raw)
    return str(raw)


def _extract_nodes(journal: Any) -> list[dict]:
    """Pull the step list out of a journal of unknown shape."""
    if isinstance(journal, list):
        return [n for n in journal if isinstance(n, dict)]
    if isinstance(journal, dict):
        for key in ("nodes", "steps", "trials", "history"):
            val = journal.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
        # last resort: any list-of-dicts value
        for val in journal.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
    raise ValueError("Could not locate step nodes in journal")


def parse_journal(path: Path, dataset: str, openml_id: int) -> list[StepRecord]:
    journal = json.loads(path.read_text())
    nodes = _extract_nodes(journal)

    records: list[StepRecord] = []
    for pos, node in enumerate(nodes):
        notes: list[str] = []

        code, k = _first_present(node, "code")
        if k is None:
            notes.append("no code key")
        code = str(code or "")

        buggy_raw, k = _first_present(node, "is_buggy")
        if k is None:
            # fall back to presence of an exception type
            buggy_raw = bool(node.get("exc_type"))
            notes.append("is_buggy inferred from exc_type")
        is_buggy = bool(buggy_raw)

        score_raw, _ = _first_present(node, "validation_score")
        score = _coerce_score(score_raw)

        out_raw, _ = _first_present(node, "execution_output")
        execution_output = _coerce_output(out_raw)

        fb_raw, _ = _first_present(node, "feedback")
        feedback = str(fb_raw or "")

        wall_raw, _ = _first_present(node, "wall_clock_sec")
        wall = float(wall_raw) if isinstance(wall_raw, (int, float)) else None

        idx_raw, k = _first_present(node, "step_idx")
        step_idx = int(idx_raw) if isinstance(idx_raw, int) else pos
        if k is None:
            notes.append("step_idx from position")

        records.append(
            StepRecord(
                dataset=dataset,
                openml_id=openml_id,
                step_idx=step_idx,
                code=code,
                is_buggy=is_buggy,
                validation_score=score,
                execution_output=execution_output,
                feedback=feedback,
                wall_clock_sec=wall,
                parse_notes=notes,
            )
        )

    records.sort(key=lambda r: r.step_idx)
    return records


def read_summary(summary_csv: Path) -> list[dict]:
    with open(summary_csv, newline="") as fh:
        return list(csv.DictReader(fh))


def load_traces(
    corpus_dir: Path,
    summary_csv: Path,
    log: list[str] | None = None,
) -> tuple[dict[str, list[StepRecord]], dict[str, dict]]:
    """Load every SUCCESS dataset in the summary CSV.

    Returns (traces, meta) where meta[dataset] is the summary CSV row.
    Datasets whose status is not SUCCESS are skipped (they are partial runs
    and would corrupt the per-dataset step statistics).
    """
    log = log if log is not None else []
    rows = read_summary(summary_csv)

    traces: dict[str, list[StepRecord]] = {}
    meta: dict[str, dict] = {}

    for row in rows:
        name = row["dataset_name"]
        if row.get("status") != "SUCCESS":
            log.append(f"SKIP {name}: status={row.get('status')} completed={row.get('completed_steps')}")
            continue

        ds_dir = corpus_dir / name
        if not ds_dir.is_dir():
            log.append(f"SKIP {name}: directory not found under {corpus_dir}")
            continue

        try:
            journal_path = discover_journal(ds_dir)
        except FileNotFoundError as exc:
            log.append(f"SKIP {name}: {exc}")
            continue

        try:
            steps = parse_journal(journal_path, name, int(row["openml_id"]))
        except Exception as exc:  # noqa: BLE001 - want to continue past one bad journal
            log.append(f"SKIP {name}: journal parse failed: {exc}")
            continue

        # --- validation against the summary CSV (warn, never abort) --------
        exp_total = int(row["completed_steps"])
        exp_ok = int(row["n_successful_steps"])
        exp_bug = int(row["n_buggy_steps"])
        got_ok = sum(1 for s in steps if not s.is_buggy)
        got_bug = sum(1 for s in steps if s.is_buggy)

        if len(steps) != exp_total:
            log.append(f"WARN {name}: {len(steps)} steps loaded, CSV says {exp_total}")
        if got_ok != exp_ok:
            log.append(f"WARN {name}: {got_ok} non-buggy, CSV says {exp_ok}")
        if got_bug != exp_bug:
            log.append(f"WARN {name}: {got_bug} buggy, CSV says {exp_bug}")

        for s in steps:
            for note in s.parse_notes:
                log.append(f"NOTE {name} step {s.step_idx}: {note}")

        traces[name] = steps
        meta[name] = row

    return traces, meta
