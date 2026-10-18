"""CP-3 re-execution harness.

Runs (possibly mutated) step code in an isolated subprocess with the dataset
staged at ./input/data.csv, exactly as AIDE presented it, and recovers the
validation metric from stdout.

Fully offline: the CSV is copied from the local data/ tree, never downloaded.
"""

from __future__ import annotations

import re
import contextlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# "Mean Macro F1: 0.7136", "Mean CV Macro F1: 0.81", "METRIC=0.9"
_EXPLICIT = re.compile(r"METRIC\s*=\s*([-+]?\d*\.?\d+)")
# Greedy label capture up to the LAST separator. A restrictive character class
# silently drops real metric lines containing "+", unicode hyphens or quotes,
# letting a library diagnostic line (e.g. binary_logloss) win instead.
_LABELLED = re.compile(r"^(.+)[:=]\s*([-+]?\d*\.\d+)\s*$")  # greedy label to the LAST separator; end-anchored scored best on the corpus
_AGGREGATE = re.compile(r"\b(mean|cv|overall|final|average|avg|total)\b", re.I)
# Score-like metrics (higher is better) - what AIDE optimises and reports.
_SCORE_METRIC = re.compile(
    r"\b(f1|auc|roc|acc|accuracy|score|precision|recall|kappa|r2|balanced)\b", re.I)
# Loss-like metrics - real metrics, but only a fallback: a library's
# per-iteration logloss must never outrank the reported F1.
_LOSS_METRIC = re.compile(r"\b(loss|logloss|rmse|mse|mae|error)\b", re.I)
# Pure parameters/diagnostics - used ONLY to filter lines that name no metric
# at all. A label may mention "threshold" and still be the metric line, e.g.
# "Mean CV macro F1 with bagged ensemble and threshold tuning: 0.9959".
_NOT_A_METRIC = re.compile(
    r"\b(threshold|param\w*|iteration|epoch|seed|round|time|sec|"
    r"n_estimators|depth|rate|fold|split|size|count|bins|points)\b", re.I)


# Any "./input/<file>" the generated code reads - the filename varies per
# dataset (data.csv for most, credit-g.csv for one).
_INPUT_REF = re.compile(r"""input[/\\]+([A-Za-z0-9_.\-]+\.csv)""")


def referenced_inputs(code: str) -> set[str]:
    """Filenames the code expects to find under ./input/."""
    return set(_INPUT_REF.findall(code))


# ---------------------------------------------------------------------------
# Catastrophic-failure filter
# ---------------------------------------------------------------------------
# A synthetic F2 injection that drops the score to near-random (or below) is
# not measuring leakage inflation - the injected code broke the pipeline.
# These are reclassified as injection failures so they don't pollute the
# leakage-inflation statistic.
#
# Thresholds:
#   MIN_VIABLE_SCORE : absolute floor - anything below this is garbage
#                      (random binary macro-F1 is ~0.50; 0.10 is well below)
#   CRASH_RATIO      : relative floor - if the injected score is less than
#                      this fraction of the baseline score the pipeline
#                      crashed rather than merely leaked

MIN_VIABLE_SCORE = 0.10
CRASH_RATIO = 0.30


def is_viable_injection(baseline_score: float, injected_score: float) -> tuple[bool, str]:
    """Decide whether a re-executed injection produced a valid measurement.

    Returns (viable, reason). When viable is False, `reason` explains why.
    A non-viable injection should still be labelled as faulty (the mutation
    happened) but excluded from leakage-inflation / degradation statistics.
    """
    if injected_score < MIN_VIABLE_SCORE:
        return False, (
            f"injected_score {injected_score:.4f} < MIN_VIABLE_SCORE {MIN_VIABLE_SCORE}"
        )
    threshold = baseline_score * CRASH_RATIO
    if injected_score < threshold:
        return False, (
            f"injected_score {injected_score:.4f} < "
            f"{CRASH_RATIO:.0%} of baseline {baseline_score:.4f} "
            f"(threshold {threshold:.4f})"
        )
    return True, ""


def extract_metric(stdout: str) -> float | None:
    """Recover the reported validation metric from captured stdout.

    Priority: an explicit METRIC= line, then the last 'label: float' line that
    mentions mean/cv/overall, then the last 'label: float' line overall.
    """
    m = _EXPLICIT.findall(stdout)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass

    labelled: list[tuple[str, float]] = []
    for line in stdout.splitlines():
        hit = _LABELLED.search(line.strip())
        if hit:
            try:
                labelled.append((hit.group(1), float(hit.group(2))))
            except ValueError:
                continue
    if not labelled:
        return None

    # Precedence: a score-like metric beats a loss-like metric beats anything
    # that at least is not obviously a parameter. Naming a metric wins even if
    # the label also mentions a parameter word.
    scored = [(l, v) for l, v in labelled if _SCORE_METRIC.search(l)]
    lossy = [(l, v) for l, v in labelled if _LOSS_METRIC.search(l)]
    neutral = [(l, v) for l, v in labelled if not _NOT_A_METRIC.search(l)]
    pool = scored or lossy or neutral or labelled

    # within the chosen tier prefer an aggregate (mean/cv/overall) line
    aggregated = [v for l, v in pool if _AGGREGATE.search(l)]
    if aggregated:
        return aggregated[-1]
    return pool[-1][1]


def resolve_data_csv(project_root: Path, dataset: str, group: str, corpus_dir: Path | None = None) -> Path | None:
    """Locate the dataset CSV in the local data tree."""
    candidates = [
        project_root / "data" / group / dataset / "data.csv",
        project_root / "data" / ("cc18" if group != "cc18" else "transfer") / dataset / "data.csv",
        project_root / "traces" / "raw" / dataset / "workspace" / "input" / "data.csv",
    ]
    if corpus_dir is not None:
        candidates.append(corpus_dir / dataset / "workspace" / "input" / "data.csv")
    for c in candidates:
        if c.exists():
            return c
    return None


def reexecute_step(
    mutated_code: str,
    dataset: str,
    group: str,
    project_root: Path,
    timeout_sec: int = 180,
    python_exe: str | None = None,
    corpus_dir: Path | None = None,
) -> dict:
    """Execute code in a subprocess, returning the recovered metric.

    Returns {success, score, stdout, stderr, wall_clock_sec, error}.
    """
    started = time.time()
    data_csv = resolve_data_csv(project_root, dataset, group, corpus_dir=corpus_dir)
    if data_csv is None:
        return {
            "success": False, "score": None, "stdout": "", "stderr": "",
            "wall_clock_sec": 0.0, "error": f"no data.csv for {dataset}",
        }

    exe = python_exe or sys.executable

    run_root = project_root / "cp3_fault_injection" / "reexec_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    # Keep baseline/mutation code and execution output as local research evidence.
    with contextlib.nullcontext(tempfile.mkdtemp(prefix=f"cp3_{dataset}_", dir=run_root)) as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "input").mkdir()
        # The generated code may read ./input/data.csv or ./input/<dataset>.csv.
        # Stage the CSV under every name the code actually references, plus the
        # two conventional defaults, so the filename never decides the outcome.
        names = referenced_inputs(mutated_code) | {"data.csv", f"{dataset}.csv"}
        for name in names:
            # copy, never symlink - the code may rewrite its input
            shutil.copy2(data_csv, tmpdir / "input" / name)
        (tmpdir / "working").mkdir(exist_ok=True)

        script = tmpdir / "step.py"
        script.write_text(mutated_code)

        try:
            proc = subprocess.run(
                [exe, str(script)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False, "score": None, "stdout": "", "stderr": "",
                "wall_clock_sec": time.time() - started, "error": f"timeout after {timeout_sec}s",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False, "score": None, "stdout": "", "stderr": "",
                "wall_clock_sec": time.time() - started, "error": f"{type(exc).__name__}: {exc}",
            }

        wall = time.time() - started
        (tmpdir / "stdout.log").write_text(proc.stdout)
        (tmpdir / "stderr.log").write_text(proc.stderr)
        score = extract_metric(proc.stdout)

        if proc.returncode != 0:
            return {
                "success": False, "score": None,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
                "wall_clock_sec": wall, "error": f"exit {proc.returncode}",
            }
        if score is None:
            return {
                "success": False, "score": None,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
                "wall_clock_sec": wall, "error": "no metric found in stdout",
            }

        return {
            "success": True, "score": score,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
            "wall_clock_sec": wall, "error": None,
        }
