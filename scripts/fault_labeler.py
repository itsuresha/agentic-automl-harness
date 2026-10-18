"""CP-3 natural fault labelling (F1 stale memory, F2 leakage, F3 partial fit)."""

from __future__ import annotations

from feature_extractor import StepConfig, canonical_hash
from trace_loader import StepRecord

ITERATIVE_MODELS = {
    "logisticregression", "sgdclassifier", "sgdregressor",
    "mlpclassifier", "mlpregressor", "svc", "svr",
    "ridgeclassifier", "perceptron", "passiveaggressiveclassifier",
}
ENSEMBLE_MODELS = {
    "randomforestclassifier", "randomforestregressor",
    "gradientboostingclassifier", "gradientboostingregressor",
    "adaboostclassifier", "adaboostregressor",
    "baggingclassifier", "baggingregressor",
    "extratreesclassifier", "extratreesregressor",
    "xgbclassifier", "xgbregressor",
    "lgbmclassifier", "lgbmregressor",
    "catboostclassifier", "catboostregressor",
    "votingclassifier", "stackingclassifier",
}
TREE_MODELS = {"decisiontreeclassifier", "decisiontreeregressor"}
GRADIENT_BOOSTING_MODELS = {
    "gradientboostingclassifier", "gradientboostingregressor",
    "xgbclassifier", "xgbregressor", "lgbmclassifier", "lgbmregressor",
    "catboostclassifier", "catboostregressor", "adaboostclassifier", "adaboostregressor",
}
LR_ALIASES = ("learning_rate", "eta")

FAULT_NAMES = {
    "F1": "Redundant Configuration Proposal (RCP)",
    "F2": "Data Leakage (DL)",
    "F3": "Training-Budget Fault (TBF)",
}

# n_estimators aliases used by the boosting libraries
N_EST_KEYS = ("n_estimators", "num_boost_round", "num_iterations", "iterations")
MAX_ITER_KEYS = ("max_iter", "max_iterations")


def label_f1(
    steps: list[StepRecord],
    configs: list[StepConfig | None],
) -> tuple[list[bool | None], list[tuple[int, int, str]]]:
    """F1: canonical config repeats an earlier non-buggy step in the same run.

    Returns (labels, collisions) where collisions are (step_i, donor_j, hash).
    """
    seen: dict[str, int] = {}
    labels: list[bool | None] = []
    collisions: list[tuple[int, int, str]] = []

    for step, cfg in zip(steps, configs):
        if step.is_buggy or cfg is None:
            labels.append(None)
            continue
        h = canonical_hash(cfg)
        if h in seen:
            labels.append(True)
            collisions.append((step.step_idx, seen[h], h))
        else:
            seen[h] = step.step_idx
            labels.append(False)
    return labels, collisions


def label_f2(steps: list[StepRecord], configs: list[StepConfig | None]) -> list[bool | None]:
    """F2: a statistic-learning preprocessor is fitted on pre-split data.

    Uses StepConfig.leaky_before_split, which already applies the Pipeline/CV
    guard and excludes vocabulary-only encoders.
    """
    labels: list[bool | None] = []
    for step, cfg in zip(steps, configs):
        if step.is_buggy or cfg is None:
            labels.append(None)
            continue
        labels.append(bool(cfg.leaky_before_split))
    return labels


def _first_key(hp: dict, keys: tuple[str, ...], default):
    for k in keys:
        if k in hp:
            return hp[k]
    return default


def _as_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _as_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def label_f3(
    steps: list[StepRecord],
    configs: list[StepConfig | None],
) -> tuple[list[bool | None], list[list[str]]]:
    """F3: fit completed but the model is inadequately trained.

    Returns (labels, reasons) - reasons[i] lists which indicators fired.
    """
    scores = [s.validation_score for s in steps if not s.is_buggy and s.validation_score is not None]
    median_score = sorted(scores)[len(scores) // 2] if scores else 0.5

    labels: list[bool | None] = []
    reasons: list[list[str]] = []

    for step, cfg in zip(steps, configs):
        if step.is_buggy or cfg is None:
            labels.append(None)
            reasons.append([])
            continue

        cls = cfg.model_class.lower()
        hp = cfg.hyperparameters
        why: list[str] = []

        # 1. premature convergence stop
        if cls in ITERATIVE_MODELS:
            if _as_int(_first_key(hp, MAX_ITER_KEYS, 100), 100) <= 2:
                why.append("max_iter<=2")

        # 2. degenerate ensemble
        if cls in ENSEMBLE_MODELS:
            if _as_int(_first_key(hp, N_EST_KEYS, 100), 100) <= 2:
                why.append("n_estimators<=2")

        # 3. trivial tree (decision stump)
        if cls in TREE_MODELS | ENSEMBLE_MODELS and _as_int(hp.get("max_depth", 0), 0) == 1:
            why.append("max_depth==1")

        # 4. convergence warning together with a below-median score
        if "ConvergenceWarning" in step.execution_output:
            if step.validation_score is not None and step.validation_score < median_score:
                why.append("ConvergenceWarning+below_median")

        # 5. absurd learning rate; inspect both spellings if both are present.
        if cls in GRADIENT_BOOSTING_MODELS:
            if any(_as_float(hp.get(k, 0.1), 0.1) >= 50.0 for k in LR_ALIASES):
                why.append("learning_rate>=50")

        labels.append(bool(why))
        reasons.append(why)

    return labels, reasons


def label_dataset(
    steps: list[StepRecord],
    configs: list[StepConfig | None],
) -> dict:
    """Run all three labellers over one dataset run."""
    f1, collisions = label_f1(steps, configs)
    f2 = label_f2(steps, configs)
    f3, f3_reasons = label_f3(steps, configs)
    return {
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "f1_collisions": collisions,
        "f3_reasons": f3_reasons,
    }
