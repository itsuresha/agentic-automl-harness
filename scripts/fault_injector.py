"""CP-3 synthetic fault injection.

F1 - config duplication: relabel only, no code change (a semantically
     identical config reproduces the donor's score by construction).
F2 - leakage: fit a target-aware selector on the FULL data before the split,
     which is the brief's sanctioned "apply fit_transform to X before
     splitting" form. Chosen over scaler leakage because ~96% of the corpus
     uses LightGBM, and tree ensembles are invariant to feature scaling -
     a scaler leak would be undetectable by construction.
F3 - partial fit: clamp the training budget via AST rewrite.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from feature_extractor import StepConfig
from fault_labeler import ENSEMBLE_MODELS, ITERATIVE_MODELS, TREE_MODELS, GRADIENT_BOOSTING_MODELS, LR_ALIASES

# --- F2 --------------------------------------------------------------------
LEAK_SNIPPET = '''
# --- CP-3 SYNTHETIC F2 INJECTION: data leakage -------------------------
# Target-mean encoding computed over the FULL dataset, before any split.
# Every row's encoded value is informed by the labels of rows that will
# later land in the validation folds, so validation scores are inflated.
# Original columns are kept, so this adds leaked signal without removing
# information (a selector-based leak loses more than it gains on trees).
import pandas as _cp3_pd, numpy as _cp3_np
try:
    _cp3_is_df = hasattr({XV}, "iloc")
    _cp3_X = {XV}.copy() if _cp3_is_df else _cp3_pd.DataFrame(_cp3_np.asarray({XV}))
    _cp3_y = _cp3_pd.Series(_cp3_np.asarray({YV}).ravel()).astype(float)
    # bin count scales with row count so the leak strength is comparable
    # across datasets of very different sizes (~{rpb} rows per bin)
    _cp3_q = max(10, len(_cp3_X) // {rpb})
    _cp3_cols = list(_cp3_X.columns)[:{ncols}]
    _cp3_new = []
    for _cp3_col in _cp3_cols:
        _cp3_s = _cp3_X[_cp3_col]
        if _cp3_s.nunique() > {card}:
            _cp3_b = _cp3_pd.qcut(_cp3_s.rank(method="first"), q=_cp3_q,
                                  labels=False, duplicates="drop")
        else:
            _cp3_b = _cp3_s.astype("category").cat.codes
        _cp3_new.append(_cp3_y.groupby(_cp3_b).transform("mean").values)
    # keep the original container type: turning an ndarray into a DataFrame
    # would break downstream positional indexing such as X[train_idx]
    if _cp3_is_df:
        for _cp3_i, _cp3_v in enumerate(_cp3_new):
            _cp3_X["cp3te_%d" % _cp3_i] = _cp3_v
        {XV} = _cp3_X
    elif _cp3_new:
        {XV} = _cp3_np.column_stack([_cp3_np.asarray({XV})] + _cp3_new)
except Exception:
    pass
# --- END CP-3 INJECTION ------------------------------------------------
'''


def inject_leakage(code: str, config: StepConfig, ncols: int = 5,
                   card: int = 20, rows_per_bin: int = 8) -> tuple[str | None, dict]:
    """Insert a pre-split target-aware selection fit.

    Returns (mutated_code, details). mutated_code is None if the step is not
    a viable target.
    """
    anchor = config.split_use_position or config.split_position
    if anchor is None:
        return None, {"skip_reason": "no split position identified"}

    lines = code.split("\n")
    insert_at = anchor - 1  # 0-based, insert before the split is applied

    if insert_at < 1 or insert_at > len(lines):
        return None, {"skip_reason": f"split line {config.split_position} out of range"}

    # X and y must already exist at the insertion point
    xv = config.feature_var or "X"
    yv = config.target_var or "y"
    head = "\n".join(lines[:insert_at])
    for var in (xv, yv):
        if not re.search(rf"^\s*{re.escape(var)}\s*(?:[,)\]]|=[^=])", head, re.M):
            return None, {"skip_reason": f"variable '{var}' not assigned before split"}

    snippet = LEAK_SNIPPET.format(ncols=ncols, card=card, rpb=rows_per_bin, XV=xv, YV=yv)
    mutated = "\n".join(lines[:insert_at]) + "\n" + snippet + "\n" + "\n".join(lines[insert_at:])

    try:
        ast.parse(mutated)
    except SyntaxError as exc:
        return None, {"skip_reason": f"mutated code does not parse: {exc.msg}"}

    return mutated, {
        "original_fit_line": config.first_fit_position,
        "mutated_fit_line": insert_at + 1,
        "preprocessor": "target_mean_encoding(full_data)",
        "n_cols_encoded": ncols,
        "rows_per_bin": rows_per_bin,
        "split_line": config.split_position,
        "anchor_line": anchor,
        "feature_var": xv,
        "target_var": yv,
    }


# --- F3 --------------------------------------------------------------------
CLAMP_ITERATIVE = {"max_iter": 1}
CLAMP_ENSEMBLE = {"n_estimators": 1}
CLAMP_TREE = {"max_depth": 1}
CLAMP_LR = {"learning_rate": 100.0}
F3_STRATEGIES = [
    {"name": "clamp_n_est", "clamps": CLAMP_ENSEMBLE,
     "applicable": lambda cls: cls in ENSEMBLE_MODELS,
     "description": "Single boosting round / single tree in ensemble"},
    {"name": "clamp_lr", "clamps": CLAMP_LR,
     "applicable": lambda cls: cls in GRADIENT_BOOSTING_MODELS,
     "description": "Absurdly high learning rate"},
    {"name": "clamp_depth", "clamps": CLAMP_TREE,
     "applicable": lambda cls: cls in ENSEMBLE_MODELS or cls in TREE_MODELS,
     "description": "Decision stumps"},
    {"name": "clamp_iter", "clamps": CLAMP_ITERATIVE,
     "applicable": lambda cls: cls in ITERATIVE_MODELS,
     "description": "Single optimisation iteration"},
]


def get_applicable_strategies(model_class: str) -> list[dict]:
    return [s for s in F3_STRATEGIES if s["applicable"](model_class.lower())]

# boosting libraries spell the budget differently
BUDGET_ALIASES = ("n_estimators", "num_boost_round", "num_iterations", "iterations")

WARN_SUPPRESS = (
    "import warnings as _cp3_warnings\n"
    "_cp3_warnings.filterwarnings('ignore')\n"
)


class _ClampTransformer(ast.NodeTransformer):
    """Force the model's training budget down to a degenerate value."""

    def __init__(self, model_class: str, clamps: dict[str, Any], explicit: bool = False) -> None:
        self.model_class = model_class
        self.clamps = clamps
        self.applied: list[tuple[str, Any, Any]] = []
        self.explicit = explicit

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)

        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""

        if name.lower() != self.model_class.lower():
            return node

        if self.explicit:
            # Remove clamped keys from **params without changing the source dict.
            # Otherwise adding an explicit keyword can cause duplicate-key errors.
            aliases = set(self.clamps)
            if "n_estimators" in aliases:
                aliases.update(BUDGET_ALIASES)
            if "learning_rate" in aliases:
                aliases.update(LR_ALIASES)
            for kw in node.keywords:
                if kw.arg is None:
                    filtered = ast.parse("{_cp3_k: _cp3_v for _cp3_k, _cp3_v in params.items() if _cp3_k not in ()}", mode="eval").body
                    filtered.generators[0].iter.func.value = kw.value
                    filtered.generators[0].ifs[0].comparators[0] = ast.Tuple(elts=[ast.Constant(k) for k in sorted(aliases)], ctx=ast.Load())
                    kw.value = filtered

        for key, new_val in self.clamps.items():
            # replace the alias actually present, else add the canonical key
            target_key = key
            for kw in node.keywords:
                if kw.arg in BUDGET_ALIASES and key == "n_estimators":
                    target_key = kw.arg
                    break
                if kw.arg in LR_ALIASES and key == "learning_rate":
                    target_key = kw.arg
                    break

            existing = next((kw for kw in node.keywords if kw.arg == target_key), None)
            old_val = None
            if existing is not None:
                try:
                    old_val = ast.literal_eval(existing.value)
                except (ValueError, SyntaxError):
                    old_val = "<expr>"
                existing.value = ast.Constant(value=new_val)
            else:
                node.keywords.append(ast.keyword(arg=target_key, value=ast.Constant(value=new_val)))

            self.applied.append((target_key, old_val, new_val))
            if self.explicit:
                aliases = BUDGET_ALIASES if key == "n_estimators" else LR_ALIASES if key == "learning_rate" else (key,)
                for kw in node.keywords:
                    if kw.arg in aliases and kw.arg != target_key:
                        kw.value = ast.Constant(value=new_val)

        return node


def inject_partial_fit(code: str, config: StepConfig, strategy: str | None = None) -> tuple[str | None, dict]:
    """Clamp the model's training budget so the fit completes but undertrains."""
    from fault_labeler import ENSEMBLE_MODELS, ITERATIVE_MODELS, TREE_MODELS

    cls = config.model_class
    if not cls:
        return None, {"skip_reason": "no model class identified"}

    low = cls.lower()
    if strategy is not None:
        chosen = next((s for s in F3_STRATEGIES if s["name"] == strategy), None)
        if chosen is None or not chosen["applicable"](low):
            return None, {"skip_reason": f"Strategy {strategy!r} is not applicable to {cls}", "mutation_strategy": strategy}
        clamps = dict(chosen["clamps"])
        if strategy == "clamp_depth" and low in {"lgbmclassifier", "lgbmregressor"}:
            clamps["num_leaves"] = 2
    elif low in ENSEMBLE_MODELS:
        clamps = dict(CLAMP_ENSEMBLE)
    elif low in ITERATIVE_MODELS:
        clamps = dict(CLAMP_ITERATIVE)
    elif low in TREE_MODELS:
        clamps = dict(CLAMP_TREE)
    else:
        return None, {"skip_reason": f"model '{cls}' is neither iterative, ensemble nor tree"}

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, {"skip_reason": f"original code does not parse: {exc.msg}"}

    strategy_name = strategy or ("clamp_n_est" if low in ENSEMBLE_MODELS else "clamp_iter" if low in ITERATIVE_MODELS else "clamp_depth")
    tf = _ClampTransformer(cls, clamps, explicit=strategy is not None)
    new_tree = tf.visit(tree)
    if not tf.applied:
        return None, {"skip_reason": f"no instantiation of '{cls}' found to clamp"}

    ast.fix_missing_locations(new_tree)
    try:
        mutated = WARN_SUPPRESS + ast.unparse(new_tree)
    except Exception as exc:  # noqa: BLE001
        return None, {"skip_reason": f"unparse failed: {exc}"}

    try:
        ast.parse(mutated)
    except SyntaxError as exc:
        return None, {"skip_reason": f"mutated code does not parse: {exc.msg}"}

    key, old, new = tf.applied[0]
    return mutated, {
        "clamped_param": key,
        "original_value": old,
        "clamped_value": new,
        "model_class": cls,
        "n_sites_clamped": len(tf.applied),
        "mutation_strategy": strategy_name,
    }
