"""CP-3 feature extraction.

Parses generated Python into a StepConfig (model, hyperparameters,
preprocessing chain, split/fit positions) and derives a canonical hash that
is stable under formatting, kwarg order and trivial value differences.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

HASH_VERSION = "v2"

FEATURE_TIERS = {
    "canonical_hash": 1,
    "model_class": 1,
    "n_hyperparams": 1,
    "n_preprocessing_steps": 1,
    "preprocessing_before_split": 1,
    "leaky_before_split": 1,
    "hash_seen_before": 1,
    "hash_first_seen_at": 1,
    "parse_quality": 1,
    "validation_score": 2,
    "wall_clock_sec": 2,
    "score_delta_from_prev": 2,
    "score_rank_in_run": 2,
    "is_best_so_far": 2,
    "steps_since_improvement": 2,
}

# --- vocabularies ----------------------------------------------------------
MODEL_SUFFIXES = ("Classifier", "Regressor")
MODEL_EXACT = {
    "SVC", "SVR", "LinearSVC", "LinearSVR", "NuSVC", "NuSVR",
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "Perceptron", "GaussianNB", "MultinomialNB", "BernoulliNB",
    "KNeighborsClassifier", "KNeighborsRegressor", "LDA", "QDA",
    "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis",
}
MODEL_PREFIXES = ("XGB", "LGBM", "CatBoost", "MLP")

PREPROCESSORS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "PCA", "TruncatedSVD", "PolynomialFeatures", "LabelEncoder", "OneHotEncoder",
    "OrdinalEncoder", "SimpleImputer", "KNNImputer", "ColumnTransformer",
    "Pipeline", "make_pipeline", "SelectKBest", "VarianceThreshold", "SMOTE",
    "FeatureHasher", "TfidfVectorizer", "CountVectorizer", "QuantileTransformer",
    "PowerTransformer", "FunctionTransformer", "LabelBinarizer",
}
FEATURE_SELECTORS = {"SelectKBest", "RFE", "RFECV", "VarianceThreshold", "SelectFromModel"}
# Preprocessors whose fit is inherently fold-safe when wrapped
PIPELINE_WRAPPERS = {"Pipeline", "make_pipeline", "ColumnTransformer", "make_column_transformer"}

# Preprocessors whose fitted state is a *vocabulary* (set of observed categories).
# Fitting these on the full dataset leaks no distributional statistic, so per the
# F2 rationale ("mean, variance, PCA loadings") they are NOT data leakage.
VOCABULARY_ONLY = {
    "LabelEncoder", "OrdinalEncoder", "LabelBinarizer", "OneHotEncoder",
}
# Preprocessors whose fitted parameters depend on the data distribution (or on y).
# Fitting these pre-split is genuine leakage.
STATISTIC_LEAKING = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "PCA", "TruncatedSVD", "SimpleImputer", "KNNImputer", "SelectKBest",
    "VarianceThreshold", "QuantileTransformer", "PowerTransformer",
    "TfidfVectorizer", "CountVectorizer", "SMOTE", "FeatureHasher",
}

SPLIT_CALLS = {
    "train_test_split", "StratifiedKFold", "KFold", "cross_val_score",
    "cross_validate", "StratifiedShuffleSplit", "ShuffleSplit",
    "GroupKFold", "TimeSeriesSplit", "RepeatedStratifiedKFold",
}
CV_WRAPPERS = {"cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}

REGEX_PATTERNS = {
    "model_class": re.compile(
        r"\b(\w*(?:Classifier|Regressor)|SVC|SVR|XGB\w+|LGBM\w+|CatBoost\w+|MLP\w+)\s*\("
    ),
    "preprocessor": re.compile(r"\b(" + "|".join(sorted(PREPROCESSORS)) + r")\s*\("),
    "split": re.compile(r"\b(" + "|".join(sorted(SPLIT_CALLS)) + r")\s*\("),
    "fit_transform": re.compile(r"\.fit_transform\s*\("),
    "fit_call": re.compile(r"\.fit\s*\("),
}


@dataclass
class StepConfig:
    model_class: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing_chain: list[str] = field(default_factory=list)
    split_position: int | None = None
    split_use_position: int | None = None   # where the split is APPLIED to data
    feature_var: str | None = None          # the X-like variable handed to the split
    target_var: str | None = None           # the y-like variable handed to the split
    first_fit_position: int | None = None
    preprocessing_before_split: bool = False   # mechanical: ANY preprocessor pre-split
    leaky_before_split: bool = False           # strict: statistic-learning preprocessor pre-split
    parse_quality: str = "unparseable"
    preprocessing_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    split_method: str | None = None
    split_params: dict[str, Any] = field(default_factory=dict)
    feature_selector: str | None = None
    notes: list[str] = field(default_factory=list)


def _is_model_name(name: str) -> bool:
    if name in MODEL_EXACT:
        return True
    if name.endswith(MODEL_SUFFIXES):
        return True
    return name.startswith(MODEL_PREFIXES)


def _call_name(node: ast.Call) -> str:
    """Return the bare callable name for Name and Attribute calls."""
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return "<expr>"


class _ConfigVisitor(ast.NodeVisitor):
    """Single pass collecting everything the StepConfig needs."""

    def __init__(self) -> None:
        self.models: list[tuple[int, str, dict[str, Any]]] = []
        self.preprocessors: list[tuple[int, str, dict[str, Any]]] = []
        self.split_lines: list[int] = []
        self.split_use_lines: list[int] = []
        self.split_args: list[tuple[int, list[str]]] = []
        self.split_methods: list[tuple[int, str, dict[str, Any]]] = []
        self.feature_selectors: list[tuple[int, str]] = []
        self.cv_lines: list[int] = []
        # variable name -> preprocessor class it holds
        self.prep_vars: dict[str, str] = {}
        # variable names holding a Pipeline/ColumnTransformer
        self.pipeline_vars: set[str] = set()
        # fit calls: (line, receiver_name, method, arg_names)
        self.fit_calls: list[tuple[int, str, str, list[str]]] = []
        # names of variables produced by a split
        self.post_split_vars: set[str] = set()
        # pipelines handed to a CV wrapper -> fold-safe
        self.pipelines_in_cv: set[str] = set()

    # -- assignments let us track which variable holds which estimator ------
    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        tuple_targets: list[str] = []
        for t in node.targets:
            if isinstance(t, ast.Tuple):
                tuple_targets += [e.id for e in t.elts if isinstance(e, ast.Name)]

        if isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name in PIPELINE_WRAPPERS:
                for t in targets:
                    self.pipeline_vars.add(t)
            elif name in PREPROCESSORS:
                for t in targets:
                    self.prep_vars[t] = name
            if name in SPLIT_CALLS:
                # X_train, X_test, y_train, y_test = train_test_split(...)
                for t in tuple_targets + targets:
                    self.post_split_vars.add(t)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)

        if _is_model_name(name):
            hp = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            self.models.append((node.lineno, name, hp))

        if name in PREPROCESSORS:
            params = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            self.preprocessors.append((node.lineno, name, params))
            if name in FEATURE_SELECTORS:
                self.feature_selectors.append((node.lineno, name))

        if name in SPLIT_CALLS:
            self.split_lines.append(node.lineno)
            params = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            self.split_methods.append((node.lineno, name, params))
            # train_test_split(...) IS the split; KFold(...) is only its constructor
            if name in {"train_test_split"}:
                self.split_use_lines.append(node.lineno)
                self.split_args.append(
                    (node.lineno, [a.id for a in node.args if isinstance(a, ast.Name)])
                )

        # cv.split(X, y) - where a CV object is actually applied to the data
        if isinstance(node.func, ast.Attribute) and node.func.attr == "split":
            self.split_use_lines.append(node.lineno)
            self.split_methods.append((node.lineno, "split", {}))
            self.split_args.append(
                (node.lineno, [a.id for a in node.args if isinstance(a, ast.Name)])
            )

        if name in CV_WRAPPERS:
            self.cv_lines.append(node.lineno)
            # any pipeline passed positionally to CV is fold-safe
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.pipelines_in_cv.add(arg.id)
                elif isinstance(arg, ast.Call) and _call_name(arg) in PIPELINE_WRAPPERS:
                    self.pipelines_in_cv.add("<inline_pipeline>")
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name):
                    self.pipelines_in_cv.add(kw.value.id)

        # fit / fit_transform / fit_predict
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "fit", "fit_transform", "fit_predict"
        }:
            recv = node.func.value
            recv_name = recv.id if isinstance(recv, ast.Name) else _call_name(recv) if isinstance(recv, ast.Call) else ""
            arg_names = [a.id for a in node.args if isinstance(a, ast.Name)]
            self.fit_calls.append((node.lineno, recv_name, node.func.attr, arg_names))

        self.generic_visit(node)


def _analyse_leakage(v: _ConfigVisitor) -> tuple[bool, bool, int | None, int | None, list[str]]:
    """Decide whether preprocessing is fitted on pre-split data.

    Returns (mechanical, strict, split_position, first_prep_fit_position, notes).

    mechanical - ANY preprocessor fitted before the split (brief section 4.1).
    strict     - a *statistic-learning* preprocessor fitted before the split.
                 Vocabulary-only encoders (LabelEncoder etc.) are excluded
                 because they fit no distributional parameter and therefore
                 cannot inflate the score, which is what F2 is defined by.

    A fit inside a Pipeline handed to a CV wrapper is never leaky, because
    sklearn refits it within each fold.
    """
    notes: list[str] = []
    split_position = min(v.split_lines) if v.split_lines else None

    prep_fits: list[int] = []          # any preprocessor
    leak_fits: list[int] = []          # statistic-learning only

    for line, recv, method, args in v.fit_calls:
        # resolve the preprocessor class behind this receiver
        if recv in v.prep_vars:
            cls = v.prep_vars[recv]
        elif recv in PREPROCESSORS:
            cls = recv                 # inline: StandardScaler().fit_transform(...)
        elif recv in v.pipeline_vars:
            cls = "Pipeline"
        elif method == "fit_transform":
            cls = "<unknown>"          # fit_transform is preprocessing-shaped
        else:
            continue                   # a model .fit() - not preprocessing

        if recv in v.pipeline_vars and recv in v.pipelines_in_cv:
            notes.append(f"line {line}: pipeline '{recv}' is CV-wrapped, fold-safe")
            continue

        if args and all(a in v.post_split_vars for a in args):
            notes.append(f"line {line}: fits on post-split vars {args}, safe")
            continue

        prep_fits.append(line)
        if cls in STATISTIC_LEAKING or cls in ("<unknown>", "Pipeline"):
            leak_fits.append(line)
        elif cls in VOCABULARY_ONLY:
            notes.append(f"line {line}: {cls} is vocabulary-only, not a leakage source")

    first_fit = min(prep_fits) if prep_fits else None
    first_leak = min(leak_fits) if leak_fits else None

    if split_position is None:
        if first_fit is not None:
            notes.append("no explicit split found; cannot order fit vs split")
        return False, False, None, first_fit, notes

    mechanical = first_fit is not None and first_fit < split_position
    strict = first_leak is not None and first_leak < split_position

    if strict:
        notes.append(f"leaking preprocessing fit at line {first_leak} precedes split at line {split_position}")
    elif mechanical:
        notes.append(f"preprocessing fit at line {first_fit} precedes split at line {split_position} (vocabulary-only)")

    return mechanical, strict, split_position, first_fit, notes



def _split_vars(v: "_ConfigVisitor") -> tuple[str | None, str | None]:
    """Names of the feature/target variables actually handed to the split.

    Generated code does not always call them X and y (e.g. X_scaled), so the
    injector must discover the real names rather than assume them.
    """
    if not v.split_args:
        return None, None
    _, args = min(v.split_args, key=lambda t: t[0])
    feat = args[0] if len(args) >= 1 else None
    targ = args[1] if len(args) >= 2 else None
    return feat, targ


def _preprocessing_chain(v: "_ConfigVisitor") -> list[str]:
    return [n for _, n, _ in sorted(v.preprocessors, key=lambda item: item[0])]


def _preprocessing_params(v: "_ConfigVisitor") -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {}
    for _, name, step_params in sorted(v.preprocessors, key=lambda item: item[0]):
        key = name
        if key in seen:
            seen[key] += 1
            key = f"{name}#{seen[name]}"
        else:
            seen[key] = 1
        if step_params:
            params[key] = step_params
    return params


def _split_method(v: "_ConfigVisitor") -> tuple[str | None, dict[str, Any]]:
    if not v.split_methods:
        return None, {}
    _, method, params = min(v.split_methods, key=lambda item: item[0])
    return method, params


def _feature_selector(v: "_ConfigVisitor") -> str | None:
    if not v.feature_selectors:
        return None
    _, name = min(v.feature_selectors, key=lambda item: item[0])
    return name


def extract_config(code: str) -> StepConfig:
    """Parse code into a StepConfig, falling back to regex when AST fails."""
    if not code.strip():
        return StepConfig(model_class="", parse_quality="unparseable", notes=["empty code"])

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _regex_fallback(code, f"SyntaxError: {exc.msg} (line {exc.lineno})")

    v = _ConfigVisitor()
    v.visit(tree)

    if not v.models:
        # AST parsed but no recognisable model - partial information only
        mech, strict, split_pos, first_fit, notes = _analyse_leakage(v)
        split_use = min(v.split_use_lines) if v.split_use_lines else None
        feat_var, targ_var = _split_vars(v)
        return StepConfig(
            model_class="",
            hyperparameters={},
            preprocessing_chain=_preprocessing_chain(v),
            split_position=split_pos,
            split_use_position=split_use,
            feature_var=feat_var,
            target_var=targ_var,
            first_fit_position=first_fit,
            preprocessing_before_split=mech,
            leaky_before_split=strict,
            parse_quality="partial_ast",
            preprocessing_params=_preprocessing_params(v),
            split_method=_split_method(v)[0],
            split_params=_split_method(v)[1],
            feature_selector=_feature_selector(v),
            notes=notes + ["no model class identified"],
        )

    # the last model instantiation is the one actually trained in most AIDE code
    _, model_class, hyperparameters = v.models[-1]
    mech, strict, split_pos, first_fit, notes = _analyse_leakage(v)
    split_use = min(v.split_use_lines) if v.split_use_lines else None
    feat_var, targ_var = _split_vars(v)
    split_method, split_params = _split_method(v)

    return StepConfig(
        model_class=model_class,
        hyperparameters=hyperparameters,
        preprocessing_chain=_preprocessing_chain(v),
        split_position=split_pos,
        split_use_position=split_use,
        feature_var=feat_var,
        target_var=targ_var,
        first_fit_position=first_fit,
        preprocessing_before_split=mech,
        leaky_before_split=strict,
        parse_quality="full_ast",
        preprocessing_params=_preprocessing_params(v),
        split_method=split_method,
        split_params=split_params,
        feature_selector=_feature_selector(v),
        notes=notes,
    )


def _regex_fallback(code: str, reason: str) -> StepConfig:
    models = REGEX_PATTERNS["model_class"].findall(code)
    preps = REGEX_PATTERNS["preprocessor"].findall(code)

    split_m = REGEX_PATTERNS["split"].search(code)
    split_pos = code[: split_m.start()].count("\n") + 1 if split_m else None

    ft_m = REGEX_PATTERNS["fit_transform"].search(code)
    fit_pos = code[: ft_m.start()].count("\n") + 1 if ft_m else None

    leaky = bool(split_pos and fit_pos and fit_pos < split_pos)

    return StepConfig(
        model_class=models[-1] if models else "",
        hyperparameters={},
        preprocessing_chain=preps,
        split_position=split_pos,
        first_fit_position=fit_pos,
        preprocessing_before_split=leaky,
        leaky_before_split=leaky,
        parse_quality="regex_fallback",
        split_method=split_m.group(1) if split_m else None,
        notes=[reason],
    )


# --- canonicalisation ------------------------------------------------------
def _normalize_value(val: Any) -> str:
    if val is None or val == "auto" or val == "None":
        return "auto"
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}"
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(_normalize_value(v) for v in sorted(val, key=str)) + "]"
    return str(val).lower()


def canonical_hash(config: StepConfig) -> str:
    """Order-preserving v2 hash of semantically meaningful config parts."""
    parts = [f"model={config.model_class.split('.')[-1].lower()}"]
    hp = ",".join(
        f"{key}={_normalize_value(config.hyperparameters[key])}"
        for key in sorted(config.hyperparameters.keys())
    )
    parts.append(f"hp={hp}")

    prep_parts = []
    used: dict[str, int] = {}
    for step_name in config.preprocessing_chain:
        used[step_name] = used.get(step_name, 0) + 1
        param_key = step_name if used[step_name] == 1 else f"{step_name}#{used[step_name]}"
        step_params = config.preprocessing_params.get(param_key, {})
        if step_params:
            param_str = ",".join(
                f"{k}={_normalize_value(v)}" for k, v in sorted(step_params.items())
            )
            prep_parts.append(f"{step_name.lower()}({param_str})")
        else:
            prep_parts.append(step_name.lower())
    parts.append(f"prep={'|'.join(prep_parts)}")

    if config.split_method:
        split = config.split_method.lower()
        if config.split_params:
            params = ",".join(
                f"{k}={_normalize_value(v)}" for k, v in sorted(config.split_params.items())
            )
            split = f"{split}({params})"
        parts.append(f"split={split}")

    if config.feature_selector:
        parts.append(f"selector={config.feature_selector.lower()}")

    canonical = "|".join(parts)
    hash_hex = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{HASH_VERSION}:{hash_hex}"


def _check_pipeline_leakage(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in PIPELINE_WRAPPERS:
                return False
    return True


def _is_leaky_fit_from_code(code: str) -> bool:
    """Code-only AST leakage check for reconciled synthetic candidates."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    fit_positions: list[int] = []
    target_transform_positions: list[int] = []
    split_positions: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"fit", "fit_transform"}:
            fit_positions.append(node.lineno)
        if isinstance(func, ast.Attribute) and func.attr in {"groupby", "transform", "map"}:
            names = {n.id.lower() for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & {"y", "target", "labels", "label", "_cp3_y"}:
                target_transform_positions.append(node.lineno)
        name = _call_name(node)
        if name == "train_test_split" or (isinstance(func, ast.Attribute) and func.attr == "split"):
            split_positions.append(node.lineno)

    if not split_positions:
        return _check_pipeline_leakage(tree)
    earliest_split = min(split_positions)
    return any(pos < earliest_split for pos in fit_positions + target_transform_positions)
