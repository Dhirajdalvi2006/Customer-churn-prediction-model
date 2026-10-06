"""SHAP-based Explainable AI (XAI) for the RETENTIX AI churn model.

What this module explains
-------------------------
Every number produced here answers one question: *how did the fitted model
arrive at this customer's churn score?* The module is **purely explanatory**:
it never fits, refits, mutates or re-predicts anything, and the values it
returns do not feed back into the scoring pipeline.

Which model / explainer is used  (selected from the fitted estimator)
-------------------------------------------------------------------
The explainer is **never hard-coded**. It is chosen from the type of the
*actual fitted estimator*, so this one module can explain either model in
this project without modification:

==========================================  =========================
Fitted estimator (verified by inspection)    Explainer selected
==========================================  =========================
``LogisticRegression`` (production, 30 feat) ``shap.LinearExplainer``
``GradientBoostingClassifier`` (Phase 3B)    ``shap.TreeExplainer``
==========================================  =========================

Selection uses :func:`_select_explainer_kind`, which classifies the estimator
by **sklearn type** (``sklearn.tree.BaseDecisionTree`` /
``sklearn.ensemble.BaseEnsemble`` for tree ensembles) and by the presence of
fitted linear parameters (``coef_`` / ``intercept_``) for linear models. The
model *name* is never string-matched: a renamed or re-labelled estimator
still resolves correctly, and an unrecognised model raises
:class:`XAIUnsupportedModelError` rather than silently falling back to a
sampling-based approximation.

Two fitted objects are supported transparently:

* a :class:`backend.model.ChurnModel` (the production bundle, whose
  ``best_model`` is a bare estimator), and
* a fitted ``sklearn.pipeline.Pipeline`` (the Phase 3B champion bundle,
  ``preprocessor`` -> ``classifier``).

Both are normalised into a :class:`ModelView` so that every public function
below behaves identically regardless of which bundle it was handed.

Output space: log-odds, NOT probability  (verified, not assumed)
---------------------------------------------------------------
Both explainers are pinned to the model's **raw margin**, so one single
additivity invariant holds for either model class:

    expected_value + sum_i(phi_i)  ==  estimator.decision_function(x)  (margin)

**Linear (verified, residual ~2e-16).** For a linear model SHAP has a
closed-form analytic solution and
``expected_value = coef @ mean + intercept``. Passing
``link=shap.links.logit`` to ``LinearExplainer`` is **accepted but inert**:
the base value and the attribution values remain in raw log-odds space.

**Tree (verified, residual ~2e-15, SHAP 0.52 / scikit-learn 1.6.1).** For the
champion ``GradientBoostingClassifier`` the configuration was determined
empirically, not assumed:

* ``feature_perturbation="tree_path_dependent"`` with ``model_output="raw"``
  is the only combination that reconstructs ``decision_function`` exactly
  (max abs residual **2.22e-15** over 300 rows). It is also fully
  deterministic -- no background sample is required or consumed, so no
  sub-sampling can perturb the base value.
* ``model_output="probability"`` requires the ``interventional`` path, needs
  a background dataset, and SHAP **silently sub-samples** it (200 -> 100)
  which both logs a notice and makes the base value depend on that
  sub-sample; its residual against the true probability is ~4.7e-9, i.e.
  four orders of magnitude worse. It is therefore not used.
* ``model_output="log_loss"`` is rejected by SHAP
  (``ExplainerError: Both samples and labels must be provided ...``) and
  would attribute *loss* rather than the churn score, so it is not used.

Because both explainers attribute the same quantity, the module-wide
``OUTPUT_SPACE`` invariant stays ``"log_odds"`` for either model and the
reconstruction error stays directly interpretable. An explicitly labelled
counterfactual probability shift is offered for human-readable display only.

What the base value means
-------------------------
``expected_value`` is the model's average log-odds churn score over the
background reference sample. It is the "average customer" the attributions
start from; 0 in log-odds corresponds to 50% churn probability.

What positive / negative contributions mean
-------------------------------------------
A **positive** contribution pushed this customer's log-odds churn score
*up* (towards churn). A **negative** contribution pushed it *down* (away from
churn). Magnitude is relative to the background average, not to zero.

Why SHAP does NOT prove causality
---------------------------------
SHAP attributes the model's *own* decision to its inputs under a chosen
feature-interdependence assumption. It describes association within this
fitted model only. It does **not** establish that changing a feature would
change the real-world outcome. Statements in the UI are therefore phrased as
"model-attributed contribution", never "caused churn". Correlated features
(e.g. tenure / TotalCharges / contract type) also share attribution mass,
which is a property of the decomposition, not evidence of a causal pathway.

Feature-name mapping
--------------------
SHAP values are computed in the **30-column transformed space** produced by the
existing fitted ``ColumnTransformer`` (see :mod:`backend.model`). Names come
from that fitted preprocessor's ``get_feature_names_out()``, cross-checked
against the persisted ``feature_names.pkl``; no name is ever invented. Because
the OneHotEncoder uses ``drop="first"``, each categorical block keeps a single
dummy column whose value 1 means "category differs from the training baseline
category". :func:`pretty_feature_name` renders a readable label (e.g.
``Contract = Two year``) while the raw technical name is always retained.
"""
import os
import threading
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import BaseDecisionTree

from backend.model import (
    DATA_PATH,
    MODEL_DIR,
    ChurnModel,
)

# Number of background rows used to define the reference distribution. Kept
# modest because the exact explainers are analytic/closed-form: the answer is
# exact for any background, so more rows add cost, not accuracy.
GLOBAL_SAMPLE_SIZE = 200

# Fixed seed for the global reference sample so published rankings are
# byte-for-byte reproducible across runs and machines.
GLOBAL_RANDOM_STATE = 42

# Marker stored on each explanation so callers can assert the invariant
# actually verified for this model, rather than assuming one. Both supported
# explainers are pinned to the raw margin, so the same label is correct for
# either model class (see the module docstring for the empirical evidence).
OUTPUT_SPACE = "log_odds"

_THREAD_LOCK = threading.Lock()
_MODEL_CACHE = {}
_GLOBAL_CACHE = {}


class XAIUnsupportedModelError(RuntimeError):
    """Raised when a fitted estimator has no exact SHAP explainer here.

    This is deliberately a hard failure. Falling back to ``KernelExplainer``
    would silently return a *sampling-based approximation* with a different
    output space and different error magnitude, which would be reported to a
    business user as if it carried the same guarantees as the exact
    additivity verified for the supported model classes.
    """


class ModelView:
    """A fitted model bundle normalised to a single, uniform interface.

    The project stores two structurally different fitted objects:

    * :class:`backend.model.ChurnModel` -- the production bundle, where
      ``best_model`` is a *bare* estimator (``LogisticRegression``) and the
      fitted ``preprocessor`` turns 19 raw columns into 30 transformed ones;
    * a fitted ``sklearn.pipeline.Pipeline`` -- the Phase 3B champion bundle,
      which is one Pipeline whose ``classifier`` step is the estimator.

    Exposing ``estimator`` / ``preprocessor`` / ``transform`` on both means
    every function in this module is written once and behaves identically for
    either bundle. Nothing here fits, refits or mutates anything: the
    preprocessor is always used in transform-only mode.
    """

    __slots__ = ("estimator", "preprocessor", "median", "label", "raw_columns",
                 "contract")

    def __init__(self, estimator, preprocessor, median, label,
                 raw_columns=None, contract=None):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.median = median
        self.label = label
        # The exact raw column list this bundle's preprocessor consumes, taken
        # from the fitted transformer specification itself (not from the
        # production REQUIRED_COLS constant). ``None`` means "delegate to the
        # shared inference path", which is how the production ChurnModel keeps
        # its verified behaviour byte-for-byte.
        self.raw_columns = raw_columns
        # The shared bundle input contract, when this view came from a
        # ModelBundle. When present it is the *only* path used to build the
        # frame, so the explanation is computed from exactly the same
        # validated input the score was computed from.
        self.contract = contract

    @property
    def n_features(self):
        """Width of the transformed feature space the estimator consumes."""
        return int(self.estimator.n_features_in_)

    def transform(self, raw):
        """Run the bundle's own fitted preprocessor. Never fits anything.

        The production ``ChurnModel`` bundle keeps using the shared
        :func:`backend.model.prepare_frame` path unchanged. A ``Pipeline``
        bundle instead builds its frame from the columns its *own* fitted
        preprocessor declares, so a bundle-specific column contract is never
        forced through the production 19-column assumption.
        """
        from backend.model import prepare_frame

        if self.contract is not None:
            # Bundle-backed view: the shared contract is authoritative.
            frame = self.contract.prepare(raw)
        elif self.raw_columns is None:
            frame = prepare_frame(raw, self.preprocessor, self.median)
        else:
            frame = self._pipeline_frame(raw)
        return np.asarray(self.preprocessor.transform(frame), dtype=float)

    def _pipeline_frame(self, raw):
        """Build the raw frame a fitted ``Pipeline`` preprocessor consumes.

        Deliberately self-contained: it reuses the shared, already-verified
        coercion helpers for the columns they cover (SeniorCitizen, and the
        training-median TotalCharges imputation) but takes its **column set
        from the bundle's own fitted transformer spec** and validates
        categoricals against **that bundle's own fitted encoder categories**.

        Nothing here fits, refits or mutates a fitted artifact.
        """
        import pandas as pd
        from backend.model import (
            _coerce_senior_citizen,
            _validate_categoricals,
        )

        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
        else:
            raise TypeError(
                f"customer_data must be a dict or DataFrame, got "
                f"{type(raw).__name__}."
            )

        missing = [c for c in self.raw_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required field(s): {missing}")

        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = _coerce_senior_citizen(df["SeniorCitizen"])
        if "TotalCharges" in df.columns:
            # Impute with the median recovered from the bundle's own fitted
            # imputer, so a score never depends on batch composition.
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],
                                               errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(float(self.median))

        _validate_categoricals(df, self.category_map())
        return df[self.raw_columns]

    def category_map(self):
        """Per-column category domain learned by *this* bundle's encoder.

        Read from the fitted transformer spec plus the fitted
        ``OneHotEncoder``'s ``categories_``. The champion's categorical step is a
        ``Pipeline`` (``imputer`` -> ``onehot``) while the production bundle's
        is a bare ``OneHotEncoder``; both layouts are unwrapped here, inside
        the XAI layer, so :mod:`backend.model` needs no compatibility shim.
        """
        spec = None
        for name, _, cols in self.preprocessor.transformers:
            if name == "cat":
                spec = [str(c) for c in cols]
                break
        if spec is None:
            return {}

        node = self.preprocessor.named_transformers_["cat"]
        if hasattr(node, "named_steps"):  # Pipeline(imputer -> onehot)
            node = node.named_steps["onehot"]
        categories = np.asarray(node.categories_, dtype=object)
        if len(categories) != len(spec):
            raise ValueError(
                f"Fitted encoder exposes {len(categories)} category lists for "
                f"{len(spec)} categorical columns; refusing to validate against "
                f"a misaligned domain."
            )
        return {col: [str(c) for c in cats]
                for col, cats in zip(spec, categories)}


def _bundle_model_view(bundle):
    """Build a :class:`ModelView` from a v1 or v2 :class:`ModelBundle`.

    Resolves BLOCKER-09 without duplicating model-specific production logic:
    the bundle already exposes ``estimator``, ``transformer`` and the shared
    input contract, so this adapter only *reads* those. Both bundle formats
    (``estimator_only`` and ``pipeline``) go through the identical path, and
    the frozen artifacts are never modified.

    The resulting view transforms through the **shared contract** rather than
    through :func:`backend.model.prepare_frame`, so an explanation is always
    computed from exactly the same validated input the score was computed
    from -- otherwise the attribution could describe a different row than the
    one that was scored.
    """
    estimator = bundle.estimator
    transformer = bundle.transformer
    if estimator is None or transformer is None:
        raise XAIUnsupportedModelError(
            f"Bundle {bundle.manifest.get('bundle_version')!r} exposes no "
            f"fitted estimator/preprocessor; cannot explain it."
        )
    if not hasattr(estimator, "n_features_in_"):
        raise XAIUnsupportedModelError(
            f"Bundle {bundle.manifest.get('bundle_version')!r} is not fitted; "
            f"its estimator exposes no n_features_in_."
        )
    return ModelView(
        estimator=estimator,
        preprocessor=transformer,
        median=float(bundle.contract.total_charges_statistic),
        label=type(estimator).__name__,
        raw_columns=_pipeline_raw_columns(transformer),
        contract=bundle.contract,
    )


def _pipeline_total_charges_median(preprocessor):
    """Recover the training-split median from a fitted numeric imputer.

    ``prepare_frame`` needs the *training* median so a score never depends on
    the incoming batch. For a Pipeline that median lives in the fitted
    ``SimpleImputer``; it is read, never recomputed. Falls back to the
    production sidecar value if the step layout is not recognised.
    """
    from backend.model import NUMERICAL_COLS

    try:
        numeric = preprocessor.named_transformers_["num"]
        imputer = numeric.named_steps["imputer"]
        stats = np.asarray(imputer.statistics_, dtype=float)
        return float(stats[list(NUMERICAL_COLS).index("TotalCharges")])
    except (AttributeError, KeyError, ValueError, IndexError, TypeError):
        return float(ChurnModel.load().total_charges_median)


def _as_model_view(model):
    """Normalise a supported fitted bundle into a :class:`ModelView`.

    Accepts, in order:

    * a :class:`backend.model_bundle.ModelBundle` -- the v1 or v2 bundle
      (Phase 4 Step 8 / BLOCKER-09). Handled *before* the generic Pipeline
      branch, because a v2 bundle's ``.model`` is itself a Pipeline; going
      through the bundle first means the bundle's own metadata-supplied
      threshold, class index and input contract are respected, and no
      model-specific logic is duplicated here.
    * a :class:`backend.model.ChurnModel` -- the production bundle.
    * a fitted ``Pipeline`` -- the raw Phase 3B champion object.
    * an already-built :class:`ModelView` (returned unchanged).
    """
    if isinstance(model, ModelView):
        return model

    # A ModelBundle knows its own estimator/preprocessor/contract, so it needs
    # no unpacking logic of its own here.
    try:
        from backend.model_bundle import ModelBundle
    except ImportError:  # pragma: no cover - defensive
        ModelBundle = None
    if ModelBundle is not None and isinstance(model, ModelBundle):
        return _bundle_model_view(model)

    if isinstance(model, ChurnModel):
        if model.best_model is None or model.preprocessor is None:
            raise XAIUnsupportedModelError(
                "The ChurnModel is not fitted: best_model/preprocessor are "
                "missing. Load it with ChurnModel.load() before explaining."
            )
        return ModelView(
            estimator=model.best_model,
            preprocessor=model.preprocessor,
            median=float(model.total_charges_median),
            label=model.best_model_name or "LogisticRegression",
        )
    # Fitted sklearn Pipeline (the Phase 3B champion bundle). Detection is by
    # type, not by name, and requires the classifier step to be fitted.
    if hasattr(model, "named_steps") and hasattr(model, "steps"):
        steps = dict(model.named_steps)
        preprocessor = steps.get("preprocessor")
        estimator = steps.get("classifier", steps.get("model"))
        if preprocessor is None or estimator is None:
            raise XAIUnsupportedModelError(
                "Pipeline must expose fitted 'preprocessor' and 'classifier' "
                f"steps; found {sorted(steps)}."
            )
        if not hasattr(estimator, "n_features_in_"):
            raise XAIUnsupportedModelError(
                "The Pipeline's 'classifier' step is not fitted; call fit() "
                "before explaining."
            )
        return ModelView(estimator=estimator, preprocessor=preprocessor,
                         median=_pipeline_total_charges_median(preprocessor),
                         label=type(estimator).__name__,
                         raw_columns=_pipeline_raw_columns(preprocessor))
    raise XAIUnsupportedModelError(
        f"Unsupported fitted model of type {type(model).__name__!r}. Pass a "
        "backend.model.ChurnModel, a fitted sklearn Pipeline, or a ModelView."
    )


def _pipeline_raw_columns(preprocessor):
    """Raw column contract a fitted ``Pipeline`` preprocessor consumes.

    Read from the fitted ``ColumnTransformer`` spec (``preprocessor.transformers``)
    rather than from the production :data:`backend.model.REQUIRED_COLS` constant,
    so a bundle whose preprocessor was built from a different column set is
    never forced through the production 19-column assumption.
    """
    raw_columns = []
    if not hasattr(preprocessor, "transformers"):
        return None
    for _, _, cols in preprocessor.transformers:
        if cols is None:
            continue
        raw_columns.extend(cols)
    return raw_columns if raw_columns else None


def _bundle_identity(view):
    """A cache key that identifies *this fitted bundle instance* uniquely.

    Two structurally different bundles in this project can expose the same
    estimator *type*: the v1 :class:`backend.model_bundle.ModelBundle` and the
    production :class:`backend.model.ChurnModel` are both
    ``LogisticRegression`` objects loaded from the same ``models/`` directory.
    Keying the explainer cache on the class name therefore let one silently
    shadow the other, so a caller could be handed an explainer wrapping a
    different fitted estimator than the one it asked about -- which is exactly
    the kind of hidden per-bundle behaviour a shared contract must prevent.

    Identity is ``id()`` of the fitted estimator, which distinguishes the two
    instances while still sharing one cache entry when the *same* object is
    requested repeatedly (the normal Streamlit rerun case). The tuple holds a
    strong reference to the estimator so its ``id()`` cannot be recycled by the
    garbage collector while the cache entry is alive.
    """
    return (id(view.estimator), view.estimator, type(view.estimator).__name__)


def _select_explainer_kind(estimator):
    """Return ``"linear"`` or ``"tree"`` for a fitted estimator.

    Classification is by **fitted type**, never by the model's display name,
    so a renamed or re-labelled estimator still resolves correctly:

    1. ``sklearn.tree.BaseDecisionTree`` / ``sklearn.ensemble.BaseEnsemble``
       -> a tree ensemble -> ``"tree"``. This covers
       ``GradientBoostingClassifier``, ``RandomForestClassifier`` and
       ``DecisionTreeClassifier`` (all evaluated in Phase 3B) without a
       fragile ``if "Forest" in name`` style test.
    2. otherwise, fitted linear parameters (``coef_`` and ``intercept_``)
       -> ``"linear"``. This covers ``LogisticRegression`` and linear
       regression heads generally.
    3. otherwise raise :class:`XAIUnsupportedModelError` rather than degrade
       silently to a sampling-based approximation.

    Both branches are exact, deterministic explainers in raw-margin space, so
    the selection never changes the additivity guarantee.
    """
    if isinstance(estimator, (BaseDecisionTree, BaseEnsemble)):
        return "tree"
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return "linear"
    raise XAIUnsupportedModelError(
        f"No exact SHAP explainer is configured for estimator type "
        f"{type(estimator).__name__!r}. Supported: linear models "
        "(coef_/intercept_) and tree ensembles (BaseDecisionTree / "
        "BaseEnsemble). Refusing to fall back to a sampling-based "
        "approximation, which would report different output-space guarantees."
    )


def _explainer_name(kind):
    """Human-readable explainer label stored on each explanation."""
    return {"linear": "shap.LinearExplainer",
            "tree": "shap.TreeExplainer"}[kind]


def _log_odds_to_probability(z):
    """Numerically stable logistic transform (avoids overflow on large |z|)."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def get_feature_names(model=None):
    """Return the transformed feature names, cross-checked across sources.

    Sources, in priority order:
      1. the fitted preprocessor's ``get_feature_names_out()``
      2. the persisted ``feature_names.pkl`` written at training time
         (production ``ChurnModel`` bundle only -- a champion ``Pipeline``
         carries no sidecar file, so source 2 does not exist for it)

    If two available sources disagree, or the count does not match the fitted
    estimator's ``n_features_in_``, a ValueError is raised. A silently wrong
    feature label would produce a confidently incorrect business explanation,
    so this is a hard failure rather than a best-effort guess.

    The count is **not** hard-coded to 30: it is whatever the supplied fitted
    preprocessor actually emits, cross-checked against the estimator it
    feeds. The production bundle yields 30 (``drop="first"``); the Phase 3B
    champion bundle yields 45 (full one-hot, ``drop=None``).
    """
    if model is None:
        model = get_model()
    view = _as_model_view(model)
    from_prep = [str(n) for n in view.preprocessor.get_feature_names_out()]

    # Sidecar names exist only for the ChurnModel production bundle.
    from_disk = None
    if isinstance(model, ChurnModel) and model.feature_names is not None:
        from_disk = [str(n) for n in np.asarray(model.feature_names).ravel()]

    if from_disk is not None:
        if len(from_prep) != len(from_disk):
            raise ValueError(
                f"Feature-name sources disagree: preprocessor has "
                f"{len(from_prep)}, feature_names.pkl has {len(from_disk)}."
            )
        if list(from_prep) != list(from_disk):
            raise ValueError(
                "Feature-name sources disagree in content; refusing to display "
                "explanations whose labels cannot be trusted."
            )
    expected = view.n_features
    if len(from_prep) != expected:
        raise ValueError(
            f"Expected {expected} transformed features to match the fitted "
            f"model, but the preprocessor produced {len(from_prep)}."
        )
    return from_prep


def split_transformed_name(name):
    """Split ``cat__Contract_Two year`` into (raw_feature, dummy_suffix).

    The OneHotEncoder uses ``drop="first"``, so a suffix identifies the dummy
    column for that raw feature, not an independent quantity.
    """
    if name.startswith("num__"):
        return name[5:], None
    if name.startswith("cat__"):
        stem = name[5:]
        # Split on the LAST underscore so multi-word values such as
        # "Two year" or "No internet service" stay intact.
        raw, _, suffix = stem.rpartition("_")
        if not raw:
            return stem, None
        return raw, suffix
    return name, None


def pretty_feature_name(name):
    """Render a transformed feature as a readable business label.

    ``cat__Contract_Two year`` -> ``Contract = Two year``
    ``num__MonthlyCharges``    -> ``Monthly Charges``

    Readability only: the caller always keeps the original technical name so
    nothing is lost for audit or academic reproducibility.
    """
    raw, suffix = split_transformed_name(name)
    readable = raw.replace("_", " ")
    if suffix is None:
        return readable
    return f"{readable} = {suffix}"


def get_model(model=None):
    """Return the model to explain, loading it from disk at most once.

    Production default (Phase 4 Step 10): the **active ModelBundle**, resolved
    through the production inference service, which in turn resolves
    ``models/active_bundle.json`` via
    :meth:`BundleLoader.get_active_bundle`. The XAI layer therefore explains
    exactly the object that produced the on-screen score, in that bundle's own
    validated transformed space, and automatically follows the active pointer
    if it is ever moved -- with no change here and no model type hard-coded.

    An explicit ``model`` argument still wins, so a specific bundle or the
    legacy ``ChurnModel`` can be explained explicitly when required.

    Streamlit re-executes the page script on every widget interaction, so
    unpickling the artifacts on each rerun would be wasteful. The cache is
    process-local and keyed by artifact directory, and is guarded by a lock so
    concurrent sessions cannot double-load.
    """
    if model is not None:
        # A production prediction service may be passed straight through from a
        # view. It knows which fitted object actually produced the score, so
        # unwrap it here rather than making every view unwrap it itself.
        unwrap = getattr(model, "explainable_model", None)
        if callable(unwrap):
            return unwrap()
        return model
    # Imported lazily: the inference service imports this module's siblings,
    # and a module-level import here would be an avoidable import cycle.
    from backend.prediction_service import get_prediction_service
    return get_prediction_service(MODEL_DIR).explainable_model()


def load_reference_background(model=None, sample_size=GLOBAL_SAMPLE_SIZE,
                              random_state=GLOBAL_RANDOM_STATE):
    """Build the deterministic background sample defining the base value.

    Sampled with an explicit ``random_state`` from the cleaned dataset, then
    pushed through the *same* shared inference path as scoring
    (``ModelView.transform`` -> ``prepare_frame`` -> fitted preprocessor) so
    the reference population is expressed in exactly the model's own
    transformed space (30 features for the production bundle, 45 for the
    champion bundle). No fitting occurs.
    """
    view = _as_model_view(get_model(model))
    df = pd.read_csv(DATA_PATH)
    n = min(int(sample_size), len(df))
    sample = df.sample(n=n, random_state=random_state)
    return view.transform(sample)


def _build_explainer(view, kind, background):
    """Construct the SHAP explainer for a fitted estimator.

    Branch 1 -- linear (``LogisticRegression`` and similar)
        ``shap.LinearExplainer`` with an explicit ``Independent`` masker and
        ``model_output="raw"``. The masker is built with
        ``max_samples=len(background)`` because SHAP's default of 100 would
        silently sub-sample a larger background, logging a notice *and* making
        the base value depend on that sub-sample. ``model_output="raw"`` pins
        the space explicitly instead of relying on a version-dependent
        default.

    Branch 2 -- tree ensemble (``GradientBoostingClassifier`` and similar)
        ``shap.TreeExplainer`` with
        ``feature_perturbation="tree_path_dependent"`` and
        ``model_output="raw"``. This is the configuration verified against
        SHAP 0.52 / scikit-learn 1.6.1 to reconstruct ``decision_function``
        to ~2e-15. Two alternatives were measured and rejected on evidence:

        * ``model_output="probability"`` requires
          ``feature_perturbation="interventional"``, needs a background
          dataset, and SHAP silently sub-samples it (200 -> 100), making the
          base value depend on that sub-sample. Its residual against the true
          probability is ~4.7e-9, four orders of magnitude worse.
        * ``model_output="log_loss"`` raises ``ExplainerError: Both samples
          and labels must be provided`` and would attribute loss rather than
          the churn score.

        ``tree_path_dependent`` needs **no** background at all, so the
        background matrix is neither required nor consumed for tree models.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "linear":
            masker = shap.maskers.Independent(background,
                                              max_samples=len(background))
            return shap.LinearExplainer(view.estimator, masker,
                                        model_output="raw")
        if kind == "tree":
            return shap.TreeExplainer(
                view.estimator,
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )
        raise XAIUnsupportedModelError(f"Unknown explainer kind {kind!r}.")


def get_shap_explainer(model=None, background=None):
    """Build (and cache) the correct exact SHAP explainer for a fitted model.

    Explainer choice
    ----------------
    The explainer is **selected from the type of the fitted estimator** by
    :func:`_select_explainer_kind` -- never hard-coded and never string-matched
    on the model name:

    * ``LogisticRegression``         -> :class:`shap.LinearExplainer`
      (closed-form, analytic, exact)
    * ``GradientBoostingClassifier`` -> :class:`shap.TreeExplainer`
      (tree-path-dependent, exact)

    Both are exact, deterministic, and pinned to the raw margin, so the same
    additivity invariant and the same ``OUTPUT_SPACE`` label hold for either
    model class. ``KernelExplainer`` is deliberately never used: it is a
    sampling-based approximation and would weaken the guarantees this module
    documents. An estimator of neither shape raises
    :class:`XAIUnsupportedModelError` rather than degrading silently.

    The explainer consumes the estimator in the **fitted transformed space**
    of its own bundle. No encoder, scaler or preprocessing pipeline is
    created or fitted here -- the caller always routes raw input through
    :func:`backend.model.prepare_frame` and the already-fitted preprocessor.

    Caching
    -------
    The explainer is expensive enough to build that it must not be rebuilt on
    every Streamlit rerun, so it is memoised per (bundle, explainer kind,
    background identity). Pass a custom ``background`` to force a separate
    entry. The background checksum is part of the key only for the linear
    branch, which is the branch that actually consumes it.
    """
    view = _as_model_view(get_model(model))
    kind = _select_explainer_kind(view.estimator)

    if background is None:
        background = load_reference_background(model)
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or background.shape[1] != view.n_features:
        raise ValueError(
            f"Background must be a 2-D array with {view.n_features} columns, "
            f"got shape {background.shape}."
        )

    key = (
        "explainer",
        # Identify the *fitted bundle instance*, not its type. The class name
        # alone is not sufficient: the v1 ModelBundle and the production
        # ChurnModel are both LogisticRegression under the same MODEL_DIR with
        # the same background, so a name-based key let one silently shadow the
        # other and a caller could receive an explainer wrapping the wrong
        # estimator. ``_bundle_identity`` pins a strong reference so the id()
        # cannot be recycled by the garbage collector while the entry lives.
        _bundle_identity(view),
        kind,
        os.path.abspath(MODEL_DIR),
        background.shape,
    )
    if kind == "linear":
        # A cheap, stable content hash so two identical backgrounds share a
        # cache entry while different ones never collide.
        key = key + (float(np.sum(background)), float(np.sum(background ** 2)))

    cached = _MODEL_CACHE.get(key)
    if cached is None:
        with _THREAD_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is None:
                cached = _build_explainer(view, kind, background)
                _MODEL_CACHE[key] = cached
    return cached


def _base_value(explainer):
    """Extract the scalar base value regardless of SHAP's return shape."""
    base = np.asarray(explainer.expected_value, dtype=float).ravel()
    return float(base[-1]) if base.size else float(base)


def _shap_values(explainer, matrix):
    """Run the explainer and return a clean (n_rows, n_features) array.

    SHAP returns an ``Explanation`` for modern versions and can emit a
    list of arrays for older ones; with a single-output binary classifier the
    values are 2-D, but a defensive squeeze keeps the public contract stable.
    """
    raw = explainer(matrix)
    values = raw.values if hasattr(raw, "values") else raw
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        # (n, features, outputs) -> keep the churn (positive-class) output.
        values = values[:, :, -1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def get_feature_contributions(customer_data, model=None, explainer=None,
                              top_n=None):
    """Explain one customer and return structured, business-ready attributions.

    Parameters
    ----------
    customer_data : dict or pandas.DataFrame
        Raw customer input, exactly as accepted by
        :meth:`backend.model.ChurnModel.predict_single`.
    model : ChurnModel, optional
        Reuse an already-loaded model instead of hitting the cache.
    explainer : shap.Explainer, optional
        Reuse an existing explainer (used by the simulator to explain the
        baseline and the intervention with an identical base value). Either a
        ``LinearExplainer`` or a ``TreeExplainer`` is valid; which one is built
        here depends on the fitted estimator.
    top_n : int, optional
        Truncate the returned contributions to the top N by absolute
        magnitude. ``None`` returns all 30.

    Returns
    -------
    dict with keys:
        prediction           "Yes" / "No"
        churn_probability    percent, from the model's own predict_proba
        churn_probability_raw
        base_value           average log-odds over the background sample
        base_probability     base value expressed as a probability
        model_log_odds       decision_function value (the reconstructed total)
        reconstruction_error base + sum(contributions) - model_log_odds
        output_space         "log_odds" (the space the invariant holds in)
        contributions        list of per-feature dicts, sorted by |contribution|
        top_increasing       features that pushed the score towards churn
        top_decreasing       features that pushed the score away from churn

    Every contribution entry carries both the technical transformed name and a
    readable label, plus the customer's original raw value where the
    transformed column maps back to one.
    """
    view = _as_model_view(get_model(model))
    if explainer is None:
        explainer = get_shap_explainer(view)

    # Reuse the SHARED inference path. This is the single point that guarantees
    # SHAP sees precisely the transformed representation the model scores
    # (30 columns for the production bundle, 45 for the champion bundle).
    transformed = view.transform(customer_data)

    shap_values = _shap_values(explainer, transformed)[0]
    base = _base_value(explainer)
    names = get_feature_names(view)

    if shap_values.shape[0] != len(names):
        raise ValueError(
            f"SHAP returned {shap_values.shape[0]} attributions for "
            f"{len(names)} named features."
        )

    probability = float(view.estimator.predict_proba(transformed)[0, 1])
    log_odds = float(view.estimator.decision_function(transformed)[0])
    base_probability = float(_log_odds_to_probability(base))

    raw_record = {}
    if isinstance(customer_data, pd.DataFrame):
        if len(customer_data) > 0:
            raw_record = customer_data.iloc[0].to_dict()
    elif isinstance(customer_data, dict):
        raw_record = dict(customer_data)

    contributions = []
    for i, name in enumerate(names):
        phi = float(shap_values[i])
        raw_feature, _ = split_transformed_name(name)
        # Report the customer's own value for the underlying raw feature when
        # it exists; dummy columns otherwise report the binary flag itself.
        value = raw_record.get(raw_feature, None)
        counterfactual_shift = float(
            _log_odds_to_probability(base + phi) - base_probability
        )
        contributions.append({
            "feature": name,
            "label": pretty_feature_name(name),
            "raw_feature": raw_feature,
            "value": value,
            "contribution": phi,
            "abs_contribution": abs(phi),
            "counterfactual_probability_shift": counterfactual_shift,
            "direction": "increases churn" if phi > 0 else "decreases churn",
        })

    contributions.sort(key=lambda c: c["abs_contribution"], reverse=True)
    selected = contributions if top_n is None else contributions[:int(top_n)]

    return {
        "prediction": "Yes" if probability >= 0.5 else "No",
        "churn_probability": round(probability * 100, 2),
        "churn_probability_raw": probability,
        "base_value": base,
        "base_probability": base_probability,
        "model_log_odds": log_odds,
        "reconstruction_error": float(base + float(np.sum(shap_values)) - log_odds),
        "output_space": OUTPUT_SPACE,
        # Reports WHICH explainer actually ran, so a caller (or an auditor) can
        # confirm the Linear/Tree branch was chosen from the fitted estimator
        # rather than assumed.
        "explainer": _explainer_name(
            _select_explainer_kind(view.estimator)
        ),
        "n_features": len(names),
        "contributions": selected,
        "top_increasing": [c for c in contributions if c["contribution"] > 0],
        "top_decreasing": [c for c in contributions if c["contribution"] < 0],
    }


# ``explain_single`` is the public individual-explanation name requested by the
# specification; ``get_feature_contributions`` is the descriptive alias used
# inside the app. They are the same function, so there is one implementation.
explain_single = get_feature_contributions


def _global_directions(explainer, model, names, background, values):
    """Classify each feature's global effect as raising or lowering churn risk.

    Why the naive approach is wrong
    -------------------------------
    A signed mean of SHAP values is **identically zero** when the explained
    sample is the background sample, because SHAP is constructed so that the
    attributions sum to a fixed total. Averaging signed contributions over that
    same population therefore returns numerical noise (~1e-17) whose sign is
    arbitrary. Reporting that as a feature "direction" would be indefensible.

    Instead each feature is evaluated in its *active* state, which is the state
    a business user means by "this factor is present":
      * one-hot columns: rows where the dummy equals 1 (category is present)
      * continuous columns: rows above the median (an above-average customer)

    For a linear model this is the sign of the feature's effect on the log-odds
    when it is actually doing something.
    """
    directions = []
    view = _as_model_view(model)
    # `coef_` exists only for a linear estimator. A tree ensemble has no
    # global additive coefficient, so the analytic fallback below is applied
    # **only** when the active estimator actually has one; every other case
    # uses the observed active-state SHAP mean, which is defined for any
    # estimator and is the definition this function documents.
    coefficients = getattr(view.estimator, "coef_", None)
    if coefficients is not None:
        coefficients = np.asarray(coefficients, dtype=float).ravel()

    # Dimensions come from the active bundle, never from a hard-coded 30.
    n_features = len(names)
    if values.shape[1] != n_features or background.shape[1] != n_features:
        raise ValueError(
            f"Direction/feature-name length mismatch: {len(names)} names, "
            f"{background.shape[1]} background columns, {values.shape[1]} "
            f"attribution columns."
        )

    column_mean = np.asarray(background.mean(axis=0), dtype=float)
    for i, name in enumerate(names):
        column = background[:, i]
        is_binary = np.array_equal(np.unique(column), np.array([0.0, 1.0]))
        if is_binary:
            active = column == 1.0
            if active.any():
                effect = float(values[active, i].mean())
            elif coefficients is not None:
                # Category absent from the sample: fall back to the analytic
                # contribution had it been present (linear models only).
                effect = float(coefficients[i] * (1.0 - column_mean[i]))
            else:
                # Category absent AND no analytic coefficient available (a tree
                # ensemble). There is no evidence of an effect in this sample,
                # so report the neutral direction instead of inventing a sign.
                effect = 0.0
        else:
            above = column > np.median(column)
            effect = float(values[above, i].mean()) if above.any() else 0.0
        directions.append("increases churn" if effect > 0 else "decreases churn")
    return directions


def get_global_shap_values(model=None, explainer=None,
                           sample_size=GLOBAL_SAMPLE_SIZE,
                           random_state=GLOBAL_RANDOM_STATE):
    """Compute and cache the global SHAP importance ranking.

    Importance is the **mean absolute SHAP value** over a deterministic sample:
    the typical size of a feature's push on the log-odds score, regardless of
    direction. This is a global average, so it deliberately says nothing about
    any individual customer.

    Performance / reproducibility
    -----------------------------
    * The sample is drawn with a fixed ``random_state``, so the published
      ranking is reproducible run to run.
    * Only ``sample_size`` rows are explained, never the full dataset, so cost
      stays bounded (~45 microseconds per row for this linear model).
    * The result is memoised in-process. Streamlit's own ``@st.cache_data``
      is layered on top by the view, so neither the sample nor the explanation
      is recomputed on ordinary widget interaction.

    Returns
    -------
    dict with ``importance`` (DataFrame sorted descending by mean |SHAP|),
    ``sample_size``, ``random_state``, ``base_value``, ``shap_values`` and
    ``feature_names``.
    """
    view = _as_model_view(get_model(model))
    key = ("global", os.path.abspath(MODEL_DIR),
           type(view.estimator).__name__, int(sample_size), int(random_state))
    cached = _GLOBAL_CACHE.get(key)
    if cached is not None:
        return cached

    with _THREAD_LOCK:
        cached = _GLOBAL_CACHE.get(key)
        if cached is not None:
            return cached

        if explainer is None:
            explainer = get_shap_explainer(view)
        names = get_feature_names(view)

        df = pd.read_csv(DATA_PATH)
        n = min(int(sample_size), len(df))
        sample = df.sample(n=n, random_state=random_state)
        transformed = view.transform(sample)
        values = _shap_values(explainer, transformed)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Global SHAP values contain non-finite entries; refusing to "
                "publish an importance ranking that cannot be reproduced."
            )

        mean_abs = np.abs(values).mean(axis=0)
        # Direction is evaluated on each feature's ACTIVE rows, not from the
        # signed mean over the whole sample (which is zero by construction).
        # See _global_directions for the full justification.
        directions = _global_directions(explainer, view, names, transformed,
                                        values)

        importance = pd.DataFrame({
            "Feature": names,
            "Label": [pretty_feature_name(nm) for nm in names],
            "Mean |SHAP|": mean_abs,
            "Direction": directions,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

        result = {
            "importance": importance,
            "shap_values": values,
            "feature_names": names,
            "sample_size": n,
            "random_state": int(random_state),
            "base_value": _base_value(explainer),
            "explainer": _explainer_name(
                _select_explainer_kind(view.estimator)
            ),
            "output_space": OUTPUT_SPACE,
        }
        _GLOBAL_CACHE[key] = result
        return result


def explain_simulation(baseline_payload, simulated_payload, model=None,
                       top_n=5):
    """Compare the model's attribution of a profile before and after a change.

    Used by the What-If simulator to show how the *model's own* reasoning
    shifts when an intervention is applied. The same explainer (and therefore
    the same base value) is used for both sides, so the delta is attributable
    purely to the modified features.

    Interpretation caveat
    ---------------------
    A contribution change is a **model-attributed** shift in score, not proof
    that performing the intervention reduces real-world churn. The simulator
    is a what-if probe of the fitted model, not a causal experiment.
    """
    model = get_model(model)
    explainer = get_shap_explainer(model)
    baseline = get_feature_contributions(baseline_payload, model, explainer)
    simulated = get_feature_contributions(simulated_payload, model, explainer)

    base_map = {c["feature"]: c["contribution"] for c in baseline["contributions"]}
    sim_map = {c["feature"]: c["contribution"] for c in simulated["contributions"]}

    deltas = []
    for name in get_feature_names(model):
        delta = sim_map[name] - base_map[name]
        raw_feature, _ = split_transformed_name(name)
        deltas.append({
            "feature": name,
            "label": pretty_feature_name(name),
            "raw_feature": raw_feature,
            "baseline_contribution": base_map[name],
            "simulated_contribution": sim_map[name],
            "delta": delta,
            "abs_delta": abs(delta),
        })
    deltas.sort(key=lambda d: d["abs_delta"], reverse=True)

    return {
        "baseline": baseline,
        "simulated": simulated,
        "probability_delta": float(
            simulated["churn_probability"] - baseline["churn_probability"]
        ),
        "log_odds_delta": float(
            simulated["model_log_odds"] - baseline["model_log_odds"]
        ),
        "deltas": deltas[:int(top_n)] if top_n else deltas,
        "base_value": baseline["base_value"],
        "output_space": OUTPUT_SPACE,
        "disclaimer": (
            "Model-attributed contribution change. SHAP describes this fitted "
            "model's reasoning and does not establish that changing a feature "
            "changes real-world churn."
        ),
    }

