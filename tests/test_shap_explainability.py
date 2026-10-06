"""Phase 2 validation for the SHAP explainability layer.

Design rule for this file: every assertion below was derived from the SHAP
output semantics **empirically verified for this fitted model**, not assumed.

The verified facts are:
  * the champion model is a ``LogisticRegression`` over 30 transformed features
  * ``shap.LinearExplainer`` is the analytic, deterministic explainer for it
  * additivity holds in **log-odds** space:
        expected_value + sum(shap_values) == model.decision_function(x)
  * the ``link=`` argument is inert for linear explainers in this SHAP
    release, so probability-space additivity must NOT be asserted

SHAP is purely explanatory, so several tests also assert that adding SHAP does
not move a single prediction.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap  # noqa: E402

import backend.explainability as bx  # noqa: E402
from backend.model import ChurnModel, load_data  # noqa: E402

# Additivity holds in the model's native output space. Verified empirically
# (residual ~1e-15); asserted here so a future SHAP/sklearn change is caught.
ADDITIVITY_ATOL = 1e-8


@pytest.fixture(scope="module")
def model():
    return ChurnModel.load()


@pytest.fixture(scope="module")
def explainer(model):
    return bx.get_shap_explainer(model)


@pytest.fixture(scope="module")
def customers():
    return load_data().drop(columns=["Churn"])


@pytest.fixture(scope="module")
def sample_profile(customers):
    return customers.iloc[0].to_dict()


# 1. SHAP module imports successfully.
def test_shap_module_imports():
    import importlib
    mod = importlib.import_module("backend.explainability")
    assert hasattr(mod, "get_shap_explainer")
    assert hasattr(mod, "explain_single")
    assert hasattr(mod, "get_global_shap_values")
    assert hasattr(mod, "get_feature_contributions")
    assert isinstance(shap.LinearExplainer, type)


# 2. Explainer initializes successfully and is the right type.
def test_explainer_initialises(explainer, model):
    assert isinstance(explainer, shap.LinearExplainer)
    assert np.isfinite(bx._base_value(explainer))
    # The explainer must wrap the persisted champion model, not a retrained one.
    assert explainer.model is model.best_model
    assert model.best_model.__class__.__name__ == "LogisticRegression"


# 3. Exactly 30 transformed features are explained.
def test_exactly_30_features_explained(model, sample_profile):
    explanation = bx.explain_single(sample_profile, model)
    assert explanation["n_features"] == 30
    assert len(explanation["contributions"]) == 30
    assert len(explanation["contributions"][0]["feature"]) > 0


# 4. Feature names match the model feature count, and come from the artifacts.
def test_feature_names_match_model(model):
    names = bx.get_feature_names(model)
    assert len(names) == 30
    assert len(names) == int(model.best_model.n_features_in_)
    # Names are taken from the fitted preprocessor, not hand-written.
    assert list(model.preprocessor.get_feature_names_out()) == names
    # And they agree with the persisted training-time artifact.
    assert [str(n) for n in np.asarray(model.feature_names).ravel()] == names
    assert not any(n.strip() == "" for n in names)


# 5. Individual explanation is deterministic.
def test_individual_explanation_deterministic(model, sample_profile):
    first = bx.explain_single(sample_profile, model)
    second = bx.explain_single(sample_profile, model)
    assert first["base_value"] == second["base_value"]
    for a, b in zip(first["contributions"], second["contributions"]):
        assert a["feature"] == b["feature"]
        assert a["contribution"] == b["contribution"]


# 6. SHAP does not modify the prediction output.
def test_explanation_does_not_change_prediction(model, customers):
    rows = [0, 7, 55, 500, 4000]
    before = [model.predict_single(customers.iloc[i].to_dict()) for i in rows]
    for i in rows:
        bx.explain_single(customers.iloc[i].to_dict(), model)
    after = [model.predict_single(customers.iloc[i].to_dict()) for i in rows]
    assert before == after


# 7. SHAP contributions are numeric and finite.
def test_contributions_numeric_and_finite(model, sample_profile):
    explanation = bx.explain_single(sample_profile, model)
    for c in explanation["contributions"]:
        value = c["contribution"]
        assert isinstance(value, float)
        assert math.isfinite(value)
    assert math.isfinite(explanation["base_value"])
    assert math.isfinite(explanation["churn_probability_raw"])


# 8. Global SHAP ranking is deterministic and reproducible.
def test_global_shap_ranking_deterministic(model, explainer):
    first = bx.get_global_shap_values(model, explainer)
    second = bx.get_global_shap_values(model, explainer)
    pd.testing.assert_frame_equal(first["importance"], second["importance"])
    # A freshly built explainer must reproduce the same ranking from scratch.
    fresh = bx.get_shap_explainer(model, bx.load_reference_background(model))
    third = bx.get_global_shap_values(model, fresh)
    assert list(first["importance"]["Feature"]) == list(third["importance"]["Feature"])


# 9. Global SHAP output contains no NaN / inf.
def test_global_shap_output_finite(model, explainer):
    result = bx.get_global_shap_values(model, explainer)
    values = result["shap_values"]
    assert values.shape[0] == result["sample_size"]
    assert values.shape[1] == 30
    assert np.all(np.isfinite(values))
    assert not result["importance"].isna().any().any()
    assert np.all(np.isfinite(result["importance"]["Mean |SHAP|"]))


# 10. Top-N contribution ordering is correct (sorted by |contribution| desc).
def test_top_n_ordering(model, sample_profile):
    full = bx.explain_single(sample_profile, model)
    abs_values = [abs(c["contribution"]) for c in full["contributions"]]
    assert abs_values == sorted(abs_values, reverse=True)

    top3 = bx.get_feature_contributions(sample_profile, model, top_n=3)
    assert len(top3["contributions"]) == 3
    assert [c["feature"] for c in top3["contributions"]] == \
        [c["feature"] for c in full["contributions"][:3]]


# 11. Both positive and negative contributions are handled.
def test_positive_and_negative_contributions(model, customers):
    found_positive = False
    found_negative = False
    for i in range(0, 40):
        explanation = bx.explain_single(customers.iloc[i].to_dict(), model)
        for c in explanation["contributions"]:
            if c["contribution"] > 0:
                found_positive = True
                assert c["direction"] == "increases churn"
            elif c["contribution"] < 0:
                found_negative = True
                assert c["direction"] == "decreases churn"
        assert explanation["top_increasing"] or explanation["top_decreasing"]
    assert found_positive and found_negative


# 12. Simulator explanation can be generated.
def test_simulator_explanation(model, sample_profile):
    simulated = dict(sample_profile)
    simulated["Contract"] = "Two year"
    simulated["PaymentMethod"] = "Credit card (automatic)"

    result = bx.explain_simulation(sample_profile, simulated, model)
    assert {"baseline", "simulated", "probability_delta", "log_odds_delta",
            "deltas", "base_value"}.issubset(result)
    # Both sides share one base value, so the delta is attributable only to
    # the modified features.
    assert result["baseline"]["base_value"] == result["simulated"]["base_value"]
    assert result["deltas"] == sorted(result["deltas"], key=lambda d: d["abs_delta"],
                                      reverse=True)
    # The intervention here is expected to lower the score, but the test only
    # asserts the bookkeeping, not the business conclusion.
    assert math.isfinite(result["log_odds_delta"])
    assert "does not establish" in result["disclaimer"]


# 13. (extra) Global direction labels must agree with the model coefficients.
def test_global_directions_match_model_coefficients(model, explainer):
    result = bx.get_global_shap_values(model, explainer)
    canonical = bx.get_feature_names(model)
    coefficients = model.best_model.coef_[0]
    for _, row in result["importance"].iterrows():
        idx = canonical.index(row["Feature"])
        expected = "increases churn" if coefficients[idx] > 0 else "decreases churn"
        assert row["Direction"] == expected, row["Feature"]


# --- The most important test: SHAP must agree with the model itself. ---
def test_shap_additivity_reconstructs_model_log_odds(model, explainer, customers):
    """base_value + sum(phi) == decision_function, verified for the exact
    output space SHAP actually used for this model (log-odds).

    This guards against the common mistake of asserting probability-space
    additivity, which does NOT hold here: ``link=`` is inert for linear
    explainers, and the sum of individual probability shifts is not equal to
    the change in probability.
    """
    matrix = np.asarray(model._transform(customers.head(200)), dtype=float)
    values = bx._shap_values(explainer, matrix)
    base = bx._base_value(explainer)
    log_odds = model.best_model.decision_function(matrix)

    assert np.max(np.abs(base + values.sum(axis=1) - log_odds)) < ADDITIVITY_ATOL

    # Sanity-check the opposite claim, so this test documents *why* the
    # log-odds identity is the correct one to assert.
    probabilities = model.best_model.predict_proba(matrix)[:, 1]
    probability_space_error = np.max(np.abs(base + values.sum(axis=1) - probabilities))
    assert probability_space_error > 1e-3


def test_single_explanation_reconstructs_model_score(model, sample_profile):
    """The public per-customer payload also reconstructs the model score."""
    explanation = bx.explain_single(sample_profile, model)
    total = explanation["base_value"] + sum(
        c["contribution"] for c in explanation["contributions"]
    )
    assert abs(total - explanation["model_log_odds"]) < ADDITIVITY_ATOL
    assert abs(explanation["reconstruction_error"]) < ADDITIVITY_ATOL
    # The reported probability must be the model's own, not a SHAP estimate.
    assert abs(
        explanation["churn_probability"]
        - model.predict_single(sample_profile)["churn_probability"]
    ) < 1e-9


def test_explainer_sees_the_same_matrix_as_the_model(model, explainer,
                                                     sample_profile):
    """SHAP must consume exactly the 30-feature prediction representation.

    Guards against a second preprocessing pipeline being introduced: the
    explainer is exercised on ``model._transform`` output and must accept all
    30 columns without error, and the explanation's probability must equal the
    scorer's probability.
    """
    matrix = np.asarray(model._transform(sample_profile), dtype=float)
    assert matrix.shape == (1, 30)
    assert bx._shap_values(explainer, matrix).shape == (1, 30)
    assert abs(
        float(model.best_model.predict_proba(matrix)[0, 1])
        - bx.explain_single(sample_profile, model)["churn_probability_raw"]
    ) < 1e-12


def test_feature_labels_are_readable_and_keep_technical_name():
    assert bx.pretty_feature_name("cat__Contract_Two year") == "Contract = Two year"
    assert bx.pretty_feature_name("num__MonthlyCharges") == "MonthlyCharges"
    raw, suffix = bx.split_transformed_name("cat__OnlineSecurity_No internet service")
    assert raw == "OnlineSecurity"
    assert suffix == "No internet service"


# =====================================================================
# Phase 4 Step 1 -- ModelView / explainer-selection architecture
# =====================================================================
#
# The architecture under test:
#
#   * ``ModelView`` normalises two structurally different fitted bundles
#     (production ``ChurnModel`` and the Phase 3B champion ``Pipeline``) to one
#     interface and delegates transformation to each bundle's OWN fitted
#     preprocessor.
#   * The explainer is chosen from the FITTED ESTIMATOR'S TYPE, never by
#     string-matching the model name.
#
# The tests ABOVE this line remain PRODUCTION-SPECIFIC regression tests, with
# their original tolerances. The tests below are split explicitly:
#   * genuinely model-agnostic contracts are written once and parameterised
#     over BOTH bundles;
#   * behaviour only true of one model class is asserted only for that bundle.
#
# Neither bundle is retrained, promoted or modified; both are loaded read-only.

CHAMPION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "experiments", "champion.pkl",
)

# Every artifact that must stay byte-identical across the whole run.
PRODUCTION_ARTIFACTS = [
    "best_model.pkl",
    "preprocessor.pkl",
    "results.pkl",
    "feature_names.pkl",
    "total_charges_median.pkl",
    os.path.join("experiments", "champion.pkl"),
]

# Expected SHA256 of each frozen artifact, captured before any XAI change.
# Asserted explicitly so a future re-serialization cannot pass unnoticed.
EXPECTED_ARTIFACT_HASHES = {
    "best_model.pkl":
        "66400195eb9cfef8afaaf9ee59ee83e19baa1e9afa70a4ddc9fc49ce80d9010e",
    "preprocessor.pkl":
        "fcc2d4e47fa218fce6056f7904e4b54f7f9c58fdd6ccbd04ff20b7a5210e028f",
    "results.pkl":
        "5b42735ce812f4ebdffeb2bb4b2abe6afacecb3fdefd3d836a0d425dc473753c",
    "feature_names.pkl":
        "b0e5ae0e650cb9421334ee5f88163fd7769e8dce9c0d27f9161fc4e10aaad9a4",
    "total_charges_median.pkl":
        "ae6e77cba1892e0da00efae88517d3eb4299d925bb6a4f0aad1283ead8cb7640",
    os.path.join("experiments", "champion.pkl"):
        "372ddf6c9ec977460d078a244a97b8781295394bc8f54acb2dc8031f3b736589",
}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_hashes():
    import backend.model as backend_model
    return {
        rel: _sha256(os.path.join(backend_model.MODEL_DIR, rel))
        for rel in PRODUCTION_ARTIFACTS
    }


@pytest.fixture(scope="module")
def artifact_hashes_before():
    """Hash the frozen artifacts before any XAI code in this run executes."""
    return _artifact_hashes()


@pytest.fixture(scope="module")
def champion():
    """The frozen Phase 3B champion Pipeline, loaded read-only. Never retrained."""
    import joblib
    from sklearn.pipeline import Pipeline

    bundle = joblib.load(CHAMPION_PATH)
    assert isinstance(bundle, Pipeline)
    return bundle


@pytest.fixture(scope="module")
def champion_view(champion):
    return bx._as_model_view(champion)


@pytest.fixture(scope="module")
def production_view(model):
    return bx._as_model_view(model)


# --- 1. ModelView normalisation -----------------------------------------

def test_model_view_wraps_production_churnmodel(production_view, model):
    """A ChurnModel normalises to a view bound to its OWN fitted objects."""
    assert isinstance(production_view, bx.ModelView)
    # Delegation, not copying: the view wraps the very same fitted instances.
    assert production_view.estimator is model.best_model
    assert production_view.preprocessor is model.preprocessor
    assert production_view.n_features == 30
    # Production keeps the shared, already-verified inference path.
    assert production_view.raw_columns is None


def test_model_view_wraps_champion_pipeline(champion_view, champion):
    """A Pipeline normalises to a view over its own preprocessor + classifier."""
    assert isinstance(champion_view, bx.ModelView)
    assert champion_view.preprocessor is champion.named_steps["preprocessor"]
    assert champion_view.estimator is champion.named_steps["classifier"]
    # The champion's transformed width is its OWN; not the production 30.
    assert champion_view.n_features == 45
    assert champion_view.n_features != 30
    # Raw column contract is read from the fitted transformer spec.
    assert champion_view.raw_columns is not None
    assert len(champion_view.raw_columns) == 19


def test_model_view_is_idempotent(production_view):
    """Re-normalising a view returns it unchanged (no nesting)."""
    assert bx._as_model_view(production_view) is production_view


def test_model_view_transform_delegates_to_own_preprocessor(
        production_view, model, champion_view, champion, customers):
    """Transformation must come from each bundle's fitted preprocessor.

    This is the core of the abstraction: a view never applies a foreign
    preprocessor and never assumes a fixed 30-column representation.
    """
    from backend.model import prepare_frame

    head = customers.head(5)

    # Production: identical, element-for-element, to the shared inference path.
    via_view = production_view.transform(head)
    via_shared = np.asarray(
        model.preprocessor.transform(
            prepare_frame(head, model.preprocessor, model.total_charges_median)
        ),
        dtype=float,
    )
    assert via_view.shape == (5, 30)
    assert np.array_equal(via_view, via_shared)

    # Champion: its own preprocessor, its own width, no production assumption.
    champ_matrix = champion_view.transform(head)
    assert champ_matrix.shape == (5, 45)
    assert champ_matrix.shape[1] == champion_view.n_features
    ordered = [c for c in champion_view.raw_columns]
    direct = np.asarray(
        champion.named_steps["preprocessor"].transform(head[ordered]),
        dtype=float,
    )
    assert np.array_equal(champ_matrix, direct)


def test_champion_view_does_not_use_production_prepare_frame(champion_view,
                                                             champion,
                                                             customers):
    """The champion bundle must not be routed through the production path.

    ``prepare_frame`` reads its category domain from the production encoder
    layout (a bare OneHotEncoder). The champion's categorical step is a
    Pipeline, so delegating to the shared path would either fail or silently
    apply production-specific assumptions. This test pins the requirement.
    """
    from backend.model import prepare_frame

    head = customers.head(3)
    # The XAI layer's own transform works for the champion ...
    assert champion_view.transform(head).shape == (3, 45)
    # ... while the production inference path genuinely cannot handle it,
    # which is exactly why ModelView must not use it for a Pipeline bundle.
    with pytest.raises(AttributeError):
        prepare_frame(head, champion.named_steps["preprocessor"],
                      champion_view.median)


def test_model_view_rejects_invalid_champion_input(champion_view, customers):
    """Input hygiene is preserved for the champion bundle, not bypassed."""
    profile = {c: customers.iloc[0][c] for c in champion_view.raw_columns}
    assert champion_view.transform(profile).shape == (1, 45)

    with pytest.raises(ValueError, match="Missing required field"):
        broken = dict(profile)
        del broken["gender"]
        champion_view.transform(broken)

    with pytest.raises(ValueError, match="Invalid categorical input"):
        broken = dict(profile)
        broken["Contract"] = "Not A Real Contract"
        champion_view.transform(broken)


def test_unsupported_model_raises_instead_of_silently_degrading():
    """An unknown bundle must fail loudly, never fall back to KernelExplainer."""
    class NotAModel:
        pass

    with pytest.raises(bx.XAIUnsupportedModelError):
        bx._as_model_view(NotAModel())

    class UnsupportedEstimator:
        """Neither a tree ensemble nor a fitted linear model."""
        pass

    with pytest.raises(bx.XAIUnsupportedModelError):
        bx._select_explainer_kind(UnsupportedEstimator())


# --- 2. Explainer selection (type-based, never by name) -----------------

def test_explainer_kind_is_selected_from_type_not_name(production_view,
                                                        champion_view):
    """Selection keys off the fitted estimator's type, not its display name."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    assert isinstance(production_view.estimator, LogisticRegression)
    assert isinstance(champion_view.estimator, GradientBoostingClassifier)
    assert bx._select_explainer_kind(production_view.estimator) == "linear"
    assert bx._select_explainer_kind(champion_view.estimator) == "tree"

    # Prove it is NOT name-based: a linear estimator carrying a deliberately
    # misleading "GradientBoosting" label must still resolve to "linear".
    disguised = bx.ModelView(
        estimator=production_view.estimator,
        preprocessor=production_view.preprocessor,
        median=production_view.median,
        label="GradientBoostingClassifier",
    )
    assert disguised.label == "GradientBoostingClassifier"
    assert bx._select_explainer_kind(disguised.estimator) == "linear"


def test_built_explainer_class_follows_selected_kind(production_view,
                                                     champion_view):
    """The explainer class built by the factory matches the selected kind."""
    background = bx.load_reference_background(production_view)
    linear = bx._build_explainer(production_view, "linear", background)
    tree = bx._build_explainer(champion_view, "tree", None)
    assert isinstance(linear, shap.LinearExplainer)
    assert isinstance(tree, shap.TreeExplainer)
    # An unknown kind is a hard error, never a silent default.
    with pytest.raises(bx.XAIUnsupportedModelError):
        bx._build_explainer(production_view, "kernel", None)


def test_get_shap_explainer_returns_correct_class_per_bundle(explainer,
                                                             champion):
    """The public entry point builds the right explainer for each bundle."""
    assert isinstance(explainer, shap.LinearExplainer)
    assert isinstance(bx.get_shap_explainer(champion), shap.TreeExplainer)


def test_explainer_cache_does_not_cross_contaminate_bundles(explainer,
                                                           champion):
    """Two bundles must never share a cache entry within one project."""
    champ_explainer = bx.get_shap_explainer(champion)
    assert champ_explainer is not explainer
    assert bx.get_shap_explainer(champion) is champ_explainer


def test_linear_explainer_consumes_whole_background(production_view):
    """The masker is pinned to the full background.

    SHAP's default masker cap of 100 would silently sub-sample a 200-row
    background, making the base value depend on an arbitrary subset. The
    max_samples=len(background) setting prevents that.
    """
    background = bx.load_reference_background(production_view)
    explainer = bx._build_explainer(production_view, "linear", background)
    assert explainer.masker.data.shape[0] == background.shape[0]


# --- 3. Feature names per bundle (never assumed) ------------------------

def test_feature_names_come_from_each_bundles_own_preprocessor(
        production_view, champion_view):
    """Each bundle reports ITS OWN fitted names; the counts differ by design."""
    prod_names = bx.get_feature_names(production_view)
    champ_names = bx.get_feature_names(champion_view)

    assert len(prod_names) == 30
    assert len(champ_names) == 45

    # Names come from each fitted preprocessor, never from a shared constant.
    assert prod_names == [str(n) for n in
                          production_view.preprocessor.get_feature_names_out()]
    assert champ_names == [str(n) for n in
                           champion_view.preprocessor.get_feature_names_out()]
    assert not any(n.strip() == "" for n in champ_names)


def test_champion_and_production_encodings_differ_as_designed(
        production_view, champion_view):
    """The 45-vs-30 gap is a real encoding difference, not a defect.

    Production one-hot uses drop="first" so each categorical keeps one dummy;
    the champion keeps the full one-hot block. Same raw columns, different
    transformed width -- exactly the expected difference between the bundles.
    """
    prod_names = bx.get_feature_names(production_view)
    champ_names = bx.get_feature_names(champion_view)
    # The champion retains a baseline dummy for a column production drops.
    assert any(n.startswith("cat__gender_Female") for n in champ_names)
    assert not any(n.startswith("cat__gender_Female") for n in prod_names)
    assert len(champ_names) > len(prod_names)


def test_champion_names_resolve_without_a_feature_names_sidecar(champion):
    """A Pipeline carries no feature_names.pkl; names still resolve fully."""
    names = bx.get_feature_names(champion)
    assert len(names) == 45
    assert names == [str(n) for n in
                     champion.named_steps["preprocessor"].get_feature_names_out()]


# --- 4. Model-agnostic XAI contract, asserted against BOTH bundles -------
#
# Each of these is a property of the XAI layer, not of one model class, so it
# is written once and parameterised over both bundles. The expected feature
# count is always read from the active view, never hard-coded to 30.

@pytest.fixture(params=["production", "champion"])
def bundle(request):
    """Resolve a named bundle into (view, estimator, n_features, kind)."""
    if request.param == "production":
        view = request.getfixturevalue("production_view")
        return view, view.estimator, view.n_features, "linear"
    view = request.getfixturevalue("champion_view")
    return view, view.estimator, view.n_features, "tree"


def test_contract_shap_values_have_expected_shape(bundle):
    view, _, n_features, _ = bundle
    background = bx.load_reference_background(view)
    explainer = bx.get_shap_explainer(view, background=background)
    values = bx._shap_values(explainer, background[:25])
    assert values.shape == (25, n_features)
    assert np.all(np.isfinite(values))


def test_contract_additivity_reconstructs_decision_function(bundle, customers):
    """base_value + sum(phi) == decision_function, for BOTH explainers.

    Both explainers are pinned to model_output="raw" (the model's raw margin),
    so this is the single additivity identity valid for either bundle. Tree
    raw output is deliberately NOT compared against probability; the next test
    documents why that would be the wrong invariant.
    """
    view, estimator, _, _ = bundle
    background = bx.load_reference_background(view)
    explainer = bx.get_shap_explainer(view, background=background)
    matrix = view.transform(customers.head(200))
    values = bx._shap_values(explainer, matrix)
    base = bx._base_value(explainer)
    margin = estimator.decision_function(matrix)
    residual = float(np.max(np.abs(base + values.sum(axis=1) - margin)))
    assert residual < ADDITIVITY_ATOL, f"additivity residual was {residual}"


def test_contract_explanations_are_deterministic(bundle, customers):
    """Repeated explanation of the same input is bit-identical."""
    view, _, _, kind = bundle
    background = bx.load_reference_background(view)
    explainer = bx.get_shap_explainer(view, background=background)
    matrix = view.transform(customers.head(10))
    first = bx._shap_values(explainer, matrix)
    assert np.array_equal(first, bx._shap_values(explainer, matrix))
    # A freshly built explainer (cache bypassed) must agree as well.
    fresh = bx._build_explainer(view, kind, background)
    assert np.array_equal(first, bx._shap_values(fresh, matrix))


def test_contract_individual_explanation_reconstructs_score(bundle, customers):
    """The public payload reconstructs the estimator's own margin."""
    view, estimator, n_features, _ = bundle
    profile = customers.iloc[0].to_dict()
    result = bx.explain_single(profile, view)
    total = result["base_value"] + sum(
        c["contribution"] for c in result["contributions"]
    )
    assert abs(total - result["model_log_odds"]) < ADDITIVITY_ATOL
    assert abs(result["reconstruction_error"]) < ADDITIVITY_ATOL
    # The reported probability is the estimator's own, not a SHAP estimate.
    matrix = view.transform(profile)
    assert abs(result["churn_probability_raw"]
               - float(estimator.predict_proba(matrix)[0, 1])) < 1e-12
    assert result["n_features"] == n_features
    assert len(result["contributions"]) == n_features
    assert result["output_space"] == bx.OUTPUT_SPACE


def test_contract_explainer_label_matches_estimator_type(bundle, customers):
    view, _, _, kind = bundle
    result = bx.explain_single(customers.iloc[0].to_dict(), view)
    expected = {"linear": "shap.LinearExplainer",
                "tree": "shap.TreeExplainer"}[kind]
    assert result["explainer"] == expected == bx._explainer_name(kind)


def test_contract_global_ranking_is_wellformed(bundle):
    view, _, n_features, kind = bundle
    result = bx.get_global_shap_values(view)
    importance = result["importance"]
    assert len(importance) == n_features
    assert list(importance.columns) == ["Feature", "Label", "Mean |SHAP|",
                                        "Direction"]
    assert np.all(np.isfinite(importance["Mean |SHAP|"]))
    assert importance["Mean |SHAP|"].is_monotonic_decreasing
    assert set(importance["Direction"]) <= {"increases churn", "decreases churn"}
    assert result["feature_names"] == bx.get_feature_names(view)
    assert result["explainer"] == bx._explainer_name(kind)
    assert result["shap_values"].shape == (result["sample_size"], n_features)


def test_contract_global_ranking_is_deterministic(bundle):
    view, _, _, kind = bundle
    first = bx.get_global_shap_values(view)
    again = bx.get_global_shap_values(view)
    assert np.array_equal(first["shap_values"], again["shap_values"])
    # Recomputed from a freshly built explainer, bypassing the cache.
    rebuilt = bx.get_global_shap_values(
        view,
        explainer=bx._build_explainer(view, kind,
                                      bx.load_reference_background(view)),
    )
    assert np.allclose(first["shap_values"], rebuilt["shap_values"])


def test_contract_explanations_do_not_mutate_the_model(bundle, customers):
    """SHAP is purely explanatory: predictions identical before/after."""
    view, estimator, _, _ = bundle
    matrix = view.transform(customers.head(20))
    before = estimator.predict_proba(matrix)
    explainer = bx.get_shap_explainer(
        view, background=bx.load_reference_background(view))
    bx._shap_values(explainer, matrix)
    bx.explain_single(customers.iloc[0].to_dict(), view)
    assert np.array_equal(before, estimator.predict_proba(matrix))


def test_contract_contributions_are_sorted_and_labelled(bundle, customers):
    view, _, n_features, _ = bundle
    contributions = bx.explain_single(customers.iloc[0].to_dict(),
                                      view)["contributions"]
    mags = [c["abs_contribution"] for c in contributions]
    assert mags == sorted(mags, reverse=True)
    for c in contributions:
        assert c["label"] == bx.pretty_feature_name(c["feature"])
        assert c["direction"] in {"increases churn", "decreases churn"}
    assert len(contributions) == n_features


# --- 5. Champion-specific guarantees (not claimed for linear) ----------

def test_champion_tree_output_is_margin_not_probability(champion_view,
                                                         customers):
    """Tree raw output is the margin, NOT the probability.

    Documents *why* the margin identity is the correct one to assert, and
    prevents a future change from silently claiming probability additivity.
    """
    background = bx.load_reference_background(champion_view)
    explainer = bx.get_shap_explainer(champion_view, background=background)
    matrix = champion_view.transform(customers.head(200))
    values = bx._shap_values(explainer, matrix)
    base = bx._base_value(explainer)
    estimator = champion_view.estimator

    margin_error = np.max(np.abs(base + values.sum(axis=1)
                                 - estimator.decision_function(matrix)))
    assert margin_error < ADDITIVITY_ATOL

    probabilities = estimator.predict_proba(matrix)[:, 1]
    probability_space_error = np.max(np.abs(base + values.sum(axis=1)
                                           - probabilities))
    assert probability_space_error > 1e-3


def test_champion_tree_explainer_needs_no_background(champion_view, customers):
    """tree_path_dependent consumes no background, so the base value is fixed.

    Guards the determinism guarantee: a different background size must not
    shift the base value, because none is consumed.
    """
    full = bx.load_reference_background(champion_view, sample_size=200)
    subset = full[:50]
    matrix = champion_view.transform(customers.head(20))
    full_explainer = bx.get_shap_explainer(champion_view, background=full)
    subset_explainer = bx.get_shap_explainer(champion_view, background=subset)
    assert bx._base_value(full_explainer) == bx._base_value(subset_explainer)
    assert np.allclose(bx._shap_values(full_explainer, matrix),
                       bx._shap_values(subset_explainer, matrix))


def test_champion_is_never_forced_into_production_representation(
        champion_view, customers):
    """The champion keeps its own 45-feature space end to end."""
    assert champion_view.n_features == 45
    assert champion_view.n_features != 30
    assert len(bx.get_feature_names(champion_view)) == 45
    assert champion_view.transform(customers.head(3)).shape[1] == 45
    # A 30-column matrix is simply not this estimator's input.
    with pytest.raises(Exception):
        champion_view.estimator.predict_proba(np.zeros((1, 30)))


# --- 6. Artifact immutability ------------------------------------------

def test_explaining_never_writes_to_any_artifact(artifact_hashes_before,
                                                 bundle, customers):
    """Explaining must not write to any fitted artifact."""
    view, _, _, _ = bundle
    bx.explain_single(customers.iloc[0].to_dict(), view)
    bx.get_global_shap_values(view)
    assert _artifact_hashes() == artifact_hashes_before


def test_all_artifacts_still_byte_identical(artifact_hashes_before):
    """Explicit expected SHA256 for the six frozen artifacts.

    Guards against a future change that rewrites or re-serializes a model
    bundle, champion.pkl included.
    """
    current = _artifact_hashes()
    assert current == EXPECTED_ARTIFACT_HASHES
    assert current == artifact_hashes_before


def test_no_model_promotion_occurred(model, champion):
    """Production stays on LogisticRegression; the champion stays frozen."""
    assert model.best_model.__class__.__name__ == "LogisticRegression"
    assert champion.named_steps["classifier"].__class__.__name__ == \
        "GradientBoostingClassifier"
    # The champion has NOT been copied over the production bundle.
    assert model.best_model is not champion.named_steps["classifier"]
    assert model.preprocessor is not champion.named_steps["preprocessor"]


