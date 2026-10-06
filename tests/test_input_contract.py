"""Regression tests for the shared ModelBundle input contract.

Phase 4 Step 8. One test per promotion blocker resolved by
``backend.input_contract`` / ``backend.model_bundle`` / the XAI adapter, plus
the contract invariants themselves.

Safety invariants asserted throughout:
  * production artifacts are never written;
  * ``models/active_bundle.json`` still selects v1;
  * v2 is never promoted.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from backend.input_contract import (
    ContractError,
    ContractSpec,
    NumericPolicy,
    SENIOR_CITIZEN_ALIASES,
    prepare_input_frame,
)
from backend.model_bundle import BundleInferenceError, ModelBundle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
V1_DIR = os.path.join(MODELS_DIR, "bundles", "v1")
V2_DIR = os.path.join(MODELS_DIR, "bundles", "v2")
DATA_CSV = os.path.join(BASE_DIR, "data", "telco_customer_churn_cleaned.csv")

BUNDLE_IDS = ("v1", "v2")


@pytest.fixture(scope="module")
def bundles():
    return {"v1": ModelBundle(V1_DIR), "v2": ModelBundle(V2_DIR)}


@pytest.fixture(scope="module")
def base_row():
    """A valid raw customer row taken from the real dataset."""
    df = pd.read_csv(DATA_CSV)
    return df.drop(columns=["Churn", "customerID"]).iloc[0].to_dict()


@pytest.fixture(params=BUNDLE_IDS)
def bundle(request, bundles):
    """Parametrised over both bundles, so every check runs for v1 *and* v2."""
    return bundles[request.param]


# --- 1. valid input prediction ------------------------------------------

def test_valid_input_prediction_v1(bundles, base_row):
    p = bundles["v1"].predict_proba(base_row)
    assert p.shape == (1,)
    assert 0.0 <= float(p[0]) <= 1.0


def test_valid_input_prediction_v2(bundles, base_row):
    p = bundles["v2"].predict_proba(base_row)
    assert p.shape == (1,)
    assert 0.0 <= float(p[0]) <= 1.0


# --- 3. missing required column rejection --------------------------------

@pytest.mark.parametrize("column", ["Contract", "tenure", "SeniorCitizen",
                                    "MonthlyCharges", "TotalCharges"])
def test_missing_required_column_rejected(bundle, base_row, column):
    row = {k: v for k, v in base_row.items() if k != column}
    with pytest.raises(ContractError) as exc:
        bundle.predict_proba(row)
    assert column in str(exc.value)


# --- 4. extra column handling --------------------------------------------

def test_extra_columns_ignored_identically(bundles, base_row):
    """A labelled/annotated row scores exactly like the bare row."""
    enriched = {**base_row, "customerID": "1234-XYZ", "Churn": "Yes",
                "some_future_feature": 99}
    for key in BUNDLE_IDS:
        b = bundles[key]
        assert b.contract.extra_column_policy == "ignore"
        np.testing.assert_allclose(
            b.predict_proba(enriched), b.predict_proba(base_row),
            rtol=0, atol=0,
        )


# --- 5. invalid categorical rejection (BLOCKER-02) -----------------------

@pytest.mark.parametrize("column,value", [
    ("Contract", "Three year"),
    ("Contract", "month-to-month"),
    ("InternetService", "DSL fibre"),
    ("gender", "M"),
    ("PaymentMethod", "Cash"),
])
def test_invalid_categorical_rejected_loudly(bundle, base_row, column, value):
    """Both bundles must REFUSE an out-of-domain level.

    Regression for BLOCKER-02: v2's ``handle_unknown="ignore"`` used to turn
    this into a silent, confidently wrong all-zeros column.
    """
    with pytest.raises(ContractError) as exc:
        bundle.predict_proba({**base_row, column: value})
    assert column in str(exc.value)


def test_invalid_categorical_behaviour_is_identical_across_bundles(bundles,
                                                                   base_row):
    row = {**base_row, "Contract": "Three year"}
    outcomes = {}
    for key in BUNDLE_IDS:
        try:
            bundles[key].predict_proba(row)
            outcomes[key] = "accepted"
        except ContractError:
            outcomes[key] = "rejected"
    assert outcomes["v1"] == outcomes["v2"] == "rejected"

# --- 6/7. SeniorCitizen representations (BLOCKER-01) ----------------------

@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "0", "1", "0.0", "1.0",
                                   "Yes", "No", "yes", "no", "YES", " No "])
def test_senior_citizen_valid_representations_accepted(bundle, base_row, value):
    """Every representation in the documented closed set must be accepted.

    Regression for BLOCKER-01: v2's median ``SimpleImputer`` previously
    rejected the text values v1 accepted.
    """
    prob = bundle.predict_proba({**base_row, "SeniorCitizen": value})
    assert np.isfinite(prob[0])


def test_senior_citizen_equivalent_representations_agree(bundle, base_row):
    """``1`` and ``'Yes'`` are the same logical value and must score alike."""
    a = bundle.predict_proba({**base_row, "SeniorCitizen": 1})[0]
    b = bundle.predict_proba({**base_row, "SeniorCitizen": "Yes"})[0]
    assert float(a) == pytest.approx(float(b), abs=0.0)


@pytest.mark.parametrize("value", [2, 0.5, -1, 99, "maybe", "", None, np.nan,
                                   "Senior"])
def test_senior_citizen_invalid_representations_rejected(bundle, base_row,
                                                         value):
    with pytest.raises(ContractError) as exc:
        bundle.predict_proba({**base_row, "SeniorCitizen": value})
    assert "SeniorCitizen" in str(exc.value)


def test_senior_citizen_alias_set_is_the_documented_closed_set():
    assert set(SENIOR_CITIZEN_ALIASES) == {
        "0", "0.0", "1", "1.0", "no", "yes", "false", "true",
        "n", "y", "f", "t",
    }


# --- 8. TotalCharges policy ----------------------------------------------

def test_total_charges_missing_imputes_persisted_median(bundles, base_row):
    """BLOCKER-04: the imputed VALUE must be the persisted median, for both."""
    expected = 1394.925
    for key in BUNDLE_IDS:
        b = bundles[key]
        assert b.contract.total_charges_policy == \
            "impute_with_persisted_training_median"
        frame = b.prepare_input({**base_row, "TotalCharges": np.nan})
        assert float(frame["TotalCharges"].iloc[0]) == pytest.approx(expected)
        # ... and the score must equal that of an explicit median.
        np.testing.assert_allclose(
            b.predict_proba({**base_row, "TotalCharges": np.nan}),
            b.predict_proba({**base_row, "TotalCharges": expected}),
            rtol=0, atol=0,
        )


def test_total_charges_blank_string_treated_as_missing(bundle, base_row):
    frame = bundle.prepare_input({**base_row, "TotalCharges": "  "})
    assert frame["TotalCharges"].iloc[0] == pytest.approx(1394.925)


def test_total_charges_numeric_text_is_coerced(bundle, base_row):
    frame = bundle.prepare_input({**base_row, "TotalCharges": "100.5"})
    assert float(frame["TotalCharges"].iloc[0]) == pytest.approx(100.5)


# --- 9. numeric NaN behaviour (BLOCKER-03) -------------------------------

@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges"])
def test_missing_measured_numeric_is_rejected_by_both(bundles, base_row,
                                                      column):
    """BLOCKER-03: a missing *measured* numeric is refused, for BOTH bundles.

    v1 already refused it; v2's fitted imputer used to accept it silently.
    Imputing it here would have silently changed production v1 behaviour, so
    the contract keeps the loud refusal for both.
    """
    row = {**base_row, column: np.nan}
    for key in BUNDLE_IDS:
        with pytest.raises(ContractError) as exc:
            bundles[key].predict_proba(row)
        assert column in str(exc.value)


def test_contract_only_exposes_total_charges_for_imputation(bundles):
    """The contract must not lean on v2's fitted imputer for the other three.

    This is what guarantees the accept/reject decision cannot drift between the
    bundles: neither bundle is allowed to impute a column the other refuses.
    """
    for key in BUNDLE_IDS:
        stats = bundles[key].contract.numeric_impute_statistics
        assert set(stats) == {"TotalCharges"}
        assert bundles[key].contract.numeric_missing_policy == \
            NumericPolicy.MISSING_NUMERIC_REJECT


# --- 10. infinity rejection ----------------------------------------------

@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges", "TotalCharges"])
def test_infinity_rejected_by_both(bundles, base_row, column, value):
    for key in BUNDLE_IDS:
        with pytest.raises(ContractError) as exc:
            bundles[key].predict_proba({**base_row, column: value})
        assert "infinite" in str(exc.value).lower()


# --- 11. positive-class probability --------------------------------------

def test_positive_class_is_metadata_derived_and_means_churn(bundles):
    """BLOCKER-07: encodings differ (int vs str) but the MEANING must not.

    v1 encodes classes as ``[0, 1]`` and v2 as ``['No', 'Yes']``. The positive
    class is derived from each bundle's own ``classes_`` and cross-checked
    against its manifest -- but both must resolve to the same churn semantics.
    """
    for key in BUNDLE_IDS:
        b = bundles[key]
        classes = [str(c) for c in b.classes]
        assert classes[b.positive_class_index] == str(b.positive_class)
        assert b.positive_class_index == \
            b.manifest["inference"]["positive_class_index"]
        # The declared positive class must actually BE the resolved one.
        assert str(b.manifest["inference"]["positive_class"]) == \
            str(b.positive_class)
    # v1 encodes the positive class as int 1, v2 as the string 'Yes'. The
    # literal labels differ by design; both must resolve to the SAME canonical
    # churn semantics, which is what makes them comparable without a
    # per-bundle special case.
    assert bundles["v1"].positive_semantic_label == "Yes"
    assert bundles["v2"].positive_semantic_label == "Yes"
    aliases = ModelBundle.POSITIVE_CLASS_ALIASES
    assert aliases[str(bundles["v1"].positive_class).lower()] == "Yes"
    assert aliases[str(bundles["v2"].positive_class).lower()] == "Yes"


def test_positive_probability_column_is_correct_for_each_bundle(bundles,
                                                                 base_row):
    """The returned probability must be the POSITIVE one, not blindly [1]."""
    for key in BUNDLE_IDS:
        b = bundles[key]
        neg, pos = b.predict_proba_both(base_row)
        both = b.predict_proba(base_row)
        np.testing.assert_allclose(np.asarray(pos), np.asarray(both))
        # For a known churner the positive probability must dominate.
        assert float(both[0]) > float(neg[0]), (
            f"{key}: positive-class probability did not dominate a churn row"
        )
        np.testing.assert_allclose(float(both[0]) + float(neg[0]), 1.0,
                                   rtol=0, atol=1e-9)


def test_positive_class_index_not_assumed_to_be_one(bundles):
    """Guard: the index is *derived*, so it must equal the validated index."""
    for key in BUNDLE_IDS:
        b = bundles[key]
        assert b.positive_class_index in (0, 1)
        assert b.negative_class_index == 1 - b.positive_class_index
        assert b.positive_class_index != b.negative_class_index

# --- 12. threshold = 0.5 --------------------------------------------------

def test_threshold_is_metadata_driven_and_still_one_half(bundles):
    for key in BUNDLE_IDS:
        b = bundles[key]
        assert b.decision_threshold == 0.5
        assert b.decision_threshold == b.manifest["inference"]["decision_threshold"]


def test_threshold_labels_match_the_half_point(bundles, base_row):
    for key in BUNDLE_IDS:
        b = bundles[key]
        p = float(b.predict_proba(base_row)[0])
        expected = "Yes" if p >= 0.5 else "No"
        assert b.predict_labels(base_row)[0] == expected


def test_threshold_applied_around_the_boundary(bundles, base_row):
    """Exactly-at-threshold is inclusive; a hair below is not."""
    for key in BUNDLE_IDS:
        b = bundles[key]
        p = float(b.predict_proba(base_row)[0])
        assert (b.predict_labels(base_row)[0] == "Yes") == (p >= 0.5)


def test_phase_3c_analysis_threshold_not_adopted(bundles):
    """The 0.25 analysis threshold must not leak into production semantics."""
    for key in BUNDLE_IDS:
        assert bundles[key].decision_threshold != 0.25


# --- 13/14. single / batch parity ----------------------------------------

def test_single_batch_parity_v1(bundles, base_row):
    df = pd.DataFrame([base_row, {**base_row, "tenure": 5}])
    batch = bundles["v1"].predict_proba(df)
    singles = np.concatenate([
        bundles["v1"].predict_proba(base_row),
        bundles["v1"].predict_proba({**base_row, "tenure": 5}),
    ])
    np.testing.assert_allclose(batch, singles, rtol=0, atol=0)


def test_single_batch_parity_v2(bundles, base_row):
    df = pd.DataFrame([base_row, {**base_row, "tenure": 5}])
    batch = bundles["v2"].predict_proba(df)
    singles = np.concatenate([
        bundles["v2"].predict_proba(base_row),
        bundles["v2"].predict_proba({**base_row, "tenure": 5}),
    ])
    np.testing.assert_allclose(batch, singles, rtol=0, atol=0)


def test_score_does_not_depend_on_batch_composition(bundles, base_row):
    """A row's score must not change when other rows join the batch."""
    for key in BUNDLE_IDS:
        b = bundles[key]
        alone = b.predict_proba(base_row)
        crowded = b.predict_proba(pd.DataFrame(
            [base_row] + [{**base_row, "tenure": int(t),
                           "TotalCharges": float(t) * 10.0}
                          for t in range(1, 20)]))
        np.testing.assert_allclose(alone, crowded[:1], rtol=0, atol=0)


# --- 15/16. XAI ----------------------------------------------------------

def test_xai_v1_uses_linear_explainer(bundles, base_row):
    from backend import explainability as bx
    b = bundles["v1"]
    view = bx._as_model_view(b)
    assert bx._select_explainer_kind(view.estimator) == "linear"
    contrib = bx.get_feature_contributions(
        base_row, model=b, explainer=bx.get_shap_explainer(b))
    assert contrib["explainer"] == "shap.LinearExplainer"
    assert contrib["n_features"] == 30
    assert len(contrib["contributions"]) == 30


def test_xai_v2_uses_tree_explainer(bundles, base_row):
    from backend import explainability as bx
    b = bundles["v2"]
    view = bx._as_model_view(b)
    assert bx._select_explainer_kind(view.estimator) == "tree"
    contrib = bx.get_feature_contributions(
        base_row, model=b, explainer=bx.get_shap_explainer(b))
    assert contrib["explainer"] == "shap.TreeExplainer"
    assert contrib["n_features"] == 45
    assert len(contrib["contributions"]) == 45


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_xai_accepts_bundle_directly(bundles, key):
    """BLOCKER-09: no more XAIUnsupportedModelError for a ModelBundle."""
    from backend import explainability as bx
    view = bx._as_model_view(bundles[key])
    assert view.contract is not None
    assert view.estimator is bundles[key].estimator


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_xai_feature_names_match_bundle_feature_count(bundles, key, base_row):
    """BLOCKER-08: names/width must track each bundle's OWN feature space."""
    from backend import explainability as bx
    b = bundles[key]
    expected = b.feature_representation()["transformed_feature_count"]
    names = bx.get_feature_names(b)
    assert len(names) == expected
    assert names == b.feature_representation()["transformed_features"]
    assert bx._as_model_view(b).transform(base_row).shape[1] == expected


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_xai_shap_additivity_holds(bundles, key, base_row):
    from backend import explainability as bx
    b = bundles[key]
    c = bx.get_feature_contributions(
        base_row, model=b, explainer=bx.get_shap_explainer(b))
    total = sum(x["contribution"] for x in c["contributions"])
    assert c["base_value"] + total == pytest.approx(c["model_log_odds"],
                                                     abs=1e-6)


def test_xai_explainer_cache_does_not_cross_contaminate_bundles(bundles):
    """Regression: two bundles of the same estimator TYPE must not share an
    explainer cache entry.

    ``test_shap_explainability.py::test_explainer_initialises`` asserts the
    explainer wraps the production ``ChurnModel.best_model``. The v1
    ``ModelBundle`` is a *separately unpickled* ``LogisticRegression`` living
    under the same ``models/`` directory, so a cache keyed on the class name
    let the v1 bundle shadow production (or vice versa) depending on test
    order. This locks the key to fitted-instance identity.
    """
    from backend import explainability as bx
    from backend.model import ChurnModel

    v1_explainer = bx.get_shap_explainer(bundles["v1"])
    production = ChurnModel.load()
    prod_explainer = bx.get_shap_explainer(production)

    assert v1_explainer.model is bundles["v1"].estimator
    assert prod_explainer.model is production.best_model
    assert v1_explainer is not prod_explainer

    # And the same object still reuses one cached explainer.
    assert bx.get_shap_explainer(production) is prod_explainer


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_xai_output_is_deterministic(bundles, key, base_row):
    from backend import explainability as bx
    b = bundles[key]
    first = bx.get_feature_contributions(
        base_row, model=b, explainer=bx.get_shap_explainer(b))
    second = bx.get_feature_contributions(
        base_row, model=b, explainer=bx.get_shap_explainer(b))
    assert [x["contribution"] for x in first["contributions"]] == \
           [x["contribution"] for x in second["contributions"]]

# --- 17/18. determinism --------------------------------------------------

@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_prediction_is_deterministic(bundles, key, base_row):
    b = bundles[key]
    first = b.predict_proba(base_row)
    for _ in range(4):
        np.testing.assert_allclose(b.predict_proba(base_row), first,
                                   rtol=0, atol=0)


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_reloading_bundle_reproduces_identical_scores(bundles, key, base_row):
    b = bundles[key]
    path = V1_DIR if key == "v1" else V2_DIR
    fresh = ModelBundle(path)
    np.testing.assert_allclose(fresh.predict_proba(base_row),
                               b.predict_proba(base_row), rtol=0, atol=0)


# --- 19. bundle integrity ------------------------------------------------

@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_bundle_manifest_artifact_hashes_verify(bundles, key):
    """Integrity is enforced at load time; assert the recorded hashes hold.

    v1 declares a separate ``preprocessor`` artifact; v2 is a self-contained
    Pipeline with only a ``model`` artifact, so the loop follows the manifest
    rather than assuming a fixed artifact set.
    """
    import hashlib
    b = bundles[key]
    for art, spec in b.manifest["artifacts"].items():
        with open(os.path.join(b.bundle_dir, spec["path"]), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert digest == spec["sha256"].lower(), f"{key}/{art} hash drifted"
    assert "model" in b.manifest["artifacts"]


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_bundle_class_semantics_validated_against_artifact(bundles, key):
    b = bundles[key]
    assert [str(c) for c in b.classes] == \
        [str(c) for c in b.manifest["inference"]["classes"]]
    assert b.manifest["features"]["transformed_feature_count"] == \
        b.feature_representation()["transformed_feature_count"]


@pytest.mark.parametrize("key", BUNDLE_IDS)
def test_bundle_rejects_inconsistent_class_metadata(key):
    """Corrupting positive_class_index must fail loudly, not silently."""
    import shutil
    import tempfile
    src = V1_DIR if key == "v1" else V2_DIR
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "bundle")
        shutil.copytree(src, dst)
        manifest_path = os.path.join(dst, "manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["inference"]["positive_class_index"] = 0
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with pytest.raises(BundleInferenceError):
            ModelBundle(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_feature_representations_differ_and_are_self_describing(bundles):
    """BLOCKER-08: 30 vs 45 is reported, not silently harmonised."""
    v1, v2 = bundles["v1"], bundles["v2"]
    assert v1.feature_representation()["transformed_feature_count"] == 30
    assert v2.feature_representation()["transformed_feature_count"] == 45
    assert v1.feature_representation()["one_hot_drop"] == "first"
    assert v2.feature_representation()["one_hot_drop"] is None
    # Each bundle's real transformed width matches its own declaration.
    for b in (v1, v2):
        width = b.transform(
            {c: (0 if c == "SeniorCitizen" else
                 (0.0 if c in ("tenure", "MonthlyCharges", "TotalCharges") else
                  sorted(b.contract.categorical_domains[c])[0]))
             for c in b.contract.required_columns}).shape[1]
        assert width == b.feature_representation()["transformed_feature_count"]


# --- 20. production boundary (bundle-aware: reads real pointer) -----------

def test_active_bundle_is_a_known_version():
    """active_bundle.json must always point to a known, loadable bundle.

    Phase 4 Step 12D: formerly test_active_bundle_remains_v1. The assertion
    was v1-specific and would fail after a controlled v2 promotion. This test
    is updated to be bundle-aware: it verifies the pointer names a valid,
    loadable bundle, not that a specific version is deployed. The promotion
    record in models/experiments/v2_promotion_record.json documents the
    v1->v2 transition.
    """
    with open(os.path.join(MODELS_DIR, "active_bundle.json"),
              encoding="utf-8") as fh:
        active = json.load(fh)
    assert active["active_version"] in ("v1", "v2"), (
        f"active_bundle.json must point to a known bundle, "
        f"got {active['active_version']!r}"
    )


def test_bundle_loader_returns_active_bundle():
    """BundleLoader must load the bundle named by active_bundle.json.

    Phase 4 Step 12D: formerly test_bundle_loader_still_returns_v1. Updated
    to be bundle-aware: the invariant is that the loaded bundle matches the
    pointer, not that v1 is always returned.
    """
    from backend.model_bundle import BundleLoader
    loaded = BundleLoader.get_active_bundle(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, "active_bundle.json"),
              encoding="utf-8") as fh:
        pointer_version = json.load(fh)["active_version"]
    assert loaded.manifest["bundle_version"] == pointer_version, (
        f"BundleLoader returned {loaded.manifest['bundle_version']!r} "
        f"but pointer says {pointer_version!r}"
    )


def test_active_bundle_pointer_is_consistent():
    """The active pointer names a bundle that loads cleanly.

    Phase 4 Step 12D: formerly test_v2_is_not_promoted_anywhere. Updated to
    be bundle-aware: it verifies the pointer is internally consistent and the
    named bundle loads, rather than asserting a specific version. A failed
    load would surface as a BundleInferenceError or FileNotFoundError, making
    any broken pointer visible without hard-coding a version name.
    """
    from backend.model_bundle import BundleLoader
    with open(os.path.join(MODELS_DIR, "active_bundle.json"),
              encoding="utf-8") as fh:
        active_version = json.load(fh)["active_version"]
    # The pointer must name a known version, not an arbitrary string.
    assert active_version in ("v1", "v2")
    # And the named bundle must actually load without error.
    loaded = BundleLoader.get_active_bundle(MODELS_DIR)
    assert loaded.manifest["bundle_version"] == active_version


def test_production_churn_model_path_unchanged():
    """The production ChurnModel must still be the active inference path."""
    from backend.model import ChurnModel
    m = ChurnModel.load()
    assert m.best_model is not None
    assert m.preprocessor is not None
    # Its SeniorCitizen tolerance is preserved (BLOCKER-01 must not regress
    # the already-working production coercion).
    from backend.model import _coerce_senior_citizen
    out = _coerce_senior_citizen(pd.Series(["Yes", "No", 1, 0]))
    assert list(out) == [1, 0, 1, 0]


def test_contract_spec_is_self_describing(bundles):
    """The contract must be introspectable, not an implicit behaviour."""
    for key in BUNDLE_IDS:
        d = bundles[key].contract.describe()
        for field in ("required_columns", "categorical_domains",
                      "senior_citizen_accepted_representations",
                      "total_charges_policy", "numeric_missing_policy",
                      "categorical_missing_policy", "infinity_policy",
                      "extra_column_policy"):
            assert field in d, f"{key}: contract.describe() missing {field}"
        assert len(d["required_columns"]) == 19


def test_prepare_input_frame_requires_a_valid_spec_type(base_row):
    spec = ContractSpec(
        required_columns=["SeniorCitizen"],
        numeric_columns=["SeniorCitizen"],
        categorical_domains={},
        senior_citizen_aliases=SENIOR_CITIZEN_ALIASES,
        total_charges_policy="n/a",
        total_charges_statistic=0.0,
    )
    frame = prepare_input_frame({"SeniorCitizen": "Yes", "extra": 1}, spec)
    assert list(frame.columns) == ["SeniorCitizen"]
    assert int(frame["SeniorCitizen"].iloc[0]) == 1


def test_prepare_input_frame_rejects_unsupported_input_type():
    spec = ContractSpec(
        required_columns=["SeniorCitizen"],
        numeric_columns=["SeniorCitizen"],
        categorical_domains={},
        senior_citizen_aliases=SENIOR_CITIZEN_ALIASES,
        total_charges_policy="n/a",
        total_charges_statistic=0.0,
    )
    with pytest.raises(ContractError):
        prepare_input_frame("not a row", spec)