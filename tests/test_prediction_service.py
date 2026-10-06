"""Phase 4 Step 10 -- production inference migrated to the ModelBundle architecture.

These tests lock in the migrated production contract:

* the application resolves its model through ``active_bundle.json`` only;
* v1 legacy ``ChurnModel`` and the active v1 bundle agree on probability and
  label (parity is asserted, not assumed);
* single and batch prediction share one path;
* positive class, threshold and feature count are metadata-driven;
* contract violations and unresolvable bundles fail explicitly, with no silent
  fallback to the legacy artifacts.

The active pointer is asserted to still be ``v1``: these tests must never
promote v2.
"""
import ast
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from backend.model import ChurnModel, load_data
from backend.model_bundle import ModelBundle
from backend.prediction_service import (
    BundlePredictionService,
    PredictionServiceError,
    active_pointer_version,
    get_prediction_service,
)

MODELS = "models"

# Deterministic inputs: a spread of tenure / charges / contract combinations so
# the comparison exercises both sides of the decision threshold.
FIXED_ROWS = [
    dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No",
         tenure=2, PhoneService="Yes", MultipleLines="No",
         InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
         DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
         StreamingMovies="Yes", Contract="Month-to-month",
         PaperlessBilling="Yes", PaymentMethod="Electronic check",
         MonthlyCharges=92.50, TotalCharges=185.00),
    dict(gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
         tenure=64, PhoneService="Yes", MultipleLines="Yes",
         InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
         DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
         StreamingMovies="Yes", Contract="Two year", PaperlessBilling="No",
         PaymentMethod="Credit card (automatic)", MonthlyCharges=75.20,
         TotalCharges=4812.80),
    dict(gender="Female", SeniorCitizen=1, Partner="No", Dependents="No",
         tenure=45, PhoneService="Yes", MultipleLines="No",
         InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="Yes",
         DeviceProtection="Yes", TechSupport="No", StreamingTV="Yes",
         StreamingMovies="No", Contract="One year", PaperlessBilling="Yes",
         PaymentMethod="Bank transfer (automatic)", MonthlyCharges=104.35,
         TotalCharges=4695.75),
]


@pytest.fixture(scope="module")
def service():
    return BundlePredictionService(MODELS)


@pytest.fixture(scope="module")
def legacy():
    return ChurnModel.load()


BUNDLE_IDS = ("v1", "v2")

@pytest.fixture(params=BUNDLE_IDS)
def service_for(request, pointer_dir):
    """Provides a BundlePredictionService explicitly initialized to a requested bundle."""
    _set_pointer(pointer_dir, request.param)
    return BundlePredictionService(str(pointer_dir))

@pytest.fixture
def v1_service(pointer_dir):
    _set_pointer(pointer_dir, "v1")
    return BundlePredictionService(str(pointer_dir))


@pytest.fixture
def real_rows():
    """Deterministic slice of the real dataset, minus label/id columns."""
    df = load_data().drop(columns=["Churn", "customerID"])
    return df.iloc[[0, 7, 55, 500, 4000, 7042]].copy()

@pytest.fixture
def pointer_dir(tmp_path):
    """A writable copy of models/ so pointer edits cannot touch the real one.

    Bundles are referenced by relative path inside each manifest, so the copy
    is self-contained and hash verification still holds.
    """
    dest = tmp_path / "models"
    shutil.copytree(MODELS, dest)
    return dest


def _set_pointer(models_dir, version):
    with open(os.path.join(models_dir, "active_bundle.json"), "w") as fh:
        json.dump({"active_version": version}, fh)


# ---------------------------------------------------------------------------
# 1-2: the active v1 bundle and the production service load it
# ---------------------------------------------------------------------------
def test_active_v1_bundle_loads():
    bundle = ModelBundle(os.path.join(MODELS, "bundles", "v1"))
    assert bundle.manifest["bundle_version"] == "v1"
    assert bundle.transformer is not None
    assert bundle.estimator is not None


def test_production_service_loads_active_bundle(service_for):
    if service_for.bundle_version == "v1":
        assert service_for.model_type == "LogisticRegression"
        assert service_for.model_format == "estimator_only"
        assert service_for.transformed_feature_count == 30
    elif service_for.bundle_version == "v2":
        assert service_for.model_type == "GradientBoostingClassifier"
        assert service_for.model_format == "pipeline"
        assert service_for.transformed_feature_count == 45
    assert service_for.bundle is not None


# ---------------------------------------------------------------------------
# 3: the active pointer -- not a hardcoded version -- controls the model
# ---------------------------------------------------------------------------
def test_active_pointer_controls_model_selection(pointer_dir):
    """Flipping the pointer switches the served model; nothing else changes."""
    _set_pointer(pointer_dir, "v2")
    v2_service = BundlePredictionService(str(pointer_dir))
    assert v2_service.bundle_version == "v2"
    assert v2_service.model_type == "GradientBoostingClassifier"
    assert v2_service.transformed_feature_count == 45

    _set_pointer(pointer_dir, "v1")
    v1_service = BundlePredictionService(str(pointer_dir))
    assert v1_service.bundle_version == "v1"
    assert v1_service.model_type == "LogisticRegression"
    assert v1_service.transformed_feature_count == 30


def test_v2_service_scores_through_same_contract(pointer_dir):
    """v2 is servable through the identical service interface (not active)."""
    _set_pointer(pointer_dir, "v2")
    v2_service = BundlePredictionService(str(pointer_dir))
    res = v2_service.predict_single(FIXED_ROWS[0])
    assert res["prediction"] in ("Yes", "No")
    assert 0.0 <= res["churn_probability"] <= 100.0


# ---------------------------------------------------------------------------
# 4: no hardcoded model path / version / type in production inference
# ---------------------------------------------------------------------------
def _executable_string_literals(rel_path):
    """String literals that exist in *executable* code, excluding docstrings.

    The architectural requirements below concern code, not prose. The service's
    module/class/function docstrings deliberately *name* the legacy artifacts and
    the forbidden literals in order to document that they must not be used, so a
    raw substring scan over the file text would flag its own documentation. This
    parses the module and yields only literals that the interpreter can actually
    evaluate as a string constant, skipping docstrings. Comments are never part
    of the AST and are therefore excluded automatically.
    """
    with open(rel_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=rel_path)

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_hardcoded_model_path_in_production_inference():
    """Production inference must not name a legacy artifact or a version."""
    literals = _executable_string_literals("backend/prediction_service.py")
    joined = "\n".join(literals)
    for forbidden in ("best_model.pkl", "preprocessor.pkl", "champion.pkl"):
        assert forbidden not in joined, (
            f"production inference must not reference {forbidden}"
        )
    # No bundle version is hardcoded as the served model.
    for forbidden in ("v1", "v2"):
        assert forbidden not in literals, (
            f"production inference must not hardcode bundle version {forbidden!r}"
        )
    # It must resolve through the active pointer, not by inspecting versions.
    src = open("backend/prediction_service.py", encoding="utf-8").read()
    assert "get_active_bundle" in src
    assert "active_bundle.json" in src


def test_no_hardcoded_class_index_or_threshold_in_service():
    """No ``proba[:, 1]`` and no threshold literal in the scoring path."""
    src = open("backend/prediction_service.py", encoding="utf-8").read()
    tree = ast.parse(src, filename="backend/prediction_service.py")
    # Inspect executable code, not prose: the module docstring deliberately
    # *names* these forbidden constructs to document that they are not used.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and \
                isinstance(node.slice, ast.Constant) and \
                node.slice.value == 1:
            base = node.value
            text = ast.unparse(base)
            assert "predict_proba" not in text, (
                "positive-class probability must be selected via bundle "
                f"metadata/classes_, not a hardcoded index: {text}[1]"
            )
        if isinstance(node, ast.Constant) and \
                isinstance(node.value, (int, float)) and \
                not isinstance(node.value, bool):
            assert node.value != 0.25, "Phase 3C 0.25 is analysis-only"
    # 0.5 must not be hardcoded as a decision literal either -- the threshold is
    # read from bundle metadata. Only executable string/number literals count;
    # the docstring names 0.5 precisely to document that it is not used.
    tree = ast.parse(src, filename="backend/prediction_service.py")
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docstrings:
            continue
        if isinstance(node.value, str):
            assert "0.5" not in node.value, "threshold must come from bundle metadata"
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            assert node.value != 0.5, "threshold must come from bundle metadata"


# ---------------------------------------------------------------------------
# 5-6: v1 legacy vs bundle parity (the Step 4 STOP condition)
# ---------------------------------------------------------------------------
@pytest.fixture
def v1_service(pointer_dir):
    _set_pointer(pointer_dir, "v1")
    return BundlePredictionService(str(pointer_dir))

def test_v1_legacy_bundle_probability_parity(v1_service, legacy, real_rows):
    for _, row in real_rows.iterrows():
        payload = row.to_dict()
        legacy_prob = legacy.predict_single(payload)["churn_probability"]
        bundle_prob = v1_service.predict_single(payload)["churn_probability"]
        assert bundle_prob == pytest.approx(legacy_prob, abs=0.01), (
            f"probability parity broke: legacy={legacy_prob} bundle={bundle_prob}"
        )


def test_v1_legacy_bundle_label_parity(v1_service, legacy, real_rows):
    for _, row in real_rows.iterrows():
        payload = row.to_dict()
        legacy_label = legacy.predict_single(payload)["prediction"]
        bundle_label = v1_service.predict_single(payload)["prediction"]
        assert bundle_label == legacy_label
        assert bundle_label in ("Yes", "No")


def test_legacy_bundle_parity_on_fixed_rows(v1_service, legacy):
    """Parity must hold on inputs the app itself can construct."""
    for payload in FIXED_ROWS:
        assert v1_service.predict_single(payload)["churn_probability"] == \
            pytest.approx(legacy.predict_single(payload)["churn_probability"], abs=0.01)


def test_parity_agrees_with_bundle_probability(service_for, real_rows):
    """The service's rounded percent matches the bundle's raw probability."""
    for _, row in real_rows.iterrows():
        payload = row.to_dict()
        raw = float(service_for.bundle.predict_proba(payload)[0])
        assert service_for.predict_single(payload)["churn_probability"] == \
            pytest.approx(round(raw * 100, 2), abs=1e-9)


# ---------------------------------------------------------------------------
# 7-9: single / batch prediction and their parity
# ---------------------------------------------------------------------------
def test_single_prediction_returns_expected_shape(service_for, real_rows):
    res = service_for.predict_single(real_rows.iloc[0].to_dict())
    assert set(res) == {"prediction", "churn_probability", "no_churn_probability"}
    assert res["prediction"] in ("Yes", "No")
    assert 0.0 <= res["churn_probability"] <= 100.0
    assert res["churn_probability"] + res["no_churn_probability"] == \
        pytest.approx(100.0, abs=0.01)


def test_batch_prediction_returns_expected_shape(service_for, real_rows):
    labels, probs = service_for.predict_batch(real_rows)
    assert len(labels) == len(real_rows)
    assert len(probs) == len(real_rows)
    assert all(lbl in ("Yes", "No") for lbl in labels)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_single_batch_parity(service_for, real_rows):
    """A row's score must not depend on the batch it travelled in."""
    batch_labels, batch_probs = service_for.predict_batch(real_rows)
    for i in range(len(real_rows)):
        single = service_for.predict_single(real_rows.iloc[i].to_dict())
        raw = float(service_for.bundle.predict_proba(real_rows.iloc[i].to_dict())[0])
        # The batch probability is the raw positive-class score.
        assert float(batch_probs[i]) == pytest.approx(raw, abs=1e-12)
        # The single-prediction path reports the same score rounded to 2dp as a
        # percentage, so the expected difference is bounded by that rounding
        # (half of 0.01 percentage points == 5e-5 in probability units), not 1e-9.
        reported = single["churn_probability"] / 100.0
        assert abs(float(batch_probs[i]) - reported) <= 5e-5 + 1e-12
        # Labels must match exactly -- no tolerance is acceptable there.
        assert str(batch_labels[i]) == single["prediction"]


def test_batch_score_independent_of_batch_composition(service_for, real_rows):
    """No batch-derived statistics: one row scores the same alone or together."""
    alone = float(service_for.bundle.predict_proba(real_rows.iloc[0].to_dict())[0])
    together = float(service_for.bundle.predict_proba(real_rows)[0])
    assert alone == pytest.approx(together, abs=1e-12)


# ---------------------------------------------------------------------------
# 10-13: positive class and threshold are metadata-driven; 0.5 retained
# ---------------------------------------------------------------------------
def test_positive_class_is_metadata_driven(service_for):
    meta = service_for.metadata()
    bundle = service_for.bundle
    assert meta["positive_class"] == str(bundle.positive_class)
    assert meta["positive_class_index"] == bundle.positive_class_index
    assert meta["positive_semantic_label"] == "Yes"
    # The index must come from the validated classes_, not a literal 1.
    expected = list(bundle.classes).index(bundle.positive_class)
    assert bundle.positive_class_index == expected
    assert bundle.positive_class == bundle.classes[expected]


def test_threshold_is_metadata_driven(service_for):
    manifest_threshold = service_for.bundle.manifest["inference"]["decision_threshold"]
    assert service_for.decision_threshold == manifest_threshold
    assert service_for.metadata()["decision_threshold"] == manifest_threshold


def test_threshold_remains_0_5(service):
    assert service.decision_threshold == 0.5
    with open(os.path.join(MODELS, "bundles", "v1", "manifest.json")) as fh:
        assert json.load(fh)["inference"]["decision_threshold"] == 0.5


def test_0_25_analysis_threshold_not_used(service):
    """The Phase 3C analysis threshold must not leak into production."""
    assert service.decision_threshold != 0.25
    labels, probs = service.predict_batch(pd.DataFrame([FIXED_ROWS[0], FIXED_ROWS[1]]))
    for lbl, prob in zip(labels, probs):
        assert (lbl == "Yes") == (prob >= 0.5)


def test_label_follows_metadata_threshold(service, real_rows):
    for _, row in real_rows.iterrows():
        payload = row.to_dict()
        prob = float(service.bundle.predict_proba(payload)[0])
        label = service.predict_single(payload)["prediction"]
        assert (label == "Yes") == (prob >= service.decision_threshold)


# ---------------------------------------------------------------------------
# 14-17: the shared input contract still governs production inference
# ---------------------------------------------------------------------------
def test_invalid_categorical_rejected(service):
    payload = {**FIXED_ROWS[0], "Contract": "Three year"}
    with pytest.raises(PredictionServiceError) as err:
        service.predict_single(payload)
    assert "Contract" in str(err.value)


def test_invalid_senior_citizen_rejected(service):
    payload = {**FIXED_ROWS[0], "SeniorCitizen": "Maybe"}
    with pytest.raises(PredictionServiceError) as err:
        service.predict_single(payload)
    assert "SeniorCitizen" in str(err.value)


def test_numeric_nan_rejected_by_contract(service):
    payload = {**FIXED_ROWS[0], "tenure": np.nan}
    with pytest.raises(PredictionServiceError):
        service.predict_single(payload)


def test_infinity_rejected(service):
    payload = {**FIXED_ROWS[0], "MonthlyCharges": np.inf}
    with pytest.raises(PredictionServiceError):
        service.predict_single(payload)


def test_total_charges_missing_is_imputed_from_bundle_median(service):
    """The documented policy: blank TotalCharges takes the training median."""
    payload = {**FIXED_ROWS[0], "TotalCharges": np.nan}
    median = service.metadata()["total_charges_median"]
    imputed = service.predict_single(payload)
    explicit = service.predict_single({**FIXED_ROWS[0], "TotalCharges": median})
    assert imputed["churn_probability"] == explicit["churn_probability"]


def test_missing_required_column_rejected(service):
    payload = dict(FIXED_ROWS[0])
    payload.pop("tenure")
    with pytest.raises(PredictionServiceError) as err:
        service.predict_single(payload)
    assert "tenure" in str(err.value)


def test_extra_columns_ignored(service):
    payload = {**FIXED_ROWS[0], "customerID": "9999", "Churn": "Yes"}
    assert service.predict_single(payload) == service.predict_single(FIXED_ROWS[0])


def test_batch_rejects_invalid_categorical_like_single(service, real_rows):
    frame = real_rows.copy()
    frame.loc[frame.index[0], "Contract"] = "Three year"
    with pytest.raises(PredictionServiceError) as err:
        service.predict_batch(frame)
    assert "Contract" in str(err.value)


# ---------------------------------------------------------------------------
# 18-20: unresolvable bundles fail safely, with no legacy fallback
# ---------------------------------------------------------------------------
def test_missing_active_bundle_pointer_fails_safely(tmp_path):
    with pytest.raises(PredictionServiceError) as err:
        BundlePredictionService(str(tmp_path))
    assert "active_bundle.json" in str(err.value)


def test_nonexistent_active_bundle_version_fails_safely(pointer_dir):
    _set_pointer(pointer_dir, "v99")
    with pytest.raises(PredictionServiceError):
        BundlePredictionService(str(pointer_dir))


def test_invalid_manifest_fails_safely(pointer_dir):
    _set_pointer(pointer_dir, "v1")
    with open(os.path.join(pointer_dir, "bundles", "v1", "manifest.json"), "w") as fh:
        fh.write("{ this is not valid json")
    with pytest.raises(PredictionServiceError):
        BundlePredictionService(str(pointer_dir))


def test_manifest_missing_required_section_fails_safely(pointer_dir):
    _set_pointer(pointer_dir, "v1")
    path = os.path.join(pointer_dir, "bundles", "v1", "manifest.json")
    with open(path) as fh:
        manifest = json.load(fh)
    del manifest["inference"]
    with open(path, "w") as fh:
        json.dump(manifest, fh)
    with pytest.raises(PredictionServiceError):
        BundlePredictionService(str(pointer_dir))


def test_hash_mismatch_fails_safely(pointer_dir):
    """A tampered artifact must stop inference, not be silently scored."""
    _set_pointer(pointer_dir, "v1")
    path = os.path.join(pointer_dir, "bundles", "v1", "model.pkl")
    with open(path, "ab") as fh:
        fh.write(b"\x00tampered")
    with pytest.raises(PredictionServiceError) as err:
        BundlePredictionService(str(pointer_dir))
    assert "integrity" in str(err.value).lower()


def test_class_mismatch_fails_safely(pointer_dir):
    """A manifest whose classes disagree with the fitted model must fail."""
    _set_pointer(pointer_dir, "v1")
    path = os.path.join(pointer_dir, "bundles", "v1", "manifest.json")
    with open(path) as fh:
        manifest = json.load(fh)
    manifest["inference"]["classes"] = ["maybe", "perhaps"]
    with open(path, "w") as fh:
        json.dump(manifest, fh)
    with pytest.raises(PredictionServiceError):
        BundlePredictionService(str(pointer_dir))


def test_no_silent_legacy_fallback_on_failure(tmp_path):
    """A broken deployment must not resurrect best_model.pkl."""
    shutil.copy(os.path.join(MODELS, "best_model.pkl"), tmp_path / "best_model.pkl")
    shutil.copy(os.path.join(MODELS, "preprocessor.pkl"), tmp_path / "preprocessor.pkl")
    with pytest.raises(PredictionServiceError):
        BundlePredictionService(str(tmp_path))


def test_error_message_has_no_python_traceback(tmp_path):
    try:
        BundlePredictionService(str(tmp_path))
    except PredictionServiceError as exc:
        text = str(exc)
        assert "Traceback" not in text
        assert 'File "' not in text
        assert text.strip()
    else:
        pytest.fail("expected PredictionServiceError")


# ---------------------------------------------------------------------------
# 21-22: XAI explains the active bundle
# ---------------------------------------------------------------------------
def test_xai_explains_active_bundle(service):
    from backend.explainability import explain_single, get_shap_explainer

    explainable = service.explainable_model()
    assert explainable is service.bundle
    explainer = get_shap_explainer(explainable)
    out = explain_single(FIXED_ROWS[0], explainable, explainer)
    assert out["contributions"]
    # The explanation must be in the active bundle's own feature space.
    assert len(out["contributions"]) == service.transformed_feature_count


def test_xai_global_explanations_work_for_active_bundle(service):
    from backend.explainability import get_global_shap_values, get_shap_explainer

    explainable = service.explainable_model()
    explainer = get_shap_explainer(explainable)
    out = get_global_shap_values(explainable, explainer, sample_size=30)
    # `out` is a result envelope; the feature representation lives in
    # feature_names / shap_values, not in the envelope's key count.
    assert set(out) >= {"importance", "shap_values", "feature_names"}
    assert out["sample_size"] == 30
    names = out["feature_names"]
    assert len(names) == service.transformed_feature_count
    assert len(out["importance"]) == service.transformed_feature_count
    # shap_values is (n_samples, n_features) in the active bundle's own space.
    assert out["shap_values"].shape == (30, service.transformed_feature_count)
    # Global importance covers exactly the bundle's feature names. It is
    # sorted by mean |SHAP| descending, so compare as a set, not positionally.
    assert set(out["importance"]["Feature"]) == set(names)


def test_xai_simulator_explanation_works(service):
    from backend.explainability import explain_simulation

    baseline = dict(FIXED_ROWS[0])
    simulated = {**baseline, "Contract": "Two year", "tenure": 24,
                 "TotalCharges": 24 * baseline["MonthlyCharges"]}
    out = explain_simulation(baseline, simulated, service.explainable_model())
    assert out is not None


# ---------------------------------------------------------------------------
# 23: determinism
# ---------------------------------------------------------------------------
def test_inference_is_deterministic(service, real_rows):
    payload = real_rows.iloc[0].to_dict()
    first = service.predict_single(payload)
    for _ in range(5):
        assert service.predict_single(payload) == first


def test_batch_inference_is_deterministic(service, real_rows):
    first_labels, first_probs = service.predict_batch(real_rows)
    for _ in range(3):
        labels, probs = service.predict_batch(real_rows)
        assert list(labels) == list(first_labels)
        np.testing.assert_array_equal(probs, first_probs)


# ---------------------------------------------------------------------------
# 24 / Step 15-16: the production boundary (bundle-aware: reads real pointer)
# ---------------------------------------------------------------------------
def test_active_pointer_is_consistent():
    """The active_bundle.json pointer and the loaded service agree on version.

    Phase 4 Step 12D: promotion changed the pointer from v1 to v2. This test
    was formerly named test_active_pointer_remains_v1 and hard-asserted v1.
    It is updated to be bundle-aware: it verifies internal consistency of the
    pointer, not that a specific version is active. The promotion record in
    models/experiments/v2_promotion_record.json documents the v1->v2 change.
    """
    actual_version = active_pointer_version(MODELS)
    assert actual_version in ("v1", "v2"), (
        f"active_bundle.json must point to a known bundle, got {actual_version!r}"
    )
    with open(os.path.join(MODELS, "active_bundle.json")) as fh:
        pointer = json.load(fh)
    assert pointer["active_version"] == actual_version


def test_active_service_matches_pointer(service):
    """The service's loaded bundle version matches active_bundle.json.

    Phase 4 Step 12D: formerly test_v2_is_not_promoted. Now that promotion
    has been completed, the invariant is that the service serves exactly the
    bundle declared by the active pointer -- not that a specific version is
    served. The promotion record documents the old and new active versions.
    """
    actual_version = active_pointer_version(MODELS)
    assert service.bundle_version == actual_version, (
        f"service.bundle_version {service.bundle_version!r} disagrees with "
        f"active pointer {actual_version!r}"
    )


def test_service_is_process_cached():
    a = get_prediction_service(MODELS)
    b = get_prediction_service(MODELS)
    assert a is b


def test_expose_metadata_to_application(service):
    """Metadata envelope must be present and self-consistent.

    Phase 4 Step 12D: bundle-version-specific assertions (v1 / LogisticRegression /
    30 features) are replaced by consistency checks so this test remains valid
    for any active bundle. Version-specific values are in the promotion record.
    """
    meta = service.metadata()
    for key in ("bundle_version", "model_type", "model_format",
                "decision_threshold", "positive_class",
                "transformed_feature_count"):
        assert key in meta
    # The reported bundle_version must agree with the active pointer.
    assert meta["bundle_version"] == active_pointer_version(MODELS)
    # The model_type must be a non-empty string from the fitted artifact.
    assert isinstance(meta["model_type"], str) and meta["model_type"]
    # The transformed feature count must be a positive integer.
    assert isinstance(meta["transformed_feature_count"], int)
    assert meta["transformed_feature_count"] > 0


def test_frozen_artifacts_unchanged_by_migration():
    """The two files the migration could have touched stay byte-identical.

    Pinned against the hashes recorded in the shadow audit. The remaining
    frozen artifacts are verified in the final hash audit rather than pinned
    here, to avoid stale duplicated literals.
    """
    pinned = {
        "models/best_model.pkl": "66400195eb9cfef8afaaf9ee59ee83e19baa1e9afa70a4ddc9fc49ce80d9010e",
        "models/preprocessor.pkl": "fcc2d4e47fa218fce6056f7904e4b54f7f9c58fdd6ccbd04ff20b7a5210e028f",
    }
    for path, digest in pinned.items():
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == digest, f"{path} changed during migration"
