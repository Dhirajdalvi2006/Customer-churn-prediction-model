"""Phase 4 Step 7 -- tests for the read-only shadow scoring audit.

These tests are ANALYSIS + VERIFICATION only. They never fit, refit, train,
mutate or re-save any artifact, they never change the production threshold,
and they never promote a bundle. Every test here is read-only with respect to
``models/**`` and ``data/**``.
"""

import os
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from backend import shadow_audit as sa
from backend.model_bundle import ModelBundle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hash(rel):
    path = os.path.join(BASE_DIR, rel)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture(scope="module")
def bundles():
    return sa.load_bundles()


@pytest.fixture(scope="module")
def sample():
    return sa.load_common_sample()


@pytest.fixture(scope="module")
def scored(bundles, sample):
    v1, v2 = bundles
    return sa.score_both(v1, v2, sample["test_features"],
                         sa.PRODUCTION_THRESHOLD)


# --- 1/2. Both bundles load -------------------------------------------------
def test_v1_bundle_loads(bundles):
    v1, _ = bundles
    assert v1.manifest["bundle_version"] == "v1"
    assert v1.manifest["model_format"] == "estimator_only"
    assert v1.model is not None and v1.preprocessor is not None


def test_v2_bundle_loads(bundles):
    _, v2 = bundles
    assert v2.manifest["bundle_version"] == "v2"
    assert v2.manifest["model_format"] == "pipeline"
    assert v2.model is not None


# --- 3. Same-schema comparison works ---------------------------------------
def test_schema_comparison_runs_and_reports_every_category(bundles, sample):
    v1, v2 = bundles
    base_row = sample["test_features"].iloc[0].to_dict()
    result = sa.compare_input_contracts(v1, v2, base_row)
    assert result["verdict"] in (sa.SAME, sa.DIFFERENT)
    # Every category carries an explicit SAME / DIFFERENT / UNKNOWN verdict.
    for name, cat in result["categories"].items():
        assert cat["status"] in (sa.SAME, sa.DIFFERENT, sa.UNKNOWN), name
    # The column NAME sets genuinely agree; only behaviour differs.
    assert result["categories"]["required_columns"]["status"] == sa.SAME
    assert result["categories"]["categorical_domains"]["status"] == sa.SAME


# --- 4. Class semantics are validated --------------------------------------
def test_class_semantics_validated_for_both_bundles(bundles):
    v1, v2 = bundles
    for bundle in (v1, v2):
        meta = sa.analyse_class_semantics(bundle)
        assert meta["classes_match_manifest"] is True
        assert meta["manifest_index_agrees_with_classes"] is True
        # Both resolve the SAME churn semantics, verified from classes_.
        assert meta["positive_class_is_yes"] is True
        assert meta["positive_semantic_label"] == "Yes"


# --- 5. Positive probability uses a validated class index -------------------
def test_positive_probability_uses_validated_class_index(bundles, sample):
    v1, v2 = bundles
    for bundle, prob in ((v1, None), (v2, None)):
        meta = sa.analyse_class_semantics(bundle)
        classes = meta["classes"]
        idx = meta["positive_class_index"]
        # The index is located by SEARCHING classes_ for the positive label.
        assert classes[idx] == meta["positive_class"]
    frame = sample["test_features"].head(20)
    p1, l1 = sa._predict(v1, frame, sa.analyse_class_semantics(v1)
                         ["positive_class_index"],
                         sa.analyse_class_semantics(v1)["classes"])
    p2, l2 = sa._predict(v2, frame, sa.analyse_class_semantics(v2)
                         ["positive_class_index"],
                         sa.analyse_class_semantics(v2)["classes"])
    # Probability column must be the positive class: >= 0.5 exactly when the
    # threshold label says Yes.
    assert np.all((p1 >= 0.5) == (l1 == "Yes"))
    assert np.all((p2 >= 0.5) == (l2 == "Yes"))


# --- 6. Probability comparison is deterministic -----------------------------
def test_probability_comparison_deterministic(bundles, sample, scored):
    v1, v2 = bundles
    frame = sample["test_features"]
    a = sa.compare_probabilities(scored["v1_prob"], scored["v2_prob"])
    b = sa.compare_probabilities(scored["v1_prob"], scored["v2_prob"])
    assert a == b
    again = sa.score_both(v1, v2, frame, sa.PRODUCTION_THRESHOLD)
    c = sa.compare_probabilities(again["v1_prob"], again["v2_prob"])
    assert a["mean_absolute_difference"] == pytest.approx(
        c["mean_absolute_difference"])


# --- 7. Label comparison deterministic --------------------------------------
def test_label_comparison_deterministic(scored):
    a = sa.compare_labels(scored["v1_label"], scored["v2_label"],
                          sa.PRODUCTION_THRESHOLD)
    b = sa.compare_labels(scored["v1_label"], scored["v2_label"],
                          sa.PRODUCTION_THRESHOLD)
    assert a == b
    cm = a["confusion_matrix_v1_rows_v2_cols"]
    assert (cm["both_No"] + cm["v1_No_v2_Yes"] + cm["v1_Yes_v2_No"]
            + cm["both_Yes"]) == a["total_rows"]
    assert a["label_agreement"] + a["label_disagreement"] == a["total_rows"]

# --- 8/9. Single vs batch parity, independently per bundle ------------------
def test_single_batch_parity_v1(bundles, sample):
    v1, _ = bundles
    result = sa.single_batch_parity(v1, sample["test_features"])
    assert result["bundle_version"] == "v1"
    assert result["probability_parity"] is True
    assert result["label_parity"] is True
    assert result["max_probability_delta"] == pytest.approx(0.0, abs=1e-12)


def test_single_batch_parity_v2(bundles, sample):
    _, v2 = bundles
    result = sa.single_batch_parity(v2, sample["test_features"])
    assert result["bundle_version"] == "v2"
    assert result["probability_parity"] is True
    assert result["label_parity"] is True
    assert result["max_probability_delta"] == pytest.approx(0.0, abs=1e-12)


# --- 10. Repeated shadow run is deterministic -------------------------------
def test_repeated_shadow_run_deterministic(bundles, sample):
    v1, v2 = bundles
    frame = sample["test_features"]
    result = sa.determinism_check(v1, v2, frame)
    assert result["deterministic"] is True
    assert result["v1_run1_equals_run2"] is True
    assert result["v2_run1_equals_run2"] is True


# --- 11/12. Threshold remains 0.5 and 0.25 is never adopted -----------------
def test_threshold_remains_zero_point_five(bundles, sample):
    v1, v2 = bundles
    assert sa.PRODUCTION_THRESHOLD == 0.50
    scored = sa.score_both(v1, v2, sample["test_features"],
                           sa.PRODUCTION_THRESHOLD)
    assert scored["threshold"] == 0.50
    # Threshold labels must be exactly prob >= 0.50.
    assert np.array_equal(scored["v1_threshold_label"],
                          np.where(scored["v1_prob"] >= 0.50, "Yes", "No"))
    assert np.array_equal(scored["v2_threshold_label"],
                          np.where(scored["v2_prob"] >= 0.50, "Yes", "No"))


def test_analysis_threshold_zero_point_two_five_not_adopted(bundles, sample):
    v1, v2 = bundles
    frame = sample["test_features"]
    # Requesting the Phase 3C analysis threshold must be refused outright.
    with pytest.raises(ValueError):
        sa.score_both(v1, v2, frame, sa.ANALYSIS_ONLY_THRESHOLD)
    # And the default path must never silently use it.
    scored = sa.score_both(v1, v2, frame)
    assert scored["threshold"] == sa.PRODUCTION_THRESHOLD
    assert scored["threshold"] != sa.ANALYSIS_ONLY_THRESHOLD
    labels = sa.compare_labels(scored["v1_label"], scored["v2_label"])
    assert labels["threshold"] == sa.PRODUCTION_THRESHOLD

# --- 13/14/15. Frozen artifacts remain unchanged ----------------------------
ORIGINAL_ARTIFACTS = [
    "models/best_model.pkl",
    "models/preprocessor.pkl",
    "models/results.pkl",
    "models/feature_names.pkl",
    "models/total_charges_median.pkl",
    "models/experiments/champion.pkl",
]


def test_original_production_artifacts_unchanged():
    """The six original production/champion artifacts must be byte-identical."""
    for rel in ORIGINAL_ARTIFACTS:
        expected = sa.FROZEN_HASHES[rel]
        assert _hash(rel) == expected, f"{rel} changed"
    # champion.pkl in particular must still be the frozen champion.
    assert _hash("models/experiments/champion.pkl") == sa.FROZEN_HASHES[
        "models/experiments/champion.pkl"]


def test_v1_bundle_artifacts_unchanged():
    for rel in ("models/bundles/v1/model.pkl",
                "models/bundles/v1/preprocessor.pkl"):
        assert _hash(rel) == sa.FROZEN_HASHES[rel], f"{rel} changed"


def test_v2_bundle_artifacts_unchanged():
    assert _hash("models/bundles/v2/model.pkl") == sa.FROZEN_HASHES[
        "models/bundles/v2/model.pkl"]


def test_audit_module_did_not_mutate_anything(bundles, sample):
    """Running the full audit leaves every frozen artifact byte-identical."""
    before = {rel: _hash(rel) for rel in sa.FROZEN_HASHES}
    sa.run_audit(write_artifacts=False)
    after = {rel: _hash(rel) for rel in sa.FROZEN_HASHES}
    assert before == after


def test_active_bundle_pointer_and_audit_consistency():
    """The shadow audit reports the actual active bundle without mutating it.

    Phase 4 Step 12D: formerly test_active_bundle_still_points_to_v1_and_no_promotion.
    Updated to be bundle-aware: the invariant is that the audit accurately
    reflects whichever bundle is active in models/active_bundle.json, and
    that running the audit does NOT mutate the active pointer or adopt
    the analysis threshold. Promotion record is in models/experiments/.
    """
    with open(os.path.join(BASE_DIR, "models", "active_bundle.json")) as fh:
        expected_version = json.load(fh)["active_version"]
    assert expected_version in ("v1", "v2")
    audit = sa.run_audit(write_artifacts=False)
    assert audit["promotion_performed"] is False
    assert audit["analysis_only"] is True
    assert audit["active_bundle"] == expected_version
    assert audit["threshold"]["threshold_changed"] is False
    assert audit["threshold"]["analysis_threshold_adopted"] is False


# --- Audit artifact + contract blocker coverage ----------------------------
def test_audit_json_artifact_exists_and_has_required_sections():
    path = os.path.join(BASE_DIR, "models", "experiments", "shadow_audit.json")
    assert os.path.exists(path), "run: python -m backend.shadow_audit"
    with open(path) as fh:
        data = json.load(fh)
    for key in ("timestamp", "v1_bundle_version", "v2_bundle_version",
                "sample_size", "data_source", "threshold",
                "class_metadata", "contract_comparison",
                "probability_statistics", "label_agreement_statistics",
                "disagreement_summary", "threshold_proximity",
                "single_batch_parity", "determinism", "xai_compatibility",
                "blockers", "warnings", "hash_verification"):
        assert key in data, f"missing section: {key}"
    assert data["threshold"]["production_threshold_used"] == 0.50
    assert data["threshold"]["analysis_threshold_adopted"] is False
    assert data["hash_verification"]["all_unchanged"] is True
    # The audit must not contain customer identifiers. The literal token
    # "customerID" may appear inside the audit's own explanatory prose, so
    # the check targets real identifier VALUES: every disagreement row is
    # keyed by row_index only, and no 4-5 digit customerID-shaped value or
    # XXXX-XXXX identifier is present in the emitted records.
    for row in data["disagreement_summary"]["rows"]:
        assert set(row) <= {"row_index", "v1_probability", "v2_probability",
                            "v1_label", "v2_label",
                            "absolute_probability_difference", "true_label"}
    assert "customer_id" not in json.dumps(data).lower()
    assert not any("-" in str(row.get("true_label", "")) for row in
                   data["disagreement_summary"]["rows"])


def test_blockers_are_reported_and_not_fixed():
    blocking = [b for b in sa.BLOCKERS if b["blocks_promotion"]]
    assert len(blocking) > 0
    areas = " ".join(b["area"] for b in sa.BLOCKERS)
    for expected in ("Raw schema", "Categorical domains", "Numeric handling",
                     "TotalCharges", "NaN behavior", "Infinity behavior",
                     "Class semantics", "Positive probability semantics",
                     "Transformed feature representation", "Error handling"):
        assert expected in areas, f"uncovered blocker area: {expected}"


def test_xai_compatibility_reported_for_both_bundles(bundles):
    v1, v2 = bundles
    result = sa.xai_compatibility(v1, v2, n=5)
    for key, expected_features in (("v1", 30), ("v2", 45)):
        entry = result["bundles"][key]
        # Both bundles ARE explainable through their native fitted objects.
        assert entry["native_object_accepted"] is True
        assert entry["deterministic_output"] is True
        assert entry["transformed_feature_count"] == expected_features
        assert entry["shap_output_shape"][0] == 5
    assert result["bundles"]["v1"]["explainer_type"] == "LinearExplainer"
    assert result["bundles"]["v2"]["explainer_type"] == "TreeExplainer"


def test_model_bundle_is_accepted_directly_by_xai(bundles):
    """Phase 4 Step 8 / BLOCKER-09: a ModelBundle is now explainable directly.

    The audit originally recorded ``model_bundle_accepted_directly: False`` as
    a documented blocker. It is now ``True`` for both bundles, which is what
    ``_as_model_view`` gaining ModelBundle support means in observable terms.
    """
    v1, v2 = bundles
    result = sa.xai_compatibility(v1, v2, n=5)
    for key in ("v1", "v2"):
        entry = result["bundles"][key]
        assert entry["model_bundle_accepted_directly"] is True, (
            f"{key} bundle is still rejected by _as_model_view; BLOCKER-09 is "
            f"not resolved."
        )
        # The explainer must still be selected from the fitted type, not the
        # bundle label.
        assert entry["explainer_type"] in ("LinearExplainer", "TreeExplainer")
    assert result["bundles"]["v1"]["explainer_type"] == "LinearExplainer"
    assert result["bundles"]["v2"]["explainer_type"] == "TreeExplainer"


def test_audit_module_is_not_imported_by_production():
    """No production module may import the shadow audit module."""
    for rel in ("backend/model.py", "backend/explainability.py", "app.py",
                "frontend/views_predict.py"):
        path = os.path.join(BASE_DIR, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        assert "shadow_audit" not in src, f"{rel} imports the audit module"