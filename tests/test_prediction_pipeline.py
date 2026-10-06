"""Regression tests for the unified RETENTIX AI prediction pipeline.

These tests lock in the Phase 1B contract: predict_single(), batch scoring and
the What-If simulator must all flow through backend.model.prepare_frame(), so a
customer's probability is identical regardless of the entry point, and the
persisted model artifacts must stay loadable and unchanged.

Run with:  python -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import (  # noqa: E402
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    REQUIRED_COLS,
    FALLBACK_TOTAL_CHARGES_MEDIAN,
    ChurnModel,
    derive_total_charges,
    encoder_categories,
    load_data,
    prepare_frame,
)

# The Phase 1B spec fixes the transformed feature count at 30.
EXPECTED_N_FEATURES = 30

# Recorded BEFORE the Phase 1B refactor for real dataset rows. These are
# observed historical TotalCharges values (no proxy involved), so they must be
# reproduced exactly after the change. Index -> (prediction, probability %).
REGRESSION_BASELINE = {
    0: ("Yes", 61.35),
    1: ("No", 4.54),
    2: ("No", 29.91),
    10: ("No", 20.65),
    100: ("No", 19.27),
    1000: ("Yes", 74.59),
    4000: ("Yes", 66.02),
    7042: ("No", 4.73),
}


@pytest.fixture(scope="module")
def model():
    return ChurnModel.load()


@pytest.fixture(scope="module")
def features(model):
    """Feature-only copy of the cleaned dataset (no target)."""
    return load_data().drop(columns=["Churn"])


@pytest.fixture(scope="module")
def sample_customer(features):
    """A representative real customer row as a plain dict."""
    return features.iloc[1000].to_dict()


# --------------------------------------------------------------------------
# A. Feature schema
# --------------------------------------------------------------------------
def test_a1_raw_schema_is_19_features():
    assert len(REQUIRED_COLS) == 19
    assert REQUIRED_COLS == CATEGORICAL_COLS + NUMERICAL_COLS
    assert len(set(REQUIRED_COLS)) == 19
    assert len(CATEGORICAL_COLS) == 15
    assert len(NUMERICAL_COLS) == 4


def test_a2_transform_produces_exactly_30_features(model, features):
    frame = prepare_frame(features, model.preprocessor, model.total_charges_median)
    transformed = model.preprocessor.transform(frame)
    assert transformed.shape == (len(features), EXPECTED_N_FEATURES)


def test_a3_saved_feature_names_match_preprocessor_output(model):
    from_preprocessor = list(model.preprocessor.get_feature_names_out())
    assert len(from_preprocessor) == EXPECTED_N_FEATURES
    # Saved feature_names.pkl must describe the same schema, in the same order.
    assert list(model.feature_names) == from_preprocessor


def test_a4_prepare_frame_returns_exact_schema(model, sample_customer):
    frame = prepare_frame(sample_customer, model.preprocessor,
                          model.total_charges_median)
    assert list(frame.columns) == REQUIRED_COLS
    assert len(frame) == 1


def test_a5_missing_required_field_raises(model, sample_customer):
    payload = dict(sample_customer)
    payload.pop("Contract")
    with pytest.raises(ValueError, match="Missing required field"):
        model.predict_single(payload)


# --------------------------------------------------------------------------
# B. Single vs batch
# --------------------------------------------------------------------------
def test_b1_single_and_batch_agree(model, features):
    _, probs = model.predict_batch(features)
    for i, (expected_pred, expected_prob) in REGRESSION_BASELINE.items():
        single = model.predict_single(features.iloc[i].to_dict())
        assert single["prediction"] == expected_pred
        assert single["churn_probability"] == pytest.approx(expected_prob, abs=0.01)
        # Batch is unrounded, so compare within the rounding tolerance.
        assert float(probs[i]) * 100 == pytest.approx(
            single["churn_probability"], abs=0.01
        )


def test_b2_single_and_batch_exact_equivalence_on_customers(model, features):
    """A customer scored alone must equal the same row scored inside a batch.

    predict_single() rounds to 2 decimals for display, so equivalence is
    asserted at the display precision; test_b1 also checks the unrounded
    values agree to within that same rounding step.
    """
    subset = features.iloc[[0, 5, 100, 4000]].reset_index(drop=True)
    _, probs = model.predict_batch(subset)
    for pos, row in subset.iterrows():
        single = model.predict_single(row.to_dict())
        assert single["churn_probability"] == round(float(probs[pos]) * 100, 2)


def test_b3_probabilities_sum_to_100(model, sample_customer):
    result = model.predict_single(sample_customer)
    assert result["churn_probability"] + result["no_churn_probability"] == \
        pytest.approx(100.0, abs=0.01)


# --------------------------------------------------------------------------
# C. Reordered input
# --------------------------------------------------------------------------
def test_c1_reordered_dict_keys_give_identical_prediction(model, sample_customer):
    forward = model.predict_single(dict(sample_customer))
    reversed_keys = dict(reversed(list(sample_customer.items())))
    reversed_result = model.predict_single(reversed_keys)
    assert forward == reversed_result


def test_c2_reordered_frame_columns_give_identical_prediction(model, features):
    row = features.iloc[100]
    forward = model.predict_single(row.to_dict())
    shuffled = model.predict_single(row.reindex(
        list(reversed(features.columns))).to_dict())
    assert forward == shuffled


# --------------------------------------------------------------------------
# D. Idempotence
# --------------------------------------------------------------------------
def test_d1_repeated_single_prediction_is_stable(model, sample_customer):
    first = model.predict_single(dict(sample_customer))
    for _ in range(4):
        assert model.predict_single(dict(sample_customer)) == first


def test_d2_repeated_batch_is_stable(model, features):
    subset = features.iloc[:200]
    _, probs_a = model.predict_batch(subset)
    _, probs_b = model.predict_batch(subset.copy())
    np.testing.assert_array_equal(probs_a, probs_b)


def test_d3_prepare_frame_does_not_mutate_caller_data(model, sample_customer):
    frame = prepare_frame(sample_customer, model.preprocessor,
                          model.total_charges_median)
    frame.iloc[0, frame.columns.get_loc("TotalCharges")] = -999.0
    # The caller's dict is untouched, so a second call is unaffected.
    assert model.predict_single(sample_customer)["churn_probability"] == \
        REGRESSION_BASELINE[1000][1]


# --------------------------------------------------------------------------
# E. NaN TotalCharges
# --------------------------------------------------------------------------
def test_e1_single_prediction_with_nan_total_charges_does_not_crash(
        model, sample_customer):
    payload = dict(sample_customer)
    payload["TotalCharges"] = np.nan
    result = model.predict_single(payload)
    assert 0.0 <= result["churn_probability"] <= 100.0


def test_e2_blank_string_total_charges_is_imputed(model, sample_customer):
    """Blank strings (the original dataset's null form) must coerce, not crash."""
    payload = dict(sample_customer)
    payload["TotalCharges"] = " "
    result = model.predict_single(payload)
    assert 0.0 <= result["churn_probability"] <= 100.0


def test_e3_batch_with_nan_total_charges_does_not_crash(model, features):
    batch = features.iloc[[0, 1, 2]].copy()
    batch["TotalCharges"] = np.nan
    preds, probs = model.predict_batch(batch)
    assert len(preds) == 3
    assert not np.isnan(probs).any()


def test_e4_all_nan_total_charges_batch_uses_training_median(model, features):
    """The all-NaN batch that previously failed now imputes cleanly."""
    batch = features.iloc[[0, 1, 2]].copy()
    batch["TotalCharges"] = np.nan
    _, probs = model.predict_batch(batch)

    expected = []
    for _, row in batch.iterrows():
        payload = row.to_dict()
        payload["TotalCharges"] = np.nan
        expected.append(model.predict_single(payload)["churn_probability"])
    for actual, exp in zip(probs * 100, expected):
        assert actual == pytest.approx(exp, abs=0.01)


def test_e5_single_and_batch_use_the_same_persisted_median(model, features):
    assert model.total_charges_median == pytest.approx(
        FALLBACK_TOTAL_CHARGES_MEDIAN
    )
    # A single-row frame with NaN must be filled with the persisted median.
    single_row = features.iloc[[0]].copy()
    single_row["TotalCharges"] = np.nan
    prepared = prepare_frame(single_row, model.preprocessor,
                             model.total_charges_median)
    assert prepared["TotalCharges"].iloc[0] == pytest.approx(
        model.total_charges_median
    )
    assert not prepared["TotalCharges"].isna().any()


def test_e6_missing_total_charges_is_never_batch_dependent(model, features):
    """The old bug: imputation used the uploaded batch's own median.

    A customer's score must not move when a wildly different row is added.
    """
    target = features.iloc[[0]].copy()
    target["TotalCharges"] = np.nan

    _, probs_alone = model.predict_batch(target)
    extra = features.iloc[[1]].copy()
    extra["TotalCharges"] = 999_999.0
    extra["MonthlyCharges"] = 150.0
    _, probs_with_extra = model.predict_batch(
        pd.concat([target, extra], ignore_index=True))
    assert float(probs_alone[0]) == pytest.approx(
        float(probs_with_extra[0]), abs=1e-12)


# --------------------------------------------------------------------------
# F. Batch stability
# --------------------------------------------------------------------------
def test_f1_adding_unrelated_rows_does_not_change_predictions(model, features):
    target = features.iloc[[42, 99]].reset_index(drop=True)
    _, base = model.predict_batch(target)
    _, with_extra = model.predict_batch(
        pd.concat([target, features.iloc[[7, 300, 1000, 5000]]],
                  ignore_index=True)
    )
    np.testing.assert_allclose(base, with_extra[:2], rtol=0, atol=1e-12)


def test_f2_adding_rows_with_extreme_values_does_not_change_predictions(
        model, features):
    target = features.iloc[[42, 99]].reset_index(drop=True)
    _, base = model.predict_batch(target)
    extreme = features.iloc[[1, 2]].copy()
    extreme["MonthlyCharges"] = 150.0
    extreme["TotalCharges"] = 1_000_000.0
    extreme["tenure"] = 72
    _, with_extra = model.predict_batch(
        pd.concat([target, extreme], ignore_index=True))
    np.testing.assert_allclose(base, with_extra[:2], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# G. Unknown categories
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [
    "Three year", "Lifetime", "month-to-month", "MONTH TO MONTH", "",
    "Fiber Optic", "two year", "credit card",
])
def test_g1_unknown_category_raises_value_error(model, sample_customer, bad_value):
    payload = dict(sample_customer)
    payload["Contract"] = bad_value
    with pytest.raises(ValueError) as excinfo:
        model.predict_single(payload)
    message = str(excinfo.value)
    assert "Contract" in message           # feature name
    assert bad_value in message             # invalid value
    assert "allowed values" in message      # permitted domain


def test_g2_unknown_category_is_not_silently_baselined(model, sample_customer):
    """A rejected value must not quietly score as the drop-first category."""
    baseline = model.predict_single(sample_customer)
    payload = dict(sample_customer)
    payload["Contract"] = "Three year"  # closest to the dropped "Month-to-month"
    with pytest.raises(ValueError):
        model.predict_single(payload)
    assert baseline["prediction"] in ("Yes", "No")


def test_g3_unknown_category_in_batch_raises(model, features):
    batch = features.iloc[[0, 1]].copy()
    batch.loc[0, "PaymentMethod"] = "Lifetime"
    with pytest.raises(ValueError, match="PaymentMethod"):
        model.predict_batch(batch)


def test_g4_error_lists_feature_value_and_allowed_values(model, sample_customer):
    payload = dict(sample_customer)
    payload["InternetService"] = "Satellite"
    with pytest.raises(ValueError) as excinfo:
        model.predict_single(payload)
    message = str(excinfo.value)
    assert "InternetService" in message
    assert "Satellite" in message
    assert "Fiber optic" in message   # an allowed value is named


def test_g5_all_trained_categories_are_accepted(model, features):
    """Every value the encoder actually saw must still pass validation."""
    known = encoder_categories(model.preprocessor)
    row = features.iloc[0].to_dict()
    for col, values in known.items():
        for value in values:
            payload = dict(row)
            payload[col] = value
            result = model.predict_single(payload)
            assert 0.0 <= result["churn_probability"] <= 100.0


def test_g6_every_categorical_column_is_validated(model, features):
    """No categorical feature may silently accept an unseen value."""
    known = encoder_categories(model.preprocessor)
    assert set(known) == set(CATEGORICAL_COLS)
    for col in CATEGORICAL_COLS:
        payload = features.iloc[0].to_dict()
        payload[col] = "__not_a_real_category__"
        with pytest.raises(ValueError, match=col):
            model.predict_single(payload)


# --------------------------------------------------------------------------
# H. SeniorCitizen
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (0, 0), (1, 1), (0.0, 0), (1.0, 1),
    ("0", 0), ("1", 1), ("0.0", 0), ("1.0", 1),
    ("Yes", 1), ("No", 0), ("yes", 1), (" no ", 0),
])
def test_h1_senior_citizen_coercion(model, sample_customer, raw, expected):
    payload = dict(sample_customer)
    payload["SeniorCitizen"] = raw
    result = model.predict_single(payload)
    reference = dict(sample_customer)
    reference["SeniorCitizen"] = expected
    assert result == model.predict_single(reference)


def test_h2_single_and_batch_share_senior_citizen_coercion(model, features):
    """The same SeniorCitizen input must score identically via either path.

    Compared at predict_single()'s 2-decimal display precision, since the
    batch path returns unrounded probabilities.
    """
    row = features.iloc[[7]].copy()
    for raw in [0, 1, "0", "1", "Yes", "No", 1.0]:
        batch_row = row.copy()
        batch_row["SeniorCitizen"] = raw
        _, batch_probs = model.predict_batch(batch_row)
        single = model.predict_single(
            {**row.iloc[0].to_dict(), "SeniorCitizen": raw})
        assert round(float(batch_probs[0]) * 100, 2) == \
            single["churn_probability"]


def test_h3_column_dtype_is_int64_on_every_path(model, features):
    payload = features.iloc[5].to_dict()
    payload["SeniorCitizen"] = "Yes"
    prepared = prepare_frame(payload, model.preprocessor,
                             model.total_charges_median)
    assert prepared["SeniorCitizen"].dtype == np.dtype("int64")
    assert prepared["SeniorCitizen"].iloc[0] == 1


def test_h4_invalid_senior_citizen_raises_clear_error(model, sample_customer):
    payload = dict(sample_customer)
    payload["SeniorCitizen"] = "maybe"
    with pytest.raises(ValueError) as excinfo:
        model.predict_single(payload)
    assert "SeniorCitizen" in str(excinfo.value)
    assert "allowed values" in str(excinfo.value)


# --------------------------------------------------------------------------
# I. Simulator
# --------------------------------------------------------------------------
def _ui_profile(features, idx):
    """Emulate what the UI builds: a proxy TotalCharges, not a measured one."""
    row = features.iloc[idx].to_dict()
    row["TotalCharges"] = derive_total_charges(row["MonthlyCharges"], row["tenure"])
    return row


def test_i1_derive_total_charges_rule_is_centred_and_stable():
    assert derive_total_charges(70.5, 24) == 70.5 * 24
    # The max(1, tenure) floor keeps zero-tenure customers non-zero.
    assert derive_total_charges(70.5, 0) == 70.5
    assert derive_total_charges(70.5, -5) == 70.5
    assert derive_total_charges(70.5, 1) == 70.5


def test_i2_simulator_baseline_no_intervention_returns_original(model, features):
    """Simulator with every lever at 'keep current' must be a no-op."""
    base = _ui_profile(features, 1000)
    original = model.predict_single(base)

    sim = dict(base)
    # Replicate views_predict.py: no contract change, no tech support, no
    # auto-pay switch, zero retention credit.
    sim["MonthlyCharges"] = max(18.0, base["MonthlyCharges"] - 0)
    sim["TotalCharges"] = derive_total_charges(sim["MonthlyCharges"],
                                               base["tenure"])
    assert model.predict_single(sim) == original


def test_i3_simulator_strict_contract_upgrade_reduces_risk(model, features):
    """Moving Month-to-month -> Two year plus autopay must not increase risk."""
    base = _ui_profile(features, 1000)
    assert base["Contract"] == "Month-to-month"
    original = model.predict_single(base)["churn_probability"]

    sim = dict(base)
    sim["Contract"] = "Two year"
    sim["PaymentMethod"] = "Credit card (automatic)"
    sim["TechSupport"] = "Yes"
    sim["OnlineSecurity"] = "Yes"
    sim["MonthlyCharges"] = max(18.0, base["MonthlyCharges"] - 10)
    sim["TotalCharges"] = derive_total_charges(sim["MonthlyCharges"],
                                               base["tenure"])
    improved = model.predict_single(sim)["churn_probability"]

    assert improved < original, (
        f"Expected risk to fall after a strict contract upgrade: "
        f"{original} -> {improved}"
    )


def test_i4_simulator_provenance_bias_is_bounded(model, features):
    """Quantify the proxy-vs-observed TotalCharges bias (documented limitation).

    The UI reconstructs TotalCharges from MonthlyCharges x tenure, whereas the
    model was fitted on true observed lifetime spend. This test guards the
    magnitude so a future change cannot make the proxy arbitrarily worse.
    """
    deltas = []
    for idx in [0, 10, 100, 1000, 4000]:
        real = features.iloc[idx].to_dict()
        proxied = _ui_profile(features, idx)
        deltas.append(abs(
            model.predict_single(proxied)["churn_probability"]
            - model.predict_single(real)["churn_probability"]
        ))
    mean_delta = float(np.mean(deltas))
    assert mean_delta < 12.0, f"Proxy bias unexpectedly large: {mean_delta:.3f}pp"
    assert max(deltas) < 25.0, f"Max proxy bias unexpectedly large: {max(deltas)}"


def test_i5_simulator_and_predictor_agree_on_identical_profiles(model, features):
    """The predictor and the simulator share one code path, so same input,
    same output."""
    profile = _ui_profile(features, 250)
    assert model.predict_single(profile) == model.predict_single(dict(profile))


# --------------------------------------------------------------------------
# J. Artifact compatibility
# --------------------------------------------------------------------------
def test_j1_model_expects_30_input_features(model):
    assert model.best_model.n_features_in_ == EXPECTED_N_FEATURES


def test_j2_artifacts_load_and_expose_expected_attributes(model):
    for attribute in ("preprocessor", "best_model", "best_model_name",
                      "feature_names", "results", "total_charges_median"):
        assert hasattr(model, attribute), f"missing artifact attribute: {attribute}"


def test_j3_persisted_median_is_a_plausible_positive_value(model):
    assert isinstance(model.total_charges_median, float)
    assert model.total_charges_median > 0
    # It must come from the training data, not the whole frame.
    full_frame_median = load_data()["TotalCharges"].median()
    assert model.total_charges_median != pytest.approx(full_frame_median)


def test_j4_prepared_frame_transforms_without_refitting(model, features):
    """prepare_frame + transform must not mutate the fitted preprocessor."""
    before = model.preprocessor.get_feature_names_out().tolist()
    scaler_mean = model.preprocessor.named_transformers_["num"].mean_.copy()
    model.predict_batch(features.iloc[:50])
    model.predict_single(features.iloc[0].to_dict())
    assert model.preprocessor.get_feature_names_out().tolist() == before
    np.testing.assert_array_equal(
        model.preprocessor.named_transformers_["num"].mean_, scaler_mean)


def test_j5_regression_baseline_unchanged_for_known_rows(model, features):
    """The headline guarantee of Phase 1B: real rows score exactly as before."""
    for i, (expected_pred, expected_prob) in REGRESSION_BASELINE.items():
        result = model.predict_single(features.iloc[i].to_dict())
        assert result["prediction"] == expected_pred
        assert result["churn_probability"] == pytest.approx(expected_prob, abs=0.01)


def test_j6_corpus_level_predictions_are_unchanged(model, features):
    """Corpus fingerprint, recorded before the refactor.

    Guards against any systematic drift across all 7,043 real customers.
    """
    _, probs = model.predict_batch(features)
    assert probs.size == 7043
    assert float(probs.mean()) == pytest.approx(0.2658696631, abs=1e-9)
    assert float(probs.std()) == pytest.approx(0.2442841503, abs=1e-9)
    assert float(probs.sum()) == pytest.approx(1872.52003749, abs=1e-6)
    assert float((probs > 0.5).mean()) == pytest.approx(0.2216385063, abs=1e-9)

