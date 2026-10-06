"""Streamlit AppTest smoke tests covering all 6 pages and the key flows.

Verifies the unified preprocessing path end-to-end through the real UI:
single prediction, batch scoring, CSV upload, What-If simulator, invalid
categorical input and missing TotalCharges.

Run with:  python -m pytest tests/test_streamlit_app_smoke.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest  # noqa: E402

from backend.model import REQUIRED_COLS, ChurnModel, load_data  # noqa: E402

APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)

PAGES = [
    "🏠 Executive Command Center",
    "📊 Exploratory Intelligence",
    "🔍 Churn Drivers & Ecosystem",
    "🔮 AI Churn Predictor & Simulator",
    "🤖 Model Leaderboard & XAI",
    "💼 Retention ROI & Playbook",
]

PREDICTOR = "🔮 AI Churn Predictor & Simulator"


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, f"App raised on load: {at.exception}"
    return at


def _switch(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"App raised on page '{page}': {at.exception}"
    return at


def _metric_map(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def _as_percent(value: str) -> float:
    """Streamlit renders metric values as display strings, e.g. '48.0%'."""
    return float(str(value).strip().rstrip("%"))


# --------------------------------------------------------------------------
# All six pages render without error
# --------------------------------------------------------------------------
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _switch(_run(), page)
    assert len(at.markdown) > 0


# --------------------------------------------------------------------------
# Single prediction + What-If simulator
# --------------------------------------------------------------------------
def test_predictor_page_renders_prediction_and_simulator():
    at = _switch(_run(), PREDICTOR)
    assert at.metric, "predictor page should render metrics"
    labels = " ".join(m.label for m in at.metric)
    assert "Original Churn Risk" in labels
    assert "Simulated Churn Risk" in labels


def test_simulator_no_intervention_returns_original_probability():
    """With every lever untouched, simulated risk == original risk."""
    metrics = _metric_map(_switch(_run(), PREDICTOR))
    original = metrics["Original Churn Risk"]
    simulated = metrics["Simulated Churn Risk"]
    assert original == simulated, (
        f"Baseline simulator should be a no-op: {original} != {simulated}"
    )


def test_simulator_contract_upgrade_does_not_increase_risk():
    at = _switch(_run(), PREDICTOR)
    original = _metric_map(at)["Original Churn Risk"]
    at.selectbox(key="sim_contract").set_value("Two year").run()
    assert not at.exception, at.exception
    simulated = _metric_map(at)["Simulated Churn Risk"]
    assert _as_percent(simulated) <= _as_percent(original)


# --------------------------------------------------------------------------
# Batch scoring + CSV upload
# --------------------------------------------------------------------------
def test_batch_sample_scoring_runs():
    at = _switch(_run(), PREDICTOR)
    buttons = [b for b in at.button if "Score Sample Batch" in b.label]
    assert buttons, "sample batch button should exist"
    buttons[0].click().run()
    assert not at.exception, f"batch scoring failed: {at.exception}"
    assert "Batch Scored" in " ".join(m.label for m in at.metric)


def test_csv_upload_scores_customers():
    at = _switch(_run(), PREDICTOR)
    sample = load_data().drop(columns=["Churn"]).head(12)
    assert at.file_uploader, "CSV uploader should exist"
    at.file_uploader[0].upload(
        "batch.csv", sample.to_csv(index=False).encode("utf-8"))
    at.run()
    assert not at.exception, f"CSV upload failed: {at.exception}"
    assert "Batch Scored" in " ".join(m.label for m in at.metric)


def test_csv_upload_with_missing_total_charges_is_imputed():
    """The old crash path: a CSV with blank TotalCharges must still score."""
    at = _switch(_run(), PREDICTOR)
    sample = load_data().drop(columns=["Churn"]).head(10).copy()
    sample["TotalCharges"] = np.nan
    at.file_uploader[0].upload(
        "batch_nan.csv", sample.to_csv(index=False).encode("utf-8"))
    at.run()
    assert not at.exception, f"NaN TotalCharges upload failed: {at.exception}"
    assert "Batch Scored" in " ".join(m.label for m in at.metric)


def test_csv_upload_with_invalid_category_reports_error():
    """An out-of-domain category must surface an error, not a silent score."""
    at = _switch(_run(), PREDICTOR)
    sample = load_data().drop(columns=["Churn"]).head(5).copy()
    sample.loc[0, "Contract"] = "Three year"
    at.file_uploader[0].upload(
        "batch_bad.csv", sample.to_csv(index=False).encode("utf-8"))
    at.run()
    assert not at.exception, "invalid category should not crash the app"
    errors = " ".join(e.value for e in at.error)
    assert "Contract" in errors, (
        f"expected a Contract validation error, got: {errors}"
    )


# --------------------------------------------------------------------------
# Backend contract used by the UI
# --------------------------------------------------------------------------
def test_ui_payload_schema_matches_backend():
    assert len(REQUIRED_COLS) == 19
    model = ChurnModel.load()
    assert model.best_model.n_features_in_ == 30
    assert model.total_charges_median > 0
