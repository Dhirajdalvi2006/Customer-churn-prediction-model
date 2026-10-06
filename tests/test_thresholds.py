"""Phase 3C tests: threshold & calibration analysis on the frozen champion.

These tests never fit or modify a model. They verify the analysis is
deterministic, correctly encodes the positive class, and that running it leaves
the production artifacts untouched.
"""

import os
import hashlib
import json

import numpy as np
import pytest

from backend.thresholds import (
    EXPERIMENT_DIR,
    CHAMPION_PATH,
    THRESHOLDS,
    POSITIVE_LABEL,
    to_binary,
    load_frozen_champion,
    get_frozen_champion_predictions,
    threshold_metrics,
    threshold_sweep,
    curve_metrics,
    candidate_thresholds,
    calibration_analysis,
    run_threshold_analysis,
    save_analysis_artifacts,
    save_analysis_charts,
)

PRODUCTION_ARTIFACTS = [
    os.path.join("models", "best_model.pkl"),
    os.path.join("models", "preprocessor.pkl"),
    os.path.join("models", "results.pkl"),
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_to_binary_encodes_positive_label_explicitly():
    """The positive class must be 1 regardless of the order labels appear in."""
    forward = to_binary(["Yes", "No", "Yes"])
    reversed_ = to_binary(["No", "Yes", "No"])

    assert list(forward) == [1, 0, 1]
    # Same mapping in both label orders -> not relying on label ordering.
    assert list(reversed_) == [0, 1, 0]
    assert POSITIVE_LABEL == "Yes"


def test_frozen_champion_is_loaded_not_refitted():
    """The champion is read from disk, and its class order is asserted."""
    champion = load_frozen_champion()
    assert list(champion.classes_) == ["No", "Yes"]
    # A fitted Pipeline exposes the final estimator; it is reused, never refit.
    assert hasattr(champion, "named_steps") and "classifier" in champion.named_steps
    assert os.path.exists(CHAMPION_PATH)


def test_predictions_use_the_phase3b_untouched_test_split():
    """Same split call as Phase 3B, so these are the same untouched rows."""
    from sklearn.model_selection import train_test_split
    from backend.model import load_data, TARGET, DROP_COLS

    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    data = get_frozen_champion_predictions()

    assert data["n_total"] == len(X_test) == len(y_test)
    # Same positive count as the reconstructed Phase 3B test fold.
    assert data["n_actual_positive"] == int((y_test == "Yes").sum())
    assert data["n_actual_positive"] + data["n_actual_negative"] == data["n_total"]
    # Probabilities are the positive-class column, in [0, 1].
    assert data["y_prob"].shape == (len(X_test),)
    assert np.all((data["y_prob"] >= 0) & (data["y_prob"] <= 1))



def test_threshold_metrics_confusion_matrix_is_positional():
    """tp/fp/tn/fn are read positionally and must sum back to the sample count."""
    y_true = ["Yes", "Yes", "No", "No", "No", "Yes"]
    y_prob = [0.90, 0.40, 0.30, 0.80, 0.10, 0.70]

    m = threshold_metrics(y_true, y_prob, 0.5)

    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y_true)
    assert m["n_predicted_positive"] == m["tp"] + m["fp"] == 3
    assert m["n_predicted_negative"] == m["tn"] + m["fn"] == 3
    # p >= threshold is inclusive at the boundary.
    assert threshold_metrics(y_true, y_prob, 0.70)["tp"] == 2
    # Precision/recall follow from the confusion matrix.
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_threshold_sweep_is_deterministic_and_monotone():
    """The sweep is ordered, reproducible, and recall never rises with threshold."""
    data = get_frozen_champion_predictions()
    y_true, y_prob = data["y_true_labels"], data["y_prob"]

    sweep_a = threshold_sweep(y_true, y_prob)
    sweep_b = threshold_sweep(y_true, y_prob)

    assert [r["threshold"] for r in sweep_a] == THRESHOLDS
    assert sweep_a == sweep_b  # exact equality, not approx

    recalls = [r["recall"] for r in sweep_a]
    assert all(a >= b for a, b in zip(recalls, recalls[1:])), \
        "recall must be non-increasing as the threshold rises"
    # More permissive thresholds never predict fewer positives.
    counts = [r["n_predicted_positive"] for r in sweep_a]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_candidate_thresholds_respect_precision_floors():
    """Candidate thresholds are grid-restricted, floor-respecting and deterministic."""
    sweep = threshold_sweep(
        ["Yes"] * 5 + ["No"] * 95,
        [0.9, 0.8, 0.7, 0.65, 0.6] + [0.05] * 95,
    )
    candidates = candidate_thresholds(sweep)

    assert candidates["analysis_only"] is True
    assert candidates["no_threshold_adopted"] is True

    # max-F1 candidate must actually be the F1 argmax over the grid.
    best = max(sweep, key=lambda r: (r["f1"], -r["threshold"]))
    assert candidates["f1_maximizing"]["threshold"] == best["threshold"]

    grid = set(THRESHOLDS)
    for key in ("recall_at_precision_0_60", "recall_at_precision_0_65",
                "recall_at_precision_0_70"):
        c = candidates[key]
        if c["found"]:
            assert c["threshold"] in grid, "candidates must come from the Part A grid"
            assert c["precision"] >= c["min_precision"]
        else:
            # Infeasible floors report why, rather than silently picking something.
            assert "reason" in c and c["min_precision"] in (0.60, 0.65, 0.70)

    assert candidate_thresholds(sweep) == candidates  # deterministic


def test_curve_metrics_and_calibration_bins_are_consistent():
    """PR/ROC scalars match the points, and calibration bins partition the test set."""
    data = get_frozen_champion_predictions()
    y_true, y_prob = data["y_true_labels"], data["y_prob"]

    curves = curve_metrics(y_true, y_prob)
    assert 0.0 <= curves["pr_auc"] <= 1.0
    assert 0.5 <= curves["roc_auc"] <= 1.0
    assert 0.0 <= curves["brier_score"] <= 1.0
    # Every PR point is a valid (recall, precision) pair.
    assert all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
               for p in curves["precision_recall_curve"])
    assert curve_metrics(y_true, y_prob) == curves  # deterministic

    cal = calibration_analysis(y_true, y_prob)
    assert cal["n_bins"] == 10
    assert "equal-width" in cal["binning"]
    # Bins partition the frozen test set: nothing dropped, nothing double-counted.
    assert sum(b["count"] for b in cal["bins"]) == data["n_total"]
    assert cal["brier_score"] == curves["brier_score"]
    # Fixed equal-width edges, ascending and contiguous.
    for b, nxt in zip(cal["bins"], cal["bins"][1:]):
        assert b["upper"] == nxt["lower"]
    for b in cal["bins"]:
        if b["count"] > 0:
            assert 0.0 <= b["observed_churn_rate"] <= 1.0
            assert b["mean_predicted_probability"] is not None
        else:
            # Empty bins carry nulls rather than fabricated numbers.
            assert b["mean_predicted_probability"] is None
            assert b["observed_churn_rate"] is None
            assert b["calibration_gap"] is None


def test_run_threshold_analysis_is_reproducible_and_analysis_only():
    """Two runs in-process agree exactly, and the result is flagged analysis-only."""
    a = run_threshold_analysis()
    b = run_threshold_analysis()

    assert a == b  # exact reproducibility
    assert a["analysis_only"] is True
    assert a["model_retrained"] is False
    assert a["champion"] == "GradientBoostingClassifier"
    assert a["split"] == {
        "test_size": 0.2,
        "random_state": 42,
        "stratified": True,
        "source": "reconstructed identically to Phase 3B (no new split)",
    }
    assert a["test_set"]["n_total"] > 0
    assert len(a["threshold_sweep"]) == len(THRESHOLDS)
    assert a["metrics_at_0_50"]["threshold"] == 0.5
    # 0.50 must be one of the audited grid points, not a special case.
    assert any(r["threshold"] == 0.5 for r in a["threshold_sweep"])



def test_analysis_writes_only_into_the_experiments_directory(tmp_path):
    """JSON/CSV/PNG artifacts land in the given dir; production files are untouched."""
    result = run_threshold_analysis()

    production_before = {
        p: _sha256(os.path.join(PROJECT_ROOT, p)) for p in PRODUCTION_ARTIFACTS
    }
    champion_before = _sha256(CHAMPION_PATH)

    target = tmp_path / "experiments"
    artifacts = save_analysis_artifacts(result, experiment_dir=str(target))
    charts = save_analysis_charts(result, experiment_dir=str(target))

    # Required JSON/CSV artifacts.
    for name in ("threshold_analysis.json", "threshold_analysis.csv",
                 "calibration_analysis.json", "pr_roc_curves.json"):
        assert name in artifacts
        assert (target / name).is_file()
    # Required charts.
    for name in ("threshold_tradeoff.png", "precision_recall_curve.png",
                 "calibration_curve.png"):
        assert name in charts
        assert (target / name).is_file()
        assert (target / name).stat().st_size > 0

    # Every written path stayed inside the target directory.
    for path in list(artifacts.values()) + list(charts.values()):
        assert os.path.dirname(os.path.abspath(path)) == str(target.resolve())

    # Production artifacts and the frozen champion are byte-for-byte unchanged.
    for p, before in production_before.items():
        assert _sha256(os.path.join(PROJECT_ROOT, p)) == before, f"{p} was modified"
    assert _sha256(CHAMPION_PATH) == champion_before, "champion.pkl was modified"

    # The written JSON must round-trip to the same numbers.
    with open(target / "threshold_analysis.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["champion"] == result["champion"]
    assert payload["model_retrained"] is False
    assert len(payload["threshold_sweep"]) == len(THRESHOLDS)


def test_charts_do_not_accumulate_open_figures(tmp_path):
    """Figures are closed after saving, so repeat calls stay leak-free.

    Charts are written to pytest's tmp_path so running the suite never touches
    the real models/experiments/ directory.
    """
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    result = run_threshold_analysis()
    plt.close("all")

    expected = ("threshold_tradeoff.png", "precision_recall_curve.png",
                "calibration_curve.png")

    for _ in range(2):
        charts = save_analysis_charts(result, experiment_dir=str(tmp_path))

        for name in expected:
            assert name in charts
            assert (tmp_path / name).is_file()
            assert (tmp_path / name).stat().st_size > 0

        # No figures left open after each call, so the second call cannot
        # accumulate on top of the first.
        assert plt.get_fignums() == [], "figures were left open after save_analysis_charts"

    plt.close("all")
