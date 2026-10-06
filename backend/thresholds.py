"""Phase 3C: threshold and probability-calibration ANALYSIS for the frozen
Phase 3B champion.

This module is deliberately analysis-only:

* It loads the already-fitted champion from ``models/experiments/champion.pkl``.
* It never fits, tunes, calibrates or re-selects any model.
* It reuses the Phase 3B train/test split verbatim (``test_size=0.2``,
  ``random_state=42``, stratified) so the "untouched final test set" analysed
  here is exactly the one Phase 3B reported on. No new split is created.
* It reads production artifacts read-only and never writes to them.

Every function is pure and deterministic: given the same champion and the same
frozen test predictions it returns identical numbers on every call and in every
process.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from backend.model import load_data, TARGET, DROP_COLS

EXPERIMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "experiments",
)
CHAMPION_PATH = os.path.join(EXPERIMENT_DIR, "champion.pkl")

# Phase 3B frozen configuration. These must match backend/experiment.py exactly;
# they are reproduced here (not imported) so this analysis stays decoupled from
# the experiment and cannot silently drift if the experiment is ever retuned.
TEST_SIZE = 0.2
RANDOM_STATE = 42
POSITIVE_LABEL = "Yes"

# Part A sweep grid, in ascending order. The order is part of the contract so
# results are deterministic and diffable between runs.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]

N_CALIBRATION_BINS = 10
CURVE_DECIMALS = 12

def to_binary(y_true, positive_label=POSITIVE_LABEL):
    """Encode labels as 1/0 with the positive class mapped to 1.

    Kept explicit (rather than relying on label ordering) so the positive class
    is correct regardless of the order the labels happen to appear in the data.
    """
    return (pd.Series(np.asarray(y_true)).astype(str) == positive_label).astype(int).to_numpy()


def load_frozen_champion(path=None):
    """Load the already-fitted Phase 3B champion pipeline from disk."""
    return joblib.load(path or CHAMPION_PATH)


def get_frozen_champion_predictions(champion_path=None):
    """Return the frozen champion's predictions on the Phase 3B test set.

    The split is reconstructed with the exact same call Phase 3B used, so these
    are the same untouched final-test rows that produced the Phase 3B metrics --
    no retraining and no new split.
    """
    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]

    # Identical to backend/experiment.py::run_experiment.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    champion = load_frozen_champion(champion_path)
    # Column 1 is the "Yes" column: classes_ is sorted and the only labels
    # present are {"No", "Yes"}, so classes_[1] == "Yes". Assert rather than
    # assume, so a label-ordering change can never silently invert the analysis.
    assert list(champion.classes_) == ["No", "Yes"], (
        f"unexpected class order {list(champion.classes_)}: "
        "column 1 may not be the positive class"
    )
    y_prob = champion.predict_proba(X_test)[:, 1]

    y_true = to_binary(y_test)
    return {
        "champion": champion,
        "y_true_labels": np.asarray(y_test),
        "y_true": y_true,
        "y_prob": np.asarray(y_prob, dtype=float),
        "n_total": int(len(y_test)),
        "n_actual_positive": int(y_true.sum()),
        "n_actual_negative": int((1 - y_true).sum()),
    }



def threshold_metrics(y_true, y_prob, threshold, positive_label=POSITIVE_LABEL):
    """Metrics and confusion-matrix components at one decision threshold.

    A sample is predicted positive when ``p >= threshold``.
    """
    y_true = to_binary(y_true, positive_label)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion_matrix with labels=[0, 1] gives rows = actual, cols = predicted,
    # so tn, fp, fn, tp are read out positionally and unambiguously.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_predicted_positive": int(tp + fp),
        "n_predicted_negative": int(tn + fn),
    }


def threshold_sweep(y_true, y_prob, thresholds=None, positive_label=POSITIVE_LABEL):
    """Part A: evaluate every threshold, preserving ascending order."""
    thresholds = list(THRESHOLDS if thresholds is None else thresholds)
    return [threshold_metrics(y_true, y_prob, t, positive_label) for t in thresholds]


def curve_metrics(y_true, y_prob, positive_label=POSITIVE_LABEL):
    """Part B: PR/ROC curve data and their scalar summaries."""
    y_true = to_binary(y_true, positive_label)
    y_prob = np.asarray(y_prob, dtype=float)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # Curve points are rounded: ample precision for a curve plot, and it keeps the
    # JSON byte-stable across runs without hiding real differences.
    d = CURVE_DECIMALS
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_recall_curve": [
            [round(float(r), d), round(float(p), d)] for p, r in zip(precision, recall)
        ],
        "roc_curve": [
            [round(float(f), d), round(float(t), d)] for f, t in zip(fpr, tpr)
        ],
        "n_roc_threshold_points": int(len(roc_thresholds)),
    }



def candidate_thresholds(sweep, current_threshold=0.5):
    """Part C: transparent, purely mathematical candidate thresholds.

    Candidates are restricted to the Part A grid. Ties are broken by the *lower*
    threshold (predicting positive less often). This is a documented tie-break
    rule, not a judgement call, and it makes the result deterministic.
    """
    if not sweep:
        return {}

    def best_recall(rows, min_precision):
        feasible = [r for r in rows if r["precision"] >= min_precision]
        if not feasible:
            best_p = max(rows, key=lambda r: r["precision"])
            return {
                "found": False,
                "min_precision": min_precision,
                "reason": (
                    f"No threshold on the evaluated grid reaches precision >= "
                    f"{min_precision:.2f}; the maximum observed precision is "
                    f"{best_p['precision']:.6f} at threshold {best_p['threshold']:.2f}."
                ),
            }
        best = max(feasible, key=lambda r: (r["recall"], r["f1"], -r["threshold"]))
        return {
            "found": True,
            "min_precision": min_precision,
            "threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        }

    f1_best = max(sweep, key=lambda r: (r["f1"], -r["threshold"]))
    at_current = next(
        (r for r in sweep if abs(r["threshold"] - current_threshold) < 1e-12), None
    )
    return {
        "analysis_only": True,
        "no_threshold_adopted": True,
        "tie_break_rule": "On ties the lower threshold is chosen.",
        "candidate_scope": "restricted to the Part A threshold grid",
        "current_threshold": current_threshold,
        "current_threshold_metrics": at_current,
        "f1_maximizing": {
            "threshold": f1_best["threshold"],
            "f1": f1_best["f1"],
            "precision": f1_best["precision"],
            "recall": f1_best["recall"],
            "accuracy": f1_best["accuracy"],
        },
        "recall_at_precision_0_60": best_recall(sweep, 0.60),
        "recall_at_precision_0_65": best_recall(sweep, 0.65),
        "recall_at_precision_0_70": best_recall(sweep, 0.70),
    }


def calibration_analysis(y_true, y_prob, n_bins=N_CALIBRATION_BINS, positive_label=POSITIVE_LABEL):
    """Part D: reliability curve over fixed equal-width probability bins.

    Bins are fixed edges on [0, 1] rather than quantiles, so the bin membership
    of each prediction never changes between runs.
    """
    y_true = to_binary(y_true, positive_label)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against the interior edges; the outer clip keeps p == 1.0 counted
    # in the final bin instead of dropping it.
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    d = CURVE_DECIMALS

    bins = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        mean_predicted = float(y_prob[mask].mean()) if count else None
        observed = float(y_true[mask].mean()) if count else None
        bins.append({
            "bin": b,
            "lower": round(float(edges[b]), d),
            "upper": round(float(edges[b + 1]), d),
            "count": count,
            "mean_predicted_probability": None if mean_predicted is None else round(mean_predicted, d),
            "observed_churn_rate": None if observed is None else round(observed, d),
            # Positive gap => the model under-states churn risk in this bin.
            "calibration_gap": (
                None if (mean_predicted is None or observed is None)
                else round(mean_predicted - observed, d)
            ),
        })

    populated = [b for b in bins if b["count"] > 0]
    return {
        "n_bins": n_bins,
        "binning": "fixed equal-width bins on [0, 1]",
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "brier_score_lower_bound": 0.0,
        "brier_score_upper_bound": 1.0,
        "overall_mean_predicted_probability": round(float(y_prob.mean()), d),
        "overall_observed_churn_rate": round(float(y_true.mean()), d),
        "mean_absolute_calibration_gap": (
            round(float(np.mean([abs(b["calibration_gap"]) for b in populated])), d)
            if populated else None
        ),
        "bins": bins,
    }




def run_threshold_analysis(champion_path=None, thresholds=None, n_bins=N_CALIBRATION_BINS):
    """Run Parts A-D against the frozen champion. Analysis only, no fitting."""
    data = get_frozen_champion_predictions(champion_path)
    y_true, y_prob = data["y_true_labels"], data["y_prob"]

    sweep = threshold_sweep(y_true, y_prob, thresholds)
    at_half = next(r for r in sweep if abs(r["threshold"] - 0.5) < 1e-12)

    return {
        "champion": type(data["champion"].named_steps["classifier"]).__name__,
        "analysis_only": True,
        "model_retrained": False,
        "positive_label": POSITIVE_LABEL,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "source": "reconstructed identically to Phase 3B (no new split)",
        },
        "test_set": {
            "n_total": data["n_total"],
            "n_actual_positive": data["n_actual_positive"],
            "n_actual_negative": data["n_actual_negative"],
            "observed_positive_rate": round(data["n_actual_positive"] / data["n_total"], CURVE_DECIMALS),
        },
        "threshold_sweep": sweep,
        "metrics_at_0_50": at_half,
        "curves": curve_metrics(y_true, y_prob),
        "candidates": candidate_thresholds(sweep),
        "calibration": calibration_analysis(y_true, y_prob, n_bins),
    }


def save_analysis_artifacts(result, experiment_dir=None):
    """Part E: write the analysis artifacts. Production artifacts are never touched."""
    experiment_dir = experiment_dir or EXPERIMENT_DIR
    os.makedirs(experiment_dir, exist_ok=True)

    curves = result["curves"]
    # The full PR/ROC point clouds go to their own file; only the scalar summary
    # stays inline so calibration_analysis.json remains readable.
    curve_summary = {k: v for k, v in curves.items()
                     if k not in ("precision_recall_curve", "roc_curve")}

    payloads = {
        "threshold_analysis.json": {
            "champion": result["champion"],
            "analysis_only": True,
            "model_retrained": False,
            "positive_label": result["positive_label"],
            "split": result["split"],
            "test_set": result["test_set"],
            "roc_auc": curves["roc_auc"],
            "pr_auc": curves["pr_auc"],
            "metrics_at_0_50": result["metrics_at_0_50"],
            "threshold_sweep": result["threshold_sweep"],
            "candidates": result["candidates"],
        },
        "calibration_analysis.json": {
            "champion": result["champion"],
            "analysis_only": True,
            "model_retrained": False,
            "positive_label": result["positive_label"],
            "split": result["split"],
            "test_set": result["test_set"],
            "curve_summary": curve_summary,
            "calibration": result["calibration"],
        },
        "pr_roc_curves.json": {
            "champion": result["champion"],
            "roc_auc": curves["roc_auc"],
            "pr_auc": curves["pr_auc"],
            "precision_recall_curve": curves["precision_recall_curve"],
            "roc_curve": curves["roc_curve"],
        },
    }

    paths = {}
    for name, payload in payloads.items():
        path = os.path.join(experiment_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        paths[name] = path

    csv_path = os.path.join(experiment_dir, "threshold_analysis.csv")
    pd.DataFrame(result["threshold_sweep"]).to_csv(csv_path, index=False)
    paths["threshold_analysis.csv"] = csv_path
    return paths






def save_analysis_charts(result, experiment_dir=None):
    """Part E charts. Headless-safe, and only ever writes into experiment_dir.

    matplotlib is imported lazily and pinned to the Agg backend here so this
    module can be imported by the app/tests on a machine with no display without
    changing global plotting state on import.
    """
    experiment_dir = experiment_dir or EXPERIMENT_DIR
    os.makedirs(experiment_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = result["threshold_sweep"]
    thresholds = [r["threshold"] for r in sweep]
    written = {}

    def _save(fig, name):
        path = os.path.join(experiment_dir, name)
        try:
            fig.savefig(path, dpi=120, bbox_inches="tight")
            written[name] = path
        finally:
            # Always close, so repeated runs in one process cannot accumulate
            # figures (and matplotlib's warning limit is never hit).
            plt.close(fig)
        return path

    # 1. Threshold trade-off
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        ax.plot(thresholds, [r[key] for r in sweep], "o-", label=label)
    ax.axvline(0.5, ls="--", c="grey", lw=1, label="Current threshold (0.50)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Frozen champion: threshold trade-off (untouched test set)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "threshold_tradeoff.png")

    # 2. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    pr = result["curves"]["precision_recall_curve"]
    ax.plot([p[0] for p in pr], [p[1] for p in pr], lw=2, label="Champion")
    f1_best = result["candidates"]["f1_maximizing"]
    ax.scatter([f1_best["recall"]], [f1_best["precision"]], c="red", zorder=3,
               label=f"max-F1 @ {f1_best['threshold']:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve (AP = {result['curves']['pr_auc']:.4f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "precision_recall_curve.png")

    # 3. Calibration curve
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = [b for b in result["calibration"]["bins"] if b["count"] > 0]
    ax.plot([b["mean_predicted_probability"] for b in bins],
            [b["observed_churn_rate"] for b in bins], "o-", label="Champion")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title(f"Reliability curve (Brier = {result['calibration']['brier_score']:.4f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "calibration_curve.png")

    return written


if __name__ == "__main__":
    # Analysis only: the champion is loaded from disk and never refit.
    analysis = run_threshold_analysis()
    artifacts = save_analysis_artifacts(analysis)
    charts = save_analysis_charts(analysis)

    curves = analysis["curves"]
    print("Phase 3C threshold & calibration analysis complete (analysis only, no retraining).")
    print(f"  champion      : {analysis['champion']} (loaded from {CHAMPION_PATH})")
    print(f"  test set      : {analysis['test_set']['n_total']} rows, "
          f"{analysis['test_set']['n_actual_positive']} positive "
          f"({analysis['test_set']['observed_positive_rate']:.2%})")
    print(f"  ROC-AUC       : {curves['roc_auc']:.6f}")
    print(f"  PR-AUC        : {curves['pr_auc']:.6f}")
    print(f"  Brier score   : {analysis['calibration']['brier_score']:.6f}")
    for key, label in (("f1_maximizing", "max-F1 threshold"),
                       ("recall_at_precision_0_60", "recall @ precision>=0.60"),
                       ("recall_at_precision_0_65", "recall @ precision>=0.65"),
                       ("recall_at_precision_0_70", "recall @ precision>=0.70")):
        c = analysis["candidates"][key]
        if c.get("found", key == "f1_maximizing"):
            print(f"  {label:<30}: {c['threshold']:.2f} "
                  f"(P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f})")
        else:
            print(f"  {label:<30}: none feasible -> {c['reason']}")
    print("  artifacts     :")
    for name in sorted(list(artifacts) + list(charts)):
        print(f"    - {name}")
    print("  NOTE: no threshold adopted. These are candidates for Phase 3D review only.")
