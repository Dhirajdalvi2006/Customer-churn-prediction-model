"""Bundle-backed production inference service (Phase 4 Step 10).

The single production entry point for churn scoring. It resolves its model
through :meth:`BundleLoader.get_active_bundle`, which reads
``models/active_bundle.json`` -- the *only* mechanism by which a bundle
version becomes active. No version, artifact filename, model type or threshold
is hard-coded here; every such value is read from the resolved bundle's own
manifest and validated fitted artifact.

Deliberate design points
------------------------
* **No silent legacy fallback.** If the active pointer, manifest, artifact hash
  or fitted estimator cannot be resolved, this raises
  :class:`PredictionServiceError`. It never falls back to ``best_model.pkl``;
  doing so would silently bypass the bundle architecture and hide a broken
  deployment.
* **One contract for every entry point.** Single prediction, batch scoring and
  the What-If simulator all reach the model through the bundle's shared
  :class:`~backend.input_contract.ContractSpec`, so a customer's score cannot
  depend on which entry point scored them or on the other rows in a batch.
* **Metadata-driven semantics.** The positive class, its column index and the
  decision threshold are resolved from bundle metadata at load time. There is
  no ``predict_proba(...)[:, 1]`` and no literal ``0.5`` in this module.
* **Legacy artifacts untouched.** ``models/best_model.pkl`` and
  ``models/preprocessor.pkl`` are *not* read by any inference path; they remain
  the rollback reference. The only legacy file still consulted is
  ``results.pkl``, a training-time *evaluation* sidecar used solely for the
  leaderboard table, loaded lazily and never involved in producing a score.
"""

import json
import os
import threading

import joblib
import numpy as np
import pandas as pd

from backend.input_contract import ContractError
from backend.model import MODEL_DIR
from backend.model_bundle import BundleInferenceError, BundleLoader

__all__ = [
    "PredictionServiceError",
    "BundlePredictionService",
    "get_prediction_service",
    "active_pointer_version",
]


class PredictionServiceError(RuntimeError):
    """Application-level failure of the production inference path.

    Deliberately message-only for end users: it names what failed in
    operational terms (pointer, integrity, contract) so the UI can surface an
    actionable message without leaking an internal stack trace.
    """


def _resolution_message(exc: Exception) -> str:
    """Render a resolution failure as an operational message, not a traceback."""
    text = (str(exc) or exc.__class__.__name__).strip()
    lowered = text.lower()
    if isinstance(exc, FileNotFoundError):
        return (
            "The active model bundle could not be found. Expected "
            "'models/active_bundle.json' and the bundle it points to. "
            "Inference is unavailable until this is restored. "
            f"(Details: {text})"
        )
    if ("hash" in lowered or "sha" in lowered
            or "integrity" in lowered or "checksum" in lowered):
        return (
            "The active model bundle failed its integrity check: its artifacts "
            "no longer match the manifest describing them. Inference is stopped "
            "to avoid scoring with a corrupted model. (Details: {0})".format(text)
        )
    if "class" in lowered and "index" in lowered:
        return (
            "The active model bundle declares a churn class that disagrees with "
            f"its fitted model. (Details: {text})"
        )
    return (
        "The active model bundle could not be loaded. Inference is unavailable "
        "rather than silently falling back to a legacy artifact. "
        f"(Details: {text})"
    )


class BundlePredictionService:
    """Production inference over whichever bundle ``active_bundle.json`` selects.

    Parameters
    ----------
    models_dir:
        Directory holding ``active_bundle.json`` and ``bundles/``. Defaults to
        the project's ``models/`` directory. It is handed straight to
        :meth:`BundleLoader.get_active_bundle`; the service never inspects it
        to decide which version to use.
    """

    # Presentation labels for the two churn outcomes. These name the *meaning*
    # of the validated semantic class, not a class index: which fitted column is
    # positive is decided from metadata at load time.
    POSITIVE_LABEL = "Yes"
    NEGATIVE_LABEL = "No"

    def __init__(self, models_dir: str = MODEL_DIR):
        self.models_dir = os.path.abspath(models_dir)
        # Resolve eagerly so a broken deployment fails at construction with a
        # clear, catchable error rather than midway through a prediction.
        self.bundle = self._load_active_bundle()
        self._evaluation_results = None

    def _load_active_bundle(self):
        """Resolve the active bundle through the active pointer, safely."""
        try:
            return BundleLoader.get_active_bundle(self.models_dir)
        except (BundleInferenceError, ContractError) as exc:
            raise PredictionServiceError(_resolution_message(exc)) from None
        except Exception as exc:
            # Includes a missing/invalid pointer file, a missing bundle
            # directory, an unreadable manifest and a hash mismatch. All fail
            # explicitly: there is deliberately no legacy fallback.
            raise PredictionServiceError(_resolution_message(exc)) from None

    # -- metadata (all bundle-derived, never hard-coded) -------------------

    @property
    def bundle_version(self) -> str:
        return str(self.bundle.manifest["bundle_version"])

    @property
    def model_format(self) -> str:
        return str(self.bundle.manifest["model_format"])

    @property
    def model_type(self) -> str:
        """Estimator class name, read from the fitted artifact."""
        return type(self.bundle.estimator).__name__

    @property
    def decision_threshold(self) -> float:
        """The bundle's own decision threshold; no literal appears here."""
        return float(self.bundle.decision_threshold)

    @property
    def positive_class(self):
        return self.bundle.positive_class

    @property
    def positive_class_index(self) -> int:
        return int(self.bundle.positive_class_index)

    @property
    def transformed_feature_count(self) -> int:
        feats = self.bundle.feature_representation()
        return int(feats["transformed_feature_count"])

    @property
    def best_model_name(self) -> str:
        """Human-readable active model name derived from bundle metadata."""
        raw = str(self.bundle.manifest.get("model_type") or self.model_type)
        parts = []
        for i, ch in enumerate(raw):
            if i and ch.isupper() and not raw[i - 1].isupper():
                parts.append(" ")
            parts.append(ch)
        return "".join(parts)

    def metadata(self) -> dict:
        """Full active-model metadata for the application/UI."""
        return {
            "bundle_version": self.bundle_version,
            "model_type": self.model_type,
            "model_type_label": self.best_model_name,
            "model_format": self.model_format,
            "decision_threshold": self.decision_threshold,
            "positive_class": str(self.positive_class),
            "positive_class_index": self.positive_class_index,
            "positive_semantic_label": self.bundle.positive_semantic_label,
            "transformed_feature_count": self.transformed_feature_count,
            "total_charges_median": float(
                self.bundle.contract.total_charges_statistic
            ),
        }

    # -- inference ---------------------------------------------------------

    def predict_single(self, customer_data: dict) -> dict:
        """Score one customer through the active bundle.

        Returns the same keys the legacy ``ChurnModel.predict_single`` exposed
        so the UI is unchanged, but every value is now bundle-derived.
        """
        negative, positive, labels = self._predict(customer_data)
        return {
            "prediction": str(labels[0]),
            "churn_probability": round(float(positive[0]) * 100, 2),
            "no_churn_probability": round(float(negative[0]) * 100, 2),
        }

    def predict_batch(self, customers_df: pd.DataFrame):
        """Score many customers through exactly the single-customer path.

        Returns ``(labels, positive_probabilities)`` with probabilities in
        ``[0, 1]``. Same contract, preprocessing, class semantics and threshold
        as :meth:`predict_single`; no batch-derived statistics.
        """
        if len(customers_df) == 0:
            return np.array([], dtype=object), np.array([], dtype=float)
        _, positive, labels = self._predict(customers_df)
        return labels, positive

    def _predict(self, customer_data):
        """The one scoring path: contract -> bundle -> metadata semantics.

        Runs the bundle's fitted pipeline exactly once and derives both the
        probabilities and the labels from that single result, so the score a
        customer sees and the label they are shown can never come from two
        differing passes.

        The positive class index and the threshold are read from the resolved
        bundle's metadata (already validated against the fitted ``classes_`` at
        load time) -- no index or threshold literal appears here.

        A :class:`ContractError` means the *caller's* payload is invalid, not
        that the deployment is broken, so it is re-raised as a service error
        carrying the contract's own message (the UI shows which field failed).
        """
        try:
            negative, positive = self.bundle.predict_proba_both(customer_data)
        except ContractError as exc:
            raise PredictionServiceError(str(exc)) from None
        labels = np.where(
            positive >= self.bundle.decision_threshold,
            self._positive_label(),
            self._negative_label(),
        )
        return negative, positive, labels

    def _positive_label(self) -> str:
        """The churn label as the application presents it.

        v1's fitted ``classes_`` are integers ``[0, 1]``, so the raw class token
        is ``"1"``. The UI (and the historical contract) has always spoken
        ``"Yes"``/``"No"`` for churn, so the *semantic* label -- already derived
        and validated from ``classes_`` when the bundle loaded -- is what gets
        returned. This keeps the label meaning identical for an integer-encoded
        bundle and a string-encoded one without hardcoding either.
        """
        return self.bundle.positive_semantic_label

    def _negative_label(self) -> str:
        """The counterpart label for the non-churn class."""
        if self.bundle.positive_semantic_label == self.POSITIVE_LABEL:
            return self.NEGATIVE_LABEL
        return self.POSITIVE_LABEL

    def explainable_model(self):
        """The object the XAI layer should explain.

        Returns the active :class:`~backend.model_bundle.ModelBundle` itself, so
        SHAP is computed in the same validated space that produced the score,
        from the same fitted estimator, with no duplicated preprocessing.
        """
        return self.bundle

    # -- reporting only: evaluation sidecar, never inference ---------------

    def evaluation_results(self) -> dict:
        """Training-time evaluation metrics for the leaderboard table.

        Explicitly *not* part of inference: no prediction method calls it, and
        it is never consulted when resolving the active bundle.
        """
        if self._evaluation_results is None:
            try:
                self._evaluation_results = joblib.load(
                    os.path.join(self.models_dir, "results.pkl")
                )
            except Exception:
                self._evaluation_results = {}
        return self._evaluation_results

    @property
    def results(self) -> dict:
        return self.evaluation_results()

    def get_feature_importance(self, top_n=None):
        """Importance of the *active* estimator, sourced from the bundle.

        Coefficients/importances come from the bundle's own fitted estimator
        and names from its own fitted preprocessor, so this stays correct for
        either bundle format and for either one-hot ``drop`` setting.
        """
        estimator = self.bundle.estimator
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)[0])
        else:
            return None

        names = [str(n) for n in self.bundle.transformer.get_feature_names_out()]
        if len(names) != len(values):
            # Never display importances whose labels might be misaligned.
            return None
        frame = pd.DataFrame({"Feature": names, "Importance": values})
        frame = frame.sort_values("Importance", ascending=False).reset_index(drop=True)
        return frame.head(int(top_n)) if top_n else frame


# ---------------------------------------------------------------------------
# Process-level cache
# ---------------------------------------------------------------------------
# Streamlit re-executes the page script on every widget interaction, so the
# bundle and its verified artifacts are unpickled once per process. The service
# holds no mutable request state, so one instance is safe to share across
# sessions. Guarded by a lock so concurrent sessions cannot double-load.
_SERVICE_CACHE = {}
_SERVICE_LOCK = threading.Lock()


def get_prediction_service(models_dir: str = MODEL_DIR) -> BundlePredictionService:
    """Return the process-wide production inference service.

    Raises :class:`PredictionServiceError` if the active bundle cannot be
    resolved. It never returns a legacy-artifact-backed substitute.
    """
    key = os.path.abspath(models_dir)
    cached = _SERVICE_CACHE.get(key)
    if cached is None:
        with _SERVICE_LOCK:
            cached = _SERVICE_CACHE.get(key)
            if cached is None:
                cached = BundlePredictionService(key)
                _SERVICE_CACHE[key] = cached
    return cached


def active_pointer_version(models_dir: str = MODEL_DIR):
    """Read ``active_bundle.json``'s declared version, for diagnostics/tests.

    Reporting helper only. Inference never consults it to choose a model; it
    always goes through :meth:`BundleLoader.get_active_bundle`. Returns
    ``None`` if the pointer file is absent or unreadable.
    """
    try:
        with open(os.path.join(models_dir, "active_bundle.json"), encoding="utf-8") as fh:
            return json.load(fh).get("active_version")
    except Exception:
        return None

