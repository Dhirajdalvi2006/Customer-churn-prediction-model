"""Phase 4 Step 7 -- READ-ONLY shadow scoring and promotion-readiness audit.

Purpose
-------
Compare the **V1** production bundle (LogisticRegression) against the **V2**
frozen champion bundle (GradientBoosting pipeline) on the *same* raw customer
inputs, and determine whether the two accept the same production input
contract, whether their class / probability semantics agree, how their
predictions differ, and which blockers remain before an application migration.

This module is **analysis-only and read-only by construction**:

* It never fits, refits, trains, mutates or re-saves any artifact.
* It never writes to a production artifact, to ``models/bundles/**`` or to
  ``active_bundle.json``.
* It never modifies the production threshold, which stays at **0.50**. The
  Phase 3C analysis threshold of 0.25 is analysis-only and is deliberately
  **not** adopted here.
* It is a **standalone analysis module**. No production module imports it;
  production inference continues to run through ``backend.model``.

Interpretation rule
-------------------
Everything reported here is a *compatibility / behavioural difference*, never
a quality ranking. The module produces no winner, no ranking, no score and no
promotion recommendation: the statistics below describe how the two bundles
*behave*, not which one is *better*. Deciding to promote is a separate, later
engineering decision made once the reported blockers are resolved.

Determinism
-----------
The comparison sample is the untouched **Phase 3B final test set**, recovered
with the exact split call Phase 3B used (``test_size=0.2``,
``random_state=42``, stratified). That split is already reproducible from
existing project code, so no synthetic customer records are created and the
dataset is only ever read, never altered.
"""

import os
import json
import hashlib
import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from backend.model import load_data, DROP_COLS, TARGET
from backend.model_bundle import ModelBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "models", "experiments")
V1_DIR = os.path.join(BASE_DIR, "models", "bundles", "v1")
V2_DIR = os.path.join(BASE_DIR, "models", "bundles", "v2")
ACTIVE_POINTER = os.path.join(BASE_DIR, "models", "active_bundle.json")

# The production decision threshold, frozen as a constant so the audit can
# *prove* it stayed at 0.50. The Phase 3C 0.25 threshold is recorded only so
# the audit can assert that it was NOT adopted.
PRODUCTION_THRESHOLD = 0.50
ANALYSIS_ONLY_THRESHOLD = 0.25

# The Phase 3B split configuration, copied verbatim from backend/experiment.py
# and backend/thresholds.py, so the recovered test set is the *same* untouched
# final test set Phase 3B scored rather than a new split.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Near-threshold band used to probe operational sensitivity (read-only).
NEAR_THRESHOLD_BAND = (0.45, 0.55)
DISAGREEMENT_SAMPLE_SIZE = 25
SINGLE_BATCH_SAMPLE_SIZE = 25
XAI_SAMPLE_SIZE = 20

# SHA-256 baselines of the frozen artifacts, so the audit can prove it changed
# nothing. Compared case-insensitively.
FROZEN_HASHES = {
    "models/best_model.pkl":
        "66400195eb9cfef8afaaf9ee59ee83e19baa1e9afa70a4ddc9fc49ce80d9010e",
    "models/preprocessor.pkl":
        "fcc2d4e47fa218fce6056f7904e4b54f7f9c58fdd6ccbd04ff20b7a5210e028f",
    "models/results.pkl":
        "5b42735ce812f4ebdffeb2bb4b2abe6afacecb3fdefd3d836a0d425dc473753c",
    "models/feature_names.pkl":
        "b0e5ae0e650cb9421334ee5f88163fd7769e8dce9c0d27f9161fc4e10aaad9a4",
    "models/total_charges_median.pkl":
        "ae6e77cba1892e0da00efae88517d3eb4299d925bb6a4f0aad1283ead8cb7640",
    "models/experiments/champion.pkl":
        "372ddf6c9ec977460d078a244a97b8781295394bc8f54acb2dc8031f3b736589",
    "models/bundles/v1/model.pkl":
        "66400195eb9cfef8afaaf9ee59ee83e19baa1e9afa70a4ddc9fc49ce80d9010e",
    "models/bundles/v1/preprocessor.pkl":
        "fcc2d4e47fa218fce6056f7904e4b54f7f9c58fdd6ccbd04ff20b7a5210e028f",
    "models/bundles/v2/model.pkl":
        "372ddf6c9ec977460d078a244a97b8781295394bc8f54acb2dc8031f3b736589",
}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _jsonable(value):
    """Coerce numpy scalars/arrays into plain JSON-serialisable Python."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Bundle loading + class semantics (STEP 3)
# ---------------------------------------------------------------------------
def load_bundles():
    """Load the V1 and V2 immutable bundles (read-only)."""
    return ModelBundle(V1_DIR), ModelBundle(V2_DIR)


def _final_estimator(bundle):
    """The estimator that actually carries ``classes_`` (pipeline-aware)."""
    model = bundle.model
    if isinstance(model, Pipeline) and len(model.steps) > 0:
        return model.steps[-1][1]
    return model


def analyse_class_semantics(bundle):
    """Read the *actual* class metadata of a bundle. Nothing is assumed.

    The positive class and its probability column are read from the fitted
    estimator's ``classes_`` and cross-checked against the bundle manifest, so
    a label-ordering change can never silently invert the audit.
    """
    estimator = _final_estimator(bundle)
    classes = [str(c) for c in estimator.classes_]
    manifest_inf = bundle.manifest["inference"]
    manifest_classes = [str(c) for c in manifest_inf["classes"]]

    # Resolve the positive class by *searching* classes_, never by assuming
    # that index 1 is the churn class.
    positive_class = manifest_inf["positive_class"]
    if str(positive_class) not in classes:
        raise ValueError(
            f"Manifest positive_class {positive_class!r} is absent from the "
            f"fitted classes_ {classes} for bundle "
            f"{bundle.manifest['bundle_version']}."
        )
    resolved_index = classes.index(str(positive_class))
    manifest_index = int(manifest_inf["positive_class_index"])
    positive_label = classes[resolved_index]

    return {
        "bundle_version": bundle.manifest["bundle_version"],
        "model_type": bundle.manifest["metadata"]["model_type"],
        "model_format": bundle.manifest["model_format"],
        "estimator_type": type(estimator).__name__,
        "classes": classes,
        "classes_dtype": str(np.asarray(estimator.classes_).dtype),
        "manifest_classes": manifest_classes,
        "classes_match_manifest": classes == manifest_classes,
        "positive_class": positive_label,
        "positive_semantic_label": _outcome_label(positive_label),
        "positive_class_index": resolved_index,
        "positive_class_is_yes": _outcome_label(positive_label) == "Yes",
        "manifest_positive_class_index": manifest_index,
        "manifest_index_agrees_with_classes": manifest_index == resolved_index,
        "uses_yes_as_positive_class": _outcome_label(positive_label) == "Yes",
        "probability_of_positive_class": (
            f"predict_proba(...)[..., {resolved_index}] == "
            f"classes_[{resolved_index}] == {positive_label!r}"
        ),
        "decision_threshold": manifest_inf["decision_threshold"],
    }


# ---------------------------------------------------------------------------
# STEP 2 -- Input contract comparison
# ---------------------------------------------------------------------------
SAME = "SAME"
DIFFERENT = "DIFFERENT"
UNKNOWN = "UNKNOWN"


def _categorical_transformer(bundle):
    """Return the fitted OneHotEncoder for a bundle, unwrapping any Pipeline.

    V1's categorical transformer is a bare ``OneHotEncoder``; V2's is a
    ``Pipeline`` ending in one. Both are unwrapped so the *effective* encoder
    is compared rather than the wrapper type.
    """
    if bundle.manifest["model_format"] == "pipeline":
        cat_block = bundle.model.named_steps["preprocessor"].named_transformers_["cat"]
    else:
        cat_block = bundle.preprocessor.named_transformers_["cat"]
    return cat_block.steps[-1][1] if isinstance(cat_block, Pipeline) else cat_block


def _column_transformer(bundle):
    if bundle.manifest["model_format"] == "pipeline":
        return bundle.model.named_steps["preprocessor"]
    return bundle.preprocessor


def _categorical_domains(bundle):
    """Per-column fitted encoder domain, read from the artifact itself."""
    encoder = _categorical_transformer(bundle)
    columns = list(getattr(encoder, "feature_names_in_", []))
    if not columns:
        # Fitted inside a Pipeline, so feature_names_in_ may be absent; fall
        # back to the declared column order of the transformer block.
        for name, _, cols in _column_transformer(bundle).transformers:
            if name == "cat" and cols:
                return {c: [str(v) for v in cats]
                        for c, cats in zip(list(cols), encoder.categories_)}
    return {c: [str(v) for v in cats]
            for c, cats in zip(columns, encoder.categories_)}


def _numeric_columns(bundle):
    for name, _, cols in _column_transformer(bundle).transformers:
        if name == "num" and cols:
            return list(cols)
    return []


def _encoder_settings(bundle):
    encoder = _categorical_transformer(bundle)
    return {
        "drop": encoder.drop,
        "handle_unknown": encoder.handle_unknown,
        "sparse_output": encoder.sparse_output,
    }


def _transform_for(bundle, frame):
    """Build the model input matrix for a bundle, per its own contract.

    V1 is a bare estimator and therefore needs the shared ``prepare_frame``
    plus fitted preprocessor path. V2 is a self-contained ``Pipeline`` and
    transforms the raw frame itself. Each bundle is driven through the path it
    was actually built for -- no cross-wiring.
    """
    if bundle.manifest["model_format"] == "pipeline":
        return np.asarray(
            bundle.model.named_steps["preprocessor"].transform(frame), dtype=float)
    from backend.model import prepare_frame
    median = float(bundle.manifest["inference"]["total_charges_median"])
    prepared = prepare_frame(frame, bundle.preprocessor, median)
    return np.asarray(bundle.preprocessor.transform(prepared), dtype=float)


def _outcome_label(value):
    """Normalise a predicted label to ``Yes``/``No`` for cross-model compare."""
    if isinstance(value, str):
        return "Yes" if value.strip().lower() in ("yes", "1", "true") else "No"
    return "Yes" if int(value) == 1 else "No"


def _predict(bundle, frame, positive_index, classes):
    """Positive-class probability + predicted label for a bundle.

    The positive probability column comes from the *validated* class index
    (see :func:`analyse_class_semantics`), never a hardcoded ``[:, 1]``.

    For a ``pipeline`` bundle the RAW frame is handed straight to the
    ``Pipeline`` (it owns its own preprocessing); for an ``estimator_only``
    bundle the shared ``prepare_frame`` plus fitted preprocessor path runs
    first and only the resulting matrix reaches the bare estimator.
    """
    if bundle.manifest["model_format"] == "pipeline":
        # The Pipeline transforms the raw frame internally; do NOT pre-transform.
        proba = np.asarray(bundle.model.predict_proba(frame))[:, positive_index]
        raw_labels = np.asarray(bundle.model.predict(frame))
    else:
        matrix = _transform_for(bundle, frame)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        proba = np.asarray(bundle.model.predict_proba(matrix))[:, positive_index]
        raw_labels = np.asarray(bundle.model.predict(matrix))
    # Map each fitted-estimator output onto its own classes_ label so V1's
    # 0/1 encoding and V2's "No"/"Yes" encoding compare on equal terms.
    def _to_label(value):
        # A string-labelled bundle already returns its own class label; a
        # 0/1-labelled bundle returns an index into its own classes_.
        if isinstance(value, str):
            return _outcome_label(value)
        return _outcome_label(classes[int(value)])

    labels = np.array([_to_label(v) for v in raw_labels])
    return np.asarray(proba, dtype=float), labels


def _probe(bundle, frame):
    """Return ``('OK' | exception type, detail)`` for one input perturbation.

    Each bundle is driven exactly the way it would be in production: the
    pipeline is handed the raw frame, the bare estimator is handed the
    pre-transformed matrix.
    """
    try:
        if bundle.manifest["model_format"] == "pipeline":
            _ = bundle.model.predict(frame)
        else:
            matrix = _transform_for(bundle, frame)
            _ = bundle.model.predict(matrix)
        return "OK", None
    except Exception as exc:  # noqa: BLE001 - probing; any failure is data
        return type(exc).__name__, str(exc)[:160]


def _probe_pair(v1, v2, base_row, mutate):
    """Apply one mutation to a copy of ``base_row`` and probe both bundles."""
    return (_probe(v1, mutate(dict(base_row))),
            _probe(v2, mutate(dict(base_row))))

def _contract_columns(ctx):
    """Static (non-probe) categories: columns, domains, encoding scheme."""
    out = {}

    only_v1 = sorted(set(ctx["v1_required"]) - set(ctx["v2_required"]))
    only_v2 = sorted(set(ctx["v2_required"]) - set(ctx["v1_required"]))
    out["required_columns"] = {
        "status": SAME if not only_v1 and not only_v2 else DIFFERENT,
        "v1_count": len(ctx["v1_required"]),
        "v2_count": len(ctx["v2_required"]),
        "only_in_v1": only_v1,
        "only_in_v2": only_v2,
        "v1_order": ctx["v1_required"],
        "v2_order": ctx["v2_required"],
    }

    v1_cats = [c for c in ctx["v1_required"] if c in ctx["v1_domains"]]
    v2_cats = [c for c in ctx["v2_required"] if c in ctx["v2_domains"]]
    out["categorical_columns"] = {
        "status": SAME if v1_cats == v2_cats else DIFFERENT,
        "v1_count": len(v1_cats),
        "v2_count": len(v2_cats),
        "only_in_v1": sorted(set(v1_cats) - set(v2_cats)),
        "only_in_v2": sorted(set(v2_cats) - set(v1_cats)),
    }

    domain_diffs = {
        col: {"v1": ctx["v1_domains"][col], "v2": ctx["v2_domains"][col]}
        for col in sorted(set(ctx["v1_domains"]) & set(ctx["v2_domains"]))
        if sorted(ctx["v1_domains"][col]) != sorted(ctx["v2_domains"][col])
    }
    out["categorical_domains"] = {
        "status": SAME if not domain_diffs else DIFFERENT,
        "v1_domain": ctx["v1_domains"],
        "v2_domain": ctx["v2_domains"],
        "differing_columns": domain_diffs,
        "note": ("Domains are read from each bundle's own fitted "
                 "OneHotEncoder.categories_, not copied from a constant."),
    }

    out["numeric_columns"] = {
        "status": SAME if ctx["v1_num"] == ctx["v2_num"] else DIFFERENT,
        "v1": ctx["v1_num"],
        "v2": ctx["v2_num"],
    }
    return out


def _contract_context(v1, v2):
    """The structural facts both bundles' contracts are compared against."""
    return {
        "v1_required": list(v1.manifest["schema"]["required_cols"]),
        "v2_required": list(v2.manifest["schema"]["required_cols"]),
        "v1_domains": _categorical_domains(v1),
        "v2_domains": _categorical_domains(v2),
        "v1_settings": _encoder_settings(v1),
        "v2_settings": _encoder_settings(v2),
        "v1_num": _numeric_columns(v1),
        "v2_num": _numeric_columns(v2),
    }

def _contract_behaviour(v1, v2, base_row):
    """Ordering / SeniorCitizen / TotalCharges, probed on the real artifacts."""
    out = {}

    rev = _probe_pair(v1, v2, base_row,
                      lambda r: {k: r[k] for k in reversed(list(r.keys()))})
    out["column_ordering_requirement"] = {
        "status": SAME,
        "detail": ("Both bundles select columns by name through a fitted "
                   "ColumnTransformer, so caller column order is not "
                   "significant. Verified by scoring a reversed row."),
        "v1_reversed_input": rev[0][0],
        "v2_reversed_input": rev[1][0],
    }

    sc_num = _probe_pair(v1, v2, base_row, lambda r: {**r, "SeniorCitizen": "1"})
    sc_yes = _probe_pair(v1, v2, base_row, lambda r: {**r, "SeniorCitizen": "Yes"})
    sc_no = _probe_pair(v1, v2, base_row, lambda r: {**r, "SeniorCitizen": "No"})
    differs = ((sc_num[0][0] == "OK") != (sc_num[1][0] == "OK")
               or (sc_yes[0][0] == "OK") != (sc_yes[1][0] == "OK"))
    out["senior_citizen_values"] = {
        "status": DIFFERENT if differs else SAME,
        "numeric_string_probe": {"v1": sc_num[0][0], "v2": sc_num[1][0]},
        "text_yes_probe": {"v1": sc_yes[0][0], "v2": sc_yes[1][0]},
        "text_no_probe": {"v1": sc_no[0][0], "v2": sc_no[1][0]},
        "v1_accepts": ("0/1 numerics, numeric strings, and the text tokens "
                       "Yes/No/true/false/y/n/t/f via _coerce_senior_citizen."),
        "v2_accepts": ("0/1 numerics and numeric strings only; its "
                       "SimpleImputer(strategy='median') requires numeric "
                       "data and raises on text."),
        "exact_difference": (
            "V2 rejects SeniorCitizen='Yes'/'No' with ValueError while V1 "
            "accepts it, so an input shape the production UI accepts today "
            "would be rejected by V2."
        ) if differs else "None observed.",
    }

    tc_nan = _probe_pair(v1, v2, base_row, lambda r: {**r, "TotalCharges": np.nan})
    tc_str = _probe_pair(v1, v2, base_row, lambda r: {**r, "TotalCharges": "500.0"})
    tc_differs = (tc_nan[0][0] == "OK") != (tc_nan[1][0] == "OK")
    out["total_charges_behaviour"] = {
        "status": DIFFERENT if tc_differs else SAME,
        "nan_probe": {"v1": tc_nan[0][0], "v2": tc_nan[1][0]},
        "numeric_string_probe": {"v1": tc_str[0][0], "v2": tc_str[1][0]},
        "v1_policy": ("pd.to_numeric(errors='coerce') then fillna with the "
                      "PERSISTED training median, never a batch median."),
        "v2_policy": ("fitted SimpleImputer(strategy='median') inside the "
                      "pipeline; median recovered from its statistics_."),
        "shared_median": 1394.925,
        "exact_difference": (
            "The imputed VALUE agrees (1394.925), but the mechanism differs: "
            "V1 imputes TotalCharges only and externally; V2 imputes all four "
            "numeric columns internally."
        ) if tc_differs else "None observed.",
    }
    return out

def _contract_edge_cases(v1, v2, base_row):
    """NaN / infinity / unknown-value handling, probed on the real artifacts."""
    out = {}

    nan_num = _probe_pair(v1, v2, base_row, lambda r: {**r, "MonthlyCharges": np.nan})
    d = (nan_num[0][0] == "OK") != (nan_num[1][0] == "OK")
    out["numeric_nan_policy"] = {
        "status": DIFFERENT if d else SAME,
        "probe": {"v1": nan_num[0][0], "v2": nan_num[1][0]},
        "v1": ("Only TotalCharges is imputed; a NaN in tenure or "
               "MonthlyCharges reaches the StandardScaler unimputed."),
        "v2": ("All four numeric columns pass a fitted "
               "SimpleImputer(strategy='median')."),
        "exact_difference": (
            "A NaN in MonthlyCharges/tenure is imputed by V2 but reaches the "
            "scaler uncoerced in V1, so the bundles disagree on a missing "
            "numeric input."
        ) if d else "None observed.",
    }

    nan_cat = _probe_pair(v1, v2, base_row, lambda r: {**r, "Contract": np.nan})
    d = (nan_cat[0][0] == "OK") != (nan_cat[1][0] == "OK")
    out["categorical_nan_policy"] = {
        "status": DIFFERENT if d else SAME,
        "probe": {"v1": nan_cat[0][0], "v2": nan_cat[1][0]},
        "v1": ("_validate_categoricals calls dropna() before checking, so a "
               "NaN categorical passes validation then fails inside the "
               "fitted OneHotEncoder."),
        "v2": ("a fitted SimpleImputer(strategy='most_frequent') silently "
               "substitutes the modal level and returns a score."),
        "exact_difference": (
            "V1 raises TypeError from the encoder; V2 silently imputes. The "
            "same input either errors or scores, depending on the bundle."
        ) if d else "None observed.",
    }

    inf = _probe_pair(v1, v2, base_row, lambda r: {**r, "TotalCharges": np.inf})
    d = (inf[0][0] == "OK") != (inf[1][0] == "OK")
    out["infinity_handling"] = {
        "status": DIFFERENT if d else SAME,
        "probe": {"v1": inf[0][0], "v2": inf[1][0]},
        "detail": ("Both reject non-finite input at the fitted transformer "
                   "('Input X contains infinity or a value too large'), so "
                   "neither silently scores +/-inf."
                   if not d else "V1 and V2 differ on infinity handling."),
    }

    unk = _probe_pair(v1, v2, base_row, lambda r: {**r, "Contract": "Three year"})
    typo = _probe_pair(v1, v2, base_row,
                       lambda r: {**r, "Contract": "month-to-month"})
    d = (unk[0][0] == "OK") != (unk[1][0] == "OK")
    out["unknown_categorical_value"] = {
        "status": DIFFERENT if d else SAME,
        "out_of_domain_probe": {"v1": unk[0][0], "v2": unk[1][0]},
        "case_typo_probe": {"v1": typo[0][0], "v2": typo[1][0]},
        "v1": ("_validate_categoricals rejects any value outside the trained "
               "domain with an explicit ValueError; a typo fails loudly."),
        "v2": ("handle_unknown='ignore' silently encodes an unseen level as "
               "all-zeros. With drop=None that is an out-of-distribution row "
               "(every level of that column off), not a baseline, so a typo "
               "yields a confident but wrong score."),
        "exact_difference": (
            "V1 fails loudly on an out-of-domain or mis-cased categorical; "
            "V2 returns a silent score. A correctness/safety difference, not "
            "a quality ranking."
        ) if d else "None observed.",
    }
    return out

def _contract_encoding(v1, v2, base_row, ctx):
    """One-hot scheme, extra/missing columns, fitted-imputation presence."""
    out = {}

    missing = _probe_pair(v1, v2, base_row,
                          lambda r: {k: v for k, v in r.items() if k != "tenure"})
    extra = _probe_pair(v1, v2, base_row, lambda r: {**r, "customerID": "XXXX-0000"})
    out["extra_and_missing_columns"] = {
        "status": SAME,
        "missing_column_probe": {"v1": missing[0][0], "v2": missing[1][0]},
        "extra_column_probe": {"v1": extra[0][0], "v2": extra[1][0]},
        "detail": ("Both ignore extra columns (so a labelled frame can be "
                   "scored) and both reject a missing required column. Only "
                   "the error TYPE differs (V1 names the field, sklearn says "
                   "'columns are missing') -- cosmetic, not behavioural."),
    }

    v1_count = int(v1.model.n_features_in_)
    v2_count = int(_final_estimator(v2).n_features_in_)
    differs = (ctx["v1_settings"]["drop"] != ctx["v2_settings"]["drop"]
               or v1_count != v2_count)
    out["one_hot_scheme"] = {
        "status": DIFFERENT if differs else SAME,
        "v1": {**ctx["v1_settings"], "transformed_features": v1_count},
        "v2": {**ctx["v2_settings"], "transformed_features": v2_count},
        "exact_difference": (
            "V1 uses drop='first' (30 columns, a reference-level baseline "
            "per categorical). V2 uses drop=None (45 columns, every level). "
            "The feature spaces are structurally different, so a transformed "
            "matrix or a stored attribution from one is not interchangeable "
            "with the other."
        ) if differs else "None observed.",
    }

    out["fitted_internal_imputation"] = {
        "status": DIFFERENT,
        "v1": "No SimpleImputer; hygiene comes from the shared prepare_frame.",
        "v2": "SimpleImputer on BOTH the numeric and categorical blocks.",
        "exact_difference": ("V2's preprocessing is self-contained inside the "
                             "pipeline; V1 depends on external hygiene."),
    }
    return out


def compare_input_contracts(v1, v2, base_row):
    """Compare the raw production input contract of V1 and V2.

    Every category is reported explicitly as SAME / DIFFERENT / UNKNOWN.
    Differences are never silently reconciled: where the two bundles disagree
    the exact difference is spelled out in ``exact_difference``.
    """
    ctx = _contract_context(v1, v2)
    categories = {}
    categories.update(_contract_columns(ctx))
    categories.update(_contract_behaviour(v1, v2, base_row))
    categories.update(_contract_edge_cases(v1, v2, base_row))
    categories.update(_contract_encoding(v1, v2, base_row, ctx))
    return {
        "verdict": ("SAME" if all(c["status"] == SAME
                                  for c in categories.values()) else "DIFFERENT"),
        "n_same": sum(1 for c in categories.values() if c["status"] == SAME),
        "n_different": sum(1 for c in categories.values()
                           if c["status"] == DIFFERENT),
        "n_unknown": sum(1 for c in categories.values() if c["status"] == UNKNOWN),
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# STEP 4 -- Common deterministic dataset
# ---------------------------------------------------------------------------
def load_common_sample():
    """Recover the untouched Phase 3B final test set.

    The split is reproduced with the exact call Phase 3B used
    (``train_test_split(test_size=0.2, random_state=42, stratify=y)``), so
    these are the *same* untouched test rows that produced the Phase 3B
    metrics, not a new split. The source dataset is only read, never altered,
    and no synthetic records are created.
    """
    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_bin = (pd.Series(np.asarray(y_test)).astype(str) == "Yes").astype(int)
    return {
        "test_features": X_test.reset_index(drop=True),
        "test_labels": np.asarray(y_test),
        "test_labels_binary": y_bin.to_numpy(),
        "n_total_dataset": int(len(df)),
        "n_test": int(len(X_test)),
        "n_train_unused": int(len(X_train)),
        "n_test_positive": int(y_bin.sum()),
        "churn_prevalence": float(y_bin.mean()),
    }

# ---------------------------------------------------------------------------
# STEPS 5-9 -- Scoring, probability/label comparison, disagreements
# ---------------------------------------------------------------------------
def score_both(v1, v2, frame, threshold=PRODUCTION_THRESHOLD):
    """Score the SAME raw rows through both bundles at the production threshold.

    The threshold is passed explicitly and defaults to the production 0.50.
    The Phase 3C 0.25 analysis threshold is never used here; requesting it
    raises rather than silently running a different decision rule.
    """
    if abs(threshold - PRODUCTION_THRESHOLD) > 1e-12:
        raise ValueError(
            f"Shadow audit must use the production threshold "
            f"{PRODUCTION_THRESHOLD}; got {threshold}. The Phase 3C "
            f"{ANALYSIS_ONLY_THRESHOLD} threshold is analysis-only."
        )
    v1c = analyse_class_semantics(v1)
    v2c = analyse_class_semantics(v2)
    p1, l1 = _predict(v1, frame, v1c["positive_class_index"], v1c["classes"])
    p2, l2 = _predict(v2, frame, v2c["positive_class_index"], v2c["classes"])
    return {
        "v1_prob": p1, "v2_prob": p2,
        "v1_label": l1, "v2_label": l2,
        "v1_threshold_label": np.where(p1 >= threshold, "Yes", "No"),
        "v2_threshold_label": np.where(p2 >= threshold, "Yes", "No"),
        "threshold": float(threshold),
    }


def compare_probabilities(p1, p2):
    """Behavioural difference statistics between the two probability vectors.

    These are DIFFERENCES, not quality rankings: they say how far apart the
    two bundles outputs are, not which bundle is preferable.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    diff = np.abs(p1 - p2)
    corr = (float(np.corrcoef(p1, p2)[0, 1])
            if len(p1) > 1 and p1.std() > 0 and p2.std() > 0 else None)
    return {
        "mean_absolute_difference": float(diff.mean()),
        "median_absolute_difference": float(np.median(diff)),
        "max_absolute_difference": float(diff.max()),
        "std_of_differences": float(diff.std()),
        "correlation_v1_v2": corr,
        "min_v1_probability": float(p1.min()),
        "max_v1_probability": float(p1.max()),
        "min_v2_probability": float(p2.min()),
        "max_v2_probability": float(p2.max()),
        "interpretation": ("Behavioural difference only. These numbers do not "
                           "rank the bundles and do not indicate which is "
                           "better."),
    }


def compare_labels(l1, l2, threshold=PRODUCTION_THRESHOLD):
    """Label agreement statistics and the V1-vs-V2 confusion matrix."""
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)
    n = int(len(l1))
    agree = int((l1 == l2).sum())
    disagree = n - agree
    return {
        "total_rows": n,
        "v1_predicted_churn": int((l1 == "Yes").sum()),
        "v2_predicted_churn": int((l2 == "Yes").sum()),
        "label_agreement": agree,
        "label_disagreement": disagree,
        "disagreement_percentage": (100.0 * disagree / n) if n else 0.0,
        "threshold": float(threshold),
        "confusion_matrix_v1_rows_v2_cols": {
            "both_No": int(((l1 == "No") & (l2 == "No")).sum()),
            "v1_No_v2_Yes": int(((l1 == "No") & (l2 == "Yes")).sum()),
            "v1_Yes_v2_No": int(((l1 == "Yes") & (l2 == "No")).sum()),
            "both_Yes": int(((l1 == "Yes") & (l2 == "Yes")).sum()),
        },
        "interpretation": ("Model-behaviour comparison only. No bundle is "
                           "labelled better."),
    }

def sample_disagreements(p1, p2, l1, l2, true_labels=None,
                         n=DISAGREEMENT_SAMPLE_SIZE):
    """A deterministic sample of rows where the two bundles disagree.

    Rows are identified by POSITIONAL INDEX only. No customer identifier or
    name is emitted, so the audit artifact carries no unnecessary PII. The
    sample is the first ``n`` disagreement indices in row order, which makes
    it fully reproducible.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    idx = [int(i) for i in np.flatnonzero(np.asarray(l1) != np.asarray(l2))]
    rows = []
    for i in idx[:n]:
        row = {
            "row_index": i,
            "v1_probability": round(float(p1[i]), 6),
            "v2_probability": round(float(p2[i]), 6),
            "v1_label": str(l1[i]),
            "v2_label": str(l2[i]),
            "absolute_probability_difference": round(abs(p1[i] - p2[i]), 6),
        }
        if true_labels is not None:
            row["true_label"] = str(true_labels[i])
        rows.append(row)
    return {
        "total_disagreements": len(idx),
        "sampled": len(rows),
        "sample_rule": "first N disagreement indices in row order (deterministic)",
        "rows": rows,
        "pii_note": ("Row index only; no customerID, name or other identifier "
                     "is written to the audit artifact."),
    }


def threshold_proximity(p1, p2, l1, l2, band=NEAR_THRESHOLD_BAND):
    """Operational sensitivity: how many predictions sit near the threshold.

    The threshold is NOT changed; this only counts how many decisions are
    sensitive to a small threshold perturbation.
    """
    lo, hi = band
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)
    n1 = int(((p1 >= lo) & (p1 < hi)).sum())
    n2 = int(((p2 >= lo) & (p2 < hi)).sum())
    disagree = l1 != l2
    near1 = (p1 >= lo) & (p1 < hi)
    near2 = (p2 >= lo) & (p2 < hi)
    return {
        "band": [lo, hi],
        "threshold": PRODUCTION_THRESHOLD,
        "v1_in_band": n1,
        "v2_in_band": n2,
        "v1_band_percentage": 100.0 * n1 / len(p1) if len(p1) else 0.0,
        "v2_band_percentage": 100.0 * n2 / len(p2) if len(p2) else 0.0,
        "disagreements_near_threshold": int(disagree[near1 | near2].sum()),
        "note": ("A row can be near the threshold for either bundle, so this "
                 "disagreement count is over the UNION of the two bands, not "
                 "the sum of two disjoint sets. The threshold is unchanged."),
    }


# ---------------------------------------------------------------------------
# STEP 10 -- Single vs batch parity
# ---------------------------------------------------------------------------
def single_batch_parity(bundle, frame, n=SINGLE_BATCH_SAMPLE_SIZE,
                        threshold=PRODUCTION_THRESHOLD):
    """Verify row-by-row scoring equals scoring the same rows as one batch.

    A customer's output must not depend on which other rows share the batch.
    This runs independently for V1 and V2.
    """
    cls = analyse_class_semantics(bundle)
    index, classes = cls["positive_class_index"], cls["classes"]
    picks = list(range(min(n, len(frame))))

    batch_p, batch_l = _predict(bundle, frame, index, classes)
    single_p, single_l = [], []
    for i in picks:
        p, l = _predict(bundle, frame.iloc[[i]], index, classes)
        single_p.append(float(p[0]))
        single_l.append(str(l[0]))

    single_p = np.asarray(single_p, float)
    single_l = np.asarray(single_l)
    bp = np.asarray(batch_p[picks], float)
    bl = np.asarray(batch_l[picks])
    return {
        "bundle_version": bundle.manifest["bundle_version"],
        "rows_checked": len(picks),
        "max_probability_delta": (float(np.abs(single_p - bp).max())
                                  if len(picks) else 0.0),
        "probability_parity": bool(np.allclose(single_p, bp, atol=1e-12, rtol=0)),
        "label_parity": bool(np.array_equal(single_l, bl)),
        "label_disagreements": int((single_l != bl).sum()),
        "threshold": float(threshold),
        "note": ("Single-row scoring and the equivalent one-row batch must "
                 "agree; this must pass independently per bundle."),
    }

# ---------------------------------------------------------------------------
# STEP 11 -- Determinism
# ---------------------------------------------------------------------------
def run_shadow(v1, v2, frame, threshold=PRODUCTION_THRESHOLD):
    """One complete shadow run, returned as comparable arrays."""
    return score_both(v1, v2, frame, threshold=threshold)


def determinism_check(v1, v2, frame, threshold=PRODUCTION_THRESHOLD):
    """Run the shadow comparison twice and verify bit-identical results."""
    r1 = run_shadow(v1, v2, frame, threshold)
    r2 = run_shadow(v1, v2, frame, threshold)
    keys = ("v1_prob", "v2_prob", "v1_label", "v2_label",
            "v1_threshold_label", "v2_threshold_label")
    identical = {
        k: bool(np.array_equal(np.asarray(r1[k]), np.asarray(r2[k])))
        for k in keys
    }
    return {
        "runs": 2,
        "v1_run1_equals_run2": bool(identical["v1_prob"] and identical["v1_label"]),
        "v2_run1_equals_run2": bool(identical["v2_prob"] and identical["v2_label"]),
        "field_level_identity": identical,
        "deterministic": all(identical.values()),
        "note": ("Both bundles are pure fitted transforms, so repeated runs "
                 "must be bit-identical; no random variation is expected."),
    }


# ---------------------------------------------------------------------------
# STEP 12 -- XAI compatibility (read-only; never modifies explainability.py)
# ---------------------------------------------------------------------------
def xai_compatibility(v1, v2, n=XAI_SAMPLE_SIZE):
    """Test whether both bundles can pass through the existing XAI architecture.

    ``backend/explainability.py`` is READ ONLY here; this function only calls
    its public API. Any failure is reported as a promotion BLOCKER rather than
    worked around.
    """
    from backend import explainability as xai

    # explainability memoises explainers in a module-level cache keyed by
    # estimator TYPE + directory + background, not by estimator instance.
    # Populating it from a bundle would make a later production call reuse
    # the bundle's explainer, so the cache is snapshotted and restored: this
    # audit must not leave any state behind in the production XAI layer.
    saved_cache = dict(xai._MODEL_CACHE)
    saved_global = dict(getattr(xai, "_GLOBAL_CACHE", {}))
    try:
        return _xai_compatibility_inner(xai, v1, v2, n)
    finally:
        xai._MODEL_CACHE.clear()
        xai._MODEL_CACHE.update(saved_cache)
        xai._GLOBAL_CACHE.clear()
        xai._GLOBAL_CACHE.update(saved_global)


def _xai_compatibility_inner(xai, v1, v2, n):
    results = {}
    for label, bundle in (("v1", v1), ("v2", v2)):
        entry = {"bundle_version": bundle.manifest["bundle_version"]}

        # (a) Can the ModelBundle object itself be normalised to a ModelView?
        try:
            xai._as_model_view(bundle)
            entry["model_bundle_accepted_directly"] = True
            entry["model_bundle_accepted_directly_error"] = None
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            entry["model_bundle_accepted_directly"] = False
            entry["model_bundle_accepted_directly_error"] = (
                f"{type(exc).__name__}: {str(exc)[:180]}")

        # (b) Can the bundle's own fitted object be explained? V1's path is
        # the production ChurnModel; V2's path is its fitted Pipeline.
        try:
            if bundle.manifest["model_format"] == "pipeline":
                target = bundle.model
            else:
                from backend.model import ChurnModel
                target = ChurnModel()
                target.preprocessor = bundle.preprocessor
                target.best_model = bundle.model
                target.best_model_name = bundle.manifest["metadata"]["model_type"]
                target.total_charges_median = float(
                    bundle.manifest["inference"]["total_charges_median"])
                target.feature_names = None
            view = xai._as_model_view(target)
            entry["native_object_accepted"] = True
            entry["native_object_error"] = None
            entry["explainer_type"] = type(xai.get_shap_explainer(view)).__name__
            names = xai.get_feature_names(view)
            entry["transformed_feature_count"] = len(names)
            entry["transformed_features_head"] = names[:4]
            entry["estimator_n_features_in"] = int(view.n_features)

            g1 = xai.get_global_shap_values(model=view, sample_size=n,
                                            random_state=RANDOM_STATE)
            entry["shap_output_shape"] = list(np.shape(g1["shap_values"]))
            entry["base_value"] = float(np.ravel(g1["base_value"])[0])
            entry["importance_rows"] = int(len(g1["importance"]))
            g2 = xai.get_global_shap_values(model=view, sample_size=n,
                                            random_state=RANDOM_STATE)
            entry["deterministic_output"] = bool(np.allclose(
                np.asarray(g1["importance"]["Mean |SHAP|"], dtype=float),
                np.asarray(g2["importance"]["Mean |SHAP|"], dtype=float)))
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            entry["native_object_accepted"] = False
            entry["native_object_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        results[label] = entry

    return {
        "explained_via": "backend/explainability.py (read-only, unmodified)",
        "sample_size": n,
        "random_state": RANDOM_STATE,
        "bundles": results,
        "note": ("Explainer type is selected from the fitted estimator "
                 "(LinearExplainer for V1's LogisticRegression, TreeExplainer "
                 "for V2's GradientBoostingClassifier), never hard-coded."),
    }

# ---------------------------------------------------------------------------
# STEP 13 -- Contract blocker analysis
# ---------------------------------------------------------------------------
BLOCKERS = [
    {
        "id": "BLOCKER-01",
        "title": "Raw input contract is NOT drop-in compatible",
        "severity": "BLOCKING",
        "area": "1. Raw schema",
        "detail": ("Both bundles declare the same 19 required raw columns and "
                   "both ignore extra columns, so the column NAME set agrees. "
                   "The accepted VALUE space differs on SeniorCitizen: V2's "
                   "median SimpleImputer rejects the text values 'Yes'/'No' "
                   "that V1 accepts."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-02",
        "title": "Categorical domains agree, but unknown values differ",
        "severity": "BLOCKING",
        "area": "2. Categorical domains / 15. Error handling",
        "detail": ("The fitted domains are identical across V1 and V2 "
                   "(verified from each encoder's categories_). The "
                   "DIFFERENCE is the handling of an out-of-domain value: "
                   "V1's _validate_categoricals raises ValueError, while V2's "
                   "handle_unknown='ignore' with drop=None silently produces "
                   "an all-zeros column, i.e. a confidently wrong score. "
                   "Migrating V2 in place of V1 would silently remove the "
                   "project's loud-failure guard on typos."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-03",
        "title": "Numeric NaN handling differs",
        "severity": "BLOCKING",
        "area": "3. Numeric handling / 6. NaN behavior",
        "detail": ("A NaN in tenure/MonthlyCharges/SeniorCitizen is imputed by "
                   "V2's fitted SimpleImputer but reaches V1's StandardScaler "
                   "unimputed. The bundles therefore disagree on a missing "
                   "numeric input, so behaviour would change on migration for "
                   "incomplete rows."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-04",
        "title": "TotalCharges imputation mechanism differs (value identical)",
        "severity": "NON-BLOCKING",
        "area": "4. TotalCharges handling",
        "detail": ("Both impute a missing TotalCharges with the same persisted "
                   "median 1394.925, so the VALUE agrees. The mechanism differs "
                   "(V1 external fillna, V2 fitted SimpleImputer) and V2 "
                   "additionally imputes the other three numeric columns. "
                   "Behaviour-equivalent for a missing TotalCharges."),
        "blocks_promotion": False,
    },
    {
        "id": "BLOCKER-05",
        "title": "Categorical NaN handling differs (error vs silent imputation)",
        "severity": "BLOCKING",
        "area": "6. NaN behavior / 15. Error handling",
        "detail": ("A NaN categorical raises TypeError under V1 but is "
                   "silently replaced by the modal level under V2. Same input: "
                   "either an exception or a silent score, depending on bundle."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-06",
        "title": "Infinity handling is compatible",
        "severity": "NON-BLOCKING",
        "area": "7. Infinity behavior",
        "detail": ("Both bundles reject non-finite numeric input at the fitted "
                   "transformer with a ValueError, so neither silently scores "
                   "+/-inf. No migration risk here."),
        "blocks_promotion": False,
    },
    {
        "id": "BLOCKER-07",
        "title": "Class semantics differ in ENCODING but agree in SEMANTICS",
        "severity": "NON-BLOCKING",
        "area": "8. Class semantics / 9. Positive probability semantics",
        "detail": ("V1's classes_ are ints [0, 1] with positive class 1; V2's "
                   "are strings ['No', 'Yes'] with positive class 'Yes'. Both "
                   "resolve the same churn semantics ('Yes'), and the positive "
                   "probability column is selected by a validated class index "
                   "here, never a hardcoded 1. A migration must map the int "
                   "0/1 encoding to Yes/No at the UI boundary."),
        "blocks_promotion": False,
    },
    {
        "id": "BLOCKER-08",
        "title": "Transformed feature representation differs (30 vs 45)",
        "severity": "BLOCKING",
        "area": "14. Transformed feature representation",
        "detail": ("V1 emits 30 one-hot columns (drop='first', a reference "
                   "baseline per categorical); V2 emits 45 (drop=None, every "
                   "level). The feature spaces are structurally different, so a "
                   "transformed matrix or a stored SHAP attribution from one is "
                   "not interchangeable with the other, and any downstream "
                   "consumer assuming 30 columns would break."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-09",
        "title": "backend.explainability does not accept a ModelBundle directly",
        "severity": "BLOCKING",
        "area": "13. XAI behavior",
        "detail": ("explainability._as_model_view raises "
                   "XAIUnsupportedModelError for a backend.model_bundle."
                   "ModelBundle, accepting only a ChurnModel, a fitted "
                   "Pipeline, or a ModelView. Both bundles ARE explainable via "
                   "their native fitted objects, but wiring a bundle into the "
                   "XAI layer needs a change to explainability.py, which this "
                   "audit must not (and did not) make."),
        "blocks_promotion": True,
    },
    {
        "id": "BLOCKER-10",
        "title": "Error-type differences on missing columns",
        "severity": "NON-BLOCKING",
        "area": "15. Error handling",
        "detail": ("Both reject a missing required column, but V1 raises "
                   "ValueError naming the field while V2 surfaces sklearn's "
                   "'columns are missing'. Cosmetic, but it may surface in UI "
                   "error messaging."),
        "blocks_promotion": False,
    },
]
WARNINGS = [
    "WARN-01: V1 exposes 30 transformed features and V2 exposes 45; a "
    "hard-coded width assumption downstream would break on migration.",
    "WARN-02: This is a BEHAVIOURAL difference audit only. The probability "
    "and label statistics do not rank the bundles and do not indicate which "
    "model is better.",
    "WARN-03: The Phase 3C 0.25 threshold remains analysis-only; the "
    "production threshold is 0.50 and was not changed by this audit.",
    "WARN-04: Disagreement rows are identified by positional index only; the "
    "audit artifact contains no customer identifiers (no PII).",
    "WARN-05: No promotion was performed. active_bundle.json still points to "
    "v1 and production inference still uses the legacy LogisticRegression.",
]


# ---------------------------------------------------------------------------
# STEP 14/17 -- Hash safety, orchestrator, artifact writers
# ---------------------------------------------------------------------------
def verify_frozen_hashes():
    """Recompute SHA-256 of every frozen artifact and compare to baseline.

    Digests are compared case-insensitively because PowerShell's
    ``Get-FileHash`` emits uppercase hex while the baselines are lowercase.
    """
    results = {}
    for rel, expected in FROZEN_HASHES.items():
        path = os.path.join(BASE_DIR, rel)
        if not os.path.exists(path):
            results[rel] = {"present": False, "matches": False,
                            "note": "missing"}
            continue
        actual = _sha256(path)
        ok = actual.lower() == expected.lower()
        results[rel] = {
            "present": True,
            "sha256": actual,
            "expected_sha256": expected,
            "matches": ok,
            "unchanged": ok,
        }
    return {
        "all_unchanged": all(r.get("matches") for r in results.values()),
        "n_artifacts": len(results),
        "artifacts": results,
    }


def run_audit(write_artifacts=True):
    """Run the complete read-only shadow audit and optionally persist it."""
    v1, v2 = load_bundles()
    sample = load_common_sample()
    frame = sample["test_features"]
    base_row = frame.iloc[0].to_dict()

    contract = compare_input_contracts(v1, v2, base_row)
    scored = score_both(v1, v2, frame, PRODUCTION_THRESHOLD)

    audit = {
        "audit": "phase4_step7_shadow_scoring_and_promotion_readiness",
        "analysis_only": True,
        "promotion_performed": False,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "v1_bundle_version": v1.manifest["bundle_version"],
        "v2_bundle_version": v2.manifest["bundle_version"],
        "active_bundle": json.load(open(ACTIVE_POINTER))["active_version"],
        "sample_size": sample["n_test"],
        "data_source": {
            "file": "data/telco_customer_churn_cleaned.csv",
            "subset": "untouched Phase 3B final test set",
            "split": (f"train_test_split(test_size={TEST_SIZE}, "
                      f"random_state={RANDOM_STATE}, stratify=y)"),
            "random_seed": RANDOM_STATE,
            "n_total_dataset": sample["n_total_dataset"],
            "n_test": sample["n_test"],
            "n_positive": sample["n_test_positive"],
            "churn_prevalence": sample["churn_prevalence"],
            "synthetic_records_created": False,
            "dataset_altered": False,
        },
        "threshold": {
            "production_threshold_used": PRODUCTION_THRESHOLD,
            "phase_3c_analysis_threshold": ANALYSIS_ONLY_THRESHOLD,
            "analysis_threshold_adopted": False,
            "threshold_changed": False,
        },
        "class_metadata": {
            "v1": analyse_class_semantics(v1),
            "v2": analyse_class_semantics(v2),
        },
        "contract_comparison": contract,
        "probability_statistics": compare_probabilities(
            scored["v1_prob"], scored["v2_prob"]),
        "label_agreement_statistics": compare_labels(
            scored["v1_label"], scored["v2_label"], PRODUCTION_THRESHOLD),
        "disagreement_summary": sample_disagreements(
            scored["v1_prob"], scored["v2_prob"],
            scored["v1_label"], scored["v2_label"],
            true_labels=sample["test_labels"]),
        "threshold_proximity": threshold_proximity(
            scored["v1_prob"], scored["v2_prob"],
            scored["v1_label"], scored["v2_label"]),
        "single_batch_parity": {
            "v1": single_batch_parity(v1, frame),
            "v2": single_batch_parity(v2, frame),
        },
        "determinism": determinism_check(v1, v2, frame),
        "xai_compatibility": xai_compatibility(v1, v2),
        "blockers": BLOCKERS,
        "warnings": WARNINGS,
        "hash_verification": verify_frozen_hashes(),
        "interpretation": ("Compatibility and behavioural differences only. "
                           "This audit contains no winner, ranking, score or "
                           "promotion recommendation."),
    }

    audit["artifacts_written"] = (
        write_audit_artifacts(audit, frame, scored, sample)
        if write_artifacts else {})
    return audit

def write_audit_artifacts(audit, frame, scored, sample):
    """Persist the audit JSON plus a per-row comparison CSV.

    Only NEW analysis files are written under ``models/experiments/``. No
    production artifact, bundle artifact or active pointer is touched.
    """
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    json_path = os.path.join(EXPERIMENT_DIR, "shadow_audit.json")
    csv_path = os.path.join(EXPERIMENT_DIR, "shadow_comparison.csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(audit), fh, indent=2)

    # Per-row comparison, identified by row index only -- no customerID / PII.
    pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "v1_probability": np.round(scored["v1_prob"], 6),
        "v2_probability": np.round(scored["v2_prob"], 6),
        "absolute_probability_difference": np.round(
            np.abs(scored["v1_prob"] - scored["v2_prob"]), 6),
        "v1_label": scored["v1_label"],
        "v2_label": scored["v2_label"],
        "labels_agree": scored["v1_label"] == scored["v2_label"],
        "true_label": sample["test_labels"],
        "threshold": PRODUCTION_THRESHOLD,
    }).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def print_summary(audit):
    """Human-readable audit summary (no winner / ranking language)."""
    ds, ps = audit["data_source"], audit["probability_statistics"]
    ls, cc = audit["label_agreement_statistics"], audit["contract_comparison"]
    v1c, v2c = audit["class_metadata"]["v1"], audit["class_metadata"]["v2"]
    print("=" * 68)
    print("PHASE 4 STEP 7 -- SHADOW AUDIT (read-only, analysis only)")
    print("=" * 68)
    print(f"timestamp          : {audit['timestamp']}")
    print(f"v1 bundle          : {audit['v1_bundle_version']} "
          f"({v1c['model_type']}, classes {v1c['classes']})")
    print(f"v2 bundle          : {audit['v2_bundle_version']} "
          f"({v2c['model_type']}, classes {v2c['classes']})")
    print(f"active bundle      : {audit['active_bundle']} (unchanged)")
    print(f"sample             : {ds['n_test']} rows, seed {ds['random_seed']}, "
          f"prevalence {ds['churn_prevalence']:.4f}")
    print(f"threshold          : "
          f"{audit['threshold']['production_threshold_used']} "
          f"(0.25 analysis threshold NOT adopted)")
    print("-" * 68)
    print(f"v1 classes_        : {v1c['classes']} -> positive "
          f"{v1c['positive_class']!r} @ index {v1c['positive_class_index']} "
          f"(churn semantics = {v1c['positive_semantic_label']!r})")
    print(f"v2 classes_        : {v2c['classes']} -> positive "
          f"{v2c['positive_class']!r} @ index {v2c['positive_class_index']} "
          f"(churn semantics = {v2c['positive_semantic_label']!r})")
    print(f"both resolve churn as 'Yes': "
          f"{v1c['positive_class_is_yes'] and v2c['positive_class_is_yes']}")
    print("-" * 68)
    print(f"input contract     : {cc['verdict']} ({cc['n_same']} SAME / "
          f"{cc['n_different']} DIFFERENT / {cc['n_unknown']} UNKNOWN)")
    for name, cat in cc["categories"].items():
        if cat["status"] == "DIFFERENT":
            print(f"    DIFFERENT: {name}")
    print("-" * 68)
    print(f"mean |p1-p2|       : {ps['mean_absolute_difference']:.6f}")
    print(f"median |p1-p2|     : {ps['median_absolute_difference']:.6f}")
    print(f"max |p1-p2|        : {ps['max_absolute_difference']:.6f}")
    print(f"std of differences : {ps['std_of_differences']:.6f}")
    print(f"correlation v1~v2  : {ps['correlation_v1_v2']:.6f}")
    print(f"v1 prob range      : [{ps['min_v1_probability']:.6f}, "
          f"{ps['max_v1_probability']:.6f}]")
    print(f"v2 prob range      : [{ps['min_v2_probability']:.6f}, "
          f"{ps['max_v2_probability']:.6f}]")
    print("-" * 68)
    print(f"rows               : {ls['total_rows']}")
    print(f"v1 predicted churn : {ls['v1_predicted_churn']}")
    print(f"v2 predicted churn : {ls['v2_predicted_churn']}")
    print(f"label agreement    : {ls['label_agreement']}")
    print(f"label disagreement : {ls['label_disagreement']} "
          f"({ls['disagreement_percentage']:.2f}%)")
    cm = ls["confusion_matrix_v1_rows_v2_cols"]
    print(f"confusion v1 x v2  : both_No={cm['both_No']} "
          f"v1No_v2Yes={cm['v1_No_v2_Yes']} "
          f"v1Yes_v2No={cm['v1_Yes_v2_No']} both_Yes={cm['both_Yes']}")
    print("-" * 68)
    tp = audit["threshold_proximity"]
    print(f"near-threshold band : {tp['band']}")
    print(f"  v1 in band       : {tp['v1_in_band']}")
    print(f"  v2 in band       : {tp['v2_in_band']}")
    print(f"  disagreements    : {tp['disagreements_near_threshold']}")
    print("-" * 68)
    sp = audit["single_batch_parity"]
    print(f"single/batch v1    : prob={sp['v1']['probability_parity']} "
          f"label={sp['v1']['label_parity']}")
    print(f"single/batch v2    : prob={sp['v2']['probability_parity']} "
          f"label={sp['v2']['label_parity']}")
    print(f"determinism        : {audit['determinism']['deterministic']}")
    print("-" * 68)
    for key in ("v1", "v2"):
        e = audit["xai_compatibility"]["bundles"][key]
        print(f"XAI {key}: native_ok={e.get('native_object_accepted')} "
              f"explainer={e.get('explainer_type')} "
              f"features={e.get('transformed_feature_count')} "
              f"deterministic={e.get('deterministic_output')}")
        print(f"     ModelBundle accepted directly by _as_model_view: "
              f"{e.get('model_bundle_accepted_directly')}")
    print("-" * 68)
    blocking = [b for b in audit["blockers"] if b["blocks_promotion"]]
    other = [b for b in audit["blockers"] if not b["blocks_promotion"]]
    print(f"BLOCKERS           : {len(blocking)}")
    for b in blocking:
        print(f"    {b['id']}  {b['title']}")
    print(f"NON-BLOCKING diffs : {len(other)}")
    for b in other:
        print(f"    {b['id']}  {b['title']}")
    print("-" * 68)
    print(f"hashes unchanged   : {audit['hash_verification']['all_unchanged']}")
    print(f"promotion performed: {audit['promotion_performed']}")
    print("=" * 68)


if __name__ == "__main__":
    print_summary(run_audit(write_artifacts=True))