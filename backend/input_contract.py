"""Shared, explicit production input contract for the ModelBundle architecture.

Why this module exists
----------------------
The Phase 4 Step 7 READ-ONLY shadow audit proved that the v1 and v2 bundles
disagree on the *same* raw input, for four separate reasons (BLOCKER-01,
BLOCKER-02, BLOCKER-03, BLOCKER-05). Those disagreements were not caused by
the frozen models: they were caused by the **absence of a single shared
validation and normalisation step** in front of them. Each bundle was
"validating on its own", so each applied its own fitted preprocessing rules
and its own error behaviour.

This module is that missing step. It is a *contract*, not a compatibility
shim: it states, in one place and explicitly, what the application accepts
before any bundle is allowed to see a row. Every bundle -- v1, v2, and any
future one -- must route raw input through :func:`prepare_input_frame` so
that the required column set, the accepted ``SeniorCitizen`` representations,
the ``TotalCharges`` policy, the numeric/categorical NaN policy, the
infinity policy, the categorical domain check and the raised error type are
all identical across bundles.

Design constraint (Phase 4 Step 8)
----------------------------------
The underlying preprocessing *representations* are deliberately NOT unified.
v1 remains ``estimator + ColumnTransformer`` emitting 30 columns; v2 remains
a self-contained ``Pipeline`` emitting 45 columns. This contract governs only
the *raw* input in front of those representations, so BLOCKER-08 is resolved
by making the difference explicit, self-describing bundle metadata rather
than by forcing two frozen feature spaces together (which would need a
refit, and is forbidden).

This module never fits, refits or mutates a model artifact, never changes a
decision threshold, and never coerces a value except where the accepted
representation set above explicitly permits it.
"""

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "ContractError",
    "NumericPolicy",
    "ContractSpec",
    "SENIOR_CITIZEN_ALIASES",
    "TOTAL_CHARGES_POLICY_IMPUTE_MEDIAN",
    "prepare_input_frame",
    "normalise_senior_citizen",
    "coerce_numeric_columns",
    "reject_infinities",
    "validate_categorical_domains",
]


class ContractError(ValueError):
    """The single exception type raised for every contract violation.

    A dedicated type so callers -- the Streamlit UI, the batch engine, the
    shadow audit -- can catch exactly one class for "this input violates the
    contract", independent of which bundle consumes it and of which rule was
    broken. This is what unifies the audit's BLOCKER-02 / BLOCKER-05 /
    BLOCKER-10 error-type differences.
    """


#: Closed set of accepted ``SeniorCitizen`` representations, keyed by the
#: lower-cased / whitespace-stripped token. The models were trained on int64
#: 0/1, so the contract's job is to funnel every accepted spelling to that
#: 0/1 and reject everything else loudly. The set is the union of what the
#: two bundles historically tolerated: a widening of v2 (BLOCKER-01) and a
#: preservation of v1, so no previously valid v1 input becomes invalid.
SENIOR_CITIZEN_ALIASES: Dict[str, int] = {
    "0": 0, "0.0": 0, "1": 1, "1.0": 1,
    "no": 0, "yes": 1, "false": 0, "true": 1, "n": 0, "y": 1, "f": 0, "t": 1,
}

#: ``TotalCharges`` is the one numeric with a documented, persisted
#: training-median imputation policy (its CSV form is historical lifetime
#: spend and is frequently blank for zero-tenure customers).
TOTAL_CHARGES_POLICY_IMPUTE_MEDIAN = "impute_with_persisted_training_median"


class NumericPolicy:
    """Named, shared policies for the numeric columns.

    These are named objects rather than magic booleans scattered through the
    validation code so the audit can assert that a given policy was *applied*,
    instead of inferring it from a downstream result.
    """

    #: A missing numeric is replaced with the bundle's persisted training
    #: statistic *before* inference, identically for every bundle. Only
    #: available for columns that actually expose such a statistic; see
    #: :data:`MISSING_NUMERIC_REJECT` for the policy applied elsewhere.
    MISSING_NUMERIC_IMPUTE = "impute_with_training_statistic"

    #: A missing *categorical* is rejected loudly rather than silently imputed
    #: to a modal level. This intentionally keeps v1's loud-failure guard
    #: (BLOCKER-05) instead of v2's silent modal fill, because a wrong
    #: category is far more dangerous than a refused row.
    MISSING_CATEGORICAL_REJECT = "reject"

    #: A missing numeric is REJECTED unless the column has an explicit,
    #: persisted training statistic to impute it with.
    #:
    #: This is the policy that closes BLOCKER-03 *without* changing production.
    #: v1 today rejects a missing ``tenure`` / ``MonthlyCharges`` (its fitted
    #: ``StandardScaler`` cannot consume NaN) but imputes a missing
    #: ``TotalCharges`` from the persisted median. v2, whose fitted imputer
    #: covers all four numerics, silently accepted all four. Imputing the
    #: other three would have quietly changed v1's production behaviour, so
    #: the contract instead adopts **v1's** semantics for both bundles: only
    #: ``TotalCharges`` (which has a documented, persisted median) is
    #: imputed; every other missing numeric is refused loudly. That removes
    #: the disagreement without touching a single frozen parameter.
    MISSING_NUMERIC_REJECT = "reject_unless_statistic_available"

    #: A non-finite (infinite) numeric is always rejected, never imputed.
    INFINITY_REJECT = "reject"



class ContractSpec:
    """The declarative contract a single bundle exposes to the application.

    A ``ContractSpec`` is *data*, not behaviour: it names the required
    columns, the numeric columns, the categorical domains, the accepted
    ``SeniorCitizen`` representations and the NaN / infinity policies.
    :func:`prepare_input_frame` is the single function that enforces a spec, so
    neither bundle carries a hand-written special case.
    """

    __slots__ = (
        "required_columns", "numeric_columns", "categorical_domains",
        "senior_citizen_aliases", "total_charges_policy",
        "total_charges_statistic", "numeric_missing_policy",
        "categorical_missing_policy", "infinity_policy", "extra_column_policy",
        "numeric_impute_statistics",
    )

    def __init__(
        self,
        required_columns: Sequence[str],
        numeric_columns: Sequence[str],
        categorical_domains: Mapping[str, Sequence[str]],
        senior_citizen_aliases: Mapping[str, int],
        total_charges_policy: str,
        total_charges_statistic: float,
        numeric_missing_policy: str = NumericPolicy.MISSING_NUMERIC_REJECT,
        categorical_missing_policy: str = NumericPolicy.MISSING_CATEGORICAL_REJECT,
        infinity_policy: str = NumericPolicy.INFINITY_REJECT,
        extra_column_policy: str = "ignore",
        numeric_impute_statistics: Mapping[str, float] = None,
    ):
        self.required_columns = list(required_columns)
        self.numeric_columns = list(numeric_columns)
        self.categorical_domains = {
            k: [str(x) for x in v] for k, v in categorical_domains.items()
        }
        self.senior_citizen_aliases = dict(senior_citizen_aliases)
        self.total_charges_policy = total_charges_policy
        self.total_charges_statistic = float(total_charges_statistic)
        self.numeric_missing_policy = numeric_missing_policy
        self.categorical_missing_policy = categorical_missing_policy
        self.infinity_policy = infinity_policy
        self.extra_column_policy = extra_column_policy
        # Per-numeric training statistics used to impute a missing value. Read
        # from the bundle's OWN fitted imputer, never recomputed from the
        # incoming batch, so a score is reproducible.
        self.numeric_impute_statistics = dict(numeric_impute_statistics or {})

    def prepare(self, customer_data) -> pd.DataFrame:
        """Enforce this contract and return a normalised, ordered frame.

        A method on the spec so any consumer holding only a contract (the XAI
        layer, the UI, the audit) enforces exactly the same rules as
        :meth:`backend.model_bundle.ModelBundle.prepare_input`.
        """
        return prepare_input_frame(customer_data, self)

    def describe(self) -> Dict[str, Any]:
        """Machine-readable description, for the audit and for the UI."""
        return {
            "required_columns": list(self.required_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_domains": {
                k: list(v) for k, v in self.categorical_domains.items()
            },
            "senior_citizen_accepted_representations": sorted(
                self.senior_citizen_aliases
            ),
            "total_charges_policy": self.total_charges_policy,
            "total_charges_statistic": self.total_charges_statistic,
            "numeric_missing_policy": self.numeric_missing_policy,
            "categorical_missing_policy": self.categorical_missing_policy,
            "infinity_policy": self.infinity_policy,
            "extra_column_policy": self.extra_column_policy,
        }



# ---------------------------------------------------------------------------
# Enforcement helpers. Each is small, pure and independently testable; each
# raises ContractError (never a bare sklearn/pandas error) so the failure mode
# is identical for v1 and v2.
# ---------------------------------------------------------------------------


def _is_missing(value) -> bool:
    """True for None/NaN/NaT/empty-string, without raising on arrays."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalise_senior_citizen(series: pd.Series) -> pd.Series:
    """Map every accepted ``SeniorCitizen`` representation to int64 0/1.

    Closes BLOCKER-01. A text ``'Yes'`` is accepted (v1's historical
    behaviour) *and* is normalised before v2's fitted median imputer ever sees
    it, so both bundles score the same row identically instead of v2 raising
    "Cannot use median strategy with non-numeric data".

    An unrecognised token, a NaN, or a non-0/1 number is rejected with the
    single :class:`ContractError` naming the offending value.
    """
    def _one(value):
        if _is_missing(value):
            raise ContractError(
                "Feature 'SeniorCitizen' is missing; accepted representations "
                f"are {sorted(SENIOR_CITIZEN_ALIASES)}."
            )
        token = str(value).strip().lower()
        if token in SENIOR_CITIZEN_ALIASES:
            return SENIOR_CITIZEN_ALIASES[token]
        # Numeric spellings that are not exactly 0/1 (e.g. 2, 0.5) are invalid.
        raise ContractError(
            f"Feature 'SeniorCitizen' received invalid value {value!r}; "
            f"accepted representations are {sorted(SENIOR_CITIZEN_ALIASES)} "
            "(0 and 1 are the only numeric values allowed)."
        )

    return series.apply(_one).astype("int64")



def reject_infinities(df: pd.DataFrame, spec: ContractSpec) -> pd.DataFrame:
    """Reject any ``+inf`` / ``-inf`` in a declared numeric column.

    Both bundles already refused infinities downstream (BLOCKER-06), but with
    sklearn's internal message at different depths. Doing it here makes the
    refusal explicit, uniform and attributable to the contract.
    """
    offenders = {}
    for col in spec.numeric_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(values.isin([float("inf"), float("-inf")]).sum())
        if n_bad:
            offenders[col] = n_bad
    if offenders:
        raise ContractError(
            "Non-finite numeric input is not allowed -> "
            + " | ".join(f"{col}: {n} infinite value(s)"
                         for col, n in offenders.items())
        )
    return df


def coerce_numeric_columns(df: pd.DataFrame, spec: ContractSpec) -> pd.DataFrame:
    """Apply the shared numeric policies, in a fixed, documented order.

    1. every declared numeric column is coerced with ``errors="coerce"`` so a
       text-numeric such as ``"70.5"`` is accepted (it is a valid number, and
       refusing it would hide a real value);
    2. a missing numeric is resolved by the bundle-shared policy
       (BLOCKER-03): impute it when this bundle persists a training statistic
       for that column (``TotalCharges``), otherwise reject it loudly;
    3. any *infinite* value is rejected (infinity policy / BLOCKER-06).

    The imputation statistic is the bundle's own persisted training value, so a
    score never depends on the composition of the incoming batch.
    """
    out = df.copy()
    stats = getattr(spec, "numeric_impute_statistics", None) or {}

    for col in spec.numeric_columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in spec.numeric_columns:
        if col not in out.columns or not out[col].isna().any():
            continue
        if spec.numeric_missing_policy in (
                NumericPolicy.MISSING_NUMERIC_IMPUTE,
                NumericPolicy.MISSING_NUMERIC_REJECT):
            if col in stats:
                out[col] = out[col].fillna(float(stats[col]))
                continue
            if spec.numeric_missing_policy == NumericPolicy.MISSING_NUMERIC_REJECT:
                raise ContractError(
                    f"Feature {col!r} is missing; this contract only imputes a "
                    f"missing value for columns that persist a training "
                    f"statistic ({sorted(stats)}). Supply {col!r} explicitly."
                )
        raise ContractError(
            f"Feature {col!r} is missing and no persisted training statistic "
            f"is available to impute it; refusing to guess."
        )

    if spec.infinity_policy == NumericPolicy.INFINITY_REJECT:
        reject_infinities(out, spec)
    return out


def validate_categorical_domains(df: pd.DataFrame,
                                 spec: ContractSpec) -> pd.DataFrame:
    """Enforce the shared categorical rules, loudly.

    Two distinct failures, two distinct messages, one exception type:

    * a **missing** categorical (NaN/None) is rejected outright
      (BLOCKER-05) -- v2's fitted imputer would have silently substituted the
      modal level, producing a confidently wrong score;
    * an **out-of-domain** categorical is rejected (BLOCKER-02) -- v2's
      ``handle_unknown="ignore"`` would have produced an all-zeros column.
    """
    errors = []

    for col, allowed in spec.categorical_domains.items():
        if col not in df.columns:
            continue
        values = df[col]

        if spec.categorical_missing_policy == NumericPolicy.MISSING_CATEGORICAL_REJECT:
            n_missing = int(values.isna().sum())
            if n_missing:
                errors.append(
                    f"Feature {col!r} received {n_missing} missing value(s); "
                    f"a missing category is rejected, not imputed "
                    f"(allowed values are {sorted(allowed)})."
                )

        bad = sorted({str(v) for v in values.dropna().unique()
                      if str(v) not in allowed})
        if bad:
            errors.append(
                f"Feature {col!r} received invalid value(s) {bad}; "
                f"allowed values are {sorted(allowed)}."
            )

    if errors:
        raise ContractError("Invalid categorical input -> " + " | ".join(errors))
    return df


def prepare_input_frame(customer_data, spec: ContractSpec) -> pd.DataFrame:
    """The ONE authoritative raw-input path for every bundle.

    Applied identically for v1 and v2, in this fixed order:

      1. materialise a ``DataFrame`` (dict -> single row) and type-check;
      2. **required columns** -- reject a missing one, naming it (unifies
         BLOCKER-10's differing error types);
      3. **extra columns** -- dropped, per the explicit contract policy, so a
         labelled dataset can be scored;
      4. ``SeniorCitizen`` -- normalised through the closed accepted set
         (BLOCKER-01);
      5. numerics -- coerced, then a missing value imputed with the bundle's
         persisted training statistic (BLOCKER-03); infinity rejected;
      6. categoricals -- missing and out-of-domain values both rejected loudly
         (BLOCKER-05, BLOCKER-02);
      7. re-index to exactly the contract's column order.

    Returns a frame carrying the contract's columns in the contract's order,
    already normalised, so the caller's fitted preprocessor is transform-only.
    """
    if isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    elif isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        raise ContractError(
            f"customer_data must be a dict or DataFrame, got "
            f"{type(customer_data).__name__}."
        )

    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        raise ContractError(
            f"Missing required field(s): {missing}. Required columns are "
            f"{list(spec.required_columns)}."
        )

    if spec.extra_column_policy == "ignore":
        df = df[list(spec.required_columns)]

    if "SeniorCitizen" in spec.required_columns:
        df["SeniorCitizen"] = normalise_senior_citizen(df["SeniorCitizen"])

    df = coerce_numeric_columns(df, spec)
    df = validate_categorical_domains(df, spec)

    return df[list(spec.required_columns)]
