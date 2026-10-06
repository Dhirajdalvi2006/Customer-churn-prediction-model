import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline

from backend.input_contract import (
    ContractError,
    ContractSpec,
    SENIOR_CITIZEN_ALIASES,
    TOTAL_CHARGES_POLICY_IMPUTE_MEDIAN,
    prepare_input_frame,
)


class BundleInferenceError(ValueError):
    """Raised when a bundle cannot resolve a class/threshold from metadata.

    Kept as a distinct type so the "positive class must be derived, never
    assumed" rule (Phase 4 Step 8) fails loudly instead of silently falling
    back to ``predict_proba(...)[:, 1]``.
    """


class ModelBundle:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = os.path.realpath(bundle_dir)
        self.manifest_path = os.path.join(self.bundle_dir, "manifest.json")
        self.manifest = self._load_and_validate_manifest()
        
        self._verify_artifact_integrity()
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()
        
        self._validate_model_consistency()
        self._validate_model_format()

        # Phase 4 Step 8: resolve positive-class and threshold semantics from
        # metadata + the fitted artifact, ONCE, at load time. Nothing downstream
        # may hardcode ``predict_proba(...)[:, 1]``.
        self._resolve_class_semantics()
        self._resolve_threshold()

        # The shared input contract this bundle exposes to the application.
        self._contract = self._build_contract()

    # -- estimator / transformer accessors ---------------------------------

    @property
    def estimator(self):
        """The fitted classifier, unwrapped from a Pipeline if needed.

        This is the one place the two bundle *formats* are bridged. It unifies
        their call signature only -- never their feature representations.
        """
        if isinstance(self.model, Pipeline):
            steps = dict(self.model.named_steps)
            for key in ("classifier", "model"):
                if key in steps:
                    return steps[key]
            return self.model.steps[-1][1]
        return self.model

    @property
    def transformer(self):
        """The fitted ColumnTransformer for this bundle.

        v1 exposes it directly; v2 reaches it through the Pipeline's
        ``preprocessor`` step. Returns ``None`` only for a bundle that declares
        no preprocessor at all.
        """
        if isinstance(self.model, Pipeline):
            return self.model.named_steps.get("preprocessor")
        return self.preprocessor

    def _load_and_validate_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        
        # 1. Required fields
        required = ["schema_version", "bundle_version", "model_type", "artifacts", "model_format", 
                    "schema", "inference", "features", "metadata"]
        for r in required:
            if r not in manifest:
                raise ValueError(f"Missing required manifest field: {r}")
        
        # 2. Artifacts
        if "model" not in manifest["artifacts"]:
            raise ValueError("Missing 'model' artifact definition")
        for art in ["model", "preprocessor"]:
            if art in manifest["artifacts"]:
                for key in ["path", "sha256"]:
                    if key not in manifest["artifacts"][art]:
                        raise ValueError(f"Missing {key} for artifact {art}")
        
        # 3. Model format
        if manifest["model_format"] not in ["estimator_only", "pipeline"]:
            raise ValueError(f"Invalid model_format: {manifest['model_format']}")
            
        # 4. Inference
        inf = manifest["inference"]
        if not isinstance(inf.get("decision_threshold"), (int, float)) or not (0 <= inf["decision_threshold"] <= 1):
            raise ValueError("Threshold must be a numeric value in [0, 1]")
            
        # 5. Features
        feat = manifest["features"]
        if len(feat["transformed_features"]) != feat["transformed_feature_count"]:
            raise ValueError("Transformed feature count mismatch")
            
        return manifest

    def _verify_artifact_integrity(self):
        for art in ["model", "preprocessor"]:
            if art not in self.manifest["artifacts"]: continue
            rel_path = self.manifest["artifacts"][art]["path"]
            abs_path = os.path.realpath(os.path.join(self.bundle_dir, rel_path))
            
            # Path security check
            if not abs_path.startswith(self.bundle_dir + os.sep):
                raise SecurityError(f"Path traversal detected: {rel_path}")
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Artifact not found: {abs_path}")
                
            # Hash check. SHA256 hex digests are case-insensitive by
            # definition, so manifests may store them in either case
            # (PowerShell ``Get-FileHash`` emits uppercase, most tooling
            # emits lowercase). Compare normalised to avoid a spurious
            # mismatch on an otherwise byte-identical artifact.
            with open(abs_path, "rb") as fh:
                sha256 = hashlib.sha256(fh.read()).hexdigest()
            expected = str(self.manifest["artifacts"][art]["sha256"]).strip().lower()
            if sha256.lower() != expected:
                raise ValueError(f"SHA256 mismatch for {art}")

    def _load_model(self):
        path = os.path.join(self.bundle_dir, self.manifest["artifacts"]["model"]["path"])
        return joblib.load(path)

    def _load_preprocessor(self):
        if "preprocessor" not in self.manifest["artifacts"]:
            return None
        path = os.path.join(self.bundle_dir, self.manifest["artifacts"]["preprocessor"]["path"])
        return joblib.load(path)

    def _validate_model_consistency(self):
        inf = self.manifest["inference"]
        
        # Determine effective estimator (handle pipeline vs estimator)
        estimator = self.model
        if isinstance(self.model, Pipeline):
            # Try to find final step
            if len(self.model.steps) > 0:
                estimator = self.model.steps[-1][1]
        
        if hasattr(estimator, "classes_"):
            if list(estimator.classes_) != inf["classes"]:
                raise ValueError(f"Model classes {list(estimator.classes_)} mismatch manifest {inf['classes']}")

    def _validate_model_format(self):
        is_pipeline = isinstance(self.model, Pipeline)
        format = self.manifest["model_format"]
        
        if format == "pipeline" and not is_pipeline:
            raise ValueError("Model format declared 'pipeline' but model is not sklearn Pipeline")
        if format == "estimator_only" and is_pipeline:
            raise ValueError("Model format declared 'estimator_only' but model is sklearn Pipeline")
    # -- Phase 4 Step 8: metadata-driven class + threshold semantics --------

    #: Semantic aliases for the positive (churn) class. v1 encodes classes as
    #: ints 0/1 while v2 uses 'No'/'Yes'; both mean the same thing, so the
    #: positive class is resolved through this map against the *actual* fitted
    #: ``classes_`` -- never assumed to be column 1.
    #:
    #: Every churn spelling collapses to the single canonical value "Yes", and
    #: every non-churn spelling to "No". That is what makes BLOCKER-07 provably
    #: safe: even though the two bundles *encode* their classes differently,
    #: they are guaranteed to resolve the same semantic label, so a UI or an
    #: audit can compare them without a per-bundle special case.
    POSITIVE_CLASS_ALIASES = {
        "1": "Yes", "yes": "Yes", "true": "Yes",
        "0": "No", "no": "No", "false": "No",
    }

    def _resolve_class_semantics(self):
        """Derive the positive class and its column index from metadata.

        Validated against the fitted estimator's own ``classes_``; the manifest
        is never trusted blindly. Raises rather than defaulting to index 1, so
        a bundle whose positive class cannot be identified cannot be used.
        """
        inf = self.manifest["inference"]
        estimator = self.estimator

        classes = getattr(estimator, "classes_", None)
        if classes is None:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: the fitted "
                f"estimator exposes no classes_, so the positive class cannot "
                f"be validated from the artifact."
            )
        classes = list(classes)
        if len(classes) != 2:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: expected a binary "
                f"estimator, found {len(classes)} classes {classes}."
            )

        declared = [str(c) for c in inf["classes"]]
        actual = [str(c) for c in classes]
        if declared != actual:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: manifest classes "
                f"{declared} disagree with the fitted classes_ {actual}."
            )

        positive = inf.get("positive_class")
        if positive is None:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: manifest declares "
                f"no positive_class."
            )
        token = str(positive).strip().lower()
        if token not in self.POSITIVE_CLASS_ALIASES:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: positive_class "
                f"{positive!r} is not a recognised churn-class encoding "
                f"(expected one of {sorted(self.POSITIVE_CLASS_ALIASES)})."
            )
        semantic = self.POSITIVE_CLASS_ALIASES[token]

        matches = [i for i, c in enumerate(actual)
                   if self.POSITIVE_CLASS_ALIASES.get(str(c).strip().lower())
                   == semantic]
        if len(matches) != 1:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: positive class "
                f"{semantic!r} matches {len(matches)} entries of classes_ "
                f"{actual}; refusing to guess an index."
            )
        index = matches[0]

        declared_index = inf.get("positive_class_index")
        if declared_index is not None and int(declared_index) != index:
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: manifest "
                f"positive_class_index {declared_index} disagrees with the "
                f"index derived from classes_ ({index})."
            )

        self.classes = classes
        self.positive_class = classes[index]
        self.positive_class_index = int(index)
        self.negative_class_index = 1 - int(index)
        self.positive_semantic_label = semantic

    def _resolve_threshold(self):
        """Read the decision threshold from bundle metadata; never change it.

        The Phase 3C 0.25 analysis threshold is deliberately NOT adopted: this
        task must not change the threshold, and 0.5 remains the production
        value in both manifests.
        """
        inf = self.manifest["inference"]
        threshold = inf.get("decision_threshold")
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise BundleInferenceError(
                f"Bundle {self.manifest['bundle_version']!r}: decision_threshold "
                f"{threshold!r} is not a number in [0, 1]."
            )
        self.decision_threshold = float(threshold)    # -- Phase 4 Step 8: the shared input contract --------------------------

    def _numeric_impute_statistics(self):
        """The persisted training statistics the contract may impute with.

        Deliberately limited to ``TotalCharges`` -- the one numeric with a
        documented, persisted median in both manifests and the one v1 already
        imputes in production.

        v2''s fitted ``SimpleImputer`` also carries medians for ``tenure``,
        ``MonthlyCharges`` and ``SeniorCitizen``. Those are deliberately NOT
        exposed to the contract: imputing them would make v2 accept a missing
        numeric that v1 has always refused, which is exactly the silent
        behaviour change BLOCKER-03 was raised to prevent. The bundle may
        *retain* its own fitted imputer; the contract simply declines to rely
        on it. Keeping the v1 semantics is what makes the two bundles agree
        without editing a frozen artifact.
        """
        stats = {}
        median = self.manifest["inference"].get("total_charges_median")
        if median is not None:
            stats["TotalCharges"] = float(median)
        return stats

    def _fitted_numeric_imputer_statistics(self):
        """The bundle''s OWN fitted imputer medians, for reporting only.

        Exposed so the audit can *show* that v2 could have imputed a missing
        ``tenure`` (and deliberately does not, under the shared contract).
        Never used to fill a value on the inference path.
        """
        stats = {}
        transformer = self.transformer
        if transformer is None:
            return stats
        try:
            numeric = transformer.named_transformers_["num"]
        except (AttributeError, KeyError):
            return stats
        if hasattr(numeric, "named_steps"):
            numeric = numeric.named_steps.get("imputer")
        if numeric is None or not hasattr(numeric, "statistics_"):
            return stats
        cols = list(self.manifest["schema"]["numeric_cols"])
        for col, value in zip(cols, np.asarray(numeric.statistics_, dtype=float)):
            stats[col] = float(value)
        return stats

    def _build_contract(self):
        """Build the shared :class:`ContractSpec` this bundle exposes.

        Both bundles produce the SAME contract because it is derived from the
        shared manifest schema plus the shared alias set -- not from
        per-bundle special cases. Only the per-numeric imputation statistics
        differ, and those are the bundle's own persisted training values by
        design.
        """
        schema = self.manifest["schema"]
        return ContractSpec(
            required_columns=schema["required_cols"],
            numeric_columns=schema["numeric_cols"],
            categorical_domains=schema["categorical_domains"],
            senior_citizen_aliases=SENIOR_CITIZEN_ALIASES,
            total_charges_policy=TOTAL_CHARGES_POLICY_IMPUTE_MEDIAN,
            total_charges_statistic=float(
                self.manifest["inference"]["total_charges_median"]
            ),
            numeric_impute_statistics=self._numeric_impute_statistics(),
        )

    @property
    def contract(self) -> ContractSpec:
        """The shared input contract for this bundle."""
        return self._contract

    def prepare_input(self, customer_data) -> pd.DataFrame:
        """Validate + normalise raw input under the shared contract.

        The single entry point every consumer of a bundle must use, so a row is
        always interpreted identically no matter which bundle scores it.
        """
        return prepare_input_frame(customer_data, self._contract)

    # -- Phase 4 Step 8: BLOCKER-08 feature-representation transparency ------

    def feature_representation(self):
        """Self-describing description of this bundle's feature space.

        BLOCKER-08 is resolved by making the 30-vs-45 difference explicit and
        machine-readable rather than by forcing two frozen feature spaces to
        become identical. Consumers (XAI, the audit) read the count and names
        from here instead of assuming 30 columns.
        """
        feat = self.manifest["features"]
        return {
            "bundle_version": self.manifest["bundle_version"],
            "transformed_feature_count": int(feat["transformed_feature_count"]),
            "transformed_features": list(feat["transformed_features"]),
            "one_hot_drop": self._one_hot_drop(),
        }

    def _one_hot_drop(self):
        """The encoder's ``drop`` setting, read from the fitted artifact.

        v1 uses ``drop='first'`` (a reference baseline per categorical, 30
        columns); v2 uses ``drop=None`` (every level, 45 columns). Reported, not
        changed -- reconciling them would require refitting, which is forbidden.
        """
        transformer = self.transformer
        if transformer is None:
            return None
        try:
            cat = transformer.named_transformers_["cat"]
        except (AttributeError, KeyError):
            return None
        if hasattr(cat, "named_steps"):
            cat = cat.named_steps.get("onehot")
        return getattr(cat, "drop", None)

    def transform(self, customer_data):
        """Contract-validate, then run the bundle's own fitted preprocessor.

        Transform-only: nothing is ever fitted or refitted here.
        """
        frame = self.prepare_input(customer_data)
        return self.transformer.transform(frame)

    def predict_proba(self, customer_data):
        """Probability of the POSITIVE class, selected by validated index.

        The positive column is ``self.positive_class_index``, derived from
        ``classes_`` at load time -- never a hardcoded 1.
        """
        matrix = self.transform(customer_data)
        proba = self.estimator.predict_proba(matrix)
        return np.asarray(proba)[:, self.positive_class_index]

    def predict_proba_both(self, customer_data):
        """``(negative_prob, positive_prob)`` in class order, for the UI."""
        matrix = self.transform(customer_data)
        proba = np.asarray(self.estimator.predict_proba(matrix))
        return (proba[:, self.negative_class_index],
                proba[:, self.positive_class_index])

    def predict_labels(self, customer_data):
        """Threshold the positive probability at the bundle's own threshold."""
        positive = self.predict_proba(customer_data)
        return np.where(positive >= self.decision_threshold, "Yes", "No")

    def describe_contract(self):
        """Full self-description: contract + class + threshold + features."""
        return {
            "bundle_version": self.manifest["bundle_version"],
            "model_format": self.manifest["model_format"],
            "estimator_type": type(self.estimator).__name__,
            "classes": [str(c) for c in self.classes],
            "positive_class": str(self.positive_class),
            "positive_class_index": self.positive_class_index,
            "positive_semantic_label": self.positive_semantic_label,
            "decision_threshold": self.decision_threshold,
            "contract": self._contract.describe(),
            "features": self.feature_representation(),
        }

class BundleLoader:
    @staticmethod
    def get_active_bundle(models_dir: str = "Models") -> ModelBundle:
        active_path = os.path.join(models_dir, "active_bundle.json")
        with open(active_path, "r") as f:
            active = json.load(f)
        
        bundle_dir = os.path.join(models_dir, "bundles", active["active_version"])
        return ModelBundle(bundle_dir)
