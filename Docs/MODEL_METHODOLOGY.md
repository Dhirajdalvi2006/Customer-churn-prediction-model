# Model Methodology

## Phase 3A: Audit Findings
An initial audit of the historical project state (Phase 3A) identified that the model artifact in production (`LogisticRegression`) was severely over-written by ad-hoc scripts (`app.py`, `backend/model.py`). Models were being repeatedly fitted over the entirety of the dataset without sequestering a proper test set, making all accuracy claims optimistic. The production model lacked a formal artifact trail mapping code versions to byte-files.

## Phase 3B: Experimental Methodology
To resolve structural unreliability, Phase 3B instituted a strict separation of cross-validation (CV) vs. final test sets.
- **Dataset Split:** The historical dataset was rigorously split (80% development, 20% test). `test_size=0.2, random_state=42`. Stratification by target `Churn` was enforced.
- **Cross-Validation:** 5-fold `StratifiedKFold` evaluated four candidate models (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`).
- **Champion Selection:** The champion was strictly selected by the highest mean ROC AUC on the 80% CV folds.
- **Results:** `GradientBoostingClassifier` prevailed on CV. The 20% test set was finally unlocked to evaluate its true untainted performance (AUC 0.8529). At this point, `champion.pkl` was permanently frozen.

## Phase 3C: Threshold and Calibration Analysis
- **Threshold:** Operational sensitivity was evaluated by sliding the decision boundary. A threshold of 0.25 dramatically improved recall, capturing 82% of actual churners compared to 54% at the default 0.50 threshold.
- **Adoption Block:** The 0.25 threshold was explicitly restricted to analysis-only. Altering the threshold mid-flight fundamentally invalidates prior metrics without stakeholder realignment. It remains in `models/experiments/threshold_analysis.json` for future business ROI discussions.
- **Calibration:** `champion.pkl` was quantified against a `CalibrationDisplay`. With an Expected Calibration Error (ECE) of ~0.038, the probabilistic output was declared reliable.

## Phase 4: Production Readiness and Architecture Realignment
Despite the champion being scientifically validated in Phase 3B, a Phase 4 architectural review revealed a critical mismatch: the production architecture was structurally hard-coded for a 30-feature `LogisticRegression` (V1). The champion was a 45-feature `Pipeline` (V2). Direct file substitution would cause catastrophic inference failure.

### Bundle Architecture (`backend/model_bundle.py`)
To solve the mismatch, the architecture was elevated to a **Bundle** abstraction.
- A bundle is an autonomous version directory (`models/bundles/v1`, `v2`) containing its specific binary artifacts, a declarative `manifest.json` outlining format (`pipeline` vs `estimator_only`), threshold, and metadata.
- A lightweight pointer (`active_bundle.json`) determines the active production target.

### Shadow Audit (`backend/shadow_audit.py`)
Before promotion, a highly controlled shadow scoring audit was executed:
- Both `v1` and `v2` models were loaded side-by-side.
- The shared 20% final test set was identically pushed through both.
- This proved that the inputs strictly adhered to the schema, and the exact probability differentials (and class semantic alignments) were fully understood and predictable.

### XAI Abstraction (`backend/explainability.py`)
The legacy XAI relied on `LinearExplainer` bounded to 30 features. A factory abstraction (`_as_model_view`) was implemented so SHAP could dynamically adapt to either bundle, ultimately routing V1 to `LinearExplainer` and V2 to `TreeExplainer` over 45 features identically securely.

### Promotion and Rollback Verification
Once the test suite was upgraded to dynamically query the active pointer (rather than hardcoding `v1`), the controlled promotion was executed (`active_bundle.json` -> `v2`).
- A rollback test verified that toggling the pointer instantly reinstates `v1` inference.
- Production is permanently unified on `V2` without overwriting historical byte-files.