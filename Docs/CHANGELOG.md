# Project Changelog

## Phase 1
- **Discovery**: Baseline Logistic Regression pipeline implemented to establish target `Churn`.

## Phase 1B
- **Prediction Pipeline Unification**: Unification of ad-hoc application prediction endpoints into a single repeatable inference contract. Removed duplicated `app.py` training calls.

## Phase 2
- **SHAP/XAI Integration**: Added operational explainers (Global, Individual, Simulation) via SHAP `LinearExplainer`, constrained to the original 30 transformed features.

## Phase 3A
- **Model-Selection Audit**: Documented critical defect in baseline iteration. The historical models were fit dynamically across the entire dataset without holding out a final test set, meaning no trustworthy accuracy bounds existed.

## Phase 3B
- **Cross-Validated Champion Selection**:
  - Stabilized dataset split (80% development, 20% test).
  - 5-fold CV enacted across four model classes.
  - Test set explicitly sequestered during CV.
  - `GradientBoostingClassifier` natively bundled with preprocessing (45 features) outperformed legacy configurations to become the official frozen experimental champion.

## Phase 3C
- **Threshold & Calibration Analysis**:
  - Swept output distributions mapping false-positive / false-negative constraints.
  - Proved setting a threshold of 0.25 captured 82% of churners vs 54% capture at default threshold 0.5.
  - Frozen as analysis-only (not deployed) due to business ROI implications. Calibration bounds formally documented in `experiments/calibration_analysis.json`.

## Phase 4
- **Production Readiness & Model Bundle Architecture (V2)**:
  - Designed generalized `ModelBundle` artifact structure mapping to legacy `v1` and new `v2`.
  - Implemented schema-strict Input Contract validating real application inputs independently from pipeline formats.
  - Abstracted the XAI core through `ModelView` to allow `TreeExplainer` on the 45-feature format and `LinearExplainer` on the 30-feature format concurrently.
  - Completed Read-Only Shadow Audit (`backend.shadow_audit`), successfully mapping input boundaries completely.
  - Executed controlled Promotion, switching application pointer to **V2** permanently.
  - Rolled back to V1 logically, then re-promoted safely to establish rollback methodology.
  - Upgraded test harnesses to verify invariants fluidly against whichever bundle is active.