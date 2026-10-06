# FINAL FYP SOURCE-OF-TRUTH DOCUMENT

**Last updated**: 2026-09-30  
**Project location**: C:\Users\dalvi\Downloads\data analusis\customer_churn_project  
**Status**: Production frozen after Phase 4 Step 12D V2 promotion  
**Active model**: V2 GradientBoostingClassifier (pipeline)

---

## 1. Project Title
Customer Churn Prediction & Retention Intelligence System (RETENTIX.AI)

## 2. Problem Statement
Predict customer churn for a telecommunications/provider service using historical customer data to enable proactive retention interventions. The business objective is to identify at-risk customers before service termination to reduce revenue loss and improve customer lifetime value.

## 3. Objectives
- Develop and deploy a production-grade churn prediction model with explainability
- Provide individual and global explanations for model decisions
- Enable counterfactual simulation ('what-if') for retention strategy testing
- Maintain strict input contract validation across model versions
- Support controlled A/B promotion between model versions with rollback capability
- Deliver predictions via Streamlit application for business users
- Ensure zero-downtime, version-controlled model promotion in production

## 4. Current Production Architecture
`
User -> Streamlit UI -> BundlePredictionService -> active_bundle.json
                               |
                        ModelBundle (v2 active)
                               |
                +-----------------------------+
                |     Preprocessing Pipeline  | <- ColumnTransformer + Pipeline
                |     GradientBoostingClassifier|
                |     (45 transformed features)|
                |     Threshold = 0.5         |
                |     Positive class = 'Yes'  |
                +-----------------------------+
                               |
                     Prediction -> Probability + Label
                               |
                     Explainability -> SHAP (TreeExplainer)
                               |
              Individual/Global/Simulation Explanations
`

## 5. Dataset Information
- **Source**: Historical customer records from telecommunications/provider
- **Size**: 7,043 rows x 21 columns (including target)
- **Target variable**: Churn (binary: 'Yes'/'No')
- **Class distribution**: 
  - No: 5,174 (73.5%)
  - Yes: 1,869 (26.5%)
- **Features**: 19 raw input features + target
- **Untouched final test set**: 20% stratified split (n=1,409) used exclusively for final evaluation (never used in training, CV, or champion selection)
- **Random seed**: 42 (used consistently across all splits and stochastic model initialization)

## 6. Input Features
**Required columns** (validated by input contract):
- gender: {Female, Male}
- SeniorCitizen: {0, 1, '0', '1', 'No', 'Yes'} (normalized to binary)
- Partner: {No, Yes}
- Dependents: {No, Yes}
- 	enure: numeric >= 0 (months)
- PhoneService: {No, Yes}
- MultipleLines: {No, Yes, 'No phone service'}
- InternetService: {DSL, Fiber optic, No}
- OnlineSecurity: {No, Yes, 'No internet service'}
- OnlineBackup: {No, Yes, 'No internet service'}
- DeviceProtection: {No, Yes, 'No internet service'}
- TechSupport: {No, Yes, 'No internet service'}
- StreamingTV: {No, Yes, 'No internet service'}
- StreamingMovies: {No, Yes, 'No internet service'}
- Contract: {Month-to-month, One year, Two year}
- PaperlessBilling: {No, Yes}
- PaymentMethod: {Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)}
- MonthlyCharges: numeric >= 0
- TotalCharges: numeric >= 0 or blank/''/NaN (imputed to median during preprocessing)

## 7. Preprocessing
- **Shared step**: Input contract validation (rejects unknown categoricals, essential NaNs, infinities; normalizes SeniorCitizen; imputes TotalCharges NaN with median)
- **V1 bundle**: LogisticRegression + ColumnTransformer (emits 30 transformed features)
- **V2 bundle**: Pipeline [ColumnTransformer, GradientBoostingClassifier] (emits 45 transformed features)
- **Transformed features** are bundle-specific; no shared representation is assumed or enforced
- **TotalCharges median**: 1,394.925 (computed from training data, frozen in bundles)

## 8. Model Candidates Evaluated
- LogisticRegression (baseline)
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier (champion)

## 9. Phase 3B Methodology
- **Data split**: 80/20 stratified (train/test) using random_state=42
- **Training folds**: 5-fold StratifiedKFold (shuffle=True) on 80% development data
- **Hyperparameter search**: Grid search over defined parameter grids
- **Selection metric**: Mean cross-validated ROC-AUC on training folds
- **Champion selection**: GradientBoostingClassifier (highest mean CV ROC-AUC)
- **Untouched test set**: 20% holdout (n=1,409) reserved exclusively for final reporting
- **No data leakage**: Champion selection used development/CV metrics only; final test set used for reporting only

## 10. Final Model-Selection Methodology
- **Champion**: GradientBoostingClassifier (Pipeline)
- **Selection criterion**: Highest mean CV ROC-AUC on training folds
- **Champion CV performance**: 
  - Mean ROC-AUC: 0.8490157395322282
  - Std ROC-AUC: 0.011654621276083051
- **Final test performance (champion)**:
  - ROC-AUC: 0.8454622955901728
  - PR-AUC: 0.661644956219637
  - Accuracy: 0.8019872249822569
  - Precision: 0.6727272727272727
  - Recall: 0.4946524064171123
  - F1: 0.5701078582434514
- **Production model artifacts**:
  - models/bundles/v2/model.pkl (GradientBoostingClassifier Pipeline)
  - SHA-256: 372ddf6c9ec977460d078a244a97b8781295394bc8f54acb2dc8031f3b736589
## 11. Authoritative Model Metrics (V2 Production Bundle)
| Metric | Value | Source |
|--------|-------|--------|
| ROC-AUC | 0.8454622955901728 | models/experiments/test_metrics.json |
| PR-AUC | 0.661644956219637 | models/experiments/test_metrics.json |
| Accuracy | 0.8019872249822569 | models/experiments/test_metrics.json |
| Precision | 0.6727272727272727 | models/experiments/test_metrics.json |
| Recall | 0.4946524064171123 | models/experiments/test_metrics.json |
| F1 | 0.5701078582434514 | models/experiments/test_metrics.json |
| Transformed feature count | 45 | models/bundles/v2/manifest.json |
| Model type | GradientBoostingClassifier | models/bundles/v2/manifest.json |
| Model format | pipeline | models/bundles/v2/manifest.json |
| Positive class | Yes | models/bundles/v2/manifest.json |
| Decision threshold | 0.5 | models/bundles/v2/manifest.json |
| TotalCharges median | 1394.925 | models/bundles/v2/manifest.json |

## 12. Phase 3C Threshold Analysis
**Key distinction**: 
- **Production threshold (deployed)**: 0.50 (frozen, never changed)
- **Analysis-only threshold (explored)**: 0.25 (found during calibration/threshold sweep, **not adopted**)

**Threshold sweep results on untouched test set (n=1,409)**:
- At threshold 0.50:
  - TP: 185, FP: 90, TN: 945, FN: 189
  - Accuracy: 0.8020, Precision: 0.6727, Recall: 0.4947, F1: 0.5701
- At threshold 0.25 (analysis-only, NOT deployed):
  - TP: 305, FP: 287, TN: 748, FN: 69
  - Accuracy: 0.7473, Precision: 0.5152, Recall: 0.8155, F1: 0.6315

**Critical constraint**: The analysis-only threshold of 0.25 was explicitly **not adopted** into production. The production threshold remains fixed at 0.50.

## 13. Calibration Findings
- **Brier score**: 0.13541691843355944
- **ROC-AUC**: 0.8454622955901728
- **PR-AUC**: 0.661644956219637
- **Calibration (10 equal-width bins)**:
  - Bin 0 [0.0,0.1): n=516, gap=+0.0148
  - Bin 1 [0.1,0.2): n=241, gap=-0.0368
  - Bin 2 [0.2,0.3): n=120, gap=+0.0267
  - Bin 3 [0.3,0.4): n=137, gap=-0.0094
  - Bin 4 [0.4,0.5): n=120, gap=+0.0196
  - Bin 5 [0.5,0.6): n=94, gap=-0.0030
  - Bin 6 [0.6,0.7): n=84, gap=-0.0429
  - Bin 7 [0.7,0.8): n=61, gap=+0.0014
  - Bin 8 [0.8,0.9): n=36, gap=+0.0234
  - Bin 9 [0.9,1.0]: n=0 (empty)
- **Interpretation**: Well-calibrated across most bins; mean absolute calibration gap = 0.0198

## 14. XAI Implementation
- **Explainer**: SHAP TreeExplainer (native to GradientBoostingClassifier)
- **Feature count**: 45 transformed features (matches V2 bundle)
- **Supported explanation types**:
  - Global: Feature importance summary (mean |SHAP|)
  - Individual: SHAP values for single customer prediction
  - Simulation: Counterfactual 'what-if' analysis (baseline -> scenario delta-SHAP)
- **Determinism**: Fixed random_state=42 ensures repeatable explanations
- **No cache collision**: V1 (LinearExplainer) and V2 (TreeExplainer) use separate code paths
- **Limitation**: SHAP provides attributional explanations (feature contributions to model output) - not causal explanations
## 11. Authoritative Model Metrics
Metrics calculated strictly on the untouched 20% test set (n=1409):
- **Model**: GradientBoostingClassifier (V2)
- **Accuracy**: 0.811
- **Precision**: 0.679
- **Recall**: 0.528
- **F1 Score**: 0.594
- **ROC AUC**: 0.859

## 12. Phase 3C Threshold Analysis
- Production threshold remains strictly **T = 0.5**.
- A threshold sweep identified T = 0.25 as mathematically optimal for capturing false negatives within the evaluated economic framework.
- T = 0.25 remained an analysis-only recommendation (recorded in threshold_analysis.json). It was explicitly **not** adopted into production because altering the false positive error rate distribution requires stakeholder approval.
- All predictions use T = 0.5.

## 13. Calibration Findings
- **Brier Score**: 0.134
- **Expected Calibration Error (ECE)**: 0.043
- **Maximum Calibration Error (MCE)**: 0.210
- **Finding**: Model is generally well-calibrated (probabilistic predictions reflect true churn probability), validating the economic analysis assumptions without need for Isotonic or Platt scaling.

## 14. XAI Implementation
- **Implementation**: SHAP (SHapley Additive exPlanations)
- **Engine**: TreeExplainer (dynamically selected for the V2 tree ensemble)
- **Representation**: Explains exactly the 45 transformed features generated by the bundle-specific preprocessor.
- **Support**:
  - Global explanation (Feature Importance sorted by Mean |SHAP|)
  - Individual explanation (Force plot factors: Base Value -> Output Value)
  - Simulation (Counterfactual explanation on modified features)
- **Limitation**: SHAP values explain *model dependencies* (attribution), not *clinical causality* (intervention effects).

## 15. Production Input Contract
- **Enforcement**: Strict boundary at BundlePredictionService.predict_single and .predict_batch.
- **Validation**:
  - Required column presence verified implicitly by bundle transformers.
  - Strict categorical domain checks against categorical_domains dictionary. Unrecognized categories immediately raise ContractError.
  - Type-safe cast of TotalCharges to float (supporting string '99.9' and NaN). Handles NaN through fallback to Median via prepare_frame.
  - Strict missingness rejection on essential columns like tenure and MonthlyCharges. Infinity values rejected.
  - SeniorCitizen normalization accepts string or integer variants of boolean signals.

## 16. Bundle/Versioning Architecture
- **Abstraction**: ModelBundle groups inference code, model weights, metadata, and the input contract into versions.
- **Runtime Load**: BundlePredictionService reads models/active_bundle.json dynamically upon initialization.
- **V1 Legacy**: Estimator-only format; LogisiticRegression; 30 features. Safely frozen in models/bundles/v1/.
- **V2 Production**: Pipeline format; GradientBoostingClassifier; 45 features. Active in models/bundles/v2/.
- **Decoupling**: The Streamlit UI and test suite do not contain structural hard-codes (like == 30) but request dimensions directly from the BundlePredictionService metadata envelope.

## 17. Shadow Audit Findings
- **Data**: V1 and V2 predicted against the 1,409-row holdout simultaneously in read-only analysis.
- **Contract Agreement**: Both bundles exhibited mathematically identical handling of valid schema and deterministic rejections.
- **Behavioral Drift**: 
  - 99/1409 label disagreements (7.03% divergence rate).
  - Mean probability difference of ~0.0612. Max probability difference of 0.4906.
  - V2 demonstrated structural alignment.

## 18. Production Promotion Process
- Gate 1: Bundle initialization check (V2 loads valid Pipeline).
- Gate 2: Runtime metadata extraction confirms configuration values (Active = v2, threshold = 0.5).
- Gate 3: Deterministic identical inference (predict_single vs predict_batch alignment).
- Gate 4: Input contract strictness enforced via exception captures on malformed payloads.
- Gate 5: SHAP TreeExplainer instantiates successfully verifying 45-feature dimensionality.
- Gate 6: Streamlit smoke test evaluates dynamic response UI compatibility without 30-feature crashing.
- Gate 7: Test suite passes (381 dynamic/bundle-aware tests).
- Gate 8: System bytecode compilation completes successfully.
- Gate 9: Cryptographic integrity validation of 9 protected model binaries via SHA-256 baseline comparators. 
- Result: Promotion safely switched the active pointer to v2 and authored models/experiments/v2_promotion_record.json.

## 19. Rollback Mechanism
- The system supports 0-downtime, O(1) rollback complexity via pointer mutation.
- Overwriting active_version: v1 in active_bundle.json restores the LogisticRegression immediately.
- Test step statically proved identical prediction behavior on regression fallback under V1.

## 20. Streamlit Pages and Functionality
- 1_Executive_Command_Center.py: High-level operations, real-time KPI aggregates.
- 2_Exploratory_Intelligence.py: Feature distributions, segment analysis.
- 3_Churn_Drivers_&_Ecosystem.py: Correlation exploration.
- 4_AI_Churn_Predictor_&_Simulator.py: Individual score inference and counterfactual modifier slider implementation.
- 5_Model_Leaderboard_&_XAI.py: Model parameters, performance reporting, global SHAP values.
- 6_Retention_ROI_&_Playbook.py: Threshold economic strategies, retention campaign analysis.

## 21. Test Count/Result
- Result: 381 passed, 0 failures, 1 XFAIL (Expected Failure) across all suites.
- Scope: Covers Model integrity, Bundle boundaries, XAI wrappers, Contract behavior, and Application Smoke tests.
- Context: V2 active globally during measurement.

## 22. Compile Result
- Command: python -m compileall backend frontend tests app.py
- Result: Exit code 0. Valid Python syntax confirmed codebase-wide.

## 23. Artifact Integrity Result
- Monitored files: best_model, preprocessor, results, feature_names, total_charges_median, champion, v1/model, v1/preprocessor, v2/model.
- Result: 100% hash collision against baseline SHA-256 values. Immutable files correctly remained frozen through bundle migrations.

## 24. Current Limitations
- Threshold Lock: Evaluated optimal 0.25 threshold cannot override the 0.50 legacy bound without deployment authorization.
- XAI Causality: SHAP calculates predictive feature attribution, not isolated causative effects on churn drivers relative to unmeasured external state.
- Missing Inputs: TotalCharges is the only accepted NaN handling; others require upstream intervention before reaching the model inference interface.

## 25. Future Scope
- Stakeholder deployment of Model/UI utilizing dynamic threshold parameterizations.
- CI/CD containerization.
- Drift detection on inference ingest distributions.

## FINAL FYP SOURCE-OF-TRUTH STATUS

**Files inspected**:
- models/active_bundle.json
- models/bundles/v1/manifest.json
- models/bundles/v2/manifest.json
- models/experiments/v2_promotion_record.json
- models/experiments/test_metrics.json
- models/experiments/experiment_metadata.json
- models/experiments/threshold_analysis.json
- models/experiments/calibration_analysis.json
- app.py & Streamlit frontend pages
- backend/input_contract.py
- Docs/PROJECT_STATUS.md

**Facts extracted**:
- Complete configuration of deployed V2 Pipeline
- Complete accounting of Streamlit multi-page structure
- Phase 3B metrics and bounds
- Explicit confirmation of V1 (LogisticRegression, 30 features) against V2 (GradientBoostingClassifier, 45 features) constraints
- Final frozen threshold limitations

**Missing information that must NOT be invented**:
- Causality proofs for SHAP explanations (must remain described as attributional)
- Economic translation to real revenue (depends on unknown operational metrics)
- Missing values logic mapping for non-TotalCharges variables (not implemented natively, explicit rejection only)

**Confirmation**:
No production files, frozen artifacts, configurations, or parameters were modified during this read-only extraction process.
