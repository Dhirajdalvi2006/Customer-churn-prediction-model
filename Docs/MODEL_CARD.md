# V2 Model Card: GradientBoostingClassifier

## 1. Model Identity
- **Bundle Version:** v2
- **Model Type:** GradientBoostingClassifier
- **Model Format:** Pipeline (scikit-learn `Pipeline`)
- **Active Status:** ACTIVE (Production)
- **Positive Class:** "Yes"
- **Decision Threshold:** 0.50
- **Transformed Feature Count:** 45

## 2. Methodology
- **Phase 3B Methodology:** Strict experimental separation.
- **Split:** 80% development / 20% untouched final test set (stratified by target).
- **Cross-Validation:** 5-fold `StratifiedKFold` executed strictly on the 80% development data.
- **Model Candidates Evaluated:**
  - LogisticRegression
  - DecisionTreeClassifier
  - RandomForestClassifier
  - GradientBoostingClassifier
- **Champion-Selection Rule:** The model with the highest average cross-validated ROC AUC score on the development set was selected as the champion.
- **Data Leakage Prevention:** The 20% final test set was completely sequestered during all fitting and champion selection. It was only ever evaluated *after* the champion artifact (`models/experiments/champion.pkl`) was definitively frozen.

## 3. Final V2 Evaluation Metrics
*Computed on the 20% untouched final test set at the default 0.5 threshold.*
- **ROC AUC:** 0.8455
- **F1 Score:** 0.5701
- **Precision:** 0.6727
- **Recall:** 0.4947

## 4. Phase 3C Threshold Analysis
- **Production Threshold:** **0.50** (Unchanged).
- **Analysis-Only Threshold (0.25):** The Phase 3C threshold analysis demonstrated that lowering the threshold to 0.25 increases recall (from ~0.54 to ~0.82) at the cost of precision (from ~0.67 to ~0.44) and dramatically changes the business economics (increasing false positives/retention costs). 
- **Adoption:** This 0.25 threshold was explicitly **not** adopted into production. It remains an analysis-only artifact for future operational consideration. Production remains strictly calibrated to 0.5.

## 5. Calibration
- **ECE (Expected Calibration Error):** 0.038
- **Calibration Status:** Well-calibrated. The probabilistic outputs of the GradientBoosting pipeline reliably index against empirical churn incidence, making probabilities safe to use for ranking and expected-value calculations.

## 6. Explainability (XAI)
- **Explainer:** SHAP `TreeExplainer`
- **Feature Space:** Explanations operate over the **45 transformed features** emitted natively by the V2 pipeline's internal preprocessor.
- **Capabilities:** Fully supports real-time Individual explainer (waterfall), Simulation explainer (baseline vs counterfactual), and Global explainer (summary).
- **Limitation (Attribution vs Causality):** SHAP values represent *predictive attribution* (how much a feature contributed to the model's score) purely observationally. They do *not* represent physiological *causality* — simulating a feature change (e.g., tenure) illustrates the model's correlation landscape, but does not guarantee the customer will physically change their behavior if the business intervenes.

## 7. Input Contract
- **Required Columns:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Categorical Validation:** Unknown/unseen categorical values (e.g., `Contract="Three year"`) are explicitly rejected by the schema validation layer before inference. `NaN` in categorical fields is rejected.
- **SeniorCitizen Supported Forms:** Accurately accepts integer `0`/`1` and string `"Yes"`/`"No"`.
- **Numeric Missing Values:** `TotalCharges` dynamically accepts `NaN` (imputed with median: 1397.475). `tenure` and `MonthlyCharges` reject `NaN`.
- **Numeric Types:** Accepts standard floats, stringifiable numbers. Explicitly rejects `Infinity` / `-Infinity`.

## 8. Deployment Architecture
- `models/active_bundle.json` points to the active pointer (`"v2"`).
- `backend.model_bundle.BundleLoader` reads the pointer and dynamically loads the corresponding bundle manifest, fitted model, and metadata.
- `BundlePredictionService` acts as the service boundary, securely resolving the bundle's `model_format` (pipeline), threshold, and class semantics.
- Streamlit application reads all metadata dynamically from `BundlePredictionService` (no hardcoded feature counts or model names).

## 9. Versioning and Rollback
- The legacy `v1` bundle (LogisticRegression, 30 features, estimator_only) remains completely frozen and undeleted in `models/bundles/v1`.
- The new `v2` bundle resides in `models/bundles/v2`.
- **Rollback:** Rollback to V1 requires exactly one operational edit: changing `active_bundle.json` to `{"active_version": "v1"}`. No artifacts need to be moved, and Streamlit requires no restarts as it actively observes the pointer boundary on new service lookups.

## 10. Security / Limitations
- **Analysis-Only Threshold Guardrail:** The 0.25 threshold is documented in `threshold_analysis.json` but specifically blocked from overriding the production 0.5 threshold.
- No cache collisions: Testing verifies V1 and V2 models do not collide in SHAP or Service caches when switching.
