# FINAL FYP CONSISTENCY AUDIT

## 1. Overall Status

PASS WITH MINOR ROUNDING DIFFERENCES

## 2. Production Identity Audit
All documents consistently identify the production model as V2 (GradientBoostingClassifier, pipeline, 45 features, threshold 0.5). No documents claim LogisticRegression as active.

## 3. Metric Consistency Audit

| Metric | Source Value | Report | Viva | Status |
| :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.80198 | 0.8020 | 0.8020 | ROUNDING DIFFERENCE |
| ROC-AUC | 0.84546 | 0.8455 | 0.8455 | ROUNDING DIFFERENCE |
| F1 Score | 0.5701 | 0.5701 | 0.5701 | PASS |

## 4. Threshold Audit
Production threshold is 0.5. All documents correctly identify 0.25 as an analysis-only threshold investigated in Phase 3C and not deployed.

## 5. XAI Audit
Consistent usage of TreeExplainer for V2 (45 features). All documents correctly state SHAP provides attribution, not causality.

## 6. Methodology Audit
Consistent reporting of 80/20 stratified split, 5-fold CV, and untouched final test set usage.

## 7. V1/V2 Audit
Consistent distinction between V1 (LogisticRegression, 30 features) and V2 (GradientBoostingClassifier, 45 features).

## 8. Input Contract Audit
Description of required columns and validation logic in documentation matches `backend/input_contract.py`.

## 9. Architecture Audit
Documentation accurately reflects the Streamlit -> Prediction Service -> Active Bundle -> Pipeline architecture.

## 10. Testing Audit
Documentation consistently reports 381 passed tests, 0 failures, 1 XFAIL.

## 11. Production Integrity Audit
V2 active, SHA-256 hashes match baseline.

## 12. Unsupported Claim Audit
No unsupported claims ("best", "guaranteed") found in frozen documentation.

## 13. Presentation Audit
*Inconsistency Found*: `Docs/RETENTIX_FINAL_PRESENTATION.md` is missing from the repository.

## 14. Viva Audit
Viva preparation guide is consistent with report and implementation.

## 15. Application/UI Audit
UI components in documentation match the 6-page implementation.

## 16. Required Corrections
1. Create `Docs/RETENTIX_FINAL_PRESENTATION.md` which is currently missing.

## 17. Final Submission Readiness
The project is internally consistent, with the exception of the missing presentation file.
