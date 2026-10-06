
# RETENTIX VIVA PREPARATION GUIDE

This guide is derived strictly from the frozen project documentation and verified production state.

---

## SECTION 1 — 30-SECOND PROJECT INTRODUCTION

**Q: "What is your project?"**

**A:** "My project, RETENTIX AI, is an explainable customer churn prediction and retention intelligence system. It leverages machine learning to identify customers at risk of churning, providing not just churn probability scores, but also actionable explanations through SHAP-based XAI. The system is deployed with a production-grade bundle architecture to ensure reliable, versioned predictions via a Streamlit dashboard, helping businesses move from reactive to proactive retention strategies."

---

## SECTION 2 — 1-MINUTE PROJECT EXPLANATION

**Q: "Explain your project in detail."**

**A:** "RETENTIX AI transforms raw customer data into actionable retention intelligence. The pipeline validates input data against a strict production contract, scores it using a validated GradientBoostingClassifier pipeline, and generates churn probabilities. Crucially, the system uses SHAP to provide global and individual feature-importance explanations, alongside a simulator that lets users test 'what-if' scenarios. Everything is managed via a versioned 'ModelBundle' architecture that ensures production reliability, complete with automated rollback capabilities and a comprehensive dashboard for monitoring, simulation, and planning retention campaigns."

---

## SECTION 3 — CORE VIVA QUESTIONS (SAMPLE)

### Q: Why did you select Gradient Boosting?
*   **Short Answer:** It provided the highest mean cross-validated ROC-AUC on development folds compared to Logistic Regression, Decision Tree, and Random Forest models.
*   **Detailed Answer:** During Phase 3B, candidate models were evaluated using 5-fold Stratified Cross-Validation on the development set. The GradientBoostingClassifier consistently demonstrated superior generalization performance based on ROC-AUC, meeting the selection criteria established for the champion.
*   **Key Point:** It was selected based on a documented, data-driven validation procedure.
*   **Source:** `experiment_metadata.json` / `test_metrics.json`

---

## SECTION 4 — ML METRICS (AUTHORITATIVE)

| Metric | GradientBoostingClassifier (Untouched Test) |
| :--- | :--- |
| ROC-AUC | 0.8455 |
| PR-AUC | 0.6616 |
| Accuracy | 0.8020 |
| Precision | 0.6727 |
| Recall | 0.4947 |
| F1 | 0.5701 |

*Note: Champion selection was based on CV ROC-AUC (mean: 0.8490).*

---

## SECTION 5 — THRESHOLD & CALIBRATION

**Q: Why is your production threshold 0.5?**
**A:** 0.5 is the standard classification threshold used for deployment in the current production environment. While Phase 3C analysis explored a 0.25 threshold as an analysis-only operating point to improve recall, it was not deployed, as threshold changes necessitate formal stakeholder and business-economic acceptance regarding the resulting shift in precision-recall trade-offs.

---

## SECTION 6 — SHAP / XAI

**Q: Does SHAP prove causality?**
**A:** No. SHAP provides model attribution, not causal proof. It identifies which features the model relied upon to generate a specific prediction, but it does not imply that changing those features will causally result in a different churn outcome in reality.

---

## SECTION 7 — PRODUCTIONIZATION (V1 → V2)

**Q: What is a ModelBundle?**
**A:** A ModelBundle is the versioned container for all artifacts required for prediction (model, preprocessor, manifest, and metadata). It encapsulates the entire pipeline, ensuring that the model is always served with the correct preprocessing logic it was trained with, preventing inference defects.

---

## SECTION 8 — "DON'T SAY THIS" GUIDE

*   **DO NOT SAY:** "Our model guarantees churn prediction."
*   **SAY:** "Our model estimates churn probability based on historical patterns."
*   **DO NOT SAY:** "SHAP proves why the customer churned."
*   **SAY:** "SHAP identifies model-attributed feature contributions."
*   **DO NOT SAY:** "V2 is definitely better than V1."
*   **SAY:** "V2 was promoted through the documented production-readiness validation process."

---

## SECTION 9 — FINAL 2-MINUTE DEMO SCRIPT

1.  **Problem:** reactive retention is ineffective.
2.  **Solution:** RETENTIX AI predictive retention intelligence.
3.  **Prediction:** Score customer churn probability.
4.  **Explainability:** Use SHAP to understand *why* a customer is high-risk.
5.  **Dashboard:** Execute retention ROI/playbook planning.
6.  **Production Architecture:** Bundled, versioned, and validated deployment.
7.  **Result:** V2 deployed Gradient Boosting Pipeline.
8.  **Conclusion:** Proactive churn management platform.

---

## SECTION 10 — VIVA QUESTIONS TO PREPARE

(Selected Questions)
1. Why Gradient Boosting?
2. Why 5-fold CV?
3. Why untouched final test data?
4. Why threshold 0.5 vs 0.25?
5. Why SHAP (attribution vs causality)?
6. Why 45 features in V2 vs 30 in V1?
7. What is ModelBundle?
8. Why V1 is retained (rollback)?
9. How was production parity verified?
10. What is an input contract?

---

# LIVE DEMO PLAN (5-7 Minutes)

1. **Executive Command Center:** Briefly introduce the system.
2. **AI Churn Predictor & Simulator:** Select a customer, predict churn, show high-risk flags.
3. **Explainability:** Show SHAP bar chart for individual risk factors.
4. **Simulator:** Change customer tenure/total charges, demonstrate probability shift.
5. **Dashboard Navigation:** Briefly show pages 1, 2, 3, 5, 6 as proof of full functionality.
6. **Production/Rollback Info:** Show the metadata panel confirming Version v2 is active and V1 exists.

---

PHASE 5 STEP 4 COMPLETE — VIVA GUIDE CREATED
