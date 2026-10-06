# RETENTIX AI: Final FYP Presentation

## Slide 1 — Title
### Key Message
Explainable Customer Churn Prediction and Retention Intelligence System.
### Slide Content
- Project: RETENTIX AI
- Explainable Customer Churn Prediction & Retention Intelligence
- Final Year Project
### Visual
[Title Slide Placeholder]
### Speaker Notes
Welcome. This presentation covers the development, validation, and productionization of RETENTIX AI, an explainable system for customer churn prediction.

## Slide 2 — Problem Statement
### Key Message
Reactive retention is inefficient; predictive, explainable intelligence is required.
### Slide Content
- High churn rates impact profitability.
- Reactive retention (after churn) misses opportunities.
- Predictive models need to be explainable for stakeholder trust.
### Visual
[Problem/Opportunity Diagram]
### Speaker Notes
Customer churn is a critical challenge. Reactive strategies are too late. RETENTIX provides predictive intelligence to allow proactive intervention.

## Slide 3 — Objectives
### Key Message
Build a production-ready, explainable, and validated churn prediction system.
### Slide Content
- Develop accurate churn prediction models.
- Implement explainability (SHAP).
- Create an end-to-end production pipeline.
- Ensure rigorous model validation.
- Build a user-friendly retention dashboard.
### Visual
[Objectives Table]
### Speaker Notes
Our objectives were to move from reactive to predictive, ensuring the system is explainable, rigorously validated, and production-ready.

## Slide 4 — Proposed Solution
### Key Message
End-to-end, explainable, and production-hardened retention intelligence.
### Slide Content
- Data Preprocessing & Validation
- ML Prediction (Gradient Boosting)
- Explainability (SHAP)
- Simulation & Retention Intelligence
- Production-hardened deployment
### Visual
[Solution Pipeline Diagram]
### Speaker Notes
RETENTIX is an end-to-end system, from input validation and prediction to simulation and actionable retention intelligence, built on hardened infrastructure.

## Slide 5 — System Architecture
### Key Message
Modular, bundle-aware architecture for production reliability.
### Slide Content
- Streamlit UI
- Prediction Service
- Active Bundle Pointer
- ModelBundle (Contract, Preprocessing, Model)
- XAI Layer (TreeExplainer)
### Visual
[ASCII Architecture Diagram]
### Speaker Notes
Our architecture is decoupled. The UI talks to a Prediction Service, which loads an Active Bundle (v2). XAI is a separate layer using the bundle's estimator.

## Slide 6 — Dataset & Features
### Key Message
Standardized dataset for robust churn prediction.
### Slide Content
- Verified final dataset (n=1409 test rows).
- 45 transformed features (V2).
- Key categories: Contract, Tenure, Charges.
- Target: Churn.
### Visual
[Feature Structure Table]
### Speaker Notes
We used a standardized dataset. V2 utilizes 45 transformed features, ensuring robustness by encoding categorical variables appropriately for Gradient Boosting.

## Slide 7 — ML Methodology
### Key Message
Rigorous, leak-free validation strategy.
### Slide Content
- 80/20 Stratified Split.
- 5-Fold Stratified CV on Development.
- Champion selection on CV performance.
- Untouched final test set.
### Visual
[Methodology Workflow]
### Speaker Notes
We prioritized rigorous validation: an 80/20 split, 5-fold CV for model selection, and crucially, keeping the final test set untouched until the end.

## Slide 8 — Model Comparison
### Key Message
Gradient Boosting selected as the champion model.
### Slide Content
- Logistic Regression (V1).
- Decision Tree.
- Random Forest.
- Gradient Boosting (V2 - Champion).
### Visual
[Model Comparison Table]
### Speaker Notes
We evaluated multiple models. Gradient Boosting (V2) demonstrated superior performance characteristics in cross-validation, leading to its selection.

## Slide 9 — Final Production Model
### Key Message
V2 is the current frozen production model.
### Slide Content
- Model: GradientBoostingClassifier
- Format: Self-contained Pipeline
- Transformed Features: 45
- Positive Class: Yes
- Threshold: 0.5
### Visual
[Model Identity Badge]
### Speaker Notes
V2, our Gradient Boosting Pipeline, is the frozen production model, replacing V1 (Logistic Regression).

## Slide 10 — Model Performance
### Key Message
Authoritative performance metrics demonstrate robustness.
### Slide Content
- CV ROC-AUC: 0.852 (±0.012)
- Untouched Test ROC-AUC: 0.859
- Accuracy: 0.811
- Brier Score: 0.138
### Visual
[Metrics Table]
### Speaker Notes
Metrics are consistent. The test set performance confirms the robustness indicated during cross-validation.

## Slide 11 — Explainable AI
### Key Message
SHAP provides attribution, not causality.
### Slide Content
- TreeExplainer for V2.
- Global, Individual, and Simulation explanations.
- Attribution only (no causality).
- Consistent with 45 features.
### Visual
[SHAP workflow]
### Speaker Notes
We use TreeExplainer for SHAP. It provides model attribution—showing how features drive probability—but we emphasize this is NOT causal.

## Slide 12 — Threshold & Calibration
### Key Message
Production threshold is 0.5.
### Slide Content
- Production Threshold: 0.50.
- Analysis-only (Phase 3C): 0.25 investigated.
- Threshold changes require stakeholder acceptance.
### Visual
[Threshold Context Diagram]
### Speaker Notes
Our production threshold is 0.5. While 0.25 was analyzed, it was not deployed, as threshold changes impact precision-recall and require business approval.

## Slide 13 — RETENTIX Dashboard
### Key Message
Actionable insights for retention intelligence.
### Slide Content
- 6 Pages: Command Center, Exploratory, Drivers, Predictor, Leaderboard, ROI.
- Real-time simulation.
- Explainable churn risk.
### Visual
[Dashboard Screenshot Placeholders]
### Speaker Notes
The dashboard is functional and intuitive, providing everything from churn drivers and simulators to retention playbooks.

## Slide 14 — Productionization & Reliability
### Key Message
Hardened deployment through versioned bundles and audit.
### Slide Content
- V1 (Legacy) → V2 (Production).
- Bundle Architecture (Manifests, SHA-256).
- Promotion Record & Shadow Audit.
- Rollback Verified.
### Visual
[Bundle Architecture Diagram]
### Speaker Notes
We moved from V1 to V2 using versioned bundles and a shadow audit to ensure safety, with rollback verified.

## Slide 15 — Testing & Validation
### Key Message
Zero-failure production validation.
### Slide Content
- Tests Passed: 381.
- Failures: 0.
- XFAIL: 1.
- Artifact Integrity: 9/9 Unchanged.
### Visual
[Validation Status Table]
### Speaker Notes
Our suite passed 381 tests with zero failures, and artifact integrity is strictly maintained.

## Slide 16 — Conclusion & Future Scope
### Key Message
RETENTIX is a validated, explainable, and ready-to-use retention system.
### Slide Content
- Conclusion: Robust churn prediction solution.
- Future Scope: (As verified in report).
### Visual
[Summary Graphic]
### Speaker Notes
RETENTIX is a solid, production-hardened system. Future scope includes further dashboard expansion based on feedback.

# LIVE DEMO PLAN
1. Open Executive Command Center
2. Show dataset overview
3. Navigate to Churn Drivers
4. Select a customer
5. Generate churn probability
6. Show XAI explanation
7. Modify simulator inputs
8. Show prediction change
9. Show Model Leaderboard/XAI
10. Show Retention ROI
11. Show production bundle/version info
