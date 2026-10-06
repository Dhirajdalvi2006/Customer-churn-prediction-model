# RETENTIX AI: Explainable Customer Churn Prediction and Retention Intelligence System

## 1. TITLE PAGE

**Project Title:** RETENTIX AI: Explainable Customer Churn Prediction and Retention Intelligence System
**Academic Report — Final Year Project**

## 2. ABSTRACT

Customer churn poses a significant operational continuity and revenue risk in competitive business environments. Identifying at-risk customers prior to attrition and understanding the underlying drivers are critical for targeted retention efforts. This project presents RETENTIX AI, an end-to-end, explainable predictive analytics system engineered to evaluate churn probabilities dynamically while offering rigorous input validation, architectural separation, and strict bundle versioning boundaries. The deployed V2 champion model utilizes a self-contained Pipeline encompassing preprocessing and a GradientBoostingClassifier architecture. Operating upon a 45-feature transformed space, it achieves 81.1% accuracy and an ROC-AUC of 0.859 against a strictly partitioned unseen test set. Explainability frameworks utilize SHAP TreeExplainer attributions to present global, individual, and simulated churn-risk scenarios. Furthermore, the system includes a modular shadow audit mechanism, SHA-256 binary validation, and an interactive Streamlit intelligence application representing modern MLOps principles.

## 3. INTRODUCTION

### 3.1 Background
The widespread adoption of predictive analytics has shifted retention strategies in telecommunications and subscription-based service models from reactive interventions to proactive engagement. Churn prediction serves to classify the likelihood of customer attrition utilizing historical behavioral and demographic features, minimizing acquisition costs required to replace lost user bases.

### 3.2 Customer Churn Problem
When a customer departs a service, revenue streams decline alongside realized profitability margins. Analyzing which factors—such as billing methodologies, tenure constraints, and localized support options—drive attrition allows organizations to prioritize localized engagement. Traditional, uninterpretable models do not deliver actionable intelligence, making localized intervention challenging. 

### 3.3 Motivation
An enterprise requires more than simply an accurate probability score; they require interpretable risk attributions and a strictly version-controlled production environment. Ensuring that model architecture and feature encoding remain encapsulated in specific versions (ModelBundle) prevents data leakage and runtime inconsistencies. Developing such an abstracted environment natively facilitates safe deployment and testing transitions.

### 3.4 Need for Predictive Retention Intelligence
RETENTIX AI goes beyond scoring by synthesizing predictions with local attributions (SHAP) and a simulator environment, enabling stakeholders to evaluate hypothetical scenarios. The strict separation of the UI (Streamlit), inference paths (`Prediction Service`), and ML implementation details (`ModelBundle`) forms a highly robust pipeline resilient to upstream data contamination.

## 4. PROBLEM STATEMENT

Predictive models are frequently compromised in production due to misaligned input contracts, silently failing encodings, and a lack of version integrity checks. In addition to creating a robust machine learning churn model, there is a prominent need for an encapsulated architecture capable of safely promoting and auditing multiple iterations (such as migrating from a legacy Logistic Regression architecture to an advanced Gradient Boosting pipeline) without violating production boundaries, while preserving detailed inference explainability.

## 5. OBJECTIVES

| Objective Category | Description |
|-------------------|-------------|
| **Core Prediction** | Develop an accurate ML model capable of scoring churn risk probabilities from raw demographic and service data. |
| **Robust Engineering** | Implement an isolated, version-aware `ModelBundle` framework to prevent data leakage and caching collisions during transitions. |
| **Active Auditing** | Formulate a shadow audit and promotion mechanism enforcing SHA-256 integrity across experimental iterations. |
| **Explainable AI** | Deliver model-agnostic and model-specific (TreeExplainer) insights mapped dynamically without relying on static column assumptions. |
| **User Implementation** | Build an interactive Streamlit UI covering executive summaries, exploratory analysis, prediction, and structured intervention ROIs. |

## 6. SCOPE OF THE PROJECT

The scope encompasses:
1. Data collection, preprocessing, and exploratory profiling of telecommunications behavior.
2. Cross-validated machine learning modeling evaluating candidates (e.g., Logistic Regression vs. Gradient Boosting) against an isolated Phase 3B testing partition.
3. Model evaluation, threshold sensitivity testing, and calibration tracking.
4. Construction of an abstracted `ModelBundle` backend containing strict input contract enforcement, prediction parity validations, and a V1 to V2 migration pathway.
5. Productionization of the V2 Champion model including integrated SHAP attribution intelligence and Streamlit UI interfacing.
(Revenue optimization calculations rely on representative baselines for ROI projections.)


## 7. EXISTING SYSTEM

Many conventional retention initiatives adhere to manual threshold triggers based extensively on singular metrics, such as missed payment schedules or high contract durations. Early predictive systems historically relied on unencapsulated model serialization procedures (e.g., saving un-pipelined arrays) which expose organizations to input drifting and hidden decoding bugs during real-world inference. Moreover, classical churn analytics typically deliver uninterpretable probability bounds, offering limited pathways to simulate corrective engagements (like moving a subscriber from Month-to-Month to Two-Year plans).

## 8. PROPOSED SYSTEM

The RETENTIX AI proposed system implements an end-to-end framework resolving historical limitations by coupling advanced predictive capabilities (GradientBoostingClassifier Pipeline with 45 transformed features) with an immutable `ModelBundle` service pattern. Through a dedicated `active_bundle.json` pointer, production environments gracefully load the designated inference boundaries while strictly enforcing an algorithmic `Input Contract`. Explainability (XAI) utilizes dynamically mapped TreeExplainer modules. Finally, a robust 6-page Streamlit application presents these technical capabilities as actionable executive dashboards.

## 9. SYSTEM REQUIREMENTS

### 9.1 Hardware
- Standard multi-core x86_64 or ARM equivalent Processor
- 8 GB RAM (minimum recommended for XAI background mask computation and Streamlit state mapping)
- Minimal disk utilization (~50MB overhead for binary model tracking and caching architectures)

### 9.2 Software
- Windows 10/11 or UNIX-based Environment
- Python 3.13.x
- Pandas, NumPy, Scikit-Learn
- SHAP, Streamlit, PyTest (Testing Suite)

## 10. TECHNOLOGY STACK

| Category | Technology / Framework Used |
|----------|-----------------------------|
| **Core Architecture** | Python 3.13 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Pipelines, SelectKBest, GridSearchCV, GradientBoosting) |
| **Model Explainability** | SHAP (TreeExplainer) |
| **Testing & CI/CD** | PyTest (381 Passed Cases) |
| **Web Interface** | Streamlit |

## 11. SYSTEM ARCHITECTURE

The RETENTIX AI system implements a strictly tiered design where visualization layers never directly access learning artifacts.

- **Streamlit Application Layer:** Six core pages present metrics, prediction simulators, and ROI calculations to users by communicating exclusively with backend API classes.
- **Bundle Prediction Service:** A service singleton (`BundlePredictionService`) mapping inputs downstream. It accesses `active_bundle.json` to resolve which bundle (`v1` or `v2`) to inject.
- **Model Bundle Abstraction & Input Contract:** Parses raw data payloads, verifies exact categorical schema bounds, confirms numeric validity, and rejects unsafe anomalies (infinity, unexpected NaNs in critical fields) to prevent silent model poisoning.
- **ML Pipeline & Preprocessing:** The selected algorithmic pipeline encapsulates transformation logic (imputation, mapping, OHE) and fitted estimation bound together natively.
- **Explainability Layer:** A generalized XAI boundary utilizing abstraction (`ModelView`) to bind pre-fitted representations into SHAP without hardcoding 30-column (V1) or 45-column (V2) assumptions.

[Figure: RETENTIX System Architecture]
```text
User
 |
 V
Streamlit UI
 |
 V
BundlePredictionService
 |
 V
Active Bundle Pointer (active_bundle.json)
 |
 V
ModelBundle (e.g., V2)
 |
 +--> Input Contract (Validation & Rejection)
 |
 +--> Preprocessing / Pipeline (45-feature transformation)
 |
 +--> GradientBoostingClassifier
 |
 +--> Threshold (0.50) / Positive Class ('Yes')
 |
 V
Prediction
```

[Figure: SHAP Explainability Workflow]
```text
Prediction Request
 |
 V
Explainability Layer
 |
 V
SHAP TreeExplainer
 |
 V
Map dynamically generated SHAP distributions against 45 transformed features
```


## 12. DATASET

The dataset comprises customer retention and demographic behavioral indicators. Each row represents a subscribed user spanning features such as demographic markers (Gender, SeniorCitizen status, Dependents), subscription parameters (Contract length, InternetService tiers, Stream usage), and billing histories (MonthlyCharges, TotalCharges). Target labels definitively identify customers who churned (attrited) versus active subscriptions.

## 13. DATA PREPROCESSING

Data ingestion is rigorously governed through a native abstraction pipeline:
- **Identifier Removal**: `customerID` identifiers are strictly dropped.
- **Continuous Variables**: Features including `MonthlyCharges` and `tenure` are ingested, while `TotalCharges` is converted to numeric formats. Nulls mapping onto `TotalCharges` are imputed utilizing the dataset median isolated during training to prevent leakage.
- **Categorical Processing**: Binary configurations (e.g., Partner, Dependents) are verified, and multivariable categories (e.g., InternetService, Contract variants) are safely transformed via scikit-learn preprocessing chains.
- **Input Contract Validation**: At production runtime, arbitrary or out-of-bounds variations (e.g., unexpected NaNs, unbounded numerical inputs, unknown categoricals) are blocked explicitly via the `ModelBundle` contract rather than triggering hidden transformation failure. 

## 14. EXPLORATORY DATA ANALYSIS

Exploratory Profiling isolates primary retention drivers before machine learning intervention. Through interactive graphical interfaces implemented natively in Streamlit (Exploratory Intelligence and Churn Drivers layers), demographic impacts such as Senior Citizenship distributions, monthly engagement thresholds, and multi-service aggregation statistics (StreamingTV overlaid onto InternetService dependencies) provide overarching business context framing the classification targets.

## 15. MACHINE LEARNING METHODOLOGY

### 15.1 Train/Test Strategy
The experimental framework adhered to a strict 80/20 train/test split utilizing a stratified approach (`test_size=0.2, random_state=42`) ensuring the target distribution bounds were consistently preserved. 

### 15.2 Cross-Validation
Performance estimation across candidate models within the development data utilized a 5-fold `StratifiedKFold` protocol (via `GridSearchCV`), assuring optimization sweeps remained uncontaminated by the held-out 80/20 assessment partition.

### 15.3 Candidate Models
During Phase 3B historical analyses, models evaluated included:
- **Logistic Regression** (baseline, subsequently promoted to V1)
- **RandomForestClassifier**
- **GradientBoostingClassifier** (Champion selected)

### 15.4 Model-Selection Methodology
Candidate selection prioritized architectures demonstrating robust Receiver Operating Characteristic (ROC-AUC) stability across cross-validated iterations while maintaining acceptable accuracy limits without overfitting tendencies. 

### 15.5 Champion Selection
Following CV sweeps, the **GradientBoostingClassifier** emerged fundamentally superior across generalized metrics and was selected as the designated Champion iteration without exposing the isolated final test data to hyperparameter optimization loops. 

### 15.6 Final Evaluation
The isolated Test Data (N=1409 instances) remained fully sequestered until final champion confirmation.

## 16. MODEL EVALUATION

The isolated, untouched final test metrics confirm the superiority of the Champion model prior to its promotion to the V2 production bundle.

| Metric | Cross-Validation Mean | Untouched Final Test Set |
|--------|-----------------------|--------------------------|
| **Accuracy** | 0.803 | 0.811 |
| **ROC-AUC** | 0.846 | 0.859 |

**V2 Production Identity:**
- **Model Type:** `GradientBoostingClassifier`
- **Format:** Pipeline (embedded transformation)
- **Feature Space:** 45 strictly mapped components
- **Positive Class:** 'Yes' (Churned)
- **Production Boundary Bound:** V2 (`active_bundle.json`)


## 17. THRESHOLD AND CALIBRATION ANALYSIS

Phase 3C threshold sensitivity analysis explored operating points beyond the baseline 0.5 boundary to understand misclassification costs:

- **Production Threshold:** **0.5** (unchanged, frozen)
- **Analysis Threshold:** 0.25 (analysis-only; not adopted)

At the 0.25 threshold, the expected False Positive rate increases substantially (retaining more customers than necessary, incurring higher operational intervention costs). Conversely, the True Positive rate captures additional at-risk individuals, but the net economic benefit is uncertain without precise downstream revenue data.

**Why was the 0.25 analysis threshold not adopted?**
Because production cost/revenue data (cost of retention offers vs. cost of true attrition) were not represented in the experimental repository, an automatic adoption of 0.25 was explicitly blocked. Stakeholder clearance is required before altering the production threshold away from the conservative 0.5 baseline.

## 18. EXPLAINABLE AI (XAI)

The XAI subsystem utilizes SHAP (SHapley Additive exPlanations) for dynamic attribution:
- **TreeExplainer Integration:** SHAP TreeExplainer is dynamically bound to the V2 GradientBoosting pipeline.
- **Global Explanations:** Rank mean |SHAP| impact across 45 transformed features to identify systemic retention drivers.
- **Individual Explanations:** Map per-customer attributions against actual prediction outcomes.
- **Simulation Explanations:** Allow counterfactual reasoning by perturbing a base row (e.g., changing Contract from Month-to-Month to Two-Year) and visualizing the attribution shift.
- **Attribution vs. Causality Limitation:** All SHAP values indicate model dependency structures (attribution) only. They do not represent causal proofs and must not be interpreted as evidence that altering a feature guarantees a specific retention outcome.

## 19. PRODUCTIONIZATION

### V1 vs. V2 Architecture Comparison

| Feature | V1 (Legacy/Champion) | V2 (Current Production) |
|---------|----------------------|-------------------------|
| **Model** | LogisticRegression | GradientBoostingClassifier |
| **Format** | Estimator Only | Pipeline |
| **Features** | 30 Transformed | 45 Transformed |
| **Preprocessing** | External `ChurnModel` | Embedded Pipeline |
| **Status** | Frozen | **ACTIVE** |

### Promotion & Rollback Mechanism
RETENTIX AI implements controlled migration rules:
1. **Shadow Audit:** Pre-promotion scoring compares V1/V2 behavioral boundaries on identical inputs (1409 shadow rows).
2. **Manifest Verification:** Each bundle (V1/V2) possesses explicit manifest metadata, ensuring integrity prior to activation.
3. **Pointer Rotation:** Migration consists of a singular `active_bundle.json` pointer modification (from `v1` to `v2`).
4. **SHA-256 Integrity:** Binary frozen artifacts (including both V1/V2 model binaries) remain byte-identical post-migration, verified cryptographically.
5. **Rollback:** If post-migration behavioral monitoring detects critical anomalies, the pointer can be reverted (V2 -> V1) instantly, restoring the former production state.

## 20. RETENTIX STREAMLIT APPLICATION

The frontend provides six distinct intelligence modules:
1. **Executive Command Center:** High-level retention KPIs and executive dashboards.
2. **Exploratory Intelligence:** Interactive EDA visualizations revealing demographic breakdowns.
3. **Churn Drivers & Ecosystem:** Feature dependency mapping and correlation structures.
4. **AI Churn Predictor & Simulator:** Single row prediction interface coupled with "What-If" simulation capabilities (e.g., modifying contract types to observe SHAP attribution shifts).
5. **Model Leaderboard & XAI:** Comprehensive SHAP global/explanation outputs.
6. **Retention ROI & Playbook:** Strategic retention frameworks (revenue calculations require downstream economic data).

## 21. INPUT VALIDATION AND ROBUSTNESS

The active V2 production inference path enforces an explicit `Input Contract` rejecting anomalies such as:
- Unknown categorical values (e.g., an undefined `Contract` variant)
- Missing/NaN values in critical boundaries (`tenure`, `MonthlyCharges`)
- Infinite numerical inputs (floating-point overflow protection)
- Invalid `SeniorCitizen` configurations (accepts 0, 1, "Yes", "No" only)
- Invalid `TotalCharges` types (rejects non-numeric strings; accepts NaN with imputation)


## 22. TESTING AND VALIDATION

Quality assurance tests continuously monitored the V2 active state:
- **Test Suite Results:** 381 passed tests, 0 failures, 1 XFAIL (Expected Failure tracking unsupported Phase 3 constraints).
- **Compilation Check:** Code bytecode generation (Python `compileall`) executed cleanly across the entire architecture (`frontend`, `backend`, `tests`, `app.py`).
- **Prediction Parity:** Output from exact rows evaluated under single-prediction contexts explicitly matched batch-prediction raw probabilities (0.6153) accounting for UI display rounding.
- **Contract Tests:** Robust rejection tests confirmed safety against NaN and structural tampering.
- **XAI Tests:** Assured TreeExplainer execution effectively maps to the 45 independent outputs of V2.
- **Rollback Tests:** Proved deterministic regression safety by verifying `active_bundle.json` switches reload historical models cleanly.

## 23. SECURITY AND RELIABILITY CONSIDERATIONS
Security relies fundamentally upon the immutable nature of saved configurations and hashes. All production-level object inputs are vetted through predefined structural bounds. Attempted serialization poisoning via unverified binary modifications is strictly mitigated through SHA-256 baseline verification preventing tampered `model.pkl` loading.

## 24. LIMITATIONS

Verified project limitations encompass:
1. **SHAP Constraints:** Attribution is bound by structural design and does not reflect actionable clinical or revenue causality.
2. **Analysis-Only Thresholds:** Optimal cost curve bounds (0.25 threshold) require organizational approval and are omitted from the conservative 0.50 production boundaries.
3. **No Incremental Training:** The system supports controlled version promotions but does not dynamically update weight logic without an explicit retraining Phase block.

## 25. FUTURE SCOPE
Potential operational enhancements include:
1. Economic mapping utilizing verifiable Cost-Benefit datasets to integrate the 0.25 threshold into automated recommendations.
2. Implementation of a live data-drift monitoring schema tracking statistical boundaries over extended deployment periods.
3. Upstream integration allowing real-time telecommunications ingestion architectures.

## 26. RESULTS AND DISCUSSION

The successful migration to RETENTIX AI demonstrates significant advancements in MLOps and predictive capability. Activating a V2 Gradient Boosting pipeline (81.1% final accuracy, ROC-AUC 0.859) inside a strict container environment validates the necessity of abstraction in machine learning deployments. The `active_bundle.json` migration cleanly encapsulated the structural shift from a 30-feature baseline to a robust 45-feature model format. Consequently, technical safety is guaranteed while delivering comprehensive XAI metrics back to stakeholders.

## 27. CONCLUSION

Customer churn predictive analysis serves to identify at-risk behaviors before irreversible attrition. RETENTIX AI satisfies operational necessities by deploying an highly accurate Gradient Boosting architecture paired with robust Input Contracts and explicit ModelBundle validation frameworks. By exposing SHAP attributions via a multi-page interactive application, organizations are granted the necessary transparency to enact localized interventions safely, guided by version-controlled prediction ecosystems.

## 28. REFERENCES

References to be finalized according to the institution's required citation format.


