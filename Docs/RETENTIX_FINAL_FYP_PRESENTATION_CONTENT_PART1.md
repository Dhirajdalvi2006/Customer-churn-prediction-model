# RETENTIX AI: Final FYP Presentation Structure (Part 1: Slides 1-8)

## Slide 1: Title
- **Key Message**: Title of the FYP.
- **Content**:
    - Project Title: RETENTIX AI: Explainable Customer Churn Prediction and Retention Intelligence System
    - Presenter Name: [Name]
    - Institution: [Institution]
    - Date: [Date]
- **Visual**: Academic title page design.
- **Speaker Notes**: Welcome the panel to the presentation for RETENTIX AI, an explainable churn prediction system.

## Slide 2: Problem Statement
- **Key Message**: Customer churn is a critical challenge.
- **Content**:
    - Customer churn reduces revenue and market share.
    - Reactive retention efforts are often too late.
    - Predictive systems are needed but lack explainability.
- **Visual**: Simple churn icon/graphic.
- **Speaker Notes**: Introduce the problem: churn is a major issue, and we need better ways to predict it and understand *why*.

## Slide 3: Objectives
- **Key Message**: Clear goals for the FYP.
- **Content**:
    - Develop a predictive churn model.
    - Implement explainability (SHAP).
    - Build a retention-intelligence dashboard.
    - Establish a robust productionization pipeline.
- **Visual**: Bulleted list of 4 objectives.
- **Speaker Notes**: Briefly outline the key objectives: predict, explain, dashboard, and productionize.

## Slide 4: Proposed Solution
- **Key Message**: End-to-end churn intelligence.
- **Content**:
    - Data validation, ML prediction, churn probability assessment, actionable retention playbooks.
- **Visual**: Flowchart (Data -> Prediction -> Explainability -> Retention).
- **Speaker Notes**: Explain the RETENTIX solution overview from data to actionable insights.

## Slide 5: System Architecture
- **Key Message**: Robust production-ready architecture.
- **Content**: 
    - Streamlit UI -> Prediction Service -> Active Bundle -> ModelBundle -> Pipeline.
- **Visual**: Architecture Diagram.
- **Speaker Notes**: Describe the production stack: Streamlit UI -> Prediction Service -> Active Bundle Pointer -> ModelBundle -> Pipeline.

## Slide 6: Dataset & Features
- **Key Message**: Verified data foundations.
- **Content**:
    - Feature Count: 45 (transformed)
    - Categorical/Numerical structure verified.
    - TotalCharges handling defined.
- **Visual**: Feature breakdown table placeholder.
- **Speaker Notes**: Detail the dataset features, emphasizing the transformation to 45 dimensions.

## Slide 7: ML Methodology
- **Key Message**: Strict validation methodology.
- **Content**:
    - 80/20 stratified split.
    - 5-Fold Stratified CV on dev set.
    - Champion Selection: GradientBoostingClassifier.
    - Untouched Final Test Set for performance evaluation.
- **Visual**: Flowchart (Split -> CV -> Champion -> Test).
- **Speaker Notes**: Explain the workflow ensuring no test data leakage into training/selection.

## Slide 8: Model Comparison
- **Key Message**: Rigorous selection process.
- **Content**:
    - Comparison: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
    - Selected Model: GradientBoostingClassifier (Pipeline format).
- **Visual**: Performance comparison table placeholder.
- **Speaker Notes**: Present the model selection results, highlighting the selection of Gradient Boosting.
