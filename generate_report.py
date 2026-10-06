import os
import joblib
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_section_line(doc):
    para = doc.add_paragraph()
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "667EEA")
    pBdr.append(bottom)
    pPr.append(pBdr)
    return para


def add_image(doc, filename, caption, width=5.5):
    fpath = os.path.join(VIZ_DIR, filename)
    if os.path.exists(fpath):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(fpath, width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap.runs[0]
        cap_run.font.italic = True
        cap_run.font.size = Pt(9)
        cap_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        doc.add_paragraph()


def create_report():
    results = joblib.load(os.path.join(MODEL_DIR, "results.pkl"))
    best_model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))

    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    h1_style = doc.styles["Heading 1"]
    h1_style.font.name = "Calibri"
    h1_style.font.size = Pt(16)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    h2_style = doc.styles["Heading 2"]
    h2_style.font.name = "Calibri"
    h2_style.font.size = Pt(13)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Customer Churn Prediction")
    r.font.name = "Calibri"
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Machine Learning Project Report")
    r.font.name = "Calibri"
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x66, 0x7E, 0xEA)

    doc.add_paragraph()

    info_table = doc.add_table(rows=4, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Student Name", "Dhiraj Dalvi"),
        ("Course", "Data Analytics / Machine Learning"),
        ("Date", "September 2026"),
        ("Dataset", "Telco Customer Churn"),
    ]
    for i, (label, value) in enumerate(info_data):
        row = info_table.rows[i]
        cell_l = row.cells[0]
        cell_r = row.cells[1]
        cell_l.text = label
        cell_r.text = value
        cell_l.paragraphs[0].runs[0].font.bold = True
        cell_l.paragraphs[0].runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        set_cell_bg(cell_l, "D6E4F0")
        set_cell_bg(cell_r, "EBF5FB")

    doc.add_page_break()

    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        "1. Introduction ............................................................................",
        "2. Dataset Description .....................................................................",
        "3. Data Preprocessing ......................................................................",
        "4. Exploratory Data Analysis ................................................................",
        "5. Feature Engineering .....................................................................",
        "6. Model Training & Evaluation ..............................................................",
        "7. Model Comparison Results .................................................................",
        "8. Feature Importance Analysis ..............................................................",
        "9. Streamlit Web Application ................................................................",
        "10. Conclusions & Business Recommendations ..................................................",
        "11. Technologies Used .......................................................................",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.left_indent = Inches(0.3)
        p.runs[0].font.size = Pt(11)

    doc.add_page_break()

    doc.add_heading("1. Introduction", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "Customer churn is one of the most pressing business challenges for telecommunications "
        "companies. It refers to the loss of clients or subscribers who discontinue using a "
        "company's services to switch to a competitor. Reducing churn is critical because "
        "acquiring a new customer costs five to seven times more than retaining an existing one."
    )
    doc.add_paragraph(
        "This project presents a complete end-to-end machine learning solution for predicting "
        "customer churn using the IBM Telco Customer Churn dataset. The solution covers the full "
        "data science pipeline: data exploration, preprocessing, feature engineering, training "
        "and comparing four classification algorithms, and deploying the best model through an "
        "interactive Streamlit web dashboard."
    )

    doc.add_heading("1.1 Project Objectives", level=2)
    objectives = [
        "Build a binary classification model to predict whether a telecom customer will churn.",
        "Identify the key customer attributes and service factors that drive churn behavior.",
        "Compare multiple machine learning models using standard evaluation metrics.",
        "Deploy the best-performing model with an interactive prediction interface.",
        "Generate actionable business recommendations to reduce customer churn.",
    ]
    for obj in objectives:
        p = doc.add_paragraph(obj, style="List Bullet")
        p.runs[0].font.size = Pt(11)

    doc.add_heading("1.2 Scope", level=2)
    doc.add_paragraph(
        "The project covers 7,043 customer records with 20 input features covering demographics, "
        "phone and internet services, account tenure, billing information, and service add-ons. "
        "Four models were trained: Logistic Regression, Decision Tree, Random Forest, and "
        "Gradient Boosting. Model selection was based on the F1-Score to balance precision and "
        "recall on the imbalanced target variable."
    )

    doc.add_heading("2. Dataset Description", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "The dataset is the IBM Telco Customer Churn dataset containing information about "
        "telecom customers and whether they churned (stopped using the service). It has 7,043 "
        "rows and 21 columns including the target variable."
    )

    doc.add_heading("2.1 Summary Statistics", level=2)
    summary_table = doc.add_table(rows=6, cols=2)
    summary_table.style = "Light Grid Accent 1"
    summary_data = [
        ("Total Customers", "7,043"),
        ("Total Features", "21 (incl. target)"),
        ("Churned Customers", "1,869 (26.54%)"),
        ("Retained Customers", "5,174 (73.46%)"),
        ("Numerical Features", "4"),
        ("Categorical Features", "15 + customerID"),
    ]
    for i, (k, v) in enumerate(summary_data):
        row = summary_table.rows[i]
        row.cells[0].text = k
        row.cells[1].text = v
        row.cells[0].paragraphs[0].runs[0].font.bold = True
        set_cell_bg(row.cells[0], "D6E4F0")

    doc.add_paragraph()

    doc.add_heading("2.2 Feature Description", level=2)
    feat_table = doc.add_table(rows=1, cols=3)
    feat_table.style = "Light Grid Accent 1"
    for i, h in enumerate(["Feature", "Type", "Description"]):
        cell = feat_table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_bg(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    features = [
        ("customerID", "String", "Unique identifier — dropped during preprocessing"),
        ("gender", "Categorical", "Male or Female"),
        ("SeniorCitizen", "Binary (0/1)", "Whether the customer is a senior citizen"),
        ("Partner", "Yes / No", "Whether the customer has a partner"),
        ("Dependents", "Yes / No", "Whether the customer has dependents"),
        ("tenure", "Numerical", "Months the customer has been with the company"),
        ("PhoneService", "Yes / No", "Whether customer has phone service"),
        ("MultipleLines", "Categorical", "No / Yes / No phone service"),
        ("InternetService", "Categorical", "DSL / Fiber optic / No"),
        ("OnlineSecurity", "Categorical", "Yes / No / No internet service"),
        ("OnlineBackup", "Categorical", "Yes / No / No internet service"),
        ("DeviceProtection", "Categorical", "Yes / No / No internet service"),
        ("TechSupport", "Categorical", "Yes / No / No internet service"),
        ("StreamingTV", "Categorical", "Yes / No / No internet service"),
        ("StreamingMovies", "Categorical", "Yes / No / No internet service"),
        ("Contract", "Categorical", "Month-to-month / One year / Two year"),
        ("PaperlessBilling", "Yes / No", "Whether customer uses paperless billing"),
        ("PaymentMethod", "Categorical", "Electronic check / Mailed check / Bank/Credit card"),
        ("MonthlyCharges", "Numerical (Float)", "Amount charged monthly in USD"),
        ("TotalCharges", "Numerical (Float)", "Total amount charged to date in USD"),
        ("Churn", "Binary — TARGET", "Yes (churned) / No (retained)"),
    ]
    for feat, ftype, desc in features:
        row = feat_table.add_row().cells
        row[0].text = feat
        row[1].text = ftype
        row[2].text = desc

    doc.add_heading("3. Data Preprocessing", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "Before training, the raw dataset was cleaned and transformed through the following steps:"
    )
    steps = [
        "TotalCharges contained blank strings for 11 records with zero tenure. These were converted to NaN and imputed with the column median.",
        "The target variable Churn was label-encoded: 'Yes' → 1 (churn), 'No' → 0 (retain).",
        "customerID was dropped as it carries no predictive signal.",
        "The 4 numerical features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) were standardized using StandardScaler (zero mean, unit variance).",
        "The 15 categorical features were one-hot encoded using OneHotEncoder with drop='first' to eliminate multicollinearity, producing ~30 binary columns.",
        "A ColumnTransformer bundled both transformers into a single reusable preprocessing pipeline.",
        "Data was split 80 / 20 (train / test) with stratified sampling to preserve class proportions.",
    ]
    for s in steps:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("4. Exploratory Data Analysis", level=1)
    add_section_line(doc)

    doc.add_heading("4.1 Churn Distribution", level=2)
    doc.add_paragraph(
        "The dataset exhibits class imbalance: 73.46% of customers did not churn while 26.54% "
        "did. Stratified splitting was used to maintain this ratio in both train and test sets."
    )
    add_image(doc, "churn_distribution.png", "Figure 1 — Churn Distribution (bar & pie charts)")

    doc.add_heading("4.2 Numerical Feature Analysis", level=2)
    doc.add_paragraph(
        "Tenure, MonthlyCharges, and TotalCharges show distinct distributions between churners "
        "and non-churners. Churned customers tend to have lower tenure, higher monthly charges, "
        "and lower total charges (consistent with short-stay high-cost customers)."
    )
    add_image(doc, "numerical_distributions.png", "Figure 2 — Numerical Feature Distributions by Churn Status")

    doc.add_heading("4.3 Categorical Feature Analysis", level=2)
    doc.add_paragraph(
        "Among categorical features, Contract type is the strongest visual predictor. "
        "Month-to-month contract holders churn at roughly 42% vs 11% for one-year and 3% for "
        "two-year contract customers. Paperless billing and fiber optic internet also show "
        "elevated churn."
    )
    add_image(doc, "categorical_churn.png", "Figure 3 — Categorical Features vs Churn Counts")

    doc.add_heading("4.4 Correlation Analysis", level=2)
    doc.add_paragraph(
        "The correlation heatmap shows that tenure is negatively correlated with churn while "
        "MonthlyCharges is positively correlated. TotalCharges is highly correlated with "
        "tenure (longer customers accumulate more total charges)."
    )
    add_image(doc, "correlation_heatmap.png", "Figure 4 — Numerical Feature Correlation Heatmap")

    doc.add_heading("4.5 Service-wise Churn Rates", level=2)
    doc.add_paragraph(
        "Customers without online security, tech support, and device protection services show "
        "significantly higher churn rates. Internet service customers without security add-ons "
        "are particularly at risk."
    )
    add_image(doc, "service_churn_rates.png", "Figure 5 — Churn Rates Across Service Add-ons")

    doc.add_heading("4.6 Key EDA Findings", level=2)
    observations = [
        "Month-to-month contracts: ~42% churn rate vs ~11% (one-year) and ~3% (two-year).",
        "Fiber optic internet customers churn more than DSL customers.",
        "Electronic check payment method has the highest churn rate among payment types.",
        "Customers with tenure < 12 months are at the highest churn risk.",
        "Senior citizens have a slightly higher churn rate than non-senior customers.",
        "Gender has negligible impact on churn — nearly equal rates for male and female.",
        "Customers without tech support or online security churn at nearly 2x the rate of those with it.",
        "Higher monthly charges consistently correlate with higher churn probability.",
    ]
    for obs in observations:
        doc.add_paragraph(obs, style="List Bullet")

    doc.add_heading("5. Feature Engineering", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "Beyond cleaning, the following transformations were applied to prepare features for "
        "machine learning:"
    )
    fe = [
        "OneHotEncoding: 15 categorical columns expanded into ~30 binary dummy variables using drop='first' to avoid the dummy variable trap.",
        "StandardScaler: 4 numerical columns normalized to mean=0, std=1 so that gradient-based and distance-based models converge correctly.",
        "ColumnTransformer: Both transformers were combined into a single sklearn pipeline step, applied consistently to train and test data.",
        "Pipeline Persistence: The fitted ColumnTransformer was serialized with joblib alongside the model to guarantee identical preprocessing at inference time.",
    ]
    for f in fe:
        doc.add_paragraph(f, style="List Bullet")

    doc.add_heading("6. Model Training & Evaluation", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "Four classification models were trained on the preprocessed 80% training split and "
        "evaluated on the 20% held-out test set. All models used random_state=42 for "
        "reproducibility."
    )

    model_details = [
        (
            "6.1 Logistic Regression",
            "A linear probabilistic classifier. max_iter=1000 to guarantee convergence on this "
            "multi-dimensional space. Works by estimating the log-odds of churn as a linear "
            "function of input features. Provides model coefficients that can be directly "
            "interpreted as feature importance (absolute value).",
        ),
        (
            "6.2 Decision Tree",
            "A tree-based non-linear classifier. max_depth=10 limits tree depth to prevent "
            "overfitting on the training set. Splits the feature space into axis-aligned "
            "rectangular regions, making it highly interpretable but prone to variance without "
            "depth constraint.",
        ),
        (
            "6.3 Random Forest",
            "An ensemble of 100 independent decision trees trained on bootstrapped subsets of "
            "the data (bagging). Each tree also sees a random subset of features at each split. "
            "Reduces variance through averaging, providing robust and generalizable predictions. "
            "n_jobs=-1 uses all CPU cores.",
        ),
        (
            "6.4 Gradient Boosting",
            "A sequential boosting ensemble of 100 shallow trees where each tree corrects the "
            "residual errors of the previous one. Typically achieves the best bias-variance "
            "balance. Slower to train than Random Forest but often yields higher accuracy on "
            "tabular data.",
        ),
    ]
    for heading, desc in model_details:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(desc)

    doc.add_heading("7. Model Comparison Results", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "All four models were evaluated on the test set using five standard classification "
        "metrics. The best model (highest F1-Score) was automatically selected for deployment."
    )

    metrics_order = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_table = doc.add_table(rows=1, cols=6)
    results_table.style = "Light Grid Accent 1"
    header_row = results_table.rows[0]
    for i, h in enumerate(["Model"] + metrics_order):
        cell = header_row.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_bg(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    best_f1 = max(v["metrics"]["F1-Score"] for v in results.values())

    for model_name, res in results.items():
        row = results_table.add_row().cells
        row[0].text = model_name
        if model_name == best_model_name:
            row[0].paragraphs[0].runs[0].font.bold = True
        for j, metric in enumerate(metrics_order):
            val = res["metrics"][metric]
            row[j + 1].text = f"{float(val):.4f}"
            if float(val) == max(float(results[m]["metrics"][metric]) for m in results):
                row[j + 1].paragraphs[0].runs[0].font.bold = True
                set_cell_bg(row[j + 1], "D5F5E3")

    doc.add_paragraph()
    note = doc.add_paragraph(
        f"★  Best Model: {best_model_name}  |  Best F1-Score: {best_f1:.4f}  |  "
        f"Bold+green cells indicate the highest value in each column."
    )
    note.runs[0].font.italic = True
    note.runs[0].font.size = Pt(10)
    note.runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    add_image(doc, "model_comparison.png", "Figure 6 — Model Performance Comparison (all metrics)")
    add_image(doc, "confusion_matrices.png", "Figure 7 — Confusion Matrices for All Four Models")
    add_image(doc, "roc_curves.png", "Figure 8 — ROC Curves (AUC comparison)")

    doc.add_heading("7.1 Metric Interpretation", level=2)
    metric_explanations = [
        ("Accuracy", "Proportion of all correct predictions. Can be misleading on imbalanced data."),
        ("Precision", "Of all predicted churners, what fraction actually churned. High precision = fewer false alarms."),
        ("Recall", "Of all actual churners, what fraction was caught. High recall = fewer missed churners (critical for business)."),
        ("F1-Score", "Harmonic mean of Precision and Recall. Best single metric for imbalanced classification."),
        ("ROC-AUC", "Area under the ROC curve. Measures the model's ability to separate churners from non-churners across all thresholds."),
    ]
    exp_table = doc.add_table(rows=1, cols=2)
    exp_table.style = "Light Grid Accent 1"
    exp_table.rows[0].cells[0].text = "Metric"
    exp_table.rows[0].cells[1].text = "Interpretation"
    exp_table.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    exp_table.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    set_cell_bg(exp_table.rows[0].cells[0], "D6E4F0")
    set_cell_bg(exp_table.rows[0].cells[1], "D6E4F0")
    for metric, explanation in metric_explanations:
        row = exp_table.add_row().cells
        row[0].text = metric
        row[1].text = explanation

    doc.add_heading("8. Feature Importance Analysis", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        f"Feature importance was extracted from the best model ({best_model_name}). "
        "For Logistic Regression, the absolute value of model coefficients is used as the "
        "importance score. For tree-based models, mean impurity decrease is used."
    )
    top_factors = [
        ("Contract — Two year", "Strong negative predictor of churn; two-year customers are most loyal"),
        ("Contract — One year", "Lower churn than month-to-month but higher than two-year"),
        ("Tenure", "Longer-tenured customers are significantly less likely to churn"),
        ("InternetService — Fiber optic", "Associated with higher churn, possibly due to pricing concerns"),
        ("MonthlyCharges", "Higher monthly bills correlate with increased churn risk"),
        ("TotalCharges", "Indirect reflection of tenure; higher total = longer relationship"),
        ("OnlineSecurity — Yes", "Customers with security add-ons are more loyal"),
        ("TechSupport — Yes", "Tech support subscribers have lower churn rates"),
        ("PaymentMethod — Electronic check", "Highest churn rate among payment methods"),
        ("PaperlessBilling — Yes", "Correlated with digital engagement and slightly higher churn"),
    ]
    fi_table = doc.add_table(rows=1, cols=2)
    fi_table.style = "Light Grid Accent 1"
    fi_table.rows[0].cells[0].text = "Feature"
    fi_table.rows[0].cells[1].text = "Business Insight"
    fi_table.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    fi_table.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    set_cell_bg(fi_table.rows[0].cells[0], "1F4E79")
    set_cell_bg(fi_table.rows[0].cells[1], "1F4E79")
    fi_table.rows[0].cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    fi_table.rows[0].cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for feat, insight in top_factors:
        row = fi_table.add_row().cells
        row[0].text = feat
        row[1].text = insight

    doc.add_paragraph()
    add_image(doc, "feature_importance.png", "Figure 9 — Top Feature Importances")

    doc.add_heading("9. Streamlit Web Application", level=1)
    add_section_line(doc)
    doc.add_paragraph(
        "An interactive web dashboard was developed using Streamlit to make the ML solution "
        "accessible to non-technical stakeholders. The application is separated into a "
        "backend layer (backend/model.py) handling all data and ML logic and a frontend "
        "layer (app.py) handling the UI and interactivity."
    )

    doc.add_heading("9.1 Application Pages", level=2)
    pages = [
        ("Project Overview", "Summary of the project, key statistics, technologies, and objectives."),
        ("Dataset Statistics", "Data types, missing values, unique value counts, and sample rows."),
        ("Exploratory Data Analysis", "Interactive Plotly charts for categorical and numerical feature distributions and correlations."),
        ("Churn Analysis", "Churn rate by feature, cross-tabulations, and distribution overlays for tenure and monthly charges."),
        ("Model Performance", "Metrics table, grouped bar chart comparison, and top feature importance bar chart."),
        ("Predict Churn", "Customer input form → real-time prediction → churn probability → gauge chart → feature importance explanation."),
    ]
    page_table = doc.add_table(rows=1, cols=2)
    page_table.style = "Light Grid Accent 1"
    page_table.rows[0].cells[0].text = "Page"
    page_table.rows[0].cells[1].text = "Description"
    page_table.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    page_table.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    set_cell_bg(page_table.rows[0].cells[0], "D6E4F0")
    set_cell_bg(page_table.rows[0].cells[1], "D6E4F0")
    for page, desc in pages:
        row = page_table.add_row().cells
        row[0].text = page
        row[1].text = desc

    doc.add_heading("9.2 Run Instructions", level=2)
    run_steps = [
        "Install dependencies:  pip install -r requirements.txt",
        "Train the model:  python backend/model.py",
        "Launch the dashboard:  streamlit run app.py",
        "Open your browser at:  http://localhost:8501",
    ]
    for step in run_steps:
        doc.add_paragraph(step, style="List Number")

    doc.add_heading("10. Conclusions & Business Recommendations", level=1)
    add_section_line(doc)

    doc.add_heading("10.1 Conclusions", level=2)
    conclusions = [
        f"The best performing model is {best_model_name} with an F1-Score of {best_f1:.4f} and ROC-AUC of {float(results[best_model_name]['metrics']['ROC-AUC']):.4f}.",
        "All four models achieve ROC-AUC above 0.74, confirming that the features carry strong signal for churn prediction.",
        "Contract type is the single most important business lever — customers on monthly contracts are over 10x more likely to churn than those on two-year plans.",
        "Short-tenure customers (< 6 months) represent the highest-risk segment and require immediate attention.",
        "Add-on services (OnlineSecurity, TechSupport, OnlineBackup) act as retention anchors — their presence significantly reduces churn probability.",
        "The deployed pipeline handles all preprocessing automatically, ensuring consistent and reproducible predictions on new customer data.",
    ]
    for c in conclusions:
        doc.add_paragraph(c, style="List Bullet")

    doc.add_heading("10.2 Business Recommendations", level=2)
    recs = [
        ("Contract Incentives", "Offer discounts (e.g., one free month) to month-to-month customers who upgrade to annual contracts."),
        ("Onboarding Programs", "Target new customers (tenure < 6 months) with proactive support calls and loyalty bonuses in the first 90 days."),
        ("Bundle Security Services", "Bundle OnlineSecurity and TechSupport with fiber optic plans at a marginal discount to increase retention."),
        ("Payment Method Migration", "Incentivize customers using electronic checks to switch to auto-pay (bank transfer or credit card) with a billing credit."),
        ("Fiber Optic Value Review", "Investigate whether fiber optic pricing is perceived as fair; high charges with fewer add-ons drive churn."),
        ("Predictive Alerts", "Deploy the churn model in production CRM to flag at-risk customers weekly and trigger automated retention outreach."),
        ("Loyalty Rewards", "Create a tenure-based loyalty tier system that rewards customers passing 12, 24, and 48-month milestones."),
    ]
    rec_table = doc.add_table(rows=1, cols=2)
    rec_table.style = "Light Grid Accent 1"
    rec_table.rows[0].cells[0].text = "Action"
    rec_table.rows[0].cells[1].text = "Recommendation"
    rec_table.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    rec_table.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    set_cell_bg(rec_table.rows[0].cells[0], "1F4E79")
    set_cell_bg(rec_table.rows[0].cells[1], "1F4E79")
    rec_table.rows[0].cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    rec_table.rows[0].cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for action, rec in recs:
        row = rec_table.add_row().cells
        row[0].text = action
        row[0].paragraphs[0].runs[0].font.bold = True
        row[1].text = rec

    doc.add_heading("11. Technologies Used", level=1)
    add_section_line(doc)
    techs = [
        ("Python 3.13", "Core programming language"),
        ("Pandas 2.x", "Data loading, cleaning, and manipulation"),
        ("NumPy 2.x", "Numerical array operations"),
        ("Scikit-learn 1.6", "ML algorithms, preprocessing pipelines, and evaluation metrics"),
        ("Matplotlib 3.x", "Static chart generation for the notebook and report"),
        ("Seaborn 0.12+", "Statistical heatmaps and enhanced matplotlib charts"),
        ("Plotly 5.x", "Interactive charts in the Streamlit dashboard"),
        ("Streamlit 1.28+", "Interactive web application framework"),
        ("Joblib 1.3+", "Serialization of trained model and preprocessing pipeline"),
        ("python-docx 0.8+", "Programmatic generation of this Word report"),
        ("Jupyter Notebook", "Interactive analysis and documentation environment"),
    ]
    tech_table = doc.add_table(rows=1, cols=2)
    tech_table.style = "Light Grid Accent 1"
    tech_table.rows[0].cells[0].text = "Technology"
    tech_table.rows[0].cells[1].text = "Purpose"
    tech_table.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    tech_table.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    set_cell_bg(tech_table.rows[0].cells[0], "D6E4F0")
    set_cell_bg(tech_table.rows[0].cells[1], "D6E4F0")
    for tech, purpose in techs:
        row = tech_table.add_row().cells
        row[0].text = tech
        row[0].paragraphs[0].runs[0].font.bold = True
        row[1].text = purpose

    doc.add_paragraph()
    add_section_line(doc)
    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer_p.add_run(
        "Customer Churn Prediction Project  ·  Dhiraj Dalvi  ·  September 2026"
    )
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    output_path = os.path.join(BASE_DIR, "DhirajDalvi_CustomerChurnReport.docx")
    doc.save(output_path)
    print(f"Report saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_report()
