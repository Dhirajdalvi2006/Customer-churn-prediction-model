# ⚡ RETENTIX AI | Telecommunications Customer Churn Intelligence & Retention Studio

**Engineered by Dhiraj Dalvi**  
*Enterprise Telecom Machine Learning & Revenue Protection System (v2.4)*

---

## 📌 Executive Overview

**RETENTIX AI** is a state-of-the-art, end-to-end customer churn intelligence platform designed for telecommunications enterprise leaders, retention managers, and data science teams. By combining high-precision machine learning inference with interactive commercial simulation, RETENTIX AI empowers organizations to identify churn risks, diagnose behavioral catalysts, simulate retention campaign ROI, and execute targeted retention strategies before revenue loss occurs.

---

## 🌟 Key Application Modules & Features

### 🏠 1. Executive Command Center
* **Macro Performance Indicators:** Live tracking of Active Accounts, Churn Rate %, Total MRR ($), At-Risk MRR ($), and Average Lifetime Spend (CLV).
* **Revenue Exposure Analysis:** Monthly Revenue at risk broken down across Contract Horizons (Month-to-month, 1-Year, 2-Year).
* **Portfolio Risk Distribution:** Interactive donut ratio comparing retained vs. churned subscriber proportions.
* **Tenure Density Mapping:** Histogram overlay showing churn hazard concentration across customer lifetime months.
* **Strategic Signals & Alerts:** Live vulnerability flags covering Month-to-Month contracts, Fiber Optic pricing anomalies, and Electronic Check payment friction.

### 📊 2. Exploratory Data Intelligence
* **Demographic & Categorical Breakdown:** Deep-dive analysis of customer attributes (Contract, Internet Service, Tech Support, etc.) with churn rates.
* **Service Hierarchy Sunburst:** Multi-level revenue allocation viz (`InternetService ➔ Contract ➔ Churn`).
* **Pearson Correlation Matrix:** Quantitative feature correlations with the churn target variable.
* **Numerical Metric Distributions:** Histograms, boxplots, and violin plots for `MonthlyCharges`, `TotalCharges`, and `tenure`.
* **Interactive Dataset Explorer:** Search, filter, and export customized cohort datasets to CSV.

### 🔍 3. Churn Drivers & Service Ecosystem
* **The "Protective Shield" Effect:** Quantification of how bundling add-on services (Online Security, Tech Support, Backup, etc.) reduces churn rate from **45%+ down to under 12%**.
* **Payment Modality Friction:** Comparative analysis revealing why **Electronic Check** subscribers exhibit >2x the churn rate of automated payment methods.
* **Value vs. Tenure Risk Quadrants:** Scatter mapping categorizing accounts into *VIP Flight Risk*, *Core Crown Jewels*, *Trial Churners*, and *Stable Budget Anchors*.

### 🔮 4. Real-Time AI Churn Predictor & Simulator
* **Single Subscriber Profiling:** Interactive form to input custom demographic, service, and billing attributes.
* **Archetype Quick-Load Presets:** One-click loading of predefined customer profiles (*High-Risk Newcomer*, *5-Year Loyal Family*, *High-Spend Streamer*, *Budget Saver*).
* **Live Inference Gauge:** Real-time scoring with instant classification into Low, Moderate, or High Risk tiers.
* **Interactive Intervention Simulator:** Test "what-if" scenarios (e.g., upgrading contract or adding Tech Support) to see instant reductions in predicted churn probability.

### 📁 5. Bulk Customer Portfolio Scoring
* **Batch CSV Upload:** Process large subscriber files simultaneously through the active model bundle.
* **Schema Validation & Template Download:** Auto-validation against `REQUIRED_COLS` with standard template download.
* **Automated Risk Tiering:** Categorization into *Low Risk* (0–30%), *Moderate Risk* (30–60%), and *High Risk* (60–100%).
* **Batch Analytics & Export:** Visual risk distribution pie chart, probability density histogram, and full scored CSV download.

### 🤖 6. Model Leaderboard & Explainable AI (XAI)
* **Algorithm Performance Benchmarks:** Comparative leaderboard across Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and AdaBoost models.
* **Global Feature Importance:** Feature ranking based on model weights and coefficient magnitude.
* **SHAP (SHapley Additive exPlanations):** Deterministic population-level mean $|SHAP|$ attributions and individual customer log-odds attributions with additivity verification.
* **Publication-Ready Gallery:** High-resolution chart artifact viewer for executive presentations.

### 💼 7. Retention ROI & Playbook Simulator
* **Financial Campaign Simulator:** Input cohort size, average monthly bill, offer cost, and expected conversion rate to compute **Net Profit ($)** and **ROI (%)**.
* **4-Pillar Executive Playbook:**
  1. *Contract Horizon Migration Program* (Targeting Month-to-month hazard).
  2. *"Security Shield" Ecosystem Bundling* (Mitigating Fiber Optic churn).
  3. *"First-90-Days" Concierge Onboarding* (Reducing early tenure drop-offs).
  4. *Frictionless Auto-Pay Migration* (Eliminating Electronic Check attrition).

---

## 🛠️ Technology Stack & Architecture

* **Frontend UI & Visualizations:** [Streamlit](https://streamlit.io/), [Plotly Express & Graph Objects](https://plotly.com/python/), Custom Glassmorphism CSS Design Tokens (`frontend/styles.py`).
* **Machine Learning & Core Pipeline:** Python 3.13+, `scikit-learn`, `pandas`, `numpy`, `joblib`.
* **Model Explainability (XAI):** `shap` (SHapley Additive exPlanations).
* **Reporting & Exporting:** `python-docx`, CSV exports, `io.StringIO`.
* **Testing & Quality Assurance:** `pytest` unit & integration testing suite (380+ tests).

---

## 📂 Project Structure

```
customer_churn_project/
├── app.py                      # Streamlit main entrypoint & view router
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation (this file)
├── backend/                    # Core machine learning & backend services
│   ├── __init__.py
│   ├── model.py                # Data loading, schema definitions & preprocessing
│   ├── prediction_service.py   # Active bundle loader & production inference service
│   ├── input_contract.py       # Payload validation & schema enforcement
│   ├── explainability.py       # SHAP global & local attribution engine
│   ├── model_bundle.py         # Bundle integrity hashing & artifact management
│   └── ...
├── frontend/                   # User Interface views & design system
│   ├── __init__.py
│   ├── styles.py               # Enterprise CSS design tokens, card containers & themes
│   ├── views_executive.py      # Executive Command Center view
│   ├── views_eda.py            # Exploratory Data Intelligence view
│   ├── views_drivers.py        # Churn Drivers & Ecosystem view
│   ├── views_predict.py        # Real-time Predictor & Simulator view
│   ├── views_bulk.py           # Bulk Portfolio Scoring view
│   ├── views_models.py         # Model Leaderboard & SHAP XAI view
│   └── views_strategy.py       # Retention ROI & Playbook view
├── models/                     # Model artifacts & bundle configuration
│   ├── active_bundle.json      # Pointer to active production bundle
│   ├── best_model.pkl          # Persisted champion model artifact
│   ├── preprocessor.pkl        # Data transformation pipeline
│   └── ...
├── data/                       # Telecommunications customer datasets
│   └── telco_customer_churn_cleaned.csv
├── visualizations/             # Generated publication-ready charts & plots
├── reports/                    # Generated docx analytical reports
└── tests/                      # Automated test suite (pytest)
    ├── test_champion_contract.py
    ├── test_input_contract.py
    ├── test_model_bundle.py
    ├── test_prediction_service.py
    ├── test_shap_explainability.py
    └── test_streamlit_app_smoke.py
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
* Python 3.10+ (Python 3.13 fully supported)
* `pip` package manager

### 2. Clone / Extract Repository
Navigated to your local project directory:
```bash
cd customer_churn_project
```

### 3. Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Application

Execute the following command in your terminal:
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

To run the complete unit and integration test suite:
```bash
# Windows CMD / PowerShell
cmd /c "set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && python -m pytest"

# Linux / macOS
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

---

## 🔧 Bug Fixes & System Stability Improvements

* **Strategic Signals HTML Formatting Fix:** Resolved a raw text display issue where `st.markdown(..., unsafe_allow_html=True)` rendered raw `<div>` tags on screen due to blank line breaks interrupting CommonMark HTML block parsing. Streamlit HTML blocks are now continuous and unindented.
* **Requirements Modernization:** Updated `requirements.txt` with clear sectioning and verified compatibility across Python 3.13, NumPy 2.x, pandas 3.0, scikit-learn 1.6+, and SHAP 0.48+.
* **Bundle Architecture Integrity:** Implemented bundle pointer validation through `models/active_bundle.json` ensuring safe fallback error messaging and schema contract adherence across both single and batch predictions.

---

## 📜 License & Citation

Designed and engineered by **Dhiraj Dalvi**.  
*For questions, commercial deployment inquiries, or custom Telecom ML integrations, please refer to the project documentation.*