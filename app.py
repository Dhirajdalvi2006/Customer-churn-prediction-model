import os
import sys

# Ensure root directory and current working directory are in sys.path for Streamlit Cloud & Vercel
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in [ROOT_DIR, os.getcwd(), os.path.abspath(".")]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd

try:
    from backend.model import load_data
    from backend.prediction_service import get_prediction_service, PredictionServiceError
    from frontend.styles import inject_custom_css
    from frontend.views_executive import show_executive_command_center
    from frontend.views_eda import show_exploratory_intelligence
    from frontend.views_drivers import show_churn_drivers_and_ecosystem
    from frontend.views_predict import show_ai_prediction_and_simulator
    from frontend.views_models import show_model_evaluation_and_explainability
    from frontend.views_strategy import show_retention_roi_and_playbook
    from frontend.views_bulk import show_bulk_prediction_studio
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from backend.model import load_data
    from backend.prediction_service import get_prediction_service, PredictionServiceError
    from frontend.styles import inject_custom_css
    from frontend.views_executive import show_executive_command_center
    from frontend.views_eda import show_exploratory_intelligence
    from frontend.views_drivers import show_churn_drivers_and_ecosystem
    from frontend.views_predict import show_ai_prediction_and_simulator
    from frontend.views_models import show_model_evaluation_and_explainability
    from frontend.views_strategy import show_retention_roi_and_playbook
    from frontend.views_bulk import show_bulk_prediction_studio


st.set_page_config(
    page_title="RETENTIX AI | Customer Churn Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_trained_model():
    """Resolve the production inference service for the active model bundle.

    Phase 4 Step 10: the application no longer loads ``best_model.pkl`` /
    ``preprocessor.pkl``. ``get_prediction_service()`` resolves
    ``models/active_bundle.json`` through ``BundleLoader.get_active_bundle()``,
    so the model is chosen solely by the active pointer.

    A resolution failure is surfaced to the user as a clear application error
    and the service is returned as ``None``. It is *not* replaced by a legacy
    fallback: silently scoring from the old artifacts would hide a broken
    deployment and defeat the bundle architecture.
    """
    try:
        return get_prediction_service()
    except PredictionServiceError as exc:
        st.error(str(exc))
        return None


def main():
    inject_custom_css()

    df = get_dataset()
    model = get_trained_model()

    with st.sidebar:
        st.markdown('''
        <div style="padding: 0.2rem 0 1.1rem 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); padding: 8px 12px; border-radius: 12px; font-weight: 800; font-size: 1.2rem; color: white;">⚡</div>
                <div>
                    <div style="font-weight: 800; font-size: 1.2rem; color: #FFFFFF; letter-spacing: -0.02em;">RETENTIX<span style="color:#6366F1;">.AI</span></div>
                    <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 500;">CHURN INTELLIGENCE STUDIO</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; margin:0.5rem 0 0.3rem 0;'>Navigation Menu</p>", unsafe_allow_html=True)
        
        pages = [
            "🏠 Executive Command Center",
            "📊 Exploratory Intelligence",
            "🔍 Churn Drivers & Ecosystem",
            "🔮 AI Churn Predictor & Simulator",
            "📁 Bulk Portfolio Prediction",
            "🤖 Model Leaderboard & XAI",
            "💼 Retention ROI & Playbook"
        ]
        
        selected_page = st.radio(
            "Select View",
            options=pages,
            label_visibility="collapsed"
        )

        st.markdown("---")
        
        # Active-model metadata is read from the resolved bundle, never
        # hard-coded, so the sidebar stays correct if the active pointer moves.
        if model is not None:
            model_name = model.best_model_name
            model_sub = f"Bundle {model.bundle_version} · {model.transformed_feature_count} features"
        else:
            model_name = "Not Loaded"
            model_sub = "Active bundle unavailable"

        st.markdown(f'''
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px;">
            <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Inference Engine</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">⚡ {model_name}</div>
            <div style="font-size: 0.7rem; color: #64748B; margin-top: 2px;">{model_sub}</div>
            <div style="display: flex; gap: 8px; margin-top: 6px; font-size: 0.72rem;">
                <span style="color:#34D399;">● ROC-AUC: 84.2%</span>
                <span style="color:#94A3B8;">|</span>
                <span style="color:#A5B4FC;">7,043 Cohorts</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Global Slicer in Sidebar
        st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; margin:0.4rem 0 0.2rem 0;'>Cohort Filter Slicer</p>", unsafe_allow_html=True)
        contract_filter = st.multiselect(
            "Contract Type",
            options=df["Contract"].unique().tolist(),
            default=df["Contract"].unique().tolist(),
            help="Filter data dynamically across analytics views"
        )
        internet_filter = st.multiselect(
            "Internet Service",
            options=df["InternetService"].unique().tolist(),
            default=df["InternetService"].unique().tolist()
        )

        filtered_df = df[
            (df["Contract"].isin(contract_filter)) & 
            (df["InternetService"].isin(internet_filter))
        ]

        st.caption(f"Active Cohort: **{len(filtered_df):,}** of {len(df):,} accounts ({len(filtered_df)/len(df)*100:.1f}%)")

        st.markdown('''
        <div style="position: relative; margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255, 255, 255, 0.06); font-size: 0.7rem; color: #64748B; text-align: center;">
            Engineered by <strong>Dhiraj Dalvi</strong><br>
            Telecom ML Intelligence System v2.4
        </div>
        ''', unsafe_allow_html=True)

    # --- ROUTE TO SELECTED PAGE ---
    if selected_page == "🏠 Executive Command Center":
        show_executive_command_center(filtered_df, df, model)
    elif selected_page == "📊 Exploratory Intelligence":
        show_exploratory_intelligence(filtered_df)
    elif selected_page == "🔍 Churn Drivers & Ecosystem":
        show_churn_drivers_and_ecosystem(filtered_df)
    elif selected_page == "📁 Bulk Portfolio Prediction":
        show_bulk_prediction_studio(df, model)

    elif selected_page == "🔮 AI Churn Predictor & Simulator":
        show_ai_prediction_and_simulator(df, model)
    elif selected_page == "🤖 Model Leaderboard & XAI":
        show_model_evaluation_and_explainability(df, model)
    elif selected_page == "💼 Retention ROI & Playbook":
        show_retention_roi_and_playbook(filtered_df)


if __name__ == "__main__":
    main()

# Top-level exports for Vercel / WSGI serverless deployment entry point checks
def handler(request=None, response=None):
    return main()

app = application = handler

