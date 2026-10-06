import streamlit as st
import pandas as pd
import io
import plotly.express as px
from backend.prediction_service import PredictionServiceError
from frontend.styles import render_hero_banner, section_card, apply_plotly_theme

def show_bulk_prediction_studio(df: pd.DataFrame, model):
    render_hero_banner(
        title="Bulk Customer Portfolio Scoring",
        subtitle="Upload a CSV file containing customer data to perform batch churn predictions and risk analysis.",
        tag="BATCH INFERENCE ENGINE"
    )

    if model is None:
        st.error(
            "⚠️ The active model bundle could not be loaded. Bulk inference is "
            "unavailable. Please check the model configuration."
        )
        return

    main_card = section_card("📁", "Upload & Scoring Control")
    with main_card:
        col_upload, col_info = st.columns([6, 4])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Customer CSV for Batch Prediction", 
                type=["csv"], 
                help="The CSV must contain the required features for churn prediction (e.g., tenure, Contract, MonthlyCharges, etc.)"
            )
            
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 10px;">
                💡 <strong>Tip:</strong> You can download the template on the right to see the required format.
            </div>
            """, unsafe_allow_html=True)
        
        with col_info:
            st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 15px;">
                <div style="font-weight: 700; color: #A5B4FC; margin-bottom: 8px; font-size: 0.9rem;">📋 Batch Schema Requirements</div>
                <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.5;">
                    The model expects columns like:<br>
                    • Contract, Tenure, MonthlyCharges<br>
                    • InternetService, TechSupport, etc.<br>
                    Additional columns are automatically ignored.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # Template download
            template_df = df.head(5).copy()
            if "Churn" in template_df.columns:
                template_df = template_df.drop(columns=["Churn"])
            
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Schema Template",
                data=csv_template,
                file_name="telco_churn_template.csv",
                mime="text/csv",
                use_container_width=True
            )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            with st.status("Performing batch inference...", expanded=True) as status:
                st.write("Validating schema...")
                
                st.write("Running predictions...")
                batch_preds, batch_prob_arr = model.predict_batch(batch_df)
                
                st.write("Post-processing results...")
                batch_df["Predicted_Churn"] = list(batch_preds)
                batch_df["Churn_Probability (%)"] = (batch_prob_arr * 100).round(2)
                batch_df["Risk_Tier"] = pd.cut(
                    batch_df["Churn_Probability (%)"],
                    bins=[-1, 30, 60, 100],
                    labels=["Low Risk", "Moderate Risk", "High Risk"]
                )
                status.update(label="Scoring Complete!", state="complete", expanded=False)

            # Results Dashboard
            st.markdown("---")
            
            # Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            total = len(batch_df)
            churners = (batch_df["Predicted_Churn"] == "Yes").sum()
            high_risk = (batch_df["Risk_Tier"] == "High Risk").sum()
            avg_prob = batch_df["Churn_Probability (%)"].mean()
            
            with m1:
                st.metric("Total Processed", f"{total:,}")
            with m2:
                st.metric("Predicted Churn", f"{churners:,}", f"{(churners/total*100 if total > 0 else 0):.1f}%")
            with m3:
                st.metric("High Risk Accounts", f"{high_risk:,}")
            with m4:
                st.metric("Avg. Churn Probability", f"{avg_prob:.1f}%")

            # Visualization
            viz_col1, viz_col2 = st.columns([1, 1])
            
            with viz_col1:
                st.markdown("##### 🎯 Risk Tier Distribution")
                risk_counts = batch_df["Risk_Tier"].value_counts().reset_index()
                risk_counts.columns = ["Risk Tier", "Count"]
                fig_risk = px.pie(
                    risk_counts, 
                    values="Count", 
                    names="Risk Tier",
                    color="Risk Tier",
                    color_discrete_map={"Low Risk": "#10B981", "Moderate Risk": "#F59E0B", "High Risk": "#F43F5E"},
                    hole=0.4
                )
                apply_plotly_theme(fig_risk)
                st.plotly_chart(fig_risk, use_container_width=True)

            with viz_col2:
                st.markdown("##### 📈 Probability Distribution")
                fig_dist = px.histogram(
                    batch_df, 
                    x="Churn_Probability (%)",
                    nbins=20,
                    color_discrete_sequence=["#6366F1"]
                )
                apply_plotly_theme(fig_dist)
                st.plotly_chart(fig_dist, use_container_width=True)

            # Data Table
            st.markdown("##### 📄 Scored Customer Data")
            cols_to_show = ["Predicted_Churn", "Churn_Probability (%)", "Risk_Tier"]
            potential_ids = ["customerID", "Name", "ID"]
            found_ids = [c for c in potential_ids if c in batch_df.columns]
            
            display_cols = found_ids + cols_to_show + [c for c in batch_df.columns if c not in found_ids + cols_to_show]
            
            st.dataframe(
                batch_df[display_cols],
                use_container_width=True,
                hide_index=True
            )

            # Final Download
            out_buf = io.StringIO()
            batch_df.to_csv(out_buf, index=False)
            st.download_button(
                "📥 Download Full Scored Results (CSV)",
                data=out_buf.getvalue(),
                file_name="batch_churn_scores.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        except PredictionServiceError as e:
            st.error(f"**Contract Violation:** {e}")
            st.info("Please ensure your CSV matches the required schema.")
        except Exception as e:
            st.error(f"**An error occurred during processing:** {e}")
