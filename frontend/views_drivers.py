import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.styles import render_hero_banner, apply_plotly_theme, section_card

def show_churn_drivers_and_ecosystem(df: pd.DataFrame):
    render_hero_banner(
        title="Churn Drivers & Service Ecosystem",
        subtitle="Uncover the 'Protective Shield' effect of bundled add-on services, tenure hazard curves, and high-value risk quadrants.",
        tag="BEHAVIORAL RISK DYNAMICS"
    )

    col1, col2 = st.columns(2)

    with col1:
        rt_card = section_card("🛡️", "The 'Protective Shield': Service Add-on Bundling Effect")
        with rt_card:
            service_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            df_addons = df.copy()
            df_addons["Addon_Count"] = 0
            for s in service_addons:
                if s in df_addons.columns:
                    df_addons["Addon_Count"] += (df_addons[s] == "Yes").astype(int)

            addon_stats = df_addons.groupby("Addon_Count")["Churn"].agg(
                Total="count",
                Churn_Rate=lambda x: (x == "Yes").mean() * 100
            ).reset_index()

            fig_addons = px.bar(
                addon_stats,
                x="Addon_Count",
                y="Churn_Rate",
                color="Churn_Rate",
                text_auto=".1f",
                color_continuous_scale="Reds_r",
                labels={"Addon_Count": "Number of Active Add-on Services", "Churn_Rate": "Churn Rate (%)"}
            )
            fig_addons = apply_plotly_theme(fig_addons, height=310)
            st.plotly_chart(fig_addons, width="stretch")
        
            st.markdown("""
            <div style="font-size:0.83rem; color:#CBD5E1;">
                📌 <strong>The Ecosystem Retention Moat:</strong> Accounts with <strong>0 add-on services</strong> experience a <strong>45%+ churn rate</strong>. Subscribing to 4+ add-ons drops churn to <strong>under 12%</strong>!
            </div>
            """, unsafe_allow_html=True)

    with col2:
        rt_card = section_card("💳", "Churn Rates Across Payment Modalities")
        with rt_card:
            pay_stats = df.groupby("PaymentMethod")["Churn"].agg(
                Total="count",
                Churn_Rate=lambda x: (x == "Yes").mean() * 100
            ).reset_index().sort_values("Churn_Rate", ascending=True)

            fig_pay = px.bar(
                pay_stats,
                x="Churn_Rate",
                y="PaymentMethod",
                orientation="h",
                color="Churn_Rate",
                text_auto=".1f",
                color_continuous_scale="Reds",
                labels={"Churn_Rate": "Churn Rate (%)", "PaymentMethod": "Payment Channel"}
            )
            fig_pay = apply_plotly_theme(fig_pay, height=310)
            st.plotly_chart(fig_pay, width="stretch")

            st.markdown("""
            <div style="font-size:0.83rem; color:#CBD5E1;">
                📌 <strong>Payment Friction:</strong> <strong>Electronic Check</strong> has more than <strong>double</strong> the churn rate of Credit Card / Bank Transfer auto-pay options.
            </div>
            """, unsafe_allow_html=True)

    rt_card = section_card("🎯", "Customer Value vs. Tenure Risk Quadrants")
    with rt_card:
        med_monthly = df["MonthlyCharges"].median()
        med_tenure = df["tenure"].median()
    
        fig_scatter = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="tenure",
            y="MonthlyCharges",
            color="Churn",
            size="TotalCharges",
            hover_data=["Contract", "PaymentMethod", "InternetService"],
            color_discrete_map={"No": "#10B981", "Yes": "#F43F5E"},
            opacity=0.6,
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Bill ($)", "TotalCharges": "Lifetime Spend ($)"}
        )
        fig_scatter.add_vline(x=med_tenure, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_scatter.add_hline(y=med_monthly, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_scatter = apply_plotly_theme(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, width="stretch")
    
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown("""
            <div style="background:rgba(244,63,94,0.15); border:1px solid rgba(244,63,94,0.3); padding:10px; border-radius:10px;">
                <div style="font-weight:700; color:#FB7185; font-size:0.85rem;">Top Left: VIP Flight Risk</div>
                <div style="font-size:0.75rem; color:#CBD5E1;">High Spend, Low Tenure. Critical danger zone. White-glove proactive retention required!</div>
            </div>
            """, unsafe_allow_html=True)
        with q2:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); padding:10px; border-radius:10px;">
                <div style="font-weight:700; color:#34D399; font-size:0.85rem;">Top Right: Core Crown Jewels</div>
                <div style="font-size:0.75rem; color:#CBD5E1;">High Spend, High Tenure. Loyal premium accounts. Best candidates for VIP loyalty perks.</div>
            </div>
            """, unsafe_allow_html=True)
        with q3:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.3); padding:10px; border-radius:10px;">
                <div style="font-weight:700; color:#FBBF24; font-size:0.85rem;">Bottom Left: Trial Churners</div>
                <div style="font-size:0.75rem; color:#CBD5E1;">Low Spend, Low Tenure. Fast drop-offs. Deploy automated digital nurture campaigns.</div>
            </div>
            """, unsafe_allow_html=True)
        with q4:
            st.markdown("""
            <div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3); padding:10px; border-radius:10px;">
                <div style="font-weight:700; color:#60A5FA; font-size:0.85rem;">Bottom Right: Stable Budget Anchors</div>
                <div style="font-size:0.75rem; color:#CBD5E1;">Low Spend, High Tenure. Steady baseline subscribers with near-zero attrition.</div>
            </div>
            """, unsafe_allow_html=True)

