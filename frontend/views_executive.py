import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.styles import render_hero_banner, apply_plotly_theme, section_card

def show_executive_command_center(df: pd.DataFrame, full_df: pd.DataFrame, model):
    render_hero_banner(
        title="Executive Retention Command Center",
        subtitle="Real-time macro performance indicators, customer portfolio risk exposure, and high-impact churn intelligence.",
        tag="PORTFOLIO METRICS & REVENUE EXPOSURE"
    )

    total_customers = len(df)
    churn_count = (df["Churn"] == "Yes").sum()
    retained_count = (df["Churn"] == "No").sum()
    churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
    total_mrr = df["MonthlyCharges"].sum()
    at_risk_mrr = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()
    pct_mrr_at_risk = (at_risk_mrr / total_mrr * 100) if total_mrr > 0 else 0
    avg_tenure = df["tenure"].mean() if total_customers > 0 else 0
    avg_clv = df["TotalCharges"].mean() if total_customers > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Accounts <span>👥</span></div>
            <div class="kpi-value">{total_customers:,}</div>
            <span class="kpi-badge badge-info">{retained_count:,} Retained</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        badge_style = "badge-danger" if churn_rate > 25 else "badge-warning"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Churn Rate <span>⚠️</span></div>
            <div class="kpi-value" style="color: {'#FB7185' if churn_rate > 25 else '#FBBF24'};">{churn_rate:.1f}%</div>
            <span class="kpi-badge {badge_style}">{churn_count:,} Churned</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total MRR <span>💳</span></div>
            <div class="kpi-value">${total_mrr/1000:,.1f}k</div>
            <span class="kpi-badge badge-success">${df['MonthlyCharges'].mean() if total_customers > 0 else 0:.1f} Avg/Sub</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">At-Risk MRR <span>🚨</span></div>
            <div class="kpi-value" style="color: #FB7185;">${at_risk_mrr/1000:,.1f}k</div>
            <span class="kpi-badge badge-danger">{pct_mrr_at_risk:.1f}% of Total</span>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Lifetime Spend <span>💎</span></div>
            <div class="kpi-value">${avg_clv:,.0f}</div>
            <span class="kpi-badge badge-info">{avg_tenure:.0f} Mo Tenure</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([6, 4])
    with col_left:
        rt_card = section_card("💵", "Monthly Revenue Exposure & At-Risk Capital by Contract")
        with rt_card:
            contract_mrr = df.groupby(["Contract", "Churn"])["MonthlyCharges"].sum().reset_index()
            fig_rev = px.bar(
                contract_mrr,
                x="Contract",
                y="MonthlyCharges",
                color="Churn",
                barmode="group",
                text_auto=".2s",
                color_discrete_map={"No": "#22C55E", "Yes": "#EF4444"},
                labels={"MonthlyCharges": "Monthly Revenue ($)", "Contract": "Contract Horizon"}
            )
            fig_rev = apply_plotly_theme(fig_rev, height=330)
            st.plotly_chart(fig_rev, width="stretch")

    with col_right:
        rt_card = section_card("🎯", "Portfolio Churn Ratio")
        with rt_card:
            churn_counts = df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Status", "Count"]
            churn_counts["Status"] = churn_counts["Status"].map({"No": "Retained Customers", "Yes": "Churned Customers"})
        
            fig_donut = px.pie(
                churn_counts,
                values="Count",
                names="Status",
                hole=0.68,
                color="Status",
                color_discrete_map={"Retained Customers": "#22C55E", "Churned Customers": "#EF4444"}
            )
            fig_donut.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color='#0F172A', width=3)))
            fig_donut = apply_plotly_theme(fig_donut, height=330)
            fig_donut.update_layout(showlegend=False)
            st.plotly_chart(fig_donut, width="stretch")

    col_a, col_b = st.columns([5, 5])
    with col_a:
        rt_card = section_card("⏳", "Churn Density Across Customer Tenure Horizons")
        with rt_card:
            fig_tenure = px.histogram(
                df,
                x="tenure",
                color="Churn",
                nbins=36,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={"No": "#22C55E", "Yes": "#EF4444"},
                labels={"tenure": "Tenure (Months in Service)"}
            )
            fig_tenure = apply_plotly_theme(fig_tenure, height=300)
            st.plotly_chart(fig_tenure, width="stretch")

    with col_b:
        rt_card = section_card("🛡️", "Strategic Intelligence & Urgent Signals")
        with rt_card:
            m2m_churn = (df[df["Contract"] == "Month-to-month"]["Churn"] == "Yes").mean() * 100 if len(df[df["Contract"] == "Month-to-month"]) > 0 else 0
            fiber_churn = (df[df["InternetService"] == "Fiber optic"]["Churn"] == "Yes").mean() * 100 if len(df[df["InternetService"] == "Fiber optic"]) > 0 else 0
            echeck_churn = (df[df["PaymentMethod"] == "Electronic check"]["Churn"] == "Yes").mean() * 100 if len(df[df["PaymentMethod"] == "Electronic check"]) > 0 else 0

            st.markdown(f"""<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 4px;">
<div style="background: rgba(244, 63, 94, 0.12); border-left: 4px solid #F43F5E; padding: 12px; border-radius: 8px;">
    <div style="font-weight: 700; color: #FB7185; font-size: 0.92rem;">🚨 Month-to-Month Contract Vulnerability</div>
    <div style="color: #CBD5E1; font-size: 0.83rem; margin-top: 2px;">
        Month-to-month accounts exhibit a <strong>{m2m_churn:.1f}% churn rate</strong>. Converting 15% to 1-Year plans would save ~${(df[df['Contract']=='Month-to-month']['MonthlyCharges'].sum() * 0.15 * 12)/1000:,.0f}k/yr.
    </div>
</div>
<div style="background: rgba(245, 158, 11, 0.12); border-left: 4px solid #F59E0B; padding: 12px; border-radius: 8px;">
    <div style="font-weight: 700; color: #FBBF24; font-size: 0.92rem;">⚡ Fiber Optic Churn Anomaly</div>
    <div style="color: #CBD5E1; font-size: 0.83rem; margin-top: 2px;">
        Fiber optic users experience a <strong>{fiber_churn:.1f}% churn rate</strong>. Pricing combined with lack of Tech Support is the primary catalyst.
    </div>
</div>
<div style="background: rgba(59, 130, 246, 0.12); border-left: 4px solid #3B82F6; padding: 12px; border-radius: 8px;">
    <div style="font-weight: 700; color: #60A5FA; font-size: 0.92rem;">💳 Electronic Check Friction</div>
    <div style="color: #CBD5E1; font-size: 0.83rem; margin-top: 2px;">
        Customers paying via Electronic Check churn at <strong>{echeck_churn:.1f}%</strong> compared to ~15% for automatic bank/credit card payment methods.
    </div>
</div>
</div>""", unsafe_allow_html=True)
