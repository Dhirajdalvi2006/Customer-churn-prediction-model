import streamlit as st
import pandas as pd
from frontend.styles import render_hero_banner, section_card

def show_retention_roi_and_playbook(df: pd.DataFrame):
    render_hero_banner(
        title="Retention Strategy & Campaign ROI Simulator",
        subtitle="Model business impact, simulate customer win-back campaign economics, and execute four proven operational retention playbooks.",
        tag="COMMERCIAL ROI & STRATEGIC PLAYBOOK"
    )

    rt_card = section_card("💰", "Retention Campaign ROI Financial Model")
    with rt_card:
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            target_cohort_size = st.number_input("Target High-Risk Cohort (Users)", 100, 5000, 1000, step=100)
        with rc2:
            avg_user_mrr = st.number_input("Average Monthly Bill ($)", 20.0, 150.0, 75.0, step=5.0)
        with rc3:
            retention_offer_cost = st.number_input("Incentive Cost ($ / Customer)", 5.0, 100.0, 25.0, step=5.0)
        with rc4:
            campaign_success_rate = st.slider("Campaign Acceptance Rate (%)", 5, 60, 25, step=1)

        saved_customers = int(target_cohort_size * (campaign_success_rate / 100))
        total_campaign_cost = target_cohort_size * retention_offer_cost
        annual_revenue_saved = saved_customers * avg_user_mrr * 12
        net_profit_generated = annual_revenue_saved - total_campaign_cost
        roi_percent = (net_profit_generated / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Customers Retained <span>👥</span></div>
                <div class="kpi-value" style="color:#34D399;">{saved_customers:,}</div>
                <span class="kpi-badge badge-success">{campaign_success_rate}% Success</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Campaign Budget <span>💸</span></div>
                <div class="kpi-value">${total_campaign_cost:,.0f}</div>
                <span class="kpi-badge badge-info">${retention_offer_cost:.0f} / Target</span>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Annual Rev Saved <span>💵</span></div>
                <div class="kpi-value" style="color:#34D399;">${annual_revenue_saved:,.0f}</div>
                <span class="kpi-badge badge-success">12-Mo Horizon</span>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Net Campaign ROI <span>📈</span></div>
                <div class="kpi-value" style="color:#A5B4FC;">{roi_percent:,.0f}%</div>
                <span class="kpi-badge badge-success">${net_profit_generated:,.0f} Net Gain</span>
            </div>
            """, unsafe_allow_html=True)


    # 4-Pillar Strategic Retention Playbook
    rt_card = section_card("📋", "The 4-Pillar Executive Retention Playbook")
    with rt_card:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("""<div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 15px;">
    <div style="font-weight: 800; font-size: 1.05rem; color: #A5B4FC; margin-bottom: 6px;">
        🏛️ PILLAR 1: Contract Horizon Migration Program
    </div>
    <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
        <strong>Objective:</strong> Transition month-to-month subscribers into 1-year and 2-year contracted plans.<br>
        <strong>Tactic:</strong> Deliver targeted "Loyalty Upgrade" promos offering 1 month free or 10% discount on 12-month commitment.<br>
        <strong>Projected Impact:</strong> Reduces segment churn hazard by up to <strong>65%</strong>.
    </div>
</div>
<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px;">
    <div style="font-weight: 800; font-size: 1.05rem; color: #34D399; margin-bottom: 6px;">
        🛡️ PILLAR 2: "Security Shield" Ecosystem Bundling
    </div>
    <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
        <strong>Objective:</strong> Increase user touchpoints and sticky platform dependencies.<br>
        <strong>Tactic:</strong> Auto-bundle free Online Security & Tech Support for all Fiber Optic accounts in their first 6 months.<br>
        <strong>Projected Impact:</strong> Cuts Fiber Optic churn from <strong>41.9% down to &lt; 18%</strong>.
    </div>
</div>""", unsafe_allow_html=True)

        with p2:
            st.markdown("""<div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 15px;">
    <div style="font-weight: 800; font-size: 1.05rem; color: #FBBF24; margin-bottom: 6px;">
        🚀 PILLAR 3: "First-90-Days" Concierge Onboarding
    </div>
    <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
        <strong>Objective:</strong> Prevent high drop-offs during the vulnerable 0-6 month tenure window.<br>
        <strong>Tactic:</strong> Automated multi-channel onboarding check-ins, VIP routing on customer support lines, and bill clarity guides.<br>
        <strong>Projected Impact:</strong> Captures early detractors and improves 1-year subscriber retention by <strong>22%</strong>.
    </div>
</div>
<div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 18px;">
    <div style="font-weight: 800; font-size: 1.05rem; color: #60A5FA; margin-bottom: 6px;">
        💳 PILLAR 4: Frictionless Auto-Pay Migration
    </div>
    <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
        <strong>Objective:</strong> Eliminate payment failure and conscious bill-paying friction.<br>
        <strong>Tactic:</strong> Offer a recurring $5 monthly bill credit for migrating from manual Electronic Check to Automatic Card / ACH billing.<br>
        <strong>Projected Impact:</strong> Decreases payment-related attrition from <strong>45% to ~15%</strong>.
    </div>
</div>""", unsafe_allow_html=True)

