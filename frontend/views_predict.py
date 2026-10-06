import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from backend.model import derive_total_charges, REQUIRED_COLS
from backend.explainability import explain_simulation
from backend.prediction_service import PredictionServiceError
from frontend.styles import render_hero_banner, apply_plotly_theme, section_card

def show_ai_prediction_and_simulator(df: pd.DataFrame, model):
    render_hero_banner(
        title="Real-Time AI Churn Predictor & Simulator",
        subtitle="Test individual subscriber profiles, evaluate live churn probabilities, and simulate retention intervention outcomes in real time.",
        tag="LIVE INFERENCE ENGINE"
    )

    if model is None:
        st.error(
            "⚠️ The active model bundle could not be loaded, so inference is "
            "unavailable. Check models/active_bundle.json and that the bundle "
            "it points to exists and passes its integrity check. Scoring will "
            "not fall back to legacy artifacts."
        )
        return

    rt_card = section_card("⚡", "Quick-Load Customer Archetype Presets")
    with rt_card:
        col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    
        if "form_gender" not in st.session_state:
            st.session_state.form_gender = "Female"
            st.session_state.form_senior = 0
            st.session_state.form_partner = "No"
            st.session_state.form_dependents = "No"
            st.session_state.form_tenure = 3
            st.session_state.form_phone = "Yes"
            st.session_state.form_lines = "No"
            st.session_state.form_internet = "Fiber optic"
            st.session_state.form_security = "No"
            st.session_state.form_backup = "No"
            st.session_state.form_protection = "No"
            st.session_state.form_tech = "No"
            st.session_state.form_tv = "Yes"
            st.session_state.form_movies = "Yes"
            st.session_state.form_contract = "Month-to-month"
            st.session_state.form_paperless = "Yes"
            st.session_state.form_payment = "Electronic check"
            st.session_state.form_monthly = 89.50

        with col_pre1:
            if st.button("🚨 High-Risk Newcomer", width="stretch"):
                st.session_state.update({
                    "form_gender": "Female", "form_senior": 0, "form_partner": "No", "form_dependents": "No",
                    "form_tenure": 2, "form_phone": "Yes", "form_lines": "No", "form_internet": "Fiber optic",
                    "form_security": "No", "form_backup": "No", "form_protection": "No", "form_tech": "No",
                    "form_tv": "Yes", "form_movies": "Yes", "form_contract": "Month-to-month",
                    "form_paperless": "Yes", "form_payment": "Electronic check", "form_monthly": 92.50
                })
                st.rerun()

        with col_pre2:
            if st.button("🛡️ 5-Year Loyal Family", width="stretch"):
                st.session_state.update({
                    "form_gender": "Male", "form_senior": 0, "form_partner": "Yes", "form_dependents": "Yes",
                    "form_tenure": 64, "form_phone": "Yes", "form_lines": "Yes", "form_internet": "DSL",
                    "form_security": "Yes", "form_backup": "Yes", "form_protection": "Yes", "form_tech": "Yes",
                    "form_tv": "Yes", "form_movies": "Yes", "form_contract": "Two year",
                    "form_paperless": "No", "form_payment": "Credit card (automatic)", "form_monthly": 75.20
                })
                st.rerun()

        with col_pre3:
            if st.button("💎 High-Spend Streamer", width="stretch"):
                st.session_state.update({
                    "form_gender": "Female", "form_senior": 1, "form_partner": "Yes", "form_dependents": "No",
                    "form_tenure": 12, "form_phone": "Yes", "form_lines": "Yes", "form_internet": "Fiber optic",
                    "form_security": "No", "form_backup": "Yes", "form_protection": "No", "form_tech": "No",
                    "form_tv": "Yes", "form_movies": "Yes", "form_contract": "Month-to-month",
                    "form_paperless": "Yes", "form_payment": "Electronic check", "form_monthly": 104.80
                })
                st.rerun()

        with col_pre4:
            if st.button("📦 Budget Saver", width="stretch"):
                st.session_state.update({
                    "form_gender": "Male", "form_senior": 0, "form_partner": "No", "form_dependents": "No",
                    "form_tenure": 24, "form_phone": "Yes", "form_lines": "No", "form_internet": "DSL",
                    "form_security": "Yes", "form_backup": "No", "form_protection": "No", "form_tech": "No",
                    "form_tv": "No", "form_movies": "No", "form_contract": "One year",
                    "form_paperless": "No", "form_payment": "Bank transfer (automatic)", "form_monthly": 29.85
                })
                st.rerun()


    col_input, col_output = st.columns([6, 5])

    with col_input:
        rt_card = section_card("📝", "Customer Account & Service Parameters")
        with rt_card:
            tab_dem, tab_srv, tab_bil = st.tabs(["👤 Demographics", "🌐 Services & Add-ons", "💳 Billing & Contract"])

            with tab_dem:
                cd1, cd2 = st.columns(2)
                with cd1:
                    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.form_gender == "Male" else 1)
                    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=st.session_state.form_senior)
                with cd2:
                    partner = st.selectbox("Has Partner", ["Yes", "No"], index=0 if st.session_state.form_partner == "Yes" else 1)
                    dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=0 if st.session_state.form_dependents == "Yes" else 1)
                tenure = st.slider("Tenure in Company (Months)", 0, 72, int(st.session_state.form_tenure))

            with tab_srv:
                cs1, cs2 = st.columns(2)
                with cs1:
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if st.session_state.form_phone == "Yes" else 1)
                    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=["Yes", "No", "No phone service"].index(st.session_state.form_lines))
                    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(st.session_state.form_internet))
                    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_security))
                with cs2:
                    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_backup))
                    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_protection))
                    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_tech))
                    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_tv))
                    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(st.session_state.form_movies))

            with tab_bil:
                cb1, cb2 = st.columns(2)
                with cb1:
                    contract = st.selectbox("Contract Duration", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state.form_contract))
                    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if st.session_state.form_paperless == "Yes" else 1)
                with cb2:
                    payment = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                    ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.form_payment))
                    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, float(st.session_state.form_monthly), step=0.50)
                
                total_charges = derive_total_charges(monthly_charges, tenure)
                st.caption(f"Estimated Cumulative Spend: **${total_charges:,.2f}**")


    # The payload is derived from the authoritative schema (REQUIRED_COLS) so the
    # 19 form values cannot silently drift from the model contract. Key order is
    # irrelevant to the backend, which re-orders columns from REQUIRED_COLS.
    form_values = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }
    assert set(form_values) == set(REQUIRED_COLS), (
        f"Form payload does not match the model schema. "
        f"Missing: {sorted(set(REQUIRED_COLS) - set(form_values))}, "
        f"extra: {sorted(set(form_values) - set(REQUIRED_COLS))}"
    )
    customer_payload = {col: form_values[col] for col in REQUIRED_COLS}

    pred_res = model.predict_single(customer_payload)
    churn_prob = pred_res["churn_probability"]

    with col_output:
        rt_card = section_card("🔮", "Live Churn Risk Assessment")
        with rt_card:
            if churn_prob >= 60:
                status_class, status_title, status_color, status_sub = "prediction-churn", "🚨 HIGH RISK OF CHURN", "#FB7185", "Critical flight risk. Immediate retention intervention recommended."
            elif churn_prob >= 30:
                status_class, status_title, status_color, status_sub = "prediction-churn", "⚠️ MODERATE RISK OF CHURN", "#FBBF24", "Moderate vulnerability. Proactive engagement advised."
            else:
                status_class, status_title, status_color, status_sub = "prediction-stay", "✅ LOW RISK / LOYAL SUBSCRIBER", "#34D399", "Healthy account engagement. Prime candidate for loyalty rewards."

            st.markdown(f"""
            <div class="prediction-hero-card {status_class}">
                <div class="prediction-status-title" style="color: {status_color};">{status_title}</div>
                <div style="font-size: 2.2rem; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #FFFFFF; margin: 6px 0;">
                    {churn_prob:.1f}% <span style="font-size: 0.95rem; color: #CBD5E1; font-weight: 500;">Churn Probability</span>
                </div>
                <div class="prediction-status-sub">{status_sub}</div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                number={'suffix': "%", 'font': {'color': "#FFFFFF", 'family': "JetBrains Mono"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#F43F5E" if churn_prob > 50 else "#10B981", 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.25)"},
                        {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [60, 100], 'color': "rgba(244, 63, 94, 0.25)"}
                    ],
                    'threshold': {'line': {'color': "#EF4444", 'width': 4}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig_gauge = apply_plotly_theme(fig_gauge, height=210)
            fig_gauge.update_layout(margin=dict(l=25, r=25, t=10, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; background:rgba(15,23,42,0.7); padding:10px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.06); font-size:0.85rem;">
                <span style="color:#94A3B8;">Annual Value at Risk:</span>
                <span style="font-weight:700; color:#FB7185; font-family:'JetBrains Mono';">${monthly_charges * 12:,.2f} / yr</span>
            </div>
            """, unsafe_allow_html=True)
    # What-If Retention Simulation Engine
    rt_card = section_card("🧪", "'What-If' Retention Scenario Simulator (Test Intervention Strategies)")
    with rt_card:
        st.markdown("<p style='color:#94A3B8; font-size:0.88rem;'>Test tactical changes to this subscriber plan to watch the exact drop in Churn Probability in real time:</p>", unsafe_allow_html=True)

        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        with sim_c1:
            sim_contract = st.selectbox("Simulate Contract Change", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(contract), key="sim_contract")
        with sim_c2:
            sim_tech = st.selectbox("Add Tech Support & Security?", ["Keep Current", "Add Tech Support & Online Security"], key="sim_tech")
        with sim_c3:
            sim_pay = st.selectbox("Switch to Auto-Pay?", ["Keep Current", "Switch to Credit Card Auto-Pay"], key="sim_pay")
        with sim_c4:
            sim_discount = st.slider("Monthly Retention Credit ($)", 0, 30, 0, step=5, key="sim_disc")

        sim_payload = customer_payload.copy()
        sim_payload["Contract"] = sim_contract
        if sim_tech == "Add Tech Support & Online Security":
            sim_payload["TechSupport"] = "Yes"
            sim_payload["OnlineSecurity"] = "Yes"
        if sim_pay == "Switch to Credit Card Auto-Pay":
            sim_payload["PaymentMethod"] = "Credit card (automatic)"
        sim_payload["MonthlyCharges"] = max(18.0, monthly_charges - sim_discount)
        sim_payload["TotalCharges"] = derive_total_charges(sim_payload["MonthlyCharges"], tenure)

        sim_res = model.predict_single(sim_payload)
        sim_prob = sim_res["churn_probability"]
        prob_delta = sim_prob - churn_prob

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Original Churn Risk", f"{churn_prob:.1f}%")
        with sc2:
            st.metric("Simulated Churn Risk", f"{sim_prob:.1f}%", delta=f"{prob_delta:.1f}%", delta_color="inverse")
        with sc3:
            saved_prob_pct = churn_prob - sim_prob
            st.metric("Risk Reduction Achieved", f"{saved_prob_pct:.1f}% Drop" if saved_prob_pct > 0 else "0% Drop")

        # ---- SHAP: how the MODEL'S OWN attribution shifts after the intervention.
        # Deliberately framed as a model-attributed change, never as a causal
        # claim: altering a field shows what this fitted model would score, not
        # what would happen in reality.
        try:
            sim_shap = explain_simulation(customer_payload, sim_payload, model)
            base_up = sim_shap["baseline"]["top_increasing"][:3]
            sim_up = sim_shap["simulated"]["top_increasing"][:3]

            st.markdown("""
            <div class="reco-box">
            <div style="font-weight:700; color:#A5B4FC; margin-bottom:8px; font-size:0.92rem;">🧠 SHAP: How the Model's Reasoning Changes</div>
            """, unsafe_allow_html=True)

            bs1, bs2 = st.columns(2)
            with bs1:
                st.markdown("**Baseline — top churn drivers (model-attributed)**")
                for r in base_up:
                    st.markdown(f"- {r['label']}: `+{r['contribution']:.3f}` log-odds")
            with bs2:
                st.markdown("**After intervention — top churn drivers**")
                for r in sim_up:
                    st.markdown(f"- {r['label']}: `+{r['contribution']:.3f}` log-odds")

            changed = [d for d in sim_shap["deltas"] if abs(d["delta"]) > 1e-9]
            if changed:
                st.markdown("**Largest model-attributed changes after the intervention:**")
                for d in changed[:5]:
                    arrow = "🔻 reduced churn score" if d["delta"] < 0 else "🔺 raised churn score"
                    st.markdown(
                        f"- {d['label']}: {d['baseline_contribution']:+.3f} → "
                        f"{d['simulated_contribution']:+.3f} log-odds "
                        f"(**{d['delta']:+.3f}**, {arrow})"
                    )
            else:
                st.caption("The selected intervention did not change any feature contribution.")

            st.caption(
                "Base value (average log-odds) = "
                f"{sim_shap['base_value']:.4f} — identical on both sides, so the "
                f"difference of {sim_shap['log_odds_delta']:+.4f} log-odds comes "
                f"only from the modified features."
            )
            st.info(
                "This is a **model-attributed contribution change**, not proof of "
                "cause and effect. SHAP shows how this fitted model scores the "
                "modified profile; it does not demonstrate that carrying out the "
                "intervention reduces real-world churn.",
                icon="⚖️",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP simulator explanation unavailable: {e}")

        st.markdown("""
        <div class="reco-box">
            <div style="font-weight:700; color:#A5B4FC; margin-bottom:8px; font-size:0.92rem;">🤖 AI Prescriptive Next-Best-Action (NBA):</div>
        """, unsafe_allow_html=True)

        if contract == "Month-to-month":
            st.markdown('<div class="reco-item">📌 <strong>Offer 1-Year Contract Lock-in:</strong> Upgrading from Month-to-Month will deliver the single largest reduction in churn hazard.</div>', unsafe_allow_html=True)
        if tech_support != "Yes" or online_security != "Yes":
            st.markdown('<div class="reco-item">🛡️ <strong>Bundle Cyber-Care Package:</strong> Free 3-month trial of Tech Support & Online Security will create strong product stickiness.</div>', unsafe_allow_html=True)
        if payment == "Electronic check":
            st.markdown('<div class="reco-item">💳 <strong>Incentivize Auto-Pay:</strong> Offer a one-time $10 account credit for switching from Electronic Check to Automatic Card Billing.</div>', unsafe_allow_html=True)
        if tenure <= 6:
            st.markdown('<div class="reco-item">📞 <strong>First-90-Day Concierge Outreach:</strong> Assign proactive check-in call from Customer Success team to verify satisfaction.</div>', unsafe_allow_html=True)
    
        st.markdown("</div>", unsafe_allow_html=True)

    # Batch Prediction & CSV Scoring Engine
    rt_card = section_card("📂", "Batch Inference & CSV Portfolio Scoring Engine")
    with rt_card:
        c_b1, c_b2 = st.columns([6, 4])
        with c_b1:
            st.markdown("<p style='color:#94A3B8; font-size:0.85rem;'>Upload an unlabelled customer CSV or click below to score a random sample of 50 accounts simultaneously.</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
            score_sample_btn = st.button("⚡ Score Sample Batch (50 Accounts)", type="secondary")

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df):,} accounts from uploaded file.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        elif score_sample_btn:
            batch_df = df.sample(min(50, len(df)), random_state=101).copy()

        if batch_df is not None:
            try:
                # Shared inference path: the production service applies the
                # bundle's own input contract (see
                # backend.input_contract.prepare_input_frame), so batch rows go
                # through exactly the same validation, imputation and encoding
                # as the single-customer predictor. Imputation uses the
                # persisted TRAINING median from bundle metadata, so a row's
                # score never depends on the other rows in the uploaded file.
                batch_preds, batch_prob_arr = model.predict_batch(batch_df)
                batch_probs = batch_prob_arr * 100

                batch_df["Predicted_Churn"] = list(batch_preds)
                batch_df["Churn_Probability (%)"] = batch_probs.round(2)
                batch_df["Risk_Tier"] = pd.cut(
                    batch_df["Churn_Probability (%)"],
                    bins=[-1, 30, 60, 100],
                    labels=["Low Risk", "Moderate Risk", "High Risk"]
                )

                high_risk_count = (batch_df["Risk_Tier"] == "High Risk").sum()
                at_risk_batch_rev = batch_df[batch_df["Risk_Tier"] == "High Risk"]["MonthlyCharges"].sum()

                b_m1, b_m2, b_m3 = st.columns(3)
                with b_m1:
                    st.metric("Batch Scored", f"{len(batch_df):,} Accounts")
                with b_m2:
                    st.metric("High Risk Accounts", f"{high_risk_count:,}", f"{(high_risk_count/len(batch_df)*100):.1f}% of cohort")
                with b_m3:
                    st.metric("High Risk Monthly Spend", f"${at_risk_batch_rev:,.2f}")

                cols_show = ["Contract", "tenure", "MonthlyCharges", "Predicted_Churn", "Churn_Probability (%)", "Risk_Tier"]
                if "customerID" in batch_df.columns:
                    cols_show = ["customerID"] + cols_show
            
                st.dataframe(batch_df[cols_show], width="stretch", hide_index=True)

                out_buf = io.StringIO()
                batch_df.to_csv(out_buf, index=False)
                st.download_button(
                    "📥 Download Scored Batch CSV",
                    data=out_buf.getvalue(),
                    file_name="scored_churn_predictions.csv",
                    mime="text/csv"
                )
            except PredictionServiceError as e:
                # A contract violation: the service already renders an
                # operational message naming the offending field, so the raw
                # traceback is never surfaced to the user.
                st.error(f"Batch prediction error: {e}")
            except Exception:
                st.error(
                    "Batch prediction failed while processing this file. "
                    "Please verify the uploaded CSV matches the expected "
                    "customer schema and try again."
                )


