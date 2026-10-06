import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from backend.model import VIZ_DIR, load_data
from backend.explainability import (
    explain_single,
    get_global_shap_values,
    get_shap_explainer,
    get_model as get_xai_model,
)
from frontend.styles import render_hero_banner, apply_plotly_theme, section_card

# Global SHAP analysis is deterministic and identical for every user, so it is
# computed once per process and reused across reruns instead of being
# recalculated whenever a widget on this page moves.
@st.cache_resource(show_spinner=False)
def _cached_explainer():
    return get_shap_explainer(get_xai_model())


@st.cache_data(show_spinner="Calculating SHAP values over a deterministic sample...")
def _cached_global_shap():
    return get_global_shap_values(get_xai_model(), _cached_explainer())


@st.cache_data(show_spinner=False)
def _cached_customer_explanation(profile_key, profile):
    """Explain one customer profile; cached per distinct profile."""
    return explain_single(profile, get_xai_model(), _cached_explainer())


CAUSALITY_DISCLAIMER = (
    "SHAP describes how **this fitted model** reached its score. Contributions "
    "are model attributions, not causal proof: a large positive contribution "
    "does not establish that changing the feature would change real-world churn."
)


def _direction_color(direction):
    return "#FB7185" if direction == "increases churn" else "#34D399"


def _render_local_bar(explanation, top_n=10):
    """Horizontal diverging bar chart of the top individual attributions."""
    rows = explanation["contributions"][:top_n]
    if not rows:
        st.info("No attributions available for this profile.")
        return
    labels = [r["label"] for r in rows][::-1]
    values = [r["contribution"] for r in rows][::-1]
    colors = [_direction_color(r["direction"]) for r in rows][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
    fig.update_layout(
        title="Model-Attributed Contribution to Churn Score (log-odds)",
        xaxis_title="SHAP value (positive = towards churn)",
        height=380, showlegend=False,
    )
    st.plotly_chart(apply_plotly_theme(fig, height=380), width="stretch")

def show_model_evaluation_and_explainability(df: pd.DataFrame, model):
    render_hero_banner(
        title="Model Leaderboard & Explainability (XAI)",
        subtitle="Benchmark classifier algorithms, inspect ROC-AUC & confusion matrix trade-offs, and explore global feature importance weights.",
        tag="ALGORITHM DIAGNOSTICS & XAI"
    )

    if model is None or not model.results:
        st.error("No model evaluation metrics loaded.")
        return

    metrics_data = []
    for name, res in model.results.items():
        row = {"Model Algorithm": name}
        for k, v in res["metrics"].items():
            row[k] = round(float(v), 4)
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data).sort_values("ROC-AUC", ascending=False)

    col_l1, col_l2 = st.columns([6, 4])
    with col_l1:
        rt_card = section_card("🏆", "Algorithm Performance Leaderboard")
        with rt_card:
            st.dataframe(
                metrics_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                    color="#312E81"
                ).format({
                    "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                    "Recall": "{:.3f}", "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}"
                }),
                width="stretch",
                hide_index=True
            )

            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 12px; margin-top: 15px;">
                🥇 <strong>Champion Model:</strong> <span style="color:#A5B4FC; font-weight:700;">{model.best_model_name}</span> achieved the top discrimination power with <strong>ROC-AUC = {metrics_df.iloc[0]['ROC-AUC']:.4f}</strong> and Accuracy = {metrics_df.iloc[0]['Accuracy']*100:.1f}%.
            </div>
            """, unsafe_allow_html=True)

    with col_l2:
        rt_card = section_card("📊", "Multi-Metric Model Comparison")
        with rt_card:
            fig_comp = go.Figure()
            metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            for idx, row in metrics_df.iterrows():
                fig_comp.add_trace(go.Bar(
                    name=row["Model Algorithm"],
                    x=metric_cols,
                    y=[row[m] for m in metric_cols]
                ))
            
            fig_comp.update_layout(barmode="group", yaxis_range=[0.4, 1.0])
            fig_comp = apply_plotly_theme(fig_comp, height=290)
            st.plotly_chart(fig_comp, width="stretch")

    # Feature Importance & Explainability
    rt_card = section_card("🔍", "Global Feature Importance Weights")
    with rt_card:
        feat_imp = model.get_feature_importance()
        if feat_imp is not None:
            c_f1, c_f2 = st.columns([7, 3])
            with c_f1:
                top_n = st.slider("Display Top N Features:", 5, min(25, len(feat_imp)), 12)
                top_df = feat_imp.head(top_n).sort_values("Importance", ascending=True)
            
                fig_imp = px.bar(
                    top_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title=f"Top {top_n} Predictive Signals ({model.best_model_name})"
                )
                fig_imp = apply_plotly_theme(fig_imp, height=360)
                st.plotly_chart(fig_imp, width="stretch")
            
            with c_f2:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 12px; font-size: 0.85rem; color: #CBD5E1; height: 100%;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">💡 Feature Insights:</div>
                    <ul style="margin: 0; padding-left: 18px; line-height: 1.6;">
                        <li><strong>Tenure & Contract Terms:</strong> Account duration and 2-year commitments are the strongest stabilizers.</li>
                        <li><strong>Internet Service Type:</strong> Fiber Optic is a strong positive churn signal due to pricing & competition.</li>
                        <li><strong>Total / Monthly Charges:</strong> Higher recurring price points increase sensitivity.</li>
                        <li><strong>Payment Method:</strong> Electronic Check increases churn probability significantly.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    # ---- SHAP Explainable AI (additive, in the model's log-odds space) ----
    # Placed AFTER the existing coefficient-based importance chart, which is
    # deliberately retained: coefficients answer "how is the model built",
    # SHAP answers "how did the model score these customers".
    try:
        global_shap = _cached_global_shap()
    except Exception as e:
        st.error(f"SHAP analysis unavailable: {e}")
        global_shap = None

    if global_shap is not None:
        rt_card = section_card("🧠", "SHAP Global Feature Attribution (Mean |SHAP|)")
        with rt_card:
            st.markdown(
                f"<p style='color:#94A3B8; font-size:0.85rem;'>"
                f"Mean absolute SHAP value over a deterministic sample of "
                f"<strong>{global_shap['sample_size']}</strong> customers "
                f"(random_state={global_shap['random_state']}). Larger bars mean "
                f"the feature moves the churn score further, on average.</p>",
                unsafe_allow_html=True,
            )
            g_top = st.slider(
                "Global SHAP Top N:",
                5, min(30, len(global_shap["importance"])), 12,
                key="shap_global_topn",
            )
            g_df = (global_shap["importance"]
                    .head(g_top)
                    .sort_values("Mean |SHAP|", ascending=True))

            g1, g2 = st.columns([7, 3])
            with g1:
                fig_glob = px.bar(
                    g_df, x="Mean |SHAP|", y="Label", orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "increases churn": "#FB7185",
                        "decreases churn": "#34D399",
                    },
                    title=f"Top {g_top} Features by Mean |SHAP|",
                )
                fig_glob = apply_plotly_theme(fig_glob, height=400)
                st.plotly_chart(fig_glob, width="stretch")
            with g2:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 12px; font-size: 0.82rem; color: #CBD5E1; height: 100%;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">📖 How to read this:</div>
                    <ul style="margin: 0; padding-left: 18px; line-height: 1.6;">
                        <li><strong>Mean |SHAP|</strong> = average size of a feature's push on the score, ignoring direction.</li>
                        <li><strong>Direction</strong> is evaluated where the feature is <em>active</em> (e.g. the category is present). A signed average is zero by construction and is never used.</li>
                        <li>This is a population average; it does not describe any single customer.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(
                global_shap["importance"]
                .head(15)[["Label", "Feature", "Mean |SHAP|", "Direction"]],
                width="stretch", hide_index=True,
            )

        # ---- Individual (local) explanation ----
        rt_card = section_card("🔬", "SHAP Individual Customer Explanation")
        with rt_card:
            try:
                xai_df = load_data().drop(columns=["Churn"])
                row_options = list(range(min(200, len(xai_df))))
                sel_row = st.selectbox(
                    "Select a customer from the dataset to explain:",
                    row_options,
                    format_func=lambda i: f"Customer row {i}",
                    key="shap_customer_row",
                )
                profile = xai_df.iloc[sel_row].to_dict()
                explanation = _cached_customer_explanation(sel_row, profile)

                im1, im2, im3 = st.columns(3)
                im1.metric("Churn Probability", f"{explanation['churn_probability']:.1f}%")
                im2.metric("Model Prediction", explanation["prediction"])
                im3.metric(
                    "Baseline (Avg Customer)",
                    f"{explanation['base_probability'] * 100:.1f}%",
                )

                st.caption(
                    f"SHAP base value (average log-odds) = {explanation['base_value']:.4f}. "
                    f"Additivity holds in {explanation['output_space']} space: "
                    f"base + sum of contributions reconstructs the model score "
                    f"(residual {explanation['reconstruction_error']:.2e})."
                )

                il1, il2 = st.columns([7, 3])
                with il1:
                    _render_local_bar(explanation, top_n=12)
                with il2:
                    up = explanation["top_increasing"][:5]
                    down = explanation["top_decreasing"][:5]
                    st.markdown("##### 🔺 Top Factors Increasing Churn")
                    if up:
                        for r in up:
                            val = r["value"]
                            val_s = f" — {val}" if val is not None else ""
                            st.markdown(
                                f"- **{r['label']}**{val_s}: "
                                f"`+{r['contribution']:.3f}` log-odds"
                            )
                    else:
                        st.caption("No feature pushes this customer towards churn.")

                    st.markdown("##### 🔻 Top Factors Decreasing Churn")
                    if down:
                        for r in down:
                            val = r["value"]
                            val_s = f" — {val}" if val is not None else ""
                            st.markdown(
                                f"- **{r['label']}**{val_s}: "
                                f"`{r['contribution']:.3f}` log-odds"
                            )
                    else:
                        st.caption("No feature pushes this customer away from churn.")

                st.info(CAUSALITY_DISCLAIMER, icon="⚖️")
            except Exception as e:
                st.error(f"Individual SHAP explanation failed: {e}")




    # Saved High-Res Visualizations Viewer
    rt_card = section_card("🖼️", "Publication-Ready Visualization Gallery")
    with rt_card:
        viz_files = [f for f in os.listdir(VIZ_DIR) if f.endswith(".png")] if os.path.exists(VIZ_DIR) else []
        if viz_files:
            selected_viz = st.selectbox("Select saved chart artifact to inspect:", viz_files)
            st.image(os.path.join(VIZ_DIR, selected_viz), width="stretch", caption=f"Saved Report Visualization: {selected_viz}")
        else:
            st.info("No static visualization artifacts found in visualizations folder.")
