import streamlit as st
import pandas as pd
import plotly.express as px
import io
from backend.model import CATEGORICAL_COLS
from frontend.styles import render_hero_banner, apply_plotly_theme, section_card

def show_exploratory_intelligence(df: pd.DataFrame):
    render_hero_banner(
        title="Exploratory Data Intelligence",
        subtitle="Deep behavioral distributions, cross-feature correlations, service hierarchies, and cohort segmentations.",
        tag="DATASET & FEATURE ANATOMY"
    )

    t1, t2, t3, t4 = st.tabs([
        "📊 Categorical & Demographics", 
        "🔗 Correlation Heatmap", 
        "📈 Numerical Distributions", 
        "📋 Live Data Explorer"
    ])

    with t1:
        rt_card = section_card("👥", "Categorical Feature Distribution & Churn Breakdown")
        with rt_card:
            c1, c2 = st.columns([4, 6])
            with c1:
                selected_cat = st.selectbox(
                    "Select Categorical Attribute",
                    CATEGORICAL_COLS,
                    index=CATEGORICAL_COLS.index("Contract") if "Contract" in CATEGORICAL_COLS else 0
                )
            
                cat_summary = df.groupby(selected_cat)["Churn"].agg(
                    Total="count",
                    Churned=lambda x: (x == "Yes").sum(),
                    Churn_Rate=lambda x: round((x == "Yes").mean() * 100, 2)
                ).reset_index()
            
                st.dataframe(
                    cat_summary.style.format({"Total": "{:,}", "Churned": "{:,}", "Churn_Rate": "{:.2f}%"}),
                    width="stretch",
                    hide_index=True
                )

            with c2:
                fig_bar = px.bar(
                    df,
                    x=selected_cat,
                    color="Churn",
                    barmode="group",
                    title=f"Distribution of {selected_cat} by Churn Status",
                    color_discrete_map={"No": "#00E676", "Yes": "#FF3D71"},
                    category_orders={"Churn": ["No", "Yes"]},
                    labels={"Churn": "Churn Status", selected_cat: selected_cat}
                )
                fig_bar = apply_plotly_theme(fig_bar, height=320)

                # Explicitly enforce vibrant high-contrast trace colors and full opacity
                for trace in fig_bar.data:
                    if trace.name == "No":
                        trace.marker.color = "#00E676"
                        trace.marker.opacity = 1.0
                        trace.opacity = 1.0
                    elif trace.name == "Yes":
                        trace.marker.color = "#FF3D71"
                        trace.marker.opacity = 1.0
                        trace.opacity = 1.0

                fig_bar.update_layout(
                    plot_bgcolor="#0F172A",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=dict(
                        text=f"Distribution of {selected_cat} by Churn Status",
                        font=dict(size=13, color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
                        x=0,
                        xanchor="left",
                    ),
                    legend=dict(
                        title=dict(text="Churn Status", font=dict(size=12, color="#F8FAFC")),
                        font=dict(size=11, color="#F1F5F9"),
                        bgcolor="rgba(15, 23, 42, 0.9)",
                        bordercolor="rgba(255, 255, 255, 0.15)",
                        borderwidth=1,
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        zerolinecolor="rgba(255, 255, 255, 0.08)",
                        tickfont=dict(size=11, color="#CBD5E1"),
                        title=dict(font=dict(size=12, color="#94A3B8")),
                        automargin=True,
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        zerolinecolor="rgba(255, 255, 255, 0.08)",
                        tickfont=dict(size=11, color="#CBD5E1"),
                        title=dict(font=dict(size=12, color="#94A3B8")),
                        automargin=True,
                    ),
                )
                st.plotly_chart(fig_bar, width="stretch")
            

        rt_card = section_card("☀️", "Service Hierarchy Sunburst (Internet Service ➔ Contract ➔ Churn)")
        with rt_card:
            fig_sun = px.sunburst(
                df,
                path=["InternetService", "Contract", "Churn"],
                values="MonthlyCharges",
                color="Churn",
                color_discrete_map={"No": "#22C55E", "Yes": "#EF4444", "(?)": "#6366F1"},
                title="Monthly Revenue Allocation across Hierarchy"
            )
            fig_sun = apply_plotly_theme(fig_sun, height=430)
            st.plotly_chart(fig_sun, width="stretch")

    with t2:
        rt_card = section_card("🔗", "Pearson Feature Correlation Matrix")
        with rt_card:
            num_cols_corr = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
            corr_df = df[num_cols_corr].copy()
            corr_df["Churn_Numeric"] = (df["Churn"] == "Yes").astype(int)
            corr_matrix = corr_df.corr().round(3)
        
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Matrix with Churn Target"
            )
            fig_corr = apply_plotly_theme(fig_corr, height=380)
            st.plotly_chart(fig_corr, width="stretch")

            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 18px; border-radius: 10px; font-size: 0.85rem; color: #94A3B8;">
                💡 <strong>Key Correlation Insights:</strong>
                <ul style="margin: 6px 0 0 0; padding-left: 20px;">
                    <li><strong>Tenure</strong> has a strong negative correlation with Churn (-0.35) — older customers are far more loyal.</li>
                    <li><strong>Monthly Charges</strong> has a positive correlation with Churn (+0.19) — higher bills trigger price sensitivity.</li>
                    <li><strong>Total Charges</strong> correlates with Tenure (+0.83) as cumulative spend compounds over time.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with t3:
        rt_card = section_card("📈", "Numerical Feature Distributions & Outliers")
        with rt_card:
            selected_num = st.selectbox("Select Numerical Metric", ["MonthlyCharges", "TotalCharges", "tenure"])
        
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                fig_kde = px.histogram(
                    df,
                    x=selected_num,
                    color="Churn",
                    marginal="box",
                    barmode="overlay",
                    opacity=0.7,
                    color_discrete_map={"No": "#22C55E", "Yes": "#EF4444"},
                    title=f"Histogram & Boxplot for {selected_num}"
                )
                fig_kde = apply_plotly_theme(fig_kde, height=340)
                st.plotly_chart(fig_kde, width="stretch")
            
            with c_v2:
                fig_box = px.violin(
                    df,
                    y=selected_num,
                    x="Contract",
                    color="Churn",
                    box=True,
                    points="all",
                    color_discrete_map={"No": "#22C55E", "Yes": "#EF4444"},
                    title=f"Violin Distribution of {selected_num} by Contract Type"
                )
                fig_box = apply_plotly_theme(fig_box, height=340)
                st.plotly_chart(fig_box, width="stretch")
            

    with t4:
        rt_card = section_card("📋", "Interactive Dataset Explorer & Quick Filter")
        with rt_card:
            c_search, c_filter = st.columns([6, 4])
            with c_search:
                search_query = st.text_input("Search in customer data (e.g. Fiber optic, Two year, Female)", "")
            with c_filter:
                show_churn_only = st.checkbox("Show Only Churned Accounts", value=False)
            
            display_df = df.copy()
            if show_churn_only:
                display_df = display_df[display_df["Churn"] == "Yes"]
            if search_query:
                mask = display_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
                display_df = display_df[mask]

            st.dataframe(
                display_df.head(100),
                width="stretch",
                hide_index=True
            )
            st.caption(f"Showing {min(100, len(display_df)):,} of {len(display_df):,} matching records.")

            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Filtered Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="telco_customer_churn_export.csv",
                mime="text/csv"
            )
