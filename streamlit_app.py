"""
Food Security & Nutrition Monitoring Dashboard
================================================
IPC Phase classification and early warning system for Northern Nigeria.
Integrates crop production, market prices, malnutrition rates, and conflict
data to track food insecurity across LGAs and flag emergency zones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_food_security_dataset
from model import META_PATH, MODEL_PATH, load_model, save_model, train

IPC_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}
IPC_LABELS = {1: "Phase 1: Minimal", 2: "Phase 2: Stressed", 3: "Phase 3: Crisis",
              4: "Phase 4: Emergency", 5: "Phase 5: Famine"}

st.set_page_config(
    page_title="Food Security Dashboard | Nigeria",
    page_icon="🌾",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_food_security_dataset()


@st.cache_resource
def get_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        return load_model()
    df = load_data()
    clf, meta = train(df)
    save_model(clf, meta)
    return clf, meta


df = load_data()
clf, meta = get_model()

st.sidebar.title("🌾 Food Security Monitor")
page = st.sidebar.radio("Navigate", ["National Overview", "IPC Phase Map", "Trend Analysis", "Model Insights"])
year_filter = st.sidebar.multiselect("Year", sorted(df["year"].unique(), reverse=True), default=[])
state_filter = st.sidebar.multiselect("State", sorted(df["state"].unique()), default=[])
conflict_filter = st.sidebar.multiselect("Conflict Level", df["conflict_level"].unique().tolist(), default=[])

filtered = df.copy()
if year_filter:
    filtered = filtered[filtered["year"].isin(year_filter)]
if state_filter:
    filtered = filtered[filtered["state"].isin(state_filter)]
if conflict_filter:
    filtered = filtered[filtered["conflict_level"].isin(conflict_filter)]

# ── national overview ─────────────────────────────────────────────────────────
if page == "National Overview":
    st.title("Food Security & Nutrition Monitoring — Northern Nigeria")
    st.markdown(
        f"Tracking **{len(filtered):,} LGA-season observations** across 16 states "
        "using IPC phase classification."
    )

    col1, col2, col3, col4 = st.columns(4)
    crisis_above = filtered[filtered["ipc_phase"] >= 3]
    col1.metric("In Crisis+ (IPC ≥3)", f"{len(crisis_above):,}", f"{len(crisis_above)/len(filtered)*100:.1f}% of LGAs")
    col2.metric("Avg Wasting Rate", f"{filtered['wasting_pct'].mean():.1f}%")
    col3.metric("Population Food Insecure", f"{filtered['pct_food_insecure'].mean()*100:.1f}%", "avg across LGAs")
    col4.metric("Emergency/Famine (IPC ≥4)", f"{(filtered['ipc_phase'] >= 4).sum():,}")

    col_a, col_b = st.columns(2)
    with col_a:
        phase_counts = filtered["ipc_phase"].value_counts().reset_index()
        phase_counts.columns = ["phase", "count"]
        phase_counts["label"] = phase_counts["phase"].map(IPC_LABELS)
        phase_counts["color"] = phase_counts["phase"].map(IPC_COLORS)
        fig = px.bar(
            phase_counts.sort_values("phase"),
            x="label", y="count",
            color="phase",
            color_discrete_map=IPC_COLORS,
            title="LGA Distribution by IPC Phase",
            labels={"label": "IPC Phase", "count": "LGA Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        state_phase = (
            filtered.groupby("state")["ipc_phase"].mean().reset_index()
            .sort_values("ipc_phase", ascending=False)
        )
        fig2 = px.bar(
            state_phase, x="ipc_phase", y="state", orientation="h",
            color="ipc_phase",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
            title="Average IPC Phase by State",
            labels={"ipc_phase": "Avg IPC Phase", "state": ""},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Conflict Level vs IPC Phase")
    fig3 = px.box(
        filtered, x="conflict_level", y="ipc_phase",
        color="conflict_level",
        category_orders={"conflict_level": ["None", "Low", "Moderate", "High", "Severe"]},
        title="IPC Phase Distribution by Conflict Level",
        labels={"ipc_phase": "IPC Phase", "conflict_level": "Conflict Level"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── ipc phase map ─────────────────────────────────────────────────────────────
elif page == "IPC Phase Map":
    st.title("IPC Phase Geographic Distribution")
    metric = st.selectbox(
        "Map Metric",
        ["ipc_phase", "pct_food_insecure", "wasting_pct", "stunting_pct", "yield_kg_ha"],
    )
    sample = filtered.sample(min(2000, len(filtered)), random_state=42)
    scale = "RdYlGn_r" if metric in ("ipc_phase", "pct_food_insecure", "wasting_pct", "stunting_pct") else "YlGn"
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color=metric, color_continuous_scale=scale,
        size="population_total", size_max=16,
        hover_data=["state", "year", "season", "conflict_level", "ipc_phase"],
        zoom=5, height=560,
        mapbox_style="carto-positron",
        title=f"LGA Map — {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Index vs Food Insecurity Rate")
    fig2 = px.scatter(
        filtered.sample(min(1000, len(filtered)), random_state=2),
        x="price_index", y="pct_food_insecure",
        color="ipc_phase", color_discrete_map=IPC_COLORS,
        opacity=0.7, trendline="ols",
        labels={"price_index": "Staple Food Price Index (₦/kg)", "pct_food_insecure": "% Food Insecure"},
        title="Staple Prices vs Food Insecurity",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── trend analysis ────────────────────────────────────────────────────────────
elif page == "Trend Analysis":
    st.title("Multi-Year Food Security Trends")

    yearly = filtered.groupby("year").agg(
        avg_ipc=("ipc_phase", "mean"),
        crisis_pct=("ipc_phase", lambda x: (x >= 3).mean()),
        avg_wasting=("wasting_pct", "mean"),
        avg_yield=("yield_kg_ha", "mean"),
        avg_price=("price_index", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(yearly, x="year", y="avg_ipc", markers=True,
                      title="Average IPC Phase Over Time",
                      labels={"avg_ipc": "Avg IPC Phase"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.line(yearly, x="year", y="crisis_pct", markers=True,
                       title="% LGAs in Crisis or Worse (IPC ≥3)",
                       labels={"crisis_pct": "Proportion in Crisis+"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.line(yearly, x="year", y="avg_yield", markers=True,
                       title="Average Crop Yield (kg/ha)",
                       labels={"avg_yield": "Yield kg/ha"})
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.line(yearly, x="year", y="avg_price", markers=True,
                       title="Average Staple Food Price Index",
                       labels={"avg_price": "Price Index ₦/kg"})
        st.plotly_chart(fig4, use_container_width=True)

# ── model insights ────────────────────────────────────────────────────────────
elif page == "Model Insights":
    st.title("IPC Phase Classifier — Model Insights")
    col1, col2 = st.columns(2)
    col1.metric("CV Accuracy", f"{meta['cv_accuracy_mean']:.4f}", f"±{meta['cv_accuracy_std']:.4f}")
    col2.metric("Training Accuracy", f"{meta['train_accuracy']:.4f}")

    fi = pd.Series(meta["feature_importances"]).sort_values(ascending=True)
    fig = px.bar(
        fi.reset_index(), x=0, y="index", orientation="h",
        title="Feature Importances (Random Forest)",
        labels={0: "Importance", "index": "Feature"},
        color=0, color_continuous_scale="Greens",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("IPC Phase Reference")
    ipc_ref = pd.DataFrame([
        {"Phase": k, "Label": v, "Description": desc}
        for (k, v), desc in zip(IPC_LABELS.items(), [
            "Adequate food access; less than 20% households with poor diet",
            "Borderline adequate food; increased coping strategies",
            "Food consumption gaps; accelerated asset depletion",
            "Large food consumption gaps; high malnutrition & mortality",
            "Starvation, death, and destitution — catastrophic scale",
        ])
    ])
    st.table(ipc_ref)
