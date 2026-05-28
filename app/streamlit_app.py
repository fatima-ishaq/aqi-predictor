"""
AQI Predictor Dashboard
-----------------------
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import aqi_category, compute_features

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
CITY      = os.getenv("CITY_NAME", "Karachi")

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "aqi_change_rate", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "pm25_pm10_ratio", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
]

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_resource
def load_models():
    models = {}
    for day in [1, 2, 3]:
        path = f"models/model_day_{day}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[day] = pickle.load(f)
    return models


@st.cache_data(ttl=3600)
def load_recent_features():
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    # Last 72 hours
    cutoff = datetime.utcnow() - timedelta(hours=72)
    docs   = list(col.find({"timestamp": {"$gte": cutoff}}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def make_predictions(df, models):
    if df.empty or not models:
        return {}

    latest = df.iloc[-1]
    X = latest[FEATURE_COLS].values.reshape(1, -1)

    predictions = {}
    for day, model in models.items():
        pred_aqi = float(model.predict(X)[0])
        pred_aqi = max(0, min(500, pred_aqi))  # clamp to valid range
        predictions[day] = {
            "aqi": round(pred_aqi),
            "category": aqi_category(pred_aqi),
            "date": (datetime.utcnow() + timedelta(days=day)).strftime("%A, %b %d"),
        }
    return predictions


# ─── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title=f"AQI Predictor — {CITY}", page_icon="🌬️", layout="wide")

st.title(f"🌬️ AQI Predictor — {CITY}")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

models  = load_models()
df      = load_recent_features()
preds   = make_predictions(df, models)

# ─── Current AQI ───────────────────────────────────────────────────────────────
if not df.empty:
    current_aqi      = int(df.iloc[-1]["aqi"])
    current_category = aqi_category(current_aqi)
    current_color    = AQI_COLORS.get(current_category, "#cccccc")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align:center; padding:20px; border-radius:16px; background:{current_color}22; border: 2px solid {current_color}'>
            <h1 style='color:{current_color}; font-size:72px; margin:0'>{current_aqi}</h1>
            <h3 style='margin:0'>Current AQI</h3>
            <p style='font-size:18px'>{current_category}</p>
        </div>
        """, unsafe_allow_html=True)

    # Alert for hazardous levels
    if current_aqi > 150:
        st.error(f"⚠️ **Air quality alert!** Current AQI is {current_aqi} ({current_category}). "
                 f"Avoid outdoor activities and wear a mask if going outside.")

# ─── 3-Day Forecast ────────────────────────────────────────────────────────────
st.subheader("📅 3-Day AQI Forecast")

if preds:
    cols = st.columns(3)
    for i, (day, info) in enumerate(preds.items()):
        color = AQI_COLORS.get(info["category"], "#cccccc")
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center; padding:16px; border-radius:12px;
                        background:{color}22; border:2px solid {color}'>
                <p style='margin:0; font-weight:bold'>{info["date"]}</p>
                <h2 style='color:{color}; margin:4px 0'>{info["aqi"]}</h2>
                <p style='margin:0; font-size:13px'>{info["category"]}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Models not found. Run the training pipeline first: `python pipelines/training_pipeline.py`")

# ─── Historical AQI Chart ──────────────────────────────────────────────────────
st.subheader("📈 AQI — Last 72 Hours")

if not df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines+markers", name="AQI",
        line=dict(color="#4f8ef7", width=2),
    ))
    # Threshold lines
    for label, threshold, color in [
    ("Moderate", 100, "#ffff00"),
    ("Unhealthy", 150, "#ff7e00"),
    ("Very Unhealthy", 200, "#ff0000")
]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color,
                      annotation_text=label)
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Time", yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)

# ─── Pollutant Breakdown ───────────────────────────────────────────────────────
st.subheader("🔬 Pollutant Levels (Last 72h)")

if not df.empty:
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
    fig2 = px.line(df, x="timestamp", y=pollutants,
                   labels={"value": "µg/m³", "variable": "Pollutant"},
                   color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─── Model Performance ─────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")

metrics_path = "models/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    mdf = pd.DataFrame(metrics).T.reset_index()
    mdf.columns = ["Forecast Horizon", "RMSE", "MAE", "R²"]
    mdf["Forecast Horizon"] = ["Day +1", "Day +2", "Day +3"]
    st.dataframe(mdf, use_container_width=True, hide_index=True)
else:
    st.info("Run training pipeline to see model metrics.")

# ─── AQI Guide ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ AQI Guide"):
    guide = pd.DataFrame([
        ("0–50",   "Good",                              "Air quality is satisfactory"),
        ("51–100", "Moderate",                          "Acceptable; some pollutants may affect sensitive people"),
        ("101–150","Unhealthy for Sensitive Groups",    "General public unaffected; sensitive groups at risk"),
        ("151–200","Unhealthy",                         "Everyone may experience health effects"),
        ("201–300","Very Unhealthy",                    "Health alert — serious effects"),
        ("301–500","Hazardous",                         "Emergency conditions"),
    ], columns=["AQI", "Category", "Health Implication"])
    st.dataframe(guide, use_container_width=True, hide_index=True)