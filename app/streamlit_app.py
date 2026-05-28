"""
AQI Predictor Dashboard
-----------------------
Run with: streamlit run app/streamlit_app.py
Models are loaded from MongoDB GridFS (works locally AND on Streamlit Cloud).
"""

import os
import sys
import pickle
import json
import gridfs
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
    "Good":                              "#00e400",
    "Moderate":                          "#ffff00",
    "Unhealthy for Sensitive Groups":    "#ff7e00",
    "Unhealthy":                         "#ff0000",
    "Very Unhealthy":                    "#8f3f97",
    "Hazardous":                         "#7e0023",
}


# ── Data / model loaders ───────────────────────────────────────────────────────

@st.cache_resource(ttl=3600)
def load_models():
    """
    Load models from MongoDB GridFS.
    Falls back to local .pkl files if GridFS has nothing (first local run).
    """
    client = MongoClient(MONGO_URI)
    db     = client["aqi_db"]
    fs     = gridfs.GridFS(db, collection="models")
    models = {}

    for day in [1, 2, 3]:
        grid_out = fs.find_one({"filename": f"model_day_{day}"})
        if grid_out:
            models[day] = pickle.loads(grid_out.read())
        else:
            # Fallback: local file (useful during development)
            local = f"models/model_day_{day}.pkl"
            if os.path.exists(local):
                with open(local, "rb") as f:
                    models[day] = pickle.load(f)

    client.close()
    return models


@st.cache_data(ttl=3600)
def load_recent_features():
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    cutoff = datetime.utcnow() - timedelta(hours=168)  # last 7 days
    docs   = list(col.find({"timestamp": {"$gte": cutoff}}, {"_id": 0}))
    client.close()
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=86400)
def load_metrics():
    # Try local file first (fast), fallback to MongoDB
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            return json.load(f)
    return {}


def make_predictions(df: pd.DataFrame, models: dict) -> dict:
    if df.empty or not models:
        return {}
    # Use the most recent complete row
    valid_rows = df[FEATURE_COLS].dropna()
    if valid_rows.empty:
        return {}
    X = valid_rows.iloc[-1].values.reshape(1, -1)
    predictions = {}
    for day, model in models.items():
        pred_aqi = float(model.predict(X)[0])
        pred_aqi = max(0, min(500, pred_aqi))
        predictions[day] = {
            "aqi":      round(pred_aqi),
            "category": aqi_category(pred_aqi),
            "date":     (datetime.utcnow() + timedelta(days=day)).strftime("%A, %b %d"),
        }
    return predictions


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"AQI Predictor — {CITY}",
    page_icon="🌬️",
    layout="wide",
)

st.title(f"🌬️ AQI Predictor — {CITY}")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

models  = load_models()
df      = load_recent_features()
preds   = make_predictions(df, models)

# ── Current AQI ────────────────────────────────────────────────────────────────

if not df.empty and "aqi" in df.columns:
    recent_aqi       = df["aqi"].dropna()
    current_aqi      = int(recent_aqi.iloc[-1]) if not recent_aqi.empty else 0
    current_category = aqi_category(current_aqi)
    current_color    = AQI_COLORS.get(current_category, "#cccccc")

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align:center;padding:24px;border-radius:16px;
                    background:{current_color}22;border:2px solid {current_color}'>
            <h1 style='color:{current_color};font-size:80px;margin:0'>{current_aqi}</h1>
            <h3 style='margin:4px 0'>Current AQI</h3>
            <p style='font-size:18px;margin:0'>{current_category}</p>
        </div>
        """, unsafe_allow_html=True)

    if current_aqi > 150:
        st.error(
            f"⚠️ **Air quality alert!** AQI is {current_aqi} ({current_category}). "
            "Limit outdoor activity and wear a mask if going outside."
        )

# ── 3-Day Forecast ─────────────────────────────────────────────────────────────

st.subheader("📅 3-Day AQI Forecast")

if preds:
    cols = st.columns(3)
    for i, (day, info) in enumerate(preds.items()):
        color = AQI_COLORS.get(info["category"], "#cccccc")
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center;padding:20px;border-radius:12px;
                        background:{color}22;border:2px solid {color}'>
                <p style='margin:0;font-weight:bold'>{info["date"]}</p>
                <h2 style='color:{color};margin:6px 0'>{info["aqi"]}</h2>
                <p style='margin:0;font-size:13px'>{info["category"]}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Models not found. Run: `python pipelines/training_pipeline.py`")

# ── Historical AQI chart ───────────────────────────────────────────────────────

st.subheader("📈 AQI — Last 7 Days")

if not df.empty and "aqi" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines+markers", name="AQI",
        line=dict(color="#4f8ef7", width=2),
        marker=dict(size=3),
    ))
    thresholds = [
        (100, "#cccc00", "Moderate"),
        (150, "#ff7e00", "Unhealthy"),
        (200, "#ff0000", "Very Unhealthy"),
    ]
    for val, color, label in thresholds:
        fig.add_hline(
            y=val, line_dash="dot", line_color=color,
            annotation_text=label, annotation_position="right",
        )
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time", yaxis_title="AQI",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Pollutant breakdown ────────────────────────────────────────────────────────

st.subheader("🔬 Pollutant Levels (Last 7 Days)")

if not df.empty:
    available = [c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"] if c in df.columns]
    if available:
        fig2 = px.line(
            df, x="timestamp", y=available,
            labels={"value": "µg/m³", "variable": "Pollutant"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ── Model performance ──────────────────────────────────────────────────────────

st.subheader("📊 Model Performance")

metrics = load_metrics()
if metrics:
    rows = []
    for horizon, m in metrics.items():
        rows.append({
            "Forecast Horizon": horizon.replace("day_", "Day +"),
            "Best Model":       m.get("best_model", "—"),
            "RMSE":             m.get("rmse"),
            "MAE":              m.get("mae"),
            "R²":               m.get("r2"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Run the training pipeline to populate metrics.")

# ── AQI guide ─────────────────────────────────────────────────────────────────

with st.expander("ℹ️ AQI Reference Guide"):
    guide = pd.DataFrame([
        ("0–50",   "Good",                           "Air quality is satisfactory"),
        ("51–100", "Moderate",                       "Acceptable; some pollutants may affect sensitive people"),
        ("101–150","Unhealthy for Sensitive Groups", "General public unaffected; sensitive groups at risk"),
        ("151–200","Unhealthy",                      "Everyone may experience health effects"),
        ("201–300","Very Unhealthy",                 "Health alert — serious effects for everyone"),
        ("301–500","Hazardous",                      "Emergency conditions — entire population at risk"),
    ], columns=["AQI", "Category", "Health Implication"])
    st.dataframe(guide, use_container_width=True, hide_index=True)
