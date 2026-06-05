# AQI Predictor

An automated machine learning system that predicts Air Quality Index (AQI) 1, 2, and 3 days in advance using weather data and historical air quality patterns.

## Overview

This system combines real-time weather and air quality data from Open-Meteo with feature engineering and multiple ML models to forecast AQI levels. The entire pipeline is automated via GitHub Actions, with results visualized on an interactive Streamlit dashboard.

**Deployed Application:** https://aqi-predictor-yxfvfm2vxstp8ugcuiphzn.streamlit.app/
**View Experiments on DagsHub:**https://dagshub.com/fatima-ishaq/aqi-predictor/experiments

Key metrics tracked per run:
- RMSE, MAE, R² for each model and horizon
- Model parameters
- SHAP explainability plots

## Features

- **Multi-day forecasting**: Predicts AQI for 24h, 48h, and 72h ahead
- **Multiple models**: Random Forest, Gradient Boosting, XGBoost, Voting Ensemble, and Keras Deep Learning
- **Automated pipelines**: Feature extraction runs hourly, model training runs daily
- **Interactive dashboard**: Real-time AQI display, 3-day forecast, historical trends, pollutant levels
- **Experiment tracking**: MLflow/DagsHub logging for model performance monitoring
- **Model registry**: MongoDB GridFS for storing and serving models

## Tech Stack

| Category | Technologies |
|----------|---------------|
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-learn, XGBoost, TensorFlow/Keras |
| **APIs** | Open-Meteo (weather + air quality) |
| **Database** | MongoDB (feature store, model artifacts, metrics) |
| **Frontend** | Streamlit, Plotly |
| **MLOps** | MLflow, DagsHub |
| **Automation** | GitHub Actions (cron schedules) |

## Project Structure

```
aqi-predictor/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── pipelines/
│   ├── backfill_pipeline.py      # Historical data (90 days)
│   ├── feature_pipeline.py       # Hourly data fetching
│   └── training_pipeline.py      # Daily model training
├── utils/
│   └── feature_engineering.py    # Feature computation & AQI conversion
├── models/                        # Local model backups
└── requirements.txt
```

## Pipelines

| Pipeline | Schedule | Description |
|----------|----------|-------------|
| **Backfill** | Manual (once) | Loads ~90 days of historical weather + AQI data |
| **Feature** | Hourly | Fetches latest data, engineers features, stores in MongoDB |
| **Training** | Daily | Trains 5 models per horizon, logs to MLflow, saves best to GridFS |

## How It Works

1. **Feature Pipeline** fetches current AQI (PM2.5, PM10, NO2, Ozone) and weather data (temperature, humidity, wind, precipitation) from Open-Meteo APIs
2. **Feature Engineering** creates time-based features (hour, day, month, weekend), rolling statistics, lag features, and target shifts (future AQI)
3. **Training Pipeline** uses 80/20 time-series split, compares 5 models, selects best by RMSE
4. **Dashboard** loads models from GridFS, shows current AQI + 3-day forecast + historical trends

## Quick Start

### 1. Clone & setup
```bash
git clone [your-repo]
cd aqi-predictor
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure `.env` file
```
MONGODB_URI=your_mongodb_connection_string
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO_NAME=your_repo
DAGSHUB_TOKEN=your_token
```

### 3. Backfill historical data (first time only)
```bash
python pipelines/backfill_pipeline.py --days 90
```

### 4. Run training locally (optional)
```bash
python pipelines/training_pipeline.py
```

### 5. Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

## Deployment

- **GitHub Actions** schedules feature pipeline (hourly) and training pipeline (daily)
- **Streamlit Cloud** hosts the dashboard with MongoDB connection
- Models are stored in MongoDB GridFS, accessible from both local and cloud

## Limitations

- Open-Meteo weather API only provides historical data for last 30-40 days (older dates have NaN weather)
- Free tier APIs may have occasional rate limits or downtime
- AQI spikes (e.g., week 21 anomaly) are difficult to predict with limited data


## Acknowledgments

- Open-Meteo for free weather and air quality APIs
- DagsHub for MLflow hosting
