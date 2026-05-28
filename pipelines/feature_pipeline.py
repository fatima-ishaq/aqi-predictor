"""
Feature Pipeline
----------------
Runs every hour via GitHub Actions.
1. Fetches current weather + air quality from Open-Meteo.
2. Engineers features.
3. Upserts into MongoDB (feature store).
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import compute_features

load_dotenv()

MONGO_URI  = os.getenv("MONGODB_URI")
LAT        = float(os.getenv("CITY_LAT", 24.8607))
LON        = float(os.getenv("CITY_LON", 67.0011))
CITY       = os.getenv("CITY_NAME", "Karachi")


def fetch_current_data(lat: float, lon: float) -> dict:
    """Fetch latest hourly data from Open-Meteo air quality + weather APIs."""

    # Air quality
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "auto", "past_days": 1, "forecast_days": 1,
    }
    aq_resp = requests.get(aq_url, params=aq_params, timeout=10).json()

    # Weather
    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "auto", "past_days": 1, "forecast_days": 1,
    }
    wx_resp = requests.get(wx_url, params=wx_params, timeout=10).json()

    return {"air_quality": aq_resp, "weather": wx_resp}


def parse_to_dataframe(raw: dict) -> pd.DataFrame:
    aq = raw["air_quality"]["hourly"]
    wx = raw["weather"]["hourly"]

    df = pd.DataFrame({
        "timestamp":            aq["time"],
        "pm2_5":                aq["pm2_5"],
        "pm10":                 aq["pm10"],
        "nitrogen_dioxide":     aq["nitrogen_dioxide"],
        "ozone":                aq["ozone"],
        "temperature_2m":       wx["temperature_2m"],
        "relative_humidity_2m": wx["relative_humidity_2m"],
        "wind_speed_10m":       wx["wind_speed_10m"],
        "precipitation":        wx["precipitation"],
    })
    df = df.dropna(subset=["pm2_5"])
    return df


def upsert_to_mongodb(df: pd.DataFrame, city: str):
    client = MongoClient(MONGO_URI)
    db     = client["aqi_db"]
    col    = db[f"features_{city.lower()}"]

    operations = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["city"] = city
        doc["timestamp"] = pd.Timestamp(doc["timestamp"]).to_pydatetime()
        operations.append(
            UpdateOne(
                {"timestamp": doc["timestamp"], "city": city},
                {"$set": doc},
                upsert=True,
            )
        )

    if operations:
        result = col.bulk_write(operations)
        print(f"[Feature Pipeline] Upserted {result.upserted_count} | Modified {result.modified_count}")

    client.close()


def run():
    print(f"[{datetime.now(timezone.utc)}] Running feature pipeline for {CITY}...")
    raw   = fetch_current_data(LAT, LON)
    df    = parse_to_dataframe(raw)
    df    = compute_features(df)
    upsert_to_mongodb(df, CITY)
    print("Done.")


if __name__ == "__main__":
    run()