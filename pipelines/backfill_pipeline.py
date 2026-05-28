"""
Backfill Pipeline
-----------------
Run ONCE manually to populate MongoDB with ~90 days of historical data.
Usage:
    python pipelines/backfill_pipeline.py --days 90
"""

import os
import argparse
import requests
import pandas as pd
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import compute_features

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
LAT       = float(os.getenv("CITY_LAT", 24.8607))
LON       = float(os.getenv("CITY_LON", 67.0011))
CITY      = os.getenv("CITY_NAME", "Karachi")


def fetch_historical(lat, lon, past_days=90):
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "auto", "past_days": past_days,
    }
    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "auto", "past_days": past_days,
    }
    aq_resp = requests.get(aq_url, params=aq_params, timeout=30).json()
    wx_resp = requests.get(wx_url, params=wx_params, timeout=30).json()
    return {"air_quality": aq_resp, "weather": wx_resp}


def parse_to_dataframe(raw):
    aq = raw["air_quality"]["hourly"]
    wx = raw["weather"]["hourly"]

    df_aq = pd.DataFrame({
        "timestamp":        aq["time"],
        "pm2_5":            aq["pm2_5"],
        "pm10":             aq["pm10"],
        "nitrogen_dioxide": aq["nitrogen_dioxide"],
        "ozone":            aq["ozone"],
    })

    df_wx = pd.DataFrame({
        "timestamp":              wx["time"],
        "temperature_2m":         wx["temperature_2m"],
        "relative_humidity_2m":   wx["relative_humidity_2m"],
        "wind_speed_10m":         wx["wind_speed_10m"],
        "precipitation":          wx["precipitation"],
    })

    # Merge on timestamp so mismatched lengths don't crash
    df = pd.merge(df_aq, df_wx, on="timestamp", how="inner")
    return df.dropna(subset=["pm2_5"])


def upsert_to_mongodb(df, city):
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{city.lower()}"]
    ops    = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["city"] = city
        doc["timestamp"] = pd.Timestamp(doc["timestamp"]).to_pydatetime()
        ops.append(UpdateOne(
            {"timestamp": doc["timestamp"], "city": city},
            {"$set": doc}, upsert=True,
        ))
    if ops:
        res = col.bulk_write(ops)
        print(f"Upserted: {res.upserted_count} | Modified: {res.modified_count}")
    client.close()


def run(past_days=90):
    print(f"Backfilling {past_days} days for {CITY}...")
    raw = fetch_historical(LAT, LON, past_days)
    df  = parse_to_dataframe(raw)
    df  = compute_features(df)
    print(f"Fetched {len(df)} rows. Storing to MongoDB...")
    upsert_to_mongodb(df, CITY)
    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run(args.days)