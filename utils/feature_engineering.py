import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw dataframe with columns:
      timestamp, pm2_5, pm10, nitrogen_dioxide, ozone,
      temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation
    Returns enriched feature dataframe.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Time-based features ---
    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- AQI calculation (simplified US EPA PM2.5 formula) ---
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)

    # --- Derived features ---
    df["aqi_change_rate"]    = df["aqi"].diff().fillna(0)
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["aqi_rolling_std_3"]  = df["aqi"].rolling(3, min_periods=1).std().fillna(0)
    df["pm25_pm10_ratio"]    = (df["pm2_5"] / (df["pm10"] + 1e-5)).round(4)

    # --- Lag features (look back 1, 2, 3 hours) ---
    for lag in [1, 2, 3]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag).bfill()

    # --- Targets: AQI N days ahead ---
    # We predict daily average AQI, so we create targets as next-day values.
    # During backfill these will be real values; during inference they are NaN.
    for day_ahead in [1, 2, 3]:
        df[f"target_day_{day_ahead}"] = (
            df["aqi"].shift(-24 * day_ahead).fillna(np.nan)
        )

    return df


def pm25_to_aqi(pm25: float) -> float:
    """Converts PM2.5 concentration (µg/m³) to US AQI."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    return 500.0


def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"