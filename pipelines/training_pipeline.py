"""
Training Pipeline
-----------------
Runs daily via GitHub Actions.
Trains 3 separate models: Day+1, Day+2, Day+3 AQI prediction.
Logs to DagsHub (MLflow). Saves best model per horizon.
"""

import os
import mlflow
import dagshub
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle, json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

MONGO_URI      = os.getenv("MONGODB_URI")
CITY           = os.getenv("CITY_NAME", "Karachi")
DAGSHUB_USER   = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO   = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN  = os.getenv("DAGSHUB_TOKEN")

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "aqi_change_rate", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "pm25_pm10_ratio", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
]


def load_from_mongodb() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    docs   = list(col.find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train_model_for_day(df: pd.DataFrame, day_ahead: int):
    target_col = f"target_day_{day_ahead}"
    sub = df[FEATURE_COLS + [target_col]].dropna()

    X = sub[FEATURE_COLS].values
    y = sub[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    candidates = {
        "random_forest":        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting":    GradientBoostingRegressor(n_estimators=200, random_state=42),
        "ridge":                Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
    }

    best_model, best_rmse, best_name = None, float("inf"), ""

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        print(f"  Day+{day_ahead} | {name:25s} RMSE={rmse:.2f} MAE={mae:.2f} R²={r2:.3f}")

        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name

    return best_model, best_name, X_train, X_test, y_train, y_test


def log_shap(model, X_train, feature_names, day_ahead):
    """Generate and save SHAP summary plot."""
    try:
        # Use TreeExplainer for tree-based models, otherwise KernelExplainer
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train[:200])
        else:
            # For pipeline (Ridge), get the final estimator
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
            shap_values = explainer.shap_values(X_train[:50])

        plt.figure()
        shap.summary_plot(shap_values, X_train[:200], feature_names=feature_names, show=False)
        path = f"/tmp/shap_day{day_ahead}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(path)
    except Exception as e:
        print(f"  SHAP skipped: {e}")


def run():
    # Setup DagsHub + MLflow
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("aqi_predictor")

    print(f"Loading data from MongoDB for {CITY}...")
    df = load_from_mongodb()
    print(f"Loaded {len(df)} rows.")

    os.makedirs("models", exist_ok=True)
    metrics_summary = {}

    for day_ahead in [1, 2, 3]:
        print(f"\nTraining model for Day+{day_ahead}...")
        with mlflow.start_run(run_name=f"day_{day_ahead}"):
            model, model_name, X_train, X_test, y_train, y_test = train_model_for_day(df, day_ahead)

            preds = model.predict(X_test)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            mae   = mean_absolute_error(y_test, preds)
            r2    = r2_score(y_test, preds)

            mlflow.log_params({"model_type": model_name, "day_ahead": day_ahead, "city": CITY})
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(model, artifact_path=f"model_day_{day_ahead}")

            # SHAP feature importance
            log_shap(model, X_train, FEATURE_COLS, day_ahead)

            # Save locally too (for Streamlit app)
            model_path = f"models/model_day_{day_ahead}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            metrics_summary[f"day_{day_ahead}"] = {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 3)}
            print(f"  Best: {model_name} | RMSE={rmse:.2f} | saved to {model_path}")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nTraining complete. Metrics:", metrics_summary)


if __name__ == "__main__":
    run()