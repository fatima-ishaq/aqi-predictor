"""
Training Pipeline
-----------------
Runs daily via GitHub Actions.
Trains 3 separate models: Day+1, Day+2, Day+3 AQI prediction.
Logs to DagsHub (MLflow). Saves best model to MongoDB GridFS + local pkl.
"""

import os
import mlflow
import tempfile
import dagshub
import gridfs
import numpy as np
import pandas as pd
import shap
import pickle
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for CI runners
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

MONGO_URI     = os.getenv("MONGODB_URI")
CITY          = os.getenv("CITY_NAME", "Karachi")
DAGSHUB_USER  = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "aqi_change_rate", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "pm25_pm10_ratio", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_from_mongodb() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    docs   = list(col.find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Model saving (GridFS) ──────────────────────────────────────────────────────

def save_model_to_mongodb(model, day_ahead: int):
    """Persist model bytes to MongoDB GridFS so Streamlit Cloud can load them."""
    client = MongoClient(MONGO_URI)
    db     = client["aqi_db"]
    fs     = gridfs.GridFS(db, collection="models")

    # Remove previous version for this day
    for old in fs.find({"filename": f"model_day_{day_ahead}"}):
        fs.delete(old._id)

    fs.put(pickle.dumps(model), filename=f"model_day_{day_ahead}", city=CITY)
    client.close()
    print(f"  Saved model_day_{day_ahead} to MongoDB GridFS")


# ── Training ───────────────────────────────────────────────────────────────────

def train_model_for_day(df: pd.DataFrame, day_ahead: int):
    target_col = f"target_day_{day_ahead}"
    sub = df[FEATURE_COLS + [target_col]].dropna()

    X = sub[FEATURE_COLS].values
    y = sub[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=42
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=6,
            learning_rate=0.05, random_state=42,
            verbosity=0, n_jobs=-1
        ),
    }

    best_model, best_rmse, best_name = None, float("inf"), ""

    print(f"\n  {'Model':<25} {'Train RMSE':>10} {'Test RMSE':>10} {'MAE':>8} {'R²':>7} {'Gap':>8}")
    print(f"  {'-'*70}")

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        rmse       = np.sqrt(mean_squared_error(y_test,  test_preds))
        mae        = mean_absolute_error(y_test, test_preds)
        r2         = r2_score(y_test, test_preds)
        gap        = rmse - train_rmse

        print(f"  {name:<25} {train_rmse:>10.2f} {rmse:>10.2f} {mae:>8.2f} {r2:>7.3f} {gap:>8.2f}")

        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name

    print(f"\n  Winner for Day+{day_ahead}: {best_name} (RMSE={best_rmse:.2f})")
    return best_model, best_name, X_train, X_test, y_train, y_test


# ── SHAP ───────────────────────────────────────────────────────────────────────

def log_shap(model, X_train, feature_names: list, day_ahead: int):
    try:
        if hasattr(model, "feature_importances_"):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_train[:200])
            plot_X     = X_train[:200]
        else:
            explainer  = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
            shap_vals  = explainer.shap_values(X_train[:50])
            plot_X     = X_train[:50]

        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, plot_X, feature_names=feature_names, show=False)

        # NamedTemporaryFile works on both Windows and Linux CI runners
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_shap_day{day_ahead}.png", delete=False
        )
        tmp.close()
        plt.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(tmp.name)
        os.unlink(tmp.name)
        print(f"  SHAP plot logged for Day+{day_ahead}")

    except Exception as e:
        print(f"  SHAP skipped for Day+{day_ahead}: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("aqi_predictor")

    print(f"Loading data from MongoDB for {CITY}...")
    df = load_from_mongodb()
    print(f"Loaded {len(df)} rows.")

    os.makedirs("models", exist_ok=True)
    metrics_summary = {}

    for day_ahead in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Training model for Day+{day_ahead}...")

        with mlflow.start_run(run_name=f"day_{day_ahead}"):
            model, model_name, X_train, X_test, y_train, y_test = \
                train_model_for_day(df, day_ahead)

            preds = model.predict(X_test)
            rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae   = float(mean_absolute_error(y_test, preds))
            r2    = float(r2_score(y_test, preds))

            # Log to DagsHub / MLflow
            mlflow.log_params({
                "model_type": model_name,
                "day_ahead":  day_ahead,
                "city":       CITY,
                "n_features": len(FEATURE_COLS),
            })
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(
                model,
                name=f"model_day_{day_ahead}",
                input_example=X_train[:1],   # silences the signature warning
            )

            # SHAP
            log_shap(model, X_train, FEATURE_COLS, day_ahead)

            # Save locally (for local Streamlit runs)
            local_path = f"models/model_day_{day_ahead}.pkl"
            with open(local_path, "wb") as f:
                pickle.dump(model, f)

            # Save to MongoDB GridFS (for Streamlit Cloud / CI)
            save_model_to_mongodb(model, day_ahead)

            metrics_summary[f"day_{day_ahead}"] = {
                "rmse": round(rmse, 2),
                "mae":  round(mae,  2),
                "r2":   round(r2,   3),
                "best_model": model_name,
            }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete.")
    for horizon, m in metrics_summary.items():
        print(f"  {horizon}: {m['best_model']:<25} RMSE={m['rmse']} R²={m['r2']}")


if __name__ == "__main__":
    run()
