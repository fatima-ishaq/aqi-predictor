"""
Training Pipeline
-----------------
Runs daily via GitHub Actions.
Trains 3 separate models: Day+1, Day+2, Day+3 AQI prediction.
Candidates: Random Forest, Gradient Boosting, XGBoost, Keras (deep learning).
Uses TimeSeriesSplit CV for stable model selection and metric logging.
Logs to DagsHub (MLflow). Saves best model to MongoDB GridFS + local pkl.
"""

import os
import mlflow
import tempfile
import gridfs
import numpy as np
import pandas as pd
import shap
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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


# ── Keras wrapper ──────────────────────────────────────────────────────────────

class KerasRegressorWrapper:
    def __init__(self, input_dim: int, epochs: int = 50, batch_size: int = 32):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self.model      = None

    def _build(self):
        from tensorflow import keras
        m = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(self.input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def fit(self, X, y):
        X_scaled   = self.scaler.fit_transform(X)
        self.model = self._build()
        self.model.fit(X_scaled, y, epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=0.1, verbose=0)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
            tmp.close()
            self.model.save(tmp.name)
            with open(tmp.name, "rb") as f:
                state["_model_bytes"] = f.read()
            os.unlink(tmp.name)
        state["model"] = None
        return state

    def __setstate__(self, state):
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes:
            from tensorflow import keras
            tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
            tmp.write(model_bytes)
            tmp.close()
            self.model = keras.models.load_model(tmp.name)
            os.unlink(tmp.name)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_from_mongodb() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    col    = client["aqi_db"][f"features_{CITY.lower()}"]
    docs   = list(col.find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model_to_mongodb(model, day_ahead: int):
    client = MongoClient(MONGO_URI)
    fs     = gridfs.GridFS(client["aqi_db"], collection="models")
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

    # Keep a final holdout for SHAP (not used for model selection)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=6,
            learning_rate=0.05, random_state=42,
            verbosity=0, n_jobs=-1,
        ),
        "keras_dense": KerasRegressorWrapper(
            input_dim=len(FEATURE_COLS), epochs=50, batch_size=32,
        ),
    }

    # TimeSeriesSplit — stable model selection across 5 windows
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_rmse   = {name: [] for name in candidates}
    cv_mae    = {name: [] for name in candidates}
    cv_r2     = {name: [] for name in candidates}
    cv_models = {name: None for name in candidates}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        for name, model in candidates.items():
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            cv_rmse[name].append(float(np.sqrt(mean_squared_error(y[test_idx], preds))))
            cv_mae[name].append(float(mean_absolute_error(y[test_idx], preds)))
            cv_r2[name].append(float(r2_score(y[test_idx], preds)))
            cv_models[name] = model

    # Aggregate CV metrics
    all_metrics = {}
    print(f"\n  {'Model':<25} {'CV RMSE':>10} {'±Std':>8} {'CV MAE':>8} {'CV R²':>8}")
    print(f"  {'-'*65}")
    for name in candidates:
        mean_rmse = float(np.mean(cv_rmse[name]))
        std_rmse  = float(np.std(cv_rmse[name]))
        mean_mae  = float(np.mean(cv_mae[name]))
        mean_r2   = float(np.mean(cv_r2[name]))
        print(f"  {name:<25} {mean_rmse:>10.2f} {std_rmse:>8.2f} {mean_mae:>8.2f} {mean_r2:>8.3f}")
        all_metrics[name] = {
            "rmse": mean_rmse,
            "mae":  mean_mae,
            "r2":   mean_r2,
            "std":  std_rmse,
        }

    # Winner = lowest average CV RMSE
    best_name  = min(cv_rmse, key=lambda k: np.mean(cv_rmse[k]))
    best_model = cv_models[best_name]
    print(f"\n  Winner for Day+{day_ahead}: {best_name} "
          f"(CV RMSE={all_metrics[best_name]['rmse']:.2f} "
          f"± {all_metrics[best_name]['std']:.2f})")

    return best_model, best_name, all_metrics, X_train, X_test, y_train, y_test


# ── SHAP ───────────────────────────────────────────────────────────────────────

def log_shap(model, X_train, feature_names: list, day_ahead: int):
    if not hasattr(model, "feature_importances_"):
        print(f"  SHAP skipped for Day+{day_ahead}: not a tree-based model")
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train[:200])
        fig, _    = plt.subplots()
        shap.summary_plot(shap_vals, X_train[:200],
                          feature_names=feature_names, show=False)
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_shap_day{day_ahead}.png", delete=False)
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
    # Direct MLflow auth — no dagshub.init OAuth needed in CI
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
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
            model, model_name, all_metrics, X_train, X_test, y_train, y_test = \
                train_model_for_day(df, day_ahead)

            # Use CV metrics for logging — stable across runs
            rmse = round(all_metrics[model_name]["rmse"], 2)
            mae  = round(all_metrics[model_name]["mae"],  2)
            r2   = round(all_metrics[model_name]["r2"],   3)
            std  = round(all_metrics[model_name]["std"],  2)

            mlflow.log_params({
                "model_type":   model_name,
                "day_ahead":    day_ahead,
                "city":         CITY,
                "n_features":   len(FEATURE_COLS),
                "n_rows":       len(df),
                "cv_folds":     5,
                **{f"cv_rmse_{k}": round(v["rmse"], 2)
                   for k, v in all_metrics.items()},
            })
            mlflow.log_metrics({
                "rmse":   rmse,
                "mae":    mae,
                "r2":     r2,
                "cv_std": std,
            })
            mlflow.sklearn.log_model(
                model,
                name=f"model_day_{day_ahead}",
                input_example=X_train[:1],
            )

            log_shap(model, X_train, FEATURE_COLS, day_ahead)

            # Save locally
            with open(f"models/model_day_{day_ahead}.pkl", "wb") as f:
                pickle.dump(model, f)

            # Save to GridFS for Streamlit Cloud
            save_model_to_mongodb(model, day_ahead)

            metrics_summary[f"day_{day_ahead}"] = {
                "rmse":       rmse,
                "mae":        mae,
                "r2":         r2,
                "cv_std":     std,
                "best_model": model_name,
                "all_models": {k: round(v["rmse"], 2)
                               for k, v in all_metrics.items()},
            }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete.")
    for horizon, m in metrics_summary.items():
        print(f"  {horizon}: {m['best_model']:<25} "
              f"CV RMSE={m['rmse']} ±{m['cv_std']} R²={m['r2']}")


if __name__ == "__main__":
    run()