"""
Water Quality Index (WQI) prediction pipeline.

This module contains the core, reusable pieces of the analysis that was
originally prototyped in ``notebook.ipynb``:

- loading and cleaning the raw water-quality CSV
- computing the ground-truth WQI from standard sub-indices
- feature engineering
- training a stacked ensemble (RandomForest, XGBoost, LightGBM, SVR)

Run this file directly to execute the full pipeline end to end:

    python -m src.model
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

NUMERIC_COLUMNS = ["pH", "DO", "BOD", "TC", "FC", "Nitrate"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + [
    "DO_BOD_ratio",
    "pH_deviation",
    "log_fecal_coliform",
    "log_total_coliform",
]


def compute_wqi(df: pd.DataFrame) -> pd.Series:
    """Compute the Water Quality Index from its five sub-indices.

    Weights follow a standard CPCB-style approximation:
    DO 0.31, pH 0.11, BOD 0.19, FC 0.28, Nitrate 0.11.
    """
    q_do = np.clip(100 - (10 - df["DO"]) * 10, 0, 100)
    q_ph = np.clip(100 - np.abs(df["pH"] - 7.0) * 20, 0, 100)
    q_bod = np.clip(100 - df["BOD"] * 12, 0, 100)
    q_fc = np.clip(100 - (np.log1p(df["FC"]) / np.log1p(10000)) * 100, 0, 100)
    q_nitrate = np.clip(100 - df["Nitrate"] * 15, 0, 100)

    return 0.31 * q_do + 0.11 * q_ph + 0.19 * q_bod + 0.28 * q_fc + 0.11 * q_nitrate


def find_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Fuzzy-match a column name in ``df`` against a list of patterns."""
    for col in df.columns:
        for pattern in patterns:
            if pattern.lower() in col.lower():
                return col
    return None


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and normalize the raw water-quality CSV into standard column names."""
    try:
        df_raw = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(csv_path, encoding="latin1")

    df_raw.columns = df_raw.columns.str.strip()

    mapping = {}
    patterns = {
        "Station_Code": ["station code", "station_code", "stn_code", "station"],
        "pH": ["ph"],
        "DO": ["d.o.", "do (mg/l)", "dissolved oxygen"],
        "BOD": ["b.o.d.", "bod"],
        "TC": ["total coliform", "tc"],
        "FC": ["fecal coliform", "fc"],
        "Nitrate": ["nitrate", "nitrite"],
    }
    for target, pats in patterns.items():
        col = find_column(df_raw, pats)
        if col:
            mapping[col] = target

    df_raw = df_raw.rename(columns=mapping)

    if "Station_Code" not in df_raw.columns:
        df_raw["Station_Code"] = "STATION_01"

    missing = [c for c in NUMERIC_COLUMNS if c not in df_raw.columns]
    if missing:
        raise KeyError(
            f"Could not map required columns: {missing}. "
            f"Detected columns: {df_raw.columns.tolist()}"
        )

    for col in NUMERIC_COLUMNS:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    if "WQI" not in df_raw.columns:
        df_filled = df_raw.copy()
        df_filled[NUMERIC_COLUMNS] = df_filled[NUMERIC_COLUMNS].fillna(
            df_filled[NUMERIC_COLUMNS].median()
        )
        df_raw["WQI"] = compute_wqi(df_filled)

    return df_raw


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific derived features used by the models."""
    df_feat = df.copy()
    df_feat["DO_BOD_ratio"] = df_feat["DO"] / (df_feat["BOD"] + 1e-5)
    df_feat["pH_deviation"] = np.abs(df_feat["pH"] - 7.0)
    df_feat["log_fecal_coliform"] = np.log1p(df_feat["FC"])
    df_feat["log_total_coliform"] = np.log1p(df_feat["TC"])
    return df_feat


def build_stacked_ensemble() -> StackingRegressor:
    """Construct the regularized stacked ensemble (RF + XGB + LGBM + SVR -> Ridge)."""
    estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
            ),
        ),
        (
            "lgb",
            LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                num_leaves=20,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        ),
        ("svr", SVR(C=5.0, epsilon=0.1)),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=10.0))


def run_pipeline(csv_path: str = "data/dataset.csv") -> dict:
    """Run the full load -> engineer -> train -> evaluate pipeline and return metrics."""
    df_raw = load_dataset(csv_path)

    imputer = KNNImputer(n_neighbors=5)
    df_raw[NUMERIC_COLUMNS] = imputer.fit_transform(df_raw[NUMERIC_COLUMNS])

    df_engineered = apply_feature_engineering(df_raw)

    X = df_engineered[FEATURE_COLUMNS]
    y = df_engineered["WQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_stacked_ensemble()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    return metrics


if __name__ == "__main__":
    results = run_pipeline()
    print("Stacked Ensemble test performance:")
    for k, v in results.items():
        print(f"  {k.upper()}: {v:.4f}")
