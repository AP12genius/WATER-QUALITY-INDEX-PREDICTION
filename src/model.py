"""Leakage-safe one-year-ahead Water Quality Index forecasting."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_SEED = 42
NUMERIC_COLUMNS = ["pH", "DO", "BOD", "TC", "FC", "Nitrate"]
WQI_COLUMNS = ["pH", "DO", "BOD", "FC", "Nitrate"]
FEATURE_COLUMNS = ["WQI", *NUMERIC_COLUMNS, *[
    "DO_BOD_ratio",
    "pH_deviation",
    "log_fecal_coliform",
    "log_total_coliform",
]]
TRAIN_END_YEAR = 2012
VALIDATION_YEAR = 2013
TEST_YEAR = 2014


def compute_wqi(df: pd.DataFrame) -> pd.Series:
    """Return the documented five-input WQI approximation on a 0--100 scale."""
    q_do = np.clip(100 - (10 - df["DO"]) * 10, 0, 100)
    q_ph = np.clip(100 - np.abs(df["pH"] - 7.0) * 20, 0, 100)
    q_bod = np.clip(100 - df["BOD"] * 12, 0, 100)
    q_fc = np.clip(100 - np.log1p(df["FC"]) / np.log1p(10000) * 100, 0, 100)
    q_nitrate = np.clip(100 - df["Nitrate"] * 15, 0, 100)
    return 0.31 * q_do + 0.11 * q_ph + 0.19 * q_bod + 0.28 * q_fc + 0.11 * q_nitrate


def find_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return the first case-insensitive partial match for ``patterns``."""
    for column in df.columns:
        if any(pattern.lower() in column.lower() for pattern in patterns):
            return column
    return None


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV and normalize its required measurement and time columns."""
    try:
        dataset = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        dataset = pd.read_csv(csv_path, encoding="latin1")

    dataset.columns = dataset.columns.str.strip()
    patterns = {
        "Station_Code": ["station code", "station_code", "stn_code"],
        "pH": ["ph"],
        "DO": ["d.o.", "do (mg/l)", "dissolved oxygen"],
        "BOD": ["b.o.d.", "bod"],
        "TC": ["total coliform", "tc"],
        "FC": ["fecal coliform", "faecal coliform", "fc"],
        "Nitrate": ["nitrate", "nitrite"],
        "year": ["year"],
    }
    mapping = {
        source: target
        for target, options in patterns.items()
        if (source := find_column(dataset, options)) is not None
    }
    dataset = dataset.rename(columns=mapping)

    required = ["Station_Code", "year", *NUMERIC_COLUMNS]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise KeyError(f"Could not map required columns: {missing}")

    dataset["Station_Code"] = dataset["Station_Code"].astype("string").str.strip()
    dataset["Station_Code"] = dataset["Station_Code"].replace({"": pd.NA, "nan": pd.NA})
    dataset["year"] = pd.to_numeric(dataset["year"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    return dataset[required].copy()


def aggregate_station_years(dataset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate station/year measurements and calculate observed WQI."""
    valid = dataset.dropna(subset=["Station_Code", "year"]).copy()
    valid["year"] = valid["year"].astype(int)
    aggregated = (
        valid.groupby(["Station_Code", "year"], as_index=False)[NUMERIC_COLUMNS]
        .mean()
        .sort_values(["Station_Code", "year"])
        .reset_index(drop=True)
    )
    aggregated["WQI"] = compute_wqi(aggregated)
    return aggregated


def apply_feature_engineering(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add derived predictors from measurements available at the current year."""
    featured = dataset.copy()
    featured["DO_BOD_ratio"] = featured["DO"] / (featured["BOD"] + 1e-5)
    featured["pH_deviation"] = np.abs(featured["pH"] - 7.0)
    featured["log_fecal_coliform"] = np.log1p(featured["FC"].clip(lower=0))
    featured["log_total_coliform"] = np.log1p(featured["TC"].clip(lower=0))
    return featured


def build_forecast_pairs(station_years: pd.DataFrame) -> pd.DataFrame:
    """Pair each station's measurements with its WQI in the next calendar year."""
    current = station_years.rename(columns={"year": "feature_year"}).copy()
    future = station_years[["Station_Code", "year", "WQI"]].rename(
        columns={"year": "target_year", "WQI": "target_wqi"}
    )
    future["feature_year"] = future["target_year"] - 1
    pairs = current.merge(future, on=["Station_Code", "feature_year"], how="inner")
    return (
        pairs.dropna(subset=["WQI", "target_wqi"])
        .sort_values("target_year")
        .reset_index(drop=True)
    )


def split_by_target_year(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create immutable chronological train, validation, and test partitions."""
    return {
        "train": pairs.loc[pairs["target_year"] <= TRAIN_END_YEAR].copy(),
        "validation": pairs.loc[pairs["target_year"] == VALIDATION_YEAR].copy(),
        "test": pairs.loc[pairs["target_year"] == TEST_YEAR].copy(),
    }


def make_preprocessor() -> Pipeline:
    """Build preprocessing that is fit only on the training partition."""
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    )


def build_models() -> dict[str, object]:
    """Build regularized tabular regressors used by the benchmark."""
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbosity=-1,
        ),
        "svr": SVR(C=5.0, epsilon=0.1),
    }


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    """Compute the benchmark's standard regression metrics."""
    actual_array = np.asarray(list(actual))
    predicted_array = np.asarray(list(predicted))
    return {
        "r2": r2_score(actual_array, predicted_array),
        "mae": mean_absolute_error(actual_array, predicted_array),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "mse": mean_squared_error(actual_array, predicted_array),
    }


def _fit_models(train: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    preprocessor = make_preprocessor()
    features = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    target = train["target_wqi"]
    models = build_models()
    for model in models.values():
        model.fit(features, target)
    return preprocessor, models


def _predictions(
    preprocessor: Pipeline, models: dict[str, object], partition: pd.DataFrame
) -> dict[str, np.ndarray]:
    features = preprocessor.transform(partition[FEATURE_COLUMNS])
    return {name: model.predict(features) for name, model in models.items()}


def chronological_oof_predictions(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Generate base-model out-of-fold predictions using only earlier target years."""
    predictions: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    years = sorted(train["target_year"].unique())
    for year in years[1:]:
        fold_train = train.loc[train["target_year"] < year]
        fold_validation = train.loc[train["target_year"] == year]
        if fold_train.empty or fold_validation.empty:
            continue
        preprocessor, models = _fit_models(fold_train)
        predictions.append(pd.DataFrame(_predictions(preprocessor, models, fold_validation)))
        targets.append(fold_validation["target_wqi"].reset_index(drop=True))
    if not predictions:
        raise ValueError("Not enough chronological training years for stacked predictions.")
    return pd.concat(predictions, ignore_index=True), pd.concat(targets, ignore_index=True)


def run_pipeline(csv_path: str | Path = "data/dataset.csv") -> dict[str, dict[str, dict[str, float]]]:
    """Run the chronological benchmark and return metrics by model and partition."""
    station_years = aggregate_station_years(load_dataset(csv_path))
    pairs = build_forecast_pairs(apply_feature_engineering(station_years))
    partitions = split_by_target_year(pairs)
    if any(partition.empty for partition in partitions.values()):
        raise ValueError("Required chronological partitions are empty.")

    preprocessor, models = _fit_models(partitions["train"])
    oof_features, oof_target = chronological_oof_predictions(partitions["train"])
    meta_model = Ridge(alpha=10.0).fit(oof_features, oof_target)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, partition in partitions.items():
        target = partition["target_wqi"]
        model_predictions = _predictions(preprocessor, models, partition)
        model_predictions["median_baseline"] = np.full(len(partition), target.median())
        model_predictions["persistence_baseline"] = partition["WQI"].to_numpy()
        base_frame = pd.DataFrame({key: model_predictions[key] for key in models})
        model_predictions["stacked_ensemble"] = meta_model.predict(base_frame)
        results[name] = {
            model: regression_metrics(target, predicted)
            for model, predicted in model_predictions.items()
        }
    return results


def print_results(results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print metrics in a compact, copyable format."""
    for partition, models in results.items():
        print(f"\n{partition.title()} metrics")
        for model, metrics in models.items():
            print(
                f"  {model:22} R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  MSE={metrics['mse']:.4f}"
            )
    print("\nGeneralization warnings (train-to-validation R2 gap > 0.10)")
    for model, train_metrics in results["train"].items():
        validation_r2 = results["validation"][model]["r2"]
        gap = train_metrics["r2"] - validation_r2
        if gap > 0.10:
            print(f"  {model}: {gap:.4f}")


def generate_shap_summary(
    csv_path: str | Path, output_path: str | Path
) -> None:
    """Save a SHAP beeswarm plot for the fitted random forest on held-out data."""
    import shap

    station_years = aggregate_station_years(load_dataset(csv_path))
    pairs = build_forecast_pairs(apply_feature_engineering(station_years))
    partitions = split_by_target_year(pairs)
    preprocessor, models = _fit_models(partitions["train"])
    test_features = preprocessor.transform(partitions["test"][FEATURE_COLUMNS])
    explanation = shap.TreeExplainer(models["random_forest"])(test_features)
    shap.plots.beeswarm(explanation, show=False, max_display=len(FEATURE_COLUMNS))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.csv", help="Path to the CSV dataset.")
    parser.add_argument("--shap-output", help="Optional path for a random-forest SHAP plot.")
    args = parser.parse_args()
    results = run_pipeline(args.data)
    print_results(results)
    if args.shap_output:
        generate_shap_summary(args.data, args.shap_output)
        print(f"\nSaved SHAP summary to {args.shap_output}")


if __name__ == "__main__":
    main()
