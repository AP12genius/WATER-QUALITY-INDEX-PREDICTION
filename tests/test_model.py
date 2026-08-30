import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    TEST_YEAR,
    TRAIN_END_YEAR,
    VALIDATION_YEAR,
    aggregate_station_years,
    apply_feature_engineering,
    build_forecast_pairs,
    compute_wqi,
    find_column,
    load_dataset,
    split_by_target_year,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def _station_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Station_Code": ["A", "A", "A", "A", "B"],
            "year": [2011, 2011, 2012, 2014, 2012],
            "pH": [7.0, 7.2, 7.1, 7.3, 7.0],
            "DO": [8.0, 8.0, 7.0, 6.0, 8.0],
            "BOD": [2.0, 2.0, 3.0, 4.0, 2.0],
            "TC": [100.0, 100.0, 200.0, 300.0, 100.0],
            "FC": [10.0, 10.0, 20.0, 30.0, 10.0],
            "Nitrate": [1.0, 1.0, 2.0, 3.0, 1.0],
        }
    )


def test_compute_wqi_perfect_water_is_near_100():
    df = pd.DataFrame(
        [{"pH": 7.0, "DO": 10.0, "BOD": 0.0, "FC": 0.0, "Nitrate": 0.0}]
    )
    assert compute_wqi(df).iloc[0] == pytest.approx(100.0, abs=1e-6)


def test_compute_wqi_output_is_bounded():
    df = pd.DataFrame(
        {
            "pH": [0.0, 14.0, 7.0],
            "DO": [0.0, 20.0, 8.0],
            "BOD": [100.0, 0.0, 2.0],
            "FC": [0.0, 1e6, 100.0],
            "Nitrate": [0.0, 50.0, 1.0],
        }
    )
    wqi = compute_wqi(df)
    assert (wqi >= 0).all() and (wqi <= 100).all()


def test_find_column_matches_case_insensitively():
    df = pd.DataFrame(columns=["Station Code", "PH", "D.O. (mg/l)"])
    assert find_column(df, ["station code"]) == "Station Code"
    assert find_column(df, ["ph"]) == "PH"


def test_aggregate_station_years_averages_duplicates():
    aggregated = aggregate_station_years(_station_data())
    row = aggregated.loc[(aggregated["Station_Code"] == "A") & (aggregated["year"] == 2011)]
    assert len(row) == 1
    assert row.iloc[0]["pH"] == pytest.approx(7.1)


def test_build_forecast_pairs_uses_only_consecutive_years():
    pairs = build_forecast_pairs(aggregate_station_years(_station_data()))
    assert pairs[["Station_Code", "feature_year", "target_year"]].to_dict("records") == [
        {"Station_Code": "A", "feature_year": 2011, "target_year": 2012}
    ]


def test_apply_feature_engineering_adds_expected_columns():
    df = _station_data().iloc[:1]
    featured = apply_feature_engineering(df)
    assert {
        "DO_BOD_ratio",
        "pH_deviation",
        "log_fecal_coliform",
        "log_total_coliform",
    }.issubset(featured)
    assert featured["log_fecal_coliform"].iloc[0] == pytest.approx(np.log1p(10.0))


def test_split_by_target_year_has_fixed_boundaries():
    pairs = pd.DataFrame({"target_year": [2012, 2013, 2014, 2015]})
    partitions = split_by_target_year(pairs)
    assert partitions["train"]["target_year"].max() <= TRAIN_END_YEAR
    assert partitions["validation"]["target_year"].tolist() == [VALIDATION_YEAR]
    assert partitions["test"]["target_year"].tolist() == [TEST_YEAR]


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset.csv not present")
def test_load_dataset_maps_required_columns():
    dataset = load_dataset(DATA_PATH)
    assert {"Station_Code", "year", "pH", "DO", "BOD", "TC", "FC", "Nitrate"}.issubset(
        dataset.columns
    )
    assert len(dataset) > 0
