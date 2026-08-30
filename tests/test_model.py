import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import apply_feature_engineering, compute_wqi, find_column, load_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def test_compute_wqi_perfect_water_is_near_100():
    df = pd.DataFrame(
        [{"pH": 7.0, "DO": 10.0, "BOD": 0.0, "FC": 0.0, "Nitrate": 0.0}]
    )
    wqi = compute_wqi(df)
    assert wqi.iloc[0] == pytest.approx(100.0, abs=1e-6)


def test_compute_wqi_severe_pollution_is_low():
    df = pd.DataFrame(
        [{"pH": 4.0, "DO": 0.5, "BOD": 50.0, "FC": 40000.0, "Nitrate": 12.0}]
    )
    wqi = compute_wqi(df)
    assert wqi.iloc[0] < 30


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
    assert find_column(df, ["d.o."]) == "D.O. (mg/l)"


def test_find_column_returns_none_when_no_match():
    df = pd.DataFrame(columns=["foo", "bar"])
    assert find_column(df, ["nonexistent"]) is None


def test_apply_feature_engineering_adds_expected_columns():
    df = pd.DataFrame(
        {"pH": [7.5], "DO": [8.0], "BOD": [1.0], "TC": [100.0], "FC": [10.0]}
    )
    df_feat = apply_feature_engineering(df)
    for col in [
        "DO_BOD_ratio",
        "pH_deviation",
        "log_fecal_coliform",
        "log_total_coliform",
    ]:
        assert col in df_feat.columns
    assert df_feat["pH_deviation"].iloc[0] == pytest.approx(0.5)
    assert df_feat["log_fecal_coliform"].iloc[0] == pytest.approx(np.log1p(10.0))


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset.csv not present")
def test_load_dataset_maps_required_columns():
    df = load_dataset(DATA_PATH)
    for col in ["pH", "DO", "BOD", "TC", "FC", "Nitrate", "WQI"]:
        assert col in df.columns
    assert len(df) > 0
