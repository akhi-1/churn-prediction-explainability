import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import build_matrix, engineer, load_raw  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_total_charges_is_numeric(raw):
    assert pd.api.types.is_numeric_dtype(raw["TotalCharges"])
    assert raw["TotalCharges"].isna().sum() == 0


def test_zero_tenure_customers_have_zero_total_charges(raw):
    zero_tenure = raw[raw["tenure"] == 0]
    assert len(zero_tenure) > 0, "expected some tenure-0 rows in this dataset"
    assert (zero_tenure["TotalCharges"] == 0).all()


def test_no_internet_service_collapsed(raw):
    df = engineer(raw)
    for col in ["OnlineSecurity", "TechSupport", "StreamingTV"]:
        assert "No internet service" not in df[col].unique()


def test_avg_monthly_spend_handles_zero_tenure(raw):
    df = engineer(raw)
    zero_tenure = df[df["tenure"] == 0]
    # Falls back to the current monthly charge rather than dividing by zero.
    assert (zero_tenure["avg_monthly_spend"] == zero_tenure["MonthlyCharges"]).all()
    assert df["avg_monthly_spend"].isna().sum() == 0


def test_num_addons_within_range(raw):
    df = engineer(raw)
    assert df["num_addons"].between(0, 6).all()


def test_matrix_has_no_target_leakage(raw):
    df = engineer(raw)
    X, y = build_matrix(df)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_matrix_is_fully_numeric(raw):
    X, _ = build_matrix(engineer(raw))
    assert X.select_dtypes(include="object").empty


def test_column_names_are_model_safe(raw):
    X, _ = build_matrix(engineer(raw))
    for c in X.columns:
        assert not any(ch in c for ch in "[]<> ")
