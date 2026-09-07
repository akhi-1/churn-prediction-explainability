"""Feature engineering for the Telco churn dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "telco.csv"

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SERVICE_COLS:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )
        
    addon_cols = [c for c in SERVICE_COLS if c != "MultipleLines"]
    df["num_addons"] = (df[addon_cols] == "Yes").sum(axis=1)

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    return df


def build_matrix(df: pd.DataFrame):
    """Return X (one-hot encoded) and y."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COL])

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X.columns = [
        c.replace(" ", "_").replace("[", "").replace("]", "").replace("-", "_")
        for c in X.columns
    ]


    return X, y


def load_prepared():
    df = engineer(load_raw())
    return build_matrix(df)
