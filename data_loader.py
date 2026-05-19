"""Data acquisition, cleaning, and summary statistics for the Telco Churn dataset."""

import os
import requests
import pandas as pd
import numpy as np

from config import DATA_PATH, MONTHLY_REVENUE

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_data() -> None:
    """Download the IBM Telco Churn CSV to DATA_PATH if it is not already present."""
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    try:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        with open(DATA_PATH, "wb") as fh:
            fh.write(resp.content)
        print(f"Dataset downloaded to {DATA_PATH}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Telco Churn dataset from {URL}: {exc}"
        ) from exc


def load_data() -> pd.DataFrame:
    """Load, clean, and return the Telco Churn dataframe with a binary Churn column."""
    download_data()
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_stats(df: pd.DataFrame) -> dict:
    """Return a summary dict with churn_rate, total_customers, avg_tenure, avg_monthly, at_risk_revenue."""
    churn_rate      = df["Churn"].mean() * 100
    total_customers = len(df)
    avg_tenure      = df["tenure"].mean()
    avg_monthly     = df["MonthlyCharges"].mean()
    churned_count   = df["Churn"].sum()
    at_risk_revenue = churned_count * MONTHLY_REVENUE
    return {
        "churn_rate":      round(churn_rate, 2),
        "total_customers": total_customers,
        "avg_tenure":      round(avg_tenure, 1),
        "avg_monthly":     round(avg_monthly, 2),
        "at_risk_revenue": round(at_risk_revenue, 0),
    }
