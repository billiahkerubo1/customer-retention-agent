import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    df["average_monthly_spend"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, np.nan)
    )

    df["average_monthly_spend"] = (
        df["average_monthly_spend"]
        .fillna(df["MonthlyCharges"])
    )

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["service_count"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    return df