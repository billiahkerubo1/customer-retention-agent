
import pandas as pd

from src.features.build_features import create_features


def sample_customers():
    return pd.DataFrame({
        "tenure": [12, 0],
        "TotalCharges": [1026.0, 0.0],
        "MonthlyCharges": [85.5, 50.0],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["Yes", "Yes"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"]
    })


def test_tenure_group_is_created_correctly():
    df = sample_customers()

    result = create_features(df)

    assert result.loc[0, "tenure_group"] == "0-12 months"
    assert result.loc[1, "tenure_group"] == "0-12 months"


def test_average_monthly_spend_is_calculated_correctly():
    df = sample_customers()

    result = create_features(df)

    assert result.loc[0, "average_monthly_spend"] == 85.5
    assert result.loc[1, "average_monthly_spend"] == 50.0


def test_service_count_is_calculated_correctly():
    df = sample_customers()

    result = create_features(df)

    assert result.loc[0, "service_count"] == 3
    assert result.loc[1, "service_count"] == 3


def test_original_dataframe_is_not_modified():
    df = sample_customers()
    original_columns = list(df.columns)

    create_features(df)

    assert list(df.columns) == original_columns
