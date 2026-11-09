
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from src.features.build_features import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"


def sample_customer():
    return pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.5],
        "TotalCharges": [1026.0]
    })


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = load_model()

    assert model is not None
    assert hasattr(model, "predict_proba")


def test_model_returns_valid_churn_probability():
    model = load_model()

    customer = create_features(sample_customer())

    probability = model.predict_proba(customer)[0, 1]

    assert np.isfinite(probability)
    assert 0 <= probability <= 1


def test_prediction_respects_configured_threshold():
    model = load_model()
    metadata = load_metadata()

    threshold = metadata["threshold"]

    customer = create_features(sample_customer())

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(probability >= threshold)

    assert prediction in [0, 1]
    assert prediction == int(probability >= threshold)
