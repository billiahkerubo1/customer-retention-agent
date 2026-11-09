
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]
    assert result["threshold"] == 0.35


def test_predict_rejects_invalid_input():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 5
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
