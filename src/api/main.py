import json
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import create_features


# --------------------------------------------------
# 1. Load the saved model and metadata
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"

print("\n--- MODEL PATH DEBUGGING ---")
print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())
print("Metadata path:", METADATA_PATH)
print("Metadata exists:", METADATA_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

THRESHOLD = metadata["threshold"]
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

THRESHOLD = metadata["threshold"]


# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Retention API",
    description="API for predicting customer churn risk.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. Define the customer input structure
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)

    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# --------------------------------------------------
# 4. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": metadata["model_name"]
    }


# --------------------------------------------------
# 5. Churn prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        # Convert incoming customer information into a DataFrame
        customer_df = pd.DataFrame([customer.model_dump()])

        # Generate the additional features
        customer_df = create_features(customer_df)

        # Predict churn probability
        probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        # Apply the selected classification threshold
        prediction = int(probability >= THRESHOLD)

        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "prediction_label": (
                "Churn risk detected"
                if prediction == 1
                else "No churn risk detected at selected threshold"
            ),
            "threshold": THRESHOLD
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )