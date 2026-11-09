
import json
import joblib
import numpy as np
import pandas as pd
import shap

from pathlib import Path
from src.features.build_features import create_features


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUSTOMER_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "telco_customer_churn_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "churn_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "model_metadata.json"
)


# Load customer dataset
if not CUSTOMER_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Customer dataset not found: {CUSTOMER_DATA_PATH}"
    )

CUSTOMER_DF = pd.read_csv(CUSTOMER_DATA_PATH)


# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Churn model not found: {MODEL_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {METADATA_PATH}"
    )

MODEL = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    MODEL_METADATA = json.load(file)

CHURN_THRESHOLD = MODEL_METADATA["threshold"]


# Initialize SHAP explainer when first needed
SHAP_EXPLAINER = None


def get_shap_explainer():
    global SHAP_EXPLAINER

    if SHAP_EXPLAINER is None:
        classifier = MODEL.named_steps["classifier"]
        SHAP_EXPLAINER = shap.TreeExplainer(classifier)

    return SHAP_EXPLAINER


# Retrieve customer record
def get_customer_record(customer_id: str):

    customer = CUSTOMER_DF.loc[
        CUSTOMER_DF["customerID"] == customer_id
    ]

    if customer.empty:
        raise ValueError(
            f"Customer {customer_id} was not found."
        )

    return customer.iloc[0]


# Tool 1: Retrieve customer profile
def get_customer_profile(customer_id: str):

    customer = get_customer_record(customer_id)

    return {
        "customer_id": customer["customerID"],
        "gender": customer["gender"],
        "senior_citizen": int(customer["SeniorCitizen"]),
        "partner": customer["Partner"],
        "dependents": customer["Dependents"],
        "tenure": int(customer["tenure"]),
        "contract": customer["Contract"],
        "monthly_charges": float(customer["MonthlyCharges"]),
        "total_charges": float(customer["TotalCharges"]),
        "internet_service": customer["InternetService"],
        "online_security": customer["OnlineSecurity"],
        "tech_support": customer["TechSupport"],
        "payment_method": customer["PaymentMethod"]
    }


# Tool 2: Predict customer churn
def predict_customer_churn(customer_id: str):

    get_customer_record(customer_id)

    customer_features = CUSTOMER_DF.loc[
        CUSTOMER_DF["customerID"] == customer_id
    ].drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    customer_features = create_features(customer_features)

    churn_probability = float(
        MODEL.predict_proba(customer_features)[0, 1]
    )

    churn_prediction = int(
        churn_probability >= CHURN_THRESHOLD
    )

    return {
        "customer_id": customer_id,
        "churn_probability": round(churn_probability, 4),
        "churn_prediction": churn_prediction,
        "prediction_label": (
            "Churn risk detected"
            if churn_prediction == 1
            else "Below selected churn-risk threshold"
        ),
        "threshold": CHURN_THRESHOLD,
        "model_name": MODEL_METADATA["model_name"]
    }


# Tool 3: Explain individual churn prediction using SHAP
def explain_customer_churn(
    customer_id: str,
    top_n: int = 5
):

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    get_customer_record(customer_id)

    customer_features = CUSTOMER_DF.loc[
        CUSTOMER_DF["customerID"] == customer_id
    ].drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # Apply the same feature engineering used during training.
    customer_features = create_features(customer_features)

    # Apply the fitted preprocessing pipeline.
    preprocessor = MODEL.named_steps["preprocessor"]

    transformed_features = preprocessor.transform(
        customer_features
    )

    if hasattr(transformed_features, "toarray"):
        transformed_features = transformed_features.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Calculate SHAP contributions.
    explainer = get_shap_explainer()

    shap_values = explainer.shap_values(
        transformed_features,
        check_additivity=False
    )

    # Handle different SHAP output formats.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]

    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    # Features pushing churn risk higher.
    increasing_risk = (
        contributions[
            contributions["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(top_n)
    )

    # Features pushing churn risk lower.
    decreasing_risk = (
        contributions[
            contributions["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(top_n)
    )

    return {
        "customer_id": customer_id,
        "explanation_method": "SHAP TreeExplainer",
        "contribution_scale": "Raw model output (log-odds)",

        "increasing_churn_risk": [
            {
                "feature": row["feature"],
                "contribution": round(
                    float(row["shap_value"]), 4
                )
            }
            for _, row in increasing_risk.iterrows()
        ],

        "decreasing_churn_risk": [
            {
                "feature": row["feature"],
                "contribution": round(
                    float(row["shap_value"]), 4
                )
            }
            for _, row in decreasing_risk.iterrows()
        ],

        "interpretation_note": (
            "Positive SHAP contributions push the model's "
            "churn score higher. Negative contributions push "
            "it lower. These explain model behaviour, not "
            "causation or the customer's actual reasons for leaving."
        )
    }
