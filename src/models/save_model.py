import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"


def save_model(model, threshold):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "XGBoost",
        "threshold": threshold,
        "target": "Churn",
        "positive_class": 1,
        "negative_class": 0,
        "features": list(model.feature_names_in_)
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")