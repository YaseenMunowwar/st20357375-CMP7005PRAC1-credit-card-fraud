from pathlib import Path
from typing import Any
import joblib
import pandas as pd

def load_models(models_dir: Path) -> dict[str, Any]:
    models = {
        "Logistic Regression": joblib.load(models_dir / "model_logistic.pkl"),
        "Random Forest": joblib.load(models_dir / "model_random_forest.pkl"),
        "Improved Random Forest (SMOTE)": {
            "model": joblib.load(models_dir / "model_random_forest_smote.pkl"),
            "preprocessor": joblib.load(models_dir / "preprocessor.pkl"),
        },
    }
    return models

def predict_transaction(model_name: str, input_data: dict[str, Any], models: dict[str, Any]) -> tuple[str, str]:
    input_df = pd.DataFrame([input_data])

    if model_name == "Improved Random Forest (SMOTE)":
        preprocessor = models[model_name]["preprocessor"]
        model = models[model_name]["model"]
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]
        explanation = (
            "The improved Random Forest model was trained using SMOTE to better learn "
            "fraudulent cases in this highly imbalanced dataset."
        )
    else:
        # Baseline models are saved as full pipelines
        pipeline = models[model_name]
        prediction = pipeline.predict(input_df)[0]
        explanation = (
            f"The prediction was generated using the {model_name} model selected from Task 3."
        )

    prediction_label = "Fraudulent Transaction" if prediction == 1 else "Non-Fraudulent Transaction"
    return prediction_label, explanation
