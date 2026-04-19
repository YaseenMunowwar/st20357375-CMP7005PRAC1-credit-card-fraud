from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import os

MODELS_DIR = Path("models")

def load_model_pipeline(name):
    try:
        path = MODELS_DIR / f"{name}.pkl"
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        return None

def load_preprocessor():
    try:
        path = MODELS_DIR / "preprocessor.pkl"
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return None

def load_models(models_dir: Path) -> dict[str, Any]:
    global MODELS_DIR
    MODELS_DIR = models_dir
    models = {
        "Logistic Regression": load_model_pipeline("model_logistic"),
        "Random Forest": load_model_pipeline("model_random_forest"),
        "Improved Random Forest (SMOTE)": {
            "model": load_model_pipeline("model_random_forest_smote"),
            "preprocessor": load_preprocessor(),
        },
    }
    return models

def predict_transaction(models: dict[str, Any], model_name: str, input_data: dict[str, Any]) -> tuple[str, str]:
    """
    Executes a fraud risk assessment using the specified machine learning model.
    
    Args:
        models: Dictionary of pre-loaded model pipelines.
        model_name: The identifier of the model to be utilized.
        input_data: Raw input dictionary from the web form.
        
    Returns:
        A tuple containing (prediction_label, and a technical explanation).
    """
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
