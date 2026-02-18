import pandas as pd
import joblib
from src.models.predict import load_model, predict
from src.utils.logger import setup_logger

logger = setup_logger()

# List of required features for the model
REQUIRED_COLUMNS = joblib.load("artifacts/required_columns.pkl")

def validate_input(input_data: dict):
    """
    Validates input payload for inference.
    Ensures all required columns are present.
    """
    missing = set(REQUIRED_COLUMNS) - set(input_data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return True

def run_inference(input_data: dict):
    """
    End-to-end inference pipeline for Job Acceptance Prediction.
    """
    logger.info("Starting job acceptance inference pipeline")

    # Validate input
    validate_input(input_data)

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Load the trained model
    model = load_model()

    # Predict acceptance
    prediction, probabilities = predict(model, input_df)

    # Build result dictionary
    result = {
        "predicted_status": prediction[0],                # 0 = Not Accepted, 1 = Accepted
        "confidence": max(probabilities[0]) if probabilities is not None else None
    }

    logger.info(f"Inference result: {result}")
    return result
