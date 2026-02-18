import pandas as pd
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------- PATHS ----------------
MODEL_PATH = "artifacts/models/final_rf_top_features.pkl"
REQUIRED_COLUMNS_PATH = "artifacts/required_columns_top_features.pkl"

# ---------------- LOAD MODEL & REQUIRED COLUMNS ----------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(REQUIRED_COLUMNS_PATH):
    logger.error("Model or required columns file not found!")
    raise FileNotFoundError("Please train the model and save required columns first.")

model = joblib.load(MODEL_PATH)
required_columns = joblib.load(REQUIRED_COLUMNS_PATH)
logger.info("Loaded Top-Feature RandomForest model and required columns successfully.")

# ---------------- PREDICT FUNCTION ----------------
def predict_candidate(input_dict: dict):
    """
    Predict placement status for a single candidate or batch.

    input_dict: dictionary with candidate features, e.g.
        {
            "technical_score": 70,
            "years_of_experience": 3,
            "interview_score": 180,
            "skills_match_percentage": 75,
            "communication_score": 65
        }
    """
    input_df = pd.DataFrame([input_dict])

    # Add missing required columns with 0
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns as required by the model
    input_df = input_df[required_columns]

    # Make prediction
    pred = model.predict(input_df)

    # Convert numeric prediction to human-readable
    result = ["Placed" if p == 1 else "Not Placed" for p in pred]

    return result

# ---------------- TEST ----------------
if __name__ == "__main__":
    # Example input
    candidate = {
        "technical_score": 70,
        "years_of_experience": 2,
        "aptitude_score": 90,
        "skills_match_percentage": 75,
        "communication_score": 70
    }
    prediction = predict_candidate(candidate)
    status = "Placed" if prediction[0] == 1 else "Not Placed"
    print(f"Candidate Prediction: {status}")
    logger.info(f"Candidate Prediction: {status}")
    
    #logger.info(f"Candidate Prediction: {prediction[0]}")
    #print(f"Candidate Prediction: {prediction[0]}")




'''import joblib
import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Paths to saved model and required columns
MODEL_PATH = "artifacts/models/final_random_forest_model.pkl"
REQUIRED_COLUMNS_PATH = "artifacts/required_columns.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained RandomForest model.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading final RandomForest model...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return model

def load_required_columns(path: str = REQUIRED_COLUMNS_PATH):
    """
    Load the list of columns required for inference.
    """
    if not os.path.exists(path):
        logger.error(f"Required columns file not found at {path}")
        raise FileNotFoundError(f"Required columns not found: {path}")

    required_columns = joblib.load(path)
    logger.info(f"Loaded required columns for inference: {required_columns}")
    return required_columns

def predict(model, input_df: pd.DataFrame, required_columns: list):
    """
    Predict placement using the trained RandomForest model.
    
    input_df: pandas DataFrame with user features.
    required_columns: list of columns expected by the model.

    Returns: list of "Placed" or "Not Placed" for each row.
    """
    if input_df.empty:
        logger.warning("Input dataframe is empty!")
        return []

    # Add missing columns with default value 0
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only required columns and reorder
    input_df = input_df[required_columns]

    logger.info("Running placement prediction...")
    preds = model.predict(input_df)

    # Convert numeric prediction to text
    result = ["Placed" if p == 1 else "Not Placed" for p in preds]
    logger.info(f"Prediction completed: {result}")
    return result'''



'''  
import joblib
import pandas as pd
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

# Path to saved best model
MODEL_PATH = "artifacts/models/final_random_forest_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Loads the trained ML pipeline (best model).
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info("Loading trained model")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    return model

def predict(model, input_df: pd.DataFrame):
    """
    Runs prediction on input dataframe.
    Returns both predicted labels and predicted probabilities.
    """
    if input_df.empty:
        logger.warning("Input dataframe is empty!")
        return None, None
    logger.info("Running inference...")
    predictions = model.predict(input_df)
    try:
        probabilities = model.predict_proba(input_df)
    except AttributeError:
        logger.warning("Model does not support predict_proba. Returning None for probabilities.")
        probabilities = None

    logger.info("Inference completed")
    return predictions, probabilities'''