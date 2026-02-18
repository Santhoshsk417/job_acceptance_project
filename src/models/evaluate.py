import json
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_and_save(y_test, y_pred, model_name: str):
    """
    Evaluate classification model and save report.
    """
    logger.info(f"Evaluating model: {model_name}")
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{model_name} F1 Score: {f1:.4f}")
    logger.info(f"{model_name} report: {report}")

    os.makedirs("artifacts/reports", exist_ok=True)
    report_path = "artifacts/reports/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return accuracy, f1, report