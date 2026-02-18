# src/data/feature_scaling.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def scale_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Scale numerical features using MinMaxScaler
    """
    logger.info(f"Scaling features: {features}...")
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    logger.info("Feature scaling completed")
    return df_scaled