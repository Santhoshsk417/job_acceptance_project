import pandas  as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv(path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise