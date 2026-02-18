import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import pandas as pd

df = pd.read_csv("data/processed/processed_dataset.csv/train_raw.csv")
print(df.columns.tolist())

from src.utils.logger import get_logger

logger = get_logger(__name__)
# ---------------- PATHS ----------------
TRAIN_PATH = "data/processed/processed_dataset.csv/train_raw.csv"  # Your saved train data
MODEL_PATH = "artifacts/models/final_rf_top_features.pkl"
REQUIRED_COLUMNS_PATH = "artifacts/required_columns_top_features.pkl"

# Ensure folders exist
os.makedirs("artifacts/models", exist_ok=True)

# ---------------- TOP 6 FEATURES ----------------
top_features = [
    "technical_score",
    "years_of_experience",
    "aptitude_score",
    "skills_match_percentage",
    "communication_score"
]

TARGET = "status"  # Target column in train_raw.csv

# ---------------- LOAD TRAIN DATA ----------------
logger.info(f"Loading train data from: {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)

# Keep only top features + target
df_train = df_train[top_features + [TARGET]]

X_train = df_train[top_features]
y_train = df_train[TARGET]

logger.info(f"Training RandomForest on top {len(top_features)} features: {top_features}")



# ---------------- TRAIN RANDOM FOREST ----------------
logger.info("Training RandomForestClassifier on top features")
rf_top = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=2,
    random_state=42
)
rf_top.fit(X_train, y_train)

logger.info("RandomForest training completed.")

# ---------------- VALIDATE ----------------
y_pred = rf_top.predict(X_train)
acc = accuracy_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
logger.info(f"Top-Feature RandomForest Accuracy: {acc:.4f}")
logger.info(f"Top-Feature RandomForest F1 Score: {f1:.4f}")
# ---------------- SAVE MODEL & REQUIRED COLUMNS ----------------
joblib.dump(rf_top, MODEL_PATH)
joblib.dump(top_features, REQUIRED_COLUMNS_PATH)

logger.info(f"Model saved at: {MODEL_PATH}")
logger.info(f"Required columns saved at: {REQUIRED_COLUMNS_PATH}")
logger.info("train_top_features.py completed successfully.")