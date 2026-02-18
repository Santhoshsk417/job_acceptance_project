from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
import os
import pandas as pd

logger = get_logger(__name__)

#  Drop target
'''drop_col=['status','interview_score', 'ssc_score', 'hsc_score', 'degree_score',
 'academic_performance_score', 'placement_prob_score',
 'company_tier_original_Tier 2', 'company_tier_original_Tier 3',
 'competition_level_competition_level_Low',
 'competition_level_competition_level_Medium',
 'experience_category_Junior', 'experience_category_Senior',
 'ssc_band_Average', 'ssc_band_Good', 'ssc_band_Excellent',
 'hsc_band_Average', 'hsc_band_Good', 'hsc_band_Excellent',
 'degree_band_Average', 'degree_band_Good', 'degree_band_Excellent',
 'academic_performance_band_Average', 'academic_performance_band_Good',
 'academic_performance_band_Excellent', 'skill_match_level_Medium',
 'skill_match_level_High', 'interview_performance_Average',
 'interview_performance_Good', 'interview_performance_Excellent',
 'placement_prob_band_Medium', 'placement_prob_band_High']'''


def split_data(df: pd.DataFrame, target: str='status_num', test_size=0.2, random_state=42,output_path: str = "data/processed",stratify=None):
    logger.info("Splitting data into train and test sets")
    # Drop columns for training
    X = df.drop(columns=['status', 'status_num'], errors='ignore')
    y = df['status_num']
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=stratify) # maintain same class proportion

    logger.info(f"Split completed: Train={X_train.shape}, Test={X_test.shape}")

    # Create processed directory
    os.makedirs(output_path, exist_ok=True)

    # Save splits
    train_df = X_train.copy()
    train_df[target] = y_train.values

    test_df = X_test.copy()
    test_df[target] = y_test.values

    train_path = os.path.join(output_path, "train_raw.csv")
    test_path = os.path.join(output_path, "test_raw.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train data saved at {train_path}")
    logger.info(f"Test data saved at {test_path}")

    return X_train, X_test, y_train, y_test

