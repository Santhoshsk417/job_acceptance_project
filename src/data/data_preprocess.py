# src/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)
def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data preprocessing (encoding)....")
    # categorical into numerical values
    # Binary encoding
    binary_cols = ['gender','internship_experience','bond_requirement','status']
    for col in binary_cols:
        df[col] = pd.factorize(df[col])[0]

    # # List of columns to one-hot encoding
    one_hot_cols = ['degree_specialization','career_switch_willingness','relevant_experience',
                    'company_tier','job_role_match','competition_level','layoff_history','relocation_willingness']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False, dtype=int)

    ordinal_cols = ['experience_category', 'academic_performance_band',
                'skill_match_level', 'interview_performance', 'placement_prob_band','ssc_band','hsc_band','degree_band']

    le = LabelEncoder()
    for col in ordinal_cols:
        df[col] = le.fit_transform(df[col])

    logger.info("Data preprocessing (encoding) completed")
    return df
