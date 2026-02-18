import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("starting data cleaning process....")
    #filling missing values
    mean_col = ['ssc_percentage', 'hsc_percentage']
    for col in mean_col:
        df[col]=df[col].fillna(df[col].mean()) 
    
    median_col=['notice_period_days','employment_gap_months']
    for col in median_col:
        df[col]=df[col].fillna(df[col].median())

    fill_unknown=['career_switch_willingness','relevant_experience','job_role_match','layoff_history','relocation_willingness']
    for col in fill_unknown:
        df[col]=df[col].fillna('Unknown')

    # Drop full-row duplicates, keep the first occurrence
    df.drop_duplicates(inplace=True)
   
    # Standardize categorical columns
    df['gender'] = df['gender'].str.strip().str.title()
    df['internship_experience'] = df['internship_experience'].str.strip().str.title()
    # Remove extra spaces
    df['company_tier'] = df['company_tier'].str.strip().str.title()
    
    logger.info("Data cleaning completed")
    return df

