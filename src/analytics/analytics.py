import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_business_kpis(df: pd.DataFrame):

    logger.info("Generating business KPIs...")

    total_candidates = len(df)

    # Placement Rate
    placement_rate = df["status"].mean() * 100 if "placement_prob_score" in df.columns else 0

     # Job Acceptance Rate
    job_acceptance_rate = df["placement_prob_score"].mean() * 100 if "placement_prob_score" in df.columns else 0

    # Averages
    avg_interview_score = df["interview_score"].mean() if "interview_score" in df.columns else 0
    avg_skills_match = df["skills_match_percentage"].mean() if "skills_match_percentage" in df.columns else 0

    # High-Risk Candidate Definition
    high_risk = df[
        (df["placement_prob_score"] < 0.30) &
        (df["interview_score"] < df["interview_score"].median())
    ]
    high_risk_percentage = (len(high_risk) / total_candidates) * 100

    # Smart Dropout Logic
    dropouts = df[
        (df["placement_prob_score"] > 0.60) &
        (df["status"] == 0)
    ]
    dropout_rate = (len(dropouts) / total_candidates) * 100

    kpis = {
        "Total Candidates": total_candidates,
        "Placement Rate (%)": round(placement_rate, 2),
        "Job Acceptance Rate (%)": round(job_acceptance_rate, 2),
        "Average Interview Score": round(avg_interview_score, 2),
        "Average Skills Match (%)": round(avg_skills_match, 2),
        "High Risk Candidate (%)": round(high_risk_percentage, 2),
        "Offer Dropout Rate (%)": round(dropout_rate, 2),
    }

    os.makedirs("artifacts/reports", exist_ok=True)
    pd.DataFrame([kpis]).to_csv("artifacts/reports/business_kpis.csv", index=False)

    logger.info("Business KPIs generated successfully")

    return kpis
