import pandas as pd
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_plot(fig, save_dir,filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

def eda_analysis(df: pd.DataFrame,save_dir: str = "eda_plots") -> pd.DataFrame:
    logger.info("Starting EDA...")

    #1)Interview score vs job acceptance
    df['interview_score']=df['aptitude_score']+df['technical_score']+df['communication_score']

    fig, ax = plt.subplots()
    sns.boxplot(x='status',y='interview_score',data=df,ax=ax)
    ax.set_title('Interview score vs job acceptance')
    save_plot(fig,save_dir,'Interview score vs job acceptance.png')

    #2)Skills match percentage impact on placement
    fig,ax=plt.subplots()
    sns.boxplot(x='status',y='skills_match_percentage',data=df,ax=ax)
    ax.set_title("Skills Match Percentage vs Job Placement")
    save_plot(fig,save_dir,'Skills Match Percentage vs Job Placement.png')


    #3)Company tier vs acceptance rate
    # If you have these one-hot columns
    pd.crosstab(df["company_tier"], df["status"])
    company_acceptance = pd.crosstab(
    df["company_tier"],
    df["status"],
    normalize="index"  # row-wise percentage
    ) * 100
    
    fig,ax=plt.subplots()
    company_acceptance.plot(kind='bar', stacked=True,figsize=(6,4),ax=ax)
    ax.set_title("Company Tier vs Acceptance Rate")
    ax.set_ylabel("Acceptance Rate")
    ax.set_xlabel("Company Tier")
    save_plot(fig,save_dir,'Company Tier vs Acceptance Rate.png')

    #4)Experience vs placement probability
    df["status_num"] = df["status"].map({"Not Placed":0, "Placed":1})
    experience_placement = df.groupby("years_of_experience")["status_num"].mean()
    fig,ax=plt.subplots()
    experience_placement.plot(x=experience_placement.index, y=experience_placement.values, marker='o',ax=ax)
    ax.set_title("Experience vs Placement Probability")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Placement Probability")
    save_plot(fig,save_dir,'Experience vs Placement Probability.png')

    #5)Competition level vs job acceptance
    fig,ax=plt.subplots()
    ax = sns.countplot(x="competition_level", hue="status", data=df,ax=ax)
    ax.set_title("Competition level vs job acceptance")
    ax.set_ylabel("Number of Candidates")
    ax.set_xlabel("Competition Level")
    save_plot(fig,save_dir,'Competition level vs job acceptance.png')

    #6)Correlation analysis among numeric features
    df["status_num"] = df["status"].map({"Not Placed":0, "Placed":1})
    numeric_cols = [
    "age_years", "ssc_percentage", "hsc_percentage", "degree_percentage",
    "technical_score", "aptitude_score", "communication_score",
    "skills_match_percentage", "certifications_count",
    "years_of_experience", "previous_ctc_lpa", "expected_ctc_lpa",
    "notice_period_days", "employment_gap_months", "interview_score"
    ]
    corr = df[numeric_cols + ["status_num"]].corr() 
    fig,ax=plt.subplots(figsize=(30,25))# bigger figure
    sns.heatmap(corr, annot=True, fmt='.2f',cmap='coolwarm', cbar=True, center=0,linewidths=0.5,ax=ax)  # turn off numbers to reduce clutter
    ax.tick_params(axis='x', rotation=45)
    ax.set_title("Correlation Matrix of Numeric Features")
    save_plot(fig,save_dir,'Correlation Matrix of Numeric Features.png')

    logger.info("EDA analysis completed successfully")
    return df

