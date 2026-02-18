import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import get_logger
from src.data.feature_scaling import scale_features


logger = get_logger(__name__)

def save_plot(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

def feature_engineering(df: pd.DataFrame,save_dir: str = "fe_plots") -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    #1)Experience category (Fresher / Junior / Senior)
    df["status_num"] = df["status"].map({"Not Placed":0, "Placed":1})
    df['experience_category']=pd.cut(df['years_of_experience'],bins=[-1,0,3,df['years_of_experience'].max()],labels=['Fresher','Junior','Senior'])

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='experience_category', hue='status', data=df,ax=ax)
    ax.set_title("Experience Category vs Placement Status out total")
    ax.set_xlabel("Experience Category")
    ax.set_ylabel("Number of Candidates")
    save_plot(fig, save_dir, "Experience Category vs Placement Status out total.png")
    sns.barplot(x='experience_category', y='status_num', data=df, ax=ax)
    ax.set_title("Experience Category vs Placement Probability out of placed")
    save_plot(fig, save_dir, "Experience Category vs Placement Probability out of placed.png")

    # 2) Academic performance bands
    # Map status to numeric
    df['status_num'] = df['status'].map({"Not Placed":0, "Placed":1})
    # Step 1: Create individual bands
    df['ssc_band'] = pd.cut(df['ssc_percentage'], bins=[0,40,65,85,100], labels=['Poor','Average','Good','Excellent'])
    df['hsc_band'] = pd.cut(df['hsc_percentage'], bins=[0,40,65,85,100], labels=['Poor','Average','Good','Excellent'])
    df['degree_band'] = pd.cut(df['degree_percentage'], bins=[0,40,65,85,100], labels=['Poor','Average','Good','Excellent'])
    # Step 2: Map bands to numeric scores
    score_map = {'Poor':0, 'Average':1, 'Good':2, 'Excellent':3}
    df['ssc_score'] = df['ssc_band'].map(score_map).astype(float)
    df['hsc_score'] = df['hsc_band'].map(score_map).astype(float)
    df['degree_score'] = df['degree_band'].map(score_map).astype(float)
    # Step 3: Average score across SSC, HSC, Degree
    df['academic_score'] = df[['ssc_score','hsc_score','degree_score']].mean(axis=1)
    # Step 4: Create combined Academic Performance Band
    df['academic_performance_band'] = pd.cut(df['academic_score'],bins=[0,1.25,2,2.75,3.05],labels=['Poor','Average','Good','Excellent'])
    # Step 5: Plot placement probability
    fig, ax = plt.subplots()
    sns.barplot(x='academic_performance_band', y='status_num', data=df,
                order=['Poor','Average','Good','Excellent'],palette='viridis', ax=ax)
    ax.set_title("Academic Performance band vs Placement Probability")
    ax.set_xlabel("Academic Performance Band")
    ax.set_ylabel("Placement Probability")
    save_plot(fig, save_dir,"Academic Performance band vs Placement Probability.png")

    #3)Skills match level (Low / Medium / High)
    # Map status to numeric if needed
    df['status_num'] = df['status'].map({"Not Placed":0, "Placed":1})
    df['skill_match_level'] = pd.cut(df['skills_match_percentage'], bins=[0,40,70,100], labels=['Low','Medium','High'])

    fig, ax = plt.subplots()
    sns.barplot(x='skill_match_level', y='status_num', data=df, order=['Low','Medium','High'],palette='viridis', ax=ax)
    ax.set_title("Skills match level (Low / Medium / High)")
    ax.set_xlabel("Skills Match Level")
    ax.set_ylabel("Placement Probability")
    save_plot(fig, save_dir,"Skills match level.png")

    #4)Interview performance category
    df['interview_performance']=pd.cut(df['interview_score'],bins=[0,100,170,240,300],labels=['Poor', 'Average', 'Good', 'Excellent'])
    df['status_num'] = df['status'].map({"Not Placed":0, "Placed":1})
    fig, ax = plt.subplots()
    sns.barplot(x='interview_performance', y='status_num', data=df, order=['Poor', 'Average', 'Good', 'Excellent'],palette='viridis', ax=ax)
    ax.set_title("Interview Performance Category vs Placement")
    ax.set_xlabel("Interview Performance Category")
    ax.set_ylabel("Placement Probability")
    save_plot(fig, save_dir,"Interview Performance Category vs Placement.png")

    #5)Placement probability score
    features = ['communication_score','technical_score','skills_match_percentage','interview_score','years_of_experience']
    df_scaled = scale_features(df, features)
    weights = {'communication_score': 0.20,
           'technical_score': 0.20,
           'skills_match_percentage': 0.20,
           'interview_score': 0.3,
           'years_of_experience': 0.1}
    df['placement_prob_score'] = sum(df_scaled[col]*w for col,w in weights.items())

    df['placement_prob_band'] = pd.cut(df['placement_prob_score'], bins=[0, 0.30, 0.60, 1],labels=['Low', 'Medium', 'High'])
    fig, ax = plt.subplots()
    sns.barplot(x='placement_prob_band', y='placement_prob_score', data=df, order=['Low','Medium','High'],palette='viridis', ax=ax)
    ax.set_title("Placement Probability Score")
    ax.set_ylim(0,1)
    save_plot(fig, save_dir, "Placement Probability Score.png")

    logger.info("Feature engineering completed successfully")
    return df

