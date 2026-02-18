import sys
import os
import pandas as pd
import streamlit as st
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.models.predict import predict_candidate
from src.analytics.analytics import generate_business_kpis

from src.features.feature_engineering import feature_engineering


# ---------------- PATHS ----------------
MODEL_PATH = "artifacts/models/final_rf_top_features.pkl"
REQUIRED_COLUMNS_PATH = "artifacts/required_columns_top_features.pkl"
DATA_PATH = "data/processed/processed_dataset.csv/train_raw.csv"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Candidate Placement Dashboard", layout="wide")
st.title("💼 HR Candidate Placement Dashboard")
st.markdown("---")

# ---------------- LOAD DATA FOR KPIS ----------------
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    from src.analytics.analytics import generate_business_kpis
    kpis = generate_business_kpis(df)
else:
    st.warning(f"Data not found at {DATA_PATH}. KPIs will not be available.")
    df = None
    kpis = None

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["Predict Candidate", "Business KPIs"])

# ---------------- TAB 1: Candidate Prediction ----------------
with tab1:
    st.header("🤖 Predict Candidate Placement")

    # Check if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(REQUIRED_COLUMNS_PATH):
        from src.models.predict import predict_candidate

        st.subheader("Enter Candidate Details")
        candidate_input = {
            "technical_score": st.number_input("Technical Score", 0, 100, 50),
            "years_of_experience": st.number_input("Years of Experience", 0, 50, 2),
            "aptitude_score": st.number_input("Aptitude Score", 0, 100, 50),
            "skills_match_percentage": st.number_input("Skills Match (%)", 0, 100, 50),
            "communication_score": st.number_input("Communication Score", 0, 100, 50)
        }

        if st.button("Predict Placement"):
            prediction = predict_candidate(candidate_input)
            st.subheader("Prediction Result")
            st.success(f"Candidate is: {prediction[0]}")

    else:
        st.warning("Model or required columns file not found! Please train the model first.")

# ---------------- TAB 2: Business KPIs ----------------
with tab2:
    st.header("📊 Business KPIs")

    if kpis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", kpis["Total Candidates"])
        col2.metric("Placement Rate (%)", kpis["Placement Rate (%)"])
        col3.metric("Offer Dropout Rate (%)", kpis["Offer Dropout Rate (%)"])

        col4, col5, col6,col7 = st.columns(4)
        col4.metric("Average Interview Score", kpis["Average Interview Score"])
        col5.metric("Average Skills Match (%)", kpis["Average Skills Match (%)"])
        col6.metric("High Risk Candidate (%)", kpis["High Risk Candidate (%)"])
        col7.metric("Job Acceptance Rate (%)", kpis["Job Acceptance Rate (%)"])

    else:
        st.warning("KPIs are not available because the data file is missing.")
