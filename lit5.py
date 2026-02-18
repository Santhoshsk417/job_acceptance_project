import streamlit as st
import pandas as pd
import joblib
import os
from src.models.predict import predict_candidate  # your predict.py

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Candidate Placement", layout="wide")
st.title("💼 HR Candidate Placement Predictor")
st.markdown("---")

# ---------------- LOAD MODEL CHECK ----------------
MODEL_PATH = "artifacts/models/final_rf_top_features.pkl"
REQUIRED_COLUMNS_PATH = "artifacts/required_columns_top_features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(REQUIRED_COLUMNS_PATH):
    st.error("Model or required columns file not found! Please train the model first.")
    st.stop()

st.success("Top-Feature RandomForest Model Loaded Successfully ✅")

# ---------------- USER INPUT ----------------
st.subheader("Enter Candidate Details")

candidate_input = {
    "technical_score": st.number_input("Technical Score", 0, 100, 50),
    "years_of_experience": st.number_input("Years of Experience", 0, 50, 2),
    "aptitude_score": st.number_input("aptitude_score", 0, 300, 150),
    "skills_match_percentage": st.number_input("Skills Match (%)", 0, 100, 50),
    "communication_score": st.number_input("Communication Score", 0, 100, 50)
}

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict Placement"):
    # Call predict function
    prediction = predict_candidate(candidate_input)
    
    # Show result
    st.subheader("Prediction Result")
    st.success(f"Candidate is: {prediction[0]}")
