# Job Acceptance Prediction ML Pipeline
End-to-end machine learning pipeline to predict HR job acceptance using Python, Random Forest, and feature engineering. Includes data preprocessing, exploratory analysis, model training, evaluation, and inference.
# Project Overview
This project implements a complete workflow to predict whether a candidate will accept a job offer.  
The pipeline includes:

- Data cleaning and preprocessing  
- Feature engineering  
- Exploratory Data Analysis (EDA) with visualizations  
- Model training using **Random Forest classifier**  
- Model evaluation and inference
## Project Structure

hr-job-placement-ml/
│
├── config/
│   ├── config.yaml          # Paths, preprocessing params, model configs
│   └── logging.yaml         # Logging config
│
├── data/
│   ├── raw/
│   │   └── HR_Job_Placement_Dataset.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── eda/
│   └── eda.py         # EDA, graphs, insights (not production)
│
├── src/
│   ├── __init__.py
│
│   ├── data/
│   │   ├── data_load.py     # Load CSV/raw data
│   │   ├── data_cleaning.py # cleaning
│   │   ├── data_preprocess.py    # Feature engineering, missing values
│   │   ├── feature_scaling.py
│   │   └── split.py         # Train/test split
│
│   ├── analytics/
│   │   └── analytics.py   
│
│   ├── features/
│   │   └── feature_engineering.py   # Derived features (placement_prob_score, bands)
│
│   ├── models/
│   │   ├── train.py         # Train all models, compare
│   │   ├── evaluate.py      # Classification report, confusion matrix
│   │   ├── predict.py       # Load model, make predictions
│   │   └── model_registry.py# Store model references, params
│
│   ├── pipelines/
│   │   └── inference_pipeline.py   # Load model, preprocess & predict as test
│
│   ├── utils/
│   │   ├── logger.py        # Centralized logging
│
│   └── visualization/
│       └── plots.py         # EDA/analysis plots
│
├── train_top_feature/
│   ├── train_top_feature.py
│
├── logs/
│   └── training.log
├── eda_plots/
│   └──#6eda images.png
├── fe_plots/
│   └── #5 feature_engineering images.png
│
├── artifacts/
│   ├── required_columns_top_features.pkl
│   ├── required_columns.pkl
│   ├── models/
│   │   └── final_random_forest_model.pkl
│   │   └── final_rf_top_features.pkl   
│   └── reports/
│       ├── classification_report.json
│       └── confusion_matrix/confusion_matrix.png
│       └── business_kpis.csv
│
├── app/
│   └── streamlit_app.py
│
├── main.py                  # Orchestrates full pipeline
├── requirements.txt
└── README.md
## Dependencies

- Python 
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- uv for dependency management

## How to Run

Install dependencies:

uv sync
uv run python main.py
