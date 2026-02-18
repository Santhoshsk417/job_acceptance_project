import pandas as pd
from src.data.data_load import load_csv
from src.data.data_cleaning import clean_data
from eda.eda import eda_analysis
from src.data.split import split_data
from src.data.data_preprocess import encode_data
from src.features.feature_engineering import feature_engineering
from src.data.feature_scaling import scale_features
from src.models.train import train_models
from src.analytics.analytics import generate_business_kpis
from src.utils.logger import get_logger
import yaml

logger = get_logger(__name__)

def main():
    logger.info("ML Pipeline started")

    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 1️⃣ Load data
    df = load_csv(config["data"]["raw_file"])

    # 2️⃣ Clean
    df = clean_data(df)

    # 5️⃣ Encoding
    #df = encode_data(df)

    # 3️⃣ EDA (on cleaned raw data)
    df = eda_analysis(df)

    # 4️⃣ Feature Engineering
    df = feature_engineering(df)

    # 5️⃣ Encoding
    df = encode_data(df)
    #categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    #df = pd.get_dummies(df, drop_first=True)

    # 5.5️⃣ Impute missing values (after get_dummies!)
    from sklearn.impute import SimpleImputer
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns  # includes new dummy columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])



    # 6️⃣ Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        target="status",
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=df['status_num'], 
        output_path=config["data"]["process_file"]
    )
    # 7️⃣ Feature Scaling (only numeric columns)
    #df = scale_features(df)

    #8️⃣ Train & select best model
    best_rf, final_pred, feature_importance = train_models(
        X_train, y_train,
        X_test, y_test,
    )

    # 🔟 Business KPIs
    kpis = generate_business_kpis(df)
    logger.info(f"Business KPIs: {kpis}")

    logger.info(
        f"Training completed. Best Model: {type(best_rf).__name__}")



if __name__ == "__main__":
    main()
