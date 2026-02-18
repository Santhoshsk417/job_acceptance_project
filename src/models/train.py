from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
from src.models.model_registry import get_candidate_models
from src.models.evaluate import evaluate_and_save
from src.visuals.plots import plot_confusion_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_models(X_train, y_train, X_test, y_test,):
    """
    Train candidate models, do hyperparameter tuning for top ones,
    check overfitting, feature importance, and save final model.
    """
     #1)Initial model comparison
     
    # 1️⃣ Initial model comparison
    models = get_candidate_models()
    best_model_name = None
    best_score = -1
    best_model = None
    best_pred = None

    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, f1, _ = evaluate_and_save(y_test, y_pred, name)

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model = model
            best_pred = y_pred

    logger.info(f"Best model: {best_model_name} with F1: {best_score:.4f}")

    #2️) Hyperparameter tuning for RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10,15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4], 
        'max_features': ['sqrt', 'log2']
    }
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        rf_model,
        rf_param_grid,
        cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    logger.info(f"Best RF params: {rf_grid.best_params_}, CV accuracy: {rf_grid.best_score_:.4f}")

    #3)Train final RF with best params
    best_rf = RandomForestClassifier(
        **rf_grid.best_params_,
        class_weight='balanced',
        random_state=42)
    best_rf.fit(X_train, y_train)
    final_pred = best_rf.predict(X_test)

    #4) Evaluate final model
    evaluate_and_save(y_test, final_pred, "RandomForest_Final")
    plot_confusion_matrix(y_test, final_pred, labels=sorted(y_test.unique()),
                          save_path="artifacts/reports/confusion_matrix.png")
    
    #5) Check overfitting
    train_acc = best_rf.score(X_train, y_train)
    test_acc = best_rf.score(X_test, y_test)
    logger.info(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    #6)Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_rf.feature_importances_
    }).sort_values(by="importance", ascending=False)
    logger.info(f"Top features:\n{feature_importance.head(10)}")

    #7)Save final model
    joblib.dump(best_rf, "artifacts/models/final_random_forest_model.pkl")
    logger.info("Final RandomForest model saved at artifacts/models/final_random_forest_model.pkl")

    #8) Save REQUIRED_COLUMNS dynamically for inference
    import os
    required_columns = X_train.columns.tolist()
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(required_columns, "artifacts/required_columns.pkl")
    logger.info("Saved REQUIRED_COLUMNS for inference at artifacts/required_columns.pkl")
    
    return best_rf, final_pred, feature_importance

