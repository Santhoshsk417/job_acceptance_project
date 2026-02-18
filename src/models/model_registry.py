from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_candidate_models(random_state: int=42):
    """
    Returns dictionary of candidate models for comparison.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "decision tree":DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200,random_state=random_state),
        "knn":KNeighborsClassifier()
        }
