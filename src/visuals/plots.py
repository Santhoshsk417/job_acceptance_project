import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

def save_plot(fig, save_dir,filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

def plot_confusion_matrix(y_test, y_pred, labels, save_path="artifacts/reports/confusion_matrix.png"):
    """
    Plots and saves confusion matrix.
    """
    logger.info("Generating confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    save_plot(fig, save_path ,'Confusion Matrix.png')

    logger.info(f"Confusion matrix saved at {save_path}")