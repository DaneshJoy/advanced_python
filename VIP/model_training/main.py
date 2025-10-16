import time
import logging
from functools import wraps
from datetime import datetime
from typing import Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


now = datetime.now().strftime('%Y_%m_%d')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f'eeg_detection_{now}.log'),
        logging.StreamHandler()
    ]    
)
logger = logging.getLogger(__name__)

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")

    return wrapper


@log_execution
def load_data(file_path: str) -> pd.DataFrame:
    arff_data = arff.loadarff(file_path)
    df = pd.DataFrame(arff_data[0])
    df["eyeDetection"] = df["eyeDetection"].apply(lambda x: int(x.decode("utf-8")))
    return df


def visualize_data(df: pd.DataFrame, range: List = [3000, 6000]):
    df.iloc[range, :].plot()
    plt.show()


@log_execution
def prepare_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop("eyeDetection", axis=1)
    y = df["eyeDetection"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


@log_execution
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    model = RandomForestClassifier(
        n_estimators=100, criterion="entropy", n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    return model


@log_execution
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, target_names=["Open", "Closed"]
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Classification Report:\n{class_report}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=model.classes_, cmap=plt.cm.Blues
    )
    plt.show()

    return {
        "predictions": y_pred,
        "accuracy": accuracy,
        "classification_report": class_report,
        "Confusion Matrix": conf_matrix,
    }


@log_execution
def save_model(model, filepath: str):
    joblib.dump(model, filepath)


@log_execution
def load_model(filepath: str):
    model = joblib.load(filepath)
    return model

@log_execution
def visualize_predictions(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, 
                          n_samples: int = 200, start_idx: int = 0, 
                          save_path: Optional[str] = None) -> None:
    """
    Visualize EEG signal with eye state predictions vs actual values.
    
    Args:
        X_test: Test features (EEG signals)
        y_test: True labels
        y_pred: Predicted labels
        n_samples: Number of samples to plot
        start_idx: Starting index for the segment
        save_path: Path to save the plot (optional)
    """
    
    samples = min(n_samples, len(y_test) - start_idx)
    end_idx = start_idx + samples
    
    # Get a segment of data
    X_segment = X_test[start_idx:end_idx]
    y_true_segment = y_test[start_idx:end_idx]
    y_pred_segment = y_pred[start_idx:end_idx]
    
    # Use first EEG channel for visualization (or mean of all channels)
    if isinstance(X_segment, pd.DataFrame):
        signal = X_segment.iloc[:, 0].values
    else:
        signal = X_segment[:, 0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Plot 1: EEG signal with true labels
    ax1 = axes[0]
    ax1.plot(signal, color='blue')
    
    # Mark eye state regions (true labels)
    for i in range(samples):
        if y_true_segment[i] == 1:  # Closed
            ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='red', label='Closed')
        else:  # Open
            ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='green', label='Open')
    
    ax1.set_ylabel('EEG Signal (Channel 1)', fontsize=10)
    ax1.set_title('EEG Signal with True Eye States', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Plot 2: EEG signal with predicted labels
    ax2 = axes[1]
    ax2.plot(signal, color='blue', linewidth=1, alpha=0.7)
    
    # Mark eye state regions (predicted labels)
    for i in range(samples):
        if y_pred_segment[i] == 1:  # Closed
            ax2.axvspan(i-0.5, i+0.5, alpha=0.3, color='red', label='Closed' if y_pred_segment[i-1] == 0 else '')
        else:  # Open
            ax2.axvspan(i-0.5, i+0.5, alpha=0.3, color='green', label='Open' if y_pred_segment[i-1] == 1 else '')
    
    ax2.set_ylabel('EEG Signal (Channel 1)', fontsize=10)
    ax2.set_title('EEG Signal with Predicted Eye States', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Mark misclassifications
    mismatches = y_true_segment != y_pred_segment
    
    # Add text with accuracy for this segment
    segment_accuracy = np.mean(y_true_segment == y_pred_segment)
    fig.text(0.5, 0.02, f'Segment Accuracy: {segment_accuracy:.2%} | Samples: {start_idx} to {end_idx}', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Log segment statistics
    logger.info(f"Segment statistics:")
    logger.info(f"  - True Open: {np.sum(y_true_segment == 0)}, Closed: {np.sum(y_true_segment == 1)}")
    logger.info(f"  - Predicted Open: {np.sum(y_pred_segment == 0)}, Closed: {np.sum(y_pred_segment == 1)}")
    logger.info(f"  - Misclassifications: {np.sum(mismatches)}")
    logger.info(f"  - Segment Accuracy: {segment_accuracy:.2%}")


def main():
    file_path = "data/EEG Eye State.arff"
    data = load_data(file_path)
    visualize_data(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    save_model(model, "model.pkl")

    # model_loaded = load_model('model.pkl')
    # evaluate_model(model_loaded, X_test, y_test)
    
    # Visualize predictions
    visualize_predictions(
        X_test,
        y_test.values, 
        results['predictions'],
        n_samples=50,
        start_idx=1000,
        save_path='predictions_plot.png'
    )


if __name__ == "__main__":
    main()
