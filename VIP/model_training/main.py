import time
from functools import wraps
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(F"Completed {func.__name__} in {elapsed:.2f}s")
        return result
    return wrapper


@log_execution
def load_data(file_path: str) -> pd.DataFrame:
    arff_data = arff.loadarff(file_path)
    df = pd.DataFrame(arff_data[0])
    df['eyeDetection'] = df['eyeDetection'].apply(lambda x: int(x.decode('utf-8')))
    return df


def visualize_data(df: pd.DataFrame, range: List=[3000, 6000]):
    df.iloc[:, :].plot()
    plt.show()


@log_execution
def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop('eyeDetection', axis=1)
    y = df['eyeDetection']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


@log_execution
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    return model


@log_execution
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Open', 'Closed'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                            display_labels=model.classes_,
                                            cmap=plt.cm.Blues)
    plt.show()
    
    return {
        'predictions': y_pred,
        'accuracy': accuracy,
        'classification_report': class_report,
        'Confusion Matrix': conf_matrix
    }
    

@log_execution
def save_model(model, filepath: str):
    joblib.dump(model, filepath)


@log_execution
def load_model(filepath:str):
    model = joblib.load(filepath)
    return model


def main():
    file_path = 'data/EEG Eye State.arff'
    data = load_data(file_path)
    visualize_data(data)
    # X_train, X_test, y_train, y_test = prepare_data(data)
    # model = train_model(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # save_model(model, 'model.pkl')
    
    # model_loaded = load_model('model.pkl')
    # evaluate_model(model_loaded, X_test, y_test)
    

if __name__ == '__main__':
    main()
