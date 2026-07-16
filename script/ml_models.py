import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from script.preprocessing import prepare_dataset
from script.config import ML_TRAIN_FILE, ML_VAL_FILE, ML_TEST_FILE, MODELS_DIR, LABEL_ENCODER_PATH

def get_model(model_type):
    if model_type == "logistic_reg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500))
        ])

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError("Modello non riconosciuto")

def train_model(model_type, X, y):
    model = get_model(model_type)
    model.fit(X, y)

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1": f1_score(y_test, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True
        )
    }
    return metrics

def cross_validate_model(model_type, X, y, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = train_model(
            model_type,
            X[train_idx],
            y[train_idx]
        )

        metrics = evaluate_model(
            model,
            X[val_idx],
            y[val_idx]
        )
        scores.append(metrics["f1"])
        print(f"Fold {fold}: F1={metrics['f1']:.4f}")

    print(
        f"\nMean F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}"
    )

    return scores
