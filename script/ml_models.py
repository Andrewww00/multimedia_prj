from tqdm import tqdm
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def save_ml_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    
def load_ml_model(path):
    if Path(path).exists():
        return joblib.load(path)
    return None

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
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "y_true": y_test,
        "y_pred": predictions
    }
    return metrics

def cross_validate_model(model_type, X, y, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(
                                                    skf.split(X, y),
                                                    total=n_splits,
                                                    desc=f"Cross Validation {model_type}"
                                                    ),
                                                    start=1
    ):
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

        final_model = get_model(model_type)
        final_model.fit(X, y)

    return scores, final_model