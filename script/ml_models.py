from tqdm import tqdm
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_ml_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_ml_model(path):
    if Path(path).exists():
        return joblib.load(path)
    return None


def train_model(model_type, X, y, params=None):    
    if params is None:
        params = {}

    if model_type == "logistic_reg":
        # Estrazione parametri PCA
        pca_params = {
            k.replace("pca__", ""): v
            for k, v in params.items()
            if k.startswith("pca__")
        }
        # Estrazione parametri Logistic Regression
        lr_params = {
            k: v
            for k, v in params.items()
            if not k.startswith("pca__")
        }

        # Inizializzazione Logistic Regression
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **lr_params
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(**pca_params)),
            ("clf", clf)
        ])

        # Addestramento
        pipeline.fit(X, y)
        return pipeline
    
    elif model_type == "random_forest":
        # Estrazione parametri PCA
        pca_params = {
            k.replace("pca__", ""): v
            for k, v in params.items()
            if k.startswith("pca__")
        }

        # Estrazione parametri Random Forest
        rf_params = {
            k: v
            for k, v in params.items()
            if not k.startswith("pca__")
        }

        # Inizializzazione della Random Forest
        clf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **rf_params
        )

        pipeline = Pipeline([
            ("pca", PCA(random_state=42, svd_solver='randomized', **pca_params)),
            ("clf", clf)
        ])

        # Addestramento
        pipeline.fit(X, y)
        return pipeline

    else:
        raise ValueError(f"Modello non riconosciuto: {model_type}")

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


def cross_validate_model(model_type, X, y, n_splits=5, params=None):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
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
            y[train_idx],
            params=params
        )

        metrics = evaluate_model(
            model,
            X[val_idx],
            y[val_idx]
        )
        cv_metrics["accuracy"].append(metrics["accuracy"])
        cv_metrics["precision"].append(metrics["precision"])
        cv_metrics["recall"].append(metrics["recall"])
        cv_metrics["f1"].append(metrics["f1"])


    final_model = train_model(model_type, X, y, params=params)

    return cv_metrics, final_model
