import os
import json
import gc
import numpy as np
import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from script.load_datasets import compose_dataset
from script.deep_models import dl_parameter_search

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Errore impostazione GPU: {e}")


class ClearMemoryCallback(tf.keras.callbacks.Callback):
    """Pulisce la RAM della GPU alla fine di ogni addestramento di KerasTuner"""

    def on_train_end(self, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()


def grid_search(model_type, X_train, y_train, X_val=None, y_val=None):
    """
    Esegue l'Hyperparameter Tuning in base al tipo di modello richiesto.
    Ritorna un dizionario con i migliori parametri trovati.
    """
    print(f"\n{'='*40}")
    print(f" AVVIO TUNING PER: {model_type.upper()}")
    print(f"{'='*40}")

    best_params = {}

    if model_type == "cnn":
        if X_val is None or y_val is None:
            raise ValueError(
                f"Per il modello {model_type} devi fornire anche X_val e y_val.")

        tuner = kt.GridSearch(
            hypermodel=lambda hp: dl_parameter_search(hp, model_type),
            objective="val_accuracy",
            max_trials=27,
            directory="tuner_results",
            project_name=f"{model_type}_grid_search",
            overwrite=True
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[ClearMemoryCallback()]
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_params = best_hp.values

        print(f"\nMIGLIORI PARAMETRI TROVATI PER {model_type.upper()}:")
        for key, value in best_params.items():
            print(f"- {key}: {value}")

    elif model_type == "transfer_learning":
        if X_val is None or y_val is None:
            raise ValueError(
                f"Per il modello {model_type} devi fornire anche X_val e y_val.")

        tuner_tl = kt.GridSearch(
            hypermodel=lambda hp: dl_parameter_search(hp, model_type),
            objective="val_accuracy",
            max_trials=27,
            directory="tuner_results",
            project_name=f"{model_type}_grid_search",
            overwrite=True
        )

        tuner_tl.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            callbacks=[ClearMemoryCallback()]
        )

        best_hp = tuner_tl.get_best_hyperparameters(num_trials=1)[0]
        best_params = best_hp.values

        print(f"\nMIGLIORI PARAMETRI TROVATI PER {model_type.upper()}:")
        for key, value in best_params.items():
            print(f"- {key}: {value}")

    else:
        if X_val is not None and y_val is not None:
            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))
        else:
            X_combined, y_combined = X_train, y_train

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        if model_type == "logistic_reg":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    solver='saga', max_iter=300, tol=1e-2, random_state=42))
            ])
            param_grid = {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__l1_ratio': [0.0, 1.0]
            }

        elif model_type == "random_forest":
            pipeline = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(
                f"Modello '{model_type}' non supportato per la Grid Search!")

        grid_search_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2
        )

        grid_search_cv.fit(X_combined, y_combined)

        best_params = {k.replace('clf__', ''): v for k,
                       v in grid_search_cv.best_params_.items()}

        print(f"\nMIGLIORI PARAMETRI TROVATI PER {model_type.upper()}:")
        print(f"Miglior F1-Score in CV: {grid_search_cv.best_score_:.4f}")
        for key, value in best_params.items():
            print(f"- {key}: {value}")

    output_folder = "tuner_results"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"{model_type}_best_params.json")

    with open(file_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nParametri salvati con successo in: {file_path}")
    return best_params


if __name__ == "__main__":
    # Opzioni: "logistic_reg", "random_forest", "cnn", "transfer_learning"
    MODEL_TO_TUNE = "transfer_learning"

    is_deep_learning = MODEL_TO_TUNE in ["cnn", "transfer_learning"]

    print(f"Caricamento dati per il tuning di: {MODEL_TO_TUNE}...")
    X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset(
        CNN=is_deep_learning)

    del X_test, y_test
    gc.collect()

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    if X_val is not None:
        X_val = np.ascontiguousarray(X_val, dtype=np.float32)

    gc.collect()

    grid_search(
        model_type=MODEL_TO_TUNE,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )