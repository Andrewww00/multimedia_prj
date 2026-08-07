import json
from script.preprocessing import prepare_dataset
from script.ml_models import cross_validate_model, train_model, save_ml_model, load_ml_model
from script.config import *
from script.load_datasets import compose_dataset, load_class_names
from script.deep_models import cnn_model, transfer_learning_model
import script.metrics_plots as mt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import BatchNormalization

ML = True
CROSS_VALIDATION = True
CNN = False
TRANSFER_LEARNING = False

# Modelli senza cross validation
model_paths = {
    "logistic_reg": LOGISTIC_MODEL_PATH,
    "random_forest": RANDOM_FOREST_MODEL_PATH
}

# Modelli con cross validation
cv_model_paths = {
    "logistic_reg": LOGISTIC_CV_MODEL_PATH,
    "random_forest": RANDOM_FOREST_CV_MODEL_PATH
}

classes_names = load_class_names()

if __name__ == "__main__":
    prepare_dataset()
    if ML:
        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset()

        # TRAIN
        models = ["logistic_reg", "random_forest"]
        results = {}

        for model in models:
            # 1. CARICAMENTO DEI MIGLIORI PARAMETRI
            json_path = Path("tuner_results") / f"{model}_best_params.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    best_params = json.load(f)
                print(
                    f"Parametri ottimali caricati per {model}: {best_params}")
            else:
                print(f"Parametri per {model} non trovati. Uso i default.")
                best_params = {}

            if CROSS_VALIDATION:
                model_path = cv_model_paths[model]
                metrics_path = MODELS_DIR / f"{model}_cv_metrics.json"
                trained_model = load_ml_model(model_path)

                if trained_model is None or not metrics_path.exists():
                    print(f"Alleno {model} con cross validation")
                    cv_metrics, trained_model = cross_validate_model(
                        model_type=model,
                        X=X_train,
                        y=y_train,
                        n_splits=5,
                        params=best_params
                    )
                    save_ml_model(trained_model, model_path)
                    print(f"{model} salvato.")

                    print(f"\n--- Risultati Cross Validation: {model.upper()} ---")
                    model_results = {"means": [], "ci": []}

                    for metric in ["accuracy", "precision", "recall", "f1"]:
                        scores_percent = [s * 100 for s in cv_metrics[metric]]
                        mean_val, err_val = mt.calc_conf_interval(scores_percent)

                        print(f"{metric.capitalize():<10}: {mean_val:.2f}% ± {err_val:.2f}%")

                        model_results["means"].append(mean_val)
                        model_results["ci"].append(err_val)

                    # Salvataggio  metriche
                    with open(metrics_path, "w") as f:
                        json.dump(model_results, f)
                        
                else:
                    print(f"{model} già allenato. Caricamento completato.")
                    with open(metrics_path, "r") as f:
                        model_results = json.load(f)
                        
                results[model] = model_results
                mt.evaluate_and_plot_model(model, trained_model, X_test, y_test, model_results)

            # No cross validation
            else:
                model_path = model_paths[model]
                trained_model = load_ml_model(model_path)
                if trained_model is None:
                    print(f"Alleno {model} senza cross validation")
                    trained_model = train_model(
                        model_type=model,
                        X=X_train,
                        y=y_train,
                        params=best_params
                    )
                    save_ml_model(trained_model, model_path)
                    print(f"{model} salvato.")
                else:
                    print(f"{model} già allenato. Caricamento completato.")

                results[model] = mt.evaluate_and_plot_model(model, trained_model, X_test, y_test, model_results)

        mt.plot_models_comparison(results)

    elif CNN:
        print("Alleno la CNN")
        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset(
            CNN=True)

        if CNN_MODEL_PATH.exists():
            print("CNN già allenata. Caricamento modello...")
            model = load_model(CNN_MODEL_PATH)
            history_path = MODELS_DIR / "cnn_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                mt.plot_training_history(history, title="CNN training history")
            else:
                print("History CNN non trovata. Salto i grafici di training.")

        else:
            print("CNN non trovata. Avvio training...")

            # Carico i best params che ho salvato in .json
            json_path = Path("tuner_results") / "cnn_best_params.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    best_params = json.load(f)
                print(
                    f"Iperparametri ottimali caricati da {json_path}: {best_params}")
            else:
                print(
                    f"Attenzione: {json_path} non trovato. Utilizzo i parametri di default.")
                best_params = {}

            # Uso i best params trovati con grid search
            model = cnn_model(best_params)

            # 1. DATA AUGMENTATION
            data_gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode="reflect"
            )
            data_gen.fit(X_train)

            # 2. DEFINIZIONE DEI CALLBACKS
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,    # Aspetta 10 epoche
                restore_best_weights=True   # Torno ai pesi prima dell'overfitting
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,      # diminuisco il learning rate se per 8 epoche non ho miglioramenti
                patience=5,
                min_lr=1e-5
            )

            checkpoint = ModelCheckpoint(
                filepath=str(CNN_MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            )

            # 3. TRAINING
            print("Inizio training")
            history = model.fit(
                data_gen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=100,  # early stopping e reduce learning rate si occupano dell'overfitting
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=1
            )

            # 4. Caricamento modello
            model = load_model(CNN_MODEL_PATH)

            history_dict = {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            }

            with open(CNN_HISTORY_PATH, "w") as f:
                json.dump(history_dict, f)

            mt.plot_training_history(history, title="CNN training History")
            print("CNN salvata.")

            # 5. VALUTAZIONE FINALE
            print("\nValuto sul Test Set")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            print(f"\nTest loss: {loss:.4f}")
            print(f"Test accuracy: {accuracy:.4f}")
            mt.predict_image(
                indices=[0, 100, 500, 1000, 2000,
                         3000, 4000, 5000, 6000, 7000],
                model=model,
                X_test=X_test,
                y_test=y_test,
                class_names=classes_names,
                save_path=PLOTS_DIR / "CNN_custom_predictions.png"
            )

    elif TRANSFER_LEARNING:
        print("Transfer Learning CNN")

        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset(
            CNN=True)

        json_path = Path("tuner_results") / \
            "transfer_learning_best_params.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                best_params = json.load(f)
            print(
                f"Iperparametri ottimali caricati da {json_path}: {best_params}")
        else:
            print(
                f"Attenzione: {json_path} non trovato. Utilizzo i parametri di default.")
            best_params = {}

        # Creo un dizionario che contiene i parametri trovati con grid search
        tl_params = {
            "learning_rate": best_params.get("learning_rate", 1e-3),
            "dense_units": best_params.get("dense_units", 256),
            "dropout_rate": best_params.get("dropout_rate", 0.3)
        }

        if TL_MODEL_PATH.exists():
            print("CNN con transfer learning già allenata. Caricamento modello...")
            model = load_model(TL_MODEL_PATH, safe_mode=False)

            if TL_HISTORY_PATH.exists():
                try:
                    with open(TL_HISTORY_PATH, "r") as f:
                        history = json.load(f)

                    mt.plot_training_history(
                        history,
                        "densenet121 training history"
                    )

                except json.JSONDecodeError:
                    print("History corrotta. Verrà ignorata.")
        else:
            print("CNN non trovata. Avvio Transfer Learning...")

            # Passo i parametri trovati tramite grid search
            model, base_model = transfer_learning_model(
                params=tl_params,
                input_shape=(32, 32, 3),
                num_classes=100
            )
            model.summary()

            # Data augmentation
            data_gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                zoom_range=0.15,
                shear_range=0.1,
                brightness_range=[0.8, 1.2],
                fill_mode="reflect"
            )

            early_stopping_1 = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True)
            early_stopping_2 = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True)
            checkpoint_1 = ModelCheckpoint(filepath=str(
                TL_MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
            checkpoint_2 = ModelCheckpoint(filepath=str(
                TL_MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
            reduce_lr_1 = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
            reduce_lr_2 = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

            # FASE 1: FEATURE EXTRACTION
            print("\nFASE 1 - Backbone congelato")

            model.compile(
                optimizer=Adam(
                    learning_rate=tl_params["learning_rate"]
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history_1 = model.fit(
                data_gen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping_1, reduce_lr_1, checkpoint_1],
                verbose=1
            )

            # FASE 2: FINE TUNING
            print("\nFASE 2 - Fine tuning")
            base_model.trainable = True

            # Congelo i primi layer
            for layer in base_model.layers[:-50]:
                layer.trainable = False

            for layer in base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False

            fine_tuning_lr = tl_params["learning_rate"] / 100

            model.compile(
                optimizer=Adam(
                    learning_rate=fine_tuning_lr
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history_2 = model.fit(
                data_gen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping_2, reduce_lr_2, checkpoint_2],
                verbose=1
            )

            # Unione history
            history = {}
            for key in history_1.history.keys():
                history[key] = (
                    history_1.history[key] +
                    history_2.history[key]
                )
            # Plot training
            mt.plot_training_history(
                history, title="densenet121 training history")

            # Salvataggio modello
            MODELS_DIR.mkdir(
                parents=True,
                exist_ok=True
            )
            model.save(TL_MODEL_PATH)
            # Salvataggio history
            history_dict = {
                key: [float(value) for value in values]
                for key, values in history.items()
            }

            with open(TL_HISTORY_PATH, "w") as f:
                json.dump(history_dict, f)
            print("CNN Transfer Learning salvata.")

        # Valutazione finale
        print("\nValutazione sul Test Set")
        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=1
        )
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy*100:.2f}%")

        mt.predict_image(
            indices=[0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=classes_names,
            save_path=PLOTS_DIR / "densenet121_predictions.png"
        )
