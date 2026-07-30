
import json
from script.preprocessing import prepare_dataset
from script.ml_models import cross_validate_model, train_model, evaluate_model, save_ml_model, load_ml_model
from script.config import *
from script.load_datasets import compose_dataset, load_class_names
from script.deep_models import cnn_model, transfer_learning_model
import script.metrics_plots as mt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import BatchNormalization

ML = False
CROSS_VALIDATION = False
CNN = False
TRANSFER_LEARNING = True

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
        score = []
        results = {}
        if CROSS_VALIDATION:
            for model in models:
                model_path = cv_model_paths[model]
                trained_model = load_ml_model(model_path)
                if trained_model is None:
                    print(f"Alleno {model} con cross validation")
                    cv_scores, trained_model = cross_validate_model(
                        model_type=model,
                        X=X_train,
                        y=y_train,
                        n_splits=5
                    )
                    save_ml_model(trained_model, model_path)
                    print(f"{model} salvato.")
                else:
                    print(f"{model} già allenato. Caricamento completato.")
                mt.evaluate_and_plot_model(
                    model,
                    trained_model,
                    X_test,
                    y_test,
                    results
                )

        elif CROSS_VALIDATION == False:
            for model in models:
                model_path = model_paths[model]
                trained_model = load_ml_model(model_path)
                if trained_model is None:
                    trained_model = train_model(
                        model_type=model,
                        X=X_train,
                        y=y_train,
                    )
                    save_ml_model(trained_model, model_path)
                    print(f"{model} salvato.")
                else:
                    print(f"{model} già allenato. Caricamento completato.")

                mt.evaluate_and_plot_model(
                    model,
                    trained_model,
                    X_test,
                    y_test,
                    results
                )
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

            model = cnn_model()

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
                factor=0.5,      # Dimezzo il learning rate se per 5 epoche non ho miglioramenti
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
                data_gen.flow(X_train, y_train, batch_size=64),
                validation_data=(X_val, y_val),
                epochs=50,  # early stopping e reduce learning rate si occupano dell'overfitting
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=1
            )

            # 4. Salvataggio modello
            model.save(CNN_MODEL_PATH)

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

        X_train = X_train * 255.0
        X_val = X_val * 255.0
        X_test = X_test * 255.0

        if TL_MODEL_PATH.exists():
            print("CNN con transfer learning già allenata. Caricamento modello...")
            model = load_model(TL_MODEL_PATH)

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
            # Creazione modello DenseNet121
            model, base_model = transfer_learning_model(
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

            # Controlli learning rate e epoche
            early_stopping_1 = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            early_stopping_2 = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            checkpoint_1 = ModelCheckpoint(
                filepath=str(TL_MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            )

            checkpoint_2 = ModelCheckpoint(
                filepath=str(TL_MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )

            # FASE 1: FEATURE EXTRACTION
            print("\nFASE 1 - Backbone congelato")
            model.compile(
                optimizer=Adam(
                    learning_rate=1e-3
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history_1 = model.fit(
                data_gen.flow(
                    X_train,
                    y_train,
                    batch_size=64
                ),
                validation_data=(X_val, y_val),
                epochs=30,
                callbacks=[
                    early_stopping_1,
                    reduce_lr,
                    checkpoint_1
                ],
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

            model.compile(
                optimizer=Adam(
                    learning_rate=1e-5
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history_2 = model.fit(
                data_gen.flow(
                    X_train,
                    y_train,
                    batch_size=64
                ),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[
                    early_stopping_2,
                    reduce_lr,
                    checkpoint_2
                ],
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
