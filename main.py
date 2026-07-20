
import os
from script.preprocessing import prepare_dataset
from script.ml_models import cross_validate_model, train_model, evaluate_model, save_ml_model, load_ml_model
from script.config import MODELS_DIR, LOGISTIC_MODEL_PATH, LOGISTIC_CV_MODEL_PATH, RANDOM_FOREST_MODEL_PATH, RANDOM_FOREST_CV_MODEL_PATH, CNN_MODEL_PATH
from script.load_datasets import compose_dataset
from script.deep_models import cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import script.metrics_plots as mt

CROSS_VALIDATION = True
CNN = False

model_paths = {
    "logistic_reg": LOGISTIC_MODEL_PATH,
    "random_forest": RANDOM_FOREST_MODEL_PATH
}

cv_model_paths = {
    "logistic_reg": LOGISTIC_CV_MODEL_PATH,
    "random_forest": RANDOM_FOREST_CV_MODEL_PATH
}

def evaluate_and_plot_model(model_name, model, X_test, y_test, results):

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1']*100:.2f}%")

    mt.plot_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        title=f"Confusion Matrix - {model_name}"
    )

    results[model_name] = [
        metrics["accuracy"] * 100,
        metrics["precision"] * 100,
        metrics["recall"] * 100,
        metrics["f1"] * 100
    ]

if __name__ == "__main__":
    prepare_dataset()
    if not CNN:
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
                evaluate_and_plot_model(
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
                        model_type= model,
                        X=X_train,
                        y=y_train,
                    )
                    save_ml_model(trained_model, model_path)
                    print(f"{model} salvato.")
                else:
                    print(f"{model} già allenato. Caricamento completato.")
                    
                evaluate_and_plot_model(
                    model,
                    trained_model,
                    X_test,
                    y_test,
                    results
                )
        mt.plot_models_comparison(results)
                
    elif CNN:
        print("Alleno la CNN")
        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset(CNN)
        
        if CNN_MODEL_PATH.exists():

            print("CNN già allenata. Caricamento modello...")
            model = load_model(CNN_MODEL_PATH)
        
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
                patience=10,    # Aspetta 15 epoche
                restore_best_weights=True   # Torno ai pesi prima dell'overfittin
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,      # Dimezzo il learning rate se per 5 epoche non ho miglioramenti
                patience=5,
                min_lr=1e-5
            )

            # 3. TRAINING
            print("Inizio training")
            history = model.fit(
                data_gen.flow(X_train, y_train, batch_size=64),
                validation_data=(X_val, y_val),
                epochs=50,  #early stopping e reduce learning rate si occupano dell'overfitting
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # 4. Salvataggio modello
            model.save(CNN_MODEL_PATH)
            print("CNN salvata.")
            
            # 5. VALUTAZIONE FINALE
            print("\nValuto sul Test Set")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            print(f"\nTest loss: {loss:.4f}")
            print(f"Test accuracy: {accuracy:.4f}")
