# import numpy as np
# import json
# import joblib
# from sklearn.preprocessing import LabelEncoder
from script.preprocessing import prepare_dataset
from script.ml_models import cross_validate_model, train_model, evaluate_model
from script.config import DATA_DIR, ML_TRAIN_FILE, ML_VAL_FILE, ML_TEST_FILE, MODELS_DIR, LABEL_ENCODER_PATH
from script.load_datasets import compose_dataset
from script.deep_models import cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

CROSS_VALIDATION = False
CNN = True

# print(tf.config.list_physical_devices('GPU'))
# print(tf.__version__)
# print(tf.config.list_physical_devices())
# print(tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    prepare_dataset()
    if not CNN:
        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset()

        # TRAIN
        models = ["logistic_reg", "random_forest"]
        results = []
        score = []
        if CROSS_VALIDATION:
            print("Alleno con la cross validation\n")
            for model in models:
                results.append(cross_validate_model(
                    model_type= model,
                    X=X_train,
                    y=y_train,  
                    n_splits=5
                ))
        else:
            print("Alleno il modello")
            for model in models:
                trained_model = train_model(
                    model_type= model,
                    X=X_train,
                    y=y_train,
                )
                score.append(evaluate_model(trained_model, X_test, y_test))

    else:
        print("Alleno la CNN")
        X_train, y_train, X_val, y_val, X_test, y_test = compose_dataset(CNN)
        model = cnn_model()

        # 1. DATA AUGMENTATION
        # Genero nuove versioni delle immagini
        data_gen = ImageDataGenerator(
            rotation_range=15,      # ruoto immagine di 15 gradi
            width_shift_range=0.1,  # sposto orizzontalmente del 10%
            height_shift_range=0.1, # sposta verticalmente del 10%
            horizontal_flip=True,   # specchio orizzontalmente
            zoom_range=0.1
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
            factor=0.5,             # Dimezzo il learning rate se per 5 epoche non ho miglioramenti
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

        # 4. VALUTAZIONE FINALE
        print("\nValuto sul Test Set")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
