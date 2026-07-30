import keras_tuner as kt
from deep_models import MODEL_TYPE, parameter_search
from load_datasets import compose_dataset
import gc
import tensorflow as tf

class ClearMemoryCallback(tf.keras.callbacks.Callback):

    def on_train_end(self, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()

X_train, y_train, X_val, y_val, _, _ = compose_dataset(CNN=True)

tuner = kt.GridSearch(
    hypermodel=parameter_search,
    objective="val_accuracy",
    max_trials=27,
    directory="tuner_results",
    project_name=f"grid_search_{MODEL_TYPE}"
)

tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[ClearMemoryCallback()]
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\nMIGLIORI PARAMETRI TROVATI PER: {MODEL_TYPE.upper()}")
print("Learning rate:", best_hp.get("learning_rate"))

if MODEL_TYPE == "cnn":
    print("Dropout base :", best_hp.get("dropout_base"))
    print("Optimizer    :", best_hp.get("optimizer"))
else:
    print("Dense units  :", best_hp.get("dense_units"))
    print("Dropout rate :", best_hp.get("dropout_rate"))
    print("Activation   :", best_hp.get("activation"))
