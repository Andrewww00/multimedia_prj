from keras.models import Sequential
from keras.layers import Rescaling, Dense, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, UpSampling2D
from keras.regularizers import l2
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.optimizers import Adam, AdamW, Nadam
from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from keras.models import Model

MODEL_TYPE = "cnn"


def cnn_model(params=None):
    if params is None:
        params = {}

    # Ricerca parametri e definizione di valori di default
    activation = params.get("activation", "relu")
    weight_decay = params.get("weight_decay", 1e-4)
    base_filters = params.get("base_filters", 64)

    dense_1 = params.get("dense_units_1", 256)
    dense_2 = params.get("dense_units_2", 128)

    dr_base = params.get("dropout_base", 0.1)
    dr_step = params.get("dropout_step", 0.1)
    dr_final = params.get("dropout_final", 0.5)

    lr = params.get("learning_rate", 5e-4)
    opt_name = params.get("optimizer", "adam")
    label_smoothing = params.get("label_smoothing", 0.1)

    model = Sequential()

    # Layer 1 (32x32 -> 16x16)
    model.add(Conv2D(filters=base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    # Strided convolution per il downsampling
    model.add(Conv2D(filters=base_filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(dr_base))

    # Layer 2 (16x16 -> 8x8)
    f2 = base_filters * 2
    dr2 = min(dr_base + dr_step, 0.5)
    model.add(Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=f2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(dr2))

    # Layer 3 (8x8 -> 4x4)
    f3 = base_filters * 4
    dr3 = min(dr2 + dr_step, 0.5)
    model.add(Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=f3, kernel_size=(3, 3), strides=(2, 2), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(dr3))

    # Layer 4 (4x4 -> 2x2)
    f4 = base_filters * 8
    dr4 = min(dr3 + dr_step, 0.5)
    model.add(Conv2D(filters=f4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=f4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                     activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(dr4))

    # Classificazione
    model.add(GlobalAveragePooling2D())

    model.add(Dense(dense_1, activation=activation,
              kernel_regularizer=l2(weight_decay)))
    model.add(Dense(dense_2, activation=activation,
              kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(dr_final))

    model.add(Dense(100, activation="softmax"))

    # Scelgo l'ottimizzatore
    if opt_name == "adamw":
        optimizer = AdamW(learning_rate=lr)
    elif opt_name == "nadam":
        optimizer = Nadam(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )

    return model


def transfer_learning_model(params=None, input_shape=(32, 32, 3), num_classes=100):
    if params is None:
        params = {}

    lr = params.get("learning_rate", 1e-3)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)

    scale = 7

    inputs = tf.keras.Input(shape=input_shape)
    x = Rescaling(255.0)(inputs)
    x = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x)
    x = preprocess_input(x)
    base_model = DenseNet121(
        input_shape=(32*scale, 32*scale, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="DenseNet121_CIFAR100")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def dl_parameter_search(hp, model_type):
    if model_type == "cnn":
        params = {
            "learning_rate": hp.Choice("learning_rate", values=[0.001, 0.0005, 0.0001]),
            "dropout_base": hp.Choice("dropout_base", values=[0.2, 0.3, 0.4]),
            "optimizer": hp.Choice("optimizer", values=["adam", "adamw", "nadam"])
        }

        return cnn_model(params)

    elif model_type == "transfer_learning":
        params = {
            "learning_rate": hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5]),
            "dense_units": hp.Choice("dense_units", [128, 256, 512]),
            "dropout_rate": hp.Choice("dropout_rate", [0.2, 0.3, 0.4])
        }
        model, _ = transfer_learning_model(params)
        return model
