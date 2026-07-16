from keras.models import Sequential    
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization   
from keras.regularizers import l2
from keras.optimizers import Adam

def cnn_model():

    model = Sequential()

    weight_decay = 1e-4

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.10))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation="relu", 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.40))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(
    256,
    activation="relu",
    kernel_regularizer=l2(weight_decay)
    ))

    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Dense(
    128,
    activation="relu",
    kernel_regularizer=l2(weight_decay)
    ))

    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Dense(100, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])    
    model.summary()

    return model