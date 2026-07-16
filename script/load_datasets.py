from script.config import *
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def load(files):
    datasets = []
    for file in files:
        with np.load(file) as data:
            X = data["X"]
            y = data["y"]
            print(X.shape, y.shape)
            datasets.append((X, y))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    return X_train, y_train, X_val, y_val, X_test, y_test



def compose_dataset(CNN = False):
    if CNN:
        files = [CNN_TRAIN_FILE, CNN_VAL_FILE, CNN_TEST_FILE]
        X_train, y_train, X_val, y_val, X_test, y_test = load(files)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # One-hot encoding delle label
        y_train = to_categorical(y_train, num_classes=100)
        y_val = to_categorical(y_val, num_classes=100)
        y_test = to_categorical(y_test, num_classes=100)
    
    else:
        files = [ML_TRAIN_FILE, ML_VAL_FILE, ML_TEST_FILE]
        X_train, y_train, X_val, y_val, X_test, y_test = load(files)
        # label encoding
        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        joblib.dump(le, LABEL_ENCODER_PATH)
        
    return X_train, y_train, X_val, y_val, X_test, y_test
    