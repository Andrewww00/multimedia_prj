from pathlib import Path
import kagglehub

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(kagglehub.dataset_download("fedesoriano/cifar100"))

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ML_DIR = PROCESSED_DIR / "ml"
CNN_DIR = PROCESSED_DIR / "cnn"

# Salvataggio modelli allenati per evitare il training
MODELS_DIR = BASE_DIR / "trained_models"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_reg.pkl"
LOGISTIC_CV_MODEL_PATH = MODELS_DIR / "logistic_reg_cv.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
RANDOM_FOREST_CV_MODEL_PATH = MODELS_DIR / "random_forest_cv.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_cifar100.keras"

# ML datasets
ML_TRAIN_FILE = ML_DIR / "train.npz"
ML_VAL_FILE = ML_DIR / "val.npz"
ML_TEST_FILE = ML_DIR / "test.npz"

# CNN datasets
CNN_TRAIN_FILE = CNN_DIR / "train.npz"
CNN_VAL_FILE = CNN_DIR / "val.npz"
CNN_TEST_FILE = CNN_DIR / "test.npz"

# Path per i plot
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)