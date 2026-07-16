import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from script.config import DATASET_DIR, ML_DIR, CNN_DIR

def unpickle(file_path):
    """ Legge i file binari pickle e restituisce un dizionario """
    with open(file_path, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

def validate_dataset(file_path):
    """ Controlla che il file npz esista, sia leggibile e contenga X e y """
    try:
        with np.load(file_path) as data:
            if "X" not in data:
                return False

            if "y" not in data:
                return False

            if len(data["X"]) == 0:
                return False

            if len(data["y"]) == 0:
                return False
        return True

    except Exception as e:
        print(f"Errore lettura {file_path}: {e}")
        file_path.unlink(missing_ok=True)
        return False

def reconstruct_image(flat_array):
    """ Funzione di ricostruzione delle immagini per la CNN """
    img_reshaped = flat_array.reshape(3, 32, 32)
    img = img_reshaped.transpose(1, 2, 0)
    return img

def normalize_hist(hist):
    """ Funzione di normalizzazione degli istogrammi """
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist.flatten()

def extract_features_ML(img, bins=[32, 32, 32]):
    """ Estrazione features per gli algoritmi di ML """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Canale H (Tonalità): i valori in OpenCV vanno da 0 a 180
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins[0]], [0, 180])
    # Canale S (Saturazione): i valori vanno da 0 a 256
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins[1]], [0, 256])
    # Canale V (Luminosità): i valori vanno da 0 a 256
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins[2]], [0, 256])

    hist_h = normalize_hist(hist_h)
    hist_s = normalize_hist(hist_s)
    hist_v = normalize_hist(hist_v)

    """ HOG FEATURES """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    feature_vector = np.concatenate([
        hist_h.flatten(), 
        hist_s.flatten(), 
        hist_v.flatten(), 
        hog_features
    ]).astype(np.float32)
    
    return feature_vector

def process_dataset(raw_images, labels, extract_ml=True, extract_cnn=True):
    """ Crazione dataset ML e CNN """
    X_ml = []
    X_cnn = []

    for idx, flat_img in enumerate(tqdm(raw_images)):
        try:
            img = reconstruct_image(flat_img)
            if extract_cnn:
                X_cnn.append(img.astype(np.float32) / 255.0)
            if extract_ml:
                features = extract_features_ML(img)
                X_ml.append(features)
        except Exception as e:
            print(f"Errore immagine {idx}: {e}")

    if extract_ml:
        out_ml = np.array(X_ml, dtype=np.float32)
    else:
        out_ml = None
    if extract_cnn:
        out_cnn = np.array(X_cnn, dtype=np.float32)
    else:
        out_cnn = None

    return out_ml, out_cnn, np.array(labels)

def save_dataset(path, X, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)

def prepare_dataset():
    """ Orchestratore che richiama le funzioni per l'estrazione, creazione e salvataggio dei dataset"""

    # Creazione variabili per controlli e cicli
    splits = ["train", "val", "test"]
    ml_files = {split: ML_DIR / f"{split}.npz" for split in splits}
    cnn_files = {split: CNN_DIR / f"{split}.npz" for split in splits}
    all_files = list(ml_files.values()) + list(cnn_files.values())
    train_dict = unpickle(DATASET_DIR / "train")
    test_dict = unpickle(DATASET_DIR / "test")

    # Estrazione dati dal dizionario
    raw_train_images = train_dict[b'data']
    raw_train_labels = train_dict[b'fine_labels']
    raw_test_images = test_dict[b'data']
    raw_test_labels = test_dict[b'fine_labels']

    """ Divisione di TRAIN in TRAIN e VALIDATION """

    train_images, val_images, train_labels, val_labels = train_test_split(
        raw_train_images,
        raw_train_labels,
        test_size=0.2,
        random_state=42,
        stratify=raw_train_labels
    )

    
    datasets_to_process = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (raw_test_images, raw_test_labels)
    }

    for split_name, (images, labels) in datasets_to_process.items():
        ml_file = ml_files[split_name]
        cnn_file = cnn_files[split_name]

        # Controlliamo l'integrità dei singoli file
        ml_valid = ml_file.exists() and validate_dataset(ml_file)
        cnn_valid = cnn_file.exists() and validate_dataset(cnn_file)

        # Se entrambi sono pronti, passiamo al prossimo split
        if ml_valid and cnn_valid:
            print(f"Split '{split_name.upper()}' (ML e CNN) già presenti e validi")
            continue

        print(f"\nProcessing {split_name.upper()}...")
        X_ml, X_cnn, y = process_dataset(
            images, 
            labels, 
            extract_ml=not ml_valid, 
            extract_cnn=not cnn_valid
        )

        # Salvo solo quello che e' stato ricalcolato
        if not ml_valid:
            if X_ml is None or len(X_ml) == 0:
                raise RuntimeError(f"Errore: estrazione ML fallita per {split_name}.")
            save_dataset(ml_file, X_ml, y)
            
        if not cnn_valid:
            if X_cnn is None or len(X_cnn) == 0:
                raise RuntimeError(f"Errore: estrazione CNN fallita per {split_name}.")
            save_dataset(cnn_file, X_cnn, y)