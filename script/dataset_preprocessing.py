import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" /  "fruits-360-original-size" / "Training"
TEST_PATH = BASE_DIR / "data" / "fruits-360-original-size" / "Test"
VAL_PATH = BASE_DIR / "data" /  "fruits-360-original-size" / "Validation"

# print("\nIl train path è:\n", TRAIN_PATH)
# print("\nIl test path è:\n", TEST_PATH)
# print("\nIl val path è:\n", VAL_PATH)

def extract_features_ML(img_path, new_shape=(100,100), color=114):
    img = cv2.imread(img_path)
    # print(img.shape)

    # Inserisco la width e la hight dell'immagine in 2 variabili
    hight, widht = img.shape[:2]

    # Calcolo il fattore di scala mantenendo le proporzioni dell'immagine
    w_new, h_new = new_shape
    scale_factor = min(w_new / widht, h_new / hight)
    h_scaled = int(hight * scale_factor)
    w_scaled = int(widht * scale_factor)

    # Ridimensiono immagine
    img_resized = cv2.resize(img, (w_scaled, h_scaled))
    #print(img_resized.shape) # (91, 100, 3) size

    # Faccio il padding per ottenere le dimensioni che voglio
    canvas = np.full((h_new, w_new, 3), color, dtype=np.uint8) 

    # Centro l'immagine calcolando l'offset
    top = (h_new - h_scaled) // 2
    left = (w_new - w_scaled) //2

    # Metto l'immagine al centro
    canvas[
            top:top + h_scaled,
            left:left + w_scaled
        ] = img_resized

    #print(canvas.shape) # (100, 100, 3) size
    return canvas


def load_process_dataset(dataset_path):
    X = []
    y = [] 
    dataset_dir = Path(dataset_path)
    
    # Estraiamo tutte le cartelle delle classi (escludendo eventuali file sparsi)
    class_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    
    print(f"Trovate {len(class_folders)} classi. Inizio l'estrazione...")
    
    for folder in class_folders:
        label = folder.name
        
        for img_path in folder.glob("*.*"):
            # Salta i file che non sono immagini
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Qui chiameremo il nostro "motore" (Livello Micro) passandogli il percorso come stringa
            # features = extract_features(str(img_path))
            
            # X.append(features)
            y.append(label)
            
    return np.array(X), np.array(y)

def plot(img_path, title="Immagine ridimensionata"):
    """
    Mostra un'immagine (OpenCV -> matplotlib).
    """
    img = extract_features_ML(img_path)
    # Conversione BGR -> RGB (OpenCV usa BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# load_process_dataset(TRAIN_PATH)

test_img_path = r"C:\Users\andri\Desktop\Fruits-Classifier\data\fruits-360-original-size\Training\Apple 11\r0_2.jpg"
plot(test_img_path)
