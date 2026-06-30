import cv2
from tqdm import tqdm
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

def extract_features_ML(img_path, new_shape=(100,100), color=255, bins=[32, 32, 32]):
    
    """--- 0. LETTURA DELLE IMMAGINI ---"""
    img = cv2.imread(img_path)
    # Controllo che l'immagine sia stata letta correttamente
    if img is None:
        raise ValueError(f"Impossibile leggere {img_path}")

    """--- 1. RIDIMENSIONAMENTO ---"""
    # Inserisco la width e la height dell'immagine in 2 variabili separate
    height, width = img.shape[:2]

    # Calcolo il fattore di scala per mantenere le proporzioni dell'immagine
    w_new, h_new = new_shape
    scale_factor = min(w_new / width, h_new / height)
    h_scaled = int(height * scale_factor)
    w_scaled = int(width * scale_factor)

    # Dopo il resize l'immagine ha shape (91, 100, 3)
    img_resized = cv2.resize(img, (w_scaled, h_scaled))
    #print(img_resized.shape)

    """--- 2. PADDING ---"""
    # Padding per ottenere le dimensioni desiderate
    canvas = np.full((h_new, w_new, 3), color, dtype=np.uint8) 

    # Centro l'immagine calcolando l'offset
    top = (h_new - h_scaled) // 2
    left = (w_new - w_scaled) //2

    # Metto l'immagine al centro
    # Dopo questa operazione la shape risulta (100, 100, 3)
    canvas[
            top:top + h_scaled,
            left:left + w_scaled
        ] = img_resized

    #print(canvas.shape)
    
    """--- 3. CONVERSIONE DA BGR A HSV ---"""
    # Conversione da BGR a HSV
    hsv_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)

    # Canale H (Tonalità): i valori in OpenCV vanno da 0 a 180
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins[0]], [0, 180])
    
    # Canale S (Saturazione): i valori vanno da 0 a 256
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins[1]], [0, 256])
    
    # Canale V (Valore/Luminosità): i valori vanno da 0 a 256
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins[2]], [0, 256])

    feature_vector = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    """--- 4. NORMALIZZAZIONE ---"""
    # Dividiamo ogni valore per la somma totale dei pixel 
    total_pixels = feature_vector.sum()
    if total_pixels > 0:
        feature_vector = feature_vector / total_pixels

    return feature_vector

def load_process_dataset(dataset_path):
    X = []
    y = [] 
    dataset_dir = Path(dataset_path)
    
    # Estraiamo tutte le cartelle delle classi (escludendo eventuali file sparsi)
    class_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    
    print(f"Trovate {len(class_folders)} classi. Inizio l'estrazione...")
    
    for folder in tqdm(class_folders):
        label = folder.name
        
        for img_path in folder.glob("*.*"):
            # Salta i file che non sono immagini
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Qui chiameremo il nostro "motore" (Livello Micro) passandogli il percorso come stringa
            features = extract_features_ML(str(img_path))
            
            X.append(features)
            y.append(label)
            
    return np.array(X), np.array(y)

# def plot(title="Immagine ridimensionata", new_shape=(100,100), color=255):
#     img_path = r"C:\Users\andri\Desktop\Fruits-Classifier\data\fruits-360-original-size\Training\Apple 11\r0_2.jpg"
    
#     """--- 0. LETTURA DELLE IMMAGINI ---"""
#     img = cv2.imread(img_path)
#     # Controllo che l'immagine sia stata letta correttamente
#     if img is None:
#         raise ValueError(f"Impossibile leggere {img_path}")

#     """--- 1. RIDIMENSIONAMENTO ---"""
#     # Inserisco la width e la height dell'immagine in 2 variabili separate
#     height, width = img.shape[:2]

#     # Calcolo il fattore di scala per mantenere le proporzioni dell'immagine
#     w_new, h_new = new_shape
#     scale_factor = min(w_new / width, h_new / height)
#     h_scaled = int(height * scale_factor)
#     w_scaled = int(width * scale_factor)

#     # Dopo il resize l'immagine ha shape (91, 100, 3)
#     img_resized = cv2.resize(img, (w_scaled, h_scaled))
#     #print(img_resized.shape)

#     """--- 2. PADDING ---"""
#     # Padding per ottenere le dimensioni desiderate
#     canvas = np.full((h_new, w_new, 3), color, dtype=np.uint8) 

#     # Centro l'immagine calcolando l'offset
#     top = (h_new - h_scaled) // 2
#     left = (w_new - w_scaled) //2

#     # Metto l'immagine al centro
#     # Dopo questa operazione la shape risulta (100, 100, 3)
#     canvas[
#             top:top + h_scaled,
#             left:left + w_scaled
#         ] = img_resized

#     plt.imshow(canvas)
#     if title:
#         plt.title(title)
#     plt.axis("off")
#     plt.show()

#plot()
X, y = load_process_dataset(TEST_PATH)
print(X.shape)
print(y.shape)