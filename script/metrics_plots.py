import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
from sklearn import metrics
import seaborn as sns
from script.config import PLOTS_DIR
from script.ml_models import evaluate_model

DISTANCE = 0.5


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

    plot_confusion_matrix(
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


def save_plot(filename):
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def calc_conf_interval(scores, confidence=0.95):
    n = len(scores)
    mean_score = np.mean(scores)
    if n < 2:
        return mean_score, 0.0

    std_err = np.std(scores, ddof=1) / np.sqrt(n)
    margin_of_error = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean_score, margin_of_error


def evaluate(y_true, y_pred, model_name="Modello"):
    """Calcolo metriche"""
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    rec = metrics.recall_score(
        y_true, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n--- Performance: {model_name} ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    return [acc * 100, prec * 100, rec * 100, f1 * 100]


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot matrice di confusione"""
    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True,
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    save_path = save_plot(filename)
    print(f"Grafico salvato in {save_path}")


def plot_training_history(history, title="Model Training History"):
    """Plot curve di Loss e Accuracy CNN"""
    plt.figure(figsize=(12, 5))

    # Estraiamo il dizionario a prescindere da come ci viene passato
    hist_dict = history.history if hasattr(history, 'history') else history

    # Plot della Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist_dict['loss'], label='Training Loss', color='blue')
    if 'val_loss' in hist_dict:
        plt.plot(hist_dict['val_loss'],
                 label='Validation Loss', color='orange')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot dell'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist_dict['accuracy'], label='Training Accuracy', color='blue')
    if 'val_accuracy' in hist_dict:
        plt.plot(hist_dict['val_accuracy'],
                 label='Validation Accuracy', color='orange')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    save_path = save_plot(filename)
    print(f"Grafico salvato in {save_path}")


def plot_models_comparison(models_dict):
    """
    Crea il grafico a barre affiancate (come nel vecchio notebook) per confrontare più modelli.
    models_dict si aspetta la struttura: {"Nome Modello": [acc, prec, rec, f1]}
    """
    models = list(models_dict.keys())

    accuracy = [models_dict[m][0] for m in models]
    precision = [models_dict[m][1] for m in models]
    recall = [models_dict[m][2] for m in models]
    f1_score = [models_dict[m][3] for m in models]

    x = np.arange(len(models)) * DISTANCE
    width = 0.1

    fig, axis = plt.subplots(figsize=(7, 6))

    axis.bar(x - 1.5 * width, accuracy, width,
             label='Accuracy', color='#4C72B0')
    axis.bar(x - 0.5 * width, precision, width,
             label='Precision', color='#55A868')
    axis.bar(x + 0.5 * width, recall, width, label='Recall', color='#C44E52')
    axis.bar(x + 1.5 * width, f1_score, width,
             label='F1 Score', color='#8172B3')

    axis.set_xlabel('Modelli')
    axis.set_ylabel('Performance (%)')
    axis.set_title('Confronto Metriche Modelli su CIFAR-100')
    axis.set_xticks(x)
    axis.set_xticklabels(models)

    # Aggiungiamo i valori numerici sopra le barre per massima chiarezza
    for container in axis.containers:
        axis.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

    axis.legend()
    plt.ylim(0, 110)  # Diamo spazio per le etichette sopra le barre
    plt.tight_layout()
    filename = "confronto_totale_ml.png"
    save_path = save_plot(filename)
    print(f"Grafico salvato in {save_path}")


def predict_image(indices, model, X_test, y_test, class_names, save_path):
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    n_images = len(indices)
    cols = 5
    rows = (n_images + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for i, index in enumerate(indices):
        image = X_test[index]
        prediction = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )
        predicted_class = np.argmax(prediction)
        true_class = y_test[index]

        if image.max() > 1:
            image = image / 255.0

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {class_names[true_class]}\n"
            f"Pred: {class_names[predicted_class]}\n"
            f"Conf: {np.max(prediction):.2f}",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Immagini salvate in {save_path}")
