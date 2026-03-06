import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# create results folder if not exists
os.makedirs("results", exist_ok=True)


def plot_loss(losses, title, filename):
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.savefig(f"results/{filename}")
    plt.close()


def plot_accuracy(acc, title, filename):
    plt.figure()
    plt.plot(acc, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.savefig(f"results/{filename}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig("results/confusion_matrix.png")
    plt.close()


def plot_model_comparison(supervised_acc, ssl_acc):
    models = ["Supervised", "SSL + FineTune"]
    acc = [supervised_acc, ssl_acc]

    plt.figure()
    plt.bar(models, acc)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")

    plt.savefig("results/model_comparison.png")
    plt.close()