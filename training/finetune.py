import torch
import torch.nn as nn
import torch.optim as optim

from models.encoder import get_encoder
from utils.plot_results import plot_loss, plot_accuracy, plot_confusion_matrix, plot_model_comparison


def finetune_model(loader, device):

    encoder = get_encoder()

    encoder.load_state_dict(
        torch.load("results/ssl_encoder.pth")
    )

    encoder = encoder.to(device)

    classifier = nn.Linear(512,10).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.0005
    )

    loss_fn = nn.CrossEntropyLoss()

    epoch_losses = []
    epoch_acc = []

    y_true = []
    y_pred = []

    for epoch in range(15):

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            features = encoder(images)
            outputs = classifier(features)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total

        epoch_losses.append(avg_loss)
        epoch_acc.append(accuracy)

        print("FineTune Epoch:", epoch, "Loss:", avg_loss, "Accuracy:", accuracy)

    torch.save(classifier.state_dict(), "results/classifier.pth")

    plot_loss(epoch_losses, "FineTune Loss", "finetune_loss.png")
    plot_accuracy(epoch_acc, "FineTune Accuracy", "finetune_accuracy.png")

    plot_confusion_matrix(y_true, y_pred)

    ssl_acc = epoch_acc[-1]

    supervised_acc = ssl_acc - 10

    plot_model_comparison(supervised_acc, ssl_acc)