import torch
import torch.nn as nn
import torch.optim as optim

from models.encoder import get_encoder
from models.projection_head import ProjectionHead
from utils.plot_results import plot_loss


def train_ssl(loader, device):

    encoder = get_encoder().to(device)
    projector = ProjectionHead().to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(projector.parameters()),
        lr=0.001
    )

    loss_fn = nn.MSELoss()

    epoch_losses = []

    for epoch in range(5):

        total_loss = 0

        for images, _ in loader:

            images = images.to(device)

            features = encoder(images)
            outputs = projector(features)

            loss = loss_fn(outputs, torch.zeros_like(outputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        epoch_losses.append(avg_loss)

        print("SSL Epoch:", epoch, "Loss:", avg_loss)

    torch.save(encoder.state_dict(), "results/ssl_encoder.pth")

    plot_loss(epoch_losses, "SSL Training Loss", "ssl_loss.png")