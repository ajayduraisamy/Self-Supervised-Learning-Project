import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.dataset_split import create_low_label_split
from training.ssl_train import train_ssl
from training.finetune import finetune_model


def main():

    print("Starting Self-Supervised Learning Project")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Basic transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

   
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    print("Total dataset size:", len(dataset))

   
    labeled_data, unlabeled_data = create_low_label_split(dataset)

    print("Labeled data size:", len(labeled_data))
    print("Unlabeled data size:", len(unlabeled_data))

    # Data loaders
    ssl_loader = DataLoader(
        unlabeled_data,
        batch_size=32,
        shuffle=True
    )

    labeled_loader = DataLoader(
        labeled_data,
        batch_size=32,
        shuffle=True
    )

    # Step 1: Self-Supervised Training
    print("\nStarting SSL Training\n")
    train_ssl(ssl_loader, device)

    # Step 2: Fine-tuning with labeled data
    print("\nStarting Fine-Tuning\n")
    finetune_model(labeled_loader, device)

    print("\nTraining Complete")
    print("Models saved in results/ folder")


if __name__ == "__main__":
    main()