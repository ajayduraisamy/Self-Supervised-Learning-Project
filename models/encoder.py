import torchvision.models as models
import torch.nn as nn

def get_encoder():

    model = models.resnet18(weights="IMAGENET1K_V1")

    model.fc = nn.Identity()

    return model