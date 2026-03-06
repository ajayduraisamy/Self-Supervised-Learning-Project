import torchvision.models as models
import torch.nn as nn

def get_encoder():

    model = models.resnet18(pretrained=False)

    model.fc = nn.Identity()

    return model