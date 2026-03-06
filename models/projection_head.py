import torch.nn as nn

class ProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self,x):
        return self.net(x)