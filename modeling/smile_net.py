import torch.nn as nn


class SmileNet(nn.Module):
    def __init__(self, input_dim, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
