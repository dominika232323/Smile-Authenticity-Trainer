import torch.nn as nn


class SimpleMultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
