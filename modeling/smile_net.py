import torch.nn as nn


class SmileNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        if hidden_dims is None:
            hidden_dims = [128, 64]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
