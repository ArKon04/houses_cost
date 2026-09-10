import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(previous_dim, hidden_dim)
            )

            layers.append(
                nn.ReLU()
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, 1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

