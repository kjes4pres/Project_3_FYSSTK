import torch
import torch.nn as nn
import torch.optim as optim

class WeatherNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: str = "relu"
    ):
        super().__init__()

        # Choose activation function
        activations = {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
        }
        act = activations[activation.lower()]

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)
        
        # Hidden layers
        for h in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)