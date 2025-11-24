import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class WeatherNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str = None
    ):  
        
        super().__init__()
        

        self.cost = nn.CrossEntropyLoss()
        self.output_layer = nn.Softmax()
        
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
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)
    
    def get_model(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)
    
    def train(self, x, y, lr: float):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        outputs = self.forward(x)
        loss = self.cost(outputs, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def pred(self, x):
        return self.model(x)

        