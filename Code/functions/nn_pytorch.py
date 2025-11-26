import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class WeatherNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_hidden_layers: int = None,
        activation: str = None
    ):  
        
        super().__init__()
        

        self.cost = nn.CrossEntropyLoss()
        #self.output_layer = nn.Softmax()
        
        if num_hidden_layers:
            # Choose activation function
            activations = {
                "relu": nn.ReLU(),
                "lrelu": nn.LeakyReLU(),
                "sigmoid": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "elu": nn.ELU()
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
            #layers.append(nn.Softmax(dim=1))

        
        else:
            layers = []
            # Input layer
            layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)
    
    def get_model(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)
    
    def train_model(self, train_loader, lr: float, epochs: int, lmb: float=None, reg_type: str = None):
        """
        reg_type: None, "L1", or "L2"
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for e in range(epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(X)
                loss = self.cost(outputs, y)

                if reg_type == "L1":
                    L1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss += lmb * L1_norm

                if reg_type == "L2":
                    L2_norm = sum(p.pow(2).sum() for p in self.parameters())
                    loss += lmb * L2_norm
                    
                loss.backward()
                optimizer.step()
    
    def evaluate(self, loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in loader:
                outputs=self.forward(X) 
                preds = outputs.argmax(dim=1) 
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total
