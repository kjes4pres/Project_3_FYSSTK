import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class WeatherNN(nn.Module):
    """
    A configurable feed-forward neural network through PyTorch.

    Parameters
    ----------
    input_dim : int
        Number of the input features.
    output_dim : int
        Number of output classes.
    hidden_dim : int, optional
        Size of each hidden layer. Required if `num_hidden_layers` is specified.
    num_hidden_layers : int, optional
        Number of hidden layers. If None, the model reduces to a single linear
        layer mapping `input_dim` to `output_dim`. Can be used for simple logistic regression.
    activation : str, optional
        Activation function to use in hidden layers. Options: {"relu", "lrelu", "sigmoid"}.
        Required if `num_hidden_layers` is specified.

    Attributes
    ----------
    cost : nn.CrossEntropyLoss
        Loss function used for training. Additional functions can be added as needed.
    model : nn.Sequential
        Layers are added to this sequential model based on the initialization parameters.
    """
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
        
        if num_hidden_layers:
            # Choose activation function
            activations = {
                "relu": nn.ReLU(),
                "lrelu": nn.LeakyReLU(),
                "sigmoid": nn.Sigmoid()
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
        
        else:
            layers = []
            # Input layer
            layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)
    
    def get_model(self):
        """
        Returns the underlying PyTorch model.
        """
        return self.model
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.model(x)
    
    def train_model(self, train_loader, lr: float, epochs: int, lmb: float=None, reg_type: str = None):
        """
        Train the neural network.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        lr : float
            Learning rate for the optimizer.
        epochs : int
            Number of training epochs.
        lmb : float, optional
            Regularization parameter. Required if `reg_type` is specified.
        reg_type : str, optional
            Type of regularization to apply. Options: {"L1", "L2"}.
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
        """
        Evaluate the model's accuracy on a given dataset.

        Parameters
        ----------
        loader : DataLoader
            DataLoader for the dataset to evaluate.
        """
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
