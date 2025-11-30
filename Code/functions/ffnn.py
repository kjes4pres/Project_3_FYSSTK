import autograd.numpy as np  
from typing import List, Callable, Tuple, Dict, Any
import numpy.typing as npt

# For reproducibility
# Same seed as project 1 & 2
np.random.seed(2018)
seed = np.random.seed(2018)

class NeuralNetwork:
    def __init__(
        self,
        network_input_size: int,
        layer_output_sizes: List[int],
        activation_funcs: List[Callable[[np.ndarray], np.ndarray]],
        activation_ders: List[Callable[[np.ndarray], np.ndarray]],
        cost_fun: Callable[[np.ndarray], np.ndarray],
        cost_der: Callable[[np.ndarray], np.ndarray],
        lamb = 0.0,
        cost_fun_type: str = None
    ):
        self.cost_der = cost_der
        self.cost_fun = cost_fun
        self.activation_ders = activation_ders
        self.activation_funcs = activation_funcs
        self.layer_output_sizes = layer_output_sizes
        self.network_input_size = network_input_size
        self.weights = self.create_layers(network_input_size, layer_output_sizes)
        self.lamb = lamb
        self.cost_fun_type = cost_fun_type
        self.training_info = {
            "Cost_history" : [],
            "RMSProp_velocities": []
            }
    
    # Returns current weights of the model
    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.weights
    
    # Returns the training info, like cost history and such.
    def get_info(self) -> Dict[str, Any]:
        return self.training_info
    
    def get_cost_fun(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.cost_fun
    
    def set_random_seed(self, seed: int) -> None:
        self.weights = self.create_layers(self.network_input_size, self.layer_output_sizes, seed)
    
    # Creates the weights with prechosen shapes.
    def create_layers(self, network_input_size: int, layer_output_sizes: List[int], seed: int = 2018) -> List[Tuple[np.ndarray, np.ndarray]]:
        layers = []
        np.random.seed(seed)
        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size) 
            b = np.random.randn(layer_output_size, ) 
            layers.append((W, b))
            i_size = layer_output_size

        return layers
    
    def _feed_forward(self, input: np.ndarray) -> np.ndarray:
        a = input
        for (W, b), activation_func in zip(self.weights, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, input: np.ndarray, target: np.ndarray) -> float:
        if self.cost_fun_type == "L1":
            reg_term = self.lamb * sum(np.sum(np.abs(W)) for W, b in self.weights)
        elif self.cost_fun_type == "L2":
            reg_term = self.lamb * sum(np.sum(W**2) for W, b in self.weights)
        else:
            reg_term = 0.0
        predict = self._feed_forward(input)
        return self.cost_fun(predict, target) + reg_term

    def _feed_forward_saver(self, inputs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        layer_inputs = [ ]
        zs = []
        a = inputs

        for (W, b), activation_func in zip(self.weights, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def backpropagation_batch(self, inputs: np.ndarray, target: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:

        layer_inputs, zs, prediction = self._feed_forward_saver(inputs)
        batch_size = inputs.shape[0]

        layer_grads = [() for l in self.weights]

        # Loop over layers backward
        for i in reversed(range(len(self.weights))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.weights) - 1:
                # Last layer: derivative of cost w.r.t activation
                if self.cost_fun.__name__ == "cross_entropy" and activation_der.__name__ == "softmax_der":
                    dC_dz = prediction - target
                else:
                    dC_da = self.cost_der(prediction, target) / batch_size
                    dC_dz = dC_da * activation_der(z)
            else:
                W, _ = self.weights[i + 1]
                dC_da = dC_dz @ W.T
                
                dC_dz = dC_da * activation_der(z)
        
            dC_dW = (layer_input.T @ dC_dz) 
            dC_db = np.mean(dC_dz, axis=0)
            
            # Applying regularization if specified
            if self.cost_fun_type == "L1":
                dC_dW += self.lamb * np.sign(self.weights[i][0])

            if self.cost_fun_type == "L2":
                dC_dW += 2*self.lamb * self.weights[i][0]

            # Gradient clipping
            clip_value = 1.0
            total_norm = np.sqrt(np.sum(dC_dW ** 2))
            if total_norm > clip_value:
                dC_dW *= clip_value / (total_norm + 1e-8)

            total_norm_b = np.sqrt(np.sum(dC_db ** 2))
            if total_norm_b > clip_value:
                dC_db *= clip_value / (total_norm_b + 1e-8)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    # Weight update methods based on optimization method
    def update_weights_Adam(self, layer_grads: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> int:
        if not hasattr(self, 't'):
            self.t = 0
        if not hasattr(self, 'm'):
            self.m = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.weights]
        if not hasattr(self, 'v'):
            self.v = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.weights]
        for i in range(len(self.weights)):
            W, b = self.weights[i]
            dC_dW, dC_db = layer_grads[i]
            m_W, m_b = self.m[i]
            v_W, v_b = self.v[i]
            
            m_W = beta1 * m_W + (1 - beta1) * dC_dW
            v_W = beta2 * v_W + (1 - beta2) * (dC_dW ** 2)
            m_b = beta1 * m_b + (1 - beta1) * dC_db
            v_b = beta2 * v_b + (1 - beta2) * (dC_db ** 2)

            m_W_hat = m_W / (1 - beta1 ** (self.t + 1))
            v_W_hat = v_W / (1 - beta2 ** (self.t + 1))

            m_b_hat = m_b / (1 - beta1 ** (self.t + 1))
            v_b_hat = v_b / (1 - beta2 ** (self.t + 1))

            W -= (learning_rate / (np.sqrt(v_W_hat) + epsilon)) * m_W_hat
            b -= (learning_rate / (np.sqrt(v_b_hat) + epsilon)) * m_b_hat

            self.m[i] = (m_W, m_b)
            self.v[i] = (v_W, v_b)
            self.weights[i] = (W, b)

        self.t += 1

    
        # Standard gradient descent
    def update_weights(self, layer_grads: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        for i in range(len(self.weights)):
            W, b = self.weights[i]
            dC_dW, dC_db = layer_grads[i]

            W -= learning_rate * dC_dW
            b -= learning_rate * dC_db

            self.weights[i] = (W, b)
    
    # Training by standard gradient descent.
    def train(self, input: np.ndarray, target: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1, optimizer: str = "gd") -> None:
        for i in range(epochs):
            grads = self.backpropagation_batch(input,target)
            if optimizer == "Adam":
                self.update_weights_Adam(grads, learning_rate)
            else:
                self.update_weights(grads, learning_rate)
            self.training_info["Cost_history"].append(self.cost(input,target))
    
    # Training with stochastic gradient descent.
    def train_SGD(self, input: np.ndarray, target: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1, batch_size: int = 100, optimizer: str = "gd") -> None:
        
        batches = int(input.shape[0] / batch_size)

        for epoch in range(epochs):
            for batch in range(batches):
                index = np.random.choice(input.shape[0], batch_size, replace=True)
                X_batch = input[index]
                y_batch = target[index]
                grads = self.backpropagation_batch(X_batch,y_batch)

                if optimizer == "Adam":
                    self.update_weights_Adam(grads, learning_rate)
                else:
                    self.update_weights(grads, learning_rate) 
            self.training_info["Cost_history"].append(self.cost(input,target))

    
