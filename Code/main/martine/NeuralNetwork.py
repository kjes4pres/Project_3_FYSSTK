# Neural Network class

# import needed packages
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

class NeuralNetwork:
    """
    Self-implemented feed-forward neural network with flexible architecture design. 
    Can be used for regression and classification tasks.

    - network_input_size: number of features per input.
    - layer_output_sizes: list of number of nodes in the hidden layers and output layer.
    - hidden_func: activation function for the hidden layers.
    - output_func: activation function for the output layer.
    - cost_func: cost function.
    - seed: random seed for reproducibility and comparability.
    """
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        hidden_func,
        output_func,
        cost_func,
        seed = None
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.classification = None
        self.regularization = None

        self._set_classification()
        self._set_regularization()
        

    def create_layers(self):
        """
        Creates a list of layers with random weights and biases.
        Each layer is a tuple (W, b).
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        layers = []

        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size) # weights, shape (output_size, input_size)
            b = np.random.randn(layer_output_size) * 0.01 # bias, shape (output_size,)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
        

    def activation_list(self):
        """
        Creates list of the activation functions for use in upcoming methods.
        """

        activation_funcs = []
        
        for i in range(len(self.layer_output_sizes)-1):
            activation_funcs.append(self.hidden_func)
        
        activation_funcs.append(self.output_func)
        
        return activation_funcs


    def feed_forward(self, inputs, layers, activation_funcs):
        """
        Simple feed-forward pass.
        """
        a = inputs

        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W.T + b #intermediary
            a = activation_func.func(z) #activation function

        return a
    
    def feed_forward_saver(self, inputs, layers, activation_funcs):
        """
        Forward pass that saves layer inputs, pre-activations, and activations for use in back-propagation.
        """
        layer_inputs = []
        zs = []
        a = inputs
        
        for (W, b), activation_func in zip(layers, activation_funcs):
            layer_inputs.append(a)
            
            z = a @ W.T + b #intermediary
            a = activation_func.func(z) #activation function

            zs.append(z)

        return layer_inputs, zs, a


    def compute_gradients(self, inputs, layers, activation_funcs, targets, lmb = None):
        """
        Returns gradients of the cost function w.r.t. the weights and w.r.t. the biases.
        """

        if lmb == None: # if no regularization
            lmb = 0.0 

        layer_inputs, zs, predict = self.feed_forward_saver(inputs, layers, activation_funcs) # feed-forward

        layer_grads = [() for layer in layers] # for storing gradients

        # Start from the last layer (output layer) and propagate backwards
        for i in reversed(range(len(layers))):
            layer_input, z, activation_func = layer_inputs[i], zs[i], activation_funcs[i]

            if i == len(layers) - 1:
                # Output layer
                (W, b) = layers[i]
                dC_da = self.cost_func.der(targets, predict, W, lmb)
            else:
                # Hidden layers
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W

            da_dz = activation_func.der(z)
            dC_dz = dC_da * da_dz
      
            if layer_input.ndim == 1: # single batch input
                dC_dW = np.outer(dC_dz, layer_input)
                dC_db = dC_dz
            else: # batched input - use the average gradients
                dC_dW_batch = []
                batch_size = layer_input.shape[0]
                for m in range(batch_size):
                    dC_dW_batch.append(np.outer(dC_dz[m,:], layer_input[m,:]))
                dC_dW = np.mean(dC_dW_batch, axis=0)
                dC_db = np.mean(dC_dz, axis=0) 

            # regularization term
            if self.regularization:
                dC_dW += self.cost_func.reg_der(layers[i][0], lmb)  

            layer_grads[i] = (dC_dW, dC_db)
        
        return layer_grads


    def update_layers(self, layers, layer_grads, schedulers_W, schedulers_b):
        """
        Update each layer's weights and biases using gradient descent.
        """

        for idx, ((W, b), (dC_dW, dC_db)) in enumerate(zip(layers, layer_grads)):
            # compute changes in parameters
            dW_change = schedulers_W[idx].update_change(dC_dW)
            db_change = schedulers_b[idx].update_change(dC_db)

            # update parameters
            W_updated = W - dW_change
            b_updated = b - db_change

            # store updated parameters back into layers
            layers[idx] = (W_updated, b_updated)

        return layers


    def train_network_GD(self, inputs, layers, activation_funcs, targets, scheduler, max_num_iters = 1000, lmb = None):
        """
        Trains the network using plain (full-batch) gradient descent.
        Returns the same list of layer parameters (weights and biases), but updated to minimize the cost function.
        (Not used in Project 2).
        """

        if lmb == None: # if no regularization
            lmb = 0.0

        print(f"{scheduler.__class__.__name__}: Learning rate = {scheduler.eta}, Regularization parameter = {lmb}")

        # Create independent scheduler copies for weights and biases
        schedulers_W = [deepcopy(scheduler) for _ in layers]
        schedulers_b = [deepcopy(scheduler) for _ in layers]
    
        # gradient descent loop
        for t in range(max_num_iters):

            old_layers = deepcopy(layers)
                        
            # back-propagate
            layer_grads = self.compute_gradients(inputs, layers, activation_funcs, targets, lmb)
            layers = self.update_layers(layers, layer_grads, schedulers_W, schedulers_b)

            # Stop iteration if the difference between new and old parameters is less than the given tolerance
            tol = 1e-6 # tolerance for when to stop the iteration
            convergence = all(np.all(np.abs(layer[0] - old_layer[0]) <= tol) and np.all(np.abs(layer[1] - old_layer[1]) <= tol) for layer, old_layer in zip(layers, old_layers))
            if convergence == True: 
                print("Number of iterations to find parameters:", t)
                break
            else:
                continue

        return layers
    

    def train_network_SGD(self, inputs, layers, activation_funcs, targets, scheduler, epochs = 100, M = 10, lmb = None, history = False):
        """
        Trains the network using stochastic (mini-batch) gradient descent.
        Returns the same list of layer parameters (weights and biases), but updated to minimize the cost function.
        
        Parameters:
        - epochs: number of epochs
        - M: batch size
        - lmb: regularization parameter
        
        history = True indicates that the model performance will be tracked after each epoch.
        """

        # set seed
        if self.seed is not None:
            np.random.seed(self.seed)

        if lmb == None: # if no regularization
            lmb = 0.0
    
        n_inputs = inputs.shape[0]
    
        m = n_inputs // M # number of batches

        print(f"{scheduler.__class__.__name__}: Learning rate = {scheduler.eta}, Regularization parameter = {lmb}")

        # Create independent scheduler copies for weights and biases
        schedulers_W = [deepcopy(scheduler) for _ in layers]
        schedulers_b = [deepcopy(scheduler) for _ in layers]

        # initialize list for containing evaluation metrics after each epoch if history = True
        if history == True:
            epoch_metrics = []

        # loop through epochs
        for e in range(epochs):
            # Shuffle data at the start of each epoch
            shuffled_indices = np.random.permutation(len(inputs))
            inputs_shuffled = inputs[shuffled_indices]
            targets_shuffled = targets[shuffled_indices]

            old_layers = deepcopy(layers)
        
            # loop through mini-batches
            for t in range(m):
                if t == m - 1:
                    # Last batch (remaining data points)
                    inputs_batch = inputs_shuffled[t * M :, :]
                    targets_batch = targets_shuffled[t * M :]
                else:
                    inputs_batch = inputs_shuffled[t * M : (t + 1) * M]
                    targets_batch = targets_shuffled[t * M : (t + 1) * M] 
            
                # back-propagate
                layer_grads = self.compute_gradients(inputs_batch, layers, activation_funcs, targets_batch, lmb)
                layers = self.update_layers(layers, layer_grads, schedulers_W, schedulers_b)

            # evaluate model after each epoch if history == True
            if history == True:
                loss = self.evaluate(inputs, layers, activation_funcs, targets)
                epoch_metrics.append(loss)

            # Stop iteration if the difference between new and old parameters is less than the given tolerance after each epoch
            tol = 1e-6 # tolerance for when to stop the iteration
            convergence = all(np.all(np.abs(layer[0] - old_layer[0]) <= tol) and np.all(np.abs(layer[1] - old_layer[1]) <= tol) for layer, old_layer in zip(layers, old_layers))
            if convergence == True:            
                print("Number of epochs to find parameters:", t)
                break
            else:
                continue

        if history == True:
            return layers, epoch_metrics
        else:
            return layers
    

    def evaluate(self, inputs, layers, activation_funcs, targets): 
        """
        Evaluates the quality of the model on the given data.
        Automatically detects regression / binary / multi-class problems and returns a dictionary with relevant metrics.
        """

        predictions = self.feed_forward(inputs, layers, activation_funcs) # predict by feed-forward
    
        results = {} # for storing metrics

        if self.classification: # Classification problems
            
            if self.layer_output_sizes[-1] == 1: # Binary classification
                predictions = (predictions.flatten() >= 0.5).astype(int) # returns 1 for values >= 0.5 and 0 otherwise
                targets = targets.flatten().astype(int)
                results["accuracy"] = accuracy_score(targets, predictions)
            
            else: # Multiclass classification
                pred_labels = np.argmax(predictions, axis=1) # convert to class labels
                true_labels = np.argmax(targets, axis=1) # convert to class labels
                results["accuracy"] = accuracy_score(true_labels, pred_labels)
        
        else: # Regression problem
            results["r2"] = R2(targets, predictions) # R2 score
            results["mse"] = MSE(targets, predictions) # MSE

        return results
    
    def predict(self, inputs, layers, activation_funcs):
        """
        Performs prediction by feed-forward after training of the network has been finished.
        """
        predictions = self.feed_forward(inputs, layers, activation_funcs)

        if self.classification: # if classification problem, convert predictions to class labels
            predictions = np.argmax(predictions, axis=1)
        
        return predictions


    
    def _set_regularization(self):
        """ 
        Decides if the cost function is without (False) or with (True) regularization, 
        sets self.regularization during init().
        """
        self.regularization = False
        if (
            self.cost_func.__class__.__name__ == "MSEL1"
            or self.cost_func.__class__.__name__ == "MSEL2"
            or self.cost_func.__class__.__name__ == "BCEL1"
            or self.cost_func.__class__.__name__ == "BCEL2"
            or self.cost_func.__class__.__name__ == "CCEL1"
            or self.cost_func.__class__.__name__ == "CCEL2"
    
        ):
            self.regularization = True

    def _set_classification(self):
        """
        Decides if the NN acts as classifier (True) og regressor (False),
        sets self.classification during init().
        """
        self.classification = False
        if (
            self.cost_func.__class__.__name__ == "BCE"
            or self.cost_func.__class__.__name__ == "BCEL1"
            or self.cost_func.__class__.__name__ == "BCEL2"
            or self.cost_func.__class__.__name__ == "CCE"
            or self.cost_func.__class__.__name__ == "CCEL1"
            or self.cost_func.__class__.__name__ == "CCEL2"
        ):
            self.classification = True