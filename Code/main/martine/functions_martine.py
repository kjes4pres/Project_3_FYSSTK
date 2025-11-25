# Functions (and classes) used in project 2
# (Neural Network class in own file)

# import needed packages
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""---------- Functions ----------"""

def Runge_func(x, n, noise = False, mag_noise = 1):
    # returns Runge's function with or without noise
    # if noise = True, noise is included in the model.
    # n is number of data points, mag_noise is the magnitude of the noise, between 0 and 1
    y = 1/(1+25*x**2) #Runge's function without noise.
    if noise == True:
        noise = np.random.randn(n) #noise given by the standard normal distribution
        y = 1/(1+25*x**2)  + mag_noise * noise #Runge's function with noise
    return y


def scale_data(X_train, X_test, model):
    # function for scaling inputs using Scikit-Learn's StandardScaler or MinMaxScaler
    # model indicates if the NN acts as a regressor or as a classifier, selected from "regressor" and "classifier".
       
    if model == "regressor":
        scaler = StandardScaler(with_mean=True, with_std=False) # center data
    elif model == "classifier":
        scaler = MinMaxScaler() # scale each feature to the interval [0,1] (normalize)
    else: 
         raise ValueError("Please choose model from 'regressor' and 'classifier'.")
    
    scaler.fit(X_train) #scale X based on the training data
    
    X_train_s = scaler.transform(X_train) #training data scaled
    X_test_s = scaler.transform(X_test) #test data scaled

    return X_train_s, X_test_s #return scaled data


def find_best_params(matrix, etas, lambdas, model):
    # returns the best model parameters lambda and eta, as well as the corresponding MSE/accuracy of the regression/classification model
    # model indicates if the NN acts as a regressor or as a classifier, selected from "regressor" and "classifier".
    
    if model == "regressor":
        i, j = np.unravel_index(np.argmin(matrix), matrix.shape) #index of the lowest value in the matrix
    elif model == "classifier":
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape) #index of the highest value in the matrix
    else: 
         raise ValueError("Please choose model from 'regressor' and 'classifier'.")
    
    best_lambda = lambdas[j] # corresponding lambda
    best_eta = etas[i] # corresponding eta
    value = matrix[i, j] # lowest MSE/highest accuracy in the model
    
    if model == "regressor":
        print(fr"Best MSE = {value:.4f} at λ = {best_lambda:.1e}, learning rate = {best_eta:.1e}.")
    elif model == "classifier":
        print(fr"Best accuracy = {value:.4f} at λ = {best_lambda:.1e}, learning rate = {best_eta:.1e}.")

    return best_lambda, best_eta, value



"""---------- Classes ----------"""

class Scheduler:
    """
    Abstract class for schedulers used to update the learning rate in gradient descent.
    Adapted from FYS-STK4155 lectures.
    """

    def __init__(self, eta):
        self.eta = eta

    def update_change(self, gradient):
        raise NotImplementedError("update_change() must be implemented in subclass.")

    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


class Momentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0.0

    def update_change(self, gradient):
        # Calculate change
        self.change = self.momentum * self.change + self.eta * gradient
        
        return self.change

    def reset(self):
        pass


class AdaGrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.r = 0.0

    def update_change(self, gradient):
        delta = 1e-7  # avoid division ny zero

        # Update gradient accumulation variables
        self.r += gradient * gradient 

        # Calculate effective learning rate
        e_eta = self.eta / (np.sqrt(self.r + delta)) 

        return e_eta * gradient

    def reset(self):
        self.r = 0.0


class RMSProp(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.r = 0.0
     
    def update_change(self, gradient):
        delta = 1e-7  # avoid division by zero
    
        # Update gradient accumulation (2nd moment) variables
        self.r = self.rho * self.r + (1 - self.rho) * gradient * gradient
            
        # Calculate effective learning rate
        e_eta = self.eta / (np.sqrt(self.r + delta))
        
        return e_eta * gradient

    def reset(self):
        self.r = 0.0


class ADAM(Scheduler):
    def __init__(self, eta, rho1, rho2):
        super().__init__(eta)
        self.rho1 = rho1
        self.rho2 = rho2
        self.s = 0.0
        self.r = 0.0
        self.time = 0

    def update_change(self, gradient):
        delta = 1e-7  # avoid division by zero

        if isinstance(self.s, float):  # first call
            self.s = np.zeros_like(gradient)
            self.r = np.zeros_like(gradient)

        # Update time
        self.time += 1

        # Update 1st moment variables and correct bias
        self.s = self.rho1 * self.s + (1 - self.rho1) * gradient
        s_hat = self.s / (1 - self.rho1**(self.time))

        # Update 2nd moment and correct bias
        self.r = self.rho2 * self.r + (1 - self.rho2) * gradient * gradient
        r_hat = self.r / (1 - self.rho2**(self.time))

        # Calculate effective learning rate
        e_eta = self.eta * s_hat / (np.sqrt(r_hat) + delta)

        return e_eta

    def reset(self):
        self.s = 0.0
        self.r = 0.0
        self.time = 0


class CostFunction:
    """
    Base class for all cost functions used in the project.
    """
    
    def func(self, targets, predictions, weights=None, lmb=0.0):
        raise NotImplementedError("func() must be implemented in subclass.")
    
    def der(self, targets, predictions, weights=None, lmb=0.0):
        raise NotImplementedError("der() must be implemented in subclass.")


class MSEBase(CostFunction):
    """
    Base class for Mean Squared Error-based cost functions.
    """

    def func(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute MSE + optional regularization.
        """
        n = targets.shape[0]
        mse = (1.0 / n) * np.sum((targets - predictions) ** 2)
        
        # regularization
        if weights is not None:
            mse += self._reg(weights, lmb)
        
        return mse

    def der(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute derivative of MSE w.r.t. predictions.
        """
        n = targets.shape[0]
        grad_pred = (-2.0 / n) * (targets - predictions)
        
        return grad_pred
    
    def reg_der(self, weights, lmb):
        """
        Compute derivative of optional regularization term w.r.t. weights.
        """
        return 0.0

    # Internal methods
    def _reg(self, weights, lmb):
        """
        Optional regularization term.
        """
        return 0.0

    
class MSE(MSEBase):
    """Mean Squared Error with no regularization."""
    pass

class MSEL2(MSEBase):
    """MSE with L2 regularization."""

    def _reg(self, weights, lmb):
        return lmb * np.sum(weights ** 2)

    def reg_der(self, weights, lmb):
        return 2 * lmb * weights

class MSEL1(MSEBase):
    """MSE with L1 regularization."""

    def _reg(self, weights, lmb):
        return lmb * np.sum(np.abs(weights))

    def reg_der(self, weights, lmb):
        return lmb * np.sign(weights)
    
class BCEBase(CostFunction):

    """
    Base class for Binary Cross-Entropy (BCE) loss for binary classification, 
    can be used with no regularization, or with L1 or L2 regularization.
    Used with Sigmoid output layer.
    
    - targets: True binary labels (0 or 1)
    - predictions: Model predictions (probabilities from sigmoid)
    """

    def func(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute BCE + optional regularization.
        """
        n = targets.shape[0]
        bce = -(1.0 / n) * np.sum((targets * np.log(predictions + 1e-10)) + ((1 - targets) * np.log(1 - predictions + 1e-10)))
        
        # regularization
        if weights is not None:
            bce += self._reg(weights, lmb)
        
        return bce

    def der(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute derivative of BCE w.r.t predictions.
        """
        n = targets.shape[0]

        targets = targets.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)
        grad_pred = -(1.0 / n) * (targets / (predictions + 1e-10) - (1 - targets) / (1 - predictions + 1e-10))
        
        return grad_pred
    
    def reg_der(self, weights, lmb):
        """
        Compute derivative of optional regularization term w.r.t. weights.
        """
        return 0.0

    # Internal methods
    def _reg(self, weights, lmb):
        """
        Optional regularization term.
        """
        return 0.0
    
class BCE(BCEBase):
    """Binary Cross Entropy with no regularization"""
    pass

class BCEL2(BCEBase):
    """Binary Cross Entropy with L2 regularization"""

    def _reg(self, weights, lmb):
        return lmb * np.sum(weights ** 2)

    def reg_der(self, weights, lmb):
        return 2 * lmb * weights

class BCEL1(BCEBase):
    """Binary Cross Entropy with L1 regularization"""

    def _reg(self, weights, lmb):
        return lmb * np.sum(np.abs(weights))

    def reg_der(self, weights, lmb):
        return lmb * np.sign(weights)


class CCEBase(CostFunction):

    """
    Class for Categorical Cross-Entropy (CCE) loss for multi-class classification (Softmax Cross Entropy).
    Important note: it is assumed that the activation function for the output layer is Softmax.
    
    targets: One-hot encoded true labels
    predictions: Model predictions (probabilities from softmax)
    """

    def func(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute CCE.
        """
        n = targets.shape[0]
        cce = -(1.0 / n) * np.sum(targets * np.log(predictions + 1e-10))

        # regularization
        if weights is not None:
            bce += self._reg(weights, lmb)

        return cce

    def der(self, targets, predictions, weights=None, lmb=0.0):
        """
        Compute derivative of CCE in combination with Softmax w.r.t. predictions.
        """
        n = targets.shape[0]
        grad_pred = (1.0 / n) * (predictions - targets)
        
        return grad_pred
    
    def reg_der(self, weights, lmb):
        """
        Compute derivative of optional regularization term w.r.t. weights.
        """
        return 0.0

    # Internal methods
    def _reg(self, weights, lmb):
        """
        Optional regularization term.
        """
        return 0.0

    
class CCE(CCEBase):
    """Categorical Cross Entropy with no regularization"""
    pass

class CCEL2(CCEBase):
    """Categorical Cross Entropy with L2 regularization"""

    def _reg(self, weights, lmb):
        return lmb * np.sum(weights ** 2)

    def reg_der(self, weights, lmb):
        return 2 * lmb * weights

class CCEL1(CCEBase):
    """Categorical Cross Entropy with L1 regularization"""

    def _reg(self, weights, lmb):
        return lmb * np.sum(np.abs(weights))

    def reg_der(self, weights, lmb):
        return lmb * np.sign(weights)
    

class ActivationFunction():
    """
    Base class for all activation functions used in the project.
    """
    
    def func(self, z):
        """
        Activation function.
        """
        raise NotImplementedError("func() must be implemented in subclass.")
    
    def der(self, z):
        """
        Activation function derivative.
        """
        raise NotImplementedError("der() must be implemented in subclass.")
    
class Identity(ActivationFunction):
    """
    Linear output.
    """
    def func(self, z):
        return z
    
    def der(self, z):
        return 1
    
class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """

    def func(self, z):
        z = np.clip(z, -500, 500) # prevent the exponential from blowing up by limiting the values in z
        return 1.0 / (1 + np.exp(-z))
    
    def der(self, z):
        """Compute derivative of the Sigmoid function."""
        s = self.func(z)
        return s * (1 - s)

class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    def func(self, z):
        return np.where(z > 0, z, 0)
    
    def der(self, z):
        """Compute derivative of the ReLU function."""
        return np.where(z > 0, 1, 0)

class LReLU(ActivationFunction):
    """
    Leaky ReLU activation function.
    """

    def func(self, z):
        alpha = 0.01
        return np.where(z > 0, z, alpha * z)
    
    def der(self, z):
        """Compute derivative of the Leaky ReLU function."""
        alpha = 0.01
        return np.where(z > 0, 1, alpha)
    
class Softmax(ActivationFunction):
    """
    Softmax activation function.
    Important note: only for use with the CCE cost function.
    """

    def func(self, z):
        z = z - np.max(z, axis=-1, keepdims=True) # ensure numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def der(self, z):
        """The derivative of the Softmax function is computed in combination with the CCE cost function."""
        return 1