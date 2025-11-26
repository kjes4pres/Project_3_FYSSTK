import numpy as np  

# Different activation functions
def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


# Derivatives of the activation functions
def derivate(func: callable):
    if func.__name__ == "RELU":

        def der_func(X):
            return np.where(X > 0, 1, 0)

        return der_func

    elif func.__name__ == "LRELU":

        def der_func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return der_func

    elif func.__name__ == "sigmoid":

        def der_func(X):
            s = sigmoid(X)
            return s * (1 - s)
        return der_func  
    
    elif func.__name__ == "softmax":

        def softmax_der(X):
            return None
        return softmax_der

    elif func.__name__ == "identity":

        def der_func(X):
            return np.ones(X.shape)
        return der_func

    else:
        return print("Derivative not implemented for this function.")
