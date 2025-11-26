import numpy as np

def cross_entropy(predict, target):
    eps = 1e-9
    return -np.mean(target * np.log(predict + eps))

def cross_entropy_der(predict, target):
    return (predict - target)