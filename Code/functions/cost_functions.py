import numpy as np

def cross_entropy(predict, target):
    return -np.sum(target*np.log(predict))

def cross_entropy_der(predict, target):
    return target/predict*(-1)