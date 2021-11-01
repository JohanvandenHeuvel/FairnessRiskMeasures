import numpy as np
from numpy.linalg import norm


def hinge_loss(actual, predicted):
    loss = np.array(1 - np.multiply(actual, predicted)).clip(0)
    return np.mean(loss)


def square_hinge_loss(actual, predicted):
    return hinge_loss(actual, predicted) ** 2


def L1_loss(actual, predicted):
    return 1 / len(actual) * norm(actual - predicted, 1)


def L2_loss(actual, predicted):
    return 1 / len(actual) * (norm(actual - predicted, 2) ** 2)
