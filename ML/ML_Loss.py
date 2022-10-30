import numpy as np


@np.vectorize
def loss_abs(yHat, y):
    return np.abs(yHat - y)


@np.vectorize
def loss_abs_(yHat, y):
    return (yHat - y) / np.abs(yHat - y)


@np.vectorize
def loss_square(yHat, y):
    return (yHat - y) ** 2


@np.vectorize
def loss_square_(yHat, y):
    return 2 * (yHat - y)


@np.vectorize
def loss_log(yHat, y):
    return - np.log(yHat) if y == 1 else - np.log(1 - yHat)


@np.vectorize
def loss_log_(yHat, y):
    return - 1 / yHat if y == 1 else 1 / (1 - yHat)
