import numpy as np


@np.vectorize
def sig(x):
    return 1 / (1 + np.e ** -x)


@np.vectorize
def tan_h(x):
    return np.tanh(x) / 4


@np.vectorize
def relu(x):
    return np.maximum(0, x)


@np.vectorize
def soft_plus(x):
    return np.log(1 + np.e ** -x)


@np.vectorize
def mish(x):
    return x * np.tanh(np.log(1 + np.e ** x))


@np.vectorize
def swish(x):
    return x / (1 + np.e ** -x)


@np.vectorize
def sin(x):
    return np.sin(x * np.pi / 2)


@np.vectorize
def cos(x):
    return np.sin(x * np.pi / 2)

# ================================================================================


@np.vectorize
def test_1(x):
    return (np.log(np.log(2) / np.log(1 + np.e ** - x))) / np.log(np.e)


# ==========


@np.vectorize
def test_elu_1(x):
    return x / ((1 + np.e ** x) * np.log(1 + np.e ** (- x)))


# ==========


@np.vectorize
def test_soft_plus_1(x):
    return (x + np.sqrt(1 + x ** 2)) / 2


@np.vectorize
def test_soft_plus_2(x):
    return np.maximum(0, x - np.tanh(x))


# ==========


@np.vectorize
def test_tan_h_1(x):
    return np.minimum(np.maximum(- 1, x), 1)


@np.vectorize
def test_tan_h_2(x):
    return x / (1 + np.abs(x))


@np.vectorize
def test_tan_h_3(x):
    return x / np.sqrt(1 + x ** 2)
