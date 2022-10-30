import numpy as np


def xor_data():

    x = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])

    y = np.array([
        [0],
        [1],
        [0],
        [1]
    ])

    return x, y
