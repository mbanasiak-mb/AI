import numpy as np


def xyz_data(model):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    px, py = np.meshgrid(x, y)

    px = px.reshape(-1)
    py = py.reshape(-1)

    X = np.array([px, py]).T
    model.X = X

    pz = model.predict()
    pz = pz.reshape(-1)

    return px, py, pz


def xyz_train(samepls):
    px = np.random.random((1, samepls))
    py = np.random.random((1, samepls))
    pz = np.array(np.random.random((1, samepls)) < 0.5, dtype=float)
    px = px.reshape(1, -1)
    py = py.reshape(1, -1)
    pz = pz.reshape(1, -1)
    return px[0, :], py[0, :], pz[0, :]
