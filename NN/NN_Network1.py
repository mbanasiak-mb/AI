import numpy as np


class Net2D:

    @staticmethod
    def f_d(f, x):
        h = 10e-12
        return (f(x + h) - f(x - h)) / (2 * h)

    def f_cost(self, yHat, y):
        return np.mean(self.f_loss(yHat, y), axis=0, keepdims=True)

    def __init__(self, bone: [int]):
        self.alpha = 0

        self.bone = bone
        self.L = 0
        self.W = []
        self.b = []
        self.F = []

        self.Z = []
        self.A = []
        self.history = []

        self.X = None
        self.Y = None

        self.f_loss = None
        self.f_loss_d = None

    def init_parameters(self):
        self.L = len(self.bone) - 1
        for l in range(0, self.L):
            W = np.random.random((self.bone[l], self.bone[l + 1]))
            b = np.zeros((1, self.bone[l + 1]))
            self.W.append(W)
            self.b.append(b)

    def feed_forward(self):
        self.Z = []
        self.A = []

        A = self.X
        for l in range(0, self.L):
            Z = np.dot(A, self.W[l]) + self.b[l]
            A = self.F[l](Z)
            self.Z.append(Z)
            self.A.append(A)

    def back_propagation(self):
        dBl = []
        dWl = []

        l = self.L - 1
        g = self.f_loss_d(self.A[l], self.Y) * Net2D.f_d(self.F[l], self.Z[l])
        dw = np.dot(self.A[l - 1].T, g)
        dBl.insert(0, g.copy())
        dWl.insert(0, dw.copy())

        for _ in range(1, self.L - 1):
            g = np.dot(g, self.W[l].T) * Net2D.f_d(self.F[l - 1], self.Z[l - 1])
            dw = np.dot(self.A[l - 2].T, g)
            dBl.insert(0, g.copy())
            dWl.insert(0, dw.copy())
            l -= 1

        g = np.dot(g, self.W[l].T) * Net2D.f_d(self.F[l - 1], self.Z[l - 1])
        dw = np.dot(self.X.T, g)
        dBl.insert(0, g.copy())
        dWl.insert(0, dw.copy())

        for l in range(0, self.L):
            self.W[l] -= self.alpha * dWl[l]
            self.b[l] -= self.alpha * np.mean(dBl[l], axis=0, keepdims=True)

    def train(self, acc):
        for _ in range(0, acc):
            self.feed_forward()
            self.back_propagation()
            cost = self.f_cost(self.A[self.L - 1], self.Y)

            self.history.append(cost.reshape(1,))

    def predict(self, X=None):
        if X is not None:
            self.X = X

        self.feed_forward()
        return self.A[self.L - 1]
