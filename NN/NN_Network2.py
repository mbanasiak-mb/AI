import numpy as np
from AI.ML.ML_Optimizer import Optimizer


class Net2D:

    @staticmethod
    def f_d(f, x):
        h = 10e-12
        return (f(x + h) - f(x - h)) / (2 * h)

    def f_cost(self, yHat, y):
        return np.mean(self.f_loss(yHat, y), axis=0, keepdims=True)

    def __init__(self, bone: [int]):
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

        self.c_optimizer_W: Optimizer = None
        self.c_optimizer_b: Optimizer = None

    def init_optimizers(self):
        self.c_optimizer_W.init_memory(self.L)
        self.c_optimizer_b.init_memory(self.L)

    def init_parameters(self):
        self.clean_parameters()
        self.L = len(self.bone) - 1
        for l in range(0, self.L):
            W = np.random.random((self.bone[l], self.bone[l + 1]))
            b = np.zeros((1, self.bone[l + 1]))
            self.W.append(W)
            self.b.append(b)

    def clean_parameters(self):
        self.W = []
        self.b = []

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
        dbl = []
        dWl = []

        l = self.L - 1
        g = self.f_loss_d(self.A[l], self.Y) * Net2D.f_d(self.F[l], self.Z[l])
        dW = np.dot(self.A[l - 1].T, g)
        dbl.insert(0, g.copy())
        dWl.insert(0, dW.copy())

        for _ in range(1, self.L - 1):
            g = np.dot(g, self.W[l].T) * Net2D.f_d(self.F[l - 1], self.Z[l - 1])
            dW = np.dot(self.A[l - 2].T, g)
            dbl.insert(0, g.copy())
            dWl.insert(0, dW.copy())
            l -= 1

        g = np.dot(g, self.W[l].T) * Net2D.f_d(self.F[l - 1], self.Z[l - 1])
        dW = np.dot(self.X.T, g)
        dbl.insert(0, g.copy())
        dWl.insert(0, dW.copy())

        for l in range(0, self.L):
            dW = self.c_optimizer_W.get_result(dWl[l], l)
            self.W[l] -= dW
            db = self.c_optimizer_b.get_result(dbl[l], l)
            self.b[l] -= np.mean(db, axis=0, keepdims=True)

    def train(self, acc):
        for _ in range(0, acc):
            self.feed_forward()
            self.back_propagation()
            cost = self.f_cost(self.A[self.L - 1], self.Y)

            self.history.append(cost.reshape(1,))

    def predict(self, x=None):
        if x is not None:
            self.X = x

        self.feed_forward()
        return self.A[self.L - 1]
