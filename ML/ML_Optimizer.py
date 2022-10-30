import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def init_memory(self, l):
        pass

    @abstractmethod
    def clean_memory(self):
        pass

    @abstractmethod
    def get_result(self, th, l):
        pass


class GD(Optimizer):
    def __init__(self, alpha):
        self.a = alpha

    def init_memory(self, l):
        pass

    def clean_memory(self):
        pass

    def get_result(self, th, l):
        o = self.a * th
        return o


class Momentum(Optimizer):
    def __init__(self, alpha, beta=0.9):
        self.a = alpha
        self.b = beta

        self.m = []
        self.t = 1

    def init_memory(self, l):
        self.clean_memory()
        self.m = [0] * l

    def clean_memory(self):
        self.m = []
        self.t = 1

    def get_result(self, th, l):
        self.m[l] = self.b * self.m[l] + (1 - self.b) * th
        mHat = self.m[l] / (1 - self.b ** self.t)
        o = self.a * mHat
        self.t += 1
        return o


class RMSProp(Optimizer):
    def __init__(self, alpha, beta=0.9, epsilon=10e-6):
        self.a = alpha
        self.b = beta
        self.e = epsilon

        self.s = []

    def init_memory(self, l):
        self.clean_memory()
        self.s = [0] * l

    def clean_memory(self):
        self.s = []

    def get_result(self, th, l):
        self.s[l] = self.b * self.s[l] + (1 - self.b) * th ** 2
        o = self.a * th / np.sqrt(self.s[l] + self.e)
        return o


class AdaM(Optimizer):
    def __init__(self, alpha, beta1=0.9, beta2=0.999, epsilon=10e-8):
        self.a = alpha
        self.b1 = beta1
        self.b2 = beta2
        self.e = epsilon

        self.m = []
        self.s = []
        self.t = 1

    def init_memory(self, l):
        self.clean_memory()
        self.m = [0] * l
        self.s = [0] * l

    def clean_memory(self):
        self.m = []
        self.s = []
        self.t = 1

    def get_result(self, th, l):
        self.m[l] = self.b1 * self.m[l] + (1 - self.b1) * th
        self.s[l] = self.b2 * self.s[l] + (1 - self.b2) * th ** 2
        mHat = self.m[l] / (1 - self.b1 ** self.t)
        sHat = self.s[l] / (1 - self.b2 ** self.t)
        o = self.a * mHat / (np.sqrt(sHat) + self.e)
        self.t += 1
        return o


# ==========

class OptT(Optimizer):
    def __init__(self, alpha, beta):
        self.a = alpha
        self.b = beta

        self.d2 = []
        self.k = []

    def init_memory(self, l):
        self.clean_memory()
        self.d2 = [None] * l
        self.k = [self.a] * l

    def clean_memory(self):
        self.d2 = []
        self.k = []

    def get_result(self, th, l):
        d = np.sign(th)
        if self.d2[l] is None:
            self.d2[l] = d

        t = self.b ** (1 - np.abs(d + self.d2[l]) / 2)
        self.k[l] /= t

        o = d * self.k[l]
        self.d2[l] = d
        return o
