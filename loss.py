import random
import numpy as np
from math import sqrt

class Loss:
    def compute(self, x : np.array) -> float :
        pass

class Linear(Loss):
    def __init__(self, n : int):
        self.n = n
        self.a = np.random.rand(1, n)
    def compute(self, x: np.array) -> float:
        return np.matmul(self.a, x)[0]
    def __str__(self):
        return "Linear"

class Quadratic(Loss):
    def __init__(self, n : int):
        self.n = n
        self.a = np.random.rand(1, n)
        self.b = np.random.rand(n, n)
    def compute(self, x : np.array) -> float:
        return (np.matmul(self.a, x) + np.matmul(x.T, np.matmul(b, x)))[0,0]
    def __str__(self):
        return "Quadratic"

class Brownian(Loss):
    def __init__(self, n : int, interval : float = 1.0):
        self.n = n
        self.t = [0]
        while len(self.t) < n:
            self.t.append(random.uniform(self.t[-1], interval))
        self.t.append(interval)
        self.b = np.zeros((n, n))
        for row in range(0, n):
            for col in range(0, row+1):
                self.b[row][col] = sqrt(self.t[col+1] - self.t[col])
    def compute(self, x : np.array) -> float:
        return np.matmul(self.b, x).max()
    def __str__(self):
        return "Brownian"
    
class NewBrownian(Loss):
    def __init__(self, n : int):
        self.b = [np.random.uniform() for i in range(0, n)]
        self.n = n
    def compute(self, x : np.array) -> float:
        x = list(x)
        return max(list(map(
            lambda i : self.b[i] * x[i],
            range(0, self.n)
        )))
    def __str__(self):
        return "New Brownian"
