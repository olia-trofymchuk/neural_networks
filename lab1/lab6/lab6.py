"""
Лабораторна робота №6
з дисципліни “Нейронні мережі”
виконала студентка групи АнД-31
Трофимчук Ольга

Тема:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Adaline:
    def __init__(self, n_iterations=100, random_state=42, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.cost = []
        self.weights = None

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # 1 = bias, X.shape[1] = number of features
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        i = 0
        while i < self.n_iterations and (not self.cost or self.cost[-1] > 0.01):
            errors = []
            i += 1
            for x, target in zip(X, y):
                output = self.predict(x)
                error = target - output
                self.weights[1:] += self.learning_rate * np.dot(x, error)
                self.weights[0] += self.learning_rate * error
                errors.append(0.5 * error ** 2)
            self.cost.append(np.mean(errors))

    def predict(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]


"""
# first iteration
x1 = np.random.normal(0, 0.3, 100)
y1 = np.random.normal(0, 0.3, 100)
x2 = np.random.normal(2, 0.3, 100)
y2 = np.random.normal(2, 0.3, 100)
"""

# second iteration
x1 = np.random.normal(2, 1, 100)
y1 = np.random.normal(1, 0.1, 100)
x2 = np.random.normal(-1, 0.4, 20)
y2 = np.random.normal(1.8, 0.4, 20)

X1 = np.column_stack((x1, y1))
y1 = np.ones(len(X1))

X2 = np.column_stack((x2, y2))
y2 = -np.ones(len(X2))

X = np.vstack((X1, X2))
y = np.concatenate((y1, y2))

df = pd.DataFrame(X, columns=["X1", "X2"])
df["label"] = y
df = df.sample(frac=1).reset_index(drop=True)

adaline = Adaline()
adaline.fit(df[["X1", "X2"]].values, df["label"].values)

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot([-10, 10], [-(adaline.weights[0] + adaline.weights[1] * (-10)) / adaline.weights[2],
                     -(adaline.weights[0] + adaline.weights[1] * 10) / adaline.weights[2]], color="r")
plt.legend(["Decision Boundary", "Class 1", "Class 2"])
plt.show()

plt.plot(adaline.cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

print("Weights:", adaline.weights)
