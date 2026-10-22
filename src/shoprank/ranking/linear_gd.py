"""Linear regression trained with batch gradient descent, written from scratch in NumPy."""

import numpy as np


class LinearRegressionGD:
    def __init__(self, lr: float = 0.01, n_steps: int = 1000):
        self.lr = lr
        self.n_steps = n_steps

    def fit(self, X, y) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w, b = np.zeros(X.shape[1]), 0.0
        self.losses_ = []
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.n_steps):
                error = X @ w + b - y
                self.losses_.append(float((error ** 2).mean()))  # MSE before this step
                w -= self.lr * 2 * (X.T @ error) / len(y)
                b -= self.lr * 2 * error.mean()
        self.coef_, self.intercept_ = w, b
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
