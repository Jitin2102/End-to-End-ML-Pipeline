import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.bias * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([7, 8, 9, 10, 11])
lr = LinearRegression()
lr.fit(X, y)
op = lr.predict([6])

rmse = np.sqrt((((op - 12) ** 2) / 5) * 100)
print(op)
print(rmse)
