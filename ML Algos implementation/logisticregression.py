import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / samples) * np.dot(X.T, (y_pred - y))
            db = (1 / samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        lr_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(lr_model)

        return np.array([1 if i > 0.5 else 0 for i in y_pred])


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 1, 0, 1, 0])
lgr = LogisticRegression()
lgr.fit(X, y)
op = lgr.predict([6])
print(op)
