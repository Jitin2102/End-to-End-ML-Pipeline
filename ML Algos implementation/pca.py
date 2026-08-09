import numpy as np


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix deriviation
        cov = np.cov(X_centered.T)

        # Eigen Values and Eigen Vectors Breakdown
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Transpose to access them as row
        eigenvectors = eigenvectors.T

        # Sort the column based on magnitude idxs
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs[: self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)


X = [[1, 2, 3], [4, 5, 6]]
y = [1, 2, 4, 5, 6]

pca = PCA()
pca.fit(X)
transformed = pca.transform(X)
print(transformed)
