import numpy as np


class KMeans:
    def __init__(self, K=3, max_iters=1000):
        self.K = K
        self.max_iters = max_iters
        self.centroids = None

    def fit_predict(self, X):
        samples, features = X.shape
        # Initialize centroids randomly from dataset indices
        random_sample_idx = np.random.choice(samples, self.K, replace=False)
        self.centroids = X[random_sample_idx]
        for _ in range(self.max_iters):
            # Assign samples to closest centroids
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            # Recompute centroids as means of assigned samples
            new_centroids = np.array(
                [
                    X[labels == k].mean(axis=0)
                    if len(X[labels == k]) > 0
                    else self.centroids[k]
                    for k in range(self.K)
                ]
            )
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return labels
