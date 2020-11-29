import numpy as np
from sklearn.utils import shuffle

class ADALINERegressor:
    def __init__(self):
        self.avg_error = []
        self.W = None

    def fit(self, X, y, alpha=0.001, epoches=10):
        X = np.c_[np.ones(len(X)), X]

        # Init weight matrix
        self.W = 0.1 * np.random.randn(X.shape[1])

        for _ in range(epoches):
            X, y = shuffle(X, y, random_state=100)
            for i in range(X.shape[0]):
                x = X[i, :]
                y_pred = self.W.dot(x)
                error = y[i] - y_pred

                # Normalize the x vector
                x_norm = x/np.linalg.norm(X[0, :])

                # Update the weights
                delta = np.outer(error, x_norm)
                self.W = self.W + alpha*delta

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        y_pred = self.W.dot(X.T)[0]
        return y_pred
