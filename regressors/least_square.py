import numpy as np


class LeastSquareRegressor:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, shuffle_=True):
        # Compute the weight matrix
        X_ = np.c_[np.ones(len(X)), X]
        self.weights = y.T.dot(np.linalg.pinv(X_).T)

    def predict(self, X):
        X_ = np.c_[np.ones(len(X)), X]
        y_score = self.weights.dot(X_.T)
        return y_score
