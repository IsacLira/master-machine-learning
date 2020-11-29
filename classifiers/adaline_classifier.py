import numpy as np
from sklearn.utils import shuffle
from .utils import encode


class ADALINEClassifier:
    def __init__(self):
        self.avg_error = []
        self.W = None

    def fit(self, X, y, alpha=0.001, epoches=10):
        Y = encode(y) # Encode labels into binary matrix
        Y = 2*Y - 1 # Transform 0 to -1

        # Init weight matrix
        self.W = 0.1*np.random.randn(Y.shape[1], X.shape[1])
        for _ in range(epoches):
            error_lis = []
            X, Y = shuffle(X, Y, random_state=100)
            for i in range(X.shape[0]):
                x = X[i, :]
                y_pred = self.W.dot(x)
                error = Y[i, :] - y_pred.T

                # Normalize x vector
                x_norm = x/np.linalg.norm(X[0, :])

                # Update the weights
                delta = np.outer(error, x_norm)
                self.W = self.W + alpha*delta
                error_lis.append(sum(error**2))
            self.avg_error.append(sum(error_lis)/len(X))

    def predict(self, X):
        Y_score = self.W.dot(X.T)
        y_pred = np.argmax(Y_score, 0)
        return y_pred
