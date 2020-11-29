import numpy as np
from sklearn.utils import shuffle
from .utils import encode

class LeastSquareClassifier:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, shuffle_=True):

        if shuffle_:
            X, y = shuffle(X, y, random_state=100)

        # Encode the labels
        Y = encode(y)

        # Compute the weight matrix
        self.weights = Y.T.dot(np.linalg.pinv(X).T)

    def predict(self, X):
        y_score = self.weights.dot(X.T)
        y_pred = np.argmax(y_score, axis=0)
        return y_pred
