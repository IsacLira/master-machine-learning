import numpy as np
from numpy.random import randn
from .utils import encode
from sklearn.utils import shuffle


class MLP1:
    """Multiperceptrom Model with one hidden layer
    """

    def __init__(self,
                 hidden_neurons,
                 epoches=1,
                 eta=0.01,
                 verbose=False):
        self.hidden_neurons = hidden_neurons
        self.epoches = epoches
        self.eta = eta
        self.verbose = verbose

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def foward_step(self, xi):
        self.hidden_act = self._sigmoid(
            self.w_hidden.dot(xi)
        )
        # Add bias term
        self.hidden_act_ = np.insert(self.hidden_act, 0, 1)
        self.output_act = self._sigmoid(
            self.w_output.dot(self.hidden_act_))

    def backward_step(self, xi, yi):

        # Compute error
        self.error = yi - self.output_act

        # Compute gradients for output layer
        local_grad = self.output_act*(1 - self.output_act)
        output_grad = self.error*local_grad

        # Compute gradiends for hidden layer
        local_grad = self.hidden_act*(1 - self.hidden_act)
        hidden_grad = local_grad*(self.w_output[:, 1:].T.dot(output_grad))

        self.w_output += self.eta*np.outer(output_grad, self.hidden_act_)
        self.w_hidden += self.eta*np.outer(hidden_grad, xi)

    def fit(self, X, y):
        Y = encode(y)
        n_class, n_feats = Y.shape[1], X.shape[1]

        # Init weights
        self.w_hidden = 0.01*randn(self.hidden_neurons, n_feats+1)
        self.w_output = 0.01*randn(n_class, self.hidden_neurons+1)

        for e in range(self.epoches):
            if self.verbose:
                print(f'Training epoch {e}')

            X, Y = shuffle(X, Y, random_state=100)

            for i in range(X.shape[0]):
                xi, yi = X[i, :], Y[i, :]

                # Add bias term
                xi = np.insert(xi, 0, 1)

                self.foward_step(xi)
                self.backward_step(xi, yi)

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            xi = np.insert(X[i, :], 0, 1)
            self.foward_step(xi)
            pred.append(np.argmax(self.output_act))
        return pred


class MLP2:
    def __init__(self,
                 layers_size,
                 epoches=1,
                 eta=0.01,
                 verbose=False):

        self.hidden_neurons_0 = layers_size[0]
        self.hidden_neurons_1 = layers_size[1]
        self.epoches = epoches
        self.eta = eta
        self.verbose = verbose

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def foward_step(self, xi):
        self.hidden_act_0 = self._sigmoid(
            self.w_hidden_0.dot(xi))
        self.hidden_act_0_ = np.insert(self.hidden_act_0, 0, 1)

        self.hidden_act_1 = self._sigmoid(
            self.w_hidden_1.dot(self.hidden_act_0_))
        self.hidden_act_1_ = np.insert(self.hidden_act_1, 0, 1)

        self.output_act = self._sigmoid(
            self.w_output.dot(self.hidden_act_1_))

    def backward_step(self, xi, yi):

        # Compute error
        self.error = yi - self.output_act

        # Compute gradients for output layer
        local_grad = self.output_act * (1 - self.output_act)
        output_grad = self.error*local_grad

        # Compute gradiends for hidden layer 1
        local_grad = self.hidden_act_1 * (1 - self.hidden_act_1)
        hidden_grad_1 = local_grad * (self.w_output[:, 1:].T.dot(output_grad))

        # Compute gradiends for hidden layer 0
        local_grad = self.hidden_act_0 * (1 - self.hidden_act_0)
        hidden_grad_0 = local_grad * \
            (self.w_hidden_1[:, 1:].T.dot(hidden_grad_1))

        self.w_output += self.eta * np.outer(output_grad, self.hidden_act_1_)
        self.w_hidden_1 += self.eta * \
            np.outer(hidden_grad_1, self.hidden_act_0_)
        self.w_hidden_0 += self.eta * np.outer(hidden_grad_0, xi)

    def fit(self, X, y):
        Y = encode(y)
        n_class, n_feats = Y.shape[1], X.shape[1]

        # Init weights
        self.w_hidden_0 = 0.01*randn(self.hidden_neurons_0, n_feats + 1)
        self.w_hidden_1 = 0.01 * \
            randn(self.hidden_neurons_1, self.hidden_neurons_0 + 1)
        self.w_output = 0.01*randn(n_class, self.hidden_neurons_1 + 1)

        for e in range(self.epoches):
            if self.verbose:
                print(f'Training epoch {e}')

            X, Y = shuffle(X, Y, random_state=100)

            for i in range(X.shape[0]):
                xi, yi = X[i, :], Y[i, :]

                # Add bias term
                xi = np.insert(xi, 0, 1)

                self.foward_step(xi)
                self.backward_step(xi, yi)

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            xi = np.insert(X[i, :], 0, 1)
            self.foward_step(xi)
            pred.append(np.argmax(self.output_act))
        return pred
