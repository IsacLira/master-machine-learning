import numpy as np
from classifiers.adaline_classifier import ADALINEClassifier
from classifiers.mlp import MLP1, MLP2
from classifiers.linear_model import LeastSquareClassifier
from sklearn.model_selection import train_test_split as split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


X, y = make_classification(1000, n_classes=4, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = split(X, y, train_size=.7)


# Instantiate the ADALINE model

adaline_clf.fit(X_train, y_train)
# Compute the performance on test set
print('Accuracy for ADALINE Model: ',
    round(100*accuracy_score(y_test, adaline_clf.predict(X_test)), 2))

# Instantiate a Least Square model
ls_clf = LeastSquareClassifier()
ls_clf.fit(X_train, y_train)
print('Accuracy for Least Square Model: ',
    round(100*accuracy_score(y_test, ls_clf.predict(X_test)), 2))

# Instantiate a MLP model with 1 hidden layer
mlp1 = MLP1(hidden_neurons=100,
          epoches=20,
          eta=0.1)
mlp1.fit(X_train, y_train)
print('Accuracy for MLP Model (1 hidden layer): ',
    round(100*accuracy_score(y_test, mlp1.predict(X_test)), 2))

# Instantiate a MLP model with 1 hidden layer
mlp2 = MLP2(layers_size=(100, 100),
          epoches=50,
          eta=0.1)
mlp2.fit(X_train, y_train)
print('Accuracy for MLP Model (2 hidden layer): ',
    round(100*accuracy_score(y_test, mlp2.predict(X_test)),2 ))
