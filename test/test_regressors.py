from regressors.least_square import LeastSquareRegressor
from regressors.adaline_regressor import ADALINERegressor
from regressors.mlp import build_model, train_model, predict
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as split

X, y = make_regression(1000, 20, noise=50)
X_train, X_test, y_train, y_test = split(X, y, train_size=.7)

ls_reg = LeastSquareRegressor()
ls_reg.fit(X_train, y_train)
score = r2_score(y_test, ls_reg.predict(X_test))
print(f'Least Square performance: {score}')

ada_reg = ADALINERegressor(alpha=0.001, epoches=50)
ada_reg.fit(X_train, y_train)
score = r2_score(y_test, ada_reg.predict(X_test))
print(f'ADALINE Regressor performance: {score}')

mlp = build_model(n_feats=X_train.shape[1],
                  layers_size=[200])
train_model(X=X_train,
            y=y_train,
            model=mlp,
            epoches=100,
            batch_size=20)

score = r2_score(y_test, predict(X_test, mlp))
print(f'MLP (1 hidden layer) performance: {score}')


mlp = build_model(n_feats=X_train.shape[1],
                  layers_size=[100, 100])
train_model(X=X_train,
            y=y_train,
            model=mlp,
            epoches=200,
            batch_size=20)

score = r2_score(y_test, predict(X_test, mlp))
print(f'MLP (2 hidden layers) performance: {score}')