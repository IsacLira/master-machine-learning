from regressors.least_square import LeastSquareRegressor
from regressors.adaline_regressor import ADALINERegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as split

X, y = make_regression(1000, 20, noise=50)
X_train, X_test, y_train, y_test = split(X, y, train_size=.7)

ls_reg = LeastSquareRegressor()
ls_reg.fit(X_train, y_train)
score = r2_score(y_test, ls_reg.predict(X_test))
print(f'Least Square performance: {score}')

ada_reg = ADALINERegressor()
ada_reg.fit(X_train, y_train, alpha=0.001, epoches=50)
score = r2_score(y_test, ada_reg.predict(X_test))
print(f'ADALINE Regressor performance: {score}')