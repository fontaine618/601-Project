from KernelLogisticRegression import KernelLogisticRegression
import numpy as np
from scipy.special import expit
from sklearn.model_selection import GridSearchCV

p = 20
n = 1000
X = np.random.normal(0, 1, (n, p))
y = np.random.binomial(1, expit(X.sum(1)), n) * 2
klr = KernelLogisticRegression(kernel="rbf", gamma=0.1)
klr.fit(X, y)
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1., 10, 100, 1000, 10000, 100000),
              "gamma": (0.001, 0.01, 0.1, 1.0, 10, 100, 100)}
clf = GridSearchCV(klr, parameters, return_train_score=True)
clf.fit(X, y)
print(clf.best_params_)
print(np.mean(y == clf.predict(X)))

X = np.random.normal(0, 1, (n, p))
y = np.random.binomial(1, expit(X.sum(1)), n) * 2
print(np.mean(y == clf.predict(X)))

clf.cv_results_

import sklearn.utils.optimize
dir(sklearn.utils.optimize)
import sklearn
sklearn.__version__